"""Train PPO on the stop-sign grid environment with optional curricula."""

import os, glob, time, argparse
from typing import List, Optional, Tuple
from PIL import Image
import torch
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecTransposeImage
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, BaseCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from envs.stop_sign_grid_env import StopSignGridEnv
from utils.uv_paint import GREEN_GLOW  # single pair; swap here if you want another
from utils.save_callbacks import SaveImprovingOverlaysCallback
from utils.tb_callbacks import TensorboardOverlayCallback, EpisodeMetricsCallback, StepMetricsCallback





# ----------------- custom CNN extractor -----------------
class StopSignFeatureExtractor(BaseFeaturesExtractor):
    """
    Lightweight CNN for sign-focused crops with optional mask channel.
    """

    def __init__(self, observation_space, features_dim: int = 512):
        super().__init__(observation_space, features_dim)
        n_input_channels = int(observation_space.shape[0])
        self.cnn = torch.nn.Sequential(
            torch.nn.Conv2d(n_input_channels, 32, kernel_size=5, stride=2, padding=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            torch.nn.ReLU(),
        )

        with torch.no_grad():
            sample = torch.zeros((1, *observation_space.shape), dtype=torch.float32)
            n_flatten = int(self.cnn(sample).view(1, -1).shape[1])

        self.linear = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(n_flatten, features_dim),
            torch.nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))


# ----------------- progress logger -----------------
class ProgressETACallback(BaseCallback):
    """
    Log steps-per-second and ETA at a fixed wall-clock interval.

    @param total_timesteps: Total training steps for ETA calculation.
    @param log_every_sec: Wall-clock logging interval in seconds.
    @param verbose: Verbosity level.
    """
    def __init__(self, total_timesteps: int, log_every_sec: float = 10.0, verbose: int = 1):
        super().__init__(verbose)
        self.total_timesteps = int(total_timesteps)
        self.log_every_sec = float(log_every_sec)
        self._t0 = None
        self._last = None

    def _on_training_start(self):
        self._t0 = time.perf_counter()
        self._last = self._t0

    def _on_step(self) -> bool:
        now = time.perf_counter()
        if (now - self._last) >= self.log_every_sec:
            done = self.num_timesteps
            sps = done / max(now - self._t0, 1e-9)
            if self.verbose:
                eta = max(self.model._total_timesteps - done, 0) / max(sps, 1e-9)
                print(f"[{done:,}/{self.model._total_timesteps:,}] {sps:,.2f} steps/s | ETA {eta/3600:,.2f}h")
            self.logger.record("progress/steps_per_sec", sps)
            self._last = now
        return True


class LinearRampCallback(BaseCallback):
    """
    Linearly ramp a scalar env attribute over a fixed number of steps.

    @param attr_name: Env method name to call for updates.
    @param start: Starting value.
    @param end: Ending value.
    @param steps: Number of steps to reach the end value.
    @param verbose: Verbosity level.
    """
    def __init__(self, attr_name: str, start: float, end: float, steps: int, verbose: int = 0):
        super().__init__(verbose)
        self.attr_name = str(attr_name)
        self.start = float(start)
        self.end = float(end)
        self.steps = max(int(steps), 1)

    def _on_training_start(self) -> None:
        self.training_env.env_method(self.attr_name, self.start)

    def _on_step(self) -> bool:
        t = min(float(self.num_timesteps) / float(self.steps), 1.0)
        value = self.start + t * (self.end - self.start)
        self.training_env.env_method(self.attr_name, value)
        return True


class LinearRampModelAttrCallback(BaseCallback):
    """
    Linearly ramp a model attribute (e.g., ent_coef) over a fixed number of steps.

    @param attr_name: Model attribute name to update.
    @param start: Starting value.
    @param end: Ending value.
    @param steps: Number of steps to reach the end value.
    @param verbose: Verbosity level.
    """
    def __init__(self, attr_name: str, start: float, end: float, steps: int, verbose: int = 0):
        super().__init__(verbose)
        self.attr_name = str(attr_name)
        self.start = float(start)
        self.end = float(end)
        self.steps = max(int(steps), 1)

    def _on_training_start(self) -> None:
        setattr(self.model, self.attr_name, float(self.start))

    def _on_step(self) -> bool:
        t = min(float(self.num_timesteps) / float(self.steps), 1.0)
        value = self.start + t * (self.end - self.start)
        setattr(self.model, self.attr_name, float(value))
        return True


def load_backgrounds(folder: str) -> List[Image.Image]:
    """
    Load a small set of background images from a folder.

    @param folder: Background image directory.
    @return: List of PIL images.
    """
    paths = sorted(glob.glob(os.path.join(folder, "*.*")))
    imgs = []
    for p in paths:
        try:
            imgs.append(Image.open(p).convert("RGB"))
        except Exception:
            pass
    if not imgs:
        raise FileNotFoundError(f"No backgrounds found in: {folder}")
    return imgs[:20]


def build_solid_backgrounds(img_size: Tuple[int, int]) -> List[Image.Image]:
    """
    Build a small set of solid-color backgrounds for curriculum phases.

    @param img_size: (W, H) image size.
    @return: List of PIL images.
    """
    colors = [(200, 200, 200), (120, 120, 120), (30, 30, 30)]
    W, H = int(img_size[0]), int(img_size[1])
    return [Image.new("RGB", (W, H), c) for c in colors]


def build_backgrounds(bg_mode: str, folder: str, img_size: Tuple[int, int]) -> List[Image.Image]:
    """
    Build backgrounds for training.

    @param bg_mode: "dataset" or "solid".
    @param folder: Background image directory.
    @param img_size: (W, H) image size.
    @return: List of PIL images.
    """
    mode = str(bg_mode or "dataset").lower().strip()
    if mode == "solid":
        return build_solid_backgrounds(img_size)
    return load_backgrounds(folder)


def find_latest_checkpoint(ckpt_dir: str) -> Optional[str]:
    """
    Return the newest checkpoint path in a directory, or None.

    @param ckpt_dir: Checkpoint directory.
    @return: Newest checkpoint path or None.
    """
    if not os.path.isdir(ckpt_dir):
        return None
    cands = glob.glob(os.path.join(ckpt_dir, "*.zip"))
    if not cands:
        return None
    cands.sort(key=lambda p: os.path.getmtime(p))
    return cands[-1]


def make_env_factory(
    stop_plain: Image.Image,
    stop_uv: Image.Image,
    pole_rgba: Optional[Image.Image],
    backgrounds: List[Image.Image],
    steps_per_episode: int,
    eval_K: int,
    grid_cell_px: int,
    uv_drop_threshold: float,
    lambda_area: float,
    lambda_iou: float,
    lambda_misclass: float,
    area_cap_frac: Optional[float],
    area_cap_penalty: float,
    area_cap_mode: str,
    yolo_wts: str,
    yolo_device: str,
    obs_size: Tuple[int, int],
    obs_margin: float,
    obs_include_mask: bool,
):
    """
    Create a factory function for VecEnv construction.

    @param stop_plain: Base stop-sign image.
    @param stop_uv: UV variant of the stop sign.
    @param pole_rgba: Pole image with alpha (or None to disable).
    @param backgrounds: Background image list.
    @param steps_per_episode: Max steps per episode.
    @param eval_K: Number of transforms per evaluation.
    @param grid_cell_px: Grid cell size in pixels.
    @param uv_drop_threshold: UV drop threshold for success.
    @param lambda_area: Area penalty weight.
    @param lambda_iou: IOU reward weight.
    @param lambda_misclass: Misclassification reward weight.
    @param area_cap_frac: Area cap fraction (or None).
    @param area_cap_penalty: Penalty when cap exceeded.
    @param area_cap_mode: "soft" or "hard" cap mode.
    @param yolo_wts: YOLO weights path.
    @param yolo_device: YOLO device spec.
    @param obs_size: Cropped observation size.
    @param obs_margin: Crop margin around sign bbox.
    @param obs_include_mask: Include overlay mask channel.
    @return: Callable that builds a monitored env.
    """
    def _init():
        return Monitor(
            StopSignGridEnv(
                stop_sign_image=stop_plain,
                stop_sign_uv_image=stop_uv,
                background_images=backgrounds,
                pole_image=pole_rgba,
                yolo_weights=yolo_wts,
                yolo_device=yolo_device,
                img_size=(640, 640),
                obs_size=(int(obs_size[0]), int(obs_size[1])),
                obs_margin=float(obs_margin),
                obs_include_mask=bool(obs_include_mask),

                steps_per_episode=steps_per_episode,
                eval_K=eval_K,
                detector_debug=True,

                grid_cell_px=grid_cell_px,
                # Optional cap: if area_cap_frac is set and max_cells is None, the env derives
                # max_cells = ceil(area_cap_frac * valid_total) and terminates with
                # info["note"]="max_cells_reached" once selected_cells hits that cap.
                max_cells=None,  # leave None because we terminate by threshold
                uv_paint=GREEN_GLOW,  # single color pair for this project
                use_single_color=True,

                uv_drop_threshold=uv_drop_threshold,
                day_tolerance=0.05,
                lambda_day=1.0,
                lambda_area=float(lambda_area),
                lambda_iou=float(lambda_iou),
                lambda_misclass=float(lambda_misclass),
                area_cap_frac=area_cap_frac,
                area_cap_penalty=area_cap_penalty,
                area_cap_mode=area_cap_mode,
            )
        )
    return _init


def parse_args():
    """
    Parse CLI arguments for training.

    @return: Parsed argparse namespace.
    """
    ap = argparse.ArgumentParser("Train PPO on grid-square UV attack over stop sign")
    ap.add_argument("--data", default="./data")
    ap.add_argument("--bgdir", default="./data/backgrounds")
    ap.add_argument("--yolo", "--yolo-weights", dest="yolo_weights", default=None)
    ap.add_argument("--yolo-version", choices=["8", "11"], default="8")
    ap.add_argument("--detector-device", default=os.getenv("YOLO_DEVICE", "auto"))
    ap.add_argument("--tb", default="./runs/tb")
    ap.add_argument("--ckpt", default="./runs/checkpoints")
    ap.add_argument("--overlays", default="./runs/overlays")
    ap.add_argument("--bg-mode", choices=["dataset", "solid"], default="dataset",
                    help="Background mode for single-phase training.")
    ap.add_argument("--no-pole", action="store_true",
                    help="Disable pole for single-phase training.")

    ap.add_argument("--num-envs", type=int, default=8)
    ap.add_argument("--vec", choices=["dummy", "subproc"], default="subproc")
    ap.add_argument("--n-steps", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--total-steps", type=int, default=800_000)
    ap.add_argument("--ent-coef", type=float, default=0.005,
                    help="Entropy coefficient for PPO.")

    ap.add_argument("--episode-steps", type=int, default=7000)
    ap.add_argument("--eval-K", type=int, default=10)
    ap.add_argument("--grid-cell", type=int, default=2, choices=[2, 4, 8, 16, 32])
    ap.add_argument("--uv-threshold", type=float, default=0.70)
    ap.add_argument("--lambda-area", type=float, default=0.30)
    ap.add_argument("--lambda-iou", type=float, default=0.40)
    ap.add_argument("--lambda-misclass", type=float, default=0.60)
    ap.add_argument("--area-cap-frac", type=float, default=0.30,
                    help="Fraction of sign grid allowed for patches; <=0 disables cap.")
    ap.add_argument("--area-cap-penalty", type=float, default=-0.20,
                    help="Reward penalty when area cap is exceeded.")
    ap.add_argument("--area-cap-mode", choices=["soft", "hard"], default="soft",
                    help="Soft applies a penalty when exceeded; hard terminates.")
    ap.add_argument("--obs-size", type=int, default=224,
                    help="Cropped observation size (square).")
    ap.add_argument("--obs-margin", type=float, default=0.10,
                    help="Crop margin around sign bbox (fraction of bbox size).")
    ap.add_argument("--obs-include-mask", type=int, default=1,
                    help="Include overlay mask channel in observations (1/0).")
    ap.add_argument("--area-cap-start", type=float, default=None,
                    help="Start value for area cap curriculum; overrides --area-cap-frac when set.")
    ap.add_argument("--area-cap-end", type=float, default=None,
                    help="End value for area cap curriculum; overrides --area-cap-frac when set.")
    ap.add_argument("--area-cap-steps", type=int, default=0,
                    help="Steps over which to ramp area cap; <=0 uses total-steps.")
    ap.add_argument("--lambda-area-start", type=float, default=None,
                    help="Start value for lambda-area curriculum.")
    ap.add_argument("--lambda-area-end", type=float, default=None,
                    help="End value for lambda-area curriculum.")
    ap.add_argument("--lambda-area-steps", type=int, default=0,
                    help="Steps over which to ramp lambda-area; <=0 uses total-steps.")
    ap.add_argument("--ent-coef-start", type=float, default=None,
                    help="Start value for entropy coefficient schedule.")
    ap.add_argument("--ent-coef-end", type=float, default=None,
                    help="End value for entropy coefficient schedule.")
    ap.add_argument("--ent-coef-steps", type=int, default=0,
                    help="Steps over which to ramp entropy coefficient; <=0 uses total-steps.")

    ap.add_argument("--multiphase", action="store_true",
                    help="Enable 3-phase curriculum (solid/no pole -> dataset + pole).")
    ap.add_argument("--phase1-steps", type=int, default=0,
                    help="Phase 1 steps (0 = auto split).")
    ap.add_argument("--phase2-steps", type=int, default=0,
                    help="Phase 2 steps (0 = auto split).")
    ap.add_argument("--phase3-steps", type=int, default=0,
                    help="Phase 3 steps (0 = auto split).")
    ap.add_argument("--phase1-eval-K", type=int, default=None,
                    help="Phase 1 eval_K override (default 1).")
    ap.add_argument("--phase2-eval-K", type=int, default=None,
                    help="Phase 2 eval_K override (default 3).")
    ap.add_argument("--phase3-eval-K", type=int, default=None,
                    help="Phase 3 eval_K override (default 5 or --eval-K if smaller).")

    ap.add_argument("--resume", action="store_true", help="resume from latest checkpoint in --ckpt")

    ap.add_argument("--save-freq-steps", type=int, default=0)
    ap.add_argument("--save-freq-updates", type=int, default=2)
    ap.add_argument("--step-log-every", type=int, default=1,
                    help="Log step metrics every N steps to TB/ndjson.")
    ap.add_argument("--step-log-keep", type=int, default=1000,
                    help="Keep last N step rows (0 = append forever).")
    ap.add_argument("--step-log-500", type=int, default=500,
                    help="Log confidence every N steps to a separate metric/log.")
    return ap.parse_args()


def resolve_yolo_weights(yolo_version: str, yolo_weights: Optional[str]) -> str:
    """
    Resolve default YOLO weights when no path is provided.

    @param yolo_version: YOLO version string.
    @param yolo_weights: Optional explicit weights path.
    @return: Weights path.
    """
    if yolo_weights:
        return yolo_weights
    defaults = {
        "8": "./weights/yolo8n.pt",
        "11": "./weights/yolo11n.pt",
    }
    return defaults[str(yolo_version)]


if __name__ == "__main__":
    # allocator knobs
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:256")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    args = parse_args()
    dev_lower = str(args.detector_device).lower()
    if "cuda" in dev_lower and args.vec == "subproc":
        print("WARN: CUDA detector + SubprocVecEnv is risky. Switching vec to dummy.")
        args.vec = "dummy"
    print("torch.cuda.is_available():", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("Using cuda device")

    yolo_weights = resolve_yolo_weights(args.yolo_version, args.yolo_weights)
    print(f"YOLO version={args.yolo_version} weights={yolo_weights}")

    # paths
    STOP_PLAIN = os.path.join(args.data, "stop_sign.png")
    STOP_UV    = os.path.join(args.data, "stop_sign_uv.png")
    POLE_PNG   = os.path.join(args.data, "pole.png")
    BG_DIR     = args.bgdir

    stop_plain = Image.open(STOP_PLAIN).convert("RGBA")
    stop_uv    = Image.open(STOP_UV).convert("RGBA") if os.path.exists(STOP_UV) else stop_plain.copy()
    pole_rgba  = Image.open(POLE_PNG).convert("RGBA") if os.path.exists(POLE_PNG) else None

    img_size = (640, 640)

    def build_env(eval_K: int, bg_mode: str, use_pole: bool):
        backgrounds = build_backgrounds(bg_mode, BG_DIR, img_size)
        pole_use = pole_rgba if use_pole else None

        use_cap_ramp = (args.area_cap_start is not None) or (args.area_cap_end is not None)
        if use_cap_ramp:
            start = args.area_cap_start if args.area_cap_start is not None else float(args.area_cap_frac)
            area_cap_frac = None if float(start) <= 0 else float(start)
        else:
            area_cap_frac = None if args.area_cap_frac <= 0 else float(args.area_cap_frac)

        use_lambda_ramp = (args.lambda_area_start is not None) or (args.lambda_area_end is not None)
        if use_lambda_ramp:
            lambda_area = float(args.lambda_area_start if args.lambda_area_start is not None else args.lambda_area)
        else:
            lambda_area = float(args.lambda_area)

        fns = [
            make_env_factory(
                stop_plain, stop_uv, pole_use, backgrounds,
                steps_per_episode=args.episode_steps,
                eval_K=eval_K,
                grid_cell_px=args.grid_cell,
                uv_drop_threshold=args.uv_threshold,
                lambda_area=lambda_area,
                lambda_iou=float(args.lambda_iou),
                lambda_misclass=float(args.lambda_misclass),
                area_cap_frac=area_cap_frac,
                area_cap_penalty=float(args.area_cap_penalty),
                area_cap_mode=str(args.area_cap_mode),
                yolo_wts=yolo_weights,
                yolo_device=args.detector_device,
                obs_size=(int(args.obs_size), int(args.obs_size)),
                obs_margin=float(args.obs_margin),
                obs_include_mask=bool(int(args.obs_include_mask)),
            ) for _ in range(args.num_envs)
        ]
        v = SubprocVecEnv(fns) if args.vec == "subproc" else DummyVecEnv(fns)
        return VecTransposeImage(v)

    def resolve_phase_steps(total: int) -> Tuple[int, int, int]:
        p1 = int(args.phase1_steps) if int(args.phase1_steps) > 0 else int(0.4 * total)
        p2 = int(args.phase2_steps) if int(args.phase2_steps) > 0 else int(0.3 * total)
        p3 = int(args.phase3_steps) if int(args.phase3_steps) > 0 else int(total - p1 - p2)
        if p3 < 0:
            p3 = 0
        if (p1 + p2 + p3) <= 0:
            p1, p2, p3 = int(total), 0, 0
        return p1, p2, p3

    total_steps = int(args.total_steps)
    if args.multiphase:
        total_steps = sum(resolve_phase_steps(total_steps))

    run_tag = f"grid_uv_yolo{args.yolo_version}"
    tb_root = os.path.join(args.tb, run_tag)

    # PPO
    os.makedirs(tb_root, exist_ok=True)
    os.makedirs(args.ckpt, exist_ok=True)
    os.makedirs(args.overlays, exist_ok=True)

    model = None

    # callbacks
    # Save top minimal-area successes, keep top-1000, log to TB
    tb_cb = TensorboardOverlayCallback(tb_root, tag_prefix=run_tag, max_images=25, verbose=1)
    
    ep_cb = EpisodeMetricsCallback(tb_root, verbose=1)
    step_cb = StepMetricsCallback(
        tb_root,
        every_n_steps=int(args.step_log_every),
        keep_last_n=int(args.step_log_keep),
        log_every_500=int(args.step_log_500),
        verbose=0,
    )
    
    saver = SaveImprovingOverlaysCallback(
        save_dir=args.overlays, threshold=0.0, mode="minimal",
        max_saved=1000, verbose=1, tb_callback=tb_cb
    )

    # checkpoint cadence
    if args.save_freq_steps and args.save_freq_steps > 0:
        SAVE_FREQ = int(args.save_freq_steps)
    else:
        SAVE_FREQ = max(int(args.n_steps) * int(args.num_envs) * max(args.save_freq_updates, 1), 1)
    ckpt_cb = CheckpointCallback(save_freq=SAVE_FREQ, save_path=args.ckpt, name_prefix="grid")

    progress = ProgressETACallback(total_timesteps=int(total_steps), log_every_sec=15, verbose=1)

    ramp_callbacks = []
    if (args.area_cap_start is not None) or (args.area_cap_end is not None):
        cap_start = float(args.area_cap_start if args.area_cap_start is not None else args.area_cap_frac)
        cap_end = float(args.area_cap_end if args.area_cap_end is not None else args.area_cap_frac)
        cap_steps = int(args.area_cap_steps) if int(args.area_cap_steps) > 0 else int(total_steps)
        ramp_callbacks.append(LinearRampCallback("set_area_cap_frac", cap_start, cap_end, cap_steps, verbose=0))

    if (args.lambda_area_start is not None) or (args.lambda_area_end is not None):
        la_start = float(args.lambda_area_start if args.lambda_area_start is not None else args.lambda_area)
        la_end = float(args.lambda_area_end if args.lambda_area_end is not None else args.lambda_area)
        la_steps = int(args.lambda_area_steps) if int(args.lambda_area_steps) > 0 else int(total_steps)
        ramp_callbacks.append(LinearRampCallback("set_lambda_area", la_start, la_end, la_steps, verbose=0))

    if (args.ent_coef_start is not None) or (args.ent_coef_end is not None):
        ent_start = float(args.ent_coef_start if args.ent_coef_start is not None else args.ent_coef)
        ent_end = float(args.ent_coef_end if args.ent_coef_end is not None else args.ent_coef)
        ent_steps = int(args.ent_coef_steps) if int(args.ent_coef_steps) > 0 else int(total_steps)
        ramp_callbacks.append(LinearRampModelAttrCallback("ent_coef", ent_start, ent_end, ent_steps, verbose=0))

    callback_list = CallbackList([tb_cb, ep_cb, step_cb, saver, ckpt_cb, progress] + ramp_callbacks)

    if not args.multiphase:
        phase_tag = "single"
        phase_log_dir = os.path.join(tb_root, phase_tag)
        tb_cb.set_log_dir(phase_log_dir, tag_prefix=f"{run_tag}/{phase_tag}")
        ep_cb.set_log_dir(phase_log_dir)
        step_cb.set_log_dir(phase_log_dir)

        env = build_env(eval_K=int(args.eval_K), bg_mode=args.bg_mode, use_pole=not args.no_pole)
        policy_kwargs = dict(
            features_extractor_class=StopSignFeatureExtractor,
            features_extractor_kwargs=dict(features_dim=512),
            net_arch=dict(pi=[256, 256], vf=[256, 256]),
        )
        model = PPO(
            "CnnPolicy",
            env,
            verbose=2,
            n_steps=int(args.n_steps),
            batch_size=int(args.batch_size),
            learning_rate=2.0e-4,
            gamma=0.995,
            gae_lambda=0.95,
            ent_coef=float(args.ent_coef),
            vf_coef=0.5,
            clip_range=0.2,
            tensorboard_log=tb_root,
            device="auto",
            policy_kwargs=policy_kwargs,
        )

        # resume if asked
        if args.resume:
            ckpt = find_latest_checkpoint(args.ckpt)
            if ckpt:
                print(f" Resuming from: {ckpt}")
                model = PPO.load(ckpt, env=env, device="auto")
                model.n_steps = int(args.n_steps)
                model.batch_size = int(args.batch_size)
                model.ent_coef = float(args.ent_coef)

        model.learn(
            total_timesteps=int(total_steps),
            callback=callback_list,
            tb_log_name=f"{run_tag}_{phase_tag}",
        )
    else:
        p1, p2, p3 = resolve_phase_steps(int(args.total_steps))
        phase1_eval = int(args.phase1_eval_K) if args.phase1_eval_K is not None else 1
        phase2_eval = int(args.phase2_eval_K) if args.phase2_eval_K is not None else 3
        phase3_eval = int(args.phase3_eval_K) if args.phase3_eval_K is not None else min(5, int(args.eval_K))
        phases = [
            {"name": "phase1_easy", "steps": p1, "eval_K": phase1_eval, "bg_mode": "solid", "use_pole": False},
            {"name": "phase2_medium", "steps": p2, "eval_K": phase2_eval, "bg_mode": "dataset", "use_pole": True},
            {"name": "phase3_full", "steps": p3, "eval_K": phase3_eval, "bg_mode": "dataset", "use_pole": True},
        ]

        for i, ph in enumerate(phases):
            if int(ph["steps"]) <= 0:
                continue
            print(f"[CURRICULUM] {ph['name']} steps={ph['steps']} eval_K={ph['eval_K']} bg={ph['bg_mode']} pole={ph['use_pole']}")
            phase_log_dir = os.path.join(tb_root, ph["name"])
            tb_cb.set_log_dir(phase_log_dir, tag_prefix=f"{run_tag}/{ph['name']}")
            ep_cb.set_log_dir(phase_log_dir)
            step_cb.set_log_dir(phase_log_dir)

            env = build_env(eval_K=int(ph["eval_K"]), bg_mode=ph["bg_mode"], use_pole=bool(ph["use_pole"]))

            if model is None:
                policy_kwargs = dict(
                    features_extractor_class=StopSignFeatureExtractor,
                    features_extractor_kwargs=dict(features_dim=512),
                    net_arch=dict(pi=[256, 256], vf=[256, 256]),
                )
                model = PPO(
                    "CnnPolicy",
                    env,
                    verbose=2,
                    n_steps=int(args.n_steps),
                    batch_size=int(args.batch_size),
                    learning_rate=2.0e-4,
                    gamma=0.995,
                    gae_lambda=0.95,
                    ent_coef=float(args.ent_coef),
                    vf_coef=0.5,
                    clip_range=0.2,
                    tensorboard_log=tb_root,
                    device="auto",
                    policy_kwargs=policy_kwargs,
                )

                if args.resume:
                    ckpt = find_latest_checkpoint(args.ckpt)
                    if ckpt:
                        print(f" Resuming from: {ckpt}")
                        model = PPO.load(ckpt, env=env, device="auto")
                        model.n_steps = int(args.n_steps)
                        model.batch_size = int(args.batch_size)
                        model.ent_coef = float(args.ent_coef)
            else:
                model.set_env(env)

            model._total_timesteps = int(total_steps)
            model.learn(
                total_timesteps=int(ph["steps"]),
                reset_num_timesteps=False,
                callback=callback_list,
                tb_log_name=f"{run_tag}_{ph['name']}",
            )

    final = os.path.join(args.ckpt, "ppo_grid_uv_final")
    model.save(final)
    print(f" Saved final model to {final}")
