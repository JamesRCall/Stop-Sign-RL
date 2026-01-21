"""Train PPO on the stop-sign grid environment with optional curricula."""

import os, glob, time, argparse
from typing import List, Optional
from PIL import Image
import torch
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecTransposeImage
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, BaseCallback

from envs.stop_sign_grid_env import StopSignGridEnv
from utils.uv_paint import GREEN_GLOW  # single pair; swap here if you want another
from utils.save_callbacks import SaveImprovingOverlaysCallback
from utils.tb_callbacks import TensorboardOverlayCallback, EpisodeMetricsCallback, StepMetricsCallback



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
    pole_rgba: Image.Image,
    backgrounds: List[Image.Image],
    steps_per_episode: int,
    eval_K: int,
    grid_cell_px: int,
    uv_drop_threshold: float,
    lambda_area: float,
    area_cap_frac: Optional[float],
    area_cap_penalty: float,
    area_cap_mode: str,
    yolo_wts: str,
    yolo_device: str,
):
    """
    Create a factory function for VecEnv construction.

    @param stop_plain: Base stop-sign image.
    @param stop_uv: UV variant of the stop sign.
    @param pole_rgba: Pole image with alpha.
    @param backgrounds: Background image list.
    @param steps_per_episode: Max steps per episode.
    @param eval_K: Number of transforms per evaluation.
    @param grid_cell_px: Grid cell size in pixels.
    @param uv_drop_threshold: UV drop threshold for success.
    @param lambda_area: Area penalty weight.
    @param area_cap_frac: Area cap fraction (or None).
    @param area_cap_penalty: Penalty when cap exceeded.
    @param area_cap_mode: "soft" or "hard" cap mode.
    @param yolo_wts: YOLO weights path.
    @param yolo_device: YOLO device spec.
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

    ap.add_argument("--num-envs", type=int, default=8)
    ap.add_argument("--vec", choices=["dummy", "subproc"], default="subproc")
    ap.add_argument("--n-steps", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--total-steps", type=int, default=800_000)

    ap.add_argument("--episode-steps", type=int, default=7000)
    ap.add_argument("--eval-K", type=int, default=10)
    ap.add_argument("--grid-cell", type=int, default=2, choices=[2, 4, 8, 16, 32])
    ap.add_argument("--uv-threshold", type=float, default=0.70)
    ap.add_argument("--lambda-area", type=float, default=0.30)
    ap.add_argument("--area-cap-frac", type=float, default=0.30,
                    help="Fraction of sign grid allowed for patches; <=0 disables cap.")
    ap.add_argument("--area-cap-penalty", type=float, default=-0.20,
                    help="Reward penalty when area cap is exceeded.")
    ap.add_argument("--area-cap-mode", choices=["soft", "hard"], default="soft",
                    help="Soft applies a penalty when exceeded; hard terminates.")
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
    pole_rgba  = Image.open(POLE_PNG).convert("RGBA")
    bgs        = load_backgrounds(BG_DIR)

    def build_env():
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
                stop_plain, stop_uv, pole_rgba, bgs,
                steps_per_episode=args.episode_steps,
                eval_K=args.eval_K,
                grid_cell_px=args.grid_cell,
                uv_drop_threshold=args.uv_threshold,
                lambda_area=lambda_area,
                area_cap_frac=area_cap_frac,
                area_cap_penalty=float(args.area_cap_penalty),
                area_cap_mode=str(args.area_cap_mode),
                yolo_wts=yolo_weights,
                yolo_device=args.detector_device,
            ) for _ in range(args.num_envs)
        ]
        v = SubprocVecEnv(fns) if args.vec == "subproc" else DummyVecEnv(fns)
        return VecTransposeImage(v)

    env = build_env()

    # PPO
    os.makedirs(args.tb, exist_ok=True)
    os.makedirs(args.ckpt, exist_ok=True)
    os.makedirs(args.overlays, exist_ok=True)

    model = PPO(
        "CnnPolicy",
        env,
        verbose=2,
        n_steps=int(args.n_steps),
        batch_size=int(args.batch_size),
        learning_rate=2.0e-4,
        gamma=0.995,
        gae_lambda=0.95,
        ent_coef=0.005,
        vf_coef=0.5,
        clip_range=0.2,
        tensorboard_log=args.tb,
        device="auto",
    )

    # resume if asked
    if args.resume:
        ckpt = find_latest_checkpoint(args.ckpt)
        if ckpt:
            print(f" Resuming from: {ckpt}")
            model = PPO.load(ckpt, env=env, device="auto")
            model.n_steps = int(args.n_steps)
            model.batch_size = int(args.batch_size)

    # callbacks
    # Save top minimal-area successes, keep top-1000, log to TB
    tb_cb = TensorboardOverlayCallback(args.tb, tag_prefix="grid_uv", max_images=25, verbose=1)
    
    ep_cb = EpisodeMetricsCallback(args.tb, verbose=1)
    step_cb = StepMetricsCallback(
        args.tb,
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

    progress = ProgressETACallback(total_timesteps=int(args.total_steps), log_every_sec=15, verbose=1)

    ramp_callbacks = []
    if (args.area_cap_start is not None) or (args.area_cap_end is not None):
        cap_start = float(args.area_cap_start if args.area_cap_start is not None else args.area_cap_frac)
        cap_end = float(args.area_cap_end if args.area_cap_end is not None else args.area_cap_frac)
        cap_steps = int(args.area_cap_steps) if int(args.area_cap_steps) > 0 else int(args.total_steps)
        ramp_callbacks.append(LinearRampCallback("set_area_cap_frac", cap_start, cap_end, cap_steps, verbose=0))

    if (args.lambda_area_start is not None) or (args.lambda_area_end is not None):
        la_start = float(args.lambda_area_start if args.lambda_area_start is not None else args.lambda_area)
        la_end = float(args.lambda_area_end if args.lambda_area_end is not None else args.lambda_area)
        la_steps = int(args.lambda_area_steps) if int(args.lambda_area_steps) > 0 else int(args.total_steps)
        ramp_callbacks.append(LinearRampCallback("set_lambda_area", la_start, la_end, la_steps, verbose=0))

    model.learn(
        total_timesteps=int(args.total_steps),
        callback=CallbackList([tb_cb, ep_cb, step_cb, saver, ckpt_cb, progress] + ramp_callbacks),
        tb_log_name="grid_uv",
    )

    final = os.path.join(args.ckpt, "ppo_grid_uv_final")
    model.save(final)
    print(f" Saved final model to {final}")
