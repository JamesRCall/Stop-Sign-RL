"""Train MaskablePPO on an alpha-masked traffic-sign grid environment.

Notes:
  - Action masking is enabled (duplicate grid cells are masked out).
  - Observations are VecNormalize'd; stats are saved alongside checkpoints.
"""

import os, glob, time, argparse, re, random
from typing import List, Optional, Tuple
from PIL import Image
import torch
import numpy as np

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecTransposeImage, VecNormalize
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, BaseCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from envs.traffic_sign_grid_env import TrafficSignGridEnv
from envs.attack_objective import ATTACK_MODES
from utils.sign_assets import resolve_sign_assets
from utils.experiment_manifest import build_experiment_manifest, write_manifest
from utils.uv_paint import (
    WHITE_GLOW,
    RED_GLOW,
    GREEN_GLOW,
    YELLOW_GLOW,
    BLUE_GLOW,
    ORANGE_GLOW,
    UVPaint,
)
from utils.save_callbacks import SaveImprovingOverlaysCallback
from utils.tb_callbacks import TensorboardOverlayCallback, EpisodeMetricsCallback, StepMetricsCallback





# ----------------- custom CNN extractor -----------------
class TrafficSignFeatureExtractor(BaseFeaturesExtractor):
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


def build_policy_kwargs(cnn_arch: str) -> dict:
    """
    Build policy kwargs for the custom CNN extractor (or NatureCNN).

    Args:
        cnn_arch: "custom" or "nature".

    Returns:
        Dict of policy kwargs.
    """
    arch = str(cnn_arch or "custom").strip().lower()
    kwargs = {"normalize_images": False}
    if arch == "nature":
        return kwargs
    kwargs.update({
        "features_extractor_class": TrafficSignFeatureExtractor,
        "features_extractor_kwargs": {"features_dim": 512},
        "net_arch": {"pi": [256, 256], "vf": [256, 256]},
    })
    return kwargs


# Keep old checkpoint deserialization/import paths working.
StopSignFeatureExtractor = TrafficSignFeatureExtractor


def resolve_paint_list(paint_name: str, paint_list: Optional[str]) -> List[UVPaint]:
    """
    Resolve paint selection from CLI args.

    Args:
        paint_name: Single paint name.
        paint_list: Optional comma-separated list.

    Returns:
        List of UVPaints (length >= 1).
    """
    mapping = {
        "white": WHITE_GLOW,
        "red": RED_GLOW,
        "green": GREEN_GLOW,
        "yellow": YELLOW_GLOW,
        "blue": BLUE_GLOW,
        "orange": ORANGE_GLOW,
    }
    paints = []
    if paint_list:
        for part in paint_list.split(","):
            key = part.strip().lower()
            if key in mapping:
                paints.append(mapping[key])
    if not paints:
        key = str(paint_name or "yellow").strip().lower()
        paints = [mapping.get(key, YELLOW_GLOW)]
    return paints


# ----------------- progress logger -----------------
class ProgressETACallback(BaseCallback):
    """
    Log steps-per-second and ETA at a fixed wall-clock interval.

    Args:
        total_timesteps: Total training steps for ETA calculation.
        log_every_sec: Wall-clock logging interval in seconds.
        verbose: Verbosity level.
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

    Args:
        attr_name: Env method name to call for updates.
        start: Starting value.
        end: Ending value.
        steps: Number of steps to reach the end value.
        verbose: Verbosity level.
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

    Args:
        attr_name: Model attribute name to update.
        start: Starting value.
        end: Ending value.
        steps: Number of steps to reach the end value.
        verbose: Verbosity level.
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


class SaveVecNormalizeCallback(BaseCallback):
    """
    Periodically save VecNormalize stats alongside checkpoints.

    Args:
        save_freq: Save frequency in training steps.
        save_dir: Directory to save vecnormalize.pkl.
        verbose: Verbosity level.
    """
    def __init__(self, save_freq: int, save_dir: str, verbose: int = 0):
        super().__init__(verbose)
        self.save_freq = max(int(save_freq), 1)
        self.save_dir = str(save_dir)

    def _on_step(self) -> bool:
        if (self.n_calls % self.save_freq) != 0:
            return True
        venv = self.model.get_env()
        if isinstance(venv, VecNormalize):
            os.makedirs(self.save_dir, exist_ok=True)
            vec_path = os.path.join(self.save_dir, "vecnormalize.pkl")
            venv.save(vec_path)
            step_vec_path = os.path.join(
                self.save_dir,
                f"vecnormalize_{int(self.num_timesteps)}_steps.pkl",
            )
            venv.save(step_vec_path)
            if self.verbose:
                print(f"[VECNORM] Saved stats to {step_vec_path}")
        return True

def load_backgrounds(folder: str) -> List[Image.Image]:
    """
    Load a small set of background images from a folder.

    Args:
        folder: Background image directory.

    Returns:
        List of PIL images.
    """
    supported = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
    paths = [
        p for p in sorted(glob.glob(os.path.join(folder, "*.*")))
        if os.path.splitext(p)[1].lower() in supported
    ]
    imgs = []
    failures = []
    for p in paths:
        try:
            imgs.append(Image.open(p).convert("RGB"))
        except Exception as exc:
            failures.append(f"{p}: {exc}")
    if failures:
        raise RuntimeError("Unreadable background assets:\n" + "\n".join(failures))
    if not imgs:
        raise FileNotFoundError(f"No backgrounds found in: {folder}")
    return imgs


def build_solid_backgrounds(img_size: Tuple[int, int]) -> List[Image.Image]:
    """
    Build a small set of solid-color backgrounds for curriculum phases.

    Args:
        img_size: (W, H) image size.

    Returns:
        List of PIL images.
    """
    colors = [(200, 200, 200), (120, 120, 120), (30, 30, 30)]
    W, H = int(img_size[0]), int(img_size[1])
    return [Image.new("RGB", (W, H), c) for c in colors]


def build_backgrounds(bg_mode: str, folder: str, img_size: Tuple[int, int]) -> List[Image.Image]:
    """
    Build backgrounds for training.

    Args:
        bg_mode: "dataset" or "solid".
        folder: Background image directory.
        img_size: (W, H) image size.

    Returns:
        List of PIL images.
    """
    mode = str(bg_mode or "dataset").lower().strip()
    if mode == "solid":
        return build_solid_backgrounds(img_size)
    return load_backgrounds(folder)


def find_latest_checkpoint(ckpt_dir: str) -> Optional[str]:
    """
    Return the newest checkpoint path in a directory, or None.

    Args:
        ckpt_dir: Checkpoint directory.

    Returns:
        Newest checkpoint path or None.
    """
    if not os.path.isdir(ckpt_dir):
        return None
    cands = glob.glob(os.path.join(ckpt_dir, "*.zip"))
    if not cands:
        return None
    cands.sort(key=lambda p: os.path.getmtime(p))
    return cands[-1]


def find_vecnormalize_for_checkpoint(checkpoint_path: str) -> str:
    """Resolve checkpoint-matched observation statistics, failing if absent."""
    folder = os.path.dirname(os.path.abspath(checkpoint_path))
    stem = os.path.splitext(os.path.basename(checkpoint_path))[0]
    match = re.search(r"_(\d+)_steps$", stem)
    candidates = []
    if match:
        candidates.append(
            os.path.join(folder, f"vecnormalize_{match.group(1)}_steps.pkl")
        )
    candidates.append(os.path.join(folder, "vecnormalize.pkl"))
    for path in candidates:
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(
        "Resume requires the VecNormalize statistics saved with the checkpoint; "
        f"checked: {', '.join(candidates)}"
    )


def make_env_factory(
    sign_plain: Image.Image,
    sign_active: Image.Image,
    pole_rgba: Optional[Image.Image],
    backgrounds: List[Image.Image],
    steps_per_episode: int,
    eval_K: int,
    grid_cell_px: int,
    uv_drop_threshold: float,
    success_conf_threshold: float,
    lambda_efficiency: float,
    efficiency_eps: float,
    transform_strength: float,
    lambda_area: float,
    area_target_frac: Optional[float],
    step_cost: float,
    step_cost_after_target: float,
    day_tolerance: float,
    lambda_day: float,
    lambda_iou: float,
    lambda_misclass: float,
    area_cap_frac: Optional[float],
    area_cap_penalty: float,
    area_cap_mode: str,
    yolo_wts: str,
    yolo_device: str,
    detector_type: str,
    detector_model: Optional[str],
    source_class,
    attack_mode: str,
    attack_target_class,
    allowed_alternative_classes,
    target_conf_threshold: float,
    min_attack_success_rate: float,
    min_clean_detection_rate: float,
    localization_iou_threshold: float,
    require_source_suppression: bool,
    require_day_preservation: bool,
    obs_size: Tuple[int, int],
    obs_margin: float,
    obs_include_mask: bool,
    uv_paints: List[UVPaint],
    cell_cover_thresh: float,
    lambda_perceptual: float,
    seed: int,
):
    """
    Create a factory function for VecEnv construction.

    Args:
        sign_plain: Base traffic-sign image.
        sign_active: Active/UV variant of the traffic sign.
        pole_rgba: Pole image with alpha (or None to disable).
        backgrounds: Background image list.
        steps_per_episode: Max steps per episode.
        eval_K: Number of transforms per evaluation.
        grid_cell_px: Grid cell size in pixels.
        uv_drop_threshold: UV drop threshold for shaping.
        success_conf_threshold: Success threshold for after-conf.
        lambda_efficiency: Efficiency bonus weight (optional).
        efficiency_eps: Denominator epsilon for efficiency bonus.
        transform_strength: Strength of sign transforms (0..1).
        lambda_area: Area penalty weight.
        area_target_frac: Target area fraction for excess penalties.
        step_cost: Per-step penalty (global).
        step_cost_after_target: Additional per-step penalty after target area.
        lambda_iou: IOU reward weight.
        lambda_misclass: Misclassification reward weight.
        area_cap_frac: Area cap fraction (or None).
        area_cap_penalty: Penalty when cap exceeded.
        area_cap_mode: "soft" or "hard" cap mode.
        area_target_frac: Target area fraction for excess penalties.
        yolo_wts: YOLO weights path.
        yolo_device: YOLO device spec.
        detector_type: Detector backend ("yolo" or "torchvision" or "rtdetr").
        detector_model: Torchvision/RT-DETR model name (optional).
        obs_size: Cropped observation size.
        obs_margin: Crop margin around sign bbox.
        obs_include_mask: Include overlay mask channel.
        uv_paints: List of paints to sample (per episode).
        cell_cover_thresh: Cell coverage threshold for grid validity.
        lambda_perceptual: Daylight visibility penalty weight.

    Returns:
        Callable that builds a monitored env.
    """
    def _init():
        paint_list = list(uv_paints) if uv_paints else [YELLOW_GLOW]
        env = TrafficSignGridEnv(
            stop_sign_image=sign_plain,
            stop_sign_uv_image=sign_active,
            background_images=backgrounds,
            pole_image=pole_rgba,
            yolo_weights=yolo_wts,
            yolo_device=yolo_device,
            detector_type=detector_type,
            detector_model=detector_model,
            img_size=(640, 640),
            obs_size=(int(obs_size[0]), int(obs_size[1])),
            obs_margin=float(obs_margin),
            obs_include_mask=bool(obs_include_mask),

            steps_per_episode=steps_per_episode,
            eval_K=eval_K,
            detector_debug=False,

            grid_cell_px=grid_cell_px,
            # Keep the optional legacy count cap disabled. Area constraints are
            # enforced with exact painted sign pixels inside the environment.
            max_cells=None,
            uv_paint=paint_list[0],
            uv_paint_list=paint_list if len(paint_list) > 1 else None,
            use_single_color=True,
            cell_cover_thresh=float(cell_cover_thresh),

            uv_drop_threshold=uv_drop_threshold,
            success_conf_threshold=success_conf_threshold,
            lambda_efficiency=lambda_efficiency,
            efficiency_eps=efficiency_eps,
            transform_strength=transform_strength,
            day_tolerance=float(day_tolerance),
            lambda_day=float(lambda_day),
            lambda_area=float(lambda_area),
            area_target_frac=area_target_frac,
            step_cost=float(step_cost),
            step_cost_after_target=float(step_cost_after_target),
            lambda_iou=float(lambda_iou),
            lambda_misclass=float(lambda_misclass),
            lambda_perceptual=float(lambda_perceptual),
            area_cap_frac=area_cap_frac,
            area_cap_penalty=area_cap_penalty,
            area_cap_mode=area_cap_mode,
            source_class=source_class,
            attack_mode=attack_mode,
            attack_target_class=attack_target_class,
            allowed_alternative_classes=allowed_alternative_classes,
            target_conf_threshold=float(target_conf_threshold),
            min_attack_success_rate=float(min_attack_success_rate),
            min_clean_detection_rate=float(min_clean_detection_rate),
            localization_iou_threshold=float(localization_iou_threshold),
            require_source_suppression=bool(require_source_suppression),
            require_day_preservation=bool(require_day_preservation),
            seed=int(seed),
        )
        env = Monitor(env)
        env = ActionMasker(env, lambda e: e.unwrapped.action_masks())
        return env
    return _init


def parse_args():
    """
    Parse CLI arguments for training.

    Returns:
        Parsed argparse namespace.
    """
    ap = argparse.ArgumentParser("Train PPO on a grid-square traffic-sign attack")
    ap.add_argument("--data", default="./data")
    ap.add_argument("--bgdir", default="./data/backgrounds")
    ap.add_argument("--sign-profile", choices=["stop", "speed_limit", "custom"], default="stop")
    ap.add_argument("--sign-image", default="",
                    help="Explicit RGBA day-state sign asset.")
    ap.add_argument("--sign-active-image", default="",
                    help="Optional RGBA active/UV-state sign asset; defaults to day asset.")
    ap.add_argument("--source-class", default="",
                    help="Detector label (or integer id) for the clean sign.")
    ap.add_argument("--attack-mode", choices=ATTACK_MODES, default="disappearance")
    ap.add_argument("--attack-target-class", default="",
                    help="Required detector label/id for targeted misclassification.")
    ap.add_argument(
        "--allowed-alternative-classes",
        default="",
        help=(
            "Comma-separated preregistered labels/ids that may count for "
            "untargeted misclassification. Empty uses every non-source label."
        ),
    )
    ap.add_argument("--target-conf", type=float, default=0.40,
                    help="Wrong/target-class confidence required for misclassification.")
    ap.add_argument("--min-attack-success-rate", type=float, default=0.80,
                    help="Required fraction of eligible EOT samples satisfying the attack.")
    ap.add_argument("--min-clean-detection-rate", type=float, default=0.80,
                    help="Required fraction of EOT samples correctly classified before attack.")
    ap.add_argument("--localization-iou", type=float, default=0.30,
                    help="Minimum detection/sign-box IoU used for attack attribution.")
    ap.add_argument("--require-source-suppression", type=int, choices=[0, 1], default=1)
    ap.add_argument("--require-day-preservation", type=int, choices=[0, 1], default=1)
    ap.add_argument("--yolo", "--yolo-weights", dest="yolo_weights", default=None)
    ap.add_argument("--yolo-version", choices=["8", "11"], default="8")
    ap.add_argument("--detector-device", default=os.getenv("YOLO_DEVICE", "auto"))
    ap.add_argument("--detector", default="yolo",
                    help="Detector backend: yolo, torchvision, or rtdetr.")
    ap.add_argument("--detector-model", default="",
                    help="Torchvision model name (e.g., fasterrcnn_resnet50_fpn_v2).")
    ap.add_argument("--tb", default="./runs/tb")
    ap.add_argument("--ckpt", default="./runs/checkpoints")
    ap.add_argument("--overlays", default="./runs/overlays")
    ap.add_argument("--bg-mode", choices=["dataset", "solid"], default="dataset",
                    help="Background mode for single-phase training.")
    ap.add_argument("--no-pole", action="store_true",
                    help="Disable pole for single-phase training.")
    ap.add_argument("--cnn", choices=["custom", "nature"], default="custom",
                    help="Feature extractor: custom CNN or SB3 NatureCNN.")

    ap.add_argument("--num-envs", type=int, default=8)
    ap.add_argument("--vec", choices=["dummy", "subproc"], default="subproc")
    ap.add_argument("--n-steps", type=int, default=1024)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--total-steps", type=int, default=800_000)
    ap.add_argument("--ent-coef", type=float, default=0.001,
                    help="Entropy coefficient for PPO.")
    ap.add_argument("--seed", type=int, default=0,
                    help="Root seed for Python, NumPy, Torch, SB3, and environments.")

    ap.add_argument("--episode-steps", type=int, default=300)
    ap.add_argument("--eval-K", type=int, default=3)
    ap.add_argument("--grid-cell", type=int, default=16)
    ap.add_argument("--uv-threshold", type=float, default=0.75)
    ap.add_argument("--success-conf", type=float, default=0.20,
                    help="Maximum localized source confidence for source suppression.")
    ap.add_argument("--lambda-area", type=float, default=0.70)
    ap.add_argument("--lambda-iou", type=float, default=0.40)
    ap.add_argument("--lambda-misclass", type=float, default=0.60)
    ap.add_argument("--lambda-perceptual", type=float, default=0.0,
                    help="Penalty for daylight visibility (lower is better).")
    ap.add_argument("--day-tolerance", type=float, default=0.05,
                    help="Maximum clean-day confidence drop allowed for joint success.")
    ap.add_argument("--lambda-day", type=float, default=1.0,
                    help="Penalty weight for daylight drop.")
    ap.add_argument("--lambda-efficiency", type=float, default=0.40,
                    help="Efficiency bonus weight (drop per area).")
    ap.add_argument("--efficiency-eps", type=float, default=0.02,
                    help="Epsilon for efficiency denominator.")
    ap.add_argument("--area-target", type=float, default=0.25,
                    help="Target area fraction for excess penalties.")
    ap.add_argument("--step-cost", type=float, default=0.012,
                    help="Per-step penalty (global).")
    ap.add_argument("--step-cost-after-target", type=float, default=0.14,
                    help="Additional per-step penalty when area exceeds the target.")
    ap.add_argument("--paint", default="yellow",
                    help="Paint name (red, green, yellow, blue, white, orange).")
    ap.add_argument("--paint-list", default="",
                    help="Comma-separated list of paint names to sample per episode.")
    ap.add_argument("--cell-cover-thresh", type=float, default=0.60,
                    help="Grid cell coverage threshold (0..1). Lower covers edges.")
    ap.add_argument("--transform-strength", type=float, default=1.0,
                    help="Strength of geometric/photometric transforms (0..1).")
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
                    help="Phase 2 eval_K override (default 2).")
    ap.add_argument("--phase3-eval-K", type=int, default=None,
                    help="Phase 3 eval_K override (default 4 or --eval-K if smaller).")
    ap.add_argument("--phase1-lambda-day", type=float, default=None,
                    help="Phase 1 lambda-day override (default 0.0).")
    ap.add_argument("--phase2-lambda-day", type=float, default=None,
                    help="Phase 2 lambda-day override (default 0.5).")
    ap.add_argument("--phase3-lambda-day", type=float, default=None,
                    help="Phase 3 lambda-day override (default 1.0).")
    ap.add_argument("--phase1-transform-strength", type=float, default=None,
                    help="Phase 1 transform strength override (default 0.4).")
    ap.add_argument("--phase2-transform-strength", type=float, default=None,
                    help="Phase 2 transform strength override (default 0.7).")
    ap.add_argument("--phase3-transform-strength", type=float, default=None,
                    help="Phase 3 transform strength override (default 1.0).")
    ap.add_argument("--phase1-step-cost", type=float, default=None,
                    help="Phase 1 per-step penalty override.")
    ap.add_argument("--phase2-step-cost", type=float, default=None,
                    help="Phase 2 per-step penalty override.")
    ap.add_argument("--phase3-step-cost", type=float, default=None,
                    help="Phase 3 per-step penalty override.")

    ap.add_argument("--resume", action="store_true",
                    help="continue from latest checkpoint for --total-steps additional timesteps (single-phase only)")
    ap.add_argument("--check-env", action="store_true",
                    help="Run SB3 env checker on a single env instance, then exit.")

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

    Args:
        yolo_version: YOLO version string.
        yolo_weights: Optional explicit weights path.

    Returns:
        Weights path.
    """
    if yolo_weights:
        return yolo_weights
    defaults = {
        "8": "./weights/yolov8n.pt",
        "11": "./weights/yolo11n.pt",
    }
    return defaults[str(yolo_version)]


if __name__ == "__main__":
    # allocator knobs
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:256")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    args = parse_args()
    if args.resume and args.multiphase:
        raise ValueError(
            "multiphase resume requires persisted phase progress and is intentionally "
            "disabled; resume a single-phase run or start a new curriculum"
        )
    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))
    if str(args.detector_device).strip().lower() == "auto":
        args.detector_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dev_lower = str(args.detector_device).lower()
    if "cuda" in dev_lower and args.vec == "subproc":
        print("WARN: CUDA detector + SubprocVecEnv is risky. Switching vec to dummy.")
        args.vec = "dummy"
    if "cuda" in dev_lower and int(args.num_envs) > 1:
        print("WARN: in-process CUDA detector uses one model per env; forcing num_envs=1.")
        args.num_envs = 1
    print("torch.cuda.is_available():", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("Using cuda device")

    yolo_weights = resolve_yolo_weights(args.yolo_version, args.yolo_weights)
    if str(args.detector).lower() == "yolo":
        print(f"YOLO version={args.yolo_version} weights={yolo_weights}")
    else:
        det_model = str(args.detector_model) if args.detector_model else "default"
        print(f"Detector={args.detector} model={det_model}")

    # Sign/profile paths. Speed-limit profiles require a fine-grained custom
    # detector label map; class resolution fails closed if the label is absent.
    sign_assets = resolve_sign_assets(
        data_dir=args.data,
        profile=args.sign_profile,
        sign_image=(args.sign_image or None),
        sign_active_image=(args.sign_active_image or None),
        source_class=(args.source_class or None),
    )
    source_class = sign_assets.source_class
    attack_target_class = args.attack_target_class or None
    POLE_PNG   = os.path.join(args.data, "pole.png")
    BG_DIR     = args.bgdir

    sign_plain = Image.open(sign_assets.day_image).convert("RGBA")
    sign_active = Image.open(sign_assets.active_image).convert("RGBA")
    pole_rgba  = Image.open(POLE_PNG).convert("RGBA") if os.path.exists(POLE_PNG) else None

    img_size = (640, 640)

    def build_single_env(bg_mode: str, use_pole: bool, transform_strength: Optional[float] = None):
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

        paint_list = resolve_paint_list(args.paint, args.paint_list)
        return TrafficSignGridEnv(
            stop_sign_image=sign_plain,
            stop_sign_uv_image=sign_active,
            background_images=backgrounds,
            pole_image=pole_use,
            yolo_weights=yolo_weights,
            yolo_device=args.detector_device,
            detector_type=str(args.detector),
            detector_model=str(args.detector_model) if args.detector_model else None,
            img_size=(640, 640),
            obs_size=(int(args.obs_size), int(args.obs_size)),
            obs_margin=float(args.obs_margin),
            obs_include_mask=bool(int(args.obs_include_mask)),

            steps_per_episode=int(args.episode_steps),
            eval_K=int(args.eval_K),
            detector_debug=True,

            grid_cell_px=int(args.grid_cell),
            max_cells=None,
            uv_paint=paint_list[0],
            uv_paint_list=paint_list if len(paint_list) > 1 else None,
            use_single_color=True,
            cell_cover_thresh=float(args.cell_cover_thresh),

            uv_drop_threshold=float(args.uv_threshold),
            success_conf_threshold=float(args.success_conf),
            lambda_efficiency=float(args.lambda_efficiency),
            efficiency_eps=float(args.efficiency_eps),
            transform_strength=float(args.transform_strength if transform_strength is None else transform_strength),
            day_tolerance=float(args.day_tolerance),
            lambda_day=float(args.lambda_day),
            lambda_area=float(lambda_area),
            area_target_frac=(float(args.area_target) if args.area_target is not None else None),
            step_cost=float(args.step_cost),
            step_cost_after_target=float(args.step_cost_after_target),
            lambda_iou=float(args.lambda_iou),
            lambda_misclass=float(args.lambda_misclass),
            lambda_perceptual=float(args.lambda_perceptual),
            area_cap_frac=area_cap_frac,
            area_cap_penalty=float(args.area_cap_penalty),
            area_cap_mode=str(args.area_cap_mode),
            source_class=source_class,
            attack_mode=str(args.attack_mode),
            attack_target_class=attack_target_class,
            allowed_alternative_classes=args.allowed_alternative_classes,
            target_conf_threshold=float(args.target_conf),
            min_attack_success_rate=float(args.min_attack_success_rate),
            min_clean_detection_rate=float(args.min_clean_detection_rate),
            localization_iou_threshold=float(args.localization_iou),
            require_source_suppression=bool(int(args.require_source_suppression)),
            require_day_preservation=bool(int(args.require_day_preservation)),
            seed=int(args.seed),
        )

    def build_env(
        eval_K: int,
        bg_mode: str,
        use_pole: bool,
        transform_strength: Optional[float] = None,
        vecnorm_path: Optional[str] = None,
    ):
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

        paint_list = resolve_paint_list(args.paint, args.paint_list)
        fns = [
            make_env_factory(
                sign_plain, sign_active, pole_use, backgrounds,
                steps_per_episode=args.episode_steps,
                eval_K=eval_K,
                grid_cell_px=args.grid_cell,
                uv_drop_threshold=args.uv_threshold,
                success_conf_threshold=float(args.success_conf),
                lambda_area=lambda_area,
                area_target_frac=(float(args.area_target) if args.area_target is not None else None),
                step_cost=float(args.step_cost),
                step_cost_after_target=float(args.step_cost_after_target),
                day_tolerance=float(args.day_tolerance),
                lambda_day=float(args.lambda_day),
                lambda_iou=float(args.lambda_iou),
                lambda_misclass=float(args.lambda_misclass),
                lambda_efficiency=float(args.lambda_efficiency),
                efficiency_eps=float(args.efficiency_eps),
                transform_strength=float(args.transform_strength if transform_strength is None else transform_strength),
                area_cap_frac=area_cap_frac,
                area_cap_penalty=float(args.area_cap_penalty),
                area_cap_mode=str(args.area_cap_mode),
                yolo_wts=yolo_weights,
                yolo_device=args.detector_device,
                detector_type=str(args.detector),
                detector_model=str(args.detector_model) if args.detector_model else None,
                source_class=source_class,
                attack_mode=str(args.attack_mode),
                attack_target_class=attack_target_class,
                allowed_alternative_classes=args.allowed_alternative_classes,
                target_conf_threshold=float(args.target_conf),
                min_attack_success_rate=float(args.min_attack_success_rate),
                min_clean_detection_rate=float(args.min_clean_detection_rate),
                localization_iou_threshold=float(args.localization_iou),
                require_source_suppression=bool(int(args.require_source_suppression)),
                require_day_preservation=bool(int(args.require_day_preservation)),
                obs_size=(int(args.obs_size), int(args.obs_size)),
                obs_margin=float(args.obs_margin),
                obs_include_mask=bool(int(args.obs_include_mask)),
                uv_paints=paint_list,
                cell_cover_thresh=float(args.cell_cover_thresh),
                lambda_perceptual=float(args.lambda_perceptual),
                seed=int(args.seed) + rank,
            ) for rank in range(args.num_envs)
        ]
        v = SubprocVecEnv(fns) if args.vec == "subproc" else DummyVecEnv(fns)
        v = VecTransposeImage(v)
        if vecnorm_path:
            if not os.path.isfile(vecnorm_path):
                raise FileNotFoundError(
                    f"VecNormalize statistics required for resume: {vecnorm_path}"
                )
            v = VecNormalize.load(vecnorm_path, v)
            v.training = True
            v.norm_reward = False
        else:
            v = VecNormalize(v, norm_obs=True, norm_reward=False, clip_obs=5.0)
        return v

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

    def _sanitize_tag(value: str) -> str:
        return re.sub(r"[^A-Za-z0-9._-]+", "-", str(value).strip())

    det = str(args.detector).lower()
    if det == "yolo":
        det_tag = f"yolo{args.yolo_version}"
    else:
        det_tag = det
        if args.detector_model:
            det_tag = f"{det_tag}_{_sanitize_tag(args.detector_model)}"
    objective_tag = _sanitize_tag(str(args.attack_mode))
    sign_tag = _sanitize_tag(str(args.sign_profile))
    target_tag = (
        f"_to_{_sanitize_tag(str(attack_target_class))}"
        if attack_target_class is not None
        else ""
    )
    run_tag = f"grid_uv_{sign_tag}_{objective_tag}{target_tag}_{det_tag}"
    tb_root = os.path.join(args.tb, run_tag)

    if args.check_env:
        print("[CHECK] Running SB3 env checker...")
        env = build_single_env(bg_mode=args.bg_mode, use_pole=not args.no_pole)
        check_env(env, warn=True)
        print("[CHECK] Env checker completed.")
        raise SystemExit(0)

    # PPO
    os.makedirs(tb_root, exist_ok=True)
    os.makedirs(args.ckpt, exist_ok=True)
    os.makedirs(args.overlays, exist_ok=True)

    background_paths = sorted(
        path for path in glob.glob(os.path.join(BG_DIR, "*.*")) if os.path.isfile(path)
    )
    manifest = build_experiment_manifest(
        config=vars(args),
        sign_day_path=sign_assets.day_image,
        sign_active_path=sign_assets.active_image,
        pole_path=(POLE_PNG if os.path.isfile(POLE_PNG) else None),
        background_paths=background_paths,
        weights_path=(yolo_weights if str(args.detector).lower() == "yolo" else None),
        source_class=source_class,
        attack_target_class=attack_target_class,
    )
    write_manifest(os.path.join(args.ckpt, "experiment_manifest.json"), manifest)

    model = None

    # callbacks
    # Save a bounded set of minimal-area joint successes for frozen certification.
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
        max_saved=25, verbose=0, tb_callback=tb_cb
    )

    # checkpoint cadence
    if args.save_freq_steps and args.save_freq_steps > 0:
        desired_save_timesteps = int(args.save_freq_steps)
    else:
        desired_save_timesteps = int(args.n_steps) * int(args.num_envs) * max(args.save_freq_updates, 1)
    # SB3 callbacks count vector steps, each of which advances num_envs timesteps.
    SAVE_FREQ = max(desired_save_timesteps // max(int(args.num_envs), 1), 1)
    ckpt_cb = CheckpointCallback(save_freq=SAVE_FREQ, save_path=args.ckpt, name_prefix="grid")
    vec_cb = SaveVecNormalizeCallback(save_freq=SAVE_FREQ, save_dir=args.ckpt, verbose=0)

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

    callback_list = CallbackList([tb_cb, ep_cb, step_cb, saver, ckpt_cb, vec_cb, progress] + ramp_callbacks)

    if not args.multiphase:
        phase_tag = "single"
        phase_log_dir = os.path.join(tb_root, phase_tag)
        tb_cb.set_log_dir(phase_log_dir, tag_prefix=f"{run_tag}/{phase_tag}")
        ep_cb.set_log_dir(phase_log_dir)
        step_cb.set_log_dir(phase_log_dir)

        ckpt = None
        vecnorm_resume = None
        if args.resume:
            ckpt = find_latest_checkpoint(args.ckpt)
            if not ckpt:
                raise FileNotFoundError(f"--resume requested but no checkpoint found in {args.ckpt}")
            vecnorm_resume = find_vecnormalize_for_checkpoint(ckpt)
        env = build_env(
            eval_K=int(args.eval_K),
            bg_mode=args.bg_mode,
            use_pole=not args.no_pole,
            vecnorm_path=vecnorm_resume,
        )
        if ckpt:
            print(f" Resuming model={ckpt} vecnormalize={vecnorm_resume}")
            model = MaskablePPO.load(ckpt, env=env, device="auto")
        else:
            policy_kwargs = build_policy_kwargs(args.cnn)
            model = MaskablePPO(
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
                seed=int(args.seed),
            )

        model.learn(
            total_timesteps=int(total_steps),
            callback=callback_list,
            tb_log_name=f"{run_tag}_{phase_tag}",
            reset_num_timesteps=not bool(args.resume),
        )
    else:
        p1, p2, p3 = resolve_phase_steps(int(args.total_steps))
        phase1_eval = int(args.phase1_eval_K) if args.phase1_eval_K is not None else 1
        phase2_eval = int(args.phase2_eval_K) if args.phase2_eval_K is not None else 2
        phase3_eval = int(args.phase3_eval_K) if args.phase3_eval_K is not None else min(4, int(args.eval_K))
        phase1_tf = float(args.phase1_transform_strength) if args.phase1_transform_strength is not None else 0.4
        phase2_tf = float(args.phase2_transform_strength) if args.phase2_transform_strength is not None else 0.7
        phase3_tf = float(args.phase3_transform_strength) if args.phase3_transform_strength is not None else 1.0
        phase1_ld = float(args.phase1_lambda_day) if args.phase1_lambda_day is not None else float(args.lambda_day)
        phase2_ld = float(args.phase2_lambda_day) if args.phase2_lambda_day is not None else float(args.lambda_day)
        phase3_ld = float(args.phase3_lambda_day) if args.phase3_lambda_day is not None else float(args.lambda_day)
        phase1_sc = float(args.phase1_step_cost) if args.phase1_step_cost is not None else float(args.step_cost)
        phase2_sc = float(args.phase2_step_cost) if args.phase2_step_cost is not None else float(args.step_cost)
        phase3_sc = float(args.phase3_step_cost) if args.phase3_step_cost is not None else float(args.step_cost)
        phases = [
            {"name": "phase1_easy", "steps": p1, "eval_K": phase1_eval, "bg_mode": "solid", "use_pole": False, "tf": phase1_tf, "lambda_day": phase1_ld, "step_cost": phase1_sc},
            {"name": "phase2_medium", "steps": p2, "eval_K": phase2_eval, "bg_mode": "dataset", "use_pole": True, "tf": phase2_tf, "lambda_day": phase2_ld, "step_cost": phase2_sc},
            {"name": "phase3_full", "steps": p3, "eval_K": phase3_eval, "bg_mode": "dataset", "use_pole": True, "tf": phase3_tf, "lambda_day": phase3_ld, "step_cost": phase3_sc},
        ]

        resume_ckpt = find_latest_checkpoint(args.ckpt) if args.resume else None
        if args.resume and not resume_ckpt:
            raise FileNotFoundError(f"--resume requested but no checkpoint found in {args.ckpt}")

        for i, ph in enumerate(phases):
            if int(ph["steps"]) <= 0:
                continue
            print(f"[CURRICULUM] {ph['name']} steps={ph['steps']} eval_K={ph['eval_K']} tf={ph['tf']} lambda_day={ph['lambda_day']} step_cost={ph['step_cost']} bg={ph['bg_mode']} pole={ph['use_pole']}")
            phase_log_dir = os.path.join(tb_root, ph["name"])
            tb_cb.set_log_dir(phase_log_dir, tag_prefix=f"{run_tag}/{ph['name']}")
            ep_cb.set_log_dir(phase_log_dir)
            step_cb.set_log_dir(phase_log_dir)

            args.lambda_day = float(ph["lambda_day"])
            args.step_cost = float(ph["step_cost"])
            vecnorm_path = None
            if model is None and resume_ckpt:
                vecnorm_path = find_vecnormalize_for_checkpoint(resume_ckpt)
            elif model is not None:
                prior_env = model.get_env()
                if not isinstance(prior_env, VecNormalize):
                    raise RuntimeError("multiphase transition expected a VecNormalize environment")
                vecnorm_path = os.path.join(args.ckpt, "vecnormalize_phase_transition.pkl")
                prior_env.save(vecnorm_path)
            env = build_env(
                eval_K=int(ph["eval_K"]),
                bg_mode=ph["bg_mode"],
                use_pole=bool(ph["use_pole"]),
                transform_strength=float(ph["tf"]),
                vecnorm_path=vecnorm_path,
            )

            if model is None:
                if resume_ckpt:
                    print(f" Resuming model={resume_ckpt} vecnormalize={vecnorm_path}")
                    model = MaskablePPO.load(resume_ckpt, env=env, device="auto")
                else:
                    policy_kwargs = build_policy_kwargs(args.cnn)
                    model = MaskablePPO(
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
                        seed=int(args.seed),
                    )
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
    # Save VecNormalize stats if present for evaluation.
    venv = model.get_env()
    if isinstance(venv, VecNormalize):
        vec_path = os.path.join(args.ckpt, "vecnormalize.pkl")
        venv.save(vec_path)
        print(f" Saved VecNormalize stats to {vec_path}")
    print(f" Saved final model to {final}")
