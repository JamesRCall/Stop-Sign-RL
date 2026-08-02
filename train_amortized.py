"""Train one prefix-valid optimizer across a preregistered task distribution.

This is the research entry point for the amortized method.  The legacy
``train_traffic_sign.py`` path still trains one scene-conditioned task and is
not evidence of cross-task amortization.
"""
from __future__ import annotations

import argparse
from collections import deque
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import random
import sys
from types import SimpleNamespace
from typing import Any, Dict

import numpy as np

from baselines.grid_utils import build_env_from_args, resolve_yolo_weights
from envs.amortized_traffic_sign_env import AmortizedTask, AmortizedTrafficSignEnv
from utils.fluorescence_transport import (
    FluorescenceCalibration,
    compute_fluorescence_transport,
    load_fluorescence_calibration,
)
from utils.experiment_manifest import file_record, package_versions, write_manifest
from utils.sign_assets import resolve_sign_assets
from utils.task_manifest import TASK_SPLITS, TaskManifest, TaskSpec, load_task_manifest
from utils.uv_paint import UVPaint


DEFAULT_ENVIRONMENT: Dict[str, Any] = {
    "data": "./data",
    "bgdir": "./data/backgrounds",
    "bg_mode": "dataset",
    "no_pole": False,
    "sign_profile": "custom",
    "sign_image": "",
    "sign_active_image": "",
    "detector": "yolo",
    "detector_model": "",
    "yolo_version": "8",
    "yolo_weights": None,
    "detector_device": "cpu",
    "paint": "yellow",
    "paint_list": "",
    "paint_action_mode": "fixed",
    "paint_palette": "",
    "episode_steps": 64,
    "eval_K": 3,
    "grid_cell": 16,
    "cell_cover_thresh": 0.60,
    "uv_threshold": 0.75,
    "success_conf": 0.20,
    "target_conf": 0.40,
    "min_attack_success_rate": 0.80,
    "min_clean_detection_rate": 0.80,
    "localization_iou": 0.30,
    "require_source_suppression": 1,
    "require_day_preservation": 1,
    "day_tolerance": 0.05,
    "lambda_day": 1.0,
    "lambda_area": 0.70,
    "lambda_iou": 0.40,
    "lambda_misclass": 0.60,
    "lambda_efficiency": 0.40,
    "efficiency_eps": 0.02,
    "lambda_perceptual": 0.0,
    "area_target": 0.25,
    "area_cap_frac": 0.30,
    "area_cap_penalty": -0.20,
    "area_cap_mode": "hard",
    "step_cost": 0.012,
    "step_cost_after_target": 0.14,
    "transform_strength": 1.0,
    "fixed_angle_deg": None,
    "obs_size": 224,
    "obs_margin": 0.10,
    "obs_include_mask": 1,
    "detector_debug": 0,
    "allowed_alternative_classes": "",
}


class RollingSuccessGate:
    """Bounded, auditable stopping rule over completed training episodes.

    Training never runs indefinitely: ``total_steps`` remains the hard upper
    bound.  When enabled, this gate permits an earlier stop only after a full
    rolling window of terminal episodes reaches the declared joint-success
    rate and the minimum policy-step count has been met.
    """

    def __init__(
        self,
        *,
        success_rate: float,
        window: int,
        minimum_steps: int,
    ) -> None:
        self.success_rate = float(success_rate)
        self.window = int(window)
        self.minimum_steps = int(minimum_steps)
        if not 0.0 <= self.success_rate <= 1.0:
            raise ValueError("early-stop-success-rate must be in [0, 1]")
        if self.window < 1:
            raise ValueError("early-stop-window must be >= 1")
        if self.minimum_steps < 0:
            raise ValueError("minimum-steps must be >= 0")
        self.history: deque[bool] = deque(maxlen=self.window)
        self.terminal_episodes = 0
        self.triggered = False
        self.trigger_step: int | None = None

    @property
    def enabled(self) -> bool:
        return self.success_rate > 0.0

    @property
    def rolling_rate(self) -> float | None:
        if not self.history:
            return None
        return float(sum(self.history) / len(self.history))

    def observe(self, *, success: bool, policy_steps: int) -> bool:
        self.terminal_episodes += 1
        self.history.append(bool(success))
        ready = bool(
            self.enabled
            and int(policy_steps) >= self.minimum_steps
            and len(self.history) == self.window
            and float(self.rolling_rate or 0.0) >= self.success_rate
        )
        if ready and not self.triggered:
            self.triggered = True
            self.trigger_step = int(policy_steps)
        return ready

    def report(self, *, final_policy_steps: int) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "required_rolling_success_rate": self.success_rate,
            "window_terminal_episodes": self.window,
            "minimum_policy_steps": self.minimum_steps,
            "terminal_episodes_observed": self.terminal_episodes,
            "final_window_size": len(self.history),
            "final_rolling_success_rate": self.rolling_rate,
            "triggered": self.triggered,
            "trigger_policy_step": self.trigger_step,
            "final_policy_steps": int(final_policy_steps),
            "stop_reason": (
                "rolling_joint_success_gate"
                if self.triggered
                else "maximum_total_steps"
            ),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train one task-conditioned, prefix-valid support-batch policy. "
            "Only manifest tasks labeled train are sampled."
        )
    )
    parser.add_argument("--task-manifest", required=True)
    parser.add_argument("--support-scenes", type=int, default=4)
    parser.add_argument("--risk-alpha", type=float, default=0.25)
    parser.add_argument("--required-support-success-rate", type=float, default=0.80)
    parser.add_argument("--required-clean-eligible-rate", type=float, default=0.80)
    parser.add_argument("--required-day-preservation-rate", type=float, default=0.80)
    parser.add_argument("--dual-learning-rate", type=float, default=0.05)
    parser.add_argument("--total-steps", type=int, default=800_000)
    parser.add_argument("--n-steps", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=2.0e-4)
    parser.add_argument("--ent-coef", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", default="./runs/amortized")
    parser.add_argument("--save-freq", type=int, default=100_000)
    parser.add_argument(
        "--minimum-steps",
        type=int,
        default=0,
        help="Minimum policy steps before the optional rolling success gate may stop training.",
    )
    parser.add_argument(
        "--early-stop-success-rate",
        type=float,
        default=0.0,
        help=(
            "Joint-success rate required across a full terminal-episode window; "
            "0 disables early stopping and total-steps remains the hard limit."
        ),
    )
    parser.add_argument(
        "--early-stop-window",
        type=int,
        default=50,
        help="Number of completed episodes in the optional rolling success gate.",
    )
    parser.add_argument("--check-env", action="store_true")
    parser.add_argument(
        "--allow-uncalibrated-simulation",
        action="store_true",
        help="Debug only: permit tasks without a validated physics calibration.",
    )
    parser.add_argument(
        "--allow-single-task-debug",
        action="store_true",
        help="Debug only: bypass the requirement for at least two training tasks.",
    )
    parser.add_argument(
        "--allow-incomplete-splits",
        action="store_true",
        help="Debug only: allow missing development/calibration/certification tasks.",
    )
    return parser.parse_args()


def _detector_signature(config: Dict[str, Any]) -> str:
    fields = (
        "detector",
        "detector_model",
        "yolo_version",
        "yolo_weights",
        "detector_device",
    )
    return json.dumps(
        {key: config.get(key) for key in fields},
        sort_keys=True,
        separators=(",", ":"),
    )


def _detector_class_binding(environment: Any) -> Dict[str, Any]:
    """Return an auditable, per-environment detector/class binding record."""
    import hashlib

    base = getattr(environment, "unwrapped", environment)
    detector = getattr(base, "det", None)
    if detector is None:
        raise RuntimeError("training environment does not expose its detector")

    raw_names = getattr(detector, "id_to_name", {}) or {}
    id_to_name = {
        int(class_id): str(class_name)
        for class_id, class_name in dict(raw_names).items()
    }
    canonical_names = json.dumps(
        [[class_id, id_to_name[class_id]] for class_id in sorted(id_to_name)],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    source_id = int(getattr(base, "source_class_id"))
    attack_target_id = getattr(base, "attack_target_id", None)
    return {
        "detector_runtime_type": (
            f"{type(detector).__module__}.{type(detector).__qualname__}"
        ),
        "label_map_class_count": len(id_to_name),
        "label_map_sha256": hashlib.sha256(canonical_names).hexdigest(),
        "id_to_name": {str(class_id): id_to_name[class_id] for class_id in sorted(id_to_name)},
        "source_class_id": source_id,
        "source_class_name": id_to_name.get(source_id),
        "attack_target_class_id": (
            int(attack_target_id) if attack_target_id is not None else None
        ),
        "attack_target_class_name": (
            id_to_name.get(int(attack_target_id))
            if attack_target_id is not None
            else None
        ),
        # Kept only to expose accidental dependence on the wrapper's legacy
        # confidence-only binding when one model is shared across source tasks.
        "legacy_wrapper_target_id": (
            int(detector.target_id) if hasattr(detector, "target_id") else None
        ),
    }


def _task_environment_config(
    manifest: TaskManifest,
    task: TaskSpec,
    *,
    seed: int,
    detector_instance: Any = None,
) -> Dict[str, Any]:
    config = dict(DEFAULT_ENVIRONMENT)
    config.update(task.resolved_environment(manifest.directory))
    config.update(
        {
            "seed": int(seed),
            "action_indexing": "canonical_full_grid",
            "terminate_on_success": 0,
            "detector_instance": detector_instance,
        }
    )
    return config


def _validate_research_splits(manifest: TaskManifest, args: argparse.Namespace) -> None:
    train_tasks = manifest.tasks_for_split("train")
    if len(train_tasks) < 2 and not args.allow_single_task_debug:
        raise ValueError(
            "amortized training requires at least two train tasks; use "
            "--allow-single-task-debug only for software checks"
        )
    if not args.allow_incomplete_splits:
        missing = [
            split
            for split in TASK_SPLITS[1:]
            if not manifest.tasks_for_split(split)
        ]
        if missing:
            raise ValueError(
                "paper-facing manifests require development, calibration, and "
                f"certification tasks; missing {missing}"
            )


def _validate_calibration(
    manifest: TaskManifest,
    task: TaskSpec,
    *,
    allow_uncalibrated: bool,
) -> FluorescenceCalibration | None:
    config = task.resolved_environment(manifest.directory)
    calibration_path = str(config.get("physics_calibration", "") or "").strip()
    if not calibration_path or not task.calibration_sha256:
        if allow_uncalibrated:
            return None
        raise ValueError(
            f"task {task.task_id!r} lacks a physics calibration path/hash; "
            "uncalibrated RGB constants cannot support the proposed method"
        )
    calibration = load_fluorescence_calibration(calibration_path)
    if calibration.canonical_sha256 != task.calibration_sha256:
        raise ValueError(
            f"task {task.task_id!r} calibration hash mismatch: manifest "
            f"{task.calibration_sha256}, loaded {calibration.canonical_sha256}"
        )
    if not calibration.provenance.is_empirical and not allow_uncalibrated:
        raise ValueError(
            f"task {task.task_id!r} uses a non-empirical calibration fixture; "
            "replace it with measured data or pass --allow-uncalibrated-simulation "
            "for a debug run that cannot support physical claims"
        )
    return calibration


def _linear_channel_to_srgb8(value: float) -> int:
    channel = max(0.0, min(float(value), 1.0))
    encoded = (
        12.92 * channel
        if channel <= 0.0031308
        else 1.055 * (channel ** (1.0 / 2.4)) - 0.055
    )
    return int(round(255.0 * max(0.0, min(encoded, 1.0))))


def _transport_paint(calibration: FluorescenceCalibration, seed: int) -> tuple[UVPaint, Any]:
    result = compute_fluorescence_transport(calibration, seed=int(seed))

    def hex_color(values) -> str:
        channels = [_linear_channel_to_srgb8(value) for value in values]
        return "#" + "".join(f"{channel:02X}" for channel in channels)

    paint = UVPaint(
        name=f"SpectralCalibration:{calibration.provenance.calibration_id}:{seed}",
        day_hex=hex_color(result.day_linear_rgb),
        active_hex=hex_color(result.triggered_linear_rgb),
        translucent=False,
        # The transport already models substrate plus coating radiance.  It is
        # therefore an effective coated-cell color, not an alpha blend against
        # the uncoated RGB asset.
        day_alpha=1.0,
        active_alpha=1.0,
    )
    return paint, result


def build_training_environment(
    manifest: TaskManifest,
    args: argparse.Namespace,
) -> AmortizedTrafficSignEnv:
    if int(args.support_scenes) < 2 and not args.allow_single_task_debug:
        raise ValueError("support-scenes must be >= 2 for a support-batch claim")
    if int(args.support_scenes) < 1:
        raise ValueError("support-scenes must be >= 1")
    _validate_research_splits(manifest, args)

    return build_amortized_environment(manifest, args, split="train")


def build_amortized_environment(
    manifest: TaskManifest,
    args: argparse.Namespace,
    *,
    split: str,
) -> AmortizedTrafficSignEnv:
    """Build one split-specific wrapper without crossing its task boundary."""
    requested_split = str(split).strip().lower()
    split_tasks = manifest.tasks_for_split(requested_split)
    if not split_tasks:
        raise ValueError(f"task manifest has no {requested_split!r} tasks")
    if int(args.support_scenes) < 1:
        raise ValueError("support-scenes must be >= 1")

    detector_cache: Dict[str, Any] = {}
    detector_signatures: Dict[str, str] = {}
    task_batches = []
    for task_index, task in enumerate(split_tasks):
        calibration = _validate_calibration(
            manifest,
            task,
            allow_uncalibrated=bool(args.allow_uncalibrated_simulation),
        )
        replicas = []
        for replica_index in range(int(args.support_scenes)):
            replica_seed = int(args.seed) + 100_003 * task_index + 997 * replica_index
            cached_detector = detector_cache.get(task.detector_id)
            config = _task_environment_config(
                manifest,
                task,
                seed=replica_seed,
                detector_instance=cached_detector,
            )
            transport_result = None
            if calibration is not None:
                paint, transport_result = _transport_paint(calibration, replica_seed)
                config["uv_paint_instance"] = paint
            signature = _detector_signature(config)
            prior_signature = detector_signatures.get(task.detector_id)
            if prior_signature is not None and prior_signature != signature:
                raise ValueError(
                    f"detector_id {task.detector_id!r} maps to multiple detector configs"
                )
            env = build_env_from_args(SimpleNamespace(**config))
            if transport_result is not None:
                env.physics_calibration_sha256 = calibration.canonical_sha256
                env.physics_transport_seed = int(replica_seed)
                env.physics_transport_result = transport_result.to_dict()
            if cached_detector is None:
                detector_cache[task.detector_id] = env.det
                detector_signatures[task.detector_id] = signature
            replicas.append(env)
        task_batches.append(AmortizedTask(task, tuple(replicas)))

    return AmortizedTrafficSignEnv(
        task_batches,
        allowed_splits=(requested_split,),
        risk_alpha=float(args.risk_alpha),
        required_support_success_rate=float(args.required_support_success_rate),
        required_clean_eligible_rate=float(args.required_clean_eligible_rate),
        required_day_preservation_rate=float(args.required_day_preservation_rate),
        dual_learning_rate=float(args.dual_learning_rate),
        initial_constraint_duals=getattr(args, "initial_constraint_duals", None),
        terminate_on_support_success=bool(
            getattr(args, "terminate_on_support_success", True)
        ),
        seed=int(args.seed),
    )


def _write_run_manifest(
    output_dir: Path,
    manifest: TaskManifest,
    args: argparse.Namespace,
    *,
    environment: AmortizedTrafficSignEnv,
) -> None:
    task_inputs = []
    environments_by_task = {
        task.spec.task_id: task.support_envs for task in environment.tasks
    }
    expected_task_ids = {
        task.task_id for task in manifest.tasks_for_split("train")
    }
    if set(environments_by_task) != expected_task_ids:
        raise RuntimeError(
            "training environment tasks do not match the manifest train split"
        )
    supported_images = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
    for task_index, task in enumerate(manifest.tasks_for_split("train")):
        config = _task_environment_config(
            manifest,
            task,
            seed=int(args.seed) + 100_003 * task_index,
        )
        backgrounds = []
        background_dir = Path(str(config.get("bgdir", "")))
        if str(config.get("bg_mode", "dataset")) == "dataset" and background_dir.is_dir():
            backgrounds = [
                file_record(path)
                for path in sorted(background_dir.iterdir())
                if path.is_file() and path.suffix.lower() in supported_images
            ]

        def optional_record(value: Any) -> Dict[str, Any] | None:
            if not isinstance(value, str) or not value.strip():
                return None
            return file_record(value)

        data_dir = Path(str(config.get("data", "./data")))
        pole_path = data_dir / "pole.png"
        detector_weights = None
        if str(config.get("detector", "yolo")).strip().lower() == "yolo":
            detector_weights = resolve_yolo_weights(
                str(config.get("yolo_version", "8")),
                config.get("yolo_weights"),
            )
        sign_assets = resolve_sign_assets(
            data_dir=data_dir,
            profile=str(config.get("sign_profile", "custom")),
            sign_image=(config.get("sign_image") or None),
            sign_active_image=(config.get("sign_active_image") or None),
            source_class=task.source_class,
        )
        class_bindings = [
            _detector_class_binding(support_env)
            for support_env in environments_by_task[task.task_id]
        ]
        if not class_bindings or any(
            binding != class_bindings[0] for binding in class_bindings[1:]
        ):
            raise RuntimeError(
                f"task {task.task_id!r} has inconsistent detector class bindings "
                "across support scenes"
            )
        task_inputs.append(
            {
                "task_id": task.task_id,
                "detector_id": task.detector_id,
                "source_class": task.source_class,
                "attack_target_class": task.target_class,
                "detector_backend": str(config.get("detector", "")),
                "detector_model_identifier": str(
                    config.get("detector_model", "") or ""
                ),
                "detector_weights": optional_record(detector_weights),
                "detector_class_binding": class_bindings[0],
                "sign_day": file_record(sign_assets.day_image),
                "sign_active": file_record(sign_assets.active_image),
                "physics_calibration": optional_record(
                    config.get("physics_calibration")
                ),
                "pole": (
                    file_record(pole_path)
                    if pole_path.is_file() and not bool(config.get("no_pole", False))
                    else None
                ),
                "backgrounds": backgrounds,
            }
        )
    record = {
        "schema_version": 1,
        "method": "task_amortized_prefix_valid_support_batch",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "task_manifest": {
            "path": manifest.source_path,
            "source_sha256": manifest.source_sha256,
            "canonical_sha256": manifest.canonical_sha256,
            "train_task_ids": [task.task_id for task in manifest.tasks_for_split("train")],
        },
        "config": vars(args),
        "training_inputs": task_inputs,
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "packages": package_versions(
                [
                    "numpy",
                    "Pillow",
                    "torch",
                    "torchvision",
                    "gymnasium",
                    "stable-baselines3",
                    "sb3-contrib",
                    "ultralytics",
                    "transformers",
                ]
            ),
        },
        "claim_limits": {
            "no_novelty_or_success_guarantee": True,
            "uncalibrated_simulation": bool(args.allow_uncalibrated_simulation),
            "single_task_debug": bool(args.allow_single_task_debug),
            "incomplete_splits": bool(args.allow_incomplete_splits),
            "offline_queries_are_not_online_queries": True,
        },
        "run_status": "initialized",
        "query_accounting": {
            "offline_training_detector_image_queries": None,
            "policy_environment_steps": None,
        },
        "final_constraint_duals": None,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_manifest(
        output_dir / "amortized_run_manifest.json",
        record,
    )


def _sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _finalize_run_manifest(
    output_dir: Path,
    *,
    environment: AmortizedTrafficSignEnv,
    policy_steps: int,
    training_stop: Dict[str, Any],
) -> None:
    path = output_dir / "amortized_run_manifest.json"
    record = json.loads(path.read_text(encoding="utf-8"))
    model_path = output_dir / "amortized_prefix_policy_final.zip"
    normalizer_path = output_dir / "vecnormalize_final.pkl"
    if not model_path.is_file() or not normalizer_path.is_file():
        raise RuntimeError("final policy artifacts are missing; query ledger not sealed")
    record["run_status"] = "completed"
    record["completed_utc"] = datetime.now(timezone.utc).isoformat()
    record["query_accounting"] = {
        "offline_training_detector_image_queries": int(
            environment._lifetime_queries
        ),
        "policy_environment_steps": int(policy_steps),
    }
    record["final_constraint_duals"] = {
        name: float(value) for name, value in environment._duals.items()
    }
    record["training_stop"] = dict(training_stop)
    record["final_artifacts"] = {
        "policy_sha256": _sha256_file(model_path),
        "vecnormalize_sha256": _sha256_file(normalizer_path),
    }
    write_manifest(path, record)


def main() -> None:
    args = parse_args()
    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    manifest = load_task_manifest(args.task_manifest)
    env = build_training_environment(manifest, args)
    output_dir = Path(args.output_dir).resolve()
    _write_run_manifest(output_dir, manifest, args, environment=env)

    # Heavy RL imports stay local so manifest/schema tooling remains usable in
    # lightweight review and CI environments.
    from sb3_contrib import MaskablePPO
    from sb3_contrib.common.wrappers import ActionMasker
    from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
    from stable_baselines3.common.env_checker import check_env
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecTransposeImage

    if args.check_env:
        check_env(env, warn=True)
        print("Amortized environment check completed.")
        env.close()
        return

    wrapped = ActionMasker(env, lambda value: value.unwrapped.action_masks())
    vector = DummyVecEnv([lambda: wrapped])
    vector = VecTransposeImage(vector)
    vector = VecNormalize(
        vector,
        norm_obs=True,
        norm_reward=False,
        clip_obs=5.0,
        norm_obs_keys=["task", "feedback"],
    )

    class SaveNormalizer(BaseCallback):
        def __init__(self, save_frequency: int):
            super().__init__()
            self.save_frequency = max(1, int(save_frequency))

        def _on_step(self) -> bool:
            if self.num_timesteps % self.save_frequency == 0:
                vector.save(
                    str(output_dir / f"vecnormalize_{self.num_timesteps}_steps.pkl")
                )
            return True

    checkpoint = CheckpointCallback(
        save_freq=max(1, int(args.save_freq)),
        save_path=str(output_dir),
        name_prefix="amortized_prefix_policy",
    )
    success_gate = RollingSuccessGate(
        success_rate=float(args.early_stop_success_rate),
        window=int(args.early_stop_window),
        minimum_steps=int(args.minimum_steps),
    )

    class StopOnRollingJointSuccess(BaseCallback):
        def _on_step(self) -> bool:
            infos = list(self.locals.get("infos", []) or [])
            dones = np.asarray(self.locals.get("dones", []), dtype=bool).reshape(-1)
            for index, done in enumerate(dones):
                if not bool(done):
                    continue
                info = infos[index] if index < len(infos) else {}
                if success_gate.observe(
                    success=bool(info.get("attack_success", False)),
                    policy_steps=int(self.num_timesteps),
                ):
                    return False
            return True

    success_callback = StopOnRollingJointSuccess()
    model = MaskablePPO(
        "MultiInputPolicy",
        vector,
        verbose=2,
        n_steps=int(args.n_steps),
        batch_size=int(args.batch_size),
        learning_rate=float(args.learning_rate),
        gamma=0.995,
        gae_lambda=0.95,
        ent_coef=float(args.ent_coef),
        tensorboard_log=str(output_dir / "tb"),
        seed=int(args.seed),
        device="auto",
    )
    try:
        model.learn(
            total_timesteps=int(args.total_steps),
            callback=[
                checkpoint,
                SaveNormalizer(int(args.save_freq)),
                success_callback,
            ],
            tb_log_name="task_amortized_prefix_valid",
        )
        model.save(str(output_dir / "amortized_prefix_policy_final"))
        vector.save(str(output_dir / "vecnormalize_final.pkl"))
        _finalize_run_manifest(
            output_dir,
            environment=env,
            policy_steps=int(model.num_timesteps),
            training_stop=success_gate.report(
                final_policy_steps=int(model.num_timesteps)
            ),
        )
    finally:
        vector.close()


if __name__ == "__main__":
    main()
