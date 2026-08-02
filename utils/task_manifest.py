"""Strict task-distribution manifests for amortized stencil experiments.

The manifest separates research tasks from command-line accidentals.  It also
performs the split-leakage checks that an integer RNG seed cannot provide.
Paths are resolved relative to the manifest, but files are not opened here so a
manifest can be inspected on a machine that does not hold restricted assets.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

from envs.attack_objective import ATTACK_MODES


SCHEMA_VERSION = 1
TASK_SPLITS = ("train", "development", "calibration", "certification")
DEFAULT_LEAKAGE_KEYS = (
    "sign_instance_id",
    "background_split_id",
    "camera_id",
    "material_batch_id",
    "physical_run_group",
)

_ROOT_KEYS = {
    "schema_version",
    "manifest_id",
    "description",
    "condition_feature_names",
    "leakage_keys",
    "tasks",
}
_TASK_KEYS = {
    "task_id",
    "split",
    "sign_instance_id",
    "background_split_id",
    "camera_id",
    "material_batch_id",
    "physical_run_group",
    "detector_id",
    "source_class",
    "target_class",
    "attack_mode",
    "calibration_sha256",
    "condition_features",
    "weight",
    "environment",
}
_ENVIRONMENT_KEYS = {
    "data",
    "bgdir",
    "bg_mode",
    "no_pole",
    "sign_profile",
    "sign_image",
    "sign_active_image",
    "detector",
    "detector_model",
    "yolo_version",
    "yolo_weights",
    "detector_device",
    "paint",
    "paint_list",
    "paint_action_mode",
    "paint_palette",
    "episode_steps",
    "eval_K",
    "grid_cell",
    "cell_cover_thresh",
    "uv_threshold",
    "success_conf",
    "target_conf",
    "min_attack_success_rate",
    "min_clean_detection_rate",
    "localization_iou",
    "require_source_suppression",
    "require_day_preservation",
    "day_tolerance",
    "lambda_day",
    "lambda_area",
    "lambda_iou",
    "lambda_misclass",
    "lambda_efficiency",
    "efficiency_eps",
    "lambda_perceptual",
    "area_target",
    "area_cap_frac",
    "area_cap_penalty",
    "area_cap_mode",
    "step_cost",
    "step_cost_after_target",
    "transform_strength",
    "fixed_angle_deg",
    "obs_size",
    "obs_margin",
    "obs_include_mask",
    "detector_debug",
    "allowed_alternative_classes",
    "physics_calibration",
}
_PATH_FIELDS = {
    "data",
    "bgdir",
    "sign_image",
    "sign_active_image",
    "yolo_weights",
    "physics_calibration",
}


class TaskManifestError(ValueError):
    """Raised when a task manifest is ambiguous, leaky, or malformed."""


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _reject_constant(value: str) -> None:
    raise TaskManifestError(f"manifest contains non-finite constant {value!r}")


def _no_duplicate_keys(pairs: Sequence[Tuple[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise TaskManifestError(f"manifest contains duplicate key {key!r}")
        result[key] = value
    return result


def _strict_keys(value: Mapping[str, Any], allowed: Iterable[str], path: str) -> None:
    allowed_set = set(allowed)
    unknown = sorted(set(value) - allowed_set)
    missing = sorted(allowed_set - set(value))
    if unknown or missing:
        parts = []
        if missing:
            parts.append(f"missing {missing}")
        if unknown:
            parts.append(f"unknown {unknown}")
        raise TaskManifestError(f"{path} has invalid fields: {', '.join(parts)}")


def _string(value: Any, path: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise TaskManifestError(f"{path} must be a string")
    result = value.strip()
    if not result and not allow_empty:
        raise TaskManifestError(f"{path} must not be empty")
    return result


def _optional_string(value: Any, path: str) -> Optional[str]:
    if value is None:
        return None
    result = _string(value, path, allow_empty=True)
    return result or None


def _finite_number(value: Any, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TaskManifestError(f"{path} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise TaskManifestError(f"{path} must be finite")
    return result


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    split: str
    sign_instance_id: str
    background_split_id: str
    camera_id: str
    material_batch_id: str
    physical_run_group: str
    detector_id: str
    source_class: str
    target_class: Optional[str]
    attack_mode: str
    calibration_sha256: str
    condition_features: Tuple[float, ...]
    weight: float
    environment: Mapping[str, Any]

    @property
    def source_target_pair(self) -> str:
        return f"{self.source_class}->{self.target_class or '*'}"

    def resolved_environment(self, manifest_directory: Path) -> Dict[str, Any]:
        result = dict(self.environment)
        for key in _PATH_FIELDS:
            value = result.get(key)
            if not isinstance(value, str) or not value.strip():
                continue
            path = Path(value)
            if not path.is_absolute():
                path = manifest_directory / path
            result[key] = str(path.resolve())
        result.update(
            {
                "source_class": self.source_class,
                "attack_mode": self.attack_mode,
                "attack_target_class": self.target_class or "",
            }
        )
        return result


@dataclass(frozen=True)
class TaskManifest:
    manifest_id: str
    description: str
    condition_feature_names: Tuple[str, ...]
    leakage_keys: Tuple[str, ...]
    tasks: Tuple[TaskSpec, ...]
    canonical_sha256: str
    source_sha256: str
    source_path: str

    @property
    def directory(self) -> Path:
        return Path(self.source_path).resolve().parent

    def tasks_for_split(self, split: str) -> Tuple[TaskSpec, ...]:
        requested = str(split).strip().lower()
        if requested not in TASK_SPLITS:
            raise TaskManifestError(f"unknown task split {split!r}")
        return tuple(task for task in self.tasks if task.split == requested)

    def task_by_id(self, task_id: str) -> TaskSpec:
        matches = [task for task in self.tasks if task.task_id == task_id]
        if not matches:
            raise KeyError(f"task_id {task_id!r} is not present in the manifest")
        return matches[0]


def _parse_task(
    value: Any,
    *,
    index: int,
    feature_count: int,
) -> TaskSpec:
    path = f"$.tasks[{index}]"
    if not isinstance(value, dict):
        raise TaskManifestError(f"{path} must be an object")
    _strict_keys(value, _TASK_KEYS, path)

    split = _string(value["split"], f"{path}.split").lower()
    if split not in TASK_SPLITS:
        raise TaskManifestError(
            f"{path}.split must be one of {', '.join(TASK_SPLITS)}"
        )
    mode = _string(value["attack_mode"], f"{path}.attack_mode")
    if mode not in ATTACK_MODES:
        raise TaskManifestError(
            f"{path}.attack_mode must be one of {', '.join(ATTACK_MODES)}"
        )
    source = _string(value["source_class"], f"{path}.source_class")
    target = _optional_string(value["target_class"], f"{path}.target_class")
    if mode == "targeted_misclassification" and target is None:
        raise TaskManifestError(f"{path}.target_class is required for targeted mode")
    if mode != "targeted_misclassification" and target is not None:
        raise TaskManifestError(
            f"{path}.target_class must be null unless attack_mode is "
            "targeted_misclassification"
        )
    if target is not None and target.casefold() == source.casefold():
        raise TaskManifestError(f"{path}.target_class must differ from source_class")

    calibration_hash = _string(
        value["calibration_sha256"], f"{path}.calibration_sha256", allow_empty=True
    ).lower()
    if calibration_hash and (
        len(calibration_hash) != 64
        or any(character not in "0123456789abcdef" for character in calibration_hash)
    ):
        raise TaskManifestError(
            f"{path}.calibration_sha256 must be empty or 64 lowercase hex characters"
        )

    features_value = value["condition_features"]
    if not isinstance(features_value, list) or len(features_value) != feature_count:
        raise TaskManifestError(
            f"{path}.condition_features must contain {feature_count} values"
        )
    features = tuple(
        _finite_number(item, f"{path}.condition_features[{item_index}]")
        for item_index, item in enumerate(features_value)
    )
    weight = _finite_number(value["weight"], f"{path}.weight")
    if weight <= 0.0:
        raise TaskManifestError(f"{path}.weight must be > 0")

    environment = value["environment"]
    if not isinstance(environment, dict):
        raise TaskManifestError(f"{path}.environment must be an object")
    unknown_environment = sorted(set(environment) - _ENVIRONMENT_KEYS)
    if unknown_environment:
        raise TaskManifestError(
            f"{path}.environment has unknown fields {unknown_environment}"
        )

    return TaskSpec(
        task_id=_string(value["task_id"], f"{path}.task_id"),
        split=split,
        sign_instance_id=_string(
            value["sign_instance_id"], f"{path}.sign_instance_id"
        ),
        background_split_id=_string(
            value["background_split_id"], f"{path}.background_split_id"
        ),
        camera_id=_string(value["camera_id"], f"{path}.camera_id"),
        material_batch_id=_string(
            value["material_batch_id"], f"{path}.material_batch_id"
        ),
        physical_run_group=_string(
            value["physical_run_group"], f"{path}.physical_run_group"
        ),
        detector_id=_string(value["detector_id"], f"{path}.detector_id"),
        source_class=source,
        target_class=target,
        attack_mode=mode,
        calibration_sha256=calibration_hash,
        condition_features=features,
        weight=weight,
        environment=dict(environment),
    )


def _validate_split_leakage(tasks: Sequence[TaskSpec], leakage_keys: Sequence[str]) -> None:
    valid_keys = set(DEFAULT_LEAKAGE_KEYS) | {"detector_id", "source_target_pair"}
    unknown = sorted(set(leakage_keys) - valid_keys)
    if unknown:
        raise TaskManifestError(f"leakage_keys contains unsupported fields {unknown}")
    for key in leakage_keys:
        seen: Dict[str, str] = {}
        for task in tasks:
            value = (
                task.source_target_pair
                if key == "source_target_pair"
                else str(getattr(task, key))
            )
            prior_split = seen.get(value)
            if prior_split is not None and prior_split != task.split:
                raise TaskManifestError(
                    f"split leakage: {key}={value!r} occurs in both "
                    f"{prior_split!r} and {task.split!r}"
                )
            seen[value] = task.split


def load_task_manifest(path: str | Path) -> TaskManifest:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"task manifest not found: {source}")
    source_bytes = source.read_bytes()
    try:
        payload = json.loads(
            source_bytes.decode("utf-8"),
            parse_constant=_reject_constant,
            object_pairs_hook=_no_duplicate_keys,
        )
    except UnicodeDecodeError as exc:
        raise TaskManifestError("task manifest must be UTF-8 JSON") from exc
    if not isinstance(payload, dict):
        raise TaskManifestError("task manifest root must be an object")
    _strict_keys(payload, _ROOT_KEYS, "$")
    if payload["schema_version"] != SCHEMA_VERSION:
        raise TaskManifestError(
            f"schema_version must be {SCHEMA_VERSION}, got {payload['schema_version']!r}"
        )

    names_value = payload["condition_feature_names"]
    if not isinstance(names_value, list):
        raise TaskManifestError("$.condition_feature_names must be an array")
    names = tuple(
        _string(item, f"$.condition_feature_names[{index}]")
        for index, item in enumerate(names_value)
    )
    if len(set(names)) != len(names):
        raise TaskManifestError("condition_feature_names must be unique")

    leakage_value = payload["leakage_keys"]
    if not isinstance(leakage_value, list) or not leakage_value:
        raise TaskManifestError("$.leakage_keys must be a non-empty array")
    leakage_keys = tuple(
        _string(item, f"$.leakage_keys[{index}]")
        for index, item in enumerate(leakage_value)
    )
    if len(set(leakage_keys)) != len(leakage_keys):
        raise TaskManifestError("leakage_keys must be unique")

    tasks_value = payload["tasks"]
    if not isinstance(tasks_value, list) or not tasks_value:
        raise TaskManifestError("$.tasks must be a non-empty array")
    tasks = tuple(
        _parse_task(item, index=index, feature_count=len(names))
        for index, item in enumerate(tasks_value)
    )
    task_ids = [task.task_id for task in tasks]
    if len(set(task_ids)) != len(task_ids):
        raise TaskManifestError("task_id values must be unique")
    _validate_split_leakage(tasks, leakage_keys)

    canonical = _canonical_json(payload)
    return TaskManifest(
        manifest_id=_string(payload["manifest_id"], "$.manifest_id"),
        description=_string(payload["description"], "$.description"),
        condition_feature_names=names,
        leakage_keys=leakage_keys,
        tasks=tasks,
        canonical_sha256=hashlib.sha256(canonical).hexdigest(),
        source_sha256=hashlib.sha256(source_bytes).hexdigest(),
        source_path=str(source.resolve()),
    )


__all__ = [
    "DEFAULT_LEAKAGE_KEYS",
    "SCHEMA_VERSION",
    "TASK_SPLITS",
    "TaskManifest",
    "TaskManifestError",
    "TaskSpec",
    "load_task_manifest",
]
