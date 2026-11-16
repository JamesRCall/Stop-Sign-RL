"""Strict configuration for staged, query-budgeted PatchRL experiments.

The configuration is intentionally small and closed: every field is required,
unknown fields are rejected, JSON duplicate keys and non-finite constants are
errors, and paper-facing compute caps are enforced while loading the file.
This makes the configuration itself a reviewable experiment contract rather
than a loose collection of trainer defaults.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence


SCHEMA_VERSION = 1
EXPERIMENT_STAGES = ("pilot", "paper")
MODEL_BACKENDS = ("yolo", "torchvision", "rtdetr")
ATTACK_GOALS = (
    "disappearance",
    "untargeted_misclassification",
    "targeted_misclassification",
)
QUERY_COUNTING_UNIT = "one_detector_evaluation_of_one_image"
QUERY_SCOPE = "per_meta_policy_seed"
MAX_PILOT_SEARCH_IMAGE_QUERIES = 25_000
MAX_PAPER_SEARCH_IMAGE_QUERIES = 50_000


class ConfigError(ValueError):
    """Raised when an experiment configuration is incomplete or ambiguous."""


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _reject_constant(value: str) -> None:
    raise ConfigError(f"configuration contains non-finite constant {value!r}")


def _unique_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ConfigError(f"configuration contains duplicate key {key!r}")
        result[key] = value
    return result


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigError(f"{path} must be an object")
    return value


def _strict_keys(value: Mapping[str, Any], expected: Iterable[str], path: str) -> None:
    expected_set = set(expected)
    missing = sorted(expected_set - set(value))
    unknown = sorted(set(value) - expected_set)
    if missing or unknown:
        parts = []
        if missing:
            parts.append(f"missing {missing}")
        if unknown:
            parts.append(f"unknown {unknown}")
        raise ConfigError(f"{path} has invalid fields: {', '.join(parts)}")


def _array(value: Any, path: str, *, nonempty: bool = False) -> list[Any]:
    if not isinstance(value, list):
        raise ConfigError(f"{path} must be an array")
    if nonempty and not value:
        raise ConfigError(f"{path} must not be empty")
    return value


def _string(value: Any, path: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise ConfigError(f"{path} must be a string")
    result = value.strip()
    if not result and not allow_empty:
        raise ConfigError(f"{path} must not be empty")
    return result


def _optional_string(value: Any, path: str) -> str | None:
    if value is None:
        return None
    result = _string(value, path, allow_empty=True)
    return result or None


def _boolean(value: Any, path: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigError(f"{path} must be a boolean")
    return value


def _integer(value: Any, path: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{path} must be an integer")
    if value < minimum:
        raise ConfigError(f"{path} must be >= {minimum}")
    return int(value)


def _number(
    value: Any,
    path: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
    minimum_inclusive: bool = True,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{path} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ConfigError(f"{path} must be finite")
    if minimum is not None:
        invalid = result < minimum if minimum_inclusive else result <= minimum
        if invalid:
            operator = ">=" if minimum_inclusive else ">"
            raise ConfigError(f"{path} must be {operator} {minimum}")
    if maximum is not None and result > maximum:
        raise ConfigError(f"{path} must be <= {maximum}")
    return result


def _identifier(value: Any, path: str) -> str:
    result = _string(value, path)
    if not re.fullmatch(r"[a-z0-9][a-z0-9_.-]*", result):
        raise ConfigError(f"{path} must be a lowercase safe identifier")
    return result


def _string_array(
    value: Any,
    path: str,
    *,
    nonempty: bool = False,
) -> tuple[str, ...]:
    raw = _array(value, path, nonempty=nonempty)
    result = tuple(_string(item, f"{path}[{index}]") for index, item in enumerate(raw))
    if len(result) != len(set(result)):
        raise ConfigError(f"{path} must not contain duplicates")
    return result


@dataclass(frozen=True)
class DetectorModelConfig:
    id: str
    backend: str
    checkpoint: str
    sampling_weight: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "backend": self.backend,
            "checkpoint": self.checkpoint,
            "sampling_weight": self.sampling_weight,
        }


@dataclass(frozen=True)
class AssetsConfig:
    sign_day: str
    sign_active: str
    backgrounds: str
    calibration: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "sign_day": self.sign_day,
            "sign_active": self.sign_active,
            "backgrounds": self.backgrounds,
            "calibration": self.calibration,
        }


@dataclass(frozen=True)
class AttackGoalConfig:
    mode: str
    enabled: bool
    source_class: str
    target_class: str | None
    alternative_classes: tuple[str, ...]
    source_confidence_threshold: float
    target_confidence_threshold: float
    minimum_eot_success_rate: float
    require_source_suppression: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "enabled": self.enabled,
            "source_class": self.source_class,
            "target_class": self.target_class,
            "alternative_classes": list(self.alternative_classes),
            "source_confidence_threshold": self.source_confidence_threshold,
            "target_confidence_threshold": self.target_confidence_threshold,
            "minimum_eot_success_rate": self.minimum_eot_success_rate,
            "require_source_suppression": self.require_source_suppression,
        }


@dataclass(frozen=True)
class MaterialBudgetConfig:
    palette: tuple[str, ...]
    maximum_area_fraction: float
    maximum_actions: int
    action_primitives: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "palette": list(self.palette),
            "maximum_area_fraction": self.maximum_area_fraction,
            "maximum_actions": self.maximum_actions,
            "action_primitives": list(self.action_primitives),
        }


@dataclass(frozen=True)
class QueryBudgetConfig:
    counting_unit: str
    scope: str
    search_image_queries: int
    feasibility_image_queries: int
    development_image_queries: int
    evaluation_image_queries: int
    checkpoints: tuple[int, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "counting_unit": self.counting_unit,
            "scope": self.scope,
            "search_image_queries": self.search_image_queries,
            "feasibility_image_queries": self.feasibility_image_queries,
            "development_image_queries": self.development_image_queries,
            "evaluation_image_queries": self.evaluation_image_queries,
            "checkpoints": list(self.checkpoints),
        }


@dataclass(frozen=True)
class PPOConfig:
    seeds: tuple[int, ...]
    initial_seed_count: int
    one_meta_policy_per_seed: bool
    maximum_environment_steps: int
    maximum_episode_actions: int
    rollout_steps: int
    batch_size: int
    vector_environments: int
    learning_rate: float
    gamma: float
    gae_lambda: float
    entropy_coefficient: float
    checkpoint_query_interval: int

    @property
    def initial_seeds(self) -> tuple[int, ...]:
        return self.seeds[: self.initial_seed_count]

    @property
    def expansion_seeds(self) -> tuple[int, ...]:
        return self.seeds[self.initial_seed_count :]

    def to_dict(self) -> dict[str, Any]:
        return {
            "initial_seed_count": self.initial_seed_count,
            "one_meta_policy_per_seed": self.one_meta_policy_per_seed,
            "maximum_environment_steps": self.maximum_environment_steps,
            "maximum_episode_actions": self.maximum_episode_actions,
            "rollout_steps": self.rollout_steps,
            "batch_size": self.batch_size,
            "vector_environments": self.vector_environments,
            "learning_rate": self.learning_rate,
            "gamma": self.gamma,
            "gae_lambda": self.gae_lambda,
            "entropy_coefficient": self.entropy_coefficient,
            "checkpoint_query_interval": self.checkpoint_query_interval,
        }


@dataclass(frozen=True)
class TrainingConfig:
    """Compute contract shared by pilot and paper runners."""

    query_budget: QueryBudgetConfig
    ppo: PPOConfig

    @property
    def search_image_queries(self) -> int:
        return self.query_budget.search_image_queries

    @property
    def maximum_environment_steps(self) -> int:
        return self.ppo.maximum_environment_steps

    def to_dict(self) -> dict[str, Any]:
        return {
            "query_budget": self.query_budget.to_dict(),
            "ppo": self.ppo.to_dict(),
        }


@dataclass(frozen=True)
class FeasibilityConfig:
    clean_scenes: int
    minimum_clean_detection_rate: float
    probe_patterns: int
    probe_scenes: int
    minimum_dense_progress: float
    minimum_source_confidence_drop: float
    no_progress_query_checkpoint: int
    target_event_query_checkpoint: int
    require_target_event_for_targeted: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "clean_scenes": self.clean_scenes,
            "minimum_clean_detection_rate": self.minimum_clean_detection_rate,
            "probe_patterns": self.probe_patterns,
            "probe_scenes": self.probe_scenes,
            "minimum_dense_progress": self.minimum_dense_progress,
            "minimum_source_confidence_drop": self.minimum_source_confidence_drop,
            "no_progress_query_checkpoint": self.no_progress_query_checkpoint,
            "target_event_query_checkpoint": self.target_event_query_checkpoint,
            "require_target_event_for_targeted": self.require_target_event_for_targeted,
        }


@dataclass(frozen=True)
class EvaluationConfig:
    episodes: int
    eot_samples: int
    confidence_level: float
    frozen_split: str
    minimum_success_rate: float
    minimum_confidence_interval_lower: float
    minimum_successful_pilot_seeds: int
    minimum_successful_seeds_per_model: int
    minimum_successful_models: int
    minimum_successful_goals: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "episodes": self.episodes,
            "eot_samples": self.eot_samples,
            "confidence_level": self.confidence_level,
            "frozen_split": self.frozen_split,
            "minimum_success_rate": self.minimum_success_rate,
            "minimum_confidence_interval_lower": self.minimum_confidence_interval_lower,
            "minimum_successful_pilot_seeds": self.minimum_successful_pilot_seeds,
            "minimum_successful_seeds_per_model": self.minimum_successful_seeds_per_model,
            "minimum_successful_models": self.minimum_successful_models,
            "minimum_successful_goals": self.minimum_successful_goals,
        }


@dataclass(frozen=True)
class ExperimentConfig:
    schema_version: int
    experiment_id: str
    stage: str
    description: str
    assets: AssetsConfig
    surrogate_models: tuple[DetectorModelConfig, ...]
    heldout_models: tuple[DetectorModelConfig, ...]
    attack_goals: tuple[AttackGoalConfig, ...]
    patch: MaterialBudgetConfig
    training: TrainingConfig
    feasibility: FeasibilityConfig
    evaluation: EvaluationConfig
    seeds: tuple[int, ...]
    source_path: str | None
    source_sha256: str
    canonical_sha256: str

    @property
    def enabled_goals(self) -> tuple[AttackGoalConfig, ...]:
        return tuple(goal for goal in self.attack_goals if goal.enabled)

    @property
    def model_ids(self) -> tuple[str, ...]:
        return tuple(model.id for model in (*self.surrogate_models, *self.heldout_models))

    @property
    def source_models(self) -> tuple[DetectorModelConfig, ...]:
        """Compatibility alias for older orchestration prototypes."""

        return self.surrogate_models

    @property
    def material_budget(self) -> MaterialBudgetConfig:
        return self.patch

    @property
    def query_budget(self) -> QueryBudgetConfig:
        return self.training.query_budget

    @property
    def ppo(self) -> PPOConfig:
        return self.training.ppo

    def goal(self, mode: str, *, require_enabled: bool = False) -> AttackGoalConfig:
        matches = [goal for goal in self.attack_goals if goal.mode == str(mode)]
        if not matches:
            raise ConfigError(f"attack goal {mode!r} is not declared")
        result = matches[0]
        if require_enabled and not result.enabled:
            raise ConfigError(f"attack goal {mode!r} is disabled")
        return result

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "experiment_id": self.experiment_id,
            "stage": self.stage,
            "description": self.description,
            "assets": self.assets.to_dict(),
            "surrogate_models": [model.to_dict() for model in self.surrogate_models],
            "heldout_models": [model.to_dict() for model in self.heldout_models],
            "attack_goals": [goal.to_dict() for goal in self.attack_goals],
            "patch": self.patch.to_dict(),
            "training": self.training.to_dict(),
            "feasibility": self.feasibility.to_dict(),
            "evaluation": self.evaluation.to_dict(),
            "seeds": list(self.seeds),
        }


_ROOT_KEYS = {
    "schema_version",
    "experiment_id",
    "stage",
    "description",
    "assets",
    "surrogate_models",
    "heldout_models",
    "attack_goals",
    "patch",
    "training",
    "feasibility",
    "evaluation",
    "seeds",
}
_ASSET_KEYS = {"sign_day", "sign_active", "backgrounds", "calibration"}
_TRAINING_KEYS = {"query_budget", "ppo"}
_MODEL_KEYS = {"id", "backend", "checkpoint", "sampling_weight"}
_GOAL_KEYS = {
    "mode",
    "enabled",
    "source_class",
    "target_class",
    "alternative_classes",
    "source_confidence_threshold",
    "target_confidence_threshold",
    "minimum_eot_success_rate",
    "require_source_suppression",
}
_MATERIAL_KEYS = {
    "palette",
    "maximum_area_fraction",
    "maximum_actions",
    "action_primitives",
}
_QUERY_KEYS = {
    "counting_unit",
    "scope",
    "search_image_queries",
    "feasibility_image_queries",
    "development_image_queries",
    "evaluation_image_queries",
    "checkpoints",
}
_PPO_KEYS = {
    "initial_seed_count",
    "one_meta_policy_per_seed",
    "maximum_environment_steps",
    "maximum_episode_actions",
    "rollout_steps",
    "batch_size",
    "vector_environments",
    "learning_rate",
    "gamma",
    "gae_lambda",
    "entropy_coefficient",
    "checkpoint_query_interval",
}
_FEASIBILITY_KEYS = {
    "clean_scenes",
    "minimum_clean_detection_rate",
    "probe_patterns",
    "probe_scenes",
    "minimum_dense_progress",
    "minimum_source_confidence_drop",
    "no_progress_query_checkpoint",
    "target_event_query_checkpoint",
    "require_target_event_for_targeted",
}
_EVALUATION_KEYS = {
    "episodes",
    "eot_samples",
    "confidence_level",
    "frozen_split",
    "minimum_success_rate",
    "minimum_confidence_interval_lower",
    "minimum_successful_pilot_seeds",
    "minimum_successful_seeds_per_model",
    "minimum_successful_models",
    "minimum_successful_goals",
}


def _parse_model(value: Any, path: str) -> DetectorModelConfig:
    raw = _mapping(value, path)
    _strict_keys(raw, _MODEL_KEYS, path)
    backend = _string(raw["backend"], f"{path}.backend").lower()
    if backend not in MODEL_BACKENDS:
        raise ConfigError(f"{path}.backend must be one of {list(MODEL_BACKENDS)}")
    return DetectorModelConfig(
        id=_identifier(raw["id"], f"{path}.id"),
        backend=backend,
        checkpoint=_string(raw["checkpoint"], f"{path}.checkpoint"),
        sampling_weight=_number(
            raw["sampling_weight"],
            f"{path}.sampling_weight",
            minimum=0.0,
            minimum_inclusive=False,
        ),
    )


def _parse_assets(value: Any, path: str) -> AssetsConfig:
    raw = _mapping(value, path)
    _strict_keys(raw, _ASSET_KEYS, path)
    return AssetsConfig(
        sign_day=_string(raw["sign_day"], f"{path}.sign_day"),
        sign_active=_string(raw["sign_active"], f"{path}.sign_active"),
        backgrounds=_string(raw["backgrounds"], f"{path}.backgrounds"),
        calibration=_optional_string(raw["calibration"], f"{path}.calibration"),
    )


def _parse_goal(value: Any, path: str) -> AttackGoalConfig:
    raw = _mapping(value, path)
    _strict_keys(raw, _GOAL_KEYS, path)
    mode = _string(raw["mode"], f"{path}.mode").lower()
    if mode not in ATTACK_GOALS:
        raise ConfigError(f"{path}.mode must be one of {list(ATTACK_GOALS)}")
    target = _optional_string(raw["target_class"], f"{path}.target_class")
    alternatives = _string_array(raw["alternative_classes"], f"{path}.alternative_classes")
    if mode == "targeted_misclassification":
        if target is None:
            raise ConfigError(f"{path}.target_class is required for targeted mode")
        if alternatives:
            raise ConfigError(f"{path}.alternative_classes must be empty for targeted mode")
    elif target is not None:
        raise ConfigError(f"{path}.target_class must be null outside targeted mode")
    if mode == "disappearance" and alternatives:
        raise ConfigError(f"{path}.alternative_classes must be empty for disappearance")
    return AttackGoalConfig(
        mode=mode,
        enabled=_boolean(raw["enabled"], f"{path}.enabled"),
        source_class=_string(raw["source_class"], f"{path}.source_class"),
        target_class=target,
        alternative_classes=alternatives,
        source_confidence_threshold=_number(
            raw["source_confidence_threshold"],
            f"{path}.source_confidence_threshold",
            minimum=0.0,
            maximum=1.0,
        ),
        target_confidence_threshold=_number(
            raw["target_confidence_threshold"],
            f"{path}.target_confidence_threshold",
            minimum=0.0,
            maximum=1.0,
        ),
        minimum_eot_success_rate=_number(
            raw["minimum_eot_success_rate"],
            f"{path}.minimum_eot_success_rate",
            minimum=0.0,
            maximum=1.0,
            minimum_inclusive=False,
        ),
        require_source_suppression=_boolean(
            raw["require_source_suppression"],
            f"{path}.require_source_suppression",
        ),
    )


def _parse_material(value: Any, path: str) -> MaterialBudgetConfig:
    raw = _mapping(value, path)
    _strict_keys(raw, _MATERIAL_KEYS, path)
    return MaterialBudgetConfig(
        palette=_string_array(raw["palette"], f"{path}.palette", nonempty=True),
        maximum_area_fraction=_number(
            raw["maximum_area_fraction"],
            f"{path}.maximum_area_fraction",
            minimum=0.0,
            maximum=1.0,
            minimum_inclusive=False,
        ),
        maximum_actions=_integer(raw["maximum_actions"], f"{path}.maximum_actions", minimum=1),
        action_primitives=_string_array(
            raw["action_primitives"], f"{path}.action_primitives", nonempty=True
        ),
    )


def _parse_query(value: Any, path: str) -> QueryBudgetConfig:
    raw = _mapping(value, path)
    _strict_keys(raw, _QUERY_KEYS, path)
    counting = _string(raw["counting_unit"], f"{path}.counting_unit")
    if counting != QUERY_COUNTING_UNIT:
        raise ConfigError(f"{path}.counting_unit must equal {QUERY_COUNTING_UNIT!r}")
    scope = _string(raw["scope"], f"{path}.scope")
    if scope != QUERY_SCOPE:
        raise ConfigError(f"{path}.scope must equal {QUERY_SCOPE!r}")
    checkpoints_raw = _array(raw["checkpoints"], f"{path}.checkpoints", nonempty=True)
    checkpoints = tuple(
        _integer(item, f"{path}.checkpoints[{index}]", minimum=1)
        for index, item in enumerate(checkpoints_raw)
    )
    if tuple(sorted(set(checkpoints))) != checkpoints:
        raise ConfigError(f"{path}.checkpoints must be strictly increasing and unique")
    result = QueryBudgetConfig(
        counting_unit=counting,
        scope=scope,
        search_image_queries=_integer(
            raw["search_image_queries"], f"{path}.search_image_queries", minimum=1
        ),
        feasibility_image_queries=_integer(
            raw["feasibility_image_queries"],
            f"{path}.feasibility_image_queries",
            minimum=1,
        ),
        development_image_queries=_integer(
            raw["development_image_queries"],
            f"{path}.development_image_queries",
            minimum=1,
        ),
        evaluation_image_queries=_integer(
            raw["evaluation_image_queries"],
            f"{path}.evaluation_image_queries",
            minimum=1,
        ),
        checkpoints=checkpoints,
    )
    if result.checkpoints[-1] != result.search_image_queries:
        raise ConfigError(
            f"{path}.checkpoints must end at the search_image_queries hard cap"
        )
    return result


def _parse_ppo(value: Any, path: str, *, seeds: tuple[int, ...]) -> PPOConfig:
    raw = _mapping(value, path)
    _strict_keys(raw, _PPO_KEYS, path)
    rollout_steps = _integer(raw["rollout_steps"], f"{path}.rollout_steps", minimum=1)
    batch_size = _integer(raw["batch_size"], f"{path}.batch_size", minimum=1)
    vector_environments = _integer(
        raw["vector_environments"], f"{path}.vector_environments", minimum=1
    )
    if (rollout_steps * vector_environments) % batch_size != 0:
        raise ConfigError(
            f"{path}.batch_size must divide rollout_steps * vector_environments"
        )
    return PPOConfig(
        seeds=seeds,
        initial_seed_count=_integer(
            raw["initial_seed_count"], f"{path}.initial_seed_count", minimum=1
        ),
        one_meta_policy_per_seed=_boolean(
            raw["one_meta_policy_per_seed"], f"{path}.one_meta_policy_per_seed"
        ),
        maximum_environment_steps=_integer(
            raw["maximum_environment_steps"],
            f"{path}.maximum_environment_steps",
            minimum=1,
        ),
        maximum_episode_actions=_integer(
            raw["maximum_episode_actions"],
            f"{path}.maximum_episode_actions",
            minimum=1,
        ),
        rollout_steps=rollout_steps,
        batch_size=batch_size,
        vector_environments=vector_environments,
        learning_rate=_number(
            raw["learning_rate"],
            f"{path}.learning_rate",
            minimum=0.0,
            maximum=1.0,
            minimum_inclusive=False,
        ),
        gamma=_number(
            raw["gamma"], f"{path}.gamma", minimum=0.0, maximum=1.0
        ),
        gae_lambda=_number(
            raw["gae_lambda"], f"{path}.gae_lambda", minimum=0.0, maximum=1.0
        ),
        entropy_coefficient=_number(
            raw["entropy_coefficient"],
            f"{path}.entropy_coefficient",
            minimum=0.0,
        ),
        checkpoint_query_interval=_integer(
            raw["checkpoint_query_interval"],
            f"{path}.checkpoint_query_interval",
            minimum=1,
        ),
    )


def _parse_training(
    value: Any,
    path: str,
    *,
    seeds: tuple[int, ...],
) -> TrainingConfig:
    raw = _mapping(value, path)
    _strict_keys(raw, _TRAINING_KEYS, path)
    return TrainingConfig(
        query_budget=_parse_query(raw["query_budget"], f"{path}.query_budget"),
        ppo=_parse_ppo(raw["ppo"], f"{path}.ppo", seeds=seeds),
    )


def _parse_feasibility(value: Any, path: str) -> FeasibilityConfig:
    raw = _mapping(value, path)
    _strict_keys(raw, _FEASIBILITY_KEYS, path)
    return FeasibilityConfig(
        clean_scenes=_integer(raw["clean_scenes"], f"{path}.clean_scenes", minimum=1),
        minimum_clean_detection_rate=_number(
            raw["minimum_clean_detection_rate"],
            f"{path}.minimum_clean_detection_rate",
            minimum=0.0,
            maximum=1.0,
            minimum_inclusive=False,
        ),
        probe_patterns=_integer(
            raw["probe_patterns"], f"{path}.probe_patterns", minimum=1
        ),
        probe_scenes=_integer(raw["probe_scenes"], f"{path}.probe_scenes", minimum=1),
        minimum_dense_progress=_number(
            raw["minimum_dense_progress"],
            f"{path}.minimum_dense_progress",
            minimum=0.0,
            minimum_inclusive=False,
        ),
        minimum_source_confidence_drop=_number(
            raw["minimum_source_confidence_drop"],
            f"{path}.minimum_source_confidence_drop",
            minimum=0.0,
            maximum=1.0,
        ),
        no_progress_query_checkpoint=_integer(
            raw["no_progress_query_checkpoint"],
            f"{path}.no_progress_query_checkpoint",
            minimum=1,
        ),
        target_event_query_checkpoint=_integer(
            raw["target_event_query_checkpoint"],
            f"{path}.target_event_query_checkpoint",
            minimum=1,
        ),
        require_target_event_for_targeted=_boolean(
            raw["require_target_event_for_targeted"],
            f"{path}.require_target_event_for_targeted",
        ),
    )


def _parse_evaluation(value: Any, path: str) -> EvaluationConfig:
    raw = _mapping(value, path)
    _strict_keys(raw, _EVALUATION_KEYS, path)
    confidence = _number(
        raw["confidence_level"],
        f"{path}.confidence_level",
        minimum=0.0,
        maximum=1.0,
        minimum_inclusive=False,
    )
    if confidence >= 1.0:
        raise ConfigError(f"{path}.confidence_level must be < 1")
    return EvaluationConfig(
        episodes=_integer(raw["episodes"], f"{path}.episodes", minimum=1),
        eot_samples=_integer(raw["eot_samples"], f"{path}.eot_samples", minimum=1),
        confidence_level=confidence,
        frozen_split=_string(raw["frozen_split"], f"{path}.frozen_split"),
        minimum_success_rate=_number(
            raw["minimum_success_rate"],
            f"{path}.minimum_success_rate",
            minimum=0.0,
            maximum=1.0,
            minimum_inclusive=False,
        ),
        minimum_confidence_interval_lower=_number(
            raw["minimum_confidence_interval_lower"],
            f"{path}.minimum_confidence_interval_lower",
            minimum=0.0,
            maximum=1.0,
        ),
        minimum_successful_pilot_seeds=_integer(
            raw["minimum_successful_pilot_seeds"],
            f"{path}.minimum_successful_pilot_seeds",
            minimum=1,
        ),
        minimum_successful_seeds_per_model=_integer(
            raw["minimum_successful_seeds_per_model"],
            f"{path}.minimum_successful_seeds_per_model",
            minimum=1,
        ),
        minimum_successful_models=_integer(
            raw["minimum_successful_models"],
            f"{path}.minimum_successful_models",
            minimum=1,
        ),
        minimum_successful_goals=_integer(
            raw["minimum_successful_goals"],
            f"{path}.minimum_successful_goals",
            minimum=1,
        ),
    )


def parse_experiment_config(
    payload: Mapping[str, Any],
    *,
    source_path: str | Path | None = None,
    source_bytes: bytes | None = None,
) -> ExperimentConfig:
    """Validate one already-decoded configuration mapping."""

    raw = _mapping(payload, "$")
    _strict_keys(raw, _ROOT_KEYS, "$")
    schema_version = _integer(raw["schema_version"], "$.schema_version", minimum=1)
    if schema_version != SCHEMA_VERSION:
        raise ConfigError(f"$.schema_version must equal {SCHEMA_VERSION}")
    stage = _string(raw["stage"], "$.stage").lower()
    if stage not in EXPERIMENT_STAGES:
        raise ConfigError(f"$.stage must be one of {list(EXPERIMENT_STAGES)}")

    assets = _parse_assets(raw["assets"], "$.assets")
    surrogate_models = tuple(
        _parse_model(value, f"$.surrogate_models[{index}]")
        for index, value in enumerate(
            _array(raw["surrogate_models"], "$.surrogate_models", nonempty=True)
        )
    )
    heldout_models = tuple(
        _parse_model(value, f"$.heldout_models[{index}]")
        for index, value in enumerate(_array(raw["heldout_models"], "$.heldout_models"))
    )
    model_ids = [model.id for model in (*surrogate_models, *heldout_models)]
    if len(model_ids) != len(set(model_ids)):
        raise ConfigError(
            "surrogate_models and heldout_models must have globally unique IDs"
        )
    if stage == "paper" and not heldout_models:
        raise ConfigError("paper configuration requires at least one heldout model")

    goals = tuple(
        _parse_goal(value, f"$.attack_goals[{index}]")
        for index, value in enumerate(_array(raw["attack_goals"], "$.attack_goals", nonempty=True))
    )
    goal_modes = [goal.mode for goal in goals]
    if len(goal_modes) != len(set(goal_modes)):
        raise ConfigError("$.attack_goals must not repeat a mode")
    enabled_goals = [goal for goal in goals if goal.enabled]
    if not enabled_goals:
        raise ConfigError("$.attack_goals must enable at least one goal")
    source_classes = {goal.source_class.casefold() for goal in enabled_goals}
    if len(source_classes) != 1:
        raise ConfigError("enabled attack goals must use one source_class")

    seeds_raw = _array(raw["seeds"], "$.seeds", nonempty=True)
    seeds = tuple(
        _integer(seed, f"$.seeds[{index}]", minimum=0)
        for index, seed in enumerate(seeds_raw)
    )
    if len(seeds) != len(set(seeds)):
        raise ConfigError("$.seeds must not contain duplicates")
    material = _parse_material(raw["patch"], "$.patch")
    training = _parse_training(raw["training"], "$.training", seeds=seeds)
    query = training.query_budget
    ppo = training.ppo
    feasibility = _parse_feasibility(raw["feasibility"], "$.feasibility")
    evaluation = _parse_evaluation(raw["evaluation"], "$.evaluation")

    search_cap = (
        MAX_PILOT_SEARCH_IMAGE_QUERIES
        if stage == "pilot"
        else MAX_PAPER_SEARCH_IMAGE_QUERIES
    )
    if query.search_image_queries > search_cap:
        raise ConfigError(
            f"{stage} search_image_queries exceeds the hard cap of {search_cap}"
        )
    if ppo.maximum_environment_steps > query.search_image_queries:
        raise ConfigError(
            "$.training.ppo.maximum_environment_steps cannot exceed the charged "
            "search-image budget"
        )
    if ppo.maximum_episode_actions != material.maximum_actions:
        raise ConfigError(
            "$.training.ppo.maximum_episode_actions must equal $.patch.maximum_actions"
        )
    if ppo.initial_seed_count > len(ppo.seeds):
        raise ConfigError("$.training.ppo.initial_seed_count cannot exceed the number of seeds")
    if not ppo.one_meta_policy_per_seed:
        raise ConfigError("$.ppo.one_meta_policy_per_seed must be true")
    if stage == "pilot" and (len(ppo.seeds) != 1 or ppo.initial_seed_count != 1):
        raise ConfigError("pilot configuration must declare exactly one initial seed")
    if stage == "paper" and len(ppo.seeds) < 3:
        raise ConfigError("paper configuration requires at least three staged seeds")
    if ppo.checkpoint_query_interval > query.search_image_queries:
        raise ConfigError("PPO checkpoint interval exceeds the search-image budget")
    if feasibility.no_progress_query_checkpoint > query.search_image_queries:
        raise ConfigError("no-progress checkpoint exceeds the search-image budget")
    if feasibility.target_event_query_checkpoint > query.search_image_queries:
        raise ConfigError("target-event checkpoint exceeds the search-image budget")
    minimum_probe_queries = feasibility.clean_scenes + (
        feasibility.probe_patterns * feasibility.probe_scenes
    )
    if minimum_probe_queries > query.feasibility_image_queries:
        raise ConfigError(
            "feasibility_image_queries cannot cover clean scenes plus all probe images"
        )
    if evaluation.minimum_successful_pilot_seeds > ppo.initial_seed_count:
        raise ConfigError(
            "minimum_successful_pilot_seeds exceeds the initial staged seed count"
        )
    if evaluation.minimum_successful_seeds_per_model > len(ppo.seeds):
        raise ConfigError(
            "minimum_successful_seeds_per_model exceeds the declared seed count"
        )
    if evaluation.minimum_successful_models > len(model_ids):
        raise ConfigError("minimum_successful_models exceeds declared model count")
    if evaluation.minimum_successful_goals > len(enabled_goals):
        raise ConfigError("minimum_successful_goals exceeds enabled goal count")

    canonical_payload = {
        "schema_version": schema_version,
        "experiment_id": _identifier(raw["experiment_id"], "$.experiment_id"),
        "stage": stage,
        "description": _string(raw["description"], "$.description"),
        "assets": assets,
        "surrogate_models": surrogate_models,
        "heldout_models": heldout_models,
        "attack_goals": goals,
        "patch": material,
        "training": training,
        "feasibility": feasibility,
        "evaluation": evaluation,
        "seeds": seeds,
    }
    canonical_json_payload = {
        "schema_version": schema_version,
        "experiment_id": canonical_payload["experiment_id"],
        "stage": stage,
        "description": canonical_payload["description"],
        "assets": assets.to_dict(),
        "surrogate_models": [model.to_dict() for model in surrogate_models],
        "heldout_models": [model.to_dict() for model in heldout_models],
        "attack_goals": [goal.to_dict() for goal in goals],
        "patch": material.to_dict(),
        "training": training.to_dict(),
        "feasibility": feasibility.to_dict(),
        "evaluation": evaluation.to_dict(),
        "seeds": list(seeds),
    }
    canonical = _canonical_json(canonical_json_payload)
    raw_source = source_bytes if source_bytes is not None else canonical
    return ExperimentConfig(
        **canonical_payload,
        source_path=(str(Path(source_path).resolve()) if source_path is not None else None),
        source_sha256=hashlib.sha256(raw_source).hexdigest(),
        canonical_sha256=hashlib.sha256(canonical).hexdigest(),
    )


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    """Load strict JSON without accepting duplicates, NaN, or infinity."""

    source = Path(path).resolve()
    try:
        data = source.read_bytes()
        payload = json.loads(
            data.decode("utf-8"),
            parse_constant=_reject_constant,
            object_pairs_hook=_unique_object,
        )
    except ConfigError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ConfigError(f"cannot read experiment configuration {source}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ConfigError("configuration root must be an object")
    return parse_experiment_config(payload, source_path=source, source_bytes=data)


__all__ = [
    "ATTACK_GOALS",
    "AttackGoalConfig",
    "AssetsConfig",
    "ConfigError",
    "DetectorModelConfig",
    "EvaluationConfig",
    "ExperimentConfig",
    "FeasibilityConfig",
    "MaterialBudgetConfig",
    "PPOConfig",
    "QueryBudgetConfig",
    "TrainingConfig",
    "load_experiment_config",
    "parse_experiment_config",
]
