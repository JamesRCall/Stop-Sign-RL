"""Deterministic fail-fast gates for staged PatchRL experiments.

Gate inputs are validated records rather than ad-hoc dictionaries.  Missing,
duplicate, over-budget, non-finite, or incomplete evidence never advances an
experiment.  The gates do not inspect trainer internals and therefore remain
usable by local runners, CI, and paper artifact tooling.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable, Mapping, Sequence

from patchrl.config import ATTACK_GOALS, ExperimentConfig


RECORD_STATUSES = ("completed", "aborted", "failed")


class GateRecordError(ValueError):
    """Raised when a gate record is malformed or internally inconsistent."""


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise GateRecordError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise GateRecordError(f"{name} must be finite")
    return result


def _rate(value: Any, name: str) -> float:
    result = _finite(value, name)
    if not 0.0 <= result <= 1.0:
        raise GateRecordError(f"{name} must be in [0, 1]")
    return result


def _integer(value: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise GateRecordError(f"{name} must be an integer")
    if value < minimum:
        raise GateRecordError(f"{name} must be >= {minimum}")
    return int(value)


def _string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise GateRecordError(f"{name} must be a non-empty string")
    return value.strip()


def _strict_record(
    value: Mapping[str, Any], expected: Iterable[str], name: str
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise GateRecordError(f"{name} must be an object")
    expected_set = set(expected)
    missing = sorted(expected_set - set(value))
    unknown = sorted(set(value) - expected_set)
    if missing or unknown:
        raise GateRecordError(
            f"{name} invalid fields: missing={missing}, unknown={unknown}"
        )
    return value


def _goal(value: Any, name: str = "goal") -> str:
    result = _string(value, name)
    if result not in ATTACK_GOALS:
        raise GateRecordError(f"{name} must be one of {list(ATTACK_GOALS)}")
    return result


@dataclass(frozen=True)
class FeasibilityRecord:
    model_id: str
    goal: str
    detector_image_queries: int
    clean_detection_rate: float
    maximum_dense_progress: float
    maximum_source_confidence_drop: float
    target_events: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "model_id", _string(self.model_id, "model_id"))
        object.__setattr__(self, "goal", _goal(self.goal))
        object.__setattr__(
            self,
            "detector_image_queries",
            _integer(self.detector_image_queries, "detector_image_queries", minimum=1),
        )
        object.__setattr__(
            self,
            "clean_detection_rate",
            _rate(self.clean_detection_rate, "clean_detection_rate"),
        )
        object.__setattr__(
            self,
            "maximum_dense_progress",
            _finite(self.maximum_dense_progress, "maximum_dense_progress"),
        )
        object.__setattr__(
            self,
            "maximum_source_confidence_drop",
            _finite(
                self.maximum_source_confidence_drop,
                "maximum_source_confidence_drop",
            ),
        )
        object.__setattr__(
            self, "target_events", _integer(self.target_events, "target_events")
        )

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "FeasibilityRecord":
        raw = _strict_record(
            value,
            {
                "model_id",
                "goal",
                "detector_image_queries",
                "clean_detection_rate",
                "maximum_dense_progress",
                "maximum_source_confidence_drop",
                "target_events",
            },
            "feasibility record",
        )
        return cls(**raw)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "goal": self.goal,
            "detector_image_queries": self.detector_image_queries,
            "clean_detection_rate": self.clean_detection_rate,
            "maximum_dense_progress": self.maximum_dense_progress,
            "maximum_source_confidence_drop": self.maximum_source_confidence_drop,
            "target_events": self.target_events,
        }


@dataclass(frozen=True)
class EvaluationRecord:
    model_id: str
    goal: str
    seed: int
    status: str
    feasibility_passed: bool
    search_image_queries: int
    evaluation_image_queries: int
    maximum_dense_progress: float
    maximum_source_confidence_drop: float
    target_events: int
    successes: int
    episodes: int
    confidence_interval_lower: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "model_id", _string(self.model_id, "model_id"))
        object.__setattr__(self, "goal", _goal(self.goal))
        object.__setattr__(self, "seed", _integer(self.seed, "seed"))
        status = _string(self.status, "status").lower()
        if status not in RECORD_STATUSES:
            raise GateRecordError(f"status must be one of {list(RECORD_STATUSES)}")
        object.__setattr__(self, "status", status)
        if not isinstance(self.feasibility_passed, bool):
            raise GateRecordError("feasibility_passed must be a boolean")
        object.__setattr__(
            self,
            "search_image_queries",
            _integer(self.search_image_queries, "search_image_queries"),
        )
        object.__setattr__(
            self,
            "evaluation_image_queries",
            _integer(self.evaluation_image_queries, "evaluation_image_queries"),
        )
        object.__setattr__(
            self,
            "maximum_dense_progress",
            _finite(self.maximum_dense_progress, "maximum_dense_progress"),
        )
        object.__setattr__(
            self,
            "maximum_source_confidence_drop",
            _finite(
                self.maximum_source_confidence_drop,
                "maximum_source_confidence_drop",
            ),
        )
        object.__setattr__(
            self, "target_events", _integer(self.target_events, "target_events")
        )
        successes = _integer(self.successes, "successes")
        episodes = _integer(self.episodes, "episodes")
        if successes > episodes:
            raise GateRecordError("successes cannot exceed episodes")
        if status == "completed" and episodes < 1:
            raise GateRecordError("a completed record requires at least one episode")
        object.__setattr__(self, "successes", successes)
        object.__setattr__(self, "episodes", episodes)
        lower = _rate(self.confidence_interval_lower, "confidence_interval_lower")
        if episodes and lower > (successes / episodes) + 1e-12:
            raise GateRecordError("confidence_interval_lower cannot exceed success_rate")
        object.__setattr__(self, "confidence_interval_lower", lower)

    @property
    def success_rate(self) -> float:
        return float(self.successes / self.episodes) if self.episodes else 0.0

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "EvaluationRecord":
        raw = _strict_record(
            value,
            {
                "model_id",
                "goal",
                "seed",
                "status",
                "feasibility_passed",
                "search_image_queries",
                "evaluation_image_queries",
                "maximum_dense_progress",
                "maximum_source_confidence_drop",
                "target_events",
                "successes",
                "episodes",
                "confidence_interval_lower",
            },
            "evaluation record",
        )
        return cls(**raw)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "goal": self.goal,
            "seed": self.seed,
            "status": self.status,
            "feasibility_passed": self.feasibility_passed,
            "search_image_queries": self.search_image_queries,
            "evaluation_image_queries": self.evaluation_image_queries,
            "maximum_dense_progress": self.maximum_dense_progress,
            "maximum_source_confidence_drop": self.maximum_source_confidence_drop,
            "target_events": self.target_events,
            "successes": self.successes,
            "episodes": self.episodes,
            "success_rate": self.success_rate,
            "confidence_interval_lower": self.confidence_interval_lower,
        }


@dataclass(frozen=True)
class GateDecision:
    stage: str
    advance: bool
    reasons: tuple[str, ...]
    evidence: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "advance": self.advance,
            "reasons": list(self.reasons),
            "evidence": dict(self.evidence),
        }


def _known_model_ids(config: ExperimentConfig) -> set[str]:
    return set(config.model_ids)


def _feasibility_failures(
    record: FeasibilityRecord,
    config: ExperimentConfig,
) -> list[str]:
    failures: list[str] = []
    if record.model_id not in {model.id for model in config.surrogate_models}:
        failures.append("model_not_declared_as_surrogate")
    try:
        goal = config.goal(record.goal, require_enabled=True)
    except ValueError:
        failures.append("goal_not_enabled")
        return failures
    thresholds = config.feasibility
    if record.detector_image_queries > config.training.query_budget.feasibility_image_queries:
        failures.append("feasibility_query_budget_exceeded")
    if record.clean_detection_rate < thresholds.minimum_clean_detection_rate:
        failures.append("clean_detection_below_threshold")
    # Zero must never pass even if a configuration threshold is accidentally
    # weakened in a later schema revision.
    if record.maximum_dense_progress <= 0.0:
        failures.append("zero_dense_progress")
    elif record.maximum_dense_progress < thresholds.minimum_dense_progress:
        failures.append("dense_progress_below_threshold")
    if (
        record.goal == "disappearance" or goal.require_source_suppression
    ) and (
        record.maximum_source_confidence_drop
        < thresholds.minimum_source_confidence_drop
    ):
        failures.append("source_confidence_drop_below_threshold")
    if (
        record.goal == "targeted_misclassification"
        and thresholds.require_target_event_for_targeted
        and record.target_events <= 0
    ):
        failures.append("target_event_absent")
    return failures


def evaluate_feasibility(
    record: FeasibilityRecord,
    config: ExperimentConfig,
) -> GateDecision:
    """Evaluate one model/goal probe record without favorable imputation."""

    failures = _feasibility_failures(record, config)
    return GateDecision(
        stage="feasibility",
        advance=not failures,
        reasons=tuple(failures),
        evidence=record.to_dict(),
    )


def decide_pilot_advancement(
    records: Sequence[FeasibilityRecord],
    config: ExperimentConfig,
) -> GateDecision:
    """Decide whether feasibility evidence permits any pilot PPO training.

    A goal is eligible only when every declared surrogate model has exactly one
    passing record for that goal.  Experiments with alternative goals may
    advance when the configured minimum number of complete goals is eligible.
    """

    by_key: dict[tuple[str, str], list[FeasibilityRecord]] = {}
    for record in records:
        by_key.setdefault((record.model_id, record.goal), []).append(record)
    eligible_goals: list[str] = []
    rejected: dict[str, list[str]] = {}
    for goal in config.enabled_goals:
        failures: list[str] = []
        for model in config.surrogate_models:
            matches = by_key.get((model.id, goal.mode), [])
            if len(matches) != 1:
                failures.append(
                    f"{model.id}:"
                    + ("missing_record" if not matches else "duplicate_records")
                )
                continue
            failures.extend(
                f"{model.id}:{reason}"
                for reason in _feasibility_failures(matches[0], config)
            )
        if failures:
            rejected[goal.mode] = failures
        else:
            eligible_goals.append(goal.mode)
    required = config.evaluation.minimum_successful_goals
    advance = len(eligible_goals) >= required
    reasons = () if advance else ("insufficient_feasible_goals",)
    return GateDecision(
        stage="pilot",
        advance=advance,
        reasons=reasons,
        evidence={
            "eligible_goals": eligible_goals,
            "rejected_goals": rejected,
            "required_successful_goals": required,
        },
    )


def _evaluation_failures(
    record: EvaluationRecord,
    config: ExperimentConfig,
) -> list[str]:
    failures: list[str] = []
    if record.model_id not in _known_model_ids(config):
        failures.append("unknown_model")
    try:
        goal = config.goal(record.goal, require_enabled=True)
    except ValueError:
        failures.append("goal_not_enabled")
        return failures
    if record.seed not in config.seeds:
        failures.append("undeclared_seed")
    if record.status != "completed":
        failures.append("run_not_completed")
    if not record.feasibility_passed:
        failures.append("feasibility_not_passed")
    if record.search_image_queries <= 0:
        failures.append("zero_search_queries")
    elif record.search_image_queries > config.training.query_budget.search_image_queries:
        failures.append("search_query_budget_exceeded")
    if record.evaluation_image_queries <= 0:
        failures.append("zero_evaluation_queries")
    elif (
        record.evaluation_image_queries
        > config.training.query_budget.evaluation_image_queries
    ):
        failures.append("evaluation_query_budget_exceeded")
    if record.maximum_dense_progress <= 0.0:
        failures.append("zero_dense_progress")
    elif record.maximum_dense_progress < config.feasibility.minimum_dense_progress:
        failures.append("dense_progress_below_threshold")
    if (
        record.goal == "disappearance" or goal.require_source_suppression
    ) and (
        record.maximum_source_confidence_drop
        < config.feasibility.minimum_source_confidence_drop
    ):
        failures.append("source_confidence_drop_below_threshold")
    if (
        record.goal == "targeted_misclassification"
        and config.feasibility.require_target_event_for_targeted
        and record.target_events <= 0
    ):
        failures.append("target_event_absent")
    if record.success_rate < config.evaluation.minimum_success_rate:
        failures.append("success_rate_below_threshold")
    if (
        record.confidence_interval_lower
        < config.evaluation.minimum_confidence_interval_lower
    ):
        failures.append("confidence_interval_below_threshold")
    return failures


def evaluate_result(
    record: EvaluationRecord,
    config: ExperimentConfig,
) -> GateDecision:
    """Return whether one frozen model/goal/seed result is advancement-ready."""

    failures = _evaluation_failures(record, config)
    return GateDecision(
        stage="result",
        advance=not failures,
        reasons=tuple(failures),
        evidence=record.to_dict(),
    )


def decide_seed_advancement(
    records: Sequence[EvaluationRecord],
    config: ExperimentConfig,
    *,
    model_id: str,
    goal: str,
) -> GateDecision:
    """Decide whether initial seed evidence warrants running expansion seeds."""

    requested_goal = config.goal(goal, require_enabled=True)
    del requested_goal
    initial_seeds = config.training.ppo.initial_seeds
    matches: dict[int, list[EvaluationRecord]] = {seed: [] for seed in initial_seeds}
    for record in records:
        if record.model_id == model_id and record.goal == goal and record.seed in matches:
            matches[record.seed].append(record)
    successful: list[int] = []
    rejected: dict[str, list[str]] = {}
    for seed, seed_records in matches.items():
        if len(seed_records) != 1:
            rejected[str(seed)] = [
                "missing_record" if not seed_records else "duplicate_records"
            ]
            continue
        failures = _evaluation_failures(seed_records[0], config)
        if failures:
            rejected[str(seed)] = failures
        else:
            successful.append(seed)
    required = config.evaluation.minimum_successful_pilot_seeds
    advance = len(successful) >= required
    return GateDecision(
        stage="seed_expansion",
        advance=advance,
        reasons=() if advance else ("insufficient_successful_pilot_seeds",),
        evidence={
            "model_id": model_id,
            "goal": goal,
            "successful_initial_seeds": successful,
            "rejected_initial_seeds": rejected,
            "required_successful_pilot_seeds": required,
            "expansion_seeds": list(config.training.ppo.expansion_seeds) if advance else [],
        },
    )


def decide_model_advancement(
    records: Sequence[EvaluationRecord],
    config: ExperimentConfig,
    *,
    goal: str,
    model_ids: Sequence[str] | None = None,
) -> GateDecision:
    """Decide whether replicated evidence warrants expanding model coverage."""

    config.goal(goal, require_enabled=True)
    candidates = tuple(
        model_ids
        if model_ids is not None
        else (model.id for model in config.surrogate_models)
    )
    if len(candidates) != len(set(candidates)):
        return GateDecision(
            stage="model_expansion",
            advance=False,
            reasons=("duplicate_candidate_models",),
            evidence={"goal": goal, "candidate_models": list(candidates)},
        )
    unknown = sorted(set(candidates) - _known_model_ids(config))
    if unknown:
        return GateDecision(
            stage="model_expansion",
            advance=False,
            reasons=("unknown_candidate_models",),
            evidence={"goal": goal, "unknown_models": unknown},
        )

    qualified: list[str] = []
    rejected: dict[str, Any] = {}
    required_seeds = config.evaluation.minimum_successful_seeds_per_model
    for model_id in candidates:
        by_seed: dict[int, list[EvaluationRecord]] = {}
        for record in records:
            if record.model_id == model_id and record.goal == goal:
                by_seed.setdefault(record.seed, []).append(record)
        passing_seeds: list[int] = []
        seed_failures: dict[str, list[str]] = {}
        for seed in config.seeds:
            seed_records = by_seed.get(seed, [])
            if len(seed_records) != 1:
                seed_failures[str(seed)] = [
                    "missing_record" if not seed_records else "duplicate_records"
                ]
                continue
            failures = _evaluation_failures(seed_records[0], config)
            if failures:
                seed_failures[str(seed)] = failures
            else:
                passing_seeds.append(seed)
        if len(passing_seeds) >= required_seeds:
            qualified.append(model_id)
        else:
            rejected[model_id] = {
                "passing_seeds": passing_seeds,
                "seed_failures": seed_failures,
            }
    required_models = config.evaluation.minimum_successful_models
    advance = len(qualified) >= required_models
    return GateDecision(
        stage="model_expansion",
        advance=advance,
        reasons=() if advance else ("insufficient_successful_models",),
        evidence={
            "goal": goal,
            "qualified_models": qualified,
            "rejected_models": rejected,
            "required_successful_models": required_models,
            "required_successful_seeds_per_model": required_seeds,
        },
    )


__all__ = [
    "EvaluationRecord",
    "FeasibilityRecord",
    "GateDecision",
    "GateRecordError",
    "decide_model_advancement",
    "decide_pilot_advancement",
    "decide_seed_advancement",
    "evaluate_feasibility",
    "evaluate_result",
]
