"""Preregistered two-phase risk certification for frozen patch prefixes.

This module implements a deliberately narrow claim:

* a finite family of prefix candidates is frozen before calibration;
* every candidate is evaluated on the same preregistered calibration trials;
* the smallest preregistered prefix order passing every task constraint is
  selected without assuming monotonic attack success;
* only that selected prefix is evaluated on a separate, untouched
  certification trial set; and
* one-sided exact binomial bounds receive an equal Bonferroni allocation over
  every task, stochastic claim, and (during calibration) candidate prefix.

The output is finite-sample sampled-population risk certification under the
sampling assumptions written in the protocol.  It is not formal verification,
an adversarial robustness certificate, or an all-world guarantee.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
import math
import re
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from utils.certification_stats import (
    binomial_upper_tail,
    bonferroni_local_alpha,
    clopper_pearson_lower,
)


SCHEMA_VERSION = 1
CLAIM_NAMES = (
    "joint_success",
    "attack_success",
    "day_preservation",
    "clean_eligibility",
)
PHASES = ("calibration", "certification")
SCOPE_LABEL = (
    "finite-sample sampled-population risk certification under the "
    "preregistered Bernoulli sampling assumptions; not formal verification "
    "and not an all-world guarantee"
)
SELECTION_RULE = (
    "minimum preregistered prefix order among candidates passing every task, "
    "stochastic claim, and exact area constraint; every candidate is evaluated "
    "and attack success monotonicity is not assumed"
)

_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class FractionSpec:
    numerator: int
    denominator: int

    @property
    def fraction(self) -> Fraction:
        return Fraction(self.numerator, self.denominator)

    @property
    def decimal(self) -> float:
        return float(self.fraction)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "numerator": int(self.numerator),
            "denominator": int(self.denominator),
            "decimal": self.decimal,
        }


@dataclass(frozen=True)
class TrialSpec:
    trial_id: str
    sample_sha256: str


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    detector_sha256: str
    calibration_trials: Tuple[TrialSpec, ...]
    certification_trials: Tuple[TrialSpec, ...]


@dataclass(frozen=True)
class AreaSpec:
    selected_pixels: int
    sign_pixels: int

    @property
    def fraction(self) -> Fraction:
        return Fraction(self.selected_pixels, self.sign_pixels)

    def as_dict(self) -> Dict[str, Any]:
        reduced = self.fraction
        return {
            "selected_pixels": int(self.selected_pixels),
            "sign_pixels": int(self.sign_pixels),
            "fraction": {
                "numerator": int(reduced.numerator),
                "denominator": int(reduced.denominator),
                "decimal": float(reduced),
            },
        }


@dataclass(frozen=True)
class TaskPatternSpec:
    task_id: str
    pattern_sha256: str
    area: AreaSpec


@dataclass(frozen=True)
class PrefixSpec:
    prefix_id: str
    order: int
    task_patterns: Tuple[TaskPatternSpec, ...]

    def task_pattern_map(self) -> Dict[str, TaskPatternSpec]:
        return {item.task_id: item for item in self.task_patterns}


@dataclass(frozen=True)
class EstimandSpec:
    population_description: str
    sampling_unit: str
    task_aggregation: str
    fixed_sample_size: bool


@dataclass(frozen=True)
class QueryAccountingSpec:
    offline_training_detector_image_queries: int
    development_selection_detector_image_queries: int
    prefix_generation_detector_image_queries: int
    expected_calibration_detector_image_queries: int
    expected_certification_detector_image_queries: int


@dataclass(frozen=True)
class ProtocolSpec:
    protocol_id: str
    prefix_family_sha256: str
    estimand: EstimandSpec
    calibration_alpha: FractionSpec
    certification_alpha: FractionSpec
    claims: Mapping[str, FractionSpec]
    area_cap: FractionSpec
    tasks: Tuple[TaskSpec, ...]
    prefixes: Tuple[PrefixSpec, ...]
    excluded_training_sha256: Tuple[str, ...]
    excluded_development_sha256: Tuple[str, ...]
    query_accounting: QueryAccountingSpec
    canonical_sha256: str

    def task_map(self) -> Dict[str, TaskSpec]:
        return {task.task_id: task for task in self.tasks}

    def prefix_map(self) -> Dict[str, PrefixSpec]:
        return {prefix.prefix_id: prefix for prefix in self.prefixes}


@dataclass(frozen=True)
class ResultRow:
    task_id: str
    prefix_id: str
    trial_id: str
    sample_sha256: str
    pattern_sha256: str
    detector_sha256: str
    outcomes: Mapping[str, bool]
    area: AreaSpec
    detector_image_queries: int


def canonical_json_bytes(value: Any) -> bytes:
    """Return deterministic UTF-8 JSON bytes, rejecting NaN and infinities."""
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def report_sha256(report: Mapping[str, Any]) -> str:
    """Hash a report while excluding its self-referential digest field."""
    payload = dict(report)
    payload.pop("report_sha256", None)
    return canonical_sha256(payload)


def _expect_mapping(value: Any, *, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be a JSON object")
    return value


def _expect_list(value: Any, *, context: str) -> List[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{context} must be a JSON list")
    return value


def _expect_exact_keys(
    value: Mapping[str, Any],
    expected: Iterable[str],
    *,
    context: str,
) -> None:
    required = set(expected)
    present = set(str(key) for key in value.keys())
    missing = sorted(required - present)
    unknown = sorted(present - required)
    if missing or unknown:
        pieces = []
        if missing:
            pieces.append("missing=" + ",".join(missing))
        if unknown:
            pieces.append("unknown=" + ",".join(unknown))
        raise ValueError(f"{context} has invalid fields ({'; '.join(pieces)})")


def _strict_int(value: Any, *, context: str, minimum: Optional[int] = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{context} must be an integer")
    number = int(value)
    if minimum is not None and number < minimum:
        raise ValueError(f"{context} must be >= {minimum}")
    return number


def _strict_bool(value: Any, *, context: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{context} must be a JSON boolean")
    return bool(value)


def _identifier(value: Any, *, context: str) -> str:
    if not isinstance(value, str) or not _IDENTIFIER_RE.fullmatch(value):
        raise ValueError(
            f"{context} must match {_IDENTIFIER_RE.pattern!r}"
        )
    return value


def _sha256(value: Any, *, context: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise ValueError(f"{context} must be a lowercase 64-character SHA-256")
    return value


def _fraction_spec(
    value: Any,
    *,
    context: str,
    allow_one: bool,
) -> FractionSpec:
    obj = _expect_mapping(value, context=context)
    _expect_exact_keys(obj, ("numerator", "denominator"), context=context)
    numerator = _strict_int(obj["numerator"], context=f"{context}.numerator", minimum=1)
    denominator = _strict_int(
        obj["denominator"], context=f"{context}.denominator", minimum=1
    )
    if math.gcd(numerator, denominator) != 1:
        raise ValueError(f"{context} must be in reduced rational form")
    if numerator > denominator or (numerator == denominator and not allow_one):
        interval = "(0, 1]" if allow_one else "(0, 1)"
        raise ValueError(f"{context} must be in {interval}")
    return FractionSpec(numerator=numerator, denominator=denominator)


def _area_spec(value: Any, *, context: str) -> AreaSpec:
    obj = _expect_mapping(value, context=context)
    _expect_exact_keys(obj, ("selected_pixels", "sign_pixels"), context=context)
    selected = _strict_int(
        obj["selected_pixels"], context=f"{context}.selected_pixels", minimum=0
    )
    sign = _strict_int(obj["sign_pixels"], context=f"{context}.sign_pixels", minimum=1)
    if selected > sign:
        raise ValueError(f"{context}.selected_pixels must not exceed sign_pixels")
    return AreaSpec(selected_pixels=selected, sign_pixels=sign)


def _trial_specs(value: Any, *, context: str) -> Tuple[TrialSpec, ...]:
    rows = _expect_list(value, context=context)
    if not rows:
        raise ValueError(f"{context} must contain at least one trial")
    out: List[TrialSpec] = []
    seen_ids = set()
    seen_samples = set()
    for index, raw in enumerate(rows):
        row_context = f"{context}[{index}]"
        obj = _expect_mapping(raw, context=row_context)
        _expect_exact_keys(obj, ("trial_id", "sample_sha256"), context=row_context)
        trial_id = _identifier(obj["trial_id"], context=f"{row_context}.trial_id")
        sample = _sha256(obj["sample_sha256"], context=f"{row_context}.sample_sha256")
        if trial_id in seen_ids:
            raise ValueError(f"{context} contains duplicate trial_id {trial_id!r}")
        if sample in seen_samples:
            raise ValueError(f"{context} contains duplicate sample_sha256 {sample}")
        seen_ids.add(trial_id)
        seen_samples.add(sample)
        out.append(TrialSpec(trial_id=trial_id, sample_sha256=sample))
    return tuple(out)


def parse_protocol(payload: Any) -> ProtocolSpec:
    """Validate a protocol with unknown-field rejection and overlap checks."""
    root = _expect_mapping(payload, context="protocol")
    _expect_exact_keys(
        root,
        (
            "schema_version",
            "protocol_id",
            "prefix_family_sha256",
            "estimand",
            "familywise_alpha",
            "claims",
            "area_cap",
            "tasks",
            "prefixes",
            "excluded_sample_sha256",
            "query_accounting",
        ),
        context="protocol",
    )
    version = _strict_int(root["schema_version"], context="protocol.schema_version")
    if version != SCHEMA_VERSION:
        raise ValueError(
            f"protocol.schema_version must be {SCHEMA_VERSION}, got {version}"
        )
    protocol_id = _identifier(root["protocol_id"], context="protocol.protocol_id")
    prefix_family_sha256 = _sha256(
        root["prefix_family_sha256"],
        context="protocol.prefix_family_sha256",
    )

    estimand_obj = _expect_mapping(root["estimand"], context="protocol.estimand")
    _expect_exact_keys(
        estimand_obj,
        (
            "population_description",
            "sampling_unit",
            "task_aggregation",
            "fixed_sample_size",
        ),
        context="protocol.estimand",
    )
    population = estimand_obj["population_description"]
    if not isinstance(population, str) or not population.strip():
        raise ValueError("protocol.estimand.population_description must be non-empty")
    sampling_unit = estimand_obj["sampling_unit"]
    if sampling_unit not in (
        "independent_scene_transform_pair",
        "independent_scene_cluster",
    ):
        raise ValueError(
            "protocol.estimand.sampling_unit must be "
            "'independent_scene_transform_pair' or 'independent_scene_cluster'"
        )
    if estimand_obj["task_aggregation"] != "all_tasks":
        raise ValueError("protocol.estimand.task_aggregation must be 'all_tasks'")
    if _strict_bool(
        estimand_obj["fixed_sample_size"],
        context="protocol.estimand.fixed_sample_size",
    ) is not True:
        raise ValueError("protocol.estimand.fixed_sample_size must be true")
    estimand = EstimandSpec(
        population_description=population.strip(),
        sampling_unit=str(sampling_unit),
        task_aggregation="all_tasks",
        fixed_sample_size=True,
    )

    alpha_obj = _expect_mapping(
        root["familywise_alpha"], context="protocol.familywise_alpha"
    )
    _expect_exact_keys(
        alpha_obj, PHASES, context="protocol.familywise_alpha"
    )
    calibration_alpha = _fraction_spec(
        alpha_obj["calibration"],
        context="protocol.familywise_alpha.calibration",
        allow_one=False,
    )
    certification_alpha = _fraction_spec(
        alpha_obj["certification"],
        context="protocol.familywise_alpha.certification",
        allow_one=False,
    )

    claims_obj = _expect_mapping(root["claims"], context="protocol.claims")
    _expect_exact_keys(claims_obj, CLAIM_NAMES, context="protocol.claims")
    claims = {
        name: _fraction_spec(
            claims_obj[name], context=f"protocol.claims.{name}", allow_one=False
        )
        for name in CLAIM_NAMES
    }
    area_cap = _fraction_spec(
        root["area_cap"], context="protocol.area_cap", allow_one=True
    )

    task_rows = _expect_list(root["tasks"], context="protocol.tasks")
    if not task_rows:
        raise ValueError("protocol.tasks must contain at least one task")
    tasks: List[TaskSpec] = []
    task_ids = set()
    calibration_samples = set()
    certification_samples = set()
    for index, raw in enumerate(task_rows):
        context = f"protocol.tasks[{index}]"
        obj = _expect_mapping(raw, context=context)
        _expect_exact_keys(
            obj,
            (
                "task_id",
                "detector_sha256",
                "calibration_trials",
                "certification_trials",
            ),
            context=context,
        )
        task_id = _identifier(obj["task_id"], context=f"{context}.task_id")
        if task_id in task_ids:
            raise ValueError(f"protocol.tasks contains duplicate task_id {task_id!r}")
        task_ids.add(task_id)
        calibration_trials = _trial_specs(
            obj["calibration_trials"], context=f"{context}.calibration_trials"
        )
        certification_trials = _trial_specs(
            obj["certification_trials"], context=f"{context}.certification_trials"
        )
        for trial in calibration_trials:
            if trial.sample_sha256 in calibration_samples:
                raise ValueError(
                    "calibration sample_sha256 values must be globally unique: "
                    + trial.sample_sha256
                )
            calibration_samples.add(trial.sample_sha256)
        for trial in certification_trials:
            if trial.sample_sha256 in certification_samples:
                raise ValueError(
                    "certification sample_sha256 values must be globally unique: "
                    + trial.sample_sha256
                )
            certification_samples.add(trial.sample_sha256)
        tasks.append(
            TaskSpec(
                task_id=task_id,
                detector_sha256=_sha256(
                    obj["detector_sha256"], context=f"{context}.detector_sha256"
                ),
                calibration_trials=calibration_trials,
                certification_trials=certification_trials,
            )
        )

    overlap = calibration_samples & certification_samples
    if overlap:
        raise ValueError(
            "calibration and certification samples overlap: " + ",".join(sorted(overlap))
        )

    prefix_rows = _expect_list(root["prefixes"], context="protocol.prefixes")
    if not prefix_rows:
        raise ValueError("protocol.prefixes must contain at least one frozen prefix")
    prefixes: List[PrefixSpec] = []
    prefix_ids = set()
    prefix_orders = set()
    expected_task_ids = set(task_ids)
    for index, raw in enumerate(prefix_rows):
        context = f"protocol.prefixes[{index}]"
        obj = _expect_mapping(raw, context=context)
        _expect_exact_keys(
            obj, ("prefix_id", "order", "task_patterns"), context=context
        )
        prefix_id = _identifier(obj["prefix_id"], context=f"{context}.prefix_id")
        order = _strict_int(obj["order"], context=f"{context}.order", minimum=0)
        if prefix_id in prefix_ids:
            raise ValueError(f"protocol.prefixes contains duplicate prefix_id {prefix_id!r}")
        if order in prefix_orders:
            raise ValueError(f"protocol.prefixes contains duplicate order {order}")
        prefix_ids.add(prefix_id)
        prefix_orders.add(order)

        pattern_rows = _expect_list(
            obj["task_patterns"], context=f"{context}.task_patterns"
        )
        patterns: List[TaskPatternSpec] = []
        pattern_task_ids = set()
        for pattern_index, pattern_raw in enumerate(pattern_rows):
            pattern_context = f"{context}.task_patterns[{pattern_index}]"
            pattern_obj = _expect_mapping(pattern_raw, context=pattern_context)
            _expect_exact_keys(
                pattern_obj,
                ("task_id", "pattern_sha256", "area"),
                context=pattern_context,
            )
            pattern_task_id = _identifier(
                pattern_obj["task_id"], context=f"{pattern_context}.task_id"
            )
            if pattern_task_id in pattern_task_ids:
                raise ValueError(
                    f"{context}.task_patterns contains duplicate task_id "
                    f"{pattern_task_id!r}"
                )
            pattern_task_ids.add(pattern_task_id)
            patterns.append(
                TaskPatternSpec(
                    task_id=pattern_task_id,
                    pattern_sha256=_sha256(
                        pattern_obj["pattern_sha256"],
                        context=f"{pattern_context}.pattern_sha256",
                    ),
                    area=_area_spec(
                        pattern_obj["area"], context=f"{pattern_context}.area"
                    ),
                )
            )
        if pattern_task_ids != expected_task_ids:
            missing = sorted(expected_task_ids - pattern_task_ids)
            extra = sorted(pattern_task_ids - expected_task_ids)
            raise ValueError(
                f"{context}.task_patterns must contain every task exactly once "
                f"(missing={missing}, extra={extra})"
            )
        prefixes.append(
            PrefixSpec(prefix_id=prefix_id, order=order, task_patterns=tuple(patterns))
        )

    # A physical prefix may add material but cannot remove it.  This validates
    # the material interpretation only; no analogous monotonicity is assumed or
    # used for attack success.
    ordered_prefixes = sorted(prefixes, key=lambda item: item.order)
    for task_id in sorted(task_ids):
        prior: Optional[AreaSpec] = None
        for prefix in ordered_prefixes:
            current = prefix.task_pattern_map()[task_id].area
            if prior is not None:
                if current.sign_pixels != prior.sign_pixels:
                    raise ValueError(
                        f"sign_pixels changed across prefixes for task {task_id!r}"
                    )
                if current.fraction < prior.fraction:
                    raise ValueError(
                        f"material area decreases with prefix order for task {task_id!r}"
                    )
            prior = current

    excluded_obj = _expect_mapping(
        root["excluded_sample_sha256"], context="protocol.excluded_sample_sha256"
    )
    _expect_exact_keys(
        excluded_obj,
        ("training", "development"),
        context="protocol.excluded_sample_sha256",
    )

    def parse_hash_list(value: Any, *, context: str) -> Tuple[str, ...]:
        raw_values = _expect_list(value, context=context)
        parsed = tuple(
            _sha256(item, context=f"{context}[{index}]")
            for index, item in enumerate(raw_values)
        )
        if len(set(parsed)) != len(parsed):
            raise ValueError(f"{context} contains duplicate hashes")
        return parsed

    training_hashes = parse_hash_list(
        excluded_obj["training"], context="protocol.excluded_sample_sha256.training"
    )
    development_hashes = parse_hash_list(
        excluded_obj["development"],
        context="protocol.excluded_sample_sha256.development",
    )
    split_sets = {
        "training": set(training_hashes),
        "development": set(development_hashes),
        "calibration": calibration_samples,
        "certification": certification_samples,
    }
    split_names = list(split_sets)
    for left_index, left_name in enumerate(split_names):
        for right_name in split_names[left_index + 1 :]:
            shared = split_sets[left_name] & split_sets[right_name]
            if shared:
                raise ValueError(
                    f"{left_name} and {right_name} sample hashes overlap: "
                    + ",".join(sorted(shared))
                )

    query_obj = _expect_mapping(
        root["query_accounting"], context="protocol.query_accounting"
    )
    query_fields = (
        "offline_training_detector_image_queries",
        "development_selection_detector_image_queries",
        "prefix_generation_detector_image_queries",
        "expected_calibration_detector_image_queries",
        "expected_certification_detector_image_queries",
    )
    _expect_exact_keys(query_obj, query_fields, context="protocol.query_accounting")
    query_values = {
        field: _strict_int(
            query_obj[field], context=f"protocol.query_accounting.{field}", minimum=0
        )
        for field in query_fields
    }
    query_accounting = QueryAccountingSpec(**query_values)

    return ProtocolSpec(
        protocol_id=protocol_id,
        prefix_family_sha256=prefix_family_sha256,
        estimand=estimand,
        calibration_alpha=calibration_alpha,
        certification_alpha=certification_alpha,
        claims=claims,
        area_cap=area_cap,
        tasks=tuple(tasks),
        prefixes=tuple(prefixes),
        excluded_training_sha256=training_hashes,
        excluded_development_sha256=development_hashes,
        query_accounting=query_accounting,
        canonical_sha256=canonical_sha256(root),
    )


def _parse_result_rows(
    payload: Any,
    protocol: ProtocolSpec,
    *,
    required_phase: str,
    selection_sha256: Optional[str] = None,
) -> Tuple[List[ResultRow], str]:
    root = _expect_mapping(payload, context="results")
    expected_root_keys = {
        "schema_version",
        "protocol_sha256",
        "phase",
        "rows",
    }
    if required_phase == "certification":
        expected_root_keys.add("selection_sha256")
    _expect_exact_keys(root, expected_root_keys, context="results")
    version = _strict_int(root["schema_version"], context="results.schema_version")
    if version != SCHEMA_VERSION:
        raise ValueError(f"results.schema_version must be {SCHEMA_VERSION}")
    if root["phase"] != required_phase:
        raise ValueError(f"results.phase must be {required_phase!r}")
    declared_protocol_hash = _sha256(
        root["protocol_sha256"], context="results.protocol_sha256"
    )
    if declared_protocol_hash != protocol.canonical_sha256:
        raise ValueError(
            "results.protocol_sha256 does not match the canonical protocol hash"
        )
    if required_phase == "certification":
        declared_selection_hash = _sha256(
            root["selection_sha256"], context="results.selection_sha256"
        )
        if selection_sha256 is None or declared_selection_hash != selection_sha256:
            raise ValueError(
                "results.selection_sha256 does not match the sealed calibration selection"
            )

    raw_rows = _expect_list(root["rows"], context="results.rows")
    if not raw_rows:
        raise ValueError("results.rows must not be empty")
    rows: List[ResultRow] = []
    seen_keys = set()
    for index, raw in enumerate(raw_rows):
        context = f"results.rows[{index}]"
        obj = _expect_mapping(raw, context=context)
        _expect_exact_keys(
            obj,
            (
                "task_id",
                "prefix_id",
                "trial_id",
                "sample_sha256",
                "pattern_sha256",
                "detector_sha256",
                "outcomes",
                "area",
                "detector_image_queries",
            ),
            context=context,
        )
        task_id = _identifier(obj["task_id"], context=f"{context}.task_id")
        prefix_id = _identifier(obj["prefix_id"], context=f"{context}.prefix_id")
        trial_id = _identifier(obj["trial_id"], context=f"{context}.trial_id")
        key = (prefix_id, task_id, trial_id)
        if key in seen_keys:
            raise ValueError(f"results.rows contains duplicate key {key!r}")
        seen_keys.add(key)

        outcomes_obj = _expect_mapping(obj["outcomes"], context=f"{context}.outcomes")
        _expect_exact_keys(outcomes_obj, CLAIM_NAMES, context=f"{context}.outcomes")
        outcomes = {
            name: _strict_bool(
                outcomes_obj[name], context=f"{context}.outcomes.{name}"
            )
            for name in CLAIM_NAMES
        }
        rows.append(
            ResultRow(
                task_id=task_id,
                prefix_id=prefix_id,
                trial_id=trial_id,
                sample_sha256=_sha256(
                    obj["sample_sha256"], context=f"{context}.sample_sha256"
                ),
                pattern_sha256=_sha256(
                    obj["pattern_sha256"], context=f"{context}.pattern_sha256"
                ),
                detector_sha256=_sha256(
                    obj["detector_sha256"], context=f"{context}.detector_sha256"
                ),
                outcomes=outcomes,
                area=_area_spec(obj["area"], context=f"{context}.area"),
                detector_image_queries=_strict_int(
                    obj["detector_image_queries"],
                    context=f"{context}.detector_image_queries",
                    minimum=0,
                ),
            )
        )
    return rows, canonical_sha256(root)


def _area_within_cap(area: AreaSpec, cap: FractionSpec) -> bool:
    # Cross multiplication is exact and avoids a floating-point boundary.
    return bool(
        area.selected_pixels * cap.denominator
        <= area.sign_pixels * cap.numerator
    )


def _validate_and_index_rows(
    rows: Sequence[ResultRow],
    protocol: ProtocolSpec,
    *,
    phase: str,
    selected_prefix_id: Optional[str] = None,
) -> Dict[Tuple[str, str, str], ResultRow]:
    prefix_map = protocol.prefix_map()
    if phase == "calibration":
        allowed_prefixes = set(prefix_map)
    else:
        if selected_prefix_id not in prefix_map:
            raise ValueError("selected prefix is absent from the protocol")
        allowed_prefixes = {str(selected_prefix_id)}

    expected: Dict[Tuple[str, str, str], Tuple[TrialSpec, TaskPatternSpec, TaskSpec]] = {}
    for prefix_id in allowed_prefixes:
        prefix = prefix_map[prefix_id]
        pattern_map = prefix.task_pattern_map()
        for task in protocol.tasks:
            trials = (
                task.calibration_trials
                if phase == "calibration"
                else task.certification_trials
            )
            for trial in trials:
                expected[(prefix_id, task.task_id, trial.trial_id)] = (
                    trial,
                    pattern_map[task.task_id],
                    task,
                )

    actual = {(row.prefix_id, row.task_id, row.trial_id): row for row in rows}
    missing = sorted(set(expected) - set(actual))
    extra = sorted(set(actual) - set(expected))
    if missing or extra:
        raise ValueError(
            "result rows must exactly match the preregistered phase rows "
            f"(missing={missing}, extra={extra})"
        )

    for key, row in actual.items():
        trial, pattern, task = expected[key]
        if row.sample_sha256 != trial.sample_sha256:
            raise ValueError(f"sample_sha256 mismatch for row {key!r}")
        if row.pattern_sha256 != pattern.pattern_sha256:
            raise ValueError(f"pattern_sha256 mismatch for row {key!r}")
        if row.detector_sha256 != task.detector_sha256:
            raise ValueError(f"detector_sha256 mismatch for row {key!r}")
        if row.area != pattern.area:
            raise ValueError(f"exact material area mismatch for row {key!r}")

        clean = bool(row.outcomes["clean_eligibility"])
        attack = bool(row.outcomes["attack_success"])
        day = bool(row.outcomes["day_preservation"])
        joint = bool(row.outcomes["joint_success"])
        if attack and not clean:
            raise ValueError(
                f"attack_success must fail closed when clean_eligibility is false: {key!r}"
            )
        if day and not clean:
            raise ValueError(
                f"day_preservation must fail closed when clean_eligibility is false: {key!r}"
            )
        area_ok = _area_within_cap(row.area, protocol.area_cap)
        expected_joint = bool(clean and attack and day and area_ok)
        if joint != expected_joint:
            raise ValueError(
                f"joint_success must equal clean AND attack AND day AND exact-area: {key!r}"
            )

    expected_query_total = (
        protocol.query_accounting.expected_calibration_detector_image_queries
        if phase == "calibration"
        else protocol.query_accounting.expected_certification_detector_image_queries
    )
    observed_query_total = sum(row.detector_image_queries for row in rows)
    if observed_query_total != expected_query_total:
        raise ValueError(
            f"{phase} detector image-query total mismatch: "
            f"expected {expected_query_total}, observed {observed_query_total}"
        )
    return actual


def _fraction_dict(value: Fraction) -> Dict[str, Any]:
    return {
        "numerator": int(value.numerator),
        "denominator": int(value.denominator),
        "decimal": float(value),
    }


def _claim_result(
    successes: int,
    trials: int,
    threshold: FractionSpec,
    local_alpha: Fraction,
) -> Dict[str, Any]:
    alpha_float = float(local_alpha)
    lower = clopper_pearson_lower(successes, trials, alpha_float)
    threshold_fraction = threshold.fraction
    exact_p_value = binomial_upper_tail(successes, trials, float(threshold_fraction))
    passes = bool(lower >= float(threshold_fraction))
    return {
        "successes": int(successes),
        "trials": int(trials),
        "point_estimate": float(successes / trials),
        "one_sided_exact_lower_bound": float(lower),
        "threshold": threshold.as_dict(),
        "exact_boundary_tail_probability": float(exact_p_value),
        "passes": passes,
    }


def _evaluate_prefix(
    prefix: PrefixSpec,
    protocol: ProtocolSpec,
    indexed_rows: Mapping[Tuple[str, str, str], ResultRow],
    *,
    phase: str,
    local_alpha: Fraction,
) -> Dict[str, Any]:
    pattern_map = prefix.task_pattern_map()
    task_reports: List[Dict[str, Any]] = []
    for task in protocol.tasks:
        trials = (
            task.calibration_trials
            if phase == "calibration"
            else task.certification_trials
        )
        task_rows = [
            indexed_rows[(prefix.prefix_id, task.task_id, trial.trial_id)]
            for trial in trials
        ]
        claims: Dict[str, Any] = {}
        for claim_name in CLAIM_NAMES:
            successes = sum(
                1 for row in task_rows if bool(row.outcomes[claim_name])
            )
            claims[claim_name] = _claim_result(
                successes,
                len(task_rows),
                protocol.claims[claim_name],
                local_alpha,
            )
        area = pattern_map[task.task_id].area
        area_ok = _area_within_cap(area, protocol.area_cap)
        area_report = {
            **area.as_dict(),
            "cap": protocol.area_cap.as_dict(),
            "comparison": (
                f"{area.selected_pixels}*{protocol.area_cap.denominator} <= "
                f"{area.sign_pixels}*{protocol.area_cap.numerator}"
            ),
            "passes": area_ok,
            "statistical_alpha_consumed": False,
        }
        task_pass = bool(
            area_ok and all(claims[name]["passes"] for name in CLAIM_NAMES)
        )
        task_reports.append(
            {
                "task_id": task.task_id,
                "trials": len(task_rows),
                "claims": claims,
                "exact_area": area_report,
                "detector_image_queries": sum(
                    row.detector_image_queries for row in task_rows
                ),
                "task_pass": task_pass,
            }
        )

    worst_case: Dict[str, Any] = {}
    for claim_name in CLAIM_NAMES:
        rows = [
            (
                task_report["task_id"],
                float(
                    task_report["claims"][claim_name][
                        "one_sided_exact_lower_bound"
                    ]
                ),
            )
            for task_report in task_reports
        ]
        worst_value = min(value for _, value in rows)
        worst_case[claim_name] = {
            "lower_bound": worst_value,
            "task_ids": [task_id for task_id, value in rows if value == worst_value],
        }

    all_tasks_pass = bool(all(task["task_pass"] for task in task_reports))
    return {
        "prefix_id": prefix.prefix_id,
        "order": int(prefix.order),
        "tasks": task_reports,
        "worst_task_lower_bounds": worst_case,
        "all_tasks_pass": all_tasks_pass,
    }


def _base_report(
    protocol: ProtocolSpec,
    *,
    phase: str,
    rows_canonical_sha256: str,
    comparison_count: int,
    local_alpha: Fraction,
) -> Dict[str, Any]:
    phase_alpha = (
        protocol.calibration_alpha
        if phase == "calibration"
        else protocol.certification_alpha
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "method": "preregistered_two_phase_exact_binomial_risk_certification",
        "phase": phase,
        "generated_at_utc": datetime.now(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z"),
        "scope": SCOPE_LABEL,
        "protocol_id": protocol.protocol_id,
        "prefix_family_sha256": protocol.prefix_family_sha256,
        "protocol_sha256": protocol.canonical_sha256,
        "results_canonical_sha256": rows_canonical_sha256,
        "estimand": {
            "population_description": protocol.estimand.population_description,
            "sampling_unit": protocol.estimand.sampling_unit,
            "task_aggregation": protocol.estimand.task_aggregation,
            "fixed_sample_size": True,
            "interpretation": SCOPE_LABEL,
        },
        "simultaneous_inference": {
            "bound": "one-sided exact Clopper-Pearson by stable binomial-tail inversion",
            "allocation": "equal Bonferroni over the preregistered family",
            "familywise_alpha": phase_alpha.as_dict(),
            "comparison_count": int(comparison_count),
            "local_alpha": _fraction_dict(local_alpha),
            "simultaneous_confidence_at_least": float(1 - phase_alpha.fraction),
            "area_is_exact_not_stochastic": True,
        },
    }


def calibrate_prefixes(protocol_payload: Any, results_payload: Any) -> Dict[str, Any]:
    """Evaluate all prefixes on calibration trials and seal one selection."""
    protocol = parse_protocol(protocol_payload)
    rows, rows_hash = _parse_result_rows(
        results_payload, protocol, required_phase="calibration"
    )
    indexed = _validate_and_index_rows(rows, protocol, phase="calibration")
    comparison_count = len(protocol.prefixes) * len(protocol.tasks) * len(CLAIM_NAMES)
    local_alpha = bonferroni_local_alpha(
        protocol.calibration_alpha.fraction, comparison_count
    )
    prefix_reports = [
        _evaluate_prefix(
            prefix,
            protocol,
            indexed,
            phase="calibration",
            local_alpha=local_alpha,
        )
        for prefix in sorted(protocol.prefixes, key=lambda item: item.order)
    ]
    passing = [item for item in prefix_reports if bool(item["all_tasks_pass"])]
    selected = min(passing, key=lambda item: item["order"]) if passing else None
    selection = {
        "status": "selected" if selected is not None else "no_prefix_passed",
        "selected_prefix_id": (
            str(selected["prefix_id"]) if selected is not None else None
        ),
        "selected_order": int(selected["order"]) if selected is not None else None,
        "selection_rule": SELECTION_RULE,
    }
    selection_digest = canonical_sha256(selection)
    query = protocol.query_accounting
    report = {
        **_base_report(
            protocol,
            phase="calibration",
            rows_canonical_sha256=rows_hash,
            comparison_count=comparison_count,
            local_alpha=local_alpha,
        ),
        "candidate_family": {
            "candidate_count": len(prefix_reports),
            "all_candidates_evaluated": True,
            "same_preregistered_trials_for_every_candidate": True,
            "success_monotonicity_assumed": False,
            "selection_rule": SELECTION_RULE,
        },
        "prefixes": prefix_reports,
        "selection": selection,
        "selection_sha256": selection_digest,
        "calibration_status": selection["status"],
        "final_certificate_issued": False,
        "queries": {
            "counting_rule": "one detector evaluation of one image is one query",
            "pre_evaluation_counts_source": (
                "preregistered protocol declarations; not reconstructed by this tool"
            ),
            "calibration_count_source": (
                "per-result-row ledger, reconciled to the preregistered total"
            ),
            "offline_training_detector_image_queries": (
                query.offline_training_detector_image_queries
            ),
            "development_selection_detector_image_queries": (
                query.development_selection_detector_image_queries
            ),
            "prefix_generation_detector_image_queries": (
                query.prefix_generation_detector_image_queries
            ),
            "expected_calibration_detector_image_queries": (
                query.expected_calibration_detector_image_queries
            ),
            "observed_calibration_detector_image_queries": sum(
                row.detector_image_queries for row in rows
            ),
            "reconciled": True,
        },
    }
    report["report_sha256"] = report_sha256(report)
    return report


def _validated_selection(
    selection_report: Any,
    protocol: ProtocolSpec,
) -> Tuple[str, str, Mapping[str, Any], int]:
    report = _expect_mapping(selection_report, context="selection_report")
    if report.get("phase") != "calibration":
        raise ValueError("selection_report.phase must be 'calibration'")
    if report.get("protocol_sha256") != protocol.canonical_sha256:
        raise ValueError("selection report belongs to a different protocol")
    declared_report_hash = _sha256(
        report.get("report_sha256"), context="selection_report.report_sha256"
    )
    if declared_report_hash != report_sha256(report):
        raise ValueError("selection report integrity hash is invalid")
    selection = _expect_mapping(
        report.get("selection"), context="selection_report.selection"
    )
    _expect_exact_keys(
        selection,
        ("status", "selected_prefix_id", "selected_order", "selection_rule"),
        context="selection_report.selection",
    )
    if selection["status"] != "selected":
        raise ValueError("calibration did not select a passing prefix")
    if selection["selection_rule"] != SELECTION_RULE:
        raise ValueError("selection report uses an unknown selection rule")
    prefix_id = _identifier(
        selection["selected_prefix_id"],
        context="selection_report.selection.selected_prefix_id",
    )
    selected_order = _strict_int(
        selection["selected_order"],
        context="selection_report.selection.selected_order",
        minimum=0,
    )
    prefix = protocol.prefix_map().get(prefix_id)
    if prefix is None or prefix.order != selected_order:
        raise ValueError("selected prefix/order is inconsistent with the protocol")
    declared_selection_hash = _sha256(
        report.get("selection_sha256"),
        context="selection_report.selection_sha256",
    )
    if declared_selection_hash != canonical_sha256(selection):
        raise ValueError("selection digest is invalid")
    calibration_queries = report.get("queries", {}).get(
        "observed_calibration_detector_image_queries"
    )
    calibration_queries = _strict_int(
        calibration_queries,
        context="selection_report.queries.observed_calibration_detector_image_queries",
        minimum=0,
    )
    if (
        calibration_queries
        != protocol.query_accounting.expected_calibration_detector_image_queries
    ):
        raise ValueError("selection report calibration-query accounting is inconsistent")
    return prefix_id, declared_selection_hash, selection, calibration_queries


def certify_selected_prefix(
    protocol_payload: Any,
    selection_report: Any,
    results_payload: Any,
) -> Dict[str, Any]:
    """Certify the sealed prefix on untouched certification trials."""
    protocol = parse_protocol(protocol_payload)
    prefix_id, selection_digest, selection, calibration_queries = _validated_selection(
        selection_report, protocol
    )
    rows, rows_hash = _parse_result_rows(
        results_payload,
        protocol,
        required_phase="certification",
        selection_sha256=selection_digest,
    )
    indexed = _validate_and_index_rows(
        rows,
        protocol,
        phase="certification",
        selected_prefix_id=prefix_id,
    )
    comparison_count = len(protocol.tasks) * len(CLAIM_NAMES)
    local_alpha = bonferroni_local_alpha(
        protocol.certification_alpha.fraction, comparison_count
    )
    prefix = protocol.prefix_map()[prefix_id]
    prefix_report = _evaluate_prefix(
        prefix,
        protocol,
        indexed,
        phase="certification",
        local_alpha=local_alpha,
    )
    certificate_pass = bool(prefix_report["all_tasks_pass"])
    query = protocol.query_accounting
    certification_queries = sum(row.detector_image_queries for row in rows)
    full_lifecycle_queries = (
        query.offline_training_detector_image_queries
        + query.development_selection_detector_image_queries
        + query.prefix_generation_detector_image_queries
        + calibration_queries
        + certification_queries
    )
    report = {
        **_base_report(
            protocol,
            phase="certification",
            rows_canonical_sha256=rows_hash,
            comparison_count=comparison_count,
            local_alpha=local_alpha,
        ),
        "selection": dict(selection),
        "selection_sha256": selection_digest,
        "selected_prefix": prefix_report,
        "aggregation": {
            "rule": "all_tasks",
            "all_tasks_pass": certificate_pass,
            "worst_task_lower_bounds": prefix_report["worst_task_lower_bounds"],
            "primary_worst_joint_lower_bound": prefix_report[
                "worst_task_lower_bounds"
            ]["joint_success"]["lower_bound"],
        },
        "certificate_status": "pass" if certificate_pass else "fail",
        "certificate_pass": certificate_pass,
        "claims_formal_or_all_world_guarantee": False,
        "queries": {
            "counting_rule": "one detector evaluation of one image is one query",
            "pre_evaluation_counts_source": (
                "preregistered protocol declarations; not reconstructed by this tool"
            ),
            "evaluation_counts_source": (
                "calibration and certification per-row ledgers, each reconciled "
                "to its preregistered total"
            ),
            "offline_training_detector_image_queries": (
                query.offline_training_detector_image_queries
            ),
            "development_selection_detector_image_queries": (
                query.development_selection_detector_image_queries
            ),
            "prefix_generation_detector_image_queries": (
                query.prefix_generation_detector_image_queries
            ),
            "calibration_detector_image_queries": calibration_queries,
            "expected_certification_detector_image_queries": (
                query.expected_certification_detector_image_queries
            ),
            "observed_certification_detector_image_queries": certification_queries,
            "full_lifecycle_detector_image_queries": full_lifecycle_queries,
            "reconciled": True,
        },
    }
    report["report_sha256"] = report_sha256(report)
    return report


__all__ = [
    "CLAIM_NAMES",
    "PHASES",
    "SCHEMA_VERSION",
    "SCOPE_LABEL",
    "SELECTION_RULE",
    "calibrate_prefixes",
    "canonical_json_bytes",
    "canonical_sha256",
    "certify_selected_prefix",
    "parse_protocol",
    "report_sha256",
]
