"""Shared, auditable protocol for budget-matched black-box comparisons."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
import operator
from typing import Any, Dict, Iterable, List, Mapping, Optional, Protocol, Sequence, Tuple


PROTOCOL_VERSION = "budgeted-black-box-v3-grouped-actions"


class BudgetError(RuntimeError):
    """Base class for a comparison-budget violation."""


class BudgetExhausted(BudgetError):
    """Raised before an evaluation that would exceed the detector budget."""


class MaterialBudgetExceeded(BudgetError):
    """Raised before an evaluation whose exact painted area is too large."""


@dataclass(frozen=True)
class BudgetSpec:
    """Hard budgets shared by every method in a comparison.

    ``detector_query_limit`` counts individual detector input images, not API
    calls or batches.  ``material_pixel_limit`` counts source-sign alpha-mask
    pixels covered by selected grid cells.  Both limits are inclusive.
    """

    detector_query_limit: int
    material_pixel_limit: int
    detector_query_scope: str = "online_search_including_clean_references"

    def __post_init__(self) -> None:
        query_limit = _exact_integer(
            self.detector_query_limit, "detector_query_limit"
        )
        material_limit = _exact_integer(
            self.material_pixel_limit, "material_pixel_limit"
        )
        if query_limit < 0:
            raise ValueError("detector_query_limit must be non-negative")
        if material_limit < 0:
            raise ValueError("material_pixel_limit must be non-negative")
        object.__setattr__(self, "detector_query_limit", query_limit)
        object.__setattr__(self, "material_pixel_limit", material_limit)
        if self.detector_query_scope != "online_search_including_clean_references":
            raise ValueError(
                "detector_query_scope must be "
                "'online_search_including_clean_references'"
            )


@dataclass(frozen=True)
class OracleObservation:
    """One oracle response before accounting metadata is attached."""

    score: float
    joint_success: bool
    metrics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not math.isfinite(float(self.score)):
            raise ValueError("oracle score must be finite")
        if type(self.joint_success) is not bool:
            raise ValueError("oracle joint_success must be a boolean")
        if not isinstance(self.metrics, Mapping):
            raise ValueError("oracle metrics must be a mapping")


class CandidateOracle(Protocol):
    """Only interface through which optimizers may observe the objective."""

    dimension: int
    selectable_indices: Sequence[int]
    candidate_group_ids: Sequence[int]
    cell_material_pixels: Sequence[int]
    sign_material_pixels: int
    objective_material_pixel_limit: Optional[int]
    detector_queries_per_evaluation: int
    initial_detector_queries: int
    objective_id: str

    def evaluate(self, selected_indices: Tuple[int, ...]) -> OracleObservation:
        """Evaluate one canonical action-token set at the fixed query cost.

        Tokens may be ordinary binary cell choices or categorical material
        choices. ``candidate_group_ids`` makes alternatives for one physical
        cell mutually exclusive without charging that cell more than once.
        """


@dataclass(frozen=True)
class CandidateEvaluation:
    evaluation_index: int
    selected_indices: Tuple[int, ...]
    score: float
    joint_success: bool
    selected_material_pixels: int
    sign_material_pixels: int
    material_fraction: float
    detector_queries_this_evaluation: int
    detector_queries_cumulative: int
    metrics: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        row = asdict(self)
        row["selected_indices"] = list(self.selected_indices)
        # Explicit alias for v3: under grouped categorical searches these are
        # action tokens, while ``selected_indices`` remains for compatibility
        # with existing result readers.
        row["selected_action_tokens"] = list(self.selected_indices)
        row["metrics"] = _json_safe(self.metrics)
        return row


@dataclass(frozen=True)
class SearchResult:
    method: str
    seed: int
    status: str
    protocol_version: str
    objective_id: str
    budget: BudgetSpec
    initial_detector_queries: int
    detector_queries_used: int
    evaluated_candidates: int
    best: Optional[CandidateEvaluation]
    trace: Tuple[CandidateEvaluation, ...]
    implementation: str
    fidelity: str
    config: Mapping[str, Any] = field(default_factory=dict)
    note: Optional[str] = None

    @property
    def best_successful(self) -> Optional[CandidateEvaluation]:
        """Highest-scoring candidate satisfying the complete joint predicate."""
        successful = (row for row in self.trace if row.joint_success)
        return max(successful, key=candidate_rank, default=None)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "method": self.method,
            "seed": int(self.seed),
            "status": self.status,
            "protocol_version": self.protocol_version,
            "objective_id": self.objective_id,
            "budget": asdict(self.budget),
            "initial_detector_queries": int(self.initial_detector_queries),
            "detector_queries_used": int(self.detector_queries_used),
            "evaluated_candidates": int(self.evaluated_candidates),
            "best": self.best.to_dict() if self.best is not None else None,
            "best_successful": (
                self.best_successful.to_dict()
                if self.best_successful is not None
                else None
            ),
            "trace": [row.to_dict() for row in self.trace],
            "implementation": self.implementation,
            "fidelity": self.fidelity,
            "config": _json_safe(self.config),
            "note": self.note,
        }


def canonical_candidate(indices: Iterable[int], dimension: int) -> Tuple[int, ...]:
    """Return a sorted, duplicate-free canonical candidate or fail closed."""
    values = tuple(
        _exact_integer(value, "selected index") for value in indices
    )
    if len(values) != len(set(values)):
        raise ValueError("selected_indices must not contain duplicates")
    if any(value < 0 or value >= int(dimension) for value in values):
        raise ValueError("selected_indices contains an out-of-range index")
    return tuple(sorted(values))


def candidate_material_pixels(
    candidate: Sequence[int], cell_material_pixels: Sequence[int]
) -> int:
    return int(sum(int(cell_material_pixels[index]) for index in candidate))


def candidate_rank(row: CandidateEvaluation) -> Tuple[float, int, int, Tuple[int, ...]]:
    """Deterministic ranking of the shared scalar objective and its ties."""
    return (
        float(row.score),
        1 if row.joint_success else 0,
        -int(row.selected_material_pixels),
        tuple(-value for value in row.selected_indices),
    )


class BudgetedEvaluator:
    """Fail-closed accounting wrapper shared by all optimizers.

    Candidate validity and the fixed query cost are checked *before* the oracle
    is invoked.  The oracle's own cumulative counter is checked when available
    so a backend cannot silently under-report detector-image use.
    """

    def __init__(self, oracle: CandidateOracle, budget: BudgetSpec):
        self.oracle = oracle
        self.budget = budget
        self.dimension = _exact_integer(oracle.dimension, "oracle dimension")
        self.cell_material_pixels = tuple(
            _exact_integer(value, "cell material cost")
            for value in oracle.cell_material_pixels
        )
        self.candidate_group_ids = tuple(
            _exact_integer(value, "candidate group id")
            for value in getattr(oracle, "candidate_group_ids", range(self.dimension))
        )
        self.selectable_indices = tuple(
            sorted(
                _exact_integer(value, "selectable index")
                for value in oracle.selectable_indices
            )
        )
        if self.dimension <= 0:
            raise ValueError("oracle dimension must be positive")
        if len(self.cell_material_pixels) != self.dimension:
            raise ValueError("cell_material_pixels length must equal dimension")
        if any(value < 0 for value in self.cell_material_pixels):
            raise ValueError("cell material costs must be non-negative")
        if len(self.candidate_group_ids) != self.dimension:
            raise ValueError("candidate_group_ids length must equal dimension")
        if any(value < 0 for value in self.candidate_group_ids):
            raise ValueError("candidate_group_ids must be non-negative")
        if len(self.selectable_indices) != len(set(self.selectable_indices)):
            raise ValueError("selectable_indices must not contain duplicates")
        if not self.selectable_indices:
            raise ValueError("oracle must expose at least one selectable index")
        if any(
            value < 0 or value >= self.dimension
            for value in self.selectable_indices
        ):
            raise ValueError("selectable_indices contains an out-of-range index")
        self._selectable_set = frozenset(self.selectable_indices)
        self.sign_material_pixels = _exact_integer(
            oracle.sign_material_pixels, "sign_material_pixels"
        )
        if self.sign_material_pixels <= 0:
            raise ValueError("sign_material_pixels must be positive")
        self.queries_per_evaluation = _exact_integer(
            oracle.detector_queries_per_evaluation,
            "detector_queries_per_evaluation",
        )
        self.initial_queries = _exact_integer(
            oracle.initial_detector_queries, "initial_detector_queries"
        )
        if self.queries_per_evaluation <= 0:
            raise ValueError("detector_queries_per_evaluation must be positive")
        if self.initial_queries < 0:
            raise ValueError("initial_detector_queries must be non-negative")
        if self.initial_queries > int(budget.detector_query_limit):
            raise BudgetExhausted(
                "reference-image queries already exceed detector_query_limit"
            )
        if int(budget.material_pixel_limit) > self.sign_material_pixels:
            raise ValueError(
                "material_pixel_limit cannot exceed sign_material_pixels"
            )
        objective_material_limit = getattr(
            oracle, "objective_material_pixel_limit", None
        )
        if objective_material_limit is not None:
            objective_material_limit = _exact_integer(
                objective_material_limit,
                "objective_material_pixel_limit",
            )
        self.objective_material_pixel_limit = objective_material_limit
        if (
            objective_material_limit is not None
            and int(objective_material_limit)
            != int(budget.material_pixel_limit)
        ):
            raise ValueError(
                "comparison material_pixel_limit does not match the exact "
                "pixel limit implied by env.area_cap_frac: comparison="
                f"{budget.material_pixel_limit}, objective="
                f"{int(objective_material_limit)}"
            )
        self._queries_used = self.initial_queries
        self._trace: List[CandidateEvaluation] = []

    @property
    def queries_used(self) -> int:
        return int(self._queries_used)

    @property
    def remaining_queries(self) -> int:
        return int(self.budget.detector_query_limit) - self.queries_used

    @property
    def trace(self) -> Tuple[CandidateEvaluation, ...]:
        return tuple(self._trace)

    @property
    def best(self) -> Optional[CandidateEvaluation]:
        return max(self._trace, key=candidate_rank) if self._trace else None

    def can_evaluate_more(self) -> bool:
        return self.remaining_queries >= self.queries_per_evaluation

    def material_pixels(self, indices: Iterable[int]) -> int:
        candidate = canonical_candidate(indices, self.dimension)
        self._validate_selectable(candidate)
        self._validate_candidate_groups(candidate)
        return candidate_material_pixels(candidate, self.cell_material_pixels)

    def is_material_feasible(self, indices: Iterable[int]) -> bool:
        return self.material_pixels(indices) <= int(self.budget.material_pixel_limit)

    def evaluate(self, indices: Iterable[int]) -> CandidateEvaluation:
        candidate = canonical_candidate(indices, self.dimension)
        self._validate_selectable(candidate)
        self._validate_candidate_groups(candidate)
        selected_pixels = candidate_material_pixels(
            candidate, self.cell_material_pixels
        )
        if selected_pixels > int(self.budget.material_pixel_limit):
            raise MaterialBudgetExceeded(
                f"candidate uses {selected_pixels} material pixels; limit is "
                f"{self.budget.material_pixel_limit}"
            )
        if not self.can_evaluate_more():
            raise BudgetExhausted(
                f"evaluation costs {self.queries_per_evaluation} detector images; "
                f"only {self.remaining_queries} remain"
            )

        counter_before = _oracle_counter(self.oracle)
        try:
            observation = self.oracle.evaluate(candidate)
        except BaseException:
            counter_after_failure = _oracle_counter(self.oracle)
            if counter_before is not None and counter_after_failure is not None:
                actual_failure_queries = (
                    int(counter_after_failure) - int(counter_before)
                )
                if actual_failure_queries < 0:
                    raise RuntimeError(
                        "oracle detector query counter decreased during failure"
                    )
                self._queries_used += actual_failure_queries
            raise
        counter_after = _oracle_counter(self.oracle)
        if counter_before is not None and counter_after is not None:
            actual = int(counter_after) - int(counter_before)
            if actual != self.queries_per_evaluation:
                if actual >= 0:
                    self._queries_used += actual
                raise RuntimeError(
                    "oracle query-accounting invariant failed: expected "
                    f"{self.queries_per_evaluation}, observed {actual}"
                )

        # The detector cost is irrevocably spent even if result serialization
        # subsequently fails closed.
        self._queries_used += self.queries_per_evaluation
        safe_metrics = _json_safe(observation.metrics)
        if not isinstance(safe_metrics, dict):
            raise ValueError("oracle metrics must serialize to a JSON object")
        row = CandidateEvaluation(
            evaluation_index=len(self._trace),
            selected_indices=candidate,
            score=float(observation.score),
            joint_success=bool(observation.joint_success),
            selected_material_pixels=selected_pixels,
            sign_material_pixels=self.sign_material_pixels,
            material_fraction=(
                float(selected_pixels) / float(self.sign_material_pixels)
            ),
            detector_queries_this_evaluation=self.queries_per_evaluation,
            detector_queries_cumulative=self.queries_used,
            metrics=safe_metrics,
        )
        self._trace.append(row)
        return row

    def _validate_selectable(self, candidate: Sequence[int]) -> None:
        invalid = [value for value in candidate if value not in self._selectable_set]
        if invalid:
            raise ValueError(
                "candidate contains non-selectable indices: "
                + ", ".join(str(value) for value in invalid[:8])
            )

    def group_id(self, index: int) -> int:
        """Return the exclusive fabrication group for one candidate token."""

        return int(self.candidate_group_ids[int(index)])

    def _validate_candidate_groups(self, candidate: Sequence[int]) -> None:
        groups = [self.group_id(index) for index in candidate]
        if len(groups) != len(set(groups)):
            raise ValueError(
                "candidate assigns multiple action tokens to one exclusive group"
            )

    def result(
        self,
        *,
        method: str,
        seed: int,
        implementation: str,
        fidelity: str = "native_implementation",
        config: Optional[Mapping[str, Any]] = None,
        status: str = "completed",
        note: Optional[str] = None,
    ) -> SearchResult:
        return SearchResult(
            method=str(method),
            seed=_exact_integer(seed, "result seed"),
            status=str(status),
            protocol_version=PROTOCOL_VERSION,
            objective_id=str(self.oracle.objective_id),
            budget=self.budget,
            initial_detector_queries=self.initial_queries,
            detector_queries_used=self.queries_used,
            evaluated_candidates=len(self._trace),
            best=self.best,
            trace=self.trace,
            implementation=str(implementation),
            fidelity=str(fidelity),
            config=dict(config or {}),
            note=note,
        )

    def contract_fingerprint(self) -> str:
        """Hash all fields that must be identical across compared methods."""
        body = {
            "protocol": PROTOCOL_VERSION,
            "objective_id": str(self.oracle.objective_id),
            "dimension": self.dimension,
            "selectable_indices": list(self.selectable_indices),
            "cell_material_pixels": list(self.cell_material_pixels),
            "candidate_group_ids": list(self.candidate_group_ids),
            "sign_material_pixels": self.sign_material_pixels,
            "objective_material_pixel_limit": self.objective_material_pixel_limit,
            "queries_per_evaluation": self.queries_per_evaluation,
            "initial_queries": self.initial_queries,
            "budget": asdict(self.budget),
        }
        payload = json.dumps(
            body,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
        return hashlib.sha256(payload).hexdigest()


def _oracle_counter(oracle: CandidateOracle) -> Optional[int]:
    value = getattr(oracle, "detector_queries_total", None)
    if value is None:
        return None
    return _exact_integer(
        value() if callable(value) else value,
        "oracle detector query counter",
    )


def _exact_integer(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer, not a boolean")
    try:
        return int(operator.index(value))
    except TypeError as exc:
        raise ValueError(f"{name} must be an integer") from exc


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("non-finite numeric value is not permitted in results")
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(number):
        raise ValueError("non-finite numeric value is not permitted in results")
    return number
