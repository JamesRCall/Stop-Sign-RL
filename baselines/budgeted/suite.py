"""Run multiple optimizers against fresh, contract-identical oracles."""
from __future__ import annotations

from dataclasses import dataclass
import inspect
import operator
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

from .external import METHOD_REGISTRY
from .json_io import atomic_write_json_new, strict_json_dumps
from .optimizers import (
    cma_es_search,
    evolution_strategy_search,
    fipatch_style_pso_proxy_search,
    genetic_search,
    greedy_search,
    random_search,
)
from .protocol import (
    PROTOCOL_VERSION,
    BudgetSpec,
    BudgetedEvaluator,
    CandidateOracle,
    SearchResult,
)


NATIVE_RUNNERS: Mapping[str, Callable[..., SearchResult]] = {
    "random_search": random_search,
    "forward_greedy": greedy_search,
    "genetic_algorithm": genetic_search,
    "gaussian_es": evolution_strategy_search,
    "fipatch_style_pso_proxy": fipatch_style_pso_proxy_search,
    "cma_es": cma_es_search,
}


@dataclass(frozen=True)
class ComparisonReport:
    protocol_version: str
    seed: int
    contract_fingerprint: str
    results: Tuple[SearchResult, ...]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "protocol_version": self.protocol_version,
            "seed": int(self.seed),
            "contract_fingerprint": self.contract_fingerprint,
            "results": [result.to_dict() for result in self.results],
            "method_registry": registry_snapshot(),
        }

    def save_json(self, path: str) -> None:
        atomic_write_json_new(Path(path), self.to_dict())


def registry_snapshot() -> Dict[str, Dict[str, str]]:
    """Machine-readable availability and claim boundary for every slot."""
    rows: Dict[str, Dict[str, str]] = {}
    for method_id, entry in METHOD_REGISTRY.items():
        if method_id in NATIVE_RUNNERS:
            availability = (
                "optional_dependency"
                if entry.execution == "optional_dependency"
                else "runnable"
            )
        else:
            availability = "requires_verified_external_runner"
        rows[method_id] = {
            "label": entry.label,
            "fidelity": entry.fidelity,
            "execution": entry.execution,
            "availability": availability,
            "claim_boundary": entry.claim_boundary,
        }
    return rows


def run_native_suite(
    oracle_factory: Callable[[], CandidateOracle],
    *,
    budget: BudgetSpec,
    methods: Sequence[str],
    seed: int,
    method_configs: Optional[Mapping[str, Mapping[str, Any]]] = None,
) -> ComparisonReport:
    """Run each method from a fresh oracle and reject contract drift.

    ``oracle_factory`` must reconstruct the same detector, scene, EOT seeds,
    objective, and grid on every call.  Its hashed contract is checked before a
    result is admitted to the report.
    """
    if not methods:
        raise ValueError("methods must not be empty")
    if isinstance(seed, bool):
        raise ValueError("seed must be an integer, not a boolean")
    try:
        normalized_seed = int(operator.index(seed))
    except TypeError as exc:
        raise ValueError("seed must be an integer") from exc
    if normalized_seed < 0:
        raise ValueError("seed must be non-negative")
    unknown = [method for method in methods if method not in NATIVE_RUNNERS]
    if unknown:
        raise ValueError(
            "native suite cannot run unavailable/external methods: "
            + ", ".join(unknown)
        )
    if len(methods) != len(set(methods)):
        raise ValueError("methods must not contain duplicates")

    configs = dict(method_configs or {})
    extra_configs = sorted(set(configs) - set(methods))
    if extra_configs:
        raise ValueError(
            "method_configs contains methods that are not selected: "
            + ", ".join(extra_configs)
        )
    normalized_configs: Dict[str, Dict[str, Any]] = {}
    for method in methods:
        raw_config = configs.get(method, {})
        if not isinstance(raw_config, Mapping):
            raise ValueError(f"method config for {method} must be a mapping")
        config = dict(raw_config)
        if "evaluator" in config or "seed" in config:
            raise ValueError(
                f"method config for {method} must not override evaluator or seed"
            )
        signature = inspect.signature(NATIVE_RUNNERS[method])
        allowed = {
            name
            for name, parameter in signature.parameters.items()
            if name not in {"evaluator", "seed"}
            and parameter.kind
            in {
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            }
        }
        unexpected = sorted(set(config) - allowed)
        if unexpected:
            raise ValueError(
                f"unknown config keys for {method}: " + ", ".join(unexpected)
            )
        # Programmatic callers receive the same finite/JSON-compatible config
        # guarantee as CLI callers.
        strict_json_dumps(config)
        normalized_configs[method] = config

    expected_fingerprint: Optional[str] = None
    results = []
    for method in methods:
        oracle = oracle_factory()
        failure: Optional[BaseException] = None
        try:
            evaluator = BudgetedEvaluator(oracle, budget)
            fingerprint = evaluator.contract_fingerprint()
            if expected_fingerprint is None:
                expected_fingerprint = fingerprint
            elif fingerprint != expected_fingerprint:
                raise RuntimeError(
                    f"comparison contract drift detected before running {method}"
                )
            result = NATIVE_RUNNERS[method](
                evaluator, seed=normalized_seed, **normalized_configs[method]
            )
            results.append(result)
        except BaseException as exc:
            failure = exc
            raise
        finally:
            close = getattr(oracle, "close", None)
            if callable(close):
                try:
                    close()
                except Exception as close_error:
                    if failure is None:
                        raise
                    if hasattr(failure, "add_note"):
                        failure.add_note(
                            f"oracle close also failed: {close_error!r}"
                        )

    assert expected_fingerprint is not None
    return ComparisonReport(
        protocol_version=PROTOCOL_VERSION,
        seed=normalized_seed,
        contract_fingerprint=expected_fingerprint,
        results=tuple(results),
    )
