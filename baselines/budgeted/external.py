"""Fail-closed adapters for exact upstream comparison methods.

These adapters do not imitate FIPatch, PatchAttack, or BAAP.  They accept an
interactive upstream runner only after a pinned artifact has been hashed and
its method identity has been checked.  All detector feedback still flows
through :class:`BudgetedEvaluator`.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Protocol, Sequence, Tuple

from .protocol import PROTOCOL_VERSION, BudgetedEvaluator, SearchResult
from .json_io import StrictJSONError, strict_json_load


EXTERNAL_RUNNER_API = "budgeted-proposal-runner-v2"
EXTERNAL_METHOD_IDS = (
    "fipatch_pso",
    "patchattack",
    "baap_2606_18318",
    "per_task_ppo",
    "wei_rl_2212_12995",
    "meta_attack_iccv21",
    "impact_de_es",
    "simulator_attack_cvpr21",
)


class ExternalMethodUnavailable(RuntimeError):
    """Raised before search when no verified exact upstream runner exists."""


@dataclass(frozen=True)
class RegistryEntry:
    method_id: str
    label: str
    fidelity: str
    execution: str
    claim_boundary: str


METHOD_REGISTRY: Mapping[str, RegistryEntry] = {
    "random_search": RegistryEntry(
        "random_search", "Random search", "native_implementation", "native",
        "Repository baseline; not a reproduction of a named paper.",
    ),
    "forward_greedy": RegistryEntry(
        "forward_greedy", "Forward greedy", "native_implementation", "native",
        "Repository baseline; not a reproduction of a named paper.",
    ),
    "genetic_algorithm": RegistryEntry(
        "genetic_algorithm", "Binary genetic algorithm", "native_implementation", "native",
        "Generic GA baseline; not a reproduction of a named paper.",
    ),
    "gaussian_es": RegistryEntry(
        "gaussian_es", "Gaussian evolution strategy", "native_implementation", "native",
        "Generic isotropic ES; not CMA-ES, NES, or BAAP.",
    ),
    "cma_es": RegistryEntry(
        "cma_es", "CMA-ES", "reference_package", "optional_dependency",
        "Runs the installed cma package and records its resolved version; a paper artifact must pin that version.",
    ),
    "fipatch_style_pso_proxy": RegistryEntry(
        "fipatch_style_pso_proxy",
        "Binary PSO family proxy",
        "proxy_not_fipatch_reproduction",
        "native",
        "Executable family proxy only; never label its result as FIPatch.",
    ),
    "fipatch_pso": RegistryEntry(
        "fipatch_pso", "FIPatch-style PSO", "requires_declared_mapping_and_independent_audit", "external_runner",
        "A pinned runner/mapping establishes identity only; equivalence requires independent semantic audit.",
    ),
    "patchattack": RegistryEntry(
        "patchattack", "PatchAttack", "requires_declared_mapping_and_independent_audit", "external_runner",
        "A pinned runner/mapping establishes identity only; equivalence requires independent semantic audit.",
    ),
    "baap_2606_18318": RegistryEntry(
        "baap_2606_18318",
        "Budget-Aware Adaptive Adversarial Patches (arXiv:2606.18318)",
        "requires_declared_mapping_and_independent_audit",
        "external_runner",
        "Thompson-sampling + NES growing-patch method; no proxy is labeled as BAAP.",
    ),
    "per_task_ppo": RegistryEntry(
        "per_task_ppo",
        "Independently trained per-task PPO",
        "requires_declared_mapping_and_independent_audit",
        "external_runner",
        "Requires a frozen per-task policy artifact and interactive proposal runner.",
    ),
    "wei_rl_2212_12995": RegistryEntry(
        "wei_rl_2212_12995",
        "Wei et al. simultaneous position/perturbation RL",
        "requires_declared_mapping_and_independent_audit",
        "external_runner",
        "arXiv:2212.12995 / TPAMI; unavailable without an exact pinned runner.",
    ),
    "meta_attack_iccv21": RegistryEntry(
        "meta_attack_iccv21",
        "ICCV 2021 Meta-Attack",
        "requires_declared_mapping_and_independent_audit",
        "external_runner",
        "Unavailable without an exact pinned upstream runner.",
    ),
    "impact_de_es": RegistryEntry(
        "impact_de_es",
        "IMPACT differential-evolution + ES",
        "requires_declared_mapping_and_independent_audit",
        "external_runner",
        "Unavailable without an exact pinned upstream runner and material mapping.",
    ),
    "simulator_attack_cvpr21": RegistryEntry(
        "simulator_attack_cvpr21",
        "CVPR 2021 Simulator Attack",
        "requires_declared_mapping_and_independent_audit",
        "external_runner",
        "Requires independent audit of patch/detector adaptation; offline simulator-training queries must be reported separately.",
    ),
}


@dataclass(frozen=True)
class ExternalArtifactManifest:
    method_id: str
    paper_url: str
    upstream_repository: str
    upstream_revision: str
    artifact_path: Path
    artifact_sha256: str
    candidate_mapping_path: Path
    candidate_mapping_sha256: str
    candidate_mapping_status: str
    offline_detector_queries: int
    runner_api: str
    manifest_path: Path

    @classmethod
    def load(
        cls, path: str, *, expected_method_id: str
    ) -> "ExternalArtifactManifest":
        manifest_path = Path(path).resolve()
        try:
            raw = strict_json_load(
                manifest_path, label="external artifact manifest"
            )
        except StrictJSONError as exc:
            raise ExternalMethodUnavailable(
                f"cannot read external artifact manifest: {exc}"
            ) from exc
        required = {
            "schema_version",
            "method_id",
            "paper_url",
            "upstream_repository",
            "upstream_revision",
            "artifact_path",
            "artifact_sha256",
            "candidate_mapping_path",
            "candidate_mapping_sha256",
            "candidate_mapping_status",
            "offline_detector_queries",
            "runner_api",
        }
        if not isinstance(raw, dict) or set(raw) != required:
            raise ExternalMethodUnavailable(
                "external manifest must contain exactly: "
                + ", ".join(sorted(required))
            )
        if type(raw["schema_version"]) is not int or raw["schema_version"] != 1:
            raise ExternalMethodUnavailable("unsupported external manifest schema")
        if not isinstance(raw["method_id"], str):
            raise ExternalMethodUnavailable("method_id must be a string")
        method_id = raw["method_id"]
        if method_id != str(expected_method_id) or method_id not in EXTERNAL_METHOD_IDS:
            raise ExternalMethodUnavailable(
                f"external manifest method_id {method_id!r} does not match "
                f"requested method {expected_method_id!r}"
            )
        for name in (
            "paper_url",
            "upstream_repository",
            "upstream_revision",
        ):
            if not isinstance(raw[name], str) or not raw[name].strip():
                raise ExternalMethodUnavailable(f"{name} must be a non-empty string")
        if not isinstance(raw["runner_api"], str) or raw["runner_api"] != EXTERNAL_RUNNER_API:
            raise ExternalMethodUnavailable("runner_api is not supported")
        if not isinstance(raw["artifact_sha256"], str):
            raise ExternalMethodUnavailable("artifact_sha256 must be a string")
        expected_hash = raw["artifact_sha256"]
        if (
            len(expected_hash) != 64
            or expected_hash != expected_hash.lower()
            or any(ch not in "0123456789abcdef" for ch in expected_hash)
        ):
            raise ExternalMethodUnavailable("artifact_sha256 must be lowercase SHA-256")
        if not isinstance(raw["artifact_path"], str) or not raw["artifact_path"].strip():
            raise ExternalMethodUnavailable("artifact_path must be a non-empty string")
        artifact_path = (manifest_path.parent / raw["artifact_path"]).resolve()
        if not artifact_path.is_file():
            raise ExternalMethodUnavailable(
                f"pinned upstream artifact does not exist: {artifact_path}"
            )
        actual_hash = _sha256_file(artifact_path)
        if actual_hash != expected_hash:
            raise ExternalMethodUnavailable(
                "pinned upstream artifact SHA-256 mismatch"
            )
        if raw["candidate_mapping_status"] != "declared_preregistered_and_audited":
            raise ExternalMethodUnavailable(
                "candidate_mapping_status must be "
                "'declared_preregistered_and_audited'"
            )
        if (
            isinstance(raw["offline_detector_queries"], bool)
            or not isinstance(raw["offline_detector_queries"], int)
            or raw["offline_detector_queries"] < 0
        ):
            raise ExternalMethodUnavailable(
                "offline_detector_queries must be a non-negative integer"
            )
        if (
            not isinstance(raw["candidate_mapping_path"], str)
            or not raw["candidate_mapping_path"].strip()
        ):
            raise ExternalMethodUnavailable(
                "candidate_mapping_path must be a non-empty string"
            )
        if not isinstance(raw["candidate_mapping_sha256"], str):
            raise ExternalMethodUnavailable(
                "candidate_mapping_sha256 must be a string"
            )
        mapping_hash = raw["candidate_mapping_sha256"]
        if (
            len(mapping_hash) != 64
            or mapping_hash != mapping_hash.lower()
            or any(ch not in "0123456789abcdef" for ch in mapping_hash)
        ):
            raise ExternalMethodUnavailable(
                "candidate_mapping_sha256 must be lowercase SHA-256"
            )
        mapping_path = (
            manifest_path.parent / raw["candidate_mapping_path"]
        ).resolve()
        if not mapping_path.is_file():
            raise ExternalMethodUnavailable(
                f"candidate-space mapping does not exist: {mapping_path}"
            )
        if _sha256_file(mapping_path) != mapping_hash:
            raise ExternalMethodUnavailable(
                "candidate-space mapping SHA-256 mismatch"
            )
        return cls(
            method_id=method_id,
            paper_url=str(raw["paper_url"]),
            upstream_repository=str(raw["upstream_repository"]),
            upstream_revision=str(raw["upstream_revision"]),
            artifact_path=artifact_path,
            artifact_sha256=expected_hash,
            candidate_mapping_path=mapping_path,
            candidate_mapping_sha256=mapping_hash,
            candidate_mapping_status=raw["candidate_mapping_status"],
            offline_detector_queries=raw["offline_detector_queries"],
            runner_api=str(raw["runner_api"]),
            manifest_path=manifest_path,
        )


class ExternalProposalRunner(Protocol):
    """Interactive runner contract implemented by exact upstream wrappers."""

    method_id: str
    artifact_sha256: str
    candidate_mapping_sha256: str
    candidate_mapping_status: str
    runner_api: str
    query_access: str

    def initialize(self, context: Mapping[str, Any]) -> None:
        """Initialize without querying the detector."""

    def propose(
        self, history: Tuple[Mapping[str, Any], ...]
    ) -> Optional[Sequence[int]]:
        """Return the next mask, or ``None`` to stop."""


def run_external_adapter(
    evaluator: BudgetedEvaluator,
    *,
    method_id: str,
    seed: int,
    artifact_manifest_path: str,
    runner: Optional[ExternalProposalRunner],
    max_evaluations: Optional[int] = None,
) -> SearchResult:
    """Run a verified upstream proposal loop under the common oracle.

    A missing or mismatched artifact/runner raises before a candidate detector
    query.  Merely naming a method is never enough to produce a result row.
    """
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if max_evaluations is not None and (
        isinstance(max_evaluations, bool)
        or not isinstance(max_evaluations, int)
        or max_evaluations < 0
    ):
        raise ValueError("max_evaluations must be a non-negative integer or None")
    manifest = ExternalArtifactManifest.load(
        artifact_manifest_path, expected_method_id=method_id
    )
    if runner is None:
        raise ExternalMethodUnavailable(
            f"{method_id} requires an exact interactive upstream runner"
        )
    if str(getattr(runner, "method_id", "")) != manifest.method_id:
        raise ExternalMethodUnavailable("runner method_id does not match manifest")
    if str(getattr(runner, "artifact_sha256", "")).lower() != manifest.artifact_sha256:
        raise ExternalMethodUnavailable("runner artifact hash does not match manifest")
    if (
        str(getattr(runner, "candidate_mapping_sha256", "")).lower()
        != manifest.candidate_mapping_sha256
    ):
        raise ExternalMethodUnavailable(
            "runner candidate-space mapping hash does not match manifest"
        )
    if (
        str(getattr(runner, "candidate_mapping_status", ""))
        != "declared_preregistered_and_audited"
    ):
        raise ExternalMethodUnavailable(
            "runner has not declared the preregistered mapping/audit status"
        )
    if str(getattr(runner, "runner_api", "")) != EXTERNAL_RUNNER_API:
        raise ExternalMethodUnavailable("runner API does not match manifest")
    if str(getattr(runner, "query_access", "")) != "budgeted_evaluator_only":
        raise ExternalMethodUnavailable(
            "runner must declare query_access='budgeted_evaluator_only'"
        )

    context: Dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "method_id": method_id,
        "seed": int(seed),
        "objective_id": str(evaluator.oracle.objective_id),
        "candidate_mapping_sha256": manifest.candidate_mapping_sha256,
        "candidate_mapping_status": manifest.candidate_mapping_status,
        "offline_detector_queries": manifest.offline_detector_queries,
        "dimension": int(evaluator.dimension),
        "selectable_indices": list(evaluator.selectable_indices),
        "cell_material_pixels": list(evaluator.cell_material_pixels),
        "sign_material_pixels": int(evaluator.sign_material_pixels),
        "budget": {
            "detector_query_limit": int(evaluator.budget.detector_query_limit),
            "detector_query_scope": evaluator.budget.detector_query_scope,
            "material_pixel_limit": int(evaluator.budget.material_pixel_limit),
            "initial_detector_queries": int(evaluator.initial_queries),
            "queries_per_evaluation": int(evaluator.queries_per_evaluation),
        },
    }
    _call_without_oracle_queries(evaluator, runner.initialize, context)
    history: Tuple[Mapping[str, Any], ...] = ()
    count = 0
    while evaluator.can_evaluate_more() and (
        max_evaluations is None or count < int(max_evaluations)
    ):
        proposal = _call_without_oracle_queries(
            evaluator, runner.propose, history
        )
        if proposal is None:
            break
        row = evaluator.evaluate(proposal)
        history = history + (row.to_dict(),)
        count += 1

    status = (
        "query_budget_exhausted"
        if not evaluator.can_evaluate_more()
        else "completed"
    )
    return evaluator.result(
        method=method_id,
        seed=seed,
        implementation=(
            f"upstream:{manifest.upstream_repository}@{manifest.upstream_revision}"
        ),
        fidelity="verified_external_adapter_with_declared_mapping",
        config={
            "paper_url": manifest.paper_url,
            "artifact_sha256": manifest.artifact_sha256,
            "candidate_mapping_sha256": manifest.candidate_mapping_sha256,
            "candidate_mapping_status": manifest.candidate_mapping_status,
            "offline_detector_queries": manifest.offline_detector_queries,
            "runner_api": manifest.runner_api,
            "max_evaluations": max_evaluations,
        },
        status=status,
        note=(
            "Hashes establish runner/mapping identity, not semantic equivalence; "
            "any named-method equivalence claim requires independent audit."
        ),
    )


def _call_without_oracle_queries(
    evaluator: BudgetedEvaluator, function: Any, *args: Any
) -> Any:
    before = getattr(evaluator.oracle, "detector_queries_total", None)
    before_value = int(before() if callable(before) else before) if before is not None else None
    result = function(*args)
    after = getattr(evaluator.oracle, "detector_queries_total", None)
    after_value = int(after() if callable(after) else after) if after is not None else None
    if (
        before_value is not None
        and after_value is not None
        and after_value != before_value
    ):
        raise RuntimeError(
            "external runner queried the detector outside BudgetedEvaluator"
        )
    return result


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
