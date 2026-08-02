"""Bind a frozen development prefix family to a preregistered risk plan.

The trial inventory supplies the statistical design and sealed sample hashes;
the prefix artifact supplies immutable task-pattern hashes, exact areas, and
pre-evaluation query totals.  The resulting document is validated by the same
strict parser used during calibration and certification.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.experiment_manifest import write_manifest  # noqa: E402
from utils.risk_certification import (  # noqa: E402
    canonical_sha256,
    parse_protocol,
)


METHOD = "task_amortized_prefix_valid_support_batch"
INVENTORY_KEYS = {
    "schema_version",
    "protocol_id",
    "estimand",
    "familywise_alpha",
    "claims",
    "area_cap",
    "tasks",
    "excluded_sample_sha256",
    "query_accounting",
}
INVENTORY_QUERY_KEYS = {
    "development_selection_detector_image_queries",
    "expected_calibration_detector_image_queries",
    "expected_certification_detector_image_queries",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build and validate a two-phase risk protocol from a frozen prefix "
            "family and a preregistered trial inventory."
        )
    )
    parser.add_argument("--prefix-family", required=True)
    parser.add_argument("--trial-inventory", required=True)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def _reject_constant(value: str) -> None:
    raise ValueError(f"JSON contains non-finite constant {value!r}")


def _no_duplicate_keys(pairs: Sequence[tuple[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"JSON contains duplicate key {key!r}")
        result[key] = value
    return result


def _load_json(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=_reject_constant,
            object_pairs_hook=_no_duplicate_keys,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read JSON file {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return value


def prefix_artifact_sha256(artifact: Mapping[str, Any]) -> str:
    payload = dict(artifact)
    payload.pop("artifact_sha256", None)
    return canonical_sha256(payload)


def _exact_keys(value: Mapping[str, Any], expected: set[str], context: str) -> None:
    actual = set(value)
    missing = sorted(expected - actual)
    unknown = sorted(actual - expected)
    if missing or unknown:
        raise ValueError(
            f"{context} has invalid fields (missing={missing}, unknown={unknown})"
        )


def _nonnegative_int(value: Any, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{context} must be a nonnegative integer")
    return int(value)


def _validate_prefix_artifact(artifact: Mapping[str, Any]) -> str:
    declared = artifact.get("artifact_sha256")
    if not isinstance(declared, str) or len(declared) != 64:
        raise ValueError("prefix family lacks a lowercase SHA-256 artifact digest")
    actual = prefix_artifact_sha256(artifact)
    if declared != actual:
        raise ValueError("prefix-family artifact digest is invalid")
    if artifact.get("schema_version") != 1 or artifact.get("method") != METHOD:
        raise ValueError("unsupported prefix-family schema or method")
    if artifact.get("status") != "frozen_candidate_family_not_a_certificate":
        raise ValueError("prefix family is not marked frozen")
    generation = artifact.get("generation")
    if not isinstance(generation, dict) or generation.get("split") != "development":
        raise ValueError("paper-facing prefix families must come from development")
    limits = artifact.get("limitations")
    if not isinstance(limits, dict):
        raise ValueError("prefix family lacks limitation flags")
    if bool(limits.get("nonempirical_debug")) or bool(
        limits.get("incomplete_training_ledger_debug")
    ):
        raise ValueError("debug prefix families cannot enter a paper-facing protocol")
    return declared


def build_protocol(
    prefix_artifact: Mapping[str, Any],
    trial_inventory: Mapping[str, Any],
) -> Dict[str, Any]:
    family_hash = _validate_prefix_artifact(prefix_artifact)
    _exact_keys(trial_inventory, INVENTORY_KEYS, "trial inventory")
    if trial_inventory["schema_version"] != 1:
        raise ValueError("trial inventory schema_version must be 1")

    inventory_queries = trial_inventory["query_accounting"]
    if not isinstance(inventory_queries, dict):
        raise ValueError("trial inventory query_accounting must be an object")
    _exact_keys(
        inventory_queries,
        INVENTORY_QUERY_KEYS,
        "trial inventory query_accounting",
    )
    artifact_queries = prefix_artifact.get("query_accounting")
    if not isinstance(artifact_queries, dict):
        raise ValueError("prefix family lacks query accounting")
    offline_queries = _nonnegative_int(
        artifact_queries.get("offline_training_detector_image_queries"),
        "offline training detector-image queries",
    )
    prefix_queries = _nonnegative_int(
        artifact_queries.get("development_prefix_detector_image_queries"),
        "development prefix detector-image queries",
    )

    inventory_tasks = trial_inventory["tasks"]
    if not isinstance(inventory_tasks, list) or not inventory_tasks:
        raise ValueError("trial inventory must contain tasks")
    inventory_task_ids = {
        row.get("task_id") for row in inventory_tasks if isinstance(row, dict)
    }
    if None in inventory_task_ids or len(inventory_task_ids) != len(inventory_tasks):
        raise ValueError("trial inventory task IDs must be present and unique")

    artifact_tasks = prefix_artifact.get("tasks")
    if not isinstance(artifact_tasks, list) or not artifact_tasks:
        raise ValueError("prefix family must contain task records")
    artifact_task_ids = {
        row.get("task_id") for row in artifact_tasks if isinstance(row, dict)
    }
    if artifact_task_ids != inventory_task_ids:
        raise ValueError(
            "prefix-family and trial-inventory task IDs differ "
            f"(prefix={sorted(artifact_task_ids)}, inventory={sorted(inventory_task_ids)})"
        )

    raw_candidates = prefix_artifact.get("candidate_prefixes")
    if not isinstance(raw_candidates, list) or not raw_candidates:
        raise ValueError("prefix family contains no common candidate prefixes")
    prefixes = []
    for candidate_index, candidate in enumerate(raw_candidates):
        if not isinstance(candidate, dict):
            raise ValueError(f"candidate_prefixes[{candidate_index}] must be an object")
        if set(candidate) != {"prefix_id", "order", "task_patterns"}:
            raise ValueError(
                f"candidate_prefixes[{candidate_index}] has an invalid shape"
            )
        patterns = candidate["task_patterns"]
        if not isinstance(patterns, list):
            raise ValueError("candidate task_patterns must be an array")
        prefixes.append(
            {
                "prefix_id": candidate["prefix_id"],
                "order": candidate["order"],
                "task_patterns": [
                    {
                        "task_id": pattern["task_id"],
                        "pattern_sha256": pattern["pattern_sha256"],
                        "area": {
                            "selected_pixels": pattern["selected_pixels"],
                            "sign_pixels": pattern["sign_pixels"],
                        },
                    }
                    for pattern in patterns
                ],
            }
        )

    protocol: Dict[str, Any] = {
        "schema_version": 1,
        "protocol_id": trial_inventory["protocol_id"],
        "prefix_family_sha256": family_hash,
        "estimand": trial_inventory["estimand"],
        "familywise_alpha": trial_inventory["familywise_alpha"],
        "claims": trial_inventory["claims"],
        "area_cap": trial_inventory["area_cap"],
        "tasks": inventory_tasks,
        "prefixes": prefixes,
        "excluded_sample_sha256": trial_inventory["excluded_sample_sha256"],
        "query_accounting": {
            "offline_training_detector_image_queries": offline_queries,
            "development_selection_detector_image_queries": _nonnegative_int(
                inventory_queries[
                    "development_selection_detector_image_queries"
                ],
                "development selection detector-image queries",
            ),
            "prefix_generation_detector_image_queries": prefix_queries,
            "expected_calibration_detector_image_queries": _nonnegative_int(
                inventory_queries[
                    "expected_calibration_detector_image_queries"
                ],
                "expected calibration detector-image queries",
            ),
            "expected_certification_detector_image_queries": _nonnegative_int(
                inventory_queries[
                    "expected_certification_detector_image_queries"
                ],
                "expected certification detector-image queries",
            ),
        },
    }
    parse_protocol(protocol)
    return protocol


def main() -> None:
    args = parse_args()
    family_path = Path(args.prefix_family).expanduser().resolve()
    inventory_path = Path(args.trial_inventory).expanduser().resolve()
    protocol = build_protocol(
        _load_json(family_path),
        _load_json(inventory_path),
    )
    destination = Path(args.out).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite protocol: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    write_manifest(destination, protocol)
    print(f"wrote {destination}")
    print(f"protocol_sha256={canonical_sha256(protocol)}")


if __name__ == "__main__":
    main()
