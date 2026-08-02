import copy
import hashlib

import pytest

from tools.build_risk_protocol import build_protocol, prefix_artifact_sha256
from utils.risk_certification import CLAIM_NAMES, parse_protocol


def _digest(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _artifact():
    artifact = {
        "schema_version": 1,
        "method": "task_amortized_prefix_valid_support_batch",
        "status": "frozen_candidate_family_not_a_certificate",
        "generation": {"split": "development"},
        "query_accounting": {
            "offline_training_detector_image_queries": 100,
            "development_prefix_detector_image_queries": 20,
        },
        "limitations": {
            "nonempirical_debug": False,
            "incomplete_training_ledger_debug": False,
        },
        "tasks": [{"task_id": "task-a"}],
        "candidate_prefixes": [
            {
                "prefix_id": "prefix-0001",
                "order": 1,
                "task_patterns": [
                    {
                        "task_id": "task-a",
                        "pattern_sha256": _digest("pattern-a-1"),
                        "selected_pixels": 1,
                        "sign_pixels": 10,
                    }
                ],
            }
        ],
    }
    artifact["artifact_sha256"] = prefix_artifact_sha256(artifact)
    return artifact


def _inventory():
    return {
        "schema_version": 1,
        "protocol_id": "bound_protocol",
        "estimand": {
            "population_description": "held-out synthetic test population",
            "sampling_unit": "independent_scene_cluster",
            "task_aggregation": "all_tasks",
            "fixed_sample_size": True,
        },
        "familywise_alpha": {
            "calibration": {"numerator": 1, "denominator": 20},
            "certification": {"numerator": 1, "denominator": 20},
        },
        "claims": {
            name: {"numerator": 1, "denominator": 2} for name in CLAIM_NAMES
        },
        "area_cap": {"numerator": 1, "denominator": 5},
        "tasks": [
            {
                "task_id": "task-a",
                "detector_sha256": _digest("detector-a"),
                "calibration_trials": [
                    {
                        "trial_id": "cal-1",
                        "sample_sha256": _digest("calibration-1"),
                    }
                ],
                "certification_trials": [
                    {
                        "trial_id": "cert-1",
                        "sample_sha256": _digest("certification-1"),
                    }
                ],
            }
        ],
        "excluded_sample_sha256": {
            "training": [_digest("train-1")],
            "development": [_digest("development-1")],
        },
        "query_accounting": {
            "development_selection_detector_image_queries": 3,
            "expected_calibration_detector_image_queries": 4,
            "expected_certification_detector_image_queries": 4,
        },
    }


def test_builder_binds_family_hash_patterns_area_and_queries():
    artifact = _artifact()
    protocol = build_protocol(artifact, _inventory())
    parsed = parse_protocol(protocol)
    assert parsed.prefix_family_sha256 == artifact["artifact_sha256"]
    assert parsed.query_accounting.offline_training_detector_image_queries == 100
    assert parsed.query_accounting.prefix_generation_detector_image_queries == 20
    pattern = parsed.prefixes[0].task_patterns[0]
    assert pattern.area.selected_pixels == 1
    assert pattern.area.sign_pixels == 10


def test_builder_rejects_tampering_debug_artifacts_and_task_mismatch():
    tampered = _artifact()
    tampered["candidate_prefixes"][0]["order"] = 2
    with pytest.raises(ValueError, match="digest"):
        build_protocol(tampered, _inventory())

    debug = _artifact()
    debug["limitations"]["nonempirical_debug"] = True
    debug["artifact_sha256"] = prefix_artifact_sha256(debug)
    with pytest.raises(ValueError, match="Debug|debug"):
        build_protocol(debug, _inventory())

    mismatch = copy.deepcopy(_inventory())
    mismatch["tasks"][0]["task_id"] = "task-b"
    with pytest.raises(ValueError, match="task IDs differ"):
        build_protocol(_artifact(), mismatch)
