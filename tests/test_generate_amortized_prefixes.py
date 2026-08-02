import copy

import pytest

from tools.generate_amortized_prefixes import (
    artifact_sha256,
    assemble_candidate_family,
    canonical_sha256,
)


def _record(task_id, count):
    return {
        "task_id": task_id,
        "prefixes": [
            {
                "order": order,
                "pattern_sha256": canonical_sha256(
                    {"task": task_id, "order": order}
                ),
                "selected_pixels": order,
                "sign_pixels": 100,
            }
            for order in range(1, count + 1)
        ],
    }


def test_candidate_family_uses_only_the_common_contiguous_prefixes():
    family = assemble_candidate_family([_record("a", 3), _record("b", 2)])
    assert [item["prefix_id"] for item in family] == [
        "prefix-0001",
        "prefix-0002",
    ]
    assert [item["order"] for item in family] == [1, 2]
    assert [row["task_id"] for row in family[1]["task_patterns"]] == ["a", "b"]


def test_candidate_family_rejects_missing_or_noncontiguous_sequences():
    with pytest.raises(ValueError, match="at least one prefix"):
        assemble_candidate_family([_record("a", 1), _record("b", 0)])
    malformed = _record("a", 2)
    malformed["prefixes"][1]["order"] = 3
    with pytest.raises(ValueError, match="contiguous"):
        assemble_candidate_family([malformed])


def test_artifact_digest_excludes_only_its_self_digest():
    artifact = {"schema_version": 1, "tasks": ["a"]}
    artifact["artifact_sha256"] = artifact_sha256(artifact)
    assert artifact_sha256(artifact) == artifact["artifact_sha256"]
    tampered = copy.deepcopy(artifact)
    tampered["tasks"].append("b")
    assert artifact_sha256(tampered) != artifact["artifact_sha256"]
