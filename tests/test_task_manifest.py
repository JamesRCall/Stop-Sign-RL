import json

import pytest

from utils.task_manifest import TaskManifestError, load_task_manifest


def _task(task_id="train-25-55", split="train", sign_instance="sign-a"):
    return {
        "task_id": task_id,
        "split": split,
        "sign_instance_id": sign_instance,
        "background_split_id": f"background-{task_id}",
        "camera_id": f"camera-{task_id}",
        "material_batch_id": f"material-{task_id}",
        "physical_run_group": f"runs-{task_id}",
        "detector_id": "fine-grained-yolo-a",
        "source_class": "speed_limit_25",
        "target_class": "speed_limit_55",
        "attack_mode": "targeted_misclassification",
        "calibration_sha256": "a" * 64,
        "condition_features": [25 / 160, 55 / 160],
        "weight": 1.0,
        "environment": {
            "sign_profile": "custom",
            "sign_image": "assets/speed25.png",
            "sign_active_image": "assets/speed25_uv.png",
            "bgdir": "backgrounds/train",
            "physics_calibration": "calibration/camera-a.json",
        },
    }


def _manifest(tasks=None):
    return {
        "schema_version": 1,
        "manifest_id": "unit-test-manifest",
        "description": "Synthetic manifest used only by unit tests.",
        "condition_feature_names": ["source_speed", "target_speed"],
        "leakage_keys": [
            "sign_instance_id",
            "background_split_id",
            "camera_id",
            "material_batch_id",
            "physical_run_group",
        ],
        "tasks": tasks or [_task()],
    }


def _write(tmp_path, payload, name="tasks.json", *, indent=None):
    path = tmp_path / name
    path.write_text(json.dumps(payload, indent=indent), encoding="utf-8")
    return path


def test_manifest_is_strict_hashed_and_resolves_relative_paths(tmp_path):
    first = load_task_manifest(_write(tmp_path, _manifest(), "a.json"))
    second = load_task_manifest(_write(tmp_path, _manifest(), "b.json", indent=2))
    assert first.canonical_sha256 == second.canonical_sha256
    assert first.source_sha256 != second.source_sha256
    task = first.tasks_for_split("train")[0]
    resolved = task.resolved_environment(first.directory)
    assert resolved["source_class"] == "speed_limit_25"
    assert resolved["attack_target_class"] == "speed_limit_55"
    assert resolved["sign_image"].endswith("assets\\speed25.png")


def test_manifest_rejects_cross_split_instance_leakage(tmp_path):
    train = _task("train", "train", "shared-sign")
    certification = _task("cert", "certification", "shared-sign")
    with pytest.raises(TaskManifestError, match="split leakage"):
        load_task_manifest(_write(tmp_path, _manifest([train, certification])))


def test_manifest_rejects_unknown_fields_and_missing_target(tmp_path):
    unknown = _manifest()
    unknown["tasks"][0]["reviewer_surprise"] = True
    with pytest.raises(TaskManifestError, match="invalid fields"):
        load_task_manifest(_write(tmp_path, unknown, "unknown.json"))

    missing_target = _manifest()
    missing_target["tasks"][0]["target_class"] = None
    with pytest.raises(TaskManifestError, match="target_class is required"):
        load_task_manifest(_write(tmp_path, missing_target, "target.json"))


def test_manifest_rejects_target_for_nontargeted_mode(tmp_path):
    ambiguous = _manifest()
    ambiguous["tasks"][0]["attack_mode"] = "untargeted_misclassification"
    with pytest.raises(TaskManifestError, match="must be null"):
        load_task_manifest(_write(tmp_path, ambiguous, "ambiguous.json"))
