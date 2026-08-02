import copy
import json
from pathlib import Path

import pytest

import tools.run_misclassification_matrix as matrix_runner
from tools.run_misclassification_matrix import (
    _output_has_experiment_content,
    validate_matrix,
)


REPOSITORY = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPOSITORY / "configs" / "misclassification_models.json"


def _payload():
    return json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))


def _write_resolved_copy(tmp_path, payload):
    resolved = copy.deepcopy(payload)
    for key in ("sign_day", "sign_active", "backgrounds"):
        value = Path(resolved["assets"][key])
        resolved["assets"][key] = str(
            (DEFAULT_CONFIG.parent / value).resolve() if not value.is_absolute() else value
        )
    for model in resolved["models"]:
        if model["weights"]:
            value = Path(model["weights"])
            model["weights"] = str(
                (DEFAULT_CONFIG.parent / value).resolve()
                if not value.is_absolute()
                else value
            )
    destination = tmp_path / "matrix.json"
    destination.write_text(json.dumps(resolved), encoding="utf-8")
    return destination


def test_default_misclassification_matrix_declares_every_supported_model_family():
    config = validate_matrix(DEFAULT_CONFIG)
    assert config["attack"] == {
        "mode": "targeted_misclassification",
        "source_class": "stop sign",
        "target_class": "traffic light",
        "allowed_alternative_classes": "",
    }
    assert config["patch"]["paint_action_mode"] == "joint_palette"
    assert len(config["models"]) == 9
    assert {model["detector"] for model in config["models"]} == {
        "yolo",
        "torchvision",
        "rtdetr",
    }


def test_matrix_rejects_duplicate_model_ids(tmp_path):
    payload = _payload()
    payload["models"][1]["id"] = payload["models"][0]["id"]
    with pytest.raises(ValueError, match="duplicate model id"):
        validate_matrix(_write_resolved_copy(tmp_path, payload))


def test_matrix_rejects_targeted_attack_without_target(tmp_path):
    payload = _payload()
    payload["attack"]["target_class"] = ""
    with pytest.raises(ValueError, match="requires.*target_class"):
        validate_matrix(_write_resolved_copy(tmp_path, payload))


def test_matrix_rejects_ambiguous_joint_palette(tmp_path):
    payload = _payload()
    payload["patch"]["paint_palette"] = "red,red"
    with pytest.raises(ValueError, match="must be unique"):
        validate_matrix(_write_resolved_copy(tmp_path, payload))


def test_launcher_files_do_not_turn_new_output_into_resume(tmp_path):
    output = tmp_path / "matrix"
    output.mkdir()
    (output / "launcher.log").write_text("nohup: ignoring input\n", encoding="utf-8")
    (output / "launcher.pid").write_text("20570\n", encoding="utf-8")

    assert not _output_has_experiment_content(output)


def test_resume_bootstraps_new_output_containing_launcher_files(tmp_path, monkeypatch):
    output = tmp_path / "matrix"
    output.mkdir()
    (output / "launcher.log").write_text("nohup: ignoring input\n", encoding="utf-8")
    (output / "launcher.pid").write_text("20570\n", encoding="utf-8")

    monkeypatch.setattr(
        matrix_runner,
        "_run_child",
        lambda **kwargs: {
            "model_id": kwargs["model"]["id"],
            "status": "completed",
            "returncode": 0,
        },
    )
    monkeypatch.setattr(matrix_runner, "_aggregate", lambda output, models: 0)

    result = matrix_runner.main(
        [
            "--config",
            str(DEFAULT_CONFIG),
            "--output",
            str(output),
            "--model-ids",
            "yolov8n",
            "--seeds",
            "0",
            "--max-steps",
            "1",
            "--minimum-steps",
            "0",
            "--save-freq",
            "1",
            "--episodes",
            "1",
            "--query-budget",
            "1",
            "--eval-k",
            "1",
            "--train-eval-k",
            "1",
            "--support-scenes",
            "1",
            "--episode-steps",
            "1",
            "--max-prefix",
            "1",
            "--resume",
            "--skip-tests",
            "--no-archive",
        ]
    )

    assert result == 0
    assert (output / "matrix_provenance.json").is_file()
    assert (output / "STATUS.txt").read_text(encoding="utf-8").startswith("COMPLETED\n")


def test_unknown_file_still_requires_resume_provenance(tmp_path):
    output = tmp_path / "matrix"
    output.mkdir()
    (output / "launcher.log").write_text("", encoding="utf-8")
    (output / "partial-result.json").write_text("{}\n", encoding="utf-8")

    assert _output_has_experiment_content(output)
