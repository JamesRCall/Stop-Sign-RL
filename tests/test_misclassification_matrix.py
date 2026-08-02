import copy
import json
from pathlib import Path

import pytest

from tools.run_misclassification_matrix import validate_matrix


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
