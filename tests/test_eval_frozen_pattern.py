import json

import pytest
from PIL import Image, ImageDraw

from detectors.class_names import resolve_class_id
from envs.stop_sign_grid_env import TrafficSignGridEnv
import tools.eval_frozen_pattern as frozen_eval
from tools.eval_frozen_pattern import (
    load_frozen_pattern,
    pattern_config_mismatch_groups,
    summarize_metric_rows,
    wilson_interval,
)


def _write_json(tmp_path, payload, name="pattern.json"):
    path = tmp_path / name
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_loads_root_actions_and_preserves_order(tmp_path):
    path = _write_json(
        tmp_path,
        {
            "actions": [8, 3, 11],
            "scene_seed": 42,
            "config": {"grid_cell": 16, "sign_profile": "stop"},
        },
    )
    pattern = load_frozen_pattern(path)
    assert pattern.pattern_type == "actions"
    assert pattern.values == (8, 3, 11)
    assert pattern.locator == "$.actions"
    assert pattern.source_seed == 42
    assert pattern.declared_grid_cell_px == 16


def test_loads_eval_episode_trace_by_explicit_index(tmp_path):
    path = _write_json(
        tmp_path,
        {
            "episodes_detail": [
                {
                    "seed": 1000,
                    "trace": {"grid_cell_px": 8, "selected_indices": [10, 12]},
                },
                {
                    "seed": 1001,
                    "trace": {"grid_cell_px": 8, "selected_indices": [20, 22]},
                },
            ]
        },
    )
    pattern = load_frozen_pattern(
        path,
        requested_type="selected_indices",
        episode_index=1,
    )
    assert pattern.values == (20, 22)
    assert pattern.source_seed == 1001
    assert pattern.locator == "$.episodes_detail[1].trace.selected_indices"
    assert pattern.declared_grid_cell_px == 8


def test_loads_callback_root_trace_and_actual_reset_seed(tmp_path):
    path = _write_json(
        tmp_path,
        {
            "episode_meta": {"reset_seed": 73},
            "attack_mode": "untargeted_misclassification",
            "trace": {
                "grid_cell_px": 16,
                "paint_name": "YellowGlow",
                "selected_indices": [5, 9],
            },
        },
    )
    pattern = load_frozen_pattern(path)
    assert pattern.values == (5, 9)
    assert pattern.locator == "$.trace.selected_indices"
    assert pattern.source_seed == 73
    assert pattern.trace_paint_name == "YellowGlow"
    assert pattern.declared_config["attack_mode"] == "untargeted_misclassification"


def test_loads_root_episode_list(tmp_path):
    path = _write_json(
        tmp_path,
        [
            {"seed": 100, "trace": {"selected_indices": [1]}},
            {"seed": 101, "trace": {"selected_indices": [2, 3]}},
        ],
    )
    pattern = load_frozen_pattern(path, episode_index=1)
    assert pattern.values == (2, 3)
    assert pattern.locator == "$[1].trace.selected_indices"
    assert pattern.source_seed == 101


def test_loads_exported_replica_stencil_flat_indices(tmp_path):
    path = _write_json(
        tmp_path,
        {
            "grid_cell_px": 8,
            "selected_indices_flat": [17, 23],
            "selected_cells_rc": [[2, 1], [2, 7]],
        },
    )
    pattern = load_frozen_pattern(path)
    assert pattern.pattern_type == "selected_indices"
    assert pattern.values == (17, 23)
    assert pattern.locator == "$.selected_indices_flat"
    assert pattern.declared_grid_cell_px == 8


def test_auto_mode_rejects_ambiguous_pattern_artifact(tmp_path):
    path = _write_json(
        tmp_path,
        {"actions": [1, 2], "selected_indices": [8, 9]},
    )
    with pytest.raises(ValueError, match="multiple candidates"):
        load_frozen_pattern(path)
    assert load_frozen_pattern(path, requested_type="actions").values == (1, 2)


@pytest.mark.parametrize(
    "bad_values, message",
    [
        ([1, 1], "duplicate"),
        ([1, 2.0], "must be an integer"),
        ([True], "must be an integer"),
        ([], "empty"),
    ],
)
def test_pattern_values_fail_closed(tmp_path, bad_values, message):
    path = _write_json(tmp_path, {"selected_indices": bad_values})
    with pytest.raises(ValueError, match=message):
        load_frozen_pattern(path)


def test_wilson_interval_matches_known_boundary_values():
    low_zero, high_zero = wilson_interval(0, 100)
    low_all, high_all = wilson_interval(100, 100)
    assert low_zero == pytest.approx(0.0)
    assert high_zero == pytest.approx(0.0369935, rel=1e-5)
    assert low_all == pytest.approx(0.9630065, rel=1e-5)
    assert high_all == pytest.approx(1.0)


def test_metric_summary_ignores_nonfinite_values():
    means, stds = summarize_metric_rows(
        [{"c_on": 0.2}, {"c_on": float("nan")}, {"c_on": 0.6}],
        ["c_on", "missing"],
    )
    assert means["c_on"] == pytest.approx(0.4)
    assert stds["c_on"] == pytest.approx(0.2)
    assert means["missing"] is None
    assert stds["missing"] is None


def test_config_checks_separate_physical_and_protocol_transfer(tmp_path):
    path = _write_json(
        tmp_path,
        {
            "actions": [0],
            "config": {
                "grid_cell": 16,
                "paint": "red",
                "attack_mode": "targeted_misclassification",
                "attack_target_class": "speed limit 55",
                "allowed_alternative_classes": "speed limit 35,speed limit 45",
            },
        },
    )
    pattern = load_frozen_pattern(path)
    requested = {
        "grid_cell": 16,
        "paint": "yellow",
        "paint_list": "",
        "attack_mode": "disappearance",
        "attack_target_class": "",
        "allowed_alternative_classes": "",
    }
    groups = pattern_config_mismatch_groups(pattern, requested)
    assert any(row.startswith("paint ") for row in groups["geometry_material"])
    assert any(row.startswith("attack_mode ") for row in groups["protocol"])
    assert any(
        row.startswith("allowed_alternative_classes ")
        for row in groups["protocol"]
    )


class _AlwaysSourceDetector:
    id_to_name = {1: "stop sign", 2: "speed limit 55"}
    target_id = 1

    def resolve_class_id(self, class_ref, *, role="class"):
        return resolve_class_id(self.id_to_name, class_ref, role=role)

    def infer_detections_batch(self, images):
        return [
            {
                "boxes": [[0.0, 0.0, float(image.width), float(image.height)]],
                "confs": [0.9],
                "clss": [1],
            }
            for image in images
        ]


class _NoStepEnvironment(TrafficSignGridEnv):
    def step(self, action):  # pragma: no cover - failure guard only
        raise AssertionError("frozen certification must never call env.step")


def _circle_sign(size=64):
    image = Image.new("RGBA", (size, size), (255, 255, 255, 0))
    ImageDraw.Draw(image).ellipse(
        (2, 2, size - 3, size - 3),
        fill=(255, 255, 255, 255),
    )
    return image


def test_certification_reuses_one_stencil_without_stepping(tmp_path, monkeypatch):
    pattern_path = _write_json(
        tmp_path,
        {
            "actions": [0],
            "scene_seed": 7,
            "config": {"grid_cell": 16, "sign_profile": "stop"},
        },
    )
    sign = _circle_sign()
    env = _NoStepEnvironment(
        stop_sign_image=sign,
        stop_sign_uv_image=sign.copy(),
        background_images=[Image.new("RGB", (128, 128), "gray")],
        pole_image=None,
        grid_cell_px=16,
        source_class="stop sign",
        detector_instance=_AlwaysSourceDetector(),
        img_size=(128, 128),
        eval_K=2,
        obs_size=(64, 64),
        transform_strength=0.0,
        localization_iou_threshold=0.01,
    )
    monkeypatch.setattr(frozen_eval, "build_env_from_args", lambda _args: env)
    args = frozen_eval.parse_args(
        [
            "--pattern-json",
            str(pattern_path),
            "--episodes",
            "3",
            "--seed-base",
            "100",
            "--eval-K",
            "2",
            "--grid-cell",
            "16",
            "--bg-mode",
            "solid",
            "--no-pole",
            "--localization-iou",
            "0.01",
        ]
    )

    report = frozen_eval.run_certification(args)

    assert [row["seed"] for row in report["rows"]] == [100, 101, 102]
    assert all(row["selected_cells"] == 1 for row in report["rows"])
    assert all(row["success"] is False for row in report["rows"])
    assert all(row["clean_baseline_image_queries"] == 4 for row in report["rows"])
    assert all(row["overlay_image_queries"] == 4 for row in report["rows"])
    assert report["detector_image_queries_total"] == 24
    assert report["protocol"]["no_policy_inference"] is True
    assert report["protocol"]["one_pattern_reused_for_every_seed"] is True
    assert report["protocol"]["seed_role"] == "fresh_rng_seed_certification"
    assert report["protocol"]["disjoint_background_dataset_split_enforced"] is False
    assert report["pattern"]["applied_stencil"]["selected_indices"]
    assert len({row["physical_stencil_sha256"] for row in report["rows"]}) == 1
    assert "misclassification_success_rate" in report["mean_metrics"]
