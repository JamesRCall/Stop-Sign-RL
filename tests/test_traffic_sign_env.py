import numpy as np
import pytest
from PIL import Image, ImageDraw

from detectors.class_names import resolve_class_id
from envs.stop_sign_grid_env import (
    StopSignGridEnv,
    TrafficSignGridEnv,
    _random_perspective_coeffs,
)
from envs.attack_objective import aggregate_metrics, summarize_detection


class FakeDetector:
    id_to_name = {1: "speed limit 25", 2: "speed limit 55"}
    target_id = 1

    def resolve_class_id(self, class_ref, *, role="class"):
        return resolve_class_id(self.id_to_name, class_ref, role=role)

    def infer_detections_batch(self, images):
        return [
            {"boxes": [], "confs": [], "clss": []}
            for _ in images
        ]


def _circle_sign(size=64):
    image = Image.new("RGBA", (size, size), (255, 255, 255, 0))
    ImageDraw.Draw(image).ellipse((2, 2, size - 3, size - 3), fill=(255, 255, 255, 255))
    return image


def test_perspective_coefficients_are_finite_without_blas_solver():
    coefficients = _random_perspective_coeffs(
        128,
        96,
        np.random.default_rng(123),
    )
    assert coefficients.shape == (8,)
    assert np.isfinite(coefficients).all()


def test_speed_sign_geometry_uses_exact_painted_pixel_fraction():
    sign = _circle_sign()
    env = TrafficSignGridEnv(
        stop_sign_image=sign,
        stop_sign_uv_image=sign.copy(),
        background_images=[Image.new("RGB", (128, 128), "gray")],
        pole_image=None,
        grid_cell_px=16,
        cell_cover_thresh=0.10,
        source_class="speed limit 25",
        detector_instance=FakeDetector(),
        img_size=(128, 128),
        eval_K=1,
        obs_size=(64, 64),
    )
    assert StopSignGridEnv is TrafficSignGridEnv
    env._episode_cells = np.zeros((env.Gh, env.Gw), dtype=bool)
    row, col = [int(v) for v in env._valid_coords[0]]
    env._episode_cells[row, col] = True
    expected = env._cell_pixel_areas[row, col] / env._sign_pixel_area
    assert env._area_frac_selected() == pytest.approx(expected)


def test_canonical_full_grid_actions_keep_coordinates_stable_across_silhouettes():
    sign = _circle_sign()
    legacy = TrafficSignGridEnv(
        stop_sign_image=sign,
        stop_sign_uv_image=sign.copy(),
        background_images=[],
        pole_image=None,
        grid_cell_px=16,
        cell_cover_thresh=0.10,
        source_class="speed limit 25",
        detector_instance=FakeDetector(),
    )
    canonical = TrafficSignGridEnv(
        stop_sign_image=sign,
        stop_sign_uv_image=sign.copy(),
        background_images=[],
        pole_image=None,
        grid_cell_px=16,
        cell_cover_thresh=0.10,
        action_indexing="canonical_full_grid",
        terminate_on_success=False,
        source_class="speed limit 25",
        detector_instance=FakeDetector(),
    )
    assert legacy.action_space.n == legacy._n_valid
    assert canonical.action_space.n == canonical.Gh * canonical.Gw
    canonical._episode_cells = np.zeros((canonical.Gh, canonical.Gw), dtype=bool)
    row, col = [int(value) for value in canonical._valid_coords[0]]
    flat_index = row * canonical.Gw + col
    mask = canonical.action_masks()
    assert mask[flat_index]
    assert not np.any(mask.reshape(canonical.Gh, canonical.Gw)[~canonical._valid_cells])


def test_hard_area_cap_masks_cells_that_cannot_be_added_exactly():
    sign = _circle_sign()
    env = TrafficSignGridEnv(
        stop_sign_image=sign,
        stop_sign_uv_image=sign.copy(),
        background_images=[],
        pole_image=None,
        grid_cell_px=16,
        cell_cover_thresh=0.10,
        area_cap_frac=0.05,
        area_cap_mode="hard",
        source_class="speed limit 25",
        detector_instance=FakeDetector(),
    )
    env._episode_cells = np.zeros((env.Gh, env.Gw), dtype=bool)
    mask = env.action_masks()
    for action, (row, col) in enumerate(env._valid_coords):
        expected = env._cell_pixel_areas[row, col] <= (
            env.area_cap_frac * env._sign_pixel_area + 1e-12
        )
        assert bool(mask[action]) is bool(expected)


def test_next_step_query_cost_counts_day_and_triggered_eot_images():
    sign = _circle_sign()
    env = TrafficSignGridEnv(
        stop_sign_image=sign,
        stop_sign_uv_image=sign.copy(),
        background_images=[],
        pole_image=None,
        eval_K=3,
        source_class="speed limit 25",
        detector_instance=FakeDetector(),
    )
    env.reset(seed=1)
    assert env.next_step_detector_query_cost() == 6


def test_impossible_hard_area_cap_fails_before_policy_sampling():
    sign = _circle_sign()
    with pytest.raises(ValueError, match="smaller than every valid"):
        TrafficSignGridEnv(
            stop_sign_image=sign,
            stop_sign_uv_image=sign.copy(),
            background_images=[],
            pole_image=None,
            grid_cell_px=16,
            cell_cover_thresh=0.10,
            area_cap_frac=1e-8,
            area_cap_mode="hard",
            source_class="speed limit 25",
            detector_instance=FakeDetector(),
        )


def test_mismatched_active_asset_is_rejected():
    sign = _circle_sign(64)
    with pytest.raises(ValueError, match="identical pixel dimensions"):
        TrafficSignGridEnv(
            stop_sign_image=sign,
            stop_sign_uv_image=_circle_sign(48),
            background_images=[],
            pole_image=None,
            detector_instance=FakeDetector(),
        )


def test_targeted_environment_requires_designated_target():
    sign = _circle_sign()
    with pytest.raises(ValueError, match="attack_target_class is required"):
        TrafficSignGridEnv(
            stop_sign_image=sign,
            stop_sign_uv_image=sign.copy(),
            background_images=[],
            pole_image=None,
            attack_mode="targeted_misclassification",
            source_class="speed limit 25",
            detector_instance=FakeDetector(),
        )


def test_untargeted_environment_resolves_preregistered_alternative_labels():
    sign = _circle_sign()
    env = TrafficSignGridEnv(
        stop_sign_image=sign,
        stop_sign_uv_image=sign.copy(),
        background_images=[],
        pole_image=None,
        attack_mode="untargeted_misclassification",
        source_class="speed limit 25",
        allowed_alternative_classes="speed limit 55",
        detector_instance=FakeDetector(),
    )
    assert env.allowed_alternative_class_ids == (2,)
    assert env.attack_config.allowed_alternative_class_ids == (2,)

    with pytest.raises(ValueError, match="exclude the source"):
        TrafficSignGridEnv(
            stop_sign_image=sign,
            stop_sign_uv_image=sign.copy(),
            background_images=[],
            pole_image=None,
            attack_mode="untargeted_misclassification",
            source_class="speed limit 25",
            allowed_alternative_classes="speed limit 25",
            detector_instance=FakeDetector(),
        )


def test_reset_seed_reproduces_scene_and_transform_samples():
    sign = _circle_sign()
    env = TrafficSignGridEnv(
        stop_sign_image=sign,
        stop_sign_uv_image=sign.copy(),
        background_images=[Image.new("RGB", (128, 128), "gray")],
        pole_image=None,
        grid_cell_px=16,
        source_class="speed limit 25",
        detector_instance=FakeDetector(),
        img_size=(128, 128),
        eval_K=2,
        obs_size=(64, 64),
        transform_strength=0.0,
    )
    env.reset(seed=123)
    first = (env._place_seed, list(env._transform_seeds), env._bg_index)
    env.reset(seed=999)
    env.reset(seed=123)
    assert (env._place_seed, list(env._transform_seeds), env._bg_index) == first


def test_joint_success_enforces_day_and_area_constraints():
    sign = _circle_sign()
    env = TrafficSignGridEnv(
        stop_sign_image=sign,
        stop_sign_uv_image=sign.copy(),
        background_images=[],
        pole_image=None,
        grid_cell_px=16,
        source_class="speed limit 25",
        attack_mode="targeted_misclassification",
        attack_target_class="speed limit 55",
        area_cap_frac=0.30,
        detector_instance=FakeDetector(),
    )
    row = summarize_detection(
        {"boxes": [[0, 0, 100, 100]], "confs": [0.9], "clss": [2]},
        [0, 0, 100, 100],
        1,
        2,
        env.attack_config,
    )
    metrics = aggregate_metrics([row])
    good = env.joint_success_components(
        metrics,
        clean_detection_rate=1.0,
        day_correct_rate=1.0,
        drop_day=0.0,
        area_frac=0.20,
    )
    bad_day = env.joint_success_components(
        metrics,
        clean_detection_rate=1.0,
        day_correct_rate=0.0,
        drop_day=0.0,
        area_frac=0.20,
    )
    over_budget = env.joint_success_components(
        metrics,
        clean_detection_rate=1.0,
        day_correct_rate=1.0,
        drop_day=0.0,
        area_frac=0.31,
    )
    assert good["attack_success"]
    assert not bad_day["attack_success"]
    assert not over_budget["attack_success"]


def test_detector_failure_propagates_instead_of_becoming_disappearance():
    class FailingDetector(FakeDetector):
        def infer_detections_batch(self, images):
            raise RuntimeError("synthetic detector failure")

    sign = _circle_sign()
    env = TrafficSignGridEnv(
        stop_sign_image=sign,
        stop_sign_uv_image=sign.copy(),
        background_images=[],
        pole_image=None,
        grid_cell_px=16,
        source_class="speed limit 25",
        detector_instance=FailingDetector(),
        transform_strength=0.0,
    )
    with pytest.raises(RuntimeError, match="synthetic detector failure"):
        env.reset(seed=1)
