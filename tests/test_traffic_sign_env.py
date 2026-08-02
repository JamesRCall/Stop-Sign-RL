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
from utils.uv_paint import UVPaint


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


class AlwaysSourceDetector(FakeDetector):
    def infer_detections_batch(self, images):
        rows = []
        for image in images:
            width, height = image.size
            rows.append(
                {
                    "boxes": [[0, 0, width, height]],
                    "confs": [0.9],
                    "clss": [self.target_id],
                }
            )
        return rows


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


def test_joint_palette_actions_encode_cell_and_material_and_mask_whole_cell():
    sign = _circle_sign()
    palette = [
        UVPaint("RedMaterial", "#220000", "#FF0000", translucent=False),
        UVPaint("GreenMaterial", "#002200", "#00FF00", translucent=False),
    ]
    env = TrafficSignGridEnv(
        stop_sign_image=sign,
        stop_sign_uv_image=sign.copy(),
        background_images=[Image.new("RGB", (128, 128), "gray")],
        pole_image=None,
        img_size=(128, 128),
        obs_size=(64, 64),
        grid_cell_px=16,
        cell_cover_thresh=0.10,
        action_indexing="canonical_full_grid",
        paint_action_mode="joint_palette",
        uv_paint_palette=palette,
        source_class="speed limit 25",
        detector_instance=AlwaysSourceDetector(),
        localization_iou_threshold=0.01,
        transform_strength=0.0,
    )
    env.reset(seed=7)

    assert env.action_space.n == env.Gh * env.Gw * len(palette)
    cell_mask = env._valid_cells.reshape(-1)
    action_mask = env.action_masks().reshape(env.Gh * env.Gw, len(palette))
    assert np.all(action_mask[cell_mask])
    assert not np.any(action_mask[~cell_mask])

    flat_cell = int(np.flatnonzero(cell_mask)[0])
    green_action = env.encode_action(flat_cell, 1)
    assert env.decode_action(green_action) == (flat_cell, 1)
    _, _, _, _, info = env.step(green_action)

    row, col = divmod(flat_cell, env.Gw)
    assert env._episode_cells[row, col]
    assert env._episode_paint_ids[row, col] == 1
    assert not np.any(
        env.action_masks().reshape(env.Gh * env.Gw, len(palette))[flat_cell]
    )
    assert info["trace"]["paint_action_mode"] == "joint_palette"
    assert (
        info["trace"]["action_encoding"]
        == "canonical_cell_major_material_minor_v1"
    )
    assert info["trace"]["selected_indices"] == [flat_cell]
    assert info["trace"]["selected_material_indices"] == [1]
    assert info["trace"]["cell_material_assignments"] == [
        {
            "cell_index": flat_cell,
            "material_index": 1,
            "material_name": "GreenMaterial",
        }
    ]


def test_joint_palette_render_uses_each_cells_assigned_day_and_active_color():
    sign = _circle_sign()
    palette = [
        UVPaint("RedMaterial", "#220000", "#FF0000", translucent=False),
        UVPaint("GreenMaterial", "#002200", "#00FF00", translucent=False),
    ]
    env = TrafficSignGridEnv(
        stop_sign_image=sign,
        stop_sign_uv_image=sign.copy(),
        background_images=[],
        pole_image=None,
        grid_cell_px=16,
        cell_cover_thresh=0.10,
        action_indexing="canonical_full_grid",
        paint_action_mode="joint_palette",
        uv_paint_palette=palette,
        source_class="speed limit 25",
        detector_instance=FakeDetector(),
    )
    env._episode_cells = np.zeros((env.Gh, env.Gw), dtype=bool)
    env._episode_paint_ids = np.full((env.Gh, env.Gw), -1, dtype=np.int16)
    selected = [tuple(int(value) for value in row) for row in env._valid_coords[:2]]
    assert len(selected) == 2
    for material_index, (row, col) in enumerate(selected):
        env._episode_cells[row, col] = True
        env._episode_paint_ids[row, col] = material_index

    day = env._render_overlay_pattern(mode="day")
    active = env._render_overlay_pattern(mode="on")
    sign_alpha = np.asarray(sign.getchannel("A"))
    for material_index, (row, col) in enumerate(selected):
        x0, y0, x1, y1 = env._cell_rects[row * env.Gw + col]
        local_y, local_x = np.argwhere(sign_alpha[y0:y1, x0:x1] > 0)[0]
        point = (x0 + int(local_x), y0 + int(local_y))
        assert day.getpixel(point)[:3] == palette[material_index].day_rgb
        assert active.getpixel(point)[:3] == palette[material_index].active_rgb


def test_joint_palette_observation_channel_preserves_material_identity():
    sign = _circle_sign()
    palette = [
        UVPaint("RedMaterial", "#D0D0D0", "#FF0000", translucent=False),
        UVPaint("GreenMaterial", "#D0D0D0", "#00FF00", translucent=False),
    ]
    env = TrafficSignGridEnv(
        stop_sign_image=sign,
        stop_sign_uv_image=sign.copy(),
        background_images=[Image.new("RGB", (128, 128), "gray")],
        pole_image=None,
        img_size=(128, 128),
        obs_size=(64, 64),
        grid_cell_px=16,
        cell_cover_thresh=0.10,
        action_indexing="canonical_full_grid",
        paint_action_mode="joint_palette",
        uv_paint_palette=palette,
        source_class="speed limit 25",
        detector_instance=FakeDetector(),
        transform_strength=0.0,
    )
    env.reset(seed=17)
    env._episode_cells[:] = False
    env._episode_paint_ids[:] = -1
    center = np.asarray([0.5 * (env.Gh - 1), 0.5 * (env.Gw - 1)])
    coordinates = sorted(
        (tuple(int(value) for value in row) for row in env._valid_coords),
        key=lambda row: float(np.linalg.norm(np.asarray(row) - center)),
    )[:2]
    for material_index, (row, col) in enumerate(coordinates):
        env._episode_cells[row, col] = True
        env._episode_paint_ids[row, col] = material_index

    observation = env._render_observation(
        kind="day",
        use_overlay=True,
        transform_seed=env._transform_seeds[0],
    )
    material_values = set(np.unique(observation[..., 3]).tolist())
    assert 0 in material_values
    assert int(round(255 / len(palette))) in material_values
    assert 255 in material_values


def test_joint_palette_configuration_fails_closed_on_ambiguous_materials():
    sign = _circle_sign()
    red = UVPaint("RepeatedMaterial", "#220000", "#FF0000", translucent=False)
    green = UVPaint("RepeatedMaterial", "#002200", "#00FF00", translucent=False)
    with pytest.raises(ValueError, match="paint names must be unique"):
        TrafficSignGridEnv(
            stop_sign_image=sign,
            stop_sign_uv_image=sign.copy(),
            background_images=[],
            pole_image=None,
            paint_action_mode="joint_palette",
            uv_paint_palette=[red, green],
            source_class="speed limit 25",
            detector_instance=FakeDetector(),
        )


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
