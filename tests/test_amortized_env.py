import numpy as np
import pytest
import gymnasium as gym
from gymnasium import spaces

from envs.amortized_traffic_sign_env import AmortizedTask, AmortizedTrafficSignEnv
from envs.robust_objective import empirical_cvar, summarize_support_risk
from utils.task_manifest import TaskSpec


class PrefixStubEnv(gym.Env):
    def __init__(self, reward_offset=0.0):
        super().__init__()
        self.action_indexing = "canonical_full_grid"
        self.terminate_on_success = False
        self.Gh = 2
        self.Gw = 2
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(0, 255, shape=(4, 4, 1), dtype=np.uint8)
        self.steps_per_episode = 3
        self.area_cap_frac = 0.75
        self.reward_offset = float(reward_offset)
        self._episode_cells = np.zeros((2, 2), dtype=bool)
        self._cell_pixel_areas = np.ones((2, 2), dtype=np.int64)
        self._sign_pixel_area = 4
        self._detector_queries = 0
        self._step = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._episode_cells[:] = False
        self._detector_queries = 2
        self._step = 0
        return np.zeros((4, 4, 1), dtype=np.uint8), {}

    def action_masks(self):
        valid = np.asarray([True, True, True, False])
        return valid & (~self._episode_cells.reshape(-1))

    def next_step_detector_query_cost(self):
        return 4

    def step(self, action):
        action = int(action)
        self._episode_cells.reshape(-1)[action] = True
        self._detector_queries += 4
        self._step += 1
        count = int(self._episode_cells.sum())
        success = count >= 2
        area = count / 4
        observation = np.full((4, 4, 1), count, dtype=np.uint8)
        info = {
            "attack_success": success,
            "objective_success": success,
            "clean_eligible": True,
            "day_preserved": True,
            "mean_source_conf": 0.1,
            "mean_attack_target_conf": 0.8 if success else 0.2,
            "mean_target_margin": 0.7 if success else 0.1,
            "drop_day": 0.01,
            "total_area_mask_frac": area,
        }
        return observation, float(count + self.reward_offset), False, self._step >= 3, info


def _spec(split="train"):
    return TaskSpec(
        task_id=f"task-{split}",
        split=split,
        sign_instance_id=f"sign-{split}",
        background_split_id=f"background-{split}",
        camera_id=f"camera-{split}",
        material_batch_id=f"material-{split}",
        physical_run_group=f"runs-{split}",
        detector_id="detector-a",
        source_class="speed_limit_25",
        target_class="speed_limit_55",
        attack_mode="targeted_misclassification",
        calibration_sha256="b" * 64,
        condition_features=(25 / 160, 55 / 160),
        weight=1.0,
        environment={},
    )


def _wrapper(split="train"):
    task = AmortizedTask(
        spec=_spec(split),
        support_envs=(PrefixStubEnv(0.0), PrefixStubEnv(-1.0)),
    )
    return AmortizedTrafficSignEnv(
        [task],
        allowed_splits=(split,),
        risk_alpha=0.5,
        required_support_success_rate=1.0,
        required_clean_eligible_rate=1.0,
        required_day_preservation_rate=1.0,
        seed=4,
    )


def test_cvar_uses_fractional_tail_mass_and_fails_on_nonfinite_values():
    assert empirical_cvar([1.0, 2.0, 10.0], alpha=0.5, tail="lower") == pytest.approx(4 / 3)
    assert empirical_cvar([1.0, 2.0, 10.0], alpha=0.5, tail="upper") == pytest.approx(22 / 3)
    with pytest.raises(ValueError, match="non-finite"):
        empirical_cvar([1.0, float("nan")])


def test_support_summary_keeps_every_scene_in_denominator():
    summary = summarize_support_risk(
        [1.0, -1.0],
        [
            {"attack_success": True, "mean_target_margin": 0.5},
            {},
        ],
        alpha=0.5,
    )
    assert summary.support_count == 2
    assert summary.joint_success_rate == pytest.approx(0.5)
    assert summary.target_margin_lower_cvar == pytest.approx(0.0)


def test_one_action_updates_every_support_scene_and_prefixes_are_monotone():
    env = _wrapper()
    observation, reset_info = env.reset(seed=9)
    assert set(observation) == {"image", "task", "feedback"}
    assert reset_info["support_count"] == 2
    assert reset_info["offline_training_detector_queries_episode"] == 4
    assert env.action_masks().tolist() == [True, True, True, False]
    assert env.next_step_detector_query_cost() == 8

    observation, reward, terminated, truncated, first = env.step(0)
    assert not terminated and not truncated
    assert first["ordered_actions"] == [0]
    assert first["selected_indices"] == [0]
    assert first["selected_pixels"] == 1
    assert first["sign_pixels"] == 4
    assert first["exact_area_fraction"] == {
        "numerator": 1,
        "denominator": 4,
        "decimal": 0.25,
    }
    assert first["prefix_valid_monotone_fabrication"] is True
    assert first["offline_training_detector_queries_episode"] == 12
    assert reward == pytest.approx(0.0)

    _, _, terminated, _, second = env.step(1)
    assert terminated
    assert second["attack_success"] is True
    assert second["ordered_actions"] == [0, 1]
    assert second["selected_indices"] == [0, 1]
    assert second["prefix_sha256"] != first["prefix_sha256"]
    for support in env.tasks[0].support_envs:
        assert np.flatnonzero(support._episode_cells).tolist() == [0, 1]


def test_invalid_or_duplicate_action_never_mutates_a_prefix():
    env = _wrapper()
    env.reset(seed=2)
    _, reward, _, _, invalid = env.step(3)
    assert reward == -1.0
    assert invalid["ordered_actions"] == []
    env.step(0)
    _, reward, _, _, duplicate = env.step(0)
    assert reward == -1.0
    assert duplicate["ordered_actions"] == [0]


def test_training_wrapper_rejects_held_out_tasks_by_default():
    task = AmortizedTask(_spec("certification"), (PrefixStubEnv(),))
    with pytest.raises(ValueError, match="not authorized"):
        AmortizedTrafficSignEnv([task])


def test_prefix_generation_can_cover_every_prefix_after_early_success():
    task = AmortizedTask(
        spec=_spec("development"),
        support_envs=(PrefixStubEnv(), PrefixStubEnv()),
    )
    env = AmortizedTrafficSignEnv(
        [task],
        allowed_splits=("development",),
        terminate_on_support_success=False,
        seed=7,
    )
    env.set_next_task_id(task.spec.task_id)
    _, info = env.reset(seed=8)
    assert info["task_id"] == task.spec.task_id
    env.step(0)
    _, _, terminated, truncated, second = env.step(1)
    assert second["attack_success"] is True
    assert not terminated and not truncated
    _, _, _, truncated, third = env.step(2)
    assert truncated
    assert third["ordered_actions"] == [0, 1, 2]


def test_frozen_constraint_duals_can_be_restored_without_online_updates():
    task = AmortizedTask(
        spec=_spec("development"),
        support_envs=(PrefixStubEnv(), PrefixStubEnv()),
    )
    expected = {"failure": 0.4, "day": 0.3, "clean": 0.2, "area": 0.1}
    env = AmortizedTrafficSignEnv(
        [task],
        allowed_splits=("development",),
        initial_constraint_duals=expected,
        dual_learning_rate=0.0,
        terminate_on_support_success=False,
    )
    env.reset(seed=3)
    env.step(0)
    env.step(1)
    _, _, _, truncated, info = env.step(2)
    assert truncated
    assert info["constraint_duals"] == expected
