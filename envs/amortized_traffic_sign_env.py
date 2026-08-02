"""Task-conditioned, support-batched environment for amortized optimization.

One action is applied to every support scene for the active task.  The selected
cell set therefore remains one persistent stencil rather than a collection of
scene-specific patches.  Actions use the base environment's opt-in canonical
full-grid indexing and can only add previously unused cells, making every
recorded prefix inclusion-monotone and directly fabricable.

This module contains no Stable-Baselines dependency.  It exposes a standard
Gymnasium Dict observation suitable for MaskablePPO's MultiInputPolicy.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import re
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from envs.attack_objective import ATTACK_MODES
from envs.robust_objective import SupportRiskSummary, summarize_support_risk
from utils.task_manifest import TaskSpec


FEEDBACK_NAMES = (
    "step_fraction",
    "area_fraction",
    "log_detector_queries",
    "joint_success_rate",
    "clean_eligible_rate",
    "day_preservation_rate",
    "objective_success_rate",
    "mean_source_confidence",
    "mean_attack_target_confidence",
    "mean_target_margin",
    "mean_day_drop",
    "mean_reward",
    "lower_cvar_reward",
    "reward_standard_deviation",
    "failure_dual",
    "day_dual",
    "clean_dual",
    "area_dual",
)


@dataclass(frozen=True)
class AmortizedTask:
    """One task descriptor and independent support-scene replicas."""

    spec: TaskSpec
    support_envs: Tuple[gym.Env, ...]

    def __post_init__(self) -> None:
        if not self.support_envs:
            raise ValueError(f"task {self.spec.task_id!r} requires support_envs")


def _base_env(env: gym.Env) -> Any:
    return getattr(env, "unwrapped", env)


def _fingerprint_vector(value: str, dimensions: int = 8) -> np.ndarray:
    digest = hashlib.sha256(str(value).encode("utf-8")).digest()
    repeated = (digest * ((dimensions + len(digest) - 1) // len(digest)))[:dimensions]
    return np.asarray([(byte / 127.5) - 1.0 for byte in repeated], dtype=np.float32)


def _numeric_sign_value(label: Optional[str]) -> float:
    if not label:
        return -1.0
    match = re.search(r"(?<!\d)(\d{1,3})(?!\d)", str(label))
    if match is None:
        return -1.0
    return float(max(0, min(int(match.group(1)), 160)) / 160.0)


def _mean_info(infos: Sequence[Mapping[str, Any]], key: str) -> float:
    values = []
    for info in infos:
        value = info.get(key, 0.0)
        if isinstance(value, bool):
            values.append(1.0 if value else 0.0)
        else:
            number = float(value)
            if not math.isfinite(number):
                raise RuntimeError(f"support metric {key!r} is non-finite")
            values.append(number)
    return float(sum(values) / len(values)) if values else 0.0


class AmortizedTrafficSignEnv(gym.Env):
    """Sample tasks while applying one irreversible stencil to support scenes.

    The default ``allowed_splits=("train",)`` is intentional.  Development,
    calibration, and certification tasks require an explicit separate wrapper,
    which prevents a training run from silently sampling held-out tasks.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        tasks: Sequence[AmortizedTask],
        *,
        allowed_splits: Sequence[str] = ("train",),
        risk_alpha: float = 0.25,
        required_support_success_rate: float = 0.80,
        required_clean_eligible_rate: float = 0.80,
        required_day_preservation_rate: float = 0.80,
        dual_learning_rate: float = 0.05,
        initial_constraint_duals: Optional[Mapping[str, float]] = None,
        terminate_on_support_success: bool = True,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.tasks = tuple(tasks)
        if not self.tasks:
            raise ValueError("AmortizedTrafficSignEnv requires at least one task")
        self.allowed_splits = tuple(str(value).strip().lower() for value in allowed_splits)
        if not self.allowed_splits:
            raise ValueError("allowed_splits must not be empty")
        disallowed = [
            task.spec.task_id
            for task in self.tasks
            if task.spec.split not in self.allowed_splits
        ]
        if disallowed:
            raise ValueError(
                "task split is not authorized for this wrapper: " + ", ".join(disallowed)
            )

        self.risk_alpha = float(risk_alpha)
        if not 0.0 < self.risk_alpha <= 1.0:
            raise ValueError("risk_alpha must be in (0, 1]")
        self.required_support_success_rate = self._rate(
            required_support_success_rate, "required_support_success_rate"
        )
        self.required_clean_eligible_rate = self._rate(
            required_clean_eligible_rate, "required_clean_eligible_rate"
        )
        self.required_day_preservation_rate = self._rate(
            required_day_preservation_rate, "required_day_preservation_rate"
        )
        self.dual_learning_rate = float(dual_learning_rate)
        if not math.isfinite(self.dual_learning_rate) or self.dual_learning_rate < 0.0:
            raise ValueError("dual_learning_rate must be finite and >= 0")
        self.terminate_on_support_success = bool(terminate_on_support_success)

        reference = _base_env(self.tasks[0].support_envs[0])
        if not isinstance(reference.observation_space, spaces.Box):
            raise TypeError("support image observation spaces must be gymnasium Box spaces")
        reference_shape = tuple(reference.observation_space.shape)
        reference_dtype = np.dtype(reference.observation_space.dtype)
        reference_actions = int(reference.action_space.n)
        reference_grid = (int(reference.Gh), int(reference.Gw))
        for task in self.tasks:
            for env in task.support_envs:
                base = _base_env(env)
                if getattr(base, "action_indexing", None) != "canonical_full_grid":
                    raise ValueError(
                        f"task {task.spec.task_id!r} must use "
                        "action_indexing='canonical_full_grid'"
                    )
                if bool(getattr(base, "terminate_on_success", True)):
                    raise ValueError(
                        f"task {task.spec.task_id!r} support envs must set "
                        "terminate_on_success=False"
                    )
                if (int(base.Gh), int(base.Gw)) != reference_grid:
                    raise ValueError(
                        "all tasks must share one canonical grid shape; got "
                        f"{(base.Gh, base.Gw)} and {reference_grid}"
                    )
                if int(base.action_space.n) != reference_actions:
                    raise ValueError("all tasks must share one canonical action space")
                if tuple(base.observation_space.shape) != reference_shape:
                    raise ValueError("all support image observations must share one shape")
                if np.dtype(base.observation_space.dtype) != reference_dtype:
                    raise ValueError("all support image observations must share one dtype")

        self.Gh, self.Gw = reference_grid
        self.action_space = spaces.Discrete(reference_actions)
        self._task_vector_length = self._make_task_vector(self.tasks[0]).size
        for task in self.tasks[1:]:
            if self._make_task_vector(task).size != self._task_vector_length:
                raise ValueError("all tasks must have equal-length condition_features")
        self.observation_space = spaces.Dict(
            {
                "image": reference.observation_space,
                "task": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(self._task_vector_length,),
                    dtype=np.float32,
                ),
                "feedback": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(len(FEEDBACK_NAMES),),
                    dtype=np.float32,
                ),
            }
        )

        weights = np.asarray([task.spec.weight for task in self.tasks], dtype=np.float64)
        self._task_probabilities = weights / weights.sum()
        self.rng = np.random.default_rng(seed)
        self._active_task: Optional[AmortizedTask] = None
        self._active_observations: list[np.ndarray] = []
        self._previous_query_counts: list[int] = []
        self._episode_queries = 0
        self._lifetime_queries = 0
        self._step = 0
        self._ordered_actions: list[int] = []
        self._prefix_sets: list[frozenset[int]] = []
        self._last_feedback = np.zeros(len(FEEDBACK_NAMES), dtype=np.float32)
        dual_names = ("failure", "day", "clean", "area")
        supplied_duals = dict(initial_constraint_duals or {})
        unknown_duals = sorted(set(supplied_duals) - set(dual_names))
        if unknown_duals:
            raise ValueError(f"unknown initial constraint duals: {unknown_duals}")
        self._duals = {}
        for name in dual_names:
            value = float(supplied_duals.get(name, 0.0))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"initial constraint dual {name!r} must be finite and >= 0")
            self._duals[name] = value
        self._next_task_id: Optional[str] = None

    @staticmethod
    def _rate(value: float, name: str) -> float:
        result = float(value)
        if not math.isfinite(result) or not 0.0 <= result <= 1.0:
            raise ValueError(f"{name} must be finite and in [0, 1]")
        return result

    @property
    def active_task_id(self) -> Optional[str]:
        return self._active_task.spec.task_id if self._active_task else None

    def set_next_task_id(self, task_id: str) -> None:
        """Select exactly one task for the next reset.

        Training normally samples tasks by their preregistered weights.  Prefix
        generation needs deterministic coverage of every development task, so
        it may arm a one-shot task selection without weakening split checks.
        """

        requested = str(task_id).strip()
        self._select_task(requested)
        self._next_task_id = requested

    def _make_task_vector(self, task: AmortizedTask) -> np.ndarray:
        spec = task.spec
        reference = _base_env(task.support_envs[0])
        objective = getattr(reference, "attack_config", None)
        mode = np.zeros(len(ATTACK_MODES), dtype=np.float32)
        mode[ATTACK_MODES.index(spec.attack_mode)] = 1.0
        source_value = _numeric_sign_value(spec.source_class)
        target_value = _numeric_sign_value(spec.target_class)
        semantic = np.asarray(
            [
                source_value,
                target_value,
                (target_value - source_value)
                if source_value >= 0.0 and target_value >= 0.0
                else 0.0,
                1.0 if spec.target_class is not None else 0.0,
            ],
            dtype=np.float32,
        )
        calibration_key = spec.calibration_sha256 or "uncalibrated"
        constraints = np.asarray(
            [
                float(getattr(objective, "source_conf_threshold", 0.0)),
                float(getattr(objective, "target_conf_threshold", 0.0)),
                float(getattr(objective, "min_success_rate", 0.0)),
                float(getattr(reference, "min_clean_detection_rate", 0.0)),
                float(getattr(objective, "localization_iou_threshold", 0.0)),
                float(getattr(reference, "day_tolerance", 0.0)),
                float(getattr(reference, "area_cap_frac", -1.0) or -1.0),
                1.0 if bool(getattr(objective, "require_source_suppression", True)) else 0.0,
                1.0 if bool(getattr(reference, "require_day_preservation", True)) else 0.0,
                float(np.asarray(reference._valid_cells, dtype=bool).mean())
                if hasattr(reference, "_valid_cells")
                else 1.0,
            ],
            dtype=np.float32,
        )
        return np.concatenate(
            [
                mode,
                _fingerprint_vector(spec.source_class),
                _fingerprint_vector(spec.target_class or "none"),
                _fingerprint_vector(spec.detector_id),
                _fingerprint_vector(spec.sign_instance_id),
                _fingerprint_vector(spec.camera_id),
                _fingerprint_vector(spec.material_batch_id),
                _fingerprint_vector(calibration_key),
                semantic,
                constraints,
                np.asarray(spec.condition_features, dtype=np.float32),
            ]
        ).astype(np.float32)

    def _select_task(self, requested_task_id: Optional[str]) -> AmortizedTask:
        if requested_task_id is None:
            index = int(self.rng.choice(len(self.tasks), p=self._task_probabilities))
            return self.tasks[index]
        matches = [task for task in self.tasks if task.spec.task_id == requested_task_id]
        if not matches:
            raise ValueError(
                f"requested task_id {requested_task_id!r} is unavailable in allowed splits"
            )
        return matches[0]

    def _observation(self) -> Dict[str, np.ndarray]:
        if self._active_task is None or not self._active_observations:
            raise RuntimeError("reset() must be called before requesting an observation")
        return {
            "image": np.asarray(self._active_observations[0]).copy(),
            "task": self._make_task_vector(self._active_task).copy(),
            "feedback": self._last_feedback.copy(),
        }

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            self.action_space.seed(seed)
        requested = (options or {}).get("task_id")
        if requested is None:
            requested = self._next_task_id
        self._next_task_id = None
        self._active_task = self._select_task(str(requested) if requested else None)
        child_seeds = self.rng.integers(
            0, 2**31 - 1, size=len(self._active_task.support_envs), dtype=np.int64
        )
        self._active_observations = []
        self._previous_query_counts = []
        self._episode_queries = 0
        self._step = 0
        self._ordered_actions = []
        self._prefix_sets = []
        self._last_feedback = np.zeros(len(FEEDBACK_NAMES), dtype=np.float32)
        for env, child_seed in zip(self._active_task.support_envs, child_seeds):
            observation, _ = env.reset(seed=int(child_seed))
            self._active_observations.append(np.asarray(observation))
            queries = int(getattr(_base_env(env), "_detector_queries", 0))
            self._previous_query_counts.append(queries)
            self._episode_queries += queries
            self._lifetime_queries += queries
        return self._observation(), {
            "task_id": self._active_task.spec.task_id,
            "task_split": self._active_task.spec.split,
            "support_count": len(self._active_task.support_envs),
            "support_detector_queries_episode": self._episode_queries,
            "offline_training_detector_queries_episode": (
                self._episode_queries
                if self._active_task.spec.split == "train"
                else None
            ),
            "physics_calibration_sha256": self._active_task.spec.calibration_sha256,
            "support_physics_transport_seeds": [
                getattr(_base_env(env), "physics_transport_seed", None)
                for env in self._active_task.support_envs
            ],
            "ordered_actions": [],
        }

    def action_masks(self) -> np.ndarray:
        if self._active_task is None:
            return np.ones(int(self.action_space.n), dtype=bool)
        masks = [
            np.asarray(_base_env(env).action_masks(), dtype=bool)
            for env in self._active_task.support_envs
        ]
        return np.logical_and.reduce(masks)

    def next_step_detector_query_cost(self) -> int:
        """Return the exact aggregate detector-image cost of one valid action."""

        if self._active_task is None:
            raise RuntimeError("reset() must be called before querying step cost")
        return sum(
            int(_base_env(env).next_step_detector_query_cost())
            for env in self._active_task.support_envs
        )

    def _update_feedback(
        self,
        infos: Sequence[Mapping[str, Any]],
        rewards: Sequence[float],
        risk: SupportRiskSummary,
    ) -> None:
        reference = _base_env(self._active_task.support_envs[0])
        step_limit = max(1, int(getattr(reference, "steps_per_episode", 1)))
        reward_std = float(np.std(np.asarray(rewards, dtype=np.float64)))
        self._last_feedback = np.asarray(
            [
                self._step / step_limit,
                max(float(info.get("total_area_mask_frac", 0.0)) for info in infos),
                math.log1p(self._episode_queries) / 20.0,
                _mean_info(infos, "attack_success"),
                _mean_info(infos, "clean_eligible"),
                _mean_info(infos, "day_preserved"),
                _mean_info(infos, "objective_success"),
                _mean_info(infos, "mean_source_conf"),
                _mean_info(infos, "mean_attack_target_conf"),
                _mean_info(infos, "mean_target_margin"),
                _mean_info(infos, "drop_day"),
                risk.reward_mean,
                risk.reward_lower_cvar,
                reward_std,
                self._duals["failure"],
                self._duals["day"],
                self._duals["clean"],
                self._duals["area"],
            ],
            dtype=np.float32,
        )

    def _prefix_hash(self) -> str:
        payload = {
            "grid_shape": [self.Gh, self.Gw],
            "ordered_actions": list(self._ordered_actions),
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

    def _exact_selected_area(self) -> Tuple[int, int]:
        if self._active_task is None:
            raise RuntimeError("reset() must be called before measuring area")
        reference = _base_env(self._active_task.support_envs[0])
        cell_areas = np.asarray(reference._cell_pixel_areas, dtype=np.int64)
        selected = np.asarray(reference._episode_cells, dtype=bool)
        sign_pixels = int(reference._sign_pixel_area)
        if cell_areas.shape != selected.shape or sign_pixels <= 0:
            raise RuntimeError("support environment has invalid exact-area metadata")
        selected_pixels = int(cell_areas[selected].sum())
        for env in self._active_task.support_envs[1:]:
            base = _base_env(env)
            peer_areas = np.asarray(base._cell_pixel_areas, dtype=np.int64)
            peer_selected = np.asarray(base._episode_cells, dtype=bool)
            peer_sign_pixels = int(base._sign_pixel_area)
            peer_selected_pixels = int(peer_areas[peer_selected].sum())
            if (
                peer_sign_pixels != sign_pixels
                or peer_selected_pixels != selected_pixels
                or not np.array_equal(peer_areas, cell_areas)
            ):
                raise RuntimeError(
                    "support replicas for one task disagree on exact stencil area"
                )
        return selected_pixels, sign_pixels

    def step(self, action: int):
        if self._active_task is None:
            raise RuntimeError("reset() must be called before step()")
        action_index = int(action)
        mask = self.action_masks()
        if action_index < 0 or action_index >= mask.size or not bool(mask[action_index]):
            info = {
                "task_id": self._active_task.spec.task_id,
                "note": "invalid_or_nonmonotone_action",
                "attack_success": False,
                "ordered_actions": list(self._ordered_actions),
                "prefix_sha256": self._prefix_hash(),
                "offline_training_detector_queries_episode": self._episode_queries,
            }
            return self._observation(), -1.0, False, False, info

        self._step += 1
        observations: list[np.ndarray] = []
        rewards: list[float] = []
        terminated_rows: list[bool] = []
        truncated_rows: list[bool] = []
        infos: list[Mapping[str, Any]] = []
        for index, env in enumerate(self._active_task.support_envs):
            observation, reward, terminated, truncated, info = env.step(action_index)
            observations.append(np.asarray(observation))
            rewards.append(float(reward))
            terminated_rows.append(bool(terminated))
            truncated_rows.append(bool(truncated))
            infos.append(info)
            queries = int(getattr(_base_env(env), "_detector_queries", 0))
            delta = queries - self._previous_query_counts[index]
            if delta < 0:
                raise RuntimeError("support detector query counter moved backwards")
            self._previous_query_counts[index] = queries
            self._episode_queries += delta
            self._lifetime_queries += delta

        reference_cells = np.asarray(
            _base_env(self._active_task.support_envs[0])._episode_cells, dtype=bool
        )
        for env in self._active_task.support_envs[1:]:
            if not np.array_equal(
                reference_cells,
                np.asarray(_base_env(env)._episode_cells, dtype=bool),
            ):
                raise RuntimeError("support scenes diverged from the shared stencil")
        self._ordered_actions.append(action_index)
        current_prefix = frozenset(int(value) for value in np.flatnonzero(reference_cells))
        if self._prefix_sets and not self._prefix_sets[-1] < current_prefix:
            raise RuntimeError("prefix fabrication invariant was violated")
        if len(current_prefix) != len(self._ordered_actions):
            raise RuntimeError("one action must add exactly one canonical grid cell")
        self._prefix_sets.append(current_prefix)
        self._active_observations = observations

        risk = summarize_support_risk(rewards, infos, alpha=self.risk_alpha)
        joint_rate = _mean_info(infos, "attack_success")
        clean_rate = _mean_info(infos, "clean_eligible")
        day_rate = _mean_info(infos, "day_preserved")
        objective_rate = _mean_info(infos, "objective_success")
        maximum_area = max(
            float(info.get("total_area_mask_frac", 0.0)) for info in infos
        )
        reference = _base_env(self._active_task.support_envs[0])
        area_cap = getattr(reference, "area_cap_frac", None)
        violations = {
            "failure": max(0.0, self.required_support_success_rate - joint_rate),
            "clean": max(0.0, self.required_clean_eligible_rate - clean_rate),
            "day": max(0.0, self.required_day_preservation_rate - day_rate),
            "area": max(0.0, maximum_area - float(area_cap))
            if area_cap is not None
            else 0.0,
        }
        constrained_reward = risk.reward_lower_cvar - sum(
            self._duals[name] * value for name, value in violations.items()
        )
        global_success = bool(
            joint_rate >= self.required_support_success_rate
            and clean_rate >= self.required_clean_eligible_rate
            and day_rate >= self.required_day_preservation_rate
            and violations["area"] <= 0.0
        )
        cannot_continue = not bool(np.any(self.action_masks())) or any(
            str(info.get("note", "")) in ("area_cap_exceeded", "no_free_cells")
            for info in infos
        )
        terminated = bool(
            cannot_continue
            or all(terminated_rows)
            or (global_success and self.terminate_on_support_success)
        )
        truncated = bool(all(truncated_rows))
        done = terminated or truncated
        if done and self.dual_learning_rate > 0.0:
            for name, violation in violations.items():
                self._duals[name] = max(
                    0.0,
                    self._duals[name] + self.dual_learning_rate * float(violation),
                )
        self._update_feedback(infos, rewards, risk)
        selected_pixels, sign_pixels = self._exact_selected_area()
        divisor = math.gcd(selected_pixels, sign_pixels)

        info: Dict[str, Any] = {
            "method": "task_amortized_prefix_valid_support_batch",
            "task_id": self._active_task.spec.task_id,
            "task_split": self._active_task.spec.split,
            "source_class": self._active_task.spec.source_class,
            "attack_target_class": self._active_task.spec.target_class,
            "attack_mode": self._active_task.spec.attack_mode,
            "support_count": len(infos),
            "risk": risk.as_dict(),
            "constraint_violations": dict(violations),
            "constraint_duals": dict(self._duals),
            "support_joint_success_rate": joint_rate,
            "support_clean_eligible_rate": clean_rate,
            "support_day_preservation_rate": day_rate,
            "support_objective_success_rate": objective_rate,
            "maximum_area_fraction": maximum_area,
            "selected_pixels": selected_pixels,
            "sign_pixels": sign_pixels,
            "exact_area_fraction": {
                "numerator": selected_pixels // divisor,
                "denominator": sign_pixels // divisor,
                "decimal": selected_pixels / sign_pixels,
            },
            "attack_success": global_success,
            "ordered_actions": list(self._ordered_actions),
            "selected_indices": sorted(current_prefix),
            "prefix_length": len(current_prefix),
            "prefix_sha256": self._prefix_hash(),
            "prefix_valid_monotone_fabrication": True,
            "support_detector_queries_episode": self._episode_queries,
            "support_detector_queries_lifetime": self._lifetime_queries,
            "next_step_detector_query_cost": (
                self.next_step_detector_query_cost()
                if bool(np.any(self.action_masks()))
                else 0
            ),
            "offline_training_detector_queries_episode": (
                self._episode_queries
                if self._active_task.spec.split == "train"
                else None
            ),
            "offline_training_detector_queries_lifetime": (
                self._lifetime_queries
                if self._active_task.spec.split == "train"
                else None
            ),
            "query_accounting_scope": "support_batch_environment_only",
            "physics_calibration_sha256": self._active_task.spec.calibration_sha256,
            "support_physics_transport_seeds": [
                getattr(_base_env(env), "physics_transport_seed", None)
                for env in self._active_task.support_envs
            ],
        }
        return self._observation(), float(constrained_reward), terminated, truncated, info

    def close(self) -> None:
        closed: set[int] = set()
        for task in self.tasks:
            for env in task.support_envs:
                identity = id(env)
                if identity not in closed:
                    env.close()
                    closed.add(identity)


__all__ = [
    "AmortizedTask",
    "AmortizedTrafficSignEnv",
    "FEEDBACK_NAMES",
]
