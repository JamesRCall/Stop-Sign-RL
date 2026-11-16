"""Compact meta-environment for black-box, prefix-valid patch construction.

The wrapper deliberately discards the base environment's image observation and
reward.  A policy instead sees the material stencil, a short history of actual
black-box responses, and task/goal/budget context.  Every non-STOP action adds
one previously unused cell, additions must remain four-connected, and detector
image queries are accounted from the wrapped environment's cumulative counter.

This module contains no Lagrange multipliers or adaptive dual variables.  Its
reward is a bounded difference of the dense potential defined in
``envs.attack_objective`` with optional, fixed material/query costs.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
from typing import Any, Mapping, Optional, Sequence

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from envs.attack_objective import (
    ATTACK_MODES,
    AggregateAttackMetrics,
    AttackObjectiveConfig,
    DenseAttackPotential,
    TransformAttackMetrics,
    aggregate_metrics,
    dense_attack_potential,
)


RESPONSE_FEATURE_NAMES = (
    "potential",
    "source_suppression",
    "challenger_confidence",
    "challenger_dominance",
    "localization",
    "objective_success",
    "attack_success",
    "query_fraction",
)

BUDGET_FEATURE_NAMES = (
    "material_budget_fraction",
    "query_budget_fraction_of_max",
    "material_used_fraction",
    "material_budget_used_fraction",
    "query_budget_used_fraction",
    "query_budget_remaining_fraction",
    "episode_progress_fraction",
)


@dataclass(frozen=True)
class MetaPatchTask:
    """One black-box optimization task exposed to the meta-policy.

    ``env`` is constructed outside this module.  Context and goal are
    deliberately numeric and explicit; callers should not rely on arbitrary
    string hashes as semantic features.
    When ``goal`` is empty, a stable vector is derived from the base
    environment's :class:`AttackObjectiveConfig`.
    """

    task_id: str
    env: gym.Env
    context: Sequence[float] = ()
    goal: Sequence[float] = ()
    weight: float = 1.0


@dataclass(frozen=True)
class _ResolvedTask:
    task_id: str
    environment: gym.Env
    context: np.ndarray
    goal: np.ndarray
    weight: float


def _base_environment(environment: gym.Env) -> Any:
    return getattr(environment, "unwrapped", environment)


def _finite_vector(values: Sequence[float], *, name: str) -> np.ndarray:
    vector = np.asarray(tuple(values), dtype=np.float32).reshape(-1)
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain only finite numeric values")
    return vector


def _goal_from_config(config: AttackObjectiveConfig) -> np.ndarray:
    mode = np.zeros(len(ATTACK_MODES), dtype=np.float32)
    mode[ATTACK_MODES.index(config.mode)] = 1.0
    return np.concatenate(
        [
            mode,
            np.asarray(
                [
                    config.source_conf_threshold,
                    config.target_conf_threshold,
                    config.min_success_rate,
                    config.localization_iou_threshold,
                    float(config.require_source_suppression),
                ],
                dtype=np.float32,
            ),
        ]
    )


def _objective_config(environment: gym.Env) -> AttackObjectiveConfig:
    config = getattr(_base_environment(environment), "attack_config", None)
    if not isinstance(config, AttackObjectiveConfig):
        raise TypeError(
            "every MetaPatchTask environment must expose an AttackObjectiveConfig "
            "as environment.unwrapped.attack_config"
        )
    return config


def _metrics_from_mapping(value: Mapping[str, Any]) -> AggregateAttackMetrics:
    """Build dense-potential inputs from the public scalar info contract."""

    nested = value.get("attack_metrics")
    if isinstance(nested, AggregateAttackMetrics):
        return nested
    sample_count = value.get(
        "eligible_transform_count",
        value.get("eval_K_used", value.get("total_transform_count", 1)),
    )
    try:
        n = max(1, int(sample_count))
    except (TypeError, ValueError):
        n = 1
    return AggregateAttackMetrics(
        n=n,
        mean_source_conf=float(value.get("mean_source_conf", value.get("c_on", 1.0))),
        mean_source_iou=float(value.get("mean_source_iou", 1.0)),
        mean_top_conf=float(value.get("mean_top_conf", 0.0)),
        mean_top_iou=float(value.get("mean_top_iou", 0.0)),
        mean_alternative_conf=float(value.get("mean_alternative_conf", 0.0)),
        mean_alternative_iou=float(value.get("mean_alternative_iou", 0.0)),
        mean_attack_target_conf=float(value.get("mean_attack_target_conf", 0.0)),
        mean_attack_target_iou=float(value.get("mean_attack_target_iou", 0.0)),
        mean_alternative_margin=float(value.get("mean_alternative_margin", 0.0)),
        mean_target_margin=float(value.get("mean_target_margin", 0.0)),
        disappearance_rate=float(value.get("disappearance_success_rate", 0.0)),
        untargeted_rate=float(value.get("misclassification_success_rate", 0.0)),
        targeted_rate=float(value.get("targeted_success_rate", 0.0)),
    )


def _clean_metrics(environment: gym.Env, reset_info: Mapping[str, Any]) -> AggregateAttackMetrics:
    nested = reset_info.get("attack_metrics")
    if isinstance(nested, AggregateAttackMetrics):
        return nested
    base = _base_environment(environment)
    rows = getattr(base, "_baseline_on_metrics", None)
    if rows and all(isinstance(row, TransformAttackMetrics) for row in rows):
        return aggregate_metrics(rows)
    if any(
        name in reset_info
        for name in (
            "mean_source_conf",
            "c_on",
            "mean_attack_target_conf",
            "mean_alternative_conf",
        )
    ):
        return _metrics_from_mapping(reset_info)
    # A clean, confidently localized source gives potential zero in all modes.
    return AggregateAttackMetrics(n=1, mean_source_conf=1.0, mean_source_iou=1.0)


class MetaPatchEnv(gym.Env):
    """Meta-RL wrapper over compatible cell/material patch environments.

    Required base-environment contract:

    - a discrete action space with cell/material tokens;
    - ``Gh``, ``Gw``, ``paint_action_count`` and ``attack_config`` attributes;
    - ``decode_action(token) -> (cell_action, material_index)`` or compatible
      cell-major/material-minor action ordering;
    - ``action_masks()`` and ``next_step_detector_query_cost()`` methods;
    - a cumulative ``_detector_queries`` counter or cumulative
      ``info['detector_queries']`` field.

    The appended STOP token is always the last action.  It never calls the base
    environment and therefore consumes no detector query.  Observations contain
    ``grid`` with shape ``(materials + 1, Gh, Gw)``, ``history`` with shape
    ``(history_length, 8)``, and one flat ``context`` vector containing the
    caller-supplied task context, attack goal, and seven budget/progress fields.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        tasks: Sequence[MetaPatchTask | gym.Env],
        material_budgets: Sequence[float] = (0.20,),
        query_budgets: Sequence[int] = (1_000,),
        history_length: int = 8,
        *,
        max_steps: Optional[int] = None,
        material_cost_weight: float = 0.0,
        query_cost_weight: float = 0.0,
        terminate_on_success: bool = True,
    ) -> None:
        super().__init__()
        if not tasks:
            raise ValueError("MetaPatchEnv requires at least one task")
        self.history_length = int(history_length)
        if self.history_length < 1:
            raise ValueError("history_length must be >= 1")

        self.material_budgets = tuple(float(value) for value in material_budgets)
        if not self.material_budgets or any(
            not math.isfinite(value) or not 0.0 < value <= 1.0
            for value in self.material_budgets
        ):
            raise ValueError("material_budgets must contain fractions in (0, 1]")
        normalized_query_budgets: list[int] = []
        for value in query_budgets:
            if isinstance(value, bool) or int(value) != value or int(value) < 1:
                raise ValueError("query_budgets must contain positive integers")
            normalized_query_budgets.append(int(value))
        if not normalized_query_budgets:
            raise ValueError("query_budgets must not be empty")
        self.query_budgets = tuple(normalized_query_budgets)
        self._maximum_query_budget = max(self.query_budgets)

        self.material_cost_weight = float(material_cost_weight)
        self.query_cost_weight = float(query_cost_weight)
        if not math.isfinite(self.material_cost_weight) or self.material_cost_weight < 0.0:
            raise ValueError("material_cost_weight must be finite and >= 0")
        if not math.isfinite(self.query_cost_weight) or self.query_cost_weight < 0.0:
            raise ValueError("query_cost_weight must be finite and >= 0")
        self.terminate_on_success = bool(terminate_on_success)

        resolved: list[_ResolvedTask] = []
        for index, supplied in enumerate(tasks):
            task = (
                supplied
                if isinstance(supplied, MetaPatchTask)
                else MetaPatchTask(task_id=f"task_{index}", env=supplied)
            )
            if not str(task.task_id).strip():
                raise ValueError("task_id must be non-empty")
            environment = task.env
            if not isinstance(environment, gym.Env):
                raise TypeError(f"task {task.task_id!r} did not produce a Gym environment")
            config = _objective_config(environment)
            context = _finite_vector(task.context, name=f"task {task.task_id} context")
            goal = _finite_vector(task.goal, name=f"task {task.task_id} goal")
            if goal.size == 0:
                goal = _goal_from_config(config)
            weight = float(task.weight)
            if not math.isfinite(weight) or weight <= 0.0:
                raise ValueError(f"task {task.task_id!r} weight must be finite and > 0")
            resolved.append(
                _ResolvedTask(str(task.task_id), environment, context, goal, weight)
            )

        context_sizes = {task.context.size for task in resolved}
        goal_sizes = {task.goal.size for task in resolved}
        if len(context_sizes) != 1 or len(goal_sizes) != 1:
            raise ValueError("all tasks must use equal-length context and goal vectors")
        task_ids = [task.task_id for task in resolved]
        if len(task_ids) != len(set(task_ids)):
            raise ValueError("task_id values must be unique")
        self.tasks = tuple(resolved)
        weights = np.asarray([task.weight for task in self.tasks], dtype=np.float64)
        self._task_probabilities = weights / weights.sum()

        reference = _base_environment(self.tasks[0].environment)
        self.Gh, self.Gw = int(reference.Gh), int(reference.Gw)
        self.material_count = int(getattr(reference, "paint_action_count", 1))
        self.base_action_count = int(self.tasks[0].environment.action_space.n)
        if self.Gh < 1 or self.Gw < 1 or self.material_count < 1:
            raise ValueError("base grid and material count must be positive")
        for task in self.tasks:
            base = _base_environment(task.environment)
            signature = (
                int(base.Gh),
                int(base.Gw),
                int(getattr(base, "paint_action_count", 1)),
                int(task.environment.action_space.n),
            )
            expected = (self.Gh, self.Gw, self.material_count, self.base_action_count)
            if signature != expected:
                raise ValueError(
                    "all tasks must share grid, material, and action-space semantics; "
                    f"expected {expected}, got {signature} for {task.task_id!r}"
                )
            if not callable(getattr(base, "action_masks", None)):
                raise TypeError(f"task {task.task_id!r} lacks action_masks()")
            if not callable(getattr(base, "next_step_detector_query_cost", None)):
                raise TypeError(
                    f"task {task.task_id!r} lacks next_step_detector_query_cost()"
                )

        self.stop_action = self.base_action_count
        self.action_space = spaces.Discrete(self.base_action_count + 1)
        self.max_steps = int(max_steps or (self.Gh * self.Gw))
        if self.max_steps < 1:
            raise ValueError("max_steps must be >= 1")
        context_length = (
            next(iter(context_sizes))
            + next(iter(goal_sizes))
            + len(BUDGET_FEATURE_NAMES)
        )
        self.observation_space = spaces.Dict(
            {
                "grid": spaces.Box(
                    0.0,
                    1.0,
                    shape=(self.material_count + 1, self.Gh, self.Gw),
                    dtype=np.float32,
                ),
                "history": spaces.Box(
                    0.0,
                    1.0,
                    shape=(self.history_length, len(RESPONSE_FEATURE_NAMES)),
                    dtype=np.float32,
                ),
                "context": spaces.Box(
                    -np.inf,
                    np.inf,
                    shape=(context_length,),
                    dtype=np.float32,
                ),
            }
        )

        self._active_task: Optional[_ResolvedTask] = None
        self._material_budget = self.material_budgets[0]
        self._query_budget = self.query_budgets[0]
        self._materials = np.full((self.Gh, self.Gw), -1, dtype=np.int16)
        self._cell_area_fractions = np.zeros((self.Gh, self.Gw), dtype=np.float64)
        self._material_used = 0.0
        self._queries_used = 0
        self._lifetime_queries = 0
        self._step_count = 0
        self._initial_potential: Optional[DenseAttackPotential] = None
        self._previous_potential: Optional[DenseAttackPotential] = None
        self._history: deque[np.ndarray] = deque(maxlen=self.history_length)
        self._last_base_info: Mapping[str, Any] = {}

    @property
    def active_task_id(self) -> Optional[str]:
        return self._active_task.task_id if self._active_task is not None else None

    @property
    def detector_queries_used(self) -> int:
        return int(self._queries_used)

    @property
    def lifetime_detector_queries(self) -> int:
        """Exact detector-image queries accumulated across all resets/tasks."""

        return int(self._lifetime_queries)

    @property
    def material_used_fraction(self) -> float:
        return float(self._material_used)

    def _active_base(self) -> Any:
        if self._active_task is None:
            raise RuntimeError("reset() must be called before using MetaPatchEnv")
        return _base_environment(self._active_task.environment)

    def _query_count(self, info: Optional[Mapping[str, Any]] = None) -> int:
        base = self._active_base()
        value = getattr(base, "_detector_queries", None)
        if value is None and info is not None:
            value = info.get("detector_queries")
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise RuntimeError(
                "base environment must expose an exact cumulative integer detector-query counter"
            )
        if int(value) < 0:
            raise RuntimeError("base detector-query counter must be nonnegative")
        return int(value)

    def _resolve_cell_areas(self) -> np.ndarray:
        base = self._active_base()
        raw = getattr(base, "_cell_pixel_areas", None)
        denominator = getattr(base, "_sign_pixel_area", None)
        if raw is not None and denominator is not None and int(denominator) > 0:
            values = np.asarray(raw, dtype=np.float64)
            if values.shape != (self.Gh, self.Gw):
                raise RuntimeError("base cell-area metadata has the wrong grid shape")
            return values / float(denominator)
        valid = np.asarray(getattr(base, "_valid_cells", np.ones((self.Gh, self.Gw))), dtype=bool)
        if valid.shape != (self.Gh, self.Gw) or not np.any(valid):
            raise RuntimeError("base valid-cell metadata has the wrong grid shape")
        return valid.astype(np.float64) / float(valid.sum())

    def _decode_action(self, action: int) -> tuple[int, int, int, int]:
        base = self._active_base()
        decoder = getattr(base, "decode_action", None)
        if callable(decoder):
            cell_action, material_index = decoder(int(action))
        else:
            cell_action, material_index = divmod(int(action), self.material_count)
        cell_action, material_index = int(cell_action), int(material_index)
        indexing = str(getattr(base, "action_indexing", "canonical_full_grid"))
        if indexing == "valid_cells":
            coordinates = np.asarray(getattr(base, "_valid_coords", ()))
            if cell_action < 0 or cell_action >= len(coordinates):
                raise ValueError("decoded compact cell action is out of range")
            row, column = (int(value) for value in coordinates[cell_action])
        else:
            row, column = divmod(cell_action, self.Gw)
        if not (0 <= row < self.Gh and 0 <= column < self.Gw):
            raise ValueError("decoded action maps outside the canonical grid")
        if not 0 <= material_index < self.material_count:
            raise ValueError("decoded action has an invalid material index")
        return cell_action, material_index, row, column

    def _is_connected_candidate(self, row: int, column: int) -> bool:
        selected = self._materials >= 0
        if not bool(np.any(selected)):
            return True
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            rr, cc = row + dr, column + dc
            if 0 <= rr < self.Gh and 0 <= cc < self.Gw and selected[rr, cc]:
                return True
        return False

    def action_masks(self) -> np.ndarray:
        mask = np.zeros(self.base_action_count + 1, dtype=bool)
        mask[self.stop_action] = True
        if self._active_task is None:
            return mask
        base = self._active_base()
        base_mask = np.asarray(base.action_masks(), dtype=bool).reshape(-1)
        if base_mask.size != self.base_action_count:
            raise RuntimeError("base action mask has the wrong size")
        predicted_queries = int(base.next_step_detector_query_cost())
        if predicted_queries < 0:
            raise RuntimeError("next-step detector-query cost must be nonnegative")
        if self._queries_used + predicted_queries > self._query_budget:
            return mask
        for action in np.flatnonzero(base_mask):
            _, _, row, column = self._decode_action(int(action))
            if self._materials[row, column] >= 0:
                continue
            if not self._is_connected_candidate(row, column):
                continue
            next_material = self._material_used + self._cell_area_fractions[row, column]
            if next_material <= self._material_budget + 1e-12:
                mask[int(action)] = True
        return mask

    def _history_row(
        self,
        potential: DenseAttackPotential,
        info: Mapping[str, Any],
        query_delta: int,
    ) -> np.ndarray:
        return np.asarray(
            [
                potential.potential,
                potential.source_suppression,
                potential.challenger_confidence,
                potential.challenger_dominance,
                potential.localization,
                float(bool(info.get("objective_success", False))),
                float(bool(info.get("attack_success", False))),
                min(1.0, float(query_delta) / float(self._query_budget)),
            ],
            dtype=np.float32,
        )

    def _observation(self) -> dict[str, np.ndarray]:
        if self._active_task is None:
            raise RuntimeError("reset() must be called before requesting an observation")
        material_grid = np.zeros(
            (self.material_count + 1, self.Gh, self.Gw), dtype=np.float32
        )
        material_grid[0] = 1.0
        for row, column in np.argwhere(self._materials >= 0):
            material = int(self._materials[row, column])
            material_grid[0, row, column] = 0.0
            material_grid[material + 1, row, column] = 1.0

        history = np.zeros(
            (self.history_length, len(RESPONSE_FEATURE_NAMES)), dtype=np.float32
        )
        if self._history:
            values = np.stack(tuple(self._history), axis=0)
            history[-len(values) :] = values
        material_budget_used = min(
            1.0, self._material_used / max(self._material_budget, 1e-12)
        )
        query_budget_used = min(1.0, self._queries_used / self._query_budget)
        budget = np.asarray(
            [
                self._material_budget,
                min(1.0, self._query_budget / self._maximum_query_budget),
                self._material_used,
                material_budget_used,
                query_budget_used,
                max(0.0, 1.0 - query_budget_used),
                min(1.0, self._step_count / self.max_steps),
            ],
            dtype=np.float32,
        )
        context = np.concatenate(
            [self._active_task.context, self._active_task.goal, budget]
        ).astype(np.float32)
        return {
            "grid": material_grid,
            "history": history,
            "context": context,
        }

    def _wrapper_info(
        self,
        base_info: Mapping[str, Any],
        *,
        query_delta: int,
        stopped: bool = False,
    ) -> dict[str, Any]:
        potential = self._previous_potential
        initial = self._initial_potential
        if potential is None or initial is None:
            raise RuntimeError("reset() must establish the dense-potential baseline")
        dense_progress = float(potential.potential - initial.potential)
        return {
            "task_id": self.active_task_id,
            "stopped": bool(stopped),
            "detector_image_queries_delta": int(query_delta),
            "detector_image_queries_used": int(self._queries_used),
            "detector_image_query_budget": int(self._query_budget),
            "material_used_fraction": float(self._material_used),
            "material_budget_fraction": float(self._material_budget),
            # Public scalar callback contract.  Progress always uses this
            # episode's clean reset state, never the preceding edit or a
            # potential from another task/episode.
            "initial_dense_potential": float(initial.potential),
            "dense_potential": float(potential.potential),
            "dense_progress": dense_progress,
            "potential": potential.as_dict(),
            "attack_success": bool(base_info.get("attack_success", False)),
            "objective_success": bool(base_info.get("objective_success", False)),
            "base_info": dict(base_info),
        }

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        super().reset(seed=seed)
        requested = dict(options or {})
        task_id = requested.get("task_id")
        if task_id is None:
            index = int(self.np_random.choice(len(self.tasks), p=self._task_probabilities))
            self._active_task = self.tasks[index]
        else:
            matches = [task for task in self.tasks if task.task_id == str(task_id)]
            if not matches:
                raise ValueError(f"unknown task_id {task_id!r}")
            self._active_task = matches[0]

        material_budget = requested.get(
            "material_budget", self.np_random.choice(self.material_budgets)
        )
        query_budget = requested.get("query_budget", self.np_random.choice(self.query_budgets))
        self._material_budget = float(material_budget)
        self._query_budget = int(query_budget)
        if not math.isfinite(self._material_budget) or not 0.0 < self._material_budget <= 1.0:
            raise ValueError("reset material_budget must be in (0, 1]")
        if self._query_budget < 1:
            raise ValueError("reset query_budget must be a positive integer")

        self._materials.fill(-1)
        self._material_used = 0.0
        self._step_count = 0
        self._history.clear()
        child_seed = int(self.np_random.integers(0, 2**31 - 1))
        _, base_info = self._active_task.environment.reset(seed=child_seed)
        self._last_base_info = dict(base_info or {})
        self._cell_area_fractions = self._resolve_cell_areas()
        self._queries_used = self._query_count(self._last_base_info)
        self._lifetime_queries += self._queries_used
        if self._queries_used > self._query_budget:
            raise ValueError(
                "query budget is smaller than the base environment's reset-query cost"
            )
        config = _objective_config(self._active_task.environment)
        self._initial_potential = dense_attack_potential(
            _clean_metrics(self._active_task.environment, self._last_base_info), config
        )
        self._previous_potential = self._initial_potential
        self._history.append(
            self._history_row(
                self._previous_potential, self._last_base_info, self._queries_used
            )
        )
        info = self._wrapper_info(
            self._last_base_info, query_delta=self._queries_used, stopped=False
        )
        info["reset_detector_image_queries"] = int(self._queries_used)
        return self._observation(), info

    def step(self, action: int):
        if (
            self._active_task is None
            or self._initial_potential is None
            or self._previous_potential is None
        ):
            raise RuntimeError("reset() must be called before step()")
        action_index = int(action)
        if action_index == self.stop_action:
            info = self._wrapper_info(self._last_base_info, query_delta=0, stopped=True)
            return self._observation(), 0.0, True, False, info
        mask = self.action_masks()
        if action_index < 0 or action_index >= self.base_action_count or not mask[action_index]:
            info = self._wrapper_info(self._last_base_info, query_delta=0, stopped=False)
            info["invalid_action"] = True
            return self._observation(), -1.0, False, False, info

        _, material_index, row, column = self._decode_action(action_index)
        base = self._active_base()
        predicted_queries = int(base.next_step_detector_query_cost())
        before = self._query_count(self._last_base_info)
        _, base_reward, base_terminated, base_truncated, base_info = (
            self._active_task.environment.step(action_index)
        )
        del base_reward  # The wrapper intentionally defines the complete PPO reward.
        self._last_base_info = dict(base_info or {})
        after = self._query_count(self._last_base_info)
        query_delta = after - before
        if query_delta < 0:
            raise RuntimeError("base detector-query counter moved backwards")
        if query_delta != predicted_queries:
            raise RuntimeError(
                "base next-step query-cost contract was inaccurate: "
                f"predicted {predicted_queries}, observed {query_delta}"
            )
        self._queries_used += query_delta
        self._lifetime_queries += query_delta
        if self._queries_used > self._query_budget:
            raise RuntimeError("base environment exceeded the declared query budget")

        self._materials[row, column] = material_index
        added_material = float(self._cell_area_fractions[row, column])
        self._material_used += added_material
        self._step_count += 1

        config = _objective_config(self._active_task.environment)
        current = dense_attack_potential(
            _metrics_from_mapping(self._last_base_info), config
        )
        potential_delta = float(current.potential - self._previous_potential.potential)
        material_cost = added_material / max(self._material_budget, 1e-12)
        query_cost = query_delta / float(self._query_budget)
        reward = float(
            np.clip(
                potential_delta
                - self.material_cost_weight * material_cost
                - self.query_cost_weight * query_cost,
                -1.0,
                1.0,
            )
        )
        self._previous_potential = current
        self._history.append(
            self._history_row(current, self._last_base_info, query_delta)
        )

        success = bool(self._last_base_info.get("attack_success", False))
        terminated = bool(base_terminated or (success and self.terminate_on_success))
        budget_exhausted = self._queries_used >= self._query_budget
        step_limit = self._step_count >= self.max_steps
        truncated = bool(base_truncated or ((budget_exhausted or step_limit) and not terminated))
        info = self._wrapper_info(
            self._last_base_info, query_delta=query_delta, stopped=False
        )
        info.update(
            {
                "potential_delta": potential_delta,
                "material_cost": material_cost,
                "query_cost": query_cost,
                "query_budget_exhausted": bool(budget_exhausted),
                "step_limit_reached": bool(step_limit),
            }
        )
        return self._observation(), reward, terminated, truncated, info

    def close(self) -> None:
        closed: set[int] = set()
        for task in self.tasks:
            identity = id(task.environment)
            if identity not in closed:
                task.environment.close()
                closed.add(identity)


__all__ = [
    "BUDGET_FEATURE_NAMES",
    "MetaPatchEnv",
    "MetaPatchTask",
    "RESPONSE_FEATURE_NAMES",
]
