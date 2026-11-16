"""Frozen, query-auditable evaluation for compact PatchRL policies."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Optional, Protocol, Sequence

import numpy as np

from patchrl.meta_env import MetaPatchEnv


class MaskedPredictor(Protocol):
    """Inference-only interface implemented by ``sb3_contrib.MaskablePPO``."""

    def predict(
        self,
        observation: Mapping[str, np.ndarray],
        *,
        action_masks: np.ndarray,
        deterministic: bool = True,
    ) -> Any: ...


@dataclass(frozen=True)
class EvaluationEpisode:
    """One preregistered task/seed/budget evaluation unit."""

    task_id: str
    seed: int
    material_budget: float
    query_budget: int

    def validated(self) -> "EvaluationEpisode":
        task_id = str(self.task_id).strip()
        if not task_id:
            raise ValueError("evaluation task_id must be non-empty")
        if isinstance(self.seed, bool) or not isinstance(self.seed, (int, np.integer)):
            raise ValueError("evaluation seed must be an integer")
        seed = int(self.seed)
        if seed < 0:
            raise ValueError("evaluation seed must be nonnegative")
        material = float(self.material_budget)
        if not math.isfinite(material) or not 0.0 < material <= 1.0:
            raise ValueError("evaluation material_budget must be in (0, 1]")
        if (
            isinstance(self.query_budget, bool)
            or not isinstance(self.query_budget, (int, np.integer))
            or int(self.query_budget) < 1
        ):
            raise ValueError("evaluation query_budget must be a positive integer")
        return EvaluationEpisode(task_id, seed, material, int(self.query_budget))

    def as_dict(self) -> dict[str, Any]:
        return {
            "task_id": str(self.task_id),
            "seed": int(self.seed),
            "material_budget": float(self.material_budget),
            "query_budget": int(self.query_budget),
        }


def wilson_interval(
    successes: int,
    total: int,
    *,
    z: float = 1.959963984540054,
) -> tuple[float, float]:
    """Return the two-sided Wilson score interval for a binomial proportion."""

    if isinstance(successes, bool) or isinstance(total, bool):
        raise ValueError("successes and total must be integers")
    if int(successes) != successes or int(total) != total:
        raise ValueError("successes and total must be integers")
    successes, total = int(successes), int(total)
    if total < 1 or successes < 0 or successes > total:
        raise ValueError("require 0 <= successes <= total and total >= 1")
    if not math.isfinite(float(z)) or float(z) <= 0.0:
        raise ValueError("z must be finite and positive")
    proportion = successes / total
    z2 = float(z) ** 2
    denominator = 1.0 + z2 / total
    center = (proportion + z2 / (2.0 * total)) / denominator
    radius = (
        float(z)
        * math.sqrt(
            proportion * (1.0 - proportion) / total
            + z2 / (4.0 * total * total)
        )
        / denominator
    )
    lower = 0.0 if successes == 0 else max(0.0, center - radius)
    upper = 1.0 if successes == total else min(1.0, center + radius)
    return lower, upper


def atomic_write_json(path: str | Path, payload: Mapping[str, Any]) -> Path:
    """Strictly serialize and atomically replace one JSON artifact.

    The temporary file is created in the destination directory, flushed, and
    replaced only after strict serialization succeeds.  NaN and infinity are
    rejected and a failed write leaves neither a partial target nor temp file.
    """

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(
        dict(payload),
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
    ) + "\n"
    temporary_name: Optional[str] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
            delete=False,
        ) as handle:
            temporary_name = handle.name
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
        temporary_name = None
    finally:
        if temporary_name is not None:
            try:
                Path(temporary_name).unlink()
            except FileNotFoundError:
                pass
    return destination


def _integer_action(prediction: Any) -> int:
    # Stable-Baselines returns ``(action, state)``; simple test/frozen policies
    # often return the action directly.  Support both without importing SB3.
    candidate = prediction[0] if isinstance(prediction, tuple) else prediction
    values = np.asarray(candidate)
    if values.size != 1:
        raise RuntimeError(
            "predictor must return exactly one action for a non-vectorized evaluator"
        )
    scalar = values.reshape(-1)[0]
    if isinstance(scalar, (bool, np.bool_)):
        raise RuntimeError("predictor returned a boolean action")
    try:
        action = int(scalar)
    except (TypeError, ValueError, OverflowError) as exc:
        raise RuntimeError("predictor returned a non-integer action") from exc
    try:
        numeric = float(scalar)
    except (TypeError, ValueError, OverflowError) as exc:
        raise RuntimeError("predictor returned a non-numeric action") from exc
    if not math.isfinite(numeric) or numeric != float(action):
        raise RuntimeError("predictor returned a non-integer action")
    return action


def _potential(info: Mapping[str, Any]) -> tuple[float, dict[str, Any]]:
    components = info.get("potential")
    if components is None:
        return 0.0, {}
    if not isinstance(components, Mapping):
        raise RuntimeError("environment potential record must be an object")
    value = float(components.get("potential", 0.0))
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise RuntimeError("environment potential must be finite and in [0, 1]")
    normalized: dict[str, Any] = {}
    for key, item in components.items():
        if key == "mode":
            normalized[str(key)] = str(item)
            continue
        numeric = float(item)
        if not math.isfinite(numeric):
            raise RuntimeError("environment potential components must be finite")
        normalized[str(key)] = numeric
    return value, normalized


def _termination_reason(
    *,
    terminated: bool,
    truncated: bool,
    step_cap_reached: bool,
    info: Mapping[str, Any],
) -> str:
    if bool(info.get("stopped", False)):
        return "policy_stop"
    if terminated and bool(info.get("attack_success", False)):
        return "attack_success"
    if terminated:
        return "environment_terminated"
    if bool(info.get("query_budget_exhausted", False)):
        return "query_budget_exhausted"
    if step_cap_reached:
        return "evaluator_step_cap"
    if truncated:
        return "environment_truncated"
    raise RuntimeError("episode ended without a termination reason")


def _summarize(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("cannot summarize zero evaluation episodes")
    successes = sum(bool(row["success"]) for row in rows)
    lower, upper = wilson_interval(successes, len(rows))
    total_queries = sum(int(row["detector_image_queries"]) for row in rows)
    by_task: dict[str, Any] = {}
    for task_id in sorted({str(row["task_id"]) for row in rows}):
        task_rows = [row for row in rows if str(row["task_id"]) == task_id]
        task_successes = sum(bool(row["success"]) for row in task_rows)
        task_lower, task_upper = wilson_interval(task_successes, len(task_rows))
        task_queries = sum(int(row["detector_image_queries"]) for row in task_rows)
        by_task[task_id] = {
            "episodes": len(task_rows),
            "successes": int(task_successes),
            "success_rate": float(task_successes / len(task_rows)),
            "wilson_95_ci": {"lower": task_lower, "upper": task_upper},
            "detector_image_queries": int(task_queries),
        }
    return {
        "episodes": len(rows),
        "successes": int(successes),
        "success_rate": float(successes / len(rows)),
        "wilson_95_ci": {"lower": lower, "upper": upper},
        "total_detector_image_queries": int(total_queries),
        "mean_detector_image_queries_per_episode": float(total_queries / len(rows)),
        "by_task": by_task,
    }


def evaluate_policy(
    predictor: MaskedPredictor,
    environment: MetaPatchEnv,
    episodes: Sequence[EvaluationEpisode],
    *,
    max_steps: int,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Evaluate a frozen masked policy without training or adapting it.

    Each episode supplies its own task ID, RNG seed, material budget, and image-
    query budget.  The evaluator passes the current action mask to every
    deterministic prediction, rejects a masked action before it can query the
    detector, and reconciles episode query counts with the environment's
    lifetime counter.
    """

    if not callable(getattr(predictor, "predict", None)):
        raise TypeError("predictor must expose predict(...)")
    if not callable(getattr(environment, "action_masks", None)):
        raise TypeError("environment must expose action_masks()")
    if isinstance(max_steps, bool) or not isinstance(max_steps, (int, np.integer)):
        raise ValueError("max_steps must be a positive integer")
    max_steps = int(max_steps)
    if max_steps < 1:
        raise ValueError("max_steps must be a positive integer")
    plan = tuple(episode.validated() for episode in episodes)
    if not plan:
        raise ValueError("evaluation requires at least one episode")
    identities = [(item.task_id, item.seed) for item in plan]
    if len(identities) != len(set(identities)):
        raise ValueError("evaluation task_id/seed pairs must be unique")

    lifetime_before_evaluation = int(environment.lifetime_detector_queries)
    rows: list[dict[str, Any]] = []
    for episode_index, specification in enumerate(plan):
        lifetime_before_episode = int(environment.lifetime_detector_queries)
        observation, reset_info = environment.reset(
            seed=specification.seed,
            options={
                "task_id": specification.task_id,
                "material_budget": specification.material_budget,
                "query_budget": specification.query_budget,
            },
        )
        if str(reset_info.get("task_id", "")) != specification.task_id:
            raise RuntimeError("environment reset selected the wrong evaluation task")
        final_info: Mapping[str, Any] = reset_info
        initial_potential, _ = _potential(reset_info)
        potential_trace = [initial_potential]
        actions: list[int] = []
        terminated = False
        truncated = False

        while not (terminated or truncated) and len(actions) < max_steps:
            mask = np.asarray(environment.action_masks(), dtype=bool).reshape(-1)
            if mask.size != int(environment.action_space.n):
                raise RuntimeError("environment returned an action mask with the wrong size")
            if not bool(np.any(mask)):
                raise RuntimeError("environment action mask contains no feasible action")
            prediction = predictor.predict(
                observation,
                action_masks=mask.copy(),
                deterministic=True,
            )
            action = _integer_action(prediction)
            if action < 0 or action >= mask.size or not bool(mask[action]):
                raise RuntimeError(
                    f"predictor selected masked or out-of-range action {action}"
                )
            observation, _, terminated, truncated, step_info = environment.step(action)
            actions.append(action)
            final_info = step_info
            current_potential, _ = _potential(step_info)
            potential_trace.append(current_potential)

        step_cap_reached = bool(
            not terminated and not truncated and len(actions) >= max_steps
        )
        if not (terminated or truncated or step_cap_reached):
            raise RuntimeError("evaluation loop exited without completing the episode")
        episode_queries = int(environment.detector_queries_used)
        lifetime_delta = int(environment.lifetime_detector_queries) - lifetime_before_episode
        if episode_queries != lifetime_delta:
            raise RuntimeError(
                "episode detector-query count disagrees with lifetime-query delta: "
                f"episode={episode_queries}, lifetime_delta={lifetime_delta}"
            )
        material_used = float(environment.material_used_fraction)
        if episode_queries > specification.query_budget:
            raise RuntimeError("evaluation exceeded its explicit detector-query budget")
        if material_used > specification.material_budget + 1e-12:
            raise RuntimeError("evaluation exceeded its explicit material budget")
        final_potential, potential_components = _potential(final_info)
        reason = _termination_reason(
            terminated=bool(terminated),
            truncated=bool(truncated),
            step_cap_reached=step_cap_reached,
            info=final_info,
        )
        rows.append(
            {
                "episode_index": int(episode_index),
                "task_id": specification.task_id,
                "seed": int(specification.seed),
                "success": bool(final_info.get("attack_success", False)),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "step_cap_reached": step_cap_reached,
                "termination_reason": reason,
                "steps": len(actions),
                "actions": actions,
                "detector_image_queries": episode_queries,
                "query_budget": int(specification.query_budget),
                "material_used_fraction": material_used,
                "material_budget_fraction": float(specification.material_budget),
                "potential": final_potential,
                "potential_components": potential_components,
                "potential_trace": potential_trace,
            }
        )

    evaluation_lifetime_delta = (
        int(environment.lifetime_detector_queries) - lifetime_before_evaluation
    )
    summary = _summarize(rows)
    if evaluation_lifetime_delta != int(summary["total_detector_image_queries"]):
        raise RuntimeError(
            "report query total disagrees with environment lifetime-query delta"
        )
    report = {
        "schema_version": 1,
        "method": "frozen_masked_meta_patch_policy_evaluation",
        "generated_at_utc": datetime.now(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z"),
        "protocol": {
            "training_performed": False,
            "adaptation_performed": False,
            "deterministic_policy": True,
            "action_masks_used": True,
            "explicit_per_episode_budgets": True,
            "step_cap": int(max_steps),
            "query_counting_unit": "one_detector_evaluation_of_one_image",
        },
        "episode_plan": [episode.as_dict() for episode in plan],
        "summary": summary,
        "episodes": rows,
    }
    if output_path is not None:
        atomic_write_json(output_path, report)
    return report


__all__ = [
    "EvaluationEpisode",
    "MaskedPredictor",
    "atomic_write_json",
    "evaluate_policy",
    "wilson_interval",
]
