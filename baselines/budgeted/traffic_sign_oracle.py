"""Traffic-sign environment adapter for the budgeted optimizer protocol."""
from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Dict, Optional, Tuple

import numpy as np

from envs.attack_objective import (
    AggregateAttackMetrics,
    reward_terms,
    success_progress,
)
from .protocol import OracleObservation


def exact_material_limit(sign_pixels: int, area_fraction: float) -> int:
    """Largest integer pixel count satisfying an inclusive fractional cap."""
    if int(sign_pixels) <= 0:
        raise ValueError("sign_pixels must be positive")
    if not 0.0 <= float(area_fraction) <= 1.0:
        raise ValueError("area_fraction must be in [0, 1]")
    return int(math.floor(float(sign_pixels) * float(area_fraction) + 1e-12))


def environment_objective_score(
    env: Any,
    *,
    attack_metrics: AggregateAttackMetrics,
    c0_day: float,
    c_day: float,
    c0_on: float,
    c_on: float,
    area_fraction: float,
    clean_detection_rate: float,
    day_correct_rate: float,
) -> Tuple[float, Dict[str, Any]]:
    """Reproduce ``TrafficSignGridEnv.step`` reward for an arbitrary mask.

    The function is path-independent: the candidate mask is evaluated as one
    black-box proposal.  The environment's constant per-proposal step cost is
    retained, while its episode counter and early-termination state are not.
    """
    c0_day = float(c0_day)
    c_day = float(c_day)
    c0_on = float(c0_on)
    c_on = float(c_on)
    area_fraction = float(area_fraction)
    drop_day = float(c0_day - c_day)
    drop_on = float(c0_on - c_on)
    joint = env.joint_success_components(
        attack_metrics,
        clean_detection_rate=float(clean_detection_rate),
        day_correct_rate=float(day_correct_rate),
        drop_day=drop_day,
        area_frac=area_fraction,
    )
    if min(c0_day, c0_on) < float(env.min_base_conf):
        return -0.05, {
            "raw_total": -0.05,
            "drop_day": drop_day,
            "drop_on": drop_on,
            **joint,
            "baseline_conf_too_low": True,
        }

    pen_day = max(0.0, drop_day - float(env.day_tolerance))
    max_drop = max(0.0, c0_on - float(env.success_conf_threshold))
    drop_blend = max(0.0, min(drop_on, max_drop))
    classification_gain, localization_gain = reward_terms(
        attack_metrics, env.attack_config
    )
    suppression_weight = 1.0 if env.attack_mode == "disappearance" else 0.25
    suppression_gain = suppression_weight * drop_blend
    efficiency_signal = (
        drop_blend
        if env.attack_mode == "disappearance"
        else classification_gain
    )
    efficiency = math.log1p(
        max(0.0, float(efficiency_signal))
        / max(area_fraction, float(env.efficiency_eps))
    )
    area_target = (
        env.area_target_frac
        if env.area_target_frac is not None
        else env.area_cap_frac
    )
    lambda_area = float(env.lambda_area)
    step_cost_penalty = float(env.step_cost)
    if (
        float(env.step_cost_after_target) > 0.0
        and area_target is not None
        and area_fraction > float(area_target)
    ):
        excess_ratio = (
            area_fraction - float(area_target)
        ) / max(float(area_target), 1e-6)
        step_cost_penalty += float(env.step_cost_after_target) * (
            1.0 + max(0.0, excess_ratio)
        )
    excess_penalty = 0.0
    if area_target is not None and area_fraction > float(area_target):
        excess = area_fraction - float(area_target)
        excess_penalty = lambda_area * 4.5 * excess + lambda_area * excess**2

    raw_core = (
        suppression_gain
        - float(env.lambda_day) * pen_day
        - lambda_area * area_fraction
        - excess_penalty
        - step_cost_penalty
        + float(env.lambda_iou) * localization_gain
        + float(env.lambda_classification) * classification_gain
        + float(env.lambda_efficiency) * efficiency
    )
    perceptual = float(env._perceptual_delta())
    raw_core -= float(env.lambda_perceptual) * perceptual
    shaping = 0.35 * math.tanh(
        3.0 * success_progress(attack_metrics, env.attack_config)
    )
    success_bonus = (
        0.2 * ((1.0 - area_fraction) ** 2)
        if bool(joint["attack_success"])
        else 0.0
    )
    raw_total = raw_core + shaping + success_bonus
    cap_exceeded = bool(
        env.area_cap_frac is not None
        and area_fraction > float(env.area_cap_frac)
    )
    if cap_exceeded and str(env.area_cap_mode) == "soft":
        if env.area_cap_frac and float(env.area_cap_frac) > 0.0:
            relative_excess = max(
                0.0,
                (area_fraction - float(env.area_cap_frac))
                / float(env.area_cap_frac),
            )
            over_penalty = abs(float(env.area_cap_penalty)) * (
                1.0 + 2.0 * relative_excess
            )
        else:
            over_penalty = abs(float(env.area_cap_penalty))
        raw_total = -over_penalty
    score = math.tanh(1.2 * raw_total)
    return float(score), {
        "raw_core": float(raw_core),
        "raw_total": float(raw_total),
        "drop_day": drop_day,
        "drop_on": drop_on,
        "classification_gain": float(classification_gain),
        "localization_gain": float(localization_gain),
        "efficiency": float(efficiency),
        "perceptual_delta": perceptual,
        "cap_exceeded": cap_exceeded,
        "baseline_conf_too_low": False,
        **joint,
    }


class TrafficSignCandidateOracle:
    """Fixed-EOT candidate oracle with measured detector-image accounting."""

    def __init__(
        self,
        env: Any,
        *,
        scene_seed: int,
        eval_k: Optional[int] = None,
        experiment_artifact_identity: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.env = env
        self._closed = False
        self.experiment_artifact_identity = dict(
            experiment_artifact_identity or {}
        )
        try:
            self._initialize(scene_seed=scene_seed, eval_k=eval_k)
        except BaseException:
            self.close()
            raise

    def _initialize(
        self, *, scene_seed: int, eval_k: Optional[int]
    ) -> None:
        env = self.env
        self.scene_seed = int(scene_seed)
        self.eval_k = int(env.eval_K_max if eval_k is None else eval_k)
        if self.eval_k < 1 or self.eval_k > int(env.eval_K_max):
            raise ValueError("eval_k must be in [1, env.eval_K_max]")
        # Adaptive K makes candidate costs history-dependent and is therefore
        # prohibited for matched comparisons.
        if int(env.eval_K_min) != int(env.eval_K_max):
            raise ValueError(
                "budgeted comparisons require fixed EOT: eval_K_min == eval_K_max"
            )
        if self.eval_k != int(env.eval_K_max):
            raise ValueError(
                "eval_k must equal env.eval_K_max so all reference queries are used"
            )

        env.reset(seed=self.scene_seed)
        base_dimension = int(env.Gh * env.Gw)
        palette_size = (
            int(env.paint_action_count)
            if str(getattr(env, "paint_action_mode", "fixed")) == "joint_palette"
            else 1
        )
        self.dimension = base_dimension * palette_size
        valid_cells = tuple(
            int(value) for value in np.flatnonzero(env._valid_cells.reshape(-1))
        )
        self.selectable_indices = tuple(
            cell * palette_size + material
            for cell in valid_cells
            for material in range(palette_size)
        )
        base_costs = tuple(int(value) for value in env._cell_pixel_areas.reshape(-1))
        self.cell_material_pixels = tuple(
            base_costs[token // palette_size] for token in range(self.dimension)
        )
        # A cell may receive exactly one material.  All material tokens for that
        # cell therefore share one exclusive candidate group.
        self.candidate_group_ids = tuple(
            token // palette_size for token in range(self.dimension)
        )
        self.palette_size = palette_size
        self.sign_material_pixels = int(env._sign_pixel_area)
        self.objective_material_pixel_limit = (
            exact_material_limit(
                self.sign_material_pixels, float(env.area_cap_frac)
            )
            if env.area_cap_frac is not None
            else None
        )
        self.detector_queries_per_evaluation = int(2 * self.eval_k)
        self.initial_detector_queries = int(env._detector_queries)
        expected_initial = int(2 * self.eval_k)
        if self.initial_detector_queries != expected_initial:
            raise RuntimeError(
                "environment reference-query invariant failed: expected "
                f"{expected_initial}, observed {self.initial_detector_queries}"
            )
        self.objective_id = self._objective_id()

    @property
    def detector_queries_total(self) -> int:
        return int(self.env._detector_queries)

    def _objective_id(self) -> str:
        env = self.env
        attack = env.attack_config
        detector = env.det
        detector_identity = {
            "class": (
                detector.__class__.__module__
                + "."
                + detector.__class__.__qualname__
            ),
            "id_to_name": {
                str(key): str(value)
                for key, value in sorted(
                    dict(getattr(detector, "id_to_name", {}) or {}).items()
                )
            },
        }
        for name in (
            "conf",
            "iou",
            "target_id",
            "model_name",
            "server_addr",
            "device",
        ):
            value = getattr(detector, name, None)
            if value is not None:
                detector_identity[name] = (
                    float(value)
                    if isinstance(value, (int, float)) and not isinstance(value, bool)
                    else str(value)
                )
        paint = env.paint
        palette = tuple(getattr(env, "paint_palette", (paint,)))
        contract = {
            "adapter": "traffic-sign-fixed-eot-v1",
            "scene_seed": self.scene_seed,
            "place_seed": int(env._place_seed),
            "transform_seeds": [
                int(value) for value in env._transform_seeds[: self.eval_k]
            ],
            "background_index": getattr(env, "_bg_index", None),
            "background_rgb_sha256": _image_like_sha256(env._bg_rgb),
            "sign_day_sha256": _image_like_sha256(env.sign_rgba_day),
            "sign_active_sha256": _image_like_sha256(env.sign_rgba_on),
            "detector": detector_identity,
            "experiment_artifacts": self.experiment_artifact_identity,
            "attack_mode": str(env.attack_mode),
            "source_class_id": int(env.source_class_id),
            "target_class_id": (
                int(env.attack_target_id)
                if env.attack_target_id is not None
                else None
            ),
            "attack_config": {
                "source_conf_threshold": float(attack.source_conf_threshold),
                "target_conf_threshold": float(attack.target_conf_threshold),
                "min_success_rate": float(attack.min_success_rate),
                "localization_iou_threshold": float(
                    attack.localization_iou_threshold
                ),
                "require_source_suppression": bool(
                    attack.require_source_suppression
                ),
                "allowed_alternative_class_ids": list(
                    attack.allowed_alternative_class_ids or ()
                ),
            },
            "joint_constraints": {
                "min_clean_detection_rate": float(env.min_clean_detection_rate),
                "require_day_preservation": bool(env.require_day_preservation),
                "day_tolerance": float(env.day_tolerance),
                "area_cap_frac": env.area_cap_frac,
            },
            "reward": {
                name: float(getattr(env, name))
                for name in (
                    "success_conf_threshold",
                    "lambda_day",
                    "lambda_area",
                    "lambda_iou",
                    "lambda_classification",
                    "lambda_efficiency",
                    "efficiency_eps",
                    "lambda_perceptual",
                    "step_cost",
                    "step_cost_after_target",
                    "area_cap_penalty",
                )
            },
            "area_target_frac": env.area_target_frac,
            "area_cap_mode": str(env.area_cap_mode),
            "paint": {
                "name": str(getattr(paint, "name", "unknown")),
                "day_rgb": list(getattr(paint, "day_rgb", ())),
                "active_rgb": list(getattr(paint, "active_rgb", ())),
                "translucent": bool(getattr(paint, "translucent", False)),
                "day_alpha": float(getattr(paint, "day_alpha", 1.0)),
                "active_alpha": float(getattr(paint, "active_alpha", 1.0)),
            },
            "paint_action": {
                "mode": str(getattr(env, "paint_action_mode", "fixed")),
                "encoding": str(getattr(env, "action_encoding", "unknown")),
                "palette": [
                    {
                        "name": str(getattr(value, "name", "unknown")),
                        "day_rgb": list(getattr(value, "day_rgb", ())),
                        "active_rgb": list(getattr(value, "active_rgb", ())),
                        "translucent": bool(getattr(value, "translucent", False)),
                        "day_alpha": float(getattr(value, "day_alpha", 1.0)),
                        "active_alpha": float(getattr(value, "active_alpha", 1.0)),
                    }
                    for value in palette
                ],
            },
            "eval_k": self.eval_k,
            "grid_shape": [int(env.Gh), int(env.Gw)],
            "cell_material_pixels_sha256": hashlib.sha256(
                np.asarray(env._cell_pixel_areas, dtype="<i8").tobytes()
            ).hexdigest(),
            "render": {
                "img_size": [int(value) for value in env.img_size],
                "transform_strength": float(env.transform_strength),
                "fixed_angle_deg": (
                    float(env.fixed_angle_deg)
                    if env.fixed_angle_deg is not None
                    else None
                ),
            },
            "physics_transport": getattr(env, "physics_transport_result", None),
        }
        payload = json.dumps(
            contract,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        return "traffic-sign-env-reward-v1:" + hashlib.sha256(payload).hexdigest()

    def evaluate(self, selected_indices: Tuple[int, ...]) -> OracleObservation:
        if self._closed:
            raise RuntimeError("traffic-sign candidate oracle is closed")
        env = self.env
        env._episode_cells[:] = False
        if getattr(env, "_episode_paint_ids", None) is not None:
            env._episode_paint_ids[:] = -1
        for token in selected_indices:
            cell_index, material_index = divmod(int(token), int(self.palette_size))
            row, col = divmod(cell_index, int(env.Gw))
            if not bool(env._valid_cells[row, col]):
                raise ValueError(f"token {token} is not a printable grid-cell material")
            env._episode_cells[row, col] = True
            if getattr(env, "_episode_paint_ids", None) is not None:
                env._episode_paint_ids[row, col] = int(material_index)

        seeds = env._transform_seeds[: self.eval_k]
        overlay = env._eval_overlay_over_K(
            seeds,
            baseline_day_confidences=env._baseline_c0_day_list[: self.eval_k],
            baseline_on_confidences=env._baseline_c0_on_list[: self.eval_k],
            baseline_day_metrics=env._baseline_day_metrics[: self.eval_k],
            baseline_on_metrics=env._baseline_on_metrics[: self.eval_k],
        )
        eligible = list(overlay.get("eligible_indices", []))
        c0_day = (
            float(np.mean([env._baseline_c0_day_list[i] for i in eligible]))
            if eligible
            else 0.0
        )
        c0_on = (
            float(np.mean([env._baseline_c0_on_list[i] for i in eligible]))
            if eligible
            else 0.0
        )
        c_day = float(overlay.get("c_day", 0.0))
        c_on = float(overlay.get("c_on", 0.0))
        selected_pixels = int(
            env._cell_pixel_areas[env._episode_cells].sum()
        )
        area_fraction = float(selected_pixels) / float(env._sign_pixel_area)
        attack_metrics = overlay.get("attack_metrics", AggregateAttackMetrics())
        clean_rate = float(overlay.get("clean_detection_rate", 0.0))
        day_rate = float(overlay.get("day_correct_rate", 0.0))
        score, breakdown = environment_objective_score(
            env,
            attack_metrics=attack_metrics,
            c0_day=c0_day,
            c_day=c_day,
            c0_on=c0_on,
            c_on=c_on,
            area_fraction=area_fraction,
            clean_detection_rate=clean_rate,
            day_correct_rate=day_rate,
        )
        metric_breakdown = dict(breakdown)
        metric_breakdown["objective_condition_success"] = bool(
            metric_breakdown.pop("objective_success")
        )
        metric_breakdown["joint_success"] = bool(
            metric_breakdown.pop("attack_success")
        )
        metrics: Dict[str, Any] = {
            "attack_mode": str(env.attack_mode),
            "c0_day": c0_day,
            "c_day": c_day,
            "c0_on": c0_on,
            "c_on": c_on,
            "selected_material_pixels": selected_pixels,
            "sign_material_pixels": int(env._sign_pixel_area),
            "material_fraction": area_fraction,
            "selected_action_tokens": [int(value) for value in selected_indices],
            "cell_material_assignments": (
                env._selected_material_assignments()
                if str(getattr(env, "paint_action_mode", "fixed")) == "joint_palette"
                else [
                    {
                        "cell_index": int(value),
                        "material_index": 0,
                        "material_name": str(getattr(env.paint, "name", "unknown")),
                    }
                    for value in selected_indices
                ]
            ),
            "clean_detection_rate": clean_rate,
            "day_correct_rate": day_rate,
            "eligible_transform_count": len(eligible),
            "total_transform_count": self.eval_k,
            **attack_metrics.as_dict(),
            **metric_breakdown,
        }
        return OracleObservation(
            score=score,
            joint_success=bool(breakdown["attack_success"]),
            metrics=metrics,
        )

    def close(self) -> None:
        """Release detector/environment resources; safe to call repeatedly."""
        if self._closed:
            return
        self._closed = True
        detector = getattr(self.env, "det", None)
        detector_close = getattr(detector, "close", None)
        try:
            if callable(detector_close):
                detector_close()
            else:
                # The repository's remote wrapper predates a public close()
                # method.  Close its owned connection without assuming one was
                # ever established.
                connection = getattr(detector, "_conn", None)
                if connection is not None:
                    connection.close()
                    detector._conn = None
        finally:
            env_close = getattr(self.env, "close", None)
            if callable(env_close):
                env_close()


def _image_like_sha256(value: Any) -> str:
    """Hash PIL images or NumPy arrays with shape/mode metadata."""
    if hasattr(value, "tobytes"):
        raw = value.tobytes()
    else:
        raw = np.asarray(value).tobytes()
    metadata = {
        "mode": getattr(value, "mode", None),
        "size": list(getattr(value, "size", ())),
        "shape": list(getattr(value, "shape", ())),
        "dtype": str(getattr(value, "dtype", "")),
    }
    digest = hashlib.sha256()
    digest.update(
        json.dumps(metadata, sort_keys=True, allow_nan=False).encode("utf-8")
    )
    digest.update(raw)
    return digest.hexdigest()
