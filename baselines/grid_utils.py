"""Shared utilities for grid-based baselines."""
from __future__ import annotations

import os
from typing import List, Tuple, Optional, Dict, Any
import math
import numpy as np
from PIL import Image
from types import SimpleNamespace

from envs.traffic_sign_grid_env import TrafficSignGridEnv
from envs.attack_objective import AggregateAttackMetrics
from utils.sign_assets import resolve_sign_assets
from utils.uv_paint import (
    WHITE_GLOW,
    RED_GLOW,
    GREEN_GLOW,
    YELLOW_GLOW,
    BLUE_GLOW,
    ORANGE_GLOW,
    UVPaint,
)


def resolve_paint_list(paint: str, paint_list: Optional[str]) -> List[UVPaint]:
    mapping = {
        "white": WHITE_GLOW,
        "red": RED_GLOW,
        "green": GREEN_GLOW,
        "yellow": YELLOW_GLOW,
        "blue": BLUE_GLOW,
        "orange": ORANGE_GLOW,
    }
    paints = []
    if paint_list:
        for part in paint_list.split(","):
            key = part.strip().lower()
            if key in mapping:
                paints.append(mapping[key])
    if not paints:
        key = str(paint or "yellow").strip().lower()
        paints = [mapping.get(key, YELLOW_GLOW)]
    return paints


def resolve_yolo_weights(yolo_version: str, yolo_weights: Optional[str]) -> str:
    if yolo_weights:
        return yolo_weights
    defaults = {"8": "./weights/yolov8n.pt", "11": "./weights/yolo11n.pt"}
    return defaults[str(yolo_version)]


def parse_angle_list(value: str) -> List[float]:
    if value is None:
        return []
    out: List[float] = []
    for part in str(value).split(","):
        part = part.strip()
        if not part:
            continue
        try:
            out.append(float(part))
        except ValueError:
            continue
    return out


def load_backgrounds(folder: str) -> List[Image.Image]:
    supported = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
    paths = sorted(
        os.path.join(folder, p)
        for p in os.listdir(folder)
        if os.path.splitext(p)[1].lower() in supported
    )
    imgs = []
    failures = []
    for p in paths:
        if not os.path.isfile(p):
            continue
        try:
            imgs.append(Image.open(p).convert("RGB"))
        except Exception as exc:
            failures.append(f"{p}: {exc}")
    if failures:
        raise RuntimeError("Unreadable background assets:\n" + "\n".join(failures))
    if not imgs:
        raise FileNotFoundError(f"No backgrounds found in: {folder}")
    return imgs


def build_solid_backgrounds(img_size: Tuple[int, int]) -> List[Image.Image]:
    colors = [(200, 200, 200), (120, 120, 120), (30, 30, 30)]
    W, H = int(img_size[0]), int(img_size[1])
    return [Image.new("RGB", (W, H), c) for c in colors]


def build_backgrounds(bg_mode: str, folder: str, img_size: Tuple[int, int]) -> List[Image.Image]:
    mode = str(bg_mode or "dataset").lower().strip()
    if mode == "solid":
        return build_solid_backgrounds(img_size)
    return load_backgrounds(folder)


def build_env_from_args(args) -> TrafficSignGridEnv:
    yolo_weights = resolve_yolo_weights(args.yolo_version, args.yolo_weights)
    detector_type = str(getattr(args, "detector", "yolo")).strip().lower()
    detector_model = str(getattr(args, "detector_model", "") or "").strip()

    sign_assets = resolve_sign_assets(
        data_dir=args.data,
        profile=getattr(args, "sign_profile", "stop"),
        sign_image=(getattr(args, "sign_image", "") or None),
        sign_active_image=(getattr(args, "sign_active_image", "") or None),
        source_class=(getattr(args, "source_class", "") or None),
    )
    stop_plain = Image.open(sign_assets.day_image).convert("RGBA")
    stop_uv = Image.open(sign_assets.active_image).convert("RGBA")
    pole_path = os.path.join(args.data, "pole.png")
    pole_rgba = Image.open(pole_path).convert("RGBA") if (os.path.exists(pole_path) and not args.no_pole) else None

    img_size = (640, 640)
    backgrounds = build_backgrounds(args.bg_mode, args.bgdir, img_size)
    paint_override = getattr(args, "uv_paint_instance", None)
    paint_list = (
        [paint_override]
        if isinstance(paint_override, UVPaint)
        else resolve_paint_list(args.paint, args.paint_list)
    )

    env = TrafficSignGridEnv(
        stop_sign_image=stop_plain,
        stop_sign_uv_image=stop_uv,
        background_images=backgrounds,
        pole_image=pole_rgba,
        yolo_weights=yolo_weights,
        yolo_device=args.detector_device,
        detector_type=detector_type,
        detector_model=(detector_model if detector_model else None),
        img_size=img_size,
        obs_size=(int(args.obs_size), int(args.obs_size)),
        obs_margin=float(args.obs_margin),
        obs_include_mask=bool(int(args.obs_include_mask)),

        steps_per_episode=int(args.episode_steps),
        eval_K=int(args.eval_K),
        detector_debug=bool(int(args.detector_debug)),

        grid_cell_px=int(args.grid_cell),
        max_cells=None,
        action_indexing=str(getattr(args, "action_indexing", "valid_cells")),
        terminate_on_success=bool(
            int(getattr(args, "terminate_on_success", 1))
        ),
        uv_paint=paint_list[0],
        uv_paint_list=paint_list if len(paint_list) > 1 else None,
        use_single_color=True,
        cell_cover_thresh=float(args.cell_cover_thresh),

        uv_drop_threshold=float(args.uv_threshold),
        success_conf_threshold=float(args.success_conf),
        lambda_efficiency=float(args.lambda_efficiency),
        efficiency_eps=float(args.efficiency_eps),
        transform_strength=float(args.transform_strength),
        fixed_angle_deg=(
            float(args.fixed_angle_deg)
            if getattr(args, "fixed_angle_deg", None) is not None
            else None
        ),
        day_tolerance=float(args.day_tolerance),
        lambda_day=float(args.lambda_day),
        lambda_area=float(args.lambda_area),
        area_target_frac=(float(args.area_target) if args.area_target is not None else None),
        step_cost=float(args.step_cost),
        step_cost_after_target=float(args.step_cost_after_target),
        lambda_iou=float(args.lambda_iou),
        lambda_misclass=float(args.lambda_misclass),
        lambda_perceptual=float(args.lambda_perceptual),
        area_cap_frac=(float(args.area_cap_frac) if args.area_cap_frac and float(args.area_cap_frac) > 0 else None),
        area_cap_penalty=float(args.area_cap_penalty),
        area_cap_mode=str(args.area_cap_mode),
        source_class=sign_assets.source_class,
        attack_mode=str(getattr(args, "attack_mode", "disappearance")),
        attack_target_class=(getattr(args, "attack_target_class", "") or None),
        allowed_alternative_classes=getattr(args, "allowed_alternative_classes", ""),
        target_conf_threshold=float(getattr(args, "target_conf", 0.40)),
        min_attack_success_rate=float(getattr(args, "min_attack_success_rate", 0.80)),
        min_clean_detection_rate=float(getattr(args, "min_clean_detection_rate", 0.80)),
        localization_iou_threshold=float(getattr(args, "localization_iou", 0.30)),
        require_source_suppression=bool(int(getattr(args, "require_source_suppression", 1))),
        require_day_preservation=bool(int(getattr(args, "require_day_preservation", 1))),
        detector_instance=getattr(args, "detector_instance", None),
        seed=int(getattr(args, "seed", 0)),
    )
    return env


def save_final_images(env: TrafficSignGridEnv, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    day = env._render_variant(kind="day", use_overlay=True, transform_seed=env._transform_seeds[0])
    on = env._render_variant(kind="on", use_overlay=True, transform_seed=env._transform_seeds[0])
    overlay = env._render_overlay_pattern(mode="on")
    day.save(os.path.join(out_dir, "final_day.png"))
    on.save(os.path.join(out_dir, "final_on.png"))
    overlay.save(os.path.join(out_dir, "final_overlay.png"))


def info_metrics(info: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract common numeric metrics from env info for logging/JSON.
    """
    if not isinstance(info, dict):
        return {}

    keys = [
        "c0_day", "c_day", "c0_on", "c_on",
        "drop_day", "drop_on", "drop_on_smooth",
        "mean_iou", "misclass_rate",
        "disappearance_success_rate", "misclassification_success_rate",
        "targeted_success_rate", "clean_detection_rate",
        "mean_attack_target_conf", "mean_target_margin",
        "mean_target_conf", "mean_top_conf",
        "reward_core", "reward_raw_total",
        "reward_efficiency", "reward_perceptual", "reward_step_cost",
        "reward", "lambda_area_used",
        "total_area_mask_frac",
        "area_target_frac", "area_cap",
        "uv_success", "attack_success", "objective_success",
        "day_preserved", "within_area_budget", "area_cap_exceeded",
        "selected_cells",
        "detector_queries",
    ]

    out: Dict[str, Any] = {}
    for k in keys:
        if k in info:
            v = info.get(k)
            if isinstance(v, (np.generic,)):
                v = v.item()
            out[k] = v

    # Include optional note and class counts if present.
    if "note" in info:
        out["note"] = str(info.get("note"))
    if "top_class_counts" in info and isinstance(info.get("top_class_counts"), dict):
        out["top_class_counts"] = info.get("top_class_counts")

    return out


def log_metrics_tb(writer, metrics: Dict[str, Any], step: int, prefix: str = "metrics/") -> None:
    if writer is None:
        return
    for k, v in metrics.items():
        if isinstance(v, (bool, int, float, np.integer, np.floating)):
            fv = float(v)
            if math.isfinite(fv):
                writer.add_scalar(f"{prefix}{k}", fv, step)


def _as_float(v: Any, default: float = float("nan")) -> float:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return default
    return x if math.isfinite(x) else default


def _apply_pattern(env, pattern_type: str, pattern: List[int]) -> None:
    env._episode_cells[:] = False
    if pattern_type == "selected_indices":
        for idx in pattern:
            i = int(idx)
            if i < 0 or i >= (env.Gh * env.Gw):
                continue
            r, c = divmod(i, env.Gw)
            if env._valid_cells[r, c]:
                env._episode_cells[r, c] = True
        return
    if pattern_type == "actions":
        for a in pattern:
            i = int(a)
            if i < 0 or i >= int(env._n_valid):
                continue
            rr, cc = env._valid_coords[i]
            env._episode_cells[int(rr), int(cc)] = True
        return
    raise ValueError(f"Unknown pattern_type: {pattern_type}")


def _eval_pattern(env, eval_k: int) -> Dict[str, float]:
    k = max(1, min(int(eval_k), len(env._transform_seeds)))
    seeds = env._transform_seeds[:k]
    overlay = env._eval_overlay_over_K(
        seeds,
        baseline_day_confidences=env._baseline_c0_day_list[:k],
        baseline_on_confidences=env._baseline_c0_on_list[:k],
        baseline_day_metrics=env._baseline_day_metrics[:k],
        baseline_on_metrics=env._baseline_on_metrics[:k],
    )
    eligible = list(overlay.get("eligible_indices", []))
    c_day = _as_float(overlay.get("c_day", float("nan")))
    c_on = _as_float(overlay.get("c_on", float("nan")))
    c0_day = _as_float(np.mean([env._baseline_c0_day_list[i] for i in eligible])) if eligible else float("nan")
    c0_on = _as_float(np.mean([env._baseline_c0_on_list[i] for i in eligible])) if eligible else float("nan")
    drop_day = _as_float(c0_day - c_day)
    drop_on = _as_float(c0_on - c_on)
    area_frac = _as_float(env._area_frac_selected())
    metrics = overlay.get("attack_metrics", AggregateAttackMetrics())
    clean_rate = _as_float(overlay.get("clean_detection_rate", 0.0), 0.0)
    day_correct_rate = _as_float(overlay.get("day_correct_rate", 0.0), 0.0)
    joint = env.joint_success_components(
        metrics,
        clean_detection_rate=clean_rate,
        day_correct_rate=day_correct_rate,
        drop_day=drop_day,
        area_frac=area_frac,
    )
    success = 1.0 if joint["attack_success"] else 0.0
    return {
        "c0_day": c0_day,
        "c0_on": c0_on,
        "c_day": c_day,
        "c_on": c_on,
        "drop_day": drop_day,
        "drop_on": drop_on,
        "area_frac": area_frac,
        "success": success,
        "objective_success": 1.0 if joint["objective_success"] else 0.0,
        "clean_eligible": 1.0 if joint["clean_eligible"] else 0.0,
        "clean_detection_rate": clean_rate,
        "day_correct_rate": day_correct_rate,
        "eligible_transform_count": float(len(eligible)),
        "total_transform_count": float(k),
        "disappearance_success_rate": float(metrics.disappearance_rate),
        "misclassification_success_rate": float(metrics.untargeted_rate),
        "targeted_success_rate": float(metrics.targeted_rate),
        "mean_source_conf": float(metrics.mean_source_conf),
        "mean_source_iou": float(metrics.mean_source_iou),
        "mean_alternative_conf": float(metrics.mean_alternative_conf),
        "mean_alternative_iou": float(metrics.mean_alternative_iou),
        "mean_attack_target_conf": float(metrics.mean_attack_target_conf),
        "mean_attack_target_iou": float(metrics.mean_attack_target_iou),
        "mean_alternative_margin": float(metrics.mean_alternative_margin),
        "mean_target_margin": float(metrics.mean_target_margin),
        "day_preserved": 1.0 if joint["day_preserved"] else 0.0,
        "within_area_budget": 1.0 if joint["within_area_budget"] else 0.0,
    }


def _default_cfg_for_env(cfg: Dict[str, Any]) -> Dict[str, Any]:
    d = dict(cfg)
    d.setdefault("data", "./data")
    d.setdefault("sign_profile", "stop")
    d.setdefault("sign_image", "")
    d.setdefault("sign_active_image", "")
    d.setdefault("source_class", "")
    d.setdefault("attack_mode", "disappearance")
    d.setdefault("attack_target_class", "")
    d.setdefault("allowed_alternative_classes", "")
    d.setdefault("target_conf", 0.40)
    d.setdefault("min_attack_success_rate", 0.80)
    d.setdefault("min_clean_detection_rate", 0.80)
    d.setdefault("localization_iou", 0.30)
    d.setdefault("require_source_suppression", 1)
    d.setdefault("require_day_preservation", 1)
    d.setdefault("bgdir", "./data/backgrounds")
    d.setdefault("bg_mode", "dataset")
    d.setdefault("no_pole", False)
    d.setdefault("yolo_version", "8")
    d.setdefault("yolo_weights", None)
    d.setdefault("detector", "yolo")
    d.setdefault("detector_model", "")
    d.setdefault("detector_device", "auto")
    d.setdefault("detector_debug", 0)
    d.setdefault("eval_K", 3)
    d.setdefault("grid_cell", 16)
    d.setdefault("episode_steps", 300)
    d.setdefault("transform_strength", 1.0)
    d.setdefault("fixed_angle_deg", None)
    d.setdefault("day_tolerance", 0.05)
    d.setdefault("lambda_area", 0.70)
    d.setdefault("lambda_efficiency", 0.40)
    d.setdefault("efficiency_eps", 0.02)
    d.setdefault("lambda_day", 1.0)
    d.setdefault("lambda_iou", 0.40)
    d.setdefault("lambda_misclass", 0.60)
    d.setdefault("lambda_perceptual", 0.0)
    d.setdefault("area_target", 0.25)
    d.setdefault("step_cost", 0.012)
    d.setdefault("step_cost_after_target", 0.14)
    d.setdefault("area_cap_frac", 0.30)
    d.setdefault("area_cap_penalty", -0.20)
    d.setdefault("area_cap_mode", "soft")
    d.setdefault("uv_threshold", 0.75)
    d.setdefault("success_conf", 0.20)
    d.setdefault("paint", "yellow")
    d.setdefault("paint_list", "")
    d.setdefault("cell_cover_thresh", 0.60)
    d.setdefault("obs_size", 224)
    d.setdefault("obs_margin", 0.10)
    d.setdefault("obs_include_mask", 1)
    return d


def eval_pattern_over_angles(
    base_args,
    pattern_type: str,
    pattern: List[int],
    seed: int,
    angles: List[float],
    eval_k: int,
    detector_device: Optional[str] = None,
) -> List[Dict[str, float]]:
    if not angles or not pattern:
        return []
    if hasattr(base_args, "__dict__"):
        base_cfg = dict(vars(base_args))
    else:
        base_cfg = dict(base_args)
    if detector_device is not None:
        base_cfg["detector_device"] = detector_device
    base_cfg["eval_K"] = int(eval_k)
    base_cfg = _default_cfg_for_env(base_cfg)

    out: List[Dict[str, float]] = []
    for angle in angles:
        cfg = dict(base_cfg)
        cfg["fixed_angle_deg"] = float(angle)
        env = build_env_from_args(SimpleNamespace(**cfg))
        env.reset(seed=int(seed))
        _apply_pattern(env, pattern_type, pattern)
        metrics = _eval_pattern(env, eval_k=eval_k)
        metrics["angle_deg"] = float(angle)
        out.append(metrics)
    return out


def eval_pattern_over_angles_in_env(
    env,
    pattern_type: str,
    pattern: List[int],
    angles: List[float],
    eval_k: int,
) -> List[Dict[str, float]]:
    if not angles or not pattern:
        return []
    orig_fixed = getattr(env, "fixed_angle_deg", None)
    orig_cells = env._episode_cells.copy()
    _apply_pattern(env, pattern_type, pattern)
    out: List[Dict[str, float]] = []
    for angle in angles:
        env.fixed_angle_deg = float(angle)
        k = max(1, min(int(eval_k), len(env._transform_seeds)))
        seeds = env._transform_seeds[:k]
        c0_day_list, c0_on_list = env._eval_plain_over_K(seeds)
        overlay = env._eval_overlay_over_K(
            seeds,
            baseline_day_confidences=c0_day_list,
            baseline_on_confidences=c0_on_list,
            baseline_day_metrics=env._last_plain_day_metrics,
            baseline_on_metrics=env._last_plain_on_metrics,
        )
        eligible = list(overlay.get("eligible_indices", []))
        c0_day = _as_float(np.mean([c0_day_list[i] for i in eligible])) if eligible else float("nan")
        c0_on = _as_float(np.mean([c0_on_list[i] for i in eligible])) if eligible else float("nan")
        c_day = _as_float(overlay.get("c_day", float("nan")))
        c_on = _as_float(overlay.get("c_on", float("nan")))
        drop_day = _as_float(c0_day - c_day)
        drop_on = _as_float(c0_on - c_on)
        area_frac = _as_float(env._area_frac_selected())
        metrics = overlay.get("attack_metrics", AggregateAttackMetrics())
        clean_rate = _as_float(overlay.get("clean_detection_rate", 0.0), 0.0)
        day_correct_rate = _as_float(overlay.get("day_correct_rate", 0.0), 0.0)
        joint = env.joint_success_components(
            metrics,
            clean_detection_rate=clean_rate,
            day_correct_rate=day_correct_rate,
            drop_day=drop_day,
            area_frac=area_frac,
        )
        success = 1.0 if joint["attack_success"] else 0.0
        out.append(
            {
                "angle_deg": float(angle),
                "c0_day": c0_day,
                "c0_on": c0_on,
                "c_day": c_day,
                "c_on": c_on,
                "drop_day": drop_day,
                "drop_on": drop_on,
                "area_frac": area_frac,
                "success": success,
                "clean_detection_rate": clean_rate,
                "day_correct_rate": day_correct_rate,
                "disappearance_success_rate": float(metrics.disappearance_rate),
                "misclassification_success_rate": float(metrics.untargeted_rate),
                "targeted_success_rate": float(metrics.targeted_rate),
            }
        )
    env.fixed_angle_deg = orig_fixed
    env._episode_cells = orig_cells
    return out
