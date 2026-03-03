"""Shared utilities for grid-based baselines."""
from __future__ import annotations

import os
from typing import List, Tuple, Optional, Dict, Any
import math
import numpy as np
from PIL import Image
from types import SimpleNamespace

from envs.stop_sign_grid_env import StopSignGridEnv
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
    defaults = {"8": "./weights/yolo8n.pt", "11": "./weights/yolo11n.pt"}
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
    paths = sorted([os.path.join(folder, p) for p in os.listdir(folder)])
    imgs = []
    for p in paths:
        if not os.path.isfile(p):
            continue
        try:
            imgs.append(Image.open(p).convert("RGB"))
        except Exception:
            continue
    if not imgs:
        raise FileNotFoundError(f"No backgrounds found in: {folder}")
    return imgs[:20]


def build_solid_backgrounds(img_size: Tuple[int, int]) -> List[Image.Image]:
    colors = [(200, 200, 200), (120, 120, 120), (30, 30, 30)]
    W, H = int(img_size[0]), int(img_size[1])
    return [Image.new("RGB", (W, H), c) for c in colors]


def build_backgrounds(bg_mode: str, folder: str, img_size: Tuple[int, int]) -> List[Image.Image]:
    mode = str(bg_mode or "dataset").lower().strip()
    if mode == "solid":
        return build_solid_backgrounds(img_size)
    return load_backgrounds(folder)


def build_env_from_args(args) -> StopSignGridEnv:
    yolo_weights = resolve_yolo_weights(args.yolo_version, args.yolo_weights)
    detector_type = str(getattr(args, "detector", "yolo")).strip().lower()
    detector_model = str(getattr(args, "detector_model", "") or "").strip()

    stop_plain = Image.open(os.path.join(args.data, "stop_sign.png")).convert("RGBA")
    stop_uv_path = os.path.join(args.data, "stop_sign_uv.png")
    stop_uv = Image.open(stop_uv_path).convert("RGBA") if os.path.exists(stop_uv_path) else stop_plain.copy()
    pole_path = os.path.join(args.data, "pole.png")
    pole_rgba = Image.open(pole_path).convert("RGBA") if (os.path.exists(pole_path) and not args.no_pole) else None

    img_size = (640, 640)
    backgrounds = build_backgrounds(args.bg_mode, args.bgdir, img_size)
    paint_list = resolve_paint_list(args.paint, args.paint_list)

    env = StopSignGridEnv(
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
    )
    return env


def save_final_images(env: StopSignGridEnv, out_dir: str) -> None:
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
        "mean_target_conf", "mean_top_conf",
        "reward_core", "reward_raw_total",
        "reward_efficiency", "reward_perceptual", "reward_step_cost",
        "reward", "lambda_area_used",
        "total_area_mask_frac",
        "area_target_frac", "area_cap",
        "uv_success", "attack_success", "area_cap_exceeded",
        "selected_cells",
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
    overlay = env._eval_overlay_over_K(seeds)
    c_day = _as_float(overlay.get("c_day", float("nan")))
    c_on = _as_float(overlay.get("c_on", float("nan")))
    c0_day = _as_float(env._mean_over_K(env._baseline_c0_day_list, k))
    drop_on = _as_float(c0_day - c_on)
    area_frac = _as_float(env._area_frac_selected())
    success = 1.0 if (math.isfinite(c_on) and c_on <= float(env.success_conf_threshold)) else 0.0
    return {
        "c0_day": c0_day,
        "c_day": c_day,
        "c_on": c_on,
        "drop_on": drop_on,
        "area_frac": area_frac,
        "success": success,
    }


def _default_cfg_for_env(cfg: Dict[str, Any]) -> Dict[str, Any]:
    d = dict(cfg)
    d.setdefault("data", "./data")
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
    d.setdefault("lambda_day", 0.0)
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
