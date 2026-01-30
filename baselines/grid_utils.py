"""Shared utilities for grid-based baselines."""
from __future__ import annotations

import os
from typing import List, Tuple, Optional, Dict, Any
import math
import numpy as np
from PIL import Image

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
