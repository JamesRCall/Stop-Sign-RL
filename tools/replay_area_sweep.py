#!/usr/bin/env python3
"""
replay_area_sweep.py

Re-render hard cases from area_sweep.ndjson using stored seeds.

Expected inputs are typically under _runs or a run-id subfolder.
"""
import argparse
import json
import math
import os
import sys
import glob
from typing import List

import numpy as np
from PIL import Image, ImageDraw

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from envs.stop_sign_grid_env import StopSignGridEnv
from utils import uv_paint


def get_paint_map() -> dict:
    paints = {}
    for name in dir(uv_paint):
        if name.endswith("_GLOW"):
            p = getattr(uv_paint, name)
            if hasattr(p, "name"):
                paints[p.name.lower()] = p
    return paints


def parse_combo(combo: str, mapping: dict):
    parts = [p.strip().lower() for p in combo.split("+") if p.strip()]
    paints = []
    for p in parts:
        if p in mapping:
            paints.append(mapping[p])
    return paints


def apply_multi_color_overlay(sign_rgba: Image.Image, mode: str, env: StopSignGridEnv, paints: List, rng) -> Image.Image:
    assert mode in ("day", "on")
    rgb = sign_rgba.convert("RGB")
    a = sign_rgba.split()[-1]
    if not paints:
        return Image.merge("RGBA", (*rgb.split(), a))

    on = np.argwhere(env._episode_cells)
    for r, c in on:
        paint = paints[int(rng.integers(0, len(paints)))]
        if mode == "day":
            color = paint.day_rgb
            alpha = paint.day_alpha if paint.translucent else 1.0
        else:
            color = paint.active_rgb
            alpha = paint.active_alpha if paint.translucent else 1.0
            if paint.translucent:
                alpha = max(alpha, env.uv_min_alpha)

        x0, y0, x1, y1 = env._cell_rects[int(r) * env.Gw + int(c)]
        mask = Image.new("L", rgb.size, 0)
        mdraw = ImageDraw.Draw(mask)
        mdraw.rectangle([x0, y0, x1 - 1, y1 - 1], fill=255)
        mask = Image.composite(mask, Image.new("L", mask.size, 0), env._sign_alpha)
        if alpha < 1.0:
            arr = (np.array(mask, dtype=np.float32) * float(alpha)).astype(np.uint8)
        mask = Image.fromarray(arr)
        rgb.paste(color, mask=mask)

    return Image.merge("RGBA", (*rgb.split(), a))


def main() -> None:
    p = argparse.ArgumentParser(description="Replay area sweep cases and save images.")
    p.add_argument("--log", default="./_debug_area_sweep/area_sweep.ndjson")
    p.add_argument("--out", default="./_debug_area_sweep/replay")
    p.add_argument("--min-area", type=float, default=0.7)
    p.add_argument("--min-conf", type=float, default=0.5)
    p.add_argument("--limit", type=int, default=50)
    p.add_argument("--grid-cell", type=int, default=16, choices=[2, 4, 8, 16, 32])
    p.add_argument("--data", default="./data")
    p.add_argument("--bgdir", default="./data/backgrounds")
    p.add_argument("--yolo", default="./weights/yolo8n.pt")
    p.add_argument("--device", default="cuda")
    p.add_argument("--detector", default="yolo",
                   help="Detector backend: yolo or torchvision.")
    p.add_argument("--detector-model", default="",
                   help="Torchvision model name (e.g., fasterrcnn_resnet50_fpn_v2).")
    args = p.parse_args()

    if not os.path.isfile(args.log):
        raise FileNotFoundError(args.log)

    rows = []
    with open(args.log, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))

    rows = [
        r for r in rows
        if float(r.get("area_frac", 0.0)) >= float(args.min_area)
        and float(r.get("c_on", 0.0)) >= float(args.min_conf)
    ]

    rows.sort(key=lambda r: float(r.get("c_on", 0.0)), reverse=True)
    rows = rows[: max(1, int(args.limit))]

    if not rows:
        print("No rows matched filters.")
        return

    os.makedirs(args.out, exist_ok=True)

    stop_day = Image.open(os.path.join(args.data, "stop_sign.png")).convert("RGBA")
    uv_path = os.path.join(args.data, "stop_sign_uv.png")
    stop_uv = Image.open(uv_path).convert("RGBA") if os.path.exists(uv_path) else stop_day.copy()
    pole_path = os.path.join(args.data, "pole.png")
    pole = Image.open(pole_path).convert("RGBA") if os.path.exists(pole_path) else None

    bgs = [Image.open(p).convert("RGB") for p in sorted(glob.glob(os.path.join(args.bgdir, "*.*")))]
    if not bgs:
        raise FileNotFoundError(f"No backgrounds in {args.bgdir}")

    paint_map = get_paint_map()

    for i, r in enumerate(rows, 1):
        combo = r.get("paint_combo", "")
        paints = parse_combo(combo, paint_map)
        if not paints:
            continue

        env = StopSignGridEnv(
            stop_sign_image=stop_day,
            stop_sign_uv_image=stop_uv,
            background_images=bgs,
            pole_image=pole,
            yolo_weights=args.yolo,
            yolo_device=args.device,
            detector_type=str(args.detector),
            detector_model=str(args.detector_model) if args.detector_model else None,
            eval_K=int(r.get("eval_k", 1)),
            grid_cell_px=int(args.grid_cell),
            uv_paint=paints[0],
            uv_paint_list=paints,
            use_single_color=True,
        )

        env.reset()
        bg_idx = int(r.get("bg_idx", 0)) % len(bgs)
        env._bg_rgb = bgs[bg_idx].resize(env.img_size, Image.BILINEAR).convert("RGB")
        env._place_seed = int(r.get("place_seed", 0))
        env._transform_seeds = [int(s) for s in r.get("transform_seeds", [0])]

        valid_coords = env._valid_coords
        total = int(valid_coords.shape[0])
        target_n = int(math.ceil(float(r.get("area_frac", 0.0)) * total))

        env._episode_cells[:] = False
        if target_n > 0:
            mask_seed = int(r.get("mask_seed", env._place_seed))
            rng_mask = np.random.default_rng(mask_seed)
            if target_n >= total:
                env._episode_cells[:, :] = env._valid_cells
            else:
                picks = rng_mask.choice(total, size=target_n, replace=False)
                coords = valid_coords[picks]
                for rr, cc in coords:
                    env._episode_cells[int(rr), int(cc)] = True

        rng = np.random.default_rng(int(r.get("mask_seed", 0)))
        over_on = apply_multi_color_overlay(env.sign_rgba_on, "on", env, paints, rng)
        preview = env._compose_on_bg(env._transform_sign(over_on, env._transform_seeds[0]), env._place_seed)
        overlay = apply_multi_color_overlay(env.sign_rgba_on, "on", env, paints, rng)

        stem = f"case_{i:03d}_c{float(r.get('c_on',0.0)):.3f}_a{float(r.get('area_frac',0.0)):.2f}"
        preview.save(os.path.join(args.out, f\"{stem}_uv_on.png\"))
        overlay.save(os.path.join(args.out, f\"{stem}_overlay.png\"))

    print(f\"Saved {len(rows)} cases to {args.out}\")


if __name__ == \"__main__\":
    main()
