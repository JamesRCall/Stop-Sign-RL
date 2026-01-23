#!/usr/bin/env python3
"""
area_sweep_debug.py

Sweep different area-cover percentages and report YOLO confidence for each.
Useful for diagnosing whether coverage actually lowers confidence.
"""
import argparse
import os
import glob
import math
from typing import List
import sys

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from envs.stop_sign_grid_env import StopSignGridEnv
from utils.uv_paint import GREEN_GLOW


def load_bgs(folder: str) -> List[Image.Image]:
    paths = sorted(glob.glob(os.path.join(folder, "*.*")))
    return [Image.open(p).convert("RGB") for p in paths]


def parse_percents(s: str) -> List[float]:
    out = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        val = float(part)
        if val > 1.0:
            val = val / 100.0
        out.append(val)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Sweep patch area coverage vs YOLO confidence.")
    p.add_argument("--data", default="./data")
    p.add_argument("--bgdir", default="./data/backgrounds")
    p.add_argument("--yolo", default="./weights/yolo8n.pt")
    p.add_argument("--device", default="cuda")
    p.add_argument("--grid-cell", type=int, default=16, choices=[2, 4, 8, 16, 32])
    p.add_argument("--eval-k", type=int, default=5)
    p.add_argument("--percentages", default="0,10,20,30,40,50,60,70,80,90,100")
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--save", action="store_true", help="Save sample images for each percentage.")
    p.add_argument("--out", default="./_debug_area_sweep")
    args = p.parse_args()

    stop_day = Image.open(os.path.join(args.data, "stop_sign.png")).convert("RGBA")
    uv_path = os.path.join(args.data, "stop_sign_uv.png")
    stop_uv = Image.open(uv_path).convert("RGBA") if os.path.exists(uv_path) else stop_day.copy()
    pole_path = os.path.join(args.data, "pole.png")
    pole = Image.open(pole_path).convert("RGBA") if os.path.exists(pole_path) else None
    bgs = load_bgs(args.bgdir)

    env = StopSignGridEnv(
        stop_sign_image=stop_day,
        stop_sign_uv_image=stop_uv,
        background_images=bgs,
        pole_image=pole,
        yolo_weights=args.yolo,
        yolo_device=args.device,
        eval_K=int(args.eval_k),
        grid_cell_px=int(args.grid_cell),
        uv_paint=GREEN_GLOW,
        use_single_color=True,
    )

    rng = np.random.default_rng(int(args.seed))
    obs, _ = env.reset()
    if args.save:
        os.makedirs(args.out, exist_ok=True)
        Image.fromarray(obs).save(os.path.join(args.out, "obs_day_baseline.png"))

    valid_coords = env._valid_coords  # (N,2)
    total = int(valid_coords.shape[0])
    if total <= 0:
        print("No valid cells in mask.")
        return

    percents = parse_percents(args.percentages)
    print(f"valid_cells={total} | eval_K={args.eval_k}")

    for pct in percents:
        pct = max(0.0, min(1.0, float(pct)))
        target_n = int(math.ceil(pct * total))
        env._episode_cells[:] = False

        if target_n > 0:
            picks = rng.choice(total, size=target_n, replace=False)
            coords = valid_coords[picks]
            for r, c in coords:
                env._episode_cells[int(r), int(c)] = True

        area_frac = float(env._episode_cells.sum()) / float(total)
        eval_seeds = env._transform_seeds[: int(args.eval_k)]
        metrics = env._eval_overlay_over_K(eval_seeds)
        c_on = float(metrics.get("c_on", 0.0))
        c_day = float(metrics.get("c_day", 0.0))
        mean_iou = float(metrics.get("mean_iou", 0.0))
        misclass_rate = float(metrics.get("misclass_rate", 0.0))
        drop_on = float(env._mean_over_K(env._baseline_c0_day_list, int(args.eval_k)) - c_on)

        print(f"area={area_frac:.3f} | c_on={c_on:.3f} | c_day={c_day:.3f} | drop_on={drop_on:.3f} | "
              f"iou={mean_iou:.3f} | misclass={misclass_rate:.3f}")

        if args.save:
            preview = env._render_variant(kind="on", use_overlay=True, transform_seed=env._transform_seeds[0])
            overlay = env._render_overlay_pattern(mode="on")
            stem = f"area_{int(area_frac*100):03d}"
            preview.save(os.path.join(args.out, f"{stem}_uv_on.png"))
            overlay.save(os.path.join(args.out, f"{stem}_overlay.png"))


if __name__ == "__main__":
    main()
