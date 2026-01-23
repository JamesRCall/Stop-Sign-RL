#!/usr/bin/env python3
"""
area_sweep_debug.py

Sweep different area-cover percentages and report YOLO confidence for each.
Thorough mode can iterate backgrounds and placements and log detailed rows.
"""
import argparse
import os
import glob
import math
import sys
import json
from typing import List

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


def parse_int_list(s: str) -> List[int]:
    out = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Sweep patch area coverage vs YOLO confidence.")
    p.add_argument("--data", default="./data")
    p.add_argument("--bgdir", default="./data/backgrounds")
    p.add_argument("--yolo", default="./weights/yolo8n.pt")
    p.add_argument("--device", default="cuda")
    p.add_argument("--grid-cell", type=int, default=16, choices=[2, 4, 8, 16, 32])
    p.add_argument("--eval-k", type=int, default=5)
    p.add_argument("--eval-k-list", default="",
                   help="Comma-separated eval-K values to sweep (overrides --eval-k).")
    p.add_argument("--percentages", default="0,10,20,30,40,50,60,70,80,90,100")
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--trials", type=int, default=5,
                   help="Number of random masks per percentage.")
    p.add_argument("--full-cover", action="store_true",
                   help="Force 100% coverage instead of percentage sweep.")
    p.add_argument("--bg-samples", type=int, default=5,
                   help="Number of background images to sample.")
    p.add_argument("--pos-samples", type=int, default=5,
                   help="Number of placement seeds per background.")
    p.add_argument("--no-pole", action="store_true",
                   help="Disable pole for debug.")
    p.add_argument("--high-conf", type=float, default=0.5,
                   help="Threshold to flag and optionally save high-confidence cases.")
    p.add_argument("--save", action="store_true", help="Save sample images for each percentage.")
    p.add_argument("--out", default="./_debug_area_sweep")
    p.add_argument("--log", default="./_debug_area_sweep/area_sweep.ndjson")
    args = p.parse_args()

    stop_day = Image.open(os.path.join(args.data, "stop_sign.png")).convert("RGBA")
    uv_path = os.path.join(args.data, "stop_sign_uv.png")
    stop_uv = Image.open(uv_path).convert("RGBA") if os.path.exists(uv_path) else stop_day.copy()
    pole_path = os.path.join(args.data, "pole.png")
    pole = Image.open(pole_path).convert("RGBA") if os.path.exists(pole_path) else None
    bgs = load_bgs(args.bgdir)
    if not bgs:
        raise FileNotFoundError(f"No backgrounds found in: {args.bgdir}")

    percents = [1.0] if args.full_cover else parse_percents(args.percentages)
    k_list = parse_int_list(args.eval_k_list) if args.eval_k_list.strip() else [int(args.eval_k)]

    for eval_k in k_list:
        env = StopSignGridEnv(
            stop_sign_image=stop_day,
            stop_sign_uv_image=stop_uv,
            background_images=bgs,
            pole_image=None if args.no_pole else pole,
            yolo_weights=args.yolo,
            yolo_device=args.device,
            eval_K=int(eval_k),
            grid_cell_px=int(args.grid_cell),
            uv_paint=GREEN_GLOW,
            use_single_color=True,
        )

        rng = np.random.default_rng(int(args.seed))
        obs, _ = env.reset()
        if args.save:
            os.makedirs(args.out, exist_ok=True)
            Image.fromarray(obs).save(os.path.join(args.out, f"obs_day_baseline_k{eval_k}.png"))

        valid_coords = env._valid_coords  # (N,2)
        total = int(valid_coords.shape[0])
        if total <= 0:
            print("No valid cells in mask.")
            return

        bg_samples = max(1, int(args.bg_samples))
        pos_samples = max(1, int(args.pos_samples))
        bg_count = min(bg_samples, len(bgs))
        bg_indices = rng.choice(len(bgs), size=bg_count, replace=False)

        if args.save:
            os.makedirs(args.out, exist_ok=True)
        os.makedirs(os.path.dirname(args.log), exist_ok=True)

        print(f"\nvalid_cells={total} | eval_K={eval_k} | trials={int(args.trials)} | "
              f"bg_samples={bg_count} pos_samples={pos_samples} full_cover={bool(args.full_cover)}")

        for pct in percents:
            pct = max(0.0, min(1.0, float(pct)))
            target_n = int(math.ceil(pct * total))
            if target_n == 0:
                trials = 1
            else:
                trials = max(1, int(args.trials))

            rows = []
            for bg_idx in bg_indices:
                env._bg_rgb = bgs[int(bg_idx)].resize(env.img_size, Image.BILINEAR).convert("RGB")
                for _ in range(pos_samples):
                    env._place_seed = int(rng.integers(0, 2**31 - 1))
                    env._transform_seeds = [int(rng.integers(0, 2**31 - 1)) for _ in range(int(eval_k))]
                    env._baseline_c0_day_list, env._baseline_c0_on_list = env._eval_plain_over_K(env._transform_seeds)

                    for _ in range(trials):
                        env._episode_cells[:] = False
                        if target_n > 0:
                            if target_n >= total:
                                env._episode_cells[:, :] = env._valid_cells
                            else:
                                picks = rng.choice(total, size=target_n, replace=False)
                                coords = valid_coords[picks]
                                for r, c in coords:
                                    env._episode_cells[int(r), int(c)] = True

                        area_frac = float(env._episode_cells.sum()) / float(total)
                        eval_seeds = env._transform_seeds[: int(eval_k)]
                        metrics = env._eval_overlay_over_K(eval_seeds)
                        c_on = float(metrics.get("c_on", 0.0))
                        c_day = float(metrics.get("c_day", 0.0))
                        mean_iou = float(metrics.get("mean_iou", 0.0))
                        misclass_rate = float(metrics.get("misclass_rate", 0.0))
                        c0_day = float(env._mean_over_K(env._baseline_c0_day_list, int(eval_k)))
                        drop_on = float(c0_day - c_on)

                        rows.append((area_frac, c_on, c_day, drop_on, mean_iou, misclass_rate))

                        rec = {
                            "eval_k": int(eval_k),
                            "bg_idx": int(bg_idx),
                            "place_seed": int(env._place_seed),
                            "transform_seeds": [int(s) for s in env._transform_seeds],
                            "area_frac": float(area_frac),
                            "c_on": float(c_on),
                            "c_day": float(c_day),
                            "c0_day": float(c0_day),
                            "drop_on": float(drop_on),
                            "mean_iou": float(mean_iou),
                            "misclass_rate": float(misclass_rate),
                            "full_cover": bool(args.full_cover),
                        }
                        with open(args.log, "a", encoding="utf-8") as f:
                            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

                        if args.save and c_on >= float(args.high_conf):
                            stem = f"area_{int(area_frac*100):03d}_k{eval_k}_bg{int(bg_idx)}_p{int(env._place_seed)}"
                            preview = env._render_variant(kind="on", use_overlay=True, transform_seed=env._transform_seeds[0])
                            overlay = env._render_overlay_pattern(mode="on")
                            preview.save(os.path.join(args.out, f"{stem}_uv_on.png"))
                            overlay.save(os.path.join(args.out, f"{stem}_overlay.png"))

            arr = np.array(rows, dtype=float)
            mean = arr.mean(axis=0)
            std = arr.std(axis=0)

            print(
                f"area={mean[0]:.3f}+/-{std[0]:.3f} | "
                f"c_on={mean[1]:.3f}+/-{std[1]:.3f} | "
                f"c_day={mean[2]:.3f}+/-{std[2]:.3f} | "
                f"drop_on={mean[3]:.3f}+/-{std[3]:.3f} | "
                f"iou={mean[4]:.3f}+/-{std[4]:.3f} | "
                f"misclass={mean[5]:.3f}+/-{std[5]:.3f}"
            )

            if args.save:
                stem = f"area_{int(mean[0]*100):03d}_k{eval_k}_mean"
                preview = env._render_variant(kind="on", use_overlay=True, transform_seed=env._transform_seeds[0])
                overlay = env._render_overlay_pattern(mode="on")
                preview.save(os.path.join(args.out, f"{stem}_uv_on.png"))
                overlay.save(os.path.join(args.out, f"{stem}_overlay.png"))


if __name__ == "__main__":
    main()
