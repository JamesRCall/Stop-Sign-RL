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
import itertools
import time
from typing import List

import numpy as np
from PIL import Image, ImageDraw

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from envs.stop_sign_grid_env import StopSignGridEnv, _iou_xyxy
from utils.uv_paint import (
    GREEN_GLOW, VIOLET_GLOW, BLUE_GLOW, YELLOW_GLOW,
    RED_GLOW, ORANGE_GLOW, CYAN_GLOW, MAGENTA_GLOW, PINK_GLOW, LIME_GLOW,
    CHARTREUSE_GLOW, SKY_GLOW, NAVY_GLOW, GOLD_GLOW, AMBER_GLOW, CORAL_GLOW,
    MAROON_GLOW, OLIVE_GLOW, MINT_GLOW, AQUA_GLOW, PURPLE_GLOW, INDIGO_GLOW,
    SLATE_GLOW, BROWN_GLOW, TAN_GLOW, PEACH_GLOW, STEEL_GLOW, LAVENDER_GLOW,
    NEON_YELLOW_GLOW, BLUE_LIGHT_GLOW, BLUE_MED_GLOW, BLUE_DARK_GLOW,
    UVPaint,
)


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


def parse_paint_list(s: str) -> List[str]:
    out = []
    for part in s.split(","):
        part = part.strip().lower()
        if part:
            out.append(part)
    return out


def resolve_paints(names: List[str]) -> List[UVPaint]:
    mapping = {
        "green": GREEN_GLOW,
        "violet": VIOLET_GLOW,
        "blue": BLUE_GLOW,
        "yellow": YELLOW_GLOW,
        "red": RED_GLOW,
        "orange": ORANGE_GLOW,
        "cyan": CYAN_GLOW,
        "magenta": MAGENTA_GLOW,
        "pink": PINK_GLOW,
        "lime": LIME_GLOW,
        "chartreuse": CHARTREUSE_GLOW,
        "sky": SKY_GLOW,
        "navy": NAVY_GLOW,
        "gold": GOLD_GLOW,
        "amber": AMBER_GLOW,
        "coral": CORAL_GLOW,
        "maroon": MAROON_GLOW,
        "olive": OLIVE_GLOW,
        "mint": MINT_GLOW,
        "aqua": AQUA_GLOW,
        "purple": PURPLE_GLOW,
        "indigo": INDIGO_GLOW,
        "slate": SLATE_GLOW,
        "brown": BROWN_GLOW,
        "tan": TAN_GLOW,
        "peach": PEACH_GLOW,
        "steel": STEEL_GLOW,
        "lavender": LAVENDER_GLOW,
        "neon_yellow": NEON_YELLOW_GLOW,
        "light_blue": BLUE_LIGHT_GLOW,
        "medium_blue": BLUE_MED_GLOW,
        "dark_blue": BLUE_DARK_GLOW,
    }
    paints = []
    for n in names:
        if n in mapping:
            paints.append(mapping[n])
    return paints


def apply_multi_color_overlay(sign_rgba: Image.Image, mode: str, env: StopSignGridEnv, paints: List[UVPaint], rng) -> Image.Image:
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
            mask = Image.fromarray(arr, mode="L")
        rgb.paste(color, mask=mask)

    return Image.merge("RGBA", (*rgb.split(), a))


def eval_overlay_over_K_custom(env: StopSignGridEnv, seeds: List[int], over_day: Image.Image, over_on: Image.Image) -> dict:
    imgs_over_day, imgs_over_on = [], []
    for t_seed in seeds:
        over_day_t = env._compose_on_bg(env._transform_sign(over_day, t_seed), env._place_seed)
        over_on_t = env._compose_on_bg(env._transform_sign(over_on, t_seed), env._place_seed)
        imgs_over_day.append(over_day_t)
        imgs_over_on.append(over_on_t)

    c_day_list = env.det.infer_confidence_batch(imgs_over_day)
    c_on_list = env.det.infer_confidence_batch(imgs_over_on)

    det_on = env.det.infer_detections_batch(imgs_over_on)
    iou_vals = []
    misclass_vals = []
    target_conf_vals = []
    top_conf_vals = []
    top_class_counts = {}
    for det in det_on:
        target_conf = float(det.get("target_conf", 0.0))
        top_conf = float(det.get("top_conf", 0.0))
        top_class = det.get("top_class", None)
        target_box = det.get("target_box", None)
        top_box = det.get("top_box", None)
        misclass = (top_class is not None and top_class != env.det.target_id and top_conf > 0.0)
        misclass_vals.append(1.0 if misclass else 0.0)
        target_conf_vals.append(target_conf)
        top_conf_vals.append(top_conf)
        if top_class is not None:
            cls = int(top_class)
            top_class_counts[cls] = top_class_counts.get(cls, 0) + 1

        iou = 0.0
        if target_box is not None and top_box is not None:
            iou = _iou_xyxy(target_box, top_box)
        iou_vals.append(float(iou))

    mean = lambda xs: float(np.mean(xs)) if len(xs) else 0.0
    return {
        "c_day": mean(c_day_list),
        "c_on": mean(c_on_list),
        "mean_iou": mean(iou_vals),
        "misclass_rate": mean(misclass_vals),
        "mean_target_conf": mean(target_conf_vals),
        "mean_top_conf": mean(top_conf_vals),
        "top_class_counts": top_class_counts,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Sweep patch area coverage vs YOLO confidence.")
    p.add_argument("--data", default="./data")
    p.add_argument("--bgdir", default="./data/backgrounds")
    p.add_argument("--yolo", default="./weights/yolo8n.pt")
    p.add_argument("--device", default="cuda")
    p.add_argument("--grid-cell", type=int, default=16, choices=[2, 4, 8, 16, 32])
    p.add_argument("--cell-cover-thresh", type=float, default=0.60,
                   help="Grid cell coverage threshold (0..1). Lower covers edges.")
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
    p.add_argument(
        "--paint-list",
        default="neon_yellow,orange,red,light_blue,medium_blue,dark_blue,purple,green",
        help="Comma-separated paint names to test.",
    )
    p.add_argument("--combo-sizes", default="1",
                   help="Comma-separated combo sizes to test (e.g., 1,2,3).")
    p.add_argument("--combo-limit", type=int, default=0,
                   help="Limit number of color combos (0 = no limit).")
    p.add_argument("--save", action="store_true", help="Save sample images for each percentage.")
    p.add_argument("--out", default="./_debug_area_sweep")
    p.add_argument("--log", default="./_debug_area_sweep/area_sweep.ndjson")
    p.add_argument("--progress-every", type=int, default=200,
                   help="Print progress/ETA every N evaluations.")
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
    paint_names = parse_paint_list(args.paint_list)
    paints = resolve_paints(paint_names)
    if not paints:
        paints = [GREEN_GLOW]
    combo_sizes = parse_int_list(args.combo_sizes)
    combo_sizes = [s for s in combo_sizes if s > 0]
    if not combo_sizes:
        combo_sizes = [1]
    combos = []
    for k in combo_sizes:
        for combo in itertools.combinations(paints, min(k, len(paints))):
            combos.append(combo)

    if args.combo_limit and len(combos) > int(args.combo_limit):
        rng_tmp = np.random.default_rng(int(args.seed))
        picks = rng_tmp.choice(len(combos), size=int(args.combo_limit), replace=False)
        combos = [combos[int(i)] for i in picks]

    for eval_k in k_list:
        rng = np.random.default_rng(int(args.seed))

        bg_samples = max(1, int(args.bg_samples))
        pos_samples = max(1, int(args.pos_samples))
        bg_count = min(bg_samples, len(bgs))
        bg_indices = rng.choice(len(bgs), size=bg_count, replace=False)

        if args.save:
            os.makedirs(args.out, exist_ok=True)
        os.makedirs(os.path.dirname(args.log), exist_ok=True)

        total_runs = None
        done_runs = 0
        t0 = time.perf_counter()
        last_print = t0

        for combo in combos:
            combo_names = "+".join([p.name for p in combo])
            env = StopSignGridEnv(
                stop_sign_image=stop_day,
                stop_sign_uv_image=stop_uv,
                background_images=bgs,
                pole_image=None if args.no_pole else pole,
                yolo_weights=args.yolo,
                yolo_device=args.device,
                eval_K=int(eval_k),
                grid_cell_px=int(args.grid_cell),
                uv_paint=combo[0],
                use_single_color=True,
                cell_cover_thresh=float(args.cell_cover_thresh),
            )

            obs, _ = env.reset()
            if args.save:
                os.makedirs(args.out, exist_ok=True)
                Image.fromarray(obs).save(os.path.join(args.out, f"obs_day_baseline_k{eval_k}_{combo_names}.png"))

            valid_coords = env._valid_coords  # (N,2)
            total = int(valid_coords.shape[0])
            if total <= 0:
                print("No valid cells in mask.")
                return
            if total_runs is None:
                total_runs = 0
                for pct in percents:
                    target_n = int(math.ceil(float(pct) * total)) if percents else 0
                    trials = 1 if target_n == 0 else max(1, int(args.trials))
                    total_runs += bg_count * pos_samples * trials
                total_runs = max(1, total_runs) * max(1, len(combos))

            print(
                f"\nvalid_cells={total} | eval_K={eval_k} | trials={int(args.trials)} | "
                f"bg_samples={bg_count} pos_samples={pos_samples} full_cover={bool(args.full_cover)} | "
                f"paint_combo={combo_names} cell_thresh={float(args.cell_cover_thresh):.2f}"
            )

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
                            over_day = apply_multi_color_overlay(env.sign_rgba_day, "day", env, list(combo), rng)
                            over_on = apply_multi_color_overlay(env.sign_rgba_on, "on", env, list(combo), rng)
                            metrics = eval_overlay_over_K_custom(env, eval_seeds, over_day, over_on)
                            c_on = float(metrics.get("c_on", 0.0))
                            c_day = float(metrics.get("c_day", 0.0))
                            mean_iou = float(metrics.get("mean_iou", 0.0))
                            misclass_rate = float(metrics.get("misclass_rate", 0.0))
                            c0_day = float(env._mean_over_K(env._baseline_c0_day_list, int(eval_k)))
                            drop_on = float(c0_day - c_on)

                            rows.append((area_frac, c_on, c_day, drop_on, mean_iou, misclass_rate))

                            rec = {
                                "eval_k": int(eval_k),
                                "paint_combo": combo_names,
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
                                "cell_cover_thresh": float(args.cell_cover_thresh),
                            }
                            with open(args.log, "a", encoding="utf-8") as f:
                                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

                            if args.save and c_on >= float(args.high_conf):
                                stem = (
                                    f"area_{int(area_frac*100):03d}_k{eval_k}_bg{int(bg_idx)}_p{int(env._place_seed)}"
                                    f"_{combo_names}"
                                )
                                preview = env._compose_on_bg(env._transform_sign(over_on, env._transform_seeds[0]), env._place_seed)
                                overlay = apply_multi_color_overlay(env.sign_rgba_on, "on", env, list(combo), rng)
                                preview.save(os.path.join(args.out, f"{stem}_uv_on.png"))
                                overlay.save(os.path.join(args.out, f"{stem}_overlay.png"))

                            done_runs += 1
                            if int(args.progress_every) > 0 and (done_runs % int(args.progress_every) == 0):
                                now = time.perf_counter()
                                elapsed = now - t0
                                rate = done_runs / max(elapsed, 1e-9)
                                remaining = max(total_runs - done_runs, 0)
                                eta = remaining / max(rate, 1e-9)
                                if now - last_print >= 1.0:
                                    print(f"[progress] {done_runs}/{total_runs} | {rate:.2f} runs/s | ETA {eta/60.0:.1f} min")
                                    last_print = now

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
                    stem = f"area_{int(mean[0]*100):03d}_k{eval_k}_mean_{combo_names}"
                    preview = env._compose_on_bg(env._transform_sign(over_on, env._transform_seeds[0]), env._place_seed)
                    overlay = apply_multi_color_overlay(env.sign_rgba_on, "on", env, list(combo), rng)
                    preview.save(os.path.join(args.out, f"{stem}_uv_on.png"))
                    overlay.save(os.path.join(args.out, f"{stem}_overlay.png"))


if __name__ == "__main__":
    main()
