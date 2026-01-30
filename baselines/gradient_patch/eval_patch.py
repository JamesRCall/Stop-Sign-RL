"""Evaluate a saved gradient patch using the PPO detector path."""
from __future__ import annotations

import os
import sys
import json
import argparse
import random
import numpy as np
from PIL import Image
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from detectors.yolo_wrapper import DetectorWrapper
from baselines.gradient_patch.renderer import (
    pil_to_tensor_rgb,
    pil_to_tensor_rgba,
    pil_to_tensor_gray,
    tensor_to_pil_rgb,
    sample_transform_params,
    apply_transform_rgba,
    compose_sign_and_pole,
    sample_placement,
    place_group_on_background,
    alpha_composite_rgba,
)


def build_backgrounds(bg_mode: str, folder: str, size):
    mode = str(bg_mode or "dataset").strip().lower()
    if mode == "solid":
        colors = [(200, 200, 200), (120, 120, 120), (30, 30, 30)]
        W, H = int(size[0]), int(size[1])
        return [pil_to_tensor_rgb(Image.new("RGB", (W, H), c)) for c in colors]
    # dataset
    paths = sorted([os.path.join(folder, p) for p in os.listdir(folder)])
    imgs = []
    for p in paths:
        if not os.path.isfile(p):
            continue
        try:
            imgs.append(pil_to_tensor_rgb(Image.open(p).convert("RGB")))
        except Exception:
            continue
        if len(imgs) >= 20:
            break
    if not imgs:
        raise FileNotFoundError(f"No backgrounds found in: {folder}")
    return imgs


def main() -> None:
    ap = argparse.ArgumentParser("Evaluate saved gradient patch with PPO detector path")
    ap.add_argument("--run", required=True, help="Path to gradient run folder.")
    ap.add_argument("--data", default="./data")
    ap.add_argument("--bgdir", default="./data/backgrounds")
    ap.add_argument("--bg-mode", choices=["dataset", "solid"], default="dataset")
    ap.add_argument("--no-pole", action="store_true")
    ap.add_argument("--weights", default="./weights/yolo8n.pt")
    ap.add_argument("--detector-device", default="auto")
    ap.add_argument("--eval-k", type=int, default=10)
    ap.add_argument("--transform-strength", type=float, default=1.0)
    ap.add_argument("--img-size", type=int, default=640)
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    rng = random.Random(int(args.seed))
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))

    # Load assets
    stop_plain = Image.open(os.path.join(args.data, "stop_sign.png")).convert("RGBA")
    stop_uv_path = os.path.join(args.data, "stop_sign_uv.png")
    stop_uv = Image.open(stop_uv_path).convert("RGBA") if os.path.exists(stop_uv_path) else stop_plain.copy()
    pole_path = os.path.join(args.data, "pole.png")
    pole_img = Image.open(pole_path).convert("RGBA") if (os.path.exists(pole_path) and not args.no_pole) else None

    sign_rgba = pil_to_tensor_rgba(stop_plain)
    sign_uv_rgba = pil_to_tensor_rgba(stop_uv)
    pole_rgba = pil_to_tensor_rgba(pole_img) if pole_img is not None else None

    overlay_day_path = os.path.join(args.run, "overlay_day.png")
    overlay_on_path = os.path.join(args.run, "overlay_on.png")
    if not (os.path.exists(overlay_day_path) and os.path.exists(overlay_on_path)):
        raise FileNotFoundError("Missing overlay_day.png / overlay_on.png in run folder.")

    overlay_day = pil_to_tensor_rgba(Image.open(overlay_day_path).convert("RGBA"))
    overlay_on = pil_to_tensor_rgba(Image.open(overlay_on_path).convert("RGBA"))

    # Compose overlays onto sign (RGBA)
    sign_day = alpha_composite_rgba(sign_rgba, overlay_day)
    sign_on = alpha_composite_rgba(sign_uv_rgba, overlay_on)

    sign_size = (int(sign_rgba.shape[2]), int(sign_rgba.shape[1]))
    img_size = (int(args.img_size), int(args.img_size))
    backgrounds = build_backgrounds(args.bg_mode, args.bgdir, img_size)

    det = DetectorWrapper(args.weights, device=args.detector_device, conf=0.10, iou=0.45, debug=False)

    c0_day_list = []
    c_day_list = []
    c_on_list = []

    for _ in range(int(args.eval_k)):
        params = sample_transform_params(rng, args.transform_strength, sign_size)
        t_plain = apply_transform_rgba(sign_rgba, params, sign_size)
        t_day = apply_transform_rgba(sign_day, params, sign_size)
        t_on = apply_transform_rgba(sign_on, params, sign_size)

        group_plain = compose_sign_and_pole(t_plain, pole_rgba)
        group_day = compose_sign_and_pole(t_day, pole_rgba)
        group_on = compose_sign_and_pole(t_on, pole_rgba)

        bg = backgrounds[rng.randint(0, len(backgrounds) - 1)]
        placement = sample_placement(rng, group_plain, img_size)
        img_plain = place_group_on_background(group_plain, bg, rng, img_size, placement=placement)
        img_day = place_group_on_background(group_day, bg, rng, img_size, placement=placement)
        img_on = place_group_on_background(group_on, bg, rng, img_size, placement=placement)

        c0_day_list.append(det.infer_confidence(tensor_to_pil_rgb(img_plain)))
        c_day_list.append(det.infer_confidence(tensor_to_pil_rgb(img_day)))
        c_on_list.append(det.infer_confidence(tensor_to_pil_rgb(img_on)))

    c0_day = float(np.mean(c0_day_list)) if c0_day_list else 0.0
    c_day = float(np.mean(c_day_list)) if c_day_list else 0.0
    c_on = float(np.mean(c_on_list)) if c_on_list else 0.0
    drop_day = float(c0_day - c_day)
    drop_on = float(c0_day - c_on)

    # Area fraction from patch mask
    mask_path = os.path.join(args.run, "patch_mask.png")
    area_frac = None
    if os.path.exists(mask_path):
        mask = pil_to_tensor_gray(Image.open(mask_path)).float()
        sign_alpha = sign_rgba[3:4]
        sign_mask = (sign_alpha > 0.0).float()
        area_frac = float((mask * sign_mask).sum() / sign_mask.sum().clamp(min=1.0))

    out = {
        "c0_day": c0_day,
        "c_day": c_day,
        "c_on": c_on,
        "drop_day": drop_day,
        "drop_on": drop_on,
        "area_frac": area_frac,
        "eval_k": int(args.eval_k),
        "weights": args.weights,
        "transform_strength": float(args.transform_strength),
        "bg_mode": args.bg_mode,
        "no_pole": bool(args.no_pole),
    }

    out_path = os.path.join(args.run, "eval_metrics.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print(json.dumps(out, indent=2))
    print(f"[DONE] Wrote {out_path}")


if __name__ == "__main__":
    main()
