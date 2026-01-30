"""Gradient-based patch optimization baseline (separate from PPO).

This script builds a differentiable renderer, runs EOT over transforms,
and optimizes a patch constrained to the UV paint palette.
"""
from __future__ import annotations

import os
import sys
import json
import time
import argparse
import random
from typing import List, Tuple
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from utils.uv_paint import (
    WHITE_GLOW,
    RED_GLOW,
    GREEN_GLOW,
    YELLOW_GLOW,
    BLUE_GLOW,
    ORANGE_GLOW,
    UVPaint,
)
from baselines.gradient_patch.yolo_grad import YoloGrad
from baselines.gradient_patch.renderer import (
    pil_to_tensor_rgb,
    pil_to_tensor_rgba,
    tensor_to_pil_rgb,
    tensor_to_pil_gray,
    apply_patch,
    sample_transform_params,
    apply_transform_rgba,
    compose_sign_and_pole,
    sample_placement,
    place_group_on_background,
)


def resolve_palette(name: str) -> List[UVPaint]:
    name = str(name or "uv6").strip().lower()
    if name == "yellow":
        return [YELLOW_GLOW]
    if name == "uv6":
        return [WHITE_GLOW, RED_GLOW, GREEN_GLOW, YELLOW_GLOW, BLUE_GLOW, ORANGE_GLOW]
    raise ValueError(f"Unknown palette '{name}'. Use 'uv6' or 'yellow'.")


def load_backgrounds(folder: str, max_count: int = 20) -> List[torch.Tensor]:
    paths = sorted([os.path.join(folder, p) for p in os.listdir(folder)])
    imgs = []
    for p in paths:
        if not os.path.isfile(p):
            continue
        try:
            imgs.append(pil_to_tensor_rgb(Image.open(p).convert("RGB")))
        except Exception:
            continue
        if len(imgs) >= max_count:
            break
    if not imgs:
        raise FileNotFoundError(f"No backgrounds found in: {folder}")
    return imgs


def build_solid_backgrounds(size: Tuple[int, int]) -> List[torch.Tensor]:
    from PIL import Image
    colors = [(200, 200, 200), (120, 120, 120), (30, 30, 30)]
    W, H = int(size[0]), int(size[1])
    return [pil_to_tensor_rgb(Image.new("RGB", (W, H), c)) for c in colors]


def build_backgrounds(bg_mode: str, folder: str, size: Tuple[int, int]) -> List[torch.Tensor]:
    mode = str(bg_mode or "dataset").strip().lower()
    if mode == "solid":
        return build_solid_backgrounds(size)
    return load_backgrounds(folder)


def save_outputs(out_dir: str, name: str, img_rgb: torch.Tensor) -> None:
    os.makedirs(out_dir, exist_ok=True)
    pil = tensor_to_pil_rgb(img_rgb)
    pil.save(os.path.join(out_dir, name))


def main() -> None:
    ap = argparse.ArgumentParser("Gradient patch baseline (YOLO)")
    ap.add_argument("--data", default="./data")
    ap.add_argument("--bgdir", default="./data/backgrounds")
    ap.add_argument("--bg-mode", choices=["dataset", "solid"], default="dataset")
    ap.add_argument("--no-pole", action="store_true")
    ap.add_argument("--weights", default="./weights/yolo8n.pt")
    ap.add_argument("--device", default="auto")

    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=3e-2)
    ap.add_argument("--eot-k", type=int, default=4)
    ap.add_argument("--transform-strength", type=float, default=1.0)
    ap.add_argument("--img-size", type=int, default=640)
    ap.add_argument("--mask-res", type=int, default=64)
    ap.add_argument("--seed", type=int, default=123)

    ap.add_argument("--palette", choices=["uv6", "yellow"], default="uv6")
    ap.add_argument("--color-mode", choices=["global", "pixel"], default="global",
                    help="global = single paint for whole patch; pixel = per-pixel mixture")
    ap.add_argument("--palette-temp", type=float, default=0.3,
                    help="Softmax temperature for palette mixing (lower = harder).")

    ap.add_argument("--area-target", type=float, default=0.25)
    ap.add_argument("--lambda-area", type=float, default=0.70)
    ap.add_argument("--lambda-day", type=float, default=0.0)
    ap.add_argument("--day-tolerance", type=float, default=0.05)
    ap.add_argument("--lambda-tv", type=float, default=2e-4)
    ap.add_argument("--area-cap-frac", type=float, default=0.0,
                    help="Optional soft cap penalty; <=0 disables.")
    ap.add_argument("--area-cap-penalty", type=float, default=0.20)
    ap.add_argument("--lambda-efficiency", type=float, default=0.0)
    ap.add_argument("--efficiency-eps", type=float, default=0.02)

    ap.add_argument("--uv-min-alpha", type=float, default=0.08)
    ap.add_argument("--save-every", type=int, default=200)
    ap.add_argument("--out", default="./baselines/gradient_patch/_runs")
    args = ap.parse_args()

    rng = random.Random(int(args.seed))
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))

    device = args.device
    if device == "auto":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        device = "cuda:0"

    # Load assets
    stop_plain = Image.open(os.path.join(args.data, "stop_sign.png")).convert("RGBA")
    stop_uv_path = os.path.join(args.data, "stop_sign_uv.png")
    stop_uv = Image.open(stop_uv_path).convert("RGBA") if os.path.exists(stop_uv_path) else stop_plain.copy()
    pole_path = os.path.join(args.data, "pole.png")
    pole_img = Image.open(pole_path).convert("RGBA") if (os.path.exists(pole_path) and not args.no_pole) else None

    sign_rgba = pil_to_tensor_rgba(stop_plain).to(device)
    sign_uv_rgba = pil_to_tensor_rgba(stop_uv).to(device)
    pole_rgba = pil_to_tensor_rgba(pole_img).to(device) if pole_img is not None else None

    sign_alpha = sign_rgba[3:4].clamp(0.0, 1.0)
    sign_mask = (sign_alpha > 0.0).float()
    sign_area = sign_mask.sum().clamp(min=1.0)

    img_size = (int(args.img_size), int(args.img_size))
    sign_size = (int(sign_rgba.shape[2]), int(sign_rgba.shape[1]))
    backgrounds = [b.to(device) for b in build_backgrounds(args.bg_mode, args.bgdir, img_size)]

    # Palette
    palette = resolve_palette(args.palette)
    day_colors = torch.tensor([p.day_rgb for p in palette], dtype=torch.float32, device=device) / 255.0
    on_colors = torch.tensor([p.active_rgb for p in palette], dtype=torch.float32, device=device) / 255.0
    day_alpha = float(palette[0].day_alpha)
    on_alpha = float(palette[0].active_alpha)

    # Params
    mask_logits = torch.nn.Parameter(torch.zeros((1, 1, args.mask_res, args.mask_res), device=device))
    if len(palette) > 1:
        if args.color_mode == "pixel":
            color_logits = torch.nn.Parameter(torch.zeros((1, len(palette), args.mask_res, args.mask_res), device=device))
        else:
            color_logits = torch.nn.Parameter(torch.zeros((len(palette),), device=device))
    else:
        color_logits = None

    opt_params = [mask_logits] + ([color_logits] if color_logits is not None else [])
    optimizer = torch.optim.Adam(opt_params, lr=float(args.lr))

    yolo = YoloGrad(args.weights, target_class="stop sign", device=device)

    run_id = time.strftime("grad_%Y%m%d_%H%M%S")
    out_dir = os.path.join(args.out, run_id)
    os.makedirs(out_dir, exist_ok=True)

    def get_mask() -> torch.Tensor:
        mask = torch.sigmoid(mask_logits)
        mask = F.interpolate(mask, size=sign_rgba.shape[1:], mode="bilinear", align_corners=False)
        mask = mask[0]  # (1, H, W)
        return (mask * sign_mask).clamp(0.0, 1.0)

    def get_colors(mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if len(palette) == 1:
            day = day_colors[0].view(3, 1, 1).expand_as(sign_rgba[:3])
            on = on_colors[0].view(3, 1, 1).expand_as(sign_rgba[:3])
            return day, on

        if args.color_mode == "pixel":
            logits = color_logits
            logits = F.interpolate(logits, size=sign_rgba.shape[1:], mode="bilinear", align_corners=False)
            weights = F.softmax(logits / max(args.palette_temp, 1e-6), dim=1)
            day = torch.einsum("bphw,pc->bchw", weights, day_colors)[0]
            on = torch.einsum("bphw,pc->bchw", weights, on_colors)[0]
        else:
            weights = F.softmax(color_logits / max(args.palette_temp, 1e-6), dim=0)
            day = (weights[:, None] * day_colors).sum(dim=0).view(3, 1, 1).expand_as(sign_rgba[:3])
            on = (weights[:, None] * on_colors).sum(dim=0).view(3, 1, 1).expand_as(sign_rgba[:3])
        return day, on

    def area_frac(mask: torch.Tensor) -> torch.Tensor:
        return (mask * sign_mask).sum() / sign_area

    metrics = []

    for step in range(1, int(args.steps) + 1):
        optimizer.zero_grad()

        # Sample background and shared placement
        bg = backgrounds[rng.randint(0, len(backgrounds) - 1)]

        mask = get_mask()
        day_color, on_color = get_colors(mask)

        # Build patched sign variants
        sign_day = apply_patch(sign_rgba, sign_alpha, mask, day_color, day_alpha, args.uv_min_alpha, False)
        sign_on = apply_patch(sign_uv_rgba, sign_alpha, mask, on_color, on_alpha, args.uv_min_alpha, True)

        # Plain sign (no patch)
        plain_day = sign_rgba

        # EOT batch
        imgs_plain = []
        imgs_day = []
        imgs_on = []
        for _ in range(int(args.eot_k)):
            params = sample_transform_params(rng, args.transform_strength, sign_size)
            t_plain = apply_transform_rgba(plain_day, params, sign_size)
            t_day = apply_transform_rgba(sign_day, params, sign_size)
            t_on = apply_transform_rgba(sign_on, params, sign_size)

            group_plain = compose_sign_and_pole(t_plain, pole_rgba)
            group_day = compose_sign_and_pole(t_day, pole_rgba)
            group_on = compose_sign_and_pole(t_on, pole_rgba)

            placement = sample_placement(rng, group_plain, img_size)
            img_plain = place_group_on_background(group_plain, bg, rng, img_size, placement=placement)
            img_day = place_group_on_background(group_day, bg, rng, img_size, placement=placement)
            img_on = place_group_on_background(group_on, bg, rng, img_size, placement=placement)

            imgs_plain.append(img_plain)
            imgs_day.append(img_day)
            imgs_on.append(img_on)

        batch_plain = torch.stack(imgs_plain, dim=0)
        batch_day = torch.stack(imgs_day, dim=0)
        batch_on = torch.stack(imgs_on, dim=0)

        c0_day = yolo.target_conf(batch_plain).mean().detach()
        c_day = yolo.target_conf(batch_day).mean()
        c_on = yolo.target_conf(batch_on).mean()

        drop_day = c0_day - c_day
        drop_on = c0_day - c_on

        area = area_frac(mask)
        pen_day = torch.relu(drop_day - float(args.day_tolerance))

        excess_penalty = 0.0
        if args.area_target and args.area_target > 0:
            excess = torch.relu(area - float(args.area_target))
            excess_penalty = float(args.lambda_area) * (4.5 * excess + excess * excess)

        cap_pen = 0.0
        if args.area_cap_frac and args.area_cap_frac > 0:
            cap_excess = torch.relu(area - float(args.area_cap_frac))
            cap_pen = float(args.area_cap_penalty) * (1.0 + 2.0 * cap_excess)

        efficiency = torch.log1p(torch.relu(drop_on) / max(float(args.efficiency_eps), 1e-6))

        # Total loss (minimize)
        loss = (
            -drop_on
            + float(args.lambda_day) * pen_day
            + float(args.lambda_area) * area
            + excess_penalty
            + cap_pen
            - float(args.lambda_efficiency) * efficiency
        )

        # Total variation for smoothness
        tv = torch.mean(torch.abs(mask[:, 1:, :] - mask[:, :-1, :])) + torch.mean(torch.abs(mask[:, :, 1:] - mask[:, :, :-1]))
        loss = loss + float(args.lambda_tv) * tv

        loss.backward()
        optimizer.step()

        if step % 20 == 0 or step == 1:
            m = {
                "step": step,
                "loss": float(loss.item()),
                "c0_day": float(c0_day.item()),
                "c_day": float(c_day.item()),
                "c_on": float(c_on.item()),
                "drop_day": float(drop_day.item()),
                "drop_on": float(drop_on.item()),
                "area": float(area.item()),
            }
            metrics.append(m)
            print(f"[{step}/{args.steps}] loss={m['loss']:.4f} drop_on={m['drop_on']:.4f} area={m['area']:.4f}")

        if args.save_every > 0 and (step % int(args.save_every) == 0 or step == int(args.steps)):
            with torch.no_grad():
                # Save overlay previews
                save_outputs(out_dir, f"preview_day_{step:06d}.png", batch_day[0])
                save_outputs(out_dir, f"preview_on_{step:06d}.png", batch_on[0])

    # Save final assets
    with torch.no_grad():
        mask = get_mask()
        day_color, on_color = get_colors(mask)
        sign_day = apply_patch(sign_rgba, sign_alpha, mask, day_color, day_alpha, args.uv_min_alpha, False)
        sign_on = apply_patch(sign_uv_rgba, sign_alpha, mask, on_color, on_alpha, args.uv_min_alpha, True)
        save_outputs(out_dir, "sign_day.png", sign_day[:3])
        save_outputs(out_dir, "sign_on.png", sign_on[:3])
        tensor_to_pil_gray(mask).save(os.path.join(out_dir, "patch_mask.png"))

    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"[DONE] Saved outputs to {out_dir}")


if __name__ == "__main__":
    main()
