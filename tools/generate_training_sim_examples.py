#!/usr/bin/env python3
"""Generate training-simulation visuals with matched DAY/UV transforms.

Creates:
1) A stop-sign panel with octagon-aware grid overlay.
2) Four DAY/UV render pairs where each pair uses the exact same background,
   placement, and transform parameters (angle/scale/etc.).
3) A composite figure suitable for paper drafting.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont


def _load_font(size: int):
    for name in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            pass
    return ImageFont.load_default()


def _affine_matrix(angle_deg: float, shear_deg: float, scale: float, tx: float, ty: float, w: int, h: int):
    angle = math.radians(angle_deg)
    shear = math.radians(shear_deg)
    cos_a, sin_a = math.cos(angle) * scale, math.sin(angle) * scale
    a = cos_a + (-sin_a) * math.tan(shear)
    b = sin_a + cos_a * math.tan(shear)
    c = tx
    d = -sin_a + cos_a * math.tan(shear)
    e = cos_a + sin_a * math.tan(shear)
    f = ty
    cx, cy = w / 2.0, h / 2.0
    c += cx - (a * cx + b * cy)
    f += cy - (d * cx + e * cy)
    return (a, b, c, d, e, f)


def _random_perspective_coeffs(w: int, h: int, rng: np.random.Generator, max_shift: float = 0.06):
    dx, dy = w * max_shift, h * max_shift
    src = [(0, 0), (w, 0), (w, h), (0, h)]
    dst = [
        (rng.uniform(-dx, dx), rng.uniform(-dy, dy)),
        (w + rng.uniform(-dx, dx), rng.uniform(-dy, dy)),
        (w + rng.uniform(-dx, dx), h + rng.uniform(-dy, dy)),
        (rng.uniform(-dx, dx), h + rng.uniform(-dy, dy)),
    ]
    a = []
    for (x, y), (u, v) in zip(src, dst):
        a.extend([[x, y, 1, 0, 0, 0, -u * x, -u * y], [0, 0, 0, x, y, 1, -v * x, -v * y]])
    a = np.array(a, dtype=np.float32)
    b = np.array([p for uv in dst for p in uv], dtype=np.float32)
    coeffs = np.linalg.lstsq(a, b, rcond=None)[0]
    return [float(x) for x in coeffs]


def _sample_transform_params(
    seed: int,
    w: int,
    h: int,
    strength: float,
    fixed_angle_deg: float | None,
    geom_gain: float = 1.0,
    photo_gain: float = 1.0,
    force_photometric: bool = False,
) -> dict[str, Any]:
    rng = np.random.default_rng(int(seed))
    # Keep parity with StopSignGridEnv._transform_sign():
    # strength is clamped to [0, 1] before use.
    s_base = max(0.0, min(1.0, float(strength)))
    s_geo = max(0.0, min(1.0, s_base * float(geom_gain)))
    s_photo = max(0.0, min(1.0, s_base * float(photo_gain)))
    if fixed_angle_deg is None and s_geo <= 0.0 and s_photo <= 0.0:
        return {
            "angle_deg": 0.0,
            "shear_deg": 0.0,
            "scale": 1.0,
            "tx": 0.0,
            "ty": 0.0,
            "perspective": None,
            "brightness": None,
            "contrast": None,
            "color": None,
            "blur_radius": None,
            "noise_sigma": None,
            "noise_seed": None,
        }

    angle = float(fixed_angle_deg) if fixed_angle_deg is not None else float(rng.uniform(-6.0 * s_geo, 6.0 * s_geo))
    shear = float(rng.uniform(-4.0 * s_geo, 4.0 * s_geo)) if s_geo > 0.0 else 0.0
    scale = float(1.0 + rng.uniform(-0.10 * s_geo, 0.10 * s_geo)) if s_geo > 0.0 else 1.0
    tx = float(rng.uniform(-0.02 * s_geo * w, 0.02 * s_geo * w)) if s_geo > 0.0 else 0.0
    ty = float(rng.uniform(-0.02 * s_geo * h, 0.02 * s_geo * h)) if s_geo > 0.0 else 0.0

    perspective = None
    if s_geo > 0.0 and rng.random() < (0.5 * s_geo):
        perspective = _random_perspective_coeffs(w, h, rng, max_shift=0.06 * s_geo)

    brightness = None
    if s_photo > 0.0 and (force_photometric or rng.random() < (0.7 * s_photo)):
        bspan = 0.10 * s_photo
        brightness = float(rng.uniform(1.0 - bspan, 1.0 + bspan))

    contrast = None
    if s_photo > 0.0 and (force_photometric or rng.random() < (0.7 * s_photo)):
        cspan = 0.10 * s_photo
        contrast = float(rng.uniform(1.0 - cspan, 1.0 + cspan))

    color = None
    if s_photo > 0.0 and (force_photometric or rng.random() < (0.3 * s_photo)):
        kspan = 0.10 * s_photo
        color = float(rng.uniform(1.0 - kspan, 1.0 + kspan))

    blur_radius = None
    if s_photo > 0.0 and (force_photometric or rng.random() < (0.4 * s_photo)):
        blur_radius = float(rng.uniform(0.0, 0.8 * s_photo))

    noise_sigma = None
    noise_seed = None
    if s_photo > 0.0 and (force_photometric or rng.random() < (0.6 * s_photo)):
        noise_sigma = float(rng.uniform(1.0 * s_photo, 3.0 * s_photo))
        noise_seed = int(rng.integers(0, 2**31 - 1))

    return {
        "angle_deg": angle,
        "shear_deg": shear,
        "scale": scale,
        "tx": tx,
        "ty": ty,
        "perspective": perspective,
        "brightness": brightness,
        "contrast": contrast,
        "color": color,
        "blur_radius": blur_radius,
        "noise_sigma": noise_sigma,
        "noise_seed": noise_seed,
    }


def _apply_transform(sign_rgba: Image.Image, p: dict[str, Any]) -> Image.Image:
    w, h = sign_rgba.size
    out = sign_rgba.copy()
    aff = _affine_matrix(float(p["angle_deg"]), float(p["shear_deg"]), float(p["scale"]), float(p["tx"]), float(p["ty"]), w, h)
    out = out.transform((w, h), Image.AFFINE, data=aff, resample=Image.BILINEAR, fillcolor=(0, 0, 0, 0))
    if p.get("perspective") is not None:
        out = out.transform((w, h), Image.PERSPECTIVE, p["perspective"], resample=Image.BILINEAR, fillcolor=(0, 0, 0, 0))

    rgb, a = out.convert("RGB"), out.split()[-1]
    if p.get("brightness") is not None:
        rgb = ImageEnhance.Brightness(rgb).enhance(float(p["brightness"]))
    if p.get("contrast") is not None:
        rgb = ImageEnhance.Contrast(rgb).enhance(float(p["contrast"]))
    if p.get("color") is not None:
        rgb = ImageEnhance.Color(rgb).enhance(float(p["color"]))
    if p.get("blur_radius") is not None:
        rgb = rgb.filter(ImageFilter.GaussianBlur(radius=float(p["blur_radius"])))
    if p.get("noise_sigma") is not None and float(p["noise_sigma"]) > 0.0:
        arr = np.array(rgb, dtype=np.int16)
        nseed = p.get("noise_seed", 0)
        rng = np.random.default_rng(int(nseed) if nseed is not None else 0)
        noise = rng.normal(0.0, float(p["noise_sigma"]), size=arr.shape)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        rgb = Image.fromarray(arr, mode="RGB")
    return Image.merge("RGBA", (*rgb.split(), a))


def _compose_sign_and_pole(
    sign_rgba: Image.Image,
    pole_rgba: Image.Image | None,
    pole_width_ratio: float = 0.12,
    bottom_len_factor: float = 4.0,
    clearance_px: int = 2,
    side_margin_frac: float = 0.06,
) -> Image.Image:
    if pole_rgba is None:
        return sign_rgba.copy()
    sign = sign_rgba.copy()
    sw, sh = sign.size
    pole = pole_rgba.convert("RGBA").copy()
    pw0, ph0 = pole.size

    target_pw = max(2, int(pole_width_ratio * sw))
    scale_w = target_pw / max(1, pw0)
    target_ph = max(1, int(ph0 * scale_w))
    pole = pole.resize((target_pw, target_ph), Image.BILINEAR)

    h_needed = clearance_px + sh + int(bottom_len_factor * sh)
    if pole.height < h_needed:
        scale_h = h_needed / pole.height
        pole = pole.resize((pole.width, int(pole.height * scale_h)), Image.BILINEAR)
    pole = pole.crop((0, 0, pole.width, h_needed))

    side_margin = int(side_margin_frac * sw)
    gw = max(pole.width, sw + 2 * side_margin)
    gh = h_needed
    group = Image.new("RGBA", (gw, gh), (0, 0, 0, 0))

    px = (gw - pole.width) // 2
    group.alpha_composite(pole, (px, 0))
    sx = (gw - sw) // 2
    sy = clearance_px
    group.alpha_composite(sign, (sx, sy))
    return group


def _place_group_on_background(group_rgba: Image.Image, bg_rgb: Image.Image, seed: int, img_size: tuple[int, int]) -> Image.Image:
    rng = np.random.default_rng(int(seed))
    w, h = img_size
    bg = bg_rgb.resize((w, h), Image.BILINEAR).convert("RGBA")

    target_w = int(rng.uniform(0.30 * w, 0.50 * w))
    scale = target_w / max(1, group_rgba.width)
    group = group_rgba.resize((target_w, int(group_rgba.height * scale)), Image.BILINEAR)

    margin = int(0.04 * w)
    max_x = max(margin, w - group.width - margin)
    max_y = max(margin, h - group.height - margin)
    left_max = max(margin, min(max_x, int(0.40 * w)))
    right_min = max(margin, min(max_x, int(0.60 * w)))
    if rng.random() < 0.5:
        x = int(rng.integers(margin, left_max + 1))
    else:
        x = int(rng.integers(right_min, max_x + 1))
    min_y = max(margin, int(0.12 * h))
    if min_y > max_y:
        min_y = max(margin, max_y)
    y = int(rng.integers(min_y, max_y + 1))

    canvas = bg.copy()
    canvas.alpha_composite(group, (x, y))
    return canvas.convert("RGB")


def _overlay_grid_on_sign_rgba(
    sign_rgba: Image.Image,
    grid_cell: int,
    cell_cover_thresh: float,
    line_rgba: tuple[int, int, int, int] = (255, 255, 255, 200),
    fill_rgba: tuple[int, int, int, int] = (255, 255, 255, 20),
    line_width: int = 2,
) -> Image.Image:
    """Overlay simulation-valid cell boundaries directly on sign RGBA."""
    sign = sign_rgba.copy().convert("RGBA")
    alpha = np.array(sign.split()[-1], dtype=np.uint8) > 0
    w, h = sign.size
    gw, gh = math.ceil(w / grid_cell), math.ceil(h / grid_cell)
    draw = ImageDraw.Draw(sign, "RGBA")

    for r in range(gh):
        for c in range(gw):
            x0, y0 = c * grid_cell, r * grid_cell
            x1, y1 = min(w, x0 + grid_cell), min(h, y0 + grid_cell)
            cell = alpha[y0:y1, x0:x1]
            cover = float(cell.mean()) if cell.size else 0.0
            if cover < float(cell_cover_thresh):
                continue
            draw.rectangle([x0, y0, x1, y1], fill=fill_rgba, outline=line_rgba, width=line_width)
    return sign


def _paste_contain(dst: Image.Image, src: Image.Image, box: tuple[int, int, int, int], bg=(245, 247, 250)) -> None:
    x0, y0, x1, y1 = box
    bw = max(1, x1 - x0)
    bh = max(1, y1 - y0)
    scale = min(bw / src.width, bh / src.height)
    nw = max(1, int(src.width * scale))
    nh = max(1, int(src.height * scale))
    rs = src.resize((nw, nh), Image.BILINEAR)
    panel = Image.new("RGB", (bw, bh), bg)
    px = (bw - nw) // 2
    py = (bh - nh) // 2
    panel.paste(rs, (px, py))
    dst.paste(panel, (x0, y0))


def _build_composite(
    grid_sign: Image.Image,
    day_uv_pairs: list[tuple[Image.Image, Image.Image]],
    out_path: Path,
) -> None:
    w, h = 2200, 1300
    canvas = Image.new("RGB", (w, h), (252, 252, 253))
    draw = ImageDraw.Draw(canvas, "RGB")
    title_font = _load_font(54)
    sub_font = _load_font(26)
    lbl_font = _load_font(28)

    draw.text((40, 28), "Training Simulation Visuals (Matched DAY/UV Transform Sets)", font=title_font, fill=(20, 28, 44))
    draw.text((42, 95), "Each pair uses identical background, placement, and transform parameters.", font=sub_font, fill=(75, 88, 110))

    # Left: grid stop sign
    left_box = (40, 160, 900, 1220)
    draw.rounded_rectangle(left_box, radius=22, outline=(180, 190, 210), width=3, fill=(248, 250, 252))
    draw.text((70, 185), "Grid-Constrained Stop Sign", font=lbl_font, fill=(28, 36, 56))
    _paste_contain(canvas, grid_sign.convert("RGB"), (70, 240, 870, 1180))

    # Right: 2x2 sets, each with DAY and UV thumbnail
    right_x0, right_y0 = 940, 160
    panel_w, panel_h = 1215, 1060
    draw.rounded_rectangle((right_x0, right_y0, right_x0 + panel_w, right_y0 + panel_h),
                           radius=22, outline=(180, 190, 210), width=3, fill=(248, 250, 252))
    draw.text((right_x0 + 30, right_y0 + 25), "Matched Render Sets", font=lbl_font, fill=(28, 36, 56))

    cols, rows = 2, 2
    cell_w = (panel_w - 70) // cols
    cell_h = (panel_h - 110) // rows
    idx = 0
    for r in range(rows):
        for c in range(cols):
            if idx >= len(day_uv_pairs):
                break
            day_img, uv_img = day_uv_pairs[idx]
            x = right_x0 + 25 + c * cell_w
            y = right_y0 + 80 + r * cell_h
            box = (x, y, x + cell_w - 20, y + cell_h - 20)
            draw.rounded_rectangle(box, radius=15, outline=(200, 208, 224), width=2, fill=(255, 255, 255))
            draw.text((x + 16, y + 12), f"Set {idx + 1}", font=sub_font, fill=(36, 48, 70))
            # split day/uv
            mx0, my0, mx1, my1 = x + 16, y + 52, x + cell_w - 36, y + cell_h - 36
            gap = 12
            mid = (mx0 + mx1 - gap) // 2
            day_box = (mx0, my0, mid, my1)
            uv_box = (mid + gap, my0, mx1, my1)
            _paste_contain(canvas, day_img, day_box)
            _paste_contain(canvas, uv_img, uv_box)
            draw.rectangle(day_box, outline=(70, 130, 220), width=3)
            draw.rectangle(uv_box, outline=(132, 74, 196), width=3)
            draw.text((day_box[0] + 10, day_box[1] + 8), "DAY", font=sub_font, fill=(40, 92, 170))
            draw.text((uv_box[0] + 10, uv_box[1] + 8), "UV", font=sub_font, fill=(106, 56, 165))
            idx += 1

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def _parse_indices_csv(text: str) -> list[int]:
    out: list[int] = []
    for tok in str(text).split(","):
        tok = tok.strip()
        if not tok:
            continue
        out.append(int(tok))
    return out


def _pick_spaced_indices(n: int, k: int) -> list[int]:
    if n <= 0 or k <= 0:
        return []
    if k == 1:
        return [0]
    if k >= n:
        return list(range(n))
    vals = []
    for i in range(k):
        idx = int(round(i * (n - 1) / (k - 1)))
        vals.append(idx)
    # dedupe while preserving order
    out: list[int] = []
    seen = set()
    for v in vals:
        if v in seen:
            continue
        seen.add(v)
        out.append(v)
    # pad if dedupe reduced count
    j = 0
    while len(out) < k and j < n:
        if j not in seen:
            out.append(j)
            seen.add(j)
        j += 1
    return out[:k]


def _load_trace_records(path: Path) -> list[dict[str, Any]]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    records: list[dict[str, Any]] = []

    def add_from_episode(ep: dict[str, Any], idx: int):
        tr = ep.get("trace", {}) if isinstance(ep, dict) else {}
        if not isinstance(tr, dict):
            return
        place_seed = tr.get("place_seed", None)
        t_seeds = tr.get("transform_seeds", [])
        if place_seed is None or not isinstance(t_seeds, list) or not t_seeds:
            return
        rec = {
            "episode_index": int(ep.get("episode_index", idx)),
            "seed": ep.get("seed", None),
            "place_seed": int(place_seed),
            "transform_seeds": [int(x) for x in t_seeds],
        }
        records.append(rec)

    if isinstance(obj, dict):
        eps = obj.get("episodes_detail", None)
        if isinstance(eps, list):
            for i, ep in enumerate(eps):
                if isinstance(ep, dict):
                    add_from_episode(ep, i)
    elif isinstance(obj, list):
        for i, ep in enumerate(obj):
            if isinstance(ep, dict):
                add_from_episode(ep, i)
    return records


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Generate matched DAY/UV training simulation visuals.")
    ap.add_argument("--data", default="./data")
    ap.add_argument("--bgdir", default="./data/backgrounds")
    ap.add_argument("--out-dir", default="./_runs/paper_data/figures/training_sim")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--sets", type=int, default=4)
    ap.add_argument("--img-size", type=int, default=640)
    ap.add_argument("--grid-cell", type=int, default=16)
    ap.add_argument("--cell-cover-thresh", type=float, default=0.60)
    ap.add_argument("--transform-strength", type=float, default=1.0)
    ap.add_argument("--fixed-angle-deg", type=float, default=None)
    ap.add_argument("--geom-gain", type=float, default=1.0, help="Multiplier for geometric transform magnitude.")
    ap.add_argument("--photo-gain", type=float, default=1.0, help="Multiplier for photometric/lighting effects.")
    ap.add_argument("--force-photometric", type=int, default=0, help="If 1, always apply photometric effects.")
    ap.add_argument(
        "--episodes-json",
        default="",
        help="Optional eval summary/episodes JSON containing episodes_detail.trace with place_seed/transform_seeds.",
    )
    ap.add_argument(
        "--episode-indices",
        default="",
        help="Optional comma-separated episode indices to use from --episodes-json (e.g., 0,25,50,75).",
    )
    ap.add_argument(
        "--bg-from-episode-seed",
        type=int,
        default=1,
        help="If 1 and episode seed exists, approximate training bg selection from that seed.",
    )
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    data_dir = Path(args.data)
    bg_dir = Path(args.bgdir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sign_day = Image.open(data_dir / "stop_sign.png").convert("RGBA")
    uv_path = data_dir / "stop_sign_uv.png"
    sign_uv = Image.open(uv_path).convert("RGBA") if uv_path.exists() else sign_day.copy()
    pole_path = data_dir / "pole.png"
    pole = Image.open(pole_path).convert("RGBA") if pole_path.exists() else None

    bg_paths = sorted([p for p in bg_dir.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}])
    if not bg_paths:
        raise FileNotFoundError(f"No backgrounds found in: {bg_dir}")
    bgs = [Image.open(p).convert("RGB") for p in bg_paths]

    pairs: list[tuple[Image.Image, Image.Image]] = []
    sets_meta: list[dict[str, Any]] = []
    grid_panel_scene: Image.Image | None = None
    trace_source = None
    trace_records: list[dict[str, Any]] = []
    if str(args.episodes_json).strip():
        jp = Path(args.episodes_json)
        if not jp.exists():
            raise FileNotFoundError(f"--episodes-json not found: {jp}")
        trace_records = _load_trace_records(jp)
        trace_source = str(jp)
        if not trace_records:
            raise ValueError(f"No usable trace records found in: {jp}")

    rng = np.random.default_rng(int(args.seed))
    use_indices: list[int] = []
    if trace_records:
        if str(args.episode_indices).strip():
            use_indices = _parse_indices_csv(args.episode_indices)
        else:
            use_indices = _pick_spaced_indices(len(trace_records), int(args.sets))

    for i in range(int(args.sets)):
        if trace_records:
            src_idx = use_indices[i % len(use_indices)]
            rec = trace_records[src_idx]
            place_seed = int(rec["place_seed"])
            t_seed = int(rec["transform_seeds"][i % len(rec["transform_seeds"])])
            ep_seed = rec.get("seed", None)
            if int(args.bg_from_episode_seed) == 1 and ep_seed is not None:
                bg_rng = np.random.default_rng(int(ep_seed))
                bg_idx = int(bg_rng.integers(0, len(bgs)))
            else:
                bg_idx = int(rng.integers(0, len(bgs)))
        else:
            src_idx = None
            rec = {}
            place_seed = int(rng.integers(0, 2**31 - 1))
            t_seed = int(rng.integers(0, 2**31 - 1))
            bg_idx = int(rng.integers(0, len(bgs)))

        # Use the same transform generator as env, seeded by trace transform seed.
        params = _sample_transform_params(
            seed=t_seed,
            w=sign_day.width,
            h=sign_day.height,
            strength=float(args.transform_strength),
            fixed_angle_deg=args.fixed_angle_deg,
            geom_gain=float(args.geom_gain),
            photo_gain=float(args.photo_gain),
            force_photometric=bool(int(args.force_photometric)),
        )
        day_t = _apply_transform(sign_day, params)
        uv_t = _apply_transform(sign_uv, params)

        day_group = _compose_sign_and_pole(day_t, pole)
        uv_group = _compose_sign_and_pole(uv_t, pole)

        day_img = _place_group_on_background(day_group, bgs[bg_idx], place_seed, (int(args.img_size), int(args.img_size)))
        uv_img = _place_group_on_background(uv_group, bgs[bg_idx], place_seed, (int(args.img_size), int(args.img_size)))

        if grid_panel_scene is None:
            # Build a "real scene" stop-sign panel with simulation-valid grid overlay on top.
            grid_sign = _overlay_grid_on_sign_rgba(
                sign_day,
                grid_cell=int(args.grid_cell),
                cell_cover_thresh=float(args.cell_cover_thresh),
            )
            grid_sign_t = _apply_transform(grid_sign, params)
            grid_group = _compose_sign_and_pole(grid_sign_t, pole)
            grid_panel_scene = _place_group_on_background(
                grid_group, bgs[bg_idx], place_seed, (int(args.img_size), int(args.img_size))
            )

        day_img.save(out_dir / f"set_{i+1:02d}_day.png")
        uv_img.save(out_dir / f"set_{i+1:02d}_uv.png")
        pairs.append((day_img, uv_img))
        sets_meta.append(
            {
                "set_index": i + 1,
                "trace_source": trace_source,
                "trace_row_index": src_idx,
                "trace_episode_index": rec.get("episode_index", None) if isinstance(rec, dict) else None,
                "trace_episode_seed": rec.get("seed", None) if isinstance(rec, dict) else None,
                "background": bg_paths[bg_idx].name,
                "place_seed": place_seed,
                "transform_seed": t_seed,
                "transform_params": params,
            }
        )

    if grid_panel_scene is None:
        raise RuntimeError("Failed to build grid panel scene.")
    grid_panel_path = out_dir / "panel_grid_stop_sign.png"
    grid_panel_scene.save(grid_panel_path)

    composite_path = out_dir / "training_sim_visual_overview.png"
    _build_composite(grid_panel_scene, pairs, composite_path)

    meta = {
        "seed": int(args.seed),
        "sets": int(args.sets),
        "img_size": int(args.img_size),
        "grid_cell": int(args.grid_cell),
        "cell_cover_thresh": float(args.cell_cover_thresh),
        "transform_strength": float(args.transform_strength),
        "geom_gain": float(args.geom_gain),
        "photo_gain": float(args.photo_gain),
        "force_photometric": bool(int(args.force_photometric)),
        "fixed_angle_deg": None if args.fixed_angle_deg is None else float(args.fixed_angle_deg),
        "outputs": {
            "grid_panel": str(grid_panel_path),
            "composite": str(composite_path),
            "pairs": [
                {"day": str(out_dir / f"set_{i+1:02d}_day.png"), "uv": str(out_dir / f"set_{i+1:02d}_uv.png")}
                for i in range(int(args.sets))
            ],
        },
        "sets_meta": sets_meta,
    }
    (out_dir / "training_sim_visual_overview.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"[SAVE] {out_dir / 'panel_grid_stop_sign.png'}")
    for i in range(int(args.sets)):
        print(f"[SAVE] {out_dir / f'set_{i+1:02d}_day.png'}")
        print(f"[SAVE] {out_dir / f'set_{i+1:02d}_uv.png'}")
    print(f"[SAVE] {composite_path}")
    print(f"[SAVE] {out_dir / 'training_sim_visual_overview.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
