from __future__ import annotations

import argparse
import json
import math
import os
import random
from typing import List, Tuple

from PIL import Image, ImageDraw


def build_grid(base: Image.Image, grid: int, cover_thresh: float) -> Tuple[int, int, List[Tuple[int,int,int,int]], List[Tuple[int,int]]]:
    w, h = base.size
    gw, gh = math.ceil(w / grid), math.ceil(h / grid)
    alpha = base.split()[-1]
    a = alpha.load()

    rects: List[Tuple[int,int,int,int]] = []
    valid: List[Tuple[int,int]] = []

    for r in range(gh):
        for c in range(gw):
            x0, y0 = c * grid, r * grid
            x1, y1 = min(w, x0 + grid), min(h, y0 + grid)
            # compute coverage ratio in alpha mask
            covered = 0
            total = 0
            for yy in range(y0, y1):
                for xx in range(x0, x1):
                    total += 1
                    if a[xx, yy] > 0:
                        covered += 1
            cover = (covered / total) if total else 0.0
            if cover >= cover_thresh:
                valid.append((r, c))
            rects.append((x0, y0, x1, y1))
    return gw, gh, rects, valid


def pick_used(valid: List[Tuple[int,int]], used_count: int, seed: int) -> List[Tuple[int,int]]:
    rng = random.Random(seed)
    if used_count <= 0:
        return []
    used_count = min(used_count, len(valid))
    return rng.sample(valid, used_count)


def pick_red(valid: List[Tuple[int,int]], used: List[Tuple[int,int]], w: int, h: int, grid: int) -> Tuple[int,int] | None:
    used_set = set(used)
    remaining = [rc for rc in valid if rc not in used_set]
    if not remaining:
        return None
    # pick the remaining cell closest to the image center for clarity
    cx, cy = w / 2.0, h / 2.0
    best = None
    best_d = None
    for (r, c) in remaining:
        x = c * grid + grid / 2.0
        y = r * grid + grid / 2.0
        d = (x - cx) ** 2 + (y - cy) ** 2
        if best is None or d < best_d:
            best = (r, c)
            best_d = d
    return best


def draw_grid_lines(draw: ImageDraw.ImageDraw, w: int, h: int, grid: int, color: Tuple[int,int,int,int], width: int) -> None:
    # vertical lines
    x = 0
    while x <= w:
        draw.line([(x, 0), (x, h)], fill=color, width=width)
        x += grid
    # horizontal lines
    y = 0
    while y <= h:
        draw.line([(0, y), (w, y)], fill=color, width=width)
        y += grid


def render_overlay(base: Image.Image, grid: int, rects: List[Tuple[int,int,int,int]], used: List[Tuple[int,int]], selected: Tuple[int,int] | None, out_path: str,
                   fill_alpha: int, select_alpha: int, grid_alpha: int, line_width: int, select_rgb: Tuple[int,int,int]) -> None:
    w, h = base.size
    canvas = base.copy()
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(overlay)

    # fill used cells
    for (r, c) in used:
        idx = r * (math.ceil(w / grid)) + c
        x0, y0, x1, y1 = rects[idx]
        d.rectangle([x0, y0, x1, y1], fill=(0, 0, 0, fill_alpha))

    # fill selected cell
    if selected is not None:
        r, c = selected
        idx = r * (math.ceil(w / grid)) + c
        x0, y0, x1, y1 = rects[idx]
        d.rectangle([x0, y0, x1, y1], fill=(select_rgb[0], select_rgb[1], select_rgb[2], select_alpha))

    # composite fills
    canvas = Image.alpha_composite(canvas, overlay)

    # grid lines on top (masked to sign alpha)
    grid_layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    gd = ImageDraw.Draw(grid_layer)
    draw_grid_lines(gd, w, h, grid, (0, 0, 0, grid_alpha), line_width)
    sign_mask = base.split()[-1]
    grid_layer = Image.composite(grid_layer, Image.new("RGBA", (w, h), (0, 0, 0, 0)), sign_mask)
    canvas = Image.alpha_composite(canvas, grid_layer)

    canvas.save(out_path)


def main() -> int:
    ap = argparse.ArgumentParser(description="Create grid overlay images for stop-sign examples.")
    ap.add_argument("--base", required=True, help="Base stop-sign image (RGBA)")
    ap.add_argument("--out-dir", required=True, help="Output directory")
    ap.add_argument("--grid", type=int, default=16, help="Grid cell size in pixels")
    ap.add_argument("--used-count", type=int, default=12, help="Number of used (black) cells to fill")
    ap.add_argument("--seed", type=int, default=7, help="Seed for used-cell sampling")
    ap.add_argument("--cover-thresh", type=float, default=0.60, help="Alpha coverage threshold for valid cells")
    ap.add_argument("--fill-alpha", type=int, default=200, help="Alpha for black used cells")
    ap.add_argument("--select-alpha", type=int, default=220, help="Alpha for selected cell")
    ap.add_argument("--select-color", default="0,200,0", help="Selected cell RGB as r,g,b (default green)")
    ap.add_argument("--grid-alpha", type=int, default=255, help="Alpha for grid lines")
    ap.add_argument("--line-width", type=int, default=1, help="Grid line width in pixels")
    ap.add_argument("--prefix", default="grid_example", help="Output filename prefix")
    args = ap.parse_args()

    base = Image.open(args.base).convert("RGBA")
    w, h = base.size
    try:
        parts = [int(p.strip()) for p in args.select_color.split(",")]
        if len(parts) != 3 or any(p < 0 or p > 255 for p in parts):
            raise ValueError
        select_rgb = (parts[0], parts[1], parts[2])
    except Exception as exc:
        raise ValueError("--select-color must be r,g,b with 0..255") from exc

    gw, gh, rects, valid = build_grid(base, args.grid, args.cover_thresh)
    used = pick_used(valid, args.used_count, args.seed)
    red = pick_red(valid, used, w, h, args.grid)

    os.makedirs(args.out_dir, exist_ok=True)

    out_used = os.path.join(args.out_dir, f"{args.prefix}_used.png")
    out_selected = os.path.join(args.out_dir, f"{args.prefix}_selected.png")

    render_overlay(
        base,
        args.grid,
        rects,
        used,
        None,
        out_used,
        fill_alpha=args.fill_alpha,
        select_alpha=args.select_alpha,
        grid_alpha=args.grid_alpha,
        line_width=args.line_width,
        select_rgb=select_rgb,
    )
    render_overlay(
        base,
        args.grid,
        rects,
        used,
        red,
        out_selected,
        fill_alpha=args.fill_alpha,
        select_alpha=args.select_alpha,
        grid_alpha=args.grid_alpha,
        line_width=args.line_width,
        select_rgb=select_rgb,
    )

    meta = {
        "base": args.base,
        "grid": args.grid,
        "grid_size": [gw, gh],
        "cover_thresh": args.cover_thresh,
        "used_count": len(used),
        "used_cells_rc": used,
        "red_cell_rc": red,
    }
    with open(os.path.join(args.out_dir, f"{args.prefix}_cells.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("Wrote:")
    print(out_used)
    print(out_selected)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
