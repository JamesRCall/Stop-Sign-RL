#!/usr/bin/env python3
"""Export a printable stencil for a replica stop sign using env-matched grid geometry.

This tool uses the sign alpha mask from `data/stop_sign.png` and the same grid-validity
logic as StopSignGridEnv (`grid_cell_px`, `cell_cover_thresh`) to create a print-ready
stencil page showing which grid squares to cut/paint on a privately owned replica sign.

Outputs (default):
  - <out-prefix>_stencil_page.png   (printable page with grid + marks + selected cells)
  - <out-prefix>_stencil_page.pdf   (same printable page)
  - <out-prefix>_cutmask.png        (exact clipped pattern mask in sign coordinates, scaled)
  - <out-prefix>_metadata.json      (grid geometry + selected cells for reproducibility)
Optional:
  - <out-prefix>_<mode>_mask.svg    (vector mask at true size; --export-svg)
  - <out-prefix>_<mode>_mask.stl    (3D-printable mask tiles in mm; --export-stl)
"""
from __future__ import annotations

import argparse
import json
import math
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont


@dataclass
class GridGeom:
    W: int
    H: int
    Gw: int
    Gh: int
    g: int
    valid: np.ndarray  # [Gh, Gw] bool
    rects: list[tuple[int, int, int, int]]
    sign_alpha: np.ndarray  # [H, W] bool


def _load_font(size: int):
    try:
        return ImageFont.truetype("arial.ttf", size=size)
    except Exception:
        try:
            return ImageFont.truetype("DejaVuSans.ttf", size=size)
        except Exception:
            return ImageFont.load_default()


def _build_grid(sign_rgba: Image.Image, grid_cell_px: int, cell_cover_thresh: float) -> GridGeom:
    W, H = sign_rgba.size
    g = int(grid_cell_px)
    Gw, Gh = math.ceil(W / g), math.ceil(H / g)

    sign_alpha = np.array(sign_rgba.split()[-1], dtype=np.uint8) > 0
    valid = np.zeros((Gh, Gw), dtype=bool)
    rects: list[tuple[int, int, int, int]] = []

    for r in range(Gh):
        for c in range(Gw):
            x0, y0 = c * g, r * g
            x1, y1 = min(W, x0 + g), min(H, y0 + g)
            cell = sign_alpha[y0:y1, x0:x1]
            cover = float(cell.mean()) if cell.size else 0.0
            valid[r, c] = (cover >= float(cell_cover_thresh))
            rects.append((x0, y0, x1, y1))

    return GridGeom(W=W, H=H, Gw=Gw, Gh=Gh, g=g, valid=valid, rects=rects, sign_alpha=sign_alpha)


def _parse_selected_rc(s: str, geom: GridGeom) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    if not s.strip():
        return out
    # Accept separators ';' and '|', pairs as r,c or r:c
    chunks = [c.strip() for c in s.replace("|", ";").split(";") if c.strip()]
    for ch in chunks:
        if ":" in ch:
            a, b = ch.split(":", 1)
        elif "," in ch:
            a, b = ch.split(",", 1)
        else:
            raise ValueError(f"Bad selected-rc token '{ch}'. Use r,c;r,c or r:c;r:c")
        r, c = int(a.strip()), int(b.strip())
        if not (0 <= r < geom.Gh and 0 <= c < geom.Gw):
            raise ValueError(f"Cell (r={r}, c={c}) out of range for grid {geom.Gh}x{geom.Gw}")
        out.append((r, c))
    return out


def _parse_selected_indices(s: str, geom: GridGeom) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    if not s.strip():
        return out
    vals = [v.strip() for v in s.replace(";", ",").split(",") if v.strip()]
    for v in vals:
        idx = int(v)
        if idx < 0 or idx >= geom.Gh * geom.Gw:
            raise ValueError(f"Flat cell index {idx} out of range for {geom.Gh}x{geom.Gw}")
        r = idx // geom.Gw
        c = idx % geom.Gw
        out.append((r, c))
    return out


def _load_selected_from_json(path: Path, geom: GridGeom, episode_index: int = 0) -> list[tuple[int, int]]:
    data = json.loads(path.read_text(encoding="utf-8"))

    def find_selected(obj: Any) -> Any:
        if isinstance(obj, dict):
            if "selected_indices" in obj and isinstance(obj["selected_indices"], list):
                return obj["selected_indices"]
            if "trace" in obj and isinstance(obj["trace"], dict):
                tr = obj["trace"]
                if "selected_indices" in tr and isinstance(tr["selected_indices"], list):
                    return tr["selected_indices"]
            if "episodes_detail" in obj and isinstance(obj["episodes_detail"], list) and obj["episodes_detail"]:
                idx = max(0, min(int(episode_index), len(obj["episodes_detail"]) - 1))
                return find_selected(obj["episodes_detail"][idx])
            for v in obj.values():
                found = find_selected(v)
                if found is not None:
                    return found
        elif isinstance(obj, list):
            # If the file itself is a per-episode list from eval, select one row.
            if obj and isinstance(obj[0], dict):
                idx = max(0, min(int(episode_index), len(obj) - 1))
                return find_selected(obj[idx])
        return None

    selected = find_selected(data)
    if selected is None:
        raise ValueError(
            f"No 'selected_indices' found in {path}. "
            "Pass --selected-rc or --selected-indices manually, or use JSON that contains trace.selected_indices."
        )
    return _parse_selected_indices(",".join(str(int(v)) for v in selected), geom)


def _dedupe_cells(cells: list[tuple[int, int]]) -> list[tuple[int, int]]:
    seen = set()
    out = []
    for rc in cells:
        if rc in seen:
            continue
        seen.add(rc)
        out.append(rc)
    return out


def _render_selected_mask_sign_coords(geom: GridGeom, selected_cells: list[tuple[int, int]]) -> np.ndarray:
    """Binary mask in sign pixel coordinates, clipped to sign alpha exactly like env overlay."""
    H, W = geom.H, geom.W
    sel = np.zeros((H, W), dtype=np.uint8)
    for r, c in selected_cells:
        if not (0 <= r < geom.Gh and 0 <= c < geom.Gw):
            continue
        x0, y0, x1, y1 = geom.rects[r * geom.Gw + c]
        sel[y0:y1, x0:x1] = 255
    sel = np.where(geom.sign_alpha, sel, 0).astype(np.uint8)
    return sel


def _to_inches(val: float, units: str) -> float:
    units = units.lower().strip()
    if units in ("in", "inch", "inches"):
        return float(val)
    if units in ("cm", "centimeter", "centimeters"):
        return float(val) / 2.54
    if units in ("mm", "millimeter", "millimeters"):
        return float(val) / 25.4
    raise ValueError(f"Unsupported units: {units}")


def _draw_registration_marks(draw: ImageDraw.ImageDraw, page_w: int, page_h: int, box: tuple[int, int, int, int]):
    x0, y0, x1, y1 = box
    col = (0, 0, 0, 255)
    L = max(12, int(0.01 * min(page_w, page_h)))
    gap = max(6, L // 3)
    # corner marks around sign placement box
    for (x, y, sx, sy) in [
        (x0, y0, +1, +1),
        (x1, y0, -1, +1),
        (x0, y1, +1, -1),
        (x1, y1, -1, -1),
    ]:
        draw.line([(x + sx * gap, y), (x + sx * (gap + L), y)], fill=col, width=2)
        draw.line([(x, y + sy * gap), (x, y + sy * (gap + L))], fill=col, width=2)
    # center crosshair
    cx = (x0 + x1) // 2
    cy = (y0 + y1) // 2
    cL = max(10, L - 2)
    draw.line([(cx - cL, cy), (cx + cL, cy)], fill=col, width=1)
    draw.line([(cx, cy - cL), (cx, cy + cL)], fill=col, width=1)


def _draw_scale_box(draw: ImageDraw.ImageDraw, dpi: int, page_margin_px: int, page_h: int, units: str):
    # 1 inch or 2 cm reference box depending on units
    if units.lower().startswith("c") or units.lower().startswith("m"):
        label = "2 cm"
        side_px = int(round((2.0 / 2.54) * dpi))
    else:
        label = "1 in"
        side_px = int(round(1.0 * dpi))
    x0 = page_margin_px
    y0 = page_h - page_margin_px - side_px
    draw.rectangle([x0, y0, x0 + side_px, y0 + side_px], outline=(0, 0, 0, 255), width=2)
    font = _load_font(max(14, side_px // 7))
    draw.text((x0 + side_px + 10, y0 + side_px // 2), label, fill=(0, 0, 0, 255), font=font, anchor="lm")


def _draw_page(
    sign_rgba: Image.Image,
    geom: GridGeom,
    selected_cells: list[tuple[int, int]],
    sign_width_in: float,
    sign_height_in: float,
    dpi: int,
    margin_in: float,
    label_cells: bool,
    units: str = "in",
) -> tuple[Image.Image, Image.Image]:
    """
    Returns:
      page_rgba: printable page with reference/grid/selected cells
      cutmask_rgba: exact clipped cut-mask in sign physical coordinates (transparent bg)
    """
    sign_w_px = int(round(sign_width_in * dpi))
    sign_h_px = int(round(sign_height_in * dpi))
    margin_px = int(round(margin_in * dpi))
    header_px = int(round(0.45 * dpi))
    footer_px = int(round(0.40 * dpi))
    page_w = sign_w_px + 2 * margin_px
    page_h = sign_h_px + 2 * margin_px + header_px + footer_px

    # Exact cut mask in sign coordinates (scaled to physical dimensions)
    sel_mask = _render_selected_mask_sign_coords(geom, selected_cells)
    sel_mask_img = Image.fromarray(sel_mask, mode="L").resize((sign_w_px, sign_h_px), Image.NEAREST)
    sign_alpha_img = Image.fromarray((geom.sign_alpha.astype(np.uint8) * 255), mode="L").resize((sign_w_px, sign_h_px), Image.NEAREST)

    cutmask = Image.new("RGBA", (sign_w_px, sign_h_px), (0, 0, 0, 0))
    # Black cutouts where selected
    cutmask.paste((0, 0, 0, 255), mask=sel_mask_img)

    # Printable page
    page = Image.new("RGBA", (page_w, page_h), (255, 255, 255, 255))
    draw = ImageDraw.Draw(page, "RGBA")
    title_font = _load_font(max(18, int(0.12 * dpi)))
    body_font = _load_font(max(12, int(0.08 * dpi)))
    mono_font = _load_font(max(10, int(0.065 * dpi)))

    # Title / instructions (minimal)
    draw.text((margin_px, margin_px // 2), "Replica Stop-Sign Grid Stencil (Private Test Only)", fill=(0, 0, 0, 255), font=title_font)
    draw.text(
        (margin_px, margin_px // 2 + int(0.16 * dpi)),
        "Align sign outline + registration marks. Cut/paint only black selected cells.",
        fill=(60, 60, 60, 255),
        font=body_font,
    )

    sign_box = (margin_px, margin_px + header_px, margin_px + sign_w_px, margin_px + header_px + sign_h_px)
    sx0, sy0, sx1, sy1 = sign_box

    # Light sign silhouette reference from actual sign alpha (scaled)
    ref = Image.new("RGBA", (sign_w_px, sign_h_px), (255, 255, 255, 0))
    ref_draw = ImageDraw.Draw(ref, "RGBA")
    # faint red fill where sign exists
    ref_draw.rectangle([0, 0, sign_w_px, sign_h_px], fill=(0, 0, 0, 0))
    ref.paste((235, 235, 235, 255), mask=sign_alpha_img)
    # overlay selected black cells
    ref.alpha_composite(cutmask, (0, 0))
    page.alpha_composite(ref, (sx0, sy0))

    # Draw sign outer contour approx from alpha edge (raster edge)
    # Using sign alpha as a mask border by dilate-erode style difference via resize and point ops avoided;
    # simply outline sign bounding box and rely on silhouette + grid for alignment.
    draw.rectangle([sx0, sy0, sx1, sy1], outline=(150, 150, 150, 255), width=2)

    # Draw grid lines scaled and validity shading
    sx = sign_w_px / geom.W
    sy = sign_h_px / geom.H
    for r in range(geom.Gh):
        for c in range(geom.Gw):
            x0, y0, x1, y1 = geom.rects[r * geom.Gw + c]
            X0 = sx0 + int(round(x0 * sx))
            Y0 = sy0 + int(round(y0 * sy))
            X1 = sx0 + int(round(x1 * sx))
            Y1 = sy0 + int(round(y1 * sy))
            if geom.valid[r, c]:
                outline = (160, 160, 160, 120)
                fill = None
            else:
                outline = (180, 180, 180, 80)
                fill = (240, 240, 240, 80)
            if fill is not None:
                draw.rectangle([X0, Y0, X1, Y1], fill=fill)
            draw.rectangle([X0, Y0, X1, Y1], outline=outline, width=1)

    # Re-paste selected cutmask on top to ensure visibility over grid lines
    page.alpha_composite(cutmask, (sx0, sy0))

    # Registration marks + center cross
    _draw_registration_marks(draw, page_w, page_h, sign_box)
    _draw_scale_box(draw, dpi=dpi, page_margin_px=margin_px, page_h=page_h, units=units)

    # Footer metadata
    footer_y = sy1 + int(0.10 * dpi)
    meta_line = (
        f"Sign size: {sign_width_in:.2f}in x {sign_height_in:.2f}in | "
        f"grid_cell_px={geom.g} | grid={geom.Gh}x{geom.Gw} | valid_cells={int(geom.valid.sum())} | "
        f"selected={len(selected_cells)}"
    )
    draw.text((margin_px, footer_y), meta_line, fill=(50, 50, 50, 255), font=body_font)

    if label_cells and selected_cells:
        # Compact selected cell listing
        cells_str = ", ".join([f"({r},{c})" for r, c in selected_cells[:80]])
        if len(selected_cells) > 80:
            cells_str += f", ... (+{len(selected_cells) - 80} more)"
        draw.text((margin_px, footer_y + int(0.12 * dpi)), f"Selected cells: {cells_str}", fill=(70, 70, 70, 255), font=mono_font)

    return page, cutmask


def _paper_size_inches(paper_size: str, orientation: str, content_size_in: tuple[float, float] | None = None) -> tuple[float, float]:
    ps = str(paper_size).lower().strip()
    if ps == "letter":
        w, h = 8.5, 11.0
    elif ps == "a4":
        w, h = 210.0 / 25.4, 297.0 / 25.4
    else:
        raise ValueError(f"Unsupported paper size: {paper_size}")

    ori = str(orientation).lower().strip()
    if ori == "landscape":
        return (max(w, h), min(w, h))
    if ori == "portrait":
        return (min(w, h), max(w, h))
    if ori != "auto":
        raise ValueError(f"Unsupported paper orientation: {orientation}")
    if content_size_in is None:
        return (min(w, h), max(w, h))
    cw, ch = content_size_in
    content_landscape = cw > ch
    if content_landscape:
        return (max(w, h), min(w, h))
    return (min(w, h), max(w, h))


def _tile_starts(total_px: int, window_px: int, stride_px: int) -> list[int]:
    if total_px <= window_px:
        return [0]
    starts = [0]
    pos = 0
    while True:
        nxt = pos + stride_px
        if nxt + window_px >= total_px:
            last = max(0, total_px - window_px)
            if last != starts[-1]:
                starts.append(last)
            break
        starts.append(nxt)
        pos = nxt
    return starts


def _tile_starts_mm(total_mm: float, window_mm: float, stride_mm: float) -> list[float]:
    if total_mm <= window_mm:
        return [0.0]
    starts = [0.0]
    pos = 0.0
    while True:
        nxt = pos + stride_mm
        if nxt + window_mm >= total_mm:
            last = max(0.0, total_mm - window_mm)
            if last - starts[-1] > 1e-6:
                starts.append(last)
            break
        starts.append(nxt)
        pos = nxt
    return starts


def _split_indices_by_mm(widths_mm: list[float], groups: int) -> list[tuple[int, int]]:
    if groups <= 1:
        return [(0, len(widths_mm))]
    total = float(sum(widths_mm))
    if total <= 0:
        return [(0, len(widths_mm))]
    splits: list[tuple[int, int]] = []
    start = 0
    used = 0.0
    for g in range(groups):
        remaining_groups = groups - g
        remaining_width = total - used
        target = remaining_width / remaining_groups
        if g == groups - 1:
            splits.append((start, len(widths_mm)))
            break
        acc = 0.0
        idx = start
        while idx < len(widths_mm):
            acc += widths_mm[idx]
            idx += 1
            cols_remaining = len(widths_mm) - idx
            if acc >= target and cols_remaining >= remaining_groups - 1:
                break
        splits.append((start, idx))
        used += acc
        start = idx
    return splits


def _mask_cells_from_mode(geom: GridGeom, selected_cells: list[tuple[int, int]], mode: str) -> list[tuple[int, int]]:
    mode = str(mode).lower().strip()
    if mode == "selected":
        return list(selected_cells)
    if mode != "cover":
        raise ValueError(f"Unsupported mask mode: {mode}")
    sel = set(selected_cells)
    out: list[tuple[int, int]] = []
    for r in range(geom.Gh):
        for c in range(geom.Gw):
            if geom.valid[r, c] and (r, c) not in sel:
                out.append((r, c))
    return out


def _rect_mm_from_cell(geom: GridGeom, r: int, c: int, scale_x: float, scale_y: float) -> tuple[float, float, float, float]:
    x0, y0, x1, y1 = geom.rects[r * geom.Gw + c]
    x0_mm = float(x0) * scale_x
    x1_mm = float(x1) * scale_x
    # Flip Y so STL uses bottom-left origin (y up)
    y0_mm = float(geom.H - y1) * scale_y
    y1_mm = float(geom.H - y0) * scale_y
    return x0_mm, y0_mm, x1_mm, y1_mm


def _export_svg(
    path: Path,
    geom: GridGeom,
    mask_cells: list[tuple[int, int]],
    sign_w_mm: float,
    sign_h_mm: float,
) -> None:
    scale_x = sign_w_mm / float(geom.W)
    scale_y = sign_h_mm / float(geom.H)
    header = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{sign_w_mm:.3f}mm" height="{sign_h_mm:.3f}mm" viewBox="0 0 {sign_w_mm:.3f} {sign_h_mm:.3f}">',
        '<g fill="#000000" stroke="none">',
    ]
    rects = []
    for r, c in mask_cells:
        x0, y0, x1, y1 = geom.rects[r * geom.Gw + c]
        x = float(x0) * scale_x
        y = float(y0) * scale_y
        w = float(x1 - x0) * scale_x
        h = float(y1 - y0) * scale_y
        rects.append(f'<rect x="{x:.3f}" y="{y:.3f}" width="{w:.3f}" height="{h:.3f}" />')
    footer = ["</g>", "</svg>"]
    svg = "\n".join(header + rects + footer) + "\n"
    path.write_text(svg, encoding="utf-8")


def _normal(v1: tuple[float, float, float], v2: tuple[float, float, float], v3: tuple[float, float, float]) -> tuple[float, float, float]:
    ax, ay, az = v2[0] - v1[0], v2[1] - v1[1], v2[2] - v1[2]
    bx, by, bz = v3[0] - v1[0], v3[1] - v1[1], v3[2] - v1[2]
    nx = ay * bz - az * by
    ny = az * bx - ax * bz
    nz = ax * by - ay * bx
    norm = math.sqrt(nx * nx + ny * ny + nz * nz)
    if norm <= 0:
        return (0.0, 0.0, 0.0)
    return (nx / norm, ny / norm, nz / norm)


def _add_tri(tris: list[tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]],
             v1: tuple[float, float, float],
             v2: tuple[float, float, float],
             v3: tuple[float, float, float]) -> None:
    tris.append((_normal(v1, v2, v3), v1, v2, v3))


def _add_quad(tris: list[tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]],
              v0: tuple[float, float, float],
              v1: tuple[float, float, float],
              v2: tuple[float, float, float],
              v3: tuple[float, float, float]) -> None:
    _add_tri(tris, v0, v1, v2)
    _add_tri(tris, v0, v2, v3)


def _write_binary_stl(path: Path, tris: list[tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]]) -> None:
    header = b"stop_sign_cover".ljust(80, b" ")
    with path.open("wb") as f:
        f.write(header[:80])
        f.write(struct.pack("<I", len(tris)))
        for n, v1, v2, v3 in tris:
            f.write(struct.pack("<3f", float(n[0]), float(n[1]), float(n[2])))
            f.write(struct.pack("<3f", float(v1[0]), float(v1[1]), float(v1[2])))
            f.write(struct.pack("<3f", float(v2[0]), float(v2[1]), float(v2[2])))
            f.write(struct.pack("<3f", float(v3[0]), float(v3[1]), float(v3[2])))
            f.write(struct.pack("<H", 0))


def _triangles_for_tile(
    geom: GridGeom,
    mask: np.ndarray,
    r0: int,
    r1: int,
    c0: int,
    c1: int,
    scale_x: float,
    scale_y: float,
    thickness_mm: float,
    tile_x0_mm: float,
    tile_y0_mm: float,
) -> list[tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]]:
    tris: list[tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]] = []
    z0 = 0.0
    z1 = float(thickness_mm)
    for r in range(r0, r1):
        for c in range(c0, c1):
            if not mask[r, c]:
                continue
            x0_mm, y0_mm, x1_mm, y1_mm = _rect_mm_from_cell(geom, r, c, scale_x, scale_y)
            x0 = x0_mm - tile_x0_mm
            x1 = x1_mm - tile_x0_mm
            y0 = y0_mm - tile_y0_mm
            y1 = y1_mm - tile_y0_mm

            # Neighbor checks within tile only; outside tile = boundary.
            has_left = (c - 1 >= c0) and mask[r, c - 1]
            has_right = (c + 1 < c1) and mask[r, c + 1]
            has_down = (r + 1 < r1) and mask[r + 1, c]
            has_up = (r - 1 >= r0) and mask[r - 1, c]

            # Top (+Z)
            _add_quad(tris,
                      (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1))
            # Bottom (-Z)
            _add_quad(tris,
                      (x0, y0, z0), (x0, y1, z0), (x1, y1, z0), (x1, y0, z0))

            # Left (-X)
            if not has_left:
                _add_quad(tris,
                          (x0, y0, z0), (x0, y0, z1), (x0, y1, z1), (x0, y1, z0))
            # Right (+X)
            if not has_right:
                _add_quad(tris,
                          (x1, y0, z0), (x1, y1, z0), (x1, y1, z1), (x1, y0, z1))
            # Down (-Y)
            if not has_down:
                _add_quad(tris,
                          (x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1))
            # Up (+Y)
            if not has_up:
                _add_quad(tris,
                          (x0, y1, z0), (x0, y1, z1), (x1, y1, z1), (x1, y1, z0))
    return tris


def _export_stl_tiles(
    out_prefix: Path,
    geom: GridGeom,
    mask_cells: list[tuple[int, int]],
    sign_w_mm: float,
    sign_h_mm: float,
    thickness_mm: float,
    tile_max_mm: float,
) -> list[str]:
    scale_x = sign_w_mm / float(geom.W)
    scale_y = sign_h_mm / float(geom.H)

    mask = np.zeros((geom.Gh, geom.Gw), dtype=bool)
    for r, c in mask_cells:
        if 0 <= r < geom.Gh and 0 <= c < geom.Gw:
            mask[r, c] = True

    # Compute per-column/row physical sizes to split on cell boundaries.
    col_widths = []
    for c in range(geom.Gw):
        x0, _, x1, _ = geom.rects[c]
        col_widths.append(float(x1 - x0) * scale_x)
    row_heights = []
    for r in range(geom.Gh):
        _, y0, _, y1 = geom.rects[r * geom.Gw]
        row_heights.append(float(y1 - y0) * scale_y)

    total_w = sum(col_widths)
    total_h = sum(row_heights)
    tile_max = float(tile_max_mm)
    if tile_max <= 0:
        tile_max = max(total_w, total_h)
    n_x = max(1, int(math.ceil(total_w / tile_max)))
    n_y = max(1, int(math.ceil(total_h / tile_max)))

    col_splits = _split_indices_by_mm(col_widths, n_x)
    row_splits = _split_indices_by_mm(row_heights, n_y)

    # Precompute column and row origins in mm (bottom-left origin for STL).
    col_x0_mm = [0.0] * geom.Gw
    acc = 0.0
    for c in range(geom.Gw):
        col_x0_mm[c] = acc
        acc += col_widths[c]

    # Row bottom coordinates in STL space (y up, origin at bottom-left).
    row_y0_mm = [0.0] * geom.Gh
    for r in range(geom.Gh):
        _, y0, _, y1 = geom.rects[r * geom.Gw]
        row_y0_mm[r] = float(geom.H - y1) * scale_y

    tile_paths: list[str] = []
    tile_idx = 0
    for ry, (r0, r1) in enumerate(row_splits):
        for cx, (c0, c1) in enumerate(col_splits):
            tile_idx += 1
            tile_x0 = col_x0_mm[c0]
            tile_y0 = row_y0_mm[r1 - 1]
            tris = _triangles_for_tile(
                geom=geom,
                mask=mask,
                r0=r0,
                r1=r1,
                c0=c0,
                c1=c1,
                scale_x=scale_x,
                scale_y=scale_y,
                thickness_mm=thickness_mm,
                tile_x0_mm=tile_x0,
                tile_y0_mm=tile_y0,
            )
            if n_x == 1 and n_y == 1:
                stl_path = out_prefix.with_name(out_prefix.name + ".stl")
            else:
                stl_path = out_prefix.with_name(out_prefix.name + f"_tile_{ry+1:02d}x{cx+1:02d}.stl")
            _write_binary_stl(stl_path, tris)
            tile_paths.append(str(stl_path))
    return tile_paths


def _draw_tile_registration(draw: ImageDraw.ImageDraw, x0: int, y0: int, x1: int, y1: int):
    col = (0, 0, 0, 160)
    L = 12
    for (x, y, sx, sy) in [(x0, y0, +1, +1), (x1, y0, -1, +1), (x0, y1, +1, -1), (x1, y1, -1, -1)]:
        draw.line([(x, y), (x + sx * L, y)], fill=col, width=1)
        draw.line([(x, y), (x, y + sy * L)], fill=col, width=1)


def _make_tiled_standard_pages(
    master_page: Image.Image,
    dpi: int,
    paper_size: str,
    paper_orientation: str,
    paper_margin_in: float,
    tile_overlap_in: float,
) -> tuple[list[Image.Image], dict[str, Any]]:
    """Tile a full-scale master page onto standard paper without resizing (print at 100%)."""
    master_w, master_h = master_page.size
    paper_w_in, paper_h_in = _paper_size_inches(
        paper_size, paper_orientation, content_size_in=(master_w / dpi, master_h / dpi)
    )
    paper_w_px = int(round(paper_w_in * dpi))
    paper_h_px = int(round(paper_h_in * dpi))
    margin_px = max(0, int(round(paper_margin_in * dpi)))
    overlap_px = max(0, int(round(tile_overlap_in * dpi)))
    printable_w = paper_w_px - 2 * margin_px
    printable_h = paper_h_px - 2 * margin_px
    if printable_w <= 0 or printable_h <= 0:
        raise ValueError("Standard paper margin too large for selected paper size/DPI.")
    if overlap_px >= printable_w or overlap_px >= printable_h:
        raise ValueError("tile_overlap is too large for the printable area.")

    stride_w = max(1, printable_w - overlap_px)
    stride_h = max(1, printable_h - overlap_px)
    xs = _tile_starts(master_w, printable_w, stride_w)
    ys = _tile_starts(master_h, printable_h, stride_h)

    pages: list[Image.Image] = []
    tile_entries: list[dict[str, Any]] = []
    total_tiles = len(xs) * len(ys)
    font_title = _load_font(max(16, int(0.10 * dpi)))
    font_meta = _load_font(max(12, int(0.075 * dpi)))

    tile_idx = 0
    for r, y0 in enumerate(ys):
        for c, x0 in enumerate(xs):
            tile_idx += 1
            x1 = min(master_w, x0 + printable_w)
            y1 = min(master_h, y0 + printable_h)
            crop = master_page.crop((x0, y0, x1, y1))

            page = Image.new("RGBA", (paper_w_px, paper_h_px), (255, 255, 255, 255))
            draw = ImageDraw.Draw(page, "RGBA")
            page.alpha_composite(crop, (margin_px, margin_px))

            cx0, cy0 = margin_px, margin_px
            cx1, cy1 = margin_px + crop.size[0], margin_px + crop.size[1]
            draw.rectangle([cx0, cy0, cx1, cy1], outline=(120, 120, 120, 200), width=2)
            _draw_tile_registration(draw, cx0, cy0, cx1, cy1)

            draw.text((margin_px, paper_h_px - margin_px - int(0.22 * dpi)),
                      f"Stencil tile {tile_idx}/{total_tiles}  (row {r+1}, col {c+1})",
                      fill=(0, 0, 0, 255), font=font_title)
            draw.text((margin_px, paper_h_px - margin_px - int(0.10 * dpi)),
                      "Print at 100% scale (disable 'fit to page'). Align tile borders + registration marks.",
                      fill=(60, 60, 60, 255), font=font_meta)

            pages.append(page)
            tile_entries.append({
                "tile_index": tile_idx,
                "row": r + 1,
                "col": c + 1,
                "crop_xywh_px": [int(x0), int(y0), int(crop.size[0]), int(crop.size[1])],
                "paste_xy_px": [int(margin_px), int(margin_px)],
            })

    meta = {
        "paper_size": str(paper_size).lower(),
        "paper_orientation": str(paper_orientation).lower(),
        "paper_width_in": float(paper_w_in),
        "paper_height_in": float(paper_h_in),
        "paper_px": [int(paper_w_px), int(paper_h_px)],
        "paper_margin_in": float(paper_margin_in),
        "paper_margin_px": int(margin_px),
        "tile_overlap_in": float(tile_overlap_in),
        "tile_overlap_px": int(overlap_px),
        "printable_px": [int(printable_w), int(printable_h)],
        "grid_tiles": {"rows": len(ys), "cols": len(xs), "count": len(pages)},
        "tiles": tile_entries,
    }
    return pages, meta


def main() -> int:
    ap = argparse.ArgumentParser(description="Export a printable grid stencil for a replica stop sign.")
    ap.add_argument("--data", default="./data", help="Repo data folder containing stop_sign.png")
    ap.add_argument("--sign-image", default="", help="Optional explicit sign image path (defaults to data/stop_sign.png)")
    ap.add_argument("--grid-cell", type=int, default=16, choices=[2, 4, 8, 16, 32], help="Grid cell size in sign-image pixels (match training)")
    ap.add_argument("--cell-cover-thresh", type=float, default=0.60, help="Valid-cell alpha coverage threshold (match training)")
    ap.add_argument("--selected-rc", default="", help="Selected cells as 'r,c;r,c;...' (row,col)")
    ap.add_argument("--selected-indices", default="", help="Selected flat indices as 'i,j,k,...' (row-major over full Gh*Gw)")
    ap.add_argument("--from-json", default="", help="Optional JSON file containing selected_indices (e.g., trace.selected_indices)")
    ap.add_argument("--episode-index", type=int, default=0, help="Episode index if JSON contains a list/episodes_detail")
    ap.add_argument("--sign-width", type=float, required=True, help="Replica sign printed width")
    ap.add_argument("--sign-height", type=float, required=True, help="Replica sign printed height")
    ap.add_argument("--units", default="in", choices=["in", "cm", "mm"], help="Units for sign width/height")
    ap.add_argument("--dpi", type=int, default=300, help="Raster print DPI (e.g., 300)")
    ap.add_argument("--margin", type=float, default=0.5, help="Page margin around sign (same units as sign size)")
    ap.add_argument("--paper-size", default="custom", choices=["custom", "letter", "a4"],
                    help="If set to letter/a4, also export tiled standard-paper pages at true scale.")
    ap.add_argument("--paper-orientation", default="auto", choices=["auto", "portrait", "landscape"],
                    help="Orientation for standard-paper tiled pages.")
    ap.add_argument("--tile-content", default="sign-only", choices=["sign-only", "full-page"],
                    help="What to tile onto standard paper: the sign region only (recommended) or the full stencil page.")
    ap.add_argument("--paper-margin", type=float, default=0.5,
                    help="Printable margin on each standard tile page (same units as sign size).")
    ap.add_argument("--tile-overlap", type=float, default=0.20,
                    help="Overlap between neighboring standard-paper tiles (same units as sign size).")
    ap.add_argument("--label-cells", action="store_true", help="Print selected cell (r,c) list on page")
    ap.add_argument("--mask-mode", default="selected", choices=["selected", "cover"],
                    help="Mask mode for SVG/STL output: selected cells (stencil) or cover (valid minus selected).")
    ap.add_argument("--export-svg", action="store_true", help="Export SVG of the selected/cover mask at real size.")
    ap.add_argument("--export-stl", action="store_true", help="Export STL (mm units) of the selected/cover mask.")
    ap.add_argument("--thickness-mm", type=float, default=1.6, help="STL extrusion thickness in mm.")
    ap.add_argument("--tile-max-mm", type=float, default=300.0,
                    help="Max tile size for STL (mm). Set <=0 to disable tiling.")
    ap.add_argument("--out-prefix", default="./_runs/paper_data/replica_stencil/replica_stop_sign", help="Output file prefix")
    args = ap.parse_args()

    data_dir = Path(args.data)
    sign_path = Path(args.sign_image) if args.sign_image else (data_dir / "stop_sign.png")
    if not sign_path.exists():
        raise FileNotFoundError(f"Sign image not found: {sign_path}")

    sign_rgba = Image.open(sign_path).convert("RGBA")
    geom = _build_grid(sign_rgba, grid_cell_px=int(args.grid_cell), cell_cover_thresh=float(args.cell_cover_thresh))

    selected_cells: list[tuple[int, int]] = []
    if args.from_json:
        selected_cells.extend(_load_selected_from_json(Path(args.from_json), geom, episode_index=int(args.episode_index)))
    if args.selected_indices:
        selected_cells.extend(_parse_selected_indices(args.selected_indices, geom))
    if args.selected_rc:
        selected_cells.extend(_parse_selected_rc(args.selected_rc, geom))
    selected_cells = _dedupe_cells(selected_cells)

    if not selected_cells:
        raise ValueError(
            "No selected cells provided. Use one of: --selected-rc, --selected-indices, or --from-json "
            "(containing trace.selected_indices)."
        )

    # Validate against valid-cell mask and warn if selected includes invalid cells
    invalid_selected = [(r, c) for (r, c) in selected_cells if not geom.valid[r, c]]
    if invalid_selected:
        print(f"[WARN] {len(invalid_selected)} selected cells are invalid under current grid settings and will still be drawn/clipped: {invalid_selected[:10]}")

    sign_w_in = _to_inches(float(args.sign_width), args.units)
    sign_h_in = _to_inches(float(args.sign_height), args.units)
    margin_in = _to_inches(float(args.margin), args.units)
    sign_w_mm = sign_w_in * 25.4
    sign_h_mm = sign_h_in * 25.4

    page, cutmask = _draw_page(
        sign_rgba=sign_rgba,
        geom=geom,
        selected_cells=selected_cells,
        sign_width_in=sign_w_in,
        sign_height_in=sign_h_in,
        dpi=int(args.dpi),
        margin_in=margin_in,
        label_cells=bool(args.label_cells),
        units=str(args.units),
    )

    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    png_page = out_prefix.with_name(out_prefix.name + "_stencil_page.png")
    pdf_page = out_prefix.with_name(out_prefix.name + "_stencil_page.pdf")
    cutmask_png = out_prefix.with_name(out_prefix.name + "_cutmask.png")
    meta_json = out_prefix.with_name(out_prefix.name + "_metadata.json")

    page.save(png_page, format="PNG", dpi=(int(args.dpi), int(args.dpi)))
    page.convert("RGB").save(pdf_page, format="PDF", resolution=int(args.dpi))
    cutmask.save(cutmask_png, format="PNG", dpi=(int(args.dpi), int(args.dpi)))

    mask_mode = str(args.mask_mode).lower().strip()
    mask_cells = _mask_cells_from_mode(geom, selected_cells, mask_mode)

    svg_path = None
    if bool(args.export_svg):
        svg_path = out_prefix.with_name(out_prefix.name + f"_{mask_mode}_mask.svg")
        _export_svg(svg_path, geom, mask_cells, sign_w_mm, sign_h_mm)

    stl_paths: list[str] = []
    if bool(args.export_stl):
        stl_prefix = out_prefix.with_name(out_prefix.name + f"_{mask_mode}_mask")
        stl_paths = _export_stl_tiles(
            out_prefix=stl_prefix,
            geom=geom,
            mask_cells=mask_cells,
            sign_w_mm=sign_w_mm,
            sign_h_mm=sign_h_mm,
            thickness_mm=float(args.thickness_mm),
            tile_max_mm=float(args.tile_max_mm),
        )

    meta = {
        "sign_image": str(sign_path),
        "grid_cell_px": int(args.grid_cell),
        "cell_cover_thresh": float(args.cell_cover_thresh),
        "grid_shape": {"Gh": int(geom.Gh), "Gw": int(geom.Gw)},
        "valid_cells_count": int(geom.valid.sum()),
        "selected_cells_rc": [[int(r), int(c)] for (r, c) in selected_cells],
        "selected_indices_flat": [int(r * geom.Gw + c) for (r, c) in selected_cells],
        "sign_physical": {
            "width_input": float(args.sign_width),
            "height_input": float(args.sign_height),
            "units": str(args.units),
            "width_in": sign_w_in,
            "height_in": sign_h_in,
            "width_mm": float(sign_w_mm),
            "height_mm": float(sign_h_mm),
        },
        "print": {"dpi": int(args.dpi), "margin_input": float(args.margin), "margin_units": str(args.units), "margin_in": margin_in},
        "outputs": {
            "stencil_page_png": str(png_page),
            "stencil_page_pdf": str(pdf_page),
            "cutmask_png": str(cutmask_png),
            "mask_mode": mask_mode,
            "mask_svg": str(svg_path) if svg_path else "",
            "mask_stl": stl_paths,
        },
    }

    # Optional standard-paper tiled output (Letter/A4) with no resizing; preserves physical scale.
    if str(args.paper_size).lower() != "custom":
        paper_margin_in = _to_inches(float(args.paper_margin), args.units)
        tile_overlap_in = _to_inches(float(args.tile_overlap), args.units)
        sign_w_px = int(round(sign_w_in * int(args.dpi)))
        sign_h_px = int(round(sign_h_in * int(args.dpi)))
        margin_px = int(round(margin_in * int(args.dpi)))
        header_px = int(round(0.45 * int(args.dpi)))
        # Crop tiled source to sign region by default to avoid wasting paper on headers/footers.
        tile_content = str(args.tile_content).lower()
        if tile_content == "sign-only":
            tile_master = page.crop((margin_px, margin_px + header_px, margin_px + sign_w_px, margin_px + header_px + sign_h_px))
        else:
            tile_master = page

        tiled_pages, tiled_meta = _make_tiled_standard_pages(
            master_page=tile_master,
            dpi=int(args.dpi),
            paper_size=str(args.paper_size),
            paper_orientation=str(args.paper_orientation),
            paper_margin_in=paper_margin_in,
            tile_overlap_in=tile_overlap_in,
        )
        tiles_dir = out_prefix.with_name(out_prefix.name + f"_tiles_{str(args.paper_size).lower()}")
        tiles_dir.mkdir(parents=True, exist_ok=True)
        tile_png_paths: list[str] = []
        for i, tile in enumerate(tiled_pages, start=1):
            p = tiles_dir / f"tile_{i:03d}.png"
            tile.save(p, format="PNG", dpi=(int(args.dpi), int(args.dpi)))
            tile_png_paths.append(str(p))
        tiled_pdf = out_prefix.with_name(out_prefix.name + f"_stencil_tiled_{str(args.paper_size).lower()}.pdf")
        if tiled_pages:
            tiled_rgb = [im.convert("RGB") for im in tiled_pages]
            tiled_rgb[0].save(
                tiled_pdf,
                format="PDF",
                save_all=True,
                append_images=tiled_rgb[1:],
                resolution=int(args.dpi),
            )
        meta["standard_paper_tiles"] = {
            "enabled": True,
            "paper_size": str(args.paper_size).lower(),
            "paper_orientation": str(args.paper_orientation).lower(),
            "tile_content": tile_content,
            "paper_margin_input": float(args.paper_margin),
            "paper_margin_units": str(args.units),
            "paper_margin_in": float(paper_margin_in),
            "tile_overlap_input": float(args.tile_overlap),
            "tile_overlap_units": str(args.units),
            "tile_overlap_in": float(tile_overlap_in),
            "tiles_dir": str(tiles_dir),
            "tile_png_paths": tile_png_paths,
            "tiled_pdf": str(tiled_pdf),
            "tiling": tiled_meta,
            "print_instructions": "Print at 100% scale / actual size. Disable fit-to-page scaling.",
        }
        print(f"[SAVE] {tiled_pdf}")
        print(f"[SAVE] {tiles_dir} ({len(tiled_pages)} tile PNGs)")
        print(f"[INFO] tiled standard-paper output: {tiled_meta['grid_tiles']['rows']}x{tiled_meta['grid_tiles']['cols']} pages")
    else:
        meta["standard_paper_tiles"] = {"enabled": False}

    meta_json.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"[SAVE] {png_page}")
    print(f"[SAVE] {pdf_page}")
    print(f"[SAVE] {cutmask_png}")
    if svg_path:
        print(f"[SAVE] {svg_path}")
    if stl_paths:
        if len(stl_paths) == 1:
            print(f"[SAVE] {stl_paths[0]}")
        else:
            print(f"[SAVE] {len(stl_paths)} STL tiles")
    print(f"[SAVE] {meta_json}")
    print(
        f"[INFO] grid={geom.Gh}x{geom.Gw} valid={int(geom.valid.sum())} selected={len(selected_cells)} "
        f"sign={sign_w_in:.2f}x{sign_h_in:.2f} in @ {int(args.dpi)} dpi"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
