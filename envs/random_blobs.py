# envs/random_blobs.py
from typing import Tuple, List, Dict, Any
from PIL import Image, ImageDraw, ImageChops
import numpy as np

def _blob_mask_to_area_frac(mask: Image.Image, allowed_mask: Image.Image, crop_to_mask: bool = True) -> float:
    """
    Fraction of allowed_mask area covered by `mask` (union so far).
    """
    am = np.array(allowed_mask, dtype=np.uint8) > 0
    mm = np.array(mask, dtype=np.uint8) > 0
    covered = (am & mm).sum()
    total = am.sum() if crop_to_mask else (mask.size[0] * mask.size[1])
    return float(covered) / float(total + 1e-9)

def draw_randomized_blobs_set(
    base_pil: Image.Image,
    count: int,
    size_scale: float,
    alpha: float,      
    color_mean: Tuple[int, int, int],
    color_std: float,                 # unused when single_color=True
    mode: str,
    rng: np.random.Generator,
    allowed_mask: Image.Image = None, # L mask: where blobs are permitted
    area_cap: float = 0.30,           # cap as fraction of allowed_mask area
    cap_relative_to_mask: bool = True,# kept for API compatibility
    single_color: bool = True
) -> Tuple[Image.Image, List[Dict[str, Any]]]:
    """
    Draw up to `count` organic blobs. All blobs use the SAME color when single_color=True.
    Returns (composited_RGB_image, per_blob_meta).
    """
    W, H = base_pil.size
    base_rgb = base_pil.convert("RGB")
    comp = base_rgb.copy()  

    if allowed_mask is None:
        allowed_mask = Image.new("L", (W, H), 255)

    allowed_np = (np.array(allowed_mask, dtype=np.uint8) > 0)
    ys, xs = np.nonzero(allowed_np)
    if len(xs) == 0:
        ys, xs = np.nonzero(np.ones((H, W), dtype=bool))  # degenerate: allow everywhere

    # one color per set
    color = tuple(int(np.clip(v, 0, 255)) for v in color_mean)

    # union of all blob masks (binary)
    union_mask = Image.new("L", (W, H), 0)

    metas: List[Dict[str, Any]] = []
    total_area_frac = 0.0

    for i in range(int(count)):
        if total_area_frac >= area_cap - 1e-6:
            break

        # center sampled INSIDE allowed region
        idx = int(rng.integers(0, len(xs)))
        cx, cy = float(xs[idx]), float(ys[idx])

        # organic polygon footprint (ragged ellipse)
        rx = size_scale * rng.uniform(0.01, 0.04) * W
        ry = size_scale * rng.uniform(0.01, 0.04) * H
        k = int(rng.integers(8, 16))
        angles = np.linspace(0, 2*np.pi, k, endpoint=False) + rng.uniform(0, 2*np.pi)
        rj = rng.uniform(0.7, 1.3, size=k)
        pts = [(cx + rx * rj[j] * np.cos(a), cy + ry * rj[j] * np.sin(a)) for j, a in enumerate(angles)]

        # make a hard-edged (binary) mask for this blob
        blob_mask = Image.new("L", (W, H), 0)
        ImageDraw.Draw(blob_mask).polygon(pts, fill=255)
        # binarize to ensure no anti-aliased alpha remains
        blob_mask = blob_mask.point(lambda v: 255 if v >= 1 else 0)

        # update union (pre-clip)
        union_mask = ImageChops.lighter(union_mask, blob_mask)

        # compute new total area inside allowed mask
        new_total = _blob_mask_to_area_frac(union_mask, allowed_mask, crop_to_mask=True)
        added_area = max(0.0, new_total - total_area_frac)
        total_area_frac = new_total

        metas.append({
            "mode": mode,
            "alpha": 1.0,  
            "color": color,
            "area_frac": float(added_area)
        })

        if total_area_frac >= area_cap - 1e-6:
            break

    # final mask: restrict union to allowed region
    if allowed_mask.mode != "L":
        allowed_mask = allowed_mask.convert("L")
    final_mask = ImageChops.multiply(union_mask, allowed_mask)

    # HARD paste: opaque color where mask=255
    comp.paste(color, mask=final_mask)

    return comp, metas
