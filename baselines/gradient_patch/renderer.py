"""Differentiable renderer for stop-sign patch optimization."""
from __future__ import annotations

from typing import List, Tuple, Optional, Dict, Any
import math
import random
import numpy as np
import torch
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode


def pil_to_tensor_rgb(pil_img) -> torch.Tensor:
    arr = np.array(pil_img.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def pil_to_tensor_rgba(pil_img) -> torch.Tensor:
    arr = np.array(pil_img.convert("RGBA"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def pil_to_tensor_gray(pil_img) -> torch.Tensor:
    arr = np.array(pil_img.convert("L"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0).contiguous()


def tensor_to_pil_rgb(t: torch.Tensor):
    t = t.detach().clamp(0.0, 1.0).cpu()
    arr = (t.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
    from PIL import Image
    return Image.fromarray(arr, mode="RGB")


def tensor_to_pil_rgba(t: torch.Tensor):
    t = t.detach().clamp(0.0, 1.0).cpu()
    arr = (t.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
    from PIL import Image
    return Image.fromarray(arr, mode="RGBA")


def tensor_to_pil_gray(t: torch.Tensor):
    t = t.detach().clamp(0.0, 1.0).cpu()
    arr = (t.squeeze(0).numpy() * 255.0).astype(np.uint8)
    from PIL import Image
    return Image.fromarray(arr, mode="L")


def alpha_composite(bg_rgb: torch.Tensor, fg_rgba: torch.Tensor) -> torch.Tensor:
    """
    Alpha composite fg over bg.

    @param bg_rgb: (3, H, W)
    @param fg_rgba: (4, H, W)
    @return: (3, H, W)
    """
    fg_rgb = fg_rgba[:3]
    a = fg_rgba[3:4].clamp(0.0, 1.0)
    return fg_rgb * a + bg_rgb * (1.0 - a)


def alpha_composite_rgba(bg_rgba: torch.Tensor, fg_rgba: torch.Tensor) -> torch.Tensor:
    """
    Alpha composite fg over bg (RGBA->RGBA).
    """
    bg_rgb = bg_rgba[:3]
    bg_a = bg_rgba[3:4].clamp(0.0, 1.0)
    fg_rgb = fg_rgba[:3]
    fg_a = fg_rgba[3:4].clamp(0.0, 1.0)
    out_a = fg_a + bg_a * (1.0 - fg_a)
    out_rgb = (fg_rgb * fg_a + bg_rgb * bg_a * (1.0 - fg_a)) / out_a.clamp(min=1e-6)
    return torch.cat([out_rgb, out_a], dim=0)


def apply_patch(
    sign_rgba: torch.Tensor,
    sign_alpha: torch.Tensor,
    mask: torch.Tensor,
    color_rgb: torch.Tensor,
    alpha: float,
    uv_min_alpha: float = 0.08,
    enforce_uv_min: bool = False,
) -> torch.Tensor:
    """
    Apply a colored patch mask to the sign.

    @param sign_rgba: (4, H, W)
    @param sign_alpha: (1, H, W) sign alpha mask in [0,1]
    @param mask: (1, H, W) patch mask in [0,1]
    @param color_rgb: (3, H, W) color tensor in [0,1]
    @param alpha: base alpha for the patch (0..1)
    @param uv_min_alpha: minimum alpha for UV mode
    @param enforce_uv_min: clamp alpha to uv_min_alpha if True
    """
    rgb = sign_rgba[:3]
    a = sign_alpha.clamp(0.0, 1.0)
    m = mask.clamp(0.0, 1.0) * a
    alpha_map = m * float(alpha)
    if enforce_uv_min and uv_min_alpha > 0.0:
        alpha_map = torch.maximum(alpha_map, m * float(uv_min_alpha))
    out_rgb = color_rgb * alpha_map + rgb * (1.0 - alpha_map)
    return torch.cat([out_rgb, a], dim=0)


def _rand_uniform(rng: random.Random, low: float, high: float) -> float:
    return low + (high - low) * rng.random()


def sample_transform_params(
    rng: random.Random,
    strength: float,
    img_size: Tuple[int, int],
) -> Dict[str, Any]:
    W, H = int(img_size[0]), int(img_size[1])
    strength = max(0.0, min(1.0, float(strength)))

    angle = _rand_uniform(rng, -6.0 * strength, 6.0 * strength)
    shear = _rand_uniform(rng, -4.0 * strength, 4.0 * strength)
    scale = 1.0 + _rand_uniform(rng, -0.10 * strength, 0.10 * strength)
    tx = _rand_uniform(rng, -0.02 * strength * W, 0.02 * strength * W)
    ty = _rand_uniform(rng, -0.02 * strength * H, 0.02 * strength * H)

    do_persp = rng.random() < 0.5 * strength
    max_shift = 0.06 * strength
    if do_persp:
        dx, dy = W * max_shift, H * max_shift
        start = [(0, 0), (W, 0), (W, H), (0, H)]
        end = [
            (_rand_uniform(rng, -dx, dx), _rand_uniform(rng, -dy, dy)),
            (W + _rand_uniform(rng, -dx, dx), _rand_uniform(rng, -dy, dy)),
            (W + _rand_uniform(rng, -dx, dx), H + _rand_uniform(rng, -dy, dy)),
            (_rand_uniform(rng, -dx, dx), H + _rand_uniform(rng, -dy, dy)),
        ]
    else:
        start = end = None

    bright = _rand_uniform(rng, 1.0 - 0.1 * strength, 1.0 + 0.1 * strength) if rng.random() < 0.7 * strength else 1.0
    contrast = _rand_uniform(rng, 1.0 - 0.1 * strength, 1.0 + 0.1 * strength) if rng.random() < 0.7 * strength else 1.0
    saturation = _rand_uniform(rng, 1.0 - 0.1 * strength, 1.0 + 0.1 * strength) if rng.random() < 0.3 * strength else 1.0
    blur_sigma = _rand_uniform(rng, 0.0, 0.8 * strength) if rng.random() < 0.4 * strength else 0.0
    noise_sigma = _rand_uniform(rng, 1.0 * strength, 3.0 * strength) if rng.random() < 0.6 * strength else 0.0

    return {
        "angle": angle,
        "shear": shear,
        "scale": scale,
        "tx": tx,
        "ty": ty,
        "persp_start": start,
        "persp_end": end,
        "bright": bright,
        "contrast": contrast,
        "saturation": saturation,
        "blur_sigma": blur_sigma,
        "noise_sigma": noise_sigma,
    }


def apply_transform_rgba(
    img_rgba: torch.Tensor,
    params: Dict[str, Any],
    img_size: Tuple[int, int],
) -> torch.Tensor:
    """
    Apply differentiable transforms to RGBA tensor (4, H, W).
    """
    W, H = int(img_size[0]), int(img_size[1])
    out = img_rgba

    # Affine
    out = TF.affine(
        out,
        angle=float(params["angle"]),
        translate=[float(params["tx"]), float(params["ty"])],
        scale=float(params["scale"]),
        shear=[float(params["shear"]), 0.0],
        interpolation=InterpolationMode.BILINEAR,
        fill=0.0,
    )

    # Perspective
    if params["persp_start"] is not None and params["persp_end"] is not None:
        out = TF.perspective(
            out,
            startpoints=params["persp_start"],
            endpoints=params["persp_end"],
            interpolation=InterpolationMode.BILINEAR,
            fill=0.0,
        )

    # Color jitter (RGB only)
    rgb = out[:3]
    a = out[3:4]
    if params["bright"] != 1.0:
        rgb = TF.adjust_brightness(rgb, params["bright"])
    if params["contrast"] != 1.0:
        rgb = TF.adjust_contrast(rgb, params["contrast"])
    if params["saturation"] != 1.0:
        rgb = TF.adjust_saturation(rgb, params["saturation"])
    out = torch.cat([rgb, a], dim=0)

    # Blur
    if params["blur_sigma"] > 0.0:
        sigma = float(params["blur_sigma"])
        k = max(1, int(round(2 * sigma + 1)))
        k = k + 1 if k % 2 == 0 else k
        out = TF.gaussian_blur(out, kernel_size=[k, k], sigma=[sigma, sigma])

    # Noise (RGB only)
    if params["noise_sigma"] > 0.0:
        sigma = float(params["noise_sigma"]) / 255.0
        noise = torch.randn_like(out[:3]) * sigma
        rgb = (out[:3] + noise).clamp(0.0, 1.0)
        out = torch.cat([rgb, out[3:4]], dim=0)

    return out


def compose_sign_and_pole(
    sign_rgba: torch.Tensor,
    pole_rgba: Optional[torch.Tensor],
    pole_width_ratio: float = 0.12,
    bottom_len_factor: float = 4.0,
    clearance_px: int = 2,
    side_margin_frac: float = 0.06,
) -> torch.Tensor:
    """
    Compose sign with pole into an RGBA group image.
    """
    if pole_rgba is None:
        return sign_rgba

    _, SH, SW = sign_rgba.shape
    pole = pole_rgba
    _, PH, PW = pole.shape

    target_pw = max(2, int(pole_width_ratio * SW))
    scale_w = target_pw / max(1, PW)
    target_ph = max(1, int(PH * scale_w))
    pole = TF.resize(pole, size=[target_ph, target_pw], interpolation=InterpolationMode.BILINEAR)

    H_needed = clearance_px + SH + int(bottom_len_factor * SH)
    if pole.shape[1] < H_needed:
        scale_h = H_needed / max(1, pole.shape[1])
        new_h = int(round(pole.shape[1] * scale_h))
        new_w = max(1, int(round(pole.shape[2] * scale_h)))
        pole = TF.resize(pole, size=[new_h, new_w], interpolation=InterpolationMode.BILINEAR)
    pole = pole[:, :H_needed, :]

    side_margin = int(side_margin_frac * SW)
    GW = max(pole.shape[2], SW + 2 * side_margin)
    GH = H_needed
    group = torch.zeros((4, GH, GW), dtype=sign_rgba.dtype, device=sign_rgba.device)

    px = (GW - pole.shape[2]) // 2
    group[:, 0:pole.shape[1], px:px + pole.shape[2]] = pole

    sx = (GW - SW) // 2
    sy = clearance_px
    sign_region = group[:, sy:sy + SH, sx:sx + SW]
    group[:, sy:sy + SH, sx:sx + SW] = alpha_composite_rgba(sign_region, sign_rgba)

    return group


def sample_placement(
    rng: random.Random,
    group_rgba: torch.Tensor,
    img_size: Tuple[int, int],
) -> Dict[str, int]:
    """
    Sample a placement (x, y, target_w, target_h) for the group.
    """
    W, H = int(img_size[0]), int(img_size[1])
    target_w = int(_rand_uniform(rng, 0.30 * W, 0.50 * W))
    scale = target_w / max(1, group_rgba.shape[2])
    target_h = max(1, int(round(group_rgba.shape[1] * scale)))

    margin = int(0.04 * W)
    max_x = max(margin, W - target_w - margin)
    max_y = max(margin, H - target_h - margin)
    left_max = max(margin, min(max_x, int(0.40 * W)))
    right_min = max(margin, min(max_x, int(0.60 * W)))
    if rng.random() < 0.5:
        x = int(rng.randint(margin, left_max if left_max >= margin else margin))
    else:
        x = int(rng.randint(right_min, max_x if max_x >= right_min else right_min))

    min_y = max(margin, int(0.12 * H))
    if min_y > max_y:
        min_y = max(margin, max_y)
    y = int(rng.randint(min_y, max_y if max_y >= min_y else min_y))
    return {"x": x, "y": y, "target_w": target_w, "target_h": target_h}


def place_group_on_background(
    group_rgba: torch.Tensor,
    bg_rgb: torch.Tensor,
    rng: random.Random,
    img_size: Tuple[int, int],
    placement: Optional[Dict[str, int]] = None,
) -> torch.Tensor:
    """
    Place the RGBA group onto the background, returning RGB canvas.
    """
    W, H = int(img_size[0]), int(img_size[1])
    bg = TF.resize(bg_rgb, size=[H, W], interpolation=InterpolationMode.BILINEAR)

    if placement is None:
        placement = sample_placement(rng, group_rgba, img_size)

    target_w = int(placement["target_w"])
    target_h = int(placement["target_h"])
    x = int(placement["x"])
    y = int(placement["y"])

    group = TF.resize(group_rgba, size=[target_h, target_w], interpolation=InterpolationMode.BILINEAR)

    # Clip placement to canvas bounds (match PIL paste behavior)
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(W, x + target_w)
    y2 = min(H, y + target_h)
    if x2 <= x1 or y2 <= y1:
        return bg  # nothing to paste

    gx1 = x1 - x
    gy1 = y1 - y
    gx2 = gx1 + (x2 - x1)
    gy2 = gy1 + (y2 - y1)
    group_crop = group[:, gy1:gy2, gx1:gx2]
    # Build a full-size RGBA overlay via padding to avoid in-place ops.
    pad_left = int(x1)
    pad_right = int(W - x2)
    pad_top = int(y1)
    pad_bottom = int(H - y2)
    overlay = torch.nn.functional.pad(
        group_crop,
        (pad_left, pad_right, pad_top, pad_bottom),
        mode="constant",
        value=0.0,
    )
    return alpha_composite(bg, overlay)
