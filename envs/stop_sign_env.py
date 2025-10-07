# envs/stop_sign_env.py
import numpy as np
from PIL import Image, ImageFilter
import gymnasium as gym
from gymnasium import spaces
from torchvision import transforms
from typing import Dict, Any, Tuple, List
import io

from detectors.yolo_wrapper import DetectorWrapper
from .random_blobs import draw_randomized_blobs_set

_to_tensor = transforms.ToTensor()  # kept for compatibility if needed elsewhere


def _rand_augment(pil: Image.Image, rng) -> Image.Image:
    """Stronger / broader augmentations for robustness evaluation."""
    from torchvision.transforms.functional import (
        affine, adjust_brightness, adjust_contrast,
        adjust_saturation, adjust_hue
    )
    img = pil

    # Geometric
    angle = float(rng.uniform(-40, 40))
    scale = float(rng.uniform(0.85, 1.15))
    shearx = float(rng.uniform(-12, 12))
    sheary = float(rng.uniform(-12, 12))
    tx = int(rng.uniform(-0.10, 0.10) * img.width)
    ty = int(rng.uniform(-0.10, 0.10) * img.height)
    img = affine(
        img, angle=angle, translate=(tx, ty), scale=scale,
        shear=(shearx, sheary), interpolation=transforms.InterpolationMode.BILINEAR
    )

    # occasional extra affine
    if rng.random() < 0.35:
        angle2 = float(rng.uniform(-8, 8))
        tx2 = int(rng.uniform(-0.04, 0.04) * img.width)
        ty2 = int(rng.uniform(-0.04, 0.04) * img.height)
        scale2 = float(rng.uniform(0.95, 1.05))
        img = affine(
            img, angle=angle2, translate=(tx2, ty2), scale=scale2,
            shear=(rng.uniform(-6, 6), rng.uniform(-6, 6)),
            interpolation=transforms.InterpolationMode.BILINEAR
        )

    # photometric jitter
    img = adjust_brightness(img, 1.0 + float(rng.uniform(-0.25, 0.25)))
    img = adjust_contrast(img, 1.0 + float(rng.uniform(-0.25, 0.25)))
    img = adjust_saturation(img, 1.0 + float(rng.uniform(-0.35, 0.35)))
    img = adjust_hue(img, float(rng.uniform(-0.05, 0.05)))

    # gamma
    if rng.random() < 0.4:
        gamma = float(rng.uniform(0.85, 1.25))
        arr = np.array(img).astype(np.float32) / 255.0
        arr = np.clip(arr ** gamma, 0.0, 1.0)
        img = Image.fromarray((arr * 255).astype(np.uint8))

    # blur
    if rng.random() < 0.5:
        radius = float(rng.uniform(0.0, 2.5))
        if radius > 0.0:
            img = img.filter(ImageFilter.GaussianBlur(radius=radius))
    if rng.random() < 0.12:
        radius_box = int(rng.integers(1, 3))  # cast to Python int
        img = img.filter(ImageFilter.BoxBlur(radius_box))

    # jpeg artifacts
    if rng.random() < 0.4:
        q = int(rng.integers(50, 95))
        bio = io.BytesIO()
        img.save(bio, format="JPEG", quality=q)
        bio.seek(0)
        img = Image.open(bio).convert("RGB")

    # noise
    if rng.random() < 0.35:
        arr = np.array(img).astype(np.float32)
        sigma = float(rng.uniform(0.0, 8.0))
        if sigma > 0:
            noise = rng.normal(0, sigma, size=arr.shape)
            arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
            img = Image.fromarray(arr)

    # occlusion / cutout
    if rng.random() < 0.35:
        from PIL import ImageDraw
        W, H = img.size
        ow = int(rng.uniform(0.03, 0.15) * W)
        oh = int(rng.uniform(0.03, 0.15) * H)
        cx = int(rng.uniform(0.25, 0.75) * W)
        cy = int(rng.uniform(0.25, 0.75) * H)
        x0 = np.clip(cx - ow // 2 + int(rng.integers(-ow // 2, ow // 2)), 0, W - 1)
        y0 = np.clip(cy - oh // 2 + int(rng.integers(-oh // 2, oh // 2)), 0, H - 1)
        draw = ImageDraw.Draw(img)
        color = tuple(int(c) for c in (rng.integers(0, 255), rng.integers(0, 255), rng.integers(0, 255)))
        draw.rectangle([x0, y0, min(W - 1, x0 + ow), min(H - 1, y0 + oh)], fill=color)

    # slight resize+resample
    if rng.random() < 0.3:
        sf = float(rng.uniform(0.9, 1.1))
        neww = max(4, int(img.width * sf))
        newh = max(4, int(img.height * sf))
        img = img.resize((neww, newh), Image.BILINEAR).resize(pil.size, Image.BILINEAR)

    return img


def _mask_from_alpha(pil: Image.Image, edge_margin_pct: float = 0.02) -> Image.Image:
    """
    Strict alpha-only mask; small dilation to ensure thin white border & letters are included.
    Requires the source image to carry transparency (RGBA/LA or 'transparency' info).
    """
    if not (pil.mode in ("RGBA", "LA") or ("transparency" in pil.info)):
        raise ValueError(
            "stop_sign_image must carry an alpha channel (transparent background). "
            "Load it without .convert('RGB') so the env can use alpha."
        )
    rgba = pil.convert("RGBA")
    A = np.array(rgba.split()[-1])
    base = (A > 0).astype(np.uint8) * 255
    mask = Image.fromarray(base, mode="L")

    k = max(3, int(round(min(pil.size) * edge_margin_pct)) | 1)
    mask = mask.filter(ImageFilter.MaxFilter(k))
    if k > 3:
        mask = mask.filter(ImageFilter.MinFilter(k - 2))
    return mask


class StopSignBlobEnv(gym.Env):
    """
    One-step task: agent proposes an opaque blob-set on the sign (no transparency).
    We evaluate YOLO confidence over K random transforms before/after,
    constrain spread/area, and return a shaped, *normalized* reward (delta/base).
    """
    metadata = {"render.modes": ["rgb_array"]}

    def __init__(
        self,
        stop_sign_image,
        yolo_weights: str,
        img_size: Tuple[int, int] = (640, 640),
        K_transforms: int = 10,
        count_max: int = 80,
        # Kept for compatibility with older training scripts; ignored.
        elite_bonus_weight: float = 0.0,
        elite_momentum: float = 0.0,
        objective: str = "adversary",
        w_area: float = 2.0,
        w_count: float = 0.1,
        area_cap: float = 0.25,
        edge_margin_pct: float = 0.02,
        resample_tries: int = 8,
        min_count_penalty: float = 0.15,     # anti-collapse: encourage >1 blob
        min_area_frac: float = 0.01,         # anti-collapse: avoid vanishing area
        min_area_penalty: float = 0.5
    ):
        super().__init__()

        self.img_size = (int(img_size[0]), int(img_size[1]))
        self.count_max = int(count_max)
        self.K = int(K_transforms)

        # keep original (with alpha), then an RGB resized for display/obs
        if isinstance(stop_sign_image, np.ndarray):
            stop_sign_image = Image.fromarray(stop_sign_image)
        self.orig_pil = stop_sign_image  # may be RGBA
        self.base_pil = stop_sign_image.convert("RGB").resize(self.img_size, Image.BILINEAR)

        # Mask strictly from alpha
        self.full_mask = _mask_from_alpha(self.orig_pil, edge_margin_pct=edge_margin_pct).resize(
            self.img_size, Image.NEAREST
        )
        self.allowed_mask = self.full_mask.copy()  # anywhere on the sign

        # YOLO detector
        self.detector = DetectorWrapper(model_path=yolo_weights, target_class="stop sign", device="auto")

        # Action space: 8-D in [-1,1]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(8,), dtype=np.float32)

        # Observation space: uint8 (H,W,C) — works with VecTransposeImage
        H, W = self.base_pil.height, self.base_pil.width
        self.observation_space = spaces.Box(low=0, high=255, shape=(H, W, 3), dtype=np.uint8)

        # RNG and misc
        self.rng = np.random.default_rng()
        self.transforms: List[int] = []

        self.objective = str(objective).lower()
        self.w_area = float(w_area)
        self.w_count = float(w_count)
        self.area_cap = float(area_cap)
        self.edge_margin_pct = float(edge_margin_pct)
        self.resample_tries = int(resample_tries)
        self.min_count_penalty = float(min_count_penalty)
        self.min_area_frac = float(min_area_frac)
        self.min_area_penalty = float(min_area_penalty)

        self._last_composite: Image.Image | None = None
        self._orig_obs = np.array(self.base_pil).astype(np.uint8)
        self._sign_area_pixels = int(np.count_nonzero(np.array(self.full_mask) > 0))

    def _map_action(self, a: np.ndarray) -> Dict[str, Any]:
        a = np.clip(a, -1, 1)
        count = int(1 + ((a[0] + 1) / 2) * (self.count_max - 1))
        size_scale = 0.5 + ((a[1] + 1) / 2) * 1.5

        # Opaque paint, single color
        alpha = 1.0
        def col(x): return int(32 + ((x + 1) / 2) * 192)
        color_mean = (col(a[3]), col(a[4]), col(a[5]))
        color_std = 0.0

        mode_logits = np.array([a[7], 0.3 * a[7] - 0.2, -a[7], 0.0], dtype=float)
        mode = ["noise", "superellipse", "metaballs", "mix"][int(np.argmax(mode_logits))]

        return dict(count=count, size_scale=size_scale, alpha=alpha,
                    color_mean=color_mean, color_std=color_std, mode=mode)

    def _conf_over_transforms(self, pil: Image.Image) -> float:
        confs: List[float] = []
        for seed in self.transforms:
            rng = np.random.default_rng(seed)
            aug = _rand_augment(pil, rng)
            confs.append(self.detector.infer_confidence(aug))
        return float(np.mean(confs)) if confs else 0.0

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.transforms = [int(self.rng.integers(0, 10_000_000)) for _ in range(self.K)]
        self._orig_obs = np.array(self.base_pil).astype(np.uint8)
        self._last_composite = None
        self.allowed_mask = self.full_mask.copy()
        return self._orig_obs, {}

    def _overlay_bbox_frac(self, base_pil: Image.Image, composited_pil: Image.Image) -> float:
        base_arr = np.array(base_pil).astype(np.int16)
        comp_arr = np.array(composited_pil).astype(np.int16)
        diff = np.abs(comp_arr - base_arr).sum(axis=2)
        mask = diff > 0
        if not mask.any():
            return 0.0
        ys, xs = np.where(mask)
        xmin, xmax = int(xs.min()), int(xs.max())
        ymin, ymax = int(ys.min()), int(ys.max())
        bbox_area = (xmax - xmin + 1) * (ymax - ymin + 1)
        return float(bbox_area) / float(max(1, self._sign_area_pixels))

    def reset_quick(self):
        obs, _ = self.reset()
        return obs

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float32)
        params = self._map_action(action)

        # Baseline confidence across transforms
        base_conf = self._conf_over_transforms(self.base_pil)

        # Draw blobs (respect area spread), retry a few times
        composited, meta, last_bbox_frac = None, None, None
        rng = np.random.default_rng()
        for _ in range(self.resample_tries):
            comp_cand, meta_cand = draw_randomized_blobs_set(
                self.base_pil,
                **params,
                rng=rng,
                allowed_mask=self.allowed_mask,
                area_cap=self.area_cap,
                cap_relative_to_mask=True,
                single_color=True
            )
            bbox_frac = self._overlay_bbox_frac(self.base_pil, comp_cand)
            last_bbox_frac = bbox_frac
            composited, meta = comp_cand, meta_cand
            if bbox_frac <= self.area_cap + 1e-9:
                break

        after_conf = self._conf_over_transforms(composited)

        # ---- Reward shaping (normalized) ----
        # raw delta: after - before (lower better for adversary)
        delta = after_conf - base_conf
        denom = max(base_conf, 1e-6)
        delta_norm = delta / denom

        sgn = +1.0 if self.objective == "helper" else -1.0
        goal_term = sgn * delta_norm

        # sparsity terms
        total_area_mask_frac = float(sum((b.get("area_frac") or 0.0) for b in meta)) if meta else 0.0
        count_norm = float(params["count"]) / float(self.count_max)
        shaped = goal_term - self.w_area * total_area_mask_frac - self.w_count * count_norm

        # anti-collapse: penalize too few blobs or too little area
        collapse_pen = 0.0
        if params["count"] <= 1:
            collapse_pen -= self.min_count_penalty
        if total_area_mask_frac < self.min_area_frac:
            collapse_pen -= self.min_area_penalty

        # spread penalty
        spread_penalty = 0.0
        if last_bbox_frac and last_bbox_frac > self.area_cap + 1e-9:
            spread_penalty = -1.0 * float((last_bbox_frac - self.area_cap) / max(1e-6, self.area_cap))

        reward = float(shaped + collapse_pen + spread_penalty)

        self._last_composite = composited

        info = {
            "objective": self.objective,
            "base_conf": float(base_conf),
            "after_conf": float(after_conf),
            "delta_conf": float(delta),
            "delta_conf_norm": float(delta_norm),
            "params": params,
            "overlays": meta,
            "total_area_mask_frac": total_area_mask_frac,
            "bbox_frac_used": float(last_bbox_frac if last_bbox_frac else 0.0),
            "area_cap": float(self.area_cap),
            "collapse_pen": float(collapse_pen),
            "spread_penalty": float(spread_penalty),
            "composited_pil": composited
        }

        # one-step episode
        terminated, truncated = True, False

        # Return uint8 (H, W, C) — safe for VecTransposeImage
        obs = np.array(composited if composited is not None else self.base_pil, dtype=np.uint8)
        return obs, reward, terminated, truncated, info

    def render(self, mode="rgb_array"):
        pil = self._last_composite if self._last_composite is not None else self.base_pil
        return np.array(pil, dtype=np.uint8)
