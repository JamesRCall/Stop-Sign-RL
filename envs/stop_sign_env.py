from __future__ import annotations
from typing import Tuple, Dict, Any, List, Optional
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageChops, ImageDraw
import gymnasium as gym
from gymnasium import spaces

from detectors.yolo_wrapper import DetectorWrapper
from .random_blobs import draw_randomized_blobs_set  # your existing drawer
from utils.uv_paint import UVPaint, VIOLET_GLOW      # define more in utils/uv_paint.py


class StopSignBlobEnv(gym.Env):
    """
    Episodic env that learns blob patterns which keep YOLO confidence high before UV
    (daylight) and low after UV activation, while matching geometry/placement/transforms.

    Per-step pipeline (pair-matched):
      1) Sample a UVPaint pair from a list (policy chooses via action).
      2) Draw blobs on the sign with a single geometry 'pattern_seed'.
      3) Apply mild transforms to the SIGN ONLY (same transform_seed for all variants).
      4) Attach transformed sign to the POLE (now it's one RGBA 'group').
      5) Place that group on the episode's BACKGROUND with one size & position (place_seed).
      6) Run YOLO on the final composites; save/log the exact tested image.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        stop_sign_image: Image.Image,
        stop_sign_uv_image: Optional[Image.Image] = None,
        background_images: Optional[List[Image.Image]] = None,
        pole_image: Optional[Image.Image] = None,
        yolo_weights: str = "yolo11n.pt",
        img_size: Tuple[int, int] = (640, 640),

        # ---- Transforms / episode ----
        steps_per_episode: int = 16,

        # ---- Blob controls ----
        count_max: int = 80,
        area_cap: float = 0.30,

        # ---- UV paints ----
        # You can pass a list like [VIOLET_GLOW, GREEN_GLOW, ...]
        uv_paints: Optional[List[UVPaint]] = None,
        # Fallback if policy doesn't choose: use this (kept for convenience)
        default_uv_paint: UVPaint = VIOLET_GLOW,

        # ---- Reward shaping (Phase B) ----
        eps_day_tolerance: float = 0.03,
        day_floor: float = 0.80,

        # ---- Curriculum controls ----
        attack_only: bool = False,           # Phase A if True
        attack_alpha: float = 1.0,           # blob opacity in Phase A

        seed: Optional[int] = None,
    ):
        super().__init__()
        self.img_size = tuple(img_size)
        self.steps_per_episode = int(steps_per_episode)

        # blobs
        self.count_max = int(count_max)
        self.area_cap = float(area_cap)

        # UV paint list
        self.uv_paints: List[UVPaint] = list(uv_paints) if uv_paints else [default_uv_paint]
        self.default_uv_paint = default_uv_paint

        # reward shaping
        self.eps_day_tolerance = float(eps_day_tolerance)
        self.day_floor = float(day_floor)

        # curriculum
        self.attack_only = bool(attack_only)
        self.attack_alpha = float(attack_alpha)

        # assets
        self.sign_rgba_plain = stop_sign_image.convert("RGBA")
        self.sign_rgba_uv = (stop_sign_uv_image or stop_sign_image).convert("RGBA")
        self.bg_list = [im.convert("RGB") for im in (background_images or [])]
        self.pole_rgba = None if pole_image is None else pole_image.convert("RGBA")

        # detector
        self.det = DetectorWrapper(yolo_weights)

        # spaces
        # action: [0]=count, [1]=size_scale, [2]=color_idx (continuous -> discrete),
        # the rest reserved (so you can add more later)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(8,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(self.img_size[1], self.img_size[0], 3), dtype=np.uint8
        )

        # RNG & episodic state
        self.rng = np.random.default_rng(seed)
        self._step_in_ep = 0
        self._bg_rgb = None  # cached background for this episode

    # ----------------------------- phase control -----------------------------
    def set_phase_attack_only(self, attack_only: bool, attack_alpha: Optional[float] = None):
        self.attack_only = bool(attack_only)
        if attack_alpha is not None:
            self.attack_alpha = float(attack_alpha)

    # ----------------------------- lifecycle ---------------------------------
    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._step_in_ep = 0
        self._bg_rgb = self._choose_bg_rgb()  # background ONLY; no pole here
        # show a plain baseline composite as the initial observation for simplicity
        plain_group = self._compose_sign_and_pole(self.sign_rgba_plain.copy())
        img0, _ = self._place_group_on_background(plain_group, self._bg_rgb, seed=int(self.rng.integers(0, 2**31 - 1)))
        obs = np.array(img0, dtype=np.uint8)
        return obs, {}

    # ----------------------------- step --------------------------------------
    def step(self, action: np.ndarray):
        self._step_in_ep += 1
        params = self._map_action(action)

        # choose uv paint pair from the list based on action's discrete index
        uv_pair = self.uv_paints[params["color_idx"]]

        # seeds to match geometry/transform/placement across variants
        pattern_seed = int(self.rng.integers(0, 2**31 - 1))
        transform_seed = int(self.rng.integers(0, 2**31 - 1))
        place_seed = int(self.rng.integers(0, 2**31 - 1))
        pat_rng = np.random.default_rng(pattern_seed)

        # -------------------- build the three variants --------------------
        # 0) plain (no blobs)
        sign_plain_t = self._apply_sign_only_transform(self.sign_rgba_plain.copy(), transform_seed)
        group_plain = self._compose_sign_and_pole(sign_plain_t)
        img_plain, place_meta = self._place_group_on_background(group_plain, self._bg_rgb, seed=place_seed)

        if self.attack_only:
            # A) attack image (use uv_pair.active_rgb as the color; alpha=self.attack_alpha)
            sign_attack, metas = self._draw_blobs_on_sign_with_pattern(
                self.sign_rgba_plain.copy(),
                color_rgb=uv_pair.active_rgb,
                alpha=self.attack_alpha,
                params=params,
                rng=pat_rng,
                return_metas=True,
            )
            total_area_mask_frac = float(sum(m.get("area_frac", 0.0) for m in metas))
            sign_attack_t = self._apply_sign_only_transform(sign_attack, transform_seed)
            group_attack = self._compose_sign_and_pole(sign_attack_t)
            img_attack, _ = self._place_group_on_background(group_attack, self._bg_rgb, seed=place_seed)

            # score on FINAL images; preview is the attacked final composite
            c0 = float(self._infer_conf(img_plain))
            ca = float(self._infer_conf(img_attack))
            preview = img_attack

            # ------------------ normalized reward ------------------
            # confidences are in [0,1]; normalize the drop by the baseline
            rel_drop = max(0.0, (c0 - ca) / max(c0, 1e-3))  # 0..~1 (higher is better for adversary)

            # area penalties: small linear discouragement + stronger penalty above a soft cap
            cap = float(self.area_cap)             # e.g., 0.20
            over = max(0.0, total_area_mask_frac - cap)
            pen_area_lin  = 0.10 * total_area_mask_frac      # light touch everywhere
            pen_area_over = 0.80 * (over ** 2)               # quadratic over-cap

            # count penalty to avoid degenerate "many tiny blobs" solutions
            cnt_norm = float(params["count"]) / max(1.0, float(self.count_max))
            pen_count = 0.10 * cnt_norm

            # combine and squash to [-1, 1] for stable PPO
            raw = (1.00 * rel_drop) - (pen_area_lin + pen_area_over + pen_count)
            import math
            reward = math.tanh(2.0 * raw)  # gain=2 tightens into [-1,1] but keeps signal

            obs = np.array(img_attack, dtype=np.uint8)  # agent "sees" what it made
            terminated = False
            truncated = (self._step_in_ep >= self.steps_per_episode)
            info = {
                "objective": "attack_only",
                "base_conf": c0,
                "after_conf": ca,
                "delta_conf": ca - c0,
                "total_area_mask_frac": total_area_mask_frac,
                "area_cap": cap,
                "params": {
                    "count": int(params["count"]),
                    "size_scale": float(params["size_scale"]),
                    "alpha": float(self.attack_alpha),
                    "color_idx": int(params["color_idx"]),
                },
                # trace for exact reproduction
                "trace": {
                    "phase": "A",
                    "pattern_seed": pattern_seed,
                    "transform_seed": transform_seed,
                    "place_seed": place_seed,
                    "count": int(params["count"]),
                    "size_scale": float(params["size_scale"]),
                    "alpha": float(self.attack_alpha),
                    "color_idx": int(params["color_idx"]),
                    "place": place_meta,
                },
                # diagnostics for TB
                "reward_raw": float(raw),
                "rel_drop": float(rel_drop),
                "pen_area_lin": float(pen_area_lin),
                "pen_area_over": float(pen_area_over),
                "pen_count": float(pen_count),
                "conf_plain": c0,
                "conf_uv_day": ca,
                "conf_uv_on": ca,
                "gap": float(c0 - ca),
                "reward": float(reward),
                # the exact tested image
                "composited_pil": preview,
            }
            return obs, reward, terminated, truncated, info


        # -------------------- Phase B (UV-day & UV-on) --------------------
        # 1) UV-day
        sign_day, metas_day = self._draw_blobs_on_sign_with_pattern(
            self.sign_rgba_plain.copy(),
            color_rgb=uv_pair.day_rgb,
            alpha=(uv_pair.day_alpha if uv_pair.translucent else 1.0),
            params=params,
            rng=pat_rng,
            return_metas=True,
        )
        total_area_mask_frac = float(sum(m.get("area_frac", 0.0) for m in metas_day))
        sign_day_t = self._apply_sign_only_transform(sign_day, transform_seed)
        group_day  = self._compose_sign_and_pole(sign_day_t)
        img_day, _ = self._place_group_on_background(group_day, self._bg_rgb, seed=place_seed)

        # 2) UV-on  (same geometry: reset RNG)
        pat_rng = np.random.default_rng(pattern_seed)
        sign_on = self._draw_blobs_on_sign_with_pattern(
            self.sign_rgba_uv.copy(),
            color_rgb=uv_pair.active_rgb,
            alpha=(uv_pair.active_alpha if uv_pair.translucent else 1.0),
            params=params,
            rng=pat_rng,
        )
        sign_on_t = self._apply_sign_only_transform(sign_on, transform_seed)
        group_on   = self._compose_sign_and_pole(sign_on_t)
        img_on, _ = self._place_group_on_background(group_on, self._bg_rgb, seed=place_seed)

        # test on FINAL images; preview the UV-on final composite
        c0 = self._infer_conf(img_plain)
        c_day = self._infer_conf(img_day)
        c_on = self._infer_conf(img_on)
        preview = img_on

        # reward: maximize day - on; keep day close to baseline and above a floor
        eps = self.eps_day_tolerance
        floor = self.day_floor
        objective = c_day - c_on
        penalty_day = max(0.0, (c0 - c_day) - eps)
        bonus_dayfloor = max(0.0, c_day - floor)
        reward = 1.0 * objective - 2.0 * penalty_day + 0.25 * bonus_dayfloor

        obs = np.array(img_day, dtype=np.uint8)  # agent observes the 'should-stay-high' variant
        terminated = False
        truncated = (self._step_in_ep >= self.steps_per_episode)
        info = {
            "objective": "adversary",
            "base_conf": float(c0),
            "after_conf": float(c_on),
            "delta_conf": float(c_on - c0),
            "total_area_mask_frac": total_area_mask_frac,
            "area_cap": float(self.area_cap),
            "params": {
                "count": int(params["count"]),
                "size_scale": float(params["size_scale"]),
                "alpha_day": float(uv_pair.day_alpha if uv_pair.translucent else 1.0),
                "alpha_on": float(uv_pair.active_alpha if uv_pair.translucent else 1.0),
                "color_idx": int(params["color_idx"]),
            },
            "trace": {
                "phase": "B",
                "pattern_seed": pattern_seed,
                "transform_seed": transform_seed,
                "place_seed": place_seed,
                "count": int(params["count"]),
                "size_scale": float(params["size_scale"]),
                "alpha_day": float(uv_pair.day_alpha if uv_pair.translucent else 1.0),
                "alpha_on": float(uv_pair.active_alpha if uv_pair.translucent else 1.0),
                "color_idx": int(params["color_idx"]),
                "place": place_meta,
            },
            "conf_plain": float(c0),
            "conf_uv_day": float(c_day),
            "conf_uv_on": float(c_on),
            "gap": float(objective),
            "reward": float(reward),
            # the exact tested image
            "composited_pil": preview,
        }
        return obs, reward, terminated, truncated, info

    # ============================ helpers ====================================

    def _infer_conf(self, pil_rgb: Image.Image) -> float:
        try:
            return float(self.det.infer_confidence(pil_rgb))
        except Exception:
            return 0.0

    def _map_action(self, a: np.ndarray) -> Dict[str, Any]:
        a = np.clip(a, -1, 1).astype(np.float32)
        count = int(1 + ((a[0] + 1) * 0.5) * (self.count_max - 1))
        size_scale = 0.5 + ((a[1] + 1) * 0.5) * 1.5
        # choose a paint index from the list using a[2]
        n_colors = max(1, len(self.uv_paints))
        idx = int(np.floor(((a[2] + 1) * 0.5) * n_colors))
        idx = min(max(idx, 0), n_colors - 1)
        return {"count": count, "size_scale": float(size_scale), "color_idx": idx, "mode": "superellipse"}

    # ------------------------ draw / compose primitives ----------------------

    def _draw_blobs_on_sign_with_pattern(
        self,
        sign_rgba: Image.Image,
        color_rgb: Tuple[int, int, int],
        alpha: float,
        params: Dict[str, Any],
        rng: np.random.Generator,
        return_metas: bool = False,
    ):
        """
        Draw a single-color blob set onto the sign bitmap using the GIVEN rng
        (so geometry repeats across day/on). Uses your random_blobs API.
        """
        sign_mask_local = sign_rgba.split()[-1]
        sign_rgb = sign_rgba.convert("RGB")
        comp_rgb, metas = draw_randomized_blobs_set(
            base_pil=sign_rgb,
            count=params["count"],
            size_scale=params["size_scale"],
            alpha=float(np.clip(alpha, 0.0, 1.0)),
            color_mean=tuple(int(v) for v in color_rgb),  # exact color via mean
            color_std=0.0,
            mode=params["mode"],
            rng=rng,
            allowed_mask=sign_mask_local,
            area_cap=self.area_cap,
            cap_relative_to_mask=True,
            single_color=True,
        )
        out = Image.merge("RGBA", (*comp_rgb.split(), sign_mask_local))
        return (out, metas) if return_metas else out

    def _apply_sign_only_transform(self, sign_rgba: Image.Image, seed: int) -> Image.Image:
        """
        Mild transforms on the SIGN ONLY (RGBA). Keeps alpha intact.
        """
        rng = np.random.default_rng(seed)
        W, H = sign_rgba.size
        out = sign_rgba.copy()

        # mild affine
        angle = rng.uniform(-6, 6)
        shear = rng.uniform(-4, 4)
        scale = 1.0 + rng.uniform(-0.10, 0.10)
        tx = rng.uniform(-0.02 * W, 0.02 * W)
        ty = rng.uniform(-0.02 * H, 0.02 * H)

        aff = _affine_matrix(angle, shear, scale, tx, ty, W, H)
        out = out.transform((W, H), Image.AFFINE, data=aff, resample=Image.BILINEAR, fillcolor=(0, 0, 0, 0))

        # optional very light perspective
        if rng.random() < 0.5:
            coeffs = _random_perspective_coeffs(W, H, rng, max_shift=0.06)
            out = out.transform((W, H), Image.PERSPECTIVE, coeffs, resample=Image.BILINEAR, fillcolor=(0, 0, 0, 0))

        # photometric (very mild) on RGB only
        rgb, a = out.convert("RGB"), out.split()[-1]
        if rng.random() < 0.7:
            rgb = ImageEnhance.Brightness(rgb).enhance(float(rng.uniform(0.9, 1.1)))
        if rng.random() < 0.7:
            rgb = ImageEnhance.Contrast(rgb).enhance(float(rng.uniform(0.9, 1.1)))
        if rng.random() < 0.3:
            rgb = ImageEnhance.Color(rgb).enhance(float(rng.uniform(0.9, 1.1)))
        if rng.random() < 0.4:
            rgb = rgb.filter(ImageFilter.GaussianBlur(radius=float(rng.uniform(0.0, 0.8))))
        if rng.random() < 0.6:
            arr = np.array(rgb, dtype=np.int16)
            sigma = rng.uniform(1.0, 3.0)
            noise = rng.normal(0.0, sigma, size=arr.shape)
            arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
            rgb = Image.fromarray(arr, mode="RGB")

        return Image.merge("RGBA", (*rgb.split(), a))


    def _place_group_on_background(self, group_rgba: Image.Image, bg_rgb: Image.Image, seed: int):
        """
        Randomly scale & place the (sign+pole) group onto the chosen background.
        Returns (final RGB composite, placement_meta).
        """
        rng = np.random.default_rng(seed)
        W, H = self.img_size
        bg_rgba = bg_rgb.resize((W, H), Image.BILINEAR).convert("RGBA")

        # overall scale relative to canvas width
        target_w = int(rng.uniform(0.30 * W, 0.55 * W))
        scale = target_w / max(1, group_rgba.width)
        group = group_rgba.resize((target_w, int(group_rgba.height * scale)), Image.BILINEAR)

        # placement with margins
        margin = int(0.04 * W)
        max_x = max(margin, W - group.width - margin)
        max_y = max(margin, H - group.height - margin)
        x = int(rng.integers(margin, max_x + 1))
        y = int(rng.integers(margin, max_y + 1))

        canvas = bg_rgba.copy()
        canvas.alpha_composite(group, (x, y))
        return canvas.convert("RGB"), {"x": x, "y": y, "scale": scale}

    def _choose_bg_rgb(self) -> Image.Image:
        W, H = self.img_size
        if self.bg_list:
            idx = int(self.rng.integers(0, len(self.bg_list)))
            return self.bg_list[idx].resize((W, H), Image.BILINEAR).convert("RGB")
        return Image.new("RGB", (W, H), (200, 200, 200))
    
    def _compose_sign_and_pole(
        self,
        sign_rgba: Image.Image,
        pole_width_ratio: float = 0.12,   # ~12% of sign width (realistic)
        bottom_len_factor: float = 4.0,   # show ~4× sign-height of pole below
        clearance_px: int = 2,            # tiny gap between pole top and sign
        side_margin_frac: float = 0.06,   # side margins around the sign in the group
    ) -> Image.Image:
        """
        Build a (sign+pole) group with realistic proportions:
        - Resize pole to pole_width_ratio * sign_width
        - Crop (or stretch slightly) so there is ~bottom_len_factor * sign_height
            of pole visible below the sign
        - Place sign at the very top of the visible pole with a tiny clearance
        - Center sign over pole; add small side margins
        """
        if self.pole_rgba is None:
            return sign_rgba

        sign = sign_rgba.copy()
        SW, SH = sign.size
        pole = self.pole_rgba.copy().convert("RGBA")
        PW0, PH0 = pole.size

        # 1) Width-normalize pole to a fraction of sign width
        target_pw = max(2, int(pole_width_ratio * SW))
        scale_w = target_pw / max(1, PW0)
        target_ph = max(1, int(PH0 * scale_w))
        pole = pole.resize((target_pw, target_ph), Image.BILINEAR)

        # 2) Ensure we have enough pole length visible below the sign; crop to needed height
        H_needed = clearance_px + SH + int(bottom_len_factor * SH)
        if pole.height < H_needed:
            # modest stretch to meet minimum; looks fine for a cylindrical pole
            scale_h = H_needed / pole.height
            pole = pole.resize((pole.width, int(pole.height * scale_h)), Image.BILINEAR)

        # Crop to exactly what we need from the top of the pole
        pole = pole.crop((0, 0, pole.width, H_needed))

        # 3) Build a group canvas with a tiny side margin around the sign
        side_margin = int(side_margin_frac * SW)
        GW = max(pole.width, SW + 2 * side_margin)
        GH = H_needed
        group = Image.new("RGBA", (GW, GH), (0, 0, 0, 0))

        # 4) Paste pole centered, then sign at top with tiny clearance
        px = (GW - pole.width) // 2
        group.alpha_composite(pole, (px, 0))

        sx = (GW - SW) // 2
        sy = clearance_px
        group.alpha_composite(sign, (sx, sy))
        return group


# ---------------------------- helpers (module level) ----------------------------

def _affine_matrix(angle_deg, shear_deg, scale, tx, ty, W, H):
    import math
    angle = math.radians(angle_deg)
    shear = math.radians(shear_deg)
    cos_a, sin_a = math.cos(angle) * scale, math.sin(angle) * scale
    a = cos_a + (-sin_a) * math.tan(shear)
    b = sin_a + cos_a * math.tan(shear)
    c = tx
    d = -sin_a + cos_a * math.tan(shear)
    e = cos_a + sin_a * math.tan(shear)
    f = ty
    cx, cy = W / 2.0, H / 2.0
    c += cx - (a * cx + b * cy)
    f += cy - (d * cx + e * cy)
    return (a, b, c, d, e, f)

def _random_perspective_coeffs(W, H, rng, max_shift=0.06):
    dx, dy = W * max_shift, H * max_shift
    src = [(0, 0), (W, 0), (W, H), (0, H)]
    dst = [
        (rng.uniform(-dx, dx), rng.uniform(-dy, dy)),
        (W + rng.uniform(-dx, dx), rng.uniform(-dy, dy)),
        (W + rng.uniform(-dx, dx), H + rng.uniform(-dy, dy)),
        (rng.uniform(-dx, dx), H + rng.uniform(-dy, dy)),
    ]
    import numpy as np
    A = []
    for (x, y), (u, v) in zip(src, dst):
        A.extend([[x, y, 1, 0, 0, 0, -u * x, -u * y],
                  [0, 0, 0, x, y, 1, -v * x, -v * y]])
    A = np.array(A, dtype=np.float32)
    B = np.array([p for uv in dst for p in uv], dtype=np.float32)
    coeffs = np.linalg.lstsq(A, B, rcond=None)[0]
    return coeffs

