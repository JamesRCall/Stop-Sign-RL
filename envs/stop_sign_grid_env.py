from __future__ import annotations
from typing import Tuple, Dict, Any, List, Optional
import math
import numpy as np
from collections import deque
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter

import gymnasium as gym
from gymnasium import spaces

from detectors.yolo_wrapper import DetectorWrapper
from utils.uv_paint import UVPaint, VIOLET_GLOW  # you can swap the paint in train file


class StopSignGridEnv(gym.Env):
    """
    Grid-square adversarial overlay over a stop sign (octagon mask only).

    Action (continuous -> discrete):
      a[0], a[1] in [-1, 1]  ->  (row, col) index into sign grid
      (cell size = 4x4 px by default; configurable via grid_cell_px)

    Per step:
      • Add 1 new grid cell (2x2 / 4x4 square) to a running episode mask.
      • Duplicate cells are disallowed: we remap to nearest free cell.
      • Render THREE matched variants on the SAME background/pole/placement/transforms:
          0) Plain (no overlay) – baseline for day and UV-on
          1) Daylight overlay (pre-activation color/alpha)
          2) UV-on overlay (activated color/alpha)
      • For robustness, evaluate each variant across K matched SIGN-only transforms,
        same placement and background for all three.
      • Compute mean confidences over K runs:
          c0_day, c_day, c0_on, c_on
        Primary objective is drop_on = c0_on - c_on (we want ≥ threshold).
        Secondary: keep day confidence high (penalize drop if it exceeds tolerance).
      • Episode terminates early when drop_on_mean ≥ uv_drop_threshold.

    Observation:
      RGB image of the *daylight* composite for the first transform (H,W,3) uint8.

    Reward (per step, normalized):
      Let raw_core = drop_on - lambda_day * max(0, drop_day - day_tolerance).
      Add a smooth shaping bonus as drop_on approaches threshold, plus a small
      success bonus once drop_on >= uv_drop_threshold, then squash:

          raw_total = raw_core + shaping + success_bonus
          reward    = tanh(2 * raw_total)    ∈ (-1, 1)

      so PPO always sees a bounded per-step reward.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        stop_sign_image: Image.Image,
        stop_sign_uv_image: Optional[Image.Image],
        background_images: List[Image.Image],
        pole_image: Optional[Image.Image],
        yolo_weights: str = "weights/yolo11n.pt",
        img_size: Tuple[int, int] = (640, 640),
        detector_debug: bool = False,


        # Episodes
        steps_per_episode: int = 7000,
        eval_K: int = 3,

        # Grid config
        grid_cell_px: int = 4,               # 🔴 default 4x4 now
        max_cells: Optional[int] = None,

        # Paint (single pair)
        uv_paint: UVPaint = VIOLET_GLOW,
        use_single_color: bool = True,

        # Threshold logic
        uv_drop_threshold: float = 0.70,
        day_tolerance: float = 0.05,
        lambda_day: float = 1.0,
        min_base_conf: float = 0.20,
        cell_cover_thresh: float = 0.6,


        # YOLO
        yolo_device: str = "cpu",
        conf_thresh: float = 0.10,
        iou_thresh: float = 0.45,
        info_image_every: int = 50,

        # Reward smoothing (Option A)
        reward_smooth_n: int = 10,
        use_ema: bool = True,
        ema_beta: float = 0.90,

        seed: Optional[int] = None,
    ):
        super().__init__()
        self.img_size = tuple(img_size)
        self.steps_per_episode = int(steps_per_episode)
        self.eval_K = int(eval_K)

        self.sign_rgba_day = stop_sign_image.convert("RGBA")
        self.sign_rgba_on  = (stop_sign_uv_image or stop_sign_image).convert("RGBA")
        self.bg_list = [im.convert("RGB") for im in (background_images or [])]
        self.pole_rgba = None if pole_image is None else pole_image.convert("RGBA")

        self.grid_cell_px = int(grid_cell_px)
        if self.grid_cell_px not in (2, 4):
            raise ValueError("grid_cell_px must be 2 or 4")

        # build sign alpha and grid on construction
        self._sign_alpha = self.sign_rgba_day.split()[-1]  # L mask of octagon
        self._build_grid_index()

        self.max_cells = int(max_cells) if max_cells is not None else None

        # UV paint pair (single)
        self.paint = uv_paint
        self.use_single_color = bool(use_single_color)

        # threshold / reward
        self.uv_drop_threshold = float(uv_drop_threshold)
        self.day_tolerance = float(day_tolerance)
        self.lambda_day = float(lambda_day)
        self.min_base_conf = float(min_base_conf)
        self.info_image_every = int(info_image_every)
        self.cell_cover_thresh = float(cell_cover_thresh)


        # Reward smoothing state (Option A)
        self.reward_smooth_n = int(reward_smooth_n)
        self.use_ema = bool(use_ema)
        self.ema_beta = float(ema_beta)
        self._drop_hist = deque(maxlen=self.reward_smooth_n)
        self._drop_ema = None

        # detector
        self.det = DetectorWrapper(
            yolo_weights, device=yolo_device, conf=conf_thresh, iou=iou_thresh, debug=detector_debug
        )


        # action/obs spaces
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        H, W = self.img_size[1], self.img_size[0]
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(H, W, 3), dtype=np.uint8
        )

        # RNG & episodic state
        self.rng = np.random.default_rng(seed)
        self._step = 0
        self._bg_rgb = None
        self._episode_cells: np.ndarray = None  # bool mask [Gh, Gw] of selected cells
        self._place_seed = None
        self._transform_seeds: List[int] = []

    # ----------------------------- grid build --------------------------------

    def _build_grid_index(self):
        """Precompute grid geometry restricted to the sign alpha (octagon)."""
        W, H = self.sign_rgba_day.size
        g = self.grid_cell_px
        Gw, Gh = math.ceil(W / g), math.ceil(H / g)

        signA = np.array(self._sign_alpha, dtype=np.uint8) > 0
        valid = np.zeros((Gh, Gw), dtype=bool)
        rects: List[Tuple[int, int, int, int]] = []

        for r in range(Gh):
            for c in range(Gw):
                x0, y0 = c * g, r * g
                x1, y1 = min(W, x0 + g), min(H, y0 + g)
                cell = signA[y0:y1, x0:x1]
                cover = float(cell.mean()) if cell.size else 0.0
                valid[r, c] = (cover >= self.cell_cover_thresh)

                rects.append((x0, y0, x1, y1))

        self.Gw, self.Gh = Gw, Gh
        self._cell_rects = rects
        self._valid_cells = valid

    # ----------------------------- lifecycle ---------------------------------

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self._step = 0
        self._episode_cells = np.zeros((self.Gh, self.Gw), dtype=bool)

        self._bg_rgb = self._choose_bg_rgb()
        self._place_seed = int(self.rng.integers(0, 2**31 - 1))
        self._transform_seeds = [int(self.rng.integers(0, 2**31 - 1)) for _ in range(self.eval_K)]

        obs = self._render_variant(kind="day", use_overlay=False, transform_seed=self._transform_seeds[0])

        # Cache plain baseline confidences (same throughout episode)
        c0_day, c0_on = self._eval_plain_over_K()
        self._baseline_c0_day = c0_day
        self._baseline_c0_on  = c0_on

        # Reset reward smoothing buffers (Option A)
        self._drop_hist.clear()
        self._drop_ema = None

        return np.array(obs, dtype=np.uint8), {}

    # ----------------------------- step --------------------------------------

    def step(self, action: np.ndarray):
        self._step += 1

        # 1) map action → cell (r,c); enforce non-duplicate; remap if needed
        r, c = self._map_to_cell(action)
        if not self._valid_cells[r, c]:
            r, c = self._nearest_valid_free_cell(r, c)
        else:
            if self._episode_cells[r, c]:
                r, c = self._nearest_valid_free_cell(r, c)

        self._episode_cells[r, c] = True

        terminated = False
        if self.max_cells is not None:
            total_on = int(self._episode_cells.sum())
            if total_on >= self.max_cells:
                terminated = True

        # 2) Evaluate K matched transforms: plain vs overlay, both day and UV-on
        c_day, c_on = self._eval_overlay_over_K()
        c0_day, c0_on = self._baseline_c0_day, self._baseline_c0_on

        drop_day = float(c0_day - c_day)
        drop_on  = float(c0_on - c_on)

        # (Option A) Smooth UV drop signal over recent steps to reduce volatility
        self._drop_hist.append(drop_on)
        if self.use_ema:
            if self._drop_ema is None:
                self._drop_ema = drop_on
            else:
                b = float(self.ema_beta)
                self._drop_ema = b * float(self._drop_ema) + (1.0 - b) * drop_on
            drop_on_s = float(self._drop_ema)
        else:
            drop_on_s = float(np.mean(self._drop_hist)) if len(self._drop_hist) else float(drop_on)

        # (5) Baseline gating: if the detector didn't see the sign in baseline,
        # don't let the agent "win" on junk scenes.
        if c0_on < self.min_base_conf:
            # Small penalty to discourage these states
            reward = -0.05
            terminated = False
            truncated = (self._step >= self.steps_per_episode)

            obs_img = self._render_variant(kind="day", use_overlay=True, transform_seed=self._transform_seeds[0])
            obs = np.array(obs_img, dtype=np.uint8)

            info = {
                "objective": "grid_uv",
                "c0_day": c0_day, "c_day": c_day,
                "c0_on": c0_on,   "c_on": c_on,
                "drop_day": drop_day, "drop_on": drop_on, "drop_on_smooth": float(drop_on_s),
                "note": "baseline_conf_too_low",
                "min_base_conf": float(self.min_base_conf),
            }
            return obs, float(reward), bool(terminated), bool(truncated), info


        # 3) reward (normalized) & early stopping on UV threshold
        # core term: big UV drop good, day drop beyond tolerance bad
        pen_day = max(0.0, drop_day - self.day_tolerance)
        raw_core = drop_on_s - self.lambda_day * pen_day

        # smooth shaping as we move toward threshold
        thr = self.uv_drop_threshold
        shaping = 0.25 * math.tanh(4.0 * (drop_on_s - 0.5 * thr))

        # success bonus once we’ve actually crossed the threshold
        success_bonus = 0.0
        if drop_on_s >= thr:
            success_bonus = 0.5

        # total “pre-normalized” reward, then squash to ~[-1, 1]
        raw_total = raw_core + shaping + success_bonus
        reward = math.tanh(2.0 * raw_total)

        if drop_on_s >= thr:
            terminated = True

        truncated = (self._step >= self.steps_per_episode)

        # 4) observation: daylight composite for transform #0
        obs_img = self._render_variant(kind="day", use_overlay=True, transform_seed=self._transform_seeds[0])
        obs = np.array(obs_img, dtype=np.uint8)

        # preview image: UV-on, transform #0
        preview_on = self._render_variant(kind="on", use_overlay=True, transform_seed=self._transform_seeds[0])

        info = {
            "objective": "grid_uv",
            "c0_day": c0_day, "c_day": c_day,
            "c0_on": c0_on,   "c_on": c_on,
            "drop_day": drop_day, "drop_on": drop_on, "drop_on_smooth": float(drop_on_s),
            "reward_core": float(raw_core),
            "reward_raw_total": float(raw_total),
            "reward": float(reward),
            "selected_cells": int(self._episode_cells.sum()),
            "grid_cell_px": int(self.grid_cell_px),
            "uv_drop_threshold": float(self.uv_drop_threshold),
            "day_tolerance": float(self.day_tolerance),
            "base_conf": float(c0_on),
            "after_conf": float(c_on),
            "total_area_mask_frac": self._area_frac_selected(),
            "params": {
                "count": int(self._episode_cells.sum()),
                "size_scale": float(self.grid_cell_px),
                "alpha": float(self.paint.active_alpha),
            },
            "trace": {
                "phase": "grid_uv",
                "grid_cell_px": int(self.grid_cell_px),
                "selected_indices": self._selected_indices_list(),
                "place_seed": int(self._place_seed),
                "transform_seeds": [int(s) for s in self._transform_seeds],
            },
        }

        # (6) Only include the PIL preview occasionally to reduce overhead
        if (self._step % self.info_image_every) == 0:
            info["composited_pil"] = preview_on

        return obs, float(reward), bool(terminated), bool(truncated), info

    # ----------------------------- helpers -----------------------------------

    def _eval_plain_over_K(self) -> Tuple[float, float]:
        imgs_plain_day, imgs_plain_on = [], []
        for t_seed in self._transform_seeds:
            plain_day = self._compose_on_bg(self._transform_sign(self.sign_rgba_day, t_seed), self._place_seed)
            plain_on  = self._compose_on_bg(self._transform_sign(self.sign_rgba_on,  t_seed), self._place_seed)
            imgs_plain_day.append(plain_day)
            imgs_plain_on.append(plain_on)
        c0_day_list = self.det.infer_confidence_batch(imgs_plain_day)
        c0_on_list  = self.det.infer_confidence_batch(imgs_plain_on)
        mean = lambda xs: float(np.mean(xs)) if len(xs) else 0.0
        return mean(c0_day_list), mean(c0_on_list)


    def _selected_indices_list(self) -> List[int]:
        idxs = np.flatnonzero(self._episode_cells.reshape(-1)).tolist()
        return idxs

    def _area_frac_selected(self) -> float:
        valid_total = int(self._valid_cells.sum())
        if valid_total == 0:
            return 0.0
        return float(int(self._episode_cells.sum())) / float(valid_total)

    def _map_to_cell(self, a: np.ndarray) -> Tuple[int, int]:
        a = np.clip(np.asarray(a, dtype=np.float32), -1.0, 1.0)
        r = int(((a[0] + 1.0) * 0.5) * self.Gh)
        c = int(((a[1] + 1.0) * 0.5) * self.Gw)
        r = max(0, min(self.Gh - 1, r))
        c = max(0, min(self.Gw - 1, c))
        return r, c

    def _nearest_valid_free_cell(self, r: int, c: int) -> Tuple[int, int]:
        free_mask = self._valid_cells & (~self._episode_cells)
        coords = np.argwhere(free_mask)  # (N, 2) array of (rr, cc)

        if coords.size == 0:
            return r, c

        dr = coords[:, 0] - r
        dc = coords[:, 1] - c
        d2 = dr * dr + dc * dc
        m = d2.min()

        # random tie-break among equally near cells
        cand = coords[d2 == m]
        i = int(self.rng.integers(0, cand.shape[0]))
        return int(cand[i, 0]), int(cand[i, 1])


    def _is_valid_free(self, r: int, c: int) -> bool:
        return (
            0 <= r < self.Gh
            and 0 <= c < self.Gw
            and self._valid_cells[r, c]
            and (not self._episode_cells[r, c])
        )

    def _render_variant(self, kind: str, use_overlay: bool, transform_seed: int) -> Image.Image:
        assert kind in ("day", "on")
        sign_src = self.sign_rgba_day if kind == "day" else self.sign_rgba_on
        sign = sign_src if not use_overlay else self._apply_grid_overlay(
            sign_src, mode=("day" if kind == "day" else "on")
        )
        sign_t = self._transform_sign(sign, transform_seed)
        img = self._compose_on_bg(sign_t, self._place_seed)
        return img

    def _eval_over_K(self) -> Tuple[float, float, float, float]:
        imgs_plain_day = []
        imgs_plain_on  = []
        imgs_over_day  = []
        imgs_over_on   = []

        # Precompute overlays once per step (huge win)
        over_sign_day = self._apply_grid_overlay(self.sign_rgba_day, mode="day")
        over_sign_on  = self._apply_grid_overlay(self.sign_rgba_on,  mode="on")

        for t_seed in self._transform_seeds:
            plain_day = self._compose_on_bg(self._transform_sign(self.sign_rgba_day, t_seed), self._place_seed)
            plain_on  = self._compose_on_bg(self._transform_sign(self.sign_rgba_on,  t_seed), self._place_seed)

            over_day  = self._compose_on_bg(self._transform_sign(over_sign_day, t_seed), self._place_seed)
            over_on   = self._compose_on_bg(self._transform_sign(over_sign_on,  t_seed), self._place_seed)

            imgs_plain_day.append(plain_day)
            imgs_plain_on.append(plain_on)
            imgs_over_day.append(over_day)
            imgs_over_on.append(over_on)

        # Batch inference: 4 calls total instead of 4*K
        c0_day_list = self.det.infer_confidence_batch(imgs_plain_day)
        c0_on_list  = self.det.infer_confidence_batch(imgs_plain_on)
        c_day_list  = self.det.infer_confidence_batch(imgs_over_day)
        c_on_list   = self.det.infer_confidence_batch(imgs_over_on)

        mean = lambda xs: float(np.mean(xs)) if len(xs) else 0.0
        return mean(c0_day_list), mean(c_day_list), mean(c0_on_list), mean(c_on_list)

    def _eval_overlay_over_K(self) -> Tuple[float, float]:
        imgs_over_day, imgs_over_on = [], []

        # Precompute overlays once per step
        over_sign_day = self._apply_grid_overlay(self.sign_rgba_day, mode="day")
        over_sign_on  = self._apply_grid_overlay(self.sign_rgba_on,  mode="on")

        for t_seed in self._transform_seeds:
            over_day = self._compose_on_bg(self._transform_sign(over_sign_day, t_seed), self._place_seed)
            over_on  = self._compose_on_bg(self._transform_sign(over_sign_on,  t_seed), self._place_seed)
            imgs_over_day.append(over_day)
            imgs_over_on.append(over_on)

        c_day_list = self.det.infer_confidence_batch(imgs_over_day)
        c_on_list  = self.det.infer_confidence_batch(imgs_over_on)

        mean = lambda xs: float(np.mean(xs)) if len(xs) else 0.0
        return mean(c_day_list), mean(c_on_list)


    def _apply_grid_overlay(self, sign_rgba: Image.Image, mode: str) -> Image.Image:
        rgb = sign_rgba.convert("RGB")
        a   = sign_rgba.split()[-1]

        draw = ImageDraw.Draw(rgb)
        if mode == "day":
            color = self.paint.day_rgb
            alpha = self.paint.day_alpha if self.paint.translucent else 1.0
        else:
            color = self.paint.active_rgb
            alpha = self.paint.active_alpha if self.paint.translucent else 1.0

        mask = Image.new("L", rgb.size, 0)
        mdraw = ImageDraw.Draw(mask)
        on = np.argwhere(self._episode_cells)
        for r, c in on:
            x0, y0, x1, y1 = self._cell_rects[r * self.Gw + c]
            mdraw.rectangle([x0, y0, x1 - 1, y1 - 1], fill=255)


        mask = Image.composite(mask, Image.new("L", mask.size, 0), self._sign_alpha)

        if alpha < 1.0:
            arr = (np.array(mask, dtype=np.float32) * float(alpha)).astype(np.uint8)
            mask = Image.fromarray(arr, mode="L")

        rgb.paste(color, mask=mask)
        return Image.merge("RGBA", (*rgb.split(), a))

    def _transform_sign(self, sign_rgba: Image.Image, seed: int) -> Image.Image:
        rng = np.random.default_rng(seed)
        W, H = sign_rgba.size
        out = sign_rgba.copy()

        angle = rng.uniform(-6, 6)
        shear = rng.uniform(-4, 4)
        scale = 1.0 + rng.uniform(-0.10, 0.10)
        tx = rng.uniform(-0.02 * W, 0.02 * W)
        ty = rng.uniform(-0.02 * H, 0.02 * H)

        aff = _affine_matrix(angle, shear, scale, tx, ty, W, H)
        out = out.transform((W, H), Image.AFFINE, data=aff, resample=Image.BILINEAR, fillcolor=(0, 0, 0, 0))

        if rng.random() < 0.5:
            coeffs = _random_perspective_coeffs(W, H, rng, max_shift=0.06)
            out = out.transform((W, H), Image.PERSPECTIVE, coeffs, resample=Image.BILINEAR, fillcolor=(0, 0, 0, 0))

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

    def _compose_on_bg(self, sign_rgba_t: Image.Image, place_seed: int) -> Image.Image:
        group = self._compose_sign_and_pole(sign_rgba_t)
        img, _ = self._place_group_on_background(group, self._bg_rgb, seed=place_seed)
        return img

    def _compose_sign_and_pole(
        self,
        sign_rgba: Image.Image,
        pole_width_ratio: float = 0.12,
        bottom_len_factor: float = 4.0,
        clearance_px: int = 2,
        side_margin_frac: float = 0.06,
    ) -> Image.Image:
        if self.pole_rgba is None:
            return sign_rgba

        sign = sign_rgba.copy()
        SW, SH = sign.size
        pole = self.pole_rgba.copy().convert("RGBA")
        PW0, PH0 = pole.size

        target_pw = max(2, int(pole_width_ratio * SW))
        scale_w = target_pw / max(1, PW0)
        target_ph = max(1, int(PH0 * scale_w))
        pole = pole.resize((target_pw, target_ph), Image.BILINEAR)

        H_needed = clearance_px + SH + int(bottom_len_factor * SH)
        if pole.height < H_needed:
            scale_h = H_needed / pole.height
            pole = pole.resize((pole.width, int(pole.height * scale_h)), Image.BILINEAR)
        pole = pole.crop((0, 0, pole.width, H_needed))

        side_margin = int(side_margin_frac * SW)
        GW = max(pole.width, SW + 2 * side_margin)
        GH = H_needed
        group = Image.new("RGBA", (GW, GH), (0, 0, 0, 0))

        px = (GW - pole.width) // 2
        group.alpha_composite(pole, (px, 0))

        sx = (GW - SW) // 2
        sy = clearance_px
        group.alpha_composite(sign, (sx, sy))
        return group

    def _place_group_on_background(self, group_rgba: Image.Image, bg_rgb: Image.Image, seed: int):
        rng = np.random.default_rng(seed)
        W, H = self.img_size
        bg_rgba = bg_rgb.resize((W, H), Image.BILINEAR).convert("RGBA")

        target_w = int(rng.uniform(0.30 * W, 0.55 * W))
        scale = target_w / max(1, group_rgba.width)
        group = group_rgba.resize((target_w, int(group_rgba.height * scale)), Image.BILINEAR)

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
        return Image.new("RGB", (200, 200, 200))


# ---------------------------- helpers (module level) ----------------------------

def _affine_matrix(angle_deg, shear_deg, scale, tx, ty, W, H):
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
    A = []
    for (x, y), (u, v) in zip(src, dst):
        A.extend([[x, y, 1, 0, 0, 0, -u * x, -u * y],
                  [0, 0, 0, x, y, 1, -v * x, -v * y]])
    A = np.array(A, dtype=np.float32)
    B = np.array([p for uv in dst for p in uv], dtype=np.float32)
    coeffs = np.linalg.lstsq(A, B, rcond=None)[0]
    return coeffs
