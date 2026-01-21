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

    Action:
      Discrete index into valid grid cells within the sign octagon.
      (cell size = grid_cell_px; configurable)

    Per step:
      - Add 1 new grid cell (grid_cell_px square) to a running episode mask.
      - Duplicate cells are disallowed: remap to a free cell.
      - Render three matched variants on the same background/pole/placement/transforms:
          0) Plain (no overlay) baseline for day and UV-on
          1) Daylight overlay (pre-activation color/alpha)
          2) UV-on overlay (activated color/alpha)
      - For robustness, evaluate each variant across K matched sign-only transforms
        with identical placement and background.
      - Compute mean confidences over K runs:
          c0_day, c_day, c0_on, c_on
        Primary objective is drop_on = c0_on - c_on (target threshold).
        Secondary objective keeps day confidence high (penalize if drop exceeds tolerance).
      - Episode terminates early when drop_on_mean meets uv_drop_threshold.

    Observation:
      RGB image of the *daylight* composite for the first transform (H,W,3) uint8.

    Reward (per step, normalized):
      Let raw_core = blend(drop_on_s, drop_on) - lambda_day * max(0, drop_day - day_tolerance)
                              - lambda_area * area_frac.
      Add a smooth shaping bonus as drop_on approaches threshold, plus a small
      success bonus once drop_on >= uv_drop_threshold, then squash:

          raw_total = raw_core + shaping + success_bonus
          reward    = tanh(1.2 * raw_total)    (-1, 1)

      so PPO always sees a bounded per-step reward.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        stop_sign_image: Image.Image,
        stop_sign_uv_image: Optional[Image.Image],
        background_images: List[Image.Image],
        pole_image: Optional[Image.Image],
        yolo_weights: str = "weights/yolo8n.pt",
        img_size: Tuple[int, int] = (640, 640),
        detector_debug: bool = False,


        # Episodes
        steps_per_episode: int = 7000,
        eval_K: int = 3,
        eval_K_min: Optional[int] = None,
        eval_K_max: Optional[int] = None,
        eval_K_ramp_threshold: Optional[float] = None,

        # Grid config
        grid_cell_px: int = 16,               # default grid size in pixels
        max_cells: Optional[int] = None,
        area_cap_frac: Optional[float] = None,

        # Paint (single pair)
        uv_paint: UVPaint = VIOLET_GLOW,
        use_single_color: bool = True,

        # Threshold logic
        uv_drop_threshold: float = 0.70,
        day_tolerance: float = 0.05,
        lambda_day: float = 1.0,
        lambda_area: float = 0.3,
        uv_min_alpha: float = 0.08,
        min_base_conf: float = 0.20,
        cell_cover_thresh: float = 0.60,
        area_cap_penalty: float = -0.20,
        area_cap_mode: str = "soft",


        # YOLO
        yolo_device: str = "cpu",
        conf_thresh: float = 0.10,
        iou_thresh: float = 0.45,
        info_image_every: int = 50,

        # Reward smoothing
        reward_smooth_n: int = 10,
        use_ema: bool = True,
        ema_beta: float = 0.90,

        seed: Optional[int] = None,
    ):
        super().__init__()
        self.img_size = tuple(img_size)
        self.steps_per_episode = int(steps_per_episode)
        self.eval_K_min = int(eval_K if eval_K_min is None else eval_K_min)
        self.eval_K_max = int(eval_K if eval_K_max is None else eval_K_max)
        if self.eval_K_min < 1:
            raise ValueError("eval_K_min must be >= 1")
        if self.eval_K_max < self.eval_K_min:
            raise ValueError("eval_K_max must be >= eval_K_min")
        self.eval_K_ramp_threshold = (
            float(eval_K_ramp_threshold)
            if eval_K_ramp_threshold is not None
            else 0.5 * float(uv_drop_threshold)
        )

        self.sign_rgba_day = stop_sign_image.convert("RGBA")
        self.sign_rgba_on  = (stop_sign_uv_image or stop_sign_image).convert("RGBA")
        self.bg_list = [im.convert("RGB") for im in (background_images or [])]
        self.pole_rgba = None if pole_image is None else pole_image.convert("RGBA")

        self.grid_cell_px = int(grid_cell_px)
        if self.grid_cell_px not in (2, 4, 8, 16, 32):
            raise ValueError("grid_cell_px must be one of: 2, 4, 8, 16, 32")

        self.cell_cover_thresh = float(cell_cover_thresh)

        # build sign alpha and grid on construction
        self._sign_alpha = self.sign_rgba_day.split()[-1]  # L mask of octagon
        self._build_grid_index()

        self.max_cells = int(max_cells) if max_cells is not None else None
        self.area_cap_frac = float(area_cap_frac) if area_cap_frac is not None else None
        self._derived_max_cells = (self.max_cells is None and self.area_cap_frac is not None)
        if self.area_cap_frac is not None and not (0.0 < self.area_cap_frac <= 1.0):
            raise ValueError("area_cap_frac must be in (0, 1]")
        if self._derived_max_cells:
            valid_total = int(self._valid_cells.sum())
            if valid_total > 0:
                derived = int(math.ceil(self.area_cap_frac * valid_total))
                self.max_cells = max(1, min(valid_total, derived))
            else:
                self.max_cells = 0

        # UV paint pair (single)
        self.paint = uv_paint
        self.use_single_color = bool(use_single_color)

        # threshold / reward
        self.uv_drop_threshold = float(uv_drop_threshold)
        self.day_tolerance = float(day_tolerance)
        self.lambda_day = float(lambda_day)
        self.lambda_area = float(lambda_area)
        self.uv_min_alpha = float(uv_min_alpha)
        self.min_base_conf = float(min_base_conf)
        self.info_image_every = int(info_image_every)
        self.area_cap_penalty = float(area_cap_penalty)
        self.area_cap_mode = str(area_cap_mode).lower().strip()
        if self.area_cap_mode not in ("soft", "hard"):
            raise ValueError("area_cap_mode must be 'soft' or 'hard'")


        # Reward smoothing state (Option A)
        self.reward_smooth_n = int(reward_smooth_n)
        self.use_ema = bool(use_ema)
        self.ema_beta = float(ema_beta)
        self._drop_hist = deque(maxlen=self.reward_smooth_n)
        self._drop_ema = None

        # detector
        dev_str = str(yolo_device)
        if dev_str.lower().startswith("server://"):
            from detectors.remote_detector import RemoteDetectorWrapper
            self.det = RemoteDetectorWrapper(
                server_addr=dev_str,
                conf=conf_thresh,
                iou=iou_thresh,
                debug=detector_debug,
            )
        else:
            self.det = DetectorWrapper(
                yolo_weights, device=yolo_device, conf=conf_thresh, iou=iou_thresh, debug=detector_debug
            )


        # action/obs spaces
        self.action_space = spaces.Discrete(self._n_valid)
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
        self._baseline_c0_day_list: List[float] = []
        self._baseline_c0_on_list: List[float] = []
        self._last_drop_on_s = 0.0

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
        self._valid_coords = np.argwhere(self._valid_cells)  # shape (N,2)
        self._n_valid = int(self._valid_coords.shape[0])


    # ----------------------------- lifecycle ---------------------------------

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self._step = 0
        self._episode_cells = np.zeros((self.Gh, self.Gw), dtype=bool)

        self._bg_rgb = self._choose_bg_rgb()
        self._place_seed = int(self.rng.integers(0, 2**31 - 1))
        self._transform_seeds = [
            int(self.rng.integers(0, 2**31 - 1)) for _ in range(self.eval_K_max)
        ]

        obs = self._render_variant(kind="day", use_overlay=False, transform_seed=self._transform_seeds[0])

        # Cache plain baseline confidences (same throughout episode)
        self._baseline_c0_day_list, self._baseline_c0_on_list = self._eval_plain_over_K(
            self._transform_seeds
        )

        # Reset reward smoothing buffers (Option A)
        self._drop_hist.clear()
        self._drop_ema = None
        self._last_drop_on_s = 0.0

        return np.array(obs, dtype=np.uint8), {}

    # ----------------------------- step --------------------------------------

    def step(self, action):
        self._step += 1

        # 1) Action is an index into valid cells (octagon-aware)
        idx = int(action)
        idx = max(0, min(idx, self._n_valid - 1))
        r, c = self._valid_coords[idx]
        r, c = int(r), int(c)

        # Enforce non-duplicate: if already used, sample a random unused valid cell
        if self._episode_cells[r, c]:
            free_mask = self._valid_cells & (~self._episode_cells)
            coords = np.argwhere(free_mask)
            if coords.size == 0:
                # no free cells left
                terminated = True
                truncated = (self._step >= self.steps_per_episode)

                obs_img = self._render_variant(kind="day", use_overlay=True, transform_seed=self._transform_seeds[0])
                obs = np.array(obs_img, dtype=np.uint8)
                area_frac = self._area_frac_selected()
                cap_exceeded = self.area_cap_frac is not None and area_frac > self.area_cap_frac
                info = {
                    "objective": "grid_uv",
                    "note": "no_free_cells",
                    "lambda_area": float(self.lambda_area),
                    "total_area_mask_frac": area_frac,
                    "area_cap": float(self.area_cap_frac) if self.area_cap_frac is not None else 0.0,
                    "uv_success": False,
                    "area_cap_exceeded": cap_exceeded,
                }
                return obs, 0.0, bool(terminated), bool(truncated), info

            pick = coords[int(self.rng.integers(0, coords.shape[0]))]
            r, c = int(pick[0]), int(pick[1])

        selected_cells = int(self._episode_cells.sum())
        valid_total = int(self._valid_cells.sum())
        if self.area_cap_frac is not None and valid_total > 0 and self.area_cap_mode == "hard":
            next_area_frac = float(selected_cells + 1) / float(valid_total)
            if next_area_frac > self.area_cap_frac:
                terminated = True
                truncated = (self._step >= self.steps_per_episode)
                obs_img = self._render_variant(kind="day", use_overlay=True, transform_seed=self._transform_seeds[0])
                obs = np.array(obs_img, dtype=np.uint8)
                area_frac = float(selected_cells) / float(valid_total)
                info = {
                    "objective": "grid_uv",
                    "note": "area_cap_exceeded",
                    "selected_cells": int(selected_cells),
                    "total_area_mask_frac": float(area_frac),
                    "area_cap": float(self.area_cap_frac),
                    "uv_success": False,
                    "area_cap_exceeded": True,
                }
                return obs, float(self.area_cap_penalty), bool(terminated), bool(truncated), info

        self._episode_cells[r, c] = True

        area_frac = self._area_frac_selected()
        cap_exceeded = self.area_cap_frac is not None and area_frac > self.area_cap_frac

        terminated = False
        max_cells_reached = False
        if self.max_cells is not None:
            if int(self._episode_cells.sum()) >= self.max_cells:
                terminated = True
                max_cells_reached = True

        # 2) Evaluate overlay vs baseline
        eval_K = self._current_eval_K(self._last_drop_on_s)
        eval_seeds = self._transform_seeds[:eval_K]
        c_day, c_on = self._eval_overlay_over_K(eval_seeds)
        c0_day = self._mean_over_K(self._baseline_c0_day_list, eval_K)
        c0_on = self._mean_over_K(self._baseline_c0_on_list, eval_K)

        drop_day = float(c0_day - c_day)
        drop_on  = float(c0_on - c_on)

        # (Option A reward smoothing) Smooth UV drop signal over recent steps
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
        self._last_drop_on_s = drop_on_s

        area_frac = self._area_frac_selected()
        cap_exceeded = self.area_cap_frac is not None and area_frac > self.area_cap_frac

        # (Baseline gating)
        if c0_on < self.min_base_conf:
            reward = -0.05
            # keep termination from max_cells if you want; I'm leaving it as-is:
            truncated = (self._step >= self.steps_per_episode)

            obs_img = self._render_variant(kind="day", use_overlay=True, transform_seed=self._transform_seeds[0])
            obs = np.array(obs_img, dtype=np.uint8)
            info = {
                "objective": "grid_uv",
                "c0_day": c0_day, "c_day": c_day,
                "c0_on": c0_on,   "c_on": c_on,
                "drop_day": drop_day, "drop_on": drop_on, "drop_on_smooth": float(drop_on_s),
                "note": "max_cells_reached" if max_cells_reached else "baseline_conf_too_low",
                "min_base_conf": float(self.min_base_conf),
                "total_area_mask_frac": float(area_frac),
                "area_cap": float(self.area_cap_frac) if self.area_cap_frac is not None else None,
                "uv_success": False,
                "area_cap_exceeded": bool(cap_exceeded),
            }
            return obs, float(reward), bool(terminated), bool(truncated), info

        # 3) Reward using smoothed UV drop
        pen_day = max(0.0, drop_day - self.day_tolerance)
        area_frac = self._area_frac_selected()
        drop_blend = 0.7 * drop_on_s + 0.3 * drop_on
        raw_core = drop_blend - self.lambda_day * pen_day - self.lambda_area * area_frac

        thr = self.uv_drop_threshold
        shaping = 0.35 * math.tanh(3.0 * (drop_on_s - 0.5 * thr))
        uv_success = drop_on_s >= thr and (self.area_cap_frac is None or area_frac <= self.area_cap_frac)
        success_bonus = 0.2 if uv_success else 0.0

        raw_total = raw_core + shaping + success_bonus
        if cap_exceeded and self.area_cap_mode == "soft":
            if self.area_cap_frac and self.area_cap_frac > 0:
                excess = max(0.0, (area_frac - self.area_cap_frac) / self.area_cap_frac)
                raw_total += float(self.area_cap_penalty) * (1.0 + 2.0 * excess)
            else:
                raw_total += float(self.area_cap_penalty)
        reward = math.tanh(1.2 * raw_total)

        if drop_on_s >= thr:
            terminated = True
            if self.area_cap_frac is not None and area_frac > self.area_cap_frac:
                uv_success = False
                cap_exceeded = True

        truncated = (self._step >= self.steps_per_episode)

        # 4) Observation and preview
        obs_img = self._render_variant(kind="day", use_overlay=True, transform_seed=self._transform_seeds[0])
        obs = np.array(obs_img, dtype=np.uint8)
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
            "eval_K_used": int(eval_K),
            "eval_K_min": int(self.eval_K_min),
            "eval_K_max": int(self.eval_K_max),
            "uv_drop_threshold": float(self.uv_drop_threshold),
            "day_tolerance": float(self.day_tolerance),
            "lambda_area": float(self.lambda_area),
            "base_conf": float(c0_on),
            "after_conf": float(c_on),
            "total_area_mask_frac": float(area_frac),
            "area_cap": float(self.area_cap_frac) if self.area_cap_frac is not None else None,
            "uv_success": bool(uv_success),
            "area_cap_exceeded": bool(cap_exceeded),
            "trace": {
                "phase": "grid_uv",
                "grid_cell_px": int(self.grid_cell_px),
                "selected_indices": self._selected_indices_list(),
                "place_seed": int(self._place_seed),
                "transform_seeds": [int(s) for s in self._transform_seeds],
            },
        }
        if max_cells_reached:
            info["note"] = "max_cells_reached"

        # Always attach the final image if we succeeded (hit threshold), so it can be saved reliably.
        if terminated and uv_success:
            info["composited_pil"] = preview_on
            info["overlay_pil"] = self._render_overlay_pattern(mode="on")
        # Otherwise only attach occasionally to reduce overhead
        elif (self._step % self.info_image_every) == 0:
            info["composited_pil"] = preview_on
            info["overlay_pil"] = self._render_overlay_pattern(mode="on")


        return obs, float(reward), bool(terminated), bool(truncated), info


    # ----------------------------- helpers -----------------------------------

    def _eval_plain_over_K(self, seeds: List[int]) -> Tuple[List[float], List[float]]:
        imgs_plain_day, imgs_plain_on = [], []
        for t_seed in seeds:
            plain_day = self._compose_on_bg(self._transform_sign(self.sign_rgba_day, t_seed), self._place_seed)
            plain_on  = self._compose_on_bg(self._transform_sign(self.sign_rgba_on,  t_seed), self._place_seed)
            imgs_plain_day.append(plain_day)
            imgs_plain_on.append(plain_on)
        c0_day_list = self.det.infer_confidence_batch(imgs_plain_day)
        c0_on_list  = self.det.infer_confidence_batch(imgs_plain_on)
        return list(c0_day_list), list(c0_on_list)

    def _mean_over_K(self, values: List[float], K: int) -> float:
        if K <= 0:
            return 0.0
        return float(np.mean(values[:K])) if values else 0.0

    def _current_eval_K(self, drop_on_s: float) -> int:
        if self.eval_K_min == self.eval_K_max:
            return int(self.eval_K_max)

        ramp_start = float(self.eval_K_ramp_threshold)
        ramp_end = float(self.uv_drop_threshold)
        if drop_on_s <= ramp_start:
            return int(self.eval_K_min)
        if ramp_end <= ramp_start:
            return int(self.eval_K_max)
        if drop_on_s >= ramp_end:
            return int(self.eval_K_max)

        t = (drop_on_s - ramp_start) / (ramp_end - ramp_start)
        k = self.eval_K_min + t * (self.eval_K_max - self.eval_K_min)
        return int(max(self.eval_K_min, min(self.eval_K_max, math.ceil(k))))


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

    def set_area_cap_frac(self, value: Optional[float]) -> None:
        """
        Update area cap and derived max_cells at runtime.

        @param value: New cap fraction (None or <=0 disables).
        """
        if value is None or float(value) <= 0.0:
            self.area_cap_frac = None
            if self._derived_max_cells:
                self.max_cells = None
            return
        v = float(value)
        if not (0.0 < v <= 1.0):
            raise ValueError("area_cap_frac must be in (0, 1]")
        self.area_cap_frac = v
        if self._derived_max_cells:
            valid_total = int(self._valid_cells.sum())
            if valid_total > 0:
                derived = int(math.ceil(self.area_cap_frac * valid_total))
                self.max_cells = max(1, min(valid_total, derived))
            else:
                self.max_cells = 0

    def set_lambda_area(self, value: float) -> None:
        """
        Update area penalty weight at runtime.

        @param value: New lambda_area value.
        """
        self.lambda_area = float(value)

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

    def _eval_overlay_over_K(self, seeds: List[int]) -> Tuple[float, float]:
        imgs_over_day, imgs_over_on = [], []

        # Precompute overlays once per step
        over_sign_day = self._apply_grid_overlay(self.sign_rgba_day, mode="day")
        over_sign_on  = self._apply_grid_overlay(self.sign_rgba_on,  mode="on")

        for t_seed in seeds:
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
            if self.paint.translucent:
                alpha = max(alpha, self.uv_min_alpha)

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

    def _render_overlay_pattern(self, mode: str) -> Image.Image:
        """Render only the overlay pattern on a transparent background."""
        assert mode in ("day", "on")
        size = self.sign_rgba_day.size
        img = Image.new("RGBA", size, (0, 0, 0, 0))

        if mode == "day":
            color = self.paint.day_rgb
            alpha = self.paint.day_alpha if self.paint.translucent else 1.0
        else:
            color = self.paint.active_rgb
            alpha = self.paint.active_alpha if self.paint.translucent else 1.0
            if self.paint.translucent:
                alpha = max(alpha, self.uv_min_alpha)

        mask = Image.new("L", size, 0)
        mdraw = ImageDraw.Draw(mask)
        on = np.argwhere(self._episode_cells)
        for r, c in on:
            x0, y0, x1, y1 = self._cell_rects[r * self.Gw + c]
            mdraw.rectangle([x0, y0, x1 - 1, y1 - 1], fill=255)

        mask = Image.composite(mask, Image.new("L", size, 0), self._sign_alpha)
        if alpha < 1.0:
            arr = (np.array(mask, dtype=np.float32) * float(alpha)).astype(np.uint8)
            mask = Image.fromarray(arr, mode="L")

        img.paste(color, mask=mask)
        return img

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

        # Wider distance variance: allow sign to appear smaller (farther) or larger (closer).
        target_w = int(rng.uniform(0.12 * W, 0.60 * W))
        scale = target_w / max(1, group_rgba.width)
        group = group_rgba.resize((target_w, int(group_rgba.height * scale)), Image.BILINEAR)

        margin = int(0.04 * W)
        max_x = max(margin, W - group.width - margin)
        max_y = max(margin, H - group.height - margin)
        # Force placement to left/right sides so the sign isn't centered.
        left_max = max(margin, min(max_x, int(0.40 * W)))
        right_min = max(margin, min(max_x, int(0.60 * W)))
        if rng.random() < 0.5:
            x = int(rng.integers(margin, left_max + 1))
        else:
            x = int(rng.integers(right_min, max_x + 1))
        # Avoid placing the sign too high in the frame.
        min_y = max(margin, int(0.12 * H))
        y = int(rng.integers(min_y, max_y + 1))

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
