from __future__ import annotations
from typing import Tuple, Dict, Any, List, Optional
import math
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter

import gymnasium as gym
from gymnasium import spaces

from detectors.yolo_wrapper import DetectorWrapper
from utils.uv_paint import UVPaint, YELLOW_GLOW  # you can swap the paint in train file


class StopSignGridEnv(gym.Env):
    """
    Grid-square adversarial overlay over a stop sign (octagon mask only).

    Action:
      Discrete index into valid grid cells within the sign octagon.
      (cell size = grid_cell_px; configurable)

    Per step:
      - Add 1 new grid cell (grid_cell_px square) to a running episode mask.
      - Duplicate cells are disallowed: action is remapped to a free cell deterministically.
      - Render three matched variants on the same background/pole/placement/transforms:
          0) Plain (no overlay) baseline for day and UV-on
          1) Daylight overlay (pre-activation color/alpha)
          2) UV-on overlay (activated color/alpha)
      - For robustness, evaluate each variant across K matched sign-only transforms
        with identical placement and background.
      - Compute mean confidences over K runs:
          c0_day, c_day, c0_on, c_on
        Primary objective is drop_on = c0_day - c_on (target threshold).
        Secondary objective keeps day confidence high (penalize if drop exceeds tolerance).
      - Additional objectives use mean IoU (target vs top detection) and misclassification rate.
      - Episode terminates early when after-conf <= success threshold.

    Observation:
      Cropped RGB image around the sign (daylight composite) with optional
      overlay-mask channel (H,W,3 or H,W,4) uint8.

    Reward (per step, normalized):
      Let raw_core = drop_on
                     - lambda_day * max(0, drop_day - day_tolerance)
                     - lambda_area_used * area_frac
                     + lambda_iou * (1 - mean_iou)
                     + lambda_misclass * misclass_rate
                     + lambda_efficiency * log1p(drop_on / area_frac).
      lambda_area_used can be adaptive (Lagrangian) when area_target_frac is set.

      Add a smooth shaping bonus as after-conf approaches the success threshold,
      plus a small success bonus once success criteria are met, then squash:

          raw_total = raw_core + shaping + success_bonus
          reward    = tanh(1.2 * raw_total)    (-1, 1)

      Efficiency bonus (optional):
        efficiency = log1p(max(0, drop_on) / max(area_frac, efficiency_eps))
        raw_core += lambda_efficiency * efficiency

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
        uv_paint: UVPaint = YELLOW_GLOW,
        uv_paint_list: Optional[List[UVPaint]] = None,
        use_single_color: bool = True,

        # Threshold logic
        uv_drop_threshold: float = 0.75,
        success_conf_threshold: float = 0.20,
        day_tolerance: float = 0.05,
        lambda_day: float = 1.0,
        lambda_area: float = 0.3,
        area_target_frac: Optional[float] = None,
        area_lagrange_lr: float = 0.0,
        area_lagrange_min: float = 0.0,
        area_lagrange_max: float = 5.0,
        step_cost: float = 0.0,
        step_cost_after_target: float = 0.0,
        lambda_iou: float = 0.4,
        lambda_misclass: float = 0.6,
        lambda_efficiency: float = 0.0,
        efficiency_eps: float = 0.02,
        lambda_perceptual: float = 0.0,
        transform_strength: float = 1.0,
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
        diag_area_thresh: float = 0.80,
        diag_conf_thresh: float = 0.70,

        # Observation crop
        obs_size: Tuple[int, int] = (224, 224),
        obs_margin: float = 0.10,
        obs_include_mask: bool = True,

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
        self.area_cap_mode = str(area_cap_mode).lower().strip()
        if self.area_cap_mode not in ("soft", "hard"):
            raise ValueError("area_cap_mode must be 'soft' or 'hard'")
        self._derived_max_cells = (
            self.max_cells is None
            and self.area_cap_frac is not None
            and self.area_cap_mode == "hard"
        )
        if self.area_cap_frac is not None and not (0.0 < self.area_cap_frac <= 1.0):
            raise ValueError("area_cap_frac must be in (0, 1]")
        if self._derived_max_cells:
            valid_total = int(self._valid_cells.sum())
            if valid_total > 0:
                derived = int(math.ceil(self.area_cap_frac * valid_total))
                self.max_cells = max(1, min(valid_total, derived))
            else:
                self.max_cells = 0

        # UV paint pair (single or list)
        self.paint_list = list(uv_paint_list) if uv_paint_list else None
        self.paint = uv_paint
        self.use_single_color = bool(use_single_color)

        # threshold / reward
        self.uv_drop_threshold = float(uv_drop_threshold)
        self.success_conf_threshold = float(success_conf_threshold)
        self.day_tolerance = float(day_tolerance)
        self.lambda_day = float(lambda_day)
        self.lambda_area = float(lambda_area)
        self.area_target_frac = float(area_target_frac) if area_target_frac is not None else None
        self.area_lagrange_lr = float(area_lagrange_lr)
        self.area_lagrange_min = float(area_lagrange_min)
        self.area_lagrange_max = float(area_lagrange_max)
        self._lambda_area_dyn = float(lambda_area)
        self.step_cost = float(step_cost)
        self.step_cost_after_target = float(step_cost_after_target)
        self.lambda_iou = float(lambda_iou)
        self.lambda_misclass = float(lambda_misclass)
        self.lambda_efficiency = float(lambda_efficiency)
        self.efficiency_eps = float(efficiency_eps)
        self.lambda_perceptual = float(lambda_perceptual)
        self.transform_strength = float(transform_strength)
        self.uv_min_alpha = float(uv_min_alpha)
        self.min_base_conf = float(min_base_conf)
        self.info_image_every = int(info_image_every)
        self.diag_area_thresh = float(diag_area_thresh)
        self.diag_conf_thresh = float(diag_conf_thresh)
        self.obs_size = (int(obs_size[0]), int(obs_size[1]))
        self.obs_margin = float(obs_margin)
        self.obs_include_mask = bool(obs_include_mask)
        self.area_cap_penalty = float(area_cap_penalty)
        if self.area_target_frac is not None:
            if not (0.0 < self.area_target_frac <= 1.0):
                raise ValueError("area_target_frac must be in (0, 1]")
        if self.area_lagrange_max < self.area_lagrange_min:
            raise ValueError("area_lagrange_max must be >= area_lagrange_min")


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
        H, W = self.obs_size[1], self.obs_size[0]
        C = 4 if self.obs_include_mask else 3
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(H, W, C), dtype=np.uint8
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
        self._diag_saved = False

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

        if self.paint_list:
            pick = int(self.rng.integers(0, len(self.paint_list)))
            self.paint = self.paint_list[pick]

        self._bg_rgb = self._choose_bg_rgb()
        self._place_seed = int(self.rng.integers(0, 2**31 - 1))
        self._transform_seeds = [
            int(self.rng.integers(0, 2**31 - 1)) for _ in range(self.eval_K_max)
        ]

        obs = self._render_observation(kind="day", use_overlay=False, transform_seed=self._transform_seeds[0])

        # Cache plain baseline confidences (same throughout episode)
        self._baseline_c0_day_list, self._baseline_c0_on_list = self._eval_plain_over_K(
            self._transform_seeds
        )

        self._last_drop_on_s = 0.0
        self._diag_saved = False

        return np.array(obs, dtype=np.uint8), {}

    # ----------------------------- step --------------------------------------

    def step(self, action):
        self._step += 1

        # 1) Action is an index into valid cells (octagon-aware)
        idx = int(action)
        idx = max(0, min(idx, self._n_valid - 1))
        free_mask = self._valid_cells & (~self._episode_cells)
        coords = np.argwhere(free_mask)
        if coords.size == 0:
            # no free cells left
            terminated = True
            truncated = (self._step >= self.steps_per_episode)

            obs = self._render_observation(kind="day", use_overlay=True, transform_seed=self._transform_seeds[0])
            area_frac = self._area_frac_selected()
            cap_exceeded = self.area_cap_frac is not None and area_frac > self.area_cap_frac
            info = {
                "objective": "grid_uv",
                "note": "no_free_cells",
                "lambda_area": float(self.lambda_area),
                "total_area_mask_frac": area_frac,
                "area_cap": float(self.area_cap_frac) if self.area_cap_frac is not None else 0.0,
                "uv_success": False,
                "attack_success": False,
                "area_cap_exceeded": cap_exceeded,
            }
            return obs, -1.0, bool(terminated), bool(truncated), info

        pick = coords[idx % coords.shape[0]]
        r, c = int(pick[0]), int(pick[1])

        selected_cells = int(self._episode_cells.sum())
        valid_total = int(self._valid_cells.sum())
        if self.area_cap_frac is not None and valid_total > 0 and self.area_cap_mode == "hard":
            next_area_frac = float(selected_cells + 1) / float(valid_total)
            if next_area_frac > self.area_cap_frac:
                terminated = True
                truncated = (self._step >= self.steps_per_episode)
                obs = self._render_observation(kind="day", use_overlay=True, transform_seed=self._transform_seeds[0])
                area_frac = float(selected_cells) / float(valid_total)
                info = {
                    "objective": "grid_uv",
                    "note": "area_cap_exceeded",
                    "selected_cells": int(selected_cells),
                    "total_area_mask_frac": float(area_frac),
                    "area_cap": float(self.area_cap_frac),
                    "uv_success": False,
                    "attack_success": False,
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
        overlay_metrics = self._eval_overlay_over_K(eval_seeds)
        c_day = overlay_metrics["c_day"]
        c_on = overlay_metrics["c_on"]
        mean_iou = overlay_metrics["mean_iou"]
        misclass_rate = overlay_metrics["misclass_rate"]
        mean_target_conf = overlay_metrics.get("mean_target_conf", 0.0)
        mean_top_conf = overlay_metrics.get("mean_top_conf", 0.0)
        top_class_counts = overlay_metrics.get("top_class_counts", {})

        c0_day = self._mean_over_K(self._baseline_c0_day_list, eval_K)
        c0_on = self._mean_over_K(self._baseline_c0_on_list, eval_K)

        drop_day = float(c0_day - c_day)
        drop_on  = float(c0_day - c_on)

        drop_on_s = float(drop_on)
        self._last_drop_on_s = drop_on_s

        area_frac = self._area_frac_selected()
        cap_exceeded = self.area_cap_frac is not None and area_frac > self.area_cap_frac

        # (Baseline gating)
        if c0_day < self.min_base_conf:
            reward = -0.05
            # keep termination from max_cells if you want; I'm leaving it as-is:
            truncated = (self._step >= self.steps_per_episode)

            obs = self._render_observation(kind="day", use_overlay=True, transform_seed=self._transform_seeds[0])
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
                "attack_success": False,
                "area_cap_exceeded": bool(cap_exceeded),
            }
            return obs, float(reward), bool(terminated), bool(truncated), info

        # 3) Reward using raw UV drop (no smoothing for core objective)
        pen_day = max(0.0, drop_day - self.day_tolerance)
        area_frac = self._area_frac_selected()
        conf_thr = self.success_conf_threshold
        max_drop = max(0.0, float(c0_day - conf_thr))
        drop_blend = min(float(drop_on), max_drop)
        eff_drop = max(0.0, float(drop_on))
        eff_denom = max(float(area_frac), float(self.efficiency_eps))
        efficiency = math.log1p(eff_drop / eff_denom)
        lambda_area_used = float(self.lambda_area)
        area_target = self.area_target_frac if self.area_target_frac is not None else self.area_cap_frac
        if self.area_lagrange_lr > 0.0 and area_target is not None and float(area_target) > 0.0:
            violation = float(area_frac) - float(area_target)
            self._lambda_area_dyn = float(
                np.clip(
                    self._lambda_area_dyn + (self.area_lagrange_lr * violation),
                    self.area_lagrange_min,
                    self.area_lagrange_max,
                )
            )
            lambda_area_used = float(self._lambda_area_dyn)
        step_cost_penalty = float(self.step_cost)
        if self.step_cost_after_target > 0.0 and area_target is not None and area_frac > float(area_target):
            excess = (float(area_frac) - float(area_target)) / max(float(area_target), 1e-6)
            step_cost_penalty += float(self.step_cost_after_target) * (1.0 + max(0.0, excess))

        excess_penalty = 0.0
        if area_target is not None and area_frac > float(area_target):
            excess = float(area_frac) - float(area_target)
            # Stronger push against exceeding the target.
            excess_penalty = (lambda_area_used * 2.5 * excess) + (lambda_area_used * (excess ** 2))

        raw_core = (
            drop_blend
            - self.lambda_day * pen_day
            - lambda_area_used * area_frac
            - excess_penalty
            - step_cost_penalty
            + self.lambda_iou * (1.0 - mean_iou)
            + self.lambda_misclass * misclass_rate
            + self.lambda_efficiency * efficiency
        )
        perceptual = self._perceptual_delta()
        raw_core -= self.lambda_perceptual * perceptual

        shaping = 0.35 * math.tanh(3.0 * (conf_thr - c_on))
        conf_success = self._is_drop_success(c_on, area_frac, conf_thr)
        attack_success = bool(conf_success)
        success_bonus = (0.2 * ((1.0 - float(area_frac)) ** 2)) if conf_success else 0.0

        raw_total = raw_core + shaping + success_bonus
        if cap_exceeded and self.area_cap_mode == "soft":
            if self.area_cap_frac and self.area_cap_frac > 0:
                excess = max(0.0, (area_frac - self.area_cap_frac) / self.area_cap_frac)
                over_pen = abs(float(self.area_cap_penalty)) * (1.0 + 2.0 * excess)
            else:
                over_pen = abs(float(self.area_cap_penalty))
            raw_total = -over_pen
        reward = math.tanh(1.2 * raw_total)

        if conf_success:
            terminated = True
        # Success is purely confidence-based; cap only affects reward (soft mode)

        truncated = (self._step >= self.steps_per_episode)

        # 4) Observation and preview
        obs = self._render_observation(kind="day", use_overlay=True, transform_seed=self._transform_seeds[0])
        preview_on = self._render_variant(kind="on", use_overlay=True, transform_seed=self._transform_seeds[0])

        target_id = getattr(self.det, "target_id", None)
        target_name = None
        id_to_name = getattr(self.det, "id_to_name", None)
        if target_id is not None and isinstance(id_to_name, dict):
            target_name = id_to_name.get(int(target_id))

        info = {
            "objective": "grid_uv",
            "c0_day": c0_day, "c_day": c_day,
            "c0_on": c0_on,   "c_on": c_on,
            "drop_day": drop_day, "drop_on": drop_on, "drop_on_smooth": float(drop_on_s),
            "reward_core": float(raw_core),
            "reward_efficiency": float(self.lambda_efficiency * efficiency),
            "reward_perceptual": float(-self.lambda_perceptual * perceptual),
            "reward_step_cost": float(-step_cost_penalty),
            "reward_raw_total": float(raw_total),
            "reward": float(reward),
            "lambda_area_used": float(lambda_area_used),
            "lambda_area_dyn": float(self._lambda_area_dyn),
            "area_target_frac": float(area_target) if area_target is not None else None,
            "area_lagrange_lr": float(self.area_lagrange_lr),
            "step_cost": float(self.step_cost),
            "step_cost_after_target": float(self.step_cost_after_target),
            "mean_iou": float(mean_iou),
            "misclass_rate": float(misclass_rate),
            "mean_target_conf": float(mean_target_conf),
            "mean_top_conf": float(mean_top_conf),
            "top_class_counts": dict(top_class_counts),
            "target_id": int(target_id) if target_id is not None else None,
            "target_name": str(target_name) if target_name is not None else None,
            "selected_cells": int(self._episode_cells.sum()),
            "grid_cell_px": int(self.grid_cell_px),
            "eval_K_used": int(eval_K),
            "eval_K_min": int(self.eval_K_min),
            "eval_K_max": int(self.eval_K_max),
            "uv_drop_threshold": float(self.uv_drop_threshold),
            "success_conf_threshold": float(self.success_conf_threshold),
            "day_tolerance": float(self.day_tolerance),
            "lambda_area": float(self.lambda_area),
            "lambda_iou": float(self.lambda_iou),
            "lambda_misclass": float(self.lambda_misclass),
            "lambda_efficiency": float(self.lambda_efficiency),
            "efficiency_eps": float(self.efficiency_eps),
            "lambda_perceptual": float(self.lambda_perceptual),
            "perceptual_delta": float(perceptual),
            "paint_name": getattr(self.paint, "name", "unknown"),
            "base_conf": float(c0_day),
            "after_conf": float(c_on),
            "total_area_mask_frac": float(area_frac),
            "area_cap": float(self.area_cap_frac) if self.area_cap_frac is not None else None,
            "uv_success": bool(conf_success),
            "attack_success": bool(attack_success),
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

        diagnostic = (
            (not self._diag_saved)
            and (area_frac >= self.diag_area_thresh)
            and (c_on >= self.diag_conf_thresh)
        )
        if diagnostic:
            self._diag_saved = True
            info["diagnostic"] = True
            info["diagnostic_reason"] = "high_coverage_high_conf"
            info["diagnostic_area_thresh"] = float(self.diag_area_thresh)
            info["diagnostic_conf_thresh"] = float(self.diag_conf_thresh)

        # Always attach the final image if we hit drop success, so it can be saved reliably.
        if terminated and conf_success:
            info["composited_pil"] = preview_on
            info["overlay_pil"] = self._render_overlay_pattern(mode="on")
        # Otherwise only attach occasionally to reduce overhead
        elif diagnostic or (self._step % self.info_image_every) == 0:
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

    def _render_variant(
        self,
        kind: str,
        use_overlay: bool,
        transform_seed: int,
        return_meta: bool = False,
    ):
        assert kind in ("day", "on")
        sign_src = self.sign_rgba_day if kind == "day" else self.sign_rgba_on
        sign = sign_src if not use_overlay else self._apply_grid_overlay(
            sign_src, mode=("day" if kind == "day" else "on")
        )
        sign_t = self._transform_sign(sign, transform_seed)
        if return_meta:
            return self._compose_on_bg(sign_t, self._place_seed, return_meta=True)
        return self._compose_on_bg(sign_t, self._place_seed)

    def _crop_observation(self, img: Image.Image, sign_bbox_bg: Tuple[float, float, float, float], resample):
        W, H = img.size
        x1, y1, x2, y2 = [float(v) for v in sign_bbox_bg]
        if x2 <= x1 or y2 <= y1:
            crop = img
        else:
            pad = int(round(max(x2 - x1, y2 - y1) * self.obs_margin))
            cx1 = max(0, int(math.floor(x1 - pad)))
            cy1 = max(0, int(math.floor(y1 - pad)))
            cx2 = min(W, int(math.ceil(x2 + pad)))
            cy2 = min(H, int(math.ceil(y2 + pad)))
            if cx2 <= cx1 or cy2 <= cy1:
                crop = img
            else:
                crop = img.crop((cx1, cy1, cx2, cy2))
        if crop.size != self.obs_size:
            crop = crop.resize(self.obs_size, resample=resample)
        return crop

    def _render_observation(self, kind: str, use_overlay: bool, transform_seed: int) -> np.ndarray:
        img, meta = self._render_variant(kind=kind, use_overlay=use_overlay, transform_seed=transform_seed, return_meta=True)
        sign_bbox_bg = meta.get("sign_bbox_bg", (0, 0, img.size[0], img.size[1]))
        crop = self._crop_observation(img, sign_bbox_bg, resample=Image.BILINEAR)

        if not self.obs_include_mask:
            return np.array(crop, dtype=np.uint8)

        # Build overlay mask aligned to the sign placement.
        overlay = self._render_overlay_pattern(mode="on")
        overlay_t = self._transform_sign(overlay, transform_seed)
        x1, y1, x2, y2 = [float(v) for v in sign_bbox_bg]
        mw = max(1, int(round(x2 - x1)))
        mh = max(1, int(round(y2 - y1)))
        overlay_t = overlay_t.resize((mw, mh), resample=Image.NEAREST)
        mask = Image.new("L", self.img_size, 0)
        alpha = overlay_t.split()[-1]
        mask.paste(alpha, (int(round(x1)), int(round(y1))))
        mask_crop = self._crop_observation(mask, sign_bbox_bg, resample=Image.NEAREST)

        rgb = np.array(crop, dtype=np.uint8)
        m = np.array(mask_crop, dtype=np.uint8)
        return np.dstack([rgb, m])

    def set_area_cap_frac(self, value: Optional[float]) -> None:
        """
        Update area cap and derived max_cells at runtime (hard mode only).

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
        self._lambda_area_dyn = float(value)

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

    def _eval_overlay_over_K(self, seeds: List[int]) -> Dict[str, Any]:
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

        det_on = self.det.infer_detections_batch(imgs_over_on)
        iou_vals = []
        misclass_vals = []
        target_conf_vals = []
        top_conf_vals = []
        top_class_counts: Dict[int, int] = {}
        for det in det_on:
            target_conf = float(det.get("target_conf", 0.0))
            top_conf = float(det.get("top_conf", 0.0))
            top_class = det.get("top_class", None)
            target_box = det.get("target_box", None)
            top_box = det.get("top_box", None)
            misclass = (top_class is not None and top_class != self.det.target_id and top_conf > 0.0)
            misclass_vals.append(1.0 if misclass else 0.0)
            target_conf_vals.append(target_conf)
            top_conf_vals.append(top_conf)
            if top_class is not None:
                cls = int(top_class)
                top_class_counts[cls] = top_class_counts.get(cls, 0) + 1

            iou = 0.0
            if target_box is not None and top_box is not None:
                iou = _iou_xyxy(target_box, top_box)
            iou_vals.append(float(iou))

        mean = lambda xs: float(np.mean(xs)) if len(xs) else 0.0
        return {
            "c_day": mean(c_day_list),
            "c_on": mean(c_on_list),
            "mean_iou": mean(iou_vals),
            "misclass_rate": mean(misclass_vals),
            "mean_target_conf": mean(target_conf_vals),
            "mean_top_conf": mean(top_conf_vals),
            "top_class_counts": top_class_counts,
        }

    def _is_attack_success(
        self,
        drop_on_s: float,
        mean_iou: float,
        misclass_rate: float,
        area_frac: float,
        threshold: float,
    ) -> bool:
        base_drop = float(drop_on_s) >= float(threshold)
        attack_signal = (float(misclass_rate) > 0.0) or (float(mean_iou) < 0.25)
        within_cap = (self.area_cap_frac is None) or (float(area_frac) <= float(self.area_cap_frac))
        return base_drop and attack_signal and within_cap

    def _is_drop_success(self, after_conf: float, area_frac: float, threshold: float) -> bool:
        conf_ok = float(after_conf) <= float(threshold)
        return conf_ok


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

    def _perceptual_delta(self) -> float:
        """
        Measure daylight-visibility delta: mean absolute RGB difference between
        baseline sign and daylight overlay, masked to the sign alpha.
        """
        base = self.sign_rgba_day.convert("RGB")
        over = self._apply_grid_overlay(self.sign_rgba_day, mode="day").convert("RGB")
        base_arr = np.array(base, dtype=np.float32)
        over_arr = np.array(over, dtype=np.float32)
        diff = np.abs(over_arr - base_arr) / 255.0
        mask = (np.array(self._sign_alpha, dtype=np.uint8) > 0).astype(np.float32)
        if mask.sum() <= 0:
            return 0.0
        diff_mean = float((diff.mean(axis=2) * mask).sum() / mask.sum())
        return diff_mean

    def _transform_sign(self, sign_rgba: Image.Image, seed: int) -> Image.Image:
        rng = np.random.default_rng(seed)
        W, H = sign_rgba.size
        strength = max(0.0, min(1.0, float(self.transform_strength)))
        out = sign_rgba.copy()

        if strength <= 0.0:
            return out

        angle = rng.uniform(-6.0 * strength, 6.0 * strength)
        shear = rng.uniform(-4.0 * strength, 4.0 * strength)
        scale = 1.0 + rng.uniform(-0.10 * strength, 0.10 * strength)
        tx = rng.uniform(-0.02 * strength * W, 0.02 * strength * W)
        ty = rng.uniform(-0.02 * strength * H, 0.02 * strength * H)

        aff = _affine_matrix(angle, shear, scale, tx, ty, W, H)
        out = out.transform((W, H), Image.AFFINE, data=aff, resample=Image.BILINEAR, fillcolor=(0, 0, 0, 0))

        if rng.random() < 0.5 * strength:
            coeffs = _random_perspective_coeffs(W, H, rng, max_shift=0.06 * strength)
            out = out.transform((W, H), Image.PERSPECTIVE, coeffs, resample=Image.BILINEAR, fillcolor=(0, 0, 0, 0))

        rgb, a = out.convert("RGB"), out.split()[-1]
        if rng.random() < 0.7 * strength:
            rgb = ImageEnhance.Brightness(rgb).enhance(float(rng.uniform(1.0 - 0.1 * strength, 1.0 + 0.1 * strength)))
        if rng.random() < 0.7 * strength:
            rgb = ImageEnhance.Contrast(rgb).enhance(float(rng.uniform(1.0 - 0.1 * strength, 1.0 + 0.1 * strength)))
        if rng.random() < 0.3 * strength:
            rgb = ImageEnhance.Color(rgb).enhance(float(rng.uniform(1.0 - 0.1 * strength, 1.0 + 0.1 * strength)))
        if rng.random() < 0.4 * strength:
            rgb = rgb.filter(ImageFilter.GaussianBlur(radius=float(rng.uniform(0.0, 0.8 * strength))))
        if rng.random() < 0.6 * strength:
            arr = np.array(rgb, dtype=np.int16)
            sigma = rng.uniform(1.0 * strength, 3.0 * strength)
            noise = rng.normal(0.0, sigma, size=arr.shape)
            arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
            rgb = Image.fromarray(arr, mode="RGB")

        return Image.merge("RGBA", (*rgb.split(), a))

    def _compose_on_bg(self, sign_rgba_t: Image.Image, place_seed: int, return_meta: bool = False):
        if return_meta:
            group, gmeta = self._compose_sign_and_pole(sign_rgba_t, return_meta=True)
        else:
            group, gmeta = self._compose_sign_and_pole(sign_rgba_t, return_meta=False), None
        img, place = self._place_group_on_background(group, self._bg_rgb, seed=place_seed)
        if not return_meta:
            return img

        sign_bbox = gmeta.get("sign_bbox", (0, 0, sign_rgba_t.width, sign_rgba_t.height))
        sx1, sy1, sx2, sy2 = [float(v) for v in sign_bbox]
        scale = float(place.get("scale", 1.0))
        x = float(place.get("x", 0.0))
        y = float(place.get("y", 0.0))
        sign_bbox_bg = (x + scale * sx1, y + scale * sy1, x + scale * sx2, y + scale * sy2)
        meta = {
            **place,
            "sign_bbox_group": sign_bbox,
            "sign_bbox_bg": sign_bbox_bg,
        }
        return img, meta

    def _compose_sign_and_pole(
        self,
        sign_rgba: Image.Image,
        pole_width_ratio: float = 0.12,
        bottom_len_factor: float = 4.0,
        clearance_px: int = 2,
        side_margin_frac: float = 0.06,
        return_meta: bool = False,
    ):
        if self.pole_rgba is None:
            if return_meta:
                return sign_rgba, {"sign_bbox": (0, 0, sign_rgba.width, sign_rgba.height)}
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
        if return_meta:
            return group, {"sign_bbox": (sx, sy, sx + SW, sy + SH)}
        return group

    def _place_group_on_background(self, group_rgba: Image.Image, bg_rgb: Image.Image, seed: int):
        rng = np.random.default_rng(seed)
        W, H = self.img_size
        bg_rgba = bg_rgb.resize((W, H), Image.BILINEAR).convert("RGBA")

        # Distance variance with a safer minimum size to avoid missed detections.
        target_w = int(rng.uniform(0.18 * W, 0.60 * W))
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
        if min_y > max_y:
            min_y = max(margin, max_y)
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


def _iou_xyxy(box_a, box_b) -> float:
    ax1, ay1, ax2, ay2 = [float(v) for v in box_a]
    bx1, by1, bx2, by2 = [float(v) for v in box_b]
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 0.0:
        return 0.0
    return float(inter / union)
