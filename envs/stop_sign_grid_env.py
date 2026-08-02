from __future__ import annotations
from typing import Tuple, Dict, Any, List, Optional, Sequence
import math
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter

import gymnasium as gym
from gymnasium import spaces

from detectors.factory import build_detector
from detectors.class_names import ClassReference, resolve_class_id
from envs.attack_objective import (
    ATTACK_MODES,
    AggregateAttackMetrics,
    AttackObjectiveConfig,
    TransformAttackMetrics,
    aggregate_metrics,
    attack_success as objective_attack_success,
    reward_terms,
    success_progress,
    summarize_detection,
)
from utils.uv_paint import UVPaint, YELLOW_GLOW


class TrafficSignGridEnv(gym.Env):
    """
    Grid-square adversarial overlay over an alpha-masked traffic sign.

    Action:
      Discrete index into valid grid cells within the sign alpha mask.
      (cell size = grid_cell_px; configurable)

    Per step:
      - Add 1 new grid cell (grid_cell_px square) to a running episode mask.
      - Duplicate cells are disallowed: invalid actions return a small penalty.
        Use action masking (MaskablePPO) to prevent duplicates entirely.
      - Render four matched variants on the same background/pole/placement/transforms:
          0) Plain (no overlay) baseline for day
          1) Plain (no overlay) baseline for UV-on
          2) Daylight overlay (pre-activation color/alpha)
          3) UV-on overlay (activated color/alpha)
      - For robustness, evaluate each variant across K matched sign-only transforms
        with identical placement and background.
      - Compute mean confidences over K runs:
          c0_day, c_day, c0_on, c_on
        Active-state suppression is drop_on = c0_on - c_on, using the matched
        clean active image rather than the daylight image.
        Secondary objective keeps day confidence high (penalize if drop exceeds tolerance).
      - Detections are localized against the known rendered sign box, so objects
        elsewhere in the background cannot count as misclassification.
      - Episode termination follows one explicit attack mode: disappearance,
        untargeted misclassification, or targeted misclassification.

    Observation:
      Cropped RGB image around the sign (daylight composite) with optional
      overlay-mask channel (H,W,3 or H,W,4) uint8.

    Masking:
      action_masks() returns a boolean mask of valid (free) cells for MaskablePPO.

    Reward (per step, normalized):
      The centralized attack objective supplies mode-specific classification and
      localization gains.  Disappearance emphasizes source suppression;
      untargeted and targeted modes emphasize a localized alternative/target
      label while optionally requiring source suppression.  The core also
      penalizes inactive-state degradation, exact painted-pixel area, steps,
      excess area, and optional perceptual change, and rewards query-efficient
      progress.  A smooth objective-progress shaping term and a small
      minimum-area joint-success bonus are then added and squashed:

          raw_total = raw_core + shaping + success_bonus
          reward    = tanh(1.2 * raw_total)    (-1, 1)

      A reported success is stricter than a positive reward: the clean matched
      day/active baselines must be eligible, the configured EOT attack rate must
      be met, the inactive/day state must be preserved when required, and the
      exact painted-pixel area must be within the configured cap.

      so PPO always sees a bounded per-step reward.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        stop_sign_image: Image.Image,
        stop_sign_uv_image: Optional[Image.Image],
        background_images: List[Image.Image],
        pole_image: Optional[Image.Image],
        yolo_weights: str = "weights/yolov8n.pt",
        img_size: Tuple[int, int] = (640, 640),
        detector_debug: bool = False,


        # Episodes
        steps_per_episode: int = 300,
        eval_K: int = 3,
        eval_K_min: Optional[int] = None,
        eval_K_max: Optional[int] = None,
        eval_K_ramp_threshold: Optional[float] = None,

        # Grid config
        grid_cell_px: int = 16,
        max_cells: Optional[int] = None,
        area_cap_frac: Optional[float] = None,
        action_indexing: str = "valid_cells",
        terminate_on_success: bool = True,

        # Paint (single pair)
        uv_paint: UVPaint = YELLOW_GLOW,
        uv_paint_list: Optional[List[UVPaint]] = None,
        use_single_color: bool = True,
        paint_action_mode: str = "fixed",
        uv_paint_palette: Optional[Sequence[UVPaint]] = None,

        # Threshold logic
        uv_drop_threshold: float = 0.75,
        success_conf_threshold: float = 0.20,
        day_tolerance: float = 0.05,
        lambda_day: float = 1.0,
        lambda_area: float = 0.3,
        area_target_frac: Optional[float] = None,
        step_cost: float = 0.0,
        step_cost_after_target: float = 0.0,
        lambda_iou: float = 0.4,
        lambda_misclass: float = 0.6,
        lambda_classification: Optional[float] = None,
        lambda_efficiency: float = 0.0,
        efficiency_eps: float = 0.02,
        lambda_perceptual: float = 0.0,
        transform_strength: float = 1.0,
        fixed_angle_deg: Optional[float] = None,
        uv_min_alpha: float = 0.08,
        min_base_conf: float = 0.20,
        cell_cover_thresh: float = 0.60,
        area_cap_penalty: float = -0.20,
        area_cap_mode: str = "soft",

        # Detector backend
        detector_type: str = "yolo",
        detector_model: Optional[str] = None,
        source_class: ClassReference = "stop sign",
        attack_mode: str = "disappearance",
        attack_target_class: Optional[ClassReference] = None,
        allowed_alternative_classes: Optional[Sequence[ClassReference]] = None,
        target_conf_threshold: float = 0.40,
        min_attack_success_rate: float = 0.80,
        min_clean_detection_rate: float = 0.80,
        localization_iou_threshold: float = 0.30,
        require_source_suppression: bool = True,
        require_day_preservation: bool = True,
        detector_instance: Optional[Any] = None,

        # Detector thresholds
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
        if self.sign_rgba_day.size != self.sign_rgba_on.size:
            raise ValueError(
                "day and active sign assets must have identical pixel dimensions"
            )
        day_alpha = np.array(self.sign_rgba_day.split()[-1], dtype=np.uint8) > 0
        on_alpha = np.array(self.sign_rgba_on.split()[-1], dtype=np.uint8) > 0
        alpha_union = int(np.logical_or(day_alpha, on_alpha).sum())
        alpha_overlap = (
            float(np.logical_and(day_alpha, on_alpha).sum()) / float(alpha_union)
            if alpha_union > 0
            else 0.0
        )
        if alpha_overlap < 0.95:
            raise ValueError(
                "day and active sign alpha masks are not registered "
                f"(mask IoU={alpha_overlap:.3f}, required >= 0.95)"
            )
        self.bg_list = [im.convert("RGB") for im in (background_images or [])]
        self.pole_rgba = None if pole_image is None else pole_image.convert("RGBA")

        self.grid_cell_px = int(grid_cell_px)
        if self.grid_cell_px <= 0:
            raise ValueError("grid_cell_px must be a positive integer")

        self.cell_cover_thresh = float(cell_cover_thresh)

        # build sign alpha and grid on construction
        self._sign_alpha = self.sign_rgba_day.split()[-1]  # L mask of octagon
        self._build_grid_index()

        # The historical policy indexes only alpha-valid cells.  That compact
        # space changes size and meaning when the sign silhouette changes, so
        # it cannot be shared by an amortized policy trained across sign types.
        # ``canonical_full_grid`` is an opt-in, fixed semantic: action i always
        # means row/column ``divmod(i, Gw)`` and invalid cells are masked out.
        indexing = str(action_indexing).strip().lower().replace("-", "_")
        indexing_aliases = {
            "valid": "valid_cells",
            "compact": "valid_cells",
            "full": "canonical_full_grid",
            "full_grid": "canonical_full_grid",
            "canonical": "canonical_full_grid",
        }
        indexing = indexing_aliases.get(indexing, indexing)
        if indexing not in ("valid_cells", "canonical_full_grid"):
            raise ValueError(
                "action_indexing must be 'valid_cells' or "
                "'canonical_full_grid'"
            )
        self.action_indexing = indexing
        self.terminate_on_success = bool(terminate_on_success)

        self.max_cells = int(max_cells) if max_cells is not None else None
        self.area_cap_frac = float(area_cap_frac) if area_cap_frac is not None else None
        self.area_cap_mode = str(area_cap_mode).lower().strip()
        if self.area_cap_mode not in ("soft", "hard"):
            raise ValueError("area_cap_mode must be 'soft' or 'hard'")
        # Pixel-exact cap checks happen before each action.  A cell-count proxy is
        # inaccurate for circular signs and partially covered edge cells.
        self._derived_max_cells = False
        if self.area_cap_frac is not None and not (0.0 < self.area_cap_frac <= 1.0):
            raise ValueError("area_cap_frac must be in (0, 1]")
        if self.area_cap_frac is not None and self.area_cap_mode == "hard":
            feasible_from_empty = self._valid_cells & (
                self._cell_pixel_areas
                <= float(self.area_cap_frac) * float(self._sign_pixel_area) + 1e-12
            )
            if not np.any(feasible_from_empty):
                raise ValueError(
                    "hard area cap is smaller than every valid canonical cell"
                )

        # UV paint pair (single or list).  ``uv_paint_list`` retains its
        # historical meaning: sample one global paint for the whole episode.
        # Joint palette mode is deliberately opt-in and uses a separate palette
        # so an old configuration cannot silently change action semantics.
        self.paint_list = list(uv_paint_list) if uv_paint_list else None
        self.paint = uv_paint
        self.use_single_color = bool(use_single_color)
        normalized_paint_mode = str(paint_action_mode).strip().lower().replace("-", "_")
        normalized_paint_mode = {
            "single": "fixed",
            "single_color": "fixed",
            "palette": "joint_palette",
            "joint": "joint_palette",
        }.get(normalized_paint_mode, normalized_paint_mode)
        if normalized_paint_mode not in ("fixed", "joint_palette"):
            raise ValueError("paint_action_mode must be 'fixed' or 'joint_palette'")
        palette = tuple(uv_paint_palette or ())
        if normalized_paint_mode == "joint_palette":
            if not palette:
                raise ValueError(
                    "uv_paint_palette must contain at least one paint in joint_palette mode"
                )
            if self.paint_list:
                raise ValueError(
                    "uv_paint_list episode randomization cannot be combined with "
                    "joint_palette actions"
                )
            if any(not isinstance(value, UVPaint) for value in palette):
                raise TypeError("every uv_paint_palette entry must be a UVPaint")
            palette_names = [str(value.name) for value in palette]
            if len(palette_names) != len(set(palette_names)):
                raise ValueError("uv_paint_palette paint names must be unique")
        elif palette:
            raise ValueError(
                "uv_paint_palette is only valid when paint_action_mode='joint_palette'"
            )
        self.paint_action_mode = normalized_paint_mode
        self.paint_palette: Tuple[UVPaint, ...] = (
            palette if self.paint_action_mode == "joint_palette" else (self.paint,)
        )
        self.paint_action_count = len(self.paint_palette)
        self.action_encoding = (
            (
                "canonical_cell_major_material_minor_v1"
                if self.action_indexing == "canonical_full_grid"
                else "valid_cell_major_material_minor_v1"
            )
            if self.paint_action_mode == "joint_palette"
            else "cell_only_v1"
        )

        # threshold / reward
        self.uv_drop_threshold = float(uv_drop_threshold)
        self.success_conf_threshold = float(success_conf_threshold)
        self.day_tolerance = float(day_tolerance)
        self.lambda_day = float(lambda_day)
        self.lambda_area = float(lambda_area)
        self.area_target_frac = float(area_target_frac) if area_target_frac is not None else None
        self.step_cost = float(step_cost)
        self.step_cost_after_target = float(step_cost_after_target)
        self.lambda_iou = float(lambda_iou)
        self.lambda_classification = float(
            lambda_misclass if lambda_classification is None else lambda_classification
        )
        # Backward-compatible metric/config key used by existing callbacks.
        self.lambda_misclass = self.lambda_classification
        self.lambda_efficiency = float(lambda_efficiency)
        self.efficiency_eps = float(efficiency_eps)
        self.lambda_perceptual = float(lambda_perceptual)
        self.transform_strength = float(transform_strength)
        self.fixed_angle_deg = (
            float(fixed_angle_deg) if fixed_angle_deg is not None else None
        )
        # Optional: enable per-episode angle evaluation (set externally).
        self.angle_eval_list: List[float] = []
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


        # Detector and explicit attack semantics.
        self.source_class = source_class
        self.attack_mode = str(attack_mode).strip().lower()
        if self.attack_mode not in ATTACK_MODES:
            raise ValueError(f"attack_mode must be one of: {', '.join(ATTACK_MODES)}")
        if self.attack_mode == "targeted_misclassification" and attack_target_class is None:
            raise ValueError(
                "attack_target_class is required for targeted_misclassification"
            )

        self.det = detector_instance or build_detector(
            detector_type=detector_type,
            detector_model=detector_model,
            yolo_weights=yolo_weights,
            device=yolo_device,
            conf=conf_thresh,
            iou=iou_thresh,
            target_class=source_class,
            debug=detector_debug,
        )
        # Detector wrappers retain ``target_id`` for the legacy confidence-only
        # API.  Amortized experiments may deliberately share one detector model
        # across tasks with different source labels, so that mutable wrapper
        # field is not a safe source-of-truth for an individual environment.
        # The structured detection path below consumes the complete class arrays;
        # resolve and retain this environment's source id independently instead.
        source_resolver = getattr(self.det, "resolve_class_id", None)
        if callable(source_resolver):
            self.source_class_id = int(
                source_resolver(source_class, role="source class")
            )
        else:
            self.source_class_id = resolve_class_id(
                getattr(self.det, "id_to_name", {}) or {},
                source_class,
                role="source class",
            )
        self.attack_target_class = attack_target_class
        self.attack_target_id: Optional[int] = None
        if attack_target_class is not None:
            resolver = getattr(self.det, "resolve_class_id", None)
            if callable(resolver):
                self.attack_target_id = int(
                    resolver(attack_target_class, role="attack target class")
                )
            else:
                self.attack_target_id = resolve_class_id(
                    getattr(self.det, "id_to_name", {}) or {},
                    attack_target_class,
                    role="attack target class",
                )
            if self.attack_target_id == self.source_class_id:
                raise ValueError("attack target class must differ from source class")

        if isinstance(allowed_alternative_classes, str):
            alternative_refs = [
                part.strip()
                for part in allowed_alternative_classes.split(",")
                if part.strip()
            ]
        else:
            alternative_refs = list(allowed_alternative_classes or [])
        alternative_ids: List[int] = []
        for class_ref in alternative_refs:
            resolver = getattr(self.det, "resolve_class_id", None)
            if callable(resolver):
                class_id = int(resolver(class_ref, role="allowed alternative class"))
            else:
                class_id = resolve_class_id(
                    getattr(self.det, "id_to_name", {}) or {},
                    class_ref,
                    role="allowed alternative class",
                )
            if class_id == self.source_class_id:
                raise ValueError("allowed alternative classes must exclude the source class")
            if class_id not in alternative_ids:
                alternative_ids.append(class_id)
        self.allowed_alternative_class_ids = (
            tuple(alternative_ids) if alternative_refs else None
        )

        self.attack_config = AttackObjectiveConfig(
            mode=self.attack_mode,
            source_conf_threshold=self.success_conf_threshold,
            target_conf_threshold=float(target_conf_threshold),
            min_success_rate=float(min_attack_success_rate),
            localization_iou_threshold=float(localization_iou_threshold),
            require_source_suppression=bool(require_source_suppression),
            allowed_alternative_class_ids=self.allowed_alternative_class_ids,
        )
        self.min_clean_detection_rate = float(min_clean_detection_rate)
        if not 0.0 <= self.min_clean_detection_rate <= 1.0:
            raise ValueError("min_clean_detection_rate must be in [0, 1]")
        self.require_day_preservation = bool(require_day_preservation)


        # action/obs spaces
        self._base_action_count = (
            self._n_valid
            if self.action_indexing == "valid_cells"
            else self.Gh * self.Gw
        )
        self.action_space = spaces.Discrete(
            self._base_action_count * self.paint_action_count
        )
        H, W = self.obs_size[1], self.obs_size[0]
        C = 4 if self.obs_include_mask else 3
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(H, W, C), dtype=np.uint8
        )

        # RNG & episodic state
        self.rng = np.random.default_rng(seed)
        self._step = 0
        self._bg_rgb = None
        self._bg_index: Optional[int] = None
        self._episode_cells: np.ndarray = None  # bool mask [Gh, Gw] of selected cells
        self._episode_paint_ids: Optional[np.ndarray] = None  # int material id [Gh, Gw]
        self._place_seed = None
        self._transform_seeds: List[int] = []
        self._baseline_c0_day_list: List[float] = []
        self._baseline_c0_on_list: List[float] = []
        self._baseline_day_metrics: List[TransformAttackMetrics] = []
        self._baseline_on_metrics: List[TransformAttackMetrics] = []
        self._last_plain_day_metrics: List[TransformAttackMetrics] = []
        self._last_plain_on_metrics: List[TransformAttackMetrics] = []
        self._last_drop_on_s = 0.0
        self._diag_saved = False
        self._detector_queries = 0
        self._detector_requests = 0

    # ----------------------------- grid build --------------------------------

    def _build_grid_index(self):
        """Precompute grid geometry and exact printable area inside the sign."""
        W, H = self.sign_rgba_day.size
        g = self.grid_cell_px
        Gw, Gh = math.ceil(W / g), math.ceil(H / g)

        signA = np.array(self._sign_alpha, dtype=np.uint8) > 0
        sign_pixel_area = int(signA.sum())
        if sign_pixel_area <= 0:
            raise ValueError("sign image must contain a non-empty alpha mask")
        valid = np.zeros((Gh, Gw), dtype=bool)
        pixel_areas = np.zeros((Gh, Gw), dtype=np.int64)
        rects: List[Tuple[int, int, int, int]] = []

        for r in range(Gh):
            for c in range(Gw):
                x0, y0 = c * g, r * g
                x1, y1 = min(W, x0 + g), min(H, y0 + g)
                cell = signA[y0:y1, x0:x1]
                cover = float(cell.mean()) if cell.size else 0.0
                valid[r, c] = (cover >= self.cell_cover_thresh)
                pixel_areas[r, c] = int(cell.sum())

                rects.append((x0, y0, x1, y1))

        self.Gw, self.Gh = Gw, Gh
        self._cell_rects = rects
        self._valid_cells = valid
        self._cell_pixel_areas = pixel_areas
        self._sign_pixel_area = sign_pixel_area
        self._valid_coords = np.argwhere(self._valid_cells)  # shape (N,2)
        self._n_valid = int(self._valid_coords.shape[0])
        if self._n_valid <= 0:
            raise ValueError(
                "grid/cell-cover settings produced no valid sign cells; reduce "
                "grid_cell_px or cell_cover_thresh"
            )


    # ----------------------------- lifecycle ---------------------------------

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            self.action_space.seed(seed)

        self._step = 0
        self._detector_queries = 0
        self._detector_requests = 0
        self._episode_cells = np.zeros((self.Gh, self.Gw), dtype=bool)
        self._episode_paint_ids = np.full((self.Gh, self.Gw), -1, dtype=np.int16)

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
        self._baseline_day_metrics = list(self._last_plain_day_metrics)
        self._baseline_on_metrics = list(self._last_plain_on_metrics)

        self._last_drop_on_s = 0.0
        self._diag_saved = False

        return np.array(obs, dtype=np.uint8), {}

    # ----------------------------- step --------------------------------------

    def step(self, action):
        self._step += 1

        # 1) Action is either a cell index (legacy fixed-paint mode) or a
        # cell-major/material-minor token (joint-palette mode).
        idx = int(action)
        idx = max(0, min(idx, int(self.action_space.n) - 1))
        cell_action, paint_index = self.decode_action(idx)
        free_mask = self._valid_cells & (~self._episode_cells)
        if not np.any(free_mask):
            # no free cells left
            terminated = True
            truncated = (self._step >= self.steps_per_episode)

            obs = self._render_observation(kind="day", use_overlay=True, transform_seed=self._transform_seeds[0])
            area_frac = self._area_frac_selected()
            cap_exceeded = self.area_cap_frac is not None and area_frac > self.area_cap_frac
            info = {
                "objective": self.attack_mode,
                "note": "no_free_cells",
                "lambda_area": float(self.lambda_area),
                "total_area_mask_frac": area_frac,
                "area_cap": float(self.area_cap_frac) if self.area_cap_frac is not None else 0.0,
                "uv_success": False,
                "attack_success": False,
                "area_cap_exceeded": cap_exceeded,
                "detector_queries": int(self._detector_queries),
            }
            return obs, -1.0, bool(terminated), bool(truncated), info

        if self.action_indexing == "valid_cells":
            pick = self._valid_coords[cell_action]
            r, c = int(pick[0]), int(pick[1])
        else:
            r, c = divmod(cell_action, self.Gw)
        if not free_mask[r, c]:
            # invalid action (duplicate or not free); should be prevented by action masking
            terminated = False
            truncated = (self._step >= self.steps_per_episode)
            obs = self._render_observation(kind="day", use_overlay=True, transform_seed=self._transform_seeds[0])
            info = {
                "objective": self.attack_mode,
                "note": "invalid_action",
                "selected_cells": int(self._episode_cells.sum()),
                "total_area_mask_frac": float(self._area_frac_selected()),
                "uv_success": False,
                "attack_success": False,
                "area_cap_exceeded": False,
                "detector_queries": int(self._detector_queries),
            }
            return obs, -0.05, bool(terminated), bool(truncated), info

        selected_cells = int(self._episode_cells.sum())
        if self.area_cap_frac is not None and self.area_cap_mode == "hard":
            next_area_frac = self._area_frac_selected() + self._cell_area_frac(r, c)
            if next_area_frac > self.area_cap_frac:
                terminated = True
                truncated = (self._step >= self.steps_per_episode)
                obs = self._render_observation(kind="day", use_overlay=True, transform_seed=self._transform_seeds[0])
                area_frac = self._area_frac_selected()
                info = {
                    "objective": self.attack_mode,
                    "note": "area_cap_exceeded",
                    "selected_cells": int(selected_cells),
                    "total_area_mask_frac": float(area_frac),
                    "area_cap": float(self.area_cap_frac),
                    "uv_success": False,
                    "attack_success": False,
                    "area_cap_exceeded": True,
                    "detector_queries": int(self._detector_queries),
                }
                return obs, float(self.area_cap_penalty), bool(terminated), bool(truncated), info

        self._episode_cells[r, c] = True
        if self._episode_paint_ids is None:
            self._episode_paint_ids = np.full((self.Gh, self.Gw), -1, dtype=np.int16)
        self._episode_paint_ids[r, c] = int(paint_index)

        area_frac = self._area_frac_selected()
        cap_exceeded = self.area_cap_frac is not None and area_frac > self.area_cap_frac

        terminated = False
        max_cells_reached = False
        if not np.any(self._valid_cells & (~self._episode_cells)):
            terminated = True
        if self.max_cells is not None:
            if int(self._episode_cells.sum()) >= self.max_cells:
                terminated = True
                max_cells_reached = True

        # 2) Evaluate overlay vs baseline
        eval_K = self._current_eval_K(self._last_drop_on_s)
        eval_seeds = self._transform_seeds[:eval_K]
        overlay_metrics = self._eval_overlay_over_K(
            eval_seeds,
            baseline_day_confidences=self._baseline_c0_day_list[:eval_K],
            baseline_on_confidences=self._baseline_c0_on_list[:eval_K],
            baseline_day_metrics=self._baseline_day_metrics[:eval_K],
            baseline_on_metrics=self._baseline_on_metrics[:eval_K],
        )
        c_day = overlay_metrics["c_day"]
        c_on = overlay_metrics["c_on"]
        mean_iou = overlay_metrics["mean_iou"]
        misclass_rate = overlay_metrics["misclass_rate"]
        mean_source_conf = overlay_metrics.get("mean_source_conf", 0.0)
        mean_top_conf = overlay_metrics.get("mean_top_conf", 0.0)
        top_class_counts = overlay_metrics.get("top_class_counts", {})
        attack_metrics: AggregateAttackMetrics = overlay_metrics["attack_metrics"]

        eligible_indices = list(overlay_metrics.get("eligible_indices", []))
        c0_day = float(np.mean([
            self._baseline_c0_day_list[i] for i in eligible_indices
        ])) if eligible_indices else 0.0
        c0_on = float(np.mean([
            self._baseline_c0_on_list[i] for i in eligible_indices
        ])) if eligible_indices else 0.0

        drop_day = float(c0_day - c_day)
        # Compare active-overlay confidence against the matched active baseline.
        # Using the daylight baseline confounds the UV source image with the patch.
        drop_on  = float(c0_on - c_on)

        drop_on_s = float(drop_on)
        self._last_drop_on_s = drop_on_s

        area_frac = self._area_frac_selected()
        cap_exceeded = self.area_cap_frac is not None and area_frac > self.area_cap_frac

        # (Baseline gating)
        if min(c0_day, c0_on) < self.min_base_conf:
            reward = -0.05
            # keep termination from max_cells if you want; I'm leaving it as-is:
            truncated = (self._step >= self.steps_per_episode)

            obs = self._render_observation(kind="day", use_overlay=True, transform_seed=self._transform_seeds[0])
            info = {
                "objective": self.attack_mode,
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
                "detector_queries": int(self._detector_queries),
            }
            return obs, float(reward), bool(terminated), bool(truncated), info

        # 3) Reward using raw UV drop (no smoothing for core objective)
        pen_day = max(0.0, drop_day - self.day_tolerance)
        area_frac = self._area_frac_selected()
        conf_thr = self.success_conf_threshold
        max_drop = max(0.0, float(c0_on - conf_thr))
        drop_blend = max(0.0, min(float(drop_on), max_drop))
        classification_gain, localization_gain = reward_terms(
            attack_metrics, self.attack_config
        )
        suppression_weight = 1.0 if self.attack_mode == "disappearance" else 0.25
        suppression_gain = suppression_weight * drop_blend
        efficiency_signal = (
            drop_blend
            if self.attack_mode == "disappearance"
            else classification_gain
        )
        eff_drop = max(0.0, float(efficiency_signal))
        eff_denom = max(float(area_frac), float(self.efficiency_eps))
        efficiency = math.log1p(eff_drop / eff_denom)
        area_target = self.area_target_frac if self.area_target_frac is not None else self.area_cap_frac
        lambda_area_used = float(self.lambda_area)
        step_cost_penalty = float(self.step_cost)
        if self.step_cost_after_target > 0.0 and area_target is not None and area_frac > float(area_target):
            excess = (float(area_frac) - float(area_target)) / max(float(area_target), 1e-6)
            step_cost_penalty += float(self.step_cost_after_target) * (1.0 + max(0.0, excess))

        excess_penalty = 0.0
        if area_target is not None and area_frac > float(area_target):
            excess = float(area_frac) - float(area_target)
            # Stronger push against exceeding the target.
            excess_penalty = (lambda_area_used * 4.5 * excess) + (lambda_area_used * (excess ** 2))

        raw_core = (
            suppression_gain
            - self.lambda_day * pen_day
            - lambda_area_used * area_frac
            - excess_penalty
            - step_cost_penalty
            + self.lambda_iou * localization_gain
            + self.lambda_classification * classification_gain
            + self.lambda_efficiency * efficiency
        )
        perceptual = self._perceptual_delta()
        raw_core -= self.lambda_perceptual * perceptual

        shaping = 0.35 * math.tanh(
            3.0 * success_progress(attack_metrics, self.attack_config)
        )
        clean_detection_rate = float(overlay_metrics.get("clean_detection_rate", 0.0))
        day_correct_rate = float(overlay_metrics.get("day_correct_rate", 0.0))
        joint = self.joint_success_components(
            attack_metrics,
            clean_detection_rate=clean_detection_rate,
            day_correct_rate=day_correct_rate,
            drop_day=drop_day,
            area_frac=area_frac,
        )
        clean_eligible = joint["clean_eligible"]
        objective_success = joint["objective_success"]
        day_preserved = joint["day_preserved"]
        within_budget = joint["within_area_budget"]
        attack_success = joint["attack_success"]
        success_bonus = (
            0.2 * ((1.0 - float(area_frac)) ** 2)
            if attack_success
            else 0.0
        )

        raw_total = raw_core + shaping + success_bonus
        if cap_exceeded and self.area_cap_mode == "soft":
            if self.area_cap_frac and self.area_cap_frac > 0:
                excess = max(0.0, (area_frac - self.area_cap_frac) / self.area_cap_frac)
                over_pen = abs(float(self.area_cap_penalty)) * (1.0 + 2.0 * excess)
            else:
                over_pen = abs(float(self.area_cap_penalty))
            raw_total = -over_pen
        reward = math.tanh(1.2 * raw_total)

        if attack_success and self.terminate_on_success:
            terminated = True

        truncated = (self._step >= self.steps_per_episode)

        # 4) Observation and preview
        obs = self._render_observation(kind="day", use_overlay=True, transform_seed=self._transform_seeds[0])
        preview_on = self._render_variant(kind="on", use_overlay=True, transform_seed=self._transform_seeds[0])

        source_id = self.source_class_id
        source_name = None
        attack_target_name = None
        id_to_name = getattr(self.det, "id_to_name", None)
        if isinstance(id_to_name, dict):
            source_name = id_to_name.get(int(source_id))
            if self.attack_target_id is not None:
                attack_target_name = id_to_name.get(int(self.attack_target_id))

        info = {
            "objective": self.attack_mode,
            "attack_mode": self.attack_mode,
            "c0_day": c0_day, "c_day": c_day,
            "c0_on": c0_on,   "c_on": c_on,
            "drop_day": drop_day, "drop_on": drop_on, "drop_on_smooth": float(drop_on_s),
            "reward_core": float(raw_core),
            "reward_efficiency": float(self.lambda_efficiency * efficiency),
            "reward_suppression": float(suppression_gain),
            "reward_classification": float(
                self.lambda_classification * classification_gain
            ),
            "reward_localization": float(self.lambda_iou * localization_gain),
            "reward_perceptual": float(-self.lambda_perceptual * perceptual),
            "reward_step_cost": float(-step_cost_penalty),
            "reward_raw_total": float(raw_total),
            "reward": float(reward),
            "lambda_area_used": float(lambda_area_used),
            "area_target_frac": float(area_target) if area_target is not None else None,
            "step_cost": float(self.step_cost),
            "step_cost_after_target": float(self.step_cost_after_target),
            "mean_iou": float(mean_iou),
            "misclass_rate": float(misclass_rate),
            "disappearance_success_rate": float(attack_metrics.disappearance_rate),
            "misclassification_success_rate": float(attack_metrics.untargeted_rate),
            "targeted_success_rate": float(attack_metrics.targeted_rate),
            "mean_source_iou": float(attack_metrics.mean_source_iou),
            "mean_alternative_iou": float(attack_metrics.mean_alternative_iou),
            "mean_attack_target_iou": float(attack_metrics.mean_attack_target_iou),
            "mean_alternative_conf": float(attack_metrics.mean_alternative_conf),
            "mean_attack_target_conf": float(attack_metrics.mean_attack_target_conf),
            "mean_alternative_margin": float(attack_metrics.mean_alternative_margin),
            "mean_target_margin": float(attack_metrics.mean_target_margin),
            "mean_source_conf": float(mean_source_conf),
            # Legacy detector-wrapper terminology: "target" here means the
            # clean/source class, not the designated attack target.
            "mean_target_conf": float(mean_source_conf),
            "mean_top_conf": float(mean_top_conf),
            "top_class_counts": dict(top_class_counts),
            "source_class_id": int(source_id),
            "source_class_name": str(source_name) if source_name is not None else str(self.source_class),
            "attack_target_id": int(self.attack_target_id) if self.attack_target_id is not None else None,
            "attack_target_name": (
                str(attack_target_name)
                if attack_target_name is not None
                else (str(self.attack_target_class) if self.attack_target_class is not None else None)
            ),
            "allowed_alternative_class_ids": (
                list(self.allowed_alternative_class_ids)
                if self.allowed_alternative_class_ids is not None
                else None
            ),
            # Legacy aliases retained for existing analysis scripts.
            "target_id": int(source_id),
            "target_name": str(source_name) if source_name is not None else str(self.source_class),
            "selected_cells": int(self._episode_cells.sum()),
            "detector_queries": int(self._detector_queries),
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
            "lambda_classification": float(self.lambda_classification),
            "lambda_efficiency": float(self.lambda_efficiency),
            "efficiency_eps": float(self.efficiency_eps),
            "lambda_perceptual": float(self.lambda_perceptual),
            "fixed_angle_deg": float(self.fixed_angle_deg) if self.fixed_angle_deg is not None else None,
            "perceptual_delta": float(perceptual),
            "paint_name": getattr(self.paint, "name", "unknown"),
            "base_conf": float(c0_day),
            "after_conf": float(c_on),
            "total_area_mask_frac": float(area_frac),
            "area_cap": float(self.area_cap_frac) if self.area_cap_frac is not None else None,
            "clean_detection_rate": float(clean_detection_rate),
            "day_correct_rate": float(day_correct_rate),
            "clean_eligible": bool(clean_eligible),
            "day_preserved": bool(day_preserved),
            "within_area_budget": bool(within_budget),
            "objective_success": bool(objective_success),
            "uv_success": bool(attack_success),
            "attack_success": bool(attack_success),
            "area_cap_exceeded": bool(cap_exceeded),
            "trace": {
                "phase": f"grid_uv_{self.attack_mode}",
                "grid_cell_px": int(self.grid_cell_px),
                "action_indexing": self.action_indexing,
                "selected_indices": self._selected_indices_list(),
                "place_seed": int(self._place_seed),
                "transform_seeds": [int(s) for s in self._transform_seeds],
                "background_index": int(self._bg_index) if self._bg_index is not None else None,
                "paint_name": getattr(self.paint, "name", "unknown"),
                "fixed_angle_deg": float(self.fixed_angle_deg) if self.fixed_angle_deg is not None else None,
            },
        }
        if self.paint_action_mode == "joint_palette":
            assignments = self._selected_material_assignments()
            info["paint_name"] = "joint_palette"
            info["paint_palette_names"] = [paint.name for paint in self.paint_palette]
            info["trace"].update(
                {
                    "paint_action_mode": self.paint_action_mode,
                    "action_encoding": self.action_encoding,
                    "paint_name": "joint_palette",
                    "paint_palette": [self._paint_descriptor(paint) for paint in self.paint_palette],
                    "material_observation_channel": {
                        "unpainted_value": 0,
                        "encoding": "round(255*(material_index+1)/palette_size)",
                        "palette_size": int(self.paint_action_count),
                    },
                    "selected_material_indices": [
                        int(row["material_index"]) for row in assignments
                    ],
                    "cell_material_assignments": assignments,
                }
            )
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
        if terminated and attack_success:
            info["composited_pil"] = preview_on
            info["overlay_pil"] = self._render_overlay_pattern(mode="on")
        # Otherwise only attach occasionally to reduce overhead
        elif diagnostic or (self._step % self.info_image_every) == 0:
            info["composited_pil"] = preview_on
            info["overlay_pil"] = self._render_overlay_pattern(mode="on")


        if (terminated or truncated) and self.angle_eval_list:
            info["angle_results"] = self._eval_angles_current(self.angle_eval_list, eval_K)
        return obs, float(reward), bool(terminated), bool(truncated), info

    def action_masks(self) -> np.ndarray:
        """
        Action mask for MaskablePPO: True where adding the cell is feasible.

        In hard-cap mode this excludes a free cell whose exact sign-pixel area
        would cross the material budget.  Masking that action is essential for
        prefix-valid callers: a returned action must add exactly one cell.
        """
        if self._episode_cells is None or self._n_valid <= 0:
            return np.ones(int(self.action_space.n), dtype=bool)
        feasible = self._valid_cells & (~self._episode_cells)
        if self.max_cells is not None and int(self._episode_cells.sum()) >= self.max_cells:
            feasible[:] = False
        if self.area_cap_frac is not None and self.area_cap_mode == "hard":
            selected_pixels = int(self._cell_pixel_areas[self._episode_cells].sum())
            maximum_pixels = float(self.area_cap_frac) * float(self._sign_pixel_area)
            feasible &= (selected_pixels + self._cell_pixel_areas) <= (
                maximum_pixels + 1e-12
            )
        if self.action_indexing == "canonical_full_grid":
            cell_mask = feasible.reshape(-1).astype(bool)
        else:
            coords = self._valid_coords
            cell_mask = feasible[coords[:, 0], coords[:, 1]].astype(bool)
        if self.paint_action_mode == "joint_palette":
            return np.repeat(cell_mask, self.paint_action_count)
        return cell_mask

    def encode_action(self, cell_action: int, material_index: int = 0) -> int:
        """Encode one base-cell action and material choice into a policy token."""

        cell = int(cell_action)
        material = int(material_index)
        if cell < 0 or cell >= int(self._base_action_count):
            raise ValueError("cell_action is outside the base action space")
        if self.paint_action_mode == "fixed":
            if material != 0:
                raise ValueError("fixed paint mode only accepts material_index=0")
            return cell
        if material < 0 or material >= int(self.paint_action_count):
            raise ValueError("material_index is outside the paint palette")
        return cell * int(self.paint_action_count) + material

    def decode_action(self, action: int) -> Tuple[int, int]:
        """Decode a policy token as ``(base_cell_action, material_index)``."""

        token = int(action)
        if token < 0 or token >= int(self.action_space.n):
            raise ValueError("action is outside the action space")
        if self.paint_action_mode == "fixed":
            return token, 0
        return divmod(token, int(self.paint_action_count))

    def next_step_detector_query_cost(self) -> int:
        """Return the exact number of detector images a valid next step uses."""

        eval_k = self._current_eval_K(self._last_drop_on_s)
        # Each transform has one inactive/day overlay and one triggered overlay.
        return 2 * int(eval_k)


    # ----------------------------- helpers -----------------------------------

    def _eval_plain_over_K(self, seeds: List[int]) -> Tuple[List[float], List[float]]:
        imgs_plain_day, imgs_plain_on = [], []
        sign_boxes_day, sign_boxes_on = [], []
        for t_seed in seeds:
            plain_day, meta_day = self._compose_on_bg(
                self._transform_sign(self.sign_rgba_day, t_seed),
                self._place_seed,
                return_meta=True,
            )
            plain_on, meta_on = self._compose_on_bg(
                self._transform_sign(self.sign_rgba_on, t_seed),
                self._place_seed,
                return_meta=True,
            )
            imgs_plain_day.append(plain_day)
            imgs_plain_on.append(plain_on)
            sign_boxes_day.append(meta_day["sign_bbox_bg"])
            sign_boxes_on.append(meta_on["sign_bbox_bg"])
        day_rows = self._summarize_detection_batch(imgs_plain_day, sign_boxes_day)
        on_rows = self._summarize_detection_batch(imgs_plain_on, sign_boxes_on)
        self._last_plain_day_metrics = list(day_rows)
        self._last_plain_on_metrics = list(on_rows)
        return (
            [float(row.source_conf) for row in day_rows],
            [float(row.source_conf) for row in on_rows],
        )

    def _summarize_detection_batch(
        self,
        images: List[Image.Image],
        sign_boxes: List[Tuple[float, float, float, float]],
    ):
        self._detector_queries += int(len(images))
        self._detector_requests += 1
        detections = self.det.infer_detections_batch(images)
        if len(detections) != len(images):
            raise RuntimeError(
                "detector returned a different number of summaries than input images"
            )
        return [
            summarize_detection(
                detection,
                sign_box,
                self.source_class_id,
                self.attack_target_id,
                self.attack_config,
            )
            for detection, sign_box in zip(detections, sign_boxes)
        ]

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

    @staticmethod
    def _paint_descriptor(paint: UVPaint) -> Dict[str, Any]:
        return {
            "name": str(paint.name),
            "day_hex": str(paint.day_hex),
            "active_hex": str(paint.active_hex),
            "translucent": bool(paint.translucent),
            "day_alpha": float(paint.day_alpha),
            "active_alpha": float(paint.active_alpha),
        }

    def _selected_material_assignments(self) -> List[Dict[str, Any]]:
        """Return canonical row-major cell/material assignments for the trace."""

        if self._episode_cells is None:
            return []
        if self._episode_paint_ids is None:
            if np.any(self._episode_cells) and self.paint_action_mode == "joint_palette":
                raise RuntimeError("selected cells are missing joint-palette assignments")
            paint_ids = np.zeros((self.Gh, self.Gw), dtype=np.int16)
        else:
            paint_ids = np.asarray(self._episode_paint_ids)
        assignments: List[Dict[str, Any]] = []
        for flat_index in self._selected_indices_list():
            row, col = divmod(int(flat_index), int(self.Gw))
            material_index = int(paint_ids[row, col])
            if material_index < 0 or material_index >= int(self.paint_action_count):
                raise RuntimeError(
                    f"selected cell {flat_index} has invalid material index "
                    f"{material_index}"
                )
            assignments.append(
                {
                    "cell_index": int(flat_index),
                    "material_index": material_index,
                    "material_name": str(self.paint_palette[material_index].name),
                }
            )
        return assignments

    def _eval_angles_current(self, angles: List[float], eval_K: int) -> List[Dict[str, Any]]:
        if not angles:
            return []
        orig_fixed = self.fixed_angle_deg
        seeds = self._transform_seeds[:eval_K]
        out: List[Dict[str, Any]] = []
        for angle in angles:
            self.fixed_angle_deg = float(angle)
            c0_day_list, c0_on_list = self._eval_plain_over_K(seeds)
            overlay = self._eval_overlay_over_K(
                seeds,
                baseline_day_confidences=c0_day_list,
                baseline_on_confidences=c0_on_list,
                baseline_day_metrics=self._last_plain_day_metrics,
                baseline_on_metrics=self._last_plain_on_metrics,
            )
            eligible = list(overlay.get("eligible_indices", []))
            c0_day = float(np.mean([c0_day_list[i] for i in eligible])) if eligible else float("nan")
            c0_on = float(np.mean([c0_on_list[i] for i in eligible])) if eligible else float("nan")
            c_day = float(overlay.get("c_day", float("nan")))
            c_on = float(overlay.get("c_on", float("nan")))
            drop_day = float(c0_day - c_day) if np.isfinite(c0_day) and np.isfinite(c_day) else float("nan")
            drop_on = float(c0_on - c_on) if np.isfinite(c0_on) and np.isfinite(c_on) else float("nan")
            area_frac = float(self._area_frac_selected())
            metrics = overlay.get("attack_metrics", AggregateAttackMetrics())
            clean_rate = float(overlay.get("clean_detection_rate", 0.0))
            day_correct_rate = float(overlay.get("day_correct_rate", 0.0))
            joint = self.joint_success_components(
                metrics,
                clean_detection_rate=clean_rate,
                day_correct_rate=day_correct_rate,
                drop_day=drop_day,
                area_frac=area_frac,
            )
            success = 1.0 if joint["attack_success"] else 0.0
            out.append(
                {
                    "angle_deg": float(angle),
                    "c0_day": c0_day,
                    "c0_on": c0_on,
                    "c_day": c_day,
                    "c_on": c_on,
                    "drop_day": drop_day,
                    "drop_on": drop_on,
                    "area_frac": area_frac,
                    "success": success,
                    "attack_mode": self.attack_mode,
                    "clean_detection_rate": clean_rate,
                    "day_correct_rate": day_correct_rate,
                    "day_preserved": bool(joint["day_preserved"]),
                    "within_area_budget": bool(joint["within_area_budget"]),
                    "disappearance_success_rate": float(metrics.disappearance_rate),
                    "misclassification_success_rate": float(metrics.untargeted_rate),
                    "targeted_success_rate": float(metrics.targeted_rate),
                }
            )
        self.fixed_angle_deg = orig_fixed
        return out

    def _area_frac_selected(self) -> float:
        if self._sign_pixel_area <= 0 or self._episode_cells is None:
            return 0.0
        selected_pixels = int(self._cell_pixel_areas[self._episode_cells].sum())
        return float(selected_pixels) / float(self._sign_pixel_area)

    def joint_success_components(
        self,
        metrics: AggregateAttackMetrics,
        *,
        clean_detection_rate: float,
        day_correct_rate: float,
        drop_day: float,
        area_frac: float,
    ) -> Dict[str, bool]:
        """Evaluate the same joint constraints for PPO and every baseline."""
        clean_eligible = float(clean_detection_rate) >= self.min_clean_detection_rate
        objective_ok = objective_attack_success(metrics, self.attack_config)
        day_preserved = bool(
            float(day_correct_rate) >= self.min_clean_detection_rate
            and float(drop_day) <= self.day_tolerance
        )
        within_budget = bool(
            self.area_cap_frac is None
            or float(area_frac) <= float(self.area_cap_frac)
        )
        success = bool(
            clean_eligible
            and objective_ok
            and within_budget
            and (day_preserved or not self.require_day_preservation)
        )
        return {
            "attack_success": success,
            "clean_eligible": clean_eligible,
            "objective_success": objective_ok,
            "day_preserved": day_preserved,
            "within_area_budget": within_budget,
        }

    def _cell_area_frac(self, row: int, col: int) -> float:
        """Return the exact fraction of sign pixels covered by one grid cell."""
        if self._sign_pixel_area <= 0:
            return 0.0
        return float(self._cell_pixel_areas[row, col]) / float(self._sign_pixel_area)


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

        # Build an auxiliary channel aligned to the sign placement.  Fixed mode
        # preserves the historical active-overlay alpha mask byte-for-byte.
        # Joint mode instead records the material ID, because the built-in
        # fluorescent paints intentionally look alike in daylight and a binary
        # mask would alias distinct policy states.
        overlay = (
            self._render_material_observation_pattern()
            if self.paint_action_mode == "joint_palette"
            else self._render_overlay_pattern(mode="on")
        )
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

    def _render_material_observation_pattern(self) -> Image.Image:
        """Render stable material-index codes in an RGBA alpha channel."""

        self._selected_material_assignments()
        size = self.sign_rgba_day.size
        codes = Image.new("L", size, 0)
        draw = ImageDraw.Draw(codes)
        palette_size = int(self.paint_action_count)
        for row, col in np.argwhere(self._episode_cells):
            material_index = int(self._episode_paint_ids[row, col])
            code = int(round(255.0 * float(material_index + 1) / float(palette_size)))
            x0, y0, x1, y1 = self._cell_rects[int(row) * self.Gw + int(col)]
            draw.rectangle([x0, y0, x1 - 1, y1 - 1], fill=code)
        codes = Image.composite(codes, Image.new("L", size, 0), self._sign_alpha)
        blank = Image.new("L", size, 0)
        return Image.merge("RGBA", (blank, blank, blank, codes))

    def set_area_cap_frac(self, value: Optional[float]) -> None:
        """
        Update area cap and derived max_cells at runtime (hard mode only).

        Args:
            value: New cap fraction (None or <=0 disables).
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

        Args:
            value: New lambda_area value.
        """
        self.lambda_area = float(value)

    def _eval_overlay_over_K(
        self,
        seeds: List[int],
        baseline_day_confidences: Optional[List[float]] = None,
        baseline_on_confidences: Optional[List[float]] = None,
        baseline_day_metrics: Optional[List[TransformAttackMetrics]] = None,
        baseline_on_metrics: Optional[List[TransformAttackMetrics]] = None,
    ) -> Dict[str, Any]:
        imgs_over_day, imgs_over_on = [], []
        sign_boxes_day, sign_boxes_on = [], []

        # Precompute overlays once per step
        over_sign_day = self._apply_grid_overlay(self.sign_rgba_day, mode="day")
        over_sign_on  = self._apply_grid_overlay(self.sign_rgba_on,  mode="on")

        for t_seed in seeds:
            over_day, meta_day = self._compose_on_bg(
                self._transform_sign(over_sign_day, t_seed),
                self._place_seed,
                return_meta=True,
            )
            over_on, meta_on = self._compose_on_bg(
                self._transform_sign(over_sign_on, t_seed),
                self._place_seed,
                return_meta=True,
            )
            imgs_over_day.append(over_day)
            imgs_over_on.append(over_on)
            sign_boxes_day.append(meta_day["sign_bbox_bg"])
            sign_boxes_on.append(meta_on["sign_bbox_bg"])

        day_rows_all = self._summarize_detection_batch(imgs_over_day, sign_boxes_day)
        on_rows_all = self._summarize_detection_batch(imgs_over_on, sign_boxes_on)
        if baseline_on_confidences is None and baseline_day_confidences is None:
            eligible_indices = list(range(len(on_rows_all)))
        else:
            eligible_indices = [
                idx
                for idx in range(len(on_rows_all))
                if (
                    baseline_on_confidences is not None
                    and idx < len(baseline_on_confidences)
                    and float(baseline_on_confidences[idx]) >= self.min_base_conf
                    and baseline_day_confidences is not None
                    and idx < len(baseline_day_confidences)
                    and float(baseline_day_confidences[idx]) >= self.min_base_conf
                    and (
                        baseline_day_metrics is None
                        or (
                            idx < len(baseline_day_metrics)
                            and baseline_day_metrics[idx].top_class == self.source_class_id
                        )
                    )
                    and (
                        baseline_on_metrics is None
                        or (
                            idx < len(baseline_on_metrics)
                            and baseline_on_metrics[idx].top_class == self.source_class_id
                        )
                    )
                )
            ]
        day_rows = [day_rows_all[idx] for idx in eligible_indices]
        on_rows = [on_rows_all[idx] for idx in eligible_indices]
        metrics = aggregate_metrics(on_rows)
        mode_iou = {
            "disappearance": metrics.mean_source_iou,
            "untargeted_misclassification": metrics.mean_alternative_iou,
            "targeted_misclassification": metrics.mean_attack_target_iou,
        }[self.attack_mode]
        return {
            "c_day": float(np.mean([row.source_conf for row in day_rows])) if day_rows else 0.0,
            "c_on": float(metrics.mean_source_conf),
            "mean_iou": float(mode_iou),
            "mean_source_iou": float(metrics.mean_source_iou),
            "mean_alternative_iou": float(metrics.mean_alternative_iou),
            "mean_attack_target_iou": float(metrics.mean_attack_target_iou),
            "misclass_rate": float(metrics.untargeted_rate),
            "disappearance_success_rate": float(metrics.disappearance_rate),
            "targeted_success_rate": float(metrics.targeted_rate),
            # Backward-compatible alias.  New consumers should use the explicit
            # source/alternative/attack-target fields below.
            "mean_target_conf": float(metrics.mean_source_conf),
            "mean_source_conf": float(metrics.mean_source_conf),
            "mean_alternative_conf": float(metrics.mean_alternative_conf),
            "mean_attack_target_conf": float(metrics.mean_attack_target_conf),
            "mean_alternative_margin": float(metrics.mean_alternative_margin),
            "mean_target_margin": float(metrics.mean_target_margin),
            "mean_top_conf": float(metrics.mean_top_conf),
            "top_class_counts": dict(metrics.top_class_counts or {}),
            "attack_metrics": metrics,
            "eligible_indices": eligible_indices,
            "eligible_transform_count": int(len(eligible_indices)),
            "total_transform_count": int(len(on_rows_all)),
            "clean_detection_rate": (
                float(len(eligible_indices)) / float(len(on_rows_all))
                if on_rows_all
                else 0.0
            ),
            "day_correct_rate": (
                float(np.mean([
                    row.top_class == self.source_class_id
                    and row.source_conf >= self.min_base_conf
                    for row in day_rows
                ]))
                if day_rows
                else 0.0
            ),
        }

    def _paint_layers(self, mode: str):
        """Yield ``(paint, selected_coordinates)`` layers for one render state."""

        if mode not in ("day", "on"):
            raise ValueError("overlay mode must be 'day' or 'on'")
        if self.paint_action_mode == "fixed":
            return [(self.paint, np.argwhere(self._episode_cells))]
        # Validate every selected cell before rendering; silently falling back to
        # palette entry zero would make saved action tokens non-reproducible.
        self._selected_material_assignments()
        return [
            (
                paint,
                np.argwhere(
                    self._episode_cells
                    & (self._episode_paint_ids == int(material_index))
                ),
            )
            for material_index, paint in enumerate(self.paint_palette)
        ]

    def _grid_layer_mask(
        self,
        size: Tuple[int, int],
        coordinates: np.ndarray,
        alpha: float,
    ) -> Image.Image:
        mask = Image.new("L", size, 0)
        mdraw = ImageDraw.Draw(mask)
        for r, c in coordinates:
            x0, y0, x1, y1 = self._cell_rects[int(r) * self.Gw + int(c)]
            mdraw.rectangle([x0, y0, x1 - 1, y1 - 1], fill=255)
        mask = Image.composite(mask, Image.new("L", size, 0), self._sign_alpha)
        if alpha < 1.0:
            arr = (np.array(mask, dtype=np.float32) * float(alpha)).astype(np.uint8)
            mask = Image.fromarray(arr)
        return mask

    def _paint_color_and_alpha(
        self, paint: UVPaint, mode: str
    ) -> Tuple[Tuple[int, int, int], float]:
        if mode == "day":
            color = paint.day_rgb
            alpha = paint.day_alpha if paint.translucent else 1.0
        else:
            color = paint.active_rgb
            alpha = paint.active_alpha if paint.translucent else 1.0
            if paint.translucent:
                alpha = max(alpha, self.uv_min_alpha)
        return color, float(alpha)

    def _apply_grid_overlay(self, sign_rgba: Image.Image, mode: str) -> Image.Image:
        rgb = sign_rgba.convert("RGB")
        source_alpha = sign_rgba.split()[-1]
        for paint, coordinates in self._paint_layers(mode):
            color, alpha = self._paint_color_and_alpha(paint, mode)
            mask = self._grid_layer_mask(rgb.size, coordinates, alpha)
            rgb.paste(color, mask=mask)
        return Image.merge("RGBA", (*rgb.split(), source_alpha))

    def _render_overlay_pattern(self, mode: str) -> Image.Image:
        """Render only the overlay pattern on a transparent background."""
        assert mode in ("day", "on")
        size = self.sign_rgba_day.size
        img = Image.new("RGBA", size, (0, 0, 0, 0))

        for paint, coordinates in self._paint_layers(mode):
            color, alpha = self._paint_color_and_alpha(paint, mode)
            mask = self._grid_layer_mask(size, coordinates, alpha)
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

        fixed_angle = self.fixed_angle_deg
        if fixed_angle is None and strength <= 0.0:
            return out

        angle = float(fixed_angle) if fixed_angle is not None else rng.uniform(-6.0 * strength, 6.0 * strength)
        shear = rng.uniform(-4.0 * strength, 4.0 * strength) if strength > 0.0 else 0.0
        scale = 1.0 + rng.uniform(-0.10 * strength, 0.10 * strength) if strength > 0.0 else 1.0
        tx = rng.uniform(-0.02 * strength * W, 0.02 * strength * W) if strength > 0.0 else 0.0
        ty = rng.uniform(-0.02 * strength * H, 0.02 * strength * H) if strength > 0.0 else 0.0

        aff = _affine_matrix(angle, shear, scale, tx, ty, W, H)
        out = out.transform((W, H), Image.AFFINE, data=aff, resample=Image.BILINEAR, fillcolor=(0, 0, 0, 0))

        if strength > 0.0 and rng.random() < 0.5 * strength:
            coeffs = _random_perspective_coeffs(W, H, rng, max_shift=0.06 * strength)
            out = out.transform((W, H), Image.PERSPECTIVE, coeffs, resample=Image.BILINEAR, fillcolor=(0, 0, 0, 0))

        rgb, a = out.convert("RGB"), out.split()[-1]
        if strength > 0.0 and rng.random() < 0.7 * strength:
            rgb = ImageEnhance.Brightness(rgb).enhance(float(rng.uniform(1.0 - 0.1 * strength, 1.0 + 0.1 * strength)))
        if strength > 0.0 and rng.random() < 0.7 * strength:
            rgb = ImageEnhance.Contrast(rgb).enhance(float(rng.uniform(1.0 - 0.1 * strength, 1.0 + 0.1 * strength)))
        if strength > 0.0 and rng.random() < 0.3 * strength:
            rgb = ImageEnhance.Color(rgb).enhance(float(rng.uniform(1.0 - 0.1 * strength, 1.0 + 0.1 * strength)))
        if strength > 0.0 and rng.random() < 0.4 * strength:
            rgb = rgb.filter(ImageFilter.GaussianBlur(radius=float(rng.uniform(0.0, 0.8 * strength))))
        if strength > 0.0 and rng.random() < 0.6 * strength:
            arr = np.array(rgb, dtype=np.int16)
            sigma = rng.uniform(1.0 * strength, 3.0 * strength)
            noise = rng.normal(0.0, sigma, size=arr.shape)
            arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
            rgb = Image.fromarray(arr)

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
        target_w = int(rng.uniform(0.30 * W, 0.50 * W))
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
            self._bg_index = idx
            return self.bg_list[idx].resize((W, H), Image.BILINEAR).convert("RGB")
        self._bg_index = None
        return Image.new("RGB", (W, H), (200, 200, 200))


# ---------------------------- helpers (module level) ----------------------------

# Backward-compatible import name used by existing checkpoints and scripts.
StopSignGridEnv = TrafficSignGridEnv

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
    B = [float(p) for uv in dst for p in uv]
    return _solve_small_linear_system(A, B)


def _solve_small_linear_system(A, B):
    """Solve the fixed 8x8 perspective system with partial pivoting.

    Avoiding a BLAS/LAPACK call here makes worker startup more robust on systems
    with incompatible MKL/OpenMP runtimes; the system is tiny and solved only
    while sampling a transform.
    """
    n = len(B)
    augmented = [
        [float(value) for value in A[row]] + [float(B[row])]
        for row in range(n)
    ]
    for column in range(n):
        pivot_row = max(
            range(column, n),
            key=lambda row: abs(augmented[row][column]),
        )
        pivot = augmented[pivot_row][column]
        if abs(pivot) < 1e-12:
            raise ValueError("degenerate perspective transform")
        if pivot_row != column:
            augmented[column], augmented[pivot_row] = (
                augmented[pivot_row],
                augmented[column],
            )
        pivot = augmented[column][column]
        augmented[column] = [value / pivot for value in augmented[column]]
        for row in range(n):
            if row == column:
                continue
            factor = augmented[row][column]
            if factor == 0.0:
                continue
            augmented[row] = [
                augmented[row][idx] - factor * augmented[column][idx]
                for idx in range(n + 1)
            ]
    return np.asarray([augmented[row][-1] for row in range(n)], dtype=np.float32)
