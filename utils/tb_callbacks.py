"""TensorBoard callbacks for overlay images and training metrics."""
import os
import numpy as np
from PIL import Image
from typing import Optional, Dict, Any

from stable_baselines3.common.callbacks import BaseCallback
from torch.utils.tensorboard import SummaryWriter


def pil_to_chw_uint8(pil: Image.Image) -> np.ndarray:
    """
    Convert PIL RGB -> CHW uint8 numpy (C,H,W).

    @param pil: Input PIL image.
    @return: CHW uint8 numpy array.
    """
    arr = np.array(pil.convert("RGB"), dtype=np.uint8)  # H,W,C
    return np.transpose(arr, (2, 0, 1))                 # C,H,W


class TensorboardOverlayCallback(BaseCallback):
    """
    Logs overlay metadata and images to TensorBoard.

    Usage:
      tb_cb = TensorboardOverlayCallback(log_dir="./runs/ppo_adversary", tag_prefix="adversary")
      # In your saver callback, when you produce `rec`:
      # tb_cb.log_overlay_record(rec, global_step=self.model.num_timesteps)
      # ...and include both callbacks in a CallbackList.
    """

    def __init__(self, log_dir: str, tag_prefix: str = "overlay", max_images: int = 100, verbose: int = 0):
        """
        @param log_dir: Base log directory.
        @param tag_prefix: TensorBoard tag prefix.
        @param max_images: Max overlay images to log.
        @param verbose: Verbosity level.
        """
        super().__init__(verbose)
        self.log_dir = os.path.abspath(log_dir)
        os.makedirs(self.log_dir, exist_ok=True)
        # keep a subdir to avoid clobbering SB3 scalar logs
        self.tb_dir = os.path.join(self.log_dir, "tb_overlays")
        os.makedirs(self.tb_dir, exist_ok=True)
        self.tag_prefix = tag_prefix
        self.max_images = int(max_images)
        self.writer: Optional[SummaryWriter] = None
        self._images_written = 0

    def _on_training_start(self) -> None:
        if self.writer is None:
            self.writer = SummaryWriter(log_dir=self.tb_dir)
            if self.verbose:
                print(f"[TB] writing overlay logs to: {self.tb_dir}")

    def _on_step(self) -> bool:
        # No-op; saver will call `log_overlay_record` explicitly.
        return True

    def set_log_dir(self, log_dir: str, tag_prefix: Optional[str] = None) -> None:
        """
        Update log directory for a new phase/run.

        @param log_dir: Base log directory.
        @param tag_prefix: Optional new TensorBoard tag prefix.
        """
        if tag_prefix is not None:
            self.tag_prefix = str(tag_prefix)
        if self.writer is not None:
            self.writer.flush()
            self.writer.close()
            self.writer = None
        self.log_dir = os.path.abspath(log_dir)
        os.makedirs(self.log_dir, exist_ok=True)
        self.tb_dir = os.path.join(self.log_dir, "tb_overlays")
        os.makedirs(self.tb_dir, exist_ok=True)
        self._images_written = 0

    def log_overlay_record(self, rec: Dict[str, Any], global_step: int) -> None:
        """
        Log one overlay record (the JSON-like dict your saver writes).

        Keys used if present:
          base_conf, after_conf (or conf_after), delta_conf
          params: {count, size_scale, alpha}
          total_area_mask_frac, area_cap
          composited_pil (PIL.Image) OR png_path (str)
        """
        if self.writer is None:
            self.writer = SummaryWriter(log_dir=self.tb_dir)

        # --- Scalars (with safe fallbacks) ---
        base_conf = float(rec.get("base_conf", rec.get("conf_base", 0.0)))
        after_conf = float(rec.get("after_conf", rec.get("conf_after", 0.0)))
        delta = float(rec.get("delta_conf", after_conf - base_conf))
        params = rec.get("params", {}) or {}
        count = float(params.get("count", 0.0))
        size_scale = float(params.get("size_scale", 0.0))
        alpha = float(params.get("alpha", 0.0))
        total_area_mask_frac = float(rec.get("total_area_mask_frac", 0.0))
        area_cap = float(rec.get("area_cap", 0.0))

        tp = self.tag_prefix
        self.writer.add_scalar(f"{tp}/base_conf", base_conf, global_step)
        self.writer.add_scalar(f"{tp}/after_conf", after_conf, global_step)
        self.writer.add_scalar(f"{tp}/delta_conf", delta, global_step)
        self.writer.add_scalar(f"{tp}/count", count, global_step)
        self.writer.add_scalar(f"{tp}/size_scale", size_scale, global_step)
        self.writer.add_scalar(f"{tp}/alpha", alpha, global_step)
        self.writer.add_scalar(f"{tp}/total_patch_area_frac", total_area_mask_frac, global_step)
        self.writer.add_scalar(f"{tp}/area_cap", area_cap, global_step)

        # --- Optional image ---
        pil_img: Optional[Image.Image] = None
        img_field = rec.get("composited_pil")
        if isinstance(img_field, Image.Image):
            pil_img = img_field
        else:
            png_path = rec.get("png_path")
            if isinstance(png_path, str) and os.path.isfile(png_path):
                try:
                    pil_img = Image.open(png_path).convert("RGB")
                except Exception:
                    pil_img = None

        if pil_img is not None and self._images_written < self.max_images:
            try:
                chw = pil_to_chw_uint8(pil_img)
                self.writer.add_image(f"{tp}/overlay_img", chw, global_step, dataformats="CHW")
                self._images_written += 1
            except Exception as e:
                if self.verbose:
                    print("[TB] image log failed:", e)

        self.writer.flush()

    def _on_training_end(self) -> None:
        if self.writer is not None:
            self.writer.flush()
            self.writer.close()
            if self.verbose:
                print("[TB] overlay writer closed")
            self.writer = None
# Episode summary metrics callback
from typing import List
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from torch.utils.tensorboard import SummaryWriter
import os


class EpisodeMetricsCallback(BaseCallback):
    """
    Logs episode-level scalars:
      - episode/area_frac_final
      - episode/length_steps
      - episode/drop_on_final, episode/drop_on_smooth_final
      - episode/base_conf_final, episode/after_conf_final
      - episode/reward_final, episode/selected_cells_final
      - episode/eval_K_used_final, episode/uv_success_final, episode/area_cap_exceeded_final
      - episode/mean_iou_final, episode/misclass_rate_final
      - episode/reward_core_final, episode/reward_raw_total_final
      - episode/reward_efficiency_final, episode/reward_perceptual_final, episode/reward_step_cost_final
      - episode/lambda_area_used_final, episode/lambda_area_dyn_final
      - episode/area_target_frac_final, episode/area_lagrange_lr_final

    X-axis is episode index (so you can see improvement run-to-run).
    Also logs *_vs_timesteps variants so you can align with training time.
    """

    def __init__(self, log_dir: str, verbose: int = 0):
        """
        @param log_dir: Base log directory.
        @param verbose: Verbosity level.
        """
        super().__init__(verbose)
        self.log_dir = os.path.abspath(log_dir)
        os.makedirs(self.log_dir, exist_ok=True)
        self.tb_dir = os.path.join(self.log_dir, "tb_episode_metrics")
        os.makedirs(self.tb_dir, exist_ok=True)

        self.writer: SummaryWriter | None = None
        self._ep_count = 0
        self._ep_len: List[int] = []

    def _on_training_start(self) -> None:
        if self.writer is None:
            self.writer = SummaryWriter(log_dir=self.tb_dir)

        # initialize per-env counters
        n_envs = getattr(self.training_env, "num_envs", 1)
        self._ep_len = [0 for _ in range(int(n_envs))]
        self._last_valid = [
            {
                "area": None,
                "drop_on": None,
                "drop_on_s": None,
                "base_conf": None,
                "after_conf": None,
                "reward": None,
                "reward_core": None,
                "reward_raw_total": None,
                "reward_efficiency": None,
                "reward_perceptual": None,
                "reward_step_cost": None,
                "selected_cells": None,
                "eval_k": None,
                "uv_success": None,
                "attack_success": None,
                "area_cap_exceeded": None,
                "mean_iou": None,
                "misclass_rate": None,
                "lambda_area_used": None,
                "lambda_area_dyn": None,
                "area_target_frac": None,
                "area_lagrange_lr": None,
            }
            for _ in range(int(n_envs))
        ]

        if self.verbose:
            print(f"[TB] episode metrics -> {self.tb_dir} (n_envs={n_envs})")

    def set_log_dir(self, log_dir: str) -> None:
        """
        Update log directory for a new phase/run.

        @param log_dir: Base log directory.
        """
        if self.writer is not None:
            self.writer.flush()
            self.writer.close()
            self.writer = None
        self.log_dir = os.path.abspath(log_dir)
        os.makedirs(self.log_dir, exist_ok=True)
        self.tb_dir = os.path.join(self.log_dir, "tb_episode_metrics")
        os.makedirs(self.tb_dir, exist_ok=True)
        self._ep_count = 0
        self._ep_len = []
        self._last_valid = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", None)
        dones = self.locals.get("dones", None)

        if infos is None or dones is None:
            return True

        # dones can be list/np array
        for env_idx, (done, info) in enumerate(zip(dones, infos)):
            # increment length counter for this env each step
            self._ep_len[env_idx] += 1

            if isinstance(info, dict):
                last = self._last_valid[env_idx]
                for key, alt_key in (
                    ("area", "total_area_mask_frac"),
                    ("drop_on", "drop_on"),
                    ("drop_on_s", "drop_on_smooth"),
                    ("base_conf", "base_conf"),
                    ("after_conf", "after_conf"),
                    ("reward", "reward"),
                    ("reward_core", "reward_core"),
                    ("reward_raw_total", "reward_raw_total"),
                    ("reward_efficiency", "reward_efficiency"),
                    ("reward_perceptual", "reward_perceptual"),
                    ("reward_step_cost", "reward_step_cost"),
                    ("selected_cells", "selected_cells"),
                    ("eval_k", "eval_K_used"),
                    ("uv_success", "uv_success"),
                    ("attack_success", "attack_success"),
                    ("area_cap_exceeded", "area_cap_exceeded"),
                    ("mean_iou", "mean_iou"),
                    ("misclass_rate", "misclass_rate"),
                    ("lambda_area_used", "lambda_area_used"),
                    ("lambda_area_dyn", "lambda_area_dyn"),
                    ("area_target_frac", "area_target_frac"),
                    ("area_lagrange_lr", "area_lagrange_lr"),
                ):
                    val = info.get(alt_key, None)
                    if isinstance(val, (int, float)) and not np.isnan(val):
                        last[key] = val

            if not done:
                continue

            self._ep_count += 1
            ep_len = int(self._ep_len[env_idx])
            self._ep_len[env_idx] = 0  # reset for next episode

            # pull key metrics from final info for this episode
            area = None
            drop_on = None
            drop_on_s = None
            base_conf = None
            after_conf = None
            reward = None
            reward_core = None
            reward_raw_total = None
            reward_efficiency = None
            reward_perceptual = None
            reward_step_cost = None
            selected_cells = None
            eval_k = None
            uv_success = None
            attack_success = None
            area_cap_exceeded = None
            mean_iou = None
            misclass_rate = None
            lambda_area_used = None
            lambda_area_dyn = None
            area_target_frac = None
            area_lagrange_lr = None
            if isinstance(info, dict):
                area = info.get("total_area_mask_frac", None)
                drop_on = info.get("drop_on", None)
                drop_on_s = info.get("drop_on_smooth", None)
                base_conf = info.get("base_conf", info.get("c0_on", None))
                after_conf = info.get("after_conf", info.get("c_on", None))
                reward = info.get("reward", None)
                reward_core = info.get("reward_core", None)
                reward_raw_total = info.get("reward_raw_total", None)
                reward_efficiency = info.get("reward_efficiency", None)
                reward_perceptual = info.get("reward_perceptual", None)
                reward_step_cost = info.get("reward_step_cost", None)
                selected_cells = info.get("selected_cells", None)
                eval_k = info.get("eval_K_used", None)
                uv_success = info.get("uv_success", None)
                attack_success = info.get("attack_success", None)
                area_cap_exceeded = info.get("area_cap_exceeded", None)
                mean_iou = info.get("mean_iou", None)
                misclass_rate = info.get("misclass_rate", None)
                lambda_area_used = info.get("lambda_area_used", None)
                lambda_area_dyn = info.get("lambda_area_dyn", None)
                area_target_frac = info.get("area_target_frac", None)
                area_lagrange_lr = info.get("area_lagrange_lr", None)

            # Fallback: use last valid per-env values, then NaN if still missing.
            last = self._last_valid[env_idx] if env_idx < len(self._last_valid) else {}
            if area is None:
                area = last.get("area", None)
            if drop_on is None:
                drop_on = last.get("drop_on", None)
            if drop_on_s is None:
                drop_on_s = last.get("drop_on_s", None)
            if base_conf is None:
                base_conf = last.get("base_conf", None)
            if after_conf is None:
                after_conf = last.get("after_conf", None)
            if reward is None:
                reward = last.get("reward", None)
            if reward_core is None:
                reward_core = last.get("reward_core", None)
            if reward_raw_total is None:
                reward_raw_total = last.get("reward_raw_total", None)
            if reward_efficiency is None:
                reward_efficiency = last.get("reward_efficiency", None)
            if reward_perceptual is None:
                reward_perceptual = last.get("reward_perceptual", None)
            if reward_step_cost is None:
                reward_step_cost = last.get("reward_step_cost", None)
            if selected_cells is None:
                selected_cells = last.get("selected_cells", None)
            if eval_k is None:
                eval_k = last.get("eval_k", None)
            if uv_success is None:
                uv_success = last.get("uv_success", None)
            if attack_success is None:
                attack_success = last.get("attack_success", None)
            if area_cap_exceeded is None:
                area_cap_exceeded = last.get("area_cap_exceeded", None)
            if mean_iou is None:
                mean_iou = last.get("mean_iou", None)
            if misclass_rate is None:
                misclass_rate = last.get("misclass_rate", None)
            if lambda_area_used is None:
                lambda_area_used = last.get("lambda_area_used", None)
            if lambda_area_dyn is None:
                lambda_area_dyn = last.get("lambda_area_dyn", None)
            if area_target_frac is None:
                area_target_frac = last.get("area_target_frac", None)
            if area_lagrange_lr is None:
                area_lagrange_lr = last.get("area_lagrange_lr", None)

            # Final fallback: NaN so TB shows gaps instead of crashing.
            area_val = float(area) if area is not None else float("nan")
            drop_on_val = float(drop_on) if drop_on is not None else float("nan")
            drop_on_s_val = float(drop_on_s) if drop_on_s is not None else float("nan")
            base_conf_val = float(base_conf) if base_conf is not None else float("nan")
            after_conf_val = float(after_conf) if after_conf is not None else float("nan")
            reward_val = float(reward) if reward is not None else float("nan")
            reward_core_val = float(reward_core) if reward_core is not None else float("nan")
            reward_raw_total_val = float(reward_raw_total) if reward_raw_total is not None else float("nan")
            reward_efficiency_val = float(reward_efficiency) if reward_efficiency is not None else float("nan")
            reward_perceptual_val = float(reward_perceptual) if reward_perceptual is not None else float("nan")
            reward_step_cost_val = float(reward_step_cost) if reward_step_cost is not None else float("nan")
            selected_cells_val = float(selected_cells) if selected_cells is not None else float("nan")
            eval_k_val = float(eval_k) if eval_k is not None else float("nan")
            uv_success_val = float(uv_success) if uv_success is not None else float("nan")
            attack_success_val = float(attack_success) if attack_success is not None else float("nan")
            area_cap_exceeded_val = float(area_cap_exceeded) if area_cap_exceeded is not None else float("nan")
            mean_iou_val = float(mean_iou) if mean_iou is not None else float("nan")
            misclass_rate_val = float(misclass_rate) if misclass_rate is not None else float("nan")
            lambda_area_used_val = float(lambda_area_used) if lambda_area_used is not None else float("nan")
            lambda_area_dyn_val = float(lambda_area_dyn) if lambda_area_dyn is not None else float("nan")
            area_target_frac_val = float(area_target_frac) if area_target_frac is not None else float("nan")
            area_lagrange_lr_val = float(area_lagrange_lr) if area_lagrange_lr is not None else float("nan")

            # log vs EPISODE INDEX (best for tracking improvement)
            if self.writer is not None:
                self.writer.add_scalar("episode/length_steps", ep_len, self._ep_count)
                self.writer.add_scalar("episode/area_frac_final", area_val, self._ep_count)
                self.writer.add_scalar("episode/drop_on_final", drop_on_val, self._ep_count)
                self.writer.add_scalar("episode/drop_on_smooth_final", drop_on_s_val, self._ep_count)
                self.writer.add_scalar("episode/base_conf_final", base_conf_val, self._ep_count)
                self.writer.add_scalar("episode/after_conf_final", after_conf_val, self._ep_count)
                self.writer.add_scalar("episode/reward_final", reward_val, self._ep_count)
                self.writer.add_scalar("episode/reward_core_final", reward_core_val, self._ep_count)
                self.writer.add_scalar("episode/reward_raw_total_final", reward_raw_total_val, self._ep_count)
                self.writer.add_scalar("episode/reward_efficiency_final", reward_efficiency_val, self._ep_count)
                self.writer.add_scalar("episode/reward_perceptual_final", reward_perceptual_val, self._ep_count)
                self.writer.add_scalar("episode/reward_step_cost_final", reward_step_cost_val, self._ep_count)
                self.writer.add_scalar("episode/selected_cells_final", selected_cells_val, self._ep_count)
                self.writer.add_scalar("episode/eval_K_used_final", eval_k_val, self._ep_count)
                self.writer.add_scalar("episode/uv_success_final", uv_success_val, self._ep_count)
                self.writer.add_scalar("episode/attack_success_final", attack_success_val, self._ep_count)
                self.writer.add_scalar("episode/area_cap_exceeded_final", area_cap_exceeded_val, self._ep_count)
                self.writer.add_scalar("episode/mean_iou_final", mean_iou_val, self._ep_count)
                self.writer.add_scalar("episode/misclass_rate_final", misclass_rate_val, self._ep_count)
                self.writer.add_scalar("episode/lambda_area_used_final", lambda_area_used_val, self._ep_count)
                self.writer.add_scalar("episode/lambda_area_dyn_final", lambda_area_dyn_val, self._ep_count)
                self.writer.add_scalar("episode/area_target_frac_final", area_target_frac_val, self._ep_count)
                self.writer.add_scalar("episode/area_lagrange_lr_final", area_lagrange_lr_val, self._ep_count)

                # also log vs global timesteps (sometimes useful)
                self.writer.add_scalar("episode/length_steps_vs_timesteps", ep_len, self.num_timesteps)
                self.writer.add_scalar("episode/area_frac_final_vs_timesteps", area_val, self.num_timesteps)
                self.writer.add_scalar("episode/drop_on_final_vs_timesteps", drop_on_val, self.num_timesteps)
                self.writer.add_scalar("episode/drop_on_smooth_final_vs_timesteps", drop_on_s_val, self.num_timesteps)
                self.writer.add_scalar("episode/base_conf_final_vs_timesteps", base_conf_val, self.num_timesteps)
                self.writer.add_scalar("episode/after_conf_final_vs_timesteps", after_conf_val, self.num_timesteps)
                self.writer.add_scalar("episode/reward_final_vs_timesteps", reward_val, self.num_timesteps)
                self.writer.add_scalar("episode/reward_core_final_vs_timesteps", reward_core_val, self.num_timesteps)
                self.writer.add_scalar("episode/reward_raw_total_final_vs_timesteps", reward_raw_total_val, self.num_timesteps)
                self.writer.add_scalar("episode/reward_efficiency_final_vs_timesteps", reward_efficiency_val, self.num_timesteps)
                self.writer.add_scalar("episode/reward_perceptual_final_vs_timesteps", reward_perceptual_val, self.num_timesteps)
                self.writer.add_scalar("episode/reward_step_cost_final_vs_timesteps", reward_step_cost_val, self.num_timesteps)
                self.writer.add_scalar("episode/selected_cells_final_vs_timesteps", selected_cells_val, self.num_timesteps)
                self.writer.add_scalar("episode/eval_K_used_final_vs_timesteps", eval_k_val, self.num_timesteps)
                self.writer.add_scalar("episode/uv_success_final_vs_timesteps", uv_success_val, self.num_timesteps)
                self.writer.add_scalar("episode/attack_success_final_vs_timesteps", attack_success_val, self.num_timesteps)
                self.writer.add_scalar("episode/area_cap_exceeded_final_vs_timesteps", area_cap_exceeded_val, self.num_timesteps)
                self.writer.add_scalar("episode/mean_iou_final_vs_timesteps", mean_iou_val, self.num_timesteps)
                self.writer.add_scalar("episode/misclass_rate_final_vs_timesteps", misclass_rate_val, self.num_timesteps)
                self.writer.add_scalar("episode/lambda_area_used_final_vs_timesteps", lambda_area_used_val, self.num_timesteps)
                self.writer.add_scalar("episode/lambda_area_dyn_final_vs_timesteps", lambda_area_dyn_val, self.num_timesteps)
                self.writer.add_scalar("episode/area_target_frac_final_vs_timesteps", area_target_frac_val, self.num_timesteps)
                self.writer.add_scalar("episode/area_lagrange_lr_final_vs_timesteps", area_lagrange_lr_val, self.num_timesteps)

                self.writer.flush()

        return True

    def _on_training_end(self) -> None:
        if self.writer is not None:
            self.writer.flush()
            self.writer.close()
            self.writer = None


class StepMetricsCallback(BaseCallback):
    """
    Logs step-level metrics to a lightweight ndjson file and TB scalars.
    Keeps a rolling window if keep_last_n > 0.
    Includes reward components and adaptive area weights when available.
    """

    def __init__(
        self,
        log_dir: str,
        every_n_steps: int = 1,
        keep_last_n: int = 1000,
        log_every_500: int = 500,
        verbose: int = 0,
    ):
        """
        @param log_dir: Base log directory.
        @param every_n_steps: Log cadence in steps.
        @param keep_last_n: Keep last N rows (0 = append forever).
        @param log_every_500: Secondary cadence for snapshot logging.
        @param verbose: Verbosity level.
        """
        super().__init__(verbose)
        self.log_dir = os.path.abspath(log_dir)
        os.makedirs(self.log_dir, exist_ok=True)
        self.tb_dir = os.path.join(self.log_dir, "tb_step_metrics")
        os.makedirs(self.tb_dir, exist_ok=True)
        self.every_n_steps = max(int(every_n_steps), 1)
        self.keep_last_n = max(int(keep_last_n), 0)
        self.log_every_500 = max(int(log_every_500), 0)
        self.writer: SummaryWriter | None = None
        self._last_logged = 0
        self._ndjson_path = os.path.join(self.tb_dir, "step_metrics.ndjson")
        self._ndjson_500_path = os.path.join(self.tb_dir, "step_metrics_500.ndjson")
        self._last_logged_500 = 0
        self._ring = []

    def _on_training_start(self) -> None:
        if self.writer is None:
            self.writer = SummaryWriter(log_dir=self.tb_dir)
        if self.verbose:
            print(f"[TB] step metrics -> {self.tb_dir} (every {self.every_n_steps} steps)")

    def set_log_dir(self, log_dir: str) -> None:
        """
        Update log directory for a new phase/run.

        @param log_dir: Base log directory.
        """
        if self.writer is not None:
            self.writer.flush()
            self.writer.close()
            self.writer = None
        self.log_dir = os.path.abspath(log_dir)
        os.makedirs(self.log_dir, exist_ok=True)
        self.tb_dir = os.path.join(self.log_dir, "tb_step_metrics")
        os.makedirs(self.tb_dir, exist_ok=True)
        self._ndjson_path = os.path.join(self.tb_dir, "step_metrics.ndjson")
        self._ndjson_500_path = os.path.join(self.tb_dir, "step_metrics_500.ndjson")
        self._last_logged = 0
        self._last_logged_500 = 0
        self._ring = []

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_logged < self.every_n_steps:
            return True

        infos = self.locals.get("infos", None)
        if not infos:
            return True

        info = None
        for it in infos:
            if isinstance(it, dict):
                info = it
                break
        if info is None:
            return True

        base_conf = info.get("base_conf", info.get("c0_on", None))
        after_conf = info.get("after_conf", info.get("c_on", None))
        delta_conf = info.get("delta_conf", None)
        if delta_conf is None and (base_conf is not None and after_conf is not None):
            delta_conf = float(after_conf) - float(base_conf)

        area_frac = info.get("total_area_mask_frac", None)
        drop_on = info.get("drop_on", None)
        drop_on_s = info.get("drop_on_smooth", None)
        mean_iou = info.get("mean_iou", None)
        misclass_rate = info.get("misclass_rate", None)
        mean_top_conf = info.get("mean_top_conf", None)
        mean_target_conf = info.get("mean_target_conf", None)
        eval_k = info.get("eval_K_used", None)
        top_class_counts = info.get("top_class_counts", None)
        reward_core = info.get("reward_core", None)
        reward_raw_total = info.get("reward_raw_total", None)
        reward_efficiency = info.get("reward_efficiency", None)
        reward_perceptual = info.get("reward_perceptual", None)
        reward_step_cost = info.get("reward_step_cost", None)
        lambda_area_used = info.get("lambda_area_used", None)
        lambda_area_dyn = info.get("lambda_area_dyn", None)
        area_target_frac = info.get("area_target_frac", None)
        area_lagrange_lr = info.get("area_lagrange_lr", None)

        row = {
            "step": int(self.num_timesteps),
            "base_conf": float(base_conf) if base_conf is not None else None,
            "after_conf": float(after_conf) if after_conf is not None else None,
            "delta_conf": float(delta_conf) if delta_conf is not None else None,
            "area_frac": float(area_frac) if area_frac is not None else None,
            "drop_on": float(drop_on) if drop_on is not None else None,
            "drop_on_smooth": float(drop_on_s) if drop_on_s is not None else None,
            "mean_iou": float(mean_iou) if mean_iou is not None else None,
            "misclass_rate": float(misclass_rate) if misclass_rate is not None else None,
            "mean_top_conf": float(mean_top_conf) if mean_top_conf is not None else None,
            "mean_target_conf": float(mean_target_conf) if mean_target_conf is not None else None,
            "eval_K_used": int(eval_k) if eval_k is not None else None,
            "top_class_counts": top_class_counts if isinstance(top_class_counts, dict) else None,
            "reward_core": float(reward_core) if reward_core is not None else None,
            "reward_raw_total": float(reward_raw_total) if reward_raw_total is not None else None,
            "reward_efficiency": float(reward_efficiency) if reward_efficiency is not None else None,
            "reward_perceptual": float(reward_perceptual) if reward_perceptual is not None else None,
            "reward_step_cost": float(reward_step_cost) if reward_step_cost is not None else None,
            "lambda_area_used": float(lambda_area_used) if lambda_area_used is not None else None,
            "lambda_area_dyn": float(lambda_area_dyn) if lambda_area_dyn is not None else None,
            "area_target_frac": float(area_target_frac) if area_target_frac is not None else None,
            "area_lagrange_lr": float(area_lagrange_lr) if area_lagrange_lr is not None else None,
        }

        if self.writer is not None:
            if row["after_conf"] is not None:
                self.writer.add_scalar("step_range/after_conf", row["after_conf"], self.num_timesteps)
            if row["delta_conf"] is not None:
                self.writer.add_scalar("step_range/delta_conf", row["delta_conf"], self.num_timesteps)
            if row["area_frac"] is not None:
                self.writer.add_scalar("step_range/area_frac", row["area_frac"], self.num_timesteps)
            if row["drop_on"] is not None:
                self.writer.add_scalar("step_range/drop_on", row["drop_on"], self.num_timesteps)
            if row["drop_on_smooth"] is not None:
                self.writer.add_scalar("step_range/drop_on_smooth", row["drop_on_smooth"], self.num_timesteps)
            if row["mean_iou"] is not None:
                self.writer.add_scalar("step_range/mean_iou", row["mean_iou"], self.num_timesteps)
            if row["misclass_rate"] is not None:
                self.writer.add_scalar("step_range/misclass_rate", row["misclass_rate"], self.num_timesteps)
            if row["mean_top_conf"] is not None:
                self.writer.add_scalar("step_range/mean_top_conf", row["mean_top_conf"], self.num_timesteps)
            if row["mean_target_conf"] is not None:
                self.writer.add_scalar("step_range/mean_target_conf", row["mean_target_conf"], self.num_timesteps)
            if row["reward_core"] is not None:
                self.writer.add_scalar("step_range/reward_core", row["reward_core"], self.num_timesteps)
            if row["reward_raw_total"] is not None:
                self.writer.add_scalar("step_range/reward_raw_total", row["reward_raw_total"], self.num_timesteps)
            if row["reward_efficiency"] is not None:
                self.writer.add_scalar("step_range/reward_efficiency", row["reward_efficiency"], self.num_timesteps)
            if row["reward_perceptual"] is not None:
                self.writer.add_scalar("step_range/reward_perceptual", row["reward_perceptual"], self.num_timesteps)
            if row["reward_step_cost"] is not None:
                self.writer.add_scalar("step_range/reward_step_cost", row["reward_step_cost"], self.num_timesteps)
            if row["lambda_area_used"] is not None:
                self.writer.add_scalar("step_range/lambda_area_used", row["lambda_area_used"], self.num_timesteps)
            if row["lambda_area_dyn"] is not None:
                self.writer.add_scalar("step_range/lambda_area_dyn", row["lambda_area_dyn"], self.num_timesteps)
            if row["area_target_frac"] is not None:
                self.writer.add_scalar("step_range/area_target_frac", row["area_target_frac"], self.num_timesteps)
            if row["area_lagrange_lr"] is not None:
                self.writer.add_scalar("step_range/area_lagrange_lr", row["area_lagrange_lr"], self.num_timesteps)
            self.writer.flush()

        try:
            import json
            if self.keep_last_n > 0:
                self._ring.append(row)
                if len(self._ring) > self.keep_last_n:
                    self._ring = self._ring[-self.keep_last_n :]
                with open(self._ndjson_path, "w", encoding="utf-8") as f:
                    for r in self._ring:
                        f.write(json.dumps(r, ensure_ascii=False) + "\n")
            else:
                with open(self._ndjson_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
        except Exception:
            pass

        self._last_logged = int(self.num_timesteps)

        if self.log_every_500 > 0 and (self.num_timesteps - self._last_logged_500) >= self.log_every_500:
            if self.writer is not None:
                if row["after_conf"] is not None:
                    self.writer.add_scalar("step_500/after_conf", row["after_conf"], self.num_timesteps)
                if row["delta_conf"] is not None:
                    self.writer.add_scalar("step_500/delta_conf", row["delta_conf"], self.num_timesteps)
                self.writer.flush()
            try:
                import json
                with open(self._ndjson_500_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            except Exception:
                pass
            self._last_logged_500 = int(self.num_timesteps)

        return True

    def _on_training_end(self) -> None:
        if self.writer is not None:
            self.writer.flush()
            self.writer.close()
            self.writer = None
