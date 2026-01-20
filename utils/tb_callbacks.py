# utils/tb_callbacks.py
import os
import numpy as np
from PIL import Image
from typing import Optional, Dict, Any

from stable_baselines3.common.callbacks import BaseCallback
from torch.utils.tensorboard import SummaryWriter


def pil_to_chw_uint8(pil: Image.Image) -> np.ndarray:
    """Convert PIL RGB -> CHW uint8 numpy (C,H,W)."""
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
# --- NEW: Episode summary metrics callback ---
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

    X-axis is episode index (so you can see improvement run-to-run).
    Also logs *_vs_timesteps variants so you can align with training time.
    """

    def __init__(self, log_dir: str, verbose: int = 0):
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

        if self.verbose:
            print(f"[TB] episode metrics -> {self.tb_dir} (n_envs={n_envs})")

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", None)
        dones = self.locals.get("dones", None)

        if infos is None or dones is None:
            return True

        # dones can be list/np array
        for env_idx, (done, info) in enumerate(zip(dones, infos)):
            # increment length counter for this env each step
            self._ep_len[env_idx] += 1

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
            selected_cells = None
            eval_k = None
            uv_success = None
            area_cap_exceeded = None
            if isinstance(info, dict):
                area = info.get("total_area_mask_frac", None)
                drop_on = info.get("drop_on", None)
                drop_on_s = info.get("drop_on_smooth", None)
                base_conf = info.get("base_conf", info.get("c0_on", None))
                after_conf = info.get("after_conf", info.get("c_on", None))
                reward = info.get("reward", None)
                selected_cells = info.get("selected_cells", None)
                eval_k = info.get("eval_K_used", None)
                uv_success = info.get("uv_success", None)
                area_cap_exceeded = info.get("area_cap_exceeded", None)

            # Fallback: if not present, store NaN (so TB shows gaps instead of crashing)
            area_val = float(area) if area is not None else float("nan")
            drop_on_val = float(drop_on) if drop_on is not None else float("nan")
            drop_on_s_val = float(drop_on_s) if drop_on_s is not None else float("nan")
            base_conf_val = float(base_conf) if base_conf is not None else float("nan")
            after_conf_val = float(after_conf) if after_conf is not None else float("nan")
            reward_val = float(reward) if reward is not None else float("nan")
            selected_cells_val = float(selected_cells) if selected_cells is not None else float("nan")
            eval_k_val = float(eval_k) if eval_k is not None else float("nan")
            uv_success_val = float(uv_success) if uv_success is not None else float("nan")
            area_cap_exceeded_val = float(area_cap_exceeded) if area_cap_exceeded is not None else float("nan")

            # log vs EPISODE INDEX (best for “is it improving?”)
            if self.writer is not None:
                self.writer.add_scalar("episode/length_steps", ep_len, self._ep_count)
                self.writer.add_scalar("episode/area_frac_final", area_val, self._ep_count)
                self.writer.add_scalar("episode/drop_on_final", drop_on_val, self._ep_count)
                self.writer.add_scalar("episode/drop_on_smooth_final", drop_on_s_val, self._ep_count)
                self.writer.add_scalar("episode/base_conf_final", base_conf_val, self._ep_count)
                self.writer.add_scalar("episode/after_conf_final", after_conf_val, self._ep_count)
                self.writer.add_scalar("episode/reward_final", reward_val, self._ep_count)
                self.writer.add_scalar("episode/selected_cells_final", selected_cells_val, self._ep_count)
                self.writer.add_scalar("episode/eval_K_used_final", eval_k_val, self._ep_count)
                self.writer.add_scalar("episode/uv_success_final", uv_success_val, self._ep_count)
                self.writer.add_scalar("episode/area_cap_exceeded_final", area_cap_exceeded_val, self._ep_count)

                # also log vs global timesteps (sometimes useful)
                self.writer.add_scalar("episode/length_steps_vs_timesteps", ep_len, self.num_timesteps)
                self.writer.add_scalar("episode/area_frac_final_vs_timesteps", area_val, self.num_timesteps)
                self.writer.add_scalar("episode/drop_on_final_vs_timesteps", drop_on_val, self.num_timesteps)
                self.writer.add_scalar("episode/drop_on_smooth_final_vs_timesteps", drop_on_s_val, self.num_timesteps)
                self.writer.add_scalar("episode/base_conf_final_vs_timesteps", base_conf_val, self.num_timesteps)
                self.writer.add_scalar("episode/after_conf_final_vs_timesteps", after_conf_val, self.num_timesteps)
                self.writer.add_scalar("episode/reward_final_vs_timesteps", reward_val, self.num_timesteps)
                self.writer.add_scalar("episode/selected_cells_final_vs_timesteps", selected_cells_val, self.num_timesteps)
                self.writer.add_scalar("episode/eval_K_used_final_vs_timesteps", eval_k_val, self.num_timesteps)
                self.writer.add_scalar("episode/uv_success_final_vs_timesteps", uv_success_val, self.num_timesteps)
                self.writer.add_scalar("episode/area_cap_exceeded_final_vs_timesteps", area_cap_exceeded_val, self.num_timesteps)

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
    """

    def __init__(
        self,
        log_dir: str,
        every_n_steps: int = 1,
        keep_last_n: int = 1000,
        log_every_500: int = 500,
        verbose: int = 0,
    ):
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

        row = {
            "step": int(self.num_timesteps),
            "base_conf": float(base_conf) if base_conf is not None else None,
            "after_conf": float(after_conf) if after_conf is not None else None,
            "delta_conf": float(delta_conf) if delta_conf is not None else None,
            "area_frac": float(area_frac) if area_frac is not None else None,
            "drop_on": float(drop_on) if drop_on is not None else None,
            "drop_on_smooth": float(drop_on_s) if drop_on_s is not None else None,
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

