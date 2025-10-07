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
