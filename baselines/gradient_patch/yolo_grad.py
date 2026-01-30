"""Differentiable YOLO target-confidence wrapper.

This avoids NMS and uses raw model outputs to compute a smooth confidence
signal for the target class. The parsing is heuristic to support common
Ultralytics output formats (v8/v11).
"""
from __future__ import annotations

from typing import Optional, Tuple
import re
import torch


def _norm_name(s: str) -> str:
    return re.sub(r"[\s\-_]+", "", str(s).strip().lower())


class YoloGrad:
    def __init__(
        self,
        weights: str,
        target_class: str = "stop sign",
        device: str = "auto",
    ) -> None:
        from ultralytics import YOLO

        dev = str(device).lower().strip()
        if dev == "auto":
            dev = "cuda:0" if torch.cuda.is_available() else "cpu"
        if dev == "cuda":
            dev = "cuda:0"
        self.device = dev

        self.yolo = YOLO(weights)
        # Underlying torch model
        self.model = self.yolo.model
        self.model.to(self.device)
        self.model.eval()

        names = getattr(self.yolo, "names", None)
        if names is None:
            names = getattr(self.model, "names", {})
        if isinstance(names, dict):
            id_to_name = {int(k): str(v) for k, v in names.items()}
        else:
            id_to_name = {i: str(n) for i, n in enumerate(names)}
        self.id_to_name = id_to_name
        name_to_id = {_norm_name(v): k for k, v in id_to_name.items()}
        tc_norm = _norm_name(target_class)
        if tc_norm in name_to_id:
            self.target_id = int(name_to_id[tc_norm])
        else:
            # fallback aliases
            for alias in ("stopsign", "stop-sign", "stop_sign", "stop"):
                if alias in name_to_id:
                    self.target_id = int(name_to_id[alias])
                    break
            else:
                self.target_id = 11  # COCO stop-sign often 11
        self.num_classes = len(self.id_to_name) if self.id_to_name else 80

    def _ensure_pred_format(self, pred: torch.Tensor) -> torch.Tensor:
        """
        Convert pred to (B, N, C) if possible.
        """
        if pred.dim() == 2:
            pred = pred.unsqueeze(0)
        if pred.dim() == 4:
            # Sometimes (B, C, H, W) - flatten spatial
            b, c, h, w = pred.shape
            pred = pred.view(b, c, h * w).permute(0, 2, 1)
            return pred
        if pred.dim() != 3:
            raise ValueError(f"Unexpected pred dims: {pred.shape}")
        b, d1, d2 = pred.shape
        # Heuristic: choose (B, N, C) where C is smallest plausible channel dim
        if d2 <= d1:
            return pred
        return pred.permute(0, 2, 1)

    def _target_conf_from_pred(self, pred: torch.Tensor) -> torch.Tensor:
        """
        Return per-image target confidence (B,) from raw model output.
        """
        pred = self._ensure_pred_format(pred)  # (B, N, C)
        c = pred.shape[-1]

        # Try to interpret channels
        # Common: [x, y, w, h, obj, cls...]
        if c >= (5 + self.num_classes):
            obj = pred[..., 4].sigmoid()
            cls = pred[..., 5 + self.target_id].sigmoid()
            conf = obj * cls
        elif c == (4 + self.num_classes):
            cls = pred[..., 4 + self.target_id].sigmoid()
            conf = cls
        elif c == self.num_classes:
            cls = pred[..., self.target_id].sigmoid()
            conf = cls
        elif c > 5 and (5 + self.target_id) < c:
            obj = pred[..., 4].sigmoid()
            cls = pred[..., 5 + self.target_id].sigmoid()
            conf = obj * cls
        else:
            # Fallback: take the last channel as a confidence proxy
            conf = pred[..., -1].sigmoid()

        # Max over locations/anchors
        return conf.max(dim=1).values

    def target_conf(self, images: torch.Tensor) -> torch.Tensor:
        """
        Compute target confidence for a batch of images.

        @param images: Tensor (B, 3, H, W) in [0,1].
        @return: Tensor (B,) target confidences.
        """
        if images.dim() == 3:
            images = images.unsqueeze(0)
        if images.device != torch.device(self.device):
            images = images.to(self.device)
        if images.dtype != torch.float32:
            images = images.float()

        pred = self.model(images)
        if isinstance(pred, (list, tuple)):
            pred = pred[0]
        return self._target_conf_from_pred(pred)

