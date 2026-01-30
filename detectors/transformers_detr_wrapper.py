"""Wrapper around Hugging Face Transformers DETR for stop-sign confidence queries."""
from __future__ import annotations

from typing import Optional, Union
import re
import warnings
import torch


def _norm(s: str) -> str:
    """Normalize class names for comparison."""
    return re.sub(r"[\s\-_]+", "", s.strip().lower())


class TransformersDetrWrapper:
    """
    DETR wrapper using Hugging Face transformers.

    Exposes the same interface as YOLO wrapper:
      - infer_confidence
      - infer_confidence_batch
      - infer_detections_batch
      - target_id, id_to_name
    """

    def __init__(
        self,
        model_name: str = "facebook/detr-resnet-50",
        target_class: Union[str, int] = "stop sign",
        device: str = "cpu",
        conf: float = 0.10,
        iou: float = 0.45,
        debug: bool = False,
    ):
        """
        @param model_name: Hugging Face model id for DETR.
        @param target_class: Target class name or id.
        @param device: Device string (cpu/cuda/auto).
        @param conf: Confidence threshold.
        @param iou: IoU threshold for optional extra NMS.
        @param debug: Enable debug logging.
        """
        dev = str(device).lower().strip()
        if dev == "auto":
            dev = "cuda:0" if torch.cuda.is_available() else "cpu"
        if dev == "cuda":
            dev = "cuda:0"
        self.device = dev
        self.conf = float(conf)
        self.iou = float(iou)
        self.debug = bool(debug)
        self.model_name = str(model_name)

        try:
            from transformers import DetrImageProcessor, DetrForObjectDetection
            from transformers.utils import logging as hf_logging
        except Exception as e:  # pragma: no cover - import guard
            raise ImportError(
                "Transformers DETR requires the 'transformers' package. "
                "Install with: pip install transformers"
            ) from e

        # Reduce noisy load-time warnings and logs from transformers/torch meta init.
        warnings.filterwarnings(
            "ignore",
            message=".*meta parameter.*no-op.*",
            category=UserWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message=".*copying from a non-meta parameter.*",
            category=UserWarning,
        )
        hf_logging.set_verbosity_error()

        self.processor = DetrImageProcessor.from_pretrained(self.model_name)
        self.model = DetrForObjectDetection.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()

        id_to_name = {}
        id2label = getattr(self.model.config, "id2label", None)
        if isinstance(id2label, dict):
            for k, v in id2label.items():
                try:
                    id_to_name[int(k)] = str(v)
                except Exception:
                    continue
        self.id_to_name = id_to_name
        self.target_id = self._resolve_target_id(target_class)

    def _resolve_target_id(self, target_class: Union[str, int]) -> int:
        if isinstance(target_class, int):
            return int(target_class)
        tc = str(target_class).strip()
        if tc.isdigit():
            return int(tc)

        tc_norm = _norm(tc)
        if isinstance(self.id_to_name, dict):
            for k, v in self.id_to_name.items():
                if _norm(v) == tc_norm:
                    return int(k)
            for alias in ["stopsign", "stop-sign", "stop_sign", "stop"]:
                for k, v in self.id_to_name.items():
                    if _norm(v) == _norm(alias):
                        return int(k)
        return 11

    def _predict(self, pil_images):
        if not pil_images:
            return []
        inputs = self.processor(images=pil_images, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
        target_sizes = torch.tensor(
            [(int(im.size[1]), int(im.size[0])) for im in pil_images],
            device=self.device,
        )
        results = self.processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=float(self.conf)
        )
        return results

    def _filter_detections(self, output):
        boxes = output.get("boxes", None)
        scores = output.get("scores", None)
        labels = output.get("labels", None)
        if boxes is None or scores is None or labels is None:
            return None, None, None
        if scores.numel() == 0:
            return None, None, None

        if self.iou is not None and 0.0 < float(self.iou) < 1.0 and boxes.shape[0] > 1:
            try:
                from torchvision.ops import batched_nms
                keep = batched_nms(boxes, scores, labels, float(self.iou))
                boxes = boxes[keep]
                scores = scores[keep]
                labels = labels[keep]
            except Exception:
                pass

        if boxes.numel() == 0:
            return None, None, None
        return boxes, scores, labels

    def infer_confidence(self, pil_image) -> float:
        return float(self.infer_confidence_batch([pil_image])[0])

    def infer_confidence_batch(self, pil_images) -> list[float]:
        results = self._predict(pil_images)
        out = []
        for r0 in results:
            boxes, scores, labels = self._filter_detections(r0)
            if boxes is None:
                out.append(0.0)
                continue
            mask = (labels == int(self.target_id))
            if not bool(mask.any().item()):
                out.append(0.0)
                continue
            out.append(float(scores[mask].max().detach().cpu().item()))
        return out

    def infer_detections_batch(self, pil_images) -> list[dict]:
        results = self._predict(pil_images)
        out = []
        for r0 in results:
            boxes, scores, labels = self._filter_detections(r0)
            if boxes is None:
                out.append(
                    {
                        "target_conf": 0.0,
                        "target_box": None,
                        "top_conf": 0.0,
                        "top_class": None,
                        "top_box": None,
                        "boxes": [],
                        "confs": [],
                        "clss": [],
                    }
                )
                continue

            boxes_np = boxes.detach().cpu().numpy()
            scores_np = scores.detach().cpu().numpy()
            labels_np = labels.detach().cpu().numpy()

            top_idx = int(scores_np.argmax()) if scores_np.size else None
            top_conf = float(scores_np[top_idx]) if top_idx is not None else 0.0
            top_class = int(labels_np[top_idx]) if top_idx is not None else None
            top_box = boxes_np[top_idx].tolist() if top_idx is not None else None

            target_mask = (labels_np == int(self.target_id))
            if target_mask.any():
                t_scores = scores_np[target_mask]
                t_boxes = boxes_np[target_mask]
                t_idx = int(t_scores.argmax())
                target_conf = float(t_scores[t_idx])
                target_box = t_boxes[t_idx].tolist()
            else:
                target_conf = 0.0
                target_box = None

            out.append(
                {
                    "target_conf": target_conf,
                    "target_box": target_box,
                    "top_conf": top_conf,
                    "top_class": top_class,
                    "top_box": top_box,
                    "boxes": boxes_np.tolist(),
                    "confs": scores_np.tolist(),
                    "clss": labels_np.tolist(),
                }
            )
        return out
