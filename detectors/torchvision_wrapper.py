"""Thin wrapper around torchvision detection models for stop-sign confidence queries."""
from __future__ import annotations

from typing import Optional, Union
import re
import torch


def _norm(s: str) -> str:
    """Normalize class names for comparison."""
    return re.sub(r"[\s\-_]+", "", s.strip().lower())


class TorchvisionDetectorWrapper:
    """
    Wrapper around torchvision detection models that exposes the same interface
    as the YOLO wrapper: infer_confidence(), infer_confidence_batch(),
    infer_detections_batch(), target_id, id_to_name.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        target_class: Union[str, int] = "stop sign",
        device: str = "cpu",
        conf: float = 0.10,
        iou: float = 0.45,
        debug: bool = False,
    ):
        """
        @param model_name: Torchvision model name (e.g., fasterrcnn_resnet50_fpn_v2).
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

        self.model_name = self._resolve_model_name(model_name)
        self.model, self.id_to_name = self._load_model(self.model_name)
        self.target_id = self._resolve_target_id(target_class)

        self.model.to(self.device)
        self.model.eval()

    def _resolve_model_name(self, name: Optional[str]) -> str:
        if not name:
            return "fasterrcnn_resnet50_fpn_v2"
        n = str(name).strip().lower()
        aliases = {
            "fasterrcnn": "fasterrcnn_resnet50_fpn_v2",
            "faster-rcnn": "fasterrcnn_resnet50_fpn_v2",
            "retinanet": "retinanet_resnet50_fpn_v2",
            "ssd": "ssd300_vgg16",
            "fcos": "fcos_resnet50_fpn",
            "centernet": "centernet_resnet50_fpn",
        }
        return aliases.get(n, n)

    def _load_model(self, name: str):
        if name == "centernet_resnet50_fpn":
            raise ValueError("CenterNet is not supported in this build. Choose another detector.")

        import torchvision.models.detection as tvd

        specs = {
            "fasterrcnn_resnet50_fpn_v2": ("fasterrcnn_resnet50_fpn_v2", "FasterRCNN_ResNet50_FPN_V2_Weights"),
            "fasterrcnn_resnet50_fpn": ("fasterrcnn_resnet50_fpn", "FasterRCNN_ResNet50_FPN_Weights"),
            "retinanet_resnet50_fpn_v2": ("retinanet_resnet50_fpn_v2", "RetinaNet_ResNet50_FPN_V2_Weights"),
            "retinanet_resnet50_fpn": ("retinanet_resnet50_fpn", "RetinaNet_ResNet50_FPN_Weights"),
            "ssd300_vgg16": ("ssd300_vgg16", "SSD300_VGG16_Weights"),
            "fcos_resnet50_fpn": ("fcos_resnet50_fpn", "FCOS_ResNet50_FPN_Weights"),
        }

        if name not in specs:
            supported = ", ".join(sorted(specs.keys()))
            raise ValueError(f"Unsupported torchvision detector '{name}'. Supported: {supported}")

        fn_name, weights_name = specs[name]
        fn = getattr(tvd, fn_name, None)
        weights_cls = getattr(tvd, weights_name, None)
        if fn is None or weights_cls is None:
            raise ValueError(
                f"Torchvision model '{name}' not available in this torchvision version."
            )

        weights = weights_cls.DEFAULT
        model = fn(weights=weights)

        categories = []
        meta = getattr(weights, "meta", None)
        if isinstance(meta, dict):
            categories = meta.get("categories", []) or []

        if categories and isinstance(categories, (list, tuple)):
            first = str(categories[0]).lower()
            has_bg = "background" in first
            if has_bg:
                id_to_name = {i: str(n) for i, n in enumerate(categories)}
            else:
                id_to_name = {i + 1: str(n) for i, n in enumerate(categories)}
        else:
            id_to_name = {}

        return model, id_to_name

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

    def _preprocess(self, pil_image):
        from torchvision.transforms.functional import to_tensor

        if pil_image.mode != "RGB":
            pil_image = pil_image.convert("RGB")
        return to_tensor(pil_image)

    def _predict(self, pil_images):
        if not pil_images:
            return []
        tensors = [self._preprocess(im).to(self.device) for im in pil_images]
        with torch.no_grad():
            return self.model(tensors)

    def _filter_detections(self, output):
        boxes = output.get("boxes", None)
        scores = output.get("scores", None)
        labels = output.get("labels", None)
        if boxes is None or scores is None or labels is None:
            return None, None, None
        if scores.numel() == 0:
            return None, None, None

        if self.conf is not None:
            keep = scores >= float(self.conf)
            boxes = boxes[keep]
            scores = scores[keep]
            labels = labels[keep]

        if boxes.numel() == 0:
            return None, None, None

        if self.iou is not None and 0.0 < float(self.iou) < 1.0 and boxes.shape[0] > 1:
            from torchvision.ops import batched_nms
            keep = batched_nms(boxes, scores, labels, float(self.iou))
            boxes = boxes[keep]
            scores = scores[keep]
            labels = labels[keep]

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
