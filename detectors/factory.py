"""Detector factory for YOLO, torchvision, or remote detector backends."""
from __future__ import annotations

from typing import Optional


def build_detector(
    *,
    detector_type: str = "yolo",
    detector_model: Optional[str] = None,
    yolo_weights: Optional[str] = None,
    device: str = "cpu",
    conf: float = 0.10,
    iou: float = 0.45,
    target_class: str = "stop sign",
    debug: bool = False,
):
    """
    Build a detector wrapper instance.

    Args:
        detector_type: "yolo" or "torchvision" (aliases: "tv").
        detector_model: Model name for torchvision (ignored for YOLO).
        yolo_weights: Path to YOLO weights.
        device: Device string (cpu/cuda/auto or server://host:port).
        conf: Confidence threshold.
        iou: IoU threshold.
        target_class: Target class name or id.
        debug: Enable debug logging.
    """
    dev_str = str(device)
    if dev_str.lower().startswith("server://"):
        from detectors.remote_detector import RemoteDetectorWrapper
        return RemoteDetectorWrapper(
            server_addr=dev_str,
            conf=conf,
            iou=iou,
            debug=debug,
        )

    dtype = str(detector_type or "yolo").strip().lower()
    if dtype in ("yolo", "ultralytics"):
        from detectors.yolo_wrapper import DetectorWrapper
        return DetectorWrapper(
            yolo_weights,
            target_class=target_class,
            device=device,
            conf=conf,
            iou=iou,
            debug=debug,
        )
    if dtype in ("torchvision", "tv"):
        from detectors.torchvision_wrapper import TorchvisionDetectorWrapper
        return TorchvisionDetectorWrapper(
            model_name=detector_model,
            target_class=target_class,
            device=device,
            conf=conf,
            iou=iou,
            debug=debug,
        )
    if dtype in ("detr", "transformers", "hf", "rtdetr", "rtdetrv2", "rt-detr", "rt-detrv2"):
        from detectors.transformers_detr_wrapper import TransformersDetrWrapper
        return TransformersDetrWrapper(
            model_name=detector_model or "facebook/detr-resnet-50",
            target_class=target_class,
            device=device,
            conf=conf,
            iou=iou,
            debug=debug,
        )

    raise ValueError(
        f"Unsupported detector_type '{detector_type}'. Use 'yolo', 'torchvision', or 'detr'."
    )
