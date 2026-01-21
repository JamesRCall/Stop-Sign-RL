"""Thin wrapper around Ultralytics YOLO for stop-sign confidence queries."""
from typing import Union
import re
import torch

def _norm(s: str) -> str:
    """Normalize class names for comparison."""
    return re.sub(r"[\s\-_]+", "", s.strip().lower())

class DetectorWrapper:
    def __init__(
        self,
        model_path: str,
        target_class: Union[str, int] = "stop sign",
        device: str = "cpu",    # default cpu; you can pass "cuda:0" to use GPU
        conf: float = 0.10,
        iou: float = 0.45,
        debug: bool = False,    # NEW: print errors once if something goes wrong
    ):
        """
        @param model_path: Path to YOLO weights.
        @param target_class: Target class name or id.
        @param device: Device string (cpu/cuda/auto).
        @param conf: Confidence threshold.
        @param iou: IoU threshold.
        @param debug: Enable debug logging.
        """
        from ultralytics import YOLO

        dev = str(device).lower().strip()
        if dev == "auto":
            dev = "cuda:0" if torch.cuda.is_available() else "cpu"
        # normalize "cuda" -> "cuda:0"
        if dev == "cuda":
            dev = "cuda:0"
        self.device = dev
        self.conf = float(conf)
        self.iou  = float(iou)
        self.debug = bool(debug)
        self._logged_names = False

        self.model = YOLO(model_path)
        # Moving the model is optional; predict(device=...) is sufficient,
        # but .to() is harmless if supported.
        try:
            self.model.to(self.device)
        except Exception:
            pass

        # Build name maps
        names_raw = getattr(self.model, "names", {})
        if isinstance(names_raw, dict):
            id_to_name = {int(k): str(v) for k, v in names_raw.items()}
        else:
            id_to_name = {i: str(n) for i, n in enumerate(names_raw)}
        self.id_to_name = id_to_name
        name_to_id_norm = {_norm(v): k for k, v in id_to_name.items()}

        if isinstance(target_class, str):
            tc_norm = _norm(target_class)
            if tc_norm in name_to_id_norm:
                self.target_id = int(name_to_id_norm[tc_norm])
            else:
                for alias in ["stopsign", "stop-sign", "stop_sign", "stop"]:
                    if alias in name_to_id_norm:
                        self.target_id = int(name_to_id_norm[alias])
                        break
                else:
                    # fallback (COCO often 11, but don't rely on it without names)
                    self.target_id = 11
        else:
            self.target_id = int(target_class)

    def infer_confidence(self, pil_image) -> float:
        """Return max confidence for the target class in a single image."""
        try:
            # Important: do NOT pass half=...; let Ultralytics decide. Always pass device explicitly.
            res = self.model.predict(
                pil_image,
                conf=self.conf,
                iou=self.iou,
                verbose=False,
                device=self.device,
            )
        except Exception as e:
            if self.debug:
                print(f"[DetectorWrapper] predict() error on device={self.device}: {e}")
            return 0.0

        if not res:
            return 0.0

        r0 = res[0]
        boxes = getattr(r0, "boxes", None)
        if boxes is None or len(boxes) == 0:
            # optional: one-time log class names to confirm mapping
            if self.debug and not self._logged_names:
                print(f"[DetectorWrapper] No boxes; target_id={self.target_id} | names={self.id_to_name}")
                self._logged_names = True
            return 0.0

        # robust tensor -> numpy
        import numpy as np
        def to_numpy(x):
            try:
                return x.detach().cpu().numpy()
            except Exception:
                return np.asarray(x)

        confs = to_numpy(boxes.conf).astype(float).reshape(-1)
        clss  = to_numpy(boxes.cls).astype(int).reshape(-1)

        mask = (clss == self.target_id)
        if not mask.any():
            if self.debug and not self._logged_names:
                print(f"[DetectorWrapper] No target class in detections; target_id={self.target_id} | names={self.id_to_name}")
                self._logged_names = True
            return 0.0
        return float(confs[mask].max())

    def infer_confidence_batch(self, pil_images) -> list[float]:
        """
        Returns a list of max confidences for target class, one per image.
        Ultralytics supports passing a list of images to predict().
        """
        try:
            results = self.model.predict(
                pil_images,
                conf=self.conf,
                iou=self.iou,
                verbose=False,
                device=self.device,
            )
        except Exception as e:
            if self.debug:
                print(f"[DetectorWrapper] batch predict() error on device={self.device}: {e}")
            return [0.0 for _ in pil_images]

        out = []
        import numpy as np

        def to_numpy(x):
            try:
                return x.detach().cpu().numpy()
            except Exception:
                return np.asarray(x)

        for r0 in results:
            boxes = getattr(r0, "boxes", None)
            if boxes is None or len(boxes) == 0:
                out.append(0.0)
                continue
            confs = to_numpy(boxes.conf).astype(float).reshape(-1)
            clss  = to_numpy(boxes.cls).astype(int).reshape(-1)
            mask = (clss == self.target_id)
            out.append(float(confs[mask].max()) if mask.any() else 0.0)
        return out

    def infer_detections_batch(self, pil_images) -> list[dict]:
        """
        Return detection summaries for each image.

        Each entry includes:
          - target_conf: max confidence for target class
          - target_box: [x1, y1, x2, y2] for max target conf (or None)
          - top_conf: max confidence across all detections
          - top_class: class id for top detection (or None)
          - top_box: [x1, y1, x2, y2] for top detection (or None)
          - boxes, confs, clss: full arrays for downstream analysis
        """
        try:
            results = self.model.predict(
                pil_images,
                conf=self.conf,
                iou=self.iou,
                verbose=False,
                device=self.device,
            )
        except Exception as e:
            if self.debug:
                print(f"[DetectorWrapper] batch predict() error on device={self.device}: {e}")
            return [
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
                for _ in pil_images
            ]

        out = []
        import numpy as np

        def to_numpy(x):
            try:
                return x.detach().cpu().numpy()
            except Exception:
                return np.asarray(x)

        for r0 in results:
            boxes = getattr(r0, "boxes", None)
            if boxes is None or len(boxes) == 0:
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

            confs = to_numpy(boxes.conf).astype(float).reshape(-1)
            clss = to_numpy(boxes.cls).astype(int).reshape(-1)
            xyxy = to_numpy(boxes.xyxy).astype(float).reshape(-1, 4)

            top_idx = int(np.argmax(confs)) if confs.size else None
            top_conf = float(confs[top_idx]) if top_idx is not None else 0.0
            top_class = int(clss[top_idx]) if top_idx is not None else None
            top_box = xyxy[top_idx].tolist() if top_idx is not None else None

            target_mask = (clss == self.target_id)
            if target_mask.any():
                t_idx = int(np.argmax(confs[target_mask]))
                t_all = np.flatnonzero(target_mask)
                t_pick = int(t_all[t_idx])
                target_conf = float(confs[t_pick])
                target_box = xyxy[t_pick].tolist()
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
                    "boxes": xyxy.tolist(),
                    "confs": confs.tolist(),
                    "clss": clss.tolist(),
                }
            )

        return out
