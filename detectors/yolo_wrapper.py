"""Thin wrapper around Ultralytics YOLO for traffic-sign confidence queries."""
import torch

from detectors.class_names import ClassReference, resolve_class_id

class DetectorWrapper:
    def __init__(
        self,
        model_path: str,
        target_class: ClassReference = "stop sign",
        device: str = "cpu",    # default cpu; you can pass "cuda:0" to use GPU
        conf: float = 0.10,
        iou: float = 0.45,
        debug: bool = False,    # NEW: print errors once if something goes wrong
    ):
        """
        Args:
            model_path: Path to YOLO weights.
            target_class: Target class name or id.
            device: Device string (cpu/cuda/auto).
            conf: Confidence threshold.
            iou: IoU threshold.
            debug: Enable debug logging.
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
        self.target_id = self.resolve_class_id(target_class, role="source class")

    def resolve_class_id(self, class_ref: ClassReference, *, role: str = "class") -> int:
        """Resolve a class against this checkpoint's label map."""
        return resolve_class_id(self.id_to_name, class_ref, role=role)

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
            raise RuntimeError("YOLO inference failed; sample is invalid") from e

        if not res:
            raise RuntimeError("YOLO returned no result object for one input image")

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
            raise RuntimeError("YOLO batch inference failed; samples are invalid") from e

        if len(results) != len(pil_images):
            raise RuntimeError(
                f"YOLO returned {len(results)} results for {len(pil_images)} images"
            )

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
            raise RuntimeError("YOLO detection inference failed; samples are invalid") from e

        if len(results) != len(pil_images):
            raise RuntimeError(
                f"YOLO returned {len(results)} results for {len(pil_images)} images"
            )

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
