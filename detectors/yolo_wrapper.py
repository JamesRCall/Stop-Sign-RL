# detectors/yolo_wrapper.py
from typing import Union
import re
import torch

def _norm(s: str) -> str:
    # normalize like "Stop Sign" -> "stopsign"
    return re.sub(r"[\s\-_]+", "", s.strip().lower())

class DetectorWrapper:
    def __init__(
        self,
        model_path: str,
        target_class: Union[str, int] = "stop sign",
        device: str = "cpu",            # <-- default cpu to avoid GPU copies in workers
        conf: float = 0.10,
        iou: float = 0.45,
    ):
        from ultralytics import YOLO

        # Resolve device
        dev = str(device).lower()
        if dev == "auto":
            dev = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = dev
        self.conf = float(conf)
        self.iou  = float(iou)

        # Load model and try to place on device (older Ultralytics may ignore .to(); we also pass device in predict)
        self.model = YOLO(model_path)
        try:
            self.model.to(self.device)
        except Exception:
            pass

        self.use_half = self.device.startswith("cuda")

        # Build id->name and normalized name->id maps
        names_raw = getattr(self.model, "names", {})
        if isinstance(names_raw, dict):
            id_to_name = {int(k): str(v) for k, v in names_raw.items()}
        else:
            id_to_name = {i: str(n) for i, n in enumerate(names_raw)}
        name_to_id_norm = {_norm(v): k for k, v in id_to_name.items()}

        if isinstance(target_class, str):
            tc_norm = _norm(target_class)
            if tc_norm in name_to_id_norm:
                self.target_id = int(name_to_id_norm[tc_norm])
            else:
                # common aliases
                for alias in ["stopsign", "stop-sign", "stop_sign", "stop"]:
                    if alias in name_to_id_norm:
                        self.target_id = int(name_to_id_norm[alias])
                        break
                else:
                    # fallback (COCO often has stop sign at 11, but don't rely on it if names available)
                    self.target_id = 11
        else:
            self.target_id = int(target_class)

    def infer_confidence(self, pil_image) -> float:
        # Always pass device explicitly; set half only on CUDA
        res = self.model.predict(
            pil_image,
            conf=self.conf,
            iou=self.iou,
            verbose=False,
            device=self.device,      # <-- FORCE device every call
            half=self.use_half,
        )
        if not res:
            return 0.0

        r0 = res[0]
        boxes = getattr(r0, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return 0.0

        # --- robust tensor -> numpy helpers ---
        import numpy as np
        def to_numpy(x):
            try:
                return x.detach().cpu().numpy()
            except Exception:
                return np.asarray(x)

        confs = to_numpy(boxes.conf).astype(float).reshape(-1)
        clss  = to_numpy(boxes.cls).astype(int).reshape(-1)

        mask = (clss == self.target_id)
        return float(confs[mask].max()) if mask.any() else 0.0
