#!/usr/bin/env python3
"""Debug detector outputs on a single image."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from detectors.factory import build_detector


def _label_for(det, cls_id: int) -> str:
    id_to_name = getattr(det, "id_to_name", None)
    if isinstance(id_to_name, dict):
        return str(id_to_name.get(int(cls_id), ""))
    return ""


def _draw_boxes(img: Image.Image, boxes, confs, clss, det, out_path: str) -> None:
    draw = ImageDraw.Draw(img)
    for box, conf, cls_id in zip(boxes, confs, clss):
        x1, y1, x2, y2 = [float(v) for v in box]
        label = _label_for(det, int(cls_id))
        text = f"{int(cls_id)} {label} {float(conf):.3f}".strip()
        draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=2)
        draw.text((x1 + 2, y1 + 2), text, fill=(255, 255, 0))
    img.save(out_path)


def main() -> int:
    ap = argparse.ArgumentParser(description="Debug detector outputs on a single image.")
    ap.add_argument("--image", required=True, help="Path to input image")
    ap.add_argument("--detector", default="yolo",
                    help="Detector backend: yolo, torchvision, rtdetr")
    ap.add_argument("--detector-model", default="",
                    help="Torchvision/transformers model id (if applicable)")
    ap.add_argument("--yolo-weights", default="./weights/yolo8n.pt",
                    help="YOLO weights path")
    ap.add_argument("--device", default="auto", help="cpu, cuda, or auto")
    ap.add_argument("--conf", type=float, default=0.10, help="Confidence threshold")
    ap.add_argument("--iou", type=float, default=0.45, help="IoU/NMS threshold")
    ap.add_argument("--target-class", default="stop sign", help="Target class name or id")
    ap.add_argument("--topk", type=int, default=0, help="Limit printed detections (0 = all)")
    ap.add_argument("--save", action="store_true", help="Save annotated image")
    ap.add_argument("--out", default="", help="Output image path if --save is set")
    args = ap.parse_args()

    img_path = Path(args.image)
    if not img_path.exists():
        raise FileNotFoundError(f"Image not found: {img_path}")

    img = Image.open(img_path).convert("RGB")
    det = build_detector(
        detector_type=str(args.detector),
        detector_model=str(args.detector_model) if args.detector_model else None,
        yolo_weights=str(args.yolo_weights),
        device=str(args.device),
        conf=float(args.conf),
        iou=float(args.iou),
        target_class=str(args.target_class),
        debug=True,
    )

    out = det.infer_detections_batch([img])[0]
    boxes = out.get("boxes", []) or []
    confs = out.get("confs", []) or []
    clss = out.get("clss", []) or []

    target_id = getattr(det, "target_id", None)
    target_label = _label_for(det, int(target_id)) if target_id is not None else ""

    print(f"detector={args.detector} model={args.detector_model or '<default>'} device={args.device}")
    print(f"target_id={target_id} label={target_label}")
    print(f"detections={len(confs)}")

    rows = list(zip(boxes, confs, clss))
    rows.sort(key=lambda r: float(r[1]), reverse=True)
    if args.topk and args.topk > 0:
        rows = rows[: int(args.topk)]

    for i, (box, conf, cls_id) in enumerate(rows, start=1):
        label = _label_for(det, int(cls_id))
        x1, y1, x2, y2 = [float(v) for v in box]
        print(
            f"[{i:03d}] cls={int(cls_id)} label={label} conf={float(conf):.4f} "
            f"box=({x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f})"
        )

    if args.save:
        out_path = args.out or str(img_path.with_name(img_path.stem + "_det.png"))
        _draw_boxes(img.copy(), boxes, confs, clss, det, out_path)
        print(f"saved={out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
