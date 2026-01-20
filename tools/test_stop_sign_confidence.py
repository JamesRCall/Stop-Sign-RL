#!/usr/bin/env python3
"""
test_stop_sign_confidence.py

Quick visual + numerical check for how your YOLO model scores a stop sign in a given image.

- Prints all detected stop-sign boxes with confidence values.
- Shows (and optionally saves) an annotated image with labeled confidence.
"""

import argparse
import os
import sys
from typing import List, Tuple

import torch
from PIL import Image, ImageDraw, ImageFont

try:
    from ultralytics import YOLO
except Exception as e:
    print("ERROR: Could not import 'ultralytics'. Install it with:\n  pip install ultralytics", file=sys.stderr)
    raise


# ---------- helper functions ----------

def pick_stop_sign_class_ids(names: dict) -> List[int]:
    """
    Return list of class IDs that look like "stop sign".

    @param names: Class id-to-name mapping.
    @return: Sorted list of stop-sign class ids.
    """
    ids = []
    for k, v in names.items():
        name = str(v).strip().lower()
        if "stop" in name and "sign" in name:
            ids.append(int(k))
    return sorted(ids)


def draw_boxes_with_conf(pil_img: Image.Image, boxes, confs, names, clses) -> Image.Image:
    """
    Draw boxes and confidence text directly on the image.

    @param pil_img: Source image.
    @param boxes: Array of bounding boxes.
    @param confs: Array of confidences.
    @param names: Class name mapping.
    @param clses: Array of class ids.
    @return: Annotated image.
    """
    img = pil_img.convert("RGB").copy()
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except:
        font = ImageFont.load_default()

    for (x1, y1, x2, y2), conf, cls_id in zip(boxes, confs, clses):
        label = f"STOP sign {conf * 100:.1f}%"

        # Draw rectangle for the detection box
        draw.rectangle([(x1, y1), (x2, y2)], outline=(255, 0, 0), width=3)

        # Compute text size (Pillow 10+ compatibility)
        try:
            # Newer Pillow
            left, top, right, bottom = draw.textbbox((x1, y1), label, font=font)
            text_w, text_h = right - left, bottom - top
        except AttributeError:
            # Older Pillow
            text_w, text_h = draw.textsize(label, font=font)

        # Draw label background
        text_bg_y1 = max(y1 - text_h - 6, 0)
        text_bg_y2 = y1
        draw.rectangle([(x1, text_bg_y1), (x1 + text_w + 8, text_bg_y2)], fill=(255, 0, 0))
        draw.text((x1 + 4, text_bg_y1 + 2), label, fill=(255, 255, 255), font=font)

    return img


# ---------- main ----------

def main():
    """
    Run a single-image confidence check and optional annotated export.

    @return: None
    """
    ap = argparse.ArgumentParser(description="Test YOLOv11 stop-sign confidence on a single image")
    ap.add_argument("--weights", default="./weights/yolo11n.pt", help="Path to YOLOv11 weights (.pt)")
    ap.add_argument("--image", required=True, help="Path to input image")
    ap.add_argument("--device", default="auto", help="cuda, cpu, or auto")
    ap.add_argument("--conf", type=float, default=0.1, help="Confidence threshold")
    ap.add_argument("--iou", type=float, default=0.45, help="IoU threshold for NMS")
    ap.add_argument("--save", action="store_true", help="Save annotated image")
    ap.add_argument("--out", default="", help="Optional output image path")
    args = ap.parse_args()

    if not os.path.isfile(args.weights):
        sys.exit(f"ERROR: weights not found: {args.weights}")
    if not os.path.isfile(args.image):
        sys.exit(f"ERROR: image not found: {args.image}")

    # pick device
    device = "cuda" if (args.device == "auto" and torch.cuda.is_available()) else args.device
    model = YOLO(args.weights)
    names = model.names if hasattr(model, "names") else {}
    stop_ids = pick_stop_sign_class_ids(names)

    if not stop_ids:
        print("WARN:  No 'stop sign' class found in model.names; detections may be empty.")

    # run inference
    results = model.predict(
        source=args.image,
        conf=args.conf,
        iou=args.iou,
        device=device,
        verbose=False
    )

    res = results[0]
    if res.boxes is None or len(res.boxes) == 0:
        print("No detections.")
        return

    boxes = res.boxes.xyxy.cpu().numpy()
    confs = res.boxes.conf.cpu().numpy()
    clses = res.boxes.cls.cpu().numpy().astype(int)

    import numpy as np
    stop_mask = np.isin(clses, stop_ids)
    stop_boxes, stop_confs, stop_clses = boxes[stop_mask], confs[stop_mask], clses[stop_mask]

    if stop_boxes.shape[0] == 0:
        print("ERROR: No STOP SIGN detected above threshold.")
        return

    # Print detections
    print(" Detected STOP SIGN(s):")
    for i, (b, c) in enumerate(zip(stop_boxes, stop_confs), 1):
        x1, y1, x2, y2 = [float(v) for v in b]
        print(f"  #{i}: conf={c:.4f} ({c*100:.1f}%)  box=[{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]")

    top_idx = int(np.argmax(stop_confs))
    print(f"\n TOP STOP SIGN confidence: {stop_confs[top_idx]*100:.2f}%")

    # Draw and save annotated image
    if args.save or args.out:
        pil_img = Image.open(args.image).convert("RGB")
        ann = draw_boxes_with_conf(pil_img, stop_boxes, stop_confs, names, stop_clses)
        out_path = args.out or f"{os.path.splitext(args.image)[0]}_annotated.jpg"
        ann.save(out_path)
        print(f" Saved annotated image -> {out_path}")


if __name__ == "__main__":
    main()
