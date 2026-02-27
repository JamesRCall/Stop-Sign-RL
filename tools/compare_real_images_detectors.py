#!/usr/bin/env python3
"""Compare real camera images across six detector backends used in this project.

Outputs:
  - per-image, per-detector detections/confidences
  - aggregate summary metrics per detector
  - optional CSV and annotated overlays

Default detector set (6 total; SSD excluded):
  - YOLOv8
  - YOLO11
  - Torchvision Faster R-CNN (v2)
  - Torchvision FCOS
  - Torchvision RetinaNet (v2)
  - RT-DETR (PekingU/rtdetr_r50vd)
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterable

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from detectors.factory import build_detector  # noqa: E402


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
EXTRA_CAMERA_EXTS = {".heic", ".heif", ".tif", ".tiff"}
IMAGE_EXTS = IMAGE_EXTS | EXTRA_CAMERA_EXTS


@dataclass
class DetectorSpec:
    name: str
    detector_type: str
    detector_model: str | None = None
    yolo_weights: str | None = None


DEFAULT_SIX = [
    DetectorSpec("yolo8", "yolo", yolo_weights="weights/yolo8n.pt"),
    DetectorSpec("yolo11", "yolo", yolo_weights="weights/yolo11n.pt"),
    DetectorSpec("fasterrcnn_v2", "torchvision", detector_model="fasterrcnn_resnet50_fpn_v2"),
    DetectorSpec("fcos", "torchvision", detector_model="fcos_resnet50_fpn"),
    DetectorSpec("retinanet_v2", "torchvision", detector_model="retinanet_resnet50_fpn_v2"),
    DetectorSpec("rtdetr_r50vd", "rtdetr", detector_model="PekingU/rtdetr_r50vd"),
]


def _label_for(det, cls_id: int | None) -> str:
    if cls_id is None:
        return ""
    id_to_name = getattr(det, "id_to_name", None)
    if isinstance(id_to_name, dict):
        return str(id_to_name.get(int(cls_id), ""))
    return ""


def _collect_images(inp: Path, recursive: bool) -> list[Path]:
    if inp.is_file():
        if inp.suffix.lower() not in IMAGE_EXTS:
            raise ValueError(f"Unsupported image extension: {inp}")
        return [inp]
    if not inp.is_dir():
        raise FileNotFoundError(f"Input not found: {inp}")
    if recursive:
        files = [p for p in inp.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    else:
        files = [p for p in inp.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    return sorted(files)


def _maybe_enable_heif_support() -> tuple[bool, str | None]:
    """Try to register HEIF/HEIC support for Pillow if pillow-heif is installed."""
    try:
        import pillow_heif  # type: ignore

        # Newer versions expose register_heif_opener directly.
        if hasattr(pillow_heif, "register_heif_opener"):
            pillow_heif.register_heif_opener()
            return True, None
        # Fallback for older versions.
        if hasattr(pillow_heif, "register_avif_opener"):
            try:
                pillow_heif.register_heif_opener()  # type: ignore[attr-defined]
            except Exception:
                pass
            return True, None
        return False, "pillow-heif installed but opener registration was not found"
    except Exception as e:
        return False, str(e)


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def _draw_boxes(img: Image.Image, boxes, confs, clss, det, out_path: Path, topk: int = 20) -> None:
    draw = ImageDraw.Draw(img)
    rows = list(zip(boxes or [], confs or [], clss or []))
    rows.sort(key=lambda r: float(r[1]), reverse=True)
    if topk and topk > 0:
        rows = rows[:topk]
    for box, conf, cls_id in rows:
        x1, y1, x2, y2 = [float(v) for v in box]
        label = _label_for(det, int(cls_id))
        text = f"{int(cls_id)} {label} {float(conf):.3f}".strip()
        draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=2)
        draw.text((x1 + 2, y1 + 2), text, fill=(255, 255, 0))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)


def _build_specs(args) -> list[DetectorSpec]:
    specs = []
    if args.only:
        wanted = {s.strip().lower() for s in args.only.split(",") if s.strip()}
    else:
        wanted = None
    for spec in DEFAULT_SIX:
        if wanted and spec.name.lower() not in wanted:
            continue
        s = DetectorSpec(**asdict(spec))
        if s.name == "yolo8" and args.yolo8_weights:
            s.yolo_weights = args.yolo8_weights
        if s.name == "yolo11" and args.yolo11_weights:
            s.yolo_weights = args.yolo11_weights
        specs.append(s)
    return specs


def _summarize_detector_rows(rows: Iterable[dict]) -> dict[str, Any]:
    rows = list(rows)
    n = len(rows)
    if n == 0:
        return {
            "images": 0,
            "mean_target_conf": 0.0,
            "median_target_conf": 0.0,
            "target_detect_rate": 0.0,
            "mean_top_conf": 0.0,
            "any_detection_rate": 0.0,
            "mean_runtime_ms": 0.0,
        }
    target_confs = sorted([_safe_float(r.get("target_conf")) for r in rows])
    top_confs = [_safe_float(r.get("top_conf")) for r in rows]
    runtimes = [_safe_float(r.get("runtime_ms")) for r in rows]
    target_missing_rate = sum(1 for r in rows if bool(r.get("target_missing", False))) / n
    top_misclass_rate = sum(1 for r in rows if bool(r.get("top_misclass", False))) / n
    detected_rows = [r for r in rows if _safe_float(r.get("target_conf")) > 0.0]
    mean_target_conf_when_detected = (
        sum(_safe_float(r.get("target_conf")) for r in detected_rows) / len(detected_rows)
        if detected_rows else 0.0
    )
    target_detect_rate = sum(1 for v in target_confs if v > 0.0) / n
    any_detection_rate = sum(1 for r in rows if int(r.get("num_detections", 0)) > 0) / n
    mid = n // 2
    median = target_confs[mid] if n % 2 == 1 else 0.5 * (target_confs[mid - 1] + target_confs[mid])
    return {
        "images": n,
        "mean_target_conf": sum(target_confs) / n,
        "median_target_conf": median,
        "target_detect_rate": target_detect_rate,
        "target_missing_rate": target_missing_rate,
        "top_misclass_rate": top_misclass_rate,
        "mean_target_conf_when_detected": mean_target_conf_when_detected,
        "mean_top_conf": sum(top_confs) / n,
        "any_detection_rate": any_detection_rate,
        "mean_runtime_ms": sum(runtimes) / n,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Compare real images across six detector backends.")
    ap.add_argument("--input", required=True, help="Path to a single image or a folder of images")
    ap.add_argument("--recursive", action="store_true", help="Recursively scan folder input")
    ap.add_argument("--device", default="auto", help="Detector device: cpu/cuda/auto")
    ap.add_argument("--conf", type=float, default=0.10, help="Detection confidence threshold")
    ap.add_argument("--iou", type=float, default=0.45, help="NMS/IoU threshold where applicable")
    ap.add_argument("--target-class", default="stop sign", help="Target class label or id")
    ap.add_argument("--yolo8-weights", default="", help="Override YOLOv8 weights path")
    ap.add_argument("--yolo11-weights", default="", help="Override YOLO11 weights path")
    ap.add_argument("--only", default="", help="Optional subset by names (comma-separated), e.g. yolo8,fcos,rtdetr_r50vd")
    ap.add_argument("--save-overlays", action="store_true", help="Save annotated overlay images per detector")
    ap.add_argument("--overlay-dir", default="_runs/real_detector_compare_overlays", help="Directory for annotated overlays")
    ap.add_argument("--overlay-topk", type=int, default=20, help="Max boxes to draw in overlays")
    ap.add_argument("--out-json", default="_runs/paper_data/real_detector_compare/results.json", help="Output JSON path")
    ap.add_argument("--out-csv", default="_runs/paper_data/real_detector_compare/results.csv", help="Output CSV path")
    ap.add_argument("--skip-csv", action="store_true", help="Do not write CSV")
    args = ap.parse_args()

    inp = Path(args.input)
    images = _collect_images(inp, recursive=bool(args.recursive))
    if not images:
        raise FileNotFoundError(f"No images found under: {inp}")

    has_heif = any(p.suffix.lower() in {".heic", ".heif"} for p in images)
    heif_enabled = False
    heif_err = None
    if has_heif:
        heif_enabled, heif_err = _maybe_enable_heif_support()
        if not heif_enabled:
            print(
                "[WARN] HEIC/HEIF files detected, but HEIC support is not enabled in Pillow. "
                "Install with: pip install pillow-heif"
            )

    specs = _build_specs(args)
    if not specs:
        raise ValueError("No detector specs selected. Check --only values.")

    print(f"[COMPARE] images={len(images)} detectors={len(specs)} device={args.device}")
    print("[COMPARE] detector set:", ", ".join(s.name for s in specs))

    detectors: dict[str, Any] = {}
    for spec in specs:
        print(f"[LOAD] {spec.name} ...")
        det = build_detector(
            detector_type=spec.detector_type,
            detector_model=spec.detector_model,
            yolo_weights=spec.yolo_weights,
            device=args.device,
            conf=float(args.conf),
            iou=float(args.iou),
            target_class=args.target_class,
            debug=False,
        )
        detectors[spec.name] = det

    per_image_results: list[dict[str, Any]] = []

    for img_path in images:
        try:
            pil = Image.open(img_path).convert("RGB")
        except Exception as e:
            if img_path.suffix.lower() in {".heic", ".heif"} and not heif_enabled:
                print(
                    f"[WARN] Failed to open {img_path.name} (HEIC). "
                    f"Install HEIC support: pip install pillow-heif | detail={heif_err}"
                )
            else:
                print(f"[WARN] Failed to open {img_path}: {e}")
            continue

        image_record = {
            "image_path": str(img_path),
            "width": pil.width,
            "height": pil.height,
            "detectors": {},
        }
        print(f"[IMG] {img_path.name}")

        for spec in specs:
            det = detectors[spec.name]
            t0 = time.perf_counter()
            out = det.infer_detections_batch([pil])[0]
            dt_ms = (time.perf_counter() - t0) * 1000.0

            boxes = out.get("boxes", []) or []
            confs = out.get("confs", []) or []
            clss = out.get("clss", []) or []
            target_conf = _safe_float(out.get("target_conf"))
            top_conf = _safe_float(out.get("top_conf"))
            top_class = out.get("top_class", None)
            target_id = getattr(det, "target_id", None)
            top_is_target = (top_class is not None and target_id is not None and int(top_class) == int(target_id))
            target_missing = (target_conf <= 0.0)
            # Proxy misclassification for stop-sign images: detector fires but top class is not target.
            top_misclass = (len(confs) > 0 and not top_is_target)
            row = {
                "detector_name": spec.name,
                "detector_type": spec.detector_type,
                "detector_model": spec.detector_model,
                "target_id": target_id,
                "target_conf": target_conf,
                "target_box": out.get("target_box"),
                "top_conf": top_conf,
                "top_class": top_class,
                "top_label": _label_for(det, top_class) if top_class is not None else "",
                "top_box": out.get("top_box"),
                "top_is_target": bool(top_is_target),
                "target_missing": bool(target_missing),
                "top_misclass": bool(top_misclass),
                "num_detections": len(confs),
                "runtime_ms": dt_ms,
                # keep full detections for downstream analysis/reproducibility
                "boxes": boxes,
                "confs": confs,
                "clss": clss,
            }
            image_record["detectors"][spec.name] = row
            print(
                f"  - {spec.name:<14} target={target_conf:.4f} top={top_conf:.4f} "
                f"n={len(confs):<3d} miss={int(top_misclass)} t={dt_ms:7.1f} ms"
            )

            if args.save_overlays:
                out_dir = Path(args.overlay_dir) / spec.name
                overlay_name = f"{img_path.stem}_{spec.name}{img_path.suffix}"
                _draw_boxes(pil.copy(), boxes, confs, clss, det, out_dir / overlay_name, topk=int(args.overlay_topk))

        per_image_results.append(image_record)

    # Aggregate summaries
    detector_rows: dict[str, list[dict[str, Any]]] = {s.name: [] for s in specs}
    for rec in per_image_results:
        for name, row in rec["detectors"].items():
            detector_rows[name].append(row)
    summary = {name: _summarize_detector_rows(rows) for name, rows in detector_rows.items()}

    payload = {
        "meta": {
            "input": str(inp),
            "recursive": bool(args.recursive),
            "num_images": len(per_image_results),
            "device": args.device,
            "conf": float(args.conf),
            "iou": float(args.iou),
            "target_class": args.target_class,
            "detectors": [asdict(s) for s in specs],
            "ssd_removed": True,
        },
        "summary": summary,
        "images": per_image_results,
    }

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[SAVE] json={out_json}")

    if not args.skip_csv:
        out_csv = Path(args.out_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "image_path",
                "detector_name",
                "detector_type",
                "detector_model",
                "target_conf",
                "top_conf",
                "top_class",
                "top_label",
                "top_is_target",
                "target_missing",
                "top_misclass",
                "num_detections",
                "runtime_ms",
                "target_box",
                "top_box",
            ])
            for rec in per_image_results:
                for name, row in rec["detectors"].items():
                    writer.writerow([
                        rec["image_path"],
                        name,
                        row.get("detector_type", ""),
                        row.get("detector_model", ""),
                        row.get("target_conf", 0.0),
                        row.get("top_conf", 0.0),
                        row.get("top_class", ""),
                        row.get("top_label", ""),
                        row.get("top_is_target", False),
                        row.get("target_missing", False),
                        row.get("top_misclass", False),
                        row.get("num_detections", 0),
                        row.get("runtime_ms", 0.0),
                        json.dumps(row.get("target_box")),
                        json.dumps(row.get("top_box")),
                    ])
        print(f"[SAVE] csv={out_csv}")

    print("[SUMMARY]")
    for name, s in summary.items():
        print(
            f"  {name:<14} mean_target={s['mean_target_conf']:.4f} "
            f"detect_rate={s['target_detect_rate']:.2%} top_miscls={s['top_misclass_rate']:.2%} "
            f"mean_runtime={s['mean_runtime_ms']:.1f} ms"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
