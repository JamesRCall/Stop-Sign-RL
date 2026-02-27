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

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from detectors.factory import build_detector  # noqa: E402


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
EXTRA_CAMERA_EXTS = {".heic", ".heif", ".tif", ".tiff"}
IMAGE_EXTS = IMAGE_EXTS | EXTRA_CAMERA_EXTS
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}


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


def _is_image(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTS


def _is_video(path: Path) -> bool:
    return path.suffix.lower() in VIDEO_EXTS


def _collect_media(inp: Path, recursive: bool, include_videos: bool) -> list[Path]:
    valid_exts = set(IMAGE_EXTS)
    if include_videos:
        valid_exts |= VIDEO_EXTS
    if inp.is_file():
        if inp.suffix.lower() not in valid_exts:
            kind = "image/video" if include_videos else "image"
            raise ValueError(f"Unsupported {kind} extension: {inp}")
        return [inp]
    if not inp.is_dir():
        raise FileNotFoundError(f"Input not found: {inp}")
    it = inp.rglob("*") if recursive else inp.iterdir()
    files = [p for p in it if p.is_file() and p.suffix.lower() in valid_exts]
    return sorted(files)


def _iter_video_frames(
    video_path: Path,
    frame_step: int,
    max_frames: int,
    start_sec: float,
    end_sec: float,
):
    """Yield sampled PIL RGB frames from a video."""
    try:
        import cv2  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            f"OpenCV is required for video support. Install with: pip install opencv-python. Error: {e}"
        )

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if fps <= 0:
        fps = 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    start_idx = max(0, int(round(start_sec * fps)))
    end_idx = (total_frames - 1) if end_sec <= 0 else int(round(end_sec * fps))
    if total_frames > 0:
        end_idx = min(end_idx, total_frames - 1)
    if end_idx < start_idx:
        cap.release()
        return

    step = max(1, int(frame_step))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_idx)
    idx = start_idx
    yielded = 0
    while True:
        if idx > end_idx:
            break
        ok, frame_bgr = cap.read()
        if not ok or frame_bgr is None:
            break
        if ((idx - start_idx) % step) == 0:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(np.asarray(frame_rgb), mode="RGB")
            ts = float(idx / fps) if fps > 0 else 0.0
            yield idx, ts, pil
            yielded += 1
            if max_frames > 0 and yielded >= int(max_frames):
                break
        idx += 1
    cap.release()


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


def _mean(vals: Iterable[float]) -> float:
    vals = [float(v) for v in vals]
    return (sum(vals) / len(vals)) if vals else 0.0


def _std(vals: Iterable[float]) -> float:
    vals = [float(v) for v in vals]
    if not vals:
        return 0.0
    mu = sum(vals) / len(vals)
    return (sum((v - mu) ** 2 for v in vals) / len(vals)) ** 0.5


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
    ap = argparse.ArgumentParser(description="Compare real images/videos across six detector backends.")
    ap.add_argument("--input", required=True, help="Path to a single image/video or a folder")
    ap.add_argument("--recursive", action="store_true", help="Recursively scan folder input")
    ap.add_argument("--include-videos", action="store_true",
                    help="Also process videos (*.mp4/*.mov/...); runs detector on sampled frames.")
    ap.add_argument("--video-frame-step", type=int, default=30,
                    help="Sample every Nth frame for videos (default: 30).")
    ap.add_argument("--video-max-frames", type=int, default=0,
                    help="Max sampled frames per video (0 = no limit).")
    ap.add_argument("--video-start-sec", type=float, default=0.0,
                    help="Start time (seconds) for video sampling.")
    ap.add_argument("--video-end-sec", type=float, default=0.0,
                    help="End time (seconds) for video sampling (<=0 means full video).")
    ap.add_argument("--device", default="auto", help="Detector device: cpu/cuda/auto")
    ap.add_argument("--conf", type=float, default=0.10, help="Detection confidence threshold")
    ap.add_argument("--iou", type=float, default=0.45, help="NMS/IoU threshold where applicable")
    ap.add_argument("--repeats", type=int, default=1,
                    help="Run each detector on each image multiple times and average metrics/runtime (default: 1).")
    ap.add_argument("--warmup-runs", type=int, default=1,
                    help="Warmup inference runs per detector before timed evaluation loop (default: 1).")
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
    include_videos = bool(args.include_videos) or _is_video(inp)
    media = _collect_media(inp, recursive=bool(args.recursive), include_videos=include_videos)
    if not media:
        kind = "media" if include_videos else "images"
        raise FileNotFoundError(f"No {kind} found under: {inp}")

    has_heif = any(p.suffix.lower() in {".heic", ".heif"} for p in media if _is_image(p))
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

    n_images = sum(1 for p in media if _is_image(p))
    n_videos = sum(1 for p in media if _is_video(p))
    print(f"[COMPARE] media={len(media)} (images={n_images}, videos={n_videos}) detectors={len(specs)} device={args.device}")
    print("[COMPARE] detector set:", ", ".join(s.name for s in specs))
    print(f"[COMPARE] repeats={int(max(1, args.repeats))} warmup_runs={int(max(0, args.warmup_runs))}")
    if include_videos:
        print(
            f"[COMPARE] video sampling: step={int(args.video_frame_step)} "
            f"max_frames={int(args.video_max_frames)} start_sec={float(args.video_start_sec):.2f} "
            f"end_sec={float(args.video_end_sec):.2f}"
        )

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

    repeats = int(max(1, args.repeats))
    warmup_runs = int(max(0, args.warmup_runs))

    per_image_results: list[dict[str, Any]] = []
    processed_image_files = 0
    processed_video_files = 0
    sampled_video_frames = 0

    def _process_sample(
        pil: Image.Image,
        image_path: str,
        source_type: str,
        source_path: str,
        overlay_stem: str,
        overlay_suffix: str,
        frame_index: int | None = None,
        frame_time_sec: float | None = None,
    ) -> None:
        image_record = {
            "image_path": image_path,
            "source_type": source_type,
            "source_path": source_path,
            "frame_index": frame_index,
            "frame_time_sec": frame_time_sec,
            "width": pil.width,
            "height": pil.height,
            "detectors": {},
        }
        print(f"[SAMPLE] {image_path}")

        for spec in specs:
            det = detectors[spec.name]
            # Optional warmup helps runtime measurement stability (especially first call/JIT/cache effects).
            for _ in range(warmup_runs):
                _ = det.infer_detections_batch([pil])[0]

            repeat_records: list[dict[str, Any]] = []
            first_out = None
            for rep in range(repeats):
                t0 = time.perf_counter()
                out_rep = det.infer_detections_batch([pil])[0]
                dt_ms_rep = (time.perf_counter() - t0) * 1000.0
                if first_out is None:
                    first_out = out_rep

                rep_target_conf = _safe_float(out_rep.get("target_conf"))
                rep_top_conf = _safe_float(out_rep.get("top_conf"))
                rep_top_class = out_rep.get("top_class", None)
                rep_target_id = getattr(det, "target_id", None)
                rep_top_is_target = (
                    rep_top_class is not None and rep_target_id is not None and int(rep_top_class) == int(rep_target_id)
                )
                rep_confs = out_rep.get("confs", []) or []
                repeat_records.append({
                    "repeat_index": int(rep),
                    "target_conf": rep_target_conf,
                    "top_conf": rep_top_conf,
                    "top_class": rep_top_class,
                    "top_label": _label_for(det, rep_top_class) if rep_top_class is not None else "",
                    "top_is_target": bool(rep_top_is_target),
                    "target_missing": bool(rep_target_conf <= 0.0),
                    "top_misclass": bool(len(rep_confs) > 0 and not rep_top_is_target),
                    "num_detections": int(len(rep_confs)),
                    "runtime_ms": dt_ms_rep,
                })

            out = first_out if first_out is not None else {}
            dt_ms = _mean(r["runtime_ms"] for r in repeat_records)

            boxes = out.get("boxes", []) or []
            confs = out.get("confs", []) or []
            clss = out.get("clss", []) or []
            target_conf = _mean(r["target_conf"] for r in repeat_records)
            top_conf = _mean(r["top_conf"] for r in repeat_records)
            target_conf_std = _std(r["target_conf"] for r in repeat_records)
            top_conf_std = _std(r["top_conf"] for r in repeat_records)
            runtime_ms_std = _std(r["runtime_ms"] for r in repeat_records)
            top_class = out.get("top_class", None)
            target_id = getattr(det, "target_id", None)
            top_is_target = (top_class is not None and target_id is not None and int(top_class) == int(target_id))
            target_missing = (target_conf <= 0.0)
            # Proxy misclassification for stop-sign images: detector fires but top class is not target.
            top_misclass = bool(any(bool(r["top_misclass"]) for r in repeat_records))
            row = {
                "detector_name": spec.name,
                "detector_type": spec.detector_type,
                "detector_model": spec.detector_model,
                "target_id": target_id,
                "target_conf": target_conf,
                "target_conf_std": target_conf_std,
                "target_box": out.get("target_box"),
                "top_conf": top_conf,
                "top_conf_std": top_conf_std,
                "top_class": top_class,
                "top_label": _label_for(det, top_class) if top_class is not None else "",
                "top_box": out.get("top_box"),
                "top_is_target": bool(top_is_target),
                "target_missing": bool(target_missing),
                "top_misclass": bool(top_misclass),
                "num_detections": len(confs),
                "runtime_ms": dt_ms,
                "runtime_ms_std": runtime_ms_std,
                "repeats": repeats,
                "warmup_runs": warmup_runs,
                "repeat_records": repeat_records,
                # keep full detections for downstream analysis/reproducibility
                "boxes": boxes,
                "confs": confs,
                "clss": clss,
            }
            image_record["detectors"][spec.name] = row
            print(
                f"  - {spec.name:<14} target={target_conf:.4f}+/-{target_conf_std:.4f} "
                f"top={top_conf:.4f}+/-{top_conf_std:.4f} n={len(confs):<3d} "
                f"mis={int(top_misclass)} t={dt_ms:7.1f}+/-{runtime_ms_std:6.1f} ms"
            )

            if args.save_overlays:
                out_dir = Path(args.overlay_dir) / spec.name
                overlay_name = f"{overlay_stem}_{spec.name}{overlay_suffix}"
                _draw_boxes(pil.copy(), boxes, confs, clss, det, out_dir / overlay_name, topk=int(args.overlay_topk))

        per_image_results.append(image_record)

    for media_path in media:
        if _is_image(media_path):
            try:
                pil = Image.open(media_path).convert("RGB")
            except Exception as e:
                if media_path.suffix.lower() in {".heic", ".heif"} and not heif_enabled:
                    print(
                        f"[WARN] Failed to open {media_path.name} (HEIC). "
                        f"Install HEIC support: pip install pillow-heif | detail={heif_err}"
                    )
                else:
                    print(f"[WARN] Failed to open {media_path}: {e}")
                continue
            processed_image_files += 1
            _process_sample(
                pil=pil,
                image_path=str(media_path),
                source_type="image",
                source_path=str(media_path),
                overlay_stem=media_path.stem,
                overlay_suffix=media_path.suffix,
            )
            continue

        if _is_video(media_path):
            processed_video_files += 1
            print(f"[VIDEO] {media_path.name}")
            try:
                for frame_idx, frame_ts, frame_pil in _iter_video_frames(
                    video_path=media_path,
                    frame_step=int(args.video_frame_step),
                    max_frames=int(args.video_max_frames),
                    start_sec=float(args.video_start_sec),
                    end_sec=float(args.video_end_sec),
                ):
                    sampled_video_frames += 1
                    _process_sample(
                        pil=frame_pil,
                        image_path=f"{media_path}::frame_{int(frame_idx):06d}",
                        source_type="video",
                        source_path=str(media_path),
                        overlay_stem=f"{media_path.stem}_f{int(frame_idx):06d}",
                        overlay_suffix=".png",
                        frame_index=int(frame_idx),
                        frame_time_sec=float(frame_ts),
                    )
            except Exception as e:
                print(f"[WARN] Failed to process video {media_path}: {e}")
            continue

        print(f"[WARN] Skipping unsupported file: {media_path}")

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
            "num_images": len(per_image_results),  # kept for backward compatibility
            "num_records": len(per_image_results),
            "num_image_files": int(processed_image_files),
            "num_video_files": int(processed_video_files),
            "num_video_frames_sampled": int(sampled_video_frames),
            "include_videos": bool(include_videos),
            "video_frame_step": int(args.video_frame_step),
            "video_max_frames": int(args.video_max_frames),
            "video_start_sec": float(args.video_start_sec),
            "video_end_sec": float(args.video_end_sec),
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
                "source_type",
                "source_path",
                "frame_index",
                "frame_time_sec",
                "detector_name",
                "detector_type",
                "detector_model",
                "target_conf",
                "target_conf_std",
                "top_conf",
                "top_conf_std",
                "top_class",
                "top_label",
                "top_is_target",
                "target_missing",
                "top_misclass",
                "num_detections",
                "runtime_ms",
                "runtime_ms_std",
                "repeats",
                "target_box",
                "top_box",
            ])
            for rec in per_image_results:
                for name, row in rec["detectors"].items():
                    writer.writerow([
                        rec["image_path"],
                        rec.get("source_type", "image"),
                        rec.get("source_path", rec["image_path"]),
                        rec.get("frame_index", ""),
                        rec.get("frame_time_sec", ""),
                        name,
                        row.get("detector_type", ""),
                        row.get("detector_model", ""),
                        row.get("target_conf", 0.0),
                        row.get("target_conf_std", 0.0),
                        row.get("top_conf", 0.0),
                        row.get("top_conf_std", 0.0),
                        row.get("top_class", ""),
                        row.get("top_label", ""),
                        row.get("top_is_target", False),
                        row.get("target_missing", False),
                        row.get("top_misclass", False),
                        row.get("num_detections", 0),
                        row.get("runtime_ms", 0.0),
                        row.get("runtime_ms_std", 0.0),
                        row.get("repeats", 1),
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
