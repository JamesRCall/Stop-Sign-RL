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
import re
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
    DetectorSpec("yolo8", "yolo", yolo_weights="weights/yolov8n.pt"),
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


def _parse_float_list_csv(text: str) -> list[float]:
    vals: list[float] = []
    for tok in str(text).split(","):
        tok = tok.strip()
        if not tok:
            continue
        vals.append(float(tok))
    return vals


def _parse_distance_range(text: str) -> tuple[float, float]:
    vals = _parse_float_list_csv(text)
    if len(vals) != 2:
        raise ValueError("--distance-range-m must be exactly two comma-separated numbers, e.g. 20,0")
    return float(vals[0]), float(vals[1])


def _assign_distance_bin(distance_m: float | None, edges: list[float]) -> str | None:
    if distance_m is None:
        return None
    if not edges or len(edges) < 2:
        return None
    x = float(distance_m)
    lo0 = float(min(edges))
    hi0 = float(max(edges))
    x = max(lo0, min(hi0, x))
    for i in range(len(edges) - 1):
        lo = float(edges[i])
        hi = float(edges[i + 1])
        if i < len(edges) - 2:
            if lo <= x < hi:
                return f"{int(lo)}-{int(hi)}m"
        else:
            if lo <= x <= hi:
                return f"{int(lo)}-{int(hi)}m"
    return None


def _parse_media_name_meta(source_path: str) -> dict[str, Any]:
    stem = Path(str(source_path)).stem.lower()
    toks = [t for t in re.split(r"[^a-z0-9]+", stem) if t]

    def has_any(opts: set[str]) -> bool:
        return any(t in opts for t in toks)

    video_type = "unknown"
    if has_any({"base", "baseline"}):
        video_type = "base"
    elif has_any({"spray"}):
        video_type = "spray"
    elif has_any({"paint"}):
        video_type = "paint"

    daynight = "unknown"
    if has_any({"day"}):
        daynight = "day"
    elif has_any({"night"}):
        daynight = "night"

    uvnorm = "unknown"
    if has_any({"uv"}):
        uvnorm = "uv"
    elif has_any({"norm", "normal"}):
        uvnorm = "norm"
    elif has_any({"baseline", "base"}):
        uvnorm = "norm"

    # Distance parsing:
    #  - "05m_*", "5m_*", "..._10m_..."
    #  - "5-20m_*" -> uses first number as the labeled capture distance.
    distance_m = None
    m_range = re.search(r"(\d+)\s*-\s*(\d+)\s*m", stem)
    if m_range:
        try:
            distance_m = float(m_range.group(1))
        except Exception:
            distance_m = None
    if distance_m is None:
        m_single = re.search(r"(\d+)\s*m\b", stem)
        if m_single:
            try:
                distance_m = float(m_single.group(1))
            except Exception:
                distance_m = None

    trial = None
    numeric_toks = [t for t in toks if t.isdigit()]
    if numeric_toks:
        try:
            trial = int(numeric_toks[-1])
        except Exception:
            trial = None

    parse_ok = (
        video_type != "unknown"
        and daynight != "unknown"
        and uvnorm != "unknown"
        and trial is not None
    )
    condition_key = f"{video_type}_{daynight}_{uvnorm}"
    trial_key = f"{condition_key}_t{trial}" if trial is not None else f"{condition_key}_t?"
    return {
        "video_type": video_type,
        "daynight": daynight,
        "uvnorm": uvnorm,
        "distance_m": distance_m,
        "trial": trial,
        "condition_key": condition_key,
        "trial_key": trial_key,
        "parse_ok": bool(parse_ok),
    }


def _aggregate_condition_summaries(
    by_video: dict[str, Any],
    detector_names: list[str],
    bin_labels: list[str],
) -> dict[str, Any]:
    """Aggregate per-video summaries into mean/std over numbered trials."""
    grouped: dict[str, list[dict[str, Any]]] = {}
    for _video_id, v in by_video.items():
        cond = str(v.get("condition_key", "unknown_unknown_unknown"))
        grouped.setdefault(cond, []).append(v)

    out: dict[str, Any] = {}
    for cond, vids in grouped.items():
        cond_payload: dict[str, Any] = {
            "n_trials": len(vids),
            "videos": [str(v.get("video_id", "")) for v in vids],
            "detectors": {},
        }
        for det_name in detector_names:
            ov_rows = []
            bin_rows: dict[str, list[dict[str, Any]]] = {b: [] for b in bin_labels}
            for v in vids:
                d = (v.get("detectors", {}) or {}).get(det_name)
                if not isinstance(d, dict):
                    continue
                if isinstance(d.get("overall"), dict):
                    ov_rows.append(d["overall"])
                bins_obj = d.get("bins", {}) or {}
                for b in bin_labels:
                    if isinstance(bins_obj.get(b), dict):
                        bin_rows[b].append(bins_obj[b])

            def _ms(rows: list[dict[str, Any]], key: str) -> tuple[float, float]:
                vals = [_safe_float(r.get(key)) for r in rows if r is not None]
                return _mean(vals), _std(vals)

            if not ov_rows and not any(bin_rows.values()):
                continue

            det_payload: dict[str, Any] = {}
            if ov_rows:
                mu_conf, sd_conf = _ms(ov_rows, "mean_target_conf")
                mu_det, sd_det = _ms(ov_rows, "target_detect_rate")
                mu_mis, sd_mis = _ms(ov_rows, "top_misclass_rate")
                det_payload["overall_mean_over_trials"] = {
                    "mean_target_conf": mu_conf,
                    "std_target_conf": sd_conf,
                    "target_detect_rate": mu_det,
                    "std_detect_rate": sd_det,
                    "top_misclass_rate": mu_mis,
                    "std_top_misclass_rate": sd_mis,
                    "n_trials": len(ov_rows),
                }

            bins_payload: dict[str, Any] = {}
            for b in bin_labels:
                rows = bin_rows.get(b, [])
                if not rows:
                    continue
                mu_conf, sd_conf = _ms(rows, "mean_target_conf")
                mu_det, sd_det = _ms(rows, "target_detect_rate")
                mu_mis, sd_mis = _ms(rows, "top_misclass_rate")
                bins_payload[b] = {
                    "mean_target_conf": mu_conf,
                    "std_target_conf": sd_conf,
                    "target_detect_rate": mu_det,
                    "std_detect_rate": sd_det,
                    "top_misclass_rate": mu_mis,
                    "std_top_misclass_rate": sd_mis,
                    "n_trials": len(rows),
                }
            det_payload["distance_bins_mean_over_trials"] = bins_payload
            cond_payload["detectors"][det_name] = det_payload
        out[cond] = cond_payload
    return out


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
    ap.add_argument("--distance-range-m", default="20,0",
                    help="Video distance mapping range as start,end in meters (default: 20,0).")
    ap.add_argument("--distance-bins-m", default="0,5,10,15,20",
                    help="Distance bin edges in meters for grouped JSON (default: 0,5,10,15,20).")
    ap.add_argument("--speed-mph", type=float, default=0.0,
                    help="If >0, estimate traveled distance per sampled video frame using this speed.")
    ap.add_argument("--distance-fps", type=float, default=0.0,
                    help="If >0, use this FPS for distance estimate (e.g., 30). Otherwise use video timestamps.")
    ap.add_argument("--initial-distance-m", type=float, default=0.0,
                    help="If >0, also report estimated remaining distance to sign: max(0, initial - traveled).")
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

    dist_start_m, dist_end_m = _parse_distance_range(args.distance_range_m)
    dist_edges = sorted(set(_parse_float_list_csv(args.distance_bins_m)))
    if len(dist_edges) < 2:
        raise ValueError("--distance-bins-m must include at least two edges.")
    bin_labels = [
        f"{int(dist_edges[i])}-{int(dist_edges[i + 1])}m"
        for i in range(len(dist_edges) - 1)
    ]

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
    print(
        "[VALIDITY] top_misclass is a legacy full-frame proxy only. Without an "
        "annotated sign ROI and paired clean/active records, it is not a valid "
        "misclassification or joint-ASR measurement."
    )
    if include_videos:
        print(
            f"[COMPARE] video sampling: step={int(args.video_frame_step)} "
            f"max_frames={int(args.video_max_frames)} start_sec={float(args.video_start_sec):.2f} "
            f"end_sec={float(args.video_end_sec):.2f}"
        )
        print(
            f"[COMPARE] quartile distance mapping: start_m={dist_start_m:.2f} "
            f"end_m={dist_end_m:.2f} bins={','.join(bin_labels)}"
        )
        if float(args.speed_mph) > 0.0:
            fps_mode = f"{float(args.distance_fps):.2f} (override)" if float(args.distance_fps) > 0.0 else "video timestamps"
            print(
                f"[COMPARE] distance est: speed_mph={float(args.speed_mph):.3f} "
                f"fps_mode={fps_mode} initial_distance_m={float(args.initial_distance_m):.3f}"
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
        distance_elapsed_sec: float | None = None,
        distance_traveled_m: float | None = None,
        distance_traveled_ft: float | None = None,
        distance_to_sign_m: float | None = None,
        distance_to_sign_ft: float | None = None,
        video_progress_0_1: float | None = None,
        distance_est_m_quarter: float | None = None,
        distance_bin_m: str | None = None,
        video_name_meta: dict[str, Any] | None = None,
    ) -> None:
        if video_name_meta is None:
            video_name_meta = {}
        image_record = {
            "image_path": image_path,
            "source_type": source_type,
            "source_path": source_path,
            "frame_index": frame_index,
            "frame_time_sec": frame_time_sec,
            "distance_elapsed_sec": distance_elapsed_sec,
            "distance_traveled_m": distance_traveled_m,
            "distance_traveled_ft": distance_traveled_ft,
            "distance_to_sign_m": distance_to_sign_m,
            "distance_to_sign_ft": distance_to_sign_ft,
            "video_progress_0_1": video_progress_0_1,
            "distance_est_m_quarter": distance_est_m_quarter,
            "distance_bin_m": distance_bin_m,
            "video_name_meta": video_name_meta,
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
            mmeta = _parse_media_name_meta(str(media_path))
            named_dist_m = mmeta.get("distance_m", None)
            named_bin = _assign_distance_bin(named_dist_m, dist_edges)
            _process_sample(
                pil=pil,
                image_path=str(media_path),
                source_type="image",
                source_path=str(media_path),
                overlay_stem=media_path.stem,
                overlay_suffix=media_path.suffix,
                distance_to_sign_m=named_dist_m,
                distance_to_sign_ft=(None if named_dist_m is None else float(named_dist_m) * 3.280839895),
                distance_est_m_quarter=named_dist_m,
                distance_bin_m=named_bin,
                video_name_meta=mmeta,
            )
            continue

        if _is_video(media_path):
            processed_video_files += 1
            print(f"[VIDEO] {media_path.name}")
            try:
                vmeta = _parse_media_name_meta(str(media_path))
                frames = list(_iter_video_frames(
                    video_path=media_path,
                    frame_step=int(args.video_frame_step),
                    max_frames=int(args.video_max_frames),
                    start_sec=float(args.video_start_sec),
                    end_sec=float(args.video_end_sec),
                ))
                n_frames = len(frames)
                if n_frames == 0:
                    print(f"[WARN] No sampled frames for video: {media_path}")
                    continue
                for order_idx, (frame_idx, frame_ts, frame_pil) in enumerate(frames):
                    sampled_video_frames += 1
                    d_elapsed_sec = None
                    d_traveled_m = None
                    d_traveled_ft = None
                    d_to_sign_m = None
                    d_to_sign_ft = None
                    progress_0_1 = 0.0 if n_frames <= 1 else float(order_idx) / float(n_frames - 1)
                    d_est_quarter = float(dist_start_m + (dist_end_m - dist_start_m) * progress_0_1)
                    d_bin_quarter = _assign_distance_bin(d_est_quarter, dist_edges)
                    speed_mph = float(args.speed_mph)
                    if speed_mph > 0.0:
                        if float(args.distance_fps) > 0.0:
                            d_fps = float(args.distance_fps)
                            start_idx_d = int(round(float(args.video_start_sec) * d_fps))
                            d_elapsed_sec = max(0.0, (float(frame_idx) - float(start_idx_d)) / d_fps)
                        else:
                            d_elapsed_sec = max(0.0, float(frame_ts) - float(args.video_start_sec))
                        speed_mps = speed_mph * 0.44704
                        d_traveled_m = d_elapsed_sec * speed_mps
                        d_traveled_ft = d_traveled_m * 3.280839895
                        if float(args.initial_distance_m) > 0.0:
                            d_to_sign_m = max(0.0, float(args.initial_distance_m) - d_traveled_m)
                            d_to_sign_ft = d_to_sign_m * 3.280839895
                    _process_sample(
                        pil=frame_pil,
                        image_path=f"{media_path}::frame_{int(frame_idx):06d}",
                        source_type="video",
                        source_path=str(media_path),
                        overlay_stem=f"{media_path.stem}_f{int(frame_idx):06d}",
                        overlay_suffix=".png",
                        frame_index=int(frame_idx),
                        frame_time_sec=float(frame_ts),
                        distance_elapsed_sec=d_elapsed_sec,
                        distance_traveled_m=d_traveled_m,
                        distance_traveled_ft=d_traveled_ft,
                        distance_to_sign_m=d_to_sign_m,
                        distance_to_sign_ft=d_to_sign_ft,
                        video_progress_0_1=progress_0_1,
                        distance_est_m_quarter=d_est_quarter,
                        distance_bin_m=d_bin_quarter,
                        video_name_meta=vmeta,
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

    # Distance-binned summaries over named captures (videos + images)
    video_distance_grouped: dict[str, Any] = {
        "distance_mapping": {
            "method": "video_quartile_progress_linear + image_named_distance",
            "range_start_m": float(dist_start_m),
            "range_end_m": float(dist_end_m),
            "bins_m": dist_edges,
            "bin_labels": bin_labels,
        },
        "by_video": {},
        "by_condition": {},
    }
    named_recs = [
        r for r in per_image_results
        if str(r.get("distance_bin_m") or "") != ""
        and isinstance((r.get("video_name_meta") or {}), dict)
    ]
    if named_recs:
        by_source: dict[str, list[dict[str, Any]]] = {}
        for r in named_recs:
            src = str(r.get("source_path", ""))
            by_source.setdefault(src, []).append(r)

        by_video: dict[str, Any] = {}
        detector_names = [s.name for s in specs]
        for src, rows_v in by_source.items():
            rows_v = sorted(rows_v, key=lambda rr: int(rr.get("frame_index") or -1))
            vm = rows_v[0].get("video_name_meta", {}) if rows_v else {}
            vid = Path(src).stem
            det_payload: dict[str, Any] = {}
            for dname in detector_names:
                d_all = [rv["detectors"][dname] for rv in rows_v if dname in rv.get("detectors", {})]
                if not d_all:
                    continue
                bins_payload: dict[str, Any] = {}
                for bl in bin_labels:
                    d_bin = [
                        rv["detectors"][dname]
                        for rv in rows_v
                        if dname in rv.get("detectors", {}) and str(rv.get("distance_bin_m") or "") == bl
                    ]
                    bsum = _summarize_detector_rows(d_bin)
                    bsum["frames"] = len(d_bin)
                    bins_payload[bl] = bsum
                ov = _summarize_detector_rows(d_all)
                ov["frames"] = len(d_all)
                det_payload[dname] = {
                    "overall": ov,
                    "bins": bins_payload,
                }
            by_video[vid] = {
                "video_id": vid,
                "source_type": rows_v[0].get("source_type", "unknown") if rows_v else "unknown",
                "source_path": src,
                "video_type": vm.get("video_type", "unknown"),
                "daynight": vm.get("daynight", "unknown"),
                "uvnorm": vm.get("uvnorm", "unknown"),
                "distance_m": vm.get("distance_m", None),
                "trial": vm.get("trial", None),
                "condition_key": vm.get("condition_key", "unknown_unknown_unknown"),
                "trial_key": vm.get("trial_key", "unknown_unknown_unknown_t?"),
                "parse_ok": bool(vm.get("parse_ok", False)),
                "num_sampled_frames": len(rows_v),
                "detectors": det_payload,
            }
        video_distance_grouped["by_video"] = by_video
        video_distance_grouped["by_condition"] = _aggregate_condition_summaries(
            by_video=by_video,
            detector_names=[s.name for s in specs],
            bin_labels=bin_labels,
        )

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
            "distance_range_m": [float(dist_start_m), float(dist_end_m)],
            "distance_bins_m": [float(x) for x in dist_edges],
            "speed_mph": float(args.speed_mph),
            "distance_fps": float(args.distance_fps),
            "initial_distance_m": float(args.initial_distance_m),
            "device": args.device,
            "conf": float(args.conf),
            "iou": float(args.iou),
            "target_class": args.target_class,
            "detectors": [asdict(s) for s in specs],
            "ssd_removed": True,
            "metric_validity": {
                "top_misclass": "legacy_global_top_proxy_not_attack_success",
                "roi_localized": False,
                "paired_clean_active": False,
                "usable_for_joint_asr": False,
            },
        },
        "summary": summary,
        "video_distance_grouped": video_distance_grouped,
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
                "distance_elapsed_sec",
                "distance_traveled_m",
                "distance_traveled_ft",
                "distance_to_sign_m",
                "distance_to_sign_ft",
                "video_progress_0_1",
                "distance_est_m_quarter",
                "distance_bin_m",
                "video_type",
                "daynight",
                "uvnorm",
                "trial",
                "condition_key",
                "trial_key",
                "parse_ok",
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
                        rec.get("distance_elapsed_sec", ""),
                        rec.get("distance_traveled_m", ""),
                        rec.get("distance_traveled_ft", ""),
                        rec.get("distance_to_sign_m", ""),
                        rec.get("distance_to_sign_ft", ""),
                        rec.get("video_progress_0_1", ""),
                        rec.get("distance_est_m_quarter", ""),
                        rec.get("distance_bin_m", ""),
                        (rec.get("video_name_meta", {}) or {}).get("video_type", ""),
                        (rec.get("video_name_meta", {}) or {}).get("daynight", ""),
                        (rec.get("video_name_meta", {}) or {}).get("uvnorm", ""),
                        (rec.get("video_name_meta", {}) or {}).get("trial", ""),
                        (rec.get("video_name_meta", {}) or {}).get("condition_key", ""),
                        (rec.get("video_name_meta", {}) or {}).get("trial_key", ""),
                        (rec.get("video_name_meta", {}) or {}).get("parse_ok", ""),
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
