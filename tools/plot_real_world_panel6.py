#!/usr/bin/env python3
"""Create one 6-panel figure for real-world detector confidence curves.

Panels:
  Row 1: Base-Norm, Spray-Norm, Paint-Norm
  Row 2: Base-UV,   Spray-UV,   Paint-UV

Each panel contains 6 detector lines vs frame index (mean over matching videos/trials).
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


DET_ORDER = [
    "yolo8",
    "yolo11",
    "fasterrcnn_v2",
    "fcos",
    "retinanet_v2",
    "rtdetr_r50vd",
]
DET_LABEL = {
    "yolo8": "YOLOv8",
    "yolo11": "YOLO11",
    "fasterrcnn_v2": "Faster R-CNN v2",
    "fcos": "FCOS",
    "retinanet_v2": "RetinaNet v2",
    "rtdetr_r50vd": "RT-DETR R50",
}
COLORS = {
    "yolo8": "#1f77b4",
    "yolo11": "#ff7f0e",
    "fasterrcnn_v2": "#2ca02c",
    "fcos": "#d62728",
    "retinanet_v2": "#9467bd",
    "rtdetr_r50vd": "#8c564b",
}

TYPE_ORDER = ["base", "spray", "paint"]
UV_ORDER = ["norm", "uv"]


def _safe_float(v: Any) -> float:
    try:
        return float(v)
    except Exception:
        return float("nan")


def _mean(vals: list[float]) -> float:
    arr = [x for x in vals if not math.isnan(x)]
    return sum(arr) / len(arr) if arr else float("nan")


def _std(vals: list[float]) -> float:
    arr = [x for x in vals if not math.isnan(x)]
    if not arr:
        return float("nan")
    mu = sum(arr) / len(arr)
    return math.sqrt(sum((x - mu) ** 2 for x in arr) / len(arr))


def _parse_video_name_meta(source_path: str) -> dict[str, str]:
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
    elif has_any({"norm", "normal", "baseline", "base"}):
        uvnorm = "norm"

    return {"video_type": video_type, "daynight": daynight, "uvnorm": uvnorm}


def _collect(payload: dict[str, Any], daynight_filter: str) -> tuple[list[int], dict[str, Any]]:
    # out[(type,uv)][det][frame] -> list[conf]
    out: dict[tuple[str, str], dict[str, dict[int, list[float]]]] = {}
    for t in TYPE_ORDER:
        for u in UV_ORDER:
            out[(t, u)] = {d: {} for d in DET_ORDER}

    frames = set()
    rows = payload.get("images") or []
    for r in rows:
        if str(r.get("source_type", "")).lower() != "video":
            continue
        vm = r.get("video_name_meta")
        if not isinstance(vm, dict):
            vm = _parse_video_name_meta(str(r.get("source_path", "")))
        vtype = str(vm.get("video_type", "unknown")).lower()
        uv = str(vm.get("uvnorm", "unknown")).lower()
        dn = str(vm.get("daynight", "unknown")).lower()
        if vtype not in TYPE_ORDER or uv not in UV_ORDER:
            vm = _parse_video_name_meta(str(r.get("source_path", "")))
            vtype = str(vm.get("video_type", "unknown")).lower()
            uv = str(vm.get("uvnorm", "unknown")).lower()
            dn = str(vm.get("daynight", "unknown")).lower()
        if vtype not in TYPE_ORDER or uv not in UV_ORDER:
            continue
        if daynight_filter != "any" and dn != daynight_filter:
            continue
        try:
            fidx = int(r.get("frame_index"))
        except Exception:
            continue
        frames.add(fidx)
        dets = r.get("detectors") or {}
        for d in DET_ORDER:
            dd = dets.get(d) or {}
            c = _safe_float(dd.get("target_conf"))
            if math.isnan(c):
                continue
            out[(vtype, uv)][d].setdefault(fidx, []).append(c)
    return sorted(frames), out


def main() -> int:
    ap = argparse.ArgumentParser(description="Create one 6-panel real-world detector confidence figure.")
    ap.add_argument("--input-json", default="_runs/paper_data/real_detector_compare/real_world_videos_20to0.json")
    ap.add_argument("--out-dir", default="_runs/paper_data/real_detector_compare/plots_distance")
    ap.add_argument("--daynight", choices=["any", "day", "night"], default="any")
    ap.add_argument("--dpi", type=int, default=240)
    args = ap.parse_args()

    in_json = Path(args.input_json)
    if not in_json.exists():
        raise FileNotFoundError(f"Input JSON not found: {in_json}")
    payload = json.loads(in_json.read_text(encoding="utf-8"))

    frames, grouped = _collect(payload, daynight_filter=args.daynight)
    if not frames:
        raise ValueError("No video-frame rows found for the requested filter.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = args.daynight

    fig, axes = plt.subplots(2, 3, figsize=(18, 9), sharex=True, sharey=True)
    handles = []
    labels = []

    for r, uv in enumerate(UV_ORDER):
        for c, t in enumerate(TYPE_ORDER):
            ax = axes[r, c]
            key = (t, uv)
            for d in DET_ORDER:
                xs = []
                ys = []
                sds = []
                for f in frames:
                    vals = grouped[key][d].get(f, [])
                    if not vals:
                        continue
                    xs.append(f)
                    ys.append(_mean(vals))
                    sds.append(_std(vals))
                if not xs:
                    continue
                line, = ax.plot(
                    xs,
                    ys,
                    linewidth=1.8,
                    color=COLORS.get(d, None),
                    alpha=0.95,
                    label=DET_LABEL.get(d, d),
                )
                if r == 0 and c == 0:
                    handles.append(line)
                    labels.append(DET_LABEL.get(d, d))
                if len(xs) > 3:
                    lo = [max(0.0, y - (0.0 if math.isnan(s) else s)) for y, s in zip(ys, sds)]
                    hi = [y + (0.0 if math.isnan(s) else s) for y, s in zip(ys, sds)]
                    ax.fill_between(xs, lo, hi, color=COLORS.get(d, None), alpha=0.10, linewidth=0)

            ax.set_title(f"{t.capitalize()} - {uv.upper()}", fontsize=12, weight="bold")
            ax.grid(True, alpha=0.25, linestyle="--")
            ax.set_ylim(bottom=0.0)
            if r == 1:
                ax.set_xlabel("Frame Index")
            if c == 0:
                ax.set_ylabel("Stop-Sign Confidence")

    # Big global legend
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=6,
        fontsize=20,
        frameon=True,
        fancybox=True,
        framealpha=0.95,
        bbox_to_anchor=(0.5, 0.03),
        borderpad=0.8,
        handlelength=2.8,
        columnspacing=1.5,
    )
    fig.tight_layout(rect=[0.02, 0.14, 0.98, 0.98])

    out_png = out_dir / f"panel6_conf_vs_frame_{suffix}.png"
    out_pdf = out_dir / f"panel6_conf_vs_frame_{suffix}.pdf"
    fig.savefig(out_png, dpi=int(args.dpi))
    fig.savefig(out_pdf)
    plt.close(fig)
    print(f"[SAVE] {out_png}")
    print(f"[SAVE] {out_pdf}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
