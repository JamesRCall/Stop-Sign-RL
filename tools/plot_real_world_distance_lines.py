#!/usr/bin/env python3
"""Plot detector confidence lines from real-world video comparison JSON.

Creates 3 line plots (Base, Spray, Paint), each containing one line per detector.
Supports:
  - mode=frames: raw per-frame confidence (no quartile point estimation in plot)
  - mode=bins: distance-bin confidence from grouped JSON
"""

from __future__ import annotations

import argparse
import csv
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

TYPE_ORDER = ["base", "spray", "paint"]
TYPE_LABEL = {"base": "Base", "spray": "Spray", "paint": "Paint"}

COLORS = {
    "yolo8": "#1f77b4",
    "yolo11": "#ff7f0e",
    "fasterrcnn_v2": "#2ca02c",
    "fcos": "#d62728",
    "retinanet_v2": "#9467bd",
    "rtdetr_r50vd": "#8c564b",
}


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


def _bin_center(label: str) -> float:
    # expects labels like "0-5m"
    txt = str(label).replace("m", "")
    lo, hi = txt.split("-", 1)
    return 0.5 * (float(lo) + float(hi))


def _bin_sort_key(label: str) -> tuple[float, str]:
    try:
        return (_bin_center(label), label)
    except Exception:
        return (1e9, label)


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


def _collect_rows_bins(payload: dict[str, Any], *, daynight: str, uvnorm: str) -> tuple[list[str], dict[str, Any]]:
    grouped = (payload.get("video_distance_grouped") or {}).get("by_video") or {}
    mapping = (payload.get("video_distance_grouped") or {}).get("distance_mapping") or {}
    bin_labels = list(mapping.get("bin_labels") or [])
    if not bin_labels:
        # fallback: discover from first detector bins
        discovered = set()
        for v in grouped.values():
            dets = v.get("detectors") or {}
            for d in dets.values():
                bins = (d.get("bins") or {}).keys()
                discovered.update(str(b) for b in bins)
        bin_labels = sorted(discovered, key=_bin_sort_key)

    # aggregate at video level -> mean/std over videos per detector/bin/type
    out: dict[str, Any] = {t: {d: {b: [] for b in bin_labels} for d in DET_ORDER} for t in TYPE_ORDER}
    for _vid, v in grouped.items():
        vtype = str(v.get("video_type", "unknown")).lower()
        if vtype not in out:
            continue
        vday = str(v.get("daynight", "unknown")).lower()
        vuv = str(v.get("uvnorm", "unknown")).lower()
        if daynight != "any" and vday != daynight:
            continue
        if uvnorm != "any" and vuv != uvnorm:
            continue
        dets = v.get("detectors") or {}
        for det_name in DET_ORDER:
            d = dets.get(det_name) or {}
            bins = d.get("bins") or {}
            for b in bin_labels:
                bb = bins.get(b) or {}
                c = _safe_float(bb.get("mean_target_conf"))
                if not math.isnan(c):
                    out[vtype][det_name][b].append(c)
    return bin_labels, out


def _collect_rows_frames(payload: dict[str, Any], *, daynight: str, uvnorm: str) -> tuple[list[int], dict[str, Any]]:
    recs = payload.get("images") or []
    # out[type][detector][frame_idx] -> list[conf]
    out: dict[str, Any] = {t: {d: {} for d in DET_ORDER} for t in TYPE_ORDER}
    frame_set: set[int] = set()
    for r in recs:
        if str(r.get("source_type", "")).lower() != "video":
            continue
        vm = r.get("video_name_meta") or {}
        if not isinstance(vm, dict):
            vm = {}
        vtype = str(vm.get("video_type", "")).lower()
        vday = str(vm.get("daynight", "")).lower()
        vuv = str(vm.get("uvnorm", "")).lower()
        if vtype not in TYPE_ORDER:
            parsed = _parse_video_name_meta(str(r.get("source_path", "")))
            vtype = parsed["video_type"]
            vday = parsed["daynight"]
            vuv = parsed["uvnorm"]
        if vtype not in TYPE_ORDER:
            continue
        if daynight != "any" and vday != daynight:
            continue
        if uvnorm != "any" and vuv != uvnorm:
            continue
        try:
            fidx = int(r.get("frame_index"))
        except Exception:
            continue
        frame_set.add(fidx)
        dets = r.get("detectors") or {}
        for det_name in DET_ORDER:
            d = dets.get(det_name) or {}
            c = _safe_float(d.get("target_conf"))
            if math.isnan(c):
                continue
            out[vtype][det_name].setdefault(fidx, []).append(c)
    frames = sorted(frame_set)
    return frames, out


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _plot_one(
    out_png: Path,
    out_pdf: Path,
    title: str,
    bin_labels: list[str],
    bin_to_stats: dict[str, dict[str, tuple[float, float, int]]],
) -> None:
    # x: far -> near (20 -> 0) for driving interpretation
    bins_sorted = sorted(bin_labels, key=_bin_sort_key)
    centers = [_bin_center(b) for b in bins_sorted]
    bins_plot = list(reversed(bins_sorted))
    x = list(reversed(centers))

    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    for det in DET_ORDER:
        ys = []
        yerr = []
        ns = []
        for b in bins_plot:
            mu, sd, n = bin_to_stats[det].get(b, (float("nan"), float("nan"), 0))
            ys.append(mu)
            yerr.append(0.0 if math.isnan(sd) else sd)
            ns.append(n)
        if all(math.isnan(v) for v in ys):
            continue
        ax.errorbar(
            x,
            ys,
            yerr=yerr,
            marker="o",
            linewidth=2.0,
            markersize=5.0,
            capsize=3.0,
            color=COLORS.get(det, None),
            label=DET_LABEL.get(det, det),
            alpha=0.95,
        )

    ax.set_title(title, fontsize=14, weight="bold")
    ax.set_xlabel("Distance to Stop Sign (m) [far -> near]")
    ax.set_ylabel("Target Confidence (stop sign)")
    ax.set_xticks(x)
    ax.set_xticklabels(bins_plot)
    ax.grid(True, alpha=0.25, linestyle="--")
    ax.set_ylim(bottom=0.0)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=220)
    fig.savefig(out_pdf)
    plt.close(fig)


def _plot_one_frames(
    out_png: Path,
    out_pdf: Path,
    title: str,
    frames: list[int],
    frame_to_stats: dict[str, dict[int, tuple[float, float, int]]],
) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    for det in DET_ORDER:
        xs: list[int] = []
        ys: list[float] = []
        sds: list[float] = []
        for f in frames:
            mu, sd, _n = frame_to_stats[det].get(f, (float("nan"), float("nan"), 0))
            if math.isnan(mu):
                continue
            xs.append(f)
            ys.append(mu)
            sds.append(0.0 if math.isnan(sd) else sd)
        if not xs:
            continue
        ax.plot(
            xs,
            ys,
            linewidth=1.8,
            color=COLORS.get(det, None),
            label=DET_LABEL.get(det, det),
            alpha=0.95,
        )
        if len(xs) > 3:
            lo = [max(0.0, y - s) for y, s in zip(ys, sds)]
            hi = [y + s for y, s in zip(ys, sds)]
            ax.fill_between(xs, lo, hi, color=COLORS.get(det, None), alpha=0.12, linewidth=0)

    ax.set_title(title, fontsize=14, weight="bold")
    ax.set_xlabel("Frame Index")
    ax.set_ylabel("Target Confidence (stop sign)")
    ax.grid(True, alpha=0.25, linestyle="--")
    ax.set_ylim(bottom=0.0)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=220)
    fig.savefig(out_pdf)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description="Plot 3 detector confidence line charts (base/spray/paint).")
    ap.add_argument("--input-json", default="_runs/paper_data/real_detector_compare/real_world_videos_20to0.json")
    ap.add_argument("--out-dir", default="_runs/paper_data/real_detector_compare/plots_distance")
    ap.add_argument("--mode", choices=["frames", "bins"], default="frames",
                    help="frames=raw per-frame plot (no quartile point estimation); bins=distance-bin means.")
    ap.add_argument("--daynight", choices=["any", "day", "night"], default="any",
                    help="Optional filter on video name tag.")
    ap.add_argument("--uvnorm", choices=["any", "uv", "norm"], default="any",
                    help="Optional filter on video name tag.")
    args = ap.parse_args()

    in_json = Path(args.input_json)
    if not in_json.exists():
        raise FileNotFoundError(f"Input JSON not found: {in_json}")
    payload = json.loads(in_json.read_text(encoding="utf-8"))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_rows: list[dict[str, Any]] = []
    suffix = f"{args.mode}_{args.daynight}_{args.uvnorm}"
    if args.mode == "bins":
        bin_labels, grouped = _collect_rows_bins(payload, daynight=args.daynight, uvnorm=args.uvnorm)
        if not bin_labels:
            raise ValueError("No distance bins found in JSON.")
        for t in TYPE_ORDER:
            per_det: dict[str, dict[str, tuple[float, float, int]]] = {d: {} for d in DET_ORDER}
            for d in DET_ORDER:
                for b in bin_labels:
                    vals = grouped[t][d][b]
                    per_det[d][b] = (_mean(vals), _std(vals), len(vals))
                    csv_rows.append({
                        "mode": "bins",
                        "type": t,
                        "detector": d,
                        "distance_bin": b,
                        "mean_target_conf": per_det[d][b][0],
                        "std_target_conf": per_det[d][b][1],
                        "n_samples": per_det[d][b][2],
                        "daynight_filter": args.daynight,
                        "uvnorm_filter": args.uvnorm,
                    })
            png = out_dir / f"line_conf_vs_distance_{t}_{suffix}.png"
            pdf = out_dir / f"line_conf_vs_distance_{t}_{suffix}.pdf"
            title = f"{TYPE_LABEL[t]}: detector confidence vs distance"
            if args.daynight != "any" or args.uvnorm != "any":
                title += f" ({args.daynight}/{args.uvnorm})"
            _plot_one(png, pdf, title, bin_labels, per_det)
            print(f"[SAVE] {png}")
            print(f"[SAVE] {pdf}")
    else:
        frames, grouped = _collect_rows_frames(payload, daynight=args.daynight, uvnorm=args.uvnorm)
        if not frames:
            raise ValueError("No video frame rows found in JSON for the selected filters.")
        for t in TYPE_ORDER:
            per_det: dict[str, dict[int, tuple[float, float, int]]] = {d: {} for d in DET_ORDER}
            for d in DET_ORDER:
                for f in frames:
                    vals = grouped[t][d].get(f, [])
                    if not vals:
                        continue
                    per_det[d][f] = (_mean(vals), _std(vals), len(vals))
                    csv_rows.append({
                        "mode": "frames",
                        "type": t,
                        "detector": d,
                        "frame_index": f,
                        "mean_target_conf": per_det[d][f][0],
                        "std_target_conf": per_det[d][f][1],
                        "n_samples": per_det[d][f][2],
                        "daynight_filter": args.daynight,
                        "uvnorm_filter": args.uvnorm,
                    })
            png = out_dir / f"line_conf_vs_frame_{t}_{suffix}.png"
            pdf = out_dir / f"line_conf_vs_frame_{t}_{suffix}.pdf"
            title = f"{TYPE_LABEL[t]}: detector confidence vs frame"
            if args.daynight != "any" or args.uvnorm != "any":
                title += f" ({args.daynight}/{args.uvnorm})"
            _plot_one_frames(png, pdf, title, frames, per_det)
            print(f"[SAVE] {png}")
            print(f"[SAVE] {pdf}")

    csv_out = out_dir / f"line_conf_stats_{suffix}.csv"
    _write_csv(csv_out, csv_rows)
    print(f"[SAVE] {csv_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
