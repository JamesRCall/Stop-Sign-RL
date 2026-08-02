from __future__ import annotations

import json
from pathlib import Path
from collections import defaultdict

import matplotlib.pyplot as plt


def _round_key(x: float, ndigits: int = 6) -> float:
    return round(float(x), ndigits)


def _mean(vals):
    return sum(vals) / len(vals) if vals else None


def _pretty_combo(name: str) -> str:
    return str(name).replace("Glow", "").strip()


def _color_for_combo(name: str) -> str | None:
    n = _pretty_combo(name).lower()
    if "red" in n:
        return "red"
    if "green" in n:
        return "green"
    if "blue" in n:
        return "blue"
    if "yellow" in n:
        return "gold"
    if "orange" in n:
        return "orange"
    if "purple" in n:
        return "purple"
    if "pink" in n:
        return "hotpink"
    if "cyan" in n:
        return "cyan"
    return None


def load_ndjson(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def main() -> int:
    src = Path("runs/area_sweep.ndjson")
    if not src.exists():
        raise SystemExit(f"Missing: {src}")

    rows = load_ndjson(src)
    det_map = {
        "yolo8": ("yolo8", "YOLOv8"),
        "yolo11": ("yolo11", "YOLOv11"),
        "fasterrcnn_resnet50_fpn_v2": ("fasterrcnn_v2", "FRCNNv2"),
        "fcos_resnet50_fpn": ("fcos", "FCOS"),
        "retinanet_resnet50_fpn_v2": ("retinanet_v2", "RetNetv2"),
        "rtdetr": ("rtdetr_r50vd", "RT-DETR"),
        # Explicitly omit SSD
        "ssd300_vgg16": None,
    }
    det_order = ["yolo8", "yolo11", "fasterrcnn_v2", "fcos", "retinanet_v2", "rtdetr_r50vd"]
    # Keep required fields only
    keep = []
    for r in rows:
        det_raw = r.get("detector_name")
        mapped = det_map.get(det_raw, (str(det_raw), str(det_raw)))
        if mapped is None:
            continue
        det_key, det_label = mapped
        combo = r.get("paint_combo")
        size = r.get("combo_size")
        area = r.get("area_frac")
        drop = r.get("drop_on")
        if det_raw is None or combo is None or size is None or area is None or drop is None:
            continue
        keep.append(
            {
                "detector": str(det_key),
                "detector_label": str(det_label),
                "combo": str(combo),
                "combo_size": int(size),
                "area_frac": float(area),
                "drop_on": float(drop),
            }
        )

    if not keep:
        raise SystemExit("No valid rows found.")

    detectors = [d for d in det_order if d in {r["detector"] for r in keep}]
    det_labels = {}
    for r in keep:
        det_labels[r["detector"]] = r["detector_label"]
    combo_sizes = sorted({r["combo_size"] for r in keep})

    out_dir = Path("runs/area_sweep_plots")
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------- Plot 1: single-color lines per detector (size=1), 3x2 grid ----------
    # Aggregate mean drop_on per (detector, combo, area_frac)
    agg_det_combo = defaultdict(list)
    for r in keep:
        key = (r["detector"], r["combo"], _round_key(r["area_frac"]))
        agg_det_combo[key].append(r["drop_on"])

    series_det_combo = defaultdict(list)
    for (det, combo, area_key), vals in agg_det_combo.items():
        series_det_combo[(det, combo)].append((area_key, _mean(vals)))

    combos_size1 = sorted({r["combo"] for r in keep if r["combo_size"] == 1})
    fig, axes = plt.subplots(2, 3, figsize=(12, 7.2), sharex=True, sharey=True)
    axes = axes.flatten()

    for i, det in enumerate(detectors[:6]):
        ax = axes[i]
        for combo in combos_size1:
            pts = sorted(series_det_combo.get((det, combo), []), key=lambda x: x[0])
            if not pts:
                continue
            xs, ys = zip(*pts)
            ax.plot(
                xs,
                ys,
                marker="o",
                linewidth=1.5,
                label=_pretty_combo(combo),
                color=_color_for_combo(combo),
            )
        ax.set_title(det_labels.get(det, det))
        ax.set_xlabel("area_frac")
        ax.set_ylabel("drop_on")
        ax.grid(True, alpha=0.25)

    # remove extra axes if fewer than 6 detectors
    for j in range(len(detectors), len(axes)):
        fig.delaxes(axes[j])

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=min(3, len(labels)))

    fig.tight_layout(rect=[0, 0, 1, 0.90])
    out_all = out_dir / "area_sweep_single_color_by_detector.pdf"
    fig.savefig(out_all)
    plt.close(fig)

    # ---------- Plot 2: combo-size tiers with lines per combo (avg over detectors), 3x2 grid ----------
    # Aggregate mean drop_on per (combo_size, combo, area_frac) averaged over detectors
    tier_combo_agg = defaultdict(list)
    for r in keep:
        key = (r["combo_size"], r["combo"], _round_key(r["area_frac"]))
        tier_combo_agg[key].append(r["drop_on"])

    series_tier_combo = defaultdict(list)
    for (size, combo, area_key), vals in tier_combo_agg.items():
        series_tier_combo[(size, combo)].append((area_key, _mean(vals)))

    fig, axes = plt.subplots(2, 3, figsize=(12, 7.2), sharex=True, sharey=True)
    axes = axes.flatten()

    for i, size in enumerate(combo_sizes[:6]):
        ax = axes[i]
        combos_for_size = sorted({r["combo"] for r in keep if r["combo_size"] == size})
        for combo in combos_for_size:
            pts = sorted(series_tier_combo.get((size, combo), []), key=lambda x: x[0])
            if not pts:
                continue
            xs, ys = zip(*pts)
            ax.plot(xs, ys, marker="o", linewidth=1.5, label=_pretty_combo(combo))
        ax.set_title(f"combo_size={size}")
        ax.set_xlabel("area_frac")
        ax.set_ylabel("drop_on")
        ax.grid(True, alpha=0.25)

    for j in range(len(combo_sizes), len(axes)):
        fig.delaxes(axes[j])

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=min(3, len(labels)))

    fig.tight_layout(rect=[0, 0, 1, 0.90])
    out_tiers = out_dir / "area_sweep_combo_size_tiers.pdf"
    fig.savefig(out_tiers)
    plt.close(fig)

    print(f"Wrote: {out_all}")
    print(f"Wrote: {out_tiers}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
