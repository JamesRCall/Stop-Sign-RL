#!/usr/bin/env python3
"""
Analyze area sweep NDJSON logs and generate summary tables + plots.
"""
import argparse
import json
import os
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt


def load_rows(path: str):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def mean_std(vals):
    arr = np.array(vals, dtype=float)
    return float(arr.mean()), float(arr.std())


def main() -> None:
    ap = argparse.ArgumentParser(description="Summarize area_sweep NDJSON results.")
    ap.add_argument("--input", default="./_debug_area_sweep/area_sweep.ndjson")
    ap.add_argument("--out", default="./_debug_area_sweep/plots")
    ap.add_argument("--top-n", type=int, default=10)
    ap.add_argument("--percentages", default="10,25,50,75,100")
    ap.add_argument("--metric", default="c_on",
                    choices=["c_on", "mean_iou", "misclass_rate"])
    args = ap.parse_args()

    rows = load_rows(args.input)
    if not rows:
        raise SystemExit(f"No rows found in {args.input}")

    os.makedirs(args.out, exist_ok=True)
    pct_list = [float(p.strip()) / 100.0 for p in args.percentages.split(",") if p.strip()]

    # bucket rows by (target_pct, combo)
    buckets = defaultdict(list)
    for r in rows:
        tp = float(r.get("target_pct", r.get("area_frac", 0.0)))
        combo = r.get("paint_combo", "unknown")
        buckets[(tp, combo)].append(r)

    # Build summaries
    summaries = []
    for (tp, combo), recs in buckets.items():
        c_on_mean, c_on_std = mean_std([r["c_on"] for r in recs])
        iou_mean, iou_std = mean_std([r["mean_iou"] for r in recs])
        mis_mean, mis_std = mean_std([r["misclass_rate"] for r in recs])
        area_mean, area_std = mean_std([r["area_frac"] for r in recs])
        summaries.append({
            "target_pct": tp,
            "paint_combo": combo,
            "area_mean": area_mean,
            "area_std": area_std,
            "c_on_mean": c_on_mean,
            "c_on_std": c_on_std,
            "iou_mean": iou_mean,
            "iou_std": iou_std,
            "misclass_mean": mis_mean,
            "misclass_std": mis_std,
            "n": len(recs),
        })

    # Save summary json
    summary_path = os.path.join(args.out, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2)

    # Plot top-N per percentage
    for tp in pct_list:
        subset = [s for s in summaries if abs(s["target_pct"] - tp) < 1e-6]
        if not subset:
            continue

        # sort by metric (lower c_on/iou is better; higher misclass is better)
        if args.metric == "misclass_rate":
            subset.sort(key=lambda s: s["misclass_mean"], reverse=True)
            vals = [s["misclass_mean"] for s in subset[: args.top_n]]
            ylabel = "Misclass rate (mean)"
            title = f"Top {args.top_n} combos @ {int(tp*100)}% (misclass)"
        elif args.metric == "mean_iou":
            subset.sort(key=lambda s: s["iou_mean"])
            vals = [s["iou_mean"] for s in subset[: args.top_n]]
            ylabel = "IoU (mean)"
            title = f"Top {args.top_n} combos @ {int(tp*100)}% (IoU)"
        else:
            subset.sort(key=lambda s: s["c_on_mean"])
            vals = [s["c_on_mean"] for s in subset[: args.top_n]]
            ylabel = "After-conf (mean)"
            title = f"Top {args.top_n} combos @ {int(tp*100)}% (after-conf)"

        labels = [s["paint_combo"] for s in subset[: args.top_n]]
        plt.figure(figsize=(10, 6))
        plt.barh(range(len(labels))[::-1], vals[::-1])
        plt.yticks(range(len(labels))[::-1], labels[::-1])
        plt.xlabel(ylabel)
        plt.title(title)
        plt.tight_layout()
        out_path = os.path.join(args.out, f"top_{args.metric}_{int(tp*100)}.png")
        plt.savefig(out_path, dpi=160)
        plt.close()


if __name__ == "__main__":
    main()
