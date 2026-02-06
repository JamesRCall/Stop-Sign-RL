#!/usr/bin/env python3
"""
Analyze area sweep NDJSON logs and generate coverage plots by detector and color-group.

This script groups results by detector and color-combo size, then plots:
  - coverage vs confidence (c_on)
  - coverage vs misclassification rate
  - coverage vs IoU
"""
import argparse
import json
import os
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt


COLOR_ABBR = {
    "WhiteGlow": "WG",
    "RedGlow": "RG",
    "GreenGlow": "GG",
    "YellowGlow": "YG",
    "BlueGlow": "BG",
    "OrangeGlow": "OG",
}


def _abbr_combo(combo: str) -> str:
    parts = [p.strip() for p in str(combo).split("+") if p.strip()]
    if not parts:
        return "unknown"
    abbr = [COLOR_ABBR.get(p, p) for p in parts]
    return "+".join(abbr)

def _parse_percent_list(s: str):
    out = []
    for part in str(s).split(","):
        part = part.strip()
        if not part:
            continue
        val = float(part)
        if val > 1.0:
            val = val / 100.0
        out.append(val)
    return out


def _safe_float(x, default=np.nan) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def _accumulate(stats, key, metric, val):
    if val is None or np.isnan(val):
        return
    s = stats[key]
    s[metric]["sum"] += float(val)
    s[metric]["count"] += 1


def _mean_from_stats(s):
    out = {}
    for metric, agg in s.items():
        if agg["count"] <= 0:
            out[metric] = np.nan
        else:
            out[metric] = float(agg["sum"] / agg["count"])
    return out


def load_summary(path: str, metrics):
    """
    Stream the NDJSON and aggregate means per (detector, combo_size, combo, pct).

    Args:
        path: Path to NDJSON log.
        metrics: Metrics to aggregate.

    Returns:
        Tuple of (summary_dict, pct_set).
    """
    stats = defaultdict(lambda: {m: {"sum": 0.0, "count": 0} for m in metrics})
    pct_set = set()

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue

            det = r.get("detector_name") or r.get("detector") or "unknown"
            combo = r.get("paint_combo", "unknown")
            combo_size = int(r.get("combo_size", 1))
            pct = _safe_float(r.get("target_pct", r.get("area_frac", np.nan)))
            if np.isnan(pct):
                continue
            pct_set.add(pct)

            key = (str(det), int(combo_size), str(combo), float(pct))
            for m in metrics:
                _accumulate(stats, key, m, _safe_float(r.get(m, np.nan)))

    return stats, pct_set


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot coverage curves by detector and color group.")
    ap.add_argument("--input", default="./_debug_area_sweep/area_sweep.ndjson")
    ap.add_argument("--out", default="./_debug_area_sweep/plots")
    ap.add_argument("--percentages", default="10,25,50,75,100")
    ap.add_argument(
        "--metrics",
        default="c_on,misclass_rate,mean_iou",
        help="Comma-separated metrics to plot.",
    )
    ap.add_argument(
        "--show-mean",
        type=int,
        default=1,
        help="Overlay mean+std across combos (1=yes, 0=no).",
    )
    args = ap.parse_args()

    metrics = [m.strip() for m in str(args.metrics).split(",") if m.strip()]
    if not metrics:
        raise SystemExit("No metrics specified.")

    stats, pct_set = load_summary(args.input, metrics)
    if not stats:
        raise SystemExit(f"No rows found in {args.input}")

    os.makedirs(args.out, exist_ok=True)
    pct_list = _parse_percent_list(args.percentages)
    if not pct_list:
        pct_list = sorted(pct_set)
    pct_list = sorted(set(float(p) for p in pct_list))

    # Build in-memory summary mapping: det -> combo_size -> combo -> pct -> metrics
    det_map = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    for (det, combo_size, combo, pct), s in stats.items():
        det_map[det][combo_size][combo][pct] = _mean_from_stats(s)

    # Save summary json for reproducibility
    summary_path = os.path.join(args.out, "summary.json")
    out_json = []
    for det, size_map in det_map.items():
        for combo_size, combo_map in size_map.items():
            for combo, pct_map in combo_map.items():
                for pct, mvals in pct_map.items():
                    rec = {
                        "detector": det,
                        "combo_size": combo_size,
                        "paint_combo": combo,
                        "target_pct": pct,
                    }
                    rec.update({f"{k}_mean": v for k, v in mvals.items()})
                    out_json.append(rec)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(out_json, f, indent=2)

    # Plot coverage curves per detector and combo_size (one line per combo)
    for det, size_map in det_map.items():
        det_dir = os.path.join(args.out, str(det))
        os.makedirs(det_dir, exist_ok=True)

        # Big per-detector plots with all combos (across sizes)
        all_combo_map = {}
        for combo_size, combo_map in size_map.items():
            for combo, pct_map in combo_map.items():
                all_combo_map[combo] = pct_map

        for metric in metrics:
            plt.figure(figsize=(14, 8))
            combos = sorted(all_combo_map.keys())
            x_vals = [p * 100.0 for p in pct_list]
            for combo in combos:
                ys = []
                for pct in pct_list:
                    mvals = all_combo_map[combo].get(pct, None)
                    ys.append(mvals.get(metric, np.nan) if mvals else np.nan)
                plt.plot(x_vals, ys, linewidth=1.0, alpha=0.6, label=_abbr_combo(combo))

            if int(args.show_mean):
                mean_ys = []
                std_ys = []
                for pct in pct_list:
                    vals = []
                    for combo in combos:
                        mvals = all_combo_map[combo].get(pct, None)
                        v = mvals.get(metric, np.nan) if mvals else np.nan
                        if not np.isnan(v):
                            vals.append(float(v))
                    if vals:
                        mean_ys.append(float(np.mean(vals)))
                        std_ys.append(float(np.std(vals)))
                    else:
                        mean_ys.append(np.nan)
                        std_ys.append(np.nan)
                plt.plot(x_vals, mean_ys, color="black", linewidth=2.0, label="mean")
                plt.fill_between(
                    x_vals,
                    np.array(mean_ys) - np.array(std_ys),
                    np.array(mean_ys) + np.array(std_ys),
                    color="black",
                    alpha=0.12,
                    linewidth=0,
                    label="std",
                )

            plt.xlabel("Coverage (%)")
            ylabel = {
                "c_on": "After-confidence (mean)",
                "misclass_rate": "Misclassification rate (mean)",
                "mean_iou": "IoU (mean)",
            }.get(metric, metric)
            plt.ylabel(ylabel)
            plt.title(f"{det} | all combos | {metric}")
            plt.grid(True, alpha=0.2)

            # Place legend above to preserve plot width.
            plt.legend(
                fontsize=6,
                ncol=6,
                loc="upper center",
                bbox_to_anchor=(0.5, 1.18),
                borderaxespad=0.0,
            )
            plt.tight_layout(rect=[0, 0, 1, 0.90])
            out_path = os.path.join(det_dir, f"coverage_all_combos_{metric}.png")
            plt.savefig(out_path, dpi=220, bbox_inches="tight")
            plt.close()

        for combo_size, combo_map in size_map.items():
            group_dir = os.path.join(det_dir, f"combo_{combo_size}")
            os.makedirs(group_dir, exist_ok=True)

            for metric in metrics:
                plt.figure(figsize=(12, 7))
                combos = sorted(combo_map.keys())

                x_vals = [p * 100.0 for p in pct_list]
                for combo in combos:
                    ys = []
                    for pct in pct_list:
                        mvals = combo_map[combo].get(pct, None)
                        ys.append(mvals.get(metric, np.nan) if mvals else np.nan)
                    plt.plot(x_vals, ys, linewidth=1.2, alpha=0.75, label=_abbr_combo(combo))

                if int(args.show_mean):
                    mean_ys = []
                    std_ys = []
                    for pct in pct_list:
                        vals = []
                        for combo in combos:
                            mvals = combo_map[combo].get(pct, None)
                            v = mvals.get(metric, np.nan) if mvals else np.nan
                            if not np.isnan(v):
                                vals.append(float(v))
                        if vals:
                            mean_ys.append(float(np.mean(vals)))
                            std_ys.append(float(np.std(vals)))
                        else:
                            mean_ys.append(np.nan)
                            std_ys.append(np.nan)
                    plt.plot(
                        x_vals,
                        mean_ys,
                        color="black",
                        linewidth=2.0,
                        label="mean",
                    )
                    plt.fill_between(
                        x_vals,
                        np.array(mean_ys) - np.array(std_ys),
                        np.array(mean_ys) + np.array(std_ys),
                        color="black",
                        alpha=0.15,
                        linewidth=0,
                        label="std",
                    )

                plt.xlabel("Coverage (%)")
                ylabel = {
                    "c_on": "After-confidence (mean)",
                    "misclass_rate": "Misclassification rate (mean)",
                    "mean_iou": "IoU (mean)",
                }.get(metric, metric)
                plt.ylabel(ylabel)
                plt.title(f"{det} | combo_size={combo_size} | {metric}")
                plt.grid(True, alpha=0.2)
                if len(combos) <= 12:
                    plt.legend(fontsize=9, ncol=2, loc="best")
                else:
                    plt.legend(
                        fontsize=8,
                        ncol=4,
                        loc="upper center",
                        bbox_to_anchor=(0.5, 1.18),
                        borderaxespad=0.0,
                    )
                    plt.tight_layout(rect=[0, 0, 1, 0.90])

                out_path = os.path.join(group_dir, f"coverage_{metric}.png")
                plt.savefig(out_path, dpi=220, bbox_inches="tight")
                plt.close()


if __name__ == "__main__":
    main()
