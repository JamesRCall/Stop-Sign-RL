#!/usr/bin/env python3
"""
Rank color combos using Pareto fronts per metric (no weighted mixing).

Outputs:
  - summary_overall.json / summary_overall.csv
  - summary_by_detector.json
  - pareto_overall_<metric>.json / .csv
  - pareto_<detector>_<metric>.json / .csv
"""
import argparse
import csv
import json
import os
from collections import defaultdict

import numpy as np


METRIC_DIRECTIONS = {
    "drop_on": "max",
    "c_on": "min",
    "misclass_rate": "max",
    "mean_iou": "min",
}


def _safe_float(x, default=np.nan) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


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


def _mean(vals):
    vals = [v for v in vals if not np.isnan(v)]
    return float(np.mean(vals)) if vals else float("nan")


def _auc(x, y):
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    mask = ~np.isnan(x) & ~np.isnan(y)
    if mask.sum() < 2:
        return float("nan")
    x = x[mask]
    y = y[mask]
    order = np.argsort(x)
    try:
        return float(np.trapezoid(y[order], x[order]))
    except AttributeError:
        return float(np.trapz(y[order], x[order]))


def _pareto_front(points, maximize_metric: bool):
    """
    points: list of (key, metric_val, area_val)
    maximize_metric: True if metric should be maximized; else minimized.
    """
    front = []
    for key, m, a in points:
        if np.isnan(m) or np.isnan(a):
            continue
        dominated = False
        for key2, m2, a2 in points:
            if np.isnan(m2) or np.isnan(a2):
                continue
            if key == key2:
                continue
            if maximize_metric:
                better_or_equal = (m2 >= m) and (a2 <= a)
                strictly_better = (m2 > m) or (a2 < a)
            else:
                better_or_equal = (m2 <= m) and (a2 <= a)
                strictly_better = (m2 < m) or (a2 < a)
            if better_or_equal and strictly_better:
                dominated = True
                break
        if not dominated:
            front.append((key, m, a))
    return front


def _write_csv(path, rows, fieldnames):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main() -> None:
    ap = argparse.ArgumentParser(description="Pareto ranking for area_sweep NDJSON.")
    ap.add_argument("--input", default="./_debug_area_sweep/area_sweep.ndjson")
    ap.add_argument("--out", default="./_debug_area_sweep/rankings")
    ap.add_argument("--percentages", default="10,25,50,75,100")
    ap.add_argument(
        "--metrics",
        default="c_on,misclass_rate,mean_iou,drop_on",
        help="Comma-separated metrics to rank.",
    )
    args = ap.parse_args()

    metrics = [m.strip() for m in str(args.metrics).split(",") if m.strip()]
    for m in metrics:
        if m not in METRIC_DIRECTIONS:
            raise SystemExit(f"Unsupported metric: {m}")

    pct_list = _parse_percent_list(args.percentages)
    if not pct_list:
        raise SystemExit("No percentages provided.")

    # stats[(detector, combo)][pct][metric] -> sum/count
    stats = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: {"sum": 0.0, "count": 0})))
    area_stats = defaultdict(lambda: defaultdict(lambda: {"sum": 0.0, "count": 0}))

    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            det = str(r.get("detector_name") or r.get("detector") or "unknown")
            combo = str(r.get("paint_combo", "unknown"))
            pct = _safe_float(r.get("target_pct", np.nan))
            if np.isnan(pct):
                continue
            # normalize pct to fraction
            if pct > 1.0:
                pct = pct / 100.0
            if pct not in pct_list:
                continue

            area = _safe_float(r.get("area_frac", np.nan))
            if not np.isnan(area):
                area_stats[(det, combo)][pct]["sum"] += float(area)
                area_stats[(det, combo)][pct]["count"] += 1

            for m in metrics:
                v = _safe_float(r.get(m, np.nan))
                if np.isnan(v):
                    continue
                stats[(det, combo)][pct][m]["sum"] += float(v)
                stats[(det, combo)][pct][m]["count"] += 1

    os.makedirs(args.out, exist_ok=True)

    def build_summary(det_filter=None):
        rows = []
        for (det, combo), pct_map in stats.items():
            if det_filter is not None and det != det_filter:
                continue
            # per-coverage means
            metric_by_pct = {m: [] for m in metrics}
            area_by_pct = []
            for pct in pct_list:
                # metric means
                for m in metrics:
                    s = pct_map.get(pct, {}).get(m, {"sum": 0.0, "count": 0})
                    mean_val = float(s["sum"] / s["count"]) if s["count"] > 0 else float("nan")
                    metric_by_pct[m].append(mean_val)
                # area mean
                a = area_stats.get((det, combo), {}).get(pct, {"sum": 0.0, "count": 0})
                area_mean = float(a["sum"] / a["count"]) if a["count"] > 0 else float("nan")
                area_by_pct.append(area_mean)

            rec = {
                "detector": det,
                "paint_combo": combo,
                "area_mean": _mean(area_by_pct),
                "area_auc": _auc(pct_list, area_by_pct),
            }
            for m in metrics:
                rec[f"{m}_mean"] = _mean(metric_by_pct[m])
                rec[f"{m}_auc"] = _auc(pct_list, metric_by_pct[m])
            rows.append(rec)
        return rows

    # Overall summary
    overall_rows = build_summary(det_filter=None)
    summary_overall_json = os.path.join(args.out, "summary_overall.json")
    with open(summary_overall_json, "w", encoding="utf-8") as f:
        json.dump(overall_rows, f, indent=2)
    if overall_rows:
        _write_csv(
            os.path.join(args.out, "summary_overall.csv"),
            overall_rows,
            fieldnames=list(overall_rows[0].keys()),
        )

    # Per-detector summary
    detectors = sorted({r["detector"] for r in overall_rows})
    by_det = {}
    for det in detectors:
        rows = build_summary(det_filter=det)
        by_det[det] = rows
    with open(os.path.join(args.out, "summary_by_detector.json"), "w", encoding="utf-8") as f:
        json.dump(by_det, f, indent=2)

    # Pareto fronts per metric (overall + per detector)
    def pareto_for_rows(rows, metric):
        maximize = METRIC_DIRECTIONS[metric] == "max"
        points = []
        for r in rows:
            points.append((r["paint_combo"], r[f"{metric}_mean"], r["area_mean"]))
        front = _pareto_front(points, maximize_metric=maximize)
        out = []
        for combo, m, a in front:
            out.append({"paint_combo": combo, metric: m, "area_mean": a})
        return out

    for m in metrics:
        front = pareto_for_rows(overall_rows, m)
        out_json = os.path.join(args.out, f"pareto_overall_{m}.json")
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(front, f, indent=2)
        if front:
            _write_csv(
                os.path.join(args.out, f"pareto_overall_{m}.csv"),
                front,
                fieldnames=list(front[0].keys()),
            )

    for det, rows in by_det.items():
        for m in metrics:
            front = pareto_for_rows(rows, m)
            out_json = os.path.join(args.out, f"pareto_{det}_{m}.json")
            with open(out_json, "w", encoding="utf-8") as f:
                json.dump(front, f, indent=2)
            if front:
                _write_csv(
                    os.path.join(args.out, f"pareto_{det}_{m}.csv"),
                    front,
                    fieldnames=list(front[0].keys()),
                )

    # Best combo per coverage percent (overall across detectors + per detector)
    def _best_for_pct(rows, metric, pct_val):
        maximize = METRIC_DIRECTIONS[metric] == "max"
        best = None
        for r in rows:
            if abs(float(r.get("target_pct", -1)) - float(pct_val)) > 1e-9:
                continue
            val = _safe_float(r.get(f"{metric}_mean", np.nan))
            if np.isnan(val):
                continue
            if metric == "misclass_rate" and val <= 0.0:
                continue
            if metric == "mean_iou" and val >= 1.0:
                continue
            if best is None:
                best = r, val
            else:
                if maximize and val > best[1]:
                    best = r, val
                if (not maximize) and val < best[1]:
                    best = r, val
        return best

    # Build per-percent summaries from summary_overall (averaged across detectors)
    overall_pct_rows = []
    # derive per-combo per-pct aggregated across detectors
    combo_pct = defaultdict(lambda: defaultdict(list))
    for (det, combo), pct_map in stats.items():
        for pct in pct_list:
            s = pct_map.get(pct, {})
            rec = {"detector": det, "paint_combo": combo, "target_pct": pct}
            for m in metrics:
                agg = s.get(m, {"sum": 0.0, "count": 0})
                rec[m] = float(agg["sum"] / agg["count"]) if agg["count"] > 0 else float("nan")
            combo_pct[combo][pct].append(rec)

    # average across detectors for each combo/pct
    overall_rows_by_pct = []
    for combo, pct_map in combo_pct.items():
        for pct, recs in pct_map.items():
            row = {"paint_combo": combo, "target_pct": pct}
            for m in metrics:
                row[f"{m}_mean"] = _mean([_safe_float(r.get(m, np.nan)) for r in recs])
            overall_rows_by_pct.append(row)

    for m in metrics:
        best_rows = []
        for pct in pct_list:
            best = _best_for_pct(overall_rows_by_pct, m, pct)
            if not best:
                best_rows.append(
                    {
                        "target_pct": float(pct),
                        "paint_combo": None,
                        f"{m}_mean": None,
                    }
                )
                continue
            rec, val = best
            best_rows.append(
                {
                    "target_pct": float(pct),
                    "paint_combo": rec["paint_combo"],
                    f"{m}_mean": float(val),
                }
            )
        out_json = os.path.join(args.out, f"best_by_pct_overall_{m}.json")
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(best_rows, f, indent=2)
        if best_rows:
            _write_csv(
                os.path.join(args.out, f"best_by_pct_overall_{m}.csv"),
                best_rows,
                fieldnames=list(best_rows[0].keys()),
            )


    # per detector best by pct
    for det, rows in by_det.items():
        # rebuild per-detector per-pct list
        rows_by_pct = []
        for (d, combo), pct_map in stats.items():
            if d != det:
                continue
            for pct in pct_list:
                s = pct_map.get(pct, {})
                rec = {"paint_combo": combo, "target_pct": pct}
                for m in metrics:
                    agg = s.get(m, {"sum": 0.0, "count": 0})
                    rec[f"{m}_mean"] = float(agg["sum"] / agg["count"]) if agg["count"] > 0 else float("nan")
                rows_by_pct.append(rec)

        for m in metrics:
            best_rows = []
            for pct in pct_list:
                best = _best_for_pct(rows_by_pct, m, pct)
                if not best:
                    best_rows.append(
                        {
                            "target_pct": float(pct),
                            "paint_combo": None,
                            f"{m}_mean": None,
                        }
                    )
                    continue
                rec, val = best
                best_rows.append(
                    {
                        "target_pct": float(pct),
                        "paint_combo": rec["paint_combo"],
                        f"{m}_mean": float(val),
                    }
                )
            out_json = os.path.join(args.out, f"best_by_pct_{det}_{m}.json")
            with open(out_json, "w", encoding="utf-8") as f:
                json.dump(best_rows, f, indent=2)
            if best_rows:
                _write_csv(
                    os.path.join(args.out, f"best_by_pct_{det}_{m}.csv"),
                    best_rows,
                    fieldnames=list(best_rows[0].keys()),
                )


if __name__ == "__main__":
    main()
