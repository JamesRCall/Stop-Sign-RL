#!/usr/bin/env python3
"""Create paper-ready distance tables from real-world video detector comparison CSV.

Expected filename pattern (from source_path basename):
  Base_Light_1.2.MP4
  Base_Uv_2.2.MP4
  Paint_Light_3.2.MP4
  Spray_Uv_1.2.MP4

This script:
1) Parses condition/replicate from source video names.
2) Uses per-frame distance_to_sign_m from compare_real_images_detectors.py output.
3) Bins frames by distance.
4) Aggregates per-video first, then across replicate videos (mean +/- std).
5) Writes CSV + PDF tables.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


CONDITION_ORDER = [
    "base_light",
    "base_uv",
    "paint_light",
    "paint_uv",
    "spray_light",
    "spray_uv",
]


def _safe_float(v: Any) -> float:
    try:
        return float(v)
    except Exception:
        return float("nan")


def _safe_bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    return s in {"1", "true", "yes", "y"}


def _mean(vals: list[float]) -> float:
    arr = [x for x in vals if not math.isnan(x)]
    return sum(arr) / len(arr) if arr else float("nan")


def _std(vals: list[float]) -> float:
    arr = [x for x in vals if not math.isnan(x)]
    if not arr:
        return float("nan")
    mu = sum(arr) / len(arr)
    return math.sqrt(sum((x - mu) ** 2 for x in arr) / len(arr))


def _fmt_pm(mu: float, sd: float, prec: int = 3) -> str:
    if math.isnan(mu):
        return ""
    if math.isnan(sd):
        return f"{mu:.{prec}f}"
    return f"{mu:.{prec}f} +/- {sd:.{prec}f}"


def _fmt_pct_pm(mu: float, sd: float, prec: int = 1) -> str:
    if math.isnan(mu):
        return ""
    if math.isnan(sd):
        return f"{100.0 * mu:.{prec}f}%"
    return f"{100.0 * mu:.{prec}f}% +/- {100.0 * sd:.{prec}f}%"


def _parse_condition_from_source(source_path: str) -> tuple[str, int | None, bool]:
    name = Path(str(source_path)).name
    stem = name.rsplit(".", 1)[0]
    m = re.match(r"(?i)^(base|paint|spray)_(light|uv)_([0-9]+)", stem)
    if not m:
        return "unknown", None, False
    condition = f"{m.group(1).lower()}_{m.group(2).lower()}"
    rep = int(m.group(3))
    return condition, rep, True


def _parse_bins(spec: str) -> list[float]:
    vals = []
    for tok in str(spec).split(","):
        tok = tok.strip()
        if not tok:
            continue
        vals.append(float(tok))
    vals = sorted(set(vals))
    if len(vals) < 2:
        raise ValueError("Need at least two distance bin edges.")
    return vals


def _bin_label(lo: float, hi: float) -> str:
    return f"{int(round(lo)):02d}-{int(round(hi)):02d}m"


def _assign_bin(distance_m: float, edges: list[float]) -> tuple[str, float] | None:
    if math.isnan(distance_m):
        return None
    # Clamp values slightly outside the bin range.
    x = max(min(distance_m, edges[-1]), edges[0])
    for i in range(len(edges) - 1):
        lo = edges[i]
        hi = edges[i + 1]
        if i < len(edges) - 2:
            if lo <= x < hi:
                return _bin_label(lo, hi), 0.5 * (lo + hi)
        else:
            if lo <= x <= hi:
                return _bin_label(lo, hi), 0.5 * (lo + hi)
    return None


def _table_pdf(rows: list[dict[str, Any]], out_pdf: Path, title: str, subtitle: str, rows_per_page: int = 22) -> None:
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with PdfPages(out_pdf) as pdf:
            fig = plt.figure(figsize=(14, 8))
            ax = fig.add_axes([0, 0, 1, 1])
            ax.axis("off")
            ax.text(0.03, 0.95, title, fontsize=16, fontweight="bold", ha="left", va="top")
            ax.text(0.03, 0.90, subtitle, fontsize=10, color="#4b5563", ha="left", va="top")
            ax.text(0.03, 0.80, "No rows.", fontsize=12, ha="left", va="top")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
        return

    cols = list(rows[0].keys())
    data = [[str(r.get(c, "")) for c in cols] for r in rows]
    chunks = [data[i:i + rows_per_page] for i in range(0, len(data), rows_per_page)]
    with PdfPages(out_pdf) as pdf:
        for pidx, chunk in enumerate(chunks, 1):
            fig = plt.figure(figsize=(15, 8.5))
            ax = fig.add_axes([0.02, 0.03, 0.96, 0.94])
            ax.axis("off")
            ax.text(0.0, 1.03, title, fontsize=15, fontweight="bold", ha="left", va="bottom", transform=ax.transAxes)
            sub = subtitle if len(chunks) == 1 else f"{subtitle} | Page {pidx}/{len(chunks)}"
            ax.text(0.0, 0.995, sub, fontsize=9.5, color="#4b5563", ha="left", va="top", transform=ax.transAxes)

            max_lens = []
            for j, c in enumerate(cols):
                m = max(len(c), *(len(row[j]) for row in chunk))
                max_lens.append(max(6, min(m, 40)))
            tot = float(sum(max_lens))
            col_widths = [m / tot for m in max_lens]

            tbl = ax.table(
                cellText=chunk,
                colLabels=cols,
                colWidths=col_widths,
                loc="upper left",
                cellLoc="left",
                colLoc="left",
                bbox=[0.0, 0.0, 1.0, 0.93],
            )
            tbl.auto_set_font_size(False)
            tbl.set_fontsize(8.8)
            tbl.scale(1, 1.16)
            for (r, c), cell in tbl.get_celld().items():
                cell.set_edgecolor("#d1d5db")
                cell.set_linewidth(0.6)
                if r == 0:
                    cell.set_facecolor("#e5e7eb")
                    cell.set_text_props(weight="bold", color="#111827")
                else:
                    cell.set_facecolor("#ffffff" if (r % 2 == 1) else "#f9fafb")

            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _sort_condition_key(cond: str) -> tuple[int, str]:
    c = str(cond).lower()
    if c in CONDITION_ORDER:
        return (CONDITION_ORDER.index(c), c)
    return (999, c)


def main() -> int:
    ap = argparse.ArgumentParser(description="Build distance-binned real-world detector tables (CSV + PDF).")
    ap.add_argument("--input-csv", default="_runs/paper_data/real_detector_compare/real_world_results.csv")
    ap.add_argument("--out-dir", default="_runs/paper_data/real_detector_compare/paper_tables_distance")
    ap.add_argument("--distance-bins", default="0,5,10,15,20,25,30,35",
                    help="Comma-separated bin edges in meters.")
    ap.add_argument("--rows-per-page", type=int, default=24)
    args = ap.parse_args()

    in_csv = Path(args.input_csv)
    if not in_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {in_csv}")

    edges = _parse_bins(args.distance_bins)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows_raw: list[dict[str, Any]] = []
    with in_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for rec in reader:
            condition, rep, ok = _parse_condition_from_source(rec.get("source_path", ""))
            d_to = _safe_float(rec.get("distance_to_sign_m"))
            b = _assign_bin(d_to, edges)
            if b is None:
                continue
            bin_label, bin_mid = b
            row = {
                "detector_name": rec.get("detector_name", ""),
                "source_path": rec.get("source_path", ""),
                "condition": condition,
                "rep": rep,
                "parse_ok": ok,
                "distance_to_sign_m": d_to,
                "distance_bin": bin_label,
                "distance_bin_mid": bin_mid,
                "target_conf": _safe_float(rec.get("target_conf")),
                "target_missing": _safe_bool(rec.get("target_missing")),
            }
            row["detect"] = 0.0 if row["target_missing"] else 1.0
            rows_raw.append(row)

    if not rows_raw:
        raise ValueError("No rows parsed. Ensure input CSV has distance_to_sign_m and expected source video names.")

    # Step 1: per-video-per-bin metrics to avoid overweighting long clips.
    g1: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for r in rows_raw:
        k = (r["detector_name"], r["source_path"], r["condition"], r["rep"], r["distance_bin"], r["distance_bin_mid"])
        g1[k].append(r)

    video_bin_rows: list[dict[str, Any]] = []
    for k, grp in g1.items():
        det, src, cond, rep, dlab, dmid = k
        confs = [float(x["target_conf"]) for x in grp]
        dets = [float(x["detect"]) for x in grp]
        video_bin_rows.append(
            {
                "Detector": det,
                "Video": src,
                "Condition": cond,
                "Rep": rep,
                "Distance Bin": dlab,
                "Distance Mid (m)": f"{float(dmid):.2f}",
                "Frames": len(grp),
                "Target Conf (video mean)": _mean(confs),
                "Detect Rate (video mean)": _mean(dets),
            }
        )

    # Step 2: aggregate across videos.
    g2: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for r in video_bin_rows:
        k = (r["Detector"], r["Condition"], r["Distance Bin"], r["Distance Mid (m)"])
        g2[k].append(r)

    long_rows: list[dict[str, Any]] = []
    for k, grp in g2.items():
        det, cond, dlab, dmid = k
        conf_mu = _mean([float(x["Target Conf (video mean)"]) for x in grp])
        conf_sd = _std([float(x["Target Conf (video mean)"]) for x in grp])
        dr_mu = _mean([float(x["Detect Rate (video mean)"]) for x in grp])
        dr_sd = _std([float(x["Detect Rate (video mean)"]) for x in grp])
        long_rows.append(
            {
                "Detector": det,
                "Condition": cond,
                "Distance Bin": dlab,
                "Distance Mid (m)": dmid,
                "N videos": len(grp),
                "Target Conf": _fmt_pm(conf_mu, conf_sd, 3),
                "Detect Rate": _fmt_pct_pm(dr_mu, dr_sd, 1),
            }
        )

    def _long_sort_key(r: dict[str, Any]) -> tuple:
        return (
            str(r["Detector"]),
            _sort_condition_key(r["Condition"]),
            -float(r["Distance Mid (m)"]),
        )

    long_rows.sort(key=_long_sort_key)

    # Pivots: per condition, rows=detector, cols=distance bins.
    by_condition_target: dict[str, list[dict[str, Any]]] = {}
    by_condition_detect: dict[str, list[dict[str, Any]]] = {}

    cond_set = sorted({str(r["Condition"]) for r in long_rows}, key=lambda c: _sort_condition_key(c))
    bin_set = sorted({str(r["Distance Bin"]) for r in long_rows},
                     key=lambda b: -float(next(r["Distance Mid (m)"] for r in long_rows if r["Distance Bin"] == b)))
    det_set = sorted({str(r["Detector"]) for r in long_rows})

    lookup = {(r["Detector"], r["Condition"], r["Distance Bin"]): r for r in long_rows}
    for cond in cond_set:
        t_rows: list[dict[str, Any]] = []
        d_rows: list[dict[str, Any]] = []
        for det in det_set:
            tr = {"Detector": det}
            dr = {"Detector": det}
            for b in bin_set:
                rr = lookup.get((det, cond, b))
                tr[b] = rr["Target Conf"] if rr else ""
                dr[b] = rr["Detect Rate"] if rr else ""
            t_rows.append(tr)
            d_rows.append(dr)
        by_condition_target[cond] = t_rows
        by_condition_detect[cond] = d_rows

    # Save outputs.
    path_video_bin_csv = out_dir / "video_bin_means.csv"
    path_long_csv = out_dir / "detector_distance_condition_long.csv"
    _write_csv(path_video_bin_csv, video_bin_rows)
    _write_csv(path_long_csv, long_rows)

    _table_pdf(
        long_rows,
        out_dir / "detector_distance_condition_long.pdf",
        title="Real-World Video Results by Distance Bin",
        subtitle="Per-video means aggregated across replicates (mean +/- std).",
        rows_per_page=int(args.rows_per_page),
    )

    for cond in cond_set:
        cond_slug = re.sub(r"[^a-z0-9_]+", "_", cond.lower()).strip("_")
        t_csv = out_dir / f"target_conf_{cond_slug}_pivot.csv"
        d_csv = out_dir / f"detect_rate_{cond_slug}_pivot.csv"
        t_pdf = out_dir / f"target_conf_{cond_slug}_pivot.pdf"
        d_pdf = out_dir / f"detect_rate_{cond_slug}_pivot.pdf"
        _write_csv(t_csv, by_condition_target[cond])
        _write_csv(d_csv, by_condition_detect[cond])
        _table_pdf(
            by_condition_target[cond],
            t_pdf,
            title=f"Target Confidence vs Distance ({cond})",
            subtitle="Rows: detectors | Cols: distance bins",
            rows_per_page=max(10, int(args.rows_per_page)),
        )
        _table_pdf(
            by_condition_detect[cond],
            d_pdf,
            title=f"Detect Rate vs Distance ({cond})",
            subtitle="Rows: detectors | Cols: distance bins",
            rows_per_page=max(10, int(args.rows_per_page)),
        )

    print(f"[SAVE] {path_video_bin_csv}")
    print(f"[SAVE] {path_long_csv}")
    print(f"[SAVE] {out_dir / 'detector_distance_condition_long.pdf'}")
    for cond in cond_set:
        cond_slug = re.sub(r"[^a-z0-9_]+", "_", cond.lower()).strip("_")
        print(f"[SAVE] {out_dir / f'target_conf_{cond_slug}_pivot.pdf'}")
        print(f"[SAVE] {out_dir / f'detect_rate_{cond_slug}_pivot.pdf'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
