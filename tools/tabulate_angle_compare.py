#!/usr/bin/env python3
"""Build angle-robustness tables from compare_summary.json files.

Expected input files are outputs of tools/aggregate_baselines.py and include:
  - ppo (eval summary JSON object)
  - greedy (aggregated summary)
  - random (aggregated summary)

This script emits:
  - angle_compare_long.csv
  - angle_compare_wide.csv
  - angle_compare_table.tex
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _as_float(x: Any) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return float("nan")
    return v if math.isfinite(v) else float("nan")


def _pm(mean_v: float, std_v: float, nd: int = 3) -> str:
    if not math.isfinite(mean_v):
        return "nan"
    if not math.isfinite(std_v):
        return f"{mean_v:.{nd}f}"
    return f"{mean_v:.{nd}f} +/- {std_v:.{nd}f}"


def _extract_angle_deg(obj: Dict[str, Any], path: Path) -> float:
    # Prefer explicit config from PPO eval summary nested inside compare_summary.
    ppo = obj.get("ppo", {})
    if isinstance(ppo, dict):
        cfg = ppo.get("config", {})
        if isinstance(cfg, dict):
            angle = cfg.get("fixed_angle_deg", None)
            if angle is not None:
                return _as_float(angle)

    # Fallback: parse from path name tokens such as angle12 / a-8.5 / deg_15
    text = str(path)
    patterns = [
        r"angle[_-]?(-?\d+(?:\.\d+)?)",
        r"deg[_-]?(-?\d+(?:\.\d+)?)",
        r"\ba(-?\d+(?:\.\d+)?)\b",
    ]
    for pat in patterns:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            return _as_float(m.group(1))
    return float("nan")


def _extract_method_row(angle: float, method: str, summ: Dict[str, Any], src: Path) -> Dict[str, Any]:
    row = {
        "angle_deg": angle,
        "method": method,
        "n": int(summ.get("n", 0)) if method != "ppo" else int(summ.get("episodes", 0)),
        "success_rate": _as_float(summ.get("success_rate", float("nan"))),
        "mean_after_conf": _as_float(summ.get("mean_after_conf", float("nan"))),
        "std_after_conf": _as_float(summ.get("std_after_conf", float("nan"))),
        "mean_drop_on": _as_float(summ.get("mean_drop_on", float("nan"))),
        "std_drop_on": _as_float(summ.get("std_drop_on", float("nan"))),
        "mean_steps": _as_float(summ.get("mean_steps", float("nan"))),
        "std_steps": _as_float(summ.get("std_steps", float("nan"))),
        "mean_area_frac": _as_float(summ.get("mean_area_frac", float("nan"))),
        "std_area_frac": _as_float(summ.get("std_area_frac", float("nan"))),
        "mean_runtime_sec": _as_float(
            summ.get("mean_runtime_sec", summ.get("runtime_per_episode_mean_sec", float("nan")))
        ),
        "std_runtime_sec": _as_float(
            summ.get("std_runtime_sec", summ.get("runtime_per_episode_std_sec", float("nan")))
        ),
        "source_json": str(src),
    }
    return row


def _load_compare(path: Path) -> List[Dict[str, Any]]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    angle = _extract_angle_deg(obj, path)
    rows: List[Dict[str, Any]] = []

    ppo = obj.get("ppo", {})
    if isinstance(ppo, dict) and ppo:
        rows.append(_extract_method_row(angle, "ppo", ppo, path))

    greedy = obj.get("greedy", {})
    if isinstance(greedy, dict) and greedy:
        rows.append(_extract_method_row(angle, "greedy", greedy, path))

    random = obj.get("random", {})
    if isinstance(random, dict) and random:
        rows.append(_extract_method_row(angle, "random", random, path))

    return rows


def _write_csv(path: Path, rows: List[Dict[str, Any]], fields: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _build_wide(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_angle: Dict[float, Dict[str, Any]] = {}
    for r in rows:
        a = r["angle_deg"]
        m = r["method"]
        if a not in by_angle:
            by_angle[a] = {"angle_deg": a}
        by_angle[a][f"{m}_success_rate"] = r["success_rate"]
        by_angle[a][f"{m}_after_conf"] = r["mean_after_conf"]
        by_angle[a][f"{m}_drop_on"] = r["mean_drop_on"]
        by_angle[a][f"{m}_steps"] = r["mean_steps"]
        by_angle[a][f"{m}_area_frac"] = r["mean_area_frac"]
        by_angle[a][f"{m}_runtime_sec"] = r["mean_runtime_sec"]
    return [by_angle[k] for k in sorted(by_angle.keys())]


def _build_latex(rows: List[Dict[str, Any]]) -> str:
    # Keep a compact table with core paper metrics.
    by_angle_method: Dict[Tuple[float, str], Dict[str, Any]] = {
        (float(r["angle_deg"]), str(r["method"])): r for r in rows
    }
    angles = sorted({float(r["angle_deg"]) for r in rows if math.isfinite(float(r["angle_deg"]))})
    methods = ["ppo", "greedy", "random"]

    out: List[str] = []
    out.append("\\begin{tabular}{l l c c c}")
    out.append("\\toprule")
    out.append("Angle & Method & Success $\\uparrow$ & After-conf $\\downarrow$ & Drop-on $\\uparrow$ \\\\")
    out.append("\\midrule")
    for a in angles:
        for i, m in enumerate(methods):
            r = by_angle_method.get((a, m))
            if r is None:
                continue
            angle_label = f"{a:.1f}^\\circ" if i == 0 else ""
            succ = _as_float(r.get("success_rate"))
            succ_s = f"{100.0 * succ:.1f}\\%" if math.isfinite(succ) else "nan"
            after_s = _pm(_as_float(r.get("mean_after_conf")), _as_float(r.get("std_after_conf")), nd=3)
            drop_s = _pm(_as_float(r.get("mean_drop_on")), _as_float(r.get("std_drop_on")), nd=3)
            out.append(f"{angle_label} & {m.upper()} & {succ_s} & {after_s} & {drop_s} \\\\")
        out.append("\\midrule")
    if out and out[-1] == "\\midrule":
        out = out[:-1]
    out.append("\\bottomrule")
    out.append("\\end{tabular}")
    return "\n".join(out) + "\n"


def main() -> int:
    p = argparse.ArgumentParser(description="Tabulate dedicated-angle PPO vs greedy/random compare summaries.")
    p.add_argument(
        "--inputs",
        nargs="*",
        default=[],
        help="Explicit list of compare_summary.json paths.",
    )
    p.add_argument(
        "--input-glob",
        default="",
        help="Glob for compare_summary.json files, e.g. '_runs/paper_data/compare/*/compare_summary.json'.",
    )
    p.add_argument(
        "--out-dir",
        default="_runs/paper_data/compare/angle_tables",
        help="Output directory for CSV/TEX tables.",
    )
    args = p.parse_args()

    paths: List[Path] = []
    for s in args.inputs:
        pp = Path(s)
        if pp.is_file():
            paths.append(pp)
    if args.input_glob:
        paths.extend(Path(p) for p in glob.glob(args.input_glob))
    # de-duplicate while preserving order
    seen = set()
    uniq: List[Path] = []
    for pth in paths:
        k = str(pth.resolve())
        if k in seen:
            continue
        seen.add(k)
        uniq.append(pth)
    paths = uniq
    if not paths:
        raise FileNotFoundError("No input compare_summary.json files found. Use --inputs or --input-glob.")

    rows: List[Dict[str, Any]] = []
    for path in paths:
        rows.extend(_load_compare(path))

    rows.sort(key=lambda r: (float(r.get("angle_deg", float("inf"))), str(r.get("method", ""))))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    long_fields = [
        "angle_deg",
        "method",
        "n",
        "success_rate",
        "mean_after_conf",
        "std_after_conf",
        "mean_drop_on",
        "std_drop_on",
        "mean_steps",
        "std_steps",
        "mean_area_frac",
        "std_area_frac",
        "mean_runtime_sec",
        "std_runtime_sec",
        "source_json",
    ]
    long_csv = out_dir / "angle_compare_long.csv"
    _write_csv(long_csv, rows, long_fields)

    wide_rows = _build_wide(rows)
    wide_csv = out_dir / "angle_compare_wide.csv"
    if wide_rows:
        wide_fields = sorted({k for r in wide_rows for k in r.keys()}, key=lambda x: (x != "angle_deg", x))
        _write_csv(wide_csv, wide_rows, wide_fields)
    else:
        wide_csv.write_text("", encoding="utf-8")

    tex_path = out_dir / "angle_compare_table.tex"
    tex_path.write_text(_build_latex(rows), encoding="utf-8")

    print(f"[SAVE] {long_csv}")
    print(f"[SAVE] {wide_csv}")
    print(f"[SAVE] {tex_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

