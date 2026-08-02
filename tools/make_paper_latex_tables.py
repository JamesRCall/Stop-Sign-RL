#!/usr/bin/env python3
"""Generate LaTeX tables for paper experiments from _runs/paper_data JSON artifacts.

Outputs:
- ppo_eval_100eps_table.tex
- baseline_compare_table.tex
- angle_success_table.tex
- angle_after_conf_table.tex
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


DETECTOR_ORDER = [
    "yolo8",
    "yolo11",
    "fasterrcnn_v2",
    "fcos",
    "retinanet_v2",
    "rtdetr",
]

DETECTOR_LABEL = {
    "yolo8": "YOLOv8",
    "yolo11": "YOLO11",
    "fasterrcnn_v2": "Faster R-CNN (v2)",
    "fcos": "FCOS",
    "retinanet_v2": "RetinaNet (v2)",
    "rtdetr": "RT-DETR (R50)",
}


def _safe_float(x: Any, default: float = float("nan")) -> float:
    try:
        v = float(x)
        return v if math.isfinite(v) else default
    except Exception:
        return default


def _pct(x: float, nd: int = 1) -> str:
    if not math.isfinite(x):
        return "--"
    return f"{100.0 * x:.{nd}f}\\%"


def _f(x: float, nd: int = 3) -> str:
    if not math.isfinite(x):
        return "--"
    return f"{x:.{nd}f}"


def _pm(mu: float, sd: float, nd: int = 3) -> str:
    if not math.isfinite(mu):
        return "--"
    if not math.isfinite(sd):
        return f"{mu:.{nd}f}"
    return f"{mu:.{nd}f} $\\pm$ {sd:.{nd}f}"


def _tex_escape(s: str) -> str:
    repl = {
        "\\": r"\textbackslash{}",
        "_": r"\_",
        "&": r"\&",
        "%": r"\%",
        "#": r"\#",
        "{": r"\{",
        "}": r"\}",
        "$": r"\$",
    }
    out = str(s)
    for k, v in repl.items():
        out = out.replace(k, v)
    return out


def _detector_key_from_name(name: str) -> str:
    n = name.lower()
    if "yolo8" in n:
        return "yolo8"
    if "yolo11" in n:
        return "yolo11"
    if "faster" in n:
        return "fasterrcnn_v2"
    if "fcos" in n:
        return "fcos"
    if "retina" in n:
        return "retinanet_v2"
    if "rtdetr" in n:
        return "rtdetr"
    return n


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def collect_eval_100(eval_dir: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for p in sorted(eval_dir.glob("*_summary.json")):
        name = p.name.lower()
        if "stencil" in name:
            continue
        obj = _load_json(p)
        if int(obj.get("episodes", 0)) != 100:
            continue
        k = _detector_key_from_name(name)
        if k not in DETECTOR_ORDER:
            continue
        out[k] = obj
    return out


def collect_compare(compare_dir: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for d in sorted(compare_dir.glob("*_N5_seed1000")):
        p = d / "compare_summary.json"
        if not p.exists():
            continue
        obj = _load_json(p)
        k = _detector_key_from_name(d.name)
        if k not in DETECTOR_ORDER:
            continue
        out[k] = obj
    return out


def collect_angle(angle_summary_json: Path) -> list[dict[str, Any]]:
    if not angle_summary_json.exists():
        return []
    arr = json.loads(angle_summary_json.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for r in arr:
        if not isinstance(r, dict):
            continue
        det = _detector_key_from_name(str(r.get("detector", "")))
        if det not in DETECTOR_ORDER:
            continue
        rows.append(
            {
                "detector": det,
                "method": str(r.get("method", "")).lower(),
                "angle_deg": _safe_float(r.get("angle_deg")),
                "success_rate": _safe_float(r.get("success_rate")),
                "after_conf_mean": _safe_float(r.get("after_conf_mean")),
            }
        )
    return rows


def make_eval_table(eval_by_det: dict[str, dict[str, Any]]) -> str:
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\setlength{\tabcolsep}{5pt}")
    lines.append(r"\begin{tabular}{lccccc}")
    lines.append(r"\hline")
    lines.append(r"Detector & Joint ASR $\uparrow$ & Source conf. $\downarrow$ & Active drop $\uparrow$ & Localized alt. rate & Steps $\downarrow$ \\")
    lines.append(r"\hline")
    for k in DETECTOR_ORDER:
        obj = eval_by_det.get(k)
        if not obj:
            continue
        lines.append(
            f"{DETECTOR_LABEL[k]} & "
            f"{_pct(_safe_float(obj.get('success_rate')))} & "
            f"{_pm(_safe_float(obj.get('mean_after_conf')), _safe_float(obj.get('std_after_conf')), 3)} & "
            f"{_pm(_safe_float(obj.get('mean_drop_on')), _safe_float(obj.get('std_drop_on')), 3)} & "
            f"{_pm(_safe_float(obj.get('mean_misclass_rate')), _safe_float(obj.get('std_misclass_rate')), 3)} & "
            f"{_pm(_safe_float(obj.get('mean_steps')), _safe_float(obj.get('std_steps')), 1)} \\\\"
        )
    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    lines.append(
        r"\caption{Scene-conditioned PPO evaluation. Joint ASR requires clean paired-state eligibility, the configured ROI-localized attack objective over EOT samples, inactive-state preservation, and the material-area budget. Fixed-stencil certification is reported separately on held-out scenes.}"
    )
    lines.append(r"\label{tab:ppo_eval_100}")
    lines.append(r"\end{table}")
    return "\n".join(lines) + "\n"


def make_compare_table(compare_by_det: dict[str, dict[str, Any]]) -> str:
    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\setlength{\tabcolsep}{4pt}")
    lines.append(r"\begin{tabular}{llccccc}")
    lines.append(r"\hline")
    lines.append(r"Detector & Method & Success $\uparrow$ & After-conf $\downarrow$ & Drop-on $\uparrow$ & Misclass $\downarrow$ & Runtime(s/ep) $\downarrow$ \\")
    lines.append(r"\hline")
    for k in DETECTOR_ORDER:
        obj = compare_by_det.get(k)
        if not obj:
            continue
        for method in ("ppo", "greedy", "random"):
            mobj = obj.get(method, {}) if isinstance(obj, dict) else {}
            if not isinstance(mobj, dict) or not mobj:
                continue
            runtime = _safe_float(mobj.get("mean_runtime_sec", mobj.get("runtime_per_episode_mean_sec")))
            lines.append(
                f"{DETECTOR_LABEL[k]} & {method.upper()} & "
                f"{_pct(_safe_float(mobj.get('success_rate')))} & "
                f"{_pm(_safe_float(mobj.get('mean_after_conf')), _safe_float(mobj.get('std_after_conf')), 3)} & "
                f"{_pm(_safe_float(mobj.get('mean_drop_on')), _safe_float(mobj.get('std_drop_on')), 3)} & "
                f"{_pm(_safe_float(mobj.get('mean_misclass_rate')), _safe_float(mobj.get('std_misclass_rate')), 3)} & "
                f"{_f(runtime, 2)} \\\\"
            )
        lines.append(r"\hline")
    if lines and lines[-1] == r"\hline":
        lines = lines[:-1]
    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    lines.append(r"\caption{PPO vs greedy vs random baseline comparison (N=5 paired seeds per detector).}")
    lines.append(r"\label{tab:baseline_compare}")
    lines.append(r"\end{table*}")
    return "\n".join(lines) + "\n"


def _pivot_angle(rows: list[dict[str, Any]], value_key: str) -> list[dict[str, Any]]:
    idx = {}
    for r in rows:
        k = (r["detector"], r["angle_deg"])
        if k not in idx:
            idx[k] = {"detector": r["detector"], "angle_deg": r["angle_deg"]}
        idx[k][r["method"]] = r
    out = []
    for k in sorted(idx.keys(), key=lambda x: (DETECTOR_ORDER.index(x[0]), x[1])):
        rec = idx[k]
        ppo = rec.get("ppo", {})
        gr = rec.get("greedy", {})
        rnd = rec.get("random", {})
        out.append(
            {
                "detector": rec["detector"],
                "angle_deg": rec["angle_deg"],
                "ppo": _safe_float(ppo.get(value_key)),
                "greedy": _safe_float(gr.get(value_key)),
                "random": _safe_float(rnd.get(value_key)),
            }
        )
    return out


def make_angle_table(rows: list[dict[str, Any]], value_key: str, caption: str, label: str, percent: bool) -> str:
    piv = _pivot_angle(rows, value_key=value_key)
    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\setlength{\tabcolsep}{4pt}")
    lines.append(r"\begin{tabular}{lcccc}")
    lines.append(r"\hline")
    lines.append(r"Detector @ Angle & PPO & Greedy & Random & Best \\")
    lines.append(r"\hline")
    for r in piv:
        det = DETECTOR_LABEL.get(r["detector"], r["detector"])
        ang = _f(_safe_float(r["angle_deg"]), 0)
        vals = {"PPO": r["ppo"], "Greedy": r["greedy"], "Random": r["random"]}
        # success: higher is better; after-conf: lower is better
        if value_key == "after_conf_mean":
            best = min(vals.items(), key=lambda x: (math.inf if not math.isfinite(x[1]) else x[1]))[0]
        else:
            best = max(vals.items(), key=lambda x: (-math.inf if not math.isfinite(x[1]) else x[1]))[0]

        def fmt(v: float) -> str:
            if percent:
                return _pct(v, 1)
            return _f(v, 3)

        lines.append(
            f"{det} @ ${ang}^\\circ$ & {fmt(r['ppo'])} & {fmt(r['greedy'])} & {fmt(r['random'])} & {best} \\\\"
        )
    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    lines.append(rf"\caption{{{caption}}}")
    lines.append(rf"\label{{{label}}}")
    lines.append(r"\end{table*}")
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate LaTeX tables from paper_data artifacts.")
    ap.add_argument("--eval-dir", default="_runs/paper_data/eval")
    ap.add_argument("--compare-dir", default="_runs/paper_data/compare")
    ap.add_argument("--angle-summary-json", default="_runs/paper_data/compare/angle_replay_N5_seed1000/angle_replay_summary.json")
    ap.add_argument("--out-dir", default="_runs/paper_data/paper_tables_latex")
    args = ap.parse_args()

    eval_dir = Path(args.eval_dir)
    compare_dir = Path(args.compare_dir)
    angle_json = Path(args.angle_summary_json)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    eval_by_det = collect_eval_100(eval_dir)
    compare_by_det = collect_compare(compare_dir)
    angle_rows = collect_angle(angle_json)

    p1 = out_dir / "ppo_eval_100eps_table.tex"
    p2 = out_dir / "baseline_compare_table.tex"
    p3 = out_dir / "angle_success_table.tex"
    p4 = out_dir / "angle_after_conf_table.tex"

    p1.write_text(make_eval_table(eval_by_det), encoding="utf-8")
    p2.write_text(make_compare_table(compare_by_det), encoding="utf-8")
    p3.write_text(
        make_angle_table(
            angle_rows,
            value_key="success_rate",
            caption="Angle robustness: success rate comparison across PPO, greedy, and random baselines.",
            label="tab:angle_success",
            percent=True,
        ),
        encoding="utf-8",
    )
    p4.write_text(
        make_angle_table(
            angle_rows,
            value_key="after_conf_mean",
            caption="Angle robustness: mean post-attack confidence (lower is better).",
            label="tab:angle_after_conf",
            percent=False,
        ),
        encoding="utf-8",
    )

    print(f"[SAVE] {p1}")
    print(f"[SAVE] {p2}")
    print(f"[SAVE] {p3}")
    print(f"[SAVE] {p4}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
