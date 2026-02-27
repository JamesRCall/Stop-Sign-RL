"""Aggregate greedy/random baseline summaries + PPO eval summary into one JSON."""
from __future__ import annotations

import os
import json
import argparse
from datetime import datetime
from typing import List, Dict, Any

import numpy as np


def load_run_summaries(list_path: str) -> List[Dict[str, Any]]:
    runs: List[Dict[str, Any]] = []
    if not list_path or not os.path.exists(list_path):
        return runs
    with open(list_path, "r", encoding="utf-8") as f:
        for line in f:
            path = line.strip()
            if not path:
                continue
            summ_path = os.path.join(path, "summary.json")
            if not os.path.exists(summ_path):
                continue
            try:
                with open(summ_path, "r", encoding="utf-8") as sf:
                    rec = json.load(sf)
                    rec["_run_dir"] = path
                    runs.append(rec)
            except Exception:
                continue
    return runs


def _as_float(value: Any) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return v if np.isfinite(v) else float("nan")


def _mean(vals: List[float]) -> float:
    vals = [v for v in vals if not np.isnan(v)]
    return float(np.mean(vals)) if vals else float("nan")


def _std(vals: List[float]) -> float:
    vals = [v for v in vals if not np.isnan(v)]
    return float(np.std(vals)) if vals else float("nan")


def _extract_episode_row_from_summary(run: Dict[str, Any], idx: int) -> Dict[str, Any]:
    details = run.get("episodes_detail", [])
    detail = details[0] if isinstance(details, list) and details else {}
    final = run.get("final", {}) if isinstance(run, dict) else {}
    metrics = final.get("metrics", {}) if isinstance(final, dict) else {}
    episode_meta = run.get("episode_meta", {}) if isinstance(run.get("episode_meta", {}), dict) else {}

    paired_seed = run.get("seed", episode_meta.get("reset_seed", detail.get("seed", None)))
    row = {
        "episode_index": int(idx),
        "seed": int(paired_seed) if paired_seed is not None else None,
        "episode_seed": int(detail.get("seed")) if detail.get("seed") is not None else None,
        "success": bool(detail.get("success", metrics.get("uv_success", False))),
        "steps": int(detail.get("steps", run.get("steps", 0))),
        "base_conf": _as_float(detail.get("base_conf", metrics.get("base_conf", metrics.get("c0_day", np.nan)))),
        "after_conf": _as_float(detail.get("after_conf", metrics.get("after_conf", metrics.get("c_on", np.nan)))),
        "drop_on": _as_float(detail.get("drop_on", metrics.get("drop_on", final.get("drop_on", np.nan)))),
        "area_frac": _as_float(detail.get("area_frac", metrics.get("total_area_mask_frac", final.get("area_frac", np.nan)))),
        "drop_per_area": _as_float(detail.get("drop_per_area", np.nan)),
        "mean_iou": _as_float(detail.get("mean_iou", metrics.get("mean_iou", np.nan))),
        "misclass_rate": _as_float(detail.get("misclass_rate", metrics.get("misclass_rate", np.nan))),
        "selected_cells": _as_float(detail.get("selected_cells", metrics.get("selected_cells", np.nan))),
        "runtime_sec": _as_float(detail.get("runtime_sec", run.get("runtime_total_sec", np.nan))),
        "runtime_per_step_sec": _as_float(detail.get("runtime_per_step_sec", run.get("runtime_per_step_sec", np.nan))),
        "run_id": str(run.get("run_id", "")),
        "run_dir": str(run.get("_run_dir", "")),
    }
    if np.isnan(row["drop_per_area"]) and not np.isnan(row["drop_on"]) and not np.isnan(row["area_frac"]) and row["area_frac"] > 0:
        row["drop_per_area"] = float(row["drop_on"] / row["area_frac"])
    return row


def _summary_from_rows(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    steps = [_as_float(r.get("steps", np.nan)) for r in rows]
    base = [_as_float(r.get("base_conf", np.nan)) for r in rows]
    after = [_as_float(r.get("after_conf", np.nan)) for r in rows]
    drop_on = [_as_float(r.get("drop_on", np.nan)) for r in rows]
    area = [_as_float(r.get("area_frac", np.nan)) for r in rows]
    eff = [_as_float(r.get("drop_per_area", np.nan)) for r in rows]
    iou = [_as_float(r.get("mean_iou", np.nan)) for r in rows]
    mis = [_as_float(r.get("misclass_rate", np.nan)) for r in rows]
    cells = [_as_float(r.get("selected_cells", np.nan)) for r in rows]
    runtime = [_as_float(r.get("runtime_sec", np.nan)) for r in rows]
    runtime_step = [_as_float(r.get("runtime_per_step_sec", np.nan)) for r in rows]
    success = [1.0 if bool(r.get("success", False)) else 0.0 for r in rows]
    return {
        "n": len(rows),
        "n_success": int(sum(success)),
        "success_rate": float(sum(success) / len(success)) if success else float("nan"),
        "mean_steps": _mean(steps),
        "std_steps": _std(steps),
        "mean_base_conf": _mean(base),
        "std_base_conf": _std(base),
        "mean_after_conf": _mean(after),
        "std_after_conf": _std(after),
        "mean_drop_on": _mean(drop_on),
        "std_drop_on": _std(drop_on),
        "mean_area_frac": _mean(area),
        "std_area_frac": _std(area),
        "mean_drop_per_area": _mean(eff),
        "std_drop_per_area": _std(eff),
        "mean_iou": _mean(iou),
        "std_iou": _std(iou),
        "mean_misclass_rate": _mean(mis),
        "std_misclass_rate": _std(mis),
        "mean_selected_cells": _mean(cells),
        "std_selected_cells": _std(cells),
        "mean_runtime_sec": _mean(runtime),
        "std_runtime_sec": _std(runtime),
        "mean_runtime_per_step_sec": _mean(runtime_step),
        "std_runtime_per_step_sec": _std(runtime_step),
    }


def _paired_delta(ref_rows: List[Dict[str, Any]], other_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    ref_by_seed = {int(r["seed"]): r for r in ref_rows if r.get("seed") is not None}
    oth_by_seed = {int(r["seed"]): r for r in other_rows if r.get("seed") is not None}
    common = sorted(set(ref_by_seed.keys()).intersection(oth_by_seed.keys()))
    metrics = [
        "success",
        "steps",
        "area_frac",
        "after_conf",
        "drop_on",
        "drop_per_area",
        "mean_iou",
        "misclass_rate",
        "runtime_sec",
        "runtime_per_step_sec",
    ]
    deltas: Dict[str, List[float]] = {k: [] for k in metrics}
    for seed in common:
        a = ref_by_seed[seed]
        b = oth_by_seed[seed]
        for m in metrics:
            av = 1.0 if m == "success" and bool(a.get("success", False)) else _as_float(a.get(m, np.nan))
            bv = 1.0 if m == "success" and bool(b.get("success", False)) else _as_float(b.get(m, np.nan))
            if not np.isnan(av) and not np.isnan(bv):
                deltas[m].append(float(av - bv))
    out: Dict[str, Any] = {"n_pairs": len(common), "seeds": common}
    for m in metrics:
        out[f"mean_delta_{m}"] = _mean(deltas[m])
        out[f"std_delta_{m}"] = _std(deltas[m])
    return out


def main() -> None:
    ap = argparse.ArgumentParser("Aggregate greedy/random baseline metrics")
    ap.add_argument("--ppo-json", default="", help="Path to PPO eval summary JSON (optional).")
    ap.add_argument("--greedy-list", default="", help="File with greedy run dirs (one per line).")
    ap.add_argument("--random-list", default="", help="File with random run dirs (one per line).")
    ap.add_argument("--out", default="./_runs/baseline_compare_summary.json")
    args = ap.parse_args()

    greedy_runs = load_run_summaries(args.greedy_list)
    random_runs = load_run_summaries(args.random_list)
    greedy_rows = [_extract_episode_row_from_summary(r, i) for i, r in enumerate(greedy_runs)]
    random_rows = [_extract_episode_row_from_summary(r, i) for i, r in enumerate(random_runs)]

    ppo_obj = None
    ppo_rows: List[Dict[str, Any]] = []
    if args.ppo_json and os.path.exists(args.ppo_json):
        with open(args.ppo_json, "r", encoding="utf-8") as f:
            ppo_obj = json.load(f)
        ppo_details = ppo_obj.get("episodes_detail", []) if isinstance(ppo_obj, dict) else []
        if isinstance(ppo_details, list):
            for i, row in enumerate(ppo_details):
                if not isinstance(row, dict):
                    continue
                ppo_rows.append({
                    "episode_index": int(row.get("episode_index", i)),
                    "seed": int(row["seed"]) if row.get("seed") is not None else None,
                    "episode_seed": int(row["seed"]) if row.get("seed") is not None else None,
                    "success": bool(row.get("success", False)),
                    "steps": int(row.get("steps", 0)),
                    "base_conf": _as_float(row.get("base_conf", np.nan)),
                    "after_conf": _as_float(row.get("after_conf", np.nan)),
                    "drop_on": _as_float(row.get("drop_on", np.nan)),
                    "area_frac": _as_float(row.get("area_frac", np.nan)),
                    "drop_per_area": _as_float(row.get("drop_per_area", np.nan)),
                    "mean_iou": _as_float(row.get("mean_iou", np.nan)),
                    "misclass_rate": _as_float(row.get("misclass_rate", np.nan)),
                    "selected_cells": _as_float(row.get("selected_cells", np.nan)),
                    "runtime_sec": _as_float(row.get("runtime_sec", np.nan)),
                    "runtime_per_step_sec": _as_float(row.get("runtime_per_step_sec", np.nan)),
                    "run_id": str(ppo_obj.get("model", "")),
                    "run_dir": "",
                })

    out = {
        "generated_at_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "greedy": _summary_from_rows(greedy_rows),
        "random": _summary_from_rows(random_rows),
        "ppo": ppo_obj,
        "episodes": {
            "greedy": greedy_rows,
            "random": random_rows,
            "ppo": ppo_rows,
        },
        "paired_by_seed": {
            "ppo_minus_greedy": _paired_delta(ppo_rows, greedy_rows) if ppo_rows else {"n_pairs": 0, "seeds": []},
            "ppo_minus_random": _paired_delta(ppo_rows, random_rows) if ppo_rows else {"n_pairs": 0, "seeds": []},
        },
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
