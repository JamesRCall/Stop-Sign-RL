"""Aggregate greedy/random baseline summaries + PPO eval summary into one JSON."""
from __future__ import annotations

import os
import json
import argparse
from typing import List, Dict, Any

import numpy as np


def load_run_summaries(list_path: str) -> List[Dict[str, Any]]:
    runs = []
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
                    runs.append(json.load(sf))
            except Exception:
                continue
    return runs


def _mean(vals):
    vals = [v for v in vals if v is not None and not np.isnan(v)]
    return float(np.mean(vals)) if vals else float("nan")


def _std(vals):
    vals = [v for v in vals if v is not None and not np.isnan(v)]
    return float(np.std(vals)) if vals else float("nan")


def extract_metrics(runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    area = []
    iou = []
    mis = []
    drop_on = []

    for r in runs:
        final = r.get("final", {}) if isinstance(r, dict) else {}
        metrics = final.get("metrics", {}) if isinstance(final, dict) else {}
        area.append(metrics.get("total_area_mask_frac", final.get("area_frac", np.nan)))
        iou.append(metrics.get("mean_iou", np.nan))
        mis.append(metrics.get("misclass_rate", np.nan))
        drop_on.append(metrics.get("drop_on", final.get("drop_on", np.nan)))

    return {
        "n": len(runs),
        "mean_area_frac": _mean(area),
        "std_area_frac": _std(area),
        "mean_iou": _mean(iou),
        "std_iou": _std(iou),
        "mean_misclass_rate": _mean(mis),
        "std_misclass_rate": _std(mis),
        "mean_drop_on": _mean(drop_on),
        "std_drop_on": _std(drop_on),
    }


def main() -> None:
    ap = argparse.ArgumentParser("Aggregate greedy/random baseline metrics")
    ap.add_argument("--ppo-json", default="", help="Path to PPO eval summary JSON (optional).")
    ap.add_argument("--greedy-list", default="", help="File with greedy run dirs (one per line).")
    ap.add_argument("--random-list", default="", help="File with random run dirs (one per line).")
    ap.add_argument("--out", default="./_runs/baseline_compare_summary.json")
    args = ap.parse_args()

    greedy_runs = load_run_summaries(args.greedy_list)
    random_runs = load_run_summaries(args.random_list)

    out = {
        "greedy": extract_metrics(greedy_runs),
        "random": extract_metrics(random_runs),
    }

    if args.ppo_json and os.path.exists(args.ppo_json):
        with open(args.ppo_json, "r", encoding="utf-8") as f:
            out["ppo"] = json.load(f)
    else:
        out["ppo"] = None

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
