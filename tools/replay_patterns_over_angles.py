#!/usr/bin/env python3
"""Replay saved PPO/greedy/random patterns across fixed angles.

This script reuses already-saved patterns from baseline-compare outputs:
  - PPO patterns from ppo_episodes.json -> trace.selected_indices
  - Greedy/Random patterns from summary.json -> actions

It evaluates each fixed pattern across user-specified angles, without re-running
search, and writes raw + aggregated CSV tables.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Tuple

import numpy as np

from baselines.grid_utils import build_env_from_args


def _as_float(v: Any, default: float = float("nan")) -> float:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return default
    return x if math.isfinite(x) else default


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_angles(s: str) -> List[float]:
    out: List[float] = []
    for p in str(s).split(","):
        p = p.strip()
        if not p:
            continue
        out.append(float(p))
    return out


def _load_ppo_patterns(compare_dir: Path) -> List[Dict[str, Any]]:
    p = compare_dir / "ppo_episodes.json"
    if not p.is_file():
        return []
    rows = json.loads(p.read_text(encoding="utf-8"))
    out: List[Dict[str, Any]] = []
    if not isinstance(rows, list):
        return out
    for r in rows:
        if not isinstance(r, dict):
            continue
        seed = r.get("seed", None)
        tr = r.get("trace", {})
        tr = tr if isinstance(tr, dict) else {}
        sel = tr.get("selected_indices", [])
        if seed is None or not isinstance(sel, list) or not sel:
            continue
        out.append(
            {
                "method": "ppo",
                "seed": int(seed),
                "pattern_type": "selected_indices",
                "pattern": [int(x) for x in sel],
            }
        )
    return out


def _load_action_patterns_from_list(list_path: Path, method: str) -> List[Dict[str, Any]]:
    if not list_path.is_file():
        return []
    out: List[Dict[str, Any]] = []
    for line in list_path.read_text(encoding="utf-8").splitlines():
        run_dir = line.strip()
        if not run_dir:
            continue
        s = Path(run_dir) / "summary.json"
        if not s.is_file():
            continue
        try:
            obj = _read_json(s)
        except Exception:
            continue
        seed = obj.get("seed", None)
        actions = obj.get("actions", [])
        if seed is None or not isinstance(actions, list) or not actions:
            continue
        out.append(
            {
                "method": method,
                "seed": int(seed),
                "pattern_type": "actions",
                "pattern": [int(a) for a in actions],
            }
        )
    return out


def _default_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    d = dict(cfg)
    d.setdefault("data", "./data")
    d.setdefault("bgdir", "./data/backgrounds")
    d.setdefault("bg_mode", "dataset")
    d.setdefault("no_pole", False)
    d.setdefault("yolo_version", "8")
    d.setdefault("yolo_weights", None)
    d.setdefault("detector", "yolo")
    d.setdefault("detector_model", "")
    d.setdefault("detector_device", "auto")
    d.setdefault("detector_debug", 0)
    d.setdefault("eval_K", 3)
    d.setdefault("grid_cell", 16)
    d.setdefault("episode_steps", 300)
    d.setdefault("transform_strength", 1.0)
    d.setdefault("fixed_angle_deg", None)
    d.setdefault("day_tolerance", 0.05)
    d.setdefault("lambda_area", 0.70)
    d.setdefault("lambda_efficiency", 0.40)
    d.setdefault("efficiency_eps", 0.02)
    d.setdefault("lambda_day", 0.0)
    d.setdefault("lambda_iou", 0.40)
    d.setdefault("lambda_misclass", 0.60)
    d.setdefault("lambda_perceptual", 0.0)
    d.setdefault("area_target", 0.25)
    d.setdefault("step_cost", 0.012)
    d.setdefault("step_cost_after_target", 0.14)
    d.setdefault("area_cap_frac", 0.30)
    d.setdefault("area_cap_penalty", -0.20)
    d.setdefault("area_cap_mode", "soft")
    d.setdefault("uv_threshold", 0.75)
    d.setdefault("success_conf", 0.20)
    d.setdefault("paint", "yellow")
    d.setdefault("paint_list", "")
    d.setdefault("cell_cover_thresh", 0.60)
    d.setdefault("obs_size", 224)
    d.setdefault("obs_margin", 0.10)
    d.setdefault("obs_include_mask", 1)
    return d


def _build_env_args(base_cfg: Dict[str, Any], angle: float, eval_k: int, detector_device: str) -> SimpleNamespace:
    cfg = _default_cfg(base_cfg)
    cfg["transform_strength"] = 0.0
    cfg["fixed_angle_deg"] = float(angle)
    cfg["eval_K"] = int(eval_k)
    cfg["detector_device"] = detector_device
    return SimpleNamespace(**cfg)


def _apply_pattern(env, pattern_type: str, pattern: List[int]) -> None:
    env._episode_cells[:] = False
    if pattern_type == "selected_indices":
        for idx in pattern:
            i = int(idx)
            if i < 0 or i >= (env.Gh * env.Gw):
                continue
            r, c = divmod(i, env.Gw)
            if env._valid_cells[r, c]:
                env._episode_cells[r, c] = True
        return
    if pattern_type == "actions":
        for a in pattern:
            i = int(a)
            if i < 0 or i >= int(env._n_valid):
                continue
            rr, cc = env._valid_coords[i]
            env._episode_cells[int(rr), int(cc)] = True
        return
    raise ValueError(f"Unknown pattern_type: {pattern_type}")


def _eval_pattern(env, eval_k: int) -> Dict[str, float]:
    k = max(1, min(int(eval_k), len(env._transform_seeds)))
    seeds = env._transform_seeds[:k]
    overlay = env._eval_overlay_over_K(seeds)
    c_day = _as_float(overlay.get("c_day", float("nan")))
    c_on = _as_float(overlay.get("c_on", float("nan")))
    c0_day = _as_float(env._mean_over_K(env._baseline_c0_day_list, k))
    drop_on = _as_float(c0_day - c_on)
    area_frac = _as_float(env._area_frac_selected())
    success = 1.0 if (math.isfinite(c_on) and c_on <= float(env.success_conf_threshold)) else 0.0
    return {
        "c0_day": c0_day,
        "c_day": c_day,
        "c_on": c_on,
        "drop_on": drop_on,
        "area_frac": area_frac,
        "success": success,
    }


def _summarize(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    key_to_vals: Dict[Tuple[str, str, float], List[Dict[str, Any]]] = {}
    for r in rows:
        k = (str(r["detector"]), str(r["method"]), float(r["angle_deg"]))
        key_to_vals.setdefault(k, []).append(r)

    out: List[Dict[str, Any]] = []
    for (det, method, angle), vals in sorted(key_to_vals.items(), key=lambda x: (x[0][0], x[0][2], x[0][1])):
        def arr(name: str) -> np.ndarray:
            return np.array([_as_float(v.get(name)) for v in vals], dtype=np.float64)
        succ = arr("success")
        after = arr("c_on")
        drop = arr("drop_on")
        area = arr("area_frac")
        out.append(
            {
                "detector": det,
                "method": method,
                "angle_deg": angle,
                "n": int(len(vals)),
                "success_rate": float(np.nanmean(succ)),
                "after_conf_mean": float(np.nanmean(after)),
                "after_conf_std": float(np.nanstd(after)),
                "drop_on_mean": float(np.nanmean(drop)),
                "drop_on_std": float(np.nanstd(drop)),
                "area_frac_mean": float(np.nanmean(area)),
                "area_frac_std": float(np.nanstd(area)),
            }
        )
    return out


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
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


def main() -> int:
    p = argparse.ArgumentParser(description="Replay saved PPO/greedy/random patterns across fixed angles.")
    p.add_argument("--compare-root", default="_runs/paper_data/compare")
    p.add_argument("--run-glob", default="*_N5_seed1000", help="Glob for detector compare run directories.")
    p.add_argument("--angles", default="-24,-18,-12,-6,0,6,12,18,24")
    p.add_argument("--eval-k", type=int, default=3)
    p.add_argument("--detector-device", default="auto")
    p.add_argument("--out-dir", default="_runs/paper_data/compare/angle_replay")
    args = p.parse_args()

    compare_root = Path(args.compare_root)
    run_dirs = sorted(Path(pth) for pth in glob.glob(str(compare_root / args.run_glob)) if Path(pth).is_dir())
    if not run_dirs:
        raise FileNotFoundError(f"No compare dirs found under {compare_root} with glob '{args.run_glob}'")

    angles = _parse_angles(args.angles)
    if not angles:
        raise ValueError("No angles parsed from --angles")

    all_rows: List[Dict[str, Any]] = []
    total_jobs = 0
    for d in run_dirs:
        ppo_summary = d / "ppo_summary.json"
        if not ppo_summary.is_file():
            print(f"[WARN] missing ppo_summary.json in {d}, skipping")
            continue
        ppo_obj = _read_json(ppo_summary)
        cfg = ppo_obj.get("config", {}) if isinstance(ppo_obj, dict) else {}
        cfg = cfg if isinstance(cfg, dict) else {}
        detector_name = d.name

        patterns = []
        patterns.extend(_load_ppo_patterns(d))
        patterns.extend(_load_action_patterns_from_list(d / "greedy_runs.txt", "greedy"))
        patterns.extend(_load_action_patterns_from_list(d / "random_runs.txt", "random"))
        if not patterns:
            print(f"[WARN] no patterns found in {d}, skipping")
            continue

        total_jobs += len(patterns) * len(angles)
        done = 0
        for pat in patterns:
            seed = int(pat["seed"])
            for angle in angles:
                env_args = _build_env_args(cfg, angle=angle, eval_k=int(args.eval_k), detector_device=str(args.detector_device))
                env = build_env_from_args(env_args)
                env.reset(seed=seed)
                _apply_pattern(env, pat["pattern_type"], pat["pattern"])
                m = _eval_pattern(env, eval_k=int(args.eval_k))
                row = {
                    "detector": detector_name,
                    "method": str(pat["method"]),
                    "seed": int(seed),
                    "angle_deg": float(angle),
                    "pattern_type": str(pat["pattern_type"]),
                    "pattern_len": int(len(pat["pattern"])),
                    **m,
                }
                all_rows.append(row)
                done += 1
                if done % 20 == 0:
                    print(f"[{detector_name}] progress {done}/{len(patterns)*len(angles)}")

    if not all_rows:
        raise RuntimeError("No rows generated. Check compare directories and input files.")

    summary_rows = _summarize(all_rows)
    out_dir = Path(args.out_dir)
    _write_csv(out_dir / "angle_replay_rows.csv", all_rows)
    _write_csv(out_dir / "angle_replay_summary.csv", summary_rows)
    print(f"[SAVE] {out_dir / 'angle_replay_rows.csv'}")
    print(f"[SAVE] {out_dir / 'angle_replay_summary.csv'}")
    print(f"[DONE] rows={len(all_rows)} summary_rows={len(summary_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

