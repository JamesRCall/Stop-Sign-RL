"""Greedy grid baseline: choose the best next cell at each step."""
from __future__ import annotations

import os
import sys
import json
import time
import argparse
import random
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from torch.utils.tensorboard import SummaryWriter
from baselines.grid_utils import build_env_from_args, save_final_images, info_metrics, log_metrics_tb


def snapshot_state(env):
    return {
        "episode_cells": env._episode_cells.copy(),
        "step": int(env._step),
        "last_drop_on_s": float(env._last_drop_on_s),
        "diag_saved": bool(env._diag_saved),
    }


def restore_state(env, state):
    env._episode_cells = state["episode_cells"].copy()
    env._step = int(state["step"])
    env._last_drop_on_s = float(state["last_drop_on_s"])
    env._diag_saved = bool(state["diag_saved"])


def score_from(info, reward, mode: str) -> float:
    mode = str(mode or "reward").lower()
    if mode == "reward":
        return float(reward)
    if mode == "drop_on":
        return float(info.get("drop_on", -1e9))
    if mode == "reward_raw_total":
        return float(info.get("reward_raw_total", reward))
    if mode == "drop_on_smooth":
        return float(info.get("drop_on_smooth", info.get("drop_on", -1e9)))
    return float(reward)


def parse_args():
    ap = argparse.ArgumentParser("Greedy grid baseline (StopSignGridEnv)")
    ap.add_argument("--data", default="./data")
    ap.add_argument("--bgdir", default="./data/backgrounds")
    ap.add_argument("--bg-mode", choices=["dataset", "solid"], default="dataset")
    ap.add_argument("--no-pole", action="store_true")

    ap.add_argument("--yolo-weights", default=None)
    ap.add_argument("--yolo-version", choices=["8", "11"], default="8")
    ap.add_argument("--detector", choices=["yolo", "torchvision", "rtdetr"], default="yolo")
    ap.add_argument("--detector-model", default="", help="Torchvision/RT-DETR detector model id.")
    ap.add_argument("--detector-device", default="auto")
    ap.add_argument("--detector-debug", type=int, default=0)

    ap.add_argument("--eval-K", type=int, default=3)
    ap.add_argument("--grid-cell", type=int, default=16, choices=[2, 4, 8, 16, 32])
    ap.add_argument("--episode-steps", type=int, default=300)
    ap.add_argument("--transform-strength", type=float, default=1.0)
    ap.add_argument("--day-tolerance", type=float, default=0.05)

    ap.add_argument("--lambda-area", type=float, default=0.70)
    ap.add_argument("--lambda-efficiency", type=float, default=0.40)
    ap.add_argument("--efficiency-eps", type=float, default=0.02)
    ap.add_argument("--lambda-day", type=float, default=0.0)
    ap.add_argument("--lambda-iou", type=float, default=0.40)
    ap.add_argument("--lambda-misclass", type=float, default=0.60)
    ap.add_argument("--lambda-perceptual", type=float, default=0.0)
    ap.add_argument("--area-target", type=float, default=0.25)
    ap.add_argument("--step-cost", type=float, default=0.012)
    ap.add_argument("--step-cost-after-target", type=float, default=0.14)
    ap.add_argument("--area-cap-frac", type=float, default=0.30)
    ap.add_argument("--area-cap-penalty", type=float, default=-0.20)
    ap.add_argument("--area-cap-mode", choices=["soft", "hard"], default="soft")
    ap.add_argument("--uv-threshold", type=float, default=0.75)
    ap.add_argument("--success-conf", type=float, default=0.20)

    ap.add_argument("--paint", default="yellow")
    ap.add_argument("--paint-list", default="")
    ap.add_argument("--cell-cover-thresh", type=float, default=0.60)

    ap.add_argument("--obs-size", type=int, default=224)
    ap.add_argument("--obs-margin", type=float, default=0.10)
    ap.add_argument("--obs-include-mask", type=int, default=1)

    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--select-by", choices=["reward", "drop_on", "reward_raw_total", "drop_on_smooth"], default="reward")
    ap.add_argument("--out", default="./baselines/greedy_grid/_runs")
    ap.add_argument("--tb", default="", help="TensorBoard log dir (default: <run_dir>/tb).")
    return ap.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    run_id = f"greedy_seed{int(args.seed)}_{time.strftime('%Y%m%d_%H%M%S')}"
    out_dir = os.path.join(args.out, run_id)
    os.makedirs(out_dir, exist_ok=True)
    tb_dir = args.tb if args.tb else os.path.join(out_dir, "tb")
    writer = SummaryWriter(log_dir=tb_dir)

    env = build_env_from_args(args)
    env.reset(seed=int(args.seed))
    episode_meta = {
        "reset_seed": int(args.seed),
        "place_seed": int(env._place_seed) if env._place_seed is not None else None,
        "transform_seeds": list(env._transform_seeds) if getattr(env, "_transform_seeds", None) else [],
    }

    action_seq = []
    step_logs = []

    done = False
    start = time.perf_counter()
    step_idx = 0
    step_runtime_sec_list = []
    while not done:
        step_t0 = time.perf_counter()
        masks = env.action_masks()
        candidates = np.where(masks)[0]
        if candidates.size == 0:
            break

        base = snapshot_state(env)
        best = None
        best_state = None
        best_out = None

        for i, a in enumerate(candidates):
            obs, reward, term, trunc, info = env.step(int(a))
            score = score_from(info, reward, args.select_by)
            state_after = snapshot_state(env)

            if (best is None) or (score > best):
                best = score
                best_state = state_after
                best_out = (obs, reward, term, trunc, info, int(a), score)

            restore_state(env, base)
            if (i + 1) % 50 == 0 or (i + 1) == candidates.size:
                elapsed = time.perf_counter() - start
                print(f"[step {step_idx+1}] eval {i+1}/{candidates.size} | elapsed {elapsed/60:.1f}m")

        if best_out is None:
            break

        # Commit best action (restore to its state)
        restore_state(env, best_state)
        _, reward, term, trunc, info, a, score = best_out
        action_seq.append(a)
        metrics = info_metrics(info)
        step_logs.append({
            "step": int(env._step),
            "action": int(a),
            "reward": float(reward),
            "score": float(score),
            "drop_on": float(info.get("drop_on", 0.0)),
            "drop_day": float(info.get("drop_day", 0.0)),
            "area_frac": float(info.get("total_area_mask_frac", 0.0)),
            "metrics": metrics,
        })

        done = bool(term) or bool(trunc)
        step_idx += 1
        step_runtime_sec = float(time.perf_counter() - step_t0)
        step_runtime_sec_list.append(step_runtime_sec)
        elapsed = time.perf_counter() - start
        print(f"[step {step_idx}] best_score={best:.4f} drop_on={step_logs[-1]['drop_on']:.4f} area={step_logs[-1]['area_frac']:.4f} elapsed {elapsed/60:.1f}m")
        writer.add_scalar("metrics/reward", float(reward), step_idx)
        writer.add_scalar("metrics/score", float(score), step_idx)
        writer.add_scalar("metrics/drop_on", float(info.get("drop_on", 0.0)), step_idx)
        writer.add_scalar("metrics/drop_day", float(info.get("drop_day", 0.0)), step_idx)
        writer.add_scalar("metrics/area_frac", float(info.get("total_area_mask_frac", 0.0)), step_idx)
        log_metrics_tb(writer, metrics, step_idx, prefix="env/")

    save_final_images(env, out_dir)
    final_step = step_logs[-1] if step_logs else {}
    final_metrics = final_step.get("metrics", {}) if isinstance(final_step, dict) else {}
    final_success = bool(final_metrics.get("uv_success", False))
    area_frac = float(final_metrics.get("total_area_mask_frac", final_step.get("area_frac", np.nan))) if final_step else float("nan")
    base_conf = float(final_metrics.get("base_conf", final_metrics.get("c0_day", np.nan))) if final_step else float("nan")
    after_conf = float(final_metrics.get("after_conf", final_metrics.get("c_on", np.nan))) if final_step else float("nan")
    drop_on = float(final_metrics.get("drop_on", final_step.get("drop_on", np.nan))) if final_step else float("nan")
    mean_iou = float(final_metrics.get("mean_iou", np.nan)) if final_step else float("nan")
    misclass = float(final_metrics.get("misclass_rate", np.nan)) if final_step else float("nan")
    selected_cells = float(final_metrics.get("selected_cells", np.nan)) if final_step else float("nan")
    drop_per_area = float("nan")
    if np.isfinite(drop_on) and np.isfinite(area_frac) and area_frac > 0:
        drop_per_area = float(drop_on / area_frac)
    runtime_total_sec = float(time.perf_counter() - start)
    runtime_per_step_sec = float(runtime_total_sec / len(step_logs)) if step_logs else float("nan")
    summary = {
        "method": "greedy",
        "run_id": run_id,
        "seed": int(args.seed),
        "detector": str(args.detector),
        "detector_model": str(args.detector_model),
        "select_by": args.select_by,
        "actions": action_seq,
        "final": final_step,
        "steps": len(step_logs),
        "n_success": 1 if final_success else 0,
        "success_rate": 1.0 if final_success else 0.0,
        "mean_steps": float(len(step_logs)),
        "mean_base_conf": base_conf,
        "mean_after_conf": after_conf,
        "mean_drop_on": drop_on,
        "mean_area_frac": area_frac,
        "mean_drop_per_area": drop_per_area,
        "mean_iou": mean_iou,
        "mean_misclass_rate": misclass,
        "mean_selected_cells": selected_cells,
        "runtime_total_sec": runtime_total_sec,
        "runtime_per_step_sec": runtime_per_step_sec,
        "runtime_per_step_mean_sec": float(np.mean(step_runtime_sec_list)) if step_runtime_sec_list else float("nan"),
        "runtime_per_step_std_sec": float(np.std(step_runtime_sec_list)) if step_runtime_sec_list else float("nan"),
        "episodes_detail": [{
            "episode_index": 0,
            "seed": int(args.seed),
            "success": bool(final_success),
            "steps": int(len(step_logs)),
            "base_conf": base_conf,
            "after_conf": after_conf,
            "drop_on": drop_on,
            "area_frac": area_frac,
            "drop_per_area": drop_per_area,
            "mean_iou": mean_iou,
            "misclass_rate": misclass,
            "selected_cells": selected_cells,
            "runtime_sec": runtime_total_sec,
            "runtime_per_step_sec": runtime_per_step_sec,
        }],
        "episode_meta": episode_meta,
        "config": vars(args),
    }
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(out_dir, "steps.json"), "w", encoding="utf-8") as f:
        json.dump(step_logs, f, indent=2)

    writer.close()
    print(f"[DONE] Saved outputs to {out_dir}")


if __name__ == "__main__":
    main()
