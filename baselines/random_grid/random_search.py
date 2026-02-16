"""Random grid baseline: random action sequences, keep best."""
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


def score_from(info, reward, mode: str) -> float:
    mode = str(mode or "reward").lower()
    if mode == "reward":
        return float(reward)
    if mode == "drop_on":
        return float(info.get("drop_on", -1e9))
    if mode == "success_area":
        metrics = info.get("metrics", {}) if isinstance(info, dict) else {}
        uv_success = bool(metrics.get("uv_success", False))
        area = float(metrics.get("total_area_mask_frac", info.get("area_frac", 1.0)))
        if uv_success:
            return 1.0 - area  # higher is better -> lower area
        # If not successful, prefer higher drop_on but keep below any success.
        return -1e3 + float(metrics.get("drop_on", info.get("drop_on", -1e9)))
    if mode == "reward_raw_total":
        return float(info.get("reward_raw_total", reward))
    if mode == "drop_on_smooth":
        return float(info.get("drop_on_smooth", info.get("drop_on", -1e9)))
    return float(reward)


def parse_args():
    ap = argparse.ArgumentParser("Random grid baseline (StopSignGridEnv)")
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
    ap.add_argument("--trials", type=int, default=50)
    ap.add_argument("--select-by", choices=["reward", "drop_on", "reward_raw_total", "drop_on_smooth", "success_area"], default="success_area")
    ap.add_argument("--out", default="./baselines/random_grid/_runs")
    ap.add_argument("--tb", default="", help="TensorBoard log dir (default: <run_dir>/tb).")
    return ap.parse_args()


def run_random_episode(env, seed: int):
    random.seed(int(seed))
    np.random.seed(int(seed))
    env.reset(seed=int(seed))
    episode_meta = {
        "reset_seed": int(seed),
        "place_seed": int(env._place_seed) if env._place_seed is not None else None,
        "transform_seeds": list(env._transform_seeds) if getattr(env, "_transform_seeds", None) else [],
    }
    action_seq = []
    step_logs = []
    done = False
    while not done:
        masks = env.action_masks()
        candidates = np.where(masks)[0]
        if candidates.size == 0:
            break
        a = int(random.choice(candidates.tolist()))
        _, reward, term, trunc, info = env.step(a)
        action_seq.append(a)
        metrics = info_metrics(info)
        step_logs.append({
            "step": int(env._step),
            "action": int(a),
            "reward": float(reward),
            "drop_on": float(info.get("drop_on", 0.0)),
            "drop_day": float(info.get("drop_day", 0.0)),
            "area_frac": float(info.get("total_area_mask_frac", 0.0)),
            "metrics": metrics,
        })
        done = bool(term) or bool(trunc)
    final = step_logs[-1] if step_logs else {}
    return action_seq, step_logs, final, episode_meta


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    run_id = f"random_seed{int(args.seed)}_{time.strftime('%Y%m%d_%H%M%S')}"
    out_dir = os.path.join(args.out, run_id)
    os.makedirs(out_dir, exist_ok=True)
    tb_dir = args.tb if args.tb else os.path.join(out_dir, "tb")
    writer = SummaryWriter(log_dir=tb_dir)

    env = build_env_from_args(args)
    run_start = time.perf_counter()

    best = None
    best_trial = None
    best_actions = None
    best_steps = None
    best_meta = None
    trial_rows = []

    for t in range(int(args.trials)):
        trial_t0 = time.perf_counter()
        trial_seed = int(args.seed) + t
        actions, steps, final, meta = run_random_episode(env, trial_seed)
        trial_runtime_sec = float(time.perf_counter() - trial_t0)
        score = score_from(final, final.get("reward", 0.0), args.select_by)
        final_metrics = final.get("metrics", {}) if isinstance(final, dict) else {}
        area = final_metrics.get("total_area_mask_frac", final.get("area_frac", 0.0))
        after_conf = final_metrics.get("c_on", final.get("after_conf", None))
        base_conf = final_metrics.get("base_conf", final_metrics.get("c0_day", np.nan))
        drop_on = final_metrics.get("drop_on", final.get("drop_on", np.nan))
        mean_iou = final_metrics.get("mean_iou", np.nan)
        misclass = final_metrics.get("misclass_rate", np.nan)
        selected_cells = final_metrics.get("selected_cells", np.nan)
        success = bool(final_metrics.get("uv_success", False))
        steps_count = len(steps)
        drop_per_area = float("nan")
        if np.isfinite(float(drop_on)) and np.isfinite(float(area)) and float(area) > 0:
            drop_per_area = float(float(drop_on) / float(area))
        trial_rows.append({
            "trial_index": int(t),
            "trial_seed": int(trial_seed),
            "score": float(score),
            "success": bool(success),
            "steps": int(steps_count),
            "base_conf": float(base_conf),
            "after_conf": float(after_conf) if after_conf is not None else float("nan"),
            "drop_on": float(drop_on),
            "area_frac": float(area),
            "drop_per_area": float(drop_per_area),
            "mean_iou": float(mean_iou),
            "misclass_rate": float(misclass),
            "selected_cells": float(selected_cells),
            "runtime_sec": trial_runtime_sec,
            "runtime_per_step_sec": float(trial_runtime_sec / steps_count) if steps_count > 0 else float("nan"),
        })
        if (best is None) or (score > best):
            best = score
            best_trial = trial_seed
            best_actions = actions
            best_steps = steps
            best_meta = meta
        if (t + 1) % 5 == 0 or (t + 1) == int(args.trials):
            if after_conf is None:
                print(f"[trial {t+1}/{args.trials}] best_score={best:.4f} area={float(area):.4f}")
            else:
                print(f"[trial {t+1}/{args.trials}] best_score={best:.4f} area={float(area):.4f} c_on={float(after_conf):.4f}")
        writer.add_scalar("metrics/trial_score", float(score), t + 1)
        writer.add_scalar("metrics/best_score", float(best), t + 1)
        writer.add_scalar("metrics/drop_on", float(final.get("drop_on", 0.0)), t + 1)
        writer.add_scalar("metrics/drop_day", float(final.get("drop_day", 0.0)), t + 1)
        writer.add_scalar("metrics/area_frac", float(final.get("area_frac", 0.0)), t + 1)
        if isinstance(final, dict) and "metrics" in final:
            log_metrics_tb(writer, final["metrics"], t + 1, prefix="env/")

    # Replay best trial for images
    if best_trial is not None:
        run_random_episode(env, best_trial)

    save_final_images(env, out_dir)
    final_step = best_steps[-1] if best_steps else {}
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
    runtime_total_sec = float(time.perf_counter() - run_start)
    trial_runtime_vals = [float(r.get("runtime_sec", np.nan)) for r in trial_rows]
    best_trial_runtime_sec = float("nan")
    if best_trial is not None:
        for row in trial_rows:
            if int(row.get("trial_seed", -1)) == int(best_trial):
                best_trial_runtime_sec = float(row.get("runtime_sec", np.nan))
                break

    summary = {
        "method": "random",
        "run_id": run_id,
        "seed": int(args.seed),
        "detector": str(args.detector),
        "detector_model": str(args.detector_model),
        "select_by": args.select_by,
        "trials": int(args.trials),
        "best_score": float(best) if best is not None else None,
        "best_seed": int(best_trial) if best_trial is not None else None,
        "actions": best_actions or [],
        "final": final_step,
        "steps": len(best_steps) if best_steps else 0,
        "n_success": 1 if final_success else 0,
        "success_rate": 1.0 if final_success else 0.0,
        "mean_steps": float(len(best_steps)) if best_steps else float("nan"),
        "mean_base_conf": base_conf,
        "mean_after_conf": after_conf,
        "mean_drop_on": drop_on,
        "mean_area_frac": area_frac,
        "mean_drop_per_area": drop_per_area,
        "mean_iou": mean_iou,
        "mean_misclass_rate": misclass,
        "mean_selected_cells": selected_cells,
        "runtime_total_sec": runtime_total_sec,
        "mean_trial_runtime_sec": float(np.mean(trial_runtime_vals)) if trial_runtime_vals else float("nan"),
        "std_trial_runtime_sec": float(np.std(trial_runtime_vals)) if trial_runtime_vals else float("nan"),
        "best_trial_runtime_sec": best_trial_runtime_sec,
        "episodes_detail": [{
            "episode_index": 0,
            "seed": int(best_trial) if best_trial is not None else None,
            "success": bool(final_success),
            "steps": int(len(best_steps)) if best_steps else 0,
            "base_conf": base_conf,
            "after_conf": after_conf,
            "drop_on": drop_on,
            "area_frac": area_frac,
            "drop_per_area": drop_per_area,
            "mean_iou": mean_iou,
            "misclass_rate": misclass,
            "selected_cells": selected_cells,
            "runtime_sec": best_trial_runtime_sec,
            "runtime_per_step_sec": float(best_trial_runtime_sec / len(best_steps)) if best_steps and np.isfinite(best_trial_runtime_sec) else float("nan"),
        }],
        "trials_detail": trial_rows,
        "episode_meta": best_meta if best_trial is not None else None,
        "config": vars(args),
    }

    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(out_dir, "steps.json"), "w", encoding="utf-8") as f:
        json.dump(best_steps or [], f, indent=2)

    writer.close()
    if best_steps:
        final = best_steps[-1]
        metrics = final.get("metrics", {})
        area = metrics.get("total_area_mask_frac", final.get("area_frac", None))
        after_conf = metrics.get("c_on", final.get("after_conf", None))
        if after_conf is None:
            print(f"[BEST] score={best:.4f} area={float(area):.4f}" if area is not None else f"[BEST] score={best:.4f}")
        else:
            print(f"[BEST] score={best:.4f} area={float(area):.4f} c_on={float(after_conf):.4f}")
    print(f"[DONE] Saved outputs to {out_dir}")


if __name__ == "__main__":
    main()
