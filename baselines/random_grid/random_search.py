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
from baselines.grid_utils import build_env_from_args, save_final_images


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
    ap = argparse.ArgumentParser("Random grid baseline (StopSignGridEnv)")
    ap.add_argument("--data", default="./data")
    ap.add_argument("--bgdir", default="./data/backgrounds")
    ap.add_argument("--bg-mode", choices=["dataset", "solid"], default="dataset")
    ap.add_argument("--no-pole", action="store_true")

    ap.add_argument("--yolo-weights", default=None)
    ap.add_argument("--yolo-version", choices=["8", "11"], default="8")
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
    ap.add_argument("--select-by", choices=["reward", "drop_on", "reward_raw_total", "drop_on_smooth"], default="drop_on")
    ap.add_argument("--out", default="./baselines/random_grid/_runs")
    ap.add_argument("--tb", default="", help="TensorBoard log dir (default: <run_dir>/tb).")
    return ap.parse_args()


def run_random_episode(env, seed: int):
    random.seed(int(seed))
    np.random.seed(int(seed))
    env.reset(seed=int(seed))
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
        step_logs.append({
            "step": int(env._step),
            "action": int(a),
            "reward": float(reward),
            "drop_on": float(info.get("drop_on", 0.0)),
            "drop_day": float(info.get("drop_day", 0.0)),
            "area_frac": float(info.get("total_area_mask_frac", 0.0)),
        })
        done = bool(term) or bool(trunc)
    final = step_logs[-1] if step_logs else {}
    return action_seq, step_logs, final


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    run_id = time.strftime("random_%Y%m%d_%H%M%S")
    out_dir = os.path.join(args.out, run_id)
    os.makedirs(out_dir, exist_ok=True)
    tb_dir = args.tb if args.tb else os.path.join(out_dir, "tb")
    writer = SummaryWriter(log_dir=tb_dir)

    env = build_env_from_args(args)

    best = None
    best_trial = None
    best_actions = None
    best_steps = None

    for t in range(int(args.trials)):
        trial_seed = int(args.seed) + t
        actions, steps, final = run_random_episode(env, trial_seed)
        score = score_from(final, final.get("reward", 0.0), args.select_by)
        if (best is None) or (score > best):
            best = score
            best_trial = trial_seed
            best_actions = actions
            best_steps = steps
        if (t + 1) % 5 == 0 or (t + 1) == int(args.trials):
            print(f"[trial {t+1}/{args.trials}] best_score={best:.4f}")
        writer.add_scalar("metrics/trial_score", float(score), t + 1)
        writer.add_scalar("metrics/best_score", float(best), t + 1)
        writer.add_scalar("metrics/drop_on", float(final.get("drop_on", 0.0)), t + 1)
        writer.add_scalar("metrics/drop_day", float(final.get("drop_day", 0.0)), t + 1)
        writer.add_scalar("metrics/area_frac", float(final.get("area_frac", 0.0)), t + 1)

    # Replay best trial for images
    if best_trial is not None:
        run_random_episode(env, best_trial)

    save_final_images(env, out_dir)

    summary = {
        "run_id": run_id,
        "select_by": args.select_by,
        "trials": int(args.trials),
        "best_score": float(best) if best is not None else None,
        "best_seed": int(best_trial) if best_trial is not None else None,
        "actions": best_actions or [],
        "final": best_steps[-1] if best_steps else {},
        "steps": len(best_steps) if best_steps else 0,
    }

    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(out_dir, "steps.json"), "w", encoding="utf-8") as f:
        json.dump(best_steps or [], f, indent=2)

    writer.close()
    print(f"[DONE] Saved outputs to {out_dir}")


if __name__ == "__main__":
    main()
