#!/usr/bin/env python3
"""Renderer checks and a fixed two-objective diagnostic, never paper updates.

Runs both methods on one predefined detector/seed under both objectives.
Reports all outcomes, including failures. No DIRP/Greedy/Random search is run.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import run_rapid_modern_baselines as runner


def check_renderer(args, output):
    import numpy as np
    from baselines.modern.oracle import AttackOracle

    env_args = SimpleNamespace(compare_root=args.compare_root, data=args.data,
        backgrounds=args.backgrounds, weights_dir=args.weights_dir, device=args.device,
        seed_values=[args.seed], search_objective="dirp_score", color_mode=args.color_mode)
    env, recorder, config = runner.make_env(env_args, args.detector)
    oracle = None
    try:
        oracle = AttackOracle(env, recorder, seed=args.seed, config=config,
            candidate_limit=4, deadline=time.monotonic()+600, folder=output/"controls")
        # Check pixel equivalence, independently of detector scores.
        empty_differences = {}
        for kind in ("day", "on"):
            clean = env._render_variant(kind, False, env._transform_seeds[0])
            empty = env._render_variant(kind, True, env._transform_seeds[0])
            clean.save(output/"controls"/f"clean_{kind}.png")
            empty_differences[kind] = int(np.abs(np.asarray(clean).astype(int)-np.asarray(empty).astype(int)).max())
            clean.close(); empty.close()
        assert not any(empty_differences.values()), "Empty overlay changes rendered image"
        zero = np.zeros(oracle.shape, dtype=bool)
        rgb = np.full((*oracle.shape, 3), env.paint.active_rgb, dtype=np.uint8)
        controls = {"empty": oracle.evaluate(zero, rgb)}
        # Match a legal grid pattern rendered by the original environment.
        coordinates = np.argwhere(env._valid_cells)
        for r, c in coordinates[len(coordinates)//3:len(coordinates)//3+12]:
            env._episode_cells[r, c] = True
        mask = np.asarray(env._grid_layer_mask(env.sign_rgba_day.size,
            np.argwhere(env._episode_cells), 1.0)) > 0
        oracle.current_mask, oracle.current_rgb = mask, rgb
        grid_differences = {}
        for kind, source in [("day", env.sign_rgba_day), ("on", env.sign_rgba_on)]:
            legacy = oracle.original_overlay(source, kind)
            adapted = env._apply_grid_overlay(source, kind)
            grid_differences[kind] = int(np.abs(np.asarray(legacy).astype(int)-np.asarray(adapted).astype(int)).max())
        assert not any(grid_differences.values()), "Adapter disagrees with original grid renderer"
        controls["grid_yellow"] = oracle.evaluate(mask, rgb)
        for name, color in [("full_sign_black", [0, 0, 0]), ("full_sign_yellow", env.paint.active_rgb)]:
            controls[name] = oracle.evaluate(oracle.sign_mask,
                np.full((*oracle.shape, 3), color, dtype=np.uint8))
            for kind in ("day", "on"):
                image = env._render_variant(kind, True, env._transform_seeds[0])
                image.save(output/"controls"/f"{name}_{kind}.png"); image.close()
        report = dict(empty_pixel_max_difference=empty_differences,
            legacy_grid_pixel_max_difference=grid_differences, controls=controls,
            note="Full-sign overlays are renderer probes, not benchmark attacks or paper results.")
        runner.save(output/"renderer_checks.json", report)
        print("Renderer checks passed: empty overlay and original grid overlay match exactly.", flush=True)
    finally:
        if oracle is not None:
            oracle.close()
        env.close()


def summarize(args, output):
    rows = []
    for objective in ("dirp_score", "suppression"):
        for method in ("FIPatch", "IMPACT"):
            folder = output/objective/args.detector/str(args.seed)/method
            path = folder/"result.json"
            if not path.exists():
                rows.append(dict(objective=objective, method=method, status="missing")); continue
            result = runner.read(path)
            trace = [json.loads(line) for line in (output/objective/result["attempt"]/"trace.jsonl").read_text().splitlines()]
            best = result.get("best") or {}
            rows.append(dict(objective=objective, method=method, status=result["status"],
                candidates=len(trace), clean_confidence=best.get("base_uv_conf"),
                selected_confidence=best.get("after_conf"),
                minimum_evaluated_confidence=min((t["after_conf"] for t in trace), default=None),
                area=best.get("area_frac"), success=best.get("success"),
                joint_success=best.get("joint_success"),
                zero_area_candidates=sum(t["area_frac"]==0 for t in trace),
                reference_confidence_difference=result.get("reference_confidence_difference")))
    runner.save(output/"pilot_summary.json", rows)
    lines = ["# Modern baseline diagnostic pilot", "",
        "Both objectives and all outcomes are reported. These are adaptation diagnostics, not final paper evidence.", "",
        "| Objective | Method | Status | Candidates | Clean UV | Selected UV | Minimum UV | Area | Success | Empty candidates |",
        "|---|---|---|---:|---:|---:|---:|---:|---|---:|"]
    for r in rows:
        fmt = lambda key: f"{r[key]:.4f}" if isinstance(r.get(key), (int, float)) else "--"
        lines.append(f"| {r['objective']} | {r['method']} | {r['status']} | {r.get('candidates', 0)} | "
            f"{fmt('clean_confidence')} | {fmt('selected_confidence')} | {fmt('minimum_evaluated_confidence')} | "
            f"{fmt('area')} | {r.get('success', '--')} | {r.get('zero_area_candidates', '--')} |")
    lines += ["", "Success still requires clean eligibility and activated confidence <= 0.20. Joint success additionally checks the original area/daylight gates.",
        "Confidence-only optimization removes area/daylight penalties from the search objective only. It is a different protocol and is not pooled with the original objective.",
        "Clean-reference differences remain recorded; equal seed IDs do not establish identical scenes to the saved three methods."]
    (output/"pilot_summary.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print("\n".join(lines), flush=True)
    return all(r["status"] in ("completed", "query_limit") for r in rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", default="runs/modern_pilot_v3")
    p.add_argument("--detector", choices=runner.DETECTORS, default="yolo8")
    p.add_argument("--seed", type=int, default=1000)
    p.add_argument("--queries", type=int, default=3078, help="512 candidates per search at K=3; shared by all four searches")
    p.add_argument("--hours-per-objective", type=float, default=1.0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--color-mode", choices=["rgb", "material"], default="rgb")
    p.add_argument("--compare-root", default="runs/paper_data/compare")
    p.add_argument("--data", default="data")
    p.add_argument("--backgrounds", default="data/backgrounds")
    p.add_argument("--weights-dir", default="weights")
    args = p.parse_args()
    if args.queries < 390 or not 0 < args.hours_per_objective <= 6:
        p.error("queries >=390 and hours-per-objective in (0,6] required")
    output = Path(args.output).resolve()
    if output.exists():
        p.error("Choose a new pilot output folder; previous results are never overwritten")
    output.mkdir(parents=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(output/".ultralytics"))
    runner.save(output/"pilot_plan.json", dict(vars(args), objectives=["dirp_score", "suppression"],
        methods=list(runner.METHODS), purpose="fixed diagnostic; not selected for attack success"))
    check_renderer(args, output)
    codes = []
    for objective in ("dirp_score", "suppression"):
        command = [sys.executable, "-u", str(ROOT/"tools/run_rapid_modern_baselines.py"),
            "--detectors", args.detector, "--seeds", str(args.seed),
            "--queries", str(args.queries), "--hours", str(args.hours_per_objective),
            "--search-objective", objective, "--color-mode", args.color_mode,
            "--budget-policy", "fixed", "--output", str(output/objective),
            "--device", args.device, "--compare-root", args.compare_root,
            "--data", args.data, "--backgrounds", args.backgrounds, "--weights-dir", args.weights_dir]
        codes.append(subprocess.run(command, cwd=ROOT).returncode)
    complete = summarize(args, output)
    return 0 if complete and not any(codes) else 2


if __name__ == "__main__":
    raise SystemExit(main())
