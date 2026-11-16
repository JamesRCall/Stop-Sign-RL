#!/usr/bin/env python3
"""Run ONLY FIPatch/IMPACT; reuse existing DIRP/Greedy/Random seed results.

The parent is a wall-clock watchdog. GPU inference runs in an interruptible
child process. No training or rerunning of the existing three methods occurs.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DETECTORS = ("yolo8", "yolo11", "fasterrcnn_v2", "fcos", "retinanet_v2", "rtdetr")
METHODS = ("FIPatch", "IMPACT")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save(path, value):
    from baselines.modern.oracle import atomic_json
    atomic_json(Path(path), value)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", default="runs/modern_baselines_14h")
    p.add_argument("--compare-root", default="runs/paper_data/compare")
    p.add_argument("--data", default="data")
    p.add_argument("--backgrounds", default="data/backgrounds")
    p.add_argument("--weights-dir", default="weights")
    p.add_argument("--detectors", default="all")
    p.add_argument("--seeds", default="1000,1001,1002,1003,1004")
    p.add_argument("--methods", default="FIPatch,IMPACT")
    p.add_argument("--hours", type=float, default=13.5,
                   help="Runtime including calibration/model loads; maximum 13.5, reserving time for export")
    p.add_argument("--queries", type=int, default=12006,
                   help="Detector input images per search including references; auto policy may lower the shared ceiling")
    p.add_argument("--budget-policy", choices=["auto", "fixed"], default="auto",
                   help="fixed honors the requested query ceiling; the wall deadline can still stop unfinished runs")
    p.add_argument("--population", type=int, default=16)
    p.add_argument("--circles", type=int, default=3)
    p.add_argument("--swarm", type=int, default=16)
    p.add_argument("--color-mode", choices=["rgb", "material"], default="rgb",
                   help="rgb uses material warm starts then free RGB; material fixes the original UV-paint color")
    p.add_argument("--search-areas", default="0.25,0.5",
                   help="Predefined nominal sign-area levels for support search; original soft penalty/metrics are unchanged")
    p.add_argument("--search-objective", choices=["dirp_score", "suppression"], default="dirp_score",
                   help="Original area-penalized reward, or confidence-only diagnostic search; success gates stay unchanged")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--report-only", action="store_true")
    p.add_argument("--prepare-only", action="store_true")
    p.add_argument("--worker", choices=["calibrate", "attack"], help=argparse.SUPPRESS)
    p.add_argument("--detector", help=argparse.SUPPRESS)
    p.add_argument("--deadline", type=float, help=argparse.SUPPRESS)
    args = p.parse_args()
    args.output = str(Path(args.output).resolve())
    args.compare_root = str(Path(args.compare_root).resolve())
    args.data = str(Path(args.data).resolve())
    args.backgrounds = str(Path(args.backgrounds).resolve())
    args.weights_dir = str(Path(args.weights_dir).resolve())
    args.detector_names = list(DETECTORS) if args.detectors == "all" else args.detectors.split(",")
    args.seed_values = list(map(int, args.seeds.split(",")))
    args.method_names = args.methods.split(",")
    try:
        args.search_area_levels = list(map(float, args.search_areas.split(",")))
    except ValueError:
        p.error("search-areas must be comma-separated fractions")
    if (not args.search_area_levels or len(args.search_area_levels) > 4
            or len(set(args.search_area_levels)) != len(args.search_area_levels)
            or any(not 0 < x <= 1 for x in args.search_area_levels)):
        p.error("Provide 1-4 distinct search-area fractions in (0,1]")
    if not args.detector_names or len(set(args.detector_names)) != len(args.detector_names) or any(n not in DETECTORS for n in args.detector_names):
        p.error("Unknown/duplicate detectors")
    if not args.method_names or len(set(args.method_names)) != len(args.method_names) or any(n not in METHODS for n in args.method_names):
        p.error("Only FIPatch and IMPACT can be run; existing methods are never rerun")
    if not args.seed_values or len(set(args.seed_values)) != len(args.seed_values) or min(args.seed_values) < 0:
        p.error("Provide distinct nonnegative seeds")
    if not 0 < args.hours <= 13.5:
        p.error("--hours must be >0 and <=13.5 to leave export/shutdown time below 14 hours")
    if args.queries < 102 or args.population < 4 or args.circles < 1 or args.swarm < 4:
        p.error("queries >=102, population/swarm >=4, circles >=1 required")
    return args


def comparison(args, detector):
    return read(Path(args.compare_root) / f"{detector}_N5_seed1000/compare_summary.json")


def environment_config(args, detector):
    from baselines.grid_utils import _default_cfg_for_env
    config = _default_cfg_for_env(comparison(args, detector)["ppo"]["config"])
    config.update(data=args.data, bgdir=args.backgrounds, detector_device=args.device,
                  seed=args.seed_values[0], info_image_every=0, search_objective=args.search_objective,
                  color_mode=getattr(args, "color_mode", "rgb"),
                  search_area_levels=getattr(args, "search_area_levels", [.25, .5]))
    if config["detector"] == "yolo":
        name = "yolo11n.pt" if detector == "yolo11" else "yolov8n.pt"
        config["yolo_weights"] = str(Path(args.weights_dir) / name)
    # Frozen policies are not loaded: their old measured results are reused.
    config.pop("model", None)
    config.pop("vecnorm", None)
    return config


def make_env(args, detector):
    from types import SimpleNamespace
    from baselines.grid_utils import build_env_from_args
    from baselines.modern.oracle import RecordingDetector
    config = environment_config(args, detector)
    env = build_env_from_args(SimpleNamespace(**config))
    recorder = RecordingDetector(env.det)
    env.det = recorder
    return env, recorder, config


def preflight(args):
    from importlib import metadata
    from baselines.modern.upstream import SOURCES
    sources = {}
    for detector in args.detector_names:
        path = Path(args.compare_root) / f"{detector}_N5_seed1000/compare_summary.json"
        value = read(path)
        for method in ["ppo", "greedy", "random"]:
            seeds = {int(r["seed"]) for r in value["episodes"][method]}
            if not set(args.seed_values).issubset(seeds):
                raise ValueError(f"{path}: {method} has no saved result for requested seeds")
        sources[str(path)] = sha(path)
        for row in value["episodes"]["random"]:
            if row["seed"] not in args.seed_values:
                continue
            trial_file = path.parent / "random" / Path(row["run_dir"]).name / "summary.json"
            data = read(trial_file)
            if len(data["trials_detail"]) != 50:
                raise ValueError(f"Expected all 50 stored Random trials: {trial_file}")
            sources[str(trial_file)] = sha(trial_file)
    for name in ["stop_sign.png", "stop_sign_uv.png"]:
        path = Path(args.data) / name
        sources[str(path)] = sha(path)
    if (Path(args.data)/"pole.png").exists():
        sources[str(Path(args.data)/"pole.png")] = sha(Path(args.data)/"pole.png")
    backgrounds = [p for p in Path(args.backgrounds).iterdir() if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"]]
    if not backgrounds:
        raise ValueError("No benchmark backgrounds; refusing a synthetic-background substitution")
    for path in sorted(backgrounds):
        sources[str(path)] = sha(path)
    for name in args.detector_names:
        if name in ["yolo8", "yolo11"]:
            path = Path(args.weights_dir) / ("yolo11n.pt" if name == "yolo11" else "yolov8n.pt")
            sources[str(path)] = sha(path)
    code_files = [Path(__file__)]
    for directory in ["baselines", "detectors", "envs", "utils"]:
        code_files.extend((ROOT/directory).rglob("*.py"))
    for path in code_files:
        sources[str(path)] = sha(path)
    versions = {}
    for package in ["torch", "torchvision", "ultralytics", "transformers", "numpy", "Pillow", "scipy", "scikit-learn", "pyswarm"]:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = "not_installed"
    return dict(detectors=args.detector_names, seeds=args.seed_values, methods=args.method_names,
                requested_queries=args.queries, population=args.population, circles=args.circles,
                adapter_revision=4, search_objective=args.search_objective,
                impact_cell_rule="actual cropped-cell coverage, identical to DIRP",
                color_mode=args.color_mode, swarm=args.swarm, budget_policy=args.budget_policy,
                search_area_levels=args.search_area_levels,
                device=args.device, input_sha256=sources, upstream=SOURCES, package_versions=versions,
                python_version=sys.version,
                adaptation="Author PSO/DE kernels; material warm starts, sign support repair, variable-radius multicircle PSO, tile-space ES; DIRP UV renderer",
                timing_note="Reused runtime measurements and new GPU measurements are not hardware-normalized speedups.")


def child_command(args, mode, deadline, detector=None):
    command = [sys.executable, "-u", str(Path(__file__).resolve()), "--worker", mode,
               "--output", args.output, "--compare-root", args.compare_root, "--data", args.data,
               "--backgrounds", args.backgrounds, "--weights-dir", args.weights_dir,
               "--detectors", args.detectors, "--seeds", args.seeds, "--methods", args.methods,
               "--hours", str(args.hours), "--queries", str(args.queries),
               "--population", str(args.population), "--circles", str(args.circles),
               "--search-objective", args.search_objective,
               "--color-mode", args.color_mode, "--swarm", str(args.swarm),
               "--search-areas", args.search_areas,
               "--budget-policy", args.budget_policy,
               "--device", args.device, "--deadline", str(deadline)]
    if detector:
        command += ["--detector", detector]
    return command


def supervised(command, deadline):
    process = subprocess.Popen(command, cwd=ROOT, start_new_session=(os.name != "nt"))
    try:
        while process.poll() is None:
            if time.monotonic() >= deadline:
                stop_child(process)
                return 124
            time.sleep(0.5)
        return process.returncode
    except BaseException:
        stop_child(process)
        raise


def stop_child(process):
    if process.poll() is not None:
        return
    if os.name != "nt":
        os.killpg(process.pid, signal.SIGTERM)
    else:
        process.terminate()
    try:
        process.wait(timeout=8)
    except subprocess.TimeoutExpired:
        if os.name != "nt":
            os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()
        process.wait(timeout=5)


def worker(args):
    import gc
    import importlib.metadata
    from types import SimpleNamespace
    import numpy as np
    import torch
    from baselines.modern.oracle import AttackOracle, SearchStopped
    from baselines.modern.upstream import load_kernels
    from baselines.modern.attacks import fipatch, impact

    torch.set_num_threads(min(4, os.cpu_count() or 1))
    if args.device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; use the Blackwell setup script")
        # Execute a real kernel: availability alone can pass on an unsupported GPU.
        (torch.ones(32, device=args.device) @ torch.ones(32, device=args.device)).item()
    source_dir = Path(args.output) / "upstream"
    fipatch_kernel, impact_kernel = load_kernels(source_dir)
    if args.worker == "calibrate":
        timings = {}
        for detector in args.detector_names:
            if time.monotonic() >= args.deadline:
                break
            print(f"[calibrate] loading {detector}", flush=True)
            env, recorder, config = make_env(args, detector)
            try:
                oracle = AttackOracle(env, recorder, seed=args.seed_values[0], config=config,
                                      candidate_limit=4, deadline=args.deadline,
                                      folder=Path(args.output)/"calibration"/detector)
                mask = np.zeros(oracle.shape, bool)
                mask[oracle.shape[0]//3:2*oracle.shape[0]//3, oracle.shape[1]//3:2*oracle.shape[1]//3] = True
                rgb = np.full((*oracle.shape, 3), 128, dtype=np.float32)
                oracle.evaluate(mask, rgb)  # Warm-up, logged separately from benchmark.
                started = time.monotonic()
                for _ in range(3):
                    oracle.evaluate(mask, rgb)
                timings[detector] = (time.monotonic()-started)/3
                oracle.close()
                save(Path(args.output)/"calibration_times.json", timings)
                print(f"[calibrate] {detector}: {timings[detector]:.3f}s per 2*K-image candidate", flush=True)
            finally:
                env.close()
                del env, recorder
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        return

    plan = read(Path(args.output)/"plan.json")
    env, recorder, config = make_env(args, args.detector)
    versions = {p: importlib.metadata.version(p) for p in ["torch", "torchvision", "ultralytics", "transformers", "numpy", "Pillow", "scikit-learn", "pyswarm"]}
    gpu = torch.cuda.get_device_name(torch.device(args.device)) if args.device.startswith("cuda") else "CPU"
    save(Path(args.output)/"environment"/(args.detector+".json"), dict(gpu=gpu, versions=versions, config=config))
    jobs = [(seed, method) for i, seed in enumerate(args.seed_values)
            for method in (args.method_names if i%2==0 else list(reversed(args.method_names)))]
    for position, (seed, method) in enumerate(jobs):
        if time.monotonic() >= args.deadline:
            break
        folder = Path(args.output)/args.detector/str(seed)/method
        result_file = folder/"result.json"
        if result_file.exists() and read(result_file).get("status") in ["completed", "query_limit"]:
            print(f"[resume] {args.detector} seed={seed} {method} already complete", flush=True)
            continue
        attempt_root = folder/"attempts"
        attempt_root.mkdir(parents=True, exist_ok=True)
        attempt = attempt_root/f"{len(list(attempt_root.iterdir()))+1:03d}"
        # Each run receives the same query ceiling and a conservative time guard.
        remaining_jobs = sum(
            not ((Path(args.output)/args.detector/str(s)/m/"result.json").exists()
                 and read(Path(args.output)/args.detector/str(s)/m/"result.json").get("status")
                 in ["completed", "query_limit"])
            for s, m in jobs[position:])
        trial_deadline = min(args.deadline, time.monotonic()+
            max(1.0, (args.deadline-time.monotonic())/max(1, remaining_jobs)))
        last_progress = [0.0]

        def progress(oracle):
            now = time.monotonic()
            if now-last_progress[0] < 10:
                return
            last_progress[0] = now
            rate = oracle.evaluations/max(now-oracle.started, 1e-9)
            eta = (oracle.candidate_limit-oracle.evaluations)/max(rate, 1e-9)
            state = dict(detector=args.detector, method=method, seed=seed,
                         candidates=oracle.evaluations, candidate_limit=oracle.candidate_limit,
                         detector_images=oracle.queries, trial_eta_seconds=eta,
                         search_stage=oracle.search_stage,
                         lowest_confidence=oracle.min_confidence,
                         remaining_wall_seconds=max(0.0, args.deadline-now))
            save(Path(args.output)/"progress.json", state)
            print(f"[{args.detector}/{seed}/{method}] {oracle.evaluations}/{oracle.candidate_limit} candidates; "
                  f"{oracle.queries} detector images; trial ETA {eta/60:.1f}m; "
                  f"lowest confidence {oracle.min_confidence:.4f}; "
                  f"detector time allowance {(args.deadline-now)/3600:.2f}h", flush=True)

        oracle = None
        reference_delta = None
        started = time.monotonic()
        status, error = "completed", None
        try:
            oracle = AttackOracle(env, recorder, seed=seed, config=config,
                                  candidate_limit=plan["candidate_limit"], deadline=trial_deadline,
                                  folder=attempt, progress=progress)
            saved = next(r for r in comparison(args, args.detector)["episodes"]["ppo"] if r["seed"]==seed)
            delta = oracle.c0_day-float(saved["base_conf"])
            reference_delta = delta
            save(attempt/"reference_check.json", dict(seed=seed, current_c0_day=oracle.c0_day,
                 saved_c0_day=saved["base_conf"], difference=delta,
                 note="Same seeds/settings do not prove identical assets or old software; retain this audit."))
            print(f"[start] {args.detector}/{seed}/{method}; reference difference={delta:+.6f}", flush=True)
            if method == "FIPatch":
                fipatch(oracle, fipatch_kernel, seed, circles=args.circles, swarm=args.swarm)
            else:
                impact(oracle, impact_kernel, seed, population=args.population)
        except SearchStopped as exc:
            status = exc.reason
        except Exception as exc:
            status, error = "error", str(exc)
            traceback.print_exc()
        finally:
            result = dict(detector=args.detector, seed=seed, method=method, status=status,
                          runtime_sec=time.monotonic()-started, attempt=str(attempt.relative_to(Path(args.output))),
                          candidate_limit=plan["candidate_limit"],
                          evaluated_candidates=oracle.evaluations if oracle else 0,
                          detector_image_queries=oracle.queries if oracle else 0,
                          best=oracle.best if oracle else None, error=error,
                          reference_confidence_difference=reference_delta,
                          adapter_revision=4, search_objective=args.search_objective,
                          color_mode=args.color_mode,
                          minimum_evaluated_confidence=oracle.min_confidence if oracle else None,
                          zero_area_candidates=oracle.zero_area_candidates if oracle else None,
                          gpu=gpu, package_versions=versions,
                          fidelity="author-kernel detector/UV adaptation; not original published protocol")
            save(result_file, result)
            if oracle is not None:
                oracle.close()
            else:
                recorder.records.clear()
            print(f"[done] {args.detector}/{seed}/{method}: {status}", flush=True)
    env.close()


def export_report(args):
    """Keep the five method names; preserve data origin and missing counts in files."""
    output = Path(args.output)
    rows = []
    for detector in args.detector_names:
        original = comparison(args, detector)
        source_folder = Path(args.compare_root)/f"{detector}_N5_seed1000"
        for method, key in [("DIRP", "ppo"), ("Greedy", "greedy")]:
            for row in original["episodes"][key]:
                if row["seed"] not in args.seed_values:
                    continue
                rows.append(dict(detector=detector, method=method, seed=row["seed"], status="completed",
                    source="saved_results", success=float(bool(row["success"])),
                    after_conf=row["after_conf"], area_frac=row["area_frac"], runtime_sec=row["runtime_sec"],
                    detector_image_queries="", clean_eligible="", n_records=1))
        for run in original["episodes"]["random"]:
            if run["seed"] not in args.seed_values:
                continue
            trials = read(source_folder/"random"/Path(run["run_dir"]).name/"summary.json")["trials_detail"]
            rows.append(dict(detector=detector, method="Random", seed=run["seed"], status="completed",
                source="saved_results", success=statistics.mean(float(t["success"]) for t in trials),
                after_conf=statistics.mean(t["after_conf"] for t in trials),
                area_frac=statistics.mean(t["area_frac"] for t in trials),
                runtime_sec=statistics.mean(t["runtime_sec"] for t in trials),
                detector_image_queries="", clean_eligible="", n_records=len(trials)))
        for method in args.method_names:
            for seed in args.seed_values:
                path = output/detector/str(seed)/method/"result.json"
                result = read(path) if path.exists() else {}
                best = result.get("best") or {}
                rows.append(dict(detector=detector, method=method, seed=seed,
                    status=result.get("status", "not_completed"), source="new_run",
                    success=float(best["success"]) if best else "", after_conf=best.get("after_conf", ""),
                    area_frac=best.get("area_frac", ""), runtime_sec=result.get("runtime_sec", ""),
                    detector_image_queries=result.get("detector_image_queries", ""),
                    clean_eligible=best.get("clean_eligible", ""), n_records=1))
                rows[-1]["reference_confidence_difference"] = result.get("reference_confidence_difference", "")
                rows[-1]["joint_success"] = best.get("joint_success", "")
                rows[-1]["drop_on"] = best.get("drop_on", "")
                rows[-1]["zero_area_candidates"] = result.get("zero_area_candidates", "")
                rows[-1]["minimum_evaluated_confidence"] = result.get("minimum_evaluated_confidence", "")
    for row in rows:
        row.setdefault("reference_confidence_difference", "")
        for key in ["joint_success", "drop_on", "zero_area_candidates", "minimum_evaluated_confidence"]:
            row.setdefault(key, "")
    output.mkdir(parents=True, exist_ok=True)
    with (output/"per_seed.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    summaries = []
    for detector in args.detector_names:
        for method in ["DIRP", "Greedy", "Random"]+args.method_names:
            group = [r for r in rows if r["detector"]==detector and r["method"]==method]
            completed = [r for r in group if r["status"] in ["completed", "query_limit"] and r["success"]!=""]
            mean = lambda key: statistics.mean(float(r[key]) for r in completed) if completed else ""
            available_mean = lambda key: (statistics.mean(float(r[key]) for r in completed)
                if completed and all(r[key] not in [None, ""] for r in completed) else "")
            summaries.append(dict(detector=detector, method=method, completed_seeds=len(completed),
                planned_seeds=len(group), incomplete_seeds=len(group)-len(completed),
                ASR=mean("success"), mean_after_conf=mean("after_conf"), mean_area=mean("area_frac"),
                joint_ASR=available_mean("joint_success"), mean_drop_on=available_mean("drop_on"),
                mean_minimum_evaluated_confidence=available_mean("minimum_evaluated_confidence"),
                mean_zero_area_candidates=available_mean("zero_area_candidates"),
                mean_minutes=mean("runtime_sec")/60 if completed else "",
                timing_unit="per Random candidate" if method=="Random" else "per search/episode",
                record_count=sum(r["n_records"] for r in completed),
                clean_ineligible_runs=sum(r["clean_eligible"] is False for r in completed),
                provenance="new_run" if method in METHODS else "saved_results"))
    with (output/"comparison.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summaries[0]))
        writer.writeheader(); writer.writerows(summaries)
    lines = ["# DIRP baseline comparison", "", "| Detector | Method | Completed/planned | ASR | Activated confidence | Area | Joint ASR | Minutes |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    for r in summaries:
        fmt = lambda k: f"{r[k]:.3f}" if r[k]!="" else "--"
        lines.append(f"| {r['detector']} | {r['method']} | {r['completed_seeds']}/{r['planned_seeds']} | "
                     f"{fmt('ASR')} | {fmt('mean_after_conf')} | {fmt('mean_area')} | {fmt('joint_ASR')} | {fmt('mean_minutes')} |")
    lines += ["", "FIPatch/IMPACT use author optimizer kernels adapted to detector scores and the DIRP UV renderer; active RGB is optimized.",
              "DIRP/Greedy/Random values are read from the saved matching-seed experiment files. Their timing measurements use the original hardware and exclude PPO training.",
              "Random timing is per recorded candidate, not a whole best-of-50 search. No cross-hardware speedup is implied.",
              "Incomplete or errored new runs are missing data, not attack failures; ASR averages completed runs only. Check the completed/planned column.",
              "The shared ceiling applies to the two new methods, not retroactively to the saved three methods.",
              "ASR is clean-eligible activated confidence <=0.20. New-method Joint ASR additionally checks the original 0.30 area and daylight gates; unavailable saved joint values are --.",
              "Reference confidence deltas and task hashes are saved per run; same seed alone does not establish exact old-scene reproduction."]
    manifest_path = output/"manifest.json"
    if manifest_path.exists():
        contract = read(manifest_path).get("contract", {})
        lines += ["", f"Adapter revision: {contract.get('adapter_revision', 1)}. "
                  f"Search objective: {contract.get('search_objective', 'dirp_score')}. "
                  f"Color mode: {contract.get('color_mode', 'rgb')}. "
                  f"Nominal support levels: {contract.get('search_area_levels', 'not recorded')}. "
                  "Search-objective changes do not change the success thresholds. These settings must be disclosed when comparing protocols."]
    drift = [r for r in rows if r["source"]=="new_run" and r["reference_confidence_difference"] not in [None, ""]
             and abs(r["reference_confidence_difference"]) > 0.001]
    if drift:
        lines += ["", f"Reference check: {len(drift)} new runs differ from the stored clean confidence by more than 0.001. "
                  "These runs are seed/config matched but are not verified reproductions of the exact original scenes. "
                  "Resolve data/order/software differences before claiming an exact paired comparison."]
    (output/"comparison.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    save(output/"comparison.json", summaries)
    order = ["DIRP", "Greedy", "Random"]+args.method_names
    protocol = read(manifest_path).get("contract", {}) if manifest_path.exists() else {}
    protocol_note = (f" New-method adapter revision {protocol.get('adapter_revision', 1)}, "
        f"color mode {protocol.get('color_mode', 'rgb')}, "
        f"search objective {protocol.get('search_objective', 'dirp_score')}.").replace("_", r"\_")
    latex = [r"\begin{table*}[t]", r"\centering", r"\scriptsize",
             r"\caption{Baseline comparison on the configured original seeds. FIPatch and IMPACT are detector/UV adaptations using author optimizer kernels. DIRP, Greedy and Random use the saved seed-matched results. Time is minutes per search/episode, except Random per candidate; PPO training is excluded and timings are not hardware-normalized. ASR uses the confidence threshold only; area and daylight gates are reported separately. Incomplete method/seed groups are shown as --."+protocol_note+"}",
             r"\label{tab:modern_baselines}", r"\resizebox{\textwidth}{!}{%",
             r"\begin{tabular}{l"+"ccc"*len(order)+"}", r"\toprule",
             "Detector & "+" & ".join(r"\multicolumn{3}{c}{"+method+"}" for method in order)+r" \\",
             " & "+" & ".join(["ASR & Area & Time"]*len(order))+r" \\", r"\midrule"]
    for detector in args.detector_names:
        cells = [detector.replace("_", r"\_")]
        for method in order:
            row = next(r for r in summaries if r["detector"]==detector and r["method"]==method)
            cells += [f"{row[k]:.3f}" if row["completed_seeds"]==row["planned_seeds"] and row[k]!="" else "--"
                      for k in ["ASR", "mean_area", "mean_minutes"]]
        latex.append(" & ".join(cells)+r" \\")
    latex += [r"\bottomrule", r"\end{tabular}}", r"\end{table*}"]
    (output/"comparison.tex").write_text("\n".join(latex)+"\n", encoding="utf-8")
    print(f"Report: {output/'comparison.md'}", flush=True)


def main():
    args = parse_args()
    os.chdir(ROOT)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(Path(args.output)/".ultralytics"))
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")
    if args.worker:
        Path(os.environ["YOLO_CONFIG_DIR"]).mkdir(parents=True, exist_ok=True)
        worker(args)
        return 0
    if args.report_only:
        export_report(args)
        return 0
    from baselines.modern.upstream import fetch_sources
    started = time.monotonic()
    deadline = started + args.hours*3600
    output = Path(args.output)
    if output.exists() and any(output.iterdir()) and not args.resume:
        raise ValueError("Output already exists; use --resume or a new --output")
    output.mkdir(parents=True, exist_ok=True)
    Path(os.environ["YOLO_CONFIG_DIR"]).mkdir(parents=True, exist_ok=True)
    contract = preflight(args)
    fingerprint = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
    manifest = output/"manifest.json"
    if manifest.exists() and read(manifest)["fingerprint"] != fingerprint:
        raise ValueError("Resume contract differs: inputs, methods, seeds, code or settings changed")
    save(manifest, dict(fingerprint=fingerprint, contract=contract))
    print("Running ONLY "+" and ".join(args.method_names)+
          "; DIRP/Greedy/Random are read from saved files.", flush=True)
    fetch_sources(output/"upstream")
    if args.prepare_only:
        print("Inputs and pinned upstream code verified. No GPU work started.")
        return 0
    plan_path = output/"plan.json"
    if not plan_path.exists():
        code = supervised(child_command(args, "calibrate", deadline), deadline)
        timings_path = output/"calibration_times.json"
        if code or not timings_path.exists() or set(read(timings_path)) != set(args.detector_names):
            export_report(args)
            raise RuntimeError("Calibration failed or timed out; see console; no partial calibration is treated as a complete benchmark")
        timings = read(timings_path)
        max_k = max(int(comparison(args, n)["ppo"]["config"]["eval_K"]) for n in args.detector_names)
        if len({int(comparison(args, n)["ppo"]["config"]["eval_K"]) for n in args.detector_names}) != 1:
            raise ValueError("A shared EOT K is required for the global query ceiling")
        jobs = len(args.detector_names)*len(args.seed_values)*len(args.method_names)
        requested = args.queries//(2*max_k)-1
        # Factor 2 allows for native clustering/DE, disk flushes, and variation
        # beyond measured inference. Allocate only 70% of remaining wall time.
        affordable = int(0.7*max(0, deadline-time.monotonic())/(jobs*max(timings.values())*2))
        count = requested if args.budget_policy == "fixed" else min(requested, affordable)
        if count < 64:
            export_report(args)
            raise RuntimeError("Too little time for even 64 candidates/run; choose fewer detectors/seeds or increase --hours")
        plan = dict(candidate_limit=count, detector_image_query_limit=2*max_k*(count+1),
                    budget_policy=args.budget_policy, adapter_revision=4,
                    trial_time_policy="remaining detector wall budget divided among unfinished searches",
                    calibration_seconds_per_candidate=timings, safety_factor=2.0, planned_new_runs=jobs)
        save(plan_path, plan)
    plan = read(plan_path)
    print(f"Shared new-run budget: {plan['candidate_limit']} candidates / "
          f"{plan['detector_image_query_limit']} detector input images, including references. "
          f"Hard process deadline in {(deadline-time.monotonic())/3600:.2f}h.", flush=True)
    interrupted = False
    try:
        for i, detector in enumerate(args.detector_names):
            if time.monotonic() >= deadline:
                break
            remaining_detectors = len(args.detector_names)-i
            detector_deadline = min(deadline, time.monotonic()+(deadline-time.monotonic())/remaining_detectors)
            code = supervised(child_command(args, "attack", detector_deadline, detector), detector_deadline)
            if code:
                print(f"{detector}: child exited {code}; incomplete runs will be marked missing", flush=True)
            export_report(args)
    except KeyboardInterrupt:
        interrupted = True
    finally:
        export_report(args)
        complete = all((Path(args.output)/detector/str(seed)/method/"result.json").exists()
                       and read(Path(args.output)/detector/str(seed)/method/"result.json").get("status") in ["completed", "query_limit"]
                       and read(Path(args.output)/detector/str(seed)/method/"result.json").get("best") is not None
                       for detector in args.detector_names for seed in args.seed_values for method in args.method_names)
        save(output/"run_status.json", dict(elapsed_seconds=time.monotonic()-started,
             time_limit_hours=args.hours, interrupted=interrupted,
             deadline_reached=time.monotonic()>=deadline, complete=complete))
    return 130 if interrupted else 0 if complete else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
