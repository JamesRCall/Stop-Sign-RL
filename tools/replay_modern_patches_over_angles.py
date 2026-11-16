#!/usr/bin/env python3
"""Replay saved FIPatch/IMPACT NPZ patches; never optimize or retrain."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def export(output, rows, seeds):
    if not rows:
        return
    with (output/"angle_rows.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    groups = sorted({(r["detector"], r["method"], r["angle_deg"]) for r in rows})
    summaries = []
    for detector, method, angle in groups:
        group = [r for r in rows if (r["detector"], r["method"], r["angle_deg"]) == (detector, method, angle)]
        mean = lambda key: statistics.mean(float(r[key]) for r in group)
        summaries.append(dict(detector=detector, method=method, angle_deg=angle,
            completed_seeds=len(group), planned_seeds=len(seeds), ASR=mean("success"),
            joint_ASR=mean("joint_success"), clean_eligible_fraction=mean("clean_eligible"),
            mean_clean_day=mean("base_conf"), mean_clean_uv=mean("base_uv_conf"),
            mean_day=mean("patched_day_conf"), mean_uv=mean("after_conf"),
            mean_area=mean("area_frac"), mean_seconds=mean("replay_seconds")))
    with (output/"angle_summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
        writer.writeheader(); writer.writerows(summaries)
    (output/"angle_summary.json").write_text(json.dumps(summaries, indent=2)+"\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="runs/modern_baselines_v3_rgb")
    parser.add_argument("--output", default="runs/modern_angles_yolo_v3")
    parser.add_argument("--detectors", default="yolo8,yolo11")
    parser.add_argument("--methods", default="FIPatch,IMPACT")
    parser.add_argument("--seeds", default="1000,1001,1002,1003,1004")
    parser.add_argument("--angles", default="-24,-18,-12,-6,0,6,12,18,24")
    parser.add_argument("--data", default="data")
    parser.add_argument("--backgrounds", default="data/backgrounds")
    parser.add_argument("--weights-dir", default="weights")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    detectors, methods = args.detectors.split(","), args.methods.split(",")
    seeds, angles = list(map(int, args.seeds.split(","))), list(map(float, args.angles.split(",")))
    import math
    for values in [detectors, methods, seeds, angles]:
        if not values or len(set(values)) != len(values):
            parser.error("Selections must be nonempty and distinct")
    if (any(d not in ["yolo8", "yolo11"] for d in detectors)
            or any(m not in ["FIPatch", "IMPACT"] for m in methods)
            or min(seeds) < 0 or not all(math.isfinite(a) for a in angles)):
        parser.error("Select YOLOv8/YOLOv11, FIPatch/IMPACT, nonnegative seeds and finite angles")
    source, output = (ROOT/args.input).resolve(), (ROOT/args.output).resolve()
    if output == source or source in output.parents or output in source.parents:
        parser.error("Output must be separate from the source run")
    if output.exists() and any(output.iterdir()) and not args.resume:
        parser.error("Output exists; choose a new folder or --resume")
    os.environ.setdefault("YOLO_CONFIG_DIR", str(output/".ultralytics"))
    import numpy as np
    import torch
    from baselines.grid_utils import build_env_from_args
    from baselines.modern.oracle import AttackOracle, RecordingDetector, atomic_json

    class ReplayOracle(AttackOracle):
        def _save_best_images(self):
            pass  # Raw metrics and patch remain saved; avoid hundreds of display PNGs.

    # Preflight every requested saved patch before loading a GPU model.
    records, input_hashes = {}, {}
    for detector in detectors:
        environment = source/"environment"/(detector+".json")
        input_hashes[str(environment)] = sha(environment)
        for seed in seeds:
            for method in methods:
                result_path = source/detector/str(seed)/method/"result.json"
                result = read(result_path)
                if result.get("status") not in ["completed", "query_limit"] or not result.get("best"):
                    raise ValueError(f"No completed source search: {result_path}")
                relative = Path(result["attempt"].replace("\\", "/"))
                attempt = (source/relative).resolve()
                if source not in attempt.parents:
                    raise ValueError("Source attempt escapes the input folder")
                patch, contract_path = attempt/"best_patch.npz", attempt/"contract.json"
                for path in [result_path, patch, contract_path]:
                    input_hashes[str(path)] = sha(path)
                records[detector, seed, method] = (result, patch, read(contract_path))
        weights = ROOT/args.weights_dir/("yolov8n.pt" if detector == "yolo8" else "yolo11n.pt")
        input_hashes[str(weights)] = sha(weights)
    for path in [ROOT/args.data/name for name in ["stop_sign.png", "stop_sign_uv.png", "pole.png"]]:
        input_hashes[str(path)] = sha(path)
    for path in sorted((ROOT/args.backgrounds).iterdir()):
        if path.is_file() and path.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"]:
            input_hashes[str(path)] = sha(path)
    code = [Path(__file__)]
    for directory in ["baselines", "envs", "detectors", "utils"]:
        code.extend((ROOT/directory).rglob("*.py"))
    signature = dict(detectors=detectors, methods=methods, seeds=seeds, angles=angles,
        input_sha256=input_hashes, code_sha256={str(p): sha(p) for p in code},
        device=args.device, torch_version=torch.__version__, purpose="fixed saved-patch rotation replay")
    if (output/"replay_manifest.json").exists() and read(output/"replay_manifest.json") != signature:
        raise ValueError("Resume settings, inputs, or code changed; choose a new output folder")
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output/"replay_manifest.json", signature)
    rows = [json.loads(line) for line in (output/"angle_rows.jsonl").read_text().splitlines()] if (output/"angle_rows.jsonl").exists() else []
    completed = {(r["detector"], r["seed"], r["method"], r["angle_deg"]) for r in rows}
    total = len(detectors)*len(methods)*len(seeds)*len(angles)
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    started, new_rows = time.monotonic(), 0
    error = None
    print(f"Replaying {len(records)} SAVED patches at {len(angles)} angles: {total} evaluations. No optimization.", flush=True)
    try:
        for detector in detectors:
            config = dict(read(source/"environment"/(detector+".json"))["config"])
            config.update(data=str((ROOT/args.data).resolve()), bgdir=str((ROOT/args.backgrounds).resolve()),
                detector_device=args.device, info_image_every=0,
                yolo_weights=str((ROOT/args.weights_dir/("yolov8n.pt" if detector=="yolo8" else "yolo11n.pt")).resolve()))
            print(f"Loading {detector} once", flush=True)
            env = build_env_from_args(SimpleNamespace(**config))
            recorder = RecordingDetector(env.det); env.det = recorder
            try:
                for seed in seeds:
                    for method in methods:
                        original, path, expected = records[detector, seed, method]
                        with np.load(path, allow_pickle=False) as patch:
                            mask, rgb = patch["mask"].copy(), patch["rgb"].copy()
                        for angle in angles:
                            if (detector, seed, method, angle) in completed:
                                continue
                            env.fixed_angle_deg = angle
                            folder = output/detector/str(seed)/method/("angle_"+f"{angle:+g}".replace("+", "p").replace("-", "m"))
                            oracle, tick = None, time.monotonic()
                            try:
                                oracle = ReplayOracle(env, recorder, seed=seed, config=config,
                                    candidate_limit=1, deadline=float("inf"), folder=folder)
                                for key in ["place_seed", "transform_seeds", "image_size", "sign_size",
                                            "sign_day_sha256", "sign_uv_sha256", "background_sha256"]:
                                    if oracle.contract[key] != expected[key]:
                                        raise ValueError(f"Source scene changed: {detector}/{seed}/{method}: {key}")
                                clean_views = [float(r["target_conf"]) for r in recorder.records[-2*oracle.k:]]
                                oracle.search_stage = "saved_patch_angle_replay"
                                metrics = oracle.evaluate(mask, rgb)
                                if not np.isclose(metrics["area_frac"], original["best"]["area_frac"], atol=1e-12, rtol=0):
                                    raise ValueError("Replayed patch area differs from the saved patch")
                                patched_views = [float(r["target_conf"]) for r in recorder.records[-2*oracle.k:]]
                                row = dict(detector=detector, method=method, seed=seed, angle_deg=angle,
                                    **metrics, replay_seconds=time.monotonic()-tick,
                                    clean_view_confidences=clean_views, patched_view_confidences=patched_views,
                                    source_patch_sha256=input_hashes[str(path)])
                                with (output/"angle_rows.jsonl").open("a", encoding="utf-8") as stream:
                                    stream.write(json.dumps(row, allow_nan=False)+"\n")
                                rows.append(row); new_rows += 1
                                completed.add((detector, seed, method, angle))
                                eta = (time.monotonic()-started)/new_rows*(total-len(rows))
                                print(f"[{len(rows)}/{total}] {detector}/{seed}/{method} {angle:+g}deg: "
                                    f"UV={metrics['after_conf']:.3f}, ASR={int(metrics['success'])}; ETA {eta/60:.1f}m", flush=True)
                            finally:
                                if oracle is not None:
                                    oracle.close()
            finally:
                env.close()
            export(output, rows, seeds)
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        export(output, rows, seeds)
        atomic_json(output/"run_status.json", dict(complete=len(rows)==total and error is None,
            completed_evaluations=len(rows), planned_evaluations=total, elapsed_seconds=time.monotonic()-started,
            error=error, metric="same confidence ASR and separate joint area/daylight checks as source run",
            angle_interpretation="in-plane sign rotation, not physical 3D viewing angle"))
    print(f"DONE. Summary: {output/'angle_summary.csv'}", flush=True)


if __name__ == "__main__":
    main()
