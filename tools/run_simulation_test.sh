#!/usr/bin/env bash
# Self-contained, non-empirical end-to-end simulation/debug experiment.
#
# This runner intentionally does not claim physical validation, held-out
# certification, or a faithful FIPatch reproduction.  It exercises the full
# software path and records every artifact needed to debug that path.

set -Eeuo pipefail

usage() {
  cat <<'EOF'
Usage: bash tools/run_simulation_test.sh

Runs the complete synthetic simulation test:
  1. dependency and asset checks
  2. matched DAY/UV simulation renders
  3. task-conditioned RL training
  4. development-prefix extraction
  5. frozen-pattern evaluation on fresh RNG seeds
  6. equal-query/material random, greedy, GA, ES, PSO-proxy, and CMA-ES runs
  7. CSV/JSON summaries, checksums, and a compressed result archive

Important environment variables:
  RUN_ID=simulation_stop_v1       Unique result name
  RESULTS=/path/to/results        Default: runs/simulation/$RUN_ID
  PYTHON_BIN=.venv/bin/python     Python executable
  DEVICE=cuda:0                   Detector and policy device (cpu is valid)
  SIM_SEEDS="0"                   Space-separated independent seeds
  SIM_STEPS=4096                  RL environment steps per seed
  SIM_EPISODES=50                 Frozen-pattern evaluation episodes per seed
  SIM_QUERY_BUDGET=1000           Detector-image limit for every baseline
  SIM_EVAL_K=2                    EOT samples per candidate
  SIM_GRID_CELL=16                Grid-cell size in sign pixels
  SIM_AREA_CAP=0.20               Exact shared material-area cap
  RUN_TESTS=1                     Run pytest before the simulation
  RESUME=1                        Skip completed stages
  RESTART_PARTIAL=1               Archive and restart incomplete training dirs

Optional custom speed-sign/misclassification inputs:
  SIM_SIGN_IMAGE=/abs/day.png
  SIM_ACTIVE_IMAGE=/abs/active.png
  SIM_SOURCE_CLASS=speed_limit_25
  SIM_ATTACK_MODE=targeted_misclassification
  SIM_TARGET_CLASS=speed_limit_55
  SIM_ALLOWED_ALTS=class_a,class_b   (untargeted mode only)
  SIM_WEIGHTS=/abs/fine_grained_detector.pt
  SIM_YOLO_VERSION=8

All outputs are labeled SYNTHETIC/NON-EMPIRICAL and must not be reported as
physical or held-out certification evidence.
EOF
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  exit 0
fi
if [[ $# -ne 0 ]]; then
  echo "ERROR: unexpected arguments: $*" >&2
  usage >&2
  exit 2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd -P)"
cd "$ROOT"

RUN_ID="${RUN_ID:-simulation_stop_v1}"
RESULTS="${RESULTS:-$ROOT/runs/simulation/$RUN_ID}"
PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-cuda:0}"
SIM_SEEDS="${SIM_SEEDS:-0}"
SIM_STEPS="${SIM_STEPS:-4096}"
SIM_EPISODES="${SIM_EPISODES:-50}"
SIM_QUERY_BUDGET="${SIM_QUERY_BUDGET:-1000}"
SIM_EVAL_K="${SIM_EVAL_K:-2}"
SIM_GRID_CELL="${SIM_GRID_CELL:-16}"
SIM_AREA_CAP="${SIM_AREA_CAP:-0.20}"
SIM_EPISODE_STEPS="${SIM_EPISODE_STEPS:-32}"
SIM_SUPPORT_SCENES="${SIM_SUPPORT_SCENES:-2}"
SIM_MAX_PREFIX="${SIM_MAX_PREFIX:-16}"
SIM_TRANSFORM_STRENGTH="${SIM_TRANSFORM_STRENGTH:-1.0}"
SIM_SIGN_IMAGE="${SIM_SIGN_IMAGE:-}"
SIM_ACTIVE_IMAGE="${SIM_ACTIVE_IMAGE:-}"
SIM_SOURCE_CLASS="${SIM_SOURCE_CLASS:-stop sign}"
SIM_ATTACK_MODE="${SIM_ATTACK_MODE:-disappearance}"
SIM_TARGET_CLASS="${SIM_TARGET_CLASS:-}"
SIM_ALLOWED_ALTS="${SIM_ALLOWED_ALTS:-}"
SIM_WEIGHTS="${SIM_WEIGHTS:-$ROOT/weights/yolov8n.pt}"
SIM_YOLO_VERSION="${SIM_YOLO_VERSION:-8}"
SIM_BGDIR="${SIM_BGDIR:-$ROOT/data/backgrounds}"
RUN_TESTS="${RUN_TESTS:-1}"
RESUME="${RESUME:-1}"
RESTART_PARTIAL="${RESTART_PARTIAL:-1}"

export ROOT RUN_ID RESULTS PYTHON_BIN DEVICE SIM_SEEDS SIM_STEPS SIM_EPISODES
export SIM_QUERY_BUDGET SIM_EVAL_K SIM_GRID_CELL SIM_AREA_CAP SIM_EPISODE_STEPS
export SIM_SUPPORT_SCENES SIM_MAX_PREFIX SIM_TRANSFORM_STRENGTH SIM_SIGN_IMAGE
export SIM_ACTIVE_IMAGE SIM_SOURCE_CLASS SIM_ATTACK_MODE SIM_TARGET_CLASS
export SIM_ALLOWED_ALTS SIM_WEIGHTS SIM_YOLO_VERSION SIM_BGDIR

mkdir -p "$RESULTS" "$RESULTS/logs" "$RESULTS/config" "$RESULTS/timing"
STATUS_FILE="$RESULTS/STATUS.txt"
START_EPOCH="$(date +%s)"

on_error() {
  local code=$?
  {
    echo "FAILED"
    echo "exit_code=$code"
    echo "finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "Inspect logs in: $RESULTS/logs"
  } > "$STATUS_FILE"
  echo "Simulation failed (exit $code). Inspect $RESULTS/logs" >&2
  exit "$code"
}
trap on_error ERR

stage_done() {
  [[ "$RESUME" == "1" && -f "$1" ]]
}

run_logged() {
  local label="$1"
  local marker="$2"
  shift 2
  if stage_done "$marker"; then
    echo "[skip] $label"
    return 0
  fi
  local start end
  start="$(date +%s)"
  echo "[run ] $label (log: $RESULTS/logs/$label.log)"
  "$@" > "$RESULTS/logs/$label.log" 2>&1
  end="$(date +%s)"
  printf '%s\n' "$((end - start))" > "$RESULTS/timing/$label.seconds"
  touch "$marker"
  echo "[done] $label ($((end - start))s)"
}

command -v "$PYTHON_BIN" >/dev/null 2>&1 || {
  echo "ERROR: Python executable not found: $PYTHON_BIN" >&2
  exit 2
}

read -r -a SEED_ARRAY <<< "$SIM_SEEDS"
if [[ ${#SEED_ARRAY[@]} -eq 0 ]]; then
  echo "ERROR: SIM_SEEDS must contain at least one integer." >&2
  exit 2
fi

"$PYTHON_BIN" - <<'PY'
import os
import pathlib

def integer(name, minimum=0):
    raw = os.environ[name]
    try:
        value = int(raw)
    except ValueError as exc:
        raise SystemExit(f"ERROR: {name} must be an integer, got {raw!r}") from exc
    if value < minimum:
        raise SystemExit(f"ERROR: {name} must be >= {minimum}, got {value}")
    return value

for item in os.environ["SIM_SEEDS"].split():
    try:
        if int(item) < 0:
            raise ValueError
    except ValueError as exc:
        raise SystemExit(f"ERROR: invalid non-negative seed {item!r}") from exc
integer("SIM_STEPS", 1)
integer("SIM_EPISODES", 1)
integer("SIM_QUERY_BUDGET", 1)
integer("SIM_EVAL_K", 1)
integer("SIM_GRID_CELL", 1)
integer("SIM_EPISODE_STEPS", 1)
integer("SIM_SUPPORT_SCENES", 1)
integer("SIM_MAX_PREFIX", 1)

mode = os.environ["SIM_ATTACK_MODE"]
valid = {"disappearance", "untargeted_misclassification", "targeted_misclassification"}
if mode not in valid:
    raise SystemExit(f"ERROR: SIM_ATTACK_MODE must be one of {sorted(valid)}, got {mode!r}")
if mode == "targeted_misclassification" and not os.environ["SIM_TARGET_CLASS"].strip():
    raise SystemExit("ERROR: SIM_TARGET_CLASS is required for targeted_misclassification")
if mode != "targeted_misclassification" and os.environ["SIM_TARGET_CLASS"].strip():
    raise SystemExit("ERROR: SIM_TARGET_CLASS must be empty outside targeted mode")
if mode == "untargeted_misclassification" and not os.environ["SIM_ALLOWED_ALTS"].strip():
    raise SystemExit("ERROR: SIM_ALLOWED_ALTS is required for untargeted_misclassification")

for name in ("SIM_AREA_CAP", "SIM_TRANSFORM_STRENGTH"):
    try:
        value = float(os.environ[name])
    except ValueError as exc:
        raise SystemExit(f"ERROR: {name} must be numeric") from exc
    if not 0.0 < value <= 1.0:
        raise SystemExit(f"ERROR: {name} must be in (0, 1]")

root = pathlib.Path(os.environ["ROOT"])
required = [
    root / "data" / "stop_sign.png",
    root / "data" / "stop_sign_uv.png",
    root / "data" / "pole.png",
    root / "data" / "synthetic" / "fluorescence_transport_v1.synthetic.json",
    pathlib.Path(os.environ["SIM_WEIGHTS"]),
    pathlib.Path(os.environ["SIM_BGDIR"]),
]
if os.environ["SIM_SIGN_IMAGE"].strip():
    required.append(pathlib.Path(os.environ["SIM_SIGN_IMAGE"]))
if os.environ["SIM_ACTIVE_IMAGE"].strip():
    required.append(pathlib.Path(os.environ["SIM_ACTIVE_IMAGE"]))
missing = [str(path) for path in required if not path.exists()]
if missing:
    raise SystemExit("ERROR: required simulation inputs are missing:\n  " + "\n  ".join(missing))
PY

"$PYTHON_BIN" - <<'PY'
import importlib
missing = []
for name in ("numpy", "PIL", "torch", "gymnasium", "stable_baselines3", "sb3_contrib", "ultralytics"):
    try:
        importlib.import_module(name)
    except Exception as exc:
        missing.append(f"{name}: {exc}")
if missing:
    raise SystemExit("ERROR: missing/broken Python dependencies:\n  " + "\n  ".join(missing))
PY

cat > "$STATUS_FILE" <<EOF
RUNNING
started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
results=$RESULTS
claim_scope=SYNTHETIC_NON_EMPIRICAL_SOFTWARE_SIMULATION_ONLY
EOF

# Write all generated configuration in one strict, machine-readable step.
"$PYTHON_BIN" - <<'PY'
import json
import os
import pathlib
import platform
import subprocess
import sys

from utils.fluorescence_transport import load_fluorescence_calibration

root = pathlib.Path(os.environ["ROOT"]).resolve()
results = pathlib.Path(os.environ["RESULTS"]).resolve()
config_dir = results / "config"
config_dir.mkdir(parents=True, exist_ok=True)

custom_day = os.environ["SIM_SIGN_IMAGE"].strip()
custom_active = os.environ["SIM_ACTIVE_IMAGE"].strip()
if custom_day:
    profile = "custom"
    day = pathlib.Path(custom_day).resolve()
    active = pathlib.Path(custom_active).resolve() if custom_active else day
else:
    profile = "stop"
    day = (root / "data" / "stop_sign.png").resolve()
    active = (root / "data" / "stop_sign_uv.png").resolve()

calibration_path = (root / "data" / "synthetic" / "fluorescence_transport_v1.synthetic.json").resolve()
calibration = load_fluorescence_calibration(calibration_path)
mode = os.environ["SIM_ATTACK_MODE"]
target = os.environ["SIM_TARGET_CLASS"].strip() or None
common_env = {
    "data": str((root / "data").resolve()),
    "bgdir": str(pathlib.Path(os.environ["SIM_BGDIR"]).resolve()),
    "bg_mode": "dataset",
    "no_pole": False,
    "sign_profile": profile,
    "sign_image": str(day),
    "sign_active_image": str(active),
    "detector": "yolo",
    "detector_model": "",
    "yolo_version": os.environ["SIM_YOLO_VERSION"],
    "yolo_weights": str(pathlib.Path(os.environ["SIM_WEIGHTS"]).resolve()),
    "detector_device": os.environ["DEVICE"],
    "paint": "yellow",
    "paint_list": "",
    "episode_steps": int(os.environ["SIM_EPISODE_STEPS"]),
    "eval_K": int(os.environ["SIM_EVAL_K"]),
    "grid_cell": int(os.environ["SIM_GRID_CELL"]),
    "cell_cover_thresh": 0.60,
    "success_conf": 0.20,
    "target_conf": 0.40,
    "min_attack_success_rate": 0.80,
    "min_clean_detection_rate": 0.80,
    "localization_iou": 0.30,
    "require_source_suppression": 1,
    "require_day_preservation": 1,
    "day_tolerance": 0.05,
    "area_target": float(os.environ["SIM_AREA_CAP"]),
    "area_cap_frac": float(os.environ["SIM_AREA_CAP"]),
    "area_cap_mode": "hard",
    "transform_strength": float(os.environ["SIM_TRANSFORM_STRENGTH"]),
    "obs_size": 224,
    "obs_margin": 0.10,
    "obs_include_mask": 1,
    "detector_debug": 0,
    "allowed_alternative_classes": os.environ["SIM_ALLOWED_ALTS"],
    "physics_calibration": str(calibration_path),
}

tasks = []
for split, suffix, feature, transform in (
    ("train", "train-a", 0.0, 0.80),
    ("train", "train-b", 0.5, 1.00),
    ("development", "development", 1.0, 1.00),
):
    environment = dict(common_env)
    environment["transform_strength"] = min(
        1.0, float(common_env["transform_strength"]) * transform
    )
    tasks.append({
        "task_id": f"simulation-{suffix}",
        "split": split,
        # These are simulation roles, not claims of distinct physical assets.
        "sign_instance_id": f"shared-synthetic-sign-{suffix}",
        "background_split_id": f"shared-synthetic-backgrounds-{suffix}",
        "camera_id": f"synthetic-camera-{suffix}",
        "material_batch_id": f"synthetic-material-{suffix}",
        "physical_run_group": f"nonphysical-debug-{suffix}",
        "detector_id": "shared-simulation-yolo",
        "source_class": os.environ["SIM_SOURCE_CLASS"],
        "target_class": target,
        "attack_mode": mode,
        "calibration_sha256": calibration.canonical_sha256,
        "condition_features": [feature],
        "weight": 1.0,
        "environment": environment,
    })

manifest = {
    "schema_version": 1,
    "manifest_id": os.environ["RUN_ID"],
    "description": (
        "SYNTHETIC NON-EMPIRICAL SOFTWARE SIMULATION. Reuses bundled assets "
        "across role labels; incomplete splits and debug flags are intentional. "
        "Not physical, held-out, certification, or novelty evidence."
    ),
    "condition_feature_names": ["simulation_transform_variant"],
    "leakage_keys": ["physical_run_group"],
    "tasks": tasks,
}

baseline_env = dict(common_env)
baseline_env.pop("physics_calibration", None)
baseline_env.update({
    "source_class": os.environ["SIM_SOURCE_CLASS"],
    "attack_mode": mode,
    "attack_target_class": target or "",
})

method_config = {
    "genetic_algorithm": {"population_size": 8, "elite_fraction": 0.25},
    "gaussian_es": {"population_size": 8, "elite_fraction": 0.25, "sigma": 1.0},
    "fipatch_style_pso_proxy": {
        "swarm_size": 8, "inertia": 0.72, "cognitive": 1.49, "social": 1.49
    },
    "cma_es": {"population_size": 8, "sigma": 1.0},
}

def dump(name, value):
    (config_dir / name).write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )

dump("simulation_tasks.json", manifest)
dump("baseline_environment.json", baseline_env)
dump("baseline_methods.json", method_config)

try:
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
    ).strip()
except Exception:
    commit = None
dump("simulation_provenance.json", {
    "claim_scope": "SYNTHETIC_NON_EMPIRICAL_SOFTWARE_SIMULATION_ONLY",
    "not_valid_evidence_for": [
        "physical robustness", "held-out certification", "novelty", "USENIX acceptance"
    ],
    "run_id": os.environ["RUN_ID"],
    "git_commit": commit,
    "python": sys.version,
    "platform": platform.platform(),
    "parameters": {
        key: os.environ[key]
        for key in sorted(os.environ)
        if key.startswith("SIM_") or key in {"DEVICE", "PYTHON_BIN"}
    },
})
PY

if [[ "$RUN_TESTS" == "1" ]]; then
  run_logged "pytest" "$RESULTS/.pytest.done" \
    "$PYTHON_BIN" -m pytest -q
fi

# This visual generator specifically represents the bundled stop-sign fixture.
if [[ -z "$SIM_SIGN_IMAGE" ]]; then
  run_logged "simulation_visuals" "$RESULTS/.visuals.done" \
    "$PYTHON_BIN" tools/generate_training_sim_examples.py \
      --data "$ROOT/data" \
      --bgdir "$SIM_BGDIR" \
      --out-dir "$RESULTS/visuals" \
      --seed "${SEED_ARRAY[0]}" \
      --sets 4 \
      --grid-cell "$SIM_GRID_CELL" \
      --cell-cover-thresh 0.60 \
      --transform-strength "$SIM_TRANSFORM_STRENGTH" \
      --force-photometric 1
else
  echo "[skip] bundled stop-sign visuals (custom sign assets configured)"
fi

for seed in "${SEED_ARRAY[@]}"; do
  SEED_DIR="$RESULTS/rl/seed_$seed"
  RUN_MANIFEST="$SEED_DIR/amortized_run_manifest.json"
  TRAIN_MARKER="$SEED_DIR/.training.done"
  mkdir -p "$RESULTS/rl"

  if [[ "$RESUME" == "1" && -f "$RUN_MANIFEST" ]] && \
     "$PYTHON_BIN" - "$RUN_MANIFEST" <<'PY' >/dev/null 2>&1
import json, pathlib, sys
obj = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
raise SystemExit(0 if obj.get("run_status") == "completed" else 1)
PY
  then
    touch "$TRAIN_MARKER"
  elif [[ -d "$SEED_DIR" && -n "$(find "$SEED_DIR" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
    if [[ "$RESTART_PARTIAL" == "1" ]]; then
      archived="$SEED_DIR.partial.$(date -u +%Y%m%dT%H%M%SZ)"
      mv "$SEED_DIR" "$archived"
      echo "[info] archived incomplete training directory to $archived"
    else
      echo "ERROR: incomplete training directory exists: $SEED_DIR" >&2
      echo "Set RESTART_PARTIAL=1 to archive it and restart." >&2
      exit 2
    fi
  fi
  mkdir -p "$SEED_DIR"

  run_logged "train_seed_$seed" "$TRAIN_MARKER" \
    "$PYTHON_BIN" -u train_amortized.py \
      --task-manifest "$RESULTS/config/simulation_tasks.json" \
      --support-scenes "$SIM_SUPPORT_SCENES" \
      --risk-alpha 0.25 \
      --required-support-success-rate 0.80 \
      --required-clean-eligible-rate 0.80 \
      --required-day-preservation-rate 0.80 \
      --total-steps "$SIM_STEPS" \
      --n-steps 128 \
      --batch-size 64 \
      --seed "$seed" \
      --save-freq 1024 \
      --output-dir "$SEED_DIR" \
      --allow-uncalibrated-simulation \
      --allow-incomplete-splits

  FAMILY="$RESULTS/rl/seed_$seed/development_prefixes.json"
  if [[ "$RESUME" == "1" && -s "$FAMILY" ]]; then
    touch "$RESULTS/rl/seed_$seed/.prefixes.done"
  fi
  run_logged "prefixes_seed_$seed" "$RESULTS/rl/seed_$seed/.prefixes.done" \
    "$PYTHON_BIN" -u tools/generate_amortized_prefixes.py \
      --task-manifest "$RESULTS/config/simulation_tasks.json" \
      --run-manifest "$RUN_MANIFEST" \
      --model "$SEED_DIR/amortized_prefix_policy_final.zip" \
      --vecnormalize "$SEED_DIR/vecnormalize_final.pkl" \
      --out-json "$FAMILY" \
      --max-prefix-length "$SIM_MAX_PREFIX" \
      --max-development-detector-queries-per-task "$SIM_QUERY_BUDGET" \
      --seed "$((100000 + seed))" \
      --device "$DEVICE" \
      --allow-nonempirical-debug

  PATTERN="$RESULTS/rl/seed_$seed/selected_pattern.json"
  if ! stage_done "$RESULTS/rl/seed_$seed/.pattern.done"; then
    "$PYTHON_BIN" - "$FAMILY" "$PATTERN" <<'PY' \
      > "$RESULTS/logs/select_pattern_seed_$seed.log" 2>&1
import json
import pathlib
import sys

family_path, output_path = map(pathlib.Path, sys.argv[1:])
family = json.loads(family_path.read_text(encoding="utf-8"))
tasks = family.get("tasks", [])
if not tasks:
    raise SystemExit("development prefix family contains no task records")
prefixes = tasks[0].get("prefixes", [])
if not prefixes:
    raise SystemExit("development policy produced no selectable prefix")
successful = [p for p in prefixes if p.get("support_measurements", {}).get("joint_success")]
chosen = min(successful, key=lambda p: int(p["order"])) if successful else prefixes[-1]
payload = {
    "selected_indices": chosen["selected_indices"],
    "selection_rule": "shortest_joint_success_else_longest_available",
    "selected_prefix_order": chosen["order"],
    "development_task_id": tasks[0].get("task_id"),
    "synthetic_non_empirical": True,
    "source_family": str(family_path.resolve()),
}
output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
    touch "$RESULTS/rl/seed_$seed/.pattern.done"
  fi

  EVAL_ARGS=(
    --pattern-json "$PATTERN"
    --pattern-type selected_indices
    --episodes "$SIM_EPISODES"
    --seed-base "$((1000000 + seed * 10000))"
    --out-json "$RESULTS/rl/seed_$seed/frozen_eval.json"
    --data "$ROOT/data"
    --bgdir "$SIM_BGDIR"
    --bg-mode dataset
    --sign-image "${SIM_SIGN_IMAGE:-$ROOT/data/stop_sign.png}"
    --sign-active-image "${SIM_ACTIVE_IMAGE:-${SIM_SIGN_IMAGE:-$ROOT/data/stop_sign_uv.png}}"
    --source-class "$SIM_SOURCE_CLASS"
    --attack-mode "$SIM_ATTACK_MODE"
    --allowed-alternative-classes "$SIM_ALLOWED_ALTS"
    --area-cap-frac "$SIM_AREA_CAP"
    --area-cap-mode hard
    --yolo-weights "$SIM_WEIGHTS"
    --yolo-version "$SIM_YOLO_VERSION"
    --detector yolo
    --detector-device "$DEVICE"
    --eval-K "$SIM_EVAL_K"
    --grid-cell "$SIM_GRID_CELL"
    --episode-steps "$SIM_EPISODE_STEPS"
    --transform-strength "$SIM_TRANSFORM_STRENGTH"
    --paint yellow
    --cell-cover-thresh 0.60
  )
  if [[ -n "$SIM_SIGN_IMAGE" ]]; then
    EVAL_ARGS+=(--sign-profile custom)
  else
    EVAL_ARGS+=(--sign-profile stop)
  fi
  if [[ "$SIM_ATTACK_MODE" == "targeted_misclassification" ]]; then
    EVAL_ARGS+=(--attack-target-class "$SIM_TARGET_CLASS")
  fi
  if [[ "$RESUME" == "1" && -s "$RESULTS/rl/seed_$seed/frozen_eval.json" ]]; then
    touch "$RESULTS/rl/seed_$seed/.frozen_eval.done"
  fi
  run_logged "frozen_eval_seed_$seed" "$RESULTS/rl/seed_$seed/.frozen_eval.done" \
    "$PYTHON_BIN" -u tools/eval_frozen_pattern.py "${EVAL_ARGS[@]}"

  if [[ "$RESUME" == "1" && -s "$RESULTS/baselines_seed_$seed.json" ]]; then
    touch "$RESULTS/baselines_seed_$seed.done"
  fi
  run_logged "baselines_seed_$seed" "$RESULTS/baselines_seed_$seed.done" \
    "$PYTHON_BIN" -u tools/run_budgeted_comparison.py \
      --environment-json "$RESULTS/config/baseline_environment.json" \
      --methods random_search,forward_greedy,genetic_algorithm,gaussian_es,fipatch_style_pso_proxy,cma_es \
      --detector-query-limit "$SIM_QUERY_BUDGET" \
      --material-area-fraction "$SIM_AREA_CAP" \
      --scene-seed "$((2000000 + seed * 10000))" \
      --optimizer-seed "$seed" \
      --method-config-json "$RESULTS/config/baseline_methods.json" \
      --output "$RESULTS/baselines_seed_$seed.json"
done

"$PYTHON_BIN" - "$RESULTS" <<'PY'
import csv
import hashlib
import json
import os
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
rows = []
for seed_text in os.environ["SIM_SEEDS"].split():
    seed = int(seed_text)
    run = root / "rl" / f"seed_{seed}"
    evaluation = json.loads((run / "frozen_eval.json").read_text(encoding="utf-8"))
    rows.append({
        "family": "rl_frozen_pattern",
        "method": "task_amortized_prefix_valid_support_batch",
        "seed": seed,
        "status": "completed",
        "success_rate": evaluation["success_rate"],
        "wilson_95_lower": evaluation["wilson_95_ci"]["lower"],
        "wilson_95_upper": evaluation["wilson_95_ci"]["upper"],
        "detector_queries": evaluation["detector_image_queries_total"],
        "evaluated_candidates": evaluation["episodes"],
        "best_joint_success": evaluation["n_success"] > 0,
        "best_score": "",
        "fidelity": "synthetic_debug_frozen_pattern",
    })
    comparison = json.loads((root / f"baselines_seed_{seed}.json").read_text(encoding="utf-8"))
    for result in comparison["results"]:
        best = result.get("best") or {}
        rows.append({
            "family": "online_budgeted_baseline",
            "method": result["method"],
            "seed": seed,
            "status": result["status"],
            "success_rate": "",
            "wilson_95_lower": "",
            "wilson_95_upper": "",
            "detector_queries": result["detector_queries_used"],
            "evaluated_candidates": result["evaluated_candidates"],
            "best_joint_success": best.get("joint_success", ""),
            "best_score": best.get("score", ""),
            "fidelity": result["fidelity"],
        })

fields = [
    "family", "method", "seed", "status", "success_rate",
    "wilson_95_lower", "wilson_95_upper", "detector_queries",
    "evaluated_candidates", "best_joint_success", "best_score", "fidelity",
]
with (root / "simulation_summary.csv").open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
(root / "simulation_summary.json").write_text(
    json.dumps({
        "claim_scope": "SYNTHETIC_NON_EMPIRICAL_SOFTWARE_SIMULATION_ONLY",
        "rows": rows,
    }, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)

PY

END_EPOCH="$(date +%s)"
printf '%s\n' "$((END_EPOCH - START_EPOCH))" > "$RESULTS/timing/total.seconds"
{
  echo "COMPLETED"
  echo "finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "elapsed_seconds=$((END_EPOCH - START_EPOCH))"
  echo "claim_scope=SYNTHETIC_NON_EMPIRICAL_SOFTWARE_SIMULATION_ONLY"
  echo "summary_csv=$RESULTS/simulation_summary.csv"
  echo "archive=$RESULTS/simulation_results.tar.gz"
} > "$STATUS_FILE"

"$PYTHON_BIN" - "$RESULTS" <<'PY'
import hashlib
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
checksums = []
for path in sorted(root.rglob("*")):
    if not path.is_file() or path.name in {"SHA256SUMS", "simulation_results.tar.gz"}:
        continue
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    checksums.append(f"{digest}  {path.relative_to(root).as_posix()}")
(root / "SHA256SUMS").write_text("\n".join(checksums) + "\n", encoding="utf-8")
PY

tar -czf "$RESULTS/simulation_results.tar.gz" \
  --exclude='./simulation_results.tar.gz' \
  -C "$RESULTS" .
trap - ERR

echo "Simulation complete."
echo "Summary: $RESULTS/simulation_summary.csv"
echo "Archive: $RESULTS/simulation_results.tar.gz"
