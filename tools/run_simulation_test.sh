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
  SIM_TRAIN_EVAL_K=2              EOT samples used only during RL training
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
  SIM_DETECTOR=yolo|torchvision|rtdetr
  SIM_DETECTOR_MODEL=model_name_or_hf_id
  SIM_WEIGHTS=/abs/fine_grained_detector.pt
  SIM_YOLO_VERSION=8
  SIM_PAINT=yellow                  Fixed pixel-patch color for this run
  SIM_PAINT_ACTION_MODE=fixed       Or joint_palette
  SIM_PAINT_PALETTE=white,red,green,yellow,blue,orange
  SIM_PHYSICS=fixed_palette         Or synthetic_transport (default)

Adaptive bounded training:
  SIM_STEPS=800000                  Hard maximum; training never runs forever
  SIM_MIN_STEPS=50000               Earliest convergence stop
  SIM_EARLY_STOP_RATE=0.80          Rolling joint-success threshold
  SIM_EARLY_STOP_WINDOW=50          Completed-episode window
  SIM_SAVE_FREQ=100000              Policy/normalizer checkpoint interval

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
SIM_TRAIN_EVAL_K="${SIM_TRAIN_EVAL_K:-$SIM_EVAL_K}"
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
SIM_DETECTOR="${SIM_DETECTOR:-yolo}"
SIM_DETECTOR="${SIM_DETECTOR,,}"
SIM_DETECTOR_MODEL="${SIM_DETECTOR_MODEL:-}"
SIM_YOLO_VERSION="${SIM_YOLO_VERSION:-8}"
SIM_WEIGHTS="${SIM_WEIGHTS-}"
if [[ "${SIM_DETECTOR,,}" == "yolo" && -z "$SIM_WEIGHTS" ]]; then
  if [[ "$SIM_YOLO_VERSION" == "11" ]]; then
    SIM_WEIGHTS="$ROOT/weights/yolo11n.pt"
  else
    SIM_WEIGHTS="$ROOT/weights/yolov8n.pt"
  fi
fi
SIM_PAINT="${SIM_PAINT:-yellow}"
SIM_PAINT="${SIM_PAINT,,}"
SIM_PAINT_ACTION_MODE="${SIM_PAINT_ACTION_MODE:-fixed}"
SIM_PAINT_ACTION_MODE="${SIM_PAINT_ACTION_MODE,,}"
SIM_PAINT_PALETTE="${SIM_PAINT_PALETTE:-}"
SIM_PHYSICS="${SIM_PHYSICS:-synthetic_transport}"
SIM_PHYSICS="${SIM_PHYSICS,,}"
SIM_BGDIR="${SIM_BGDIR:-$ROOT/data/backgrounds}"
SIM_MIN_STEPS="${SIM_MIN_STEPS:-0}"
SIM_EARLY_STOP_RATE="${SIM_EARLY_STOP_RATE:-0.0}"
SIM_EARLY_STOP_WINDOW="${SIM_EARLY_STOP_WINDOW:-50}"
SIM_SAVE_FREQ="${SIM_SAVE_FREQ:-100000}"
RUN_TESTS="${RUN_TESTS:-1}"
RUN_VISUALS="${RUN_VISUALS:-1}"
RESUME="${RESUME:-1}"
RESTART_PARTIAL="${RESTART_PARTIAL:-1}"
CREATE_ARCHIVE="${CREATE_ARCHIVE:-1}"

for toggle in RUN_TESTS RUN_VISUALS RESUME RESTART_PARTIAL CREATE_ARCHIVE; do
  value="${!toggle}"
  if [[ "$value" != "0" && "$value" != "1" ]]; then
    echo "ERROR: $toggle must be 0 or 1, got '$value'" >&2
    exit 2
  fi
done

if [[ -e "$RESULTS" && ! -d "$RESULTS" ]]; then
  echo "ERROR: RESULTS exists but is not a directory: $RESULTS" >&2
  exit 2
fi
if [[ "$RESUME" == "0" && -d "$RESULTS" && \
      -n "$(find "$RESULTS" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
  echo "ERROR: RESULTS is not empty and RESUME=0: $RESULTS" >&2
  echo "Choose a new RESULTS directory or use RESUME=1 for an identical run." >&2
  exit 2
fi

export ROOT RUN_ID RESULTS PYTHON_BIN DEVICE SIM_SEEDS SIM_STEPS SIM_EPISODES
export SIM_QUERY_BUDGET SIM_EVAL_K SIM_TRAIN_EVAL_K SIM_GRID_CELL SIM_AREA_CAP SIM_EPISODE_STEPS
export SIM_SUPPORT_SCENES SIM_MAX_PREFIX SIM_TRANSFORM_STRENGTH SIM_SIGN_IMAGE
export SIM_ACTIVE_IMAGE SIM_SOURCE_CLASS SIM_ATTACK_MODE SIM_TARGET_CLASS
export SIM_ALLOWED_ALTS SIM_WEIGHTS SIM_YOLO_VERSION SIM_BGDIR
export SIM_DETECTOR SIM_DETECTOR_MODEL SIM_PAINT SIM_PHYSICS SIM_MIN_STEPS
export SIM_EARLY_STOP_RATE SIM_EARLY_STOP_WINDOW
export SIM_PAINT_ACTION_MODE SIM_PAINT_PALETTE
export SIM_SAVE_FREQ

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

valid_json_artifact() {
  local path="$1"
  local kind="$2"
  local expected="${3:-}"
  [[ -s "$path" ]] || return 1
  "$PYTHON_BIN" - "$path" "$kind" "$expected" <<'PY' >/dev/null 2>&1
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
kind = sys.argv[2]
expected = sys.argv[3]
value = json.loads(path.read_text(encoding="utf-8"))
if not isinstance(value, dict):
    raise SystemExit(1)
if kind == "prefix":
    tasks = value.get("tasks")
    accounting = value.get("query_accounting")
    valid = (
        isinstance(tasks, list)
        and bool(tasks)
        and all(
            isinstance(task, dict)
            and isinstance(task.get("prefixes"), list)
            and bool(task["prefixes"])
            for task in tasks
        )
        and isinstance(accounting, dict)
        and isinstance(
            accounting.get("offline_training_detector_image_queries"), int
        )
        and isinstance(
            accounting.get("development_prefix_detector_image_queries"), int
        )
    )
elif kind == "frozen":
    valid = (
        value.get("episodes") == int(expected)
        and isinstance(value.get("wilson_95_ci"), dict)
        and isinstance(value.get("detector_image_queries_total"), int)
    )
elif kind == "baseline":
    methods = {
        "random_search",
        "forward_greedy",
        "genetic_algorithm",
        "gaussian_es",
        "fipatch_style_pso_proxy",
        "cma_es",
    }
    results = value.get("results")
    valid = (
        isinstance(results, list)
        and len(results) == len(methods)
        and all(isinstance(item, dict) for item in results)
        and {item.get("method") for item in results} == methods
        and isinstance(value.get("invocation"), dict)
    )
elif kind == "pattern":
    valid = bool(value.get("actions") or value.get("selected_indices"))
else:
    valid = False
raise SystemExit(0 if valid else 1)
PY
}

archive_invalid_artifact() {
  local path="$1"
  local marker="$2"
  local label="$3"
  local stamp archived
  stamp="$(date -u +%Y%m%dT%H%M%SZ).${BASHPID:-$$}"
  if [[ -e "$path" ]]; then
    archived="$path.invalid.$stamp"
    mv -- "$path" "$archived"
    echo "[info] archived invalid $label artifact to $archived"
  fi
  if [[ -e "$marker" ]]; then
    archived="$marker.invalid.$stamp"
    mv -- "$marker" "$archived"
    echo "[info] archived stale $label marker to $archived"
  fi
}

command -v "$PYTHON_BIN" >/dev/null 2>&1 || {
  echo "ERROR: Python executable not found: $PYTHON_BIN" >&2
  exit 2
}
if [[ "$CREATE_ARCHIVE" == "1" ]]; then
  command -v tar >/dev/null 2>&1 || {
    echo "ERROR: tar is required to package simulation artifacts" >&2
    exit 2
  }
fi

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
integer("SIM_TRAIN_EVAL_K", 1)
integer("SIM_GRID_CELL", 1)
integer("SIM_EPISODE_STEPS", 1)
integer("SIM_SUPPORT_SCENES", 1)
integer("SIM_MAX_PREFIX", 1)
integer("SIM_MIN_STEPS", 0)
integer("SIM_EARLY_STOP_WINDOW", 1)
integer("SIM_SAVE_FREQ", 1)
if int(os.environ["SIM_MIN_STEPS"]) > int(os.environ["SIM_STEPS"]):
    raise SystemExit("ERROR: SIM_MIN_STEPS cannot exceed SIM_STEPS")

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
if mode != "untargeted_misclassification" and os.environ["SIM_ALLOWED_ALTS"].strip():
    raise SystemExit("ERROR: SIM_ALLOWED_ALTS must be empty outside untargeted mode")
source_class = os.environ["SIM_SOURCE_CLASS"].strip()
target_class = os.environ["SIM_TARGET_CLASS"].strip()
if not source_class:
    raise SystemExit("ERROR: SIM_SOURCE_CLASS must not be empty")
if target_class and target_class.casefold() == source_class.casefold():
    raise SystemExit("ERROR: source and target classes must differ")

detector = os.environ["SIM_DETECTOR"].strip().lower()
if detector not in {"yolo", "torchvision", "rtdetr"}:
    raise SystemExit("ERROR: SIM_DETECTOR must be yolo, torchvision, or rtdetr")
if detector == "rtdetr" and not os.environ["SIM_DETECTOR_MODEL"].strip():
    raise SystemExit("ERROR: SIM_DETECTOR_MODEL is required for rtdetr")
if detector == "torchvision" and not os.environ["SIM_DETECTOR_MODEL"].strip():
    raise SystemExit("ERROR: SIM_DETECTOR_MODEL is required for torchvision")
if detector == "yolo" and os.environ["SIM_DETECTOR_MODEL"].strip():
    raise SystemExit("ERROR: SIM_DETECTOR_MODEL must be empty for yolo")
if detector == "torchvision" and os.environ["SIM_DETECTOR_MODEL"].strip() not in {
    "fasterrcnn_resnet50_fpn_v2",
    "fasterrcnn_resnet50_fpn",
    "retinanet_resnet50_fpn_v2",
    "retinanet_resnet50_fpn",
    "ssd300_vgg16",
    "fcos_resnet50_fpn",
}:
    raise SystemExit("ERROR: unsupported SIM_DETECTOR_MODEL for torchvision")
if detector != "yolo" and os.environ["SIM_WEIGHTS"].strip():
    raise SystemExit(f"ERROR: SIM_WEIGHTS must be empty for detector {detector}")
if os.environ["SIM_YOLO_VERSION"] not in {"8", "11"}:
    raise SystemExit("ERROR: SIM_YOLO_VERSION must be 8 or 11")
if os.environ["SIM_PHYSICS"] not in {"synthetic_transport", "fixed_palette"}:
    raise SystemExit("ERROR: SIM_PHYSICS must be synthetic_transport or fixed_palette")
if os.environ["SIM_PAINT"].strip().lower() not in {
    "white", "red", "green", "yellow", "blue", "orange"
}:
    raise SystemExit("ERROR: SIM_PAINT must be white, red, green, yellow, blue, or orange")
paint_mode = os.environ["SIM_PAINT_ACTION_MODE"].strip().lower()
if paint_mode not in {"fixed", "joint_palette"}:
    raise SystemExit("ERROR: SIM_PAINT_ACTION_MODE must be fixed or joint_palette")
allowed_paints = {"white", "red", "green", "yellow", "blue", "orange"}
palette = [
    part.strip().lower()
    for part in os.environ["SIM_PAINT_PALETTE"].split(",")
    if part.strip()
]
if paint_mode == "joint_palette" and len(palette) < 2:
    raise SystemExit("ERROR: joint_palette mode requires at least two SIM_PAINT_PALETTE entries")
if len(palette) != len(set(palette)):
    raise SystemExit("ERROR: SIM_PAINT_PALETTE entries must be unique")
unknown_paints = sorted(set(palette) - allowed_paints)
if unknown_paints:
    raise SystemExit(
        "ERROR: unknown SIM_PAINT_PALETTE entries: " + ", ".join(unknown_paints)
    )
if paint_mode == "fixed" and palette:
    raise SystemExit("ERROR: SIM_PAINT_PALETTE must be empty in fixed mode")
if paint_mode == "joint_palette" and os.environ["SIM_PHYSICS"] != "fixed_palette":
    raise SystemExit("ERROR: joint_palette currently requires SIM_PHYSICS=fixed_palette")

for name in ("SIM_AREA_CAP", "SIM_TRANSFORM_STRENGTH"):
    try:
        value = float(os.environ[name])
    except ValueError as exc:
        raise SystemExit(f"ERROR: {name} must be numeric") from exc
    if not 0.0 < value <= 1.0:
        raise SystemExit(f"ERROR: {name} must be in (0, 1]")
try:
    early_rate = float(os.environ["SIM_EARLY_STOP_RATE"])
except ValueError as exc:
    raise SystemExit("ERROR: SIM_EARLY_STOP_RATE must be numeric") from exc
if not 0.0 <= early_rate <= 1.0:
    raise SystemExit("ERROR: SIM_EARLY_STOP_RATE must be in [0, 1]")

root = pathlib.Path(os.environ["ROOT"])
custom_day = os.environ["SIM_SIGN_IMAGE"].strip()
custom_active = os.environ["SIM_ACTIVE_IMAGE"].strip()
if custom_active and not custom_day:
    raise SystemExit("ERROR: SIM_ACTIVE_IMAGE requires SIM_SIGN_IMAGE")
required_files = [root / "data" / "pole.png"]
if custom_day:
    required_files.append(pathlib.Path(custom_day))
    required_files.append(pathlib.Path(custom_active or custom_day))
else:
    required_files.extend(
        [root / "data" / "stop_sign.png", root / "data" / "stop_sign_uv.png"]
    )
if os.environ["SIM_PHYSICS"] == "synthetic_transport":
    required_files.append(
        root / "data" / "synthetic" / "fluorescence_transport_v1.synthetic.json"
    )
if detector == "yolo":
    required_files.append(pathlib.Path(os.environ["SIM_WEIGHTS"]))
missing = [str(path) for path in required_files if not path.is_file()]
if missing:
    raise SystemExit(
        "ERROR: required simulation files are missing or not files:\n  "
        + "\n  ".join(missing)
    )
backgrounds = pathlib.Path(os.environ["SIM_BGDIR"])
if not backgrounds.is_dir():
    raise SystemExit(f"ERROR: SIM_BGDIR is not a directory: {backgrounds}")
supported = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
if not any(
    path.is_file() and path.suffix.lower() in supported
    for path in backgrounds.iterdir()
):
    raise SystemExit(f"ERROR: SIM_BGDIR contains no supported images: {backgrounds}")
PY

"$PYTHON_BIN" - <<'PY'
import importlib
import os
missing = []
required = ["numpy", "PIL", "torch", "gymnasium", "stable_baselines3", "sb3_contrib"]
detector = os.environ["SIM_DETECTOR"].strip().lower()
if detector == "yolo":
    required.append("ultralytics")
elif detector == "torchvision":
    required.append("torchvision")
elif detector == "rtdetr":
    required.append("transformers")
for name in required:
    try:
        importlib.import_module(name)
    except Exception as exc:
        missing.append(f"{name}: {exc}")
if missing:
    raise SystemExit("ERROR: missing/broken Python dependencies:\n  " + "\n  ".join(missing))
PY

if [[ "$RESUME" == "1" && -e "$STATUS_FILE" && \
      ! -f "$RESULTS/config/simulation_provenance.json" ]]; then
  completed_artifact="$(
    find "$RESULTS" -type f \
      \( -name '*.done' -o -name '*.zip' -o -name 'frozen_eval.json' \
         -o -name 'baselines_seed_*.json' \) \
      -print -quit 2>/dev/null
  )"
  if [[ -n "$completed_artifact" ]]; then
    echo "ERROR: cannot safely resume completed stages without simulation_provenance.json" >&2
    echo "Choose a new RESULTS directory." >&2
    exit 2
  fi
  echo "[info] restarting an uninitialized/preflight-failed result directory"
fi
if [[ "$RESUME" == "1" && -f "$RESULTS/config/simulation_provenance.json" ]]; then
  "$PYTHON_BIN" - "$RESULTS/config/simulation_provenance.json" <<'PY'
import json
import os
import pathlib
import subprocess
import sys

path = pathlib.Path(sys.argv[1])
try:
    saved = json.loads(path.read_text(encoding="utf-8"))
except Exception as exc:
    raise SystemExit(f"ERROR: invalid resume provenance {path}: {exc}") from exc
saved_parameters = saved.get("parameters")
if not isinstance(saved_parameters, dict):
    raise SystemExit("ERROR: resume provenance lacks a parameters object")
current_parameters = {
    key: os.environ[key]
    for key in sorted(os.environ)
    if key.startswith("SIM_") or key in {"DEVICE", "PYTHON_BIN"}
}
changed = sorted(
    key
    for key in set(saved_parameters) | set(current_parameters)
    if saved_parameters.get(key) != current_parameters.get(key)
)
try:
    current_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=os.environ["ROOT"],
        text=True,
        stderr=subprocess.DEVNULL,
    ).strip()
except Exception:
    current_commit = None
if saved.get("git_commit") != current_commit:
    changed.append("git_commit")
if changed:
    raise SystemExit(
        "ERROR: refusing to mix incompatible resume artifacts; changed: "
        + ", ".join(changed)
        + ". Choose a new RESULTS directory."
    )
PY
fi

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
calibration = (
    load_fluorescence_calibration(calibration_path)
    if os.environ["SIM_PHYSICS"] == "synthetic_transport"
    else None
)
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
    "detector": os.environ["SIM_DETECTOR"],
    "detector_model": os.environ["SIM_DETECTOR_MODEL"],
    "yolo_version": os.environ["SIM_YOLO_VERSION"],
    "yolo_weights": (
        str(pathlib.Path(os.environ["SIM_WEIGHTS"]).resolve())
        if os.environ["SIM_DETECTOR"] == "yolo"
        else None
    ),
    "detector_device": os.environ["DEVICE"],
    "paint": os.environ["SIM_PAINT"],
    "paint_list": "",
    "paint_action_mode": os.environ["SIM_PAINT_ACTION_MODE"],
    "paint_palette": os.environ["SIM_PAINT_PALETTE"],
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
}
if os.environ["SIM_PHYSICS"] == "synthetic_transport":
    common_env["physics_calibration"] = str(calibration_path)

tasks = []
for split, suffix, feature, transform in (
    ("train", "train-a", 0.0, 0.80),
    ("train", "train-b", 0.5, 1.00),
    ("development", "development", 1.0, 1.00),
):
    environment = dict(common_env)
    environment["eval_K"] = (
        int(os.environ["SIM_TRAIN_EVAL_K"])
        if split == "train"
        else int(os.environ["SIM_EVAL_K"])
    )
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
        "detector_id": (
            f"shared-simulation-{os.environ['SIM_DETECTOR']}-"
            f"{os.environ['SIM_DETECTOR_MODEL'] or os.environ['SIM_YOLO_VERSION']}"
        ),
        "source_class": os.environ["SIM_SOURCE_CLASS"],
        "target_class": target,
        "attack_mode": mode,
        "calibration_sha256": (
            calibration.canonical_sha256 if calibration is not None else ""
        ),
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
    "action_indexing": "canonical_full_grid",
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
    "$PYTHON_BIN" -m pytest -q tests
fi

# This visual generator specifically represents the bundled stop-sign fixture.
if [[ "$RUN_VISUALS" == "1" && -z "$SIM_SIGN_IMAGE" ]]; then
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
elif [[ "$RUN_VISUALS" == "1" ]]; then
  echo "[skip] bundled stop-sign visuals (custom sign assets configured)"
else
  echo "[skip] simulation visuals (RUN_VISUALS=0)"
fi

for seed in "${SEED_ARRAY[@]}"; do
  SEED_DIR="$RESULTS/rl/seed_$seed"
  RUN_MANIFEST="$SEED_DIR/amortized_run_manifest.json"
  TRAIN_MARKER="$SEED_DIR/.training.done"
  mkdir -p "$RESULTS/rl"

  if [[ "$RESUME" == "1" && -f "$RUN_MANIFEST" ]] && \
     "$PYTHON_BIN" - "$RUN_MANIFEST" <<'PY' >/dev/null 2>&1
import hashlib
import json
import pathlib
import sys

obj = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
root = pathlib.Path(sys.argv[1]).parent
artifacts = obj.get("final_artifacts", {})
accounting = obj.get("query_accounting", {})

def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()

policy = root / "amortized_prefix_policy_final.zip"
normalizer = root / "vecnormalize_final.pkl"
valid = (
    obj.get("run_status") == "completed"
    and policy.is_file()
    and normalizer.is_file()
    and artifacts.get("policy_sha256") == digest(policy)
    and artifacts.get("vecnormalize_sha256") == digest(normalizer)
    and isinstance(accounting.get("offline_training_detector_image_queries"), int)
    and isinstance(accounting.get("policy_environment_steps"), int)
)
raise SystemExit(0 if valid else 1)
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
      --minimum-steps "$SIM_MIN_STEPS" \
      --early-stop-success-rate "$SIM_EARLY_STOP_RATE" \
      --early-stop-window "$SIM_EARLY_STOP_WINDOW" \
      --n-steps 128 \
      --batch-size 64 \
      --seed "$seed" \
      --save-freq "$SIM_SAVE_FREQ" \
      --output-dir "$SEED_DIR" \
      --allow-uncalibrated-simulation \
      --allow-incomplete-splits

  FAMILY="$RESULTS/rl/seed_$seed/development_prefixes.json"
  PREFIX_MARKER="$RESULTS/rl/seed_$seed/.prefixes.done"
  if [[ "$RESUME" == "1" ]] && valid_json_artifact "$FAMILY" prefix; then
    touch "$PREFIX_MARKER"
  elif [[ "$RESUME" == "1" && ( -e "$FAMILY" || -e "$PREFIX_MARKER" ) ]]; then
    archive_invalid_artifact "$FAMILY" "$PREFIX_MARKER" "development-prefix"
  fi
  run_logged "prefixes_seed_$seed" "$PREFIX_MARKER" \
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
  PATTERN_MARKER="$RESULTS/rl/seed_$seed/.pattern.done"
  if [[ "$RESUME" == "1" && -e "$PATTERN_MARKER" ]] && \
     ! valid_json_artifact "$PATTERN" pattern; then
    archive_invalid_artifact "$PATTERN" "$PATTERN_MARKER" "selected-pattern"
  fi
  if ! stage_done "$PATTERN_MARKER"; then
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
descriptor = chosen.get("pattern_descriptor", {})
joint_palette = descriptor.get("paint_action_mode") == "joint_palette"
payload = {
    "selection_rule": "shortest_joint_success_else_longest_available",
    "selected_prefix_order": chosen["order"],
    "development_task_id": tasks[0].get("task_id"),
    "synthetic_non_empirical": True,
    "source_family": str(family_path.resolve()),
    "config": descriptor["config"],
}
if joint_palette:
    payload.update({
        "actions": chosen["ordered_actions"],
        "paint_action_mode": "joint_palette",
        "action_indexing": "canonical_full_grid",
        "action_encoding": descriptor["action_encoding"],
        "paint_palette": descriptor["paint_palette"],
        "paint_palette_sha256": descriptor["paint_palette_sha256"],
        "cell_material_assignments": descriptor["cell_material_assignments"],
    })
else:
    payload["selected_indices"] = chosen["selected_indices"]
output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
    touch "$PATTERN_MARKER"
  fi

  if [[ "$SIM_PAINT_ACTION_MODE" == "joint_palette" ]]; then
    PATTERN_TYPE="actions"
  else
    PATTERN_TYPE="selected_indices"
  fi
  EVAL_ARGS=(
    --pattern-json "$PATTERN"
    --pattern-type "$PATTERN_TYPE"
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
    --detector "$SIM_DETECTOR"
    --detector-device "$DEVICE"
    --eval-K "$SIM_EVAL_K"
    --grid-cell "$SIM_GRID_CELL"
    --episode-steps "$SIM_EPISODE_STEPS"
    --transform-strength "$SIM_TRANSFORM_STRENGTH"
    --paint "$SIM_PAINT"
    --paint-action-mode "$SIM_PAINT_ACTION_MODE"
    --paint-palette "$SIM_PAINT_PALETTE"
    --action-indexing canonical_full_grid
    --cell-cover-thresh 0.60
  )
  if [[ -n "$SIM_DETECTOR_MODEL" ]]; then
    EVAL_ARGS+=(--detector-model "$SIM_DETECTOR_MODEL")
  fi
  if [[ -n "$SIM_SIGN_IMAGE" ]]; then
    EVAL_ARGS+=(--sign-profile custom)
  else
    EVAL_ARGS+=(--sign-profile stop)
  fi
  if [[ "$SIM_ATTACK_MODE" == "targeted_misclassification" ]]; then
    EVAL_ARGS+=(--attack-target-class "$SIM_TARGET_CLASS")
  fi
  FROZEN_EVAL="$RESULTS/rl/seed_$seed/frozen_eval.json"
  FROZEN_MARKER="$RESULTS/rl/seed_$seed/.frozen_eval.done"
  if [[ "$RESUME" == "1" ]] && \
     valid_json_artifact "$FROZEN_EVAL" frozen "$SIM_EPISODES"; then
    touch "$FROZEN_MARKER"
  elif [[ "$RESUME" == "1" && ( -e "$FROZEN_EVAL" || -e "$FROZEN_MARKER" ) ]]; then
    archive_invalid_artifact "$FROZEN_EVAL" "$FROZEN_MARKER" "frozen-evaluation"
  fi
  run_logged "frozen_eval_seed_$seed" "$FROZEN_MARKER" \
    "$PYTHON_BIN" -u tools/eval_frozen_pattern.py "${EVAL_ARGS[@]}"

  BASELINE_REPORT="$RESULTS/baselines_seed_$seed.json"
  BASELINE_MARKER="$RESULTS/baselines_seed_$seed.done"
  if [[ "$RESUME" == "1" ]] && valid_json_artifact "$BASELINE_REPORT" baseline; then
    touch "$BASELINE_MARKER"
  elif [[ "$RESUME" == "1" && ( -e "$BASELINE_REPORT" || -e "$BASELINE_MARKER" ) ]]; then
    archive_invalid_artifact "$BASELINE_REPORT" "$BASELINE_MARKER" "baseline"
  fi
  run_logged "baselines_seed_$seed" "$BASELINE_MARKER" \
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
    family = json.loads(
        (run / "development_prefixes.json").read_text(encoding="utf-8")
    )
    training = json.loads(
        (run / "amortized_run_manifest.json").read_text(encoding="utf-8")
    )
    accounting = family.get("query_accounting", {})

    def nonnegative_integer(name):
        value = accounting.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"invalid {name} in development prefix accounting: {value!r}")
        return value

    training_queries = nonnegative_integer("offline_training_detector_image_queries")
    development_queries = nonnegative_integer(
        "development_prefix_detector_image_queries"
    )
    frozen_queries = int(evaluation["detector_image_queries_total"])
    training_accounting = training.get("query_accounting", {})
    policy_steps = training_accounting.get("policy_environment_steps")
    training_stop = training.get("training_stop", {})
    if isinstance(policy_steps, bool) or not isinstance(policy_steps, int) or policy_steps < 1:
        raise ValueError(f"invalid finalized policy_environment_steps: {policy_steps!r}")
    stop_reason = training_stop.get("stop_reason")
    if stop_reason not in {"rolling_joint_success_gate", "maximum_total_steps"}:
        raise ValueError(f"invalid finalized training stop reason: {stop_reason!r}")
    rows.append({
        "family": "rl_frozen_pattern",
        "method": "task_amortized_prefix_valid_support_batch",
        "seed": seed,
        "status": "completed",
        "success_rate": evaluation["success_rate"],
        "wilson_95_lower": evaluation["wilson_95_ci"]["lower"],
        "wilson_95_upper": evaluation["wilson_95_ci"]["upper"],
        "detector_queries": training_queries + development_queries + frozen_queries,
        "offline_training_detector_queries": training_queries,
        "development_prefix_detector_queries": development_queries,
        "frozen_evaluation_detector_queries": frozen_queries,
        "policy_environment_steps": policy_steps,
        "training_stop_reason": stop_reason,
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
            "offline_training_detector_queries": 0,
            "development_prefix_detector_queries": 0,
            "frozen_evaluation_detector_queries": 0,
            "policy_environment_steps": "",
            "training_stop_reason": "",
            "evaluated_candidates": result["evaluated_candidates"],
            "best_joint_success": best.get("joint_success", ""),
            "best_score": best.get("score", ""),
            "fidelity": result["fidelity"],
        })

fields = [
    "family", "method", "seed", "status", "success_rate",
    "wilson_95_lower", "wilson_95_upper", "detector_queries",
    "offline_training_detector_queries", "development_prefix_detector_queries",
    "frozen_evaluation_detector_queries",
    "policy_environment_steps", "training_stop_reason",
    "evaluated_candidates", "best_joint_success", "best_score", "fidelity",
]
with (root / "simulation_summary.csv").open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
(root / "simulation_summary.json").write_text(
    json.dumps({
        "claim_scope": "SYNTHETIC_NON_EMPIRICAL_SOFTWARE_SIMULATION_ONLY",
        "comparison_limit": (
            "RL detector_queries includes offline policy training, development-prefix "
            "selection, and frozen evaluation. Baseline detector_queries is its online "
            "search ledger; baselines are not independently frozen-evaluated here."
        ),
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
  if [[ "$CREATE_ARCHIVE" == "1" ]]; then
    echo "archive=$RESULTS/simulation_results.tar.gz"
  else
    echo "archive=disabled"
  fi
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

if [[ "$CREATE_ARCHIVE" == "1" ]]; then
  tar -czf "$RESULTS/simulation_results.tar.gz" \
    --exclude='./simulation_results.tar.gz' \
    -C "$RESULTS" .
fi
trap - ERR

echo "Simulation complete."
echo "Summary: $RESULTS/simulation_summary.csv"
if [[ "$CREATE_ARCHIVE" == "1" ]]; then
  echo "Archive: $RESULTS/simulation_results.tar.gz"
else
  echo "Archive: disabled (raw artifacts and SHA256SUMS retained)"
fi
