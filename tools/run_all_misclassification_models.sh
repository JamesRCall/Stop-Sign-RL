#!/usr/bin/env bash
# Run the full configured targeted-misclassification detector matrix without
# streaming high-volume training output into an interactive terminal.
set -Eeuo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$ROOT"

if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x "$ROOT/.venv/bin/python" ]]; then
    PYTHON_BIN="$ROOT/.venv/bin/python"
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
  elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python)"
  else
    echo "ERROR: no Python executable found; set PYTHON_BIN explicitly" >&2
    exit 2
  fi
fi
CONFIG="${CONFIG:-$ROOT/configs/misclassification_models.json}"
OUTPUT="${OUTPUT:-$ROOT/runs/misclassification_matrix}"
DEVICE="${DEVICE:-auto}"
SEEDS="${SEEDS:-0 1 2 3 4}"
MODEL_IDS="${MODEL_IDS:-}"
MAX_STEPS="${MAX_STEPS:-800000}"
MINIMUM_STEPS="${MINIMUM_STEPS:-50000}"
EARLY_STOP_SUCCESS_RATE="${EARLY_STOP_SUCCESS_RATE:-0.80}"
EARLY_STOP_WINDOW="${EARLY_STOP_WINDOW:-50}"
SAVE_FREQ="${SAVE_FREQ:-100000}"
EVAL_EPISODES="${EVAL_EPISODES:-200}"
QUERY_BUDGET="${QUERY_BUDGET:-10000}"
EVAL_K="${EVAL_K:-8}"
TRAIN_EVAL_K="${TRAIN_EVAL_K:-2}"
SUPPORT_SCENES="${SUPPORT_SCENES:-4}"
EPISODE_STEPS="${EPISODE_STEPS:-64}"
MAX_PREFIX="${MAX_PREFIX:-64}"
RESUME="${RESUME:-1}"
RUN_TESTS="${RUN_TESTS:-1}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-1}"
ARCHIVE="${ARCHIVE:-1}"

for toggle in RESUME RUN_TESTS CONTINUE_ON_ERROR ARCHIVE; do
  value="${!toggle}"
  if [[ "$value" != "0" && "$value" != "1" ]]; then
    echo "ERROR: $toggle must be 0 or 1, got '$value'" >&2
    exit 2
  fi
done

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "ERROR: Python executable not found: $PYTHON_BIN" >&2
  exit 2
fi

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  "$PYTHON_BIN" tools/run_misclassification_matrix.py --help
  exit 0
fi
if [[ $# -ne 0 ]]; then
  echo "ERROR: this wrapper accepts configuration through environment variables only" >&2
  exit 2
fi

args=(
  --config "$CONFIG"
  --output "$OUTPUT"
  --python-bin "$PYTHON_BIN"
  --device "$DEVICE"
  --seeds "$SEEDS"
  --max-steps "$MAX_STEPS"
  --minimum-steps "$MINIMUM_STEPS"
  --early-stop-success-rate "$EARLY_STOP_SUCCESS_RATE"
  --early-stop-window "$EARLY_STOP_WINDOW"
  --save-freq "$SAVE_FREQ"
  --episodes "$EVAL_EPISODES"
  --query-budget "$QUERY_BUDGET"
  --eval-k "$EVAL_K"
  --train-eval-k "$TRAIN_EVAL_K"
  --support-scenes "$SUPPORT_SCENES"
  --episode-steps "$EPISODE_STEPS"
  --max-prefix "$MAX_PREFIX"
)
[[ -n "$MODEL_IDS" ]] && args+=(--model-ids "$MODEL_IDS")
[[ "$RESUME" == "1" ]] && args+=(--resume)
[[ "$RUN_TESTS" != "1" ]] && args+=(--skip-tests)
[[ "$CONTINUE_ON_ERROR" == "1" ]] && args+=(--continue-on-error)
[[ "$ARCHIVE" != "1" ]] && args+=(--no-archive)

exec "$PYTHON_BIN" -u tools/run_misclassification_matrix.py "${args[@]}"
