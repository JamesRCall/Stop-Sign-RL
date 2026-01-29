#!/usr/bin/env bash
set -euo pipefail

# ==============================
# Defaults (override via env or CLI)
# ==============================
YOLO_DEVICE="${YOLO_DEVICE:-cuda:0}"
EPISODES="${EPISODES:-20}"
DETERMINISTIC="${DETERMINISTIC:-1}"
TB_DIR="${TB_DIR:-./_runs/tb_eval}"
TB_TAG="${TB_TAG:-eval}"
CKPT_DIR="${CKPT_DIR:-./_runs/checkpoints}"
MODEL="${MODEL:-}"
VECNORM="${VECNORM:-}"
START_TB="${START_TB:-1}"
PORT="${PORT:-6006}"

usage() {
  cat <<EOF
Usage: $0 [options]

Options:
  --episodes N         (default: $EPISODES)
  --deterministic {0|1} (default: $DETERMINISTIC)
  --tb DIR             (default: $TB_DIR)
  --tb-tag TAG         (default: $TB_TAG)
  --port PORT          (default: $PORT)
  --no-tb              (do not auto-start TensorBoard)
  --ckpt DIR           (default: $CKPT_DIR)
  --model PATH         (default: latest in --ckpt)
  --vecnorm PATH       (optional VecNormalize stats .pkl)
  -h, --help
EOF
}

TB_PID=""

cleanup() {
  echo ""
  echo "[CLEANUP] Stopping background services..."
  [[ -n "${TB_PID}" ]] && kill "${TB_PID}" 2>/dev/null || true
}
trap cleanup EXIT

while [[ $# -gt 0 ]]; do
  case "$1" in
    --episodes) EPISODES="$2"; shift 2;;
    --deterministic) DETERMINISTIC="$2"; shift 2;;
    --tb) TB_DIR="$2"; shift 2;;
    --tb-tag) TB_TAG="$2"; shift 2;;
    --port) PORT="$2"; shift 2;;
    --no-tb) START_TB="0"; shift 1;;
    --ckpt) CKPT_DIR="$2"; shift 2;;
    --model) MODEL="$2"; shift 2;;
    --vecnorm) VECNORM="$2"; shift 2;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown option: $1"; usage; exit 1;;
  esac
done

if [[ -z "${VECNORM}" ]]; then
  DEFAULT_VN="${CKPT_DIR}/vecnormalize.pkl"
  if [[ -f "${DEFAULT_VN}" ]]; then
    VECNORM="${DEFAULT_VN}"
  fi
fi

EXTRA_ARGS=()
if [[ -n "${MODEL}" ]]; then
  EXTRA_ARGS+=(--model "${MODEL}")
fi
if [[ -n "${VECNORM}" ]]; then
  EXTRA_ARGS+=(--vecnorm "${VECNORM}")
fi

echo "[EVAL] Running evaluation:"
echo "       episodes=${EPISODES} deterministic=${DETERMINISTIC} tb=${TB_DIR} tag=${TB_TAG}"
echo ""

if [[ "${START_TB}" == "1" ]]; then
  mkdir -p "${TB_DIR}"
  echo "[TB] Starting TensorBoard on port ${PORT}, logdir=${TB_DIR}"
  tensorboard --logdir "${TB_DIR}" --port "${PORT}" > "${TB_DIR}/tensorboard.log" 2>&1 &
  TB_PID=$!
  sleep 2
  echo "[TB] PID=${TB_PID} | log: ${TB_DIR}/tensorboard.log"
  echo "[TB] Open: http://localhost:${PORT}"
fi

YOLO_DEVICE="${YOLO_DEVICE}" \
python tools/eval_policy.py \
  --ckpt "${CKPT_DIR}" \
  --episodes "${EPISODES}" \
  --deterministic "${DETERMINISTIC}" \
  --tb "${TB_DIR}" \
  --tb-tag "${TB_TAG}" \
  "${EXTRA_ARGS[@]}"
