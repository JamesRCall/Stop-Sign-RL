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

usage() {
  cat <<EOF
Usage: $0 [options]

Options:
  --episodes N         (default: $EPISODES)
  --deterministic {0|1} (default: $DETERMINISTIC)
  --tb DIR             (default: $TB_DIR)
  --tb-tag TAG         (default: $TB_TAG)
  --ckpt DIR           (default: $CKPT_DIR)
  --model PATH         (default: latest in --ckpt)
  --vecnorm PATH       (optional VecNormalize stats .pkl)
  -h, --help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --episodes) EPISODES="$2"; shift 2;;
    --deterministic) DETERMINISTIC="$2"; shift 2;;
    --tb) TB_DIR="$2"; shift 2;;
    --tb-tag) TB_TAG="$2"; shift 2;;
    --ckpt) CKPT_DIR="$2"; shift 2;;
    --model) MODEL="$2"; shift 2;;
    --vecnorm) VECNORM="$2"; shift 2;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown option: $1"; usage; exit 1;;
  esac
done

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

YOLO_DEVICE="${YOLO_DEVICE}" \
python tools/eval_policy.py \
  --ckpt "${CKPT_DIR}" \
  --episodes "${EPISODES}" \
  --deterministic "${DETERMINISTIC}" \
  --tb "${TB_DIR}" \
  --tb-tag "${TB_TAG}" \
  "${EXTRA_ARGS[@]}"
