#!/usr/bin/env bash
set -euo pipefail

# Simple runner to compare PPO vs greedy/random baselines over multiple seeds.
# Configure via env vars or CLI flags below.

N="${N:-20}"
SEED_BASE="${SEED_BASE:-123}"
EVAL_K="${EVAL_K:-3}"
GRID_CELL="${GRID_CELL:-16}"
PAINT="${PAINT:-yellow}"
YOLO_VERSION="${YOLO_VERSION:-8}"
YOLO_WEIGHTS="${YOLO_WEIGHTS:-./weights/yolo8n.pt}"
DETECTOR="${DETECTOR:-yolo}"
DETECTOR_MODEL="${DETECTOR_MODEL:-}"
PPO_MODEL="${PPO_MODEL:-}"
PPO_CKPT_DIR="${PPO_CKPT_DIR:-./_runs/checkpoints}"
PPO_VECNORM="${PPO_VECNORM:-}"
BG_MODE="${BG_MODE:-dataset}"
TRANSFORM_STRENGTH="${TRANSFORM_STRENGTH:-1.0}"
AREA_TARGET="${AREA_TARGET:-0.25}"
LAMBDA_AREA="${LAMBDA_AREA:-0.70}"
LAMBDA_DAY="${LAMBDA_DAY:-0.0}"
RANDOM_TRIALS="${RANDOM_TRIALS:-50}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-./_runs/baseline_compare_${RUN_TAG}}"
mkdir -p "${OUT_ROOT}"
GREEDY_LIST="${OUT_ROOT}/greedy_runs.txt"
RANDOM_LIST="${OUT_ROOT}/random_runs.txt"
PPO_SUMMARY_JSON="${OUT_ROOT}/ppo_summary.json"
PPO_EPISODES_JSON="${OUT_ROOT}/ppo_episodes.json"
GREEDY_OUT_ROOT="${OUT_ROOT}/greedy"
RANDOM_OUT_ROOT="${OUT_ROOT}/random"
mkdir -p "${GREEDY_OUT_ROOT}" "${RANDOM_OUT_ROOT}"
> "${GREEDY_LIST}"
> "${RANDOM_LIST}"

DETECTOR_ARGS=(--detector "${DETECTOR}")
if [[ -n "${DETECTOR_MODEL}" ]]; then
  DETECTOR_ARGS+=(--detector-model "${DETECTOR_MODEL}")
fi

echo "[RUN] N=${N} seed_base=${SEED_BASE} eval_K=${EVAL_K} grid=${GRID_CELL} paint=${PAINT} detector=${DETECTOR} model=${DETECTOR_MODEL:-<default>}"

# 1) PPO eval over N episodes using seed base
if [[ -n "${PPO_MODEL}" ]]; then
  echo "[PPO] Evaluating PPO over ${N} episodes with seed base ${SEED_BASE}"
  VECNORM_ARG=()
  if [[ -z "${PPO_VECNORM}" ]]; then
    # Auto-detect VecNormalize stats in the checkpoint dir if present.
    if [[ -f "${PPO_CKPT_DIR}/vecnormalize.pkl" ]]; then
      PPO_VECNORM="${PPO_CKPT_DIR}/vecnormalize.pkl"
    fi
  fi
  if [[ -n "${PPO_VECNORM}" ]]; then
    VECNORM_ARG=(--vecnorm "${PPO_VECNORM}")
  fi
  python tools/eval_policy.py \
    --model "${PPO_MODEL}" \
    --ckpt "${PPO_CKPT_DIR}" \
    --episodes "${N}" \
    --seed "${SEED_BASE}" \
    --yolo-version "${YOLO_VERSION}" \
    --eval-K "${EVAL_K}" \
    --grid-cell "${GRID_CELL}" \
    --paint "${PAINT}" \
    --yolo-weights "${YOLO_WEIGHTS}" \
    "${DETECTOR_ARGS[@]}" \
    --bg-mode "${BG_MODE}" \
    --transform-strength "${TRANSFORM_STRENGTH}" \
    --area-target "${AREA_TARGET}" \
    --lambda-area "${LAMBDA_AREA}" \
    --lambda-day "${LAMBDA_DAY}" \
    --out-json "${PPO_SUMMARY_JSON}" \
    --out-episodes-json "${PPO_EPISODES_JSON}" \
    "${VECNORM_ARG[@]}"
else
  echo "[PPO] PPO_MODEL not set; skipping PPO eval."
fi

# 2) Greedy + Random over the same seeds
for ((i=0; i<${N}; i++)); do
  seed=$((SEED_BASE + i))
  echo "[SEED ${seed}] greedy"
  python baselines/greedy_grid/greedy_search.py \
    --seed "${seed}" \
    --out "${GREEDY_OUT_ROOT}" \
    --yolo-version "${YOLO_VERSION}" \
    --eval-K "${EVAL_K}" \
    --grid-cell "${GRID_CELL}" \
    --paint "${PAINT}" \
    --yolo-weights "${YOLO_WEIGHTS}" \
    "${DETECTOR_ARGS[@]}" \
    --bg-mode "${BG_MODE}" \
    --transform-strength "${TRANSFORM_STRENGTH}" \
    --area-target "${AREA_TARGET}" \
    --lambda-area "${LAMBDA_AREA}" \
    --lambda-day "${LAMBDA_DAY}"
  latest_greedy="$(ls -td "${GREEDY_OUT_ROOT}"/greedy_* 2>/dev/null | head -n 1 || true)"
  [[ -n "${latest_greedy}" ]] && echo "${latest_greedy}" >> "${GREEDY_LIST}"

  echo "[SEED ${seed}] random (trials=${RANDOM_TRIALS})"
  python baselines/random_grid/random_search.py \
    --seed "${seed}" \
    --out "${RANDOM_OUT_ROOT}" \
    --yolo-version "${YOLO_VERSION}" \
    --trials "${RANDOM_TRIALS}" \
    --select-by "success_area" \
    --eval-K "${EVAL_K}" \
    --grid-cell "${GRID_CELL}" \
    --paint "${PAINT}" \
    --yolo-weights "${YOLO_WEIGHTS}" \
    "${DETECTOR_ARGS[@]}" \
    --bg-mode "${BG_MODE}" \
    --transform-strength "${TRANSFORM_STRENGTH}" \
    --area-target "${AREA_TARGET}" \
    --lambda-area "${LAMBDA_AREA}" \
    --lambda-day "${LAMBDA_DAY}"
  latest_random="$(ls -td "${RANDOM_OUT_ROOT}"/random_* 2>/dev/null | head -n 1 || true)"
  [[ -n "${latest_random}" ]] && echo "${latest_random}" >> "${RANDOM_LIST}"
done

python tools/aggregate_baselines.py \
  --ppo-json "${PPO_SUMMARY_JSON}" \
  --greedy-list "${GREEDY_LIST}" \
  --random-list "${RANDOM_LIST}" \
  --out "${OUT_ROOT}/compare_summary.json"

echo "[DONE] Completed PPO + baselines over ${N} seeds."
echo "[DONE] Compare summary: ${OUT_ROOT}/compare_summary.json"
