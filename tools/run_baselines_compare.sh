#!/usr/bin/env bash
set -euo pipefail

# Simple runner to compare PPO vs greedy/random baselines over multiple seeds.
# Configure via env vars or CLI flags below.

N="${N:-30}"
SEED_BASE="${SEED_BASE:-100000}"
DATA_DIR="${DATA_DIR:-./data}"
BG_DIR="${BG_DIR:-./data/backgrounds}"
NO_POLE="${NO_POLE:-0}"
EVAL_K="${EVAL_K:-3}"
GRID_CELL="${GRID_CELL:-16}"
PAINT="${PAINT:-yellow}"
YOLO_VERSION="${YOLO_VERSION:-8}"
YOLO_WEIGHTS="${YOLO_WEIGHTS:-./weights/yolov8n.pt}"
DETECTOR="${DETECTOR:-yolo}"
DETECTOR_MODEL="${DETECTOR_MODEL:-}"
PPO_MODEL="${PPO_MODEL:-}"
PPO_CKPT_DIR="${PPO_CKPT_DIR:-./_runs/checkpoints}"
PPO_VECNORM="${PPO_VECNORM:-}"
BG_MODE="${BG_MODE:-dataset}"
TRANSFORM_STRENGTH="${TRANSFORM_STRENGTH:-1.0}"
FIXED_ANGLE_DEG="${FIXED_ANGLE_DEG:-}"
AREA_TARGET="${AREA_TARGET:-0.25}"
LAMBDA_AREA="${LAMBDA_AREA:-0.70}"
LAMBDA_DAY="${LAMBDA_DAY:-1.0}"
DAY_TOLERANCE="${DAY_TOLERANCE:-0.05}"
RANDOM_TRIALS="${RANDOM_TRIALS:-1}"
SIGN_PROFILE="${SIGN_PROFILE:-stop}"
SIGN_IMAGE="${SIGN_IMAGE:-}"
SIGN_ACTIVE_IMAGE="${SIGN_ACTIVE_IMAGE:-}"
SOURCE_CLASS="${SOURCE_CLASS:-}"
ATTACK_MODE="${ATTACK_MODE:-disappearance}"
ATTACK_TARGET_CLASS="${ATTACK_TARGET_CLASS:-}"
ALLOWED_ALTERNATIVE_CLASSES="${ALLOWED_ALTERNATIVE_CLASSES:-}"
TARGET_CONF="${TARGET_CONF:-0.40}"
MIN_ATTACK_SUCCESS_RATE="${MIN_ATTACK_SUCCESS_RATE:-0.80}"
MIN_CLEAN_DETECTION_RATE="${MIN_CLEAN_DETECTION_RATE:-0.80}"
LOCALIZATION_IOU="${LOCALIZATION_IOU:-0.30}"
REQUIRE_SOURCE_SUPPRESSION="${REQUIRE_SOURCE_SUPPRESSION:-1}"
REQUIRE_DAY_PRESERVATION="${REQUIRE_DAY_PRESERVATION:-1}"
ANGLE_LIST="${ANGLE_LIST:--24,-18,-12,-6,0,6,12,18,24}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-./_runs/baseline_compare_${RUN_TAG}}"
if [[ -e "${OUT_ROOT}" ]]; then
  base="${OUT_ROOT}"
  i=1
  while [[ -e "${base}_v${i}" ]]; do
    i=$((i + 1))
  done
  OUT_ROOT="${base}_v${i}"
fi
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

OBJECTIVE_ARGS=(
  --data "${DATA_DIR}"
  --bgdir "${BG_DIR}"
  --sign-profile "${SIGN_PROFILE}"
  --attack-mode "${ATTACK_MODE}"
  --allowed-alternative-classes "${ALLOWED_ALTERNATIVE_CLASSES}"
  --target-conf "${TARGET_CONF}"
  --min-attack-success-rate "${MIN_ATTACK_SUCCESS_RATE}"
  --min-clean-detection-rate "${MIN_CLEAN_DETECTION_RATE}"
  --localization-iou "${LOCALIZATION_IOU}"
  --require-source-suppression "${REQUIRE_SOURCE_SUPPRESSION}"
  --require-day-preservation "${REQUIRE_DAY_PRESERVATION}"
  --day-tolerance "${DAY_TOLERANCE}"
)
if [[ "${NO_POLE}" == "1" ]]; then
  OBJECTIVE_ARGS+=(--no-pole)
fi
if [[ -n "${SIGN_IMAGE}" ]]; then
  OBJECTIVE_ARGS+=(--sign-image "${SIGN_IMAGE}")
fi
if [[ -n "${SIGN_ACTIVE_IMAGE}" ]]; then
  OBJECTIVE_ARGS+=(--sign-active-image "${SIGN_ACTIVE_IMAGE}")
fi
if [[ -n "${SOURCE_CLASS}" ]]; then
  OBJECTIVE_ARGS+=(--source-class "${SOURCE_CLASS}")
fi
if [[ -n "${ATTACK_TARGET_CLASS}" ]]; then
  OBJECTIVE_ARGS+=(--attack-target-class "${ATTACK_TARGET_CLASS}")
fi

echo "[RUN] N=${N} seed_base=${SEED_BASE} eval_K=${EVAL_K} grid=${GRID_CELL} paint=${PAINT} detector=${DETECTOR} model=${DETECTOR_MODEL:-<default>}"
echo "[RUN] sign=${SIGN_PROFILE} source=${SOURCE_CLASS:-profile-default} attack=${ATTACK_MODE} target=${ATTACK_TARGET_CLASS:-none} random_trials=${RANDOM_TRIALS}"
if [[ -n "${FIXED_ANGLE_DEG}" ]]; then
  echo "[RUN] fixed_angle_deg=${FIXED_ANGLE_DEG}"
fi

ANGLE_ARGS=()
if [[ -n "${FIXED_ANGLE_DEG}" ]]; then
  ANGLE_ARGS=(--fixed-angle-deg "${FIXED_ANGLE_DEG}")
fi

ANGLE_LIST_ARGS=()
if [[ -n "${ANGLE_LIST}" ]]; then
  angles="${ANGLE_LIST:--24,-18,-12,-6,0,6,12,18,24}"
  ANGLE_LIST_ARGS=(--angle-list="${angles}")
fi

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
    "${OBJECTIVE_ARGS[@]}" \
    --bg-mode "${BG_MODE}" \
    --transform-strength "${TRANSFORM_STRENGTH}" \
    "${ANGLE_ARGS[@]}" \
    "${ANGLE_LIST_ARGS[@]}" \
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
    "${OBJECTIVE_ARGS[@]}" \
    --bg-mode "${BG_MODE}" \
    --transform-strength "${TRANSFORM_STRENGTH}" \
    "${ANGLE_ARGS[@]}" \
    "${ANGLE_LIST_ARGS[@]}" \
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
    "${OBJECTIVE_ARGS[@]}" \
    --bg-mode "${BG_MODE}" \
    --transform-strength "${TRANSFORM_STRENGTH}" \
    "${ANGLE_ARGS[@]}" \
    "${ANGLE_LIST_ARGS[@]}" \
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
