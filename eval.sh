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
OUT_JSON="${OUT_JSON:-}"
OUT_EPISODES_JSON="${OUT_EPISODES_JSON:-}"
SAVE_OVERLAY_DIR="${SAVE_OVERLAY_DIR:-}"
SAVE_COMPOSITED_DIR="${SAVE_COMPOSITED_DIR:-}"
SEED="${SEED:-}"

# Env defaults (mirror train.sh)
EVAL_K="${EVAL_K:-3}"
GRID_CELL="${GRID_CELL:-16}"
LAMBDA_AREA="${LAMBDA_AREA:-0.70}"
LAMBDA_EFFICIENCY="${LAMBDA_EFFICIENCY:-0.40}"
EFFICIENCY_EPS="${EFFICIENCY_EPS:-0.02}"
LAMBDA_PERCEPTUAL="${LAMBDA_PERCEPTUAL:-0.0}"
LAMBDA_DAY="${LAMBDA_DAY:-0.0}"
AREA_TARGET="${AREA_TARGET:-0.25}"
STEP_COST="${STEP_COST:-0.012}"
STEP_COST_AFTER_TARGET="${STEP_COST_AFTER_TARGET:-0.14}"
AREA_CAP_FRAC="${AREA_CAP_FRAC:-0.30}"
AREA_CAP_PENALTY="${AREA_CAP_PENALTY:--0.20}"
AREA_CAP_MODE="${AREA_CAP_MODE:-soft}"
AREA_CAP_START="${AREA_CAP_START:-0.80}"
AREA_CAP_END="${AREA_CAP_END:-0.30}"
LAMBDA_AREA_START="${LAMBDA_AREA_START:-}"
LAMBDA_AREA_END="${LAMBDA_AREA_END:-}"
OBS_SIZE="${OBS_SIZE:-224}"
OBS_MARGIN="${OBS_MARGIN:-0.10}"
OBS_INCLUDE_MASK="${OBS_INCLUDE_MASK:-1}"
CELL_COVER_THRESH="${CELL_COVER_THRESH:-0.60}"
SUCCESS_CONF="${SUCCESS_CONF:-0.20}"
TRANSFORM_STRENGTH="${TRANSFORM_STRENGTH:-1.0}"
FIXED_ANGLE_DEG="${FIXED_ANGLE_DEG:-}"
PAINT="${PAINT:-yellow}"
PAINT_LIST="${PAINT_LIST:-}"
EPISODE_STEPS="${EPISODE_STEPS:-300}"
UV_THRESHOLD="${UV_THRESHOLD:-0.75}"
YOLO_VERSION="${YOLO_VERSION:-8}"
YOLO_WEIGHTS="${YOLO_WEIGHTS:-}"
DETECTOR="${DETECTOR:-yolo}"
DETECTOR_MODEL="${DETECTOR_MODEL:-}"
START_TB="${START_TB:-1}"
PORT="${PORT:-6006}"

usage() {
  cat <<EOF
Usage: $0 [options]

Options:
  --episodes N         (default: $EPISODES)
  --seed N             (default: $SEED)
  --deterministic {0|1} (default: $DETERMINISTIC)
  --tb DIR             (default: $TB_DIR)
  --tb-tag TAG         (default: $TB_TAG)
  --port PORT          (default: $PORT)
  --no-tb              (do not auto-start TensorBoard)
  --ckpt DIR           (default: $CKPT_DIR)
  --model PATH         (default: latest in --ckpt)
  --vecnorm PATH       (optional VecNormalize stats .pkl)
  --out-json PATH      (default: <tb_run_dir>/summary.json)
  --out-episodes-json PATH (default: <tb_run_dir>/episodes.json)
  --save-overlay-dir PATH (optional: save per-episode overlay pattern PNGs)
  --save-composited-dir PATH (optional: save per-episode composited preview PNGs)
  --eval-k K           (default: $EVAL_K)
  --grid-cell N        (default: $GRID_CELL)
  --lambda-area X      (default: $LAMBDA_AREA or --lambda-area-end if set)
  --lambda-efficiency X (default: $LAMBDA_EFFICIENCY)
  --efficiency-eps X   (default: $EFFICIENCY_EPS)
  --lambda-perceptual X (default: $LAMBDA_PERCEPTUAL)
  --lambda-day X       (default: $LAMBDA_DAY)
  --area-target F      (default: $AREA_TARGET)
  --step-cost X        (default: $STEP_COST)
  --step-cost-after-target X (default: $STEP_COST_AFTER_TARGET)
  --area-cap-frac F    (default: $AREA_CAP_FRAC or --area-cap-end if set)
  --area-cap-penalty P (default: $AREA_CAP_PENALTY)
  --area-cap-mode MODE (default: $AREA_CAP_MODE)
  --obs-size N         (default: $OBS_SIZE)
  --obs-margin X       (default: $OBS_MARGIN)
  --obs-include-mask {0|1} (default: $OBS_INCLUDE_MASK)
  --cell-cover-thresh X (default: $CELL_COVER_THRESH)
  --success-conf X     (default: $SUCCESS_CONF)
  --transform-strength X (default: $TRANSFORM_STRENGTH)
  --fixed-angle-deg X  (default: $FIXED_ANGLE_DEG)
  --paint NAME         (default: $PAINT)
  --paint-list LIST    (default: $PAINT_LIST)
  --episode-steps N    (default: $EPISODE_STEPS)
  --uv-threshold X     (default: $UV_THRESHOLD)
  --yolo-version {8|11} (default: $YOLO_VERSION)
  --yolo-weights PATH  (default: $YOLO_WEIGHTS)
  --detector {yolo|torchvision|rtdetr} (default: $DETECTOR)
  --detector-model NAME (torchvision/transformers model id)
  -h, --help
EOF
}

TB_PID=""
TB_RUN_DIR=""

cleanup() {
  echo ""
  echo "[CLEANUP] Stopping background services..."
  [[ -n "${TB_PID}" ]] && kill "${TB_PID}" 2>/dev/null || true
}
trap cleanup EXIT

while [[ $# -gt 0 ]]; do
  case "$1" in
    --episodes) EPISODES="$2"; shift 2;;
    --seed) SEED="$2"; shift 2;;
    --deterministic) DETERMINISTIC="$2"; shift 2;;
    --tb) TB_DIR="$2"; shift 2;;
    --tb-tag) TB_TAG="$2"; shift 2;;
    --port) PORT="$2"; shift 2;;
    --no-tb) START_TB="0"; shift 1;;
    --ckpt) CKPT_DIR="$2"; shift 2;;
    --model) MODEL="$2"; shift 2;;
    --vecnorm) VECNORM="$2"; shift 2;;
    --out-json) OUT_JSON="$2"; shift 2;;
    --out-episodes-json) OUT_EPISODES_JSON="$2"; shift 2;;
    --save-overlay-dir) SAVE_OVERLAY_DIR="$2"; shift 2;;
    --save-composited-dir) SAVE_COMPOSITED_DIR="$2"; shift 2;;
    --eval-k) EVAL_K="$2"; shift 2;;
    --grid-cell) GRID_CELL="$2"; shift 2;;
    --lambda-area) LAMBDA_AREA="$2"; shift 2;;
    --lambda-efficiency) LAMBDA_EFFICIENCY="$2"; shift 2;;
    --efficiency-eps) EFFICIENCY_EPS="$2"; shift 2;;
    --lambda-perceptual) LAMBDA_PERCEPTUAL="$2"; shift 2;;
    --lambda-day) LAMBDA_DAY="$2"; shift 2;;
    --area-target) AREA_TARGET="$2"; shift 2;;
    --step-cost) STEP_COST="$2"; shift 2;;
    --step-cost-after-target) STEP_COST_AFTER_TARGET="$2"; shift 2;;
    --area-cap-frac) AREA_CAP_FRAC="$2"; shift 2;;
    --area-cap-penalty) AREA_CAP_PENALTY="$2"; shift 2;;
    --area-cap-mode) AREA_CAP_MODE="$2"; shift 2;;
    --obs-size) OBS_SIZE="$2"; shift 2;;
    --obs-margin) OBS_MARGIN="$2"; shift 2;;
    --obs-include-mask) OBS_INCLUDE_MASK="$2"; shift 2;;
    --cell-cover-thresh) CELL_COVER_THRESH="$2"; shift 2;;
    --success-conf) SUCCESS_CONF="$2"; shift 2;;
    --transform-strength) TRANSFORM_STRENGTH="$2"; shift 2;;
    --fixed-angle-deg) FIXED_ANGLE_DEG="$2"; shift 2;;
    --paint) PAINT="$2"; shift 2;;
    --paint-list) PAINT_LIST="$2"; shift 2;;
    --episode-steps) EPISODE_STEPS="$2"; shift 2;;
    --uv-threshold) UV_THRESHOLD="$2"; shift 2;;
    --yolo-version) YOLO_VERSION="$2"; shift 2;;
    --yolo-weights) YOLO_WEIGHTS="$2"; shift 2;;
    --detector) DETECTOR="$2"; shift 2;;
    --detector-model) DETECTOR_MODEL="$2"; shift 2;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown option: $1"; usage; exit 1;;
  esac
done

if [[ -z "${MODEL}" ]]; then
  if [[ -d "${CKPT_DIR}" ]]; then
    # Newest zip anywhere under CKPT_DIR (including subfolders).
    MODEL="$(ls -t "${CKPT_DIR}"/*.zip "${CKPT_DIR}"/*/*.zip 2>/dev/null | head -n 1 || true)"
    if [[ -n "${MODEL}" ]]; then
      CKPT_DIR="$(dirname "${MODEL}")"
    fi
  fi
fi

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
if [[ -n "${SEED}" ]]; then
  EXTRA_ARGS+=(--seed "${SEED}")
fi
if [[ -n "${YOLO_WEIGHTS}" ]]; then
  EXTRA_ARGS+=(--yolo-weights "${YOLO_WEIGHTS}")
fi
if [[ -n "${DETECTOR}" ]]; then
  EXTRA_ARGS+=(--detector "${DETECTOR}")
fi
if [[ -n "${DETECTOR_MODEL}" ]]; then
  EXTRA_ARGS+=(--detector-model "${DETECTOR_MODEL}")
fi
if [[ -n "${FIXED_ANGLE_DEG}" ]]; then
  EXTRA_ARGS+=(--fixed-angle-deg "${FIXED_ANGLE_DEG}")
fi

# If curriculum end values are provided, use those for evaluation.
if [[ -n "${LAMBDA_AREA_END}" ]]; then
  LAMBDA_AREA="${LAMBDA_AREA_END}"
fi
if [[ -n "${AREA_CAP_END}" ]]; then
  AREA_CAP_FRAC="${AREA_CAP_END}"
fi

# Use an isolated TensorBoard run directory per eval invocation to avoid
# mixing with stale root-level event files.
tb_safe_tag="$(echo "${TB_TAG}" | sed -E 's/[^A-Za-z0-9._-]+/_/g; s/^[_ .-]+//; s/[_ .-]+$//')"
if [[ -z "${tb_safe_tag}" ]]; then
  tb_safe_tag="eval"
fi
tb_stamp="$(date +%Y%m%d_%H%M%S)"
TB_RUN_DIR="${TB_DIR}/${tb_safe_tag}_${tb_stamp}"
if [[ -z "${OUT_JSON}" ]]; then
  OUT_JSON="${TB_RUN_DIR}/summary.json"
fi
if [[ -z "${OUT_EPISODES_JSON}" ]]; then
  OUT_EPISODES_JSON="${TB_RUN_DIR}/episodes.json"
fi
if [[ -n "${OUT_JSON}" ]]; then
  EXTRA_ARGS+=(--out-json "${OUT_JSON}")
fi
if [[ -n "${OUT_EPISODES_JSON}" ]]; then
  EXTRA_ARGS+=(--out-episodes-json "${OUT_EPISODES_JSON}")
fi
if [[ -n "${SAVE_OVERLAY_DIR}" ]]; then
  EXTRA_ARGS+=(--save-overlay-dir "${SAVE_OVERLAY_DIR}")
fi
if [[ -n "${SAVE_COMPOSITED_DIR}" ]]; then
  EXTRA_ARGS+=(--save-composited-dir "${SAVE_COMPOSITED_DIR}")
fi

echo "[EVAL] Running evaluation:"
echo "       episodes=${EPISODES} deterministic=${DETERMINISTIC} tb=${TB_RUN_DIR} tag=${TB_TAG}"
echo "       out_json=${OUT_JSON}"
echo "       out_episodes_json=${OUT_EPISODES_JSON}"
if [[ -n "${SAVE_OVERLAY_DIR}" ]]; then
  echo "       save_overlay_dir=${SAVE_OVERLAY_DIR}"
fi
if [[ -n "${SAVE_COMPOSITED_DIR}" ]]; then
  echo "       save_composited_dir=${SAVE_COMPOSITED_DIR}"
fi
echo ""

if [[ "${START_TB}" == "1" ]]; then
  mkdir -p "${TB_RUN_DIR}"
  echo "[TB] Starting TensorBoard on port ${PORT}, logdir=${TB_RUN_DIR}"
  tensorboard --logdir "${TB_RUN_DIR}" --port "${PORT}" > "${TB_RUN_DIR}/tensorboard.log" 2>&1 &
  TB_PID=$!
  sleep 2
  echo "[TB] PID=${TB_PID} | log: ${TB_RUN_DIR}/tensorboard.log"
  echo "[TB] Open: http://localhost:${PORT}"
fi

YOLO_DEVICE="${YOLO_DEVICE}" \
python tools/eval_policy.py \
  --ckpt "${CKPT_DIR}" \
  --episodes "${EPISODES}" \
  --deterministic "${DETERMINISTIC}" \
  --tb "${TB_RUN_DIR}" \
  --tb-tag "${TB_TAG}" \
  --eval-K "${EVAL_K}" \
  --grid-cell "${GRID_CELL}" \
  --lambda-area "${LAMBDA_AREA}" \
  --lambda-efficiency "${LAMBDA_EFFICIENCY}" \
  --efficiency-eps "${EFFICIENCY_EPS}" \
  --lambda-perceptual "${LAMBDA_PERCEPTUAL}" \
  --lambda-day "${LAMBDA_DAY}" \
  --area-target "${AREA_TARGET}" \
  --step-cost "${STEP_COST}" \
  --step-cost-after-target "${STEP_COST_AFTER_TARGET}" \
  --area-cap-frac "${AREA_CAP_FRAC}" \
  --area-cap-penalty "${AREA_CAP_PENALTY}" \
  --area-cap-mode "${AREA_CAP_MODE}" \
  --obs-size "${OBS_SIZE}" \
  --obs-margin "${OBS_MARGIN}" \
  --obs-include-mask "${OBS_INCLUDE_MASK}" \
  --cell-cover-thresh "${CELL_COVER_THRESH}" \
  --success-conf "${SUCCESS_CONF}" \
  --transform-strength "${TRANSFORM_STRENGTH}" \
  --paint "${PAINT}" \
  --paint-list "${PAINT_LIST}" \
  --episode-steps "${EPISODE_STEPS}" \
  --uv-threshold "${UV_THRESHOLD}" \
  --yolo-version "${YOLO_VERSION}" \
  "${EXTRA_ARGS[@]}"
