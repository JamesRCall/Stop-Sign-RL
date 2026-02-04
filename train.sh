#!/usr/bin/env bash
set -euo pipefail

# ==============================
# GPU + allocator knobs
# ==============================
export YOLO_DEVICE="${YOLO_DEVICE:-cuda:0}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:256}"

# Optional: keeps CPU from thrashing when you use lots of env logic
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

# Optional (often helps on big GPUs; safe if it doesn't)
export TORCH_CUDNN_V8_API_ENABLED="${TORCH_CUDNN_V8_API_ENABLED:-1}"

# ==============================
# Defaults (override via env or CLI)
# ==============================
NUM_ENVS="${NUM_ENVS:-1}"            # with CUDA detector, keep 1 unless you build a detector server
VEC="${VEC:-dummy}"                 # must be dummy for CUDA+YOLO in current architecture

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
AREA_CAP_STEPS="${AREA_CAP_STEPS:-500000}"
LAMBDA_AREA_START="${LAMBDA_AREA_START:-}"
LAMBDA_AREA_END="${LAMBDA_AREA_END:-}"
LAMBDA_AREA_STEPS="${LAMBDA_AREA_STEPS:-0}"
OBS_SIZE="${OBS_SIZE:-224}"
OBS_MARGIN="${OBS_MARGIN:-0.10}"
OBS_INCLUDE_MASK="${OBS_INCLUDE_MASK:-1}"
CELL_COVER_THRESH="${CELL_COVER_THRESH:-0.60}"
SUCCESS_CONF="${SUCCESS_CONF:-0.20}"
TRANSFORM_STRENGTH="${TRANSFORM_STRENGTH:-1.0}"
PAINT="${PAINT:-yellow}"
PAINT_LIST="${PAINT_LIST:-}"
CNN="${CNN:-custom}"
PHASE1_TRANSFORM_STRENGTH="${PHASE1_TRANSFORM_STRENGTH:-}"
PHASE2_TRANSFORM_STRENGTH="${PHASE2_TRANSFORM_STRENGTH:-}"
PHASE3_TRANSFORM_STRENGTH="${PHASE3_TRANSFORM_STRENGTH:-}"
PHASE1_LAMBDA_DAY="${PHASE1_LAMBDA_DAY:-}"
PHASE2_LAMBDA_DAY="${PHASE2_LAMBDA_DAY:-}"
PHASE3_LAMBDA_DAY="${PHASE3_LAMBDA_DAY:-}"
PHASE1_STEP_COST="${PHASE1_STEP_COST:-}"
PHASE2_STEP_COST="${PHASE2_STEP_COST:-}"
PHASE3_STEP_COST="${PHASE3_STEP_COST:-}"

YOLO_VERSION="${YOLO_VERSION:-8}"
YOLO_WEIGHTS="${YOLO_WEIGHTS:-}"
DETECTOR="${DETECTOR:-yolo}"
DETECTOR_MODEL="${DETECTOR_MODEL:-${DET_SERVER_MODEL:-}}"
START_DET_SERVER="${START_DET_SERVER:-0}"
DET_SERVER_PORT="${DET_SERVER_PORT:-5009}"
DET_SERVER_DEVICE="${DET_SERVER_DEVICE:-cuda:0}"
DET_SERVER_MODEL="${DET_SERVER_MODEL:-${DETECTOR_MODEL}}"

N_STEPS="${N_STEPS:-1024}"
BATCH="${BATCH:-1024}"              # default to rollout size for num_envs=1
TOTAL_STEPS="${TOTAL_STEPS:-800000}"
ENT_COEF="${ENT_COEF:-0.001}"
ENT_COEF_START="${ENT_COEF_START:-}"
ENT_COEF_END="${ENT_COEF_END:-}"
ENT_COEF_STEPS="${ENT_COEF_STEPS:-0}"

TB_DIR="${TB_DIR:-./_runs/tb}"
CKPT_DIR="${CKPT_DIR:-./_runs/checkpoints}"
OVR_DIR="${OVR_DIR:-./_runs/overlays}"
PORT="${PORT:-6006}"

SAVE_FREQ_UPDATES="${SAVE_FREQ_UPDATES:-2}"
STEP_LOG_EVERY="${STEP_LOG_EVERY:-1}"
STEP_LOG_KEEP="${STEP_LOG_KEEP:-1000}"
STEP_LOG_500="${STEP_LOG_500:-500}"
PY_MAIN="${PY_MAIN:-train_single_stop_sign.py}"
MULTIPHASE="${MULTIPHASE:-0}"
RESUME="${RESUME:-0}"
CHECK_ENV="${CHECK_ENV:-1}"

# Monitoring
MON_INTERVAL="${MON_INTERVAL:-5}"
ENABLE_MON="${ENABLE_MON:-1}"
MON_LOG="./_runs/monitor.log"

# ==============================
# CLI overrides
# ==============================
usage() {
  cat <<EOF
Usage: $0 [options]

Options:
  --num-envs N                (default: $NUM_ENVS)
  --vec {dummy|subproc}       (default: $VEC)

  --eval-k K                  (default: $EVAL_K)
  --grid-cell {2|4|8|16|32}           (default: $GRID_CELL)
  --lambda-area X             (default: $LAMBDA_AREA)
  --lambda-efficiency X       (default: $LAMBDA_EFFICIENCY)
  --efficiency-eps X          (default: $EFFICIENCY_EPS)
  --lambda-perceptual X       (default: $LAMBDA_PERCEPTUAL)
  --lambda-day X              (default: $LAMBDA_DAY)
  --area-target F             (default: $AREA_TARGET)
  --step-cost X                (default: $STEP_COST)
  --step-cost-after-target X   (default: $STEP_COST_AFTER_TARGET)
  --success-conf X            (default: $SUCCESS_CONF)
  --transform-strength X      (default: $TRANSFORM_STRENGTH)
  --paint NAME                (default: $PAINT)
  --paint-list LIST           (comma-separated)
  --cnn {custom|nature}       (default: $CNN)
  --cell-cover-thresh X       (default: $CELL_COVER_THRESH)
  --phase1-transform-strength X
  --phase2-transform-strength X
  --phase3-transform-strength X
  --phase1-lambda-day X
  --phase2-lambda-day X
  --phase3-lambda-day X
  --phase1-step-cost X
  --phase2-step-cost X
  --phase3-step-cost X
  --area-cap-frac F           (default: $AREA_CAP_FRAC)
  --area-cap-penalty P        (default: $AREA_CAP_PENALTY)
  --area-cap-mode {soft|hard} (default: $AREA_CAP_MODE)
  --area-cap-start F          (default: $AREA_CAP_START)
  --area-cap-end F            (default: $AREA_CAP_END)
  --area-cap-steps N          (default: $AREA_CAP_STEPS)
  --lambda-area-start X       (default: $LAMBDA_AREA_START)
  --lambda-area-end X         (default: $LAMBDA_AREA_END)
  --lambda-area-steps N       (default: $LAMBDA_AREA_STEPS)
  --ent-coef-start X          (default: $ENT_COEF_START)
  --ent-coef-end X            (default: $ENT_COEF_END)
  --ent-coef-steps N          (default: $ENT_COEF_STEPS)
  --obs-size N                (default: $OBS_SIZE)
  --obs-margin X              (default: $OBS_MARGIN)
  --obs-include-mask {0|1}    (default: $OBS_INCLUDE_MASK)

  --yolo-version {8|11}        (default: $YOLO_VERSION)
  --yolo-weights PATH          (default: $YOLO_WEIGHTS)
  --detector {yolo|torchvision|detr|rtdetr|rtdetrv2} (default: $DETECTOR)
  --detector-model NAME        (torchvision/transformers model id)
  --start-detector-server      (start local detector server)
  --detector-port P            (default: $DET_SERVER_PORT)
  --detector-device DEV        (default: $DET_SERVER_DEVICE)
  --detector-server-model PATH (default: $DET_SERVER_MODEL)

  --n-steps N                 (default: $N_STEPS)
  --batch N                   (default: $BATCH)
  --total-steps N             (default: $TOTAL_STEPS)
  --ent-coef X                (default: $ENT_COEF)
  --step-log-every N          (default: $STEP_LOG_EVERY)
  --step-log-keep N           (default: $STEP_LOG_KEEP)
  --step-log-500 N            (default: $STEP_LOG_500)

  --tb DIR                    (default: $TB_DIR)
  --ckpt DIR                  (default: $CKPT_DIR)
  --overlays DIR              (default: $OVR_DIR)
  --multiphase                (enable 3-phase curriculum)
  --resume                    (resume from latest checkpoint)
  --check-env                 (run SB3 env checker before training)
  --no-check-env              (skip SB3 env checker)

  --port P                    (default: $PORT)
  --mon-interval SEC          (default: $MON_INTERVAL)
  --no-monitor                (disable resource monitor)

  -h, --help
EOF
}

NUM_ENVS_SET=0
VEC_SET=0
N_STEPS_SET=0
BATCH_SET=0
TB_SET=0
CKPT_SET=0
OVR_SET=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --num-envs) NUM_ENVS="$2"; NUM_ENVS_SET=1; shift 2;;
    --vec) VEC="$2"; VEC_SET=1; shift 2;;

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
    --success-conf) SUCCESS_CONF="$2"; shift 2;;
    --transform-strength) TRANSFORM_STRENGTH="$2"; shift 2;;
    --paint) PAINT="$2"; shift 2;;
    --paint-list) PAINT_LIST="$2"; shift 2;;
    --cnn) CNN="$2"; shift 2;;
    --cell-cover-thresh) CELL_COVER_THRESH="$2"; shift 2;;
    --phase1-transform-strength) PHASE1_TRANSFORM_STRENGTH="$2"; shift 2;;
    --phase2-transform-strength) PHASE2_TRANSFORM_STRENGTH="$2"; shift 2;;
    --phase3-transform-strength) PHASE3_TRANSFORM_STRENGTH="$2"; shift 2;;
    --phase1-lambda-day) PHASE1_LAMBDA_DAY="$2"; shift 2;;
    --phase2-lambda-day) PHASE2_LAMBDA_DAY="$2"; shift 2;;
    --phase3-lambda-day) PHASE3_LAMBDA_DAY="$2"; shift 2;;
    --phase1-step-cost) PHASE1_STEP_COST="$2"; shift 2;;
    --phase2-step-cost) PHASE2_STEP_COST="$2"; shift 2;;
    --phase3-step-cost) PHASE3_STEP_COST="$2"; shift 2;;
    --area-cap-frac) AREA_CAP_FRAC="$2"; shift 2;;
    --area-cap-penalty) AREA_CAP_PENALTY="$2"; shift 2;;
    --area-cap-mode) AREA_CAP_MODE="$2"; shift 2;;
    --area-cap-start) AREA_CAP_START="$2"; shift 2;;
    --area-cap-end) AREA_CAP_END="$2"; shift 2;;
    --area-cap-steps) AREA_CAP_STEPS="$2"; shift 2;;
    --lambda-area-start) LAMBDA_AREA_START="$2"; shift 2;;
    --lambda-area-end) LAMBDA_AREA_END="$2"; shift 2;;
    --lambda-area-steps) LAMBDA_AREA_STEPS="$2"; shift 2;;
    --ent-coef-start) ENT_COEF_START="$2"; shift 2;;
    --ent-coef-end) ENT_COEF_END="$2"; shift 2;;
    --ent-coef-steps) ENT_COEF_STEPS="$2"; shift 2;;
    --obs-size) OBS_SIZE="$2"; shift 2;;
    --obs-margin) OBS_MARGIN="$2"; shift 2;;
    --obs-include-mask) OBS_INCLUDE_MASK="$2"; shift 2;;

    --yolo-version) YOLO_VERSION="$2"; shift 2;;
    --yolo-weights) YOLO_WEIGHTS="$2"; shift 2;;
    --detector) DETECTOR="$2"; shift 2;;
    --detector-model) DETECTOR_MODEL="$2"; DET_SERVER_MODEL="$2"; shift 2;;
    --start-detector-server) START_DET_SERVER="1"; shift 1;;
    --detector-port) DET_SERVER_PORT="$2"; shift 2;;
    --detector-device) DET_SERVER_DEVICE="$2"; shift 2;;
    --detector-server-model) DET_SERVER_MODEL="$2"; shift 2;;

    --n-steps) N_STEPS="$2"; N_STEPS_SET=1; shift 2;;
    --batch) BATCH="$2"; BATCH_SET=1; shift 2;;
    --total-steps) TOTAL_STEPS="$2"; shift 2;;
    --ent-coef) ENT_COEF="$2"; shift 2;;
    --step-log-every) STEP_LOG_EVERY="$2"; shift 2;;
    --step-log-keep) STEP_LOG_KEEP="$2"; shift 2;;
    --step-log-500) STEP_LOG_500="$2"; shift 2;;

    --tb) TB_DIR="$2"; TB_SET=1; shift 2;;
    --ckpt) CKPT_DIR="$2"; CKPT_SET=1; shift 2;;
    --overlays) OVR_DIR="$2"; OVR_SET=1; shift 2;;
    --multiphase) MULTIPHASE="1"; shift 1;;
    --resume) RESUME="1"; shift 1;;
    --check-env) CHECK_ENV="1"; shift 1;;
    --no-check-env) CHECK_ENV="0"; shift 1;;

    --port) PORT="$2"; shift 2;;
    --mon-interval) MON_INTERVAL="$2"; shift 2;;
    --no-monitor) ENABLE_MON="0"; shift 1;;

    -h|--help) usage; exit 0;;
    *) echo "Unknown option: $1"; usage; exit 1;;
  esac
done

# ==============================
# Run directories (avoid overwriting old runs)
# ==============================
TB_ROOT="${TB_DIR}"
CKPT_ROOT="${CKPT_DIR}"
OVR_ROOT="${OVR_DIR}"

if [[ "${RESUME}" == "1" && "${CKPT_SET}" -eq 0 ]]; then
  latest_ckpt_dir="$(ls -td "${CKPT_ROOT}"/*/ 2>/dev/null | head -n 1 || true)"
  if [[ -n "${latest_ckpt_dir}" ]]; then
    CKPT_DIR="${latest_ckpt_dir%/}"
    run_id="$(basename "${CKPT_DIR}")"
    [[ "${TB_SET}" -eq 0 ]] && TB_DIR="${TB_ROOT}/${run_id}"
    [[ "${OVR_SET}" -eq 0 ]] && OVR_DIR="${OVR_ROOT}/${run_id}"
  else
    echo "WARN: --resume set but no checkpoint folders found in ${CKPT_ROOT}. Starting new run."
  fi
fi

if [[ "${RESUME}" != "1" && "${CKPT_SET}" -eq 0 && "${TB_SET}" -eq 0 && "${OVR_SET}" -eq 0 ]]; then
  # Auto-increment per detector: yolo8_1, torchvision_ssd300_vgg16_1, detr_facebook-detr-resnet-50_1, ...
  if [[ "${DETECTOR}" == "yolo" ]]; then
    run_prefix="yolo${YOLO_VERSION}"
  else
    run_prefix="${DETECTOR}"
    if [[ -n "${DETECTOR_MODEL}" ]]; then
      safe_model="${DETECTOR_MODEL//\//-}"
      safe_model="${safe_model// /-}"
      safe_model="${safe_model//:/-}"
      run_prefix="${run_prefix}_${safe_model}"
    fi
  fi
  prefix="${run_prefix}_"
  max_id=0
  for d in "${CKPT_ROOT}/${prefix}"*; do
    [[ -d "${d}" ]] || continue
    base="$(basename "${d}")"
    num="${base##${prefix}}"
    if [[ "${num}" =~ ^[0-9]+$ ]]; then
      (( num > max_id )) && max_id="${num}"
    fi
  done
  run_id="${prefix}$((max_id + 1))"
  TB_DIR="${TB_ROOT}/${run_id}"
  CKPT_DIR="${CKPT_ROOT}/${run_id}"
  OVR_DIR="${OVR_ROOT}/${run_id}"
fi

mkdir -p "$TB_DIR" "$CKPT_DIR" "$OVR_DIR" ./_runs

# If using remote detector server, prefer multi-env defaults unless user overrides.
if [[ "${YOLO_DEVICE}" == server://* ]]; then
  [[ "${VEC_SET}" -eq 0 ]] && VEC="subproc"
  [[ "${NUM_ENVS_SET}" -eq 0 ]] && NUM_ENVS="4"
  [[ "${N_STEPS_SET}" -eq 0 ]] && N_STEPS="256"
  [[ "${BATCH_SET}" -eq 0 ]] && BATCH="1024"
fi

# ==============================
# Safety: CUDA + subproc is unstable for in-process YOLO inference
# ==============================
if [[ "${YOLO_DEVICE}" == cuda* ]] && [[ "${VEC}" == "subproc" ]]; then
  echo "WARN: CUDA YOLO + SubprocVecEnv is unstable in current design. Forcing --vec dummy."
  VEC="dummy"
fi

# PPO constraint: batch_size must be <= n_steps * num_envs
ROLLOUT=$(( N_STEPS * NUM_ENVS ))
if (( BATCH > ROLLOUT )); then
  echo "WARN: batch (${BATCH}) > rollout (${ROLLOUT}). Clamping batch -> ${ROLLOUT}."
  BATCH="${ROLLOUT}"
fi

# ==============================
# Helpers
# ==============================
TB_PID=""
MON_PID=""
DET_PID=""

cleanup() {
  echo ""
  echo "[CLEANUP] Stopping background services..."
  [[ -n "${TB_PID}" ]] && kill "$TB_PID" 2>/dev/null || true
  [[ -n "${MON_PID}" ]] && kill "$MON_PID" 2>/dev/null || true
  [[ -n "${DET_PID}" ]] && kill "$DET_PID" 2>/dev/null || true
}
trap cleanup EXIT

start_tensorboard() {
  echo "[TB] Starting TensorBoard on port ${PORT}, logdir=${TB_DIR}"
  tensorboard --logdir "${TB_DIR}" --host localhost --port "${PORT}" > ./_runs/tensorboard.log 2>&1 &
  TB_PID=$!
  sleep 2
  echo "[TB] PID=${TB_PID} | log: ./_runs/tensorboard.log"
  echo "[TB] Open: http://localhost:${PORT}"
}

start_monitor() {
  [[ "${ENABLE_MON}" != "1" ]] && { echo "[MON] Monitor disabled"; return; }
  echo "[MON] Starting resource monitor @ ${MON_INTERVAL}s -> ${MON_LOG}"
  {
    echo "=== Resource monitor started: $(date) ==="
    echo "interval=${MON_INTERVAL}s"
    echo ""
    has_cmd() { command -v "$1" >/dev/null 2>&1; }
    while true; do
      ts="$(date '+%F %T')"
      echo "----- ${ts} -----"
      if has_cmd nvidia-smi; then
        nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total \
          --format=csv,noheader,nounits | awk '{printf "[GPU] util=%s%% memUtil=%s%% mem=%s/%s MiB\n",$1,$2,$3,$4}'
      else
        echo "[GPU] nvidia-smi not found"
      fi
      if has_cmd free; then
        free -m | awk '/Mem:/ {printf "[RAM] used=%d MiB / total=%d MiB (%.1f%%)\n",$3,$2,($3/$2)*100}'
      fi
      echo ""
      sleep "${MON_INTERVAL}"
    done
  } >> "${MON_LOG}" 2>&1 &
  MON_PID=$!
  echo "[MON] PID=${MON_PID} | log: ${MON_LOG}"
}

# ==============================
# Optional: start local detector server
# ==============================
start_detector_server() {
  [[ "${START_DET_SERVER}" != "1" ]] && return
  if [[ -z "${DET_SERVER_MODEL}" ]]; then
    DET_SERVER_MODEL="${DETECTOR_MODEL}"
  fi
  if [[ -z "${DET_SERVER_MODEL}" && "${DETECTOR}" == "yolo" ]]; then
    DET_SERVER_MODEL="${YOLO_WEIGHTS}"
  fi
  if [[ "${DETECTOR}" == "yolo" && -z "${DET_SERVER_MODEL}" ]]; then
    echo "[DET] detector model not set. Use --detector-model or --yolo-weights."
    exit 1
  fi
  echo "[DET] Starting detector server on port ${DET_SERVER_PORT}"
  python tools/detector_server.py \
    --detector "${DETECTOR}" \
    --detector-model "${DET_SERVER_MODEL}" \
    --model "${DET_SERVER_MODEL}" \
    --device "${DET_SERVER_DEVICE}" \
    --port "${DET_SERVER_PORT}" \
    > ./_runs/detector_server.log 2>&1 &
  DET_PID=$!
  sleep 2
  echo "[DET] PID=${DET_PID} | log: ./_runs/detector_server.log"
  export YOLO_DEVICE="server://127.0.0.1:${DET_SERVER_PORT}"
}

EXTRA_ARGS=()
if [[ -n "${YOLO_WEIGHTS}" ]]; then
  EXTRA_ARGS+=(--yolo-weights "${YOLO_WEIGHTS}")
fi
if [[ -n "${DETECTOR_MODEL}" ]]; then
  EXTRA_ARGS+=(--detector-model "${DETECTOR_MODEL}")
fi
if [[ "${MULTIPHASE}" == "1" ]]; then
  EXTRA_ARGS+=(--multiphase)
fi
if [[ "${RESUME}" == "1" ]]; then
  EXTRA_ARGS+=(--resume)
fi
if [[ -n "${ENT_COEF_START}" ]]; then
  EXTRA_ARGS+=(--ent-coef-start "${ENT_COEF_START}")
fi
if [[ -n "${ENT_COEF_END}" ]]; then
  EXTRA_ARGS+=(--ent-coef-end "${ENT_COEF_END}")
fi
if [[ -n "${AREA_TARGET}" ]]; then
  EXTRA_ARGS+=(--area-target "${AREA_TARGET}")
fi
if [[ -n "${PHASE1_TRANSFORM_STRENGTH}" ]]; then
  EXTRA_ARGS+=(--phase1-transform-strength "${PHASE1_TRANSFORM_STRENGTH}")
fi
if [[ -n "${PHASE2_TRANSFORM_STRENGTH}" ]]; then
  EXTRA_ARGS+=(--phase2-transform-strength "${PHASE2_TRANSFORM_STRENGTH}")
fi
if [[ -n "${PHASE3_TRANSFORM_STRENGTH}" ]]; then
  EXTRA_ARGS+=(--phase3-transform-strength "${PHASE3_TRANSFORM_STRENGTH}")
fi
if [[ -n "${PHASE1_LAMBDA_DAY}" ]]; then
  EXTRA_ARGS+=(--phase1-lambda-day "${PHASE1_LAMBDA_DAY}")
fi
if [[ -n "${PHASE2_LAMBDA_DAY}" ]]; then
  EXTRA_ARGS+=(--phase2-lambda-day "${PHASE2_LAMBDA_DAY}")
fi
if [[ -n "${PHASE3_LAMBDA_DAY}" ]]; then
  EXTRA_ARGS+=(--phase3-lambda-day "${PHASE3_LAMBDA_DAY}")
fi
if [[ -n "${PHASE1_STEP_COST}" ]]; then
  EXTRA_ARGS+=(--phase1-step-cost "${PHASE1_STEP_COST}")
fi
if [[ -n "${PHASE2_STEP_COST}" ]]; then
  EXTRA_ARGS+=(--phase2-step-cost "${PHASE2_STEP_COST}")
fi
if [[ -n "${PHASE3_STEP_COST}" ]]; then
  EXTRA_ARGS+=(--phase3-step-cost "${PHASE3_STEP_COST}")
fi
if [[ -n "${PAINT_LIST}" ]]; then
  EXTRA_ARGS+=(--paint-list "${PAINT_LIST}")
fi
if [[ -n "${LAMBDA_AREA_START}" ]]; then
  EXTRA_ARGS+=(--lambda-area-start "${LAMBDA_AREA_START}")
fi
if [[ -n "${LAMBDA_AREA_END}" ]]; then
  EXTRA_ARGS+=(--lambda-area-end "${LAMBDA_AREA_END}")
fi
if [[ -n "${LAMBDA_AREA_STEPS}" && "${LAMBDA_AREA_STEPS}" -gt 0 ]]; then
  EXTRA_ARGS+=(--lambda-area-steps "${LAMBDA_AREA_STEPS}")
fi

if [[ "${CHECK_ENV}" == "1" ]]; then
  echo "[CHECK] Running SB3 env checker..."
    python "${PY_MAIN}" \
      --detector-device "${YOLO_DEVICE}" \
      --detector "${DETECTOR}" \
      --yolo-version "${YOLO_VERSION}" \
    --num-envs 1 \
    --vec dummy \
    --eval-K "${EVAL_K}" \
    --grid-cell "${GRID_CELL}" \
    --lambda-area "${LAMBDA_AREA}" \
    --lambda-efficiency "${LAMBDA_EFFICIENCY}" \
    --efficiency-eps "${EFFICIENCY_EPS}" \
    --lambda-perceptual "${LAMBDA_PERCEPTUAL}" \
    --lambda-day "${LAMBDA_DAY}" \
    --step-cost "${STEP_COST}" \
    --step-cost-after-target "${STEP_COST_AFTER_TARGET}" \
    --success-conf "${SUCCESS_CONF}" \
    --transform-strength "${TRANSFORM_STRENGTH}" \
    --paint "${PAINT}" \
    --cnn "${CNN}" \
    --cell-cover-thresh "${CELL_COVER_THRESH}" \
    --area-cap-frac "${AREA_CAP_FRAC}" \
    --area-cap-penalty "${AREA_CAP_PENALTY}" \
    --area-cap-mode "${AREA_CAP_MODE}" \
    --area-cap-start "${AREA_CAP_START}" \
    --area-cap-end "${AREA_CAP_END}" \
    --area-cap-steps "${AREA_CAP_STEPS}" \
    --obs-size "${OBS_SIZE}" \
    --obs-margin "${OBS_MARGIN}" \
    --obs-include-mask "${OBS_INCLUDE_MASK}" \
    --check-env \
    "${EXTRA_ARGS[@]}"
fi

# ==============================
# Start services
# ==============================
start_detector_server
start_tensorboard
start_monitor

# ==============================
# Run training
# ==============================
echo "[TRAIN] Launching GPU training:"
echo "        YOLO_DEVICE=${YOLO_DEVICE}"
echo "        yolo-version=${YOLO_VERSION} yolo-weights=${YOLO_WEIGHTS:-<default>}"
echo "        num-envs=${NUM_ENVS} vec=${VEC} eval_K=${EVAL_K} grid=${GRID_CELL}"
echo "        lambda-area=${LAMBDA_AREA} lambda-eff=${LAMBDA_EFFICIENCY} lambda-perc=${LAMBDA_PERCEPTUAL} lambda-day=${LAMBDA_DAY} step-cost=${STEP_COST} step-cost-after-target=${STEP_COST_AFTER_TARGET} area-target=${AREA_TARGET:-<cap>} success-conf=${SUCCESS_CONF} tf=${TRANSFORM_STRENGTH} paint=${PAINT} cnn=${CNN} area-cap-frac=${AREA_CAP_FRAC} area-cap-penalty=${AREA_CAP_PENALTY} mode=${AREA_CAP_MODE}"
echo "        cap-ramp=${AREA_CAP_START}->${AREA_CAP_END} over ${AREA_CAP_STEPS} steps"
if [[ -n "${LAMBDA_AREA_START}" || -n "${LAMBDA_AREA_END}" || ( -n "${LAMBDA_AREA_STEPS}" && "${LAMBDA_AREA_STEPS}" -gt 0 ) ]]; then
  echo "        lambda-ramp=${LAMBDA_AREA_START:-<unset>}->${LAMBDA_AREA_END:-<unset>} over ${LAMBDA_AREA_STEPS} steps"
else
  echo "        lambda-ramp=disabled"
fi
echo "        n-steps=${N_STEPS} batch=${BATCH} total-steps=${TOTAL_STEPS}"
echo "        tb=${TB_DIR} ckpt=${CKPT_DIR} overlays=${OVR_DIR}"
echo ""

python "${PY_MAIN}" \
  --detector-device "${YOLO_DEVICE}" \
  --detector "${DETECTOR}" \
  --yolo-version "${YOLO_VERSION}" \
  --num-envs "${NUM_ENVS}" \
  --vec "${VEC}" \
  --n-steps "${N_STEPS}" \
  --batch-size "${BATCH}" \
  --total-steps "${TOTAL_STEPS}" \
  --ent-coef "${ENT_COEF}" \
  --eval-K "${EVAL_K}" \
  --grid-cell "${GRID_CELL}" \
  --lambda-area "${LAMBDA_AREA}" \
  --lambda-efficiency "${LAMBDA_EFFICIENCY}" \
  --efficiency-eps "${EFFICIENCY_EPS}" \
  --lambda-perceptual "${LAMBDA_PERCEPTUAL}" \
  --lambda-day "${LAMBDA_DAY}" \
  --step-cost "${STEP_COST}" \
  --step-cost-after-target "${STEP_COST_AFTER_TARGET}" \
  --success-conf "${SUCCESS_CONF}" \
  --transform-strength "${TRANSFORM_STRENGTH}" \
  --paint "${PAINT}" \
  --cnn "${CNN}" \
  --cell-cover-thresh "${CELL_COVER_THRESH}" \
  --area-cap-frac "${AREA_CAP_FRAC}" \
  --area-cap-penalty "${AREA_CAP_PENALTY}" \
  --area-cap-mode "${AREA_CAP_MODE}" \
  --area-cap-start "${AREA_CAP_START}" \
  --area-cap-end "${AREA_CAP_END}" \
  --area-cap-steps "${AREA_CAP_STEPS}" \
  --ent-coef-steps "${ENT_COEF_STEPS}" \
  --obs-size "${OBS_SIZE}" \
  --obs-margin "${OBS_MARGIN}" \
  --obs-include-mask "${OBS_INCLUDE_MASK}" \
  --step-log-every "${STEP_LOG_EVERY}" \
  --step-log-keep "${STEP_LOG_KEEP}" \
  --step-log-500 "${STEP_LOG_500}" \
  --save-freq-updates "${SAVE_FREQ_UPDATES}" \
  --tb "${TB_DIR}" \
  --ckpt "${CKPT_DIR}" \
  --overlays "${OVR_DIR}" \
  "${EXTRA_ARGS[@]}"
