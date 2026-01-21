#!/usr/bin/env bash
set -euo pipefail

# ==============================
# GPU + allocator knobs
# ==============================
export YOLO_DEVICE="${YOLO_DEVICE:-cuda:0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:256}"

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
LAMBDA_AREA="${LAMBDA_AREA:-0.30}"
AREA_CAP_FRAC="${AREA_CAP_FRAC:-0.30}"
AREA_CAP_PENALTY="${AREA_CAP_PENALTY:--0.20}"
AREA_CAP_MODE="${AREA_CAP_MODE:-soft}"
AREA_CAP_START="${AREA_CAP_START:-0.80}"
AREA_CAP_END="${AREA_CAP_END:-0.30}"
AREA_CAP_STEPS="${AREA_CAP_STEPS:-500000}"
LAMBDA_AREA_START="${LAMBDA_AREA_START:-0.10}"
LAMBDA_AREA_END="${LAMBDA_AREA_END:-0.30}"
LAMBDA_AREA_STEPS="${LAMBDA_AREA_STEPS:-200000}"
OBS_SIZE="${OBS_SIZE:-224}"
OBS_MARGIN="${OBS_MARGIN:-0.10}"
OBS_INCLUDE_MASK="${OBS_INCLUDE_MASK:-1}"

YOLO_VERSION="${YOLO_VERSION:-8}"
YOLO_WEIGHTS="${YOLO_WEIGHTS:-}"
START_DET_SERVER="${START_DET_SERVER:-0}"
DET_SERVER_PORT="${DET_SERVER_PORT:-5009}"
DET_SERVER_DEVICE="${DET_SERVER_DEVICE:-cuda:0}"
DET_SERVER_MODEL="${DET_SERVER_MODEL:-}"

N_STEPS="${N_STEPS:-512}"
BATCH="${BATCH:-512}"              # default to rollout size for num_envs=1
TOTAL_STEPS="${TOTAL_STEPS:-800000}"
ENT_COEF="${ENT_COEF:-0.005}"
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
  --start-detector-server      (start local detector server)
  --detector-port P            (default: $DET_SERVER_PORT)
  --detector-device DEV        (default: $DET_SERVER_DEVICE)
  --detector-model PATH        (default: $DET_SERVER_MODEL)

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

while [[ $# -gt 0 ]]; do
  case "$1" in
    --num-envs) NUM_ENVS="$2"; NUM_ENVS_SET=1; shift 2;;
    --vec) VEC="$2"; VEC_SET=1; shift 2;;

    --eval-k) EVAL_K="$2"; shift 2;;
    --grid-cell) GRID_CELL="$2"; shift 2;;
    --lambda-area) LAMBDA_AREA="$2"; shift 2;;
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
    --start-detector-server) START_DET_SERVER="1"; shift 1;;
    --detector-port) DET_SERVER_PORT="$2"; shift 2;;
    --detector-device) DET_SERVER_DEVICE="$2"; shift 2;;
    --detector-model) DET_SERVER_MODEL="$2"; shift 2;;

    --n-steps) N_STEPS="$2"; N_STEPS_SET=1; shift 2;;
    --batch) BATCH="$2"; BATCH_SET=1; shift 2;;
    --total-steps) TOTAL_STEPS="$2"; shift 2;;
    --ent-coef) ENT_COEF="$2"; shift 2;;
    --step-log-every) STEP_LOG_EVERY="$2"; shift 2;;
    --step-log-keep) STEP_LOG_KEEP="$2"; shift 2;;
    --step-log-500) STEP_LOG_500="$2"; shift 2;;

    --tb) TB_DIR="$2"; shift 2;;
    --ckpt) CKPT_DIR="$2"; shift 2;;
    --overlays) OVR_DIR="$2"; shift 2;;
    --multiphase) MULTIPHASE="1"; shift 1;;

    --port) PORT="$2"; shift 2;;
    --mon-interval) MON_INTERVAL="$2"; shift 2;;
    --no-monitor) ENABLE_MON="0"; shift 1;;

    -h|--help) usage; exit 0;;
    *) echo "Unknown option: $1"; usage; exit 1;;
  esac
done

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
    DET_SERVER_MODEL="${YOLO_WEIGHTS}"
  fi
  if [[ -z "${DET_SERVER_MODEL}" ]]; then
    echo "[DET] detector model not set. Use --detector-model or --yolo-weights."
    exit 1
  fi
  echo "[DET] Starting detector server on port ${DET_SERVER_PORT}"
  python tools/detector_server.py \
    --model "${DET_SERVER_MODEL}" \
    --device "${DET_SERVER_DEVICE}" \
    --port "${DET_SERVER_PORT}" > ./_runs/detector_server.log 2>&1 &
  DET_PID=$!
  sleep 2
  echo "[DET] PID=${DET_PID} | log: ./_runs/detector_server.log"
  export YOLO_DEVICE="server://127.0.0.1:${DET_SERVER_PORT}"
}

# ==============================
# Start services
# ==============================
start_detector_server
start_tensorboard
start_monitor

# ==============================
# Run training
# ==============================
EXTRA_ARGS=()
if [[ -n "${YOLO_WEIGHTS}" ]]; then
  EXTRA_ARGS+=(--yolo-weights "${YOLO_WEIGHTS}")
fi
if [[ "${MULTIPHASE}" == "1" ]]; then
  EXTRA_ARGS+=(--multiphase)
fi
if [[ -n "${ENT_COEF_START}" ]]; then
  EXTRA_ARGS+=(--ent-coef-start "${ENT_COEF_START}")
fi
if [[ -n "${ENT_COEF_END}" ]]; then
  EXTRA_ARGS+=(--ent-coef-end "${ENT_COEF_END}")
fi

echo "[TRAIN] Launching GPU training:"
echo "        YOLO_DEVICE=${YOLO_DEVICE}"
echo "        yolo-version=${YOLO_VERSION} yolo-weights=${YOLO_WEIGHTS:-<default>}"
echo "        num-envs=${NUM_ENVS} vec=${VEC} eval_K=${EVAL_K} grid=${GRID_CELL}"
echo "        lambda-area=${LAMBDA_AREA} area-cap-frac=${AREA_CAP_FRAC} area-cap-penalty=${AREA_CAP_PENALTY} mode=${AREA_CAP_MODE}"
echo "        cap-ramp=${AREA_CAP_START}->${AREA_CAP_END} over ${AREA_CAP_STEPS} steps"
echo "        lambda-ramp=${LAMBDA_AREA_START}->${LAMBDA_AREA_END} over ${LAMBDA_AREA_STEPS} steps"
echo "        n-steps=${N_STEPS} batch=${BATCH} total-steps=${TOTAL_STEPS}"
echo "        tb=${TB_DIR} ckpt=${CKPT_DIR} overlays=${OVR_DIR}"
echo ""

python "${PY_MAIN}" \
  --detector-device "${YOLO_DEVICE}" \
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
  --area-cap-frac "${AREA_CAP_FRAC}" \
  --area-cap-penalty "${AREA_CAP_PENALTY}" \
  --area-cap-mode "${AREA_CAP_MODE}" \
  --area-cap-start "${AREA_CAP_START}" \
  --area-cap-end "${AREA_CAP_END}" \
  --area-cap-steps "${AREA_CAP_STEPS}" \
  --lambda-area-start "${LAMBDA_AREA_START}" \
  --lambda-area-end "${LAMBDA_AREA_END}" \
  --lambda-area-steps "${LAMBDA_AREA_STEPS}" \
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
