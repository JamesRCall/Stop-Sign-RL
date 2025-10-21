#!/usr/bin/env bash
set -euo pipefail

# ==============================
# Defaults (can be overridden)
# ==============================
MODE="${MODE:-both}"                    # attack | uv | both
NUM_ENVS="${NUM_ENVS:-32}"
N_STEPS="${N_STEPS:-512}"
BATCH="${BATCH:-4096}"                  # trainer can read PPO_BATCH_SIZE or --batch-size
VEC="${VEC:-subproc}"                   # dummy | subproc

TB_DIR="${TB_DIR:-./_runs/tb}"          # keep this outside OneDrive to avoid file locks
CKPT_DIR="${CKPT_DIR:-./_runs/checkpoints}"
OVR_DIR="${OVR_DIR:-./_runs/overlays}"

PORT="${PORT:-6006}"
NGROK_URL="${NGROK_URL:-https://curliest-ally-sobersidedly.ngrok-free.dev}"

SAVE_FREQ_UPDATES="${SAVE_FREQ_UPDATES:-2}"  # checkpoints every K PPO rollouts
PY_MAIN="${PY_MAIN:-train_single_stop_sign.py}"

# Monitoring
MON_INTERVAL="${MON_INTERVAL:-5}"       # seconds between samples
ENABLE_MON="${ENABLE_MON:-1}"           # 1=on, 0=off
MON_LOG="./_runs/monitor.log"

# ==============================
# CLI overrides
# ==============================
usage() {
  cat <<EOF
Usage: $0 [options]

Options:
  --mode {attack|uv|both}     (default: $MODE)
  --num-envs N                (default: $NUM_ENVS)
  --n-steps N                 (default: $N_STEPS)
  --batch N                   (default: $BATCH)  # requires trainer to read PPO_BATCH_SIZE or --batch-size
  --vec {dummy|subproc}       (default: $VEC)

  --tb DIR                    (default: $TB_DIR)
  --ckpt DIR                  (default: $CKPT_DIR)
  --overlays DIR              (default: $OVR_DIR)

  --port P                    (default: $PORT)
  --ngrok-url URL             (default: $NGROK_URL)

  --mon-interval SEC          (default: $MON_INTERVAL)
  --no-monitor                (disable resource monitor)

  -h, --help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode) MODE="$2"; shift 2;;
    --num-envs) NUM_ENVS="$2"; shift 2;;
    --n-steps) N_STEPS="$2"; shift 2;;
    --batch) BATCH="$2"; shift 2;;
    --vec) VEC="$2"; shift 2;;

    --tb) TB_DIR="$2"; shift 2;;
    --ckpt) CKPT_DIR="$2"; shift 2;;
    --overlays) OVR_DIR="$2"; shift 2;;

    --port) PORT="$2"; shift 2;;
    --ngrok-url) NGROK_URL="$2"; shift 2;;

    --mon-interval) MON_INTERVAL="$2"; shift 2;;
    --no-monitor) ENABLE_MON="0"; shift 1;;

    -h|--help) usage; exit 0;;
    *) echo "Unknown option: $1"; usage; exit 1;;
  esac
done

mkdir -p "$TB_DIR" "$CKPT_DIR" "$OVR_DIR" ./_runs

# ==============================
# Helpers
# ==============================
TB_PID=""
NGROK_PID=""
MON_PID=""

cleanup() {
  echo ""
  echo "[CLEANUP] Stopping background services..."
  [[ -n "${TB_PID}" ]] && kill "$TB_PID" 2>/dev/null || true
  [[ -n "${NGROK_PID}" ]] && kill "$NGROK_PID" 2>/dev/null || true
  [[ -n "${MON_PID}" ]] && kill "$MON_PID" 2>/dev/null || true
}
trap cleanup EXIT

start_tensorboard() {
  echo "[TB] Starting TensorBoard on port ${PORT}, logdir=${TB_DIR}"
  tensorboard --logdir "${TB_DIR}" --host 0.0.0.0 --port "${PORT}" > ./_runs/tensorboard.log 2>&1 &
  TB_PID=$!
  sleep 2
  echo "[TB] PID=${TB_PID} | log: ./_runs/tensorboard.log"
}

start_ngrok() {
  echo "[NGROK] Starting tunnel for TensorBoard (${PORT}) → ${NGROK_URL}"
  if ngrok http --help 2>/dev/null | grep -q -- "--url"; then
    ngrok http --url="${NGROK_URL}" "${PORT}" > ./_runs/ngrok.log 2>&1 &
  else
    ngrok http --domain="$(echo "${NGROK_URL}" | sed 's~https\?://~~')" "${PORT}" > ./_runs/ngrok.log 2>&1 &
  fi
  NGROK_PID=$!
  sleep 3
  echo "[NGROK] PID=${NGROK_PID} | log: ./_runs/ngrok.log"
  echo "[NGROK] If you see auth errors, run once: ngrok config add-authtoken <YOUR_TOKEN>"
}

start_monitor() {
  [[ "${ENABLE_MON}" != "1" ]] && { echo "[MON] Monitor disabled"; return; }
  echo "[MON] Starting resource monitor @ ${MON_INTERVAL}s → ${MON_LOG}"
  {
    echo "=== Resource monitor started: $(date) ==="
    echo "interval=${MON_INTERVAL}s"
    echo ""

    has_cmd() { command -v "$1" >/dev/null 2>&1; }

    while true; do
      ts="$(date '+%F %T')"
      echo "----- ${ts} -----"

      # GPU + VRAM (nvidia-smi)
      if has_cmd nvidia-smi; then
        echo "[GPU] nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits"
        nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits | awk '{printf "GPU util=%s%%, mem=%s/%s MiB\n",$1,$2,$3}'
      else
        echo "[GPU] nvidia-smi not found"
      fi

      # CPU
      if has_cmd mpstat; then
        mpstat 1 1 | awk '/all/ {printf "[CPU] usr=%.1f%% sys=%.1f%% idle=%.1f%%\n",$3,$5,$12}'
      elif has_cmd top; then
        top -b -n1 | awk -F',' '/Cpu\(s\)/{print "[CPU]" $0}'
      else
        echo "[CPU] mpstat/top not found"
      fi

      # RAM
      if has_cmd free; then
        free -m | awk '/Mem:/ {printf "[RAM] used=%d MiB / total=%d MiB (%.1f%%)\n",$3,$2,($3/$2)*100}'
      else
        echo "[RAM] free not found"
      fi

      echo ""
      sleep "${MON_INTERVAL}"
    done
  } >> "${MON_LOG}" 2>&1 &
  MON_PID=$!
  echo "[MON] PID=${MON_PID} | log: ${MON_LOG}"
}

# ==============================
# Start services
# ==============================
start_tensorboard
start_ngrok
start_monitor

# ==============================
# Run training (foreground)
# ==============================
echo "[TRAIN] Launching training:"
echo "        mode=${MODE} num-envs=${NUM_ENVS} vec=${VEC} n-steps=${N_STEPS} batch=${BATCH}"
echo "        tb=${TB_DIR} ckpt=${CKPT_DIR} overlays=${OVR_DIR} save-freq-updates=${SAVE_FREQ_UPDATES}"
echo "        monitor: interval=${MON_INTERVAL}s → ${MON_LOG}"
echo ""

# If your trainer supports --batch-size, append it. If not, it will be ignored.
# Also export PPO_BATCH_SIZE for trainers that read it from env.
export PPO_BATCH_SIZE="${BATCH}"

python "${PY_MAIN}" \
  --mode "${MODE}" \
  --num-envs "${NUM_ENVS}" \
  --vec "${VEC}" \
  --n-steps "${N_STEPS}" \
  --save-freq-updates "${SAVE_FREQ_UPDATES}" \
  --tb "${TB_DIR}" \
  --ckpt "${CKPT_DIR}" \
  --overlays "${OVR_DIR}"

# NOTE: if your train_single_stop_sign.py accepts --batch-size, add:
#   --batch-size "${BATCH}"
# And make sure the parser passes it into PPO(batch_size=...).
