#!/usr/bin/env bash
set -euo pipefail

# ==============================
# Defaults (can be overridden)
# ==============================
MODE="${MODE:-both}"                    # attack | uv | both
NUM_ENVS="${NUM_ENVS:-8}"
N_STEPS="${N_STEPS:-512}"
BATCH="${BATCH:-128}"                   # NOTE: your trainer must read PPO_BATCH_SIZE env or accept --batch-size
VEC="${VEC:-subproc}"                   # dummy | subproc

TB_DIR="${TB_DIR:-./_runs/tb}"          # keep this outside OneDrive to avoid file locks
CKPT_DIR="${CKPT_DIR:-./_runs/checkpoints}"
OVR_DIR="${OVR_DIR:-./_runs/overlays}"

PORT="${PORT:-6006}"
NGROK_URL="${NGROK_URL:-https://curliest-ally-sobersidedly.ngrok-free.dev}"

SAVE_FREQ_UPDATES="${SAVE_FREQ_UPDATES:-2}"  # checkpoints every K PPO rollouts

PY_MAIN="${PY_MAIN:-train_single_stop_sign.py}"

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

    -h|--help) usage; exit 0;;
    *) echo "Unknown option: $1"; usage; exit 1;;
  esac
done

mkdir -p "$TB_DIR" "$CKPT_DIR" "$OVR_DIR" ./_runs

# ==============================
# Helpers
# ==============================
cleanup() {
  echo ""
  echo "[CLEANUP] Stopping background services..."
  [[ -n "${TB_PID:-}" ]] && kill "$TB_PID" 2>/dev/null || true
  [[ -n "${NGROK_PID:-}" ]] && kill "$NGROK_PID" 2>/dev/null || true
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
  # ngrok v3 prefers --url, older builds used --domain; try --url then fallback.
  if ngrok http --help 2>/dev/null | grep -q -- "--url"; then
    ngrok http --url="${NGROK_URL}" "${PORT}" > ./_runs/ngrok.log 2>&1 &
  else
    # try older flag
    ngrok http --domain="$(echo "${NGROK_URL}" | sed 's~https\?://~~')" "${PORT}" > ./_runs/ngrok.log 2>&1 &
  fi
  NGROK_PID=$!
  sleep 3
  echo "[NGROK] PID=${NGROK_PID} | log: ./_runs/ngrok.log"
  echo "[NGROK] If you see auth errors, run once: ngrok config add-authtoken <YOUR_TOKEN>"
}

# ==============================
# Start services
# ==============================
start_tensorboard
start_ngrok

# ==============================
# Run training (foreground)
# ==============================
echo "[TRAIN] Launching training:"
echo "        mode=${MODE} num-envs=${NUM_ENVS} vec=${VEC} n-steps=${N_STEPS} batch=${BATCH}"
echo "        tb=${TB_DIR} ckpt=${CKPT_DIR} overlays=${OVR_DIR} save-freq-updates=${SAVE_FREQ_UPDATES}"
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
