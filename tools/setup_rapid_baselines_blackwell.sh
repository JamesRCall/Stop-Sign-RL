#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

# Use a dedicated Linux environment; do not reuse the old CUDA 12.4 environment.
PYTHON_BIN="${PYTHON_BIN:-python3.11}"
VENV_PATH="${VENV_PATH:-.venv-blackwell-baselines}"
"${PYTHON_BIN}" -m venv "${VENV_PATH}"
PYTHON="${VENV_PATH}/bin/python"
"${PYTHON}" -m pip install --upgrade pip
# PyTorch 2.7+ / CUDA 12.8 provides Blackwell wheels, unlike the paper's old 2.6/cu124 environment.
"${PYTHON}" -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
"${PYTHON}" -m pip install -r requirements-rapid-baselines.txt
"${PYTHON}" - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable. Check the server driver and GPU visibility.")
print("GPU:", torch.cuda.get_device_name(0))
print("Capability:", torch.cuda.get_device_capability(0))
print("PyTorch:", torch.__version__, "CUDA:", torch.version.cuda)
x = torch.ones((32, 32), device="cuda")
print("CUDA kernel check:", (x @ x)[0, 0].item())
PY
echo "Ready. Run: bash tools/run_rapid_baselines_linux.sh"
