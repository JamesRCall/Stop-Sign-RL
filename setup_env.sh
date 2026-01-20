#!/usr/bin/env bash
# Create a venv and install requirements.
set -euo pipefail

VENV_DIR="${VENV_DIR:-.venv}"
PYTHON_BIN="${PYTHON_BIN:-}"

if [[ -n "${PYTHON_BIN}" ]]; then
  PYTHON="${PYTHON_BIN}"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON="python3"
elif command -v python >/dev/null 2>&1; then
  PYTHON="python"
elif command -v py >/dev/null 2>&1; then
  PYTHON="py"
else
  echo "Python not found. Install Python or set PYTHON_BIN."
  exit 1
fi

if [[ ! -d "${VENV_DIR}" ]]; then
  "${PYTHON}" -m venv "${VENV_DIR}"
fi

if [[ -x "${VENV_DIR}/bin/python" ]]; then
  VENV_PY="${VENV_DIR}/bin/python"
elif [[ -x "${VENV_DIR}/Scripts/python.exe" ]]; then
  VENV_PY="${VENV_DIR}/Scripts/python.exe"
elif [[ -x "${VENV_DIR}/Scripts/python" ]]; then
  VENV_PY="${VENV_DIR}/Scripts/python"
else
  echo "Venv python not found in ${VENV_DIR}. Recreate the venv or set VENV_DIR."
  exit 1
fi

"${VENV_PY}" -m pip install --upgrade pip
"${VENV_PY}" -m pip install -r requirements.txt

echo "Environment ready."
echo "Activate (bash): source ${VENV_DIR}/bin/activate"
echo "Activate (Windows bash): source ${VENV_DIR}/Scripts/activate"
