#!/usr/bin/env bash
# Build the dedicated `wildlife` conda env for detector training + tracking.
# Runs on the login node (network/IO bound — no GPU compute here).
# Env lives alongside acidvlm/cosmo under the user envs prefix.
set -euo pipefail

ENV_PREFIX=/work/nvme/bgte/tislam6/envs/wildlife

echo "[setup_env] loading miniforge module"
source /etc/profile.d/modules.sh 2>/dev/null || true
module load miniforge3-python 2>/dev/null || module load anaconda3_gpu 2>/dev/null || true

# Locate a conda we can drive non-interactively.
CONDA_BIN="$(command -v conda || true)"
if [ -z "${CONDA_BIN}" ]; then
  CONDA_BIN=/sw/external/python/mambaforge/bin/conda
fi
echo "[setup_env] using conda: ${CONDA_BIN}"

if [ ! -d "${ENV_PREFIX}" ]; then
  echo "[setup_env] creating env at ${ENV_PREFIX}"
  "${CONDA_BIN}" create -y -p "${ENV_PREFIX}" python=3.11
fi

# pip from inside the env
PIP="${ENV_PREFIX}/bin/pip"
echo "[setup_env] torch (CUDA 12.1 wheels for A100)"
"${PIP}" install --no-input torch torchvision --index-url https://download.pytorch.org/whl/cu121

echo "[setup_env] ultralytics + opencv + utils"
"${PIP}" install --no-input "ultralytics>=8.3" "opencv-python-headless>=4.8" numpy pandas matplotlib python-docx

echo "[setup_env] versions:"
"${ENV_PREFIX}/bin/python" - <<'PY'
import torch, ultralytics, cv2, numpy
print("python  ", __import__("sys").version.split()[0])
print("torch   ", torch.__version__)
print("ultralytics", ultralytics.__version__)
print("cv2     ", cv2.__version__)
print("cuda_build", torch.version.cuda, "(cuda runtime check happens on a GPU node)")
PY
echo "[setup_env] DONE -> ${ENV_PREFIX}"
