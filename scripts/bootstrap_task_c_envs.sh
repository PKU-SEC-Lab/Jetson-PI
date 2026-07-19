#!/usr/bin/env bash
# Build isolated Python 3.11 policy-server and LIBERO-client environments.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT="${TASK_C_ROOT:-/home/pinyarash/dev/pinyarash/jetson-pi-task-c}"
POLICY_ENV="${TASK_C_POLICY_ENV:-${ROOT}/envs/policy}"
LIBERO_ENV="${TASK_C_LIBERO_ENV:-${ROOT}/envs/libero}"

export PYTHONNOUSERSITE=1
export GIT_LFS_SKIP_SMUDGE=1

mkdir -p "${ROOT}/envs"
UV_PROJECT_ENVIRONMENT="${POLICY_ENV}" uv sync --frozen --no-dev --python 3.11
uv pip install --python "${POLICY_ENV}/bin/python" 'pytest>=8,<9'

TRANSFORMERS_DIR="$("${POLICY_ENV}/bin/python" -c 'import pathlib, transformers; print(pathlib.Path(transformers.__file__).parent)')"
cp -r "${REPO}/src/openpi/models_pytorch/transformers_replace/." "${TRANSFORMERS_DIR}/"

uv venv --python 3.11 "${LIBERO_ENV}"
uv pip install --python "${LIBERO_ENV}/bin/python" \
  'numpy==1.26.4' \
  'imageio[ffmpeg]==2.37.0' \
  'opencv-python==4.6.0.66' \
  'pillow>=10,<12' \
  'tqdm>=4.67,<5' \
  'tyro>=0.9,<1' \
  'pyyaml>=6,<7' \
  'easydict==1.9' \
  'bddl==1.0.1' \
  'cloudpickle==2.1.0' \
  'future==0.18.2' \
  'gym==0.25.2' \
  'matplotlib==3.5.3' \
  'mujoco==2.3.2' \
  'robosuite==1.4.1'
uv pip install --python "${LIBERO_ENV}/bin/python" \
  --index-url https://download.pytorch.org/whl/cpu \
  'torch==2.5.1'
uv pip install --python "${LIBERO_ENV}/bin/python" -e "${REPO}/packages/openpi-client"
uv pip install --python "${LIBERO_ENV}/bin/python" --no-deps -e "${REPO}/third_party/libero" \
  --config-settings editable_mode=compat

"${POLICY_ENV}/bin/python" -c \
  'import jax; assert jax.__version__ == "0.5.3"; print("policy", jax.__version__, jax.devices())'
MUJOCO_GL=egl LIBERO_CONFIG_PATH="${ROOT}/libero-config" \
  "${LIBERO_ENV}/bin/python" -c \
  'import numpy, torch, mujoco, robosuite; print("libero", numpy.__version__, torch.__version__, mujoco.__version__, robosuite.__version__)'
