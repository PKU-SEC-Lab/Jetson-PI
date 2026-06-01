#!/usr/bin/env bash
# LIBERO-spatial evaluation with Pi0.5 + trained future correction module (FAAC).
#
# Starts serve_policy.py, runs examples/libero/main.py, writes logs/videos under logs/.
# Requires checkpoints/<EXP_NAME>/world_model_step_<STEP>/.
#
# Usage:
#   export PI0_CHECKPOINT=PATH/TO/CHECKPOINT/pi05_libero
#   export PY_SERVER=PATH/TO/PYTHON
#   export WM=PATH/TO/future-correction-module
#   bash scripts/eval_wm_libero_spatial.sh
#
# Or derive WM from EXP_NAME + STEP:
#   export EXP_NAME=your_training_exp_name
#   export STEP=<N>
#   bash scripts/eval_wm_libero_spatial.sh
#
# Adaptive multi-rollout (confidence-based scheduling):
#   export LIBERO_WM_EVAL_ADAPTIVE_KAPPA=1
#   export LIBERO_WM_EVAL_KAPPA_DELTA=0.4
#   bash scripts/eval_wm_libero_spatial.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

: "${PI0_CHECKPOINT:?Set PI0_CHECKPOINT (Pi0.5 LIBERO checkpoint, e.g. PATH/TO/CHECKPOINT/pi05_libero)}"
: "${PY_SERVER:?Set PY_SERVER (Python with JAX for serve_policy)}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PORT="${PORT:-8000}"

export LIBERO_WM_EVAL_TASK_SUITE="${LIBERO_WM_EVAL_TASK_SUITE:-libero_spatial}"
export LIBERO_WM_EVAL_NUM_TRIALS="${LIBERO_WM_EVAL_NUM_TRIALS:-50}"
export LIBERO_WM_EVAL_NO_WM=0

# Async chunking: H=10, trigger at K, overlap = H-K
export AH="${AH:-10}"
export K="${K:-9}"
export OVERLAP="${OVERLAP:-$((AH - K))}"

if [[ -z "${WM:-}" ]]; then
  : "${EXP_NAME:?Set WM (e.g. PATH/TO/future-correction-module) or EXP_NAME + STEP}"
  : "${STEP:?Set WM or EXP_NAME + STEP}"
  CKPT_ROOT="${CKPT_ROOT:-${REPO_ROOT}/checkpoints/${EXP_NAME}}"
  export WM="${CKPT_ROOT}/world_model_step_${STEP}"
else
  export WM
fi

if [[ -z "${EXP_NAME:-}" ]]; then
  export EXP_NAME="$(basename "$(dirname "${WM}")")"
fi

if [[ -z "${STEP:-}" ]]; then
  WM_BASENAME="$(basename "${WM}")"
  if [[ "${WM_BASENAME}" =~ ^world_model_step_([0-9]+)$ ]]; then
    export STEP="${BASH_REMATCH[1]}"
  else
    export STEP=unknown
  fi
fi

export POLICY_CKPT_DIR="${POLICY_CKPT_DIR:-${PI0_CHECKPOINT}}"
export PI0_NORM_CHECKPOINT_DIR="${PI0_NORM_CHECKPOINT_DIR:-${PI0_CHECKPOINT}}"

if [[ ! -d "${WM}/params" ]]; then
  echo "Missing future correction module checkpoint: ${WM}/params" >&2
  echo "Set WM to the future correction module dir (e.g. PATH/TO/future-correction-module; must contain params/)." >&2
  exit 1
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
export LIBERO_WM_EVAL_OUT_ROOT="${LIBERO_WM_EVAL_OUT_ROOT:-${REPO_ROOT}/logs/${LIBERO_WM_EVAL_TASK_SUITE}_${EXP_NAME}_wm${STEP}_h${AH}_k${K}_t${LIBERO_WM_EVAL_NUM_TRIALS}_${STAMP}}"

if [[ "${LIBERO_WM_EVAL_ADAPTIVE_KAPPA:-0}" == "1" ]]; then
  KAPPA_DELTA="${LIBERO_WM_EVAL_KAPPA_DELTA:-0.4}"
  DELTA="${LIBERO_WM_EVAL_DELTA:-$(( (OVERLAP + 3 - 1) / 3 ))}"
  export LIBERO_WM_EVAL_EXTRA_TYRO="--async-wm-multi-rollout --async-wm-multi-rollout-adaptive-kappa --async-wm-multi-rollout-adaptive-kappa-low-replan --async-wm-rollout-delta-t ${DELTA}.0 --async-wm-multi-rollout-kappa-delta=${KAPPA_DELTA} --wm-confidence-jsonl ${LIBERO_WM_EVAL_OUT_ROOT}/wm_confidence.jsonl"
fi

exec bash "${REPO_ROOT}/scripts/libero_wm_eval_spatial_bundle_step_one.sh"
