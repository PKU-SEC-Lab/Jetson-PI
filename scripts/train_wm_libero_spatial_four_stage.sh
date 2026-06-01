#!/usr/bin/env bash
# Four-stage Pi0.5 + World Model training on LIBERO-spatial (LeRobot export).
#
# Default recipe (recommended):
#   Stage1=30k  Stage2=15k  Stage3=55k  Stage4=0
#   Stage3: L_cond trains WM without reducer; L_act detaches WM mu (updates Pi0 AE + full LLM)
#   batch=16, action_encoder=transformer_block, handover H=10
#
# Usage:
#   export PI0_CHECKPOINT=PATH/TO/CHECKPOINT/pi05_libero
#   export OPENPI_LIBERO_LOCAL_DATASET_DIR=PATH/TO/DATASET/libero
#   export PY=PATH/TO/PYTHON
#   bash scripts/train_wm_libero_spatial_four_stage.sh
#
# Optional overrides: CUDA_VISIBLE_DEVICES, STAGE{1,2,3}_STEPS, BATCH_SIZE, EXP_NAME, NUM_WORKERS, ...
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

: "${PI0_CHECKPOINT:?Set PI0_CHECKPOINT (Pi0.5 LIBERO checkpoint dir, e.g. PATH/TO/CHECKPOINT/pi05_libero)}"
: "${OPENPI_LIBERO_LOCAL_DATASET_DIR:?Set OPENPI_LIBERO_LOCAL_DATASET_DIR (LeRobot libero root)}"
: "${PY:?Set PY (Python with JAX, e.g. PATH/TO/PYTHON)}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}"
export TRANSFORMERS_NO_TF="${TRANSFORMERS_NO_TF:-1}"
export USE_TF="${USE_TF:-0}"

export STAGE1_STEPS="${STAGE1_STEPS:-30000}"
export STAGE2_STEPS="${STAGE2_STEPS:-15000}"
export STAGE3_STEPS="${STAGE3_STEPS:-55000}"
export BATCH_SIZE="${BATCH_SIZE:-16}"
export GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-1}"
export NUM_WORKERS="${NUM_WORKERS:-4}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-1000}"
export LOG_INTERVAL="${LOG_INTERVAL:-100}"

RUN_TS="$(date +%Y%m%d_%H%M%S)"
export EXP_NAME="${EXP_NAME:-wm_four_stage_pi05_libero_spatial_s1_${STAGE1_STEPS}_s2_${STAGE2_STEPS}_s3_${STAGE3_STEPS}_bs${BATCH_SIZE}_${RUN_TS}}"
LOG="${WM_LOG_FILE:-${REPO_ROOT}/logs/${EXP_NAME}.log}"
mkdir -p "$(dirname "${LOG}")" "${REPO_ROOT}/checkpoints"

echo "LOG=${LOG}"
echo "EXP_NAME=${EXP_NAME}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "PI0_CHECKPOINT=${PI0_CHECKPOINT}"
echo "OPENPI_LIBERO_LOCAL_DATASET_DIR=${OPENPI_LIBERO_LOCAL_DATASET_DIR}"
echo "STAGE1=${STAGE1_STEPS} STAGE2=${STAGE2_STEPS} STAGE3=${STAGE3_STEPS} BATCH=${BATCH_SIZE}"

set -o pipefail
"${PY}" -u "${REPO_ROOT}/scripts/train_world_model_four_stage.py" \
  --data-config-name pi05_libero \
  --assets-base-dir "${OPENPI_LIBERO_LOCAL_DATASET_DIR}" \
  --checkpoint-base-dir "${REPO_ROOT}/checkpoints" \
  --exp-name "${EXP_NAME}" \
  --pi0-checkpoint "${PI0_CHECKPOINT}" \
  --libero-task-index-min 30 \
  --libero-task-index-max 40 \
  --stage1-steps "${STAGE1_STEPS}" \
  --stage2-steps "${STAGE2_STEPS}" \
  --stage3-steps "${STAGE3_STEPS}" \
  --stage4-steps 0 \
  --max-delta-t 10 \
  --handover-horizon-min 10 \
  --handover-horizon-max 10 \
  --token-reducer-kind learned_cross_attn \
  --action-encoder-kind transformer_block \
  --four-stage1-condition-source future_prefix \
  --four-stage1-prefix-source future_prefix \
  --lact-prefix-source future_prefix \
  --lambda-act 1.0 \
  --lambda-sg 0.1 \
  --four-stage3-lambda-cond 1.0 \
  --four-stage3-lcond-train-wm-no-reducer \
  --four-stage3-detach-wm-mu-for-lact \
  --wm-gru-hidden-dim 384 \
  --wm-gru-num-layers 3 \
  --wm-num-future-heads 8 \
  --wm-num-reducer-heads 8 \
  --batch-size "${BATCH_SIZE}" \
  --grad-accum-steps "${GRAD_ACCUM_STEPS}" \
  --num-workers "${NUM_WORKERS}" \
  --save-interval "${SAVE_INTERVAL}" \
  --log-interval "${LOG_INTERVAL}" \
  --seed 42 \
  --full-llm-trainable \
  --no-wandb-enabled \
  2>&1 | stdbuf -oL -eL tee -a "${LOG}"

exit "${PIPESTATUS[0]}"
