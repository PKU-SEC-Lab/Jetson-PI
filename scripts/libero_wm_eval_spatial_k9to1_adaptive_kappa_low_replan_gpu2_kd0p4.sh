#!/usr/bin/env bash
# K-sweep (9 -> 1) with adaptive kappa + low replan on libero_spatial.
#
#   export WM=PATH/TO/future-correction-module
#   export PI0_CHECKPOINT=PATH/TO/CHECKPOINT/pi05_libero
#   export PY_SERVER=PATH/TO/PYTHON
#   export LIBERO_WM_EVAL_KAPPA_DELTA=0.4
#   bash scripts/libero_wm_eval_spatial_k9to1_adaptive_kappa_low_replan_gpu2_kd0p4.sh
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

BUNDLE="${REPO}/scripts/libero_wm_eval_spatial_bundle_step_one.sh"
if [[ ! -f "${BUNDLE}" ]]; then
  echo "FATAL: missing ${BUNDLE}" >&2
  exit 1
fi

export PYTHONUNBUFFERED=1
export PYTHONPATH="${REPO}/src${PYTHONPATH:+:$PYTHONPATH}"

: "${WM:?Set WM (e.g. PATH/TO/future-correction-module)}"
export POLICY_CKPT_DIR="${POLICY_CKPT_DIR:-${PI0_CHECKPOINT:-PATH/TO/CHECKPOINT/pi05_libero}}"

export LIBERO_WM_EVAL_TASK_SUITE="${LIBERO_WM_EVAL_TASK_SUITE:-libero_spatial}"
H=10
NUM_TRIALS="${LIBERO_WM_EVAL_NUM_TRIALS:-50}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
FIRST_PORT="${FIRST_PORT:-8100}"
KAPPA_DELTA="${LIBERO_WM_EVAL_KAPPA_DELTA:-0.4}"

_feasible_adaptive() {
  local H_="$1" O_="$2" d_="$3"
  if ((O_ < 1 || d_ < 1 || O_ >= H_)); then
    return 1
  fi
  local n1=$(( (H_ - O_) / d_ + 1 ))
  local n2=999999
  if ((H_ >= 2 * O_)); then
    n2=$(( (H_ - 2 * O_) / d_ + 1 ))
  fi
  local mc=$(( (H_ - O_ - 1) / d_ + 1 ))
  ((mc < 1)) && mc=1
  local mr=$n1
  ((n2 < mr)) && mr=$n2
  ((mc < mr)) && mr=$mc
  ((mr < 1)) && return 1
  if ((O_ + (mr - 1) * d_ > H_)); then
    return 1
  fi
  if ((2 * O_ + (mr - 1) * d_ > H_)); then
    return 1
  fi
  return 0
}

SWEEP_STAMP="${SWEEP_STAMP:-$(date +%Y%m%d_%H%M%S)}"

for K in $(seq 9 -1 1); do
  O=$((H - K))
  DELTA=$(( (O + 3 - 1) / 3 ))
  if ! _feasible_adaptive "${H}" "${O}" "${DELTA}"; then
    echo "[skip] K=${K} overlap=${O} delta_idx=${DELTA} infeasible for adaptive WM+AE"
    continue
  fi
  STAMP=$(date +%Y%m%d_%H%M%S)
  PORT=$((FIRST_PORT + K))
  OUT="${REPO}/logs/${LIBERO_WM_EVAL_TASK_SUITE}_k_sweep_kd${KAPPA_DELTA}_h${H}_k${K}_o${O}_d${DELTA}_t${NUM_TRIALS}_gpu${CUDA_VISIBLE_DEVICES}_${SWEEP_STAMP}_${STAMP}"
  mkdir -p "${OUT}"
  {
    echo "mode=wm_multi_adaptive_kappa_low_replan"
    echo "WM=${WM}"
    echo "KAPPA_DELTA=${KAPPA_DELTA}"
  } >>"${OUT}/sweep_case.txt"

  export LIBERO_WM_EVAL_EXTRA_TYRO="--async-wm-multi-rollout --async-wm-multi-rollout-adaptive-kappa --async-wm-multi-rollout-adaptive-kappa-low-replan --async-wm-rollout-delta-t ${DELTA}.0 --async-wm-multi-rollout-kappa-delta=${KAPPA_DELTA} --wm-confidence-jsonl ${OUT}/wm_confidence.jsonl"

  echo "======== ${LIBERO_WM_EVAL_TASK_SUITE} K=${K} o=${O} d=${DELTA} kd=${KAPPA_DELTA} port=${PORT} OUT=${OUT}"
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
    WM="${WM}" \
    POLICY_CKPT_DIR="${POLICY_CKPT_DIR}" \
    AH="${H}" K="${K}" OVERLAP="${O}" \
    LIBERO_WM_EVAL_NUM_TRIALS="${NUM_TRIALS}" \
    LIBERO_WM_EVAL_TASK_SUITE="${LIBERO_WM_EVAL_TASK_SUITE}" \
    PORT="${PORT}" \
    LIBERO_WM_EVAL_OUT_ROOT="${OUT}" \
    bash "${BUNDLE}" \
    || echo "[warn] K=${K} exited non-zero; see ${OUT}"
done

echo "k_sweep_done sweep=${SWEEP_STAMP}"
