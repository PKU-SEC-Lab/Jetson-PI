#!/usr/bin/env bash
# Single LIBERO eval run: serve_policy.py + examples/libero/main.py.
# Prefer scripts/eval_wm_libero_spatial.sh; this script is the low-level bundle.
#
#   export WM=PATH/TO/future-correction-module
#   export PI0_CHECKPOINT=PATH/TO/CHECKPOINT/pi05_libero
#   export PY_SERVER=PATH/TO/PYTHON
#   bash scripts/libero_wm_eval_spatial_bundle_step_one.sh
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -n "${LIBERO_WM_EVAL_EXTRA_TYRO:-}" ]]; then
  # shellcheck disable=SC2206
  EXTRA_MAIN_FLAGS=(${LIBERO_WM_EVAL_EXTRA_TYRO})
else
  EXTRA_MAIN_FLAGS=()
fi
cd "$REPO"

export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"

if [[ -z "${WM:-}" ]]; then
  : "${EXP_NAME:?Set WM or EXP_NAME + STEP}"
  : "${STEP:?Set WM or EXP_NAME + STEP}"
  CKPT_ROOT="${CKPT_ROOT:-$REPO/checkpoints/${EXP_NAME}}"
  WM="${CKPT_ROOT}/world_model_step_${STEP}"
fi
export WM

if [[ -z "${EXP_NAME:-}" ]]; then
  EXP_NAME="$(basename "$(dirname "${WM}")")"
fi
STEP="${STEP:-unknown}"
if [[ "${STEP}" == "unknown" ]]; then
  WM_BASENAME="$(basename "${WM}")"
  if [[ "${WM_BASENAME}" =~ ^world_model_step_([0-9]+)$ ]]; then
    STEP="${BASH_REMATCH[1]}"
  fi
fi

CKPT_ROOT="${CKPT_ROOT:-$REPO/checkpoints/${EXP_NAME}}"
ORBAX_STEP="${ORBAX_STEP:-${STEP}}"
POLICY_CKPT_DIR="${POLICY_CKPT_DIR:-PATH/TO/CHECKPOINT/pi05_libero}"

AH="${AH:-10}"
K="${K:-9}"
OVERLAP="${OVERLAP:-1}"
NUM_TRIALS="${LIBERO_WM_EVAL_NUM_TRIALS:-50}"
PORT="${PORT:-8000}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
TASK_SUITE="${LIBERO_WM_EVAL_TASK_SUITE:-libero_spatial}"
NO_WM="${LIBERO_WM_EVAL_NO_WM:-0}"
OUT_ROOT="${LIBERO_WM_EVAL_OUT_ROOT:-$REPO/logs/${TASK_SUITE}_${EXP_NAME}_wm${STEP}_h${AH}_k${K}_o${OVERLAP}_${STAMP}}"

PY="${PY:-$REPO/examples/libero/.venv/bin/python}"
PY_SERVER="${PY_SERVER:-PATH/TO/PYTHON}"
if ! test -x "$PY"; then PY="$PY_SERVER"; fi

export PYTHONPATH="${REPO}/packages/openpi-client/src:${REPO}/src:${REPO}/third_party/libero${PYTHONPATH:+:$PYTHONPATH}"
XVFB_OPTS="${XVFB_OPTS:--screen 0 1024x768x24}"
SERVER_SHUTDOWN_TIMEOUT_SEC="${SERVER_SHUTDOWN_TIMEOUT_SEC:-45}"

LIBERO_CFG_DIR="${LIBERO_CONFIG_PATH:-$REPO/.libero_eval}"
mkdir -p "${LIBERO_CFG_DIR}"
cat > "${LIBERO_CFG_DIR}/config.yaml" <<EOF
assets: ${REPO}/third_party/libero/libero/libero/assets
bddl_files: ${REPO}/third_party/libero/libero/libero/bddl_files
benchmark_root: ${REPO}/third_party/libero/libero/libero
datasets: ${REPO}/third_party/libero/libero/datasets
init_states: ${REPO}/third_party/libero/libero/libero/init_files
EOF
export LIBERO_CONFIG_PATH="${LIBERO_CFG_DIR}"

if ! test -d "${POLICY_CKPT_DIR}/params"; then
  echo "Missing policy checkpoint params: ${POLICY_CKPT_DIR}/params" >&2
  exit 1
fi
if [[ "${NO_WM}" != "1" ]] && ! test -d "${WM}/params"; then
  echo "Missing WM params: ${WM}/params" >&2
  exit 1
fi

mkdir -p "${OUT_ROOT}/videos"
LOG_S="${OUT_ROOT}/serve.log"
LOG_C="${OUT_ROOT}/client.log"
META="${OUT_ROOT}/run_meta.txt"
# Pin both ``serve_policy`` and ``main.py`` to the same device(s). Callers set ``CUDA_VISIBLE_DEVICES`` (e.g. ``2`` for a single physical GPU).
EVAL_CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

stop_server() {
  local spid="$1"
  local port="${2:-}"
  [[ -z "${spid}" ]] && return 0
  kill "${spid}" 2>/dev/null || true
  local _j _max="${SERVER_SHUTDOWN_TIMEOUT_SEC}"
  for ((_j = 0; _j < _max; _j++)); do
    if ! kill -0 "${spid}" 2>/dev/null; then
      wait "${spid}" 2>/dev/null || true
      break
    fi
    sleep 1
  done
  kill -9 "${spid}" 2>/dev/null || true
  wait "${spid}" 2>/dev/null || true
  if [[ -n "${port}" ]] && command -v fuser >/dev/null 2>&1; then
    fuser -k "${port}/tcp" 2>/dev/null || true
    sleep 1
  fi
}

wait_serve_ready() {
  local p="$1" max="${2:-900}" _i
  for ((_i = 1; _i <= max; _i++)); do
    if ! kill -0 "${SPID}" 2>/dev/null; then
      echo "serve_policy exited early (pid ${SPID}); see ${LOG_S}" | tee -a "${LOG_C}" "${META}"
      return 1
    fi
    if grep -q "server listening" "${LOG_S}" 2>/dev/null; then
      if python3 - "$p" <<'PY' >/dev/null 2>&1
import sys, urllib.request

port = int(sys.argv[1])
try:
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/healthz", timeout=3) as r:
        raise SystemExit(0 if int(getattr(r, "status", 200) or 200) == 200 else 1)
except Exception:
    raise SystemExit(1)
PY
      then
        return 0
      fi
    fi
    sleep 1
  done
  echo "TIMEOUT waiting for serve_policy (log 'server listening' + GET /healthz) on port ${p}" | tee -a "${LOG_C}" "${META}"
  return 1
}

{
  echo "START $(date -Is)"
  echo "EXP_NAME=${EXP_NAME}"
  echo "STEP=${STEP}"
  echo "ORBAX_STEP=${ORBAX_STEP}"
  echo "POLICY_CKPT_DIR=${POLICY_CKPT_DIR}"
  echo "WM=${WM}"
  echo "H=${AH} K=${K} overlap=${OVERLAP} trials_per_task=${NUM_TRIALS}"
  echo "LIBERO_WM_EVAL_TASK_SUITE=${TASK_SUITE}"
  echo "LIBERO_WM_EVAL_NO_WM=${NO_WM}"
  echo "PORT=${PORT}"
  echo "OUT_ROOT=${OUT_ROOT}"
  echo "CUDA_VISIBLE_DEVICES=${EVAL_CUDA_VISIBLE_DEVICES}"
} | tee "${META}"

SERVE_WM_ARGS=(--world-model-checkpoint "${WM}" --world-model-token-reducer-kind learned_cross_attn
  --world-model-action-encoder-kind transformer_block)
if [[ "${NO_WM}" == "1" ]]; then
  SERVE_WM_ARGS=()
fi
ASYNC_WM_FLAG=(--async-use-world-model)
if [[ "${NO_WM}" == "1" ]]; then
  ASYNC_WM_FLAG=(--no-async-use-world-model)
fi

: >"${LOG_S}"
CUDA_VISIBLE_DEVICES="${EVAL_CUDA_VISIBLE_DEVICES}" \
  "${PY_SERVER}" -u "${REPO}/scripts/serve_policy.py" --env LIBERO --port "${PORT}" \
  "${SERVE_WM_ARGS[@]}" \
  --async-ae-proprio-source prefix_t \
  policy:checkpoint --policy.config pi05_libero --policy.dir "${POLICY_CKPT_DIR}" \
  >>"${LOG_S}" 2>&1 &
SPID=$!
echo "server_pid=${SPID}" >>"${LOG_S}"

if ! wait_serve_ready "${PORT}" 900; then
  stop_server "${SPID}" "${PORT}"
  exit 1
fi
echo "serve_ready port=${PORT} healthz_ok" | tee -a "${LOG_C}"

if ! kill -0 "${SPID}" 2>/dev/null; then
  echo "FATAL: serve_policy no longer running before client start (pid ${SPID}); see ${LOG_S}" | tee -a "${LOG_C}" "${META}"
  stop_server "${SPID}" "${PORT}"
  exit 1
fi

set +e
if command -v xvfb-run >/dev/null 2>&1; then
  CUDA_VISIBLE_DEVICES="${EVAL_CUDA_VISIBLE_DEVICES}" MUJOCO_GL=glx \
    xvfb-run -a -s "${XVFB_OPTS}" \
    "${PY}" -u "${REPO}/examples/libero/main.py" \
    --host 127.0.0.1 --port "${PORT}" \
    --task-suite-name "${TASK_SUITE}" \
    --num-trials-per-task "${NUM_TRIALS}" \
    --async-inference \
    "${ASYNC_WM_FLAG[@]}" \
    --action-horizon "${AH}" \
    --async-trigger-step "${K}" \
    --overlap-skip "${OVERLAP}" \
    --pi0-norm-checkpoint-dir "${PI0_NORM_CHECKPOINT_DIR:-${POLICY_CKPT_DIR}}" \
    --video-out-path "${OUT_ROOT}/videos" \
    "${EXTRA_MAIN_FLAGS[@]}" \
    2>&1 | tee -a "${LOG_C}"
else
  echo "xvfb-run not found; MUJOCO_GL=egl" | tee -a "${LOG_C}"
  CUDA_VISIBLE_DEVICES="${EVAL_CUDA_VISIBLE_DEVICES}" MUJOCO_GL=egl \
    "${PY}" -u "${REPO}/examples/libero/main.py" \
    --host 127.0.0.1 --port "${PORT}" \
    --task-suite-name "${TASK_SUITE}" \
    --num-trials-per-task "${NUM_TRIALS}" \
    --async-inference \
    "${ASYNC_WM_FLAG[@]}" \
    --action-horizon "${AH}" \
    --async-trigger-step "${K}" \
    --overlap-skip "${OVERLAP}" \
    --pi0-norm-checkpoint-dir "${PI0_NORM_CHECKPOINT_DIR:-${POLICY_CKPT_DIR}}" \
    --video-out-path "${OUT_ROOT}/videos" \
    "${EXTRA_MAIN_FLAGS[@]}" \
    2>&1 | tee -a "${LOG_C}"
fi
EC=${PIPESTATUS[0]}
set -e

stop_server "${SPID}" "${PORT}"
echo "client_exit=${EC}" | tee -a "${META}"
echo "END $(date -Is)" | tee -a "${META}"
exit "${EC}"
