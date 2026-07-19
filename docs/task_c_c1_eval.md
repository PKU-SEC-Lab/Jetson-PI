# Task-C C1: PI0.5 confidence scheduling

This receipt path measures the released JAX PI0.5 + FAAC scheduler on paired
LIBERO-Spatial trajectories. C1 is fixed to H=10, K=9, seed 42, and 30 initial
states per each of the ten tasks. It compares `faac_only` with confidence
thresholds 0.2, 0.4, and 0.8. It does not run the gated K/Delta or OOD suites.

## Isolated environments and checkpoint

```bash
export TASK_C_ROOT=/home/pinyarash/dev/pinyarash/jetson-pi-task-c
bash scripts/bootstrap_task_c_envs.sh

PY="$TASK_C_ROOT/envs/policy/bin/python"
PYTHONPATH=packages/openpi-client/src:src:. "$PY" -m scripts.task_c_c1 checkpoint-manifest
PYTHONPATH=packages/openpi-client/src:src:. "$PY" -m scripts.task_c_c1 preflight
```

The policy environment is resolved by the repository `uv.lock`. The simulator
client uses `scripts/requirements-task-c-libero.txt`, compiled from the adjacent
`.in` file for Python 3.11 with the PyTorch CPU wheel index. This is intentional:
the upstream LIBERO example lock is Python-3.8-era and cannot resolve in the
approved Python 3.11 environment. Each run also records both environment freezes
and the client-lock SHA-256.

The checkpoint checkout must be detached at
`a3a803da176b10ab87dc5e29720d47c772848b43`. The manifest command hashes every
file and checks each LFS object against both its Git LFS OID and the pinned
ModelScope file API.

## Smoke and C1 blocks

Use a fresh output directory for every invocation; the runner refuses to append
to an existing receipt.

```bash
ROOT=/home/pinyarash/dev/pinyarash/jetson-pi-task-c
PYTHONPATH=packages/openpi-client/src:src:. "$ROOT/envs/policy/bin/python" \
  -m scripts.task_c_c1 run-condition \
  --condition kappa_0p4 --trials-per-task 2 --task-id-count 1 \
  --out "$ROOT/smoke/kappa_0p4"

for condition in faac_only kappa_0p2 kappa_0p4 kappa_0p8; do
  PYTHONPATH=packages/openpi-client/src:src:. "$ROOT/envs/policy/bin/python" \
    -m scripts.task_c_c1 run-condition \
    --condition "$condition" --out "$ROOT/c1/$condition"
done

PYTHONPATH=packages/openpi-client/src:src:. "$ROOT/envs/policy/bin/python" \
  -m scripts.task_c_analysis aggregate-c1 "$ROOT/c1/aggregate" \
  --faac-only "$ROOT/c1/faac_only" \
  --kappa-0p2 "$ROOT/c1/kappa_0p2" \
  --kappa-0p4 "$ROOT/c1/kappa_0p4" \
  --kappa-0p8 "$ROOT/c1/kappa_0p8"
```

## Boundaries and outputs

`c_tier0_ms` covers the jitted 40M world-model call, including the learned
4×1024 reducer and the device-side `-mean(log_var)` calculation, plus host
materialization of kappa and the actual confidence-threshold comparison. It excludes VLM, action expert, simulator, websocket,
mu copying, and trace I/O. The first 30 WM calls are marked warmup and excluded
from timing percentiles.

Each condition emits:

- `run_manifest.json`, `episodes.jsonl`, `steps_raw.jsonl`, and
  `steps_labeled.jsonl`;
- `server_trace/wm_calls.jsonl` and `server_trace/policy_calls.jsonl`;
- `mu/calibration/` and `mu/eval/` f16 NumPy shards plus row indexes;
- `summary.json`, logs, environment freezes, and an out-of-band-hashed
  `SHA256SUMS` receipt.

Trajectory parity fixes the split before inference: even initialization indices
are calibration and odd indices are eval. Condition and success are excluded
from the split key. Phase is a declared proxy: approach precedes the first of
two consecutive positive gripper-close commands; contact begins at that command.
The raw end-effector delta and gripper qpos are retained on every control step.

The aggregate reports exact paired McNemar tests and a deterministic paired
bootstrap confidence interval for candidate-minus-baseline success rate. A skip
rate is deployable only when McNemar is non-significant at 0.05 and the one-sided
95% lower bound is at least -5 percentage points. A statistically significant
improvement is also accepted; McNemar significance is never treated as harm by
itself. Raw, success-conditioned, and
deployable-valid skip rates remain separate fields.

The C1 aggregator fails closed unless all four conditions contain exactly the
same 300 trajectories: tasks 0–9, initialization indices 0–29, and seed 42.
This prevents a smoke or partial block from being labeled as the completed
1,200-episode experiment.

## Gated follow-on: RAPID comparator

After C1 and only with a separate gate, add a training-free O(1) proprio trigger
using end-effector translation/rotation velocity and gripper-state transitions.
Run it on the identical task/seed trajectories and report the same success,
noninferiority, raw/valid-skip, and approach/contact schema. Keep its trigger
cost separate from `c_tier0_ms`; this A/B determines whether the learned 40M WM
earns its additional compute over a kinematic heuristic.
