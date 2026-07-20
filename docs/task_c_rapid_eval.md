# Task-C RAPID: sealed success-anchored A/B

RAPID is a training-free, constant-time kinematic routing trigger for the
released JAX PI0.5 + FAAC scheduler. It replaces the learned 40M kappa value as
the skip/infer routing signal. It does **not** remove the world model: FAAC still
uses the world model, so every emitted summary records
`wm_still_required_for_faac=true`.

## Frozen trigger

For end-effector translation `p_t`, axis-angle orientation `r_t`, corresponding
rotation matrix `R_t`, and proprio tick interval `dt=1`, RAPID computes

```text
v_t     = ||p_t - p_(t-1)|| / dt
a_t     = ||p_t - 2 p_(t-1) + p_(t-2)|| / dt^2
omega_t = angle(R_(t-1)^T R_t) / dt
g_t     = q_left - q_right
```

The gripper is closed when `g_t <= theta_closed`, open when
`g_t >= theta_open`, and transitioning inside the hysteresis band. A change
between stable open/closed states or occupancy of the band starts a two-tick
transition cooldown. A sign change in the already-issued fixed-horizon gripper
commands is a separate transition veto.

RAPID returns `SKIP` only when the current and previous samples are both safe:
all three motion features are at or below their frozen thresholds, neither
observed nor issued gripper commands transition, and all inputs are finite with
three-sample history. Every malformed, non-finite, cold-start, transition, or
unsafe case returns `INFER`.

The sealed q95 thresholds are:

| Parameter | Value |
| --- | ---: |
| `theta_v` | 0.0325055209451041 |
| `theta_a` | 0.0244409472863782 |
| `theta_omega` | 0.02518825690298463 |
| `theta_closed` | 0.02946965101485451 |
| `theta_open` | 0.05416228474738698 |
| transition cooldown | 2 ticks |

`configs/task_c/rapid_thresholds.json` has SHA-256
`959ee301d68086a71351efbcd101239cb84edb46b4173b2cfdc826f28056b399`
and was committed, with `configs/task_c/SHA256SUMS`, at
`e2bc7c206a7a7bb23ead5d240b04309a95aada13` before the first final-eval
episode loaded.

## Disjoint calibration and hard gates

The numeric gates were committed at
`0d7928bb87c8efb4eae13350109ac231840e989b` before calibration. Cal-Fit used
init states 30--39 (100 pairs), Cal-Confirm used 40--49 (100 pairs), and final
evaluation used 0--29 (300 pairs). All use tasks 0--9 and seed 42. Code verifies
zero trajectory overlap.

Cal-Fit selected the highest success-anchored call-level skip rate among ladder
candidates with exact paired McNemar `p >= 0.05` and candidate-minus-baseline
success delta `>= 0`; ties select the lower quantile. The computed winner was
q95: 98/100 versus 98/100, delta 0, McNemar p=1.0, and 85.659% valid skip.

Cal-Confirm compared 97/100 RAPID with 99/100 always-infer: delta -2 pp,
McNemar p=0.625, and one-sided paired-percentile lower bound -5 pp. This passed
the precommitted strict abort rule: abort only for harmful McNemar p<0.05,
delta<-5 pp, or lower bound<-5 pp. No retuning followed confirmation.

Before the candidate final arm, the always-infer result of 97.667% reconciled
against the sealed C1 result of 97.0%. Absolute drift was 0.667 pp, below the
committed 3 pp hard-abort tolerance.

## Final 600-episode paired result

The matrix was LIBERO-Spatial, K=9, tasks 0--9, seed 42, init states 0--29:
300 episodes per arm and 600 total. Both manifests verify the same infer,
fallback, FAAC, and rollout implementation digest; `routing_policy` is the only
manipulated variable.

| Success-primary metric | Always-infer | RAPID |
| --- | ---: | ---: |
| Successes | 293/300 | 293/300 |
| Success rate | 97.667% | 97.667% |
| Candidate delta |  | 0.0 pp |

There were 12 discordant pairs: six baseline-failure/RAPID-success and six
baseline-success/RAPID-failure. The two-sided exact paired McNemar p-value is
1.0. The 50,000-sample paired percentile interval has one-sided 95% lower bound
-2.0 pp and two-sided 95% interval [-2.333 pp, +2.333 pp]. RAPID therefore
passes the precommitted -5 pp non-inferiority margin.

Per-task baseline/RAPID successes were `30/30`, `30/30`, `30/30`, `29/30`,
`27/27`, `30/30`, `30/29`, `29/28`, `30/30`, and `28/29`, respectively.
Per-task intervals are descriptive at 30 pairs; the 300-pair pooled comparison
is the success-primary test.

## Routing and timing metrics

| Metric | Result |
| --- | ---: |
| Raw policy-call skip | 2,847/3,288 (86.588%) |
| Success-anchored valid skip | 2,665/3,050 (87.377%) |
| Rollout eligible-step skip | 19,929/20,370 (97.835%) |
| Agreement with same-step 40M kappa decision | 2,847/3,288 (86.588%) |
| Kappa comparison coverage | 100% |
| Trigger compute, mean | 105.866 us |
| Trigger compute, p50 / p95 / p99 | 99.297 / 143.978 / 152.908 us |

The kappa comparison is defined as the first eligible 40M kappa gate at the
same trigger environment step, using kappa delta 0.4.

Approach decisions skipped 1,248/1,260 (99.048%); contact decisions skipped
1,599/2,028 (78.846%). All seven RAPID failure episodes contained skips in both
phases. Across the six harmful discordant failures, approach contained 27 skip
decisions and contact contained 83. These phase counts are descriptive and do
not by themselves assign causality.

## Receipts and verification

The immutable external result roots are:

- always-infer: `/home/pinyarash/dev/pinyarash/jetson-pi-task-c/rapid/final/always_infer`,
  receipt SHA-256 `6b330e258adbc241b9102345ade52e175b89c963ba0a31c83e751be95257474c`;
- RAPID: `/home/pinyarash/dev/pinyarash/jetson-pi-task-c/rapid/final/rapid`,
  receipt SHA-256 `dace80e55ba1e6d7d218732b3a1923d741e50bc261309d1e7f3683c88c35d707`;
- aggregate: `/home/pinyarash/dev/pinyarash/jetson-pi-task-c/rapid/final/aggregate`,
  receipt SHA-256 `4c343046f6967217d4e4530aea2826f4e222046d57ce2c55c4618c9fa37817bc`.

The aggregate `SHA256SUMS` verifies `summary.json`, `run_manifest.json`, and all
3,288 raw RAPID policy-call rows. Focused verification completed with 57 passing
tests, clean Ruff output, and Pyright reporting 0 errors for the RAPID
implementation modules.
