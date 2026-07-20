"""Raw-row calibration, sealing, and paired analysis for Task-C RAPID."""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Mapping, Sequence
import hashlib
import json
import math
import pathlib
import subprocess
from typing import Any, cast

import numpy as np
from openpi_client import rapid_trigger
from openpi_client import task_c_trace

CALIB_GATES_SCHEMA_VERSION = "jetson-pi-task-c-rapid-calib-gates-v1"
SHARED_EXECUTION_PATH_VERSION = "jetson-pi-task-c-rapid-shared-path-v1"
DEFAULT_KAPPA_DELTA = 0.4


def _unique(records: Sequence[Mapping[str, Any]], key: str) -> dict[str, Mapping[str, Any]]:
    output: dict[str, Mapping[str, Any]] = {}
    for record in records:
        value = str(record[key])
        if value in output:
            raise task_c_trace.TaskCTraceError(f"duplicate raw-row {key}: {value}")
        output[value] = record
    return output


def _ratio(numerator: int, denominator: int) -> float | None:
    return float(numerator / denominator) if denominator else None


def _rapid_policy_rows(
    wm_calls: Sequence[Mapping[str, Any]],
    *,
    kappa_delta: float,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for call in wm_calls:
        grouped[str(call["policy_call_id"])].append(call)
    output: list[dict[str, Any]] = []
    for policy_call_id, grouped_calls in sorted(grouped.items()):
        calls = sorted(grouped_calls, key=lambda item: int(item["round_index"]))
        rapid_payloads = [call.get("rapid") for call in calls]
        if all(payload is None for payload in rapid_payloads):
            continue
        if any(not isinstance(payload, Mapping) for payload in rapid_payloads):
            raise task_c_trace.TaskCTraceError(f"partial RAPID payloads for {policy_call_id}")
        typed_payloads = [cast(Mapping[str, Any], payload) for payload in rapid_payloads if isinstance(payload, Mapping)]
        canonical_payloads = {task_c_trace.canonical_json_bytes(dict(payload)) for payload in typed_payloads}
        if len(canonical_payloads) != 1:
            raise task_c_trace.TaskCTraceError(f"RAPID payload changed within policy call {policy_call_id}")
        rapid = dict(typed_payloads[0])
        decision = rapid.get("decision")
        if decision not in {"skip", "infer"}:
            raise task_c_trace.TaskCTraceError(f"invalid RAPID route for {policy_call_id}: {decision!r}")
        routing_policies = {str(call.get("routing_policy")) for call in calls}
        if len(routing_policies) != 1:
            raise task_c_trace.TaskCTraceError(f"invalid RAPID routing-policy trace for {policy_call_id}")
        routing_policy = routing_policies.pop()
        if routing_policy not in {"always_infer", "rapid"}:
            raise task_c_trace.TaskCTraceError(f"invalid RAPID routing-policy trace for {policy_call_id}")
        round_zero = [call for call in calls if int(call["round_index"]) == 0]
        if len(round_zero) != 1:
            raise task_c_trace.TaskCTraceError(f"RAPID policy call {policy_call_id} needs one round zero")
        kappa0 = float(round_zero[0]["kappa"])
        eligible = [call for call in calls if int(call["round_index"]) > 0]
        if not eligible:
            raise task_c_trace.TaskCTraceError(f"RAPID policy call {policy_call_id} has no eligible round")
        kappa_route = "infer" if any(float(call["kappa"]) < kappa0 - kappa_delta for call in eligible) else "skip"
        trigger_compute_ns = rapid.get("trigger_compute_ns")
        if isinstance(trigger_compute_ns, bool) or not isinstance(trigger_compute_ns, int) or trigger_compute_ns < 0:
            raise task_c_trace.TaskCTraceError(f"invalid trigger timing for {policy_call_id}")
        context = calls[0]
        output.append(
            {
                "policy_call_id": policy_call_id,
                "trajectory_id": str(context["trajectory_id"]),
                "task_id": int(context["task_id"]),
                "episode_idx": int(context["episode_idx"]),
                "env_step": int(context["env_step"]),
                "routing_policy": routing_policy,
                "rapid_decision": decision,
                "executed_decision": "infer" if routing_policy == "always_infer" else decision,
                "kappa_decision": kappa_route,
                "decision_agreement": decision == kappa_route,
                "trigger_compute_ns": trigger_compute_ns,
                "rapid": rapid,
            }
        )
    return output


def _phase_by_key(steps: Sequence[Mapping[str, Any]]) -> dict[tuple[str, int], str]:
    return task_c_trace.contact_phase_by_trajectory(steps)


def _skip_metrics(calls: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    skips = sum(str(call["rapid_decision"]) == "skip" for call in calls)
    return {"decisions": len(calls), "skips": skips, "skip_rate": _ratio(skips, len(calls))}


def paired_raw_result(
    baseline_episodes: Sequence[Mapping[str, Any]],
    candidate_episodes: Sequence[Mapping[str, Any]],
    candidate_wm_calls: Sequence[Mapping[str, Any]],
    candidate_steps: Sequence[Mapping[str, Any]],
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
    noninferiority_margin: float,
    kappa_delta: float,
) -> dict[str, Any]:
    """Compute all success and trigger metrics directly from append-only rows."""

    baseline = _unique(baseline_episodes, "trajectory_id")
    candidate = _unique(candidate_episodes, "trajectory_id")
    if set(baseline) != set(candidate):
        raise task_c_trace.TaskCTraceError("paired raw episode trajectory sets differ")
    trajectory_ids = sorted(baseline)
    baseline_success = [bool(baseline[key]["success"]) for key in trajectory_ids]
    candidate_success = [bool(candidate[key]["success"]) for key in trajectory_ids]
    paired_mcnemar = task_c_trace.exact_mcnemar(baseline_success, candidate_success)
    paired_bootstrap = task_c_trace.paired_bootstrap_success_delta(
        baseline_success,
        candidate_success,
        samples=bootstrap_samples,
        seed=bootstrap_seed,
    )
    paired_bootstrap["method"] = "paired_percentile"
    delta = float(paired_bootstrap["success_rate_delta"])
    rapid_calls = _rapid_policy_rows(candidate_wm_calls, kappa_delta=kappa_delta)
    unknown = sorted({str(call["trajectory_id"]) for call in rapid_calls}.difference(candidate))
    if unknown:
        raise task_c_trace.TaskCTraceError(f"RAPID calls reference unknown trajectories: {unknown[:5]}")
    preserved = {
        key for key in trajectory_ids if bool(baseline[key]["success"]) and bool(candidate[key]["success"])
    }
    raw_skip = _skip_metrics(rapid_calls)
    anchored_calls = [call for call in rapid_calls if str(call["trajectory_id"]) in preserved]
    anchored_skip = _skip_metrics(anchored_calls)
    phase_map = _phase_by_key(candidate_steps)
    per_phase: dict[str, Any] = {}
    harmful = {
        key for key in trajectory_ids if bool(baseline[key]["success"]) and not bool(candidate[key]["success"])
    }
    candidate_failures = {key for key in trajectory_ids if not bool(candidate[key]["success"])}
    for phase in ("approach", "contact"):
        phase_calls: list[Mapping[str, Any]] = []
        for call in rapid_calls:
            key = (str(call["trajectory_id"]), int(call["env_step"]))
            if key not in phase_map:
                raise task_c_trace.TaskCTraceError(f"RAPID call has no phase row: {key}")
            if phase_map[key] == phase:
                phase_calls.append(call)
        phase_skips = [call for call in phase_calls if str(call["rapid_decision"]) == "skip"]
        harmful_skips = [call for call in phase_skips if str(call["trajectory_id"]) in harmful]
        failure_skips = [call for call in phase_skips if str(call["trajectory_id"]) in candidate_failures]
        per_phase[phase] = {
            **_skip_metrics(phase_calls),
            "harmful_discordant_failure_episodes_with_skip": len(
                {str(call["trajectory_id"]) for call in harmful_skips}
            ),
            "harmful_discordant_skip_decisions": len(harmful_skips),
            "candidate_failure_episodes_with_skip": len({str(call["trajectory_id"]) for call in failure_skips}),
            "candidate_failure_skip_decisions": len(failure_skips),
        }
    timing_us = [float(call["trigger_compute_ns"]) / 1000.0 for call in rapid_calls]
    timing = task_c_trace.percentile_summary(timing_us) if timing_us else None
    agreements = sum(bool(call["decision_agreement"]) for call in rapid_calls)
    agreement = {
        "decisions": len(rapid_calls),
        "agreements": agreements,
        "disagreements": len(rapid_calls) - agreements,
        "agreement_rate": _ratio(agreements, len(rapid_calls)),
        "kappa_delta": float(kappa_delta),
    }
    ni_pass = float(paired_bootstrap["lower_95_one_sided"]) >= noninferiority_margin
    mcnemar_no_harm = not (
        float(paired_mcnemar["p_value_two_sided_exact"]) < 0.05
        and int(paired_mcnemar["baseline_success_candidate_failure"])
        > int(paired_mcnemar["baseline_failure_candidate_success"])
    )
    validity = ni_pass and mcnemar_no_harm
    return {
        "computed_from_raw_rows": True,
        "n_pairs": len(trajectory_ids),
        "baseline_successes": sum(baseline_success),
        "candidate_successes": sum(candidate_success),
        "baseline_success_rate": sum(baseline_success) / len(trajectory_ids),
        "candidate_success_rate": sum(candidate_success) / len(trajectory_ids),
        "success_rate_delta": delta,
        "paired_mcnemar": paired_mcnemar,
        "paired_bootstrap": paired_bootstrap,
        "noninferiority_margin": float(noninferiority_margin),
        "noninferiority_pass": ni_pass,
        "mcnemar_no_evidence_of_harm": mcnemar_no_harm,
        "validity_gate_pass": validity,
        "preserved_success_pairs": len(preserved),
        "raw_skip_rate": raw_skip["skip_rate"],
        "raw_skip": raw_skip,
        "success_anchored_skip_rate": anchored_skip["skip_rate"],
        "success_anchored_skip": anchored_skip,
        "valid_skip_rate": anchored_skip["skip_rate"] if validity else None,
        "contact_vs_approach_failures": per_phase,
        "trigger_compute_us": timing,
        "kappa_decision_agreement": agreement,
        "rapid_policy_calls": rapid_calls,
    }


def select_cal_fit_candidate(
    candidate_results: Mapping[float, Mapping[str, Any]],
    gate_config: Mapping[str, Any],
) -> dict[str, Any]:
    fit = gate_config["cal_fit"]
    eligible: list[tuple[float, Mapping[str, Any]]] = []
    audit: list[dict[str, Any]] = []
    for quantile, result in sorted(candidate_results.items()):
        p_value = float(result["paired_mcnemar"]["p_value_two_sided_exact"])
        delta = float(result["success_rate_delta"])
        skip_rate = result["success_anchored_skip_rate"]
        passes = (
            p_value >= float(fit["mcnemar_p_value_min_inclusive"])
            and delta >= float(fit["success_rate_delta_min_inclusive"])
            and skip_rate is not None
        )
        row = {
            "motion_quantile": float(quantile),
            "mcnemar_p_value": p_value,
            "success_rate_delta": delta,
            "success_anchored_skip_rate": skip_rate,
            "eligible": passes,
        }
        audit.append(row)
        if passes:
            eligible.append((float(quantile), result))
    if not eligible:
        raise task_c_trace.TaskCTraceError("Cal-Fit selected no ladder candidate under the committed numeric gates")
    selected_quantile, selected_result = min(
        eligible,
        key=lambda item: (-float(item[1]["success_anchored_skip_rate"]), item[0]),
    )
    return {
        "selection_computed_from_raw_rows": True,
        "selection_primary": fit["selection_primary"],
        "tie_break": fit["tie_break"],
        "selected_motion_quantile": selected_quantile,
        "selected_success_anchored_skip_rate": selected_result["success_anchored_skip_rate"],
        "candidate_audit": audit,
    }


def cal_confirm_decision(result: Mapping[str, Any], gate_config: Mapping[str, Any]) -> dict[str, Any]:
    gate = gate_config["cal_confirm"]
    p_value = float(result["paired_mcnemar"]["p_value_two_sided_exact"])
    harmful = int(result["paired_mcnemar"]["baseline_success_candidate_failure"]) > int(
        result["paired_mcnemar"]["baseline_failure_candidate_success"]
    )
    delta = float(result["success_rate_delta"])
    lower = float(result["paired_bootstrap"]["lower_95_one_sided"])
    reasons: list[str] = []
    if harmful and p_value < float(gate["abort_if_harmful_mcnemar_p_below"]):
        reasons.append("harmful_paired_mcnemar")
    if delta < float(gate["abort_if_success_rate_delta_below"]):
        reasons.append("success_rate_delta")
    if lower < float(gate["abort_if_paired_percentile_lower_one_sided_below"]):
        reasons.append("paired_percentile_lower_one_sided")
    return {
        "computed_from_raw_rows": True,
        "abort": bool(reasons),
        "abort_reasons": reasons,
        "observed": {
            "paired_mcnemar_p_value": p_value,
            "harmful_direction": harmful,
            "success_rate_delta": delta,
            "paired_percentile_lower_one_sided": lower,
        },
    }


def _two_cluster_centers(values: np.ndarray) -> tuple[float, float]:
    if values.size < 4 or not np.isfinite(values).all():
        raise task_c_trace.TaskCTraceError("gripper calibration requires at least four finite apertures")
    low, high = float(np.min(values)), float(np.max(values))
    if low == high:
        raise task_c_trace.TaskCTraceError("gripper calibration aperture has no open/closed separation")
    for _ in range(100):
        low_mask = np.abs(values - low) <= np.abs(values - high)
        if not low_mask.any() or low_mask.all():
            raise task_c_trace.TaskCTraceError("gripper calibration produced an empty aperture cluster")
        new_low = float(np.median(values[low_mask]))
        new_high = float(np.median(values[~low_mask]))
        if abs(new_low - low) < 1e-12 and abs(new_high - high) < 1e-12:
            break
        low, high = new_low, new_high
    return min(low, high), max(low, high)


def derive_threshold_ladder(
    episodes: Sequence[Mapping[str, Any]],
    proprio_observations: Sequence[Mapping[str, Any]],
    *,
    motion_quantiles: Sequence[float],
    transition_cooldown_steps: int,
    dt: float,
) -> list[dict[str, Any]]:
    """Freeze motion quantiles from successful, disjoint Cal-Fit rows only."""

    episode_by_id = _unique(episodes, "trajectory_id")
    successful = {key for key, episode in episode_by_id.items() if bool(episode["success"])}
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in proprio_observations:
        trajectory_id = str(row["trajectory_id"])
        if trajectory_id not in episode_by_id:
            raise task_c_trace.TaskCTraceError(f"proprio row has unknown trajectory {trajectory_id}")
        if trajectory_id in successful:
            grouped[trajectory_id].append(row)
    if set(grouped) != successful:
        missing = sorted(successful.difference(grouped))
        raise task_c_trace.TaskCTraceError(f"successful Cal-Fit trajectories missing proprio rows: {missing[:5]}")
    features: list[dict[str, float | bool | None]] = []
    aperture_values: list[float] = []
    for trajectory_id, grouped_rows in sorted(grouped.items()):
        rows = sorted(grouped_rows, key=lambda item: int(item["env_step"]))
        env_steps = [int(row["env_step"]) for row in rows]
        if len(env_steps) != len(set(env_steps)):
            raise task_c_trace.TaskCTraceError(f"duplicate trigger env_step rows for {trajectory_id}")
        states = np.asarray([row["state"] for row in rows], dtype=np.float64)
        trajectory_features = rapid_trigger.finite_difference_feature_rows(states, dt=dt)
        features.extend(row for row in trajectory_features if bool(row["history_valid"]))
        aperture_values.extend(
            float(row["gripper_aperture"])
            for row in trajectory_features
            if row["gripper_aperture"] is not None
        )
    if not features:
        raise task_c_trace.TaskCTraceError("Cal-Fit has no three-sample kinematic feature rows")
    speeds = np.asarray([float(row["ee_speed"]) for row in features if row["ee_speed"] is not None])
    accelerations = np.asarray(
        [float(row["ee_acceleration"]) for row in features if row["ee_acceleration"] is not None]
    )
    angular_speeds = np.asarray(
        [float(row["ee_angular_speed"]) for row in features if row["ee_angular_speed"] is not None]
    )
    closed_center, open_center = _two_cluster_centers(np.asarray(aperture_values, dtype=np.float64))
    gap = open_center - closed_center
    theta_closed = closed_center + gap / 3.0
    theta_open = open_center - gap / 3.0
    rows_sha = task_c_trace.sha256_bytes(task_c_trace.canonical_json_bytes(list(proprio_observations)))
    ladder: list[dict[str, Any]] = []
    for quantile in motion_quantiles:
        if not 0.0 < float(quantile) <= 1.0:
            raise ValueError(f"motion quantile must be in (0,1], got {quantile}")
        ladder.append(
            {
                "schema_version": rapid_trigger.THRESHOLD_SCHEMA_VERSION,
                "formula_version": rapid_trigger.FORMULA_VERSION,
                "motion_quantile": float(quantile),
                "thresholds": {
                    "theta_v": float(np.quantile(speeds, quantile)),
                    "theta_a": float(np.quantile(accelerations, quantile)),
                    "theta_omega": float(np.quantile(angular_speeds, quantile)),
                    "theta_closed": theta_closed,
                    "theta_open": theta_open,
                    "transition_cooldown_steps": int(transition_cooldown_steps),
                    "dt": float(dt),
                },
                "calibration": {
                    "source": "successful Cal-Fit raw proprio observations only",
                    "successful_trajectories": len(successful),
                    "kinematic_feature_rows": len(features),
                    "proprio_observations_sha256": rows_sha,
                    "gripper_cluster_centers": {"closed": closed_center, "open": open_center},
                    "gripper_hysteresis_formula": "theta_closed=closed+gap/3; theta_open=open-gap/3",
                },
            }
        )
    return ladder


def assert_eval_arm_equivalence(
    baseline_manifest: Mapping[str, Any],
    candidate_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    if baseline_manifest.get("shared_execution_path") != candidate_manifest.get("shared_execution_path"):
        raise task_c_trace.TaskCTraceError("eval arms do not have a byte-identical shared execution path")
    shared = baseline_manifest.get("shared_execution_path")
    if not isinstance(shared, Mapping) or shared.get("arms_share_byte_identical_paths") is not True:
        raise task_c_trace.TaskCTraceError("shared execution path lacks the mandatory equality assertion")
    if baseline_manifest.get("rapid_thresholds_sha256") != candidate_manifest.get("rapid_thresholds_sha256"):
        raise task_c_trace.TaskCTraceError("eval arms loaded different frozen RAPID threshold bytes")
    if baseline_manifest.get("only_manipulated_variable") != "routing_policy" or candidate_manifest.get(
        "only_manipulated_variable"
    ) != "routing_policy":
        raise task_c_trace.TaskCTraceError("eval manifests do not isolate routing_policy as the manipulated variable")
    if baseline_manifest.get("routing_policy") != "always_infer" or candidate_manifest.get("routing_policy") != "rapid":
        raise task_c_trace.TaskCTraceError("eval routing arms must be always_infer and rapid")
    return {
        "verified": True,
        "only_manipulated_variable": "routing_policy",
        "shared_execution_path_digest": shared.get("digest"),
    }


def shared_execution_path(repo: pathlib.Path) -> dict[str, Any]:
    relative_paths = [
        "examples/libero/main.py",
        "packages/openpi-client/src/openpi_client/action_chunk_broker.py",
        "packages/openpi-client/src/openpi_client/rapid_trigger.py",
        "src/openpi/policies/pi0_async_inference_policy.py",
    ]
    files = [
        {"path": relative, "sha256": task_c_trace.sha256_file(repo / relative)} for relative in relative_paths
    ]
    digest = task_c_trace.sha256_bytes(task_c_trace.canonical_json_bytes(files))
    return {
        "schema_version": SHARED_EXECUTION_PATH_VERSION,
        "digest": digest,
        "files": files,
        "infer_path": "Pi0AsyncInferencePolicy._infer_wm_multi_rollout_ae -> _LowKappaFullPi0Fallback",
        "fallback_path": "_make_wm_low_replan_two_phase_fn -> WebsocketClientPolicy.infer",
        "faac_path": "world-model mu -> shared action-expert sampling/merge",
        "routing_seam": "openpi_client.rapid_trigger.route_decision",
        "arms_share_byte_identical_paths": True,
    }


def _run_git(repo: pathlib.Path, args: Sequence[str]) -> bytes:
    completed = subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)
    return completed.stdout


def verify_git_sealed_file(repo: pathlib.Path, path: pathlib.Path) -> dict[str, Any]:
    repo = repo.resolve()
    path = path.resolve()
    relative = path.relative_to(repo).as_posix()
    working = path.read_bytes()
    committed = _run_git(repo, ["show", f"HEAD:{relative}"])
    if working != committed:
        raise task_c_trace.TaskCTraceError(f"sealed file differs from HEAD: {relative}")
    status = _run_git(repo, ["status", "--porcelain", "--", relative]).decode().strip()
    if status:
        raise task_c_trace.TaskCTraceError(f"sealed file is dirty: {relative}: {status}")
    return {
        "path": relative,
        "sha256": hashlib.sha256(working).hexdigest(),
        "git_head": _run_git(repo, ["rev-parse", "HEAD"]).decode().strip(),
        "committed_and_clean": True,
    }


def verify_threshold_sha_manifest(repo: pathlib.Path, threshold_path: pathlib.Path, sums_path: pathlib.Path) -> dict[str, Any]:
    threshold_receipt = verify_git_sealed_file(repo, threshold_path)
    sums_receipt = verify_git_sealed_file(repo, sums_path)
    expected: dict[str, str] = {}
    for line in sums_path.read_text(encoding="utf-8").splitlines():
        digest, separator, name = line.partition("  ")
        if separator != "  " or len(digest) != 64:
            raise task_c_trace.TaskCTraceError(f"invalid threshold SHA256SUMS line: {line!r}")
        expected[name] = digest
    relative_name = threshold_path.relative_to(sums_path.parent).as_posix()
    if expected.get(relative_name) != threshold_receipt["sha256"]:
        raise task_c_trace.TaskCTraceError("threshold SHA256SUMS does not bind rapid_thresholds.json")
    return {"threshold": threshold_receipt, "sha256s": sums_receipt}


def load_gate_config(path: pathlib.Path) -> dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict) or document.get("schema_version") != CALIB_GATES_SCHEMA_VERSION:
        raise task_c_trace.TaskCTraceError("unexpected RAPID calibration gate schema")
    for key in ("calibration_partitions", "ladder", "bootstrap", "cal_fit", "cal_confirm", "baseline_reconciliation", "final_eval"):
        if key not in document:
            raise task_c_trace.TaskCTraceError(f"RAPID calibration gates missing {key}")
    if document["bootstrap"].get("method") != "paired_percentile":
        raise task_c_trace.TaskCTraceError("RAPID bootstrap method must be paired_percentile")
    return document


def verify_condition_receipt(root: pathlib.Path) -> dict[str, Any]:
    from scripts import task_c_analysis

    digest = task_c_analysis._verify_sha_manifest(root)  # noqa: SLF001 - experiment receipt verifier
    return {"root": str(root.resolve()), "sha256s_sha256": digest}


def validate_partition(
    episodes: Sequence[Mapping[str, Any]],
    partition: Mapping[str, Any],
    *,
    label: str,
) -> dict[str, Any]:
    expected_tasks = {int(value) for value in partition["task_ids"]}
    start = int(partition["episode_idx_start"])
    stop = int(partition["episode_idx_stop_exclusive"])
    expected_seed = int(partition["seed"])
    expected_pairs = int(partition["expected_pairs"])
    if len(episodes) != expected_pairs:
        raise task_c_trace.TaskCTraceError(f"{label} has {len(episodes)} episodes, expected {expected_pairs}")
    task_ids = {int(row["task_id"]) for row in episodes}
    if task_ids != expected_tasks:
        raise task_c_trace.TaskCTraceError(f"{label} task set {sorted(task_ids)} != {sorted(expected_tasks)}")
    if any(int(row["seed"]) != expected_seed for row in episodes):
        raise task_c_trace.TaskCTraceError(f"{label} seed differs from {expected_seed}")
    expected_indices = set(range(start, stop))
    for task_id in sorted(expected_tasks):
        actual = {int(row["episode_idx"]) for row in episodes if int(row["task_id"]) == task_id}
        if actual != expected_indices:
            raise task_c_trace.TaskCTraceError(
                f"{label} task {task_id} init states {sorted(actual)} != {sorted(expected_indices)}"
            )
    trajectory_ids = {str(row["trajectory_id"]) for row in episodes}
    if len(trajectory_ids) != expected_pairs:
        raise task_c_trace.TaskCTraceError(f"{label} contains duplicate trajectory ids")
    return {
        "label": label,
        "pairs": expected_pairs,
        "task_ids": sorted(expected_tasks),
        "episode_idx_start": start,
        "episode_idx_stop_exclusive": stop,
        "seed": expected_seed,
        "trajectory_ids_sha256": task_c_trace.sha256_bytes(
            task_c_trace.canonical_json_bytes(sorted(trajectory_ids))
        ),
    }


def _raw_result_from_roots(
    baseline_root: pathlib.Path,
    candidate_root: pathlib.Path,
    gates: Mapping[str, Any],
    *,
    bootstrap_seed: int,
) -> dict[str, Any]:
    baseline = task_c_trace.read_jsonl(baseline_root / "episodes.jsonl")
    candidate, calls, steps = _load_condition_rows(candidate_root)
    return paired_raw_result(
        baseline,
        candidate,
        calls,
        steps,
        bootstrap_samples=int(gates["bootstrap"]["samples"]),
        bootstrap_seed=bootstrap_seed,
        noninferiority_margin=float(gates["final_eval"]["noninferiority_margin"]),
        kappa_delta=float(gates["final_eval"]["kappa_shadow_delta"]),
    )


def cal_fit_from_roots(
    baseline_root: pathlib.Path,
    candidate_roots: Mapping[float, pathlib.Path],
    gates: Mapping[str, Any],
) -> dict[str, Any]:
    partition = gates["calibration_partitions"]["cal_fit"]
    baseline_receipt = verify_condition_receipt(baseline_root)
    baseline = task_c_trace.read_jsonl(baseline_root / "episodes.jsonl")
    baseline_partition = validate_partition(baseline, partition, label="Cal-Fit baseline")
    results: dict[float, dict[str, Any]] = {}
    candidate_receipts: dict[str, Any] = {}
    for quantile, root in sorted(candidate_roots.items()):
        candidate_receipts[str(quantile)] = verify_condition_receipt(root)
        candidate = task_c_trace.read_jsonl(root / "episodes.jsonl")
        validate_partition(candidate, partition, label=f"Cal-Fit q={quantile}")
        result = _raw_result_from_roots(
            baseline_root,
            root,
            gates,
            bootstrap_seed=int(gates["bootstrap"]["seed"]) + round(quantile * 100),
        )
        results[float(quantile)] = result
    selection = select_cal_fit_candidate(results, gates)
    compact_results = {
        str(quantile): {key: value for key, value in result.items() if key != "rapid_policy_calls"}
        for quantile, result in sorted(results.items())
    }
    return {
        "schema_version": "jetson-pi-task-c-rapid-cal-fit-v1",
        "computed_from_raw_rows": True,
        "partition": baseline_partition,
        "baseline_receipt": baseline_receipt,
        "candidate_receipts": candidate_receipts,
        "selection": selection,
        "candidates": compact_results,
    }


def cal_confirm_from_roots(
    baseline_root: pathlib.Path,
    candidate_root: pathlib.Path,
    gates: Mapping[str, Any],
) -> dict[str, Any]:
    partition = gates["calibration_partitions"]["cal_confirm"]
    baseline_receipt = verify_condition_receipt(baseline_root)
    candidate_receipt = verify_condition_receipt(candidate_root)
    baseline = task_c_trace.read_jsonl(baseline_root / "episodes.jsonl")
    candidate = task_c_trace.read_jsonl(candidate_root / "episodes.jsonl")
    partition_receipt = validate_partition(baseline, partition, label="Cal-Confirm baseline")
    validate_partition(candidate, partition, label="Cal-Confirm candidate")
    result = _raw_result_from_roots(
        baseline_root,
        candidate_root,
        gates,
        bootstrap_seed=int(gates["bootstrap"]["seed"]) + 10_000,
    )
    decision = cal_confirm_decision(result, gates)
    return {
        "schema_version": "jetson-pi-task-c-rapid-cal-confirm-v1",
        "computed_from_raw_rows": True,
        "partition": partition_receipt,
        "baseline_receipt": baseline_receipt,
        "candidate_receipt": candidate_receipt,
        "decision": decision,
        "result": {key: value for key, value in result.items() if key != "rapid_policy_calls"},
    }


def frozen_threshold_document(
    selected_document: Mapping[str, Any],
    cal_fit_receipt_path: pathlib.Path,
    cal_confirm_receipt_path: pathlib.Path,
    gates_path: pathlib.Path,
) -> dict[str, Any]:
    fit = json.loads(cal_fit_receipt_path.read_text(encoding="utf-8"))
    confirm = json.loads(cal_confirm_receipt_path.read_text(encoding="utf-8"))
    if bool(confirm["decision"]["abort"]):
        raise task_c_trace.TaskCTraceError("cannot freeze RAPID thresholds after a failed Cal-Confirm gate")
    selected_quantile = float(fit["selection"]["selected_motion_quantile"])
    if float(selected_document["motion_quantile"]) != selected_quantile:
        raise task_c_trace.TaskCTraceError("selected threshold document does not match computed Cal-Fit winner")
    document = dict(selected_document)
    document["freeze"] = {
        "selection_computed_from_raw_rows": True,
        "selected_motion_quantile": selected_quantile,
        "cal_fit_receipt": str(cal_fit_receipt_path.resolve()),
        "cal_fit_receipt_sha256": task_c_trace.sha256_file(cal_fit_receipt_path),
        "cal_confirm_receipt": str(cal_confirm_receipt_path.resolve()),
        "cal_confirm_receipt_sha256": task_c_trace.sha256_file(cal_confirm_receipt_path),
        "cal_confirm_passed": True,
        "calib_gates": str(gates_path.resolve()),
        "calib_gates_sha256": task_c_trace.sha256_file(gates_path),
        "eval_rows_observed": 0,
        "retuning_after_cal_confirm": False,
    }
    return document


def baseline_reconciliation(
    baseline_root: pathlib.Path,
    gates: Mapping[str, Any],
) -> dict[str, Any]:
    verify_condition_receipt(baseline_root)
    baseline_manifest = json.loads((baseline_root / "run_manifest.json").read_text(encoding="utf-8"))
    if baseline_manifest.get("calibration_stage") != "final_eval":
        raise task_c_trace.TaskCTraceError("baseline reconciliation requires a final_eval manifest")
    final_partition = gates["calibration_partitions"]["final_eval"]
    baseline_episodes = task_c_trace.read_jsonl(baseline_root / "episodes.jsonl")
    validate_partition(baseline_episodes, final_partition, label="final always-infer baseline")
    sealed_root = pathlib.Path(gates["baseline_reconciliation"]["sealed_c1_faac_only_root"])
    sealed_receipt = verify_condition_receipt(sealed_root)
    sealed_episodes = task_c_trace.read_jsonl(sealed_root / "episodes.jsonl")
    sealed_rate = sum(bool(row["success"]) for row in sealed_episodes) / len(sealed_episodes)
    baseline_rate = sum(bool(row["success"]) for row in baseline_episodes) / len(baseline_episodes)
    delta = baseline_rate - sealed_rate
    tolerance = float(gates["baseline_reconciliation"]["max_abs_success_rate_delta"])
    passed = abs(delta) <= tolerance
    receipt = {
        "hard_abort_on_failure": True,
        "passed": passed,
        "sealed_c1_receipt": sealed_receipt,
        "sealed_c1_success_rate": sealed_rate,
        "rapid_always_infer_success_rate": baseline_rate,
        "success_rate_delta": delta,
        "absolute_success_rate_delta": abs(delta),
        "max_abs_success_rate_delta": tolerance,
    }
    if not passed:
        raise task_c_trace.TaskCTraceError(f"baseline-vs-sealed-C1 reconciliation breached: {receipt}")
    return receipt


def discordant_pair_power(
    *,
    n_pairs: int,
    noninferiority_margin: float,
    assumed_discordant_rates: Sequence[float],
) -> dict[str, Any]:
    z_alpha = 1.6448536269514722
    scenarios: list[dict[str, float]] = []
    for discordant_rate in assumed_discordant_rates:
        standard_error = math.sqrt(float(discordant_rate) / n_pairs)
        standardized = (0.0 - noninferiority_margin) / standard_error - z_alpha
        power = 0.5 * (1.0 + math.erf(standardized / math.sqrt(2.0)))
        scenarios.append(
            {
                "assumed_discordant_pair_rate": float(discordant_rate),
                "assumed_true_success_delta": 0.0,
                "normal_approx_one_sided_power": power,
            }
        )
    return {
        "method": "paired-difference normal approximation",
        "n_pairs": int(n_pairs),
        "one_sided_alpha": 0.05,
        "noninferiority_margin": float(noninferiority_margin),
        "scenarios": scenarios,
        "caveat": "Power depends on the discordant-pair rate; scenarios were committed before final aggregation.",
    }


def _write_sha_manifest(root: pathlib.Path) -> dict[str, str]:
    manifest = root / "SHA256SUMS"
    files = sorted(path for path in root.rglob("*") if path.is_file() and path != manifest)
    manifest.write_text(
        "".join(f"{task_c_trace.sha256_file(path)}  {path.relative_to(root).as_posix()}\n" for path in files),
        encoding="utf-8",
    )
    return {"path": str(manifest), "sha256": task_c_trace.sha256_file(manifest)}


def aggregate_final_eval(
    baseline_root: pathlib.Path,
    candidate_root: pathlib.Path,
    gates: Mapping[str, Any],
    output_root: pathlib.Path,
) -> dict[str, Any]:
    baseline_receipt = verify_condition_receipt(baseline_root)
    candidate_receipt = verify_condition_receipt(candidate_root)
    baseline_manifest = json.loads((baseline_root / "run_manifest.json").read_text(encoding="utf-8"))
    candidate_manifest = json.loads((candidate_root / "run_manifest.json").read_text(encoding="utf-8"))
    path_equivalence = assert_eval_arm_equivalence(baseline_manifest, candidate_manifest)
    reconciliation = baseline_reconciliation(baseline_root, gates)
    final_partition = gates["calibration_partitions"]["final_eval"]
    baseline = task_c_trace.read_jsonl(baseline_root / "episodes.jsonl")
    candidate = task_c_trace.read_jsonl(candidate_root / "episodes.jsonl")
    validate_partition(baseline, final_partition, label="final always-infer baseline")
    validate_partition(candidate, final_partition, label="final RAPID candidate")
    overall = _raw_result_from_roots(
        baseline_root,
        candidate_root,
        gates,
        bootstrap_seed=int(gates["bootstrap"]["seed"]) + 20_000,
    )
    rapid_rows = overall.pop("rapid_policy_calls")
    per_task: dict[str, Any] = {}
    candidate_calls = task_c_trace.read_jsonl(candidate_root / "server_trace" / "wm_calls.jsonl")
    candidate_steps = task_c_trace.read_jsonl(candidate_root / "steps_raw.jsonl")
    for task_id in sorted({int(row["task_id"]) for row in baseline}):
        task_result = paired_raw_result(
            [row for row in baseline if int(row["task_id"]) == task_id],
            [row for row in candidate if int(row["task_id"]) == task_id],
            [row for row in candidate_calls if int(row["task_id"]) == task_id],
            [row for row in candidate_steps if int(row["task_id"]) == task_id],
            bootstrap_samples=int(gates["bootstrap"]["samples"]),
            bootstrap_seed=int(gates["bootstrap"]["seed"]) + 21_000 + task_id,
            noninferiority_margin=float(gates["final_eval"]["noninferiority_margin"]),
            kappa_delta=float(gates["final_eval"]["kappa_shadow_delta"]),
        )
        task_result.pop("rapid_policy_calls")
        per_task[str(task_id)] = task_result
    output_root.mkdir(parents=True, exist_ok=False)
    rapid_path = output_root / "rapid_calls.jsonl"
    with rapid_path.open("wb") as stream:
        for row in rapid_rows:
            stream.write(task_c_trace.canonical_json_bytes(row) + b"\n")
    summary = {
        "schema_version": "jetson-pi-task-c-rapid-final-v1",
        "experiment": "RAPID_success_anchored_AB_libero_spatial_K9",
        "eval_matrix": {
            "suite": "libero_spatial",
            "trigger_k": 9,
            "paired_episodes_per_arm": len(baseline),
            "total_episodes": len(baseline) + len(candidate),
            "seed": 42,
            "task_ids": list(range(10)),
            "init_indices": [0, 29],
        },
        "wm_still_required_for_faac": True,
        "baseline_reconciliation": reconciliation,
        "arm_path_equivalence": path_equivalence,
        "success_primary": overall,
        "per_task": per_task,
        "discordant_pair_power": discordant_pair_power(
            n_pairs=len(baseline),
            noninferiority_margin=float(gates["final_eval"]["noninferiority_margin"]),
            assumed_discordant_rates=gates["final_eval"]["power_assumed_discordant_pair_rates"],
        ),
        "bootstrap_ci_method": gates["bootstrap"]["method"],
        "condition_receipts": {"always_infer": baseline_receipt, "rapid": candidate_receipt},
        "raw_rapid_calls": {
            "path": rapid_path.name,
            "rows": len(rapid_rows),
            "sha256": task_c_trace.sha256_file(rapid_path),
        },
    }
    write_json(output_root / "summary.json", summary)
    write_json(
        output_root / "run_manifest.json",
        {
            "schema_version": summary["schema_version"],
            "status": "complete",
            "summary_sha256": task_c_trace.sha256_file(output_root / "summary.json"),
            "thresholds_sha256": baseline_manifest["rapid_thresholds_sha256"],
            "calib_gates_sha256": baseline_manifest["rapid_calib_gates"]["sha256"],
            "condition_receipts": summary["condition_receipts"],
        },
    )
    manifest_receipt = _write_sha_manifest(output_root)
    return {**summary, "receipt_manifest": manifest_receipt}


def write_json(path: pathlib.Path, value: Any) -> None:
    task_c_trace.write_json_atomic(path, value)


def _load_condition_rows(root: pathlib.Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    return (
        task_c_trace.read_jsonl(root / "episodes.jsonl"),
        task_c_trace.read_jsonl(root / "server_trace" / "wm_calls.jsonl"),
        task_c_trace.read_jsonl(root / "steps_raw.jsonl"),
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    derive = subparsers.add_parser("derive-ladder")
    derive.add_argument("--baseline", type=pathlib.Path, required=True)
    derive.add_argument("--gates", type=pathlib.Path, required=True)
    derive.add_argument("--out", type=pathlib.Path, required=True)
    pair = subparsers.add_parser("pair-raw")
    pair.add_argument("--baseline", type=pathlib.Path, required=True)
    pair.add_argument("--candidate", type=pathlib.Path, required=True)
    pair.add_argument("--gates", type=pathlib.Path, required=True)
    pair.add_argument("--out", type=pathlib.Path, required=True)
    fit = subparsers.add_parser("cal-fit")
    fit.add_argument("--baseline", type=pathlib.Path, required=True)
    fit.add_argument("--candidate", action="append", required=True, help="MOTION_QUANTILE=CONDITION_ROOT")
    fit.add_argument("--gates", type=pathlib.Path, required=True)
    fit.add_argument("--out", type=pathlib.Path, required=True)
    confirm = subparsers.add_parser("cal-confirm")
    confirm.add_argument("--baseline", type=pathlib.Path, required=True)
    confirm.add_argument("--candidate", type=pathlib.Path, required=True)
    confirm.add_argument("--gates", type=pathlib.Path, required=True)
    confirm.add_argument("--out", type=pathlib.Path, required=True)
    freeze = subparsers.add_parser("freeze-thresholds")
    freeze.add_argument("--selected-document", type=pathlib.Path, required=True)
    freeze.add_argument("--cal-fit-receipt", type=pathlib.Path, required=True)
    freeze.add_argument("--cal-confirm-receipt", type=pathlib.Path, required=True)
    freeze.add_argument("--gates", type=pathlib.Path, required=True)
    freeze.add_argument("--out", type=pathlib.Path, required=True)
    reconcile = subparsers.add_parser("reconcile-baseline")
    reconcile.add_argument("--baseline", type=pathlib.Path, required=True)
    reconcile.add_argument("--gates", type=pathlib.Path, required=True)
    reconcile.add_argument("--out", type=pathlib.Path, required=True)
    aggregate = subparsers.add_parser("aggregate-eval")
    aggregate.add_argument("--baseline", type=pathlib.Path, required=True)
    aggregate.add_argument("--candidate", type=pathlib.Path, required=True)
    aggregate.add_argument("--gates", type=pathlib.Path, required=True)
    aggregate.add_argument("--out", type=pathlib.Path, required=True)
    seal = subparsers.add_parser("verify-threshold-seal")
    seal.add_argument("--repo", type=pathlib.Path, required=True)
    seal.add_argument("--thresholds", type=pathlib.Path, required=True)
    seal.add_argument("--sha256s", type=pathlib.Path, required=True)
    init_states = subparsers.add_parser("verify-init-states")
    init_states.add_argument("--suite", default="libero_spatial")
    init_states.add_argument("--minimum", type=int, default=50)
    init_states.add_argument("--out", type=pathlib.Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.command == "verify-threshold-seal":
        result: Any = verify_threshold_sha_manifest(args.repo, args.thresholds, args.sha256s)
    elif args.command == "verify-init-states":
        from libero.libero import benchmark  # pyright: ignore[reportMissingImports]

        suite = benchmark.get_benchmark_dict()[args.suite]()
        counts = {str(task_id): len(suite.get_task_init_states(task_id)) for task_id in range(suite.n_tasks)}
        too_short = {key: value for key, value in counts.items() if value < args.minimum}
        if too_short:
            raise task_c_trace.TaskCTraceError(
                f"{args.suite} tasks need at least {args.minimum} init states: {too_short}"
            )
        result = {
            "suite": args.suite,
            "minimum_required": args.minimum,
            "task_count": suite.n_tasks,
            "counts": counts,
            "all_tasks_pass": True,
        }
        write_json(args.out, result)
    else:
        gates = load_gate_config(args.gates)
        if args.command == "derive-ladder":
            episodes = task_c_trace.read_jsonl(args.baseline / "episodes.jsonl")
            observations = task_c_trace.read_jsonl(args.baseline / "proprio_observations.jsonl")
            ladder = derive_threshold_ladder(
                episodes,
                observations,
                motion_quantiles=gates["ladder"]["motion_quantiles"],
                transition_cooldown_steps=gates["ladder"]["transition_cooldown_steps"],
                dt=gates["ladder"]["dt"],
            )
            args.out.mkdir(parents=True, exist_ok=False)
            for document in ladder:
                quantile = round(float(document["motion_quantile"]) * 100)
                write_json(args.out / f"rapid_thresholds_q{quantile:02d}.json", document)
            result = {"documents": len(ladder), "out": str(args.out)}
        elif args.command == "pair-raw":
            baseline, _, _ = _load_condition_rows(args.baseline)
            candidate, calls, steps = _load_condition_rows(args.candidate)
            result = paired_raw_result(
                baseline,
                candidate,
                calls,
                steps,
                bootstrap_samples=int(gates["bootstrap"]["samples"]),
                bootstrap_seed=int(gates["bootstrap"]["seed"]),
                noninferiority_margin=float(gates["final_eval"]["noninferiority_margin"]),
                kappa_delta=float(gates["final_eval"]["kappa_shadow_delta"]),
            )
            persisted = {key: value for key, value in result.items() if key != "rapid_policy_calls"}
            write_json(args.out, persisted)
        elif args.command == "cal-fit":
            candidates: dict[float, pathlib.Path] = {}
            for spec in args.candidate:
                quantile, separator, root = spec.partition("=")
                if separator != "=" or not root:
                    raise ValueError(f"invalid --candidate {spec!r}; expected QUANTILE=ROOT")
                candidates[float(quantile)] = pathlib.Path(root)
            result = cal_fit_from_roots(args.baseline, candidates, gates)
            write_json(args.out, result)
        elif args.command == "cal-confirm":
            result = cal_confirm_from_roots(args.baseline, args.candidate, gates)
            write_json(args.out, result)
            if result["decision"]["abort"]:
                raise task_c_trace.TaskCTraceError(
                    f"Cal-Confirm hard abort: {result['decision']['abort_reasons']}"
                )
        elif args.command == "freeze-thresholds":
            selected = json.loads(args.selected_document.read_text(encoding="utf-8"))
            result = frozen_threshold_document(
                selected,
                args.cal_fit_receipt,
                args.cal_confirm_receipt,
                args.gates,
            )
            write_json(args.out, result)
        elif args.command == "reconcile-baseline":
            result = baseline_reconciliation(args.baseline, gates)
            write_json(args.out, result)
        else:
            result = aggregate_final_eval(args.baseline, args.candidate, gates, args.out)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
