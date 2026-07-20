"""Finalize and analyze paired Jetson-PI Task-C scheduling receipts."""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
import json
import os
import pathlib
from typing import Any

import numpy as np
from openpi_client import task_c_trace

MU_ROWS_PER_SHARD = 4096
NONINFERIORITY_MARGIN = -0.05
C3_SUITES = ("libero_object", "libero_goal", "libero_10")


def _write_jsonl(path: pathlib.Path, records: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("wb") as stream:
        for record in records:
            stream.write(task_c_trace.canonical_json_bytes(dict(record)) + b"\n")
    os.replace(temporary, path)


def _write_sha_manifest(root: pathlib.Path, *, filename: str = "SHA256SUMS") -> tuple[pathlib.Path, str]:
    manifest = root / filename
    excluded = {manifest.resolve()}
    files = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.resolve() not in excluded and ".tmp-" not in path.name
    )
    lines = [f"{task_c_trace.sha256_file(path)}  {path.relative_to(root).as_posix()}" for path in files]
    temporary = manifest.with_name(f".{manifest.name}.tmp-{os.getpid()}")
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.replace(temporary, manifest)
    return manifest, task_c_trace.sha256_file(manifest)


def _verify_sha_manifest(root: pathlib.Path, *, filename: str = "SHA256SUMS") -> str:
    root = root.resolve()
    manifest = root / filename
    if not manifest.is_file():
        raise task_c_trace.TaskCTraceError(f"missing receipt manifest: {manifest}")
    expected: dict[str, str] = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        digest, separator, relative = line.partition("  ")
        try:
            valid_digest = len(digest) == 64 and int(digest, 16) >= 0
        except ValueError:
            valid_digest = False
        relative_path = pathlib.PurePosixPath(relative)
        if (
            separator != "  "
            or not valid_digest
            or not relative
            or relative_path.is_absolute()
            or ".." in relative_path.parts
            or relative in expected
        ):
            raise task_c_trace.TaskCTraceError(f"invalid receipt manifest line: {line!r}")
        expected[relative] = digest
    actual_paths = {
        path.relative_to(root).as_posix(): path
        for path in root.rglob("*")
        if path.is_file() and path.resolve() != manifest.resolve() and ".tmp-" not in path.name
    }
    if set(actual_paths) != set(expected):
        raise task_c_trace.TaskCTraceError(
            f"receipt file set mismatch: missing={sorted(set(expected) - set(actual_paths))}, "
            f"extra={sorted(set(actual_paths) - set(expected))}"
        )
    for relative, path in sorted(actual_paths.items()):
        actual = task_c_trace.sha256_file(path)
        if actual != expected[relative]:
            raise task_c_trace.TaskCTraceError(f"SHA-256 mismatch for {path}: {actual} != {expected[relative]}")
    return task_c_trace.sha256_file(manifest)


def _verify_c3_condition_contract(root: pathlib.Path, *, condition: str, suite: str) -> None:
    manifest_path = root / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {
        "status": "complete",
        "condition": condition,
        "suite": suite,
        "experiment": f"C3_{suite}_k9",
        "trigger_k": 9,
        "action_horizon": 10,
        "seed": 42,
        "task_id_start": 0,
        "task_id_count": 10,
        "trials_per_task": 30,
        "expected_episodes": 300,
        "actual_episodes": 300,
        "overlap": 1,
        "wm_delta_t": 1.0,
        "confidence_schedule": condition == "kappa_0p4",
        "kappa_delta": 0.4 if condition == "kappa_0p4" else None,
    }
    for field, expected_value in expected.items():
        if manifest.get(field) != expected_value:
            raise task_c_trace.TaskCTraceError(
                f"C3 {condition} manifest {field}={manifest.get(field)!r} != {expected_value!r}"
            )
    world_model = manifest.get("world_model")
    if not isinstance(world_model, dict):
        raise task_c_trace.TaskCTraceError(f"C3 {condition} manifest has no world_model receipt")
    if world_model.get("training_suite") != "libero_spatial" or world_model.get("out_of_training_suite") is not True:
        raise task_c_trace.TaskCTraceError(f"C3 {condition} is not labeled WM-out-of-training-suite")


def _unique_by(records: Sequence[Mapping[str, Any]], key: str) -> dict[str, Mapping[str, Any]]:
    out: dict[str, Mapping[str, Any]] = {}
    for record in records:
        value = str(record[key])
        if value in out:
            raise task_c_trace.TaskCTraceError(f"duplicate {key}: {value}")
        out[value] = record
    return out


def _count_kappa_decisions(calls: Sequence[Mapping[str, Any]]) -> tuple[int, int]:
    invalid = [str(call.get("decision")) for call in calls if call.get("decision") not in {"skip_vlm", "infer_vlm"}]
    if invalid:
        raise task_c_trace.TaskCTraceError(f"unexpected eligible decision labels: {sorted(set(invalid))}")
    skipped = sum(call["decision"] == "skip_vlm" for call in calls)
    reinferred = sum(call["decision"] == "infer_vlm" for call in calls)
    return skipped, reinferred


def _validate_mu_and_write_shards(
    output_root: pathlib.Path,
    wm_calls: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    server_root = output_root / "server_trace"
    indexes = task_c_trace.read_jsonl(server_root / "mu_raw" / "index.jsonl")
    wm_by_id = _unique_by(wm_calls, "wm_call_id")
    index_by_id = _unique_by(indexes, "wm_call_id")
    if set(wm_by_id) != set(index_by_id):
        raise task_c_trace.TaskCTraceError("wm_calls and mu_raw index have different wm_call_id sets")

    receipt: dict[str, Any] = {}
    bytes_per_row = int(np.prod(task_c_trace.MU_SHAPE)) * task_c_trace.MU_DTYPE.itemsize
    for split in ("calibration", "eval"):
        split_indexes = sorted(
            (item for item in indexes if item["split"] == split),
            key=lambda item: int(item["raw_row"]),
        )
        for expected_row, item in enumerate(split_indexes):
            if int(item["raw_row"]) != expected_row:
                raise task_c_trace.TaskCTraceError(
                    f"non-contiguous {split} mu raw rows: got {item['raw_row']} at index {expected_row}"
                )
        raw_path = server_root / "mu_raw" / f"{split}.f16"
        if not raw_path.is_file():
            raise task_c_trace.TaskCTraceError(f"missing {split} mu raw file")
        expected_size = len(split_indexes) * bytes_per_row
        if raw_path.stat().st_size != expected_size:
            raise task_c_trace.TaskCTraceError(
                f"{split} mu raw size {raw_path.stat().st_size} != expected {expected_size}"
            )
        raw = np.memmap(raw_path, mode="r", dtype=task_c_trace.MU_DTYPE)
        raw = raw.reshape(len(split_indexes), *task_c_trace.MU_SHAPE)
        shard_dir = output_root / "mu" / split
        shard_dir.mkdir(parents=True, exist_ok=True)
        remapped: list[dict[str, Any]] = []
        shard_receipts: list[dict[str, Any]] = []
        for start in range(0, len(split_indexes), MU_ROWS_PER_SHARD):
            stop = min(start + MU_ROWS_PER_SHARD, len(split_indexes))
            shard_index = start // MU_ROWS_PER_SHARD
            shard_path = shard_dir / f"mu-{shard_index:05d}.npy"
            np.save(shard_path, np.asarray(raw[start:stop], dtype=task_c_trace.MU_DTYPE), allow_pickle=False)
            loaded = np.load(shard_path, mmap_mode="r", allow_pickle=False)
            if loaded.shape != (stop - start, *task_c_trace.MU_SHAPE) or loaded.dtype != task_c_trace.MU_DTYPE:
                raise task_c_trace.TaskCTraceError(f"mu shard validation failed: {shard_path}")
            shard_sha = task_c_trace.sha256_file(shard_path)
            shard_receipts.append(
                {
                    "path": shard_path.relative_to(output_root).as_posix(),
                    "sha256": shard_sha,
                    "rows": stop - start,
                    "shape": list(loaded.shape),
                    "dtype": str(loaded.dtype),
                }
            )
            for local_row, item in enumerate(split_indexes[start:stop]):
                raw_bytes = np.ascontiguousarray(loaded[local_row]).tobytes(order="C")
                if task_c_trace.sha256_bytes(raw_bytes) != item["mu_sha256"]:
                    raise task_c_trace.TaskCTraceError(f"mu row SHA mismatch for {item['wm_call_id']} in {shard_path}")
                remapped.append(
                    {
                        **item,
                        "shard": shard_path.relative_to(output_root).as_posix(),
                        "shard_row": local_row,
                    }
                )
        index_path = shard_dir / "index.jsonl"
        _write_jsonl(index_path, remapped)
        split_manifest, split_manifest_sha = _write_sha_manifest(shard_dir)
        receipt[split] = {
            "rows": len(split_indexes),
            "unique_trajectories": len({item["trajectory_id"] for item in split_indexes}),
            "shards": shard_receipts,
            "index": index_path.relative_to(output_root).as_posix(),
            "index_sha256": task_c_trace.sha256_file(index_path),
            "manifest": split_manifest.relative_to(output_root).as_posix(),
            "manifest_sha256": split_manifest_sha,
        }
    calib_trajectories = {item["trajectory_id"] for item in indexes if item["split"] == "calibration"}
    eval_trajectories = {item["trajectory_id"] for item in indexes if item["split"] == "eval"}
    overlap = sorted(calib_trajectories.intersection(eval_trajectories))
    if overlap:
        raise task_c_trace.TaskCTraceError(f"mu trajectory leakage across calibration/eval: {overlap[:5]}")
    receipt["trajectory_overlap_count"] = 0
    return receipt


def _phase_counts(
    eligible_calls: Sequence[Mapping[str, Any]],
    phase_by_key: Mapping[tuple[str, int], str],
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for phase in ("approach", "contact"):
        calls = [
            call
            for call in eligible_calls
            if phase_by_key[(str(call["trajectory_id"]), int(call["env_step"]))] == phase
        ]
        skipped, reinferred = _count_kappa_decisions(calls)
        output[phase] = {
            "eligible_decisions": len(calls),
            "skip_decisions": skipped,
            "infer_decisions": reinferred,
            "kappa_forced_reinfer_count": reinferred,
            "kappa_ever_forced_reinfer": reinferred > 0,
            "raw_skip_rate": task_c_trace.ratio(skipped, len(calls)),
        }
    return output


def finalize_condition(output_root: pathlib.Path) -> dict[str, Any]:
    from scripts import task_c_rapid

    output_root = output_root.resolve()
    manifest_path = output_root / "run_manifest.json"
    if not manifest_path.is_file():
        raise task_c_trace.TaskCTraceError(f"missing run manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    condition = str(manifest["condition"])
    expected_episodes = int(manifest["expected_episodes"])

    episodes = task_c_trace.read_jsonl(output_root / "episodes.jsonl")
    steps = task_c_trace.read_jsonl(output_root / "steps_raw.jsonl")
    wm_calls = task_c_trace.read_jsonl(output_root / "server_trace" / "wm_calls.jsonl")
    policy_calls = task_c_trace.read_jsonl(output_root / "server_trace" / "policy_calls.jsonl")
    proprio_path = output_root / "proprio_observations.jsonl"
    proprio_observations = task_c_trace.read_jsonl(proprio_path) if proprio_path.is_file() else []
    if len(episodes) != expected_episodes:
        raise task_c_trace.TaskCTraceError(f"episode count {len(episodes)} != expected {expected_episodes}")
    episode_by_trajectory = _unique_by(episodes, "trajectory_id")
    if any(item.get("error") for item in episodes):
        raise task_c_trace.TaskCTraceError("at least one episode contains an evaluation error")
    if any(item["condition"] != condition for item in [*episodes, *steps, *wm_calls, *policy_calls]):
        raise task_c_trace.TaskCTraceError("mixed condition values in condition receipt")
    _write_jsonl(output_root / "wm_calls.jsonl", wm_calls)

    step_counts: dict[str, int] = defaultdict(int)
    for step in steps:
        step_counts[str(step["trajectory_id"])] += 1
    for trajectory_id, episode in episode_by_trajectory.items():
        if step_counts[trajectory_id] != int(episode["control_steps"]):
            raise task_c_trace.TaskCTraceError(
                f"step count mismatch for {trajectory_id}: {step_counts[trajectory_id]} != {episode['control_steps']}"
            )

    phase_by_key = task_c_trace.contact_phase_by_trajectory(steps)
    wm_by_step: dict[tuple[str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for call in wm_calls:
        key = (str(call["trajectory_id"]), int(call["env_step"]))
        if key not in phase_by_key:
            raise task_c_trace.TaskCTraceError(f"WM call has no matching client step: {call['wm_call_id']} key={key}")
        wm_by_step[key].append(call)

    labeled_steps: list[dict[str, Any]] = []
    for step in steps:
        key = (str(step["trajectory_id"]), int(step["env_step"]))
        episode = episode_by_trajectory[str(step["trajectory_id"])]
        linked = sorted(wm_by_step.get(key, []), key=lambda call: int(call["wm_call_index"]))
        labeled_steps.append(
            {
                **step,
                "episode_success": bool(episode["success"]),
                "phase": phase_by_key[key],
                "contact_proxy": {
                    "rule": "first_of_two_consecutive_gripper_commands_gt_0",
                    "close_threshold": 0.0,
                    "sustained_steps": 2,
                },
                "wm_events": [
                    {
                        "wm_call_id": call["wm_call_id"],
                        "policy_call_id": call["policy_call_id"],
                        "round_index": call["round_index"],
                        "kappa": call["kappa"],
                        "decision": call["decision"],
                        "decision_eligible": call["decision_eligible"],
                        "c_tier0_ms": call["c_tier0_ms"],
                        "routing_policy": call.get("routing_policy"),
                        "rapid": call.get("rapid"),
                    }
                    for call in linked
                ],
            }
        )
    _write_jsonl(output_root / "steps_labeled.jsonl", labeled_steps)

    mu_receipt = _validate_mu_and_write_shards(output_root, wm_calls)
    measured_timing = [float(call["c_tier0_ms"]) for call in wm_calls if not call["timing_warmup"]]
    if not measured_timing:
        raise task_c_trace.TaskCTraceError("no measured C_tier0 calls remain after warmup exclusion")
    eligible = [call for call in wm_calls if call["decision_eligible"]]
    skips, reinfers = _count_kappa_decisions(eligible)
    rapid_calls = task_c_rapid._rapid_policy_rows(  # noqa: SLF001 - shared raw-row experiment seam
        wm_calls,
        kappa_delta=float(manifest.get("kappa_delta") or task_c_rapid.DEFAULT_KAPPA_DELTA),
    )
    if rapid_calls:
        _write_jsonl(output_root / "rapid_calls.jsonl", rapid_calls)
    rounds_by_policy: dict[str, int] = defaultdict(int)
    for call in wm_calls:
        rounds_by_policy[str(call["policy_call_id"])] += 1
    task_summary: dict[str, Any] = {}
    for task_id in sorted({int(item["task_id"]) for item in episodes}):
        task_eps = [item for item in episodes if int(item["task_id"]) == task_id]
        task_summary[str(task_id)] = {
            "episodes": len(task_eps),
            "successes": sum(bool(item["success"]) for item in task_eps),
            "success_rate": sum(bool(item["success"]) for item in task_eps) / len(task_eps),
        }
    summary = {
        "schema_version": task_c_trace.SCHEMA_VERSION,
        "condition": condition,
        "episodes": len(episodes),
        "successes": sum(bool(item["success"]) for item in episodes),
        "success_rate": sum(bool(item["success"]) for item in episodes) / len(episodes),
        "wm_still_required_for_faac": bool(manifest.get("wm_still_required_for_faac", True)),
        "per_task": task_summary,
        "policy_calls": {
            "total_vlm_calls": len(policy_calls),
            "wm_conditioned_vlm_calls": len(rounds_by_policy),
        },
        "scheduling": {
            "eligible_decisions": len(eligible),
            "skip_decisions": skips,
            "infer_decisions": reinfers,
            "kappa_forced_reinfer_count": reinfers,
            "kappa_ever_forced_reinfer": reinfers > 0,
            "raw_skip_rate": task_c_trace.ratio(skips, len(eligible)),
            "mean_wm_ae_rounds_per_wm_conditioned_vlm_call": (
                float(np.mean(list(rounds_by_policy.values()))) if rounds_by_policy else None
            ),
            "per_phase": _phase_counts(eligible, phase_by_key),
        },
        "c_tier0_ms": task_c_trace.percentile_summary(measured_timing),
        "c_tier0_warmup_calls_discarded": sum(bool(call["timing_warmup"]) for call in wm_calls),
        "rapid": (
            {
                "policy_call_decisions": len(rapid_calls),
                "shadow_skip_decisions": sum(call["rapid_decision"] == "skip" for call in rapid_calls),
                "raw_skip_rate": task_c_trace.ratio(
                    sum(call["rapid_decision"] == "skip" for call in rapid_calls), len(rapid_calls)
                ),
                "executed_skip_decisions": sum(call["executed_decision"] == "skip" for call in rapid_calls),
                "trigger_compute_us": task_c_trace.percentile_summary(
                    [float(call["trigger_compute_ns"]) / 1000.0 for call in rapid_calls]
                ),
                "per_step_decision_agreement_with_40m_kappa": {
                    "agreements": sum(bool(call["decision_agreement"]) for call in rapid_calls),
                    "disagreements": sum(not bool(call["decision_agreement"]) for call in rapid_calls),
                    "agreement_rate": task_c_trace.ratio(
                        sum(bool(call["decision_agreement"]) for call in rapid_calls), len(rapid_calls)
                    ),
                    "semantics": "first eligible 40M kappa gate at the same trigger env_step",
                    "coverage_rate": 1.0,
                },
                "raw_rows": "rapid_calls.jsonl",
            }
            if rapid_calls
            else None
        ),
        "rapid_proprio_observations": {
            "rows": len(proprio_observations),
            "sha256": task_c_trace.sha256_file(proprio_path) if proprio_observations else None,
        },
        "mu": mu_receipt,
    }
    task_c_trace.write_json_atomic(output_root / "summary.json", summary)
    manifest["status"] = "complete"
    manifest["actual_episodes"] = len(episodes)
    manifest["summary_sha256"] = task_c_trace.sha256_file(output_root / "summary.json")
    manifest["steps_labeled_sha256"] = task_c_trace.sha256_file(output_root / "steps_labeled.jsonl")
    task_c_trace.write_json_atomic(manifest_path, manifest)
    sha_path, sha = _write_sha_manifest(output_root)
    # A manifest cannot contain its own digest without a circular dependency.
    # Return the out-of-band digest to the caller, but keep the persisted files
    # exactly as covered by this authoritative manifest.
    return {**summary, "receipt_manifest": sha_path.name, "receipt_manifest_sha256": sha}


def _decision_stats(
    calls: Sequence[Mapping[str, Any]],
    *,
    preserved_trajectories: set[str] | None = None,
    task_id: int | None = None,
    phase: str | None = None,
    phase_by_key: Mapping[tuple[str, int], str] | None = None,
) -> dict[str, Any]:
    selected = [call for call in calls if bool(call["decision_eligible"])]
    if preserved_trajectories is not None:
        selected = [call for call in selected if str(call["trajectory_id"]) in preserved_trajectories]
    if task_id is not None:
        selected = [call for call in selected if int(call["task_id"]) == task_id]
    if phase is not None:
        if phase_by_key is None:
            raise ValueError("phase_by_key is required for phase filtering")
        selected = [
            call for call in selected if phase_by_key[(str(call["trajectory_id"]), int(call["env_step"]))] == phase
        ]
    skipped, reinferred = _count_kappa_decisions(selected)
    return {
        "eligible_decisions": len(selected),
        "skip_decisions": skipped,
        "infer_decisions": reinferred,
        "kappa_forced_reinfer_count": reinferred,
        "kappa_ever_forced_reinfer": reinferred > 0,
        "skip_rate": task_c_trace.ratio(skipped, len(selected)),
    }


def _paired_result(
    baseline: Sequence[Mapping[str, Any]],
    candidate: Sequence[Mapping[str, Any]],
    calls: Sequence[Mapping[str, Any]],
    phase_by_key: Mapping[tuple[str, int], str],
    *,
    bootstrap_seed: int,
) -> dict[str, Any]:
    base_by_id = _unique_by(baseline, "trajectory_id")
    cand_by_id = _unique_by(candidate, "trajectory_id")
    if set(base_by_id) != set(cand_by_id):
        raise task_c_trace.TaskCTraceError("paired condition trajectory sets differ")
    trajectory_ids = sorted(base_by_id)
    base_success = [bool(base_by_id[key]["success"]) for key in trajectory_ids]
    cand_success = [bool(cand_by_id[key]["success"]) for key in trajectory_ids]
    mcnemar = task_c_trace.exact_mcnemar(base_success, cand_success)
    bootstrap = task_c_trace.paired_bootstrap_success_delta(
        base_success,
        cand_success,
        seed=bootstrap_seed,
    )
    success_rate_delta = float(bootstrap["success_rate_delta"])
    mcnemar_nonsignificant = float(mcnemar["p_value_two_sided_exact"]) >= 0.05
    mcnemar_no_evidence_of_harm = mcnemar_nonsignificant or success_rate_delta >= 0.0
    noninferior = float(bootstrap["lower_95_one_sided"]) >= NONINFERIORITY_MARGIN
    validity = mcnemar_no_evidence_of_harm and noninferior
    preserved = {key for key in trajectory_ids if bool(base_by_id[key]["success"]) and bool(cand_by_id[key]["success"])}
    raw = _decision_stats(calls)
    success_conditioned = _decision_stats(calls, preserved_trajectories=preserved)
    phases: dict[str, Any] = {}
    for phase in ("approach", "contact"):
        raw_phase = _decision_stats(calls, phase=phase, phase_by_key=phase_by_key)
        preserved_phase = _decision_stats(
            calls,
            preserved_trajectories=preserved,
            phase=phase,
            phase_by_key=phase_by_key,
        )
        phases[phase] = {
            "raw": raw_phase,
            "success_conditioned": preserved_phase,
            "deployable_valid_skip_rate": preserved_phase["skip_rate"] if validity else None,
        }
    return {
        "n_pairs": len(trajectory_ids),
        "baseline_successes": sum(base_success),
        "candidate_successes": sum(cand_success),
        "baseline_success_rate": sum(base_success) / len(base_success),
        "candidate_success_rate": sum(cand_success) / len(cand_success),
        "paired_mcnemar": mcnemar,
        "paired_bootstrap": bootstrap,
        "noninferiority_margin": NONINFERIORITY_MARGIN,
        "mcnemar_nonsignificant": mcnemar_nonsignificant,
        "mcnemar_no_evidence_of_harm": mcnemar_no_evidence_of_harm,
        "noninferior": noninferior,
        "validity_gate_pass": validity,
        "preserved_success_pairs": len(preserved),
        "raw_skip": raw,
        "success_conditioned_skip": success_conditioned,
        "deployable_valid_skip_rate": success_conditioned["skip_rate"] if validity else None,
        "per_phase": phases,
    }


def _phase_map_from_labeled_steps(root: pathlib.Path) -> dict[tuple[str, int], str]:
    steps = task_c_trace.read_jsonl(root / "steps_labeled.jsonl")
    return {(str(step["trajectory_id"]), int(step["env_step"])): str(step["phase"]) for step in steps}


def _aggregate_paired_experiment(
    output_root: pathlib.Path,
    condition_roots: Mapping[str, pathlib.Path],
    *,
    suite: str,
    experiment: str,
    required: set[str],
) -> dict[str, Any]:
    if set(condition_roots) != required:
        raise task_c_trace.TaskCTraceError(
            f"{experiment} requires condition roots {sorted(required)}, got {sorted(condition_roots)}"
        )
    output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    baseline = task_c_trace.read_jsonl(condition_roots["faac_only"] / "episodes.jsonl")

    expected_tasks = set(range(10))
    expected_trajectory_ids: set[str] | None = None
    for condition, root in sorted(condition_roots.items()):
        episodes = task_c_trace.read_jsonl(root / "episodes.jsonl")
        if len(episodes) != 300:
            raise task_c_trace.TaskCTraceError(
                f"{experiment} {condition} must contain exactly 300 episodes, got {len(episodes)}"
            )
        task_ids = {int(item["task_id"]) for item in episodes}
        if task_ids != expected_tasks:
            raise task_c_trace.TaskCTraceError(f"{experiment} {condition} task ids {sorted(task_ids)} != 0..9")
        if any(str(item["suite"]) != suite or int(item["seed"]) != 42 for item in episodes):
            raise task_c_trace.TaskCTraceError(f"{experiment} {condition} must use {suite} and seed 42")
        for task_id in sorted(expected_tasks):
            task_episodes = [item for item in episodes if int(item["task_id"]) == task_id]
            episode_indices = {int(item["episode_idx"]) for item in task_episodes}
            if len(task_episodes) != 30 or episode_indices != set(range(30)):
                raise task_c_trace.TaskCTraceError(
                    f"{experiment} {condition} task {task_id} must contain episode_idx 0..29 exactly once"
                )
        trajectory_ids = {str(item["trajectory_id"]) for item in episodes}
        if expected_trajectory_ids is None:
            expected_trajectory_ids = trajectory_ids
        elif trajectory_ids != expected_trajectory_ids:
            raise task_c_trace.TaskCTraceError(f"{experiment} {condition} is not paired to the baseline trajectory set")
    condition_summaries: dict[str, Any] = {}
    paired: dict[str, Any] = {}
    all_mu_indexes: list[dict[str, Any]] = []
    for condition, root in sorted(condition_roots.items()):
        condition_summaries[condition] = json.loads((root / "summary.json").read_text(encoding="utf-8"))
        for split in ("calibration", "eval"):
            indexes = task_c_trace.read_jsonl(root / "mu" / split / "index.jsonl")
            all_mu_indexes.extend({"condition": condition, "condition_root": str(root), **item} for item in indexes)
        if condition == "faac_only":
            continue
        candidate = task_c_trace.read_jsonl(root / "episodes.jsonl")
        calls = task_c_trace.read_jsonl(root / "server_trace" / "wm_calls.jsonl")
        phase_map = _phase_map_from_labeled_steps(root)
        result = _paired_result(
            baseline,
            candidate,
            calls,
            phase_map,
            bootstrap_seed=20260719 + int(condition[-1]),
        )
        per_task: dict[str, Any] = {}
        for task_id in sorted({int(item["task_id"]) for item in baseline}):
            base_task = [item for item in baseline if int(item["task_id"]) == task_id]
            cand_task = [item for item in candidate if int(item["task_id"]) == task_id]
            task_calls = [call for call in calls if int(call["task_id"]) == task_id]
            task_result = _paired_result(
                base_task,
                cand_task,
                task_calls,
                phase_map,
                bootstrap_seed=20260719 + 100 * task_id + int(condition[-1]),
            )
            if int(task_result["n_pairs"]) < 30:
                raise task_c_trace.TaskCTraceError(f"paired task {task_id} has fewer than 30 episodes")
            per_task[str(task_id)] = task_result
        result["per_task"] = per_task
        paired[condition] = result

    split_by_trajectory: dict[str, str] = {}
    for item in all_mu_indexes:
        trajectory_id = str(item["trajectory_id"])
        split = str(item["split"])
        previous = split_by_trajectory.setdefault(trajectory_id, split)
        if previous != split:
            raise task_c_trace.TaskCTraceError(f"mu split leakage for {trajectory_id}: {previous} vs {split}")
    mu_shards: dict[str, Any] = {"calibration": [], "eval": []}
    for condition, root in sorted(condition_roots.items()):
        for split in ("calibration", "eval"):
            for shard in sorted((root / "mu" / split).glob("mu-*.npy")):
                mu_shards[split].append(
                    {
                        "condition": condition,
                        "path": str(shard),
                        "sha256": task_c_trace.sha256_file(shard),
                        "rows": int(np.load(shard, mmap_mode="r", allow_pickle=False).shape[0]),
                    }
                )
    mu_manifest = {
        "schema_version": task_c_trace.SCHEMA_VERSION,
        "trajectory_split_rule": "episode_idx_even=calibration; episode_idx_odd=eval; condition excluded",
        "trajectory_overlap_count": 0,
        "unique_calibration_trajectories": sum(value == "calibration" for value in split_by_trajectory.values()),
        "unique_eval_trajectories": sum(value == "eval" for value in split_by_trajectory.values()),
        "shards": mu_shards,
    }
    task_c_trace.write_json_atomic(output_root / "mu_shards.json", mu_manifest)
    summary = {
        "schema_version": task_c_trace.SCHEMA_VERSION,
        "experiment": experiment,
        "suite": suite,
        "world_model_training_suite": "libero_spatial",
        "wm_out_of_training_suite": suite != "libero_spatial",
        "conditions": condition_summaries,
        "paired": paired,
        "mu_shards_manifest": "mu_shards.json",
        "mu_shards_manifest_sha256": task_c_trace.sha256_file(output_root / "mu_shards.json"),
    }
    task_c_trace.write_json_atomic(output_root / "summary.json", summary)
    run_manifest = {
        "schema_version": task_c_trace.SCHEMA_VERSION,
        "experiment": experiment,
        "suite": suite,
        "world_model_training_suite": "libero_spatial",
        "wm_out_of_training_suite": suite != "libero_spatial",
        "status": "complete",
        "condition_receipts": {
            condition: {
                "root": str(root),
                "manifest_sha256": task_c_trace.sha256_file(root / "SHA256SUMS"),
            }
            for condition, root in sorted(condition_roots.items())
        },
        "summary_sha256": task_c_trace.sha256_file(output_root / "summary.json"),
        "mu_shards_manifest_sha256": task_c_trace.sha256_file(output_root / "mu_shards.json"),
    }
    task_c_trace.write_json_atomic(output_root / "run_manifest.json", run_manifest)
    _, manifest_sha = _write_sha_manifest(output_root)
    return {**summary, "receipt_manifest_sha256": manifest_sha}


def aggregate_c1(c1_root: pathlib.Path, condition_roots: Mapping[str, pathlib.Path]) -> dict[str, Any]:
    return _aggregate_paired_experiment(
        c1_root,
        condition_roots,
        suite="libero_spatial",
        experiment="C1_libero_spatial_k9",
        required={"faac_only", "kappa_0p2", "kappa_0p4", "kappa_0p8"},
    )


def aggregate_paired_suite(
    output_root: pathlib.Path,
    condition_roots: Mapping[str, pathlib.Path],
    *,
    suite: str,
) -> dict[str, Any]:
    if suite not in C3_SUITES:
        raise task_c_trace.TaskCTraceError(f"C3 requires a harder LIBERO suite, got {suite!r}")
    required = {"faac_only", "kappa_0p4"}
    if set(condition_roots) != required:
        raise task_c_trace.TaskCTraceError(
            f"C3_{suite}_k9 requires condition roots {sorted(required)}, got {sorted(condition_roots)}"
        )
    for condition, root in condition_roots.items():
        _verify_sha_manifest(root)
        _verify_c3_condition_contract(root, condition=condition, suite=suite)
    return _aggregate_paired_experiment(
        output_root,
        condition_roots,
        suite=suite,
        experiment=f"C3_{suite}_k9",
        required={"faac_only", "kappa_0p4"},
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    finalize = subparsers.add_parser("finalize-condition")
    finalize.add_argument("output", type=pathlib.Path)
    aggregate = subparsers.add_parser("aggregate-c1")
    aggregate.add_argument("output", type=pathlib.Path)
    aggregate.add_argument("--faac-only", type=pathlib.Path, required=True)
    aggregate.add_argument("--kappa-0p2", type=pathlib.Path, required=True)
    aggregate.add_argument("--kappa-0p4", type=pathlib.Path, required=True)
    aggregate.add_argument("--kappa-0p8", type=pathlib.Path, required=True)
    paired = subparsers.add_parser("aggregate-paired-suite")
    paired.add_argument("output", type=pathlib.Path)
    paired.add_argument("--suite", choices=C3_SUITES, required=True)
    paired.add_argument("--faac-only", type=pathlib.Path, required=True)
    paired.add_argument("--kappa-0p4", type=pathlib.Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.command == "finalize-condition":
        result = finalize_condition(args.output)
    elif args.command == "aggregate-c1":
        result = aggregate_c1(
            args.output,
            {
                "faac_only": args.faac_only,
                "kappa_0p2": args.kappa_0p2,
                "kappa_0p4": args.kappa_0p4,
                "kappa_0p8": args.kappa_0p8,
            },
        )
    else:
        result = aggregate_paired_suite(
            args.output,
            {
                "faac_only": args.faac_only,
                "kappa_0p4": args.kappa_0p4,
            },
            suite=args.suite,
        )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
