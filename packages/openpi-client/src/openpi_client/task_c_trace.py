"""Fail-closed trace helpers for the Jetson-PI Task-C scheduling evaluation.

The module deliberately depends only on NumPy and the Python standard library so
the policy-server and LIBERO-client environments can write the same schema.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import pathlib
import threading
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np


SCHEMA_VERSION = "jetson-pi-task-c-v1"
TRACE_KEY = "openpi/task_c_trace"
MU_SHAPE = (4, 1024)
MU_DTYPE = np.dtype("<f2")
REQUIRED_CONTEXT_FIELDS = frozenset(
    {
        "run_id",
        "condition",
        "suite",
        "task_id",
        "episode_idx",
        "seed",
        "trajectory_id",
        "split",
        "env_step",
    }
)


class TaskCTraceError(RuntimeError):
    """Raised when trace provenance or accounting is incomplete."""


def canonical_trajectory_id(*, suite: str, task_id: int, episode_idx: int, seed: int) -> str:
    """Return the condition-independent key used for paired comparisons."""

    return f"{suite}/task-{task_id:02d}/seed-{seed}/init-{episode_idx:03d}"


def trajectory_split(*, task_id: int, episode_idx: int) -> str:
    """Deterministically split trajectories without consulting outcomes.

    Even initialization indices are calibration trajectories and odd indices are
    evaluation trajectories.  The condition is intentionally absent, so paired
    copies of a trajectory can never cross the split boundary.
    """

    del task_id
    return "calibration" if episode_idx % 2 == 0 else "eval"


def trace_context(
    *,
    run_id: str,
    condition: str,
    suite: str,
    task_id: int,
    episode_idx: int,
    seed: int,
    env_step: int,
) -> dict[str, Any]:
    """Build one validated client-to-server trace context."""

    context = {
        "run_id": str(run_id),
        "condition": str(condition),
        "suite": str(suite),
        "task_id": int(task_id),
        "episode_idx": int(episode_idx),
        "seed": int(seed),
        "trajectory_id": canonical_trajectory_id(
            suite=suite,
            task_id=int(task_id),
            episode_idx=int(episode_idx),
            seed=int(seed),
        ),
        "split": trajectory_split(task_id=int(task_id), episode_idx=int(episode_idx)),
        "env_step": int(env_step),
    }
    validate_trace_context(context, expected_condition=condition, expected_run_id=run_id)
    return context


def validate_trace_context(
    context: Mapping[str, Any] | None,
    *,
    expected_condition: str | None = None,
    expected_run_id: str | None = None,
) -> dict[str, Any]:
    """Normalize and validate a trace context, including its derived fields."""

    if not isinstance(context, Mapping):
        raise TaskCTraceError("Task-C trace mode requires a mapping trace context on every policy call")
    missing = sorted(REQUIRED_CONTEXT_FIELDS.difference(context))
    if missing:
        raise TaskCTraceError(f"Task-C trace context is missing fields: {missing}")
    out = dict(context)
    for key in ("run_id", "condition", "suite", "trajectory_id", "split"):
        if not isinstance(out[key], str) or not out[key]:
            raise TaskCTraceError(f"Task-C trace field {key!r} must be a non-empty string")
    for key in ("task_id", "episode_idx", "seed", "env_step"):
        if isinstance(out[key], bool):
            raise TaskCTraceError(f"Task-C trace field {key!r} must be an integer, not bool")
        out[key] = int(out[key])
    if min(out["task_id"], out["episode_idx"], out["env_step"]) < 0:
        raise TaskCTraceError("Task-C task_id, episode_idx, and env_step must be non-negative")
    expected_trajectory = canonical_trajectory_id(
        suite=out["suite"],
        task_id=out["task_id"],
        episode_idx=out["episode_idx"],
        seed=out["seed"],
    )
    if out["trajectory_id"] != expected_trajectory:
        raise TaskCTraceError(f"Task-C trajectory_id mismatch: {out['trajectory_id']!r} != {expected_trajectory!r}")
    expected_split = trajectory_split(task_id=out["task_id"], episode_idx=out["episode_idx"])
    if out["split"] != expected_split:
        raise TaskCTraceError(f"Task-C split mismatch: {out['split']!r} != {expected_split!r}")
    if expected_condition is not None and out["condition"] != expected_condition:
        raise TaskCTraceError(f"Task-C condition mismatch: {out['condition']!r} != {expected_condition!r}")
    if expected_run_id is not None and out["run_id"] != expected_run_id:
        raise TaskCTraceError(f"Task-C run_id mismatch: {out['run_id']!r} != {expected_run_id!r}")
    return out


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json_atomic(path: pathlib.Path, value: Any) -> None:
    """Write canonical JSON without exposing a partially written receipt."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_bytes(canonical_json_bytes(value) + b"\n")
    os.replace(temporary, path)


def _append_bytes(path: pathlib.Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o644)
    try:
        view = memoryview(payload)
        while view:
            written = os.write(fd, view)
            if written <= 0:
                raise TaskCTraceError(f"short write while appending {path}")
            view = view[written:]
    finally:
        os.close(fd)


def append_jsonl(path: pathlib.Path, record: Mapping[str, Any]) -> None:
    _append_bytes(path, canonical_json_bytes(dict(record)) + b"\n")


def read_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise TaskCTraceError(f"missing required JSONL file: {path}")
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise TaskCTraceError(f"invalid JSON at {path}:{line_no}: {exc}") from exc
            if not isinstance(item, dict):
                raise TaskCTraceError(f"expected JSON object at {path}:{line_no}")
            records.append(item)
    return records


class ServerTraceRecorder:
    """Synchronous, append-only policy-server trace recorder.

    The measured WM interval ends before the recorder receives the values.  The
    synchronous append therefore cannot leak trace I/O into ``c_tier0_ms``.
    """

    def __init__(
        self,
        root: pathlib.Path,
        *,
        run_id: str,
        condition: str,
        timing_warmup_calls: int = 30,
    ) -> None:
        if timing_warmup_calls < 0:
            raise ValueError("timing_warmup_calls must be non-negative")
        self.root = root.resolve()
        self.run_id = str(run_id)
        self.condition = str(condition)
        self.timing_warmup_calls = int(timing_warmup_calls)
        self._lock = threading.Lock()
        self._policy_call_index = 0
        self._wm_call_index = 0
        self._mu_rows = {"calibration": 0, "eval": 0}
        self.root.mkdir(parents=True, exist_ok=True)
        guarded = [
            self.root / "policy_calls.jsonl",
            self.root / "wm_calls.jsonl",
            self.root / "mu_raw" / "index.jsonl",
            self.root / "mu_raw" / "calibration.f16",
            self.root / "mu_raw" / "eval.f16",
        ]
        dirty = [str(path) for path in guarded if path.exists() and path.stat().st_size]
        if dirty:
            raise TaskCTraceError(f"refusing to append to non-empty Task-C server trace: {dirty}")

    @classmethod
    def from_env(cls) -> ServerTraceRecorder | None:
        raw_root = os.environ.get("OPENPI_TASK_C_TRACE_ROOT")
        if not raw_root:
            return None
        run_id = os.environ.get("OPENPI_TASK_C_RUN_ID")
        condition = os.environ.get("OPENPI_TASK_C_CONDITION")
        if not run_id or not condition:
            raise TaskCTraceError("OPENPI_TASK_C_TRACE_ROOT requires OPENPI_TASK_C_RUN_ID and OPENPI_TASK_C_CONDITION")
        warmup = int(os.environ.get("OPENPI_TASK_C_TIMING_WARMUP_CALLS", "30"))
        return cls(pathlib.Path(raw_root), run_id=run_id, condition=condition, timing_warmup_calls=warmup)

    def begin_policy_call(self, context: Mapping[str, Any] | None, *, kind: str) -> str:
        ctx = validate_trace_context(
            context,
            expected_condition=self.condition,
            expected_run_id=self.run_id,
        )
        if kind not in {"plain_vlm", "faac_refresh", "kappa_schedule", "rapid_schedule"}:
            raise TaskCTraceError(f"unsupported Task-C policy-call kind: {kind!r}")
        with self._lock:
            idx = self._policy_call_index
            self._policy_call_index += 1
            call_id = f"{self.condition}:policy:{idx:08d}"
            append_jsonl(
                self.root / "policy_calls.jsonl",
                {
                    "schema_version": SCHEMA_VERSION,
                    "policy_call_id": call_id,
                    "policy_call_index": idx,
                    "kind": kind,
                    "is_vlm_call": True,
                    **ctx,
                },
            )
        return call_id

    def record_wm_call(
        self,
        context: Mapping[str, Any] | None,
        *,
        policy_call_id: str | None,
        round_index: int,
        mu: np.ndarray,
        kappa: float,
        wm_forward_kappa_ms: float,
        kappa_host_check_ms: float,
        kappa_decision_ms: float,
        decision: str,
        decision_eligible: bool,
        action_expert_executed: bool,
        routing_policy: str | None = None,
        rapid: Mapping[str, Any] | None = None,
    ) -> str:
        ctx = validate_trace_context(
            context,
            expected_condition=self.condition,
            expected_run_id=self.run_id,
        )
        if not policy_call_id or not policy_call_id.startswith(f"{self.condition}:policy:"):
            raise TaskCTraceError(f"invalid Task-C policy_call_id: {policy_call_id!r}")
        if round_index < 0:
            raise TaskCTraceError("round_index must be non-negative")
        if decision not in {"faac_refresh", "seed_round", "skip_vlm", "infer_vlm"}:
            raise TaskCTraceError(f"unsupported Task-C decision: {decision!r}")
        if (round_index == 0) == bool(decision_eligible):
            raise TaskCTraceError("only WM rounds after round zero may be scheduling-decision eligible")
        if not math.isfinite(float(kappa)):
            raise TaskCTraceError("non-finite kappa")
        if not math.isfinite(float(wm_forward_kappa_ms)) or wm_forward_kappa_ms <= 0:
            raise TaskCTraceError("wm_forward_kappa_ms must be finite and > 0")
        if not math.isfinite(float(kappa_host_check_ms)) or kappa_host_check_ms < 0:
            raise TaskCTraceError("kappa_host_check_ms must be finite and >= 0")
        if not math.isfinite(float(kappa_decision_ms)) or kappa_decision_ms < 0:
            raise TaskCTraceError("kappa_decision_ms must be finite and >= 0")
        rapid_fields: dict[str, Any] = {}
        if routing_policy is not None:
            if routing_policy not in {"kappa", "always_infer", "rapid"}:
                raise TaskCTraceError(f"unsupported Task-C routing_policy: {routing_policy!r}")
            rapid_fields["routing_policy"] = routing_policy
        if rapid is not None:
            rapid_record = dict(rapid)
            if rapid_record.get("decision") not in {"skip", "infer"}:
                raise TaskCTraceError("RAPID trace decision must be skip or infer")
            compute_ns = rapid_record.get("trigger_compute_ns")
            if isinstance(compute_ns, bool) or not isinstance(compute_ns, int) or compute_ns < 0:
                raise TaskCTraceError("RAPID trigger_compute_ns must be a non-negative integer")
            rapid_fields["rapid"] = rapid_record

        mu_array = np.asarray(mu)
        if mu_array.shape == (1, *MU_SHAPE):
            mu_array = mu_array[0]
        if mu_array.shape != MU_SHAPE:
            raise TaskCTraceError(f"WM reducer mu shape mismatch: {mu_array.shape} != {MU_SHAPE}")
        if not np.isfinite(mu_array).all():
            raise TaskCTraceError("WM reducer mu contains NaN or inf")
        mu_f16 = np.ascontiguousarray(mu_array, dtype=MU_DTYPE)
        mu_bytes = mu_f16.tobytes(order="C")
        if len(mu_bytes) != math.prod(MU_SHAPE) * MU_DTYPE.itemsize:
            raise TaskCTraceError("WM reducer mu byte-size mismatch")

        split = ctx["split"]
        with self._lock:
            wm_idx = self._wm_call_index
            self._wm_call_index += 1
            row = self._mu_rows[split]
            self._mu_rows[split] += 1
            wm_call_id = f"{self.condition}:wm:{wm_idx:09d}"
            mu_sha = sha256_bytes(mu_bytes)
            _append_bytes(self.root / "mu_raw" / f"{split}.f16", mu_bytes)
            mu_index = {
                "schema_version": SCHEMA_VERSION,
                "wm_call_id": wm_call_id,
                "policy_call_id": policy_call_id,
                "raw_file": f"{split}.f16",
                "raw_row": row,
                "shape": list(MU_SHAPE),
                "dtype": "float16-le",
                "mu_sha256": mu_sha,
                **ctx,
            }
            append_jsonl(self.root / "mu_raw" / "index.jsonl", mu_index)
            c_tier0_ms = float(wm_forward_kappa_ms) + float(kappa_host_check_ms) + float(kappa_decision_ms)
            append_jsonl(
                self.root / "wm_calls.jsonl",
                {
                    "schema_version": SCHEMA_VERSION,
                    "wm_call_id": wm_call_id,
                    "wm_call_index": wm_idx,
                    "policy_call_id": policy_call_id,
                    "round_index": int(round_index),
                    "kappa": float(kappa),
                    "decision": decision,
                    "decision_eligible": bool(decision_eligible),
                    "action_expert_executed": bool(action_expert_executed),
                    "wm_forward_kappa_ms": float(wm_forward_kappa_ms),
                    "kappa_host_check_ms": float(kappa_host_check_ms),
                    "kappa_decision_ms": float(kappa_decision_ms),
                    "c_tier0_ms": c_tier0_ms,
                    "timing_warmup": wm_idx < self.timing_warmup_calls,
                    "mu_sha256": mu_sha,
                    "mu_raw_row": row,
                    **rapid_fields,
                    **ctx,
                },
            )
        return wm_call_id


class ClientTraceRecorder:
    """Append-only LIBERO step and episode recorder."""

    def __init__(self, root: pathlib.Path, *, run_id: str, condition: str) -> None:
        self.root = root.resolve()
        self.run_id = str(run_id)
        self.condition = str(condition)
        self.root.mkdir(parents=True, exist_ok=True)
        guarded = [
            self.root / "episodes.jsonl",
            self.root / "steps_raw.jsonl",
            self.root / "proprio_observations.jsonl",
        ]
        dirty = [str(path) for path in guarded if path.exists() and path.stat().st_size]
        if dirty:
            raise TaskCTraceError(f"refusing to append to non-empty Task-C client trace: {dirty}")

    def record_proprio_observation(self, context: Mapping[str, Any], *, state: np.ndarray) -> None:
        """Record the exact raw proprio row consumed by an O(1) client trigger."""

        ctx = validate_trace_context(
            context,
            expected_condition=self.condition,
            expected_run_id=self.run_id,
        )
        state_array = np.asarray(state, dtype=np.float32).reshape(-1)
        if state_array.shape != (8,):
            raise TaskCTraceError(f"LIBERO trigger proprio shape mismatch: {state_array.shape} != (8,)")
        if not np.isfinite(state_array).all():
            raise TaskCTraceError("non-finite LIBERO trigger proprio")
        append_jsonl(
            self.root / "proprio_observations.jsonl",
            {
                "schema_version": SCHEMA_VERSION,
                "state": [float(value) for value in state_array],
                **ctx,
            },
        )

    def record_step(
        self,
        context: Mapping[str, Any],
        *,
        source: str = "main",
        task_description: str,
        action: np.ndarray,
        state_before: np.ndarray,
        state_after: np.ndarray,
        done_after_step: bool,
        policy_kappa: np.ndarray | None,
    ) -> None:
        ctx = validate_trace_context(
            context,
            expected_condition=self.condition,
            expected_run_id=self.run_id,
        )
        if source not in {"main", "low_replan_rollout"}:
            raise TaskCTraceError(f"unsupported Task-C simulator-step source: {source!r}")
        action_array = np.asarray(action, dtype=np.float32).reshape(-1)
        before = np.asarray(state_before, dtype=np.float32).reshape(-1)
        after = np.asarray(state_after, dtype=np.float32).reshape(-1)
        if action_array.shape != (7,):
            raise TaskCTraceError(f"LIBERO action shape mismatch: {action_array.shape} != (7,)")
        if before.shape != (8,) or after.shape != (8,):
            raise TaskCTraceError(f"LIBERO proprio shape mismatch: before={before.shape}, after={after.shape}")
        if not (np.isfinite(action_array).all() and np.isfinite(before).all() and np.isfinite(after).all()):
            raise TaskCTraceError("non-finite LIBERO step data")
        ee_delta = after[:3] - before[:3]
        kappa_values = None
        if policy_kappa is not None:
            kappa_array = np.asarray(policy_kappa, dtype=np.float64).reshape(-1)
            if not np.isfinite(kappa_array).all():
                raise TaskCTraceError("non-finite policy kappa in client step")
            kappa_values = [float(value) for value in kappa_array]
        append_jsonl(
            self.root / "steps_raw.jsonl",
            {
                "schema_version": SCHEMA_VERSION,
                "source": source,
                "task_description": str(task_description),
                "action": [float(value) for value in action_array],
                "gripper_command": float(action_array[6]),
                "state_before": [float(value) for value in before],
                "state_after": [float(value) for value in after],
                "ee_velocity_proxy": [float(value) for value in ee_delta],
                "ee_speed_proxy": float(np.linalg.norm(ee_delta)),
                "gripper_qpos_before": [float(value) for value in before[6:8]],
                "gripper_qpos_after": [float(value) for value in after[6:8]],
                "done_after_step": bool(done_after_step),
                "policy_kappa": kappa_values,
                **ctx,
            },
        )

    def record_episode(
        self,
        context: Mapping[str, Any],
        *,
        task_description: str,
        success: bool,
        control_steps: int,
        main_control_steps: int | None = None,
        error: str | None,
    ) -> None:
        ctx = validate_trace_context(
            context,
            expected_condition=self.condition,
            expected_run_id=self.run_id,
        )
        if ctx["env_step"] != 0:
            raise TaskCTraceError("episode records must use env_step=0 context")
        append_jsonl(
            self.root / "episodes.jsonl",
            {
                "schema_version": SCHEMA_VERSION,
                "task_description": str(task_description),
                "success": bool(success),
                "control_steps": int(control_steps),
                "main_control_steps": int(main_control_steps) if main_control_steps is not None else int(control_steps),
                "error": error,
                **ctx,
            },
        )


def percentile_summary(values: Sequence[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or not np.isfinite(array).all():
        raise TaskCTraceError("percentile_summary requires at least one finite sample")
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
        "jitter": float(np.std(array)),
    }


def exact_mcnemar(baseline: Sequence[bool], candidate: Sequence[bool]) -> dict[str, Any]:
    """Exact two-sided paired McNemar test (binomial conditional test)."""

    if len(baseline) != len(candidate) or not baseline:
        raise TaskCTraceError("McNemar inputs must be non-empty and paired")
    b = sum(bool(base) and not bool(cand) for base, cand in zip(baseline, candidate, strict=True))
    c = sum(not bool(base) and bool(cand) for base, cand in zip(baseline, candidate, strict=True))
    discordant = b + c
    if discordant == 0:
        p_value = 1.0
    else:
        tail = sum(math.comb(discordant, index) for index in range(min(b, c) + 1)) / (2**discordant)
        p_value = min(1.0, 2.0 * tail)
    return {
        "baseline_success_candidate_failure": b,
        "baseline_failure_candidate_success": c,
        "discordant": discordant,
        "p_value_two_sided_exact": float(p_value),
    }


def paired_bootstrap_success_delta(
    baseline: Sequence[bool],
    candidate: Sequence[bool],
    *,
    seed: int = 20260719,
    samples: int = 50_000,
) -> dict[str, float | int | str]:
    """Deterministic paired-bootstrap interval for candidate minus baseline SR."""

    if len(baseline) != len(candidate) or not baseline:
        raise TaskCTraceError("bootstrap inputs must be non-empty and paired")
    if samples < 1000:
        raise TaskCTraceError("paired bootstrap requires at least 1000 resamples")
    base = np.asarray(baseline, dtype=np.float64)
    cand = np.asarray(candidate, dtype=np.float64)
    diffs = cand - base
    rng = np.random.default_rng(seed)
    # Bound peak memory while preserving a deterministic draw stream.
    means: list[np.ndarray] = []
    remaining = samples
    while remaining:
        chunk = min(remaining, 4096)
        indexes = rng.integers(0, diffs.size, size=(chunk, diffs.size), endpoint=False)
        means.append(np.mean(diffs[indexes], axis=1))
        remaining -= chunk
    boot = np.concatenate(means)
    return {
        "method": "paired_percentile",
        "n_pairs": int(diffs.size),
        "samples": int(samples),
        "seed": int(seed),
        "success_rate_delta": float(np.mean(diffs)),
        "lower_95_one_sided": float(np.percentile(boot, 5)),
        "lower_95_two_sided": float(np.percentile(boot, 2.5)),
        "upper_95_two_sided": float(np.percentile(boot, 97.5)),
    }


def contact_phase_by_trajectory(
    steps: Iterable[Mapping[str, Any]],
    *,
    close_threshold: float = 0.0,
    sustained_steps: int = 2,
) -> dict[tuple[str, int], str]:
    """Label approach/contact using a frozen sustained-close command proxy.

    LIBERO's action convention uses positive gripper commands to close and the
    official warm-up command ``-1`` to remain open.  Contact starts at the first
    of ``sustained_steps`` consecutive commands above ``close_threshold``.
    """

    if sustained_steps < 1:
        raise ValueError("sustained_steps must be >= 1")
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for step in steps:
        grouped.setdefault(str(step["trajectory_id"]), []).append(step)
    labels: dict[tuple[str, int], str] = {}
    for trajectory_id, records in grouped.items():
        ordered = sorted(records, key=lambda item: int(item["env_step"]))
        onset: int | None = None
        for idx in range(0, len(ordered) - sustained_steps + 1):
            band = ordered[idx : idx + sustained_steps]
            if all(float(item["gripper_command"]) > close_threshold for item in band):
                onset = int(ordered[idx]["env_step"])
                break
        for item in ordered:
            env_step = int(item["env_step"])
            labels[(trajectory_id, env_step)] = "contact" if onset is not None and env_step >= onset else "approach"
    return labels


def ratio(numerator: int, denominator: int) -> float | None:
    return float(numerator / denominator) if denominator else None
