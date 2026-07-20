from __future__ import annotations

import json

import numpy as np
import pytest

from openpi_client import task_c_trace


def _context() -> dict[str, object]:
    return task_c_trace.trace_context(
        run_id="rapid-test",
        condition="rapid",
        suite="libero_spatial",
        task_id=0,
        episode_idx=30,
        seed=42,
        env_step=9,
    )


def test_server_trace_persists_routing_and_rapid_payload(tmp_path) -> None:
    recorder = task_c_trace.ServerTraceRecorder(tmp_path, run_id="rapid-test", condition="rapid")
    policy_call_id = recorder.begin_policy_call(_context(), kind="rapid_schedule")
    rapid = {"decision": "skip", "trigger_compute_ns": 1234, "history_valid": True}
    recorder.record_wm_call(
        _context(),
        policy_call_id=policy_call_id,
        round_index=0,
        mu=np.zeros(task_c_trace.MU_SHAPE),
        kappa=0.5,
        wm_forward_kappa_ms=1.0,
        kappa_host_check_ms=0.1,
        kappa_decision_ms=0.01,
        decision="seed_round",
        decision_eligible=False,
        action_expert_executed=True,
        routing_policy="rapid",
        rapid=rapid,
    )

    row = json.loads((tmp_path / "wm_calls.jsonl").read_text(encoding="utf-8"))
    assert row["routing_policy"] == "rapid"
    assert row["rapid"] == rapid


def test_server_trace_fails_closed_on_invalid_rapid_timing(tmp_path) -> None:
    recorder = task_c_trace.ServerTraceRecorder(tmp_path, run_id="rapid-test", condition="rapid")
    policy_call_id = recorder.begin_policy_call(_context(), kind="rapid_schedule")
    with pytest.raises(task_c_trace.TaskCTraceError, match="trigger_compute_ns"):
        recorder.record_wm_call(
            _context(),
            policy_call_id=policy_call_id,
            round_index=0,
            mu=np.zeros(task_c_trace.MU_SHAPE),
            kappa=0.5,
            wm_forward_kappa_ms=1.0,
            kappa_host_check_ms=0.1,
            kappa_decision_ms=0.01,
            decision="seed_round",
            decision_eligible=False,
            action_expert_executed=True,
            routing_policy="rapid",
            rapid={"decision": "skip", "trigger_compute_ns": -1},
        )


def test_client_trace_records_exact_trigger_proprio_rows(tmp_path) -> None:
    recorder = task_c_trace.ClientTraceRecorder(tmp_path, run_id="rapid-test", condition="rapid")
    state = np.arange(8, dtype=np.float32)

    recorder.record_proprio_observation(_context(), state=state)

    row = json.loads((tmp_path / "proprio_observations.jsonl").read_text(encoding="utf-8"))
    assert row["state"] == [float(value) for value in state]
    assert row["trajectory_id"] == _context()["trajectory_id"]
