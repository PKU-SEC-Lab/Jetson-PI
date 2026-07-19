from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
from openpi_client import task_c_trace
import pytest

from scripts import task_c_analysis


def _context(*, episode_idx: int, env_step: int = 0, suite: str = "libero_spatial", task_id: int = 0) -> dict:
    return task_c_trace.trace_context(
        run_id="test-run",
        condition="kappa_0p4",
        suite=suite,
        task_id=task_id,
        episode_idx=episode_idx,
        seed=42,
        env_step=env_step,
    )


def _write_jsonl(path: pathlib.Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"".join(task_c_trace.canonical_json_bytes(record) + b"\n" for record in records))


def _paired_suite_roots(tmp_path: pathlib.Path, *, suite: str = "libero_object") -> dict[str, pathlib.Path]:
    roots: dict[str, pathlib.Path] = {}
    for condition in ("faac_only", "kappa_0p4"):
        root = tmp_path / condition
        episodes: list[dict] = []
        calls: list[dict] = []
        steps: list[dict] = []
        for task_id in range(10):
            for episode_idx in range(30):
                context = _context(episode_idx=episode_idx, suite=suite, task_id=task_id)
                episodes.append({**context, "condition": condition, "success": True})
                if condition == "kappa_0p4":
                    calls.append(
                        {
                            **context,
                            "condition": condition,
                            "decision_eligible": True,
                            "decision": "skip_vlm",
                        }
                    )
                    steps.append({**context, "condition": condition, "phase": "approach"})
        _write_jsonl(root / "episodes.jsonl", episodes)
        _write_jsonl(root / "server_trace" / "wm_calls.jsonl", calls)
        _write_jsonl(root / "steps_labeled.jsonl", steps)
        _write_jsonl(root / "mu" / "calibration" / "index.jsonl", [])
        _write_jsonl(root / "mu" / "eval" / "index.jsonl", [])
        (root / "summary.json").write_text("{}\n", encoding="utf-8")
        task_c_trace.write_json_atomic(
            root / "run_manifest.json",
            {
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
                "kappa_delta": None if condition == "faac_only" else 0.4,
                "confidence_schedule": condition != "faac_only",
                "world_model": {
                    "training_suite": "libero_spatial",
                    "out_of_training_suite": True,
                },
            },
        )
        (root / "client.log").write_text("sealed\n", encoding="utf-8")
        task_c_analysis._write_sha_manifest(root)  # noqa: SLF001 - construct a production-format receipt
        roots[condition] = root
    return roots


def test_trace_context_is_condition_independent_and_disjoint() -> None:
    calibration = _context(episode_idx=0)
    evaluation = _context(episode_idx=1)
    other_condition = dict(calibration, condition="kappa_0p8")

    assert calibration["split"] == "calibration"
    assert evaluation["split"] == "eval"
    assert calibration["trajectory_id"] != evaluation["trajectory_id"]
    assert other_condition["trajectory_id"] == calibration["trajectory_id"]


def test_server_trace_rejects_wrong_mu_shape(tmp_path: pathlib.Path) -> None:
    recorder = task_c_trace.ServerTraceRecorder(
        tmp_path,
        run_id="test-run",
        condition="kappa_0p4",
        timing_warmup_calls=0,
    )
    call_id = recorder.begin_policy_call(_context(episode_idx=0), kind="kappa_schedule")
    with pytest.raises(task_c_trace.TaskCTraceError, match="mu shape mismatch"):
        recorder.record_wm_call(
            _context(episode_idx=0),
            policy_call_id=call_id,
            round_index=0,
            mu=np.zeros((4, 16), dtype=np.float32),
            kappa=1.0,
            wm_forward_kappa_ms=1.0,
            kappa_host_check_ms=0.01,
            kappa_decision_ms=0.001,
            decision="seed_round",
            decision_eligible=False,
            action_expert_executed=True,
        )


def test_contact_phase_uses_sustained_close_proxy() -> None:
    steps = [
        {"trajectory_id": "trajectory", "env_step": 0, "gripper_command": -1.0},
        {"trajectory_id": "trajectory", "env_step": 1, "gripper_command": 0.5},
        {"trajectory_id": "trajectory", "env_step": 2, "gripper_command": -0.1},
        {"trajectory_id": "trajectory", "env_step": 3, "gripper_command": 0.2},
        {"trajectory_id": "trajectory", "env_step": 4, "gripper_command": 0.3},
    ]
    phases = task_c_trace.contact_phase_by_trajectory(steps)

    assert [phases[("trajectory", index)] for index in range(5)] == [
        "approach",
        "approach",
        "approach",
        "contact",
        "contact",
    ]


def test_exact_mcnemar_and_noninferiority_are_paired() -> None:
    baseline = [True] * 30
    candidate = [True] * 28 + [False, False]

    mcnemar = task_c_trace.exact_mcnemar(baseline, candidate)
    interval = task_c_trace.paired_bootstrap_success_delta(baseline, candidate, samples=2000)

    assert mcnemar["baseline_success_candidate_failure"] == 2
    assert mcnemar["baseline_failure_candidate_success"] == 0
    assert mcnemar["p_value_two_sided_exact"] == 0.5
    assert interval["success_rate_delta"] == pytest.approx(-2 / 30)


def test_valid_skip_rate_requires_preserved_paired_success() -> None:
    baseline = []
    candidate = []
    calls = []
    phase_map = {}
    for episode_idx in range(30):
        context = _context(episode_idx=episode_idx)
        episode = {**context, "success": True}
        baseline.append(episode)
        candidate.append(episode)
        phase = "approach" if episode_idx < 15 else "contact"
        phase_map[(context["trajectory_id"], 0)] = phase
        calls.append(
            {
                **context,
                "decision_eligible": True,
                "decision": "skip_vlm",
            }
        )

    result = task_c_analysis._paired_result(  # noqa: SLF001 - verify the analysis contract directly
        baseline,
        candidate,
        calls,
        phase_map,
        bootstrap_seed=42,
    )

    assert result["validity_gate_pass"] is True
    assert result["raw_skip"]["skip_rate"] == 1.0
    assert result["success_conditioned_skip"]["skip_rate"] == 1.0
    assert result["deployable_valid_skip_rate"] == 1.0
    assert result["per_phase"]["approach"]["deployable_valid_skip_rate"] == 1.0
    assert result["per_phase"]["contact"]["deployable_valid_skip_rate"] == 1.0


def test_significantly_better_candidate_is_not_rejected_by_mcnemar() -> None:
    baseline = []
    candidate = []
    calls = []
    phase_map = {}
    for episode_idx in range(30):
        context = _context(episode_idx=episode_idx)
        baseline.append({**context, "success": episode_idx < 20})
        candidate.append({**context, "success": True})
        phase_map[(context["trajectory_id"], 0)] = "approach"
        calls.append({**context, "decision_eligible": True, "decision": "skip_vlm"})

    result = task_c_analysis._paired_result(  # noqa: SLF001 - verify the analysis contract directly
        baseline,
        candidate,
        calls,
        phase_map,
        bootstrap_seed=42,
    )

    assert result["mcnemar_nonsignificant"] is False
    assert result["mcnemar_no_evidence_of_harm"] is True
    assert result["validity_gate_pass"] is True


def test_decision_stats_count_exact_kappa_reinfers_and_reject_unknown_labels() -> None:
    calls = [
        {"decision_eligible": True, "decision": "skip_vlm"},
        {"decision_eligible": True, "decision": "infer_vlm"},
    ]

    result = task_c_analysis._decision_stats(calls)  # noqa: SLF001 - verify the analysis contract directly

    assert result["kappa_forced_reinfer_count"] == 1
    assert result["kappa_ever_forced_reinfer"] is True
    with pytest.raises(task_c_trace.TaskCTraceError, match="unexpected eligible decision"):
        task_c_analysis._decision_stats(  # noqa: SLF001 - verify fail-closed receipt analysis
            [{"decision_eligible": True, "decision": "seed_round"}]
        )


def test_c1_aggregate_rejects_an_incomplete_design(tmp_path: pathlib.Path) -> None:
    roots = {}
    for condition in ("faac_only", "kappa_0p2", "kappa_0p4", "kappa_0p8"):
        root = tmp_path / condition
        root.mkdir()
        context = _context(episode_idx=0)
        (root / "episodes.jsonl").write_text(
            task_c_trace.canonical_json_bytes({**context, "condition": condition, "success": True}).decode() + "\n",
            encoding="utf-8",
        )
        roots[condition] = root

    with pytest.raises(task_c_trace.TaskCTraceError, match="exactly 300 episodes"):
        task_c_analysis.aggregate_c1(tmp_path / "aggregate", roots)


def test_paired_suite_aggregate_accepts_exact_c3_design(tmp_path: pathlib.Path) -> None:
    roots = _paired_suite_roots(tmp_path)

    result = task_c_analysis.aggregate_paired_suite(
        tmp_path / "aggregate",
        roots,
        suite="libero_object",
    )

    assert result["experiment"] == "C3_libero_object_k9"
    assert result["suite"] == "libero_object"
    assert result["world_model_training_suite"] == "libero_spatial"
    assert result["wm_out_of_training_suite"] is True
    assert result["paired"]["kappa_0p4"]["validity_gate_pass"] is True
    assert result["paired"]["kappa_0p4"]["deployable_valid_skip_rate"] == 1.0


def test_paired_suite_aggregate_rejects_tampered_condition_receipt(tmp_path: pathlib.Path) -> None:
    roots = _paired_suite_roots(tmp_path)
    (roots["kappa_0p4"] / "client.log").write_text("tampered\n", encoding="utf-8")

    with pytest.raises(task_c_trace.TaskCTraceError, match="SHA-256 mismatch"):
        task_c_analysis.aggregate_paired_suite(
            tmp_path / "aggregate",
            roots,
            suite="libero_object",
        )


def test_paired_suite_aggregate_rejects_non_k9_condition(tmp_path: pathlib.Path) -> None:
    roots = _paired_suite_roots(tmp_path)
    root = roots["kappa_0p4"]
    manifest_path = root / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["trigger_k"] = 8
    task_c_trace.write_json_atomic(manifest_path, manifest)
    task_c_analysis._write_sha_manifest(root)  # noqa: SLF001 - reseal an internally consistent invalid receipt

    with pytest.raises(task_c_trace.TaskCTraceError, match="trigger_k"):
        task_c_analysis.aggregate_paired_suite(
            tmp_path / "aggregate",
            roots,
            suite="libero_object",
        )


def test_paired_suite_aggregate_cli_accepts_c3_inputs(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "task_c_analysis.py",
            "aggregate-paired-suite",
            "/tmp/c3-object/aggregate",
            "--suite",
            "libero_object",
            "--faac-only",
            "/tmp/c3-object/faac_only",
            "--kappa-0p4",
            "/tmp/c3-object/kappa_0p4",
        ],
    )

    args = task_c_analysis._parse_args()  # noqa: SLF001 - exercise the public CLI parser

    assert args.command == "aggregate-paired-suite"
    assert args.suite == "libero_object"


def test_finalize_condition_emits_required_schema(tmp_path: pathlib.Path) -> None:
    manifest = {
        "schema_version": task_c_trace.SCHEMA_VERSION,
        "condition": "kappa_0p4",
        "expected_episodes": 2,
        "status": "running",
    }
    (tmp_path / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    client = task_c_trace.ClientTraceRecorder(tmp_path, run_id="test-run", condition="kappa_0p4")
    server = task_c_trace.ServerTraceRecorder(
        tmp_path / "server_trace",
        run_id="test-run",
        condition="kappa_0p4",
        timing_warmup_calls=0,
    )
    for episode_idx in range(2):
        context = _context(episode_idx=episode_idx)
        client.record_step(
            context,
            task_description="test task",
            action=np.array([0, 0, 0, 0, 0, 0, -1], dtype=np.float32),
            state_before=np.zeros(8, dtype=np.float32),
            state_after=np.ones(8, dtype=np.float32),
            done_after_step=True,
            policy_kappa=np.array([1.0, 0.9], dtype=np.float32),
        )
        client.record_episode(
            context,
            task_description="test task",
            success=True,
            control_steps=1,
            error=None,
        )
        policy_call_id = server.begin_policy_call(context, kind="kappa_schedule")
        server.record_wm_call(
            context,
            policy_call_id=policy_call_id,
            round_index=0,
            mu=np.full((1, 4, 1024), episode_idx, dtype=np.float32),
            kappa=1.0,
            wm_forward_kappa_ms=1.0,
            kappa_host_check_ms=0.01,
            kappa_decision_ms=0.001,
            decision="seed_round",
            decision_eligible=False,
            action_expert_executed=True,
        )
        server.record_wm_call(
            context,
            policy_call_id=policy_call_id,
            round_index=1,
            mu=np.full((1, 4, 1024), episode_idx + 0.5, dtype=np.float32),
            kappa=0.9,
            wm_forward_kappa_ms=1.1,
            kappa_host_check_ms=0.02,
            kappa_decision_ms=0.002,
            decision="skip_vlm",
            decision_eligible=True,
            action_expert_executed=True,
        )

    summary = task_c_analysis.finalize_condition(tmp_path)

    assert summary["episodes"] == 2
    assert summary["success_rate"] == 1.0
    assert summary["scheduling"]["raw_skip_rate"] == 1.0
    assert summary["mu"]["trajectory_overlap_count"] == 0
    assert summary["mu"]["calibration"]["rows"] == 2
    assert summary["mu"]["eval"]["rows"] == 2
    assert (tmp_path / "steps_labeled.jsonl").is_file()
    assert (tmp_path / "summary.json").is_file()
    assert (tmp_path / "SHA256SUMS").is_file()
