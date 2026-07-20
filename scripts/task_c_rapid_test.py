from __future__ import annotations

import copy

from openpi_client import task_c_trace
import pytest

from scripts import task_c_c1
from scripts import task_c_rapid


def _episodes(successes: list[bool], *, condition: str) -> list[dict[str, object]]:
    return [
        {
            "trajectory_id": f"libero_spatial/task-00/seed-42/init-{index:03d}",
            "condition": condition,
            "suite": "libero_spatial",
            "task_id": 0,
            "episode_idx": index,
            "seed": 42,
            "success": success,
        }
        for index, success in enumerate(successes)
    ]


def _rapid_wm_calls(decisions: list[str]) -> list[dict[str, object]]:
    calls: list[dict[str, object]] = []
    for index, decision in enumerate(decisions):
        trajectory_id = f"libero_spatial/task-00/seed-42/init-{index:03d}"
        common = {
            "policy_call_id": f"rapid:policy:{index:08d}",
            "trajectory_id": trajectory_id,
            "condition": "rapid",
            "suite": "libero_spatial",
            "task_id": 0,
            "episode_idx": index,
            "seed": 42,
            "env_step": 0,
            "routing_policy": "rapid",
            "rapid": {"decision": decision, "trigger_compute_ns": 2_000, "history_valid": True},
        }
        calls.extend(
            [
                {**common, "wm_call_id": f"wm-{index}-0", "round_index": 0, "kappa": 1.0},
                {
                    **common,
                    "wm_call_id": f"wm-{index}-1",
                    "round_index": 1,
                    "kappa": 0.9 if decision == "skip" else 0.5,
                },
            ]
        )
    return calls


def _steps(count: int) -> list[dict[str, object]]:
    return [
        {
            "trajectory_id": f"libero_spatial/task-00/seed-42/init-{index:03d}",
            "env_step": 0,
            "gripper_command": -1.0,
        }
        for index in range(count)
    ]


def _gate_config() -> dict[str, object]:
    return {
        "bootstrap": {"method": "paired_percentile", "samples": 1000, "seed": 17},
        "cal_fit": {
            "mcnemar_p_value_min_inclusive": 0.05,
            "success_rate_delta_min_inclusive": 0.0,
            "selection_primary": "max_success_anchored_skip_rate",
            "tie_break": "lowest_motion_quantile",
        },
        "cal_confirm": {
            "abort_if_harmful_mcnemar_p_below": 0.05,
            "abort_if_success_rate_delta_below": -0.05,
            "abort_if_paired_percentile_lower_one_sided_below": -0.05,
        },
        "final_eval": {"noninferiority_margin": -0.05, "mcnemar_alpha": 0.05},
    }


def test_raw_pair_result_deduplicates_policy_calls_and_reports_kappa_agreement() -> None:
    baseline = _episodes([True, True, True, False], condition="rapid_always_infer")
    candidate = _episodes([True, True, True, False], condition="rapid")
    calls = _rapid_wm_calls(["skip", "skip", "infer", "infer"])

    result = task_c_rapid.paired_raw_result(
        baseline,
        candidate,
        calls,
        _steps(4),
        bootstrap_samples=1000,
        bootstrap_seed=17,
        noninferiority_margin=-0.05,
        kappa_delta=0.4,
    )

    assert result["n_pairs"] == 4
    assert result["raw_skip_rate"] == 0.5
    assert result["success_anchored_skip_rate"] == pytest.approx(2 / 3)
    assert result["trigger_compute_us"]["mean"] == 2.0
    assert result["kappa_decision_agreement"]["agreement_rate"] == 1.0
    assert result["paired_bootstrap"]["method"] == "paired_percentile"


def test_cal_fit_selection_is_computed_and_never_eyeballed() -> None:
    baseline = _episodes([True] * 10, condition="rapid_always_infer")
    conservative = task_c_rapid.paired_raw_result(
        baseline,
        _episodes([True] * 10, condition="rapid_q50"),
        _rapid_wm_calls(["skip"] * 4 + ["infer"] * 6),
        _steps(10),
        bootstrap_samples=1000,
        bootstrap_seed=17,
        noninferiority_margin=-0.05,
        kappa_delta=0.4,
    )
    aggressive = task_c_rapid.paired_raw_result(
        baseline,
        _episodes([True] * 10, condition="rapid_q80"),
        _rapid_wm_calls(["skip"] * 8 + ["infer"] * 2),
        _steps(10),
        bootstrap_samples=1000,
        bootstrap_seed=17,
        noninferiority_margin=-0.05,
        kappa_delta=0.4,
    )

    selected = task_c_rapid.select_cal_fit_candidate(
        {0.50: conservative, 0.80: aggressive},
        _gate_config(),
    )

    assert selected["selected_motion_quantile"] == 0.80
    assert selected["selection_computed_from_raw_rows"] is True


def test_cal_confirm_aborts_on_predeclared_noninferiority_failure() -> None:
    result = {
        "success_rate_delta": -0.06,
        "paired_mcnemar": {
            "p_value_two_sided_exact": 0.5,
            "baseline_success_candidate_failure": 6,
            "baseline_failure_candidate_success": 0,
        },
        "paired_bootstrap": {"lower_95_one_sided": -0.08},
    }

    decision = task_c_rapid.cal_confirm_decision(result, _gate_config())

    assert decision["abort"] is True
    assert "success_rate_delta" in decision["abort_reasons"]
    assert "paired_percentile_lower_one_sided" in decision["abort_reasons"]


def test_threshold_ladder_uses_only_successful_raw_proprio_trajectories() -> None:
    episodes = _episodes([True, False], condition="rapid_always_infer")
    observations: list[dict[str, object]] = []
    for trajectory_index, scale in ((0, 1.0), (1, 100.0)):
        for env_step, x in enumerate((0.0, 0.1, 0.2, 0.3)):
            aperture = 0.08 if env_step < 2 else 0.005
            observations.append(
                {
                    "trajectory_id": f"libero_spatial/task-00/seed-42/init-{trajectory_index:03d}",
                    "env_step": env_step,
                    "state": [x * scale, 0.0, 0.0, 0.0, 0.0, 0.0, aperture / 2, -aperture / 2],
                }
            )

    ladder = task_c_rapid.derive_threshold_ladder(
        episodes,
        observations,
        motion_quantiles=[0.5, 0.9],
        transition_cooldown_steps=2,
        dt=1.0,
    )

    assert [item["motion_quantile"] for item in ladder] == [0.5, 0.9]
    assert all(item["calibration"]["successful_trajectories"] == 1 for item in ladder)
    assert all(item["thresholds"]["theta_v"] < 1.0 for item in ladder)
    assert all(item["thresholds"]["theta_closed"] < item["thresholds"]["theta_open"] for item in ladder)


def test_eval_arm_contract_rejects_any_shared_path_drift() -> None:
    baseline = {
        "condition": "rapid_always_infer",
        "routing_policy": "always_infer",
        "rapid_thresholds_sha256": "threshold-sha",
        "shared_execution_path": {"digest": "same", "arms_share_byte_identical_paths": True},
        "only_manipulated_variable": "routing_policy",
    }
    candidate = copy.deepcopy(baseline)
    candidate.update(condition="rapid", routing_policy="rapid")
    assert task_c_rapid.assert_eval_arm_equivalence(baseline, candidate)["verified"] is True

    candidate["shared_execution_path"]["digest"] = "different"
    with pytest.raises(task_c_trace.TaskCTraceError, match="shared execution path"):
        task_c_rapid.assert_eval_arm_equivalence(baseline, candidate)


def test_rapid_runner_spec_keeps_threshold_and_shared_scheduler_flags_identical() -> None:
    baseline = task_c_c1.rapid_condition_spec(
        condition="rapid_always_infer",
        routing_policy="always_infer",
        rapid_thresholds_path="configs/task_c/rapid_thresholds.json",
    )
    candidate = task_c_c1.rapid_condition_spec(
        condition="rapid",
        routing_policy="rapid",
        rapid_thresholds_path="configs/task_c/rapid_thresholds.json",
    )

    assert baseline["shared_scheduler_flags"] == candidate["shared_scheduler_flags"]
    assert baseline["rapid_thresholds_path"] == candidate["rapid_thresholds_path"]
    assert baseline["routing_policy"] == "always_infer"
    assert candidate["routing_policy"] == "rapid"
