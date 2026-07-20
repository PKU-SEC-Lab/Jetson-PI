from __future__ import annotations

import math
import json

import numpy as np
import pytest

from openpi_client import rapid_trigger


def _state(*, x: float, axis_x: float = 0.0, aperture: float = 0.08) -> np.ndarray:
    return np.array([x, 0.0, 0.0, axis_x, 0.0, 0.0, aperture / 2, -aperture / 2], dtype=np.float32)


def _thresholds() -> rapid_trigger.RapidThresholds:
    return rapid_trigger.RapidThresholds(
        theta_v=0.2,
        theta_a=0.05,
        theta_omega=0.2,
        theta_closed=0.02,
        theta_open=0.06,
        transition_cooldown_steps=2,
        dt=1.0,
    )


def test_trigger_skips_only_after_two_safe_finite_difference_samples() -> None:
    trigger = rapid_trigger.RapidKinematicTrigger(_thresholds())

    decisions = [
        trigger.observe(_state(x=0.0)),
        trigger.observe(_state(x=0.1)),
        trigger.observe(_state(x=0.2)),
        trigger.observe(_state(x=0.3)),
    ]

    assert [item.decision for item in decisions] == ["infer", "infer", "infer", "skip"]
    assert decisions[-1].ee_speed == pytest.approx(0.1, abs=1e-7)
    assert decisions[-1].ee_acceleration == pytest.approx(0.0, abs=1e-7)
    assert decisions[-1].gripper_state == "open"


def test_trigger_fails_closed_on_acceleration_and_gripper_transition() -> None:
    trigger = rapid_trigger.RapidKinematicTrigger(_thresholds())
    for x in (0.0, 0.1, 0.2, 0.3):
        trigger.observe(_state(x=x))

    acceleration = trigger.observe(_state(x=0.7))
    transition = trigger.observe(_state(x=0.8, aperture=0.04))
    cooldown_1 = trigger.observe(_state(x=0.9, aperture=0.01))
    cooldown_2 = trigger.observe(_state(x=1.0, aperture=0.01))

    assert acceleration.decision == "infer"
    assert acceleration.ee_acceleration > 0.05
    assert transition.decision == "infer"
    assert transition.gripper_transition is True
    assert cooldown_1.gripper_transition is True
    assert cooldown_2.gripper_transition is True


def test_trigger_uses_relative_so3_rotation_across_axis_angle_wrap() -> None:
    trigger = rapid_trigger.RapidKinematicTrigger(_thresholds())

    trigger.observe(_state(x=0.0, axis_x=math.pi - 0.01))
    result = trigger.observe(_state(x=0.0, axis_x=-math.pi + 0.01))

    assert result.ee_angular_speed == pytest.approx(0.02, abs=1e-6)


def test_command_flip_veto_and_nonfinite_state_force_infer() -> None:
    trigger = rapid_trigger.RapidKinematicTrigger(_thresholds())
    for x in (0.0, 0.1, 0.2, 0.3):
        last = trigger.observe(_state(x=x))
    assert last.decision == "skip"

    vetoed = rapid_trigger.apply_gripper_command_veto(last, [-1.0, -1.0, 1.0])
    invalid = trigger.observe(np.full(8, np.nan, dtype=np.float32))

    assert vetoed.decision == "infer"
    assert vetoed.command_transition is True
    assert invalid.decision == "infer"
    assert invalid.fail_closed_reason == "nonfinite_proprio"


def test_threshold_document_loader_rejects_wrong_formula(tmp_path) -> None:
    path = tmp_path / "rapid_thresholds.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "jetson-pi-task-c-rapid-thresholds-v1",
                "formula_version": "rapid-kinematic-v1",
                "thresholds": {
                    "theta_v": 0.2,
                    "theta_a": 0.05,
                    "theta_omega": 0.2,
                    "theta_closed": 0.02,
                    "theta_open": 0.06,
                    "transition_cooldown_steps": 2,
                    "dt": 1.0,
                },
            }
        ),
        encoding="utf-8",
    )

    loaded = rapid_trigger.load_threshold_document(path)
    assert loaded.theta_v == 0.2

    path.write_text(path.read_text(encoding="utf-8").replace("rapid-kinematic-v1", "unknown"), encoding="utf-8")
    with pytest.raises(ValueError, match="formula_version"):
        rapid_trigger.load_threshold_document(path)


def test_routing_policy_changes_only_the_gate_decision() -> None:
    assert rapid_trigger.route_decision("always_infer", rapid_route=None, kappa=1.0, kappa0=1.0, delta=0.4) == "infer"
    assert rapid_trigger.route_decision("rapid", rapid_route="skip", kappa=0.0, kappa0=1.0, delta=0.4) == "skip"
    assert rapid_trigger.route_decision("rapid", rapid_route="infer", kappa=1.0, kappa0=1.0, delta=0.4) == "infer"
    assert rapid_trigger.route_decision("kappa", rapid_route=None, kappa=0.59, kappa0=1.0, delta=0.4) == "infer"
    assert rapid_trigger.route_decision("kappa", rapid_route=None, kappa=0.60, kappa0=1.0, delta=0.4) == "skip"


def test_finite_difference_feature_rows_use_the_public_formula() -> None:
    states = np.zeros((4, 8), dtype=np.float64)
    states[:, 0] = [0.0, 1.0, 3.0, 6.0]
    states[:, 5] = [0.0, 0.1, 0.3, 0.6]

    rows = rapid_trigger.finite_difference_feature_rows(states, dt=2.0)

    assert rows[0] == {
        "ee_speed": None,
        "ee_acceleration": None,
        "ee_angular_speed": None,
        "gripper_aperture": 0.0,
        "history_valid": False,
    }
    assert rows[1]["ee_speed"] == pytest.approx(0.5)
    assert rows[1]["ee_acceleration"] is None
    assert rows[1]["ee_angular_speed"] == pytest.approx(0.05)
    assert rows[2]["ee_speed"] == pytest.approx(1.0)
    assert rows[2]["ee_acceleration"] == pytest.approx(0.25)
    assert rows[2]["ee_angular_speed"] == pytest.approx(0.1)
    assert rows[2]["history_valid"] is True


def test_finite_difference_feature_rows_reject_bad_trajectories() -> None:
    with pytest.raises(ValueError, match="shape"):
        rapid_trigger.finite_difference_feature_rows(np.zeros((3, 7)), dt=1.0)
    states = np.zeros((3, 8))
    states[1, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        rapid_trigger.finite_difference_feature_rows(states, dt=1.0)
