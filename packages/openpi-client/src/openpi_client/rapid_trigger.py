"""Training-free O(1) kinematic routing for Task-C RAPID evaluations."""

from __future__ import annotations

import dataclasses
import json
import math
import pathlib
import time
from collections import deque
from collections.abc import Mapping, Sequence
from typing import Literal

import numpy as np


RapidRoute = Literal["skip", "infer"]
RoutingPolicy = Literal["kappa", "always_infer", "rapid"]
GripperState = Literal["open", "closed", "transition"]
ROUTING_POLICIES = frozenset({"kappa", "always_infer", "rapid"})
RAPID_ROUTING_POLICIES = frozenset({"always_infer", "rapid"})
THRESHOLD_SCHEMA_VERSION = "jetson-pi-task-c-rapid-thresholds-v1"
FORMULA_VERSION = "rapid-kinematic-v1"


@dataclasses.dataclass(frozen=True)
class RapidThresholds:
    theta_v: float
    theta_a: float
    theta_omega: float
    theta_closed: float
    theta_open: float
    transition_cooldown_steps: int = 2
    dt: float = 1.0

    def __post_init__(self) -> None:
        finite = (
            self.theta_v,
            self.theta_a,
            self.theta_omega,
            self.theta_closed,
            self.theta_open,
            self.dt,
        )
        if not all(math.isfinite(float(value)) for value in finite):
            raise ValueError("RAPID thresholds must be finite")
        if min(self.theta_v, self.theta_a, self.theta_omega, self.theta_closed) < 0:
            raise ValueError("RAPID motion and closed-aperture thresholds must be non-negative")
        if self.theta_closed >= self.theta_open:
            raise ValueError("RAPID gripper hysteresis requires theta_closed < theta_open")
        if self.transition_cooldown_steps < 0:
            raise ValueError("RAPID transition cooldown must be non-negative")
        if self.dt <= 0:
            raise ValueError("RAPID dt must be positive")

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> RapidThresholds:
        required = {
            "theta_v",
            "theta_a",
            "theta_omega",
            "theta_closed",
            "theta_open",
            "transition_cooldown_steps",
            "dt",
        }
        if set(value) != required:
            raise ValueError(f"RAPID threshold keys must be exactly {sorted(required)}")
        def number(key: str) -> float:
            raw = value[key]
            if isinstance(raw, bool) or not isinstance(raw, (int, float)):
                raise ValueError(f"RAPID {key} must be numeric")
            return float(raw)

        cooldown = value["transition_cooldown_steps"]
        if isinstance(cooldown, bool) or not isinstance(cooldown, int):
            raise ValueError("RAPID transition_cooldown_steps must be an integer")
        return cls(
            theta_v=number("theta_v"),
            theta_a=number("theta_a"),
            theta_omega=number("theta_omega"),
            theta_closed=number("theta_closed"),
            theta_open=number("theta_open"),
            transition_cooldown_steps=cooldown,
            dt=number("dt"),
        )


@dataclasses.dataclass(frozen=True)
class RapidDecision:
    decision: RapidRoute
    ee_speed: float | None
    ee_acceleration: float | None
    ee_angular_speed: float | None
    gripper_aperture: float | None
    gripper_state: GripperState | None
    gripper_transition: bool
    command_transition: bool
    history_valid: bool
    safe_current: bool
    safe_previous: bool
    trigger_compute_ns: int
    fail_closed_reason: str | None = None

    def as_dict(self) -> dict[str, object]:
        return dataclasses.asdict(self)


def _axis_angle_to_matrix(axis_angle: np.ndarray) -> np.ndarray:
    angle = float(np.linalg.norm(axis_angle))
    if angle < 1e-12:
        return np.eye(3, dtype=np.float64)
    axis = np.asarray(axis_angle, dtype=np.float64) / angle
    x, y, z = axis
    skew = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float64)
    return np.eye(3, dtype=np.float64) + math.sin(angle) * skew + (1.0 - math.cos(angle)) * (skew @ skew)


def _relative_rotation_angle(previous: np.ndarray, current: np.ndarray) -> float:
    relative = _axis_angle_to_matrix(previous).T @ _axis_angle_to_matrix(current)
    cosine = float(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0))
    sine = float(
        np.linalg.norm(
            np.array(
                [
                    relative[2, 1] - relative[1, 2],
                    relative[0, 2] - relative[2, 0],
                    relative[1, 0] - relative[0, 1],
                ],
                dtype=np.float64,
            )
        )
        / 2.0
    )
    return math.atan2(sine, cosine)


def finite_difference_feature_rows(states: np.ndarray, *, dt: float) -> list[dict[str, float | bool | None]]:
    """Compute the frozen RAPID motion features for one ordered trajectory."""

    values = np.asarray(states, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 8:
        raise ValueError(f"RAPID trajectory states must have shape (N, 8), got {values.shape}")
    if not math.isfinite(float(dt)) or dt <= 0:
        raise ValueError("RAPID feature dt must be finite and positive")
    if not np.isfinite(values).all():
        raise ValueError("RAPID trajectory states must be finite")

    rows: list[dict[str, float | bool | None]] = []
    for index, current in enumerate(values):
        speed = None
        acceleration = None
        angular_speed = None
        if index >= 1:
            previous = values[index - 1]
            speed = float(np.linalg.norm(current[:3] - previous[:3]) / dt)
            angular_speed = float(_relative_rotation_angle(previous[3:6], current[3:6]) / dt)
        if index >= 2:
            previous = values[index - 1]
            previous_previous = values[index - 2]
            acceleration = float(
                np.linalg.norm(current[:3] - 2.0 * previous[:3] + previous_previous[:3]) / dt**2
            )
        rows.append(
            {
                "ee_speed": speed,
                "ee_acceleration": acceleration,
                "ee_angular_speed": angular_speed,
                "gripper_aperture": float(current[6] - current[7]),
                "history_valid": index >= 2,
            }
        )
    return rows


class RapidKinematicTrigger:
    """Stateful three-sample finite-difference trigger with fail-closed routing."""

    def __init__(self, thresholds: RapidThresholds) -> None:
        self.thresholds = thresholds
        self._history: deque[np.ndarray] = deque(maxlen=3)
        self._last_stable_gripper: Literal["open", "closed"] | None = None
        self._cooldown_remaining = 0
        self._previous_safe = False
        self._latest_decision: RapidDecision | None = None

    def reset(self) -> None:
        self._history.clear()
        self._last_stable_gripper = None
        self._cooldown_remaining = 0
        self._previous_safe = False
        self._latest_decision = None

    @property
    def latest_decision(self) -> RapidDecision | None:
        return self._latest_decision

    def observe(self, proprio: np.ndarray) -> RapidDecision:
        started_ns = time.perf_counter_ns()
        state = np.asarray(proprio, dtype=np.float64).reshape(-1)
        if state.shape != (8,):
            self._history.clear()
            self._previous_safe = False
            self._latest_decision = self._fail_closed(started_ns, "proprio_shape")
            return self._latest_decision
        if not np.isfinite(state).all():
            self._history.clear()
            self._previous_safe = False
            self._latest_decision = self._fail_closed(started_ns, "nonfinite_proprio")
            return self._latest_decision

        aperture = float(state[6] - state[7])
        gripper_state, transition_event = self._gripper_observation(aperture)
        if transition_event:
            self._cooldown_remaining = self.thresholds.transition_cooldown_steps
            gripper_transition = True
        elif self._cooldown_remaining > 0:
            gripper_transition = True
            self._cooldown_remaining -= 1
        else:
            gripper_transition = False

        self._history.append(np.array(state, dtype=np.float64, copy=True))
        history_valid = len(self._history) == 3
        speed: float | None = None
        acceleration: float | None = None
        angular_speed: float | None = None
        safe_current = False
        safe_previous = self._previous_safe
        if len(self._history) >= 2:
            previous, current = self._history[-2], self._history[-1]
            dt = self.thresholds.dt
            speed = float(np.linalg.norm(current[:3] - previous[:3]) / dt)
            angular_speed = float(_relative_rotation_angle(previous[3:6], current[3:6]) / dt)
        if history_valid:
            previous_previous, previous, current = self._history
            dt = self.thresholds.dt
            acceleration = float(np.linalg.norm(current[:3] - 2.0 * previous[:3] + previous_previous[:3]) / dt**2)
            if speed is None or angular_speed is None:
                raise AssertionError("RAPID two-sample kinematics missing with three-sample history")
            safe_current = (
                speed <= self.thresholds.theta_v
                and acceleration <= self.thresholds.theta_a
                and angular_speed <= self.thresholds.theta_omega
                and not gripper_transition
            )

        decision: RapidRoute = "skip" if safe_current and safe_previous else "infer"
        self._previous_safe = safe_current
        self._latest_decision = RapidDecision(
            decision=decision,
            ee_speed=speed,
            ee_acceleration=acceleration,
            ee_angular_speed=angular_speed,
            gripper_aperture=aperture,
            gripper_state=gripper_state,
            gripper_transition=gripper_transition,
            command_transition=False,
            history_valid=history_valid,
            safe_current=safe_current,
            safe_previous=safe_previous,
            trigger_compute_ns=time.perf_counter_ns() - started_ns,
        )
        return self._latest_decision

    def _gripper_observation(self, aperture: float) -> tuple[GripperState, bool]:
        if aperture >= self.thresholds.theta_open:
            state: GripperState = "open"
            stable: Literal["open", "closed"] | None = "open"
        elif aperture <= self.thresholds.theta_closed:
            state = "closed"
            stable = "closed"
        else:
            state = "transition"
            stable = None
        transition = state == "transition"
        if stable is not None:
            transition = transition or (
                self._last_stable_gripper is not None and stable != self._last_stable_gripper
            )
            self._last_stable_gripper = stable
        return state, transition

    @staticmethod
    def _fail_closed(started_ns: int, reason: str) -> RapidDecision:
        return RapidDecision(
            decision="infer",
            ee_speed=None,
            ee_acceleration=None,
            ee_angular_speed=None,
            gripper_aperture=None,
            gripper_state=None,
            gripper_transition=False,
            command_transition=False,
            history_valid=False,
            safe_current=False,
            safe_previous=False,
            trigger_compute_ns=time.perf_counter_ns() - started_ns,
            fail_closed_reason=reason,
        )


def apply_gripper_command_veto(decision: RapidDecision, gripper_commands: Sequence[float]) -> RapidDecision:
    """Force INFER if the already-issued fixed-horizon gripper command changes sign."""

    started_ns = time.perf_counter_ns()
    commands = np.asarray(gripper_commands, dtype=np.float64).reshape(-1)
    invalid = commands.size == 0 or not np.isfinite(commands).all()
    signs = commands > 0.0
    transition = invalid or bool(np.any(signs[1:] != signs[:-1]))
    elapsed = time.perf_counter_ns() - started_ns
    if invalid:
        return dataclasses.replace(
            decision,
            decision="infer",
            command_transition=True,
            trigger_compute_ns=decision.trigger_compute_ns + elapsed,
            fail_closed_reason="invalid_gripper_commands",
        )
    return dataclasses.replace(
        decision,
        decision="infer" if transition else decision.decision,
        command_transition=transition,
        trigger_compute_ns=decision.trigger_compute_ns + elapsed,
    )


def load_threshold_document(path: pathlib.Path) -> RapidThresholds:
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError("RAPID threshold document must be a JSON object")
    if document.get("schema_version") != THRESHOLD_SCHEMA_VERSION:
        raise ValueError(f"unexpected RAPID threshold schema_version: {document.get('schema_version')!r}")
    if document.get("formula_version") != FORMULA_VERSION:
        raise ValueError(f"unexpected RAPID formula_version: {document.get('formula_version')!r}")
    thresholds = document.get("thresholds")
    if not isinstance(thresholds, dict):
        raise ValueError("RAPID threshold document is missing a thresholds object")
    return RapidThresholds.from_mapping(thresholds)


def route_decision(
    routing_policy: RoutingPolicy,
    *,
    rapid_route: RapidRoute | None,
    kappa: float,
    kappa0: float,
    delta: float,
) -> RapidRoute:
    """Return the sole manipulated skip/infer branch for the shared scheduler."""

    if routing_policy == "always_infer":
        return "infer"
    if routing_policy == "rapid":
        if rapid_route not in ("skip", "infer"):
            raise ValueError(f"RAPID routing requires a valid rapid_route, got {rapid_route!r}")
        return rapid_route
    if routing_policy == "kappa":
        if not all(math.isfinite(value) for value in (kappa, kappa0, delta)) or delta < 0:
            raise ValueError("kappa routing requires finite kappa/kappa0 and non-negative delta")
        return "infer" if kappa < kappa0 - delta else "skip"
    raise ValueError(f"unknown routing_policy: {routing_policy!r}")
