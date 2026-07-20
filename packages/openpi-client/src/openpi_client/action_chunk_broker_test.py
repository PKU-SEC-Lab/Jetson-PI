from __future__ import annotations

from typing import Dict

import numpy as np

from openpi_client import action_chunk_broker
from openpi_client import base_policy


class _FakePolicy(base_policy.BasePolicy):
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def infer(self, obs: Dict) -> Dict:  # noqa: UP006
        self.events.append(f"policy:{obs['tick']}")
        return {"actions": np.zeros((3, 7), dtype=np.float32)}


def test_async_broker_observation_hook_runs_once_before_routing_work() -> None:
    events: list[str] = []
    broker = action_chunk_broker.AsyncActionBufferBroker(
        _FakePolicy(events),
        action_horizon=3,
        async_trigger_step=2,
        observation_step_hook=lambda obs: events.append(f"observe:{obs['tick']}"),
    )
    try:
        broker.infer({"tick": 0})
        broker.infer({"tick": 1})
    finally:
        broker.shutdown()

    assert events[0:2] == ["observe:0", "policy:0"]
    assert events.count("observe:0") == 1
    assert events.count("observe:1") == 1
