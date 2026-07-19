from __future__ import annotations

import sys
from typing import Any

from scripts import task_c_c1


def test_run_condition_cli_accepts_c3_suite(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "task_c_c1.py",
            "run-condition",
            "--condition",
            "kappa_0p4",
            "--suite",
            "libero_object",
            "--out",
            "/tmp/c3-object",
        ],
    )

    args = task_c_c1._parse_args()  # noqa: SLF001 - exercise the public CLI parser

    assert args.suite == "libero_object"


def test_run_condition_cli_forwards_suite(monkeypatch) -> None:
    captured: dict[str, Any] = {}

    def fake_run_condition(**kwargs: Any) -> None:
        captured.update(kwargs)

    monkeypatch.setattr(task_c_c1, "run_condition", fake_run_condition)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "task_c_c1.py",
            "run-condition",
            "--condition",
            "faac_only",
            "--suite",
            "libero_goal",
            "--out",
            "/tmp/c3-goal",
        ],
    )

    task_c_c1.main()

    assert captured["suite"] == "libero_goal"
