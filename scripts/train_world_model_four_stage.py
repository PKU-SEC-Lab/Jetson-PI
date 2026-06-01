#!/usr/bin/env python3
# ruff: noqa: RUF002

import dataclasses
import datetime as _dt
import json
import os
import socket
import sys

from openpi.training import world_model_training_four_stage as wm_four


def main() -> None:
    cfg = wm_four.cli()

    # Repro header: always print to stdout early so `tee` captures it.
    ts = _dt.datetime.now().astimezone().isoformat(timespec="seconds")
    env_keys = [
        "CUDA_VISIBLE_DEVICES",
        "PYTHONNOUSERSITE",
        "PYTHONPATH",
        "OPENPI_LIBERO_LOCAL_DATASET_DIR",
        "OPENPI_WM_TIMESTAMP_CHECK_MODE",
        "XLA_PYTHON_CLIENT_PREALLOCATE",
        "XLA_PYTHON_CLIENT_MEM_FRACTION",
        "XLA_PYTHON_CLIENT_ALLOCATOR",
        "HF_ENDPOINT",
        "HF_HOME",
        "WANDB_MODE",
        "WANDB_DISABLED",
    ]
    env = {k: os.environ.get(k) for k in env_keys if os.environ.get(k) is not None}
    header = {
        "timestamp": ts,
        "cwd": os.getcwd(),
        "host": socket.gethostname(),
        "python": sys.executable,
        "argv": sys.argv,
        "env": env,
        "config": dataclasses.asdict(cfg),
    }
    print("# RUN_CONFIG_HEADER")
    print(json.dumps(header, ensure_ascii=False, sort_keys=True))
    sys.stdout.flush()

    wm_four.run_training(cfg)


if __name__ == "__main__":
    main()

