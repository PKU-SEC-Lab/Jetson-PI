# ruff: noqa: RUF002, RUF003, FBT003
"""Field descriptions for async World Model data collection (design doc section 4).

No I/O: structured metadata for logging schemas or dataset columns.
"""

from __future__ import annotations

import dataclasses
from typing import Any


@dataclasses.dataclass(frozen=True)
class AsyncWorldModelRecordingField:

    name: str
    dtype: str
    description: str
    required: bool = True


# Design doc 4.1 raw signals
RAW_SIGNAL_FIELDS: tuple[AsyncWorldModelRecordingField, ...] = (
    AsyncWorldModelRecordingField(
        "observation.images.*",
        "uint8|float32, HxWx3",
        "Multi-camera RGB; store native res, resize/tokenize offline",
        True,
    ),
    AsyncWorldModelRecordingField(
        "observation.images.*.timestamp_ns",
        "int64",
        "Per-camera exposure or arrival timestamp",
        True,
    ),
    AsyncWorldModelRecordingField(
        "observation.state",
        "float32[D_state]",
        "Proprio aligned to images (joints, EE pose, gripper, ...)",
        True,
    ),
    AsyncWorldModelRecordingField(
        "action.command",
        "float32[D_act]",
        "Executed action each control step (source of committed prefix)",
        True,
    ),
    AsyncWorldModelRecordingField(
        "task.prompt",
        "str",
        "Language instruction or task id for Pi0 prompt",
        False,
    ),
    AsyncWorldModelRecordingField(
        "task.scene_id",
        "str|int",
        "Scene or object id",
        False,
    ),
    AsyncWorldModelRecordingField(
        "timing.image_ready_ns",
        "int64",
        "Image ready time",
        False,
    ),
    AsyncWorldModelRecordingField(
        "timing.vlm_start_ns",
        "int64",
        "VLM forward start",
        False,
    ),
    AsyncWorldModelRecordingField(
        "timing.vlm_end_ns",
        "int64",
        "VLM forward end",
        False,
    ),
    AsyncWorldModelRecordingField(
        "timing.ae_start_ns",
        "int64",
        "Action expert sampling start",
        False,
    ),
    AsyncWorldModelRecordingField(
        "timing.handover_ns",
        "int64",
        "True control handover time (not only infer-done)",
        True,
    ),
    AsyncWorldModelRecordingField(
        "async.chunk_start_step",
        "int32",
        "Chunk start index in trajectory",
        False,
    ),
    AsyncWorldModelRecordingField(
        "async.async_trigger_step",
        "int32",
        "Prefetch trigger step k",
        False,
    ),
    AsyncWorldModelRecordingField(
        "async.action_horizon",
        "int32",
        "Chunk length H",
        False,
    ),
    AsyncWorldModelRecordingField(
        "episode.success",
        "bool",
        "Success flag (failures help uncertainty)",
        False,
    ),
)


def recording_schema_dict() -> dict[str, Any]:
    """Compact schema for JSON export."""
    return {
        "raw_signals": [dataclasses.asdict(f) for f in RAW_SIGNAL_FIELDS],
        "offline_sample_notes": (
            "Per time t sample control delta and handover H in [1,5]; "
            "A_comm = a_t..a_{t+delta-1}; supervise a_{t+delta:t+delta+H}; "
            "target C* = stopgrad(Q(H_{t+delta}))."
        ),
    }
