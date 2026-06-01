# ruff: noqa: RUF002, RUF003
"""World Model async training data: sample (t, delta, H) from LeRobot trajectories into batches.

Requires ``LeRobotDataset`` with long enough ``delta_timestamps``; ``valid_indices`` drops out-of-episode tails.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
import dataclasses
import json
import logging
import os
from pathlib import Path

import numpy as np
import torch
import tree

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.training.config as _config
from openpi.training.data_loader import _collate_fn
import openpi.transforms as _transforms

logger = logging.getLogger("openpi")

_DEFAULT_ACTION_KEYS: tuple[str, ...] = ("actions",)

_LIBERO_PI_SUITE_TO_TASK_RANGE: dict[str, tuple[int, int]] = {
    "spatial": (0, 10),
    "libero_spatial": (0, 10),
    "object": (10, 20),
    "libero_object": (10, 20),
    "goal": (20, 30),
    "libero_goal": (20, 30),
    "libero_10": (30, 40),
    "libero10": (30, 40),
    "10": (30, 40),
}


def _timestamp_check_marker_path(repo_id: str) -> Path:
    # Prefer HF cache root to keep marker colocated with dataset caches.
    hf_home = os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface")
    safe_repo = repo_id.replace("/", "__")
    return Path(hf_home) / "openpi" / "wm_timestamp_check" / f"{safe_repo}.ok"


def _episode_ids_cache_path(repo_id: str, lo: int, hi: int) -> Path:
    hf_home = os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface")
    safe_repo = repo_id.replace("/", "__")
    return Path(hf_home) / "openpi" / "wm_episode_ids" / f"{safe_repo}_{lo}_{hi}.json"


def _should_run_timestamp_check(repo_id: str) -> bool:
    """Policy for expensive timestamp consistency checks.

    OPENPI_WM_TIMESTAMP_CHECK_MODE:
      - once   (default): run once per repo_id, then reuse marker
      - always: run every process start
      - never : skip always
    """
    mode = os.environ.get("OPENPI_WM_TIMESTAMP_CHECK_MODE", "once").strip().lower()
    if mode == "always":
        return True
    if mode == "never":
        return False
    marker = _timestamp_check_marker_path(repo_id)
    return not marker.exists()


def _mark_timestamp_check_done(repo_id: str) -> None:
    marker = _timestamp_check_marker_path(repo_id)
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text("ok\n", encoding="utf-8")


def _create_lerobot_dataset(lerobot_dataset_module, *, skip_timestamp_check: bool, **kwargs):
    """Create LeRobotDataset with optional timestamp-check bypass."""
    if not skip_timestamp_check:
        return lerobot_dataset_module.LeRobotDataset(**kwargs)
    original = lerobot_dataset_module.check_timestamps_sync
    lerobot_dataset_module.check_timestamps_sync = lambda *args, **kws: None
    try:
        return lerobot_dataset_module.LeRobotDataset(**kwargs)
    finally:
        lerobot_dataset_module.check_timestamps_sync = original


def _load_cached_episode_ids(repo_id: str, lo: int, hi: int) -> list[int] | None:
    p = _episode_ids_cache_path(repo_id, lo, hi)
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        ids = data.get("episode_ids", [])
        if not isinstance(ids, list) or not ids:
            return None
        return [int(x) for x in ids]
    except Exception:
        return None


def _save_cached_episode_ids(repo_id: str, lo: int, hi: int, episode_ids: list[int]) -> None:
    p = _episode_ids_cache_path(repo_id, lo, hi)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps(
            {
                "repo_id": repo_id,
                "task_index_min": int(lo),
                "task_index_max": int(hi),
                "episode_ids": [int(x) for x in episode_ids],
            },
            ensure_ascii=True,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _episode_indices_for_task_index_range(hf_dataset, lo: int, hi: int) -> list[int]:
    ei = np.asarray(hf_dataset["episode_index"], dtype=np.int64).reshape(-1)
    ti = np.asarray(hf_dataset["task_index"], dtype=np.int64).reshape(-1)
    mask = (ti >= lo) & (ti < hi)
    if not np.any(mask):
        raise ValueError(f"No frames with task_index in [{lo}, {hi}) for this dataset.")
    return sorted(np.unique(ei[mask]).tolist())


def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _slice_time_stack(obj: object, frame_idx: int, stack_len: int) -> object:
    if isinstance(obj, dict):
        return {k: _slice_time_stack(v, frame_idx, stack_len) for k, v in obj.items()}
    if isinstance(obj, torch.Tensor):
        obj = _to_numpy(obj)
    if isinstance(obj, np.ndarray) and obj.ndim > 0 and obj.shape[0] == stack_len:
        return np.array(obj[frame_idx], copy=False)
    return obj


def _build_valid_frame_indices(lerobot_ds, need_future: int) -> list[int]:
    hf = lerobot_ds.hf_dataset
    ep_to = lerobot_ds.episode_data_index["to"].numpy()
    valid: list[int] = []
    n = len(lerobot_ds)
    for idx in range(n):
        ep = int(hf[idx]["episode_index"])
        if idx + need_future <= ep_to[ep]:
            valid.append(idx)
    logger.info("WorldModel dataset: %d / %d frames valid (need_future=%d)", len(valid), n, need_future)
    return valid


def _make_delta_timestamps(
    *,
    fps: float,
    stack_len: int,
    action_keys: Sequence[str],
    camera_keys: Sequence[str],
    state_key: str = "state",
) -> dict[str, list[float]]:
    offsets = [i / fps for i in range(stack_len)]
    ts: dict[str, list[float]] = {k: list(offsets) for k in action_keys}
    ts[state_key] = list(offsets)
    for cam in camera_keys:
        ts[cam] = list(offsets)
    return ts


@dataclasses.dataclass
class WorldModelDataConfig:

    max_delta_t: int = 10
    action_horizon_min: int = 1
    action_horizon_max: int = 5
    action_keys: Sequence[str] = dataclasses.field(default_factory=lambda: _DEFAULT_ACTION_KEYS)
    state_feature_key: str = "state"
    l_act_targets_from_t: bool = False


class WorldModelLeRobotDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        lerobot_ds: torch.utils.data.Dataset,
        *,
        valid_indices: Sequence[int],
        stack_len: int,
        sample_transform: Callable[[dict], dict],
        wm_cfg: WorldModelDataConfig,
        action_dim: int,
        pi0_action_horizon: int,
        seed: int = 0,
        lerobot_prompt_tasks: dict[int, str] | None = None,
    ) -> None:
        self._ds = lerobot_ds
        self._valid = list(valid_indices)
        self._stack_len = stack_len
        self._transform = sample_transform
        self._wm = wm_cfg
        self._action_dim = action_dim
        self._ah = pi0_action_horizon
        self._seed = seed
        self._lerobot_prompt_tasks = lerobot_prompt_tasks

    def __len__(self) -> int:
        return len(self._valid)

    @property
    def valid_frame_indices(self) -> list[int]:
        return list(self._valid)

    def __getitem__(self, index: int) -> dict:
        rng = np.random.default_rng([self._seed, index])
        idx = int(self._valid[index])
        raw = self._ds[idx]
        raw = tree.map_structure(_to_numpy, raw)

        ep = int(np.asarray(raw["episode_index"]).reshape(-1)[0])
        ep_end = int(self._ds.episode_data_index["to"][ep])
        dmax = min(self._wm.max_delta_t, ep_end - idx - self._wm.action_horizon_min)
        if dmax < 1:
            raise RuntimeError("invalid valid_indices: cannot sample delta")
        delta = int(rng.integers(1, dmax + 1))
        h_hi = min(self._wm.action_horizon_max, ep_end - idx - delta)
        if h_hi < self._wm.action_horizon_min:
            raise RuntimeError("invalid valid_indices: cannot sample H")
        if self._wm.action_horizon_min == self._wm.action_horizon_max:
            h = self._wm.action_horizon_min
        else:
            h = int(rng.integers(self._wm.action_horizon_min, h_hi + 1))
        if self._wm.l_act_targets_from_t:
            dmax_lact = int(min(dmax, h - 1))
            if dmax_lact < 1:
                raise RuntimeError("invalid valid_indices: l_act_targets_from_t needs h>1 and room for delta")
            delta = int(rng.integers(1, dmax_lact + 1))
            if ep_end - idx < h:
                raise RuntimeError("invalid valid_indices: l_act_targets_from_t needs ep_end - idx >= H")
        delta1 = int(rng.integers(1, delta)) if delta >= 2 else 0
        semigroup_valid = delta >= 2 and delta1 > 0 and delta1 < delta

        def tf(i: int) -> dict:
            sl = _slice_time_stack(raw, i, self._stack_len)
            if self._lerobot_prompt_tasks is not None:
                sl = _transforms.PromptFromLeRobotTask(self._lerobot_prompt_tasks)(sl)
            return self._transform(sl)

        per_step = [tf(t) for t in range(self._stack_len)]
        out0 = per_step[0]
        out_d = per_step[delta]
        out_d1 = per_step[delta1] if semigroup_valid else out0

        def _actions_vec(o: dict) -> np.ndarray:
            a = np.asarray(o["actions"], dtype=np.float32)
            if a.ndim == 2 and a.shape[0] == 1:
                a = a[0]
            if a.ndim != 1:
                raise ValueError(f"expected 1D actions per time step from transform, got {a.shape}")
            return a

        actions_seq = np.stack([_actions_vec(p) for p in per_step], axis=0)

        a_pad = np.zeros((self._wm.max_delta_t, self._action_dim), dtype=np.float32)
        prefix = actions_seq[:delta]
        a_pad[:delta] = _transforms.pad_to_dim(prefix, self._action_dim, axis=-1)
        prefix_mask = np.zeros((self._wm.max_delta_t,), dtype=bool)
        prefix_mask[:delta] = True

        if self._wm.l_act_targets_from_t:
            handover = actions_seq[0:h]
            handover = _transforms.pad_to_dim(handover, self._action_dim, axis=-1)
            actions_handover = np.zeros((self._ah, self._action_dim), dtype=np.float32)
            actions_handover[:h] = handover
            handover_valid = np.zeros((self._ah,), dtype=bool)
            if delta < h:
                handover_valid[delta:h] = True
        else:
            handover = actions_seq[delta : delta + h]
            handover = _transforms.pad_to_dim(handover, self._action_dim, axis=-1)
            actions_handover = np.zeros((self._ah, self._action_dim), dtype=np.float32)
            actions_handover[:h] = handover
            handover_valid = np.zeros((self._ah,), dtype=bool)
            handover_valid[:h] = True

        if semigroup_valid:
            seg2 = actions_seq[delta1:delta]
            ap2 = np.zeros((self._wm.max_delta_t, self._action_dim), dtype=np.float32)
            ln = seg2.shape[0]
            ap2[:ln] = _transforms.pad_to_dim(seg2, self._action_dim, axis=-1)
            mask2 = np.zeros((self._wm.max_delta_t,), dtype=bool)
            mask2[:ln] = True
        else:
            ap2 = np.zeros((self._wm.max_delta_t, self._action_dim), dtype=np.float32)
            mask2 = np.zeros((self._wm.max_delta_t,), dtype=bool)

        merged = {k: v for k, v in out0.items() if k != "actions"}
        merged["_wm_obs_future"] = dict(out_d)
        merged["_wm_obs_td1"] = dict(out_d1)
        merged["actions"] = actions_seq
        merged["wm_action_prefix_pad"] = a_pad
        merged["wm_prefix_mask"] = prefix_mask
        merged["wm_delta_t"] = np.float32(delta)
        merged["wm_actions_handover"] = actions_handover
        merged["wm_handover_valid"] = handover_valid
        merged["wm_semigroup_valid"] = np.bool_(semigroup_valid)
        merged["wm_delta1"] = np.int32(delta1 if semigroup_valid else 0)
        merged["wm_delta2"] = np.int32(delta - delta1 if semigroup_valid else 0)
        merged["wm_sg_prefix_pad"] = ap2
        merged["wm_sg_prefix_mask"] = mask2
        return merged


_WM_META_KEYS = frozenset(
    {
        "_wm_obs_td1",
        "_wm_obs_future",
        "wm_action_prefix_pad",
        "wm_prefix_mask",
        "wm_delta_t",
        "wm_actions_handover",
        "wm_handover_valid",
        "wm_semigroup_valid",
        "wm_delta1",
        "wm_delta2",
        "wm_sg_prefix_pad",
        "wm_sg_prefix_mask",
        "actions",
    }
)


def world_model_collate(batch: list[dict]) -> dict:
    return _collate_fn(batch)


def batch_to_observations_and_wm_tensors(
    batch: dict,
) -> tuple[_model.Observation, _model.Observation, _model.Observation, dict]:
    obs_t = {k: v for k, v in batch.items() if k not in _WM_META_KEYS}
    obs_f = batch["_wm_obs_future"]
    obs_td1 = batch["_wm_obs_td1"]
    extra = {
        "wm_action_prefix_pad": batch["wm_action_prefix_pad"],
        "wm_prefix_mask": batch["wm_prefix_mask"],
        "wm_delta_t": batch["wm_delta_t"],
        "wm_actions_handover": batch["wm_actions_handover"],
        "wm_handover_valid": batch["wm_handover_valid"],
        "wm_semigroup_valid": batch["wm_semigroup_valid"],
        "wm_delta1": batch["wm_delta1"],
        "wm_delta2": batch["wm_delta2"],
        "wm_sg_prefix_pad": batch["wm_sg_prefix_pad"],
        "wm_sg_prefix_mask": batch["wm_sg_prefix_mask"],
    }
    with at.disable_typechecking():
        o0 = _model.Observation.from_dict(obs_t)
        o1 = _model.Observation.from_dict(obs_f)
        o2 = _model.Observation.from_dict(obs_td1)
    return o0, o1, o2, extra


class WorldModelFakeDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        *,
        model_config: _model.BaseModelConfig,
        wm_cfg: WorldModelDataConfig,
        num_samples: int = 256,
        seed: int = 0,
    ) -> None:
        self._num = num_samples
        self._spec_o, _ = model_config.inputs_spec()
        self._wm = wm_cfg
        self._mc = model_config
        self._seed = seed

    def __len__(self) -> int:
        return self._num

    def __getitem__(self, index: int) -> dict:
        import jax
        import jax.numpy as jnp

        rng = jax.random.key(self._seed + index)

        def make_from_spec(spec: jax.ShapeDtypeStruct):
            nonlocal rng
            rng, data_rng = jax.random.split(rng)
            shape = spec.shape[1:]
            if spec.dtype == jnp.float32:
                return jax.random.uniform(data_rng, shape=shape, minval=-1.0, maxval=1.0)
            if spec.dtype == jnp.int32:
                return jax.random.randint(data_rng, shape=shape, minval=0, maxval=2048)
            return jnp.zeros(shape=shape, dtype=spec.dtype)

        observation = jax.tree.map(make_from_spec, self._spec_o)
        out0 = jax.tree.map(np.asarray, observation.to_dict())
        if self._wm.l_act_targets_from_t:
            delta, h = 2, min(5, self._mc.action_horizon)
            delta1 = 1
        else:
            delta, h = 3, 2
            delta1 = 1
        semigroup_valid = True
        a_pad = np.zeros((self._wm.max_delta_t, self._mc.action_dim), dtype=np.float32)
        rng_a = np.random.default_rng(self._seed + index)
        a_pad[:delta] = rng_a.standard_normal((delta, self._mc.action_dim)).astype(np.float32) * 0.1
        prefix_mask = np.zeros((self._wm.max_delta_t,), dtype=bool)
        prefix_mask[:delta] = True
        ah = self._mc.action_horizon
        actions_handover = np.zeros((ah, self._mc.action_dim), dtype=np.float32)
        actions_handover[:h] = rng_a.standard_normal((h, self._mc.action_dim)).astype(np.float32) * 0.1
        handover_valid = np.zeros((ah,), dtype=bool)
        if self._wm.l_act_targets_from_t and delta < h:
            handover_valid[delta:h] = True
        else:
            handover_valid[:h] = True
        ap2 = np.zeros_like(a_pad)
        ln = delta - delta1
        ap2[:ln] = a_pad[delta1:delta]
        mask2 = np.zeros_like(prefix_mask)
        mask2[:ln] = True
        merged = dict(out0)
        merged["_wm_obs_future"] = dict(out0)
        merged["_wm_obs_td1"] = dict(out0)
        merged["wm_action_prefix_pad"] = a_pad
        merged["wm_prefix_mask"] = prefix_mask
        merged["wm_delta_t"] = np.float32(delta)
        merged["wm_actions_handover"] = actions_handover
        merged["wm_handover_valid"] = handover_valid
        merged["wm_semigroup_valid"] = np.bool_(semigroup_valid)
        merged["wm_delta1"] = np.int32(delta1)
        merged["wm_delta2"] = np.int32(delta - delta1)
        merged["wm_sg_prefix_pad"] = ap2
        merged["wm_sg_prefix_mask"] = mask2
        return merged


def create_world_model_lerobot_dataset(
    data_config: _config.DataConfig,
    *,
    model_config: _model.BaseModelConfig,
    wm_cfg: WorldModelDataConfig,
    libero_suite: str | None = None,
    libero_task_index_min: int | None = None,
    libero_task_index_max: int | None = None,
    libero_scratch_download_videos: bool = False,
) -> tuple[WorldModelLeRobotDataset, int]:
    try:
        import lerobot.common.datasets.lerobot_dataset as lerobot_dataset  # type: ignore
    except Exception:  # noqa: BLE001
        lerobot_dataset = None  # type: ignore

    if data_config.repo_id is None or data_config.repo_id == "fake":
        raise ValueError("World model training requires a real LeRobot repo_id in DataConfig.")

    repo_id = data_config.repo_id
    local_dir = os.environ.get("OPENPI_LIBERO_LOCAL_DATASET_DIR", "").strip()
    if local_dir and (Path(local_dir).expanduser().resolve() / "meta" / "info.json").is_file():
        if lerobot_dataset is not None:
            logger.info(
                "OPENPI_LIBERO_LOCAL_DATASET_DIR is set and contains meta/info.json; using local parquet (no HF metadata fetch)."
            )
        lerobot_dataset = None

    if lerobot_dataset is None:
        if not local_dir:
            raise ModuleNotFoundError(
                "Missing `lerobot.common.datasets.lerobot_dataset` in this Python environment. "
                "Either install a compatible `lerobot` package, or set "
                "OPENPI_LIBERO_LOCAL_DATASET_DIR=/path/to/local/lerobot-parquet-dataset (with meta/ and data/)."
            )

        # ---------------------------
        # ---------------------------
        import bisect
        import io
        import json
        import pathlib

        import pyarrow.parquet as pq
        from PIL import Image

        root = pathlib.Path(local_dir).expanduser().resolve()
        info_path = root / "meta" / "info.json"
        tasks_path = root / "meta" / "tasks.jsonl"
        episodes_path = root / "meta" / "episodes.jsonl"
        if not info_path.exists():
            raise FileNotFoundError(f"Local dataset missing meta/info.json: {info_path}")
        if not tasks_path.exists():
            raise FileNotFoundError(f"Local dataset missing meta/tasks.jsonl: {tasks_path}")
        if not episodes_path.exists():
            raise FileNotFoundError(f"Local dataset missing meta/episodes.jsonl: {episodes_path}")

        info = json.loads(info_path.read_text(encoding="utf-8"))
        fps = float(info.get("fps", 10))
        features = info.get("features", {}) or {}
        camera_keys = [k for k, v in features.items() if isinstance(v, dict) and v.get("dtype") == "image"]
        if not camera_keys:
            raise ValueError("Local dataset has no image features in meta/info.json; cannot build WM samples.")

        # tasks: {task_index: task_str}
        lerobot_prompt_tasks: dict[int, str] = {}
        for line in tasks_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            obj = json.loads(line)
            lerobot_prompt_tasks[int(obj["task_index"])] = str(obj["task"])

        # episodes lengths (expect contiguous episode_index)
        ep_lengths: list[int] = []
        for line in episodes_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            obj = json.loads(line)
            ep_idx = int(obj["episode_index"])
            ln = int(obj["length"])
            if ep_idx != len(ep_lengths):
                raise ValueError(
                    f"episodes.jsonl not contiguous: got episode_index={ep_idx}, expected={len(ep_lengths)}"
                )
            ep_lengths.append(ln)
        total_episodes = len(ep_lengths)
        if total_episodes <= 0:
            raise ValueError("Local dataset has zero episodes.")

        # episode end indices in *global frame* space (inclusive)
        ep_to: list[int] = []
        s = -1
        for ln in ep_lengths:
            s += ln
            ep_to.append(s)
        total_frames = ep_to[-1] + 1

        def _ep_for_global(idx: int) -> int:
            return bisect.bisect_left(ep_to, idx)

        # Determine per-episode task_index by reading the first row of each parquet (fast: 1 row, 1 column).
        cache_ti = _episode_ids_cache_path(repo_id, -999, -998)  # special key for local task_index map
        task_index_per_ep: list[int] | None = None
        try:
            if cache_ti.exists():
                task_index_per_ep = json.loads(cache_ti.read_text(encoding="utf-8")).get("task_index_per_ep")
        except Exception:
            task_index_per_ep = None
        if not isinstance(task_index_per_ep, list) or len(task_index_per_ep) != total_episodes:
            task_index_per_ep = []
            chunks_size = int(info.get("chunks_size", 1000) or 1000)
            for ep in range(total_episodes):
                ep_chunk = ep // chunks_size
                parquet_path = root / "data" / f"chunk-{ep_chunk:03d}" / f"episode_{ep:06d}.parquet"
                if not parquet_path.exists():
                    raise FileNotFoundError(f"Missing parquet for episode {ep}: {parquet_path}")
                t = pq.read_table(parquet_path, columns=["task_index"]).slice(0, 1)
                task_index_per_ep.append(int(t["task_index"][0].as_py()))
            cache_ti.parent.mkdir(parents=True, exist_ok=True)
            cache_ti.write_text(
                json.dumps({"repo_id": repo_id, "task_index_per_ep": task_index_per_ep}, ensure_ascii=True) + "\n",
                encoding="utf-8",
            )

        # Minimal hf-like columns for existing helper functions
        episode_index_col = np.empty((total_frames,), dtype=np.int64)
        task_index_col = np.empty((total_frames,), dtype=np.int64)
        start = 0
        for ep, ln in enumerate(ep_lengths):
            end = start + ln
            episode_index_col[start:end] = ep
            task_index_col[start:end] = int(task_index_per_ep[ep])
            start = end
        hf_dataset = {"episode_index": episode_index_col, "task_index": task_index_col}

        class _LocalLeRobotDataset(torch.utils.data.Dataset):
            def __init__(self) -> None:
                self.hf_dataset = hf_dataset
                self.episode_data_index = {"to": torch.as_tensor(np.asarray(ep_to, dtype=np.int64))}

            def __len__(self) -> int:
                return int(total_frames)

            def __getitem__(self, idx: int) -> dict:
                idx = int(idx)
                ep = _ep_for_global(idx)
                ep_start = 0 if ep == 0 else (ep_to[ep - 1] + 1)
                t0 = idx - ep_start
                stack_len_local = wm_cfg.max_delta_t + wm_cfg.action_horizon_max

                chunks_size = int(info.get("chunks_size", 1000) or 1000)
                ep_chunk = ep // chunks_size
                parquet_path = root / "data" / f"chunk-{ep_chunk:03d}" / f"episode_{ep:06d}.parquet"

                cols = [
                    "image",
                    "wrist_image",
                    "state",
                    "actions",
                    "timestamp",
                    "frame_index",
                    "episode_index",
                    "index",
                    "task_index",
                ]
                table = pq.read_table(parquet_path, columns=cols).slice(t0, stack_len_local)

                out: dict[str, object] = {}
                for k in ("image", "wrist_image"):
                    if k in table.column_names:
                        col = table[k]
                        if hasattr(col, "combine_chunks"):
                            col = col.combine_chunks()
                        bytes_arr = col.field("bytes")
                        imgs = []
                        for j in range(len(bytes_arr)):
                            b = bytes_arr[j].as_py()
                            if b is None:
                                raise ValueError(f"Missing {k}.bytes at episode={ep} t={t0+j}")
                            im = Image.open(io.BytesIO(b))
                            imgs.append(np.asarray(im.convert("RGB"), dtype=np.uint8))
                        out[k] = np.stack(imgs, axis=0)

                for k in table.column_names:
                    if k in ("image", "wrist_image"):
                        continue
                    out[k] = table[k].to_numpy(zero_copy_only=False)
                return out

        stack_len = wm_cfg.max_delta_t + wm_cfg.action_horizon_max
        need_future = stack_len

        # suite/task filter
        episode_ids: list[int] | None = None
        if libero_task_index_min is not None and libero_task_index_max is not None:
            t_lo, t_hi = libero_task_index_min, libero_task_index_max
        elif libero_suite is not None:
            key = libero_suite.lower().replace("-", "_")
            if key not in _LIBERO_PI_SUITE_TO_TASK_RANGE:
                raise ValueError(
                    f"Unknown libero_suite={libero_suite!r}. "
                    f"Try one of: {sorted(set(_LIBERO_PI_SUITE_TO_TASK_RANGE))} "
                    "or pass libero_task_index_min/max."
                )
            t_lo, t_hi = _LIBERO_PI_SUITE_TO_TASK_RANGE[key]
        else:
            t_lo = t_hi = None
        if t_lo is not None and t_hi is not None:
            cached_ids = _load_cached_episode_ids(repo_id, int(t_lo), int(t_hi))
            if cached_ids is not None:
                episode_ids = cached_ids
            else:
                episode_ids = [ep for ep, ti in enumerate(task_index_per_ep) if int(ti) >= t_lo and int(ti) < t_hi]
                _save_cached_episode_ids(repo_id, int(t_lo), int(t_hi), episode_ids)

        valid: list[int] = []
        allow = set(episode_ids) if episode_ids is not None else None
        for idx in range(total_frames):
            ep = _ep_for_global(idx)
            if allow is not None and ep not in allow:
                continue
            ep_end = ep_to[ep]
            if idx + need_future <= ep_end:
                valid.append(idx)

        logger.info(
            "Local LeRobot parquet dataset enabled: root=%s | episodes=%d | frames=%d | valid_frames=%d | fps=%s | camera_keys=%s",
            str(root),
            total_episodes,
            total_frames,
            len(valid),
            fps,
            camera_keys,
        )

        sample_transform = _transforms.compose(
            [
                *data_config.repack_transforms.inputs,
                *data_config.data_transforms.inputs,
                _transforms.Normalize(data_config.norm_stats or {}, use_quantiles=data_config.use_quantile_norm),
                *data_config.model_transforms.inputs,
            ]
        )

        ds_local = _LocalLeRobotDataset()
        wm_ds = WorldModelLeRobotDataset(
            ds_local,
            valid_indices=valid,
            stack_len=stack_len,
            sample_transform=sample_transform,
            wm_cfg=wm_cfg,
            action_dim=int(model_config.action_dim),
            pi0_action_horizon=int(model_config.action_horizon),
            seed=int(data_config.seed) if hasattr(data_config, "seed") else 0,
            lerobot_prompt_tasks=lerobot_prompt_tasks,
        )
        return wm_ds, int(total_frames)
    run_timestamp_check = _should_run_timestamp_check(repo_id)
    logger.info(
        "WorldModel timestamp check policy: mode=%s, run_now=%s (repo_id=%s)",
        os.environ.get("OPENPI_WM_TIMESTAMP_CHECK_MODE", "once"),
        run_timestamp_check,
        repo_id,
    )
    meta = lerobot_dataset.LeRobotDatasetMetadata(repo_id)
    stack_len = wm_cfg.max_delta_t + wm_cfg.action_horizon_max
    action_keys = tuple(data_config.action_sequence_keys)
    cam_keys = list(meta.camera_keys)
    if not cam_keys:
        raise ValueError("Dataset has no camera keys; cannot build multi-frame world-model samples.")
    state_key = wm_cfg.state_feature_key
    delta_ts = _make_delta_timestamps(
        fps=float(meta.fps),
        stack_len=stack_len,
        action_keys=action_keys,
        camera_keys=cam_keys,
        state_key=state_key,
    )

    episode_ids: list[int] | None = None
    if libero_task_index_min is not None and libero_task_index_max is not None:
        t_lo, t_hi = libero_task_index_min, libero_task_index_max
    elif libero_suite is not None:
        key = libero_suite.lower().replace("-", "_")
        if key not in _LIBERO_PI_SUITE_TO_TASK_RANGE:
            raise ValueError(
                f"Unknown libero_suite={libero_suite!r}. "
                f"Try one of: {sorted(set(_LIBERO_PI_SUITE_TO_TASK_RANGE))} "
                "or pass libero_task_index_min/max."
            )
        t_lo, t_hi = _LIBERO_PI_SUITE_TO_TASK_RANGE[key]
    else:
        t_lo = t_hi = None

    if t_lo is not None and t_hi is not None:
        cached_ids = _load_cached_episode_ids(repo_id, t_lo, t_hi)
        if cached_ids is not None:
            episode_ids = cached_ids
            logger.info(
                "Loaded cached episode_ids for task_index [%s, %s): %d episodes (repo_id=%s)",
                t_lo,
                t_hi,
                len(episode_ids),
                repo_id,
            )
        else:
            if "libero" not in repo_id.lower():
                logger.warning(
                    "libero_suite / task_index filter is meant for physical-intelligence/libero-style layout; "
                    "repo_id=%s may use different task_index ordering.",
                    repo_id,
                )
            scratch = _create_lerobot_dataset(
                lerobot_dataset_module=lerobot_dataset,
                skip_timestamp_check=not run_timestamp_check,
                repo_id=repo_id,
                delta_timestamps=None,
                download_videos=libero_scratch_download_videos,
            )
            episode_ids = _episode_indices_for_task_index_range(scratch.hf_dataset, t_lo, t_hi)
            _save_cached_episode_ids(repo_id, t_lo, t_hi, episode_ids)
            logger.info(
                "Computed+cached episode_ids for task_index [%s, %s): %d episodes (repo_id=%s)",
                t_lo,
                t_hi,
                len(episode_ids),
                repo_id,
            )
            del scratch

    base = _create_lerobot_dataset(
        lerobot_dataset_module=lerobot_dataset,
        skip_timestamp_check=not run_timestamp_check,
        repo_id=repo_id,
        episodes=episode_ids,
        delta_timestamps=delta_ts,
    )
    if run_timestamp_check:
        _mark_timestamp_check_done(repo_id)
        logger.info("WorldModel timestamp check completed and marked done for repo_id=%s", repo_id)
    valid = _build_valid_frame_indices(base, stack_len)

    transforms_list = [
        *data_config.repack_transforms.inputs,
        *data_config.data_transforms.inputs,
        _transforms.Normalize(data_config.norm_stats, use_quantiles=data_config.use_quantile_norm),
        *data_config.model_transforms.inputs,
    ]
    sample_transform = _transforms.compose(transforms_list)

    prompt_tasks = meta.tasks if data_config.prompt_from_task else None

    ds = WorldModelLeRobotDataset(
        base,
        valid_indices=valid,
        stack_len=stack_len,
        sample_transform=sample_transform,
        wm_cfg=wm_cfg,
        action_dim=model_config.action_dim,
        pi0_action_horizon=model_config.action_horizon,
        lerobot_prompt_tasks=prompt_tasks,
    )
    return ds, stack_len
