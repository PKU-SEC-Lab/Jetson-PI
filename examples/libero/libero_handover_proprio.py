
from __future__ import annotations

import json
import pathlib
from typing import Literal

import numpy as np

ModelActionDim = 32


def _libero_delta_actions_numpy(ap_pad: np.ndarray, st_pad: np.ndarray) -> np.ndarray:
    ap_pad = np.asarray(ap_pad, dtype=np.float32)
    st_pad = np.asarray(st_pad, dtype=np.float32).reshape(-1)
    mask = np.array([True, True, True, True, True, True, False], dtype=bool)
    dims = 7
    delta = np.where(mask, st_pad[:dims], np.float32(0)).astype(np.float32)
    out = np.array(ap_pad, dtype=np.float32, copy=True)
    out[:, :dims] -= delta
    return out


def _pad_to_dim(x: np.ndarray, dim: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    if x.shape[0] >= dim:
        return x[:dim].astype(np.float32, copy=False)
    out = np.zeros((dim,), dtype=np.float32)
    out[: x.shape[0]] = x
    return out


def _zscore(x: np.ndarray, mean: list | np.ndarray, std: list | np.ndarray) -> np.ndarray:
    m = np.asarray(mean, dtype=np.float64)
    s = np.asarray(std, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    return ((x - m) / (s + 1e-6)).astype(np.float32)


def _unzscore(x: np.ndarray, mean: list | np.ndarray, std: list | np.ndarray) -> np.ndarray:
    m = np.asarray(mean, dtype=np.float64)
    s = np.asarray(std, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    return (x * s + m).astype(np.float32)


def load_pi0_libero_norm_stats(checkpoint_parent: pathlib.Path | str) -> tuple[dict, dict]:
    ckpt = pathlib.Path(checkpoint_parent)
    p = ckpt / "assets" / "physical-intelligence" / "libero" / "norm_stats.json"
    if not p.is_file():
        raise FileNotFoundError(f"norm_stats not found: {p}")
    raw = json.loads(p.read_text(encoding="utf-8"))
    ns = raw["norm_stats"]
    return ns["state"], ns["actions"]


def rollforward_np(state_n: np.ndarray, actions_n: np.ndarray, mask: np.ndarray) -> np.ndarray:
    state_n = np.asarray(state_n, dtype=np.float32).reshape(-1)
    actions_n = np.asarray(actions_n, dtype=np.float32)
    mask = np.asarray(mask, dtype=np.float32).reshape(-1)
    acc = np.sum(mask[:, None] * actions_n, axis=0)
    d = min(int(state_n.shape[-1]), int(acc.shape[-1]))
    if d <= 0:
        return state_n.copy()
    out = state_n.copy()
    out[:d] = out[:d] + acc[:d]
    return out


def last_valid_prefix_action_np(actions: np.ndarray, mask: np.ndarray) -> np.ndarray:
    actions = np.asarray(actions, dtype=np.float32)
    m = np.asarray(mask, dtype=bool).reshape(-1)
    if not np.any(m):
        return actions[0].copy()
    last_i = np.where(m)[0][-1]
    return actions[last_i].copy()


def vlash_last_action_q_np(
    state_n: np.ndarray,
    actions_n: np.ndarray,
    mask: np.ndarray,
    *,
    merge_action_dims: int = 7,
) -> np.ndarray:
    state_n = np.asarray(state_n, dtype=np.float32).reshape(-1)
    last_a = last_valid_prefix_action_np(actions_n, mask)
    d = min(merge_action_dims, int(state_n.shape[-1]), int(last_a.shape[-1]))
    if d <= 0:
        return state_n.copy()
    out = state_n.copy()
    out[:d] = last_a[:d]
    return out


def handover_suffix_state_raw(
    *,
    state_k_raw: np.ndarray,
    action_prefix_raw: np.ndarray,
    state_stats: dict,
    action_stats: dict,
    mode: Literal["vlash_last_action", "future_rollout"],
    raw_proprio_dim: int = 8,
    vlash_merge_action_dims: int = 7,
    apply_delta_actions: bool = True,
) -> np.ndarray:
    st = _pad_to_dim(np.asarray(state_k_raw, dtype=np.float32), ModelActionDim)

    ap = np.asarray(action_prefix_raw, dtype=np.float32)
    if ap.ndim != 2:
        raise ValueError(f"action_prefix_raw expected (L, A), got {ap.shape}")
    ap_pad = np.stack([_pad_to_dim(ap[i], ModelActionDim) for i in range(ap.shape[0])], axis=0)
    if apply_delta_actions:
        ap_pad = _libero_delta_actions_numpy(ap_pad, st)

    st_n = _zscore(st, state_stats["mean"], state_stats["std"])
    ap_n = np.stack(
        [_zscore(ap_pad[i], action_stats["mean"], action_stats["std"]) for i in range(ap_pad.shape[0])],
        axis=0,
    )

    mask = np.ones((ap_n.shape[0],), dtype=bool)
    if mode == "future_rollout":
        q_n = rollforward_np(st_n, ap_n, mask)
    else:
        q_n = vlash_last_action_q_np(st_n, ap_n, mask, merge_action_dims=vlash_merge_action_dims)

    q_full = _unzscore(q_n, state_stats["mean"], state_stats["std"])
    return np.asarray(q_full[:raw_proprio_dim], dtype=np.float32).copy()
