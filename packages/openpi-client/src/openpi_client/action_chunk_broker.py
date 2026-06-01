from __future__ import annotations

import concurrent.futures
import dataclasses
import logging
from collections import deque
from collections.abc import Callable
from typing import Dict, Optional

import numpy as np
import tree
from typing_extensions import override

from openpi_client import base_policy as _base_policy

logger = logging.getLogger(__name__)


def wm_stitch_n_from_policy_full(full: Dict) -> int | None:  # noqa: UP006
    """``openpi/wm_stitch_n`` from server (multi-round WM); ``None`` if absent."""
    v = full.get("openpi/wm_stitch_n")
    if v is None:
        return None
    return int(np.asarray(v, dtype=np.int64).reshape(()))


@dataclasses.dataclass(frozen=True)
class PrefetchContext:
    """Passed to ``prefetch_obs_hook`` when ``AsyncActionChunkBroker`` starts a background infer."""

    chunk_actions: np.ndarray
    chunk_start_step: int
    async_trigger_step: int
    action_horizon: int
    overlap_skip: int
    wm_stitch_n: int | None = None
    wm_overlap_exec_band: np.ndarray | None = None


class ActionChunkBroker(_base_policy.BasePolicy):
    def __init__(self, policy: _base_policy.BasePolicy, action_horizon: int):
        self._policy = policy
        self._action_horizon = action_horizon
        self._cur_step: int = 0
        self._last_results: Dict[str, np.ndarray] | None = None

    @override
    def infer(self, obs: Dict) -> Dict:  # noqa: UP006
        if self._last_results is None:
            self._last_results = self._policy.infer(obs)
            self._cur_step = 0

        def slicer(x):
            if isinstance(x, np.ndarray):
                return x[self._cur_step, ...]
            return x

        results = tree.map_structure(slicer, self._last_results)
        self._cur_step += 1
        if self._cur_step >= self._action_horizon:
            self._last_results = None
        return results

    @override
    def reset(self) -> None:
        self._policy.reset()
        self._last_results = None
        self._cur_step = 0


def _snapshot_observation(obs: Dict) -> Dict:  # noqa: UP006
    def _copy(x):
        if isinstance(x, np.ndarray):
            return np.array(x, copy=True)
        return x

    return tree.map_structure(_copy, obs)


class AsyncActionChunkBroker(_base_policy.BasePolicy):
    def __init__(
        self,
        policy: _base_policy.BasePolicy,
        action_horizon: int,
        *,
        async_trigger_step: int = 4,
        overlap_skip: int = 0,
        chunk_exec_steps: int | None = None,
        allow_trigger_at_chunk_start: bool = False,
        prefetch_obs_hook: Callable[[Dict, PrefetchContext], Dict] | None = None,
        observation_state_chunk_index: int | None = None,
        prefetch_handover_true_state: bool = False,
        handover_snapshot_hook: Callable[[Dict, PrefetchContext], Dict] | None = None,
        handover_merged_state_fn: Optional[Callable[[Dict, Dict, np.ndarray], np.ndarray]] = None,
    ) -> None:
        if not (0 <= overlap_skip < action_horizon):
            raise ValueError(
                f"overlap_skip must be in [0, action_horizon), got overlap_skip={overlap_skip}, "
                f"action_horizon={action_horizon}."
            )
        if allow_trigger_at_chunk_start:
            valid_trigger = 0 <= async_trigger_step <= action_horizon
            trigger_msg = (
                "async_trigger_step must satisfy 0 <= async_trigger_step <= action_horizon "
                f"(got async_trigger_step={async_trigger_step}, action_horizon={action_horizon})."
            )
        else:
            valid_trigger = 0 < async_trigger_step <= action_horizon
            trigger_msg = (
                "async_trigger_step must satisfy 0 < async_trigger_step <= action_horizon "
                f"(got async_trigger_step={async_trigger_step}, action_horizon={action_horizon})."
            )
        if not valid_trigger:
            raise ValueError(trigger_msg)
        if chunk_exec_steps is not None and not (1 <= chunk_exec_steps <= action_horizon):
            raise ValueError(
                "chunk_exec_steps must be in [1, action_horizon] or None, "
                f"got chunk_exec_steps={chunk_exec_steps}, action_horizon={action_horizon}."
            )
        if observation_state_chunk_index is not None and not (0 <= observation_state_chunk_index < action_horizon):
            raise ValueError(
                "observation_state_chunk_index must be in [0, action_horizon) or None, "
                f"got {observation_state_chunk_index}, action_horizon={action_horizon}."
            )
        if prefetch_handover_true_state and prefetch_obs_hook is not None:
            raise ValueError("prefetch_handover_true_state=True is incompatible with prefetch_obs_hook.")
        if handover_snapshot_hook is not None and not prefetch_handover_true_state:
            raise ValueError("handover_snapshot_hook requires prefetch_handover_true_state=True.")
        if handover_merged_state_fn is not None and not prefetch_handover_true_state:
            raise ValueError("handover_merged_state_fn requires prefetch_handover_true_state=True.")

        self._policy = policy
        self._action_horizon = action_horizon
        self._async_trigger_step = async_trigger_step
        self._overlap_skip = overlap_skip
        self._chunk_exec_steps = chunk_exec_steps
        self._allow_trigger_at_chunk_start = allow_trigger_at_chunk_start
        self._prefetch_obs_hook = prefetch_obs_hook
        self._observation_state_chunk_index = observation_state_chunk_index
        self._prefetch_handover_true_state = prefetch_handover_true_state
        self._handover_snapshot_hook = handover_snapshot_hook
        self._handover_merged_state_fn = handover_merged_state_fn

        self._cur_step: int = 0
        self._chunk_start_step: int = 0
        self._last_results: Dict[str, np.ndarray] | None = None
        self._pending_future: concurrent.futures.Future | None = None
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self._first_chunk_after_reset = True
        self._state_history: list[np.ndarray] = []
        self._saved_visual_snap_at_k: Dict | None = None
        self._saved_handover_action_prefix: np.ndarray | None = None

    def _drain_pending(self) -> None:
        if self._pending_future is None:
            return
        fut = self._pending_future
        self._pending_future = None
        try:
            fut.result(timeout=600)
        except Exception:
            pass

    def _build_obs_in(self, obs: Dict) -> Dict:  # noqa: UP006
        if self._observation_state_chunk_index is None:
            return obs
        st = np.array(obs["observation/state"], copy=True)
        self._state_history.append(st)
        idx = self._observation_state_chunk_index
        if len(self._state_history) > idx:
            obs_in = dict(obs)
            obs_in["observation/state"] = np.array(self._state_history[idx], copy=True)
            return obs_in
        return obs

    @override
    def infer(self, obs: Dict) -> Dict:  # noqa: UP006
        exec_limit = self._chunk_exec_steps if self._chunk_exec_steps is not None else self._action_horizon
        if self._last_results is None:
            if self._observation_state_chunk_index is not None:
                self._state_history = []
            if self._pending_future is not None:
                self._last_results = self._pending_future.result()
                self._pending_future = None
                if self._first_chunk_after_reset or self._chunk_exec_steps is not None:
                    self._cur_step = 0
                    self._chunk_start_step = 0
                else:
                    self._cur_step = self._overlap_skip
                    self._chunk_start_step = self._overlap_skip

        obs_in = self._build_obs_in(obs)

        if self._last_results is None:
            if self._prefetch_handover_true_state and self._saved_visual_snap_at_k is not None:
                merged = _snapshot_observation(self._saved_visual_snap_at_k)
                if self._handover_merged_state_fn is not None:
                    if self._saved_handover_action_prefix is None:
                        raise RuntimeError("handover merge: missing saved action prefix (broker internal error)")
                    merged["observation/state"] = np.asarray(
                        self._handover_merged_state_fn(merged, obs_in, self._saved_handover_action_prefix),
                        dtype=np.float32,
                    )
                else:
                    merged["observation/state"] = np.array(obs_in["observation/state"], copy=True)
                infer_obs = merged
                self._saved_visual_snap_at_k = None
                self._saved_handover_action_prefix = None
            else:
                infer_obs = obs_in
            self._last_results = self._policy.infer(infer_obs)
            if self._first_chunk_after_reset or self._chunk_exec_steps is not None:
                self._cur_step = 0
                self._chunk_start_step = 0
                self._first_chunk_after_reset = False
            else:
                self._cur_step = self._overlap_skip
                self._chunk_start_step = self._overlap_skip

        if (
            not self._prefetch_handover_true_state
            and self._pending_future is None
            and self._cur_step == self._async_trigger_step
        ):
            snap = _snapshot_observation(obs_in)
            if self._prefetch_obs_hook is not None:
                actions = self._last_results["actions"]
                if not isinstance(actions, np.ndarray):
                    actions = np.asarray(actions)
                ctx = PrefetchContext(
                    chunk_actions=actions,
                    chunk_start_step=self._chunk_start_step,
                    async_trigger_step=self._async_trigger_step,
                    action_horizon=self._action_horizon,
                    overlap_skip=self._overlap_skip,
                )
                snap = self._prefetch_obs_hook(snap, ctx)
            self._pending_future = self._executor.submit(self._policy.infer, snap)
        elif self._prefetch_handover_true_state and self._cur_step == self._async_trigger_step:
            actions = self._last_results["actions"]
            if not isinstance(actions, np.ndarray):
                actions = np.asarray(actions)
            snap = _snapshot_observation(obs_in)
            if self._handover_snapshot_hook is not None:
                ctx = PrefetchContext(
                    chunk_actions=actions,
                    chunk_start_step=self._chunk_start_step,
                    async_trigger_step=self._async_trigger_step,
                    action_horizon=self._action_horizon,
                    overlap_skip=self._overlap_skip,
                )
                snap = self._handover_snapshot_hook(snap, ctx)
            self._saved_visual_snap_at_k = snap
            self._saved_handover_action_prefix = np.array(
                np.asarray(actions[self._async_trigger_step : self._action_horizon]),
                dtype=np.float32,
                copy=True,
            )

        def slicer(x):
            if isinstance(x, np.ndarray):
                return x[self._cur_step, ...]
            return x

        results = tree.map_structure(slicer, self._last_results)
        self._cur_step += 1
        if self._cur_step >= exec_limit:
            self._last_results = None
        return results

    @override
    def reset(self) -> None:
        self._policy.reset()
        self._drain_pending()
        self._last_results = None
        self._cur_step = 0
        self._chunk_start_step = 0
        self._first_chunk_after_reset = True
        self._state_history = []
        self._saved_visual_snap_at_k = None
        self._saved_handover_action_prefix = None

    def shutdown(self) -> None:
        self._drain_pending()
        self._executor.shutdown(wait=False)


def _split_leading_horizon(full: Dict, H: int) -> list[Dict]:  # noqa: UP006
    """Split policy outputs whose leading dim is ``H`` into ``H`` per-step dicts."""

    def one(i: int) -> Dict:
        def sel(x):
            if isinstance(x, np.ndarray) and x.ndim >= 1 and int(x.shape[0]) == H:
                # np.asarray(..., copy=) requires NumPy>=2; use np.array for 1.x compat.
                return np.array(x[i, ...], copy=True)
            return x

        return tree.map_structure(sel, full)

    return [one(i) for i in range(H)]


class AsyncActionBufferBroker(_base_policy.BasePolicy):

    def __init__(
        self,
        policy: _base_policy.BasePolicy,
        action_horizon: int,
        *,
        async_trigger_step: int = 4,
        overlap_skip: int = 0,
        chunk_exec_steps: int | None = None,
        allow_trigger_at_chunk_start: bool = False,
        prefetch_obs_hook: Callable[[Dict, PrefetchContext], Dict] | None = None,
        observation_state_chunk_index: int | None = None,
        wm_infer_complete_hook: Callable[[Dict], None] | None = None,
        low_replan_two_phase_fn: Callable[[Dict, Dict], Dict] | None = None,
    ) -> None:
        if chunk_exec_steps is not None:
            raise ValueError("AsyncActionBufferBroker does not support chunk_exec_steps; pass None.")
        if not (0 <= overlap_skip < action_horizon):
            raise ValueError(
                f"overlap_skip must be in [0, action_horizon), got overlap_skip={overlap_skip}, "
                f"action_horizon={action_horizon}."
            )
        if allow_trigger_at_chunk_start:
            valid_trigger = 0 <= async_trigger_step <= action_horizon
            trigger_msg = (
                "async_trigger_step must satisfy 0 <= async_trigger_step <= action_horizon "
                f"(got async_trigger_step={async_trigger_step}, action_horizon={action_horizon})."
            )
        else:
            valid_trigger = 0 < async_trigger_step <= action_horizon
            trigger_msg = (
                "async_trigger_step must satisfy 0 < async_trigger_step <= action_horizon "
                f"(got async_trigger_step={async_trigger_step}, action_horizon={action_horizon})."
            )
        if not valid_trigger:
            raise ValueError(trigger_msg)
        if observation_state_chunk_index is not None and not (0 <= observation_state_chunk_index < action_horizon):
            raise ValueError(
                "observation_state_chunk_index must be in [0, action_horizon) or None, "
                f"got {observation_state_chunk_index}, action_horizon={action_horizon}."
            )

        self._policy = policy
        self._action_horizon = action_horizon
        self._async_trigger_step = int(async_trigger_step)
        self._overlap_skip = int(overlap_skip)
        self._prefetch_obs_hook = prefetch_obs_hook
        self._observation_state_chunk_index = observation_state_chunk_index
        self._wm_infer_complete_hook = wm_infer_complete_hook
        self._low_replan_two_phase_fn = low_replan_two_phase_fn

        self._buf: deque[Dict] = deque()
        self._last_infer_full: np.ndarray | None = None
        self._last_wm_stitch_n: int | None = None
        self._last_wm_overlap_exec_band: np.ndarray | None = None
        self._pending_future: concurrent.futures.Future | None = None
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self._state_history: list[np.ndarray] = []
        # κ-adaptive early WM stop (``openpi/wm_adaptive_early_stop``): enqueue only ``[0:L)``, stash
        # ``[L:L+overlap)`` for glue before ``[overlap:H)`` of the next infer; prefetch at obs after L pops.
        self._wm_adapt_pops_left: int = 0
        self._wm_adapt_overlap_parts: list[Dict] | None = None
        self._wm_adapt_prefetch_on_next: bool = False
        self._wm_adapt_glue_next: bool = False
        #: Snapshot of the dict passed to ``policy.infer`` for the in-flight prefetch (for WM low-replan two-phase).
        self._prefetch_two_phase_obs_snap: Dict | None = None

    def _maybe_apply_low_replan_two_phase(self, obs_snap: Dict, full: Dict) -> Dict:  # noqa: UP006
        if not bool(full.get("openpi/wm_low_replan_two_phase")):
            return full
        if self._low_replan_two_phase_fn is None:
            raise RuntimeError(
                "Policy returned openpi/wm_low_replan_two_phase=True but AsyncActionBufferBroker was constructed "
                "with low_replan_two_phase_fn=None."
            )
        return dict(self._low_replan_two_phase_fn(obs_snap, full))

    def _invoke_wm_infer_complete_hook(self, full: Dict) -> None:  # noqa: UP006
        if self._wm_infer_complete_hook is None:
            return
        try:
            self._wm_infer_complete_hook(full)
        except Exception:
            logger.exception("wm_infer_complete_hook raised")

    def _drain_pending(self) -> None:
        if self._pending_future is None:
            return
        fut = self._pending_future
        self._pending_future = None
        try:
            fut.result(timeout=600)
        except Exception:
            pass

    def _build_obs_in(self, obs: Dict) -> Dict:  # noqa: UP006
        if self._observation_state_chunk_index is None:
            return obs
        st = np.array(obs["observation/state"], copy=True)
        self._state_history.append(st)
        idx = self._observation_state_chunk_index
        if len(self._state_history) > idx:
            obs_in = dict(obs)
            obs_in["observation/state"] = np.array(self._state_history[idx], copy=True)
            return obs_in
        return obs

    def _append_from_full_first(self, full: Dict) -> None:  # noqa: UP006
        parts = _split_leading_horizon(full, self._action_horizon)
        self._buf.extend(parts)
        acts = full.get("actions")
        if acts is not None:
            a = np.asarray(acts, dtype=np.float32)
            H, O = self._action_horizon, self._overlap_skip
            self._last_infer_full = np.array(a, dtype=np.float32, copy=True)
            self._last_wm_stitch_n = wm_stitch_n_from_policy_full(full)
            self._last_wm_overlap_exec_band = np.array(a[H - O : H], dtype=np.float32, copy=True)

    def _append_from_full_merged(self, full: Dict) -> None:  # noqa: UP006
        parts = _split_leading_horizon(full, self._action_horizon)
        for j in range(self._overlap_skip, self._action_horizon):
            self._buf.append(parts[j])
        acts = full.get("actions")
        if acts is not None:
            a = np.asarray(acts, dtype=np.float32)
            H, O = self._action_horizon, self._overlap_skip
            self._last_infer_full = np.array(a, dtype=np.float32, copy=True)
            self._last_wm_stitch_n = wm_stitch_n_from_policy_full(full)
            self._last_wm_overlap_exec_band = np.array(a[H - O : H], dtype=np.float32, copy=True)

    def _append_adaptive_wm_early(self, full: Dict) -> None:  # noqa: UP006
        """Partial enqueue + stash overlap band for glue-merge after prefetch (obs after L pops)."""
        H = self._action_horizon
        o = self._overlap_skip
        if not bool(full.get("openpi/wm_adaptive_early_stop")):
            raise RuntimeError("internal: _append_adaptive_wm_early without adaptive flag")
        L = int(full["openpi/wm_adaptive_exec_len"])
        if not (0 < L < H):
            raise RuntimeError(f"wm_adaptive_exec_len must be in (0,H), got L={L}, H={H}")
        if L + o > H:
            raise RuntimeError(f"wm_adaptive L+overlap must be <= H, got L={L}, overlap={o}, H={H}")
        parts = _split_leading_horizon(full, H)
        for i in range(L):
            self._buf.append(parts[i])
        self._wm_adapt_overlap_parts = [parts[i] for i in range(L, L + o)]
        self._wm_adapt_pops_left = L
        self._wm_adapt_prefetch_on_next = False
        self._wm_adapt_glue_next = False
        acts = full.get("actions")
        if acts is not None:
            a = np.asarray(acts, dtype=np.float32)
            self._last_infer_full = np.array(a, dtype=np.float32, copy=True)
            self._last_wm_stitch_n = wm_stitch_n_from_policy_full(full)
            self._last_wm_overlap_exec_band = np.array(a[L : L + o], dtype=np.float32, copy=True)

    def _append_adaptive_glue_merge(self, full: Dict) -> None:  # noqa: UP006
        """After early-stop prefetch: old ``[L:L+overlap)`` at front, then new ``[overlap:H)`` at end."""
        if self._wm_adapt_overlap_parts is None:
            raise RuntimeError("internal: adaptive glue without stashed overlap parts")
        H = self._action_horizon
        o = self._overlap_skip
        parts = _split_leading_horizon(full, H)
        for p in reversed(self._wm_adapt_overlap_parts):
            self._buf.appendleft(p)
        for j in range(o, H):
            self._buf.append(parts[j])
        self._wm_adapt_overlap_parts = None
        self._wm_adapt_pops_left = 0
        acts = full.get("actions")
        if acts is not None:
            a = np.asarray(acts, dtype=np.float32)
            H, O = self._action_horizon, self._overlap_skip
            self._last_infer_full = np.array(a, dtype=np.float32, copy=True)
            self._last_wm_stitch_n = wm_stitch_n_from_policy_full(full)
            self._last_wm_overlap_exec_band = np.array(a[H - O : H], dtype=np.float32, copy=True)

    def _consume_infer_full(self, full: Dict) -> None:  # noqa: UP006
        if bool(full.get("openpi/wm_adaptive_early_stop")):
            self._append_adaptive_wm_early(full)
        else:
            self._append_from_full_merged(full)

    def _blocking_refill(self, obs: Dict) -> None:  # noqa: UP006
        obs_in = self._build_obs_in(obs)
        full = self._policy.infer(obs_in)
        full = self._maybe_apply_low_replan_two_phase(_snapshot_observation(obs_in), full)
        self._invoke_wm_infer_complete_hook(full)
        if bool(full.get("openpi/wm_adaptive_early_stop")):
            self._append_adaptive_wm_early(full)
        elif self._last_infer_full is not None:
            self._append_from_full_merged(full)
        else:
            self._append_from_full_first(full)

    def _start_prefetch(self, obs: Dict) -> None:  # noqa: UP006
        obs_in = self._build_obs_in(obs)
        snap = _snapshot_observation(obs_in)
        if self._prefetch_obs_hook is not None:
            if self._last_infer_full is None:
                raise RuntimeError("AsyncActionBufferBroker: prefetch_hook requires prior infer (last_infer_full).")
            ctx = PrefetchContext(
                chunk_actions=self._last_infer_full,
                chunk_start_step=0,
                async_trigger_step=self._async_trigger_step,
                action_horizon=self._action_horizon,
                overlap_skip=self._overlap_skip,
                wm_stitch_n=self._last_wm_stitch_n,
                wm_overlap_exec_band=self._last_wm_overlap_exec_band,
            )
            snap = self._prefetch_obs_hook(snap, ctx)
        self._prefetch_two_phase_obs_snap = _snapshot_observation(snap)
        self._pending_future = self._executor.submit(self._policy.infer, snap)

    @override
    def infer(self, obs: Dict) -> Dict:  # noqa: UP006
        if self._wm_adapt_prefetch_on_next:
            self._start_prefetch(obs)
            self._wm_adapt_prefetch_on_next = False
            self._wm_adapt_glue_next = True

        if self._pending_future is not None and self._pending_future.done():
            full = self._pending_future.result(timeout=600)
            self._pending_future = None
            snap = self._prefetch_two_phase_obs_snap
            self._prefetch_two_phase_obs_snap = None
            obs_in = self._build_obs_in(obs)
            full = self._maybe_apply_low_replan_two_phase(
                snap if snap is not None else _snapshot_observation(obs_in), full
            )
            self._invoke_wm_infer_complete_hook(full)
            if self._wm_adapt_glue_next:
                self._append_adaptive_glue_merge(full)
                self._wm_adapt_glue_next = False
            else:
                self._consume_infer_full(full)

        while len(self._buf) == 0:
            if self._pending_future is not None:
                full = self._pending_future.result(timeout=600)
                self._pending_future = None
                snap = self._prefetch_two_phase_obs_snap
                self._prefetch_two_phase_obs_snap = None
                obs_in = self._build_obs_in(obs)
                full = self._maybe_apply_low_replan_two_phase(
                    snap if snap is not None else _snapshot_observation(obs_in), full
                )
                self._invoke_wm_infer_complete_hook(full)
                if self._wm_adapt_glue_next:
                    self._append_adaptive_glue_merge(full)
                    self._wm_adapt_glue_next = False
                else:
                    self._consume_infer_full(full)
            else:
                self._blocking_refill(obs)

        need = self._action_horizon - self._async_trigger_step
        if (
            self._pending_future is None
            and len(self._buf) == need
            and not self._wm_adapt_prefetch_on_next
            and self._wm_adapt_overlap_parts is None
        ):
            self._start_prefetch(obs)

        out = self._buf.popleft()
        if self._wm_adapt_pops_left > 0:
            self._wm_adapt_pops_left -= 1
            if self._wm_adapt_pops_left == 0:
                self._wm_adapt_prefetch_on_next = True
        return out

    @override
    def reset(self) -> None:
        self._policy.reset()
        self._drain_pending()
        self._buf.clear()
        self._last_infer_full = None
        self._last_wm_stitch_n = None
        self._last_wm_overlap_exec_band = None
        self._pending_future = None
        self._state_history = []
        self._wm_adapt_pops_left = 0
        self._wm_adapt_overlap_parts = None
        self._wm_adapt_prefetch_on_next = False
        self._wm_adapt_glue_next = False
        self._prefetch_two_phase_obs_snap = None

    def shutdown(self) -> None:
        self._drain_pending()
        self._executor.shutdown(wait=False)
