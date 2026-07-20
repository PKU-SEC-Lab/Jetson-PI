from collections.abc import Callable
import dataclasses
import json
import logging
import math
import pathlib
import sys
from typing import Literal

import imageio
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import numpy as np
from openpi_client import action_chunk_broker
from openpi_client import image_tools
from openpi_client import rapid_trigger
from openpi_client import task_c_trace
from openpi_client import websocket_client_policy as _websocket_client_policy
import tqdm
import tyro

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256
OPENPI_ASYNC_KEY = "openpi/async"


def _wm_multi_stitch_seed_and_prev_tail(
    prev_full: np.ndarray,
    *,
    H: int,  # noqa: N803 - mathematical horizon notation used throughout this released scheduler
    O: int,  # noqa: E741,N803 - mathematical overlap notation used throughout this released scheduler
    delta_idx: int,
    stitch_n: int | None,
    default_n: int,
    overlap_exec_band: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    patched = np.array(prev_full, dtype=np.float32, copy=True)
    if overlap_exec_band is not None:
        band = np.asarray(overlap_exec_band, dtype=np.float32)
        if band.ndim != 2 or int(band.shape[0]) != O or int(band.shape[1]) != int(prev_full.shape[1]):
            raise ValueError(
                "wm_multi stitch: overlap_exec_band must be (O, A) matching overlap_skip and prev_full, "
                f"got band.shape={band.shape}, O={O}, prev_full.shape={prev_full.shape}"
            )
        band = np.array(band, dtype=np.float32, copy=True)
    else:
        n = int(stitch_n) if stitch_n is not None else int(default_n)
        if n < 1:
            raise ValueError(f"wm_multi stitch: n must be >= 1, got {n}")
        lo = O + (n - 1) * delta_idx
        hi = 2 * O + (n - 1) * delta_idx
        if lo < 0 or hi > H or hi - lo != O:
            raise ValueError(
                "wm_multi stitch: invalid band [O+(n-1)*Δ : 2*O+(n-1)*Δ) "
                f"(O={O}, delta_idx={delta_idx}, n={n}, lo={lo}, hi={hi}, H={H})."
            )
        band = np.array(prev_full[lo:hi], dtype=np.float32, copy=True)
    patched[0:O] = band
    return patched, band


def _mean_int(xs: list[int]) -> float:
    return float(sum(xs) / len(xs)) if xs else float("nan")


_LIBERO_HANDOVER_MODES = frozenset({"handover_true", "handover_vlash_last_action", "handover_future_rollout"})


def _make_libero_handover_norm_proxy_fn(
    checkpoint_parent: pathlib.Path,
    mode: Literal["vlash_last_action", "future_rollout"],
) -> Callable[[dict, dict, np.ndarray], np.ndarray]:
    ex = pathlib.Path(__file__).resolve().parent
    if str(ex) not in sys.path:
        sys.path.insert(0, str(ex))
    import libero_handover_proprio as lhp

    st_s, ac_s = lhp.load_pi0_libero_norm_stats(checkpoint_parent)

    def _fn(snap_k: dict, _obs_now: dict, ap: np.ndarray) -> np.ndarray:
        sk = np.asarray(snap_k["observation/state"], dtype=np.float32)
        return lhp.handover_suffix_state_raw(
            state_k_raw=sk,
            action_prefix_raw=ap,
            state_stats=st_s,
            action_stats=ac_s,
            mode=mode,
        )

    return _fn


def _pack_libero_proprio_vector(obs: dict) -> np.ndarray:
    return np.concatenate(
        (
            obs["robot0_eef_pos"],
            _quat2axisangle(obs["robot0_eef_quat"]),
            obs["robot0_gripper_qpos"],
        )
    ).astype(np.float32)


class _WmAeInferRoundStatsLogger:
    __slots__ = (
        "_kappa_drop_edge_den",
        "_kappa_drop_edge_num",
        "_n_exception_fallback",
        "_n_infers_kappa_vec",
        "_n_infers_wm",
        "_n_low_replan_fallback",
        "_rounds_sum",
        "_sum_infer_mean_kappa",
    )

    def __init__(self) -> None:
        self._n_infers_wm = 0
        self._rounds_sum = 0
        self._sum_infer_mean_kappa = 0.0
        self._n_infers_kappa_vec = 0
        self._kappa_drop_edge_num = 0.0
        self._kappa_drop_edge_den = 0
        self._n_low_replan_fallback = 0
        self._n_exception_fallback = 0

    def on_infer_complete(self, full: dict) -> None:
        sn = action_chunk_broker.wm_stitch_n_from_policy_full(full)
        low_fb = bool(full.get("openpi/wm_low_replan_fallback_full_pi0"))
        exc_fb = bool(full.get("openpi/wm_exception_fallback_full_pi0"))
        if sn is None and not low_fb and not exc_fb:
            return

        self._n_infers_wm += 1
        if low_fb:
            self._n_low_replan_fallback += 1
        elif exc_fb:
            self._n_exception_fallback += 1

        partial_raw = full.get("openpi/wm_low_replan_partial_wm_ae_rounds")
        partial_wm_ae = None if partial_raw is None else int(np.asarray(partial_raw, dtype=np.int64).reshape(()))

        if sn is not None:
            self._rounds_sum += int(sn)
            stitch_last = int(sn)
        elif low_fb and partial_wm_ae is not None:
            self._rounds_sum += int(partial_wm_ae)
            stitch_last = int(partial_wm_ae)
        else:
            stitch_last = 0

        run_stitch = self._rounds_sum / self._n_infers_wm

        kap = full.get("openpi/wm_confidence_kappa")
        k = np.asarray(kap, dtype=np.float64).reshape(-1) if kap is not None else np.empty(0, dtype=np.float64)
        if k.size > 0:
            this_mean_k = float(np.mean(k))
            self._sum_infer_mean_kappa += this_mean_k
            self._n_infers_kappa_vec += 1
            run_mean_k = self._sum_infer_mean_kappa / self._n_infers_kappa_vec
        else:
            this_mean_k = float("nan")
            run_mean_k = (
                self._sum_infer_mean_kappa / self._n_infers_kappa_vec if self._n_infers_kappa_vec > 0 else float("nan")
            )

        this_mean_drop_s = "na"
        run_mean_drop_s = "na"
        if k.size >= 2:
            diffs = k[:-1] - k[1:]
            edge_mass = float(np.sum(diffs))
            n_edges = int(k.size - 1)
            drop = edge_mass / float(n_edges)
            this_mean_drop_s = f"{drop:.6f}"
            self._kappa_drop_edge_num += edge_mass
            self._kappa_drop_edge_den += n_edges
            run_mean_drop_s = (
                f"{self._kappa_drop_edge_num / float(self._kappa_drop_edge_den):.6f}"
                if self._kappa_drop_edge_den > 0
                else "na"
            )
        elif k.size == 1:
            this_mean_drop_s = "0.000000"
            if self._kappa_drop_edge_den > 0:
                run_mean_drop_s = f"{self._kappa_drop_edge_num / float(self._kappa_drop_edge_den):.6f}"
            else:
                run_mean_drop_s = "na"

        low_partial_s = str(partial_wm_ae) if (low_fb and partial_wm_ae is not None) else "na"
        logging.info(
            "wm_ae_infer_stats: wm_multi_infer_idx=%d openpi_wm_stitch_n_last=%d "
            "low_replan_partial_wm_ae_rounds=%s "
            "cumulative_wm_ae_rounds=%d running_mean_wm_ae_rounds_per_llm_infer=%.6f "
            "this_infer_mean_kappa_rounds=%.6f this_infer_mean_inter_round_kappa_drop=%s "
            "running_mean_kappa_across_rounds_per_infer=%.6f "
            "running_mean_inter_round_kappa_drop_per_infer=%s "
            "(kappa_drop_wm_inter_round_edges=%d)",
            self._n_infers_wm,
            stitch_last,
            low_partial_s,
            self._rounds_sum,
            run_stitch,
            this_mean_k,
            this_mean_drop_s,
            run_mean_k,
            run_mean_drop_s,
            int(self._kappa_drop_edge_den),
        )


def _libero_oracle_handover_prefetch_hook(env_holder: dict):
    def hook(obs: dict, ctx: action_chunk_broker.PrefetchContext) -> dict:
        env = env_holder.get("env")
        if env is None:
            raise RuntimeError("oracle_handover prefetch: env_holder['env'] not set")
        out = dict(obs)
        K, H = int(ctx.async_trigger_step), int(ctx.action_horizon)  # noqa: N806 - published K/H notation
        ap = np.asarray(ctx.chunk_actions[K:H], dtype=np.float32)
        saved = env.get_sim_state()
        rob_obs = None
        try:
            for j in range(ap.shape[0]):
                rob_obs, _, _, _ = env.step(ap[j].tolist())
            if rob_obs is None:
                raise RuntimeError("oracle_handover: empty action prefix")
            out["observation/state"] = _pack_libero_proprio_vector(rob_obs)
        finally:
            env.set_state(saved)
            env.env.sim.forward()
            env._post_process()  # noqa: SLF001 - robosuite exposes no public equivalent after state restore
            env._update_observables(force=True)  # noqa: SLF001 - same state-restore boundary
        return out

    return hook


def _libero_wm_prefetch_hook(async_key: str, *, ae_proprio_source: str | None = None):
    def hook(obs: dict, ctx: action_chunk_broker.PrefetchContext) -> dict:
        out = dict(obs)
        K, H = int(ctx.async_trigger_step), int(ctx.action_horizon)  # noqa: N806 - published K/H notation
        ap = np.asarray(ctx.chunk_actions[K:H], dtype=np.float32)
        if ap.shape[0] == 0:
            raise RuntimeError(
                "WM prefetch: empty action tail (need async_trigger_step < action_horizon, "
                f"got async_trigger_step={K}, action_horizon={H})."
            )
        eff_ae = "prefix_t" if ae_proprio_source is None else ae_proprio_source
        out[async_key] = {
            "use_world_model": True,
            "action_prefix": ap,
            "prefix_mask": np.ones((ap.shape[0],), dtype=bool),
            "delta_t": float(H - K),
            "wm_proprio_t": np.asarray(obs["observation/state"], dtype=np.float32),
            "ae_proprio_source": eff_ae,
        }
        return out

    return hook


def _libero_wm_multi_rollout_prefetch_hook(
    async_key: str,
    *,
    num_rounds: int,
    wm_rollout_delta_t: float,
    ae_proprio_source: str | None = None,
):
    def hook(obs: dict, ctx: action_chunk_broker.PrefetchContext) -> dict:
        out = dict(obs)
        H = int(ctx.action_horizon)  # noqa: N806 - published horizon notation
        O = int(ctx.overlap_skip)  # noqa: E741,N806 - published overlap notation
        delta_idx = round(float(wm_rollout_delta_t))
        if O + (num_rounds - 1) * delta_idx > H:
            raise ValueError(
                "wm_multi_rollout prefetch: need overlap_skip + (num_rounds-1)*round(wm_rollout_delta_t) <= "
                f"action_horizon (got overlap_skip={O}, num_rounds={num_rounds}, wm_rollout_delta_t={wm_rollout_delta_t}, "
                f"delta_idx={delta_idx}, H={H})."
            )
        if ctx.wm_overlap_exec_band is None and 2 * O + (num_rounds - 1) * delta_idx > H:
            raise ValueError(
                "wm_multi_rollout prefetch: stitch band needs 2*overlap_skip + (num_rounds-1)*delta_idx <= "
                f"action_horizon (got O={O}, num_rounds={num_rounds}, delta_idx={delta_idx}, H={H})."
            )
        seed = np.asarray(ctx.chunk_actions, dtype=np.float32)
        if int(seed.shape[0]) != H:
            raise ValueError(f"wm_multi_rollout prefetch: chunk_actions must have length H={H}, got {seed.shape[0]}")
        eff_ae = "prefix_t" if ae_proprio_source is None else ae_proprio_source
        wm_seed, prev_tail = _wm_multi_stitch_seed_and_prev_tail(
            seed,
            H=H,
            O=O,
            delta_idx=delta_idx,
            stitch_n=ctx.wm_stitch_n,
            default_n=int(num_rounds),
            overlap_exec_band=ctx.wm_overlap_exec_band,
        )
        out[async_key] = {
            "use_world_model": True,
            "wm_multi_rollout": {
                "enabled": True,
                "num_rounds": int(num_rounds),
                "delta_t": float(wm_rollout_delta_t),
                "overlap": O,
                "prev_chunk_tail": prev_tail,
            },
            "wm_seed_chunk": wm_seed,
            "wm_proprio_t": np.asarray(obs["observation/state"], dtype=np.float32),
            "ae_proprio_source": eff_ae,
        }
        return out

    return hook


def _libero_wm_multi_rollout_adaptive_prefetch_hook(
    async_key: str,
    *,
    wm_rollout_delta_t: float,
    kappa_delta: float,
    ae_proprio_source: str | None = None,
    adaptive_kappa_low_replan: bool = False,
    routing_policy: rapid_trigger.RoutingPolicy = "kappa",
    rapid_observer: rapid_trigger.RapidKinematicTrigger | None = None,
):
    def hook(obs: dict, ctx: action_chunk_broker.PrefetchContext) -> dict:
        from openpi.policies.wm_multi_rollout_schedule import wm_multi_rollout_adaptive_max_rounds

        out = dict(obs)
        H = int(ctx.action_horizon)  # noqa: N806 - published horizon notation
        O = int(ctx.overlap_skip)  # noqa: E741,N806 - published overlap notation
        delta_idx = round(float(wm_rollout_delta_t))
        max_r = wm_multi_rollout_adaptive_max_rounds(h=H, overlap=O, delta_idx=delta_idx)
        if O + (max_r - 1) * delta_idx > H:
            raise ValueError(
                "wm_multi_rollout adaptive prefetch: need overlap_skip + (N-1)*delta_idx <= action_horizon "
                f"(got overlap_skip={O}, max_r={max_r}, wm_rollout_delta_t={wm_rollout_delta_t}, delta_idx={delta_idx}, H={H})."
            )
        if ctx.wm_overlap_exec_band is None and 2 * O + (max_r - 1) * delta_idx > H:
            raise ValueError(
                "wm_multi_rollout adaptive prefetch: stitch band needs 2*overlap_skip + (max_r-1)*delta_idx <= "
                f"action_horizon (got O={O}, max_r={max_r}, delta_idx={delta_idx}, H={H})."
            )
        seed = np.asarray(ctx.chunk_actions, dtype=np.float32)
        if int(seed.shape[0]) != H:
            raise ValueError(f"wm_multi_rollout prefetch: chunk_actions must have length H={H}, got {seed.shape[0]}")
        eff_ae = "prefix_t" if ae_proprio_source is None else ae_proprio_source
        wm_seed, prev_tail = _wm_multi_stitch_seed_and_prev_tail(
            seed,
            H=H,
            O=O,
            delta_idx=delta_idx,
            stitch_n=ctx.wm_stitch_n,
            default_n=int(max_r),
            overlap_exec_band=ctx.wm_overlap_exec_band,
        )
        rapid_payload = None
        if rapid_observer is not None:
            latest = rapid_observer.latest_decision
            if latest is None:
                raise RuntimeError("RAPID observer has no real-proprio sample at prefetch")
            latest = rapid_trigger.apply_gripper_command_veto(latest, seed[:, 6])
            rapid_payload = latest.as_dict()
        if routing_policy == "rapid" and rapid_payload is None:
            raise RuntimeError("RAPID routing requires a configured kinematic observer")
        out[async_key] = {
            "use_world_model": True,
            "wm_multi_rollout": {
                "enabled": True,
                "adaptive_kappa": True,
                "adaptive_kappa_low_replan": bool(adaptive_kappa_low_replan),
                "routing_policy": routing_policy,
                "rapid": rapid_payload,
                "kappa_delta": float(kappa_delta),
                "num_rounds": int(max_r),
                "delta_t": float(wm_rollout_delta_t),
                "overlap": O,
                "prev_chunk_tail": prev_tail,
            },
            "wm_seed_chunk": wm_seed,
            "wm_proprio_t": np.asarray(obs["observation/state"], dtype=np.float32),
            "ae_proprio_source": eff_ae,
        }
        return out

    return hook


def _make_wm_low_replan_two_phase_fn(
    *,
    client: _websocket_client_policy.WebsocketClientPolicy,
    env_holder: dict,
    args,  # tyro Args (``Args``); defined below to avoid forward-ref issues
    openpi_async_key: str,
) -> Callable[[dict, dict], dict]:
    def low_replan_two_phase(_obs_snap: dict, p1: dict) -> dict:
        if not bool(p1.get("openpi/wm_low_replan_two_phase")):
            return p1
        env = env_holder.get("env")
        if env is None:
            raise RuntimeError("wm low_replan_two_phase: env_holder['env'] not set")
        task_description = str(env_holder.get("task_description") or "")
        roll = np.asarray(p1["openpi/wm_low_replan_rollout_actions"], dtype=np.float32)
        glue = np.asarray(p1["openpi/wm_low_replan_glue_actions"], dtype=np.float32)
        n_roll = int(np.asarray(p1["openpi/wm_low_replan_rollout_len"], dtype=np.int64).reshape(()))
        if int(roll.shape[0]) != n_roll:
            raise RuntimeError(f"wm low_replan_two_phase: rollout_len={n_roll} vs roll.shape[0]={roll.shape[0]}")
        H = int(args.action_horizon)  # noqa: N806 - published horizon notation
        O = int(args.overlap_skip)  # noqa: E741,N806 - published overlap notation
        if int(glue.shape[0]) != O:
            raise RuntimeError(f"wm low_replan_two_phase: glue rows {glue.shape[0]} != overlap_skip={O}")
        obs = None
        for i in range(n_roll):
            arow = roll[i]
            state_before = None
            task_c_recorder = env_holder.get("task_c_recorder")
            task_c_context = _obs_snap.get(task_c_trace.TRACE_KEY)
            if task_c_recorder is not None and task_c_context is not None:
                latest_obs = env_holder.get("task_c_latest_obs")
                if latest_obs is None:
                    raise task_c_trace.TaskCTraceError("missing latest LIBERO observation for low-replan trace")
                state_before = _pack_libero_proprio_vector(latest_obs)
            obs, _reward, done, _info = env.step(np.asarray(arow, dtype=np.float32).reshape(-1).tolist())
            env_holder["task_c_latest_obs"] = obs
            if task_c_recorder is not None and task_c_context is not None:
                base_context = task_c_trace.validate_trace_context(task_c_context)
                hidden_context = {
                    **base_context,
                    "env_step": int(env_holder.get("task_c_sim_step", 0)),
                }
                task_c_recorder.record_step(
                    hidden_context,
                    source="low_replan_rollout",
                    task_description=task_description,
                    action=arow,
                    state_before=state_before,
                    state_after=_pack_libero_proprio_vector(obs),
                    done_after_step=bool(done),
                    policy_kappa=None,
                )
                env_holder["task_c_sim_step"] = int(env_holder.get("task_c_sim_step", 0)) + 1
            if done:
                break
        if obs is None:
            raise RuntimeError("wm low_replan_two_phase: rollout produced no env observation")
        img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
        wri = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
        img = image_tools.convert_to_uint8(image_tools.resize_with_pad(img, args.resize_size, args.resize_size))
        wri = image_tools.convert_to_uint8(image_tools.resize_with_pad(wri, args.resize_size, args.resize_size))
        plain = {
            "observation/image": img,
            "observation/wrist_image": wri,
            "observation/state": _pack_libero_proprio_vector(obs),
            "prompt": task_description,
        }
        if task_c_trace.TRACE_KEY in _obs_snap:
            # Preserve the original trigger-step identity on the mandatory
            # fresh-VLM half of a low-kappa two-phase replan, but advance its
            # simulator-step coordinate past the just-executed rollout.
            low_replan_context = dict(_obs_snap[task_c_trace.TRACE_KEY])
            low_replan_context["env_step"] = int(env_holder.get("task_c_sim_step", 0))
            plain[task_c_trace.TRACE_KEY] = low_replan_context
        p2 = client.infer(plain)
        a2 = np.asarray(p2["actions"], dtype=np.float32)
        if a2.ndim != 2 or int(a2.shape[0]) != H:
            raise RuntimeError(f"wm low_replan_two_phase: expected Pi0 actions (H, A) with H={H}, got {a2.shape}")
        merged = np.concatenate([glue, a2[O:]], axis=0).astype(np.float32, copy=False)
        if int(merged.shape[0]) != H:
            raise RuntimeError(f"wm low_replan_two_phase: merged len {merged.shape[0]} != H={H}")
        out = dict(p2)
        out["actions"] = merged
        out["openpi/wm_low_replan_fallback_full_pi0"] = True
        out["openpi/wm_low_replan_partial_wm_ae_rounds"] = p1.get("openpi/wm_low_replan_partial_wm_ae_rounds")
        out["openpi/wm_confidence_kappa"] = p1.get("openpi/wm_confidence_kappa")
        out.pop("openpi/wm_low_replan_two_phase", None)
        out.pop(openpi_async_key, None)
        return out

    return low_replan_two_phase


def _validate_async_chunk_params(
    *,
    action_horizon: int,
    overlap_skip: int,
    async_trigger_step: int,
    chunk_exec_steps: int | None = None,
    allow_trigger_at_chunk_start: bool = False,
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
            "async_chunk_exec_steps must be in [1, action_horizon] or None, "
            f"got async_chunk_exec_steps={chunk_exec_steps}, action_horizon={action_horizon}."
        )


@dataclasses.dataclass
class Args:
    host: str = "0.0.0.0"
    port: int = 8000
    resize_size: int = 224
    action_horizon: int = 5
    async_inference: bool = False
    async_trigger_step: int = 3
    overlap_skip: int = 0
    async_use_world_model: bool = False
    async_chunk_exec_steps: int | None = None
    async_allow_trigger_at_chunk_start: bool = False
    async_prefetch_proprio: Literal[
        "trigger",
        "oracle_handover",
        "handover_true",
        "handover_vlash_last_action",
        "handover_future_rollout",
    ] = "trigger"
    pi0_norm_checkpoint_dir: str = "PATH/TO/CHECKPOINT/pi05_libero"
    async_proprio_state_chunk_index: int | None = None
    async_use_proprio_state_at_h_minus_k: bool = False
    async_wm_multi_rollout: bool = False
    async_wm_multi_rollout_num_rounds: int = 3
    async_wm_multi_rollout_adaptive_kappa: bool = False
    async_wm_multi_rollout_adaptive_kappa_low_replan: bool = False
    async_wm_routing_policy: Literal["kappa", "always_infer", "rapid"] = "kappa"
    async_wm_multi_rollout_kappa_delta: float = 0.2
    async_wm_rollout_delta_t: float = 2.0
    task_suite_name: str = "libero_spatial"
    num_steps_wait: int = 10
    num_trials_per_task: int = 50
    episode_idx_start: int = 0
    video_out_path: str = "data/libero/videos"
    wm_confidence_jsonl: str | None = None
    seed: int = 7
    task_id_start: int = 0
    task_id_count: int | None = None
    write_videos: bool = True
    task_c_trace_dir: str | None = None
    task_c_run_id: str | None = None
    task_c_condition: str | None = None
    rapid_thresholds: str | None = None


def eval_libero(args: Args) -> None:
    np.random.seed(args.seed)

    task_c_recorder = None
    task_c_run_id = args.task_c_run_id or ""
    task_c_condition = args.task_c_condition or ""
    if args.task_c_trace_dir is not None:
        if not args.task_c_run_id or not args.task_c_condition:
            raise ValueError("task_c_trace_dir requires task_c_run_id and task_c_condition")
        task_c_recorder = task_c_trace.ClientTraceRecorder(
            pathlib.Path(args.task_c_trace_dir),
            run_id=task_c_run_id,
            condition=task_c_condition,
        )
    elif args.task_c_run_id is not None or args.task_c_condition is not None:
        raise ValueError("task_c_run_id/task_c_condition require task_c_trace_dir")

    if args.async_use_world_model and not args.async_inference:
        raise ValueError("async_use_world_model=True requires async_inference=True")
    if args.async_wm_multi_rollout and not args.async_use_world_model:
        raise ValueError("async_wm_multi_rollout=True requires async_use_world_model=True")
    if args.async_wm_multi_rollout_adaptive_kappa and not args.async_wm_multi_rollout:
        raise ValueError("async_wm_multi_rollout_adaptive_kappa=True requires async_wm_multi_rollout=True")
    if args.async_wm_multi_rollout_adaptive_kappa_low_replan and not args.async_wm_multi_rollout_adaptive_kappa:
        raise ValueError(
            "async_wm_multi_rollout_adaptive_kappa_low_replan=True requires async_wm_multi_rollout_adaptive_kappa=True"
        )
    if args.async_wm_routing_policy not in rapid_trigger.ROUTING_POLICIES:
        raise ValueError(f"unknown async_wm_routing_policy: {args.async_wm_routing_policy!r}")
    if args.async_wm_routing_policy != "kappa":
        if not args.async_wm_multi_rollout_adaptive_kappa_low_replan:
            raise ValueError(
                "async_wm_routing_policy other than kappa requires adaptive-kappa low-replan shared paths"
            )
        if args.async_wm_routing_policy == "rapid" and args.rapid_thresholds is None:
            raise ValueError("async_wm_routing_policy=rapid requires rapid_thresholds")
    if args.async_wm_multi_rollout:
        if args.async_prefetch_proprio != "trigger":
            raise ValueError("async_wm_multi_rollout=True requires async_prefetch_proprio=trigger")
        d_idx = round(float(args.async_wm_rollout_delta_t))
        if args.overlap_skip < 1:
            raise ValueError("async_wm_multi_rollout=True requires overlap_skip>=1 (non-empty first WM prefix)")
        if d_idx < 1:
            raise ValueError("async_wm_rollout_delta_t must round to a positive integer index step")
        if not args.async_wm_multi_rollout_adaptive_kappa:
            if args.overlap_skip + (args.async_wm_multi_rollout_num_rounds - 1) * d_idx > args.action_horizon:
                raise ValueError(
                    "async_wm_multi_rollout: need overlap_skip + (num_rounds-1)*round(async_wm_rollout_delta_t) <= "
                    f"action_horizon (got overlap_skip={args.overlap_skip}, num_rounds={args.async_wm_multi_rollout_num_rounds}, "
                    f"async_wm_rollout_delta_t={args.async_wm_rollout_delta_t}, action_horizon={args.action_horizon})."
                )
            if 2 * args.overlap_skip + (args.async_wm_multi_rollout_num_rounds - 1) * d_idx > args.action_horizon:
                raise ValueError(
                    "async_wm_multi_rollout: chunk stitching needs 2*overlap_skip + (num_rounds-1)*round(async_wm_rollout_delta_t) "
                    f"<= action_horizon (got overlap_skip={args.overlap_skip}, num_rounds={args.async_wm_multi_rollout_num_rounds}, "
                    f"d_idx={d_idx}, action_horizon={args.action_horizon})."
                )
        else:
            from openpi.policies.wm_multi_rollout_schedule import wm_multi_rollout_adaptive_max_rounds

            max_r = wm_multi_rollout_adaptive_max_rounds(
                h=args.action_horizon, overlap=args.overlap_skip, delta_idx=d_idx
            )
            if args.overlap_skip + (max_r - 1) * d_idx > args.action_horizon:
                raise ValueError(
                    "async_wm_multi_rollout adaptive: internal schedule error "
                    f"(overlap_skip={args.overlap_skip}, max_r={max_r}, d_idx={d_idx}, H={args.action_horizon})."
                )
    if args.async_prefetch_proprio == "oracle_handover" and not args.async_inference:
        raise ValueError("async_prefetch_proprio=oracle_handover requires async_inference=True")
    if args.async_prefetch_proprio == "oracle_handover" and args.async_use_world_model:
        raise ValueError("async_prefetch_proprio=oracle_handover is incompatible with async_use_world_model=True")
    if args.async_prefetch_proprio in _LIBERO_HANDOVER_MODES:
        if not args.async_inference:
            raise ValueError(f"async_prefetch_proprio={args.async_prefetch_proprio} requires async_inference=True")
        if args.async_use_world_model and args.async_prefetch_proprio != "handover_true":
            raise ValueError(
                f"async_prefetch_proprio={args.async_prefetch_proprio} is incompatible with async_use_world_model=True"
            )
        if args.async_use_proprio_state_at_h_minus_k or args.async_proprio_state_chunk_index is not None:
            raise ValueError(
                "Libero handover mode (image@K + q@handover) cannot be combined with proprio-state-chunk-index mode"
            )
    if args.async_prefetch_proprio in ("handover_vlash_last_action", "handover_future_rollout"):
        _ns = (
            pathlib.Path(args.pi0_norm_checkpoint_dir)
            / "assets"
            / "physical-intelligence"
            / "libero"
            / "norm_stats.json"
        )
        if not _ns.is_file():
            raise ValueError(f"missing norm_stats: {_ns}")
    if (args.async_proprio_state_chunk_index is not None or args.async_use_proprio_state_at_h_minus_k) and (
        not args.async_inference
    ):
        raise ValueError(
            "async_proprio_state_chunk_index / async_use_proprio_state_at_h_minus_k require async_inference=True"
        )
    if args.async_use_proprio_state_at_h_minus_k and args.async_proprio_state_chunk_index is not None:
        raise ValueError("set only one of: --async-use-proprio-state-at-h-minus-k or --async-proprio-state-chunk-index")
    if args.async_inference:
        _validate_async_chunk_params(
            action_horizon=args.action_horizon,
            overlap_skip=args.overlap_skip,
            async_trigger_step=args.async_trigger_step,
            chunk_exec_steps=args.async_chunk_exec_steps,
            allow_trigger_at_chunk_start=args.async_allow_trigger_at_chunk_start,
        )

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    logging.info("Task suite: %s", args.task_suite_name)

    if args.task_id_start < 0 or args.task_id_start >= num_tasks_in_suite:
        raise ValueError(f"task_id_start must be in [0,{num_tasks_in_suite}), got {args.task_id_start}")
    task_id_stop = num_tasks_in_suite
    if args.task_id_count is not None:
        if args.task_id_count < 1:
            raise ValueError("task_id_count must be >= 1")
        task_id_stop = min(num_tasks_in_suite, args.task_id_start + args.task_id_count)
    if args.episode_idx_start < 0:
        raise ValueError("episode_idx_start must be non-negative")
    required_init_states = args.episode_idx_start + args.num_trials_per_task
    initial_states_by_task: dict[int, np.ndarray] = {}
    initial_state_counts: dict[int, int] = {}
    for check_task_id in range(args.task_id_start, task_id_stop):
        states = task_suite.get_task_init_states(check_task_id)
        initial_states_by_task[check_task_id] = states
        initial_state_counts[check_task_id] = len(states)
    too_short = {task_id: count for task_id, count in initial_state_counts.items() if count < required_init_states}
    if too_short:
        raise ValueError(
            f"selected tasks need at least {required_init_states} init states for episode_idx_start="
            f"{args.episode_idx_start}, got {too_short}"
        )
    logging.info("LIBERO init-state preflight counts: %s", initial_state_counts)

    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)

    if args.task_suite_name == "libero_spatial":
        max_steps = 220
    elif args.task_suite_name == "libero_object":
        max_steps = 280
    elif args.task_suite_name == "libero_goal":
        max_steps = 300
    elif args.task_suite_name == "libero_10":
        max_steps = 520
    elif args.task_suite_name == "libero_90":
        max_steps = 400
    else:
        raise ValueError(f"Unknown task suite: {args.task_suite_name}")

    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)

    env_holder: dict = {
        "env": None,
        "task_description": "",
        "task_c_recorder": task_c_recorder,
        "task_c_sim_step": 0,
        "task_c_latest_obs": None,
    }
    proprio_chunk_idx = None
    rapid_observer: rapid_trigger.RapidKinematicTrigger | None = None
    if args.rapid_thresholds is not None:
        thresholds = rapid_trigger.load_threshold_document(pathlib.Path(args.rapid_thresholds))
        rapid_observer = rapid_trigger.RapidKinematicTrigger(thresholds)

    def record_and_observe_proprio(observation: dict) -> None:
        state = observation["observation/state"]
        if task_c_recorder is not None:
            context = observation.get(task_c_trace.TRACE_KEY)
            if context is None:
                raise task_c_trace.TaskCTraceError("Task-C trigger observation is missing its trace context")
            task_c_recorder.record_proprio_observation(context, state=state)
        if rapid_observer is not None:
            rapid_observer.observe(state)

    if args.async_inference:
        if args.async_use_proprio_state_at_h_minus_k:
            proprio_chunk_idx = args.action_horizon - args.async_trigger_step
        elif args.async_proprio_state_chunk_index is not None:
            proprio_chunk_idx = args.async_proprio_state_chunk_index
        prefetch_handover_true_state = args.async_prefetch_proprio in _LIBERO_HANDOVER_MODES
        if args.async_chunk_exec_steps is not None and not prefetch_handover_true_state:
            raise ValueError(
                "async_chunk_exec_steps is only supported with Libero handover prefetch modes "
                f"({sorted(_LIBERO_HANDOVER_MODES)}); non-handover async uses AsyncActionBufferBroker, which "
                "executes one policy action per env step from a rolling buffer (pass async_chunk_exec_steps=None)."
            )
        handover_merged_state_fn: Callable[[dict, dict, np.ndarray], np.ndarray] | None = None
        if args.async_prefetch_proprio == "handover_vlash_last_action":
            handover_merged_state_fn = _make_libero_handover_norm_proxy_fn(
                pathlib.Path(args.pi0_norm_checkpoint_dir), "vlash_last_action"
            )
        elif args.async_prefetch_proprio == "handover_future_rollout":
            handover_merged_state_fn = _make_libero_handover_norm_proxy_fn(
                pathlib.Path(args.pi0_norm_checkpoint_dir), "future_rollout"
            )

        handover_snapshot_hook: Callable[[dict, action_chunk_broker.PrefetchContext], dict] | None = None
        if args.async_use_world_model and args.async_wm_multi_rollout and args.async_wm_multi_rollout_adaptive_kappa:
            prefetch_hook = _libero_wm_multi_rollout_adaptive_prefetch_hook(
                OPENPI_ASYNC_KEY,
                wm_rollout_delta_t=args.async_wm_rollout_delta_t,
                kappa_delta=args.async_wm_multi_rollout_kappa_delta,
                ae_proprio_source=None,
                adaptive_kappa_low_replan=args.async_wm_multi_rollout_adaptive_kappa_low_replan,
                routing_policy=args.async_wm_routing_policy,
                rapid_observer=rapid_observer,
            )
        elif args.async_use_world_model and args.async_wm_multi_rollout:
            prefetch_hook = _libero_wm_multi_rollout_prefetch_hook(
                OPENPI_ASYNC_KEY,
                num_rounds=args.async_wm_multi_rollout_num_rounds,
                wm_rollout_delta_t=args.async_wm_rollout_delta_t,
                ae_proprio_source=None,
            )
        elif args.async_use_world_model and args.async_prefetch_proprio == "handover_true":
            prefetch_hook = None
            handover_snapshot_hook = _libero_wm_prefetch_hook(OPENPI_ASYNC_KEY, ae_proprio_source="prefix_t")
        elif args.async_use_world_model:
            prefetch_hook = _libero_wm_prefetch_hook(OPENPI_ASYNC_KEY)
        elif args.async_prefetch_proprio == "oracle_handover":
            prefetch_hook = _libero_oracle_handover_prefetch_hook(env_holder)
        else:
            prefetch_hook = None
        if prefetch_handover_true_state:
            chunk_policy = action_chunk_broker.AsyncActionChunkBroker(
                policy=client,
                action_horizon=args.action_horizon,
                async_trigger_step=args.async_trigger_step,
                overlap_skip=args.overlap_skip,
                chunk_exec_steps=args.async_chunk_exec_steps,
                allow_trigger_at_chunk_start=args.async_allow_trigger_at_chunk_start,
                prefetch_obs_hook=prefetch_hook,
                observation_state_chunk_index=proprio_chunk_idx,
                prefetch_handover_true_state=prefetch_handover_true_state,
                handover_snapshot_hook=handover_snapshot_hook,
                handover_merged_state_fn=handover_merged_state_fn,
            )
        else:
            _wm_ae_infer_stats = _WmAeInferRoundStatsLogger()
            low_replan_two_phase_fn = None
            if args.async_wm_multi_rollout_adaptive_kappa_low_replan:
                low_replan_two_phase_fn = _make_wm_low_replan_two_phase_fn(
                    client=client,
                    env_holder=env_holder,
                    args=args,
                    openpi_async_key=OPENPI_ASYNC_KEY,
                )
            chunk_policy = action_chunk_broker.AsyncActionBufferBroker(
                policy=client,
                action_horizon=args.action_horizon,
                async_trigger_step=args.async_trigger_step,
                overlap_skip=args.overlap_skip,
                chunk_exec_steps=None,
                allow_trigger_at_chunk_start=args.async_allow_trigger_at_chunk_start,
                prefetch_obs_hook=prefetch_hook,
                observation_state_chunk_index=proprio_chunk_idx,
                wm_infer_complete_hook=_wm_ae_infer_stats.on_infer_complete,
                low_replan_two_phase_fn=low_replan_two_phase_fn,
                observation_step_hook=(
                    record_and_observe_proprio if task_c_recorder is not None or rapid_observer is not None else None
                ),
            )
    else:
        chunk_policy = action_chunk_broker.ActionChunkBroker(policy=client, action_horizon=args.action_horizon)

    try:
        total_episodes, total_successes = 0, 0
        suite_control_steps_all: list[int] = []
        suite_control_steps_success: list[int] = []
        per_task_mean_all: list[float] = []
        per_task_mean_success: list[float] = []
        for task_id in tqdm.tqdm(range(args.task_id_start, task_id_stop)):
            task = task_suite.get_task(task_id)
            initial_states = initial_states_by_task[task_id]
            env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)
            env_holder["env"] = env
            env_holder["task_description"] = str(task_description)

            task_episodes, task_successes = 0, 0
            task_control_steps_all: list[int] = []
            task_control_steps_success: list[int] = []
            episode_stop = args.episode_idx_start + args.num_trials_per_task
            for episode_idx in tqdm.tqdm(range(args.episode_idx_start, episode_stop)):
                logging.info("\nTask: %s", task_description)
                chunk_policy.reset()
                if rapid_observer is not None:
                    rapid_observer.reset()
                env.reset()
                obs = env.set_init_state(initial_states[episode_idx])
                env_holder["task_c_sim_step"] = 0
                env_holder["task_c_latest_obs"] = obs
                t = 0
                replay_images = []
                done = False
                ep_control_steps = 0
                episode_error: str | None = None

                logging.info("Starting episode %d...", task_episodes + 1)
                last_kappa_sig: tuple[float, ...] | None = None
                ep_kappa_events: list[list[float]] = []
                while t < max_steps + args.num_steps_wait:
                    try:
                        if t < args.num_steps_wait:
                            obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                            t += 1
                            continue

                        img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                        wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                        img = image_tools.convert_to_uint8(
                            image_tools.resize_with_pad(img, args.resize_size, args.resize_size)
                        )
                        wrist_img = image_tools.convert_to_uint8(
                            image_tools.resize_with_pad(wrist_img, args.resize_size, args.resize_size)
                        )
                        replay_images.append(img)

                        infer_element = {
                            "observation/image": img,
                            "observation/wrist_image": wrist_img,
                            "observation/state": _pack_libero_proprio_vector(obs),
                            "prompt": str(task_description),
                        }
                        step_trace_context = None
                        if task_c_recorder is not None:
                            step_trace_context = task_c_trace.trace_context(
                                run_id=task_c_run_id,
                                condition=task_c_condition,
                                suite=args.task_suite_name,
                                task_id=task_id,
                                episode_idx=episode_idx,
                                seed=args.seed,
                                env_step=int(env_holder["task_c_sim_step"]),
                            )
                            infer_element[task_c_trace.TRACE_KEY] = step_trace_context
                        env_holder["task_c_latest_obs"] = obs

                        policy_out = chunk_policy.infer(infer_element)
                        if args.wm_confidence_jsonl:
                            kap = policy_out.get("openpi/wm_confidence_kappa")
                            if kap is not None:
                                kap_list = [float(x) for x in np.asarray(kap, dtype=np.float64).reshape(-1)]
                                sig = tuple(kap_list)
                                if sig != last_kappa_sig:
                                    last_kappa_sig = sig
                                    ep_kappa_events.append(kap_list)
                        action = np.asarray(policy_out["actions"])
                        latest_obs = env_holder.get("task_c_latest_obs") or obs
                        state_before = _pack_libero_proprio_vector(latest_obs)
                        ep_control_steps += 1
                        obs, reward, done, info = env.step(action.tolist())
                        env_holder["task_c_latest_obs"] = obs
                        if task_c_recorder is not None:
                            if step_trace_context is None:
                                raise task_c_trace.TaskCTraceError("missing Task-C step trace context")
                            action_trace_context = task_c_trace.trace_context(
                                run_id=task_c_run_id,
                                condition=task_c_condition,
                                suite=args.task_suite_name,
                                task_id=task_id,
                                episode_idx=episode_idx,
                                seed=args.seed,
                                env_step=int(env_holder["task_c_sim_step"]),
                            )
                            task_c_recorder.record_step(
                                action_trace_context,
                                task_description=str(task_description),
                                action=action,
                                state_before=state_before,
                                state_after=_pack_libero_proprio_vector(obs),
                                done_after_step=bool(done),
                                policy_kappa=policy_out.get("openpi/wm_confidence_kappa"),
                            )
                            env_holder["task_c_sim_step"] = int(env_holder["task_c_sim_step"]) + 1
                        if done:
                            task_successes += 1
                            total_successes += 1
                            break
                        t += 1
                    except Exception as e:
                        logging.error("Caught exception: %s", e)
                        episode_error = f"{type(e).__name__}: {e}"
                        if task_c_recorder is not None:
                            raise
                        break

                task_control_steps_all.append(ep_control_steps)
                if done:
                    task_control_steps_success.append(ep_control_steps)

                task_episodes += 1
                total_episodes += 1

                suffix = "success" if done else "failure"
                task_segment = task_description.replace(" ", "_")
                if args.write_videos and replay_images:
                    imageio.mimwrite(
                        pathlib.Path(args.video_out_path) / f"rollout_{task_segment}_{suffix}.mp4",
                        [np.asarray(x) for x in replay_images],
                        fps=10,
                    )

                if task_c_recorder is not None:
                    episode_context = task_c_trace.trace_context(
                        run_id=task_c_run_id,
                        condition=task_c_condition,
                        suite=args.task_suite_name,
                        task_id=task_id,
                        episode_idx=episode_idx,
                        seed=args.seed,
                        env_step=0,
                    )
                    task_c_recorder.record_episode(
                        episode_context,
                        task_description=str(task_description),
                        success=bool(done),
                        control_steps=int(env_holder["task_c_sim_step"]),
                        main_control_steps=ep_control_steps,
                        error=episode_error,
                    )

                logging.info("Success: %s", done)
                logging.info("# episodes completed so far: %d", total_episodes)
                logging.info("# successes: %d (%.1f%%)", total_successes, total_successes / total_episodes * 100)

                if args.wm_confidence_jsonl and ep_kappa_events:
                    p = pathlib.Path(args.wm_confidence_jsonl)
                    p.parent.mkdir(parents=True, exist_ok=True)
                    wm_meta: dict = {
                        "task_id": int(task_id),
                        "task_description": task_description,
                        "episode_idx": int(episode_idx),
                        "success": bool(done),
                        "wm_kappa_rounds_per_infer": ep_kappa_events,
                        "action_horizon": int(args.action_horizon),
                        "async_trigger_step": int(args.async_trigger_step),
                        "overlap_skip": int(args.overlap_skip),
                        "async_wm_rollout_delta_t": float(args.async_wm_rollout_delta_t),
                        "async_wm_multi_rollout": bool(args.async_wm_multi_rollout),
                        "async_wm_multi_rollout_adaptive_kappa": bool(args.async_wm_multi_rollout_adaptive_kappa),
                        "async_wm_multi_rollout_kappa_delta": float(args.async_wm_multi_rollout_kappa_delta),
                        "async_wm_multi_rollout_adaptive_kappa_low_replan": bool(
                            args.async_wm_multi_rollout_adaptive_kappa_low_replan
                        ),
                    }
                    with p.open("a", encoding="utf-8") as fc:
                        fc.write(json.dumps(wm_meta, ensure_ascii=False) + "\n")

            m_all = _mean_int(task_control_steps_all)
            m_succ = _mean_int(task_control_steps_success)
            logging.info(
                "Task step summary: task=%r trials=%d num_steps_wait=%d "
                "mean_control_steps_all_episodes=%.2f mean_control_steps_success_episodes=%s (n_success=%d) "
                "(one control step = one policy env.step after warmup)",
                task_description,
                len(task_control_steps_all),
                args.num_steps_wait,
                m_all,
                (f"{m_succ:.2f}" if task_control_steps_success else "nan"),
                len(task_control_steps_success),
            )
            suite_control_steps_all.extend(task_control_steps_all)
            suite_control_steps_success.extend(task_control_steps_success)
            per_task_mean_all.append(m_all)
            if task_control_steps_success:
                per_task_mean_success.append(m_succ)

            logging.info("Current task success rate: %s", float(task_successes) / float(task_episodes))
            logging.info("Current total success rate: %s", float(total_successes) / float(total_episodes))

        logging.info("Total success rate: %s", float(total_successes) / float(total_episodes))
        logging.info("Total episodes: %d", total_episodes)
        pooled_all = _mean_int(suite_control_steps_all)
        pooled_succ = _mean_int(suite_control_steps_success)
        mean_of_task_means_all = (
            float(sum(per_task_mean_all) / len(per_task_mean_all)) if per_task_mean_all else float("nan")
        )
        mean_of_task_means_succ = (
            float(sum(per_task_mean_success) / len(per_task_mean_success)) if per_task_mean_success else float("nan")
        )
        logging.info(
            "Suite step summary: suite=%s num_tasks=%d trials_per_task=%d num_steps_wait=%d "
            "pooled_mean_control_steps_all_episodes=%.2f (N=%d) "
            "pooled_mean_control_steps_success_episodes=%s (N_succ_ep=%d) "
            "mean_of_per_task_mean_control_steps_all=%.2f mean_of_per_task_mean_control_steps_success=%s "
            "(control step = one policy env.step after warmup; success-mean is over tasks with >=1 success)",
            args.task_suite_name,
            num_tasks_in_suite,
            args.num_trials_per_task,
            args.num_steps_wait,
            pooled_all,
            len(suite_control_steps_all),
            (f"{pooled_succ:.2f}" if suite_control_steps_success else "nan"),
            len(suite_control_steps_success),
            mean_of_task_means_all,
            (f"{mean_of_task_means_succ:.2f}" if per_task_mean_success else "nan"),
        )
    finally:
        if isinstance(
            chunk_policy,
            action_chunk_broker.AsyncActionChunkBroker | action_chunk_broker.AsyncActionBufferBroker,
        ):
            chunk_policy.shutdown()


def _get_libero_env(task, resolution, seed):
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)
    return env, task_description


def _quat2axisangle(quat):
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    eval_libero(tyro.cli(Args))
