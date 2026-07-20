# ruff: noqa: SLF001, RUF002, RUF003
from __future__ import annotations

import logging
import time
from typing import Any, Literal

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
from openpi_client import base_policy as _base_policy
from openpi_client import rapid_trigger as _rapid_trigger
from openpi_client import task_c_trace as _task_c_trace
from typing_extensions import override

from openpi.models import model as _model
from openpi.models.pi0 import Pi0
from openpi.models.pi0_world_model import Pi0FutureWorldModel
from openpi.models.pi0_world_model import global_confidence_from_log_var
from openpi.policies import wm_inference_verify as _wm_verify
import openpi.policies.policy as _policy
from openpi.policies.wm_multi_rollout_schedule import wm_multi_rollout_adaptive_max_rounds
import openpi.shared.nnx_utils as nnx_utils
import openpi.transforms as transforms

logger = logging.getLogger("openpi")


class _LowKappaFullPi0Fallback(Exception):  # noqa: N818 - retained released internal exception name
    """WM multi-rollout saw κ_r < κ_0 - kappa_delta at round ``r>=1`` (after WM, before that round's AE).

    ``wm_ae_rounds_completed`` counts full WM→AE cycles finished (rounds ``0..r-1``).
    ``kappa_per_round_np`` has length ``r+1`` (one κ per WM forward, including the round that triggered fallback).

    Low-replan two-phase (client): execute ``rollout_actions_model`` for ``rollout_len`` steps in sim, run full Pi0
    on the post-rollout image, then stitch ``glue_actions_model`` with ``full_pi0_actions[overlap:]`` (model-space
    rows before ``_output_transform`` on the server; the infer ``except`` path converts them for the payload).
    """

    __slots__ = (
        "glue_actions_model",
        "kappa_per_round_np",
        "rollout_actions_model",
        "rollout_len",
        "wm_ae_rounds_completed",
    )

    def __init__(
        self,
        *,
        wm_ae_rounds_completed: int,
        kappa_per_round_np: np.ndarray,
        rollout_len: int,
        rollout_actions_model: np.ndarray,
        glue_actions_model: np.ndarray,
    ) -> None:
        super().__init__()
        self.wm_ae_rounds_completed = int(wm_ae_rounds_completed)
        self.kappa_per_round_np = kappa_per_round_np
        self.rollout_len = int(rollout_len)
        self.rollout_actions_model = rollout_actions_model
        self.glue_actions_model = glue_actions_model


ASYNC_KEY = "openpi/async"
AeProprioSource = Literal["prefix_t", "future_rollout", "vlash_last_action"]


def _rollforward_proprio_batched(state: jnp.ndarray, actions: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
    m = mask.astype(jnp.float32)[..., None]
    acc = jnp.sum(m * actions, axis=1)
    d = min(int(state.shape[-1]), int(acc.shape[-1]))
    if d <= 0:
        return state
    return state.at[..., :d].add(acc[..., :d])


def _last_valid_prefix_action_batched(actions: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
    m = mask.astype(jnp.float32)
    rev = m[:, ::-1]
    cum = jnp.cumsum(rev, axis=1)
    pick_rev = (cum == 1) & (rev > 0.5)
    pick = pick_rev[:, ::-1]
    has_any = jnp.sum(m, axis=1, keepdims=True) > 0
    weights = jnp.where(has_any, pick.astype(jnp.float32), jnp.zeros_like(pick).at[:, 0].set(1.0))
    return jnp.einsum("bla,bl->ba", actions, weights)


def _vlash_last_action_proprio_batched(state: jnp.ndarray, actions: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
    last_a = _last_valid_prefix_action_batched(actions, mask)
    d = min(int(state.shape[-1]), int(last_a.shape[-1]))
    if d <= 0:
        return state
    return state.at[..., :d].set(last_a[..., :d])


class Pi0AsyncInferencePolicy(_base_policy.BasePolicy):
    def __init__(
        self,
        inner: _policy.Policy,
        *,
        pi0: Pi0,
        world_model: Pi0FutureWorldModel | None,
        action_norm: transforms.Normalize | None,
        model_action_dim: int,
        ae_proprio_source: AeProprioSource = "vlash_last_action",
    ):
        self._inner = inner
        self._pi0 = pi0
        self._world_model = world_model
        self._action_norm = action_norm
        self._model_action_dim = model_action_dim
        self._ae_proprio_source: AeProprioSource = ae_proprio_source
        self._prefix_states = nnx_utils.module_jit(pi0.prefix_hidden_states)
        self._sample_with_future = nnx_utils.module_jit(pi0.sample_actions)
        if world_model is not None:
            wm_gd, wm_st = nnx.split(world_model)

            def _wm_jitted(h_t, proprio, action_prefix_pad, prefix_mask, delta_t, rng_key):
                wm = nnx.merge(wm_gd, wm_st)
                out = wm(
                    h_t,
                    proprio,
                    action_prefix_pad,
                    prefix_mask,
                    delta_t,
                    kv_mask=None,
                    rngs=nnx.Rngs(rng_key),
                    train=False,
                    return_current_tokens=False,
                )
                kappa = global_confidence_from_log_var(out.log_var)
                return out.mu, kappa

            self._wm_forward = jax.jit(_wm_jitted)
        else:
            self._wm_forward = None
        self._task_c_trace = _task_c_trace.ServerTraceRecorder.from_env()

    @property
    def metadata(self) -> dict[str, Any]:
        return self._inner.metadata

    @override
    def infer(self, obs: dict) -> dict:
        d = dict(obs)
        trace_context = d.pop(_task_c_trace.TRACE_KEY, None)
        meta = d.pop(ASYNC_KEY, None)
        if not isinstance(meta, dict) or not meta.get("use_world_model"):
            if self._task_c_trace is not None:
                self._task_c_trace.begin_policy_call(trace_context, kind="plain_vlm")
            return self._inner.infer(d)
        if self._wm_forward is None or self._world_model is None:
            logger.warning(
                "openpi/async.use_world_model is set but no world model checkpoint was loaded; using standard Pi0."
            )
            return self._inner.infer(d)

        inputs = self._inner._input_transform(d)
        t0 = time.monotonic()
        used_wm_multi = isinstance(meta.get("wm_multi_rollout"), dict) and bool(meta["wm_multi_rollout"].get("enabled"))
        policy_call_id = None
        if self._task_c_trace is not None:
            routing_policy = None
            if used_wm_multi:
                routing_policy = meta["wm_multi_rollout"].get("routing_policy", "kappa")
            policy_call_id = self._task_c_trace.begin_policy_call(
                trace_context,
                kind="rapid_schedule" if routing_policy in {"always_infer", "rapid"} else (
                    "kappa_schedule" if used_wm_multi else "faac_refresh"
                ),
            )
        wm_extras: dict[str, Any] = {}
        try:
            if used_wm_multi:
                actions_batched, kappa_rounds, wm_extras = self._infer_wm_multi_rollout_ae(
                    inputs,
                    meta,
                    trace_context=trace_context,
                    policy_call_id=policy_call_id,
                )
            else:
                actions_batched, kappa_rounds = self._infer_with_world_model(
                    inputs,
                    meta,
                    trace_context=trace_context,
                    policy_call_id=policy_call_id,
                )
        except _LowKappaFullPi0Fallback as ex:
            logger.info(
                "wm_multi_rollout adaptive low_replan: κ dropped below κ₀ - kappa_delta "
                "(completed_wm_ae_rounds=%d wm_kappa_rounds=%s); two-phase full Pi0 "
                "(rollout_len=%d then glue+pi0[overlap:]).",
                ex.wm_ae_rounds_completed,
                np.asarray(ex.kappa_per_round_np, dtype=np.float64).reshape(-1).tolist(),
                ex.rollout_len,
            )
            stack = np.concatenate(
                [
                    np.asarray(ex.rollout_actions_model, dtype=np.float32),
                    np.asarray(ex.glue_actions_model, dtype=np.float32),
                ],
                axis=0,
            )
            # ``_output_transform`` includes ``Unnormalize(..., strict=True)``, which requires every norm_stats key
            # (e.g. ``state`` and ``actions``) to be present; mirror the success-path dict.
            batched = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
            state_for_out = np.asarray(batched["state"][0, ...])
            out_tf = self._inner._output_transform({"state": state_for_out, "actions": stack})
            act_all = np.asarray(out_tf["actions"], dtype=np.float32)
            n_roll = int(ex.rollout_len)
            roll_client = np.array(act_all[:n_roll], dtype=np.float32, copy=True)
            glue_client = np.array(act_all[n_roll:], dtype=np.float32, copy=True)
            return {
                "openpi/wm_low_replan_two_phase": True,
                "openpi/wm_low_replan_fallback_full_pi0": True,
                "openpi/wm_low_replan_partial_wm_ae_rounds": int(ex.wm_ae_rounds_completed),
                "openpi/wm_confidence_kappa": np.asarray(ex.kappa_per_round_np, dtype=np.float32).reshape(-1).copy(),
                "openpi/wm_low_replan_rollout_len": n_roll,
                "openpi/wm_low_replan_rollout_actions": roll_client,
                "openpi/wm_low_replan_glue_actions": glue_client,
                "policy_timing": {"infer_ms": (time.monotonic() - t0) * 1000},
            }
        except Exception:
            if self._task_c_trace is not None:
                # A silent standard-Pi0 fallback would corrupt the scheduling
                # comparison and leave missing mu/decision rows.  Trace runs
                # therefore fail closed while legacy runs retain the fallback.
                raise
            logger.exception("World model inference failed; falling back to standard Pi0.")
            out = self._inner.infer(d)
            if isinstance(out, dict):
                out = dict(out)
                out["openpi/wm_exception_fallback_full_pi0"] = True
                return out
            return out

        batched = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
        outputs = {"state": batched["state"], "actions": actions_batched}
        outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)
        outputs = self._inner._output_transform(outputs)
        outputs["policy_timing"] = {"infer_ms": (time.monotonic() - t0) * 1000}
        outputs["openpi/async_world_model"] = True
        if used_wm_multi:
            outputs["openpi/async_wm_multi_rollout"] = True
            if wm_extras.get("wm_stitch_n") is not None:
                outputs["openpi/wm_stitch_n"] = int(wm_extras["wm_stitch_n"])
            if wm_extras.get("adaptive_max_rounds") is not None:
                outputs["openpi/wm_adaptive_max_rounds"] = int(wm_extras["adaptive_max_rounds"])
            if wm_extras.get("adaptive_exec_len") is not None:
                outputs["openpi/wm_adaptive_exec_len"] = int(wm_extras["adaptive_exec_len"])
            if wm_extras.get("adaptive_early_stop"):
                outputs["openpi/wm_adaptive_early_stop"] = True
        kr = np.asarray(jax.device_get(kappa_rounds), dtype=np.float32).reshape(-1)
        outputs["openpi/wm_confidence_kappa"] = kr
        logger.info("wm_confidence_kappa_rounds=%s", kr.tolist())
        if wm_extras.get("adaptive_max_rounds") is not None:
            logger.info(
                "wm_multi_rollout_adaptive: max_rounds=%s early_stop=%s exec_len=%s",
                wm_extras.get("adaptive_max_rounds"),
                wm_extras.get("adaptive_early_stop"),
                wm_extras.get("adaptive_exec_len"),
            )
        return outputs

    def _forward_world_model_task_c(
        self,
        h_t: Any,
        proprio: jax.Array,
        ap_j: jax.Array,
        mask_j: jax.Array,
        delta: jax.Array,
        k_wm: jax.Array,
    ) -> tuple[jax.Array, jax.Array, dict[str, Any] | None]:
        if self._wm_forward is None:
            raise RuntimeError("world-model forward is unavailable")
        if self._task_c_trace is None:
            mu, kappa = self._wm_forward(h_t, proprio, ap_j, mask_j, delta, k_wm)
            return mu, kappa, None

        started_ns = time.perf_counter_ns()
        mu, kappa = self._wm_forward(h_t, proprio, ap_j, mask_j, delta, k_wm)
        jax.block_until_ready((mu, kappa))
        forward_done_ns = time.perf_counter_ns()
        host_check_started_ns = time.perf_counter_ns()
        kappa_float = float(np.asarray(jax.device_get(kappa.reshape(()))))
        host_check_done_ns = time.perf_counter_ns()
        return (
            mu,
            kappa,
            {
                "kappa": kappa_float,
                "wm_forward_kappa_ms": (forward_done_ns - started_ns) / 1_000_000.0,
                "kappa_host_check_ms": (host_check_done_ns - host_check_started_ns) / 1_000_000.0,
                "_kappa_decision_started_ns": host_check_done_ns,
            },
        )

    @staticmethod
    def _complete_task_c_wm_measurement(measurement: dict[str, Any] | None, mu: jax.Array) -> None:
        """Close the tier-0 boundary after the scheduling decision, before trace I/O."""

        if measurement is None:
            return
        started_ns = measurement.pop("_kappa_decision_started_ns", None)
        if started_ns is None or "kappa_decision_ms" in measurement:
            raise _task_c_trace.TaskCTraceError("Task-C WM measurement boundary was closed more than once")
        measurement["kappa_decision_ms"] = (time.perf_counter_ns() - int(started_ns)) / 1_000_000.0
        # Mu persistence is deliberately outside the tier-0 boundary.
        measurement["mu"] = np.asarray(jax.device_get(mu), dtype=np.float16)

    def _record_task_c_wm(
        self,
        measurement: dict[str, Any] | None,
        *,
        trace_context: dict | None,
        policy_call_id: str | None,
        round_index: int,
        decision: str,
        decision_eligible: bool,
        action_expert_executed: bool,
        routing_policy: str | None = None,
        rapid: dict[str, Any] | None = None,
    ) -> None:
        if self._task_c_trace is None:
            return
        if measurement is None:
            raise _task_c_trace.TaskCTraceError("missing Task-C world-model measurement")
        self._task_c_trace.record_wm_call(
            trace_context,
            policy_call_id=policy_call_id,
            round_index=round_index,
            mu=measurement["mu"],
            kappa=measurement["kappa"],
            wm_forward_kappa_ms=measurement["wm_forward_kappa_ms"],
            kappa_host_check_ms=measurement["kappa_host_check_ms"],
            kappa_decision_ms=measurement["kappa_decision_ms"],
            decision=decision,
            decision_eligible=decision_eligible,
            action_expert_executed=action_expert_executed,
            routing_policy=routing_policy,
            rapid=rapid,
        )

    def _infer_with_world_model(
        self,
        inputs: dict,
        meta: dict,
        *,
        trace_context: dict | None = None,
        policy_call_id: str | None = None,
    ) -> tuple[jax.Array, jax.Array]:
        self._inner._rng, k_wm, k_sample = jax.random.split(self._inner._rng, 3)
        batched = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
        observation = _model.Observation.from_dict(batched)
        h_t = self._prefix_states(observation)

        wm_proprio_t = meta.get("wm_proprio_t")
        if wm_proprio_t is None:
            proprio = batched["state"]
        else:
            wm_q = np.asarray(wm_proprio_t, dtype=np.float32).reshape(-1)
            # Normalize raw proprio (e.g. Libero 8-D) with ``state`` stats, then pad to model width for WM.
            if self._action_norm is not None:
                wm_q = self._action_norm({"state": wm_q})["state"]
            wm_q = transforms.pad_to_dim(wm_q, self._model_action_dim, axis=-1)
            proprio = jnp.asarray(wm_q)[jnp.newaxis, ...]

        ap = np.asarray(meta["action_prefix"], dtype=np.float32)
        if ap.ndim != 2:
            raise ValueError("action_prefix must be 2D (L, A)")
        # Same as proprio: normalize with ``actions`` stats at native width (e.g. Libero 7-D), then pad for WM.
        if self._action_norm is not None:
            ap = self._action_norm({"actions": ap})["actions"]
        ap = transforms.pad_to_dim(ap, self._model_action_dim, axis=-1)

        prefix_mask = np.asarray(meta["prefix_mask"], dtype=bool)
        if prefix_mask.shape[0] != ap.shape[0]:
            raise ValueError("prefix_mask length must match action_prefix length")

        delta_t = float(np.asarray(meta["delta_t"]).reshape(()))
        delta = jnp.asarray([delta_t], dtype=jnp.float32)
        ap_j = jnp.asarray(ap)[jnp.newaxis, ...]
        mask_j = jnp.asarray(prefix_mask)[jnp.newaxis, ...]
        mu, kappa, measurement = self._forward_world_model_task_c(h_t, proprio, ap_j, mask_j, delta, k_wm)
        self._complete_task_c_wm_measurement(measurement, mu)

        ae_src = meta.get("ae_proprio_source")
        if ae_src is not None:
            if ae_src not in ("prefix_t", "future_rollout", "vlash_last_action"):
                raise ValueError(
                    "openpi/async.ae_proprio_source must be 'prefix_t', 'future_rollout', or 'vlash_last_action', "
                    f"got {ae_src!r}"
                )
            effective_ae: AeProprioSource = ae_src
        else:
            effective_ae = self._ae_proprio_source

        obs_for_ae = observation
        if effective_ae == "future_rollout":
            state_ae = _rollforward_proprio_batched(batched["state"], ap_j, mask_j)
            obs_for_ae = _model.Observation.from_dict({**batched, "state": state_ae})
        elif effective_ae == "vlash_last_action":
            state_ae = _vlash_last_action_proprio_batched(batched["state"], ap_j, mask_j)
            obs_for_ae = _model.Observation.from_dict({**batched, "state": state_ae})

        if self._world_model is not None and _wm_verify.wm_inference_verify_mode() != "off":
            _wm_verify.run_wm_inference_verification(
                pi0=self._pi0,
                world_model=self._world_model,
                observation=obs_for_ae,
                mu=mu,
            )

        actions = self._sample_with_future(
            k_sample,
            obs_for_ae,
            num_steps=self._inner._sample_kwargs.get("num_steps", 10),
            future_condition_tokens=mu,
        )
        self._record_task_c_wm(
            measurement,
            trace_context=trace_context,
            policy_call_id=policy_call_id,
            round_index=0,
            decision="faac_refresh",
            decision_eligible=False,
            action_expert_executed=True,
        )
        return actions, jnp.reshape(kappa, (-1,))

    def _observation_for_ae_from_prefix(
        self,
        *,
        batched: dict,
        observation: _model.Observation,
        ap_j: jnp.ndarray,
        mask_j: jnp.ndarray,
        effective_ae: AeProprioSource,
    ) -> _model.Observation:
        if effective_ae == "future_rollout":
            state_ae = _rollforward_proprio_batched(batched["state"], ap_j, mask_j)
            return _model.Observation.from_dict({**batched, "state": state_ae})
        if effective_ae == "vlash_last_action":
            state_ae = _vlash_last_action_proprio_batched(batched["state"], ap_j, mask_j)
            return _model.Observation.from_dict({**batched, "state": state_ae})
        return observation

    def _infer_wm_multi_rollout_ae(
        self,
        inputs: dict,
        meta: dict,
        *,
        trace_context: dict | None = None,
        policy_call_id: str | None = None,
    ) -> tuple[jax.Array, jax.Array, dict[str, Any]]:
        mr = meta["wm_multi_rollout"]
        num_rounds = int(mr["num_rounds"])
        delta_t_wm = float(mr["delta_t"])
        overlap = int(mr["overlap"])
        adaptive = bool(mr.get("adaptive_kappa"))
        low_replan = bool(mr.get("adaptive_kappa_low_replan"))
        routing_policy = str(mr.get("routing_policy", "kappa"))
        if routing_policy not in _rapid_trigger.ROUTING_POLICIES:
            raise ValueError(f"unknown wm_multi_rollout.routing_policy: {routing_policy!r}")
        rapid_payload = mr.get("rapid")
        if rapid_payload is not None and not isinstance(rapid_payload, dict):
            raise ValueError("wm_multi_rollout.rapid must be a mapping or null")
        rapid_route = None if rapid_payload is None else rapid_payload.get("decision")
        if routing_policy == "rapid" and rapid_route not in {"skip", "infer"}:
            raise ValueError("RAPID routing requires rapid.decision=skip or infer")
        if routing_policy != "kappa" and not low_replan:
            raise ValueError("non-kappa routing requires the shared low-replan fallback path")
        kappa_th = float(mr.get("kappa_delta", 0.2))
        if num_rounds < 1:
            raise ValueError("wm_multi_rollout.num_rounds must be >= 1")
        if overlap < 1:
            raise ValueError("wm_multi_rollout.overlap must be >= 1 for non-empty first WM prefix")
        delta_idx = round(delta_t_wm)
        if delta_idx < 1:
            raise ValueError("wm_multi_rollout.delta_t must round to a positive integer index step")

        seed = meta.get("wm_seed_chunk")
        if seed is None:
            raise ValueError("wm_multi_rollout mode requires meta['wm_seed_chunk'] (H, A) from prefetch")
        seed_np = np.asarray(seed, dtype=np.float32)
        if seed_np.ndim != 2:
            raise ValueError("wm_seed_chunk must be 2D (H, A)")

        batched = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
        observation = _model.Observation.from_dict(batched)
        h_t = self._prefix_states(observation)

        wm_proprio_t = meta.get("wm_proprio_t")
        if wm_proprio_t is None:
            proprio = batched["state"]
        else:
            wm_q = np.asarray(wm_proprio_t, dtype=np.float32).reshape(-1)
            if self._action_norm is not None:
                wm_q = self._action_norm({"state": wm_q})["state"]
            wm_q = transforms.pad_to_dim(wm_q, self._model_action_dim, axis=-1)
            proprio = jnp.asarray(wm_q)[jnp.newaxis, ...]

        if self._action_norm is not None:
            w_seed = self._action_norm({"actions": seed_np})["actions"]
            w_seed = transforms.pad_to_dim(w_seed, self._model_action_dim, axis=-1)
        else:
            w_seed = transforms.pad_to_dim(seed_np, self._model_action_dim, axis=-1)
        working = jnp.asarray(w_seed, dtype=jnp.float32)
        h = int(working.shape[0])
        if adaptive:
            max_r = wm_multi_rollout_adaptive_max_rounds(h=h, overlap=overlap, delta_idx=delta_idx)
            if overlap + (max_r - 1) * delta_idx > h:
                raise ValueError(
                    "wm_multi_rollout adaptive: internal max_r violates overlap+(N-1)*delta<=H "
                    f"(overlap={overlap}, max_r={max_r}, delta_idx={delta_idx}, H={h})"
                )
        else:
            max_r = num_rounds
            if overlap + (num_rounds - 1) * delta_idx > h:
                raise ValueError(
                    f"wm_multi_rollout: need overlap + (num_rounds-1)*delta_idx <= H, got "
                    f"overlap={overlap}, num_rounds={num_rounds}, delta_idx={delta_idx}, H={h}"
                )
        # AE merge uses ``start = overlap + r*delta``; require ``start < H`` on the last round r = max_r-1``.
        merge_round_cap = max(1, (h - overlap - 1) // delta_idx + 1)
        if max_r > merge_round_cap:
            logger.warning(
                "wm_multi_rollout: clamping max_r %d -> %d so overlap+r*delta < H on last merge (H=%d overlap=%d delta_idx=%d)",
                max_r,
                merge_round_cap,
                h,
                overlap,
                delta_idx,
            )
            max_r = merge_round_cap

        ae_src = meta.get("ae_proprio_source")
        if ae_src is not None:
            if ae_src not in ("prefix_t", "future_rollout", "vlash_last_action"):
                raise ValueError(
                    "openpi/async.ae_proprio_source must be 'prefix_t', 'future_rollout', or 'vlash_last_action', "
                    f"got {ae_src!r}"
                )
            effective_ae: AeProprioSource = ae_src
        else:
            effective_ae = self._ae_proprio_source

        delta = jnp.asarray([delta_t_wm], dtype=jnp.float32)
        logger.info(
            "wm_multi_rollout: adaptive=%s rounds=%d (max_r=%d) overlap=%d delta_t_wm=%s delta_idx=%d H=%d ae=%s",
            adaptive,
            num_rounds,
            max_r,
            overlap,
            delta_t_wm,
            delta_idx,
            h,
            effective_ae,
        )

        kappa_per_round: list[jax.Array] = []
        kappa0_f: float | None = None
        early_exec_len: int | None = None
        for r in range(max_r):
            self._inner._rng, k_wm, k_sample = jax.random.split(self._inner._rng, 3)
            if r == 0:
                if overlap > h:
                    raise RuntimeError(f"wm_multi_rollout: overlap {overlap} > H {h} at r={r}")
                ap_body = working[0:overlap]
            elif r >= 1 and mr.get("prev_chunk_tail") is not None:
                pct_np = np.asarray(mr["prev_chunk_tail"], dtype=np.float32)
                if pct_np.ndim != 2 or int(pct_np.shape[0]) != overlap:
                    raise ValueError(
                        "wm_multi_rollout.prev_chunk_tail must be 2D (overlap, A) with overlap="
                        f"{overlap}, got shape {pct_np.shape}"
                    )
                end_new = overlap + r * delta_idx
                if end_new > h:
                    raise RuntimeError(
                        f"wm_multi_rollout: prefix new-chunk slice end {end_new} > H={h} at r={r} overlap={overlap} delta_idx={delta_idx}"
                    )
                if self._action_norm is not None:
                    pt = self._action_norm({"actions": pct_np})["actions"]
                    pt = transforms.pad_to_dim(pt, self._model_action_dim, axis=-1)
                else:
                    pt = transforms.pad_to_dim(pct_np, self._model_action_dim, axis=-1)
                prev_rows = jnp.asarray(pt, dtype=jnp.float32)
                new_band = working[overlap:end_new]
                ap_body = jnp.concatenate([prev_rows, new_band], axis=0)
            else:
                lo0 = h - overlap
                if lo0 < 0 or lo0 > h:
                    raise RuntimeError(f"wm_multi_rollout: bad tail slice lo0={lo0} at r={r}")
                parts: list[jax.Array] = [working[lo0:h]]
                for j in range(1, r + 1):
                    lo_j = h - overlap - j * delta_idx
                    hi_j = h - overlap - (j - 1) * delta_idx
                    if lo_j < 0:
                        raise RuntimeError(
                            f"wm_multi_rollout: prefix block j={j} lo_j={lo_j}<0 at r={r} (H={h} overlap={overlap} delta_idx={delta_idx})"
                        )
                    if lo_j >= hi_j or hi_j > h:
                        raise RuntimeError(
                            f"wm_multi_rollout: bad prefix block j={j} lo={lo_j} hi={hi_j} at r={r} (H={h})"
                        )
                    parts.append(working[lo_j:hi_j])
                ap_body = jnp.concatenate(parts, axis=0)
            if ap_body.shape[0] == 0:
                raise RuntimeError(f"wm_multi_rollout: empty WM prefix at round r={r}")
            ap_j = ap_body[jnp.newaxis, ...]
            mask_j = jnp.ones((1, ap_body.shape[0]), dtype=jnp.bool_)
            mu, kappa, measurement = self._forward_world_model_task_c(h_t, proprio, ap_j, mask_j, delta, k_wm)
            kappa_per_round.append(kappa.reshape(()))
            k_f = (
                float(measurement["kappa"])
                if measurement is not None
                else float(np.asarray(jax.device_get(kappa.reshape(()))))
            )
            decision = "seed_round" if r == 0 else "skip_vlm"
            if adaptive:
                if r == 0:
                    kappa0_f = k_f
                elif kappa0_f is not None:
                    if low_replan:
                        route = _rapid_trigger.route_decision(
                            routing_policy,  # type: ignore[arg-type]
                            rapid_route=rapid_route,  # type: ignore[arg-type]
                            kappa=k_f,
                            kappa0=kappa0_f,
                            delta=kappa_th,
                        )
                        if route == "infer":
                            self._complete_task_c_wm_measurement(measurement, mu)
                            self._record_task_c_wm(
                                measurement,
                                trace_context=trace_context,
                                policy_call_id=policy_call_id,
                                round_index=r,
                                decision="infer_vlm",
                                decision_eligible=True,
                                action_expert_executed=False,
                                routing_policy=routing_policy,
                                rapid=rapid_payload,
                            )
                            kappa_np = np.asarray(
                                jax.device_get(jnp.stack(kappa_per_round, axis=0)), dtype=np.float32
                            ).reshape(-1)
                            n_roll = int(overlap + int(r) * int(delta_idx))
                            if n_roll + overlap > int(h):
                                raise RuntimeError(
                                    "wm_multi_rollout low_replan: rollout_len+overlap exceeds H "
                                    f"(n_roll={n_roll}, overlap={overlap}, H={h}, r={r}, delta_idx={delta_idx})"
                                )
                            roll_w = working[:n_roll, ...]
                            glue_w = working[n_roll : n_roll + overlap, ...]
                            raise _LowKappaFullPi0Fallback(
                                wm_ae_rounds_completed=int(r),
                                kappa_per_round_np=kappa_np,
                                rollout_len=n_roll,
                                rollout_actions_model=np.asarray(jax.device_get(roll_w), dtype=np.float32),
                                glue_actions_model=np.asarray(jax.device_get(glue_w), dtype=np.float32),
                            )
                    elif k_f > kappa0_f + kappa_th:
                        self._complete_task_c_wm_measurement(measurement, mu)
                        self._record_task_c_wm(
                            measurement,
                            trace_context=trace_context,
                            policy_call_id=policy_call_id,
                            round_index=r,
                            decision="infer_vlm",
                            decision_eligible=True,
                            action_expert_executed=False,
                        )
                        early_exec_len = overlap + delta_idx * (r - 1)
                        break

            self._complete_task_c_wm_measurement(measurement, mu)

            obs_for_ae = self._observation_for_ae_from_prefix(
                batched=batched,
                observation=observation,
                ap_j=ap_j,
                mask_j=mask_j,
                effective_ae=effective_ae,
            )
            if (
                self._world_model is not None
                and _wm_verify.wm_inference_verify_mode() != "off"
                and ((adaptive and r == max_r - 1) or (not adaptive and r == num_rounds - 1))
            ):
                _wm_verify.run_wm_inference_verification(
                    pi0=self._pi0,
                    world_model=self._world_model,
                    observation=obs_for_ae,
                    mu=mu,
                )

            ae_actions = self._sample_with_future(
                k_sample,
                obs_for_ae,
                num_steps=self._inner._sample_kwargs.get("num_steps", 10),
                future_condition_tokens=mu,
            )
            start = overlap + r * delta_idx
            if start >= h:
                raise RuntimeError(f"wm_multi_rollout: merge start {start} >= H {h} at r={r}")
            working = working.at[start:h].set(ae_actions[0, start:h, ...])
            self._record_task_c_wm(
                measurement,
                trace_context=trace_context,
                policy_call_id=policy_call_id,
                round_index=r,
                decision=decision,
                decision_eligible=r > 0,
                action_expert_executed=True,
                routing_policy=routing_policy,
                rapid=rapid_payload,
            )

        extras: dict[str, Any] = {"wm_stitch_n": len(kappa_per_round)}
        if adaptive:
            extras["adaptive_max_rounds"] = max_r
            extras["adaptive_early_stop"] = early_exec_len is not None
            extras["adaptive_exec_len"] = early_exec_len
            if early_exec_len is not None:
                el = int(early_exec_len)
                tail = el + overlap
                if tail < h:
                    working = working.at[tail:h].set(w_seed[tail:h, ...])

        return working[jnp.newaxis, ...], jnp.stack(kappa_per_round, axis=0), extras
