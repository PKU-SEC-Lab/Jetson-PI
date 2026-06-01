# ruff: noqa: RUF002, RUF003

from __future__ import annotations

import dataclasses
import functools
import gc
import json
import logging
import os
import sys
from typing import Any, Literal
import pathlib

import etils.epath as epath
import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
import torch
import tqdm_loggable.auto as tqdm
import tyro
import wandb

import openpi.models.model as _model
from openpi.models import pi0_config as pi0_config_mod
import openpi.models.pi0_world_model as wm_mod
import openpi.shared.array_typing as at
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _config
import openpi.training.utils as training_utils
import openpi.training.weight_loaders as weight_loaders
import openpi.training.world_model_data as wm_data

logger = logging.getLogger("openpi")

_checkpoint_cpu_snapshot_mode_logged = False


def serializable_wm_config(obj: Any) -> Any:
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {k: serializable_wm_config(v) for k, v in dataclasses.asdict(obj).items()}
    if isinstance(obj, dict):
        return {str(k): serializable_wm_config(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [serializable_wm_config(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return float(obj) if isinstance(obj, np.floating) else int(obj)
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


def append_wm_training_log(training_log_file: str | None, record: dict[str, Any]) -> None:
    if not training_log_file:
        return
    p = pathlib.Path(training_log_file).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


StageId = Literal[1, 2, 3]

LactPrefixSource = Literal["prefix_t", "future_prefix", "prefix_rollout"]


def _rollforward_suffix_state(
    state: jnp.ndarray,
    actions: jnp.ndarray,
    mask: jnp.ndarray,
) -> jnp.ndarray:
    m = mask.astype(jnp.float32)[..., None]
    acc = jnp.sum(m * actions, axis=1)
    d = min(int(state.shape[-1]), int(acc.shape[-1]))
    if d <= 0:
        return state
    return state.at[..., :d].add(acc[..., :d])


def observation_for_lact_suffix_q(
    observation_t: _model.Observation,
    observation_f: _model.Observation,
    *,
    lact_prefix_source: LactPrefixSource,
    wm_action_prefix_pad: at.Float[at.Array, "b l a"] | None = None,
    wm_prefix_mask: at.Bool[at.Array, "b l"] | None = None,
) -> _model.Observation:
    if lact_prefix_source == "prefix_t":
        return observation_t
    if lact_prefix_source == "future_prefix":
        return observation_t.replace(state=observation_f.state)
    if lact_prefix_source == "prefix_rollout":
        if wm_action_prefix_pad is None or wm_prefix_mask is None:
            raise ValueError("prefix_rollout requires wm_action_prefix_pad and wm_prefix_mask")
        new_state = _rollforward_suffix_state(observation_t.state, wm_action_prefix_pad, wm_prefix_mask)
        return observation_t.replace(state=new_state)
    raise ValueError(f"unknown lact_prefix_source: {lact_prefix_source!r}")


class Pi0WorldModelTrainBundle(nnx.Module):

    def __init__(self, pi0: pi0_mod.Pi0, wm: wm_mod.Pi0FutureWorldModel):
        self.pi0 = pi0
        self.wm = wm


def load_bundle_with_wm_export(
    bundle: Pi0WorldModelTrainBundle,
    ckpt_root: epath.Path,
    step: int,
) -> Pi0WorldModelTrainBundle:
    export_dir = ckpt_root / f"world_model_step_{step}"
    params_dir = export_dir / "params"
    if not params_dir.is_dir():
        raise FileNotFoundError(
            f"WM export missing: {params_dir} (need world_model_step_{step} from a prior run)"
        )
    loaded_wm = wm_mod.load_pi0_future_world_model(export_dir, config=bundle.wm.cfg)
    return Pi0WorldModelTrainBundle(bundle.pi0, loaded_wm)


def _resolve_pi0_checkpoint_path(path: str) -> str:
    if path.startswith("gs://") or path.startswith("s3://"):
        return path
    p = pathlib.Path(path).expanduser().resolve()
    if not p.is_dir():
        return str(p)
    if (p / "_METADATA").exists():
        return str(p)
    params_sub = p / "params"
    if params_sub.is_dir() and (params_sub / "_METADATA").exists():
        logger.info("Resolved --pi0-checkpoint: %s -> %s", p, params_sub)
        return str(params_sub)
    return str(p)


def _load_pi0_weights(pi0: pi0_mod.Pi0, params_path: str) -> pi0_mod.Pi0:
    loader = weight_loaders.CheckpointWeightLoader(params_path)
    graphdef, st = nnx.split(pi0)
    loaded = loader.load(st.to_pure_dict())
    st.replace_by_pure_dict(loaded)
    return nnx.merge(graphdef, st)


def trainable_filter_stage1(*, freeze_token_reducer: bool) -> nnx.filterlib.Filter:
    # PathRegex uses fullmatch on "/"-joined path parts, e.g. "wm/..." (no leading slash).
    wm_params = nnx.All(nnx.Param, nnx_utils.PathRegex(r"wm/.*"))
    no_vlm_to_token = nnx.Not(nnx_utils.PathRegex(r"wm/reducer_vlm_to_token/.*"))
    if freeze_token_reducer:
        return nnx.All(
            wm_params,
            no_vlm_to_token,
            nnx.Not(nnx_utils.PathRegex(r"wm/token_reducer/.*")),
        )
    return nnx.All(wm_params, no_vlm_to_token)


def trainable_filter_stage2() -> nnx.filterlib.Filter:
    return trainable_filter_stage2_with_llm(full_llm_trainable=False)


def trainable_filter_stage2_with_llm(*, full_llm_trainable: bool) -> nnx.filterlib.Filter:
    llm_pat = (
        r"pi0/PaliGemma/llm/.*"
        if full_llm_trainable
        else r"pi0/PaliGemma/llm/.*_1/.*"
    )
    return nnx.All(
        nnx.Param,
        nnx.Any(
            nnx_utils.PathRegex(r"wm/.*"),
            nnx_utils.PathRegex(r"pi0/state_proj/.*"),
            nnx_utils.PathRegex(r"pi0/action_in_proj/.*"),
            nnx_utils.PathRegex(r"pi0/action_time_mlp_in/.*"),
            nnx_utils.PathRegex(r"pi0/action_time_mlp_out/.*"),
            nnx_utils.PathRegex(r"pi0/action_out_proj/.*"),
            nnx_utils.PathRegex(llm_pat),
        ),
    )


def trainable_filter_stage3() -> nnx.filterlib.Filter:
    return trainable_filter_stage2_with_llm(full_llm_trainable=False)


def trainable_filter_wm_logvar_head_only() -> nnx.filterlib.Filter:
    return nnx.All(nnx.Param, nnx_utils.PathRegex(r"wm/.*/logvar_head/.*"))


@at.typecheck
def train_step_stage1(
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    observation_t: _model.Observation,
    observation_f: _model.Observation,
    wm_action_prefix_pad: at.Float[at.Array, "b l a"],
    wm_prefix_mask: at.Bool[at.Array, "b l"],
    wm_delta_t: at.Float[at.Array, " b"],
    *,
    trainable_filter: nnx.filterlib.Filter,
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    bundle = nnx.merge(state.model_def, state.params)

    def loss_fn(bundle: Pi0WorldModelTrainBundle, rng2: at.KeyArrayLike):
        rng_wm, rng_pi = jax.random.split(rng2)
        h_t = jax.lax.stop_gradient(bundle.pi0.prefix_hidden_states(observation_t))
        h_f = jax.lax.stop_gradient(bundle.pi0.prefix_hidden_states(observation_f))
        target = jax.lax.stop_gradient(bundle.wm.reduce_tokens(h_f))
        out = bundle.wm(
            h_t,
            observation_t.state,
            wm_action_prefix_pad,
            wm_prefix_mask,
            wm_delta_t,
            kv_mask=None,
            rngs=nnx.Rngs(rng_wm),
            train=True,
            return_current_tokens=False,
        )
        return wm_mod.heteroscedastic_gaussian_nll(target, out.mu, out.log_var)

    diff = nnx.DiffState(0, trainable_filter)
    train_rng = jax.random.fold_in(rng, state.step)
    loss, grads = nnx.value_and_grad(loss_fn, argnums=diff)(bundle, train_rng)

    params = state.params.filter(trainable_filter)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    new_params = optax.apply_updates(params, updates)
    nnx.update(bundle, new_params)
    new_full = nnx.state(bundle)

    new_state = dataclasses.replace(
        state, step=state.step + 1, params=new_full, opt_state=new_opt_state
    )
    return new_state, {"loss": loss, "l_cond": loss}


@at.typecheck
def train_step_stage1_no_logvar(
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    observation_t: _model.Observation,
    observation_f: _model.Observation,
    wm_action_prefix_pad: at.Float[at.Array, "b l a"],
    wm_prefix_mask: at.Bool[at.Array, "b l"],
    wm_delta_t: at.Float[at.Array, " b"],
    *,
    trainable_filter: nnx.filterlib.Filter,
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    """Like stage1, but do NOT train variance head (log_var path).

    This preserves the original stage1 implementation by adding a separate step
    that stops gradients through `log_var` while still optimizing μ to fit t+Δt.
    """
    bundle = nnx.merge(state.model_def, state.params)

    def loss_fn(bundle: Pi0WorldModelTrainBundle, rng2: at.KeyArrayLike):
        rng_wm, rng_pi = jax.random.split(rng2)
        h_t = jax.lax.stop_gradient(bundle.pi0.prefix_hidden_states(observation_t))
        h_f = jax.lax.stop_gradient(bundle.pi0.prefix_hidden_states(observation_f))
        target = jax.lax.stop_gradient(bundle.wm.reduce_tokens(h_f))
        out = bundle.wm(
            h_t,
            observation_t.state,
            wm_action_prefix_pad,
            wm_prefix_mask,
            wm_delta_t,
            kv_mask=None,
            rngs=nnx.Rngs(rng_wm),
            train=True,
            return_current_tokens=False,
        )
        log_var_sg = jax.lax.stop_gradient(out.log_var)
        return wm_mod.heteroscedastic_gaussian_nll(target, out.mu, log_var_sg)

    diff = nnx.DiffState(0, trainable_filter)
    train_rng = jax.random.fold_in(rng, state.step)
    loss, grads = nnx.value_and_grad(loss_fn, argnums=diff)(bundle, train_rng)

    params = state.params.filter(trainable_filter)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    new_params = optax.apply_updates(params, updates)
    nnx.update(bundle, new_params)
    new_full = nnx.state(bundle)

    new_state = dataclasses.replace(
        state, step=state.step + 1, params=new_full, opt_state=new_opt_state
    )
    return new_state, {"loss": loss, "l_cond": loss}


@at.typecheck
def train_step_stage2_wm_logvar_only(
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    observation_t: _model.Observation,
    observation_f: _model.Observation,
    wm_action_prefix_pad: at.Float[at.Array, "b l a"],
    wm_prefix_mask: at.Bool[at.Array, "b l"],
    wm_delta_t: at.Float[at.Array, " b"],
    wm_actions_handover: at.Float[at.Array, "b ah a"],
    wm_handover_valid: at.Bool[at.Array, "b ah"],
    *,
    trainable_filter: nnx.filterlib.Filter,
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    del wm_actions_handover, wm_handover_valid
    bundle = nnx.merge(state.model_def, state.params)

    def loss_fn(bundle: Pi0WorldModelTrainBundle, rng2: at.KeyArrayLike):
        rng_wm, _ = jax.random.split(rng2)
        h_t = jax.lax.stop_gradient(bundle.pi0.prefix_hidden_states(observation_t))
        h_f = jax.lax.stop_gradient(bundle.pi0.prefix_hidden_states(observation_f))
        target = jax.lax.stop_gradient(bundle.wm.reduce_tokens(h_f))
        out = bundle.wm(
            h_t,
            observation_t.state,
            wm_action_prefix_pad,
            wm_prefix_mask,
            wm_delta_t,
            kv_mask=None,
            rngs=nnx.Rngs(rng_wm),
            train=True,
            return_current_tokens=False,
            detach_wm_features_for_logvar=True,
        )
        l_cal = wm_mod.logvar_calibration_loss(target, out.mu, out.log_var)
        return l_cal, {"l_logvar_calib": l_cal}

    diff = nnx.DiffState(0, trainable_filter)
    train_rng = jax.random.fold_in(rng, state.step)
    (loss, aux), grads = nnx.value_and_grad(loss_fn, argnums=diff, has_aux=True)(
        bundle, train_rng
    )

    params = state.params.filter(trainable_filter)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    new_params = optax.apply_updates(params, updates)
    nnx.update(bundle, new_params)
    new_full = nnx.state(bundle)

    new_state = dataclasses.replace(
        state, step=state.step + 1, params=new_full, opt_state=new_opt_state
    )
    return new_state, {"loss": loss, **aux}


@at.typecheck
def train_step_stage2(
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    observation_t: _model.Observation,
    observation_f: _model.Observation,
    wm_action_prefix_pad: at.Float[at.Array, "b l a"],
    wm_prefix_mask: at.Bool[at.Array, "b l"],
    wm_delta_t: at.Float[at.Array, " b"],
    wm_actions_handover: at.Float[at.Array, "b ah a"],
    wm_handover_valid: at.Bool[at.Array, "b ah"],
    *,
    trainable_filter: nnx.filterlib.Filter,
    lambda_act: float,
    lambda_cond: float = 1.0,
    lact_prefix_source: LactPrefixSource = "future_prefix",
    detach_wm_mu_for_lact: bool = False,
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    bundle = nnx.merge(state.model_def, state.params)

    def loss_fn(bundle: Pi0WorldModelTrainBundle, rng2: at.KeyArrayLike):
        rng_wm, _, rng_act = jax.random.split(rng2, 3)
        h_t = jax.lax.stop_gradient(bundle.pi0.prefix_hidden_states(observation_t))
        h_f = jax.lax.stop_gradient(bundle.pi0.prefix_hidden_states(observation_f))
        target = jax.lax.stop_gradient(bundle.wm.reduce_tokens(h_f))
        out = bundle.wm(
            h_t,
            observation_t.state,
            wm_action_prefix_pad,
            wm_prefix_mask,
            wm_delta_t,
            kv_mask=None,
            rngs=nnx.Rngs(rng_wm),
            train=True,
            return_current_tokens=False,
        )
        l_cond = wm_mod.heteroscedastic_gaussian_nll(target, out.mu, out.log_var)
        obs_lact = observation_for_lact_suffix_q(
            observation_t,
            observation_f,
            lact_prefix_source=lact_prefix_source,
            wm_action_prefix_pad=wm_action_prefix_pad,
            wm_prefix_mask=wm_prefix_mask,
        )
        mu_for_lact = jax.lax.stop_gradient(out.mu) if detach_wm_mu_for_lact else out.mu
        l_act = bundle.pi0.compute_flow_matching_loss_with_future(
            rng_act,
            obs_lact,
            wm_actions_handover,
            train=True,
            future_condition_tokens=mu_for_lact,
            action_valid_mask=wm_handover_valid,
        )
        la = jnp.asarray(lambda_act, dtype=jnp.float32)
        lc = jnp.asarray(lambda_cond, dtype=jnp.float32)
        total = lc * l_cond + la * l_act
        aux = {
            "l_cond": l_cond,
            "lambda_cond": lc,
            "lambda_cond_times_l_cond": lc * l_cond,
            "l_act": l_act,
            "lambda_act": la,
            "lambda_act_times_l_act": la * l_act,
            "detach_wm_mu_for_lact": jnp.asarray(1.0 if detach_wm_mu_for_lact else 0.0, dtype=jnp.float32),
        }
        return total, aux

    diff = nnx.DiffState(0, trainable_filter)
    train_rng = jax.random.fold_in(rng, state.step)
    (loss, aux), grads = nnx.value_and_grad(loss_fn, argnums=diff, has_aux=True)(
        bundle, train_rng
    )

    params = state.params.filter(trainable_filter)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    new_params = optax.apply_updates(params, updates)
    nnx.update(bundle, new_params)
    new_full = nnx.state(bundle)

    new_state = dataclasses.replace(
        state, step=state.step + 1, params=new_full, opt_state=new_opt_state
    )
    return new_state, {"loss": loss, **aux}


@at.typecheck
def compute_grads_stage2(
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    observation_t: _model.Observation,
    observation_f: _model.Observation,
    wm_action_prefix_pad: at.Float[at.Array, "b l a"],
    wm_prefix_mask: at.Bool[at.Array, "b l"],
    wm_delta_t: at.Float[at.Array, " b"],
    wm_actions_handover: at.Float[at.Array, "b ah a"],
    wm_handover_valid: at.Bool[at.Array, "b ah"],
    *,
    trainable_filter: nnx.filterlib.Filter,
    lambda_act: float,
    lambda_cond: float = 1.0,
    lact_prefix_source: LactPrefixSource = "future_prefix",
    detach_wm_mu_for_lact: bool = False,
) -> tuple[at.PyTree, dict[str, at.Array]]:
    """Compute grads for stage2/l_cond+l_act, but do NOT update optimizer.

    Useful for gradient accumulation where we want to apply `tx.update` only once.
    """
    bundle = nnx.merge(state.model_def, state.params)

    def loss_fn(bundle: Pi0WorldModelTrainBundle, rng2: at.KeyArrayLike):
        rng_wm, _, rng_act = jax.random.split(rng2, 3)
        h_t = jax.lax.stop_gradient(bundle.pi0.prefix_hidden_states(observation_t))
        h_f = jax.lax.stop_gradient(bundle.pi0.prefix_hidden_states(observation_f))
        target = jax.lax.stop_gradient(bundle.wm.reduce_tokens(h_f))
        out = bundle.wm(
            h_t,
            observation_t.state,
            wm_action_prefix_pad,
            wm_prefix_mask,
            wm_delta_t,
            kv_mask=None,
            rngs=nnx.Rngs(rng_wm),
            train=True,
            return_current_tokens=False,
        )
        l_cond = wm_mod.heteroscedastic_gaussian_nll(target, out.mu, out.log_var)
        obs_lact = observation_for_lact_suffix_q(
            observation_t,
            observation_f,
            lact_prefix_source=lact_prefix_source,
            wm_action_prefix_pad=wm_action_prefix_pad,
            wm_prefix_mask=wm_prefix_mask,
        )
        mu_for_lact = jax.lax.stop_gradient(out.mu) if detach_wm_mu_for_lact else out.mu
        l_act = bundle.pi0.compute_flow_matching_loss_with_future(
            rng_act,
            obs_lact,
            wm_actions_handover,
            train=True,
            future_condition_tokens=mu_for_lact,
            action_valid_mask=wm_handover_valid,
        )
        la = jnp.asarray(lambda_act, dtype=jnp.float32)
        lc = jnp.asarray(lambda_cond, dtype=jnp.float32)
        total = lc * l_cond + la * l_act
        aux = {
            "l_cond": l_cond,
            "lambda_cond": lc,
            "lambda_cond_times_l_cond": lc * l_cond,
            "l_act": l_act,
            "lambda_act": la,
            "lambda_act_times_l_act": la * l_act,
            "detach_wm_mu_for_lact": jnp.asarray(1.0 if detach_wm_mu_for_lact else 0.0, dtype=jnp.float32),
        }
        return total, aux

    diff = nnx.DiffState(0, trainable_filter)
    train_rng = jax.random.fold_in(rng, state.step)
    (loss, aux), grads = nnx.value_and_grad(loss_fn, argnums=diff, has_aux=True)(
        bundle, train_rng
    )
    info = {"loss": loss, **aux}
    return grads, info


def apply_grads_stage2(
    state: training_utils.TrainState,
    grads: at.PyTree,
    *,
    trainable_filter: nnx.filterlib.Filter,
) -> training_utils.TrainState:
    """Apply grads for stage2 and increment `state.step` by 1."""
    bundle = nnx.merge(state.model_def, state.params)
    params = state.params.filter(trainable_filter)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    new_params = optax.apply_updates(params, updates)
    nnx.update(bundle, new_params)
    new_full = nnx.state(bundle)
    return dataclasses.replace(
        state, step=state.step + 1, params=new_full, opt_state=new_opt_state
    )


def train_step_stage2_grad_accum(
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    observation_t: _model.Observation,
    observation_f: _model.Observation,
    wm_action_prefix_pad: at.Float[at.Array, " g b l a"],
    wm_prefix_mask: at.Bool[at.Array, " g b l"],
    wm_delta_t: at.Float[at.Array, " g b"],
    wm_actions_handover: at.Float[at.Array, " g ah b a"],
    wm_handover_valid: at.Bool[at.Array, " g b ah"],
    *,
    trainable_filter: nnx.filterlib.Filter,
    lambda_act: float,
    lambda_cond: float = 1.0,
    lact_prefix_source: LactPrefixSource = "future_prefix",
    detach_wm_mu_for_lact: bool = False,
    grad_accum_steps: int,
    axis_name: str | None = None,
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    bundle = nnx.merge(state.model_def, state.params)

    diff = nnx.DiffState(0, trainable_filter)
    rngs = jax.random.split(rng, grad_accum_steps)

    la = jnp.asarray(lambda_act, dtype=jnp.float32)
    lc = jnp.asarray(lambda_cond, dtype=jnp.float32)

    def _micro_grads_and_info(i: int):
        # Slice micro-batch i (leading dim = grad_accum_steps).
        obs_t_i = jax.tree.map(lambda x: x[i], observation_t)
        obs_f_i = jax.tree.map(lambda x: x[i], observation_f)

        wap_i = wm_action_prefix_pad[i]
        wpm_i = wm_prefix_mask[i]
        dt_i = wm_delta_t[i]
        wah_i = wm_actions_handover[i]
        hv_i = wm_handover_valid[i]

        def loss_fn(bundle: Pi0WorldModelTrainBundle, rng2: at.KeyArrayLike):
            rng_wm, _, rng_act = jax.random.split(rng2, 3)
            h_t = jax.lax.stop_gradient(bundle.pi0.prefix_hidden_states(obs_t_i))
            h_f = jax.lax.stop_gradient(bundle.pi0.prefix_hidden_states(obs_f_i))
            target = jax.lax.stop_gradient(bundle.wm.reduce_tokens(h_f))
            out = bundle.wm(
                h_t,
                obs_t_i.state,
                wap_i,
                wpm_i,
                dt_i,
                kv_mask=None,
                rngs=nnx.Rngs(rng_wm),
                train=True,
                return_current_tokens=False,
            )
            l_cond = wm_mod.heteroscedastic_gaussian_nll(target, out.mu, out.log_var)

            obs_lact = observation_for_lact_suffix_q(
                obs_t_i,
                obs_f_i,
                lact_prefix_source=lact_prefix_source,
                wm_action_prefix_pad=wap_i,
                wm_prefix_mask=wpm_i,
            )
            mu_for_lact = jax.lax.stop_gradient(out.mu) if detach_wm_mu_for_lact else out.mu
            l_act = bundle.pi0.compute_flow_matching_loss_with_future(
                rng_act,
                obs_lact,
                wah_i,
                train=True,
                future_condition_tokens=mu_for_lact,
                action_valid_mask=hv_i,
            )

            total = lc * l_cond + la * l_act
            aux = {
                "l_cond": l_cond,
                "lambda_cond": lc,
                "lambda_cond_times_l_cond": lc * l_cond,
                "l_act": l_act,
                "lambda_act": la,
                "lambda_act_times_l_act": la * l_act,
                "detach_wm_mu_for_lact": jnp.asarray(
                    1.0 if detach_wm_mu_for_lact else 0.0, dtype=jnp.float32
                ),
            }
            return total, aux

        train_rng = jax.random.fold_in(rngs[i], state.step)
        (loss, aux), grads = nnx.value_and_grad(loss_fn, argnums=diff, has_aux=True)(
            bundle, train_rng
        )
        info = {"loss": loss, **aux}
        return grads, info

    # IMPORTANT: do NOT use Python loops under jit (will unroll and explode memory).
    grads0, info0 = _micro_grads_and_info(0)

    def body(i, carry):
        grads_sum, info_sum = carry
        g, info = _micro_grads_and_info(i)
        grads_sum = jax.tree.map(lambda a, b: a + b, grads_sum, g)
        info_sum = {k: info_sum[k] + v for k, v in info.items()}
        return grads_sum, info_sum

    grads_sum, info_sum = jax.lax.fori_loop(
        1, grad_accum_steps, body, (grads0, info0)
    )

    grads_mean = jax.tree.map(lambda g: g / grad_accum_steps, grads_sum)

    params = state.params.filter(trainable_filter)
    if axis_name is not None:
        grads_mean = jax.lax.pmean(grads_mean, axis_name)
    updates, new_opt_state = state.tx.update(grads_mean, state.opt_state, params)
    new_params = optax.apply_updates(params, updates)
    nnx.update(bundle, new_params)
    new_full = nnx.state(bundle)

    new_state = dataclasses.replace(
        state, step=state.step + 1, params=new_full, opt_state=new_opt_state
    )
    info = {k: v / grad_accum_steps for k, v in info_sum.items()}
    if axis_name is not None:
        info = jax.tree.map(lambda x: jax.lax.pmean(x, axis_name), info)
    return new_state, info


def train_step_stage2_grad_accum_dp(
    rng: at.KeyArrayLike,
    step: at.Int[at.ArrayLike, ""],
    params: nnx.State,
    opt_state: optax.OptState,
    observation_t: _model.Observation,
    observation_f: _model.Observation,
    wm_action_prefix_pad: at.Float[at.Array, " g b l a"],
    wm_prefix_mask: at.Bool[at.Array, " g b l"],
    wm_delta_t: at.Float[at.Array, " g b"],
    wm_actions_handover: at.Float[at.Array, " g ah b a"],
    wm_handover_valid: at.Bool[at.Array, " g b ah"],
    *,
    model_def: nnx.GraphDef[Pi0WorldModelTrainBundle],
    tx: optax.GradientTransformation,
    trainable_filter: nnx.filterlib.Filter,
    lambda_act: float,
    lambda_cond: float = 1.0,
    lact_prefix_source: LactPrefixSource = "future_prefix",
    detach_wm_mu_for_lact: bool = False,
    grad_accum_steps: int,
    axis_name: str,
) -> tuple[at.Int[at.ArrayLike, ""], nnx.State, optax.OptState, dict[str, at.Array]]:
    """Data-parallel + grad-accum stage2 step for `jax.pmap`.

    This avoids mapping over `TrainState` directly (it contains non-mapped static leaves like `model_def`).
    """
    bundle = nnx.merge(model_def, params)

    diff = nnx.DiffState(0, trainable_filter)
    rngs = jax.random.split(rng, grad_accum_steps)

    la = jnp.asarray(lambda_act, dtype=jnp.float32)
    lc = jnp.asarray(lambda_cond, dtype=jnp.float32)

    def _micro_grads_and_info(i: int):
        obs_t_i = jax.tree.map(lambda x: x[i], observation_t)
        obs_f_i = jax.tree.map(lambda x: x[i], observation_f)

        wap_i = wm_action_prefix_pad[i]
        wpm_i = wm_prefix_mask[i]
        dt_i = wm_delta_t[i]
        wah_i = wm_actions_handover[i]
        hv_i = wm_handover_valid[i]

        def loss_fn(bundle: Pi0WorldModelTrainBundle, rng2: at.KeyArrayLike):
            rng_wm, _, rng_act = jax.random.split(rng2, 3)
            h_t = jax.lax.stop_gradient(bundle.pi0.prefix_hidden_states(obs_t_i))
            h_f = jax.lax.stop_gradient(bundle.pi0.prefix_hidden_states(obs_f_i))
            target = jax.lax.stop_gradient(bundle.wm.reduce_tokens(h_f))
            out = bundle.wm(
                h_t,
                obs_t_i.state,
                wap_i,
                wpm_i,
                dt_i,
                kv_mask=None,
                rngs=nnx.Rngs(rng_wm),
                train=True,
                return_current_tokens=False,
            )
            l_cond = wm_mod.heteroscedastic_gaussian_nll(target, out.mu, out.log_var)

            obs_lact = observation_for_lact_suffix_q(
                obs_t_i,
                obs_f_i,
                lact_prefix_source=lact_prefix_source,
                wm_action_prefix_pad=wap_i,
                wm_prefix_mask=wpm_i,
            )
            mu_for_lact = jax.lax.stop_gradient(out.mu) if detach_wm_mu_for_lact else out.mu
            l_act = bundle.pi0.compute_flow_matching_loss_with_future(
                rng_act,
                obs_lact,
                wah_i,
                train=True,
                future_condition_tokens=mu_for_lact,
                action_valid_mask=hv_i,
            )

            total = lc * l_cond + la * l_act
            aux = {
                "l_cond": l_cond,
                "lambda_cond": lc,
                "lambda_cond_times_l_cond": lc * l_cond,
                "l_act": l_act,
                "lambda_act": la,
                "lambda_act_times_l_act": la * l_act,
                "detach_wm_mu_for_lact": jnp.asarray(
                    1.0 if detach_wm_mu_for_lact else 0.0, dtype=jnp.float32
                ),
            }
            return total, aux

        train_rng = jax.random.fold_in(rngs[i], step)
        (loss, aux), grads = nnx.value_and_grad(loss_fn, argnums=diff, has_aux=True)(
            bundle, train_rng
        )
        info = {"loss": loss, **aux}
        return grads, info

    grads0, info0 = _micro_grads_and_info(0)

    def body(i, carry):
        grads_sum, info_sum = carry
        g, info = _micro_grads_and_info(i)
        grads_sum = jax.tree.map(lambda a, b: a + b, grads_sum, g)
        info_sum = {k: info_sum[k] + v for k, v in info.items()}
        return grads_sum, info_sum

    grads_sum, info_sum = jax.lax.fori_loop(1, grad_accum_steps, body, (grads0, info0))
    grads_mean = jax.tree.map(lambda g: g / grad_accum_steps, grads_sum)
    grads_mean = jax.lax.pmean(grads_mean, axis_name)

    params_tr = params.filter(trainable_filter)
    updates, new_opt_state = tx.update(grads_mean, opt_state, params_tr)
    new_params_tr = optax.apply_updates(params_tr, updates)
    nnx.update(bundle, new_params_tr)
    new_full = nnx.state(bundle)

    info = {k: v / grad_accum_steps for k, v in info_sum.items()}
    info = jax.tree.map(lambda x: jax.lax.pmean(x, axis_name), info)
    return step + 1, new_full, new_opt_state, info


@at.typecheck
def train_step_stage3(
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    observation_t: _model.Observation,
    observation_f: _model.Observation,
    observation_td1: _model.Observation,
    wm_action_prefix_pad: at.Float[at.Array, "b l a"],
    wm_prefix_mask: at.Bool[at.Array, "b l"],
    wm_delta_t: at.Float[at.Array, " b"],
    wm_actions_handover: at.Float[at.Array, "b ah a"],
    wm_handover_valid: at.Bool[at.Array, "b ah"],
    wm_semigroup_valid: at.Bool[at.Array, " b"],
    wm_delta2: at.Float[at.Array, " b"],
    wm_sg_prefix_pad: at.Float[at.Array, "b l a"],
    wm_sg_prefix_mask: at.Bool[at.Array, "b l"],
    *,
    trainable_filter: nnx.filterlib.Filter,
    lambda_act: float,
    lambda_sg: float,
    lact_prefix_source: LactPrefixSource = "future_prefix",
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    bundle = nnx.merge(state.model_def, state.params)

    def loss_fn(bundle: Pi0WorldModelTrainBundle, rng2: at.KeyArrayLike):
        rng_wm, _, rng_act, rng_sg = jax.random.split(rng2, 4)
        h_t = jax.lax.stop_gradient(bundle.pi0.prefix_hidden_states(observation_t))
        h_f = jax.lax.stop_gradient(bundle.pi0.prefix_hidden_states(observation_f))
        target = jax.lax.stop_gradient(bundle.wm.reduce_tokens(h_f))
        out = bundle.wm(
            h_t,
            observation_t.state,
            wm_action_prefix_pad,
            wm_prefix_mask,
            wm_delta_t,
            kv_mask=None,
            rngs=nnx.Rngs(rng_wm),
            train=True,
            return_current_tokens=False,
        )
        l_cond = wm_mod.heteroscedastic_gaussian_nll(target, out.mu, out.log_var)
        obs_lact = observation_for_lact_suffix_q(
            observation_t,
            observation_f,
            lact_prefix_source=lact_prefix_source,
            wm_action_prefix_pad=wm_action_prefix_pad,
            wm_prefix_mask=wm_prefix_mask,
        )
        l_act = bundle.pi0.compute_flow_matching_loss_with_future(
            rng_act,
            obs_lact,
            wm_actions_handover,
            train=True,
            future_condition_tokens=out.mu,
            action_valid_mask=wm_handover_valid,
        )
        h_d1 = jax.lax.stop_gradient(bundle.pi0.prefix_hidden_states(observation_td1))
        out2 = bundle.wm(
            h_d1,
            observation_td1.state,
            wm_sg_prefix_pad,
            wm_sg_prefix_mask,
            wm_delta2,
            kv_mask=None,
            rngs=nnx.Rngs(rng_sg),
            train=True,
            return_current_tokens=False,
        )
        sg_err = jnp.mean(jnp.square(out.mu - out2.mu), axis=(1, 2))
        l_sg = jnp.sum(sg_err * wm_semigroup_valid.astype(jnp.float32)) / jnp.maximum(
            jnp.sum(wm_semigroup_valid.astype(jnp.float32)), 1.0
        )
        la = jnp.asarray(lambda_act, dtype=jnp.float32)
        ls = jnp.asarray(lambda_sg, dtype=jnp.float32)
        total = l_cond + la * l_act + ls * l_sg
        aux = {
            "l_cond": l_cond,
            "l_act": l_act,
            "lambda_act": la,
            "lambda_act_times_l_act": la * l_act,
            "l_sg": l_sg,
            "lambda_sg": ls,
            "lambda_sg_times_l_sg": ls * l_sg,
        }
        return total, aux

    diff = nnx.DiffState(0, trainable_filter)
    train_rng = jax.random.fold_in(rng, state.step)
    (loss, aux), grads = nnx.value_and_grad(loss_fn, argnums=diff, has_aux=True)(
        bundle, train_rng
    )

    params = state.params.filter(trainable_filter)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    new_params = optax.apply_updates(params, updates)
    nnx.update(bundle, new_params)
    new_full = nnx.state(bundle)

    new_state = dataclasses.replace(
        state, step=state.step + 1, params=new_full, opt_state=new_opt_state
    )
    return new_state, {"loss": loss, **aux}


@dataclasses.dataclass
class WorldModelTrainConfig:

    data_config_name: str = "pi0_libero"
    assets_base_dir: str = "./assets"
    checkpoint_base_dir: str = "./checkpoints"
    exp_name: str = "world_model"

    pi0_checkpoint: str = "gs://openpi-assets/checkpoints/pi0_base/params"
    skip_pi0_weight_load: bool = False

    stage1_steps: int = 10_000
    stage2_steps: int = 10_000
    stage3_steps: int = 5_000

    max_delta_t: int = 10
    handover_horizon_min: int = 1
    handover_horizon_max: int = 5
    lambda_act: float = 1.0
    lambda_sg: float = 0.1
    lambda_cond: float = 1.0
    freeze_token_reducer_stage1: bool = False
    token_reducer_kind: wm_mod.TokenReducerKind = "learned_cross_attn"
    action_encoder_kind: wm_mod.ActionEncoderKind = "gru"
    # If true, make the entire LLM subtree trainable (not just the *_1 shard).
    full_llm_trainable: bool = False

    wm_vlm_hidden_dim: int | None = None
    wm_token_dim: int | None = None
    wm_num_condition_tokens: int | None = None
    wm_num_reducer_heads: int | None = None
    wm_num_future_heads: int | None = None
    wm_action_embed_dim: int | None = None
    wm_gru_hidden_dim: int | None = None
    wm_gru_num_layers: int | None = None
    wm_transformer_num_heads: int | None = None
    wm_transformer_ffn_multiplier: int | None = None

    batch_size: int = 8
    seed: int = 42
    learning_rate: float = 3e-4
    warmup_steps: int = 500
    weight_decay: float = 1e-10
    log_interval: int = 50
    save_interval: int = 1000
    disable_checkpoints: bool = False
    keep_period: int | None = None
    checkpoint_max_to_keep: int | None = None
    overwrite: bool = False
    resume: bool = False
    resume_wm_export_step: int | None = None
    resume_wm_export_ckpt_root: str | None = None
    wm_init_from_export_step: int | None = None
    wm_init_from_export_ckpt_root: str | None = None
    resume_checkpoint_step: int | None = None
    wandb_enabled: bool = False
    project_name: str = "openpi_world_model"
    num_workers: int = 0
    fake_data: bool = False
    fake_data_size: int = 512

    # physical-intelligence/libero: filter by suite (spatial -> task_index 0..9); see world_model_data
    libero_suite: str | None = None
    libero_task_index_min: int | None = None
    libero_task_index_max: int | None = None
    libero_scratch_download_videos: bool = False
    l_act_targets_from_t: bool = False
    wm_logvar_only_finetune: bool = False
    lact_prefix_source: LactPrefixSource = "future_prefix"
    training_log_file: str | None = None


def _jax_tree_block_until_ready(tree: at.PyTree) -> None:
    jax.tree.map(
        lambda x: jax.block_until_ready(x) if isinstance(x, jax.Array) else x,
        tree,
    )


def _jax_array_to_host_for_checkpoint(x: jax.Array):
    h = jax.device_get(x)
    if jax.dtypes.issubdtype(h.dtype, jax.dtypes.prng_key):
        cpu = jax.devices("cpu")
        return jax.device_put(h, cpu[0]) if cpu else h
    return np.asarray(h)


def _train_state_numpy_snapshot(state: training_utils.TrainState) -> training_utils.TrainState:

    def to_host(x):
        if isinstance(x, jax.Array):
            return _jax_array_to_host_for_checkpoint(x)
        return x

    return jax.tree.map(to_host, state)


def _save_bundle_params_only(checkpoint_dir: epath.Path, train_state: training_utils.TrainState, step: int) -> None:
    bundle = nnx.merge(train_state.model_def, train_state.params)
    _, pi0_state = nnx.split(bundle.pi0)
    _, wm_state = nnx.split(bundle.wm)
    pi0_pure = pi0_state.to_pure_dict()
    wm_pure = wm_state.to_pure_dict()

    def to_host_leaf(x):
        return _jax_array_to_host_for_checkpoint(x) if isinstance(x, jax.Array) else x

    pi0_host = jax.tree.map(to_host_leaf, pi0_pure)
    wm_host = jax.tree.map(to_host_leaf, wm_pure)
    payload = {"pi0": pi0_host, "wm": wm_host}
    out = (checkpoint_dir / f"bundle_params_step_{step}" / "params").resolve()
    with ocp.PyTreeCheckpointer() as ckptr:
        ckptr.save(str(out), {"params": payload}, force=True)
    logger.info("Checkpoint saved (bundle params only: pi0+wm): %s (training step=%d)", out, step)


def _save_wm_params_only(checkpoint_dir: epath.Path, train_state: training_utils.TrainState, step: int) -> None:
    bundle = nnx.merge(train_state.model_def, train_state.params)
    _, wm_state = nnx.split(bundle.wm)
    pure = wm_state.to_pure_dict()
    pure_host = jax.tree.map(
        lambda x: _jax_array_to_host_for_checkpoint(x) if isinstance(x, jax.Array) else x,
        pure,
    )
    out = (checkpoint_dir / f"world_model_step_{step}" / "params").resolve()
    with ocp.PyTreeCheckpointer() as ckptr:
        # Do not mkdir(out): an empty dir makes Orbax raise "already exists". force=True overwrites on re-save.
        ckptr.save(str(out), {"params": pure_host}, force=True)
    logger.info("Checkpoint saved (world_model params only): %s (training step=%d)", out, step)


def _orbax_should_use_cpu_numpy_snapshot(
    *, checkpoint_manager: ocp.CheckpointManager, step: int
) -> tuple[bool, list[int]]:
    raw = os.environ.get("WM_CHECKPOINT_CPU_SNAPSHOT", "").strip().lower()
    if raw in ("1", "true", "yes"):
        return True, []
    if raw in ("0", "false", "no"):
        return False, []
    if os.environ.get("WM_CHECKPOINT_DISABLE_AUTO_CPU_SNAPSHOT", "0").strip().lower() in (
        "1",
        "true",
        "yes",
    ):
        return False, []
    try:
        gpus = jax.devices("gpu")
    except Exception:
        gpus = []
    if not (len(gpus) == 1 and jax.process_count() == 1):
        return False, []
    prior = sorted(s for s in checkpoint_manager.all_steps() if s < step)
    return (len(prior) > 0), prior


def _persist_training_checkpoint(
    *,
    checkpoint_manager: ocp.CheckpointManager,
    train_state: training_utils.TrainState,
    data_config: _config.DataConfig,
    ckpt_root: epath.Path,
    step: int,
) -> None:
    _jax_tree_block_until_ready(train_state)
    gc.collect()
    if hasattr(jax, "clear_caches"):
        jax.clear_caches()
    use_snap, prior_lt = _orbax_should_use_cpu_numpy_snapshot(
        checkpoint_manager=checkpoint_manager, step=int(step)
    )
    global _checkpoint_cpu_snapshot_mode_logged
    if not _checkpoint_cpu_snapshot_mode_logged:
        logger.info(
            "Checkpoint persist: WM_CHECKPOINT_CPU_SNAPSHOT=%r WM_CHECKPOINT_DISABLE_AUTO_CPU_SNAPSHOT=%r | "
            "single-GPU auto = host numpy for Orbax only when prior completed steps exist (see _orbax_should_use_cpu_numpy_snapshot).",
            os.environ.get("WM_CHECKPOINT_CPU_SNAPSHOT", ""),
            os.environ.get("WM_CHECKPOINT_DISABLE_AUTO_CPU_SNAPSHOT", "0"),
        )
        _checkpoint_cpu_snapshot_mode_logged = True

    logger.info("Checkpoint persist step=%d: writing bundle params-only export (pi0+wm) first", int(step))
    _save_bundle_params_only(ckpt_root, train_state, step)
    gc.collect()
    if hasattr(jax, "clear_caches"):
        jax.clear_caches()

    logger.info("Checkpoint persist step=%d: writing WM-only export (eval / resume_wm_export compatibility)", int(step))
    _save_wm_params_only(ckpt_root, train_state, step)
    gc.collect()
    if hasattr(jax, "clear_caches"):
        jax.clear_caches()

    logger.info(
        "Checkpoint persist step=%d: Orbax payload prior_steps_lt=%s → cpu_numpy_snapshot=%s",
        int(step),
        prior_lt,
        use_snap,
    )
    to_save = _train_state_numpy_snapshot(train_state) if use_snap else train_state
    if use_snap:
        gc.collect()
        if hasattr(jax, "clear_caches"):
            jax.clear_caches()
    _checkpoints.save_state(
        checkpoint_manager,
        to_save,
        _DummyDataLoader(data_config),
        step,
    )
    logger.info("Checkpoint persist step=%d: Orbax full save finished", int(step))


def run_training(cfg: WorldModelTrainConfig) -> None:
    if cfg.disable_checkpoints:
        logger.info("disable_checkpoints=True: skipping all checkpoint writes for this run.")
    if cfg.wm_logvar_only_finetune and cfg.stage2_steps <= 0:
        raise ValueError("wm_logvar_only_finetune requires stage2_steps > 0")

    base_train = _config.get_config(cfg.data_config_name)
    pi0_cfg = base_train.model
    if not isinstance(pi0_cfg, pi0_config_mod.Pi0Config):
        raise TypeError("World model training currently requires Pi0Config (not FAST).")

    assets_dirs = epath.Path(cfg.assets_base_dir).expanduser()
    data_config = base_train.data.create(assets_dirs, pi0_cfg)
    wm_data_cfg = wm_data.WorldModelDataConfig(
        max_delta_t=cfg.max_delta_t,
        action_horizon_min=cfg.handover_horizon_min,
        action_horizon_max=cfg.handover_horizon_max,
        action_keys=tuple(data_config.action_sequence_keys),
        l_act_targets_from_t=cfg.l_act_targets_from_t,
    )

    if cfg.fake_data:
        ds = wm_data.WorldModelFakeDataset(
            model_config=pi0_cfg, wm_cfg=wm_data_cfg, num_samples=cfg.fake_data_size, seed=cfg.seed
        )
    else:
        ds, _ = wm_data.create_world_model_lerobot_dataset(
            data_config,
            model_config=pi0_cfg,
            wm_cfg=wm_data_cfg,
            libero_suite=cfg.libero_suite,
            libero_task_index_min=cfg.libero_task_index_min,
            libero_task_index_max=cfg.libero_task_index_max,
            libero_scratch_download_videos=cfg.libero_scratch_download_videos,
        )

    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=wm_data.world_model_collate,
        drop_last=True,
        persistent_workers=cfg.num_workers > 0,
    )
    # PyTorch DataLoader iterators stop after one epoch; training may need more steps than len(dataset)//batch.
    _wm_iter_box: list = [iter(loader)]

    def _next_wm_batch():
        try:
            return next(_wm_iter_box[0])
        except StopIteration:
            _wm_iter_box[0] = iter(loader)
            return next(_wm_iter_box[0])

    ckpt_root = (epath.Path(cfg.checkpoint_base_dir).expanduser() / cfg.exp_name).resolve()
    checkpoint_manager, resuming = _checkpoints.initialize_checkpoint_dir(
        ckpt_root,
        keep_period=cfg.keep_period,
        overwrite=cfg.overwrite,
        resume=cfg.resume,
        max_to_keep=cfg.checkpoint_max_to_keep,
    )

    if cfg.wandb_enabled:
        wandb.init(name=cfg.exp_name, project=cfg.project_name, config=dataclasses.asdict(cfg))

    global_step = 0
    rng_master = jax.random.key(cfg.seed)
    total_steps = cfg.stage1_steps + cfg.stage2_steps + cfg.stage3_steps

    wm_data_summary = {
        "max_delta_t": wm_data_cfg.max_delta_t,
        "action_horizon_min": wm_data_cfg.action_horizon_min,
        "action_horizon_max": wm_data_cfg.action_horizon_max,
    }

    init_rng = jax.random.fold_in(rng_master, 0)
    pi0 = pi0_cfg.create(init_rng)
    if not cfg.skip_pi0_weight_load:
        pi0_ckpt = _resolve_pi0_checkpoint_path(cfg.pi0_checkpoint)
        pi0 = _load_pi0_weights(pi0, pi0_ckpt)
    wm_defaults = wm_mod.Pi0WorldModelConfig()
    wm_cfg = wm_mod.Pi0WorldModelConfig(
        vlm_hidden_dim=cfg.wm_vlm_hidden_dim if cfg.wm_vlm_hidden_dim is not None else wm_defaults.vlm_hidden_dim,
        token_dim=cfg.wm_token_dim if cfg.wm_token_dim is not None else wm_defaults.token_dim,
        num_condition_tokens=(
            cfg.wm_num_condition_tokens
            if cfg.wm_num_condition_tokens is not None
            else wm_defaults.num_condition_tokens
        ),
        proprio_dim=pi0_cfg.action_dim,
        action_dim=pi0_cfg.action_dim,
        num_reducer_heads=(
            cfg.wm_num_reducer_heads if cfg.wm_num_reducer_heads is not None else wm_defaults.num_reducer_heads
        ),
        num_future_heads=(
            cfg.wm_num_future_heads if cfg.wm_num_future_heads is not None else wm_defaults.num_future_heads
        ),
        action_embed_dim=(
            cfg.wm_action_embed_dim if cfg.wm_action_embed_dim is not None else wm_defaults.action_embed_dim
        ),
        gru_hidden_dim=cfg.wm_gru_hidden_dim if cfg.wm_gru_hidden_dim is not None else wm_defaults.gru_hidden_dim,
        gru_num_layers=cfg.wm_gru_num_layers if cfg.wm_gru_num_layers is not None else wm_defaults.gru_num_layers,
        transformer_num_heads=(
            cfg.wm_transformer_num_heads
            if cfg.wm_transformer_num_heads is not None
            else wm_defaults.transformer_num_heads
        ),
        transformer_ffn_multiplier=(
            cfg.wm_transformer_ffn_multiplier
            if cfg.wm_transformer_ffn_multiplier is not None
            else wm_defaults.transformer_ffn_multiplier
        ),
        token_reducer_kind=cfg.token_reducer_kind,
        action_encoder_kind=cfg.action_encoder_kind,
    )
    logger.info(
        "World model token_reducer_kind=%s, action_encoder_kind=%s",
        cfg.token_reducer_kind,
        cfg.action_encoder_kind,
    )
    wm = wm_mod.Pi0FutureWorldModel(wm_cfg, rngs=nnx.Rngs(jax.random.fold_in(init_rng, 1)))
    bundle = Pi0WorldModelTrainBundle(pi0, wm)
    params = nnx.state(bundle)
    model_def = nnx.graphdef(bundle)

    skip_stage1_for_wm_export = False
    use_orbax_resume = resuming
    if cfg.resume_wm_export_step is not None:
        if cfg.wm_init_from_export_step is not None:
            raise ValueError("resume_wm_export_step and wm_init_from_export_step are mutually exclusive")
        if not cfg.resume:
            raise ValueError("resume_wm_export_step requires resume=True")
        wm_export_root = (
            epath.Path(cfg.resume_wm_export_ckpt_root).expanduser().resolve()
            if cfg.resume_wm_export_ckpt_root
            else ckpt_root
        )
        bundle = load_bundle_with_wm_export(bundle, wm_export_root, int(cfg.resume_wm_export_step))
        params = nnx.state(bundle)
        global_step = int(cfg.resume_wm_export_step)
        skip_stage1_for_wm_export = True
        use_orbax_resume = False
        total_steps = int(cfg.resume_wm_export_step) + cfg.stage2_steps + cfg.stage3_steps
        logger.info(
            "resume_wm_export_step=%d: loaded WM from %s; skipping stage 1; global_step=%d (fresh stage-2+ optimizer).",
            cfg.resume_wm_export_step,
            wm_export_root / f"world_model_step_{int(cfg.resume_wm_export_step)}",
            global_step,
        )
    elif cfg.wm_init_from_export_step is not None:
        if not cfg.wm_init_from_export_ckpt_root:
            raise ValueError(
                "wm_init_from_export_step requires wm_init_from_export_ckpt_root "
                "(parent of world_model_step_* in the **source** experiment, not the new exp_name dir)"
            )
        if cfg.stage1_steps != 0:
            raise ValueError("wm_init_from_export_step requires stage1_steps=0 (stage 1 is skipped after loading WM export)")
        wm_init_root = epath.Path(cfg.wm_init_from_export_ckpt_root).expanduser().resolve()
        bundle = load_bundle_with_wm_export(bundle, wm_init_root, int(cfg.wm_init_from_export_step))
        params = nnx.state(bundle)
        skip_stage1_for_wm_export = True
        logger.info(
            "wm_init_from_export_step=%d: loaded WM weights from %s into a new run (Pi0 from --pi0-checkpoint; "
            "global_step starts at 0 and increments in this exp dir).",
            int(cfg.wm_init_from_export_step),
            wm_init_root / f"world_model_step_{int(cfg.wm_init_from_export_step)}",
        )

    _s2_flt = (
        trainable_filter_wm_logvar_head_only()
        if cfg.wm_logvar_only_finetune
        else trainable_filter_stage2()
    )
    _s2_extra: dict[str, float] = (
        {"lambda_act": 0.0} if cfg.wm_logvar_only_finetune else {"lambda_act": cfg.lambda_act}
    )
    stages: list[tuple[StageId, int, nnx.filterlib.Filter, dict[str, float] | None]] = [
        (1, cfg.stage1_steps, trainable_filter_stage1(freeze_token_reducer=cfg.freeze_token_reducer_stage1), None),
        (2, cfg.stage2_steps, _s2_flt, _s2_extra),
        (3, cfg.stage3_steps, trainable_filter_stage3(), {"lambda_act": cfg.lambda_act, "lambda_sg": cfg.lambda_sg}),
    ]

    append_wm_training_log(
        cfg.training_log_file,
        {
            "event": "run_start",
            "strategy": "three_stage",
            "pi0_action_horizon": int(pi0_cfg.action_horizon),
            "wm_data": wm_data_summary,
            "total_steps": int(total_steps),
            "config": serializable_wm_config(cfg),
        },
    )

    resume_stage2_from_step: int | None = None

    for stage_id, n_steps, flt, extra in stages:
        pstep = None
        if n_steps <= 0:
            logger.info("Skipping stage %d (n_steps=0)", stage_id)
            continue
        logger.info("Starting stage %d for %d steps", stage_id, n_steps)
        if skip_stage1_for_wm_export and stage_id == 1:
            assert cfg.resume_wm_export_step is not None or cfg.wm_init_from_export_step is not None
            if cfg.resume_wm_export_step is not None:
                logger.info(
                    "Skipping stage 1 (resume_wm_export_step=%d, WM loaded from export).",
                    cfg.resume_wm_export_step,
                )
            else:
                logger.info(
                    "Skipping stage 1 (wm_init_from_export_step=%d, WM loaded from export).",
                    int(cfg.wm_init_from_export_step),
                )
            skip_stage1_for_wm_export = False
            continue
        if stage_id > 1:
            try:
                del train_state
            except (NameError, UnboundLocalError):
                pass
            gc.collect()
            if hasattr(jax, "clear_caches"):
                jax.clear_caches()
        schedule = optax.warmup_cosine_decay_schedule(
            init_value=cfg.learning_rate / (cfg.warmup_steps + 1),
            peak_value=cfg.learning_rate,
            warmup_steps=cfg.warmup_steps,
            decay_steps=max(cfg.warmup_steps + 1, n_steps),
            end_value=cfg.learning_rate * 0.1,
        )
        tx = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adamw(learning_rate=schedule, b1=0.9, b2=0.95, weight_decay=cfg.weight_decay),
        )
        if stage_id > 1:
            gc.collect()
            if hasattr(jax, "clear_caches"):
                jax.clear_caches()
        # For resume restores, avoid allocating a fresh (potentially huge) opt_state
        # on device before Orbax restore. Use an abstract template instead; Orbax will
        # materialize the real opt_state/params from checkpoint.
        if (stage_id == 2 and resume_stage2_from_step is not None) or (use_orbax_resume and stage_id == 1):
            opt_state = jax.eval_shape(lambda p: tx.init(p), params.filter(flt))
        else:
            opt_state = tx.init(params.filter(flt))
        train_state = training_utils.TrainState(
            step=0,
            params=params,
            model_def=model_def,
            tx=tx,
            opt_state=opt_state,
            ema_decay=None,
            ema_params=None,
        )

        if stage_id == 2 and resume_stage2_from_step is not None:
            train_state = _checkpoints.restore_state(
                checkpoint_manager,
                train_state,
                _DummyDataLoader(data_config),
                step=resume_stage2_from_step,
            )
            params = train_state.params
            global_step = resume_stage2_from_step
            _rs2 = resume_stage2_from_step
            resume_stage2_from_step = None
            logger.info(
                "Restored stage 2 from checkpoint global step=%d (train_state.step=%s).",
                _rs2,
                int(jnp.asarray(train_state.step).item()),
            )

        if use_orbax_resume and stage_id == 1:
            latest = _checkpoints.resolve_resume_step(
                checkpoint_manager, cfg.resume_checkpoint_step
            )
            if latest is not None and 0 < latest < cfg.stage1_steps:
                train_state = _checkpoints.restore_state(
                    checkpoint_manager,
                    train_state,
                    _DummyDataLoader(data_config),
                    step=latest,
                )
                global_step = int(jnp.asarray(train_state.step).item())
                logger.info("Resuming stage 1 from train_state.step=%d (checkpoint step=%d).", global_step, latest)
            elif latest is not None and latest > cfg.stage1_steps:
                e2 = cfg.stage1_steps + cfg.stage2_steps
                if latest < e2:
                    resume_stage2_from_step = int(latest)
                    logger.info(
                        "Resume: checkpoint global step %d lies in stage 2 (<%d); skipping stage 1, "
                        "will full-restore train state at stage 2 start.",
                        resume_stage2_from_step,
                        e2,
                    )
                    continue
                raise NotImplementedError(
                    f"Checkpoint step {latest} is at or past stage 2 end ({e2}); resuming stage 3 is not "
                    f"implemented yet. Only: within stage 1, at end of stage 1, or within stage 2."
                )
            elif latest is not None and latest == cfg.stage1_steps:
                train_state = _checkpoints.restore_state(
                    checkpoint_manager,
                    train_state,
                    _DummyDataLoader(data_config),
                    step=latest,
                )
                params = train_state.params
                global_step = int(latest)
                logger.info(
                    "Resume: skipping stage 1 (checkpoint at stage1_steps=%d).",
                    cfg.stage1_steps,
                )
                continue

        loop_start = int(jnp.asarray(train_state.step).item())
        if loop_start > n_steps:
            raise ValueError(f"train_state.step={loop_start} exceeds stage {stage_id} n_steps={n_steps}")

        append_wm_training_log(
            cfg.training_log_file,
            {
                "event": "stage_start",
                "strategy": "three_stage",
                "global_step": int(global_step),
                "stage_id": int(stage_id),
                "loop_start": int(loop_start),
                "n_steps": int(n_steps),
                "lambda_act": float(extra["lambda_act"]) if extra else None,
                "lambda_sg": float(extra["lambda_sg"]) if extra and "lambda_sg" in extra else None,
            },
        )

        if stage_id in (2, 3):
            gc.collect()
            if hasattr(jax, "clear_caches"):
                jax.clear_caches()

        _jit = functools.partial(jax.jit, donate_argnums=(1,))
        if stage_id == 1:
            pstep = _jit(functools.partial(train_step_stage1, trainable_filter=flt))
        elif stage_id == 2:
            assert extra is not None
            if cfg.wm_logvar_only_finetune:
                pstep = _jit(functools.partial(train_step_stage2_wm_logvar_only, trainable_filter=flt))
            else:
                pstep = _jit(
                    functools.partial(
                        train_step_stage2,
                        trainable_filter=flt,
                        lambda_act=extra["lambda_act"],
                        lambda_cond=cfg.lambda_cond,
                        lact_prefix_source=cfg.lact_prefix_source,
                    )
                )
        else:
            assert extra is not None
            pstep = _jit(
                functools.partial(
                    train_step_stage3,
                    trainable_filter=flt,
                    lambda_act=extra["lambda_act"],
                    lambda_sg=extra["lambda_sg"],
                    lact_prefix_source=cfg.lact_prefix_source,
                )
            )

        infos: list[dict] = []

        # When stdout is not a TTY (e.g. piped through `tee`), tqdm's dynamic refresh produces
        # carriage-return artifacts in the log file (often rendered as lots of blank lines).
        # In that case, disable tqdm and use plain `print(..., flush=True)` for stable logs.
        _is_tty = sys.stdout.isatty()
        it = range(loop_start, n_steps)
        pbar = tqdm.tqdm(it, dynamic_ncols=True, desc=f"stage{stage_id}") if _is_tty else it

        for _ in pbar:
            batch = _next_wm_batch()
            obs_t, obs_f, obs_td1, wm = wm_data.batch_to_observations_and_wm_tensors(batch)
            obs_t = jax.tree.map(jnp.asarray, obs_t)
            obs_f = jax.tree.map(jnp.asarray, obs_f)
            obs_td1 = jax.tree.map(jnp.asarray, obs_td1)
            wm_j = jax.tree.map(jnp.asarray, wm)

            rng_master, rng_step = jax.random.split(rng_master)
            if stage_id == 1:
                train_state, info = pstep(
                    rng_step,
                    train_state,
                    obs_t,
                    obs_f,
                    wm_j["wm_action_prefix_pad"],
                    wm_j["wm_prefix_mask"],
                    wm_j["wm_delta_t"],
                )
            elif stage_id == 2:
                train_state, info = pstep(
                    rng_step,
                    train_state,
                    obs_t,
                    obs_f,
                    wm_j["wm_action_prefix_pad"],
                    wm_j["wm_prefix_mask"],
                    wm_j["wm_delta_t"],
                    wm_j["wm_actions_handover"],
                    wm_j["wm_handover_valid"],
                )
            else:
                train_state, info = pstep(
                    rng_step,
                    train_state,
                    obs_t,
                    obs_f,
                    obs_td1,
                    wm_j["wm_action_prefix_pad"],
                    wm_j["wm_prefix_mask"],
                    wm_j["wm_delta_t"],
                    wm_j["wm_actions_handover"],
                    wm_j["wm_handover_valid"],
                    wm_j["wm_semigroup_valid"],
                    wm_j["wm_delta2"].astype(jnp.float32),
                    wm_j["wm_sg_prefix_pad"],
                    wm_j["wm_sg_prefix_mask"],
                )

            train_state = jax.block_until_ready(train_state)
            info = jax.device_get(info)

            global_step += 1
            infos.append(info)
            if global_step % cfg.log_interval == 0:
                stacked = jax.tree.map(lambda *xs: np.mean(np.stack(xs)), *infos)
                logs = {f"s{stage_id}/{k}": float(v) for k, v in stacked.items()}
                logs["global_step"] = global_step
                msg = (
                    f"step {global_step} "
                    + ", ".join(f"{k}={v:.4f}" for k, v in logs.items() if k != "global_step")
                )
                if _is_tty:
                    pbar.write(msg)  # type: ignore[union-attr]
                else:
                    print(msg, flush=True)
                if cfg.wandb_enabled:
                    wandb.log(logs, step=global_step)
                append_wm_training_log(
                    cfg.training_log_file,
                    {
                        "event": "metrics",
                        "strategy": "three_stage",
                        "global_step": int(global_step),
                        "stage_id": int(stage_id),
                        "log_interval": int(cfg.log_interval),
                        "metrics_mean": {k: float(v) for k, v in stacked.items()},
                        "wandb_prefixed": logs,
                    },
                )
                infos = []

            if (
                not cfg.disable_checkpoints
                and cfg.save_interval > 0
                and (
                    global_step % cfg.save_interval == 0 or global_step == total_steps
                )
            ):
                _persist_training_checkpoint(
                    checkpoint_manager=checkpoint_manager,
                    train_state=train_state,
                    data_config=data_config,
                    ckpt_root=ckpt_root,
                    step=global_step,
                )

        if not cfg.disable_checkpoints and stage_id in (1, 2, 3):
            gs = global_step
            loop_already_saved = cfg.save_interval > 0 and (
                gs % cfg.save_interval == 0 or (stage_id == 3 and gs == total_steps)
            )
            if not loop_already_saved:
                logger.info(
                    "Saving checkpoint at end of stage %d (global_step=%d, save_interval did not hit this step).",
                    stage_id,
                    gs,
                )
                _persist_training_checkpoint(
                    checkpoint_manager=checkpoint_manager,
                    train_state=train_state,
                    data_config=data_config,
                    ckpt_root=ckpt_root,
                    step=gs,
                )
            else:
                logger.info(
                    "Stage %d finished at global_step=%d; checkpoint already written in-loop (save_interval/total_steps).",
                    stage_id,
                    gs,
                )
            logger.info(
                "Stage %d checkpoint: Orbax step %s under %s | bundle params-only: %s | WM-only: %s",
                stage_id,
                gs,
                str((ckpt_root / str(gs)).resolve()),
                str((ckpt_root / f"bundle_params_step_{int(gs)}" / "params").resolve()),
                str((ckpt_root / f"world_model_step_{int(gs)}" / "params").resolve()),
            )
        elif cfg.disable_checkpoints and stage_id in (1, 2, 3):
            logger.info(
                "Stage %d finished at global_step=%d (disable_checkpoints=True, no save).",
                stage_id,
                global_step,
            )

        if pstep is not None:
            del pstep
            pstep = None
        params = train_state.params
        gc.collect()
        if hasattr(jax, "clear_caches"):
            jax.clear_caches()

    if cfg.wandb_enabled:
        wandb.finish()
    checkpoint_manager.wait_until_finished()
    logger.info("Done. Checkpoints under %s", ckpt_root)


class _DummyDataLoader:
    def __init__(self, data_config: _config.DataConfig):
        self._dc = data_config

    def data_config(self) -> _config.DataConfig:
        return self._dc


def cli() -> WorldModelTrainConfig:
    return tyro.cli(WorldModelTrainConfig)
