# ruff: noqa: RUF002, RUF003

from __future__ import annotations

import dataclasses
import functools
import gc
import logging
import sys
from typing import Any, Literal

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
import openpi.training.world_model_data as wm_data
import openpi.training.world_model_training as wm_train

logger = logging.getLogger("openpi")

FourStageId = Literal[1, 2, 3, 4]
FourStage1ConditionSource = Literal["prefix_t", "future_prefix"]
FourStage1PrefixSource = wm_train.LactPrefixSource


def _count_param_bytes_and_llm_leaves(st: nnx.State) -> tuple[int, int, int]:
    """Return (num_leaves, total_bytes, llm_leaves) for a filtered nnx.State."""
    pure = st.to_pure_dict()
    leaves = jax.tree_util.tree_leaves(pure)
    total_bytes = 0
    for x in leaves:
        if hasattr(x, "nbytes"):
            total_bytes += int(x.nbytes)
    # crude but robust: count leaves in the llm subtree by key path substring
    llm_leaves = 0
    flat = jax.tree_util.tree_flatten_with_path(pure)[0]
    for path, value in flat:
        k = jax.tree_util.keystr(path)
        if "pi0/PaliGemma/llm" in k:
            llm_leaves += 1
    return len(leaves), total_bytes, llm_leaves


@at.typecheck
def train_step_four_stage1(
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    observation_t: _model.Observation,
    observation_f: _model.Observation,
    wm_action_prefix_pad: at.Float[at.Array, "b l a"],
    wm_prefix_mask: at.Bool[at.Array, "b l"],
    wm_actions_handover: at.Float[at.Array, "b ah a"],
    wm_handover_valid: at.Bool[at.Array, "b ah"],
    *,
    trainable_filter: nnx.filterlib.Filter,
    four_stage1_condition_source: FourStage1ConditionSource,
    four_stage1_prefix_source: FourStage1PrefixSource,
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    bundle = nnx.merge(state.model_def, state.params)

    def loss_fn(bundle: wm_train.Pi0WorldModelTrainBundle, rng2: at.KeyArrayLike):
        rng_pi = rng2
        if four_stage1_condition_source == "future_prefix":
            h_cond = jax.lax.stop_gradient(bundle.pi0.prefix_hidden_states(observation_f))
        else:
            h_cond = jax.lax.stop_gradient(bundle.pi0.prefix_hidden_states(observation_t))
        cond_toks = bundle.wm.reduce_tokens(h_cond, kv_mask=None)
        obs_prefix = wm_train.observation_for_lact_suffix_q(
            observation_t,
            observation_f,
            lact_prefix_source=four_stage1_prefix_source,
            wm_action_prefix_pad=wm_action_prefix_pad,
            wm_prefix_mask=wm_prefix_mask,
        )
        return bundle.pi0.compute_flow_matching_loss_with_future(
            rng_pi,
            obs_prefix,
            wm_actions_handover,
            train=True,
            future_condition_tokens=cond_toks,
            action_valid_mask=wm_handover_valid,
        )

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
    return new_state, {"loss": loss, "l_act": loss}


def trainable_filter_four_stage1() -> nnx.filterlib.Filter:
    return trainable_filter_four_stage1_with_llm(full_llm_trainable=False)


def trainable_filter_four_stage1_ae_only() -> nnx.filterlib.Filter:
    return nnx.All(
        nnx.Param,
        nnx.Any(
            nnx_utils.PathRegex(r"pi0/state_proj/.*"),
            nnx_utils.PathRegex(r"pi0/action_in_proj/.*"),
            nnx_utils.PathRegex(r"pi0/action_time_mlp_in/.*"),
            nnx_utils.PathRegex(r"pi0/action_time_mlp_out/.*"),
            nnx_utils.PathRegex(r"pi0/action_out_proj/.*"),
        ),
    )


def trainable_filter_four_stage1_reducer_only() -> nnx.filterlib.Filter:
    return nnx.All(
        nnx.Param,
        nnx.Any(
            nnx_utils.PathRegex(r"wm/token_reducer/.*"),
            nnx_utils.PathRegex(r"wm/reducer_vlm_to_token/.*"),
        ),
    )


def trainable_filter_four_stage_wm_full() -> nnx.filterlib.Filter:
    return nnx.All(nnx.Param, nnx_utils.PathRegex(r"wm/.*"))


def trainable_filter_four_stage1_with_llm(*, full_llm_trainable: bool) -> nnx.filterlib.Filter:
    llm_pat = (
        r"pi0/PaliGemma/llm/.*"
        if full_llm_trainable
        else r"pi0/PaliGemma/llm/.*_1/.*"
    )
    ae = nnx.All(
        nnx.Param,
        nnx.Any(
            nnx_utils.PathRegex(r"pi0/state_proj/.*"),
            nnx_utils.PathRegex(r"pi0/action_in_proj/.*"),
            nnx_utils.PathRegex(r"pi0/action_time_mlp_in/.*"),
            nnx_utils.PathRegex(r"pi0/action_time_mlp_out/.*"),
            nnx_utils.PathRegex(r"pi0/action_out_proj/.*"),
            nnx_utils.PathRegex(llm_pat),
        ),
    )
    reducer = nnx.All(
        nnx.Param,
        nnx.Any(
            nnx_utils.PathRegex(r"wm/token_reducer/.*"),
            nnx_utils.PathRegex(r"wm/reducer_vlm_to_token/.*"),
        ),
    )
    return nnx.Any(ae, reducer)


def trainable_filter_four_stage2() -> nnx.filterlib.Filter:
    wm_all = nnx.All(nnx.Param, nnx_utils.PathRegex(r"wm/.*"))
    return nnx.All(
        wm_all,
        nnx.Not(nnx_utils.PathRegex(r"wm/token_reducer/.*")),
        nnx.Not(nnx_utils.PathRegex(r"wm/reducer_vlm_to_token/.*")),
    )


def trainable_filter_four_stage2_no_logvar_head() -> nnx.filterlib.Filter:
    """Stage2 variant: do not train variance head (logvar_head)."""
    return nnx.All(trainable_filter_four_stage2(), nnx.Not(nnx_utils.PathRegex(r"wm/.*/logvar_head/.*")))


def trainable_filter_four_stage3_lcond_wm_no_reducer_lact_pi0(*, full_llm_trainable: bool) -> nnx.filterlib.Filter:
    llm_pat = (
        r"pi0/PaliGemma/llm/.*"
        if full_llm_trainable
        else r"pi0/PaliGemma/llm/.*_1/.*"
    )
    pi0_layers = nnx.All(
        nnx.Param,
        nnx.Any(
            nnx_utils.PathRegex(r"pi0/state_proj/.*"),
            nnx_utils.PathRegex(r"pi0/action_in_proj/.*"),
            nnx_utils.PathRegex(r"pi0/action_time_mlp_in/.*"),
            nnx_utils.PathRegex(r"pi0/action_time_mlp_out/.*"),
            nnx_utils.PathRegex(r"pi0/action_out_proj/.*"),
            nnx_utils.PathRegex(llm_pat),
        ),
    )
    return nnx.Any(trainable_filter_four_stage2(), pi0_layers)


def _merge_wm_grads_lcond_subset_into_lact_full(g_act: Any, g_cond: Any) -> Any:
    d_cond: dict[str, Any] = {}
    for path, leaf in jax.tree_util.tree_leaves_with_path(g_cond):
        d_cond[jax.tree_util.keystr(path)] = leaf

    def _add(path: Any, act_leaf: Any) -> Any:
        ks = jax.tree_util.keystr(path)
        c_leaf = d_cond.get(ks)
        if c_leaf is None:
            return act_leaf
        return act_leaf + c_leaf

    return jax.tree_util.tree_map_with_path(_add, g_act)


@at.typecheck
def train_step_four_stage3_lcond_no_reducer_lact_full_wm_frozen_pi0(
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
    lambda_act: float,
    lambda_cond: float,
    lact_prefix_source: wm_train.LactPrefixSource,
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    bundle = nnx.merge(state.model_def, state.params)
    flt_cond = trainable_filter_four_stage2_no_logvar_head()
    flt_act = trainable_filter_four_stage_wm_full()

    def loss_cond_fn(bundle: wm_train.Pi0WorldModelTrainBundle, rng2: at.KeyArrayLike):
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
        )
        l_cond = wm_mod.heteroscedastic_gaussian_nll(target, out.mu, out.log_var)
        lc = jnp.asarray(lambda_cond, dtype=jnp.float32)
        aux = {
            "l_cond": l_cond,
            "lambda_cond": lc,
            "lambda_cond_times_l_cond": lc * l_cond,
        }
        return lc * l_cond, aux

    def loss_act_fn(bundle: wm_train.Pi0WorldModelTrainBundle, rng2: at.KeyArrayLike):
        rng_wm, _, rng_act = jax.random.split(rng2, 3)
        h_t = jax.lax.stop_gradient(bundle.pi0.prefix_hidden_states(observation_t))
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
        obs_lact = wm_train.observation_for_lact_suffix_q(
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
        la = jnp.asarray(lambda_act, dtype=jnp.float32)
        aux = {
            "l_act": l_act,
            "lambda_act": la,
            "lambda_act_times_l_act": la * l_act,
        }
        return la * l_act, aux

    diff_c = nnx.DiffState(0, flt_cond)
    diff_a = nnx.DiffState(0, flt_act)
    train_rng = jax.random.fold_in(rng, state.step)
    rng_c, rng_a = jax.random.split(train_rng)
    (loss_c, aux_c), grads_c = nnx.value_and_grad(loss_cond_fn, argnums=diff_c, has_aux=True)(bundle, rng_c)
    (loss_a, aux_a), grads_a = nnx.value_and_grad(loss_act_fn, argnums=diff_a, has_aux=True)(bundle, rng_a)
    grads = _merge_wm_grads_lcond_subset_into_lact_full(grads_a, grads_c)
    loss = loss_c + loss_a
    aux = {
        **aux_c,
        **aux_a,
        "loss": loss,
        "detach_wm_mu_for_lact": jnp.asarray(0.0, dtype=jnp.float32),
    }

    params = state.params.filter(flt_act)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    new_params = optax.apply_updates(params, updates)
    nnx.update(bundle, new_params)
    new_full = nnx.state(bundle)

    new_state = dataclasses.replace(
        state, step=state.step + 1, params=new_full, opt_state=new_opt_state
    )
    return new_state, aux


@dataclasses.dataclass
class FourStageWorldModelTrainConfig(wm_train.WorldModelTrainConfig):

    stage4_steps: int = 5_000
    four_stage1_condition_source: FourStage1ConditionSource = "future_prefix"
    four_stage1_prefix_source: FourStage1PrefixSource = "future_prefix"
    four_stage3_lambda_cond: float = 1.0
    four_stage3_detach_wm_mu_for_lact: bool = False
    four_stage3_lcond_train_wm_no_reducer: bool = False
    four_stage1_ae_only: bool = False
    four_stage3_ae_only: bool = False
    four_stage1_reducer_only: bool = False
    four_stage3_lcond_no_reducer_lact_full_wm: bool = False
    resume_four_stage_orbax_ckpt_root: str | None = None
    grad_accum_steps: int = 1
    # --- Optional data-parallel (multi-GPU) for stage3 training loop ---
    # If >1, use `jax.pmap` data-parallel for four-stage 3 (joint stage) with cross-device gradient sync.
    data_parallel_devices: int = 1
    # Optional: explicitly set per-device batch size for data-parallel runs.
    # If None, `batch_size` is treated as per-device and DataLoader batch becomes batch_size * data_parallel_devices.
    per_device_batch_size: int | None = None
    data_parallel_axis_name: str = "dp"


def _four_stage_end_global_steps(cfg: FourStageWorldModelTrainConfig) -> tuple[int, int, int, int]:
    s1, s2, s3, s4 = cfg.stage1_steps, cfg.stage2_steps, cfg.stage3_steps, cfg.stage4_steps
    e1 = s1
    e2 = s1 + s2
    e3 = s1 + s2 + s3
    e4 = s1 + s2 + s3 + s4
    return e1, e2, e3, e4


def run_training(cfg: FourStageWorldModelTrainConfig) -> None:
    if cfg.disable_checkpoints:
        logger.info("disable_checkpoints=True: skipping all checkpoint writes for this run.")
    if cfg.wm_logvar_only_finetune and cfg.stage2_steps <= 0:
        raise ValueError("wm_logvar_only_finetune requires stage2_steps > 0")

    if cfg.four_stage1_reducer_only and cfg.four_stage1_ae_only:
        raise ValueError("four_stage1_reducer_only and four_stage1_ae_only are mutually exclusive.")
    if cfg.four_stage3_lcond_no_reducer_lact_full_wm:
        if cfg.four_stage3_ae_only or cfg.four_stage3_lcond_train_wm_no_reducer:
            raise ValueError(
                "four_stage3_lcond_no_reducer_lact_full_wm is incompatible with "
                "four_stage3_ae_only / four_stage3_lcond_train_wm_no_reducer."
            )
        if cfg.four_stage3_detach_wm_mu_for_lact:
            raise ValueError(
                "four_stage3_lcond_no_reducer_lact_full_wm requires four_stage3_detach_wm_mu_for_lact=False."
            )
        if int(cfg.grad_accum_steps) > 1:
            raise ValueError("four_stage3_lcond_no_reducer_lact_full_wm currently only supports grad_accum_steps=1.")
        _dp_chk = int(cfg.data_parallel_devices) if cfg.data_parallel_devices else 1
        if _dp_chk > 1:
            raise ValueError("four_stage3_lcond_no_reducer_lact_full_wm does not support data_parallel_devices>1.")

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

    # Data-parallel (multi-GPU) support: DataLoader emits a global batch which will be reshaped to
    # [n_devices, per_device_batch, ...] inside the stage3 training loop.
    dp = int(cfg.data_parallel_devices) if cfg.data_parallel_devices else 1
    if dp < 1:
        raise ValueError(f"data_parallel_devices must be >=1, got {dp}")
    if dp > 1:
        ldc = int(jax.local_device_count())
        if ldc < dp:
            raise RuntimeError(
                "data_parallel_devices=%d requested, but jax.local_device_count()=%d. "
                "This Python/JAX environment can only see %d CUDA device(s). "
                "Fix JAX multi-GPU visibility first (e.g., ensure the runtime exposes multiple GPUs and "
                "JAX is installed with CUDA multi-GPU support), then rerun."
                % (dp, ldc, ldc)
            )
    per_dev_bs = int(cfg.per_device_batch_size) if cfg.per_device_batch_size is not None else int(cfg.batch_size)
    loader_batch = per_dev_bs * dp if dp > 1 else int(cfg.batch_size)
    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=loader_batch,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=wm_data.world_model_collate,
        drop_last=True,
        persistent_workers=cfg.num_workers > 0,
    )
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

    resume_s3_orbax_mngr: ocp.CheckpointManager | None = None
    resume_s3_orbax_step: int | None = None
    resume_s3_orbax_root: epath.Path | None = None
    if cfg.resume_four_stage_orbax_ckpt_root:
        if cfg.resume_wm_export_step is not None:
            raise ValueError(
                "resume_four_stage_orbax_ckpt_root and resume_wm_export_step are mutually exclusive; use only one resume mode."
            )
        if cfg.wm_init_from_export_step is not None:
            raise ValueError(
                "resume_four_stage_orbax_ckpt_root and wm_init_from_export_step are mutually exclusive; use only one init mode."
            )
        if not cfg.resume:
            raise ValueError("resume_four_stage_orbax_ckpt_root requires --resume.")
        if cfg.resume_checkpoint_step is None:
            raise ValueError(
                "resume_four_stage_orbax_ckpt_root requires --resume-checkpoint-step=<orbax step>."
            )
        if cfg.stage1_steps != 0 or cfg.stage2_steps != 0 or cfg.stage3_steps <= 0:
            raise ValueError(
                "resume_four_stage_orbax_ckpt_root only supports stage1_steps=stage2_steps=0 with stage3_steps>0."
            )
        resume_s3_orbax_root = epath.Path(cfg.resume_four_stage_orbax_ckpt_root).expanduser().resolve()
        resume_s3_orbax_mngr, _ = _checkpoints.initialize_checkpoint_dir(
            resume_s3_orbax_root,
            keep_period=None,
            overwrite=False,
            resume=True,
            max_to_keep=1,
        )
        resume_s3_orbax_step = _checkpoints.resolve_resume_step(
            resume_s3_orbax_mngr, cfg.resume_checkpoint_step
        )
        assert resume_s3_orbax_step is not None
        logger.info(
            "Stage-3 Orbax resume source: root=%s step=%d (writes still go to exp ckpt_root=%s).",
            resume_s3_orbax_root,
            int(resume_s3_orbax_step),
            ckpt_root,
        )

    if cfg.wandb_enabled:
        wandb.init(name=cfg.exp_name, project=cfg.project_name, config=dataclasses.asdict(cfg))

    global_step = 0
    rng_master = jax.random.key(cfg.seed)
    total_steps = cfg.stage1_steps + cfg.stage2_steps + cfg.stage3_steps + cfg.stage4_steps
    if (
        resume_s3_orbax_step is not None
        and cfg.stage1_steps == 0
        and cfg.stage2_steps == 0
        and cfg.stage4_steps == 0
        and cfg.stage3_steps > 0
    ):
        total_steps = int(resume_s3_orbax_step) + cfg.stage3_steps
    ends = _four_stage_end_global_steps(cfg)

    init_rng = jax.random.fold_in(rng_master, 0)
    pi0 = pi0_cfg.create(init_rng)
    if not cfg.skip_pi0_weight_load:
        pi0_ckpt = wm_train._resolve_pi0_checkpoint_path(cfg.pi0_checkpoint)
        pi0 = wm_train._load_pi0_weights(pi0, pi0_ckpt)
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
        "Four-stage WM training | token_reducer_kind=%s | action_encoder_kind=%s | stage1_cond=%s | stage1_prefix=%s | lact_prefix=%s | ends @ global_step %s",
        cfg.token_reducer_kind,
        cfg.action_encoder_kind,
        cfg.four_stage1_condition_source,
        cfg.four_stage1_prefix_source,
        cfg.lact_prefix_source,
        ends,
    )
    wm_data_summary = {
        "max_delta_t": wm_data_cfg.max_delta_t,
        "action_horizon_min": wm_data_cfg.action_horizon_min,
        "action_horizon_max": wm_data_cfg.action_horizon_max,
    }
    wm = wm_mod.Pi0FutureWorldModel(wm_cfg, rngs=nnx.Rngs(jax.random.fold_in(init_rng, 1)))
    bundle = wm_train.Pi0WorldModelTrainBundle(pi0, wm)
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
        bundle = wm_train.load_bundle_with_wm_export(bundle, wm_export_root, int(cfg.resume_wm_export_step))
        params = nnx.state(bundle)
        global_step = int(cfg.resume_wm_export_step)
        skip_stage1_for_wm_export = True
        use_orbax_resume = False
        logger.info(
            "resume_wm_export_step=%d: loaded WM from %s; skipping four-stage 1; global_step=%d (fresh optimizer).",
            cfg.resume_wm_export_step,
            wm_export_root / f"world_model_step_{int(cfg.resume_wm_export_step)}",
            global_step,
        )
        logger.warning(
            "``world_model_step_*`` contains WM only (incl. token_reducer); Pi0 Action Expert still comes from --pi0-checkpoint. "
            "Stage-1 AE weights cannot be restored from this export; use a full Orbax checkpoint (e.g. increase max_to_keep)."
        )
        total_steps = int(cfg.resume_wm_export_step) + cfg.stage2_steps + cfg.stage3_steps + cfg.stage4_steps
    elif cfg.wm_init_from_export_step is not None:
        if not cfg.wm_init_from_export_ckpt_root:
            raise ValueError(
                "wm_init_from_export_step requires wm_init_from_export_ckpt_root (source experiment checkpoint root, not the new exp_name)"
            )
        if cfg.stage1_steps != 0:
            raise ValueError("wm_init_from_export_step requires four_stage stage1_steps=0")
        wm_init_root = epath.Path(cfg.wm_init_from_export_ckpt_root).expanduser().resolve()
        bundle = wm_train.load_bundle_with_wm_export(bundle, wm_init_root, int(cfg.wm_init_from_export_step))
        params = nnx.state(bundle)
        skip_stage1_for_wm_export = True
        logger.info(
            "wm_init_from_export_step=%d: loaded WM init from %s (global_step in new exp dir starts from training loop).",
            int(cfg.wm_init_from_export_step),
            wm_init_root / f"world_model_step_{int(cfg.wm_init_from_export_step)}",
        )
        logger.warning(
            "``world_model_step_*`` contains WM only; Pi0 (incl. AE) comes from --pi0-checkpoint."
        )

    if cfg.four_stage1_reducer_only:
        s1_flt = trainable_filter_four_stage1_reducer_only()
        s1_tag = "four_s1_reducer_only"
    elif cfg.four_stage1_ae_only:
        s1_flt = trainable_filter_four_stage1_ae_only()
        s1_tag = "four_s1_ae_only"
    else:
        s1_flt = trainable_filter_four_stage1_with_llm(full_llm_trainable=cfg.full_llm_trainable)
        s1_tag = "four_s1_ae_reducer"

    if cfg.four_stage3_ae_only:
        s3_flt = trainable_filter_four_stage1_ae_only()
        s3_tag = "four_s3_ae_only_lact"
    elif cfg.four_stage3_lcond_no_reducer_lact_full_wm:
        s3_flt = trainable_filter_four_stage_wm_full()
        s3_tag = "four_s3_lcond_nored_lact_fullwm_frozen_pi0"
    elif cfg.four_stage3_lcond_train_wm_no_reducer:
        s3_flt = trainable_filter_four_stage3_lcond_wm_no_reducer_lact_pi0(full_llm_trainable=cfg.full_llm_trainable)
        s3_tag = "four_s3_lcond_nored_lact_pi0"
    else:
        s3_flt = wm_train.trainable_filter_stage2_with_llm(full_llm_trainable=cfg.full_llm_trainable)
        s3_tag = "four_s3_joint_lcond_lact"

    if cfg.wm_logvar_only_finetune:
        s2_flt = wm_train.trainable_filter_wm_logvar_head_only()
        s2_tag = "four_s2_wm_logvar_only"
    else:
        s2_flt = trainable_filter_four_stage2_no_logvar_head()
        s2_tag = "four_s2_wm_lcond"

    stages: list[
        tuple[FourStageId, int, nnx.filterlib.Filter, dict[str, float] | None, str]
    ] = [
        (
            1,
            cfg.stage1_steps,
            s1_flt,
            None,
            s1_tag,
        ),
        (2, cfg.stage2_steps, s2_flt, None, s2_tag),
        (
            3,
            cfg.stage3_steps,
            s3_flt,
            {"lambda_act": cfg.lambda_act},
            s3_tag,
        ),
        (
            4,
            cfg.stage4_steps,
            wm_train.trainable_filter_stage2_with_llm(full_llm_trainable=cfg.full_llm_trainable),
            {"lambda_act": cfg.lambda_act, "lambda_sg": cfg.lambda_sg},
            "four_s4_semigroup",
        ),
    ]

    wm_train.append_wm_training_log(
        cfg.training_log_file,
        {
            "event": "run_start",
            "strategy": "four_stage",
            "four_stage1_condition_source": cfg.four_stage1_condition_source,
            "four_stage1_prefix_source": cfg.four_stage1_prefix_source,
            "lact_prefix_source": cfg.lact_prefix_source,
            "four_stage_ends_global_step": {
                "end_stage1": ends[0],
                "end_stage2": ends[1],
                "end_stage3": ends[2],
                "end_stage4": ends[3],
            },
            "pi0_action_horizon": int(pi0_cfg.action_horizon),
            "wm_data": wm_data_summary,
            "total_steps": int(total_steps),
            "config": wm_train.serializable_wm_config(cfg),
        },
    )

    resume_four_stage2_from_step: int | None = None
    # If resume lands exactly at the end of four-stage 2 (global_step == stage1+stage2), we must restore
    # the full train_state and continue into four-stage 3 from that point (previously NotImplemented).
    resume_four_stage3_from_step: int | None = None
    # Only for the following loop-range adjustment.
    resume_stage3_extra_from_orbax: bool = False

    for stage_id, n_steps, flt, extra, tag in stages:
        pstep = None
        pstep_accum = None
        pstep_accum_dp = None
        if n_steps <= 0:
            logger.info("Skipping four-stage %d (%s) n_steps=0", stage_id, tag)
            continue
        # Resume at global_step == stage1+stage2: stage-1 handler sets ``resume_four_stage3_from_step``
        # and ``continue``s — do not run stage 2 on a fresh TrainState (would log 4s2 / wrong graph).
        if stage_id == 2 and resume_four_stage3_from_step is not None:
            logger.info(
                "Skipping four-stage 2: resume checkpoint global_step=%d is end of stage 2; "
                "full train_state restores at four-stage 3 entry.",
                int(resume_four_stage3_from_step),
            )
            continue
        logger.info("Starting four-stage %d (%s) for %d steps", stage_id, tag, n_steps)
        if skip_stage1_for_wm_export and stage_id == 1:
            assert cfg.resume_wm_export_step is not None or cfg.wm_init_from_export_step is not None
            if cfg.resume_wm_export_step is not None:
                logger.info(
                    "Skipping four-stage 1 (resume_wm_export_step=%d, WM loaded from export).",
                    cfg.resume_wm_export_step,
                )
            else:
                logger.info(
                    "Skipping four-stage 1 (wm_init_from_export_step=%d, WM loaded from export).",
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
        if (
            (stage_id == 2 and resume_four_stage2_from_step is not None)
            or (use_orbax_resume and stage_id == 1)
            or (stage_id == 3 and resume_s3_orbax_step is not None)
            or (stage_id == 3 and resume_four_stage3_from_step is not None)
        ):
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

        # Log what is actually trainable in this stage (helps verify LLM is included).
        try:
            n_leaves, n_bytes, llm_leaves = _count_param_bytes_and_llm_leaves(params.filter(flt))
            logger.info(
                "Trainable params (four-stage %d %s): leaves=%d, bytes=%.3f GiB, llm_leaves=%d",
                stage_id,
                tag,
                n_leaves,
                n_bytes / (1024**3),
                llm_leaves,
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("Could not summarize trainable params: %s", e)

        if stage_id == 2 and resume_four_stage2_from_step is not None:
            train_state = _checkpoints.restore_state(
                checkpoint_manager,
                train_state,
                wm_train._DummyDataLoader(data_config),
                step=resume_four_stage2_from_step,
            )
            params = train_state.params
            global_step = resume_four_stage2_from_step
            _rs2 = resume_four_stage2_from_step
            resume_four_stage2_from_step = None
            logger.info(
                "Restored four-stage 2 from checkpoint global step=%d (train_state.step=%s).",
                _rs2,
                int(jnp.asarray(train_state.step).item()),
            )

        if stage_id == 3 and resume_s3_orbax_step is not None:
            assert resume_s3_orbax_mngr is not None and resume_s3_orbax_root is not None
            _rs3 = int(resume_s3_orbax_step)
            try:
                train_state = _checkpoints.restore_state(
                    resume_s3_orbax_mngr,
                    train_state,
                    wm_train._DummyDataLoader(data_config),
                    step=_rs3,
                )
            except ValueError as e:
                logger.warning(
                    "Full train_state restore from stage-3 Orbax source failed (step=%d): %s. "
                    "Falling back to params-only restore with fresh stage-3 optimizer state.",
                    _rs3,
                    e,
                )
                train_state = _checkpoints.restore_params_only(
                    resume_s3_orbax_mngr,
                    train_state,
                    step=_rs3,
                )
                train_state = dataclasses.replace(train_state, opt_state=tx.init(train_state.params.filter(flt)))
            params = train_state.params
            global_step = _rs3
            resume_s3_orbax_step = None
            resume_stage3_extra_from_orbax = True
            logger.info(
                "Restored four-stage 3 from Orbax root=%s step=%d → global_step=%d; "
                "continue from restored train_state.step=%d for %d extra substeps.",
                resume_s3_orbax_root,
                _rs3,
                global_step,
                int(jnp.asarray(train_state.step).item()),
                n_steps,
            )

        if stage_id == 3 and resume_four_stage3_from_step is not None:
            _rs3c = int(resume_four_stage3_from_step)
            try:
                train_state = _checkpoints.restore_state(
                    checkpoint_manager,
                    train_state,
                    wm_train._DummyDataLoader(data_config),
                    step=_rs3c,
                )
            except ValueError as e:
                # Stage-3 strategy flags may change trainable filters vs stage-2 checkpoint,
                # making opt_state tree shapes incompatible. In this case, restore params only
                # and keep freshly initialized optimizer state for stage 3.
                logger.warning(
                    "Full train_state restore at four-stage 3 entry failed (step=%d): %s. "
                    "Falling back to params-only restore with fresh stage-3 optimizer state.",
                    _rs3c,
                    e,
                )
                train_state = _checkpoints.restore_params_only(
                    checkpoint_manager,
                    train_state,
                    step=_rs3c,
                )
                train_state = dataclasses.replace(train_state, opt_state=tx.init(train_state.params.filter(flt)))
            params = train_state.params
            global_step = _rs3c
            resume_four_stage3_from_step = None
            _prev_sub = int(jnp.asarray(train_state.step).item())
            # Orbax directory step is global (e.g. 35000); ``train_state.step`` on disk is the stage-2
            # substep counter (0..stage2_steps). Stage 3 loops use range(0, stage3_steps) like a fresh
            # stage handoff — reset substep so we run the full ``n_steps`` with correct ``global_step``.
            train_state = train_state.replace(step=jnp.asarray(0, dtype=jnp.asarray(train_state.step).dtype))
            logger.info(
                "Restored four-stage 3 entry from experiment checkpoint global_step=%d "
                "(train_state.step reset %d→0 for stage-3 substep counter; params/opt intact).",
                global_step,
                _prev_sub,
            )

        if use_orbax_resume and stage_id == 1:
            latest = _checkpoints.resolve_resume_step(
                checkpoint_manager, cfg.resume_checkpoint_step
            )
            if latest is not None and 0 < latest < cfg.stage1_steps:
                train_state = _checkpoints.restore_state(
                    checkpoint_manager,
                    train_state,
                    wm_train._DummyDataLoader(data_config),
                    step=latest,
                )
                global_step = int(jnp.asarray(train_state.step).item())
                logger.info(
                    "Resuming four-stage 1 from train_state.step=%d (checkpoint step=%d).",
                    global_step,
                    latest,
                )
            elif latest is not None and latest == cfg.stage1_steps:
                train_state = _checkpoints.restore_state(
                    checkpoint_manager,
                    train_state,
                    wm_train._DummyDataLoader(data_config),
                    step=latest,
                )
                params = train_state.params
                global_step = int(latest)
                logger.info(
                    "Resume: skipping four-stage 1 (checkpoint at stage1_steps=%d).",
                    cfg.stage1_steps,
                )
                continue
            elif latest is not None and latest > cfg.stage1_steps:
                e2 = cfg.stage1_steps + cfg.stage2_steps
                if latest < e2:
                    resume_four_stage2_from_step = int(latest)
                    logger.info(
                        "Resume: four-stage checkpoint global step %d lies in stage 2 (<%d); skipping stage 1, "
                        "will full-restore train state at four-stage 2 start.",
                        resume_four_stage2_from_step,
                        e2,
                    )
                    continue
                if latest == e2:
                    resume_four_stage3_from_step = int(latest)
                    logger.info(
                        "Resume: four-stage checkpoint global step %d is exactly end of stage 2 (%d); "
                        "skipping stages 1–2, will full-restore train state at four-stage 3 entry.",
                        resume_four_stage3_from_step,
                        e2,
                    )
                    continue
                raise NotImplementedError(
                    f"Four-stage resume past stage 2 end ({e2}) is not implemented (latest={latest}). "
                    "Supported: within stage 1, end of stage 1, within stage 2, or exactly end of stage 2 "
                    "(continue into stage 3)."
                )

        loop_start = int(jnp.asarray(train_state.step).item())
        if not (stage_id == 3 and resume_stage3_extra_from_orbax) and loop_start > n_steps:
            raise ValueError(
                f"train_state.step={loop_start} exceeds four-stage {stage_id} n_steps={n_steps}"
            )

        wm_train.append_wm_training_log(
            cfg.training_log_file,
            {
                "event": "stage_start",
                "strategy": "four_stage",
                "global_step": int(global_step),
                "four_stage_id": int(stage_id),
                "stage_tag": tag,
                "loop_start": int(loop_start),
                "n_steps": int(n_steps),
                "lambda_act": float(extra["lambda_act"]) if extra and "lambda_act" in extra else None,
                "lambda_sg": float(extra["lambda_sg"]) if extra and "lambda_sg" in extra else None,
            },
        )

        if stage_id in (2, 3, 4):
            gc.collect()
            if hasattr(jax, "clear_caches"):
                jax.clear_caches()

        _jit = functools.partial(jax.jit, donate_argnums=(1,))
        if stage_id == 1:
            pstep = _jit(
                functools.partial(
                    train_step_four_stage1,
                    trainable_filter=flt,
                    four_stage1_condition_source=cfg.four_stage1_condition_source,
                    four_stage1_prefix_source=cfg.four_stage1_prefix_source,
                )
            )
        elif stage_id == 2:
            if cfg.wm_logvar_only_finetune:
                pstep = _jit(
                    functools.partial(wm_train.train_step_stage2_wm_logvar_only, trainable_filter=flt)
                )
            else:
                # Keep original stage2 structure, but do not train variance head.
                pstep = _jit(functools.partial(wm_train.train_step_stage1_no_logvar, trainable_filter=flt))
        elif stage_id == 3:
            assert extra is not None
            # Stage2 -> Stage3 is a sharp change in compiled graph + peak memory (esp. grad accumulation).
            # Do an extra flush right before compiling Stage3 to reduce the chance of OOM at the boundary.
            gc.collect()
            if hasattr(jax, "clear_caches"):
                jax.clear_caches()
            logger.info(
                "Four-stage %d (%s): flushed Python/JAX caches before compiling step (grad_accum_steps=%d).",
                stage_id,
                tag,
                int(cfg.grad_accum_steps),
            )
            if cfg.four_stage3_lcond_no_reducer_lact_full_wm:
                pstep = _jit(
                    functools.partial(
                        train_step_four_stage3_lcond_no_reducer_lact_full_wm_frozen_pi0,
                        lambda_act=extra["lambda_act"],
                        lambda_cond=cfg.four_stage3_lambda_cond,
                        lact_prefix_source=cfg.lact_prefix_source,
                    )
                )
            elif cfg.grad_accum_steps > 1:
                pstep_accum = jax.jit(
                    functools.partial(
                        wm_train.train_step_stage2_grad_accum,
                        trainable_filter=flt,
                        lambda_act=extra["lambda_act"],
                        lambda_cond=cfg.four_stage3_lambda_cond,
                        lact_prefix_source=cfg.lact_prefix_source,
                        detach_wm_mu_for_lact=cfg.four_stage3_detach_wm_mu_for_lact,
                        grad_accum_steps=cfg.grad_accum_steps,
                    )
                )
                if dp > 1:
                    pstep_accum_dp = jax.pmap(
                        functools.partial(
                            wm_train.train_step_stage2_grad_accum_dp,
                            model_def=model_def,
                            tx=tx,
                            trainable_filter=flt,
                            lambda_act=extra["lambda_act"],
                            lambda_cond=cfg.four_stage3_lambda_cond,
                            lact_prefix_source=cfg.lact_prefix_source,
                            detach_wm_mu_for_lact=cfg.four_stage3_detach_wm_mu_for_lact,
                            grad_accum_steps=cfg.grad_accum_steps,
                            axis_name=cfg.data_parallel_axis_name,
                        ),
                        axis_name=cfg.data_parallel_axis_name,
                        in_axes=(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
                    )
            else:
                pstep = _jit(
                    functools.partial(
                        wm_train.train_step_stage2,
                        trainable_filter=flt,
                        lambda_act=extra["lambda_act"],
                        lambda_cond=cfg.four_stage3_lambda_cond,
                        lact_prefix_source=cfg.lact_prefix_source,
                        detach_wm_mu_for_lact=cfg.four_stage3_detach_wm_mu_for_lact,
                    )
                )
        else:
            assert extra is not None
            pstep = _jit(
                functools.partial(
                    wm_train.train_step_stage3,
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
        loop_end = loop_start + n_steps if (stage_id == 3 and resume_stage3_extra_from_orbax) else n_steps
        it = range(loop_start, loop_end)
        pbar = tqdm.tqdm(it, dynamic_ncols=True, desc=f"4stg{stage_id}") if _is_tty else it

        for _ in pbar:
            if stage_id == 3 and cfg.grad_accum_steps > 1 and dp > 1:
                # Data-parallel + grad accumulation for stage3.
                obs_t_list = []
                obs_f_list = []
                wap_list = []
                wpm_list = []
                dt_list = []
                wah_list = []
                hv_list = []

                rng_master, rng_step = jax.random.split(rng_master)
                rng_devs = jax.random.split(rng_step, dp)

                for _micro in range(cfg.grad_accum_steps):
                    batch = _next_wm_batch()
                    obs_t, obs_f, _obs_td1, wm = wm_data.batch_to_observations_and_wm_tensors(batch)
                    obs_t = jax.tree.map(jnp.asarray, obs_t)
                    obs_f = jax.tree.map(jnp.asarray, obs_f)
                    wm_j = jax.tree.map(jnp.asarray, wm)

                    obs_t_list.append(obs_t)
                    obs_f_list.append(obs_f)
                    wap_list.append(wm_j["wm_action_prefix_pad"])
                    wpm_list.append(wm_j["wm_prefix_mask"])
                    dt_list.append(wm_j["wm_delta_t"])
                    wah_list.append(wm_j["wm_actions_handover"])
                    hv_list.append(wm_j["wm_handover_valid"])

                g = int(cfg.grad_accum_steps)

                def _stack_tree(xs):
                    return jax.tree.map(lambda *ys: jnp.stack(ys, axis=0), *xs)

                obs_t_stacked = _stack_tree(obs_t_list)  # (g, gb, ...)
                obs_f_stacked = _stack_tree(obs_f_list)
                wap_stacked = jnp.stack(wap_list, axis=0)
                wpm_stacked = jnp.stack(wpm_list, axis=0)
                dt_stacked = jnp.stack(dt_list, axis=0)
                wah_stacked = jnp.stack(wah_list, axis=0)
                hv_stacked = jnp.stack(hv_list, axis=0)

                def _to_dp(x):
                    # x: (g, gb, ...) -> (dp, g, per_dev_bs, ...)
                    rest = x.shape[2:]
                    y = x.reshape((g, dp, per_dev_bs) + rest)
                    axes = (1, 0, 2) + tuple(range(3, y.ndim))
                    return jnp.transpose(y, axes)

                obs_t_dp = jax.tree.map(_to_dp, obs_t_stacked)
                obs_f_dp = jax.tree.map(_to_dp, obs_f_stacked)
                wap_dp = _to_dp(wap_stacked)
                wpm_dp = _to_dp(wpm_stacked)
                dt_dp = _to_dp(dt_stacked)
                wah_dp = _to_dp(wah_stacked)
                hv_dp = _to_dp(hv_stacked)

                # Initialize dp array state on first use.
                if not hasattr(run_training, "_dp_state_box"):
                    pass
                # Store dp state in python locals to avoid changing outer logic.
                if "_dp_state" not in locals():
                    dp_step = jnp.asarray(train_state.step)
                    dp_params = train_state.params
                    dp_opt_state = train_state.opt_state
                    devices = jax.local_devices()[:dp]
                    dp_step = jax.device_put_replicated(dp_step, devices)
                    dp_params = jax.device_put_replicated(dp_params, devices)
                    dp_opt_state = jax.device_put_replicated(dp_opt_state, devices)

                assert pstep_accum_dp is not None
                dp_step, dp_params, dp_opt_state, info = pstep_accum_dp(
                    rng_devs,
                    dp_step,
                    dp_params,
                    dp_opt_state,
                    obs_t_dp,
                    obs_f_dp,
                    wap_dp,
                    wpm_dp,
                    dt_dp,
                    wah_dp,
                    hv_dp,
                )
                dp_step = jax.block_until_ready(dp_step)
                info = jax.device_get(info)
                info = jax.tree.map(lambda x: x[0], info)
                # Reconstruct single-device TrainState view for logging / checkpointing.
                train_state = training_utils.TrainState(
                    step=jax.tree.map(lambda x: x[0], dp_step),
                    params=jax.tree.map(lambda x: x[0], dp_params),
                    model_def=model_def,
                    opt_state=jax.tree.map(lambda x: x[0], dp_opt_state),
                    tx=tx,
                    ema_decay=None,
                    ema_params=None,
                )
            elif stage_id == 3 and cfg.grad_accum_steps > 1:
                obs_t_list = []
                obs_f_list = []
                wap_list = []
                wpm_list = []
                dt_list = []
                wah_list = []
                hv_list = []

                rng_master, rng_step = jax.random.split(rng_master)
                for _micro in range(cfg.grad_accum_steps):
                    batch = _next_wm_batch()
                    obs_t, obs_f, _obs_td1, wm = wm_data.batch_to_observations_and_wm_tensors(batch)
                    obs_t = jax.tree.map(jnp.asarray, obs_t)
                    obs_f = jax.tree.map(jnp.asarray, obs_f)
                    wm_j = jax.tree.map(jnp.asarray, wm)

                    obs_t_list.append(obs_t)
                    obs_f_list.append(obs_f)
                    wap_list.append(wm_j["wm_action_prefix_pad"])
                    wpm_list.append(wm_j["wm_prefix_mask"])
                    dt_list.append(wm_j["wm_delta_t"])
                    wah_list.append(wm_j["wm_actions_handover"])
                    hv_list.append(wm_j["wm_handover_valid"])

                assert pstep_accum is not None
                obs_t_stacked = jax.tree.map(lambda *xs: jnp.stack(xs, axis=0), *obs_t_list)
                obs_f_stacked = jax.tree.map(lambda *xs: jnp.stack(xs, axis=0), *obs_f_list)
                wap_stacked = jnp.stack(wap_list, axis=0)
                wpm_stacked = jnp.stack(wpm_list, axis=0)
                dt_stacked = jnp.stack(dt_list, axis=0)
                wah_stacked = jnp.stack(wah_list, axis=0)
                hv_stacked = jnp.stack(hv_list, axis=0)

                train_state, info = pstep_accum(
                    rng_step,
                    train_state,
                    obs_t_stacked,
                    obs_f_stacked,
                    wap_stacked,
                    wpm_stacked,
                    dt_stacked,
                    wah_stacked,
                    hv_stacked,
                )
                train_state = jax.block_until_ready(train_state)
                info = jax.device_get(info)
            else:
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
                        wm_j["wm_actions_handover"],
                        wm_j["wm_handover_valid"],
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
                    )
                elif stage_id == 3:
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
            if not (stage_id == 3 and cfg.grad_accum_steps > 1):
                info = jax.device_get(info)

            global_step += 1
            infos.append(info)
            if global_step % cfg.log_interval == 0:
                stacked = jax.tree.map(lambda *xs: np.mean(np.stack(xs)), *infos)
                logs = {f"4s{stage_id}/{k}": float(v) for k, v in stacked.items()}
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
                wm_train.append_wm_training_log(
                    cfg.training_log_file,
                    {
                        "event": "metrics",
                        "strategy": "four_stage",
                        "global_step": int(global_step),
                        "four_stage_id": int(stage_id),
                        "stage_tag": tag,
                        "log_interval": int(cfg.log_interval),
                        "metrics_mean": {k: float(v) for k, v in stacked.items()},
                        "wandb_prefixed": logs,
                    },
                )
                infos = []

            if (
                not cfg.disable_checkpoints
                and cfg.save_interval > 0
                and (global_step % cfg.save_interval == 0 or global_step == total_steps)
            ):
                wm_train._persist_training_checkpoint(
                    checkpoint_manager=checkpoint_manager,
                    train_state=train_state,
                    data_config=data_config,
                    ckpt_root=ckpt_root,
                    step=global_step,
                )

        if not cfg.disable_checkpoints and stage_id in (1, 2, 3, 4):
            gs = global_step
            loop_already_saved = cfg.save_interval > 0 and (
                gs % cfg.save_interval == 0 or (stage_id == 4 and gs == total_steps)
            )
            if not loop_already_saved:
                logger.info(
                    "Saving checkpoint at end of four-stage %d (global_step=%d, save_interval did not hit this step).",
                    stage_id,
                    gs,
                )
                wm_train._persist_training_checkpoint(
                    checkpoint_manager=checkpoint_manager,
                    train_state=train_state,
                    data_config=data_config,
                    ckpt_root=ckpt_root,
                    step=gs,
                )
            else:
                logger.info(
                    "Four-stage %d finished at global_step=%d; checkpoint already written in-loop (save_interval/total_steps).",
                    stage_id,
                    gs,
                )
            logger.info(
                "Four-stage %d checkpoint: Orbax step %s under %s | bundle params-only: %s | WM-only: %s",
                stage_id,
                gs,
                str((ckpt_root / str(gs)).resolve()),
                str((ckpt_root / f"bundle_params_step_{int(gs)}" / "params").resolve()),
                str((ckpt_root / f"world_model_step_{int(gs)}" / "params").resolve()),
            )
        elif cfg.disable_checkpoints and stage_id in (1, 2, 3, 4):
            logger.info(
                "Four-stage %d finished at global_step=%d (disable_checkpoints=True, no save).",
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
    logger.info("Done (four-stage). Checkpoints under %s", ckpt_root)


def cli() -> FourStageWorldModelTrainConfig:
    return tyro.cli(FourStageWorldModelTrainConfig)