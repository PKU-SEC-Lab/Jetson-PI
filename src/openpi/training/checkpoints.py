from __future__ import annotations

import asyncio
import concurrent.futures as futures
import dataclasses
import gc
import logging
from typing import Any, Protocol

from etils import epath
import flax.nnx as nnx
import jax
import orbax.checkpoint as ocp
from orbax.checkpoint import checkpoint_utils as ocp_checkpoint_utils
from orbax.checkpoint._src.serialization import type_handlers as ocp_type_handlers
import orbax.checkpoint.future as future

from openpi.shared import array_typing as at
from openpi.shared import nnx_utils
import openpi.shared.normalize as _normalize
import openpi.training.data_loader as _data_loader
import openpi.training.utils as training_utils

logger = logging.getLogger(__name__)

# World-model Orbax checkpoints written by ``save_state_wm_train`` store only the ``wm/`` subtree under
# the ``params`` item; Pi0 (LLM / AE) weights are not duplicated on disk. ``restore_state_wm_train`` uses
# this manifest to merge WM weights back into the in-memory Pi0+WM bundle loaded from ``pi0_checkpoint``.
_WM_ORBAX_PARAMS_MANIFEST = "wm_params_only.json"


def _wm_subtree_param_paths_filter() -> Any:
    return nnx_utils.PathRegex(r"wm/.*")


def _merge_wm_checkpoint_params_into_state(
    *,
    model_def: nnx.GraphDef[Any],
    full_params_template: at.Params,
    wm_params_from_disk: at.Params,
) -> at.Params:
    bundle = nnx.merge(model_def, full_params_template)
    nnx.update(bundle, wm_params_from_disk)
    return nnx.state(bundle)


def _orbax_wm_params_manifest_path(checkpoint_manager: ocp.CheckpointManager, step: int) -> epath.Path:
    return epath.Path(str(checkpoint_manager.directory)).resolve() / str(int(step)) / _WM_ORBAX_PARAMS_MANIFEST


def _orbax_checkpoint_has_wm_only_params(checkpoint_manager: ocp.CheckpointManager, step: int) -> bool:
    return _orbax_wm_params_manifest_path(checkpoint_manager, step).exists()


def _reshard_tree_for_restore(tree: at.PyTree) -> at.PyTree:
    devices = jax.local_devices()
    if not devices:
        raise RuntimeError(
            "No JAX local devices for checkpoint restore. Check CUDA_VISIBLE_DEVICES / JAX_PLATFORM."
        )
    sd = jax.sharding.SingleDeviceSharding(devices[0])

    def _leaf_sharding(x: object):
        # Allow restoring into abstract templates (e.g. jax.ShapeDtypeStruct) to avoid
        # allocating a fresh opt_state on device before Orbax restore. This significantly
        # reduces peak GPU memory during resume.
        return sd if isinstance(x, (jax.Array, jax.ShapeDtypeStruct)) else None

    return jax.tree.map(_leaf_sharding, tree)


def _strip_prng_key_dtype_from_restore_args(restore_args_tree: at.PyTree) -> at.PyTree:
    if restore_args_tree is None:
        return restore_args_tree

    def _fix_leaf(arg: object) -> object:
        if isinstance(arg, ocp_type_handlers.ArrayRestoreArgs):
            dt = arg.dtype
            if dt is not None:
                try:
                    if jax.dtypes.issubdtype(dt, jax.dtypes.prng_key):
                        return dataclasses.replace(arg, dtype=None)
                except (TypeError, ValueError):
                    if str(dt) == "key<fry>" or "key<" in str(dt):
                        return dataclasses.replace(arg, dtype=None)
        return arg

    return jax.tree.map(_fix_leaf, restore_args_tree)


def _array_restore_args_set_non_strict(restore_args_tree: at.PyTree) -> at.PyTree:
    if restore_args_tree is None:
        return restore_args_tree

    def _one(arg: object) -> object:
        if isinstance(arg, ocp_type_handlers.ArrayRestoreArgs) and getattr(arg, "strict", True):
            return dataclasses.replace(arg, strict=False)
        return arg

    return jax.tree.map(_one, restore_args_tree)


def initialize_checkpoint_dir(
    checkpoint_dir: epath.Path | str,
    *,
    keep_period: int | None,
    overwrite: bool,
    resume: bool,
    max_to_keep: int | None = 1,
) -> tuple[ocp.CheckpointManager, bool]:
    checkpoint_dir = epath.Path(checkpoint_dir).resolve()
    resuming = False
    if checkpoint_dir.exists():
        if overwrite:
            checkpoint_dir.rmtree()
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            logging.info(f"Wiped checkpoint directory {checkpoint_dir}")
        elif resume:
            resuming = True
        else:
            # Allow restarting into an *empty* experiment directory without requiring `--overwrite`
            # (which would delete any accidentally-present checkpoints) and without `--resume`
            # (which would be misleading when there is nothing to restore yet).
            try:
                has_any = next(checkpoint_dir.iterdir(), None) is not None
            except FileNotFoundError:
                has_any = False
            if has_any:
                raise FileExistsError(
                    f"Checkpoint directory {checkpoint_dir} already exists and is non-empty. "
                    "Use --resume to continue an existing run, or --overwrite to wipe it."
                )
            logging.info(
                "Checkpoint directory %s exists but is empty; continuing without --overwrite/--resume.",
                checkpoint_dir,
            )

    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    mngr = ocp.CheckpointManager(
        checkpoint_dir,
        item_handlers={
            "assets": CallbackHandler(),
            "train_state": ocp.PyTreeCheckpointHandler(),
            "params": ocp.PyTreeCheckpointHandler(),
        },
        options=ocp.CheckpointManagerOptions(
            max_to_keep=max_to_keep,
            keep_period=keep_period,
            create=False,
            enable_async_checkpointing=False,
            cleanup_tmp_directories=True,
            async_options=ocp.AsyncOptions(timeout_secs=7200),
        ),
    )

    # Special case: the checkpoint directory exists and the user requests to resume training, but the training run did
    # not get to the first checkpoint saved. In this case, we don't actually want the train script to try and restore a
    # checkpoint, since it will fail.
    if resuming and tuple(mngr.all_steps()) in [(), (0,)]:
        logging.info("Checkpoint directory exists, but does not contain any checkpoints. Aborting resume.")
        resuming = False

    return mngr, resuming


def resolve_resume_step(
    checkpoint_manager: ocp.CheckpointManager, resume_checkpoint_step: int | None
) -> int | None:
    if resume_checkpoint_step is None:
        return checkpoint_manager.latest_step()
    saved = tuple(checkpoint_manager.all_steps())
    if resume_checkpoint_step not in saved:
        raise ValueError(
            f"resume_checkpoint_step={resume_checkpoint_step} not found in Orbax checkpoints {saved}. "
            "Pick an existing step directory under the experiment checkpoint folder."
        )
    return resume_checkpoint_step


def _ensure_orbax_metrics_file(checkpoint_root: epath.Path | str, step: int) -> None:
    """Orbax scans ``{step}/metrics/metrics`` when listing checkpoints; saves without ``best_fn`` omit that file."""
    root = epath.Path(checkpoint_root)
    path = root / str(step) / "metrics" / "metrics"
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{}\n")


def save_state(
    checkpoint_manager: ocp.CheckpointManager,
    state: training_utils.TrainState,
    data_loader: _data_loader.DataLoader,
    step: int,
):
    def save_assets(directory: epath.Path):
        # Save the normalization stats.
        data_config = data_loader.data_config()
        norm_stats = data_config.norm_stats
        if norm_stats is not None and data_config.asset_id is not None:
            _normalize.save(directory / data_config.asset_id, norm_stats)

    # Split params that can be used for inference into a separate item.
    with at.disable_typechecking():
        train_state, params = _split_params(state)
    gc.collect()
    items = {
        "assets": save_assets,
        "train_state": train_state,
        "params": {"params": params},
    }
    checkpoint_manager.save(step, items)
    checkpoint_manager.wait_until_finished()
    _ensure_orbax_metrics_file(checkpoint_manager.directory, step)
    step_dir = epath.Path(str(checkpoint_manager.directory)).resolve() / str(step)
    logger.info("Saved checkpoint (Orbax train_state+params): %s (step=%d)", step_dir, step)


def save_state_wm_train(
    checkpoint_manager: ocp.CheckpointManager,
    state: training_utils.TrainState,
    data_loader: _data_loader.DataLoader,
    step: int,
):
    """Like ``save_state``, but the Orbax ``params`` item stores only the ``wm/`` subtree (no Pi0 / LLM / AE).

    Writes ``{step}/wm_params_only.json`` so ``restore_state_wm_train`` / ``restore_params_only_wm_train`` can
    reload and merge WM weights into the bundle built from ``--pi0-checkpoint``.
    """

    def save_assets(directory: epath.Path):
        data_config = data_loader.data_config()
        norm_stats = data_config.norm_stats
        if norm_stats is not None and data_config.asset_id is not None:
            _normalize.save(directory / data_config.asset_id, norm_stats)

    with at.disable_typechecking():
        train_state, full_params = _split_params(state)
        params_wm = full_params.filter(_wm_subtree_param_paths_filter())
    gc.collect()
    items = {
        "assets": save_assets,
        "train_state": train_state,
        "params": {"params": params_wm},
    }
    checkpoint_manager.save(step, items)
    checkpoint_manager.wait_until_finished()
    _ensure_orbax_metrics_file(checkpoint_manager.directory, step)
    manifest = _orbax_wm_params_manifest_path(checkpoint_manager, int(step))
    manifest.write_text('{"version":1,"params":"wm_subtree_only"}\n')
    step_dir = epath.Path(str(checkpoint_manager.directory)).resolve() / str(step)
    logger.info(
        "Saved world-model checkpoint (Orbax train_state + wm-only params, no Pi0 duplicate): %s (step=%d)",
        step_dir,
        step,
    )


def restore_state(
    checkpoint_manager: ocp.CheckpointManager,
    state: training_utils.TrainState,
    data_loader: _data_loader.DataLoader,
    step: int | None = None,
) -> training_utils.TrainState:
    del data_loader

    with at.disable_typechecking():
        # Split params that can be used for inference into a separate item.
        train_state, params = _split_params(state)
        params_item = {"params": params}
        restore_args_ts = _array_restore_args_set_non_strict(
            _strip_prng_key_dtype_from_restore_args(
                ocp_checkpoint_utils.construct_restore_args(
                    train_state,
                    sharding_tree=_reshard_tree_for_restore(train_state),
                )
            )
        )
        restore_args_params = {
            "params": _array_restore_args_set_non_strict(
                _strip_prng_key_dtype_from_restore_args(
                    ocp_checkpoint_utils.construct_restore_args(
                        params,
                        sharding_tree=_reshard_tree_for_restore(params),
                    )
                )
            )
        }
        restored = checkpoint_manager.restore(
            step,
            items={
                "train_state": train_state,
                "params": params_item,
            },
            restore_kwargs={
                "train_state": {"restore_args": restore_args_ts},
                "params": {"restore_args": restore_args_params},
            },
        )
    return _merge_params(restored["train_state"], restored["params"])


def _resolve_wm_restore_step(checkpoint_manager: ocp.CheckpointManager, step: int | None) -> int:
    if step is not None:
        return int(step)
    latest = checkpoint_manager.latest_step()
    if latest is None:
        raise ValueError(f"No Orbax checkpoint found under {checkpoint_manager.directory!r}.")
    return int(latest)


def restore_state_wm_train(
    checkpoint_manager: ocp.CheckpointManager,
    state: training_utils.TrainState,
    data_loader: _data_loader.DataLoader,
    step: int | None = None,
) -> training_utils.TrainState:
    """Restore world-model training checkpoints (``save_state_wm_train`` or legacy full ``save_state``)."""
    resolved = _resolve_wm_restore_step(checkpoint_manager, step)
    if not _orbax_checkpoint_has_wm_only_params(checkpoint_manager, resolved):
        return restore_state(checkpoint_manager, state, data_loader, step=step)

    with at.disable_typechecking():
        train_state, template_full = _split_params(state)
        params_wm_template = template_full.filter(_wm_subtree_param_paths_filter())
        params_item = {"params": params_wm_template}
        restore_args_ts = _array_restore_args_set_non_strict(
            _strip_prng_key_dtype_from_restore_args(
                ocp_checkpoint_utils.construct_restore_args(
                    train_state,
                    sharding_tree=_reshard_tree_for_restore(train_state),
                )
            )
        )
        restore_args_params = {
            "params": _array_restore_args_set_non_strict(
                _strip_prng_key_dtype_from_restore_args(
                    ocp_checkpoint_utils.construct_restore_args(
                        params_wm_template,
                        sharding_tree=_reshard_tree_for_restore(params_wm_template),
                    )
                )
            )
        }
        restored = checkpoint_manager.restore(
            resolved,
            items={
                "train_state": train_state,
                "params": params_item,
            },
            restore_kwargs={
                "train_state": {"restore_args": restore_args_ts},
                "params": {"restore_args": restore_args_params},
            },
        )
    merged_full = _merge_wm_checkpoint_params_into_state(
        model_def=state.model_def,
        full_params_template=state.params,
        wm_params_from_disk=restored["params"]["params"],
    )
    return _merge_params(restored["train_state"], {"params": merged_full})


def restore_params_only(
    checkpoint_manager: ocp.CheckpointManager,
    state: training_utils.TrainState,
    step: int | None = None,
) -> training_utils.TrainState:
    """Restore only params from Orbax and keep current optimizer state."""
    with at.disable_typechecking():
        _, params = _split_params(state)
        params_item = {"params": params}
        restore_args_params = {
            "params": _array_restore_args_set_non_strict(
                _strip_prng_key_dtype_from_restore_args(
                    ocp_checkpoint_utils.construct_restore_args(
                        params,
                        sharding_tree=_reshard_tree_for_restore(params),
                    )
                )
            )
        }
        restored_params = checkpoint_manager.restore(
            step,
            items={"params": params_item},
            restore_kwargs={
                "params": {"restore_args": restore_args_params},
            },
        )
    return dataclasses.replace(state, params=restored_params["params"]["params"])


def restore_params_only_wm_train(
    checkpoint_manager: ocp.CheckpointManager,
    state: training_utils.TrainState,
    step: int | None = None,
) -> training_utils.TrainState:
    """Like ``restore_params_only``, but merges ``wm/`` weights when the checkpoint was saved with ``save_state_wm_train``."""
    resolved = _resolve_wm_restore_step(checkpoint_manager, step)
    if not _orbax_checkpoint_has_wm_only_params(checkpoint_manager, resolved):
        return restore_params_only(checkpoint_manager, state, step=step)

    with at.disable_typechecking():
        _, template_full = _split_params(state)
        params_wm_template = template_full.filter(_wm_subtree_param_paths_filter())
        params_item = {"params": params_wm_template}
        restore_args_params = {
            "params": _array_restore_args_set_non_strict(
                _strip_prng_key_dtype_from_restore_args(
                    ocp_checkpoint_utils.construct_restore_args(
                        params_wm_template,
                        sharding_tree=_reshard_tree_for_restore(params_wm_template),
                    )
                )
            )
        }
        restored_params = checkpoint_manager.restore(
            resolved,
            items={"params": params_item},
            restore_kwargs={
                "params": {"restore_args": restore_args_params},
            },
        )
    merged = _merge_wm_checkpoint_params_into_state(
        model_def=state.model_def,
        full_params_template=state.params,
        wm_params_from_disk=restored_params["params"]["params"],
    )
    return dataclasses.replace(state, params=merged)


def load_norm_stats(assets_dir: epath.Path | str, asset_id: str) -> dict[str, _normalize.NormStats] | None:
    norm_stats_dir = epath.Path(assets_dir) / asset_id
    norm_stats = _normalize.load(norm_stats_dir)
    logging.info(f"Loaded norm stats from {norm_stats_dir}")
    return norm_stats


class Callback(Protocol):
    def __call__(self, directory: epath.Path) -> None: ...


class CallbackHandler(ocp.AsyncCheckpointHandler):
    """A CheckpointHandler for calling an arbitrary function asynchronously. Only for saving, not for restoring."""

    def save(self, directory: epath.Path, args: CallbackSave):
        if jax.process_index() == 0:
            args.callback(directory)

    async def async_save(self, directory: epath.Path, args: CallbackSave) -> list[futures.Future]:
        return [future.CommitFutureAwaitingContractedSignals(asyncio.to_thread(self.save, directory, args))]

    def restore(self, *args, **kwargs):
        raise NotImplementedError("CallbackHandler does not support restore")


@ocp.args.register_with_handler(CallbackHandler, for_save=True)
@dataclasses.dataclass
class CallbackSave(ocp.args.CheckpointArgs):
    callback: Callback


@ocp.args.register_with_handler(CallbackHandler, for_restore=True)
class CallbackRestore(ocp.args.CheckpointArgs): ...


def _split_params(state: training_utils.TrainState) -> tuple[training_utils.TrainState, at.Params]:
    if state.ema_params is not None:
        params = state.ema_params
        train_state = dataclasses.replace(state, ema_params=None)
    else:
        params = state.params
        train_state = dataclasses.replace(state, params={})
    return train_state, params


def _merge_params(train_state: training_utils.TrainState, params: dict[str, at.Params]) -> training_utils.TrainState:
    # Revert the logic inside `_split_params`. Assumes that existence of `params` means that EMA params were used during the split.
    if train_state.params:
        return dataclasses.replace(train_state, ema_params=params["params"])
    return dataclasses.replace(train_state, params=params["params"])
