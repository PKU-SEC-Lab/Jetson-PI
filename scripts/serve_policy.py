import dataclasses
import enum
import logging
import os
import pathlib
import socket
from typing import Any, Literal

import jax.numpy as jnp
import tyro

from openpi.models import model as _model
from openpi.models.pi0 import Pi0
from openpi.policies import pi0_async_inference_policy as _async_policy
from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.serving import websocket_policy_server
import openpi.shared.download as download
from openpi.training import checkpoints as _checkpoints
from openpi.training import config as _config
import openpi.transforms as transforms

try:
    from openpi.models.pi0_world_model import (
        ActionEncoderKind,
        Pi0WorldModelConfig,
        TokenReducerKind,
        load_pi0_future_world_model,
    )
except ImportError:
    Pi0WorldModelConfig = None
    load_pi0_future_world_model = None
    TokenReducerKind = str
    ActionEncoderKind = str


class EnvMode(enum.Enum):
    """Supported environments."""

    ALOHA = "aloha"
    ALOHA_SIM = "aloha_sim"
    DROID = "droid"
    LIBERO = "libero"


@dataclasses.dataclass
class Checkpoint:
    """Load a policy from a trained checkpoint."""

    # Training config name (e.g., "pi0_aloha_sim").
    config: str
    # Checkpoint directory (e.g., "checkpoints/pi0_aloha_sim/exp/10000").
    dir: str


@dataclasses.dataclass
class Default:
    """Use the default policy for the given environment."""


@dataclasses.dataclass
class Args:
    """Arguments for the serve_policy script."""

    # Environment to serve the policy for. This is only used when serving default policies.
    env: EnvMode = EnvMode.ALOHA_SIM

    # If provided, will be used in case the "prompt" key is not present in the data, or if the model doesn't have a default
    # prompt.
    default_prompt: str | None = None

    # Port to serve the policy on.
    port: int = 8000
    # Record the policy's behavior for debugging.
    record: bool = False

    # Specifies how to load the policy. If not provided, the default policy for the environment will be used.
    policy: Checkpoint | Default = dataclasses.field(default_factory=Default)

    enable_async_pi0_api: bool = True
    world_model_checkpoint: str | None = None
    world_model_token_reducer_kind: TokenReducerKind = "learned_cross_attn"
    world_model_action_encoder_kind: ActionEncoderKind = "gru"
    async_ae_proprio_source: Literal["prefix_t", "future_rollout", "vlash_last_action"] = "vlash_last_action"


# Default checkpoints that should be used for each environment.
DEFAULT_CHECKPOINT: dict[EnvMode, Checkpoint] = {
    EnvMode.ALOHA: Checkpoint(
        config="pi05_aloha",
        dir="gs://openpi-assets/checkpoints/pi05_base",
    ),
    EnvMode.ALOHA_SIM: Checkpoint(
        config="pi0_aloha_sim",
        dir="gs://openpi-assets/checkpoints/pi0_aloha_sim",
    ),
    EnvMode.DROID: Checkpoint(
        config="pi05_droid",
        dir="gs://openpi-assets/checkpoints/pi05_droid",
    ),
    EnvMode.LIBERO: Checkpoint(
        config="pi05_libero",
        dir="gs://openpi-assets/checkpoints/pi05_libero",
    ),
}


def _resolve_train_config_and_dir(args: Args) -> tuple[_config.TrainConfig, str]:
    match args.policy:
        case Checkpoint():
            return _config.get_config(args.policy.config), args.policy.dir
        case Default():
            d = DEFAULT_CHECKPOINT[args.env]
            return _config.get_config(d.config), d.dir


def _pi0_params_from_restored(raw: Any) -> Any:
    """WM four-stage checkpoints store ``{'pi0': ..., 'wm': ...}``; release Pi0 trees have ``PaliGemma`` at top level."""
    if isinstance(raw, dict) and "pi0" in raw and "PaliGemma" not in raw:
        return raw["pi0"]
    return raw


def _libero_norm_checkpoint_root() -> pathlib.Path:
    """Norm stats only: prefer local dirs to avoid ``maybe_download(gs://...)`` pulling the full ~12GiB bundle."""
    rel = pathlib.Path("assets") / "physical-intelligence" / "libero" / "norm_stats.json"
    for key in ("OPENPI_LIBERO_NORM_CHECKPOINT_DIR", "OPENPI_NORM_STATS_CHECKPOINT_DIR"):
        raw = os.environ.get(key)
        if raw:
            p = pathlib.Path(raw).expanduser().resolve()
            if (p / rel).is_file():
                logging.info("Norm-stats fallback: using %s=%s", key, p)
                return p
    for p in (
        pathlib.Path("PATH/TO/CHECKPOINT/pi05_libero"),
    ):
        if (p / rel).is_file():
            logging.info("Norm-stats fallback: using local Pi05 Libero tree %s", p)
            return p
    logging.warning(
        "Norm-stats fallback: no local pi05_libero with %s; downloading OpenPI default (large). "
        "Set OPENPI_LIBERO_NORM_CHECKPOINT_DIR to a local checkpoint root to skip.",
        rel,
    )
    return pathlib.Path(download.maybe_download(DEFAULT_CHECKPOINT[EnvMode.LIBERO].dir))


def create_policy(args: Args) -> _policy.Policy:
    """Create a policy from the given arguments (single restore path)."""
    train_cfg, raw_dir = _resolve_train_config_and_dir(args)
    checkpoint_dir = pathlib.Path(download.maybe_download(str(raw_dir)))

    # Preserve existing PI05 behavior: policy_config still handles native PyTorch checkpoints.
    # For JAX checkpoints, pre-load once and pass model to avoid duplicate restore.
    model = None
    norm_stats_for_policy: dict | None = None
    params_dir = checkpoint_dir / "params"
    if params_dir.exists():
        raw_params = _model.restore_params(params_dir, dtype=jnp.bfloat16)
        pi0_params = _pi0_params_from_restored(raw_params)
        if pi0_params is not raw_params:
            logging.info(
                "Bundled JAX checkpoint at %s; loading Pi0 weights from 'pi0' subtree (top-level keys: %s).",
                params_dir,
                list(raw_params.keys()) if isinstance(raw_params, dict) else type(raw_params),
            )
        model = train_cfg.model.load(pi0_params)
        data_config_pre = train_cfg.data.create(train_cfg.assets_dirs, train_cfg.model)
        if data_config_pre.asset_id is not None:
            try:
                norm_stats_for_policy = _checkpoints.load_norm_stats(
                    checkpoint_dir / "assets", data_config_pre.asset_id
                )
            except (FileNotFoundError, OSError, ValueError) as e:
                if args.env is not EnvMode.LIBERO:
                    raise
                logging.warning(
                    "Norm stats not found under %s (%s); loading LIBERO norm_stats from fallback checkpoint root.",
                    checkpoint_dir / "assets",
                    e,
                )
                norm_root = _libero_norm_checkpoint_root()
                norm_stats_for_policy = _checkpoints.load_norm_stats(
                    norm_root / "assets", data_config_pre.asset_id
                )

    policy = _policy_config.create_trained_policy(
        train_cfg,
        checkpoint_dir,
        default_prompt=args.default_prompt,
        model=model,
        norm_stats=norm_stats_for_policy,
    )

    if not args.enable_async_pi0_api:
        return policy
    if not isinstance(model, Pi0):
        return policy

    data_config = train_cfg.data.create(train_cfg.assets_dirs, train_cfg.model)
    norm_stats = norm_stats_for_policy
    if norm_stats is None and data_config.asset_id is not None:
        norm_stats = _checkpoints.load_norm_stats(checkpoint_dir / "assets", data_config.asset_id)
    action_norm = transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm) if norm_stats else None

    wm = None
    if args.world_model_checkpoint:
        if load_pi0_future_world_model is None or Pi0WorldModelConfig is None:
            logging.warning(
                "world_model_checkpoint is set but pi0_world_model module is unavailable in this build; async WM disabled."
            )
        else:
            wm_dir = download.maybe_download(args.world_model_checkpoint)
            wm_cfg = Pi0WorldModelConfig(
                proprio_dim=train_cfg.model.action_dim,
                action_dim=train_cfg.model.action_dim,
                token_reducer_kind=args.world_model_token_reducer_kind,
                action_encoder_kind=args.world_model_action_encoder_kind,
            )
            wm = load_pi0_future_world_model(wm_dir, config=wm_cfg)
            logging.info(
                "Loaded world model from %s (token_reducer_kind=%s, action_encoder_kind=%s)",
                wm_dir,
                args.world_model_token_reducer_kind,
                args.world_model_action_encoder_kind,
            )

    return _async_policy.Pi0AsyncInferencePolicy(
        policy,
        pi0=model,
        world_model=wm,
        action_norm=action_norm,
        model_action_dim=train_cfg.model.action_dim,
        ae_proprio_source=args.async_ae_proprio_source,
    )


def main(args: Args) -> None:
    policy = create_policy(args)
    policy_metadata = policy.metadata

    # Record the policy's behavior.
    if args.record:
        policy = _policy.PolicyRecorder(policy, "policy_records")

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)

    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=policy_metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
