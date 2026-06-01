# ruff: noqa: SLF001

from __future__ import annotations

import logging
import os
from typing import Any, Literal

import jax.numpy as jnp

from openpi.models import model as _model
from openpi.models.pi0 import Pi0
from openpi.models.pi0_world_model import Pi0FutureWorldModel

logger = logging.getLogger("openpi")

_VERIFY_ENV = "OPENPI_WM_INFERENCE_VERIFY"
_MIN_ABS_ENV = "OPENPI_WM_INFERENCE_VERIFY_MIN_ABS"


def wm_inference_verify_mode() -> Literal["off", "warn", "error", "log"]:
    v = (os.environ.get(_VERIFY_ENV) or "").strip().lower()
    if v in ("", "0", "false", "no", "off", "none"):
        return "off"
    if v in ("1", "true", "yes", "error", "strict"):
        return "error"
    if v in ("warn", "warning"):
        return "warn"
    if v in ("log", "2", "verbose", "debug"):
        return "log"
    return "off"


def _mu_min_abs_threshold() -> float:
    raw = (os.environ.get(_MIN_ABS_ENV) or "").strip()
    if not raw:
        return 0.0
    try:
        return float(raw)
    except ValueError:
        return 0.0


def verify_mu_shape_and_finite(*, mu: Any, world_model: Pi0FutureWorldModel) -> dict[str, Any]:
    if mu is None:
        return {"ok": False, "reason": "mu is None (missing world-model output)"}
    cfg = world_model.cfg
    expected_m = int(cfg.num_condition_tokens)
    expected_d = int(cfg.token_dim)
    got = tuple(int(x) for x in mu.shape)
    want = (int(mu.shape[0]), expected_m, expected_d)
    if got != want:
        return {"ok": False, "reason": f"mu.shape {got} != expected {want} (from WM cfg)"}
    if not bool(jnp.all(jnp.isfinite(mu))):
        return {"ok": False, "reason": "mu contains nan/inf"}
    max_abs = float(jnp.max(jnp.abs(jnp.asarray(mu, dtype=jnp.float32))))
    thr = _mu_min_abs_threshold()
    if thr <= 0.0:
        if not bool(jnp.any(mu != 0)):
            return {"ok": False, "reason": "mu is all zeros (exact)"}
    elif max_abs < thr:
        return {
            "ok": False,
            "reason": f"mu is near-zero: max|mu|={max_abs:g} < OPENPI_WM_INFERENCE_VERIFY_MIN_ABS={thr:g}",
        }
    return {"ok": True, "mu_shape": got, "mu_max_abs": max_abs}


def verify_suffix_token_delta_matches_mu(
    *,
    pi0: Pi0,
    observation: _model.Observation,
    mu: Any,
) -> dict[str, Any]:
    obs = _model.preprocess_observation(None, observation, train=False)
    b = int(obs.state.shape[0])
    ah = int(pi0.action_horizon)
    ad = int(pi0.action_dim)
    x_t = jnp.zeros((b, ah, ad), dtype=jnp.float32)
    time = jnp.full((b,), 0.5, dtype=jnp.float32)

    st0, _, _, _ = pi0.embed_suffix(obs, x_t, time, future_condition_tokens=None)
    st1, _, _, _ = pi0.embed_suffix(obs, x_t, time, future_condition_tokens=mu)
    len0 = int(st0.shape[1])
    len1 = int(st1.shape[1])
    m = int(mu.shape[1])
    delta = len1 - len0
    if delta != m:
        return {
            "ok": False,
            "reason": f"suffix token delta {len1}-{len0}={delta} != mu.shape[1]={m}",
            "suffix_len_no_mu": len0,
            "suffix_len_with_mu": len1,
        }
    return {
        "ok": True,
        "suffix_len_no_mu": len0,
        "suffix_len_with_mu": len1,
        "mu_token_slots": m,
    }


def run_wm_inference_verification(
    *,
    pi0: Pi0,
    world_model: Pi0FutureWorldModel,
    observation: _model.Observation,
    mu: Any,
) -> None:
    mode = wm_inference_verify_mode()
    if mode == "off":
        return

    r_mu = verify_mu_shape_and_finite(mu=mu, world_model=world_model)
    if not r_mu.get("ok"):
        msg = f"WM μ check failed: {r_mu}"
        if mode == "error":
            raise RuntimeError(msg)
        logger.warning("%s", msg)
        return

    r_suf = verify_suffix_token_delta_matches_mu(pi0=pi0, observation=observation, mu=mu)
    if not r_suf.get("ok"):
        msg = f"WM AE suffix check failed: {r_suf}"
        if mode == "error":
            raise RuntimeError(msg)
        logger.warning("%s", msg)
        return

    if mode == "log":
        logger.info(
            "WM inference verify OK: mu_shape=%s mu_max_abs=%s suffix_no_mu=%s suffix_with_mu=%s (delta=%s)",
            r_mu.get("mu_shape"),
            r_mu.get("mu_max_abs"),
            r_suf.get("suffix_len_no_mu"),
            r_suf.get("suffix_len_with_mu"),
            int(r_suf.get("suffix_len_with_mu", 0)) - int(r_suf.get("suffix_len_no_mu", 0)),
        )
