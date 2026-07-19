"""Stage, preflight, and run the gated Jetson-PI Task-C C1 experiment."""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import json
import os
import pathlib
import signal
import socket
import subprocess
import time
from typing import Any
import urllib.request

from openpi_client import task_c_trace

from scripts import task_c_analysis

MODEL_ID = "zebinyang/Jetson-PI-pi05"
MODEL_REVISION = "a3a803da176b10ab87dc5e29720d47c772848b43"
MODEL_API = (
    f"https://modelscope.cn/api/v1/models/zebinyang/Jetson-PI-pi05/repo/files?Revision={MODEL_REVISION}&Recursive=true"
)
CONDITIONS = {
    "faac_only": None,
    "kappa_0p2": 0.2,
    "kappa_0p4": 0.4,
    "kappa_0p8": 0.8,
}
SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")


def _now() -> str:
    return dt.datetime.now(tz=dt.timezone.utc).isoformat()  # noqa: UP017 -- pyright bundled here lacks dt.UTC


def _run(
    command: list[str],
    *,
    cwd: pathlib.Path,
    env: dict[str, str] | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=cwd,
        env=env,
        check=check,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )


def _repo_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[1]


def _checkpoint_files(checkpoint: pathlib.Path) -> list[pathlib.Path]:
    return sorted(path for path in checkpoint.rglob("*") if path.is_file() and ".git" not in path.parts)


def checkpoint_manifest(checkpoint: pathlib.Path, artifact_root: pathlib.Path) -> dict[str, Any]:
    checkpoint = checkpoint.resolve()
    artifact_root = artifact_root.resolve()
    head = _run(["git", "rev-parse", "HEAD"], cwd=checkpoint).stdout.strip()
    if head != MODEL_REVISION:
        raise task_c_trace.TaskCTraceError(f"checkpoint revision {head} != pinned {MODEL_REVISION}")

    with urllib.request.urlopen(MODEL_API, timeout=60) as response:
        api_payload = json.load(response)
    if not api_payload.get("Success") or int(api_payload.get("Code", 0)) != 200:
        raise task_c_trace.TaskCTraceError(f"ModelScope file API failed: {api_payload}")
    api_path = artifact_root / "provenance" / "modelscope_files_api.json"
    task_c_trace.write_json_atomic(api_path, api_payload)
    api_files = {str(item["Path"]): item for item in api_payload["Data"]["Files"] if item.get("Type") != "tree"}

    lfs_output = _run(["git", "lfs", "ls-files", "--long"], cwd=checkpoint).stdout
    lfs_oids: dict[str, str] = {}
    for line in lfs_output.splitlines():
        if not line.strip():
            continue
        oid, marker, path = line.split(maxsplit=2)
        if marker not in {"*", "-"}:
            raise task_c_trace.TaskCTraceError(f"unexpected git-lfs marker: {line}")
        lfs_oids[path] = oid

    file_receipts: list[dict[str, Any]] = []
    for path in _checkpoint_files(checkpoint):
        relative = path.relative_to(checkpoint).as_posix()
        sha = task_c_trace.sha256_file(path)
        size = path.stat().st_size
        if relative in lfs_oids and sha != lfs_oids[relative]:
            raise task_c_trace.TaskCTraceError(f"git-lfs SHA mismatch for {relative}: {sha} != {lfs_oids[relative]}")
        api_item = api_files.get(relative)
        if api_item is None:
            raise task_c_trace.TaskCTraceError(f"local checkpoint file missing from pinned ModelScope API: {relative}")
        api_sha = str(api_item.get("Sha256") or "")
        if api_sha and sha != api_sha:
            raise task_c_trace.TaskCTraceError(f"ModelScope SHA mismatch for {relative}: {sha} != {api_sha}")
        if int(api_item["Size"]) != size:
            raise task_c_trace.TaskCTraceError(f"ModelScope size mismatch for {relative}: {size} != {api_item['Size']}")
        file_receipts.append(
            {
                "path": relative,
                "size": size,
                "sha256": sha,
                "git_lfs": relative in lfs_oids,
                "modelscope_last_file_revision": api_item["Revision"],
            }
        )
    if set(api_files) != {item["path"] for item in file_receipts}:
        missing = sorted(set(api_files).difference(item["path"] for item in file_receipts))
        raise task_c_trace.TaskCTraceError(f"pinned ModelScope files missing locally: {missing}")
    manifest = {
        "schema_version": task_c_trace.SCHEMA_VERSION,
        "model_id": MODEL_ID,
        "resolved_snapshot_revision": head,
        "remote_head_observed_before_download": "95207c5790aacbb3722b009111bbca1578993b3b",
        "modelscope_files_api": MODEL_API,
        "modelscope_files_api_sha256": task_c_trace.sha256_file(api_path),
        "checkpoint_root": str(checkpoint),
        "total_files": len(file_receipts),
        "total_bytes": sum(item["size"] for item in file_receipts),
        "files": file_receipts,
    }
    manifest_path = artifact_root / "provenance" / "checkpoint_manifest.json"
    task_c_trace.write_json_atomic(manifest_path, manifest)
    sums = artifact_root / "provenance" / "CHECKPOINT_SHA256SUMS"
    sums.write_text(
        "".join(f"{item['sha256']}  {item['path']}\n" for item in file_receipts),
        encoding="utf-8",
    )
    result = {
        **manifest,
        "manifest_path": str(manifest_path),
        "manifest_sha256": task_c_trace.sha256_file(manifest_path),
        "sha256s_path": str(sums),
        "sha256s_sha256": task_c_trace.sha256_file(sums),
    }
    print(json.dumps(result, sort_keys=True))
    return result


def verify_checkpoint_manifest(checkpoint: pathlib.Path, manifest_path: pathlib.Path) -> dict[str, Any]:
    """Re-hash the pinned snapshot before every model-bearing operation."""

    checkpoint = checkpoint.resolve()
    manifest_path = manifest_path.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("model_id") != MODEL_ID or manifest.get("resolved_snapshot_revision") != MODEL_REVISION:
        raise task_c_trace.TaskCTraceError("checkpoint manifest model identity or revision is not the pinned target")
    head = _run(["git", "rev-parse", "HEAD"], cwd=checkpoint).stdout.strip()
    if head != MODEL_REVISION:
        raise task_c_trace.TaskCTraceError(f"checkpoint revision {head} != pinned {MODEL_REVISION}")
    expected = {str(item["path"]): item for item in manifest["files"]}
    actual = {path.relative_to(checkpoint).as_posix(): path for path in _checkpoint_files(checkpoint)}
    if set(actual) != set(expected):
        raise task_c_trace.TaskCTraceError(
            f"checkpoint file set changed: missing={sorted(set(expected) - set(actual))}, "
            f"extra={sorted(set(actual) - set(expected))}"
        )
    verified_bytes = 0
    for relative, path in actual.items():
        receipt = expected[relative]
        size = path.stat().st_size
        if size != int(receipt["size"]):
            raise task_c_trace.TaskCTraceError(f"checkpoint size changed for {relative}")
        sha = task_c_trace.sha256_file(path)
        if sha != str(receipt["sha256"]):
            raise task_c_trace.TaskCTraceError(f"checkpoint SHA-256 changed for {relative}: {sha}")
        verified_bytes += size
    return {
        "manifest": str(manifest_path),
        "manifest_sha256": task_c_trace.sha256_file(manifest_path),
        "revision": head,
        "verified_files": len(actual),
        "verified_bytes": verified_bytes,
    }


def _base_env(repo: pathlib.Path) -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONNOUSERSITE"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    prefixes = [str(repo / "packages" / "openpi-client" / "src"), str(repo / "src"), str(repo)]
    if env.get("PYTHONPATH"):
        prefixes.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(prefixes)
    return env


def preflight(
    *,
    repo: pathlib.Path,
    policy_python: pathlib.Path,
    checkpoint: pathlib.Path,
    wm_checkpoint: pathlib.Path,
    checkpoint_manifest_path: pathlib.Path,
    output: pathlib.Path,
    cuda_device: str,
) -> dict[str, Any]:
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_verification = verify_checkpoint_manifest(checkpoint.parent, checkpoint_manifest_path)
    preflight_script = """
import json
import jax
import jax.numpy as jnp
import numpy as np
from scripts import serve_policy

args = serve_policy.Args(
    env=serve_policy.EnvMode.LIBERO,
    policy=serve_policy.Checkpoint(config="pi05_libero", dir=__PI_CHECKPOINT__),
    world_model_checkpoint=__WM_CHECKPOINT__,
    world_model_token_reducer_kind="learned_cross_attn",
    world_model_action_encoder_kind="transformer_block",
    async_ae_proprio_source="prefix_t",
)
policy = serve_policy.create_policy(args)
assert policy._pi0.action_dim == 32
assert policy._pi0.action_horizon == 10
assert policy._world_model is not None
cfg = policy._world_model.cfg
assert cfg.num_condition_tokens == 4
assert cfg.token_dim == 1024
assert cfg.vlm_hidden_dim == 2048
h = jnp.zeros((1, 16, cfg.vlm_hidden_dim), dtype=jnp.bfloat16)
q = jnp.zeros((1, cfg.proprio_dim), dtype=jnp.float32)
a = jnp.zeros((1, 1, cfg.action_dim), dtype=jnp.float32)
m = jnp.ones((1, 1), dtype=jnp.bool_)
d = jnp.ones((1,), dtype=jnp.float32)
mu, kappa = policy._wm_forward(h, q, a, m, d, jax.random.key(0))
jax.block_until_ready((mu, kappa))
mu_np = np.asarray(mu)
kappa_np = np.asarray(kappa)
assert mu_np.shape == (1, 4, 1024)
assert kappa_np.shape == (1,)
assert np.isfinite(mu_np).all() and np.isfinite(kappa_np).all()
print(json.dumps({
    "jax_version": jax.__version__,
    "backend": jax.default_backend(),
    "devices": [str(device) for device in jax.devices()],
    "device_kinds": [device.device_kind for device in jax.devices()],
    "policy_action_dim": policy._pi0.action_dim,
    "policy_action_horizon": policy._pi0.action_horizon,
    "mu_shape": list(mu_np.shape),
    "kappa_shape": list(kappa_np.shape),
    "mu_finite": True,
    "kappa_finite": True,
    "wm_config": {
        "vlm_hidden_dim": cfg.vlm_hidden_dim,
        "token_dim": cfg.token_dim,
        "num_condition_tokens": cfg.num_condition_tokens,
        "token_reducer_kind": cfg.token_reducer_kind,
        "action_encoder_kind": cfg.action_encoder_kind,
    },
}, sort_keys=True))
""".replace("__PI_CHECKPOINT__", repr(str(checkpoint))).replace("__WM_CHECKPOINT__", repr(str(wm_checkpoint)))
    env = _base_env(repo)
    env["CUDA_VISIBLE_DEVICES"] = cuda_device
    completed = _run([str(policy_python), "-c", preflight_script], cwd=repo, env=env)
    (output / "preflight.log").write_text(completed.stdout, encoding="utf-8")
    json_lines = [line for line in completed.stdout.splitlines() if line.startswith("{")]
    if not json_lines:
        raise task_c_trace.TaskCTraceError("preflight emitted no JSON receipt")
    receipt = json.loads(json_lines[-1])
    if receipt["backend"] != "gpu":
        raise task_c_trace.TaskCTraceError(f"preflight backend is not GPU: {receipt['backend']}")
    if receipt["jax_version"] != "0.5.3":
        raise task_c_trace.TaskCTraceError(f"preflight JAX version is not 0.5.3: {receipt['jax_version']}")
    if not receipt["device_kinds"] or any("H100" not in kind for kind in receipt["device_kinds"]):
        raise task_c_trace.TaskCTraceError(f"preflight did not run on H100: {receipt['device_kinds']}")
    receipt["checkpoint_verification"] = checkpoint_verification
    task_c_trace.write_json_atomic(output / "preflight.json", receipt)
    print(json.dumps(receipt, sort_keys=True))
    return receipt


def _find_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_server(process: subprocess.Popen[Any], port: int, *, timeout: float = 1800.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise task_c_trace.TaskCTraceError(f"policy server exited before readiness (rc={process.returncode})")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/healthz", timeout=2) as response:
                if int(getattr(response, "status", 200)) == 200:
                    return
        except Exception:
            pass
        time.sleep(1)
    raise task_c_trace.TaskCTraceError(f"timed out waiting for policy server on port {port}")


def _stop_process(process: subprocess.Popen[Any]) -> None:
    if process.poll() is not None:
        return
    process.send_signal(signal.SIGTERM)
    try:
        process.wait(timeout=60)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=15)


def _git_receipt(repo: pathlib.Path) -> dict[str, Any]:
    head = _run(["git", "rev-parse", "HEAD"], cwd=repo).stdout.strip()
    diff = _run(["git", "diff", "--binary", "HEAD"], cwd=repo).stdout.encode()
    status = _run(["git", "status", "--short"], cwd=repo).stdout.splitlines()
    return {
        "head": head,
        "diff_sha256": task_c_trace.sha256_bytes(diff),
        "dirty_paths": status,
    }


def _device_receipt(cuda_device: str) -> dict[str, Any]:
    query = _run(
        [
            "nvidia-smi",
            f"--id={cuda_device}",
            "--query-gpu=name,uuid,driver_version,memory.total,memory.free,compute_cap",
            "--format=csv,noheader,nounits",
        ],
        cwd=_repo_root(),
    ).stdout.strip()
    values = [value.strip() for value in query.split(",")]
    if len(values) != 6:
        raise task_c_trace.TaskCTraceError(f"unexpected nvidia-smi receipt: {query!r}")
    return {
        "name": values[0],
        "uuid": values[1],
        "driver": values[2],
        "memory_total_mib": int(values[3]),
        "memory_free_mib_before_load": int(values[4]),
        "compute_capability": values[5],
        "cuda_visible_devices": cuda_device,
    }


def _freeze(python: pathlib.Path, output: pathlib.Path) -> dict[str, Any]:
    completed = _run(["uv", "pip", "freeze", "--python", str(python)], cwd=_repo_root())
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(completed.stdout, encoding="utf-8")
    return {"path": str(output), "sha256": task_c_trace.sha256_file(output)}


def run_condition(
    *,
    repo: pathlib.Path,
    policy_python: pathlib.Path,
    libero_python: pathlib.Path,
    checkpoint: pathlib.Path,
    wm_checkpoint: pathlib.Path,
    checkpoint_manifest_path: pathlib.Path,
    preflight_receipt_path: pathlib.Path,
    condition: str,
    suite: str,
    output: pathlib.Path,
    trials_per_task: int,
    task_id_start: int,
    task_id_count: int,
    seed: int,
    cuda_device: str,
) -> dict[str, Any]:
    if condition not in CONDITIONS:
        raise ValueError(f"unknown condition {condition!r}")
    if suite not in SUITES:
        raise ValueError(f"unknown suite {suite!r}")
    output = output.resolve()
    if output.exists() and any(output.iterdir()):
        raise task_c_trace.TaskCTraceError(f"refusing to overwrite non-empty condition output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    (output / "server_trace").mkdir()
    (output / "videos").mkdir()
    experiment = "C1_libero_spatial_k9" if suite == "libero_spatial" else f"C3_{suite}_k9"
    run_id = f"{experiment.lower()}-{condition}-seed{seed}-tasks{task_id_start}-{task_id_start + task_id_count - 1}"
    port = _find_port()
    provenance = output / "provenance"
    checkpoint_verification = verify_checkpoint_manifest(checkpoint.parent, checkpoint_manifest_path)
    checkpoint_manifest_sha = task_c_trace.sha256_file(checkpoint_manifest_path)
    preflight_receipt_path = preflight_receipt_path.resolve()
    preflight_receipt = json.loads(preflight_receipt_path.read_text(encoding="utf-8"))
    if (
        preflight_receipt.get("backend") != "gpu"
        or preflight_receipt.get("jax_version") != "0.5.3"
        or not preflight_receipt.get("device_kinds")
        or any("H100" not in kind for kind in preflight_receipt["device_kinds"])
    ):
        raise task_c_trace.TaskCTraceError("run requires a successful JAX 0.5.3 H100 preflight receipt")
    manifest = {
        "schema_version": task_c_trace.SCHEMA_VERSION,
        "status": "running",
        "started_at": _now(),
        "run_id": run_id,
        "condition": condition,
        "kappa_delta": CONDITIONS[condition],
        "experiment": experiment,
        "suite": suite,
        "seed": seed,
        "task_id_start": task_id_start,
        "task_id_count": task_id_count,
        "trials_per_task": trials_per_task,
        "expected_episodes": task_id_count * trials_per_task,
        "action_horizon": 10,
        "trigger_k": 9,
        "overlap": 1,
        "wm_delta_t": 1.0,
        "confidence_schedule": condition != "faac_only",
        "checkpoint": {
            "model_id": MODEL_ID,
            "resolved_revision": MODEL_REVISION,
            "root": str(checkpoint.parent),
            "manifest": str(checkpoint_manifest_path),
            "manifest_sha256": checkpoint_manifest_sha,
            "verification": checkpoint_verification,
        },
        "preflight": {
            "path": str(preflight_receipt_path),
            "sha256": task_c_trace.sha256_file(preflight_receipt_path),
        },
        "world_model": {
            "root": str(wm_checkpoint),
            "token_reducer_kind": "learned_cross_attn",
            "action_encoder_kind": "transformer_block",
            "training_suite": "libero_spatial",
            "out_of_training_suite": suite != "libero_spatial",
        },
        "repo": _git_receipt(repo),
        "device": _device_receipt(cuda_device),
        "environments": {
            "policy": _freeze(policy_python, provenance / "policy-freeze.txt"),
            "libero": _freeze(libero_python, provenance / "libero-freeze.txt"),
            "libero_lock": {
                "path": str(repo / "scripts" / "requirements-task-c-libero.txt"),
                "sha256": task_c_trace.sha256_file(repo / "scripts" / "requirements-task-c-libero.txt"),
            },
        },
        "timing": {
            "c_tier0_boundary": "jitted 40M WM forward including latent reducer and device kappa reduction, host kappa materialization, and the threshold decision",
            "warmup_wm_calls_discarded": 30,
        },
        "phase_proxy": {
            "rule": "approach before first of two consecutive gripper commands > 0; contact from that command onward",
            "ee_velocity_proxy": "L2 and vector delta of end-effector xyz per simulator control step",
        },
        "mu_split": "episode_idx even calibration; odd eval; split excludes condition and outcome",
        "denoise_pairing_caveat": "not applicable: this JAX evaluation uses the released policy RNG stream per condition",
    }
    task_c_trace.write_json_atomic(output / "run_manifest.json", manifest)

    common_env = _base_env(repo)
    common_env["CUDA_VISIBLE_DEVICES"] = cuda_device
    server_env = dict(common_env)
    server_env.update(
        {
            "OPENPI_TASK_C_TRACE_ROOT": str(output / "server_trace"),
            "OPENPI_TASK_C_RUN_ID": run_id,
            "OPENPI_TASK_C_CONDITION": condition,
            "OPENPI_TASK_C_TIMING_WARMUP_CALLS": "30",
            "OPENPI_LIBERO_NORM_CHECKPOINT_DIR": str(checkpoint),
        }
    )
    server_command = [
        str(policy_python),
        "-u",
        str(repo / "scripts" / "serve_policy.py"),
        "--env",
        "LIBERO",
        "--port",
        str(port),
        "--world-model-checkpoint",
        str(wm_checkpoint),
        "--world-model-token-reducer-kind",
        "learned_cross_attn",
        "--world-model-action-encoder-kind",
        "transformer_block",
        "--async-ae-proprio-source",
        "prefix_t",
        "policy:checkpoint",
        "--policy.config",
        "pi05_libero",
        "--policy.dir",
        str(checkpoint),
    ]
    client_command = [
        str(libero_python),
        "-u",
        str(repo / "examples" / "libero" / "main.py"),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--task-suite-name",
        suite,
        "--num-trials-per-task",
        str(trials_per_task),
        "--task-id-start",
        str(task_id_start),
        "--task-id-count",
        str(task_id_count),
        "--seed",
        str(seed),
        "--async-inference",
        "--async-use-world-model",
        "--action-horizon",
        "10",
        "--async-trigger-step",
        "9",
        "--overlap-skip",
        "1",
        "--pi0-norm-checkpoint-dir",
        str(checkpoint),
        "--video-out-path",
        str(output / "videos"),
        "--no-write-videos",
        "--task-c-trace-dir",
        str(output),
        "--task-c-run-id",
        run_id,
        "--task-c-condition",
        condition,
    ]
    kappa_delta = CONDITIONS[condition]
    if kappa_delta is not None:
        client_command.extend(
            [
                "--async-wm-multi-rollout",
                "--async-wm-multi-rollout-adaptive-kappa",
                "--async-wm-multi-rollout-adaptive-kappa-low-replan",
                "--async-wm-rollout-delta-t",
                "1.0",
                "--async-wm-multi-rollout-kappa-delta",
                str(kappa_delta),
            ]
        )
    task_c_trace.write_json_atomic(output / "commands.json", {"server": server_command, "client": client_command})
    server_log = (output / "serve.log").open("wb")
    client_log = (output / "client.log").open("wb")
    server = subprocess.Popen(
        server_command,
        cwd=repo,
        env=server_env,
        stdout=server_log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    try:
        _wait_server(server, port)
        client_env = dict(common_env)
        client_env["MUJOCO_GL"] = "egl"
        client_env["LIBERO_CONFIG_PATH"] = str(output / "libero_config")
        config_dir = output / "libero_config"
        config_dir.mkdir()
        libero_package = repo / "third_party" / "libero" / "libero" / "libero"
        required_libero_paths = [
            libero_package,
            libero_package / "assets",
            libero_package / "bddl_files",
            libero_package / "init_files",
        ]
        missing_libero_paths = [str(path) for path in required_libero_paths if not path.is_dir()]
        if missing_libero_paths:
            raise task_c_trace.TaskCTraceError(f"missing required LIBERO package data: {missing_libero_paths}")
        task_c_trace.write_json_atomic(
            config_dir / "config.yaml.json",
            {
                "assets": str(libero_package / "assets"),
                "bddl_files": str(libero_package / "bddl_files"),
                "benchmark_root": str(libero_package),
                "datasets": str(libero_package.parent / "datasets"),
                "init_states": str(libero_package / "init_files"),
            },
        )
        # LIBERO requires YAML at this exact path; JSON is valid YAML and keeps
        # the provenance writer deterministic.
        os.replace(config_dir / "config.yaml.json", config_dir / "config.yaml")
        client = subprocess.run(
            client_command,
            cwd=repo,
            env=client_env,
            stdout=client_log,
            stderr=subprocess.STDOUT,
            check=False,
        )
        if client.returncode != 0:
            manifest["status"] = "failed"
            manifest["client_returncode"] = client.returncode
            manifest["ended_at"] = _now()
            task_c_trace.write_json_atomic(output / "run_manifest.json", manifest)
            raise task_c_trace.TaskCTraceError(
                f"LIBERO client failed for {condition} with rc={client.returncode}; see {output / 'client.log'}"
            )
    finally:
        _stop_process(server)
        server_log.close()
        client_log.close()
    manifest["ended_at"] = _now()
    task_c_trace.write_json_atomic(output / "run_manifest.json", manifest)
    summary = task_c_analysis.finalize_condition(output)
    print(json.dumps(summary, sort_keys=True))
    return summary


def _parse_args() -> argparse.Namespace:
    root = pathlib.Path(os.environ.get("TASK_C_ROOT", "/home/pinyarash/dev/pinyarash/jetson-pi-task-c"))
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=pathlib.Path, default=_repo_root())
    parser.add_argument("--artifact-root", type=pathlib.Path, default=root)
    parser.add_argument(
        "--checkpoint-root",
        type=pathlib.Path,
        default=root / "checkpoints" / "Jetson-PI-pi05",
    )
    parser.add_argument("--policy-python", type=pathlib.Path, default=root / "envs" / "policy" / "bin" / "python")
    parser.add_argument("--libero-python", type=pathlib.Path, default=root / "envs" / "libero" / "bin" / "python")
    parser.add_argument("--cuda-device", default="0")
    parser.add_argument("--preflight-receipt", type=pathlib.Path, default=root / "preflight" / "preflight.json")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("checkpoint-manifest")
    preflight_parser = subparsers.add_parser("preflight")
    preflight_parser.add_argument("--out", type=pathlib.Path, default=root / "preflight")
    run_parser = subparsers.add_parser("run-condition")
    run_parser.add_argument("--condition", choices=sorted(CONDITIONS), required=True)
    run_parser.add_argument("--suite", choices=SUITES, default="libero_spatial")
    run_parser.add_argument("--out", type=pathlib.Path, required=True)
    run_parser.add_argument("--trials-per-task", type=int, default=30)
    run_parser.add_argument("--task-id-start", type=int, default=0)
    run_parser.add_argument("--task-id-count", type=int, default=10)
    run_parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    repo = args.repo.resolve()
    checkpoint_root = args.checkpoint_root.resolve()
    pi_checkpoint = checkpoint_root / "pi05_libero"
    wm_checkpoint = checkpoint_root / "future_correction_module"
    if args.command == "checkpoint-manifest":
        checkpoint_manifest(checkpoint_root, args.artifact_root)
    elif args.command == "preflight":
        preflight(
            repo=repo,
            policy_python=args.policy_python,
            checkpoint=pi_checkpoint,
            wm_checkpoint=wm_checkpoint,
            checkpoint_manifest_path=args.artifact_root / "provenance" / "checkpoint_manifest.json",
            output=args.out,
            cuda_device=args.cuda_device,
        )
    else:
        run_condition(
            repo=repo,
            policy_python=args.policy_python,
            libero_python=args.libero_python,
            checkpoint=pi_checkpoint,
            wm_checkpoint=wm_checkpoint,
            checkpoint_manifest_path=args.artifact_root / "provenance" / "checkpoint_manifest.json",
            preflight_receipt_path=args.preflight_receipt,
            condition=args.condition,
            suite=args.suite,
            output=args.out,
            trials_per_task=args.trials_per_task,
            task_id_start=args.task_id_start,
            task_id_count=args.task_id_count,
            seed=args.seed,
            cuda_device=args.cuda_device,
        )


if __name__ == "__main__":
    with contextlib.suppress(BrokenPipeError):
        main()
