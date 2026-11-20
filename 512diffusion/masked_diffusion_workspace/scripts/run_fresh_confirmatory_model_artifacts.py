from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = (
    WORKSPACE_ROOT
    / "configs"
    / "fresh_confirmatory_models_20260901"
    / "frozen_manifest.json"
)
DEFAULT_STATE = (
    WORKSPACE_ROOT
    / "results"
    / "fresh_confirmatory_models_20260901"
    / "orchestrator_status.json"
)
GLOBAL_LOCK_PATH = (
    WORKSPACE_ROOT
    / "results"
    / "fresh_confirmatory_models_20260901"
    / ".orchestrator.lock"
)
AUTHORIZATION_TOKEN = "FRESH_SCENE_TRAINING_APPROVED"
EXPECTED_PROTOCOL = "fresh_confirmatory_models_20260901_v1"
PROCESS_MARKERS = (
    "phase_10_train_masked_diffusion.py",
    "phase_11_compare_guidance_models.py",
    "phase_12_export_runtime_model.py",
    "phase_13_run_formal_benchmark.py",
    "mitsuba",
)


class FreshConfirmatoryRunError(RuntimeError):
    """Raised when provenance or serial-execution safeguards fail."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Verify and, only with an explicit token, run the frozen fresh-scene "
            "training/export DAG strictly serially. The default is read-only."
        )
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--state", type=Path, default=DEFAULT_STATE)
    parser.add_argument(
        "--python-executable",
        default=sys.executable,
        help="Python interpreter with Torch installed; checked before execution.",
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--authorization-token",
        default="",
        help=f"Required with --execute; exact value: {AUTHORIZATION_TOKEN}",
    )
    parser.add_argument(
        "--max-jobs",
        type=int,
        default=None,
        help="Stop cleanly after at most this many newly executed jobs.",
    )
    parser.add_argument(
        "--rerun-stale",
        action="store_true",
        help="Permit replacement of outputs whose registered hash is stale.",
    )
    return parser.parse_args()


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FreshConfirmatoryRunError(f"cannot load JSON {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise FreshConfirmatoryRunError(f"expected JSON object: {path}")
    return payload


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def workspace_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path.resolve()
    return (WORKSPACE_ROOT / path).resolve()


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def acquire_runner_lock() -> Any:
    """Acquire a nonblocking process lock held for the caller's lifetime."""

    lock_path = GLOBAL_LOCK_PATH
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = lock_path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        handle.seek(0)
        owner = handle.read().strip() or "unknown owner"
        handle.close()
        raise FreshConfirmatoryRunError(
            f"another orchestrator holds {lock_path}: {owner}"
        ) from exc
    handle.seek(0)
    handle.truncate()
    handle.write(f"pid={os.getpid()} acquired_at={now_utc()}\n")
    handle.flush()
    return handle


def validate_manifest(manifest_path: Path) -> tuple[dict[str, Any], str]:
    manifest_path = manifest_path.resolve()
    manifest = load_json(manifest_path)
    if manifest.get("protocol_id") != EXPECTED_PROTOCOL:
        raise FreshConfirmatoryRunError(
            f"unexpected protocol: {manifest.get('protocol_id')!r}"
        )
    jobs = manifest.get("jobs")
    if not isinstance(jobs, list) or len(jobs) != 60:
        raise FreshConfirmatoryRunError("frozen manifest must contain 60 jobs")
    code_provenance = manifest.get("code_provenance")
    if not isinstance(code_provenance, dict) or not code_provenance:
        raise FreshConfirmatoryRunError("frozen code provenance is missing")
    for path_text, expected_sha in code_provenance.items():
        code_path = workspace_path(str(path_text))
        if not code_path.is_file():
            raise FreshConfirmatoryRunError(f"missing frozen code input: {code_path}")
        actual_sha = sha256_file(code_path)
        if actual_sha != expected_sha:
            raise FreshConfirmatoryRunError(
                f"frozen code SHA-256 mismatch for {path_text}: {actual_sha}"
            )
    observed_ids: set[str] = set()
    completed_ids: set[str] = set()
    kinds: dict[str, int] = {}
    for job in jobs:
        if not isinstance(job, dict):
            raise FreshConfirmatoryRunError("job entry is not an object")
        job_id = str(job.get("job_id", ""))
        if not job_id or job_id in observed_ids:
            raise FreshConfirmatoryRunError(f"invalid/duplicate job id: {job_id!r}")
        observed_ids.add(job_id)
        kind = str(job.get("kind", ""))
        kinds[kind] = kinds.get(kind, 0) + 1
        script_path = workspace_path(str(job.get("script", "")))
        if not script_path.is_file():
            raise FreshConfirmatoryRunError(f"missing job script for {job_id}")
        if sha256_file(script_path) != job.get("script_sha256"):
            raise FreshConfirmatoryRunError(
                f"job script SHA-256 mismatch for {job_id}: {script_path}"
            )
        for dependency in job.get("dependencies", []):
            if dependency not in completed_ids:
                raise FreshConfirmatoryRunError(
                    f"job {job_id} has missing or forward dependency {dependency}"
                )
        config_path = workspace_path(str(job["config_path"]))
        if not config_path.is_file():
            raise FreshConfirmatoryRunError(f"missing config for {job_id}: {config_path}")
        actual_config_sha = sha256_file(config_path)
        if actual_config_sha != job.get("config_sha256"):
            raise FreshConfirmatoryRunError(
                f"config SHA-256 mismatch for {job_id}: {actual_config_sha}"
            )
        json.loads(config_path.read_text(encoding="utf-8"))
        completed_ids.add(job_id)
    expected_kinds = {
        "phase10_ordered_training": 15,
        "phase11_simple_training": 15,
        "phase12_ordered_export": 15,
        "phase12_simple_export": 15,
    }
    if kinds != expected_kinds:
        raise FreshConfirmatoryRunError(f"unexpected job counts: {kinds}")
    if manifest.get("parameter_budget", {}).get("passes") is not True:
        raise FreshConfirmatoryRunError("frozen parameter-count budget did not pass")
    for scene in manifest.get("scenes", []):
        replay_path = workspace_path(str(scene["replay_path"]))
        if not replay_path.is_file():
            raise FreshConfirmatoryRunError(f"missing replay: {replay_path}")
        actual_replay_sha = sha256_file(replay_path)
        if actual_replay_sha != scene.get("replay_sha256"):
            raise FreshConfirmatoryRunError(
                f"replay SHA-256 mismatch for {scene.get('scene')}: {actual_replay_sha}"
            )
    return manifest, sha256_file(manifest_path)


def blank_state(manifest: dict[str, Any], manifest_sha: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "protocol_id": EXPECTED_PROTOCOL,
        "manifest_sha256": manifest_sha,
        "created_at": now_utc(),
        "updated_at": now_utc(),
        "jobs": {
            job["job_id"]: {
                "status": "pending",
                "attempts": [],
                "config_sha256": job["config_sha256"],
                "output_sha256": {},
                "artifact_sha256": {},
            }
            for job in manifest["jobs"]
        },
    }


def load_or_prepare_state(
    *, state_path: Path, manifest: dict[str, Any], manifest_sha: str
) -> dict[str, Any]:
    if not state_path.is_file():
        return blank_state(manifest, manifest_sha)
    state = load_json(state_path)
    if state.get("protocol_id") != EXPECTED_PROTOCOL:
        raise FreshConfirmatoryRunError("state protocol does not match manifest")
    if state.get("manifest_sha256") != manifest_sha:
        raise FreshConfirmatoryRunError(
            "state was created from a different frozen manifest; preserve it and use "
            "a new --state path"
        )
    registered = state.get("jobs")
    if not isinstance(registered, dict):
        raise FreshConfirmatoryRunError("state jobs block is invalid")
    expected_ids = {job["job_id"] for job in manifest["jobs"]}
    if set(registered) != expected_ids:
        raise FreshConfirmatoryRunError("state job set does not match manifest")
    return state


def output_hashes(job: dict[str, Any]) -> dict[str, str] | None:
    hashes: dict[str, str] = {}
    for output in job.get("expected_outputs", []):
        path = workspace_path(str(output))
        if not path.is_file():
            return None
        hashes[str(output)] = sha256_file(path)
    return hashes


def artifact_hashes(job: dict[str, Any]) -> dict[str, str] | None:
    """Hash every substantive file beneath a job's unique output directory."""

    config = load_json(workspace_path(str(job["config_path"])))
    output_dir_text = config.get("output_dir")
    if not isinstance(output_dir_text, str):
        return None
    output_dir = workspace_path(output_dir_text)
    if not output_dir.is_dir():
        return None
    hashes = {
        path.relative_to(WORKSPACE_ROOT).as_posix(): sha256_file(path)
        for path in sorted(output_dir.rglob("*"))
        if path.is_file() and not path.name.startswith(".")
    }
    return hashes or None


def classify_job_state(job: dict[str, Any], record: dict[str, Any]) -> str:
    status = str(record.get("status", "pending"))
    if status == "succeeded":
        current = output_hashes(job)
        current_artifacts = artifact_hashes(job)
        if (
            current is None
            or current != record.get("output_sha256")
            or current_artifacts is None
            or current_artifacts != record.get("artifact_sha256")
        ):
            return "stale"
        return "succeeded"
    if status == "running":
        return "stale"
    if status not in {"pending", "failed", "stale"}:
        return "stale"
    # Even a partial tree can be a remnant from a killed process. Never
    # overwrite it without the caller's explicit stale-output acknowledgement.
    if status in {"pending", "failed"} and artifact_hashes(job) is not None:
        return "stale"
    return status


def active_conflicting_processes() -> list[str]:
    result = subprocess.run(
        ["ps", "-axo", "pid=,command="],
        check=True,
        capture_output=True,
        text=True,
    )
    conflicts: list[str] = []
    for line in result.stdout.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        pid_text, _, command = stripped.partition(" ")
        try:
            pid = int(pid_text)
        except ValueError:
            continue
        if pid == os.getpid():
            continue
        if any(marker in command.lower() for marker in PROCESS_MARKERS):
            conflicts.append(stripped)
    return conflicts


def ensure_execution_preflight(python_executable: str) -> dict[str, Any]:
    conflicts = active_conflicting_processes()
    if conflicts:
        raise FreshConfirmatoryRunError(
            "refusing to overlap an existing training/export/render process:\n"
            + "\n".join(conflicts)
        )
    resolved_python = shutil.which(python_executable)
    if resolved_python is None:
        raise FreshConfirmatoryRunError(
            f"Python executable is not resolvable: {python_executable}"
        )
    probe = (
        "import json,platform,sys,torch; "
        "print(json.dumps({'python_executable':sys.executable,"
        "'python_version':platform.python_version(),'torch_version':torch.__version__,"
        "'mps_available':bool(torch.backends.mps.is_available())},sort_keys=True))"
    )
    result = subprocess.run(
        [resolved_python, "-c", probe],
        cwd=WORKSPACE_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise FreshConfirmatoryRunError(
            f"selected Python cannot import torch: {python_executable}\n{result.stderr}"
        )
    try:
        environment = json.loads(result.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError) as exc:
        raise FreshConfirmatoryRunError(
            f"could not parse Python/Torch preflight: {result.stdout!r}"
        ) from exc
    # Preserve the virtual-environment launcher path. Resolving its symlink to
    # the base interpreter can drop the venv's Torch site-packages.
    environment["resolved_python"] = os.path.abspath(resolved_python)
    return environment


def input_hashes(job: dict[str, Any]) -> dict[str, str]:
    config_path = workspace_path(str(job["config_path"]))
    config = load_json(config_path)
    paths: list[str] = [str(job["config_path"]), str(job["script"])]
    paths.extend(str(item) for item in config.get("replay_paths", []))
    checkpoint = config.get("masked_diffusion_checkpoint") or config.get(
        "checkpoint_path"
    )
    if checkpoint:
        paths.append(str(checkpoint))
    hashes: dict[str, str] = {}
    for value in paths:
        path = workspace_path(value)
        if not path.is_file():
            raise FreshConfirmatoryRunError(
                f"missing input for {job['job_id']}: {value}"
            )
        hashes[value] = sha256_file(path)
    return hashes


def inspect_checkpoint(python_executable: str, checkpoint_path: Path) -> dict[str, Any]:
    probe = """
import json
import sys
import torch

path = sys.argv[1]
try:
    payload = torch.load(path, map_location="cpu", weights_only=False)
except TypeError:
    payload = torch.load(path, map_location="cpu")
state = payload["model_state_dict"]
print(json.dumps({
    "parameter_count": sum(int(value.numel()) for value in state.values()),
    "config": payload["config"],
}, sort_keys=True))
"""
    result = subprocess.run(
        [python_executable, "-c", probe, str(checkpoint_path)],
        cwd=WORKSPACE_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise FreshConfirmatoryRunError(
            f"checkpoint inspection failed for {checkpoint_path}: {result.stderr}"
        )
    try:
        return json.loads(result.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError) as exc:
        raise FreshConfirmatoryRunError(
            f"invalid checkpoint inspection output for {checkpoint_path}"
        ) from exc


def validate_checkpoint_contract(
    *, job: dict[str, Any], config: dict[str, Any], inspection: dict[str, Any]
) -> dict[str, Any]:
    expected_count = int(config["expected_parameter_count"])
    if int(inspection.get("parameter_count", -1)) != expected_count:
        raise FreshConfirmatoryRunError(
            f"{job['job_id']} parameter count={inspection.get('parameter_count')}, "
            f"expected {expected_count}"
        )
    saved = inspection.get("config")
    if not isinstance(saved, dict):
        raise FreshConfirmatoryRunError(f"{job['job_id']} checkpoint config is missing")
    for key in ("seed", "replay_paths", "batch_size", "val_fraction"):
        if saved.get(key) != config.get(key):
            raise FreshConfirmatoryRunError(
                f"{job['job_id']} checkpoint {key}={saved.get(key)!r}, "
                f"expected {config.get(key)!r}"
            )
    if job["kind"] == "phase10_ordered_training":
        expected = {
            "batch_size": 16,
            "epochs": 32,
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
            "val_fraction": 0.25,
            "max_records": None,
            "num_workers": 0,
            "early_stop_patience": None,
            "grad_clip_norm": 0.5,
            "direction_loss_weight": 1.0,
            "direction_loss_type": "vmf_mixture",
            "direction_vmf_kappa": 1.0,
            "descriptor_weight_mode": "log_energy",
            "descriptor_weight_quantile": 0.99,
            "direction_weight_mode": "log_energy",
            "direction_weight_quantile": 0.99,
            "direction_distribution": "vmf_mixture",
            "vmf_components": 4,
            "vmf_max_kappa": 64.0,
            "max_nonfinite_batches": 4,
            "direction_space": "s2",
            "full_mask_probability": 0.25,
            "stochastic_diffusion": False,
            "diffusion_objective": "predict_x0",
            "diffusion_train_steps": 16,
            "beta_schedule": "linear",
            "beta_start": 0.0005,
            "beta_end": 0.05,
            "reverse_noise_scale": 1.0,
            "backbone_type": "mlp",
            "hidden_dim": 72,
            "depth": 3,
            "dropout": 0.1,
            "token_embedding_dim": 24,
            "time_embedding_dim": 16,
            "discrete_diffusion_mode": "ordered_absorbing",
        }
        observed = saved
    else:
        expected = {
            "epochs": 32,
            "hidden_dim": 126,
            "depth": 4,
            "dropout": 0.1,
            "vmf_components": 4,
            "probabilistic_direction": True,
            "min_kappa": 0.001,
            "max_kappa": 64.0,
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
            "grad_clip_norm": 0.5,
        }
        observed = saved.get("simple_baseline", {})
        for key, expected_value in {
            "batch_size": 16,
            "val_fraction": 0.25,
            "max_records": None,
            "num_workers": 0,
            "direction_loss_weight": 1.0,
            "direction_loss_type": "vmf_mixture",
            "direction_vmf_kappa": 1.0,
            "diffusion_steps": 1,
            "stochastic_diffusion": False,
            "diffusion_train_steps": 1,
            "beta_schedule": "linear",
            "beta_start": 0.0005,
            "beta_end": 0.05,
            "reverse_noise_scale": 1.0,
        }.items():
            if saved.get(key) != expected_value:
                raise FreshConfirmatoryRunError(
                    f"{job['job_id']} checkpoint {key}={saved.get(key)!r}, "
                    f"expected {expected_value!r}"
                )
    for key, expected_value in expected.items():
        if observed.get(key) != expected_value:
            raise FreshConfirmatoryRunError(
                f"{job['job_id']} checkpoint {key}={observed.get(key)!r}, "
                f"expected {expected_value!r}"
            )
    return {
        "checkpoint_parameter_count": expected_count,
        "checkpoint_seed": saved["seed"],
        "checkpoint_replay_paths": saved["replay_paths"],
    }


def read_runtime_header(path: Path) -> tuple[str, dict[str, str]]:
    values: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as handle:
        signature = handle.readline().strip()
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("tensor "):
                break
            key, separator, value = stripped.partition(" ")
            if separator:
                values[key] = value
    return signature, values


def validate_job_outputs(
    *, job: dict[str, Any], python_executable: str
) -> dict[str, Any]:
    """Reject stale/wrong artifacts even when the child process exits zero."""

    config = load_json(workspace_path(str(job["config_path"])))
    kind = str(job["kind"])
    if kind == "phase10_ordered_training":
        checkpoint = workspace_path(str(job["expected_outputs"][0]))
        inspection = inspect_checkpoint(python_executable, checkpoint)
        return validate_checkpoint_contract(job=job, config=config, inspection=inspection)
    if kind == "phase11_simple_training":
        checkpoint = workspace_path(str(job["expected_outputs"][0]))
        inspection = inspect_checkpoint(python_executable, checkpoint)
        return validate_checkpoint_contract(job=job, config=config, inspection=inspection)

    checkpoint = workspace_path(str(config["checkpoint_path"]))
    inspection = inspect_checkpoint(python_executable, checkpoint)
    synthetic_kind = (
        "phase10_ordered_training"
        if config["model_kind"] == "masked_diffusion"
        else "phase11_simple_training"
    )
    checkpoint_config = load_json(
        workspace_path(
            str(
                next(
                    candidate["config_path"]
                    for candidate in _ACTIVE_MANIFEST_JOBS
                    if candidate["job_id"] in job.get("dependencies", [])
                    and candidate["kind"] == synthetic_kind
                )
            )
        )
    )
    synthetic_job = dict(job)
    synthetic_job["kind"] = synthetic_kind
    provenance = validate_checkpoint_contract(
        job=synthetic_job, config=checkpoint_config, inspection=inspection
    )
    runtime_path = workspace_path(str(job["expected_outputs"][0]))
    summary_path = workspace_path(str(job["expected_outputs"][1]))
    summary = load_json(summary_path)
    summary_config = summary.get("config", {})
    for key in ("checkpoint_path", "output_dir", "output_name", "model_kind"):
        if summary_config.get(key) != config.get(key):
            raise FreshConfirmatoryRunError(
                f"{job['job_id']} export summary {key}={summary_config.get(key)!r}, "
                f"expected {config.get(key)!r}"
            )
    summary_training = summary.get("training_config", {})
    for key in ("seed", "replay_paths"):
        if summary_training.get(key) != inspection["config"].get(key):
            raise FreshConfirmatoryRunError(
                f"{job['job_id']} export training {key} provenance mismatch"
            )
    signature, header = read_runtime_header(runtime_path)
    expected_header = {
        "masked_diffusion": {
            "backbone_type": "mlp",
            "hidden_dim": "72",
            "depth": "3",
            "direction_distribution": "vmf_mixture",
            "vmf_components": "4",
            "vmf_max_kappa": "64",
            "runtime_input_mode": "full_mlp",
            "discrete_diffusion_mode": "ordered_absorbing",
        },
        "simple_mlp": {
            "backbone_type": "plain_mlp",
            "hidden_dim": "126",
            "depth": "4",
            "direction_distribution": "vmf_mixture",
            "vmf_components": "4",
            "vmf_min_kappa": "0.001",
            "vmf_max_kappa": "64",
            "runtime_input_mode": "condition_only",
            "discrete_diffusion_mode": "legacy",
        },
    }[config["model_kind"]]
    if signature != "masked_diffusion_runtime_v1":
        raise FreshConfirmatoryRunError(
            f"{job['job_id']} runtime signature={signature!r}"
        )
    for key, value in expected_header.items():
        if header.get(key) != value:
            raise FreshConfirmatoryRunError(
                f"{job['job_id']} runtime {key}={header.get(key)!r}, expected {value!r}"
            )
    provenance["runtime_header"] = header
    provenance["runtime_signature"] = signature
    return provenance


# Set by main after manifest validation; kept module-local so export validation
# can resolve the exact training config from the frozen dependency edge.
_ACTIVE_MANIFEST_JOBS: list[dict[str, Any]] = []


def run_one_job(
    *,
    job: dict[str, Any],
    record: dict[str, Any],
    state: dict[str, Any],
    state_path: Path,
    python_executable: str,
    manifest_path: Path,
    manifest_sha256: str,
) -> bool:
    attempts = record.setdefault("attempts", [])
    attempt_number = len(attempts) + 1
    log_dir = state_path.parent / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = log_dir / f"{job['job_id']}.attempt{attempt_number}.stdout.log"
    stderr_path = log_dir / f"{job['job_id']}.attempt{attempt_number}.stderr.log"
    command = [
        python_executable,
        str(workspace_path(str(job["script"]))),
        "--config",
        str(workspace_path(str(job["config_path"]))),
    ]
    attempt: dict[str, Any] = {
        "attempt": attempt_number,
        "started_at": now_utc(),
        "command": command,
        "input_sha256": input_hashes(job),
        "stdout_log": str(stdout_path),
        "stderr_log": str(stderr_path),
    }
    attempts.append(attempt)
    record["status"] = "running"
    state["updated_at"] = now_utc()
    atomic_write_json(state_path, state)

    started = time.monotonic()
    with stdout_path.open("w", encoding="utf-8") as stdout_handle, stderr_path.open(
        "w", encoding="utf-8"
    ) as stderr_handle:
        result = subprocess.run(
            command,
            cwd=WORKSPACE_ROOT,
            check=False,
            stdout=stdout_handle,
            stderr=stderr_handle,
            text=True,
        )
    attempt["finished_at"] = now_utc()
    attempt["duration_seconds"] = time.monotonic() - started
    attempt["return_code"] = result.returncode
    current_outputs = output_hashes(job)
    current_artifacts = artifact_hashes(job)
    validation_error: str | None = None
    output_provenance: dict[str, Any] | None = None
    try:
        _, current_manifest_sha = validate_manifest(manifest_path)
        if current_manifest_sha != manifest_sha256:
            raise FreshConfirmatoryRunError("frozen manifest changed during the job")
        after_inputs = input_hashes(job)
        attempt["post_input_sha256"] = after_inputs
        if after_inputs != attempt["input_sha256"]:
            raise FreshConfirmatoryRunError("job inputs changed while it was running")
        if result.returncode == 0 and current_outputs is not None:
            output_provenance = validate_job_outputs(
                job=job, python_executable=python_executable
            )
    except (FreshConfirmatoryRunError, KeyError, OSError, StopIteration, ValueError) as exc:
        validation_error = f"{type(exc).__name__}: {exc}"
    if (
        result.returncode == 0
        and current_outputs is not None
        and current_artifacts is not None
        and validation_error is None
        and output_provenance is not None
    ):
        record["status"] = "succeeded"
        record["output_sha256"] = current_outputs
        record["artifact_sha256"] = current_artifacts
        attempt["output_sha256"] = current_outputs
        attempt["artifact_sha256"] = current_artifacts
        attempt["output_provenance"] = output_provenance
        record["output_provenance"] = output_provenance
        succeeded = True
    else:
        record["status"] = "failed"
        record["output_sha256"] = current_outputs or {}
        record["artifact_sha256"] = current_artifacts or {}
        attempt["error"] = (
            f"return code {result.returncode}; "
            f"expected_outputs_complete={current_outputs is not None}; "
            f"validation_error={validation_error}"
        )
        succeeded = False
    state["updated_at"] = now_utc()
    atomic_write_json(state_path, state)
    return succeeded


def summarize(
    manifest: dict[str, Any], state: dict[str, Any]
) -> tuple[dict[str, int], list[str]]:
    counts: dict[str, int] = {}
    stale: list[str] = []
    classifications: dict[str, str] = {}
    for job in manifest["jobs"]:
        job_id = job["job_id"]
        classification = classify_job_state(job, state["jobs"][job_id])
        if any(
            classifications.get(dependency) != "succeeded"
            for dependency in job.get("dependencies", [])
        ) and classification == "succeeded":
            # A valid-looking child is still stale when its recorded input was
            # produced by a dependency that must be regenerated.
            classification = "stale"
        classifications[job_id] = classification
        counts[classification] = counts.get(classification, 0) + 1
        if classification == "stale":
            stale.append(job_id)
    return counts, stale


def main() -> int:
    global _ACTIVE_MANIFEST_JOBS

    args = parse_args()
    if args.max_jobs is not None and args.max_jobs <= 0:
        raise FreshConfirmatoryRunError("--max-jobs must be positive")
    manifest_path = args.manifest.resolve()
    state_path = args.state.resolve()
    manifest, manifest_sha = validate_manifest(manifest_path)
    state = load_or_prepare_state(
        state_path=state_path, manifest=manifest, manifest_sha=manifest_sha
    )
    counts, stale = summarize(manifest, state)
    print(f"protocol={EXPECTED_PROTOCOL}")
    print(f"manifest_sha256={manifest_sha}")
    print(f"state={state_path}")
    print("status=" + ",".join(f"{key}:{counts[key]}" for key in sorted(counts)))

    if not args.execute:
        print("check-only: no state, training, export, or render was written")
        if stale:
            print("stale_jobs=" + ",".join(stale))
        return 0
    if args.authorization_token != AUTHORIZATION_TOKEN:
        raise FreshConfirmatoryRunError(
            f"--execute requires --authorization-token {AUTHORIZATION_TOKEN}"
        )
    # The descriptor remains open until main returns, so flock prevents a
    # second orchestrator from passing the same preflight concurrently.
    runner_lock = acquire_runner_lock()
    manifest, locked_manifest_sha = validate_manifest(manifest_path)
    if locked_manifest_sha != manifest_sha:
        raise FreshConfirmatoryRunError("frozen manifest changed before lock acquisition")
    state = load_or_prepare_state(
        state_path=state_path, manifest=manifest, manifest_sha=manifest_sha
    )
    _ACTIVE_MANIFEST_JOBS = list(manifest["jobs"])
    environment = ensure_execution_preflight(args.python_executable)
    previous_environment = state.get("execution_environment")
    if previous_environment is not None and previous_environment != environment:
        raise FreshConfirmatoryRunError(
            "Python/Torch environment differs from the environment registered in state"
        )
    state["execution_environment"] = environment
    resolved_python = str(environment["resolved_python"])
    print(
        f"torch={environment['torch_version']} "
        f"python={resolved_python} lock_fd={runner_lock.fileno()}"
    )

    counts, stale = summarize(manifest, state)

    if stale and not args.rerun_stale:
        raise FreshConfirmatoryRunError(
            "stale/interrupted/unregistered outputs detected; inspect them, then use "
            "--rerun-stale to replace only these protocol-scoped outputs: "
            + ", ".join(stale)
        )
    if stale:
        stale_set = set(stale)
        for job in manifest["jobs"]:
            if job["job_id"] in stale_set:
                record = state["jobs"][job["job_id"]]
                record["status"] = "pending"
                record["output_sha256"] = {}
                record["artifact_sha256"] = {}

    # State is first materialized only after the two explicit execution gates.
    atomic_write_json(state_path, state)
    executed = 0
    for job in manifest["jobs"]:
        _, current_manifest_sha = validate_manifest(manifest_path)
        if current_manifest_sha != manifest_sha:
            raise FreshConfirmatoryRunError("frozen manifest changed before next job")
        job_id = job["job_id"]
        record = state["jobs"][job_id]
        if classify_job_state(job, record) == "succeeded":
            print(f"SKIP {job_id} (registered hashes match)")
            continue
        for dependency in job.get("dependencies", []):
            dependency_job = next(
                item for item in manifest["jobs"] if item["job_id"] == dependency
            )
            dependency_record = state["jobs"][dependency]
            if classify_job_state(dependency_job, dependency_record) != "succeeded":
                raise FreshConfirmatoryRunError(
                    f"dependency {dependency} is not a verified success for {job_id}"
                )
        conflicts = active_conflicting_processes()
        if conflicts:
            raise FreshConfirmatoryRunError(
                "a conflicting process appeared before the next job:\n"
                + "\n".join(conflicts)
            )
        print(f"RUN {job_id}", flush=True)
        if not run_one_job(
            job=job,
            record=record,
            state=state,
            state_path=state_path,
            python_executable=resolved_python,
            manifest_path=manifest_path,
            manifest_sha256=manifest_sha,
        ):
            print(f"FAILED {job_id}; stopped serial runner", file=sys.stderr)
            return 1
        print(f"DONE {job_id}", flush=True)
        executed += 1
        if args.max_jobs is not None and executed >= args.max_jobs:
            print(f"max_jobs reached after {executed} new jobs")
            break

    counts, _ = summarize(manifest, state)
    print("final_status=" + ",".join(f"{key}:{counts[key]}" for key in sorted(counts)))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except FreshConfirmatoryRunError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
