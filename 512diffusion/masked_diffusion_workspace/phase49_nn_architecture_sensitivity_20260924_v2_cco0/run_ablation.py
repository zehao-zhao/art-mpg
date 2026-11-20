#!/usr/bin/env python3
"""Resumable CCO0-only controller for Phase-49 NN sensitivity."""

from __future__ import annotations

import argparse
import csv
import fcntl
import json
import math
import os
import platform
import shutil
import socket
import statistics
import subprocess
import sys
import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from queue import Queue
from typing import Any, Iterator


PACKAGE_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = PACKAGE_ROOT.parent
PROJECT_ROOT = WORKSPACE_ROOT.parent
for entry in (str(PACKAGE_ROOT), str(WORKSPACE_ROOT / "scripts"), str(WORKSPACE_ROOT / "src")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

import protocol  # noqa: E402
import overlay_runtime  # noqa: E402


class ControllerError(RuntimeError):
    """Raised when execution cannot proceed without violating the contract."""


METRIC_FIELDS = (
    "config_id", "direction_conditioning_mode", "epochs", "run_id",
    "hidden_dim", "depth", "vmf_components", "token_embedding_dim",
    "time_embedding_dim", "dropout", "parameter_count",
    "scene", "sample_count", "model_seed", "render_seed",
    "mse", "psnr", "reference_peak", "render_seconds", "output_path",
    "output_sha256", "reference_path", "reference_sha256", "host_id",
    "task_id",
)
TRAINING_METRIC_FIELDS = (
    "config_id", "direction_conditioning_mode", "epochs", "scene",
    "hidden_dim", "depth", "vmf_components", "token_embedding_dim",
    "time_embedding_dim", "dropout", "parameter_count",
    "model_seed", "training_seconds", "epochs_completed", "final_epoch",
    "final_val_loss", "final_val_chain_length_loss", "final_val_chain_type_loss",
    "final_val_direction_loss", "best_epoch", "best_val_loss",
    "best_val_chain_length_loss", "best_val_chain_type_loss",
    "best_val_direction_loss", "checkpoint_path", "checkpoint_sha256",
    "summary_path", "summary_sha256", "metrics_json_path",
    "metrics_json_sha256", "host_id", "task_id",
)


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def atomic_json(path: Path, value: object) -> None:
    atomic_bytes(path, protocol.json_bytes(value))


def append_jsonl(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(value, sort_keys=True, allow_nan=False) + "\n"
    with path.open("a", encoding="utf-8") as stream:
        stream.write(line)
        stream.flush()
        os.fsync(stream.fileno())


@contextmanager
def exclusive_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    stream = path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        stream.close()
        raise ControllerError(f"another Phase49 controller holds {path}") from exc
    try:
        yield
    finally:
        fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        stream.close()


def host_root(manifest: dict[str, Any], host_id: str) -> Path:
    return protocol.workspace_path(manifest["output_root"]) / "hosts" / host_id


def _parse_cpu_list(value: str) -> tuple[int, ...]:
    cpus: set[int] = set()
    for field in value.strip().split(","):
        if not field:
            continue
        bounds = field.split("-", 1)
        first = int(bounds[0])
        last = int(bounds[-1])
        if first < 0 or last < first:
            raise ControllerError(f"invalid CPU-list field: {field}")
        cpus.update(range(first, last + 1))
    return tuple(sorted(cpus))


def _physical_cpu_representatives() -> tuple[int, ...]:
    if platform.system() != "Linux":
        allowed = (
            sorted(os.sched_getaffinity(0))
            if hasattr(os, "sched_getaffinity")
            else list(range(os.cpu_count() or 1))
        )
        return tuple(allowed)

    # The controller may be pinned to a CPU reserved from compute workers. Its
    # process affinity therefore cannot be used to discover the worker pool.
    # Read the cgroup-effective set (falling back to the online set) so the
    # preflight validates CPUs that child taskset invocations may use.
    cgroup_effective = Path("/sys/fs/cgroup/cpuset.cpus.effective")
    online = Path("/sys/devices/system/cpu/online")
    try:
        allowed_text = cgroup_effective.read_text().strip()
        if not allowed_text:
            allowed_text = online.read_text().strip()
        allowed = _parse_cpu_list(allowed_text)
    except (OSError, ValueError) as exc:
        raise ControllerError("cannot read the host-effective CPU set") from exc
    result: list[int] = []
    seen: set[tuple[str, str]] = set()
    for cpu in allowed:
        topology = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
        try:
            key = (
                (topology / "physical_package_id").read_text().strip(),
                (topology / "core_id").read_text().strip(),
            )
        except OSError as exc:
            raise ControllerError(f"cannot read physical topology for CPU{cpu}") from exc
        if key not in seen:
            seen.add(key)
            result.append(cpu)
    return tuple(result)


def _host_matches(expected: str) -> bool:
    actual = {socket.gethostname().lower(), socket.getfqdn().lower()}
    expected_values = {expected.lower(), expected.split(".")[0].lower()}
    return any(value in expected_values or value.split(".")[0] in expected_values for value in actual)


def _python_probe(python: Path) -> dict[str, Any]:
    if not python.is_file() or not os.access(python, os.X_OK):
        raise ControllerError(f"configured Python is not executable: {python}")
    code = (
        "import cv2,json,numpy,torch;"
        "cv2.setNumThreads(1);"
        "print(json.dumps({'opencv':cv2.__version__,'numpy':numpy.__version__,"
        "'torch':torch.__version__,'opencv_threads':cv2.getNumThreads(),"
        "'cuda':torch.cuda.is_available()}))"
    )
    completed = subprocess.run(
        [str(python), "-B", "-c", code], cwd=WORKSPACE_ROOT,
        env=_worker_env(1), capture_output=True, text=True, timeout=60, check=False,
    )
    if completed.returncode != 0:
        raise ControllerError(f"Python dependency probe failed: {completed.stderr.strip()}")
    try:
        value = json.loads(completed.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError) as exc:
        raise ControllerError("Python dependency probe returned invalid JSON") from exc
    if value.get("opencv_threads") != 1:
        raise ControllerError("OpenCV thread cap is not effective")
    return value


def _worker_env(threads: int) -> dict[str, str]:
    env = dict(os.environ)
    env.update({
        "PYTHONNOUSERSITE": "1",
        "CUDA_VISIBLE_DEVICES": "",
        "OPENCV_IO_ENABLE_OPENEXR": "1",
        "OPENCV_FOR_THREADS_NUM": "1",
        "OMP_NUM_THREADS": str(threads),
        "MKL_NUM_THREADS": str(threads),
        "OPENBLAS_NUM_THREADS": str(threads),
        "NUMEXPR_NUM_THREADS": str(threads),
        "VECLIB_MAXIMUM_THREADS": str(threads),
    })
    return env


def _implementation_gate(manifest: dict[str, Any]) -> dict[str, Any]:
    python_overlay = overlay_runtime.activate_python_overlay(manifest)
    native_runtime = overlay_runtime.verify_native_runtime(manifest)
    from masked_diffusion_mpg.model import SUPPORTED_DIRECTION_CONDITIONING_MODES

    required_mode = manifest["implementation"]["required_direction_conditioning_mode"]
    if required_mode not in set(SUPPORTED_DIRECTION_CONDITIONING_MODES):
        raise ControllerError("Python implementation does not expose full conditioning")
    header = protocol.workspace_path(manifest["phase48_overlay"]["native_header_snapshot"])
    plugin = protocol.project_path(manifest["implementation"]["mpg_plugin"])
    header_text = header.read_text(encoding="utf-8")
    plugin_bytes = plugin.read_bytes()
    required_tokens = {
        "direction_conditioning_mode", "full", "hidden_dim", "depth", "vmf_components",
    }
    for token in required_tokens:
        if token not in header_text:
            raise ControllerError(f"native runtime header lacks ablation token: {token}")
        if token.encode() not in plugin_bytes:
            raise ControllerError(
                f"compiled MPG plugin lacks ablation token {token}; rebuild before launch"
            )
    return {
        "required_mode": required_mode, "native_architecture_tokens_present": True,
        "python_overlay": python_overlay, "native_runtime": native_runtime,
    }


def preflight(
    manifest: dict[str, Any], host_id: str, python: Path, *, write_receipt: bool,
) -> dict[str, Any]:
    if platform.system() != "Linux":
        raise ControllerError("Phase49 execution preflight is Linux-only")
    expected_host = manifest["hosts"][host_id]["hostname"]
    if not _host_matches(expected_host):
        raise ControllerError(
            f"host assignment mismatch: expected {expected_host}, actual {socket.getfqdn()}"
        )
    expected_root = Path(manifest["required_server_project_root"]).resolve()
    if PROJECT_ROOT.resolve() != expected_root:
        raise ControllerError(f"project root mismatch: {PROJECT_ROOT} != {expected_root}")
    if python.resolve() != Path(manifest["required_worker_python"]).resolve():
        raise ControllerError("worker Python differs from manifest")

    physical = _physical_cpu_representatives()
    if physical != protocol.PHYSICAL_CPU_IDS:
        raise ControllerError(
            "CCO0 physical-core representatives must be exactly the even logical "
            f"CPU IDs 0--164; observed {physical}"
        )
    protocol.assert_cpu_contract(protocol.render_slots(), "render")
    protocol.assert_cpu_contract(protocol.render_slots(), "export")
    protocol.assert_cpu_contract(protocol.training_slots(), "train")
    if shutil.which("taskset") is None:
        raise ControllerError("taskset is required for the CCO0 physical-core allocation")
    inputs = protocol.verify_source_inputs(manifest, require_payloads=True)
    implementation_gate = _implementation_gate(manifest)
    python_runtime = _python_probe(python)
    inputs["phase48_overlay"] = {
        "python_files_sha256": implementation_gate["python_overlay"]["files_sha256"],
        "python_inventory_sha256": implementation_gate["python_overlay"]["inventory_sha256"],
        "phase48_manifest_sha256": implementation_gate["python_overlay"]["phase48_manifest_sha256"],
        "actual_global_plugin_sha256": implementation_gate["native_runtime"]["actual_global_plugin_sha256"],
    }
    inputs["python_runtime"] = python_runtime
    result = {
        "schema_version": 1,
        "protocol_id": protocol.PROTOCOL_ID,
        "status": "preflight_passed",
        "host_id": host_id,
        "hostname": socket.getfqdn(),
        "manifest_sha256": protocol.sha256_file(protocol.MANIFEST_PATH),
        "plan_sha256": protocol.sha256_file(protocol.PLAN_PATH),
        "input_snapshot": inputs,
        "input_snapshot_sha256": protocol.canonical_sha256(inputs),
        "implementation_gate": implementation_gate,
        "python": python_runtime,
        "physical_cpu_representatives": list(physical),
        "authorized_cpu_ids": list(protocol.PHYSICAL_CPU_IDS),
        "training_cpu_ids": list(protocol.TRAINING_CPU_IDS),
        "training_unused_authorized_cpu_ids": list(protocol.TRAINING_UNUSED_CPU_IDS),
        "reserved_cpu_ids": list(protocol.SMT_SIBLING_CPU_IDS),
        "task_counts": protocol.task_counts(manifest)["hosts"][host_id],
        "checked_at": now_utc(),
    }
    if write_receipt:
        root = host_root(manifest, host_id) / "preflight_receipts"
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        path = root / f"preflight_{stamp}_{os.getpid()}.json"
        atomic_json(path, result)
        result["receipt_path"] = protocol.portable_workspace(path)
        result["receipt_sha256"] = protocol.sha256_file(path)
    return result


def _initial_state(manifest: dict[str, Any], host_id: str, inputs: dict[str, Any]) -> dict[str, Any]:
    stages: dict[str, Any] = {}
    for stage in protocol.STAGES:
        tasks = protocol.build_tasks(manifest, host_id, stage)
        stages[stage] = {
            "status": "pending",
            "tasks_sha256": protocol.canonical_sha256([task.payload() for task in tasks]),
            "jobs": {
                task.task_id: {
                    "definition": task.payload(),
                    "definition_sha256": protocol.canonical_sha256(task.payload()),
                    "status": "pending",
                    "attempts": [],
                }
                for task in tasks
            },
        }
    timestamp = now_utc()
    return {
        "schema_version": 1,
        "protocol_id": protocol.PROTOCOL_ID,
        "host_id": host_id,
        "manifest_sha256": protocol.sha256_file(protocol.MANIFEST_PATH),
        "input_snapshot_sha256": protocol.canonical_sha256(inputs),
        "created_at": timestamp,
        "updated_at": timestamp,
        "status": "running",
        "stages": stages,
    }


def _load_state(manifest: dict[str, Any], host_id: str, inputs: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    path = host_root(manifest, host_id) / "state.json"
    if not path.exists():
        state = _initial_state(manifest, host_id, inputs)
        atomic_json(path, state)
        return path, state
    state = protocol.load_json(path)
    expected = {
        "protocol_id": protocol.PROTOCOL_ID,
        "host_id": host_id,
        "manifest_sha256": protocol.sha256_file(protocol.MANIFEST_PATH),
        "input_snapshot_sha256": protocol.canonical_sha256(inputs),
    }
    for key, value in expected.items():
        if state.get(key) != value:
            raise ControllerError(f"resume identity mismatch: {key}")
    fresh = _initial_state(manifest, host_id, inputs)["stages"]
    for stage in protocol.STAGES:
        record = state.get("stages", {}).get(stage)
        if not isinstance(record, dict) or record.get("tasks_sha256") != fresh[stage]["tasks_sha256"]:
            raise ControllerError(f"resume task grid changed: {stage}")
        if set(record.get("jobs", {})) != set(fresh[stage]["jobs"]):
            raise ControllerError(f"resume task identifiers changed: {stage}")
    return path, state


def _result_artifacts(result: dict[str, Any]) -> dict[str, str]:
    pairs: list[tuple[object, object]]
    kind = result.get("kind")
    if kind == "train_diffusion":
        pairs = [
            (result.get("checkpoint_path"), result.get("checkpoint_sha256")),
            (result.get("summary_path"), result.get("summary_sha256")),
            (result.get("metrics_json_path"), result.get("metrics_json_sha256")),
            (result.get("metrics_csv_path"), result.get("metrics_csv_sha256")),
        ]
    elif kind == "export_diffusion":
        pairs = [
            (result.get("runtime_path"), result.get("runtime_sha256")),
            (result.get("summary_path"), result.get("summary_sha256")),
        ]
    elif kind == "render":
        pairs = [(result.get("output_path"), result.get("output_sha256"))]
    else:
        raise ControllerError(f"unexpected worker result kind: {kind}")
    artifacts: dict[str, str] = {}
    for relative, expected in pairs:
        path = protocol.workspace_path(relative) if str(relative).startswith("results/") else protocol.project_path(relative)
        if protocol.sha256_file(path) != expected:
            raise ControllerError(f"worker artifact hash mismatch: {path}")
        artifacts[str(relative)] = str(expected)
    return artifacts


def _valid_success(record: dict[str, Any]) -> bool:
    if record.get("status") != "succeeded" or not record.get("attempts"):
        return False
    attempt = record["attempts"][-1]
    result = attempt.get("result")
    expected = attempt.get("artifacts")
    if not isinstance(result, dict) or not isinstance(expected, dict):
        return False
    try:
        return _result_artifacts(result) == expected
    except (ControllerError, protocol.ProtocolError, OSError):
        return False


def _successful_result(state: dict[str, Any], stage: str, task_id: str) -> dict[str, Any]:
    record = state["stages"][stage]["jobs"].get(task_id)
    if not isinstance(record, dict) or not _valid_success(record):
        raise ControllerError(f"dependency is incomplete or stale: {stage}/{task_id}")
    return dict(record["attempts"][-1]["result"])


def _worker_spec(
    manifest: dict[str, Any], state: dict[str, Any], task: protocol.Task,
    attempt_root: Path, threads: int,
) -> dict[str, Any]:
    config = manifest["configs"][task.config_id]
    scene = protocol.scene_entry(manifest, task.scene)
    spec: dict[str, Any] = {
        "schema_version": 1,
        "protocol_id": protocol.PROTOCOL_ID,
        "manifest_sha256": protocol.sha256_file(protocol.MANIFEST_PATH),
        "attempt_root": protocol.portable_workspace(attempt_root),
        **task.payload(),
        "kind": task.stage,
        "job_id": scene["job_id"],
        "scene_path": scene["scene_path"],
        "threads": threads,
        "direction_conditioning_mode": config["direction_conditioning_mode"],
        "epochs": config["epochs"],
        "hidden_dim": config["hidden_dim"],
        "depth": config["depth"],
        "vmf_components": config["vmf_components"],
        "token_embedding_dim": config["token_embedding_dim"],
        "time_embedding_dim": config["time_embedding_dim"],
        "dropout": config["dropout"],
        "parameter_count": config["parameter_count"],
    }
    if task.stage == "train":
        replay = manifest["replays"][task.scene]
        spec.update({
            "replay_path": replay["path"],
            "replay_sha256": replay["sha256"],
            "training": protocol.training_config(manifest, task.config_id),
        })
    elif task.stage == "export":
        dependency_id = (
            f"{protocol.TASK_PREFIX}__{task.config_id}__{task.scene}"
            f"__train__t{task.model_seed}"
        )
        trained = _successful_result(state, "train", dependency_id)
        spec.update({
            "checkpoint_path": trained["checkpoint_path"],
            "checkpoint_sha256": trained["checkpoint_sha256"],
        })
    else:
        dependency_id = (
            f"{protocol.TASK_PREFIX}__{task.config_id}__{task.scene}"
            f"__export__t{task.model_seed}"
        )
        exported = _successful_result(state, "export", dependency_id)
        reference = protocol.reference_entry(manifest, task.scene)
        controls = manifest["render"]
        spec.update({
            "runtime_path": exported["runtime_path"],
            "runtime_sha256": exported["runtime_sha256"],
            "reference_path": reference["path"],
            "reference_sha256": reference["sha256"],
            "render": {
                **controls,
                "sample_count": task.sample_count,
            },
        })
    return spec


def _parse_worker_result(stdout: Path) -> dict[str, Any]:
    lines = [line for line in stdout.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        raise ControllerError("worker emitted no result")
    try:
        value = json.loads(lines[-1])
    except json.JSONDecodeError as exc:
        raise ControllerError("worker final stdout line is not JSON") from exc
    if not isinstance(value, dict):
        raise ControllerError("worker result must be an object")
    return value


def _run_worker(
    python: Path, spec_path: Path, attempt_root: Path, cpu_ids: tuple[int, ...],
    threads: int, timeout: float,
) -> dict[str, Any]:
    stdout = attempt_root / "worker.stdout.log"
    stderr = attempt_root / "worker.stderr.log"
    command = [
        "taskset", "--cpu-list", ",".join(map(str, cpu_ids)),
        str(python), "-B", str(PACKAGE_ROOT / "worker.py"),
        "--spec", str(spec_path), "--attempt-root", str(attempt_root),
    ]
    started = time.monotonic()
    try:
        with stdout.open("x", encoding="utf-8") as stdout_stream, stderr.open("x", encoding="utf-8") as stderr_stream:
            completed = subprocess.run(
                command, cwd=WORKSPACE_ROOT, env=_worker_env(threads),
                stdout=stdout_stream, stderr=stderr_stream, text=True,
                timeout=timeout, check=False,
            )
        return_code = completed.returncode
    except subprocess.TimeoutExpired:
        return_code = 124
        with stderr.open("a", encoding="utf-8") as stream:
            stream.write(f"\ncontroller timeout after {timeout:.1f}s\n")
    outcome: dict[str, Any] = {
        "return_code": return_code,
        "duration_seconds": time.monotonic() - started,
        "finished_at": now_utc(),
    }
    if return_code != 0:
        outcome.update({"status": "failed", "error": f"worker exited {return_code}"})
        return outcome
    try:
        result = _parse_worker_result(stdout)
        artifacts = _result_artifacts(result)
        result_path = attempt_root / "result.json"
        atomic_json(result_path, result)
        outcome.update({
            "status": "succeeded", "result": result, "artifacts": artifacts,
            "result_sha256": protocol.sha256_file(result_path),
        })
    except Exception as exc:
        outcome.update({"status": "failed", "error": f"invalid worker result: {type(exc).__name__}: {exc}"})
    return outcome


def _stage_slots(stage: str) -> tuple[tuple[int, ...], ...]:
    slots = protocol.training_slots() if stage == "train" else protocol.render_slots()
    protocol.assert_cpu_contract(slots, stage)
    return slots


def _stage_timeout(stage: str, manifest: dict[str, Any]) -> float:
    if stage == "render":
        return float(manifest["render"]["watchdog_seconds"]) + 300.0
    return 21600.0 if stage == "train" else 3600.0


def execute_stage(
    manifest: dict[str, Any], state: dict[str, Any], state_path: Path,
    host_id: str, stage: str, python: Path, *, rerun_stale: bool,
) -> None:
    stage_state = state["stages"][stage]
    tasks = protocol.build_tasks(manifest, host_id, stage)
    pending: list[protocol.Task] = []
    for task in tasks:
        record = stage_state["jobs"][task.task_id]
        if _valid_success(record):
            continue
        if record.get("status") == "succeeded" and not rerun_stale:
            raise ControllerError(f"artifact drift for {task.task_id}; pass --rerun-stale")
        if record.get("status") == "running":
            record["status"] = "interrupted"
        pending.append(task)
    if not pending:
        stage_state["status"] = "succeeded"
        state["updated_at"] = now_utc()
        atomic_json(state_path, state)
        print(f"{host_id}/{stage}: already complete ({len(tasks)}/{len(tasks)})", flush=True)
        return

    slots = _stage_slots(stage)
    prepared: list[tuple[protocol.Task, Path]] = []
    for task in pending:
        record = stage_state["jobs"][task.task_id]
        job_root = host_root(manifest, host_id) / "stages" / stage / "jobs" / task.task_id
        existing_numbers = []
        if job_root.is_dir():
            for path in job_root.iterdir():
                if path.is_dir() and path.name.startswith("attempt_"):
                    try:
                        existing_numbers.append(int(path.name.removeprefix("attempt_")))
                    except ValueError:
                        continue
        # An abrupt stop can leave a directory between mkdir and state commit.
        # Never reuse or delete it; advance monotonically to a fresh attempt.
        attempt_number = max([len(record["attempts"]) + 1, *(number + 1 for number in existing_numbers)])
        attempt_root = job_root / f"attempt_{attempt_number:03d}"
        attempt_root.mkdir(parents=True)
        record["status"] = "queued"
        record["attempts"].append({
            "attempt_number": attempt_number,
            "attempt_root": protocol.portable_workspace(attempt_root),
            "status": "queued",
        })
        prepared.append((task, attempt_root))
        state["updated_at"] = now_utc()
        atomic_json(state_path, state)

    stage_state["status"] = "running"
    stage_state.setdefault("started_at", now_utc())
    stage_state["worker_slots"] = len(slots)
    stage_state["allocated_cpu_ids"] = sorted({cpu for slot in slots for cpu in slot})
    stage_state["reserved_cpu_ids"] = list(protocol.SMT_SIBLING_CPU_IDS)
    if stage == "train":
        stage_state["unused_authorized_cpu_ids"] = list(protocol.TRAINING_UNUSED_CPU_IDS)
    state["updated_at"] = now_utc()
    atomic_json(state_path, state)

    # The provisional slots above only make specifications deterministic. The
    # queue below owns the actual exclusive CPU assignment for each process.
    actual_slots: Queue[tuple[int, ...]] = Queue()
    for slot in slots:
        actual_slots.put(slot)

    def one(task: protocol.Task, attempt_root: Path) -> tuple[str, tuple[int, ...], str, dict[str, Any]]:
        slot = actual_slots.get()
        try:
            spec = _worker_spec(manifest, state, task, attempt_root, len(slot))
            spec_path = attempt_root / "worker_spec.json"
            atomic_json(spec_path, spec)
            outcome = _run_worker(
                python, spec_path, attempt_root, slot, len(slot),
                _stage_timeout(stage, manifest),
            )
            return task.task_id, slot, protocol.sha256_file(spec_path), outcome
        finally:
            actual_slots.put(slot)

    failures: list[str] = []
    completed = 0
    with ThreadPoolExecutor(max_workers=len(slots), thread_name_prefix=f"p49-{stage}") as pool:
        futures: dict[Future[tuple[str, tuple[int, ...], str, dict[str, Any]]], protocol.Task] = {
            pool.submit(one, task, attempt_root): task
            for task, attempt_root in prepared
        }
        for future in as_completed(futures):
            task = futures[future]
            try:
                task_id, cpu_ids, spec_sha256, outcome = future.result()
            except Exception as exc:
                task_id, cpu_ids, spec_sha256 = task.task_id, (), None
                outcome = {"status": "failed", "error": f"{type(exc).__name__}: {exc}", "finished_at": now_utc()}
            record = stage_state["jobs"][task_id]
            record["attempts"][-1].update(outcome)
            record["attempts"][-1]["cpu_ids"] = list(cpu_ids)
            record["attempts"][-1]["spec_sha256"] = spec_sha256
            record["status"] = outcome["status"]
            append_jsonl(host_root(manifest, host_id) / "events.jsonl", {
                "event": f"{stage}_{outcome['status']}", "task_id": task_id,
                "attempt": record["attempts"][-1]["attempt_number"],
                "cpu_ids": list(cpu_ids), "at": now_utc(),
                "error": outcome.get("error"),
            })
            if outcome["status"] != "succeeded":
                failures.append(f"{task_id}: {outcome.get('error')}")
            completed += 1
            state["updated_at"] = now_utc()
            atomic_json(state_path, state)
            if completed % 25 == 0 or completed == len(prepared):
                print(json.dumps({
                    "host_id": host_id, "stage": stage,
                    "completed_this_run": completed,
                    "remaining_this_run": len(prepared) - completed,
                    "failures": len(failures),
                }, sort_keys=True), flush=True)

    succeeded = sum(_valid_success(record) for record in stage_state["jobs"].values())
    stage_state["status"] = "succeeded" if succeeded == len(tasks) else "failed"
    stage_state["counts"] = {"succeeded": succeeded, "total": len(tasks), "failed_this_run": len(failures)}
    stage_state["finished_at"] = now_utc()
    state["updated_at"] = now_utc()
    atomic_json(state_path, state)
    if stage_state["status"] != "succeeded":
        raise ControllerError(
            f"{host_id}/{stage} incomplete: {succeeded}/{len(tasks)}; rerun resumes completed tasks"
        )


def _metric_row(result: dict[str, Any]) -> dict[str, Any]:
    return {field: result.get(field) for field in METRIC_FIELDS}


def _training_metric_row(result: dict[str, Any]) -> dict[str, Any]:
    metrics = result.get("training_metrics", {})
    final = metrics.get("final", {})
    best = metrics.get("best", {})
    row = {
        "config_id": result.get("config_id"),
        "direction_conditioning_mode": result.get("direction_conditioning_mode"),
        "epochs": result.get("epochs"),
        "hidden_dim": result.get("hidden_dim"),
        "depth": result.get("depth"),
        "vmf_components": result.get("vmf_components"),
        "token_embedding_dim": result.get("token_embedding_dim"),
        "time_embedding_dim": result.get("time_embedding_dim"),
        "dropout": result.get("dropout"),
        "parameter_count": result.get("parameter_count"),
        "scene": result.get("scene"),
        "model_seed": result.get("model_seed"),
        "training_seconds": result.get("training_seconds"),
        "epochs_completed": result.get("epochs_completed"),
        "checkpoint_path": result.get("checkpoint_path"),
        "checkpoint_sha256": result.get("checkpoint_sha256"),
        "summary_path": result.get("summary_path"),
        "summary_sha256": result.get("summary_sha256"),
        "metrics_json_path": result.get("metrics_json_path"),
        "metrics_json_sha256": result.get("metrics_json_sha256"),
        "host_id": result.get("host_id"),
        "task_id": result.get("task_id"),
    }
    for prefix, values in (("final", final), ("best", best)):
        row[f"{prefix}_epoch"] = values.get("epoch")
        row[f"{prefix}_val_loss"] = values.get("val_loss")
        row[f"{prefix}_val_chain_length_loss"] = values.get("val_chain_length_loss")
        row[f"{prefix}_val_chain_type_loss"] = values.get("val_chain_type_loss")
        row[f"{prefix}_val_direction_loss"] = values.get("val_direction_loss")
    if any(row.get(field) is None for field in TRAINING_METRIC_FIELDS):
        raise ControllerError(f"incomplete training metric row: {result.get('task_id')}")
    return row


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    payload = "".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in rows)
    atomic_bytes(path, payload.encode("utf-8"))


def write_host_metrics(manifest: dict[str, Any], state: dict[str, Any], host_id: str) -> dict[str, Any]:
    render_rows = [
        _metric_row(record["attempts"][-1]["result"])
        for record in state["stages"]["render"]["jobs"].values()
        if _valid_success(record)
    ]
    render_rows.sort(key=lambda row: (row["config_id"], row["scene"], int(row["sample_count"]), row["run_id"]))
    expected = len(protocol.build_tasks(manifest, host_id, "render"))
    if len(render_rows) != expected:
        raise ControllerError(f"cannot write host metrics: {len(render_rows)}/{expected} valid rows")
    training_rows = [
        _training_metric_row(record["attempts"][-1]["result"])
        for record in state["stages"]["train"]["jobs"].values()
        if _valid_success(record)
    ]
    training_rows.sort(key=lambda row: (row["config_id"], row["scene"], int(row["model_seed"])))
    expected_training = len(protocol.build_tasks(manifest, host_id, "train"))
    if len(training_rows) != expected_training:
        raise ControllerError(
            f"cannot write host training metrics: {len(training_rows)}/{expected_training} valid rows"
        )
    metrics_root = host_root(manifest, host_id) / "metrics"
    render_jsonl = metrics_root / "render_rows.jsonl"
    render_csv = metrics_root / "render_rows.csv"
    training_jsonl = metrics_root / "training_rows.jsonl"
    training_csv = metrics_root / "training_rows.csv"
    _write_jsonl(render_jsonl, render_rows)
    _write_csv(render_csv, render_rows, METRIC_FIELDS)
    _write_jsonl(training_jsonl, training_rows)
    _write_csv(training_csv, training_rows, TRAINING_METRIC_FIELDS)
    return {
        "render": {
            "row_count": len(render_rows),
            "jsonl_path": protocol.portable_workspace(render_jsonl),
            "jsonl_sha256": protocol.sha256_file(render_jsonl),
            "csv_path": protocol.portable_workspace(render_csv),
            "csv_sha256": protocol.sha256_file(render_csv),
            "rows_sha256": protocol.canonical_sha256(render_rows),
        },
        "training": {
            "row_count": len(training_rows),
            "jsonl_path": protocol.portable_workspace(training_jsonl),
            "jsonl_sha256": protocol.sha256_file(training_jsonl),
            "csv_path": protocol.portable_workspace(training_csv),
            "csv_sha256": protocol.sha256_file(training_csv),
            "rows_sha256": protocol.canonical_sha256(training_rows),
        },
    }


def finish_host(manifest: dict[str, Any], state: dict[str, Any], state_path: Path, host_id: str) -> dict[str, Any]:
    if any(state["stages"][stage].get("status") != "succeeded" for stage in protocol.STAGES):
        raise ControllerError("cannot complete host before train/export/render all succeed")
    metrics = write_host_metrics(manifest, state, host_id)
    state["status"] = "complete"
    state["updated_at"] = now_utc()
    atomic_json(state_path, state)
    receipt = {
        "schema_version": 1,
        "protocol_id": protocol.PROTOCOL_ID,
        "status": "host_complete",
        "host_id": host_id,
        "manifest_sha256": state["manifest_sha256"],
        "input_snapshot_sha256": state["input_snapshot_sha256"],
        "task_counts": protocol.task_counts(manifest)["hosts"][host_id],
        "state_path": protocol.portable_workspace(state_path),
        "state_sha256": protocol.sha256_file(state_path),
        "metrics": metrics,
        "completed_at": now_utc(),
    }
    path = host_root(manifest, host_id) / "completion_receipt.json"
    atomic_json(path, receipt)
    return {**receipt, "receipt_sha256": protocol.sha256_file(path)}


def _resume_receipt(manifest: dict[str, Any], state: dict[str, Any], host_id: str) -> dict[str, Any]:
    counts: dict[str, Any] = {}
    for stage in protocol.STAGES:
        jobs = state["stages"][stage]["jobs"]
        succeeded = sum(_valid_success(record) for record in jobs.values())
        counts[stage] = {"valid_cached": succeeded, "pending": len(jobs) - succeeded}
    receipt = {
        "schema_version": 1, "protocol_id": protocol.PROTOCOL_ID,
        "status": "run_started_or_resumed", "host_id": host_id,
        "counts": counts, "pid": os.getpid(), "at": now_utc(),
    }
    root = host_root(manifest, host_id) / "resume_receipts"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    path = root / f"resume_{stamp}_{os.getpid()}.json"
    atomic_json(path, receipt)
    append_jsonl(host_root(manifest, host_id) / "events.jsonl", receipt)
    return {**receipt, "receipt_path": protocol.portable_workspace(path), "receipt_sha256": protocol.sha256_file(path)}


def status_payload(manifest: dict[str, Any], host_id: str) -> dict[str, Any]:
    state_path = host_root(manifest, host_id) / "state.json"
    if not state_path.is_file():
        return {
            "protocol_id": protocol.PROTOCOL_ID, "host_id": host_id,
            "status": "not_started", "expected": protocol.task_counts(manifest)["hosts"][host_id],
        }
    state = protocol.load_json(state_path)
    counts: dict[str, Any] = {}
    for stage in protocol.STAGES:
        jobs = state["stages"][stage]["jobs"]
        valid = sum(_valid_success(record) for record in jobs.values())
        counts[stage] = {"valid": valid, "total": len(jobs), "remaining": len(jobs) - valid}
    return {"protocol_id": protocol.PROTOCOL_ID, "host_id": host_id, "status": state.get("status"), "stages": counts}


def _summary_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, int], list[float]] = {}
    for row in rows:
        key = (str(row["config_id"]), str(row["scene"]), int(row["sample_count"]))
        groups.setdefault(key, []).append(float(row["mse"]))
    result = []
    for (config_id, scene, spp), values in sorted(groups.items()):
        if len(values) != 10:
            raise ControllerError(f"metric group does not contain D01--D10: {config_id}/{scene}/{spp}")
        result.append({
            "config_id": config_id, "scene": scene, "sample_count": spp,
            "n": len(values), "median_mse": statistics.median(values),
            "mean_mse": statistics.fmean(values), "sample_sd_mse": statistics.stdev(values),
        })
    return result


def collect(manifest: dict[str, Any]) -> dict[str, Any]:
    all_rows: list[dict[str, Any]] = []
    all_training_rows: list[dict[str, Any]] = []
    receipts: dict[str, str] = {}
    snapshots: set[str] = set()
    for host_id in protocol.HOST_IDS:
        receipt_path = host_root(manifest, host_id) / "completion_receipt.json"
        receipt = protocol.load_json(receipt_path)
        if (
            receipt.get("protocol_id") != protocol.PROTOCOL_ID
            or receipt.get("status") != "host_complete"
            or receipt.get("host_id") != host_id
            or receipt.get("manifest_sha256") != protocol.sha256_file(protocol.MANIFEST_PATH)
        ):
            raise ControllerError(f"invalid completion receipt: {host_id}")
        state_path = protocol.workspace_path(receipt["state_path"])
        if protocol.sha256_file(state_path) != receipt.get("state_sha256"):
            raise ControllerError(f"host state changed after completion: {host_id}")
        state = protocol.load_json(state_path)
        rows = [
            _metric_row(record["attempts"][-1]["result"])
            for record in state["stages"]["render"]["jobs"].values()
            if _valid_success(record)
        ]
        expected = protocol.task_counts(manifest)["hosts"][host_id]["render"]
        if len(rows) != expected:
            raise ControllerError(f"host render row count mismatch: {host_id}")
        rows.sort(key=lambda row: (row["config_id"], row["scene"], int(row["sample_count"]), row["run_id"]))
        if protocol.canonical_sha256(rows) != receipt["metrics"]["render"]["rows_sha256"]:
            raise ControllerError(f"host metrics digest changed: {host_id}")
        training_rows = [
            _training_metric_row(record["attempts"][-1]["result"])
            for record in state["stages"]["train"]["jobs"].values()
            if _valid_success(record)
        ]
        training_rows.sort(key=lambda row: (row["config_id"], row["scene"], int(row["model_seed"])))
        expected_training = protocol.task_counts(manifest)["hosts"][host_id]["train"]
        if len(training_rows) != expected_training:
            raise ControllerError(f"host training metric count mismatch: {host_id}")
        if protocol.canonical_sha256(training_rows) != receipt["metrics"]["training"]["rows_sha256"]:
            raise ControllerError(f"host training metrics digest changed: {host_id}")
        all_rows.extend(rows)
        all_training_rows.extend(training_rows)
        snapshots.add(str(receipt["input_snapshot_sha256"]))
        receipts[host_id] = protocol.sha256_file(receipt_path)
    if len(snapshots) != 1:
        raise ControllerError("the host completion receipt has an invalid input/code snapshot set")
    if len(all_rows) != 3500 or len({row["task_id"] for row in all_rows}) != 3500:
        raise ControllerError("sensitivity grid is not exactly 3,500 unique rows")
    if len(all_training_rows) != 140 or len({row["task_id"] for row in all_training_rows}) != 140:
        raise ControllerError("combined training metrics are not exactly 140 unique jobs")
    all_rows.sort(key=lambda row: (row["config_id"], row["scene"], int(row["sample_count"]), row["run_id"]))
    all_training_rows.sort(key=lambda row: (row["config_id"], row["scene"], int(row["model_seed"])))
    summaries = _summary_rows(all_rows)
    metrics_root = protocol.workspace_path(manifest["output_root"]) / "metrics"
    rows_jsonl = metrics_root / "sensitivity_rows.jsonl"
    rows_csv = metrics_root / "sensitivity_rows.csv"
    summary_csv = metrics_root / "sensitivity_summary.csv"
    training_jsonl = metrics_root / "training_rows.jsonl"
    training_csv = metrics_root / "training_rows.csv"
    _write_jsonl(rows_jsonl, all_rows)
    _write_csv(rows_csv, all_rows, METRIC_FIELDS)
    summary_fields = ("config_id", "scene", "sample_count", "n", "median_mse", "mean_mse", "sample_sd_mse")
    _write_csv(summary_csv, summaries, summary_fields)
    _write_jsonl(training_jsonl, all_training_rows)
    _write_csv(training_csv, all_training_rows, TRAINING_METRIC_FIELDS)
    receipt = {
        "schema_version": 1,
        "protocol_id": protocol.PROTOCOL_ID,
        "status": "complete",
        "manifest_sha256": protocol.sha256_file(protocol.MANIFEST_PATH),
        "input_snapshot_sha256": next(iter(snapshots)),
        "host_completion_receipts": receipts,
        "new_row_count": 3500,
        "combined_row_count": len(all_rows),
        "training_job_metric_count": len(all_training_rows),
        "recorded_runs_interpretation": manifest["recorded_runs_interpretation"],
        "summary_group_count": len(summaries),
        "metrics": {
            "jsonl": {"path": protocol.portable_workspace(rows_jsonl), "sha256": protocol.sha256_file(rows_jsonl)},
            "csv": {"path": protocol.portable_workspace(rows_csv), "sha256": protocol.sha256_file(rows_csv)},
            "summary_csv": {"path": protocol.portable_workspace(summary_csv), "sha256": protocol.sha256_file(summary_csv)},
            "training_jsonl": {"path": protocol.portable_workspace(training_jsonl), "sha256": protocol.sha256_file(training_jsonl)},
            "training_csv": {"path": protocol.portable_workspace(training_csv), "sha256": protocol.sha256_file(training_csv)},
        },
        "completed_at": now_utc(),
    }
    receipt_path = protocol.workspace_path(manifest["output_root"]) / "completion_receipt.json"
    atomic_json(receipt_path, receipt)
    return {**receipt, "receipt_sha256": protocol.sha256_file(receipt_path)}


def run_host(args: argparse.Namespace, manifest: dict[str, Any]) -> dict[str, Any]:
    if not args.execute or args.authorization_token != manifest["authorization_token"]:
        raise ControllerError("exact Phase49 execution token and --execute are required")
    # Keep the configured virtual-environment path rather than resolving its
    # interpreter symlink. Python uses that path to locate the environment's
    # site-packages; preflight separately compares resolved identities.
    worker_python = args.python.absolute()
    preflight_result = preflight(manifest, args.host_id, worker_python, write_receipt=True)
    root = host_root(manifest, args.host_id)
    with exclusive_lock(root / "coordinator.lock"):
        state_path, state = _load_state(manifest, args.host_id, preflight_result["input_snapshot"])
        resume = _resume_receipt(manifest, state, args.host_id)
        stages = protocol.STAGES if args.stage == "all" else (args.stage,)
        for stage in stages:
            prior = protocol.STAGES[:protocol.STAGES.index(stage)]
            if any(state["stages"][name].get("status") != "succeeded" for name in prior):
                raise ControllerError(f"stage {stage} requires completed stages: {prior}")
            execute_stage(
                manifest, state, state_path, args.host_id, stage,
                worker_python, rerun_stale=args.rerun_stale,
            )
        if all(state["stages"][stage].get("status") == "succeeded" for stage in protocol.STAGES):
            result = finish_host(manifest, state, state_path, args.host_id)
            result["resume_receipt"] = resume
            return result
        return {"protocol_id": protocol.PROTOCOL_ID, "status": "stage_complete", "host_id": args.host_id, "resume_receipt": resume}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("plan")
    for command in ("preflight", "status"):
        child = subparsers.add_parser(command)
        child.add_argument("--host-id", choices=protocol.HOST_IDS, required=True)
        if command == "preflight":
            child.add_argument("--python", type=Path, default=Path(sys.executable))
    run = subparsers.add_parser("run")
    run.add_argument("--host-id", choices=protocol.HOST_IDS, required=True)
    run.add_argument("--python", type=Path, default=Path(sys.executable))
    run.add_argument("--stage", choices=("all", *protocol.STAGES), default="all")
    run.add_argument("--rerun-stale", action="store_true")
    run.add_argument("--execute", action="store_true")
    run.add_argument("--authorization-token")
    subparsers.add_parser("collect")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = protocol.load_manifest()
    if args.command == "plan":
        print(json.dumps({
            "protocol_id": protocol.PROTOCOL_ID,
            "plan": protocol.load_json(protocol.PLAN_PATH),
            "computed_task_counts": protocol.task_counts(manifest),
        }, indent=2, sort_keys=True))
        return 0
    if args.command == "preflight":
        result = preflight(manifest, args.host_id, args.python.absolute(), write_receipt=True)
    elif args.command == "status":
        result = status_payload(manifest, args.host_id)
    elif args.command == "run":
        result = run_host(args, manifest)
    else:
        result = collect(manifest)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ControllerError, protocol.ProtocolError) as exc:
        print(json.dumps({
            "protocol_id": protocol.PROTOCOL_ID,
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
        }, indent=2, sort_keys=True), file=sys.stderr)
        raise SystemExit(1)
