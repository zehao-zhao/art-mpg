#!/usr/bin/env python3
"""Crash-safe coordinator for the preregistered MPGD SPP--MSE sweep.

This successor never trains a model or renders a reference.  It verifies and
reuses the frozen Phase-20 model exports, references, and 64-SPP observations,
then renders only the preregistered 1--32 SPP grid with the frozen Phase-20
worker.  Every mutable output lives below a new, isolated result root.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import json
import math
import os
import platform
import shutil
import statistics
import subprocess
import sys
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


PACKAGE_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = PACKAGE_ROOT.parent
PROJECT_ROOT = WORKSPACE_ROOT.parent
SCRIPTS_ROOT = WORKSPACE_ROOT / "scripts"
SRC_ROOT = WORKSPACE_ROOT / "src"
for entry in (str(PACKAGE_ROOT), str(SCRIPTS_ROOT), str(SRC_ROOT)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

import protocol  # noqa: E402


class SweepError(RuntimeError):
    """Raised on any fail-closed successor condition."""


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def atomic_json(path: Path, payload: object) -> None:
    atomic_bytes(path, protocol.json_bytes(payload))


def immutable_bytes(path: Path, payload: bytes, label: str) -> str:
    if path.exists():
        if path.read_bytes() != payload:
            raise SweepError(f"refusing to overwrite changed {label}: {path}")
    else:
        atomic_bytes(path, payload)
    return protocol.sha256_file(path)


def immutable_json(path: Path, payload: object, label: str) -> str:
    return immutable_bytes(path, protocol.json_bytes(payload), label)


def tree_hashes(root: Path) -> dict[str, str]:
    if not root.is_dir():
        return {}
    return {
        path.relative_to(root).as_posix(): protocol.sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def verify_tree(root: Path, expected: object) -> bool:
    return isinstance(expected, dict) and bool(expected) and tree_hashes(root) == expected


@contextmanager
def coordinator_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        handle.close()
        raise SweepError(f"another successor coordinator holds {path}") from exc
    try:
        yield
    finally:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def _scene_tree_digest(scene_xml: Path) -> dict[str, Any]:
    root = scene_xml.parent
    files = [
        path
        for path in root.rglob("*")
        if path.is_file() and path.name != ".DS_Store" and ".phase" not in path.name
    ]
    rows = {
        path.relative_to(root).as_posix(): protocol.sha256_file(path)
        for path in sorted(files)
    }
    return {"file_count": len(rows), "tree_sha256": protocol.sha256_json(rows)}


def _verify_file(path: Path, expected_sha: object, label: str) -> str:
    if not path.is_file() or path.is_symlink():
        raise SweepError(f"missing or symlinked frozen {label}: {path}")
    actual = protocol.sha256_file(path)
    if actual != expected_sha:
        raise SweepError(
            f"frozen {label} changed: expected {expected_sha}, observed {actual}: {path}"
        )
    return actual


def _resolve_project_artifact(value: object, label: str) -> Path:
    path = Path(str(value))
    result = path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()
    try:
        result.relative_to(PROJECT_ROOT.resolve())
    except ValueError as exc:
        raise SweepError(f"{label} escapes the project: {value}") from exc
    return result


def validate_package_receipt() -> dict[str, Any]:
    path = PACKAGE_ROOT / "preregistration_receipt.json"
    receipt = protocol.load_json(path)
    if (
        receipt.get("protocol_id") != protocol.PROTOCOL_ID
        or receipt.get("status") != "package_frozen_before_execution"
        or int(receipt.get("new_render_jobs", 0)) != 540
        or int(receipt.get("reused_64spp_rows_if_all_provenance_checks_pass", 0)) != 90
    ):
        raise SweepError("preregistration receipt identity changed")
    files = receipt.get("package_files")
    if not isinstance(files, dict) or not files:
        raise SweepError("preregistration package inventory is missing")
    observed: dict[str, str] = {}
    for relative, expected in sorted(files.items()):
        candidate = (PACKAGE_ROOT / relative).resolve()
        try:
            candidate.relative_to(PACKAGE_ROOT.resolve())
        except ValueError as exc:
            raise SweepError(f"preregistration path escapes package: {relative}") from exc
        observed[relative] = _verify_file(candidate, expected, f"package file {relative}")
    return {
        "receipt_sha256": protocol.sha256_file(path),
        "files": observed,
    }


def _source_row_key(row: dict[str, Any]) -> tuple[str, str, int | None, int]:
    model_seed = row.get("model_seed")
    return (
        str(row.get("scene")),
        str(row.get("method")),
        None if model_seed is None else int(model_seed),
        int(row.get("render_seed", -1)),
    )


def select_and_validate_reused_rows(
    manifest: dict[str, Any], source_results: dict[str, Any], *, verify_images: bool
) -> list[dict[str, Any]]:
    if source_results.get("protocol_id") != protocol.SOURCE_PROTOCOL_ID:
        raise SweepError("source results protocol mismatch")
    if int(source_results.get("sample_count", 0)) != protocol.REUSED_SAMPLE_COUNT:
        raise SweepError("source results are not the frozen 64-SPP evaluation")
    if tuple(source_results.get("render_seeds", ())) != protocol.RENDER_SEEDS:
        raise SweepError("source render seeds changed")
    if tuple(source_results.get("model_seeds", ())) != protocol.MODEL_SEEDS:
        raise SweepError("source model seeds changed")
    if source_results.get("selected_controls") != manifest["source_phase20"]["selected_controls"]:
        raise SweepError("source MPGD controls changed")
    rows = source_results.get("rows")
    if not isinstance(rows, list):
        raise SweepError("source results rows are missing")

    selected: dict[tuple[str, str, int | None, int], dict[str, Any]] = {}
    for raw in rows:
        if not isinstance(raw, dict):
            raise SweepError("source result row is not an object")
        if raw.get("scene") not in protocol.SCENES or raw.get("method") not in protocol.METHODS:
            continue
        key = _source_row_key(raw)
        if key in selected:
            raise SweepError(f"duplicate source row: {key}")
        selected[key] = raw

    expected: set[tuple[str, str, int | None, int]] = set()
    for scene in protocol.SCENES:
        for method in protocol.METHODS:
            model_seeds: tuple[int | None, ...] = (
                protocol.MODEL_SEEDS if method in protocol.LEARNED_METHODS else (None,)
            )
            for model_seed in model_seeds:
                for render_seed in protocol.RENDER_SEEDS:
                    expected.add((scene, method, model_seed, render_seed))
    if set(selected) != expected:
        missing = sorted(expected - set(selected), key=str)
        extra = sorted(set(selected) - expected, key=str)
        raise SweepError(f"source 64-SPP row identity mismatch; missing={missing}, extra={extra}")

    validated: list[dict[str, Any]] = []
    for key in sorted(selected, key=str):
        row = selected[key]
        if int(row.get("sample_count", 0)) != protocol.REUSED_SAMPLE_COUNT:
            raise SweepError(f"source row is not 64 SPP: {key}")
        for metric in ("mse", "psnr", "reference_peak", "render_seconds"):
            value = float(row.get(metric, math.nan))
            if not math.isfinite(value) or (metric == "mse" and value < 0.0):
                raise SweepError(f"invalid source {metric}: {key}")
        output = _resolve_project_artifact(row.get("output_path"), f"source image {key}")
        if verify_images:
            _verify_file(output, row.get("output_sha256"), f"64-SPP image {key}")
        validated.append(dict(row))
    if len(validated) != protocol.expected_reused_rows():
        raise SweepError("wrong number of hash-verified 64-SPP rows")
    return validated


def validate_source_state_binding(
    manifest: dict[str, Any], source_state: dict[str, Any], rows: list[dict[str, Any]]
) -> None:
    """Bind each reused receipt row to its exact successful Phase-20 worker result."""
    source = manifest["source_phase20"]
    if source_state.get("protocol_id") != protocol.SOURCE_PROTOCOL_ID:
        raise SweepError("source coordinator protocol mismatch")
    if source_state.get("status") != "complete":
        raise SweepError("source coordinator did not complete")
    if source_state.get("manifest_sha256") != source["manifest"]["sha256"]:
        raise SweepError("source coordinator manifest hash mismatch")
    if source_state.get("final_results_sha256") != source["results_receipt"]["sha256"]:
        raise SweepError("source coordinator final-result hash mismatch")
    if source_state.get("selection_override") != source["selected_controls"]:
        raise SweepError("source coordinator selected controls changed")
    jobs = source_state.get("stages", {}).get("primary", {}).get("jobs", {})
    if not isinstance(jobs, dict):
        raise SweepError("source coordinator primary stage is missing")
    for row in rows:
        scene = str(row["scene"])
        method = str(row["method"])
        render_seed = int(row["render_seed"])
        model_seed = row.get("model_seed")
        model_token = "" if model_seed is None else f"__t{int(model_seed)}"
        task_id = f"primary__{scene}__{method}{model_token}__r{render_seed}"
        job = jobs.get(task_id)
        if not isinstance(job, dict) or job.get("status") != "succeeded":
            raise SweepError(f"source coordinator job is not successful: {task_id}")
        attempts = job.get("attempts")
        if not isinstance(attempts, list) or not attempts:
            raise SweepError(f"source coordinator attempt is missing: {task_id}")
        result = attempts[-1].get("result")
        if not isinstance(result, dict):
            raise SweepError(f"source coordinator result is missing: {task_id}")
        if (
            result.get("output_path") != row.get("output_path")
            or result.get("output_sha256") != row.get("output_sha256")
            or result.get("logical_method") != method
            or result.get("render_seed") != render_seed
            or result.get("model_seed") != model_seed
        ):
            raise SweepError(f"source coordinator/result row mismatch: {task_id}")
        if method in protocol.LEARNED_METHODS:
            runtime = manifest["model_exports"][f"{scene}|{method}|{int(model_seed)}"]
            if (
                result.get("runtime_path") != runtime["path"]
                or result.get("runtime_sha256") != runtime["sha256"]
            ):
                raise SweepError(f"source learned runtime mismatch: {task_id}")
        elif result.get("runtime_path") is not None or result.get("runtime_sha256") is not None:
            raise SweepError(f"source nonlearned row unexpectedly used a runtime: {task_id}")


def validate_static_inputs(
    manifest: dict[str, Any], *, require_linux: bool, verify_reused_images: bool = True
) -> dict[str, Any]:
    if require_linux and platform.system() != "Linux":
        raise SweepError("render execution is Linux-server-only")

    source = manifest["source_phase20"]
    receipts: dict[str, str] = {}
    payloads: dict[str, dict[str, Any]] = {}
    for name in (
        "manifest",
        "coordinator_state",
        "results_receipt",
        "reference_registry",
        "npm_step_selection_receipt",
    ):
        entry = source[name]
        path = protocol.workspace_path(entry["path"], name)
        receipts[name] = _verify_file(path, entry["sha256"], name)
        payloads[name] = protocol.load_json(path)

    source_manifest = payloads["manifest"]
    if source_manifest.get("protocol_id") != protocol.SOURCE_PROTOCOL_ID:
        raise SweepError("frozen source manifest protocol changed")
    primary = source_manifest.get("primary", {})
    if (
        int(primary.get("sample_count", 0)) != 64
        or int(primary.get("width", 0)) != 64
        or int(primary.get("height", 0)) != 64
        or tuple(primary.get("render_seeds", ())) != protocol.RENDER_SEEDS
        or tuple(source_manifest.get("model_seeds", ())) != protocol.MODEL_SEEDS
    ):
        raise SweepError("source Phase-20 evaluation identity changed")
    if not set(protocol.METHODS).issubset(set(primary.get("methods", ()))):
        raise SweepError("source Phase-20 report lacks a preregistered method")
    if source_manifest.get("render_controls") != {
        key: manifest["render_controls"][key]
        for key in (
            "threads_per_worker",
            "samples_per_pass",
            "diffusion_steps",
            "learned_direction_flip_to_receiver_hemisphere",
            "learned_proposal_inv_prob_clamp",
            "learned_proposal_contribution_clamp",
            "learned_proposal_max_trials",
            "mpg_train_budget",
            "mpg_train_fusion",
        )
    }:
        raise SweepError("successor render controls no longer match Phase 20")

    source_results = payloads["results_receipt"]
    if (
        source_results.get("reference_registry_sha256")
        != source["reference_registry"]["sha256"]
    ):
        raise SweepError("source results/reference-registry binding changed")
    reused_rows = select_and_validate_reused_rows(
        manifest, source_results, verify_images=verify_reused_images
    )
    validate_source_state_binding(manifest, payloads["coordinator_state"], reused_rows)
    npm_receipt = payloads["npm_step_selection_receipt"]
    if (
        npm_receipt.get("protocol_id") != protocol.SOURCE_PROTOCOL_ID
        or int(npm_receipt.get("selected_steps", 0))
        != manifest["render_controls"]["npm_selected_training_steps"]
    ):
        raise SweepError("frozen NPM renderer selection changed")

    registry = payloads["reference_registry"]
    if registry.get("protocol_id") != protocol.SOURCE_PROTOCOL_ID:
        raise SweepError("reference registry protocol changed")
    reference_hashes: dict[str, str] = {}
    for scene in protocol.SCENES:
        expected = source["references"][scene]
        path = protocol.workspace_path(expected["path"], f"reference {scene}")
        reference_hashes[scene] = _verify_file(path, expected["sha256"], f"reference {scene}")
        entry = registry.get("references", {}).get(scene)
        if not isinstance(entry, dict):
            raise SweepError(f"reference registry lacks {scene}")
        registry_path = _resolve_project_artifact(entry.get("exr_path"), f"registry reference {scene}")
        if (
            registry_path != path.resolve()
            or entry.get("exr_sha256") != expected["sha256"]
            or entry.get("active_config_sha256") != expected["active_config_sha256"]
            or int(entry.get("sample_count", 0)) != int(manifest["reference_sample_count"])
            or int(entry.get("sampler_seed", -1)) != int(manifest["reference_seed"])
        ):
            raise SweepError(f"reference registry entry changed: {scene}")

    code_hashes: dict[str, str] = {}
    for name, entry in manifest["frozen_code"].items():
        path = protocol.workspace_path(entry["path"], f"code {name}")
        code_hashes[name] = _verify_file(path, entry["sha256"], f"code {name}")
    runtime_hashes: dict[str, str] = {}
    for name, entry in manifest["frozen_runtime"].items():
        path = protocol.workspace_path(entry["path"], f"runtime {name}")
        runtime_hashes[name] = _verify_file(path, entry["sha256"], f"runtime {name}")

    scene_hashes: dict[str, Any] = {}
    for scene in protocol.SCENES:
        entry = manifest["scenes"][scene]
        path = protocol.project_path(entry["scene_path"], f"scene {scene}")
        _verify_file(path, entry["scene_xml_sha256"], f"scene XML {scene}")
        tree = _scene_tree_digest(path)
        if tree != {
            "file_count": entry["scene_tree_file_count"],
            "tree_sha256": entry["scene_tree_sha256"],
        }:
            raise SweepError(f"scene dependency tree changed: {scene}")
        scene_hashes[scene] = {
            "scene_xml_sha256": entry["scene_xml_sha256"],
            **tree,
        }

    export_hashes: dict[str, str] = {}
    for key, entry in sorted(manifest["model_exports"].items()):
        path = protocol.workspace_path(entry["path"], f"model export {key}")
        export_hashes[key] = _verify_file(path, entry["sha256"], f"model export {key}")

    successor_code = {
        path.name: protocol.sha256_file(path)
        for path in (
            PACKAGE_ROOT / "protocol.py",
            PACKAGE_ROOT / "run_sweep.py",
            PACKAGE_ROOT / "plot_results.py",
        )
    }

    return {
        "schema_version": 1,
        "protocol_id": protocol.PROTOCOL_ID,
        "source_protocol_id": protocol.SOURCE_PROTOCOL_ID,
        "source_receipts": receipts,
        "source_64spp_rows": len(reused_rows),
        "source_64spp_rows_sha256": protocol.sha256_json(reused_rows),
        "references": reference_hashes,
        "model_exports": export_hashes,
        "code": code_hashes,
        "runtime": runtime_hashes,
        "scenes": scene_hashes,
        "successor_code": successor_code,
        "preregistration_package": validate_package_receipt(),
    }


def _physical_cpu_representatives() -> list[int]:
    allowed = (
        sorted(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else list(range(os.cpu_count() or 1))
    )
    if platform.system() != "Linux":
        return allowed
    representatives: list[int] = []
    seen: set[tuple[str, str]] = set()
    for cpu in allowed:
        topology = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
        try:
            key = (
                (topology / "physical_package_id").read_text().strip(),
                (topology / "core_id").read_text().strip(),
            )
        except OSError as exc:
            raise SweepError(f"cannot read CPU topology for cpu{cpu}") from exc
        if key not in seen:
            seen.add(key)
            representatives.append(cpu)
    return representatives


def cpu_slots(workers: int, threads: int) -> tuple[tuple[int, ...], ...]:
    physical = _physical_cpu_representatives()
    if workers <= 0 or threads <= 0 or workers * threads > len(physical):
        raise SweepError(
            f"workers*threads must be in [1,{len(physical)}] physical cores"
        )
    chosen = physical[: workers * threads]
    return tuple(
        tuple(chosen[index * threads : (index + 1) * threads])
        for index in range(workers)
    )


def python_probe(python: Path, threads: int) -> dict[str, Any]:
    if not python.is_file() or not os.access(python, os.X_OK):
        raise SweepError(f"Python is not executable: {python}")
    env = dict(os.environ)
    env.update({
        "OMP_NUM_THREADS": str(threads),
        "MKL_NUM_THREADS": str(threads),
        "OPENBLAS_NUM_THREADS": str(threads),
        "NUMEXPR_NUM_THREADS": str(threads),
        "OPENCV_IO_ENABLE_OPENEXR": "1",
        "CUDA_VISIBLE_DEVICES": "",
    })
    code = (
        "import json,platform,torch,numpy,cv2;"
        "print(json.dumps({'python':platform.python_version(),'torch':torch.__version__,"
        "'numpy':numpy.__version__,'opencv':cv2.__version__,"
        "'cuda':torch.cuda.is_available()}))"
    )
    completed = subprocess.run(
        [str(python), "-c", code], cwd=WORKSPACE_ROOT, env=env,
        capture_output=True, text=True, timeout=60, check=False,
    )
    if completed.returncode != 0:
        raise SweepError(f"Python dependency probe failed: {completed.stderr.strip()}")
    try:
        payload = json.loads(completed.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError) as exc:
        raise SweepError("Python dependency probe returned invalid JSON") from exc
    if payload.get("cuda") is not False:
        raise SweepError("the frozen CPU protocol unexpectedly exposes CUDA")
    return payload


def build_worker_spec(manifest: dict[str, Any], task: protocol.Task, threads: int) -> dict[str, Any]:
    controls = manifest["render_controls"]
    spec: dict[str, Any] = {
        # The exact frozen Phase-20 worker validates this source protocol ID.
        "protocol_id": protocol.SOURCE_PROTOCOL_ID,
        "successor_protocol_id": protocol.PROTOCOL_ID,
        "task_id": task.task_id,
        "kind": "render",
        "scene": task.scene,
        "job_id": manifest["scenes"][task.scene]["job_id"],
        "scene_path": manifest["scenes"][task.scene]["scene_path"],
        "threads": threads,
        "model_seed": task.model_seed,
        "logical_method": task.method,
        "phase13_method": {
            "diffusion_mlp": "masked_runtime",
            "npm_adapted": "npm_adapted_seed_direction",
            "mpg_original": "mpg",
            "pt": "pt",
            "sms_star": "sms_star",
        }[task.method],
        "render_seed": task.render_seed,
        "control_id": None,
        "bounce_alpha": (
            controls["diffusion_bounce_alpha"]
            if task.method == "diffusion_mlp"
            else controls["npm_bounce_alpha"]
            if task.method == "npm_adapted"
            else 0.0
        ),
        "kappa_scale": (
            controls["diffusion_kappa_scale"]
            if task.method == "diffusion_mlp"
            else controls["npm_kappa_scale"]
            if task.method == "npm_adapted"
            else 1.0
        ),
        "render": {
            key: controls[key]
            for key in (
                "samples_per_pass",
                "diffusion_steps",
                "learned_direction_flip_to_receiver_hemisphere",
                "learned_proposal_inv_prob_clamp",
                "learned_proposal_contribution_clamp",
                "learned_proposal_max_trials",
                "mpg_train_budget",
                "mpg_train_fusion",
                "watchdog_seconds",
            )
        },
    }
    spec["render"].update({
        "width": manifest["width"],
        "height": manifest["height"],
        "sample_count": task.sample_count,
    })
    if task.method in protocol.LEARNED_METHODS:
        key = f"{task.scene}|{task.method}|{task.model_seed}"
        runtime = manifest["model_exports"][key]
        spec.update({
            "runtime_path": runtime["path"],
            "runtime_sha256": runtime["sha256"],
        })
    return spec


def _run_prepared(
    *, python: Path, worker: Path, spec_path: Path, attempt_root: Path,
    stdout_path: Path, stderr_path: Path, cpus: tuple[int, ...],
    threads: int, timeout: float,
) -> dict[str, Any]:
    command = [
        str(python), "-B", str(worker), "--spec", str(spec_path),
        "--attempt-root", str(attempt_root),
    ]
    taskset = shutil.which("taskset")
    if platform.system() == "Linux":
        if not taskset:
            raise SweepError("taskset is required on Linux")
        command = [taskset, "--cpu-list", ",".join(str(cpu) for cpu in cpus), *command]
    env = dict(os.environ)
    for name in (
        "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS",
    ):
        env[name] = str(threads)
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["OPENCV_IO_ENABLE_OPENEXR"] = "1"
    started = time.monotonic()
    try:
        with stdout_path.open("x", encoding="utf-8") as stdout, stderr_path.open(
            "x", encoding="utf-8"
        ) as stderr:
            completed = subprocess.run(
                command, cwd=WORKSPACE_ROOT, env=env, stdout=stdout, stderr=stderr,
                text=True, timeout=timeout, check=False,
            )
        return_code = completed.returncode
    except subprocess.TimeoutExpired:
        return_code = 124
        with stderr_path.open("a", encoding="utf-8") as stderr:
            stderr.write(f"\ncoordinator timeout after {timeout:.1f}s\n")
    outcome: dict[str, Any] = {
        "return_code": return_code,
        "duration_seconds": time.monotonic() - started,
        "finished_at": now_utc(),
    }
    if return_code != 0:
        outcome.update({"status": "failed", "error": f"worker exited {return_code}"})
        return outcome
    try:
        lines = [line for line in stdout_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        payload = json.loads(lines[-1])
        if not isinstance(payload, dict):
            raise ValueError("last worker line is not an object")
        outcome.update({"status": "succeeded", "result": payload})
    except (OSError, IndexError, json.JSONDecodeError, ValueError) as exc:
        outcome.update({"status": "failed", "error": f"invalid worker result: {exc}"})
    return outcome


def initial_state(
    manifest: dict[str, Any], manifest_sha: str, inputs: dict[str, Any], probe: dict[str, Any]
) -> dict[str, Any]:
    timestamp = now_utc()
    tasks = protocol.build_tasks(manifest)
    task_digest = protocol.sha256_json([task.payload() for task in tasks])
    return {
        "schema_version": 1,
        "protocol_id": protocol.PROTOCOL_ID,
        "manifest_sha256": manifest_sha,
        "input_snapshot": inputs,
        "python": probe,
        "tasks_sha256": task_digest,
        "created_at": timestamp,
        "updated_at": timestamp,
        "status": "running",
        "jobs": {
            task.task_id: {
                "definition": task.payload(),
                "definition_sha256": protocol.sha256_json(task.payload()),
                "status": "pending",
                "attempts": [],
            }
            for task in tasks
        },
    }


def load_or_create_state(
    path: Path, *, manifest_sha: str, inputs: dict[str, Any], probe: dict[str, Any],
    manifest: dict[str, Any],
) -> dict[str, Any]:
    tasks = protocol.build_tasks(manifest)
    if not path.exists():
        state = initial_state(manifest, manifest_sha, inputs, probe)
        atomic_json(path, state)
        return state
    state = protocol.load_json(path)
    expected = {
        "protocol_id": protocol.PROTOCOL_ID,
        "manifest_sha256": manifest_sha,
        "input_snapshot": inputs,
        "python": probe,
        "tasks_sha256": protocol.sha256_json([task.payload() for task in tasks]),
    }
    for key, value in expected.items():
        if state.get(key) != value:
            raise SweepError(f"resume identity mismatch: {key}")
    if set(state.get("jobs", {})) != {task.task_id for task in tasks}:
        raise SweepError("resume task set changed")
    return state


def classify_job(record: dict[str, Any]) -> str:
    if record.get("status") != "succeeded":
        return str(record.get("status", "pending"))
    attempts = record.get("attempts", [])
    if not attempts:
        return "stale"
    attempt = attempts[-1]
    try:
        root = protocol.workspace_path(attempt["attempt_root"])
    except (KeyError, protocol.ProtocolError):
        return "stale"
    return "succeeded" if verify_tree(root, attempt.get("artifact_sha256")) else "stale"


def _verify_worker_result(task: protocol.Task, result: dict[str, Any]) -> None:
    expected = {
        "kind": "render",
        "scene": task.scene,
        "logical_method": task.method,
        "render_seed": task.render_seed,
        "model_seed": task.model_seed,
    }
    for key, value in expected.items():
        if result.get(key) != value:
            raise SweepError(f"worker result identity mismatch for {task.task_id}: {key}")
    output = _resolve_project_artifact(result.get("output_path"), task.task_id)
    _verify_file(output, result.get("output_sha256"), f"render output {task.task_id}")


def execute_tasks(
    *, manifest: dict[str, Any], state: dict[str, Any], state_path: Path,
    output_root: Path, python: Path, workers: int, threads: int, rerun_stale: bool,
) -> None:
    tasks = protocol.build_tasks(manifest)
    pending: list[protocol.Task] = []
    for task in tasks:
        record = state["jobs"][task.task_id]
        status = classify_job(record)
        if status == "succeeded":
            continue
        if status == "stale" and not rerun_stale:
            raise SweepError(f"artifact drift for {task.task_id}; pass --rerun-stale")
        if status == "running":
            record["status"] = "interrupted"
        pending.append(task)
    if not pending:
        print(f"render grid already complete ({len(tasks)}/{len(tasks)})", flush=True)
        return

    slots = cpu_slots(min(workers, len(pending)), threads)
    state["status"] = "rendering"
    state.setdefault("render_started_at", now_utc())
    state["workers"] = len(slots)
    state["threads_per_worker"] = threads
    state["allocated_physical_cores"] = len(slots) * threads
    state["updated_at"] = now_utc()
    atomic_json(state_path, state)

    worker = protocol.workspace_path(manifest["frozen_code"]["source_worker"]["path"])
    queued = iter(pending)
    active: dict[Future[dict[str, Any]], tuple[protocol.Task, int, Path]] = {}
    failed = False

    def submit(slot_index: int, executor: ThreadPoolExecutor) -> bool:
        nonlocal failed
        if failed:
            return False
        try:
            task = next(queued)
        except StopIteration:
            return False
        record = state["jobs"][task.task_id]
        attempt_number = len(record["attempts"]) + 1
        attempt_root = output_root / "jobs" / task.task_id / f"attempt_{attempt_number:03d}"
        if attempt_root.exists():
            raise SweepError(f"attempt directory already exists: {attempt_root}")
        attempt_root.mkdir(parents=True)
        spec = build_worker_spec(manifest, task, threads)
        spec_path = attempt_root / "worker_spec.json"
        atomic_json(spec_path, spec)
        stdout_path = attempt_root / "worker.stdout.log"
        stderr_path = attempt_root / "worker.stderr.log"
        attempt = {
            "attempt_number": attempt_number,
            "attempt_root": protocol.portable_workspace(attempt_root),
            "spec_sha256": protocol.sha256_file(spec_path),
            "cpu_ids": list(slots[slot_index]),
            "started_at": now_utc(),
            "status": "running",
        }
        record["attempts"].append(attempt)
        record["status"] = "running"
        state["updated_at"] = now_utc()
        atomic_json(state_path, state)
        future = executor.submit(
            _run_prepared,
            python=python,
            worker=worker,
            spec_path=spec_path,
            attempt_root=attempt_root,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            cpus=slots[slot_index],
            threads=threads,
            timeout=float(manifest["render_controls"]["watchdog_seconds"]) + 300.0,
        )
        active[future] = (task, slot_index, attempt_root)
        return True

    with ThreadPoolExecutor(max_workers=len(slots), thread_name_prefix="phase23-spp") as executor:
        for index in range(len(slots)):
            if not submit(index, executor):
                break
        while active:
            finished, _ = wait(tuple(active), return_when=FIRST_COMPLETED)
            for future in finished:
                task, slot_index, attempt_root = active.pop(future)
                outcome = future.result()
                if outcome.get("status") == "succeeded":
                    try:
                        _verify_worker_result(task, outcome["result"])
                    except SweepError as exc:
                        outcome = {
                            **outcome,
                            "status": "failed",
                            "error": str(exc),
                        }
                outcome["artifact_sha256"] = tree_hashes(attempt_root)
                record = state["jobs"][task.task_id]
                record["attempts"][-1].update(outcome)
                record["status"] = outcome["status"]
                if outcome["status"] != "succeeded":
                    failed = True
                state["updated_at"] = now_utc()
                atomic_json(state_path, state)
                print(f"{task.task_id}: {outcome['status']}", flush=True)
                submit(slot_index, executor)

    counts = {
        name: sum(classify_job(record) == name for record in state["jobs"].values())
        for name in ("succeeded", "failed", "pending", "interrupted", "stale")
    }
    state["counts"] = counts
    state["status"] = "render_complete" if counts["succeeded"] == len(tasks) else "failed"
    state["render_finished_at"] = now_utc()
    state["updated_at"] = now_utc()
    atomic_json(state_path, state)
    if state["status"] != "render_complete":
        raise SweepError(f"render grid incomplete: {counts}")


def _successful_result(state: dict[str, Any], task: protocol.Task) -> dict[str, Any]:
    record = state.get("jobs", {}).get(task.task_id)
    if not isinstance(record, dict) or classify_job(record) != "succeeded":
        raise SweepError(f"task incomplete or stale: {task.task_id}")
    result = record["attempts"][-1].get("result")
    if not isinstance(result, dict):
        raise SweepError(f"task result missing: {task.task_id}")
    _verify_worker_result(task, result)
    return result


def _geomean(values: list[float]) -> float:
    if not values or any(not math.isfinite(value) or value < 0.0 for value in values):
        raise SweepError("geometric mean requires finite nonnegative values")
    return math.exp(math.fsum(math.log(max(value, 1e-300)) for value in values) / len(values))


def _quartiles(values: list[float]) -> tuple[float, float]:
    if len(values) < 2:
        return values[0], values[0]
    quartiles = statistics.quantiles(values, n=4, method="inclusive")
    return quartiles[0], quartiles[2]


def _csv_bytes(rows: list[dict[str, Any]], fields: list[str]) -> bytes:
    import io

    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def finalize(
    manifest: dict[str, Any], manifest_sha: str, state: dict[str, Any], output_root: Path
) -> dict[str, Any]:
    inputs = validate_static_inputs(manifest, require_linux=True, verify_reused_images=True)
    if state.get("manifest_sha256") != manifest_sha or state.get("input_snapshot") != inputs:
        raise SweepError("inputs changed between rendering and finalization")
    tasks = protocol.build_tasks(manifest)
    if any(classify_job(state["jobs"][task.task_id]) != "succeeded" for task in tasks):
        raise SweepError("cannot finalize before all new renders succeed")

    import phase_13_run_formal_benchmark as phase13

    source_results = protocol.load_json(
        protocol.workspace_path(manifest["source_phase20"]["results_receipt"]["path"])
    )
    reused = select_and_validate_reused_rows(manifest, source_results, verify_images=True)
    rows: list[dict[str, Any]] = []
    for row in reused:
        rows.append({
            "scene": row["scene"],
            "method": row["method"],
            "public_method": manifest["public_method_names"][row["method"]],
            "sample_count": 64,
            "model_seed": row.get("model_seed"),
            "render_seed": row["render_seed"],
            "mse": row["mse"],
            "psnr": row["psnr"],
            "reference_peak": row["reference_peak"],
            "render_seconds": row["render_seconds"],
            "output_path": row["output_path"],
            "output_sha256": row["output_sha256"],
            "provenance": "hash_verified_phase20_64spp_reuse",
        })

    for task in tasks:
        result = _successful_result(state, task)
        prediction = _resolve_project_artifact(result["output_path"], task.task_id)
        reference_entry = manifest["source_phase20"]["references"][task.scene]
        reference = protocol.workspace_path(reference_entry["path"], f"reference {task.scene}")
        mse, psnr, peak = phase13.compute_metrics(prediction, reference)
        if not all(math.isfinite(value) for value in (mse, psnr, peak)) or mse < 0.0:
            raise SweepError(f"invalid metric result: {task.task_id}")
        rows.append({
            "scene": task.scene,
            "method": task.method,
            "public_method": manifest["public_method_names"][task.method],
            "sample_count": task.sample_count,
            "model_seed": task.model_seed,
            "render_seed": task.render_seed,
            "mse": mse,
            "psnr": psnr,
            "reference_peak": peak,
            "render_seconds": result["record"]["seconds"],
            "output_path": result["output_path"],
            "output_sha256": result["output_sha256"],
            "provenance": "new_successor_render",
        })
    rows.sort(key=lambda row: (
        protocol.SCENES.index(row["scene"]),
        protocol.METHODS.index(row["method"]),
        int(row["sample_count"]),
        -1 if row["model_seed"] is None else int(row["model_seed"]),
        int(row["render_seed"]),
    ))
    expected_total = len(tasks) + protocol.expected_reused_rows()
    if len(rows) != expected_total:
        raise SweepError(f"wrong final row count: {len(rows)} != {expected_total}")

    summary_rows: list[dict[str, Any]] = []
    for scene in protocol.SCENES:
        for method in protocol.METHODS:
            for spp in (*protocol.NEW_SAMPLE_COUNTS, protocol.REUSED_SAMPLE_COUNT):
                values = [
                    float(row["mse"])
                    for row in rows
                    if row["scene"] == scene
                    and row["method"] == method
                    and int(row["sample_count"]) == spp
                ]
                expected_n = 15 if method in protocol.LEARNED_METHODS else 5
                if len(values) != expected_n:
                    raise SweepError(
                        f"wrong replicate count for {scene}/{method}/{spp}: {len(values)}"
                    )
                q1, q3 = _quartiles(values)
                summary_rows.append({
                    "scene": scene,
                    "method": method,
                    "public_method": manifest["public_method_names"][method],
                    "sample_count": spp,
                    "replicates": len(values),
                    "mean_mse": statistics.fmean(values),
                    "median_mse": statistics.median(values),
                    "sample_std_mse": statistics.stdev(values),
                    "geometric_mean_mse": _geomean(values),
                    "q25_mse": q1,
                    "q75_mse": q3,
                    "min_mse": min(values),
                    "max_mse": max(values),
                })

    report = {
        "schema_version": 1,
        "protocol_id": protocol.PROTOCOL_ID,
        "status": "complete",
        "source_protocol_id": protocol.SOURCE_PROTOCOL_ID,
        "metric": manifest["metric"],
        "scenes": list(protocol.SCENES),
        "methods": list(protocol.METHODS),
        "public_method_names": manifest["public_method_names"],
        "sample_counts": [*protocol.NEW_SAMPLE_COUNTS, protocol.REUSED_SAMPLE_COUNT],
        "render_seeds": list(protocol.RENDER_SEEDS),
        "model_seeds": list(protocol.MODEL_SEEDS),
        "new_render_rows": len(tasks),
        "reused_64spp_rows": protocol.expected_reused_rows(),
        "total_rows": len(rows),
        "rows": rows,
        "summary": summary_rows,
    }
    reports = output_root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    report_path = reports / "spp_mse_results.json"
    report_sha = immutable_json(report_path, report, "SPP--MSE report")

    row_fields = list(rows[0])
    rows_path = reports / "spp_mse_rows.csv"
    rows_sha = immutable_bytes(rows_path, _csv_bytes(rows, row_fields), "SPP--MSE rows CSV")
    summary_fields = list(summary_rows[0])
    summary_path = reports / "spp_mse_summary.csv"
    summary_sha = immutable_bytes(
        summary_path, _csv_bytes(summary_rows, summary_fields), "SPP--MSE summary CSV"
    )

    plotter = PACKAGE_ROOT / "plot_results.py"
    completed = subprocess.run(
        [sys.executable, "-B", str(plotter), "--report", str(report_path),
         "--output-dir", str(reports)],
        cwd=WORKSPACE_ROOT,
        env={**os.environ, "MPLBACKEND": "Agg"},
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    if completed.returncode != 0:
        raise SweepError(f"plot generation failed: {completed.stderr.strip()}")
    plot_paths = [reports / "spp_mse_decay.pdf", reports / "spp_mse_decay.svg"]
    plot_artifacts = {
        path.name: {
            "path": protocol.portable_workspace(path),
            "sha256": protocol.sha256_file(path),
            "size": path.stat().st_size,
        }
        for path in plot_paths
        if path.is_file()
    }
    if len(plot_artifacts) != 2:
        raise SweepError("plot generator did not produce both PDF and SVG")

    completion = {
        "schema_version": 1,
        "protocol_id": protocol.PROTOCOL_ID,
        "status": "complete",
        "manifest_sha256": manifest_sha,
        "counts": {
            "new_render_jobs": len(tasks),
            "reused_64spp_rows": protocol.expected_reused_rows(),
            "total_metric_rows": len(rows),
            "summary_points": len(summary_rows),
        },
        "artifacts": {
            report_path.name: {
                "path": protocol.portable_workspace(report_path),
                "sha256": report_sha,
                "size": report_path.stat().st_size,
            },
            rows_path.name: {
                "path": protocol.portable_workspace(rows_path),
                "sha256": rows_sha,
                "size": rows_path.stat().st_size,
            },
            summary_path.name: {
                "path": protocol.portable_workspace(summary_path),
                "sha256": summary_sha,
                "size": summary_path.stat().st_size,
            },
            **plot_artifacts,
        },
    }
    completion_path = reports / "completion_receipt.json"
    immutable_json(completion_path, completion, "completion receipt")
    seal_sha = protocol.sha256_file(completion_path)
    immutable_bytes(
        reports / "completion_seal.sha256",
        f"{seal_sha}  completion_receipt.json\n".encode("utf-8"),
        "completion seal",
    )
    return {**completion, "completion_receipt_sha256": seal_sha}


def status_payload(manifest: dict[str, Any]) -> dict[str, Any]:
    root = protocol.workspace_path(manifest["output_root"])
    state_path = root / "coordinator_state.json"
    if not state_path.is_file():
        return {
            "protocol_id": protocol.PROTOCOL_ID,
            "status": "not_started",
            "new_render_jobs": len(protocol.build_tasks(manifest)),
            "reused_64spp_rows": protocol.expected_reused_rows(),
        }
    state = protocol.load_json(state_path)
    counts = {
        name: sum(classify_job(record) == name for record in state.get("jobs", {}).values())
        for name in ("succeeded", "failed", "pending", "running", "interrupted", "stale")
    }
    return {
        "protocol_id": protocol.PROTOCOL_ID,
        "status": state.get("status"),
        "counts": counts,
        "updated_at": state.get("updated_at"),
        "completion_receipt": state.get("completion_receipt"),
        "completion_receipt_sha256": state.get("completion_receipt_sha256"),
    }


def print_plan(manifest: dict[str, Any]) -> None:
    learned_per_scene_spp = len(protocol.LEARNED_METHODS) * len(protocol.MODEL_SEEDS) * len(protocol.RENDER_SEEDS)
    nonlearned_per_scene_spp = (
        len(protocol.METHODS) - len(protocol.LEARNED_METHODS)
    ) * len(protocol.RENDER_SEEDS)
    print(json.dumps({
        "protocol_id": protocol.PROTOCOL_ID,
        "classification": manifest["classification"],
        "server_only": True,
        "scenes": list(protocol.SCENES),
        "methods": manifest["public_method_names"],
        "new_sample_counts": list(protocol.NEW_SAMPLE_COUNTS),
        "reused_sample_count": protocol.REUSED_SAMPLE_COUNT,
        "render_seeds": list(protocol.RENDER_SEEDS),
        "model_seeds_for_learned_methods": list(protocol.MODEL_SEEDS),
        "new_jobs_per_scene_per_spp": learned_per_scene_spp + nonlearned_per_scene_spp,
        "new_render_jobs": len(protocol.build_tasks(manifest)),
        "reused_64spp_rows": protocol.expected_reused_rows(),
        "final_metric_rows": len(protocol.build_tasks(manifest)) + protocol.expected_reused_rows(),
        "renders_references_or_trains_models": False,
    }, indent=2, sort_keys=True))


def run_pipeline(args: argparse.Namespace) -> int:
    manifest, manifest_sha = protocol.load_and_validate_manifest(args.manifest)
    if args.threads_per_worker != manifest["render_controls"]["threads_per_worker"]:
        raise SweepError("threads-per-worker must remain the frozen value 4")
    if not args.execute or args.authorization_token != manifest["execution_authorization_token"]:
        raise SweepError("run requires the exact execution flag and authorization token")
    inputs = validate_static_inputs(manifest, require_linux=True, verify_reused_images=True)
    python = Path(os.path.abspath(os.fspath(args.python)))
    probe = python_probe(python, args.threads_per_worker)
    root = protocol.workspace_path(manifest["output_root"])
    state_path = root / "coordinator_state.json"
    lock_path = protocol.workspace_path(manifest["global_lock_path"])
    with coordinator_lock(lock_path):
        state = load_or_create_state(
            state_path,
            manifest_sha=manifest_sha,
            inputs=inputs,
            probe=probe,
            manifest=manifest,
        )
        execute_tasks(
            manifest=manifest,
            state=state,
            state_path=state_path,
            output_root=root,
            python=python,
            workers=args.workers,
            threads=args.threads_per_worker,
            rerun_stale=args.rerun_stale,
        )
        completion = finalize(manifest, manifest_sha, state, root)
        state["status"] = "complete"
        state["completion_receipt"] = protocol.portable_workspace(
            root / "reports" / "completion_receipt.json"
        )
        state["completion_receipt_sha256"] = completion["completion_receipt_sha256"]
        state["updated_at"] = now_utc()
        atomic_json(state_path, state)
    print(json.dumps(completion, indent=2, sort_keys=True))
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "preflight", "run", "status", "finalize"))
    parser.add_argument("--manifest", type=Path, default=protocol.DEFAULT_MANIFEST)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--threads-per-worker", type=int, default=4)
    parser.add_argument("--rerun-stale", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--authorization-token")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        manifest, manifest_sha = protocol.load_and_validate_manifest(args.manifest)
        if args.command == "plan":
            print_plan(manifest)
            return 0
        if args.command == "status":
            print(json.dumps(status_payload(manifest), indent=2, sort_keys=True))
            return 0
        if args.command == "preflight":
            inputs = validate_static_inputs(manifest, require_linux=True, verify_reused_images=True)
            slots = cpu_slots(args.workers, args.threads_per_worker)
            probe = python_probe(Path(os.path.abspath(os.fspath(args.python))), args.threads_per_worker)
            print(json.dumps({
                "protocol_id": protocol.PROTOCOL_ID,
                "status": "ready",
                "manifest_sha256": manifest_sha,
                "python": probe,
                "new_render_jobs": len(protocol.build_tasks(manifest)),
                "reused_64spp_rows": protocol.expected_reused_rows(),
                "workers": len(slots),
                "threads_per_worker": args.threads_per_worker,
                "allocated_physical_cores": sum(len(slot) for slot in slots),
            }, indent=2, sort_keys=True))
            return 0
        if args.command == "finalize":
            root = protocol.workspace_path(manifest["output_root"])
            state = protocol.load_json(root / "coordinator_state.json")
            completion = finalize(manifest, manifest_sha, state, root)
            print(json.dumps(completion, indent=2, sort_keys=True))
            return 0
        return run_pipeline(args)
    except (SweepError, protocol.ProtocolError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
