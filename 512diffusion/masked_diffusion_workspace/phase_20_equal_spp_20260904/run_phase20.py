#!/usr/bin/env python3
"""Crash-safe coordinator and finalizer for Phase 20 equal-render-SPP runs."""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
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
from masked_diffusion_mpg.manifest import collect_missing_assets  # noqa: E402


class Phase20Error(RuntimeError):
    """Raised for a fail-closed Phase 20 condition."""


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(protocol.json_bytes(payload))
    os.replace(temporary, path)


def immutable_json(path: Path, payload: object, label: str) -> str:
    if path.exists():
        if protocol.load_json(path) != payload:
            raise Phase20Error(f"refusing to overwrite changed {label}: {path}")
    else:
        atomic_json(path, payload)
    return protocol.sha256_file(path)


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
        raise Phase20Error(f"another Phase 20 coordinator holds {path}") from exc
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


def validate_static_inputs(manifest: dict[str, Any], *, require_linux: bool) -> dict[str, Any]:
    if require_linux and platform.system() != "Linux":
        raise Phase20Error("Phase 20 execution is Linux-server-only; Mac execution is disabled")
    legacy_reference = protocol.workspace_path(manifest["legacy_reference_manifest"])
    legacy_state = protocol.workspace_path(manifest["legacy_diffusion_state"])
    npm_receipt = protocol.workspace_path(manifest["npm_step_selection_receipt"])
    phase19_selection = protocol.workspace_path(manifest["phase19_selection"])
    if protocol.sha256_file(legacy_reference) != manifest["legacy_reference_manifest_sha256"]:
        raise Phase20Error("legacy reference manifest hash changed")
    if protocol.sha256_file(legacy_state) != manifest["legacy_diffusion_state_sha256"]:
        raise Phase20Error("legacy Diffusion-MLP state hash changed")
    if protocol.sha256_file(npm_receipt) != manifest["npm_step_selection_receipt_sha256"]:
        raise Phase20Error("NPM step-tuning receipt hash changed")
    if protocol.sha256_file(phase19_selection) != manifest["phase19_selection_sha256"]:
        raise Phase20Error("Phase 19 final-v3 selection hash changed")
    inputs: dict[str, Any] = {
        "manifest_sha256": protocol.sha256_file(protocol.DEFAULT_MANIFEST),
        "code": {},
        "runtime": {},
        "scenes": {},
    }
    code_paths = {
        "coordinator": Path(__file__).resolve(),
        "worker": PACKAGE_ROOT / "worker.py",
        "protocol": PACKAGE_ROOT / "protocol.py",
        "phase13": protocol.workspace_path(manifest["phase13_path"]),
        "phase09": protocol.workspace_path(manifest["phase09_path"]),
        "diffusion_train": protocol.workspace_path(manifest["diffusion_train_path"]),
        "runtime_export": protocol.workspace_path(manifest["runtime_export_path"]),
        "npm_train": protocol.workspace_path(manifest["npm_train_path"]),
        "npm_module": SRC_ROOT / "masked_diffusion_mpg" / "npm_adapted.py",
        "tail_diagnostics": SRC_ROOT / "masked_diffusion_mpg" / "tail_diagnostics.py",
        "runtime_header": PROJECT_ROOT / "mitsuba" / "src" / "integrators" / "MPG" / "masked_diffusion_runtime.h",
    }
    for name, path in code_paths.items():
        if not path.is_file():
            raise Phase20Error(f"missing code input {name}: {path}")
        inputs["code"][name] = protocol.sha256_file(path)

    executable = protocol.project_path(manifest["runtime"]["mitsuba_executable"])
    plugin = protocol.project_path(manifest["runtime"]["mpg_plugin"])
    for name, path in (("mitsuba", executable), ("mpg_plugin", plugin)):
        if not path.is_file():
            raise Phase20Error(f"missing runtime artifact {name}: {path}")
        inputs["runtime"][name] = protocol.sha256_file(path)
    if require_linux and b"npm_adapted" not in plugin.read_bytes():
        raise Phase20Error(
            "server MPG plugin was not rebuilt with NPM-adapted runtime support"
        )

    for scene, entry in manifest["scenes"].items():
        scene_xml = protocol.project_path(entry["scene_path"])
        if not scene_xml.is_file():
            raise Phase20Error(f"missing scene XML: {scene_xml}")
        missing = collect_missing_assets(scene_xml)
        if missing:
            raise Phase20Error(f"scene {scene} has missing assets: {missing}")
        inputs["scenes"][scene] = {
            "scene_xml_sha256": protocol.sha256_file(scene_xml),
            **_scene_tree_digest(scene_xml),
        }
    inputs["legacy_reference_manifest"] = protocol.sha256_file(legacy_reference)
    inputs["legacy_diffusion_state"] = protocol.sha256_file(legacy_state)
    inputs["npm_step_selection_receipt"] = protocol.sha256_file(npm_receipt)
    inputs["phase19_selection"] = protocol.sha256_file(phase19_selection)
    inputs["legacy_replays"] = {
        scene: protocol.sha256_file(
            protocol.workspace_path(manifest["scenes"][scene]["legacy_replay_path"])
        )
        for scene in protocol.EXISTING_SCENES
    }
    inputs["legacy_diffusion_runtimes"] = {
        f"{scene}|t{seed}": _legacy_diffusion_runtime(manifest, scene, seed)[
            "runtime_sha256"
        ]
        for scene in protocol.EXISTING_SCENES
        for seed in manifest["model_seeds"]
    }
    return inputs


def _physical_cpu_representatives() -> list[int]:
    allowed = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else list(range(os.cpu_count() or 1))
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
            raise Phase20Error(f"cannot read CPU topology for cpu{cpu}") from exc
        if key not in seen:
            seen.add(key)
            representatives.append(cpu)
    return representatives


def cpu_slots(workers: int, threads: int) -> tuple[tuple[int, ...], ...]:
    physical = _physical_cpu_representatives()
    if workers <= 0 or threads <= 0 or workers * threads > len(physical):
        raise Phase20Error(
            f"workers*threads must be in [1,{len(physical)}] physical cores"
        )
    selected = physical[: workers * threads]
    return tuple(
        tuple(selected[index * threads : (index + 1) * threads])
        for index in range(workers)
    )


def _python_probe(python: Path, threads: int) -> dict[str, Any]:
    if not python.is_file() or not os.access(python, os.X_OK):
        raise Phase20Error(f"Python is not executable: {python}")
    env = dict(os.environ)
    env.update({
        "OMP_NUM_THREADS": str(threads),
        "MKL_NUM_THREADS": str(threads),
        "OPENBLAS_NUM_THREADS": str(threads),
        "NUMEXPR_NUM_THREADS": str(threads),
        "OPENCV_IO_ENABLE_OPENEXR": "1",
    })
    code = (
        "import json,platform,torch,numpy,cv2;"
        "print(json.dumps({'python':platform.python_version(),'torch':torch.__version__,"
        "'numpy':numpy.__version__,'opencv':cv2.__version__,"
        "'cuda':torch.cuda.is_available()}))"
    )
    completed = subprocess.run(
        [str(python), "-c", code],
        cwd=WORKSPACE_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    if completed.returncode != 0:
        raise Phase20Error(f"Python dependency probe failed: {completed.stderr.strip()}")
    try:
        return json.loads(completed.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError) as exc:
        raise Phase20Error("Python dependency probe returned invalid JSON") from exc


def initial_state(
    manifest_sha: str,
    inputs: dict[str, Any],
    python_probe: dict[str, Any],
    selection_override: dict[str, Any] | None,
) -> dict[str, Any]:
    timestamp = now_utc()
    return {
        "schema_version": 1,
        "protocol_id": protocol.PROTOCOL_ID,
        "manifest_sha256": manifest_sha,
        "input_snapshot": inputs,
        "python": python_probe,
        "selection_override": selection_override,
        "created_at": timestamp,
        "updated_at": timestamp,
        "status": "running",
        "stages": {},
    }


def load_or_create_state(
    path: Path,
    *,
    manifest_sha: str,
    inputs: dict[str, Any],
    python_probe: dict[str, Any],
    selection_override: dict[str, Any] | None,
) -> dict[str, Any]:
    if not path.exists():
        state = initial_state(manifest_sha, inputs, python_probe, selection_override)
        atomic_json(path, state)
        return state
    state = protocol.load_json(path)
    expected = {
        "protocol_id": protocol.PROTOCOL_ID,
        "manifest_sha256": manifest_sha,
        "input_snapshot": inputs,
        "python": python_probe,
        "selection_override": selection_override,
    }
    for key, value in expected.items():
        if state.get(key) != value:
            raise Phase20Error(f"resume identity mismatch: {key}")
    return state


def ensure_stage(state: dict[str, Any], manifest: dict[str, Any], stage: str) -> dict[str, Any]:
    tasks = protocol.build_tasks(manifest, stage)
    digest = protocol.sha256_json([task.payload() for task in tasks])
    stages = state.setdefault("stages", {})
    if stage not in stages:
        stages[stage] = {
            "status": "pending",
            "tasks_sha256": digest,
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
    record = stages[stage]
    if record.get("tasks_sha256") != digest or set(record.get("jobs", {})) != {
        task.task_id for task in tasks
    }:
        raise Phase20Error(f"stage definition changed: {stage}")
    return record


def successful_result(state: dict[str, Any], stage: str, task_id: str) -> dict[str, Any]:
    record = state.get("stages", {}).get(stage, {}).get("jobs", {}).get(task_id)
    if not isinstance(record, dict) or record.get("status") != "succeeded":
        raise Phase20Error(f"dependency is incomplete: {stage}/{task_id}")
    attempts = record.get("attempts", [])
    if not attempts:
        raise Phase20Error(f"dependency has no attempt: {stage}/{task_id}")
    attempt = attempts[-1]
    root = protocol.workspace_path(attempt["attempt_root"])
    if not verify_tree(root, attempt.get("artifact_sha256")):
        raise Phase20Error(f"dependency artifacts changed: {stage}/{task_id}")
    result = attempt.get("result")
    if not isinstance(result, dict):
        raise Phase20Error(f"dependency result missing: {stage}/{task_id}")
    return result


def _legacy_diffusion_runtime(
    manifest: dict[str, Any], scene: str, model_seed: int
) -> dict[str, str]:
    state_path = protocol.workspace_path(manifest["legacy_diffusion_state"])
    state = protocol.load_json(state_path)
    key = f"final:{scene}:ordered_mlp_current:seed{model_seed}:export"
    job = state.get("jobs", {}).get(key)
    if not isinstance(job, dict) or job.get("status") != "succeeded":
        raise Phase20Error(f"legacy Diffusion-MLP runtime missing: {key}")
    result = job.get("result", {})
    path = protocol.workspace_path(result.get("runtime_path"), "legacy runtime")
    digest = result.get("runtime_sha256")
    if not path.is_file() or protocol.sha256_file(path) != digest:
        raise Phase20Error(f"legacy Diffusion-MLP runtime hash mismatch: {scene}")
    return {"runtime_path": protocol.portable_workspace(path), "runtime_sha256": digest}


def _selection_override(
    manifest: dict[str, Any], *, selection_json: Path | None,
    bounce_alpha: float | None, kappa_scale: float | None,
) -> dict[str, Any] | None:
    if selection_json is not None and (bounce_alpha is not None or kappa_scale is not None):
        raise Phase20Error("use either --selection-json or explicit alpha/kappa, not both")
    if (bounce_alpha is None) != (kappa_scale is None):
        raise Phase20Error("--bounce-alpha and --kappa-scale must be supplied together")
    if selection_json is not None:
        payload = protocol.load_json(selection_json.resolve())
        chosen = payload.get("selected_controls", payload.get("selected", payload))
        if not isinstance(chosen, dict):
            raise Phase20Error("selection JSON lacks selected_controls")
        alpha = float(chosen["bounce_alpha"])
        kappa = float(chosen["kappa_scale"])
        source_sha = protocol.sha256_file(selection_json.resolve())
        source = str(selection_json.resolve())
    elif bounce_alpha is not None and kappa_scale is not None:
        alpha, kappa = float(bounce_alpha), float(kappa_scale)
        source_sha, source = None, "cli"
    else:
        source_path = protocol.workspace_path(manifest["phase19_selection"])
        payload = protocol.load_json(source_path)
        chosen = payload.get("selected")
        if not isinstance(chosen, dict) or chosen.get("eligible") is not True:
            raise Phase20Error("Phase 19 final-v3 has no eligible selected controls")
        alpha = float(chosen["bounce_alpha"])
        kappa = float(chosen["kappa_scale"])
        source_sha = protocol.sha256_file(source_path)
        source = protocol.portable_workspace(source_path)
    if not 0.0 <= alpha <= 1.0 or not math.isfinite(kappa) or kappa < 0.0:
        raise Phase20Error("invalid runtime control selection")
    return {
        "control_id": f"a{round(alpha * 100):03d}_k{round(kappa * 100):03d}",
        "bounce_alpha": alpha,
        "kappa_scale": kappa,
        "source": source,
        "source_sha256": source_sha,
    }


def _selected_controls(
    manifest: dict[str, Any], state: dict[str, Any], output_root: Path
) -> dict[str, Any]:
    override = state.get("selection_override")
    if isinstance(override, dict):
        return override
    path = output_root / "selection" / "selected_controls.json"
    if not path.is_file():
        raise Phase20Error("64-SPP control selection has not completed")
    payload = protocol.load_json(path)
    if payload.get("protocol_id") != protocol.PROTOCOL_ID:
        raise Phase20Error("selected-controls protocol mismatch")
    selected = payload.get("selected_controls")
    if not isinstance(selected, dict):
        raise Phase20Error("selected-controls payload is invalid")
    return selected


def _replay_dependency(
    manifest: dict[str, Any], state: dict[str, Any], scene: str
) -> dict[str, Any]:
    if scene in protocol.EXISTING_SCENES:
        entry = manifest["scenes"][scene]
        path = protocol.workspace_path(entry["legacy_replay_path"], "legacy replay")
        digest = str(entry["legacy_replay_sha256"])
        if not path.is_file() or protocol.sha256_file(path) != digest:
            raise Phase20Error(f"legacy replay changed or is missing: {scene}")
        return {
            "replay_path": protocol.portable_workspace(path),
            "replay_sha256": digest,
            "source": "reused_hash_bound_phase09_aggregate",
        }
    return successful_result(state, "replay", f"replay__{scene}")


def _runtime_dependency(
    manifest: dict[str, Any], state: dict[str, Any], scene: str,
    method: str, model_seed: int,
) -> dict[str, str]:
    if method == "diffusion_mlp" and scene in protocol.EXISTING_SCENES:
        return _legacy_diffusion_runtime(manifest, scene, model_seed)
    if method == "diffusion_mlp":
        result = successful_result(
            state, "export_diffusion_new", f"export_diffusion__{scene}__t{model_seed}"
        )
    elif method == "npm_adapted":
        result = successful_result(
            state, "export_npm", f"export_npm__{scene}__t{model_seed}"
        )
    else:
        raise Phase20Error(f"method has no runtime dependency: {method}")
    return {
        "runtime_path": str(result["runtime_path"]),
        "runtime_sha256": str(result["runtime_sha256"]),
    }


def _selected_npm_steps(state: dict[str, Any], output_root: Path) -> int:
    path = output_root / "selection" / "selected_npm_steps.json"
    if not path.is_file():
        raise Phase20Error("NPM 64-SPP renderer gate has not completed")
    payload = protocol.load_json(path)
    if payload.get("protocol_id") != protocol.PROTOCOL_ID:
        raise Phase20Error("NPM step selection protocol mismatch")
    steps = int(payload.get("selected_steps", 0))
    if steps <= 0:
        raise Phase20Error("NPM step selection is invalid")
    return steps


def build_worker_spec(
    manifest: dict[str, Any], state: dict[str, Any], task: protocol.Task,
    output_root: Path, threads: int,
) -> dict[str, Any]:
    scene = manifest["scenes"][task.scene]
    spec: dict[str, Any] = {
        "protocol_id": protocol.PROTOCOL_ID,
        "task_id": task.task_id,
        "kind": task.kind,
        "scene": task.scene,
        "job_id": scene["job_id"],
        "scene_path": scene["scene_path"],
        "threads": threads,
        "model_seed": task.model_seed,
    }
    if task.kind == "replay":
        spec["replay"] = {
            **manifest["replay"],
            "width": manifest["primary"]["width"],
            "height": manifest["primary"]["height"],
            "samples_per_pass": manifest["render_controls"]["samples_per_pass"],
        }
    elif task.kind in {"train_diffusion", "train_npm"}:
        replay = _replay_dependency(manifest, state, task.scene)
        training = dict(manifest[
            "diffusion_training" if task.kind == "train_diffusion" else "npm_training"
        ])
        if task.kind == "train_npm":
            training["max_steps"] = (
                int(str(task.control_id).removeprefix("steps"))
                if task.stage == "train_npm_gate"
                else _selected_npm_steps(state, output_root)
            )
        spec.update({
            "replay_path": replay["replay_path"],
            "replay_sha256": replay["replay_sha256"],
            "training": training,
        })
    elif task.kind == "export_npm_gate":
        steps = int(str(task.control_id).removeprefix("steps"))
        trained = successful_result(
            state,
            "train_npm_gate",
            f"train_npm_gate__{task.scene}__steps{steps}",
        )
        spec.update({
            "kind": "export_npm",
            "method": "npm_adapted",
            "checkpoint_path": trained["checkpoint_path"],
            "checkpoint_sha256": trained["checkpoint_sha256"],
        })
    elif task.kind in {"export_diffusion", "export_npm"}:
        train_stage = "train_diffusion_new" if task.kind == "export_diffusion" else "train_npm"
        train_kind = "train_diffusion" if task.kind == "export_diffusion" else "train_npm"
        trained = successful_result(
            state, train_stage, f"{train_kind}__{task.scene}__t{task.model_seed}"
        )
        spec.update({
            "method": task.method,
            "checkpoint_path": trained["checkpoint_path"],
            "checkpoint_sha256": trained["checkpoint_sha256"],
        })
    elif task.kind in {"reference", "render"}:
        controls = manifest["render_controls"]
        if task.kind == "reference":
            render = {
                **controls,
                "width": manifest["primary"]["width"],
                "height": manifest["primary"]["height"],
                "sample_count": manifest["reference"]["sample_count"],
                "watchdog_seconds": manifest["reference"]["watchdog_seconds"],
            }
            spec.update({
                "logical_method": "path_reference",
                "phase13_method": "path_reference",
                "render_seed": task.render_seed,
                "bounce_alpha": 0.0,
                "kappa_scale": 1.0,
                "render": render,
            })
        else:
            if task.stage == "npm_step_validation":
                selected = {"bounce_alpha": 1.0, "kappa_scale": 1.0}
            else:
                selected = _selected_controls(manifest, state, output_root)
            render = {
                **controls,
                "width": manifest["primary"]["width"],
                "height": manifest["primary"]["height"],
                "sample_count": (
                    manifest["npm_render_gate"]["sample_count"]
                    if task.stage == "npm_step_validation"
                    else manifest["primary"]["sample_count"]
                ),
                "watchdog_seconds": manifest["primary"]["watchdog_seconds"],
            }
            spec.update({
                "logical_method": task.method,
                "phase13_method": {
                    "diffusion_mlp": "masked_runtime",
                    "npm_adapted": "npm_adapted_seed_direction",
                    "mpg_original": "mpg",
                    "pt": "pt",
                    "ppg_style": "ppg",
                    "sms_star": "sms_star",
                }[str(task.method)],
                "render_seed": task.render_seed,
                "control_id": task.control_id,
                "bounce_alpha": (
                    1.0 if task.method == "npm_adapted" else (
                    0.0 if task.method not in protocol.LEARNED_METHODS else selected["bounce_alpha"]
                    )
                ),
                "kappa_scale": (
                    1.0 if task.method != "diffusion_mlp"
                    else selected["kappa_scale"]
                ),
                "render": render,
            })
            if task.method in protocol.LEARNED_METHODS:
                if task.stage == "npm_step_validation":
                    steps = int(str(task.control_id).removeprefix("steps"))
                    dependency = successful_result(
                        state,
                        "export_npm_gate",
                        f"export_npm_gate__{task.scene}__steps{steps}",
                    )
                    spec.update({
                        "runtime_path": dependency["runtime_path"],
                        "runtime_sha256": dependency["runtime_sha256"],
                    })
                else:
                    spec.update(_runtime_dependency(
                        manifest, state, task.scene, str(task.method), int(task.model_seed)
                    ))
    return spec


def _task_timeout(manifest: dict[str, Any], task: protocol.Task) -> float:
    if task.kind == "replay":
        return (
            len(manifest["replay"]["seed_ids"])
            * float(manifest["replay"]["timeout_seconds"])
            + 600.0
        )
    if task.kind in {"render", "reference"}:
        limit = (
            manifest["reference"]["watchdog_seconds"]
            if task.kind == "reference"
            else manifest["primary"]["watchdog_seconds"]
        )
        return float(limit) + 300.0
    return 21600.0


def _run_prepared(
    *, python: Path, spec_path: Path, attempt_root: Path,
    stdout_path: Path, stderr_path: Path, cpus: tuple[int, ...],
    threads: int, timeout: float,
) -> dict[str, Any]:
    command = [
        str(python), str(PACKAGE_ROOT / "worker.py"),
        "--spec", str(spec_path), "--attempt-root", str(attempt_root),
    ]
    if platform.system() == "Linux":
        taskset = shutil.which("taskset")
        if not taskset:
            raise Phase20Error("taskset is required on Linux")
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
                command,
                cwd=WORKSPACE_ROOT,
                env=env,
                stdout=stdout,
                stderr=stderr,
                text=True,
                timeout=timeout,
                check=False,
            )
        return_code = completed.returncode
    except subprocess.TimeoutExpired:
        return_code = 124
        with stderr_path.open("a", encoding="utf-8") as stderr:
            stderr.write(f"\ncoordinator timeout after {timeout:.1f}s\n")
    duration = time.monotonic() - started
    result: dict[str, Any] = {
        "return_code": return_code,
        "duration_seconds": duration,
        "finished_at": now_utc(),
    }
    if return_code != 0:
        result.update({"status": "failed", "error": f"worker exited {return_code}"})
        return result
    try:
        lines = [line for line in stdout_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        payload = json.loads(lines[-1])
        if not isinstance(payload, dict):
            raise ValueError("last line is not an object")
        result.update({"status": "succeeded", "result": payload})
    except (OSError, IndexError, json.JSONDecodeError, ValueError) as exc:
        result.update({"status": "failed", "error": f"invalid worker result: {exc}"})
    return result


def _classify_job(record: dict[str, Any]) -> str:
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


def execute_stage(
    *, manifest: dict[str, Any], state: dict[str, Any], state_path: Path,
    output_root: Path, stage: str, python: Path, workers: int, threads: int,
    rerun_stale: bool,
) -> None:
    stage_state = ensure_stage(state, manifest, stage)
    tasks = protocol.build_tasks(manifest, stage)
    pending: list[protocol.Task] = []
    for task in tasks:
        record = stage_state["jobs"][task.task_id]
        status = _classify_job(record)
        if status == "succeeded":
            continue
        if status == "stale" and not rerun_stale:
            raise Phase20Error(f"artifact drift for {task.task_id}; pass --rerun-stale")
        if status == "running":
            record["status"] = "interrupted"
        pending.append(task)
    if not pending:
        stage_state["status"] = "succeeded"
        atomic_json(state_path, state)
        print(f"{stage}: already complete ({len(tasks)}/{len(tasks)})", flush=True)
        return

    slots = cpu_slots(min(workers, len(pending)), threads)
    stage_state["status"] = "running"
    stage_state.setdefault("started_at", now_utc())
    stage_state["workers"] = len(slots)
    stage_state["threads_per_worker"] = threads
    stage_state["allocated_physical_cores"] = len(slots) * threads
    stage_state["allocated_physical_percent"] = round(
        100.0 * len(slots) * threads / len(_physical_cpu_representatives()), 3
    )
    state["updated_at"] = now_utc()
    atomic_json(state_path, state)

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
        record = stage_state["jobs"][task.task_id]
        attempt_number = len(record["attempts"]) + 1
        attempt_root = (
            output_root / "stages" / stage / "jobs" / task.task_id
            / f"attempt_{attempt_number:03d}"
        )
        if attempt_root.exists():
            raise Phase20Error(f"attempt directory already exists: {attempt_root}")
        attempt_root.mkdir(parents=True)
        spec = build_worker_spec(manifest, state, task, output_root, threads)
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
            spec_path=spec_path,
            attempt_root=attempt_root,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            cpus=slots[slot_index],
            threads=threads,
            timeout=_task_timeout(manifest, task),
        )
        active[future] = (task, slot_index, attempt_root)
        return True

    with ThreadPoolExecutor(max_workers=len(slots), thread_name_prefix=f"phase20-{stage}") as executor:
        for index in range(len(slots)):
            if not submit(index, executor):
                break
        while active:
            finished, _ = wait(tuple(active), return_when=FIRST_COMPLETED)
            for future in finished:
                task, slot_index, attempt_root = active.pop(future)
                outcome = future.result()
                outcome["artifact_sha256"] = tree_hashes(attempt_root)
                record = stage_state["jobs"][task.task_id]
                record["attempts"][-1].update(outcome)
                record["status"] = outcome["status"]
                if outcome["status"] != "succeeded":
                    failed = True
                state["updated_at"] = now_utc()
                atomic_json(state_path, state)
                print(f"{stage}: {task.task_id}: {outcome['status']}", flush=True)
                submit(slot_index, executor)
    counts = {
        name: sum(_classify_job(record) == name for record in stage_state["jobs"].values())
        for name in ("succeeded", "failed", "pending", "interrupted", "stale")
    }
    stage_state["counts"] = counts
    stage_state["status"] = "succeeded" if counts["succeeded"] == len(tasks) else "failed"
    stage_state["finished_at"] = now_utc()
    state["updated_at"] = now_utc()
    atomic_json(state_path, state)
    if stage_state["status"] != "succeeded":
        raise Phase20Error(f"stage {stage} incomplete: {counts}")


def _resolve_project_artifact(value: object) -> Path:
    path = Path(str(value))
    resolved = path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()
    try:
        resolved.relative_to(PROJECT_ROOT.resolve())
    except ValueError as exc:
        raise Phase20Error(f"artifact escapes project: {path}") from exc
    return resolved


def build_reference_registry(
    manifest: dict[str, Any], state: dict[str, Any], output_root: Path
) -> dict[str, Any]:
    legacy_path = protocol.workspace_path(manifest["legacy_reference_manifest"])
    legacy = protocol.load_json(legacy_path)
    references: dict[str, Any] = {}
    for scene in protocol.EXISTING_SCENES:
        key = manifest["scenes"][scene]["legacy_reference_key"]
        entry = legacy.get("references", {}).get(key)
        if not isinstance(entry, dict):
            raise Phase20Error(f"legacy reference missing: {scene}")
        exr = _resolve_project_artifact(entry["exr_path"])
        if protocol.sha256_file(exr) != entry["exr_sha256"]:
            raise Phase20Error(f"legacy reference EXR changed: {scene}")
        references[scene] = {
            **entry,
            "source": "reused_locked_ten_scene_reference",
            "source_scene_id": key,
        }
    for scene in protocol.NEW_SCENES:
        task_id = f"reference__{scene}__r{manifest['reference']['seed']}"
        result = successful_result(state, "references_new", task_id)
        exr = _resolve_project_artifact(result["output_path"])
        if protocol.sha256_file(exr) != result["output_sha256"]:
            raise Phase20Error(f"new reference EXR changed: {scene}")
        references[scene] = {
            "scene_id": scene,
            "exr_path": result["output_path"],
            "exr_sha256": result["output_sha256"],
            "sample_count": manifest["reference"]["sample_count"],
            "sampler_seed": manifest["reference"]["seed"],
            "active_config_sha256": result["active_config_sha256"],
            "width": manifest["primary"]["width"],
            "height": manifest["primary"]["height"],
            "crop_width": manifest["primary"]["width"],
            "crop_height": manifest["primary"]["height"],
            "source": "fresh_phase20_reference",
        }
    payload = {
        "schema_version": 1,
        "protocol_id": protocol.PROTOCOL_ID,
        "status": "ready",
        "reference_set_id": protocol.sha256_json(references),
        "references": references,
    }
    path = output_root / "references" / "reference_registry.json"
    immutable_json(path, payload, "reference registry")
    return payload


def _metric(prediction: Path, reference: Path) -> tuple[float, float, float]:
    import phase_13_run_formal_benchmark as phase13

    return phase13.compute_metrics(prediction, reference)


def _geomean(values: list[float]) -> float:
    if not values or any(not math.isfinite(value) or value < 0.0 for value in values):
        raise Phase20Error("geometric mean requires finite nonnegative values")
    return math.exp(math.fsum(math.log(max(value, 1e-300)) for value in values) / len(values))


def finalize_npm_step_selection(
    manifest: dict[str, Any], state: dict[str, Any], output_root: Path
) -> int:
    references = build_reference_registry_for_existing(manifest)
    rows: list[dict[str, Any]] = []
    gate = manifest["npm_render_gate"]
    for task in protocol.build_tasks(manifest, "npm_step_validation"):
        result = successful_result(state, "npm_step_validation", task.task_id)
        output = _resolve_project_artifact(result["output_path"])
        mse, psnr, peak = _metric(output, references[task.scene])
        rows.append({
            "scene": task.scene,
            "steps": int(str(task.control_id).removeprefix("steps")),
            "render_seed": task.render_seed,
            "mse": mse,
            "psnr": psnr,
            "reference_peak": peak,
            "top1_luminance_share": result["tail"]["top1_luminance_share"],
            "output_path": result["output_path"],
            "output_sha256": result["output_sha256"],
        })
    candidates: dict[str, Any] = {}
    for steps in gate["candidate_steps"]:
        per_scene: dict[str, Any] = {}
        for scene in gate["anchor_scenes"]:
            subset = [row for row in rows if row["scene"] == scene and row["steps"] == steps]
            per_scene[scene] = {
                "mean_mse": statistics.fmean(row["mse"] for row in subset),
                "geometric_mean_mse": _geomean([row["mse"] for row in subset]),
                "median_top1_luminance_share": statistics.median(
                    row["top1_luminance_share"] for row in subset
                ),
            }
        candidates[str(steps)] = {"steps": steps, "scenes": per_scene}
    for scene in gate["anchor_scenes"]:
        order = sorted(
            gate["candidate_steps"],
            key=lambda steps: (candidates[str(steps)]["scenes"][scene]["geometric_mean_mse"], steps),
        )
        for rank, steps in enumerate(order, 1):
            candidates[str(steps)]["scenes"][scene]["mse_rank"] = rank
    best_by_scene = {
        scene: min(
            candidates[str(steps)]["scenes"][scene]["geometric_mean_mse"]
            for steps in gate["candidate_steps"]
        )
        for scene in gate["anchor_scenes"]
    }
    for steps in gate["candidate_steps"]:
        item = candidates[str(steps)]
        item["mean_scene_rank"] = statistics.fmean(
            item["scenes"][scene]["mse_rank"] for scene in gate["anchor_scenes"]
        )
        item["geometric_mean_scene_ratio_to_best"] = _geomean([
            item["scenes"][scene]["geometric_mean_mse"] / best_by_scene[scene]
            for scene in gate["anchor_scenes"]
        ])
    selected = min(
        gate["candidate_steps"],
        key=lambda steps: (
            candidates[str(steps)]["mean_scene_rank"],
            candidates[str(steps)]["geometric_mean_scene_ratio_to_best"],
            steps,
        ),
    )
    payload = {
        "schema_version": 1,
        "protocol_id": protocol.PROTOCOL_ID,
        "status": "renderer_quality_gate_complete",
        "selection_rule": gate["selection_rule"],
        "training_seed": gate["training_seed"],
        "render_seeds": gate["render_seeds"],
        "sample_count": gate["sample_count"],
        "selected_steps": selected,
        "candidates": candidates,
        "rows": rows,
    }
    path = output_root / "selection" / "selected_npm_steps.json"
    immutable_json(path, payload, "NPM renderer step selection")
    return int(selected)


def build_reference_registry_for_existing(manifest: dict[str, Any]) -> dict[str, Path]:
    legacy = protocol.load_json(protocol.workspace_path(manifest["legacy_reference_manifest"]))
    references: dict[str, Path] = {}
    for scene in protocol.EXISTING_SCENES:
        key = manifest["scenes"][scene]["legacy_reference_key"]
        entry = legacy.get("references", {}).get(key)
        if not isinstance(entry, dict):
            raise Phase20Error(f"legacy reference missing: {scene}")
        exr = _resolve_project_artifact(entry["exr_path"])
        if protocol.sha256_file(exr) != entry["exr_sha256"]:
            raise Phase20Error(f"legacy reference hash mismatch: {scene}")
        references[scene] = exr
    return references


def finalize_primary(
    manifest: dict[str, Any], state: dict[str, Any], output_root: Path
) -> dict[str, Any]:
    registry = build_reference_registry(manifest, state, output_root)
    rows: list[dict[str, Any]] = []
    for task in protocol.build_tasks(manifest, "primary"):
        result = successful_result(state, "primary", task.task_id)
        reference_entry = registry["references"][task.scene]
        reference = _resolve_project_artifact(reference_entry["exr_path"])
        prediction = _resolve_project_artifact(result["output_path"])
        if protocol.sha256_file(prediction) != result["output_sha256"]:
            raise Phase20Error(f"primary output changed: {task.task_id}")
        mse, psnr, peak = _metric(prediction, reference)
        rows.append({
            "scene": task.scene,
            "cohort": manifest["scenes"][task.scene]["cohort"],
            "method": task.method,
            "model_seed": task.model_seed,
            "render_seed": task.render_seed,
            "sample_count": manifest["primary"]["sample_count"],
            "mse": mse,
            "psnr": psnr,
            "reference_peak": peak,
            "render_seconds": result["record"]["seconds"],
            "top1_luminance_share": result["tail"]["top1_luminance_share"],
            "output_path": result["output_path"],
            "output_sha256": result["output_sha256"],
        })
    scene_table: dict[str, Any] = {}
    for scene in protocol.ALL_SCENES:
        methods: dict[str, Any] = {}
        for method in protocol.PRIMARY_METHODS:
            subset = [row for row in rows if row["scene"] == scene and row["method"] == method]
            methods[method] = {
                "replicates": len(subset),
                "mean_mse": statistics.fmean(row["mse"] for row in subset),
                "median_mse": statistics.median(row["mse"] for row in subset),
                "geometric_mean_mse": _geomean([row["mse"] for row in subset]),
                "mean_psnr": statistics.fmean(row["psnr"] for row in subset),
                "median_render_seconds": statistics.median(
                    row["render_seconds"] for row in subset
                ),
                "median_top1_luminance_share": statistics.median(
                    row["top1_luminance_share"] for row in subset
                ),
                "max_top1_luminance_share": max(
                    row["top1_luminance_share"] for row in subset
                ),
            }
        baseline = methods["mpg_original"]["mean_mse"]
        scene_table[scene] = {
            "cohort": manifest["scenes"][scene]["cohort"],
            "methods": methods,
            "mse_ratio_to_mpg": {
                method: methods[method]["mean_mse"] / baseline
                for method in protocol.LEARNED_METHODS
            },
        }
    aggregate: dict[str, Any] = {}
    for method in protocol.LEARNED_METHODS:
        comparisons: dict[str, Any] = {}
        for baseline_method in ("mpg_original", "pt", "ppg_style", "sms_star"):
            ratios = [
                scene_table[scene]["methods"][method]["mean_mse"]
                / max(
                    scene_table[scene]["methods"][baseline_method]["mean_mse"],
                    1e-300,
                )
                for scene in protocol.ALL_SCENES
            ]
            comparisons[baseline_method] = {
                "scene_wins": sum(value < 1.0 for value in ratios),
                "scene_ties": sum(value == 1.0 for value in ratios),
                "scene_losses": sum(value > 1.0 for value in ratios),
                "geometric_mean_scene_mse_ratio": _geomean(ratios),
                "median_scene_mse_ratio": statistics.median(ratios),
            }
        aggregate[method] = {"comparisons": comparisons}
    payload = {
        "schema_version": 1,
        "protocol_id": protocol.PROTOCOL_ID,
        "metric": "equal-render-SPP; offline training cost reported separately",
        "sample_count": manifest["primary"]["sample_count"],
        "render_seeds": manifest["primary"]["render_seeds"],
        "model_seeds": manifest["model_seeds"],
        "selected_controls": _selected_controls(manifest, state, output_root),
        "reference_registry_sha256": protocol.sha256_file(
            output_root / "references" / "reference_registry.json"
        ),
        "scene_table": scene_table,
        "aggregate": aggregate,
        "rows": rows,
    }
    reports = output_root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    immutable_json(reports / "phase20_equal_spp_results.json", payload, "final results")
    with (reports / "phase20_equal_spp_rows.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# Phase 20 equal-render-SPP results",
        "",
        f"- Candidate render budget: **{manifest['primary']['sample_count']} SPP**.",
        f"- Render seeds: `{manifest['primary']['render_seeds']}`.",
        f"- Selected controls: `{payload['selected_controls']}`.",
        "- Offline training is not counted in render SPP and is reported in the ledger.",
        "",
        "| Scene | Cohort | Diffusion-MLP | NPM-adapted | MPG | PT | PPG | SMS* | Diff./MPG | NPM/MPG |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for scene in protocol.ALL_SCENES:
        item = scene_table[scene]
        methods = item["methods"]
        ratio = item["mse_ratio_to_mpg"]
        lines.append(
            f"| {scene} | {item['cohort']} | {methods['diffusion_mlp']['mean_mse']:.8g} | "
            f"{methods['npm_adapted']['mean_mse']:.8g} | {methods['mpg_original']['mean_mse']:.8g} | "
            f"{methods['pt']['mean_mse']:.8g} | {methods['ppg_style']['mean_mse']:.8g} | "
            f"{methods['sms_star']['mean_mse']:.8g} | "
            f"{ratio['diffusion_mlp']:.4f} | {ratio['npm_adapted']:.4f} |"
        )
    lines.extend(["", "## Across-scene summary", ""])
    for method in protocol.LEARNED_METHODS:
        item = aggregate[method]["comparisons"]["mpg_original"]
        lines.append(
            f"- **{method} vs original MPG**: {item['scene_wins']}/14 wins; "
            f"geometric-mean scene MSE ratio `{item['geometric_mean_scene_mse_ratio']:.4f}`."
        )
    (reports / "phase20_equal_spp_results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return payload


def status_payload(manifest: dict[str, Any]) -> dict[str, Any]:
    root = protocol.workspace_path(manifest["output_root"])
    state_path = root / "coordinator_state.json"
    if not state_path.is_file():
        return {"protocol_id": protocol.PROTOCOL_ID, "status": "not_started"}
    state = protocol.load_json(state_path)
    stages: dict[str, Any] = {}
    total, succeeded = 0, 0
    for stage in protocol.STAGES:
        stage_state = state.get("stages", {}).get(stage)
        expected = len(protocol.build_tasks(manifest, stage))
        if not isinstance(stage_state, dict):
            stages[stage] = {"status": "not_started", "succeeded": 0, "total": expected}
            total += expected
            continue
        done = sum(_classify_job(record) == "succeeded" for record in stage_state["jobs"].values())
        total += expected
        succeeded += done
        stages[stage] = {
            "status": stage_state.get("status"),
            "succeeded": done,
            "total": expected,
            "allocated_physical_percent": stage_state.get("allocated_physical_percent"),
        }
    return {
        "protocol_id": state.get("protocol_id"),
        "status": state.get("status"),
        "updated_at": state.get("updated_at"),
        "progress": {"succeeded": succeeded, "total": total, "percent": round(100*succeeded/total, 2)},
        "stages": stages,
    }


def run_pipeline(args: argparse.Namespace) -> int:
    manifest, manifest_sha = protocol.load_and_validate_manifest(args.manifest, require_files=True)
    inputs = validate_static_inputs(manifest, require_linux=True)
    python = Path(os.path.abspath(os.fspath(args.python)))
    probe = _python_probe(python, args.threads_per_worker)
    selection_override = _selection_override(
        manifest,
        selection_json=args.selection_json,
        bounce_alpha=args.bounce_alpha,
        kappa_scale=args.kappa_scale,
    )
    output_root = protocol.workspace_path(manifest["output_root"])
    state_path = output_root / "coordinator_state.json"
    lock_path = protocol.workspace_path(manifest["global_lock_path"])
    with coordinator_lock(lock_path):
        state = load_or_create_state(
            state_path,
            manifest_sha=manifest_sha,
            inputs=inputs,
            python_probe=probe,
            selection_override=selection_override,
        )
        stages = list(protocol.STAGES)
        for stage in stages:
            execute_stage(
                manifest=manifest,
                state=state,
                state_path=state_path,
                output_root=output_root,
                stage=stage,
                python=python,
                workers=args.workers,
                threads=args.threads_per_worker,
                rerun_stale=args.rerun_stale,
            )
            current = validate_static_inputs(manifest, require_linux=True)
            if current != state["input_snapshot"]:
                raise Phase20Error(f"immutable input changed during {stage}")
            if stage == "npm_step_validation":
                finalize_npm_step_selection(manifest, state, output_root)
            elif stage == "references_new":
                build_reference_registry(manifest, state, output_root)
            if args.stop_after == stage:
                state["status"] = f"stopped_after_{stage}"
                state["updated_at"] = now_utc()
                atomic_json(state_path, state)
                return 0
        final = finalize_primary(manifest, state, output_root)
        del final
        state["status"] = "complete"
        state["updated_at"] = now_utc()
        final_path = output_root / "reports" / "phase20_equal_spp_results.json"
        state["final_results_path"] = protocol.portable_workspace(final_path)
        state["final_results_sha256"] = protocol.sha256_file(final_path)
        atomic_json(state_path, state)
    return 0


def print_plan(manifest: dict[str, Any]) -> None:
    counts = protocol.task_counts(manifest)
    print(json.dumps({
        "protocol_id": protocol.PROTOCOL_ID,
        "server_only": True,
        "primary_metric": "equal-render-SPP",
        "primary_spp": manifest["primary"]["sample_count"],
        "scene_count": len(protocol.ALL_SCENES),
        "existing_scenes": list(protocol.EXISTING_SCENES),
        "new_qualified_scenes": list(protocol.NEW_SCENES),
        "methods": list(protocol.PRIMARY_METHODS),
        "dit_included": False,
        "task_counts": counts,
        "total_tasks": sum(counts.values()),
    }, indent=2, sort_keys=True))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("plan", "preflight", "run", "status", "finalize")
    )
    parser.add_argument("--manifest", type=Path, default=protocol.DEFAULT_MANIFEST)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--threads-per-worker", type=int, default=4)
    parser.add_argument("--rerun-stale", action="store_true")
    parser.add_argument("--selection-json", type=Path)
    parser.add_argument("--bounce-alpha", type=float)
    parser.add_argument("--kappa-scale", type=float)
    parser.add_argument("--stop-after", choices=protocol.STAGES)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        manifest, _ = protocol.load_and_validate_manifest(
            args.manifest, require_files=args.command != "plan"
        )
        if args.command == "plan":
            print_plan(manifest)
            return 0
        if args.command == "status":
            print(json.dumps(status_payload(manifest), indent=2, sort_keys=True))
            return 0
        if args.command == "preflight":
            inputs = validate_static_inputs(manifest, require_linux=True)
            slots = cpu_slots(args.workers, args.threads_per_worker)
            probe = _python_probe(Path(os.path.abspath(args.python)), args.threads_per_worker)
            print(json.dumps({
                "protocol_id": protocol.PROTOCOL_ID,
                "status": "ready",
                "input_snapshot_sha256": protocol.sha256_json(inputs),
                "python": probe,
                "workers": args.workers,
                "threads_per_worker": args.threads_per_worker,
                "allocated_physical_cores": sum(len(slot) for slot in slots),
                "detected_physical_cores": len(_physical_cpu_representatives()),
                "allocated_physical_percent": round(
                    100 * sum(len(slot) for slot in slots) / len(_physical_cpu_representatives()), 2
                ),
                "task_counts": protocol.task_counts(manifest),
            }, indent=2, sort_keys=True))
            return 0
        if args.command == "finalize":
            root = protocol.workspace_path(manifest["output_root"])
            state = protocol.load_json(root / "coordinator_state.json")
            finalize_npm_step_selection(manifest, state, root)
            finalize_primary(manifest, state, root)
            return 0
        return run_pipeline(args)
    except (Phase20Error, protocol.ProtocolError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
