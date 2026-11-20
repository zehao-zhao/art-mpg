#!/usr/bin/env python3
"""Thin Phase-48 adapter around the frozen Phase-20 worker."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any


PACKAGE_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = PACKAGE_ROOT.parent
PROJECT_ROOT = WORKSPACE_ROOT.parent
for entry in (str(PACKAGE_ROOT), str(WORKSPACE_ROOT / "scripts"), str(WORKSPACE_ROOT / "src")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

import protocol  # noqa: E402
import overlay_runtime  # noqa: E402


class WorkerError(RuntimeError):
    """Raised when a Phase-48 task violates its contract."""


def _load_phase20_worker(manifest: dict[str, Any]) -> Any:
    source_protocol_path = protocol.workspace_path(
        manifest["source"]["phase20_protocol"]["path"]
    )
    source_worker_path = protocol.workspace_path(
        manifest["source"]["phase20_worker"]["path"]
    )
    if protocol.sha256_file(source_protocol_path) != manifest["source"]["phase20_protocol"]["sha256"]:
        raise WorkerError("frozen Phase20 protocol hash changed")
    if protocol.sha256_file(source_worker_path) != manifest["source"]["phase20_worker"]["sha256"]:
        raise WorkerError("frozen Phase20 worker hash changed")

    protocol_module_name = "phase48_frozen_phase20_protocol"
    protocol_spec = importlib.util.spec_from_file_location(protocol_module_name, source_protocol_path)
    if protocol_spec is None or protocol_spec.loader is None:
        raise WorkerError("cannot load frozen Phase20 protocol")
    source_protocol = importlib.util.module_from_spec(protocol_spec)
    previous_protocol = sys.modules.get("protocol")
    previous_named_protocol = sys.modules.get(protocol_module_name)
    sys.modules["protocol"] = source_protocol
    # dataclasses resolves annotations through sys.modules[cls.__module__]
    # while executing the frozen Python-3.11 Phase20 protocol.
    sys.modules[protocol_module_name] = source_protocol
    try:
        protocol_spec.loader.exec_module(source_protocol)
        worker_spec = importlib.util.spec_from_file_location("phase48_frozen_phase20_worker", source_worker_path)
        if worker_spec is None or worker_spec.loader is None:
            raise WorkerError("cannot load frozen Phase20 worker")
        source_worker = importlib.util.module_from_spec(worker_spec)
        worker_spec.loader.exec_module(source_worker)
    finally:
        if previous_protocol is None:
            sys.modules.pop("protocol", None)
        else:
            sys.modules["protocol"] = previous_protocol
        if previous_named_protocol is None:
            sys.modules.pop(protocol_module_name, None)
        else:
            sys.modules[protocol_module_name] = previous_named_protocol
    if Path(source_worker.protocol.__file__).resolve() != source_protocol_path.resolve():
        raise WorkerError("Phase20 worker imported the wrong protocol module")
    return source_worker


def _inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _read_spec(path: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    spec = protocol.load_json(path)
    if spec.get("protocol_id") != protocol.PROTOCOL_ID:
        raise WorkerError("worker specification protocol mismatch")
    if spec.get("manifest_sha256") != protocol.sha256_file(protocol.MANIFEST_PATH):
        raise WorkerError("worker specification manifest binding changed")
    if spec.get("kind") not in {"train", "export", "render"}:
        raise WorkerError("unsupported Phase48 worker kind")
    config_id = str(spec.get("config_id"))
    if config_id not in manifest["configs"]:
        raise WorkerError("unknown ablation configuration")
    config = manifest["configs"][config_id]
    if spec.get("direction_conditioning_mode") != config["direction_conditioning_mode"]:
        raise WorkerError("worker conditioning mode differs from manifest")
    if int(spec.get("epochs", 0)) != int(config["epochs"]):
        raise WorkerError("worker epoch count differs from manifest")
    if str(spec.get("scene")) not in protocol.SCENES:
        raise WorkerError("worker scene is outside the frozen suite")
    if int(spec.get("model_seed", -1)) not in protocol.MODEL_SEEDS:
        raise WorkerError("worker model seed is outside the frozen pair")
    return spec


def _source_spec(spec: dict[str, Any]) -> dict[str, Any]:
    common = {
        "protocol_id": protocol.SOURCE_PROTOCOL_ID,
        "task_id": spec["task_id"],
        "scene": spec["scene"],
        "job_id": spec["job_id"],
        "scene_path": spec["scene_path"],
        "threads": int(spec["threads"]),
        "model_seed": int(spec["model_seed"]),
    }
    if spec["kind"] == "train":
        return {
            **common,
            "kind": "train_diffusion",
            "replay_path": spec["replay_path"],
            "replay_sha256": spec["replay_sha256"],
            "training": spec["training"],
        }
    if spec["kind"] == "export":
        return {
            **common,
            "kind": "export_diffusion",
            "method": "diffusion_mlp",
            "checkpoint_path": spec["checkpoint_path"],
            "checkpoint_sha256": spec["checkpoint_sha256"],
        }
    render = spec["render"]
    return {
        **common,
        "kind": "render",
        "logical_method": "diffusion_mlp",
        "phase13_method": "masked_runtime",
        "render_seed": int(spec["render_seed"]),
        "control_id": spec["config_id"],
        "bounce_alpha": float(render["bounce_alpha"]),
        "kappa_scale": float(render["kappa_scale"]),
        "runtime_path": spec["runtime_path"],
        "runtime_sha256": spec["runtime_sha256"],
        "render": {
            key: value for key, value in render.items()
            if key not in {"bounce_alpha", "kappa_scale"}
        },
    }


def _resolve_project_artifact(value: object) -> Path:
    path = Path(str(value))
    result = path.resolve() if path.is_absolute() else (protocol.PROJECT_ROOT / path).resolve()
    if not _inside(result, protocol.PROJECT_ROOT):
        raise WorkerError(f"artifact escapes project: {value}")
    return result


def _runtime_mode(path: Path) -> str:
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) == 2 and fields[0] == "direction_conditioning_mode":
            return fields[1]
    raise WorkerError(f"runtime export lacks direction_conditioning_mode: {path}")


def _add_training_metrics(spec: dict[str, Any], result: dict[str, Any]) -> None:
    checkpoint = protocol.workspace_path(result["checkpoint_path"], "training checkpoint")
    metrics_json = checkpoint.parent / "metrics.json"
    metrics_csv = checkpoint.parent / "metrics.csv"
    summary_path = protocol.workspace_path(result["summary_path"], "training summary")
    try:
        history = json.loads(metrics_json.read_text(encoding="utf-8"))
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise WorkerError(f"cannot read training metrics: {exc}") from exc
    if not isinstance(history, list) or len(history) != int(spec["epochs"]):
        raise WorkerError(
            f"training produced {len(history) if isinstance(history, list) else 'invalid'} "
            f"epochs, expected {spec['epochs']}"
        )
    required = (
        "epoch", "val_loss", "val_chain_length_loss", "val_chain_type_loss",
        "val_direction_loss",
    )
    if any(not isinstance(row, dict) or any(row.get(key) is None for key in required) for row in history):
        raise WorkerError("training history lacks finite validation components")
    if any(not math.isfinite(float(row[key])) for row in history for key in required[1:]):
        raise WorkerError("training history contains nonfinite validation components")
    final = history[-1]
    best = min(history, key=lambda row: (float(row["val_loss"]), int(row["epoch"])))
    if not isinstance(summary, dict) or int(summary.get("epochs_completed", -1)) != int(spec["epochs"]):
        raise WorkerError("training summary epoch count differs from the ablation contract")
    if not math.isclose(float(summary["best_validation_loss"]), float(best["val_loss"]), rel_tol=1e-10, abs_tol=1e-12):
        raise WorkerError("training summary best loss differs from epoch metrics")
    result.update({
        "metrics_json_path": protocol.portable_workspace(metrics_json),
        "metrics_json_sha256": protocol.sha256_file(metrics_json),
        "metrics_csv_path": protocol.portable_workspace(metrics_csv),
        "metrics_csv_sha256": protocol.sha256_file(metrics_csv),
        "training_seconds": float(summary["training_seconds"]),
        "epochs_completed": int(summary["epochs_completed"]),
        "training_metrics": {
            "final": {key: final[key] for key in required},
            "best": {key: best[key] for key in required},
        },
    })


def _add_render_metrics(spec: dict[str, Any], result: dict[str, Any]) -> None:
    from phase_13_run_formal_benchmark import compute_metrics

    output = _resolve_project_artifact(result["output_path"])
    reference = protocol.workspace_path(spec["reference_path"], "reference")
    if protocol.sha256_file(reference) != spec["reference_sha256"]:
        raise WorkerError(f"reference hash changed: {spec['scene']}")
    mse, psnr, peak = compute_metrics(output, reference)
    if not all(math.isfinite(value) for value in (mse, psnr, peak)) or mse < 0.0:
        raise WorkerError("computed nonfinite/negative render metric")
    result.update({
        "run_id": spec["run_id"],
        "sample_count": int(spec["render"]["sample_count"]),
        "mse": mse,
        "psnr": psnr,
        "reference_peak": peak,
        "reference_path": spec["reference_path"],
        "reference_sha256": spec["reference_sha256"],
        "render_seconds": float(result["record"]["seconds"]),
    })


def run(spec: dict[str, Any], attempt_root: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    declared_root = protocol.workspace_path(spec["attempt_root"], "attempt root")
    if declared_root != attempt_root.resolve():
        raise WorkerError("worker attempt root differs from specification")
    output_root = protocol.workspace_path(manifest["output_root"], "output root")
    if not _inside(attempt_root, output_root):
        raise WorkerError("attempt root escapes isolated Phase48 output")

    overlay_runtime.activate_python_overlay(manifest)
    source_worker = _load_phase20_worker(manifest)
    overlay_runtime.install_phase20_runtime_override(source_worker, manifest)
    result = source_worker.run(_source_spec(spec), attempt_root)
    if not isinstance(result, dict):
        raise WorkerError("Phase20 worker returned no result")
    if spec["kind"] == "train":
        _add_training_metrics(spec, result)
    elif spec["kind"] == "export":
        runtime = protocol.workspace_path(result["runtime_path"], "runtime export")
        observed_mode = _runtime_mode(runtime)
        if observed_mode != spec["direction_conditioning_mode"]:
            raise WorkerError(
                f"runtime conditioning mode mismatch: {observed_mode} != "
                f"{spec['direction_conditioning_mode']}"
            )
    elif spec["kind"] == "render":
        _add_render_metrics(spec, result)
    result.update({
        "protocol_id": protocol.PROTOCOL_ID,
        "task_id": spec["task_id"],
        "host_id": spec["host_id"],
        "config_id": spec["config_id"],
        "direction_conditioning_mode": spec["direction_conditioning_mode"],
        "epochs": int(spec["epochs"]),
    })
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--attempt-root", type=Path, required=True)
    args = parser.parse_args(argv)
    manifest = protocol.load_manifest()
    spec = _read_spec(args.spec.resolve(), manifest)
    result = run(spec, args.attempt_root.resolve(), manifest)
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (WorkerError, protocol.ProtocolError) as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
