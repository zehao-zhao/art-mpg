#!/usr/bin/env python3
"""Thin Phase-49 CCO0 architecture-sensitivity adapter around Phase 20."""

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
    """Raised when a Phase-49 task violates its contract."""


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

    protocol_module_name = "phase49_v2_cco0_frozen_phase20_protocol"
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
        worker_spec = importlib.util.spec_from_file_location(
            "phase49_v2_cco0_frozen_phase20_worker", source_worker_path
        )
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
        raise WorkerError("unsupported Phase49 worker kind")
    config_id = str(spec.get("config_id"))
    if config_id not in manifest["configs"]:
        raise WorkerError("unknown ablation configuration")
    config = manifest["configs"][config_id]
    if spec.get("direction_conditioning_mode") != config["direction_conditioning_mode"]:
        raise WorkerError("worker conditioning mode differs from manifest")
    if int(spec.get("epochs", 0)) != int(config["epochs"]):
        raise WorkerError("worker epoch count differs from manifest")
    for key in (
        "hidden_dim", "depth", "vmf_components", "token_embedding_dim",
        "time_embedding_dim", "parameter_count",
    ):
        if int(spec.get(key, -1)) != int(config[key]):
            raise WorkerError(f"worker architecture field differs from manifest: {key}")
    if not math.isclose(float(spec.get("dropout", -1)), float(config["dropout"])):
        raise WorkerError("worker dropout differs from manifest")
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


def _runtime_architecture(path: Path) -> dict[str, int | float | str]:
    wanted = {
        "hidden_dim", "depth", "vmf_components", "token_embedding_dim",
        "time_embedding_dim", "direction_conditioning_mode", "dropout",
    }
    # Dropout is a training-only setting and therefore is not serialized in
    # the runtime.  It is validated from the checkpoint instead.
    wanted.remove("dropout")
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) == 2 and fields[0] in wanted:
            values[fields[0]] = fields[1]
    if set(values) != wanted:
        raise WorkerError(f"runtime export lacks architecture tokens: {sorted(wanted - set(values))}")
    return {
        "hidden_dim": int(values["hidden_dim"]),
        "depth": int(values["depth"]),
        "vmf_components": int(values["vmf_components"]),
        "token_embedding_dim": int(values["token_embedding_dim"]),
        "time_embedding_dim": int(values["time_embedding_dim"]),
        "direction_conditioning_mode": values["direction_conditioning_mode"],
    }


def _expected_architecture(spec: dict[str, Any]) -> dict[str, int | float | str]:
    return {
        "hidden_dim": int(spec["hidden_dim"]),
        "depth": int(spec["depth"]),
        "vmf_components": int(spec["vmf_components"]),
        "token_embedding_dim": int(spec["token_embedding_dim"]),
        "time_embedding_dim": int(spec["time_embedding_dim"]),
        "direction_conditioning_mode": str(spec["direction_conditioning_mode"]),
    }


def _verify_checkpoint_architecture(
    spec: dict[str, Any], checkpoint: Path, summary: dict[str, Any],
) -> dict[str, Any]:
    import torch

    try:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    except TypeError:  # compatibility with older torch
        payload = torch.load(checkpoint, map_location="cpu")
    if not isinstance(payload, dict) or not isinstance(payload.get("config"), dict):
        raise WorkerError("training checkpoint lacks a configuration payload")
    config = payload["config"]
    expected = _expected_architecture(spec)
    for key, value in expected.items():
        observed = config.get(key)
        if isinstance(value, int):
            observed = int(observed)
        if observed != value:
            raise WorkerError(f"checkpoint architecture mismatch for {key}: {observed} != {value}")
    if not math.isclose(float(config.get("dropout", -1)), float(spec["dropout"])):
        raise WorkerError("checkpoint dropout differs from the sensitivity contract")
    state = payload.get("model_state_dict")
    if not isinstance(state, dict):
        raise WorkerError("training checkpoint lacks model_state_dict")
    parameter_count = sum(int(tensor.numel()) for tensor in state.values())
    expected_count = int(spec["parameter_count"])
    if parameter_count != expected_count:
        raise WorkerError(f"checkpoint parameter count {parameter_count} != {expected_count}")
    if int(summary.get("model_parameter_count", -1)) != expected_count:
        raise WorkerError("training summary model parameter count differs from the contract")
    if int(summary.get("trainable_parameter_count", -1)) != expected_count:
        raise WorkerError("training summary trainable parameter count differs from the contract")
    return {**expected, "dropout": float(spec["dropout"]), "parameter_count": parameter_count}


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
    architecture = _verify_checkpoint_architecture(spec, checkpoint, summary)
    result.update({
        "metrics_json_path": protocol.portable_workspace(metrics_json),
        "metrics_json_sha256": protocol.sha256_file(metrics_json),
        "metrics_csv_path": protocol.portable_workspace(metrics_csv),
        "metrics_csv_sha256": protocol.sha256_file(metrics_csv),
        "training_seconds": float(summary["training_seconds"]),
        "epochs_completed": int(summary["epochs_completed"]),
        "architecture_verification": architecture,
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
    runtime = protocol.workspace_path(spec["runtime_path"], "runtime export")
    observed_architecture = _runtime_architecture(runtime)
    expected_architecture = _expected_architecture(spec)
    if observed_architecture != expected_architecture:
        raise WorkerError(
            f"render runtime architecture mismatch: {observed_architecture} != {expected_architecture}"
        )
    record = result.get("record")
    if not isinstance(record, dict) or not record.get("stdout_log"):
        raise WorkerError("render result lacks renderer stdout log")
    renderer_log = _resolve_project_artifact(record["stdout_log"])
    log_text = renderer_log.read_text(encoding="utf-8", errors="replace")
    forbidden = (
        "Disable masked diffusion runtime fallback", "Unknown token in masked diffusion runtime file",
        "Failed to open masked diffusion runtime file", "Unexpected masked diffusion runtime magic",
    )
    if any(token in log_text for token in forbidden):
        raise WorkerError("renderer log reports masked-runtime load/fallback failure")
    if (
        "Loaded masked diffusion runtime model from" not in log_text
        or "direction_conditioning_mode=full" not in log_text
        or str(runtime) not in log_text
    ):
        raise WorkerError("renderer log does not confirm the exact full-conditioning runtime load")
    print(
        "PHASE49_RUNTIME_CANARY "
        f"config={spec['config_id']} hidden_dim={observed_architecture['hidden_dim']} "
        f"depth={observed_architecture['depth']} K={observed_architecture['vmf_components']} "
        "mode=full fallback=none",
        flush=True,
    )
    result.update({
        "run_id": spec["run_id"],
        "sample_count": int(spec["render"]["sample_count"]),
        "mse": mse,
        "psnr": psnr,
        "reference_peak": peak,
        "reference_path": spec["reference_path"],
        "reference_sha256": spec["reference_sha256"],
        "render_seconds": float(result["record"]["seconds"]),
        "architecture_verification": {
            **observed_architecture,
            "parameter_count": int(spec["parameter_count"]),
            "renderer_log_sha256": protocol.sha256_file(renderer_log),
            "renderer_confirmed_runtime_load": True,
            "renderer_fallback_detected": False,
        },
    })


def run(spec: dict[str, Any], attempt_root: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    declared_root = protocol.workspace_path(spec["attempt_root"], "attempt root")
    if declared_root != attempt_root.resolve():
        raise WorkerError("worker attempt root differs from specification")
    output_root = protocol.workspace_path(manifest["output_root"], "output root")
    if not _inside(attempt_root, output_root):
        raise WorkerError("attempt root escapes isolated Phase49 output")

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
        observed_architecture = _runtime_architecture(runtime)
        expected_architecture = _expected_architecture(spec)
        if observed_architecture != expected_architecture:
            raise WorkerError(
                f"runtime architecture mismatch: {observed_architecture} != {expected_architecture}"
            )
        result["architecture_verification"] = {
            **observed_architecture,
            "parameter_count": int(spec["parameter_count"]),
        }
    elif spec["kind"] == "render":
        _add_render_metrics(spec, result)
    result.update({
        "protocol_id": protocol.PROTOCOL_ID,
        "task_id": spec["task_id"],
        "host_id": spec["host_id"],
        "config_id": spec["config_id"],
        "direction_conditioning_mode": spec["direction_conditioning_mode"],
        "epochs": int(spec["epochs"]),
        "hidden_dim": int(spec["hidden_dim"]),
        "depth": int(spec["depth"]),
        "vmf_components": int(spec["vmf_components"]),
        "token_embedding_dim": int(spec["token_embedding_dim"]),
        "time_embedding_dim": int(spec["time_embedding_dim"]),
        "dropout": float(spec["dropout"]),
        "parameter_count": int(spec["parameter_count"]),
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
