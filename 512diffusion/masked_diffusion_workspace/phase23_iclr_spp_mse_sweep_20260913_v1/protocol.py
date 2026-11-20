#!/usr/bin/env python3
"""Frozen task contract for the ICLR MPGD SPP--MSE successor sweep."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


PACKAGE_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = PACKAGE_ROOT.parent
PROJECT_ROOT = WORKSPACE_ROOT.parent
DEFAULT_MANIFEST = PACKAGE_ROOT / "phase23_iclr_spp_mse_protocol_v1.json"

PROTOCOL_ID = "phase23_iclr_mpgd_spp_mse_living_slabs_20260913_v1"
SOURCE_PROTOCOL_ID = "phase20_equal_render_spp_20260904_v2"
SCENES = ("living", "slabs")
METHODS = ("diffusion_mlp", "npm_adapted", "mpg_original", "pt", "sms_star")
LEARNED_METHODS = ("diffusion_mlp", "npm_adapted")
MODEL_SEEDS = (104729, 130363, 155921)
RENDER_SEEDS = (1729, 31415, 271828, 424242, 8675309)
NEW_SAMPLE_COUNTS = (1, 2, 4, 8, 16, 32)
REUSED_SAMPLE_COUNT = 64


class ProtocolError(RuntimeError):
    """Raised when the preregistered successor contract is violated."""


@dataclass(frozen=True, slots=True)
class Task:
    task_id: str
    scene: str
    method: str
    sample_count: int
    render_seed: int
    model_seed: int | None

    def payload(self) -> dict[str, Any]:
        return asdict(self)


def json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ProtocolError(f"JSON object required: {path}")
    return value


def _relative_path(value: object, base: Path, boundary: Path, label: str) -> Path:
    if not isinstance(value, str) or not value or Path(value).is_absolute():
        raise ProtocolError(f"{label} must be a nonempty relative path")
    result = (base / value).resolve()
    try:
        result.relative_to(boundary.resolve())
    except ValueError as exc:
        raise ProtocolError(f"{label} escapes the project: {value}") from exc
    return result


def workspace_path(value: object, label: str = "workspace path") -> Path:
    return _relative_path(value, WORKSPACE_ROOT, PROJECT_ROOT, label)


def project_path(value: object, label: str = "project path") -> Path:
    return _relative_path(value, PROJECT_ROOT, PROJECT_ROOT, label)


def portable_workspace(path: Path) -> str:
    try:
        return path.resolve().relative_to(WORKSPACE_ROOT.resolve()).as_posix()
    except ValueError as exc:
        raise ProtocolError(f"path is outside workspace: {path}") from exc


def portable_project(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError as exc:
        raise ProtocolError(f"path is outside project: {path}") from exc


def _sha(value: object, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ProtocolError(f"{label} must be a SHA-256 hex digest")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ProtocolError(f"{label} must be a SHA-256 hex digest") from exc
    return value


def load_and_validate_manifest(path: Path = DEFAULT_MANIFEST) -> tuple[dict[str, Any], str]:
    manifest = load_json(path.resolve())
    if manifest.get("schema_version") != 1:
        raise ProtocolError("schema_version must be 1")
    if manifest.get("protocol_id") != PROTOCOL_ID:
        raise ProtocolError("protocol_id changed")
    if manifest.get("profile_id") != "linux_native":
        raise ProtocolError("the successor is Linux-server-only")
    if manifest.get("output_root") != f"results/{PROTOCOL_ID}":
        raise ProtocolError("output_root must remain isolated from Phase 20")
    if manifest.get("global_lock_path") != "results/.phase23_iclr_spp_mse_v1.lock":
        raise ProtocolError("global lock path changed")
    if tuple(manifest.get("scenes", {})) != SCENES:
        raise ProtocolError("scene set/order must remain Living + Slabs")
    if tuple(manifest.get("methods", ())) != METHODS:
        raise ProtocolError("method set/order changed")
    if tuple(manifest.get("learned_methods", ())) != LEARNED_METHODS:
        raise ProtocolError("learned method set changed")
    if tuple(manifest.get("model_seeds", ())) != MODEL_SEEDS:
        raise ProtocolError("model seeds changed")
    if tuple(manifest.get("render_seeds", ())) != RENDER_SEEDS:
        raise ProtocolError("render seeds changed")
    if tuple(manifest.get("new_sample_counts", ())) != NEW_SAMPLE_COUNTS:
        raise ProtocolError("new SPP grid changed")
    if int(manifest.get("reused_sample_count", 0)) != REUSED_SAMPLE_COUNT:
        raise ProtocolError("the only reusable SPP point must remain 64")
    if int(manifest.get("width", 0)) != 64 or int(manifest.get("height", 0)) != 64:
        raise ProtocolError("resolution must remain 64 x 64")
    if int(manifest.get("reference_sample_count", 0)) != 32768:
        raise ProtocolError("reference SPP changed")
    if int(manifest.get("reference_seed", -1)) != 314159:
        raise ProtocolError("reference seed changed")

    controls = manifest.get("render_controls")
    expected_controls = {
        "threads_per_worker": 4,
        "samples_per_pass": 1,
        "diffusion_steps": 10,
        "learned_direction_flip_to_receiver_hemisphere": False,
        "learned_proposal_inv_prob_clamp": 512.0,
        "learned_proposal_contribution_clamp": 32.0,
        "learned_proposal_max_trials": 2048,
        "mpg_train_budget": 0.3,
        "mpg_train_fusion": True,
        "watchdog_seconds": 1800.0,
        "diffusion_bounce_alpha": 0.0,
        "diffusion_kappa_scale": 1.0,
        "npm_bounce_alpha": 1.0,
        "npm_kappa_scale": 1.0,
        "npm_selected_training_steps": 300,
    }
    if controls != expected_controls:
        raise ProtocolError("render controls changed from the preregistration")

    source = manifest.get("source_phase20")
    if not isinstance(source, dict) or source.get("protocol_id") != SOURCE_PROTOCOL_ID:
        raise ProtocolError("source Phase 20 identity changed")
    for name in (
        "manifest",
        "coordinator_state",
        "results_receipt",
        "reference_registry",
        "npm_step_selection_receipt",
    ):
        entry = source.get(name)
        if not isinstance(entry, dict):
            raise ProtocolError(f"missing source receipt: {name}")
        workspace_path(entry.get("path"), name)
        _sha(entry.get("sha256"), f"{name}.sha256")
    if source.get("selected_controls") != {
        "bounce_alpha": 0.0,
        "control_id": "a000_k100",
        "kappa_scale": 1.0,
        "source": "results/phase_19_diffusion_mlp_equal_spp_stability_sweep/server_execution_final_v3/selection.json",
        "source_sha256": "1ca537894f2ae8c4de4db2b1e6013762de760f3cf4a501fc3136a2a82f1dea67",
    }:
        raise ProtocolError("frozen MPGD controls changed")
    for scene in SCENES:
        scene_entry = manifest["scenes"][scene]
        project_path(scene_entry.get("scene_path"), f"scene path {scene}")
        _sha(scene_entry.get("scene_xml_sha256"), f"scene XML {scene}")
        ref = source.get("references", {}).get(scene)
        if not isinstance(ref, dict):
            raise ProtocolError(f"missing frozen reference: {scene}")
        workspace_path(ref.get("path"), f"reference {scene}")
        _sha(ref.get("sha256"), f"reference {scene}")
        _sha(ref.get("active_config_sha256"), f"reference config {scene}")

    for group in ("frozen_code", "frozen_runtime"):
        entries = manifest.get(group)
        if not isinstance(entries, dict) or not entries:
            raise ProtocolError(f"{group} is missing")
        for name, entry in entries.items():
            if not isinstance(entry, dict):
                raise ProtocolError(f"invalid {group} entry: {name}")
            workspace_path(entry.get("path"), f"{group}.{name}")
            _sha(entry.get("sha256"), f"{group}.{name}.sha256")

    exports = manifest.get("model_exports")
    expected_keys = {
        f"{scene}|{method}|{seed}"
        for scene in SCENES
        for method in LEARNED_METHODS
        for seed in MODEL_SEEDS
    }
    if not isinstance(exports, dict) or set(exports) != expected_keys:
        raise ProtocolError("the twelve frozen learned runtime exports changed")
    for key, entry in exports.items():
        if not isinstance(entry, dict):
            raise ProtocolError(f"invalid model export: {key}")
        workspace_path(entry.get("path"), f"model export {key}")
        _sha(entry.get("sha256"), f"model export {key}")

    metric = manifest.get("metric")
    if not isinstance(metric, dict) or metric.get("name") != "full_image_mse":
        raise ProtocolError("metric must remain full-image MSE")
    if metric.get("domain") != "raw_linear_rgb":
        raise ProtocolError("metric domain must remain raw linear RGB")
    if metric.get("aggregation_for_curve") != "median_with_interquartile_range":
        raise ProtocolError("curve aggregation changed")
    for value in (controls["watchdog_seconds"], controls["mpg_train_budget"]):
        if not math.isfinite(float(value)):
            raise ProtocolError("nonfinite render control")
    return manifest, sha256_file(path.resolve())


def task_sort_key(task: Task) -> str:
    return hashlib.sha256(task.task_id.encode("utf-8")).hexdigest()


def build_tasks(manifest: dict[str, Any]) -> tuple[Task, ...]:
    tasks: list[Task] = []
    for scene in SCENES:
        for sample_count in NEW_SAMPLE_COUNTS:
            for method in METHODS:
                for render_seed in RENDER_SEEDS:
                    model_seeds: tuple[int | None, ...] = (
                        MODEL_SEEDS if method in LEARNED_METHODS else (None,)
                    )
                    for model_seed in model_seeds:
                        model_token = "" if model_seed is None else f"__t{model_seed}"
                        task_id = (
                            f"curve__{scene}__{method}__spp{sample_count}"
                            f"{model_token}__r{render_seed}"
                        )
                        tasks.append(Task(
                            task_id=task_id,
                            scene=scene,
                            method=method,
                            sample_count=sample_count,
                            render_seed=render_seed,
                            model_seed=model_seed,
                        ))
    return tuple(sorted(tasks, key=task_sort_key))


def expected_reused_rows() -> int:
    replicates_per_scene = (
        len(LEARNED_METHODS) * len(MODEL_SEEDS) * len(RENDER_SEEDS)
        + (len(METHODS) - len(LEARNED_METHODS)) * len(RENDER_SEEDS)
    )
    return len(SCENES) * replicates_per_scene


def expected_new_rows() -> int:
    return len(build_tasks(load_and_validate_manifest()[0]))
