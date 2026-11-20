from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
import xml.etree.ElementTree as et
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = WORKSPACE_ROOT.parent
DEFAULT_MANIFEST = (
    WORKSPACE_ROOT
    / "configs"
    / "fresh_confirmatory_evaluation_dag_20260901.json"
)
EXPECTED_PROTOCOL = "fresh_confirmatory_evaluation_20260901_v1"
MODEL_RECEIPT_PROTOCOL = "fresh_scene_smoke_registration_20260901_v1"
SMOKE_RECEIPT_PROTOCOL = "fresh_scene_smoke_gate_20260901_v1"
TRAINING_PROTOCOL = "fresh_confirmatory_models_20260901_v1"
AUTHORIZATION_TOKEN = "FRESH_CONFIRMATORY_EVALUATION_APPROVED"
EXPECTED_SCENES = ("plane", "stone", "sphere", "pool", "living")
EXPECTED_TRAIN_SEEDS = (104729, 130363, 155921)
EXPECTED_RENDER_SEEDS = (1729, 31415, 271828, 424242, 8675309)
EXPECTED_REFERENCE_NOISE_SEED = 161803
EXPECTED_METHODS = ("ordered", "simple", "mpg")
REGIMES = ("equal_spp", "equal_time")
SMOKE_PINNED_INPUT_KEYS = (
    "configs/baseline_manifest.json",
    "configs/runtime_profiles.json",
    "masked_diffusion_workspace/scripts/run_fresh_scene_smoke_gate.py",
    "mitsuba/build_macos_arm64_py312/dist/mitsuba",
    "mitsuba/build_macos_arm64_py312/dist/plugins/manifold_path_guiding.dylib",
    "mitsuba/build_macos_arm64_py312/dist/plugins/path.dylib",
    "registrar",
    "registration_receipt",
    "scripts/phase_13_run_formal_benchmark.py",
    "scripts/run_phase_13_status_only.py",
    "smoke_generator",
    "smoke_manifest",
    "src/masked_diffusion_mpg/__init__.py",
    "src/masked_diffusion_mpg/build.py",
    "src/masked_diffusion_mpg/manifest.py",
    "src/masked_diffusion_mpg/runtime.py",
    "training_manifest",
    "training_state",
)
RUNTIME_MARKER = "Loaded masked diffusion runtime model from"
TRAINING_STATE = (
    WORKSPACE_ROOT
    / "results"
    / "fresh_confirmatory_models_20260901"
    / "orchestrator_status.json"
)
CONFLICT_MARKERS = (
    "phase_10_train_masked_diffusion.py",
    "phase_11_compare_guidance_models.py",
    "phase_12_export_runtime_model.py",
    "phase_13_run_formal_benchmark.py",
    "finalize_phase13_deferred_metrics.py",
    "run_fresh_confirmatory_model_artifacts.py",
    "register_fresh_scene_smoke_artifacts.py",
    "mitsuba",
)


class FreshEvaluationError(RuntimeError):
    """Raised when a frozen protocol or information-barrier check fails."""


@dataclass(frozen=True, slots=True)
class DagNode:
    node_id: str
    stage: str
    dependencies: tuple[str, ...]
    render_count: int = 0
    quality_access: bool = False
    conditional: bool = False
    scene: str | None = None
    method: str | None = None
    train_seed: int | None = None
    render_seed: int | None = None
    sample_count: int | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Check the frozen fresh-scene evaluation DAG. The default is strictly "
            "read-only; execution additionally requires a reviewed manifest, all "
            "hash-pinned dependency receipts, and an exact authorization token."
        )
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--authorization-token", default="")
    parser.add_argument(
        "--python-executable",
        default=sys.executable,
        help="Python interpreter that can import NumPy and the Phase 13 modules.",
    )
    parser.add_argument(
        "--max-nodes",
        type=int,
        default=None,
        help="Future resumable execution cap; rejected in check-only mode.",
    )
    parser.add_argument(
        "--print-node-ids",
        action="store_true",
        help="Print the expanded logical node IDs without writing them.",
    )
    parser.add_argument(
        "--rerun-stale",
        action="store_true",
        help="Allow a new attempt for stale protocol-scoped outputs where safe.",
    )
    return parser.parse_args()


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FreshEvaluationError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise FreshEvaluationError(f"expected a JSON object: {path}")
    return payload


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def workspace_path(value: str) -> Path:
    path = Path(value)
    resolved = path.resolve() if path.is_absolute() else (WORKSPACE_ROOT / path).resolve()
    try:
        resolved.relative_to(WORKSPACE_ROOT.resolve())
    except ValueError as exc:
        raise FreshEvaluationError(f"path escapes workspace: {value}") from exc
    return resolved


def project_input_path(value: str) -> Path:
    """Resolve immutable renderer inputs, whose paths are project-root relative."""

    path = Path(value)
    resolved = path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()
    try:
        resolved.relative_to(PROJECT_ROOT.resolve())
    except ValueError as exc:
        raise FreshEvaluationError(f"path escapes project: {value}") from exc
    return resolved


def portable_project_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError as exc:
        raise FreshEvaluationError(f"asset escapes project: {resolved}") from exc


def _expand_scene_filename(value: str, defaults: dict[str, str]) -> str:
    """Resolve the exact `$name`/`${name}` filename defaults used by Mitsuba XML."""

    expanded = value.strip()
    seen: set[str] = set()
    while expanded.startswith("$"):
        name = (
            expanded[2:-1]
            if expanded.startswith("${") and expanded.endswith("}")
            else expanded[1:]
        )
        if not name or name in seen or name not in defaults:
            raise FreshEvaluationError(f"unresolved/cyclic scene filename variable: {value}")
        seen.add(name)
        expanded = defaults[name].strip()
    return expanded


def discover_transitive_scene_assets(scene_path: Path) -> tuple[Path, ...]:
    """Discover every file referenced by scene XML, recursively following includes."""

    root_scene = scene_path.resolve()
    try:
        root_scene.relative_to(PROJECT_ROOT.resolve())
    except ValueError as exc:
        raise FreshEvaluationError(f"scene XML escapes project: {scene_path}") from exc
    pending = [root_scene]
    parsed_xml: set[Path] = set()
    assets: set[Path] = set()
    while pending:
        xml_path = pending.pop()
        if xml_path in parsed_xml:
            continue
        if not xml_path.is_file():
            raise FreshEvaluationError(f"missing scene XML/include: {xml_path}")
        parsed_xml.add(xml_path)
        root = et.parse(xml_path).getroot()
        defaults = {
            str(element.attrib.get("name")): str(element.attrib.get("value"))
            for element in root.iter("default")
            if element.attrib.get("name") and element.attrib.get("value") is not None
        }
        parent_map = {child: parent for parent in root.iter() for child in parent}
        references: list[str] = []
        for element in root.iter("include"):
            if element.attrib.get("filename"):
                references.append(str(element.attrib["filename"]))
        for element in root.iter("string"):
            if element.attrib.get("name") != "filename":
                continue
            parent = parent_map.get(element)
            if parent is not None and parent.tag == "film":
                continue
            if element.attrib.get("value"):
                references.append(str(element.attrib["value"]))
        for raw_value in references:
            value = _expand_scene_filename(raw_value, defaults)
            if "://" in value:
                raise FreshEvaluationError(
                    f"scene asset cannot be hash-pinned because it is remote: {value}"
                )
            candidate = (xml_path.parent / value).resolve()
            try:
                candidate.relative_to(PROJECT_ROOT.resolve())
            except ValueError as exc:
                raise FreshEvaluationError(
                    f"scene asset escapes project: {xml_path} -> {value}"
                ) from exc
            if not candidate.is_file():
                raise FreshEvaluationError(
                    f"missing scene asset: {xml_path} -> {value}"
                )
            assets.add(candidate)
            if candidate.suffix.lower() == ".xml":
                pending.append(candidate)
    return tuple(sorted(assets, key=portable_project_path))


def validate_scene_asset_inventory(
    manifest: dict[str, Any],
) -> tuple[Path, dict[str, Path]]:
    """Validate the exact discovered asset set and every pinned asset byte hash."""

    spec = dict(manifest.get("scene_asset_inventory", {}))
    if spec.get("protocol_id") != "fresh_confirmatory_scene_assets_20260901_v1":
        raise FreshEvaluationError("scene asset inventory manifest protocol changed")
    inventory_path = validate_hash_pinned_file(spec, "scene asset inventory")
    inventory = load_json(inventory_path)
    if (
        inventory.get("schema_version") != 1
        or inventory.get("protocol_id")
        != "fresh_confirmatory_scene_assets_20260901_v1"
        or inventory.get("path_base") != "project_root"
    ):
        raise FreshEvaluationError("scene asset inventory identity changed")
    entries = inventory.get("scenes")
    if not isinstance(entries, dict) or set(entries) != set(EXPECTED_SCENES):
        raise FreshEvaluationError("scene asset inventory scene set changed")
    all_assets: dict[str, Path] = {}
    total = 0
    for scene in manifest["study"]["scenes"]:
        scene_key = str(scene["key"])
        entry = entries.get(scene_key)
        if not isinstance(entry, dict):
            raise FreshEvaluationError(f"missing scene asset entry: {scene_key}")
        if (
            entry.get("scene_path") != scene["scene_path"]
            or entry.get("scene_sha256") != scene["scene_sha256"]
        ):
            raise FreshEvaluationError(f"scene asset root drift: {scene_key}")
        expected_assets = entry.get("assets")
        if not isinstance(expected_assets, dict):
            raise FreshEvaluationError(f"scene asset map is invalid: {scene_key}")
        scene_path = project_input_path(str(scene["scene_path"]))
        discovered = {
            portable_project_path(path): path
            for path in discover_transitive_scene_assets(scene_path)
        }
        if set(discovered) != set(expected_assets):
            missing = sorted(set(expected_assets) - set(discovered))
            added = sorted(set(discovered) - set(expected_assets))
            raise FreshEvaluationError(
                f"scene asset inventory drift for {scene_key}; "
                f"missing={missing}, added={added}"
            )
        for path_text, path in discovered.items():
            expected_sha = expected_assets[path_text]
            if not isinstance(expected_sha, str) or sha256_file(path) != expected_sha:
                raise FreshEvaluationError(f"scene asset hash drift: {path_text}")
            if path_text in all_assets and all_assets[path_text] != path:
                raise FreshEvaluationError(f"ambiguous scene asset path: {path_text}")
            all_assets[path_text] = path
        total += len(discovered)
    if inventory.get("asset_reference_count") != total:
        raise FreshEvaluationError("scene asset inventory reference count changed")
    if inventory.get("unique_asset_count") != len(all_assets):
        raise FreshEvaluationError("scene asset inventory unique count changed")
    return inventory_path, all_assets


def require_exact_keys(payload: dict[str, Any], keys: Iterable[str], label: str) -> None:
    missing = sorted(set(keys) - set(payload))
    if missing:
        raise FreshEvaluationError(f"{label} is missing keys: {', '.join(missing)}")


def validate_hash_pinned_file(spec: dict[str, Any], label: str) -> Path:
    require_exact_keys(spec, ("path", "sha256"), label)
    expected = spec.get("sha256")
    if not isinstance(expected, str) or len(expected) != 64:
        raise FreshEvaluationError(f"{label}.sha256 is not pinned")
    path = workspace_path(str(spec["path"]))
    if not path.is_file():
        raise FreshEvaluationError(f"missing {label}: {path}")
    actual = sha256_file(path)
    if actual != expected:
        raise FreshEvaluationError(
            f"{label} SHA-256 mismatch: expected {expected}, got {actual}"
        )
    return path


def dotted_get(payload: dict[str, Any], dotted: str) -> Any:
    current: Any = payload
    for part in dotted.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def placeholder_blockers(manifest: dict[str, Any]) -> list[str]:
    blockers: list[str] = []
    declared = manifest.get("fail_closed_placeholders", [])
    if not isinstance(declared, list):
        raise FreshEvaluationError("fail_closed_placeholders must be a list")
    for dotted in declared:
        value = dotted_get(manifest, str(dotted))
        if value is None or value == "" or value == "pending_interface_hash":
            blockers.append(str(dotted))
    interface_status = dotted_get(manifest, "runner_interfaces.shared_reference.status")
    if interface_status != "ready":
        blockers.append("runner_interfaces.shared_reference.status")
    if manifest.get("execution_authorized") is not True:
        blockers.append("execution_authorized")
    return sorted(set(blockers))


def _validate_study(manifest: dict[str, Any]) -> None:
    study = manifest.get("study")
    if not isinstance(study, dict):
        raise FreshEvaluationError("study must be an object")
    if study.get("profile_id") != "macos_arm64_native":
        raise FreshEvaluationError("runtime profile differs from preregistration")
    if study.get("resolution") != {
        "width": 64,
        "height": 64,
        "crop_width": 64,
        "crop_height": 64,
    }:
        raise FreshEvaluationError("render resolution/crop differs from preregistration")
    scenes = study.get("scenes")
    if not isinstance(scenes, list):
        raise FreshEvaluationError("study.scenes must be a list")
    scene_keys = tuple(str(scene.get("key")) for scene in scenes)
    if scene_keys != EXPECTED_SCENES:
        raise FreshEvaluationError(f"scene order/set changed: {scene_keys}")
    for scene in scenes:
        expected_job_id = f"paper_{scene['key']}_mpg"
        if scene.get("job_id") != expected_job_id:
            raise FreshEvaluationError(
                f"scene job ID changed for {scene['key']}: {scene.get('job_id')!r}"
            )
        path = project_input_path(str(scene.get("scene_path", "")))
        if not path.is_file():
            raise FreshEvaluationError(f"missing scene XML: {path}")
        actual = sha256_file(path)
        if actual != scene.get("scene_sha256"):
            raise FreshEvaluationError(
                f"scene XML hash mismatch for {scene.get('key')}: {actual}"
            )
    if tuple(int(seed) for seed in study.get("training_seeds", [])) != EXPECTED_TRAIN_SEEDS:
        raise FreshEvaluationError("training seeds differ from preregistration")
    if tuple(int(seed) for seed in study.get("render_seeds", [])) != EXPECTED_RENDER_SEEDS:
        raise FreshEvaluationError("render seeds differ from preregistration")
    if int(study.get("reference_seed", -1)) != 314159:
        raise FreshEvaluationError("reference seed differs from preregistration")
    if int(study.get("reference_noise_seed", -1)) != EXPECTED_REFERENCE_NOISE_SEED:
        raise FreshEvaluationError(
            "reference-noise seed differs from preregistration"
        )
    methods = study.get("methods")
    if not isinstance(methods, dict) or tuple(methods) != EXPECTED_METHODS:
        raise FreshEvaluationError("method order/set differs from preregistration")
    expected_method_controls = {
        "ordered": (
            "ordered_diffusion_mlp_h72_k4",
            True,
            0.0,
            1.0,
            False,
            55599,
        ),
        "simple": ("simple_mlp_k4", True, 0.5, 0.25, False, 55857),
        "mpg": ("mpg", False, 0.5, 1.0, False, None),
    }
    for method, expected in expected_method_controls.items():
        spec = methods[method]
        observed = (
            str(spec.get("phase13_method")),
            spec.get("train_seeded"),
            spec.get("bounce_alpha"),
            spec.get("learned_direction_kappa_scale"),
            spec.get("learned_direction_flip_to_receiver_hemisphere"),
            spec.get("expected_parameters"),
        )
        if observed != expected:
            raise FreshEvaluationError(
                f"frozen controls changed for {method}: {observed}"
            )
    if (
        methods["mpg"].get("mpg_train_budget") != 0.2
        or methods["mpg"].get("mpg_train_fusion") is not True
    ):
        raise FreshEvaluationError("MPG online-training controls changed")
    runtime = study.get("common_runtime", {})
    expected_runtime = {
        "samples_per_pass": 1,
        "threads": 4,
        "diffusion_steps": 10,
        "learned_proposal_inv_prob_clamp": 512.0,
        "learned_proposal_contribution_clamp": 32.0,
        "learned_proposal_max_trials": 2048,
        "time_budget_seconds": 300.0,
    }
    for key, expected in expected_runtime.items():
        if runtime.get(key) != expected:
            raise FreshEvaluationError(
                f"common runtime {key}={runtime.get(key)!r}, expected {expected!r}"
            )


def _validate_renderer_inputs(manifest: dict[str, Any]) -> None:
    inputs = manifest.get("renderer_inputs")
    if not isinstance(inputs, dict) or set(inputs) != {
        "baseline_manifest",
        "runtime_profiles",
        "mitsuba_cli",
        "manifold_path_guiding_plugin",
    }:
        raise FreshEvaluationError("renderer input set changed")
    for label, spec in inputs.items():
        base = spec.get("path_base")
        if base == "workspace_root":
            path = workspace_path(str(spec.get("path", "")))
        elif base == "project_root":
            path = project_input_path(str(spec.get("path", "")))
        else:
            raise FreshEvaluationError(f"invalid path base for renderer input {label}")
        if not path.is_file():
            raise FreshEvaluationError(f"missing renderer input {label}: {path}")
        actual = sha256_file(path)
        if actual != spec.get("sha256"):
            raise FreshEvaluationError(
                f"renderer input SHA-256 mismatch for {label}: {actual}"
            )


def _validate_frozen_training_inputs(training_manifest: dict[str, Any]) -> None:
    """Recheck the frozen 12-code/60-config/5-replay training boundary."""

    code = training_manifest.get("code_provenance")
    if not isinstance(code, dict) or len(code) != 12:
        raise FreshEvaluationError("frozen training code provenance is not 12 files")
    for path_text, expected_sha in code.items():
        path = workspace_path(str(path_text))
        if not path.is_file() or sha256_file(path) != expected_sha:
            raise FreshEvaluationError(f"frozen training code drift: {path_text}")
    jobs = training_manifest.get("jobs")
    if not isinstance(jobs, list) or len(jobs) != 60:
        raise FreshEvaluationError("frozen training manifest no longer has 60 jobs")
    observed: set[str] = set()
    completed: set[str] = set()
    for job in jobs:
        job_id = str(job.get("job_id", ""))
        if not job_id or job_id in observed:
            raise FreshEvaluationError(f"invalid frozen training job ID: {job_id!r}")
        observed.add(job_id)
        for dependency in job.get("dependencies", []):
            if dependency not in completed:
                raise FreshEvaluationError(
                    f"frozen training job has forward/missing dependency: {job_id}"
                )
        config_path = workspace_path(str(job.get("config_path", "")))
        script_path = workspace_path(str(job.get("script", "")))
        if not config_path.is_file() or sha256_file(config_path) != job.get(
            "config_sha256"
        ):
            raise FreshEvaluationError(f"frozen training config drift: {job_id}")
        if not script_path.is_file() or sha256_file(script_path) != job.get(
            "script_sha256"
        ):
            raise FreshEvaluationError(f"frozen training job script drift: {job_id}")
        load_json(config_path)
        completed.add(job_id)
    scenes = training_manifest.get("scenes")
    if not isinstance(scenes, list) or len(scenes) != 5:
        raise FreshEvaluationError("frozen training replay registry is not five scenes")
    for scene in scenes:
        replay_path = workspace_path(str(scene.get("replay_path", "")))
        if not replay_path.is_file() or sha256_file(replay_path) != scene.get(
            "replay_sha256"
        ):
            raise FreshEvaluationError(
                f"frozen replay drift: {scene.get('scene')}"
            )


def _validate_stage_contract(manifest: dict[str, Any]) -> None:
    stages = manifest.get("stages")
    if not isinstance(stages, dict):
        raise FreshEvaluationError("stages must be an object")
    expected_counts = {
        "shared_references.render_count": 5,
        "shared_references.sample_count": 32768,
        "reference_noise_check.render_count": 5,
        "reference_noise_check.sample_count": 16384,
        "equal_spp.sample_count": 64,
        "equal_spp.ordered_render_count": 75,
        "equal_spp.simple_render_count": 75,
        "equal_spp.mpg_render_count": 25,
        "equal_spp.total_render_count": 175,
        "timing_calibration.cells": 15,
        "timing_calibration.base_probe_count": 45,
        "equal_time.ordered_render_count": 75,
        "equal_time.simple_render_count": 75,
        "equal_time.mpg_render_count": 25,
        "equal_time.total_render_count": 175,
        "metric_barrier.candidate_render_count_required": 350,
        "final_summary.learned_vs_simple_pairs_per_scene": 15,
        "final_summary.learned_vs_mpg_pairs_per_scene": 5,
        "final_summary.strong_claim_scene_majority": 3,
    }
    for dotted, expected in expected_counts.items():
        value = dotted_get({"stages": stages}, f"stages.{dotted}")
        if value != expected:
            raise FreshEvaluationError(
                f"stage contract {dotted}={value!r}, expected {expected!r}"
            )
    dependency_gate = stages["dependency_gates"]
    if dependency_gate != {
        "model_artifact_receipt_count": 1,
        "status_only_smoke_receipt_count": 1,
        "block_until_both_hash_pinned": True,
    }:
        raise FreshEvaluationError("dependency-gate contract changed")
    shared = stages["shared_references"]
    if shared.get("method") != "path_reference" or shared.get(
        "single_shared_manifest"
    ) is not True:
        raise FreshEvaluationError("shared-reference method/manifest contract changed")
    if stages["equal_spp"].get("defer_metrics") is not True:
        raise FreshEvaluationError("equal-SPP metrics must be deferred")
    if stages["equal_time"].get("defer_metrics") is not True:
        raise FreshEvaluationError("equal-time metrics must be deferred")
    if (
        stages["reference_noise_check"].get("method") != "pt"
        or stages["reference_noise_check"].get("metric_deferred_until_unlock")
        is not True
    ):
        raise FreshEvaluationError(
            "reference-noise checks must use deferred external-reference pt"
        )
    if stages["reference_noise_check"].get("watchdog_seconds") != 1800.0:
        raise FreshEvaluationError(
            "reference-noise checks require the frozen 1800-second watchdog"
        )
    if stages["timing_calibration"].get("quality_metrics_forbidden") is not True:
        raise FreshEvaluationError("timing calibration must remain quality-blind")
    if stages["timing_calibration"].get("repeat_trigger") != (
        "absolute_difference_between_two_16spp_whole_process_times_"
        "divided_by_their_mean"
    ):
        raise FreshEvaluationError("timing repeat trigger changed")
    if stages["timing_calibration"].get("mandatory_16spp_repeats_per_cell") != 2:
        raise FreshEvaluationError("calibration must use two initial 16-spp probes")
    if stages["timing_calibration"].get("fit_time_source") != (
        "whole_process_wall_seconds_including_startup"
    ):
        raise FreshEvaluationError("calibration must include process startup")
    timing = stages["timing_calibration"]
    expected_timing = {
        "blind": True,
        "probe_sample_counts": [1, 16],
        "representative_training_seed": 104729,
        "conditional_16spp_repeat_count_max": 15,
        "repeat_threshold_fraction": 0.1,
        "target_seconds": 120.0,
        "watchdog_seconds": 300.0,
        "fit": "time_seconds = startup_seconds + spp * marginal_seconds",
        "rounding": "largest_positive_integer_spp_predicted_not_to_exceed_target",
    }
    for key, expected in expected_timing.items():
        if timing.get(key) != expected:
            raise FreshEvaluationError(
                f"timing calibration {key} differs from preregistration"
            )
    if stages["equal_time"].get("spp_source") != "frozen_calibration_map":
        raise FreshEvaluationError("equal-time SPP source changed")
    barrier = stages["metric_barrier"]
    if barrier.get("require_all_successful") is not True:
        raise FreshEvaluationError("metric barrier must require every candidate")
    if barrier.get("require_candidate_metrics_null_before_unlock") is not True:
        raise FreshEvaluationError("metric barrier must require null candidate metrics")
    if (
        barrier.get("require_candidate_summary_hashes") is not True
        or barrier.get("require_reference_manifest_hash") is not True
    ):
        raise FreshEvaluationError("metric barrier provenance requirements changed")
    final_summary = stages["final_summary"]
    expected_summary = {
        "primary_metric": "linear_rgb_full_frame_mse_float64",
        "mse_floor": 1e-20,
        "bootstrap_seed": 20260901,
        "bootstrap_replicates": 10000,
        "equal_scene_weighting": True,
        "hierarchical_bootstrap": True,
    }
    for key, expected in expected_summary.items():
        if final_summary.get(key) != expected:
            raise FreshEvaluationError(
                f"final summary {key} differs from preregistration"
            )


def validate_manifest(manifest_path: Path) -> tuple[dict[str, Any], str]:
    manifest_path = manifest_path.resolve()
    manifest = load_json(manifest_path)
    if manifest.get("schema_version") != 1:
        raise FreshEvaluationError("unexpected manifest schema version")
    if manifest.get("protocol_id") != EXPECTED_PROTOCOL:
        raise FreshEvaluationError(
            f"unexpected protocol: {manifest.get('protocol_id')!r}"
        )
    schema_path = validate_hash_pinned_file(
        dict(manifest.get("manifest_schema", {})), "evaluation manifest schema"
    )
    schema = load_json(schema_path)
    if schema.get("$id") != "fresh_confirmatory_evaluation_dag_20260901.schema.json":
        raise FreshEvaluationError("evaluation manifest schema identity mismatch")
    plan_path = validate_hash_pinned_file(dict(manifest.get("plan", {})), "plan")
    if plan_path.name != "FRESH_SCENE_EVALUATION_PLAN_20260901.md":
        raise FreshEvaluationError("wrong preregistration plan")
    training_spec = dict(manifest.get("frozen_training_manifest", {}))
    training_path = validate_hash_pinned_file(
        training_spec, "frozen training manifest"
    )
    training_manifest = load_json(training_path)
    if training_spec.get("read_only") is not True:
        raise FreshEvaluationError("training manifest must be declared read-only")
    if training_manifest.get("protocol_id") != TRAINING_PROTOCOL:
        raise FreshEvaluationError("training manifest protocol mismatch")
    _validate_frozen_training_inputs(training_manifest)
    _validate_study(manifest)
    validate_scene_asset_inventory(manifest)
    _validate_renderer_inputs(manifest)
    _validate_stage_contract(manifest)
    validate_runner_interfaces(manifest)
    return manifest, sha256_file(manifest_path)


def _candidate_nodes(manifest: dict[str, Any], regime: str) -> list[DagNode]:
    nodes: list[DagNode] = []
    scenes = tuple(scene["key"] for scene in manifest["study"]["scenes"])
    train_seeds = tuple(manifest["study"]["training_seeds"])
    render_seeds = tuple(manifest["study"]["render_seeds"])
    dependency = (
        "reference:shared_block"
        if regime == "equal_spp"
        else "calibration:freeze_spp_map"
    )
    for method in ("ordered", "simple"):
        for scene in scenes:
            for train_seed in train_seeds:
                for render_seed in render_seeds:
                    nodes.append(
                        DagNode(
                            node_id=(
                                f"candidate:{regime}:{scene}:{method}:"
                                f"train{train_seed}:render{render_seed}"
                            ),
                            stage=regime,
                            dependencies=(
                                "gate:status_only_smoke",
                                dependency,
                            ),
                            render_count=1,
                            scene=scene,
                            method=method,
                            train_seed=int(train_seed),
                            render_seed=int(render_seed),
                            sample_count=(64 if regime == "equal_spp" else None),
                        )
                    )
    for scene in scenes:
        for render_seed in render_seeds:
            nodes.append(
                DagNode(
                    node_id=f"candidate:{regime}:{scene}:mpg:render{render_seed}",
                    stage=regime,
                    dependencies=("gate:status_only_smoke", dependency),
                    render_count=1,
                    scene=scene,
                    method="mpg",
                    render_seed=int(render_seed),
                    sample_count=(64 if regime == "equal_spp" else None),
                )
            )
    return nodes


def build_dag(manifest: dict[str, Any]) -> tuple[DagNode, ...]:
    nodes: list[DagNode] = [
        DagNode("gate:model_artifacts", "dependency_gate", ()),
        DagNode(
            "gate:status_only_smoke",
            "dependency_gate",
            ("gate:model_artifacts",),
        ),
        DagNode(
            "reference:shared_block",
            "shared_references",
            ("gate:status_only_smoke",),
            render_count=5,
            sample_count=32768,
        ),
    ]
    scenes = tuple(scene["key"] for scene in manifest["study"]["scenes"])
    for scene in scenes:
        nodes.append(
            DagNode(
                f"reference_noise:{scene}",
                "reference_noise_check",
                ("reference:shared_block",),
                render_count=1,
                scene=scene,
                method="pt",
                sample_count=16384,
            )
        )
    equal_spp = _candidate_nodes(manifest, "equal_spp")
    nodes.extend(equal_spp)
    representative_seed = int(
        manifest["stages"]["timing_calibration"]["representative_training_seed"]
    )
    base_calibration_ids: list[str] = []
    repeat_calibration_ids: list[str] = []
    timing_seed_1, timing_seed_2, timing_seed_3 = tuple(
        int(seed) for seed in manifest["study"]["render_seeds"][:3]
    )
    for scene in scenes:
        for method in EXPECTED_METHODS:
            for node_suffix, spp, render_seed in (
                ("spp1", 1, timing_seed_1),
                ("spp16:rep1", 16, timing_seed_1),
                ("spp16:rep2", 16, timing_seed_2),
            ):
                node_id = f"calibration:{scene}:{method}:{node_suffix}"
                base_calibration_ids.append(node_id)
                nodes.append(
                    DagNode(
                        node_id,
                        "timing_calibration",
                        ("gate:status_only_smoke", "reference:shared_block"),
                        render_count=1,
                        scene=scene,
                        method=method,
                        train_seed=(representative_seed if method != "mpg" else None),
                        render_seed=render_seed,
                        sample_count=spp,
                    )
                )
            repeat_id = f"calibration:{scene}:{method}:spp16:conditional_repeat"
            repeat_calibration_ids.append(repeat_id)
            nodes.append(
                DagNode(
                    repeat_id,
                    "timing_calibration_repeat",
                    (
                        f"calibration:{scene}:{method}:spp16:rep1",
                        f"calibration:{scene}:{method}:spp16:rep2",
                    ),
                    render_count=1,
                    scene=scene,
                    method=method,
                    train_seed=(representative_seed if method != "mpg" else None),
                    render_seed=timing_seed_3,
                    sample_count=16,
                    conditional=True,
                )
            )
    nodes.append(
        DagNode(
            "calibration:freeze_spp_map",
            "freeze_spp_map",
            tuple(base_calibration_ids + repeat_calibration_ids),
        )
    )
    equal_time = _candidate_nodes(manifest, "equal_time")
    nodes.extend(equal_time)
    candidate_ids = tuple(node.node_id for node in equal_spp + equal_time)
    barrier_dependencies = (
        "reference:shared_block",
        *(f"reference_noise:{scene}" for scene in scenes),
        *candidate_ids,
    )
    nodes.append(
        DagNode(
            "barrier:unlock_candidate_metrics",
            "metric_barrier",
            tuple(barrier_dependencies),
        )
    )
    finalizer_ids: list[str] = []
    for scene in scenes:
        node_id = f"finalize:reference_noise:{scene}"
        finalizer_ids.append(node_id)
        nodes.append(
            DagNode(
                node_id,
                "deferred_metric_finalization",
                (
                    "barrier:unlock_candidate_metrics",
                    f"reference_noise:{scene}",
                ),
                quality_access=True,
                scene=scene,
                method="path_reference_noise_check",
            )
        )
    for candidate in equal_spp + equal_time:
        node_id = "finalize:" + candidate.node_id.removeprefix("candidate:")
        finalizer_ids.append(node_id)
        nodes.append(
            DagNode(
                node_id,
                "deferred_metric_finalization",
                ("barrier:unlock_candidate_metrics", candidate.node_id),
                quality_access=True,
                scene=candidate.scene,
                method=candidate.method,
                train_seed=candidate.train_seed,
                render_seed=candidate.render_seed,
            )
        )
    nodes.append(
        DagNode(
            "summary:hierarchical",
            "hierarchical_summary",
            tuple(finalizer_ids),
            quality_access=True,
        )
    )
    validate_dag(tuple(nodes), manifest)
    return tuple(nodes)


def validate_dag(nodes: tuple[DagNode, ...], manifest: dict[str, Any]) -> None:
    node_by_id: dict[str, DagNode] = {}
    for node in nodes:
        if node.node_id in node_by_id:
            raise FreshEvaluationError(f"duplicate DAG node: {node.node_id}")
        for dependency in node.dependencies:
            if dependency not in node_by_id:
                raise FreshEvaluationError(
                    f"forward/missing dependency {dependency} for {node.node_id}"
                )
        node_by_id[node.node_id] = node
    counts = Counter(node.stage for node in nodes)
    expected = {
        "dependency_gate": 2,
        "shared_references": 1,
        "reference_noise_check": 5,
        "equal_spp": 175,
        "timing_calibration": 45,
        "timing_calibration_repeat": 15,
        "freeze_spp_map": 1,
        "equal_time": 175,
        "metric_barrier": 1,
        "deferred_metric_finalization": 355,
        "hierarchical_summary": 1,
    }
    if counts != Counter(expected):
        raise FreshEvaluationError(f"unexpected DAG cardinality: {dict(counts)}")
    barrier_id = "barrier:unlock_candidate_metrics"
    candidate_nodes = [
        node for node in nodes if node.stage in {"equal_spp", "equal_time"}
    ]
    if sum(node.render_count for node in candidate_nodes) != 350:
        raise FreshEvaluationError("candidate render count is not exactly 350")
    for node in nodes:
        if node.quality_access and barrier_id not in node.dependencies and node.stage != "hierarchical_summary":
            raise FreshEvaluationError(
                f"quality-access node bypasses metric barrier: {node.node_id}"
            )
        if node.stage in {"equal_spp", "equal_time", "timing_calibration", "timing_calibration_repeat"} and node.quality_access:
            raise FreshEvaluationError(f"pre-unlock render may access quality: {node.node_id}")
    summary = node_by_id["summary:hierarchical"]
    if len(summary.dependencies) != 355:
        raise FreshEvaluationError(
            "hierarchical summary must wait for 350 candidates and 5 noise checks"
        )
    if int(
        manifest["stages"]["metric_barrier"]["candidate_render_count_required"]
    ) != len(candidate_nodes):
        raise FreshEvaluationError("manifest barrier count and expanded DAG differ")


def fit_equal_time_spp(
    *, seconds_at_1: float, seconds_at_16: float, target_seconds: float = 120.0
) -> dict[str, float | int]:
    """Fit the preregistered affine timing model without looking at image quality."""

    values = (seconds_at_1, seconds_at_16, target_seconds)
    if not all(math.isfinite(value) and value > 0.0 for value in values):
        raise FreshEvaluationError("timing inputs must be finite and positive")
    marginal = (seconds_at_16 - seconds_at_1) / 15.0
    if not math.isfinite(marginal) or marginal <= 0.0:
        raise FreshEvaluationError("nonpositive marginal time; calibration is invalid")
    startup = max(0.0, seconds_at_1 - marginal)
    budget = target_seconds - startup
    if budget < marginal:
        raise FreshEvaluationError("120-second target cannot fit one sample")
    spp = max(1, int(math.floor(budget / marginal)))
    predicted = startup + spp * marginal
    while spp > 1 and predicted > target_seconds:
        spp -= 1
        predicted = startup + spp * marginal
    return {
        "startup_seconds": startup,
        "marginal_seconds_per_spp": marginal,
        "target_seconds": target_seconds,
        "sample_count": spp,
        "predicted_seconds": predicted,
    }


def validate_model_receipt(
    manifest: dict[str, Any], receipt_spec: dict[str, Any]
) -> tuple[Path, dict[str, Any]]:
    receipt_path = validate_hash_pinned_file(receipt_spec, "model artifact receipt")
    receipt = load_json(receipt_path)
    if receipt.get("schema_version") != 1 or receipt.get("protocol_id") != MODEL_RECEIPT_PROTOCOL:
        raise FreshEvaluationError("model artifact receipt protocol mismatch")
    if receipt.get("quality_metrics_inspected") is not False:
        raise FreshEvaluationError("model artifact receipt crossed the quality barrier")
    if int(receipt.get("validated_training_jobs", -1)) != 60:
        raise FreshEvaluationError("model artifact receipt did not validate 60 jobs")
    expected_training_sha = manifest["frozen_training_manifest"]["sha256"]
    if receipt.get("training_manifest_sha256") != expected_training_sha:
        raise FreshEvaluationError("model receipt training-manifest hash mismatch")
    if not TRAINING_STATE.is_file():
        raise FreshEvaluationError(f"missing pinned training state: {TRAINING_STATE}")
    if sha256_file(TRAINING_STATE) != receipt.get("training_state_sha256"):
        raise FreshEvaluationError("training state differs from model artifact receipt")
    training_state = load_json(TRAINING_STATE)
    statuses = Counter(
        str(record.get("status", "unknown"))
        for record in training_state.get("jobs", {}).values()
    )
    if statuses != Counter({"succeeded": 60}):
        raise FreshEvaluationError(
            f"pinned training state is not 60/60 successful: {dict(statuses)}"
        )
    selected = receipt.get("selected_artifacts")
    if not isinstance(selected, dict) or set(selected) != set(EXPECTED_SCENES):
        raise FreshEvaluationError("model artifact receipt scene set is incomplete")
    for scene, methods in selected.items():
        if not isinstance(methods, dict) or set(methods) != {
            "ordered_diffusion_mlp_h72_k4",
            "simple_mlp_k4",
        }:
            raise FreshEvaluationError(f"model artifact methods changed for {scene}")
        for method, artifact in methods.items():
            for path_key, hash_key in (
                ("checkpoint_path", "checkpoint_sha256"),
                ("runtime_path", "runtime_sha256"),
            ):
                path = workspace_path(str(artifact.get(path_key, "")))
                if not path.is_file() or sha256_file(path) != artifact.get(hash_key):
                    raise FreshEvaluationError(
                        f"registered {scene}/{method} {path_key} hash drift"
                    )
    return receipt_path, receipt


def validate_smoke_receipt(
    *, model_receipt_path: Path, receipt_spec: dict[str, Any]
) -> tuple[Path, dict[str, Any]]:
    receipt_path = validate_hash_pinned_file(receipt_spec, "status-only smoke receipt")
    receipt = load_json(receipt_path)
    if receipt.get("schema_version") != 1 or receipt.get("protocol_id") != SMOKE_RECEIPT_PROTOCOL:
        raise FreshEvaluationError("status-only smoke receipt protocol mismatch")
    if receipt.get("all_passed") is not True:
        raise FreshEvaluationError("status-only smoke receipt is not an all-pass")
    if receipt.get("quality_metrics_inspected") is not False:
        raise FreshEvaluationError("smoke receipt says quality metrics were inspected")
    if receipt.get("quality_metrics_computed") is not False:
        raise FreshEvaluationError("smoke receipt does not prove status-only execution")
    model_sha = sha256_file(model_receipt_path)
    linked = receipt.get("registration_receipt_sha256")
    if linked != model_sha:
        raise FreshEvaluationError(
            "smoke receipt does not hash-link the model artifact receipt"
        )
    registration_path = phase13_artifact_path(
        str(receipt.get("registration_receipt_path", ""))
    )
    if registration_path != model_receipt_path.resolve():
        raise FreshEvaluationError("smoke receipt links the wrong model receipt path")
    gate_state_path = phase13_artifact_path(str(receipt.get("gate_state_path", "")))
    if (
        not gate_state_path.is_file()
        or sha256_file(gate_state_path) != receipt.get("gate_state_sha256")
    ):
        raise FreshEvaluationError("smoke gate state hash mismatch")
    smoke_manifest_path = phase13_artifact_path(
        str(receipt.get("smoke_manifest_path", ""))
    )
    if (
        not smoke_manifest_path.is_file()
        or sha256_file(smoke_manifest_path) != receipt.get("smoke_manifest_sha256")
    ):
        raise FreshEvaluationError("smoke prerequisite manifest hash mismatch")
    generated = receipt.get("generated_config_sha256")
    if not isinstance(generated, dict) or len(generated) != 15:
        raise FreshEvaluationError("smoke receipt lacks 15 generated-config hashes")
    for path_text, expected_sha in generated.items():
        path = phase13_artifact_path(str(path_text))
        if not path.is_file() or sha256_file(path) != expected_sha:
            raise FreshEvaluationError(f"smoke config hash drift: {path_text}")
    phase_inputs = receipt.get("phase13_input_sha256")
    if not isinstance(phase_inputs, dict):
        raise FreshEvaluationError("smoke receipt lacks Phase 13 input hashes")
    for path_text, expected_sha in phase_inputs.items():
        if "/" not in str(path_text):
            continue
        if str(path_text).startswith("mitsuba/"):
            path = project_input_path(str(path_text))
        else:
            path = phase13_artifact_path(str(path_text))
        if not path.is_file() or sha256_file(path) != expected_sha:
            raise FreshEvaluationError(f"smoke Phase 13 input drift: {path_text}")
    expected_phase13_sha = receipt.get("phase13_input_sha256", {}).get(
        "scripts/phase_13_run_formal_benchmark.py"
    )
    if expected_phase13_sha != sha256_file(
        workspace_path("scripts/phase_13_run_formal_benchmark.py")
    ):
        raise FreshEvaluationError("smoke/final Phase 13 runner hash differs")
    if receipt.get("validated_job_count") != 15:
        raise FreshEvaluationError("smoke receipt must validate exactly 15 configs")
    if receipt.get("validated_record_count") != 30:
        raise FreshEvaluationError("smoke receipt must validate exactly 30 records")
    checks = receipt.get("jobs")
    if not isinstance(checks, list) or len(checks) != 15:
        raise FreshEvaluationError("smoke receipt must contain 15 per-config checks")
    for index, check in enumerate(checks):
        if not isinstance(check, dict):
            raise FreshEvaluationError(f"smoke check {index} is not an object")
        validation = check.get("status_validation")
        if not isinstance(validation, dict):
            raise FreshEvaluationError(f"smoke check {index} lacks status validation")
        if validation.get("quality_metrics_inspected") is not False:
            raise FreshEvaluationError(f"smoke check {index} crossed quality barrier")
        if validation.get("quality_metrics_computed") is not False:
            raise FreshEvaluationError(f"smoke check {index} computed quality metrics")
        records = validation.get("records")
        if not isinstance(records, list) or len(records) != 2:
            raise FreshEvaluationError(f"smoke check {index} must validate two records")
        bundle = str(check.get("bundle"))
        learned_method = bundle in {"ordered", "simple"}
        expected_methods = {
            "baseline": {"path_reference", "mpg"},
            "ordered": {"path_reference", "ordered_diffusion_mlp_h72_k4"},
            "simple": {"path_reference", "simple_mlp_k4"},
        }
        if bundle not in expected_methods or {
            str(item.get("method")) for item in records
        } != expected_methods[bundle]:
            raise FreshEvaluationError(f"smoke check {index} method set changed")
        loaded = [bool(record.get("loaded_runtime_model")) for record in records]
        if learned_method and sum(loaded) != 1:
            raise FreshEvaluationError(f"smoke check {index} missed runtime load")
        if not learned_method and any(loaded):
            raise FreshEvaluationError(f"smoke check {index} unexpectedly loaded runtime")
        for record in records:
            if (
                record.get("return_code") != 0
                or record.get("output_exists") is not True
                or record.get("integrator_matches") is not True
                or record.get("runtime_marker_matches") is not True
                or record.get("controls_match") is not True
                or not isinstance(record.get("output_sha256"), str)
                or len(record["output_sha256"]) != 64
                or not isinstance(record.get("patched_xml_sha256"), str)
                or len(record["patched_xml_sha256"]) != 64
            ):
                raise FreshEvaluationError(f"smoke check {index} has invalid provenance")
            method = str(record["method"])
            controls = record.get("expected_controls")
            if method == "path_reference":
                expected_controls: dict[str, Any] = {}
            elif method == "mpg":
                expected_controls = {"bounce_alpha": 0.5}
            elif method == "ordered_diffusion_mlp_h72_k4":
                expected_controls = {
                    "bounce_alpha": 0.0,
                    "learned_direction_kappa_scale": 1.0,
                    "learned_direction_flip_to_receiver_hemisphere": False,
                }
            elif method == "simple_mlp_k4":
                expected_controls = {
                    "bounce_alpha": 0.5,
                    "learned_direction_kappa_scale": 0.25,
                    "learned_direction_flip_to_receiver_hemisphere": False,
                }
            else:
                raise FreshEvaluationError(f"unknown smoke method: {method}")
            if not isinstance(controls, dict):
                raise FreshEvaluationError(f"smoke check {index} lacks controls")
            for key, expected_value in expected_controls.items():
                if controls.get(key) != expected_value:
                    raise FreshEvaluationError(
                        f"smoke check {index} control mismatch: {key}"
                    )
            if learned_method and method != "path_reference":
                runtime_path = controls.get("runtime_model_path")
                if not isinstance(runtime_path, str) or not runtime_path:
                    raise FreshEvaluationError(
                        f"smoke check {index} lacks expected runtime path"
                    )
        artifact_hashes = check.get("artifact_sha256")
        if not isinstance(artifact_hashes, dict) or not artifact_hashes:
            raise FreshEvaluationError(f"smoke check {index} lacks artifact hashes")
        for path_text, expected_sha in artifact_hashes.items():
            path = phase13_artifact_path(str(path_text))
            if not path.is_file() or sha256_file(path) != expected_sha:
                raise FreshEvaluationError(
                    f"smoke artifact hash drift in check {index}: {path_text}"
                )
    return receipt_path, receipt


def validate_runner_interfaces(manifest: dict[str, Any]) -> None:
    interfaces = manifest["runner_interfaces"]
    validate_hash_pinned_file(interfaces["orchestrator"], "evaluation orchestrator")
    phase13 = interfaces["phase13"]
    phase13_path = validate_hash_pinned_file(phase13, "Phase 13 runner")
    source = phase13_path.read_text(encoding="utf-8")
    for field in phase13.get("required_config_fields", []):
        if str(field) not in source:
            raise FreshEvaluationError(
                f"Phase 13 runner lacks required interface field {field!r}"
            )
    finalizer = interfaces["deferred_metric_finalizer"]
    validate_hash_pinned_file(finalizer, "deferred metric finalizer")
    validate_hash_pinned_file(
        interfaces["hierarchical_summarizer"], "hierarchical summarizer"
    )
    shared = interfaces["shared_reference"]
    if shared.get("status") != "ready":
        raise FreshEvaluationError("shared-reference interface is not frozen ready")
    if shared.get("expected_status") != "ready" or shared.get("expected_scene_count") != 5:
        raise FreshEvaluationError("shared-reference manifest contract changed")
    producer_path = validate_hash_pinned_file(
        dict(shared.get("producer_config", {})),
        "shared-reference producer config",
    )
    producer = load_json(producer_path)
    expected_producer = {
        "profile_id": "macos_arm64_native",
        "job_ids": [f"paper_{scene}_mpg" for scene in EXPECTED_SCENES],
        "width": 64,
        "height": 64,
        "crop_width": 64,
        "crop_height": 64,
        "reference_sample_count": 32768,
        "seed": 314159,
        "reference_seed": 314159,
        "methods": ["path_reference"],
        "shared_reference_manifest_output": shared["manifest_path"],
    }
    for key, expected in expected_producer.items():
        if producer.get(key) != expected:
            raise FreshEvaluationError(
                f"shared-reference producer {key}={producer.get(key)!r}, "
                f"expected {expected!r}"
            )


def smoke_pinned_input_paths(manifest: dict[str, Any]) -> dict[str, Path]:
    """Resolve and verify every input byte named by the smoke-gate receipt."""

    model_path = workspace_path(
        str(manifest["dependency_receipts"]["model_artifacts"]["path"])
    )
    smoke_path = workspace_path(
        str(manifest["dependency_receipts"]["status_only_smoke"]["path"])
    )
    model_receipt = load_json(model_path)
    smoke_receipt = load_json(smoke_path)
    expected_hashes = smoke_receipt.get("phase13_input_sha256")
    if not isinstance(expected_hashes, dict) or set(expected_hashes) != set(
        SMOKE_PINNED_INPUT_KEYS
    ):
        raise FreshEvaluationError("smoke-pinned Phase 13 input set changed")
    aliases = {
        "registrar": workspace_path(str(model_receipt.get("registrar_path", ""))),
        "registration_receipt": model_path,
        "training_manifest": workspace_path(
            str(manifest["frozen_training_manifest"]["path"])
        ),
        "training_state": TRAINING_STATE.resolve(),
        "smoke_manifest": phase13_artifact_path(
            str(smoke_receipt.get("smoke_manifest_path", ""))
        ),
        "smoke_generator": workspace_path(
            str(model_receipt.get("smoke_generator_path", ""))
        ),
    }
    resolved: dict[str, Path] = {}
    for key, expected_sha in expected_hashes.items():
        if key in aliases:
            path = aliases[key]
        elif str(key).startswith("mitsuba/"):
            path = project_input_path(str(key))
        else:
            path = phase13_artifact_path(str(key))
        if not path.is_file() or sha256_file(path) != expected_sha:
            raise FreshEvaluationError(f"smoke-pinned input drift: {key}")
        resolved[str(key)] = path
    return resolved


def validate_dependencies(manifest: dict[str, Any]) -> dict[str, str]:
    receipts = manifest["dependency_receipts"]
    model_path, _ = validate_model_receipt(manifest, receipts["model_artifacts"])
    smoke_path, _ = validate_smoke_receipt(
        model_receipt_path=model_path,
        receipt_spec=receipts["status_only_smoke"],
    )
    smoke_pinned_input_paths(manifest)
    validate_runner_interfaces(manifest)
    return {
        "model_artifact_receipt": sha256_file(model_path),
        "status_only_smoke_receipt": sha256_file(smoke_path),
    }


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
        lowered = command.lower()
        if any(marker in lowered for marker in CONFLICT_MARKERS):
            conflicts.append(stripped)
    return conflicts


def acquire_lock(path: Path, manifest_sha: str) -> Any:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        handle.seek(0)
        owner = handle.read().strip() or "unknown owner"
        handle.close()
        raise FreshEvaluationError(f"evaluation lock is held: {owner}") from exc
    handle.seek(0)
    handle.truncate()
    handle.write(
        f"pid={os.getpid()} acquired_at={now_utc()} manifest_sha256={manifest_sha}\n"
    )
    handle.flush()
    os.fsync(handle.fileno())
    return handle


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    temporary.replace(path)


def blank_state(
    *, manifest_sha: str, nodes: tuple[DagNode, ...], dependency_hashes: dict[str, str]
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "protocol_id": EXPECTED_PROTOCOL,
        "manifest_sha256": manifest_sha,
        "dependency_sha256": dependency_hashes,
        "created_at": now_utc(),
        "updated_at": now_utc(),
        "metric_barrier": "locked",
        "nodes": {
            node.node_id: {
                "status": "pending",
                "stage": node.stage,
                "dependencies": list(node.dependencies),
                "attempts": [],
                "output_sha256": {},
            }
            for node in nodes
        },
    }


def load_or_prepare_state(
    *,
    state_path: Path,
    manifest_sha: str,
    nodes: tuple[DagNode, ...],
    dependency_hashes: dict[str, str],
) -> dict[str, Any]:
    """Load a resumable ledger, rejecting protocol/hash/node-set drift."""

    if not state_path.is_file():
        return blank_state(
            manifest_sha=manifest_sha,
            nodes=nodes,
            dependency_hashes=dependency_hashes,
        )
    state = load_json(state_path)
    if state.get("schema_version") != 1 or state.get("protocol_id") != EXPECTED_PROTOCOL:
        raise FreshEvaluationError("resume state protocol mismatch")
    if state.get("manifest_sha256") != manifest_sha:
        raise FreshEvaluationError(
            "resume state belongs to a different evaluation manifest; preserve it "
            "and use a new output root"
        )
    if state.get("dependency_sha256") != dependency_hashes:
        raise FreshEvaluationError("hash-pinned dependency receipts changed since start")
    records = state.get("nodes")
    if not isinstance(records, dict):
        raise FreshEvaluationError("resume state nodes block is invalid")
    expected_ids = {node.node_id for node in nodes}
    if set(records) != expected_ids:
        raise FreshEvaluationError("resume state node set differs from expanded DAG")
    for node in nodes:
        record = records[node.node_id]
        if record.get("stage") != node.stage:
            raise FreshEvaluationError(f"resume state stage drift: {node.node_id}")
        if tuple(record.get("dependencies", [])) != node.dependencies:
            raise FreshEvaluationError(
                f"resume state dependency drift: {node.node_id}"
            )
    barrier_status = classify_node_record(records["barrier:unlock_candidate_metrics"])
    expected_barrier = "unlocked" if barrier_status == "succeeded" else "locked"
    if state.get("metric_barrier") != expected_barrier:
        raise FreshEvaluationError("resume state metric-barrier status is inconsistent")
    return state


def classify_node_record(record: dict[str, Any]) -> str:
    """Classify a node without trusting a prior success label alone."""

    status = str(record.get("status", "pending"))
    if status == "running":
        return "stale"
    if status == "skipped":
        return "skipped"
    if status != "succeeded":
        if status in {"pending", "failed"} and record.get("output_sha256"):
            return "stale"
        return status if status in {"pending", "failed", "stale"} else "stale"
    hashes = record.get("output_sha256")
    if not isinstance(hashes, dict) or not hashes:
        return "stale"
    for path_text, expected_sha in hashes.items():
        try:
            path = workspace_path(str(path_text))
        except FreshEvaluationError:
            return "stale"
        if not path.is_file() or sha256_file(path) != expected_sha:
            return "stale"
    return "succeeded"


def immutable_input_snapshot(
    *, manifest_path: Path, manifest: dict[str, Any]
) -> dict[str, str]:
    """Hash every immutable root needed for per-node TOCTOU checks."""

    asset_inventory_path, scene_assets = validate_scene_asset_inventory(manifest)
    paths = {
        "evaluation_manifest": manifest_path.resolve(),
        "evaluation_manifest_schema": workspace_path(
            str(manifest["manifest_schema"]["path"])
        ),
        "scene_asset_inventory": asset_inventory_path,
        "evaluation_orchestrator": workspace_path(
            str(manifest["runner_interfaces"]["orchestrator"]["path"])
        ),
        "plan": workspace_path(str(manifest["plan"]["path"])),
        "training_manifest": workspace_path(
            str(manifest["frozen_training_manifest"]["path"])
        ),
        "model_artifact_receipt": workspace_path(
            str(manifest["dependency_receipts"]["model_artifacts"]["path"])
        ),
        "status_only_smoke_receipt": workspace_path(
            str(manifest["dependency_receipts"]["status_only_smoke"]["path"])
        ),
        "phase13_runner": workspace_path(
            str(manifest["runner_interfaces"]["phase13"]["path"])
        ),
        "deferred_metric_finalizer": workspace_path(
            str(manifest["runner_interfaces"]["deferred_metric_finalizer"]["path"])
        ),
        "hierarchical_summarizer": workspace_path(
            str(manifest["runner_interfaces"]["hierarchical_summarizer"]["path"])
        ),
        "shared_reference_producer_config": workspace_path(
            str(
                manifest["runner_interfaces"]["shared_reference"][
                    "producer_config"
                ]["path"]
            )
        ),
        "training_state": TRAINING_STATE.resolve(),
    }
    for scene in manifest["study"]["scenes"]:
        paths[f"scene:{scene['key']}"] = project_input_path(str(scene["scene_path"]))
    for path_text, path in scene_assets.items():
        paths[f"scene_asset:{path_text}"] = path
    for label, spec in manifest["renderer_inputs"].items():
        resolver = workspace_path if spec["path_base"] == "workspace_root" else project_input_path
        paths[f"renderer:{label}"] = resolver(str(spec["path"]))
    for label, path in smoke_pinned_input_paths(manifest).items():
        paths[f"smoke_input:{label}"] = path
    training_manifest = load_json(
        workspace_path(str(manifest["frozen_training_manifest"]["path"]))
    )
    for job in training_manifest["jobs"]:
        if not str(job.get("kind", "")).startswith("phase12_"):
            continue
        paths[f"runtime:{job['job_id']}"] = workspace_path(
            str(job["expected_outputs"][0])
        )
    snapshot: dict[str, str] = {}
    for label, path in paths.items():
        if not path.is_file():
            raise FreshEvaluationError(f"missing immutable input {label}: {path}")
        snapshot[label] = sha256_file(path)
    return snapshot


def assert_snapshot_unchanged(
    *, manifest_path: Path, manifest: dict[str, Any], expected: dict[str, str]
) -> None:
    current = immutable_input_snapshot(manifest_path=manifest_path, manifest=manifest)
    if current != expected:
        changed = sorted(
            key
            for key in set(current) | set(expected)
            if current.get(key) != expected.get(key)
        )
        raise FreshEvaluationError(
            "immutable inputs changed during execution: " + ", ".join(changed)
        )


def validate_metric_unlock_eligibility(
    *,
    state: dict[str, Any],
    nodes: tuple[DagNode, ...],
    manifest: dict[str, Any],
) -> dict[str, Any]:
    """Validate only null-quality/status provenance before opening the barrier."""

    if state.get("metric_barrier") != "locked":
        raise FreshEvaluationError("metric barrier is not in its unique locked state")
    verify_global_preunlock_quality_barrier(
        manifest=manifest,
        state=state,
        nodes=nodes,
    )
    records = state.get("nodes", {})
    candidate_nodes = [
        node for node in nodes if node.stage in {"equal_spp", "equal_time"}
    ]
    if len(candidate_nodes) != 350:
        raise FreshEvaluationError("unlock expected exactly 350 candidate nodes")
    candidate_hashes: dict[str, dict[str, str]] = {}
    for node in candidate_nodes:
        record = records.get(node.node_id)
        if not isinstance(record, dict) or classify_node_record(record) != "succeeded":
            raise FreshEvaluationError(
                f"candidate is not a hash-verified success: {node.node_id}"
            )
        if record.get("metrics_deferred") is not True:
            raise FreshEvaluationError(f"candidate was not metric-deferred: {node.node_id}")
        if record.get("mse") is not None or record.get("psnr") is not None:
            raise FreshEvaluationError(
                f"candidate quality was populated before unlock: {node.node_id}"
            )
        if record.get("nonfinite_pixel_boolean") is not False:
            raise FreshEvaluationError(f"candidate is nonfinite/unvalidated: {node.node_id}")
        verify_preunlock_run_tree(record=record, node_id=node.node_id)
        candidate_hashes[node.node_id] = dict(record["output_sha256"])
    reference = records.get("reference:shared_block")
    if not isinstance(reference, dict) or classify_node_record(reference) != "succeeded":
        raise FreshEvaluationError("shared reference block is not hash-verified")
    reference_manifest_sha = reference.get("shared_reference_manifest_sha256")
    if not isinstance(reference_manifest_sha, str) or len(reference_manifest_sha) != 64:
        raise FreshEvaluationError("shared reference manifest is not hash-pinned")
    for node in candidate_nodes:
        if records[node.node_id].get("reference_manifest_sha256") != reference_manifest_sha:
            raise FreshEvaluationError(
                f"candidate used a different reference manifest: {node.node_id}"
            )
    reference_noise_hashes: dict[str, dict[str, str]] = {}
    for node in nodes:
        if node.stage != "reference_noise_check":
            continue
        record = records.get(node.node_id)
        if not isinstance(record, dict) or classify_node_record(record) != "succeeded":
            raise FreshEvaluationError(
                f"reference-noise check is not hash-verified: {node.node_id}"
            )
        if record.get("metrics_deferred") is not True:
            raise FreshEvaluationError(
                f"reference-noise metric was not deferred: {node.node_id}"
            )
        if record.get("mse") is not None or record.get("psnr") is not None:
            raise FreshEvaluationError(
                f"reference-noise quality was populated before unlock: {node.node_id}"
            )
        if record.get("nonfinite_pixel_boolean") is not False:
            raise FreshEvaluationError(
                f"reference-noise output is nonfinite/unvalidated: {node.node_id}"
            )
        verify_preunlock_run_tree(record=record, node_id=node.node_id)
        reference_noise_hashes[node.node_id] = dict(record["output_sha256"])
        if record.get("reference_manifest_sha256") != reference_manifest_sha:
            raise FreshEvaluationError(
                f"reference-noise check used a different manifest: {node.node_id}"
            )
    frozen_map = records.get("calibration:freeze_spp_map")
    if not isinstance(frozen_map, dict) or classify_node_record(frozen_map) != "succeeded":
        raise FreshEvaluationError("equal-time SPP map is not hash-verified")
    spp_map_sha = frozen_map.get("spp_map_sha256")
    if not isinstance(spp_map_sha, str) or len(spp_map_sha) != 64:
        raise FreshEvaluationError("equal-time SPP map is not hash-pinned")
    for node in candidate_nodes:
        if node.stage == "equal_time" and records[node.node_id].get(
            "spp_map_sha256"
        ) != spp_map_sha:
            raise FreshEvaluationError(
                f"equal-time candidate used a different SPP map: {node.node_id}"
            )
    return {
        "protocol_id": EXPECTED_PROTOCOL,
        "candidate_count": len(candidate_nodes),
        "candidate_output_sha256": candidate_hashes,
        "reference_noise_output_sha256": reference_noise_hashes,
        "shared_reference_manifest_sha256": reference_manifest_sha,
        "equal_time_spp_map_sha256": spp_map_sha,
        "quality_fields_verified_null": True,
    }


def verify_preunlock_run_tree(*, record: dict[str, Any], node_id: str) -> None:
    summary_path = workspace_path(str(record.get("summary_path", "")))
    run_dir = summary_path.parent
    current = tree_hashes(run_dir)
    registered = record.get("output_sha256", {})
    expected = {
        path_text: digest
        for path_text, digest in registered.items()
        if workspace_path(str(path_text)).parent == run_dir
        or run_dir in workspace_path(str(path_text)).parents
    }
    if current != expected:
        raise FreshEvaluationError(
            f"unregistered/changed artifact exists before metric unlock: {node_id}"
        )
    if any(run_dir.glob("benchmark_summary_with_metrics.*")):
        raise FreshEvaluationError(
            f"candidate metrics were finalized before global unlock: {node_id}"
        )


def verify_global_preunlock_quality_barrier(
    *,
    manifest: dict[str, Any],
    state: dict[str, Any],
    nodes: tuple[DagNode, ...],
) -> None:
    """Reject metrics globally while ignoring ordinary orphaned failed attempts."""

    root = workspace_path(str(manifest["outputs"]["candidate_quarantine"]))
    if not root.is_dir():
        return
    finalized = sorted(root.rglob("benchmark_summary_with_metrics.*"))
    if finalized:
        raise FreshEvaluationError(
            "metric-bearing artifacts exist before global unlock: "
            + ", ".join(portable_workspace_path(path) for path in finalized[:10])
        )
    render_stages = {
        "reference_noise_check",
        "equal_spp",
        "timing_calibration",
        "timing_calibration_repeat",
        "equal_time",
    }
    for node in nodes:
        if node.stage not in render_stages:
            continue
        record = state["nodes"][node.node_id]
        classification = classify_node_record(record)
        if classification == "skipped" and node.conditional:
            continue
        if classification != "succeeded":
            raise FreshEvaluationError(
                f"pre-unlock render is incomplete or stale: {node.node_id}"
            )
        summary_path = workspace_path(str(record.get("summary_path", "")))
        require_within(summary_path, root, "registered pre-unlock summary")
        verify_preunlock_run_tree(record=record, node_id=node.node_id)
        summary = load_json(summary_path)
        if summary.get("metrics_status") != "deferred":
            raise FreshEvaluationError(
                "pre-unlock render summary is not metric-deferred: "
                + portable_workspace_path(summary_path)
            )
        rows = summary.get("records")
        if (
            not isinstance(rows, list)
            or any(not isinstance(row, dict) for row in rows)
            or any(
                row.get(key) is not None
                for row in rows
                for key in ("mse", "psnr", "max_value")
            )
        ):
            raise FreshEvaluationError(
                "pre-unlock render summary contains quality fields: "
                + portable_workspace_path(summary_path)
            )


def opencv_build_has_openexr(build_line: str) -> bool:
    """Interpret OpenCV's dedicated OpenEXR build-information field."""

    match = re.fullmatch(r"\s*OpenEXR\s*:\s*(.*?)\s*", build_line)
    if match is None:
        return False
    value = match.group(1).strip()
    if not value:
        return False
    disabled = {"NO", "OFF", "FALSE", "DISABLED", "NONE", "N/A"}
    return value.upper() not in disabled


def probe_python(python_executable: str) -> dict[str, str]:
    resolved = shutil.which(python_executable)
    if resolved is None:
        raise FreshEvaluationError(f"Python executable is not resolvable: {python_executable}")
    probe = (
        "import os; os.environ['OPENCV_IO_ENABLE_OPENEXR']='1'; "
        "import cv2,json,numpy,platform,sys; "
        "source=numpy.array([[[0.25,0.5,0.75]]],dtype=numpy.float32); "
        "encoded_ok,encoded=cv2.imencode('.exr',source); "
        "decoded=cv2.imdecode(encoded,cv2.IMREAD_UNCHANGED) if encoded_ok else None; "
        "roundtrip=bool(encoded_ok and decoded is not None and "
        "decoded.shape==(1,1,3) and numpy.isfinite(decoded).all()); "
        "build_lines=[line.strip() for line in cv2.getBuildInformation().splitlines() "
        "if line.strip().startswith('OpenEXR:')]; "
        "print(json.dumps({'executable':sys.executable,"
        "'python':platform.python_version(),'numpy':numpy.__version__,"
        "'cv2':cv2.__version__,'opencv_openexr_roundtrip':roundtrip,"
        "'opencv_openexr_build':build_lines[0] if len(build_lines)==1 else ''},"
        "sort_keys=True))"
    )
    result = subprocess.run(
        [resolved, "-c", probe],
        cwd=WORKSPACE_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise FreshEvaluationError(
            f"selected Python cannot import NumPy/OpenCV: {result.stderr.strip()}"
        )
    try:
        environment = json.loads(result.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError) as exc:
        raise FreshEvaluationError("invalid Python environment probe") from exc
    environment["resolved_executable"] = os.path.abspath(resolved)
    if not opencv_build_has_openexr(str(environment.get("opencv_openexr_build", ""))):
        raise FreshEvaluationError(
            "selected Python OpenCV reports OpenEXR disabled or unavailable"
        )
    if environment.get("opencv_openexr_roundtrip") is not True:
        raise FreshEvaluationError(
            "selected Python OpenCV failed an in-memory OpenEXR encode/decode probe"
        )
    return {str(key): str(value) for key, value in environment.items()}


def portable_workspace_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(WORKSPACE_ROOT.resolve()).as_posix()
    except ValueError as exc:
        raise FreshEvaluationError(f"artifact escapes workspace: {resolved}") from exc


def phase13_artifact_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        resolved = path.resolve()
    elif path.parts and path.parts[0] == WORKSPACE_ROOT.name:
        resolved = (PROJECT_ROOT / path).resolve()
    else:
        resolved = (WORKSPACE_ROOT / path).resolve()
    try:
        resolved.relative_to(PROJECT_ROOT.resolve())
    except ValueError as exc:
        raise FreshEvaluationError(f"Phase 13 artifact escapes project: {value}") from exc
    return resolved


def runtime_registry(manifest: dict[str, Any]) -> dict[tuple[str, str, int], dict[str, str]]:
    """Resolve all 30 runtimes only through the receipt-pinned training state."""

    training_manifest = load_json(
        workspace_path(str(manifest["frozen_training_manifest"]["path"]))
    )
    training_state = load_json(TRAINING_STATE)
    state_jobs = training_state.get("jobs", {})
    registry: dict[tuple[str, str, int], dict[str, str]] = {}
    kinds = {
        "phase12_ordered_export": "ordered",
        "phase12_simple_export": "simple",
    }
    for job in training_manifest["jobs"]:
        method = kinds.get(str(job.get("kind")))
        if method is None:
            continue
        job_id = str(job["job_id"])
        record = state_jobs.get(job_id)
        if not isinstance(record, dict) or record.get("status") != "succeeded":
            raise FreshEvaluationError(f"runtime export is not successful: {job_id}")
        runtime_text = str(job["expected_outputs"][0])
        runtime_path = workspace_path(runtime_text)
        expected_sha = record.get("output_sha256", {}).get(runtime_text)
        if not runtime_path.is_file() or sha256_file(runtime_path) != expected_sha:
            raise FreshEvaluationError(f"runtime hash drift: {job_id}")
        key = (str(job["scene"]), method, int(job["seed"]))
        if key in registry:
            raise FreshEvaluationError(f"duplicate runtime registration: {key}")
        registry[key] = {"path": runtime_text, "sha256": str(expected_sha)}
    expected_keys = {
        (scene, method, seed)
        for scene in EXPECTED_SCENES
        for method in ("ordered", "simple")
        for seed in EXPECTED_TRAIN_SEEDS
    }
    if set(registry) != expected_keys:
        raise FreshEvaluationError("receipt-pinned runtime registry is incomplete")
    return registry


def training_accounting(manifest: dict[str, Any]) -> dict[str, Any]:
    """Collect preregistered cost/size provenance without reading quality curves."""

    training_manifest = load_json(
        workspace_path(str(manifest["frozen_training_manifest"]["path"]))
    )
    training_state = load_json(TRAINING_STATE)
    split_by_scene_seed = {
        (str(scene["scene"]), int(seed_entry["seed"])): seed_entry
        for scene in training_manifest["scenes"]
        for seed_entry in scene["seeds"]
    }
    rows: list[dict[str, Any]] = []
    for job in training_manifest["jobs"]:
        record = training_state["jobs"][job["job_id"]]
        attempts = [
            attempt
            for attempt in record.get("attempts", [])
            if attempt.get("return_code") == 0
        ]
        if record.get("status") != "succeeded" or not attempts:
            raise FreshEvaluationError(f"training accounting job incomplete: {job['job_id']}")
        provenance = record.get("output_provenance", {})
        is_training = "training" in str(job["kind"])
        config = load_json(workspace_path(str(job["config_path"])))
        epochs: int | None = None
        train_records: int | None = None
        examples_seen: int | None = None
        optimizer_steps: int | None = None
        if is_training:
            epochs_value = config.get("epochs")
            if epochs_value is None:
                epochs_value = config.get("simple_baseline", {}).get("epochs")
            epochs = int(epochs_value)
            split = split_by_scene_seed[(str(job["scene"]), int(job["seed"]))]
            train_records = int(split["split"]["train_records"])
            examples_seen = train_records * epochs
            optimizer_steps = int(split["optimizer_steps_per_model"])
        duration = float(attempts[-1]["duration_seconds"])
        rows.append(
            {
                "job_id": job["job_id"],
                "kind": job["kind"],
                "scene": job["scene"],
                "seed": job["seed"],
                "duration_seconds": duration,
                "parameter_count": provenance.get("checkpoint_parameter_count"),
                "epochs": epochs,
                "train_records_per_epoch": train_records,
                "examples_seen": examples_seen,
                "optimizer_steps": optimizer_steps,
                "peak_memory_bytes": None,
                "peak_memory_status": "unavailable_not_recorded_by_frozen_runner",
                "amortized_confirmatory_images": 10 if is_training else None,
                "duration_seconds_per_amortized_image": (
                    duration / 10.0 if is_training else None
                ),
                "output_sha256": record["output_sha256"],
            }
        )
    training_rows = [row for row in rows if "training" in row["kind"]]
    export_rows = [row for row in rows if "export" in row["kind"]]
    if len(training_rows) != 30 or len(export_rows) != 30:
        raise FreshEvaluationError("training accounting expected 30 train + 30 export jobs")
    return {
        "jobs": rows,
        "training_job_count": len(training_rows),
        "export_job_count": len(export_rows),
        "training_seconds": sum(row["duration_seconds"] for row in training_rows),
        "export_seconds": sum(row["duration_seconds"] for row in export_rows),
        "parameter_counts": {
            method: manifest["study"]["methods"][method]["expected_parameters"]
            for method in ("ordered", "simple")
        },
        "peak_memory_status": "unavailable_not_recorded_by_frozen_runner",
        "amortization_policy": (
            "each learned scene/method/train-seed model is used for five render "
            "seeds in each of two regimes (10 confirmatory images)"
        ),
    }


def _common_phase13_config(manifest: dict[str, Any]) -> dict[str, Any]:
    study = manifest["study"]
    resolution = study["resolution"]
    runtime = study["common_runtime"]
    return {
        "profile_id": study["profile_id"],
        "width": resolution["width"],
        "height": resolution["height"],
        "crop_width": resolution["crop_width"],
        "crop_height": resolution["crop_height"],
        "reference_sample_count": manifest["stages"]["shared_references"][
            "sample_count"
        ],
        "samples_per_pass": runtime["samples_per_pass"],
        "threads": runtime["threads"],
        "reference_seed": study["reference_seed"],
        "mpg_train_budget": 0.2,
        "masked_diffusion_train_budget": 0.2,
        "mpg_train_fusion": True,
        "learned_proposal_inv_prob_clamp": runtime[
            "learned_proposal_inv_prob_clamp"
        ],
        "learned_proposal_contribution_clamp": runtime[
            "learned_proposal_contribution_clamp"
        ],
        "learned_proposal_max_trials": runtime["learned_proposal_max_trials"],
        "diffusion_steps": runtime["diffusion_steps"],
        "time_budget_seconds": runtime["time_budget_seconds"],
        "postprocess_crop_from_full_exr": False,
    }


def _scene_job_id(manifest: dict[str, Any], scene: str) -> str:
    return str(
        next(item["job_id"] for item in manifest["study"]["scenes"] if item["key"] == scene)
    )


def generated_config_for_node(
    *,
    node: DagNode,
    manifest: dict[str, Any],
    runtimes: dict[tuple[str, str, int], dict[str, str]],
    spp_map: dict[str, int] | None,
) -> dict[str, Any]:
    if node.stage not in {
        "reference_noise_check",
        "equal_spp",
        "equal_time",
        "timing_calibration",
        "timing_calibration_repeat",
    }:
        raise FreshEvaluationError(f"node has no generated Phase 13 config: {node.node_id}")
    if node.scene is None or node.method is None:
        raise FreshEvaluationError(f"render node lacks scene/method: {node.node_id}")
    config = _common_phase13_config(manifest)
    method_spec: dict[str, Any]
    if node.stage == "reference_noise_check":
        phase13_method = "pt"
        sample_count = int(node.sample_count or 0)
        render_seed = int(manifest["study"]["reference_noise_seed"])
        config["time_budget_seconds"] = float(
            manifest["stages"]["reference_noise_check"]["watchdog_seconds"]
        )
        method_spec = {
            "bounce_alpha": 0.5,
            "learned_direction_kappa_scale": 1.0,
            "learned_direction_flip_to_receiver_hemisphere": False,
        }
    else:
        method_spec = manifest["study"]["methods"][node.method]
        phase13_method = str(method_spec["phase13_method"])
        if node.stage == "equal_time":
            if spp_map is None:
                raise FreshEvaluationError("equal-time node has no frozen SPP map")
            key = f"{node.scene}|{node.method}"
            if key not in spp_map:
                raise FreshEvaluationError(f"SPP map is missing {key}")
            sample_count = int(spp_map[key])
        else:
            sample_count = int(node.sample_count or 0)
        if node.render_seed is None:
            render_seed = int(manifest["study"]["render_seeds"][0])
        else:
            render_seed = int(node.render_seed)
    if sample_count <= 0:
        raise FreshEvaluationError(f"invalid SPP for {node.node_id}: {sample_count}")
    output_root = workspace_path(str(manifest["outputs"]["candidate_quarantine"]))
    output_dir = output_root / node.stage / node.node_id.replace(":", "__")
    config.update(
        {
            "job_ids": [_scene_job_id(manifest, node.scene)],
            "compare_sample_count": sample_count,
            "seed": render_seed,
            "output_dir": portable_workspace_path(output_dir),
            "methods": [phase13_method],
            "external_reference_manifest": manifest["runner_interfaces"][
                "shared_reference"
            ]["manifest_path"],
            "defer_metrics": True,
            "bounce_alpha": method_spec["bounce_alpha"],
            "learned_direction_kappa_scale": method_spec[
                "learned_direction_kappa_scale"
            ],
            "learned_direction_flip_to_receiver_hemisphere": method_spec[
                "learned_direction_flip_to_receiver_hemisphere"
            ],
        }
    )
    if node.method in {"ordered", "simple"}:
        if node.train_seed is None:
            raise FreshEvaluationError(f"learned node lacks train seed: {node.node_id}")
        runtime = runtimes[(node.scene, node.method, int(node.train_seed))]
        config["runtime_model_paths"] = {phase13_method: runtime["path"]}
        config["runtime_only_methods"] = [phase13_method]
    else:
        config["runtime_only_methods"] = []
    return config


def write_frozen_config(path: Path, payload: dict[str, Any]) -> str:
    serialized = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != serialized:
            raise FreshEvaluationError(f"refusing to replace differing config: {path}")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    return sha256_file(path)


def summary_snapshot(output_root: Path) -> dict[Path, str]:
    if not output_root.is_dir():
        return {}
    return {
        path.resolve(): sha256_file(path)
        for path in sorted(output_root.glob("run_*/benchmark_summary.json"))
        if path.is_file()
    }


def find_new_summary(output_root: Path, before: dict[Path, str]) -> Path:
    after = summary_snapshot(output_root)
    changed = [path for path, digest in after.items() if before.get(path) != digest]
    if len(changed) != 1:
        raise FreshEvaluationError(
            f"expected exactly one new Phase 13 summary in {output_root}, "
            f"found {len(changed)}"
        )
    return changed[0]


def phase13_failure_status(summary_path: Path) -> dict[str, Any]:
    """Extract only operational failure/timeout fields from a failed run summary."""

    payload = load_json(summary_path)
    rows = payload.get("records")
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise FreshEvaluationError("failed Phase 13 summary has invalid records")
    return_codes: list[int] = []
    for row in rows:
        try:
            return_codes.append(int(row.get("return_code")))
        except (TypeError, ValueError) as exc:
            raise FreshEvaluationError(
                "failed Phase 13 summary has invalid record return code"
            ) from exc
    return {
        "phase13_render_record_count": len(return_codes),
        "phase13_record_return_codes": return_codes,
        "phase13_failed_record_count": sum(code != 0 for code in return_codes),
        "phase13_timeout_record_count": sum(code in {124, -9} for code in return_codes),
    }


def tree_hashes(root: Path) -> dict[str, str]:
    if not root.is_dir():
        return {}
    return {
        portable_workspace_path(path): sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and not path.name.startswith(".")
    }


def require_within(path: Path, parent: Path, label: str) -> None:
    try:
        path.resolve().relative_to(parent.resolve())
    except ValueError as exc:
        raise FreshEvaluationError(f"{label} escapes run directory: {path}") from exc


def _xml_property(integrator: et.Element, name: str) -> str | None:
    for child in integrator:
        if child.attrib.get("name") == name:
            return child.attrib.get("value")
    return None


def _validate_learned_xml(
    *, xml_path: Path, node: DagNode, manifest: dict[str, Any]
) -> None:
    root = et.parse(xml_path).getroot()
    integrator = root.find("integrator")
    if integrator is None or integrator.attrib.get("type") != "manifold_path_guiding":
        raise FreshEvaluationError(f"wrong learned integrator in {xml_path}")
    spec = manifest["study"]["methods"][str(node.method)]
    expected = {
        "bounce_alpha": float(spec["bounce_alpha"]),
        "learned_direction_kappa_scale": float(
            spec["learned_direction_kappa_scale"]
        ),
        "learned_proposal_inv_prob_clamp": 512.0,
        "learned_proposal_contribution_clamp": 32.0,
    }
    for name, expected_value in expected.items():
        observed = _xml_property(integrator, name)
        try:
            matches = observed is not None and math.isclose(
                float(observed), expected_value, rel_tol=0.0, abs_tol=1e-12
            )
        except ValueError:
            matches = False
        if not matches:
            raise FreshEvaluationError(
                f"{node.node_id} XML {name}={observed!r}, expected {expected_value}"
            )
    expected_ints = {
        # The Phase 13 config uses the descriptive key
        # `learned_proposal_max_trials`, while the renderer plugin's XML
        # property is the legacy `max_trials` name.
        "max_trials": 2048,
        "diffusion_steps": 10,
    }
    for name, expected_value in expected_ints.items():
        observed = _xml_property(integrator, name)
        if observed is None or int(observed) != expected_value:
            raise FreshEvaluationError(
                f"{node.node_id} XML {name}={observed!r}, expected {expected_value}"
            )
    flip = _xml_property(integrator, "learned_direction_flip_to_receiver_hemisphere")
    if flip is None or flip.lower() != "false":
        raise FreshEvaluationError(f"{node.node_id} XML hemisphere flip is not false")


def validate_shared_reference_manifest(
    *, path: Path, manifest: dict[str, Any]
) -> dict[str, Any]:
    payload = load_json(path)
    if payload.get("schema_version") != 1 or payload.get("status") != "ready":
        raise FreshEvaluationError("shared-reference manifest is not ready schema 1")
    if payload.get("path_base") != "project_root":
        raise FreshEvaluationError("shared-reference path base changed")
    references = payload.get("references")
    if not isinstance(references, dict):
        raise FreshEvaluationError("shared-reference entries are missing")
    expected_ids = {
        _scene_job_id(manifest, scene).removesuffix("_mpg")
        for scene in EXPECTED_SCENES
    }
    if set(references) != expected_ids:
        raise FreshEvaluationError("shared-reference scene set changed")
    canonical_reference_id = hashlib.sha256(
        json.dumps(references, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if payload.get("reference_set_id") != canonical_reference_id:
        raise FreshEvaluationError("shared-reference set ID mismatch")
    source_path = phase13_artifact_path(str(payload.get("source_summary", "")))
    if not source_path.is_file() or sha256_file(source_path) != payload.get(
        "source_summary_sha256"
    ):
        raise FreshEvaluationError("shared-reference source summary hash mismatch")
    for scene_id, reference in references.items():
        expected = {
            "scene_id": scene_id,
            "sample_count": 32768,
            "sampler_seed": 314159,
            "width": 64,
            "height": 64,
            "crop_width": 64,
            "crop_height": 64,
        }
        for key, value in expected.items():
            if reference.get(key) != value:
                raise FreshEvaluationError(
                    f"shared reference {scene_id} {key} changed"
                )
        exr = phase13_artifact_path(str(reference.get("exr_path", "")))
        if (
            not exr.is_file()
            or exr.stat().st_size != int(reference.get("output_size", -1))
            or sha256_file(exr) != reference.get("exr_sha256")
        ):
            raise FreshEvaluationError(f"shared reference EXR drift: {scene_id}")
    return payload


def validate_phase13_summary(
    *,
    summary_path: Path,
    node: DagNode,
    config: dict[str, Any],
    manifest: dict[str, Any],
    reference_manifest_sha: str | None,
) -> dict[str, Any]:
    summary = load_json(summary_path)
    saved_config = summary.get("config")
    if not isinstance(saved_config, dict):
        raise FreshEvaluationError(f"summary has no config: {node.node_id}")
    for key, expected in config.items():
        if saved_config.get(key) != expected:
            raise FreshEvaluationError(f"summary config drift for {node.node_id}: {key}")
    expected_total = node.render_count
    if summary.get("total") != expected_total or summary.get("passed") != expected_total:
        raise FreshEvaluationError(f"Phase 13 did not fully pass: {node.node_id}")
    if summary.get("failed") != 0:
        raise FreshEvaluationError(f"Phase 13 recorded failures: {node.node_id}")
    records = summary.get("records")
    if not isinstance(records, list) or len(records) != expected_total:
        raise FreshEvaluationError(f"wrong record count: {node.node_id}")
    if node.stage == "shared_references":
        if summary.get("metrics_status") != "computed":
            raise FreshEvaluationError("reference producer metrics status changed")
        expected_methods = {"path_reference"}
    else:
        if summary.get("metrics_status") != "deferred":
            raise FreshEvaluationError(f"metrics were not deferred: {node.node_id}")
        expected_methods = {str(config["methods"][0])}
        provenance = summary.get("reference_provenance", {})
        if provenance.get("manifest_sha256") != reference_manifest_sha:
            raise FreshEvaluationError(f"reference provenance drift: {node.node_id}")
    expected_scene_ids = {
        str(job_id).removesuffix("_mpg") for job_id in config["job_ids"]
    }
    if {str(record.get("scene_id")) for record in records} != expected_scene_ids:
        raise FreshEvaluationError(f"scene identity mismatch: {node.node_id}")
    for record in records:
        method = str(record.get("method"))
        if method not in expected_methods or record.get("active_method") != method:
            raise FreshEvaluationError(f"active method mismatch: {node.node_id}")
        if record.get("ok") is not True or record.get("return_code") != 0:
            raise FreshEvaluationError(f"render failed: {node.node_id}")
        expected_spp = (
            int(config["reference_sample_count"])
            if node.stage == "shared_references"
            else int(config["compare_sample_count"])
        )
        expected_seed = (
            int(config["reference_seed"])
            if node.stage == "shared_references"
            else int(config["seed"])
        )
        if int(record.get("sample_count", -1)) != expected_spp:
            raise FreshEvaluationError(f"SPP mismatch: {node.node_id}")
        if int(record.get("sampler_seed", -1)) != expected_seed:
            raise FreshEvaluationError(f"sampler seed mismatch: {node.node_id}")
        if record.get("output_exists") is not True or int(record.get("output_size", 0)) <= 0:
            raise FreshEvaluationError(f"render output missing: {node.node_id}")
        output_path = phase13_artifact_path(str(record["output_path"]))
        output_sha = record.get("output_sha256")
        if (
            not output_path.is_file()
            or output_path.stat().st_size != int(record.get("output_size", -1))
            or sha256_file(output_path) != output_sha
        ):
            raise FreshEvaluationError(f"render output hash drift: {node.node_id}")
        xml_path = phase13_artifact_path(str(record["patched_scene_record"]))
        stdout_path = phase13_artifact_path(str(record["stdout_log"]))
        stderr_path = phase13_artifact_path(str(record["stderr_log"]))
        for artifact_path, label in (
            (output_path, "output"),
            (xml_path, "patched XML"),
            (stdout_path, "stdout log"),
            (stderr_path, "stderr log"),
        ):
            require_within(artifact_path, summary_path.parent, label)
            if not artifact_path.is_file():
                raise FreshEvaluationError(
                    f"missing {label} for {node.node_id}: {artifact_path}"
                )
        if not xml_path.is_file() or sha256_file(xml_path) != record.get(
            "active_config_sha256"
        ):
            raise FreshEvaluationError(f"patched XML hash drift: {node.node_id}")
        if node.stage != "shared_references":
            if (
                record.get("mse") is not None
                or record.get("psnr") is not None
                or record.get("max_value") is not None
            ):
                raise FreshEvaluationError(f"quality leaked before unlock: {node.node_id}")
            if record.get("nonfinite_pixel_boolean") is not False:
                raise FreshEvaluationError(
                    f"candidate finite/shape validation failed: {node.node_id}"
                )
        learned = node.method in {"ordered", "simple"}
        if learned:
            if record.get("loaded_runtime_model") is not True:
                raise FreshEvaluationError(f"runtime was not loaded: {node.node_id}")
            expected_runtime_text = config["runtime_model_paths"][method]
            expected_runtime = workspace_path(str(expected_runtime_text))
            observed_runtime = phase13_artifact_path(
                str(record.get("runtime_model_path", ""))
            )
            if observed_runtime != expected_runtime:
                raise FreshEvaluationError(f"runtime path mismatch: {node.node_id}")
            logs = stdout_path.read_text(
                encoding="utf-8", errors="replace"
            ) + stderr_path.read_text(encoding="utf-8", errors="replace")
            if RUNTIME_MARKER not in logs or str(expected_runtime) not in logs:
                raise FreshEvaluationError(
                    f"runtime load marker/path missing: {node.node_id}"
                )
            _validate_learned_xml(xml_path=xml_path, node=node, manifest=manifest)
        else:
            if record.get("loaded_runtime_model") is not False:
                raise FreshEvaluationError(f"unexpected runtime load: {node.node_id}")
            logs = stdout_path.read_text(
                encoding="utf-8", errors="replace"
            ) + stderr_path.read_text(encoding="utf-8", errors="replace")
            if RUNTIME_MARKER in logs:
                raise FreshEvaluationError(
                    f"unexpected runtime marker for {node.node_id}"
                )
    return summary


def node_input_hashes(
    *,
    config_path: Path,
    config: dict[str, Any],
    reference_manifest_path: Path | None,
) -> dict[str, str]:
    hashes = {portable_workspace_path(config_path): sha256_file(config_path)}
    if reference_manifest_path is not None:
        hashes[portable_workspace_path(reference_manifest_path)] = sha256_file(
            reference_manifest_path
        )
    runtime_paths = config.get("runtime_model_paths", {})
    for path_text in runtime_paths.values():
        path = workspace_path(str(path_text))
        hashes[portable_workspace_path(path)] = sha256_file(path)
    return hashes


def recover_published_shared_reference(
    *,
    node: DagNode,
    config_path: Path,
    config: dict[str, Any],
    record: dict[str, Any],
    state: dict[str, Any],
    state_path: Path,
    manifest: dict[str, Any],
    manifest_path: Path,
    immutable_snapshot: dict[str, str],
) -> bool:
    """Adopt a fully published reference block after an outer-runner crash."""

    if node.stage != "shared_references" or not record.get("attempts"):
        return False
    shared_path = workspace_path(str(config["shared_reference_manifest_output"]))
    if not shared_path.is_file():
        return False
    try:
        assert_snapshot_unchanged(
            manifest_path=manifest_path,
            manifest=manifest,
            expected=immutable_snapshot,
        )
        shared = validate_shared_reference_manifest(path=shared_path, manifest=manifest)
        summary_path = phase13_artifact_path(str(shared.get("source_summary", "")))
        output_root = workspace_path(str(config["output_dir"]))
        require_within(summary_path, output_root, "recovered reference summary")
        summary = validate_phase13_summary(
            summary_path=summary_path,
            node=node,
            config=config,
            manifest=manifest,
            reference_manifest_sha=None,
        )
        artifacts = tree_hashes(summary_path.parent)
        artifacts[portable_workspace_path(config_path)] = sha256_file(config_path)
        artifacts[portable_workspace_path(shared_path)] = sha256_file(shared_path)
        if not artifacts:
            raise FreshEvaluationError("recovered shared-reference tree is empty")
        second_shared = validate_shared_reference_manifest(
            path=shared_path, manifest=manifest
        )
        second_summary = validate_phase13_summary(
            summary_path=summary_path,
            node=node,
            config=config,
            manifest=manifest,
            reference_manifest_sha=None,
        )
        second_artifacts = tree_hashes(summary_path.parent)
        second_artifacts[portable_workspace_path(config_path)] = sha256_file(config_path)
        second_artifacts[portable_workspace_path(shared_path)] = sha256_file(shared_path)
        if second_shared != shared or second_summary != summary or second_artifacts != artifacts:
            raise FreshEvaluationError(
                "published shared-reference artifacts changed during recovery"
            )
    except (FreshEvaluationError, OSError, ValueError, KeyError, et.ParseError) as exc:
        moved = quarantine_recovery_paths(
            node_id=node.node_id,
            paths=(shared_path,),
            state=state,
            manifest=manifest,
        )
        record.setdefault("recovery_quarantine", []).extend(moved)
        record.setdefault("recovery_events", []).append(
            {
                "at": now_utc(),
                "action": "quarantined_invalid_published_shared_manifest",
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
        state["updated_at"] = now_utc()
        atomic_write_json(state_path, state)
        return False
    attempt = record["attempts"][-1]
    attempt["recovered_after_publish_crash"] = True
    attempt["recovered_at"] = now_utc()
    attempt["output_sha256"] = artifacts
    phase13_seconds = sum(float(item["seconds"]) for item in summary["records"])
    attempt.setdefault("duration_seconds", phase13_seconds)
    record.update(
        {
            "status": "succeeded",
            "output_sha256": artifacts,
            "summary_path": portable_workspace_path(summary_path),
            "config_path": portable_workspace_path(config_path),
            "config_sha256": sha256_file(config_path),
            "process_wall_seconds": float(
                attempt.get("duration_seconds", phase13_seconds)
            ),
            "phase13_seconds": phase13_seconds,
            "shared_reference_manifest_sha256": sha256_file(shared_path),
        }
    )
    state["updated_at"] = now_utc()
    atomic_write_json(state_path, state)
    return True


def run_phase13_node(
    *,
    node: DagNode,
    config_path: Path,
    config: dict[str, Any],
    record: dict[str, Any],
    state: dict[str, Any],
    state_path: Path,
    manifest: dict[str, Any],
    manifest_path: Path,
    immutable_snapshot: dict[str, str],
    python_executable: str,
) -> bool:
    output_root = workspace_path(str(config["output_dir"]))
    if recover_published_shared_reference(
        node=node,
        config_path=config_path,
        config=config,
        record=record,
        state=state,
        state_path=state_path,
        manifest=manifest,
        manifest_path=manifest_path,
        immutable_snapshot=immutable_snapshot,
    ):
        return True
    before = summary_snapshot(output_root)
    if not record.get("attempts") and before:
        raise FreshEvaluationError(
            f"unregistered Phase 13 summaries already exist for {node.node_id}"
        )
    if state.get("metric_barrier") == "locked" and output_root.is_dir() and any(
        output_root.rglob("benchmark_summary_with_metrics.*")
    ):
        raise FreshEvaluationError(
            f"pre-unlock metric artifacts already exist for {node.node_id}"
        )
    reference_path = None
    reference_sha = None
    if config.get("external_reference_manifest") is not None:
        reference_path = workspace_path(str(config["external_reference_manifest"]))
        validate_shared_reference_manifest(path=reference_path, manifest=manifest)
        reference_sha = sha256_file(reference_path)
        registered_reference_sha = state["nodes"]["reference:shared_block"].get(
            "shared_reference_manifest_sha256"
        )
        if reference_sha != registered_reference_sha:
            raise FreshEvaluationError(
                f"shared-reference manifest drift before {node.node_id}"
            )
    inputs_before = node_input_hashes(
        config_path=config_path,
        config=config,
        reference_manifest_path=reference_path,
    )
    attempts = record.setdefault("attempts", [])
    attempt_number = len(attempts) + 1
    log_dir = workspace_path(str(manifest["outputs"]["logs"]))
    log_dir.mkdir(parents=True, exist_ok=True)
    stem = node.node_id.replace(":", "__") + f".attempt{attempt_number}"
    stdout_path = log_dir / f"{stem}.stdout.log"
    stderr_path = log_dir / f"{stem}.stderr.log"
    command = [python_executable, str(workspace_path(manifest["runner_interfaces"]["phase13"]["path"])), "--config", str(config_path)]
    attempt: dict[str, Any] = {
        "attempt": attempt_number,
        "started_at": now_utc(),
        "command": command,
        "input_sha256": inputs_before,
        "stdout_log": portable_workspace_path(stdout_path),
        "stderr_log": portable_workspace_path(stderr_path),
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
    process_seconds = time.monotonic() - started
    attempt["finished_at"] = now_utc()
    attempt["duration_seconds"] = process_seconds
    attempt["return_code"] = result.returncode
    error: str | None = None
    summary: dict[str, Any] | None = None
    artifacts: dict[str, str] = {}
    try:
        assert_snapshot_unchanged(
            manifest_path=manifest_path,
            manifest=manifest,
            expected=immutable_snapshot,
        )
        inputs_after = node_input_hashes(
            config_path=config_path,
            config=config,
            reference_manifest_path=reference_path,
        )
        attempt["post_input_sha256"] = inputs_after
        if inputs_after != inputs_before:
            raise FreshEvaluationError("node input changed while rendering")
        summary_path: Path | None = None
        try:
            summary_path = find_new_summary(output_root, before)
        except FreshEvaluationError as exc:
            if result.returncode == 0:
                raise
            attempt["failed_summary_discovery_error"] = str(exc)
        if result.returncode != 0:
            if summary_path is not None:
                artifacts = tree_hashes(summary_path.parent)
                artifacts[portable_workspace_path(config_path)] = sha256_file(
                    config_path
                )
                attempt.update(phase13_failure_status(summary_path))
                attempt["failed_summary_path"] = portable_workspace_path(summary_path)
                attempt["failed_summary_sha256"] = sha256_file(summary_path)
                attempt["output_sha256"] = artifacts
            raise FreshEvaluationError(f"Phase 13 returned {result.returncode}")
        if summary_path is None:
            raise FreshEvaluationError("successful Phase 13 run has no unique summary")
        summary = validate_phase13_summary(
            summary_path=summary_path,
            node=node,
            config=config,
            manifest=manifest,
            reference_manifest_sha=reference_sha,
        )
        artifacts = tree_hashes(summary_path.parent)
        artifacts[portable_workspace_path(config_path)] = sha256_file(config_path)
        if node.stage == "shared_references":
            shared_path = workspace_path(
                str(config["shared_reference_manifest_output"])
            )
            validate_shared_reference_manifest(path=shared_path, manifest=manifest)
            artifacts[portable_workspace_path(shared_path)] = sha256_file(shared_path)
        second_summary = validate_phase13_summary(
            summary_path=summary_path,
            node=node,
            config=config,
            manifest=manifest,
            reference_manifest_sha=reference_sha,
        )
        if second_summary != summary:
            raise FreshEvaluationError("Phase 13 summary changed during validation")
        second_artifacts = tree_hashes(summary_path.parent)
        second_artifacts[portable_workspace_path(config_path)] = sha256_file(config_path)
        if node.stage == "shared_references":
            second_artifacts[portable_workspace_path(shared_path)] = sha256_file(
                shared_path
            )
        if second_artifacts != artifacts:
            raise FreshEvaluationError("Phase 13 artifacts changed during validation")
    except (FreshEvaluationError, OSError, ValueError, KeyError, et.ParseError) as exc:
        error = f"{type(exc).__name__}: {exc}"
    if error is None and summary is not None and artifacts:
        record["status"] = "succeeded"
        record["output_sha256"] = artifacts
        record["summary_path"] = portable_workspace_path(summary_path)
        record["config_path"] = portable_workspace_path(config_path)
        record["config_sha256"] = sha256_file(config_path)
        record["process_wall_seconds"] = process_seconds
        record["phase13_seconds"] = sum(
            float(item["seconds"]) for item in summary["records"]
        )
        if node.stage != "shared_references":
            item = summary["records"][0]
            record["metrics_deferred"] = summary["metrics_status"] == "deferred"
            record["mse"] = item["mse"]
            record["psnr"] = item["psnr"]
            record["nonfinite_pixel_boolean"] = item["nonfinite_pixel_boolean"]
            record["sample_count"] = item["sample_count"]
            record["sampler_seed"] = item["sampler_seed"]
            record["reference_manifest_sha256"] = reference_sha
            if node.stage == "equal_time":
                record["spp_map_sha256"] = state["nodes"][
                    "calibration:freeze_spp_map"
                ]["spp_map_sha256"]
        else:
            shared_path = workspace_path(str(config["shared_reference_manifest_output"]))
            record["shared_reference_manifest_sha256"] = sha256_file(shared_path)
        attempt["output_sha256"] = artifacts
        succeeded = True
    else:
        record["status"] = "failed"
        record["output_sha256"] = artifacts
        if artifacts:
            attempt["output_sha256"] = artifacts
        attempt["error"] = error
        succeeded = False
    state["updated_at"] = now_utc()
    atomic_write_json(state_path, state)
    return succeeded


def dependency_is_complete(record: dict[str, Any]) -> bool:
    return classify_node_record(record) in {"succeeded", "skipped"}


def completed_descendants(
    *, node_id: str, nodes: tuple[DagNode, ...], state: dict[str, Any]
) -> list[str]:
    descendants: set[str] = set()
    frontier = {node_id}
    while frontier:
        next_frontier: set[str] = set()
        for node in nodes:
            if node.node_id in descendants or node.node_id == node_id:
                continue
            if any(dependency in frontier for dependency in node.dependencies):
                descendants.add(node.node_id)
                next_frontier.add(node.node_id)
        frontier = next_frontier
    return sorted(
        descendant
        for descendant in descendants
        if classify_node_record(state["nodes"][descendant])
        in {"succeeded", "skipped"}
    )


def quarantine_recovery_paths(
    *,
    node_id: str,
    paths: Iterable[Path],
    state: dict[str, Any],
    manifest: dict[str, Any],
) -> list[dict[str, str]]:
    """Transactionally preserve outputs; resume an interrupted move by its plan."""

    requested = sorted({path.resolve() for path in paths})
    quarantine_root = (
        workspace_path(str(manifest["outputs"]["root"]))
        / "recovery_quarantine"
        / node_id.replace(":", "__")
        / (
            "attempt_"
            + str(
                len(state["nodes"][node_id].get("attempts", []))
                + len(state["nodes"][node_id].get("recovery_events", []))
                + 1
            )
        )
    )
    plan_path = quarantine_root / "quarantine_plan.json"
    if plan_path.is_file():
        plan = load_json(plan_path)
        if (
            plan.get("schema_version") != 1
            or plan.get("protocol_id") != "fresh_evaluation_recovery_quarantine_v1"
            or plan.get("node_id") != node_id
        ):
            raise FreshEvaluationError(f"invalid recovery quarantine plan: {plan_path}")
        entries = plan.get("entries")
        if not isinstance(entries, list) or not entries:
            raise FreshEvaluationError(f"empty recovery quarantine plan: {plan_path}")
    else:
        existing = [path for path in requested if path.is_file()]
        if not existing:
            return []
        basenames = [path.name for path in existing]
        if len(set(basenames)) != len(basenames):
            raise FreshEvaluationError("recovery paths have colliding basenames")
        quarantine_root.mkdir(parents=True, exist_ok=True)
        unexpected = [path for path in quarantine_root.iterdir() if path != plan_path]
        if unexpected:
            raise FreshEvaluationError(
                f"unplanned files in recovery quarantine: {quarantine_root}"
            )
        entries = [
            {
                "source": portable_workspace_path(source),
                "quarantine": portable_workspace_path(quarantine_root / source.name),
                "sha256": sha256_file(source),
            }
            for source in existing
        ]
        atomic_write_json(
            plan_path,
            {
                "schema_version": 1,
                "protocol_id": "fresh_evaluation_recovery_quarantine_v1",
                "node_id": node_id,
                "entries": entries,
            },
        )
    requested_text = {portable_workspace_path(path) for path in requested}
    planned_text = {str(entry.get("source", "")) for entry in entries}
    if not planned_text <= requested_text:
        raise FreshEvaluationError(
            f"recovery quarantine plan source set changed: {plan_path}"
        )
    unexpected_sources = [
        path
        for path in requested
        if path.is_file() and portable_workspace_path(path) not in planned_text
    ]
    if unexpected_sources:
        raise FreshEvaluationError(
            "new unplanned outputs appeared during recovery quarantine: "
            + ", ".join(portable_workspace_path(path) for path in unexpected_sources)
        )
    expected_files = {plan_path.resolve()}
    moved: list[dict[str, str]] = []
    for raw_entry in entries:
        if not isinstance(raw_entry, dict):
            raise FreshEvaluationError(f"invalid quarantine entry: {plan_path}")
        source = workspace_path(str(raw_entry.get("source", "")))
        destination = workspace_path(str(raw_entry.get("quarantine", "")))
        if destination.parent != quarantine_root.resolve():
            raise FreshEvaluationError(f"quarantine destination drift: {destination}")
        expected_files.add(destination)
        expected_sha = str(raw_entry.get("sha256", ""))
        source_exists = source.is_file()
        destination_exists = destination.is_file()
        if source_exists and destination_exists:
            raise FreshEvaluationError(
                f"both source and quarantine destination exist: {source}"
            )
        if destination_exists:
            if sha256_file(destination) != expected_sha:
                raise FreshEvaluationError(
                    f"resumed quarantine hash mismatch: {destination}"
                )
        elif source_exists:
            if sha256_file(source) != expected_sha:
                raise FreshEvaluationError(f"quarantine source hash drift: {source}")
            source.replace(destination)
            if sha256_file(destination) != expected_sha:
                raise FreshEvaluationError(f"quarantine hash mismatch: {destination}")
        else:
            raise FreshEvaluationError(
                f"quarantine lost both source and destination: {source}"
            )
        moved.append(
            {
                "source": portable_workspace_path(source),
                "quarantine": portable_workspace_path(destination),
                "sha256": expected_sha,
                "plan": portable_workspace_path(plan_path),
                "plan_sha256": sha256_file(plan_path),
            }
        )
    unexpected = [
        path.resolve()
        for path in quarantine_root.iterdir()
        if path.resolve() not in expected_files
    ]
    if unexpected:
        raise FreshEvaluationError(
            f"unregistered recovery quarantine artifacts: {quarantine_root}"
        )
    return moved


def quarantine_stale_metric_outputs(
    *,
    node: DagNode,
    state: dict[str, Any],
    manifest: dict[str, Any],
) -> list[dict[str, str]]:
    """Preserve partial post-unlock outputs before a metric-only retry."""

    candidates: list[Path] = []
    if node.stage == "deferred_metric_finalization":
        source_id = source_node_for_finalizer(node)
        source_summary = workspace_path(str(state["nodes"][source_id]["summary_path"]))
        candidates.extend(
            source_summary.parent / name
            for name in (
                "benchmark_summary_with_metrics.json",
                "benchmark_summary_with_metrics.csv",
                "benchmark_summary_with_metrics.md",
            )
        )
    elif node.stage == "hierarchical_summary":
        candidates.extend(
            [
                workspace_path(str(manifest["outputs"]["metric_index"])),
                workspace_path(str(manifest["stages"]["final_summary"]["output_json"])),
                workspace_path(
                    str(manifest["stages"]["final_summary"]["output_markdown"])
                ),
            ]
        )
    else:
        raise FreshEvaluationError(
            f"post-unlock rerun is not metric-only: {node.node_id}"
        )
    return quarantine_recovery_paths(
        node_id=node.node_id,
        paths=candidates,
        state=state,
        manifest=manifest,
    )


def complete_receipt_gate(
    *, node: DagNode, record: dict[str, Any], receipt_path: Path, state: dict[str, Any], state_path: Path
) -> None:
    digest = sha256_file(receipt_path)
    record.update(
        {
            "status": "succeeded",
            "output_sha256": {portable_workspace_path(receipt_path): digest},
            "receipt_sha256": digest,
        }
    )
    state["updated_at"] = now_utc()
    atomic_write_json(state_path, state)


def repeat_is_required(
    *, node: DagNode, state: dict[str, Any], manifest: dict[str, Any]
) -> tuple[bool, float]:
    if node.scene is None or node.method is None:
        raise FreshEvaluationError("conditional calibration node lacks identity")
    first_id = f"calibration:{node.scene}:{node.method}:spp16:rep1"
    second_id = f"calibration:{node.scene}:{node.method}:spp16:rep2"
    first = state["nodes"][first_id]
    second = state["nodes"][second_id]
    if classify_node_record(first) != "succeeded" or classify_node_record(
        second
    ) != "succeeded":
        raise FreshEvaluationError(
            f"initial 16-spp timing probes are incomplete: {node.scene}/{node.method}"
        )
    first_seconds = float(first["process_wall_seconds"])
    second_seconds = float(second["process_wall_seconds"])
    mean_seconds = (first_seconds + second_seconds) / 2.0
    if first_seconds <= 0.0 or second_seconds <= 0.0 or mean_seconds <= 0.0:
        raise FreshEvaluationError(
            f"invalid timing probes: {node.scene}/{node.method}"
        )
    fraction = abs(first_seconds - second_seconds) / mean_seconds
    threshold = float(
        manifest["stages"]["timing_calibration"]["repeat_threshold_fraction"]
    )
    return fraction > threshold, fraction


def freeze_spp_map(
    *,
    manifest: dict[str, Any],
    state: dict[str, Any],
    record: dict[str, Any],
    state_path: Path,
) -> None:
    cells: dict[str, Any] = {}
    spp_map: dict[str, int] = {}
    for scene in EXPECTED_SCENES:
        for method in EXPECTED_METHODS:
            one_id = f"calibration:{scene}:{method}:spp1"
            sixteen_ids = [
                f"calibration:{scene}:{method}:spp16:rep1",
                f"calibration:{scene}:{method}:spp16:rep2",
            ]
            repeat_id = f"calibration:{scene}:{method}:spp16:conditional_repeat"
            one = state["nodes"][one_id]
            sixteen_records = [state["nodes"][node_id] for node_id in sixteen_ids]
            repeat = state["nodes"][repeat_id]
            if classify_node_record(one) != "succeeded" or any(
                classify_node_record(item) != "succeeded"
                for item in sixteen_records
            ):
                raise FreshEvaluationError(f"incomplete calibration cell: {scene}/{method}")
            times_16 = [
                float(item["process_wall_seconds"]) for item in sixteen_records
            ]
            repeat_status = classify_node_record(repeat)
            if repeat_status == "succeeded":
                times_16.append(float(repeat["process_wall_seconds"]))
            elif repeat_status != "skipped":
                raise FreshEvaluationError(
                    f"conditional calibration unresolved: {scene}/{method}"
                )
            ordered_times = sorted(times_16)
            middle = len(ordered_times) // 2
            seconds_16 = (
                ordered_times[middle]
                if len(ordered_times) % 2 == 1
                else (ordered_times[middle - 1] + ordered_times[middle]) / 2.0
            )
            fitted = fit_equal_time_spp(
                seconds_at_1=float(one["process_wall_seconds"]),
                seconds_at_16=seconds_16,
                target_seconds=float(
                    manifest["stages"]["timing_calibration"]["target_seconds"]
                ),
            )
            key = f"{scene}|{method}"
            spp_map[key] = int(fitted["sample_count"])
            cells[key] = {
                **fitted,
                "one_spp_node": one_id,
                "sixteen_spp_nodes": [
                    *sixteen_ids,
                    *([repeat_id] if repeat_status == "succeeded" else []),
                ],
                "one_spp_process_wall_seconds": one["process_wall_seconds"],
                "sixteen_spp_process_wall_seconds": times_16,
                "repeat_status": repeat_status,
            }
    payload = {
        "schema_version": 1,
        "protocol_id": EXPECTED_PROTOCOL,
        "fit_time_source": manifest["stages"]["timing_calibration"][
            "fit_time_source"
        ],
        "target_seconds": manifest["stages"]["timing_calibration"][
            "target_seconds"
        ],
        "scene_method_compare_sample_counts": spp_map,
        "cells": cells,
        "quality_metrics_inspected": False,
    }
    output_path = workspace_path(
        str(manifest["stages"]["timing_calibration"]["frozen_map_path"])
    )
    digest = write_frozen_config(output_path, payload)
    record.update(
        {
            "status": "succeeded",
            "output_sha256": {portable_workspace_path(output_path): digest},
            "spp_map_path": portable_workspace_path(output_path),
            "spp_map_sha256": digest,
        }
    )
    state["updated_at"] = now_utc()
    atomic_write_json(state_path, state)


def load_frozen_spp_map(
    *, manifest: dict[str, Any], state: dict[str, Any]
) -> dict[str, int]:
    record = state["nodes"]["calibration:freeze_spp_map"]
    if classify_node_record(record) != "succeeded":
        raise FreshEvaluationError("equal-time SPP map node is not successful")
    path = workspace_path(str(record["spp_map_path"]))
    if sha256_file(path) != record.get("spp_map_sha256"):
        raise FreshEvaluationError("equal-time SPP map hash drift")
    payload = load_json(path)
    if payload.get("protocol_id") != EXPECTED_PROTOCOL:
        raise FreshEvaluationError("equal-time SPP map protocol mismatch")
    values = payload.get("scene_method_compare_sample_counts")
    if not isinstance(values, dict) or set(values) != {
        f"{scene}|{method}"
        for scene in EXPECTED_SCENES
        for method in EXPECTED_METHODS
    }:
        raise FreshEvaluationError("equal-time SPP map cell set changed")
    return {str(key): int(value) for key, value in values.items()}


def write_metric_unlock(
    *,
    manifest: dict[str, Any],
    manifest_sha: str,
    state: dict[str, Any],
    record: dict[str, Any],
    state_path: Path,
    dependency_hashes: dict[str, str],
) -> None:
    base_payload = validate_metric_unlock_eligibility(
        state=state,
        nodes=build_dag(manifest),
        manifest=manifest,
    )
    base_payload.update(
        {
            "schema_version": 1,
            "status": "metrics_unlocked",
            "manifest_sha256": manifest_sha,
            "dependency_sha256": dependency_hashes,
        }
    )
    output_path = workspace_path(
        str(manifest["stages"]["metric_barrier"]["unlock_receipt_path"])
    )
    if output_path.exists():
        recovery_error: str | None = None
        try:
            existing = load_json(output_path)
            if set(existing) != set(base_payload) | {"unlocked_at"}:
                raise FreshEvaluationError("existing unlock receipt keys changed")
            if not isinstance(existing.get("unlocked_at"), str) or not existing[
                "unlocked_at"
            ]:
                raise FreshEvaluationError("existing unlock receipt lacks timestamp")
            for key, expected in base_payload.items():
                if existing.get(key) != expected:
                    raise FreshEvaluationError(
                        f"existing unlock receipt mismatch: {key}"
                    )
        except (FreshEvaluationError, OSError, ValueError, KeyError) as exc:
            recovery_error = f"{type(exc).__name__}: {exc}"
        if recovery_error is None:
            digest = sha256_file(output_path)
            record.update(
                {
                    "status": "succeeded",
                    "output_sha256": {portable_workspace_path(output_path): digest},
                    "unlock_receipt_path": portable_workspace_path(output_path),
                    "unlock_receipt_sha256": digest,
                    "recovered_after_publish_crash": True,
                }
            )
            state["metric_barrier"] = "unlocked"
            state["updated_at"] = now_utc()
            atomic_write_json(state_path, state)
            return
        moved = quarantine_recovery_paths(
            node_id="barrier:unlock_candidate_metrics",
            paths=(output_path,),
            state=state,
            manifest=manifest,
        )
        record.setdefault("recovery_quarantine", []).extend(moved)
        record.setdefault("recovery_events", []).append(
            {
                "at": now_utc(),
                "action": "quarantined_invalid_unlock_receipt",
                "error": recovery_error,
            }
        )
        state["updated_at"] = now_utc()
        atomic_write_json(state_path, state)
    payload = dict(base_payload)
    payload["unlocked_at"] = now_utc()
    atomic_write_json(output_path, payload)
    digest = sha256_file(output_path)
    record.update(
        {
            "status": "succeeded",
            "output_sha256": {portable_workspace_path(output_path): digest},
            "unlock_receipt_path": portable_workspace_path(output_path),
            "unlock_receipt_sha256": digest,
        }
    )
    state["metric_barrier"] = "unlocked"
    state["updated_at"] = now_utc()
    atomic_write_json(state_path, state)


def source_node_for_finalizer(node: DagNode) -> str:
    suffix = node.node_id.removeprefix("finalize:")
    if suffix.startswith("reference_noise:"):
        return suffix
    return "candidate:" + suffix


def run_finalizer_node(
    *,
    node: DagNode,
    record: dict[str, Any],
    state: dict[str, Any],
    state_path: Path,
    manifest: dict[str, Any],
    manifest_path: Path,
    immutable_snapshot: dict[str, str],
    python_executable: str,
) -> bool:
    source_id = source_node_for_finalizer(node)
    source_record = state["nodes"][source_id]
    if classify_node_record(source_record) != "succeeded":
        raise FreshEvaluationError(f"finalizer source is not verified: {source_id}")
    source_path = workspace_path(str(source_record["summary_path"]))
    source_sha = sha256_file(source_path)
    run_dir = source_path.parent
    output_json = run_dir / "benchmark_summary_with_metrics.json"
    output_csv = run_dir / "benchmark_summary_with_metrics.csv"
    output_md = run_dir / "benchmark_summary_with_metrics.md"
    if any(path.exists() for path in (output_json, output_csv, output_md)):
        raise FreshEvaluationError(
            f"unregistered finalized metric output already exists for {node.node_id}"
        )
    attempts = record.setdefault("attempts", [])
    attempt_number = len(attempts) + 1
    logs = workspace_path(str(manifest["outputs"]["logs"]))
    logs.mkdir(parents=True, exist_ok=True)
    stem = node.node_id.replace(":", "__") + f".attempt{attempt_number}"
    stdout_path = logs / f"{stem}.stdout.log"
    stderr_path = logs / f"{stem}.stderr.log"
    command = [
        python_executable,
        str(
            workspace_path(
                str(manifest["runner_interfaces"]["deferred_metric_finalizer"]["path"])
            )
        ),
        "--summary",
        str(source_path),
        "--unlock-metrics",
    ]
    attempt = {
        "attempt": attempt_number,
        "started_at": now_utc(),
        "command": command,
        "source_summary_sha256": source_sha,
        "stdout_log": portable_workspace_path(stdout_path),
        "stderr_log": portable_workspace_path(stderr_path),
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
    attempt["duration_seconds"] = time.monotonic() - started
    attempt["finished_at"] = now_utc()
    attempt["return_code"] = result.returncode
    error: str | None = None
    metric_row: dict[str, Any] | None = None
    output_hashes: dict[str, str] = {}
    try:
        assert_snapshot_unchanged(
            manifest_path=manifest_path,
            manifest=manifest,
            expected=immutable_snapshot,
        )
        if sha256_file(source_path) != source_sha:
            raise FreshEvaluationError("deferred source summary changed during finalization")
        if result.returncode != 0:
            raise FreshEvaluationError(f"metric finalizer returned {result.returncode}")
        if not all(path.is_file() for path in (output_json, output_csv, output_md)):
            raise FreshEvaluationError("metric finalizer did not write all three outputs")
        summary = load_json(output_json)
        source_summary = load_json(source_path)
        if summary.get("metrics_status") != "computed_after_deferred_unlock":
            raise FreshEvaluationError("finalized metric status is invalid")
        source_link = summary.get("deferred_source_summary", {})
        if source_link.get("sha256") != source_sha:
            raise FreshEvaluationError("finalized summary source hash mismatch")
        rows = summary.get("records")
        if not isinstance(rows, list) or len(rows) != 1:
            raise FreshEvaluationError("finalized summary must contain one record")
        row = rows[0]
        source_rows = source_summary.get("records")
        if not isinstance(source_rows, list) or len(source_rows) != 1:
            raise FreshEvaluationError("deferred source summary row count changed")
        if summary.get("config") != source_summary.get("config"):
            raise FreshEvaluationError("finalized config differs from deferred source")
        if summary.get("reference_provenance") != source_summary.get(
            "reference_provenance"
        ):
            raise FreshEvaluationError("finalized reference provenance changed")
        for key in (
            "scene_id",
            "method",
            "sample_count",
            "sampler_seed",
            "active_config_sha256",
            "output_sha256",
        ):
            if row.get(key) != source_rows[0].get(key):
                raise FreshEvaluationError(f"finalized row changed identity field {key}")
        mse = float(row.get("mse", math.nan))
        psnr = float(row.get("psnr", math.nan))
        if not math.isfinite(mse) or mse < 0.0:
            raise FreshEvaluationError("finalized MSE is invalid")
        if not (math.isfinite(psnr) or math.isinf(psnr)):
            raise FreshEvaluationError("finalized PSNR is invalid")
        output_hashes = {
            portable_workspace_path(path): sha256_file(path)
            for path in (output_json, output_csv, output_md)
        }
        metric_row = {
            "scene": node.scene,
            "method": node.method,
            "mse": mse,
            "psnr": psnr,
            "sample_count": int(row["sample_count"]),
            "render_seed": int(row["sampler_seed"]),
            "seconds": float(source_record["process_wall_seconds"]),
            "phase13_seconds": float(source_record["phase13_seconds"]),
            "finalized_summary_path": portable_workspace_path(output_json),
            "finalized_summary_sha256": output_hashes[
                portable_workspace_path(output_json)
            ],
            "deferred_summary_path": portable_workspace_path(source_path),
            "deferred_summary_sha256": source_sha,
        }
        source_node = next(item for item in build_dag(manifest) if item.node_id == source_id)
        metric_row["regime"] = source_node.stage
        metric_row["train_seed"] = source_node.train_seed
        metric_row["render_seed"] = source_node.render_seed or int(row["sampler_seed"])
        if sha256_file(source_path) != source_sha:
            raise FreshEvaluationError("deferred source changed after finalization")
        if {
            portable_workspace_path(path): sha256_file(path)
            for path in (output_json, output_csv, output_md)
        } != output_hashes:
            raise FreshEvaluationError("finalized outputs changed during validation")
    except (FreshEvaluationError, OSError, ValueError, KeyError) as exc:
        error = f"{type(exc).__name__}: {exc}"
        # Register every partial metric artifact even on failure. This makes the
        # failed node classify as stale on resume, so the post-unlock recovery
        # path can hash-verify and quarantine the partials before retrying.
        output_hashes = {
            portable_workspace_path(path): sha256_file(path)
            for path in (output_json, output_csv, output_md)
            if path.is_file()
        }
    if error is None and metric_row is not None:
        record["status"] = "succeeded"
        record["output_sha256"] = output_hashes
        record["metric_row"] = metric_row
        attempt["output_sha256"] = output_hashes
        succeeded = True
    else:
        record["status"] = "failed"
        record["output_sha256"] = output_hashes
        attempt["error"] = error
        succeeded = False
    state["updated_at"] = now_utc()
    atomic_write_json(state_path, state)
    return succeeded


def run_hierarchical_summary_node(
    *,
    record: dict[str, Any],
    state: dict[str, Any],
    state_path: Path,
    manifest: dict[str, Any],
    manifest_path: Path,
    immutable_snapshot: dict[str, str],
    nodes: tuple[DagNode, ...],
    python_executable: str,
) -> bool:
    unlock_record = state["nodes"]["barrier:unlock_candidate_metrics"]
    if classify_node_record(unlock_record) != "succeeded":
        raise FreshEvaluationError("metric unlock receipt is not verified")
    unlock_path = workspace_path(str(unlock_record["unlock_receipt_path"]))
    candidate_rows: list[dict[str, Any]] = []
    noise_rows: list[dict[str, Any]] = []
    for node_id, node_record in state["nodes"].items():
        if not node_id.startswith("finalize:"):
            continue
        if classify_node_record(node_record) != "succeeded":
            raise FreshEvaluationError(f"finalized metric is not verified: {node_id}")
        metric_row = dict(node_record["metric_row"])
        if metric_row["regime"] == "reference_noise_check":
            noise_rows.append(metric_row)
        else:
            candidate_rows.append(metric_row)
    if len(candidate_rows) != 350 or len(noise_rows) != 5:
        raise FreshEvaluationError("metric index cardinality is incomplete")
    stage_by_node = {item.node_id: item.stage for item in nodes}
    index_path = workspace_path(str(manifest["outputs"]["metric_index"]))
    index = {
        "schema_version": 1,
        "protocol_id": "fresh_confirmatory_metric_index_20260901_v1",
        "unlock_receipt_path": portable_workspace_path(unlock_path),
        "unlock_receipt_sha256": sha256_file(unlock_path),
        "records": sorted(
            candidate_rows,
            key=lambda row: (
                row["regime"],
                row["scene"],
                row["method"],
                -1 if row["train_seed"] is None else row["train_seed"],
                row["render_seed"],
            ),
        ),
        "reference_noise_records": sorted(noise_rows, key=lambda row: row["scene"]),
        "training_accounting": training_accounting(manifest),
        "protocol_limitations": [manifest["study"]["training_pairing_limitation"]],
        "evaluation_accounting": {
            "process_wall_seconds_by_stage": {
                stage: sum(
                    sum(
                        float(attempt.get("duration_seconds", 0.0))
                        for attempt in node_record.get("attempts", [])
                    )
                    for node_id, node_record in state["nodes"].items()
                    if stage_by_node[node_id] == stage
                )
                for stage in (
                    "shared_references",
                    "reference_noise_check",
                    "equal_spp",
                    "timing_calibration",
                    "timing_calibration_repeat",
                    "equal_time",
                )
            },
            "conditional_timing_repeats_executed": sum(
                classify_node_record(state["nodes"][item.node_id]) == "succeeded"
                for item in nodes
                if item.stage == "timing_calibration_repeat"
            ),
            "failed_attempts": sum(
                attempt.get("return_code") not in (None, 0)
                or bool(attempt.get("error"))
                for node_record in state["nodes"].values()
                for attempt in node_record.get("attempts", [])
            ),
            "timeout_attempts": sum(
                attempt.get("return_code") in {124, -9}
                or int(attempt.get("phase13_timeout_record_count", 0)) > 0
                or "timeout" in str(attempt.get("error", "")).lower()
                for node_record in state["nodes"].values()
                for attempt in node_record.get("attempts", [])
            ),
            "failed_render_records": sum(
                int(attempt.get("phase13_failed_record_count", 0))
                for node_record in state["nodes"].values()
                for attempt in node_record.get("attempts", [])
            ),
            "timed_out_render_records": sum(
                int(attempt.get("phase13_timeout_record_count", 0))
                for node_record in state["nodes"].values()
                for attempt in node_record.get("attempts", [])
            ),
        },
    }
    index_sha = write_frozen_config(index_path, index)
    output_json = workspace_path(
        str(manifest["stages"]["final_summary"]["output_json"])
    )
    output_md = workspace_path(
        str(manifest["stages"]["final_summary"]["output_markdown"])
    )
    if output_json.exists() or output_md.exists():
        raise FreshEvaluationError("unregistered final summary already exists")
    summarizer = workspace_path(
        str(manifest["runner_interfaces"]["hierarchical_summarizer"]["path"])
    )
    command = [
        python_executable,
        str(summarizer),
        "--unlock-receipt",
        str(unlock_path),
        "--metric-index",
        str(index_path),
        "--output-json",
        str(output_json),
        "--output-markdown",
        str(output_md),
        "--unlock-metrics",
        "--bootstrap-replicates",
        str(manifest["stages"]["final_summary"]["bootstrap_replicates"]),
        "--bootstrap-seed",
        str(manifest["stages"]["final_summary"]["bootstrap_seed"]),
    ]
    attempts = record.setdefault("attempts", [])
    attempt = {"attempt": len(attempts) + 1, "started_at": now_utc(), "command": command}
    attempts.append(attempt)
    record["status"] = "running"
    state["updated_at"] = now_utc()
    atomic_write_json(state_path, state)
    result = subprocess.run(
        command,
        cwd=WORKSPACE_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    attempt["finished_at"] = now_utc()
    attempt["return_code"] = result.returncode
    attempt["stdout"] = result.stdout
    attempt["stderr"] = result.stderr
    assert_snapshot_unchanged(
        manifest_path=manifest_path,
        manifest=manifest,
        expected=immutable_snapshot,
    )
    for source_node in nodes:
        if source_node.stage != "deferred_metric_finalization":
            continue
        if classify_node_record(state["nodes"][source_node.node_id]) != "succeeded":
            raise FreshEvaluationError(
                f"finalized metric changed during aggregation: {source_node.node_id}"
            )
    if result.returncode != 0 or not output_json.is_file() or not output_md.is_file():
        record["status"] = "failed"
        record["output_sha256"] = {
            portable_workspace_path(path): sha256_file(path)
            for path in (index_path, output_json, output_md)
            if path.is_file()
        }
        atomic_write_json(state_path, state)
        return False
    summary = load_json(output_json)
    if summary.get("candidate_records") != 350 or set(summary.get("regimes", {})) != set(REGIMES):
        raise FreshEvaluationError("final hierarchical summary contract mismatch")
    hashes = {
        portable_workspace_path(path): sha256_file(path)
        for path in (index_path, output_json, output_md)
    }
    if hashes[portable_workspace_path(index_path)] != index_sha:
        raise FreshEvaluationError("metric index changed during final aggregation")
    record["status"] = "succeeded"
    record["output_sha256"] = hashes
    attempt["output_sha256"] = hashes
    state["updated_at"] = now_utc()
    atomic_write_json(state_path, state)
    return True


def status_summary(nodes: tuple[DagNode, ...]) -> dict[str, int]:
    counts = Counter(node.stage for node in nodes)
    counts["candidate_renders"] = sum(
        node.render_count for node in nodes if node.stage in {"equal_spp", "equal_time"}
    )
    counts["always_rendered_rasters"] = sum(
        node.render_count
        for node in nodes
        if node.stage
        in {
            "shared_references",
            "reference_noise_check",
            "equal_spp",
            "timing_calibration",
            "equal_time",
        }
    )
    counts["conditional_calibration_rasters_max"] = sum(
        node.render_count for node in nodes if node.conditional
    )
    return dict(counts)


def main() -> int:
    args = parse_args()
    if args.max_nodes is not None and args.max_nodes <= 0:
        raise FreshEvaluationError("--max-nodes must be positive")
    if args.max_nodes is not None and not args.execute:
        raise FreshEvaluationError("--max-nodes is meaningful only with --execute")

    manifest_path = args.manifest.resolve()
    manifest, manifest_sha = validate_manifest(manifest_path)
    nodes = build_dag(manifest)
    blockers = placeholder_blockers(manifest)
    dependency_blockers = [
        blocker for blocker in blockers if blocker != "execution_authorized"
    ]
    checked_dependency_hashes: dict[str, str] | None = None
    if not dependency_blockers:
        checked_dependency_hashes = validate_dependencies(manifest)
    counts = status_summary(nodes)
    print(f"protocol={EXPECTED_PROTOCOL}")
    print(f"manifest={manifest_path}")
    print(f"manifest_sha256={manifest_sha}")
    print(f"logical_nodes={len(nodes)}")
    print(f"candidate_renders={counts['candidate_renders']}")
    print(f"always_rendered_rasters={counts['always_rendered_rasters']}")
    print(
        "conditional_calibration_rasters_max="
        f"{counts['conditional_calibration_rasters_max']}"
    )
    print(
        "estimated_render_only_hours="
        f"{manifest['estimate']['render_only_hours_low']}-"
        f"{manifest['estimate']['render_only_hours_high']}"
    )
    print("metric_barrier=locked")
    if blockers:
        print("blocked_by=" + ",".join(blockers))
    else:
        print("blocked_by=none")
    print(
        "dependency_gates="
        + ("verified" if checked_dependency_hashes is not None else "pending")
    )
    if args.print_node_ids:
        for node in nodes:
            print(node.node_id)

    if not args.execute:
        print("check-only: no state, config, reference, render, metric, or lock was written")
        return 0
    if args.authorization_token != AUTHORIZATION_TOKEN:
        raise FreshEvaluationError(
            f"--execute requires --authorization-token {AUTHORIZATION_TOKEN}"
        )
    if blockers:
        raise FreshEvaluationError(
            "execution is fail-closed until every placeholder is reviewed and pinned: "
            + ", ".join(blockers)
        )

    outputs = manifest["outputs"]
    lock_path = workspace_path(str(outputs["lock"]))
    runner_lock = acquire_lock(lock_path, manifest_sha)
    if runner_lock.fileno() < 0:
        raise FreshEvaluationError("could not retain evaluation lock")
    manifest_after, manifest_sha_after = validate_manifest(manifest_path)
    if manifest_sha_after != manifest_sha or manifest_after != manifest:
        raise FreshEvaluationError("evaluation manifest changed during lock acquisition")
    dependency_hashes = validate_dependencies(manifest)
    if (
        checked_dependency_hashes is not None
        and dependency_hashes != checked_dependency_hashes
    ):
        raise FreshEvaluationError("dependency receipts changed before lock acquisition")
    conflicts = active_conflicting_processes()
    if conflicts:
        raise FreshEvaluationError(
            "refusing to overlap training/export/render processes:\n"
            + "\n".join(conflicts)
        )
    immutable_snapshot = immutable_input_snapshot(
        manifest_path=manifest_path,
        manifest=manifest,
    )
    runtimes = runtime_registry(manifest)
    state_path = workspace_path(str(outputs["state"]))
    state = load_or_prepare_state(
        state_path=state_path,
        manifest_sha=manifest_sha,
        nodes=nodes,
        dependency_hashes=dependency_hashes,
    )
    environment = probe_python(args.python_executable)
    if state.get("execution_environment") not in (None, environment):
        raise FreshEvaluationError("Python/NumPy environment differs from resume state")
    state["execution_environment"] = environment
    resolved_python = environment["resolved_executable"]

    stale = [
        node.node_id
        for node in nodes
        if classify_node_record(state["nodes"][node.node_id]) == "stale"
    ]
    if stale and not args.rerun_stale:
        raise FreshEvaluationError(
            "stale/interrupted outputs require review and --rerun-stale: "
            + ", ".join(stale)
        )
    if stale:
        unlocked = state.get("metric_barrier") == "unlocked"
        for node_id in stale:
            stale_node = next(item for item in nodes if item.node_id == node_id)
            if unlocked and stale_node.stage not in {
                "deferred_metric_finalization",
                "hierarchical_summary",
            }:
                raise FreshEvaluationError(
                    "render/calibration/reference reruns are forbidden after metric "
                    f"unlock: {node_id}; start a new protocol/output root"
                )
            descendants = completed_descendants(
                node_id=node_id,
                nodes=nodes,
                state=state,
            )
            if descendants:
                raise FreshEvaluationError(
                    f"cannot rerun stale parent {node_id}; completed descendants "
                    "would be invalidated: "
                    + ", ".join(descendants[:10])
                )
            quarantined: list[dict[str, str]] = []
            if unlocked:
                quarantined = quarantine_stale_metric_outputs(
                    node=stale_node,
                    state=state,
                    manifest=manifest,
                )
                state["nodes"][node_id].setdefault(
                    "recovery_quarantine", []
                ).extend(quarantined)
            state["nodes"][node_id].update(
                {"status": "pending", "output_sha256": {}}
            )
            state["nodes"][node_id].pop("metric_row", None)
    atomic_write_json(state_path, state)

    executed = 0
    model_receipt_path = workspace_path(
        str(manifest["dependency_receipts"]["model_artifacts"]["path"])
    )
    smoke_receipt_path = workspace_path(
        str(manifest["dependency_receipts"]["status_only_smoke"]["path"])
    )
    for node in nodes:
        _, current_manifest_sha = validate_manifest(manifest_path)
        if current_manifest_sha != manifest_sha:
            raise FreshEvaluationError("evaluation manifest changed before next node")
        assert_snapshot_unchanged(
            manifest_path=manifest_path,
            manifest=manifest,
            expected=immutable_snapshot,
        )
        record = state["nodes"][node.node_id]
        classification = classify_node_record(record)
        if classification in {"succeeded", "skipped"}:
            print(f"SKIP {node.node_id} ({classification})")
            continue
        for dependency in node.dependencies:
            dependency_record = state["nodes"][dependency]
            if not dependency_is_complete(dependency_record):
                raise FreshEvaluationError(
                    f"dependency {dependency} is not complete for {node.node_id}"
                )
        conflicts = active_conflicting_processes()
        if conflicts:
            raise FreshEvaluationError(
                "a conflicting process appeared before the next node:\n"
                + "\n".join(conflicts)
            )
        print(f"RUN {node.node_id}", flush=True)
        succeeded = True
        if node.node_id == "gate:model_artifacts":
            complete_receipt_gate(
                node=node,
                record=record,
                receipt_path=model_receipt_path,
                state=state,
                state_path=state_path,
            )
        elif node.node_id == "gate:status_only_smoke":
            complete_receipt_gate(
                node=node,
                record=record,
                receipt_path=smoke_receipt_path,
                state=state,
                state_path=state_path,
            )
        elif node.node_id == "reference:shared_block":
            config_path = workspace_path(
                str(
                    manifest["runner_interfaces"]["shared_reference"][
                        "producer_config"
                    ]["path"]
                )
            )
            config = load_json(config_path)
            succeeded = run_phase13_node(
                node=node,
                config_path=config_path,
                config=config,
                record=record,
                state=state,
                state_path=state_path,
                manifest=manifest,
                manifest_path=manifest_path,
                immutable_snapshot=immutable_snapshot,
                python_executable=resolved_python,
            )
        elif node.stage in {
            "reference_noise_check",
            "equal_spp",
            "equal_time",
            "timing_calibration",
            "timing_calibration_repeat",
        }:
            if node.conditional:
                required, fraction = repeat_is_required(
                    node=node,
                    state=state,
                    manifest=manifest,
                )
                record["repeat_trigger_fraction"] = fraction
                if not required:
                    record["status"] = "skipped"
                    record["skip_reason"] = "timing_difference_not_above_10_percent"
                    state["updated_at"] = now_utc()
                    atomic_write_json(state_path, state)
                    print(f"DONE {node.node_id} (conditional skip)", flush=True)
                    executed += 1
                    if args.max_nodes is not None and executed >= args.max_nodes:
                        print(f"max_nodes reached after {executed} new nodes")
                        break
                    continue
            spp_map = (
                load_frozen_spp_map(manifest=manifest, state=state)
                if node.stage == "equal_time"
                else None
            )
            config = generated_config_for_node(
                node=node,
                manifest=manifest,
                runtimes=runtimes,
                spp_map=spp_map,
            )
            config_root = workspace_path(str(outputs["generated_configs"]))
            config_path = config_root / f"{node.node_id.replace(':', '__')}.json"
            write_frozen_config(config_path, config)
            succeeded = run_phase13_node(
                node=node,
                config_path=config_path,
                config=config,
                record=record,
                state=state,
                state_path=state_path,
                manifest=manifest,
                manifest_path=manifest_path,
                immutable_snapshot=immutable_snapshot,
                python_executable=resolved_python,
            )
        elif node.stage == "freeze_spp_map":
            freeze_spp_map(
                manifest=manifest,
                state=state,
                record=record,
                state_path=state_path,
            )
        elif node.stage == "metric_barrier":
            write_metric_unlock(
                manifest=manifest,
                manifest_sha=manifest_sha,
                state=state,
                record=record,
                state_path=state_path,
                dependency_hashes=dependency_hashes,
            )
        elif node.stage == "deferred_metric_finalization":
            succeeded = run_finalizer_node(
                node=node,
                record=record,
                state=state,
                state_path=state_path,
                manifest=manifest,
                manifest_path=manifest_path,
                immutable_snapshot=immutable_snapshot,
                python_executable=resolved_python,
            )
        elif node.stage == "hierarchical_summary":
            succeeded = run_hierarchical_summary_node(
                record=record,
                state=state,
                state_path=state_path,
                manifest=manifest,
                manifest_path=manifest_path,
                immutable_snapshot=immutable_snapshot,
                nodes=nodes,
                python_executable=resolved_python,
            )
        else:
            raise FreshEvaluationError(f"no dispatcher for node {node.node_id}")
        if not succeeded:
            print(f"FAILED {node.node_id}; stopped serial DAG", flush=True)
            return 1
        print(f"DONE {node.node_id}", flush=True)
        executed += 1
        if args.max_nodes is not None and executed >= args.max_nodes:
            print(f"max_nodes reached after {executed} new nodes")
            break

    counts = Counter(
        classify_node_record(state["nodes"][node.node_id]) for node in nodes
    )
    print("final_status=" + ",".join(f"{key}:{counts[key]}" for key in sorted(counts)))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except FreshEvaluationError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
