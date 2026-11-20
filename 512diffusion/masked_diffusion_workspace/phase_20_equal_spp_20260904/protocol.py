#!/usr/bin/env python3
"""Validation and deterministic task construction for Phase 20.

The protocol is deliberately separate from the frozen ten-scene studies.  It
adds four scene configurations, a disjoint-seed hyperparameter validation
stage, and an NPM-adapted baseline without changing any earlier manifest.
"""

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
DEFAULT_MANIFEST = PACKAGE_ROOT / "phase20_equal_spp_protocol_v2.json"
PROTOCOL_ID = "phase20_equal_render_spp_20260904_v2"

EXISTING_SCENES = (
    "living",
    "lamp",
    "plane",
    "slabs",
    "stone",
    "pool",
    "sphere",
    "water",
    "ring_and_slab",
    "pillars",
)
NEW_SCENES = ("slab_x1", "slab_x3", "pillar_x3", "cornell_glass")
ALL_SCENES = EXISTING_SCENES + NEW_SCENES
PRIMARY_METHODS = (
    "diffusion_mlp",
    "npm_adapted",
    "mpg_original",
    "pt",
    "ppg_style",
    "sms_star",
)
LEARNED_METHODS = ("diffusion_mlp", "npm_adapted")
STAGES = (
    "replay",
    "train_npm_gate",
    "export_npm_gate",
    "npm_step_validation",
    "references_new",
    "train_diffusion_new",
    "export_diffusion_new",
    "train_npm",
    "export_npm",
    "primary",
)


class ProtocolError(RuntimeError):
    """Raised when the predeclared Phase 20 contract is invalid."""


@dataclass(frozen=True, slots=True)
class Task:
    task_id: str
    stage: str
    kind: str
    scene: str
    method: str | None = None
    render_seed: int | None = None
    model_seed: int | None = None
    control_id: str | None = None

    def payload(self) -> dict[str, Any]:
        return asdict(self)


def json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_json(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ProtocolError(f"JSON object required: {path}")
    return value


def _relative_path(value: object, base: Path, label: str) -> Path:
    if not isinstance(value, str) or not value or Path(value).is_absolute():
        raise ProtocolError(f"{label} must be a nonempty relative path")
    result = (base / value).resolve()
    try:
        result.relative_to(base.resolve())
    except ValueError as exc:
        raise ProtocolError(f"{label} escapes its root: {value}") from exc
    return result


def workspace_path(value: object, label: str = "workspace path") -> Path:
    return _relative_path(value, WORKSPACE_ROOT, label)


def project_path(value: object, label: str = "project path") -> Path:
    return _relative_path(value, PROJECT_ROOT, label)


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


def _positive_int(value: object, label: str) -> int:
    if isinstance(value, bool):
        raise ProtocolError(f"{label} must be a positive integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ProtocolError(f"{label} must be a positive integer") from exc
    if parsed <= 0:
        raise ProtocolError(f"{label} must be a positive integer")
    return parsed


def _finite(value: object, label: str) -> float:
    if isinstance(value, bool):
        raise ProtocolError(f"{label} must be finite")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ProtocolError(f"{label} must be finite") from exc
    if not math.isfinite(parsed):
        raise ProtocolError(f"{label} must be finite")
    return parsed


def load_and_validate_manifest(
    path: Path = DEFAULT_MANIFEST, *, require_files: bool = False
) -> tuple[dict[str, Any], str]:
    manifest = load_json(path)
    if manifest.get("schema_version") != 1:
        raise ProtocolError("schema_version must be 1")
    if manifest.get("protocol_id") != PROTOCOL_ID:
        raise ProtocolError("protocol_id changed")
    if manifest.get("profile_id") != "linux_native":
        raise ProtocolError("Phase 20 is server-only (linux_native)")
    if manifest.get("output_root") != f"results/{PROTOCOL_ID}":
        raise ProtocolError("output_root must be the isolated Phase 20 result root")

    scenes = manifest.get("scenes")
    if not isinstance(scenes, dict) or tuple(scenes) != ALL_SCENES:
        raise ProtocolError("scene order/set must be the frozen 10 + 4 list")
    for scene, entry in scenes.items():
        if not isinstance(entry, dict):
            raise ProtocolError(f"scene entry must be an object: {scene}")
        expected_kind = "existing" if scene in EXISTING_SCENES else "new"
        if entry.get("cohort") != expected_kind:
            raise ProtocolError(f"wrong scene cohort: {scene}")
        if entry.get("job_id") != f"{scene}_mpg":
            raise ProtocolError(f"job_id must be {scene}_mpg")
        scene_path = project_path(entry.get("scene_path"), f"scene {scene}")
        if require_files and not scene_path.is_file():
            raise ProtocolError(f"missing scene XML: {scene_path}")
        if scene in EXISTING_SCENES:
            replay_path = workspace_path(
                entry.get("legacy_replay_path"), f"legacy replay {scene}"
            )
            digest = entry.get("legacy_replay_sha256")
            if not isinstance(digest, str) or len(digest) != 64:
                raise ProtocolError(f"legacy replay hash is invalid: {scene}")
            if require_files and (
                not replay_path.is_file() or sha256_file(replay_path) != digest
            ):
                raise ProtocolError(f"legacy replay changed or is missing: {scene}")
        elif "legacy_replay_path" in entry or "legacy_replay_sha256" in entry:
            raise ProtocolError(f"new scene must not claim a legacy replay: {scene}")

    primary = manifest.get("primary")
    if not isinstance(primary, dict):
        raise ProtocolError("primary block is missing")
    if _positive_int(primary.get("sample_count"), "primary.sample_count") != 64:
        raise ProtocolError("primary evaluation must use exactly 64 render SPP")
    seeds = tuple(int(value) for value in primary.get("render_seeds", []))
    if len(seeds) != 5 or len(set(seeds)) != 5 or any(seed < 0 for seed in seeds):
        raise ProtocolError("primary must contain five distinct nonnegative render seeds")
    methods = tuple(primary.get("methods", []))
    if methods != PRIMARY_METHODS or any("dit" in method.lower() for method in methods):
        raise ProtocolError("primary methods changed or DiT was reintroduced")

    selection_path = workspace_path(manifest.get("phase19_selection"), "Phase 19 selection")
    expected_selection_sha = manifest.get("phase19_selection_sha256")
    if not isinstance(expected_selection_sha, str) or len(expected_selection_sha) != 64:
        raise ProtocolError("Phase 19 selection SHA-256 is invalid")
    if require_files:
        selection = load_json(selection_path)
        if sha256_file(selection_path) != expected_selection_sha:
            raise ProtocolError("Phase 19 final-v3 selection hash changed")
        selected = selection.get("selected")
        if (
            not isinstance(selected, dict)
            or selected.get("eligible") is not True
            or float(selected.get("bounce_alpha", -1)) != 0.0
            or float(selected.get("kappa_scale", -1)) != 1.0
        ):
            raise ProtocolError("Phase 19 final-v3 did not select alpha=0/kappa=1")

    model_seeds = tuple(int(value) for value in manifest.get("model_seeds", []))
    if model_seeds != (104729, 130363, 155921):
        raise ProtocolError("the primary model-seed replication must remain 3 seeds")
    replay = manifest.get("replay")
    if not isinstance(replay, dict) or len(replay.get("seed_ids", [])) < 3:
        raise ProtocolError("fresh replay requires at least three independent seeds")
    if float(replay.get("timeout_seconds", 0.0)) <= 0.0:
        raise ProtocolError("replay timeout must be positive")
    reference = manifest.get("reference")
    if not isinstance(reference, dict) or int(reference.get("sample_count", 0)) < 4096:
        raise ProtocolError("new-scene references must be high-SPP")

    npm = manifest.get("npm_training")
    if not isinstance(npm, dict):
        raise ProtocolError("npm_training block is missing")
    model = npm.get("model")
    if not isinstance(model, dict) or model.get("grid_resolutions") != [8, 11, 16, 22, 31, 44, 61, 86]:
        raise ProtocolError("NPM grid pyramid no longer matches the configured profile")
    if (
        model.get("features_per_level") != 4
        or model.get("sh_degree") != 2
        or model.get("hidden_dim") != 64
        or model.get("hidden_layers") != 3
        or model.get("vmf_components") != 8
        or model.get("descriptor_policy_id")
        != "geometric_length_trace_inferred_type_v1"
    ):
        raise ProtocolError("NPM architecture/policy no longer matches the configured profile")
    npm_gate = manifest.get("npm_render_gate")
    if not isinstance(npm_gate, dict):
        raise ProtocolError("npm_render_gate block is missing")
    if npm_gate.get("anchor_scenes") != ["pillars", "slabs"]:
        raise ProtocolError("NPM renderer gate anchors changed")
    if npm_gate.get("candidate_steps") != [100, 300, 1000]:
        raise ProtocolError("NPM renderer gate step candidates changed")
    if _positive_int(npm_gate.get("sample_count"), "npm gate SPP") != 64:
        raise ProtocolError("NPM renderer gate must use 64 SPP")
    npm_gate_seeds = tuple(int(value) for value in npm_gate.get("render_seeds", []))
    if len(npm_gate_seeds) != 5 or set(npm_gate_seeds) & set(seeds):
        raise ProtocolError("NPM renderer-gate seeds must be five and evaluation-disjoint")

    for key in (
        "phase13_path",
        "phase09_path",
        "diffusion_train_path",
        "runtime_export_path",
        "npm_train_path",
        "npm_step_selection_receipt",
        "phase19_selection",
        "legacy_reference_manifest",
        "legacy_diffusion_state",
    ):
        target = workspace_path(manifest.get(key), key)
        if require_files and not target.is_file():
            raise ProtocolError(f"missing required input: {target}")
    return manifest, sha256_file(path)


def task_sort_key(task: Task) -> str:
    return hashlib.sha256(task.task_id.encode("utf-8")).hexdigest()


def build_tasks(manifest: dict[str, Any], stage: str) -> tuple[Task, ...]:
    if stage not in STAGES:
        raise ProtocolError(f"unknown stage: {stage}")
    tasks: list[Task] = []
    model_seeds = tuple(int(value) for value in manifest["model_seeds"])
    if stage == "replay":
        tasks.extend(
            Task(
                task_id=f"replay__{scene}", stage=stage, kind="replay", scene=scene
            )
            for scene in NEW_SCENES
        )
    elif stage == "train_npm_gate":
        for scene in manifest["npm_render_gate"]["anchor_scenes"]:
            for steps in manifest["npm_render_gate"]["candidate_steps"]:
                tasks.append(Task(
                    task_id=f"train_npm_gate__{scene}__steps{steps}",
                    stage=stage,
                    kind="train_npm",
                    scene=scene,
                    method="npm_adapted",
                    model_seed=int(manifest["npm_render_gate"]["training_seed"]),
                    control_id=f"steps{steps}",
                ))
    elif stage == "export_npm_gate":
        for scene in manifest["npm_render_gate"]["anchor_scenes"]:
            for steps in manifest["npm_render_gate"]["candidate_steps"]:
                tasks.append(Task(
                    task_id=f"export_npm_gate__{scene}__steps{steps}",
                    stage=stage,
                    kind="export_npm_gate",
                    scene=scene,
                    method="npm_adapted",
                    model_seed=int(manifest["npm_render_gate"]["training_seed"]),
                    control_id=f"steps{steps}",
                ))
    elif stage == "npm_step_validation":
        for scene in manifest["npm_render_gate"]["anchor_scenes"]:
            for steps in manifest["npm_render_gate"]["candidate_steps"]:
                for seed in manifest["npm_render_gate"]["render_seeds"]:
                    tasks.append(Task(
                        task_id=f"npm_gate__{scene}__steps{steps}__r{seed}",
                        stage=stage,
                        kind="render",
                        scene=scene,
                        method="npm_adapted",
                        render_seed=int(seed),
                        model_seed=int(manifest["npm_render_gate"]["training_seed"]),
                        control_id=f"steps{steps}",
                    ))
    elif stage == "references_new":
        seed = int(manifest["reference"]["seed"])
        tasks.extend(
            Task(
                task_id=f"reference__{scene}__r{seed}",
                stage=stage,
                kind="reference",
                scene=scene,
                render_seed=seed,
            )
            for scene in NEW_SCENES
        )
    elif stage in {"train_diffusion_new", "export_diffusion_new"}:
        kind = "train_diffusion" if stage.startswith("train") else "export_diffusion"
        for scene in NEW_SCENES:
            for model_seed in model_seeds:
                tasks.append(Task(
                    task_id=f"{kind}__{scene}__t{model_seed}",
                    stage=stage,
                    kind=kind,
                    scene=scene,
                    method="diffusion_mlp",
                    model_seed=model_seed,
                ))
    elif stage in {"train_npm", "export_npm"}:
        kind = "train_npm" if stage == "train_npm" else "export_npm"
        for scene in ALL_SCENES:
            for model_seed in model_seeds:
                tasks.append(Task(
                    task_id=f"{kind}__{scene}__t{model_seed}",
                    stage=stage,
                    kind=kind,
                    scene=scene,
                    method="npm_adapted",
                    model_seed=model_seed,
                ))
    elif stage == "primary":
        for scene in ALL_SCENES:
            for render_seed in manifest["primary"]["render_seeds"]:
                for method in PRIMARY_METHODS:
                    if method in LEARNED_METHODS:
                        for model_seed in model_seeds:
                            tasks.append(Task(
                                task_id=(
                                    f"primary__{scene}__{method}__t{model_seed}"
                                    f"__r{render_seed}"
                                ),
                                stage=stage,
                                kind="render",
                                scene=scene,
                                method=method,
                                render_seed=int(render_seed),
                                model_seed=model_seed,
                            ))
                    else:
                        tasks.append(Task(
                            task_id=f"primary__{scene}__{method}__r{render_seed}",
                            stage=stage,
                            kind="render",
                            scene=scene,
                            method=method,
                            render_seed=int(render_seed),
                        ))
    return tuple(sorted(tasks, key=task_sort_key))


def task_counts(manifest: dict[str, Any]) -> dict[str, int]:
    return {stage: len(build_tasks(manifest, stage)) for stage in STAGES}
