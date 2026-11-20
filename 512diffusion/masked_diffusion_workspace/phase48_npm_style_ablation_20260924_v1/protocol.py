#!/usr/bin/env python3
"""Immutable task construction and input validation for Phase 48."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


PACKAGE_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = PACKAGE_ROOT.parent
PROJECT_ROOT = WORKSPACE_ROOT.parent
MANIFEST_PATH = PACKAGE_ROOT / "manifest.json"
PLAN_PATH = PACKAGE_ROOT / "plan.json"
PROTOCOL_ID = "phase48_npm_style_ablation_20260924_v1"
SOURCE_PROTOCOL_ID = "phase20_equal_render_spp_20260904_v2"
STAGES = ("train", "export", "render")
SCENES = (
    "pool", "living", "lamp", "plane", "slabs", "stone", "sphere",
    "water", "ring_and_slab", "pillars", "slab_x1", "slab_x3",
    "pillar_x3", "cornell_glass",
)
CONFIG_IDS = (
    "length_only_e32", "full_e8", "no_topology_e32", "full_e16",
    "full_e32", "no_topology_e8", "no_topology_e16",
)
HOST_IDS = ("lla0", "scu0", "maz0")
MODES = ("full", "length_only", "no_topology")
MODEL_SEEDS = (104729, 155921)
FULL_SPP = (12, 24, 36, 48, 60)


class ProtocolError(RuntimeError):
    """Raised when the Phase-48 contract or a bound input is invalid."""


@dataclass(frozen=True, slots=True)
class Task:
    task_id: str
    stage: str
    host_id: str
    config_id: str
    scene: str
    model_seed: int
    run_id: str | None = None
    render_seed: int | None = None
    sample_count: int | None = None

    def payload(self) -> dict[str, Any]:
        return asdict(self)


def json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def canonical_sha256(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def sha256_file(path: Path) -> str:
    if not path.is_file() or path.is_symlink():
        raise ProtocolError(f"regular non-symlink file required: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"cannot read JSON object {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ProtocolError(f"JSON object required: {path}")
    return value


def _contained(relative: object, root: Path, label: str) -> Path:
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ProtocolError(f"{label} must be a nonempty relative path")
    result = (root / relative).resolve()
    try:
        result.relative_to(root.resolve())
    except ValueError as exc:
        raise ProtocolError(f"{label} escapes its root: {relative}") from exc
    return result


def workspace_path(relative: object, label: str = "workspace path") -> Path:
    return _contained(relative, WORKSPACE_ROOT, label)


def project_path(relative: object, label: str = "project path") -> Path:
    return _contained(relative, PROJECT_ROOT, label)


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


def _source_path(manifest: dict[str, Any], name: str) -> Path:
    entry = manifest.get("source", {}).get(name)
    if not isinstance(entry, dict):
        raise ProtocolError(f"missing source binding: {name}")
    return workspace_path(entry.get("path"), f"source.{name}.path")


def _verify_bound(path: Path, expected: object, label: str) -> None:
    if not isinstance(expected, str) or len(expected) != 64:
        raise ProtocolError(f"invalid SHA-256 binding for {label}")
    if sha256_file(path) != expected:
        raise ProtocolError(f"bound input changed: {label}: {path}")


def scene_tree_digest(scene_xml: Path) -> dict[str, Any]:
    root = scene_xml.parent
    rows = {
        path.relative_to(root).as_posix(): sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and not path.is_symlink()
        and path.name != ".DS_Store" and ".phase" not in path.name
    }
    return {"file_count": len(rows), "tree_sha256": canonical_sha256(rows)}


def _recorded_run_tuples(manifest: dict[str, Any]) -> tuple[tuple[str, int, int], ...]:
    runs = manifest.get("recorded_runs")
    if not isinstance(runs, list):
        raise ProtocolError("recorded_runs must be a list")
    result: list[tuple[str, int, int]] = []
    for index, row in enumerate(runs, start=1):
        if not isinstance(row, dict):
            raise ProtocolError("recorded run must be an object")
        result.append((
            str(row.get("run_id")),
            int(row.get("model_seed", -1)), int(row.get("render_seed", -1)),
        ))
        if result[-1][0] != f"D{index:02d}":
            raise ProtocolError("recorded run labels must be D01--D10 in order")
    if len(result) != 10 or len({row[0] for row in result}) != 10:
        raise ProtocolError("exactly ten distinct recorded runs are required")
    if {row[1] for row in result} != set(MODEL_SEEDS):
        raise ProtocolError("recorded runs must exercise both declared model seeds")
    if len({row[2] for row in result}) != 10:
        raise ProtocolError("render seeds must be distinct")
    return tuple(result)


def load_manifest(path: Path = MANIFEST_PATH) -> dict[str, Any]:
    manifest = load_json(path)
    if manifest.get("schema_version") != 1 or manifest.get("protocol_id") != PROTOCOL_ID:
        raise ProtocolError("manifest schema/protocol identity changed")
    if manifest.get("output_root") != f"results/{PROTOCOL_ID}":
        raise ProtocolError("output_root must remain isolated")
    package_code = manifest.get("package_code")
    expected_code = {
        "protocol": "protocol.py",
        "worker": "worker.py",
        "controller": "run_ablation.py",
        "overlay_runtime": "overlay_runtime.py",
        "bootstrap": "bootstrap_overlay.py",
        "plan": "plan.json",
    }
    if not isinstance(package_code, dict) or set(package_code) != set(expected_code):
        raise ProtocolError("package code inventory changed")
    for name, relative in expected_code.items():
        entry = package_code[name]
        if not isinstance(entry, dict) or entry.get("path") != relative:
            raise ProtocolError(f"package code path changed: {name}")
        if sha256_file(PACKAGE_ROOT / relative) != entry.get("sha256"):
            raise ProtocolError(f"package code hash changed: {name}")
    if tuple(manifest.get("scenes", ())) != SCENES:
        raise ProtocolError("scene order/set changed")
    if tuple(int(value) for value in manifest.get("model_seeds", ())) != MODEL_SEEDS:
        raise ProtocolError("model seeds must be exactly 104729 and 155921")
    _recorded_run_tuples(manifest)
    interpretation = manifest.get("recorded_runs_interpretation")
    if not isinstance(interpretation, str) or not interpretation.strip():
        raise ProtocolError("recorded_runs_interpretation must be a nonempty description")

    configs = manifest.get("configs")
    if not isinstance(configs, dict) or tuple(configs) != CONFIG_IDS:
        raise ProtocolError("configuration order/set changed")
    expected = {
        "length_only_e32": ("length_only", 32, FULL_SPP),
        "full_e8": ("full", 8, (60,)),
        "no_topology_e32": ("no_topology", 32, FULL_SPP),
        "full_e16": ("full", 16, (60,)),
        "full_e32": ("full", 32, FULL_SPP),
        "no_topology_e8": ("no_topology", 8, (60,)),
        "no_topology_e16": ("no_topology", 16, (60,)),
    }
    for config_id, want in expected.items():
        entry = configs[config_id]
        got = (
            entry.get("direction_conditioning_mode"), int(entry.get("epochs", 0)),
            tuple(int(value) for value in entry.get("sample_counts", ())),
        )
        if got != want:
            raise ProtocolError(f"ablation configuration changed: {config_id}: {got}")

    hosts = manifest.get("hosts")
    if not isinstance(hosts, dict) or tuple(hosts) != HOST_IDS:
        raise ProtocolError("host allocation changed")
    assigned_scenes: list[str] = []
    for host_id in HOST_IDS:
        entry = hosts[host_id]
        if not isinstance(entry.get("hostname"), str):
            raise ProtocolError(f"hostname missing: {host_id}")
        if tuple(entry.get("configs", ())) != CONFIG_IDS:
            raise ProtocolError(f"host must run every configuration: {host_id}")
        host_scenes = tuple(entry.get("scenes", ()))
        if not host_scenes or any(scene not in SCENES for scene in host_scenes):
            raise ProtocolError(f"invalid scene shard: {host_id}")
        assigned_scenes.extend(host_scenes)
    if len(assigned_scenes) != len(set(assigned_scenes)) or set(assigned_scenes) != set(SCENES):
        raise ProtocolError("host scene shards must partition the fourteen scenes")

    resources = manifest.get("resources", {})
    if (
        int(resources.get("physical_cores", 0)) != 96
        or resources.get("reserved_cpu_ids") != [95]
        or int(resources.get("render_workers", 0)) != 95
        or int(resources.get("render_threads", 0)) != 1
        or int(resources.get("training_slots", 0)) != 19
        or resources.get("training_slot_threads") != ([5] * 19)
        or sum(resources["training_slot_threads"]) != 95
        or int(resources.get("export_workers", 0)) != 95
        or int(resources.get("export_threads", 0)) != 1
    ):
        raise ProtocolError("95-of-96 CPU allocation contract changed")
    if set(manifest.get("replays", {})) != set(SCENES):
        raise ProtocolError("replay closure must contain all fourteen scenes")
    if manifest.get("implementation", {}).get("required_direction_conditioning_modes") != list(MODES):
        raise ProtocolError("conditioning-mode contract changed")
    overlay = manifest.get("experiment_overlay", {})
    if (
        not isinstance(overlay, dict)
        or not isinstance(overlay.get("inventory_sha256"), str)
        or len(overlay["inventory_sha256"]) != 64
        or not isinstance(overlay.get("files_sha256"), str)
        or len(overlay["files_sha256"]) != 64
    ):
        raise ProtocolError("frozen overlay hashes are missing")

    plan = load_json(PLAN_PATH)
    if plan.get("protocol_id") != PROTOCOL_ID:
        raise ProtocolError("plan protocol identity changed")
    if task_counts(manifest)["all"] != plan.get("new_totals"):
        raise ProtocolError("manifest and plan task counts differ")
    return manifest


def build_tasks(manifest: dict[str, Any], host_id: str, stage: str) -> tuple[Task, ...]:
    if host_id not in HOST_IDS or stage not in STAGES:
        raise ProtocolError(f"invalid host/stage: {host_id}/{stage}")
    tasks: list[Task] = []
    config_ids = tuple(manifest["hosts"][host_id]["configs"])
    scenes = tuple(manifest["hosts"][host_id]["scenes"])
    for config_id in config_ids:
        config = manifest["configs"][config_id]
        if stage in {"train", "export"}:
            for scene in scenes:
                for model_seed in MODEL_SEEDS:
                    tasks.append(Task(
                        task_id=f"p48__{config_id}__{scene}__{stage}__t{model_seed}",
                        stage=stage, host_id=host_id, config_id=config_id,
                        scene=scene, model_seed=model_seed,
                    ))
        else:
            for scene in scenes:
                for run_id, model_seed, render_seed in _recorded_run_tuples(manifest):
                    for spp in config["sample_counts"]:
                        tasks.append(Task(
                            task_id=(f"p48__{config_id}__{scene}__{run_id}"
                                     f"__spp{spp}__t{model_seed}__r{render_seed}"),
                            stage=stage, host_id=host_id, config_id=config_id,
                            scene=scene, model_seed=model_seed, run_id=run_id,
                            render_seed=render_seed,
                            sample_count=int(spp),
                        ))
    return tuple(sorted(tasks, key=lambda task: hashlib.sha256(task.task_id.encode()).hexdigest()))


def task_counts(manifest: dict[str, Any]) -> dict[str, Any]:
    hosts: dict[str, dict[str, int]] = {}
    for host_id in HOST_IDS:
        hosts[host_id] = {stage: len(build_tasks(manifest, host_id, stage)) for stage in STAGES}
    return {
        "hosts": hosts,
        "all": {
            "training_jobs": sum(value["train"] for value in hosts.values()),
            "export_jobs": sum(value["export"] for value in hosts.values()),
            "render_jobs": sum(value["render"] for value in hosts.values()),
        },
    }


def verify_source_inputs(manifest: dict[str, Any], *, require_payloads: bool) -> dict[str, Any]:
    snapshot: dict[str, Any] = {"source": {}, "implementation": {}, "replays": {}, "scenes": {}, "references": {}}
    for name in (
        "phase20_manifest", "phase46_manifest", "phase46_bindings",
        "phase13_runner", "runtime_profiles", "phase20_reference_registry",
    ):
        entry = manifest["source"][name]
        path = _source_path(manifest, name)
        if require_payloads:
            _verify_bound(path, entry.get("sha256"), f"source.{name}")
        snapshot["source"][name] = {"path": entry["path"], "sha256": entry["sha256"]}

    phase20 = load_json(_source_path(manifest, "phase20_manifest"))
    if phase20.get("protocol_id") != SOURCE_PROTOCOL_ID:
        raise ProtocolError("Phase20 source protocol changed")
    base_training = phase20.get("diffusion_training")
    if not isinstance(base_training, dict):
        raise ProtocolError("Phase20 diffusion training configuration is missing")
    if base_training.get("backbone_type") != "mlp" or base_training.get("discrete_diffusion_mode") != "ordered_absorbing":
        raise ProtocolError("Phase48 requires the ordered-absorbing MLP profile")

    phase46 = load_json(_source_path(manifest, "phase46_manifest"))
    bindings = load_json(_source_path(manifest, "phase46_bindings"))
    if phase46.get("scene_order") != list(SCENES):
        raise ProtocolError("Phase46 scene order changed")
    if set(bindings.get("scenes", {})) != set(SCENES) or set(bindings.get("references", {})) != set(SCENES):
        raise ProtocolError("Phase46 scene/reference closure changed")
    for source_name, binding_name in (
        ("phase13_runner", "phase13_runner"),
        ("runtime_profiles", "runtime_profiles"),
    ):
        source_entry = manifest["source"].get(source_name)
        bound_entry = bindings.get("frozen_code", {}).get(binding_name)
        if source_entry != bound_entry:
            raise ProtocolError(
                f"Phase48 {source_name} binding differs from frozen Phase46"
            )
    registry = load_json(_source_path(manifest, "phase20_reference_registry"))
    if set(registry.get("references", {})) != set(SCENES):
        raise ProtocolError("Phase20 reference registry closure changed")

    for scene in SCENES:
        replay = manifest["replays"][scene]
        replay_path = workspace_path(replay.get("path"), f"replay {scene}")
        if require_payloads:
            _verify_bound(replay_path, replay.get("sha256"), f"replay {scene}")
        snapshot["replays"][scene] = dict(replay)

        scene_entry = bindings["scenes"][scene]
        scene_path = project_path(scene_entry.get("scene_path"), f"scene {scene}")
        if require_payloads:
            _verify_bound(scene_path, scene_entry.get("scene_xml_sha256"), f"scene {scene}")
            tree = scene_tree_digest(scene_path)
            if (
                tree["file_count"] != int(scene_entry.get("scene_tree_file_count", -1))
                or tree["tree_sha256"] != scene_entry.get("scene_tree_sha256")
            ):
                raise ProtocolError(f"Phase46 scene dependency tree changed: {scene}")
        snapshot["scenes"][scene] = {
            "path": scene_entry["scene_path"],
            "sha256": scene_entry["scene_xml_sha256"],
            "file_count": scene_entry["scene_tree_file_count"],
            "tree_sha256": scene_entry["scene_tree_sha256"],
        }

        reference = bindings["references"][scene]
        registry_entry = registry["references"][scene]
        if reference.get("sha256") != registry_entry.get("exr_sha256"):
            raise ProtocolError(f"Phase20/Phase46 reference mismatch: {scene}")
        reference_path = workspace_path(reference.get("path"), f"reference {scene}")
        if require_payloads:
            _verify_bound(reference_path, reference.get("sha256"), f"reference {scene}")
        snapshot["references"][scene] = dict(reference)

    # Python and MPG-plugin execution comes only from the frozen Phase48
    # overlay. The base Mitsuba executable remains external and must match the
    # Phase46 binding; production Python/header/plugin files are deliberately
    # excluded from the cross-host execution snapshot.
    executable = project_path(manifest["implementation"]["mitsuba_executable"])
    expected_executable = bindings["frozen_runtime"]["mitsuba"]["sha256"]
    if require_payloads:
        _verify_bound(executable, expected_executable, "base Mitsuba executable")
    snapshot["implementation"]["mitsuba_executable"] = {
        "path": manifest["implementation"]["mitsuba_executable"],
        "sha256": expected_executable,
    }
    return snapshot


def training_config(manifest: dict[str, Any], config_id: str) -> dict[str, Any]:
    phase20 = load_json(_source_path(manifest, "phase20_manifest"))
    value = dict(phase20["diffusion_training"])
    ablation = manifest["configs"][config_id]
    value["epochs"] = int(ablation["epochs"])
    value["direction_conditioning_mode"] = ablation["direction_conditioning_mode"]
    if value.get("backbone_type") != "mlp" or value.get("discrete_diffusion_mode") != "ordered_absorbing":
        raise ProtocolError("non-full modes require ordered-absorbing MLP")
    return value


def reference_entry(manifest: dict[str, Any], scene: str) -> dict[str, Any]:
    bindings = load_json(_source_path(manifest, "phase46_bindings"))
    entry = bindings.get("references", {}).get(scene)
    if not isinstance(entry, dict):
        raise ProtocolError(f"reference binding missing: {scene}")
    return dict(entry)


def scene_entry(manifest: dict[str, Any], scene: str) -> dict[str, Any]:
    bindings = load_json(_source_path(manifest, "phase46_bindings"))
    entry = bindings.get("scenes", {}).get(scene)
    if not isinstance(entry, dict):
        raise ProtocolError(f"scene binding missing: {scene}")
    return dict(entry)


def render_slots() -> tuple[tuple[int, ...], ...]:
    return tuple((cpu,) for cpu in range(95))


def training_slots() -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(range(start, start + 5)) for start in range(0, 95, 5))


def assert_cpu_contract(slots: Iterable[Iterable[int]]) -> None:
    flat = [cpu for slot in slots for cpu in slot]
    if sorted(flat) != list(range(95)) or len(flat) != len(set(flat)) or 95 in flat:
        raise ProtocolError("CPU slot layout must use CPUs 0--94 exactly once and reserve CPU95")
