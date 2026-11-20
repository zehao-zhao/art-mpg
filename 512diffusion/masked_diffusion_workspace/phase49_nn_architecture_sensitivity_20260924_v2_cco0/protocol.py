#!/usr/bin/env python3
"""Immutable task construction and input validation for Phase 49 sensitivity."""

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
PROTOCOL_ID = "phase49_nn_architecture_sensitivity_20260924_v2_cco0"
SOURCE_PROTOCOL_ID = "phase20_equal_render_spp_20260904_v2"
TASK_PREFIX = "p49v2cco0"
STAGES = ("train", "export", "render")
SCENES = (
    "pool", "living", "lamp", "plane", "slabs", "stone", "sphere",
    "water", "ring_and_slab", "pillars", "slab_x1", "slab_x3",
    "pillar_x3", "cornell_glass",
)
CONFIG_IDS = ("h36_d3", "h72_d2", "h72_d3_control", "h72_d4", "h144_d3")
HOST_IDS = ("cco0",)
MODE = "full"
MODEL_SEEDS = (104729, 155921)
FULL_SPP = (12, 24, 36, 48, 60)
PHYSICAL_CPU_IDS = tuple(range(0, 166, 2))
TRAINING_CPU_IDS = PHYSICAL_CPU_IDS[:80]
TRAINING_UNUSED_CPU_IDS = PHYSICAL_CPU_IDS[80:]
SMT_SIBLING_CPU_IDS = tuple(range(1, 166, 2))


class ProtocolError(RuntimeError):
    """Raised when the Phase-49 contract or a bound input is invalid."""


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


def expected_mlp_parameter_count(hidden_dim: int, depth: int) -> int:
    """Return the exact Phase48 MLP parameter count for fixed token/time/K settings."""

    # Embeddings: 11*24 + 3*24. Input projection: 281*h + h.
    # Each residual block: LayerNorm (2*h) and two h-by-h affine layers.
    # Final LayerNorm: 2*h. Length/type/vMF heads: 39*h + 39.
    return 375 + (323 + 4 * depth) * hidden_dim + 2 * depth * hidden_dim * hidden_dim


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
        "analyzer": "analyze_sensitivity.py",
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
        "h36_d3": (36, 3, 20_211),
        "h72_d2": (72, 2, 44_943),
        "h72_d3_control": (72, 3, 55_599),
        "h72_d4": (72, 4, 66_255),
        "h144_d3": (144, 3, 173_031),
    }
    for config_id, want in expected.items():
        entry = configs[config_id]
        got = (
            int(entry.get("hidden_dim", 0)), int(entry.get("depth", 0)),
            int(entry.get("parameter_count", 0)),
        )
        if got != want:
            raise ProtocolError(f"sensitivity configuration changed: {config_id}: {got}")
        if entry["parameter_count"] != expected_mlp_parameter_count(*want[:2]):
            raise ProtocolError(f"derived parameter count changed: {config_id}")
        if (
            entry.get("direction_conditioning_mode") != MODE
            or int(entry.get("epochs", 0)) != 32
            or int(entry.get("vmf_components", 0)) != 4
            or int(entry.get("token_embedding_dim", 0)) != 24
            or int(entry.get("time_embedding_dim", 0)) != 16
            or float(entry.get("dropout", -1)) != 0.1
            or tuple(int(value) for value in entry.get("sample_counts", ())) != FULL_SPP
        ):
            raise ProtocolError(f"fixed sensitivity controls changed: {config_id}")

    hosts = manifest.get("hosts")
    if not isinstance(hosts, dict) or tuple(hosts) != HOST_IDS:
        raise ProtocolError("host allocation changed")
    assigned_scenes: list[str] = []
    for host_id in HOST_IDS:
        entry = hosts[host_id]
        if entry.get("hostname") != "devvm65036.cco0":
            raise ProtocolError("CCO0 hostname changed")
        if tuple(entry.get("configs", ())) != CONFIG_IDS:
            raise ProtocolError(f"host must run every configuration: {host_id}")
        host_scenes = tuple(entry.get("scenes", ()))
        if host_scenes != SCENES:
            raise ProtocolError("CCO0 must run all fourteen scenes in frozen order")
        assigned_scenes.extend(host_scenes)
    if len(assigned_scenes) != len(set(assigned_scenes)) or set(assigned_scenes) != set(SCENES):
        raise ProtocolError("host scene shards must partition the fourteen scenes")

    resources = manifest.get("resources", {})
    if (
        int(resources.get("physical_cores", 0)) != 83
        or int(resources.get("logical_cpus", 0)) != 166
        or resources.get("authorized_cpu_ids") != list(PHYSICAL_CPU_IDS)
        or resources.get("reserved_cpu_ids") != list(SMT_SIBLING_CPU_IDS)
        or int(resources.get("render_workers", 0)) != 83
        or int(resources.get("render_threads", 0)) != 1
        or int(resources.get("training_slots", 0)) != 16
        or resources.get("training_slot_threads") != ([5] * 16)
        or resources.get("training_cpu_ids") != list(TRAINING_CPU_IDS)
        or resources.get("training_unused_authorized_cpu_ids") != list(TRAINING_UNUSED_CPU_IDS)
        or sum(resources["training_slot_threads"]) != 80
        or int(resources.get("export_workers", 0)) != 83
        or int(resources.get("export_threads", 0)) != 1
    ):
        raise ProtocolError("CCO0 83-core CPU allocation contract changed")
    if set(manifest.get("replays", {})) != set(SCENES):
        raise ProtocolError("replay closure must contain all fourteen scenes")
    implementation = manifest.get("implementation", {})
    if implementation.get("required_direction_conditioning_mode") != MODE:
        raise ProtocolError("conditioning-mode contract changed")
    if implementation.get("expected_global_mpg_plugin_sha256") != (
        "577aaa4c95026d8993ad8eae3971c3a40b8a5e51bf3f9cac2fbde1ca6c336035"
    ):
        raise ProtocolError("actual global plugin binding changed")
    overlay = manifest.get("phase48_overlay", {})
    if (
        not isinstance(overlay, dict)
        or not isinstance(overlay.get("manifest_sha256"), str)
        or len(overlay["manifest_sha256"]) != 64
        or not isinstance(overlay.get("files_sha256"), str)
        or len(overlay["files_sha256"]) != 64
    ):
        raise ProtocolError("frozen overlay hashes are missing")

    plan = load_json(PLAN_PATH)
    if plan.get("protocol_id") != PROTOCOL_ID:
        raise ProtocolError("plan protocol identity changed")
    counts = task_counts(manifest)
    if counts["all"] != plan.get("new_totals"):
        raise ProtocolError("manifest and plan task counts differ")
    plan_hosts = plan.get("hosts")
    if not isinstance(plan_hosts, dict) or tuple(plan_hosts) != HOST_IDS:
        raise ProtocolError("plan must contain only the CCO0 host allocation")
    plan_cco0 = plan_hosts["cco0"]
    if (
        tuple(plan_cco0.get("scenes", ())) != SCENES
        or tuple(plan_cco0.get("configs", ())) != CONFIG_IDS
        or {
            "train": int(plan_cco0.get("training_jobs", -1)),
            "export": int(plan_cco0.get("export_jobs", -1)),
            "render": int(plan_cco0.get("render_jobs", -1)),
        } != counts["hosts"]["cco0"]
    ):
        raise ProtocolError("CCO0 manifest and plan allocation differ")
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
                            task_id=f"{TASK_PREFIX}__{config_id}__{scene}__{stage}__t{model_seed}",
                        stage=stage, host_id=host_id, config_id=config_id,
                        scene=scene, model_seed=model_seed,
                    ))
        else:
            for scene in scenes:
                for run_id, model_seed, render_seed in _recorded_run_tuples(manifest):
                    for spp in config["sample_counts"]:
                        tasks.append(Task(
                            task_id=(f"{TASK_PREFIX}__{config_id}__{scene}__{run_id}"
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
        raise ProtocolError("Phase49 requires the ordered-absorbing MLP profile")

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
                f"Phase49 {source_name} binding differs from frozen Phase46"
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

    # Python and MPG-plugin execution comes only from the bound frozen Phase48
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
    value["hidden_dim"] = int(ablation["hidden_dim"])
    value["depth"] = int(ablation["depth"])
    value["vmf_components"] = int(ablation["vmf_components"])
    value["token_embedding_dim"] = int(ablation["token_embedding_dim"])
    value["time_embedding_dim"] = int(ablation["time_embedding_dim"])
    value["dropout"] = float(ablation["dropout"])
    if value.get("backbone_type") != "mlp" or value.get("discrete_diffusion_mode") != "ordered_absorbing":
        raise ProtocolError("architecture sensitivity requires ordered-absorbing MLP")
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
    return tuple((cpu,) for cpu in PHYSICAL_CPU_IDS)


def training_slots() -> tuple[tuple[int, ...], ...]:
    return tuple(
        TRAINING_CPU_IDS[start:start + 5]
        for start in range(0, len(TRAINING_CPU_IDS), 5)
    )


def assert_cpu_contract(slots: Iterable[Iterable[int]], stage: str) -> None:
    normalized = tuple(tuple(int(cpu) for cpu in slot) for slot in slots)
    if stage == "train":
        expected = training_slots()
    elif stage in {"export", "render"}:
        expected = render_slots()
    else:
        raise ProtocolError(f"unknown CPU-contract stage: {stage}")
    flat = [cpu for slot in normalized for cpu in slot]
    if normalized != expected or len(flat) != len(set(flat)):
        raise ProtocolError(f"CPU slot layout changed for {stage}")
    if any(cpu not in PHYSICAL_CPU_IDS for cpu in flat):
        raise ProtocolError(f"{stage} uses an odd SMT sibling or unknown CPU")
    if stage == "train" and tuple(sorted(flat)) != TRAINING_CPU_IDS:
        raise ProtocolError("training must use 16 disjoint five-core slots over even CPUs 0--158")
    if stage in {"export", "render"} and tuple(sorted(flat)) != PHYSICAL_CPU_IDS:
        raise ProtocolError(f"{stage} must use all 83 even physical-core representatives")
