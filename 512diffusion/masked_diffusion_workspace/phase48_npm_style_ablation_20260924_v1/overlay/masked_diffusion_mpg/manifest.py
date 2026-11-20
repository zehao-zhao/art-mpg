from __future__ import annotations

import json
import xml.etree.ElementTree as et
from dataclasses import dataclass
from pathlib import Path


ASSET_SUFFIXES = {
    ".bmp",
    ".dds",
    ".exr",
    ".hdr",
    ".jpeg",
    ".jpg",
    ".mtl",
    ".obj",
    ".pfm",
    ".ply",
    ".png",
    ".serialized",
    ".tga",
}


@dataclass(slots=True, frozen=True)
class RenderJob:
    """Rendered baseline scene description pulled from manifests."""

    job_id: str
    group: str
    method: str
    variant: str
    scene_path: str
    expected_integrator: str


@dataclass(slots=True, frozen=True)
class TrainingJob:
    """Training harness entry mirrored from manifests."""

    job_id: str
    group: str
    script_path: str
    args: tuple[str, ...]


@dataclass(slots=True, frozen=True)
class BaselineManifest:
    """Full render + training manifests used by experiment scripts."""

    render_jobs: tuple[RenderJob, ...]
    training_jobs: tuple[TrainingJob, ...]


def load_baseline_manifest(config_path: Path) -> BaselineManifest:
    """Hydrate immutable BaselineManifest objects from disk JSON.

    Args:
        config_path: Path to authored `baseline_manifest.json`.

    Returns:
        Parsed dataclass tying render/train tuples together.
    """

    payload = json.loads(config_path.read_text(encoding="utf-8"))
    render_jobs = tuple(
        RenderJob(
            job_id=item["id"],
            group=item["group"],
            method=item["method"],
            variant=item["variant"],
            scene_path=item["scene_path"],
            expected_integrator=item["expected_integrator"],
        )
        for item in payload["render_jobs"]
    )
    training_jobs = tuple(
        TrainingJob(
            job_id=item["id"],
            group=item["group"],
            script_path=item["script_path"],
            args=tuple(item["args"]),
        )
        for item in payload["training_jobs"]
    )
    return BaselineManifest(render_jobs=render_jobs, training_jobs=training_jobs)


def get_parent_map(root: et.Element) -> dict[et.Element, et.Element]:
    """Map XML elements to immediate parents."""

    return {child: parent for parent in root.iter() for child in parent}


def get_integrator_type(scene_path: Path) -> str:
    """Return integrator `type=` declared in an XML scene file.

    Args:
        scene_path: Scene filesystem path Mitsuba parses.

    Returns:
        Empty string when no integrator child exists yet.
    """

    root = et.parse(scene_path).getroot()
    integrator = root.find("integrator")
    if integrator is None:
        return ""
    return integrator.attrib.get("type", "")


def collect_missing_assets(scene_path: Path) -> list[str]:
    """List referenced filenames with supported extensions that disk lacks."""

    root = et.parse(scene_path).getroot()
    parent_map = get_parent_map(root)
    missing_assets: list[str] = []
    for elem in root.iter("string"):
        if elem.attrib.get("name") != "filename":
            continue
        value = elem.attrib.get("value", "")
        if not value or value.startswith("$") or "://" in value:
            continue
        if Path(value).suffix.lower() not in ASSET_SUFFIXES:
            continue
        parent = parent_map.get(elem)
        if parent is not None and parent.tag == "film":
            continue
        candidate = (scene_path.parent / value).resolve()
        if not candidate.exists():
            missing_assets.append(value)
    return sorted(set(missing_assets))


def validate_manifest_paths(
    manifest: BaselineManifest, project_root: Path
) -> list[str]:
    """Enumerate manifest paths that disappear relative to ``project_root``."""

    missing_paths: list[str] = []
    for job in manifest.render_jobs:
        if not (project_root / job.scene_path).exists():
            missing_paths.append(job.scene_path)
    for job in manifest.training_jobs:
        if not (project_root / job.script_path).exists():
            missing_paths.append(job.script_path)
    return missing_paths
