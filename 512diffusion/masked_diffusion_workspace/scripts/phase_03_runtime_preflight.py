from __future__ import annotations

import json
import logging
import sys
import xml.etree.ElementTree as et
from dataclasses import asdict, dataclass
from pathlib import Path


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = WORKSPACE_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from masked_diffusion_mpg.runtime import (  # noqa: E402
    ResolvedRuntimeProfile,
    detect_host_runtime,
    load_runtime_profiles,
    resolve_runtime_profile,
)


LOGGER = logging.getLogger("phase_03_runtime_preflight")
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


@dataclass(slots=True)
class RuntimeInfo:
    """Host capability snapshot for Mitsuba runtime checks."""

    profile_id: str
    platform_name: str
    python_version: str
    mitsuba_binary: str
    binary_format: str
    render_runtime_supported: bool
    python_binding_ready: bool
    python_binding_reason: str
    reason: str


@dataclass(slots=True)
class JobPreflight:
    """Structured readiness rollup for baseline jobs."""

    job_id: str
    group: str
    job_type: str
    ready: bool
    blocked_by_runtime: bool
    expected_integrator: str | None
    actual_integrator: str | None
    missing_assets: list[str]
    detail: str


def configure_logging() -> None:
    """Initialize logging."""

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def select_primary_runtime_profile(
    profiles: tuple[ResolvedRuntimeProfile, ...],
    *,
    host_platform: str,
) -> ResolvedRuntimeProfile:
    """Pick the canonical runtime profile for the detected host OS.

    Args:
        profiles: Resolved runtime tuples returned by loaders.
        host_platform: Normalized OS string from detectors.

    Returns:
        Highest-priority ResolvedRuntimeProfile for this machine.

    Raises:
        RuntimeError: Raised when manifests produced zero usable profiles.
    """

    if not profiles:
        raise RuntimeError("missing runtime profiles")

    preferred_profile_ids = {
        "windows": "windows_native",
        "linux": "linux_native",
        "darwin": "darwin_native",
    }
    preferred_id = preferred_profile_ids.get(host_platform, "")
    for profile in profiles:
        if profile.profile_id == preferred_id:
            return profile
    return profiles[0]


def get_parent_map(root: et.Element) -> dict[et.Element, et.Element]:
    """Map each XML node back to direct parent pointers."""

    return {child: parent for parent in root.iter() for child in parent}


def get_integrator_type(scene_path: Path) -> str:
    """Read integrator XML type attribute from disk."""

    root = et.parse(scene_path).getroot()
    integrator = root.find("integrator")
    if integrator is None:
        return ""
    return integrator.attrib.get("type", "")


def collect_missing_assets(scene_path: Path) -> list[str]:
    """List referenced assets that failed to resolve on disk.

    Only typical texture/mesh filenames are consulted; `<film>` outputs are ignored.
    """

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


def render_preflight(
    project_root: Path,
    item: dict[str, str],
    runtime_info: RuntimeInfo,
) -> JobPreflight:
    """Run render-specific checks (integrator parity + deps)."""

    scene_path = project_root / item["scene_path"]
    actual_integrator = get_integrator_type(scene_path)
    missing_assets = collect_missing_assets(scene_path)
    integrator_ok = actual_integrator == item["expected_integrator"]
    ready = (
        integrator_ok and not missing_assets and runtime_info.render_runtime_supported
    )
    blocked_by_runtime = (
        integrator_ok
        and not missing_assets
        and not runtime_info.render_runtime_supported
    )

    if ready:
        detail = "scene ready"
    elif blocked_by_runtime:
        detail = runtime_info.reason
    elif missing_assets:
        detail = f"missing assets: {missing_assets}"
    else:
        detail = f"integrator mismatch: {actual_integrator}"

    return JobPreflight(
        job_id=item["id"],
        group=item["group"],
        job_type="render",
        ready=ready,
        blocked_by_runtime=blocked_by_runtime,
        expected_integrator=item["expected_integrator"],
        actual_integrator=actual_integrator,
        missing_assets=missing_assets,
        detail=detail,
    )


def training_preflight(project_root: Path, item: dict[str, object]) -> JobPreflight:
    """Ensure training invocation targets exist locally."""

    script_path = project_root / str(item["script_path"])
    ready = script_path.exists()
    detail = str(script_path) if ready else f"missing script: {script_path}"
    return JobPreflight(
        job_id=str(item["id"]),
        group=str(item["group"]),
        job_type="training",
        ready=ready,
        blocked_by_runtime=False,
        expected_integrator=None,
        actual_integrator=None,
        missing_assets=[],
        detail=detail,
    )


def write_markdown_report(
    output_path: Path,
    runtime_info: RuntimeInfo,
    jobs: list[JobPreflight],
) -> None:
    """Emit Markdown narration for reviewers."""

    lines = [
        "# Phase 03 Runtime Preflight",
        "",
        "## Runtime",
        "",
        f"- profile_id: `{runtime_info.profile_id}`",
        f"- platform: `{runtime_info.platform_name}`",
        f"- python: `{runtime_info.python_version}`",
        f"- mitsuba_binary: `{runtime_info.mitsuba_binary}`",
        f"- binary_format: `{runtime_info.binary_format}`",
        f"- render_runtime_supported: `{runtime_info.render_runtime_supported}`",
        f"- python_binding_ready: `{runtime_info.python_binding_ready}`",
        f"- python_binding_reason: `{runtime_info.python_binding_reason}`",
        f"- reason: `{runtime_info.reason}`",
        "",
        "## Jobs",
        "",
    ]
    for job in jobs:
        lines.extend(
            [
                f"### {job.job_id}",
                "",
                f"- type: `{job.job_type}`",
                f"- group: `{job.group}`",
                f"- ready: `{job.ready}`",
                f"- blocked_by_runtime: `{job.blocked_by_runtime}`",
                f"- detail: `{job.detail}`",
                "",
            ]
        )
        if job.expected_integrator is not None:
            lines.append(f"- expected_integrator: `{job.expected_integrator}`")
            lines.append(f"- actual_integrator: `{job.actual_integrator}`")
            lines.append("")
        if job.missing_assets:
            lines.append(f"- missing_assets: `{job.missing_assets}`")
            lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    """Smoke-check baseline assets before heavier phases."""

    configure_logging()

    workspace_root = WORKSPACE_ROOT
    project_root = workspace_root.parent
    results_dir = workspace_root / "results"
    manifest_path = workspace_root / "configs" / "baseline_manifest.json"

    host = detect_host_runtime(project_root)
    runtime_profiles = tuple(
        resolve_runtime_profile(template, host)
        for template in load_runtime_profiles(
            workspace_root / "configs" / "runtime_profiles.json"
        )
    )
    primary_profile = select_primary_runtime_profile(
        runtime_profiles,
        host_platform=host.host_platform,
    )
    runtime_info = RuntimeInfo(
        profile_id=primary_profile.profile_id,
        platform_name=host.host_platform,
        python_version=host.python_version,
        mitsuba_binary=str(primary_profile.binary_path),
        binary_format=primary_profile.binary_format,
        render_runtime_supported=primary_profile.ready,
        python_binding_ready=primary_profile.python_binding_ready,
        python_binding_reason=primary_profile.python_binding_reason,
        reason=primary_profile.reason,
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    jobs: list[JobPreflight] = []
    for item in manifest["render_jobs"]:
        jobs.append(render_preflight(project_root, item, runtime_info))
    for item in manifest["training_jobs"]:
        jobs.append(training_preflight(project_root, item))

    payload = {
        "runtime": asdict(runtime_info),
        "jobs": [asdict(job) for job in jobs],
    }

    json_path = results_dir / "phase_03_runtime_preflight.json"
    md_path = results_dir / "phase_03_runtime_preflight.md"
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_markdown_report(md_path, runtime_info, jobs)

    ready_jobs = sum(1 for job in jobs if job.ready)
    blocked_jobs = sum(1 for job in jobs if job.blocked_by_runtime)
    LOGGER.info(
        "runtime preflight finished: ready=%s blocked=%s", ready_jobs, blocked_jobs
    )
    LOGGER.info("json report: %s", json_path)
    LOGGER.info("markdown report: %s", md_path)


if __name__ == "__main__":
    main()
