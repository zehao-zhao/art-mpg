from __future__ import annotations

import json
import logging
import sys
from dataclasses import asdict, dataclass
from pathlib import Path


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = WORKSPACE_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from masked_diffusion_mpg.build import (  # noqa: E402
    CMakeCacheSummary,
    ResolvedBuildProfile,
    load_build_profiles,
    resolve_build_profile,
    summarize_cmake_cache,
)
from masked_diffusion_mpg.runtime import (  # noqa: E402
    collect_python_extension_tags,
    detect_binary_format,
    detect_host_python_extension_tag,
    detect_host_runtime,
    expected_binary_format,
)


LOGGER = logging.getLogger("phase_07_build_matrix")


@dataclass(slots=True, frozen=True)
class DistArtifactSummary:
    """Artifact state within one build directory."""

    binary_path: str
    binary_format: str
    python_module_dir: str
    python_extension_tags: tuple[str, ...]


@dataclass(slots=True, frozen=True)
class BuildMatrixRecord:
    """One matrix row per build profile."""

    profile_id: str
    ready: bool
    blocked_reason: str
    rebuild_required: bool
    rebuild_reason: str
    python_bindings_enabled: bool
    cache_summary: CMakeCacheSummary
    dist_summary: DistArtifactSummary
    configure_command: str
    build_command: str


def configure_logging() -> None:
    """Configure root logging formatting."""

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def cache_matches_python_version(
    cache_summary: CMakeCacheSummary,
    python_version_prefix: str,
) -> bool:
    """Check whether cached CMake picks the expected Python version.

    Args:
        cache_summary: Cached CMake summaries.
        python_version_prefix: Expected major.minor prefix string.

    Returns:
        True when the cached python paths mention that prefix.
    """

    haystack = " ".join(
        [
            cache_summary.python_executable,
            cache_summary.python_library,
            cache_summary.python_module_extension,
        ]
    )
    return python_version_prefix in haystack


def summarize_dist_artifacts(profile: ResolvedBuildProfile) -> DistArtifactSummary:
    """Summarize `dist/` artefacts for one resolved CMake profile.

    Args:
        profile: Resolved build-profile metadata.

    Returns:
        DistArtifactSummary with binary/module paths plus format tags.
    """

    binary_name = "mitsuba.exe" if profile.target_platform == "windows" else "mitsuba"
    binary_path = profile.build_dir / "dist" / binary_name
    python_module_dir = profile.build_dir / "dist" / "python"
    return DistArtifactSummary(
        binary_path=str(binary_path),
        binary_format=detect_binary_format(binary_path),
        python_module_dir=str(python_module_dir),
        python_extension_tags=collect_python_extension_tags(python_module_dir),
    )


def compute_rebuild_requirement(
    host_platform: str,
    profile: ResolvedBuildProfile,
    cache_summary: CMakeCacheSummary,
    dist_summary: DistArtifactSummary,
    host_python_tag: str,
) -> tuple[bool, str]:
    """Decide whether CMake rebuild is required given current artifacts.

    Args:
        host_platform: Host OS id.
        profile: Resolved build profile.
        cache_summary: CMake cache summary for this build dir.
        dist_summary: dist/ artifact summary for this profile.
        host_python_tag: ABI tag for the interpreter on this machine.

    Returns:
        Tuple of (rebuild_needed, textual reason).
    """

    expected_format = expected_binary_format(profile.target_platform)
    if dist_summary.binary_format != expected_format:
        return (
            True,
            f"current dist binary format {dist_summary.binary_format} != {expected_format}",
        )

    if not cache_summary.exists:
        return True, "missing existing CMakeCache.txt"

    if cache_summary.generator != profile.generator:
        return (
            True,
            f"generator mismatch: {cache_summary.generator} vs {profile.generator}",
        )

    expected_python_switch = "ON" if profile.python_bindings_enabled else "OFF"
    if cache_summary.mts_enable_python != expected_python_switch:
        return (
            True,
            "MTS_ENABLE_PYTHON mismatch: "
            f"{cache_summary.mts_enable_python} vs {expected_python_switch}",
        )

    if (
        profile.target_platform == "windows"
        and "MSVC" not in cache_summary.cxx_compiler
        and "cl.exe" not in cache_summary.cxx_compiler.lower()
    ):
        return True, f"existing compiler is not MSVC: {cache_summary.cxx_compiler}"

    if (
        profile.target_platform == "linux"
        and "/usr/bin/" not in cache_summary.cxx_compiler
    ):
        return (
            True,
            f"existing compiler is not linux native: {cache_summary.cxx_compiler}",
        )

    if profile.python_bindings_enabled and not cache_matches_python_version(
        cache_summary, profile.python_version_prefix
    ):
        return (
            True,
            "existing cache python version does not match "
            f"{profile.python_version_prefix}: {cache_summary.python_executable}",
        )

    if profile.python_bindings_enabled and not dist_summary.python_extension_tags:
        return True, "missing compiled python modules"

    if (
        profile.python_bindings_enabled
        and profile.command_mode == "native"
        and profile.target_platform == host_platform
    ):
        if not host_python_tag:
            return True, "host python extension tag unavailable"
        if host_python_tag not in dist_summary.python_extension_tags:
            return (
                True,
                "current dist python tags do not match host python: "
                f"{dist_summary.python_extension_tags} vs {host_python_tag}",
            )

    return False, "existing dist matches current target"


def write_markdown_report(
    output_path: Path,
    *,
    host_platform: str,
    python_version: str,
    host_python_tag: str,
    profiles: tuple[ResolvedBuildProfile, ...],
    records: tuple[BuildMatrixRecord, ...],
) -> None:
    """Write build-matrix Markdown."""

    lines = [
        "# Phase 07 Build Matrix",
        "",
        "## Host",
        "",
        f"- platform: `{host_platform}`",
        f"- python: `{python_version}`",
        f"- host_python_tag: `{host_python_tag}`",
        "",
        "## Profiles",
        "",
    ]
    for profile in profiles:
        tool_lines = ", ".join(
            f"{tool.name}={'ok' if tool.available else 'missing'}"
            for tool in profile.toolchain.tools
        )
        lines.extend(
            [
                f"### {profile.profile_id}",
                "",
                f"- ready: `{profile.ready}`",
                f"- reason: `{profile.reason}`",
                f"- generator: `{profile.generator}`",
                f"- build_dir: `{profile.build_dir_text}`",
                f"- python: `{profile.python_text}`",
                f"- python_bindings_enabled: `{profile.python_bindings_enabled}`",
                f"- tools: `{tool_lines}`",
                "",
            ]
        )

    lines.extend(["## Commands", ""])
    for record in records:
        lines.extend(
            [
                f"### {record.profile_id}",
                "",
                f"- ready: `{record.ready}`",
                f"- blocked_reason: `{record.blocked_reason}`",
                f"- rebuild_required: `{record.rebuild_required}`",
                f"- rebuild_reason: `{record.rebuild_reason}`",
                f"- binary_path: `{record.dist_summary.binary_path}`",
                f"- binary_format: `{record.dist_summary.binary_format}`",
                f"- python_module_dir: `{record.dist_summary.python_module_dir}`",
                f"- python_extension_tags: `{record.dist_summary.python_extension_tags}`",
                f"- cache_exists: `{record.cache_summary.exists}`",
                f"- cache_generator: `{record.cache_summary.generator}`",
                f"- cache_cxx_compiler: `{record.cache_summary.cxx_compiler}`",
                f"- cache_mts_enable_python: `{record.cache_summary.mts_enable_python}`",
                "",
                "```text",
                record.configure_command,
                record.build_command,
                "```",
                "",
            ]
        )

    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    """CLI entry emitting the Phase 07 Mitsuba rebuild matrix."""

    configure_logging()

    project_root = WORKSPACE_ROOT.parent
    config_dir = WORKSPACE_ROOT / "configs"
    results_dir = WORKSPACE_ROOT / "results"
    # Ensure results exist when cloning into a bare tree.
    results_dir.mkdir(parents=True, exist_ok=True)

    host = detect_host_runtime(project_root)
    profiles = tuple(
        resolve_build_profile(template, host)
        for template in load_build_profiles(config_dir / "build_profiles.json")
    )
    host_python_tag = detect_host_python_extension_tag()

    records = tuple(
        BuildMatrixRecord(
            profile_id=profile.profile_id,
            ready=profile.ready,
            blocked_reason="" if profile.ready else profile.reason,
            rebuild_required=(
                rebuild := compute_rebuild_requirement(
                    host.host_platform,
                    profile,
                    cache_summary := summarize_cmake_cache(
                        profile.build_dir / "CMakeCache.txt"
                    ),
                    dist_summary := summarize_dist_artifacts(profile),
                    host_python_tag,
                )
            )[0],
            rebuild_reason=rebuild[1],
            python_bindings_enabled=profile.python_bindings_enabled,
            cache_summary=cache_summary,
            dist_summary=dist_summary,
            configure_command=profile.configure_command,
            build_command=profile.build_command,
        )
        for profile in profiles
    )

    payload = {
        "host": {
            "platform": host.host_platform,
            "python_version": host.python_version,
            "host_python_tag": host_python_tag,
        },
        "profiles": [asdict(profile) for profile in profiles],
        "records": [asdict(record) for record in records],
    }

    json_path = results_dir / "phase_07_build_matrix.json"
    md_path = results_dir / "phase_07_build_matrix.md"
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    write_markdown_report(
        md_path,
        host_platform=host.host_platform,
        python_version=host.python_version,
        host_python_tag=host_python_tag,
        profiles=profiles,
        records=records,
    )

    ready_records = sum(1 for record in records if record.ready)
    rebuild_records = sum(1 for record in records if record.rebuild_required)
    LOGGER.info("build matrix ready records: %s / %s", ready_records, len(records))
    LOGGER.info(
        "build matrix rebuild-needed records: %s / %s", rebuild_records, len(records)
    )
    LOGGER.info("json report: %s", json_path)
    LOGGER.info("markdown report: %s", md_path)


if __name__ == "__main__":
    main()
