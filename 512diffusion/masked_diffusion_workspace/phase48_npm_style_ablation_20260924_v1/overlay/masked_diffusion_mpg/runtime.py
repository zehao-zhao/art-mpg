from __future__ import annotations

import json
import platform
import re
import shutil
import subprocess
import sys
import sysconfig
from dataclasses import dataclass
from pathlib import Path


PYTHON_EXTENSION_TAG_PATTERN = re.compile(
    r"\.(?P<tag>(?:cp\d+|cpython-\d+)[^.]+)\.(?:so|pyd)$"
)


@dataclass(slots=True, frozen=True)
class RuntimeProfileTemplate:
    """Portable template describing Mitsuba binaries + PYTHONPATH knobs."""

    profile_id: str
    label: str
    command_mode: str
    host_platforms: tuple[str, ...]
    target_platform: str
    binary_path: str
    plugin_dir: str
    python_module_dir: str
    path_entries: tuple[str, ...]
    library_entries: tuple[str, ...]


@dataclass(slots=True, frozen=True)
class HostRuntime:
    """Lightweight fingerprint of interpreter + filesystem assumptions."""

    host_platform: str
    python_version: str
    repo_root: Path
    wsl_available: bool


@dataclass(slots=True, frozen=True)
class ResolvedRuntimeProfile:
    """Hydrated snapshot after joining templates with local repo paths."""

    profile_id: str
    label: str
    command_mode: str
    target_platform: str
    binary_path: Path
    plugin_dir: Path
    python_module_dir: Path
    repo_root_text: str
    binary_text: str
    plugin_text: str
    python_module_text: str
    path_entries: tuple[str, ...]
    library_entries: tuple[str, ...]
    binary_format: str
    python_extension_tags: tuple[str, ...]
    host_python_extension_tag: str
    python_binding_ready: bool
    python_binding_reason: str
    ready: bool
    reason: str


def normalize_platform_name(value: str) -> str:
    """Map arbitrary platform strings onto normalized ids.

    Args:
        value: Raw `platform.system()` style label.

    Returns:
        Canonical `windows`, `linux`, `darwin`, or echo lowercased token.
    """

    lowered = value.lower()
    if lowered.startswith("win"):
        return "windows"
    if lowered.startswith("linux"):
        return "linux"
    if lowered.startswith("darwin"):
        return "darwin"
    return lowered


def is_wsl_available() -> bool:
    """Return True when WSL tooling answers lightweight smoke probes."""

    if normalize_platform_name(platform.system()) != "windows":
        return False
    if shutil.which("wsl.exe") is None:
        return False
    # Cheap probes avoid long `bash -lc` warmup false negatives on cold WSL VMs.
    for command, marker in (
        (["wsl.exe", "-l", "-v"], None),
        (["wsl.exe", "-e", "bash", "-lc", "printf ok"], "ok"),
    ):
        try:
            result = subprocess.run(
                command,
                check=False,
                capture_output=True,
                encoding="utf-8",
                errors="ignore",
                timeout=20,
            )
        except Exception:
            continue
        stdout_text = result.stdout.replace("\x00", "")
        if result.returncode != 0:
            continue
        if marker is None or marker in stdout_text:
            return True
    return False


def detect_host_runtime(repo_root: Path) -> HostRuntime:
    """Gather Python version, OS id, repo root anchor, plus WSL reachability.

    Args:
        repo_root: Checked-out Mitsuba/monorepo root pointing at `mitsuba/`.

    Returns:
        Frozen HostRuntime struct mirroring instantaneous environment facts.
    """

    return HostRuntime(
        host_platform=normalize_platform_name(platform.system()),
        python_version=sys.version.split()[0],
        repo_root=repo_root,
        wsl_available=is_wsl_available(),
    )


def detect_binary_format(binary_path: Path) -> str:
    """Inspect magic bytes describing Mitsuba executables.

    Args:
        binary_path: Candidate binary on disk (`mitsuba.exe`, etc.).

    Returns:
        `ELF`, `PE`, `Mach-O`, literal `missing`, or `unknown` sentinel.
    """

    if not binary_path.exists():
        return "missing"
    magic = binary_path.read_bytes()[:4]
    if magic.startswith(b"\x7fELF"):
        return "ELF"
    if magic.startswith(b"MZ"):
        return "PE"
    if magic in {
        b"\xfe\xed\xfa\xce",
        b"\xfe\xed\xfa\xcf",
        b"\xcf\xfa\xed\xfe",
        b"\xca\xfe\xba\xbe",
    }:
        return "Mach-O"
    return "unknown"


def expected_binary_format(target_platform: str) -> str:
    """Return canonical binary container id for templated Mitsuba bundles.

    Args:
        target_platform: Resolved template key such as windows/linux/darwin.

    Returns:
        String token identifying the binary format (`ELF`, ...).
    """

    if target_platform == "windows":
        return "PE"
    if target_platform == "linux":
        return "ELF"
    if target_platform == "darwin":
        return "Mach-O"
    return "unknown"


def extract_python_extension_tag(filename: str) -> str:
    """Parse cpXXXX-style ABI tags out of Mitsuba-built extension filenames.

    Args:
        filename: Basename emitted under `dist/python`.

    Returns:
        ABI substring or blank when regex misses.
    """

    match = PYTHON_EXTENSION_TAG_PATTERN.search(filename)
    if match is None:
        return ""
    return match.group("tag")


def collect_python_extension_tags(python_module_dir: Path) -> tuple[str, ...]:
    """Collect ABI tags from Mitsuba Python extension modules.

    Args:
        python_module_dir: `dist/python` directory.

    Returns:
        Deduped, sorted tuple of ABI tags.
    """

    if not python_module_dir.exists():
        return ()
    tags = {
        tag
        for path in python_module_dir.rglob("*")
        if path.is_file()
        for tag in [extract_python_extension_tag(path.name)]
        if tag
    }
    return tuple(sorted(tags))


def detect_host_python_extension_tag() -> str:
    """Return the ABI tag expected by the current host Python interpreter.

    Returns:
        Interpreter ABI tag, or empty string when it cannot be parsed.
    """

    ext_suffix = sysconfig.get_config_var("EXT_SUFFIX") or ""
    return extract_python_extension_tag(ext_suffix)


def evaluate_python_binding_status(
    template: RuntimeProfileTemplate,
    host: HostRuntime,
    python_module_dir: Path,
) -> tuple[tuple[str, ...], str, bool, str]:
    """Evaluate compatibility of Mitsuba Python bindings with the current host.

    Args:
        template: Runtime profile template.
        host: Host environment fingerprint.
        python_module_dir: `dist/python` directory.

    Returns:
        ABI tag tuple, host ABI tag, compatibility flag, and reason string.
    """

    python_extension_tags = collect_python_extension_tags(python_module_dir)
    host_python_extension_tag = detect_host_python_extension_tag()

    if not python_extension_tags:
        return (
            python_extension_tags,
            host_python_extension_tag,
            False,
            "missing compiled python modules",
        )

    # Cross-target: ABI cannot be validated on this host; rely on blocker text only.
    if (
        template.command_mode != "native"
        or template.target_platform != host.host_platform
    ):
        return (
            python_extension_tags,
            host_python_extension_tag,
            False,
            "cross-target python bindings require in-target validation",
        )

    if not host_python_extension_tag:
        return (
            python_extension_tags,
            host_python_extension_tag,
            False,
            "host python extension tag unavailable",
        )

    if host_python_extension_tag not in python_extension_tags:
        return (
            python_extension_tags,
            host_python_extension_tag,
            False,
            f"python extension tag mismatch: have {python_extension_tags}, need {host_python_extension_tag}",
        )

    return (
        python_extension_tags,
        host_python_extension_tag,
        True,
        "python bindings ready",
    )


def to_wsl_path(path: Path) -> str:
    """Convert a Windows absolute path to a WSL mount path string.

    Args:
        path: Windows absolute Path.

    Returns:
        Equivalent `/mnt/<drive>/...` style path.
    """

    drive = path.drive.rstrip(":").lower()
    parts = [part for part in path.parts[1:] if part not in {path.drive, "\\"}]
    return "/mnt/" + drive + "/" + "/".join(parts)


def load_runtime_profiles(config_path: Path) -> tuple[RuntimeProfileTemplate, ...]:
    """Load portable runtime profile templates.

    Args:
        config_path: Path to `runtime_profiles.json`.

    Returns:
        Tuple of parsed RuntimeProfileTemplate entries.
    """

    payload = json.loads(config_path.read_text(encoding="utf-8"))
    return tuple(
        RuntimeProfileTemplate(
            profile_id=item["id"],
            label=item["label"],
            command_mode=item["command_mode"],
            host_platforms=tuple(item["host_platforms"]),
            target_platform=item["target_platform"],
            binary_path=item["binary_path"],
            plugin_dir=item["plugin_dir"],
            python_module_dir=item["python_module_dir"],
            path_entries=tuple(item["path_entries"]),
            library_entries=tuple(item["library_entries"]),
        )
        for item in payload["profiles"]
    )


def build_env_entries(
    repo_root_text: str, relative_entries: tuple[str, ...]
) -> tuple[str, ...]:
    """Expand relative PATH/LD_LIBRARY_PATH entries inside the runtime root.

    Args:
        repo_root_text: Repo root as seen by the target runtime.
        relative_entries: Paths relative to that repo root.

    Returns:
        Tuple of joined absolute path strings.
    """

    return tuple(
        f"{repo_root_text}/{entry}".replace("\\", "/") for entry in relative_entries
    )


def to_target_text(repo_root: Path, repo_root_text: str, absolute_path: Path) -> str:
    """Rewrite an on-disk repo path using the runtime's repo root string form.

    Args:
        repo_root: Host-visible repo root path.
        repo_root_text: Repo root string understood by that runtime (e.g., WSL).
        absolute_path: Absolute path anchored under repo_root.

    Returns:
        Path text consumable directly inside the target runtime.
    """

    relative_path = absolute_path.relative_to(repo_root)
    return f"{repo_root_text}/{relative_path.as_posix()}"


def resolve_runtime_profile(
    template: RuntimeProfileTemplate,
    host: HostRuntime,
) -> ResolvedRuntimeProfile:
    """Resolve a runtime template against the host and annotate readiness bits.

    Args:
        template: Runtime profile template from JSON.
        host: Host fingerprint from detect_host_runtime.

    Returns:
        ResolvedRuntimeProfile hydrated with textual paths plus status fields.
    """

    binary_path = host.repo_root / template.binary_path
    plugin_dir = host.repo_root / template.plugin_dir
    python_module_dir = host.repo_root / template.python_module_dir
    binary_format = detect_binary_format(binary_path)

    if template.command_mode == "wsl":
        repo_root_text = to_wsl_path(host.repo_root)
    else:
        repo_root_text = str(host.repo_root)

    binary_text = to_target_text(host.repo_root, repo_root_text, binary_path)
    plugin_text = to_target_text(host.repo_root, repo_root_text, plugin_dir)
    python_module_text = to_target_text(
        host.repo_root, repo_root_text, python_module_dir
    )
    path_entries = build_env_entries(repo_root_text, template.path_entries)
    library_entries = build_env_entries(repo_root_text, template.library_entries)
    (
        python_extension_tags,
        host_python_extension_tag,
        python_binding_ready,
        python_binding_reason,
    ) = evaluate_python_binding_status(template, host, python_module_dir)

    if host.host_platform not in template.host_platforms:
        ready = False
        reason = f"host {host.host_platform} not supported"
    elif template.command_mode == "wsl" and not host.wsl_available:
        ready = False
        reason = "wsl unavailable"
    elif not binary_path.exists():
        ready = False
        reason = f"missing binary: {binary_path}"
    elif binary_format != expected_binary_format(template.target_platform):
        ready = False
        reason = (
            f"binary format {binary_format} != "
            f"{expected_binary_format(template.target_platform)}"
        )
    else:
        ready = True
        reason = "runtime ready"

    return ResolvedRuntimeProfile(
        profile_id=template.profile_id,
        label=template.label,
        command_mode=template.command_mode,
        target_platform=template.target_platform,
        binary_path=binary_path,
        plugin_dir=plugin_dir,
        python_module_dir=python_module_dir,
        repo_root_text=repo_root_text,
        binary_text=binary_text,
        plugin_text=plugin_text,
        python_module_text=python_module_text,
        path_entries=path_entries,
        library_entries=library_entries,
        binary_format=binary_format,
        python_extension_tags=python_extension_tags,
        host_python_extension_tag=host_python_extension_tag,
        python_binding_ready=python_binding_ready,
        python_binding_reason=python_binding_reason,
        ready=ready,
        reason=reason,
    )


def make_shell_export_block(
    profile: ResolvedRuntimeProfile,
    *,
    windows_shell: bool,
) -> str:
    """Emit shell snippets that export PLUGIN_PATH/PYTHONPATH/PATH (+LD_LIBRARY_PATH).

    Args:
        profile: Resolved runtime snapshot.
        windows_shell: When True, emit PowerShell syntax instead of POSIX sh.

    Returns:
        Executable text assigning the Mitsuba-facing environment knobs.
    """

    if windows_shell:
        path_expr = ";".join(profile.path_entries) + ";$env:PATH"
        pythonpath_expr = f"{profile.python_module_text};$env:PYTHONPATH"
        lines = [
            f'$env:MTS_PLUGIN_PATH="{profile.plugin_text}"',
            f'$env:PYTHONPATH="{pythonpath_expr}"',
            f'$env:PATH="{path_expr}"',
        ]
        return "\n".join(lines)

    path_expr = ":".join(profile.path_entries) + ":${PATH:-}"
    pythonpath_expr = f"{profile.python_module_text}:${{PYTHONPATH:-}}"
    library_expr = ":".join(profile.library_entries) + ":${LD_LIBRARY_PATH:-}"
    lines = [
        f'export MTS_PLUGIN_PATH="{profile.plugin_text}"',
        f'export PYTHONPATH="{pythonpath_expr}"',
        f'export PATH="{path_expr}"',
        f'export LD_LIBRARY_PATH="{library_expr}"',
    ]
    return "\n".join(lines)
