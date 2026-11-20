from __future__ import annotations

import json
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from masked_diffusion_mpg.runtime import HostRuntime, detect_host_runtime, to_wsl_path


@dataclass(slots=True, frozen=True)
class BuildProfileTemplate:
    """Describe a cross-platform Mitsuba build template."""

    profile_id: str
    label: str
    command_mode: str
    host_platforms: tuple[str, ...]
    target_platform: str
    generator: str
    build_dir: str
    python_mode: str
    python_executable: str
    python_version_prefix: str
    build_config: str
    required_tools: tuple[str, ...]
    configure_args: tuple[str, ...]
    build_target: str


@dataclass(slots=True, frozen=True)
class ToolAvailability:
    """Whether a toolchain binary is discoverable."""

    name: str
    available: bool
    location: str


@dataclass(slots=True, frozen=True)
class ToolchainSnapshot:
    """Captured probe results for toolchain binaries."""

    command_mode: str
    tools: tuple[ToolAvailability, ...]


@dataclass(slots=True, frozen=True)
class CMakeCacheSummary:
    """Selected fields scraped from CMakeCache."""

    exists: bool
    generator: str
    c_compiler: str
    cxx_compiler: str
    mts_enable_python: str
    mts_mpg_fast_iteration: str
    python_executable: str
    python_library: str
    python_module_extension: str


@dataclass(slots=True, frozen=True)
class ResolvedBuildProfile:
    """Materialized CLI commands for configuring/building Mitsuba."""

    profile_id: str
    label: str
    command_mode: str
    target_platform: str
    generator: str
    build_dir: Path
    build_dir_text: str
    source_dir_text: str
    python_text: str
    python_version_prefix: str
    python_bindings_enabled: bool
    ready: bool
    reason: str
    toolchain: ToolchainSnapshot
    configure_command: str
    build_command: str


def load_build_profiles(config_path: Path) -> tuple[BuildProfileTemplate, ...]:
    """Load build profile descriptions from disk.

    Args:
        config_path: Path to `build_profiles.json`.

    Returns:
        Parsed profile templates.
    """

    payload = json.loads(config_path.read_text(encoding="utf-8"))
    return tuple(
        BuildProfileTemplate(
            profile_id=item["id"],
            label=item["label"],
            command_mode=item["command_mode"],
            host_platforms=tuple(item["host_platforms"]),
            target_platform=item["target_platform"],
            generator=item["generator"],
            build_dir=item["build_dir"],
            python_mode=item["python_mode"],
            python_executable=item["python_executable"],
            python_version_prefix=item["python_version_prefix"],
            build_config=item["build_config"],
            required_tools=tuple(item["required_tools"]),
            configure_args=tuple(item["configure_args"]),
            build_target=item["build_target"],
        )
        for item in payload["profiles"]
    )


def quote_powershell(value: str) -> str:
    """Minimal PowerShell quoting for interpolated commands.

    Args:
        value: Raw string.

    Returns:
        Text safe inside double quotes for PowerShell.
    """

    escaped = value.replace('"', '`"')
    return f'"{escaped}"'


def quote_bash(value: str) -> str:
    """Minimal bash quoting helper.

    Args:
        value: Raw string.

    Returns:
        Text safe after `shlex.quote`.
    """

    return shlex.quote(value)


def iter_windows_sdk_bin_dirs() -> tuple[Path, ...]:
    """List candidate dirs that may ship Windows SDK tools.

    Returns:
        Search order tuples (preferred first).
    """

    sdk_root = Path(r"C:\Program Files (x86)\Windows Kits\10\bin")
    if not sdk_root.exists():
        return ()

    versioned_dirs = sorted(
        (
            path / "x64"
            for path in sdk_root.iterdir()
            if path.is_dir() and path.name[:1].isdigit() and (path / "x64").exists()
        ),
        reverse=True,
    )
    fallback_dirs = tuple(
        path
        for path in (sdk_root / "x64", sdk_root / "x86", sdk_root / "arm64")
        if path.exists()
    )
    return tuple(versioned_dirs) + fallback_dirs


def find_windows_sdk_tool(tool_name: str) -> str:
    """Locate an executable under Windows Kits.

    Args:
        tool_name: Binary base name (`rc`, `mt`, ...).

    Returns:
        Absolute path string, empty if unresolved.
    """

    for candidate_dir in iter_windows_sdk_bin_dirs():
        candidate = candidate_dir / f"{tool_name}.exe"
        if candidate.exists():
            return str(candidate)
    return ""


def find_vsdevcmd() -> str:
    """Find Visual Studio VsDevCmd helper.

    Returns:
        Absolute `VsDevCmd.bat` path, empty if unresolved.
    """

    roots = (
        Path(r"C:\Program Files\Microsoft Visual Studio"),
        Path(r"C:\Program Files (x86)\Microsoft Visual Studio"),
    )
    for root in roots:
        if not root.exists():
            continue
        for candidate in sorted(root.glob(r"2022/*/Common7/Tools/VsDevCmd.bat")):
            if candidate.exists():
                return str(candidate)
    return ""


def resolve_native_tool_path(tool_name: str, host_platform: str) -> str:
    """Resolve host-native toolchain binary paths.

    Args:
        tool_name: Binary name queried by profiles.
        host_platform: Identifier for current host OS.

    Returns:
        Executable path string, empty if unresolved.
    """

    direct_hit = shutil.which(tool_name)
    if direct_hit:
        return direct_hit
    # The project bootstrap installs CMake and Ninja beside the active virtual
    # environment's Python, which need not be present on the parent shell PATH.
    if host_platform != "windows":
        venv_candidate = Path(sys.executable).parent / tool_name
        if venv_candidate.is_file():
            return str(venv_candidate)
    if host_platform == "windows" and tool_name == "vsdevcmd":
        return find_vsdevcmd()
    if host_platform == "windows" and tool_name in {"rc", "mt"}:
        return find_windows_sdk_tool(tool_name)
    return ""


def probe_native_tools(
    tool_names: tuple[str, ...],
    *,
    host_platform: str,
) -> ToolchainSnapshot:
    """Probe toolchain binaries visible on native host shells."""

    tools = tuple(
        ToolAvailability(
            name=tool_name,
            available=bool(
                location := resolve_native_tool_path(tool_name, host_platform)
            ),
            location=location,
        )
        for tool_name in tool_names
    )
    return ToolchainSnapshot(command_mode="native", tools=tools)


def probe_wsl_tools(tool_names: tuple[str, ...]) -> ToolchainSnapshot:
    """Probe toolchain binaries reachable inside default WSL."""

    script = (
        "for tool in "
        + " ".join(quote_bash(tool_name) for tool_name in tool_names)
        + "; do "
        + 'tool_path=$(command -v "$tool" || true); '
        + 'printf "%s=%s\\n" "$tool" "$tool_path"; '
        + "done"
    )
    try:
        result = subprocess.run(
            ["wsl.exe", "-e", "sh", "-lc", script],
            check=False,
            capture_output=True,
            encoding="utf-8",
            errors="ignore",
            timeout=10,
        )
    except Exception:
        result = None

    resolved_paths: dict[str, str] = {tool_name: "" for tool_name in tool_names}
    if result is not None and result.returncode == 0:
        for line in result.stdout.splitlines():
            if "=" not in line:
                continue
            tool_name, tool_path = line.split("=", 1)
            resolved_paths[tool_name] = tool_path.strip()

    tools = tuple(
        ToolAvailability(
            name=tool_name,
            available=bool(resolved_paths[tool_name]),
            location=resolved_paths[tool_name],
        )
        for tool_name in tool_names
    )
    return ToolchainSnapshot(command_mode="wsl", tools=tools)


def detect_toolchain(
    template: BuildProfileTemplate,
    host: HostRuntime,
) -> ToolchainSnapshot:
    """Resolve toolchain requirements for requested template/host."""

    if template.command_mode == "wsl" and host.wsl_available:
        return probe_wsl_tools(template.required_tools)
    if template.command_mode == "wsl":
        return ToolchainSnapshot(
            command_mode="wsl",
            tools=tuple(
                ToolAvailability(name=tool_name, available=False, location="")
                for tool_name in template.required_tools
            ),
        )
    return probe_native_tools(
        template.required_tools,
        host_platform=host.host_platform,
    )


def get_missing_tools(toolchain: ToolchainSnapshot) -> tuple[str, ...]:
    """Return names of unavailable tools.

    Args:
        toolchain: Latest probe snapshot.

    Returns:
        Tuple of unfound binaries.
    """

    return tuple(tool.name for tool in toolchain.tools if not tool.available)


def get_python_text(
    template: BuildProfileTemplate,
    host: HostRuntime,
    toolchain: ToolchainSnapshot,
) -> str:
    """Resolve Python interpreter string for template execution.

    Args:
        template: CMake profile template.
        host: Host runtime hints.
        toolchain: Latest probe snapshot.

    Returns:
        Python path or shorthand command used by subprocess.
    """

    if template.python_mode == "host_python":
        return sys.executable
    if resolved_python := get_tool_location(toolchain, template.python_executable):
        return resolved_python
    return template.python_executable


def detect_python_bindings_enabled(template: BuildProfileTemplate) -> bool:
    """Return True when template enables Mitsuba Python bindings.

    Args:
        template: Profile template.

    Returns:
        Whether `-DMTS_ENABLE_PYTHON=ON` is present.
    """

    return any(arg == "-DMTS_ENABLE_PYTHON=ON" for arg in template.configure_args)


def get_tool_location(
    toolchain: ToolchainSnapshot,
    tool_name: str,
) -> str:
    """Look up executable path cached inside toolchain snapshot.

    Args:
        toolchain: Latest probe snapshot.
        tool_name: Tool queried by CMake wrapper.

    Returns:
        Executable path string, empty if unknown.
    """

    for tool in toolchain.tools:
        if tool.name == tool_name:
            return tool.location
    return ""


def normalize_cmake_path_text(value: str) -> str:
    """Normalize Windows paths into CMake-friendly slash form.

    Args:
        value: Raw filesystem string.

    Returns:
        Slash-normalized CMake cache value.
    """

    return value.replace("\\", "/")


def read_cmake_cache_value(cache_path: Path, key: str) -> str:
    """Extract one field from CMakeCache.txt.

    Args:
        cache_path: Path to CMakeCache.
        key: Cache variable key.

    Returns:
        Value substring, empty if missing file/key.
    """

    if not cache_path.exists():
        return ""
    prefix = f"{key}:"
    for line in cache_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line.startswith(prefix):
            continue
        _, _, value = line.partition("=")
        return value.strip()
    return ""


def summarize_cmake_cache(cache_path: Path) -> CMakeCacheSummary:
    """Snapshot selected Mitsuba-related CMake entries."""

    return CMakeCacheSummary(
        exists=cache_path.exists(),
        generator=read_cmake_cache_value(cache_path, "CMAKE_GENERATOR"),
        c_compiler=read_cmake_cache_value(cache_path, "CMAKE_C_COMPILER"),
        cxx_compiler=read_cmake_cache_value(cache_path, "CMAKE_CXX_COMPILER"),
        mts_enable_python=read_cmake_cache_value(cache_path, "MTS_ENABLE_PYTHON"),
        mts_mpg_fast_iteration=read_cmake_cache_value(
            cache_path, "MTS_MPG_FAST_ITERATION"
        ),
        python_executable=read_cmake_cache_value(cache_path, "PYTHON_EXECUTABLE"),
        python_library=read_cmake_cache_value(cache_path, "PYTHON_LIBRARY"),
        python_module_extension=read_cmake_cache_value(
            cache_path, "PYTHON_MODULE_EXTENSION"
        ),
    )


def make_configure_command(
    template: BuildProfileTemplate,
    toolchain: ToolchainSnapshot,
    build_dir_text: str,
    source_dir_text: str,
    python_text: str,
) -> str:
    """Produce configure-step shell snippet."""

    cmake_command = get_tool_location(toolchain, "cmake") or "cmake"
    base_args = [
        cmake_command,
        "-S",
        source_dir_text,
        "-B",
        build_dir_text,
        "-G",
        template.generator,
        *template.configure_args,
        f"-DPYTHON_EXECUTABLE={normalize_cmake_path_text(python_text)}",
    ]
    if (
        template.generator == "Ninja"
        and (ninja_location := get_tool_location(toolchain, "ninja"))
    ):
        base_args.append(
            f"-DCMAKE_MAKE_PROGRAM={normalize_cmake_path_text(ninja_location)}"
        )
    if template.target_platform == "windows" and template.command_mode == "native":
        if rc_location := get_tool_location(toolchain, "rc"):
            base_args.append(
                f"-DCMAKE_RC_COMPILER={normalize_cmake_path_text(rc_location)}"
            )
        if mt_location := get_tool_location(toolchain, "mt"):
            base_args.append(f"-DCMAKE_MT={normalize_cmake_path_text(mt_location)}")
    if template.command_mode == "wsl":
        bash_command = " ".join(quote_bash(arg) for arg in base_args)
        return f"wsl.exe -e bash -lc {quote_powershell(bash_command)}"
    native_command = " ".join(quote_powershell(arg) for arg in base_args)
    if template.target_platform == "windows" and template.command_mode == "native":
        if vsdevcmd_location := get_tool_location(toolchain, "vsdevcmd"):
            cmd_payload = (
                f'call "{normalize_cmake_path_text(vsdevcmd_location)}" '
                f"-arch=x64 -host_arch=x64 && {native_command}"
            )
            return f"cmd /c {quote_powershell(cmd_payload)}"
    return native_command


def make_build_command(
    template: BuildProfileTemplate,
    toolchain: ToolchainSnapshot,
    build_dir_text: str,
) -> str:
    """Produce build-step shell snippet."""

    cmake_command = get_tool_location(toolchain, "cmake") or "cmake"
    base_args = [
        cmake_command,
        "--build",
        build_dir_text,
        "--target",
        template.build_target,
        "--parallel",
    ]
    if template.build_config:
        base_args.extend(["--config", template.build_config])
    if template.command_mode == "wsl":
        bash_command = " ".join(quote_bash(arg) for arg in base_args)
        return f"wsl.exe -e bash -lc {quote_powershell(bash_command)}"
    native_command = " ".join(quote_powershell(arg) for arg in base_args)
    if template.target_platform == "windows" and template.command_mode == "native":
        if vsdevcmd_location := get_tool_location(toolchain, "vsdevcmd"):
            cmd_payload = (
                f'call "{normalize_cmake_path_text(vsdevcmd_location)}" '
                f"-arch=x64 -host_arch=x64 && {native_command}"
            )
            return f"cmd /c {quote_powershell(cmd_payload)}"
    return native_command


def resolve_build_profile(
    template: BuildProfileTemplate,
    host: HostRuntime,
) -> ResolvedBuildProfile:
    """Materialize ResolvedBuildProfile for template + detected host.

    Args:
        template: Repo-defined profile row.
        host: Repo/host capabilities.

    Returns:
        Fully interpolated commands and readiness flags.
    """

    toolchain = detect_toolchain(template, host)
    build_dir = host.repo_root / template.build_dir

    if template.command_mode == "wsl":
        repo_root_text = to_wsl_path(host.repo_root)
        build_dir_text = to_wsl_path(build_dir)
    else:
        repo_root_text = normalize_cmake_path_text(str(host.repo_root))
        build_dir_text = normalize_cmake_path_text(str(build_dir))

    source_dir_text = f"{repo_root_text}/mitsuba".replace("\\", "/")
    python_text = get_python_text(template, host, toolchain)

    if host.host_platform not in template.host_platforms:
        ready = False
        reason = f"host {host.host_platform} not supported"
    elif template.command_mode == "wsl" and not host.wsl_available:
        ready = False
        reason = "wsl unavailable"
    elif missing_tools := get_missing_tools(toolchain):
        ready = False
        reason = f"missing tools: {missing_tools}"
    else:
        ready = True
        reason = "build toolchain ready"

    return ResolvedBuildProfile(
        profile_id=template.profile_id,
        label=template.label,
        command_mode=template.command_mode,
        target_platform=template.target_platform,
        generator=template.generator,
        build_dir=build_dir,
        build_dir_text=build_dir_text,
        source_dir_text=source_dir_text,
        python_text=python_text,
        python_version_prefix=template.python_version_prefix,
        python_bindings_enabled=detect_python_bindings_enabled(template),
        ready=ready,
        reason=reason,
        toolchain=toolchain,
        configure_command=make_configure_command(
            template, toolchain, build_dir_text, source_dir_text, python_text
        ),
        build_command=make_build_command(template, toolchain, build_dir_text),
    )


def select_host_native_profile_id(host: HostRuntime) -> str:
    """Map host OS onto default native build profile identifiers.

    Args:
        host: Active host/runtime descriptor.

    Returns:
        Canonical native profile ID string for the active host.

    Raises:
        RuntimeError: Unsupported host platform flag.
    """

    if host.host_platform == "windows":
        return "windows_native"
    if host.host_platform == "linux":
        return "linux_native"
    if host.host_platform == "darwin":
        return "macos_arm64_native"
    raise RuntimeError(
        f"unsupported host platform for native profile: {host.host_platform}"
    )


def normalize_runtime_profile_id(profile_id: str, host: HostRuntime) -> str:
    """Validate explicit native profile tokens against the active host.

    Args:
        profile_id: Identifier coming from manifests/CLI.
        host: Detected runtime.

    Returns:
        The unchanged profile identifier when it is valid for this host.

    Raises:
        RuntimeError: An explicit native profile targets another platform.
    """

    if profile_id in {
        "windows_native",
        "linux_native",
        "macos_arm64_native",
    }:
        host_profile_id = select_host_native_profile_id(host)
        if profile_id != host_profile_id:
            raise RuntimeError(
                "requested native runtime profile "
                f"{profile_id!r} does not match host platform "
                f"{host.host_platform!r} ({host_profile_id!r}); "
                "refusing implicit cross-platform remap"
            )
    return profile_id


def get_runtime_binary_name(target_platform: str) -> str:
    """Return Mitsuba executable basename for CMake target_platform."""

    if target_platform == "windows":
        return "mitsuba.exe"
    return "mitsuba"


def ensure_runtime_profile_built(repo_root: Path, profile_id: str) -> None:
    """Configure + build Mitsuba when missing for `profile_id`.

    Args:
        repo_root: Repository root holding `mitsuba/`.
        profile_id: Build profile slug from configs.

    Raises:
        RuntimeError: Missing template / toolchain prerequisites.
        subprocess.CalledProcessError: CMake or build subprocess failed.
    """

    host = detect_host_runtime(repo_root)
    templates = load_build_profiles(
        repo_root / "masked_diffusion_workspace" / "configs" / "build_profiles.json"
    )
    resolved_profiles = tuple(
        resolve_build_profile(template, host) for template in templates
    )
    profile = next(
        (item for item in resolved_profiles if item.profile_id == profile_id),
        None,
    )
    if profile is None:
        raise RuntimeError(f"unknown build profile: {profile_id}")
    if not profile.ready:
        raise RuntimeError(f"build profile not ready: {profile.reason}")

    runtime_binary_path = (
        profile.build_dir / "dist" / get_runtime_binary_name(profile.target_platform)
    )
    if runtime_binary_path.exists():
        return

    if host.host_platform == "windows":
        subprocess.run(
            ["powershell.exe", "-NoProfile", "-Command", profile.configure_command],
            check=True,
        )
        subprocess.run(
            ["powershell.exe", "-NoProfile", "-Command", profile.build_command],
            check=True,
        )
        return

    subprocess.run(profile.configure_command, shell=True, check=True)
    subprocess.run(profile.build_command, shell=True, check=True)
