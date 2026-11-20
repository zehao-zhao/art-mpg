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

from masked_diffusion_mpg.manifest import (  # noqa: E402
    BaselineManifest,
    RenderJob,
    TrainingJob,
    collect_missing_assets,
    get_integrator_type,
    load_baseline_manifest,
    validate_manifest_paths,
)
from masked_diffusion_mpg.runtime import (  # noqa: E402
    HostRuntime,
    ResolvedRuntimeProfile,
    detect_host_runtime,
    load_runtime_profiles,
    make_shell_export_block,
    resolve_runtime_profile,
)


LOGGER = logging.getLogger("phase_05_runtime_matrix")


@dataclass(slots=True, frozen=True)
class JobMatrixRecord:
    """Rendered command entry for manifest jobs under some runtime."""

    profile_id: str
    job_id: str
    job_type: str
    ready: bool
    blocked_reason: str
    command: str


def configure_logging() -> None:
    """Initialize logging."""

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def quote_powershell(value: str) -> str:
    """Minimal PowerShell quoting for interpolated strings."""

    escaped = value.replace('"', '`"')
    return f'"{escaped}"'


def quote_bash(value: str) -> str:
    """Minimal bash quoting for interpolated strings."""

    escaped = value.replace('"', '\\"')
    return f'"{escaped}"'


def render_command(profile: ResolvedRuntimeProfile, job: RenderJob) -> str:
    """Produce an executable snippet for rendering one scene.

    Args:
        profile: Resolved runtime describing shell layout.
        job: Serialized render job pulled from manifests.

    Returns:
        Full command string honoring host-specific shell glue.
    """

    scene_text = f"{profile.repo_root_text}/{job.scene_path}".replace("\\", "/")
    out_dir = f"{profile.repo_root_text}/masked_diffusion_workspace/results/baseline_runs".replace(
        "\\", "/"
    )
    out_path = f"{out_dir}/{job.job_id}.exr"

    if profile.target_platform == "windows" and profile.command_mode == "native":
        export_block = make_shell_export_block(profile, windows_shell=True)
        mkdir_command = f"New-Item -ItemType Directory -Force -Path {quote_powershell(out_dir)} | Out-Null"
        command = (
            f"& {quote_powershell(profile.binary_text)} -m {job.variant} "
            f"-o {quote_powershell(out_path)} {quote_powershell(scene_text)}"
        )
        return export_block + "\n" + mkdir_command + "\n" + command

    bash_block = make_shell_export_block(profile, windows_shell=False)
    mkdir_command = f"mkdir -p {quote_bash(out_dir)}"
    bash_command = (
        f"{quote_bash(profile.binary_text)} -m {job.variant} -t $(nproc) "
        f"-o {quote_bash(out_path)} {quote_bash(scene_text)}"
    )
    if profile.command_mode == "wsl":
        return (
            "wsl.exe -e bash -lc '\n"
            + bash_block
            + "\n"
            + mkdir_command
            + "\n"
            + bash_command
            + "\n'"
        )
    return bash_block + "\n" + mkdir_command + "\n" + bash_command


def training_command(
    project_root: Path, profile: ResolvedRuntimeProfile, job: TrainingJob
) -> str:
    """Produce runnable training helper command text."""

    if profile.command_mode == "wsl":
        script_text = f"{profile.repo_root_text}/{job.script_path}".replace("\\", "/")
        args_text = " ".join(quote_bash(arg) for arg in job.args)
        return (
            "wsl.exe -e bash -lc '\npython "
            + quote_bash(script_text)
            + " "
            + args_text
            + "\n'"
        )

    script_path = project_root / job.script_path
    args_text = " ".join(quote_powershell(arg) for arg in job.args)
    return f"python {quote_powershell(str(script_path))} {args_text}".strip()


def make_render_record(
    project_root: Path,
    profile: ResolvedRuntimeProfile,
    job: RenderJob,
) -> JobMatrixRecord:
    """Build matrix row reflecting render prerequisites."""

    scene_path = project_root / job.scene_path
    actual_integrator = get_integrator_type(scene_path)
    missing_assets = collect_missing_assets(scene_path)
    command = render_command(profile, job)

    if not profile.ready:
        return JobMatrixRecord(
            profile_id=profile.profile_id,
            job_id=job.job_id,
            job_type="render",
            ready=False,
            blocked_reason=profile.reason,
            command=command,
        )
    if actual_integrator != job.expected_integrator:
        return JobMatrixRecord(
            profile_id=profile.profile_id,
            job_id=job.job_id,
            job_type="render",
            ready=False,
            blocked_reason=f"integrator mismatch: {actual_integrator}",
            command=command,
        )
    if missing_assets:
        return JobMatrixRecord(
            profile_id=profile.profile_id,
            job_id=job.job_id,
            job_type="render",
            ready=False,
            blocked_reason=f"missing assets: {missing_assets}",
            command=command,
        )
    return JobMatrixRecord(
        profile_id=profile.profile_id,
        job_id=job.job_id,
        job_type="render",
        ready=True,
        blocked_reason="",
        command=command,
    )


def make_training_record(
    project_root: Path,
    profile: ResolvedRuntimeProfile,
    job: TrainingJob,
) -> JobMatrixRecord:
    """Build matrix row reflecting training prerequisites."""

    script_path = project_root / job.script_path
    command = training_command(project_root, profile, job)
    if not script_path.exists():
        return JobMatrixRecord(
            profile_id=profile.profile_id,
            job_id=job.job_id,
            job_type="training",
            ready=False,
            blocked_reason=f"missing script: {script_path}",
            command=command,
        )
    return JobMatrixRecord(
        profile_id=profile.profile_id,
        job_id=job.job_id,
        job_type="training",
        ready=True,
        blocked_reason="",
        command=command,
    )


def write_markdown_report(
    output_path: Path,
    host: HostRuntime,
    profiles: tuple[ResolvedRuntimeProfile, ...],
    records: tuple[JobMatrixRecord, ...],
) -> None:
    """Write Markdown matrix snapshot for inspection."""

    lines = [
        "# Phase 05 Runtime Matrix",
        "",
        "## Host",
        "",
        f"- platform: `{host.host_platform}`",
        f"- python: `{host.python_version}`",
        f"- repo_root: `{host.repo_root}`",
        f"- wsl_available: `{host.wsl_available}`",
        "",
        "## Profiles",
        "",
    ]
    for profile in profiles:
        lines.extend(
            [
                f"### {profile.profile_id}",
                "",
                f"- label: `{profile.label}`",
                f"- target_platform: `{profile.target_platform}`",
                f"- command_mode: `{profile.command_mode}`",
                f"- binary_format: `{profile.binary_format}`",
                f"- ready: `{profile.ready}`",
                f"- reason: `{profile.reason}`",
                "",
            ]
        )
    lines.extend(["## Commands", ""])
    for record in records:
        lines.extend(
            [
                f"### {record.profile_id} / {record.job_id}",
                "",
                f"- type: `{record.job_type}`",
                f"- ready: `{record.ready}`",
                f"- blocked_reason: `{record.blocked_reason}`",
                "",
                "```text",
                record.command,
                "```",
                "",
            ]
        )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    """Emit platform-local render/train command stubs."""

    configure_logging()

    project_root = WORKSPACE_ROOT.parent
    config_dir = WORKSPACE_ROOT / "configs"
    results_dir = WORKSPACE_ROOT / "results"
    (results_dir / "baseline_runs").mkdir(parents=True, exist_ok=True)

    manifest: BaselineManifest = load_baseline_manifest(
        config_dir / "baseline_manifest.json"
    )
    missing_paths = validate_manifest_paths(manifest, project_root)
    if missing_paths:
        raise FileNotFoundError(f"missing paths: {missing_paths}")

    host = detect_host_runtime(project_root)
    profiles = tuple(
        resolve_runtime_profile(template, host)
        for template in load_runtime_profiles(config_dir / "runtime_profiles.json")
    )

    records: list[JobMatrixRecord] = []
    for profile in profiles:
        for job in manifest.render_jobs:
            records.append(make_render_record(project_root, profile, job))
        for job in manifest.training_jobs:
            records.append(make_training_record(project_root, profile, job))

    payload = {
        "host": asdict(host),
        "profiles": [asdict(profile) for profile in profiles],
        "records": [asdict(record) for record in records],
    }

    json_path = results_dir / "phase_05_runtime_matrix.json"
    md_path = results_dir / "phase_05_runtime_matrix.md"
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    write_markdown_report(md_path, host, profiles, tuple(records))

    ready_records = sum(1 for record in records if record.ready)
    LOGGER.info("runtime matrix ready records: %s / %s", ready_records, len(records))
    LOGGER.info("json report: %s", json_path)
    LOGGER.info("markdown report: %s", md_path)


if __name__ == "__main__":
    main()
