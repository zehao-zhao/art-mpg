from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shlex
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = WORKSPACE_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from masked_diffusion_mpg.manifest import RenderJob, load_baseline_manifest  # noqa: E402
from masked_diffusion_mpg.build import (  # noqa: E402
    ensure_runtime_profile_built,
    normalize_runtime_profile_id,
)
from masked_diffusion_mpg.runtime import (  # noqa: E402
    ResolvedRuntimeProfile,
    detect_host_runtime,
    load_runtime_profiles,
    resolve_runtime_profile,
    to_wsl_path,
)


LOGGER = logging.getLogger("phase_12_run_runtime_smoke")


class Phase12RuntimeSmokeError(RuntimeError):
    """Describes expected failures during the Phase 12 runtime smoke stage."""


@dataclass(slots=True, frozen=True)
class RuntimeSmokeConfig:
    """Describes Phase 12 runtime smoke settings.

    Attributes:
        profile_id: Runtime profile identifier.
        job_ids: Render jobs for this run.
        width: Output width.
        height: Output height.
        crop_width: Crop width.
        crop_height: Crop height.
        sample_count: Smoke sample count.
        samples_per_pass: Samples per pass.
        threads: Mitsuba thread count.
        runtime_model_path: Relative path to the runtime model.
    """

    profile_id: str
    job_ids: tuple[str, ...]
    width: int
    height: int
    crop_width: int
    crop_height: int
    sample_count: int
    samples_per_pass: int
    threads: int
    runtime_model_path: str | None = None
    runtime_model_paths: dict[str, str] | None = None


@dataclass(slots=True, frozen=True)
class RuntimeSmokeRecord:
    """Records the outcome of one runtime smoke job.

    Attributes:
        job_id: Job identifier.
        scene_path: Original scene path.
        ok: Whether the job passed.
        return_code: Mitsuba exit code.
        seconds: Total wall-clock time.
        output_path: Output image path.
        output_exists: Whether the output image exists.
        output_size: Output image byte size.
        patched_scene_record: Path to saved patched scene.
        stdout_log: Standard output log path.
        stderr_log: Standard error log path.
        command: Command line actually executed.
        loaded_runtime_model: Whether logs show the runtime model loaded.
        disabled_runtime_model: Whether runtime fallback was disabled.
    """

    job_id: str
    scene_path: str
    ok: bool
    return_code: int
    seconds: float
    output_path: str
    output_exists: bool
    output_size: int
    patched_scene_record: str
    stdout_log: str
    stderr_log: str
    command: list[str]
    loaded_runtime_model: bool
    disabled_runtime_model: bool
    runtime_model_path: str


def configure_logging() -> None:
    """Initialize logging."""

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description="Run Phase 12 runtime smoke")
    parser.add_argument(
        "--config",
        type=Path,
        default=WORKSPACE_ROOT / "configs" / "phase_12_runtime_smoke.json",
    )
    return parser.parse_args()


def load_config(config_path: Path) -> RuntimeSmokeConfig:
    """Load Phase 12 runtime smoke configuration.

    Args:
        config_path: Path to the config file.

    Returns:
        Parsed configuration object.
    """

    payload = json.loads(config_path.read_text(encoding="utf-8"))
    return RuntimeSmokeConfig(
        profile_id=str(payload["profile_id"]),
        job_ids=tuple(str(item) for item in payload["job_ids"]),
        width=int(payload["width"]),
        height=int(payload["height"]),
        crop_width=int(payload["crop_width"]),
        crop_height=int(payload["crop_height"]),
        sample_count=int(payload["sample_count"]),
        samples_per_pass=int(payload["samples_per_pass"]),
        threads=int(payload["threads"]),
        runtime_model_path=(
            None
            if payload.get("runtime_model_path") is None
            else str(payload["runtime_model_path"])
        ),
        runtime_model_paths=(
            None
            if payload.get("runtime_model_paths") is None
            else {
                str(key): str(value)
                for key, value in dict(payload["runtime_model_paths"]).items()
            }
        ),
    )


def resolve_runtime_model_path(
    *,
    workspace_root: Path,
    config: RuntimeSmokeConfig,
    job: RenderJob,
) -> Path:
    """Resolve the runtime model path for the given job."""

    candidate_keys = (job.job_id, job.job_id.removesuffix("_mpg"))
    if config.runtime_model_paths is not None:
        for candidate_key in candidate_keys:
            runtime_path = config.runtime_model_paths.get(candidate_key)
            if runtime_path is None:
                continue
            resolved_path = (workspace_root / runtime_path).resolve()
            if not resolved_path.exists():
                raise Phase12RuntimeSmokeError(
                    f"runtime model missing for {candidate_key}: {resolved_path}"
                )
            return resolved_path

    if config.runtime_model_path is None:
        raise Phase12RuntimeSmokeError("runtime_model_path is missing")
    resolved_path = (workspace_root / config.runtime_model_path).resolve()
    if not resolved_path.exists():
        raise Phase12RuntimeSmokeError(f"runtime model missing: {resolved_path}")
    return resolved_path


def select_runtime_profile(
    profiles: tuple[ResolvedRuntimeProfile, ...],
    profile_id: str,
) -> ResolvedRuntimeProfile:
    """Select the resolved runtime profile for profile_id.

    Args:
        profiles: Resolved runtime profiles.
        profile_id: Target profile id.

    Returns:
        Matching runtime profile.
    """

    for profile in profiles:
        if profile.profile_id == profile_id:
            return profile
    raise Phase12RuntimeSmokeError(f"unknown runtime profile: {profile_id}")


def select_render_jobs(
    all_jobs: tuple[RenderJob, ...],
    job_ids: tuple[str, ...],
) -> tuple[RenderJob, ...]:
    """Filter render jobs from the baseline manifest.

    Args:
        all_jobs: All render jobs.
        job_ids: Target job identifiers.

    Returns:
        Jobs to run in this invocation.
    """

    job_map = {job.job_id: job for job in all_jobs}
    selected: list[RenderJob] = []
    for job_id in job_ids:
        job = job_map.get(job_id)
        if job is None:
            raise Phase12RuntimeSmokeError(f"unknown render job: {job_id}")
        if job.method != "mpg":
            raise Phase12RuntimeSmokeError(
                f"phase_12 runtime smoke only supports mpg jobs: {job_id}"
            )
        selected.append(job)
    return tuple(selected)


def prepend_env_path(
    base_env: dict[str, str], key: str, entries: tuple[str, ...]
) -> None:
    """Prepend path entries onto an environment variable.

    Args:
        base_env: Target environment dictionary.
        key: Environment variable name.
        entries: Paths to prepend.
    """

    current_value = base_env.get(key, "")
    normalized_entries = [entry for entry in entries if entry]
    if current_value:
        normalized_entries.append(current_value)
    base_env[key] = os.pathsep.join(normalized_entries)


def build_runtime_env(profile: ResolvedRuntimeProfile) -> dict[str, str]:
    """Build Mitsuba CLI environment for rendering.

    Args:
        profile: Resolved runtime profile.

    Returns:
        Environment dict suitable for subprocess.
    """

    if not profile.ready:
        raise Phase12RuntimeSmokeError(f"runtime profile not ready: {profile.reason}")

    if profile.command_mode == "native":
        env = os.environ.copy()
        env["MTS_PLUGIN_PATH"] = str(profile.plugin_dir)
        prepend_env_path(env, "PYTHONPATH", (str(profile.python_module_dir),))
        prepend_env_path(env, "PATH", tuple(profile.path_entries))
        return env

    if profile.command_mode == "wsl":
        return {}

    raise Phase12RuntimeSmokeError(f"unsupported runtime mode: {profile.command_mode}")


def to_profile_path(profile: ResolvedRuntimeProfile, path: Path) -> str:
    """Convert a local path to the form the target runtime expects."""

    resolved = path.resolve()
    if profile.command_mode == "wsl":
        return to_wsl_path(resolved)
    return str(resolved)


def build_wsl_command(
    *,
    profile: ResolvedRuntimeProfile,
    command: list[str],
) -> list[str]:
    """Wrap a Mitsuba command so it runs under WSL."""

    path_prefix = ":".join(profile.path_entries)
    library_prefix = ":".join(profile.library_entries)
    command_text = " ".join(shlex.quote(item) for item in command)
    shell_lines = [
        f"cd {shlex.quote(profile.repo_root_text)}",
        f"export MTS_PLUGIN_PATH={shlex.quote(profile.plugin_text)}",
        (
            f"export PYTHONPATH={shlex.quote(profile.python_module_text)}"
            "${PYTHONPATH:+:$PYTHONPATH}"
        ),
        f"export PATH={shlex.quote(path_prefix)}${{PATH:+:$PATH}}",
        (
            f"export LD_LIBRARY_PATH={shlex.quote(library_prefix)}"
            "${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
        ),
        command_text,
    ]
    return ["wsl.exe", "-e", "bash", "-lc", " && ".join(shell_lines)]


def replace_value(text: str, tag: str, name: str, value: str) -> tuple[str, bool]:
    """Replace the value attribute on a named XML tag.

    Args:
        text: Original XML text.
        tag: Node tag name.
        name: Attribute name.
        value: New value string.

    Returns:
        Updated text and whether any replacement occurred.
    """

    pattern = re.compile(
        rf'(<{tag}\s+name="{re.escape(name)}"\s+value=")[^"]*(")',
        flags=re.IGNORECASE,
    )
    updated_text, count = pattern.subn(
        lambda match: f"{match.group(1)}{value}{match.group(2)}",
        text,
    )
    return updated_text, count > 0


def ensure_integrator_property(text: str, tag: str, name: str, value: str) -> str:
    """Ensure the integrator block contains the target property.

    Args:
        text: Original XML text.
        tag: XML tag for the new property element.
        name: Attribute name.
        value: Attribute value.

    Returns:
        Updated XML text.
    """

    updated_text, replaced = replace_value(text, tag, name, value)
    if replaced:
        return updated_text

    integrator_pattern = re.compile(r"(<integrator\b[^>]*>)", flags=re.IGNORECASE)
    inserted_text, count = integrator_pattern.subn(
        lambda match: (
            f'{match.group(1)}\n\t\t<{tag} name="{name}" value="{value}" />'
        ),
        text,
        count=1,
    )
    if count == 0:
        raise Phase12RuntimeSmokeError("scene missing integrator node")
    return inserted_text


def patch_scalar_property(text: str, *, name: str, value: str) -> str:
    """Patch scalar properties across default/integer/float/string nodes.

    Args:
        text: Original XML text.
        name: Property name.
        value: Desired value.

    Returns:
        Updated XML text.
    """

    patched = text
    for tag in ("default", "integer", "float", "string"):
        patched = replace_value(patched, tag, name, value)[0]
    return patched


def patch_boolean_property(text: str, *, name: str, value: str) -> str:
    """Patch boolean properties across default/boolean nodes.

    Args:
        text: Original XML text.
        name: Property name.
        value: Desired value.

    Returns:
        Updated XML text.
    """

    patched = text
    for tag in ("default", "boolean"):
        patched = replace_value(patched, tag, name, value)[0]
    return patched


def patch_scene_text(
    source_text: str,
    *,
    config: RuntimeSmokeConfig,
    runtime_model_value: str,
) -> str:
    """Build patched scene XML text for runtime smoke.

    Args:
        source_text: Original scene XML.
        config: Smoke configuration.
        runtime_model_value: Model path readable in the target runtime.

    Returns:
        Patched scene XML.
    """

    patched = source_text
    scalar_values = {
        "imgwidth": str(config.width),
        "imgheight": str(config.height),
        "crop_offset_x": "0",
        "crop_offset_y": "0",
        "crop_width": str(config.crop_width),
        "crop_height": str(config.crop_height),
        "width": str(config.width),
        "height": str(config.height),
        "sample_count": str(config.sample_count),
        "spp": str(config.sample_count),
        "samples_per_pass": str(config.samples_per_pass),
        "timeout": "0",
    }
    for name, value in scalar_values.items():
        patched = patch_scalar_property(patched, name=name, value=value)

    boolean_values = {
        "train_auto": "false",
        "guide": "false",
        "guided": "false",
        "budget_is_time": "false",
        "selective": "false",
    }
    for name, value in boolean_values.items():
        patched = patch_boolean_property(patched, name=name, value=value)
        # `guide` appears only as `default` in some legacy scenes; do not invent new XML attrs.
        if name != "guide":
            patched = ensure_integrator_property(patched, "boolean", name, value)

    # Inject fallback runtime model via XML string attribute so CLI paths stay explicit.
    patched = patch_scalar_property(
        patched,
        name="masked_diffusion_runtime_path",
        value=runtime_model_value,
    )
    patched = ensure_integrator_property(
        patched,
        "string",
        "masked_diffusion_runtime_path",
        runtime_model_value,
    )
    return patched


def run_runtime_smoke_job(
    *,
    project_root: Path,
    profile: ResolvedRuntimeProfile,
    job: RenderJob,
    config: RuntimeSmokeConfig,
    runtime_model_path: Path,
    output_root: Path,
    runtime_env: dict[str, str],
) -> RuntimeSmokeRecord:
    """Execute one runtime smoke job.

    Args:
        project_root: Repository root.
        profile: Active runtime profile.
        job: Render job.
        config: Smoke configuration.
        runtime_model_path: Exported runtime model path.
        output_root: Output root for this phase.
        runtime_env: Environment for Mitsuba subprocess.

    Returns:
        Smoke record for this job.
    """

    scene_path = project_root / job.scene_path
    scene_text = scene_path.read_text(encoding="utf-8")
    patched_text = patch_scene_text(
        scene_text,
        config=config,
        runtime_model_value=to_profile_path(profile, runtime_model_path),
    )

    scene_record_dir = output_root / "scenes"
    output_dir = output_root / "outputs"
    log_dir = output_root / "logs"
    scene_record_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    patched_scene_record = scene_record_dir / f"{job.job_id}.xml"
    patched_scene_record.write_text(patched_text, encoding="utf-8")

    temp_scene_path = scene_path.with_name(
        f"{scene_path.stem}.phase12_{job.job_id}.xml"
    )
    temp_scene_path.write_text(patched_text, encoding="utf-8")

    output_path = output_dir / f"{job.job_id}.exr"
    stdout_log = log_dir / f"{job.job_id}.stdout.txt"
    stderr_log = log_dir / f"{job.job_id}.stderr.txt"

    command = [
        profile.binary_text
        if profile.command_mode == "wsl"
        else str(profile.binary_path),
        "-m",
        job.variant,
        "-t",
        str(config.threads),
        "-o",
        to_profile_path(profile, output_path),
        to_profile_path(profile, temp_scene_path),
    ]
    process_command = (
        build_wsl_command(profile=profile, command=command)
        if profile.command_mode == "wsl"
        else command
    )

    t0 = time.perf_counter()
    try:
        result = subprocess.run(
            process_command,
            cwd=None if profile.command_mode == "wsl" else project_root,
            env=None if profile.command_mode == "wsl" else runtime_env,
            capture_output=True,
            text=True,
            check=False,
            encoding="utf-8",
            errors="replace",
        )
    finally:
        temp_scene_path.unlink(missing_ok=True)
    dt = time.perf_counter() - t0

    stdout_log.write_text(result.stdout, encoding="utf-8", errors="ignore")
    stderr_log.write_text(result.stderr, encoding="utf-8", errors="ignore")
    combined_output = result.stdout + "\n" + result.stderr

    loaded_runtime_model = (
        "Loaded masked diffusion runtime model from" in combined_output
    )
    disabled_runtime_model = (
        "Disable masked diffusion runtime fallback due to load failure."
        in combined_output
    )
    ok = (
        result.returncode == 0
        and output_path.exists()
        and loaded_runtime_model
        and not disabled_runtime_model
    )

    return RuntimeSmokeRecord(
        job_id=job.job_id,
        scene_path=job.scene_path,
        ok=ok,
        return_code=result.returncode,
        seconds=dt,
        output_path=str(output_path),
        output_exists=output_path.exists(),
        output_size=output_path.stat().st_size if output_path.exists() else 0,
        patched_scene_record=str(patched_scene_record),
        stdout_log=str(stdout_log),
        stderr_log=str(stderr_log),
        command=process_command,
        loaded_runtime_model=loaded_runtime_model,
        disabled_runtime_model=disabled_runtime_model,
        runtime_model_path=str(runtime_model_path),
    )


def main() -> int:
    """Run Phase 12 runtime smoke."""

    configure_logging()
    args = parse_args()
    config = load_config(args.config)
    project_root = WORKSPACE_ROOT.parent

    host = detect_host_runtime(project_root)
    requested_profile_id = normalize_runtime_profile_id(config.profile_id, host)
    if requested_profile_id != config.profile_id:
        LOGGER.info(
            "phase_12 runtime profile remapped: %s -> %s",
            config.profile_id,
            requested_profile_id,
        )
    runtime_profiles = tuple(
        resolve_runtime_profile(template, host)
        for template in load_runtime_profiles(
            WORKSPACE_ROOT / "configs" / "runtime_profiles.json"
        )
    )
    profile = select_runtime_profile(runtime_profiles, requested_profile_id)
    if not profile.ready:
        ensure_runtime_profile_built(project_root, requested_profile_id)
        runtime_profiles = tuple(
            resolve_runtime_profile(template, host)
            for template in load_runtime_profiles(
                WORKSPACE_ROOT / "configs" / "runtime_profiles.json"
            )
        )
        profile = select_runtime_profile(runtime_profiles, requested_profile_id)
    runtime_env = build_runtime_env(profile)

    manifest = load_baseline_manifest(
        WORKSPACE_ROOT / "configs" / "baseline_manifest.json"
    )
    jobs = select_render_jobs(manifest.render_jobs, config.job_ids)
    output_root = WORKSPACE_ROOT / "results" / "phase_12_runtime_smoke"
    records = tuple(
        run_runtime_smoke_job(
            project_root=project_root,
            profile=profile,
            job=job,
            config=config,
            runtime_model_path=resolve_runtime_model_path(
                workspace_root=WORKSPACE_ROOT,
                config=config,
                job=job,
            ),
            output_root=output_root,
            runtime_env=runtime_env,
        )
        for job in jobs
    )

    payload = {
        "config": asdict(config),
        "total": len(records),
        "passed": sum(1 for record in records if record.ok),
        "failed": sum(1 for record in records if not record.ok),
        "records": [asdict(record) for record in records],
    }
    summary_path = WORKSPACE_ROOT / "results" / "phase_12_runtime_smoke_summary.json"
    summary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    LOGGER.info(
        "phase_12 runtime smoke passed: %s / %s",
        payload["passed"],
        payload["total"],
    )
    LOGGER.info("summary_json: %s", summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
