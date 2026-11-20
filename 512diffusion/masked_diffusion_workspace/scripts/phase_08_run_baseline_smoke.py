from __future__ import annotations

import json
import logging
import os
import re
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
)


LOGGER = logging.getLogger("phase_08_run_baseline_smoke")
PHASE08_SMOKE_JOB_IDS = (
    "paper_living_mpg",
    "paper_lamp_mpg",
    "resource_living_mpg",
    "resource_lamp_mpg",
    "resource_living_uniform",
    "resource_lamp_uniform",
)


@dataclass(slots=True, frozen=True)
class SmokeConfig:
    """Shared parameters for baseline smoke runs."""

    profile_id: str
    width: int
    height: int
    crop_width: int
    crop_height: int
    mpg_spp: int
    uniform_spp: int
    samples_per_pass: int
    mpg_timeout_seconds: float
    mpg_threads: int


@dataclass(slots=True, frozen=True)
class SmokeRunRecord:
    """Outcome of one baseline smoke render job."""

    job_id: str
    method: str
    scene_path: str
    variant: str
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
    stree_query_count: int | None
    cd_query_count: int | None
    manifold_query_count: int | None
    manifold_executed_count: int | None


def configure_logging() -> None:
    """Initialize logging."""

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def select_runtime_profile(
    profiles: tuple[ResolvedRuntimeProfile, ...],
    profile_id: str,
) -> ResolvedRuntimeProfile:
    """Pick resolved runtime profile by profile_id."""

    for profile in profiles:
        if profile.profile_id == profile_id:
            return profile
    raise KeyError(f"unknown runtime profile: {profile_id}")


def select_smoke_jobs(
    all_jobs: tuple[RenderJob, ...],
) -> tuple[RenderJob, ...]:
    """Select fixed jobs required by the default Phase 08 smoke."""

    job_map = {job.job_id: job for job in all_jobs}
    selected: list[RenderJob] = []
    missing_job_ids: list[str] = []
    for job_id in PHASE08_SMOKE_JOB_IDS:
        job = job_map.get(job_id)
        if job is None:
            missing_job_ids.append(job_id)
            continue
        selected.append(job)
    if missing_job_ids:
        raise KeyError(f"phase_08 missing smoke jobs: {missing_job_ids}")
    return tuple(selected)


def prepend_env_path(
    base_env: dict[str, str], key: str, entries: tuple[str, ...]
) -> None:
    """Prefix additional entries ahead of PATH/PYTHONPATH-style env vars."""

    current_value = base_env.get(key, "")
    normalized_entries = [entry for entry in entries if entry]
    if current_value:
        normalized_entries.append(current_value)
    base_env[key] = os.pathsep.join(normalized_entries)


def build_runtime_env(profile: ResolvedRuntimeProfile) -> dict[str, str]:
    """Build environment variables for native CLI rendering.

    Args:
        profile: Resolved runtime profile.

    Returns:
        Environment dict usable with subprocess.run.

    Raises:
        RuntimeError: Profile is not a runnable native runtime.
    """

    if profile.command_mode != "native":
        raise RuntimeError(
            f"phase_08 only supports native runtime: {profile.profile_id}"
        )
    if not profile.ready:
        raise RuntimeError(f"runtime profile not ready: {profile.reason}")

    env = os.environ.copy()
    env["MTS_PLUGIN_PATH"] = str(profile.plugin_dir)
    prepend_env_path(env, "PYTHONPATH", (str(profile.python_module_dir),))
    prepend_env_path(env, "PATH", tuple(profile.path_entries))
    return env


def replace_value(text: str, tag: str, name: str, value: str) -> str:
    """Replace the quoted value for a given `<tag name="…">`."""

    pattern = re.compile(
        rf'(<{tag}\s+name="{re.escape(name)}"\s+value=")[^"]*(")',
        flags=re.IGNORECASE,
    )
    return pattern.sub(lambda match: f"{match.group(1)}{value}{match.group(2)}", text)


def patch_scene_text(source_text: str, job: RenderJob, config: SmokeConfig) -> str:
    """Create temporary patched scene XML for smoke.

    Args:
        source_text: Original scene XML.
        job: Current render job.
        config: Smoke parameters.

    Returns:
        Scene XML after smoke patching.
    """

    patched = source_text

    # Shrink resolution and crop so default 1k/2k scenes stay fast.
    size_values = {
        "imgwidth": str(config.width),
        "imgheight": str(config.height),
        "crop_offset_x": "0",
        "crop_offset_y": "0",
        "crop_width": str(config.crop_width),
        "crop_height": str(config.crop_height),
        "width": str(config.width),
        "height": str(config.height),
        "sample_count": str(
            config.mpg_spp if job.method == "mpg" else config.uniform_spp
        ),
        "spp": str(config.mpg_spp if job.method == "mpg" else config.uniform_spp),
        "samples_per_pass": str(config.samples_per_pass),
    }
    for name, value in size_values.items():
        patched = replace_value(patched, "default", name, value)
        patched = replace_value(patched, "integer", name, value)

    if job.method == "mpg":
        bool_values = {
            "train_auto": "true",
            "guide": "true",
            "guided": "true",
            "budget_is_time": "true",
            "selective": "false",
        }
        float_values = {"timeout": str(config.mpg_timeout_seconds)}
    else:
        bool_values = {
            "train_auto": "false",
            "guide": "false",
            "guided": "false",
            "budget_is_time": "false",
            "selective": "false",
        }
        float_values = {"timeout": "0"}

    for name, value in bool_values.items():
        patched = replace_value(patched, "default", name, value)
        patched = replace_value(patched, "boolean", name, value)
    for name, value in float_values.items():
        patched = replace_value(patched, "default", name, value)
        patched = replace_value(patched, "float", name, value)

    return patched


def parse_counter(output_text: str, key: str) -> int | None:
    """Parse integer counters echoed in Mitsuba stdout/stderr."""

    pattern = re.compile(rf"{re.escape(key)}\s+(-?\d+)")
    match = pattern.search(output_text)
    if match is None:
        return None
    return int(match.group(1))


def write_markdown_report(
    output_path: Path,
    *,
    config: SmokeConfig,
    records: tuple[SmokeRunRecord, ...],
) -> None:
    """Write smoke Markdown summary."""

    lines = [
        "# Phase 08 Baseline Smoke",
        "",
        "## Config",
        "",
        f"- profile_id: `{config.profile_id}`",
        f"- width: `{config.width}`",
        f"- height: `{config.height}`",
        f"- crop_width: `{config.crop_width}`",
        f"- crop_height: `{config.crop_height}`",
        f"- mpg_spp: `{config.mpg_spp}`",
        f"- uniform_spp: `{config.uniform_spp}`",
        f"- samples_per_pass: `{config.samples_per_pass}`",
        f"- mpg_timeout_seconds: `{config.mpg_timeout_seconds}`",
        f"- mpg_threads: `{config.mpg_threads}`",
        "",
        "## Runs",
        "",
    ]
    for record in records:
        lines.extend(
            [
                f"### {record.job_id}",
                "",
                f"- ok: `{record.ok}`",
                f"- return_code: `{record.return_code}`",
                f"- seconds: `{record.seconds:.3f}`",
                f"- output_exists: `{record.output_exists}`",
                f"- output_size: `{record.output_size}`",
                f"- stree_query_count: `{record.stree_query_count}`",
                f"- cd_query_count: `{record.cd_query_count}`",
                f"- manifold_query_count: `{record.manifold_query_count}`",
                f"- manifold_executed_count: `{record.manifold_executed_count}`",
                "",
            ]
        )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def run_smoke_job(
    *,
    project_root: Path,
    profile: ResolvedRuntimeProfile,
    job: RenderJob,
    config: SmokeConfig,
    output_root: Path,
    runtime_env: dict[str, str],
) -> SmokeRunRecord:
    """Run one baseline smoke render.

    Args:
        project_root: Repository root.
        profile: Resolved runtime profile.
        job: Current render job.
        config: Smoke parameters.
        output_root: Phase output directory.
        runtime_env: Rendering environment.

    Returns:
        SmokeRunRecord for this job.
    """

    scene_path = project_root / job.scene_path
    scene_text = scene_path.read_text(encoding="utf-8")
    patched_text = patch_scene_text(scene_text, job, config)

    scene_record_dir = output_root / "scenes"
    output_dir = output_root / "outputs"
    log_dir = output_root / "logs"
    scene_record_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    patched_scene_record = scene_record_dir / f"{job.job_id}.xml"
    patched_scene_record.write_text(patched_text, encoding="utf-8")

    temp_scene_path = scene_path.with_name(
        f"{scene_path.stem}.phase08_{job.job_id}.xml"
    )
    temp_scene_path.write_text(patched_text, encoding="utf-8")

    output_path = output_dir / f"{job.job_id}.exr"
    stdout_log = log_dir / f"{job.job_id}.stdout.txt"
    stderr_log = log_dir / f"{job.job_id}.stderr.txt"

    command = [
        str(profile.binary_path),
        "-m",
        job.variant,
    ]
    if job.method == "mpg":
        # Native MPG online training on Windows is stable single-thread only; smoke uses that.
        command.extend(["-t", str(config.mpg_threads)])
    command.extend(["-o", str(output_path), str(temp_scene_path)])

    t0 = time.perf_counter()
    try:
        result = subprocess.run(
            command,
            cwd=project_root,
            env=runtime_env,
            capture_output=True,
            text=True,
            check=False,
        )
    finally:
        temp_scene_path.unlink(missing_ok=True)
    dt = time.perf_counter() - t0

    stdout_log.write_text(result.stdout, encoding="utf-8", errors="ignore")
    stderr_log.write_text(result.stderr, encoding="utf-8", errors="ignore")
    combined_output = result.stdout + "\n" + result.stderr

    return SmokeRunRecord(
        job_id=job.job_id,
        method=job.method,
        scene_path=job.scene_path,
        variant=job.variant,
        ok=result.returncode == 0 and output_path.exists(),
        return_code=result.returncode,
        seconds=dt,
        output_path=str(output_path),
        output_exists=output_path.exists(),
        output_size=output_path.stat().st_size if output_path.exists() else 0,
        patched_scene_record=str(patched_scene_record),
        stdout_log=str(stdout_log),
        stderr_log=str(stderr_log),
        command=command,
        stree_query_count=parse_counter(combined_output, "STree stats perf_num_query"),
        cd_query_count=parse_counter(combined_output, "CD stats perf_num_query"),
        manifold_query_count=parse_counter(
            combined_output, "Manifold perf_manifold_query"
        ),
        manifold_executed_count=parse_counter(
            combined_output, "Manifold perf_manifold_executed"
        ),
    )


def main() -> None:
    """Run a small baseline smoke covering living/lamp with uniform and mpg."""

    configure_logging()

    config = SmokeConfig(
        profile_id="windows_native",
        width=64,
        height=64,
        crop_width=64,
        crop_height=64,
        mpg_spp=999999,
        uniform_spp=16,
        samples_per_pass=1,
        mpg_timeout_seconds=1.0,
        mpg_threads=1,
    )

    project_root = WORKSPACE_ROOT.parent
    results_dir = WORKSPACE_ROOT / "results"
    output_root = results_dir / "phase_08_smoke"

    host = detect_host_runtime(project_root)
    requested_profile_id = normalize_runtime_profile_id(config.profile_id, host)
    if requested_profile_id != config.profile_id:
        LOGGER.info(
            "phase_08 runtime profile remapped: %s -> %s",
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
    smoke_jobs = select_smoke_jobs(manifest.render_jobs)
    records = tuple(
        run_smoke_job(
            project_root=project_root,
            profile=profile,
            job=job,
            config=config,
            output_root=output_root,
            runtime_env=runtime_env,
        )
        for job in smoke_jobs
    )

    payload = {
        "config": asdict(config),
        "total": len(records),
        "passed": sum(1 for record in records if record.ok),
        "failed": sum(1 for record in records if not record.ok),
        "records": [asdict(record) for record in records],
    }

    json_path = results_dir / "phase_08_smoke_summary.json"
    md_path = results_dir / "phase_08_smoke_summary.md"
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_markdown_report(md_path, config=config, records=records)

    LOGGER.info("phase_08 smoke passed: %s / %s", payload["passed"], payload["total"])
    LOGGER.info("json report: %s", json_path)
    LOGGER.info("markdown report: %s", md_path)


if __name__ == "__main__":
    main()
