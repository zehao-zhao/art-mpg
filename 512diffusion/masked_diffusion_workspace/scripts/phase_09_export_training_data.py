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
from typing import Callable, TypeVar


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = WORKSPACE_ROOT.parent
SRC_ROOT = WORKSPACE_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from masked_diffusion_mpg.data_contract import (  # noqa: E402
    DataContractError,
    ReplayRecord,
    make_replay_record,
    make_raw_subpath_sample_key,
    make_replay_record_key,
    read_raw_subpath_samples_jsonl,
    read_replay_records_jsonl,
    raw_subpath_sample_to_jsonable,
    write_jsonl,
    write_replay_records_jsonl,
)
from masked_diffusion_mpg.build import (  # noqa: E402
    ensure_runtime_profile_built,
    normalize_runtime_profile_id,
)
from masked_diffusion_mpg.manifest import (  # noqa: E402
    RenderJob,
    collect_missing_assets,
    load_baseline_manifest,
)
from masked_diffusion_mpg.runtime import (  # noqa: E402
    ResolvedRuntimeProfile,
    detect_host_runtime,
    load_runtime_profiles,
    resolve_runtime_profile,
    to_wsl_path,
)


LOGGER = logging.getLogger("phase_09_export_training_data")
T = TypeVar("T")


class Phase09ExportError(RuntimeError):
    """Describes an expected failure during Phase 09 export."""


@dataclass(slots=True, frozen=True)
class ExportConfig:
    """Describes Phase 09 export configuration."""

    profile_id: str
    job_ids: tuple[str, ...]
    seed_ids: tuple[int, ...]
    width: int
    height: int
    crop_width: int
    crop_height: int
    sample_count: int
    samples_per_pass: int
    threads: int
    timeout_seconds: float
    train_budget: float
    train_fusion: bool
    guided: bool
    allow_partial_aggregate: bool
    min_successful_seeds: int
    min_aggregate_raw_samples: int
    integrator_integer_overrides: dict[str, int]
    integrator_float_overrides: dict[str, float]
    integrator_boolean_overrides: dict[str, bool]
    integrator_string_overrides: dict[str, str]


@dataclass(slots=True, frozen=True)
class ExportRunRecord:
    """Records the result of a single export job."""

    job_id: str
    seed_id: int
    scene_path: str
    ok: bool
    error: str
    return_code: int
    seconds: float
    output_image: str
    raw_samples_path: str
    replay_path: str
    raw_sample_count: int
    replay_record_count: int
    stdout_log: str
    stderr_log: str
    patched_scene_record: str
    command: list[str]


@dataclass(slots=True, frozen=True)
class ExportAggregateRecord:
    """Records aggregated replay results for one task."""

    job_id: str
    seed_ids: tuple[int, ...]
    ok: bool
    error: str
    raw_samples_path: str
    replay_path: str
    raw_sample_count: int
    replay_record_count: int
    raw_duplicate_count: int
    replay_duplicate_count: int


def configure_logging() -> None:
    """Initialize logging."""

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def deduplicate_items(
    items: list[T],
    *,
    key_fn: Callable[[T], str],
) -> tuple[list[T], int]:
    """Deduplicate by stable key, preserving first-seen order."""

    seen: set[str] = set()
    deduplicated: list[T] = []
    duplicate_count = 0
    for item in items:
        item_key = key_fn(item)
        if item_key in seen:
            duplicate_count += 1
            continue
        seen.add(item_key)
        deduplicated.append(item)
    return deduplicated, duplicate_count


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description="Run Phase 09 training data export")
    parser.add_argument(
        "--config",
        type=Path,
        default=WORKSPACE_ROOT / "configs" / "phase_09_export_config.json",
    )
    return parser.parse_args()


def load_export_config(config_path: Path) -> ExportConfig:
    """Load Phase 09 export configuration."""

    payload = json.loads(config_path.read_text(encoding="utf-8"))
    seed_ids = tuple(int(item) for item in payload.get("seed_ids", [2023]))
    allow_partial_aggregate = bool(payload.get("allow_partial_aggregate", False))
    default_min_successful_seeds = 1 if allow_partial_aggregate else len(seed_ids)
    return ExportConfig(
        profile_id=str(payload["profile_id"]),
        job_ids=tuple(str(item) for item in payload["job_ids"]),
        seed_ids=seed_ids,
        width=int(payload["width"]),
        height=int(payload["height"]),
        crop_width=int(payload["crop_width"]),
        crop_height=int(payload["crop_height"]),
        sample_count=int(payload["sample_count"]),
        samples_per_pass=int(payload["samples_per_pass"]),
        threads=int(payload["threads"]),
        timeout_seconds=float(payload["timeout_seconds"]),
        train_budget=float(payload["train_budget"]),
        train_fusion=bool(payload["train_fusion"]),
        guided=bool(payload["guided"]),
        allow_partial_aggregate=allow_partial_aggregate,
        min_successful_seeds=max(
            1, int(payload.get("min_successful_seeds", default_min_successful_seeds))
        ),
        min_aggregate_raw_samples=max(
            1, int(payload.get("min_aggregate_raw_samples", 1))
        ),
        integrator_integer_overrides={
            str(key): int(value)
            for key, value in payload.get("integrator_integer_overrides", {}).items()
        },
        integrator_float_overrides={
            str(key): float(value)
            for key, value in payload.get("integrator_float_overrides", {}).items()
        },
        integrator_boolean_overrides={
            str(key): bool(value)
            for key, value in payload.get("integrator_boolean_overrides", {}).items()
        },
        integrator_string_overrides={
            str(key): str(value)
            for key, value in payload.get("integrator_string_overrides", {}).items()
        },
    )


def select_runtime_profile(
    profiles: tuple[ResolvedRuntimeProfile, ...],
    profile_id: str,
) -> ResolvedRuntimeProfile:
    """Select runtime profile by profile_id."""

    for profile in profiles:
        if profile.profile_id == profile_id:
            return profile
    raise Phase09ExportError(f"unknown runtime profile: {profile_id}")


def select_render_jobs(
    all_jobs: tuple[RenderJob, ...],
    job_ids: tuple[str, ...],
) -> tuple[RenderJob, ...]:
    """Filter target export jobs from the baseline manifest."""

    selected: list[RenderJob] = []
    job_map = {job.job_id: job for job in all_jobs}
    for job_id in job_ids:
        job = job_map.get(job_id)
        if job is None:
            raise Phase09ExportError(f"unknown render job: {job_id}")
        if job.method != "mpg":
            raise Phase09ExportError(f"phase_09 only supports mpg jobs: {job_id}")
        selected.append(job)
    return tuple(selected)


def prepend_env_path(
    base_env: dict[str, str], key: str, entries: tuple[str, ...]
) -> None:
    """Prepend path entries to an environment variable."""

    current_value = base_env.get(key, "")
    normalized_entries = [entry for entry in entries if entry]
    if current_value:
        normalized_entries.append(current_value)
    base_env[key] = os.pathsep.join(normalized_entries)


def build_runtime_env(profile: ResolvedRuntimeProfile) -> dict[str, str]:
    """Build environment variables needed for native CLI runs."""

    if not profile.ready:
        raise Phase09ExportError(f"runtime profile not ready: {profile.reason}")

    if profile.command_mode == "native":
        env = os.environ.copy()
        env["MTS_PLUGIN_PATH"] = str(profile.plugin_dir)
        prepend_env_path(env, "PYTHONPATH", (str(profile.python_module_dir),))
        prepend_env_path(env, "PATH", tuple(profile.path_entries))
        return env

    if profile.command_mode == "wsl":
        return {}

    raise Phase09ExportError(
        f"phase_09 unsupported runtime mode: {profile.command_mode}"
    )


def to_profile_path(profile: ResolvedRuntimeProfile, path: Path) -> str:
    """Convert local paths to paths the target runtime understands."""

    resolved = path.resolve()
    if profile.command_mode == "wsl":
        return to_wsl_path(resolved)
    return str(resolved)


def build_wsl_command(
    *,
    profile: ResolvedRuntimeProfile,
    command: list[str],
) -> list[str]:
    """Wrap export commands for execution under WSL."""

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
    """Replace the value field of the given XML node."""

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
    """Ensure the integrator defines the given property."""

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
        raise Phase09ExportError("scene missing integrator node")
    return inserted_text


def find_scalar_property_value(text: str, *, name: str) -> str | None:
    """Read an existing scalar property value from the scene."""

    for tag in ("default", "integer", "float", "string"):
        pattern = re.compile(
            rf'<{tag}\s+name="{re.escape(name)}"\s+value="([^"]*)"',
            flags=re.IGNORECASE,
        )
        match = pattern.search(text)
        if match is not None:
            return match.group(1)
    return None


def replace_integrator_block(text: str, integrator_block: str) -> str:
    """Replace the integrator block in the scene wholesale."""

    pattern = re.compile(
        r"<integrator\b[^>]*>.*?</integrator>",
        flags=re.IGNORECASE | re.DOTALL,
    )
    replaced_text, count = pattern.subn(
        lambda _: integrator_block,
        text,
        count=1,
    )
    if count == 0:
        raise Phase09ExportError("scene missing integrator block")
    return replaced_text


def build_export_integrator_block(
    *,
    max_depth: str,
    config: ExportConfig,
    export_samples_path: str,
    seed_id: int,
) -> str:
    """Build the MPG integrator snippet used uniformly during export."""

    return (
        '<integrator type="manifold_path_guiding">\n'
        f'\t\t<integer name="seedid" value="{seed_id}" />\n'
        f'\t\t<string name="export_samples_path" value="{export_samples_path}" />\n'
        f'\t\t<float name="train_budget" value="{config.train_budget}" />\n'
        '\t\t<boolean name="train_auto" value="true" />\n'
        f'\t\t<boolean name="guided" value="{"true" if config.guided else "false"}" />\n'
        '\t\t<boolean name="budget_is_time" value="true" />\n'
        f'\t\t<float name="timeout" value="{config.timeout_seconds}" />\n'
        f'\t\t<boolean name="train_fusion" value="{"true" if config.train_fusion else "false"}" />\n'
        f'\t\t<integer name="max_depth" value="{max_depth}" />\n'
        f'\t\t<integer name="samples_per_pass" value="{config.samples_per_pass}" />\n'
        "\t</integrator>"
    )


def patch_scene_text(
    source_text: str,
    *,
    config: ExportConfig,
    export_samples_path: str,
    seed_id: int,
) -> str:
    """Generate temporary scene text for an export job."""

    max_depth = find_scalar_property_value(source_text, name="max_depth") or "15"
    patched = source_text
    if 'type="manifold_path_guiding"' not in source_text:
        patched = replace_integrator_block(
            patched,
            build_export_integrator_block(
                max_depth=max_depth,
                config=config,
                export_samples_path=export_samples_path,
                seed_id=seed_id,
            ),
        )

    default_values = {
        "imgwidth": str(config.width),
        "imgheight": str(config.height),
        "crop_offset_x": "0",
        "crop_offset_y": "0",
        "crop_width": str(config.crop_width),
        "crop_height": str(config.crop_height),
        "spp": str(config.sample_count),
        "samples_per_pass": str(config.samples_per_pass),
        "train_auto": "true",
        "guide": "true" if config.guided else "false",
        "budget_is_time": "true",
        "timeout": str(config.timeout_seconds),
        "selective": "false",
    }
    for name, value in default_values.items():
        patched = replace_value(patched, "default", name, value)[0]

    patched = ensure_integrator_property(
        patched,
        "boolean",
        "train_fusion",
        "true" if config.train_fusion else "false",
    )
    patched = ensure_integrator_property(
        patched, "float", "train_budget", str(config.train_budget)
    )
    patched = ensure_integrator_property(
        patched,
        "string",
        "export_samples_path",
        export_samples_path,
    )
    patched = ensure_integrator_property(
        patched,
        "integer",
        "seedid",
        str(seed_id),
    )
    for name, value in config.integrator_integer_overrides.items():
        patched = ensure_integrator_property(patched, "integer", name, str(value))
    for name, value in config.integrator_float_overrides.items():
        patched = ensure_integrator_property(patched, "float", name, str(value))
    for name, value in config.integrator_boolean_overrides.items():
        patched = ensure_integrator_property(
            patched, "boolean", name, "true" if value else "false"
        )
    for name, value in config.integrator_string_overrides.items():
        patched = ensure_integrator_property(patched, "string", name, value)
    return patched


def write_markdown_report(
    output_path: Path,
    *,
    config: ExportConfig,
    records: tuple[ExportRunRecord, ...],
    aggregate_records: tuple[ExportAggregateRecord, ...],
) -> None:
    """Write the Phase 09 Markdown summary."""

    lines = [
        "# Phase 09 Export Training Data",
        "",
        "## Config",
        "",
        f"- profile_id: `{config.profile_id}`",
        f"- job_ids: `{list(config.job_ids)}`",
        f"- seed_ids: `{list(config.seed_ids)}`",
        f"- width: `{config.width}`",
        f"- height: `{config.height}`",
        f"- crop_width: `{config.crop_width}`",
        f"- crop_height: `{config.crop_height}`",
        f"- sample_count: `{config.sample_count}`",
        f"- samples_per_pass: `{config.samples_per_pass}`",
        f"- threads: `{config.threads}`",
        f"- timeout_seconds: `{config.timeout_seconds}`",
        f"- train_budget: `{config.train_budget}`",
        f"- train_fusion: `{config.train_fusion}`",
        f"- guided: `{config.guided}`",
        f"- allow_partial_aggregate: `{config.allow_partial_aggregate}`",
        f"- min_successful_seeds: `{config.min_successful_seeds}`",
        f"- min_aggregate_raw_samples: `{config.min_aggregate_raw_samples}`",
        f"- integrator_integer_overrides: `{config.integrator_integer_overrides}`",
        f"- integrator_float_overrides: `{config.integrator_float_overrides}`",
        f"- integrator_boolean_overrides: `{config.integrator_boolean_overrides}`",
        f"- integrator_string_overrides: `{config.integrator_string_overrides}`",
        "",
        "## Runs",
        "",
    ]
    for record in records:
        lines.extend(
            [
                f"### {record.job_id} / seed={record.seed_id}",
                "",
                f"- ok: `{record.ok}`",
                f"- error: `{record.error}`",
                f"- return_code: `{record.return_code}`",
                f"- seconds: `{record.seconds:.3f}`",
                f"- raw_sample_count: `{record.raw_sample_count}`",
                f"- replay_record_count: `{record.replay_record_count}`",
                f"- raw_samples_path: `{record.raw_samples_path}`",
                f"- replay_path: `{record.replay_path}`",
                "",
            ]
        )
    lines.extend(["## Aggregates", ""])
    for aggregate in aggregate_records:
        lines.extend(
            [
                f"### {aggregate.job_id}",
                "",
                f"- ok: `{aggregate.ok}`",
                f"- error: `{aggregate.error}`",
                f"- seed_ids: `{list(aggregate.seed_ids)}`",
                f"- raw_sample_count: `{aggregate.raw_sample_count}`",
                f"- replay_record_count: `{aggregate.replay_record_count}`",
                f"- raw_duplicate_count: `{aggregate.raw_duplicate_count}`",
                f"- replay_duplicate_count: `{aggregate.replay_duplicate_count}`",
                f"- raw_samples_path: `{aggregate.raw_samples_path}`",
                f"- replay_path: `{aggregate.replay_path}`",
                "",
            ]
        )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def run_export_job(
    *,
    profile: ResolvedRuntimeProfile,
    config: ExportConfig,
    job: RenderJob,
    seed_id: int,
    output_root: Path,
) -> ExportRunRecord:
    """Run a single export job."""

    source_scene = PROJECT_ROOT / job.scene_path
    missing_assets = collect_missing_assets(source_scene)
    if missing_assets:
        raise Phase09ExportError(f"missing assets: {missing_assets}")

    logs_dir = output_root / "logs"
    outputs_dir = output_root / "outputs"
    raw_dir = output_root / "raw" / "per_seed"
    replay_dir = output_root / "replay" / "per_seed"
    scene_records_dir = output_root / "scenes"
    for directory in (
        logs_dir,
        outputs_dir,
        raw_dir,
        replay_dir,
        scene_records_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    raw_samples_path = raw_dir / f"{job.job_id}.seed{seed_id}.jsonl"
    replay_path = replay_dir / f"{job.job_id}.seed{seed_id}.jsonl"
    output_image = outputs_dir / f"{job.job_id}.seed{seed_id}.exr"
    stdout_log = logs_dir / f"{job.job_id}.seed{seed_id}.stdout.txt"
    stderr_log = logs_dir / f"{job.job_id}.seed{seed_id}.stderr.txt"
    patched_scene_record = scene_records_dir / f"{job.job_id}.seed{seed_id}.xml"
    patched_scene = source_scene.with_name(
        f"{source_scene.stem}.phase09_{job.job_id}.seed{seed_id}.xml"
    )

    source_text = source_scene.read_text(encoding="utf-8")
    patched_text = patch_scene_text(
        source_text,
        config=config,
        export_samples_path=to_profile_path(profile, raw_samples_path),
        seed_id=seed_id,
    )
    patched_scene.write_text(patched_text, encoding="utf-8")
    patched_scene_record.write_text(patched_text, encoding="utf-8")

    env = build_runtime_env(profile)
    command = [
        profile.binary_text
        if profile.command_mode == "wsl"
        else str(profile.binary_path),
        "-m",
        job.variant,
        "-t",
        str(config.threads),
        "-o",
        to_profile_path(profile, output_image),
        to_profile_path(profile, patched_scene),
    ]
    process_command = (
        build_wsl_command(profile=profile, command=command)
        if profile.command_mode == "wsl"
        else command
    )
    LOGGER.info("running export job %s", job.job_id)

    start = time.perf_counter()
    try:
        result = subprocess.run(
            process_command,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=None if profile.command_mode == "wsl" else env,
            cwd=None if profile.command_mode == "wsl" else PROJECT_ROOT,
        )
    finally:
        patched_scene.unlink(missing_ok=True)
    seconds = time.perf_counter() - start
    stdout_log.write_text(result.stdout, encoding="utf-8")
    stderr_log.write_text(result.stderr, encoding="utf-8")

    raw_sample_count = 0
    replay_record_count = 0
    error = ""
    ok = result.returncode == 0 and raw_samples_path.exists()

    if result.returncode != 0:
        error = f"mitsuba exited with code {result.returncode}"
    elif not raw_samples_path.exists():
        error = "missing exported raw samples"
        ok = False
    else:
        try:
            samples = read_raw_subpath_samples_jsonl(raw_samples_path)
            replay_records: tuple[ReplayRecord, ...] = tuple(
                make_replay_record(sample) for sample in samples
            )
            write_replay_records_jsonl(replay_path, replay_records)
            raw_sample_count = len(samples)
            replay_record_count = len(replay_records)
            if raw_sample_count == 0:
                error = "exported raw samples are empty"
                ok = False
        except DataContractError as exc:
            error = str(exc)
            ok = False

    return ExportRunRecord(
        job_id=job.job_id,
        seed_id=seed_id,
        scene_path=job.scene_path,
        ok=ok,
        error=error,
        return_code=result.returncode,
        seconds=seconds,
        output_image=str(output_image),
        raw_samples_path=str(raw_samples_path),
        replay_path=str(replay_path),
        raw_sample_count=raw_sample_count,
        replay_record_count=replay_record_count,
        stdout_log=str(stdout_log),
        stderr_log=str(stderr_log),
        patched_scene_record=str(patched_scene_record),
        command=process_command,
    )


def aggregate_export_runs(
    *,
    job: RenderJob,
    config: ExportConfig,
    records: tuple[ExportRunRecord, ...],
    output_root: Path,
) -> ExportAggregateRecord:
    """Aggregate multi-seed replay data for the same task."""

    if config.allow_partial_aggregate:
        successful_records = tuple(
            record for record in records if record.ok and record.raw_sample_count > 0
        )
    else:
        successful_records = tuple(record for record in records if record.ok)
    aggregate_raw_path = output_root / "raw" / f"{job.job_id}.jsonl"
    aggregate_replay_path = output_root / "replay" / f"{job.job_id}.jsonl"

    if len(successful_records) < config.min_successful_seeds:
        failed_seed_ids = [record.seed_id for record in records if not record.ok]
        return ExportAggregateRecord(
            job_id=job.job_id,
            seed_ids=config.seed_ids,
            ok=False,
            error=(
                "successful seeds below threshold: "
                f"{len(successful_records)} < {config.min_successful_seeds}; "
                f"failed seeds: {failed_seed_ids}"
            ),
            raw_samples_path=str(aggregate_raw_path),
            replay_path=str(aggregate_replay_path),
            raw_sample_count=0,
            replay_record_count=0,
            raw_duplicate_count=0,
            replay_duplicate_count=0,
        )

    raw_samples = []
    replay_records: list[ReplayRecord] = []
    for record in successful_records:
        raw_samples.extend(
            read_raw_subpath_samples_jsonl(Path(record.raw_samples_path))
        )
        replay_records.extend(read_replay_records_jsonl(Path(record.replay_path)))

    if not raw_samples or not replay_records:
        return ExportAggregateRecord(
            job_id=job.job_id,
            seed_ids=config.seed_ids,
            ok=False,
            error="aggregated replay data is empty",
            raw_samples_path=str(aggregate_raw_path),
            replay_path=str(aggregate_replay_path),
            raw_sample_count=0,
            replay_record_count=0,
            raw_duplicate_count=0,
            replay_duplicate_count=0,
        )

    deduplicated_raw_samples, raw_duplicate_count = deduplicate_items(
        raw_samples,
        key_fn=make_raw_subpath_sample_key,
    )
    deduplicated_replay_records, replay_duplicate_count = deduplicate_items(
        replay_records,
        key_fn=make_replay_record_key,
    )

    if (
        len(deduplicated_raw_samples) < config.min_aggregate_raw_samples
        or len(deduplicated_replay_records) < config.min_aggregate_raw_samples
    ):
        return ExportAggregateRecord(
            job_id=job.job_id,
            seed_ids=config.seed_ids,
            ok=False,
            error=(
                "aggregated replay data below threshold: "
                f"{len(deduplicated_replay_records)} < "
                f"{config.min_aggregate_raw_samples}"
            ),
            raw_samples_path=str(aggregate_raw_path),
            replay_path=str(aggregate_replay_path),
            raw_sample_count=len(deduplicated_raw_samples),
            replay_record_count=len(deduplicated_replay_records),
            raw_duplicate_count=raw_duplicate_count,
            replay_duplicate_count=replay_duplicate_count,
        )

    write_jsonl(
        aggregate_raw_path,
        tuple(
            raw_subpath_sample_to_jsonable(sample)
            for sample in deduplicated_raw_samples
        ),
    )
    write_replay_records_jsonl(
        aggregate_replay_path,
        tuple(deduplicated_replay_records),
    )
    return ExportAggregateRecord(
        job_id=job.job_id,
        seed_ids=config.seed_ids,
        ok=True,
        error="",
        raw_samples_path=str(aggregate_raw_path),
        replay_path=str(aggregate_replay_path),
        raw_sample_count=len(deduplicated_raw_samples),
        replay_record_count=len(deduplicated_replay_records),
        raw_duplicate_count=raw_duplicate_count,
        replay_duplicate_count=replay_duplicate_count,
    )


def main() -> int:
    """Run Phase 09 training data export."""

    configure_logging()
    args = parse_args()
    config_path = args.config
    manifest_path = WORKSPACE_ROOT / "configs" / "baseline_manifest.json"
    runtime_path = WORKSPACE_ROOT / "configs" / "runtime_profiles.json"
    output_root = WORKSPACE_ROOT / "results" / "phase_09_export"

    export_config = load_export_config(config_path)
    manifest = load_baseline_manifest(manifest_path)
    host = detect_host_runtime(PROJECT_ROOT)
    requested_profile_id = normalize_runtime_profile_id(export_config.profile_id, host)
    if requested_profile_id != export_config.profile_id:
        LOGGER.info(
            "phase_09 runtime profile remapped: %s -> %s",
            export_config.profile_id,
            requested_profile_id,
        )
    profiles = tuple(
        resolve_runtime_profile(template, host)
        for template in load_runtime_profiles(runtime_path)
    )
    profile = select_runtime_profile(profiles, requested_profile_id)
    if not profile.ready:
        ensure_runtime_profile_built(PROJECT_ROOT, requested_profile_id)
        profiles = tuple(
            resolve_runtime_profile(template, host)
            for template in load_runtime_profiles(runtime_path)
        )
        profile = select_runtime_profile(profiles, requested_profile_id)
    jobs = select_render_jobs(manifest.render_jobs, export_config.job_ids)

    records: list[ExportRunRecord] = []
    aggregate_records: list[ExportAggregateRecord] = []
    for job in jobs:
        job_records: list[ExportRunRecord] = []
        for seed_id in export_config.seed_ids:
            try:
                record = run_export_job(
                    profile=profile,
                    config=export_config,
                    job=job,
                    seed_id=seed_id,
                    output_root=output_root,
                )
            except Phase09ExportError as exc:
                LOGGER.error(
                    "phase_09 export failed for %s / seed=%s: %s",
                    job.job_id,
                    seed_id,
                    exc,
                )
                record = ExportRunRecord(
                    job_id=job.job_id,
                    seed_id=seed_id,
                    scene_path=job.scene_path,
                    ok=False,
                    error=str(exc),
                    return_code=-1,
                    seconds=0.0,
                    output_image=str(
                        output_root / "outputs" / f"{job.job_id}.seed{seed_id}.exr"
                    ),
                    raw_samples_path=str(
                        output_root
                        / "raw"
                        / "per_seed"
                        / f"{job.job_id}.seed{seed_id}.jsonl"
                    ),
                    replay_path=str(
                        output_root
                        / "replay"
                        / "per_seed"
                        / f"{job.job_id}.seed{seed_id}.jsonl"
                    ),
                    raw_sample_count=0,
                    replay_record_count=0,
                    stdout_log=str(
                        output_root / "logs" / f"{job.job_id}.seed{seed_id}.stdout.txt"
                    ),
                    stderr_log=str(
                        output_root / "logs" / f"{job.job_id}.seed{seed_id}.stderr.txt"
                    ),
                    patched_scene_record=str(
                        output_root / "scenes" / f"{job.job_id}.seed{seed_id}.xml"
                    ),
                    command=[],
                )
            job_records.append(record)
            records.append(record)
        aggregate_records.append(
            aggregate_export_runs(
                job=job,
                config=export_config,
                records=tuple(job_records),
                output_root=output_root,
            )
        )

    output_root.mkdir(parents=True, exist_ok=True)
    summary_json = output_root.with_name("phase_09_export_summary.json")
    summary_md = output_root.with_name("phase_09_export_summary.md")
    summary_payload = {
        "config": asdict(export_config),
        "total": len(records),
        "passed": sum(1 for record in records if record.ok),
        "failed": sum(1 for record in records if not record.ok),
        "records": [asdict(record) for record in records],
        "aggregate_records": [asdict(record) for record in aggregate_records],
        "aggregate_passed": sum(1 for record in aggregate_records if record.ok),
        "aggregate_failed": sum(1 for record in aggregate_records if not record.ok),
        "partial_seed_failures_allowed": export_config.allow_partial_aggregate,
    }
    summary_json.write_text(
        json.dumps(summary_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_markdown_report(
        summary_md,
        config=export_config,
        records=tuple(records),
        aggregate_records=tuple(aggregate_records),
    )
    LOGGER.info(
        "phase_09 export finished: passed=%s failed=%s aggregate_passed=%s aggregate_failed=%s",
        summary_payload["passed"],
        summary_payload["failed"],
        summary_payload["aggregate_passed"],
        summary_payload["aggregate_failed"],
    )
    if summary_payload["aggregate_failed"] != 0:
        return 1
    if export_config.allow_partial_aggregate:
        return 0
    return 0 if summary_payload["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
