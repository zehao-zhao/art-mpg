from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import os
import random
import re
import shlex
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = WORKSPACE_ROOT.parent
SRC_ROOT = WORKSPACE_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from masked_diffusion_mpg.manifest import (  # noqa: E402
    RenderJob,
    collect_missing_assets,
    load_baseline_manifest,
)
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


LOGGER = logging.getLogger("phase_13_run_formal_benchmark")

NPM_ADAPTED_METHODS = {
    "npm_adapted",
    "npm_adapted_seed_direction",
}

RUNTIME_MODEL_METHODS = {
    "masked_runtime",
    "masked_diffusion",
    "masked_diffusion_vmf",
    "masked_hybrid",
    "simple_mlp",
    "simple_mlp_k4",
    *NPM_ADAPTED_METHODS,
    "diffusion_mlp",
    "diffusion_transformer",
    "ordered_diffusion_mlp",
    "ordered_diffusion_transformer",
    "ordered_diffusion_dit",
    "ordered_diffusion_mlp_h72_k1",
    "ordered_diffusion_mlp_h72_k4",
    "ordered_diffusion_mlp_desc_q95",
    "ordered_diffusion_mlp_desc_q99",
    "ordered_diffusion_mlp_dir_q95",
    "ordered_diffusion_mlp_dir_q99",
}


# Original SMS in the paper fixes chain length/type; only Lamp and Double Slabs
# ship presets here that faithfully mirror that hypothesis.
SMS_SCENE_PRESETS: dict[str, dict[str, int | str]] = {
    "paper_lamp": {"tau": 6, "bounce": 2, "chain": "RT"},
    "resource_lamp": {"tau": 6, "bounce": 2, "chain": "RT"},
    "paper_slabs": {"tau": 31, "bounce": 4, "chain": "TTTT"},
}


class Phase13BenchmarkError(RuntimeError):
    """Known failure modes for Phase 13 benchmarking."""


@dataclass(slots=True, frozen=True)
class Phase13BenchmarkConfig:
    """Phase 13 benchmark settings.

    Attributes:
        profile_id: Runtime launcher profile slug.
        job_ids: Render jobs queued for this run.
        width: Output width before crop.
        height: Output height before crop.
        crop_width: Final crop width.
        crop_height: Final crop height.
        reference_sample_count: Path tracer reference SPP budget.
        compare_sample_count: Default SPP budget for comparative methods.
        samples_per_pass: Samples emitted per Mitsuba pass.
        threads: Renderer thread budget.
        seed: Seeds Python RNG, Mitsuba's sampler, and MPG XML knobs.
        mpg_train_budget: Fraction of render time earmarked for online MPG training.
        mpg_train_fusion: Whether MPG fuses streamed training shards.
        runtime_model_path: Relative path for exported runtime text checkpoint.
        bounce_alpha: Geometric share of the runtime chain-length proposal.
        learned_direction_kappa_scale: Multiplier for learned vMF concentrations.
        learned_direction_flip_to_receiver_hemisphere: Apply the legacy normal fold.
        external_reference_manifest: Ready shared-reference manifest consumed by a
            candidate-only run.
        shared_reference_manifest_output: Stable ready-manifest path written by a
            successful reference-only run.
        defer_metrics: Preserve render outputs and hashes without reading images or
            computing MSE/PSNR until an explicit post-barrier finalization step.
        output_dir: Benchmark artifact directory (repo-relative path text).
        methods: Rendering methods exercised this run.
    """

    profile_id: str
    job_ids: tuple[str, ...]
    width: int
    height: int
    crop_width: int
    crop_height: int
    reference_sample_count: int
    compare_sample_count: int
    samples_per_pass: int
    threads: int
    seed: int
    mpg_train_budget: float
    masked_diffusion_train_budget: float | None
    mpg_train_fusion: bool
    output_dir: str
    methods: tuple[str, ...]
    runtime_model_path: str | None = None
    runtime_model_paths: dict[str, str] | None = None
    bounce_alpha: float = 0.5
    learned_direction_kappa_scale: float = 1.0
    learned_direction_flip_to_receiver_hemisphere: bool = True
    masked_diffusion_prob_uniform: float | None = None
    masked_diffusion_prob_uniform_train: float | None = None
    learned_proposal_inv_prob_clamp: float | None = None
    learned_proposal_contribution_clamp: float | None = None
    learned_proposal_max_trials: int | None = None
    masked_diffusion_spatial_filter: float | None = None
    masked_diffusion_ablation: int | None = None
    runtime_only_methods: tuple[str, ...] = ()
    method_compare_sample_counts: dict[str, int] | None = None
    scene_method_compare_sample_counts: dict[str, int] | None = None
    time_budget_seconds: float | None = None
    fixed_spp: int | None = None
    postprocess_crop_from_full_exr: bool = False
    diffusion_steps: int = 1
    reference_seed: int | None = None
    external_reference_manifest: str | None = None
    shared_reference_manifest_output: str | None = None
    defer_metrics: bool = False


@dataclass(slots=True, frozen=True)
class ResolvedSharedReference:
    """One immutable external reference resolved from a ready manifest."""

    scene_id: str
    exr_path: Path
    exr_sha256: str
    output_size: int
    sample_count: int
    sampler_seed: int
    active_config_sha256: str
    width: int
    height: int
    crop_width: int
    crop_height: int


@dataclass(slots=True, frozen=True)
class ResolvedSharedReferenceBundle:
    """Validated shared-reference manifest plus selected scene entries."""

    manifest_path: Path
    manifest_sha256: str
    source_summary_path: Path
    source_summary_sha256: str
    references: dict[str, ResolvedSharedReference]


@dataclass(slots=True, frozen=True)
class BenchmarkRecord:
    """One scene/method rasterization outcome.

    Attributes:
        scene_id: Scene/job identifier token.
        scene_path: Canonical scene XML relative path text.
        method: Rendering method slug.
        ok: Mitsuba finished without fatal errors.
        return_code: Subprocess exit code.
        seconds: Wall-clock elapsed time.
        sample_count: Renderer sample throughput for this combo.
        sampler_seed: Mitsuba sampler base seed written into the scene XML.
        active_method: Method identity used to construct the integrator block.
        active_config_sha256: SHA-256 of the exact patched XML sent to Mitsuba.
        output_path: EXR relative path emitted by Mitsuba.
        output_exists: EXR reachable on disk.
        output_size: Bytes on disk (0 when missing).
        output_sha256: SHA-256 of the completed EXR, or null on failure.
        nonfinite_pixel_boolean: False only after candidate pixels were decoded
            and verified finite; True for detected NaN/Inf; null if unavailable.
        patched_scene_record: Temporary XML patched for integrator swaps.
        stdout_log: Captured Mitsuba stdout.
        stderr_log: Captured Mitsuba stderr.
        loaded_runtime_model: Runtime text successfully parsed.
        mse: MSE versus path tracer reference raster.
        psnr: Peak-SNR computed with reference radiance normalization.
        max_value: HDR peak reused for PSNR denominator.
        command: Executable argv list echoed for debugging.
    """

    scene_id: str
    scene_path: str
    method: str
    ok: bool
    return_code: int
    seconds: float
    sample_count: int
    sampler_seed: int
    active_method: str
    active_config_sha256: str
    output_path: str
    output_exists: bool
    output_size: int
    output_sha256: str | None
    patched_scene_record: str
    stdout_log: str
    stderr_log: str
    loaded_runtime_model: bool
    mse: float | None
    psnr: float | None
    max_value: float | None
    runtime_model_path: str | None
    command: list[str]
    nonfinite_pixel_boolean: bool | None = None


def configure_logging() -> None:
    """Configure root logger verbosity."""

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def parse_args() -> argparse.Namespace:
    """CLI entrypoint."""

    parser = argparse.ArgumentParser(description="Run Phase 13 formal benchmark suite")
    parser.add_argument(
        "--config",
        type=Path,
        default=WORKSPACE_ROOT / "configs" / "phase_13_formal_benchmark.json",
        help="Path to Phase 13 JSON config.",
    )
    parser.add_argument(
        "--bounce-alpha",
        type=float,
        default=None,
        help=(
            "Override the geometric share of the runtime length proposal mixture "
            "(inclusive range [0, 1]; default comes from config or 0.5)."
        ),
    )
    parser.add_argument(
        "--learned-direction-kappa-scale",
        type=float,
        default=None,
        help=(
            "Override the runtime vMF concentration multiplier "
            "(finite and nonnegative; default comes from config or 1.0)."
        ),
    )
    parser.add_argument(
        "--learned-direction-flip-to-receiver-hemisphere",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Enable or disable the legacy receiver-normal hemisphere fold for "
            "learned runtime directions (default comes from config or true)."
        ),
    )
    return parser.parse_args()


def set_global_seed(seed: int) -> None:
    """Seed Python RNGs for repeatable CI-style runs.

    Args:
        seed: Integer RNG seed propagated into Mitsuba knobs.
    """

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def create_run_dir(output_root: Path) -> Path:
    """Materialize dated run subdirectory under benchmark root.

    Args:
        output_root: Repo-relative benchmark output root resolved to pathlib.

    Returns:
        Fresh run directory accepting EXR/logs.
    """

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_root / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def relativize_export_path(path_text: str | None, repo_root: Path) -> str | None:
    """Coerce exporter paths into repo-relative POSIX fragments.

    Args:
        path_text: Raw filesystem literal from Mitsuba/logs.
        repo_root: Monorepo root.

    Returns:
        Relative posix string; falls back to original text when relativization fails.
    """

    if path_text is None:
        return None
    path = Path(path_text)
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except (OSError, RuntimeError, ValueError):
        return path.as_posix()


def serialize_record(record: BenchmarkRecord, repo_root: Path) -> dict[str, object]:
    """Convert BenchmarkRecord paths to portable JSON payloads.

    Args:
        record: In-memory telemetry row for one render.
        repo_root: Monorepo root for relativization targets.

    Returns:
        Serializable dict honoring JSON/CSV export schema.
    """

    payload = asdict(record)
    for key in (
        "scene_path",
        "output_path",
        "patched_scene_record",
        "stdout_log",
        "stderr_log",
        "runtime_model_path",
    ):
        payload[key] = relativize_export_path(
            None if payload.get(key) is None else str(payload[key]),
            repo_root,
        )
    return payload


def validate_bounce_alpha(value: object) -> float:
    """Return a finite runtime length-mixture weight in the closed unit interval."""

    try:
        bounce_alpha = float(value)
    except (TypeError, ValueError) as exc:
        raise Phase13BenchmarkError(
            f"bounce_alpha must be a finite float in [0, 1], got {value!r}"
        ) from exc
    if not math.isfinite(bounce_alpha) or not 0.0 <= bounce_alpha <= 1.0:
        raise Phase13BenchmarkError(
            f"bounce_alpha must be a finite float in [0, 1], got {value!r}"
        )
    return bounce_alpha


def validate_learned_direction_kappa_scale(value: object) -> float:
    """Return a finite, nonnegative runtime vMF concentration multiplier."""

    try:
        kappa_scale = float(value)
    except (TypeError, ValueError) as exc:
        raise Phase13BenchmarkError(
            "learned_direction_kappa_scale must be finite and nonnegative, "
            f"got {value!r}"
        ) from exc
    if not math.isfinite(kappa_scale) or kappa_scale < 0.0:
        raise Phase13BenchmarkError(
            "learned_direction_kappa_scale must be finite and nonnegative, "
            f"got {value!r}"
        )
    return kappa_scale


def validate_boolean(value: object, *, label: str) -> bool:
    """Require a real JSON boolean rather than accepting truthy scalars."""

    if not isinstance(value, bool):
        raise Phase13BenchmarkError(f"{label} must be a boolean, got {value!r}")
    return value


def validate_learned_direction_flip(value: object) -> bool:
    """Require a real JSON boolean for the learned-direction hemisphere fold."""

    return validate_boolean(
        value,
        label="learned_direction_flip_to_receiver_hemisphere",
    )


def validate_reference_mode(config: Phase13BenchmarkConfig) -> None:
    """Reject ambiguous shared-reference producer/consumer configurations."""

    external = config.external_reference_manifest
    output = config.shared_reference_manifest_output
    if external is not None and not external.strip():
        raise Phase13BenchmarkError("external_reference_manifest must be non-empty")
    if output is not None and not output.strip():
        raise Phase13BenchmarkError(
            "shared_reference_manifest_output must be non-empty"
        )
    if external is not None and output is not None:
        raise Phase13BenchmarkError(
            "external_reference_manifest and shared_reference_manifest_output "
            "are mutually exclusive"
        )
    if external is not None and "path_reference" in config.methods:
        raise Phase13BenchmarkError(
            "external-reference runs must be candidate-only; remove path_reference"
        )
    if output is not None and config.methods != ("path_reference",):
        raise Phase13BenchmarkError(
            "shared-reference production requires methods=['path_reference']"
        )
    if (external is not None or output is not None) and config.reference_seed is None:
        raise Phase13BenchmarkError(
            "shared-reference runs require an explicit reference_seed"
        )
    if config.reference_seed is not None and config.reference_seed < 0:
        raise Phase13BenchmarkError("reference_seed must be non-negative")
    if external is not None or output is not None:
        if not config.job_ids:
            raise Phase13BenchmarkError("shared-reference runs require at least one job")
        if len(config.job_ids) != len(set(config.job_ids)):
            raise Phase13BenchmarkError("shared-reference job_ids must be unique")
        if not config.methods:
            raise Phase13BenchmarkError("shared-reference runs require at least one method")
        if len(config.methods) != len(set(config.methods)):
            raise Phase13BenchmarkError("shared-reference methods must be unique")
        if config.reference_sample_count <= 0:
            raise Phase13BenchmarkError(
                "shared-reference reference_sample_count must be positive"
            )
        if min(config.width, config.height, config.crop_width, config.crop_height) <= 0:
            raise Phase13BenchmarkError(
                "shared-reference image and crop dimensions must be positive"
            )
    if config.defer_metrics and external is None:
        raise Phase13BenchmarkError(
            "defer_metrics requires external_reference_manifest"
        )


def load_config(config_path: Path) -> Phase13BenchmarkConfig:
    """Parse Phase 13 JSON configuration into typed dataclass.

    Args:
        config_path: User-provided Phase 13 config file.

    Returns:
        Hydrated Phase13BenchmarkConfig instance.
    """

    payload = json.loads(config_path.read_text(encoding="utf-8"))
    config = Phase13BenchmarkConfig(
        profile_id=str(payload["profile_id"]),
        job_ids=tuple(str(item) for item in payload["job_ids"]),
        width=int(payload["width"]),
        height=int(payload["height"]),
        crop_width=int(payload["crop_width"]),
        crop_height=int(payload["crop_height"]),
        reference_sample_count=int(payload["reference_sample_count"]),
        compare_sample_count=int(payload["compare_sample_count"]),
        samples_per_pass=int(payload["samples_per_pass"]),
        threads=int(payload["threads"]),
        seed=int(payload["seed"]),
        mpg_train_budget=float(payload["mpg_train_budget"]),
        masked_diffusion_train_budget=(
            None
            if payload.get("masked_diffusion_train_budget") is None
            else float(payload["masked_diffusion_train_budget"])
        ),
        mpg_train_fusion=bool(payload["mpg_train_fusion"]),
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
        bounce_alpha=validate_bounce_alpha(payload.get("bounce_alpha", 0.5)),
        learned_direction_kappa_scale=validate_learned_direction_kappa_scale(
            payload.get("learned_direction_kappa_scale", 1.0)
        ),
        learned_direction_flip_to_receiver_hemisphere=(
            validate_learned_direction_flip(
                payload.get(
                    "learned_direction_flip_to_receiver_hemisphere",
                    True,
                )
            )
        ),
        masked_diffusion_prob_uniform=(
            None
            if payload.get("masked_diffusion_prob_uniform") is None
            else float(payload["masked_diffusion_prob_uniform"])
        ),
        masked_diffusion_prob_uniform_train=(
            None
            if payload.get("masked_diffusion_prob_uniform_train") is None
            else float(payload["masked_diffusion_prob_uniform_train"])
        ),
        learned_proposal_inv_prob_clamp=(
            None
            if payload.get("learned_proposal_inv_prob_clamp") is None
            else float(payload["learned_proposal_inv_prob_clamp"])
        ),
        learned_proposal_contribution_clamp=(
            None
            if payload.get("learned_proposal_contribution_clamp") is None
            else float(payload["learned_proposal_contribution_clamp"])
        ),
        learned_proposal_max_trials=(
            None
            if payload.get("learned_proposal_max_trials") is None
            else int(payload["learned_proposal_max_trials"])
        ),
        masked_diffusion_spatial_filter=(
            None
            if payload.get("masked_diffusion_spatial_filter") is None
            else float(payload["masked_diffusion_spatial_filter"])
        ),
        masked_diffusion_ablation=(
            None
            if payload.get("masked_diffusion_ablation") is None
            else int(payload["masked_diffusion_ablation"])
        ),
        runtime_only_methods=tuple(
            str(item) for item in payload.get("runtime_only_methods", [])
        ),
        method_compare_sample_counts=(
            None
            if payload.get("method_compare_sample_counts") is None
            else {
                str(key): int(value)
                for key, value in dict(payload["method_compare_sample_counts"]).items()
            }
        ),
        scene_method_compare_sample_counts=(
            None
            if payload.get("scene_method_compare_sample_counts") is None
            else {
                str(key): int(value)
                for key, value in dict(
                    payload["scene_method_compare_sample_counts"]
                ).items()
            }
        ),
        time_budget_seconds=(
            None
            if payload.get("time_budget_seconds") is None
            else float(payload["time_budget_seconds"])
        ),
        fixed_spp=(
            None if payload.get("fixed_spp") is None else int(payload["fixed_spp"])
        ),
        postprocess_crop_from_full_exr=bool(
            payload.get("postprocess_crop_from_full_exr", False)
        ),
        diffusion_steps=max(1, int(payload.get("diffusion_steps", 1))),
        reference_seed=(
            None
            if payload.get("reference_seed") is None
            else int(payload["reference_seed"])
        ),
        external_reference_manifest=(
            None
            if payload.get("external_reference_manifest") is None
            else str(payload["external_reference_manifest"])
        ),
        shared_reference_manifest_output=(
            None
            if payload.get("shared_reference_manifest_output") is None
            else str(payload["shared_reference_manifest_output"])
        ),
        defer_metrics=validate_boolean(
            payload.get("defer_metrics", False), label="defer_metrics"
        ),
        output_dir=str(payload["output_dir"]),
        methods=tuple(
            str(item)
            for item in payload.get(
                "methods",
                [
                    "path_reference",
                    "pt",
                    "sms_star",
                    "mpg",
                    "masked_diffusion",
                    "masked_diffusion_vmf",
                    "simple_mlp",
                ],
            )
        ),
    )
    validate_reference_mode(config)
    return config


def resolve_runtime_model_path(
    *,
    workspace_root: Path,
    config: Phase13BenchmarkConfig,
    job: RenderJob,
    method: str,
) -> Path:
    """Pick neural runtime checkpoints honoring per-job overrides."""

    candidate_keys = (
        f"{job.job_id}|{method}",
        f"{job.job_id.removesuffix('_mpg')}|{method}",
        method,
        job.job_id,
        job.job_id.removesuffix("_mpg"),
    )
    if config.runtime_model_paths is not None:
        for candidate_key in candidate_keys:
            runtime_path = config.runtime_model_paths.get(candidate_key)
            if runtime_path is None:
                continue
            resolved_path = (workspace_root / runtime_path).resolve()
            if resolved_path.exists():
                return resolved_path
            LOGGER.warning(
                "runtime model missing for %s, fallback to general runtime: %s",
                candidate_key,
                resolved_path,
            )

    if config.runtime_model_path is None:
        raise Phase13BenchmarkError("runtime_model_path is missing")
    resolved_path = (workspace_root / config.runtime_model_path).resolve()
    if not resolved_path.exists():
        raise Phase13BenchmarkError(f"runtime model missing: {resolved_path}")
    return resolved_path


def method_requires_runtime_model(method: str) -> bool:
    """Whether method depends on exported runtime text blobs."""

    return method in RUNTIME_MODEL_METHODS


def select_runtime_profile(
    profiles: tuple[ResolvedRuntimeProfile, ...],
    profile_id: str,
) -> ResolvedRuntimeProfile:
    """Fetch resolved ResolvedRuntimeProfile by ID.

    Args:
        profiles: Tuple returned from load_runtime_profiles.
        profile_id: User-selected launcher profile.

    Returns:
        Matching ResolvedRuntimeProfile.
    """

    for profile in profiles:
        if profile.profile_id == profile_id:
            return profile
    raise Phase13BenchmarkError(f"unknown runtime profile: {profile_id}")


def select_render_jobs(
    all_jobs: tuple[RenderJob, ...],
    job_ids: tuple[str, ...],
) -> tuple[RenderJob, ...]:
    """Intersect manifest entries with CLI job whitelist.

    Args:
        all_jobs: Canonical jobs loaded from baseline manifest.
        job_ids: Ordered whitelist of MPG reference jobs.

    Returns:
        Tuple of validated RenderJob objects ready for patching.
    """

    job_map = {job.job_id: job for job in all_jobs}
    selected: list[RenderJob] = []
    for job_id in job_ids:
        job = job_map.get(job_id)
        if job is None:
            raise Phase13BenchmarkError(f"unknown render job: {job_id}")
        if job.method != "mpg":
            raise Phase13BenchmarkError(
                f"phase_13 requires mpg source scenes for patching: {job_id}"
            )
        selected.append(job)
    return tuple(selected)


def prepend_env_path(
    base_env: dict[str, str], key: str, entries: tuple[str, ...]
) -> None:
    """Prefix PATH-like env vars without duplicating separators.

    Args:
        base_env: Destination env dict mutated in place.
        key: Existing env-var name (`PATH`, `PYTHONPATH`, ...).
        entries: Entries prepended newest-first before existing suffix.
    """

    current_value = base_env.get(key, "")
    normalized_entries = [entry for entry in entries if entry]
    if current_value:
        normalized_entries.append(current_value)
    base_env[key] = os.pathsep.join(normalized_entries)


def build_runtime_env(profile: ResolvedRuntimeProfile) -> dict[str, str]:
    """Produce subprocess env overrides for Mitsuba.

    Args:
        profile: Host-resolved Mitsuba toolchain metadata.

    Returns:
        Mapping suitable for subprocess.Popen(env=...).
    """

    if not profile.ready:
        raise Phase13BenchmarkError(f"runtime profile not ready: {profile.reason}")

    if profile.command_mode == "native":
        env = os.environ.copy()
        env["MTS_PLUGIN_PATH"] = str(profile.plugin_dir)
        prepend_env_path(env, "PYTHONPATH", (str(profile.python_module_dir),))
        prepend_env_path(env, "PATH", tuple(profile.path_entries))
        if profile.target_platform == "darwin":
            prepend_env_path(env, "DYLD_LIBRARY_PATH", tuple(profile.library_entries))
        elif profile.target_platform == "linux":
            prepend_env_path(env, "LD_LIBRARY_PATH", tuple(profile.library_entries))
        return env

    if profile.command_mode == "wsl":
        return {}

    raise Phase13BenchmarkError(f"unsupported runtime mode: {profile.command_mode}")


def to_profile_path(profile: ResolvedRuntimeProfile, path: Path) -> str:
    """Translate filesystem paths for native vs WSL launchers."""

    resolved = path.resolve()
    if profile.command_mode == "wsl":
        return to_wsl_path(resolved)
    return str(resolved)


def build_wsl_command(
    *,
    profile: ResolvedRuntimeProfile,
    command: list[str],
) -> list[str]:
    """Wrap argv list inside `wsl bash -lc` indirection."""

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
    """Mutate Mitsuba `<tag name=...>` XML attribute pairs.

    Args:
        text: Serialized scene XML.
        tag: Element tag (`integer`, ...).
        name: Mitsuba parameter name slug.
        value: Replacement textual value.

    Returns:
        Possibly updated XML plus bool indicating replacement happened.
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


def patch_scalar_property(text: str, *, name: str, value: str) -> str:
    """Bulk-edit Mitsuba `<default|integer|float|string>` nodes sharing a name.

    Args:
        text: Scene XML text.
        name: Parameter name attribute.
        value: Replacement string literal.

    Returns:
        Possibly edited XML text.
    """

    patched = text
    for tag in ("default", "integer", "float", "string"):
        patched = replace_value(patched, tag, name, value)[0]
    return patched


def patch_sampler_seed(text: str, *, sampler_seed: int) -> str:
    """Set the base RNG seed on every concrete Mitsuba sampler block.

    The MPG ``seedid`` integrator property controls deterministic MPG data-
    structure construction, not the renderer's sample stream. Mitsuba samplers
    read their RNG offset from an ``<integer name="seed" ...>`` child, so this
    edit must be scoped to ``<sampler>`` blocks rather than applied globally.

    Args:
        text: Scene XML after ordinary scalar substitutions.
        sampler_seed: Non-negative Mitsuba sampler base seed.

    Returns:
        XML with an existing sampler seed replaced or a new one inserted.
    """

    if sampler_seed < 0:
        raise Phase13BenchmarkError("sampler seed must be non-negative")

    sampler_pattern = re.compile(
        r"(?P<open><sampler\b[^>]*>)(?P<body>.*?)(?P<close></sampler>)",
        flags=re.IGNORECASE | re.DOTALL,
    )

    def rewrite_sampler(match: re.Match[str]) -> str:
        sampler_block = match.group(0)
        replaced_block, replaced = replace_value(
            sampler_block,
            "integer",
            "seed",
            str(sampler_seed),
        )
        if replaced:
            return replaced_block

        body = match.group("body")
        newline = "\r\n" if "\r\n" in sampler_block else "\n"
        child_indent_match = re.search(r"(?:\r?\n)([ \t]+)<", body)
        child_indent = child_indent_match.group(1) if child_indent_match else "\t"
        seed_node = (
            f'{newline}{child_indent}<integer name="seed" '
            f'value="{sampler_seed}" />'
        )
        return (
            match.group("open")
            + seed_node
            + body
            + match.group("close")
        )

    patched, sampler_count = sampler_pattern.subn(rewrite_sampler, text)
    if sampler_count == 0:
        raise Phase13BenchmarkError("scene missing concrete sampler block")
    return patched


def compute_scene_config_sha256(scene_text: str) -> str:
    """Return a stable identity for the exact XML supplied to Mitsuba."""

    return hashlib.sha256(scene_text.encode("utf-8")).hexdigest()


def compute_file_sha256(path: Path) -> str:
    """Return the SHA-256 digest of a file without loading it all into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_scalar_property_value(text: str, *, name: str) -> str | None:
    """Read first Mitsuba scalar property matching `name`.

    Args:
        text: Scene XML blob.
        name: Parameter name.

    Returns:
        Stored string attribute or None if unset.
    """

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
    """Swap entire `<integrator>...</integrator>` subtree.

    Args:
        text: Original scene markup.
        integrator_block: Replacement integrator subtree.

    Returns:
        Rewritten XML string.
    """

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
        raise Phase13BenchmarkError("scene missing integrator block")
    return replaced_text


def build_integrator_block(
    *,
    job: RenderJob,
    method: str,
    config: Phase13BenchmarkConfig,
    runtime_model_value: str,
) -> str:
    """Construct integrator subtree XML emitted into patched scenes.

    Args:
        method: Renderer slug selected for this job row.
        config: Hydrated Phase 13 knobs (samples, masking, ...).
        runtime_model_value: Runtime text path understood by Mitsuba host.

    Returns:
        Full `<integrator ...>...</integrator>` XML string.
    """

    masked_diffusion_train_budget = (
        config.mpg_train_budget
        if config.masked_diffusion_train_budget is None
        else config.masked_diffusion_train_budget
    )

    if method in {"path_reference", "pt"}:
        return (
            '<integrator type="path">\n'
            '\t\t<integer name="max_depth" value="$max_depth" />\n'
            "\t</integrator>"
        )

    # Standalone guiding baselines deliberately bypass the MPG main path.
    if method == "ppg":
        return (
            '<integrator type="ppg">\n'
            '\t\t<integer name="max_depth" value="$max_depth" />\n'
            f'\t\t<integer name="samples_per_pass" value="{config.samples_per_pass}" />\n'
            '\t\t<boolean name="train_auto" value="true" />\n'
            f'\t\t<float name="train_budget" value="{config.mpg_train_budget}" />\n'
            '\t\t<float name="guide_fraction" value="0.5" />\n'
            "\t</integrator>"
        )

    if method == "vapg":
        return (
            '<integrator type="vapg">\n'
            '\t\t<integer name="max_depth" value="$max_depth" />\n'
            f'\t\t<integer name="samples_per_pass" value="{config.samples_per_pass}" />\n'
            '\t\t<boolean name="train_auto" value="true" />\n'
            f'\t\t<float name="train_budget" value="{config.mpg_train_budget}" />\n'
            '\t\t<float name="guide_fraction" value="0.5" />\n'
            "\t</integrator>"
        )

    if method == "vmmpg":
        return (
            '<integrator type="vmmpg">\n'
            '\t\t<integer name="max_depth" value="$max_depth" />\n'
            f'\t\t<integer name="samples_per_pass" value="{config.samples_per_pass}" />\n'
            '\t\t<boolean name="train_auto" value="true" />\n'
            f'\t\t<float name="train_budget" value="{config.mpg_train_budget}" />\n'
            '\t\t<float name="guide_fraction" value="0.5" />\n'
            '\t\t<float name="mixture_balance" value="0.5" />\n'
            "\t</integrator>"
        )

    shared_prefix = (
        '<integrator type="manifold_path_guiding">\n'
        '\t\t<integer name="max_depth" value="$max_depth" />\n'
        f'\t\t<integer name="samples_per_pass" value="{config.samples_per_pass}" />\n'
        '\t\t<boolean name="selective" value="false" />\n'
        '\t\t<boolean name="budget_is_time" value="false" />\n'
        '\t\t<float name="timeout" value="0" />\n'
        f'\t\t<integer name="seedid" value="{config.seed}" />\n'
        f'\t\t<float name="bounce_alpha" value="{config.bounce_alpha}" />\n'
    )

    def runtime_model_controls_xml() -> str:
        flip_to_receiver = str(
            config.learned_direction_flip_to_receiver_hemisphere
        ).lower()
        return (
            f'\t\t<integer name="diffusion_steps" value="{config.diffusion_steps}" />\n'
            + '\t\t<float name="learned_direction_kappa_scale" '
            f'value="{config.learned_direction_kappa_scale}" />\n'
            + '\t\t<boolean name="learned_direction_flip_to_receiver_hemisphere" '
            f'value="{flip_to_receiver}" />\n'
        )

    def learned_proposal_stabilization_xml(*, include_ablation: bool = True) -> str:
        xml = ""
        if config.learned_proposal_inv_prob_clamp is not None:
            xml += (
                '\t\t<float name="learned_proposal_inv_prob_clamp" '
                f'value="{config.learned_proposal_inv_prob_clamp}" />\n'
            )
        if config.learned_proposal_contribution_clamp is not None:
            xml += (
                '\t\t<float name="learned_proposal_contribution_clamp" '
                f'value="{config.learned_proposal_contribution_clamp}" />\n'
            )
        if config.learned_proposal_max_trials is not None:
            xml += (
                '\t\t<integer name="max_trials" '
                f'value="{config.learned_proposal_max_trials}" />\n'
            )
        if include_ablation and config.masked_diffusion_ablation is not None:
            xml += (
                '\t\t<integer name="ablation" '
                f'value="{config.masked_diffusion_ablation}" />\n'
            )
        return xml

    if method == "sms":
        scene_id = job.job_id.removesuffix("_mpg")
        preset = SMS_SCENE_PRESETS.get(scene_id)
        if preset is None:
            raise Phase13BenchmarkError(f"sms preset missing for scene: {scene_id}")
        # Canonical SMS insists on deterministic chain configs; Fig. 9 presets live in SMS_SCENE_PRESETS.
        return (
            shared_prefix
            + '\t\t<boolean name="train_auto" value="false" />\n'
            + '\t\t<boolean name="guided" value="false" />\n'
            + '\t\t<float name="prob_uniform" value="1.0" />\n'
            + f'\t\t<integer name="sms_bounce_force" value="{preset["bounce"]}" />\n'
            + f'\t\t<integer name="sms_tau_force" value="{preset["tau"]}" />\n'
            + learned_proposal_stabilization_xml(include_ablation=False)
            + "\t</integrator>"
        )

    # Historic alias `uniform` maps to SMS* in the publication; retain both keys for manifests.
    if method in {"uniform", "sms_star"}:
        return (
            shared_prefix
            + '\t\t<boolean name="train_auto" value="false" />\n'
            + '\t\t<boolean name="guided" value="false" />\n'
            + learned_proposal_stabilization_xml(include_ablation=False)
            + "\t</integrator>"
        )

    if method == "mpg":
        return (
            shared_prefix
            + '\t\t<boolean name="train_auto" value="true" />\n'
            + '\t\t<boolean name="guided" value="true" />\n'
            + f'\t\t<float name="train_budget" value="{config.mpg_train_budget}" />\n'
            + (
                '\t\t<boolean name="train_fusion" value="true" />\n'
                if config.mpg_train_fusion
                else '\t\t<boolean name="train_fusion" value="false" />\n'
            )
            + "\t</integrator>"
        )

    if method == "mpg_selective":
        return (
            shared_prefix
            + '\t\t<boolean name="train_auto" value="true" />\n'
            + '\t\t<boolean name="guided" value="true" />\n'
            + '\t\t<boolean name="selective" value="true" />\n'
            + f'\t\t<float name="train_budget" value="{config.mpg_train_budget}" />\n'
            + (
                '\t\t<boolean name="train_fusion" value="true" />\n'
                if config.mpg_train_fusion
                else '\t\t<boolean name="train_fusion" value="false" />\n'
            )
            + "\t</integrator>"
        )

    if method == "mpg_prod":
        return (
            shared_prefix
            + '\t\t<boolean name="train_auto" value="true" />\n'
            + '\t\t<boolean name="guided" value="true" />\n'
            + '\t\t<boolean name="product_sampling" value="true" />\n'
            + f'\t\t<float name="train_budget" value="{config.mpg_train_budget}" />\n'
            + (
                '\t\t<boolean name="train_fusion" value="true" />\n'
                if config.mpg_train_fusion
                else '\t\t<boolean name="train_fusion" value="false" />\n'
            )
            + "\t</integrator>"
        )

    if method == "mpg_nofilter":
        return (
            shared_prefix
            + '\t\t<boolean name="train_auto" value="true" />\n'
            + '\t\t<boolean name="guided" value="true" />\n'
            + '\t\t<float name="spatial_filter" value="0" />\n'
            + f'\t\t<float name="train_budget" value="{config.mpg_train_budget}" />\n'
            + (
                '\t\t<boolean name="train_fusion" value="true" />\n'
                if config.mpg_train_fusion
                else '\t\t<boolean name="train_fusion" value="false" />\n'
            )
            + "\t</integrator>"
        )

    if method == "mpg_stocfilter":
        return (
            shared_prefix
            + '\t\t<boolean name="train_auto" value="true" />\n'
            + '\t\t<boolean name="guided" value="true" />\n'
            + '\t\t<float name="spatial_filter" value="-1" />\n'
            + f'\t\t<float name="train_budget" value="{config.mpg_train_budget}" />\n'
            + (
                '\t\t<boolean name="train_fusion" value="true" />\n'
                if config.mpg_train_fusion
                else '\t\t<boolean name="train_fusion" value="false" />\n'
            )
            + "\t</integrator>"
        )

    if method == "mpg_knn":
        return (
            shared_prefix
            + '\t\t<boolean name="train_auto" value="true" />\n'
            + '\t\t<boolean name="guided" value="true" />\n'
            + '\t\t<integer name="spatial_struct" value="0" />\n'
            + f'\t\t<float name="train_budget" value="{config.mpg_train_budget}" />\n'
            + (
                '\t\t<boolean name="train_fusion" value="true" />\n'
                if config.mpg_train_fusion
                else '\t\t<boolean name="train_fusion" value="false" />\n'
            )
            + "\t</integrator>"
        )

    if method == "mpg_m19":
        return (
            shared_prefix
            + '\t\t<boolean name="train_auto" value="true" />\n'
            + '\t\t<boolean name="guided" value="true" />\n'
            + '\t\t<integer name="spatial_struct" value="2" />\n'
            + '\t\t<integer name="directional_struct" value="1" />\n'
            + '\t\t<float name="spatial_filter" value="-1" />\n'
            + '\t\t<integer name="knn_k" value="-4000" />\n'
            + f'\t\t<float name="train_budget" value="{config.mpg_train_budget}" />\n'
            + (
                '\t\t<boolean name="train_fusion" value="true" />\n'
                if config.mpg_train_fusion
                else '\t\t<boolean name="train_fusion" value="false" />\n'
            )
            + "\t</integrator>"
        )

    if method == "masked_runtime":
        return (
            shared_prefix
            + '\t\t<boolean name="train_auto" value="false" />\n'
            + '\t\t<boolean name="guided" value="false" />\n'
            + learned_proposal_stabilization_xml()
            + runtime_model_controls_xml()
            + (
                '\t\t<string name="masked_diffusion_runtime_path" '
                f'value="{runtime_model_value}" />\n'
            )
            + "\t</integrator>"
        )

    if method in {
        "masked_diffusion",
        "masked_diffusion_vmf",
        "masked_hybrid",
        "simple_mlp",
        "simple_mlp_k4",
        *NPM_ADAPTED_METHODS,
        "diffusion_mlp",
        "diffusion_transformer",
        "ordered_diffusion_mlp",
        "ordered_diffusion_transformer",
        "ordered_diffusion_dit",
        "ordered_diffusion_mlp_h72_k1",
        "ordered_diffusion_mlp_h72_k4",
        "ordered_diffusion_mlp_desc_q95",
        "ordered_diffusion_mlp_desc_q99",
        "ordered_diffusion_mlp_dir_q95",
        "ordered_diffusion_mlp_dir_q99",
    }:
        if method in config.runtime_only_methods or method in NPM_ADAPTED_METHODS:
            return (
                shared_prefix
                + '\t\t<boolean name="train_auto" value="false" />\n'
                + '\t\t<boolean name="guided" value="false" />\n'
                + learned_proposal_stabilization_xml()
                + runtime_model_controls_xml()
                + (
                    '\t\t<string name="masked_diffusion_runtime_path" '
                    f'value="{runtime_model_value}" />\n'
                )
                + "\t</integrator>"
            )

        masked_prob_uniform = (
            0.5
            if config.masked_diffusion_prob_uniform is None
            else config.masked_diffusion_prob_uniform
        )
        masked_prob_uniform_train = (
            masked_prob_uniform
            if config.masked_diffusion_prob_uniform_train is None
            else config.masked_diffusion_prob_uniform_train
        )
        # Benchmark “main” combos mean MPG backbone + neural runtime fallback.
        # `simple_mlp` and masked vMF checkpoints share identical proposal plumbing—only weights differ.
        return (
            shared_prefix
            + '\t\t<boolean name="train_auto" value="true" />\n'
            + '\t\t<boolean name="guided" value="true" />\n'
            + (
                ""
                if config.masked_diffusion_spatial_filter is None
                else (
                    '\t\t<float name="spatial_filter" '
                    f'value="{config.masked_diffusion_spatial_filter}" />\n'
                )
            )
            # Neural proposals only spike in when MPG stats under-cover unstable regions—keeps MPG primary.
            + f'\t\t<float name="prob_uniform" value="{masked_prob_uniform}" />\n'
            + (
                '\t\t<float name="prob_uniform_train" '
                f'value="{masked_prob_uniform_train}" />\n'
            )
            + learned_proposal_stabilization_xml()
            + (
                '\t\t<float name="train_budget" '
                f'value="{masked_diffusion_train_budget}" />\n'
            )
            + (
                '\t\t<boolean name="train_fusion" value="true" />\n'
                if config.mpg_train_fusion
                else '\t\t<boolean name="train_fusion" value="false" />\n'
            )
            + runtime_model_controls_xml()
            + (
                '\t\t<string name="masked_diffusion_runtime_path" '
                f'value="{runtime_model_value}" />\n'
            )
            + "\t</integrator>"
        )

    raise Phase13BenchmarkError(f"unknown method: {method}")


def patch_scene_text(
    source_text: str,
    *,
    profile: ResolvedRuntimeProfile,
    job: RenderJob,
    method: str,
    config: Phase13BenchmarkConfig,
    runtime_model_path: Path | None,
) -> tuple[str, int]:
    """Patch scene XML for `method`, returning text + finalized sample budget.

    Args:
        source_text: Original Mitsuba XML string.
        profile: Resolved host profile (paths differ for WSL).
        job: MPG manifest entry informing scene assets.
        method: Renderer slug.
        config: Global benchmark knobs lifted from Phase 13 JSON.
        runtime_model_path: Exported neural checkpoint, if applicable.

    Returns:
        Patched ASCII XML plus authoritative sample_count for bookkeeping.
    """

    sample_count = resolve_compare_sample_count(
        config=config,
        job=job,
        method=method,
    )
    max_depth = find_scalar_property_value(source_text, name="max_depth") or "15"
    patched = source_text
    runtime_model_value = (
        "" if runtime_model_path is None else to_profile_path(profile, runtime_model_path)
    )
    patched = replace_integrator_block(
        patched,
        build_integrator_block(
            job=job,
            method=method,
            config=config,
            runtime_model_value=runtime_model_value,
        ),
    )
    scalar_values = {
        "imgwidth": str(config.width),
        "imgheight": str(config.height),
        "crop_offset_x": "0",
        "crop_offset_y": "0",
        "crop_width": str(config.crop_width),
        "crop_height": str(config.crop_height),
        "width": str(config.width),
        "height": str(config.height),
        "max_depth": max_depth,
        "sample_count": str(sample_count),
        "spp": str(sample_count),
        "samples_per_pass": str(config.samples_per_pass),
        "timeout": "0",
    }
    for name, value in scalar_values.items():
        patched = patch_scalar_property(patched, name=name, value=value)
    sampler_seed = (
        config.reference_seed
        if method == "path_reference" and config.reference_seed is not None
        else config.seed
    )
    patched = patch_sampler_seed(patched, sampler_seed=sampler_seed)
    return patched, sample_count


def resolve_compare_sample_count(
    *,
    config: Phase13BenchmarkConfig,
    job: RenderJob,
    method: str,
) -> int:
    """Determine per-method sample budget respecting overrides.

    Args:
        config: Phase13BenchmarkConfig aggregates.
        job: Render manifest row owning scene identifiers.
        method: Renderer slug (reference vs comparator).

    Returns:
        Effective sample integral used for Mitsuba XML injection.
    """

    if method == "path_reference":
        return config.reference_sample_count

    scene_id = job.job_id.removesuffix("_mpg")
    override_keys = (
        f"{scene_id}|{method}",
        f"{job.job_id}|{method}",
        f"{scene_id}:{method}",
        f"{job.job_id}:{method}",
    )
    if config.scene_method_compare_sample_counts is not None:
        for override_key in override_keys:
            override_value = config.scene_method_compare_sample_counts.get(override_key)
            if override_value is None:
                continue
            return max(1, int(override_value))

    if config.method_compare_sample_counts is not None:
        method_value = config.method_compare_sample_counts.get(method)
        if method_value is not None:
            return max(1, int(method_value))

    return config.compare_sample_count


def load_exr_rgb(path: Path) -> np.ndarray:
    """Robust OpenCV EXR ingestion returning `HxWx3 float32`.

    Args:
        path: EXR location on filesystem.

    Returns:
        Finite float32 HDR buffer arranged RGB order.
    """

    os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
    try:
        import cv2
    except ModuleNotFoundError as exc:
        raise Phase13BenchmarkError(
            "phase_13 requires cv2 built with OpenEXR support "
            "(set OPENCV_IO_ENABLE_OPENEXR=1 and install contrib builds)"
        ) from exc

    if not path.exists():
        raise Phase13BenchmarkError(f"missing exr: {path}")

    stable_size = -1
    image = None
    for _ in range(6):
        current_size = path.stat().st_size if path.exists() else 0
        if current_size <= 0:
            time.sleep(0.25)
            continue
        if stable_size == current_size:
            image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if image is not None:
                break
        stable_size = current_size
        time.sleep(0.25)
    if image is None:
        file_size = path.stat().st_size if path.exists() else -1
        raise Phase13BenchmarkError(f"failed to read exr: {path} (size={file_size})")
    if image.ndim == 2:
        image = np.stack([image, image, image], axis=-1)
    if image.ndim != 3:
        raise Phase13BenchmarkError(f"unexpected exr ndim: {image.ndim}")
    if image.shape[2] >= 4:
        image = image[:, :, :3]
    if image.shape[2] == 1:
        image = np.repeat(image, 3, axis=2)
    if image.shape[2] != 3:
        raise Phase13BenchmarkError(f"unexpected exr channels: {image.shape[2]}")
    image = image[:, :, ::-1].copy().astype(np.float32)
    if not np.all(np.isfinite(image)):
        raise Phase13BenchmarkError(f"nonfinite pixels in exr: {path}")
    return image


def _require_sha256(value: object, *, label: str) -> str:
    """Validate a lowercase hexadecimal SHA-256 string."""

    text = str(value)
    if re.fullmatch(r"[0-9a-f]{64}", text) is None:
        raise Phase13BenchmarkError(f"invalid SHA-256 for {label}: {value!r}")
    return text


def _resolve_project_artifact(path_text: object, *, label: str) -> Path:
    """Resolve a manifest path relative to PROJECT_ROOT without allowing escape."""

    relative = Path(str(path_text))
    if relative.is_absolute():
        raise Phase13BenchmarkError(f"{label} must be project-relative: {relative}")
    resolved = (PROJECT_ROOT / relative).resolve()
    try:
        resolved.relative_to(PROJECT_ROOT.resolve())
    except ValueError as exc:
        raise Phase13BenchmarkError(
            f"{label} escapes project root: {relative}"
        ) from exc
    if not resolved.is_file():
        raise Phase13BenchmarkError(f"missing {label}: {resolved}")
    return resolved


def load_external_reference_bundle(
    *,
    workspace_root: Path,
    config: Phase13BenchmarkConfig,
    jobs: tuple[RenderJob, ...],
    validate_pixels: bool = True,
) -> ResolvedSharedReferenceBundle:
    """Load and fully verify a ready shared-reference manifest.

    The manifest and source summary are hashed, every selected EXR is rehashed,
    and dimensions/SPP/seed are checked before any candidate render starts.
    Pixel decoding can be deferred for blinded runs; the byte hash is always
    verified.
    """

    if config.external_reference_manifest is None:
        raise Phase13BenchmarkError("external_reference_manifest is missing")
    manifest_path = Path(config.external_reference_manifest)
    if not manifest_path.is_absolute():
        manifest_path = (workspace_root / manifest_path).resolve()
    else:
        manifest_path = manifest_path.resolve()
    try:
        manifest_path.relative_to(PROJECT_ROOT.resolve())
    except ValueError as exc:
        raise Phase13BenchmarkError(
            f"external reference manifest must stay under project root: {manifest_path}"
        ) from exc
    if not manifest_path.is_file():
        raise Phase13BenchmarkError(
            f"external reference manifest missing: {manifest_path}"
        )
    manifest_sha256 = compute_file_sha256(manifest_path)
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Phase13BenchmarkError(
            f"invalid external reference manifest: {manifest_path}"
        ) from exc
    if payload.get("schema_version") != 1 or payload.get("status") != "ready":
        raise Phase13BenchmarkError(
            "external reference manifest must have schema_version=1 and status='ready'"
        )
    if payload.get("path_base") != "project_root":
        raise Phase13BenchmarkError(
            "external reference manifest path_base must be 'project_root'"
        )

    source_summary_path = _resolve_project_artifact(
        payload.get("source_summary"), label="shared-reference source summary"
    )
    source_summary_sha256 = _require_sha256(
        payload.get("source_summary_sha256"), label="source summary"
    )
    if compute_file_sha256(source_summary_path) != source_summary_sha256:
        raise Phase13BenchmarkError(
            f"shared-reference source summary hash mismatch: {source_summary_path}"
        )

    raw_references = payload.get("references")
    if not isinstance(raw_references, dict):
        raise Phase13BenchmarkError("shared-reference manifest references must be an object")
    reference_set_id = _require_sha256(
        payload.get("reference_set_id"), label="shared reference set"
    )
    computed_reference_set_id = hashlib.sha256(
        json.dumps(
            raw_references,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    if computed_reference_set_id != reference_set_id:
        raise Phase13BenchmarkError("shared-reference reference_set_id mismatch")
    references: dict[str, ResolvedSharedReference] = {}
    for job in jobs:
        scene_id = job.job_id.removesuffix("_mpg")
        raw = raw_references.get(scene_id)
        if not isinstance(raw, dict):
            raise Phase13BenchmarkError(
                f"shared-reference manifest missing scene: {scene_id}"
            )
        if raw.get("scene_id") != scene_id:
            raise Phase13BenchmarkError(
                f"shared-reference scene identity mismatch for {scene_id}"
            )
        exr_path = _resolve_project_artifact(
            raw.get("exr_path"), label=f"shared reference EXR for {scene_id}"
        )
        exr_sha256 = _require_sha256(
            raw.get("exr_sha256"), label=f"reference EXR {scene_id}"
        )
        if compute_file_sha256(exr_path) != exr_sha256:
            raise Phase13BenchmarkError(
                f"shared reference EXR hash mismatch for {scene_id}: {exr_path}"
            )
        expected_values = {
            "sample_count": config.reference_sample_count,
            "sampler_seed": config.reference_seed,
            "width": config.width,
            "height": config.height,
            "crop_width": config.crop_width,
            "crop_height": config.crop_height,
        }
        for field, expected in expected_values.items():
            if raw.get(field) != expected:
                raise Phase13BenchmarkError(
                    f"shared reference {scene_id} has {field}={raw.get(field)!r}; "
                    f"expected {expected!r}"
                )
        output_size = int(raw.get("output_size", -1))
        if output_size <= 0 or exr_path.stat().st_size != output_size:
            raise Phase13BenchmarkError(
                f"shared reference size mismatch for {scene_id}: {exr_path}"
            )
        active_config_sha256 = _require_sha256(
            raw.get("active_config_sha256"),
            label=f"reference XML config {scene_id}",
        )
        if validate_pixels:
            image = load_exr_rgb(exr_path)
            expected_shape = (config.crop_height, config.crop_width)
            if image.shape[:2] != expected_shape:
                raise Phase13BenchmarkError(
                    f"shared reference shape mismatch for {scene_id}: "
                    f"{image.shape[:2]} != {expected_shape}"
                )
        references[scene_id] = ResolvedSharedReference(
            scene_id=scene_id,
            exr_path=exr_path,
            exr_sha256=exr_sha256,
            output_size=output_size,
            sample_count=int(raw["sample_count"]),
            sampler_seed=int(raw["sampler_seed"]),
            active_config_sha256=active_config_sha256,
            width=int(raw["width"]),
            height=int(raw["height"]),
            crop_width=int(raw["crop_width"]),
            crop_height=int(raw["crop_height"]),
        )
    return ResolvedSharedReferenceBundle(
        manifest_path=manifest_path,
        manifest_sha256=manifest_sha256,
        source_summary_path=source_summary_path,
        source_summary_sha256=source_summary_sha256,
        references=references,
    )


def verify_external_reference_bundle(bundle: ResolvedSharedReferenceBundle) -> None:
    """Fail if a ready manifest, source summary, or reference changed in place."""

    if compute_file_sha256(bundle.manifest_path) != bundle.manifest_sha256:
        raise Phase13BenchmarkError(
            f"external reference manifest changed during run: {bundle.manifest_path}"
        )
    if compute_file_sha256(bundle.source_summary_path) != bundle.source_summary_sha256:
        raise Phase13BenchmarkError(
            "shared-reference source summary changed during run: "
            f"{bundle.source_summary_path}"
        )
    for reference in bundle.references.values():
        if compute_file_sha256(reference.exr_path) != reference.exr_sha256:
            raise Phase13BenchmarkError(
                f"shared reference EXR changed during run: {reference.exr_path}"
            )


def serialize_external_reference_bundle(
    bundle: ResolvedSharedReferenceBundle,
) -> dict[str, object]:
    """Serialize verified external-reference provenance for a run summary."""

    return {
        "mode": "external_manifest",
        "manifest_path": relativize_export_path(
            str(bundle.manifest_path), PROJECT_ROOT
        ),
        "manifest_sha256": bundle.manifest_sha256,
        "source_summary": relativize_export_path(
            str(bundle.source_summary_path), PROJECT_ROOT
        ),
        "source_summary_sha256": bundle.source_summary_sha256,
        "references": {
            scene_id: {
                "scene_id": reference.scene_id,
                "exr_path": relativize_export_path(
                    str(reference.exr_path), PROJECT_ROOT
                ),
                "exr_sha256": reference.exr_sha256,
                "output_size": reference.output_size,
                "sample_count": reference.sample_count,
                "sampler_seed": reference.sampler_seed,
                "active_config_sha256": reference.active_config_sha256,
                "width": reference.width,
                "height": reference.height,
                "crop_width": reference.crop_width,
                "crop_height": reference.crop_height,
            }
            for scene_id, reference in sorted(bundle.references.items())
        },
    }


def compute_metrics(
    prediction_path: Path,
    reference_path: Path,
) -> tuple[float, float, float]:
    """MSE / PSNR / reference peak diagnostics relative to HDR reference.

    Args:
        prediction_path: Comparator EXR.
        reference_path: Path-tracer reference EXR for same crop.

    Returns:
        Tuple `(mse, psnr, max_reference_magnitude)` for logging.
    """

    prediction = load_exr_rgb(prediction_path)
    reference = load_exr_rgb(reference_path)
    if prediction.shape != reference.shape:
        raise Phase13BenchmarkError(
            f"shape mismatch: {prediction.shape} vs {reference.shape}"
        )

    diff = prediction.astype(np.float64) - reference.astype(np.float64)
    mse = float(np.mean(diff * diff))
    # Peak-SNR denominators reuse reference radiance peaks—prediction spikes must not inflate this term.
    max_value = float(max(float(np.max(np.abs(reference))), 1e-6))
    psnr = float("inf") if mse <= 0.0 else float(10.0 * np.log10(max_value**2 / mse))
    return mse, psnr, max_value


def verify_render_output(output_path: Path) -> tuple[bool, int]:
    """Return `(readable, nbytes)` heuristic for Mitsuba dumps.

    Args:
        output_path: Candidate EXR on disk after render subprocess.

    Returns:
        Bool readability flag plus byte-length for telemetry.
    """

    if not output_path.exists():
        return False, 0
    file_size = output_path.stat().st_size
    if file_size <= 0:
        return False, file_size
    try:
        load_exr_rgb(output_path)
    except Phase13BenchmarkError:
        return False, file_size
    return True, file_size


def validate_deferred_candidate_output(
    output_path: Path,
    *,
    expected_height: int,
    expected_width: int,
) -> tuple[bool, int, bool | None]:
    """Validate only candidate structure/finite pixels without opening a reference.

    This operational check intentionally exposes no pixel statistics. It makes a
    deferred record eligible for the experiment barrier while keeping MSE, PSNR,
    reference pixels, and visual inspection locked.
    """

    if not output_path.exists():
        return False, 0, None
    file_size = output_path.stat().st_size
    if file_size <= 0:
        return False, file_size, None
    try:
        image = load_exr_rgb(output_path)
    except Phase13BenchmarkError as exc:
        detected_nonfinite = "nonfinite pixels in exr" in str(exc)
        return False, file_size, True if detected_nonfinite else None
    if image.shape != (expected_height, expected_width, 3):
        return False, file_size, False
    return True, file_size, False


def should_retry_interrupted_render(
    result: subprocess.CompletedProcess[str],
) -> bool:
    """Heuristic Mitsuba flaky-failure classifier (EINTR quirks).

    Args:
        result: Completed Mitsuba subprocess record.

    Returns:
        True when logs mention interrupted syscalls / driver error-code 4.
    """

    combined_output = f"{result.stdout}\n{result.stderr}"
    if "Interrupted system call" in combined_output:
        return True
    if result.returncode == 4 and "error code 4" in combined_output:
        return True
    return False


def merge_attempt_logs(
    attempts: tuple[subprocess.CompletedProcess[str], ...],
    *,
    stream_name: str,
) -> str:
    """Flatten ordered subprocess transcripts for artifact logs.

    Args:
        attempts: Tuple of CompletedProcess clones from rerun loop.
        stream_name: `stdout` / `stderr` attribute name.

    Returns:
        Concatenated annotated log blob.
    """

    chunks: list[str] = []
    for index, attempt in enumerate(attempts, start=1):
        stream_text = getattr(attempt, stream_name)
        chunks.append(f"===== attempt {index} / {stream_name} =====\n{stream_text}")
    return "\n".join(chunks)


def build_render_log_metadata(
    *,
    scene_id: str,
    job_id: str,
    active_method: str,
    sampler_seed: int,
    active_config_sha256: str,
    patched_scene_record: Path,
    runtime_model_path: Path | None,
    command: list[str],
) -> str:
    """Build a machine-readable provenance header for renderer logs."""

    payload = {
        "scene_id": scene_id,
        "job_id": job_id,
        "active_method": active_method,
        "sampler_seed": sampler_seed,
        "active_config_sha256": active_config_sha256,
        "patched_scene_record": str(patched_scene_record),
        "runtime_model_path": (
            None if runtime_model_path is None else str(runtime_model_path)
        ),
        "command": command,
    }
    return (
        "===== phase13 render metadata =====\n"
        + json.dumps(payload, sort_keys=True, ensure_ascii=False)
        + "\n"
    )


def phase13_temp_scene_discriminator(
    *,
    run_dir: Path,
    active_config_sha256: str,
) -> str:
    """Return a stable, compact identity for one run/config pairing."""

    identity = f"{run_dir.resolve()}\0{active_config_sha256}"
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]


def create_phase13_temp_scene(
    *,
    source_scene: Path,
    run_dir: Path,
    profile_id: str,
    scene_id: str,
    method: str,
    active_config_sha256: str,
    patched_text: str,
) -> Path:
    """Atomically create a collision-proof scene copy beside source assets.

    The readable prefix records both the process and a stable run/config
    discriminator. ``NamedTemporaryFile`` adds an atomically allocated suffix,
    so even concurrent allocations from the same process cannot alias.
    """

    discriminator = phase13_temp_scene_discriminator(
        run_dir=run_dir,
        active_config_sha256=active_config_sha256,
    )
    prefix = (
        f"{source_scene.stem}.phase13_pid{os.getpid()}_{discriminator}_"
        f"{profile_id}_{scene_id}.{method}."
    )
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=source_scene.parent,
        prefix=prefix,
        suffix=".xml",
        delete=False,
    ) as handle:
        handle.write(patched_text)
        return Path(handle.name)


def run_render(
    *,
    profile: ResolvedRuntimeProfile,
    runtime_env: dict[str, str],
    config: Phase13BenchmarkConfig,
    job: RenderJob,
    method: str,
    run_dir: Path,
    runtime_model_path: Path | None,
) -> BenchmarkRecord:
    """Render one Manifest job / method pairing and hydrate BenchmarkRecord telemetry.

    Args:
        profile: Resolved Mitsuba host profile metadata.
        runtime_env: Only used for native mode subprocess env overlays.
        config: Hydrated Phase 13 knobs.
        job: Scene manifest MPG row referencing asset paths.
        method: Rendering strategy slug patched into XML integrator subtree.
        run_dir: Timestamped directory staging logs + EXRs.
        runtime_model_path: Optional neural artifact path.

    Returns:
        Populated BenchmarkRecord prior to attaching PSNR overlays.
    """

    source_scene = PROJECT_ROOT / job.scene_path
    missing_assets = collect_missing_assets(source_scene)
    if missing_assets:
        raise Phase13BenchmarkError(
            f"scene missing assets for {job.job_id}: {missing_assets}"
        )

    output_dir = run_dir / "outputs"
    scene_dir = run_dir / "scenes"
    log_dir = run_dir / "logs"
    output_dir.mkdir(parents=True, exist_ok=True)
    scene_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    scene_id = job.job_id.removesuffix("_mpg")
    patched_text, sample_count = patch_scene_text(
        source_scene.read_text(encoding="utf-8"),
        profile=profile,
        job=job,
        method=method,
        config=config,
        runtime_model_path=runtime_model_path,
    )
    sampler_seed = (
        config.reference_seed
        if method == "path_reference" and config.reference_seed is not None
        else config.seed
    )
    active_method = method
    active_config_sha256 = compute_scene_config_sha256(patched_text)

    patched_scene_record = scene_dir / f"{scene_id}.{method}.xml"
    patched_scene_record.write_text(patched_text, encoding="utf-8")
    # Copies must neighbor original assets so relative textured paths keep resolving.
    temp_scene_path = create_phase13_temp_scene(
        source_scene=source_scene,
        run_dir=run_dir,
        profile_id=profile.profile_id,
        scene_id=scene_id,
        method=method,
        active_config_sha256=active_config_sha256,
        patched_text=patched_text,
    )

    output_path = output_dir / f"{scene_id}.{method}.exr"
    stdout_log = log_dir / f"{scene_id}.{method}.stdout.txt"
    stderr_log = log_dir / f"{scene_id}.{method}.stderr.txt"
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
    log_metadata = build_render_log_metadata(
        scene_id=scene_id,
        job_id=job.job_id,
        active_method=active_method,
        sampler_seed=sampler_seed,
        active_config_sha256=active_config_sha256,
        patched_scene_record=patched_scene_record,
        runtime_model_path=runtime_model_path,
        command=process_command,
    )
    LOGGER.info(
        "render identity: scene=%s method=%s sampler_seed=%s config_sha256=%s",
        scene_id,
        active_method,
        sampler_seed,
        active_config_sha256,
    )

    def invoke_render() -> tuple[subprocess.CompletedProcess[str], float]:
        """Spawn Mitsuba subprocess once."""

        t0 = time.perf_counter()
        render_timeout = config.time_budget_seconds
        if render_timeout is not None and render_timeout <= 0:
            render_timeout = None
        try:
            result = subprocess.run(
                process_command,
                cwd=None if profile.command_mode == "wsl" else PROJECT_ROOT,
                env=None if profile.command_mode == "wsl" else runtime_env,
                capture_output=True,
                text=True,
                check=False,
                encoding="utf-8",
                errors="replace",
                timeout=render_timeout,
            )
        except subprocess.TimeoutExpired as exc:
            def decoded_output(value: str | bytes | None) -> str:
                if value is None:
                    return ""
                if isinstance(value, bytes):
                    return value.decode("utf-8", errors="replace")
                return value

            timeout_message = (
                f"render exceeded watchdog ({render_timeout:.1f} seconds)"
            )
            result = subprocess.CompletedProcess(
                args=process_command,
                returncode=124,
                stdout=decoded_output(exc.stdout),
                stderr=(decoded_output(exc.stderr) + "\n" + timeout_message).lstrip(),
            )
        return result, time.perf_counter() - t0

    attempts: list[subprocess.CompletedProcess[str]] = []
    seconds = 0.0
    output_valid = False
    output_size = 0
    nonfinite_pixel_boolean: bool | None = None
    try:
        for attempt_index in range(2):
            result, attempt_seconds = invoke_render()
            attempts.append(result)
            seconds += attempt_seconds
            if config.defer_metrics:
                (
                    output_valid,
                    output_size,
                    nonfinite_pixel_boolean,
                ) = validate_deferred_candidate_output(
                    output_path,
                    expected_height=config.crop_height,
                    expected_width=config.crop_width,
                )
            else:
                output_valid, output_size = verify_render_output(output_path)
                nonfinite_pixel_boolean = False if output_valid else None

            retry_reason: str | None = None
            if should_retry_interrupted_render(result):
                retry_reason = f"interrupted system call (return={result.returncode})"
            # Retry once after success-with-unreadable EXR races (slow NFS / writer flush latency).
            elif result.returncode == 0 and not output_valid:
                retry_reason = f"unreadable exr (size={output_size})"
            else:
                output_size = output_path.stat().st_size if output_path.exists() else 0

            if retry_reason is not None and attempt_index == 0:
                LOGGER.warning(
                    "rerender once for %s: %s / %s",
                    retry_reason,
                    job.job_id,
                    method,
                )
                continue
            break
    finally:
        temp_scene_path.unlink(missing_ok=True)

    stdout_log.write_text(
        log_metadata + merge_attempt_logs(tuple(attempts), stream_name="stdout"),
        encoding="utf-8",
    )
    stderr_log.write_text(
        log_metadata + merge_attempt_logs(tuple(attempts), stream_name="stderr"),
        encoding="utf-8",
    )
    loaded_runtime_model = "Loaded masked diffusion runtime model from" in (
        result.stdout + "\n" + result.stderr
    )
    runtime_model_ok = (
        not method_requires_runtime_model(method) or loaded_runtime_model
    )
    ok = (
        result.returncode == 0
        and output_path.exists()
        and output_valid
        and runtime_model_ok
    )

    return BenchmarkRecord(
        scene_id=scene_id,
        scene_path=job.scene_path,
        method=method,
        ok=ok,
        return_code=result.returncode,
        seconds=seconds,
        sample_count=sample_count,
        sampler_seed=sampler_seed,
        active_method=active_method,
        active_config_sha256=active_config_sha256,
        output_path=str(output_path),
        output_exists=output_path.exists(),
        output_size=output_size,
        output_sha256=(compute_file_sha256(output_path) if output_valid else None),
        patched_scene_record=str(patched_scene_record),
        stdout_log=str(stdout_log),
        stderr_log=str(stderr_log),
        loaded_runtime_model=loaded_runtime_model,
        mse=None,
        psnr=None,
        max_value=None,
        runtime_model_path=(
            str(runtime_model_path) if method_requires_runtime_model(method) else None
        ),
        command=process_command,
        nonfinite_pixel_boolean=nonfinite_pixel_boolean,
    )


def attach_metrics(
    records: tuple[BenchmarkRecord, ...],
    *,
    external_reference_paths: dict[str, Path] | None = None,
) -> tuple[BenchmarkRecord, ...]:
    """Populate MSE/PSNR comparing every successful row against HDR references.

    Args:
        records: Raw BenchmarkRecord tuples emitted by Mitsuba launchers.
        external_reference_paths: Explicit verified scene-to-EXR mapping. When
            present, implicit lookup of an older run is disabled.

    Returns:
        Tuple with metrics filled (or flagged missing) alongside originals.
    """

    reference_map = (
        dict(external_reference_paths)
        if external_reference_paths is not None
        else {
            record.scene_id: Path(record.output_path)
            for record in records
            if record.method == "path_reference" and record.ok
        }
    )
    enriched: list[BenchmarkRecord] = []
    for record in records:
        if not record.ok:
            enriched.append(record)
            continue
        if record.method == "path_reference":
            enriched.append(
                BenchmarkRecord(
                    **{
                        **asdict(record),
                        "mse": 0.0,
                        "psnr": float("inf"),
                        "max_value": None,
                    }
                )
            )
            continue

        reference_path = reference_map.get(record.scene_id)
        if reference_path is None and external_reference_paths is None:
            candidate_root = Path(record.output_path).resolve().parents[2]
            candidate_references = sorted(
                candidate_root.glob(
                    f"run_*/outputs/{record.scene_id}.path_reference.exr"
                ),
                key=lambda item: item.stat().st_mtime,
            )
            if candidate_references:
                reference_path = candidate_references[-1]
        if reference_path is None:
            LOGGER.warning("missing path reference for metrics: %s / %s", record.scene_id, record.method)
            enriched.append(
                BenchmarkRecord(
                    **{
                        **asdict(record),
                        "mse": None,
                        "psnr": None,
                        "max_value": None,
                    }
                )
            )
            continue
        mse, psnr, max_value = compute_metrics(Path(record.output_path), reference_path)
        enriched.append(
            BenchmarkRecord(
                **{
                    **asdict(record),
                    "mse": mse,
                    "psnr": psnr,
                    "max_value": max_value,
                }
            )
        )
    return tuple(enriched)


def build_reference_provenance(
    *,
    config: Phase13BenchmarkConfig,
    records: tuple[BenchmarkRecord, ...],
    external_bundle: ResolvedSharedReferenceBundle | None,
) -> dict[str, object]:
    """Describe the exact reference bytes used by this benchmark."""

    if external_bundle is not None:
        return serialize_external_reference_bundle(external_bundle)

    references: dict[str, dict[str, object]] = {}
    for record in records:
        if record.method != "path_reference" or not record.ok:
            continue
        if record.output_sha256 is None:
            raise Phase13BenchmarkError(
                f"successful reference is missing output hash: {record.scene_id}"
            )
        references[record.scene_id] = {
            "scene_id": record.scene_id,
            "exr_path": relativize_export_path(record.output_path, PROJECT_ROOT),
            "exr_sha256": record.output_sha256,
            "output_size": record.output_size,
            "sample_count": record.sample_count,
            "sampler_seed": record.sampler_seed,
            "active_config_sha256": record.active_config_sha256,
            "width": config.width,
            "height": config.height,
            "crop_width": config.crop_width,
            "crop_height": config.crop_height,
        }
    return {
        "mode": "rendered_in_run" if references else "unavailable",
        "manifest_path": None,
        "manifest_sha256": None,
        "source_summary": None,
        "source_summary_sha256": None,
        "references": references,
    }


def write_shared_reference_manifest(
    *,
    workspace_root: Path,
    output_reference: str,
    summary_path: Path,
    summary: dict[str, object],
) -> Path:
    """Atomically publish a ready manifest after a reference-only run succeeds."""

    if int(summary["failed"]) != 0:
        raise Phase13BenchmarkError(
            "cannot publish shared references from a failed benchmark"
        )
    provenance = summary["reference_provenance"]
    if provenance["mode"] != "rendered_in_run":
        raise Phase13BenchmarkError("shared-reference source has no internal references")
    references = provenance["references"]
    expected_scenes = {
        str(record["scene_id"])
        for record in summary["records"]
        if record["method"] == "path_reference"
    }
    if set(references) != expected_scenes or not references:
        raise Phase13BenchmarkError("shared-reference source set is incomplete")
    if summary.get("metrics_status") != "computed":
        raise Phase13BenchmarkError(
            "shared-reference source must complete its ordinary validation"
        )

    # Recheck the published bytes immediately before making the ready sentinel
    # visible. This closes the gap between per-render hashing and publication.
    for scene_id, reference in references.items():
        exr_path = _resolve_project_artifact(
            reference.get("exr_path"),
            label=f"shared reference EXR for {scene_id}",
        )
        expected_hash = _require_sha256(
            reference.get("exr_sha256"),
            label=f"reference EXR {scene_id}",
        )
        if compute_file_sha256(exr_path) != expected_hash:
            raise Phase13BenchmarkError(
                f"shared reference EXR changed before publication: {exr_path}"
            )
        if exr_path.stat().st_size != int(reference.get("output_size", -1)):
            raise Phase13BenchmarkError(
                f"shared reference EXR size changed before publication: {exr_path}"
            )

    output_path = Path(output_reference)
    if not output_path.is_absolute():
        output_path = (workspace_root / output_path).resolve()
    try:
        output_path.relative_to(PROJECT_ROOT.resolve())
    except ValueError as exc:
        raise Phase13BenchmarkError(
            f"shared reference manifest must stay under project root: {output_path}"
        ) from exc
    if output_path.exists():
        raise Phase13BenchmarkError(
            f"refusing to overwrite shared reference manifest: {output_path}"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    source_summary_sha256 = compute_file_sha256(summary_path)
    reference_set_id = hashlib.sha256(
        json.dumps(references, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    payload = {
        "schema_version": 1,
        "status": "ready",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "path_base": "project_root",
        "reference_set_id": reference_set_id,
        "source_summary": relativize_export_path(str(summary_path), PROJECT_ROOT),
        "source_summary_sha256": source_summary_sha256,
        "references": references,
    }
    temporary_path = output_path.with_name(
        f".{output_path.name}.{os.getpid()}.tmp"
    )
    temporary_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    try:
        try:
            os.link(temporary_path, output_path)
        except FileExistsError as exc:
            raise Phase13BenchmarkError(
                f"refusing to overwrite shared reference manifest: {output_path}"
            ) from exc
    finally:
        temporary_path.unlink(missing_ok=True)
    return output_path


def write_csv(records: tuple[BenchmarkRecord, ...], output_path: Path) -> None:
    """Dump flattened BenchmarkRecord tuples to CSV."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    serialized_rows = [serialize_record(record, PROJECT_ROOT) for record in records]
    fieldnames = list(serialized_rows[0].keys())
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(serialized_rows)


def build_summary_payload(
    *,
    config: Phase13BenchmarkConfig,
    run_dir: Path,
    records: tuple[BenchmarkRecord, ...],
    reference_provenance: dict[str, object],
    metrics_status: str,
) -> dict[str, object]:
    """Shape JSON blobs consumed by Markdown + tooling downstream.

    Args:
        config: Phase13 knobs copied verbatim into summary artifact.
        run_dir: Location containing EXR/log scaffolding.
        records: Hydrated telemetry rows (`attach_metrics` should run first).
        reference_provenance: Hash-verified internal/external reference identity.
        metrics_status: Whether metrics were computed now or intentionally deferred.

    Returns:
        JSON-serializable dict mirroring Markdown tables + aggregate stats.
    """

    compare_records = tuple(
        record for record in records if record.method != "path_reference" and record.ok
    )
    compare_methods = sorted({record.method for record in compare_records})
    average_by_method: dict[str, dict[str, float | None]] = {}
    for method in compare_methods:
        method_records = [
            record for record in compare_records if record.method == method
        ]
        if not method_records:
            continue
        mse_values = [float(record.mse) for record in method_records if record.mse is not None]
        psnr_values = [float(record.psnr) for record in method_records if record.psnr is not None]
        average_by_method[method] = {
            "mean_mse": float(np.mean(mse_values)) if mse_values else None,
            "mean_psnr": float(np.mean(psnr_values)) if psnr_values else None,
            "mean_seconds": float(
                np.mean([float(record.seconds) for record in method_records])
            ),
        }

    baseline_methods = tuple(
        method
        for method in compare_methods
        if method
        in {
            "pt",
            "sms_star",
            "uniform",
            "mpg",
            "mpg_nofilter",
            "mpg_stocfilter",
            "mpg_knn",
            "mpg_m19",
        }
    )
    candidate_methods = tuple(
        method for method in compare_methods if method not in baseline_methods
    )
    wins: dict[str, int] = {}
    for method in candidate_methods:
        for baseline_method in baseline_methods:
            wins[f"{method}_better_than_{baseline_method}_mse"] = 0
            wins[f"{method}_better_than_{baseline_method}_psnr"] = 0
    scenes = sorted({record.scene_id for record in compare_records})
    for scene_id in scenes:
        scene_map = {
            record.method: record
            for record in compare_records
            if record.scene_id == scene_id
        }
        for method in candidate_methods:
            candidate = scene_map.get(method)
            if candidate is None:
                continue
            for baseline_method in baseline_methods:
                baseline_record = scene_map.get(baseline_method)
                if baseline_record is None:
                    continue
                if (
                    candidate.mse is not None
                    and baseline_record.mse is not None
                    and float(candidate.mse) <= float(baseline_record.mse)
                ):
                    wins[f"{method}_better_than_{baseline_method}_mse"] += 1
                if (
                    candidate.psnr is not None
                    and baseline_record.psnr is not None
                    and float(candidate.psnr) >= float(baseline_record.psnr)
                ):
                    wins[f"{method}_better_than_{baseline_method}_psnr"] += 1

    serialized_records = [serialize_record(record, PROJECT_ROOT) for record in records]
    return {
        "config": {
            **asdict(config),
            "sampler_seed": config.seed,
            "reference_sampler_seed": (
                config.seed if config.reference_seed is None else config.reference_seed
            ),
        },
        "run_dir": relativize_export_path(str(run_dir), PROJECT_ROOT),
        "total": len(records),
        "passed": sum(1 for record in records if record.ok),
        "failed": sum(1 for record in records if not record.ok),
        "metrics_status": metrics_status,
        "reference_provenance": reference_provenance,
        "records": serialized_records,
        "average_by_method": average_by_method,
        "wins": wins,
    }


def write_markdown(summary: dict[str, object], output_path: Path) -> None:
    """Render human-readable Markdown tables from structured summary blobs."""

    records = tuple(BenchmarkRecord(**record) for record in summary["records"])
    lines = [
        "# Phase 13 Formal Benchmark",
        "",
        "## Config",
        "",
        f"- profile_id: `{summary['config']['profile_id']}`",
        f"- width: `{summary['config']['width']}`",
        f"- height: `{summary['config']['height']}`",
        f"- reference_sample_count: `{summary['config']['reference_sample_count']}`",
        f"- compare_sample_count: `{summary['config']['compare_sample_count']}`",
        f"- threads: `{summary['config']['threads']}`",
        f"- sampler_seed: `{summary['config']['sampler_seed']}`",
        f"- diffusion_steps: `{summary['config'].get('diffusion_steps', 1)}`",
        f"- bounce_alpha: `{summary['config'].get('bounce_alpha', 0.5)}`",
        (
            "- learned_direction_kappa_scale: `"
            f"{summary['config'].get('learned_direction_kappa_scale', 1.0)}`"
        ),
        (
            "- learned_direction_flip_to_receiver_hemisphere: `"
            f"{summary['config'].get('learned_direction_flip_to_receiver_hemisphere', True)}`"
        ),
        f"- metrics_status: `{summary.get('metrics_status', 'computed')}`",
        (
            "- external_reference_manifest: `"
            f"{summary['config'].get('external_reference_manifest')}`"
        ),
        "",
        "## Reference provenance",
        "",
        f"- mode: `{summary['reference_provenance']['mode']}`",
        (
            "- manifest_sha256: `"
            f"{summary['reference_provenance'].get('manifest_sha256')}`"
        ),
        "",
        "| scene | spp | sampler seed | EXR SHA-256 | XML SHA-256 | path |",
        "| --- | ---: | ---: | --- | --- | --- |",
        *[
            "| {scene} | {spp} | {seed} | {exr_hash} | {xml_hash} | {path} |".format(
                scene=scene_id,
                spp=reference["sample_count"],
                seed=reference["sampler_seed"],
                exr_hash=reference["exr_sha256"],
                xml_hash=reference["active_config_sha256"],
                path=reference["exr_path"],
            )
            for scene_id, reference in sorted(
                summary["reference_provenance"]["references"].items()
            )
        ],
        "",
        "## Records",
        "",
        "| scene | method | active_method | sampler_seed | config_sha256 | ok | seconds | spp | mse | psnr | runtime_model |",
        "| --- | --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for record in records:
        mse_text = "n/a" if record.mse is None else f"{record.mse:.6f}"
        psnr_text = "n/a" if record.psnr is None else f"{record.psnr:.4f}"
        lines.append(
            f"| {record.scene_id} | {record.method} | {record.active_method} | "
            f"{record.sampler_seed} | {record.active_config_sha256[:12]} | {int(record.ok)} | "
            f"{record.seconds:.3f} | {record.sample_count} | {mse_text} | {psnr_text} | "
            f"{int(record.loaded_runtime_model)} |"
        )

    lines.extend(
        [
            "",
            "## Averages",
            "",
            "| method | mean_mse | mean_psnr | mean_seconds |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    average_by_method = summary["average_by_method"]
    for method in average_by_method:
        values = average_by_method.get(method)
        if values is None:
            continue
        mean_mse = values["mean_mse"]
        mean_psnr = values["mean_psnr"]
        mean_mse_text = "n/a" if mean_mse is None else f"{mean_mse:.6f}"
        mean_psnr_text = "n/a" if mean_psnr is None else f"{mean_psnr:.4f}"
        lines.append(
            f"| {method} | {mean_mse_text} | "
            f"{mean_psnr_text} | {values['mean_seconds']:.3f} |"
        )

    wins = summary["wins"]
    lines.extend(
        [
            "",
            "## Wins",
            "",
        ]
    )
    for key, value in wins.items():
        lines.append(f"- {key}: `{value}`")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    """Phase 13 CLI dispatcher.

    Returns:
        POSIX exit code (`0` success, `1` failure).
    """

    configure_logging()
    args = parse_args()
    config = load_config(args.config)
    if args.bounce_alpha is not None:
        config = replace(
            config,
            bounce_alpha=validate_bounce_alpha(args.bounce_alpha),
        )
    if args.learned_direction_kappa_scale is not None:
        config = replace(
            config,
            learned_direction_kappa_scale=(
                validate_learned_direction_kappa_scale(
                    args.learned_direction_kappa_scale
                )
            ),
        )
    if args.learned_direction_flip_to_receiver_hemisphere is not None:
        config = replace(
            config,
            learned_direction_flip_to_receiver_hemisphere=(
                args.learned_direction_flip_to_receiver_hemisphere
            ),
        )
    set_global_seed(config.seed)

    manifest = load_baseline_manifest(
        WORKSPACE_ROOT / "configs" / "baseline_manifest.json"
    )
    jobs = select_render_jobs(manifest.render_jobs, config.job_ids)
    external_bundle = (
        load_external_reference_bundle(
            workspace_root=WORKSPACE_ROOT,
            config=config,
            jobs=jobs,
            validate_pixels=not config.defer_metrics,
        )
        if config.external_reference_manifest is not None
        else None
    )
    host = detect_host_runtime(PROJECT_ROOT)
    requested_profile_id = normalize_runtime_profile_id(config.profile_id, host)
    if requested_profile_id != config.profile_id:
        LOGGER.info(
            "phase_13 runtime profile remapped: %s -> %s",
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
        ensure_runtime_profile_built(PROJECT_ROOT, requested_profile_id)
        runtime_profiles = tuple(
            resolve_runtime_profile(template, host)
            for template in load_runtime_profiles(
                WORKSPACE_ROOT / "configs" / "runtime_profiles.json"
            )
        )
        profile = select_runtime_profile(runtime_profiles, requested_profile_id)
    runtime_env = build_runtime_env(profile)

    run_dir = create_run_dir((WORKSPACE_ROOT / config.output_dir).resolve())
    (run_dir / "benchmark_config.json").write_text(
        json.dumps(
            {
                **asdict(config),
                "sampler_seed": config.seed,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    methods = config.methods
    records: list[BenchmarkRecord] = []
    for job in jobs:
        for method in methods:
            LOGGER.info("benchmark render: %s / %s", job.job_id, method)
            record = run_render(
                profile=profile,
                runtime_env=runtime_env,
                config=config,
                job=job,
                method=method,
                run_dir=run_dir,
                runtime_model_path=(
                    resolve_runtime_model_path(
                        workspace_root=WORKSPACE_ROOT,
                        config=config,
                        job=job,
                        method=method,
                    )
                    if method_requires_runtime_model(method)
                    else None
                ),
            )
            records.append(record)

    if external_bundle is not None:
        verify_external_reference_bundle(external_bundle)
    if config.defer_metrics:
        enriched_records = tuple(records)
        metrics_status = "deferred"
    else:
        enriched_records = attach_metrics(
            tuple(records),
            external_reference_paths=(
                None
                if external_bundle is None
                else {
                    scene_id: reference.exr_path
                    for scene_id, reference in external_bundle.references.items()
                }
            ),
        )
        metrics_status = "computed"
    reference_provenance = build_reference_provenance(
        config=config,
        records=tuple(records),
        external_bundle=external_bundle,
    )
    summary = build_summary_payload(
        config=config,
        run_dir=run_dir,
        records=enriched_records,
        reference_provenance=reference_provenance,
        metrics_status=metrics_status,
    )
    summary_json = run_dir / "benchmark_summary.json"
    summary_csv = run_dir / "benchmark_summary.csv"
    summary_md = run_dir / "benchmark_summary.md"
    summary_json.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_csv(enriched_records, summary_csv)
    write_markdown(summary, summary_md)

    if config.shared_reference_manifest_output is not None:
        shared_manifest_path = write_shared_reference_manifest(
            workspace_root=WORKSPACE_ROOT,
            output_reference=config.shared_reference_manifest_output,
            summary_path=summary_json,
            summary=summary,
        )
        LOGGER.info("shared_reference_manifest: %s", shared_manifest_path)

    LOGGER.info(
        "phase_13 benchmark finished: passed=%s failed=%s",
        summary["passed"],
        summary["failed"],
    )
    LOGGER.info("summary_json: %s", summary_json)
    return 0 if int(summary["failed"]) == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
