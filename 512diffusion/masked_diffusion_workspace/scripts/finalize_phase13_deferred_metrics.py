#!/usr/bin/env python3
"""Compute Phase13 image metrics after an explicit experiment barrier unlock."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = WORKSPACE_ROOT.parent
SCRIPTS_ROOT = WORKSPACE_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

import phase_13_run_formal_benchmark as phase13  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Materialize metrics for one hash-locked deferred Phase13 run."
    )
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument(
        "--unlock-metrics",
        action="store_true",
        help="Required acknowledgement that the experiment-wide barrier is complete.",
    )
    return parser.parse_args()


def hydrate_config(payload: dict[str, Any]) -> phase13.Phase13BenchmarkConfig:
    """Rebuild a typed Phase13 config from its serialized summary copy."""

    allowed = {field.name for field in fields(phase13.Phase13BenchmarkConfig)}
    values = {key: value for key, value in payload.items() if key in allowed}
    for key in ("job_ids", "methods", "runtime_only_methods"):
        values[key] = tuple(values.get(key, ()))
    config = phase13.Phase13BenchmarkConfig(**values)
    phase13.validate_reference_mode(config)
    return config


def resolve_record_path(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def hydrate_records(rows: list[dict[str, Any]]) -> tuple[phase13.BenchmarkRecord, ...]:
    """Restore absolute artifact paths and verify frozen candidate bytes."""

    records: list[phase13.BenchmarkRecord] = []
    for row in rows:
        if row.get("method") == "path_reference":
            raise phase13.Phase13BenchmarkError(
                "deferred external-reference summary must be candidate-only"
            )
        if row.get("mse") is not None or row.get("psnr") is not None:
            raise phase13.Phase13BenchmarkError(
                "source summary already contains image metrics"
            )
        output_path = resolve_record_path(str(row["output_path"]))
        expected_hash = phase13._require_sha256(
            row.get("output_sha256"), label=f"candidate output {row.get('scene_id')}"
        )
        if phase13.compute_file_sha256(output_path) != expected_hash:
            raise phase13.Phase13BenchmarkError(
                f"candidate output hash mismatch: {output_path}"
            )
        hydrated = dict(row)
        for key in ("output_path", "patched_scene_record", "stdout_log", "stderr_log"):
            hydrated[key] = str(resolve_record_path(str(row[key])))
        runtime_path = row.get("runtime_model_path")
        if runtime_path is not None:
            hydrated["runtime_model_path"] = str(resolve_record_path(str(runtime_path)))
        records.append(phase13.BenchmarkRecord(**hydrated))
    return tuple(records)


def validate_deferred_source(
    *,
    source: dict[str, Any],
    config: phase13.Phase13BenchmarkConfig,
    jobs: tuple[phase13.RenderJob, ...],
    bundle: phase13.ResolvedSharedReferenceBundle,
) -> None:
    """Bind a deferred candidate summary to its original references and grid."""

    if source.get("metrics_status") != "deferred":
        raise phase13.Phase13BenchmarkError(
            "source summary is not a deferred-metrics run"
        )
    if int(source.get("failed", -1)) != 0:
        raise phase13.Phase13BenchmarkError(
            "cannot finalize an incomplete deferred candidate run"
        )
    expected_provenance = phase13.serialize_external_reference_bundle(bundle)
    if source.get("reference_provenance") != expected_provenance:
        raise phase13.Phase13BenchmarkError(
            "source summary reference provenance no longer matches the locked bundle"
        )

    expected_scenes = {job.job_id.removesuffix("_mpg") for job in jobs}
    jobs_by_scene = {job.job_id.removesuffix("_mpg"): job for job in jobs}
    expected_pairs = {
        (scene_id, method)
        for scene_id in expected_scenes
        for method in config.methods
    }
    records = source.get("records")
    if not isinstance(records, list):
        raise phase13.Phase13BenchmarkError("source summary records must be a list")
    actual_pairs: list[tuple[str, str]] = []
    for row in records:
        if not isinstance(row, dict):
            raise phase13.Phase13BenchmarkError(
                "source summary record must be an object"
            )
        scene_id = str(row.get("scene_id"))
        method = str(row.get("method"))
        actual_pairs.append((scene_id, method))
        job = jobs_by_scene.get(scene_id)
        if job is None or method not in config.methods:
            continue
        expected_spp = phase13.resolve_compare_sample_count(
            config=config,
            job=job,
            method=method,
        )
        if row.get("ok") is not True or int(row.get("return_code", -1)) != 0:
            raise phase13.Phase13BenchmarkError(
                f"deferred candidate did not complete successfully: {scene_id}/{method}"
            )
        if row.get("active_method") != method:
            raise phase13.Phase13BenchmarkError(
                f"active method mismatch: {scene_id}/{method}"
            )
        if int(row.get("sampler_seed", -1)) != config.seed:
            raise phase13.Phase13BenchmarkError(
                f"candidate sampler seed mismatch: {scene_id}/{method}"
            )
        if int(row.get("sample_count", -1)) != expected_spp:
            raise phase13.Phase13BenchmarkError(
                f"candidate SPP mismatch: {scene_id}/{method}"
            )
        if row.get("nonfinite_pixel_boolean") is not False:
            raise phase13.Phase13BenchmarkError(
                f"candidate pixels were not validated finite: {scene_id}/{method}"
            )
        if phase13.method_requires_runtime_model(method) and row.get(
            "loaded_runtime_model"
        ) is not True:
            raise phase13.Phase13BenchmarkError(
                f"runtime-loaded marker missing: {scene_id}/{method}"
            )
    if len(actual_pairs) != len(expected_pairs) or set(actual_pairs) != expected_pairs:
        raise phase13.Phase13BenchmarkError(
            "deferred candidate record grid differs from the configured scene/method grid"
        )


def write_exclusive(path: Path, text: str) -> None:
    """Write a post-barrier result once; never overwrite prior evidence."""

    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as handle:
            handle.write(text)
    except FileExistsError as exc:
        raise phase13.Phase13BenchmarkError(
            f"refusing to overwrite finalized metrics: {path}"
        ) from exc


def main() -> int:
    args = parse_args()
    if not args.unlock_metrics:
        raise phase13.Phase13BenchmarkError(
            "refusing to read images before explicit --unlock-metrics"
        )
    source_summary_path = args.summary.resolve()
    source_summary_sha256 = phase13.compute_file_sha256(source_summary_path)
    source = json.loads(source_summary_path.read_text(encoding="utf-8"))
    config = hydrate_config(source["config"])
    manifest = phase13.load_baseline_manifest(
        WORKSPACE_ROOT / "configs" / "baseline_manifest.json"
    )
    jobs = phase13.select_render_jobs(manifest.render_jobs, config.job_ids)
    bundle = phase13.load_external_reference_bundle(
        workspace_root=WORKSPACE_ROOT,
        config=config,
        jobs=jobs,
        validate_pixels=True,
    )
    phase13.verify_external_reference_bundle(bundle)
    validate_deferred_source(
        source=source,
        config=config,
        jobs=jobs,
        bundle=bundle,
    )
    records = hydrate_records(source["records"])
    enriched = phase13.attach_metrics(
        records,
        external_reference_paths={
            scene_id: reference.exr_path
            for scene_id, reference in bundle.references.items()
        },
    )
    phase13.verify_external_reference_bundle(bundle)
    run_dir = source_summary_path.parent
    summary = phase13.build_summary_payload(
        config=config,
        run_dir=run_dir,
        records=enriched,
        reference_provenance=phase13.serialize_external_reference_bundle(bundle),
        metrics_status="computed_after_deferred_unlock",
    )
    summary["deferred_source_summary"] = {
        "path": phase13.relativize_export_path(
            str(source_summary_path), PROJECT_ROOT
        ),
        "sha256": source_summary_sha256,
    }
    summary["metrics_unlocked_at_utc"] = datetime.now(timezone.utc).isoformat()

    output_json = run_dir / "benchmark_summary_with_metrics.json"
    output_csv = run_dir / "benchmark_summary_with_metrics.csv"
    output_markdown = run_dir / "benchmark_summary_with_metrics.md"
    existing = [
        path for path in (output_json, output_csv, output_markdown) if path.exists()
    ]
    if existing:
        raise phase13.Phase13BenchmarkError(
            f"refusing to overwrite finalized metrics: {existing}"
        )
    write_exclusive(
        output_json,
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
    )
    phase13.write_csv(enriched, output_csv)
    phase13.write_markdown(summary, output_markdown)
    print(output_json)
    print(output_markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
