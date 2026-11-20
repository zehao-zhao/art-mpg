from __future__ import annotations

import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = WORKSPACE_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

import phase_13_run_formal_benchmark as phase13  # noqa: E402


def preserve_status_records(
    records: tuple[Any, ...], *, external_reference_paths: dict[str, Path] | None = None
) -> tuple[Any, ...]:
    """Return render status records without decoding a reference/candidate pair."""

    del external_reference_paths
    return records


def reject_quality_metric_computation(*args: object, **kwargs: object) -> None:
    """Fail closed if a future Phase 13 path tries to compute a quality metric."""

    del args, kwargs
    raise phase13.Phase13BenchmarkError(
        "quality-metric computation is disabled in status-only smoke mode"
    )


def build_status_only_summary(
    *,
    config: Any,
    run_dir: Path,
    records: tuple[Any, ...],
    reference_provenance: dict[str, object],
    metrics_status: str,
) -> dict[str, object]:
    """Build the Phase 13 summary without aggregating quality measurements."""

    del metrics_status
    if any(
        record.mse is not None
        or record.psnr is not None
        or record.max_value is not None
        for record in records
    ):
        raise phase13.Phase13BenchmarkError(
            "status-only smoke received an unexpected quality value"
        )
    return {
        "config": {
            **asdict(config),
            "sampler_seed": config.seed,
            "reference_sampler_seed": (
                config.seed if config.reference_seed is None else config.reference_seed
            ),
        },
        "run_dir": phase13.relativize_export_path(str(run_dir), phase13.PROJECT_ROOT),
        "total": len(records),
        "passed": sum(1 for record in records if record.ok),
        "failed": sum(1 for record in records if not record.ok),
        "metrics_status": "status_only",
        "quality_metrics_computed": False,
        "quality_metrics_inspected": False,
        "reference_provenance": reference_provenance,
        "records": [
            phase13.serialize_record(record, phase13.PROJECT_ROOT) for record in records
        ],
        "average_by_method": {},
        "wins": {},
    }


def write_status_only_markdown(summary: dict[str, object], output_path: Path) -> None:
    """Write only execution/provenance fields; never surface image-quality fields."""

    lines = [
        "# Phase 13 status-only smoke",
        "",
        f"- metrics_status: `{summary['metrics_status']}`",
        f"- total: `{summary['total']}`",
        f"- passed: `{summary['passed']}`",
        f"- failed: `{summary['failed']}`",
        "",
        "| scene | method | ok | return code | seconds | spp | runtime loaded | output |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for record in summary["records"]:
        lines.append(
            f"| {record['scene_id']} | {record['method']} | {int(record['ok'])} | "
            f"{record['return_code']} | {record['seconds']:.3f} | "
            f"{record['sample_count']} | {int(record['loaded_runtime_model'])} | "
            f"{record['output_path']} |"
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def install_status_only_mode() -> None:
    phase13.compute_metrics = reject_quality_metric_computation
    phase13.attach_metrics = preserve_status_records
    phase13.build_summary_payload = build_status_only_summary
    phase13.write_markdown = write_status_only_markdown


def main() -> int:
    install_status_only_mode()
    return phase13.main()


if __name__ == "__main__":
    raise SystemExit(main())
