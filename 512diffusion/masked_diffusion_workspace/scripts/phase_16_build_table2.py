from __future__ import annotations

import argparse
import csv
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
LOGGER = logging.getLogger("phase_16_build_table2")


@dataclass(slots=True, frozen=True)
class TableRowSpec:
    """Metadata for one paper-table row."""

    scene_id: str
    display_name: str
    figure: str
    chain_types: str
    summary_path: Path
    method: str


def configure_logging() -> None:
    """Initialize logging."""

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Build Table-2-style statistics from benchmark logs."
    )
    parser.add_argument(
        "--config", type=Path, required=True, help="JSON config for table rows."
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, object]:
    """Load JSON from disk."""

    return json.loads(path.read_text(encoding="utf-8"))


def load_row_specs(path: Path) -> tuple[Path, tuple[TableRowSpec, ...]]:
    """Load Table 2 generator config.

    Args:
        path: Config file path.

    Returns:
        Output directory tuple with row specs.
    """

    payload = load_json(path)
    output_dir = (WORKSPACE_ROOT / str(payload["output_dir"])).resolve()
    rows = tuple(
        TableRowSpec(
            scene_id=str(item["scene_id"]),
            display_name=str(item["display_name"]),
            figure=str(item["figure"]),
            chain_types=str(item["chain_types"]),
            summary_path=(WORKSPACE_ROOT / str(item["summary_path"])).resolve(),
            method=str(item["method"]),
        )
        for item in payload["rows"]
    )
    return output_dir, rows


def extract_float(text: str, pattern: str) -> float | None:
    """Extract a floating point metric from logs."""

    match = re.search(pattern, text)
    if match is None:
        return None
    return float(match.group(1))


def extract_int(text: str, pattern: str) -> int | None:
    """Extract an integer metric from logs."""

    match = re.search(pattern, text)
    if match is None:
        return None
    return int(match.group(1))


def parse_render_log(stdout_log: Path) -> dict[str, float | int | None]:
    """Collect table fields from Mitsuba stdout logs.

    Args:
        stdout_log: Path to Mitsuba stdout log.

    Returns:
        Metric dictionary for spreadsheet export.
    """

    text = stdout_log.read_text(encoding="utf-8", errors="replace")
    return {
        "guide_seconds": extract_float(
            text,
            r"GuidedManifoldSampler stats perf_time_guide ([0-9eE.+-]+)",
        ),
        "pdf_seconds": extract_float(
            text,
            r"GuidedManifoldSampler stats perf_time_eval ([0-9eE.+-]+)",
        ),
        "specular_total_seconds": extract_float(
            text,
            r"GuidedManifoldSampler stats perf_time_all ([0-9eE.+-]+)",
        ),
        "subpath_samples": extract_int(text, r"Manifold perf_manifold_query (\d+)"),
        "logged_sample_count": extract_int(
            text,
            r"Starting render job \([0-9]+x[0-9]+,\s*([0-9]+)\s+samples",
        ),
    }


def pick_record(
    summary: dict[str, object], scene_id: str, method: str
) -> dict[str, object]:
    """Pick the benchmark record matching scene/method."""

    for record in summary["records"]:
        if str(record["scene_id"]) == scene_id and str(record["method"]) == method:
            return dict(record)
    raise RuntimeError(f"summary missing record: {scene_id} / {method}")


def build_row(spec: TableRowSpec) -> dict[str, object]:
    """Assemble spreadsheet row dictionaries."""

    summary = load_json(spec.summary_path)
    record = pick_record(summary, spec.scene_id, spec.method)
    log_metrics = parse_render_log(Path(str(record["stdout_log"])).resolve())
    summary_config = dict(summary["config"])

    total_budget_seconds = summary_config.get("time_budget_seconds")
    if total_budget_seconds is None:
        total_budget_seconds = float(record["seconds"])
    total_budget_minutes = float(total_budget_seconds) / 60.0

    train_budget_ratio = (
        float(summary_config.get("mpg_train_budget", 0.0))
        if spec.method
        in {
            "mpg",
            "masked_diffusion",
            "masked_hybrid",
            "ppg",
            "vapg",
            "vmmpg",
        }
        else 0.0
    )
    sample_count = int(record["sample_count"])
    training_spp = int(round(sample_count * train_budget_ratio))
    rendering_spp = max(0, sample_count - training_spp)

    return {
        "scene": spec.display_name,
        "figure": spec.figure,
        "chain_types": spec.chain_types,
        "budget_total_min": round(total_budget_minutes, 3),
        "budget_training_min": round(total_budget_minutes * train_budget_ratio, 3),
        "spp_training": training_spp,
        "spp_rendering": rendering_spp,
        "time_specular_guide_min": round(
            float(log_metrics["guide_seconds"] or 0.0) / 60.0, 3
        ),
        "time_specular_pdf_min": round(
            float(log_metrics["pdf_seconds"] or 0.0) / 60.0, 3
        ),
        "time_specular_total_min": round(
            float(log_metrics["specular_total_seconds"] or 0.0) / 60.0, 3
        ),
        "subpath_samples": int(log_metrics["subpath_samples"] or 0),
        "psnr": (
            None if record.get("psnr") is None else round(float(record["psnr"]), 6)
        ),
        "method": spec.method,
        "summary_path": str(spec.summary_path),
        "stdout_log": str(record["stdout_log"]),
    }


def write_csv(output_path: Path, rows: list[dict[str, object]]) -> None:
    """Write rows to CSV."""

    fieldnames = list(rows[0].keys())
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(output_path: Path, rows: list[dict[str, object]]) -> None:
    """Write Markdown table artifact."""

    lines = [
        "# Table 2 Style Statistics",
        "",
        "| Scene | Figure | Chain types | Budget Total (min.) | Budget Training (min.) | Sample per pixel Training | Sample per pixel Rendering | Time on specular Guide (min.) | Time on specular PDF (min.) | Time on specular Total (min.) | #Sub-path samples | PSNR |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        psnr_text = "-" if row["psnr"] is None else f"{row['psnr']:.6f}"
        lines.append(
            f"| {row['scene']} | {row['figure']} | {row['chain_types']} | "
            f"{row['budget_total_min']:.3f} | {row['budget_training_min']:.3f} | "
            f"{row['spp_training']} | {row['spp_rendering']} | "
            f"{row['time_specular_guide_min']:.3f} | {row['time_specular_pdf_min']:.3f} | "
            f"{row['time_specular_total_min']:.3f} | {row['subpath_samples']} | "
            f"{psnr_text} |"
        )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    """CLI entry producing CSV + Markdown summaries."""

    configure_logging()
    args = parse_args()
    output_dir, row_specs = load_row_specs(args.config.resolve())
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [build_row(row_spec) for row_spec in row_specs]

    (output_dir / "table2_style.json").write_text(
        json.dumps({"rows": rows}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_csv(output_dir / "table2_style.csv", rows)
    write_markdown(output_dir / "table2_style.md", rows)
    LOGGER.info("table2 written: %s", output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
