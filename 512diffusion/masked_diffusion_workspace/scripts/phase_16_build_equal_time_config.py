from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
LOGGER = logging.getLogger("phase_16_build_equal_time_config")


@dataclass(slots=True, frozen=True)
class Phase16BuildConfig:
    """Parameters used to regenerate equal-wall-time benchmark configs.

    Attributes:
        source_summary_path: Input benchmark summary JSON.
        output_config_path: Output Phase 13 config path.
        output_dir: Output directory string for rerun benchmark.
        target_seconds: Time budget per method in seconds.
        scale_safety: Safety factor when extrapolating sample counts.
        reference_sample_count: Optional overrides for reference pass samples.
        min_sample_count: Lower bound clamp for sample counts.
        max_sample_count: Upper bound clamp for sample counts.
    """

    source_summary_path: Path
    output_config_path: Path
    output_dir: str
    target_seconds: float
    scale_safety: float
    reference_sample_count: int | None
    min_sample_count: int
    max_sample_count: int


def configure_logging() -> None:
    """Initialize logging."""

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(
        description="Derive unified time-budget configs from prior benchmark runs.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to Phase 16 JSON describing inputs and constraints.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, object]:
    """Load JSON blob from disk.

    Args:
        path: JSON filesystem path.

    Returns:
        Parsed mapping.
    """

    return json.loads(path.read_text(encoding="utf-8"))


def load_config(path: Path) -> Phase16BuildConfig:
    """Load validated Phase 16 configuration.

    Args:
        path: Config path.

    Returns:
        Parsed config dataclass instance.
    """

    payload = load_json(path)
    return Phase16BuildConfig(
        source_summary_path=(
            WORKSPACE_ROOT / str(payload["source_summary_path"])
        ).resolve(),
        output_config_path=(
            WORKSPACE_ROOT / str(payload["output_config_path"])
        ).resolve(),
        output_dir=str(payload["output_dir"]),
        target_seconds=float(payload["target_seconds"]),
        scale_safety=float(payload.get("scale_safety", 0.95)),
        reference_sample_count=(
            None
            if payload.get("reference_sample_count") is None
            else int(payload["reference_sample_count"])
        ),
        min_sample_count=int(payload.get("min_sample_count", 1)),
        max_sample_count=int(payload.get("max_sample_count", 2_000_000)),
    )


def estimate_sample_count(
    *,
    sample_count: int,
    seconds: float,
    target_seconds: float,
    scale_safety: float,
    min_sample_count: int,
    max_sample_count: int,
) -> int:
    """Estimate target sample count assuming linear scaling.

    Args:
        sample_count: Current sample budget.
        seconds: Observed elapsed seconds.
        target_seconds: Desired seconds per method run.
        scale_safety: Multiplicative safety fudge.
        min_sample_count: Lower clamp.
        max_sample_count: Upper clamp.

    Returns:
        Estimated sample budget after clamps.
    """

    safe_seconds = max(seconds, 1e-6)
    estimated = int(round(sample_count * target_seconds / safe_seconds * scale_safety))
    return max(min_sample_count, min(max_sample_count, estimated))


def build_output_config(config: Phase16BuildConfig) -> dict[str, object]:
    """Produce an updated Phase 13 config leveraging past benchmark timing.

    Args:
        config: Phase 16 build parameters.

    Returns:
        Writable Phase 13 config dictionary payload.
    """

    summary = load_json(config.source_summary_path)
    source_config = dict(summary["config"])
    scene_method_compare_sample_counts: dict[str, int] = {}

    for record in summary["records"]:
        method = str(record["method"])
        if method == "path_reference" or not bool(record["ok"]):
            continue
        scene_id = str(record["scene_id"])
        estimated = estimate_sample_count(
            sample_count=int(record["sample_count"]),
            seconds=float(record["seconds"]),
            target_seconds=config.target_seconds,
            scale_safety=config.scale_safety,
            min_sample_count=config.min_sample_count,
            max_sample_count=config.max_sample_count,
        )
        scene_method_compare_sample_counts[f"{scene_id}|{method}"] = estimated

    source_config["scene_method_compare_sample_counts"] = (
        scene_method_compare_sample_counts
    )
    source_config["output_dir"] = config.output_dir
    source_config["time_budget_seconds"] = config.target_seconds
    if config.reference_sample_count is not None:
        source_config["reference_sample_count"] = config.reference_sample_count
    return source_config


def main() -> int:
    """Emit equal-time configs for downstream benchmarking."""

    configure_logging()
    args = parse_args()
    config = load_config(args.config)
    output_config = build_output_config(config)
    config.output_config_path.parent.mkdir(parents=True, exist_ok=True)
    config.output_config_path.write_text(
        json.dumps(output_config, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    LOGGER.info("equal-time config written: %s", config.output_config_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
