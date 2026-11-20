from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = WORKSPACE_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from masked_diffusion_mpg.runtime_export import (  # noqa: E402
    export_runtime_model,
    load_runtime_export_config,
)


LOGGER = logging.getLogger("phase_12_export_runtime_model")


def configure_logging() -> None:
    """Initialize logging helpers."""

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description="Run Phase 12 runtime export")
    parser.add_argument(
        "--config",
        type=Path,
        default=WORKSPACE_ROOT / "configs" / "phase_12_runtime_export.json",
    )
    return parser.parse_args()


def main() -> int:
    """Export Torch weights packaged for Mitsuba runtime consumption."""

    configure_logging()
    args = parse_args()
    config = load_runtime_export_config(args.config)
    artifacts = export_runtime_model(workspace_root=WORKSPACE_ROOT, config=config)
    LOGGER.info("runtime_path: %s", artifacts.runtime_path)
    LOGGER.info("summary_json: %s", artifacts.summary_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
