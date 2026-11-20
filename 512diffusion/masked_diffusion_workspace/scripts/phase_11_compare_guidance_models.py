from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = WORKSPACE_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from masked_diffusion_mpg.comparison import (  # noqa: E402
    load_comparison_config,
    run_model_comparison,
)


LOGGER = logging.getLogger("phase_11_compare_guidance_models")


def configure_logging() -> None:
    """Initialize logging helpers."""

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description="Run Phase 11 model comparison")
    parser.add_argument(
        "--config",
        type=Path,
        default=WORKSPACE_ROOT / "configs" / "phase_11_compare_models.json",
    )
    return parser.parse_args()


def main() -> int:
    """Compare masked diffusion vs simple NN on offline replay dumps."""

    configure_logging()
    args = parse_args()
    config = load_comparison_config(args.config)
    artifacts = run_model_comparison(workspace_root=WORKSPACE_ROOT, config=config)
    LOGGER.info("run_dir: %s", artifacts.run_dir)
    LOGGER.info("comparison_json: %s", artifacts.comparison_json)
    LOGGER.info("comparison_md: %s", artifacts.comparison_md)
    LOGGER.info("simple_checkpoint: %s", artifacts.simple_checkpoint)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
