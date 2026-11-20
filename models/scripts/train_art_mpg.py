from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import replace
from pathlib import Path


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = WORKSPACE_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from masked_diffusion_mpg.training import (  # noqa: E402
    load_training_config,
    run_training,
)


LOGGER = logging.getLogger("train_art_mpg")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description="Train masked diffusion MPG baseline")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max-records", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    """Launch ART-MPG masked diffusion training loops."""

    args = parse_args()
    config = load_training_config(args.config)
    if args.epochs is not None:
        config = replace(config, epochs=args.epochs)
    if args.max_records is not None:
        config = replace(config, max_records=args.max_records)

    artifacts = run_training(
        workspace_root=WORKSPACE_ROOT,
        config=config,
    )
    LOGGER.info("run_dir: %s", artifacts.run_dir)
    LOGGER.info("best_checkpoint: %s", artifacts.best_checkpoint)
    LOGGER.info("metrics_json: %s", artifacts.metrics_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
