from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = WORKSPACE_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from masked_diffusion_mpg.npm_adapted import (  # noqa: E402
    load_npm_adapted_training_config,
    run_npm_adapted_training,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train the isolated NPM-adapted seed-direction baseline"
    )
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    config = load_npm_adapted_training_config(args.config)
    artifacts = run_npm_adapted_training(
        workspace_root=WORKSPACE_ROOT,
        config=config,
    )
    logging.info("run_dir: %s", artifacts.run_dir)
    logging.info("checkpoint: %s", artifacts.checkpoint_path)
    logging.info("summary: %s", artifacts.summary_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
