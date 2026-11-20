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
    export_npm_adapted_python_bundle,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export EMA weights for Python NPM-adapted inference"
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    artifacts = export_npm_adapted_python_bundle(
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
    )
    logging.info("weights: %s", artifacts.weights_path)
    logging.info("manifest: %s", artifacts.manifest_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
