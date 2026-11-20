from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = WORKSPACE_ROOT.parent
LOGGER = logging.getLogger("phase_18_fixed_spp_sweep")


def configure_logging() -> None:
    """Initialize logging."""

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def main() -> int:
    """Delegate to ``paper_runtime/scripts/run_fixed_spp_sweep.py``."""

    configure_logging()
    script_path = PROJECT_ROOT / "paper_runtime" / "scripts" / "run_fixed_spp_sweep.py"
    command = [sys.executable, str(script_path), *sys.argv[1:]]
    LOGGER.info("run: %s", " ".join(command))
    subprocess.run(command, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
