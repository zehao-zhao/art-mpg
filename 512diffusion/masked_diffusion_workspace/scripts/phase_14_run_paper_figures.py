from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path


DEFAULT_FIGURES = (
    "fig07_plane.py",
    "fig08_living.py",
    "fig09_lamp.py",
    "fig09_slabs.py",
    "fig14_lamp.py",
    "fig14_stone.py",
)


@dataclass(slots=True)
class FigureRunRecord:
    """Stdout/stderr bookkeeping for one paper figure script."""

    script: str
    ok: bool
    return_code: int
    seconds: float
    stdout_log: str
    stderr_log: str
    result_dir: str


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(
        description="Run reproducibility wrappers for canonical paper figures.",
    )
    parser.add_argument(
        "--figures",
        nargs="*",
        default=list(DEFAULT_FIGURES),
        help="Subset of experiment/*.py wrappers; omit to run the default bundle.",
    )
    return parser.parse_args()


def get_repo_paths() -> tuple[Path, Path, Path]:
    """Return workspace root, experiments directory, and log root."""
    workspace_root = Path(__file__).resolve().parents[1]
    repo_root = workspace_root.parent
    experiments_dir = repo_root / "experiments"
    results_root = workspace_root / "results" / "phase_14_paper_figures"
    return workspace_root, experiments_dir, results_root


def run_figure(
    script_name: str, experiments_dir: Path, run_dir: Path
) -> FigureRunRecord:
    """Execute one figure harness and tee logs to disk."""
    script_path = experiments_dir / script_name
    if not script_path.exists():
        raise FileNotFoundError(f"Figure script missing: {script_path}")
    logs_dir = run_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    stdout_log = logs_dir / f"{script_path.stem}.stdout.txt"
    stderr_log = logs_dir / f"{script_path.stem}.stderr.txt"
    t0 = time.perf_counter()
    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=experiments_dir,
        capture_output=True,
        text=True,
        check=False,
    )
    t1 = time.perf_counter()
    stdout_log.write_text(result.stdout, encoding="utf-8")
    stderr_log.write_text(result.stderr, encoding="utf-8")
    return FigureRunRecord(
        script=script_name,
        ok=result.returncode == 0,
        return_code=result.returncode,
        seconds=t1 - t0,
        stdout_log=str(stdout_log),
        stderr_log=str(stderr_log),
        result_dir=str(experiments_dir / "results" / script_path.stem),
    )


def write_summary(run_dir: Path, records: list[FigureRunRecord]) -> None:
    """Persist JSON/Markdown rollups describing the orchestration."""
    summary = {
        "run_dir": str(run_dir),
        "total": len(records),
        "passed": sum(1 for record in records if record.ok),
        "failed": sum(1 for record in records if not record.ok),
        "records": [asdict(record) for record in records],
    }
    summary_json = run_dir / "figure_summary.json"
    summary_json.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    summary_md = run_dir / "figure_summary.md"
    lines = [
        "# Paper Figure Summary",
        "",
        f"- total: {summary['total']}",
        f"- passed: {summary['passed']}",
        f"- failed: {summary['failed']}",
        "",
        "| script | ok | seconds | result_dir |",
        "| --- | --- | ---: | --- |",
    ]
    for record in records:
        lines.append(
            f"| {record.script} | {record.ok} | {record.seconds:.2f} | {record.result_dir} |"
        )
    summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    """CLI dispatcher for batch figure regeneration."""
    args = parse_args()
    _, experiments_dir, results_root = get_repo_paths()
    run_dir = results_root / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    records: list[FigureRunRecord] = []
    for script_name in args.figures:
        print(f"INFO phase_14 run figure: {script_name}")
        records.append(run_figure(script_name, experiments_dir, run_dir))
    write_summary(run_dir, records)
    failed = [record.script for record in records if not record.ok]
    if failed:
        print(f"ERROR phase_14 figure failed: {', '.join(failed)}")
        return 1
    print(f"INFO phase_14 figure finished: passed={len(records)} failed=0")
    print(f"INFO summary_json: {run_dir / 'figure_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
