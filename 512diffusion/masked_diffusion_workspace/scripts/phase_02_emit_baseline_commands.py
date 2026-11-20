from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path


LOGGER = logging.getLogger("phase_02_emit_baseline_commands")


@dataclass(slots=True)
class RenderJob:
    """Phase-02 shim mirroring serialized render manifests."""

    job_id: str
    group: str
    method: str
    variant: str
    scene_path: str
    expected_integrator: str


@dataclass(slots=True)
class TrainingJob:
    """Phase-02 shim mirroring serialized training manifests."""

    job_id: str
    group: str
    script_path: str
    args: list[str]


def configure_logging() -> None:
    """Initialize logging."""

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def quote_bash(value: str) -> str:
    """Escape bash interpolations minimally."""

    escaped = value.replace('"', '\\"')
    return f'"{escaped}"'


def render_command(job: RenderJob) -> str:
    """Generate bash invoking Mitsuba render for legacy verification."""

    scene_expr = f"$REPO_ROOT/{job.scene_path}"
    out_expr = f"$RESULT_DIR/{job.job_id}.exr"
    return (
        f'./dist/mitsuba -m {job.variant} -t "$(nproc)" '
        f"-o {quote_bash(out_expr)} {quote_bash(scene_expr)}"
    )


def training_command(job: TrainingJob) -> str:
    """Generate bash invoking ancillary training tooling."""

    args = " ".join(quote_bash(arg) for arg in job.args)
    script_expr = f"$REPO_ROOT/{job.script_path}"
    return f"python {quote_bash(script_expr)} {args}".strip()


def main() -> None:
    """Materialize Markdown command cheat sheet sourced from manifests."""

    configure_logging()

    workspace_root = Path(__file__).resolve().parents[1]
    project_root = workspace_root.parent
    config_path = workspace_root / "configs" / "baseline_manifest.json"
    output_path = workspace_root / "results" / "phase_02_baseline_commands.md"

    manifest = json.loads(config_path.read_text(encoding="utf-8"))

    render_jobs: list[RenderJob] = [
        RenderJob(
            job_id=item["id"],
            group=item["group"],
            method=item["method"],
            variant=item["variant"],
            scene_path=item["scene_path"],
            expected_integrator=item["expected_integrator"],
        )
        for item in manifest["render_jobs"]
    ]
    training_jobs: list[TrainingJob] = [
        TrainingJob(
            job_id=item["id"],
            group=item["group"],
            script_path=item["script_path"],
            args=list(item["args"]),
        )
        for item in manifest["training_jobs"]
    ]

    missing_paths: list[str] = []
    for job in render_jobs:
        if not (project_root / job.scene_path).exists():
            missing_paths.append(job.scene_path)
    for job in training_jobs:
        if not (project_root / job.script_path).exists():
            missing_paths.append(job.script_path)
    if missing_paths:
        raise FileNotFoundError(f"missing paths: {missing_paths}")

    lines: list[str] = [
        "# Phase 02 Baseline Commands",
        "",
        "Commands below feed the downstream baseline verification flow.",
        "",
        "## Environment variables",
        "",
        "```bash",
        'export REPO_ROOT="/path/to/discrete_diffusion/discrete_diffusion"',
        'export BUILD_DIR="$REPO_ROOT/mitsuba/build"',
        'export RESULT_DIR="$REPO_ROOT/masked_diffusion_workspace/results/baseline_runs"',
        'mkdir -p "$RESULT_DIR"',
        'cd "$BUILD_DIR"',
        'export MTS_PLUGIN_PATH="$BUILD_DIR/dist/plugins"',
        'export LD_LIBRARY_PATH="$BUILD_DIR/src/libcore:$BUILD_DIR/src/librender:$BUILD_DIR/dist:$BUILD_DIR/dist/plugins:${LD_LIBRARY_PATH:-}"',
        "```",
        "",
        "## Render Jobs",
        "",
    ]

    for job in render_jobs:
        lines.extend(
            [
                f"### {job.job_id}",
                "",
                f"- group: `{job.group}`",
                f"- method: `{job.method}`",
                f"- scene: `{job.scene_path}`",
                f"- expected_integrator: `{job.expected_integrator}`",
                "",
                "```bash",
                render_command(job),
                "```",
                "",
            ]
        )

    lines.extend(["## Training Jobs", ""])

    for job in training_jobs:
        lines.extend(
            [
                f"### {job.job_id}",
                "",
                f"- group: `{job.group}`",
                f"- script: `{job.script_path}`",
                "",
                "```bash",
                training_command(job),
                "```",
                "",
            ]
        )

    output_path.write_text("\n".join(lines), encoding="utf-8")
    LOGGER.info("baseline command file: %s", output_path)


if __name__ == "__main__":
    main()
