#!/usr/bin/env python3
"""Generate, execute, and resume the NPM-adapted training matrix.

The driver keeps every scene/seed/step-budget in an isolated directory and
writes a machine-readable receipt.  It is intentionally independent from the
historical ``simple_mlp`` coordinator so old artifacts cannot be relabelled as
the arXiv:2504.04315-adapted baseline.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = WORKSPACE_ROOT / "scripts" / "train_npm_adapted.py"

REPLAY_PATHS = {
    "lamp": "results/phase_09_export/replay/paper_lamp_mpg.jsonl",
    "living": "results/phase_09_export/replay/paper_living_mpg.jsonl",
    "pillars": "results/phase_09_export/replay/paper_pillars_mpg.jsonl",
    "plane": "results/phase_09_export/replay/paper_plane_mpg.jsonl",
    "pool": "results/phase_09_export/replay/paper_pool_mpg.jsonl",
    "ring_and_slab": "results/phase_09_export/replay/paper_ring_and_slab_mpg.jsonl",
    "slabs": "results/phase_09_export/replay/paper_slabs_mpg.jsonl",
    "sphere": "results/phase_09_export/replay/paper_sphere_mpg.jsonl",
    "stone": "results/phase_09_export/replay/paper_stone_mpg.jsonl",
    "water": "results/phase_09_export/replay/paper_water_mpg.jsonl",
    "slab_x1": (
        "scene_intake_20260904/reports/phase09_export_20260904_v1/"
        "replay/intake_slab_x1.jsonl"
    ),
    "slab_x3": (
        "scene_intake_20260904/reports/phase09_export_20260904_v1/"
        "replay/intake_slab_x3.jsonl"
    ),
    "pillar_x3": (
        "scene_intake_20260904/reports/phase09_export_20260904_v1/"
        "replay/intake_pillar_x3.jsonl"
    ),
    "cornell_glass": (
        "scene_intake_20260904/reports/phase09_export_20260904_v1/"
        "replay/intake_cornell_glass.jsonl"
    ),
}

MODEL = {
    "grid_resolutions": [8, 11, 16, 22, 31, 44, 61, 86],
    "features_per_level": 4,
    "sh_degree": 2,
    "hidden_dim": 64,
    "hidden_layers": 3,
    "vmf_components": 8,
    "min_kappa": 0.0001,
    "max_kappa": 128.0,
    "descriptor_policy_id": "geometric_length_trace_inferred_type_v1",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def latest_summary(output_dir: Path) -> Path | None:
    candidates = list(output_dir.glob("run_*/npm_adapted_training_summary.json"))
    return max(candidates, key=lambda item: item.stat().st_mtime) if candidates else None


def build_entry(
    *, output_root: Path, scene: str, seed: int, max_steps: int
) -> dict[str, Any]:
    replay_relative = REPLAY_PATHS[scene]
    replay_path = WORKSPACE_ROOT / replay_relative
    if not replay_path.is_file():
        raise FileNotFoundError(f"missing replay for {scene}: {replay_path}")
    identity = f"{scene}__seed{seed}__steps{max_steps}"
    output_dir = output_root / "runs" / identity
    try:
        output_relative = output_dir.relative_to(WORKSPACE_ROOT)
    except ValueError as exc:
        raise ValueError("output-root must be under the workspace") from exc
    config = {
        "replay_paths": [replay_relative],
        "output_dir": str(output_relative),
        "seed": seed,
        "batch_size": 262144,
        "max_steps": max_steps,
        "learning_rate": 0.005,
        "ema_decay": 0.99,
        "val_fraction": 0.25,
        "num_workers": 0,
        "device": "cpu",
        "position_bounds_padding": 0.01,
        "model": MODEL,
    }
    config_path = output_root / "configs" / f"{identity}.json"
    write_json(config_path, config)
    return {
        "id": identity,
        "scene": scene,
        "seed": seed,
        "max_steps": max_steps,
        "replay_path": replay_relative,
        "replay_sha256": sha256(replay_path),
        "config_path": str(config_path),
        "config_sha256": sha256(config_path),
        "output_dir": str(output_dir),
        "status": "emitted",
    }


def execute_entry(
    entry: dict[str, Any], *, python: str, rerun: bool, threads_per_job: int
) -> dict[str, Any]:
    output_dir = Path(entry["output_dir"])
    existing = latest_summary(output_dir)
    if existing is not None and not rerun:
        result = dict(entry)
        result.update(
            {
                "status": "reused",
                "return_code": 0,
                "summary_path": str(existing),
                "summary_sha256": sha256(existing),
            }
        )
        return result
    env = dict(os.environ)
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        env[name] = str(threads_per_job)
    completed = subprocess.run(
        [python, str(TRAIN_SCRIPT), "--config", entry["config_path"]],
        cwd=WORKSPACE_ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    summary = latest_summary(output_dir)
    result = dict(entry)
    result.update(
        {
            "status": "complete" if completed.returncode == 0 and summary else "failed",
            "return_code": completed.returncode,
            "summary_path": None if summary is None else str(summary),
            "summary_sha256": None if summary is None else sha256(summary),
            "log_tail": completed.stdout[-8000:],
        }
    )
    return result


def parse_csv(value: str, converter: type) -> list[Any]:
    return [converter(token.strip()) for token in value.split(",") if token.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenes", default=",".join(REPLAY_PATHS))
    parser.add_argument("--seeds", default="104729,130363,155921")
    parser.add_argument("--steps", default="1000")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=WORKSPACE_ROOT / "results" / "npm_adapted_training_matrix_20260904_v1",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--threads-per-job", type=int, default=2)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--rerun", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    scenes = parse_csv(args.scenes, str)
    unknown = sorted(set(scenes) - set(REPLAY_PATHS))
    if unknown:
        raise SystemExit(f"unknown scenes: {unknown}")
    seeds = parse_csv(args.seeds, int)
    steps = parse_csv(args.steps, int)
    if not seeds or not steps or any(value <= 0 for value in steps):
        raise SystemExit("seeds and positive step budgets are required")
    output_root = args.output_root.resolve()
    entries = [
        build_entry(
            output_root=output_root,
            scene=scene,
            seed=seed,
            max_steps=max_steps,
        )
        for scene in scenes
        for seed in seeds
        for max_steps in steps
    ]
    if args.execute:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [
                executor.submit(
                    execute_entry,
                    entry,
                    python=args.python,
                    rerun=args.rerun,
                    threads_per_job=args.threads_per_job,
                )
                for entry in entries
            ]
            entries = []
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                entries.append(result)
                print(f"{result['id']}: {result['status']}", flush=True)
        entries.sort(key=lambda item: item["id"])
    manifest = {
        "schema_version": 1,
        "protocol_id": "npm_adapted_training_matrix_20260904_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "paper": "arXiv:2504.04315",
        "model_profile": MODEL,
        "executed": bool(args.execute),
        "workers": args.workers,
        "threads_per_job": args.threads_per_job,
        "entries": entries,
    }
    write_json(output_root / "run_manifest.json", manifest)
    failed = [entry for entry in entries if entry["status"] == "failed"]
    print(output_root / "run_manifest.json")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
