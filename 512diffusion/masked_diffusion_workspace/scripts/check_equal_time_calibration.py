#!/usr/bin/env python3
"""Make the one allowed equal-time SPP calibration using timing only.

This script intentionally never reads an image-quality field.  Its JSON output
is the immutable timing-only decision record used by the final summarizer.
"""

from __future__ import annotations

import argparse
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
SCENES = ("paper_lamp", "paper_slabs")
SEEDS = (104729, 130363, 155921)
METHODS = {
    "diffusion": "ordered_diffusion_mlp_h72_k4",
    "simple": "simple_mlp_k4",
}
EXPECTED_SIMPLE_SPP = {"paper_lamp": 165, "paper_slabs": 184}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--initial-root",
        type=Path,
        default=WORKSPACE_ROOT / "results/equal_render_time_dev_macos_20260901/initial",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=WORKSPACE_ROOT
        / "results/equal_render_time_dev_macos_20260901/calibration_decision.json",
    )
    return parser.parse_args()


def summaries(root: Path, role: str) -> list[tuple[Path, dict[str, Any]]]:
    paths = sorted(root.glob(f"{role}_seed*/run_*/benchmark_summary.json"))
    if len(paths) != len(SEEDS):
        raise ValueError(f"expected {len(SEEDS)} {role} summaries, got {len(paths)}")
    return [(path, json.loads(path.read_text(encoding="utf-8"))) for path in paths]


def timing_rows(
    items: list[tuple[Path, dict[str, Any]]], role: str
) -> list[dict[str, Any]]:
    method = METHODS[role]
    rows: list[dict[str, Any]] = []
    for path, payload in items:
        config = payload["config"]
        seed = int(config["seed"])
        expected_knobs = (
            (0.0, 1.0, 64)
            if role == "diffusion"
            else (0.5, 0.25, None)
        )
        if bool(config["learned_direction_flip_to_receiver_hemisphere"]):
            raise ValueError(f"hemisphere flip enabled in {path}")
        if float(config["bounce_alpha"]) != expected_knobs[0]:
            raise ValueError(f"wrong bounce alpha in {path}")
        if float(config["learned_direction_kappa_scale"]) != expected_knobs[1]:
            raise ValueError(f"wrong kappa scale in {path}")
        if int(config["reference_sample_count"]) != 1024:
            raise ValueError(f"wrong reference spp in {path}")
        for record in payload["records"]:
            if record["method"] != method:
                continue
            scene = str(record["scene_id"])
            sample_count = int(record["sample_count"])
            expected_spp = (
                64 if role == "diffusion" else EXPECTED_SIMPLE_SPP[scene]
            )
            if sample_count != expected_spp:
                raise ValueError(
                    f"wrong {role} spp for {scene}: {sample_count} != {expected_spp}"
                )
            if not bool(record["ok"]) or not bool(record["loaded_runtime_model"]):
                raise ValueError(f"invalid runtime record in {path}: {scene}/{method}")
            rows.append(
                {
                    "role": role,
                    "scene": scene,
                    "seed": seed,
                    "sample_count": sample_count,
                    "seconds": float(record["seconds"]),
                    "summary": str(path.resolve()),
                }
            )
    return rows


def main() -> None:
    args = parse_args()
    initial_root = args.initial_root.resolve()
    rows = timing_rows(summaries(initial_root, "diffusion"), "diffusion")
    rows.extend(timing_rows(summaries(initial_root, "simple"), "simple"))
    if {row["seed"] for row in rows} != set(SEEDS):
        raise ValueError("unexpected seed set")

    decisions: dict[str, dict[str, Any]] = {}
    for scene in SCENES:
        diffusion_seconds = [
            row["seconds"]
            for row in rows
            if row["scene"] == scene and row["role"] == "diffusion"
        ]
        simple_seconds = [
            row["seconds"]
            for row in rows
            if row["scene"] == scene and row["role"] == "simple"
        ]
        if len(diffusion_seconds) != len(SEEDS) or len(simple_seconds) != len(SEEDS):
            raise ValueError(f"incomplete timing rows for {scene}")
        diffusion_mean = statistics.fmean(diffusion_seconds)
        simple_mean = statistics.fmean(simple_seconds)
        ratio = simple_mean / diffusion_mean
        initial_spp = EXPECTED_SIMPLE_SPP[scene]
        needs_calibration = abs(ratio - 1.0) > 0.10
        proposed_spp = max(1, round(initial_spp * diffusion_mean / simple_mean))
        decisions[scene] = {
            "diffusion_mean_seconds": diffusion_mean,
            "simple_mean_seconds": simple_mean,
            "simple_to_diffusion_time_ratio": ratio,
            "absolute_time_deviation_fraction": abs(ratio - 1.0),
            "threshold_fraction": 0.10,
            "initial_simple_spp": initial_spp,
            "needs_one_blind_calibration": needs_calibration,
            "calibrated_simple_spp": proposed_spp if needs_calibration else initial_spp,
        }

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "policy": "one timing-only calibration iff absolute mean-time deviation exceeds 10%",
        "quality_metrics_read": False,
        "rows": rows,
        "decisions": decisions,
    }
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
