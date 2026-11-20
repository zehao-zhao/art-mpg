#!/usr/bin/env python3
"""Summarize the Lamp/Slabs development equal-render-time benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = WORKSPACE_ROOT.parent
SCENES = ("paper_lamp", "paper_slabs")
SEEDS = (104729, 130363, 155921)
DIFFUSION_METHOD = "ordered_diffusion_mlp_h72_k4"
SIMPLE_METHOD = "simple_mlp_k4"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--initial-root",
        type=Path,
        default=WORKSPACE_ROOT / "results/equal_render_time_dev_macos_20260901/initial",
    )
    parser.add_argument(
        "--calibrated-root",
        type=Path,
        default=WORKSPACE_ROOT / "results/equal_render_time_dev_macos_20260901/calibrated",
    )
    parser.add_argument(
        "--calibration-decision",
        type=Path,
        default=WORKSPACE_ROOT
        / "results/equal_render_time_dev_macos_20260901/calibration_decision.json",
    )
    parser.add_argument(
        "--equal-spp-report",
        type=Path,
        default=WORKSPACE_ROOT
        / "results/model_specific_kappa_sweep_macos_20260901/model_specific_kappa_summary.json",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=WORKSPACE_ROOT
        / "results/equal_render_time_dev_macos_20260901/equal_time_summary.json",
    )
    parser.add_argument(
        "--output-markdown",
        type=Path,
        default=WORKSPACE_ROOT
        / "results/equal_render_time_dev_macos_20260901/equal_time_summary.md",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_summaries(root: Path, role: str) -> list[tuple[Path, dict[str, Any]]]:
    paths = sorted(root.glob(f"{role}_seed*/run_*/benchmark_summary.json"))
    return [(path, json.loads(path.read_text(encoding="utf-8"))) for path in paths]


def record_map(
    summaries: list[tuple[Path, dict[str, Any]]],
    *,
    role: str,
) -> dict[tuple[str, int], dict[str, Any]]:
    method = DIFFUSION_METHOD if role == "diffusion" else SIMPLE_METHOD
    result: dict[tuple[str, int], dict[str, Any]] = {}
    for path, payload in summaries:
        config = payload["config"]
        seed = int(config["seed"])
        expected_alpha = 0.0 if role == "diffusion" else 0.5
        expected_kappa = 1.0 if role == "diffusion" else 0.25
        checks = {
            "reference_sample_count": 1024,
            "reference_seed": 99173,
            "bounce_alpha": expected_alpha,
            "learned_direction_kappa_scale": expected_kappa,
            "learned_direction_flip_to_receiver_hemisphere": False,
            "learned_proposal_inv_prob_clamp": 512.0,
            "learned_proposal_contribution_clamp": 32.0,
            "learned_proposal_max_trials": 2048,
        }
        for field, expected in checks.items():
            if config.get(field) != expected:
                raise ValueError(f"{path}: {field}={config.get(field)!r}, expected {expected!r}")
        references = {
            row["scene_id"]: row
            for row in payload["records"]
            if row["method"] == "path_reference"
        }
        for row in payload["records"]:
            if row["method"] != method:
                continue
            if not row["ok"] or not row["loaded_runtime_model"] or row["mse"] is None:
                raise ValueError(f"invalid {role} record in {path}")
            scene = str(row["scene_id"])
            key = (scene, seed)
            if key in result:
                raise ValueError(f"duplicate {role} row {key}")
            reference = references[scene]
            result[key] = {
                "scene": scene,
                "seed": seed,
                "method": method,
                "mse": float(row["mse"]),
                "seconds": float(row["seconds"]),
                "sample_count": int(row["sample_count"]),
                "runtime_model_path": row["runtime_model_path"],
                "active_config_sha256": row["active_config_sha256"],
                "reference_config_sha256": reference["active_config_sha256"],
                "reference_output_path": reference["output_path"],
                "summary": str(path.resolve()),
            }
    return result


def geometric_mean(values: list[float]) -> float:
    return math.exp(statistics.fmean(math.log(value) for value in values))


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "pair_count": len(rows),
        "geometric_mse_ratio": geometric_mean([row["diffusion_to_simple_mse_ratio"] for row in rows]),
        "diffusion_wins": sum(row["diffusion_to_simple_mse_ratio"] < 1.0 for row in rows),
        "win_rate": statistics.fmean(row["diffusion_to_simple_mse_ratio"] < 1.0 for row in rows),
        "mean_diffusion_seconds": statistics.fmean(row["diffusion_seconds"] for row in rows),
        "mean_simple_seconds": statistics.fmean(row["simple_seconds"] for row in rows),
        "geometric_runtime_ratio": geometric_mean([row["diffusion_to_simple_runtime_ratio"] for row in rows]),
        "mean_absolute_time_deviation_fraction": statistics.fmean(row["absolute_time_deviation_fraction"] for row in rows),
        "max_absolute_time_deviation_fraction": max(row["absolute_time_deviation_fraction"] for row in rows),
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Development equal-render-time benchmark",
        "",
        "Current ordered MLP uses 64 spp, `alpha=0`, `kappa_scale=1`, and no "
        "hemisphere flip. Simple K4 uses `alpha=.5`, `kappa_scale=.25`, no flip, "
        "and scene-specific SPP selected from timing only. Both use the practical "
        "512/32/2048 clamps and a 1024-spp fixed reference.",
        "",
        f"- Calibration decision: `{report['calibration']['decision_path']}`",
        f"- Quality metrics read during calibration: `{str(report['calibration']['quality_metrics_read']).lower()}`",
        "- Seeds: 104729, 130363, 155921",
        "",
        "| Scene | Seed | Diff spp | Simple spp | Diff MSE | Simple MSE | D/S MSE | Diff s | Simple s | D/S time | time mismatch |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["pairs"]:
        lines.append(
            "| {scene} | {seed} | {diff_spp} | {simple_spp} | {diff_mse:.8f} | "
            "{simple_mse:.8f} | {mse_ratio:.6f} | {diff_s:.3f} | {simple_s:.3f} | "
            "{time_ratio:.4f} | {time_dev:.2f}% |".format(
                scene=row["scene"],
                seed=row["seed"],
                diff_spp=row["diffusion_spp"],
                simple_spp=row["simple_spp"],
                diff_mse=row["diffusion_mse"],
                simple_mse=row["simple_mse"],
                mse_ratio=row["diffusion_to_simple_mse_ratio"],
                diff_s=row["diffusion_seconds"],
                simple_s=row["simple_seconds"],
                time_ratio=row["diffusion_to_simple_runtime_ratio"],
                time_dev=100.0 * row["absolute_time_deviation_fraction"],
            )
        )
    lines.extend(["", "## Aggregates", ""])
    for scene in (*SCENES, "pooled"):
        aggregate_row = report["aggregates"][scene]
        lines.append(
            "- `{scene}`: geometric D/S MSE `{mse:.6f}`, wins "
            "`{wins}/{pairs}`; mean seconds D/S `{ds:.3f}/{ss:.3f}`, "
            "geometric runtime ratio `{runtime:.4f}`, max time mismatch "
            "`{mismatch:.2f}%`.".format(
                scene=scene,
                mse=aggregate_row["geometric_mse_ratio"],
                wins=aggregate_row["diffusion_wins"],
                pairs=aggregate_row["pair_count"],
                ds=aggregate_row["mean_diffusion_seconds"],
                ss=aggregate_row["mean_simple_seconds"],
                runtime=aggregate_row["geometric_runtime_ratio"],
                mismatch=100.0 * aggregate_row["max_absolute_time_deviation_fraction"],
            )
        )
    comparison = report["comparison_to_equal_spp"]
    lines.extend(
        [
            "",
            "## Comparison with equal SPP",
            "",
            f"- Equal-SPP pooled geometric D/S MSE: `{comparison['equal_spp_geometric_mse_ratio']:.6f}`.",
            f"- Equal-time pooled geometric D/S MSE: `{comparison['equal_time_geometric_mse_ratio']:.6f}`.",
            f"- Ratio change (equal-time / equal-SPP): `{comparison['ratio_of_ratios']:.6f}`.",
            "",
            report["conclusion"],
            "",
            "These are development-scene results only. Timing equality addresses one "
            "compute-budget comparison, but it is not final-scene evidence.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    calibration_path = args.calibration_decision.resolve()
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    if calibration.get("quality_metrics_read") is not False:
        raise ValueError("calibration decision is not certified timing-only")

    diffusion = record_map(load_summaries(args.initial_root.resolve(), "diffusion"), role="diffusion")
    simple_initial = record_map(load_summaries(args.initial_root.resolve(), "simple"), role="simple")
    calibrated_summaries = load_summaries(args.calibrated_root.resolve(), "simple")
    simple_calibrated = (
        record_map(calibrated_summaries, role="simple") if calibrated_summaries else {}
    )
    expected_keys = {(scene, seed) for scene in SCENES for seed in SEEDS}
    if set(diffusion) != expected_keys or set(simple_initial) != expected_keys:
        raise ValueError("initial run set is incomplete")

    pairs: list[dict[str, Any]] = []
    for scene in SCENES:
        decision = calibration["decisions"][scene]
        use_calibrated = bool(decision["needs_one_blind_calibration"])
        for seed in SEEDS:
            key = (scene, seed)
            diff = diffusion[key]
            simple = simple_calibrated[key] if use_calibrated else simple_initial[key]
            if diff["reference_config_sha256"] != simple["reference_config_sha256"]:
                raise ValueError(f"reference config differs for {key}")
            if diff["sample_count"] != 64:
                raise ValueError(f"diffusion SPP is not 64 for {key}")
            expected_simple_spp = int(decision["calibrated_simple_spp"])
            if simple["sample_count"] != expected_simple_spp:
                raise ValueError(f"simple SPP mismatch for {key}")
            time_ratio = diff["seconds"] / simple["seconds"]
            pairs.append(
                {
                    "scene": scene,
                    "seed": seed,
                    "diffusion_spp": diff["sample_count"],
                    "simple_spp": simple["sample_count"],
                    "diffusion_mse": diff["mse"],
                    "simple_mse": simple["mse"],
                    "diffusion_to_simple_mse_ratio": diff["mse"] / simple["mse"],
                    "diffusion_seconds": diff["seconds"],
                    "simple_seconds": simple["seconds"],
                    "diffusion_to_simple_runtime_ratio": time_ratio,
                    "absolute_time_deviation_fraction": abs(time_ratio - 1.0),
                    "diffusion_summary": diff["summary"],
                    "simple_summary": simple["summary"],
                    "reference_config_sha256": diff["reference_config_sha256"],
                }
            )

    aggregates = {
        scene: aggregate([row for row in pairs if row["scene"] == scene])
        for scene in SCENES
    }
    aggregates["pooled"] = aggregate(pairs)
    equal_spp_path = args.equal_spp_report.resolve()
    equal_spp = json.loads(equal_spp_path.read_text(encoding="utf-8"))
    equal_spp_ratio = float(
        equal_spp["tuned_comparison_pooled"]["mse"]["geometric_mean_ratio"]
    )
    equal_time_ratio = aggregates["pooled"]["geometric_mse_ratio"]
    conclusion = (
        "At approximately equal render time, diffusion has lower pooled geometric "
        "MSE than simple K4."
        if equal_time_ratio < 1.0
        else "At approximately equal render time, diffusion does not beat simple K4 "
        "on pooled geometric MSE."
    )
    report = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "development_equal_render_time",
        "protocol": {
            "scenes": list(SCENES),
            "seeds": list(SEEDS),
            "reference_seed": 99173,
            "reference_spp": 1024,
            "diffusion": {"spp": 64, "bounce_alpha": 0.0, "kappa_scale": 1.0},
            "simple": {
                "bounce_alpha": 0.5,
                "kappa_scale": 0.25,
                "scene_spp": {
                    scene: int(calibration["decisions"][scene]["calibrated_simple_spp"])
                    for scene in SCENES
                },
            },
            "hemisphere_flip": False,
            "practical_clamps": {
                "inverse_probability": 512.0,
                "contribution": 32.0,
                "max_trials": 2048,
            },
        },
        "calibration": {
            "decision_path": str(calibration_path),
            "decision_sha256": sha256_file(calibration_path),
            "quality_metrics_read": calibration["quality_metrics_read"],
            "policy": calibration["policy"],
            "decisions": calibration["decisions"],
        },
        "pairs": pairs,
        "aggregates": aggregates,
        "comparison_to_equal_spp": {
            "equal_spp_report": str(equal_spp_path),
            "equal_spp_geometric_mse_ratio": equal_spp_ratio,
            "equal_time_geometric_mse_ratio": equal_time_ratio,
            "ratio_of_ratios": equal_time_ratio / equal_spp_ratio,
        },
        "conclusion": conclusion,
    }
    args.output_json.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.output_json.resolve().write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    args.output_markdown.resolve().write_text(render_markdown(report), encoding="utf-8")
    print(args.output_json.resolve())
    print(args.output_markdown.resolve())


if __name__ == "__main__":
    main()
