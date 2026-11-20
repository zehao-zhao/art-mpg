#!/usr/bin/env python3
"""Robust, descriptive analysis for Phase-49 NN architecture sensitivity."""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import os
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


PACKAGE_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = PACKAGE_ROOT.parent
PROTOCOL_ID = "phase49_nn_architecture_sensitivity_20260924_v2_cco0"
RESULT_ROOT = WORKSPACE_ROOT / "results" / PROTOCOL_ID
DEFAULT_ROWS = RESULT_ROOT / "metrics" / "sensitivity_rows.csv"
DEFAULT_TRAINING = RESULT_ROOT / "metrics" / "training_rows.csv"
DEFAULT_OUTPUT = RESULT_ROOT / "analysis"
SCENES = (
    "pool", "living", "lamp", "plane", "slabs", "stone", "sphere",
    "water", "ring_and_slab", "pillars", "slab_x1", "slab_x3",
    "pillar_x3", "cornell_glass",
)
FULL_SPP = (12, 24, 36, 48, 60)
CONTROL = "h72_d3_control"
CONFIG_META = {
    "h36_d3": (36, 3, 20_211),
    "h72_d2": (72, 2, 44_943),
    CONTROL: (72, 3, 55_599),
    "h72_d4": (72, 4, 66_255),
    "h144_d3": (144, 3, 173_031),
}
LABELS = {
    "h36_d3": r"$h=36,d=3$",
    "h72_d2": r"$h=72,d=2$",
    CONTROL: r"$h=72,d=3$ (control)",
    "h72_d4": r"$h=72,d=4$",
    "h144_d3": r"$h=144,d=3$",
}


class AnalysisError(RuntimeError):
    """Raised when collected data violate the declared sensitivity contract."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def _atomic_json(path: Path, value: object) -> None:
    _atomic_bytes(path, _json_bytes(value))


def _float(value: object, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise AnalysisError(f"invalid float for {label}: {value!r}") from exc
    if not math.isfinite(result):
        raise AnalysisError(f"nonfinite value for {label}")
    return result


def _int(value: object, label: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise AnalysisError(f"invalid integer for {label}: {value!r}") from exc


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise AnalysisError(f"missing CSV: {path}")
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def normalize_metric_rows(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    required = {
        "config_id", "direction_conditioning_mode", "epochs", "hidden_dim", "depth",
        "vmf_components", "token_embedding_dim", "time_embedding_dim", "dropout",
        "parameter_count", "run_id", "scene",
        "sample_count", "model_seed", "render_seed", "mse", "render_seconds",
    }
    for index, source in enumerate(rows, start=2):
        if not required.issubset(source):
            raise AnalysisError(f"metric line {index} lacks {sorted(required - set(source))}")
        row = dict(source)
        for key in (
            "epochs", "hidden_dim", "depth", "vmf_components", "token_embedding_dim",
            "time_embedding_dim", "parameter_count",
            "sample_count", "model_seed", "render_seed",
        ):
            row[key] = _int(row[key], f"line {index} {key}")
        row["mse"] = _float(row["mse"], f"line {index} mse")
        row["render_seconds"] = _float(row["render_seconds"], f"line {index} render_seconds")
        row["dropout"] = _float(row["dropout"], f"line {index} dropout")
        if row["mse"] <= 0 or row["render_seconds"] < 0:
            raise AnalysisError(f"metric line {index} has nonpositive MSE or negative time")
        result.append(row)
    return result


def normalize_training_rows(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    required = {
        "config_id", "direction_conditioning_mode", "epochs", "hidden_dim", "depth",
        "vmf_components", "token_embedding_dim", "time_embedding_dim", "dropout",
        "parameter_count", "scene", "model_seed", "training_seconds",
        "epochs_completed", "final_val_loss", "best_val_loss",
    }
    for index, source in enumerate(rows, start=2):
        if not required.issubset(source):
            raise AnalysisError(f"training line {index} lacks {sorted(required - set(source))}")
        row = dict(source)
        for key in (
            "epochs", "hidden_dim", "depth", "vmf_components", "token_embedding_dim",
            "time_embedding_dim", "parameter_count",
            "model_seed", "epochs_completed",
        ):
            row[key] = _int(row[key], f"training line {index} {key}")
        for key in ("training_seconds", "final_val_loss", "best_val_loss"):
            row[key] = _float(row[key], f"training line {index} {key}")
        row["dropout"] = _float(row["dropout"], f"training line {index} dropout")
        result.append(row)
    return result


def _manifest_runs() -> dict[str, tuple[int, int]]:
    value = json.loads((PACKAGE_ROOT / "manifest.json").read_text(encoding="utf-8"))
    return {
        row["run_id"]: (int(row["model_seed"]), int(row["render_seed"]))
        for row in value["recorded_runs"]
    }


def validate_rows(rows: Sequence[Mapping[str, Any]], *, strict: bool) -> None:
    runs = _manifest_runs()
    seen: set[tuple[str, str, int, str]] = set()
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        config = str(row["config_id"])
        if config not in CONFIG_META:
            raise AnalysisError(f"unknown configuration: {config}")
        if row["scene"] not in SCENES or int(row["sample_count"]) not in FULL_SPP:
            raise AnalysisError("row is outside the 14-scene/five-SPP grid")
        identity = runs.get(str(row["run_id"]))
        if identity != (int(row["model_seed"]), int(row["render_seed"])):
            raise AnalysisError("row does not match the recorded D01--D10 identity")
        key = (config, str(row["scene"]), int(row["sample_count"]), str(row["run_id"]))
        if key in seen:
            raise AnalysisError(f"duplicate metric row: {key}")
        seen.add(key)
        counts[config] += 1
        expected = CONFIG_META[config]
        if (
            str(row["direction_conditioning_mode"]) != "full"
            or int(row["epochs"]) != 32
            or (int(row["hidden_dim"]), int(row["depth"]), int(row["parameter_count"])) != expected
            or int(row["vmf_components"]) != 4
            or int(row["token_embedding_dim"]) != 24
            or int(row["time_embedding_dim"]) != 16
            or not math.isclose(float(row["dropout"]), 0.1)
        ):
            raise AnalysisError(f"row architecture metadata changed: {key}")
    if strict:
        if len(rows) != 3500 or set(counts) != set(CONFIG_META):
            raise AnalysisError("strict analysis requires five fresh 700-row grids")
        if any(counts[config] != 700 for config in CONFIG_META):
            raise AnalysisError(f"strict configuration counts differ from 700: {dict(counts)}")


def validate_training_rows(rows: Sequence[Mapping[str, Any]], *, strict: bool) -> None:
    seen: set[tuple[str, str, int]] = set()
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        config = str(row["config_id"])
        if config not in CONFIG_META:
            raise AnalysisError(f"unknown training configuration: {config}")
        if row["scene"] not in SCENES or int(row["model_seed"]) not in {104729, 155921}:
            raise AnalysisError("training row is outside the recorded scene/model-seed grid")
        key = (config, str(row["scene"]), int(row["model_seed"]))
        if key in seen:
            raise AnalysisError(f"duplicate training row: {key}")
        seen.add(key)
        expected = CONFIG_META[config]
        if (
            str(row["direction_conditioning_mode"]) != "full"
            or int(row["epochs"]) != 32
            or int(row["epochs_completed"]) != 32
            or (int(row["hidden_dim"]), int(row["depth"]), int(row["parameter_count"])) != expected
            or int(row["vmf_components"]) != 4
            or int(row["token_embedding_dim"]) != 24
            or int(row["time_embedding_dim"]) != 16
            or not math.isclose(float(row["dropout"]), 0.1)
        ):
            raise AnalysisError(f"training architecture metadata changed: {key}")
        counts[config] += 1
    if strict:
        expected_keys = {
            (config, scene, seed)
            for config in CONFIG_META for scene in SCENES for seed in (104729, 155921)
        }
        if len(rows) != 140 or seen != expected_keys or any(counts[c] != 28 for c in CONFIG_META):
            raise AnalysisError(
                f"strict analysis requires the exact 14x2 training grid per architecture: {dict(counts)}"
            )


def _median_mse(rows: Sequence[Mapping[str, Any]], config: str, scene: str, spp: int) -> float:
    values = [
        float(row["mse"]) for row in rows
        if row["config_id"] == config and row["scene"] == scene and int(row["sample_count"]) == spp
    ]
    if len(values) != 10:
        raise AnalysisError(f"expected ten runs for {config}/{scene}/{spp}, found {len(values)}")
    return statistics.median(values)


def _gm(values: Sequence[float]) -> float:
    if not values or any(value <= 0 for value in values):
        raise AnalysisError("geometric mean requires positive values")
    return math.exp(statistics.fmean(math.log(value) for value in values))


def scene_median_ratios(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for config, (hidden, depth, count) in CONFIG_META.items():
        for scene in SCENES:
            for spp in FULL_SPP:
                control = _median_mse(rows, CONTROL, scene, spp)
                variant = _median_mse(rows, config, scene, spp)
                output.append({
                    "config_id": config, "hidden_dim": hidden, "depth": depth,
                    "parameter_count": count, "scene": scene, "sample_count": spp,
                    "variant_median_mse": variant, "control_median_mse": control,
                    "scene_median_mse_ratio": variant / control,
                })
    return output


def architecture_effects(ratios: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for config, (hidden, depth, count) in CONFIG_META.items():
        for spp in FULL_SPP:
            selected = [row for row in ratios if row["config_id"] == config and row["sample_count"] == spp]
            values = [float(row["scene_median_mse_ratio"]) for row in selected]
            no_pool = [
                float(row["scene_median_mse_ratio"]) for row in selected if row["scene"] != "pool"
            ]
            output.append({
                "config_id": config, "hidden_dim": hidden, "depth": depth,
                "parameter_count": count, "sample_count": spp, "scene_count": len(values),
                "median_scene_ratio": statistics.median(values),
                "scene_balanced_geometric_mean_ratio": _gm(values),
                "worse_scene_count": sum(value > 1.0 for value in values),
                "outside_5pct_scene_count": sum(value < 0.95 or value > 1.05 for value in values),
                "without_pool_median_scene_ratio": statistics.median(no_pool),
                "without_pool_geometric_mean_ratio": _gm(no_pool),
                "without_pool_worse_scene_count": sum(value > 1.0 for value in no_pool),
                "without_pool_outside_5pct_scene_count": sum(
                    value < 0.95 or value > 1.05 for value in no_pool
                ),
            })
    return output


def cost_diagnostics(
    metric_rows: Sequence[Mapping[str, Any]], training_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    output = []
    for config, (hidden, depth, count) in CONFIG_META.items():
        train = [float(row["training_seconds"]) for row in training_rows if row["config_id"] == config]
        render = [float(row["render_seconds"]) for row in metric_rows if row["config_id"] == config]
        if not train or not render:
            raise AnalysisError(f"missing timing rows for {config}")
        output.append({
            "config_id": config, "hidden_dim": hidden, "depth": depth,
            "parameter_count": count, "training_job_count": len(train),
            "median_training_seconds": statistics.median(train),
            "total_training_core_hours": sum(train) * 5.0 / 3600.0,
            "render_job_count": len(render), "median_render_seconds": statistics.median(render),
            "total_render_core_hours": sum(render) / 3600.0,
        })
    return output


def tail_diagnostics(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, int], list[float]] = defaultdict(list)
    for row in rows:
        if row["config_id"] in CONFIG_META:
            groups[(str(row["config_id"]), str(row["scene"]), int(row["sample_count"]))].append(float(row["mse"]))
    output = []
    for (config, scene, spp), values in sorted(groups.items()):
        if len(values) != 10:
            raise AnalysisError(f"tail diagnostic group is incomplete: {config}/{scene}/{spp}")
        median = statistics.median(values)
        output.append({
            "config_id": config, "scene": scene, "sample_count": spp,
            "median_mse": median, "mean_mse": statistics.fmean(values),
            "sample_sd_mse": statistics.stdev(values), "maximum_mse": max(values),
            "mean_to_median_ratio": statistics.fmean(values) / median,
            "maximum_to_median_ratio": max(values) / median,
        })
    return output


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise AnalysisError(f"refusing to write empty CSV: {path}")
    fields = list(rows[0])
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _tex_escape(value: str) -> str:
    return value.replace("_", r"\_").replace("%", r"\%")


def latex_table(
    ratios: Sequence[Mapping[str, Any]], effects: Sequence[Mapping[str, Any]],
    costs: Sequence[Mapping[str, Any]],
) -> str:
    effect60 = {row["config_id"]: row for row in effects if row["sample_count"] == 60}
    cost = {row["config_id"]: row for row in costs}
    lines = [
        r"\begin{table}[t]", r"\centering", r"\small", r"\setlength{\tabcolsep}{4pt}",
        r"\caption{Neural-architecture sensitivity at 60~SPP. Ratios are formed from each scene's D01--D10 median MSE relative to the $h=72,d=3$ control; lower is better. The median across scenes is primary, while the geometric mean (GM) is secondary.}",
        r"\label{tab:art-mpg-architecture-sensitivity}",
        r"\begin{tabular}{lrrrrrr}", r"\toprule",
        r"Architecture & Params & Median & GM & Worse & Outside $\pm5\%$ & Train s/job \\",
        r"\midrule",
    ]
    for config in CONFIG_META:
        row = effect60[config]
        c = cost[config]
        reference = "---" if config == CONTROL else f"{row['median_scene_ratio']:.3f}"
        gm = "---" if config == CONTROL else f"{row['scene_balanced_geometric_mean_ratio']:.3f}"
        worse = "Reference" if config == CONTROL else f"{row['worse_scene_count']}/14"
        outside = "Reference" if config == CONTROL else str(row["outside_5pct_scene_count"])
        lines.append(
            f"{LABELS[config]} & {int(row['parameter_count']):,} & {reference} & {gm} & "
            f"{worse} & {outside} & {c['median_training_seconds']:.1f} \\\\"
        )
    lines.extend([
        r"\bottomrule", r"\end{tabular}",
        r"\end{table}", "",
    ])
    return "\n".join(lines)


def _svg_text(x: float, y: float, value: str, *, size: int = 11, anchor: str = "middle") -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="sans-serif" font-size="{size}" '
        f'text-anchor="{anchor}" fill="#222">{html.escape(value)}</text>'
    )


def sensitivity_svg(effects: Sequence[Mapping[str, Any]]) -> str:
    width, height = 820, 470
    left, right, top, bottom = 75, 25, 35, 70
    plot_w, plot_h = width - left - right, height - top - bottom
    variants = [config for config in CONFIG_META if config != CONTROL]
    values = [float(row["median_scene_ratio"]) for row in effects if row["config_id"] != CONTROL]
    lo = min(0.9, min(values) * 0.98)
    hi = max(1.1, max(values) * 1.02)
    if hi <= lo:
        hi = lo + 0.2
    x = {spp: left + index * plot_w / (len(FULL_SPP) - 1) for index, spp in enumerate(FULL_SPP)}
    y = lambda value: top + (hi - value) * plot_h / (hi - lo)
    colors = {"h36_d3": "#D55E00", "h72_d2": "#E69F00", "h72_d4": "#009E73", "h144_d3": "#0072B2"}
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">', '<rect width="100%" height="100%" fill="white"/>']
    for tick in range(6):
        value = lo + tick * (hi - lo) / 5
        yy = y(value)
        parts.append(f'<line x1="{left}" y1="{yy:.1f}" x2="{width-right}" y2="{yy:.1f}" stroke="#ddd"/>')
        parts.append(_svg_text(left - 8, yy + 4, f"{value:.2f}", anchor="end"))
    parts.append(f'<line x1="{left}" y1="{y(1):.1f}" x2="{width-right}" y2="{y(1):.1f}" stroke="#222" stroke-dasharray="5 4"/>')
    for spp in FULL_SPP:
        parts.append(_svg_text(x[spp], height - 40, str(spp)))
    for index, config in enumerate(variants):
        rows = sorted((row for row in effects if row["config_id"] == config), key=lambda row: row["sample_count"])
        points = " ".join(f"{x[int(row['sample_count'])]:.1f},{y(float(row['median_scene_ratio'])):.1f}" for row in rows)
        color = colors[config]
        parts.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2.5"/>')
        for row in rows:
            parts.append(f'<circle cx="{x[int(row["sample_count"])] :.1f}" cy="{y(float(row["median_scene_ratio"])):.1f}" r="3.2" fill="{color}"/>')
        parts.append(f'<line x1="{left + 18 + index*172}" y1="{height-14}" x2="{left+42+index*172}" y2="{height-14}" stroke="{color}" stroke-width="3"/>')
        parts.append(_svg_text(left + 48 + index*172, height - 10, LABELS[config].replace("$", ""), anchor="start"))
    parts.append(_svg_text(left + plot_w / 2, height - 48, "Samples per pixel (SPP)", size=12))
    parts.append(f'<text x="18" y="{top+plot_h/2:.1f}" transform="rotate(-90 18 {top+plot_h/2:.1f})" font-family="sans-serif" font-size="12" text-anchor="middle">Median across scene-level median-MSE ratios</text>')
    parts.append("</svg>\n")
    return "\n".join(parts)


def cli_source_bindings(rows: Path, training: Path, *, strict: bool) -> dict[str, Any]:
    bindings = {
        "experiment_manifest": {
            "path": str(PACKAGE_ROOT / "manifest.json"),
            "sha256": _sha256_file(PACKAGE_ROOT / "manifest.json"),
        },
        "collected_metric_csv": {"path": str(rows), "sha256": _sha256_file(rows)},
        "collected_training_csv": {"path": str(training), "sha256": _sha256_file(training)},
        "analysis_implementation": {
            "path": str(Path(__file__).resolve()), "sha256": _sha256_file(Path(__file__).resolve())
        },
    }
    receipt_path = rows.parents[1] / "completion_receipt.json"
    if not receipt_path.is_file():
        if strict:
            raise AnalysisError(f"strict analysis requires collection receipt: {receipt_path}")
        return bindings
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("protocol_id") != PROTOCOL_ID or receipt.get("status") != "complete":
        raise AnalysisError("collection receipt has wrong protocol or status")
    if receipt.get("metrics", {}).get("csv", {}).get("sha256") != bindings["collected_metric_csv"]["sha256"]:
        raise AnalysisError("metric CSV hash differs from collection receipt")
    if receipt.get("metrics", {}).get("training_csv", {}).get("sha256") != bindings["collected_training_csv"]["sha256"]:
        raise AnalysisError("training CSV hash differs from collection receipt")
    if receipt.get("manifest_sha256") != bindings["experiment_manifest"]["sha256"]:
        raise AnalysisError("manifest hash differs from collection receipt")
    bindings["collection_receipt"] = {"path": str(receipt_path), "sha256": _sha256_file(receipt_path)}
    return bindings


def analyze(
    metric_rows: Sequence[Mapping[str, Any]], training_rows: Sequence[Mapping[str, Any]],
    output_root: Path, *, strict: bool, source_bindings: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    validate_rows(metric_rows, strict=strict)
    validate_training_rows(training_rows, strict=strict)
    ratios = scene_median_ratios(metric_rows)
    effects = architecture_effects(ratios)
    costs = cost_diagnostics(metric_rows, training_rows)
    tails = tail_diagnostics(metric_rows)
    output_root.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "scene_median_ratios.csv": ratios,
        "architecture_effects.csv": effects,
        "architecture_costs.csv": costs,
        "tail_diagnostics.csv": tails,
    }
    for name, rows in artifacts.items():
        _write_csv(output_root / name, rows)
    _atomic_bytes(output_root / "architecture_sensitivity_60spp.tex", latex_table(ratios, effects, costs).encode())
    _atomic_bytes(output_root / "architecture_sensitivity.svg", sensitivity_svg(effects).encode())
    summary = {
        "protocol_id": PROTOCOL_ID,
        "status": "analysis_complete",
        "primary_estimand": "median across the 14 ratios of D01--D10 scene-median MSE",
        "secondary_estimand": "scene-balanced geometric mean of those 14 ratios",
        "control_config": CONTROL,
        "metric_row_count": len(metric_rows),
        "training_row_count": len(training_rows),
        "effects": effects,
        "costs": costs,
        "tail_summary": {
            "group_count": len(tails),
            "groups_mean_over_10x_median": sum(float(row["mean_to_median_ratio"]) > 10 for row in tails),
            "maximum_mean_to_median_ratio": max(float(row["mean_to_median_ratio"]) for row in tails),
            "maximum_max_to_median_ratio": max(float(row["maximum_to_median_ratio"]) for row in tails),
        },
    }
    _atomic_json(output_root / "sensitivity_summary.json", summary)
    artifact_hashes = {
        path.name: {"sha256": _sha256_file(path), "bytes": path.stat().st_size}
        for path in sorted(output_root.iterdir())
        if path.is_file() and path.name != "analysis_receipt.json"
    }
    receipt = {
        "schema_version": 1, "protocol_id": PROTOCOL_ID, "status": "analysis_complete",
        "strict_grid_validation": strict, "source_bindings": dict(source_bindings or {}),
        "artifacts": artifact_hashes, "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    _atomic_json(output_root / "analysis_receipt.json", receipt)
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=Path, default=DEFAULT_ROWS)
    parser.add_argument("--training-rows", type=Path, default=DEFAULT_TRAINING)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args(argv)
    rows = normalize_metric_rows(read_csv(args.rows))
    training = normalize_training_rows(read_csv(args.training_rows))
    strict = not args.allow_partial
    bindings = cli_source_bindings(args.rows, args.training_rows, strict=strict)
    receipt = analyze(rows, training, args.output, strict=strict, source_bindings=bindings)
    print(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AnalysisError as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
