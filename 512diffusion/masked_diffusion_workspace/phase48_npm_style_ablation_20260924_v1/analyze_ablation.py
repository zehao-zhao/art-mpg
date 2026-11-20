#!/usr/bin/env python3
"""Analyze descriptor and training-budget ablations."""

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
from typing import Any, Callable, Iterable, Mapping, Sequence


PACKAGE_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = PACKAGE_ROOT.parent
PROTOCOL_ID = "phase48_npm_style_ablation_20260924_v1"
RESULT_ROOT = WORKSPACE_ROOT / "results" / PROTOCOL_ID
DEFAULT_ROWS = RESULT_ROOT / "metrics" / "ablation_rows.csv"
DEFAULT_TRAINING_ROWS = RESULT_ROOT / "metrics" / "training_rows.csv"
DEFAULT_OUTPUT = RESULT_ROOT / "analysis"

SCENES = (
    "pool", "living", "lamp", "plane", "slabs", "stone", "sphere",
    "water", "ring_and_slab", "pillars", "slab_x1", "slab_x3",
    "pillar_x3", "cornell_glass",
)
RUN_IDS = tuple(f"D{index:02d}" for index in range(1, 11))
FULL_SPP = (12, 24, 36, 48, 60)
CONFIG_META: dict[str, tuple[str, int, tuple[int, ...]]] = {
    "full_e32": ("full", 32, FULL_SPP),
    "length_only_e32": ("length_only", 32, FULL_SPP),
    "no_topology_e32": ("no_topology", 32, FULL_SPP),
    "full_e8": ("full", 8, (60,)),
    "full_e16": ("full", 16, (60,)),
    "no_topology_e8": ("no_topology", 8, (60,)),
    "no_topology_e16": ("no_topology", 16, (60,)),
}
IDENTITY_FIELDS = ("scene", "sample_count", "run_id", "model_seed", "render_seed")
INTERPRETATION = (
    "All statistics are matched descriptive summaries, not independent-replicate confidence "
    "intervals or hypothesis tests."
)
CONFIGURATION_NOTE = (
    "The no_topology configuration still predicts length and R/T descriptors; "
    "only the direction head is denied those predicted descriptors. All arms "
    "are retrained, so shared-backbone gradients may also change descriptor PMFs."
)


class AnalysisError(RuntimeError):
    """Raised when an input or comparison violates the analysis contract."""


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def _write_json(path: Path, value: object) -> None:
    payload = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    _atomic_write(path, payload)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _finite_float(value: object, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise AnalysisError(f"{label} must be numeric: {value!r}") from exc
    if not math.isfinite(result):
        raise AnalysisError(f"{label} must be finite: {value!r}")
    return result


def _integer(value: object, label: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise AnalysisError(f"{label} must be an integer: {value!r}") from exc


def load_metric_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise AnalysisError(f"collected metric CSV does not exist: {path}")
    required = {
        "config_id", "direction_conditioning_mode", "epochs", "run_id",
        "scene", "sample_count", "model_seed", "render_seed",
        "mse", "output_sha256",
    }
    rows: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise AnalysisError(f"metric CSV lacks fields: {sorted(missing)}")
        for line_number, source in enumerate(reader, start=2):
            row = dict(source)
            row["epochs"] = _integer(row["epochs"], f"line {line_number} epochs")
            row["sample_count"] = _integer(row["sample_count"], f"line {line_number} sample_count")
            row["model_seed"] = _integer(row["model_seed"], f"line {line_number} model_seed")
            row["render_seed"] = _integer(row["render_seed"], f"line {line_number} render_seed")
            row["mse"] = _finite_float(row["mse"], f"line {line_number} mse")
            if row["mse"] <= 0:
                raise AnalysisError(f"line {line_number} MSE must be positive for ratio analysis")
            rows.append(row)
    return rows


def load_training_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    required = {
        "config_id", "direction_conditioning_mode", "epochs", "scene",
        "model_seed", "training_seconds", "epochs_completed", "final_val_loss",
        "best_val_loss",
    }
    source_rows: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise AnalysisError(f"training CSV lacks fields: {sorted(missing)}")
        source_rows.extend(dict(row) for row in reader)
    return normalize_training_rows(source_rows)


def normalize_metric_rows(source_rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Normalize in-memory rows; useful for tests and downstream reuse."""
    rows: list[dict[str, Any]] = []
    for index, source in enumerate(source_rows, start=1):
        row = dict(source)
        for field in ("config_id", "direction_conditioning_mode", "run_id", "scene"):
            if field not in row or not str(row[field]):
                raise AnalysisError(f"row {index} lacks {field}")
            row[field] = str(row[field])
        for field in ("epochs", "sample_count", "model_seed", "render_seed"):
            row[field] = _integer(row.get(field), f"row {index} {field}")
        row["mse"] = _finite_float(row.get("mse"), f"row {index} mse")
        if row["mse"] <= 0:
            raise AnalysisError(f"row {index} MSE must be positive for ratio analysis")
        row["output_sha256"] = str(row.get("output_sha256", ""))
        rows.append(row)
    return rows


def normalize_training_rows(source_rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, source in enumerate(source_rows, start=1):
        row = dict(source)
        for field in ("config_id", "direction_conditioning_mode", "scene"):
            if field not in row or not str(row[field]):
                raise AnalysisError(f"training row {index} lacks {field}")
            row[field] = str(row[field])
        row["epochs"] = _integer(row.get("epochs"), f"training row {index} epochs")
        row["epochs_completed"] = _integer(
            row.get("epochs_completed"), f"training row {index} epochs_completed"
        )
        row["model_seed"] = _integer(row.get("model_seed"), f"training row {index} model_seed")
        for field in ("training_seconds", "final_val_loss", "best_val_loss"):
            row[field] = _finite_float(row.get(field), f"training row {index} {field}")
        if row["training_seconds"] < 0:
            raise AnalysisError(f"training row {index} has negative duration")
        rows.append(row)
    return rows


def validate_rows(rows: Sequence[Mapping[str, Any]], *, strict: bool) -> None:
    if not rows:
        raise AnalysisError("no metric rows were provided")
    seen: set[tuple[Any, ...]] = set()
    run_bindings: dict[str, tuple[int, int]] = {}
    for row in rows:
        config_id = str(row["config_id"])
        if config_id not in CONFIG_META:
            raise AnalysisError(f"unexpected configuration: {config_id}")
        expected_mode, expected_epochs, expected_spp = CONFIG_META[config_id]
        if str(row["direction_conditioning_mode"]) != expected_mode:
            raise AnalysisError(f"mode mismatch for {config_id}")
        if int(row["epochs"]) != expected_epochs:
            raise AnalysisError(f"epoch mismatch for {config_id}")
        if int(row["sample_count"]) not in expected_spp:
            raise AnalysisError(f"SPP mismatch for {config_id}: {row['sample_count']}")
        key = (config_id, *(row[field] for field in IDENTITY_FIELDS))
        if key in seen:
            raise AnalysisError(f"duplicate metric row: {key}")
        seen.add(key)
        run_id = str(row["run_id"])
        binding = (
            int(row["model_seed"]), int(row["render_seed"])
        )
        prior = run_bindings.setdefault(run_id, binding)
        if prior != binding:
            raise AnalysisError(
                f"run identity {run_id} maps to inconsistent model/render seeds"
            )

    if not strict:
        return
    expected_scenes = set(SCENES)
    expected_runs = set(RUN_IDS)
    if set(str(row["config_id"]) for row in rows) != set(CONFIG_META):
        raise AnalysisError("strict analysis requires all seven collected configurations")
    manifest_path = PACKAGE_ROOT / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected_bindings = {
            str(row["run_id"]): (
                int(row["model_seed"]), int(row["render_seed"])
            )
            for row in manifest["recorded_runs"]
        }
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise AnalysisError(f"cannot load canonical run bindings from {manifest_path}") from exc
    if run_bindings != expected_bindings:
        raise AnalysisError("strict run-to-model/render bindings differ from manifest")
    for config_id, (_, _, spp_values) in CONFIG_META.items():
        selected = [row for row in rows if row["config_id"] == config_id]
        expected_count = len(SCENES) * len(RUN_IDS) * len(spp_values)
        if len(selected) != expected_count:
            raise AnalysisError(
                f"strict grid count mismatch for {config_id}: {len(selected)} != {expected_count}"
            )
        if {str(row["scene"]) for row in selected} != expected_scenes:
            raise AnalysisError(f"strict scene grid mismatch for {config_id}")
        if {str(row["run_id"]) for row in selected} != expected_runs:
            raise AnalysisError(f"strict run grid mismatch for {config_id}")
        cell_runs: dict[tuple[str, int], set[str]] = defaultdict(set)
        for row in selected:
            cell_runs[(str(row["scene"]), int(row["sample_count"]))].add(str(row["run_id"]))
        if (
            any(value != expected_runs for value in cell_runs.values())
            or len(cell_runs) != len(SCENES) * len(spp_values)
        ):
            raise AnalysisError(f"strict D01--D10 cells are incomplete for {config_id}")
    if len(rows) != 2660:
        raise AnalysisError(f"strict combined row count must be 2660, observed {len(rows)}")


def _identity(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return tuple(row[field] for field in IDENTITY_FIELDS)


def paired_rows(
    rows: Sequence[Mapping[str, Any]], treatment: str, control: str,
    *, sample_counts: set[int] | None = None,
) -> list[dict[str, Any]]:
    def selected(config_id: str) -> dict[tuple[Any, ...], Mapping[str, Any]]:
        result: dict[tuple[Any, ...], Mapping[str, Any]] = {}
        for row in rows:
            if row["config_id"] != config_id:
                continue
            if sample_counts is not None and int(row["sample_count"]) not in sample_counts:
                continue
            key = _identity(row)
            if key in result:
                raise AnalysisError(f"duplicate pairing identity for {config_id}: {key}")
            result[key] = row
        return result

    treatment_rows = selected(treatment)
    control_rows = selected(control)
    if not treatment_rows or not control_rows:
        raise AnalysisError(f"empty comparison: {treatment} versus {control}")
    if set(treatment_rows) != set(control_rows):
        missing_treatment = sorted(set(control_rows) - set(treatment_rows))[:3]
        missing_control = sorted(set(treatment_rows) - set(control_rows))[:3]
        raise AnalysisError(
            f"unmatched comparison {treatment} versus {control}; "
            f"missing treatment={missing_treatment}, missing control={missing_control}"
        )
    result: list[dict[str, Any]] = []
    for key in sorted(treatment_rows):
        treatment_row = treatment_rows[key]
        control_row = control_rows[key]
        treatment_mse = float(treatment_row["mse"])
        control_mse = float(control_row["mse"])
        ratio = treatment_mse / control_mse
        delta = treatment_mse - control_mse
        result.append({
            **dict(zip(IDENTITY_FIELDS, key)),
            "treatment_config": treatment,
            "control_config": control,
            "treatment_mse": treatment_mse,
            "control_mse": control_mse,
            "signed_delta_mse": delta,
            "absolute_delta_mse": abs(delta),
            "relative_delta": ratio - 1.0,
            "absolute_relative_delta": abs(ratio - 1.0),
            "mse_ratio": ratio,
            "log_mse_ratio": math.log(ratio),
            "treatment_output_sha256": treatment_row.get("output_sha256", ""),
            "control_output_sha256": control_row.get("output_sha256", ""),
            "output_sha256_equal": bool(
                treatment_row.get("output_sha256")
                and treatment_row.get("output_sha256") == control_row.get("output_sha256")
            ),
        })
    return result


def _geometric_mean(values: Sequence[float]) -> float:
    if not values or any(value <= 0 or not math.isfinite(value) for value in values):
        raise AnalysisError("geometric mean requires finite positive values")
    return math.exp(statistics.fmean(math.log(value) for value in values))


def _pearson(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) != len(right) or not left:
        raise AnalysisError("Pearson inputs must be nonempty and equally sized")
    left_mean = statistics.fmean(left)
    right_mean = statistics.fmean(right)
    numerator = sum((x - left_mean) * (y - right_mean) for x, y in zip(left, right))
    left_ss = sum((x - left_mean) ** 2 for x in left)
    right_ss = sum((y - right_mean) ** 2 for y in right)
    if left_ss == 0 or right_ss == 0:
        return None
    return numerator / math.sqrt(left_ss * right_ss)


def summarize_pairs(pairs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not pairs:
        raise AnalysisError("cannot summarize an empty paired comparison")
    treatment = [float(row["treatment_mse"]) for row in pairs]
    control = [float(row["control_mse"]) for row in pairs]
    ratios = [float(row["mse_ratio"]) for row in pairs]
    signed = [float(row["signed_delta_mse"]) for row in pairs]
    absolute = [float(row["absolute_delta_mse"]) for row in pairs]
    absolute_relative = [float(row["absolute_relative_delta"]) for row in pairs]
    return {
        "n": len(pairs),
        "geometric_mean_mse_ratio": _geometric_mean(ratios),
        "median_mse_ratio": statistics.median(ratios),
        "mean_mse_ratio": statistics.fmean(ratios),
        "median_relative_delta_pct": 100.0 * statistics.median(row["relative_delta"] for row in pairs),
        "mean_signed_delta_mse": statistics.fmean(signed),
        "mean_absolute_delta_mse": statistics.fmean(absolute),
        "median_absolute_delta_mse": statistics.median(absolute),
        "max_absolute_delta_mse": max(absolute),
        "median_absolute_relative_delta": statistics.median(absolute_relative),
        "max_absolute_relative_delta": max(absolute_relative),
        "rmse_of_delta": math.sqrt(statistics.fmean(value * value for value in signed)),
        "pearson_r": _pearson(treatment, control),
        "treatment_lower_count": sum(x < y for x, y in zip(treatment, control)),
        "equal_mse_count": sum(x == y for x, y in zip(treatment, control)),
        "treatment_higher_count": sum(x > y for x, y in zip(treatment, control)),
        "treatment_lower_fraction": sum(x < y for x, y in zip(treatment, control)) / len(pairs),
    }


def _group_summaries(
    pairs: Sequence[Mapping[str, Any]],
    groupers: Sequence[tuple[str, Callable[[Mapping[str, Any]], Any]]],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for scope, grouper in groupers:
        groups: dict[Any, list[Mapping[str, Any]]] = defaultdict(list)
        for row in pairs:
            groups[grouper(row)].append(row)
        for group_value, selected in sorted(groups.items(), key=lambda item: str(item[0])):
            scene: str | None = None
            sample_count: int | None = None
            if scope == "scene":
                scene = str(group_value)
            elif scope == "sample_count":
                sample_count = int(group_value)
            elif scope == "scene_sample_count":
                scene, sample_count = str(group_value[0]), int(group_value[1])
            result.append({
                "scope": scope, "scene": scene, "sample_count": sample_count,
                **summarize_pairs(selected),
            })
    return result


PAIR_SUMMARY_FIELDS = (
    "comparison", "variant_config", "control_config", "scope", "scene", "sample_count",
    "n", "geometric_mean_mse_ratio", "median_mse_ratio", "mean_mse_ratio",
    "median_relative_delta_pct", "mean_signed_delta_mse", "mean_absolute_delta_mse",
    "median_absolute_delta_mse", "max_absolute_delta_mse",
    "median_absolute_relative_delta", "max_absolute_relative_delta", "rmse_of_delta",
    "pearson_r", "treatment_lower_count", "equal_mse_count", "treatment_higher_count",
    "treatment_lower_fraction",
)


def topology_analysis(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for variant in ("length_only_e32", "no_topology_e32"):
        pairs = paired_rows(rows, variant, "full_e32")
        summaries = [{"scope": "overall", "scene": None, "sample_count": None, **summarize_pairs(pairs)}]
        summaries.extend(_group_summaries(pairs, (
            ("scene", lambda row: row["scene"]),
            ("sample_count", lambda row: row["sample_count"]),
            ("scene_sample_count", lambda row: (row["scene"], row["sample_count"])),
        )))
        for summary in summaries:
            result.append({
                "comparison": f"{variant}_over_full_e32",
                "variant_config": variant,
                "control_config": "full_e32",
                **summary,
            })
    return result


def training_budget_analysis(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for config_id in (
        "full_e8", "full_e16", "full_e32",
        "no_topology_e8", "no_topology_e16", "no_topology_e32",
    ):
        mode, epochs, _ = CONFIG_META[config_id]
        control_config = "full_e32" if mode == "full" else "no_topology_e32"
        pairs = paired_rows(rows, config_id, control_config, sample_counts={60})
        summaries = [{"scope": "overall", "scene": None, "sample_count": 60, **summarize_pairs(pairs)}]
        summaries.extend(_group_summaries(pairs, (("scene", lambda row: row["scene"]),)))
        for summary in summaries:
            result.append({
                "comparison": f"{config_id}_over_{control_config}_at_60spp",
                "variant_config": config_id,
                "control_config": control_config,
                "direction_conditioning_mode": mode,
                "epochs": epochs,
                **summary,
            })
    return result


def training_efficiency_analysis(rows: Sequence[Mapping[str, Any]], *, strict: bool) -> list[dict[str, Any]]:
    if not rows:
        return []
    seen: set[tuple[str, str, int]] = set()
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        config_id = str(row["config_id"])
        if config_id not in CONFIG_META:
            raise AnalysisError(f"unexpected training configuration: {config_id}")
        expected_mode, expected_epochs, _ = CONFIG_META[config_id]
        if str(row["direction_conditioning_mode"]) != expected_mode or int(row["epochs"]) != expected_epochs:
            raise AnalysisError(f"training metadata mismatch for {config_id}")
        if int(row["epochs_completed"]) != expected_epochs:
            raise AnalysisError(f"training did not complete its declared epochs for {config_id}")
        key = (config_id, str(row["scene"]), int(row["model_seed"]))
        if key in seen:
            raise AnalysisError(f"duplicate training metric row: {key}")
        seen.add(key)
        groups[config_id].append(row)
    if strict:
        if set(groups) != set(CONFIG_META):
            raise AnalysisError("strict analysis requires training metrics for all seven new configurations")
        for config_id, selected in groups.items():
            if len(selected) != 28:
                raise AnalysisError(f"strict training row count for {config_id} must be 28")
            if {str(row["scene"]) for row in selected} != set(SCENES):
                raise AnalysisError(f"strict training scene grid mismatch for {config_id}")
            observed_grid = {
                (str(row["scene"]), int(row["model_seed"])) for row in selected
            }
            expected_grid = {
                (scene, model_seed)
                for scene in SCENES for model_seed in (104729, 155921)
            }
            if observed_grid != expected_grid:
                raise AnalysisError(f"strict training model-seed grid mismatch for {config_id}")
    result: list[dict[str, Any]] = []
    for config_id in CONFIG_META:
        selected = groups.get(config_id, [])
        if not selected:
            continue
        durations = [float(row["training_seconds"]) for row in selected]
        final_losses = [float(row["final_val_loss"]) for row in selected]
        best_losses = [float(row["best_val_loss"]) for row in selected]
        mode, epochs, _ = CONFIG_META[config_id]
        result.append({
            "config_id": config_id,
            "direction_conditioning_mode": mode,
            "epochs": epochs,
            "n_training_jobs": len(selected),
            "total_training_hours": sum(durations) / 3600.0,
            "mean_training_seconds": statistics.fmean(durations),
            "median_training_seconds": statistics.median(durations),
            "min_training_seconds": min(durations),
            "max_training_seconds": max(durations),
            "mean_final_val_loss": statistics.fmean(final_losses),
            "median_final_val_loss": statistics.median(final_losses),
            "mean_best_val_loss": statistics.fmean(best_losses),
            "median_best_val_loss": statistics.median(best_losses),
        })
    return result


def _median_mse(
    rows: Sequence[Mapping[str, Any]], config_id: str, scene: str, sample_count: int,
) -> float:
    values = [
        float(row["mse"]) for row in rows
        if row["config_id"] == config_id and row["scene"] == scene
        and int(row["sample_count"]) == sample_count
    ]
    if not values:
        raise AnalysisError(f"missing median cell: {config_id}/{scene}/{sample_count}")
    return statistics.median(values)


def _latex_escape(value: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
        "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
    }
    return "".join(replacements.get(character, character) for character in value)


def _latex_number(value: float, *, digits: int = 3) -> str:
    if value == 0:
        return "$0$"
    exponent = math.floor(math.log10(abs(value)))
    if exponent <= -3 or exponent >= 4:
        mantissa = value / (10 ** exponent)
        return f"${mantissa:.{digits}f}\\!\\times\\!10^{{{exponent}}}$"
    return f"${value:.{digits}g}$"


def topology_latex(rows: Sequence[Mapping[str, Any]], topology: Sequence[Mapping[str, Any]]) -> str:
    scene_order = [scene for scene in SCENES if any(row["scene"] == scene for row in rows)]
    lines = [
        r"\begin{table*}[t]", r"\centering", r"\small",
        r"\setlength{\tabcolsep}{4.2pt}", r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lrrrrr}", r"\toprule",
        r"Scene & Full & Length only & Context only & Length/full & Context/full \\",
        r"\midrule",
    ]
    scene_effects = {
        (str(row["variant_config"]), str(row["scene"])): row
        for row in topology
        if row["scope"] == "scene_sample_count" and row["sample_count"] == 60
    }
    for scene in scene_order:
        full = _median_mse(rows, "full_e32", scene, 60)
        length = _median_mse(rows, "length_only_e32", scene, 60)
        none = _median_mse(rows, "no_topology_e32", scene, 60)
        length_ratio = scene_effects[("length_only_e32", scene)]["geometric_mean_mse_ratio"]
        context_ratio = scene_effects[("no_topology_e32", scene)]["geometric_mean_mse_ratio"]
        lines.append(
            f"{_latex_escape(scene.replace('_', ' ').title())} & {_latex_number(full)} & "
            f"{_latex_number(length)} & {_latex_number(none)} & {length_ratio:.3f} & "
            f"{context_ratio:.3f} \\\\" 
        )
    at_60_spp = {
        row["variant_config"]: row for row in topology
        if row["scope"] == "sample_count" and row["sample_count"] == 60
    }
    lines.extend([
        r"\midrule",
        "Paired geometric mean ratio & -- & -- & -- & "
        f"{at_60_spp['length_only_e32']['geometric_mean_mse_ratio']:.3f} & "
        f"{at_60_spp['no_topology_e32']['geometric_mean_mse_ratio']:.3f}" + r" \\",
        r"\bottomrule", r"\end{tabular}%", r"}",
        r"\caption{Direction-conditioning ablation at 60 samples per pixel. "
        r"The first three numeric columns are median MSE over the recorded D01--D10 runs. "
        r"The final two columns and final row are geometric means of matched per-run "
        r"MSE ratios; values above one mean that removing conditioning increases error. "
        r"Context only removes predicted length/R/T conditioning from the direction "
        r"head, while descriptor prediction remains active and every arm is retrained. The final "
        r"row is the geometric mean of matched per-run ratios over all scenes at 60 SPP, "
        r"not a ratio of globally pooled MSE values.}",
        r"\label{tab:art-mpg-topology-ablation}", r"\end{table*}", "",
    ])
    return "\n".join(lines)


def budget_latex(
    budget: Sequence[Mapping[str, Any]], efficiency: Sequence[Mapping[str, Any]],
) -> str:
    overall = {
        row["variant_config"]: row for row in budget if row["scope"] == "overall"
    }
    efficiency_by_config = {row["config_id"]: row for row in efficiency}
    lines = [
        r"\begin{table}[t]", r"\centering", r"\small",
        r"\resizebox{\columnwidth}{!}{%", r"\begin{tabular}{lrrr}", r"\toprule",
        r"Direction conditioning / metric & 8 epochs & 16 epochs & 32 epochs \\",
        r"\midrule",
    ]
    for label, prefix in (("Full", "full"), ("Context-only direction", "no_topology")):
        configs = [f"{prefix}_e{epochs}" for epochs in (8, 16, 32)]
        ratios = [overall[config]["geometric_mean_mse_ratio"] for config in configs]
        seconds = [
            "--" if efficiency_by_config.get(config) is None
            else f"{efficiency_by_config[config]['median_training_seconds']:.1f}"
            for config in configs
        ]
        lines.append(
            f"{label}: MSE ratio & {ratios[0]:.3f} & {ratios[1]:.3f} & {ratios[2]:.3f} \\\\" 
        )
        lines.append(
            f"{label}: median train s/job & {seconds[0]} & {seconds[1]} & {seconds[2]} \\\\" 
        )
        if prefix == "full":
            lines.append(r"\addlinespace")
    lines.extend([
        r"\bottomrule", r"\end{tabular}%", r"}",
        r"\caption{Training-budget ablation at 60 samples per pixel. Each MSE entry is "
        r"the geometric mean of matched per-run ratios to the corresponding 32-epoch "
        r"conditioning mode; lower is better and each 32-epoch control is exactly 1.000. "
        r"Training times summarize per-scene, per-model-seed jobs and are descriptive "
        r"wall times. Context only removes descriptor conditioning from the direction "
        r"head; the descriptor heads remain active.}",
        r"\label{tab:art-mpg-training-budget-ablation}", r"\end{table}", "",
    ])
    return "\n".join(lines)


def _svg_text(x: float, y: float, value: str, *, size: int = 12, anchor: str = "start", weight: str = "normal") -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="sans-serif" font-size="{size}" '
        f'text-anchor="{anchor}" font-weight="{weight}">{html.escape(value)}</text>'
    )


def topology_svg(rows: Sequence[Mapping[str, Any]]) -> str:
    configs = ("full_e32", "length_only_e32", "no_topology_e32")
    labels = {
        "full_e32": "Full",
        "length_only_e32": "Length only",
        "no_topology_e32": "Context only (direction)",
    }
    colors = {"full_e32": "#0072B2", "length_only_e32": "#E69F00", "no_topology_e32": "#D55E00"}
    scenes = [scene for scene in SCENES if any(row["scene"] == scene for row in rows)]
    columns = 4
    rows_count = math.ceil(len(scenes) / columns)
    panel_width, panel_height = 290, 210
    width, height = columns * panel_width, 70 + rows_count * panel_height
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        _svg_text(width / 2, 24, "ART-MPG direction-conditioning ablation", size=17, anchor="middle", weight="bold"),
    ]
    legend_x = width / 2 - 210
    for index, config in enumerate(configs):
        x = legend_x + 145 * index
        svg.append(f'<line x1="{x:.1f}" y1="46" x2="{x + 24:.1f}" y2="46" stroke="{colors[config]}" stroke-width="2.5"/>')
        svg.append(_svg_text(x + 30, 50, labels[config], size=11))

    for scene_index, scene in enumerate(scenes):
        column, row_index = scene_index % columns, scene_index // columns
        x0, y0 = column * panel_width + 48, 70 + row_index * panel_height + 22
        plot_width, plot_height = 220, 142
        values = {
            config: [_median_mse(rows, config, scene, spp) for spp in FULL_SPP]
            for config in configs
        }
        all_logs = [math.log10(value) for series in values.values() for value in series]
        lower, upper = min(all_logs), max(all_logs)
        padding = max((upper - lower) * 0.08, 0.04)
        lower, upper = lower - padding, upper + padding
        if upper == lower:
            upper += 0.1
        map_x = lambda spp: x0 + (spp - FULL_SPP[0]) / (FULL_SPP[-1] - FULL_SPP[0]) * plot_width
        map_y = lambda value: y0 + plot_height - (math.log10(value) - lower) / (upper - lower) * plot_height
        svg.extend([
            f'<rect x="{x0}" y="{y0}" width="{plot_width}" height="{plot_height}" fill="none" stroke="#777" stroke-width="0.8"/>',
            _svg_text(x0 + plot_width / 2, y0 - 7, scene.replace("_", " ").title(), size=12, anchor="middle", weight="bold"),
        ])
        for fraction in (0.0, 0.5, 1.0):
            y = y0 + plot_height * (1.0 - fraction)
            log_value = lower + fraction * (upper - lower)
            svg.append(f'<line x1="{x0}" y1="{y:.1f}" x2="{x0 + plot_width}" y2="{y:.1f}" stroke="#ddd" stroke-width="0.7"/>')
            svg.append(_svg_text(x0 - 5, y + 4, f"{10 ** log_value:.1e}", size=8, anchor="end"))
        for spp in FULL_SPP:
            x = map_x(spp)
            svg.append(_svg_text(x, y0 + plot_height + 16, str(spp), size=9, anchor="middle"))
        for config in configs:
            points = [(map_x(spp), map_y(value)) for spp, value in zip(FULL_SPP, values[config])]
            svg.append(
                f'<polyline points="{" ".join(f"{x:.1f},{y:.1f}" for x, y in points)}" '
                f'fill="none" stroke="{colors[config]}" stroke-width="2"/>'
            )
            for x, y in points:
                svg.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="2.7" fill="{colors[config]}"/>')
        svg.append(_svg_text(x0 + plot_width / 2, y0 + plot_height + 31, "SPP", size=9, anchor="middle"))
    svg.append(
        f'<text x="16" y="{height / 2:.1f}" font-family="sans-serif" font-size="11" '
        f'text-anchor="middle" transform="rotate(-90 16 {height / 2:.1f})">'
        "Median MSE (log scale)</text>"
    )
    svg.append("</svg>\n")
    return "\n".join(svg)


def budget_svg(budget: Sequence[Mapping[str, Any]]) -> str:
    overall = {row["variant_config"]: row for row in budget if row["scope"] == "overall"}
    series = {
        "Full": [float(overall[f"full_e{epoch}"]["geometric_mean_mse_ratio"]) for epoch in (8, 16, 32)],
        "Context only (direction)": [
            float(overall[f"no_topology_e{epoch}"]["geometric_mean_mse_ratio"])
            for epoch in (8, 16, 32)
        ],
    }
    colors = {"Full": "#0072B2", "Context only (direction)": "#D55E00"}
    width, height = 680, 430
    x0, y0, plot_width, plot_height = 85, 65, 545, 285
    all_values = [value for values in series.values() for value in values] + [1.0]
    lower, upper = min(all_values), max(all_values)
    padding = max((upper - lower) * 0.12, 0.04)
    lower = max(0.0, lower - padding)
    upper += padding
    map_x = lambda epoch: x0 + (epoch - 8) / 24 * plot_width
    map_y = lambda value: y0 + plot_height - (value - lower) / (upper - lower) * plot_height
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        _svg_text(width / 2, 28, "Training-budget ablation at 60 SPP", size=17, anchor="middle", weight="bold"),
        f'<rect x="{x0}" y="{y0}" width="{plot_width}" height="{plot_height}" fill="none" stroke="#777"/>',
    ]
    for tick in range(6):
        value = lower + tick * (upper - lower) / 5
        y = map_y(value)
        svg.append(f'<line x1="{x0}" y1="{y:.1f}" x2="{x0 + plot_width}" y2="{y:.1f}" stroke="#ddd"/>')
        svg.append(_svg_text(x0 - 8, y + 4, f"{value:.2f}", size=10, anchor="end"))
    reference_y = map_y(1.0)
    svg.append(f'<line x1="{x0}" y1="{reference_y:.1f}" x2="{x0 + plot_width}" y2="{reference_y:.1f}" stroke="#555" stroke-dasharray="5,4"/>')
    for epoch in (8, 16, 32):
        x = map_x(epoch)
        svg.append(_svg_text(x, y0 + plot_height + 22, str(epoch), size=11, anchor="middle"))
    for label, values in series.items():
        points = [(map_x(epoch), map_y(value)) for epoch, value in zip((8, 16, 32), values)]
        svg.append(
            f'<polyline points="{" ".join(f"{x:.1f},{y:.1f}" for x, y in points)}" '
            f'fill="none" stroke="{colors[label]}" stroke-width="2.8"/>'
        )
        for x, y in points:
            svg.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{colors[label]}"/>')
        legend_y = 48
        legend_x = 165 if label == "Full" else 325
        svg.append(f'<line x1="{legend_x}" y1="{legend_y}" x2="{legend_x + 25}" y2="{legend_y}" stroke="{colors[label]}" stroke-width="2.8"/>')
        svg.append(_svg_text(legend_x + 32, legend_y + 4, label, size=11))
    svg.extend([
        _svg_text(x0 + plot_width / 2, height - 43, "Training epochs", size=12, anchor="middle"),
        f'<text x="20" y="{height / 2:.1f}" font-family="sans-serif" font-size="11" '
        f'text-anchor="middle" transform="rotate(-90 20 {height / 2:.1f})">'
        "Geometric mean MSE ratio to mode-matched e32</text>",
        _svg_text(width / 2, height - 14, "Matched D01--D10 ratios; lower is better", size=10, anchor="middle"),
        "</svg>\n",
    ])
    return "\n".join(svg)


TRAINING_EFFICIENCY_FIELDS = (
    "config_id", "direction_conditioning_mode", "epochs", "n_training_jobs",
    "total_training_hours", "mean_training_seconds", "median_training_seconds",
    "min_training_seconds", "max_training_seconds", "mean_final_val_loss",
    "median_final_val_loss", "mean_best_val_loss", "median_best_val_loss",
)


def analyze(
    metric_rows: Iterable[Mapping[str, Any]], training_rows: Sequence[Mapping[str, Any]],
    output_root: Path, *, strict: bool = True,
    source_bindings: Mapping[str, Mapping[str, str]] | None = None,
) -> dict[str, Any]:
    rows = normalize_metric_rows(metric_rows)
    normalized_training_rows = normalize_training_rows(training_rows)
    validate_rows(rows, strict=strict)
    topology = topology_analysis(rows)
    budget = training_budget_analysis(rows)
    efficiency = training_efficiency_analysis(normalized_training_rows, strict=strict)
    bound_sources: dict[str, Mapping[str, str]] = {
        "analysis_implementation": {
            "path": str(Path(__file__).resolve()),
            "sha256": _sha256_file(Path(__file__).resolve()),
        },
        "experiment_manifest": {
            "path": str((PACKAGE_ROOT / "manifest.json").resolve()),
            "sha256": _sha256_file(PACKAGE_ROOT / "manifest.json"),
        },
    }
    if source_bindings:
        bound_sources.update(source_bindings)

    output_root.mkdir(parents=True, exist_ok=True)
    paths = {
        "topology_effects_csv": output_root / "topology_effects.csv",
        "training_budget_effects_csv": output_root / "training_budget_effects.csv",
        "training_efficiency_csv": output_root / "training_efficiency.csv",
        "ablation_summary_json": output_root / "ablation_summary.json",
        "topology_table_tex": output_root / "topology_ablation_60spp.tex",
        "training_budget_table_tex": output_root / "training_budget_60spp.tex",
        "tables_tex": output_root / "ablation_tables.tex",
        "topology_plot_svg": output_root / "topology_curves.svg",
        "training_budget_plot_svg": output_root / "training_budget.svg",
    }
    _write_csv(paths["topology_effects_csv"], topology, PAIR_SUMMARY_FIELDS)
    _write_csv(
        paths["training_budget_effects_csv"], budget,
        (*PAIR_SUMMARY_FIELDS, "direction_conditioning_mode", "epochs"),
    )
    _write_csv(paths["training_efficiency_csv"], efficiency, TRAINING_EFFICIENCY_FIELDS)

    topology_tex = topology_latex(rows, topology)
    budget_tex = budget_latex(budget, efficiency)
    _atomic_write(paths["topology_table_tex"], topology_tex.encode())
    _atomic_write(paths["training_budget_table_tex"], budget_tex.encode())
    _atomic_write(paths["tables_tex"], (topology_tex + "\n" + budget_tex).encode())
    _atomic_write(paths["topology_plot_svg"], topology_svg(rows).encode())
    _atomic_write(paths["training_budget_plot_svg"], budget_svg(budget).encode())

    summary = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "status": "analysis_complete",
        "interpretation": INTERPRETATION,
        "configuration_note": CONFIGURATION_NOTE,
        "metric_row_count": len(rows),
        "training_metric_row_count": len(normalized_training_rows),
        "normalized_metric_rows_sha256": hashlib.sha256(
            json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "normalized_training_rows_sha256": hashlib.sha256(
            json.dumps(normalized_training_rows, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "source_bindings": bound_sources,
        "strict_grid_validation": strict,
        "topology_effects": topology,
        "training_budget_effects": budget,
        "training_efficiency": efficiency,
    }
    _write_json(paths["ablation_summary_json"], summary)
    outputs = {
        name: {"path": str(path), "sha256": _sha256_file(path)}
        for name, path in paths.items()
    }
    receipt = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "status": "analysis_complete",
        "created_at": _now_utc(),
        "strict_grid_validation": strict,
        "metric_row_count": len(rows),
        "training_metric_row_count": len(normalized_training_rows),
        "normalized_metric_rows_sha256": summary["normalized_metric_rows_sha256"],
        "normalized_training_rows_sha256": summary["normalized_training_rows_sha256"],
        "source_bindings": bound_sources,
        "outputs": outputs,
    }
    receipt_path = output_root / "analysis_receipt.json"
    _write_json(receipt_path, receipt)
    return {**receipt, "receipt_path": str(receipt_path), "receipt_sha256": _sha256_file(receipt_path)}


def cli_source_bindings(
    metric_path: Path, training_path: Path, *, strict: bool,
) -> dict[str, dict[str, str]]:
    bindings = {
        "collected_metric_csv": {
            "path": str(metric_path.resolve()),
            "sha256": _sha256_file(metric_path),
        },
    }
    if training_path.is_file():
        bindings["collected_training_csv"] = {
            "path": str(training_path.resolve()),
            "sha256": _sha256_file(training_path),
        }
    completion_path = metric_path.resolve().parent.parent / "completion_receipt.json"
    if not completion_path.is_file():
        if strict:
            raise AnalysisError(f"strict analysis requires collection receipt: {completion_path}")
        return bindings
    try:
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
        if completion.get("protocol_id") != PROTOCOL_ID or completion.get("status") != "complete":
            raise AnalysisError("collection receipt has the wrong protocol or status")
        expected_metric_hash = completion["metrics"]["csv"]["sha256"]
        expected_training_hash = completion["metrics"]["training_csv"]["sha256"]
        expected_manifest_hash = completion["manifest_sha256"]
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise AnalysisError(f"invalid collection receipt: {completion_path}") from exc
    if bindings["collected_metric_csv"]["sha256"] != expected_metric_hash:
        raise AnalysisError("metric CSV hash differs from the collection receipt")
    if (
        "collected_training_csv" not in bindings
        or bindings["collected_training_csv"]["sha256"] != expected_training_hash
    ):
        raise AnalysisError("training CSV hash differs from the collection receipt")
    manifest_hash = _sha256_file(PACKAGE_ROOT / "manifest.json")
    if manifest_hash != expected_manifest_hash:
        raise AnalysisError("experiment manifest hash differs from the collection receipt")
    bindings["collection_receipt"] = {
        "path": str(completion_path),
        "sha256": _sha256_file(completion_path),
    }
    return bindings


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=Path, default=DEFAULT_ROWS, help="collected ablation_rows.csv")
    parser.add_argument(
        "--training-rows", type=Path, default=DEFAULT_TRAINING_ROWS,
        help="collected training_rows.csv (optional only with --allow-partial)",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--allow-partial", action="store_true",
        help="permit a reduced scene/run grid for development; never use for the final report",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    rows = load_metric_rows(args.rows)
    training_rows = load_training_rows(args.training_rows)
    if not args.allow_partial and not training_rows:
        raise AnalysisError(f"strict analysis requires training metrics: {args.training_rows}")
    bindings = cli_source_bindings(
        args.rows, args.training_rows, strict=not args.allow_partial,
    )
    receipt = analyze(
        rows, training_rows, args.output, strict=not args.allow_partial,
        source_bindings=bindings,
    )
    print(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AnalysisError as exc:
        print(json.dumps({
            "protocol_id": PROTOCOL_ID,
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
        }, indent=2, sort_keys=True), file=os.sys.stderr)
        raise SystemExit(1)
