#!/usr/bin/env python3
"""Build preregistered fresh-scene summaries after the global metric unlock."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = WORKSPACE_ROOT.parent
EXPECTED_PROTOCOL = "fresh_confirmatory_evaluation_20260901_v1"
INDEX_PROTOCOL = "fresh_confirmatory_metric_index_20260901_v1"
EXPECTED_SCENES = ("plane", "stone", "sphere", "pool", "living")
TRAIN_SEEDS = (104729, 130363, 155921)
RENDER_SEEDS = (1729, 31415, 271828, 424242, 8675309)
REGIMES = ("equal_spp", "equal_time")
METHODS = ("ordered", "simple", "mpg")
MSE_FLOOR = 1e-20


class FreshSummaryError(RuntimeError):
    """Raised for a barrier, provenance, or aggregation contract violation."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize all fresh-scene metrics after the global barrier."
    )
    parser.add_argument("--unlock-receipt", type=Path, required=True)
    parser.add_argument("--metric-index", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    parser.add_argument(
        "--unlock-metrics",
        action="store_true",
        help="Required before this program opens any metric-bearing file.",
    )
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260901)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FreshSummaryError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise FreshSummaryError(f"expected a JSON object: {path}")
    return payload


def resolve_artifact(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        resolved = path.resolve()
    else:
        workspace_candidate = (WORKSPACE_ROOT / path).resolve()
        project_candidate = (PROJECT_ROOT / path).resolve()
        resolved = workspace_candidate if workspace_candidate.is_file() else project_candidate
    try:
        resolved.relative_to(PROJECT_ROOT.resolve())
    except ValueError as exc:
        raise FreshSummaryError(f"artifact escapes project root: {value}") from exc
    return resolved


def validate_finalized_summary(row: dict[str, Any], label: object) -> dict[str, Any]:
    summary_path = resolve_artifact(str(row.get("finalized_summary_path", "")))
    expected_sha = row.get("finalized_summary_sha256")
    if not summary_path.is_file() or sha256_file(summary_path) != expected_sha:
        raise FreshSummaryError(f"finalized summary hash mismatch for {label}")
    summary = load_json(summary_path)
    if summary.get("metrics_status") != "computed_after_deferred_unlock":
        raise FreshSummaryError(f"finalized summary status mismatch for {label}")
    records = summary.get("records")
    if not isinstance(records, list) or len(records) != 1:
        raise FreshSummaryError(f"finalized summary row count mismatch for {label}")
    record = records[0]
    for index_key, record_key in (
        ("mse", "mse"),
        ("psnr", "psnr"),
        ("sample_count", "sample_count"),
        ("render_seed", "sampler_seed"),
    ):
        if row.get(index_key) != record.get(record_key):
            raise FreshSummaryError(
                f"metric index differs from finalized summary for {label}: {index_key}"
            )
    output_path = resolve_artifact(str(record.get("output_path", "")))
    if not output_path.is_file() or sha256_file(output_path) != record.get(
        "output_sha256"
    ):
        raise FreshSummaryError(f"candidate EXR hash drift for {label}")
    source = summary.get("deferred_source_summary")
    if not isinstance(source, dict):
        raise FreshSummaryError(f"finalized summary lacks deferred source for {label}")
    source_path = resolve_artifact(str(source.get("path", "")))
    if not source_path.is_file() or sha256_file(source_path) != source.get("sha256"):
        raise FreshSummaryError(f"deferred summary hash drift for {label}")
    provenance = summary.get("reference_provenance")
    if not isinstance(provenance, dict):
        raise FreshSummaryError(f"finalized reference provenance missing for {label}")
    manifest_path = resolve_artifact(str(provenance.get("manifest_path", "")))
    if not manifest_path.is_file() or sha256_file(manifest_path) != provenance.get(
        "manifest_sha256"
    ):
        raise FreshSummaryError(f"shared-reference manifest hash drift for {label}")
    return summary


def validate_unlock_receipt(path: Path) -> tuple[dict[str, Any], str]:
    receipt = load_json(path.resolve())
    if receipt.get("schema_version") != 1 or receipt.get("protocol_id") != EXPECTED_PROTOCOL:
        raise FreshSummaryError("metric unlock receipt protocol mismatch")
    if receipt.get("status") != "metrics_unlocked":
        raise FreshSummaryError("metric unlock receipt is not unlocked")
    if receipt.get("candidate_count") != 350:
        raise FreshSummaryError("metric unlock receipt does not cover 350 candidates")
    if receipt.get("quality_fields_verified_null") is not True:
        raise FreshSummaryError("pre-unlock null-quality check is missing")
    candidates = receipt.get("candidate_output_sha256")
    if not isinstance(candidates, dict) or len(candidates) != 350:
        raise FreshSummaryError("unlock receipt lacks all candidate output hashes")
    noise = receipt.get("reference_noise_output_sha256")
    if not isinstance(noise, dict) or len(noise) != 5:
        raise FreshSummaryError("unlock receipt lacks five reference-noise outputs")
    for key in ("shared_reference_manifest_sha256", "equal_time_spp_map_sha256"):
        value = receipt.get(key)
        if not isinstance(value, str) or len(value) != 64:
            raise FreshSummaryError(f"unlock receipt lacks {key}")
    return receipt, sha256_file(path.resolve())


def _expected_keys() -> set[tuple[str, str, str, int | None, int]]:
    keys: set[tuple[str, str, str, int | None, int]] = set()
    for regime in REGIMES:
        for scene in EXPECTED_SCENES:
            for method in ("ordered", "simple"):
                for train_seed in TRAIN_SEEDS:
                    for render_seed in RENDER_SEEDS:
                        keys.add((regime, scene, method, train_seed, render_seed))
            for render_seed in RENDER_SEEDS:
                keys.add((regime, scene, "mpg", None, render_seed))
    return keys


def validate_metric_index(
    *, path: Path, unlock_sha: str, unlock_receipt: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    index = load_json(path.resolve())
    if index.get("schema_version") != 1 or index.get("protocol_id") != INDEX_PROTOCOL:
        raise FreshSummaryError("metric index protocol mismatch")
    if index.get("unlock_receipt_sha256") != unlock_sha:
        raise FreshSummaryError("metric index does not bind the unlock receipt")
    rows = index.get("records")
    noise = index.get("reference_noise_records")
    if not isinstance(rows, list) or len(rows) != 350:
        raise FreshSummaryError("metric index must contain 350 candidate rows")
    if not isinstance(noise, list) or len(noise) != 5:
        raise FreshSummaryError("metric index must contain five noise-check rows")
    expected = _expected_keys()
    unlock_candidates = unlock_receipt["candidate_output_sha256"]
    expected_node_ids = {
        (
            f"candidate:{regime}:{scene}:{method}:train{train_seed}:render{render_seed}"
            if method != "mpg"
            else f"candidate:{regime}:{scene}:mpg:render{render_seed}"
        )
        for regime, scene, method, train_seed, render_seed in expected
    }
    if set(unlock_candidates) != expected_node_ids:
        raise FreshSummaryError("unlock receipt candidate node set changed")
    observed: set[tuple[str, str, str, int | None, int]] = set()
    for row in rows:
        if not isinstance(row, dict):
            raise FreshSummaryError("candidate metric row is not an object")
        train_seed = row.get("train_seed")
        key = (
            str(row.get("regime")),
            str(row.get("scene")),
            str(row.get("method")),
            None if train_seed is None else int(train_seed),
            int(row.get("render_seed", -1)),
        )
        if key in observed:
            raise FreshSummaryError(f"duplicate candidate metric key: {key}")
        observed.add(key)
        mse = float(row.get("mse", math.nan))
        psnr = float(row.get("psnr", math.nan))
        seconds = float(row.get("seconds", math.nan))
        spp = int(row.get("sample_count", 0))
        if not math.isfinite(mse) or mse < 0.0:
            raise FreshSummaryError(f"invalid MSE for {key}: {mse}")
        if not math.isfinite(psnr) and not math.isinf(psnr):
            raise FreshSummaryError(f"invalid PSNR for {key}: {psnr}")
        if not math.isfinite(seconds) or seconds <= 0.0 or spp <= 0:
            raise FreshSummaryError(f"invalid timing/SPP for {key}")
        finalized = validate_finalized_summary(row, key)
        node_id = (
            f"candidate:{key[0]}:{key[1]}:{key[2]}:train{key[3]}:render{key[4]}"
            if key[2] != "mpg"
            else f"candidate:{key[0]}:{key[1]}:mpg:render{key[4]}"
        )
        deferred_path_text = str(row.get("deferred_summary_path", ""))
        deferred_sha = row.get("deferred_summary_sha256")
        if unlock_candidates[node_id].get(deferred_path_text) != deferred_sha:
            raise FreshSummaryError(
                f"deferred summary is not bound to unlock receipt: {node_id}"
            )
        finalized_record = finalized["records"][0]
        candidate_path_text = str(finalized_record["output_path"])
        candidate_path = resolve_artifact(candidate_path_text)
        try:
            workspace_candidate_path = candidate_path.relative_to(
                WORKSPACE_ROOT.resolve()
            ).as_posix()
        except ValueError:
            workspace_candidate_path = candidate_path_text
        if unlock_candidates[node_id].get(workspace_candidate_path) != finalized_record.get(
            "output_sha256"
        ):
            raise FreshSummaryError(
                f"candidate EXR is not bound to unlock receipt: {node_id}"
            )
        if finalized.get("reference_provenance", {}).get(
            "manifest_sha256"
        ) != unlock_receipt["shared_reference_manifest_sha256"]:
            raise FreshSummaryError(
                f"candidate reference is not bound to unlock receipt: {node_id}"
            )
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise FreshSummaryError(
            f"metric index key mismatch: missing={missing[:3]}, extra={extra[:3]}"
        )
    noise_scenes = {str(row.get("scene")) for row in noise}
    if noise_scenes != set(EXPECTED_SCENES):
        raise FreshSummaryError("reference-noise rows do not cover all five scenes")
    unlock_noise = unlock_receipt["reference_noise_output_sha256"]
    expected_noise_ids = {
        f"reference_noise:{scene}" for scene in EXPECTED_SCENES
    }
    if set(unlock_noise) != expected_noise_ids:
        raise FreshSummaryError("unlock receipt reference-noise node set changed")
    for row in noise:
        mse = float(row.get("mse", math.nan))
        if not math.isfinite(mse) or mse < 0.0:
            raise FreshSummaryError("invalid reference-noise MSE")
        scene = str(row.get("scene"))
        finalized = validate_finalized_summary(row, f"reference-noise/{scene}")
        node_id = f"reference_noise:{scene}"
        deferred_path_text = str(row.get("deferred_summary_path", ""))
        deferred_sha = row.get("deferred_summary_sha256")
        if unlock_noise[node_id].get(deferred_path_text) != deferred_sha:
            raise FreshSummaryError(
                f"noise deferred summary is not bound to unlock receipt: {node_id}"
            )
        finalized_record = finalized["records"][0]
        noise_output_path = resolve_artifact(str(finalized_record["output_path"]))
        try:
            noise_output_key = noise_output_path.relative_to(
                WORKSPACE_ROOT.resolve()
            ).as_posix()
        except ValueError:
            noise_output_key = str(finalized_record["output_path"])
        if unlock_noise[node_id].get(noise_output_key) != finalized_record.get(
            "output_sha256"
        ):
            raise FreshSummaryError(
                f"noise EXR is not bound to unlock receipt: {node_id}"
            )
        if finalized.get("reference_provenance", {}).get(
            "manifest_sha256"
        ) != unlock_receipt["shared_reference_manifest_sha256"]:
            raise FreshSummaryError(
                f"noise reference is not bound to unlock receipt: {node_id}"
            )
    accounting = index.get("training_accounting")
    if not isinstance(accounting, dict):
        raise FreshSummaryError("metric index lacks training/export accounting")
    if accounting.get("training_job_count") != 30 or accounting.get(
        "export_job_count"
    ) != 30:
        raise FreshSummaryError("training/export accounting cardinality changed")
    for key in ("training_seconds", "export_seconds"):
        value = float(accounting.get(key, math.nan))
        if not math.isfinite(value) or value < 0.0:
            raise FreshSummaryError(f"invalid {key}")
    accounting_jobs = accounting.get("jobs")
    if not isinstance(accounting_jobs, list) or len(accounting_jobs) != 60:
        raise FreshSummaryError("training accounting must include all 60 jobs")
    training_jobs = [
        row for row in accounting_jobs if "training" in str(row.get("kind"))
    ]
    for row in training_jobs:
        if int(row.get("epochs", -1)) != 32:
            raise FreshSummaryError("training accounting epoch count changed")
        if int(row.get("examples_seen", -1)) <= 0 or int(
            row.get("optimizer_steps", -1)
        ) <= 0:
            raise FreshSummaryError("training accounting examples/steps are invalid")
        if row.get("peak_memory_bytes") is not None or row.get(
            "peak_memory_status"
        ) != "unavailable_not_recorded_by_frozen_runner":
            raise FreshSummaryError("peak-memory limitation is not explicit")
        if row.get("amortized_confirmatory_images") != 10:
            raise FreshSummaryError("training amortization count changed")
    limitations = index.get("protocol_limitations")
    if not isinstance(limitations, list) or len(limitations) != 1:
        raise FreshSummaryError(
            "metric index must disclose the training pairing limitation"
        )
    if "minibatch permutations are not guaranteed" not in str(limitations[0]):
        raise FreshSummaryError("training pairing limitation was weakened")
    evaluation = index.get("evaluation_accounting")
    if not isinstance(evaluation, dict) or not isinstance(
        evaluation.get("process_wall_seconds_by_stage"), dict
    ):
        raise FreshSummaryError("metric index lacks evaluation accounting")
    for stage, value in evaluation["process_wall_seconds_by_stage"].items():
        seconds = float(value)
        if not math.isfinite(seconds) or seconds < 0.0:
            raise FreshSummaryError(f"invalid evaluation seconds for {stage}")
    for key in (
        "conditional_timing_repeats_executed",
        "failed_attempts",
        "timeout_attempts",
        "failed_render_records",
        "timed_out_render_records",
    ):
        value = evaluation.get(key)
        if not isinstance(value, int) or value < 0:
            raise FreshSummaryError(f"invalid evaluation counter {key}")
    return rows, noise, index


def safe_log_mse(value: float) -> float:
    return math.log(max(float(value), MSE_FLOOR))


def geometric_mean(values: Iterable[float]) -> float:
    materialized = [max(float(value), MSE_FLOOR) for value in values]
    if not materialized:
        raise FreshSummaryError("geometric mean of empty values")
    return math.exp(sum(math.log(value) for value in materialized) / len(materialized))


def percentile(values: list[float], probability: float) -> float:
    if not values:
        raise FreshSummaryError("percentile of empty values")
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def index_rows(
    rows: list[dict[str, Any]], regime: str
) -> dict[tuple[str, str, int | None, int], dict[str, Any]]:
    return {
        (
            str(row["scene"]),
            str(row["method"]),
            None if row.get("train_seed") is None else int(row["train_seed"]),
            int(row["render_seed"]),
        ): row
        for row in rows
        if row["regime"] == regime
    }


def scene_log_ratios(
    table: dict[tuple[str, str, int | None, int], dict[str, Any]],
    *,
    scene: str,
    numerator: str,
    denominator: str,
) -> list[float]:
    values: list[float] = []
    if denominator == "simple":
        for train_seed in TRAIN_SEEDS:
            for render_seed in RENDER_SEEDS:
                numerator_mse = float(
                    table[(scene, numerator, train_seed, render_seed)]["mse"]
                )
                denominator_mse = float(
                    table[(scene, denominator, train_seed, render_seed)]["mse"]
                )
                values.append(safe_log_mse(numerator_mse) - safe_log_mse(denominator_mse))
        return values
    if denominator != "mpg":
        raise FreshSummaryError(f"unsupported comparator: {denominator}")
    for render_seed in RENDER_SEEDS:
        learned_logs = [
            safe_log_mse(float(table[(scene, numerator, seed, render_seed)]["mse"]))
            for seed in TRAIN_SEEDS
        ]
        mpg_log = safe_log_mse(
            float(table[(scene, "mpg", None, render_seed)]["mse"])
        )
        values.append(sum(learned_logs) / len(learned_logs) - mpg_log)
    return values


def hierarchical_bootstrap(
    table: dict[tuple[str, str, int | None, int], dict[str, Any]],
    *,
    numerator: str,
    denominator: str,
    replicates: int,
    rng: random.Random,
) -> tuple[float, float]:
    estimates: list[float] = []
    for _ in range(replicates):
        sampled_scene_logs: list[float] = []
        for _scene_slot in EXPECTED_SCENES:
            scene = rng.choice(EXPECTED_SCENES)
            render_draws = [rng.choice(RENDER_SEEDS) for _ in RENDER_SEEDS]
            pair_logs: list[float] = []
            if denominator == "simple":
                train_draws = [rng.choice(TRAIN_SEEDS) for _ in TRAIN_SEEDS]
                for train_seed in train_draws:
                    for render_seed in render_draws:
                        numerator_mse = float(
                            table[(scene, numerator, train_seed, render_seed)]["mse"]
                        )
                        denominator_mse = float(
                            table[(scene, denominator, train_seed, render_seed)]["mse"]
                        )
                        pair_logs.append(
                            safe_log_mse(numerator_mse)
                            - safe_log_mse(denominator_mse)
                        )
            else:
                train_draws = [rng.choice(TRAIN_SEEDS) for _ in TRAIN_SEEDS]
                for render_seed in render_draws:
                    learned_logs = [
                        safe_log_mse(
                            float(table[(scene, numerator, seed, render_seed)]["mse"])
                        )
                        for seed in train_draws
                    ]
                    mpg_log = safe_log_mse(
                        float(table[(scene, "mpg", None, render_seed)]["mse"])
                    )
                    pair_logs.append(sum(learned_logs) / len(learned_logs) - mpg_log)
            sampled_scene_logs.append(sum(pair_logs) / len(pair_logs))
        estimates.append(math.exp(sum(sampled_scene_logs) / len(sampled_scene_logs)))
    return percentile(estimates, 0.025), percentile(estimates, 0.975)


def summarize_comparison(
    table: dict[tuple[str, str, int | None, int], dict[str, Any]],
    *,
    numerator: str,
    denominator: str,
    replicates: int,
    rng: random.Random,
) -> dict[str, Any]:
    per_scene: dict[str, Any] = {}
    scene_ratios: list[float] = []
    scene_mean_logs: list[float] = []
    for scene in EXPECTED_SCENES:
        logs = scene_log_ratios(
            table,
            scene=scene,
            numerator=numerator,
            denominator=denominator,
        )
        mean_log_ratio = sum(logs) / len(logs)
        ratio = math.exp(mean_log_ratio)
        scene_ratios.append(ratio)
        scene_mean_logs.append(mean_log_ratio)
        per_scene[scene] = {
            "geometric_mse_ratio": ratio,
            "wins": sum(value < 0.0 for value in logs),
            "pairs": len(logs),
        }
    # MSE_FLOOR is applied once to each underlying MSE by safe_log_mse().
    # Reusing geometric_mean(scene_ratios) would incorrectly floor a legitimate
    # already-computed ratio below 1e-20 a second time.
    pooled = math.exp(sum(scene_mean_logs) / len(scene_mean_logs))
    lower, upper = hierarchical_bootstrap(
        table,
        numerator=numerator,
        denominator=denominator,
        replicates=replicates,
        rng=rng,
    )
    majority = sum(ratio < 1.0 for ratio in scene_ratios)
    return {
        "numerator": numerator,
        "denominator": denominator,
        "per_scene": per_scene,
        "pooled_equal_scene_geometric_mse_ratio": pooled,
        "hierarchical_bootstrap_95_percent": [lower, upper],
        "scene_wins": majority,
        "strong_claim_passed": pooled < 1.0 and majority >= 3,
    }


def build_summary(
    *,
    rows: list[dict[str, Any]],
    noise_rows: list[dict[str, Any]],
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    if bootstrap_replicates <= 0:
        raise FreshSummaryError("bootstrap replicates must be positive")
    regimes: dict[str, Any] = {}
    for regime_index, regime in enumerate(REGIMES):
        table = index_rows(rows, regime)
        comparisons: dict[str, Any] = {}
        for comparison_index, (numerator, denominator) in enumerate(
            (("ordered", "simple"), ("ordered", "mpg"), ("simple", "mpg"))
        ):
            rng = random.Random(
                bootstrap_seed + regime_index * 1009 + comparison_index * 9176
            )
            comparisons[f"{numerator}_vs_{denominator}"] = summarize_comparison(
                table,
                numerator=numerator,
                denominator=denominator,
                replicates=bootstrap_replicates,
                rng=rng,
            )
        timing: dict[str, Any] = {}
        for method in METHODS:
            method_rows = [
                row
                for row in rows
                if row["regime"] == regime and row["method"] == method
            ]
            timing[method] = {
                "images": len(method_rows),
                "mean_seconds": sum(float(row["seconds"]) for row in method_rows)
                / len(method_rows),
                "min_spp": min(int(row["sample_count"]) for row in method_rows),
                "max_spp": max(int(row["sample_count"]) for row in method_rows),
                "individual_images_outside_108_132_seconds": (
                    sum(
                        not 108.0 <= float(row["seconds"]) <= 132.0
                        for row in method_rows
                    )
                    if regime == "equal_time"
                    else None
                ),
            }
        timing_cells: dict[str, Any] = {}
        for scene in EXPECTED_SCENES:
            for method in METHODS:
                cell_rows = [
                    row
                    for row in rows
                    if row["regime"] == regime
                    and row["scene"] == scene
                    and row["method"] == method
                ]
                expected_count = 15 if method in {"ordered", "simple"} else 5
                if len(cell_rows) != expected_count:
                    raise FreshSummaryError(
                        f"timing cell {regime}/{scene}/{method} has {len(cell_rows)} rows"
                    )
                sample_counts = {int(row["sample_count"]) for row in cell_rows}
                if len(sample_counts) != 1:
                    raise FreshSummaryError(
                        f"timing cell {regime}/{scene}/{method} used multiple SPP values"
                    )
                median_seconds = float(
                    statistics.median(float(row["seconds"]) for row in cell_rows)
                )
                timing_cells[f"{scene}|{method}"] = {
                    "images": len(cell_rows),
                    "sample_count": next(iter(sample_counts)),
                    "median_process_wall_seconds": median_seconds,
                    "outside_108_132_seconds": (
                        not 108.0 <= median_seconds <= 132.0
                        if regime == "equal_time"
                        else None
                    ),
                }
        regimes[regime] = {
            "comparisons": comparisons,
            "timing": timing,
            "timing_cells": timing_cells,
            "timing_cells_outside_108_132_seconds": (
                sum(
                    bool(cell["outside_108_132_seconds"])
                    for cell in timing_cells.values()
                )
                if regime == "equal_time"
                else None
            ),
        }
    return {
        "schema_version": 1,
        "protocol_id": EXPECTED_PROTOCOL,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "candidate_records": len(rows),
        "reference_noise_mse": {
            str(row["scene"]): float(row["mse"]) for row in noise_rows
        },
        "mse_floor": MSE_FLOOR,
        "bootstrap": {
            "replicates": bootstrap_replicates,
            "seed": bootstrap_seed,
            "resampling": "scenes, then paired training/render seeds within scenes",
        },
        "regimes": regimes,
        "raw_records": rows,
    }


def render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Fresh confirmatory evaluation",
        "",
        f"Candidate records: `{summary['candidate_records']}`",
        "",
    ]
    for regime in REGIMES:
        lines.extend([f"## {regime}", ""])
        lines.append(
            "| Comparison | Pooled geometric MSE ratio | 95% hierarchical CI | "
            "Scene wins | Strong claim |"
        )
        lines.append("|---|---:|---:|---:|---:|")
        comparisons = summary["regimes"][regime]["comparisons"]
        for name, row in comparisons.items():
            lower, upper = row["hierarchical_bootstrap_95_percent"]
            lines.append(
                f"| {name} | {row['pooled_equal_scene_geometric_mse_ratio']:.6g} | "
                f"[{lower:.6g}, {upper:.6g}] | {row['scene_wins']}/5 | "
                f"{row['strong_claim_passed']} |"
            )
        lines.extend(["", "### Per-scene ratios", ""])
        lines.append("| Comparison | Scene | Ratio | Wins |")
        lines.append("|---|---|---:|---:|")
        for name, row in comparisons.items():
            for scene in EXPECTED_SCENES:
                scene_row = row["per_scene"][scene]
                lines.append(
                    f"| {name} | {scene} | {scene_row['geometric_mse_ratio']:.6g} | "
                    f"{scene_row['wins']}/{scene_row['pairs']} |"
                )
        lines.append("")
        lines.extend(["### Scene/method median wall time", ""])
        lines.append("| Cell | SPP | Median seconds | Outside 108–132 s |")
        lines.append("|---|---:|---:|---:|")
        for cell_name, cell in summary["regimes"][regime]["timing_cells"].items():
            outside = cell["outside_108_132_seconds"]
            lines.append(
                f"| {cell_name} | {cell['sample_count']} | "
                f"{cell['median_process_wall_seconds']:.3f} | "
                f"{'n/a' if outside is None else outside} |"
            )
        lines.append("")
    lines.extend(["## Reference-noise check", ""])
    for scene in EXPECTED_SCENES:
        lines.append(f"- {scene}: `{summary['reference_noise_mse'][scene]:.8g}` MSE")
    accounting = summary.get("training_accounting")
    if isinstance(accounting, dict):
        lines.extend(
            [
                "",
                "## Offline accounting",
                "",
                f"- training jobs: `{accounting['training_job_count']}`; "
                f"seconds: `{accounting['training_seconds']:.3f}`",
                f"- export jobs: `{accounting['export_job_count']}`; "
                f"seconds: `{accounting['export_seconds']:.3f}`",
                f"- parameter counts: `{accounting['parameter_counts']}`",
                "- epochs per learned model: `32`; total examples seen: "
                f"`{sum(int(row['examples_seen']) for row in accounting['jobs'] if row['examples_seen'] is not None)}`",
                f"- peak memory: `{accounting['peak_memory_status']}`",
                f"- amortization: {accounting['amortization_policy']}",
            ]
        )
    evaluation = summary.get("evaluation_accounting")
    if isinstance(evaluation, dict):
        lines.extend(["", "## Evaluation wall time", ""])
        for stage, seconds in evaluation["process_wall_seconds_by_stage"].items():
            lines.append(f"- {stage}: `{seconds:.3f}` seconds")
        lines.append(
            "- conditional 16-spp repeats executed: "
            f"`{evaluation['conditional_timing_repeats_executed']}`"
        )
        lines.append(f"- failed attempts: `{evaluation['failed_attempts']}`")
        lines.append(f"- timeout attempts: `{evaluation['timeout_attempts']}`")
        lines.append(
            f"- failed Phase 13 render records: `{evaluation['failed_render_records']}`"
        )
        lines.append(
            "- timed-out Phase 13 render records: "
            f"`{evaluation['timed_out_render_records']}`"
        )
    lines.extend(["", "## Protocol limitation", ""])
    for limitation in summary.get("protocol_limitations", []):
        lines.append(f"- {limitation}")
    lines.extend(
        [
            "",
            "Strong claim requires pooled ratio < 1 and at least 3/5 scene ratios < 1 "
            "within the same regime.",
            "",
        ]
    )
    return "\n".join(lines)


def write_exclusive(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as handle:
            handle.write(text)
    except FileExistsError as exc:
        raise FreshSummaryError(f"refusing to overwrite final summary: {path}") from exc


def main() -> int:
    args = parse_args()
    if not args.unlock_metrics:
        raise FreshSummaryError(
            "refusing to open metric-bearing input without --unlock-metrics"
        )
    unlock_receipt, unlock_sha = validate_unlock_receipt(args.unlock_receipt)
    rows, noise_rows, index = validate_metric_index(
        path=args.metric_index,
        unlock_sha=unlock_sha,
        unlock_receipt=unlock_receipt,
    )
    summary = build_summary(
        rows=rows,
        noise_rows=noise_rows,
        bootstrap_replicates=args.bootstrap_replicates,
        bootstrap_seed=args.bootstrap_seed,
    )
    summary["unlock_receipt"] = {
        "path": str(args.unlock_receipt.resolve()),
        "sha256": unlock_sha,
    }
    summary["metric_index"] = {
        "path": str(args.metric_index.resolve()),
        "sha256": sha256_file(args.metric_index.resolve()),
        "source_finalized_summary_count": len(index["records"])
        + len(index["reference_noise_records"]),
    }
    summary["training_accounting"] = index["training_accounting"]
    summary["evaluation_accounting"] = index["evaluation_accounting"]
    summary["protocol_limitations"] = index["protocol_limitations"]
    write_exclusive(
        args.output_json.resolve(),
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
    )
    write_exclusive(args.output_markdown.resolve(), render_markdown(summary))
    print(args.output_json.resolve())
    print(args.output_markdown.resolve())
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except FreshSummaryError as exc:
        print(f"ERROR: {exc}")
        raise SystemExit(2)
