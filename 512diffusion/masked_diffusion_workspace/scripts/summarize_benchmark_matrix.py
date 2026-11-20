#!/usr/bin/env python3
"""Aggregate Phase-13 runs with paired, scale-free MSE comparisons."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import Any


class BenchmarkSummaryError(RuntimeError):
    """Raised when Phase-13 summaries cannot be compared safely."""


@dataclass(frozen=True, slots=True)
class PairedRecord:
    """One candidate and its baseline from the same render run."""

    run: str
    scene_id: str
    sampler_seed: int | None
    sample_count: int
    method: str
    baseline: str
    mse: float
    baseline_mse: float
    seconds: float

    @property
    def mse_ratio(self) -> float:
        return self.mse / self.baseline_mse


def _finite_positive(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result) or result <= 0.0:
        return None
    return result


def load_paired_records(
    summary_paths: list[Path],
    *,
    methods: tuple[str, ...],
    baseline: str,
) -> list[PairedRecord]:
    """Pair methods against a baseline only within the same Phase-13 run."""

    pairs: list[PairedRecord] = []
    for summary_path in summary_paths:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        records = [record for record in payload.get("records", []) if record.get("ok")]
        by_scene_method = {
            (str(record.get("scene_id")), str(record.get("method"))): record
            for record in records
        }
        scene_ids = sorted({str(record.get("scene_id")) for record in records})
        for scene_id in scene_ids:
            baseline_record = by_scene_method.get((scene_id, baseline))
            if baseline_record is None:
                continue
            baseline_mse = _finite_positive(baseline_record.get("mse"))
            if baseline_mse is None:
                continue
            baseline_spp = int(baseline_record.get("sample_count", 0))
            for method in methods:
                record = by_scene_method.get((scene_id, method))
                if record is None:
                    continue
                mse = _finite_positive(record.get("mse"))
                seconds = _finite_positive(record.get("seconds"))
                if mse is None or seconds is None:
                    continue
                sample_count = int(record.get("sample_count", 0))
                if sample_count != baseline_spp:
                    raise BenchmarkSummaryError(
                        f"SPP mismatch in {summary_path}: {scene_id} {method}="
                        f"{sample_count}, {baseline}={baseline_spp}"
                    )
                pairs.append(
                    PairedRecord(
                        run=str(summary_path),
                        scene_id=scene_id,
                        sampler_seed=(
                            None
                            if record.get("sampler_seed") is None
                            else int(record["sampler_seed"])
                        ),
                        sample_count=sample_count,
                        method=method,
                        baseline=baseline,
                        mse=mse,
                        baseline_mse=baseline_mse,
                        seconds=seconds,
                    )
                )
    return pairs


def aggregate_pairs(pairs: list[PairedRecord]) -> dict[str, dict[str, float | int]]:
    """Compute paired ratios so bright scenes do not dominate the aggregate."""

    by_method: dict[str, list[PairedRecord]] = {}
    for pair in pairs:
        by_method.setdefault(pair.method, []).append(pair)

    aggregates: dict[str, dict[str, float | int]] = {}
    for method, method_pairs in sorted(by_method.items()):
        ratios = [pair.mse_ratio for pair in method_pairs]
        aggregates[method] = {
            "paired_count": len(method_pairs),
            "win_count": sum(ratio < 1.0 for ratio in ratios),
            "win_rate": fmean(ratio < 1.0 for ratio in ratios),
            "geometric_mean_mse_ratio": math.exp(fmean(math.log(ratio) for ratio in ratios)),
            "arithmetic_mean_mse_ratio": fmean(ratios),
            "mean_render_seconds": fmean(pair.seconds for pair in method_pairs),
        }
    return aggregates


def render_markdown(
    *,
    baseline: str,
    pairs: list[PairedRecord],
    aggregates: dict[str, dict[str, float | int]],
) -> str:
    """Render a compact summary table."""

    lines = [
        "# Paired Benchmark Matrix",
        "",
        f"Baseline: `{baseline}`. Ratios are paired within one scene/run at equal SPP.",
        "",
        "| method | pairs | wins | win rate | geometric mean MSE ratio | mean render s |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for method, item in aggregates.items():
        lines.append(
            f"| {method} | {item['paired_count']} | {item['win_count']} | "
            f"{float(item['win_rate']):.3f} | "
            f"{float(item['geometric_mean_mse_ratio']):.6f} | "
            f"{float(item['mean_render_seconds']):.3f} |"
        )
    lines.extend(
        [
            "",
            "## Paired observations",
            "",
            "| scene | seed | spp | method | MSE | baseline MSE | ratio |",
            "| --- | ---: | ---: | --- | ---: | ---: | ---: |",
        ]
    )
    for pair in pairs:
        seed = "n/a" if pair.sampler_seed is None else str(pair.sampler_seed)
        lines.append(
            f"| {pair.scene_id} | {seed} | {pair.sample_count} | {pair.method} | "
            f"{pair.mse:.9g} | {pair.baseline_mse:.9g} | {pair.mse_ratio:.6f} |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summaries", nargs="+", type=Path)
    parser.add_argument("--baseline", default="simple_mlp")
    parser.add_argument("--methods", nargs="+", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()

    pairs = load_paired_records(
        args.summaries,
        methods=tuple(args.methods),
        baseline=args.baseline,
    )
    if not pairs:
        raise BenchmarkSummaryError("no valid paired records found")
    aggregates = aggregate_pairs(pairs)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(
            {
                "baseline": args.baseline,
                "summaries": [str(path) for path in args.summaries],
                "aggregates": aggregates,
                "pairs": [
                    {
                        "run": pair.run,
                        "scene_id": pair.scene_id,
                        "sampler_seed": pair.sampler_seed,
                        "sample_count": pair.sample_count,
                        "method": pair.method,
                        "baseline": pair.baseline,
                        "mse": pair.mse,
                        "baseline_mse": pair.baseline_mse,
                        "mse_ratio": pair.mse_ratio,
                        "seconds": pair.seconds,
                    }
                    for pair in pairs
                ],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    args.output_md.write_text(
        render_markdown(baseline=args.baseline, pairs=pairs, aggregates=aggregates),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
