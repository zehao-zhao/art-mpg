#!/usr/bin/env python3
"""Analyze Diffusion-MLP heavy-tail evidence without modifying the renderer."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = WORKSPACE_ROOT.parent
SRC_ROOT = WORKSPACE_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from masked_diffusion_mpg.tail_diagnostics import (  # noqa: E402
    TailDiagnosticsError,
    analyze_image,
    analyze_jsonl,
    analyze_renderer_log,
    read_runtime_model_metadata,
)


def _parse_labelled_path(value: str) -> tuple[str, Path]:
    label, separator, path_text = value.partition("=")
    if not separator or not label.strip() or not path_text.strip():
        raise argparse.ArgumentTypeError("expected LABEL=PATH")
    return label.strip(), Path(path_text).expanduser()


def _resolve_artifact(path_text: str, *, summary_path: Path | None = None) -> Path:
    candidate = Path(path_text).expanduser()
    if candidate.is_absolute():
        return candidate
    candidates = [PROJECT_ROOT / candidate, WORKSPACE_ROOT / candidate]
    if summary_path is not None:
        candidates.insert(0, summary_path.parent / candidate)
    for resolved in candidates:
        if resolved.exists():
            return resolved.resolve()
    return candidates[0].resolve()


def _candidate_event_paths(log_path: Path) -> tuple[Path, ...]:
    base_name = log_path.name
    for suffix in (".stdout.txt", ".stderr.txt", ".txt"):
        if base_name.endswith(suffix):
            base_name = base_name[: -len(suffix)]
            break
    return (
        log_path.with_name(base_name + ".tail.jsonl"),
        log_path.with_name(base_name + ".events.jsonl"),
    )


def _metric_indicator(
    summary: dict[str, Any], *, label: str
) -> list[dict[str, Any]]:
    """Emit transparent heuristic flags, not hypothesis-test conclusions."""

    if not summary.get("available"):
        return []
    flags: list[dict[str, Any]] = []
    top_share = summary.get("top_tail_share", {})
    effective_fraction = summary.get("effective_sample_size_fraction")
    max_to_median = summary.get("max_to_median")
    top_one = top_share.get("1pct")
    if top_one is not None and top_one >= 0.25:
        flags.append(
            {
                "metric": label,
                "indicator": "top_1pct_mass_share_ge_25pct",
                "value": top_one,
            }
        )
    if effective_fraction is not None and effective_fraction <= 0.1:
        flags.append(
            {
                "metric": label,
                "indicator": "ess_fraction_le_10pct",
                "value": effective_fraction,
            }
        )
    if max_to_median is not None and max_to_median >= 100.0:
        flags.append(
            {
                "metric": label,
                "indicator": "max_to_median_ge_100",
                "value": max_to_median,
            }
        )
    return flags


def _collect_indicators(report: dict[str, Any]) -> list[dict[str, Any]]:
    indicators: list[dict[str, Any]] = []
    for index, record in enumerate(report.get("benchmark_records", [])):
        image = record.get("image", {})
        if image.get("available"):
            indicators.extend(
                {
                    **item,
                    "source": f"benchmark_records[{index}].image",
                    "scene_id": record.get("scene_id"),
                    "method": record.get("method"),
                }
                for item in _metric_indicator(
                    image.get("luminance", {}), label="pixel_luminance"
                )
            )
    for label, sample in report.get("sample_streams", {}).items():
        contribution = sample.get("metrics", {}).get(
            "contribution_magnitude", {}
        )
        indicators.extend(
            {
                **item,
                "source": f"sample_streams.{label}",
                "scene_id": label,
            }
            for item in _metric_indicator(
                contribution, label="sample_contribution_proxy"
            )
        )
    for label, event in report.get("event_streams", {}).items():
        for metric_name in ("inverse_probability", "contribution_magnitude"):
            indicators.extend(
                {
                    **item,
                    "source": f"event_streams.{label}",
                    "scene_id": label,
                }
                for item in _metric_indicator(
                    event.get("metrics", {}).get(metric_name, {}),
                    label=metric_name,
                )
            )
    return indicators


def analyze_benchmark_summary(summary_path: Path) -> list[dict[str, Any]]:
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    records: list[dict[str, Any]] = []
    for source_record in payload.get("records", []):
        record: dict[str, Any] = {
            "scene_id": source_record.get("scene_id"),
            "method": source_record.get("method"),
            "ok": bool(source_record.get("ok")),
            "sample_count": source_record.get("sample_count"),
            "sampler_seed": source_record.get("sampler_seed"),
            "seconds": source_record.get("seconds"),
            "active_config_sha256": source_record.get("active_config_sha256"),
        }

        stdout_path = _resolve_artifact(
            str(source_record.get("stdout_log", "")), summary_path=summary_path
        )
        if stdout_path.is_file():
            record["renderer_log"] = analyze_renderer_log(stdout_path)
        else:
            record["renderer_log"] = {
                "source_path": str(stdout_path),
                "available": False,
                "error": "stdout log missing",
            }

        output_path = _resolve_artifact(
            str(source_record.get("output_path", "")), summary_path=summary_path
        )
        record["image"] = (
            analyze_image(output_path)
            if output_path.is_file()
            else {
                "source_path": str(output_path),
                "available": False,
                "error": "render output missing",
            }
        )

        model_text = source_record.get("runtime_model_path")
        if model_text:
            model_path = _resolve_artifact(str(model_text), summary_path=summary_path)
            record["runtime_model"] = read_runtime_model_metadata(model_path)

        auto_events = next(
            (path for path in _candidate_event_paths(stdout_path) if path.is_file()),
            None,
        )
        if auto_events is not None:
            record["events"] = analyze_jsonl(auto_events)

        renderer_log = record["renderer_log"]
        counters = renderer_log.get("diagnostic_counters", {})
        event_report = record.get("events", {})
        event_metrics = event_report.get("metrics", {})
        record["coverage"] = {
            "proposal_or_inverse_probability": bool(
                event_metrics.get("proposal_probability", {}).get("available")
                or event_metrics.get("inverse_probability", {}).get("available")
            ),
            "retry_or_failure_counts": bool(counters)
            or bool(event_metrics.get("retry_count", {}).get("available")),
            "solver_failure": "newton_fail" in counters
            or bool(event_report.get("solver_outcomes", {}).get("available")),
            "solver_residual": bool(
                event_metrics.get("solver_residual", {}).get("available")
            ),
            "per_path_contribution": bool(
                event_metrics.get("contribution_magnitude", {}).get("available")
            ),
            "pixel_contribution_proxy": bool(record["image"].get("available")),
            "component_usage": bool(
                event_report.get("component_usage", {}).get("available")
            ),
        }
        records.append(record)
    return records


def build_report(
    *,
    benchmark_summaries: list[Path],
    renderer_logs: list[tuple[str, Path]],
    event_streams: list[tuple[str, Path]],
    sample_streams: list[tuple[str, Path]],
    images: list[tuple[str, Path]],
    runtime_models: list[tuple[str, Path]],
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Diffusion-MLP heavy-tail evidence",
        "benchmark_summaries": [str(path) for path in benchmark_summaries],
        "benchmark_records": [],
        "renderer_logs": {},
        "event_streams": {},
        "sample_streams": {},
        "images": {},
        "runtime_models": {},
        "limitations": [
            "Aggregate renderer counters cannot recover per-path distributions.",
            "Pixel luminance tails are an image-space proxy, not individual path contributions.",
            "Phase-09 replay weight/energy is a training-export proxy and may not equal the final estimator contribution.",
            "Heuristic tail indicators are screening signals, not proof of infinite variance or estimator bias.",
        ],
        "recommended_cpp_hook_points": [
            {
                "file": "mitsuba/src/integrators/MPG/manifold_path_guiding.cpp",
                "site": "after finalize_seed_inverse_probability(inv_prob_estimate)",
                "fields": ["inverse_probability", "trial_count", "bernoulli_failed"],
            },
            {
                "file": "mitsuba/src/integrators/MPG/manifold_path_guiding.cpp",
                "site": "before/after clamp_learned_proposal_contribution(manifold_contribution)",
                "fields": ["contribution", "contribution_after_clamp"],
            },
            {
                "file": "mitsuba/src/integrators/MPG/manifold_path_guiding.cpp",
                "site": "length and seed-direction proposal branches",
                "fields": ["proposal_component", "proposal_probability", "chain_length"],
            },
            {
                "file": "mitsuba/src/integrators/MPG/manifold_path_guiding.h",
                "site": "Manifold_Walk::newton_solver convergence loop",
                "fields": ["solver_residual", "solver_iterations", "solver_failed"],
            },
        ],
    }
    for summary_path in benchmark_summaries:
        report["benchmark_records"].extend(analyze_benchmark_summary(summary_path))
    for label, path in renderer_logs:
        report["renderer_logs"][label] = analyze_renderer_log(path)
    for label, path in event_streams:
        report["event_streams"][label] = analyze_jsonl(path)
    for label, path in sample_streams:
        report["sample_streams"][label] = analyze_jsonl(path)
    for label, path in images:
        report["images"][label] = analyze_image(path)
    for label, path in runtime_models:
        report["runtime_models"][label] = read_runtime_model_metadata(path)
    report["tail_indicators"] = _collect_indicators(report)
    return report


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Diffusion-MLP Heavy-Tail Diagnostics",
        "",
        "Tail flags below are deterministic screening heuristics, not statistical proof of bias or infinite variance.",
        "",
        "## Fresh benchmark records",
        "",
        "| scene | method | seed | spp | ok | Newton fail/seed | Bernoulli fail/seed | pixel top 1% share |",
        "| --- | --- | ---: | ---: | --- | ---: | ---: | ---: |",
    ]
    for record in report.get("benchmark_records", []):
        renderer = record.get("renderer_log", {})
        rates = renderer.get("derived_rates", {})
        image = record.get("image", {})
        top_one = image.get("luminance", {}).get("top_tail_share", {}).get("1pct")
        lines.append(
            "| {scene} | {method} | {seed} | {spp} | {ok} | {newton} | {bernoulli} | {tail} |".format(
                scene=record.get("scene_id", "n/a"),
                method=record.get("method", "n/a"),
                seed=record.get("sampler_seed", "n/a"),
                spp=record.get("sample_count", "n/a"),
                ok=record.get("ok", False),
                newton=_format_optional(rates.get("newton_fail_per_seed_started")),
                bernoulli=_format_optional(rates.get("bernoulli_fail_per_seed_started")),
                tail=_format_optional(top_one),
            )
        )

    lines.extend(
        [
            "",
            "## Per-sample contribution proxies",
            "",
            "| stream | samples | median | q99 | max | top 1% share | ESS fraction |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for label, sample in report.get("sample_streams", {}).items():
        metric = sample.get("metrics", {}).get("contribution_magnitude", {})
        quantiles = metric.get("quantiles", {})
        shares = metric.get("top_tail_share", {})
        lines.append(
            "| {label} | {count} | {median} | {q99} | {maximum} | {share} | {ess} |".format(
                label=label,
                count=metric.get("finite_count", 0),
                median=_format_optional(quantiles.get("50pct")),
                q99=_format_optional(quantiles.get("99pct")),
                maximum=_format_optional(metric.get("max")),
                share=_format_optional(shares.get("1pct")),
                ess=_format_optional(metric.get("effective_sample_size_fraction")),
            )
        )

    lines.extend(["", "## Coverage", ""])
    if report.get("benchmark_records"):
        for record in report["benchmark_records"]:
            available = [key for key, value in record.get("coverage", {}).items() if value]
            missing = [key for key, value in record.get("coverage", {}).items() if not value]
            lines.append(
                f"- `{record.get('scene_id')}/{record.get('method')}` available: "
                + (", ".join(available) if available else "none")
                + "; unavailable: "
                + (", ".join(missing) if missing else "none")
            )
    else:
        lines.append("- No benchmark summary supplied.")

    lines.extend(["", "## Tail indicators", ""])
    indicators = report.get("tail_indicators", [])
    if indicators:
        for item in indicators:
            lines.append(
                f"- `{item.get('scene_id', item.get('source'))}` "
                f"{item['metric']}: {item['indicator']} "
                f"({_format_optional(item['value'])})"
            )
    else:
        lines.append("- No configured heuristic threshold was crossed.")

    lines.extend(["", "## Interpretation limits", ""])
    lines.extend(f"- {item}" for item in report.get("limitations", []))
    return "\n".join(lines) + "\n"


def _format_optional(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-summary", action="append", type=Path, default=[])
    parser.add_argument("--renderer-log", action="append", type=_parse_labelled_path, default=[])
    parser.add_argument("--events", action="append", type=_parse_labelled_path, default=[])
    parser.add_argument("--sample-jsonl", action="append", type=_parse_labelled_path, default=[])
    parser.add_argument("--image", action="append", type=_parse_labelled_path, default=[])
    parser.add_argument("--runtime-model", action="append", type=_parse_labelled_path, default=[])
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-md", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = build_report(
        benchmark_summaries=[path.resolve() for path in args.benchmark_summary],
        renderer_logs=args.renderer_log,
        event_streams=args.events,
        sample_streams=args.sample_jsonl,
        images=args.image,
        runtime_models=args.runtime_model,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    if args.output_md is not None:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(render_markdown(report), encoding="utf-8")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except TailDiagnosticsError as exc:
        raise SystemExit(f"tail diagnostics failed: {exc}") from exc

