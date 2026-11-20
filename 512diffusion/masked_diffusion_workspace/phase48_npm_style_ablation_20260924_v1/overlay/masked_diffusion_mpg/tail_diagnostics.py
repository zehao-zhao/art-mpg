"""Heavy-tail diagnostics for renderer logs and per-path event streams.

The renderer currently exposes aggregate ``GuidedManifoldSampler`` counters.
This module deliberately consumes those counters without changing the sampling
code.  It can also consume optional JSONL records from a future/instrumented
renderer.  Missing observables stay explicitly unavailable; they are never
silently reported as zero.

Supported event keys (aliases are accepted) include:

* ``proposal_probability`` / ``proposal_pdf`` / ``proposal_prob``
* ``inverse_probability`` / ``inverse_pdf`` / ``inv_prob``
* ``retry_count`` / ``retries`` and ``trial_count`` / ``trials``
* ``solver_residual`` / ``newton_residual``
* ``solver_failed`` / ``solver_success``
* ``contribution`` / ``path_contribution`` / ``energy`` / ``weight``
* ``proposal_component`` / ``component`` / ``proposal_source``

The Phase-09 replay files use ``weight`` as a training/export energy proxy.
Reports retain the originating field so it cannot be mistaken for a literal
per-render Monte Carlo contribution.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence


DEFAULT_QUANTILES = (0.5, 0.9, 0.95, 0.99, 0.999)
DEFAULT_TAIL_FRACTIONS = (0.1, 0.01, 0.001)

_DIAGNOSTIC_RE = re.compile(
    r"GuidedManifoldSampler\s+(diag|stats)\s+([A-Za-z0-9_.-]+)\s+"
    r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
)
_ATTEMPT_RE = re.compile(r"^===== attempt\s+(\d+)\s+/", re.MULTILINE)
_METADATA_MARKER = "===== phase13 render metadata ====="

_ALIASES: dict[str, tuple[str, ...]] = {
    "proposal_probability": (
        "proposal_probability",
        "proposal_pdf",
        "proposal_prob",
    ),
    "inverse_probability": (
        "inverse_probability",
        "inverse_pdf",
        "inv_probability",
        "inv_prob",
        "inv_prob_estimate",
    ),
    "retry_count": ("retry_count", "retries"),
    "trial_count": ("trial_count", "trials", "iterations"),
    "solver_residual": (
        "solver_residual",
        "newton_residual",
        "constraint_residual",
    ),
    "contribution": (
        "contribution",
        "path_contribution",
        "radiance",
        "energy",
        "weight",
    ),
    "component": ("proposal_component", "component", "proposal_source"),
    "solver_failed": ("solver_failed", "newton_failed"),
    "solver_success": ("solver_success", "newton_success"),
    "sample_success": ("success", "sample_success"),
    "chain_length": ("chain_length", "bounce"),
    "chain_type": ("chain_type", "tau"),
}


class TailDiagnosticsError(RuntimeError):
    """Raised for malformed diagnostics inputs."""


@dataclass(slots=True)
class _EventAccumulator:
    event_count: int = 0
    invalid_json_lines: int = 0
    metrics: dict[str, list[float]] = field(default_factory=dict)
    source_fields: dict[str, Counter[str]] = field(default_factory=dict)
    components: Counter[str] = field(default_factory=Counter)
    chain_lengths: Counter[str] = field(default_factory=Counter)
    chain_types: Counter[str] = field(default_factory=Counter)
    solver_failure_count: int = 0
    solver_outcome_count: int = 0
    sample_failure_count: int = 0
    sample_outcome_count: int = 0
    probability_consistency_products: list[float] = field(default_factory=list)


def _float_or_none(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _bool_or_none(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "yes", "1", "success", "ok"}:
            return True
        if normalized in {"false", "no", "0", "failure", "failed"}:
            return False
    return None


def _flatten_mapping(
    value: Mapping[str, Any], prefix: str = ""
) -> dict[str, tuple[Any, str]]:
    """Flatten nested mappings while retaining the original dotted key."""

    flattened: dict[str, tuple[Any, str]] = {}
    for raw_key, item in value.items():
        key = str(raw_key)
        dotted = f"{prefix}.{key}" if prefix else key
        if isinstance(item, Mapping):
            flattened.update(_flatten_mapping(item, dotted))
        else:
            # The first leaf with a short name wins. Dotted aliases remain
            # addressable and avoid throwing away explicit top-level fields.
            flattened.setdefault(key, (item, dotted))
            flattened[dotted] = (item, dotted)
    return flattened


def _lookup(
    flattened: Mapping[str, tuple[Any, str]], aliases: Sequence[str]
) -> tuple[Any, str] | tuple[None, None]:
    for alias in aliases:
        if alias in flattened:
            return flattened[alias]
    return None, None


def _scalar_magnitude(value: Any) -> float | None:
    """Convert a scalar/RGB-like value to a conservative nonnegative magnitude."""

    scalar = _float_or_none(value)
    if scalar is not None:
        return abs(scalar)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        channels = [_float_or_none(item) for item in value]
        finite_channels = [
            abs(item) for item in channels if item is not None and math.isfinite(item)
        ]
        return max(finite_channels) if finite_channels else None
    if isinstance(value, Mapping):
        channels = [
            _float_or_none(value.get(key)) for key in ("r", "g", "b", "x", "y", "z")
        ]
        finite_channels = [
            abs(item) for item in channels if item is not None and math.isfinite(item)
        ]
        return max(finite_channels) if finite_channels else None
    return None


def quantile(values: Sequence[float], probability: float) -> float:
    """Return a deterministic, linearly interpolated sample quantile."""

    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be in [0, 1]")
    if not values:
        raise ValueError("quantile requires at least one value")
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    location = probability * (len(ordered) - 1)
    lower = int(math.floor(location))
    upper = int(math.ceil(location))
    if lower == upper:
        return ordered[lower]
    fraction = location - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def summarize_numeric(
    values: Iterable[float],
    *,
    mass_uses_absolute_value: bool = True,
    quantiles: Sequence[float] = DEFAULT_QUANTILES,
    tail_fractions: Sequence[float] = DEFAULT_TAIL_FRACTIONS,
) -> dict[str, Any]:
    """Summarize a numeric stream with tail concentration and ESS proxies."""

    all_values = [float(value) for value in values]
    finite = [value for value in all_values if math.isfinite(value)]
    nonfinite_count = len(all_values) - len(finite)
    result: dict[str, Any] = {
        "count": len(all_values),
        "finite_count": len(finite),
        "nonfinite_count": nonfinite_count,
        "negative_count": sum(value < 0.0 for value in finite),
        "zero_count": sum(value == 0.0 for value in finite),
        "available": bool(finite),
    }
    if not finite:
        result.update(
            {
                "min": None,
                "max": None,
                "mean": None,
                "quantiles": {},
                "top_tail_share": {},
                "effective_sample_size": None,
                "effective_sample_size_fraction": None,
                "max_to_median": None,
            }
        )
        return result

    result["min"] = min(finite)
    result["max"] = max(finite)
    # Divide before summing so a long stream of individually finite values does
    # not overflow merely while computing its mean.
    result["mean"] = math.fsum(value / len(finite) for value in finite)
    result["quantiles"] = {
        _probability_label(probability): quantile(finite, probability)
        for probability in quantiles
    }

    mass = [abs(value) if mass_uses_absolute_value else max(value, 0.0) for value in finite]
    ordered_mass = sorted(mass, reverse=True)
    scale = ordered_mass[0]
    normalized_mass = (
        [value / scale for value in ordered_mass] if scale > 0.0 else ordered_mass
    )
    total_mass = math.fsum(normalized_mass)
    tail_shares: dict[str, float | None] = {}
    for fraction in tail_fractions:
        if not 0.0 < fraction <= 1.0:
            raise ValueError("tail fractions must be in (0, 1]")
        take = max(1, int(math.ceil(len(ordered_mass) * fraction)))
        label = _probability_label(fraction)
        tail_shares[label] = (
            math.fsum(normalized_mass[:take]) / total_mass
            if total_mass > 0.0
            else None
        )
    result["top_tail_share"] = tail_shares

    square_mass = math.fsum(value * value for value in normalized_mass)
    effective_sample_size = (
        total_mass * total_mass / square_mass if square_mass > 0.0 else None
    )
    result["effective_sample_size"] = effective_sample_size
    result["effective_sample_size_fraction"] = (
        effective_sample_size / len(finite)
        if effective_sample_size is not None
        else None
    )
    median = quantile(mass, 0.5)
    result["max_to_median"] = max(mass) / median if median > 0.0 else None
    return result


def _probability_label(value: float) -> str:
    percentage = value * 100.0
    if percentage.is_integer():
        return f"{int(percentage)}pct"
    return f"{percentage:g}pct"


def _record_metric(
    accumulator: _EventAccumulator,
    metric: str,
    value: float | None,
    source_field: str | None,
) -> None:
    if value is None:
        return
    accumulator.metrics.setdefault(metric, []).append(value)
    if source_field is not None:
        accumulator.source_fields.setdefault(metric, Counter())[source_field] += 1


def accumulate_event(accumulator: _EventAccumulator, event: Mapping[str, Any]) -> None:
    """Accumulate one structured per-path/per-sample diagnostics event."""

    accumulator.event_count += 1
    flattened = _flatten_mapping(event)

    probability_raw, probability_field = _lookup(
        flattened, _ALIASES["proposal_probability"]
    )
    inverse_raw, inverse_field = _lookup(flattened, _ALIASES["inverse_probability"])
    probability = _float_or_none(probability_raw)
    inverse_probability = _float_or_none(inverse_raw)
    _record_metric(
        accumulator, "proposal_probability", probability, probability_field
    )
    if inverse_probability is None and probability is not None and probability > 0.0:
        inverse_probability = 1.0 / probability
        inverse_field = f"derived:1/{probability_field}"
    _record_metric(
        accumulator, "inverse_probability", inverse_probability, inverse_field
    )
    if (
        probability is not None
        and inverse_probability is not None
        and math.isfinite(probability)
        and math.isfinite(inverse_probability)
    ):
        accumulator.probability_consistency_products.append(
            probability * inverse_probability
        )

    for metric in ("retry_count", "trial_count", "solver_residual"):
        raw_value, source_field = _lookup(flattened, _ALIASES[metric])
        _record_metric(
            accumulator, metric, _float_or_none(raw_value), source_field
        )

    contribution_raw, contribution_field = _lookup(
        flattened, _ALIASES["contribution"]
    )
    _record_metric(
        accumulator,
        "contribution_magnitude",
        _scalar_magnitude(contribution_raw),
        contribution_field,
    )

    component, _ = _lookup(flattened, _ALIASES["component"])
    if component is not None:
        accumulator.components[str(component)] += 1

    for canonical, counter in (
        ("chain_length", accumulator.chain_lengths),
        ("chain_type", accumulator.chain_types),
    ):
        raw_value, _ = _lookup(flattened, _ALIASES[canonical])
        if raw_value is not None:
            counter[str(raw_value)] += 1

    solver_failed_raw, _ = _lookup(flattened, _ALIASES["solver_failed"])
    solver_success_raw, _ = _lookup(flattened, _ALIASES["solver_success"])
    solver_failed = _bool_or_none(solver_failed_raw)
    solver_success = _bool_or_none(solver_success_raw)
    if solver_failed is not None:
        accumulator.solver_outcome_count += 1
        accumulator.solver_failure_count += int(solver_failed)
    elif solver_success is not None:
        accumulator.solver_outcome_count += 1
        accumulator.solver_failure_count += int(not solver_success)

    sample_success_raw, _ = _lookup(flattened, _ALIASES["sample_success"])
    sample_success = _bool_or_none(sample_success_raw)
    if sample_success is not None:
        accumulator.sample_outcome_count += 1
        accumulator.sample_failure_count += int(not sample_success)


def _counter_payload(counter: Counter[str]) -> dict[str, Any]:
    total = sum(counter.values())
    return {
        "available": total > 0,
        "total": total,
        "counts": dict(sorted(counter.items())),
        "fractions": {
            key: count / total for key, count in sorted(counter.items())
        }
        if total
        else {},
    }


def finalize_events(accumulator: _EventAccumulator) -> dict[str, Any]:
    """Turn an event accumulator into a JSON-serializable report."""

    metric_names = (
        "proposal_probability",
        "inverse_probability",
        "retry_count",
        "trial_count",
        "solver_residual",
        "contribution_magnitude",
    )
    metrics = {
        name: summarize_numeric(accumulator.metrics.get(name, ()))
        for name in metric_names
    }
    for name, summary in metrics.items():
        summary["source_fields"] = dict(
            sorted(accumulator.source_fields.get(name, Counter()).items())
        )

    consistency = summarize_numeric(accumulator.probability_consistency_products)
    consistency["interpretation"] = (
        "proposal_probability * inverse_probability; values near 1 indicate "
        "internally consistent paired fields"
    )
    return {
        "event_count": accumulator.event_count,
        "invalid_json_lines": accumulator.invalid_json_lines,
        "metrics": metrics,
        "solver_outcomes": {
            "available": accumulator.solver_outcome_count > 0,
            "count": accumulator.solver_outcome_count,
            "failure_count": accumulator.solver_failure_count,
            "failure_rate": (
                accumulator.solver_failure_count / accumulator.solver_outcome_count
                if accumulator.solver_outcome_count
                else None
            ),
        },
        "sample_outcomes": {
            "available": accumulator.sample_outcome_count > 0,
            "count": accumulator.sample_outcome_count,
            "failure_count": accumulator.sample_failure_count,
            "failure_rate": (
                accumulator.sample_failure_count / accumulator.sample_outcome_count
                if accumulator.sample_outcome_count
                else None
            ),
        },
        "component_usage": _counter_payload(accumulator.components),
        "chain_length_usage": _counter_payload(accumulator.chain_lengths),
        "chain_type_usage": _counter_payload(accumulator.chain_types),
        "proposal_inverse_consistency": consistency,
    }


def analyze_events(events: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Analyze an in-memory event iterable."""

    accumulator = _EventAccumulator()
    for event in events:
        if not isinstance(event, Mapping):
            raise TailDiagnosticsError("each diagnostics event must be an object")
        accumulate_event(accumulator, event)
    return finalize_events(accumulator)


def iter_jsonl(path: Path) -> Iterator[Mapping[str, Any]]:
    """Yield JSON objects from JSONL, reporting the exact malformed line."""

    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise TailDiagnosticsError(
                    f"invalid JSON in {path}:{line_number}: {exc.msg}"
                ) from exc
            if not isinstance(payload, Mapping):
                raise TailDiagnosticsError(
                    f"expected JSON object in {path}:{line_number}"
                )
            yield payload


def analyze_jsonl(path: Path) -> dict[str, Any]:
    """Analyze a per-path event or Phase-09 replay JSONL file."""

    report = analyze_events(iter_jsonl(path))
    report["source_path"] = str(path)
    return report


def parse_phase13_metadata(text: str) -> dict[str, Any] | None:
    """Parse the JSON provenance header written by the Phase-13 runner."""

    marker_index = text.find(_METADATA_MARKER)
    if marker_index < 0:
        return None
    after_marker = text[marker_index + len(_METADATA_MARKER) :].lstrip("\r\n")
    first_line = after_marker.splitlines()[0] if after_marker else ""
    try:
        payload = json.loads(first_line)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def parse_renderer_log(text: str) -> dict[str, Any]:
    """Parse aggregate diagnostics already emitted by the MPG renderer."""

    diagnostics: Counter[str] = Counter()
    statistics: dict[str, float] = {}
    for match in _DIAGNOSTIC_RE.finditer(text):
        category, name, raw_value = match.groups()
        value = float(raw_value)
        if category == "diag":
            diagnostics[name] += int(value)
        else:
            statistics[name] = statistics.get(name, 0.0) + value

    attempts = [int(value) for value in _ATTEMPT_RE.findall(text)]
    attempt_count = max(attempts, default=0)
    seed_started = diagnostics.get("seed_started", 0)
    emitter_outcomes = diagnostics.get("emitter_success", 0) + diagnostics.get(
        "emitter_fail", 0
    )
    ordered_cached = diagnostics.get("ordered_descriptor_cached", 0)
    derived_rates: dict[str, float | None] = {
        "newton_fail_per_seed_started": _safe_ratio(
            diagnostics.get("newton_fail", 0), seed_started
        ),
        "bernoulli_fail_per_seed_started": _safe_ratio(
            diagnostics.get("bernoulli_fail", 0), seed_started
        ),
        "emitter_fail_fraction": _safe_ratio(
            diagnostics.get("emitter_fail", 0), emitter_outcomes
        ),
        "stored_per_seed_started": _safe_ratio(
            diagnostics.get("stored", 0), seed_started
        ),
        "ordered_descriptor_uses_per_cached": _safe_ratio(
            diagnostics.get("ordered_descriptor_uses", 0), ordered_cached
        ),
    }
    unavailable = []
    if not diagnostics:
        unavailable.append("aggregate_failure_counters")
    # These observables do not exist in the current aggregate stdout format.
    unavailable.extend(
        [
            "per_path_proposal_probability",
            "per_path_inverse_probability",
            "per_path_retry_count",
            "solver_residual",
            "per_path_contribution",
            "proposal_component_usage",
        ]
    )
    return {
        "metadata": parse_phase13_metadata(text),
        "attempt_count": attempt_count,
        "outer_retry_count": max(0, attempt_count - 1),
        "diagnostic_counters": dict(sorted(diagnostics.items())),
        "timing_seconds": dict(sorted(statistics.items())),
        "derived_rates": derived_rates,
        "unavailable_observables": unavailable,
    }


def _safe_ratio(numerator: int | float, denominator: int | float) -> float | None:
    return float(numerator) / float(denominator) if denominator else None


def analyze_renderer_log(path: Path) -> dict[str, Any]:
    """Analyze one renderer stdout/stderr transcript."""

    report = parse_renderer_log(path.read_text(encoding="utf-8", errors="replace"))
    report["source_path"] = str(path)
    return report


def summarize_rgb_pixels(pixels: Iterable[Sequence[float]]) -> dict[str, Any]:
    """Summarize RGB pixels using Rec.709 luminance and channel-peak tails."""

    luminance: list[float] = []
    channel_peak: list[float] = []
    invalid_pixel_count = 0
    for pixel in pixels:
        if len(pixel) < 3:
            invalid_pixel_count += 1
            continue
        rgb = tuple(_float_or_none(pixel[index]) for index in range(3))
        if any(value is None or not math.isfinite(value) for value in rgb):
            invalid_pixel_count += 1
            continue
        red, green, blue = (float(value) for value in rgb)
        luminance.append(0.2126 * red + 0.7152 * green + 0.0722 * blue)
        channel_peak.append(max(abs(red), abs(green), abs(blue)))
    return {
        "pixel_count": len(luminance) + invalid_pixel_count,
        "valid_pixel_count": len(luminance),
        "invalid_pixel_count": invalid_pixel_count,
        "luminance": summarize_numeric(luminance),
        "channel_peak": summarize_numeric(channel_peak),
    }


def analyze_image(path: Path) -> dict[str, Any]:
    """Load EXR/HDR imagery through OpenCV and report pixel-tail evidence."""

    try:
        image = load_rgb_image(path)
    except TailDiagnosticsError as exc:
        return {
            "source_path": str(path),
            "available": False,
            "error": str(exc),
        }
    report = summarize_rgb_pixels(image.reshape(-1, 3).tolist())
    report.update(
        {
            "source_path": str(path),
            "available": True,
            "height": int(image.shape[0]),
            "width": int(image.shape[1]),
            "interpretation": (
                "pixel luminance is an aggregate firefly indicator, not a "
                "substitute for per-path contribution instrumentation"
            ),
        }
    )
    return report


def load_rgb_image(path: Path) -> Any:
    """Load an image into an ``HxWx3`` RGB float64 array.

    NumPy/OpenCV stay optional so log and JSONL diagnostics require only the
    Python standard library.
    """

    # OpenCV gates its bundled OpenEXR codec behind this explicit opt-in. It
    # must be set before importing cv2 on builds that inspect it at load time.
    import os

    os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
    try:
        import cv2  # type: ignore[import-not-found]
        import numpy as np  # type: ignore[import-not-found]
    except ModuleNotFoundError as exc:
        raise TailDiagnosticsError(
            f"optional image dependency unavailable: {exc.name}"
        ) from exc

    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise TailDiagnosticsError(f"image decoder returned no pixels: {path}")
    if image.ndim == 2:
        image = np.repeat(image[:, :, None], 3, axis=2)
    if image.ndim != 3 or image.shape[2] < 3:
        raise TailDiagnosticsError(
            f"unsupported image shape {tuple(image.shape)}: {path}"
        )
    return image[:, :, :3][:, :, ::-1].astype(np.float64, copy=False)


def read_runtime_model_metadata(path: Path) -> dict[str, Any]:
    """Read the small textual header before the first runtime tensor."""

    metadata: dict[str, Any] = {}
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line_number, line in enumerate(handle):
                if line_number == 0:
                    metadata["format"] = line.strip()
                    continue
                if line.startswith("tensor "):
                    break
                key, separator, value = line.strip().partition(" ")
                if separator:
                    metadata[key] = value
    except OSError as exc:
        return {"source_path": str(path), "available": False, "error": str(exc)}
    return {"source_path": str(path), "available": True, "metadata": metadata}
