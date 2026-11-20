#!/usr/bin/env python3
"""Generate deterministic vector SPP--MSE plots using only the Python stdlib."""

from __future__ import annotations

import argparse
import html
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Callable


PROTOCOL_ID = "phase23_iclr_mpgd_spp_mse_living_slabs_20260913_v1"
SCENES = ("living", "slabs")
METHODS = ("diffusion_mlp", "npm_adapted", "mpg_original", "pt", "sms_star")
SAMPLE_COUNTS = (1, 2, 4, 8, 16, 32, 64)
COLORS = {
    "diffusion_mlp": (0.84, 0.15, 0.16),
    "npm_adapted": (0.12, 0.47, 0.71),
    "mpg_original": (0.17, 0.63, 0.17),
    "pt": (0.30, 0.30, 0.30),
    "sms_star": (0.58, 0.40, 0.74),
}


class PlotError(RuntimeError):
    pass


def load_report(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PlotError(f"cannot read report: {exc}") from exc
    if not isinstance(value, dict) or value.get("protocol_id") != PROTOCOL_ID:
        raise PlotError("report protocol mismatch")
    if value.get("status") != "complete":
        raise PlotError("report is not complete")
    return value


def _immutable_bytes(target: Path, payload: bytes) -> None:
    if target.exists():
        if target.read_bytes() != payload:
            raise PlotError(f"refusing to overwrite changed plot: {target}")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, target)


def _validate_summary(report: dict[str, Any]) -> dict[tuple[str, str, int], dict[str, Any]]:
    summary = report.get("summary")
    if not isinstance(summary, list):
        raise PlotError("report summary is missing")
    index: dict[tuple[str, str, int], dict[str, Any]] = {}
    for row in summary:
        if not isinstance(row, dict):
            raise PlotError("summary row is invalid")
        key = (str(row.get("scene")), str(row.get("method")), int(row.get("sample_count", 0)))
        if key in index:
            raise PlotError(f"duplicate summary point: {key}")
        index[key] = row
    expected = {
        (scene, method, spp)
        for scene in SCENES
        for method in METHODS
        for spp in SAMPLE_COUNTS
    }
    if set(index) != expected:
        raise PlotError("summary grid is incomplete or contains extra points")
    for key, row in index.items():
        for field in ("median_mse", "q25_mse", "q75_mse"):
            value = float(row.get(field, math.nan))
            if not math.isfinite(value) or value <= 0.0:
                raise PlotError(f"nonpositive/nonfinite log-scale metric: {key}/{field}")
    return index


def _pdf_escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _pdf_document(content: str, width: float, height: float) -> bytes:
    stream = content.encode("ascii")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {width:.3f} {height:.3f}] "
            "/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>"
        ).encode("ascii"),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Length " + str(len(stream)).encode("ascii") + b" >>\nstream\n" + stream + b"endstream",
    ]
    payload = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = [0]
    for object_index, obj in enumerate(objects, 1):
        offsets.append(len(payload))
        payload.extend(f"{object_index} 0 obj\n".encode("ascii"))
        payload.extend(obj)
        payload.extend(b"\nendobj\n")
    xref = len(payload)
    payload.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    payload.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        payload.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    payload.extend(
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode("ascii")
    )
    return bytes(payload)


def _make_geometry(
    index: dict[tuple[str, str, int], dict[str, Any]]
) -> tuple[float, float, list[dict[str, Any]]]:
    page_width, page_height = 540.0, 220.0
    panels: list[dict[str, Any]] = []
    for scene_index, scene in enumerate(SCENES):
        left = 46.0 + scene_index * 267.0
        right = left + 220.0
        bottom, top = 34.0, 176.0
        values = [
            float(index[(scene, method, spp)][field])
            for method in METHODS
            for spp in SAMPLE_COUNTS
            for field in ("q25_mse", "q75_mse")
        ]
        log_min = math.log10(min(values))
        log_max = math.log10(max(values))
        padding = max(0.08, 0.08 * (log_max - log_min))
        log_min -= padding
        log_max += padding

        def x_coord(spp: int, *, lo: float = left, hi: float = right) -> float:
            return lo + math.log2(spp) / 6.0 * (hi - lo)

        def y_coord(value: float, *, lo: float = bottom, hi: float = top) -> float:
            return lo + (math.log10(value) - log_min) / (log_max - log_min) * (hi - lo)

        panels.append({
            "scene": scene,
            "left": left,
            "right": right,
            "bottom": bottom,
            "top": top,
            "log_min": log_min,
            "log_max": log_max,
            "x": x_coord,
            "y": y_coord,
        })
    return page_width, page_height, panels


def _pdf_plot(
    report: dict[str, Any], index: dict[tuple[str, str, int], dict[str, Any]]
) -> bytes:
    width, height, panels = _make_geometry(index)
    commands: list[str] = ["1 J 1 j"]

    def text(x: float, y: float, size: float, value: str) -> None:
        commands.append(
            f"0 0 0 rg BT /F1 {size:.2f} Tf {x:.3f} {y:.3f} Td "
            f"({_pdf_escape(value)}) Tj ET"
        )

    def rotated_text(x: float, y: float, size: float, value: str) -> None:
        commands.append(
            f"0 0 0 rg BT /F1 {size:.2f} Tf 0 1 -1 0 {x:.3f} {y:.3f} Tm "
            f"({_pdf_escape(value)}) Tj ET"
        )

    display = report["public_method_names"]
    legend_x = 70.0
    for method in METHODS:
        red, green, blue = COLORS[method]
        commands.append(f"{red:.3f} {green:.3f} {blue:.3f} RG 1.6 w {legend_x:.2f} 201 m {legend_x + 17:.2f} 201 l S")
        commands.append(f"{red:.3f} {green:.3f} {blue:.3f} rg {legend_x + 7:.2f} 198 6 6 re f")
        text(legend_x + 21.0, 198.4, 7.2, str(display[method]))
        legend_x += 78.0
    text(170.0, 187.0, 7.1, "Median MSE; shaded band = interquartile range")

    for panel in panels:
        scene = panel["scene"]
        left, right = panel["left"], panel["right"]
        bottom, top = panel["bottom"], panel["top"]
        x_coord: Callable[[int], float] = panel["x"]
        y_coord: Callable[[float], float] = panel["y"]
        commands.append("0.86 0.86 0.86 RG 0.35 w")
        for spp in SAMPLE_COUNTS:
            x = x_coord(spp)
            commands.append(f"{x:.3f} {bottom:.3f} m {x:.3f} {top:.3f} l S")
            text(x - (2.3 if spp < 10 else 4.6), bottom - 11.5, 6.3, str(spp))
        tick_logs = [
            panel["log_min"] + fraction * (panel["log_max"] - panel["log_min"])
            for fraction in (0.0, 1 / 3, 2 / 3, 1.0)
        ]
        for value in tick_logs:
            y = y_coord(10 ** value)
            commands.append(f"{left:.3f} {y:.3f} m {right:.3f} {y:.3f} l S")
            text(left - 38.0, y - 2.2, 5.8, f"{10 ** value:.1e}")
        commands.append(f"0 0 0 RG 0.65 w {left:.3f} {bottom:.3f} m {right:.3f} {bottom:.3f} l {right:.3f} {top:.3f} l {left:.3f} {top:.3f} l h S")
        text((left + right) / 2 - 14.0, top + 4.5, 8.5, scene.capitalize())
        text((left + right) / 2 - 25.0, 10.5, 7.0, "Samples per pixel (SPP)")
        if scene == SCENES[0]:
            rotated_text(10.0, 78.0, 6.7, "MSE (linear RGB)")

        for method in METHODS:
            points = [index[(scene, method, spp)] for spp in SAMPLE_COUNTS]
            lows = [(x_coord(spp), y_coord(float(point["q25_mse"]))) for spp, point in zip(SAMPLE_COUNTS, points, strict=True)]
            highs = [(x_coord(spp), y_coord(float(point["q75_mse"]))) for spp, point in zip(SAMPLE_COUNTS, points, strict=True)]
            medians = [(x_coord(spp), y_coord(float(point["median_mse"]))) for spp, point in zip(SAMPLE_COUNTS, points, strict=True)]
            red, green, blue = COLORS[method]
            fill = tuple(0.82 + 0.18 * component for component in (red, green, blue))
            polygon = lows + list(reversed(highs))
            commands.append(f"{fill[0]:.3f} {fill[1]:.3f} {fill[2]:.3f} rg")
            commands.append(f"{polygon[0][0]:.3f} {polygon[0][1]:.3f} m")
            commands.extend(f"{x:.3f} {y:.3f} l" for x, y in polygon[1:])
            commands.append("h f")
            commands.append(f"{red:.3f} {green:.3f} {blue:.3f} RG 1.35 w")
            commands.append(f"{medians[0][0]:.3f} {medians[0][1]:.3f} m")
            commands.extend(f"{x:.3f} {y:.3f} l" for x, y in medians[1:])
            commands.append("S")
            commands.append(f"{red:.3f} {green:.3f} {blue:.3f} rg")
            commands.extend(f"{x - 1.8:.3f} {y - 1.8:.3f} 3.6 3.6 re f" for x, y in medians)
    return _pdf_document("\n".join(commands) + "\n", width, height)


def _svg_plot(
    report: dict[str, Any], index: dict[tuple[str, str, int], dict[str, Any]]
) -> bytes:
    width, height, panels = _make_geometry(index)

    def rgb(color: tuple[float, float, float]) -> str:
        return "#" + "".join(f"{round(component * 255):02x}" for component in color)

    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width:.0f}" height="{height:.0f}" viewBox="0 0 {width:.0f} {height:.0f}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<g font-family="Helvetica,Arial,sans-serif" fill="#111">',
    ]
    display = report["public_method_names"]
    legend_x = 70.0
    for method in METHODS:
        color = rgb(COLORS[method])
        elements.append(f'<line x1="{legend_x:.2f}" y1="19" x2="{legend_x + 17:.2f}" y2="19" stroke="{color}" stroke-width="1.6"/>')
        elements.append(f'<rect x="{legend_x + 7:.2f}" y="16" width="6" height="6" fill="{color}"/>')
        elements.append(f'<text x="{legend_x + 21:.2f}" y="22" font-size="7.2">{html.escape(str(display[method]))}</text>')
        legend_x += 78.0
    elements.append('<text x="170" y="34" font-size="7.1">Median MSE; shaded band = interquartile range</text>')

    for panel in panels:
        scene = panel["scene"]
        left, right = panel["left"], panel["right"]
        bottom_pdf, top_pdf = panel["bottom"], panel["top"]
        top = height - top_pdf
        bottom = height - bottom_pdf
        x_coord: Callable[[int], float] = panel["x"]
        y_pdf: Callable[[float], float] = panel["y"]
        y_coord = lambda value: height - y_pdf(value)
        for spp in SAMPLE_COUNTS:
            x = x_coord(spp)
            elements.append(f'<line x1="{x:.3f}" y1="{top:.3f}" x2="{x:.3f}" y2="{bottom:.3f}" stroke="#d8d8d8" stroke-width="0.35"/>')
            elements.append(f'<text x="{x:.3f}" y="{bottom + 11.5:.3f}" text-anchor="middle" font-size="6.3">{spp}</text>')
        tick_logs = [
            panel["log_min"] + fraction * (panel["log_max"] - panel["log_min"])
            for fraction in (0.0, 1 / 3, 2 / 3, 1.0)
        ]
        for value in tick_logs:
            y = y_coord(10 ** value)
            elements.append(f'<line x1="{left:.3f}" y1="{y:.3f}" x2="{right:.3f}" y2="{y:.3f}" stroke="#d8d8d8" stroke-width="0.35"/>')
            elements.append(f'<text x="{left - 3:.3f}" y="{y + 2:.3f}" text-anchor="end" font-size="5.8">{10 ** value:.1e}</text>')
        elements.append(f'<rect x="{left:.3f}" y="{top:.3f}" width="{right-left:.3f}" height="{bottom-top:.3f}" fill="none" stroke="#111" stroke-width="0.65"/>')
        elements.append(f'<text x="{(left + right) / 2:.3f}" y="{top - 5:.3f}" text-anchor="middle" font-size="8.5">{scene.capitalize()}</text>')
        elements.append(f'<text x="{(left + right) / 2:.3f}" y="{height - 10.5:.3f}" text-anchor="middle" font-size="7">Samples per pixel (SPP)</text>')
        if scene == SCENES[0]:
            middle = (top + bottom) / 2
            elements.append(f'<text x="8" y="{middle:.3f}" transform="rotate(-90 8 {middle:.3f})" text-anchor="middle" font-size="6.7">MSE (linear RGB)</text>')
        for method in METHODS:
            points = [index[(scene, method, spp)] for spp in SAMPLE_COUNTS]
            lows = [(x_coord(spp), y_coord(float(point["q25_mse"]))) for spp, point in zip(SAMPLE_COUNTS, points, strict=True)]
            highs = [(x_coord(spp), y_coord(float(point["q75_mse"]))) for spp, point in zip(SAMPLE_COUNTS, points, strict=True)]
            medians = [(x_coord(spp), y_coord(float(point["median_mse"]))) for spp, point in zip(SAMPLE_COUNTS, points, strict=True)]
            color = rgb(COLORS[method])
            polygon = " ".join(f"{x:.3f},{y:.3f}" for x, y in lows + list(reversed(highs)))
            line = " ".join(f"{x:.3f},{y:.3f}" for x, y in medians)
            elements.append(f'<polygon points="{polygon}" fill="{color}" fill-opacity="0.18"/>')
            elements.append(f'<polyline points="{line}" fill="none" stroke="{color}" stroke-width="1.35"/>')
            elements.extend(f'<rect x="{x - 1.8:.3f}" y="{y - 1.8:.3f}" width="3.6" height="3.6" fill="{color}"/>' for x, y in medians)
    elements.extend(["</g>", "</svg>"])
    return ("\n".join(elements) + "\n").encode("utf-8")


def generate(report: dict[str, Any], output_dir: Path) -> tuple[Path, Path]:
    index = _validate_summary(report)
    pdf = output_dir / "spp_mse_decay.pdf"
    svg = output_dir / "spp_mse_decay.svg"
    _immutable_bytes(pdf, _pdf_plot(report, index))
    _immutable_bytes(svg, _svg_plot(report, index))
    return pdf, svg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        pdf, svg = generate(load_report(args.report.resolve()), args.output_dir.resolve())
        print(json.dumps({
            "protocol_id": PROTOCOL_ID,
            "status": "plots_ready",
            "pdf": str(pdf),
            "svg": str(svg),
        }, indent=2, sort_keys=True))
        return 0
    except PlotError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
