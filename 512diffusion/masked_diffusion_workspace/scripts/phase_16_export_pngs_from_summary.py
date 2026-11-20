from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

import numpy as np


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = WORKSPACE_ROOT.parent
LOGGER = logging.getLogger("phase_16_export_pngs_from_summary")


def configure_logging() -> None:
    """Initialize logging."""

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(
        description="Export per-method PNGs from benchmark summary JSON"
    )
    parser.add_argument(
        "--summary", type=Path, required=True, help="benchmark_summary.json"
    )
    parser.add_argument("--output-dir", type=Path, required=True, help="PNG output folder")
    parser.add_argument(
        "--quantile",
        type=float,
        default=0.995,
        help="Tone-map white-point quantile.",
    )
    parser.add_argument(
        "--methods",
        type=str,
        nargs="*",
        default=None,
        help=(
            "Only export these methods; omit to emit every succeeded method "
            "in the summary."
        ),
    )
    parser.add_argument(
        "--display-clamp",
        type=float,
        default=None,
        help="Linear radiance clamp for PNG display only (EXR unchanged).",
    )
    return parser.parse_args()


def load_summary(path: Path) -> dict[str, object]:
    """Load summary JSON."""

    return json.loads(path.read_text(encoding="utf-8"))


def load_exr_rgb(path: Path) -> np.ndarray:
    """Load EXR and return RGB float image.

    Args:
        path: EXR filepath.

    Returns:
        Floating-point RGB tensor.
    """

    os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
    import cv2

    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise RuntimeError(f"unable to read EXR: {path}")
    if image.ndim == 2:
        image = np.stack([image, image, image], axis=-1)
    if image.shape[2] >= 4:
        image = image[:, :, :3]
    image = image[:, :, ::-1].astype(np.float32, copy=False)
    return np.nan_to_num(image, copy=False)


def compute_white_point(image: np.ndarray, quantile: float) -> float:
    """Compute visualization white-point for one EXR.

    Args:
        image: Float RGB raster.
        quantile: High quantile controlling highlight roll-off.

    Returns:
        Scalar white point for tone mapping.
    """

    white_point = float(np.quantile(np.maximum(image, 0.0), quantile))
    return max(white_point, 1e-6)


def tone_map(image: np.ndarray, white_point: float) -> np.ndarray:
    """Map HDR radiance to displayable PNG RGB.

    Args:
        image: Float RGB raster.
        white_point: Shared white-point per scene reference.

    Returns:
        UInt8 PNG buffer.
    """

    mapped = np.clip(image / white_point, 0.0, 1.0)
    mapped = np.power(mapped, 1.0 / 2.2)
    return np.clip(mapped * 255.0 + 0.5, 0, 255).astype(np.uint8)


def apply_display_clamp(image: np.ndarray, limit: float | None) -> np.ndarray:
    """Apply a figure-only clamp before tone mapping."""

    if limit is None or limit <= 0.0:
        return image
    return np.minimum(image, np.float32(limit))


def write_png(path: Path, image: np.ndarray) -> None:
    """Write PNG raster."""

    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image[:, :, ::-1])


def relativize_to_output(path: Path, output_dir: Path) -> str:
    """Encode path relative to output directory."""

    return path.resolve().relative_to(output_dir.resolve()).as_posix()


def resolve_record_path(path_text: str, *, summary_path: Path) -> Path:
    """Resolve relative paths emitted inside summary blobs.

    Args:
        path_text: Path literal stored in benchmark record.
        summary_path: On-disk summary path (used for legacy layouts).

    Returns:
        Resolved absolute filepath.
    """

    candidate = Path(path_text)
    if candidate.is_absolute():
        return candidate.resolve()

    # New summaries are repo-root-relative; stitch against project root first.
    project_candidate = (PROJECT_ROOT / candidate).resolve()
    if project_candidate.exists():
        return project_candidate

    # Fallback for older summaries or temp-directory relative exports.
    summary_candidate = (summary_path.parent / candidate).resolve()
    if summary_candidate.exists():
        return summary_candidate

    return project_candidate


def main() -> int:
    """Run PNG export."""

    configure_logging()
    args = parse_args()
    summary = load_summary(args.summary.resolve())
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    allowed_methods = (
        None if args.methods is None else {str(method) for method in args.methods}
    )

    scene_white_points: dict[str, float] = {}
    for record in summary["records"]:
        if not bool(record["ok"]):
            continue
        if str(record["method"]) != "path_reference":
            continue
        scene_id = str(record["scene_id"])
        exr_path = resolve_record_path(
            str(record["output_path"]),
            summary_path=args.summary.resolve(),
        )
        scene_white_points[scene_id] = compute_white_point(
            load_exr_rgb(exr_path),
            args.quantile,
        )

    manifest_records: list[dict[str, object]] = []
    for record in summary["records"]:
        if not bool(record["ok"]):
            continue
        scene_id = str(record["scene_id"])
        method = str(record["method"])
        if allowed_methods is not None and method not in allowed_methods:
            continue
        exr_path = resolve_record_path(
            str(record["output_path"]),
            summary_path=args.summary.resolve(),
        )
        image = apply_display_clamp(load_exr_rgb(exr_path), args.display_clamp)
        white_point = scene_white_points.get(
            scene_id,
            compute_white_point(image, args.quantile),
        )
        png_path = output_dir / scene_id / f"{method}.png"
        write_png(png_path, tone_map(image, white_point))
        manifest_records.append(
            {
                "scene_id": scene_id,
                "method": method,
                "png_path": relativize_to_output(png_path, output_dir),
                "seconds": float(record["seconds"]),
                "sample_count": int(record["sample_count"]),
                "mse": record.get("mse"),
                "psnr": record.get("psnr"),
            }
        )

    manifest_path = output_dir / "png_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {"records": manifest_records},
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    LOGGER.info("png export finished: %s", manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
