from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np


LOGGER = logging.getLogger("phase_18_crop_exr_regions")


@dataclass(slots=True, frozen=True)
class CropRegion:
    """Single cropped EXR window."""

    name: str
    x: int
    y: int
    width: int
    height: int


def configure_logging() -> None:
    """Initialize logging."""

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(
        description="Post-process a full-frame EXR into multiple cropped views"
    )
    parser.add_argument("--input-exr", type=Path, required=True, help="Source EXR")
    parser.add_argument("--output-dir", type=Path, required=True, help="Destination folder")
    parser.add_argument(
        "--crop",
        action="append",
        default=None,
        help=(
            "Crop spec, repeatable. Format `name:x,y,w,h` or `x,y,w,h`, "
            "e.g. `--crop lamp_head:128,64,256,256`."
        ),
    )
    parser.add_argument(
        "--crop-json",
        type=Path,
        default=None,
        help="JSON crops: accept object array or mapping name -> [x,y,w,h].",
    )
    parser.add_argument(
        "--quantile",
        type=float,
        default=0.995,
        help="Tone-map highlight quantile for PNG preview.",
    )
    parser.add_argument(
        "--display-clamp",
        type=float,
        default=None,
        help="Radiance clamp for PNG display only (EXR untouched).",
    )
    parser.add_argument(
        "--png-only",
        action="store_true",
        help="Emit PNG previews only.",
    )
    parser.add_argument(
        "--exr-only",
        action="store_true",
        help="Emit cropped EXR tensors only.",
    )
    return parser.parse_args()


def parse_crop_text(text: str, index: int) -> CropRegion:
    """Parse textual crop descriptor from CLI."""

    if ":" in text:
        name, coords_text = text.split(":", 1)
        name = name.strip()
    else:
        name = f"crop_{index:03d}"
        coords_text = text
    parts = [int(item.strip()) for item in coords_text.split(",")]
    if len(parts) != 4:
        raise ValueError(f"invalid crop format: {text}")
    x, y, width, height = parts
    return CropRegion(name=name, x=x, y=y, width=width, height=height)


def load_json_crops(path: Path) -> list[CropRegion]:
    """Load CropRegion list authored in JSON."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    regions: list[CropRegion] = []
    if isinstance(payload, dict):
        iterable = payload.items()
        for name, coords in iterable:
            if not isinstance(coords, list) or len(coords) != 4:
                raise ValueError(f"invalid crop entry for {name}: {coords}")
            regions.append(
                CropRegion(
                    name=str(name),
                    x=int(coords[0]),
                    y=int(coords[1]),
                    width=int(coords[2]),
                    height=int(coords[3]),
                )
            )
        return regions
    if isinstance(payload, list):
        for index, item in enumerate(payload):
            if not isinstance(item, dict):
                raise ValueError(f"invalid crop entry at {index}: {item}")
            regions.append(
                CropRegion(
                    name=str(item.get("name", f"crop_{index:03d}")),
                    x=int(item["x"]),
                    y=int(item["y"]),
                    width=int(item["width"]),
                    height=int(item["height"]),
                )
            )
        return regions
    raise ValueError(f"invalid crop json root: {path}")


def collect_regions(args: argparse.Namespace) -> list[CropRegion]:
    """Merge JSON + CLI crop definitions."""

    regions: list[CropRegion] = []
    if args.crop_json is not None:
        regions.extend(load_json_crops(args.crop_json))
    if args.crop is not None:
        regions.extend(
            parse_crop_text(text, index + len(regions))
            for index, text in enumerate(args.crop)
        )
    if not regions:
        raise ValueError("at least one --crop or --crop-json is required")
    return regions


def load_exr_rgb(path: Path) -> np.ndarray:
    """Read EXR as RGB float32."""

    os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
    import cv2

    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise RuntimeError(f"failed to read EXR: {path}")
    if image.ndim == 2:
        image = np.stack([image, image, image], axis=-1)
    if image.shape[2] >= 4:
        image = image[:, :, :3]
    image = image[:, :, ::-1].astype(np.float32, copy=False)
    return np.nan_to_num(image, copy=False)


def validate_region(region: CropRegion, image_shape: tuple[int, ...]) -> CropRegion:
    """Ensure crops stay inside raster bounds."""

    image_height, image_width = image_shape[:2]
    if region.width <= 0 or region.height <= 0:
        raise ValueError(f"invalid crop size: {region}")
    if region.x < 0 or region.y < 0:
        raise ValueError(f"invalid crop offset: {region}")
    if region.x + region.width > image_width:
        raise ValueError(f"crop exceeds image width: {region}")
    if region.y + region.height > image_height:
        raise ValueError(f"crop exceeds image height: {region}")
    return region


def crop_image(image: np.ndarray, region: CropRegion) -> np.ndarray:
    """Crop numpy RGB buffer."""

    return image[
        region.y : region.y + region.height,
        region.x : region.x + region.width,
        :,
    ].copy()


def compute_white_point(image: np.ndarray, quantile: float) -> float:
    """PNG preview white-point heuristic."""

    white_point = float(np.quantile(np.maximum(image, 0.0), quantile))
    return max(white_point, 1e-6)


def tone_map(image: np.ndarray, white_point: float) -> np.ndarray:
    """Convert HDR crop to uint8 PNG buffer."""

    mapped = np.clip(image / white_point, 0.0, 1.0)
    mapped = np.power(mapped, 1.0 / 2.2)
    return np.clip(mapped * 255.0 + 0.5, 0, 255).astype(np.uint8)


def apply_display_clamp(image: np.ndarray, limit: float | None) -> np.ndarray:
    """Optional luminance clamp for PNG preview."""

    if limit is None or limit <= 0.0:
        return image
    return np.minimum(image, np.float32(limit))


def write_exr(path: Path, image: np.ndarray) -> None:
    """Write RGB EXR via OpenCV."""

    os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), image[:, :, ::-1].astype(np.float32, copy=False))
    if not ok:
        raise RuntimeError(f"failed to write EXR: {path}")


def write_png(path: Path, image: np.ndarray) -> None:
    """Write uint8 PNG via OpenCV."""

    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), image[:, :, ::-1])
    if not ok:
        raise RuntimeError(f"failed to write PNG: {path}")


def main() -> int:
    """Crop EXR into derivative assets."""

    configure_logging()
    args = parse_args()
    if args.png_only and args.exr_only:
        raise ValueError("--png-only and --exr-only cannot be used together")

    input_exr = args.input_exr.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    image = load_exr_rgb(input_exr)
    regions = [validate_region(region, image.shape) for region in collect_regions(args)]

    manifest_records: list[dict[str, object]] = []
    for region in regions:
        cropped = crop_image(image, region)
        record: dict[str, object] = {
            "name": region.name,
            "x": region.x,
            "y": region.y,
            "width": region.width,
            "height": region.height,
        }
        if not args.png_only:
            exr_path = output_dir / f"{region.name}.exr"
            write_exr(exr_path, cropped)
            record["exr_path"] = exr_path.relative_to(output_dir).as_posix()
        if not args.exr_only:
            png_source = apply_display_clamp(cropped, args.display_clamp)
            white_point = compute_white_point(png_source, args.quantile)
            png_path = output_dir / f"{region.name}.png"
            write_png(png_path, tone_map(png_source, white_point))
            record["png_path"] = png_path.relative_to(output_dir).as_posix()
            record["white_point"] = white_point
        manifest_records.append(record)

    manifest_path = output_dir / "crop_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "input_exr": str(input_exr),
                "image_width": int(image.shape[1]),
                "image_height": int(image.shape[0]),
                "records": manifest_records,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    LOGGER.info("crop export finished: %s", manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
