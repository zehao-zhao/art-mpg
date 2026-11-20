from __future__ import annotations

import argparse
import hashlib
import json
import math
import xml.etree.ElementTree as et
from pathlib import Path


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = WORKSPACE_ROOT.parent
METHODS = ("ordered_diffusion_mlp_h72_k4", "simple_mlp_k4")
SCENE_SEEDS = {"paper_lamp": 1729, "paper_slabs": 424242}
EXPECTED_ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize a Phase 13 bounce-alpha sweep")
    parser.add_argument(
        "--input-root",
        type=Path,
        default=WORKSPACE_ROOT / "results" / "bounce_alpha_sweep_macos_20260901",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--output-markdown",
        type=Path,
        default=None,
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def geometric_mean(values: list[float]) -> float:
    if not values or any(value <= 0.0 for value in values):
        raise ValueError("geometric mean requires positive values")
    return math.exp(sum(math.log(value) for value in values) / len(values))


def resolve_record_path(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_xml_bounce_alpha(path: Path) -> float:
    root = et.fromstring(path.read_text(encoding="utf-8"))
    node = root.find("./integrator/float[@name='bounce_alpha']")
    if node is None or "value" not in node.attrib:
        raise RuntimeError(f"missing bounce_alpha in patched XML: {path}")
    return float(node.attrib["value"])


def load_curve(input_root: Path) -> dict[str, list[dict[str, object]]]:
    curves: dict[str, list[dict[str, object]]] = {
        scene: [] for scene in SCENE_SEEDS
    }
    summary_paths = sorted(input_root.glob("*/run_*/benchmark_summary.json"))
    if len(summary_paths) != len(SCENE_SEEDS) * len(EXPECTED_ALPHAS):
        raise RuntimeError(
            f"expected {len(SCENE_SEEDS) * len(EXPECTED_ALPHAS)} summaries, "
            f"found {len(summary_paths)} under {input_root}"
        )

    seen: set[tuple[str, float]] = set()
    for summary_path in summary_paths:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        config = payload["config"]
        alpha = float(config["bounce_alpha"])
        records = {record["method"]: record for record in payload["records"]}
        scene = str(records["path_reference"]["scene_id"])
        key = (scene, alpha)
        if scene not in curves or alpha not in EXPECTED_ALPHAS or key in seen:
            raise RuntimeError(f"unexpected or duplicate sweep point: {key}")
        seen.add(key)

        if tuple(config["methods"]) != ("path_reference", *METHODS):
            raise RuntimeError(f"unexpected method set in {summary_path}")
        if int(config["reference_sample_count"]) != 1024 or int(
            config["compare_sample_count"]
        ) != 64:
            raise RuntimeError(f"unexpected SPP budget in {summary_path}")
        if int(config["seed"]) != SCENE_SEEDS[scene]:
            raise RuntimeError(f"unexpected render seed in {summary_path}")
        if int(payload["passed"]) != 3 or int(payload["failed"]) != 0:
            raise RuntimeError(f"failed render record in {summary_path}")

        reference = records["path_reference"]
        row: dict[str, object] = {
            "alpha": alpha,
            "summary_path": summary_path.relative_to(WORKSPACE_ROOT).as_posix(),
            "reference_config_sha256": reference["active_config_sha256"],
            "reference_image_sha256": sha256_file(
                resolve_record_path(str(reference["output_path"]))
            ),
            "methods": {},
        }
        method_values: dict[str, object] = {}
        for method in METHODS:
            record = records[method]
            if not bool(record["ok"]) or not bool(record["loaded_runtime_model"]):
                raise RuntimeError(f"runtime was not loaded for {scene}/{alpha}/{method}")
            xml_alpha = read_xml_bounce_alpha(
                resolve_record_path(str(record["patched_scene_record"]))
            )
            if not math.isclose(xml_alpha, alpha, rel_tol=0.0, abs_tol=1e-12):
                raise RuntimeError(
                    f"patched XML alpha mismatch for {scene}/{alpha}/{method}: {xml_alpha}"
                )
            method_values[method] = {
                "mse": float(record["mse"]),
                "seconds": float(record["seconds"]),
                "loaded_runtime_model": True,
                "active_config_sha256": record["active_config_sha256"],
                "runtime_model_path": record["runtime_model_path"],
                "xml_bounce_alpha": xml_alpha,
            }
        row["methods"] = method_values
        curves[scene].append(row)

    for scene, rows in curves.items():
        rows.sort(key=lambda row: float(row["alpha"]))
        observed = tuple(float(row["alpha"]) for row in rows)
        if observed != EXPECTED_ALPHAS:
            raise RuntimeError(f"incomplete alpha grid for {scene}: {observed}")
        reference_hashes = {str(row["reference_image_sha256"]) for row in rows}
        if len(reference_hashes) != 1:
            raise RuntimeError(f"reference image changed across alpha for {scene}")
    return curves


def build_report(input_root: Path) -> dict[str, object]:
    curves = load_curve(input_root)
    best_by_scene: dict[str, dict[str, object]] = {}
    for scene, rows in curves.items():
        best_by_scene[scene] = {}
        for method in METHODS:
            best_row = min(
                rows,
                key=lambda row: float(row["methods"][method]["mse"]),
            )
            best_by_scene[scene][method] = {
                "alpha": float(best_row["alpha"]),
                "mse": float(best_row["methods"][method]["mse"]),
            }

    pooled_curve: list[dict[str, object]] = []
    for alpha in EXPECTED_ALPHAS:
        pooled_row: dict[str, object] = {"alpha": alpha, "methods": {}}
        for method in METHODS:
            scene_mses = {
                scene: float(
                    next(row for row in rows if float(row["alpha"]) == alpha)[
                        "methods"
                    ][method]["mse"]
                )
                for scene, rows in curves.items()
            }
            pooled_row["methods"][method] = {
                "geometric_mean_mse": geometric_mean(list(scene_mses.values())),
                "scene_mses": scene_mses,
            }
        pooled_curve.append(pooled_row)

    pooled_selection: dict[str, object] = {}
    for method in METHODS:
        best_row = min(
            pooled_curve,
            key=lambda row: float(row["methods"][method]["geometric_mean_mse"]),
        )
        pooled_selection[method] = {
            "alpha": float(best_row["alpha"]),
            "geometric_mean_mse": float(
                best_row["methods"][method]["geometric_mean_mse"]
            ),
        }

    tuned_comparison_by_scene = {
        scene: {
            "diffusion_to_simple_mse_ratio": (
                float(methods[METHODS[0]]["mse"])
                / float(methods[METHODS[1]]["mse"])
            ),
            "diffusion_relative_mse_percent": 100.0
            * (
                float(methods[METHODS[0]]["mse"])
                / float(methods[METHODS[1]]["mse"])
                - 1.0
            ),
        }
        for scene, methods in best_by_scene.items()
    }
    pooled_tuned_ratio = (
        float(pooled_selection[METHODS[0]]["geometric_mean_mse"])
        / float(pooled_selection[METHODS[1]]["geometric_mean_mse"])
    )

    return {
        "status": "screening_only_single_seed_per_scene",
        "protocol": {
            "scenes": list(SCENE_SEEDS),
            "render_seeds": SCENE_SEEDS,
            "reference_seed": 99173,
            "candidate_spp": 64,
            "reference_spp": 1024,
            "alphas": list(EXPECTED_ALPHAS),
            "methods": list(METHODS),
            "render_record_count": 30,
            "neural_record_count": 20,
            "runtime_loaded_verified": True,
            "neural_patched_xml_alpha_verified": True,
            "caps": {
                "inverse_probability": 512.0,
                "contribution": 32.0,
                "max_trials": 2048,
            },
        },
        "curves": curves,
        "best_by_scene": best_by_scene,
        "pooled_curve": pooled_curve,
        "pooled_selection": pooled_selection,
        "tuned_comparison": {
            "by_scene": tuned_comparison_by_scene,
            "pooled_diffusion_to_simple_mse_ratio": pooled_tuned_ratio,
            "pooled_diffusion_relative_mse_percent": 100.0
            * (pooled_tuned_ratio - 1.0),
        },
    }


def write_markdown(report: dict[str, object], output_path: Path) -> None:
    curves = report["curves"]
    lines = [
        "# Bounce-alpha Development Sweep",
        "",
        "Screening only: one fixed render seed per development scene. Models, caps,",
        "resolution, SPP, and reference seed are fixed; only `bounce_alpha` changes.",
        "All 20 neural runs loaded their requested runtime model, and every neural",
        "patched XML recorded the requested alpha.",
        "",
    ]
    for scene, rows in curves.items():
        lines.extend(
            [
                f"## {scene}",
                "",
                "| alpha | ordered diffusion MSE | simple MLP K4 MSE | diffusion/simple | diffusion sec | simple sec |",
                "| ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in rows:
            diffusion = row["methods"][METHODS[0]]
            simple = row["methods"][METHODS[1]]
            lines.append(
                f"| {row['alpha']:.2f} | {diffusion['mse']:.9f} | "
                f"{simple['mse']:.9f} | {diffusion['mse'] / simple['mse']:.6f} | "
                f"{diffusion['seconds']:.3f} | {simple['seconds']:.3f} |"
            )
        lines.append("")

    lines.extend(
        [
            "## Best alpha by scene and method",
            "",
            "| scene | method | best alpha | MSE |",
            "| --- | --- | ---: | ---: |",
        ]
    )
    for scene, methods in report["best_by_scene"].items():
        for method, values in methods.items():
            lines.append(
                f"| {scene} | {method} | {values['alpha']:.2f} | {values['mse']:.9f} |"
            )

    lines.extend(
        [
            "",
            "## Pooled selection across both scenes",
            "",
            "The pooled score is the geometric mean of Lamp and Slabs MSE at the",
            "same alpha. Lower is better.",
            "",
            "| alpha | ordered diffusion geometric MSE | simple MLP K4 geometric MSE |",
            "| ---: | ---: | ---: |",
        ]
    )
    for row in report["pooled_curve"]:
        lines.append(
            f"| {row['alpha']:.2f} | "
            f"{row['methods'][METHODS[0]]['geometric_mean_mse']:.9f} | "
            f"{row['methods'][METHODS[1]]['geometric_mean_mse']:.9f} |"
        )

    lines.extend(["", "Selected pooled settings:", ""])
    for method, values in report["pooled_selection"].items():
        lines.append(
            f"- `{method}`: alpha `{values['alpha']:.2f}`, geometric MSE "
            f"`{values['geometric_mean_mse']:.9f}`"
        )
    tuned = report["tuned_comparison"]
    lines.extend(
        [
            "",
            "## Fair model-specific selection outcome",
            "",
            "Each model is compared at its own selected alpha.",
            "",
            "| scope | diffusion/simple MSE ratio | diffusion relative MSE |",
            "| --- | ---: | ---: |",
        ]
    )
    for scene, values in tuned["by_scene"].items():
        lines.append(
            f"| {scene} | {values['diffusion_to_simple_mse_ratio']:.6f} | "
            f"{values['diffusion_relative_mse_percent']:+.3f}% |"
        )
    lines.append(
        f"| pooled geometric MSE | "
        f"{tuned['pooled_diffusion_to_simple_mse_ratio']:.6f} | "
        f"{tuned['pooled_diffusion_relative_mse_percent']:+.3f}% |"
    )
    lines.extend(
        [
            "",
            "These are development-set choices, not final evidence. Re-evaluate the",
            "selected values with multiple fresh seeds without tuning on Ring/Pillars.",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    input_root = args.input_root.resolve()
    output_json = (
        input_root / "bounce_alpha_curve.json"
        if args.output_json is None
        else args.output_json.resolve()
    )
    output_markdown = (
        input_root / "bounce_alpha_curve.md"
        if args.output_markdown is None
        else args.output_markdown.resolve()
    )
    report = build_report(input_root)
    output_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    write_markdown(report, output_markdown)
    print(output_json)
    print(output_markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
