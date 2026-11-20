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
SCENES = ("paper_lamp", "paper_slabs")
SEEDS = (104729, 130363, 155921)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize the hemisphere-flip A/B")
    parser.add_argument(
        "--input-root",
        type=Path,
        default=WORKSPACE_ROOT / "results" / "hemisphere_ab_macos_20260901",
    )
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-markdown", type=Path, default=None)
    return parser.parse_args()


def geometric_mean(values: list[float]) -> float:
    if not values or any(value <= 0.0 for value in values):
        raise ValueError("geometric mean requires positive values")
    return math.exp(sum(math.log(value) for value in values) / len(values))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_record_path(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_runtime_xml_controls(path: Path) -> tuple[float, float, bool]:
    root = et.fromstring(path.read_text(encoding="utf-8"))
    integrator = root.find("./integrator")
    if integrator is None:
        raise RuntimeError(f"missing integrator in {path}")

    def property_value(tag: str, name: str) -> str:
        node = integrator.find(f"./{tag}[@name='{name}']")
        if node is None or "value" not in node.attrib:
            raise RuntimeError(f"missing {name} in {path}")
        return node.attrib["value"]

    return (
        float(property_value("float", "bounce_alpha")),
        float(property_value("float", "learned_direction_kappa_scale")),
        property_value(
            "boolean", "learned_direction_flip_to_receiver_hemisphere"
        ).lower()
        == "true",
    )


def load_runs(input_root: Path) -> list[dict[str, object]]:
    summary_paths = sorted(input_root.glob("*/run_*/benchmark_summary.json"))
    expected_count = len(SCENES) * len(SEEDS) * 2
    if len(summary_paths) != expected_count:
        raise RuntimeError(
            f"expected {expected_count} summaries, found {len(summary_paths)} "
            f"under {input_root}"
        )

    rows: list[dict[str, object]] = []
    seen: set[tuple[str, int, bool]] = set()
    for summary_path in summary_paths:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        config = payload["config"]
        records = {record["method"]: record for record in payload["records"]}
        scene = str(records["path_reference"]["scene_id"])
        seed = int(config["seed"])
        flip = bool(config["learned_direction_flip_to_receiver_hemisphere"])
        key = (scene, seed, flip)
        if scene not in SCENES or seed not in SEEDS or key in seen:
            raise RuntimeError(f"unexpected or duplicate A/B point: {key}")
        seen.add(key)

        if tuple(config["methods"]) != ("path_reference", *METHODS):
            raise RuntimeError(f"unexpected method set in {summary_path}")
        expected_controls = (
            int(config["reference_sample_count"]) == 1024
            and int(config["compare_sample_count"]) == 64
            and math.isclose(float(config["bounce_alpha"]), 0.5)
            and math.isclose(float(config["learned_direction_kappa_scale"]), 1.0)
        )
        if not expected_controls:
            raise RuntimeError(f"unexpected fixed controls in {summary_path}")
        if int(payload["passed"]) != 3 or int(payload["failed"]) != 0:
            raise RuntimeError(f"failed render record in {summary_path}")

        reference = records["path_reference"]
        method_values: dict[str, object] = {}
        for method in METHODS:
            record = records[method]
            if not bool(record["ok"]) or not bool(record["loaded_runtime_model"]):
                raise RuntimeError(f"runtime not loaded for {scene}/{seed}/{flip}/{method}")
            xml_controls = read_runtime_xml_controls(
                resolve_record_path(str(record["patched_scene_record"]))
            )
            if xml_controls != (0.5, 1.0, flip):
                raise RuntimeError(
                    f"runtime XML controls mismatch for {scene}/{seed}/{flip}/{method}: "
                    f"{xml_controls}"
                )
            stdout_text = resolve_record_path(str(record["stdout_log"])).read_text(
                encoding="utf-8"
            )
            native_marker = (
                "Masked diffusion runtime learned_direction_kappa_scale=1 "
                f"flip_to_receiver_hemisphere={int(flip)}"
            )
            if native_marker not in stdout_text:
                raise RuntimeError(
                    f"native runtime control marker missing for {scene}/{seed}/{flip}/{method}"
                )
            method_values[method] = {
                "mse": float(record["mse"]),
                "seconds": float(record["seconds"]),
                "loaded_runtime_model": True,
                "runtime_model_path": record["runtime_model_path"],
                "active_config_sha256": record["active_config_sha256"],
            }

        rows.append(
            {
                "scene": scene,
                "seed": seed,
                "flip": flip,
                "summary_path": summary_path.relative_to(WORKSPACE_ROOT).as_posix(),
                "reference_image_sha256": sha256_file(
                    resolve_record_path(str(reference["output_path"]))
                ),
                "reference_config_sha256": reference["active_config_sha256"],
                "methods": method_values,
            }
        )

    for scene in SCENES:
        reference_hashes = {
            str(row["reference_image_sha256"])
            for row in rows
            if row["scene"] == scene
        }
        if len(reference_hashes) != 1:
            raise RuntimeError(f"reference image changed across A/B runs for {scene}")
    return sorted(rows, key=lambda row: (str(row["scene"]), int(row["seed"]), bool(row["flip"])))


def paired_summary(ratios: list[float]) -> dict[str, object]:
    return {
        "pair_count": len(ratios),
        "geometric_mean_ratio": geometric_mean(ratios),
        "wins": sum(ratio < 1.0 for ratio in ratios),
        "win_rate": sum(ratio < 1.0 for ratio in ratios) / len(ratios),
    }


def build_report(input_root: Path) -> dict[str, object]:
    runs = load_runs(input_root)
    config_paths = sorted(
        (WORKSPACE_ROOT / "configs").glob("phase_13_hemisphere_ab_*_macos.json")
    )
    expected_config_count = len(SCENES) * len(SEEDS) * 2
    if len(config_paths) != expected_config_count:
        raise RuntimeError(
            f"expected {expected_config_count} reproducible configs, "
            f"found {len(config_paths)}"
        )
    by_key = {
        (str(row["scene"]), int(row["seed"]), bool(row["flip"])): row
        for row in runs
    }

    comparisons: list[dict[str, object]] = []
    for scene in SCENES:
        for seed in SEEDS:
            flip_row = by_key[(scene, seed, True)]
            noflip_row = by_key[(scene, seed, False)]
            method_ratios = {
                method: (
                    float(noflip_row["methods"][method]["mse"])
                    / float(flip_row["methods"][method]["mse"])
                )
                for method in METHODS
            }
            noflip_diffusion_simple_ratio = (
                float(noflip_row["methods"][METHODS[0]]["mse"])
                / float(noflip_row["methods"][METHODS[1]]["mse"])
            )
            flip_diffusion_simple_ratio = (
                float(flip_row["methods"][METHODS[0]]["mse"])
                / float(flip_row["methods"][METHODS[1]]["mse"])
            )
            comparisons.append(
                {
                    "scene": scene,
                    "seed": seed,
                    "flip_mse": {
                        method: float(flip_row["methods"][method]["mse"])
                        for method in METHODS
                    },
                    "noflip_mse": {
                        method: float(noflip_row["methods"][method]["mse"])
                        for method in METHODS
                    },
                    "noflip_over_flip_ratio": method_ratios,
                    "flip_diffusion_over_simple_ratio": flip_diffusion_simple_ratio,
                    "noflip_diffusion_over_simple_ratio": noflip_diffusion_simple_ratio,
                    "interaction_ratio": (
                        noflip_diffusion_simple_ratio / flip_diffusion_simple_ratio
                    ),
                }
            )

    by_scene: dict[str, object] = {}
    for scene in SCENES:
        scene_pairs = [row for row in comparisons if row["scene"] == scene]
        by_scene[scene] = {
            "noflip_vs_flip": {
                method: paired_summary(
                    [float(row["noflip_over_flip_ratio"][method]) for row in scene_pairs]
                )
                for method in METHODS
            },
            "noflip_diffusion_vs_simple": paired_summary(
                [
                    float(row["noflip_diffusion_over_simple_ratio"])
                    for row in scene_pairs
                ]
            ),
            "flip_diffusion_vs_simple": paired_summary(
                [
                    float(row["flip_diffusion_over_simple_ratio"])
                    for row in scene_pairs
                ]
            ),
            "interaction_noflip_over_flip_relative_ratio": paired_summary(
                [float(row["interaction_ratio"]) for row in scene_pairs]
            ),
        }

    pooled = {
        "noflip_vs_flip": {
            method: paired_summary(
                [float(row["noflip_over_flip_ratio"][method]) for row in comparisons]
            )
            for method in METHODS
        },
        "noflip_diffusion_vs_simple": paired_summary(
            [
                float(row["noflip_diffusion_over_simple_ratio"])
                for row in comparisons
            ]
        ),
        "flip_diffusion_vs_simple": paired_summary(
            [float(row["flip_diffusion_over_simple_ratio"]) for row in comparisons]
        ),
        "interaction_noflip_over_flip_relative_ratio": paired_summary(
            [float(row["interaction_ratio"]) for row in comparisons]
        ),
    }
    return {
        "status": "development_screening_three_seeds_per_scene",
        "protocol": {
            "scenes": list(SCENES),
            "seeds": list(SEEDS),
            "reference_seed": 99173,
            "candidate_spp": 64,
            "reference_spp": 1024,
            "bounce_alpha": 0.5,
            "learned_direction_kappa_scale": 1.0,
            "flip_values": [True, False],
            "methods": list(METHODS),
            "config_paths": [
                path.relative_to(WORKSPACE_ROOT).as_posix() for path in config_paths
            ],
            "runtime_loaded_verified": True,
            "runtime_native_control_marker_verified": True,
            "patched_xml_controls_verified": True,
            "caps": {
                "inverse_probability": 512.0,
                "contribution": 32.0,
                "max_trials": 2048,
            },
        },
        "runs": runs,
        "paired_rows": comparisons,
        "aggregate_by_scene": by_scene,
        "aggregate_pooled": pooled,
        "decision": {
            "mse_selected_diffusion_flip": True,
            "backward_compatible_default_remains_flip": True,
            "correctness_consistent_setting": "noflip",
            "recommendation": (
                "Keep flip=true as the compatibility/default and current MSE choice, "
                "but use noflip as the correctness branch in the next controlled "
                "experiment unless the folded density q(w)+q(-w) is implemented."
            ),
            "density_caveat": (
                "With flip=true, sampling folds negative directions into the receiver "
                "hemisphere while pdf_direction still evaluates the original, unfolded "
                "vMF density. The current Bernoulli re-hit estimator does not call "
                "pdf_direction directly, but the API pair is not a matched explicit "
                "proposal density under the fold."
            ),
        },
    }


def percent(ratio: float) -> str:
    return f"{100.0 * (ratio - 1.0):+.3f}%"


def write_markdown(report: dict[str, object], output_path: Path) -> None:
    lines = [
        "# Learned-direction Hemisphere Flip A/B",
        "",
        "Development screening on Lamp and Slabs with three paired render seeds.",
        "`noflip/flip < 1` means disabling the legacy receiver-hemisphere fold",
        "reduced MSE; `D/S < 1` means diffusion beat simple MLP; interaction `< 1`",
        "means no-flip improved diffusion relative to simple MLP. All runtime loads,",
        "native control markers, XML controls, and",
        "fixed-reference hashes were verified by the summarizer.",
        "",
    ]
    for scene in SCENES:
        lines.extend(
            [
                f"## {scene}",
                "",
                "| seed | diffusion flip | diffusion noflip | D noflip/flip | simple flip | simple noflip | S noflip/flip | flip D/S | noflip D/S | interaction |",
                "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in report["paired_rows"]:
            if row["scene"] != scene:
                continue
            d_flip = row["flip_mse"][METHODS[0]]
            d_noflip = row["noflip_mse"][METHODS[0]]
            s_flip = row["flip_mse"][METHODS[1]]
            s_noflip = row["noflip_mse"][METHODS[1]]
            lines.append(
                f"| {row['seed']} | {d_flip:.9f} | {d_noflip:.9f} | "
                f"{row['noflip_over_flip_ratio'][METHODS[0]]:.6f} | "
                f"{s_flip:.9f} | {s_noflip:.9f} | "
                f"{row['noflip_over_flip_ratio'][METHODS[1]]:.6f} | "
                f"{row['flip_diffusion_over_simple_ratio']:.6f} | "
                f"{row['noflip_diffusion_over_simple_ratio']:.6f} | "
                f"{row['interaction_ratio']:.6f} |"
            )
        lines.append("")

    lines.extend(
        [
            "## Paired aggregates",
            "",
            "| scope | comparison | geometric ratio | relative MSE | wins | win rate |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    scopes = {**report["aggregate_by_scene"], "pooled": report["aggregate_pooled"]}
    for scope, aggregate in scopes.items():
        for method in METHODS:
            values = aggregate["noflip_vs_flip"][method]
            lines.append(
                f"| {scope} | {method}: noflip / flip | "
                f"{values['geometric_mean_ratio']:.6f} | "
                f"{percent(values['geometric_mean_ratio'])} | "
                f"{values['wins']}/{values['pair_count']} | {values['win_rate']:.1%} |"
            )
        values = aggregate["noflip_diffusion_vs_simple"]
        lines.append(
            f"| {scope} | noflip: diffusion / simple | "
            f"{values['geometric_mean_ratio']:.6f} | "
            f"{percent(values['geometric_mean_ratio'])} | "
            f"{values['wins']}/{values['pair_count']} | {values['win_rate']:.1%} |"
        )
        values = aggregate["flip_diffusion_vs_simple"]
        lines.append(
            f"| {scope} | flip: diffusion / simple | "
            f"{values['geometric_mean_ratio']:.6f} | "
            f"{percent(values['geometric_mean_ratio'])} | "
            f"{values['wins']}/{values['pair_count']} | {values['win_rate']:.1%} |"
        )
        values = aggregate["interaction_noflip_over_flip_relative_ratio"]
        lines.append(
            f"| {scope} | interaction: (noflip D/S) / (flip D/S) | "
            f"{values['geometric_mean_ratio']:.6f} | "
            f"{percent(values['geometric_mean_ratio'])} | "
            f"{values['wins']}/{values['pair_count']} | {values['win_rate']:.1%} |"
        )

    lines.extend(
        [
            "",
            "## Decision",
            "",
            "- MSE selection for the diffusion model keeps `flip=true`; disabling it",
            "  did not improve absolute diffusion MSE on this screen.",
            "- Keep `flip=true` as the backward-compatible default.",
            "- For a correctness-aligned explicit proposal, prefer `flip=false` in the",
            "  next controlled experiment, unless `pdf_direction` is changed to the",
            "  folded density `q(w) + q(-w)` on the retained hemisphere.",
            "- The current renderer estimates proposal probability through Bernoulli",
            "  re-hits rather than calling `pdf_direction` directly, but the flip=true",
            "  sampler and the exported unfurled vMF density are still not a matched",
            "  explicit sampling/PDF pair.",
            "",
            "This remains a development result. The chosen setting requires a",
            "fresh-seed confirmation before any paper claim; Ring/Pillars were not",
            "used for tuning. The configured clamps remain active, so these runs are",
            "practical biased-renderer screening rather than an unbiased estimator test.",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    input_root = args.input_root.resolve()
    output_json = (
        input_root / "hemisphere_ab_summary.json"
        if args.output_json is None
        else args.output_json.resolve()
    )
    output_markdown = (
        input_root / "hemisphere_ab_summary.md"
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
