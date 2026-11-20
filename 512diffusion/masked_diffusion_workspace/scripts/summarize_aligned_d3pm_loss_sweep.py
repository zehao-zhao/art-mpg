#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Subset


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = WORKSPACE_ROOT / "src"
if str(SRC_ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(SRC_ROOT))

from masked_diffusion_mpg.dataset import ReplayTrainingDataset  # noqa: E402
from masked_diffusion_mpg.model import (  # noqa: E402
    DIRECTION_DISTRIBUTION_SPHERICAL_D3PM_VMF,
    SPHERICAL_D3PM_GEOMETRY_LEARNED_EMISSION,
)
from masked_diffusion_mpg.training import split_dataset_indices  # noqa: E402
from summarize_spherical_d3pm_offline import (  # noqa: E402
    best_epoch_metadata,
    evaluate_direction_endpoint,
    load_checkpoint,
    resolve_path,
    sha256_file,
    validate_shared_protocol,
)


EXPECTED_WEIGHTS = (0.0, 0.01, 0.025, 0.05, 0.10)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize the pre-registered aligned spherical-D3PM loss sweep."
    )
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--workspace-root", type=Path, default=WORKSPACE_ROOT)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    return parser.parse_args()


def training_metadata(run_dir: Path) -> dict[str, float | int | str | None]:
    summary_path = run_dir / "run_summary.json"
    if not summary_path.exists():
        return {
            "training_seconds": None,
            "epochs_completed": None,
            "optimizer_steps": None,
            "training_device": None,
        }
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    return {
        "training_seconds": float(summary["training_seconds"]),
        "epochs_completed": int(summary["epochs_completed"]),
        "optimizer_steps": int(summary["optimizer_steps"]),
        "training_device": str(summary["device"]),
    }


def pareto_labels(rows: list[dict[str, Any]]) -> set[str]:
    """Return labels not dominated in both endpoint NLL and direction MSE."""

    frontier: set[str] = set()
    for candidate in rows:
        dominated = any(
            other is not candidate
            and other["endpoint_vmf_nll"] <= candidate["endpoint_vmf_nll"]
            and other["direction_mse"] <= candidate["direction_mse"]
            and (
                other["endpoint_vmf_nll"] < candidate["endpoint_vmf_nll"]
                or other["direction_mse"] < candidate["direction_mse"]
            )
            for other in rows
        )
        if not dominated:
            frontier.add(str(candidate["label"]))
    return frontier


def build_preregistered_conclusion(models: list[dict[str, Any]]) -> dict[str, Any]:
    """Apply the rule fixed before observing the new sweep results.

    The T1 reference is a coordinate-wise oracle over its five weights.  A
    *single* T4 row must strictly beat both oracle coordinates and both ordered
    K4 coordinates.  This deliberately prevents selecting a weak T1 weight or
    mixing two different T4 weights after seeing the results.
    """

    baseline = next(row for row in models if row["role"] == "ordered_k4_baseline")
    t1_rows = [row for row in models if row["role"] == "aligned_t1_sweep"]
    t4_rows = [row for row in models if row["role"] == "aligned_t4_sweep"]
    if not t1_rows or not t4_rows:
        raise ValueError("both aligned T1 and aligned T4 sweep rows are required")

    best_t1_nll = min(t1_rows, key=lambda row: row["endpoint_vmf_nll"])
    best_t1_mse = min(t1_rows, key=lambda row: row["direction_mse"])
    best_t4_nll = min(t4_rows, key=lambda row: row["endpoint_vmf_nll"])
    best_t4_mse = min(t4_rows, key=lambda row: row["direction_mse"])
    t1_nll_threshold = float(best_t1_nll["endpoint_vmf_nll"])
    t1_mse_threshold = float(best_t1_mse["direction_mse"])
    baseline_nll = float(baseline["endpoint_vmf_nll"])
    baseline_mse = float(baseline["direction_mse"])

    candidate_gates: list[dict[str, Any]] = []
    for row in sorted(t4_rows, key=lambda item: item["denoising_weight"]):
        beats_t1_nll = row["endpoint_vmf_nll"] < t1_nll_threshold
        beats_t1_mse = row["direction_mse"] < t1_mse_threshold
        beats_baseline_nll = row["endpoint_vmf_nll"] < baseline_nll
        beats_baseline_mse = row["direction_mse"] < baseline_mse
        candidate_gates.append(
            {
                "label": row["label"],
                "denoising_weight": row["denoising_weight"],
                "beats_t1_oracle_nll": beats_t1_nll,
                "beats_t1_oracle_mse": beats_t1_mse,
                "beats_ordered_k4_nll": beats_baseline_nll,
                "beats_ordered_k4_mse": beats_baseline_mse,
                "passes": (
                    beats_t1_nll
                    and beats_t1_mse
                    and beats_baseline_nll
                    and beats_baseline_mse
                ),
                "nll_margin_vs_t1_oracle": (
                    t1_nll_threshold - row["endpoint_vmf_nll"]
                ),
                "mse_gain_vs_t1_oracle_fraction": (
                    1.0 - row["direction_mse"] / t1_mse_threshold
                ),
                "nll_margin_vs_ordered_k4": (
                    baseline_nll - row["endpoint_vmf_nll"]
                ),
                "mse_gain_vs_ordered_k4_fraction": (
                    1.0 - row["direction_mse"] / baseline_mse
                ),
            }
        )

    passing = [row for row in candidate_gates if row["passes"]]
    gate_passed = bool(passing)
    return {
        "selection_rule": (
            "T1 coordinate-wise oracle over five pre-registered weights; one T4 "
            "weight must strictly beat the oracle T1 NLL and MSE and ordered K4 "
            "NLL and MSE."
        ),
        "t1_oracle_endpoint_nll": {
            "value": t1_nll_threshold,
            "label": best_t1_nll["label"],
            "denoising_weight": best_t1_nll["denoising_weight"],
        },
        "t1_oracle_direction_mse": {
            "value": t1_mse_threshold,
            "label": best_t1_mse["label"],
            "denoising_weight": best_t1_mse["denoising_weight"],
        },
        "best_t4_endpoint_nll": {
            "value": best_t4_nll["endpoint_vmf_nll"],
            "label": best_t4_nll["label"],
            "denoising_weight": best_t4_nll["denoising_weight"],
        },
        "best_t4_direction_mse": {
            "value": best_t4_mse["direction_mse"],
            "label": best_t4_mse["label"],
            "denoising_weight": best_t4_mse["denoising_weight"],
        },
        "ordered_k4": {
            "endpoint_vmf_nll": baseline_nll,
            "direction_mse": baseline_mse,
        },
        "t4_candidate_gates": candidate_gates,
        "native_gate_passed": gate_passed,
        "passing_t4_weights": [row["denoising_weight"] for row in passing],
        "recommendation": (
            "Proceed with a minimal native spherical-D3PM prototype."
            if gate_passed
            else "Permanently stop this spherical-D3PM direction and do not "
            "implement it in the native renderer."
        ),
    }


def validate_sweep(rows: list[tuple[dict[str, Any], Any, Any]]) -> None:
    d3pm_rows = [row for row in rows if row[0]["role"] != "ordered_k4_baseline"]
    for role, steps in (("aligned_t1_sweep", 1), ("aligned_t4_sweep", 4)):
        role_rows = [row for row in d3pm_rows if row[0]["role"] == role]
        observed = sorted(round(float(row[2].direction_denoising_weight), 6) for row in role_rows)
        expected = sorted(round(value, 6) for value in EXPECTED_WEIGHTS)
        if observed != expected:
            raise ValueError(f"{role} weights {observed} do not match {expected}")
        for _entry, _model, config in role_rows:
            if config.direction_distribution != DIRECTION_DISTRIBUTION_SPHERICAL_D3PM_VMF:
                raise ValueError(f"{role} contains a non-spherical-D3PM model")
            if config.direction_state_geometry != SPHERICAL_D3PM_GEOMETRY_LEARNED_EMISSION:
                raise ValueError(f"{role} contains non-aligned state geometry")
            expected_fields = {
                "direction_diffusion_steps": steps,
                "direction_states": 32,
                "direction_grid_kappa": 64.0,
                "hidden_dim": 68,
                "depth": 3,
                "epochs": 32,
                "seed": 42,
            }
            for field, expected_value in expected_fields.items():
                if getattr(config, field) != expected_value:
                    raise ValueError(
                        f"{role} has unmatched {field}: "
                        f"{getattr(config, field)!r} != {expected_value!r}"
                    )


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Slabs aligned spherical D3PM denoising-weight sweep",
        "",
        "This is the final bounded sweep. All rows use the same frozen validation "
        "indices and ground-truth length/type descriptors.",
        "",
        f"- Validation records: {report['protocol']['num_validation_records']}",
        f"- Split fingerprint: `{report['protocol']['validation_index_sha256']}`",
        f"- Replay SHA-256: `{report['protocol']['replay_files'][0]['sha256']}`",
        "- New eight-run training time: "
        f"{report['protocol']['new_8_runs_training_seconds']:.2f} seconds.",
        f"- Pre-registered rule: {report['conclusion']['selection_rule']}",
        "- T1 is a single-step control, not genuine multistep diffusion.",
        "",
        "| Model | T | KL weight | Pareto | Params | Train seconds | Best epoch | Logged best val | Endpoint NLL | Direction MSE | Mean cosine | Effective K |",
        "|---|---:|---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["models"]:
        lines.append(
            "| {label} | {steps} | {weight} | {pareto} | {params:,} | "
            "{seconds} | {epoch} | {val} | {nll:.6f} | {mse:.8f} | "
            "{cosine:.6f} | {effective_k:.4f} |".format(
                label=row["label"],
                steps=row.get("diffusion_steps") or "-",
                weight=(
                    "-"
                    if row.get("denoising_weight") is None
                    else f"{row['denoising_weight']:.3f}"
                ),
                pareto=("yes" if row.get("pareto_optimal") else "-"),
                params=row["parameter_count"],
                seconds=(
                    "-"
                    if row.get("training_seconds") is None
                    else f"{row['training_seconds']:.2f}"
                ),
                epoch=row.get("best_epoch") or "-",
                val=(
                    "-"
                    if row.get("best_logged_val_loss") is None
                    else f"{row['best_logged_val_loss']:.6f}"
                ),
                nll=row["endpoint_vmf_nll"],
                mse=row["direction_mse"],
                cosine=row["mean_cosine"],
                effective_k=row["effective_k"],
            )
        )

    conclusion = report["conclusion"]
    lines.extend(
        [
            "",
            "## Pre-registered decision",
            "",
            conclusion["recommendation"],
            "",
            "- T1 oracle NLL: {value:.6f} at weight {weight:.3f}.".format(
                value=conclusion["t1_oracle_endpoint_nll"]["value"],
                weight=conclusion["t1_oracle_endpoint_nll"]["denoising_weight"],
            ),
            "- T1 oracle MSE: {value:.8f} at weight {weight:.3f}.".format(
                value=conclusion["t1_oracle_direction_mse"]["value"],
                weight=conclusion["t1_oracle_direction_mse"]["denoising_weight"],
            ),
            "- T4 gate rows:",
            *[
                "  - weight {weight:.3f}: T1 NLL/MSE pass `{nll}`/`{mse}`; "
                "ordered K4 NLL/MSE pass `{base_nll}`/`{base_mse}`; overall `{overall}`.".format(
                    weight=gate["denoising_weight"],
                    nll=str(gate["beats_t1_oracle_nll"]).lower(),
                    mse=str(gate["beats_t1_oracle_mse"]).lower(),
                    base_nll=str(gate["beats_ordered_k4_nll"]).lower(),
                    base_mse=str(gate["beats_ordered_k4_mse"]).lower(),
                    overall=str(gate["passes"]).lower(),
                )
                for gate in conclusion["t4_candidate_gates"]
            ],
            "",
            "Pareto status is computed separately within T1 and T4 over endpoint "
            "NLL and direction MSE. Logged validation loss includes the selected "
            "KL weight and is not used across weights for the gate.",
            "Offline direction metrics remain a screening gate; they are not "
            "rendering-MSE evidence.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    workspace_root = args.workspace_root.resolve()
    spec_path = args.spec.resolve()
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    device = torch.device(args.device)

    loaded = []
    for entry in spec["models"]:
        checkpoint_path = resolve_path(workspace_root, entry["checkpoint"])
        run_dir = resolve_path(workspace_root, entry["run_dir"])
        model, config, _payload = load_checkpoint(checkpoint_path, device=device)
        loaded.append(
            (
                dict(
                    entry,
                    checkpoint_path=str(checkpoint_path),
                    run_dir=str(run_dir),
                ),
                model,
                config,
            )
        )

    validate_shared_protocol([config for _entry, _model, config in loaded])
    validate_sweep(loaded)
    protocol_config = loaded[0][2]
    replay_paths = tuple(
        resolve_path(workspace_root, value) for value in protocol_config.replay_paths
    )
    dataset = ReplayTrainingDataset(replay_paths, max_records=protocol_config.max_records)
    _train_indices, val_indices = split_dataset_indices(
        len(dataset),
        val_fraction=protocol_config.val_fraction,
        seed=protocol_config.seed,
        group_keys=dataset.group_keys,
    )
    loader = DataLoader(
        Subset(dataset, val_indices),
        batch_size=min(args.batch_size, len(val_indices)),
        shuffle=False,
        num_workers=0,
    )
    val_fingerprint = hashlib.sha256(
        ",".join(str(value) for value in val_indices).encode("ascii")
    ).hexdigest()

    model_rows: list[dict[str, Any]] = []
    for entry, model, config in loaded:
        metrics = evaluate_direction_endpoint(
            model=model,
            config=config,
            loader=loader,
            device=device,
        )
        row = {
            "label": entry["label"],
            "role": entry["role"],
            "new_run": bool(entry.get("new_run", False)),
            "checkpoint": entry["checkpoint_path"],
            "run_dir": entry["run_dir"],
            "direction_distribution": config.direction_distribution,
            "direction_state_geometry": (
                config.direction_state_geometry
                if config.direction_distribution
                == DIRECTION_DISTRIBUTION_SPHERICAL_D3PM_VMF
                else None
            ),
            "diffusion_steps": (
                config.direction_diffusion_steps
                if config.direction_distribution
                == DIRECTION_DISTRIBUTION_SPHERICAL_D3PM_VMF
                else None
            ),
            "denoising_weight": (
                config.direction_denoising_weight
                if config.direction_distribution
                == DIRECTION_DISTRIBUTION_SPHERICAL_D3PM_VMF
                else None
            ),
            "parameter_count": sum(
                parameter.numel() for parameter in model.parameters()
            ),
            **best_epoch_metadata(Path(entry["run_dir"])),
            **training_metadata(Path(entry["run_dir"])),
            **asdict(metrics),
        }
        model_rows.append(row)

    for role in ("aligned_t1_sweep", "aligned_t4_sweep"):
        role_rows = [row for row in model_rows if row["role"] == role]
        frontier = pareto_labels(role_rows)
        for row in role_rows:
            row["pareto_optimal"] = row["label"] in frontier
    for row in model_rows:
        row.setdefault("pareto_optimal", None)

    sweep_seconds = sum(
        float(row["training_seconds"])
        for row in model_rows
        if row["role"] in {"aligned_t1_sweep", "aligned_t4_sweep"}
        and row["training_seconds"] is not None
    )
    new_run_seconds = sum(
        float(row["training_seconds"])
        for row in model_rows
        if row["new_run"] and row["training_seconds"] is not None
    )
    report: dict[str, Any] = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "spec": str(spec_path),
        "protocol": {
            "replay_paths": [str(path) for path in replay_paths],
            "replay_files": [
                {"path": str(path), "sha256": sha256_file(path)}
                for path in replay_paths
            ],
            "dataset_records": len(dataset),
            "num_validation_records": len(val_indices),
            "seed": protocol_config.seed,
            "val_fraction": protocol_config.val_fraction,
            "validation_index_sha256": val_fingerprint,
            "descriptor_conditioning": "ground_truth_length_and_type",
            "all_10_sweep_rows_training_seconds": sweep_seconds,
            "new_8_runs_training_seconds": new_run_seconds,
        },
        "models": model_rows,
    }
    report["curves"] = {
        role: [
            {
                key: row[key]
                for key in (
                    "label",
                    "denoising_weight",
                    "endpoint_vmf_nll",
                    "direction_mse",
                    "mean_cosine",
                    "effective_k",
                    "pareto_optimal",
                    "training_seconds",
                )
            }
            for row in sorted(
                (item for item in model_rows if item["role"] == role),
                key=lambda item: item["denoising_weight"],
            )
        ]
        for role in ("aligned_t1_sweep", "aligned_t4_sweep")
    }
    report["conclusion"] = build_preregistered_conclusion(model_rows)

    output_json = resolve_path(workspace_root, spec["output_json"])
    output_md = resolve_path(workspace_root, spec["output_markdown"])
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    output_md.write_text(render_markdown(report), encoding="utf-8")
    print(output_json)
    print(output_md)


if __name__ == "__main__":
    main()
