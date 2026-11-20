#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = WORKSPACE_ROOT / "src"
if str(SRC_ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(SRC_ROOT))

from masked_diffusion_mpg.dataset import ReplayTrainingDataset  # noqa: E402
from masked_diffusion_mpg.model import (  # noqa: E402
    DIRECTION_DISTRIBUTION_SPHERICAL_D3PM_VMF,
    MaskedSeedDiffusionModel,
    build_ordered_direction_inputs,
    predict_spherical_d3pm_endpoint,
)
from masked_diffusion_mpg.training import (  # noqa: E402
    MaskedDiffusionTrainingConfig,
    build_model_config,
    compute_vmf_mixture_nll,
    split_dataset_indices,
)


@dataclass(frozen=True)
class OfflineMetrics:
    endpoint_vmf_nll: float
    direction_mse: float
    mean_cosine: float
    effective_k: float
    num_validation_records: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate ordered vMF and spherical-D3PM checkpoints on one frozen "
            "ground-truth-descriptor validation split."
        )
    )
    parser.add_argument(
        "--spec",
        type=Path,
        required=True,
        help="JSON manifest containing output paths and model checkpoint entries.",
    )
    parser.add_argument(
        "--workspace-root",
        type=Path,
        default=WORKSPACE_ROOT,
        help="Masked-diffusion workspace root.",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    return parser.parse_args()


def resolve_path(workspace_root: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (workspace_root / path).resolve()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_checkpoint(
    path: Path,
    *,
    device: torch.device,
) -> tuple[MaskedSeedDiffusionModel, MaskedDiffusionTrainingConfig, dict[str, Any]]:
    payload = torch.load(path, map_location=device)
    config = MaskedDiffusionTrainingConfig(**payload["config"])
    model = MaskedSeedDiffusionModel(build_model_config(config)).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model, config, payload


def validate_shared_protocol(configs: list[MaskedDiffusionTrainingConfig]) -> None:
    if not configs:
        raise ValueError("the report spec contains no models")
    reference = configs[0]
    fields = ("replay_paths", "seed", "val_fraction", "max_records")
    for index, config in enumerate(configs[1:], start=1):
        for field in fields:
            if getattr(config, field) != getattr(reference, field):
                raise ValueError(
                    f"model {index} has a different {field}; a shared validation "
                    "split cannot be claimed"
                )


def best_epoch_metadata(run_dir: Path) -> dict[str, float | int | None]:
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        return {"best_epoch": None, "best_logged_val_loss": None}
    rows = json.loads(metrics_path.read_text(encoding="utf-8"))
    candidates = [row for row in rows if row.get("val_loss") is not None]
    if not candidates:
        return {"best_epoch": None, "best_logged_val_loss": None}
    best = min(candidates, key=lambda row: float(row["val_loss"]))
    return {
        "best_epoch": int(best["epoch"]),
        "best_logged_val_loss": float(best["val_loss"]),
    }


@torch.no_grad()
def evaluate_direction_endpoint(
    *,
    model: MaskedSeedDiffusionModel,
    config: MaskedDiffusionTrainingConfig,
    loader: DataLoader[dict[str, torch.Tensor]],
    device: torch.device,
) -> OfflineMetrics:
    endpoint_nll: list[torch.Tensor] = []
    direction_mse: list[torch.Tensor] = []
    cosine: list[torch.Tensor] = []
    effective_k: list[torch.Tensor] = []

    for batch in loader:
        batch = {key: value.to(device) for key, value in batch.items()}
        if (
            config.direction_distribution
            == DIRECTION_DISTRIBUTION_SPHERICAL_D3PM_VMF
        ):
            outputs = predict_spherical_d3pm_endpoint(
                model,
                condition=batch["condition"],
                chain_length=batch["chain_length"],
                chain_type_bits=batch["chain_type_bits"],
            )
            max_kappa = max(
                float(config.vmf_max_kappa),
                float(config.direction_grid_kappa),
            )
        else:
            outputs = model(
                **build_ordered_direction_inputs(
                    condition=batch["condition"],
                    chain_length=batch["chain_length"],
                    chain_type_bits=batch["chain_type_bits"],
                )
            )
            max_kappa = float(config.vmf_max_kappa)

        target = F.normalize(batch["direction"].float(), dim=1, eps=1e-6)
        prediction = F.normalize(outputs["direction"].float(), dim=1, eps=1e-6)
        endpoint_nll.append(
            compute_vmf_mixture_nll(
                outputs,
                target,
                vmf_max_kappa=max_kappa,
            ).cpu()
        )
        direction_mse.append((prediction - target).square().mean(dim=1).cpu())
        cosine.append((prediction * target).sum(dim=1).clamp(-1.0, 1.0).cpu())
        weights = torch.softmax(outputs["vmf_logits"].float(), dim=1)
        entropy = -(weights * torch.log(weights.clamp_min(1e-30))).sum(dim=1)
        effective_k.append(torch.exp(entropy).cpu())

    nll = torch.cat(endpoint_nll)
    mse = torch.cat(direction_mse)
    cos = torch.cat(cosine)
    eff_k = torch.cat(effective_k)
    return OfflineMetrics(
        endpoint_vmf_nll=float(nll.mean()),
        direction_mse=float(mse.mean()),
        mean_cosine=float(cos.mean()),
        effective_k=float(eff_k.mean()),
        num_validation_records=int(nll.numel()),
    )


def build_conclusion(models: list[dict[str, Any]]) -> dict[str, Any]:
    baseline = next(item for item in models if item["role"] == "ordered_k4_baseline")
    aligned_t1 = next(
        item for item in models if item["role"] == "aligned_single_step_control"
    )
    aligned_multistep = sorted(
        (
            item
            for item in models
            if item["role"] == "aligned_multistep_candidate"
        ),
        key=lambda item: int(item["diffusion_steps"]),
    )

    def compare(candidate: dict[str, Any], reference: dict[str, Any]) -> dict[str, Any]:
        mse_gain = 1.0 - candidate["direction_mse"] / reference["direction_mse"]
        nll_improvement = (
            reference["endpoint_vmf_nll"] - candidate["endpoint_vmf_nll"]
        )
        return {
            "reference": reference["label"],
            "mse_gain_fraction": mse_gain,
            "nll_improvement": nll_improvement,
            "better_on_both_metrics": mse_gain > 0.0 and nll_improvement > 0.0,
        }

    aligned_gates: list[dict[str, Any]] = []
    for candidate in aligned_multistep:
        comparisons = {
            "vs_aligned_t1": compare(candidate, aligned_t1),
            "vs_ordered_k4": compare(candidate, baseline),
        }
        aligned_gates.append(
            {
                "candidate": candidate["label"],
                "diffusion_steps": candidate["diffusion_steps"],
                "comparisons": comparisons,
                "passes_native_gate": all(
                    value["better_on_both_metrics"]
                    for value in comparisons.values()
                ),
            }
        )
    passing = [item for item in aligned_gates if item["passes_native_gate"]]

    fixed_t1 = next(
        item for item in models if item["role"] == "single_step_control_low_kl"
    )
    fixed_low_kl = sorted(
        (
            item
            for item in models
            if item["role"] == "multistep_candidate"
            and item.get("direction_state_geometry") == "fixed_anchor"
            and item.get("direction_kappa") == 64.0
            and item.get("denoising_weight") == 0.05
        ),
        key=lambda item: int(item["diffusion_steps"]),
    )
    fixed_low_kl_comparisons = [
        {
            "candidate": candidate["label"],
            "diffusion_steps": candidate["diffusion_steps"],
            "vs_fixed_t1": compare(candidate, fixed_t1),
        }
        for candidate in fixed_low_kl
    ]

    best_endpoint = min(models, key=lambda item: item["endpoint_vmf_nll"])
    best_mse = min(models, key=lambda item: item["direction_mse"])
    return {
        "t1_is_multistep_diffusion": False,
        "best_endpoint_nll": best_endpoint["label"],
        "best_direction_mse": best_mse["label"],
        "aligned_t1_control": aligned_t1["label"],
        "aligned_candidate_gates": aligned_gates,
        "fixed_low_kl_comparisons": fixed_low_kl_comparisons,
        "native_offline_gate_passed": bool(passing),
        "passing_aligned_candidates": [item["candidate"] for item in passing],
        "recommendation": (
            "Recommend a minimal native prototype: at least one aligned T>1 "
            "candidate beats both its matched aligned T1 control and ordered K4 "
            "on endpoint NLL and direction MSE."
            if passing
            else "Reject native multistep implementation at this stage. Neither "
            "aligned T2 nor aligned T4 beats both its matched aligned T1 control "
            "and ordered K4 simultaneously on endpoint NLL and direction MSE."
        ),
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Slabs spherical D3PM offline comparison",
        "",
        "All rows use the exact same frozen validation indices and ground-truth "
        "length/type descriptors. Direction MSE is the mean of three coordinate-wise "
        "squared errors. Effective K is `exp(entropy(mixture weights))`, averaged per record.",
        "",
        f"- Validation records: {report['protocol']['num_validation_records']}",
        f"- Split fingerprint: `{report['protocol']['validation_index_sha256']}`",
        f"- Replay SHA-256: `{report['protocol']['replay_files'][0]['sha256']}`",
        "- T1 is a single-step control, not genuine multistep diffusion.",
        "",
        "| Model | Geometry | T | kappa | KL weight | Params | Best epoch | Logged best val | Endpoint NLL | Direction MSE | Mean cosine | Effective K |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in report["models"]:
        lines.append(
            "| {label} | {geometry} | {steps} | {kappa} | {kl_weight} | {params:,} | "
            "{best_epoch} | {best_val:.6f} | {nll:.6f} | {mse:.8f} | "
            "{cosine:.6f} | {effective_k:.4f} |".format(
                label=item["label"],
                geometry=item.get("direction_state_geometry") or "ordered",
                steps=item.get("diffusion_steps") or "-",
                kappa=item.get("direction_kappa") or "-",
                kl_weight=(
                    "-"
                    if item.get("denoising_weight") is None
                    else f"{item['denoising_weight']:.2f}"
                ),
                params=item["parameter_count"],
                best_epoch=item["best_epoch"],
                best_val=item["best_logged_val_loss"],
                nll=item["endpoint_vmf_nll"],
                mse=item["direction_mse"],
                cosine=item["mean_cosine"],
                effective_k=item["effective_k"],
            )
        )
    conclusion = report["conclusion"]
    lines.extend(
        [
            "",
            "## Decision",
            "",
            conclusion["recommendation"],
            "",
            f"- Best endpoint NLL row: `{conclusion['best_endpoint_nll']}`.",
            f"- Best direction MSE row: `{conclusion['best_direction_mse']}`.",
            "- Required aligned-geometry native gates:",
            *[
                "  - {candidate}: versus aligned T1 MSE/NLL = {t1_mse:.2f}% / "
                "{t1_nll:.6f}; versus ordered K4 = {base_mse:.2f}% / "
                "{base_nll:.6f}; pass = `{passes}`.".format(
                    candidate=gate["candidate"],
                    t1_mse=100.0
                    * gate["comparisons"]["vs_aligned_t1"]["mse_gain_fraction"],
                    t1_nll=gate["comparisons"]["vs_aligned_t1"]["nll_improvement"],
                    base_mse=100.0
                    * gate["comparisons"]["vs_ordered_k4"]["mse_gain_fraction"],
                    base_nll=gate["comparisons"]["vs_ordered_k4"]["nll_improvement"],
                    passes=str(gate["passes_native_gate"]).lower(),
                )
                for gate in conclusion["aligned_candidate_gates"]
            ],
            "- Fixed-anchor matched KL=.05 controls (historical context):",
            *[
                "  - {candidate} versus fixed T1: MSE gain {mse:.2f}%, "
                "NLL improvement {nll:.6f}, better on both = `{both}`.".format(
                    candidate=item["candidate"],
                    mse=100.0
                    * item["vs_fixed_t1"]["mse_gain_fraction"],
                    nll=item["vs_fixed_t1"]["nll_improvement"],
                    both=str(
                        item["vs_fixed_t1"]["better_on_both_metrics"]
                    ).lower(),
                )
                for item in conclusion["fixed_low_kl_comparisons"]
            ],
            "",
            "Logged best validation loss is the run's combined training objective; "
            "it is shown for checkpoint provenance, not used to compare unlike loss weights.",
            "Offline direction metrics are a screening gate and do not establish lower rendering MSE.",
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

    loaded: list[
        tuple[dict[str, Any], MaskedSeedDiffusionModel, MaskedDiffusionTrainingConfig]
    ] = []
    for entry in spec["models"]:
        checkpoint_path = resolve_path(workspace_root, entry["checkpoint"])
        run_dir = resolve_path(workspace_root, entry["run_dir"])
        model, config, _payload = load_checkpoint(checkpoint_path, device=device)
        loaded.append((dict(entry, checkpoint_path=str(checkpoint_path), run_dir=str(run_dir)), model, config))

    validate_shared_protocol([config for _entry, _model, config in loaded])
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
        epoch_metadata = best_epoch_metadata(Path(entry["run_dir"]))
        row = {
            "label": entry["label"],
            "role": entry["role"],
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
            "direction_kappa": (
                config.direction_grid_kappa
                if config.direction_distribution
                == DIRECTION_DISTRIBUTION_SPHERICAL_D3PM_VMF
                else config.vmf_max_kappa
            ),
            "denoising_weight": (
                config.direction_denoising_weight
                if config.direction_distribution
                == DIRECTION_DISTRIBUTION_SPHERICAL_D3PM_VMF
                else None
            ),
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            **epoch_metadata,
            **asdict(metrics),
        }
        model_rows.append(row)

    report = {
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
            "direction_mse_definition": "mean((prediction-target)^2 over xyz and records)",
            "effective_k_definition": "mean(exp(entropy(softmax(vmf_logits))))",
        },
        "models": model_rows,
    }
    report["conclusion"] = build_conclusion(model_rows)

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
