from __future__ import annotations

import csv
import json
import logging
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, Subset

from .dataset import (
    CONDITION_DIM,
    MASK_CHAIN_LENGTH_TOKEN,
    MASK_CHAIN_TYPE_TOKEN,
    MAX_CHAIN_TYPE_BITS,
    ReplayTrainingDataset,
)
from .model import (
    DISCRETE_DIFFUSION_ORDERED_ABSORBING,
    DIRECTION_DISTRIBUTION_SPHERICAL_D3PM_VMF,
    MaskedSeedDiffusionModel,
    build_ordered_direction_inputs,
    build_ordered_length_inputs,
    build_ordered_type_inputs,
    ordered_valid_length_log_probs,
    predict_spherical_d3pm_endpoint,
)
from .training import (
    EpochMetrics,
    MaskedDiffusionTrainingConfig,
    build_model_config,
    compute_losses,
    create_run_dir,
    detect_device,
    log_vmf_normalizer_3d,
    make_alpha_bar_schedule,
    make_progress,
    move_batch_to_device,
    sample_unit_directions,
    set_global_seed,
    split_dataset_indices,
)


LOGGER = logging.getLogger("masked_diffusion_mpg.comparison")


class ModelComparisonError(RuntimeError):
    """Expected failure surfaces for Phase 11 offline comparisons.

    Attributes:
        message: Caller-provided rationale (mirrors RuntimeError semantics).
    """


@dataclass(slots=True, frozen=True)
class SimpleBaselineConfig:
    """Hyper-parameters for the lightweight MLP ablation baseline.

    Attributes:
        hidden_dim: Hidden-unit width shared across plain MLP layers.
        depth: Number of plain hidden Linear-SiLU-Dropout layers.
        dropout: Dropout probability between expansions.
        epochs: Gradient steps across the whole dataset.
        learning_rate: AdamW base LR.
        weight_decay: AdamW decoupled L2 coefficient.
        grad_clip_norm: Optional maximum gradient norm applied before each update.
        vmf_components: Number of learned vMF components when probabilistic.
    """

    hidden_dim: int = 128
    depth: int = 3
    dropout: float = 0.1
    epochs: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip_norm: float | None = None
    probabilistic_direction: bool = False
    vmf_components: int = 1
    min_kappa: float = 1e-3
    max_kappa: float = 128.0


@dataclass(slots=True, frozen=True)
class ComparisonConfig:
    """Top-level knobs for Phase 11 experiment orchestration.

    Attributes:
        replay_paths: Replay JSONL paths relative to the workspace root.
        masked_diffusion_checkpoint: Checkpoint path consumed by Torch reload.
        output_dir: Subdirectory beneath workspace for artifacts.
        seed: RNG seed propagated through dataloaders.
        batch_size: Shared train/eval minibatch cardinality.
        val_fraction: Portion carved out for offline validation indices.
        max_records: Optional cap on streamed replay samples.
        num_workers: `DataLoader` worker processes.
        device: Torch device specifier or `"auto"` passthrough.
        direction_loss_weight: Scalar weight applied to cosine/MSE combos.
        diffusion_steps: Autoregressive denoise iterations for MPG model.
        simple_baseline: Nested config for supervised MLP comparator.
    """

    replay_paths: tuple[str, ...]
    masked_diffusion_checkpoint: str
    output_dir: str
    seed: int = 42
    batch_size: int = 8
    val_fraction: float = 0.25
    max_records: int | None = None
    num_workers: int = 4
    device: str = "auto"
    direction_loss_weight: float = 0.5
    diffusion_steps: int = 4
    stochastic_diffusion: bool = False
    diffusion_train_steps: int = 16
    beta_schedule: str = "linear"
    beta_start: float = 5e-4
    beta_end: float = 5e-2
    reverse_noise_scale: float = 1.0
    direction_loss_type: str = "vmf"
    direction_vmf_kappa: float = 1.0
    simple_baseline: SimpleBaselineConfig = SimpleBaselineConfig()


@dataclass(slots=True, frozen=True)
class EvaluationMetrics:
    """Normalized validation metrics surfaced for bookkeeping.

    Attributes:
        num_samples: Rows contributing to denominators below.
        reconstruction_loss: Aggregate reconstruction objective.
        chain_length_loss: Supervised categorical head loss.
        chain_type_loss: BCE-style supervision on discrete bits.
        direction_loss: Weighted cosine/MSE mixture for normals.
        chain_length_accuracy: Exact-match accuracy on quantized lengths.
        chain_type_bit_accuracy: Bit accuracy respecting validity masks only.
        direction_mse: Per-sample L2 drift on normalized vectors.
        direction_cosine: Mean cosine similarity for directions.
    """

    num_samples: int
    reconstruction_loss: float
    chain_length_loss: float
    chain_type_loss: float
    direction_loss: float
    chain_length_accuracy: float
    chain_type_bit_accuracy: float
    direction_mse: float
    direction_cosine: float


@dataclass(slots=True, frozen=True)
class ComparisonArtifacts:
    """Paths emitted after a comparative training/eval loop.

    Attributes:
        run_dir: Timestamped subdirectory per invocation.
        comparison_json: Machine-readable KPI dump.
        comparison_md: Human-readable Markdown appendix.
        simple_checkpoint: Best-effort Torch snapshot for supervised MLP.
        simple_history_json: JSON table of epoch metrics for simple model.
        simple_history_csv: Flattened epoch metrics for spreadsheets.
    """

    run_dir: Path
    comparison_json: Path
    comparison_md: Path
    simple_checkpoint: Path
    simple_history_json: Path
    simple_history_csv: Path


def load_comparison_config(config_path: Path) -> ComparisonConfig:
    """Parse JSON manifests into ComparisonConfig structs.

    Args:
        config_path: On-disk Phase 11 JSON blueprint.

    Returns:
        Hydrated ComparisonConfig immutable record.
    """

    payload = json.loads(config_path.read_text(encoding="utf-8"))
    simple_payload = payload.get("simple_baseline", {})
    return ComparisonConfig(
        replay_paths=tuple(str(item) for item in payload["replay_paths"]),
        masked_diffusion_checkpoint=str(payload["masked_diffusion_checkpoint"]),
        output_dir=str(payload["output_dir"]),
        seed=int(payload.get("seed", 42)),
        batch_size=int(payload.get("batch_size", 8)),
        val_fraction=float(payload.get("val_fraction", 0.25)),
        max_records=(
            None if payload.get("max_records") is None else int(payload["max_records"])
        ),
        num_workers=int(payload.get("num_workers", 4)),
        device=str(payload.get("device", "auto")),
        direction_loss_weight=float(payload.get("direction_loss_weight", 0.5)),
        diffusion_steps=int(payload.get("diffusion_steps", 4)),
        stochastic_diffusion=bool(payload.get("stochastic_diffusion", False)),
        diffusion_train_steps=int(payload.get("diffusion_train_steps", payload.get("diffusion_steps", 16))),
        beta_schedule=str(payload.get("beta_schedule", "linear")),
        beta_start=float(payload.get("beta_start", 5e-4)),
        beta_end=float(payload.get("beta_end", 5e-2)),
        reverse_noise_scale=float(payload.get("reverse_noise_scale", 1.0)),
        direction_loss_type=str(payload.get("direction_loss_type", "vmf")),
        direction_vmf_kappa=float(payload.get("direction_vmf_kappa", 1.0)),
        simple_baseline=SimpleBaselineConfig(
            hidden_dim=int(simple_payload.get("hidden_dim", 128)),
            depth=int(simple_payload.get("depth", 3)),
            dropout=float(simple_payload.get("dropout", 0.1)),
            epochs=int(simple_payload.get("epochs", 64)),
            learning_rate=float(simple_payload.get("learning_rate", 1e-3)),
            weight_decay=float(simple_payload.get("weight_decay", 1e-4)),
            grad_clip_norm=(
                None
                if simple_payload.get("grad_clip_norm") is None
                else float(simple_payload["grad_clip_norm"])
            ),
            probabilistic_direction=bool(simple_payload.get("probabilistic_direction", False)),
            vmf_components=int(simple_payload.get("vmf_components", 1)),
            min_kappa=float(simple_payload.get("min_kappa", 1e-3)),
            max_kappa=float(simple_payload.get("max_kappa", 128.0)),
        ),
    )


def resolve_masked_diffusion_checkpoint(
    *,
    workspace_root: Path,
    checkpoint_reference: str,
) -> Path:
    """Normalize workspace-relative checkpoints.

    Args:
        workspace_root: Masked-diffusion tooling root (`masked_diffusion_workspace`).
        checkpoint_reference: JSON-authored relative Torch bundle.

    Returns:
        Absolute filesystem checkpoint path guaranteed to exist.

    Raises:
        ModelComparisonError: Missing checkpoint on disk after resolution.
    """

    checkpoint_path = (workspace_root / checkpoint_reference).resolve()
    if not checkpoint_path.exists():
        raise ModelComparisonError(f"missing checkpoint: {checkpoint_path}")
    return checkpoint_path


class SimpleGuidanceMLP(nn.Module):
    """Plain supervised MLP mapping conditions onto seed descriptors.

    This baseline intentionally avoids residual refinement, layer normalization,
    timestep conditioning, and iterative denoising. It still keeps the same output
    heads as before, including optional learned single- or multi-component vMF
    direction heads.
    """

    def __init__(self, input_dim: int, config: SimpleBaselineConfig) -> None:
        """Construct a plain feed-forward MLP backbone plus prediction heads.

        Args:
            input_dim: Flattened condition vector dimensionality.
            config: Architectural hyper-parameters for the comparator. The existing
                ``depth`` field is kept unchanged and now controls the number of
                plain hidden Linear-SiLU-Dropout layers.
        """

        super().__init__()
        if config.vmf_components <= 0:
            raise ValueError("vmf_components must be positive")
        if config.min_kappa <= 0.0:
            raise ValueError("min_kappa must be positive")
        if config.max_kappa < config.min_kappa:
            raise ValueError("max_kappa must be greater than or equal to min_kappa")
        hidden_depth = max(1, int(config.depth))
        layers: list[nn.Module] = [
            nn.Linear(input_dim, config.hidden_dim),
            nn.SiLU(),
            nn.Dropout(config.dropout),
        ]
        for _ in range(hidden_depth - 1):
            layers.extend(
                [
                    nn.Linear(config.hidden_dim, config.hidden_dim),
                    nn.SiLU(),
                    nn.Dropout(config.dropout),
                ]
            )
        self.backbone = nn.Sequential(*layers)

        self.chain_length_head = nn.Linear(config.hidden_dim, MASK_CHAIN_LENGTH_TOKEN)
        self.chain_type_head = nn.Linear(config.hidden_dim, 9)
        self.probabilistic_direction = bool(config.probabilistic_direction)
        self.vmf_components = int(config.vmf_components)
        self.min_kappa = float(config.min_kappa)
        self.max_kappa = float(config.max_kappa)
        if self.probabilistic_direction:
            if self.vmf_components == 1:
                # Keep the original parameter names and shapes so all existing
                # probabilistic K=1 checkpoints load without migration.
                self.direction_head = nn.Linear(config.hidden_dim, 3)
                self.direction_kappa_head = nn.Linear(config.hidden_dim, 1)
            else:
                self.vmf_logits_head = nn.Linear(
                    config.hidden_dim,
                    self.vmf_components,
                )
                self.vmf_mean_head = nn.Linear(
                    config.hidden_dim,
                    self.vmf_components * 3,
                )
                self.vmf_kappa_head = nn.Linear(
                    config.hidden_dim,
                    self.vmf_components,
                )
        else:
            self.direction_head = nn.Linear(config.hidden_dim, 3)

    def forward(self, condition: torch.Tensor) -> dict[str, torch.Tensor]:
        """Single forward pass yielding logits analogous to MPG heads.

        Args:
            condition: Batched embeddings fed from replay buffers.

        Returns:
            Dict mirroring MPG tensor keys for downstream losses. Probabilistic
            K=1 retains the legacy ``direction_kappa`` key; K>1 emits the same
            ``vmf_logits``, ``vmf_means``, and ``vmf_kappas`` keys as the masked
            diffusion models.
        """

        hidden = self.backbone(condition)

        outputs = {
            "chain_length_logits": self.chain_length_head(hidden),
            "chain_type_logits": self.chain_type_head(hidden),
        }
        if self.probabilistic_direction:
            if self.vmf_components == 1:
                direction = F.normalize(self.direction_head(hidden), dim=1, eps=1e-6)
                raw_kappa = self.direction_kappa_head(hidden).squeeze(1)
                kappa = F.softplus(raw_kappa) + self.min_kappa
                kappa = torch.clamp(kappa, min=self.min_kappa, max=self.max_kappa)
                outputs["direction"] = direction
                outputs["direction_kappa"] = kappa
            else:
                vmf_logits = self.vmf_logits_head(hidden)
                vmf_means = F.normalize(
                    self.vmf_mean_head(hidden).view(
                        hidden.shape[0],
                        self.vmf_components,
                        3,
                    ),
                    dim=2,
                    eps=1e-6,
                )
                vmf_kappas = F.softplus(self.vmf_kappa_head(hidden)) + self.min_kappa
                vmf_kappas = torch.clamp(
                    vmf_kappas,
                    min=self.min_kappa,
                    max=self.max_kappa,
                )
                vmf_weights = torch.softmax(vmf_logits, dim=1)
                outputs.update(
                    {
                        "direction": F.normalize(
                            (vmf_weights.unsqueeze(2) * vmf_means).sum(dim=1),
                            dim=1,
                            eps=1e-6,
                        ),
                        "vmf_logits": vmf_logits,
                        "vmf_means": vmf_means,
                        "vmf_kappas": vmf_kappas,
                    }
                )
        else:
            outputs["direction"] = F.normalize(
                self.direction_head(hidden),
                dim=1,
                eps=1e-6,
            )
        return outputs


def build_data_loaders(
    dataset: ReplayTrainingDataset,
    *,
    seed: int,
    val_fraction: float,
    batch_size: int,
    num_workers: int,
) -> tuple[
    Dataset[dict[str, torch.Tensor]],
    Dataset[dict[str, torch.Tensor]] | None,
    DataLoader[dict[str, torch.Tensor]],
    DataLoader[dict[str, torch.Tensor]] | None,
]:
    """Create deterministic train/eval loaders with reproducible splitting.

    Args:
        dataset: Materialized ReplayTrainingDataset wrapping JSONL shards.
        seed: RNG governs index shuffling/group-aware splits.
        val_fraction: Portion earmarked exclusively for offline evaluation.
        batch_size: User-facing minibatch size (automatically shrunk when tiny).
        num_workers: `DataLoader` multiprocessing helpers.

    Returns:
        Tuple of `(train_ds, val_ds, train_loader, val_loader)`; val pieces may be `None`.
    """

    train_indices, val_indices = split_dataset_indices(
        len(dataset),
        val_fraction=val_fraction,
        seed=seed,
        group_keys=dataset.group_keys,
    )
    train_dataset: Dataset[dict[str, torch.Tensor]] = Subset(dataset, train_indices)
    val_dataset: Dataset[dict[str, torch.Tensor]] | None = (
        Subset(dataset, val_indices) if val_indices else None
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=min(batch_size, len(train_indices)),
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=min(batch_size, len(val_indices)),
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
        )

    return train_dataset, val_dataset, train_loader, val_loader


def load_masked_diffusion_model(
    checkpoint_path: Path,
    *,
    device: torch.device,
) -> MaskedSeedDiffusionModel:
    """Hydrate Torch modules from serialized training checkpoints.

    Args:
        checkpoint_path: Absolute `.pt` bundle containing config + tensors.
        device: Torch device used for eagerly materializing tensors.

    Returns:
        MPG architecture placed in inference mode (`model.eval()`).
    """

    if not checkpoint_path.exists():
        raise ModelComparisonError(f"missing checkpoint: {checkpoint_path}")

    payload = torch.load(checkpoint_path, map_location=device)
    training_config = MaskedDiffusionTrainingConfig(**payload["config"])
    model = MaskedSeedDiffusionModel(build_model_config(training_config)).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model


def _select_reverse_timestep_indices(
    *,
    diffusion_train_steps: int,
    diffusion_steps: int,
    device: torch.device,
) -> list[int]:
    """Choose monotonically decreasing train-time indices for offline denoising."""

    train_steps = max(1, int(diffusion_train_steps))
    reverse_steps = max(1, int(diffusion_steps))
    if reverse_steps == 1:
        return [train_steps]

    values = torch.linspace(
        train_steps,
        1,
        steps=reverse_steps,
        device=device,
        dtype=torch.float32,
    ).round().to(dtype=torch.long).tolist()

    indices: list[int] = []
    for value in values:
        item = int(max(1, min(train_steps, value)))
        if not indices or item != indices[-1]:
            indices.append(item)
    if indices[-1] != 1:
        indices.append(1)
    return indices


@torch.no_grad()
def predict_with_ordered_absorbing_map(
    model: MaskedSeedDiffusionModel,
    batch: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Greedily decode an ordered descriptor, then predict its direction.

    Length is selected from the renderer-valid classes 1 through 9 using the
    condition-only all-mask state. Type bits are then selected left-to-right
    from their prefix states. The completed predicted descriptor, rather than
    any ground-truth descriptor fields in ``batch``, conditions the final
    direction pass.

    Returned length/type logits correspond to the decisions made during the
    rollout. All direction-pass outputs are retained so vMF-mixture checkpoints
    remain compatible with the standard Phase 11 loss evaluator.
    """

    condition = batch["condition"]
    batch_size = int(condition.shape[0])
    if batch_size <= 0:
        raise ModelComparisonError("ordered absorbing rollout batch is empty")

    was_training = model.training
    model.eval()
    try:
        length_outputs = model(**build_ordered_length_inputs(condition))
        valid_length_log_probs = ordered_valid_length_log_probs(
            length_outputs["chain_length_logits"]
        )
        predicted_chain_length = valid_length_log_probs.argmax(dim=1) + 1

        # Retain the class layout expected by the shared evaluator while giving
        # invalid class zero exactly zero probability.
        chain_length_logits = torch.full_like(
            length_outputs["chain_length_logits"],
            fill_value=-torch.inf,
        )
        chain_length_logits[:, 1 : MAX_CHAIN_TYPE_BITS + 1] = (
            valid_length_log_probs
        )

        predicted_chain_type_bits = torch.zeros(
            batch_size,
            MAX_CHAIN_TYPE_BITS,
            device=condition.device,
            dtype=torch.long,
        )
        chain_type_logits = torch.zeros(
            batch_size,
            MAX_CHAIN_TYPE_BITS,
            device=condition.device,
            dtype=valid_length_log_probs.dtype,
        )

        for prefix in range(MAX_CHAIN_TYPE_BITS):
            active_indices = torch.nonzero(
                predicted_chain_length > prefix,
                as_tuple=False,
            ).squeeze(1)
            if active_indices.numel() == 0:
                break

            active_chain_length = predicted_chain_length.index_select(
                0,
                active_indices,
            )
            type_outputs = model(
                **build_ordered_type_inputs(
                    condition=condition.index_select(0, active_indices),
                    chain_length=active_chain_length,
                    chain_type_bits=predicted_chain_type_bits.index_select(
                        0,
                        active_indices,
                    ),
                    prefix_length=torch.full_like(active_chain_length, prefix),
                )
            )
            next_logits = type_outputs["chain_type_logits"][:, prefix]
            chain_type_logits[active_indices, prefix] = next_logits
            predicted_chain_type_bits[active_indices, prefix] = (
                next_logits >= 0.0
            ).long()

        if (
            getattr(model.config, "direction_distribution", None)
            == DIRECTION_DISTRIBUTION_SPHERICAL_D3PM_VMF
        ):
            direction_outputs = predict_spherical_d3pm_endpoint(
                model,
                condition=condition,
                chain_length=predicted_chain_length,
                chain_type_bits=predicted_chain_type_bits,
            )
        else:
            direction_outputs = model(
                **build_ordered_direction_inputs(
                    condition=condition,
                    chain_length=predicted_chain_length,
                    chain_type_bits=predicted_chain_type_bits,
                )
            )
        outputs = dict(direction_outputs)
        outputs.update(
            {
                "chain_length_logits": chain_length_logits,
                "chain_type_logits": chain_type_logits,
                "direction": F.normalize(
                    direction_outputs["direction"],
                    dim=1,
                    eps=1e-6,
                ),
            }
        )
        return outputs
    finally:
        model.train(was_training)


def predict_with_masked_diffusion(
    model: MaskedSeedDiffusionModel,
    batch: dict[str, torch.Tensor],
    *,
    diffusion_steps: int,
    stochastic_diffusion: bool = False,
    diffusion_train_steps: int = 16,
    beta_schedule: str = "linear",
    beta_start: float = 5e-4,
    beta_end: float = 5e-2,
    reverse_noise_scale: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Autoregressively unwind masked MPG predictions offline.

    If ``stochastic_diffusion`` is disabled, this preserves the existing
    deterministic Phase 11 rollout. If enabled, it starts from a random unit
    direction x_T and performs a DDPM-style directional reverse loop using the
    same beta schedule as training. This is an offline diagnostic path; it does
    not change Mitsuba/C++ rendering yet.
    """

    if (
        getattr(getattr(model, "config", None), "discrete_diffusion_mode", None)
        == DISCRETE_DIFFUSION_ORDERED_ABSORBING
    ):
        return predict_with_ordered_absorbing_map(model, batch)

    batch_size = int(batch["condition"].shape[0])
    device = batch["condition"].device
    dtype = batch["condition"].dtype

    masked_chain_length = torch.full_like(
        batch["chain_length"],
        fill_value=MASK_CHAIN_LENGTH_TOKEN,
    )
    masked_chain_type_bits = torch.full_like(
        batch["chain_type_bits"],
        fill_value=MASK_CHAIN_TYPE_TOKEN,
    )

    if stochastic_diffusion:
        train_steps = max(1, int(diffusion_train_steps))
        timestep_indices = _select_reverse_timestep_indices(
            diffusion_train_steps=train_steps,
            diffusion_steps=diffusion_steps,
            device=device,
        )
        alpha_bar = make_alpha_bar_schedule(
            diffusion_train_steps=train_steps,
            beta_schedule=beta_schedule,
            beta_start=beta_start,
            beta_end=beta_end,
            device=device,
            dtype=torch.float32,
        ).to(device=device, dtype=dtype)

        noisy_direction = sample_unit_directions(batch["direction"])
        final_outputs: dict[str, torch.Tensor] | None = None

        for index, t_index in enumerate(timestep_indices):
            timesteps = torch.full(
                (batch_size,),
                fill_value=float(t_index) / float(train_steps),
                device=device,
                dtype=dtype,
            )
            outputs = model(
                condition=batch["condition"],
                noisy_direction=noisy_direction,
                masked_chain_length=masked_chain_length,
                masked_chain_type_bits=masked_chain_type_bits,
                timesteps=timesteps,
            )
            x0_hat = F.normalize(outputs["direction"], dim=1, eps=1e-6)
            masked_chain_length = outputs["chain_length_logits"].argmax(dim=1)
            masked_chain_type_bits = (
                torch.sigmoid(outputs["chain_type_logits"]) >= 0.5
            ).long()
            final_outputs = outputs

            if index < len(timestep_indices) - 1:
                prev_t = timestep_indices[index + 1]
                a_prev = alpha_bar[prev_t].reshape(1, 1)
                eta = sample_unit_directions(x0_hat)
                noisy_direction = F.normalize(
                    torch.sqrt(a_prev) * x0_hat
                    + float(reverse_noise_scale)
                    * torch.sqrt((1.0 - a_prev).clamp_min(0.0))
                    * eta,
                    dim=1,
                    eps=1e-6,
                )

        assert final_outputs is not None
        return {
            "chain_length_logits": final_outputs["chain_length_logits"],
            "chain_type_logits": final_outputs["chain_type_logits"],
            "direction": F.normalize(final_outputs["direction"], dim=1, eps=1e-6),
        }

    # Legacy deterministic rollout.
    noisy_direction = torch.zeros_like(batch["direction"])

    if diffusion_steps <= 1:
        outputs = model(
            condition=batch["condition"],
            noisy_direction=noisy_direction,
            masked_chain_length=masked_chain_length,
            masked_chain_type_bits=masked_chain_type_bits,
            timesteps=torch.zeros(
                batch_size,
                device=device,
                dtype=dtype,
            ),
        )
        return {
            "chain_length_logits": outputs["chain_length_logits"],
            "chain_type_logits": outputs["chain_type_logits"],
            "direction": F.normalize(outputs["direction"], dim=1, eps=1e-6),
        }

    # Start from fully masked tokens and sequentially reveal discrete logits + normals.
    for step in range(diffusion_steps, 0, -1):
        timesteps = torch.full(
            (batch_size,),
            fill_value=float(step) / float(diffusion_steps),
            device=device,
            dtype=dtype,
        )
        outputs = model(
            condition=batch["condition"],
            noisy_direction=noisy_direction,
            masked_chain_length=masked_chain_length,
            masked_chain_type_bits=masked_chain_type_bits,
            timesteps=timesteps,
        )
        masked_chain_length = outputs["chain_length_logits"].argmax(dim=1)
        masked_chain_type_bits = (
            torch.sigmoid(outputs["chain_type_logits"]) >= 0.5
        ).long()
        noisy_direction = F.normalize(outputs["direction"], dim=1, eps=1e-6)

    final_timesteps = torch.zeros(
        batch_size,
        device=device,
        dtype=dtype,
    )
    final_outputs = model(
        condition=batch["condition"],
        noisy_direction=noisy_direction,
        masked_chain_length=masked_chain_length,
        masked_chain_type_bits=masked_chain_type_bits,
        timesteps=final_timesteps,
    )
    return {
        "chain_length_logits": final_outputs["chain_length_logits"],
        "chain_type_logits": final_outputs["chain_type_logits"],
        "direction": F.normalize(final_outputs["direction"], dim=1, eps=1e-6),
    }


def _learned_vmf_direction_nll_per_sample(
    *,
    direction_mu: torch.Tensor,
    direction_kappa: torch.Tensor,
    direction_target: torch.Tensor,
    vmf_max_kappa: float,
) -> torch.Tensor:
    """Return normalized single-component vMF NLL values on S^2."""

    mu = F.normalize(direction_mu.float(), dim=1, eps=1e-6)
    target = F.normalize(direction_target.float(), dim=1, eps=1e-6)
    kappa = direction_kappa.reshape(-1).float().clamp(
        min=1e-6,
        max=float(vmf_max_kappa),
    )
    cosine = (mu * target).sum(dim=1).clamp(min=-1.0, max=1.0)
    return -(log_vmf_normalizer_3d(kappa) + kappa * cosine)


def learned_vmf_direction_nll(
    *,
    direction_mu: torch.Tensor,
    direction_kappa: torch.Tensor,
    direction_target: torch.Tensor,
    vmf_max_kappa: float = 128.0,
) -> torch.Tensor:
    """Mean normalized NLL for a single 3D vMF distribution on S^2.

    Density: C_3(kappa) exp(kappa * mu^T x), where
    C_3(kappa) = kappa / (4*pi*sinh(kappa)).
    The normalization constant is retained so this value is directly comparable
    to the K=1 case of the masked model's vMF-mixture NLL. The scalar-mean return
    contract is preserved for existing callers.
    """

    return _learned_vmf_direction_nll_per_sample(
        direction_mu=direction_mu,
        direction_kappa=direction_kappa,
        direction_target=direction_target,
        vmf_max_kappa=vmf_max_kappa,
    ).mean()


def compute_model_comparison_losses(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    *,
    direction_loss_weight: float,
    direction_loss_type: str = "vmf",
    direction_vmf_kappa: float = 2.5,
    vmf_max_kappa: float = 128.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Loss wrapper used by Phase 11 comparison.

    Learned single-component and mixture vMF heads use their normalized NLLs.
    Other predictors retain the standard ``compute_losses`` behavior.
    """

    vmf_mixture_keys = {"vmf_logits", "vmf_means", "vmf_kappas"}
    if vmf_mixture_keys.issubset(outputs):
        return compute_losses(
            outputs,
            batch,
            direction_loss_weight=direction_loss_weight,
            direction_loss_type="vmf_mixture",
            direction_vmf_kappa=direction_vmf_kappa,
            vmf_max_kappa=vmf_max_kappa,
        )

    if "direction_kappa" not in outputs:
        return compute_losses(
            outputs,
            batch,
            direction_loss_weight=direction_loss_weight,
            direction_loss_type=direction_loss_type,
            direction_vmf_kappa=direction_vmf_kappa,
        )

    base_loss, parts = compute_losses(
        outputs,
        batch,
        direction_loss_weight=0.0,
        direction_loss_type="vmf",
        direction_vmf_kappa=2.5,
    )
    direction_loss_per_sample = _learned_vmf_direction_nll_per_sample(
        direction_mu=outputs["direction"],
        direction_kappa=outputs["direction_kappa"],
        direction_target=batch["direction"],
        vmf_max_kappa=vmf_max_kappa,
    )
    sample_weight = batch["sample_weight"].to(
        device=direction_loss_per_sample.device,
        dtype=direction_loss_per_sample.dtype,
    )
    weighted_direction_loss = (
        direction_loss_per_sample * sample_weight
    ).sum() / sample_weight.sum().clamp_min(1e-6)
    total_loss = base_loss + float(direction_loss_weight) * weighted_direction_loss
    parts = dict(parts)
    parts["direction"] = direction_loss_per_sample.detach().mean()
    parts["direction_kappa_mean"] = outputs["direction_kappa"].detach().mean()
    return total_loss, parts


def evaluate_predictor(
    *,
    data_loader: DataLoader[dict[str, torch.Tensor]],
    device: torch.device,
    direction_loss_weight: float,
    direction_loss_type: str = "vmf",
    direction_vmf_kappa: float = 2.5,
    vmf_max_kappa: float = 128.0,
    predictor: Callable[[dict[str, torch.Tensor]], dict[str, torch.Tensor]],
) -> EvaluationMetrics:
    """Run `predictor` across `data_loader`, aggregating validation KPIs.

    Args:
        data_loader: Torch loader streaming validation batches already on CPU.
        device: Target accel used before moving tensors for inference math.
        direction_loss_weight: Passed through unchanged to `compute_losses`.
        predictor: Callable bridging raw batches onto MPG-like outputs.

    Returns:
        Reduced EvaluationMetrics aggregated over every batch encountered.
    """

    total_samples = 0
    total_loss = 0.0
    total_chain_length_loss = 0.0
    total_chain_type_loss = 0.0
    total_direction_loss = 0.0
    total_length_correct = 0.0
    total_bit_correct = 0.0
    total_bit_count = 0.0
    total_direction_mse = 0.0
    total_direction_cosine = 0.0

    with torch.no_grad():
        for batch in make_progress(data_loader, desc="eval", total=len(data_loader)):
            batch = move_batch_to_device(batch, device)
            outputs = predictor(batch)
            loss, parts = compute_model_comparison_losses(
                outputs,
                batch,
                direction_loss_weight=direction_loss_weight,
                direction_loss_type=direction_loss_type,
                direction_vmf_kappa=direction_vmf_kappa,
                vmf_max_kappa=vmf_max_kappa,
            )

            batch_size = int(batch["condition"].shape[0])
            total_samples += batch_size
            total_loss += float(loss.item()) * batch_size
            total_chain_length_loss += float(parts["chain_length"].item()) * batch_size
            total_chain_type_loss += float(parts["chain_type"].item()) * batch_size
            total_direction_loss += float(parts["direction"].item()) * batch_size

            predicted_length = outputs["chain_length_logits"].argmax(dim=1)
            total_length_correct += float(
                (predicted_length == batch["chain_length"]).float().sum().item()
            )

            predicted_bits = (torch.sigmoid(outputs["chain_type_logits"]) >= 0.5).long()
            valid_mask = batch["chain_type_valid"] > 0.5
            total_bit_correct += float(
                ((predicted_bits == batch["chain_type_bits"]) & valid_mask).sum().item()
            )
            total_bit_count += float(valid_mask.sum().item())

            direction = F.normalize(outputs["direction"], dim=1, eps=1e-6)
            direction_target = batch["direction"]
            total_direction_mse += float(
                F.mse_loss(direction, direction_target, reduction="none")
                .mean(dim=1)
                .sum()
                .item()
            )
            total_direction_cosine += float(
                F.cosine_similarity(direction, direction_target, dim=1).sum().item()
            )

    if total_samples <= 0:
        raise ModelComparisonError("validation dataset is empty")

    return EvaluationMetrics(
        num_samples=total_samples,
        reconstruction_loss=total_loss / total_samples,
        chain_length_loss=total_chain_length_loss / total_samples,
        chain_type_loss=total_chain_type_loss / total_samples,
        direction_loss=total_direction_loss / total_samples,
        chain_length_accuracy=total_length_correct / total_samples,
        chain_type_bit_accuracy=(total_bit_correct / max(total_bit_count, 1.0)),
        direction_mse=total_direction_mse / total_samples,
        direction_cosine=total_direction_cosine / total_samples,
    )


def save_simple_checkpoint(
    checkpoint_path: Path,
    *,
    model: SimpleGuidanceMLP,
    optimizer: AdamW,
    epoch: int,
    metrics: EpochMetrics,
    config: ComparisonConfig,
) -> None:
    """Serialize optimizer + RNG metadata for reproducible warm-starts.

    Args:
        checkpoint_path: Intended `.pt` destination on disk.
        model: Lightweight supervised comparator instance.
        optimizer: AdamW bookkeeping object tied to aforementioned model weights.
        epoch: Integer epoch stamp stored alongside tensors purely for bookkeeping.
        metrics: Latest EpochMetrics emitted by training/eval telemetry hooks.
        config: Canonical ComparisonConfig echoed into checkpoint payloads.
    """

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": asdict(metrics),
            "config": asdict(config),
        },
        checkpoint_path,
    )


def write_epoch_history_csv(path: Path, history: tuple[EpochMetrics, ...]) -> None:
    """Emit CSV rows summarizing supervised baseline epochs.

    Args:
        path: Target CSV filesystem path.
        history: EpochMetrics tuples streamed through training telemetry.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(history[0]).keys()))
        writer.writeheader()
        for item in history:
            writer.writerow(asdict(item))


def train_simple_baseline(
    *,
    train_loader: DataLoader[dict[str, torch.Tensor]],
    val_loader: DataLoader[dict[str, torch.Tensor]] | None,
    device: torch.device,
    config: ComparisonConfig,
    run_dir: Path,
) -> tuple[SimpleGuidanceMLP, tuple[EpochMetrics, ...], Path]:
    """Fit the lightweight MLP on identical replay shards as MPG.

    Args:
        train_loader: Supervised minibatch loader for stochastic updates.
        val_loader: Optional loader for periodic validation KPIs or `None`.
        device: Accelerator hosting both model tensors and batches.
        config: Holds nested `simple_baseline` hyper-parameters alongside losses.
        run_dir: Parent directory emitted under comparison outputs.

    Returns:
        Trained comparator, chronological epoch metrics tuple, champion checkpoint path.
    """

    model = SimpleGuidanceMLP(CONDITION_DIM, config.simple_baseline).to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=config.simple_baseline.learning_rate,
        weight_decay=config.simple_baseline.weight_decay,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    best_val_loss = float("inf")
    history: list[EpochMetrics] = []
    best_checkpoint = run_dir / "simple_baseline_best.pt"
    last_checkpoint = run_dir / "simple_baseline_last.pt"

    for epoch in range(1, config.simple_baseline.epochs + 1):
        model.train()
        train_loss = 0.0
        train_chain_length_loss = 0.0
        train_chain_type_loss = 0.0
        train_direction_loss = 0.0

        for batch in make_progress(
            train_loader,
            desc=f"simple train {epoch}",
            total=len(train_loader),
        ):
            batch = move_batch_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(
                device_type=device.type,
                enabled=device.type == "cuda",
            ):
                outputs = model(batch["condition"])
                loss, parts = compute_model_comparison_losses(
                    outputs,
                    batch,
                    direction_loss_weight=config.direction_loss_weight,
                    direction_loss_type=config.direction_loss_type,
                    direction_vmf_kappa=config.direction_vmf_kappa,
                    vmf_max_kappa=config.simple_baseline.max_kappa,
                )

            scaler.scale(loss).backward()
            if config.simple_baseline.grad_clip_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=float(config.simple_baseline.grad_clip_norm),
                )
            scaler.step(optimizer)
            scaler.update()

            train_loss += float(loss.item())
            train_chain_length_loss += float(parts["chain_length"].item())
            train_chain_type_loss += float(parts["chain_type"].item())
            train_direction_loss += float(parts["direction"].item())

        batch_count = float(len(train_loader))
        val_loss: float | None = None
        val_chain_length_loss: float | None = None
        val_chain_type_loss: float | None = None
        val_direction_loss: float | None = None
        if val_loader is not None and len(val_loader) > 0:
            val_metrics = evaluate_predictor(
                data_loader=val_loader,
                device=device,
                direction_loss_weight=config.direction_loss_weight,
                direction_loss_type=config.direction_loss_type,
                direction_vmf_kappa=config.direction_vmf_kappa,
                vmf_max_kappa=config.simple_baseline.max_kappa,
                predictor=lambda current_batch: model(current_batch["condition"]),
            )
            val_loss = val_metrics.reconstruction_loss
            val_chain_length_loss = val_metrics.chain_length_loss
            val_chain_type_loss = val_metrics.chain_type_loss
            val_direction_loss = val_metrics.direction_loss

        metrics = EpochMetrics(
            epoch=epoch,
            train_loss=train_loss / batch_count,
            train_chain_length_loss=train_chain_length_loss / batch_count,
            train_chain_type_loss=train_chain_type_loss / batch_count,
            train_direction_loss=train_direction_loss / batch_count,
            val_loss=val_loss,
            val_chain_length_loss=val_chain_length_loss,
            val_chain_type_loss=val_chain_type_loss,
            val_direction_loss=val_direction_loss,
        )
        history.append(metrics)

        save_simple_checkpoint(
            last_checkpoint,
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            metrics=metrics,
            config=config,
        )
        if val_loss is None:
            if epoch == config.simple_baseline.epochs:
                save_simple_checkpoint(
                    best_checkpoint,
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    metrics=metrics,
                    config=config,
                )
        elif val_loss <= best_val_loss:
            best_val_loss = val_loss
            save_simple_checkpoint(
                best_checkpoint,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                metrics=metrics,
                config=config,
            )

        LOGGER.info(
            "simple epoch=%s train_loss=%.6f val_loss=%s",
            epoch,
            metrics.train_loss,
            "n/a" if metrics.val_loss is None else f"{metrics.val_loss:.6f}",
        )

    return model, tuple(history), best_checkpoint


def write_comparison_markdown(
    output_path: Path,
    *,
    config: ComparisonConfig,
    dataset: ReplayTrainingDataset,
    masked_metrics: EvaluationMetrics,
    simple_metrics: EvaluationMetrics,
) -> None:
    """Render Markdown tables summarizing cross-model parity.

    Args:
        output_path: Destination Markdown artefact beneath run dir roots.
        config: Echoed configuration blocks for inspection.
        dataset: Dataset stats inlined into narrative appendix.
        masked_metrics: Quantitative KPIs aggregated for MPG rollout.
        simple_metrics: Quantitative KPIs aggregated for comparator MLP rollout.
    """

    lines = [
        "# Phase 11 Model Comparison",
        "",
        "## Dataset",
        "",
        f"- records: `{dataset.stats.num_records}`",
        f"- source_files: `{dataset.stats.num_source_files}`",
        f"- filtered_records: `{dataset.stats.num_filtered_records}`",
        f"- mean_sample_weight: `{dataset.stats.mean_sample_weight:.6f}`",
        "",
        "## Config",
        "",
        f"- checkpoint: `{config.masked_diffusion_checkpoint}`",
        f"- diffusion_steps: `{config.diffusion_steps}`",
        f"- stochastic_diffusion: `{config.stochastic_diffusion}`",
        f"- diffusion_train_steps: `{config.diffusion_train_steps}`",
        f"- beta_schedule: `{config.beta_schedule}`",
        f"- reverse_noise_scale: `{config.reverse_noise_scale}`",
        f"- direction_loss_type: `{config.direction_loss_type}`",
        f"- direction_vmf_kappa: `{config.direction_vmf_kappa}`",
        f"- val_fraction: `{config.val_fraction}`",
        f"- simple_epochs: `{config.simple_baseline.epochs}`",
        f"- simple_grad_clip_norm: `{config.simple_baseline.grad_clip_norm}`",
        f"- simple_vmf_components: `{config.simple_baseline.vmf_components}`",
        "",
        "## Metrics",
        "",
        "| model | reconstruction_loss | chain_length_accuracy | chain_type_bit_accuracy | direction_mse | direction_cosine |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
        (
            f"| masked_diffusion | {masked_metrics.reconstruction_loss:.6f} | "
            f"{masked_metrics.chain_length_accuracy:.6f} | "
            f"{masked_metrics.chain_type_bit_accuracy:.6f} | "
            f"{masked_metrics.direction_mse:.6f} | "
            f"{masked_metrics.direction_cosine:.6f} |"
        ),
        (
            f"| simple_mlp | {simple_metrics.reconstruction_loss:.6f} | "
            f"{simple_metrics.chain_length_accuracy:.6f} | "
            f"{simple_metrics.chain_type_bit_accuracy:.6f} | "
            f"{simple_metrics.direction_mse:.6f} | "
            f"{simple_metrics.direction_cosine:.6f} |"
        ),
        "",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def run_model_comparison(
    *,
    workspace_root: Path,
    config: ComparisonConfig,
) -> ComparisonArtifacts:
    """End-to-end offline evaluation comparing MPG vs supervised MLP.

    Args:
        workspace_root: `masked_diffusion_workspace` filesystem anchor.
        config: Hydrated knobs describing loaders, checkpoints, device choice.

    Returns:
        ComparisonArtifacts documenting every persisted artifact emitted here.
    """

    set_global_seed(config.seed)
    device = detect_device(config.device)

    replay_paths = tuple(
        (workspace_root / path).resolve() for path in config.replay_paths
    )
    dataset = ReplayTrainingDataset(replay_paths, max_records=config.max_records)
    LOGGER.info(
        "comparison dataset: records=%s source_files=%s filtered=%s mean_weight=%.4f",
        dataset.stats.num_records,
        dataset.stats.num_source_files,
        dataset.stats.num_filtered_records,
        dataset.stats.mean_sample_weight,
    )

    (
        _train_dataset,
        val_dataset,
        train_loader,
        val_loader,
    ) = build_data_loaders(
        dataset,
        seed=config.seed,
        val_fraction=config.val_fraction,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
    )
    if val_dataset is None or val_loader is None or len(val_loader) == 0:
        raise ModelComparisonError("phase_11 requires a non-empty validation split")

    output_root = (workspace_root / config.output_dir).resolve()
    run_dir = create_run_dir(output_root)
    (run_dir / "compare_config.json").write_text(
        json.dumps(asdict(config), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    simple_training_start = time.perf_counter()
    simple_model, history, simple_checkpoint = train_simple_baseline(
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        config=config,
        run_dir=run_dir,
    )
    simple_training_seconds = time.perf_counter() - simple_training_start

    masked_checkpoint_path = resolve_masked_diffusion_checkpoint(
        workspace_root=workspace_root,
        checkpoint_reference=config.masked_diffusion_checkpoint,
    )
    masked_model = load_masked_diffusion_model(masked_checkpoint_path, device=device)

    simple_metrics = evaluate_predictor(
        data_loader=val_loader,
        device=device,
        direction_loss_weight=config.direction_loss_weight,
        direction_loss_type=config.direction_loss_type,
        direction_vmf_kappa=config.direction_vmf_kappa,
        vmf_max_kappa=config.simple_baseline.max_kappa,
        predictor=lambda batch: simple_model(batch["condition"]),
    )
    masked_metrics = evaluate_predictor(
        data_loader=val_loader,
        device=device,
        direction_loss_weight=config.direction_loss_weight,
        direction_loss_type=config.direction_loss_type,
        direction_vmf_kappa=config.direction_vmf_kappa,
        vmf_max_kappa=masked_model.config.vmf_max_kappa,
        predictor=lambda batch: predict_with_masked_diffusion(
            masked_model,
            batch,
            diffusion_steps=config.diffusion_steps,
            stochastic_diffusion=config.stochastic_diffusion,
            diffusion_train_steps=config.diffusion_train_steps,
            beta_schedule=config.beta_schedule,
            beta_start=config.beta_start,
            beta_end=config.beta_end,
            reverse_noise_scale=config.reverse_noise_scale,
        ),
    )

    better_model = (
        "masked_diffusion"
        if masked_metrics.reconstruction_loss <= simple_metrics.reconstruction_loss
        else "simple_mlp"
    )

    comparison_json = run_dir / "comparison.json"
    comparison_md = run_dir / "comparison.md"
    simple_history_json = run_dir / "simple_history.json"
    simple_history_csv = run_dir / "simple_history.csv"
    comparison_json.write_text(
        json.dumps(
            {
                "config": asdict(config),
                "dataset_stats": asdict(dataset.stats),
                "masked_diffusion": asdict(masked_metrics),
                "simple_mlp": asdict(simple_metrics),
                "better_model": better_model,
                "simple_parameter_count": sum(
                    parameter.numel() for parameter in simple_model.parameters()
                ),
                "masked_parameter_count": sum(
                    parameter.numel() for parameter in masked_model.parameters()
                ),
                "simple_training_seconds": simple_training_seconds,
                "simple_epochs_completed": len(history),
                "simple_optimizer_steps": len(history) * len(train_loader),
                "simple_checkpoint": str(simple_checkpoint),
                "masked_checkpoint": str(masked_checkpoint_path),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    simple_history_json.write_text(
        json.dumps([asdict(item) for item in history], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_epoch_history_csv(simple_history_csv, history)
    # Persist a stable symlink-style copy for downstream runtime export + benchmarking.
    stable_simple_checkpoint = output_root / "best_simple_mlp.pt"
    shutil.copy2(simple_checkpoint, stable_simple_checkpoint)
    write_comparison_markdown(
        comparison_md,
        config=config,
        dataset=dataset,
        masked_metrics=masked_metrics,
        simple_metrics=simple_metrics,
    )
    LOGGER.info(
        "comparison finished: masked_loss=%.6f simple_loss=%.6f better=%s",
        masked_metrics.reconstruction_loss,
        simple_metrics.reconstruction_loss,
        better_model,
    )

    return ComparisonArtifacts(
        run_dir=run_dir,
        comparison_json=comparison_json,
        comparison_md=comparison_md,
        simple_checkpoint=simple_checkpoint,
        simple_history_json=simple_history_json,
        simple_history_csv=simple_history_csv,
    )
