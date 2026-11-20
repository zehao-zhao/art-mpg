from __future__ import annotations

import csv
import json
import logging
import math
import os
import random
import shutil
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
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
    DISCRETE_DIFFUSION_LEGACY,
    DISCRETE_DIFFUSION_ORDERED_ABSORBING,
    DIRECTION_DISTRIBUTION_SPHERICAL_D3PM_VMF,
    SPHERICAL_D3PM_GEOMETRY_FIXED_ANCHOR,
    SPHERICAL_D3PM_GEOMETRY_LEARNED_EMISSION,
    SUPPORTED_SPHERICAL_D3PM_GEOMETRIES,
    SUPPORTED_DISCRETE_DIFFUSION_MODES,
    MaskedDiffusionModelConfig,
    MaskedSeedDiffusionModel,
    build_ordered_direction_inputs,
    build_ordered_length_inputs,
    build_ordered_type_inputs,
    ordered_valid_length_log_probs,
    predict_spherical_d3pm_endpoint,
)


LOGGER = logging.getLogger("masked_diffusion_mpg.training")

DESCRIPTOR_WEIGHT_LOG_ENERGY = "log_energy"
DESCRIPTOR_WEIGHT_CLIPPED_RAW_ENERGY = "clipped_raw_energy"
SUPPORTED_DESCRIPTOR_WEIGHT_MODES = frozenset(
    {
        DESCRIPTOR_WEIGHT_LOG_ENERGY,
        DESCRIPTOR_WEIGHT_CLIPPED_RAW_ENERGY,
    }
)
SUPPORTED_DESCRIPTOR_WEIGHT_QUANTILES = frozenset({0.95, 0.99})
DIRECTION_WEIGHT_LOG_ENERGY = "log_energy"
DIRECTION_WEIGHT_CLIPPED_RAW_ENERGY = "clipped_raw_energy"
SUPPORTED_DIRECTION_WEIGHT_MODES = frozenset(
    {
        DIRECTION_WEIGHT_LOG_ENERGY,
        DIRECTION_WEIGHT_CLIPPED_RAW_ENERGY,
    }
)
SUPPORTED_DIRECTION_WEIGHT_QUANTILES = frozenset({0.95, 0.99})

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


class MaskedDiffusionTrainingError(RuntimeError):
    """Masked diffusion training failure."""


@dataclass(slots=True, frozen=True)
class MaskedDiffusionTrainingConfig:
    """Full configuration for the training script."""

    replay_paths: tuple[str, ...]
    output_dir: str
    seed: int = 42
    batch_size: int = 16
    epochs: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    val_fraction: float = 0.25
    max_records: int | None = None
    num_workers: int = 4
    device: str = "auto"
    early_stop_patience: int | None = None
    grad_clip_norm: float | None = None
    direction_loss_weight: float = 0.5
    direction_loss_type: str = "mse"
    direction_vmf_kappa: float = 1.0
    # Descriptor and direction weights are configured independently. In both
    # cases, ``log_energy`` preserves the historical weighting and
    # ``clipped_raw_energy`` uses a cap fitted on the training split only.
    descriptor_weight_mode: str = DESCRIPTOR_WEIGHT_LOG_ENERGY
    descriptor_weight_quantile: float = 0.99
    direction_weight_mode: str = DIRECTION_WEIGHT_LOG_ENERGY
    direction_weight_quantile: float = 0.99
    direction_distribution: str = "vector"
    vmf_components: int = 4
    vmf_max_kappa: float = 32.0
    direction_states: int = 32
    direction_diffusion_steps: int = 4
    direction_grid_kappa: float = 32.0
    direction_state_geometry: str = SPHERICAL_D3PM_GEOMETRY_FIXED_ANCHOR
    direction_denoising_weight: float = 0.25
    max_nonfinite_batches: int = 4
    direction_space: str = "r3"
    full_mask_probability: float = 0.25
    # True directional DDPM-style noising for training-side stochastic diffusion.
    # When disabled, the original Gaussian corruption path is preserved.
    stochastic_diffusion: bool = False
    diffusion_objective: str = "predict_x0"
    diffusion_train_steps: int = 16
    beta_schedule: str = "linear"
    beta_start: float = 5e-4
    beta_end: float = 5e-2
    reverse_noise_scale: float = 1.0
    backbone_type: str = "mlp"
    hidden_dim: int = 256
    depth: int = 4
    dropout: float = 0.1
    time_embedding_dim: int = 64
    token_embedding_dim: int = 32
    transformer_heads: int = 4
    transformer_feedforward_dim: int | None = None
    discrete_diffusion_mode: str = DISCRETE_DIFFUSION_LEGACY


@dataclass(slots=True, frozen=True)
class EpochMetrics:
    """Core metrics for one epoch."""

    epoch: int
    train_loss: float
    train_chain_length_loss: float
    train_chain_type_loss: float
    train_direction_loss: float
    val_loss: float | None
    val_chain_length_loss: float | None
    val_chain_type_loss: float | None
    val_direction_loss: float | None


@dataclass(slots=True, frozen=True)
class TrainingArtifacts:
    """Paths to training artifacts."""

    run_dir: Path
    metrics_json: Path
    metrics_csv: Path
    best_checkpoint: Path
    last_checkpoint: Path


def configure_logging() -> None:
    """Initialize training logs."""

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def validate_loss_weight_config(
    config: MaskedDiffusionTrainingConfig,
) -> None:
    """Validate per-head weight settings without inspecting replay data."""

    if config.descriptor_weight_mode not in SUPPORTED_DESCRIPTOR_WEIGHT_MODES:
        supported = ", ".join(sorted(SUPPORTED_DESCRIPTOR_WEIGHT_MODES))
        raise MaskedDiffusionTrainingError(
            "unsupported descriptor_weight_mode: "
            f"{config.descriptor_weight_mode!r}; expected one of {supported}"
        )
    if config.descriptor_weight_quantile not in SUPPORTED_DESCRIPTOR_WEIGHT_QUANTILES:
        supported = ", ".join(
            str(value) for value in sorted(SUPPORTED_DESCRIPTOR_WEIGHT_QUANTILES)
        )
        raise MaskedDiffusionTrainingError(
            "descriptor_weight_quantile must be one of "
            f"{supported}; got {config.descriptor_weight_quantile}"
        )
    if config.direction_weight_mode not in SUPPORTED_DIRECTION_WEIGHT_MODES:
        supported = ", ".join(sorted(SUPPORTED_DIRECTION_WEIGHT_MODES))
        raise MaskedDiffusionTrainingError(
            "unsupported direction_weight_mode: "
            f"{config.direction_weight_mode!r}; expected one of {supported}"
        )
    if config.direction_weight_quantile not in SUPPORTED_DIRECTION_WEIGHT_QUANTILES:
        supported = ", ".join(
            str(value) for value in sorted(SUPPORTED_DIRECTION_WEIGHT_QUANTILES)
        )
        raise MaskedDiffusionTrainingError(
            "direction_weight_quantile must be one of "
            f"{supported}; got {config.direction_weight_quantile}"
        )
    if (
        config.direction_distribution
        == DIRECTION_DISTRIBUTION_SPHERICAL_D3PM_VMF
    ):
        if config.discrete_diffusion_mode != DISCRETE_DIFFUSION_ORDERED_ABSORBING:
            raise MaskedDiffusionTrainingError(
                "spherical_d3pm_vmf currently requires ordered_absorbing mode"
            )
        if config.direction_states < 2:
            raise MaskedDiffusionTrainingError("direction_states must be at least 2")
        if config.direction_diffusion_steps <= 0:
            raise MaskedDiffusionTrainingError(
                "direction_diffusion_steps must be positive"
            )
        if config.direction_state_geometry not in SUPPORTED_SPHERICAL_D3PM_GEOMETRIES:
            raise MaskedDiffusionTrainingError(
                "unsupported direction_state_geometry: "
                f"{config.direction_state_geometry}"
            )
        if not math.isfinite(config.direction_grid_kappa) or config.direction_grid_kappa <= 0.0:
            raise MaskedDiffusionTrainingError(
                "direction_grid_kappa must be finite and positive"
            )
        if (
            not math.isfinite(config.direction_denoising_weight)
            or config.direction_denoising_weight < 0.0
        ):
            raise MaskedDiffusionTrainingError(
                "direction_denoising_weight must be finite and nonnegative"
            )


def load_training_config(config_path: Path) -> MaskedDiffusionTrainingConfig:
    """Load training settings from JSON config."""

    payload = json.loads(config_path.read_text(encoding="utf-8").replace("${ARTMPG_ROOT}", str(Path(__file__).resolve().parents[3])))
    config = MaskedDiffusionTrainingConfig(
        replay_paths=tuple(payload["replay_paths"]),
        output_dir=payload["output_dir"],
        seed=int(payload.get("seed", 42)),
        batch_size=int(payload.get("batch_size", 16)),
        epochs=int(payload.get("epochs", 64)),
        learning_rate=float(payload.get("learning_rate", 1e-3)),
        weight_decay=float(payload.get("weight_decay", 1e-4)),
        val_fraction=float(payload.get("val_fraction", 0.25)),
        max_records=(
            None if payload.get("max_records") is None else int(payload["max_records"])
        ),
        num_workers=int(payload.get("num_workers", 4)),
        device=str(payload.get("device", "auto")),
        early_stop_patience=(
            None
            if payload.get("early_stop_patience") is None
            else int(payload["early_stop_patience"])
        ),
        grad_clip_norm=(
            None
            if payload.get("grad_clip_norm") is None
            else float(payload["grad_clip_norm"])
        ),
        direction_loss_weight=float(payload.get("direction_loss_weight", 0.5)),
        direction_loss_type=str(payload.get("direction_loss_type", "mse")),
        direction_vmf_kappa=float(payload.get("direction_vmf_kappa", 1.0)),
        descriptor_weight_mode=str(
            payload.get("descriptor_weight_mode", DESCRIPTOR_WEIGHT_LOG_ENERGY)
        ),
        descriptor_weight_quantile=float(
            payload.get("descriptor_weight_quantile", 0.99)
        ),
        direction_weight_mode=str(
            payload.get("direction_weight_mode", DIRECTION_WEIGHT_LOG_ENERGY)
        ),
        direction_weight_quantile=float(
            payload.get("direction_weight_quantile", 0.99)
        ),
        direction_distribution=str(payload.get("direction_distribution", "vector")),
        vmf_components=int(payload.get("vmf_components", 4)),
        vmf_max_kappa=float(payload.get("vmf_max_kappa", 32.0)),
        direction_states=int(payload.get("direction_states", 32)),
        direction_diffusion_steps=int(
            payload.get("direction_diffusion_steps", 4)
        ),
        direction_grid_kappa=float(payload.get("direction_grid_kappa", 32.0)),
        direction_state_geometry=str(
            payload.get(
                "direction_state_geometry",
                SPHERICAL_D3PM_GEOMETRY_FIXED_ANCHOR,
            )
        ),
        direction_denoising_weight=float(
            payload.get("direction_denoising_weight", 0.25)
        ),
        max_nonfinite_batches=int(payload.get("max_nonfinite_batches", 4)),
        direction_space=str(payload.get("direction_space", "r3")),
        full_mask_probability=float(payload.get("full_mask_probability", 0.25)),
        stochastic_diffusion=bool(payload.get("stochastic_diffusion", False)),
        diffusion_objective=str(payload.get("diffusion_objective", "predict_x0")),
        diffusion_train_steps=int(payload.get("diffusion_train_steps", 16)),
        beta_schedule=str(payload.get("beta_schedule", "linear")),
        beta_start=float(payload.get("beta_start", 5e-4)),
        beta_end=float(payload.get("beta_end", 5e-2)),
        reverse_noise_scale=float(payload.get("reverse_noise_scale", 1.0)),
        backbone_type=str(payload.get("backbone_type", "mlp")),
        hidden_dim=int(payload.get("hidden_dim", 256)),
        depth=int(payload.get("depth", 4)),
        dropout=float(payload.get("dropout", 0.1)),
        time_embedding_dim=int(payload.get("time_embedding_dim", 64)),
        token_embedding_dim=int(payload.get("token_embedding_dim", 32)),
        transformer_heads=int(payload.get("transformer_heads", 4)),
        transformer_feedforward_dim=(
            None
            if payload.get("transformer_feedforward_dim") is None
            else int(payload["transformer_feedforward_dim"])
        ),
        discrete_diffusion_mode=str(
            payload.get("discrete_diffusion_mode", DISCRETE_DIFFUSION_LEGACY)
        ),
    )
    validate_loss_weight_config(config)
    return config


def set_global_seed(seed: int) -> None:
    """Seed every RNG used in training."""

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def detect_device(device_name: str) -> torch.device:
    """Resolve training device."""

    if device_name == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(device_name)
    if device.type == "cuda":
        device_index = (
            device.index if device.index is not None else torch.cuda.current_device()
        )
        LOGGER.info("device: cuda (%s)", torch.cuda.get_device_name(device_index))
    else:
        LOGGER.info("device: %s", device)
    return device


def build_model_config(
    config: MaskedDiffusionTrainingConfig,
) -> MaskedDiffusionModelConfig:
    """Map training knobs to model-architecture settings."""

    direction_distribution = config.direction_distribution
    if (
        config.direction_loss_type == "vmf_mixture"
        and direction_distribution
        != DIRECTION_DISTRIBUTION_SPHERICAL_D3PM_VMF
    ):
        direction_distribution = "vmf_mixture"

    return MaskedDiffusionModelConfig(
        condition_dim=CONDITION_DIM,
        hidden_dim=config.hidden_dim,
        depth=config.depth,
        dropout=config.dropout,
        time_embedding_dim=config.time_embedding_dim,
        token_embedding_dim=config.token_embedding_dim,
        backbone_type=config.backbone_type,
        transformer_heads=config.transformer_heads,
        transformer_feedforward_dim=config.transformer_feedforward_dim,
        direction_distribution=direction_distribution,
        vmf_components=config.vmf_components,
        vmf_max_kappa=config.vmf_max_kappa,
        direction_states=config.direction_states,
        direction_diffusion_steps=config.direction_diffusion_steps,
        direction_grid_kappa=config.direction_grid_kappa,
        direction_state_geometry=config.direction_state_geometry,
        discrete_diffusion_mode=config.discrete_diffusion_mode,
    )


def split_dataset_indices(
    dataset_size: int,
    *,
    val_fraction: float,
    seed: int,
    group_keys: tuple[str, ...] | None = None,
) -> tuple[list[int], list[int]]:
    """Split train/validation indices with a deterministic RNG."""

    indices = list(range(dataset_size))
    if dataset_size <= 1 or val_fraction <= 0:
        return indices, []

    rng = random.Random(seed)
    if group_keys is None:
        rng.shuffle(indices)
        val_size = max(1, int(round(dataset_size * val_fraction)))
        val_size = min(val_size, dataset_size - 1)
        return indices[val_size:], indices[:val_size]

    if len(group_keys) != dataset_size:
        raise MaskedDiffusionTrainingError("group_keys length does not match dataset")

    group_to_indices: dict[str, list[int]] = {}
    for index, group_key in enumerate(group_keys):
        group_to_indices.setdefault(group_key, []).append(index)

    unique_group_keys = list(group_to_indices.keys())
    if len(unique_group_keys) <= 1:
        return indices, []

    rng.shuffle(unique_group_keys)
    target_val_size = max(1, int(round(dataset_size * val_fraction)))
    val_group_keys: list[str] = []
    val_count = 0
    for group_key in unique_group_keys:
        if len(val_group_keys) >= len(unique_group_keys) - 1:
            break
        val_group_keys.append(group_key)
        val_count += len(group_to_indices[group_key])
        if val_count >= target_val_size:
            break

    val_group_key_set = set(val_group_keys)
    train_indices = [
        index
        for index, group_key in enumerate(group_keys)
        if group_key not in val_group_key_set
    ]
    val_indices = [
        index
        for index, group_key in enumerate(group_keys)
        if group_key in val_group_key_set
    ]

    if not train_indices or not val_indices:
        return indices, []
    return train_indices, val_indices


def make_progress(iterable: Any, *, desc: str, total: int) -> Any:
    """Wrap iterable with tqdm when available."""

    if tqdm is None:
        return iterable
    return tqdm(iterable, desc=desc, total=total, leave=False)


def move_batch_to_device(
    batch: dict[str, torch.Tensor],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Move batch tensors to device."""

    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}



def make_beta_schedule(
    *,
    diffusion_train_steps: int,
    beta_schedule: str,
    beta_start: float,
    beta_end: float,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Create a beta schedule for directional stochastic diffusion.

    The returned tensor has shape ``[diffusion_train_steps]`` and is clamped to
    ``(0, 0.999)`` for numerical stability.
    """

    if diffusion_train_steps <= 0:
        raise MaskedDiffusionTrainingError("diffusion_train_steps must be positive")
    if beta_start <= 0.0 or beta_end <= 0.0:
        raise MaskedDiffusionTrainingError("beta_start and beta_end must be positive")

    schedule = str(beta_schedule).lower()
    if schedule == "linear":
        betas = torch.linspace(
            float(beta_start),
            float(beta_end),
            int(diffusion_train_steps),
            device=device,
            dtype=dtype,
        )
    elif schedule == "cosine":
        # Nichol-Dhariwal style alpha_bar, converted to beta_t. We keep the
        # same output convention as the linear branch.
        steps = int(diffusion_train_steps)
        s = 0.008
        x = torch.linspace(0, steps, steps + 1, device=device, dtype=dtype)
        alpha_bar = torch.cos(((x / steps) + s) / (1.0 + s) * math.pi * 0.5) ** 2
        alpha_bar = alpha_bar / alpha_bar[0].clamp_min(1e-12)
        betas = 1.0 - (alpha_bar[1:] / alpha_bar[:-1].clamp_min(1e-12))
    else:
        raise MaskedDiffusionTrainingError(f"unsupported beta_schedule: {beta_schedule}")

    return betas.clamp(min=1e-8, max=0.999)


def make_alpha_bar_schedule(
    *,
    diffusion_train_steps: int,
    beta_schedule: str,
    beta_start: float,
    beta_end: float,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return alpha_bar with alpha_bar[0] = 1 and length T + 1."""

    betas = make_beta_schedule(
        diffusion_train_steps=diffusion_train_steps,
        beta_schedule=beta_schedule,
        beta_start=beta_start,
        beta_end=beta_end,
        device=device,
        dtype=dtype,
    )
    alphas = 1.0 - betas
    alpha_bar = torch.cat(
        [
            torch.ones(1, device=device, dtype=dtype),
            torch.cumprod(alphas, dim=0),
        ],
        dim=0,
    )
    return alpha_bar.clamp(min=1e-8, max=1.0)


def sample_unit_directions(reference: torch.Tensor) -> torch.Tensor:
    """Sample random unit directions with the same shape/device as reference."""

    noise = torch.randn_like(reference)
    return F.normalize(noise, dim=1, eps=1e-6)


def build_stochastic_noisy_direction(
    target_direction: torch.Tensor,
    *,
    t_indices: torch.Tensor,
    alpha_bar: torch.Tensor,
) -> torch.Tensor:
    """Build x_t = normalize(sqrt(a_bar_t) x0 + sqrt(1-a_bar_t) eps).

    This is a lightweight directional analogue of DDPM forward noising on S^2.
    The target is normalized before noising because the downstream proposal is a
    direction distribution.
    """

    x0 = F.normalize(target_direction, dim=1, eps=1e-6)
    eps = sample_unit_directions(x0)
    a_bar_t = alpha_bar[t_indices].reshape(-1, 1).to(dtype=x0.dtype)
    noisy = torch.sqrt(a_bar_t) * x0 + torch.sqrt((1.0 - a_bar_t).clamp_min(0.0)) * eps
    return F.normalize(noisy, dim=1, eps=1e-6)

def build_corrupted_batch(
    batch: dict[str, torch.Tensor],
    *,
    device: torch.device,
    full_mask_probability: float,
    direction_space: str,
    stochastic_diffusion: bool = False,
    diffusion_train_steps: int = 16,
    beta_schedule: str = "linear",
    beta_start: float = 5e-4,
    beta_end: float = 5e-2,
) -> dict[str, torch.Tensor]:
    """Corrupt discrete and continuous latent variables for masked diffusion.

    When ``stochastic_diffusion`` is False this preserves the original training
    corruption: random scalar timestep, token masking, and small Gaussian
    perturbations. When it is True, the continuous direction corruption becomes
    DDPM-style directional noising:

        x_t = normalize(sqrt(alpha_bar_t) * x_0
                        + sqrt(1 - alpha_bar_t) * eps)

    with integer t sampled uniformly from {1, ..., T}. Full-mask samples use
    t=T and a random unit direction to match stochastic cold-start inference.
    """

    batch_size = int(batch["condition"].shape[0])

    if stochastic_diffusion:
        if str(getattr(stochastic_diffusion, "__class__", "")).endswith("'str'>"):
            stochastic_diffusion = str(stochastic_diffusion).lower() in {"1", "true", "yes"}
        steps = max(1, int(diffusion_train_steps))
        alpha_bar = make_alpha_bar_schedule(
            diffusion_train_steps=steps,
            beta_schedule=beta_schedule,
            beta_start=beta_start,
            beta_end=beta_end,
            device=device,
            dtype=torch.float32,
        )
        t_indices = torch.randint(
            low=1,
            high=steps + 1,
            size=(batch_size,),
            device=device,
            dtype=torch.long,
        )
        timesteps = t_indices.to(dtype=torch.float32) / float(steps)
        mask_probability = 0.15 + 0.75 * timesteps

        length_mask = torch.rand(batch_size, device=device) < mask_probability
        masked_chain_length = batch["chain_length"].clone()
        masked_chain_length[length_mask] = MASK_CHAIN_LENGTH_TOKEN

        bit_mask = torch.rand(
            batch_size,
            batch["chain_type_bits"].shape[1],
            device=device,
        ) < mask_probability.unsqueeze(1)
        masked_chain_type_bits = batch["chain_type_bits"].clone()
        masked_chain_type_bits[bit_mask] = MASK_CHAIN_TYPE_TOKEN

        noisy_direction = build_stochastic_noisy_direction(
            batch["direction"],
            t_indices=t_indices,
            alpha_bar=alpha_bar,
        )

        # Inject fully masked cold-start samples. Unlike the legacy path, the
        # continuous latent is a random unit direction instead of zero, so the
        # model learns to denoise from a stochastic prior.
        full_mask = torch.rand(batch_size, device=device) < full_mask_probability
        if bool(full_mask.any()):
            masked_chain_length[full_mask] = MASK_CHAIN_LENGTH_TOKEN
            masked_chain_type_bits[full_mask] = MASK_CHAIN_TYPE_TOKEN
            timesteps[full_mask] = 1.0
            noisy_direction[full_mask] = sample_unit_directions(
                noisy_direction[full_mask]
            )

        return {
            "condition": batch["condition"],
            "timesteps": timesteps,
            "masked_chain_length": masked_chain_length,
            "masked_chain_type_bits": masked_chain_type_bits,
            "noisy_direction": noisy_direction,
        }

    # Legacy corruption path kept for backward compatibility.
    timesteps = torch.rand(batch_size, device=device, dtype=torch.float32)
    mask_probability = 0.15 + 0.75 * timesteps

    length_mask = torch.rand(batch_size, device=device) < mask_probability
    masked_chain_length = batch["chain_length"].clone()
    masked_chain_length[length_mask] = MASK_CHAIN_LENGTH_TOKEN

    bit_mask = torch.rand(
        batch_size,
        batch["chain_type_bits"].shape[1],
        device=device,
    ) < mask_probability.unsqueeze(1)
    masked_chain_type_bits = batch["chain_type_bits"].clone()
    masked_chain_type_bits[bit_mask] = MASK_CHAIN_TYPE_TOKEN

    sigma = 0.05 + 0.45 * timesteps.unsqueeze(1)
    noisy_direction = batch["direction"] + sigma * torch.randn_like(batch["direction"])
    if direction_space == "s2":
        # S^2 variant reprojects noisy directions onto the sphere (direction-only).
        noisy_direction = F.normalize(noisy_direction, dim=1, eps=1e-6)

    # Inject fully masked samples so training matches inference cold-start masking.
    full_mask = torch.rand(batch_size, device=device) < full_mask_probability
    if bool(full_mask.any()):
        masked_chain_length[full_mask] = MASK_CHAIN_LENGTH_TOKEN
        masked_chain_type_bits[full_mask] = MASK_CHAIN_TYPE_TOKEN
        noisy_direction[full_mask] = 0.0

    return {
        "condition": batch["condition"],
        "timesteps": timesteps,
        "masked_chain_length": masked_chain_length,
        "masked_chain_type_bits": masked_chain_type_bits,
        "noisy_direction": noisy_direction,
    }


def log_vmf_normalizer_3d(kappa: torch.Tensor) -> torch.Tensor:
    """Log normalization constant for vMF on S^2."""

    safe_kappa = kappa.float().clamp_min(1e-6)
    result = torch.empty_like(safe_kappa)
    small_mask = safe_kappa < 1e-5
    large_mask = safe_kappa > 20.0
    middle_mask = ~(small_mask | large_mask)

    result[small_mask] = -math.log(4.0 * math.pi)
    if large_mask.any():
        large_kappa = safe_kappa[large_mask]
        log_sinh = (
            large_kappa
            - math.log(2.0)
            + torch.log1p(-torch.exp(-2.0 * large_kappa))
        )
        result[large_mask] = (
            torch.log(large_kappa) - math.log(4.0 * math.pi) - log_sinh
        )
    if middle_mask.any():
        middle_kappa = safe_kappa[middle_mask]
        log_sinh = torch.log(torch.sinh(middle_kappa).clamp_min(1e-20))
        result[middle_mask] = (
            torch.log(middle_kappa) - math.log(4.0 * math.pi) - log_sinh
        )
    return result


def compute_vmf_mixture_nll(
    outputs: dict[str, torch.Tensor],
    target_direction: torch.Tensor,
    *,
    vmf_max_kappa: float,
) -> torch.Tensor:
    """Direction negative log-likelihood for a K-component vMF mixture."""

    required_keys = {"vmf_logits", "vmf_means", "vmf_kappas"}
    if not required_keys.issubset(outputs):
        raise MaskedDiffusionTrainingError(
            "direction_loss_type=vmf_mixture requires vMF mixture outputs"
        )

    target_direction = F.normalize(target_direction.float(), dim=1, eps=1e-6)
    log_weights = F.log_softmax(outputs["vmf_logits"].float(), dim=1)
    means = F.normalize(outputs["vmf_means"].float(), dim=2, eps=1e-6)
    kappas = outputs["vmf_kappas"].float().clamp(min=1e-6, max=vmf_max_kappa)
    cosine = (
        means * target_direction.unsqueeze(1)
    ).sum(dim=2).clamp(min=-1.0, max=1.0)
    component_log_prob = (
        log_weights + log_vmf_normalizer_3d(kappas) + kappas * cosine
    )
    return -torch.logsumexp(component_log_prob, dim=1)


def make_spherical_d3pm_forward_matrices(
    *,
    state_count: int,
    step_count: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build uniform-replacement forward kernels and cumulative products.

    ``Q_t = (1-beta_t) I + beta_t U`` with ``beta_T = 1`` makes the terminal
    law exactly uniform.  The cumulative tensor includes ``Qbar_0 = I``.
    """

    if state_count < 2:
        raise MaskedDiffusionTrainingError("state_count must be at least 2")
    if step_count <= 0:
        raise MaskedDiffusionTrainingError("step_count must be positive")
    identity = torch.eye(state_count, device=device, dtype=dtype)
    uniform = torch.full_like(identity, 1.0 / float(state_count))
    betas = torch.arange(
        1,
        step_count + 1,
        device=device,
        dtype=dtype,
    ) / float(step_count)
    forward = torch.stack(
        [(1.0 - beta) * identity + beta * uniform for beta in betas],
        dim=0,
    )
    cumulative = [identity]
    for transition in forward:
        cumulative.append(cumulative[-1] @ transition)
    return forward, torch.stack(cumulative, dim=0)


def spherical_d3pm_target_states(
    target_direction: torch.Tensor,
    anchors: torch.Tensor,
) -> torch.Tensor:
    """Quantize directions to their nearest fixed spherical state."""

    target = F.normalize(target_direction.float(), dim=1, eps=1e-6)
    geometry = anchors.detach().to(device=target.device, dtype=target.dtype)
    if geometry.ndim == 2:
        normalized_anchors = F.normalize(geometry, dim=1, eps=1e-12)
        scores = target @ normalized_anchors.transpose(0, 1)
    elif geometry.ndim == 3 and geometry.shape[0] == target.shape[0]:
        normalized_anchors = F.normalize(geometry, dim=2, eps=1e-12)
        scores = torch.bmm(
            normalized_anchors,
            target.unsqueeze(2),
        ).squeeze(2)
    else:
        raise MaskedDiffusionTrainingError(
            "anchors must have shape [states, 3] or [batch, states, 3]"
        )
    # State assignment is a discrete supervisory label. Stop gradients through
    # the learned geometry while allowing transition similarity to train it.
    return scores.argmax(dim=1)


def spherical_d3pm_posterior_kl_per_sample(
    *,
    target_direction: torch.Tensor,
    anchors: torch.Tensor,
    reverse_matrices: torch.Tensor,
) -> torch.Tensor:
    """Exact expected posterior KL for every reverse diffusion step.

    ``reverse_matrices`` is ordered ``P_T, ..., P_1``.  Instead of sampling a
    noisy state, this function sums over every ``z_t`` under the known forward
    process, producing a deterministic low-variance denoising objective.
    """

    if reverse_matrices.ndim != 4:
        raise MaskedDiffusionTrainingError(
            "reverse_matrices must have shape [batch, steps, states, states]"
        )
    batch_size, step_count, state_count, destination_count = (
        reverse_matrices.shape
    )
    if destination_count != state_count:
        raise MaskedDiffusionTrainingError("reverse matrices must be square")
    if target_direction.shape[0] != batch_size:
        raise MaskedDiffusionTrainingError(
            "target direction batch does not match reverse matrices"
        )
    if anchors.ndim == 2:
        if anchors.shape != (state_count, 3):
            raise MaskedDiffusionTrainingError(
                "anchor shape does not match reverse state count"
            )
    elif anchors.shape != (batch_size, state_count, 3):
        raise MaskedDiffusionTrainingError(
            "batched anchor shape does not match reverse states"
        )

    work_dtype = torch.float32
    reverse = reverse_matrices.to(dtype=work_dtype)
    forward, cumulative = make_spherical_d3pm_forward_matrices(
        state_count=state_count,
        step_count=step_count,
        device=reverse.device,
        dtype=work_dtype,
    )
    target_states = spherical_d3pm_target_states(
        target_direction,
        anchors,
    )
    total_kl = torch.zeros(batch_size, device=reverse.device, dtype=work_dtype)

    for step in range(1, step_count + 1):
        q_previous = cumulative[step - 1].index_select(0, target_states)
        q_current = cumulative[step].index_select(0, target_states)
        # joint[b, previous, current] = q(z_{t-1}|z0) q(z_t|z_{t-1})
        joint = q_previous.unsqueeze(2) * forward[step - 1].unsqueeze(0)
        posterior = joint.transpose(1, 2) / q_current.unsqueeze(2).clamp_min(1e-30)
        predicted = reverse[:, step_count - step, :, :].clamp_min(1e-30)
        posterior_log = torch.log(posterior.clamp_min(1e-30))
        predicted_log = torch.log(predicted)
        conditional_kl = (
            posterior * (posterior_log - predicted_log)
        ).sum(dim=2)
        total_kl = total_kl + (q_current * conditional_kl).sum(dim=1)

    return total_kl / float(step_count)


def compute_spherical_d3pm_direction_losses(
    *,
    model: MaskedSeedDiffusionModel,
    condition: torch.Tensor,
    chain_length: torch.Tensor,
    chain_type_bits: torch.Tensor,
    target_direction: torch.Tensor,
    denoising_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Compute exact endpoint NLL plus posterior denoising KL per record."""

    endpoint = predict_spherical_d3pm_endpoint(
        model,
        condition=condition,
        chain_length=chain_length,
        chain_type_bits=chain_type_bits,
    )
    endpoint_nll = compute_vmf_mixture_nll(
        endpoint,
        target_direction,
        vmf_max_kappa=float(model.config.direction_grid_kappa),
    )
    if (
        model.config.direction_state_geometry
        == SPHERICAL_D3PM_GEOMETRY_LEARNED_EMISSION
    ):
        target_geometry = endpoint["direction_emission_means"].detach()
    else:
        target_geometry = model.direction_anchors
    posterior_kl = spherical_d3pm_posterior_kl_per_sample(
        target_direction=target_direction,
        anchors=target_geometry,
        reverse_matrices=endpoint["direction_reverse_matrices"],
    )
    total = endpoint_nll + float(denoising_weight) * posterior_kl
    return total, endpoint_nll, posterior_kl, endpoint


def compute_direction_loss_per_sample(
    outputs: dict[str, torch.Tensor],
    target_direction: torch.Tensor,
    *,
    direction_loss_type: str,
    direction_vmf_kappa: float,
    vmf_max_kappa: float = 32.0,
    direction_space: str = "r3",
) -> torch.Tensor:
    """Compute one directional loss value per replay record."""

    predicted_direction = outputs["direction"]
    if direction_space == "s2":
        target_direction = F.normalize(target_direction, dim=1, eps=1e-6)
        predicted_direction = F.normalize(predicted_direction, dim=1, eps=1e-6)
    if direction_loss_type == "mse":
        return F.mse_loss(
            predicted_direction,
            target_direction,
            reduction="none",
        ).mean(dim=1)
    if direction_loss_type == "vmf":
        # Constant-kappa vMF NLL, up to the additive normalization constant.
        predicted_direction = F.normalize(predicted_direction, dim=1, eps=1e-6)
        target_direction = F.normalize(target_direction, dim=1, eps=1e-6)
        cosine_similarity = (
            (predicted_direction * target_direction)
            .sum(dim=1)
            .clamp(min=-1.0, max=1.0)
        )
        return (1.0 - cosine_similarity) * direction_vmf_kappa
    if direction_loss_type == "vmf_mixture":
        return compute_vmf_mixture_nll(
            outputs,
            target_direction,
            vmf_max_kappa=vmf_max_kappa,
        )
    raise MaskedDiffusionTrainingError(
        f"unsupported direction_loss_type: {direction_loss_type}"
    )


def _fit_energy_weight_cap(
    dataset: Dataset[dict[str, torch.Tensor]],
    train_indices: list[int],
    *,
    weight_mode: str,
    weight_quantile: float,
    weight_name: str,
) -> float | None:
    """Fit one optional raw-energy cap using training records only.

    The default ``log_energy`` path needs no fitted state and preserves the
    historical sample weights.  For ``clipped_raw_energy``, validation records
    are intentionally excluded so their energies cannot influence training or
    checkpoint selection.
    """

    if weight_mode == DESCRIPTOR_WEIGHT_LOG_ENERGY:
        return None
    if weight_mode != DESCRIPTOR_WEIGHT_CLIPPED_RAW_ENERGY:
        raise MaskedDiffusionTrainingError(
            f"unsupported {weight_name}_weight_mode: {weight_mode!r}"
        )
    if weight_quantile not in SUPPORTED_DESCRIPTOR_WEIGHT_QUANTILES:
        raise MaskedDiffusionTrainingError(
            f"{weight_name}_weight_quantile must be 0.95 or 0.99"
        )
    if not train_indices:
        raise MaskedDiffusionTrainingError(
            f"cannot fit {weight_name} energy cap from an empty training split"
        )

    raw_energies = torch.stack(
        [dataset[index]["raw_energy"].float() for index in train_indices]
    )
    if not bool(torch.isfinite(raw_energies).all()):
        raise MaskedDiffusionTrainingError(
            f"raw replay energies must be finite to fit {weight_name} weights"
        )
    raw_energies = raw_energies.clamp_min(0.0)
    cap = float(torch.quantile(raw_energies, weight_quantile).item())
    if not math.isfinite(cap) or cap <= 0.0:
        raise MaskedDiffusionTrainingError(
            "clipped_raw_energy requires a positive train-split energy "
            f"quantile for {weight_name}"
        )
    return cap


def fit_descriptor_weight_cap(
    dataset: Dataset[dict[str, torch.Tensor]],
    train_indices: list[int],
    *,
    descriptor_weight_mode: str,
    descriptor_weight_quantile: float,
) -> float | None:
    """Fit the descriptor raw-energy cap from training records only."""

    return _fit_energy_weight_cap(
        dataset,
        train_indices,
        weight_mode=descriptor_weight_mode,
        weight_quantile=descriptor_weight_quantile,
        weight_name="descriptor",
    )


def fit_direction_weight_cap(
    dataset: Dataset[dict[str, torch.Tensor]],
    train_indices: list[int],
    *,
    direction_weight_mode: str,
    direction_weight_quantile: float,
) -> float | None:
    """Fit the direction raw-energy cap from training records only."""

    return _fit_energy_weight_cap(
        dataset,
        train_indices,
        weight_mode=direction_weight_mode,
        weight_quantile=direction_weight_quantile,
        weight_name="direction",
    )


def _sample_weights_for_mode(
    batch: dict[str, torch.Tensor],
    *,
    weight_mode: str,
    weight_cap: float | None,
    weight_name: str,
) -> torch.Tensor:
    """Resolve one head family's weights from a replay batch."""

    if weight_mode == DESCRIPTOR_WEIGHT_LOG_ENERGY:
        return batch["sample_weight"]
    if weight_mode != DESCRIPTOR_WEIGHT_CLIPPED_RAW_ENERGY:
        raise MaskedDiffusionTrainingError(
            f"unsupported {weight_name}_weight_mode: {weight_mode!r}"
        )
    if weight_cap is None or not math.isfinite(weight_cap):
        raise MaskedDiffusionTrainingError(
            f"clipped_raw_energy requires a finite {weight_name}_weight_cap"
        )
    if weight_cap <= 0.0:
        raise MaskedDiffusionTrainingError(
            f"{weight_name}_weight_cap must be positive"
        )
    if "raw_energy" not in batch:
        raise MaskedDiffusionTrainingError(
            "clipped_raw_energy requires raw_energy in every replay batch"
        )
    raw_energy = batch["raw_energy"].to(
        device=batch["sample_weight"].device,
        dtype=batch["sample_weight"].dtype,
    )
    if not bool(torch.isfinite(raw_energy).all()):
        raise MaskedDiffusionTrainingError("raw replay energies must be finite")
    return raw_energy.clamp(min=0.0, max=float(weight_cap))


def descriptor_sample_weights(
    batch: dict[str, torch.Tensor],
    *,
    descriptor_weight_mode: str = DESCRIPTOR_WEIGHT_LOG_ENERGY,
    descriptor_weight_cap: float | None = None,
) -> torch.Tensor:
    """Return per-record weights for length/type losses.

    This is independent of direction weighting, enabling descriptor-only and
    both-head ablations without changing the historical default.
    """

    return _sample_weights_for_mode(
        batch,
        weight_mode=descriptor_weight_mode,
        weight_cap=descriptor_weight_cap,
        weight_name="descriptor",
    )


def direction_sample_weights(
    batch: dict[str, torch.Tensor],
    *,
    direction_weight_mode: str = DIRECTION_WEIGHT_LOG_ENERGY,
    direction_weight_cap: float | None = None,
) -> torch.Tensor:
    """Return independently configured per-record direction-loss weights."""

    return _sample_weights_for_mode(
        batch,
        weight_mode=direction_weight_mode,
        weight_cap=direction_weight_cap,
        weight_name="direction",
    )


def weighted_mean_per_sample(
    values: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """Reduce per-record values with a numerically safe normalized weight."""

    weights = weights.to(device=values.device, dtype=values.dtype)
    return (values * weights).sum() / weights.sum().clamp_min(1e-6)


def compute_losses(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    *,
    direction_loss_weight: float,
    direction_loss_type: str,
    direction_vmf_kappa: float,
    vmf_max_kappa: float = 32.0,
    direction_space: str = "r3",
    descriptor_weight_mode: str = DESCRIPTOR_WEIGHT_LOG_ENERGY,
    descriptor_weight_cap: float | None = None,
    direction_weight_mode: str = DIRECTION_WEIGHT_LOG_ENERGY,
    direction_weight_cap: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute training losses."""

    chain_length_loss = F.cross_entropy(
        outputs["chain_length_logits"],
        batch["chain_length"],
        reduction="none",
    )
    chain_type_loss = F.binary_cross_entropy_with_logits(
        outputs["chain_type_logits"],
        batch["chain_type_bits"].float(),
        reduction="none",
    )
    valid_mask = batch["chain_type_valid"]
    chain_type_loss = (chain_type_loss * valid_mask).sum(dim=1) / valid_mask.sum(
        dim=1
    ).clamp_min(1.0)
    direction_loss = compute_direction_loss_per_sample(
        outputs,
        batch["direction"],
        direction_loss_type=direction_loss_type,
        direction_vmf_kappa=direction_vmf_kappa,
        vmf_max_kappa=vmf_max_kappa,
        direction_space=direction_space,
    )

    direction_weight = direction_sample_weights(
        batch,
        direction_weight_mode=direction_weight_mode,
        direction_weight_cap=direction_weight_cap,
    )
    descriptor_weight = descriptor_sample_weights(
        batch,
        descriptor_weight_mode=descriptor_weight_mode,
        descriptor_weight_cap=descriptor_weight_cap,
    )
    descriptor_loss = weighted_mean_per_sample(
        chain_length_loss + chain_type_loss,
        descriptor_weight,
    )
    weighted_direction_loss = weighted_mean_per_sample(
        direction_loss,
        direction_weight,
    )
    total_loss = descriptor_loss + direction_loss_weight * weighted_direction_loss

    return total_loss, {
        "chain_length": chain_length_loss.detach().mean(),
        "chain_type": chain_type_loss.detach().mean(),
        "direction": direction_loss.detach().mean(),
    }


def compute_ordered_absorbing_losses(
    *,
    model: nn.Module,
    batch: dict[str, torch.Tensor],
    direction_loss_weight: float,
    direction_loss_type: str,
    direction_vmf_kappa: float,
    vmf_max_kappa: float = 32.0,
    direction_space: str = "r3",
    direction_denoising_weight: float = 0.25,
    sample_one_type_transition: bool = True,
    descriptor_weight_mode: str = DESCRIPTOR_WEIGHT_LOG_ENERGY,
    descriptor_weight_cap: float | None = None,
    direction_weight_mode: str = DIRECTION_WEIGHT_LOG_ENERGY,
    direction_weight_cap: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute leakage-free ordered absorbing descriptor losses.

    Length is predicted only from the all-mask state. Type diffusion reveals a
    strict left-to-right prefix and supervises only the next valid bit. During
    training, one uniformly sampled transition multiplied by chain length is an
    unbiased estimator of the full descriptor NLL. Validation can enumerate all
    valid transitions in one expanded batch. Direction is predicted in a
    separate pass conditioned on the completed descriptor and never receives
    the target direction as input.
    """

    chain_length = batch["chain_length"].long()
    chain_type_bits = batch["chain_type_bits"].long()
    batch_size = int(chain_length.shape[0])
    if batch_size <= 0:
        raise MaskedDiffusionTrainingError("ordered absorbing batch is empty")
    if bool(
        ((chain_length < 1) | (chain_length > MAX_CHAIN_TYPE_BITS)).any().item()
    ):
        raise MaskedDiffusionTrainingError(
            "ordered absorbing mode requires chain lengths in "
            f"[1, {MAX_CHAIN_TYPE_BITS}]"
        )

    length_outputs = model(**build_ordered_length_inputs(batch["condition"]))
    length_log_probs = ordered_valid_length_log_probs(
        length_outputs["chain_length_logits"]
    )
    chain_length_loss = F.nll_loss(
        length_log_probs,
        chain_length - 1,
        reduction="none",
    )

    if sample_one_type_transition:
        prefix_length = torch.floor(
            torch.rand(
                batch_size,
                device=chain_length.device,
                dtype=batch["condition"].dtype,
            )
            * chain_length.to(dtype=batch["condition"].dtype)
        ).long()
        type_outputs = model(
            **build_ordered_type_inputs(
                condition=batch["condition"],
                chain_length=chain_length,
                chain_type_bits=chain_type_bits,
                prefix_length=prefix_length,
            )
        )
        next_logits = type_outputs["chain_type_logits"].gather(
            1,
            prefix_length.unsqueeze(1),
        ).squeeze(1)
        next_targets = chain_type_bits.gather(
            1,
            prefix_length.unsqueeze(1),
        ).squeeze(1).float()
        transition_loss = F.binary_cross_entropy_with_logits(
            next_logits,
            next_targets,
            reduction="none",
        )
        # E_{k uniform in [0,n)}[n * loss_k] is the exact type-string NLL.
        chain_type_loss = transition_loss * chain_length.to(
            dtype=transition_loss.dtype
        )
    else:
        positions = torch.arange(
            MAX_CHAIN_TYPE_BITS,
            device=chain_length.device,
            dtype=torch.long,
        ).unsqueeze(0).expand(batch_size, -1)
        active = positions < chain_length.unsqueeze(1)
        sample_indices = torch.arange(
            batch_size,
            device=chain_length.device,
            dtype=torch.long,
        ).unsqueeze(1).expand_as(positions)[active]
        prefix_length = positions[active]
        expanded_bits = chain_type_bits[sample_indices]
        type_outputs = model(
            **build_ordered_type_inputs(
                condition=batch["condition"][sample_indices],
                chain_length=chain_length[sample_indices],
                chain_type_bits=expanded_bits,
                prefix_length=prefix_length,
            )
        )
        next_logits = type_outputs["chain_type_logits"].gather(
            1,
            prefix_length.unsqueeze(1),
        ).squeeze(1)
        next_targets = expanded_bits.gather(
            1,
            prefix_length.unsqueeze(1),
        ).squeeze(1).float()
        transition_loss = F.binary_cross_entropy_with_logits(
            next_logits,
            next_targets,
            reduction="none",
        )
        chain_type_loss = torch.zeros(
            batch_size,
            device=transition_loss.device,
            dtype=transition_loss.dtype,
        ).scatter_add(0, sample_indices, transition_loss)

    if (
        getattr(getattr(model, "config", None), "direction_distribution", None)
        == DIRECTION_DISTRIBUTION_SPHERICAL_D3PM_VMF
    ):
        (
            direction_loss,
            direction_endpoint_nll,
            direction_posterior_kl,
            _direction_outputs,
        ) = compute_spherical_d3pm_direction_losses(
            model=model,
            condition=batch["condition"],
            chain_length=chain_length,
            chain_type_bits=chain_type_bits,
            target_direction=batch["direction"],
            denoising_weight=direction_denoising_weight,
        )
    else:
        direction_outputs = model(
            **build_ordered_direction_inputs(
                condition=batch["condition"],
                chain_length=chain_length,
                chain_type_bits=chain_type_bits,
            )
        )
        direction_loss = compute_direction_loss_per_sample(
            direction_outputs,
            batch["direction"],
            direction_loss_type=direction_loss_type,
            direction_vmf_kappa=direction_vmf_kappa,
            vmf_max_kappa=vmf_max_kappa,
            direction_space=direction_space,
        )
        direction_endpoint_nll = direction_loss
        direction_posterior_kl = torch.zeros_like(direction_loss)

    direction_weight = direction_sample_weights(
        batch,
        direction_weight_mode=direction_weight_mode,
        direction_weight_cap=direction_weight_cap,
    )
    descriptor_weight = descriptor_sample_weights(
        batch,
        descriptor_weight_mode=descriptor_weight_mode,
        descriptor_weight_cap=descriptor_weight_cap,
    )
    descriptor_loss = weighted_mean_per_sample(
        chain_length_loss + chain_type_loss,
        descriptor_weight,
    )
    weighted_direction_loss = weighted_mean_per_sample(
        direction_loss,
        direction_weight,
    )
    total_loss = descriptor_loss + direction_loss_weight * weighted_direction_loss

    return total_loss, {
        "chain_length": chain_length_loss.detach().mean(),
        "chain_type": chain_type_loss.detach().mean(),
        "direction": direction_loss.detach().mean(),
        "direction_endpoint_nll": direction_endpoint_nll.detach().mean(),
        "direction_posterior_kl": direction_posterior_kl.detach().mean(),
    }


def evaluate_model(
    *,
    model: nn.Module,
    data_loader: DataLoader[dict[str, torch.Tensor]],
    device: torch.device,
    direction_loss_weight: float,
    direction_loss_type: str,
    direction_vmf_kappa: float,
    vmf_max_kappa: float,
    full_mask_probability: float,
    direction_space: str,
    direction_denoising_weight: float = 0.25,
    stochastic_diffusion: bool = False,
    diffusion_train_steps: int = 16,
    beta_schedule: str = "linear",
    beta_start: float = 5e-4,
    beta_end: float = 5e-2,
    discrete_diffusion_mode: str = DISCRETE_DIFFUSION_LEGACY,
    descriptor_weight_mode: str = DESCRIPTOR_WEIGHT_LOG_ENERGY,
    descriptor_weight_cap: float | None = None,
    direction_weight_mode: str = DIRECTION_WEIGHT_LOG_ENERGY,
    direction_weight_cap: float | None = None,
) -> tuple[float, float, float, float]:
    """Evaluate model on validation loader."""

    model.eval()
    total_loss = 0.0
    total_chain_length_loss = 0.0
    total_chain_type_loss = 0.0
    total_direction_loss = 0.0
    finite_batches = 0

    with torch.no_grad():
        for batch in make_progress(data_loader, desc="val", total=len(data_loader)):
            batch = move_batch_to_device(batch, device)
            with torch.amp.autocast(
                device_type=device.type,
                enabled=device.type == "cuda",
            ):
                if discrete_diffusion_mode == DISCRETE_DIFFUSION_ORDERED_ABSORBING:
                    loss, parts = compute_ordered_absorbing_losses(
                        model=model,
                        batch=batch,
                        direction_loss_weight=direction_loss_weight,
                        direction_loss_type=direction_loss_type,
                        direction_vmf_kappa=direction_vmf_kappa,
                        vmf_max_kappa=vmf_max_kappa,
                        direction_space=direction_space,
                        direction_denoising_weight=direction_denoising_weight,
                        sample_one_type_transition=False,
                        descriptor_weight_mode=descriptor_weight_mode,
                        descriptor_weight_cap=descriptor_weight_cap,
                        direction_weight_mode=direction_weight_mode,
                        direction_weight_cap=direction_weight_cap,
                    )
                elif discrete_diffusion_mode == DISCRETE_DIFFUSION_LEGACY:
                    corrupted = build_corrupted_batch(
                        batch,
                        device=device,
                        full_mask_probability=full_mask_probability,
                        direction_space=direction_space,
                        stochastic_diffusion=stochastic_diffusion,
                        diffusion_train_steps=diffusion_train_steps,
                        beta_schedule=beta_schedule,
                        beta_start=beta_start,
                        beta_end=beta_end,
                    )
                    outputs = model(**corrupted)
                    loss, parts = compute_losses(
                        outputs,
                        batch,
                        direction_loss_weight=direction_loss_weight,
                        direction_loss_type=direction_loss_type,
                        direction_vmf_kappa=direction_vmf_kappa,
                        vmf_max_kappa=vmf_max_kappa,
                        direction_space=direction_space,
                        descriptor_weight_mode=descriptor_weight_mode,
                        descriptor_weight_cap=descriptor_weight_cap,
                        direction_weight_mode=direction_weight_mode,
                        direction_weight_cap=direction_weight_cap,
                    )
                else:
                    raise MaskedDiffusionTrainingError(
                        "unsupported discrete_diffusion_mode: "
                        f"{discrete_diffusion_mode}"
                    )
            if not torch.isfinite(loss):
                LOGGER.warning("skip non-finite validation loss")
                continue
            total_loss += float(loss.item())
            total_chain_length_loss += float(parts["chain_length"].item())
            total_chain_type_loss += float(parts["chain_type"].item())
            total_direction_loss += float(parts["direction"].item())
            finite_batches += 1

    if finite_batches == 0:
        return (float("inf"), float("inf"), float("inf"), float("inf"))
    batch_count = float(finite_batches)
    return (
        total_loss / batch_count,
        total_chain_length_loss / batch_count,
        total_chain_type_loss / batch_count,
        total_direction_loss / batch_count,
    )


def save_checkpoint(
    checkpoint_path: Path,
    *,
    model: nn.Module,
    optimizer: AdamW,
    epoch: int,
    metrics: EpochMetrics,
    config: MaskedDiffusionTrainingConfig,
) -> None:
    """Persist training checkpoint."""

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


def write_metrics_csv(path: Path, history: tuple[EpochMetrics, ...]) -> None:
    """Flush per-epoch metrics to CSV."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(history[0]).keys()))
        writer.writeheader()
        for row in history:
            writer.writerow(asdict(row))


def create_run_dir(output_root: Path) -> Path:
    """Create timestamped directory for current training run."""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_root / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def publish_general_checkpoint(
    *,
    output_root: Path,
    best_checkpoint: Path,
) -> Path:
    """Copy best checkpoint into a stable entry-point filename.

    Args:
        output_root: phase_10 output root directory.
        best_checkpoint: Best checkpoint discovered this run.

    Returns:
        Path to copied stable checkpoint.
    """

    stable_checkpoint = output_root / "best_general.pt"
    # phase_11/phase_12 cold starts read this symlink-free entry.
    shutil.copy2(best_checkpoint, stable_checkpoint)
    return stable_checkpoint


def run_training(
    *,
    workspace_root: Path,
    config: MaskedDiffusionTrainingConfig,
) -> TrainingArtifacts:
    """Run masked diffusion training end-to-end."""

    configure_logging()
    validate_loss_weight_config(config)
    set_global_seed(config.seed)
    device = detect_device(config.device)

    replay_paths = tuple(
        (workspace_root / path).resolve() for path in config.replay_paths
    )
    dataset = ReplayTrainingDataset(replay_paths, max_records=config.max_records)
    LOGGER.info(
        "dataset: records=%s source_files=%s filtered=%s mean_weight=%.4f",
        dataset.stats.num_records,
        dataset.stats.num_source_files,
        dataset.stats.num_filtered_records,
        dataset.stats.mean_sample_weight,
    )

    train_indices, val_indices = split_dataset_indices(
        len(dataset),
        val_fraction=config.val_fraction,
        seed=config.seed,
        group_keys=dataset.group_keys,
    )
    descriptor_weight_cap = fit_descriptor_weight_cap(
        dataset,
        train_indices,
        descriptor_weight_mode=config.descriptor_weight_mode,
        descriptor_weight_quantile=config.descriptor_weight_quantile,
    )
    direction_weight_cap = fit_direction_weight_cap(
        dataset,
        train_indices,
        direction_weight_mode=config.direction_weight_mode,
        direction_weight_quantile=config.direction_weight_quantile,
    )
    LOGGER.info(
        "descriptor weighting: mode=%s quantile=%.2f cap=%s",
        config.descriptor_weight_mode,
        config.descriptor_weight_quantile,
        "n/a" if descriptor_weight_cap is None else f"{descriptor_weight_cap:.6g}",
    )
    LOGGER.info(
        "direction weighting: mode=%s quantile=%.2f cap=%s",
        config.direction_weight_mode,
        config.direction_weight_quantile,
        "n/a" if direction_weight_cap is None else f"{direction_weight_cap:.6g}",
    )
    train_dataset: Dataset[dict[str, torch.Tensor]] = Subset(dataset, train_indices)
    val_dataset: Dataset[dict[str, torch.Tensor]] | None = (
        Subset(dataset, val_indices) if val_indices else None
    )

    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_dataset,
        batch_size=min(config.batch_size, len(train_indices)),
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=min(config.batch_size, len(val_indices)),
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=pin_memory,
            drop_last=False,
        )

    model = MaskedSeedDiffusionModel(build_model_config(config)).to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    output_root = (workspace_root / config.output_dir).resolve()
    run_dir = create_run_dir(output_root)
    (run_dir / "train_config.json").write_text(
        json.dumps(asdict(config), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    history: list[EpochMetrics] = []
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    optimizer_steps = 0
    best_checkpoint = run_dir / "best.pt"
    last_checkpoint = run_dir / "last.pt"
    training_start = time.perf_counter()

    for epoch in range(1, config.epochs + 1):
        model.train()
        train_loss = 0.0
        train_chain_length_loss = 0.0
        train_chain_type_loss = 0.0
        train_direction_loss = 0.0
        finite_batches = 0
        nonfinite_batches = 0

        for batch in make_progress(
            train_loader,
            desc=f"train {epoch}",
            total=len(train_loader),
        ):
            batch = move_batch_to_device(batch, device)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(
                device_type=device.type,
                enabled=device.type == "cuda",
            ):
                if (
                    config.discrete_diffusion_mode
                    == DISCRETE_DIFFUSION_ORDERED_ABSORBING
                ):
                    loss, parts = compute_ordered_absorbing_losses(
                        model=model,
                        batch=batch,
                        direction_loss_weight=config.direction_loss_weight,
                        direction_loss_type=config.direction_loss_type,
                        direction_vmf_kappa=config.direction_vmf_kappa,
                        vmf_max_kappa=config.vmf_max_kappa,
                        direction_space=config.direction_space,
                        direction_denoising_weight=config.direction_denoising_weight,
                        sample_one_type_transition=True,
                        descriptor_weight_mode=config.descriptor_weight_mode,
                        descriptor_weight_cap=descriptor_weight_cap,
                        direction_weight_mode=config.direction_weight_mode,
                        direction_weight_cap=direction_weight_cap,
                    )
                else:
                    corrupted = build_corrupted_batch(
                        batch,
                        device=device,
                        full_mask_probability=config.full_mask_probability,
                        direction_space=config.direction_space,
                        stochastic_diffusion=config.stochastic_diffusion,
                        diffusion_train_steps=config.diffusion_train_steps,
                        beta_schedule=config.beta_schedule,
                        beta_start=config.beta_start,
                        beta_end=config.beta_end,
                    )
                    outputs = model(**corrupted)
                    loss, parts = compute_losses(
                        outputs,
                        batch,
                        direction_loss_weight=config.direction_loss_weight,
                        direction_loss_type=config.direction_loss_type,
                        direction_vmf_kappa=config.direction_vmf_kappa,
                        vmf_max_kappa=config.vmf_max_kappa,
                        direction_space=config.direction_space,
                        descriptor_weight_mode=config.descriptor_weight_mode,
                        descriptor_weight_cap=descriptor_weight_cap,
                        direction_weight_mode=config.direction_weight_mode,
                        direction_weight_cap=direction_weight_cap,
                    )
            if not torch.isfinite(loss):
                nonfinite_batches += 1
                LOGGER.warning(
                    "skip non-finite training loss at epoch=%s batch=%s/%s",
                    epoch,
                    nonfinite_batches,
                    config.max_nonfinite_batches,
                )
                if nonfinite_batches >= config.max_nonfinite_batches:
                    LOGGER.warning(
                        "stop epoch=%s after repeated non-finite training losses",
                        epoch,
                    )
                    break
                continue
            scaler.scale(loss).backward()
            if config.grad_clip_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=float(config.grad_clip_norm),
                )
            scaler.step(optimizer)
            scaler.update()
            optimizer_steps += 1

            train_loss += float(loss.item())
            train_chain_length_loss += float(parts["chain_length"].item())
            train_chain_type_loss += float(parts["chain_type"].item())
            train_direction_loss += float(parts["direction"].item())
            finite_batches += 1

        if finite_batches == 0:
            LOGGER.warning("early stop: no finite training batch at epoch=%s", epoch)
            break
        batch_count = float(finite_batches)
        val_loss: float | None = None
        val_chain_length_loss: float | None = None
        val_chain_type_loss: float | None = None
        val_direction_loss: float | None = None
        if val_loader is not None and len(val_loader) > 0:
            (
                val_loss,
                val_chain_length_loss,
                val_chain_type_loss,
                val_direction_loss,
            ) = evaluate_model(
                model=model,
                data_loader=val_loader,
                device=device,
                direction_loss_weight=config.direction_loss_weight,
                direction_loss_type=config.direction_loss_type,
                direction_vmf_kappa=config.direction_vmf_kappa,
                vmf_max_kappa=config.vmf_max_kappa,
                full_mask_probability=config.full_mask_probability,
                direction_space=config.direction_space,
                direction_denoising_weight=config.direction_denoising_weight,
                stochastic_diffusion=config.stochastic_diffusion,
                diffusion_train_steps=config.diffusion_train_steps,
                beta_schedule=config.beta_schedule,
                beta_start=config.beta_start,
                beta_end=config.beta_end,
                discrete_diffusion_mode=config.discrete_diffusion_mode,
                descriptor_weight_mode=config.descriptor_weight_mode,
                descriptor_weight_cap=descriptor_weight_cap,
                direction_weight_mode=config.direction_weight_mode,
                direction_weight_cap=direction_weight_cap,
            )

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
        save_checkpoint(
            last_checkpoint,
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            metrics=metrics,
            config=config,
        )
        if val_loss is None:
            if epoch == config.epochs:
                save_checkpoint(
                    best_checkpoint,
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    metrics=metrics,
                    config=config,
                )
                publish_general_checkpoint(
                    output_root=output_root,
                    best_checkpoint=best_checkpoint,
                )
        elif math.isfinite(val_loss) and val_loss <= best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            save_checkpoint(
                best_checkpoint,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                metrics=metrics,
                config=config,
            )
            publish_general_checkpoint(
                output_root=output_root,
                best_checkpoint=best_checkpoint,
            )
        elif val_loss is not None and config.early_stop_patience is not None:
            epochs_without_improvement += 1

        LOGGER.info(
            "epoch=%s train_loss=%.6f val_loss=%s",
            epoch,
            metrics.train_loss,
            "n/a" if metrics.val_loss is None else f"{metrics.val_loss:.6f}",
        )
        if (
            config.early_stop_patience is not None
            and val_loss is not None
            and epochs_without_improvement >= config.early_stop_patience
        ):
            LOGGER.info(
                "early stop: no validation improvement for %s epochs",
                config.early_stop_patience,
            )
            break

    if not history:
        raise MaskedDiffusionTrainingError("training stopped before any finite epoch")

    training_seconds = time.perf_counter() - training_start

    if not best_checkpoint.exists() and last_checkpoint.exists():
        shutil.copy2(last_checkpoint, best_checkpoint)

    metrics_json = run_dir / "metrics.json"
    metrics_csv = run_dir / "metrics.csv"
    metrics_json.write_text(
        json.dumps([asdict(item) for item in history], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_metrics_csv(metrics_csv, tuple(history))
    publish_general_checkpoint(output_root=output_root, best_checkpoint=best_checkpoint)
    (run_dir / "run_summary.json").write_text(
        json.dumps(
            {
                "model_parameter_count": sum(
                    parameter.numel() for parameter in model.parameters()
                ),
                "trainable_parameter_count": sum(
                    parameter.numel()
                    for parameter in model.parameters()
                    if parameter.requires_grad
                ),
                "training_seconds": training_seconds,
                "epochs_completed": len(history),
                "optimizer_steps": optimizer_steps,
                "device": str(device),
                "dataset_stats": asdict(dataset.stats),
                "descriptor_weighting": {
                    "mode": config.descriptor_weight_mode,
                    "quantile": config.descriptor_weight_quantile,
                    "train_split_cap": descriptor_weight_cap,
                },
                "direction_weighting": {
                    "mode": config.direction_weight_mode,
                    "quantile": config.direction_weight_quantile,
                    "train_split_cap": direction_weight_cap,
                },
                "best_validation_loss": (
                    None if not math.isfinite(best_val_loss) else best_val_loss
                ),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    return TrainingArtifacts(
        run_dir=run_dir,
        metrics_json=metrics_json,
        metrics_csv=metrics_csv,
        best_checkpoint=best_checkpoint,
        last_checkpoint=last_checkpoint,
    )
