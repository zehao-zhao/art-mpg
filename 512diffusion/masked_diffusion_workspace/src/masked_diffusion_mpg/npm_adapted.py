"""NPM-adapted seed-direction baseline.

This module intentionally lives beside, rather than inside, ``comparison.py``.
The historical ``simple_mlp`` comparator and all of its checkpoints therefore
retain their original name, architecture, and serialization contract.

The model follows Neural Parametric Mixtures (arXiv:2504.04315) where the MPG
replay contract permits it: eight learned dense 3-D feature grids, spherical
harmonic direction/normal features, a small ReLU decoder, and eight vMF
components with the paper's softmax/exp/logistic output mappings.  It predicts
only the continuous seed direction.  Chain length/type are deliberately owned
by a separately frozen, shared policy so comparisons cannot gain parameters by
quietly learning new discrete heads.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Subset

from .dataset import ReplayTrainingDataset
from .training import (
    compute_vmf_mixture_nll,
    create_run_dir,
    detect_device,
    move_batch_to_device,
    set_global_seed,
    split_dataset_indices,
)


NPM_ADAPTED_MODEL_KIND = "npm_adapted_seed_direction"
NPM_ADAPTED_FORMAT_VERSION = "npm_adapted_seed_direction_v1"
NPM_PAPER_GRID_RESOLUTIONS = (8, 11, 16, 22, 31, 44, 61, 86)


class NPMAdaptedError(RuntimeError):
    """Raised for invalid NPM-adapted configuration or artifacts."""


@dataclass(slots=True, frozen=True)
class NPMAdaptedModelConfig:
    """Architecture and immutable comparison-policy metadata.

    Defaults are the phase-1 paper-aligned profile requested for this project.
    ``position_min`` and ``position_max`` are replaced with train-split bounds
    by :func:`run_npm_adapted_training` and persisted in every checkpoint.
    """

    grid_resolutions: tuple[int, ...] = NPM_PAPER_GRID_RESOLUTIONS
    features_per_level: int = 4
    sh_degree: int = 2
    hidden_dim: int = 64
    hidden_layers: int = 3
    vmf_components: int = 8
    min_kappa: float = 1e-4
    max_kappa: float = 128.0
    position_min: tuple[float, float, float] = (-1.0, -1.0, -1.0)
    position_max: tuple[float, float, float] = (1.0, 1.0, 1.0)
    descriptor_policy_id: str = "geometric_length_trace_inferred_type_v1"

    def __post_init__(self) -> None:
        if not self.grid_resolutions or any(r < 2 for r in self.grid_resolutions):
            raise ValueError("grid_resolutions must contain values >= 2")
        if self.features_per_level <= 0:
            raise ValueError("features_per_level must be positive")
        if self.sh_degree != 2:
            raise ValueError("phase-1 NPM supports degree-2 SH only")
        if self.hidden_dim <= 0 or self.hidden_layers <= 0:
            raise ValueError("decoder dimensions must be positive")
        if self.vmf_components <= 0:
            raise ValueError("vmf_components must be positive")
        if self.min_kappa <= 0.0 or self.max_kappa < self.min_kappa:
            raise ValueError("invalid vMF concentration bounds")
        if len(self.position_min) != 3 or len(self.position_max) != 3:
            raise ValueError("position bounds must be 3-D")
        if any(hi <= lo for lo, hi in zip(self.position_min, self.position_max)):
            raise ValueError("each position_max component must exceed position_min")
        if not self.descriptor_policy_id.strip():
            raise ValueError("descriptor_policy_id must name the shared frozen policy")

    @property
    def grid_feature_dim(self) -> int:
        return len(self.grid_resolutions) * self.features_per_level

    @property
    def decoder_input_dim(self) -> int:
        # spatial embedding + degree-2 SH(direction) + degree-2 SH(normal)
        # + scalar roughness
        return self.grid_feature_dim + 9 + 9 + 1

    @property
    def paper_profile_matches(self) -> bool:
        return (
            self.grid_resolutions == NPM_PAPER_GRID_RESOLUTIONS
            and self.features_per_level == 4
            and self.sh_degree == 2
            and self.hidden_dim == 64
            and self.hidden_layers == 3
            and self.vmf_components == 8
        )


@dataclass(slots=True, frozen=True)
class NPMAdaptedTrainingConfig:
    """Offline training settings for the NPM-adapted direction baseline."""

    replay_paths: tuple[str, ...]
    output_dir: str
    seed: int = 42
    batch_size: int = 262_144
    max_steps: int = 1_000
    learning_rate: float = 0.005
    ema_decay: float = 0.99
    val_fraction: float = 0.25
    max_records: int | None = None
    num_workers: int = 4
    device: str = "auto"
    position_bounds_padding: float = 0.01
    model: NPMAdaptedModelConfig = NPMAdaptedModelConfig()

    def __post_init__(self) -> None:
        if not self.replay_paths:
            raise ValueError("replay_paths is empty")
        if self.batch_size <= 0 or self.max_steps <= 0:
            raise ValueError("batch_size and max_steps must be positive")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if not 0.0 <= self.ema_decay < 1.0:
            raise ValueError("ema_decay must lie in [0, 1)")
        if not 0.0 <= self.val_fraction < 1.0:
            raise ValueError("val_fraction must lie in [0, 1)")
        if self.num_workers < 0:
            raise ValueError("num_workers must be nonnegative")
        if self.position_bounds_padding < 0.0:
            raise ValueError("position_bounds_padding must be nonnegative")


@dataclass(slots=True, frozen=True)
class NPMAdaptedTrainingArtifacts:
    """Files produced by one isolated NPM-adapted training run."""

    run_dir: Path
    checkpoint_path: Path
    summary_json: Path


@dataclass(slots=True, frozen=True)
class NPMAdaptedExportArtifacts:
    """Python inference bundle and an explicit capability manifest."""

    weights_path: Path
    manifest_json: Path


def real_sh_degree2(direction: torch.Tensor) -> torch.Tensor:
    """Evaluate the nine real SH basis functions through degree two."""

    if direction.ndim != 2 or direction.shape[1] != 3:
        raise ValueError("direction must have shape [batch, 3]")
    unit = F.normalize(direction.float(), dim=1, eps=1e-8)
    x, y, z = unit.unbind(dim=1)
    return torch.stack(
        (
            torch.full_like(x, 0.28209479177387814),
            0.4886025119029199 * y,
            0.4886025119029199 * z,
            0.4886025119029199 * x,
            1.0925484305920792 * x * y,
            1.0925484305920792 * y * z,
            0.31539156525252005 * (3.0 * z.square() - 1.0),
            1.0925484305920792 * x * z,
            0.5462742152960396 * (x.square() - y.square()),
        ),
        dim=1,
    )


class MultiResolutionGridEncoding(nn.Module):
    """Dense learned 3-D grids with trilinear interpolation per level."""

    def __init__(self, config: NPMAdaptedModelConfig) -> None:
        super().__init__()
        self.resolutions = tuple(int(value) for value in config.grid_resolutions)
        self.features_per_level = int(config.features_per_level)
        self.grids = nn.ParameterList(
            [
                nn.Parameter(
                    torch.empty(
                        resolution**3,
                        self.features_per_level,
                        dtype=torch.float32,
                    )
                )
                for resolution in self.resolutions
            ]
        )
        self.register_buffer(
            "position_min",
            torch.tensor(config.position_min, dtype=torch.float32),
        )
        self.register_buffer(
            "position_max",
            torch.tensor(config.position_max, dtype=torch.float32),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Use the small uniform initialization common to learned grids."""

        for grid in self.grids:
            nn.init.uniform_(grid, -1e-4, 1e-4)

    def _normalized_position(self, position: torch.Tensor) -> torch.Tensor:
        if position.ndim != 2 or position.shape[1] != 3:
            raise ValueError("position must have shape [batch, 3]")
        extent = (self.position_max - self.position_min).clamp_min(1e-8)
        return ((position.float() - self.position_min) / extent).clamp(0.0, 1.0)

    @staticmethod
    def _interpolate_level(
        normalized: torch.Tensor,
        grid: torch.Tensor,
        resolution: int,
    ) -> torch.Tensor:
        scaled = normalized * float(resolution - 1)
        lower = torch.floor(scaled).to(dtype=torch.long)
        upper = torch.clamp(lower + 1, max=resolution - 1)
        fraction = scaled - lower.to(dtype=scaled.dtype)
        result = torch.zeros(
            normalized.shape[0],
            grid.shape[1],
            device=grid.device,
            dtype=grid.dtype,
        )
        for x_high in (0, 1):
            for y_high in (0, 1):
                for z_high in (0, 1):
                    x_index = upper[:, 0] if x_high else lower[:, 0]
                    y_index = upper[:, 1] if y_high else lower[:, 1]
                    z_index = upper[:, 2] if z_high else lower[:, 2]
                    flat_index = (x_index * resolution + y_index) * resolution + z_index
                    x_weight = fraction[:, 0] if x_high else 1.0 - fraction[:, 0]
                    y_weight = fraction[:, 1] if y_high else 1.0 - fraction[:, 1]
                    z_weight = fraction[:, 2] if z_high else 1.0 - fraction[:, 2]
                    weight = (x_weight * y_weight * z_weight).unsqueeze(1)
                    result = result + weight * grid[flat_index]
        return result

    def forward(self, position: torch.Tensor) -> torch.Tensor:
        normalized = self._normalized_position(position)
        return torch.cat(
            [
                self._interpolate_level(normalized, grid, resolution)
                for grid, resolution in zip(self.grids, self.resolutions)
            ],
            dim=1,
        )


class NPMAdaptedSeedDirection(nn.Module):
    """Paper-shaped NPM decoder for the MPG initial seed direction only."""

    def __init__(self, config: NPMAdaptedModelConfig) -> None:
        super().__init__()
        self.config = config
        self.grid_encoding = MultiResolutionGridEncoding(config)
        decoder: list[nn.Module] = []
        input_dim = config.decoder_input_dim
        for _ in range(config.hidden_layers):
            decoder.extend((nn.Linear(input_dim, config.hidden_dim), nn.ReLU()))
            input_dim = config.hidden_dim
        self.decoder = nn.Sequential(*decoder)
        self.output_layer = nn.Linear(config.hidden_dim, config.vmf_components * 4)

    def forward(
        self,
        position: torch.Tensor,
        outgoing_direction: torch.Tensor,
        normal: torch.Tensor,
        roughness: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Return an eight-component vMF mixture using NPM output mappings."""

        if roughness.ndim == 1:
            roughness = roughness.unsqueeze(1)
        if roughness.ndim != 2 or roughness.shape[1] != 1:
            raise ValueError("roughness must have shape [batch] or [batch, 1]")
        features = torch.cat(
            (
                self.grid_encoding(position),
                real_sh_degree2(outgoing_direction),
                real_sh_degree2(normal),
                roughness.float().clamp(0.0, 1.0),
            ),
            dim=1,
        )
        raw = self.output_layer(self.decoder(features)).view(
            features.shape[0],
            self.config.vmf_components,
            4,
        )
        logits = raw[:, :, 0]
        kappas = torch.exp(
            raw[:, :, 1].clamp(
                min=math.log(self.config.min_kappa),
                max=math.log(self.config.max_kappa),
            )
        )
        theta = math.pi * torch.sigmoid(raw[:, :, 2])
        phi = 2.0 * math.pi * torch.sigmoid(raw[:, :, 3])
        sin_theta = torch.sin(theta)
        means = torch.stack(
            (
                sin_theta * torch.cos(phi),
                sin_theta * torch.sin(phi),
                torch.cos(theta),
            ),
            dim=2,
        )
        weights = torch.softmax(logits, dim=1)
        direction = F.normalize(
            (weights.unsqueeze(2) * means).sum(dim=1),
            dim=1,
            eps=1e-8,
        )
        return {
            "direction": direction,
            "vmf_logits": logits,
            "vmf_weights": weights,
            "vmf_means": means,
            "vmf_kappas": kappas,
        }

    def forward_batch(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Forward the explicit NPM feature keys emitted by the replay dataset."""

        return self(
            batch["npm_position"],
            batch["npm_outgoing_direction"],
            batch["npm_normal"],
            batch["npm_roughness"],
        )


class ExponentialMovingAverage:
    """EMA state used for every exported/inference NPM checkpoint."""

    def __init__(self, model: nn.Module, *, decay: float) -> None:
        if not 0.0 <= decay < 1.0:
            raise ValueError("EMA decay must lie in [0, 1)")
        self.decay = float(decay)
        self.shadow = {
            name: tensor.detach().clone() for name, tensor in model.state_dict().items()
        }

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for name, tensor in model.state_dict().items():
            shadow = self.shadow[name]
            if tensor.is_floating_point():
                shadow.mul_(self.decay).add_(tensor.detach(), alpha=1.0 - self.decay)
            else:
                shadow.copy_(tensor.detach())

    def model_state_dict(self) -> dict[str, torch.Tensor]:
        return {name: tensor.detach().clone() for name, tensor in self.shadow.items()}


def build_npm_adapted_optimizer(
    model: NPMAdaptedSeedDirection,
    *,
    learning_rate: float = 0.005,
) -> Adam:
    """Build the paper optimizer (Adam, default learning rate 0.005)."""

    return Adam(model.parameters(), lr=learning_rate)


def compute_npm_adapted_loss(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    *,
    max_kappa: float,
) -> torch.Tensor:
    """Replay-weighted mixture NLL, the available proxy for NPM's KL loss."""

    per_sample = compute_vmf_mixture_nll(
        outputs,
        batch["direction"],
        vmf_max_kappa=max_kappa,
    )
    weights = batch["sample_weight"].float().clamp_min(0.0)
    return (per_sample * weights).sum() / weights.sum().clamp_min(1e-8)


def fit_position_bounds(
    dataset: ReplayTrainingDataset,
    indices: Iterable[int],
    *,
    padding_fraction: float,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Fit a non-degenerate grid AABB from the training split only."""

    selected = list(indices)
    if not selected:
        raise NPMAdaptedError("cannot fit grid bounds from an empty training split")
    positions = torch.stack([dataset[index]["npm_position"] for index in selected])
    lower = positions.amin(dim=0)
    upper = positions.amax(dim=0)
    span = upper - lower
    fallback = torch.where(span > 1e-6, span, torch.ones_like(span))
    pad = fallback * float(padding_fraction)
    lower = lower - pad
    upper = upper + pad
    upper = torch.where(upper > lower + 1e-6, upper, lower + 1.0)
    return tuple(float(v) for v in lower), tuple(float(v) for v in upper)


def _cpu_state_dict(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    return {name: tensor.detach().cpu() for name, tensor in state_dict.items()}


def train_npm_adapted_model(
    *,
    model: NPMAdaptedSeedDirection,
    train_loader: DataLoader,
    config: NPMAdaptedTrainingConfig,
    device: torch.device,
) -> tuple[ExponentialMovingAverage, tuple[float, ...]]:
    """Train the direction-only NPM for a fixed optimizer-step budget."""

    model.to(device)
    model.train()
    optimizer = build_npm_adapted_optimizer(
        model,
        learning_rate=config.learning_rate,
    )
    ema = ExponentialMovingAverage(model, decay=config.ema_decay)
    iterator = iter(train_loader)
    history: list[float] = []
    for _step in range(1, config.max_steps + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            batch = next(iterator)
        batch = move_batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        outputs = model.forward_batch(batch)
        loss = compute_npm_adapted_loss(
            outputs,
            batch,
            max_kappa=model.config.max_kappa,
        )
        if not bool(torch.isfinite(loss)):
            raise NPMAdaptedError("non-finite NPM-adapted training loss")
        loss.backward()
        optimizer.step()
        ema.update(model)
        history.append(float(loss.detach().item()))
    return ema, tuple(history)


@torch.no_grad()
def evaluate_npm_adapted(
    *,
    model: NPMAdaptedSeedDirection,
    data_loader: DataLoader | None,
    device: torch.device,
) -> float | None:
    if data_loader is None:
        return None
    model.eval()
    numerator = 0.0
    denominator = 0
    for batch in data_loader:
        batch = move_batch_to_device(batch, device)
        loss = compute_npm_adapted_loss(
            model.forward_batch(batch),
            batch,
            max_kappa=model.config.max_kappa,
        )
        batch_size = int(batch["direction"].shape[0])
        numerator += float(loss.item()) * batch_size
        denominator += batch_size
    return numerator / denominator if denominator else None


def _model_config_from_mapping(payload: dict[str, object]) -> NPMAdaptedModelConfig:
    normalized = dict(payload)
    for key in ("grid_resolutions", "position_min", "position_max"):
        if key in normalized:
            normalized[key] = tuple(normalized[key])
    return NPMAdaptedModelConfig(**normalized)


def load_npm_adapted_training_config(path: Path) -> NPMAdaptedTrainingConfig:
    """Load a JSON training config without affecting legacy comparison configs."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    model_payload = dict(payload.pop("model", {}))
    model = _model_config_from_mapping(model_payload)
    payload["replay_paths"] = tuple(payload["replay_paths"])
    return NPMAdaptedTrainingConfig(model=model, **payload)


def run_npm_adapted_training(
    *,
    workspace_root: Path,
    config: NPMAdaptedTrainingConfig,
) -> NPMAdaptedTrainingArtifacts:
    """Train in an isolated run directory and persist online plus EMA weights."""

    set_global_seed(config.seed)
    replay_paths = tuple(
        (workspace_root / value).resolve() for value in config.replay_paths
    )
    dataset = ReplayTrainingDataset(replay_paths, max_records=config.max_records)
    train_indices, val_indices = split_dataset_indices(
        len(dataset),
        val_fraction=config.val_fraction,
        seed=config.seed,
        group_keys=dataset.group_keys,
    )
    position_min, position_max = fit_position_bounds(
        dataset,
        train_indices,
        padding_fraction=config.position_bounds_padding,
    )
    model_config = replace(
        config.model,
        position_min=position_min,
        position_max=position_max,
    )
    effective_config = replace(config, model=model_config)
    device = detect_device(config.device)
    train_loader = DataLoader(
        Subset(dataset, train_indices),
        batch_size=min(config.batch_size, len(train_indices)),
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )
    val_loader = (
        DataLoader(
            Subset(dataset, val_indices),
            batch_size=min(config.batch_size, len(val_indices)),
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=device.type == "cuda",
            drop_last=False,
        )
        if val_indices
        else None
    )
    model = NPMAdaptedSeedDirection(model_config)
    ema, history = train_npm_adapted_model(
        model=model,
        train_loader=train_loader,
        config=effective_config,
        device=device,
    )
    online_state = _cpu_state_dict(model.state_dict())
    ema_state = _cpu_state_dict(ema.model_state_dict())
    model.load_state_dict(ema_state)
    model.to(device)
    validation_loss = evaluate_npm_adapted(
        model=model,
        data_loader=val_loader,
        device=device,
    )

    output_root = (workspace_root / config.output_dir).resolve()
    run_dir = create_run_dir(output_root)
    checkpoint_path = run_dir / "npm_adapted_seed_direction.pt"
    proxy_count = sum(
        int(dataset[index]["npm_direction_is_proxy"].item())
        for index in range(len(dataset))
    )
    default_roughness_count = sum(
        int(dataset[index]["npm_roughness_is_default"].item())
        for index in range(len(dataset))
    )
    grid_parameter_count = sum(
        parameter.numel() for parameter in model.grid_encoding.parameters()
    )
    total_parameter_count = sum(parameter.numel() for parameter in model.parameters())
    checkpoint_payload = {
        "format_version": NPM_ADAPTED_FORMAT_VERSION,
        "model_kind": NPM_ADAPTED_MODEL_KIND,
        "model_config": asdict(model_config),
        "training_config": asdict(effective_config),
        "model_state_dict": online_state,
        "ema_model_state_dict": ema_state,
        "inference_weights": "ema_model_state_dict",
        "descriptor_policy": {
            "id": model_config.descriptor_policy_id,
            "frozen": True,
            "predicted_by_this_model": False,
            "length_policy": "existing_mpg_geometric",
            "type_policy": "infer_from_realized_specular_trace",
        },
        "dataset_provenance": {
            "records": len(dataset),
            "train_records": len(train_indices),
            "validation_records": len(val_indices),
            "proxy_direction_records": proxy_count,
            "default_roughness_records": default_roughness_count,
        },
        "architecture_accounting": {
            "grid_scalar_parameters": grid_parameter_count,
            "total_scalar_parameters": total_parameter_count,
            "fp32_parameter_bytes": total_parameter_count * 4,
            "paper_reports_about_2m_and_under_10mb": True,
            "paper_source_states_feature_dtype": False,
            "reconciliation": (
                "the stated dense F=4, D8=86 grid alone exceeds 2M scalar "
                "parameters; under-10MB is compatible with half precision"
            ),
        },
        "training_loss": {
            "first": history[0],
            "last": history[-1],
            "validation_ema": validation_loss,
        },
    }
    torch.save(checkpoint_payload, checkpoint_path)
    summary_json = run_dir / "npm_adapted_training_summary.json"
    summary_json.write_text(
        json.dumps(
            {
                key: value
                for key, value in checkpoint_payload.items()
                if key
                not in {
                    "model_state_dict",
                    "ema_model_state_dict",
                }
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return NPMAdaptedTrainingArtifacts(
        run_dir=run_dir,
        checkpoint_path=checkpoint_path,
        summary_json=summary_json,
    )


def load_npm_adapted_checkpoint(
    checkpoint_path: Path,
    *,
    use_ema: bool = True,
    map_location: str | torch.device = "cpu",
) -> tuple[NPMAdaptedSeedDirection, dict[str, object]]:
    """Load a phase-1 checkpoint, selecting EMA weights by default."""

    if not checkpoint_path.exists():
        raise NPMAdaptedError(f"missing checkpoint: {checkpoint_path}")
    payload = torch.load(checkpoint_path, map_location=map_location)
    if payload.get("model_kind") != NPM_ADAPTED_MODEL_KIND:
        raise NPMAdaptedError("checkpoint is not an npm_adapted_seed_direction model")
    config = _model_config_from_mapping(dict(payload["model_config"]))
    model = NPMAdaptedSeedDirection(config)
    state_key = "ema_model_state_dict" if use_ema else "model_state_dict"
    if state_key not in payload:
        raise NPMAdaptedError(f"checkpoint is missing {state_key}")
    model.load_state_dict(payload[state_key])
    model.eval()
    return model, payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def export_npm_adapted_python_bundle(
    *,
    checkpoint_path: Path,
    output_dir: Path,
) -> NPMAdaptedExportArtifacts:
    """Export EMA weights for Python inference, explicitly excluding C++ parity."""

    model, checkpoint = load_npm_adapted_checkpoint(checkpoint_path, use_ema=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    weights_path = output_dir / "npm_adapted_seed_direction_ema.pt"
    torch.save(
        {
            "format_version": NPM_ADAPTED_FORMAT_VERSION,
            "model_kind": NPM_ADAPTED_MODEL_KIND,
            "model_config": asdict(model.config),
            "model_state_dict": _cpu_state_dict(model.state_dict()),
            "descriptor_policy": checkpoint["descriptor_policy"],
        },
        weights_path,
    )
    manifest_json = output_dir / "npm_adapted_seed_direction_manifest.json"
    manifest_json.write_text(
        json.dumps(
            {
                "format_version": NPM_ADAPTED_FORMAT_VERSION,
                "model_kind": NPM_ADAPTED_MODEL_KIND,
                "source_checkpoint": str(checkpoint_path.resolve()),
                "source_checkpoint_sha256": _sha256(checkpoint_path),
                "weights_path": str(weights_path.resolve()),
                "weights_sha256": _sha256(weights_path),
                "inference_weights": "ema",
                "descriptor_policy": checkpoint["descriptor_policy"],
                "parameter_count": sum(
                    parameter.numel() for parameter in model.parameters()
                ),
                "this_bundle_format": "python_state_dict",
                "native_cpp_runtime_supported_by_this_bundle": False,
                "native_cpp_text_export_available": True,
                "native_cpp_export_model_kind": NPM_ADAPTED_MODEL_KIND,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return NPMAdaptedExportArtifacts(
        weights_path=weights_path,
        manifest_json=manifest_json,
    )
