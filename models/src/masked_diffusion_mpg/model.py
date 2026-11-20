from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import torch
from torch import nn

from .dataset import (
    MASK_CHAIN_LENGTH_TOKEN,
    MASK_CHAIN_TYPE_TOKEN,
    MAX_CHAIN_LENGTH,
    MAX_CHAIN_TYPE_BITS,
)


DISCRETE_DIFFUSION_LEGACY = "legacy"
DISCRETE_DIFFUSION_ORDERED_ABSORBING = "ordered_absorbing"
DIRECTION_DISTRIBUTION_SPHERICAL_D3PM_VMF = "spherical_d3pm_vmf"
SPHERICAL_D3PM_GEOMETRY_FIXED_ANCHOR = "fixed_anchor"
SPHERICAL_D3PM_GEOMETRY_LEARNED_EMISSION = "learned_emission"
SUPPORTED_SPHERICAL_D3PM_GEOMETRIES = frozenset(
    {
        SPHERICAL_D3PM_GEOMETRY_FIXED_ANCHOR,
        SPHERICAL_D3PM_GEOMETRY_LEARNED_EMISSION,
    }
)
SUPPORTED_DIRECTION_DISTRIBUTIONS = frozenset(
    {
        "vector",
        "vmf_mixture",
        DIRECTION_DISTRIBUTION_SPHERICAL_D3PM_VMF,
    }
)
SUPPORTED_DISCRETE_DIFFUSION_MODES = frozenset(
    {
        DISCRETE_DIFFUSION_LEGACY,
        DISCRETE_DIFFUSION_ORDERED_ABSORBING,
    }
)
SUPPORTED_BACKBONE_TYPES = frozenset({"mlp", "transformer", "dit"})


@dataclass(slots=True, frozen=True)
class MaskedDiffusionModelConfig:
    """Describe the masked diffusion denoiser architecture."""

    condition_dim: int = 22
    hidden_dim: int = 256
    depth: int = 4
    dropout: float = 0.1
    time_embedding_dim: int = 64
    token_embedding_dim: int = 32
    backbone_type: str = "mlp"
    transformer_heads: int = 4
    transformer_feedforward_dim: int | None = None
    direction_distribution: str = "vector"
    vmf_components: int = 4
    vmf_max_kappa: float = 32.0
    direction_states: int = 32
    direction_diffusion_steps: int = 4
    direction_grid_kappa: float = 32.0
    direction_state_geometry: str = SPHERICAL_D3PM_GEOMETRY_FIXED_ANCHOR
    discrete_diffusion_mode: str = DISCRETE_DIFFUSION_LEGACY


@dataclass(slots=True, frozen=True)
class OrderedDescriptorSample:
    """One exact sample from the ordered absorbing descriptor proposal.

    ``chain_type_bits`` contains only the valid prefix for ``chain_length``.
    ``probability`` is the normalized descriptor mass, not a directional PDF.
    """

    chain_length: int
    chain_type_bits: tuple[int, ...]
    log_probability: float
    probability: float


def build_sinusoidal_time_embedding(
    timesteps: torch.Tensor,
    embedding_dim: int,
) -> torch.Tensor:
    """Build standard sinusoidal timestep embeddings."""

    if embedding_dim <= 0:
        raise ValueError("embedding_dim must be positive")
    half_dim = embedding_dim // 2
    if half_dim == 0:
        return timesteps.unsqueeze(1)

    scale = math.log(10000.0) / max(half_dim - 1, 1)
    exponents = torch.exp(
        torch.arange(half_dim, device=timesteps.device, dtype=timesteps.dtype)
        * (-scale)
    )
    angles = timesteps.unsqueeze(1) * exponents.unsqueeze(0)
    embedding = torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)
    if embedding_dim % 2 == 1:
        embedding = torch.cat(
            [embedding, torch.zeros_like(embedding[:, :1])],
            dim=1,
        )
    return embedding


def spherical_fibonacci_anchors(
    state_count: int,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return deterministic approximately equal-area anchors on ``S^2``.

    Anchors are materialized into spherical-D3PM checkpoints as a persistent
    buffer.  A future native exporter must serialize this buffer verbatim
    instead of regenerating it with platform-specific trigonometry.
    """

    if state_count < 2:
        raise ValueError("direction_states must be at least 2")
    indices = torch.arange(state_count, device=device, dtype=torch.float64)
    z = 1.0 - 2.0 * (indices + 0.5) / float(state_count)
    golden_angle = math.pi * (3.0 - math.sqrt(5.0))
    phi = indices * golden_angle
    radius = torch.sqrt((1.0 - z.square()).clamp_min(0.0))
    anchors = torch.stack(
        [radius * torch.cos(phi), radius * torch.sin(phi), z],
        dim=1,
    )
    return anchors.to(dtype=dtype)


def spherical_reverse_matrix(
    state_logits: torch.Tensor,
    locality: torch.Tensor,
    anchors: torch.Tensor,
) -> torch.Tensor:
    """Construct normalized reverse kernels for the finite spherical D3PM.

    Rows index the current state ``z_t`` and columns the destination state
    ``z_{t-1}``.  The learned destination field is combined with a positive
    spherical locality term, preserving exact row normalization.
    """

    if state_logits.ndim != 2:
        raise ValueError("state_logits must have shape [batch, states]")
    batch_size, state_count = state_logits.shape
    if anchors.ndim == 2:
        if anchors.shape != (state_count, 3):
            raise ValueError("anchors must have shape [states, 3]")
        anchors = anchors.unsqueeze(0).expand(batch_size, -1, -1)
    elif anchors.shape != (batch_size, state_count, 3):
        raise ValueError(
            "batched anchors must have shape [batch, states, 3]"
        )
    if locality.ndim == 2 and locality.shape[1] == 1:
        locality = locality.squeeze(1)
    if locality.shape != (batch_size,):
        raise ValueError("locality must have shape [batch] or [batch, 1]")

    normalized_anchors = torch.nn.functional.normalize(
        anchors.to(device=state_logits.device, dtype=state_logits.dtype),
        dim=2,
        eps=1e-12,
    )
    similarity = torch.bmm(
        normalized_anchors,
        normalized_anchors.transpose(1, 2),
    )
    transition_logits = (
        state_logits.unsqueeze(1)
        + locality.to(dtype=state_logits.dtype).reshape(batch_size, 1, 1)
        * similarity
    )
    return torch.softmax(transition_logits, dim=2)


def spherical_terminal_distribution(
    reverse_matrices: Sequence[torch.Tensor] | torch.Tensor,
) -> torch.Tensor:
    """Exactly marginalize a finite reverse chain from a uniform terminal law.

    Matrices must be ordered in sampling order, ``P_T, ..., P_1``.  This dynamic
    program is the tractability boundary: no sampled latent trajectory is
    cached or treated as an additional physical descriptor stratum.
    """

    if isinstance(reverse_matrices, torch.Tensor):
        if reverse_matrices.ndim != 4:
            raise ValueError(
                "reverse_matrices tensor must have shape [batch, steps, states, states]"
            )
        matrices = tuple(reverse_matrices.unbind(dim=1))
    else:
        matrices = tuple(reverse_matrices)
    if not matrices:
        raise ValueError("at least one reverse matrix is required")

    first = matrices[0]
    if first.ndim != 3 or first.shape[1] != first.shape[2]:
        raise ValueError("each reverse matrix must have shape [batch, states, states]")
    batch_size, state_count, _ = first.shape
    probabilities = torch.full(
        (batch_size, state_count),
        1.0 / float(state_count),
        device=first.device,
        dtype=first.dtype,
    )
    for matrix in matrices:
        if matrix.shape != first.shape:
            raise ValueError("all reverse matrices must have identical shapes")
        probabilities = torch.bmm(probabilities.unsqueeze(1), matrix).squeeze(1)
    return probabilities / probabilities.sum(dim=1, keepdim=True).clamp_min(1e-30)


class ResidualMLPBlock(nn.Module):
    """Simple residual MLP block."""

    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply one residual update."""

        return x + self.block(x)


def _modulate_tokens(
    tokens: torch.Tensor,
    shift: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    """Apply feature-wise adaptive LayerNorm modulation to token sequences."""

    return tokens * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class CompactDiTBlock(nn.Module):
    """A compact DiT block with condition-driven AdaLN-Zero residuals.

    This follows the defining DiT structure: the scene/timestep embedding does
    not participate as an ordinary sequence token.  Instead it produces the
    shift, scale, and residual gate for both attention and feed-forward paths.
    Zero-initialized gates make every block an identity map at initialization.
    """

    def __init__(
        self,
        *,
        hidden_dim: int,
        num_heads: int,
        feedforward_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.attention = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, feedforward_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feedforward_dim, hidden_dim),
            nn.Dropout(dropout),
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 6 * hidden_dim),
        )
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def forward(
        self,
        tokens: torch.Tensor,
        conditioning: torch.Tensor,
    ) -> torch.Tensor:
        """Apply adaptive attention and MLP updates."""

        (
            shift_attention,
            scale_attention,
            gate_attention,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        ) = self.adaLN_modulation(conditioning).chunk(6, dim=1)
        attention_input = _modulate_tokens(
            self.norm1(tokens),
            shift_attention,
            scale_attention,
        )
        attention_output = self.attention(
            attention_input,
            attention_input,
            attention_input,
            need_weights=False,
        )[0]
        tokens = tokens + gate_attention.unsqueeze(1) * attention_output

        mlp_input = _modulate_tokens(
            self.norm2(tokens),
            shift_mlp,
            scale_mlp,
        )
        tokens = tokens + gate_mlp.unsqueeze(1) * self.mlp(mlp_input)
        return tokens


class AdaptiveFinalLayerNorm(nn.Module):
    """DiT-style adaptive final normalization conditioned on scene and time."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 2 * hidden_dim),
        )
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def forward(
        self,
        tokens: torch.Tensor,
        conditioning: torch.Tensor,
    ) -> torch.Tensor:
        """Normalize and modulate every token with one global condition."""

        shift, scale = self.adaLN_modulation(conditioning).chunk(2, dim=1)
        return _modulate_tokens(self.norm(tokens), shift, scale)


class MaskedSeedDiffusionModel(nn.Module):
    """Minimal masked diffusion denoiser for MPG replay records."""

    def __init__(self, config: MaskedDiffusionModelConfig) -> None:
        super().__init__()
        if config.backbone_type not in SUPPORTED_BACKBONE_TYPES:
            raise ValueError(f"unsupported backbone_type: {config.backbone_type}")
        if config.discrete_diffusion_mode not in SUPPORTED_DISCRETE_DIFFUSION_MODES:
            raise ValueError(
                "unsupported discrete_diffusion_mode: "
                f"{config.discrete_diffusion_mode}"
            )
        if config.direction_distribution not in SUPPORTED_DIRECTION_DISTRIBUTIONS:
            raise ValueError(
                f"unsupported direction_distribution: {config.direction_distribution}"
            )
        if config.vmf_components <= 0:
            raise ValueError("vmf_components must be positive")
        if config.vmf_max_kappa <= 0.0:
            raise ValueError("vmf_max_kappa must be positive")
        if config.direction_states < 2:
            raise ValueError("direction_states must be at least 2")
        if config.direction_diffusion_steps <= 0:
            raise ValueError("direction_diffusion_steps must be positive")
        if config.direction_grid_kappa <= 0.0:
            raise ValueError("direction_grid_kappa must be positive")
        if config.direction_state_geometry not in SUPPORTED_SPHERICAL_D3PM_GEOMETRIES:
            raise ValueError(
                "unsupported direction_state_geometry: "
                f"{config.direction_state_geometry}"
            )
        if (
            config.backbone_type in {"transformer", "dit"}
            and config.hidden_dim % config.transformer_heads != 0
        ):
            raise ValueError("hidden_dim must be divisible by transformer_heads")

        self.config = config

        self.chain_length_embedding = nn.Embedding(
            num_embeddings=MASK_CHAIN_LENGTH_TOKEN + 1,
            embedding_dim=config.token_embedding_dim,
        )
        self.chain_type_embedding = nn.Embedding(
            num_embeddings=MASK_CHAIN_TYPE_TOKEN + 1,
            embedding_dim=config.token_embedding_dim,
        )

        if config.backbone_type == "mlp":
            input_dim = (
                config.condition_dim
                + 3
                + config.time_embedding_dim
                + config.token_embedding_dim
                + config.token_embedding_dim * MAX_CHAIN_TYPE_BITS
            )
            self.input_projection = nn.Sequential(
                nn.Linear(input_dim, config.hidden_dim),
                nn.SiLU(),
                nn.Dropout(config.dropout),
            )
            self.blocks = nn.ModuleList(
                ResidualMLPBlock(config.hidden_dim, config.dropout)
                for _ in range(config.depth)
            )
        elif config.backbone_type == "transformer":
            feedforward_dim = (
                config.transformer_feedforward_dim
                if config.transformer_feedforward_dim is not None
                else config.hidden_dim * 4
            )
            token_count = 3 + MAX_CHAIN_TYPE_BITS
            self.condition_projection = nn.Sequential(
                nn.Linear(config.condition_dim, config.hidden_dim),
                nn.SiLU(),
                nn.Dropout(config.dropout),
            )
            self.direction_time_projection = nn.Sequential(
                nn.Linear(3 + config.time_embedding_dim, config.hidden_dim),
                nn.SiLU(),
                nn.Dropout(config.dropout),
            )
            self.chain_length_projection = nn.Linear(
                config.token_embedding_dim,
                config.hidden_dim,
            )
            self.chain_type_projection = nn.Linear(
                config.token_embedding_dim,
                config.hidden_dim,
            )
            self.position_embedding = nn.Parameter(
                torch.zeros(1, token_count, config.hidden_dim)
            )
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=config.hidden_dim,
                nhead=config.transformer_heads,
                dim_feedforward=feedforward_dim,
                dropout=config.dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.transformer = nn.TransformerEncoder(
                encoder_layer,
                num_layers=config.depth,
            )
            self.chain_type_token_head = nn.Linear(config.hidden_dim, 1)

        else:
            feedforward_dim = (
                config.transformer_feedforward_dim
                if config.transformer_feedforward_dim is not None
                else config.hidden_dim * 4
            )
            token_count = 2 + MAX_CHAIN_TYPE_BITS
            self.dit_condition_projection = nn.Linear(
                config.condition_dim,
                config.hidden_dim,
            )
            self.dit_time_projection = nn.Sequential(
                nn.Linear(config.time_embedding_dim, config.hidden_dim),
                nn.SiLU(),
                nn.Linear(config.hidden_dim, config.hidden_dim),
            )
            self.dit_direction_projection = nn.Linear(3, config.hidden_dim)
            self.chain_length_projection = nn.Linear(
                config.token_embedding_dim,
                config.hidden_dim,
            )
            self.chain_type_projection = nn.Linear(
                config.token_embedding_dim,
                config.hidden_dim,
            )
            self.position_embedding = nn.Parameter(
                torch.zeros(1, token_count, config.hidden_dim)
            )
            nn.init.normal_(self.position_embedding, std=0.02)
            self.dit_blocks = nn.ModuleList(
                CompactDiTBlock(
                    hidden_dim=config.hidden_dim,
                    num_heads=config.transformer_heads,
                    feedforward_dim=feedforward_dim,
                    dropout=config.dropout,
                )
                for _ in range(config.depth)
            )
            self.chain_type_token_head = nn.Linear(config.hidden_dim, 1)

        self.final_norm: nn.Module
        if config.backbone_type == "dit":
            self.final_norm = AdaptiveFinalLayerNorm(config.hidden_dim)
        else:
            self.final_norm = nn.LayerNorm(config.hidden_dim)
        self.chain_length_head = nn.Linear(config.hidden_dim, MAX_CHAIN_LENGTH)
        if config.backbone_type == "mlp":
            self.chain_type_head = nn.Linear(config.hidden_dim, MAX_CHAIN_TYPE_BITS)
        if config.direction_distribution == "vmf_mixture":
            self.vmf_logits_head = nn.Linear(config.hidden_dim, config.vmf_components)
            self.vmf_mean_head = nn.Linear(config.hidden_dim, config.vmf_components * 3)
            self.vmf_kappa_head = nn.Linear(config.hidden_dim, config.vmf_components)
        elif (
            config.direction_distribution
            == DIRECTION_DISTRIBUTION_SPHERICAL_D3PM_VMF
        ):
            self.direction_state_head = nn.Linear(
                config.hidden_dim,
                config.direction_states,
            )
            self.direction_locality_head = nn.Linear(config.hidden_dim, 1)
            self.direction_emission_mean_head = nn.Linear(
                config.hidden_dim,
                config.direction_states * 3,
            )
            self.register_buffer(
                "direction_anchors",
                spherical_fibonacci_anchors(config.direction_states),
                persistent=True,
            )
            # Start from the correctness-first fixed-grid proposal while letting
            # the extra t=0 emission pass learn continuous, condition-dependent
            # component locations.  Zero weights also make initialization fully
            # deterministic across descriptors.
            nn.init.zeros_(self.direction_emission_mean_head.weight)
            with torch.no_grad():
                self.direction_emission_mean_head.bias.copy_(
                    self.direction_anchors.flatten()
                )
        else:
            self.direction_head = nn.Linear(config.hidden_dim, 3)

    def _encode_mlp(
        self,
        *,
        condition: torch.Tensor,
        noisy_direction: torch.Tensor,
        masked_chain_length: torch.Tensor,
        masked_chain_type_bits: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode legacy flat MLP inputs."""

        time_embedding = build_sinusoidal_time_embedding(
            timesteps,
            self.config.time_embedding_dim,
        )
        chain_length_embedding = self.chain_length_embedding(masked_chain_length)
        chain_type_embedding = self.chain_type_embedding(
            masked_chain_type_bits
        ).flatten(start_dim=1)

        hidden = torch.cat(
            [
                condition,
                noisy_direction,
                time_embedding,
                chain_length_embedding,
                chain_type_embedding,
            ],
            dim=1,
        )
        hidden = self.input_projection(hidden)
        for block in self.blocks:
            hidden = block(hidden)
        hidden = self.final_norm(hidden)
        return hidden, hidden, hidden

    def _encode_transformer(
        self,
        *,
        condition: torch.Tensor,
        noisy_direction: torch.Tensor,
        masked_chain_length: torch.Tensor,
        masked_chain_type_bits: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode PDF-style tokenized Transformer inputs."""

        time_embedding = build_sinusoidal_time_embedding(
            timesteps,
            self.config.time_embedding_dim,
        )
        condition_token = self.condition_projection(condition).unsqueeze(1)
        direction_token = self.direction_time_projection(
            torch.cat([noisy_direction, time_embedding], dim=1)
        ).unsqueeze(1)
        chain_length_token = self.chain_length_projection(
            self.chain_length_embedding(masked_chain_length)
        ).unsqueeze(1)
        chain_type_tokens = self.chain_type_projection(
            self.chain_type_embedding(masked_chain_type_bits)
        )
        tokens = torch.cat(
            [
                condition_token,
                direction_token,
                chain_length_token,
                chain_type_tokens,
            ],
            dim=1,
        )
        tokens = tokens + self.position_embedding[:, : tokens.shape[1], :]
        tokens = self.transformer(tokens)
        tokens = self.final_norm(tokens)
        direction_hidden = tokens[:, 1, :]
        chain_length_hidden = tokens[:, 2, :]
        chain_type_hidden = tokens[:, 3:, :]
        return chain_length_hidden, chain_type_hidden, direction_hidden

    def _encode_dit(
        self,
        *,
        condition: torch.Tensor,
        noisy_direction: torch.Tensor,
        masked_chain_length: torch.Tensor,
        masked_chain_type_bits: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode ordered states with scene/time-conditioned AdaLN blocks."""

        time_embedding = build_sinusoidal_time_embedding(
            timesteps,
            self.config.time_embedding_dim,
        )
        conditioning = self.dit_condition_projection(condition)
        conditioning = conditioning + self.dit_time_projection(time_embedding)

        direction_token = self.dit_direction_projection(noisy_direction).unsqueeze(1)
        chain_length_token = self.chain_length_projection(
            self.chain_length_embedding(masked_chain_length)
        ).unsqueeze(1)
        chain_type_tokens = self.chain_type_projection(
            self.chain_type_embedding(masked_chain_type_bits)
        )
        tokens = torch.cat(
            [direction_token, chain_length_token, chain_type_tokens],
            dim=1,
        )
        tokens = tokens + self.position_embedding[:, : tokens.shape[1], :]
        for block in self.dit_blocks:
            tokens = block(tokens, conditioning)
        tokens = self.final_norm(tokens, conditioning)
        direction_hidden = tokens[:, 0, :]
        chain_length_hidden = tokens[:, 1, :]
        chain_type_hidden = tokens[:, 2:, :]
        return chain_length_hidden, chain_type_hidden, direction_hidden

    def _direction_outputs(self, hidden: torch.Tensor) -> dict[str, torch.Tensor]:
        """Output direction vectors or vMF mixture parameters."""

        if self.config.direction_distribution == "vmf_mixture":
            batch_size = hidden.shape[0]
            logits = self.vmf_logits_head(hidden)
            means = self.vmf_mean_head(hidden).view(
                batch_size,
                self.config.vmf_components,
                3,
            )
            means = torch.nn.functional.normalize(means, dim=2, eps=1e-6)
            kappas = torch.nn.functional.softplus(self.vmf_kappa_head(hidden))
            kappas = (kappas + 1e-4).clamp(max=self.config.vmf_max_kappa)
            weights = torch.softmax(logits, dim=1)
            direction = torch.nn.functional.normalize(
                (weights.unsqueeze(2) * means).sum(dim=1),
                dim=1,
                eps=1e-6,
            )
            return {
                "direction": direction,
                "vmf_logits": logits,
                "vmf_means": means,
                "vmf_kappas": kappas,
            }

        if (
            self.config.direction_distribution
            == DIRECTION_DISTRIBUTION_SPHERICAL_D3PM_VMF
        ):
            state_logits = self.direction_state_head(hidden)
            locality = torch.nn.functional.softplus(
                self.direction_locality_head(hidden)
            ).squeeze(1)
            locality = locality.clamp(max=64.0)
            one_step_weights = torch.softmax(state_logits, dim=1)
            emission_means = self.direction_emission_mean_head(hidden).view(
                hidden.shape[0],
                self.config.direction_states,
                3,
            )
            emission_means = torch.nn.functional.normalize(
                emission_means,
                dim=2,
                eps=1e-6,
            )
            direction = torch.nn.functional.normalize(
                (one_step_weights.unsqueeze(2) * emission_means).sum(dim=1),
                dim=1,
                eps=1e-6,
            )
            return {
                "direction": direction,
                "direction_state_logits": state_logits,
                "direction_locality": locality,
                "direction_emission_means": emission_means,
            }

        return {"direction": self.direction_head(hidden)}

    def forward(
        self,
        *,
        condition: torch.Tensor,
        noisy_direction: torch.Tensor,
        masked_chain_length: torch.Tensor,
        masked_chain_type_bits: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Run one forward denoising pass."""

        if self.config.backbone_type in {"transformer", "dit"}:
            (
                chain_length_hidden,
                chain_type_hidden,
                direction_hidden,
            ) = (
                self._encode_transformer(
                    condition=condition,
                    noisy_direction=noisy_direction,
                    masked_chain_length=masked_chain_length,
                    masked_chain_type_bits=masked_chain_type_bits,
                    timesteps=timesteps,
                )
                if self.config.backbone_type == "transformer"
                else self._encode_dit(
                    condition=condition,
                    noisy_direction=noisy_direction,
                    masked_chain_length=masked_chain_length,
                    masked_chain_type_bits=masked_chain_type_bits,
                    timesteps=timesteps,
                )
            )
            chain_type_logits = self.chain_type_token_head(chain_type_hidden).squeeze(2)
        else:
            (
                chain_length_hidden,
                chain_type_hidden,
                direction_hidden,
            ) = self._encode_mlp(
                condition=condition,
                noisy_direction=noisy_direction,
                masked_chain_length=masked_chain_length,
                masked_chain_type_bits=masked_chain_type_bits,
                timesteps=timesteps,
            )
            chain_type_logits = self.chain_type_head(chain_type_hidden)

        outputs = {
            "chain_length_logits": self.chain_length_head(chain_length_hidden),
            "chain_type_logits": chain_type_logits,
        }
        outputs.update(self._direction_outputs(direction_hidden))
        return outputs


def _validate_ordered_descriptor_batch(
    condition: torch.Tensor,
    chain_length: torch.Tensor,
    chain_type_bits: torch.Tensor,
) -> None:
    """Validate tensors shared by ordered absorbing state constructors."""

    if condition.ndim != 2:
        raise ValueError("condition must have shape [batch, condition_dim]")
    batch_size = int(condition.shape[0])
    if chain_length.shape != (batch_size,):
        raise ValueError("chain_length must have shape [batch]")
    if chain_type_bits.shape != (batch_size, MAX_CHAIN_TYPE_BITS):
        raise ValueError(
            "chain_type_bits must have shape "
            f"[batch, {MAX_CHAIN_TYPE_BITS}]"
        )
    if bool(
        ((chain_length < 1) | (chain_length > MAX_CHAIN_TYPE_BITS)).any().item()
    ):
        raise ValueError(
            f"chain_length must be in [1, {MAX_CHAIN_TYPE_BITS}]"
        )


def build_ordered_length_inputs(
    condition: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Build the leakage-free all-mask state used to predict chain length."""

    if condition.ndim != 2:
        raise ValueError("condition must have shape [batch, condition_dim]")
    batch_size = int(condition.shape[0])
    device = condition.device
    return {
        "condition": condition,
        "noisy_direction": torch.zeros(
            batch_size,
            3,
            device=device,
            dtype=condition.dtype,
        ),
        "masked_chain_length": torch.full(
            (batch_size,),
            fill_value=MASK_CHAIN_LENGTH_TOKEN,
            device=device,
            dtype=torch.long,
        ),
        "masked_chain_type_bits": torch.full(
            (batch_size, MAX_CHAIN_TYPE_BITS),
            fill_value=MASK_CHAIN_TYPE_TOKEN,
            device=device,
            dtype=torch.long,
        ),
        "timesteps": torch.ones(
            batch_size,
            device=device,
            dtype=condition.dtype,
        ),
    }


def build_ordered_type_inputs(
    *,
    condition: torch.Tensor,
    chain_length: torch.Tensor,
    chain_type_bits: torch.Tensor,
    prefix_length: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Build left-to-right absorbing states for the next type transition.

    For row ``b``, positions below ``prefix_length[b]`` are visible, remaining
    valid positions are MASK, and positions outside the sampled chain are the
    canonical zero padding used by :mod:`dataset`.  The caller supervises only
    the logit at ``prefix_length[b]``.
    """

    _validate_ordered_descriptor_batch(condition, chain_length, chain_type_bits)
    batch_size = int(condition.shape[0])
    if prefix_length.shape != (batch_size,):
        raise ValueError("prefix_length must have shape [batch]")
    if bool(((prefix_length < 0) | (prefix_length >= chain_length)).any().item()):
        raise ValueError("prefix_length must identify a valid next type bit")

    device = condition.device
    positions = torch.arange(MAX_CHAIN_TYPE_BITS, device=device).unsqueeze(0)
    active = positions < chain_length.unsqueeze(1)
    visible = positions < prefix_length.unsqueeze(1)
    state_bits = torch.zeros_like(chain_type_bits, dtype=torch.long, device=device)
    state_bits[active] = MASK_CHAIN_TYPE_TOKEN
    reveal = active & visible
    state_bits[reveal] = chain_type_bits.to(device=device, dtype=torch.long)[reveal]

    remaining_fraction = (
        (chain_length - prefix_length).to(dtype=condition.dtype)
        / chain_length.to(dtype=condition.dtype)
    )
    return {
        "condition": condition,
        "noisy_direction": torch.zeros(
            batch_size,
            3,
            device=device,
            dtype=condition.dtype,
        ),
        "masked_chain_length": chain_length.to(device=device, dtype=torch.long),
        "masked_chain_type_bits": state_bits,
        "timesteps": remaining_fraction,
    }


def build_ordered_direction_inputs(
    *,
    condition: torch.Tensor,
    chain_length: torch.Tensor,
    chain_type_bits: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Build the clean descriptor state used by the conditional direction head."""

    _validate_ordered_descriptor_batch(condition, chain_length, chain_type_bits)
    batch_size = int(condition.shape[0])
    device = condition.device
    positions = torch.arange(MAX_CHAIN_TYPE_BITS, device=device).unsqueeze(0)
    active = positions < chain_length.unsqueeze(1)
    state_bits = torch.zeros_like(chain_type_bits, dtype=torch.long, device=device)
    source_bits = chain_type_bits.to(device=device, dtype=torch.long)
    state_bits[active] = source_bits[active]
    return {
        "condition": condition,
        "noisy_direction": torch.zeros(
            batch_size,
            3,
            device=device,
            dtype=condition.dtype,
        ),
        "masked_chain_length": chain_length.to(device=device, dtype=torch.long),
        "masked_chain_type_bits": state_bits,
        "timesteps": torch.zeros(
            batch_size,
            device=device,
            dtype=condition.dtype,
        ),
    }


def build_ordered_direction_transition_inputs(
    *,
    condition: torch.Tensor,
    chain_length: torch.Tensor,
    chain_type_bits: torch.Tensor,
    timestep: float | torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Build one completed-descriptor state for a direction reverse step."""

    inputs = build_ordered_direction_inputs(
        condition=condition,
        chain_length=chain_length,
        chain_type_bits=chain_type_bits,
    )
    batch_size = int(condition.shape[0])
    if isinstance(timestep, torch.Tensor):
        timesteps = timestep.to(device=condition.device, dtype=condition.dtype)
        if timesteps.ndim == 0:
            timesteps = timesteps.expand(batch_size)
        if timesteps.shape != (batch_size,):
            raise ValueError("timestep tensor must be scalar or have shape [batch]")
    else:
        timesteps = torch.full(
            (batch_size,),
            float(timestep),
            device=condition.device,
            dtype=condition.dtype,
        )
    inputs["timesteps"] = timesteps
    return inputs


def predict_spherical_d3pm_endpoint(
    model: MaskedSeedDiffusionModel,
    *,
    condition: torch.Tensor,
    chain_length: torch.Tensor,
    chain_type_bits: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Return the exact endpoint vMF mixture of the finite reverse process.

    The reverse latent chain is analytically marginalized.  Sampling from the
    returned mixture is distributionally identical to ancestral reverse-chain
    sampling followed by the descriptor-conditioned vMF emission.
    """

    if (
        model.config.direction_distribution
        != DIRECTION_DISTRIBUTION_SPHERICAL_D3PM_VMF
    ):
        raise ValueError("model is not configured for spherical_d3pm_vmf")
    _validate_ordered_descriptor_batch(condition, chain_length, chain_type_bits)

    batch_size = int(condition.shape[0])
    emission_outputs = model(
        **build_ordered_direction_transition_inputs(
            condition=condition,
            chain_length=chain_length,
            chain_type_bits=chain_type_bits,
            timestep=0.0,
        )
    )
    means = emission_outputs["direction_emission_means"]
    if (
        model.config.direction_state_geometry
        == SPHERICAL_D3PM_GEOMETRY_LEARNED_EMISSION
    ):
        transition_geometry = means
    else:
        transition_geometry = model.direction_anchors

    reverse_matrices: list[torch.Tensor] = []
    reverse_state_logits: list[torch.Tensor] = []
    reverse_localities: list[torch.Tensor] = []
    step_count = int(model.config.direction_diffusion_steps)
    for step in range(step_count, 0, -1):
        outputs = model(
            **build_ordered_direction_transition_inputs(
                condition=condition,
                chain_length=chain_length,
                chain_type_bits=chain_type_bits,
                timestep=float(step) / float(step_count),
            )
        )
        state_logits = outputs["direction_state_logits"]
        locality = outputs["direction_locality"]
        reverse_state_logits.append(state_logits)
        reverse_localities.append(locality)
        reverse_matrices.append(
            spherical_reverse_matrix(
                state_logits,
                locality,
                transition_geometry,
            )
        )

    matrix_stack = torch.stack(reverse_matrices, dim=1)
    terminal_weights = spherical_terminal_distribution(matrix_stack)
    kappas = torch.full(
        (batch_size, model.config.direction_states),
        float(model.config.direction_grid_kappa),
        device=condition.device,
        dtype=condition.dtype,
    )
    direction = torch.nn.functional.normalize(
        (terminal_weights.unsqueeze(2) * means).sum(dim=1),
        dim=1,
        eps=1e-6,
    )
    return {
        "direction": direction,
        "vmf_logits": torch.log(terminal_weights.clamp_min(1e-30)),
        "vmf_means": means,
        "vmf_kappas": kappas,
        "direction_terminal_weights": terminal_weights,
        "direction_reverse_matrices": matrix_stack,
        "direction_reverse_state_logits": torch.stack(
            reverse_state_logits,
            dim=1,
        ),
        "direction_reverse_localities": torch.stack(
            reverse_localities,
            dim=1,
        ),
        "direction_emission_means": means,
        "direction_transition_geometry": (
            transition_geometry
            if transition_geometry.ndim == 3
            else transition_geometry.unsqueeze(0).expand(batch_size, -1, -1)
        ),
    }


def ordered_valid_length_log_probs(
    chain_length_logits: torch.Tensor,
) -> torch.Tensor:
    """Normalize length logits over the renderer-valid classes 1 through 9."""

    if chain_length_logits.ndim != 2:
        raise ValueError("chain_length_logits must have shape [batch, classes]")
    if chain_length_logits.shape[1] <= MAX_CHAIN_TYPE_BITS:
        raise ValueError("chain_length_logits does not contain every valid class")
    return torch.log_softmax(
        chain_length_logits[:, 1 : MAX_CHAIN_TYPE_BITS + 1],
        dim=1,
    )


def _as_single_condition(condition: torch.Tensor) -> torch.Tensor:
    """Normalize a single condition vector to a one-row batch."""

    if condition.ndim == 1:
        condition = condition.unsqueeze(0)
    if condition.ndim != 2 or condition.shape[0] != 1:
        raise ValueError("ordered descriptor helpers require one condition")
    return condition


def _canonical_padded_bits(
    bits: Sequence[int],
    *,
    chain_length: int,
    device: torch.device,
) -> torch.Tensor:
    """Convert a valid descriptor bit prefix into one padded tensor row."""

    if chain_length < 1 or chain_length > MAX_CHAIN_TYPE_BITS:
        raise ValueError(
            f"chain_length must be in [1, {MAX_CHAIN_TYPE_BITS}]"
        )
    if len(bits) < chain_length:
        raise ValueError("chain_type_bits is shorter than chain_length")
    valid_bits = tuple(int(value) for value in bits[:chain_length])
    if any(value not in (0, 1) for value in valid_bits):
        raise ValueError("chain_type_bits must contain only 0 or 1")
    padded = torch.zeros(
        1,
        MAX_CHAIN_TYPE_BITS,
        device=device,
        dtype=torch.long,
    )
    if valid_bits:
        padded[0, :chain_length] = torch.tensor(
            valid_bits,
            device=device,
            dtype=torch.long,
        )
    return padded


@torch.no_grad()
def ordered_descriptor_log_prob(
    model: nn.Module,
    condition: torch.Tensor,
    *,
    chain_length: int,
    chain_type_bits: Sequence[int],
) -> float:
    """Evaluate the exact normalized mass of one ordered descriptor.

    The function teacher-forces every left-to-right reverse transition.  It is
    deterministic in evaluation mode and restores the model's previous mode.
    """

    condition = _as_single_condition(condition)
    device = condition.device
    length_tensor = torch.tensor([chain_length], device=device, dtype=torch.long)
    padded_bits = _canonical_padded_bits(
        chain_type_bits,
        chain_length=chain_length,
        device=device,
    )

    was_training = model.training
    model.eval()
    try:
        length_outputs = model(**build_ordered_length_inputs(condition))
        length_log_probs = ordered_valid_length_log_probs(
            length_outputs["chain_length_logits"]
        )
        log_probability = length_log_probs[0, chain_length - 1]

        for prefix in range(chain_length):
            prefix_tensor = torch.tensor([prefix], device=device, dtype=torch.long)
            type_outputs = model(
                **build_ordered_type_inputs(
                    condition=condition,
                    chain_length=length_tensor,
                    chain_type_bits=padded_bits,
                    prefix_length=prefix_tensor,
                )
            )
            logit = type_outputs["chain_type_logits"][0, prefix]
            bit = int(padded_bits[0, prefix].item())
            log_probability = log_probability + (
                torch.nn.functional.logsigmoid(logit)
                if bit == 1
                else torch.nn.functional.logsigmoid(-logit)
            )
        return float(log_probability.item())
    finally:
        model.train(was_training)


def _unit_interval(value: float) -> float:
    """Validate and canonicalize an explicit uniform variate."""

    result = float(value)
    if not math.isfinite(result) or result < 0.0 or result > 1.0:
        raise ValueError("uniform variates must be finite and in [0, 1]")
    if result == 1.0:
        result = math.nextafter(1.0, 0.0)
    return result


@torch.no_grad()
def sample_ordered_descriptor_from_uniforms(
    model: nn.Module,
    condition: torch.Tensor,
    *,
    uniforms: Sequence[float],
) -> OrderedDescriptorSample:
    """Sample an ordered descriptor using an explicit deterministic RNG stream.

    ``uniforms[0]`` samples length and subsequent entries sample each revealed
    bit.  Supplying explicit variates makes Python/C++ parity tests exact.
    """

    if not uniforms:
        raise ValueError("at least one uniform variate is required")
    condition = _as_single_condition(condition)
    device = condition.device

    was_training = model.training
    model.eval()
    try:
        length_outputs = model(**build_ordered_length_inputs(condition))
        length_log_probs = ordered_valid_length_log_probs(
            length_outputs["chain_length_logits"]
        )[0]
        length_probs = torch.exp(length_log_probs)
        length_u = _unit_interval(uniforms[0])
        cumulative = torch.cumsum(length_probs, dim=0)
        length_index = int(
            torch.searchsorted(
                cumulative,
                torch.tensor(length_u, device=device, dtype=cumulative.dtype),
                right=False,
            ).item()
        )
        length_index = min(length_index, MAX_CHAIN_TYPE_BITS - 1)
        chain_length = length_index + 1
        if len(uniforms) < chain_length + 1:
            raise ValueError(
                f"sampling length {chain_length} requires {chain_length + 1} uniforms"
            )

        length_tensor = torch.tensor(
            [chain_length],
            device=device,
            dtype=torch.long,
        )
        padded_bits = torch.zeros(
            1,
            MAX_CHAIN_TYPE_BITS,
            device=device,
            dtype=torch.long,
        )
        log_probability = length_log_probs[length_index]
        sampled_bits: list[int] = []

        for prefix in range(chain_length):
            prefix_tensor = torch.tensor([prefix], device=device, dtype=torch.long)
            outputs = model(
                **build_ordered_type_inputs(
                    condition=condition,
                    chain_length=length_tensor,
                    chain_type_bits=padded_bits,
                    prefix_length=prefix_tensor,
                )
            )
            logit = outputs["chain_type_logits"][0, prefix]
            probability_one = torch.sigmoid(logit)
            bit = 1 if _unit_interval(uniforms[prefix + 1]) < float(probability_one.item()) else 0
            padded_bits[0, prefix] = bit
            sampled_bits.append(bit)
            log_probability = log_probability + (
                torch.nn.functional.logsigmoid(logit)
                if bit == 1
                else torch.nn.functional.logsigmoid(-logit)
            )

        log_probability_value = float(log_probability.item())
        return OrderedDescriptorSample(
            chain_length=chain_length,
            chain_type_bits=tuple(sampled_bits),
            log_probability=log_probability_value,
            probability=math.exp(log_probability_value),
        )
    finally:
        model.train(was_training)
