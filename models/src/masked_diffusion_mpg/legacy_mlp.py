from __future__ import annotations
from dataclasses import dataclass
import torch
import torch.nn.functional as F
from torch import nn
from .dataset import MASK_CHAIN_LENGTH_TOKEN

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


