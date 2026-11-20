from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from .comparison import SimpleBaselineConfig, SimpleGuidanceMLP
from .dataset import (
    CONDITION_DIM,
    MASK_CHAIN_LENGTH_TOKEN,
    MASK_CHAIN_TYPE_TOKEN,
    MAX_CHAIN_TYPE_BITS,
    NPM_ADAPTED_DEFAULT_ROUGHNESS,
)
from .model import (
    DIRECTION_CONDITIONING_FULL,
    DIRECTION_DISTRIBUTION_SPHERICAL_D3PM_VMF,
    MaskedSeedDiffusionModel,
    build_sinusoidal_time_embedding,
)
from .npm_adapted import (
    NPM_ADAPTED_MODEL_KIND,
    load_npm_adapted_checkpoint,
)
from .training import (
    MaskedDiffusionTrainingConfig,
    build_model_config,
)


class RuntimeExportError(RuntimeError):
    """Describes Phase 12 runtime export failures.

    Attributes:
        message: Error description.
    """


@dataclass(slots=True, frozen=True)
class RuntimeExportConfig:
    """Phase 12 runtime export configuration.

    Attributes:
        checkpoint_path: Relative path to checkpoint required for export.
        output_dir: Export output directory.
        output_name: Exported model filename.
        model_kind: Export model kind.
    """

    checkpoint_path: str
    output_dir: str
    output_name: str = "masked_diffusion_runtime.txt"
    model_kind: str = "masked_diffusion"


@dataclass(slots=True, frozen=True)
class RuntimeExportArtifacts:
    """Paths produced by Phase 12 export.

    Attributes:
        runtime_path: Runtime model text path.
        summary_json: Export summary JSON path.
    """

    runtime_path: Path
    summary_json: Path


def load_runtime_export_config(config_path: Path) -> RuntimeExportConfig:
    """Load runtime export configuration.

    Args:
        config_path: Path to JSON config.

    Returns:
        Parsed config object.
    """

    payload = json.loads(config_path.read_text(encoding="utf-8"))
    return RuntimeExportConfig(
        checkpoint_path=str(payload["checkpoint_path"]),
        output_dir=str(payload["output_dir"]),
        output_name=str(payload.get("output_name", "masked_diffusion_runtime.txt")),
        model_kind=str(payload.get("model_kind", "masked_diffusion")),
    )


def resolve_runtime_export_checkpoint(
    *,
    workspace_root: Path,
    checkpoint_reference: str,
) -> Path:
    """Resolve checkpoint path for runtime export.

    Args:
        workspace_root: Workspace root directory.
        checkpoint_reference: Checkpoint path relative to config.

    Returns:
        Absolute checkpoint path.

    Raises:
        RuntimeExportError: Missing checkpoint file.
    """

    checkpoint_path = (workspace_root / checkpoint_reference).resolve()
    if not checkpoint_path.exists():
        raise RuntimeExportError(f"missing checkpoint: {checkpoint_path}")
    return checkpoint_path


def load_masked_diffusion_checkpoint(
    checkpoint_path: Path,
) -> tuple[MaskedSeedDiffusionModel, MaskedDiffusionTrainingConfig]:
    """Load trained masked-diffusion checkpoint.

    Args:
        checkpoint_path: Path to checkpoint.

    Returns:
        Model instance and training config.
    """

    if not checkpoint_path.exists():
        raise RuntimeExportError(f"missing checkpoint: {checkpoint_path}")

    payload = torch.load(checkpoint_path, map_location="cpu")
    training_config = MaskedDiffusionTrainingConfig(**payload["config"])
    model = MaskedSeedDiffusionModel(build_model_config(training_config))
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model, training_config


def load_simple_baseline_checkpoint(
    checkpoint_path: Path,
) -> tuple[SimpleGuidanceMLP, dict[str, object]]:
    """Load simple MLP baseline checkpoint.

    Args:
        checkpoint_path: Path to checkpoint.

    Returns:
        Simple baseline model and raw config dictionary.
    """

    if not checkpoint_path.exists():
        raise RuntimeExportError(f"missing checkpoint: {checkpoint_path}")

    payload = torch.load(checkpoint_path, map_location="cpu")
    simple_config = SimpleBaselineConfig(**payload["config"]["simple_baseline"])
    model = SimpleGuidanceMLP(CONDITION_DIM, simple_config)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model, dict(payload["config"])


def make_fixed_runtime_inputs(
    model: MaskedSeedDiffusionModel,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build fixed non-conditioning inputs baked into runtime.

    Args:
        model: Loaded masked-diffusion model.

    Returns:
        Zero direction, timestep embedding, length token embedding, type token embedding.
    """

    with torch.no_grad():
        noisy_direction = torch.zeros(1, 3, dtype=torch.float32)
        timesteps = torch.zeros(1, dtype=torch.float32)
        time_embedding = build_sinusoidal_time_embedding(
            timesteps,
            model.config.time_embedding_dim,
        )
        chain_length_embedding = model.chain_length_embedding(
            torch.tensor([MASK_CHAIN_LENGTH_TOKEN], dtype=torch.long)
        )
        chain_type_embedding = model.chain_type_embedding(
            torch.full(
                (1, MAX_CHAIN_TYPE_BITS),
                fill_value=MASK_CHAIN_TYPE_TOKEN,
                dtype=torch.long,
            )
        ).flatten(start_dim=1)
    return (
        noisy_direction.squeeze(0),
        time_embedding.squeeze(0),
        chain_length_embedding.squeeze(0),
        chain_type_embedding.squeeze(0),
    )


def fold_runtime_input_projection(
    model: MaskedSeedDiffusionModel,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fold fixed inputs into the first-layer bias.

    Args:
        model: Loaded masked-diffusion model.

    Returns:
        First-layer weights for condition-only input and fused bias.
    """

    linear = model.input_projection[0]
    if not isinstance(linear, torch.nn.Linear):
        raise RuntimeExportError("unexpected input_projection layout")

    condition_dim = model.config.condition_dim
    (
        noisy_direction,
        time_embedding,
        chain_length_embedding,
        chain_type_embedding,
    ) = make_fixed_runtime_inputs(model)
    constant_tail = torch.cat(
        [
            noisy_direction,
            time_embedding,
            chain_length_embedding,
            chain_type_embedding,
        ],
        dim=0,
    )

    weight = linear.weight.detach().cpu()
    bias = linear.bias.detach().cpu()
    folded_weight = weight[:, :condition_dim].contiguous()
    folded_bias = (
        bias + torch.matmul(weight[:, condition_dim:], constant_tail)
    ).contiguous()
    return folded_weight, folded_bias


def full_runtime_input_projection(
    model: MaskedSeedDiffusionModel,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the full first-layer projection used by multi-step runtime.

    Unlike ``fold_runtime_input_projection``, this keeps the noisy direction,
    timestep embedding, chain-length token embedding, and chain-type token
    embeddings as dynamic runtime inputs. The C++ renderer can then reproduce
    Phase11-style autoregressive denoising.
    """

    linear = model.input_projection[0]
    if not isinstance(linear, torch.nn.Linear):
        raise RuntimeExportError("unexpected input_projection layout")
    return (
        linear.weight.detach().cpu().contiguous(),
        linear.bias.detach().cpu().contiguous(),
    )


def fold_simple_runtime_input_projection(
    model: SimpleGuidanceMLP,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Export simple MLP input-layer weights.

    Args:
        model: Loaded simple baseline model.

    Returns:
        First-layer weights and bias for simple MLP.
    """

    linear = model.input_projection[0]
    if not isinstance(linear, torch.nn.Linear):
        raise RuntimeExportError("unexpected simple input_projection layout")
    return (
        linear.weight.detach().cpu().contiguous(),
        linear.bias.detach().cpu().contiguous(),
    )


def flatten_tensor(tensor: torch.Tensor) -> list[float]:
    """Flatten tensor to Python float list.

    Args:
        tensor: Source tensor.

    Returns:
        Flattened floats.
    """

    return [float(value) for value in tensor.detach().cpu().reshape(-1).tolist()]


def write_tensor_block(handle, name: str, tensor: torch.Tensor) -> None:
    """Write one tensor blob to text.

    Args:
        handle: Open text handle.
        name: Tensor name.
        tensor: Tensor payload.
    """

    dims = " ".join(str(dim) for dim in tensor.shape)
    ndim = tensor.ndim
    values = " ".join(f"{value:.9g}" for value in flatten_tensor(tensor))
    handle.write(f"tensor {name} {ndim} {dims}\n")
    handle.write(values)
    handle.write("\n")


def write_runtime_text_header(
    *,
    handle,
    hidden_dim: int,
    depth: int,
    condition_dim: int,
    backbone_type: str = "mlp",
    direction_distribution: str = "vector",
    vmf_components: int = 0,
    vmf_min_kappa: float = 1e-4,
    vmf_max_kappa: float = 128.0,
    transformer_heads: int = 0,
    transformer_feedforward_dim: int = 0,
    token_count: int = 0,
    time_embedding_dim: int = 0,
    token_embedding_dim: int = 0,
    runtime_input_mode: str = "condition_only",
    discrete_diffusion_mode: str = "legacy",
    direction_conditioning_mode: str = DIRECTION_CONDITIONING_FULL,
) -> None:
    """Write runtime header section.

    Args:
        handle: Open text handle.
        hidden_dim: Hidden width.
        depth: Depth of residual stack.
        condition_dim: Condition vector dimension.
    """

    # C++ loader shares one text protocol, so simple MLP and masked diffusion reuse this header shape.
    handle.write("masked_diffusion_runtime_v1\n")
    handle.write(f"hidden_dim {hidden_dim}\n")
    handle.write(f"depth {depth}\n")
    handle.write(f"condition_dim {condition_dim}\n")
    handle.write(f"backbone_type {backbone_type}\n")
    handle.write(f"direction_distribution {direction_distribution}\n")
    handle.write(f"vmf_components {vmf_components}\n")
    if direction_distribution == "vmf_mixture":
        handle.write(f"vmf_min_kappa {vmf_min_kappa:.8g}\n")
    handle.write(f"vmf_max_kappa {vmf_max_kappa:.8g}\n")
    handle.write(f"transformer_heads {transformer_heads}\n")
    handle.write(f"transformer_feedforward_dim {transformer_feedforward_dim}\n")
    handle.write(f"token_count {token_count}\n")
    handle.write(f"time_embedding_dim {time_embedding_dim}\n")
    handle.write(f"token_embedding_dim {token_embedding_dim}\n")
    handle.write(f"runtime_input_mode {runtime_input_mode}\n")
    handle.write(f"discrete_diffusion_mode {discrete_diffusion_mode}\n")
    handle.write(f"direction_conditioning_mode {direction_conditioning_mode}\n")
    handle.write(f"chain_length_classes {MASK_CHAIN_LENGTH_TOKEN}\n")
    handle.write(f"chain_type_bits {MAX_CHAIN_TYPE_BITS}\n")
    handle.write("layer_norm_eps 1e-05\n")


def write_residual_runtime_blocks(
    *,
    handle,
    blocks: torch.nn.ModuleList,
) -> None:
    """Write shared residual-block weights.

    Args:
        handle: Open text handle.
        blocks: Module list of residual blocks.
    """

    for index, block in enumerate(blocks):
        write_tensor_block(handle, f"block_{index}_norm_weight", block.block[0].weight)
        write_tensor_block(handle, f"block_{index}_norm_bias", block.block[0].bias)
        write_tensor_block(
            handle, f"block_{index}_linear1_weight", block.block[1].weight
        )
        write_tensor_block(handle, f"block_{index}_linear1_bias", block.block[1].bias)
        write_tensor_block(
            handle, f"block_{index}_linear2_weight", block.block[4].weight
        )
        write_tensor_block(handle, f"block_{index}_linear2_bias", block.block[4].bias)


def make_transformer_constant_tokens(
    model: MaskedSeedDiffusionModel,
) -> torch.Tensor:
    """Export Transformer constant tokens fused at runtime."""

    with torch.no_grad():
        noisy_direction = torch.zeros(1, 3, dtype=torch.float32)
        timesteps = torch.zeros(1, dtype=torch.float32)
        time_embedding = build_sinusoidal_time_embedding(
            timesteps,
            model.config.time_embedding_dim,
        )
        direction_token = model.direction_time_projection(
            torch.cat([noisy_direction, time_embedding], dim=1)
        )
        chain_length_token = model.chain_length_projection(
            model.chain_length_embedding(
                torch.tensor([MASK_CHAIN_LENGTH_TOKEN], dtype=torch.long)
            )
        )
        chain_type_tokens = model.chain_type_projection(
            model.chain_type_embedding(
                torch.full(
                    (1, MAX_CHAIN_TYPE_BITS),
                    fill_value=MASK_CHAIN_TYPE_TOKEN,
                    dtype=torch.long,
                )
            )
        ).squeeze(0)
    return torch.cat(
        [
            direction_token.squeeze(0).unsqueeze(0),
            chain_length_token.squeeze(0).unsqueeze(0),
            chain_type_tokens,
        ],
        dim=0,
    ).contiguous()


def write_transformer_runtime_blocks(
    *,
    handle,
    model: MaskedSeedDiffusionModel,
) -> None:
    """Write Transformer backbone tensors for runtime."""

    if not hasattr(model, "transformer"):
        raise RuntimeExportError("model is not a transformer backbone")

    write_tensor_block(
        handle,
        "condition_token_weight",
        model.condition_projection[0].weight,
    )
    write_tensor_block(
        handle, "condition_token_bias", model.condition_projection[0].bias
    )
    # Ordered absorbing inference changes the length/type/timestep state between
    # transitions. Export the unfused projections and embeddings so the C++
    # runtime can reconstruct those tokens exactly. Legacy runtimes continue to
    # use ``runtime_token_constants`` and ignore these additional tensors.
    write_tensor_block(
        handle,
        "direction_time_token_weight",
        model.direction_time_projection[0].weight,
    )
    write_tensor_block(
        handle,
        "direction_time_token_bias",
        model.direction_time_projection[0].bias,
    )
    write_tensor_block(
        handle,
        "chain_length_token_weight",
        model.chain_length_projection.weight,
    )
    write_tensor_block(
        handle,
        "chain_length_token_bias",
        model.chain_length_projection.bias,
    )
    write_tensor_block(
        handle,
        "chain_type_token_weight",
        model.chain_type_projection.weight,
    )
    write_tensor_block(
        handle,
        "chain_type_token_bias",
        model.chain_type_projection.bias,
    )
    write_tensor_block(
        handle,
        "chain_length_embedding_weight",
        model.chain_length_embedding.weight,
    )
    write_tensor_block(
        handle,
        "chain_type_embedding_weight",
        model.chain_type_embedding.weight,
    )
    write_tensor_block(
        handle,
        "runtime_token_constants",
        make_transformer_constant_tokens(model),
    )
    write_tensor_block(
        handle, "transformer_position_embedding", model.position_embedding
    )
    for index, layer in enumerate(model.transformer.layers):
        prefix = f"transformer_{index}"
        write_tensor_block(
            handle,
            f"{prefix}_self_attn_in_proj_weight",
            layer.self_attn.in_proj_weight,
        )
        write_tensor_block(
            handle,
            f"{prefix}_self_attn_in_proj_bias",
            layer.self_attn.in_proj_bias,
        )
        write_tensor_block(
            handle,
            f"{prefix}_self_attn_out_proj_weight",
            layer.self_attn.out_proj.weight,
        )
        write_tensor_block(
            handle,
            f"{prefix}_self_attn_out_proj_bias",
            layer.self_attn.out_proj.bias,
        )
        write_tensor_block(handle, f"{prefix}_linear1_weight", layer.linear1.weight)
        write_tensor_block(handle, f"{prefix}_linear1_bias", layer.linear1.bias)
        write_tensor_block(handle, f"{prefix}_linear2_weight", layer.linear2.weight)
        write_tensor_block(handle, f"{prefix}_linear2_bias", layer.linear2.bias)
        write_tensor_block(handle, f"{prefix}_norm1_weight", layer.norm1.weight)
        write_tensor_block(handle, f"{prefix}_norm1_bias", layer.norm1.bias)
        write_tensor_block(handle, f"{prefix}_norm2_weight", layer.norm2.weight)
        write_tensor_block(handle, f"{prefix}_norm2_bias", layer.norm2.bias)


def write_dit_runtime_blocks(
    *,
    handle,
    model: MaskedSeedDiffusionModel,
) -> None:
    """Write compact DiT tensors, including every AdaLN modulation."""

    if not hasattr(model, "dit_blocks"):
        raise RuntimeExportError("model is not a DiT backbone")

    write_tensor_block(
        handle,
        "dit_condition_weight",
        model.dit_condition_projection.weight,
    )
    write_tensor_block(
        handle,
        "dit_condition_bias",
        model.dit_condition_projection.bias,
    )
    write_tensor_block(
        handle,
        "dit_time_linear1_weight",
        model.dit_time_projection[0].weight,
    )
    write_tensor_block(
        handle,
        "dit_time_linear1_bias",
        model.dit_time_projection[0].bias,
    )
    write_tensor_block(
        handle,
        "dit_time_linear2_weight",
        model.dit_time_projection[2].weight,
    )
    write_tensor_block(
        handle,
        "dit_time_linear2_bias",
        model.dit_time_projection[2].bias,
    )
    write_tensor_block(
        handle,
        "dit_direction_token_weight",
        model.dit_direction_projection.weight,
    )
    write_tensor_block(
        handle,
        "dit_direction_token_bias",
        model.dit_direction_projection.bias,
    )
    write_tensor_block(
        handle,
        "chain_length_token_weight",
        model.chain_length_projection.weight,
    )
    write_tensor_block(
        handle,
        "chain_length_token_bias",
        model.chain_length_projection.bias,
    )
    write_tensor_block(
        handle,
        "chain_type_token_weight",
        model.chain_type_projection.weight,
    )
    write_tensor_block(
        handle,
        "chain_type_token_bias",
        model.chain_type_projection.bias,
    )
    write_tensor_block(
        handle,
        "chain_length_embedding_weight",
        model.chain_length_embedding.weight,
    )
    write_tensor_block(
        handle,
        "chain_type_embedding_weight",
        model.chain_type_embedding.weight,
    )
    write_tensor_block(handle, "dit_position_embedding", model.position_embedding)

    for index, block in enumerate(model.dit_blocks):
        prefix = f"dit_{index}"
        write_tensor_block(
            handle,
            f"{prefix}_self_attn_in_proj_weight",
            block.attention.in_proj_weight,
        )
        write_tensor_block(
            handle,
            f"{prefix}_self_attn_in_proj_bias",
            block.attention.in_proj_bias,
        )
        write_tensor_block(
            handle,
            f"{prefix}_self_attn_out_proj_weight",
            block.attention.out_proj.weight,
        )
        write_tensor_block(
            handle,
            f"{prefix}_self_attn_out_proj_bias",
            block.attention.out_proj.bias,
        )
        write_tensor_block(handle, f"{prefix}_linear1_weight", block.mlp[0].weight)
        write_tensor_block(handle, f"{prefix}_linear1_bias", block.mlp[0].bias)
        write_tensor_block(handle, f"{prefix}_linear2_weight", block.mlp[3].weight)
        write_tensor_block(handle, f"{prefix}_linear2_bias", block.mlp[3].bias)
        write_tensor_block(
            handle,
            f"{prefix}_adaln_weight",
            block.adaLN_modulation[1].weight,
        )
        write_tensor_block(
            handle,
            f"{prefix}_adaln_bias",
            block.adaLN_modulation[1].bias,
        )

    write_tensor_block(
        handle,
        "dit_final_adaln_weight",
        model.final_norm.adaLN_modulation[1].weight,
    )
    write_tensor_block(
        handle,
        "dit_final_adaln_bias",
        model.final_norm.adaLN_modulation[1].bias,
    )


def write_direction_runtime_heads(
    *,
    handle,
    model: MaskedSeedDiffusionModel,
) -> None:
    """Write direction-output heads."""

    if model.config.direction_distribution == "vmf_mixture":
        write_tensor_block(
            handle, "vmf_logits_head_weight", model.vmf_logits_head.weight
        )
        write_tensor_block(handle, "vmf_logits_head_bias", model.vmf_logits_head.bias)
        write_tensor_block(handle, "vmf_mean_head_weight", model.vmf_mean_head.weight)
        write_tensor_block(handle, "vmf_mean_head_bias", model.vmf_mean_head.bias)
        write_tensor_block(handle, "vmf_kappa_head_weight", model.vmf_kappa_head.weight)
        write_tensor_block(handle, "vmf_kappa_head_bias", model.vmf_kappa_head.bias)
    else:
        write_tensor_block(handle, "direction_head_weight", model.direction_head.weight)
        write_tensor_block(handle, "direction_head_bias", model.direction_head.bias)


def export_runtime_model(
    *,
    workspace_root: Path,
    config: RuntimeExportConfig,
) -> RuntimeExportArtifacts:
    """Export one-step weights readable by the C++ runtime.

    Args:
        workspace_root: Masked-diffusion workspace root.
        config: Export settings.

    Returns:
        Artifact paths.
    """

    checkpoint_path = resolve_runtime_export_checkpoint(
        workspace_root=workspace_root,
        checkpoint_reference=config.checkpoint_path,
    )
    output_root = (workspace_root / config.output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    runtime_path = output_root / config.output_name
    if config.model_kind == "masked_diffusion":
        model, training_config = load_masked_diffusion_checkpoint(checkpoint_path)
        if (
            model.config.direction_distribution
            == DIRECTION_DISTRIBUTION_SPHERICAL_D3PM_VMF
        ):
            raise RuntimeExportError(
                "spherical_d3pm_vmf is currently Python-only; native runtime "
                "export is intentionally disabled until C++ parity is implemented"
            )
        discrete_diffusion_mode = str(
            getattr(training_config, "discrete_diffusion_mode", "legacy")
        ).lower()
        if discrete_diffusion_mode not in {"legacy", "ordered_absorbing"}:
            raise RuntimeExportError(
                "unsupported discrete_diffusion_mode: " f"{discrete_diffusion_mode}"
            )
        if model.config.backbone_type == "mlp":
            runtime_input_mode = "full_mlp"
        elif model.config.backbone_type == "dit":
            runtime_input_mode = "full_dit"
        else:
            runtime_input_mode = (
                "full_transformer"
                if discrete_diffusion_mode == "ordered_absorbing"
                else "condition_only"
            )
        with runtime_path.open("w", encoding="utf-8", newline="\n") as handle:
            write_runtime_text_header(
                handle=handle,
                hidden_dim=model.config.hidden_dim,
                depth=model.config.depth,
                condition_dim=model.config.condition_dim,
                backbone_type=model.config.backbone_type,
                direction_distribution=model.config.direction_distribution,
                vmf_components=(
                    model.config.vmf_components
                    if model.config.direction_distribution == "vmf_mixture"
                    else 0
                ),
                vmf_max_kappa=(
                    model.config.vmf_max_kappa
                    if model.config.direction_distribution == "vmf_mixture"
                    else 128.0
                ),
                transformer_heads=(
                    model.config.transformer_heads
                    if model.config.backbone_type in {"transformer", "dit"}
                    else 0
                ),
                transformer_feedforward_dim=(
                    (
                        model.config.transformer_feedforward_dim
                        if model.config.transformer_feedforward_dim is not None
                        else model.config.hidden_dim * 4
                    )
                    if model.config.backbone_type in {"transformer", "dit"}
                    else 0
                ),
                token_count=(
                    (
                        3 + MAX_CHAIN_TYPE_BITS
                        if model.config.backbone_type == "transformer"
                        else 2 + MAX_CHAIN_TYPE_BITS
                    )
                    if model.config.backbone_type in {"transformer", "dit"}
                    else 0
                ),
                time_embedding_dim=(
                    model.config.time_embedding_dim
                    if runtime_input_mode
                    in {"full_mlp", "full_transformer", "full_dit"}
                    else 0
                ),
                token_embedding_dim=(
                    model.config.token_embedding_dim
                    if runtime_input_mode
                    in {"full_mlp", "full_transformer", "full_dit"}
                    else 0
                ),
                runtime_input_mode=runtime_input_mode,
                discrete_diffusion_mode=discrete_diffusion_mode,
                direction_conditioning_mode=model.config.direction_conditioning_mode,
            )
            if model.config.backbone_type == "transformer":
                write_transformer_runtime_blocks(handle=handle, model=model)
            elif model.config.backbone_type == "dit":
                write_dit_runtime_blocks(handle=handle, model=model)
            else:
                input_weight, input_bias = full_runtime_input_projection(model)
                write_tensor_block(handle, "input_weight", input_weight)
                write_tensor_block(handle, "input_bias", input_bias)
                write_tensor_block(
                    handle,
                    "chain_length_embedding_weight",
                    model.chain_length_embedding.weight,
                )
                write_tensor_block(
                    handle,
                    "chain_type_embedding_weight",
                    model.chain_type_embedding.weight,
                )
                write_residual_runtime_blocks(handle=handle, blocks=model.blocks)
            if model.config.backbone_type != "dit":
                write_tensor_block(handle, "final_norm_weight", model.final_norm.weight)
                write_tensor_block(handle, "final_norm_bias", model.final_norm.bias)
            write_tensor_block(
                handle, "chain_length_head_weight", model.chain_length_head.weight
            )
            write_tensor_block(
                handle, "chain_length_head_bias", model.chain_length_head.bias
            )
            if model.config.backbone_type in {"transformer", "dit"}:
                write_tensor_block(
                    handle,
                    "chain_type_token_head_weight",
                    model.chain_type_token_head.weight,
                )
                write_tensor_block(
                    handle,
                    "chain_type_token_head_bias",
                    model.chain_type_token_head.bias,
                )
            else:
                write_tensor_block(
                    handle, "chain_type_head_weight", model.chain_type_head.weight
                )
                write_tensor_block(
                    handle, "chain_type_head_bias", model.chain_type_head.bias
                )
            write_direction_runtime_heads(handle=handle, model=model)
        training_metadata: dict[str, object] = asdict(training_config)
        model_metadata = {
            "condition_dim": model.config.condition_dim,
            "hidden_dim": model.config.hidden_dim,
            "depth": model.config.depth,
            "dropout": model.config.dropout,
            "time_embedding_dim": model.config.time_embedding_dim,
            "token_embedding_dim": model.config.token_embedding_dim,
            "backbone_type": model.config.backbone_type,
            "direction_distribution": model.config.direction_distribution,
            "vmf_components": model.config.vmf_components,
            "vmf_max_kappa": model.config.vmf_max_kappa,
            "transformer_heads": model.config.transformer_heads,
            "transformer_feedforward_dim": model.config.transformer_feedforward_dim,
            "runtime_input_mode": runtime_input_mode,
            "discrete_diffusion_mode": discrete_diffusion_mode,
            "direction_conditioning_mode": model.config.direction_conditioning_mode,
            "runtime_multistep_supported": bool(
                model.config.backbone_type == "mlp"
                and discrete_diffusion_mode == "legacy"
            ),
            "runtime_ordered_descriptor_supported": bool(
                discrete_diffusion_mode == "ordered_absorbing"
                and runtime_input_mode in {"full_mlp", "full_transformer", "full_dit"}
            ),
            # The MPG integrator keeps its existing length-only mixture, then
            # samples the ordered type-bit transitions conditional on that
            # selected length before predicting direction from the completed
            # descriptor state.
            "runtime_integrator_ordered_descriptor_wired": bool(
                discrete_diffusion_mode == "ordered_absorbing"
                and runtime_input_mode in {"full_mlp", "full_transformer", "full_dit"}
            ),
        }
    elif config.model_kind in {"npm_adapted", NPM_ADAPTED_MODEL_KIND}:
        model, checkpoint_payload = load_npm_adapted_checkpoint(
            checkpoint_path,
            use_ema=True,
        )
        model_config = model.config
        if any(character.isspace() for character in model_config.descriptor_policy_id):
            raise RuntimeExportError(
                "NPM descriptor_policy_id cannot contain whitespace in the text format"
            )

        with runtime_path.open("w", encoding="utf-8", newline="\n") as handle:
            write_runtime_text_header(
                handle=handle,
                hidden_dim=model_config.hidden_dim,
                depth=model_config.hidden_layers,
                condition_dim=CONDITION_DIM,
                backbone_type="npm_adapted",
                direction_distribution="npm_vmf_mixture",
                vmf_components=model_config.vmf_components,
                vmf_min_kappa=model_config.min_kappa,
                vmf_max_kappa=model_config.max_kappa,
                runtime_input_mode="npm_geometry_proxy",
                discrete_diffusion_mode="legacy",
            )
            handle.write(f"npm_grid_levels {len(model_config.grid_resolutions)}\n")
            handle.write(f"npm_grid_features {model_config.features_per_level}\n")
            handle.write(f"npm_sh_degree {model_config.sh_degree}\n")
            handle.write(f"npm_roughness_default {NPM_ADAPTED_DEFAULT_ROUGHNESS:.8g}\n")
            handle.write("npm_direction_source condition_delta_proxy\n")
            handle.write(
                f"npm_descriptor_policy_id {model_config.descriptor_policy_id}\n"
            )
            write_tensor_block(
                handle,
                "npm_position_min",
                model.grid_encoding.position_min,
            )
            write_tensor_block(
                handle,
                "npm_position_max",
                model.grid_encoding.position_max,
            )
            for index, (grid, resolution) in enumerate(
                zip(model.grid_encoding.grids, model_config.grid_resolutions)
            ):
                write_tensor_block(
                    handle,
                    f"npm_grid_{index}",
                    grid.view(
                        resolution,
                        resolution,
                        resolution,
                        model_config.features_per_level,
                    ),
                )
            linear_layers = [
                layer for layer in model.decoder if isinstance(layer, torch.nn.Linear)
            ]
            if len(linear_layers) != model_config.hidden_layers:
                raise RuntimeExportError("unexpected NPM decoder layout")
            for index, layer in enumerate(linear_layers):
                write_tensor_block(
                    handle,
                    f"npm_decoder_{index}_weight",
                    layer.weight,
                )
                write_tensor_block(
                    handle,
                    f"npm_decoder_{index}_bias",
                    layer.bias,
                )
            write_tensor_block(handle, "npm_output_weight", model.output_layer.weight)
            write_tensor_block(handle, "npm_output_bias", model.output_layer.bias)

        training_metadata = dict(checkpoint_payload.get("training_config", {}))
        training_metadata["npm_adapted_checkpoint"] = str(checkpoint_path)
        model_metadata = {
            **asdict(model_config),
            "model_kind": NPM_ADAPTED_MODEL_KIND,
            "backbone_type": "npm_adapted",
            "direction_distribution": "npm_vmf_mixture",
            "runtime_input_mode": "npm_geometry_proxy",
            "direction_source": "condition_delta_proxy",
            "roughness_source": (
                f"fixed_{NPM_ADAPTED_DEFAULT_ROUGHNESS:.8g}_for_legacy_condition_vector"
            ),
            "uses_ema_weights": True,
            "direction_only": True,
            "descriptor_policy_external": True,
            "native_runtime_direction_supported": True,
            "native_runtime_integrator_direction_wired": True,
            "parameter_count": sum(
                parameter.numel() for parameter in model.parameters()
            ),
        }
    elif config.model_kind == "simple_mlp":
        model, comparison_config = load_simple_baseline_checkpoint(checkpoint_path)

        simple_baseline_config = comparison_config.get("simple_baseline", {})
        probabilistic_direction = bool(
            simple_baseline_config.get("probabilistic_direction", False)
        )
        vmf_components = int(simple_baseline_config.get("vmf_components", 1))
        has_k1_heads = hasattr(model, "direction_kappa_head")
        has_mixture_heads = all(
            hasattr(model, name)
            for name in ("vmf_logits_head", "vmf_mean_head", "vmf_kappa_head")
        )
        use_vmf_mixture = probabilistic_direction and (
            (vmf_components == 1 and has_k1_heads)
            or (vmf_components > 1 and has_mixture_heads)
        )
        if probabilistic_direction and not use_vmf_mixture:
            raise RuntimeExportError(
                "simple_mlp probabilistic head layout does not match vmf_components"
            )

        # New plain-MLP baseline from comparison.py:
        #   self.backbone = nn.Sequential(Linear, SiLU, Dropout, ...)
        is_plain_mlp = hasattr(model, "backbone") and not hasattr(
            model,
            "input_projection",
        )

        with runtime_path.open("w", encoding="utf-8", newline="\n") as handle:
            if is_plain_mlp:
                linear_layers = [
                    module
                    for module in model.backbone
                    if isinstance(module, torch.nn.Linear)
                ]
                if not linear_layers:
                    raise RuntimeExportError(
                        "plain simple_mlp backbone has no Linear layers"
                    )

                write_runtime_text_header(
                    handle=handle,
                    hidden_dim=int(model.chain_length_head.in_features),
                    depth=len(linear_layers),
                    condition_dim=CONDITION_DIM,
                    backbone_type="plain_mlp",
                    direction_distribution=(
                        "vmf_mixture" if use_vmf_mixture else "vector"
                    ),
                    vmf_components=(vmf_components if use_vmf_mixture else 0),
                    vmf_min_kappa=float(simple_baseline_config.get("min_kappa", 1e-3)),
                    vmf_max_kappa=float(simple_baseline_config.get("max_kappa", 128.0)),
                )

                for index, layer in enumerate(linear_layers):
                    write_tensor_block(
                        handle,
                        f"plain_layer_{index}_weight",
                        layer.weight,
                    )
                    write_tensor_block(handle, f"plain_layer_{index}_bias", layer.bias)

            else:
                folded_weight, folded_bias = fold_simple_runtime_input_projection(model)
                write_runtime_text_header(
                    handle=handle,
                    hidden_dim=int(model.chain_length_head.in_features),
                    depth=len(model.blocks),
                    condition_dim=CONDITION_DIM,
                    backbone_type="mlp",
                    direction_distribution=(
                        "vmf_mixture" if use_vmf_mixture else "vector"
                    ),
                    vmf_components=(vmf_components if use_vmf_mixture else 0),
                    vmf_min_kappa=float(simple_baseline_config.get("min_kappa", 1e-3)),
                    vmf_max_kappa=float(simple_baseline_config.get("max_kappa", 128.0)),
                )
                write_tensor_block(handle, "input_weight", folded_weight)
                write_tensor_block(handle, "input_bias", folded_bias)
                write_residual_runtime_blocks(handle=handle, blocks=model.blocks)
                write_tensor_block(handle, "final_norm_weight", model.final_norm.weight)
                write_tensor_block(handle, "final_norm_bias", model.final_norm.bias)

            write_tensor_block(
                handle, "chain_length_head_weight", model.chain_length_head.weight
            )
            write_tensor_block(
                handle, "chain_length_head_bias", model.chain_length_head.bias
            )
            write_tensor_block(
                handle, "chain_type_head_weight", model.chain_type_head.weight
            )
            write_tensor_block(
                handle, "chain_type_head_bias", model.chain_type_head.bias
            )

            if use_vmf_mixture:
                if vmf_components == 1:
                    hidden_dim = int(model.chain_length_head.in_features)
                    write_tensor_block(
                        handle,
                        "vmf_logits_head_weight",
                        torch.zeros(1, hidden_dim, dtype=torch.float32),
                    )
                    write_tensor_block(
                        handle,
                        "vmf_logits_head_bias",
                        torch.zeros(1, dtype=torch.float32),
                    )
                    write_tensor_block(
                        handle,
                        "vmf_mean_head_weight",
                        model.direction_head.weight,
                    )
                    write_tensor_block(
                        handle,
                        "vmf_mean_head_bias",
                        model.direction_head.bias,
                    )
                    write_tensor_block(
                        handle,
                        "vmf_kappa_head_weight",
                        model.direction_kappa_head.weight,
                    )
                    write_tensor_block(
                        handle,
                        "vmf_kappa_head_bias",
                        model.direction_kappa_head.bias,
                    )
                else:
                    write_tensor_block(
                        handle,
                        "vmf_logits_head_weight",
                        model.vmf_logits_head.weight,
                    )
                    write_tensor_block(
                        handle,
                        "vmf_logits_head_bias",
                        model.vmf_logits_head.bias,
                    )
                    write_tensor_block(
                        handle,
                        "vmf_mean_head_weight",
                        model.vmf_mean_head.weight,
                    )
                    write_tensor_block(
                        handle,
                        "vmf_mean_head_bias",
                        model.vmf_mean_head.bias,
                    )
                    write_tensor_block(
                        handle,
                        "vmf_kappa_head_weight",
                        model.vmf_kappa_head.weight,
                    )
                    write_tensor_block(
                        handle,
                        "vmf_kappa_head_bias",
                        model.vmf_kappa_head.bias,
                    )
            else:
                write_tensor_block(
                    handle,
                    "direction_head_weight",
                    model.direction_head.weight,
                )
                write_tensor_block(
                    handle,
                    "direction_head_bias",
                    model.direction_head.bias,
                )

        training_metadata = comparison_config
        training_metadata["simple_baseline_checkpoint"] = str(checkpoint_path)
        model_metadata = {
            "condition_dim": CONDITION_DIM,
            "hidden_dim": int(model.chain_length_head.in_features),
            "depth": (
                len([m for m in model.backbone if isinstance(m, torch.nn.Linear)])
                if is_plain_mlp
                else len(model.blocks)
            ),
            "dropout": float(comparison_config["simple_baseline"]["dropout"]),
            "backbone_type": "plain_mlp" if is_plain_mlp else "mlp",
            "direction_distribution": "vmf_mixture" if use_vmf_mixture else "vector",
            "vmf_components": vmf_components if use_vmf_mixture else 0,
            "vmf_min_kappa": float(simple_baseline_config.get("min_kappa", 1e-3)),
            "vmf_max_kappa": float(simple_baseline_config.get("max_kappa", 128.0)),
            "uses_learned_kappa": bool(use_vmf_mixture),
        }
    else:
        raise RuntimeExportError(
            f"unknown runtime export model kind: {config.model_kind}"
        )

    summary_json = output_root / "runtime_export_summary.json"
    summary_json.write_text(
        json.dumps(
            {
                "config": asdict(config),
                "checkpoint_path": str(checkpoint_path),
                "runtime_path": str(runtime_path),
                "training_config": training_metadata,
                "model_config": model_metadata,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    return RuntimeExportArtifacts(
        runtime_path=runtime_path,
        summary_json=summary_json,
    )
