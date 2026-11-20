from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import Dataset

from .data_contract import (
    ReplayRecord,
    make_replay_record_key,
    normalize_vector,
    read_replay_records_jsonl,
)


MAX_CHAIN_LENGTH = 10
MAX_CHAIN_TYPE_BITS = MAX_CHAIN_LENGTH - 1
MASK_CHAIN_LENGTH_TOKEN = MAX_CHAIN_LENGTH
MASK_CHAIN_TYPE_TOKEN = 2
CONDITION_DIM = 22
NPM_ADAPTED_DEFAULT_ROUGHNESS = 0.5


class ReplayDatasetError(RuntimeError):
    """Raised when replay dataset construction fails."""


@dataclass(slots=True, frozen=True)
class ReplayDatasetStats:
    """Replay dataset aggregate statistics snapshot."""

    num_records: int
    num_unique_records: int
    num_duplicate_records: int
    num_source_files: int
    num_filtered_records: int
    mean_sample_weight: float
    max_chain_length: int


def flatten_condition(record: ReplayRecord) -> tuple[float, ...]:
    """Flatten conditioning features into a fixed-length numeric vector."""

    delta_x, delta_y, delta_z = record.condition.delta
    distance = max(float(record.condition.distance), 1e-6)
    delta_dir = (delta_x / distance, delta_y / distance, delta_z / distance)
    receiver_alignment = (
        delta_dir[0] * record.condition.n_d[0]
        + delta_dir[1] * record.condition.n_d[1]
        + delta_dir[2] * record.condition.n_d[2]
    )
    emitter_alignment = (
        -delta_dir[0] * record.condition.n_l[0]
        - delta_dir[1] * record.condition.n_l[1]
        - delta_dir[2] * record.condition.n_l[2]
    )
    emission_alignment = (
        delta_dir[0] * record.condition.d_l[0]
        + delta_dir[1] * record.condition.d_l[1]
        + delta_dir[2] * record.condition.d_l[2]
    )

    return (
        *record.condition.x_d,
        *record.condition.x_l,
        *record.condition.delta,
        float(record.condition.distance),
        *record.condition.n_d,
        *record.condition.n_l,
        *record.condition.d_l,
        receiver_alignment,
        emitter_alignment,
        emission_alignment,
    )


def pad_chain_type_bits(
    bits: tuple[int, ...],
    *,
    chain_length: int,
) -> tuple[int, ...]:
    """Right-pad a variable-length chain-type bitmask up to MAX_CHAIN_TYPE_BITS."""

    clamped_length = max(0, min(chain_length, MAX_CHAIN_TYPE_BITS))
    padded = [0] * MAX_CHAIN_TYPE_BITS
    for index in range(clamped_length):
        padded[index] = int(bits[index]) if index < len(bits) else 0
    return tuple(padded)


def make_chain_type_valid_mask(chain_length: int) -> tuple[float, ...]:
    """Build floats marking which bitmask positions participate in supervision."""

    clamped_length = max(0, min(chain_length, MAX_CHAIN_TYPE_BITS))
    mask = [0.0] * MAX_CHAIN_TYPE_BITS
    for index in range(clamped_length):
        mask[index] = 1.0
    return tuple(mask)


def normalize_sample_weight(weight: float) -> float:
    """Map emitter energies onto a logarithmically stabilized training scale."""

    return 1.0 + math.log1p(max(weight, 0.0))


def make_npm_adapted_features(
    record: ReplayRecord,
) -> dict[str, torch.Tensor]:
    """Build the paper-aligned inputs available from one MPG replay row.

    New replay records may carry the true outgoing direction and roughness.
    Historical records carry neither.  For those rows we use the normalized
    receiver-to-emitter displacement as a non-target direction proxy and a
    neutral roughness value of 0.5.  Separate provenance flags make this
    approximation explicit.  The target ``initial_direction`` is never used
    as an input, preventing label leakage.
    """

    has_outgoing = record.condition.outgoing_direction is not None
    has_roughness = record.condition.roughness is not None
    outgoing_direction = (
        record.condition.outgoing_direction
        if has_outgoing
        else normalize_vector(record.condition.delta)
    )
    roughness = (
        float(record.condition.roughness)
        if has_roughness
        else NPM_ADAPTED_DEFAULT_ROUGHNESS
    )
    if not math.isfinite(roughness):
        raise ReplayDatasetError("NPM roughness must be finite")
    roughness = min(max(roughness, 0.0), 1.0)

    return {
        "npm_position": torch.tensor(record.condition.x_d, dtype=torch.float32),
        "npm_outgoing_direction": torch.tensor(
            outgoing_direction,
            dtype=torch.float32,
        ),
        "npm_normal": torch.tensor(record.condition.n_d, dtype=torch.float32),
        "npm_roughness": torch.tensor([roughness], dtype=torch.float32),
        "npm_direction_is_proxy": torch.tensor(
            0.0 if has_outgoing else 1.0,
            dtype=torch.float32,
        ),
        "npm_roughness_is_default": torch.tensor(
            0.0 if has_roughness else 1.0,
            dtype=torch.float32,
        ),
    }


class ReplayTrainingDataset(Dataset[dict[str, torch.Tensor]]):
    """Expose replay-buffer JSON lines as Torch tensors."""

    def __init__(
        self,
        replay_paths: tuple[Path, ...],
        *,
        max_records: int | None = None,
    ) -> None:
        if not replay_paths:
            raise ReplayDatasetError("replay_paths is empty")

        samples: list[dict[str, torch.Tensor]] = []
        group_keys: list[str] = []
        num_filtered_records = 0
        max_chain_length = 0

        for replay_path in replay_paths:
            if not replay_path.exists():
                raise ReplayDatasetError(f"missing replay file: {replay_path}")
            for record in read_replay_records_jsonl(replay_path):
                record_key = make_replay_record_key(record)
                chain_length = max(
                    0,
                    min(int(record.seed.chain_length), MAX_CHAIN_TYPE_BITS),
                )
                max_chain_length = max(max_chain_length, chain_length)
                if record.filtered:
                    num_filtered_records += 1

                samples.append(
                    {
                        "condition": torch.tensor(
                            flatten_condition(record),
                            dtype=torch.float32,
                        ),
                        "chain_length": torch.tensor(
                            chain_length,
                            dtype=torch.long,
                        ),
                        "chain_type_bits": torch.tensor(
                            pad_chain_type_bits(
                                record.seed.chain_type_bits,
                                chain_length=chain_length,
                            ),
                            dtype=torch.long,
                        ),
                        "chain_type_valid": torch.tensor(
                            make_chain_type_valid_mask(chain_length),
                            dtype=torch.float32,
                        ),
                        "direction": torch.tensor(
                            record.seed.initial_direction,
                            dtype=torch.float32,
                        ),
                        "sample_weight": torch.tensor(
                            normalize_sample_weight(record.weight),
                            dtype=torch.float32,
                        ),
                        # Retain the uncompressed contribution proxy so a
                        # training-only split can fit robust descriptor
                        # weights without reconstructing it from log space.
                        # ``sample_weight`` remains the historical
                        # ``1 + log1p(energy)`` value used by direction losses.
                        "raw_energy": torch.tensor(
                            max(float(record.weight), 0.0),
                            dtype=torch.float32,
                        ),
                        "filtered": torch.tensor(
                            1.0 if record.filtered else 0.0,
                            dtype=torch.float32,
                        ),
                        **make_npm_adapted_features(record),
                    }
                )
                group_keys.append(record_key)
                if max_records is not None and len(samples) >= max_records:
                    break
            if max_records is not None and len(samples) >= max_records:
                break

        if not samples:
            raise ReplayDatasetError("no replay samples loaded")

        mean_sample_weight = sum(
            float(sample["sample_weight"].item()) for sample in samples
        ) / len(samples)
        num_unique_records = len(set(group_keys))

        self._samples = tuple(samples)
        self.group_keys = tuple(group_keys)
        self.stats = ReplayDatasetStats(
            num_records=len(self._samples),
            num_unique_records=num_unique_records,
            num_duplicate_records=len(self._samples) - num_unique_records,
            num_source_files=len(replay_paths),
            num_filtered_records=num_filtered_records,
            mean_sample_weight=mean_sample_weight,
            max_chain_length=max_chain_length,
        )

    def __len__(self) -> int:
        """Dataset length in samples."""

        return len(self._samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """Return a cloned feature dict suitable for minibatch collation."""

        sample = self._samples[index]
        return {key: value.clone() for key, value in sample.items()}
