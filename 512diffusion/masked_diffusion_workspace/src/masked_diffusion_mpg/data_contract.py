from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from math import sqrt
from pathlib import Path
from typing import Iterable, Mapping


Vector3 = tuple[float, float, float]


class DataContractError(ValueError):
    """Raised when masked-diffusion replay JSON violates the schema."""


@dataclass(slots=True, frozen=True)
class RawSubpathSample:
    """Minimal faithful mirror of the C++ `SubpathSample`.

    Attributes:
        x_d: Separator vertex on the diffusion subpath endpoint.
        x_l: Separator vertex adjacent to the light subpath endpoint.
        x_1: First hit sampled along the initial seed ray from `x_d`.
        energy: Contribution weight recorded with the vertex event.
        bounce: Discrete chain depth for this reservoir entry.
        chain_type: Packed per-bounce categorical flags from Mitsuba.
        filtered: Indicates samples inserted by overlap-aware filtering passes.
    """

    x_d: Vector3
    x_l: Vector3
    x_1: Vector3
    energy: float
    bounce: int
    chain_type: int
    n_d: Vector3 = (0.0, 0.0, 0.0)
    n_l: Vector3 = (0.0, 0.0, 0.0)
    d_l: Vector3 = (0.0, 0.0, 0.0)
    filtered: bool = False
    # Optional NPM inputs.  Historical MPG replay ledgers do not contain these
    # fields, so keeping them nullable preserves the old on-disk contract and
    # lets new captures opt into the more faithful NPM conditioning path.
    outgoing_direction: Vector3 | None = None
    roughness: float | None = None


@dataclass(slots=True, frozen=True)
class ConditionFeatures:
    """Geometry-only conditioning latent derived without BSDF/category tags.

    We intentionally avoid normals-only BSDF payloads or categorical material IDs;
    downstream learning only sees reliably exported differential quantities.
    """

    x_d: Vector3
    x_l: Vector3
    delta: Vector3
    distance: float
    n_d: Vector3
    n_l: Vector3
    d_l: Vector3
    initial_direction: Vector3
    outgoing_direction: Vector3 | None = None
    roughness: float | None = None


@dataclass(slots=True, frozen=True)
class SeedDescriptor:
    """Masked-diffusion latent describing the stochastic seed choice.

    v0 materializes only bounce depth, unpacked chain-type bits, and the first
    hop direction—all directly recoverable from the native C++ buffer export.
    """

    chain_length: int
    chain_type_bits: tuple[int, ...]
    initial_direction: Vector3


@dataclass(slots=True, frozen=True)
class ReplayRecord:
    """Unified JSON-ready tuple persisted in the Mitsuba replay buffer ledger.

    Attributes:
        condition: Geometry bundle fed to Torch heads.
        seed: Targets predicted by the latent diffusion surrogate.
        weight: Effective importance multiplier for minibatch balancing.
        success: Known acceptance label when reservoirs record it; unknown rows keep `None`.
        filtered: True when the originating engine pass applied spatial pruning.
    """

    condition: ConditionFeatures
    seed: SeedDescriptor
    weight: float
    success: bool | None
    filtered: bool


def vector_sub(a: Vector3, b: Vector3) -> Vector3:
    """Component-wise subtraction for R^3 tuples."""

    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def vector_norm(v: Vector3) -> float:
    """Euclidean norm of an R^3 tuple."""

    return sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])


def normalize_vector(v: Vector3) -> Vector3:
    """Unit-vector projection with deterministic zero-vector fallback."""

    norm = vector_norm(v)
    if norm <= 1e-12:
        return (0.0, 0.0, 0.0)
    return (v[0] / norm, v[1] / norm, v[2] / norm)


def get_chaintype_bit(chain_type: int, index: int) -> int:
    """Match the bitwise contract of Mitsuba `get_chaintype_bit`."""

    return (chain_type >> index) & 1


def decode_chain_type_bits(chain_type: int, chain_length: int) -> tuple[int, ...]:
    """Unpack Mitsuba-packed chain-type bits along bounce progression."""

    if chain_length <= 0:
        return ()
    return tuple(get_chaintype_bit(chain_type, index) for index in range(chain_length))


def derive_condition_features(sample: RawSubpathSample) -> ConditionFeatures:
    """Hydrate Torch-friendly conditioning cues from exporter-native samples."""

    delta = vector_sub(sample.x_l, sample.x_d)
    initial_direction = normalize_vector(vector_sub(sample.x_1, sample.x_d))
    return ConditionFeatures(
        x_d=sample.x_d,
        x_l=sample.x_l,
        delta=delta,
        distance=vector_norm(delta),
        n_d=sample.n_d,
        n_l=sample.n_l,
        d_l=sample.d_l,
        initial_direction=initial_direction,
        outgoing_direction=sample.outgoing_direction,
        roughness=sample.roughness,
    )


def make_seed_descriptor(sample: RawSubpathSample) -> SeedDescriptor:
    """Produce the stochastic seed tuple referenced by masking heads."""

    condition = derive_condition_features(sample)
    return SeedDescriptor(
        chain_length=sample.bounce,
        chain_type_bits=decode_chain_type_bits(sample.chain_type, sample.bounce),
        initial_direction=condition.initial_direction,
    )


def make_replay_record(
    sample: RawSubpathSample,
    *,
    success: bool | None = None,
    weight: float | None = None,
) -> ReplayRecord:
    """Normalize exporter rows into Torch-ready replay artefacts.

    Args:
        sample: Native subpath reservoir row coming from Mitsuba tooling.
        success: Explicit Boolean label override; unresolved rows stay `None`.
        weight: Optional importance override; defaults to `sample.energy`.

    Returns:
        Canonical ReplayRecord suitable for persistence or JSON dumping.
    """

    condition = derive_condition_features(sample)
    seed = make_seed_descriptor(sample)
    return ReplayRecord(
        condition=condition,
        seed=seed,
        weight=sample.energy if weight is None else weight,
        success=success,
        filtered=sample.filtered,
    )


def vector_to_list(v: Vector3) -> list[float]:
    """Flatten R^3 tuples to JSON-compatible float lists."""

    return [float(v[0]), float(v[1]), float(v[2])]


def coerce_vector3(value: object, *, field_name: str) -> Vector3:
    """Strictly coerce JSON scalars/lists into validated R^3 tuples."""

    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise DataContractError(f"{field_name} must be a 3-element sequence")
    try:
        return (float(value[0]), float(value[1]), float(value[2]))
    except (TypeError, ValueError) as exc:
        raise DataContractError(f"{field_name} contains non-numeric values") from exc


def raw_subpath_sample_to_jsonable(sample: RawSubpathSample) -> dict[str, object]:
    """Marshal RawSubpathSample fields into deterministic JSON payloads."""

    payload: dict[str, object] = {
        "x_d": vector_to_list(sample.x_d),
        "x_l": vector_to_list(sample.x_l),
        "x_1": vector_to_list(sample.x_1),
        "n_d": vector_to_list(sample.n_d),
        "n_l": vector_to_list(sample.n_l),
        "d_l": vector_to_list(sample.d_l),
        "energy": float(sample.energy),
        "bounce": int(sample.bounce),
        "chain_type": int(sample.chain_type),
        "filtered": bool(sample.filtered),
    }
    if sample.outgoing_direction is not None:
        payload["outgoing_direction"] = vector_to_list(sample.outgoing_direction)
    if sample.roughness is not None:
        payload["roughness"] = float(sample.roughness)
    return payload


def raw_subpath_sample_from_jsonable(
    payload: Mapping[str, object],
) -> RawSubpathSample:
    """Hydrate exporter dicts emitted by Mitsuba-guided sampling binaries."""

    try:
        energy = float(payload["energy"])
        bounce = int(payload["bounce"])
        chain_type = int(payload["chain_type"])
    except KeyError as exc:
        raise DataContractError(f"missing required field: {exc.args[0]}") from exc
    except (TypeError, ValueError) as exc:
        raise DataContractError("invalid scalar field in raw subpath sample") from exc

    return RawSubpathSample(
        x_d=coerce_vector3(payload.get("x_d"), field_name="x_d"),
        x_l=coerce_vector3(payload.get("x_l"), field_name="x_l"),
        x_1=coerce_vector3(payload.get("x_1"), field_name="x_1"),
        n_d=coerce_vector3(payload.get("n_d", [0.0, 0.0, 0.0]), field_name="n_d"),
        n_l=coerce_vector3(payload.get("n_l", [0.0, 0.0, 0.0]), field_name="n_l"),
        d_l=coerce_vector3(payload.get("d_l", [0.0, 0.0, 0.0]), field_name="d_l"),
        energy=energy,
        bounce=bounce,
        chain_type=chain_type,
        filtered=bool(payload.get("filtered", False)),
        outgoing_direction=(
            None
            if payload.get("outgoing_direction") is None
            else coerce_vector3(
                payload.get("outgoing_direction"),
                field_name="outgoing_direction",
            )
        ),
        roughness=(
            None if payload.get("roughness") is None else float(payload["roughness"])
        ),
    )


def replay_record_to_jsonable(record: ReplayRecord) -> dict[str, object]:
    """Flatten ReplayRecord into JSON-friendly associative maps."""

    condition_payload: dict[str, object] = {
        "x_d": vector_to_list(record.condition.x_d),
        "x_l": vector_to_list(record.condition.x_l),
        "delta": vector_to_list(record.condition.delta),
        "distance": float(record.condition.distance),
        "n_d": vector_to_list(record.condition.n_d),
        "n_l": vector_to_list(record.condition.n_l),
        "d_l": vector_to_list(record.condition.d_l),
        "initial_direction": vector_to_list(record.condition.initial_direction),
    }
    if record.condition.outgoing_direction is not None:
        condition_payload["outgoing_direction"] = vector_to_list(
            record.condition.outgoing_direction
        )
    if record.condition.roughness is not None:
        condition_payload["roughness"] = float(record.condition.roughness)

    return {
        "condition": condition_payload,
        "seed": {
            "chain_length": int(record.seed.chain_length),
            "chain_type_bits": [int(bit) for bit in record.seed.chain_type_bits],
            "initial_direction": vector_to_list(record.seed.initial_direction),
        },
        "weight": float(record.weight),
        "success": record.success,
        "filtered": bool(record.filtered),
    }


def replay_record_from_jsonable(payload: Mapping[str, object]) -> ReplayRecord:
    """Rebuild ReplayRecord objects from validated JSON payloads."""

    try:
        condition_payload = payload["condition"]
        seed_payload = payload["seed"]
        weight = float(payload["weight"])
    except KeyError as exc:
        raise DataContractError(f"missing required field: {exc.args[0]}") from exc
    except (TypeError, ValueError) as exc:
        raise DataContractError("invalid scalar field in replay record") from exc

    if not isinstance(condition_payload, Mapping):
        raise DataContractError("condition must be an object")
    if not isinstance(seed_payload, Mapping):
        raise DataContractError("seed must be an object")

    try:
        chain_length = int(seed_payload["chain_length"])
        chain_type_bits_raw = seed_payload["chain_type_bits"]
    except KeyError as exc:
        raise DataContractError(f"missing seed field: {exc.args[0]}") from exc
    except (TypeError, ValueError) as exc:
        raise DataContractError("invalid seed scalar field") from exc

    if not isinstance(chain_type_bits_raw, (list, tuple)):
        raise DataContractError("chain_type_bits must be a sequence")

    try:
        chain_type_bits = tuple(int(bit) for bit in chain_type_bits_raw)
    except (TypeError, ValueError) as exc:
        raise DataContractError("chain_type_bits contains non-integer values") from exc

    success_value = payload.get("success")
    if success_value is not None and not isinstance(success_value, bool):
        raise DataContractError("success must be bool or null")

    return ReplayRecord(
        condition=ConditionFeatures(
            x_d=coerce_vector3(
                condition_payload.get("x_d"), field_name="condition.x_d"
            ),
            x_l=coerce_vector3(
                condition_payload.get("x_l"), field_name="condition.x_l"
            ),
            delta=coerce_vector3(
                condition_payload.get("delta"), field_name="condition.delta"
            ),
            distance=float(condition_payload.get("distance")),
            n_d=coerce_vector3(
                condition_payload.get("n_d", [0.0, 0.0, 0.0]),
                field_name="condition.n_d",
            ),
            n_l=coerce_vector3(
                condition_payload.get("n_l", [0.0, 0.0, 0.0]),
                field_name="condition.n_l",
            ),
            d_l=coerce_vector3(
                condition_payload.get("d_l", [0.0, 0.0, 0.0]),
                field_name="condition.d_l",
            ),
            initial_direction=coerce_vector3(
                condition_payload.get("initial_direction"),
                field_name="condition.initial_direction",
            ),
            outgoing_direction=(
                None
                if condition_payload.get("outgoing_direction") is None
                else coerce_vector3(
                    condition_payload.get("outgoing_direction"),
                    field_name="condition.outgoing_direction",
                )
            ),
            roughness=(
                None
                if condition_payload.get("roughness") is None
                else float(condition_payload["roughness"])
            ),
        ),
        seed=SeedDescriptor(
            chain_length=chain_length,
            chain_type_bits=chain_type_bits,
            initial_direction=coerce_vector3(
                seed_payload.get("initial_direction"),
                field_name="seed.initial_direction",
            ),
        ),
        weight=weight,
        success=success_value,
        filtered=bool(payload.get("filtered", False)),
    )


def make_raw_subpath_sample_key(sample: RawSubpathSample) -> str:
    """Deterministic fingerprint for exporter-native samples."""

    payload = raw_subpath_sample_to_jsonable(sample)
    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha1(serialized.encode("utf-8")).hexdigest()


def make_replay_record_key(record: ReplayRecord) -> str:
    """Deterministic fingerprint for replay-buffer rows."""

    payload = replay_record_to_jsonable(record)
    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha1(serialized.encode("utf-8")).hexdigest()


def write_jsonl(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    """Persist sorted-key JSON blobs as newline-delimited rows."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def read_raw_subpath_samples_jsonl(path: Path) -> tuple[RawSubpathSample, ...]:
    """Stream exporter JSONL blobs into immutable RawSubpathSample tuples."""

    samples: list[RawSubpathSample] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise DataContractError(
                    f"invalid JSON at line {line_number} in {path}"
                ) from exc
            if not isinstance(payload, dict):
                raise DataContractError(
                    f"line {line_number} in {path} is not a JSON object"
                )
            samples.append(raw_subpath_sample_from_jsonable(payload))
    return tuple(samples)


def read_replay_records_jsonl(path: Path) -> tuple[ReplayRecord, ...]:
    """Slurp validated replay ledger JSONLines into ReplayRecord structs."""

    records: list[ReplayRecord] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise DataContractError(
                    f"invalid JSON at line {line_number} in {path}"
                ) from exc
            if not isinstance(payload, dict):
                raise DataContractError(
                    f"line {line_number} in {path} is not a JSON object"
                )
            records.append(replay_record_from_jsonable(payload))
    return tuple(records)


def write_replay_records_jsonl(path: Path, records: Iterable[ReplayRecord]) -> None:
    """Dump ReplayRecord iterators back to canonical JSONLines."""

    write_jsonl(path, (replay_record_to_jsonable(record) for record in records))
