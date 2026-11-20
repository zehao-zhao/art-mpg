from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = WORKSPACE_ROOT / "configs" / "fresh_confirmatory_models_20260901"
MANIFEST_PATH = OUTPUT_DIR / "frozen_manifest.json"
PROTOCOL_ID = "fresh_confirmatory_models_20260901_v1"
SEEDS = (104729, 130363, 155921)
CODE_PATHS = (
    "scripts/generate_fresh_confirmatory_model_configs.py",
    "scripts/run_fresh_confirmatory_model_artifacts.py",
    "scripts/phase_10_train_masked_diffusion.py",
    "scripts/phase_11_compare_guidance_models.py",
    "scripts/phase_12_export_runtime_model.py",
    "src/masked_diffusion_mpg/__init__.py",
    "src/masked_diffusion_mpg/data_contract.py",
    "src/masked_diffusion_mpg/dataset.py",
    "src/masked_diffusion_mpg/model.py",
    "src/masked_diffusion_mpg/training.py",
    "src/masked_diffusion_mpg/comparison.py",
    "src/masked_diffusion_mpg/runtime_export.py",
)

SCENES: tuple[dict[str, Any], ...] = (
    {
        "key": "plane",
        "replay_path": "results/phase_09_export/replay/paper_plane_mpg.jsonl",
        "replay_sha256": "442919f705d14013a8b5df88a686ea1d10433fa3974d353490da4cc35cb35dd3",
        "records": 11127,
        "train_records": 8345,
        "validation_records": 2782,
        "split_notes": "All rows are L1:0.",
    },
    {
        "key": "stone",
        "replay_path": "results/phase_09_export/replay/paper_stone_mpg.jsonl",
        "replay_sha256": "11210cb96edda08746dcf731bad4fcb1f4dc5bf9d248b91758b081666ef2cade",
        "records": 3528,
        "train_records": 2646,
        "validation_records": 882,
        "split_notes": (
            "Both L1:1 and L2:11 occur in train and validation for every frozen seed."
        ),
    },
    {
        "key": "sphere",
        "replay_path": "results/phase_09_export/replay/paper_sphere_mpg.jsonl",
        "replay_sha256": "9b9661173d9e75a59bc6335b98b244756787740f657681f68d3c9a1e4450ea67",
        "records": 344,
        "train_records": 258,
        "validation_records": 86,
        "split_notes": "All rows are L1:1.",
    },
    {
        "key": "pool",
        "replay_path": "results/phase_09_export/replay/paper_pool_mpg.jsonl",
        "replay_sha256": "c2771ff6219c2282aeae99847d4c46f08a8721e029a8074b2784ce842c47bdaf",
        "records": 57,
        "train_records": 43,
        "validation_records": 14,
        "split_notes": (
            "For every frozen seed, train has 42 L1:1 plus the sole L1:0 row; "
            "validation has 14 L1:1 and therefore no rare L1:0 row."
        ),
    },
    {
        "key": "living",
        "replay_path": "results/phase_09_export/replay/paper_living_mpg.jsonl",
        "replay_sha256": "ad353ba8e4edc6daf7a32c00403f4ea52893a15757bb8f91d3d4fe3a6740d1c0",
        "records": 27,
        "train_records": 20,
        "validation_records": 7,
        "split_notes": (
            "For every frozen seed, train has 7 L2:11 and 13 L4:1111; "
            "validation has 3 L2:11 and 4 L4:1111."
        ),
    },
)

ORDERED_PARAMETER_COUNT = 55_599
SIMPLE_PARAMETER_COUNT = 55_857
BATCH_SIZE = 16
EPOCHS = 32

# Measured on the development scenes with the same selected recipes.
ORDERED_SECONDS_PER_STEP = (0.0404828932, 0.1162838111)
SIMPLE_SECONDS_PER_STEP = (0.00936366865, 0.00995707314)


class FreshConfirmatoryConfigError(RuntimeError):
    """Raised when the frozen inputs or generated package drift."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Emit frozen fresh-scene Phase 10/11 training and Phase 12 "
            "export configs. This script never trains or exports a model."
        )
    )
    parser.add_argument(
        "--emit",
        action="store_true",
        help="Write the 60 configs and frozen manifest after all replay checks pass.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace a generated file only when its bytes differ.",
    )
    return parser.parse_args()


def json_bytes(payload: object) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative(path: Path) -> str:
    return path.relative_to(WORKSPACE_ROOT).as_posix()


def ordered_config(scene: dict[str, Any], seed: int, output_dir: str) -> dict[str, Any]:
    """Build the selected ordered h72/d3/K4 recipe with effective defaults explicit."""

    return {
        "protocol_id": PROTOCOL_ID,
        "scene_key": scene["key"],
        "replay_sha256": scene["replay_sha256"],
        "replay_records": scene["records"],
        "expected_parameter_count": ORDERED_PARAMETER_COUNT,
        "replay_paths": [scene["replay_path"]],
        "output_dir": output_dir,
        "seed": seed,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "val_fraction": 0.25,
        "max_records": None,
        "num_workers": 0,
        "device": "auto",
        "early_stop_patience": None,
        "grad_clip_norm": 0.5,
        "direction_loss_weight": 1.0,
        "direction_loss_type": "vmf_mixture",
        "direction_vmf_kappa": 1.0,
        "descriptor_weight_mode": "log_energy",
        "descriptor_weight_quantile": 0.99,
        "direction_weight_mode": "log_energy",
        "direction_weight_quantile": 0.99,
        "direction_distribution": "vmf_mixture",
        "vmf_components": 4,
        "vmf_max_kappa": 64.0,
        "max_nonfinite_batches": 4,
        "direction_space": "s2",
        "full_mask_probability": 0.25,
        "stochastic_diffusion": False,
        "diffusion_objective": "predict_x0",
        "diffusion_train_steps": 16,
        "beta_schedule": "linear",
        "beta_start": 0.0005,
        "beta_end": 0.05,
        "reverse_noise_scale": 1.0,
        "backbone_type": "mlp",
        "hidden_dim": 72,
        "depth": 3,
        "dropout": 0.1,
        "time_embedding_dim": 16,
        "token_embedding_dim": 24,
        "transformer_heads": 4,
        "transformer_feedforward_dim": None,
        "discrete_diffusion_mode": "ordered_absorbing",
    }


def simple_config(
    scene: dict[str, Any], seed: int, ordered_checkpoint: str, output_dir: str
) -> dict[str, Any]:
    """Build the fair simple h126/d4/K4 recipe consumed by Phase 11."""

    return {
        "protocol_id": PROTOCOL_ID,
        "scene_key": scene["key"],
        "replay_sha256": scene["replay_sha256"],
        "replay_records": scene["records"],
        "expected_parameter_count": SIMPLE_PARAMETER_COUNT,
        "replay_paths": [scene["replay_path"]],
        "masked_diffusion_checkpoint": ordered_checkpoint,
        "output_dir": output_dir,
        "seed": seed,
        "batch_size": BATCH_SIZE,
        "val_fraction": 0.25,
        "max_records": None,
        "num_workers": 0,
        "device": "auto",
        "direction_loss_weight": 1.0,
        "direction_loss_type": "vmf_mixture",
        "direction_vmf_kappa": 1.0,
        "diffusion_steps": 1,
        "stochastic_diffusion": False,
        "diffusion_train_steps": 1,
        "beta_schedule": "linear",
        "beta_start": 0.0005,
        "beta_end": 0.05,
        "reverse_noise_scale": 1.0,
        "simple_baseline": {
            "hidden_dim": 126,
            "depth": 4,
            "dropout": 0.1,
            "epochs": EPOCHS,
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
            "grad_clip_norm": 0.5,
            "probabilistic_direction": True,
            "vmf_components": 4,
            "min_kappa": 0.001,
            "max_kappa": 64.0,
        },
    }


def export_config(
    *, scene_key: str, model: str, checkpoint: str, output_dir: str
) -> dict[str, Any]:
    if model == "ordered":
        model_kind = "masked_diffusion"
        output_name = "ordered_diffusion_mlp_h72_k4_runtime.txt"
        count = ORDERED_PARAMETER_COUNT
    else:
        model_kind = "simple_mlp"
        output_name = "simple_mlp_k4_runtime.txt"
        count = SIMPLE_PARAMETER_COUNT
    return {
        "protocol_id": PROTOCOL_ID,
        "scene_key": scene_key,
        "model": model,
        "expected_parameter_count": count,
        "checkpoint_path": checkpoint,
        "output_dir": output_dir,
        "output_name": output_name,
        "model_kind": model_kind,
    }


def add_config(
    files: dict[Path, bytes], filename: str, payload: dict[str, Any]
) -> tuple[str, str]:
    path = OUTPUT_DIR / filename
    content = json_bytes(payload)
    files[path] = content
    return relative(path), sha256_bytes(content)


def build_package() -> tuple[dict[Path, bytes], dict[str, Any]]:
    files: dict[Path, bytes] = {}
    jobs: list[dict[str, Any]] = []
    scene_entries: list[dict[str, Any]] = []

    for scene in SCENES:
        seed_entries: list[dict[str, Any]] = []
        steps = math.ceil(scene["train_records"] / BATCH_SIZE) * EPOCHS
        for seed in SEEDS:
            stem = f"fresh_{scene['key']}_seed{seed}"
            result_root = (
                f"results/fresh_confirmatory_models_20260901/{scene['key']}/seed{seed}"
            )
            phase10_dir = f"{result_root}/phase10/ordered_h72_k4"
            phase11_dir = f"{result_root}/phase11/simple_h126_k4"
            ordered_export_dir = f"{result_root}/phase12/ordered_h72_k4"
            simple_export_dir = f"{result_root}/phase12/simple_h126_k4"
            ordered_checkpoint = f"{phase10_dir}/best_general.pt"
            simple_checkpoint = f"{phase11_dir}/best_simple_mlp.pt"

            p10_path, p10_sha = add_config(
                files,
                f"phase_10_{stem}_ordered_h72_k4_e32_macos.json",
                ordered_config(scene, seed, phase10_dir),
            )
            p11_path, p11_sha = add_config(
                files,
                f"phase_11_{stem}_simple_h126_k4_e32_macos.json",
                simple_config(scene, seed, ordered_checkpoint, phase11_dir),
            )
            p12_ordered_path, p12_ordered_sha = add_config(
                files,
                f"phase_12_{stem}_ordered_h72_k4_macos.json",
                export_config(
                    scene_key=scene["key"],
                    model="ordered",
                    checkpoint=ordered_checkpoint,
                    output_dir=ordered_export_dir,
                ),
            )
            p12_simple_path, p12_simple_sha = add_config(
                files,
                f"phase_12_{stem}_simple_h126_k4_macos.json",
                export_config(
                    scene_key=scene["key"],
                    model="simple",
                    checkpoint=simple_checkpoint,
                    output_dir=simple_export_dir,
                ),
            )

            job_specs = (
                (
                    f"{scene['key']}_seed{seed}_phase10_ordered",
                    "phase10_ordered_training",
                    "scripts/phase_10_train_masked_diffusion.py",
                    p10_path,
                    p10_sha,
                    [],
                    [ordered_checkpoint],
                ),
                (
                    f"{scene['key']}_seed{seed}_phase11_simple",
                    "phase11_simple_training",
                    "scripts/phase_11_compare_guidance_models.py",
                    p11_path,
                    p11_sha,
                    [f"{scene['key']}_seed{seed}_phase10_ordered"],
                    [simple_checkpoint],
                ),
                (
                    f"{scene['key']}_seed{seed}_phase12_ordered",
                    "phase12_ordered_export",
                    "scripts/phase_12_export_runtime_model.py",
                    p12_ordered_path,
                    p12_ordered_sha,
                    [f"{scene['key']}_seed{seed}_phase10_ordered"],
                    [
                        f"{ordered_export_dir}/ordered_diffusion_mlp_h72_k4_runtime.txt",
                        f"{ordered_export_dir}/runtime_export_summary.json",
                    ],
                ),
                (
                    f"{scene['key']}_seed{seed}_phase12_simple",
                    "phase12_simple_export",
                    "scripts/phase_12_export_runtime_model.py",
                    p12_simple_path,
                    p12_simple_sha,
                    [f"{scene['key']}_seed{seed}_phase11_simple"],
                    [
                        f"{simple_export_dir}/simple_mlp_k4_runtime.txt",
                        f"{simple_export_dir}/runtime_export_summary.json",
                    ],
                ),
            )
            for job_id, kind, script, config_path, config_sha, dependencies, outputs in job_specs:
                jobs.append(
                    {
                        "job_id": job_id,
                        "kind": kind,
                        "scene": scene["key"],
                        "seed": seed,
                        "script": script,
                        "script_sha256": sha256_file(WORKSPACE_ROOT / script),
                        "config_path": config_path,
                        "config_sha256": config_sha,
                        "dependencies": dependencies,
                        "expected_outputs": outputs,
                        "initial_status": "pending",
                    }
                )

            seed_entries.append(
                {
                    "seed": seed,
                    "split": {
                        "train_records": scene["train_records"],
                        "validation_records": scene["validation_records"],
                        "train_batches_per_epoch": math.ceil(
                            scene["train_records"] / BATCH_SIZE
                        ),
                        "validation_batches": math.ceil(
                            scene["validation_records"] / BATCH_SIZE
                        ),
                    },
                    "optimizer_steps_per_model": steps,
                    "configs": {
                        "phase10_ordered": p10_path,
                        "phase11_simple": p11_path,
                        "phase12_ordered": p12_ordered_path,
                        "phase12_simple": p12_simple_path,
                    },
                }
            )
        scene_entries.append(
            {
                "scene": scene["key"],
                "replay_path": scene["replay_path"],
                "replay_sha256": scene["replay_sha256"],
                "replay_records": scene["records"],
                "split_notes": scene["split_notes"],
                "seeds": seed_entries,
            }
        )

    ordered_steps = sum(
        math.ceil(scene["train_records"] / BATCH_SIZE) * EPOCHS * len(SEEDS)
        for scene in SCENES
    )
    simple_steps = ordered_steps
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "frozen_date": "2026-09-01",
        "generation_state": "prepared_only_not_executed",
        "purpose": (
            "Frozen, scene-independent confirmatory training/export package for five "
            "fresh scenes. Generation performs no training, export, or rendering."
        ),
        "generator": "scripts/generate_fresh_confirmatory_model_configs.py",
        "orchestrator": "scripts/run_fresh_confirmatory_model_artifacts.py",
        "entry_points": {
            "ordered_training": "scripts/phase_10_train_masked_diffusion.py",
            "simple_training": "scripts/phase_11_compare_guidance_models.py",
            "runtime_export": "scripts/phase_12_export_runtime_model.py",
        },
        "code_provenance": {
            path: sha256_file(WORKSPACE_ROOT / path) for path in CODE_PATHS
        },
        "scope": {
            "scenes": [scene["key"] for scene in SCENES],
            "training_seeds": list(SEEDS),
            "training_configs": 30,
            "phase10_ordered_configs": 15,
            "phase11_simple_configs": 15,
            "phase12_export_configs": 30,
            "jobs": 60,
            "note": (
                "The repository's fair simple trainer is Phase 11; Phase 10 only "
                "constructs MaskedSeedDiffusionModel."
            ),
        },
        "frozen_recipe": {
            "common": {
                "batch_size": BATCH_SIZE,
                "epochs": EPOCHS,
                "learning_rate": 0.001,
                "weight_decay": 0.0001,
                "val_fraction": 0.25,
                "max_records": None,
                "num_workers": 0,
                "grad_clip_norm": 0.5,
                "loss_weighting": "1 + log1p(max(replay_weight, 0))",
            },
            "ordered": {
                "backbone": "mlp",
                "hidden_dim": 72,
                "depth": 3,
                "token_embedding_dim": 24,
                "time_embedding_dim": 16,
                "dropout": 0.1,
                "vmf_components": 4,
                "epochs": EPOCHS,
                "discrete_diffusion_mode": "ordered_absorbing",
                "direction_space": "s2",
            },
            "simple": {
                "backbone": "plain_mlp",
                "hidden_dim": 126,
                "depth": 4,
                "dropout": 0.1,
                "probabilistic_direction": True,
                "vmf_components": 4,
                "epochs": EPOCHS,
            },
        },
        "parameter_budget": {
            "ordered_parameters": ORDERED_PARAMETER_COUNT,
            "simple_parameters": SIMPLE_PARAMETER_COUNT,
            "absolute_difference": SIMPLE_PARAMETER_COUNT - ORDERED_PARAMETER_COUNT,
            "simple_over_ordered_ratio": SIMPLE_PARAMETER_COUNT
            / ORDERED_PARAMETER_COUNT,
            "relative_difference_percent": 100.0
            * (SIMPLE_PARAMETER_COUNT - ORDERED_PARAMETER_COUNT)
            / ORDERED_PARAMETER_COUNT,
            "maximum_relative_difference_percent": 1.0,
            "passes": (
                abs(SIMPLE_PARAMETER_COUNT - ORDERED_PARAMETER_COUNT)
                / ORDERED_PARAMETER_COUNT
                <= 0.01
            ),
            "evidence": [
                "results/ordered_absorbing_sweep_macos_20260901/mlp_h72_k4_e32/run_20260901_185417/run_summary.json",
                "results/ordered_absorbing_slabs_simple_k4_macos_20260901/phase11/simple_h126_k4_e32/run_20260901_193650/comparison.json",
            ],
        },
        "estimated_training_cost": {
            "ordered_optimizer_steps": ordered_steps,
            "simple_optimizer_steps": simple_steps,
            "total_optimizer_steps": ordered_steps + simple_steps,
            "ordered_seconds_per_step_observed_range": list(
                ORDERED_SECONDS_PER_STEP
            ),
            "simple_seconds_per_step_observed_range": list(SIMPLE_SECONDS_PER_STEP),
            "ordered_minutes_range": [
                ordered_steps * ORDERED_SECONDS_PER_STEP[0] / 60.0,
                ordered_steps * ORDERED_SECONDS_PER_STEP[1] / 60.0,
            ],
            "simple_minutes_range": [
                simple_steps * SIMPLE_SECONDS_PER_STEP[0] / 60.0,
                simple_steps * SIMPLE_SECONDS_PER_STEP[1] / 60.0,
            ],
            "total_minutes_range": [
                ordered_steps * ORDERED_SECONDS_PER_STEP[0] / 60.0
                + simple_steps * SIMPLE_SECONDS_PER_STEP[0] / 60.0,
                ordered_steps * ORDERED_SECONDS_PER_STEP[1] / 60.0
                + simple_steps * SIMPLE_SECONDS_PER_STEP[1] / 60.0,
            ],
            "planning_envelope": "1.2-3.0 hours including serial orchestration/export overhead",
        },
        "scenes": scene_entries,
        "jobs": jobs,
    }
    return files, manifest


def validate_inputs() -> None:
    for scene in SCENES:
        path = WORKSPACE_ROOT / scene["replay_path"]
        if not path.is_file():
            raise FreshConfirmatoryConfigError(f"missing replay: {path}")
        actual_sha = sha256_file(path)
        if actual_sha != scene["replay_sha256"]:
            raise FreshConfirmatoryConfigError(
                f"replay hash mismatch for {scene['key']}: {actual_sha}"
            )
        with path.open("rb") as handle:
            records = sum(1 for _ in handle)
        if records != scene["records"]:
            raise FreshConfirmatoryConfigError(
                f"replay row count mismatch for {scene['key']}: {records}"
            )


def write_generated(path: Path, content: bytes, *, force: bool) -> None:
    if path.exists():
        existing = path.read_bytes()
        if existing == content:
            return
        if not force:
            raise FreshConfirmatoryConfigError(
                f"refusing to replace changed generated file without --force: {path}"
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def validate_package(files: dict[Path, bytes], manifest: dict[str, Any]) -> None:
    if len(files) != 60 or len(manifest["jobs"]) != 60:
        raise FreshConfirmatoryConfigError("expected exactly 60 configs and jobs")
    kinds: dict[str, int] = {}
    for job in manifest["jobs"]:
        kinds[job["kind"]] = kinds.get(job["kind"], 0) + 1
        path = WORKSPACE_ROOT / job["config_path"]
        content = files.get(path)
        if content is None or sha256_bytes(content) != job["config_sha256"]:
            raise FreshConfirmatoryConfigError(
                f"internal config hash mismatch: {job['job_id']}"
            )
        json.loads(content)
    expected = {
        "phase10_ordered_training": 15,
        "phase11_simple_training": 15,
        "phase12_ordered_export": 15,
        "phase12_simple_export": 15,
    }
    if kinds != expected:
        raise FreshConfirmatoryConfigError(f"unexpected job counts: {kinds}")
    if not manifest["parameter_budget"]["passes"]:
        raise FreshConfirmatoryConfigError("parameter budget exceeds 1 percent")


def validate_existing_package(files: dict[Path, bytes], manifest_content: bytes) -> None:
    """When a package exists, make check-only verify its exact frozen bytes."""

    if not OUTPUT_DIR.exists():
        return
    expected = dict(files)
    expected[MANIFEST_PATH] = manifest_content
    existing = set(OUTPUT_DIR.glob("*.json"))
    if existing != set(expected):
        missing = sorted(path.name for path in set(expected) - existing)
        extra = sorted(path.name for path in existing - set(expected))
        raise FreshConfirmatoryConfigError(
            f"generated package file set drift: missing={missing}, extra={extra}"
        )
    for path, content in expected.items():
        if path.read_bytes() != content:
            raise FreshConfirmatoryConfigError(
                f"generated package content drift: {path}; review and use --emit --force"
            )


def main() -> int:
    args = parse_args()
    validate_inputs()
    files, manifest = build_package()
    validate_package(files, manifest)
    manifest_content = json_bytes(manifest)

    if args.emit:
        for path in sorted(files):
            write_generated(path, files[path], force=args.force)
        write_generated(MANIFEST_PATH, manifest_content, force=args.force)
    else:
        validate_existing_package(files, manifest_content)

    action = "validated and emitted" if args.emit else "validated (check-only)"
    print(f"{action}: protocol={PROTOCOL_ID}")
    print("training_configs=30 (phase10_ordered=15, phase11_simple=15)")
    print("phase12_export_configs=30 jobs=60")
    print(
        "parameter_budget="
        f"{SIMPLE_PARAMETER_COUNT}/{ORDERED_PARAMETER_COUNT} "
        f"({manifest['parameter_budget']['relative_difference_percent']:.3f}% delta, pass)"
    )
    low, high = manifest["estimated_training_cost"]["total_minutes_range"]
    print(f"estimated_training_minutes={low:.2f}-{high:.2f}")
    print(f"manifest={relative(MANIFEST_PATH)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
