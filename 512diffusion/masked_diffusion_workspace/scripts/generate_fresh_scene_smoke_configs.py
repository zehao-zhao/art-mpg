from __future__ import annotations

import argparse
import hashlib
import json
import sys
import xml.etree.ElementTree as et
from pathlib import Path
from typing import Any


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = WORKSPACE_ROOT.parent
DEFAULT_MANIFEST = (
    WORKSPACE_ROOT / "configs" / "fresh_scene_smoke_prerequisites_20260901.json"
)
EXPECTED_SCENE_KEYS = ("plane", "stone", "sphere", "pool", "living")
EXPECTED_SMOKE_VALUES = {
    "width": 16,
    "height": 16,
    "crop_width": 16,
    "crop_height": 16,
    "reference_sample_count": 1,
    "compare_sample_count": 1,
    "samples_per_pass": 1,
}
EXPECTED_BUNDLES = {
    "baseline": {
        "methods": ["path_reference", "mpg"],
        "bounce_alpha": 0.5,
        "learned_direction_kappa_scale": 1.0,
        "learned_direction_flip_to_receiver_hemisphere": False,
    },
    "ordered": {
        "methods": ["path_reference", "ordered_diffusion_mlp_h72_k4"],
        "bounce_alpha": 0.0,
        "learned_direction_kappa_scale": 1.0,
        "learned_direction_flip_to_receiver_hemisphere": False,
    },
    "simple": {
        "methods": ["path_reference", "simple_mlp_k4"],
        "bounce_alpha": 0.5,
        "learned_direction_kappa_scale": 0.25,
        "learned_direction_flip_to_receiver_hemisphere": False,
    },
}


class FreshSceneSmokeConfigError(RuntimeError):
    """Raised when a prerequisite or selected runtime identity is invalid."""


def parse_args() -> argparse.Namespace:
    """Parse a safe-by-default config-generation command line."""

    parser = argparse.ArgumentParser(
        description=(
            "Check fresh-scene smoke prerequisites and, only when all selected "
            "artifacts are registered, emit Phase 13 configs. This script never "
            "trains a model or launches Mitsuba."
        )
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--emit-ready",
        action="store_true",
        help="Write all 15 Phase 13 configs after every scene passes validation.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Allow --emit-ready to replace existing generated config files.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise FreshSceneSmokeConfigError(f"expected JSON object: {path}")
    return payload


def workspace_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path.resolve()
    return (WORKSPACE_ROOT / path).resolve()


def project_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path.resolve()
    return (PROJECT_ROOT / path).resolve()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def runtime_header(path: Path) -> tuple[str, dict[str, str]]:
    """Read only the small text header preceding the first tensor payload."""

    values: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as handle:
        signature = handle.readline().strip()
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("tensor "):
                break
            key, separator, value = stripped.partition(" ")
            if separator:
                values[key] = value
    return signature, values


def values_match(actual: object, expected: object) -> bool:
    if isinstance(expected, bool):
        return actual is expected
    if isinstance(expected, int):
        try:
            return int(actual) == expected
        except (TypeError, ValueError):
            return False
    if isinstance(expected, float):
        try:
            return float(actual) == expected
        except (TypeError, ValueError):
            return False
    return str(actual) == str(expected)


def validate_frozen_protocol(manifest: dict[str, Any]) -> None:
    """Reject accidental drift from the preregistered smoke controls."""

    smoke = manifest.get("smoke")
    if not isinstance(smoke, dict):
        raise FreshSceneSmokeConfigError("manifest smoke block is missing")
    for key, expected in EXPECTED_SMOKE_VALUES.items():
        if not values_match(smoke.get(key), expected):
            raise FreshSceneSmokeConfigError(
                f"smoke {key}={smoke.get(key)!r}, expected {expected!r}"
            )

    replacement = manifest.get("integrator_replacement")
    if not isinstance(replacement, dict) or replacement.get("required") is not True:
        raise FreshSceneSmokeConfigError(
            "Phase 13 integrator replacement must remain explicit and required"
        )

    bundle_root = manifest.get("bundle_layout")
    bundles = [] if not isinstance(bundle_root, dict) else bundle_root.get(
        "configs_per_ready_scene", []
    )
    observed = {
        str(bundle.get("key")): {
            key: bundle.get(key) for key in expected
        }
        for bundle in bundles
        if isinstance(bundle, dict)
        for expected in [EXPECTED_BUNDLES.get(str(bundle.get("key")), {})]
    }
    if observed != EXPECTED_BUNDLES:
        raise FreshSceneSmokeConfigError(
            "bundle methods or selected alpha/kappa/no-flip controls changed"
        )

    scene_keys = tuple(
        str(scene.get("scene_key"))
        for scene in manifest.get("scenes", [])
        if isinstance(scene, dict)
    )
    if scene_keys != EXPECTED_SCENE_KEYS:
        raise FreshSceneSmokeConfigError(
            f"scene order/set={scene_keys!r}, expected {EXPECTED_SCENE_KEYS!r}"
        )


def validate_hashed_path(
    *,
    label: str,
    path_text: object,
    expected_sha256: object,
    resolver: Any,
) -> tuple[Path | None, list[str]]:
    errors: list[str] = []
    if not isinstance(path_text, str) or not path_text:
        return None, [f"{label}_path is unset"]
    if not isinstance(expected_sha256, str) or len(expected_sha256) != 64:
        errors.append(f"{label}_sha256 is unset or malformed")
    path = resolver(path_text)
    if not path.is_file():
        errors.append(f"{label} does not exist: {path_text}")
        return path, errors
    if isinstance(expected_sha256, str) and len(expected_sha256) == 64:
        actual_sha256 = sha256_file(path)
        if actual_sha256 != expected_sha256:
            errors.append(
                f"{label} SHA-256 mismatch: expected {expected_sha256}, "
                f"found {actual_sha256}"
            )
    return path, errors


def validate_selected_artifact(
    *,
    scene: dict[str, Any],
    method: str,
    contract: dict[str, Any],
    required_seed: int,
) -> tuple[Path | None, list[str]]:
    errors: list[str] = []
    artifacts = scene.get("selected_artifacts", {}).get(method)
    if not isinstance(artifacts, dict):
        return None, [f"{method} artifact registration is missing"]

    checkpoint, checkpoint_errors = validate_hashed_path(
        label=f"{method} checkpoint",
        path_text=artifacts.get("checkpoint_path"),
        expected_sha256=artifacts.get("checkpoint_sha256"),
        resolver=workspace_path,
    )
    runtime, runtime_errors = validate_hashed_path(
        label=f"{method} runtime",
        path_text=artifacts.get("runtime_path"),
        expected_sha256=artifacts.get("runtime_sha256"),
        resolver=workspace_path,
    )
    errors.extend(checkpoint_errors)
    errors.extend(runtime_errors)
    for label, path in (("checkpoint", checkpoint), ("runtime", runtime)):
        if path is None:
            continue
        try:
            path.relative_to(WORKSPACE_ROOT)
        except ValueError:
            errors.append(f"{method} {label} is outside the workspace: {path}")

    summary_text = artifacts.get("runtime_export_summary_path")
    if not isinstance(summary_text, str) or not summary_text:
        errors.append(f"{method} runtime_export_summary_path is unset")
        summary_path = None
    else:
        summary_path = workspace_path(summary_text)
        if not summary_path.is_file():
            errors.append(
                f"{method} runtime export summary does not exist: {summary_text}"
            )

    if runtime is not None and runtime.is_file():
        signature, header = runtime_header(runtime)
        if signature != str(contract["header"]):
            errors.append(
                f"{method} runtime header {signature!r} != {contract['header']!r}"
            )
        for key in (
            "backbone_type",
            "hidden_dim",
            "depth",
            "direction_distribution",
            "vmf_components",
            "discrete_diffusion_mode",
        ):
            if not values_match(header.get(key), contract[key]):
                errors.append(
                    f"{method} runtime {key}={header.get(key)!r}, "
                    f"expected {contract[key]!r}"
                )

    if summary_path is not None and summary_path.is_file():
        try:
            summary = load_json(summary_path)
        except (OSError, json.JSONDecodeError, FreshSceneSmokeConfigError) as exc:
            errors.append(f"{method} invalid runtime export summary: {exc}")
        else:
            summary_config = summary.get("config") or {}
            model_config = summary.get("model_config") or {}
            training_config = summary.get("training_config") or {}
            if summary_config.get("model_kind") != contract["model_kind"]:
                errors.append(
                    f"{method} model_kind={summary_config.get('model_kind')!r}, "
                    f"expected {contract['model_kind']!r}"
                )
            for key in (
                "backbone_type",
                "hidden_dim",
                "depth",
                "direction_distribution",
                "vmf_components",
            ):
                if not values_match(model_config.get(key), contract[key]):
                    errors.append(
                        f"{method} summary {key}={model_config.get(key)!r}, "
                        f"expected {contract[key]!r}"
                    )
            # Older simple-MLP export summaries omit this field even though
            # the authoritative runtime header records ``legacy``.
            if (
                "discrete_diffusion_mode" in model_config
                and not values_match(
                    model_config["discrete_diffusion_mode"],
                    contract["discrete_diffusion_mode"],
                )
            ):
                errors.append(
                    f"{method} summary discrete_diffusion_mode="
                    f"{model_config['discrete_diffusion_mode']!r}, expected "
                    f"{contract['discrete_diffusion_mode']!r}"
                )
            if not values_match(training_config.get("seed"), required_seed):
                errors.append(
                    f"{method} training seed={training_config.get('seed')!r}, "
                    f"expected {required_seed}"
                )
            replay_names = {
                Path(str(value)).name
                for value in training_config.get("replay_paths", [])
            }
            expected_replay_name = Path(str(scene["replay_path"])).name
            if replay_names != {expected_replay_name}:
                errors.append(
                    f"{method} replay provenance={sorted(replay_names)!r}, "
                    f"expected {[expected_replay_name]!r}"
                )

            if checkpoint is not None:
                summary_checkpoint = summary.get("checkpoint_path")
                if not isinstance(summary_checkpoint, str) or (
                    workspace_path(summary_checkpoint) != checkpoint
                ):
                    errors.append(
                        f"{method} summary checkpoint does not match registered path"
                    )
            if runtime is not None:
                summary_runtime = summary.get("runtime_path")
                if not isinstance(summary_runtime, str) or (
                    workspace_path(summary_runtime) != runtime
                ):
                    errors.append(
                        f"{method} summary runtime does not match registered path"
                    )

    return runtime, errors


def validate_scene(
    *, manifest: dict[str, Any], scene: dict[str, Any], jobs: dict[str, Any]
) -> tuple[dict[str, Path], list[str]]:
    errors: list[str] = []
    scene_key = str(scene.get("scene_key", "<missing>"))
    job_id = scene.get("job_id")
    job = jobs.get(job_id)
    if job is None:
        errors.append(f"unknown baseline-manifest job_id: {job_id!r}")
    elif job.get("scene_path") != scene.get("scene_path"):
        errors.append(
            f"scene path disagrees with baseline manifest: {scene.get('scene_path')!r}"
        )

    scene_path, scene_errors = validate_hashed_path(
        label="scene XML",
        path_text=scene.get("scene_path"),
        expected_sha256=scene.get("scene_sha256"),
        resolver=project_path,
    )
    replay, replay_errors = validate_hashed_path(
        label="replay",
        path_text=scene.get("replay_path"),
        expected_sha256=scene.get("replay_sha256"),
        resolver=workspace_path,
    )
    errors.extend(scene_errors)
    errors.extend(replay_errors)
    if scene_path is not None and scene_path.is_file() and job is not None:
        try:
            root = et.parse(scene_path).getroot()
        except et.ParseError as exc:
            errors.append(f"scene XML parse failed: {exc}")
        else:
            integrator = root.find("integrator")
            if integrator is None:
                errors.append(
                    "scene XML has no integrator block for Phase 13 replacement"
                )
            else:
                observed_type = integrator.attrib.get("type")
                expected_type = job.get("expected_integrator")
                if observed_type != expected_type:
                    errors.append(
                        f"source integrator={observed_type!r}, "
                        f"baseline manifest expects {expected_type!r}"
                    )
    if replay is not None and replay.is_file():
        nonblank_records = sum(
            1
            for line in replay.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
        if nonblank_records != int(scene.get("replay_records", -1)):
            errors.append(
                f"replay record count={nonblank_records}, "
                f"expected {scene.get('replay_records')!r}"
            )

    contract_root = manifest["selected_artifact_contract"]
    required_seed = int(contract_root["smoke_training_seed"])
    runtimes: dict[str, Path] = {}
    for method in ("ordered_diffusion_mlp_h72_k4", "simple_mlp_k4"):
        runtime, artifact_errors = validate_selected_artifact(
            scene=scene,
            method=method,
            contract=contract_root[method],
            required_seed=required_seed,
        )
        errors.extend(artifact_errors)
        if runtime is not None:
            runtimes[method] = runtime

    if errors:
        return {}, [f"{scene_key}: {error}" for error in errors]
    return runtimes, []


def base_config(
    *, manifest: dict[str, Any], scene: dict[str, Any], bundle: dict[str, Any]
) -> dict[str, Any]:
    smoke = manifest["smoke"]
    scene_key = str(scene["scene_key"])
    bundle_key = str(bundle["key"])
    return {
        "profile_id": manifest["profile_id"],
        "job_ids": [scene["job_id"]],
        "width": smoke["width"],
        "height": smoke["height"],
        "crop_width": smoke["crop_width"],
        "crop_height": smoke["crop_height"],
        "reference_sample_count": smoke["reference_sample_count"],
        "compare_sample_count": smoke["compare_sample_count"],
        "samples_per_pass": smoke["samples_per_pass"],
        "threads": smoke["threads"],
        "seed": smoke["seed"],
        "reference_seed": smoke["reference_seed"],
        "mpg_train_budget": smoke["mpg_train_budget"],
        "masked_diffusion_train_budget": smoke[
            "masked_diffusion_train_budget"
        ],
        "mpg_train_fusion": smoke["mpg_train_fusion"],
        "runtime_only_methods": [],
        "learned_proposal_inv_prob_clamp": smoke[
            "learned_proposal_inv_prob_clamp"
        ],
        "learned_proposal_contribution_clamp": smoke[
            "learned_proposal_contribution_clamp"
        ],
        "learned_proposal_max_trials": smoke["learned_proposal_max_trials"],
        "diffusion_steps": smoke["diffusion_steps"],
        "bounce_alpha": bundle["bounce_alpha"],
        "learned_direction_kappa_scale": bundle[
            "learned_direction_kappa_scale"
        ],
        "learned_direction_flip_to_receiver_hemisphere": bundle[
            "learned_direction_flip_to_receiver_hemisphere"
        ],
        "time_budget_seconds": smoke["time_budget_seconds"],
        "postprocess_crop_from_full_exr": smoke[
            "postprocess_crop_from_full_exr"
        ],
        "output_dir": (
            "results/fresh_scene_feasibility_smoke_macos_20260901/"
            f"{scene_key}/{bundle_key}"
        ),
        "methods": list(bundle["methods"]),
    }


def build_scene_configs(
    *, manifest: dict[str, Any], scene: dict[str, Any], runtimes: dict[str, Path]
) -> dict[str, dict[str, Any]]:
    configs: dict[str, dict[str, Any]] = {}
    output_dir = Path(str(manifest["generated_config_dir"]))
    scene_key = str(scene["scene_key"])
    for bundle in manifest["bundle_layout"]["configs_per_ready_scene"]:
        bundle_key = str(bundle["key"])
        payload = base_config(manifest=manifest, scene=scene, bundle=bundle)
        learned_methods = [
            method for method in bundle["methods"] if method in runtimes
        ]
        if learned_methods:
            payload["runtime_model_paths"] = {
                method: runtimes[method].relative_to(WORKSPACE_ROOT).as_posix()
                for method in learned_methods
            }
            payload["runtime_only_methods"] = learned_methods
        relative_path = output_dir / (
            f"phase_13_fresh_scene_smoke_{scene_key}_{bundle_key}_macos.json"
        )
        configs[relative_path.as_posix()] = payload
    return configs


def main() -> int:
    args = parse_args()
    manifest_path = args.manifest.resolve()
    manifest = load_json(manifest_path)
    if manifest.get("schema_version") != 1:
        raise FreshSceneSmokeConfigError("unsupported prerequisite schema_version")
    validate_frozen_protocol(manifest)

    baseline = load_json(WORKSPACE_ROOT / "configs" / "baseline_manifest.json")
    jobs = {item["id"]: item for item in baseline.get("render_jobs", [])}

    all_configs: dict[str, dict[str, Any]] = {}
    blocked: dict[str, list[str]] = {}
    for scene in manifest.get("scenes", []):
        runtimes, errors = validate_scene(manifest=manifest, scene=scene, jobs=jobs)
        scene_key = str(scene.get("scene_key", "<missing>"))
        if errors:
            blocked[scene_key] = errors
            print(f"BLOCKED {scene_key}")
            for error in errors:
                print(f"  - {error}")
            continue
        print(f"READY {scene_key}")
        all_configs.update(
            build_scene_configs(manifest=manifest, scene=scene, runtimes=runtimes)
        )

    total_scenes = len(manifest.get("scenes", []))
    ready_scenes = total_scenes - len(blocked)
    print(
        f"summary: ready_scenes={ready_scenes} blocked_scenes={len(blocked)} "
        f"candidate_configs={len(all_configs)}"
    )

    if not args.emit_ready:
        print("check-only: no files written and no training/rendering launched")
        return 0

    if blocked:
        print(
            "refusing partial emission: register and validate every selected "
            "checkpoint/runtime first",
            file=sys.stderr,
        )
        return 2

    for relative_path, payload in all_configs.items():
        destination = WORKSPACE_ROOT / relative_path
        if destination.exists() and not args.force:
            raise FreshSceneSmokeConfigError(
                f"refusing to replace existing config without --force: {destination}"
            )

    for relative_path, payload in all_configs.items():
        destination = WORKSPACE_ROOT / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(payload, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"WROTE {relative_path}")

    print("configs emitted only; no training or rendering launched")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
