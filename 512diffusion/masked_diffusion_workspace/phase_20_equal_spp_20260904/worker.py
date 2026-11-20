#!/usr/bin/env python3
"""Isolated Phase 20 worker.

The coordinator gives each worker one immutable JSON specification and one
fresh attempt directory.  The worker prints exactly one JSON result on its
last stdout line; verbose library output remains in the coordinator log.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any


PACKAGE_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = PACKAGE_ROOT.parent
PROJECT_ROOT = WORKSPACE_ROOT.parent
SCRIPTS_ROOT = WORKSPACE_ROOT / "scripts"
SRC_ROOT = WORKSPACE_ROOT / "src"
for entry in (str(PACKAGE_ROOT), str(SCRIPTS_ROOT), str(SRC_ROOT)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

import protocol  # noqa: E402


class WorkerError(RuntimeError):
    """Raised when a worker cannot produce a valid artifact."""


def _read_spec(path: Path) -> dict[str, Any]:
    payload = protocol.load_json(path)
    if payload.get("protocol_id") != protocol.PROTOCOL_ID:
        raise WorkerError("worker specification protocol mismatch")
    return payload


def _sha(path: Path) -> str:
    if not path.is_file():
        raise WorkerError(f"missing expected artifact: {path}")
    return protocol.sha256_file(path)


def _workspace_relative(path: Path) -> str:
    return protocol.portable_workspace(path)


def _project_relative(path: Path) -> str:
    return protocol.portable_project(path)


def _latest(root: Path, pattern: str) -> Path:
    matches = sorted(root.glob(pattern), key=lambda item: item.stat().st_mtime_ns)
    if len(matches) != 1:
        raise WorkerError(f"expected one artifact matching {pattern}, found {len(matches)}")
    return matches[0]


def _runtime_profile():
    import phase_13_run_formal_benchmark as phase13
    from masked_diffusion_mpg.runtime import (
        detect_host_runtime,
        load_runtime_profiles,
        resolve_runtime_profile,
    )

    host = detect_host_runtime(PROJECT_ROOT)
    profiles = tuple(
        resolve_runtime_profile(template, host)
        for template in load_runtime_profiles(
            WORKSPACE_ROOT / "configs" / "runtime_profiles.json"
        )
    )
    profile = phase13.select_runtime_profile(profiles, "linux_native")
    if not profile.ready:
        raise WorkerError(f"linux runtime profile is not ready: {profile.reason}")
    return phase13, profile


def _run_render(spec: dict[str, Any], attempt_root: Path) -> dict[str, Any]:
    import phase_13_run_formal_benchmark as phase13
    from masked_diffusion_mpg.manifest import RenderJob
    from masked_diffusion_mpg.tail_diagnostics import analyze_image, analyze_renderer_log

    phase13, profile = _runtime_profile()
    method = str(spec["phase13_method"])
    logical_method = str(spec["logical_method"])
    runtime_path = None
    if spec.get("runtime_path") is not None:
        runtime_path = protocol.workspace_path(spec["runtime_path"], "runtime model")
        if _sha(runtime_path) != spec.get("runtime_sha256"):
            raise WorkerError(f"runtime hash mismatch: {runtime_path}")
    job = RenderJob(
        job_id=str(spec["job_id"]),
        group="phase20",
        method="mpg",
        variant="scalar_rgb",
        scene_path=str(spec["scene_path"]),
        expected_integrator="manifold_path_guiding",
    )
    is_reference = logical_method == "path_reference"
    render = spec["render"]
    config = phase13.Phase13BenchmarkConfig(
        profile_id="linux_native",
        job_ids=(job.job_id,),
        width=int(render["width"]),
        height=int(render["height"]),
        crop_width=int(render["width"]),
        crop_height=int(render["height"]),
        reference_sample_count=int(render["sample_count"]),
        compare_sample_count=int(render["sample_count"]),
        samples_per_pass=int(render["samples_per_pass"]),
        threads=int(spec["threads"]),
        seed=int(spec["render_seed"]),
        reference_seed=int(spec["render_seed"]),
        mpg_train_budget=float(render["mpg_train_budget"]),
        masked_diffusion_train_budget=float(render["mpg_train_budget"]),
        mpg_train_fusion=bool(render["mpg_train_fusion"]),
        output_dir=_workspace_relative(attempt_root / "artifacts"),
        methods=(method,),
        runtime_model_path=(None if runtime_path is None else _workspace_relative(runtime_path)),
        bounce_alpha=float(spec["bounce_alpha"]),
        learned_direction_kappa_scale=float(spec["kappa_scale"]),
        learned_direction_flip_to_receiver_hemisphere=bool(
            render["learned_direction_flip_to_receiver_hemisphere"]
        ),
        learned_proposal_inv_prob_clamp=float(render["learned_proposal_inv_prob_clamp"]),
        learned_proposal_contribution_clamp=float(
            render["learned_proposal_contribution_clamp"]
        ),
        learned_proposal_max_trials=int(render["learned_proposal_max_trials"]),
        runtime_only_methods=((method,) if runtime_path is not None else ()),
        time_budget_seconds=float(render["watchdog_seconds"]),
        fixed_spp=int(render["sample_count"]),
        postprocess_crop_from_full_exr=False,
        diffusion_steps=int(render["diffusion_steps"]),
        defer_metrics=not is_reference,
    )
    phase13.set_global_seed(int(spec["render_seed"]))
    run_dir = attempt_root / "artifacts" / "phase13"
    runtime_env = phase13.build_runtime_env(profile)
    record = phase13.run_render(
        profile=profile,
        runtime_env=runtime_env,
        config=config,
        job=job,
        method=method,
        run_dir=run_dir,
        runtime_model_path=runtime_path,
    )
    if not record.ok:
        raise WorkerError(
            f"render failed: method={logical_method} return={record.return_code}"
        )
    output_path = Path(record.output_path)
    tail = analyze_image(output_path)
    if not tail.get("available") or int(tail.get("invalid_pixel_count", 1)) != 0:
        raise WorkerError(f"render has unavailable/nonfinite pixel diagnostics: {output_path}")
    log_diagnostics = analyze_renderer_log(Path(record.stdout_log))
    record_payload = phase13.serialize_record(record, PROJECT_ROOT)
    result = {
        "kind": "reference" if is_reference else "render",
        "scene": spec["scene"],
        "logical_method": logical_method,
        "phase13_method": method,
        "render_seed": int(spec["render_seed"]),
        "model_seed": spec.get("model_seed"),
        "control_id": spec.get("control_id"),
        "record": record_payload,
        "output_path": _project_relative(output_path),
        "output_sha256": _sha(output_path),
        "active_config_sha256": record.active_config_sha256,
        "runtime_path": None if runtime_path is None else _workspace_relative(runtime_path),
        "runtime_sha256": None if runtime_path is None else _sha(runtime_path),
        "tail": {
            "pixel_count": tail["pixel_count"],
            "invalid_pixel_count": tail["invalid_pixel_count"],
            "top1_luminance_share": tail["luminance"]["top_tail_share"]["1pct"],
            "luminance_max_to_median": tail["luminance"]["max_to_median"],
        },
        "renderer_diagnostics": {
            "diagnostic_counters": log_diagnostics.get("diagnostic_counters", {}),
            "derived_rates": log_diagnostics.get("derived_rates", {}),
        },
    }
    return result


def _run_replay(spec: dict[str, Any], attempt_root: Path) -> dict[str, Any]:
    import phase_09_export_training_data as phase09
    from masked_diffusion_mpg.manifest import RenderJob
    from masked_diffusion_mpg.runtime import (
        detect_host_runtime,
        load_runtime_profiles,
        resolve_runtime_profile,
    )

    replay = spec["replay"]
    config = phase09.ExportConfig(
        profile_id="linux_native",
        job_ids=(str(spec["job_id"]),),
        seed_ids=tuple(int(value) for value in replay["seed_ids"]),
        width=int(replay["width"]),
        height=int(replay["height"]),
        crop_width=int(replay["width"]),
        crop_height=int(replay["height"]),
        sample_count=int(replay["sample_count"]),
        samples_per_pass=int(replay["samples_per_pass"]),
        threads=int(spec["threads"]),
        timeout_seconds=float(replay["timeout_seconds"]),
        train_budget=float(replay["train_budget"]),
        train_fusion=True,
        guided=True,
        allow_partial_aggregate=False,
        min_successful_seeds=len(replay["seed_ids"]),
        min_aggregate_raw_samples=int(replay["min_aggregate_records"]),
        integrator_integer_overrides={},
        integrator_float_overrides={},
        integrator_boolean_overrides={},
        integrator_string_overrides={},
    )
    job = RenderJob(
        job_id=str(spec["job_id"]),
        group="phase20",
        method="mpg",
        variant="scalar_rgb",
        scene_path=str(spec["scene_path"]),
        expected_integrator="manifold_path_guiding",
    )
    host = detect_host_runtime(PROJECT_ROOT)
    profiles = tuple(
        resolve_runtime_profile(template, host)
        for template in load_runtime_profiles(
            WORKSPACE_ROOT / "configs" / "runtime_profiles.json"
        )
    )
    profile = phase09.select_runtime_profile(profiles, "linux_native")
    if not profile.ready:
        raise WorkerError(f"linux runtime profile is not ready: {profile.reason}")
    output_root = attempt_root / "artifacts" / "replay_export"
    records = tuple(
        phase09.run_export_job(
            profile=profile,
            config=config,
            job=job,
            seed_id=seed,
            output_root=output_root,
        )
        for seed in config.seed_ids
    )
    aggregate = phase09.aggregate_export_runs(
        job=job,
        config=config,
        records=records,
        output_root=output_root,
    )
    if not aggregate.ok or aggregate.replay_record_count < int(
        replay["min_aggregate_records"]
    ):
        raise WorkerError(f"replay gate failed: {aggregate.error}")
    summary_path = output_root / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "protocol_id": protocol.PROTOCOL_ID,
                "scene": spec["scene"],
                "records": [asdict(record) for record in records],
                "aggregate": asdict(aggregate),
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    replay_path = Path(aggregate.replay_path)
    return {
        "kind": "replay",
        "scene": spec["scene"],
        "replay_path": _workspace_relative(replay_path),
        "replay_sha256": _sha(replay_path),
        "replay_record_count": aggregate.replay_record_count,
        "raw_sample_count": aggregate.raw_sample_count,
        "summary_path": _workspace_relative(summary_path),
        "summary_sha256": _sha(summary_path),
    }


def _run_train_diffusion(spec: dict[str, Any], attempt_root: Path) -> dict[str, Any]:
    from masked_diffusion_mpg.training import load_training_config, run_training

    replay_path = protocol.workspace_path(spec["replay_path"], "replay")
    if _sha(replay_path) != spec["replay_sha256"]:
        raise WorkerError("diffusion training replay hash mismatch")
    payload = dict(spec["training"])
    payload.update(
        {
            "replay_paths": [_workspace_relative(replay_path)],
            "output_dir": _workspace_relative(attempt_root / "artifacts"),
            "seed": int(spec["model_seed"]),
        }
    )
    config_path = attempt_root / "training_config.json"
    config_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    artifacts = run_training(
        workspace_root=WORKSPACE_ROOT,
        config=load_training_config(config_path),
    )
    summary = artifacts.run_dir / "run_summary.json"
    return {
        "kind": "train_diffusion",
        "scene": spec["scene"],
        "model_seed": int(spec["model_seed"]),
        "checkpoint_path": _workspace_relative(artifacts.best_checkpoint),
        "checkpoint_sha256": _sha(artifacts.best_checkpoint),
        "summary_path": _workspace_relative(summary),
        "summary_sha256": _sha(summary),
        "config_sha256": _sha(config_path),
    }


def _run_train_npm(spec: dict[str, Any], attempt_root: Path) -> dict[str, Any]:
    from masked_diffusion_mpg.npm_adapted import (
        load_npm_adapted_training_config,
        run_npm_adapted_training,
    )

    replay_path = protocol.workspace_path(spec["replay_path"], "replay")
    if _sha(replay_path) != spec["replay_sha256"]:
        raise WorkerError("NPM training replay hash mismatch")
    payload = dict(spec["training"])
    payload.update(
        {
            "replay_paths": [_workspace_relative(replay_path)],
            "output_dir": _workspace_relative(attempt_root / "artifacts"),
            "seed": int(spec["model_seed"]),
        }
    )
    config_path = attempt_root / "training_config.json"
    config_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    artifacts = run_npm_adapted_training(
        workspace_root=WORKSPACE_ROOT,
        config=load_npm_adapted_training_config(config_path),
    )
    return {
        "kind": "train_npm",
        "scene": spec["scene"],
        "model_seed": int(spec["model_seed"]),
        "checkpoint_path": _workspace_relative(artifacts.checkpoint_path),
        "checkpoint_sha256": _sha(artifacts.checkpoint_path),
        "summary_path": _workspace_relative(artifacts.summary_json),
        "summary_sha256": _sha(artifacts.summary_json),
        "config_sha256": _sha(config_path),
    }


def _run_export(spec: dict[str, Any], attempt_root: Path) -> dict[str, Any]:
    from masked_diffusion_mpg.runtime_export import (
        RuntimeExportConfig,
        export_runtime_model,
    )

    checkpoint = protocol.workspace_path(spec["checkpoint_path"], "checkpoint")
    if _sha(checkpoint) != spec["checkpoint_sha256"]:
        raise WorkerError("runtime export checkpoint hash mismatch")
    model_kind = (
        "masked_diffusion"
        if spec["kind"] == "export_diffusion"
        else "npm_adapted_seed_direction"
    )
    artifacts = export_runtime_model(
        workspace_root=WORKSPACE_ROOT,
        config=RuntimeExportConfig(
            checkpoint_path=_workspace_relative(checkpoint),
            output_dir=_workspace_relative(attempt_root / "artifacts"),
            output_name=f"{spec['scene']}_{spec['method']}_runtime.txt",
            model_kind=model_kind,
        ),
    )
    return {
        "kind": spec["kind"],
        "scene": spec["scene"],
        "method": spec["method"],
        "model_seed": int(spec["model_seed"]),
        "runtime_path": _workspace_relative(artifacts.runtime_path),
        "runtime_sha256": _sha(artifacts.runtime_path),
        "summary_path": _workspace_relative(artifacts.summary_json),
        "summary_sha256": _sha(artifacts.summary_json),
    }


def run(spec: dict[str, Any], attempt_root: Path) -> dict[str, Any]:
    kind = spec.get("kind")
    if kind in {"render", "reference"}:
        return _run_render(spec, attempt_root)
    if kind == "replay":
        return _run_replay(spec, attempt_root)
    if kind == "train_diffusion":
        return _run_train_diffusion(spec, attempt_root)
    if kind == "train_npm":
        return _run_train_npm(spec, attempt_root)
    if kind in {"export_diffusion", "export_npm"}:
        return _run_export(spec, attempt_root)
    raise WorkerError(f"unknown worker kind: {kind}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--attempt-root", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    spec = _read_spec(args.spec.resolve())
    attempt_root = args.attempt_root.resolve()
    attempt_root.mkdir(parents=True, exist_ok=True)
    result = run(spec, attempt_root)
    # Protect the coordinator parser from NaN/Infinity payloads.
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
