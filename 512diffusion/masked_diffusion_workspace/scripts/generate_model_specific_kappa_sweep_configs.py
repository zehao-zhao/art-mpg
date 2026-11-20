from __future__ import annotations

import json
from pathlib import Path


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = WORKSPACE_ROOT / "configs"
SCENES = {
    "lamp": {
        "job_id": "paper_lamp_mpg",
        "runtime_paths": {
            "ordered_diffusion_mlp_h72_k4": (
                "results/ordered_absorbing_lamp_macos_20260901/phase12/"
                "mlp_h72_k4/ordered_diffusion_mlp_h72_k4_runtime.txt"
            ),
            "simple_mlp_k4": (
                "results/ordered_absorbing_lamp_simple_k4_macos_20260901/phase12/"
                "simple_h126_k4/simple_mlp_k4_runtime.txt"
            ),
        },
    },
    "slabs": {
        "job_id": "paper_slabs_mpg",
        "runtime_paths": {
            "ordered_diffusion_mlp_h72_k4": (
                "results/ordered_absorbing_sweep_macos_20260901/phase12/"
                "mlp_h72_k4/ordered_diffusion_mlp_h72_k4_runtime.txt"
            ),
            "simple_mlp_k4": (
                "results/ordered_absorbing_slabs_simple_k4_macos_20260901/phase12/"
                "simple_h126_k4/simple_mlp_k4_runtime.txt"
            ),
        },
    },
}
METHODS = {
    "diffusion": {
        "method": "ordered_diffusion_mlp_h72_k4",
        "bounce_alpha": 0.0,
    },
    "simple": {
        "method": "simple_mlp_k4",
        "bounce_alpha": 0.5,
    },
}
SEEDS = (104729, 130363, 155921)
KAPPA_SCALES = (0.25, 0.5, 1.0, 2.0)


def scale_token(value: float) -> str:
    return f"{int(round(value * 100)):03d}"


def build_config(
    *, scene_key: str, method_key: str, seed: int, kappa_scale: float
) -> dict[str, object]:
    scene = SCENES[scene_key]
    method_spec = METHODS[method_key]
    method = str(method_spec["method"])
    token = scale_token(kappa_scale)
    return {
        "profile_id": "macos_arm64_native",
        "job_ids": [scene["job_id"]],
        "width": 64,
        "height": 64,
        "crop_width": 64,
        "crop_height": 64,
        "reference_sample_count": 1024,
        "compare_sample_count": 64,
        "samples_per_pass": 1,
        "threads": 4,
        "seed": seed,
        "reference_seed": 99173,
        "mpg_train_budget": 0.2,
        "masked_diffusion_train_budget": 0.2,
        "mpg_train_fusion": True,
        "runtime_model_paths": {method: scene["runtime_paths"][method]},
        "runtime_only_methods": [method],
        "learned_proposal_inv_prob_clamp": 512.0,
        "learned_proposal_contribution_clamp": 32.0,
        "learned_proposal_max_trials": 2048,
        "diffusion_steps": 10,
        "bounce_alpha": method_spec["bounce_alpha"],
        "learned_direction_kappa_scale": kappa_scale,
        "learned_direction_flip_to_receiver_hemisphere": False,
        "time_budget_seconds": 900,
        "output_dir": (
            "results/model_specific_kappa_sweep_macos_20260901/"
            f"{scene_key}_{method_key}_seed{seed}_k{token}"
        ),
        "methods": ["path_reference", method],
    }


def main() -> int:
    written: list[Path] = []
    for scene_key in SCENES:
        for method_key in METHODS:
            for seed in SEEDS:
                for kappa_scale in KAPPA_SCALES:
                    token = scale_token(kappa_scale)
                    path = CONFIG_DIR / (
                        f"phase_13_kappa_scale_{scene_key}_{method_key}_"
                        f"seed{seed}_k{token}_noflip_macos.json"
                    )
                    payload = build_config(
                        scene_key=scene_key,
                        method_key=method_key,
                        seed=seed,
                        kappa_scale=kappa_scale,
                    )
                    path.write_text(
                        json.dumps(payload, indent=2) + "\n",
                        encoding="utf-8",
                    )
                    written.append(path)
    print(f"wrote {len(written)} configs")
    for path in written:
        print(path.relative_to(WORKSPACE_ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
