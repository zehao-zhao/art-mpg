# ART-MPG reproduction

Linux CPU source package. Install CMake, Ninja, a C++17 compiler, OpenEXR development libraries and the Python packages in `requirements.txt`. Run commands from this directory; `python` denotes your Python environment.

## Build

Mitsuba's bundled third-party dependency sources and licenses are included under `512diffusion/mitsuba/ext/`. These dependencies are stored directly in this repository, so no separate submodule initialization is needed for them.

```sh
cmake -S 512diffusion/mitsuba -B work/renderer -G Ninja -DCMAKE_BUILD_TYPE=Release -DMTS_ENABLE_PYTHON=OFF -DMTS_ENABLE_GUI=OFF -DMTS_ENABLE_EMBREE=OFF -DMTS_ENABLE_LTO=OFF
cmake --build work/renderer --parallel 4
python -B experiments/equal_time/build_timed_methods.py --base-build work/renderer --output work/timed
```

The main renderer integrates the PPG delta-light/delta-BSDF MIS and guide consistency fixes, and the directional-emitter zero-PDF and direction fixes. The three timed kernels preserve the implementations and controls used for R1–R2, including their existing counter overhead. Their names are `art_joint`, `mpg`, `npm`. The joint kernel uses cached CDF/direction outputs, context spacing 0.25 and 4,096 local slots. `old_code/` holds only the exact v1 originals of the three replaced renderer files, with package-relative paths; it is not built or imported.

## Original paper code

Renderer, scene and model workspace code remains at its original `512diffusion/` location, including the original paper experiment/ablation scripts and configs. Paper training configs and recorded model/seed bindings are in `recorded_experiments/`. Descriptive entry points under `models/` and `experiments/` are additional reproduction interfaces.

## Inputs and models

Scene assets, the paper ART-MPG exports and the six equal-time joint R/T exports are included. External replay datasets, reference EXRs, checkpoints and large NPM runtime exports are not included. Input paths are listed in `models/configs/`, `experiments/equal_spp/models.json`, and `experiments/equal_time/models.json`. Training configs expand `${ARTMPG_ROOT}` at load time.

For paper training use `models/scripts/train_art_mpg.py --config models/configs/SCENE/diffusion_mlp_seed104729.json` and `train_npm.py` with the matching NPM config. Repeat for seed 155921. `export_art_mpg.py --config EXPORT.json` writes native ART-MPG or NPM runtime text: the JSON fields are `checkpoint_path`, `model_kind` (`masked_diffusion` or `npm_adapted`), `output_dir`, `output_name`. `export_npm.py` additionally exports a Python inference bundle. `export_replay.py --raw RAW.jsonl --output REPLAY.jsonl` converts MPG's native subpath export (integrator property `export_samples_path`) to training data.

To train the equal-time models, provide the exact six replay files and hashes specified in `experiments/equal_time/training_data.json`. Cornell Glass uses 206,811 records. Joint R/T is trained for 32 epochs with seed 104729, one CPU thread, hidden width 40, depth 3 and four vMF components. NPM uses the fixed scene config and four physical cores.

```sh
python -B experiments/equal_time/train_models.py --data /path/to/replays --output work/offline_models --joint-cpu 0 --npm-cpus 2,4,6,8
```

This writes native models, a model-path JSON and `training_times.csv`. The joint clock includes training, validation, checkpoint and runtime export; the NPM clock measures its training subprocess, excluding runtime export. Both exclude replay acquisition. Recorded clocks are hardware dependent.

## Table 1, Figure 1 and appendix

`paper_references.json` maps each scene to its matching original 64×64 reference EXR. The ten render seeds were randomly selected for the paper experiments. Their recorded values are reused for reproduction. The D01–D10 values are listed below. The model seed identifies the separately trained model used for each run; it is distinct from the render seed. Model-seed bindings are the same across all scenes for ART-MPG and NPM; the other methods do not use a pre-trained model. See [SEEDS.md](SEEDS.md) and `experiments/recorded_seeds.json` for the complete seed settings.

| Run | Render seed | ART-MPG/NPM model seed |
|---|---:|---:|
| D01 | 1668592294 | 104729 |
| D02 | 994471241 | 104729 |
| D03 | 472027829 | 104729 |
| D04 | 2129530871 | 104729 |
| D05 | 1936485002 | 104729 |
| D06 | 610686297 | 104729 |
| D07 | 1239430601 | 104729 |
| D08 | 1710829629 | 104729 |
| D09 | 457121851 | 155921 |
| D10 | 797665335 | 104729 |

Quantitative jobs use 64×64 and 12/24/36/48/60 SPP. All native crops use D01, 60 SPP, the original 1152×1152 film, two 288×288 ROIs and fixed exposure/sRGB transform.

```sh
python -B experiments/equal_spp/run_experiments.py --binary work/renderer/dist/mitsuba --output work/paper_runs --cpu 0 --references /path/to/paper_references.json
python -B experiments/equal_spp/make_tables.py --runs work/paper_runs --output work/paper_tables
python -B experiments/equal_spp/run_experiments.py --binary work/renderer/dist/mitsuba --output work/paper_crops --cpu 0 --native-crops
python -B experiments/equal_spp/make_figures.py --statistics work/paper_tables/statistics.csv --crops work/paper_crops --output work/paper_figures
```

The table script also accepts a per-run CSV with scene, method, seed, SPP and MSE. It requires all ten seeds and computes ordinary medians, means and sample SD (ddof=1). Figures reuse the submitted PDF strip geometry, fonts, order, labels, captions and reference pixels; only method crop pixels, metric numbers and curve points are replaced. Include the generated `figure1.tex` and `appendix_figures.tex` from their output directory. Include `table1.tex` and `appendix_table.tex` from the table output directory. Crop appearance and numerical MSE use their original separate resolutions.

## Tables R1–R2

Exactly six scenes (Living, Slabs, Cornell Glass, Pool, Water, Stone), three methods, 180/300-second budgets, five recorded seeds, 256×256, 16 physical cores per render. MPG's online learning is inside the budget; only its final film contributes to reported SPP. ART-MPG joint and NPM are pre-trained.

`time_references.json` maps each scene to two matching 256×256 reference EXRs (seeds 314159/314160, 32,768 SPP each); their linear RGB average is the evaluation reference. Living/Cornell Glass use SMS with weight controls disabled; Slabs/Pool/Water/Stone use the paper-policy PT references. Keep these inputs fixed when reproducing the recorded tables.

```sh
python -B experiments/equal_time/run_experiments.py --build work/timed --output work/time_runs --references /path/to/time_references.json --cpus 0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30
python -B experiments/equal_time/make_tables_R1_R2.py --runs work/time_runs --training work/offline_models/training_times.csv --output work/time_tables
```

Supply `--models work/offline_models/models.json` to use freshly trained exports. Without it, install NPM exports at the paths in `experiments/equal_time/models.json`; the included joint exports are used. A single job uses `run_render.py --scene living --method art_joint --seed 1729 --seconds 10 --cpus ... --build work/timed --output work/one_run --reference REF1.exr REF2.exr`.

Outputs include R1/R2 Markdown and LaTeX, all 36 median MSE/SPP cells and 12 offline clocks. Aggregation requires all five seeds and rejects duplicates or missing cells. CPU lists must contain distinct physical cores; package launchers lock physical cores including SMT siblings. Run other timed processes on disjoint cores. Output directories must be new; completed matrices can be resumed. Rendered SPP, errors and timings vary with hardware and scheduling. Models and reference inputs must remain fixed.
