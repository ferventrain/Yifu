# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository purpose

This repository implements an LSFM (light-sheet fluorescence microscopy) processing pipeline for large 3D datasets. The main supported end-to-end flow is:

1. configurable TIFF preprocessing
2. TIFF → Zarr conversion
3. registration-channel downsampling to NIfTI
4. segmentation on Zarr volumes
5. ANTs atlas registration
6. atlas-label conversion back to Zarr
7. brain-region signal analysis to Excel

The orchestration entrypoint is `main.py`, and the pipeline is driven by JSON config files (`config.json`, `config_template.json`, `config/config*.json`).

## Common commands

Project command-formatting rule: when giving the user shell commands, always provide each command as a single line so it can be pasted directly. Do not use line continuations, multi-line command blocks, or split arguments across lines unless the user explicitly asks for a formatted/multi-line version.

## Environment setup

The repo defines its main environment in `environment.yml`.

```bash
conda env create -f environment.yml
conda activate yifu
```

If the environment already exists:

```bash
conda env update -f environment.yml --prune
conda activate yifu
```

## Docker

The Dockerfile builds an image that defaults to running the environment smoke test:

```bash
docker build -t yifu .
docker run --rm yifu
```

## Environment / dependency smoke test

```bash
python test_env.py
```

This checks imports for ANTs, Cellpose, Torch, Dask, and runs small simulated registration / segmentation checks.

## Run the full pipeline

```bash
python main.py --config config.json --sample_dir "/path/to/sample"
```

Common skip flags:

```bash
python main.py --config config.json --sample_dir "/path/to/sample" --skip_preprocessing
python main.py --config config.json --sample_dir "/path/to/sample" --skip_segmentation
python main.py --config config.json --sample_dir "/path/to/sample" --skip_registration
python main.py --config config.json --sample_dir "/path/to/sample" --skip_analysis
```

Pipeline smoke test:

```bash
python main.py --test
```

Note: in current code, `main.py --test` only runs the Cellpose model check (`pipeline_modules/segmentation/cellpose_distributed.py --test`), not the full end-to-end pipeline.

## Run individual modules

### TIFF to Zarr

```bash
python pipeline_modules/preprocessing/tiff_to_zarr.py \
  --input "/path/to/tiff_folder" \
  --output "/path/to/output.zarr" \
  --chunk_size "128,256,256"
```

### Downsample registration channel

```bash
python pipeline_modules/preprocessing/downsample.py \
  --input_folder "/path/to/sample/ch1" \
  --factor "0.0800,0.0720,0.0720"
```

### Threshold segmentation

```bash
python pipeline_modules/segmentation/intensity_threshold_segmentor.py \
  --input_zarr "/path/to/input.zarr" \
  --output_zarr "/path/to/mask.zarr" \
  --threshold 1000 \
  --sigma 0
```

Threshold segmentation test mode only processes chunks that physically exist in the input Zarr store:

```bash
python pipeline_modules/segmentation/intensity_threshold_segmentor.py \
  --input_zarr "/path/to/input.zarr" \
  --output_zarr "/path/to/mask.zarr" \
  --test
```

### Distributed Cellpose segmentation

```bash
python pipeline_modules/segmentation/cellpose_distributed.py \
  --input_zarr "/path/to/input.zarr" \
  --output_zarr "/path/to/mask.zarr" \
  --pretrained_model cyto3 \
  --diameter 30 \
  --block_size "128,256,256" \
  --workers 4
```

Disable GPU if needed:

```bash
python pipeline_modules/segmentation/cellpose_distributed.py \
  --input_zarr "/path/to/input.zarr" \
  --output_zarr "/path/to/mask.zarr" \
  --no_gpu
```

### Single-image Cellpose check

```bash
python pipeline_modules/segmentation/test_single_image.py \
  --input "/path/to/image.tif" \
  --output output_test \
  --model cyto3
```

### Registration

```bash
python pipeline_modules/registration/ANTs_registration.py \
  --sample_dir "/path/to/sample" \
  --signal_channel 0 \
  --register_channel 1 \
  --atlas_image "/path/to/atlas.tiff" \
  --atlas_label "/path/to/atlas_label.tiff" \
  --mode atlas2image \
  --save_registered_image \
  --save_transforms \
  --config config.json
```

### Region analysis

```bash
python pipeline_modules/registration/region_signal_analysis_zarr_graph.py \
  --mask_zarr "/path/to/mask.zarr" \
  --label_zarr "/path/to/upsampled_atlas_label.zarr" \
  --signal_zarr "/path/to/signal.zarr" \
  --cfg pipeline_modules/registration/Region_Csv_Rev1_updated.CSV \
  --output density_results.xlsx \
  --foreground_mode equal \
  --foreground_label 1 \
  --resolution_xyz "1.8,1.8,2.0"
```

### Tubule reconstruction

This module is present but not yet wired into `main.py`.

```bash
python pipeline_modules/tubule_reconstruction/kimimaro_reconstruction.py --help
python pipeline_modules/tubule_reconstruction/view_skeleton_napari.py --help
```

## Testing / validation reality

There is no repo-level `pytest`, `tox`, or lint configuration in the current tree. Validation is mainly done through:

- `python test_env.py`
- `python main.py --test`
- standalone module scripts in `pipeline_modules/segmentation/`
- running the target module on a sample directory or small local dataset

Do not invent `pytest` or lint commands unless such tooling is added later.

## High-level architecture

## 1. `main.py` is a thin orchestrator over script-style modules

`main.py` does very little image processing itself. It:

- loads JSON config
- resolves sample/channel paths from `input.channels`
- runs module scripts via shell commands
- skips work when expected intermediate outputs already exist

The current main pipeline is fixed to `atlas2image` registration mode even if config contains another mode.

## 2. Storage strategy is intentionally mixed by stage

The codebase uses different formats for different steps:

- raw / preprocessed data: TIFF stacks
- segmentation + analysis working format: Zarr, usually with dataset `0`
- registration working volume: downsampled NIfTI (`volume.nii.gz`)
- registration outputs for atlas labels: TIFF stack, then converted back to Zarr
- final reports: Excel / CSV / JSON depending on module

Future edits should preserve this stage-specific format handoff unless the task is explicitly to unify formats.

## 3. Module responsibilities

### `pipeline_modules/preprocessing`

This directory owns:

- configurable image enhancement (`preprocessor.py`)
- TIFF → Zarr conversion (`tiff_to_zarr.py`)
- registration-channel downsampling + NIfTI export (`downsample.py`)
- mask upsampling (`upsample_mask.py`)
- channel subtraction utilities (`channel_subtraction.py`)
- **3D tubular enhancement** (`tubular_enhancement.py`) — Frangi/Meijering/Sato filters on Zarr volumes
- **Edge signal removal** (`edge_signal_removal.py`) — suppresses bright sheet-like noise at brain edges using atlas label

Important implementation detail: `Preprocessor` builds its enhancement pipeline by iterating the JSON `preprocessing` object in order, skipping bookkeeping sections such as `downsample`, `zarr`, `channel_subtraction`, `tubular_enhancement`, and `edge_signal_removal`. If preprocessing order matters, change the config order rather than assuming a hardcoded sequence.

### `pipeline_modules/segmentation`

Two segmentation modes are wired into `main.py`:

- `cellpose` via `cellpose_distributed.py`
- `threshold` via `intensity_threshold_segmentor.py`

Both operate on Zarr input and write Zarr masks. `main.py` then exports masks to TIFF when needed for downstream reuse.

The Cellpose path is Dask/distributed-oriented and is meant for chunked large-volume inference. The threshold path is simpler and can emit either connected-component labels or binary masks.

### `pipeline_modules/registration`

This directory combines two concerns:

- ANTs-based registration (`ANTs_registration.py`)
- region-level quantitative analysis (`region_signal_analysis_zarr_graph.py`, older `region_signal_analysis.py`)

The important architectural point is that region analysis is downstream of atlas labels already warped into sample space, so the default end-to-end path is atlas → image, not image → atlas.

### `pipeline_modules/visualization`

Contains separate visualization utilities such as `heatmap.py`. These are not currently part of the default `main.py` flow.

### `pipeline_modules/tubule_reconstruction`

Experimental / extension module for vessel skeleton reconstruction from binary mask Zarr using kimimaro/TEASAR. It has its own README and CLI scripts, but is currently outside the main orchestrated workflow.

## 4. Data flow through the default pipeline

For a sample with signal channel `chX` and registration channel `chY`, `main.py` currently expects / produces a flow like:

1. input TIFF stack: `sample_dir/chX/`
2. optional enhanced TIFF stack: `sample_dir/chX_preprocessed/`
3. signal Zarr: `sample_dir/chX.zarr`
4. registration downsample output: `sample_dir/chY_downsample/volume.nii.gz`
5. warped atlas label TIFF stack: `sample_dir/upsampled_atlas_label/`
6. warped atlas label Zarr: `sample_dir/upsampled_atlas_label.zarr`
7. optional cleaned Zarr (edge removal): `sample_dir/chX_clean.zarr`
8. optional enhanced Zarr (tubular): `sample_dir/chX_enhanced.zarr`
9. segmentation mask Zarr: `sample_dir/chX_mask.zarr`
10. exported mask TIFF stack: `sample_dir/chX_mask/`
11. region stats Excel: `sample_dir/density_results_chX.xlsx`

A lot of the repo logic assumes these names. Preserve them unless the task is specifically to redesign path conventions.

## 5. Configuration model

The active configuration format is JSON, not YAML.

Key top-level sections are:

- `input`
- `preprocessing`
- `registration`
- `segmentation`
- `analysis`
- optional `tubule_reconstruction`

Important config behaviors from code:

- `input.channels.signal` and `input.channels.registration` drive path naming like `ch0`, `ch1`
- `preprocessing.downsample.target_resolution_xyz` is used to derive Z/Y/X downsample factors for registration
- `preprocessing.edge_signal_removal.apply` gates the 3D edge noise removal (needs atlas label from registration)
- `preprocessing.tubular_enhancement.apply` gates the 3D Frangi/Meijering tubular filter
- `segmentation.method` selects `cellpose`, `threshold`, or `cfos_unet`
- `analysis.density_config` is resolved relative to the project root if not absolute
- `tubule_reconstruction` is already represented in `config_template.json`, but not yet called by `main.py`

## 6. Intermediate-result reuse is a core behavior

The pipeline is designed to skip expensive steps if outputs already exist. Before changing this behavior, check whether the task is actually asking to force recomputation.

Examples from current code:

- existing signal Zarr skips TIFF → Zarr conversion
- existing mask TIFF directory skips segmentation
- existing mask Zarr skips segmentation but may still export TIFF
- existing `upsampled_atlas_label/` skips registration

## Project-specific implementation notes

- Zarr readers throughout the repo commonly assume OME-Zarr-like layout with a single dataset named `0`. Many modules fall back to the only array if the group contains exactly one array.
- `downsample.py` currently uses nearest-neighbor interpolation for both masks and intensity images; do not assume linear interpolation is active just because older docs mention it.
- `ANTs_registration.py` forces atlas and register image direction matrices to identity to avoid orientation/reflection issues.
- `region_signal_analysis_zarr_graph.py` is the current Zarr-native analysis path and is the main analysis implementation to prefer over the older TIFF-oriented script.
- Hardware expectations in `hardware_requirements.md` are high (32+ vCPU, 128+ GB RAM, multi-GPU A100-class setup), so avoid casually introducing memory-heavy full-volume operations when chunked paths already exist.

## Documentation sources worth trusting

When repository docs disagree, prefer these sources in roughly this order:

1. current code in `main.py` and the module scripts it invokes
2. `config_template.json`
3. `PROJECT_DESIGN.md`
4. `README.md`
5. older per-module READMEs

Some READMEs describe planned or older flows that are not fully wired into the current main pipeline.

## Pipeline script generation skill

When the user describes a new sample analysis with image characteristics, use the `generate-pipeline` skill
(`.claude/skills/generate-pipeline.md`) to generate a shell script that runs the full or partial pipeline
from TIFF preprocessing through density analysis. The user triggers this by saying things like:

- "帮我生成一个pipeline脚本，样本在 X，有神经纤维和边缘噪声"
- "只跑 tubular enhancement 和 segmentation"
- "新样本来了，需要完整的分析流程"
