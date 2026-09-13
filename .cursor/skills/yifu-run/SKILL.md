---
name: yifu-run
description: Generate a sample config.json from image characteristics and requested analysis, navigate to the matching pipeline modules, and enqueue the sample in the Yifu pipeline harness Active queue. Use when the user describes a new LSFM sample, wants a config.json, wants to run the analysis pipeline, or mentions Active queue / arivis-analysis / harness.
---

# Yifu run mode

Primary product is `config.json` plus optional Active-queue enrollment. Do not start from a shell script.

## Defaults

- Analysis root: `H:\arivis-analysis` (`YIFU_ANALYSIS_ROOT`)
- Active queue: `H:\arivis-analysis\_active` (`YIFU_ACTIVE_DIR`)
- Config destination: `<sample_dir>/config.json`
- Operator UI: `python -m apps.pipeline_harness --host 127.0.0.1 --port 8766`
- Enqueue: `python -m pipeline_modules.harness enqueue --sample-dir "<sample_dir>"`

## Workflow

1. Read `config/config_template.json` and `capabilities.json`.
2. If `sample_dir` is missing, ask for it. It must be under the analysis root. Sample folders may sit under `_active/<batch>/`, but never use `_active/jobs` or `_active/runs`.
3. Copy the template, set `project_name`, `input.channels`, preprocessing flags, `segmentation.method`, and `tubule_reconstruction.enabled`.
4. Write `<sample_dir>/config.json`. Do not overwrite an existing config unless the user asked to replace it.
5. Tell the user the main.py steps that will run (1 downsample, 2 registration, 3 preprocess/Zarr, 4 segmentation, 5 density or Spotiflow, 6 vessels if enabled).
6. Open the matching module files from the table below (read them; do not dump their contents).
7. Enqueue if the user wants the sample in Active: `python -m pipeline_modules.harness enqueue --sample-dir "<sample_dir>"`.
8. Remind them the operator UI is progress-only: adding a job auto-starts it after the current run finishes. Config is not generated in the Web form. Test/dev modes are skills, not UI tabs.

## Image → config mapping

Start from the template, then apply:

- bright sheet / edge noise → `preprocessing.edge_signal_removal.apply=true`
- uneven dye / scattering → `scattering_removal.apply=true` and optionally `clahe.apply=true`
- autofluorescence bleed-through → `channel_subtraction.apply=true`
- fibers / tophat background → `tophat.apply=true`
- cFos cells → `segmentation.method=cfos_unet`
- spots / nuclear dots → `segmentation.method=spotiflow`
- simple intensity mask → `segmentation.method=threshold`
- vessels / tubules → `tubule_reconstruction.enabled=true` (needs a mask, not Spotiflow-only)
- skip atlas → keep registration outputs if they already exist; tell the user the UI still runs `main.py` (existing intermediates are skipped)

### Scenario A: standard cFos

`channel_subtraction` + `tophat` + `median_filter` + `scattering_removal`; `segmentation.method=cfos_unet`; tubule off unless asked.

### Scenario B: fibers + edge noise

`tophat` + `edge_signal_removal`; threshold or existing mask method as requested.

### Scenario C: vessels

Threshold or existing mask; `tubule_reconstruction.enabled=true`.

## Modules to open

Use `capabilities.json` plus each module's `capability_manifest.json`. Typical mapping:

| Need | Open |
|------|------|
| Preprocessing / TIFF→Zarr / downsample | `pipeline_modules/preprocessing/capability_manifest.json`, `preprocessor.py`, `tiff_to_zarr.py`, `downsample.py` |
| Edge noise | `pipeline_modules/preprocessing/edge_signal_removal.py` |
| ANTs registration | `pipeline_modules/registration/capability_manifest.json`, `ANTs_registration.py` |
| Threshold / cFos U-Net / Spotiflow | `pipeline_modules/segmentation/capability_manifest.json` and the matching script |
| Region Excel | `pipeline_modules/registration/region_signal_analysis_zarr_graph.py` |
| Vessels | `pipeline_modules/tubule_reconstruction/capability_manifest.json` |
| Queue / UI | `pipeline_modules/harness/capability_manifest.json`, `apps/pipeline_harness/main.py` |
| Output paths | `pipeline_modules/utils/sample_layout.py` |

## After a run

Results live under the sample directory. Excel: `results/<sample>_<channel>_brain_distribution_stats.xlsx`. The UI lists paths that actually exist.

Do not invent pytest commands. Do not run the full pipeline yourself unless the user explicitly asks; the harness UI owns long jobs.
