---
name: generate-pipeline
description: Generate a sample config.json for the LSFM pipeline from image characteristics and requested analysis, then enqueue it in the harness Active queue. Prefer config.json over a shell script. Use when the user describes a new sample, wants a pipeline config, or asks to run analysis.
---

# generate-pipeline Skill

When the user provides analysis requirements and image characteristics, **write `config.json` first**. Running happens in the pipeline harness UI, not via a generated `.sh` unless the user explicitly wants a script.

Also follow `.cursor/skills/yifu-run/SKILL.md`.

## Invocation pattern

User says something like:
- "Generate a pipeline script for sample X with ch1, enhance fibers and remove edge signal"
- "Run only segmentation for sample Y"
- "I have a new sample, it has bright sheet-like noise at edges"

## Primary outputs

1. `<sample_dir>/config.json` based on `config/config_template.json`
2. Optional Active enqueue: `python -m pipeline_modules.harness enqueue --sample-dir "<sample_dir>"`
3. Operator UI: `python -m apps.pipeline_harness --host 127.0.0.1 --port 8766`
4. A shell script only if the user still wants one

Sample directories live under `H:\arivis-analysis`. The Active queue is `H:\arivis-analysis\_active`.

## Pipeline architecture

### New pipeline order (main.py)

```
Step 1/6: Registration channel downsample
Step 2/6: Atlas registration and label outputs
Step 3/6: Signal preprocessing and Zarr conversion
Step 4/6: Segmentation
Step 5/6: Region density analysis (or Spotiflow summary)
Step 6/6: Vessel network reconstruction (if enabled)
```

Optional edge signal removal runs inside Step 3 when `preprocessing.edge_signal_removal.apply=true` and atlas label TIFF is available.

### Available preprocessing steps (config-based, in preprocessor.py)

These run at Step 3 (TIFF-level 2D per-slice processing), controlled by config.json `preprocessing` section:

```json
"channel_subtraction": {"apply": true, ...}
"tophat": {"apply": true, "kernel_size": 21}
"rolling_ball": {"apply": false, "radius": 50}
"scattering_removal": {"apply": true, "sigma": 25.0, "weight": 1.0}
"median_filter": {"apply": true, "kernel_size": 5}
"clahe": {"apply": false, "clip_limit": 2.0, "tile_grid_size": 8}
```

### Available 3D Zarr-level preprocessing steps

These run as independent CLI modules and are wired into main.py when enabled:

```json
"edge_signal_removal": {
  "apply": true/false,
  "inward_px": 50,
  "suppression_weight": 0.8,
  "brightness_pct": 90.0,
  "smooth_sigma": 5.0
}
```

### Segmentation methods

- `cellpose` -- distributed 3D via Dask, needs GPU
- `threshold` -- simple intensity threshold
- `cfos_unet` -- custom U-Net inference
- `spotiflow` -- spot detection; step 5 becomes Spotiflow region counts

### Skip flags supported by main.py

- `--skip_registration` -- skip ANTs registration

### Standalone module paths (for partial runs)

All modules are in `pipeline_modules/preprocessing/`:

| Module | CLI entry | Description |
|--------|-----------|-------------|
| `tiff_to_zarr.py` | `python -m pipeline_modules.preprocessing.tiff_to_zarr --input ... --output ... --chunk_size "128,256,256"` | TIFF to Zarr |
| `edge_signal_removal.py` | `python -m pipeline_modules.preprocessing.edge_signal_removal --input_dir ... --label_dir ... --output_dir ...` | Edge signal removal |
| `preprocessor.py` | `python -m pipeline_modules.preprocessing.preprocessor --config config.json --sample_dir ...` | 2D TIFF preprocessing |
| `downsample.py` | `python pipeline_modules/preprocessing/downsample.py --input_folder ... --factor "z,y,x"` | Registration downsampling |

## Config generation rules

1. Start from `config/config_template.json`
2. Write to `<sample_dir>/config.json` (analysis root `H:\arivis-analysis`)
3. Open matching files from `capabilities.json` / each `capability_manifest.json`
4. Enqueue with `python -m pipeline_modules.harness enqueue --sample-dir "<sample_dir>"` when the user wants it in Active
5. Do not start a long `main.py` run yourself; the harness UI owns that
6. Config tuning from image characteristics:
   - "bright sheet noise at brain edge" → enable `edge_signal_removal`
   - "uneven dye intensity" → enable `scattering_removal` and/or `clahe`
   - "autofluorescence bleed-through" → enable `channel_subtraction`
7. Output file naming convention:
   - Signal Zarr: `sample_dir/ch{SIGNAL_CH}.zarr`
   - Label Zarr: `sample_dir/upsampled_atlas_label.zarr`
   - Mask Zarr: `sample_dir/ch{SIGNAL_CH}_mask.zarr`
   - Density Excel: `sample_dir/results/{sample}_{channel}_brain_distribution_stats.xlsx`

## Optional shell script

Only if the user explicitly wants a `.sh`:

1. **Shebang + preamble**: `#!/usr/bin/env bash`, `set -euo pipefail`
2. **Conda environment**: Use `micromamba run -n yifu python ...` for every command
3. **Variable header**: Let user change `SAMPLE_DIR`, `CONFIG`, `SIGNAL_CH`, `REG_CH` at top
4. Prefer a single `main.py` invocation after config is written, or `python -m pipeline_modules.harness enqueue`

```bash
#!/usr/bin/env bash
set -euo pipefail

SAMPLE_DIR="<sample_dir>"
CONFIG="$SAMPLE_DIR/config.json"

python -m pipeline_modules.harness enqueue --sample-dir "$SAMPLE_DIR" --config "$CONFIG"
echo "Added to Active. Start the UI: python -m apps.pipeline_harness --host 127.0.0.1 --port 8766"
```

## Smart defaults for common scenarios

### Scenario A: "Standard cFos analysis"
- 2D preprocessing: channel_subtraction + tophat + median_filter + scattering_removal
- segmentation: method=cfos_unet
- Full pipeline (steps 1-5); tubule off unless asked

### Scenario B: "Quick check, no registration"
- Skip registration entirely if outputs already exist
- Just TIFF → Zarr → segmentation

### Scenario C: "Edge noise removal only"
- Registration must already be done (or run it first)
- Enable `edge_signal_removal` in config
