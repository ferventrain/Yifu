---
name: generate-pipeline
description: Generate a shell script to run the LSFM image analysis pipeline based on sample path, config, and module selection.
---

# generate-pipeline Skill

When the user provides analysis requirements and image characteristics, generate a `.sh` script that runs the full or partial LSFM pipeline.

## Invocation pattern

User says something like:
- "Generate a pipeline script for sample X with ch1, enhance fibers and remove edge signal"
- "Run only tubular enhancement and segmentation for sample Y"
- "I have a new sample, it has bright sheet-like noise at edges and thin neural fibers"

## Pipeline architecture

### New pipeline order (main.py)

```
Step 1:   TIFF 2D preprocessing + Zarr conversion
Step 1.5: Registration channel downsample
Step 2:   ANTs registration (atlas label warped to sample space)
Step 3:   Edge signal removal (uses atlas label, optional)
Step 4:   Tubular enhancement (Frangi/Meijering/Sato, optional)
Step 5:   Segmentation (Cellpose/threshold/cfos_unet)
Step 6:   Density analysis
```

### Available preprocessing steps (config-based, in preprocessor.py)

These run at Step 1 (TIFF-level 2D per-slice processing), controlled by config.json `preprocessing` section:

```json
"channel_subtraction": {"apply": true, ...}
"tophat": {"apply": true, "kernel_size": 21}
"rolling_ball": {"apply": false, "radius": 50}
"scattering_removal": {"apply": true, "sigma": 25.0, "weight": 1.0}
"median_filter": {"apply": true, "kernel_size": 5}
"clahe": {"apply": false, "clip_limit": 2.0, "tile_grid_size": 8}
```

### Available 3D Zarr-level preprocessing steps

These run at Steps 3-4 and are independent CLI modules:

```json
"edge_signal_removal": {
  "apply": true/false,
  "edge_width_px": 20,
  "suppression_weight": 0.8,
  "brightness_pct": 90.0,
  "smooth_sigma": 5.0
}
"tubular_enhancement": {
  "apply": true/false,
  "method": "frangi" | "meijering" | "sato",
  "sigmas": [1, 2, 4, 8],
  "black_ridges": false,
  "export_tiff": false
}
```

### Segmentation methods

- `cellpose` -- distributed 3D via Dask, needs GPU
- `threshold` -- simple intensity threshold
- `cfos_unet` -- custom U-Net inference

### Skip flags supported by main.py

- `--skip_registration` -- skip ANTs registration

### Standalone module paths (for partial runs)

All modules are in `pipeline_modules/preprocessing/`:

| Module | CLI entry | Description |
|--------|-----------|-------------|
| `tiff_to_zarr.py` | `python -m pipeline_modules.preprocessing.tiff_to_zarr --input ... --output ... --chunk_size "128,256,256"` | TIFF to Zarr |
| `tubular_enhancement.py` | `python -m pipeline_modules.preprocessing.tubular_enhancement --input_zarr ... --output_zarr ... --method frangi --sigmas "1,2,4,8"` | 3D tubular enhancement |
| `edge_signal_removal.py` | `python -m pipeline_modules.preprocessing.edge_signal_removal --input_zarr ... --label_zarr ... --output_zarr ...` | Edge signal removal |
| `preprocessor.py` | `python -m pipeline_modules.preprocessing.preprocessor --config config.json --sample_dir ...` | 2D TIFF preprocessing |
| `downsample.py` | `python pipeline_modules/preprocessing/downsample.py --input_folder ... --factor "z,y,x"` | Registration downsampling |

## Script generation rules

1. **Shebang + preamble**: `#!/usr/bin/env bash`, `set -euo pipefail`
2. **Conda environment**: Use `micromamba run -n yifu python ...` for every command
3. **Variable header**: Let user change `SAMPLE_DIR`, `CONFIG`, `SIGNAL_CH`, `REG_CH` at top
4. **Step comments**: Each step starts with `echo "===== Step X: Description ====="`
5. **Resume behavior**: The pipeline already skips if outputs exist, but add `|| true` checks
6. **Partial runs**: If user says "only tubular enhancement and segmentation", only generate those steps
7. **Config tuning**: If user describes image characteristics, suggest config values:
   - "bright sheet noise at brain edge" → enable `edge_signal_removal`
   - "thin neural fibers" → enable `tubular_enhancement` with smaller sigmas like [1, 2, 3]
   - "large blob-like structures" → enable `tubular_enhancement` with larger sigmas like [4, 8, 16]
   - "uneven dye intensity" → use `meijering` method instead of `frangi`
   - "dark fibers on bright background" → add `--black_ridges` to tubular enhancement
8. **Export TIFF option**: Add `--export_tiff "path"` when user asks for visual preview
9. **Output file naming convention**:
   - Signal Zarr: `sample_dir/ch{SIGNAL_CH}.zarr`
   - Clean Zarr (after edge removal): `sample_dir/ch{SIGNAL_CH}_clean.zarr`
   - Enhanced Zarr (after tubular): `sample_dir/ch{SIGNAL_CH}_enhanced.zarr`
   - Label Zarr: `sample_dir/upsampled_atlas_label.zarr`
   - Mask Zarr: `sample_dir/ch{SIGNAL_CH}_mask.zarr`
   - Density Excel: `sample_dir/density_results_ch{SIGNAL_CH}.xlsx`

## Template structure

```bash
#!/usr/bin/env bash
set -euo pipefail

# ---- Configuration (edit these) ----
SAMPLE_DIR="<sample_dir>"
CONFIG="config.json"
SIGNAL_CH="<ch_number>"
REG_CH="<ch_number>"

run() {
  micromamba run -n yifu python "$@"
}

# ---- Step 1: TIFF preprocessing + Zarr ----
echo "===== Step 1: TIFF preprocessing + Zarr ====="
run -m pipeline_modules.preprocessing.preprocessor \
  --config "$CONFIG" \
  --sample_dir "$SAMPLE_DIR"

# ... etc
```

## Smart defaults for common scenarios

### Scenario A: "Standard fiber analysis"
- 2D preprocessing: tophat + median_filter + scattering_removal
- edge_signal_removal: enabled
- tubular_enhancement: method=frangi, sigmas=[1, 2, 4, 8]
- segmentation: method=threshold or cfos_unet
- Full pipeline (steps 1-6)

### Scenario B: "Quick check, no registration"
- Skip registration entirely
- Just TIFF → Zarr → tubular enhancement → segmentation
- Add `--skip_registration` to main.py call

### Scenario C: "Edge noise removal only"
- Registration must already be done (or run it first)
- Only run edge_signal_removal + export_tiff for preview

### Scenario D: "Just tubular enhancement"
- Skip everything except tubular_enhancement + export_tiff
