# mask_zarr_to_atlas_points.py

## Purpose

Convert `sample_dir/chX_mask.zarr` into a compact point-cloud CSV that can be rendered by `render_points_brainrender.py`.

Instead of exporting every foreground voxel, the script bins the binary mask onto a target physical grid, default `25,25,25` um. Each occupied bin becomes one point at the foreground-voxel centroid inside that bin. This keeps the display light while preserving the original signal distribution better than using bin centers.

## Usage

From a sample directory:

```bash
micromamba run -n yifu python pipeline_modules/visualization/mask_zarr_to_atlas_points.py \
  --sample_dir "S:\path\to\sample" \
  --signal_ch ch1 \
  --resolution_xyz 1.8,1.8,2.0 \
  --target_resolution_xyz 25,25,25
```

Explicit mask path:

```bash
micromamba run -n yifu python pipeline_modules/visualization/mask_zarr_to_atlas_points.py \
  --mask_zarr "S:\path\to\sample\ch1_mask.zarr" \
  --resolution_xyz 1.8,1.8,2.0 \
  --output "S:\path\to\sample\ch1_mask_atlas_points.csv"
```

Then render:

```bash
micromamba run -n napari python pipeline_modules/visualization/render_points_brainrender.py \
  --points_csv "S:\path\to\sample\ch1_mask_atlas_points.csv" \
  --point_color "#ff3366" \
  --point_radius 35 \
  --root_alpha 0.12
```

## Output

The CSV contains:

- `x,y,z`: physical point centroid in microns, readable by `render_points_brainrender.py`
- `grid_x,grid_y,grid_z`: target-grid bin index
- `voxel_count`: number of original foreground voxels represented by this point
- `signal_volume_um3`: represented foreground volume
- `coordinate_space`: metadata tag, default `sample`

A JSON summary is written next to the CSV.

## Important Coordinate Note

`chX_mask.zarr` is usually in sample space. For a true standard-brain view in `brainrender`, the signal should already be transformed into atlas space first, for example by running the existing `image2atlas` registration flow. If you use this script directly on `chX_mask.zarr`, the point cloud is a faithful 25 um physical summary of the sample mask, but it is not automatically registered to the Allen standard brain.
