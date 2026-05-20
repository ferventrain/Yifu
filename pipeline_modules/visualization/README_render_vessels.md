# render_vessels_pyvista.py — Vessel Topology Renderer

## Purpose

Renders SWC skeleton files as smooth 3D branch curves with red branch-point spheres. Each branch between two non-degree-2 nodes is rendered as a single smooth spline and colored by its mean vessel diameter, with a lower-right colorbar.

## Input

A directory of `.swc` files (one per skeleton/connected component), as produced by `kimimaro_reconstruction.py --save_swc`.

SWC format per line: `id type x y z radius parent`

## Pipeline Position

```
binary mask Zarr
  → kimimaro_reconstruction.py --save_skeleton --save_swc
    → swc/ directory
      → render_vessels_pyvista.py --swc_dir swc/
```

## CLI Usage

```bash
micromamba run -n yifu python pipeline_modules/visualization/render_vessels_pyvista.py \
    --swc_dir <path_to_swc_directory> \
    [options]
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--swc_dir` | path | required | Directory containing `.swc` files |
| `--max_files` | int | 0 | Limit number of SWC files loaded (0 = all) |
| `--spline_factor` | int | 12 | Interpolation density multiplier. Higher = smoother curves |
| `--smooth_window` | int | 5 | Moving-average window for jitter removal. Odd values work best. 1 = no smoothing |
| `--min_spacing` | float | 1.0 | Minimum distance (um) between control points. Removes redundant dense points |
| `--line_width` | float | 5.0 | Visual thickness of rendered curves |
| `--branch_point_size` | float | 2.0 | Radius (um) of red branch-point spheres |
| `--color_mode` | diameter/solid/branch | diameter | `diameter` = color by mean branch diameter, `branch` = different color per branch, `solid` = uniform color |
| `--branch_color` | str | #0072b2 | Color when `color_mode=solid` |
| `--colormap` | str | bright | Matplotlib colormap for branch coloring |
| `--background` | dark/paper/black/white | dark | Scene background color |
| `--diameter_clip_percentiles` | str | 2,98 | Percentile range used for diameter color clipping. Use `0,100` to disable clipping |
| `--output` | path | None | Save screenshot to file instead of opening interactive window |
| `--dpi` | int | 300 | Screenshot resolution scale |
| `--window_size` | str | 1920,1080 | Window dimensions as `width,height` |
| `--no_frame` | flag | false | Disable the scene bounding frame |
| `--frame_line_width` | float | 2.4 | Scene frame line width |
| `--camera` | iso/xy/xz/yz | iso | Camera preset for non-interactive rendering |
| `--parallel_projection` | flag | false | Use orthographic instead of perspective projection |
| `--save_on_close` | path | None | Interactive mode only: press `S` in the viewer to save the current view |
| `--interactive_output` | path | None | Interactive mode only: press `S` in the viewer to save the current view |

## Smoothing Pipeline

Each branch goes through three stages before rendering:

1. **min_spacing filter** — removes control points closer than `min_spacing` um to the previous kept point. Eliminates redundant dense vertices that cause spline overshoot.

2. **Moving-average smooth** — applies a sliding window average (size = `smooth_window`) to interior points. Endpoints are preserved exactly. Removes high-frequency jitter from skeletonization noise.

3. **Spline interpolation** — `pv.Spline(points, n)` fits a smooth parametric curve through the cleaned control points. `n = len(points) * spline_factor` controls output density.

## Recommended Settings

| Scenario | Command |
|----------|---------|
| Quick preview | `--spline_factor 12 --smooth_window 5 --min_spacing 1.0` |
| Publication quality | `--spline_factor 20 --smooth_window 7 --min_spacing 2.0 --line_width 5` |
| Very noisy skeleton | `--spline_factor 20 --smooth_window 11 --min_spacing 3.0` |
| Save to file | add `--output render.png --dpi 300` |

## Output

- **Interactive mode** (default): Opens a PyVista 3D viewer. Rotate/zoom with mouse.
- **File mode** (`--output path.png`): Saves a screenshot and exits.

## Dependencies

- pyvista
- numpy
- matplotlib (for colormaps)

## Notes

- The default renderer uses SWC radius values for branch color only. Visual tube width remains uniform and is controlled by `--line_width`.
- Branch points are identified as nodes with more than one child in the SWC tree.
- Each SWC file is treated as one skeleton. Branch IDs are globally unique across all loaded files.
- For best results, run `kimimaro_reconstruction.py` with `--prune_spurs_max_length_um 5.0` to reduce visual clutter from short terminal branches.
