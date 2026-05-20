# render_points_brainrender.py

## Purpose

Use `brainrender` to display atlas-space punctate fluorescence signals as colored dots inside a transparent standard mouse brain.

This is the first of the three planned standard-brain visualization modes:

1. transparent 3D brain + colored point signals
2. reserved for later
3. reserved for later

## Input

A CSV file containing point coordinates that are already in atlas space.

Supported coordinate column styles:

- `x,y,z`
- `ap,dv,ml`

You can also pass custom names with `--columns`.

## Usage

Interactive:

```bash
micromamba run -n napari python pipeline_modules/visualization/render_points_brainrender.py \
  --points_csv "S:\path\to\atlas_points.csv" \
  --point_color "#00bcd4" \
  --point_radius 35 \
  --root_alpha 0.10
```

Export PNG:

```bash
micromamba run -n napari python pipeline_modules/visualization/render_points_brainrender.py \
  --points_csv "S:\path\to\atlas_points.csv" \
  --columns ap,dv,ml \
  --point_color "#ff3366" \
  --point_radius 40 \
  --root_alpha 0.12 \
  --camera three_quarters \
  --output "S:\path\to\brainrender_points.png"
```

## Parameters

- `--points_csv`: atlas-space point CSV
- `--columns`: coordinate columns in `x_col,y_col,z_col` order
- `--atlas_name`: BrainGlobe atlas name, default `allen_mouse_25um`
- `--point_color`: point color
- `--point_alpha`: point opacity
- `--point_radius`: point radius in atlas units
- `--root_alpha`: whole-brain transparency
- `--background`: background color
- `--camera`: brainrender camera preset
- `--title`: optional title
- `--output`: if provided, save PNG instead of opening the interactive window
- `--screenshot_scale`: screenshot resolution multiplier
- `--show_axes`: show atlas axes

## Notes

- Coordinates must already be in standard atlas space before rendering.
- The script redirects `brainrender` runtime files to a temporary folder, which avoids the Windows log-file lock issue in some environments.
- `brainrender` internally handles atlas mesh orientation, so the CSV should stay in atlas coordinate order and should not be manually flipped for this script.
