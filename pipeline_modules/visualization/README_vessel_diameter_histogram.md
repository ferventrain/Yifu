# vessel_diameter_histogram.py

Generate a publication-ready vessel diameter histogram directly from `kimimaro_reconstruction.py` branch metrics.

## Input

- `vessel_branch_metrics.csv`
- Required column: `mean_radius_um`

The script converts branch mean radius to diameter with:

`diameter_um = mean_radius_um * 2`

Invalid values (`NaN`, `inf`, `<= 0`) are filtered automatically.

## Usage

```bash
python pipeline_modules/visualization/vessel_diameter_histogram.py \
  --branch_csv "S:\可视化素材\血管\skeleton\vessel_branch_metrics.csv"
```

This writes `vessel_diameter_histogram.png` next to the CSV by default.

## Example With Explicit Output

```bash
python pipeline_modules/visualization/vessel_diameter_histogram.py \
  --branch_csv "S:\可视化素材\血管\skeleton\vessel_branch_metrics.csv" \
  --output "S:\可视化素材\血管\skeleton\vessel_diameter_histogram.png" \
  --bins 28 \
  --dpi 300 \
  --figsize 8.4,5.6
```

## CLI Options

- `--branch_csv`: input branch metrics CSV
- `--output`: output image path
- `--bins`: histogram bin count
- `--dpi`: output DPI
- `--title`: figure title
- `--xlabel`: x-axis label
- `--ylabel`: y-axis label
- `--figsize`: figure size as `width,height`
- `--xlim`: optional x-axis limits as `min,max`

## Figure Contents

- White-background publication layout with a restrained bio-journal palette
- Histogram of branch mean vessel diameter
- Thin count outline on top of bars
- Mean and median reference lines
- Summary box with `n`, mean, median, std, min, max
