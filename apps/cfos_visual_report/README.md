# cFos Visual Report (v0.1)

Interactive three-panel report for current cFos pipeline Excel outputs (`region_signal_analysis_zarr_graph`).

## Features

### P0
- Normalizes all `Level_*` Excel sheets into report-ready JSON
- Searchable Allen region tree with multi-select
- 2D atlas slice heatmaps for cFos count / voxel density / laterality
- Overview statistics, system activation load / enrichment, exploratory findings
- CSV export for selected regions

### P1 / v0.1
- AP / DV / ML **cFos count** distribution histograms (prefers atlas volume TIFF when available)
- Click histogram bin to jump the 2D slice to the linked plane/coordinate
- 2D heatmap modes (same as `heatmap.py`): **region metric** vs **local signal density**
- Region select → jump to coronal centroid slice and show **Bregma (mm)**
- Bookmark slices and export PNG heatmaps as ZIP (region and/or signal modes)
- Browser 3D point-cloud viewer (Three.js) when `points.csv` or cached atlas volume exists
- 3D region surface focus, optional brain outline toggle, trackball-style camera
- 3D screenshot export (PNG)
- Saved camera view per sample in browser localStorage

### P3 / two-sample comparison
- Multi-sample group manifest (CSV or JSON)
- **Exactly 2 samples:** top differential regions table, Pearson scatter plot (region-level)
- 2D heatmap modes for compare sample:
  - **hemisphere** — left/right metrics from same sample (L count vs R count)
  - **dual** — side-by-side panels (sample A | sample B)
  - **split_lr** — one slice, ML-left = A, ML-right = B
  - **diff** — A − B (diverging cmap, same as `heatmap.py`)
  - **fold** — log2(A/B)
- Welch / Mann–Whitney, BH q-values, volcano, group heatmap, PCA

## Install

From the repo root:

```bash
pip install -r apps/cfos_visual_report/requirements.txt
```

Uses the main `yifu` environment for atlas rendering dependencies (matplotlib, pandas, tifffile, scipy, etc.).

Optional for UMAP:

```bash
pip install umap-learn
```

## Build report JSON only

```bash
python pipeline_modules/visualization/cfos_report_data.py --sample_dir "S:\path\to\sample" --signal_ch ch1 --group control
```

Output defaults to `sample_dir/visualization/<sample>_cfos_report.json`.

## Run the web app

```bash
python apps/cfos_visual_report/main.py --host 127.0.0.1 --port 8765
```

Open `http://127.0.0.1:8765`, enter a sample path in the header, and click **Load sample**. No sample path is required at startup.

## Expected inputs

Primary statistics workbook:

`sample_dir/results/<sample>_<channel>_brain_distribution_stats.xlsx`

Optional spatial assets for P1:

- `sample_dir/visualization/points.csv`
- `sample_dir/visualization/<sample>_<channel>_heatmap_3d_volume.tiff`
- Spotiflow points CSV under the sample tree

### Group manifest (P3)

CSV with columns:

| column | required | description |
|---|---|---|
| `sample_dir` | yes | Path to each sample folder |
| `group` | yes | Group label (e.g. `control`, `treatment`) |
| `signal_ch` | no | Defaults to `ch1` |
| `sample_id` | no | Defaults to folder name |

Example `group_manifest.csv`:

```csv
sample_dir,group,signal_ch
S:\study\mouse01,control,ch1
S:\study\mouse02,control,ch1
S:\study\mouse03,treatment,ch1
S:\study\mouse04,treatment,ch1
```

## API endpoints

- `GET /api/report?sample_dir=...`
- `GET /api/slice.png?sample_dir=...&metric=cfos_count&plane=coronal&coordinate=216&color_mode=region|signal`
- `GET /api/region/slice-focus?sample_dir=...&region_id=315`
- `GET /api/spatial/axes?sample_dir=...`
- `GET /api/spatial/points?sample_dir=...`
- `GET /api/spatial/region-surface?sample_dir=...&region_id=997`
- `GET /api/spatial/brain-outline-surface?sample_dir=...`
- `POST /api/export/slice-bookmarks.zip` — body: `{ "sample_dir": "...", "bookmarks": [...], "color_modes": ["region","signal"] }`
- `POST /api/group/analyze` — body: `{ "manifest_path": "...", "level": "Level_8", "metric": "cfos_count", "group_a": "control", "group_b": "treatment" }`
- `GET /api/group/analyze?manifest_path=...`
- `GET /api/group/export/differential-regions.csv?manifest_path=...`
- `GET /api/export/regions.csv?sample_dir=...&region_ids=1,2,3`

## Tests

```bash
pytest tests/test_cfos_report_data.py tests/test_cfos_report_spatial.py tests/test_cfos_report_group_stats.py -q
```
