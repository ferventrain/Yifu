# cFos Interactive Visual Report Spec

## 1. Goal

Build an interactive report page for current cFos pipeline outputs. The report should combine Allen 25 um atlas anatomy, cFos region statistics, atlas-space heatmaps, 3D point/activation rendering, spatial pattern analysis, hotspot discovery, marker relationships, group statistics, and exportable tables/figures.

The first implementation must align with the data that this repository already produces. New features that require additional analysis outputs are listed explicitly in section 8.

## 2. Current Pipeline Alignment

### 2.1 Primary Per-Region Statistics Source

Use the Excel workbook produced by:

`pipeline_modules.registration.region_signal_analysis_zarr_graph`

Typical path:

`sample_dir/results/<sample>_<channel>_brain_distribution_stats.xlsx`

The workbook contains multiple `Level_*` sheets. Each sheet has one row per Allen region at that hierarchy level.

Existing required columns:

| Current column | Report meaning |
|---|---|
| `Name` | Allen region display name, currently often formatted as `full name,acronym` |
| `Total Voxels` | Region volume in voxels |
| `Signal Voxels` | cFos-positive foreground voxels assigned to the region |
| `Voxel Density` | `Signal Voxels / Total Voxels` |
| `Signal Count` | cFos object/component count assigned to the region |
| `Sum Intensity` | Sum of signal intensity over assigned cFos foreground voxels |

If hemisphere analysis is enabled, the workbook may also contain:

| Current column | Report meaning |
|---|---|
| `Left Total Voxels` | Left hemisphere region volume in voxels |
| `Left Signal Voxels` | Left cFos-positive foreground voxels |
| `Left Voxel Density` | Left `Signal Voxels / Total Voxels` |
| `Left Signal Count` | Left cFos object count |
| `Left Sum Intensity` | Left signal intensity sum |
| `Right Total Voxels` | Right hemisphere region volume in voxels |
| `Right Signal Voxels` | Right cFos-positive foreground voxels |
| `Right Voxel Density` | Right `Signal Voxels / Total Voxels` |
| `Right Signal Count` | Right cFos object count |
| `Right Sum Intensity` | Right signal intensity sum |

The report should normalize these into frontend metric names:

| Frontend metric | Source / formula |
|---|---|
| `cfos_count` | `Signal Count` |
| `signal_voxels` | `Signal Voxels` |
| `voxel_density` | `Voxel Density` |
| `region_volume_voxels` | `Total Voxels` |
| `sum_intensity` | `Sum Intensity` |
| `mean_cfos_intensity` | `Sum Intensity / Signal Voxels`, zero if `Signal Voxels == 0` |
| `left_cfos_count` | `Left Signal Count`, if present |
| `right_cfos_count` | `Right Signal Count`, if present |
| `left_voxel_density` | `Left Voxel Density`, if present |
| `right_voxel_density` | `Right Voxel Density`, if present |
| `count_laterality_index` | `(Right Signal Count - Left Signal Count) / (Right Signal Count + Left Signal Count)` |
| `density_laterality_index` | `(Right Voxel Density - Left Voxel Density) / (Right Voxel Density + Left Voxel Density)` |

Note: the pipeline currently provides voxel density, not true cells/mm3 density, unless additional conversion metadata is supplied. If the report displays `cell_density`, it must either:

- label the current value as `Voxel Density`, or
- compute `cells/mm3` only when atlas/sample voxel size and object-count volume normalization are explicitly available.

### 2.2 Region Metadata Source

Use:

`pipeline_modules/registration/Region_Csv_Rev1_updated.CSV`

Required fields already used by the pipeline:

| Column | Usage |
|---|---|
| `id` | Allen region ID |
| `name` | Display name used to match Excel `Name` |
| `acronym` | Region acronym |
| `structure_id_path` | Region hierarchy and parent lookup |

The report should build the left panel region tree from this CSV. Search should match:

- region ID
- full English name
- acronym
- optional Chinese alias if a later alias table is provided

### 2.3 Atlas / Heatmap Sources

Existing reusable modules:

| Capability | Existing module |
|---|---|
| 2D Allen atlas slice extraction | `pipeline_modules.visualization.atlas_slice` |
| Region metric slice heatmap from Excel | `pipeline_modules.visualization.atlas_slice` |
| 3D atlas-space heatmap volume/render output | `pipeline_modules.visualization.heatmap` |
| Batch AP-like density slices | `pipeline_modules.visualization.heatmap --mode batch-cell-density-slices` |
| Atlas-space points export | `pipeline_modules.visualization.warp_mask_zarr_to_atlas_points` |
| Brainrender point rendering | `pipeline_modules.visualization.render_points_brainrender` |

Important axis convention:

Atlas TIFF volumes are interpreted as `(DV, AP, ML)`.

Supported slice planes:

| Plane | Fixed axis |
|---|---|
| `coronal` | AP |
| `sagittal` | ML |
| `horizontal` | DV |

### 2.4 Spotiflow Point Outputs

If the segmentation method is Spotiflow, use:

`pipeline_modules.segmentation.spotiflow_inference`

Point CSV fields:

| Column | Meaning |
|---|---|
| `z` | Z coordinate in sample/atlas-aligned array coordinates |
| `y` | Y coordinate |
| `x` | X coordinate |
| `region_id` | Atlas region ID, if label Zarr was provided |
| `region_name` | Allen region name |
| `region_acronym` | Allen acronym |
| `tile_z0`, `tile_y0`, `tile_x0` | Source tile origin |

Region count CSV fields:

| Column | Meaning |
|---|---|
| `region_id` | Allen region ID |
| `region_name` | Allen region name |
| `region_acronym` | Allen acronym |
| `signal_count` | Detected spot count |

Current limitation: these outputs are point detections and region counts. They do not yet include hotspot clusters, local cluster density, cluster centroid tables, or per-spot intensity values.

### 2.5 Existing Group / Coarse Region Outputs

Reusable scripts:

| Existing module | What it provides |
|---|---|
| `coarse_region_metric_plot.py` | Single-sample coarse-region CSV/XLSX and atlas-order/sorted bar plots |
| `group_density_coarse_region_bar.py` | Multi-sample grouped coarse-region bar plot, no inferential statistics |
| `top_level7_density_ratio.py` | Two-sample top region log-ratio ranking, no p-values |
| `region_group_signal_count.py` | Configured system/region-group summary for a metric |

Default coarse regions currently include:

`Isocortex`, `Hippocampal formation`, `Olfactory areas`, `Cortical subplate`, `Striatum`, `Pallidum`, `Thalamus`, `Hypothalamus`, `Midbrain`, `Pons`, `Medulla`, `Cerebellum`, `fiber tracts`, `ventricular systems`.

## 3. Data Model for the Report

The report backend should convert current pipeline files into a normalized JSON-ready model.

### 3.1 `Sample`

| Field | Source |
|---|---|
| `sample_id` | sample folder name or metadata file |
| `group` | user-provided group table; not currently inferred by pipeline |
| `density_excel` | region statistics Excel path |
| `atlas_volume_tiff` | heatmap atlas-space volume, if present |
| `points_csv` | atlas-space points CSV, if present |
| `spotiflow_points_csv` | Spotiflow points CSV, if present |
| `atlas_version` | default `allen_mouse_25um` unless configured |

### 3.2 `RegionMetric`

| Field | Source / formula |
|---|---|
| `sample_id` | sample |
| `region_id` | region metadata lookup by `Name` |
| `region_name` | `Name` split before final comma where possible |
| `region_acronym` | region metadata or `Name` suffix |
| `structure_id_path` | region CSV |
| `level` | `Level_*` sheet name |
| `cfos_count` | `Signal Count` |
| `signal_voxels` | `Signal Voxels` |
| `voxel_density` | `Voxel Density` |
| `region_volume_voxels` | `Total Voxels` |
| `sum_intensity` | `Sum Intensity` |
| `mean_cfos_intensity` | `Sum Intensity / Signal Voxels` |
| `left_*`, `right_*` | hemisphere columns if available |
| `laterality_index` | configurable, default count-based LI |
| `rank_by_count` | rank descending by `cfos_count` within sample |
| `rank_by_density` | rank descending by `voxel_density` within sample |

### 3.3 `SystemMetric`

Use `config/region_groups.json` and/or coarse Allen parent regions.

| Field | Source / formula |
|---|---|
| `system_name` | group name |
| `member_region_ids` | resolved from acronyms in group config |
| `system_cfos_count` | sum of `Signal Count` |
| `system_signal_voxels` | sum of `Signal Voxels` |
| `system_total_voxels` | sum of `Total Voxels` |
| `system_voxel_density` | `system_signal_voxels / system_total_voxels` |
| `activation_load` | `system_cfos_count / whole_brain_cfos_count` |
| `enrichment_score` | `(system_cfos_count / whole_brain_cfos_count) / (system_total_voxels / whole_brain_total_voxels)` |
| `activated_region_count` | count of member regions above activation threshold |
| `top_region` | max region by selected metric |

### 3.4 `ClusterMetric`

This is a new required output for hotspot analysis.

| Field | Required source |
|---|---|
| `cluster_id` | new hotspot clustering module |
| `sample_id` | sample |
| `region_id` | atlas label at cluster centroid or majority region |
| `cluster_size` | number of points/objects in cluster |
| `cluster_density` | local density |
| `centroid_z/y/x` | cluster centroid in atlas array coordinates |
| `centroid_ap/dv/ml` | converted coordinates |
| `peak_intensity` | max or local mean intensity, if signal volume available |
| `spot_score` | cluster size/density/intensity combined score |

## 4. Page Layout

Use a three-panel application layout.

### 4.1 Left Panel: Region Browser

Required behavior:

- Hierarchical Allen region tree from `Region_Csv_Rev1_updated.CSV`
- Search by English name, acronym, ID, and later Chinese alias
- Multi-select regions
- Filter by activated-only, top-N, brain system, hemisphere, and level
- Selected regions highlight in the central atlas/3D view

### 4.2 Center Panel: Atlas / Heatmap / 3D View

Required modes:

- 2D atlas slice with region metric overlay
- 3D atlas-space activation view
- Full-brain heatmap from cached atlas volume if available
- Point cloud view from `points.csv` or Spotiflow points CSV if available

Required controls:

- Metric switch: `cfos_count`, `signal_voxels`, `voxel_density`, `mean_cfos_intensity`, `laterality_index`, and later `hotspot_score`
- Plane switch: coronal / sagittal / horizontal
- Slice coordinate slider
- Rotate / zoom / pan in 3D mode
- Click region
- Click cluster, after hotspot module exists

### 4.3 Right Panel: Statistics / Details

Default state:

- Total cFos count
- Activated region count
- Top activated regions
- System-level activation summary
- Left/right whole-brain comparison, if hemisphere columns exist
- Existing data/output status
- Missing feature status for hotspot, marker, and group-stat modules

Region-click state:

- Region name, acronym, ID
- cFos count
- Signal voxels
- Voxel density
- Region volume in voxels
- Mean cFos intensity
- Left/right metrics if present
- Laterality index if present
- System membership
- Whole-brain ranking
- Group statistics if loaded
- Hotspot statistics once cluster module exists

Cluster-click state:

- Cluster ID
- Region
- Cluster size
- Cluster density
- Centroid
- Peak intensity
- Spot score
- Local slice preview

## 5. Analytical Modules

### 5.1 Overview

Required:

- Whole-brain 2D/3D heatmap
- Top activated regions table
- Metric ranking by `Signal Count`, `Signal Voxels`, `Voxel Density`, `Sum Intensity`, `mean_cfos_intensity`
- Activated region threshold:
  - default: `Signal Count > 0`
  - optional: top percentile or z-score threshold

### 5.2 System-Level Interpretation

Required:

- System summary table from region groups/coarse regions
- Activation load
- Enrichment score
- System-level heatmap/bar chart
- Click system to highlight member regions

Existing support:

- `region_group_signal_count.py`
- `coarse_region_metric_plot.py`

Missing:

- Web-native system heatmap data endpoint
- Consistent group membership resolver for all region tree levels

### 5.3 Spatial Pattern

Required:

- AP, DV, ML axis distribution
- Click axis bin to show corresponding 2D heatmap
- Laterality analysis

Current support:

- 2D slices can be rendered by `atlas_slice.py` and `heatmap.py`
- Batch coronal/bregma slices exist
- Hemisphere columns exist if config enables hemisphere analysis

Missing:

- A general AP/DV/ML histogram module from atlas-space points or heatmap volume
- Interactive bin-to-slice linkage
- Laterality index precomputation table

Laterality formula:

```text
LI = (Right - Left) / (Right + Left)
```

Default count-based LI:

```text
count_laterality_index =
  (Right Signal Count - Left Signal Count) /
  (Right Signal Count + Left Signal Count)
```

### 5.4 Hotspot / Cluster Analysis

Required:

- Cluster count
- Cluster region table
- Hotspot overlay
- Click cluster to highlight in atlas/3D view
- Local 2D slice around cluster

Current support:

- Spotiflow can output point detections and region counts
- Heatmap volume can show smoothed density

Missing:

- Cluster detection module, e.g. DBSCAN/HDBSCAN over atlas-space points
- Cluster-to-region assignment
- Cluster density / spot score computation
- Cluster table export
- Cluster click/highlight frontend state

### 5.5 Multi-Marker Relationship

Required:

- Marker colocalization heatmap
- Region-marker enrichment heatmap
- Brain region ranking by colocalization with cFos

Current support:

- Pipeline can organize channels and process signal channels, but no unified multi-marker result table is present.

Missing:

- Multi-channel/marker sample manifest
- Per-marker region metrics table
- Point-level or voxel-level colocalization computation
- Marker-pair overlap/correlation metrics
- Region-level marker enrichment ranking

### 5.6 Group Statistics

Required:

- Load multiple samples and user-provided group labels
- Differential region list
- Differential system list
- Group heatmap
- Volcano plot
- UMAP/PCA
- Region-click group comparison panel

Current support:

- Multi-sample grouped coarse-region bar plot
- Two-sample log-ratio top-region plot

Missing:

- Group metadata import schema
- Replicate-aware statistical tests
- p-value / q-value calculation
- Effect size calculation
- Volcano plot data
- UMAP/PCA embedding
- Per-region box/violin data endpoint

## 6. Exploratory Findings

The report should auto-generate findings from available data. Each finding should be clickable and highlight the relevant region or cluster.

Supported in MVP:

- High activation regions:
  - based on top percentile or z-score of `Signal Count` / `Voxel Density`
- Strong laterality regions:
  - if hemisphere columns exist
- High system enrichment:
  - based on system enrichment score

Requires new modules:

- Local hotspot findings
- Multi-marker colocalization findings
- Group differential findings with p/q-values

## 7. Export Requirements

Required MVP exports:

- Selected region metrics as CSV/XLSX
- Top activated regions as CSV
- System summary as CSV
- Current 2D heatmap as PNG/SVG
- Current 3D screenshot as PNG, if the 3D viewer supports screenshot capture

Later exports:

- Cluster table CSV/XLSX
- Multi-marker colocalization matrix CSV
- Group differential table CSV/XLSX
- Exploratory findings HTML/PDF

Every export should include:

- `sample_id`
- `group`, if available
- selected metric
- atlas version
- source file paths
- analysis timestamp
- relevant parameters

## 8. Missing Functionality Checklist

### 8.1 Data Normalization Layer

Needed:

- Read all `Level_*` Excel sheets into one normalized table
- Map `Name` to `region_id`, acronym, hierarchy path
- Compute `mean_cfos_intensity`
- Compute laterality indices when left/right columns exist
- Emit report-ready JSON/Parquet/CSV

Status: missing as a unified report backend layer. Pieces exist in visualization utilities.

### 8.2 Interactive Web Report

Needed:

- Web app shell with three panels
- Region tree search/multi-select
- 2D atlas slice viewer
- 3D point/heatmap viewer
- Region click state
- Export UI

Status: missing.

### 8.3 True Web 3D Atlas Viewer

Needed:

- Browser-based 3D view using atlas mesh/volume/points
- Rotate, zoom, pan
- Region selection/highlight
- Screenshot export

Status: partially supported offline by `render_points_brainrender.py`; missing web-native integration.

### 8.4 AP/DV/ML Axis Distribution

Needed:

- Compute histograms from atlas-space point CSV or atlas-space heatmap volume
- Link bins to 2D slice rendering

Status: missing.

### 8.5 Hotspot Clustering

Needed:

- DBSCAN/HDBSCAN or connected local maxima over atlas-space points/volume
- Cluster metric table
- Cluster-to-region assignment
- Hotspot score

Status: missing.

### 8.6 Multi-Marker Analysis

Needed:

- Marker/channel metadata
- Per-marker region metrics
- Colocalization and enrichment matrices
- Marker relationship plots

Status: missing.

### 8.7 Group Statistical Testing

Needed:

- Sample group metadata file
- Long-format multi-sample metric table
- Statistical test selection
- p-value/q-value/effect-size output
- Volcano, group heatmap, UMAP/PCA

Status: partially supported by simple grouped plots and two-sample ratios; inferential statistics are missing.

### 8.8 Chinese Region Names

Needed:

- Chinese alias table keyed by Allen region ID or acronym

Status: missing.

## 9. Implementation Priority

### P0: Current-Data MVP

- Data normalization layer for current Excel outputs
- Three-panel static/interactively-filtered report
- Region tree from Allen CSV
- Region search/multi-select
- 2D atlas slice heatmap from `Signal Count` / `Voxel Density`
- Top activated regions
- Region-click details
- System summary using region groups/coarse regions
- Hemisphere/laterality display when left/right columns exist
- CSV export of selected region metrics

### P1: Spatial + 3D

- Browser 3D point/heatmap view
- AP/DV/ML distributions
- Bin-to-slice interaction
- 3D screenshot export
- Current camera/view persistence

### P2: Hotspots

- Hotspot clustering module
- Cluster table and region assignment
- Cluster overlay and click detail
- Local hotspot slices

### P3: Multi-Sample Statistics

- Group metadata import
- Replicate-aware statistics
- Differential region/system tables
- Group heatmap
- Volcano plot
- UMAP/PCA

### P4: Multi-Marker + Report Narrative

- Multi-marker colocalization metrics
- Marker-region ranking
- Exploratory findings generator
- HTML/PDF report export

## 10. MVP Acceptance Criteria

- Loads one sample Excel workbook from `region_signal_analysis_zarr_graph`
- Parses all `Level_*` sheets
- Maps regions to Allen IDs and hierarchy
- Displays top activated regions by `Signal Count` and `Voxel Density`
- Displays a 2D atlas slice heatmap for a selected metric
- Shows a searchable, selectable Allen region tree
- Clicking a region updates the right detail panel
- Computes and displays `mean_cfos_intensity`
- Displays laterality metrics if left/right columns exist
- Displays system-level activation load and enrichment score
- Exports selected region metrics as CSV
- Clearly marks unavailable modules such as hotspot, multi-marker, and inferential group statistics instead of silently showing empty results
