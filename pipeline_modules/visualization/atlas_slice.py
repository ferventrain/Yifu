"""Render Allen atlas label slices from standardized coordinates.

The default bregma conversion is an approximate Allen CCF 25 um convention,
not a surgical-coordinate ground truth. Pass ``bregma_index`` (or CLI
``--bregma-index``) to override it for a specific atlas release/orientation.
Atlas TIFF volumes in this project are interpreted as ``(DV, AP, ML)``.
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import matplotlib

matplotlib.use("Agg")

from matplotlib import colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
import tifffile
from scipy import ndimage as ndi
from skimage.measure import find_contours

SVG_NS = "http://www.w3.org/2000/svg"
ET.register_namespace("", SVG_NS)

Plane = Literal["coronal", "sagittal", "horizontal"]
CoordinateSystem = Literal["bregma-mm", "ccf-um", "index"]

DEFAULT_ATLAS_LABEL = Path(__file__).resolve().parents[2] / "data" / "reference" / "atlas_label.tiff"
DEFAULT_BREGMA_INDEX = (18, 216, 228)  # (DV, AP, ML), approximate 25 um CCF bregma index.

PLANE_TO_FIXED_AXIS: dict[str, int] = {
    "horizontal": 0,  # DV
    "coronal": 1,  # AP
    "sagittal": 2,  # ML
}
PLANE_TO_OUTPUT_AXES: dict[str, tuple[int, int]] = {
    "coronal": (0, 2),  # rows DV, cols ML
    "sagittal": (0, 1),  # rows DV, cols AP
    "horizontal": (1, 2),  # rows AP, cols ML
}
AXIS_NAMES = ("DV", "AP", "ML")


def axis_index_for_name(axis_name: str) -> int:
    normalized = str(axis_name).strip().upper()
    if normalized not in AXIS_NAMES:
        raise ValueError(f"axis must be one of {AXIS_NAMES}, got: {axis_name}")
    return AXIS_NAMES.index(normalized)


def index_to_bregma_mm(
    axis: int | str,
    index: float,
    *,
    bregma_index: tuple[int, int, int] = DEFAULT_BREGMA_INDEX,
    resolution_um: float = 25.0,
) -> float:
    """Convert an atlas array index to mm relative to bregma along DV/AP/ML."""

    axis_idx = axis_index_for_name(axis) if isinstance(axis, str) else int(axis)
    if axis_idx == 1:  # AP: anterior is positive bregma mm.
        return (float(bregma_index[axis_idx]) - float(index)) * float(resolution_um) / 1000.0
    return (float(index) - float(bregma_index[axis_idx])) * float(resolution_um) / 1000.0


def bregma_mm_for_plane_index(
    plane: Plane,
    index: float,
    *,
    bregma_index: tuple[int, int, int] = DEFAULT_BREGMA_INDEX,
    resolution_um: float = 25.0,
) -> float:
    fixed_axis = PLANE_TO_FIXED_AXIS[plane]
    return index_to_bregma_mm(
        fixed_axis,
        index,
        bregma_index=bregma_index,
        resolution_um=resolution_um,
    )


@dataclass(frozen=True)
class AtlasSliceSpec:
    """Coordinate request for a 2D atlas slice.

    ``coordinate`` is interpreted along the plane's fixed axis:
    coronal=AP, sagittal=ML, horizontal=DV.
    """

    plane: Plane
    coordinate_system: CoordinateSystem
    coordinate: float
    atlas_resolution_um: float = 25.0
    bregma_index: tuple[int, int, int] = DEFAULT_BREGMA_INDEX


@dataclass(frozen=True)
class AtlasSlice:
    """Extracted 2D atlas label slice plus coordinate metadata for overlays."""

    image: np.ndarray
    plane: Plane
    index: int
    coordinate_system: CoordinateSystem
    coordinate: float
    coordinate_label: str
    atlas_resolution_um: float
    bregma_index: tuple[int, int, int]
    extent_mm: tuple[float, float, float, float]
    x_axis: str
    y_axis: str


WHITE_ORANGE_RED_BLACK_CMAP = LinearSegmentedColormap.from_list(
    "white_orange_red_black",
    [
        (0.0, "#ffffff"),
        (0.25, "#ff9900"),
        (0.5, "#ff0000"),
        (1.0, "#000000"),
    ],
)

WHITE_BLUE_RED_CMAP = LinearSegmentedColormap.from_list(
    "white_blue_red",
    [
        (0.000, "#000000"),
        (0.125, "#1e094f"),
        (0.250, "#3f0761"),
        (0.375, "#71176e"),
        (0.500, "#bd334e"),
        (0.625, "#e04f31"),
        (0.750, "#f98b0e"),
        (0.875, "#ebf377"),
        (1.000, "#ffffff"),
    ],
)

# Diverging colormap for sample A minus sample B signal-count differences.
# Colorbar top -> bottom: #702136, #ea8c70, #f7f6f7, #75b2d4, #092c57
SIGNAL_COUNT_DIFF_CMAP = LinearSegmentedColormap.from_list(
    "signal_count_diff",
    [
        (0.0, "#092c57"),
        (0.25, "#75b2d4"),
        (0.5, "#f7f6f7"),
        (0.75, "#ea8c70"),
        (1.0, "#702136"),
    ],
)


@dataclass(frozen=True)
class SliceHeatmapStyle:
    metric_name: str
    vmin: float
    vmax: float
    cmap_name: str = "white_orange_red_black"


@dataclass(frozen=True)
class AtlasSliceHeatmap:
    atlas_slice: AtlasSlice
    region_values: dict[int, float]
    style: SliceHeatmapStyle


def _normalize_plane(plane: str) -> Plane:
    value = str(plane).strip().lower()
    if value not in PLANE_TO_FIXED_AXIS:
        raise ValueError(f"plane must be one of {sorted(PLANE_TO_FIXED_AXIS)}, got: {plane}")
    return value  # type: ignore[return-value]


def _normalize_coordinate_system(coordinate_system: str) -> CoordinateSystem:
    value = str(coordinate_system).strip().lower().replace("_", "-")
    aliases = {
        "bregma": "bregma-mm",
        "bregma-mm": "bregma-mm",
        "ccf": "ccf-um",
        "ccf-um": "ccf-um",
        "index": "index",
        "slice-index": "index",
    }
    if value not in aliases:
        raise ValueError("coordinate_system must be 'bregma-mm', 'ccf-um', or 'index', got: " + str(coordinate_system))
    return aliases[value]  # type: ignore[return-value]


def _validate_spec(spec: AtlasSliceSpec) -> AtlasSliceSpec:
    plane = _normalize_plane(spec.plane)
    coordinate_system = _normalize_coordinate_system(spec.coordinate_system)
    if spec.atlas_resolution_um <= 0:
        raise ValueError(f"atlas_resolution_um must be positive, got: {spec.atlas_resolution_um}")
    if len(spec.bregma_index) != 3:
        raise ValueError(f"bregma_index must contain three values (DV, AP, ML), got: {spec.bregma_index}")
    return AtlasSliceSpec(
        plane=plane,
        coordinate_system=coordinate_system,
        coordinate=float(spec.coordinate),
        atlas_resolution_um=float(spec.atlas_resolution_um),
        bregma_index=tuple(int(value) for value in spec.bregma_index),
    )


def coordinate_to_index(spec: AtlasSliceSpec, shape: tuple[int, int, int]) -> int:
    """Convert a standardized coordinate to an atlas array index.

    ``shape`` must be the atlas volume shape in ``(DV, AP, ML)`` order.
    """

    spec = _validate_spec(spec)
    if len(shape) != 3:
        raise ValueError(f"atlas shape must be 3D (DV, AP, ML), got: {shape}")

    fixed_axis = PLANE_TO_FIXED_AXIS[spec.plane]
    coordinate = float(spec.coordinate)

    if spec.coordinate_system == "index":
        index = int(round(coordinate))
    elif spec.coordinate_system == "ccf-um":
        index = int(round(coordinate / spec.atlas_resolution_um))
    else:
        bregma_value = int(spec.bregma_index[fixed_axis])
        offset_voxels = coordinate * 1000.0 / spec.atlas_resolution_um
        if fixed_axis == 1:  # AP: anterior is positive, lower CCF AP index.
            index = int(round(bregma_value - offset_voxels))
        else:  # DV and ML increase with positive bregma coordinates in this array convention.
            index = int(round(bregma_value + offset_voxels))

    axis_size = int(shape[fixed_axis])
    if index < 0 or index >= axis_size:
        raise ValueError(
            "Coordinate is outside atlas bounds: "
            f"plane={spec.plane}, coordinate_system={spec.coordinate_system}, "
            f"coordinate={spec.coordinate}, index={index}, shape={tuple(shape)}"
        )
    return index


def _read_label_volume(label_path: str | Path) -> np.ndarray:
    label_path = Path(label_path)
    if not label_path.exists():
        raise FileNotFoundError(f"Atlas label file not found: {label_path}")
    try:
        volume = tifffile.memmap(str(label_path))
    except Exception:
        volume = tifffile.imread(str(label_path))
    if volume.ndim != 3:
        raise ValueError(f"Atlas label must be a 3D TIFF volume in (DV, AP, ML) order, got shape: {volume.shape}")
    return volume


def _slice_volume(volume: np.ndarray, plane: Plane, index: int) -> np.ndarray:
    if plane == "horizontal":
        image = volume[index, :, :]
    elif plane == "coronal":
        image = volume[:, index, :]
    else:
        image = volume[:, :, index]
    return np.asarray(image)


def _axis_mm_values(axis: int, size: int, *, resolution_um: float, bregma_index: tuple[int, int, int]) -> np.ndarray:
    indices = np.arange(size, dtype=np.float64)
    if axis == 1:  # AP: positive is anterior, opposite array index direction.
        return (float(bregma_index[axis]) - indices) * resolution_um / 1000.0
    return (indices - float(bregma_index[axis])) * resolution_um / 1000.0


def _slice_extent_mm(
    shape_2d: tuple[int, int],
    plane: Plane,
    *,
    resolution_um: float,
    bregma_index: tuple[int, int, int],
) -> tuple[tuple[float, float, float, float], str, str]:
    row_axis, col_axis = PLANE_TO_OUTPUT_AXES[plane]
    row_values = _axis_mm_values(row_axis, shape_2d[0], resolution_um=resolution_um, bregma_index=bregma_index)
    col_values = _axis_mm_values(col_axis, shape_2d[1], resolution_um=resolution_um, bregma_index=bregma_index)
    extent = (float(col_values[0]), float(col_values[-1]), float(row_values[-1]), float(row_values[0]))
    return extent, AXIS_NAMES[col_axis], AXIS_NAMES[row_axis]


def _coordinate_label(spec: AtlasSliceSpec, index: int) -> str:
    axis_name = AXIS_NAMES[PLANE_TO_FIXED_AXIS[spec.plane]]
    if spec.coordinate_system == "bregma-mm":
        return f"{spec.plane} {axis_name} {spec.coordinate:g} mm from bregma (index {index})"
    if spec.coordinate_system == "ccf-um":
        return f"{spec.plane} {axis_name} {spec.coordinate:g} um CCF (index {index})"
    return f"{spec.plane} {axis_name} index {index}"


def extract_atlas_slice(label_path: str | Path, spec: AtlasSliceSpec) -> AtlasSlice:
    """Extract a 2D atlas label slice from a 3D label TIFF."""

    spec = _validate_spec(spec)
    volume = _read_label_volume(label_path)
    index = coordinate_to_index(spec, tuple(int(value) for value in volume.shape))
    image = _slice_volume(volume, spec.plane, index)
    extent_mm, x_axis, y_axis = _slice_extent_mm(
        tuple(int(value) for value in image.shape),
        spec.plane,
        resolution_um=spec.atlas_resolution_um,
        bregma_index=spec.bregma_index,
    )

    return AtlasSlice(
        image=image.copy(),
        plane=spec.plane,
        index=index,
        coordinate_system=spec.coordinate_system,
        coordinate=spec.coordinate,
        coordinate_label=_coordinate_label(spec, index),
        atlas_resolution_um=spec.atlas_resolution_um,
        bregma_index=spec.bregma_index,
        extent_mm=extent_mm,
        x_axis=x_axis,
        y_axis=y_axis,
    )


def _parse_structure_id_path(path_text: str) -> list[int]:
    path_text = str(path_text).strip()
    if not path_text:
        return []
    if path_text.startswith("/"):
        return [int(part) for part in path_text.strip("/").split("/") if part]
    values = ast.literal_eval(path_text)
    return [int(value) for value in values]


def load_region_metadata(cfg_path: str | Path) -> pd.DataFrame:
    cfg_path = Path(cfg_path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Region CSV not found: {cfg_path}")

    region_df = pd.read_csv(cfg_path)
    required = {"id", "name", "structure_id_path"}
    missing = required.difference(region_df.columns)
    if missing:
        raise ValueError(f"Region CSV is missing required columns: {sorted(missing)}")

    rows = []
    for _, row in region_df.iterrows():
        rows.append(
            {
                "region_id": int(row["id"]),
                "excel_name": str(row["name"]),
                "structure_id_path": _parse_structure_id_path(row["structure_id_path"]),
            }
        )
    return pd.DataFrame(rows)


# Excel density columns are ratios of inclusive aggregates; rebuild from exclusive
# component columns when painting atlas label voxels.
DENSITY_METRIC_COMPONENTS: dict[str, tuple[str, str]] = {
    "Voxel Density": ("Signal Voxels", "Total Voxels"),
    "Left Voxel Density": ("Left Signal Voxels", "Left Total Voxels"),
    "Right Voxel Density": ("Right Signal Voxels", "Right Total Voxels"),
}


def load_excel_metric_by_name(input_excel: str | Path, metric: str) -> dict[str, float]:
    input_excel = Path(input_excel)
    if not input_excel.exists():
        raise FileNotFoundError(f"Input Excel not found: {input_excel}")

    sheets = pd.read_excel(input_excel, sheet_name=None)
    metric_by_name: dict[str, float] = {}
    found_metric = False
    for sheet_name, frame in sheets.items():
        if not str(sheet_name).startswith("Level_"):
            continue
        if "Name" not in frame.columns:
            continue
        if metric not in frame.columns:
            continue
        found_metric = True
        for _, row in frame.iterrows():
            name = str(row["Name"])
            value = pd.to_numeric(pd.Series([row[metric]]), errors="coerce").iloc[0]
            if pd.isna(value):
                continue
            metric_by_name[name] = float(value)

    if not found_metric:
        raise ValueError(f"Metric column not found in any Level_* sheet: {metric}")
    if not metric_by_name:
        raise ValueError(f"No numeric values found for metric '{metric}' in {input_excel}")
    return metric_by_name


def children_by_region_id(path_by_region_id: dict[int, list[int]]) -> dict[int, list[int]]:
    """Map each region to its direct children using Allen structure_id_path."""
    children: dict[int, list[int]] = {}
    for region_id, path in path_by_region_id.items():
        if len(path) < 2:
            continue
        parent_id = int(path[-2])
        children.setdefault(parent_id, []).append(int(region_id))
    return children


def to_direct_label_metric_values(
    value_by_region_id: dict[int, float],
    path_by_region_id: dict[int, list[int]],
) -> dict[int, float]:
    """Convert inclusive hierarchy aggregates to direct-label-only values.

    Analysis Excel rows store subtree sums (parent = own voxels + all descendants).
    Atlas voxels labeled as a parent (e.g. root interstitial gaps) must be painted
    with only that label's exclusive count, not the whole-brain total.
    """
    children = children_by_region_id(path_by_region_id)
    exclusive: dict[int, float] = {}
    for region_id, value in value_by_region_id.items():
        child_sum = 0.0
        for child_id in children.get(int(region_id), []):
            if child_id in value_by_region_id:
                child_sum += float(value_by_region_id[child_id])
        exclusive[int(region_id)] = max(0.0, float(value) - child_sum)
    return exclusive


def _match_metric_names_to_region_ids(
    metric_by_name: dict[str, float],
    region_table: pd.DataFrame,
) -> dict[int, float]:
    value_by_region_id: dict[int, float] = {}
    for _, row in region_table.iterrows():
        excel_name = str(row["excel_name"])
        if excel_name in metric_by_name:
            value_by_region_id[int(row["region_id"])] = float(metric_by_name[excel_name])
    return value_by_region_id


def build_region_metric_lookup(
    input_excel: str | Path,
    *,
    cfg_path: str | Path,
    metric: str,
    direct_label_only: bool = False,
) -> tuple[dict[int, float], dict[int, list[int]]]:
    """Build region_id -> metric lookup from analysis Excel.

    When ``direct_label_only`` is True, convert inclusive subtree aggregates into
    exclusive per-label values suitable for painting atlas voxels (so root gaps
    show only gap cFos, not the whole-brain sum).
    """
    region_table = load_region_metadata(cfg_path)
    path_by_region_id: dict[int, list[int]] = {
        int(row["region_id"]): [int(value) for value in row["structure_id_path"]]
        for _, row in region_table.iterrows()
    }

    if direct_label_only and metric in DENSITY_METRIC_COMPONENTS:
        numerator_metric, denominator_metric = DENSITY_METRIC_COMPONENTS[metric]
        numerator_by_id = _match_metric_names_to_region_ids(
            load_excel_metric_by_name(input_excel, numerator_metric),
            region_table,
        )
        denominator_by_id = _match_metric_names_to_region_ids(
            load_excel_metric_by_name(input_excel, denominator_metric),
            region_table,
        )
        numerator_by_id = to_direct_label_metric_values(numerator_by_id, path_by_region_id)
        denominator_by_id = to_direct_label_metric_values(denominator_by_id, path_by_region_id)
        value_by_region_id = {
            region_id: (
                float(numerator_by_id.get(region_id, 0.0)) / float(denominator_by_id[region_id])
                if float(denominator_by_id.get(region_id, 0.0)) > 0
                else 0.0
            )
            for region_id in set(numerator_by_id) | set(denominator_by_id)
        }
    else:
        metric_by_name = load_excel_metric_by_name(input_excel, metric)
        value_by_region_id = _match_metric_names_to_region_ids(metric_by_name, region_table)
        if direct_label_only:
            value_by_region_id = to_direct_label_metric_values(value_by_region_id, path_by_region_id)

    if not value_by_region_id:
        raise ValueError(f"Could not match any atlas regions to Excel values from {input_excel}")
    return value_by_region_id, path_by_region_id


def subtract_region_metric_values(
    minuend_by_region_id: dict[int, float],
    subtrahend_by_region_id: dict[int, float],
) -> dict[int, float]:
    """Return per-region metric values for minuend minus subtrahend."""
    region_ids = set(minuend_by_region_id) | set(subtrahend_by_region_id)
    return {
        int(region_id): float(minuend_by_region_id.get(region_id, 0.0) - subtrahend_by_region_id.get(region_id, 0.0))
        for region_id in region_ids
    }


def fold_change_region_metric_values(
    numerator_by_region_id: dict[int, float],
    denominator_by_region_id: dict[int, float],
    *,
    pseudocount: float = 1.0,
) -> dict[int, float]:
    """Return per-region log2 fold change: numerator / denominator."""
    region_ids = set(numerator_by_region_id) | set(denominator_by_region_id)
    pseudo = float(pseudocount)
    return {
        int(region_id): float(
            np.log2((float(numerator_by_region_id.get(region_id, 0.0)) + pseudo) / (float(denominator_by_region_id.get(region_id, 0.0)) + pseudo))
        )
        for region_id in region_ids
    }


def ratio_region_metric_values(
    numerator_by_region_id: dict[int, float],
    denominator_by_region_id: dict[int, float],
    *,
    pseudocount: float = 1.0,
) -> dict[int, float]:
    """Return per-region direct ratio: numerator / denominator (with pseudocount)."""
    region_ids = set(numerator_by_region_id) | set(denominator_by_region_id)
    pseudo = float(pseudocount)
    return {
        int(region_id): float(
            (float(numerator_by_region_id.get(region_id, 0.0)) + pseudo)
            / (float(denominator_by_region_id.get(region_id, 0.0)) + pseudo)
        )
        for region_id in region_ids
    }


def _as_integer_label_volume(label_volume: np.ndarray) -> np.ndarray:
    volume = np.asarray(label_volume)
    if np.issubdtype(volume.dtype, np.integer):
        return volume
    return np.rint(volume).astype(np.int32, copy=False)


def count_region_connected_components(
    label_volume: np.ndarray,
    *,
    connectivity: int = 26,
    progress_every: int = 0,
) -> dict[int, int]:
    """Count 3D connected components for each non-zero atlas region id.

    Uses a bounding-box crop per region so large atlases stay tractable.
    Prefers ``cc3d`` (26-connect); falls back to ``scipy.ndimage.label``.
    """
    try:
        import cc3d

        use_cc3d = True
    except ImportError:  # pragma: no cover - depends on environment
        cc3d = None
        use_cc3d = False

    if connectivity not in (6, 18, 26):
        raise ValueError(f"Unsupported connectivity={connectivity}; expected 6, 18, or 26")

    # scipy structure: 6-connect = faces only; 26-connect = all neighbors.
    if connectivity == 6:
        structure = ndi.generate_binary_structure(3, 1)
    else:
        structure = ndi.generate_binary_structure(3, 3)

    volume = _as_integer_label_volume(label_volume)
    region_ids = [int(value) for value in np.unique(volume) if int(value) != 0]
    counts: dict[int, int] = {}
    total = len(region_ids)
    for index, region_id in enumerate(region_ids, start=1):
        mask = volume == region_id
        if not np.any(mask):
            counts[region_id] = 0
            continue
        coords = np.where(mask)
        z0, z1 = int(coords[0].min()), int(coords[0].max()) + 1
        y0, y1 = int(coords[1].min()), int(coords[1].max()) + 1
        x0, x1 = int(coords[2].min()), int(coords[2].max()) + 1
        cropped = mask[z0:z1, y0:y1, x0:x1]
        if use_cc3d:
            labeled = cc3d.connected_components(
                cropped.astype(np.uint8, copy=False),
                connectivity=int(connectivity),
            )
            counts[region_id] = int(labeled.max())
        else:
            labeled, n_features = ndi.label(cropped, structure=structure)
            counts[region_id] = int(n_features)
        if progress_every > 0 and (index % progress_every == 0 or index == total):
            print(f"  CC progress: {index}/{total} regions", flush=True)
    return counts


def is_paired_region(n_cc: int) -> bool:
    """Paired when atlas has at least two 3D connected components."""
    return int(n_cc) >= 2


def _label_components(
    mask: np.ndarray,
    *,
    connectivity: int = 26,
) -> tuple[np.ndarray, int]:
    """Return (labeled, n_components) for a binary mask."""
    try:
        import cc3d

        labeled = cc3d.connected_components(mask.astype(np.uint8, copy=False), connectivity=int(connectivity))
        return labeled, int(labeled.max())
    except ImportError:  # pragma: no cover - depends on environment
        structure = ndi.generate_binary_structure(3, 1 if connectivity == 6 else 3)
        labeled, n_features = ndi.label(mask, structure=structure)
        return labeled, int(n_features)


def analyze_region_cc_geometry(
    label_volume: np.ndarray,
    *,
    ml_mid_index: int | None = None,
    connectivity: int = 26,
    resolution_um: float = 25.0,
    min_cc_voxels: int = 50,
    progress_every: int = 0,
    name_by_region_id: dict[int, str] | None = None,
) -> list[dict[str, object]]:
    """Classify multi-CC regions as LR-pair vs AP/DV-split (not true hemispheric pairs).

    Atlas axes are ``(DV, AP, ML)``. For each region with ``n_cc >= 2``, the two
    largest components are compared:

    - ``lr_pair``: centroids on opposite ML sides and ML separation dominates
    - ``ap_split``: AP separation dominates (front/back components), often midline-crossing
    - ``dv_split``: DV separation dominates
    - ``same_side``: both centroids on the same ML side
    - ``ambiguous``: mixed / weak geometry

    Returns one record per multi-CC region (sorted with suspects first).
    """
    volume = _as_integer_label_volume(label_volume)
    if ml_mid_index is None:
        ml_mid_index = int(volume.shape[2] // 2)
    ml_mid_index = int(ml_mid_index)
    res = float(resolution_um)
    min_cc_voxels = int(min_cc_voxels)

    region_ids = [int(value) for value in np.unique(volume) if int(value) != 0]
    records: list[dict[str, object]] = []
    total = len(region_ids)

    for index, region_id in enumerate(region_ids, start=1):
        mask = volume == region_id
        if not np.any(mask):
            continue
        coords = np.where(mask)
        z0, z1 = int(coords[0].min()), int(coords[0].max()) + 1
        y0, y1 = int(coords[1].min()), int(coords[1].max()) + 1
        x0, x1 = int(coords[2].min()), int(coords[2].max()) + 1
        cropped = mask[z0:z1, y0:y1, x0:x1]
        labeled, n_cc = _label_components(cropped, connectivity=connectivity)
        if n_cc < 2:
            continue

        cc_stats: list[dict[str, float | int | bool]] = []
        for cc_id in range(1, n_cc + 1):
            cc_mask = labeled == cc_id
            n_vox = int(cc_mask.sum())
            if n_vox < min_cc_voxels:
                continue
            zz, yy, xx = np.where(cc_mask)
            # Convert cropped indices back to full-volume coordinates.
            dv = zz.astype(np.float64) + z0
            ap = yy.astype(np.float64) + y0
            ml = xx.astype(np.float64) + x0
            left_n = int(np.sum(ml < ml_mid_index))
            right_n = int(np.sum(ml >= ml_mid_index))
            cc_stats.append(
                {
                    "cc_id": int(cc_id),
                    "n_voxels": n_vox,
                    "centroid_dv": float(dv.mean()),
                    "centroid_ap": float(ap.mean()),
                    "centroid_ml": float(ml.mean()),
                    "left_voxels": left_n,
                    "right_voxels": right_n,
                    "crosses_midline": bool(left_n > 0 and right_n > 0),
                    "ml_side": "left" if float(ml.mean()) < ml_mid_index else "right",
                }
            )

        if len(cc_stats) < 2:
            # All but one CC were dust below min_cc_voxels.
            continue

        cc_stats.sort(key=lambda item: int(item["n_voxels"]), reverse=True)
        a = cc_stats[0]
        b = cc_stats[1]
        d_dv = abs(float(a["centroid_dv"]) - float(b["centroid_dv"])) * res
        d_ap = abs(float(a["centroid_ap"]) - float(b["centroid_ap"])) * res
        d_ml = abs(float(a["centroid_ml"]) - float(b["centroid_ml"])) * res
        opposite_sides = str(a["ml_side"]) != str(b["ml_side"])
        either_crosses = bool(a["crosses_midline"] or b["crosses_midline"])
        region_left = int(np.sum(coords[2] < ml_mid_index))
        region_right = int(np.sum(coords[2] >= ml_mid_index))
        region_crosses = region_left > 0 and region_right > 0

        dominant = max(d_dv, d_ap, d_ml)
        if opposite_sides and d_ml >= d_ap and d_ml >= d_dv and d_ml >= (0.5 * dominant if dominant > 0 else 0):
            geometry = "lr_pair"
        elif d_ap >= d_ml and d_ap >= d_dv:
            geometry = "ap_split"
        elif d_dv >= d_ml and d_dv >= d_ap:
            geometry = "dv_split"
        elif not opposite_sides:
            geometry = "same_side"
        else:
            geometry = "ambiguous"

        # Suspect: currently treated as paired by n_cc, but unlikely true L/R pair.
        suspect_force_unpaired = geometry in {"ap_split", "dv_split", "same_side"} or (
            geometry == "ambiguous" and (either_crosses or not opposite_sides)
        )

        record: dict[str, object] = {
            "region_id": int(region_id),
            "name": str(name_by_region_id.get(region_id, "")) if name_by_region_id else "",
            "n_cc": int(n_cc),
            "n_cc_kept": len(cc_stats),
            "geometry": geometry,
            "suspect_force_unpaired": bool(suspect_force_unpaired),
            "delta_dv_um": round(d_dv, 1),
            "delta_ap_um": round(d_ap, 1),
            "delta_ml_um": round(d_ml, 1),
            "opposite_ml_sides": bool(opposite_sides),
            "either_cc_crosses_midline": bool(either_crosses),
            "region_crosses_midline": bool(region_crosses),
            "cc1_voxels": int(a["n_voxels"]),
            "cc2_voxels": int(b["n_voxels"]),
            "cc1_ml_side": str(a["ml_side"]),
            "cc2_ml_side": str(b["ml_side"]),
            "cc1_crosses_midline": bool(a["crosses_midline"]),
            "cc2_crosses_midline": bool(b["crosses_midline"]),
        }
        records.append(record)

        if progress_every > 0 and (index % progress_every == 0 or index == total):
            print(f"  geometry progress: {index}/{total} regions", flush=True)

    def _sort_key(item: dict[str, object]) -> tuple:
        return (
            0 if item.get("suspect_force_unpaired") else 1,
            str(item.get("geometry") or ""),
            -float(item.get("delta_ap_um") or 0.0),
            int(item.get("region_id") or 0),
        )

    records.sort(key=_sort_key)
    return records


def find_midplane_bisected_paired_regions(
    label_volume: np.ndarray,
    paired_by_region_id: dict[int, bool],
    *,
    ml_mid_index: int | None = None,
    min_region_slice_pixels: int = 20,
    min_side_fraction: float = 0.15,
    min_bisected_fraction: float = 0.15,
    min_bisected_slices: int = 3,
    progress_every: int = 0,
    name_by_region_id: dict[int, str] | None = None,
) -> list[dict[str, object]]:
    """Detect paired regions that are often one continuous blob across the midplane.

    Current heatmap painting uses a geometric ML cut: left columns get ``A_L/B_L``,
    right columns get ``A_R/B_R``. That looks artificial when a paired region appears
    on a coronal slice as a single 2D connected component spanning both hemispheres.

    A coronal slice is counted as bisected when any 2D CC of the region has both
    sides of the midplane occupied above ``min_side_fraction``.
    """
    volume = _as_integer_label_volume(label_volume)
    if ml_mid_index is None:
        ml_mid_index = int(volume.shape[2] // 2)
    ml_mid_index = int(ml_mid_index)
    structure_2d = ndi.generate_binary_structure(2, 2)

    paired_ids = sorted(int(rid) for rid, paired in paired_by_region_id.items() if paired)
    records: list[dict[str, object]] = []
    total = len(paired_ids)
    n_ap = int(volume.shape[1])

    for index, region_id in enumerate(paired_ids, start=1):
        mask3d = volume == region_id
        if not np.any(mask3d):
            continue
        ap_indices = np.flatnonzero(np.any(mask3d, axis=(0, 2)))
        n_visible = 0
        n_bisected = 0
        max_cross_frac = 0.0
        for ap in ap_indices:
            slice2d = mask3d[:, int(ap), :]
            n_pix = int(slice2d.sum())
            if n_pix < int(min_region_slice_pixels):
                continue
            n_visible += 1
            labeled, n_cc = ndi.label(slice2d, structure=structure_2d)
            slice_bisected = False
            for cc_id in range(1, int(n_cc) + 1):
                cc = labeled == cc_id
                cols = np.where(cc)[1]
                if cols.size == 0:
                    continue
                left_n = int(np.sum(cols < ml_mid_index))
                right_n = int(np.sum(cols >= ml_mid_index))
                cc_n = left_n + right_n
                if cc_n <= 0:
                    continue
                left_frac = left_n / cc_n
                right_frac = right_n / cc_n
                if left_frac >= float(min_side_fraction) and right_frac >= float(min_side_fraction):
                    slice_bisected = True
                    max_cross_frac = max(max_cross_frac, min(left_frac, right_frac))
            if slice_bisected:
                n_bisected += 1

        if n_visible <= 0:
            continue
        frac = float(n_bisected) / float(n_visible)
        flagged = n_bisected >= int(min_bisected_slices) and frac >= float(min_bisected_fraction)
        if not flagged:
            continue
        records.append(
            {
                "region_id": int(region_id),
                "name": str(name_by_region_id.get(region_id, "")) if name_by_region_id else "",
                "n_visible_coronal_slices": int(n_visible),
                "n_bisected_slices": int(n_bisected),
                "bisected_fraction": round(frac, 4),
                "max_minor_side_fraction": round(float(max_cross_frac), 4),
                "suggest_force_unpaired": True,
                "reason": "paired_but_often_one_2d_blob_crossing_midplane",
            }
        )
        if progress_every > 0 and (index % progress_every == 0 or index == total):
            print(f"  midplane-bisect progress: {index}/{total} paired regions", flush=True)

    records.sort(key=lambda row: (-float(row["bisected_fraction"]), -int(row["n_bisected_slices"]), int(row["region_id"])))
    return records


def paint_hemisphere_ratio_slice(
    label_slice: np.ndarray,
    left_values: dict[int, float],
    right_values: dict[int, float],
    paired_by_region_id: dict[int, bool],
    *,
    ml_mid_index: int,
    min_side_fraction: float = 0.15,
) -> np.ndarray:
    """Paint A/B ratios without geometrically sawing continuous midplane-crossing blobs.

    - Unpaired / same L&R value: whole region one color
    - Paired with separate L/R 2D blobs: each blob colored by its centroid side
    - Paired but a 2D blob spans both hemispheres: keep one color (mean of L/R)
      so the midplane does not invent a hard color cut through continuous tissue
    """
    labels = np.asarray(label_slice)
    painted = np.full(labels.shape, np.nan, dtype=np.float32)
    ml_mid_index = int(max(ml_mid_index, 0))
    structure_2d = ndi.generate_binary_structure(2, 2)

    for region_id in np.unique(labels):
        rid = int(region_id)
        if rid == 0:
            continue
        if rid not in left_values and rid not in right_values:
            continue
        left_val = float(left_values.get(rid, right_values.get(rid, 0.0)))
        right_val = float(right_values.get(rid, left_values.get(rid, 0.0)))
        mask = labels == rid
        paired = bool(paired_by_region_id.get(rid, False))
        if (not paired) or abs(left_val - right_val) <= 1e-12:
            painted[mask] = left_val
            continue

        labeled, n_cc = ndi.label(mask, structure=structure_2d)
        for cc_id in range(1, int(n_cc) + 1):
            cc = labeled == cc_id
            cols = np.where(cc)[1]
            if cols.size == 0:
                continue
            left_n = int(np.sum(cols < ml_mid_index))
            right_n = int(np.sum(cols >= ml_mid_index))
            total = left_n + right_n
            if total <= 0:
                continue
            left_frac = left_n / total
            right_frac = right_n / total
            if left_frac >= float(min_side_fraction) and right_frac >= float(min_side_fraction):
                painted[cc] = 0.5 * (left_val + right_val)
            elif float(cols.mean()) < ml_mid_index:
                painted[cc] = left_val
            else:
                painted[cc] = right_val

    inside_brain = labels > 0
    painted[inside_brain & np.isnan(painted)] = 0.0
    return painted


def default_region_pairing_reference_path() -> Path:
    """Preferred cache path next to the resolved Allen atlas label."""
    try:
        from pipeline_modules.utils.data_paths import get_yifu_data_dir

        data_dir = get_yifu_data_dir(required=False)
        if data_dir is not None:
            return Path(data_dir) / "reference" / "region_pairing.json"
    except Exception:
        pass
    return Path(__file__).resolve().parents[2] / "data" / "reference" / "region_pairing.json"


def build_region_pairing_records(
    n_cc_by_region_id: dict[int, int],
    *,
    name_by_region_id: dict[int, str] | None = None,
) -> dict[str, dict[str, object]]:
    """Serialize region_id -> {n_cc, paired, optional name} for JSON."""
    records: dict[str, dict[str, object]] = {}
    for region_id, n_cc in sorted((int(rid), int(count)) for rid, count in n_cc_by_region_id.items()):
        record: dict[str, object] = {
            "n_cc": int(n_cc),
            "paired": bool(is_paired_region(n_cc)),
        }
        if name_by_region_id and region_id in name_by_region_id:
            record["name"] = str(name_by_region_id[region_id])
        records[str(region_id)] = record
    return records


def save_region_pairing_reference(
    output_path: str | Path,
    n_cc_by_region_id: dict[int, int],
    *,
    atlas_label: str | Path | None = None,
    connectivity: int = 26,
    name_by_region_id: dict[int, str] | None = None,
    extra_meta: dict[str, object] | None = None,
) -> Path:
    """Write a reusable region pairing reference JSON."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object] = {
        "atlas_label": str(atlas_label) if atlas_label is not None else "",
        "connectivity": int(connectivity),
        "paired_rule": "n_cc >= 2",
        "regions": build_region_pairing_records(n_cc_by_region_id, name_by_region_id=name_by_region_id),
    }
    if extra_meta:
        payload.update(extra_meta)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    return output_path


def load_region_pairing_reference(path: str | Path) -> tuple[dict[int, int], dict[int, bool]]:
    """Load ``region_id -> n_cc`` and ``region_id -> paired`` from a reference JSON."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Region pairing reference not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    n_cc_by_region_id: dict[int, int] = {}
    paired_by_region_id: dict[int, bool] = {}

    if isinstance(payload, dict) and "regions" in payload:
        regions = payload["regions"]
    else:
        regions = payload

    if not isinstance(regions, dict):
        raise ValueError(f"Invalid region pairing reference format: {path}")

    for key, value in regions.items():
        region_id = int(key)
        if isinstance(value, dict):
            if "n_cc" in value:
                n_cc = int(value["n_cc"])
            elif "paired" in value:
                n_cc = 2 if bool(value["paired"]) else 1
            else:
                raise ValueError(f"Region {region_id} entry missing n_cc/paired in {path}")
            paired = bool(value["paired"]) if "paired" in value else is_paired_region(n_cc)
        else:
            n_cc = int(value)
            paired = is_paired_region(n_cc)
        n_cc_by_region_id[region_id] = n_cc
        paired_by_region_id[region_id] = paired
    return n_cc_by_region_id, paired_by_region_id


def build_hemisphere_ab_ratio_lookups(
    *,
    a_left: dict[int, float],
    a_right: dict[int, float],
    b_left: dict[int, float],
    b_right: dict[int, float],
    n_cc_by_region_id: dict[int, int] | None = None,
    paired_by_region_id: dict[int, bool] | None = None,
    pseudocount: float = 1.0,
) -> tuple[dict[int, float], dict[int, float]]:
    """Build left/right A/B ratio maps using paired vs midline (single-CC) rules.

    - Paired (``n_cc >= 2`` or ``paired=True``): left = A_L/B_L, right = A_R/B_R
    - Unpaired (single CC): left = right = (A_L+A_R)/(B_L+B_R)
    - Missing from pairing maps: treated as unpaired (conservative whole-region color)
    """
    pseudo = float(pseudocount)
    region_ids = set(a_left) | set(a_right) | set(b_left) | set(b_right)
    if n_cc_by_region_id:
        region_ids |= set(n_cc_by_region_id)
    if paired_by_region_id:
        region_ids |= set(paired_by_region_id)

    left_values: dict[int, float] = {}
    right_values: dict[int, float] = {}
    for region_id in region_ids:
        rid = int(region_id)
        if paired_by_region_id is not None and rid in paired_by_region_id:
            paired = bool(paired_by_region_id[rid])
        elif n_cc_by_region_id is not None and rid in n_cc_by_region_id:
            paired = is_paired_region(n_cc_by_region_id[rid])
        else:
            paired = False

        a_l = float(a_left.get(rid, 0.0))
        a_r = float(a_right.get(rid, 0.0))
        b_l = float(b_left.get(rid, 0.0))
        b_r = float(b_right.get(rid, 0.0))

        if paired:
            left_values[rid] = (a_l + pseudo) / (b_l + pseudo)
            right_values[rid] = (a_r + pseudo) / (b_r + pseudo)
        else:
            whole = (a_l + a_r + pseudo) / (b_l + b_r + pseudo)
            left_values[rid] = whole
            right_values[rid] = whole
    return left_values, right_values


def compute_ratio_color_limits(
    values: list[float] | tuple[float, ...],
    *,
    percentile: float = 99.0,
    explicit_vmax: float | None = None,
    center: float = 1.0,
) -> tuple[float, float]:
    """Symmetric multiplicative limits around ``center`` (default 1.0 for A/B)."""
    center = float(center)
    if explicit_vmax is not None:
        upper = abs(float(explicit_vmax))
        if upper <= center:
            upper = center * 2.0 if center > 0 else center + 1.0
        lower = (center * center / upper) if center > 0 else center - (upper - center)
        if lower <= 0:
            lower = min(center * 0.5, upper * 0.25) if center > 0 else lower
        return float(lower), float(upper)

    finite = [float(value) for value in values if np.isfinite(value) and float(value) > 0]
    if not finite:
        return 0.5, 2.0
    factors = [max(value / center, center / value) for value in finite]
    factor = float(np.percentile(factors, float(percentile)))
    if not np.isfinite(factor) or factor < 1.0:
        factor = max(factors) if factors else 2.0
    if factor < 1.0 + 1e-6:
        factor = 1.0 + 1e-6
    lower = center / factor
    upper = center * factor
    return float(lower), float(upper)


def paint_hemisphere_split_slice(
    label_slice: np.ndarray,
    left_values: dict[int, float],
    right_values: dict[int, float],
    path_by_region_id: dict[int, list[int]],
    *,
    ml_mid_index: int,
) -> np.ndarray:
    """Paint left/right Excel metrics on the ML-left / ML-right sides of a coronal slice."""
    from pipeline_modules.visualization.heatmap import _paint_region_values_on_slice

    labels = np.asarray(label_slice)
    left_painted = _paint_region_values_on_slice(labels, left_values)
    right_painted = _paint_region_values_on_slice(labels, right_values)
    ml_mid_index = int(max(ml_mid_index, 0))
    combined = np.full(labels.shape, np.nan, dtype=np.float32)
    combined[:, :ml_mid_index] = left_painted[:, :ml_mid_index]
    combined[:, ml_mid_index:] = right_painted[:, ml_mid_index:]
    inside_brain = labels > 0
    combined[inside_brain & np.isnan(combined)] = 0.0
    return combined


def paint_lr_sample_split_slice(
    label_slice: np.ndarray,
    sample_a_values: dict[int, float],
    sample_b_values: dict[int, float],
    *,
    ml_mid_index: int,
) -> np.ndarray:
    """Paint sample A on ML-left and sample B on ML-right within one coronal slice."""
    from pipeline_modules.visualization.heatmap import _paint_region_values_on_slice

    labels = np.asarray(label_slice)
    painted_a = _paint_region_values_on_slice(labels, sample_a_values)
    painted_b = _paint_region_values_on_slice(labels, sample_b_values)
    ml_mid_index = int(max(ml_mid_index, 0))
    combined = np.full(labels.shape, np.nan, dtype=np.float32)
    combined[:, :ml_mid_index] = painted_a[:, :ml_mid_index]
    combined[:, ml_mid_index:] = painted_b[:, ml_mid_index:]
    inside_brain = labels > 0
    combined[inside_brain & np.isnan(combined)] = 0.0
    return combined


def compute_symmetric_metric_limits(
    values: list[float] | tuple[float, ...],
    *,
    percentile: float = 99.5,
    explicit_vmax: float | None = None,
) -> tuple[float, float]:
    """Symmetric vmin/vmax for diverging difference maps centered at zero."""
    if explicit_vmax is not None:
        limit = abs(float(explicit_vmax))
        if limit <= 0:
            limit = 1e-6
        return -limit, limit
    finite = [float(value) for value in values if np.isfinite(value)]
    if not finite:
        return -1.0, 1.0
    positive = [abs(value) for value in finite if abs(value) > 0]
    if not positive:
        return -1.0, 1.0
    limit = float(np.percentile(positive, float(percentile)))
    if limit <= 0:
        limit = max(positive)
    if limit <= 0:
        limit = 1e-6
    return -limit, limit


def lookup_region_metric_value(
    region_id: int,
    value_by_region_id: dict[int, float],
    path_by_region_id: dict[int, list[int]],
    *,
    inherit_ancestors: bool = True,
) -> float | None:
    """Return direct or inherited metric value, or None when no data exists."""
    if region_id in value_by_region_id:
        return float(value_by_region_id[region_id])
    if not inherit_ancestors:
        return None
    path = path_by_region_id.get(region_id, [])
    for ancestor_id in reversed(path[:-1]):
        if ancestor_id in value_by_region_id:
            return float(value_by_region_id[ancestor_id])
    return None


def collect_regions_missing_metric_data(
    label_image: np.ndarray,
    value_by_region_id: dict[int, float],
    path_by_region_id: dict[int, list[int]],
    *,
    inherit_ancestors: bool = True,
) -> list[int]:
    """Brain region ids visible on the slice that have no metric data."""
    missing: list[int] = []
    for label in np.unique(np.asarray(label_image)):
        region_id = int(label)
        if region_id == 0:
            continue
        if (
            lookup_region_metric_value(
                region_id,
                value_by_region_id,
                path_by_region_id,
                inherit_ancestors=inherit_ancestors,
            )
            is None
        ):
            missing.append(region_id)
    return missing


def build_region_name_lookup(cfg_path: str | Path) -> dict[int, str]:
    region_table = load_region_metadata(cfg_path)
    return {int(row["region_id"]): str(row["excel_name"]) for _, row in region_table.iterrows()}


def resolve_slice_region_values(
    label_image: np.ndarray,
    value_by_region_id: dict[int, float],
    path_by_region_id: dict[int, list[int]],
    *,
    inherit_ancestors: bool = True,
) -> dict[int, float]:
    resolved: dict[int, float] = {}
    for label in np.unique(np.asarray(label_image)):
        region_id = int(label)
        if region_id == 0:
            continue
        value = lookup_region_metric_value(
            region_id,
            value_by_region_id,
            path_by_region_id,
            inherit_ancestors=inherit_ancestors,
        )
        resolved[region_id] = 0.0 if value is None else float(value)
    return resolved


def build_atlas_slice_heatmap(
    atlas_slice: AtlasSlice,
    *,
    input_excel: str | Path,
    cfg_path: str | Path,
    metric: str = "Voxel Density",
    vmin: float = 0.0,
    vmax: float | None = None,
    cmap_name: str = "white_orange_red_black",
) -> AtlasSliceHeatmap:
    value_by_region_id, path_by_region_id = build_region_metric_lookup(
        input_excel,
        cfg_path=cfg_path,
        metric=metric,
        direct_label_only=True,
    )
    region_values = resolve_slice_region_values(
        atlas_slice.image,
        value_by_region_id,
        path_by_region_id,
        inherit_ancestors=False,
    )
    upper = float(vmax) if vmax is not None else max(value_by_region_id.values())
    if upper <= float(vmin):
        upper = float(vmin) + 1e-12
    display_metric_name = "Signal Density" if metric == "Voxel Density" else metric
    return AtlasSliceHeatmap(
        atlas_slice=atlas_slice,
        region_values=region_values,
        style=SliceHeatmapStyle(metric_name=display_metric_name, vmin=float(vmin), vmax=upper, cmap_name=cmap_name),
    )


def _boundary_segments(label_image: np.ndarray, *, include_outer: bool) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    labels = np.asarray(label_image)
    height, width = labels.shape
    segments: list[tuple[tuple[float, float], tuple[float, float]]] = []

    horizontal_diff = labels[:-1, :] != labels[1:, :]
    if include_outer:
        horizontal_keep = horizontal_diff & ((labels[:-1, :] > 0) | (labels[1:, :] > 0))
    else:
        horizontal_keep = horizontal_diff & (labels[:-1, :] > 0) & (labels[1:, :] > 0)
    for row, col in np.argwhere(horizontal_keep):
        y = float(row) + 0.5
        segments.append(((float(col) - 0.5, y), (float(col) + 0.5, y)))

    vertical_diff = labels[:, :-1] != labels[:, 1:]
    if include_outer:
        vertical_keep = vertical_diff & ((labels[:, :-1] > 0) | (labels[:, 1:] > 0))
    else:
        vertical_keep = vertical_diff & (labels[:, :-1] > 0) & (labels[:, 1:] > 0)
    for row, col in np.argwhere(vertical_keep):
        x = float(col) + 0.5
        segments.append(((x, float(row) - 0.5), (x, float(row) + 0.5)))

    if include_outer:
        top_cols = np.flatnonzero(labels[0, :] > 0)
        for col in top_cols:
            segments.append(((float(col) - 0.5, -0.5), (float(col) + 0.5, -0.5)))
        bottom_cols = np.flatnonzero(labels[-1, :] > 0)
        for col in bottom_cols:
            y = float(height) - 0.5
            segments.append(((float(col) - 0.5, y), (float(col) + 0.5, y)))
        left_rows = np.flatnonzero(labels[:, 0] > 0)
        for row in left_rows:
            segments.append(((-0.5, float(row) - 0.5), (-0.5, float(row) + 0.5)))
        right_rows = np.flatnonzero(labels[:, -1] > 0)
        for row in right_rows:
            x = float(width) - 0.5
            segments.append(((x, float(row) - 0.5), (x, float(row) + 0.5)))

    return segments


def _add_segments(
    ax: plt.Axes,
    segments: list[tuple[tuple[float, float], tuple[float, float]]],
    *,
    linewidth: float,
    alpha: float = 1.0,
) -> None:
    if not segments or linewidth <= 0:
        return
    collection = LineCollection(
        segments,
        colors="white",
        linewidths=linewidth,
        alpha=alpha,
        antialiaseds=True,
        capstyle="butt",
        joinstyle="miter",
    )
    ax.add_collection(collection)


def _smooth_contour(contour: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0 or len(contour) < 5:
        return contour
    closed = np.linalg.norm(contour[0] - contour[-1]) < 1.5
    mode = "wrap" if closed else "nearest"
    smoothed = np.empty_like(contour, dtype=np.float64)
    smoothed[:, 0] = ndi.gaussian_filter1d(contour[:, 0].astype(np.float64), sigma=sigma, mode=mode)
    smoothed[:, 1] = ndi.gaussian_filter1d(contour[:, 1].astype(np.float64), sigma=sigma, mode=mode)
    if closed:
        smoothed[-1] = smoothed[0]
    return smoothed


def _mask_contour_lines(mask: np.ndarray, *, smoothing: float, min_points: int = 8) -> list[np.ndarray]:
    padded = np.pad(mask.astype(np.float32), 1, mode="constant", constant_values=0)
    lines: list[np.ndarray] = []
    for contour in find_contours(padded, level=0.5, fully_connected="high"):
        if len(contour) < min_points:
            continue
        contour = contour - 1.0
        contour = _smooth_contour(contour, smoothing)
        lines.append(np.column_stack([contour[:, 1], contour[:, 0]]))
    return lines


def _label_contour_lines(
    label_image: np.ndarray,
    *,
    smoothing: float,
    min_region_pixels: int = 6,
) -> list[np.ndarray]:
    labels = np.asarray(label_image)
    lines: list[np.ndarray] = []
    for label in np.unique(labels):
        if label == 0:
            continue
        mask = labels == label
        if int(np.count_nonzero(mask)) < min_region_pixels:
            continue
        lines.extend(_mask_contour_lines(mask, smoothing=smoothing))
    return lines


def _add_lines(
    ax: plt.Axes,
    lines: list[np.ndarray],
    *,
    linewidth: float,
    alpha: float = 1.0,
) -> None:
    if not lines or linewidth <= 0:
        return
    collection = LineCollection(
        lines,
        colors="white",
        linewidths=linewidth,
        alpha=alpha,
        antialiaseds=True,
        capstyle="round",
        joinstyle="round",
    )
    ax.add_collection(collection)


def _contour_to_svg_d(contour: np.ndarray) -> str | None:
    if len(contour) < 2:
        return None

    points = np.asarray(contour, dtype=np.float64)
    closed = np.linalg.norm(points[0] - points[-1]) < 1.5
    if closed and len(points) > 2:
        points = points[:-1]

    if len(points) < 2:
        return None
    if len(points) < 4:
        parts = [f"M {points[0,0]:g},{points[0,1]:g}"]
        for point in points[1:]:
            parts.append(f"L {point[0]:g},{point[1]:g}")
        if closed:
            parts.append("Z")
        return " ".join(parts)

    parts = [f"M {points[0,0]:g},{points[0,1]:g}"]
    count = len(points)
    segment_count = count if closed else count - 1
    for i in range(segment_count):
        p0 = points[(i - 1) % count] if closed else points[max(i - 1, 0)]
        p1 = points[i]
        p2 = points[(i + 1) % count]
        p3 = points[(i + 2) % count] if closed else points[min(i + 2, count - 1)]

        c1 = p1 + (p2 - p0) / 6.0
        c2 = p2 - (p3 - p1) / 6.0
        parts.append(
            f"C {c1[0]:g},{c1[1]:g} {c2[0]:g},{c2[1]:g} {p2[0]:g},{p2[1]:g}"
        )

    if closed:
        parts.append("Z")
    return " ".join(parts)


def _contours_to_svg_d(contours: list[np.ndarray]) -> list[str]:
    paths: list[str] = []
    for contour in contours:
        path = _contour_to_svg_d(contour)
        if path:
            paths.append(path)
    return paths


def _colormap_by_name(name: str):
    if name == "white_orange_red_black":
        return WHITE_ORANGE_RED_BLACK_CMAP
    if name == "white_blue_red":
        return WHITE_BLUE_RED_CMAP
    if name == "signal_count_diff":
        return SIGNAL_COUNT_DIFF_CMAP
    cmap = plt.get_cmap(name)
    if not isinstance(cmap, LinearSegmentedColormap):
        return LinearSegmentedColormap.from_list(name, cmap(np.linspace(0.0, 1.0, 256)))
    return cmap


def _format_svg_color(value: tuple[float, float, float, float]) -> str:
    return mcolors.to_hex(value, keep_alpha=False)


def _build_region_fill_paths(
    label_image: np.ndarray,
    region_values: dict[int, float],
) -> list[tuple[str, float]]:
    outputs: list[tuple[str, float]] = []
    labels = np.asarray(label_image)
    for region_id, value in sorted(region_values.items()):
        mask = labels == int(region_id)
        if not np.any(mask):
            continue
        contours = _mask_contour_lines(mask, smoothing=0.0)
        for path_d in _contours_to_svg_d(contours):
            outputs.append((path_d, float(value)))
    return outputs


def _render_svg(
    output_path: Path,
    width: int,
    height: int,
    brain_d_paths: list[str],
    region_d_paths: list[str],
    *,
    line_width: float = 0.3,
    brain_outline_width: float = 0.0,
) -> Path:
    root = ET.Element(f"{{{SVG_NS}}}svg", {
        "width": str(width),
        "height": str(height),
        "viewBox": f"0 0 {width} {height}",
    })
    ET.SubElement(root, f"{{{SVG_NS}}}rect", {
        "x": "0", "y": "0",
        "width": str(width), "height": str(height),
        "fill": "black",
    })

    g = ET.SubElement(root, f"{{{SVG_NS}}}g", {
        "fill": "none",
        "stroke": "white",
        "stroke-linecap": "butt",
        "stroke-linejoin": "miter",
        "vector-effect": "non-scaling-stroke",
    })

    _line_width = float(line_width)
    if _line_width > 0 and region_d_paths:
        region_g = ET.SubElement(g, f"{{{SVG_NS}}}g", {
            "stroke-width": f"{_line_width:g}",
        })
        for d_str in region_d_paths:
            ET.SubElement(region_g, f"{{{SVG_NS}}}path", {"d": d_str})

    _brain_width = float(brain_outline_width)
    if _brain_width > 0 and brain_d_paths:
        brain_g = ET.SubElement(g, f"{{{SVG_NS}}}g", {
            "stroke-width": f"{_brain_width:g}",
        })
        for d_str in brain_d_paths:
            ET.SubElement(brain_g, f"{{{SVG_NS}}}path", {"d": d_str})

    tree = ET.ElementTree(root)
    tree.write(output_path, encoding="utf-8", xml_declaration=True)
    return output_path


def render_atlas_slice_heatmap(
    atlas_heatmap: AtlasSliceHeatmap,
    output_path: str | Path,
    *,
    line_width: float = 0.3,
    brain_outline_width: float = 0.0,
    colorbar_width: int = 48,
    colorbar_padding: int = 14,
    font_size: int = 9,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    image = np.asarray(atlas_heatmap.atlas_slice.image)
    if image.ndim != 2:
        raise ValueError(f"AtlasSlice.image must be 2D, got shape: {image.shape}")
    if output_path.suffix.lower() != ".svg":
        raise ValueError("Heatmap rendering currently supports SVG output only.")

    height, width = image.shape
    right_margin = 40
    total_width = width + colorbar_padding + colorbar_width + right_margin
    root = ET.Element(f"{{{SVG_NS}}}svg", {
        "width": str(total_width),
        "height": str(height),
        "viewBox": f"0 0 {total_width} {height}",
    })
    ET.SubElement(root, f"{{{SVG_NS}}}rect", {
        "x": "0", "y": "0",
        "width": str(total_width), "height": str(height),
        "fill": "black",
    })

    defs = ET.SubElement(root, f"{{{SVG_NS}}}defs")
    gradient = ET.SubElement(defs, f"{{{SVG_NS}}}linearGradient", {
        "id": "colorbar-gradient",
        "x1": "0%", "y1": "100%",
        "x2": "0%", "y2": "0%",
    })
    cmap = _colormap_by_name(atlas_heatmap.style.cmap_name)
    colorbar_stops = (("0%", 0.0, "white"), ("25%", 0.25, "orange"), ("50%", 0.5, "red"), ("100%", 1.0, "black"))
    for offset, value, _ in colorbar_stops:
        ET.SubElement(gradient, f"{{{SVG_NS}}}stop", {
            "offset": offset,
            "stop-color": _format_svg_color(cmap(value)),
        })

    fill_paths = _build_region_fill_paths(image, atlas_heatmap.region_values)
    norm = mcolors.Normalize(vmin=atlas_heatmap.style.vmin, vmax=atlas_heatmap.style.vmax, clip=True)

    fill_group = ET.SubElement(root, f"{{{SVG_NS}}}g")
    for path_d, value in fill_paths:
        ET.SubElement(fill_group, f"{{{SVG_NS}}}path", {
            "d": path_d,
            "fill": _format_svg_color(cmap(norm(value))),
            "stroke": "none",
        })

    brain_mask = image > 0
    brain_d_paths: list[str] = []
    region_d_paths: list[str] = []
    if np.any(brain_mask):
        brain_lines = _mask_contour_lines(brain_mask, smoothing=1.8)
        brain_d_paths = _contours_to_svg_d(brain_lines)
    region_lines = _label_contour_lines(image, smoothing=1.8)
    region_d_paths = _contours_to_svg_d(region_lines)

    outline_group = ET.SubElement(root, f"{{{SVG_NS}}}g", {
        "fill": "none",
        "stroke": "white",
        "stroke-linecap": "round",
        "stroke-linejoin": "round",
        "vector-effect": "non-scaling-stroke",
    })
    if float(line_width) > 0 and region_d_paths:
        region_g = ET.SubElement(outline_group, f"{{{SVG_NS}}}g", {"stroke-width": f"{float(line_width):g}"})
        for d_str in region_d_paths:
            ET.SubElement(region_g, f"{{{SVG_NS}}}path", {"d": d_str})
    if float(brain_outline_width) > 0 and brain_d_paths:
        brain_g = ET.SubElement(outline_group, f"{{{SVG_NS}}}g", {"stroke-width": f"{float(brain_outline_width):g}"})
        for d_str in brain_d_paths:
            ET.SubElement(brain_g, f"{{{SVG_NS}}}path", {"d": d_str})

    bar_x = width + colorbar_padding
    bar_width = 14
    bar_height = max(int(height * 0.58), 20)
    bar_y = max(int((height - bar_height) / 2), 10)
    ET.SubElement(root, f"{{{SVG_NS}}}rect", {
        "x": str(bar_x),
        "y": str(bar_y),
        "width": str(bar_width),
        "height": str(bar_height),
        "fill": "url(#colorbar-gradient)",
        "stroke": "white",
        "stroke-width": "0.8",
    })

    text_group = ET.SubElement(root, f"{{{SVG_NS}}}g", {
        "fill": "white",
        "font-size": str(font_size),
        "font-family": "Arial, Helvetica, sans-serif",
    })
    ET.SubElement(text_group, f"{{{SVG_NS}}}text", {
        "x": str(bar_x + bar_width + 8),
        "y": str(bar_y + 4),
        "dominant-baseline": "hanging",
    }).text = f"{atlas_heatmap.style.vmax:.4g}"
    ET.SubElement(text_group, f"{{{SVG_NS}}}text", {
        "x": str(bar_x + bar_width + 8),
        "y": str(bar_y + bar_height * 0.5),
        "dominant-baseline": "middle",
    }).text = f"{(atlas_heatmap.style.vmin + atlas_heatmap.style.vmax) / 2:.4g}"
    ET.SubElement(text_group, f"{{{SVG_NS}}}text", {
        "x": str(bar_x + bar_width + 8),
        "y": str(bar_y + bar_height - 2),
        "dominant-baseline": "auto",
    }).text = f"{atlas_heatmap.style.vmin:.4g}"
    ET.SubElement(text_group, f"{{{SVG_NS}}}text", {
        "x": str(bar_x + bar_width // 2),
        "y": str(max(bar_y - 6, 10)),
        "text-anchor": "middle",
        "font-size": "9",
    }).text = atlas_heatmap.style.metric_name


    tree = ET.ElementTree(root)
    tree.write(output_path, encoding="utf-8", xml_declaration=True)
    return output_path


def render_atlas_slice(
    atlas_slice: AtlasSlice,
    output_path: str | Path,
    *,
    dpi: int = 300,
    line_width: float = 0.3,
    brain_outline_width: float = 0.0,
    contour_smoothing: float = 1.8,
    show_regions: bool = True,
) -> Path:
    """Render a white-background, black-outline atlas slice."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    image = np.asarray(atlas_slice.image)
    if image.ndim != 2:
        raise ValueError(f"AtlasSlice.image must be 2D, got shape: {image.shape}")

    height, width = image.shape

    # --- Native SVG rendering branch ---
    if output_path.suffix.lower() == ".svg":
        brain_mask = image > 0
        brain_d_paths: list[str] = []
        region_d_paths: list[str] = []

        if np.any(brain_mask):
            brain_lines = _mask_contour_lines(brain_mask, smoothing=float(contour_smoothing))
            brain_d_paths = _contours_to_svg_d(brain_lines)

        if show_regions:
            region_lines = _label_contour_lines(image, smoothing=float(contour_smoothing))
            region_d_paths = _contours_to_svg_d(region_lines)
        else:
            region_d_paths = []

        return _render_svg(
            output_path,
            width=width,
            height=height,
            brain_d_paths=brain_d_paths,
            region_d_paths=region_d_paths,
            line_width=float(line_width),
            brain_outline_width=float(brain_outline_width),
        )

    # --- Matplotlib rendering branch (PNG, PDF, etc.) ---
    aspect = width / max(height, 1)
    long_side = 6.0
    if aspect >= 1:
        figsize = (long_side, max(long_side / aspect, 1.0))
    else:
        figsize = (max(long_side * aspect, 1.0), long_side)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")
    ax.imshow(np.ones((height, width), dtype=np.uint8), cmap="gray", vmin=0, vmax=1, interpolation="nearest")

    brain_mask = image > 0
    if show_regions:
        region_lines = _label_contour_lines(image, smoothing=float(contour_smoothing))
        _add_lines(ax, region_lines, linewidth=float(line_width))

    if np.any(brain_mask):
        brain_lines = _mask_contour_lines(brain_mask, smoothing=float(contour_smoothing))
        _add_lines(ax, brain_lines, linewidth=float(brain_outline_width))

    ax.set_axis_off()
    ax.set_xlim(-0.5, width - 0.5)
    ax.set_ylim(height - 0.5, -0.5)
    fig.savefig(output_path, dpi=dpi, facecolor="black", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    return output_path


def parse_bregma_index(value: str) -> tuple[int, int, int]:
    parts = [part.strip() for part in str(value).split(",") if part.strip()]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("bregma index must be in dv,ap,ml format, e.g. 18,216,228")
    try:
        return tuple(int(part) for part in parts)  # type: ignore[return-value]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("bregma index values must be integers") from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Render a white-background Allen atlas label slice with black outlines. "
            "Atlas TIFF axes are interpreted as (DV, AP, ML)."
        )
    )
    parser.add_argument("--label", default=str(DEFAULT_ATLAS_LABEL), help="3D atlas label TIFF path")
    parser.add_argument("--plane", required=True, choices=sorted(PLANE_TO_FIXED_AXIS), help="Slice plane")
    parser.add_argument(
        "--coord-system",
        required=True,
        choices=["bregma-mm", "ccf-um", "index"],
        help="Coordinate system for --coord along the plane's fixed axis",
    )
    parser.add_argument("--coord", required=True, type=float, help="Coordinate value for the selected coordinate system")
    parser.add_argument("--output", required=True, help="Output image path (.png, .svg, .pdf, etc.)")
    parser.add_argument("--atlas-resolution-um", type=float, default=25.0, help="Atlas voxel size in microns")
    parser.add_argument(
        "--bregma-index",
        type=parse_bregma_index,
        default=DEFAULT_BREGMA_INDEX,
        help="Approximate bregma index as dv,ap,ml; default: 18,216,228",
    )
    parser.add_argument("--dpi", type=int, default=300, help="Output DPI for raster formats")
    parser.add_argument("--line-width", type=float, default=0.12, help="Region boundary line width in points")
    parser.add_argument("--brain-outline-width", type=float, default=0.0, help="Outer brain boundary line width in points (0=off)")
    parser.add_argument("--contour-smoothing", type=float, default=1.8, help="Gaussian smoothing sigma for contour coordinates")
    parser.add_argument("--hide-regions", action="store_true", help="Draw only the outer brain outline")
    parser.add_argument("--input-excel", default=None, help="Excel workbook with Level_* sheets for region metrics")
    parser.add_argument(
        "--region-cfg",
        default=str(Path(__file__).resolve().parents[1] / "registration" / "Region_Csv_Rev1_updated.CSV"),
        help="Path to region CSV used to map Excel region names to atlas ids",
    )
    parser.add_argument("--metric", default="Voxel Density", help="Metric column to render when --input-excel is set")
    parser.add_argument("--vmin", type=float, default=0.0, help="Lower bound for heatmap color scaling")
    parser.add_argument("--vmax", type=float, default=None, help="Upper bound for heatmap color scaling")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        spec = AtlasSliceSpec(
            plane=args.plane,
            coordinate_system=args.coord_system,
            coordinate=args.coord,
            atlas_resolution_um=args.atlas_resolution_um,
            bregma_index=args.bregma_index,
        )
        atlas_slice = extract_atlas_slice(args.label, spec)
        if args.input_excel:
            atlas_heatmap = build_atlas_slice_heatmap(
                atlas_slice,
                input_excel=args.input_excel,
                cfg_path=args.region_cfg,
                metric=args.metric,
                vmin=args.vmin,
                vmax=args.vmax,
            )
            output_path = render_atlas_slice_heatmap(
                atlas_heatmap,
                args.output,
                line_width=args.line_width,
                brain_outline_width=args.brain_outline_width,
            )
        else:
            atlas_heatmap = None
            output_path = render_atlas_slice(
                atlas_slice,
                args.output,
                dpi=args.dpi,
                line_width=args.line_width,
                brain_outline_width=args.brain_outline_width,
                contour_smoothing=args.contour_smoothing,
                show_regions=not args.hide_regions,
            )
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    payload = {
        "output_path": str(output_path),
        "plane": atlas_slice.plane,
        "index": atlas_slice.index,
        "coordinate_label": atlas_slice.coordinate_label,
        "extent_mm": atlas_slice.extent_mm,
        "x_axis": atlas_slice.x_axis,
        "y_axis": atlas_slice.y_axis,
        "spec": asdict(spec),
    }
    if atlas_heatmap is not None:
        payload["metric"] = atlas_heatmap.style.metric_name
        payload["vmin"] = atlas_heatmap.style.vmin
        payload["vmax"] = atlas_heatmap.style.vmax
        payload["colored_region_count"] = len(atlas_heatmap.region_values)
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
