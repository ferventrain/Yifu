"""Spatial pattern utilities for the cFos visual report (P1)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import tifffile
from scipy.ndimage import binary_erosion, gaussian_filter
from skimage.measure import marching_cubes

from pipeline_modules.visualization.atlas_slice import (
    AXIS_NAMES,
    DEFAULT_ATLAS_LABEL,
    PLANE_TO_FIXED_AXIS,
    bregma_mm_for_plane_index,
    index_to_bregma_mm,
)

AxisName = Literal["DV", "AP", "ML"]
SourceKind = Literal["points_csv", "spotiflow_points_csv", "atlas_volume_tiff", "none"]

AXIS_INDEX = {"DV": 0, "AP": 1, "ML": 2}
AXIS_TO_PLANE = {"AP": "coronal", "ML": "sagittal", "DV": "horizontal"}
RESOLUTION_UM = 25.0
DEFAULT_ATLAS_RESOLUTION_UM_DV_AP_ML = (RESOLUTION_UM, RESOLUTION_UM, RESOLUTION_UM)
ALLEN_ROOT_REGION_ID = 997


def _spatial_metadata(
    atlas_shape: tuple[int, int, int],
    resolution_um: tuple[float, float, float] = DEFAULT_ATLAS_RESOLUTION_UM_DV_AP_ML,
) -> dict[str, Any]:
    return {
        "atlas_shape_dv_ap_ml": list(atlas_shape),
        "atlas_resolution_um_dv_ap_ml": [float(value) for value in resolution_um],
        "coordinate_space": "atlas_array_index",
        "axis_order": ["DV", "AP", "ML"],
    }


def get_atlas_shape(atlas_label: str | Path = DEFAULT_ATLAS_LABEL) -> tuple[int, int, int]:
    atlas_label = Path(atlas_label)
    if not atlas_label.exists():
        return (456, 528, 320)
    with tifffile.TiffFile(atlas_label) as handle:
        shape = tuple(int(value) for value in handle.series[0].shape)
    if len(shape) != 3:
        raise ValueError(f"Expected 3D atlas label, got shape {shape}")
    return shape  # (DV, AP, ML)


def resolve_spatial_source(sample_assets: dict[str, Any]) -> tuple[SourceKind, Path | None]:
    for key in ("points_csv", "spotiflow_points_csv", "atlas_volume_tiff"):
        path_text = sample_assets.get(key)
        if path_text and Path(path_text).exists():
            return key, Path(path_text)  # type: ignore[return-value]
    return "none", None


def _normalize_points_frame(frame: pd.DataFrame, source_kind: SourceKind) -> pd.DataFrame:
    working = frame.copy()

    if {"grid_x", "grid_y", "grid_z"}.issubset(working.columns):
        working["index_ap"] = pd.to_numeric(working["grid_x"], errors="coerce")
        working["index_dv"] = pd.to_numeric(working["grid_y"], errors="coerce")
        working["index_ml"] = pd.to_numeric(working["grid_z"], errors="coerce")
    elif source_kind == "spotiflow_points_csv" and {"z", "y", "x"}.issubset(working.columns):
        # Spotiflow exports sample-space z,y,x; treat as atlas indices when already warped.
        working["index_dv"] = pd.to_numeric(working["z"], errors="coerce")
        working["index_ap"] = pd.to_numeric(working["y"], errors="coerce")
        working["index_ml"] = pd.to_numeric(working["x"], errors="coerce")
    elif {"x", "y", "z"}.issubset(working.columns):
        # warp_mask_zarr_to_atlas_points CSV: x=AP, y=DV, z=ML in microns.
        working["index_ap"] = pd.to_numeric(working["x"], errors="coerce") / RESOLUTION_UM
        working["index_dv"] = pd.to_numeric(working["y"], errors="coerce") / RESOLUTION_UM
        working["index_ml"] = pd.to_numeric(working["z"], errors="coerce") / RESOLUTION_UM
    else:
        raise ValueError(
            "Points table must contain grid_x/grid_y/grid_z or x/y/z columns. "
            f"Found: {', '.join(map(str, working.columns))}"
        )

    for column in ("index_ap", "index_dv", "index_ml"):
        working[column] = working[column].astype(float)
    working = working.dropna(subset=["index_ap", "index_dv", "index_ml"])
    return working


def load_points_frame(path: str | Path, source_kind: SourceKind) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Spatial source not found: {path}")
    frame = pd.read_csv(path)
    if frame.empty:
        return frame
    return _normalize_points_frame(frame, source_kind)


def _atlas_volume_to_points(
    atlas_volume_zyx: np.ndarray,
    *,
    atlas_resolution_xyz: tuple[float, float, float] = (RESOLUTION_UM, RESOLUTION_UM, RESOLUTION_UM),
    max_points: int,
) -> pd.DataFrame:
    z_idx, y_idx, x_idx = np.nonzero(np.asarray(atlas_volume_zyx))
    ap_idx = y_idx
    dv_idx = z_idx
    ml_idx = x_idx
    table = pd.DataFrame(
        {
            "x": (ap_idx.astype(np.float64) + 0.5) * atlas_resolution_xyz[0],
            "y": (dv_idx.astype(np.float64) + 0.5) * atlas_resolution_xyz[1],
            "z": (ml_idx.astype(np.float64) + 0.5) * atlas_resolution_xyz[2],
            "grid_x": ap_idx.astype(np.int64),
            "grid_y": dv_idx.astype(np.int64),
            "grid_z": ml_idx.astype(np.int64),
        }
    )
    if max_points and max_points > 0 and len(table) > max_points:
        table = table.sample(n=int(max_points), random_state=0).sort_values(["grid_z", "grid_y", "grid_x"])
    return table.reset_index(drop=True)


def points_frame_from_atlas_volume(path: str | Path, *, max_points: int = 150_000) -> pd.DataFrame:
    volume = tifffile.imread(str(path))
    table = _atlas_volume_to_points(np.asarray(volume), max_points=max_points)
    return _normalize_points_frame(table, "atlas_volume_tiff")


def load_spatial_points(
    sample_assets: dict[str, Any],
    *,
    max_points: int = 80_000,
) -> tuple[pd.DataFrame, SourceKind, Path | None]:
    source_kind, source_path = resolve_spatial_source(sample_assets)
    if source_path is None:
        return pd.DataFrame(), "none", None

    if source_kind == "atlas_volume_tiff":
        frame = points_frame_from_atlas_volume(source_path, max_points=max_points)
    else:
        frame = load_points_frame(source_path, source_kind)
        if max_points and len(frame) > max_points:
            frame = frame.sample(n=int(max_points), random_state=0).reset_index(drop=True)
    return frame, source_kind, source_path


def _histogram_from_indices(values: np.ndarray, *, bins: int, axis_size: int) -> dict[str, Any]:
    clean = values[np.isfinite(values)]
    if clean.size == 0:
        return {
            "counts": [],
            "bin_edges": [],
            "bin_centers_index": [],
            "total": 0,
        }
    clipped = np.clip(clean, 0, max(axis_size - 1, 0))
    counts, bin_edges = np.histogram(clipped, bins=int(bins), range=(0, axis_size))
    centers = ((bin_edges[:-1] + bin_edges[1:]) / 2.0).astype(float)
    return {
        "counts": [int(value) for value in counts.tolist()],
        "bin_edges": [float(value) for value in bin_edges.tolist()],
        "bin_centers_index": [float(value) for value in centers.tolist()],
        "total": int(clean.size),
    }


def _histogram_from_volume(volume: np.ndarray, axis: AxisName, *, bins: int) -> dict[str, Any]:
    axis_index = AXIS_INDEX[axis]
    axis_size = int(volume.shape[axis_index])
    if axis_size <= 0:
        return {"counts": [], "bin_edges": [], "bin_centers_index": [], "total": 0}

    counts_per_index = []
    for index in range(axis_size):
        section = np.take(volume, index, axis=axis_index)
        if np.issubdtype(section.dtype, np.floating):
            counts_per_index.append(float(section.sum()))
        else:
            counts_per_index.append(float(np.count_nonzero(section)))

    per_index = np.asarray(counts_per_index, dtype=float)
    edges = np.linspace(0, axis_size, int(bins) + 1)
    bin_counts = np.zeros(int(bins), dtype=float)
    for index, value in enumerate(per_index):
        if value <= 0:
            continue
        bin_idx = min(int(index / max(axis_size, 1) * bins), bins - 1)
        bin_counts[bin_idx] += value

    centers = ((edges[:-1] + edges[1:]) / 2.0).astype(float)
    return {
        "counts": [float(value) for value in bin_counts.tolist()],
        "bin_edges": [float(value) for value in edges.tolist()],
        "bin_centers_index": [float(value) for value in centers.tolist()],
        "total": float(per_index.sum()),
    }


def _attach_bregma_bin_centers(
    histogram: dict[str, Any],
    *,
    axis: AxisName,
    bregma_index: tuple[int, int, int],
    resolution_um: float,
) -> None:
    centers = histogram.get("bin_centers_index") or []
    if not centers:
        return
    axis_idx = AXIS_INDEX[axis]
    histogram["bin_centers_bregma_mm"] = [
        round(index_to_bregma_mm(axis_idx, value, bregma_index=bregma_index, resolution_um=resolution_um), 3)
        for value in centers
    ]


def build_axis_histogram(
    *,
    axis: AxisName,
    points_frame: pd.DataFrame | None,
    atlas_shape: tuple[int, int, int],
    volume: np.ndarray | None = None,
    bins: int = 32,
    bregma_index: tuple[int, int, int] = (18, 216, 228),
    resolution_um: float = RESOLUTION_UM,
) -> dict[str, Any]:
    axis_size = int(atlas_shape[AXIS_INDEX[axis]])
    column_map = {"AP": "index_ap", "DV": "index_dv", "ML": "index_ml"}
    if points_frame is not None and not points_frame.empty:
        histogram = _histogram_from_indices(
            points_frame[column_map[axis]].to_numpy(dtype=float),
            bins=bins,
            axis_size=axis_size,
        )
        source = "points"
    elif volume is not None:
        histogram = _histogram_from_volume(volume, axis, bins=bins)
        source = "volume"
    else:
        histogram = {"counts": [], "bin_edges": [], "bin_centers_index": [], "total": 0}
        source = "none"

    payload = {
        "axis": axis,
        "plane": AXIS_TO_PLANE[axis],
        "fixed_axis_index": AXIS_INDEX[axis],
        "axis_size": axis_size,
        "source": source,
        "measure": "cfos_count",
        "value_unit": "count",
        **histogram,
    }
    _attach_bregma_bin_centers(
        payload,
        axis=axis,
        bregma_index=bregma_index,
        resolution_um=resolution_um,
    )
    return payload


def build_spatial_payload(
    sample_assets: dict[str, Any],
    *,
    atlas_label: str | Path = DEFAULT_ATLAS_LABEL,
    bins: int = 32,
    max_points: int = 80_000,
    bregma_index: tuple[int, int, int] = (18, 216, 228),
    resolution_um: float = RESOLUTION_UM,
) -> dict[str, Any]:
    atlas_shape = get_atlas_shape(atlas_label)
    points_frame, source_kind, source_path = load_spatial_points(sample_assets, max_points=max_points)

    volume = None
    histogram_source_kind = source_kind
    if sample_assets.get("atlas_volume_tiff") and Path(sample_assets["atlas_volume_tiff"]).exists():
        volume = np.asarray(tifffile.imread(str(sample_assets["atlas_volume_tiff"])))
        histogram_source_kind = "atlas_volume_tiff"

    histogram_points = None if volume is not None else (points_frame if not points_frame.empty else None)

    axes = {
        axis: build_axis_histogram(
            axis=axis,
            points_frame=histogram_points,
            atlas_shape=atlas_shape,
            volume=volume,
            bins=bins,
            bregma_index=bregma_index,
            resolution_um=resolution_um,
        )
        for axis in AXIS_NAMES
    }

    return {
        "available": source_kind != "none",
        "source_kind": source_kind,
        "histogram_source_kind": histogram_source_kind,
        "source_path": str(source_path) if source_path else None,
        "atlas_shape_dv_ap_ml": list(atlas_shape),
        "axis_names": list(AXIS_NAMES),
        "plane_by_axis": dict(AXIS_TO_PLANE),
        "bins": int(bins),
        "point_count": int(len(points_frame)),
        "measure": "cfos_count",
        "value_unit": "count",
        "axes": axes,
    }


@lru_cache(maxsize=4)
def _load_atlas_centroid_stats(atlas_label: str) -> dict[str, Any] | None:
    """One-pass voxel sums per atlas region id; cached per atlas label path."""
    atlas_path = Path(atlas_label)
    if not atlas_path.exists():
        return None

    labels = np.asarray(tifffile.memmap(str(atlas_path)))
    if labels.size == 0:
        return None

    flat = np.rint(labels).astype(np.int64, copy=False).ravel()
    unique_ids, inverse = np.unique(flat, return_inverse=True)
    dv_grid, ap_grid, ml_grid = np.indices(labels.shape)
    dv_flat = dv_grid.ravel().astype(np.float64, copy=False)
    ap_flat = ap_grid.ravel().astype(np.float64, copy=False)
    ml_flat = ml_grid.ravel().astype(np.float64, copy=False)
    bucket_count = int(unique_ids.size)
    counts = np.bincount(inverse, minlength=bucket_count).astype(np.int64)
    sum_dv = np.bincount(inverse, weights=dv_flat, minlength=bucket_count)
    sum_ap = np.bincount(inverse, weights=ap_flat, minlength=bucket_count)
    sum_ml = np.bincount(inverse, weights=ml_flat, minlength=bucket_count)

    by_id: dict[int, dict[str, float | int]] = {}
    for index, region_id in enumerate(unique_ids):
        region_id = int(region_id)
        count = int(counts[index])
        if region_id == 0 or count <= 0:
            continue
        by_id[region_id] = {
            "count": count,
            "sum_dv": float(sum_dv[index]),
            "sum_ap": float(sum_ap[index]),
            "sum_ml": float(sum_ml[index]),
        }

    return {
        "shape": tuple(int(value) for value in labels.shape),
        "by_id": by_id,
    }


def _centroid_from_region_ids(
    region_ids: frozenset[int],
    stats: dict[str, Any],
) -> dict[str, float] | None:
    by_id = stats["by_id"]
    total_count = 0
    total_dv = 0.0
    total_ap = 0.0
    total_ml = 0.0
    for region_id in region_ids:
        row = by_id.get(int(region_id))
        if not row:
            continue
        count = int(row["count"])
        if count <= 0:
            continue
        total_count += count
        total_dv += float(row["sum_dv"])
        total_ap += float(row["sum_ap"])
        total_ml += float(row["sum_ml"])
    if total_count <= 0:
        return None
    return {
        "index_dv": total_dv / total_count,
        "index_ap": total_ap / total_count,
        "index_ml": total_ml / total_count,
    }


def build_atlas_region_centroids_payload(
    atlas_label: str | Path = DEFAULT_ATLAS_LABEL,
) -> dict[str, Any]:
    """Compact per-region voxel sums for client-side slice focus."""
    stats = _load_atlas_centroid_stats(str(Path(atlas_label).resolve()))
    if stats is None:
        return {"available": False}

    regions: dict[str, dict[str, float | int]] = {
        str(region_id): dict(row) for region_id, row in stats["by_id"].items()
    }

    return {
        "available": True,
        "atlas_shape_dv_ap_ml": list(stats["shape"]),
        "regions": regions,
    }


def compute_region_centroid_indices(
    region_ids: frozenset[int],
    atlas_label: str | Path = DEFAULT_ATLAS_LABEL,
) -> dict[str, Any]:
    atlas_label = Path(atlas_label)
    atlas_shape = get_atlas_shape(atlas_label)
    if not atlas_label.exists() or not region_ids:
        return {
            "available": False,
            "atlas_shape_dv_ap_ml": list(atlas_shape),
            "region_ids": sorted(region_ids),
        }

    stats = _load_atlas_centroid_stats(str(atlas_label.resolve()))
    if stats is None:
        return {
            "available": False,
            "atlas_shape_dv_ap_ml": list(atlas_shape),
            "region_ids": sorted(region_ids),
        }

    centroid = _centroid_from_region_ids(region_ids, stats)
    if centroid is None:
        return {
            "available": False,
            "atlas_shape_dv_ap_ml": list(atlas_shape),
            "region_ids": sorted(region_ids),
        }

    return {
        "available": True,
        "atlas_shape_dv_ap_ml": list(atlas_shape),
        "region_ids": sorted(region_ids),
        "centroid_index_dv_ap_ml": [
            round(centroid["index_dv"], 2),
            round(centroid["index_ap"], 2),
            round(centroid["index_ml"], 2),
        ],
        **centroid,
    }


def build_region_slice_focus_payload(
    region_ids: frozenset[int],
    *,
    atlas_label: str | Path = DEFAULT_ATLAS_LABEL,
    bregma_index: tuple[int, int, int] = (18, 216, 228),
    resolution_um: float = RESOLUTION_UM,
    plane: str = "coronal",
) -> dict[str, Any]:
    centroid_payload = compute_region_centroid_indices(region_ids, atlas_label=atlas_label)
    if not centroid_payload.get("available"):
        return {
            **centroid_payload,
            "plane": plane,
        }

    ap_index = int(round(float(centroid_payload["index_ap"])))
    dv_index = int(round(float(centroid_payload["index_dv"])))
    ml_index = int(round(float(centroid_payload["index_ml"])))
    atlas_shape = tuple(int(value) for value in centroid_payload["atlas_shape_dv_ap_ml"])
    ap_index = int(min(max(ap_index, 0), atlas_shape[1] - 1))
    dv_index = int(min(max(dv_index, 0), atlas_shape[0] - 1))
    ml_index = int(min(max(ml_index, 0), atlas_shape[2] - 1))

    plane_index = {
        "coronal": ap_index,
        "horizontal": dv_index,
        "sagittal": ml_index,
    }.get(plane, ap_index)

    return {
        **centroid_payload,
        "plane": plane,
        "coordinate_system": "index",
        "coordinate": plane_index,
        "recommended_plane": "coronal",
        "recommended_index_ap": ap_index,
        "recommended_index_dv": dv_index,
        "recommended_index_ml": ml_index,
        "bregma_mm_ap": round(
            index_to_bregma_mm(1, ap_index, bregma_index=bregma_index, resolution_um=resolution_um),
            3,
        ),
        "bregma_mm_dv": round(
            index_to_bregma_mm(0, dv_index, bregma_index=bregma_index, resolution_um=resolution_um),
            3,
        ),
        "bregma_mm_ml": round(
            index_to_bregma_mm(2, ml_index, bregma_index=bregma_index, resolution_um=resolution_um),
            3,
        ),
        "bregma_mm": round(
            bregma_mm_for_plane_index(plane, plane_index, bregma_index=bregma_index, resolution_um=resolution_um),
            3,
        ),
    }


def _clip_indices(dv: float, ap: float, ml: float, shape: tuple[int, int, int]) -> tuple[int, int, int]:
    return (
        int(min(max(round(dv), 0), shape[0] - 1)),
        int(min(max(round(ap), 0), shape[1] - 1)),
        int(min(max(round(ml), 0), shape[2] - 1)),
    )


def _lookup_region_id(labels: np.ndarray, dv: float, ap: float, ml: float) -> int:
    dv_i, ap_i, ml_i = _clip_indices(dv, ap, ml, labels.shape)
    return int(labels[dv_i, ap_i, ml_i])


@lru_cache(maxsize=32)
def _cached_region_surface_mesh(
    atlas_label: str,
    region_ids_key: tuple[int, ...],
    stride: int,
    smooth_sigma: float,
) -> tuple[tuple[tuple[float, float, float], ...], tuple[tuple[int, int, int], ...]]:
    labels = tifffile.memmap(atlas_label)
    mask = np.isin(np.asarray(labels[::stride, ::stride, ::stride]), list(region_ids_key))
    if not np.any(mask):
        return ((), ())

    field = mask.astype(np.float32)
    if smooth_sigma > 0:
        field = gaussian_filter(field, sigma=float(smooth_sigma))

    try:
        verts, faces, _, _ = marching_cubes(field, level=0.5)
    except (ValueError, RuntimeError):
        return ((), ())

    if verts.size == 0 or faces.size == 0:
        return ((), ())

    dv = verts[:, 0] * float(stride)
    ap = verts[:, 1] * float(stride)
    ml = verts[:, 2] * float(stride)
    vertices = tuple((float(dv[i]), float(ap[i]), float(ml[i])) for i in range(len(dv)))
    triangles = tuple((int(a), int(b), int(c)) for a, b, c in faces)
    return vertices, triangles


@lru_cache(maxsize=8)
def _cached_brain_outline_surface_mesh(
    atlas_label: str,
    stride: int,
    smooth_sigma: float,
) -> tuple[tuple[tuple[float, float, float], ...], tuple[tuple[int, int, int], ...]]:
    labels = tifffile.memmap(atlas_label)
    mask = np.asarray(labels[::stride, ::stride, ::stride]) > 0
    if not np.any(mask):
        return ((), ())

    field = mask.astype(np.float32)
    if smooth_sigma > 0:
        field = gaussian_filter(field, sigma=float(smooth_sigma))

    try:
        verts, faces, _, _ = marching_cubes(field, level=0.5)
    except (ValueError, RuntimeError):
        return ((), ())

    if verts.size == 0 or faces.size == 0:
        return ((), ())

    dv = verts[:, 0] * float(stride)
    ap = verts[:, 1] * float(stride)
    ml = verts[:, 2] * float(stride)
    vertices = tuple((float(dv[i]), float(ap[i]), float(ml[i])) for i in range(len(dv)))
    triangles = tuple((int(a), int(b), int(c)) for a, b, c in faces)
    return vertices, triangles


def _surface_payload_from_mesh(
    *,
    atlas_shape: tuple[int, int, int],
    vertices: tuple[tuple[float, float, float], ...],
    faces: tuple[tuple[int, int, int], ...],
    region_ids: list[int] | None = None,
    kind: str = "region",
) -> dict[str, Any]:
    payload = {
        "available": bool(vertices),
        "kind": kind,
        "vertex_count": len(vertices),
        "face_count": len(faces),
        "vertices": [{"dv": dv, "ap": ap, "ml": ml} for dv, ap, ml in vertices],
        "faces": [list(triangle) for triangle in faces],
        **_spatial_metadata(atlas_shape),
    }
    if region_ids is not None:
        payload["region_ids"] = region_ids
    return payload


def build_region_surface_payload(
    *,
    atlas_label: str | Path = DEFAULT_ATLAS_LABEL,
    region_ids: frozenset[int],
    stride: int = 2,
    smooth_sigma: float = 1.2,
) -> dict[str, Any]:
    atlas_label = Path(atlas_label)
    atlas_shape = get_atlas_shape(atlas_label)
    if not atlas_label.exists() or not region_ids:
        return {
            "available": False,
            "kind": "region",
            "vertex_count": 0,
            "face_count": 0,
            "region_ids": sorted(region_ids),
            "vertices": [],
            "faces": [],
            **_spatial_metadata(atlas_shape),
        }

    ids_key = tuple(sorted(int(value) for value in region_ids))
    vertices, faces = _cached_region_surface_mesh(
        str(atlas_label),
        ids_key,
        int(stride),
        float(smooth_sigma),
    )
    return _surface_payload_from_mesh(
        atlas_shape=atlas_shape,
        vertices=vertices,
        faces=faces,
        region_ids=list(ids_key),
        kind="region",
    )


def build_brain_outline_surface_payload(
    *,
    atlas_label: str | Path = DEFAULT_ATLAS_LABEL,
    stride: int = 2,
    smooth_sigma: float = 1.4,
) -> dict[str, Any]:
    atlas_label = Path(atlas_label)
    atlas_shape = get_atlas_shape(atlas_label)
    if not atlas_label.exists():
        return {
            "available": False,
            "kind": "brain_outline",
            "vertex_count": 0,
            "face_count": 0,
            "vertices": [],
            "faces": [],
            **_spatial_metadata(atlas_shape),
        }

    vertices, faces = _cached_brain_outline_surface_mesh(
        str(atlas_label),
        int(stride),
        float(smooth_sigma),
    )
    return _surface_payload_from_mesh(
        atlas_shape=atlas_shape,
        vertices=vertices,
        faces=faces,
        kind="brain_outline",
    )


def build_points_viewer_payload(
    sample_assets: dict[str, Any],
    *,
    max_points: int = 50_000,
    atlas_label: str | Path = DEFAULT_ATLAS_LABEL,
    in_brain_only: bool = True,
) -> dict[str, Any]:
    atlas_shape = get_atlas_shape(atlas_label)
    points_frame, source_kind, source_path = load_spatial_points(sample_assets, max_points=max_points)
    if points_frame.empty:
        return {
            "available": False,
            "source_kind": source_kind,
            "source_path": str(source_path) if source_path else None,
            "point_count": 0,
            "display_count": 0,
            "points": [],
            **_spatial_metadata(atlas_shape),
        }

    labels = tifffile.memmap(str(atlas_label)) if Path(atlas_label).exists() else None
    points = []
    for row in points_frame.itertuples(index=False):
        region_id = (
            _lookup_region_id(labels, float(row.index_dv), float(row.index_ap), float(row.index_ml))
            if labels is not None
            else 0
        )
        if in_brain_only and int(region_id) <= 0:
            continue
        points.append(
            {
                "ap": float(row.index_ap),
                "dv": float(row.index_dv),
                "ml": float(row.index_ml),
                "region_id": region_id,
            }
        )

    return {
        "available": True,
        "source_kind": source_kind,
        "source_path": str(source_path) if source_path else None,
        "point_count": int(len(points_frame)),
        "display_count": len(points),
        "points": points,
        **_spatial_metadata(atlas_shape),
    }


def slice_linkage_for_bin(axis: AxisName, bin_center_index: float) -> dict[str, Any]:
    plane = AXIS_TO_PLANE[axis]
    return {
        "axis": axis,
        "plane": plane,
        "coordinate_system": "index",
        "coordinate": float(bin_center_index),
        "fixed_axis_index": PLANE_TO_FIXED_AXIS[plane],
    }


def build_region_pick_payload(
    *,
    atlas_label: str | Path,
    plane: str,
    coordinate: float,
    coordinate_system: str = "index",
    pixel_x: float,
    pixel_y: float,
    image_width: float,
    image_height: float,
) -> dict[str, Any]:
    from pipeline_modules.visualization.atlas_slice import AtlasSliceSpec, extract_atlas_slice

    if image_width <= 0 or image_height <= 0:
        raise ValueError("image_width and image_height must be positive.")

    spec = AtlasSliceSpec(
        plane=plane,
        coordinate_system=coordinate_system,
        coordinate=float(coordinate),
    )
    atlas_slice = extract_atlas_slice(atlas_label, spec)
    slice_h, slice_w = atlas_slice.image.shape
    col = int(round((float(pixel_x) / float(image_width)) * max(slice_w - 1, 0)))
    row = int(round((float(pixel_y) / float(image_height)) * max(slice_h - 1, 0)))
    col = int(np.clip(col, 0, slice_w - 1))
    row = int(np.clip(row, 0, slice_h - 1))
    region_id = int(atlas_slice.image[row, col])
    return {
        "available": region_id > 0,
        "region_id": region_id,
        "plane": plane,
        "coordinate": float(coordinate),
        "pixel_x": col,
        "pixel_y": row,
        "slice_shape_yx": [int(slice_h), int(slice_w)],
    }
