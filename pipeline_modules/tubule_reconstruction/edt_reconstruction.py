"""Chunked vessel centerlines via Euclidean distance transform + 3D thinning.

Designed for large LSFM binary vessel masks in Zarr: each chunk is skeletonized
independently, then adjacent-face endpoints are stitched so vessels that cross
block boundaries stay connected.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.spatial import cKDTree

try:
    from skimage.morphology import skeletonize as _skeletonize
except ImportError:  # older skimage
    from skimage.morphology import skeletonize_3d as _skeletonize

try:
    from pipeline_modules.tubule_reconstruction.kimimaro_reconstruction import (
        chunk_index_to_slices,
        expand_slices,
        open_zarr_dataset,
        parse_resolution_xyz,
        parse_triplet_int,
        resolution_xyz_to_zyx,
    )
except ImportError:
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.append(str(project_root))
    from pipeline_modules.tubule_reconstruction.kimimaro_reconstruction import (
        chunk_index_to_slices,
        expand_slices,
        open_zarr_dataset,
        parse_resolution_xyz,
        parse_triplet_int,
        resolution_xyz_to_zyx,
    )

logger = logging.getLogger(__name__)

# 26-neighborhood, one direction only so each voxel pair is emitted once.
_FORWARD_OFFSETS = tuple(
    (dz, dy, dx)
    for dz in range(-1, 2)
    for dy in range(-1, 2)
    for dx in range(-1, 2)
    if (dz, dy, dx) > (0, 0, 0)
)


def parse_chunk_index(text: str) -> tuple[int, int, int]:
    parts = [part.strip() for part in str(text).replace(",", ".").split(".") if part.strip()]
    if len(parts) != 3:
        raise ValueError(f"chunk_index must be z.y.x, got: {text}")
    return tuple(int(part) for part in parts)


def _remove_small_components(binary: np.ndarray, dust_threshold: int) -> np.ndarray:
    if dust_threshold <= 1:
        return binary
    labeled, n_features = ndimage.label(binary.astype(np.uint8, copy=False))
    if n_features == 0:
        return binary
    counts = np.bincount(labeled.ravel())
    keep = np.zeros(counts.shape[0], dtype=bool)
    keep[1:] = counts[1:] >= int(dust_threshold)
    return keep[labeled]


def skeletonize_binary_edt(
    binary_mask: np.ndarray,
    *,
    resolution_xyz=(1.0, 1.0, 1.0),
    dust_threshold: int = 100,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a 1-voxel-wide skeleton and the anisotropic DBF (microns)."""
    if binary_mask.ndim != 3:
        raise ValueError(f"Expected a 3D mask, got shape={binary_mask.shape}")
    binary = np.asarray(binary_mask, dtype=bool)
    if dust_threshold:
        binary = _remove_small_components(binary, dust_threshold)
    if not np.any(binary):
        return np.zeros(binary.shape, dtype=bool), np.zeros(binary.shape, dtype=np.float32)

    sampling = resolution_xyz_to_zyx(resolution_xyz)
    dbf = ndimage.distance_transform_edt(binary, sampling=sampling).astype(np.float32, copy=False)
    skeleton = np.asarray(_skeletonize(binary), dtype=bool)
    skeleton &= binary
    return skeleton, dbf


def skeleton_to_graph(skeleton: np.ndarray):
    """26-connected graph from a binary skeleton. Vertices are local z,y,x."""
    coords = np.argwhere(skeleton)
    if len(coords) == 0:
        return (
            np.empty((0, 3), dtype=np.int32),
            np.empty((0, 2), dtype=np.int32),
            np.empty((0,), dtype=np.int32),
        )

    index_of = {tuple(int(v) for v in row): i for i, row in enumerate(coords)}
    edges: list[tuple[int, int]] = []
    shape = skeleton.shape
    for i, (z, y, x) in enumerate(coords):
        z_i, y_i, x_i = int(z), int(y), int(x)
        for dz, dy, dx in _FORWARD_OFFSETS:
            zz, yy, xx = z_i + dz, y_i + dy, x_i + dx
            if not (0 <= zz < shape[0] and 0 <= yy < shape[1] and 0 <= xx < shape[2]):
                continue
            j = index_of.get((zz, yy, xx))
            if j is not None:
                edges.append((i, j))

    edge_arr = np.asarray(edges, dtype=np.int32) if edges else np.empty((0, 2), dtype=np.int32)
    degrees = np.zeros(len(coords), dtype=np.int32)
    if len(edge_arr):
        np.add.at(degrees, edge_arr[:, 0], 1)
        np.add.at(degrees, edge_arr[:, 1], 1)
    return coords.astype(np.int32, copy=False), edge_arr, degrees


def _face_endpoint_mask(coords, degrees, core_shape, axis: int, side: str, border: int = 1):
    """Endpoints that sit on a core face (local coordinates)."""
    if len(coords) == 0:
        return np.zeros((0,), dtype=bool)
    terminal = degrees == 1
    coord = coords[:, axis]
    if side == "min":
        on_face = coord <= int(border)
    else:
        on_face = coord >= int(core_shape[axis] - 1 - border)
    return terminal & on_face


def stitch_chunk_graphs(chunks: list[dict], *, max_distance_um: float) -> pd.DataFrame:
    """Join degree-1 endpoints on adjacent faces of neighboring chunks."""
    if len(chunks) < 2:
        return pd.DataFrame()

    by_index = {tuple(item["chunk_index"]): item for item in chunks}
    face_pairs = [
        ("z", 0, "max", "min", (1, 0, 0)),
        ("y", 1, "max", "min", (0, 1, 0)),
        ("x", 2, "max", "min", (0, 0, 1)),
    ]
    rows = []
    used = set()
    stitch_id = 0
    for chunk_index, item in by_index.items():
        for axis_name, axis, side_a, side_b, delta in face_pairs:
            neighbor_index = tuple(chunk_index[i] + delta[i] for i in range(3))
            other = by_index.get(neighbor_index)
            if other is None:
                continue
            mask_a = _face_endpoint_mask(item["local_coords"], item["degrees"], item["core_shape"], axis, side_a)
            mask_b = _face_endpoint_mask(other["local_coords"], other["degrees"], other["core_shape"], axis, side_b)
            if not np.any(mask_a) or not np.any(mask_b):
                continue
            coords_a = item["global_coords_um"][mask_a]
            coords_b = other["global_coords_um"][mask_b]
            ids_a = np.flatnonzero(mask_a)
            ids_b = np.flatnonzero(mask_b)
            tree = cKDTree(coords_b)
            distances, indices = tree.query(coords_a, distance_upper_bound=float(max_distance_um))
            candidates = []
            for a_i, (distance_um, b_i) in enumerate(zip(distances, indices)):
                if not np.isfinite(distance_um) or int(b_i) >= len(ids_b):
                    continue
                candidates.append((float(distance_um), int(ids_a[a_i]), int(ids_b[int(b_i)])))
            candidates.sort(key=lambda row: row[0])
            for distance_um, a_local, b_local in candidates:
                a_key = (chunk_index, int(a_local))
                b_key = (neighbor_index, int(b_local))
                if a_key in used or b_key in used:
                    continue
                used.add(a_key)
                used.add(b_key)
                src = item["global_coords_um"][a_local]
                dst = other["global_coords_um"][b_local]
                rows.append(
                    {
                        "stitch_id": stitch_id,
                        "axis": axis_name,
                        "chunk_a": ".".join(str(v) for v in chunk_index),
                        "chunk_b": ".".join(str(v) for v in neighbor_index),
                        "source_z_um": float(src[0]),
                        "source_y_um": float(src[1]),
                        "source_x_um": float(src[2]),
                        "target_z_um": float(dst[0]),
                        "target_y_um": float(dst[1]),
                        "target_x_um": float(dst[2]),
                        "length_um": float(distance_um),
                        "is_stitch": True,
                    }
                )
                stitch_id += 1
    return pd.DataFrame(rows)


def stitch_face_endpoints(chunk_metas: list[dict], *, max_distance_um: float) -> pd.DataFrame:
    """Stitch using per-face endpoint arrays stored on each chunk meta."""
    by_index = {tuple(item["chunk_index"]): item for item in chunk_metas}
    face_pairs = [
        ("z", "z_max", "z_min", (1, 0, 0)),
        ("y", "y_max", "y_min", (0, 1, 0)),
        ("x", "x_max", "x_min", (0, 0, 1)),
    ]
    rows = []
    used = set()
    stitch_id = 0
    for chunk_index, item in by_index.items():
        faces_a = item.get("face_endpoints_um") or {}
        for axis_name, key_a, key_b, delta in face_pairs:
            neighbor_index = tuple(chunk_index[i] + delta[i] for i in range(3))
            other = by_index.get(neighbor_index)
            if other is None:
                continue
            coords_a = np.asarray(faces_a.get(key_a, []), dtype=np.float64)
            coords_b = np.asarray((other.get("face_endpoints_um") or {}).get(key_b, []), dtype=np.float64)
            if coords_a.size == 0 or coords_b.size == 0:
                continue
            coords_a = np.atleast_2d(coords_a)
            coords_b = np.atleast_2d(coords_b)
            tree = cKDTree(coords_b)
            distances, indices = tree.query(coords_a, distance_upper_bound=float(max_distance_um))
            candidates = []
            for a_i, (distance_um, b_i) in enumerate(zip(distances, indices)):
                if not np.isfinite(distance_um) or int(b_i) >= len(coords_b):
                    continue
                candidates.append((float(distance_um), a_i, int(b_i)))
            candidates.sort(key=lambda row: row[0])
            for distance_um, a_i, b_i in candidates:
                a_key = (chunk_index, key_a, a_i)
                b_key = (neighbor_index, key_b, b_i)
                if a_key in used or b_key in used:
                    continue
                used.add(a_key)
                used.add(b_key)
                src = coords_a[a_i]
                dst = coords_b[b_i]
                rows.append(
                    {
                        "stitch_id": stitch_id,
                        "axis": axis_name,
                        "chunk_a": ".".join(str(v) for v in chunk_index),
                        "chunk_b": ".".join(str(v) for v in neighbor_index),
                        "source_z_um": float(src[0]),
                        "source_y_um": float(src[1]),
                        "source_x_um": float(src[2]),
                        "target_z_um": float(dst[0]),
                        "target_y_um": float(dst[1]),
                        "target_x_um": float(dst[2]),
                        "length_um": float(distance_um),
                        "is_stitch": True,
                    }
                )
                stitch_id += 1
    return pd.DataFrame(rows)


def process_chunk(
    mask_zarr,
    chunk_index,
    *,
    resolution_xyz,
    foreground_label=1,
    dust_threshold=100,
    halo_zyx=(2, 4, 4),
    build_edge_table: bool = True,
) -> dict:
    chunks = tuple(int(v) for v in mask_zarr.chunks)
    shape = tuple(int(v) for v in mask_zarr.shape)
    chunk_index = tuple(int(v) for v in chunk_index)
    core_slices = chunk_index_to_slices(chunk_index, chunks, shape)
    core_shape = tuple(cs.stop - cs.start for cs in core_slices)
    empty = {
        "chunk_index": chunk_index,
        "core_slices": core_slices,
        "core_shape": core_shape,
        "skeleton": np.zeros(core_shape, dtype=bool),
        "dbf": np.zeros(core_shape, dtype=np.float32),
        "local_coords": np.empty((0, 3), dtype=np.int32),
        "edges": np.empty((0, 2), dtype=np.int32),
        "degrees": np.empty((0,), dtype=np.int32),
        "global_coords_um": np.empty((0, 3), dtype=np.float64),
        "radii_um": np.empty((0,), dtype=np.float32),
        "edge_table": pd.DataFrame(),
        "num_skeleton_voxels": 0,
        "num_edges": 0,
        "mask_voxels": 0,
        "face_endpoints_um": {},
    }

    core = np.asarray(mask_zarr[core_slices])
    if foreground_label is None:
        core_binary = core > 0
    else:
        core_binary = core == foreground_label
    core_mask_voxels = int(np.count_nonzero(core_binary))
    if core_mask_voxels == 0:
        return empty

    expanded_slices = expand_slices(core_slices, halo_zyx, shape)
    expanded = np.asarray(mask_zarr[expanded_slices])
    if foreground_label is None:
        binary = expanded > 0
    else:
        binary = expanded == foreground_label

    skeleton_exp, dbf = skeletonize_binary_edt(
        binary,
        resolution_xyz=resolution_xyz,
        dust_threshold=dust_threshold,
    )

    core_offset = tuple(cs.start - es.start for cs, es in zip(core_slices, expanded_slices))
    core_view = tuple(slice(off, off + size) for off, size in zip(core_offset, core_shape))
    skeleton_core = skeleton_exp[core_view]
    dbf_core = dbf[core_view]
    local_coords, edges, degrees = skeleton_to_graph(skeleton_core)

    origin_zyx = np.array([s.start for s in core_slices], dtype=np.float64)
    scale_zyx = np.array(resolution_xyz_to_zyx(resolution_xyz), dtype=np.float64)
    global_voxel = local_coords.astype(np.float64) + origin_zyx
    global_um = global_voxel * scale_zyx
    radii = dbf_core[tuple(local_coords.T)] if len(local_coords) else np.empty((0,), dtype=np.float32)
    face_endpoints_um = {
        f"{axis}_{side}": global_um[_face_endpoint_mask(local_coords, degrees, core_shape, axis_i, side)]
        for axis_i, axis in enumerate(("z", "y", "x"))
        for side in ("min", "max")
    }

    edge_table = pd.DataFrame()
    if build_edge_table and len(edges):
        edge_rows = []
        for edge_id, (u, v) in enumerate(edges):
            src = global_um[int(u)]
            dst = global_um[int(v)]
            edge_rows.append(
                {
                    "chunk_index": ".".join(str(v) for v in chunk_index),
                    "edge_id": int(edge_id),
                    "source_z_um": float(src[0]),
                    "source_y_um": float(src[1]),
                    "source_x_um": float(src[2]),
                    "target_z_um": float(dst[0]),
                    "target_y_um": float(dst[1]),
                    "target_x_um": float(dst[2]),
                    "length_um": float(np.linalg.norm(dst - src)),
                    "source_radius_um": float(radii[int(u)]),
                    "target_radius_um": float(radii[int(v)]),
                    "is_stitch": False,
                }
            )
        edge_table = pd.DataFrame(edge_rows)

    return {
        "chunk_index": chunk_index,
        "core_slices": core_slices,
        "core_shape": core_shape,
        "skeleton": skeleton_core,
        "dbf": dbf_core,
        "local_coords": local_coords,
        "edges": edges,
        "degrees": degrees,
        "global_coords_um": global_um,
        "radii_um": radii,
        "edge_table": edge_table,
        "num_skeleton_voxels": int(len(local_coords)),
        "num_edges": int(len(edges)),
        "mask_voxels": core_mask_voxels,
        "face_endpoints_um": face_endpoints_um,
    }


def _union_roi(chunk_results: list[dict]):
    starts = np.array([[s.start for s in item["core_slices"]] for item in chunk_results])
    stops = np.array([[s.stop for s in item["core_slices"]] for item in chunk_results])
    start = starts.min(axis=0)
    stop = stops.max(axis=0)
    return tuple(slice(int(a), int(b)) for a, b in zip(start, stop)), tuple(int(v) for v in start)


def edge_table_to_vectors(edge_table: pd.DataFrame, *, origin_zyx, resolution_xyz):
    if edge_table is None or edge_table.empty:
        return np.empty((0, 2, 3), dtype=np.float64)
    scale_zyx = np.array(resolution_xyz_to_zyx(resolution_xyz), dtype=np.float64)
    origin = np.array(origin_zyx, dtype=np.float64)
    vectors = []
    for _, row in edge_table.iterrows():
        src = np.array([row["source_z_um"], row["source_y_um"], row["source_x_um"]], dtype=np.float64)
        dst = np.array([row["target_z_um"], row["target_y_um"], row["target_x_um"]], dtype=np.float64)
        src_px = src / scale_zyx - origin
        dst_px = dst / scale_zyx - origin
        vectors.append(np.stack([src_px, dst_px - src_px], axis=0))
    return np.asarray(vectors, dtype=np.float64)


def mosaic_skeleton(chunk_results: list[dict], roi, origin_zyx) -> np.ndarray:
    shape = tuple(s.stop - s.start for s in roi)
    out = np.zeros(shape, dtype=np.uint8)
    origin = np.array(origin_zyx, dtype=np.int64)
    for item in chunk_results:
        local = item["local_coords"]
        if len(local) == 0:
            continue
        core_origin = np.array([s.start for s in item["core_slices"]], dtype=np.int64)
        placed = local + (core_origin - origin)
        out[placed[:, 0], placed[:, 1], placed[:, 2]] = 1
    return out


def graph_to_polylines(coords: np.ndarray, edges: np.ndarray, degrees: np.ndarray) -> list[np.ndarray]:
    """Collapse voxel graphs into polylines between endpoints/junctions."""
    if len(coords) == 0:
        return []
    adjacency: list[list[int]] = [[] for _ in range(len(coords))]
    for u, v in edges:
        adjacency[int(u)].append(int(v))
        adjacency[int(v)].append(int(u))

    visited_edges: set[tuple[int, int]] = set()
    polylines: list[np.ndarray] = []

    def walk(start: int, nxt: int) -> list[int]:
        path = [start, nxt]
        prev, current = start, nxt
        while degrees[current] == 2:
            nbrs = adjacency[current]
            candidates = [n for n in nbrs if n != prev]
            if not candidates:
                break
            prev, current = current, candidates[0]
            path.append(current)
            if current == start:
                break
        return path

    starts = [i for i, d in enumerate(degrees) if d != 2]
    if not starts:
        starts = [0]
    for start in starts:
        for nbr in adjacency[start]:
            key = (min(start, nbr), max(start, nbr))
            if key in visited_edges:
                continue
            path = walk(start, nbr)
            for a, b in zip(path, path[1:]):
                visited_edges.add((min(a, b), max(a, b)))
            polylines.append(coords[np.asarray(path, dtype=np.int64)].astype(np.float32))
    return polylines


def show_in_napari(
    image,
    mask,
    skeleton,
    intra_vectors,
    stitch_vectors,
    *,
    polylines=None,
    screenshot_path: Path | None = None,
    view: bool = True,
):
    import napari

    viewer = napari.Viewer(ndisplay=3)
    if image is not None:
        viewer.add_image(image, name="image", blending="additive", colormap="gray")
    if mask is not None:
        viewer.add_labels((mask > 0).astype(np.uint8), name="mask", opacity=0.25)
    if skeleton is not None:
        viewer.add_labels(skeleton.astype(np.uint8), name="skeleton", opacity=0.9)
    if polylines:
        viewer.add_shapes(
            polylines,
            name="edges",
            shape_type="path",
            edge_color="yellow",
            edge_width=0.8,
            face_color="transparent",
        )
    elif len(intra_vectors):
        viewer.add_vectors(
            intra_vectors,
            name="edges",
            edge_width=0.6,
            edge_color="yellow",
            vector_style="line",
            length=1.0,
        )
    if len(stitch_vectors):
        viewer.add_vectors(
            stitch_vectors,
            name="stitch_edges",
            edge_width=1.6,
            edge_color="cyan",
            vector_style="line",
            length=1.0,
        )
    if screenshot_path is not None:
        try:
            screenshot_path.parent.mkdir(parents=True, exist_ok=True)
            viewer.screenshot(str(screenshot_path), canvas_only=True)
            logger.info("Wrote screenshot %s", screenshot_path)
        except Exception:
            logger.exception("Could not write napari screenshot")
    if view:
        napari.run()
    else:
        viewer.close()
    return viewer


def run_preview(
    mask_zarr_path,
    image_zarr_path,
    chunk_indices,
    *,
    output_dir,
    dataset_name="0",
    resolution_xyz=(1.8, 1.8, 2.0),
    foreground_label=1,
    dust_threshold=100,
    halo_zyx=(2, 4, 4),
    stitch_max_distance_um=5.0,
    view=True,
    screenshot_path=None,
) -> dict:
    mask_zarr = open_zarr_dataset(mask_zarr_path, dataset_name=dataset_name)
    image_zarr = open_zarr_dataset(image_zarr_path, dataset_name=dataset_name) if image_zarr_path else None
    resolution_xyz = parse_resolution_xyz(resolution_xyz)
    halo_zyx = parse_triplet_int(halo_zyx)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    chunk_results = []
    for chunk_index in chunk_indices:
        logger.info("Skeletonizing chunk %s", chunk_index)
        result = process_chunk(
            mask_zarr,
            chunk_index,
            resolution_xyz=resolution_xyz,
            foreground_label=foreground_label,
            dust_threshold=dust_threshold,
            halo_zyx=halo_zyx,
        )
        logger.info(
            "  %s: mask=%d skeleton=%d edges=%d",
            chunk_index,
            result["mask_voxels"],
            result["num_skeleton_voxels"],
            result["num_edges"],
        )
        chunk_results.append(result)

    stitch_table = stitch_chunk_graphs(chunk_results, max_distance_um=stitch_max_distance_um)
    edge_table = pd.concat(
        [item["edge_table"] for item in chunk_results if not item["edge_table"].empty],
        ignore_index=True,
    ) if any(not item["edge_table"].empty for item in chunk_results) else pd.DataFrame()
    if not stitch_table.empty:
        extra = stitch_table.rename(columns={"chunk_a": "chunk_index"}).copy()
        extra["edge_id"] = np.arange(len(extra))
        extra["source_radius_um"] = np.nan
        extra["target_radius_um"] = np.nan
        keep = [
            "chunk_index",
            "edge_id",
            "source_z_um",
            "source_y_um",
            "source_x_um",
            "target_z_um",
            "target_y_um",
            "target_x_um",
            "length_um",
            "source_radius_um",
            "target_radius_um",
            "is_stitch",
        ]
        edge_table = pd.concat([edge_table, extra[keep]], ignore_index=True)

    roi, origin_zyx = _union_roi(chunk_results)
    image = np.asarray(image_zarr[roi]) if image_zarr is not None else None
    mask = np.asarray(mask_zarr[roi])
    skeleton = mosaic_skeleton(chunk_results, roi, origin_zyx)
    intra = edge_table_to_vectors(
        edge_table.loc[~edge_table["is_stitch"]] if not edge_table.empty else edge_table,
        origin_zyx=origin_zyx,
        resolution_xyz=resolution_xyz,
    )
    stitch_vectors = edge_table_to_vectors(
        edge_table.loc[edge_table["is_stitch"]] if not edge_table.empty else edge_table,
        origin_zyx=origin_zyx,
        resolution_xyz=resolution_xyz,
    )
    origin = np.array(origin_zyx, dtype=np.float64)
    polylines = []
    for item in chunk_results:
        paths = graph_to_polylines(item["local_coords"], item["edges"], item["degrees"])
        core_origin = np.array([s.start for s in item["core_slices"]], dtype=np.float64)
        shift = core_origin - origin
        for path in paths:
            if len(path) < 2:
                continue
            polylines.append(path.astype(np.float32) + shift.astype(np.float32))

    edge_csv = output_root / "skeleton_edges.csv"
    stitch_csv = output_root / "stitch_edges.csv"
    summary_path = output_root / "preview_summary.json"
    if not edge_table.empty:
        edge_table.to_csv(edge_csv, index=False)
    if not stitch_table.empty:
        stitch_table.to_csv(stitch_csv, index=False)
    summary = {
        "chunks": [".".join(str(v) for v in item["chunk_index"]) for item in chunk_results],
        "roi_start_zyx": list(origin_zyx),
        "roi_stop_zyx": [int(s.stop) for s in roi],
        "num_edges": int(len(edge_table)),
        "num_stitch_edges": int(len(stitch_table)),
        "per_chunk": [
            {
                "chunk_index": ".".join(str(v) for v in item["chunk_index"]),
                "mask_voxels": item["mask_voxels"],
                "num_skeleton_voxels": item["num_skeleton_voxels"],
                "num_edges": item["num_edges"],
            }
            for item in chunk_results
        ],
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    np.save(output_root / "skeleton.npy", skeleton)
    np.save(output_root / "mask_crop.npy", (mask > 0).astype(np.uint8))
    if image is not None:
        np.save(output_root / "image_crop.npy", image)

    if view or screenshot_path:
        show_in_napari(
            image,
            mask,
            skeleton,
            intra,
            stitch_vectors,
            polylines=polylines,
            screenshot_path=Path(screenshot_path) if screenshot_path else (output_root / "napari_overlay.png" if view else None),
            view=view,
        )
    return summary


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="EDT+thinning vessel centerlines on selected Zarr chunks, viewed in napari."
    )
    parser.add_argument("--mask_zarr", required=True)
    parser.add_argument("--image_zarr", default="")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--chunks",
        default="1.20.16,1.20.17,1.20.18",
        help="Comma-separated chunk indices z.y.x",
    )
    parser.add_argument("--dataset_name", default="0")
    parser.add_argument("--resolution_xyz", default="1.8,1.8,2.0")
    parser.add_argument("--dust_threshold", type=int, default=100)
    parser.add_argument("--halo_zyx", default="2,4,4")
    parser.add_argument("--stitch_max_distance_um", type=float, default=5.0)
    parser.add_argument("--no_view", action="store_true", help="Do not open the interactive napari window")
    parser.add_argument("--screenshot", default="", help="Optional screenshot path; default is output_dir/napari_overlay.png")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    chunk_indices = [parse_chunk_index(part) for part in args.chunks.split(",") if part.strip()]
    run_preview(
        args.mask_zarr,
        args.image_zarr or None,
        chunk_indices,
        output_dir=args.output_dir,
        dataset_name=args.dataset_name,
        resolution_xyz=args.resolution_xyz,
        dust_threshold=args.dust_threshold,
        halo_zyx=args.halo_zyx,
        stitch_max_distance_um=args.stitch_max_distance_um,
        view=not args.no_view,
        screenshot_path=args.screenshot or None,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
