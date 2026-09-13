"""VesselExpress-style reconstruction on an occupancy-downsampled vessel mask.

VesselExpress (https://github.com/RUB-Bioinf/VesselExpress) loads a binary volume,
runs scikit-image Lee thinning, builds a 26-connected graph, and measures
segment diameter from the binary mask.

This module keeps that skeletonizer, but:

- occupancy-downsamples a Zarr mask instead of max-pool (less diameter inflation)
- runs Lee thinning in-memory only when the downsampled uint8 array fits
- samples radius from the **native** mask via chunked EDT, so coarse voxels do
  not quantize capillary diameter to one downsampled voxel
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import shutil
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage
from tqdm import tqdm

from pipeline_modules.tubule_reconstruction.edt_reconstruction import (
    graph_to_polylines,
    parse_chunk_index,
    skeleton_to_graph,
)
from pipeline_modules.tubule_reconstruction.kimimaro_reconstruction import (
    chunk_index_to_slices,
    expand_slices,
    open_zarr_dataset,
    parse_resolution_xyz,
    parse_triplet_int,
    resolution_xyz_to_zyx,
)
from pipeline_modules.tubule_reconstruction.mask_downsample import (
    block_reduce_binary,
    downsample_binary_mask_zarr,
    estimate_downsampled_memory,
)

logger = logging.getLogger(__name__)

# ~8 GiB uint8. Lee thinning peak is a few copies; 128 GB hosts ~6e9 voxels at 4x.
DEFAULT_MAX_IN_MEMORY_VOXELS = 8_000_000_000


try:
    from skimage.morphology import skeletonize as _sk_skeletonize
except ImportError:  # older skimage
    from skimage.morphology import skeletonize_3d as _sk_skeletonize


def _now_iso() -> str:
    return datetime.now().astimezone().replace(microsecond=0).isoformat()


def _align_slice_to_factor(core_slice: slice, dim: int, factor: int) -> slice:
    start = (int(core_slice.start) // factor) * factor
    stop = int(np.ceil(int(core_slice.stop) / factor) * factor)
    return slice(start, min(stop, int(dim)))


def upsample_coarse_coords(coarse_zyx: np.ndarray, native_shape, factor: int) -> np.ndarray:
    """Map downsampled voxel indices to native-resolution voxel centres."""
    factor = max(1, int(factor))
    pts = (np.asarray(coarse_zyx, dtype=np.float64) + 0.5) * factor
    clipped = np.clip(np.rint(pts), 0, np.asarray(native_shape, dtype=np.float64) - 1)
    return clipped.astype(np.int64)


def rasterize_points(shape, points_zyx: np.ndarray) -> np.ndarray:
    volume = np.zeros(tuple(int(v) for v in shape), dtype=np.uint8)
    if len(points_zyx) == 0:
        return volume
    pts = np.asarray(points_zyx, dtype=np.int64)
    volume[pts[:, 0], pts[:, 1], pts[:, 2]] = 1
    return volume


def prune_terminal_spurs(coords, edges, degrees, scale_zyx, max_length_um: float):
    """Remove terminal branches shorter than ``max_length_um`` (iterative).

    Length is Euclidean in microns using ``coords * scale_zyx``. Junctions that
    drop to degree 2 after pruning are no longer counted as branch points.
    Isolated short filaments (terminal-to-terminal) are kept.
    """
    coords = np.asarray(coords)
    edges = np.asarray(edges)
    degrees = np.asarray(degrees)
    if max_length_um <= 0 or len(coords) == 0 or len(edges) == 0:
        return coords, edges, degrees, 0

    um = coords.astype(np.float64) * np.asarray(scale_zyx, dtype=np.float64)
    adj: list[set[int]] = [set() for _ in range(len(coords))]
    for u, v in edges:
        adj[int(u)].add(int(v))
        adj[int(v)].add(int(u))
    alive = np.ones(len(coords), dtype=bool)
    pruned = 0
    changed = True
    while changed:
        changed = False
        terminals = [i for i in range(len(coords)) if alive[i] and sum(1 for n in adj[i] if alive[n]) == 1]
        for terminal in terminals:
            if not alive[terminal]:
                continue
            live = [n for n in adj[terminal] if alive[n]]
            if len(live) != 1:
                continue
            path = [terminal]
            prev = None
            current = terminal
            length = 0.0
            while True:
                nxts = [n for n in adj[current] if alive[n] and n != prev]
                if not nxts:
                    break
                nxt = nxts[0]
                length += float(np.linalg.norm(um[nxt] - um[current]))
                path.append(nxt)
                deg = sum(1 for n in adj[nxt] if alive[n])
                if deg != 2:
                    break
                prev, current = current, nxt
            if length > float(max_length_um) or len(path) < 2:
                continue
            junction = path[-1]
            if sum(1 for n in adj[junction] if alive[n]) < 3:
                continue
            for node in path[:-1]:
                alive[node] = False
            pruned += 1
            changed = True

    keep = np.flatnonzero(alive)
    remap = -np.ones(len(coords), dtype=np.int64)
    remap[keep] = np.arange(len(keep))
    new_coords = coords[keep]
    new_edges = []
    for u, v in edges:
        a, b = remap[int(u)], remap[int(v)]
        if a >= 0 and b >= 0:
            new_edges.append((int(a), int(b)))
    new_edge_arr = np.asarray(new_edges, dtype=np.int32) if new_edges else np.empty((0, 2), dtype=np.int32)
    new_degrees = np.zeros(len(new_coords), dtype=np.int32)
    if len(new_edge_arr):
        np.add.at(new_degrees, new_edge_arr[:, 0], 1)
        np.add.at(new_degrees, new_edge_arr[:, 1], 1)
    return new_coords, new_edge_arr, new_degrees, int(pruned)


def _adj_from_edges(n_nodes: int, edges: np.ndarray) -> list[set[int]]:
    adj: list[set[int]] = [set() for _ in range(n_nodes)]
    for u, v in np.asarray(edges):
        adj[int(u)].add(int(v))
        adj[int(v)].add(int(u))
    return adj


def _rebuild_graph(coords, adj, alive: np.ndarray):
    coords = np.asarray(coords)
    keep = np.flatnonzero(alive)
    remap = -np.ones(len(coords), dtype=np.int64)
    remap[keep] = np.arange(len(keep))
    new_coords = coords[keep]
    new_edges = []
    for i in keep:
        for j in adj[int(i)]:
            if not alive[int(j)] or int(j) <= int(i):
                continue
            a, b = remap[int(i)], remap[int(j)]
            if a >= 0 and b >= 0:
                new_edges.append((int(a), int(b)))
    new_edge_arr = np.asarray(new_edges, dtype=np.int32) if new_edges else np.empty((0, 2), dtype=np.int32)
    new_degrees = np.zeros(len(new_coords), dtype=np.int32)
    if len(new_edge_arr):
        np.add.at(new_degrees, new_edge_arr[:, 0], 1)
        np.add.at(new_degrees, new_edge_arr[:, 1], 1)
    return new_coords, new_edge_arr, new_degrees


def _unit(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    if norm < 1e-9:
        return vector
    return vector / norm


def resolve_triple_cliques(coords, edges, degrees, scale_zyx):
    """Drop the longest edge of every 3-voxel clique (VesselExpress)."""
    coords = np.asarray(coords)
    edges = np.asarray(edges)
    degrees = np.asarray(degrees)
    if len(coords) == 0 or len(edges) == 0:
        return coords, edges, degrees, 0
    um = coords.astype(np.float64) * np.asarray(scale_zyx, dtype=np.float64)
    adj = _adj_from_edges(len(coords), edges)
    triangles = []
    for a in range(len(coords)):
        for b in adj[a]:
            if b <= a:
                continue
            for c in adj[a] & adj[b]:
                if c <= b:
                    continue
                triangles.append((a, b, c))
    removed = 0
    for a, b, c in triangles:
        if b not in adj[a] or c not in adj[a] or c not in adj[b]:
            continue
        trio = ((a, b), (b, c), (a, c))
        lengths = [float(np.linalg.norm(um[i] - um[j])) for i, j in trio]
        u, v = trio[int(np.argmax(lengths))]
        if v in adj[u]:
            adj[u].discard(v)
            adj[v].discard(u)
            removed += 1
    alive = np.array([len(adj[i]) > 0 for i in range(len(coords))], dtype=bool)
    new_coords, new_edges, new_degrees = _rebuild_graph(coords, adj, alive)
    return new_coords, new_edges, new_degrees, int(removed)


def merge_nearby_branch_points(coords, edges, degrees, scale_zyx, distance_um: float):
    """Collapse branch-point clusters closer than ``distance_um`` to one centroid."""
    from scipy.spatial import cKDTree

    coords = np.asarray(coords, dtype=np.float64)
    edges = np.asarray(edges)
    degrees = np.asarray(degrees)
    if distance_um <= 0 or len(coords) == 0:
        return coords, edges, degrees, 0
    um = coords * np.asarray(scale_zyx, dtype=np.float64)
    bp = np.flatnonzero(degrees >= 3)
    if len(bp) < 2:
        return coords.astype(np.int32, copy=False) if coords.dtype != np.int32 else coords, edges, degrees, 0

    tree = cKDTree(um[bp])
    pairs = tree.query_pairs(r=float(distance_um))
    if not pairs:
        return coords, edges, degrees, 0

    parent = list(range(len(bp)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i, j in pairs:
        union(i, j)
    clusters: dict[int, list[int]] = {}
    for i in range(len(bp)):
        clusters.setdefault(find(i), []).append(i)

    adj = _adj_from_edges(len(coords), edges)
    alive = np.ones(len(coords), dtype=bool)
    merged = 0
    for members in clusters.values():
        if len(members) < 2:
            continue
        keys = [int(bp[m]) for m in members]
        degrees_now = [len(adj[k]) for k in keys]
        representative = keys[int(np.argmax(degrees_now))]
        coords[representative] = np.mean(coords[keys], axis=0)
        for k in keys:
            if k == representative:
                continue
            for neighbor in list(adj[k]):
                adj[neighbor].discard(k)
                if neighbor != representative and alive[neighbor]:
                    adj[neighbor].add(representative)
                    adj[representative].add(neighbor)
            adj[k].clear()
            alive[k] = False
            merged += 1
        adj[representative].discard(representative)
    new_coords, new_edges, new_degrees = _rebuild_graph(coords, adj, alive)
    return new_coords, new_edges, new_degrees, int(merged)


def suppress_collinear_kinks(
    coords,
    edges,
    degrees,
    scale_zyx,
    *,
    through_angle_deg: float = 150.0,
    kink_align_deg: float = 45.0,
):
    """Remove degree-3 voxel kinks on an otherwise straight 26-connected path.

    If two arms form an almost-straight through line and the third arm lies
    within ``kink_align_deg`` of that line, the extra edge is a staircase
    artifact, not an anatomical branch.
    """
    coords = np.asarray(coords)
    edges = np.asarray(edges)
    degrees = np.asarray(degrees)
    if len(coords) == 0 or len(edges) == 0:
        return coords, edges, degrees, 0
    um = coords.astype(np.float64) * np.asarray(scale_zyx, dtype=np.float64)
    adj = _adj_from_edges(len(coords), edges)
    through_dot_max = float(np.cos(np.deg2rad(through_angle_deg)))
    align_min = float(np.cos(np.deg2rad(kink_align_deg)))
    removed = 0
    for i in range(len(coords)):
        nbrs = sorted(adj[i])
        if len(nbrs) != 3:
            continue
        units = [_unit(um[n] - um[i]) for n in nbrs]
        best = (0, 1, 1.0)
        for a in range(3):
            for b in range(a + 1, 3):
                dot = float(np.dot(units[a], units[b]))
                if dot < best[2]:
                    best = (a, b, dot)
        if best[2] > through_dot_max:
            continue
        leftover = ({0, 1, 2} - {best[0], best[1]}).pop()
        left_u = units[leftover]
        align = max(abs(float(np.dot(left_u, units[best[0]]))), abs(float(np.dot(left_u, units[best[1]]))))
        if align + 1e-9 < align_min:
            continue
        extra = nbrs[leftover]
        if extra in adj[i]:
            adj[i].discard(extra)
            adj[extra].discard(i)
            removed += 1
    alive = np.array([len(adj[i]) > 0 for i in range(len(coords))], dtype=bool)
    new_coords, new_edges, new_degrees = _rebuild_graph(coords, adj, alive)
    return new_coords, new_edges, new_degrees, int(removed)


def clean_skeleton_topology(
    coords,
    edges,
    degrees,
    scale_zyx,
    *,
    prune_spurs_max_length_um: float = 0.0,
    merge_branch_points_distance_um: float = 0.0,
    through_angle_deg: float = 150.0,
    kink_align_deg: float = 45.0,
):
    """Clique split, kink suppression, spur prune, then nearby branch-point merge."""
    stats = {"cliques": 0, "kinks": 0, "spurs": 0, "merged_branch_points": 0}
    coords, edges, degrees, n_clique = resolve_triple_cliques(coords, edges, degrees, scale_zyx)
    stats["cliques"] = int(n_clique)
    coords, edges, degrees, n_kink = suppress_collinear_kinks(
        coords,
        edges,
        degrees,
        scale_zyx,
        through_angle_deg=through_angle_deg,
        kink_align_deg=kink_align_deg,
    )
    stats["kinks"] = int(n_kink)
    coords, edges, degrees, n_spur = prune_terminal_spurs(
        coords, edges, degrees, scale_zyx, prune_spurs_max_length_um
    )
    stats["spurs"] = int(n_spur)
    coords, edges, degrees, n_merge = merge_nearby_branch_points(
        coords, edges, degrees, scale_zyx, merge_branch_points_distance_um
    )
    stats["merged_branch_points"] = int(n_merge)
    coords, edges, degrees, n_spur2 = prune_terminal_spurs(
        coords, edges, degrees, scale_zyx, prune_spurs_max_length_um
    )
    stats["spurs"] += int(n_spur2)
    return coords, edges, degrees, stats


def skeletonize_lee(binary: np.ndarray) -> np.ndarray:
    """Lee 3D thinning, the same call VesselExpress uses."""
    binary = np.asarray(binary, dtype=bool)
    if not np.any(binary):
        return np.zeros(binary.shape, dtype=bool)
    try:
        skeleton = _sk_skeletonize(binary, method="lee")
    except TypeError:
        skeleton = _sk_skeletonize(binary)
    return np.asarray(skeleton, dtype=bool) & binary


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


def _binary_from_volume(volume: np.ndarray, foreground_label) -> np.ndarray:
    if foreground_label is None:
        return np.asarray(volume) > 0
    return np.asarray(volume) == foreground_label


def fill_small_holes(binary: np.ndarray, max_voxels: int) -> tuple[np.ndarray, int, int]:
    """Fill enclosed background cavities no larger than ``max_voxels``.

    Background that touches the volume border is never filled, so interstitial
    space in a cropped capillary bed is left open unless it is a true cavity.
    ``max_voxels <= 0`` fills every enclosed cavity.

    Returns ``(filled_mask, n_holes, voxels_filled)``.
    """
    binary = np.asarray(binary, dtype=bool)
    filled_all = ndimage.binary_fill_holes(binary)
    holes = filled_all & ~binary
    if not np.any(holes):
        return binary, 0, 0
    labeled, n_features = ndimage.label(holes)
    if n_features == 0:
        return binary, 0, 0
    counts = np.bincount(labeled.ravel())
    keep = np.zeros(counts.shape[0], dtype=bool)
    if max_voxels <= 0:
        keep[1:] = True
    else:
        keep[1:] = counts[1:] <= int(max_voxels)
    selected = keep[labeled]
    n_holes = int(keep[1:].sum())
    voxels_filled = int(np.count_nonzero(selected))
    if voxels_filled == 0:
        return binary, 0, 0
    return binary | selected, n_holes, voxels_filled


def hole_diameter_to_voxels(max_hole_diameter_um: float, resolution_xyz) -> int:
    """Voxel budget of a sphere with the given diameter in microns."""
    if max_hole_diameter_um <= 0:
        return 0
    resolution_xyz = parse_resolution_xyz(resolution_xyz)
    voxel_um3 = float(resolution_xyz[0] * resolution_xyz[1] * resolution_xyz[2])
    radius = float(max_hole_diameter_um) / 2.0
    volume_um3 = (4.0 / 3.0) * np.pi * radius**3
    return max(1, int(np.round(volume_um3 / voxel_um3)))


def despike_thick_vessels(binary: np.ndarray, *, thin_max_voxels: int = 2) -> tuple[np.ndarray, int]:
    """Smooth burrs on thick vessels without opening 1-2 voxel filaments.

    Voxels whose EDT is below ``thin_max_voxels`` (thin capillaries) are copied
    unchanged. Only the dilated thick core is 6-connected opened, then unioned
    back with the thin remainder.
    """
    binary = np.asarray(binary, dtype=bool)
    thin_max_voxels = max(1, int(thin_max_voxels))
    edt = ndimage.distance_transform_edt(binary)
    thick_core = edt >= float(thin_max_voxels)
    if not np.any(thick_core):
        return binary, 0
    thick = ndimage.binary_dilation(thick_core, iterations=thin_max_voxels) & binary
    opened = ndimage.binary_opening(thick, structure=ndimage.generate_binary_structure(3, 3))
    out = opened | (binary & ~thick)
    removed = int(np.count_nonzero(binary)) - int(np.count_nonzero(out))
    return out, max(0, removed)


def sample_native_edt_radii(
    mask_zarr,
    coarse_zyx: np.ndarray,
    *,
    factor: int = 1,
    resolution_xyz=(1.0, 1.0, 1.0),
    foreground_label: int | None = 1,
    halo_zyx=(32, 32, 32),
    workers: int = 1,
    mask_zarr_path=None,
    dataset_name: str = "0",
    progress_callback=None,
) -> np.ndarray:
    """Native-resolution EDT radius (um) for each downsampled skeleton voxel.

    Each coarse voxel maps to a ``factor^3`` native window. The radius is the
    maximum EDT inside that window, so the measurement sits on the medial axis
    even when the coarse skeleton is quantized to a neighbouring coarse cell.
    """
    coarse_zyx = np.asarray(coarse_zyx, dtype=np.int64)
    n_points = int(coarse_zyx.shape[0])
    radii = np.full((n_points,), np.nan, dtype=np.float32)
    if n_points == 0:
        return radii

    factor = max(1, int(factor))
    shape = tuple(int(v) for v in mask_zarr.shape)
    chunks = tuple(int(v) for v in (getattr(mask_zarr, "chunks", None) or shape))
    scale_zyx = np.array(resolution_xyz_to_zyx(resolution_xyz), dtype=np.float64)
    halo_zyx = tuple(int(v) + int(factor) for v in halo_zyx)

    native_start = coarse_zyx * factor
    native_start[:, 0] = np.clip(native_start[:, 0], 0, max(shape[0] - 1, 0))
    native_start[:, 1] = np.clip(native_start[:, 1], 0, max(shape[1] - 1, 0))
    native_start[:, 2] = np.clip(native_start[:, 2], 0, max(shape[2] - 1, 0))

    chunk_of = np.stack(
        [native_start[:, axis] // max(chunks[axis], 1) for axis in range(3)],
        axis=1,
    )
    groups: dict[tuple[int, int, int], list[int]] = {}
    for i, key in enumerate(chunk_of):
        groups.setdefault(tuple(int(v) for v in key), []).append(i)

    items = list(groups.items())
    workers = max(1, int(workers))
    done = 0
    total = len(items)
    if progress_callback is not None:
        progress_callback(done, total)

    def _apply_hits(hits: list[tuple[int, float]]) -> None:
        for i, value in hits:
            radii[i] = value

    def _local_job(chunk_index, indices):
        idx = np.asarray(indices, dtype=np.int64)
        return {
            "mask_path": str(mask_zarr_path) if mask_zarr_path else "",
            "dataset_name": dataset_name,
            "chunks": list(chunks),
            "halo_zyx": list(halo_zyx),
            "factor": int(factor),
            "foreground_label": foreground_label,
            "scale_zyx": scale_zyx.tolist(),
            "native_start": native_start[idx].tolist(),
            "global_indices": [int(v) for v in idx],
            "chunk_index": list(chunk_index),
        }

    if workers == 1 or total <= 1 or not mask_zarr_path:
        for chunk_index, indices in tqdm(items, desc="Native EDT radii"):
            core_slices = chunk_index_to_slices(chunk_index, chunks, shape)
            expanded = expand_slices(core_slices, halo_zyx, shape)
            block = np.asarray(mask_zarr[expanded])
            binary = _binary_from_volume(block, foreground_label)
            origin = np.array([s.start for s in expanded], dtype=np.int64)
            idx = np.asarray(indices, dtype=np.int64)
            hits = _edt_radii_from_binary(
                binary,
                origin,
                native_start[idx],
                [int(v) for v in idx],
                factor,
                scale_zyx,
            )
            _apply_hits(hits)
            done += 1
            if progress_callback is not None:
                try:
                    progress_callback(done, total)
                except Exception:
                    logger.debug("Native EDT progress callback failed", exc_info=True)
        return radii

    jobs = [_local_job(chunk_index, indices) for chunk_index, indices in items]
    in_flight = set()
    job_iter = iter(jobs)
    max_in_flight = max(workers * 2, workers)
    progress_bar = tqdm(total=total, desc="Native EDT radii")
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_init_edt_worker,
        initargs=(str(mask_zarr_path), dataset_name),
    ) as pool:
        try:
            for _ in range(min(max_in_flight, total)):
                in_flight.add(pool.submit(_edt_chunk_worker, next(job_iter)))
        except StopIteration:
            pass
        while in_flight:
            finished, in_flight = wait(in_flight, return_when=FIRST_COMPLETED)
            for future in finished:
                _apply_hits(future.result())
                done += 1
                progress_bar.update(1)
                if progress_callback is not None:
                    try:
                        progress_callback(done, total)
                    except Exception:
                        logger.debug("Native EDT progress callback failed", exc_info=True)
                try:
                    in_flight.add(pool.submit(_edt_chunk_worker, next(job_iter)))
                except StopIteration:
                    pass
    progress_bar.close()
    return radii


def _component_tables(
    skeleton: np.ndarray,
    radii_um: np.ndarray,
    *,
    scale_zyx: np.ndarray,
    origin_zyx: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split a whole-volume skeleton into connected-component tables."""
    vertex_rows: list[dict] = []
    edge_rows: list[dict] = []
    branch_rows: list[dict] = []
    if not np.any(skeleton):
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    labeled, n_features = ndimage.label(skeleton.astype(np.uint8), structure=np.ones((3, 3, 3), dtype=np.uint8))
    radius_volume = np.zeros(skeleton.shape, dtype=np.float32)
    skel_coords = np.argwhere(skeleton)
    if len(skel_coords) == len(radii_um):
        radius_volume[skel_coords[:, 0], skel_coords[:, 1], skel_coords[:, 2]] = radii_um

    for skeleton_id in range(1, n_features + 1):
        component = labeled == skeleton_id
        coords, edges, degrees = skeleton_to_graph(component)
        if len(coords) == 0:
            continue
        global_um = (coords.astype(np.float64) + 0.5 + origin_zyx) * scale_zyx
        local_radii = radius_volume[coords[:, 0], coords[:, 1], coords[:, 2]]
        for node_id, (point, radius) in enumerate(zip(global_um, local_radii)):
            vertex_rows.append(
                {
                    "skeleton_id": int(skeleton_id - 1),
                    "node_id": int(node_id),
                    "z_um": float(point[0]),
                    "y_um": float(point[1]),
                    "x_um": float(point[2]),
                    "radius_um": float(radius),
                }
            )
        for edge_id, (u, v) in enumerate(edges):
            src = global_um[int(u)]
            dst = global_um[int(v)]
            edge_rows.append(
                {
                    "skeleton_id": int(skeleton_id - 1),
                    "edge_id": int(edge_id),
                    "source_node": int(u),
                    "target_node": int(v),
                    "source_z_um": float(src[0]),
                    "source_y_um": float(src[1]),
                    "source_x_um": float(src[2]),
                    "target_z_um": float(dst[0]),
                    "target_y_um": float(dst[1]),
                    "target_x_um": float(dst[2]),
                    "edge_length_um": float(np.linalg.norm(dst - src)),
                    "source_radius_um": float(local_radii[int(u)]),
                    "target_radius_um": float(local_radii[int(v)]),
                    "is_stitch": False,
                }
            )
        index_of = {tuple(int(v) for v in row): i for i, row in enumerate(coords)}
        for branch_id, path in enumerate(graph_to_polylines(coords, edges, degrees)):
            if len(path) < 2:
                continue
            um = (path.astype(np.float64) + 0.5 + origin_zyx) * scale_zyx
            seg = np.linalg.norm(np.diff(um, axis=0), axis=1)
            radius_vals = []
            for zyx in np.rint(path).astype(np.int32):
                i = index_of.get((int(zyx[0]), int(zyx[1]), int(zyx[2])))
                if i is not None:
                    radius_vals.append(float(local_radii[i]))
            src, dst = um[0], um[-1]
            branch_rows.append(
                {
                    "skeleton_id": int(skeleton_id - 1),
                    "branch_id": int(branch_id),
                    "source_z_um": float(src[0]),
                    "source_y_um": float(src[1]),
                    "source_x_um": float(src[2]),
                    "target_z_um": float(dst[0]),
                    "target_y_um": float(dst[1]),
                    "target_x_um": float(dst[2]),
                    "length_um": float(seg.sum()) if len(seg) else 0.0,
                    "num_points": int(len(path)),
                    "mean_radius_um": float(np.mean(radius_vals)) if radius_vals else np.nan,
                    "is_stitch": False,
                }
            )

    return pd.DataFrame(vertex_rows), pd.DataFrame(edge_rows), pd.DataFrame(branch_rows)


def _component_tables_from_graph(coords, edges, degrees, radii_um, *, scale_zyx, origin_zyx):
    """Build CSV tables from the skeleton graph (no full-volume connected-component scan)."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    coords = np.asarray(coords)
    edges = np.asarray(edges)
    degrees = np.asarray(degrees)
    radii_um = np.asarray(radii_um, dtype=np.float32)
    n = int(len(coords))
    if n == 0:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    if len(edges):
        e = edges.astype(np.int64, copy=False)
        graph = coo_matrix(
            (
                np.ones(len(e) * 2, dtype=np.uint8),
                (np.concatenate([e[:, 0], e[:, 1]]), np.concatenate([e[:, 1], e[:, 0]])),
            ),
            shape=(n, n),
        )
        _, labels = connected_components(csgraph=graph, directed=False)
    else:
        labels = np.arange(n, dtype=np.int32)
    order = np.argsort(labels, kind="mergesort")
    labels_sorted = labels[order]
    splits = np.flatnonzero(np.diff(labels_sorted)) + 1
    groups = np.split(order, splits)
    local_id = np.empty(n, dtype=np.int64)
    for idx in groups:
        local_id[idx] = np.arange(len(idx), dtype=np.int64)

    global_um = (coords.astype(np.float64) + 0.5 + origin_zyx) * scale_zyx
    if len(radii_um) != n:
        radii_um = np.full(n, np.nan, dtype=np.float32)
    vertex_table = pd.DataFrame(
        {
            "skeleton_id": labels.astype(np.int32, copy=False),
            "node_id": local_id.astype(np.int32, copy=False),
            "z_um": global_um[:, 0],
            "y_um": global_um[:, 1],
            "x_um": global_um[:, 2],
            "radius_um": radii_um.astype(np.float64, copy=False),
        }
    )
    edge_rows: list[dict] = []
    branch_rows: list[dict] = []
    if len(edges):
        u = edges[:, 0].astype(np.int64, copy=False)
        v = edges[:, 1].astype(np.int64, copy=False)
        src_um = global_um[u]
        dst_um = global_um[v]
        edge_table = pd.DataFrame(
            {
                "skeleton_id": labels[u].astype(np.int32, copy=False),
                "edge_id": np.zeros(len(u), dtype=np.int32),
                "source_node": local_id[u].astype(np.int32, copy=False),
                "target_node": local_id[v].astype(np.int32, copy=False),
                "source_z_um": src_um[:, 0],
                "source_y_um": src_um[:, 1],
                "source_x_um": src_um[:, 2],
                "target_z_um": dst_um[:, 0],
                "target_y_um": dst_um[:, 1],
                "target_x_um": dst_um[:, 2],
                "edge_length_um": np.linalg.norm(dst_um - src_um, axis=1),
                "source_radius_um": radii_um[u],
                "target_radius_um": radii_um[v],
                "is_stitch": False,
            }
        )
        edge_table["edge_id"] = edge_table.groupby("skeleton_id").cumcount().astype(np.int32)
    else:
        edge_table = pd.DataFrame()

    for skeleton_id, idx in enumerate(groups):
        idx = np.asarray(idx, dtype=np.int64)
        sub_coords = coords[idx]
        sub_radii = radii_um[idx]
        sub_degrees = degrees[idx]
        if len(edges):
            keep = labels[u] == labels[idx[0]]
            sub_edges = np.stack([local_id[u[keep]], local_id[v[keep]]], axis=1) if np.any(keep) else np.empty((0, 2), dtype=np.int64)
        else:
            sub_edges = np.empty((0, 2), dtype=np.int64)
        index_of = {tuple(int(v) for v in row): i for i, row in enumerate(sub_coords)}
        for branch_id, path in enumerate(graph_to_polylines(sub_coords, sub_edges, sub_degrees)):
            if len(path) < 2:
                continue
            um = (path.astype(np.float64) + 0.5 + origin_zyx) * scale_zyx
            seg = np.linalg.norm(np.diff(um, axis=0), axis=1)
            radius_vals = []
            for zyx in np.rint(path).astype(np.int32):
                i = index_of.get((int(zyx[0]), int(zyx[1]), int(zyx[2])))
                if i is not None:
                    radius_vals.append(float(sub_radii[i]))
            src, dst = um[0], um[-1]
            length = float(seg.sum()) if len(seg) else 0.0
            euclidean = float(np.linalg.norm(dst - src))
            p0 = tuple(int(v) for v in np.rint(path[0]))
            p1 = tuple(int(v) for v in np.rint(path[-1]))
            start_i = index_of.get(p0)
            end_i = index_of.get(p1)
            branch_rows.append(
                {
                    "skeleton_id": int(skeleton_id),
                    "branch_id": int(branch_id),
                    "start_node": int(start_i) if start_i is not None else -1,
                    "end_node": int(end_i) if end_i is not None else -1,
                    "start_degree": int(sub_degrees[start_i]) if start_i is not None else 0,
                    "end_degree": int(sub_degrees[end_i]) if end_i is not None else 0,
                    "source_z_um": float(src[0]),
                    "source_y_um": float(src[1]),
                    "source_x_um": float(src[2]),
                    "target_z_um": float(dst[0]),
                    "target_y_um": float(dst[1]),
                    "target_x_um": float(dst[2]),
                    "length_um": length,
                    "branch_length_um": length,
                    "euclidean_length_um": euclidean,
                    "tortuosity": float(length / euclidean) if euclidean > 0 else float("nan"),
                    "num_points": int(len(path)),
                    "mean_radius_um": float(np.mean(radius_vals)) if radius_vals else np.nan,
                    "is_stitch": False,
                }
            )
    branch_table = pd.DataFrame(branch_rows)
    return vertex_table, edge_table, branch_table


def _write_progress(output_root: Path, payload: dict) -> None:
    path = output_root / "_progress.json"
    tmp = path.with_name(path.name + ".tmp")
    try:
        tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        replaced = False
        for _ in range(8):
            try:
                os.replace(tmp, path)
                replaced = True
                break
            except PermissionError:
                time.sleep(0.05)
        if not replaced:
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass
            logger.debug("Could not replace progress file %s", path)
    except OSError:
        logger.debug("Could not write progress file %s", path, exc_info=True)
    progress_file = os.environ.get("YIFU_PROGRESS_FILE")
    if not progress_file:
        return
    try:
        from pipeline_modules.harness.progress import ProgressWriter, read_progress

        writer = ProgressWriter(progress_file)
        writer.state = read_progress(Path(progress_file))
        writer.set_units(
            int(payload.get("unit_done") or 0),
            int(payload.get("unit_total") or 0),
            phase=str(payload.get("phase") or ""),
            phase_started_at=payload.get("phase_started_at"),
        )
    except Exception:
        logger.debug("Could not update harness progress file", exc_info=True)


_EDT_WORKER_MASK = None
_EDT_BACKEND_LOGGED = False


def _init_edt_worker(mask_path: str, dataset_name: str) -> None:
    global _EDT_WORKER_MASK
    _EDT_WORKER_MASK = open_zarr_dataset(mask_path, dataset_name=dataset_name)


def _fast_edt_um(binary: np.ndarray, sampling_zyx) -> np.ndarray:
    """Distance to background in microns. Prefers the C ``edt`` package over scipy."""
    global _EDT_BACKEND_LOGGED
    sampling = tuple(float(v) for v in sampling_zyx)
    binary = np.ascontiguousarray(binary)
    try:
        import edt as edt_mod

        out = edt_mod.edt(binary, anisotropy=sampling, black_border=False).astype(np.float32, copy=False)
        backend = "edt"
    except Exception:
        out = ndimage.distance_transform_edt(binary, sampling=sampling).astype(np.float32, copy=False)
        backend = "scipy"
    if not _EDT_BACKEND_LOGGED:
        logger.info("EDT backend=%s", backend)
        _EDT_BACKEND_LOGGED = True
    return out


def _crop_binary_with_background(binary: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    coords = np.argwhere(binary)
    lo = np.maximum(coords.min(axis=0) - 1, 0)
    hi = np.minimum(coords.max(axis=0) + 2, binary.shape)
    cropped = binary[lo[0] : hi[0], lo[1] : hi[1], lo[2] : hi[2]]
    # Pad a one-voxel background shell so solid chunks still have a finite EDT.
    cropped = np.pad(cropped, 1, mode="constant", constant_values=False)
    return cropped, lo.astype(np.int64) - 1


def _window_max_radii(edt: np.ndarray, origin: np.ndarray, local_starts: np.ndarray, global_indices, factor: int):
    out: list[tuple[int, float]] = []
    block_shape = edt.shape
    factor = max(1, int(factor))
    for local_i, i in enumerate(global_indices):
        start = local_starts[local_i] - origin
        stop = start + factor
        z0, y0, x0 = (int(max(0, start[0])), int(max(0, start[1])), int(max(0, start[2])))
        z1, y1, x1 = (
            int(min(block_shape[0], stop[0])),
            int(min(block_shape[1], stop[1])),
            int(min(block_shape[2], stop[2])),
        )
        if z1 <= z0 or y1 <= y0 or x1 <= x0:
            continue
        window = edt[z0:z1, y0:y1, x0:x1]
        if window.size:
            out.append((int(i), float(np.max(window))))
    return out


def _edt_radii_from_binary(binary, origin, local_starts, global_indices, factor, scale_zyx):
    if not np.any(binary):
        return []
    cropped, lo = _crop_binary_with_background(binary)
    edt = _fast_edt_um(cropped, scale_zyx)
    origin = np.asarray(origin, dtype=np.int64) + lo
    return _window_max_radii(edt, origin, local_starts, global_indices, factor)


def _edt_chunk_worker(payload: dict) -> list[tuple[int, float]]:
    """Process-pool worker: native EDT radii for one mask chunk."""
    mask_zarr = _EDT_WORKER_MASK
    if mask_zarr is None:
        mask_zarr = open_zarr_dataset(payload["mask_path"], dataset_name=payload["dataset_name"])
    shape = tuple(int(v) for v in mask_zarr.shape)
    chunks = tuple(int(v) for v in payload["chunks"])
    halo_zyx = tuple(int(v) for v in payload["halo_zyx"])
    factor = int(payload["factor"])
    foreground_label = payload["foreground_label"]
    scale_zyx = np.array(payload["scale_zyx"], dtype=np.float64)
    local_starts = np.asarray(payload["native_start"], dtype=np.int64)
    global_indices = [int(v) for v in payload["global_indices"]]
    chunk_index = tuple(int(v) for v in payload["chunk_index"])
    core_slices = chunk_index_to_slices(chunk_index, chunks, shape)
    expanded = expand_slices(core_slices, halo_zyx, shape)
    block = np.asarray(mask_zarr[expanded])
    binary = _binary_from_volume(block, foreground_label)
    origin = np.array([s.start for s in expanded], dtype=np.int64)
    return _edt_radii_from_binary(binary, origin, local_starts, global_indices, factor, scale_zyx)


def reconstruct_vessel_express(
    mask_zarr_path,
    output_dir,
    *,
    dataset_name: str = "0",
    resolution_xyz=(1.8, 1.8, 2.0),
    foreground_label: int | None = 1,
    downsample_factor: int = 4,
    downsample_method: str = "max_pool",
    occupancy_threshold: float | None = None,
    dust_threshold: int = 100,
    keep_downsampled_mask: bool = True,
    max_in_memory_voxels: int = DEFAULT_MAX_IN_MEMORY_VOXELS,
    radius_halo_zyx=(32, 32, 32),
    prune_spurs_max_length_um: float = 20.0,
    merge_branch_points_distance_um: float = 15.0,
    through_angle_deg: float = 150.0,
    kink_align_deg: float = 45.0,
    workers: int = 8,
) -> dict:
    """Occupancy-downsample, Lee-skeletonize, then native-EDT radii."""
    started = time.time()
    resolution_xyz = parse_resolution_xyz(resolution_xyz)
    downsample_factor = max(1, int(downsample_factor))
    workers = max(1, int(workers))
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    phase_started = _now_iso()

    def report(phase: str, done: int = 0, total: int = 0, **extra):
        payload = {
            "phase": phase,
            "updated_at": _now_iso(),
            "unit_done": int(done),
            "unit_total": int(total),
            "phase_started_at": phase_started,
            **extra,
        }
        _write_progress(output_root, payload)

    src = open_zarr_dataset(mask_zarr_path, dataset_name=dataset_name)
    mem = estimate_downsampled_memory(src.shape, downsample_factor)
    logger.info(
        "Native shape=%s voxels=%.3e | ds%dx shape=%s uint8=%.2f GiB Lee-peak~%.2f GiB",
        tuple(src.shape),
        mem["native_voxels"],
        downsample_factor,
        tuple(mem["downsampled_shape_zyx"]),
        mem["uint8_gib"],
        mem["lee_peak_gib_est"],
    )
    report("Downsampling", 0, 1, memory=mem)

    working_path = Path(mask_zarr_path)
    working_resolution = resolution_xyz
    if downsample_factor > 1:
        working_path = output_root / f"mask_ds{downsample_factor}_{downsample_method}.zarr"
        expected = tuple(int(v) for v in mem["downsampled_shape_zyx"])
        reused = False
        if working_path.exists():
            try:
                existing = open_zarr_dataset(working_path, dataset_name="0")
                reused = tuple(int(v) for v in existing.shape) == expected
            except Exception:
                reused = False
        if reused:
            logger.info("Reusing downsampled mask %s shape=%s", working_path, expected)
            report("Downsampling", 1, 1, memory=mem, reused=True)
        else:
            downsample_binary_mask_zarr(
                mask_zarr_path,
                working_path,
                factor=downsample_factor,
                dataset_name=dataset_name,
                foreground_label=foreground_label,
                method=downsample_method,
                occupancy_threshold=occupancy_threshold,
                workers=workers,
                progress_callback=lambda done, total: report("Downsampling", done, total, memory=mem),
            )
        working_resolution = tuple(float(v) * downsample_factor for v in resolution_xyz)

    ds = open_zarr_dataset(working_path, dataset_name="0")
    ds_voxels = int(np.prod(np.asarray(ds.shape, dtype=np.int64)))
    if ds_voxels > int(max_in_memory_voxels):
        raise MemoryError(
            f"Downsampled mask has {ds_voxels} voxels "
            f"(limit {int(max_in_memory_voxels)}). Increase --downsample_factor "
            "or --max_in_memory_voxels. VesselExpress Lee thinning is whole-volume."
        )

    phase_started = _now_iso()
    report("Lee skeletonize", 0, 1, memory=mem)
    ds_label = foreground_label if downsample_factor == 1 else 1
    binary = _binary_from_volume(np.asarray(ds[:]), ds_label)
    binary = _remove_small_components(binary, dust_threshold)
    skeleton = skeletonize_lee(binary)
    mask_voxels_downsampled = int(np.count_nonzero(binary))
    logger.info("Lee skeleton voxels=%d / mask voxels=%d", int(np.count_nonzero(skeleton)), mask_voxels_downsampled)

    phase_started = _now_iso()
    report("Topology cleanup", 0, 1)
    working_scale_zyx = np.array(resolution_xyz_to_zyx(working_resolution), dtype=np.float64)
    coords, edges, degrees = skeleton_to_graph(skeleton)
    coords, edges, degrees, topo = clean_skeleton_topology(
        coords,
        edges,
        degrees,
        working_scale_zyx,
        prune_spurs_max_length_um=prune_spurs_max_length_um,
        merge_branch_points_distance_um=merge_branch_points_distance_um,
        through_angle_deg=through_angle_deg,
        kink_align_deg=kink_align_deg,
    )
    if len(coords):
        coords_i = np.rint(coords).astype(np.int64)
        coords_i[:, 0] = np.clip(coords_i[:, 0], 0, skeleton.shape[0] - 1)
        coords_i[:, 1] = np.clip(coords_i[:, 1], 0, skeleton.shape[1] - 1)
        coords_i[:, 2] = np.clip(coords_i[:, 2], 0, skeleton.shape[2] - 1)
    else:
        coords_i = np.empty((0, 3), dtype=np.int64)
    logger.info("Topology cleanup %s; skeleton voxels=%d", topo, int(len(coords_i)))
    report("Topology cleanup", 1, 1, topology_clean=topo)
    del binary, skeleton
    gc.collect()

    native_mask = open_zarr_dataset(mask_zarr_path, dataset_name=dataset_name)
    phase_started = _now_iso()
    report("Native EDT radii", 0, 1)
    edt_workers = max(int(workers), min(16, os.cpu_count() or int(workers)))
    radii = sample_native_edt_radii(
        native_mask,
        coords_i,
        factor=downsample_factor,
        resolution_xyz=resolution_xyz,
        foreground_label=foreground_label,
        halo_zyx=radius_halo_zyx,
        workers=edt_workers,
        mask_zarr_path=mask_zarr_path,
        dataset_name=dataset_name,
        progress_callback=lambda done, total: report("Native EDT radii", done, total),
    )

    scale_zyx = np.array(resolution_xyz_to_zyx(working_resolution), dtype=np.float64)
    origin_zyx = np.zeros(3, dtype=np.float64)
    vertex_table, edge_table, branch_table = _component_tables_from_graph(
        coords_i,
        edges,
        degrees,
        radii,
        scale_zyx=scale_zyx,
        origin_zyx=origin_zyx,
    )

    vertex_csv = output_root / "skeleton_vertices.csv"
    edge_csv = output_root / "skeleton_edges.csv"
    branch_csv = output_root / "vessel_branch_metrics.csv"
    if vertex_table.empty:
        pd.DataFrame(columns=["skeleton_id", "node_id", "z_um", "y_um", "x_um", "radius_um"]).to_csv(vertex_csv, index=False)
        pd.DataFrame().to_csv(edge_csv, index=False)
        pd.DataFrame().to_csv(branch_csv, index=False)
    else:
        vertex_table.to_csv(vertex_csv, index=False)
        edge_table.to_csv(edge_csv, index=False)
        branch_table.to_csv(branch_csv, index=False)

    mean_radius = float(np.nanmean(radii)) if len(radii) else float("nan")
    summary = {
        "mode": "vessel_express_lee",
        "downsample_factor": int(downsample_factor),
        "downsample_method": str(downsample_method),
        "occupancy_threshold": occupancy_threshold,
        "dust_threshold": int(dust_threshold),
        "native_shape_zyx": [int(v) for v in src.shape],
        "downsampled_shape_zyx": [int(v) for v in ds.shape],
        "memory": mem,
        "mask_voxels_downsampled": int(mask_voxels_downsampled),
        "num_skeleton_voxels": int(len(coords_i)),
        "num_skeletons": int(vertex_table["skeleton_id"].nunique()) if not vertex_table.empty else 0,
        "mean_radius_um": mean_radius,
        "mean_diameter_um": mean_radius * 2.0 if np.isfinite(mean_radius) else float("nan"),
        "resolution_xyz_um": list(resolution_xyz),
        "working_resolution_xyz_um": list(working_resolution),
        "elapsed_min": round((time.time() - started) / 60.0, 2),
        "in_memory": True,
        "radius_source": "native_edt",
        "topology_clean": topo,
        "prune_spurs_max_length_um": float(prune_spurs_max_length_um),
        "merge_branch_points_distance_um": float(merge_branch_points_distance_um),
        "workers": int(workers),
    }
    (output_root / "vessel_network_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    report("Done", 1, 1, summary=summary)

    if downsample_factor > 1 and not keep_downsampled_mask:
        shutil.rmtree(working_path, ignore_errors=True)

    logger.info("VesselExpress reconstruction done in %.2f min", summary["elapsed_min"])
    return summary


def run_preview_chunks(
    mask_zarr_path,
    output_dir,
    chunk_indices,
    *,
    image_zarr_path=None,
    dataset_name: str = "0",
    resolution_xyz=(1.8, 1.8, 2.0),
    foreground_label: int | None = 1,
    downsample_factor: int = 4,
    downsample_method: str = "max_pool",
    occupancy_threshold: float | None = None,
    dust_threshold: int = 5,
    prune_spurs_max_length_um: float = 0.0,
    merge_branch_points_distance_um: float = 0.0,
    through_angle_deg: float = 150.0,
    kink_align_deg: float = 45.0,
    fill_holes: bool = False,
    max_hole_diameter_um: float = 25.0,
    despike_thick: bool = False,
    thin_max_voxels: int = 2,
    view: bool = True,
    screenshot_path=None,
) -> dict:
    """Fast ROI check: downsample a few native chunks, Lee-skeletonize, overlay on native mask."""
    downsample_factor = max(1, int(downsample_factor))
    resolution_xyz = parse_resolution_xyz(resolution_xyz)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    mask_zarr = open_zarr_dataset(mask_zarr_path, dataset_name=dataset_name)
    image_zarr = open_zarr_dataset(image_zarr_path, dataset_name=dataset_name) if image_zarr_path else None
    shape = tuple(int(v) for v in mask_zarr.shape)
    chunks = tuple(int(v) for v in mask_zarr.chunks)
    parsed = [parse_chunk_index(item) if isinstance(item, str) else tuple(int(v) for v in item) for item in chunk_indices]
    roi_slices = [chunk_index_to_slices(idx, chunks, shape) for idx in parsed]
    union = tuple(
        slice(min(s[axis].start for s in roi_slices), max(s[axis].stop for s in roi_slices))
        for axis in range(3)
    )
    aligned = tuple(_align_slice_to_factor(union[axis], shape[axis], downsample_factor) for axis in range(3))

    native_mask = np.asarray(mask_zarr[aligned])
    native_image = np.asarray(image_zarr[aligned]) if image_zarr is not None else None
    binary = _binary_from_volume(native_mask, foreground_label)
    n_holes = 0
    voxels_filled = 0
    if fill_holes:
        max_voxels = hole_diameter_to_voxels(max_hole_diameter_um, resolution_xyz)
        binary, n_holes, voxels_filled = fill_small_holes(binary, max_voxels)
        logger.info(
            "Filled %d enclosed holes (%d voxels, max_diameter=%.1f um)",
            n_holes,
            voxels_filled,
            max_hole_diameter_um,
        )
    n_despike = 0
    if despike_thick:
        binary, n_despike = despike_thick_vessels(binary, thin_max_voxels=thin_max_voxels)
        logger.info("Despiked thick vessels: removed %d surface voxels (thin_max=%d)", n_despike, thin_max_voxels)
    reduced = block_reduce_binary(
        binary,
        downsample_factor,
        method=downsample_method,
        occupancy_threshold=occupancy_threshold,
    )
    reduced = _remove_small_components(reduced, dust_threshold)
    skeleton_ds = skeletonize_lee(reduced)
    coords, edges, degrees = skeleton_to_graph(skeleton_ds)
    working_scale_zyx = np.array(resolution_xyz_to_zyx(resolution_xyz), dtype=np.float64) * float(downsample_factor)
    coords, edges, degrees, topo = clean_skeleton_topology(
        coords,
        edges,
        degrees,
        working_scale_zyx,
        prune_spurs_max_length_um=prune_spurs_max_length_um,
        merge_branch_points_distance_um=merge_branch_points_distance_um,
        through_angle_deg=through_angle_deg,
        kink_align_deg=kink_align_deg,
    )
    n_pruned = int(topo["spurs"])
    native_pts = upsample_coarse_coords(coords, native_mask.shape, downsample_factor)
    skeleton_native = rasterize_points(native_mask.shape, native_pts)
    branch_pts = native_pts[degrees >= 3] if len(coords) else np.empty((0, 3), dtype=np.int64)
    end_pts = native_pts[degrees == 1] if len(coords) else np.empty((0, 3), dtype=np.int64)

    polylines = []
    for path in graph_to_polylines(coords, edges, degrees):
        if len(path) < 2:
            continue
        polylines.append(upsample_coarse_coords(path, native_mask.shape, downsample_factor).astype(np.float32))

    np.save(output_root / "mask_crop.npy", binary.astype(np.uint8))
    np.save(output_root / "skeleton_native.npy", skeleton_native)
    if native_image is not None:
        np.save(output_root / "image_crop.npy", native_image)
    pd.DataFrame(native_pts, columns=["z", "y", "x"]).assign(degree=degrees).to_csv(
        output_root / "skeleton_points_native.csv", index=False
    )
    if len(branch_pts):
        pd.DataFrame(branch_pts, columns=["z", "y", "x"]).to_csv(output_root / "branch_points_native.csv", index=False)

    summary = {
        "mode": "vessel_express_preview",
        "chunks": [".".join(str(v) for v in idx) for idx in parsed],
        "roi_start_zyx": [int(s.start) for s in aligned],
        "roi_stop_zyx": [int(s.stop) for s in aligned],
        "downsample_factor": int(downsample_factor),
        "downsample_method": str(downsample_method),
        "dust_threshold": int(dust_threshold),
        "prune_spurs_max_length_um": float(prune_spurs_max_length_um),
        "merge_branch_points_distance_um": float(merge_branch_points_distance_um),
        "through_angle_deg": float(through_angle_deg),
        "kink_align_deg": float(kink_align_deg),
        "num_spurs_pruned": int(n_pruned),
        "topology_clean": topo,
        "fill_holes": bool(fill_holes),
        "max_hole_diameter_um": float(max_hole_diameter_um) if fill_holes else None,
        "n_holes_filled": int(n_holes),
        "hole_voxels_filled": int(voxels_filled),
        "despike_thick": bool(despike_thick),
        "despike_voxels_removed": int(n_despike),
        "native_crop_shape": list(native_mask.shape),
        "downsampled_shape": list(reduced.shape),
        "mask_voxels": int(np.count_nonzero(binary)),
        "num_skeleton_voxels": int(len(coords)),
        "num_branch_points": int(len(branch_pts)),
        "num_end_points": int(len(end_pts)),
        "num_polylines": int(len(polylines)),
        "resolution_xyz_um": list(resolution_xyz),
    }
    (output_root / "preview_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    logger.info(
        "Preview ds%dx %s: skeleton=%d branch_points=%d clean=%s roi=%s",
        downsample_factor,
        downsample_method,
        len(coords),
        len(branch_pts),
        topo,
        summary["native_crop_shape"],
    )

    shot = Path(screenshot_path) if screenshot_path else (output_root / "napari_overlay.png")
    if view or screenshot_path is not None:
        import napari

        viewer = napari.Viewer(ndisplay=3)
        if native_image is not None:
            viewer.add_image(native_image, name="image", blending="additive", colormap="gray")
        viewer.add_labels(binary.astype(np.uint8), name="mask", opacity=0.25)
        viewer.add_labels(skeleton_native, name="skeleton_ds_upsampled", opacity=0.9)
        if polylines:
            viewer.add_shapes(
                polylines,
                name="centerlines",
                shape_type="path",
                edge_color="yellow",
                edge_width=1.2,
                face_color="transparent",
            )
        if len(branch_pts):
            viewer.add_points(branch_pts, name="branch_points", size=6, face_color="magenta", ndim=3)
        if len(end_pts):
            viewer.add_points(end_pts, name="end_points", size=4, face_color="cyan", ndim=3)
        try:
            viewer.screenshot(str(shot), canvas_only=True)
        except Exception:
            logger.exception("Could not write napari screenshot")
        if view:
            napari.run()
        else:
            viewer.close()
    return summary


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="VesselExpress Lee thinning on an occupancy-downsampled mask, with native-EDT radii."
    )
    parser.add_argument("--mask_zarr", help="Native-resolution binary mask Zarr")
    parser.add_argument("--output_dir", help="Output directory")
    parser.add_argument("--dataset_name", default="0")
    parser.add_argument("--resolution_xyz", default="1.8,1.8,2.0")
    parser.add_argument("--foreground_label", type=int, default=1)
    parser.add_argument("--downsample_factor", type=int, default=4)
    parser.add_argument(
        "--downsample_method",
        choices=("occupancy", "max_pool", "majority"),
        default="max_pool",
        help="max_pool keeps 1-2 px capillaries; occupancy reduces border inflation",
    )
    parser.add_argument(
        "--occupancy_threshold",
        type=float,
        default=None,
        help="Fill fraction for occupancy downsample; default is 1/factor^2",
    )
    parser.add_argument("--dust_threshold", type=int, default=100)
    parser.add_argument("--workers", type=int, default=8, help="Parallel workers for downsample and native EDT")
    parser.add_argument("--keep_downsampled_mask", action="store_true", default=True)
    parser.add_argument("--discard_downsampled_mask", action="store_true")
    parser.add_argument("--max_in_memory_voxels", type=int, default=DEFAULT_MAX_IN_MEMORY_VOXELS)
    parser.add_argument("--radius_halo_zyx", default="32,32,32")
    parser.add_argument(
        "--estimate",
        action="store_true",
        help="Print RAM estimate for the (optionally real) mask and exit",
    )
    parser.add_argument(
        "--native_shape_zyx",
        default="",
        help="For --estimate without a Zarr: z,y,x voxel counts",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Fast ROI check on --chunks: 4x skeleton upsampled onto the native mask in napari",
    )
    parser.add_argument(
        "--chunks",
        default="1.20.16,1.20.17,1.20.18",
        help="Comma-separated native Zarr chunk indices z.y.x for --preview",
    )
    parser.add_argument("--image_zarr", default="", help="Optional intensity Zarr for --preview overlay")
    parser.add_argument("--no_view", action="store_true", help="Do not open napari")
    parser.add_argument("--screenshot", default="", help="Optional napari screenshot path")
    parser.add_argument(
        "--prune_spurs_max_length_um",
        type=float,
        default=20.0,
        help="Delete terminal branches shorter than this (um). 0=off",
    )
    parser.add_argument(
        "--merge_branch_points_distance_um",
        type=float,
        default=15.0,
        help="Merge branch points closer than this (um) into one. 0=off",
    )
    parser.add_argument(
        "--through_angle_deg",
        type=float,
        default=150.0,
        help="Min angle of a through-line used to detect voxel kinks",
    )
    parser.add_argument(
        "--kink_align_deg",
        type=float,
        default=45.0,
        help="Drop extra arm if it aligns within this angle of the through-line",
    )
    parser.add_argument(
        "--fill_holes",
        action="store_true",
        help="Fill enclosed cavities in the native mask before downsample/skeletonize",
    )
    parser.add_argument(
        "--max_hole_diameter_um",
        type=float,
        default=25.0,
        help="Only fill enclosed holes smaller than this sphere diameter (um). 0=all enclosed holes",
    )
    parser.add_argument(
        "--despike_thick",
        action="store_true",
        help="Open only thick-vessel surfaces; keep 1-2 voxel filaments",
    )
    parser.add_argument(
        "--thin_max_voxels",
        type=int,
        default=2,
        help="EDT below this (voxels) is treated as thin and never opened",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    if args.estimate:
        if args.native_shape_zyx:
            shape = parse_triplet_int(args.native_shape_zyx)
        elif args.mask_zarr:
            src = open_zarr_dataset(args.mask_zarr, dataset_name=args.dataset_name)
            shape = tuple(int(v) for v in src.shape)
        else:
            raise SystemExit("--estimate needs --mask_zarr or --native_shape_zyx")
        report = estimate_downsampled_memory(shape, args.downsample_factor)
        print(json.dumps(report, indent=2))
        return 0

    if not args.mask_zarr or not args.output_dir:
        raise SystemExit("--mask_zarr and --output_dir are required unless --estimate")

    if args.preview:
        chunk_indices = [part.strip() for part in str(args.chunks).split(",") if part.strip()]
        run_preview_chunks(
            args.mask_zarr,
            args.output_dir,
            chunk_indices,
            image_zarr_path=args.image_zarr or None,
            dataset_name=args.dataset_name,
            resolution_xyz=args.resolution_xyz,
            foreground_label=None if args.foreground_label < 0 else args.foreground_label,
            downsample_factor=args.downsample_factor,
            downsample_method=args.downsample_method,
            occupancy_threshold=args.occupancy_threshold,
            dust_threshold=args.dust_threshold,
            prune_spurs_max_length_um=args.prune_spurs_max_length_um,
            merge_branch_points_distance_um=args.merge_branch_points_distance_um,
            through_angle_deg=args.through_angle_deg,
            kink_align_deg=args.kink_align_deg,
            fill_holes=args.fill_holes,
            max_hole_diameter_um=args.max_hole_diameter_um,
            despike_thick=args.despike_thick,
            thin_max_voxels=args.thin_max_voxels,
            view=not args.no_view,
            screenshot_path=args.screenshot or None,
        )
        return 0

    reconstruct_vessel_express(
        args.mask_zarr,
        args.output_dir,
        dataset_name=args.dataset_name,
        resolution_xyz=args.resolution_xyz,
        foreground_label=None if args.foreground_label < 0 else args.foreground_label,
        downsample_factor=args.downsample_factor,
        downsample_method=args.downsample_method,
        occupancy_threshold=args.occupancy_threshold,
        dust_threshold=args.dust_threshold,
        keep_downsampled_mask=not args.discard_downsampled_mask,
        max_in_memory_voxels=args.max_in_memory_voxels,
        radius_halo_zyx=parse_triplet_int(args.radius_halo_zyx),
        prune_spurs_max_length_um=args.prune_spurs_max_length_um,
        merge_branch_points_distance_um=args.merge_branch_points_distance_um,
        through_angle_deg=args.through_angle_deg,
        kink_align_deg=args.kink_align_deg,
        workers=args.workers,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
