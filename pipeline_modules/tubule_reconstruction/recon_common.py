"""Shared helpers for the tubule_reconstruction package.

These functions were previously hosted by ``kimimaro_reconstruction`` and are
relocated here (verbatim) so the remaining reconstruction / analysis programs
(``vessel_express_reconstruction``, ``edt_reconstruction``, ``mask_downsample``,
``region_vessel_analysis``, ``vessel_diameter_analysis``, ``view_skeleton_napari``,
``csv_to_swc``) keep working after the kimimaro engine was removed.

``open_zarr_dataset`` and ``parse_resolution_xyz`` are thin re-exports of the
canonical utilities in ``pipeline_modules.utils`` so the package import surface
stays stable.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from pipeline_modules.utils.parsing import parse_resolution_xyz
from pipeline_modules.utils.zarr_io import open_zarr_array as open_zarr_dataset

__all__ = [
    "parse_resolution_xyz",
    "open_zarr_dataset",
    "resolution_xyz_to_zyx",
    "parse_triplet_int",
    "chunk_index_to_slices",
    "iter_all_chunk_indices",
    "expand_slices",
    "empty_branch_table",
    "branch_table_from_tables",
    "summarize_branch_table",
    "write_swc_files",
]


# ---------------------------------------------------------------------------
# Geometry / chunking
# ---------------------------------------------------------------------------
def resolution_xyz_to_zyx(resolution_xyz):
    resolution_xyz = parse_resolution_xyz(resolution_xyz)
    return (float(resolution_xyz[2]), float(resolution_xyz[1]), float(resolution_xyz[0]))


def parse_triplet_int(text):
    if isinstance(text, (tuple, list)):
        if len(text) != 3:
            raise ValueError(f"Expected 3 integers, got: {text}")
        return tuple(int(v) for v in text)

    parts = [part.strip() for part in str(text).split(",") if part.strip()]
    if len(parts) != 3:
        raise ValueError(f"Expected 3 comma-separated integers, got: {text}")
    return tuple(int(part) for part in parts)


def chunk_index_to_slices(chunk_index, chunks, shape):
    slices = []
    for axis, chunk_idx in enumerate(chunk_index):
        start = int(chunk_idx) * int(chunks[axis])
        stop = min(start + int(chunks[axis]), int(shape[axis]))
        slices.append(slice(start, stop))
    return tuple(slices)


def iter_all_chunk_indices(shape, chunks):
    grid_shape = tuple((int(dim) + int(chunk) - 1) // int(chunk) for dim, chunk in zip(shape, chunks))
    for z in range(grid_shape[0]):
        for y in range(grid_shape[1]):
            for x in range(grid_shape[2]):
                yield (z, y, x)


def expand_slices(core_slices, halo_zyx, shape):
    expanded = []
    for axis, (core_slice, halo, dim) in enumerate(zip(core_slices, halo_zyx, shape)):
        start = max(0, int(core_slice.start) - int(halo))
        stop = min(int(dim), int(core_slice.stop) + int(halo))
        expanded.append(slice(start, stop))
    return tuple(expanded)


# ---------------------------------------------------------------------------
# Branch table reconstruction + summary
# ---------------------------------------------------------------------------
_BRANCH_TABLE_COLUMNS = [
    "skeleton_id",
    "branch_id",
    "start_node",
    "end_node",
    "num_points",
    "num_edges",
    "is_loop",
    "start_degree",
    "end_degree",
    "branch_depth",
    "is_terminal_branch",
    "is_branch_to_branch",
    "is_root_branch",
    "branch_length_um",
    "euclidean_length_um",
    "tortuosity",
    "mean_radius_um",
    "max_radius_um",
    "min_radius_um",
]


def empty_branch_table():
    return pd.DataFrame(columns=_BRANCH_TABLE_COLUMNS)


def branch_table_from_tables(vertex_table, edge_table):
    """Recompute branch metrics from cleaned vertex/edge tables."""
    if vertex_table.empty or edge_table.empty:
        return empty_branch_table()

    all_branches = []
    for skeleton_id, skel_verts in vertex_table.groupby("skeleton_id"):
        skel_edges = edge_table[edge_table["skeleton_id"] == skeleton_id]
        if skel_edges.empty:
            continue

        node_ids = skel_verts["node_id"].tolist()
        node_id_set = set(node_ids)
        node_coords = {}
        node_radii = {}
        for row in skel_verts.itertuples(index=False):
            nid = int(row.node_id)
            node_coords[nid] = np.array([row.z_um, row.y_um, row.x_um], dtype=np.float64)
            node_radii[nid] = float(row.radius_um) if not pd.isna(row.radius_um) else np.nan

        local_adj = {nid: [] for nid in node_id_set}
        edge_list = []
        for row in skel_edges.itertuples(index=False):
            src, tgt = int(row.source_node), int(row.target_node)
            if src in node_id_set and tgt in node_id_set:
                eidx = len(edge_list)
                edge_list.append((src, tgt))
                local_adj[src].append((tgt, eidx))
                local_adj[tgt].append((src, eidx))

        degrees = {nid: len(local_adj[nid]) for nid in node_id_set}
        branch_nodes = {nid for nid, d in degrees.items() if d != 2}
        visited_edges = set()
        branch_id = 0

        for node in sorted(branch_nodes):
            for neighbor, eidx in local_adj[node]:
                if eidx in visited_edges:
                    continue

                path_nodes = [node, neighbor]
                path_edges = [eidx]
                visited_edges.add(eidx)
                prev, current = node, neighbor

                while degrees.get(current, 0) == 2:
                    candidates = [(n, ei) for n, ei in local_adj[current] if n != prev and ei not in visited_edges]
                    if not candidates:
                        break
                    next_n, next_ei = candidates[0]
                    visited_edges.add(next_ei)
                    path_nodes.append(next_n)
                    path_edges.append(next_ei)
                    prev, current = current, next_n

                if len(path_nodes) < 2:
                    continue

                pts = np.array([node_coords[n] for n in path_nodes], dtype=np.float64)
                seg_lengths = np.linalg.norm(pts[1:] - pts[:-1], axis=1)
                branch_length = float(seg_lengths.sum())
                euclidean = float(np.linalg.norm(pts[-1] - pts[0]))
                tortuosity = branch_length / euclidean if euclidean > 1e-12 else np.nan

                path_radii = [node_radii[n] for n in path_nodes if not np.isnan(node_radii.get(n, np.nan))]

                start_node = path_nodes[0]
                end_node = path_nodes[-1]
                start_deg = degrees.get(start_node, 0)
                end_deg = degrees.get(end_node, 0)

                all_branches.append({
                    "skeleton_id": int(skeleton_id),
                    "branch_id": branch_id,
                    "start_node": start_node,
                    "end_node": end_node,
                    "num_points": len(path_nodes),
                    "num_edges": len(path_edges),
                    "is_loop": False,
                    "start_degree": start_deg,
                    "end_degree": end_deg,
                    "branch_depth": np.nan,
                    "is_terminal_branch": bool(start_deg == 1 or end_deg == 1),
                    "is_branch_to_branch": bool(start_deg >= 3 and end_deg >= 3),
                    "is_root_branch": False,
                    "branch_length_um": branch_length,
                    "euclidean_length_um": euclidean,
                    "tortuosity": tortuosity,
                    "mean_radius_um": float(np.mean(path_radii)) if path_radii else np.nan,
                    "max_radius_um": float(np.max(path_radii)) if path_radii else np.nan,
                    "min_radius_um": float(np.min(path_radii)) if path_radii else np.nan,
                })
                branch_id += 1

    if not all_branches:
        return empty_branch_table()
    return pd.DataFrame(all_branches)


def _finite_column_values(table, column):
    if table.empty or column not in table.columns:
        return np.empty(0, dtype=np.float64)
    values = table[column].to_numpy(dtype=np.float64)
    return values[np.isfinite(values)]


def _branch_to_branch_lengths(branch_table):
    if branch_table.empty or "branch_length_um" not in branch_table.columns:
        return np.empty(0, dtype=np.float64)
    if "is_branch_to_branch" in branch_table.columns:
        mask = branch_table["is_branch_to_branch"].fillna(False).astype(bool).to_numpy()
    elif {"start_degree", "end_degree"}.issubset(branch_table.columns):
        mask = (
            (branch_table["start_degree"].to_numpy(dtype=np.float64) >= 3)
            & (branch_table["end_degree"].to_numpy(dtype=np.float64) >= 3)
        )
    else:
        mask = np.ones(len(branch_table), dtype=bool)
    lengths = branch_table.loc[mask, "branch_length_um"].to_numpy(dtype=np.float64)
    return lengths[np.isfinite(lengths)]


def _count_branch_points_from_branch_table(branch_table):
    if branch_table.empty:
        return 0
    points = set()
    if {"skeleton_id", "start_node", "start_degree"}.issubset(branch_table.columns):
        starts = branch_table.loc[
            branch_table["start_degree"].to_numpy(dtype=np.float64) >= 3,
            ["skeleton_id", "start_node"],
        ]
        points.update((int(row.skeleton_id), int(row.start_node)) for row in starts.itertuples(index=False))
    if {"skeleton_id", "end_node", "end_degree"}.issubset(branch_table.columns):
        ends = branch_table.loc[
            branch_table["end_degree"].to_numpy(dtype=np.float64) >= 3,
            ["skeleton_id", "end_node"],
        ]
        points.update((int(row.skeleton_id), int(row.end_node)) for row in ends.itertuples(index=False))
    return int(len(points))


def summarize_branch_table(branch_table):
    if branch_table.empty:
        return {
            "num_branch_points": 0,
            "mask_voxels": 0,
            "vessel_volume_um3": 0.0,
            "branch_point_path_length_sum_um": 0.0,
            "branch_point_path_length_mean_um": np.nan,
            "branch_point_path_length_sd_um": np.nan,
            "mean_tortuosity": np.nan,
            "mean_branch_depth": np.nan,
        }

    path_lengths = _branch_to_branch_lengths(branch_table)
    tortuosities = _finite_column_values(branch_table, "tortuosity")
    valid_tortuosities = tortuosities[np.isfinite(tortuosities)]
    branch_depths = _finite_column_values(branch_table, "branch_depth")
    valid_branch_depths = branch_depths[np.isfinite(branch_depths)]

    return {
        "num_branch_points": _count_branch_points_from_branch_table(branch_table),
        "mask_voxels": None,
        "vessel_volume_um3": None,
        "branch_point_path_length_sum_um": float(path_lengths.sum()) if path_lengths.size > 0 else 0.0,
        "branch_point_path_length_mean_um": float(path_lengths.mean()) if path_lengths.size > 0 else np.nan,
        "branch_point_path_length_sd_um": float(path_lengths.std(ddof=1)) if path_lengths.size > 1 else np.nan,
        "mean_tortuosity": float(valid_tortuosities.mean()) if valid_tortuosities.size > 0 else np.nan,
        "mean_branch_depth": float(valid_branch_depths.mean()) if valid_branch_depths.size > 0 else np.nan,
    }


# ---------------------------------------------------------------------------
# Per-skeleton SWC export
# ---------------------------------------------------------------------------
def _normalize_edge_table_schema(edge_table):
    edge_table = edge_table.copy()
    if "source_skeleton_id" not in edge_table.columns:
        edge_table["source_skeleton_id"] = edge_table["skeleton_id"]
    if "target_skeleton_id" not in edge_table.columns:
        edge_table["target_skeleton_id"] = edge_table["skeleton_id"]
    if "is_stitch" not in edge_table.columns:
        edge_table["is_stitch"] = False
    return edge_table


def _write_single_swc(vertex_table, edge_table, swc_path):
    if vertex_table.empty:
        return

    vertex_table = vertex_table.copy().sort_values(["node_id"]).reset_index(drop=True)
    node_ids = set(vertex_table["node_id"].astype(int).tolist())

    # Build undirected adjacency from edge table
    adj = {nid: [] for nid in node_ids}
    if not edge_table.empty:
        for row in edge_table.itertuples(index=False):
            src, tgt = int(row.source_node), int(row.target_node)
            if src in node_ids and tgt in node_ids:
                adj[src].append(tgt)
                adj[tgt].append(src)

    # BFS from root to assign parent for every reachable node
    parent_map = {}
    visited = set()

    unvisited = set(node_ids)
    while unvisited:
        # Pick root: prefer node with degree 1 (endpoint), else smallest id
        root = None
        for nid in sorted(unvisited):
            if len(adj.get(nid, [])) == 1:
                root = nid
                break
        if root is None:
            root = min(unvisited)

        parent_map[root] = -1
        queue = [root]
        visited.add(root)
        unvisited.discard(root)

        while queue:
            current = queue.pop(0)
            for neighbor in adj[current]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    unvisited.discard(neighbor)
                    parent_map[neighbor] = current
                    queue.append(neighbor)

    lines = ["# id type x y z radius parent"]
    for row in vertex_table.itertuples(index=False):
        node_id = int(row.node_id)
        swc_id = node_id + 1
        parent_node = parent_map.get(node_id, -1)
        parent_id = parent_node + 1 if parent_node >= 0 else -1
        radius = float(row.radius_um) if not pd.isna(row.radius_um) else 1.0

        lines.append(
            f"{swc_id} 0 {float(row.x_um):.6f} {float(row.y_um):.6f} "
            f"{float(row.z_um):.6f} {radius:.6f} {parent_id}"
        )

    swc_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_swc_files(vertex_table, edge_table, output_root):
    output_root = Path(output_root)
    swc_dir = output_root / "swc"
    swc_dir.mkdir(parents=True, exist_ok=True)

    if vertex_table.empty:
        return swc_dir, []

    edge_table = _normalize_edge_table_schema(edge_table)
    written_paths = []

    for skeleton_id, skeleton_vertices in vertex_table.groupby("skeleton_id"):
        if edge_table.empty:
            skeleton_edges = edge_table.iloc[0:0].copy()
        else:
            skeleton_edges = edge_table.loc[
                (edge_table["source_skeleton_id"].astype(int) == int(skeleton_id))
                | (edge_table["target_skeleton_id"].astype(int) == int(skeleton_id))
                | (edge_table["skeleton_id"].astype(int) == int(skeleton_id))
            ].copy()

        swc_path = swc_dir / f"skeleton_{int(skeleton_id):06d}.swc"
        _write_single_swc(skeleton_vertices, skeleton_edges, swc_path)
        written_paths.append(swc_path)

    return swc_dir, written_paths
