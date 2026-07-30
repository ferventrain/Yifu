"""Per-brain-region vessel parameter analysis.

Given skeleton CSV outputs from ``kimimaro_reconstruction`` (vertex/edge tables
in micron coordinates) plus a registered Allen-style annotation label Zarr and
a region CSV, compute vessel parameters for a user-supplied list of brain
regions. Regions can be specified by acronym, full name, or integer id, and
each query includes all descendants in the region tree.
"""

from __future__ import annotations

import argparse
import ast
import json
import logging
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from .kimimaro_reconstruction import (
    _branch_table_from_tables,
    iter_all_chunk_indices,
    open_zarr_dataset,
    parse_resolution_xyz,
    resolution_xyz_to_zyx,
)

try:
    from pipeline_modules.utils.errors import ErrorCode, PipelineError
    from pipeline_modules.utils.run_manifest import write_run_manifest
except ImportError:
    PipelineError = None  # type: ignore[assignment,misc]
    ErrorCode = None  # type: ignore[assignment]
    write_run_manifest = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


def _parse_acronym_text(acronym_text):
    try:
        acronym_values = ast.literal_eval(str(acronym_text))
        if isinstance(acronym_values, list) and acronym_values:
            return str(acronym_values[-1])
    except Exception:
        pass
    return str(acronym_text)


def _parse_structure_id_path(path_text):
    text = str(path_text).strip()
    if text.startswith("/") and text.endswith("/"):
        return [int(v) for v in text.strip("/").split("/") if v]
    values = ast.literal_eval(text)
    return [int(v) for v in values]


def load_region_tree_with_lookups(region_csv_path):
    """Load Allen-style region CSV and build id / acronym / name lookups.

    Returns ``(nodes_by_id, acronym_to_ids, name_to_ids)``. Lookup keys are
    lowercased for case-insensitive matching. Allen CSV ``name`` fields of
    the form ``"Frontal pole, cerebral cortex,FRP"`` are split so both the
    full display name and the raw field can be queried.
    """
    region_df = pd.read_csv(region_csv_path)
    region_df = region_df.reset_index(drop=True)

    nodes_by_id = {}
    acronym_to_ids = {}
    name_to_ids = {}
    for _, row in region_df.iterrows():
        structure_id = int(row["id"])
        structure_path = _parse_structure_id_path(row["structure_id_path"])
        parent_structure_id = structure_path[-2] if len(structure_path) >= 2 else None
        acronym = _parse_acronym_text(row["acronym"])
        raw_name = str(row["name"]) if pd.notna(row["name"]) else str(structure_id)
        display_name = raw_name.rsplit(",", 1)[0] if "," in raw_name else raw_name

        node = {
            "id": structure_id,
            "name": display_name,
            "raw_name": raw_name,
            "acronym": acronym,
            "parent_structure_id": parent_structure_id,
            "children": [],
        }
        nodes_by_id[structure_id] = node
        acronym_to_ids.setdefault(acronym.lower(), []).append(structure_id)
        name_to_ids.setdefault(display_name.lower(), []).append(structure_id)
        if display_name != raw_name:
            name_to_ids.setdefault(raw_name.lower(), []).append(structure_id)

    for node in nodes_by_id.values():
        parent_structure_id = node["parent_structure_id"]
        if parent_structure_id in nodes_by_id:
            nodes_by_id[parent_structure_id]["children"].append(node)

    return nodes_by_id, acronym_to_ids, name_to_ids


def _collect_subtree_ids(node):
    ids = [int(node["id"])]
    for child in node["children"]:
        ids.extend(_collect_subtree_ids(child))
    return ids


def resolve_region_query(query, nodes_by_id, acronym_to_ids, name_to_ids):
    """Resolve a single query (integer id, acronym, or full name) to a node."""
    query_str = str(query).strip()
    if not query_str:
        raise ValueError("Empty region query")

    if re.fullmatch(r"-?\d+", query_str):
        region_id = int(query_str)
        if region_id in nodes_by_id:
            return nodes_by_id[region_id]
        raise KeyError(f"Region id not found: {region_id}")

    key = query_str.lower()
    matched_ids = acronym_to_ids.get(key, [])
    source = "acronym"
    if not matched_ids:
        matched_ids = name_to_ids.get(key, [])
        source = "name"
    if not matched_ids:
        raise KeyError(f"Region not found by acronym or name: {query_str}")
    if len(matched_ids) > 1:
        raise ValueError(
            f"Region query '{query_str}' is ambiguous via {source}: ids={matched_ids}. "
            f"Please pass a unique id instead."
        )
    return nodes_by_id[matched_ids[0]]


def parse_region_groups(region_groups):
    """Parse region groups from JSON, a JSON file, or CLI-friendly shorthand."""
    if region_groups is None or not isinstance(region_groups, str):
        return region_groups

    candidate = region_groups.strip()
    if not candidate:
        return {}

    if Path(candidate).is_file():
        with open(candidate, "r", encoding="utf-8") as fh:
            return json.load(fh)

    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass

    if not (candidate.startswith("{") and candidate.endswith("}")):
        raise ValueError(
            "region_groups must be JSON, a JSON file path, or shorthand like "
            "{PFC:[FRP,ACA,PL,ILA,ORB,DP]}"
        )

    groups = {}
    body = candidate[1:-1].strip()
    if not body:
        return groups

    for match in re.finditer(r"([^:,\{\}\[\]]+)\s*:\s*\[([^\]]*)\]", body):
        name = match.group(1).strip().strip("'\"")
        values = [
            item.strip().strip("'\"")
            for item in match.group(2).split(",")
            if item.strip()
        ]
        if name:
            groups[name] = values

    if not groups:
        raise ValueError(
            "Could not parse region_groups. Use JSON like "
            "'{\"PFC\":[\"FRP\",\"ACA\",\"PL\",\"ILA\",\"ORB\",\"DP\"]}' "
            "or shorthand {PFC:[FRP,ACA,PL,ILA,ORB,DP]}."
        )
    return groups


def parse_region_list(text):
    """Split a region list string on comma / semicolon / newline."""
    if isinstance(text, (list, tuple)):
        items = list(text)
    else:
        items = re.split(r"[,;\n]", str(text))
    return [item.strip() for item in items if str(item).strip()]


def _load_input_resolution_xyz_from_config(config_path):
    config_path = Path(config_path)
    with open(config_path, "r", encoding="utf-8") as handle:
        cfg = json.load(handle)
    try:
        resolution = cfg["input"]["resolution_xyz"]
    except KeyError as exc:
        raise ValueError(
            f"Config file is missing input.resolution_xyz: {config_path}"
        ) from exc
    return tuple(float(v) for v in resolution)


def _default_config_path():
    candidate = Path(__file__).resolve().parents[2] / "config" / "config.json"
    return candidate if candidate.exists() else None


def _default_annotation_zarr_path(vertex_csv_path):
    vertex_csv_path = Path(vertex_csv_path)
    output_dir = vertex_csv_path.parent
    sample_dir = output_dir.parent if output_dir.name == "tubule_reconstruction" else output_dir
    return sample_dir / "upsampled_atlas_label.zarr"


def _default_mask_zarr_path(vertex_csv_path):
    vertex_csv_path = Path(vertex_csv_path)
    output_dir = vertex_csv_path.parent
    sample_dir = output_dir.parent if output_dir.name == "tubule_reconstruction" else output_dir
    candidates = sorted(sample_dir.glob("*_mask.zarr"))
    if len(candidates) == 1:
        return candidates[0]
    preferred = sample_dir / "ch1_mask.zarr"
    return preferred if preferred.exists() else sample_dir / "ch0_mask.zarr"


def _default_branch_csv_path(vertex_csv_path):
    return Path(vertex_csv_path).parent / "vessel_branch_metrics.csv"


def sample_annotation_labels_at_points_um(
    points_zyx_um,
    annotation_zarr,
    annotation_resolution_xyz,
):
    """Sample annotation labels at ``(z_um, y_um, x_um)`` points.

    Points outside the annotation volume receive label 0 (background).
    """
    points_zyx_um = np.asarray(points_zyx_um, dtype=np.float64)
    if points_zyx_um.size == 0:
        return np.empty(0, dtype=np.int64)

    resolution_zyx = np.asarray(
        resolution_xyz_to_zyx(annotation_resolution_xyz), dtype=np.float64
    )
    voxel_indices = np.floor(points_zyx_um / resolution_zyx).astype(np.int64)

    shape = np.asarray(annotation_zarr.shape, dtype=np.int64)
    in_bounds = np.all((voxel_indices >= 0) & (voxel_indices < shape[None, :]), axis=1)
    labels = np.zeros(len(points_zyx_um), dtype=np.int64)
    if not np.any(in_bounds):
        return labels

    clipped = voxel_indices[in_bounds]
    chunk_grid = np.asarray(annotation_zarr.chunks, dtype=np.int64)
    shape_arr = np.asarray(shape, dtype=np.int64)
    chunk_ijk = clipped // chunk_grid
    local_ijk = clipped - chunk_ijk * chunk_grid

    df = pd.DataFrame({
        "cz": chunk_ijk[:, 0], "cy": chunk_ijk[:, 1], "cx": chunk_ijk[:, 2],
        "lz": local_ijk[:, 0], "ly": local_ijk[:, 1], "lx": local_ijk[:, 2],
    })

    sampled = np.empty(len(clipped), dtype=np.int64)
    for (cz, cy, cx), grp in df.groupby(["cz", "cy", "cx"], sort=False):
        z0 = int(cz * chunk_grid[0])
        y0 = int(cy * chunk_grid[1])
        x0 = int(cx * chunk_grid[2])
        chunk_data = np.asarray(
            annotation_zarr[
                z0 : min(z0 + int(chunk_grid[0]), int(shape_arr[0])),
                y0 : min(y0 + int(chunk_grid[1]), int(shape_arr[1])),
                x0 : min(x0 + int(chunk_grid[2]), int(shape_arr[2])),
            ]
        )
        local = grp[["lz", "ly", "lx"]].to_numpy(dtype=np.int64)
        sampled[grp.index.to_numpy()] = chunk_data[local[:, 0], local[:, 1], local[:, 2]]

    labels[in_bounds] = sampled
    return labels


def _compute_degrees(edge_table):
    if edge_table.empty:
        return pd.Series(dtype=np.int64)
    all_nodes = pd.concat([
        edge_table[["skeleton_id", "source_node"]].rename(columns={"source_node": "node_id"}),
        edge_table[["skeleton_id", "target_node"]].rename(columns={"target_node": "node_id"}),
    ])
    return all_nodes.groupby(["skeleton_id", "node_id"], sort=False).size()


def _finite_values(table, column):
    if table.empty or column not in table.columns:
        return np.empty(0, dtype=np.float64)
    values = table[column].to_numpy(dtype=np.float64)
    return values[np.isfinite(values)]


def _branch_to_branch_mask(branch_table):
    if branch_table.empty:
        return np.zeros(0, dtype=bool)
    if "is_branch_to_branch" in branch_table.columns:
        return branch_table["is_branch_to_branch"].fillna(False).astype(bool).to_numpy()
    if {"start_degree", "end_degree"}.issubset(branch_table.columns):
        return (
            (branch_table["start_degree"].to_numpy(dtype=np.float64) >= 3)
            & (branch_table["end_degree"].to_numpy(dtype=np.float64) >= 3)
        )
    return np.ones(len(branch_table), dtype=bool)


def _attach_branch_midpoints(branch_table, vertex_table):
    if branch_table.empty:
        table = branch_table.copy()
        for col in ("mid_z_um", "mid_y_um", "mid_x_um"):
            table[col] = pd.Series(dtype=np.float64)
        return table
    required = {"skeleton_id", "start_node", "end_node"}
    if not required.issubset(branch_table.columns):
        raise ValueError(f"Branch CSV is missing required columns: {sorted(required - set(branch_table.columns))}")

    coords = vertex_table.drop_duplicates(
        subset=["skeleton_id", "node_id"], keep="last"
    ).set_index(["skeleton_id", "node_id"])[["z_um", "y_um", "x_um"]]

    table = branch_table.copy()
    start_index = pd.MultiIndex.from_frame(
        table[["skeleton_id", "start_node"]].rename(columns={"start_node": "node_id"})
    )
    end_index = pd.MultiIndex.from_frame(
        table[["skeleton_id", "end_node"]].rename(columns={"end_node": "node_id"})
    )
    start_coords = coords.reindex(start_index).to_numpy(dtype=np.float64)
    end_coords = coords.reindex(end_index).to_numpy(dtype=np.float64)
    midpoints = (start_coords + end_coords) / 2.0
    table["mid_z_um"] = midpoints[:, 0]
    table["mid_y_um"] = midpoints[:, 1]
    table["mid_x_um"] = midpoints[:, 2]
    return table


def _accumulate_vessel_label_histogram(
    mask_zarr,
    annotation_zarr,
    foreground_label=1,
):
    """Single-pass histogram of annotation labels on vessel voxels."""
    if tuple(mask_zarr.shape) != tuple(annotation_zarr.shape):
        raise ValueError(
            f"Mask and annotation Zarr shapes must match for direct region volume statistics: "
            f"mask={mask_zarr.shape}, annotation={annotation_zarr.shape}"
        )

    counts: dict[int, int] = {}
    chunks = tuple(int(v) for v in getattr(mask_zarr, "chunks", mask_zarr.shape))
    for chunk_index in tqdm(
        list(iter_all_chunk_indices(mask_zarr.shape, chunks)),
        desc="Region mask volume",
    ):
        slices = tuple(
            slice(
                int(chunk_index[axis]) * chunks[axis],
                min((int(chunk_index[axis]) + 1) * chunks[axis], int(mask_zarr.shape[axis])),
            )
            for axis in range(3)
        )
        mask_chunk = np.asarray(mask_zarr[slices])
        if foreground_label is None:
            vessel_mask = mask_chunk > 0
        else:
            vessel_mask = mask_chunk == foreground_label
        if not np.any(vessel_mask):
            continue
        label_chunk = np.asarray(annotation_zarr[slices])
        vessel_labels = label_chunk[vessel_mask]
        if vessel_labels.size == 0:
            continue
        unique, unique_counts = np.unique(vessel_labels, return_counts=True)
        for label_id, count in zip(unique.tolist(), unique_counts.tolist()):
            lid = int(label_id)
            if lid == 0:
                continue
            counts[lid] = counts.get(lid, 0) + int(count)
    return counts


def _compute_region_mask_volumes(
    mask_zarr,
    annotation_zarr,
    resolved,
    resolution_xyz,
    foreground_label=1,
):
    if mask_zarr is None:
        return {entry["query"]: {"mask_voxels": None, "vessel_volume_um3": np.nan} for entry in resolved}

    label_counts = _accumulate_vessel_label_histogram(
        mask_zarr=mask_zarr,
        annotation_zarr=annotation_zarr,
        foreground_label=foreground_label,
    )
    voxel_volume = float(np.prod(tuple(float(v) for v in resolution_xyz)))
    out = {}
    for entry in resolved:
        total = 0
        for lid in entry["subtree_ids"]:
            total += int(label_counts.get(int(lid), 0))
        out[entry["query"]] = {
            "mask_voxels": int(total),
            "vessel_volume_um3": float(total * voxel_volume),
        }
    return out


def _preaggregate_topology_by_label(vertex_table, edge_table, vertex_labels, branch_table, branch_labels):
    """Aggregate branch-point / branch metrics per annotation leaf label."""
    from collections import defaultdict

    local_degrees = _compute_degrees(edge_table)
    branch_points_by_label = defaultdict(int)
    if not vertex_table.empty and vertex_labels.size and not local_degrees.empty:
        deg_values = local_degrees.reindex(
            pd.MultiIndex.from_frame(vertex_table[["skeleton_id", "node_id"]]),
            fill_value=0,
        ).to_numpy(dtype=np.int64)
        for label_id, is_bp in zip(vertex_labels.tolist(), (deg_values >= 3).tolist()):
            if is_bp and int(label_id) != 0:
                branch_points_by_label[int(label_id)] += 1

    path_sum = defaultdict(float)
    path_count = defaultdict(int)
    path_sq_sum = defaultdict(float)
    tort_sum = defaultdict(float)
    tort_count = defaultdict(int)
    depth_sum = defaultdict(float)
    depth_count = defaultdict(int)

    if not branch_table.empty and branch_labels.size:
        b2b = _branch_to_branch_mask(branch_table)
        lengths = (
            branch_table["branch_length_um"].to_numpy(dtype=np.float64)
            if "branch_length_um" in branch_table.columns
            else np.full(len(branch_table), np.nan)
        )
        tort = (
            branch_table["tortuosity"].to_numpy(dtype=np.float64)
            if "tortuosity" in branch_table.columns
            else np.full(len(branch_table), np.nan)
        )
        depths = (
            branch_table["branch_depth"].to_numpy(dtype=np.float64)
            if "branch_depth" in branch_table.columns
            else np.full(len(branch_table), np.nan)
        )
        for idx, label_id in enumerate(branch_labels.tolist()):
            lid = int(label_id)
            if lid == 0:
                continue
            if b2b[idx] and np.isfinite(lengths[idx]):
                path_sum[lid] += float(lengths[idx])
                path_sq_sum[lid] += float(lengths[idx]) ** 2
                path_count[lid] += 1
            if np.isfinite(tort[idx]):
                tort_sum[lid] += float(tort[idx])
                tort_count[lid] += 1
            if np.isfinite(depths[idx]):
                depth_sum[lid] += float(depths[idx])
                depth_count[lid] += 1

    return {
        "branch_points_by_label": dict(branch_points_by_label),
        "path_sum": dict(path_sum),
        "path_count": dict(path_count),
        "path_sq_sum": dict(path_sq_sum),
        "tort_sum": dict(tort_sum),
        "tort_count": dict(tort_count),
        "depth_sum": dict(depth_sum),
        "depth_count": dict(depth_count),
    }


def _regional_summary_from_preagg(region_node, subtree_ids, preagg, volume_stats):
    subtree_list = [int(v) for v in subtree_ids]
    num_branch_points = sum(
        int(preagg["branch_points_by_label"].get(lid, 0)) for lid in subtree_list
    )
    path_sum = sum(float(preagg["path_sum"].get(lid, 0.0)) for lid in subtree_list)
    path_count = sum(int(preagg["path_count"].get(lid, 0)) for lid in subtree_list)
    path_sq_sum = sum(float(preagg["path_sq_sum"].get(lid, 0.0)) for lid in subtree_list)
    tort_sum = sum(float(preagg["tort_sum"].get(lid, 0.0)) for lid in subtree_list)
    tort_count = sum(int(preagg["tort_count"].get(lid, 0)) for lid in subtree_list)
    depth_sum = sum(float(preagg["depth_sum"].get(lid, 0.0)) for lid in subtree_list)
    depth_count = sum(int(preagg["depth_count"].get(lid, 0)) for lid in subtree_list)

    if path_count > 0:
        path_mean = path_sum / path_count
        if path_count > 1:
            variance = max(0.0, (path_sq_sum - (path_sum ** 2) / path_count) / (path_count - 1))
            path_sd = float(np.sqrt(variance))
        else:
            path_sd = np.nan
    else:
        path_mean = np.nan
        path_sd = np.nan

    return {
        "region_id": int(region_node["id"]),
        "region_acronym": region_node["acronym"],
        "region_name": region_node["name"],
        "num_subtree_ids": int(len(subtree_list)),
        "num_branch_points": int(num_branch_points),
        "branch_point_path_length_sum_um": float(path_sum) if path_count else 0.0,
        "branch_point_path_length_mean_um": float(path_mean),
        "branch_point_path_length_sd_um": float(path_sd) if path_count > 1 else np.nan,
        "mask_voxels": volume_stats.get("mask_voxels"),
        "vessel_volume_um3": volume_stats.get("vessel_volume_um3", np.nan),
        "mean_tortuosity": float(tort_sum / tort_count) if tort_count else np.nan,
        "mean_branch_depth": float(depth_sum / depth_count) if depth_count else np.nan,
    }


def _regional_summary(
    region_node,
    subtree_ids,
    vertex_table,
    edge_table,
    vertex_labels,
    branch_table,
    branch_labels,
    volume_stats,
):
    subtree_list = [int(v) for v in subtree_ids]

    if vertex_table.empty or vertex_labels.size == 0:
        vertex_mask = np.zeros(len(vertex_table), dtype=bool)
    else:
        vertex_mask = np.isin(vertex_labels, subtree_list)
    vertices_in = vertex_table.loc[vertex_mask] if vertex_mask.any() else vertex_table.iloc[0:0]

    # Degrees are computed from the full graph, then counted for vertices that lie in the region.
    local_degrees = _compute_degrees(edge_table)
    num_branch_points = 0
    if not vertices_in.empty and not local_degrees.empty:
        deg_values = local_degrees.reindex(
            pd.MultiIndex.from_frame(vertices_in[["skeleton_id", "node_id"]]),
            fill_value=0,
        ).to_numpy(dtype=np.int64)
        num_branch_points = int(np.sum(deg_values >= 3))

    if branch_table.empty or branch_labels.size == 0:
        branches_in = branch_table.iloc[0:0]
    else:
        branch_region_mask = np.isin(branch_labels, subtree_list)
        branches_in = branch_table.loc[branch_region_mask] if branch_region_mask.any() else branch_table.iloc[0:0]

    path_lengths = (
        branches_in.loc[_branch_to_branch_mask(branches_in), "branch_length_um"].to_numpy(dtype=np.float64)
        if "branch_length_um" in branches_in.columns
        else np.empty(0, dtype=np.float64)
    )
    path_lengths = path_lengths[np.isfinite(path_lengths)]
    tortuosities = _finite_values(branches_in, "tortuosity")
    branch_depths = _finite_values(branches_in, "branch_depth")

    return {
        "region_id": int(region_node["id"]),
        "region_acronym": region_node["acronym"],
        "region_name": region_node["name"],
        "num_subtree_ids": int(len(subtree_list)),
        "num_branch_points": int(num_branch_points),
        "branch_point_path_length_sum_um": float(path_lengths.sum()) if path_lengths.size else 0.0,
        "branch_point_path_length_mean_um": float(path_lengths.mean()) if path_lengths.size else np.nan,
        "branch_point_path_length_sd_um": float(path_lengths.std(ddof=1)) if path_lengths.size > 1 else np.nan,
        "mask_voxels": volume_stats.get("mask_voxels"),
        "vessel_volume_um3": volume_stats.get("vessel_volume_um3", np.nan),
        "mean_tortuosity": float(tortuosities.mean()) if tortuosities.size else np.nan,
        "mean_branch_depth": float(branch_depths.mean()) if branch_depths.size else np.nan,
    }


def analyze_regions_from_skeleton(
    vertex_csv_path,
    edge_csv_path,
    branch_csv_path=None,
    mask_zarr_path=None,
    annotation_zarr_path=None,
    region_cfg_csv=None,
    regions=None,
    region_groups=None,
    all_regions=False,
    output_dir=None,
    mask_dataset_name="0",
    foreground_label=1,
    annotation_dataset_name="0",
    annotation_resolution_xyz=None,
    config_path=None,
):
    """Compute per-region vessel parameters from existing skeleton CSV outputs.

    Parameters
    ----------
    all_regions:
        If True, export every region id present in ``region_cfg_csv`` (whole-brain
        morphology table). Overrides an empty ``regions`` list.
    """
    if region_cfg_csv is None:
        raise ValueError("region_cfg_csv is required.")
    region_queries = parse_region_list(regions) if regions else []
    if region_groups is not None:
        region_groups = parse_region_groups(region_groups)
    if not all_regions and not region_queries and not region_groups:
        raise ValueError("regions, region_groups, or all_regions=True is required.")

    logger.info("Loading region tree from %s", region_cfg_csv)
    nodes_by_id, acronym_to_ids, name_to_ids = load_region_tree_with_lookups(region_cfg_csv)
    resolved = []
    if all_regions:
        logger.info("Resolving all %d regions from CSV", len(nodes_by_id))
        for region_id in sorted(nodes_by_id):
            node = nodes_by_id[region_id]
            subtree_ids = _collect_subtree_ids(node)
            resolved.append({
                "query": str(region_id),
                "node": node,
                "subtree_ids": subtree_ids,
            })
    if region_queries:
        logger.info("Resolving %d region query(ies): %s", len(region_queries), region_queries)
        for query in region_queries:
            node = resolve_region_query(query, nodes_by_id, acronym_to_ids, name_to_ids)
            subtree_ids = _collect_subtree_ids(node)
            resolved.append({"query": query, "node": node, "subtree_ids": subtree_ids})
    if region_groups:
        logger.info("Resolving %d region group(s): %s", len(region_groups), list(region_groups.keys()))
        for group_name, sub_queries in region_groups.items():
            merged_ids = set()
            for sq in sub_queries:
                node = resolve_region_query(str(sq), nodes_by_id, acronym_to_ids, name_to_ids)
                merged_ids.update(_collect_subtree_ids(node))
            resolved.append({
                "query": group_name,
                "node": {"id": -1, "acronym": group_name, "name": group_name},
                "subtree_ids": sorted(merged_ids),
            })

    logger.info("Loading vertex CSV: %s", vertex_csv_path)
    vertex_table = pd.read_csv(vertex_csv_path)
    logger.info("Loaded %d vertices", len(vertex_table))
    logger.info("Loading edge CSV: %s", edge_csv_path)
    edge_table = pd.read_csv(edge_csv_path)
    logger.info("Loaded %d edges", len(edge_table))

    if branch_csv_path is None:
        default_branch_csv_path = _default_branch_csv_path(vertex_csv_path)
        branch_csv_path = default_branch_csv_path if default_branch_csv_path.exists() else None
    if branch_csv_path is not None and Path(branch_csv_path).exists():
        logger.info("Loading branch CSV: %s", branch_csv_path)
        branch_table = pd.read_csv(branch_csv_path)
    else:
        logger.info("Branch CSV not provided/found; reconstructing branch metrics from vertex/edge tables")
        branch_table = _branch_table_from_tables(vertex_table, edge_table)
    logger.info("Loaded %d branch paths", len(branch_table))

    required_vertex_cols = {"skeleton_id", "node_id", "z_um", "y_um", "x_um"}
    missing_vcols = required_vertex_cols - set(vertex_table.columns)
    if missing_vcols:
        raise ValueError(f"Vertex CSV is missing required columns: {sorted(missing_vcols)}")
    required_edge_cols = {
        "skeleton_id", "source_node", "target_node", "edge_length_um",
        "source_z_um", "source_y_um", "source_x_um",
        "target_z_um", "target_y_um", "target_x_um",
    }
    missing_ecols = required_edge_cols - set(edge_table.columns)
    if missing_ecols:
        raise ValueError(f"Edge CSV is missing required columns: {sorted(missing_ecols)}")

    if annotation_zarr_path is None:
        annotation_zarr_path = _default_annotation_zarr_path(vertex_csv_path)
    annotation_zarr_path = Path(annotation_zarr_path)
    if not annotation_zarr_path.exists():
        raise FileNotFoundError(
            f"Annotation Zarr not found: {annotation_zarr_path}. "
            "Pass --annotation_zarr explicitly or ensure sample_dir/upsampled_atlas_label.zarr exists."
        )

    if annotation_resolution_xyz is None:
        if config_path is None:
            config_path = _default_config_path()
        if config_path is None:
            raise ValueError(
                "annotation_resolution_xyz was not provided and config/config.json was not found. "
                "Pass --config or --annotation_resolution_xyz."
            )
        annotation_resolution_xyz = _load_input_resolution_xyz_from_config(config_path)
    annotation_resolution_xyz = parse_resolution_xyz(annotation_resolution_xyz)

    annotation_zarr = open_zarr_dataset(annotation_zarr_path, dataset_name=annotation_dataset_name)
    if len(annotation_zarr.shape) != 3:
        raise ValueError(f"Annotation Zarr must be 3D, got shape={annotation_zarr.shape}")

    if mask_zarr_path is None:
        mask_zarr_path = _default_mask_zarr_path(vertex_csv_path)
    mask_zarr_path = Path(mask_zarr_path)
    if not mask_zarr_path.exists():
        raise FileNotFoundError(
            f"Mask Zarr not found: {mask_zarr_path}. "
            "Pass --mask_zarr explicitly or ensure sample_dir/<signal_ch>_mask.zarr exists."
        )
    mask_zarr = open_zarr_dataset(mask_zarr_path, dataset_name=mask_dataset_name)
    if len(mask_zarr.shape) != 3:
        raise ValueError(f"Mask Zarr must be 3D, got shape={mask_zarr.shape}")

    _started_at = time.time()
    result = _finalize_region_analysis(
        vertex_table=vertex_table,
        edge_table=edge_table,
        branch_table=branch_table,
        mask_zarr=mask_zarr,
        annotation_zarr=annotation_zarr,
        annotation_resolution_xyz=annotation_resolution_xyz,
        resolved=resolved,
        output_dir=output_dir,
        foreground_label=foreground_label,
    )

    if output_dir is not None and write_run_manifest is not None:
        _output_files = [v for k, v in result.items() if k.endswith("_path")]
        result["manifest_path"] = write_run_manifest(
            Path(output_dir),
            module="tubule_reconstruction.region_vessel_analysis",
            entrypoint="analyze_regions_from_skeleton",
            inputs={
                "vertex_csv_path": str(vertex_csv_path),
                "edge_csv_path": str(edge_csv_path),
                "branch_csv_path": str(branch_csv_path) if branch_csv_path is not None else None,
                "mask_zarr_path": str(mask_zarr_path),
                "annotation_zarr_path": str(annotation_zarr_path),
                "region_cfg_csv": str(region_cfg_csv),
                "regions": regions,
                "region_groups": region_groups,
                "all_regions": all_regions,
                "mask_dataset_name": mask_dataset_name,
                "foreground_label": foreground_label,
                "annotation_dataset_name": annotation_dataset_name,
                "annotation_resolution_xyz": annotation_resolution_xyz,
                "config_path": str(config_path) if config_path is not None else None,
            },
            outputs=_output_files,
            started_at=_started_at,
        )

    return result


_SUMMARY_COLUMN_ORDER = [
    "query",
    "region_id",
    "region_acronym",
    "region_name",
    "num_subtree_ids",
    "num_branch_points",
    "branch_point_path_length_sum_um",
    "branch_point_path_length_mean_um",
    "branch_point_path_length_sd_um",
    "mask_voxels",
    "vessel_volume_um3",
    "mean_tortuosity",
    "mean_branch_depth",
]


def _finalize_region_analysis(
    vertex_table,
    edge_table,
    branch_table,
    mask_zarr,
    annotation_zarr,
    annotation_resolution_xyz,
    resolved,
    output_dir,
    foreground_label=1,
):
    if not vertex_table.empty:
        vertex_points = vertex_table[["z_um", "y_um", "x_um"]].to_numpy(dtype=np.float64)
    else:
        vertex_points = np.empty((0, 3), dtype=np.float64)
    logger.info("Sampling annotation labels for %d vertices ...", len(vertex_points))
    vertex_labels = sample_annotation_labels_at_points_um(
        vertex_points, annotation_zarr, annotation_resolution_xyz
    )

    branch_table = _attach_branch_midpoints(branch_table, vertex_table)
    if not branch_table.empty:
        midpoints = branch_table[["mid_z_um", "mid_y_um", "mid_x_um"]].to_numpy(dtype=np.float64)
        finite_midpoints = np.all(np.isfinite(midpoints), axis=1)
    else:
        finite_midpoints = np.zeros(0, dtype=bool)
        midpoints = np.empty((0, 3), dtype=np.float64)
    logger.info("Sampling annotation labels for %d branch path midpoints ...", int(finite_midpoints.sum()))
    branch_labels = np.zeros(len(branch_table), dtype=np.int64)
    if finite_midpoints.any():
        branch_labels[finite_midpoints] = sample_annotation_labels_at_points_um(
            midpoints[finite_midpoints], annotation_zarr, annotation_resolution_xyz
        )

    volume_stats_by_query = _compute_region_mask_volumes(
        mask_zarr=mask_zarr,
        annotation_zarr=annotation_zarr,
        resolved=resolved,
        resolution_xyz=annotation_resolution_xyz,
        foreground_label=foreground_label,
    )

    # Pre-aggregate topology by leaf label once, then sum over each region's subtree.
    # Critical for all_regions (hundreds–thousands of queries).
    logger.info("Pre-aggregating topology metrics by annotation label ...")
    preagg = _preaggregate_topology_by_label(
        vertex_table=vertex_table,
        edge_table=edge_table,
        vertex_labels=vertex_labels,
        branch_table=branch_table,
        branch_labels=branch_labels,
    )

    rows = []
    logger.info("Computing per-region vessel statistics ...")
    for entry in tqdm(resolved, desc="Regions"):
        summary_row = _regional_summary_from_preagg(
            region_node=entry["node"],
            subtree_ids=entry["subtree_ids"],
            preagg=preagg,
            volume_stats=volume_stats_by_query.get(entry["query"], {}),
        )
        summary_row["query"] = entry["query"]
        rows.append(summary_row)

    summary_table = pd.DataFrame(rows)
    summary_table = summary_table[[c for c in _SUMMARY_COLUMN_ORDER if c in summary_table.columns]]

    result = {
        "summary_table": summary_table,
        "resolved_regions": resolved,
    }

    if output_dir is not None:
        output_root = Path(output_dir)
        output_root.mkdir(parents=True, exist_ok=True)
        summary_csv_path = output_root / "region_vessel_summary.csv"
        summary_json_path = output_root / "region_vessel_summary.json"

        summary_table.to_csv(summary_csv_path, index=False)

        json_rows = []
        for row in summary_table.to_dict(orient="records"):
            sanitized = {}
            for key, value in row.items():
                if isinstance(value, float) and np.isnan(value):
                    sanitized[key] = None
                elif isinstance(value, np.generic):
                    sanitized[key] = value.item()
                else:
                    sanitized[key] = value
            json_rows.append(sanitized)
        with open(summary_json_path, "w", encoding="utf-8") as handle:
            json.dump(json_rows, handle, indent=2, ensure_ascii=False)

        result["summary_csv_path"] = summary_csv_path
        result["summary_json_path"] = summary_json_path

    return result


def build_argparser():
    parser = argparse.ArgumentParser(
        description=(
            "Compute per-brain-region vessel parameters from existing "
            "kimimaro skeleton CSV outputs (vertex/edge tables in um)."
        )
    )
    parser.add_argument("--vertex_csv", required=True, help="Path to skeleton_vertices.csv")
    parser.add_argument("--edge_csv", required=True, help="Path to skeleton_edges.csv")
    parser.add_argument(
        "--branch_csv",
        help="Path to vessel_branch_metrics.csv. Defaults to vertex_csv directory/vessel_branch_metrics.csv.",
    )
    parser.add_argument(
        "--mask_zarr",
        help=(
            "Binary vessel mask Zarr for direct volume statistics. Defaults to "
            "<sample_dir>/<signal_ch>_mask.zarr when vertex_csv is in "
            "<sample_dir>/tubule_reconstruction/."
        ),
    )
    parser.add_argument("--mask_dataset_name", default="0", help="Dataset name inside mask Zarr")
    parser.add_argument(
        "--foreground_label",
        default="1",
        help="Mask voxel value treated as vessel foreground. Use empty string to treat any nonzero value as foreground.",
    )
    parser.add_argument(
        "--annotation_zarr",
        help=(
            "Registered annotation label Zarr. Defaults to "
            "<sample_dir>/upsampled_atlas_label.zarr when vertex_csv is in "
            "<sample_dir>/tubule_reconstruction/."
        ),
    )
    parser.add_argument("--annotation_dataset_name", default="0", help="Dataset name inside annotation Zarr")
    parser.add_argument(
        "--annotation_resolution_xyz",
        help="Annotation voxel size in um as x,y,z. Defaults to input.resolution_xyz from --config.",
    )
    parser.add_argument("--config", help="Path to config.json; used for input.resolution_xyz by default")
    parser.add_argument("--cfg", required=True, help="Allen region CSV path")
    parser.add_argument(
        "--regions",
        help="Comma/semicolon separated region queries (acronym, full name, or integer id)",
    )
    parser.add_argument(
        "--region_groups",
        help=(
            "JSON dict mapping group name to list of region queries, "
            "e.g. '{\"PFC\":[\"FRP\",\"ACA\",\"PL\",\"ILA\",\"ORB\",\"DP\"]}'. "
            "Alternatively, pass a path to a .json file containing the dict. "
            "All sub-regions of listed queries are merged into a single output row."
        ),
    )
    parser.add_argument(
        "--all_regions",
        action="store_true",
        help="Export every region id in --cfg (whole-brain morphology table)",
    )
    parser.add_argument("--output_dir", required=True, help="Directory for summary CSV/JSON outputs")
    parser.add_argument(
        "--json_logs",
        action="store_true",
        help="Emit NDJSON log records to stderr instead of plain text",
    )
    return parser


def main():
    import sys as _sys

    args = build_argparser().parse_args()

    if args.json_logs:
        class _JsonFormatter(logging.Formatter):
            def format(self, record):
                return json.dumps({
                    "level": record.levelname,
                    "logger": record.name,
                    "message": record.getMessage(),
                })

        _handler = logging.StreamHandler(_sys.stderr)
        _handler.setFormatter(_JsonFormatter())
        logging.root.addHandler(_handler)
        logging.root.setLevel(logging.INFO)
    else:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    try:
        result = analyze_regions_from_skeleton(
            vertex_csv_path=args.vertex_csv,
            edge_csv_path=args.edge_csv,
            branch_csv_path=args.branch_csv,
            mask_zarr_path=args.mask_zarr,
            annotation_zarr_path=args.annotation_zarr,
            region_cfg_csv=args.cfg,
            regions=args.regions,
            region_groups=args.region_groups,
            all_regions=args.all_regions,
            output_dir=args.output_dir,
            mask_dataset_name=args.mask_dataset_name,
            foreground_label=None if str(args.foreground_label).strip() == "" else int(args.foreground_label),
            annotation_dataset_name=args.annotation_dataset_name,
            annotation_resolution_xyz=args.annotation_resolution_xyz,
            config_path=args.config,
        )

        summary_table = result["summary_table"]
        logger.info("Per-region vessel summary: %d row(s)", len(summary_table))
        if "summary_csv_path" in result:
            logger.info("Summary CSV:  %s", result["summary_csv_path"])
        if "summary_json_path" in result:
            logger.info("Summary JSON: %s", result["summary_json_path"])
    except Exception as exc:
        if PipelineError is not None and isinstance(exc, PipelineError):
            print(json.dumps({"error_code": exc.code.value, "message": str(exc.message)}), file=_sys.stderr)
            _sys.exit(exc.exit_code)
        logger.exception("Unhandled error: %s", exc)
        _sys.exit(1)


if __name__ == "__main__":
    main()
