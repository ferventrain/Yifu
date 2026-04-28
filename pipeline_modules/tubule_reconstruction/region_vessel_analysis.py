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

from .kimimaro_reconstruction import (
    open_zarr_dataset,
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
    values = ast.literal_eval(str(path_text))
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


def parse_region_list(text):
    """Split a region list string on comma / semicolon / newline."""
    if isinstance(text, (list, tuple)):
        items = list(text)
    else:
        items = re.split(r"[,;\n]", str(text))
    return [item.strip() for item in items if str(item).strip()]


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
    sampled = np.empty(len(clipped), dtype=np.int64)
    # Point-by-point read keeps memory bounded; Zarr caches chunk reads.
    for i, (z, y, x) in enumerate(clipped):
        sampled[i] = int(annotation_zarr[int(z), int(y), int(x)])
    labels[in_bounds] = sampled
    return labels


def _compute_degrees(edge_table):
    degree_map = {}
    if edge_table.empty:
        return degree_map
    for row in edge_table.itertuples(index=False):
        src_key = (int(row.skeleton_id), int(row.source_node))
        tgt_key = (int(row.skeleton_id), int(row.target_node))
        degree_map[src_key] = degree_map.get(src_key, 0) + 1
        degree_map[tgt_key] = degree_map.get(tgt_key, 0) + 1
    return degree_map


def _regional_summary(
    region_node,
    subtree_ids,
    vertex_table,
    edge_table,
    vertex_labels,
    edge_labels,
    radius_lookup,
):
    subtree_list = [int(v) for v in subtree_ids]

    if vertex_table.empty or vertex_labels.size == 0:
        vertex_mask = np.zeros(len(vertex_table), dtype=bool)
    else:
        vertex_mask = np.isin(vertex_labels, subtree_list)
    if edge_table.empty or edge_labels.size == 0:
        edge_mask = np.zeros(len(edge_table), dtype=bool)
    else:
        edge_mask = np.isin(edge_labels, subtree_list)

    vertices_in = vertex_table.loc[vertex_mask] if vertex_mask.any() else vertex_table.iloc[0:0]
    edges_in = edge_table.loc[edge_mask] if edge_mask.any() else edge_table.iloc[0:0]

    edge_lengths = (
        edges_in["edge_length_um"].to_numpy(dtype=np.float64)
        if not edges_in.empty
        else np.empty(0, dtype=np.float64)
    )
    total_length_um = float(edge_lengths.sum()) if edge_lengths.size else 0.0
    mean_edge_length_um = float(edge_lengths.mean()) if edge_lengths.size else 0.0

    if "radius_um" in vertices_in.columns and not vertices_in.empty:
        radius_values = vertices_in["radius_um"].to_numpy(dtype=np.float64)
        valid_radii = radius_values[np.isfinite(radius_values)]
    else:
        valid_radii = np.empty(0, dtype=np.float64)

    # Degrees computed from edges whose midpoint also lies in the region.
    local_degrees = _compute_degrees(edges_in)
    num_branch_points = 0
    num_end_points = 0
    for row in vertices_in.itertuples(index=False):
        deg = local_degrees.get((int(row.skeleton_id), int(row.node_id)), 0)
        if deg >= 3:
            num_branch_points += 1
        elif deg == 1:
            num_end_points += 1

    skeleton_ids_in = set()
    if not vertices_in.empty:
        skeleton_ids_in.update(vertices_in["skeleton_id"].astype(int).tolist())
    if not edges_in.empty:
        skeleton_ids_in.update(edges_in["skeleton_id"].astype(int).tolist())

    # Approximate vessel volume: sum over edges of length * pi * r_mean^2.
    vessel_volume_um3 = np.nan
    if not edges_in.empty and radius_lookup:
        volumes = []
        for row in edges_in.itertuples(index=False):
            r_src = radius_lookup.get((int(row.skeleton_id), int(row.source_node)), np.nan)
            r_tgt = radius_lookup.get((int(row.skeleton_id), int(row.target_node)), np.nan)
            if np.isfinite(r_src) and np.isfinite(r_tgt):
                r_mean = 0.5 * (r_src + r_tgt)
                volumes.append(float(row.edge_length_um) * np.pi * r_mean * r_mean)
        vessel_volume_um3 = float(np.sum(volumes)) if volumes else 0.0

    return {
        "region_id": int(region_node["id"]),
        "region_acronym": region_node["acronym"],
        "region_name": region_node["name"],
        "num_subtree_ids": int(len(subtree_list)),
        "num_vertices": int(len(vertices_in)),
        "num_edges": int(len(edges_in)),
        "num_skeletons": int(len(skeleton_ids_in)),
        "num_branch_points": int(num_branch_points),
        "num_end_points": int(num_end_points),
        "total_length_um": total_length_um,
        "mean_edge_length_um": mean_edge_length_um,
        "mean_radius_um": float(valid_radii.mean()) if valid_radii.size else np.nan,
        "median_radius_um": float(np.median(valid_radii)) if valid_radii.size else np.nan,
        "min_radius_um": float(valid_radii.min()) if valid_radii.size else np.nan,
        "max_radius_um": float(valid_radii.max()) if valid_radii.size else np.nan,
        "approx_vessel_volume_um3": vessel_volume_um3,
    }


def analyze_regions_from_skeleton(
    vertex_csv_path,
    edge_csv_path,
    annotation_zarr_path,
    region_cfg_csv,
    regions,
    output_dir=None,
    annotation_dataset_name="0",
    annotation_resolution_xyz=(25.0, 25.0, 25.0),
):
    """Compute per-region vessel parameters from existing skeleton CSV outputs."""
    region_queries = parse_region_list(regions)
    if not region_queries:
        raise ValueError("No regions provided.")

    nodes_by_id, acronym_to_ids, name_to_ids = load_region_tree_with_lookups(region_cfg_csv)
    resolved = []
    for query in region_queries:
        node = resolve_region_query(query, nodes_by_id, acronym_to_ids, name_to_ids)
        subtree_ids = _collect_subtree_ids(node)
        resolved.append({"query": query, "node": node, "subtree_ids": subtree_ids})

    vertex_table = pd.read_csv(vertex_csv_path)
    edge_table = pd.read_csv(edge_csv_path)

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

    annotation_zarr = open_zarr_dataset(annotation_zarr_path, dataset_name=annotation_dataset_name)
    if len(annotation_zarr.shape) != 3:
        raise ValueError(f"Annotation Zarr must be 3D, got shape={annotation_zarr.shape}")

    _started_at = time.time()
    result = _finalize_region_analysis(
        vertex_table=vertex_table,
        edge_table=edge_table,
        annotation_zarr=annotation_zarr,
        annotation_resolution_xyz=annotation_resolution_xyz,
        resolved=resolved,
        output_dir=output_dir,
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
                "annotation_zarr_path": str(annotation_zarr_path),
                "region_cfg_csv": str(region_cfg_csv),
                "regions": regions,
                "annotation_dataset_name": annotation_dataset_name,
                "annotation_resolution_xyz": annotation_resolution_xyz,
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
    "num_skeletons",
    "num_vertices",
    "num_edges",
    "num_branch_points",
    "num_end_points",
    "total_length_um",
    "mean_edge_length_um",
    "mean_radius_um",
    "median_radius_um",
    "min_radius_um",
    "max_radius_um",
    "approx_vessel_volume_um3",
]


def _finalize_region_analysis(
    vertex_table,
    edge_table,
    annotation_zarr,
    annotation_resolution_xyz,
    resolved,
    output_dir,
):
    if not vertex_table.empty:
        vertex_points = vertex_table[["z_um", "y_um", "x_um"]].to_numpy(dtype=np.float64)
    else:
        vertex_points = np.empty((0, 3), dtype=np.float64)
    vertex_labels = sample_annotation_labels_at_points_um(
        vertex_points, annotation_zarr, annotation_resolution_xyz
    )

    if not edge_table.empty:
        src = edge_table[["source_z_um", "source_y_um", "source_x_um"]].to_numpy(dtype=np.float64)
        tgt = edge_table[["target_z_um", "target_y_um", "target_x_um"]].to_numpy(dtype=np.float64)
        midpoints = (src + tgt) / 2.0
    else:
        midpoints = np.empty((0, 3), dtype=np.float64)
    edge_labels = sample_annotation_labels_at_points_um(
        midpoints, annotation_zarr, annotation_resolution_xyz
    )

    radius_lookup = {}
    if "radius_um" in vertex_table.columns and not vertex_table.empty:
        for row in vertex_table.itertuples(index=False):
            radius_lookup[(int(row.skeleton_id), int(row.node_id))] = float(row.radius_um)

    rows = []
    for entry in resolved:
        summary_row = _regional_summary(
            region_node=entry["node"],
            subtree_ids=entry["subtree_ids"],
            vertex_table=vertex_table,
            edge_table=edge_table,
            vertex_labels=vertex_labels,
            edge_labels=edge_labels,
            radius_lookup=radius_lookup,
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
    parser.add_argument("--annotation_zarr", required=True, help="Registered annotation label Zarr")
    parser.add_argument("--annotation_dataset_name", default="0", help="Dataset name inside annotation Zarr")
    parser.add_argument("--annotation_resolution_xyz", default="25,25,25", help="Annotation voxel size in um as x,y,z")
    parser.add_argument("--cfg", required=True, help="Allen region CSV path")
    parser.add_argument(
        "--regions",
        required=True,
        help="Comma/semicolon separated region queries (acronym, full name, or integer id)",
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
            annotation_zarr_path=args.annotation_zarr,
            region_cfg_csv=args.cfg,
            regions=args.regions,
            output_dir=args.output_dir,
            annotation_dataset_name=args.annotation_dataset_name,
            annotation_resolution_xyz=args.annotation_resolution_xyz,
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
