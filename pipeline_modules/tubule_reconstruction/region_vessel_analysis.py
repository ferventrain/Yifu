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
    if not vertices_in.empty and not local_degrees.empty:
        deg_values = local_degrees.reindex(
            pd.MultiIndex.from_frame(vertices_in[["skeleton_id", "node_id"]]),
            fill_value=0,
        ).to_numpy(dtype=np.int64)
        num_branch_points = int(np.sum(deg_values >= 3))
        num_end_points = int(np.sum(deg_values == 1))

    skeleton_ids_in = set()
    if not vertices_in.empty:
        skeleton_ids_in.update(vertices_in["skeleton_id"].astype(int).tolist())
    if not edges_in.empty:
        skeleton_ids_in.update(edges_in["skeleton_id"].astype(int).tolist())

    vessel_volume_um3 = np.nan
    if not edges_in.empty and radius_lookup is not None and not radius_lookup.empty:
        src_radii = radius_lookup.reindex(
            pd.MultiIndex.from_frame(edges_in[["skeleton_id", "source_node"]]),
            fill_value=np.nan,
        ).to_numpy(dtype=np.float64)
        tgt_radii = radius_lookup.reindex(
            pd.MultiIndex.from_frame(edges_in[["skeleton_id", "target_node"]]),
            fill_value=np.nan,
        ).to_numpy(dtype=np.float64)
        valid = np.isfinite(src_radii) & np.isfinite(tgt_radii)
        if valid.any():
            r_mean = 0.5 * (src_radii[valid] + tgt_radii[valid])
            lengths = edges_in["edge_length_um"].to_numpy(dtype=np.float64)[valid]
            vessel_volume_um3 = float(np.sum(lengths * np.pi * r_mean * r_mean))
        else:
            vessel_volume_um3 = 0.0

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
    annotation_zarr_path=None,
    region_cfg_csv=None,
    regions=None,
    region_groups=None,
    output_dir=None,
    annotation_dataset_name="0",
    annotation_resolution_xyz=None,
    config_path=None,
):
    """Compute per-region vessel parameters from existing skeleton CSV outputs."""
    if region_cfg_csv is None:
        raise ValueError("region_cfg_csv is required.")
    if regions is None and region_groups is None:
        raise ValueError("regions or region_groups is required.")
    region_queries = parse_region_list(regions) if regions else []
    if region_groups is not None:
        if isinstance(region_groups, str):
            candidate = region_groups.strip()
            if Path(candidate).is_file():
                with open(candidate, "r", encoding="utf-8") as fh:
                    region_groups = json.load(fh)
            else:
                region_groups = json.loads(candidate)
    if not region_queries and not region_groups:
        raise ValueError("No regions or region groups provided.")

    logger.info("Loading region tree from %s", region_cfg_csv)
    nodes_by_id, acronym_to_ids, name_to_ids = load_region_tree_with_lookups(region_cfg_csv)
    if region_queries:
        logger.info("Resolving %d region query(ies): %s", len(region_queries), region_queries)
    resolved = []
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
                "region_groups": region_groups,
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
    logger.info("Sampling annotation labels for %d vertices ...", len(vertex_points))
    vertex_labels = sample_annotation_labels_at_points_um(
        vertex_points, annotation_zarr, annotation_resolution_xyz
    )

    if not edge_table.empty:
        src = edge_table[["source_z_um", "source_y_um", "source_x_um"]].to_numpy(dtype=np.float64)
        tgt = edge_table[["target_z_um", "target_y_um", "target_x_um"]].to_numpy(dtype=np.float64)
        midpoints = (src + tgt) / 2.0
    else:
        midpoints = np.empty((0, 3), dtype=np.float64)
    logger.info("Sampling annotation labels for %d edge midpoints ...", len(midpoints))
    edge_labels = sample_annotation_labels_at_points_um(
        midpoints, annotation_zarr, annotation_resolution_xyz
    )

    radius_lookup = None
    if "radius_um" in vertex_table.columns and not vertex_table.empty:
        radius_lookup = vertex_table.drop_duplicates(
            subset=["skeleton_id", "node_id"], keep="last"
        ).set_index(["skeleton_id", "node_id"])["radius_um"]

    rows = []
    logger.info("Computing per-region vessel statistics ...")
    for entry in tqdm(resolved, desc="Regions"):
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
            region_groups=args.region_groups,
            output_dir=args.output_dir,
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
