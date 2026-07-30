"""Export region-specific vessel branch diameter bins from skeleton outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .kimimaro_reconstruction import open_zarr_dataset, parse_resolution_xyz
from .region_vessel_analysis import (
    _attach_branch_midpoints,
    _collect_subtree_ids,
    load_region_tree_with_lookups,
    parse_region_list,
    resolve_region_query,
    sample_annotation_labels_at_points_um,
)


DEFAULT_BIN_EDGES_UM = (0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 15.0, 20.0, 30.0)


def parse_bin_edges(value: str | tuple[float, ...] | list[float]) -> tuple[float, ...]:
    if isinstance(value, str):
        values = tuple(float(part.strip()) for part in value.split(",") if part.strip())
    else:
        values = tuple(float(part) for part in value)
    if len(values) < 2 or values[0] < 0 or any(right <= left for left, right in zip(values, values[1:])):
        raise ValueError("diameter bin edges must be at least two strictly increasing non-negative values")
    return values


def _format_value(value: float) -> str:
    return f"{value:g}"


def _resolve_regions(region_cfg_csv, regions):
    nodes_by_id, acronym_to_ids, name_to_ids = load_region_tree_with_lookups(region_cfg_csv)
    resolved = []
    for query in parse_region_list(regions):
        node = resolve_region_query(query, nodes_by_id, acronym_to_ids, name_to_ids)
        resolved.append({"query": query, "node": node, "subtree_ids": _collect_subtree_ids(node)})
    if not resolved:
        raise ValueError("At least one region query is required")
    return resolved


def summarize_region_vessel_diameter_bins(
    vertex_csv_path,
    branch_csv_path,
    annotation_zarr_path,
    region_cfg_csv,
    regions,
    *,
    annotation_resolution_xyz,
    annotation_dataset_name="0",
    bin_edges_um=DEFAULT_BIN_EDGES_UM,
) -> pd.DataFrame:
    """Bin mean branch diameters for each requested atlas region.

    Branches are assigned by the atlas label at their endpoint midpoint, the
    same assignment rule used by ``region_vessel_analysis``.
    """
    bin_edges_um = parse_bin_edges(bin_edges_um)
    vertex_table = pd.read_csv(vertex_csv_path)
    branch_table = pd.read_csv(branch_csv_path)
    required_columns = {"mean_radius_um", "branch_length_um"}
    missing = required_columns.difference(branch_table.columns)
    if missing:
        raise ValueError(f"Branch CSV is missing required columns: {sorted(missing)}")

    annotation_zarr = open_zarr_dataset(annotation_zarr_path, dataset_name=annotation_dataset_name)
    branch_table = _attach_branch_midpoints(branch_table, vertex_table)
    midpoints = branch_table[["mid_z_um", "mid_y_um", "mid_x_um"]].to_numpy(dtype=np.float64)
    finite_midpoints = np.all(np.isfinite(midpoints), axis=1)
    branch_labels = np.zeros(len(branch_table), dtype=np.int64)
    if finite_midpoints.any():
        branch_labels[finite_midpoints] = sample_annotation_labels_at_points_um(
            midpoints[finite_midpoints],
            annotation_zarr,
            parse_resolution_xyz(annotation_resolution_xyz),
        )

    diameters = pd.to_numeric(branch_table["mean_radius_um"], errors="coerce").to_numpy(dtype=np.float64) * 2.0
    lengths = pd.to_numeric(branch_table["branch_length_um"], errors="coerce").to_numpy(dtype=np.float64)
    valid_diameter = np.isfinite(diameters) & (diameters > 0)
    rows = []
    for entry in _resolve_regions(region_cfg_csv, regions):
        in_region = np.isin(branch_labels, np.asarray(entry["subtree_ids"], dtype=np.int64)) & valid_diameter
        region_diameters = diameters[in_region]
        region_lengths = lengths[in_region]
        total_count = int(region_diameters.size)
        valid_lengths = np.where(np.isfinite(region_lengths), region_lengths, 0.0)
        total_length = float(valid_lengths.sum())

        finite_edges = list(bin_edges_um)
        for index, lower in enumerate(finite_edges):
            upper = finite_edges[index + 1] if index + 1 < len(finite_edges) else np.inf
            in_bin = (region_diameters >= lower) & (region_diameters < upper)
            count = int(np.count_nonzero(in_bin))
            length_sum = float(valid_lengths[in_bin].sum())
            label = f"{_format_value(lower)}-<{_format_value(upper)}" if np.isfinite(upper) else f">={_format_value(lower)}"
            rows.append(
                {
                    "query": entry["query"],
                    "region_id": int(entry["node"]["id"]),
                    "region_acronym": entry["node"]["acronym"],
                    "region_name": entry["node"]["name"],
                    "diameter_bin_um": label,
                    "lower_um": float(lower),
                    "upper_um": float(upper) if np.isfinite(upper) else np.nan,
                    "branch_count": count,
                    "branch_percent": float(count / total_count * 100.0) if total_count else 0.0,
                    "total_branch_length_um": length_sum,
                    "length_percent": float(length_sum / total_length * 100.0) if total_length else 0.0,
                    "total_valid_branch_count": total_count,
                }
            )
    return pd.DataFrame(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export atlas-region vessel branch diameter bins.")
    parser.add_argument("--vertex_csv", required=True, help="Path to skeleton_vertices.csv")
    parser.add_argument("--branch_csv", required=True, help="Path to vessel_branch_metrics.csv")
    parser.add_argument("--annotation_zarr", required=True, help="Registered atlas label Zarr")
    parser.add_argument("--annotation_dataset_name", default="0", help="Dataset name inside annotation Zarr")
    parser.add_argument("--annotation_resolution_xyz", required=True, help="Annotation voxel size in um as x,y,z")
    parser.add_argument("--cfg", required=True, help="Allen-style region CSV")
    parser.add_argument("--regions", required=True, help="Comma/semicolon separated region queries")
    parser.add_argument("--bin_edges_um", default="0,2,4,6,8,10,12,15,20,30", help="Increasing lower bin edges in um")
    parser.add_argument("--output", required=True, help="Output CSV path")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        table = summarize_region_vessel_diameter_bins(
            args.vertex_csv,
            args.branch_csv,
            args.annotation_zarr,
            args.cfg,
            args.regions,
            annotation_resolution_xyz=args.annotation_resolution_xyz,
            annotation_dataset_name=args.annotation_dataset_name,
            bin_edges_um=args.bin_edges_um,
        )
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(output_path, index=False)
    except Exception as exc:
        print(f"Error: {exc}")
        return 1
    print(f"Saved vessel diameter bins to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
