"""Vessel diameter analysis from kimimaro branch metrics, one entry point, two modes.

global mode
    Publication-ready whole-sample diameter histogram (PNG) from
    ``vessel_branch_metrics.csv`` (diameter_um = mean_radius_um * 2).
region mode
    Per-atlas-region diameter bin table (CSV). Branches are assigned by the
    atlas label at their endpoint midpoint, the same rule used by
    ``region_vessel_analysis``.

Run as a module:
    python -m pipeline_modules.tubule_reconstruction.vessel_diameter_analysis global --branch_csv vessel_branch_metrics.csv
    python -m pipeline_modules.tubule_reconstruction.vessel_diameter_analysis region --vertex_csv skeleton_vertices.csv --branch_csv vessel_branch_metrics.csv ...
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator

from .kimimaro_reconstruction import open_zarr_dataset, parse_resolution_xyz
from .region_vessel_analysis import (
    _attach_branch_midpoints,
    _collect_subtree_ids,
    load_region_tree_with_lookups,
    parse_region_list,
    resolve_region_query,
    sample_annotation_labels_at_points_um,
)


DEFAULT_BRANCH_CSV = "vessel_branch_metrics.csv"
DEFAULT_HISTOGRAM_OUTPUT = "vessel_diameter_histogram.png"
DEFAULT_BIN_EDGES_UM = (0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 15.0, 20.0, 30.0)


def _parse_pair(value: str, name: str) -> tuple[float, float]:
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 2:
        raise ValueError(f"{name} must be in 'a,b' format, got: {value}")
    try:
        return float(parts[0]), float(parts[1])
    except ValueError as exc:
        raise ValueError(f"{name} must contain numeric values, got: {value}") from exc


def parse_figsize(value: str) -> tuple[float, float]:
    return _parse_pair(value, "figsize")


def parse_xlim(value: str | None) -> tuple[float, float] | None:
    if value is None:
        return None
    lo, hi = _parse_pair(value, "xlim")
    if hi <= lo:
        raise ValueError(f"xlim upper bound must be greater than lower bound, got: {value}")
    return lo, hi


def load_branch_diameters(branch_csv_path: str | Path) -> np.ndarray:
    branch_csv_path = Path(branch_csv_path)
    if not branch_csv_path.exists():
        raise FileNotFoundError(f"Branch CSV not found: {branch_csv_path}")

    table = pd.read_csv(branch_csv_path)
    required_column = "mean_radius_um"
    if required_column not in table.columns:
        raise ValueError(
            f"Branch CSV must contain '{required_column}' column: {branch_csv_path}"
        )

    radii = pd.to_numeric(table[required_column], errors="coerce").to_numpy(dtype=np.float64)
    valid = np.isfinite(radii) & (radii > 0)
    diameters = radii[valid] * 2.0
    if diameters.size == 0:
        raise ValueError(
            "No valid vessel diameters found after filtering NaN, inf, and non-positive values."
        )
    return diameters


def summarize_diameters(diameters_um: np.ndarray) -> dict[str, float]:
    return {
        "n": int(diameters_um.size),
        "mean": float(np.mean(diameters_um)),
        "median": float(np.median(diameters_um)),
        "std": float(np.std(diameters_um, ddof=1)) if diameters_um.size > 1 else 0.0,
        "min": float(np.min(diameters_um)),
        "max": float(np.max(diameters_um)),
    }


def resolve_output_path(branch_csv_path: Path, output: str | None) -> Path:
    if output:
        return Path(output)
    return branch_csv_path.with_name(DEFAULT_HISTOGRAM_OUTPUT)


def plot_vessel_diameter_histogram(
    diameters_um: np.ndarray,
    *,
    output_path: str | Path,
    bins: int = 24,
    dpi: int = 300,
    title: str = "Vessel Diameter Distribution",
    xlabel: str = "Vessel diameter (um)",
    ylabel: str = "Section count",
    figsize: tuple[float, float] = (8.4, 5.6),
    xlim: tuple[float, float] | None = None,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    summary = summarize_diameters(diameters_um)

    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#243447",
            "axes.linewidth": 1.0,
            "axes.labelcolor": "#16202a",
            "axes.titleweight": "semibold",
            "xtick.color": "#243447",
            "ytick.color": "#243447",
            "font.size": 11,
            "font.family": "DejaVu Sans",
            "legend.labelcolor": "#16202a",
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
        }
    )

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)

    hist_color = "#325d88"
    edge_color = "#17324d"
    counts, bin_edges = np.histogram(diameters_um, bins=bins)

    ax.hist(
        diameters_um,
        bins=bin_edges,
        color=hist_color,
        edgecolor="#f4f7fa",
        linewidth=0.9,
        alpha=0.95,
    )
    ax.stairs(counts, bin_edges, color=edge_color, linewidth=1.5, alpha=0.95)

    ax.set_title(title, fontsize=16, pad=14)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    if xlim is not None:
        ax.set_xlim(*xlim)

    ax.grid(axis="y", color="#d7e3ea", linewidth=0.8, alpha=0.75)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    stats_text = "\n".join(
        [
            f"n = {summary['n']}",
            f"std = {summary['std']:.2f} um",
            f"min = {summary['min']:.2f} um",
            f"max = {summary['max']:.2f} um",
        ]
    )
    ax.text(
        0.98,
        0.97,
        stats_text,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10.5,
        color="#163042",
        bbox={
            "boxstyle": "round,pad=0.45",
            "facecolor": "#fbfdfe",
            "edgecolor": "#ccd9e2",
            "linewidth": 0.9,
        },
    )

    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    return output_path


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


def run_global(args) -> int:
    branch_csv_path = Path(args.branch_csv)
    diameters_um = load_branch_diameters(branch_csv_path)
    output_path = resolve_output_path(branch_csv_path, args.output)
    plot_vessel_diameter_histogram(
        diameters_um,
        output_path=output_path,
        bins=args.bins,
        dpi=args.dpi,
        title=args.title,
        xlabel=args.xlabel,
        ylabel=args.ylabel,
        figsize=parse_figsize(args.figsize),
        xlim=parse_xlim(args.xlim),
    )
    print(f"Saved vessel diameter histogram to: {output_path}")
    return 0


def run_region(args) -> int:
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
    print(f"Saved vessel diameter bins to: {output_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m pipeline_modules.tubule_reconstruction.vessel_diameter_analysis",
        description="Vessel diameter analysis: whole-sample histogram PNG (global) or per-atlas-region diameter bins CSV (region).",
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)

    hist = subparsers.add_parser(
        "global",
        help="Whole-sample diameter histogram PNG from vessel_branch_metrics.csv",
    )
    hist.add_argument(
        "--branch_csv",
        default=DEFAULT_BRANCH_CSV,
        help="Path to vessel_branch_metrics.csv",
    )
    hist.add_argument(
        "--output",
        default=None,
        help="Output image path. Defaults to vessel_diameter_histogram.png next to the CSV.",
    )
    hist.add_argument("--bins", type=int, default=24, help="Number of histogram bins")
    hist.add_argument("--dpi", type=int, default=300, help="Output image DPI")
    hist.add_argument("--title", default="Vessel Diameter Distribution")
    hist.add_argument("--xlabel", default="Vessel diameter (um)")
    hist.add_argument("--ylabel", default="Section count")
    hist.add_argument(
        "--figsize",
        default="8.4,5.6",
        help="Figure size in inches as width,height",
    )
    hist.add_argument(
        "--xlim",
        default=None,
        help="Optional x-axis limits as min,max",
    )
    hist.set_defaults(handler=run_global)

    region = subparsers.add_parser(
        "region",
        help="Per-atlas-region diameter bin CSV; requires annotation zarr and Allen-style region CSV",
    )
    region.add_argument("--vertex_csv", required=True, help="Path to skeleton_vertices.csv")
    region.add_argument("--branch_csv", required=True, help="Path to vessel_branch_metrics.csv")
    region.add_argument("--annotation_zarr", required=True, help="Registered atlas label Zarr")
    region.add_argument("--annotation_dataset_name", default="0", help="Dataset name inside annotation Zarr")
    region.add_argument("--annotation_resolution_xyz", required=True, help="Annotation voxel size in um as x,y,z")
    region.add_argument("--cfg", required=True, help="Allen-style region CSV")
    region.add_argument("--regions", required=True, help="Comma/semicolon separated region queries")
    region.add_argument(
        "--bin_edges_um",
        default="0,2,4,6,8,10,12,15,20,30",
        help="Increasing lower bin edges in um",
    )
    region.add_argument("--output", required=True, help="Output CSV path")
    region.set_defaults(handler=run_region)

    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        return args.handler(args)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
