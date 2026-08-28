"""Comprehensive vessel network analysis report from kimimaro reconstruction outputs.

Computes length / diameter / tortuosity / branch-point / volume / surface / loop
metrics from the skeleton CSVs produced by ``kimimaro_reconstruction_fast.py``
and writes a per-mouse statistics workbook plus distribution figures.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DEFAULT_LENGTH_BIN_EDGES = (0.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0, 1000.0)
DEFAULT_DIAM_BIN_EDGES = (0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 15.0, 20.0, 30.0)
DEFAULT_TORT_BIN_EDGES = (1.0, 1.1, 1.2, 1.5, 2.0, 3.0, 5.0)

EDGE_CHUNK = 5_000_000
VERTEX_CHUNK = 5_000_000
BRANCH_CHUNK = 1_000_000


def parse_resolution_xyz(value):
    return tuple(float(part) for part in str(value).split(","))


def fmt_interval(lower, upper):
    if np.isfinite(upper):
        return f"{lower:g}-<{upper:g}"
    return f">={lower:g}"


def bincount_with_nan(values):
    clean = values[~np.isnan(values)]
    return clean, np.bincount(clean.astype(np.int64), minlength=0)


def stream_edges(edge_csv):
    total_length = 0.0
    num_edges = 0
    num_stitch = 0
    stitch_length = 0.0
    stitch_columns = [c for c in ("edge_length_um", "is_stitch") if c]
    usecols = stitch_columns
    for chunk in pd.read_csv(edge_csv, usecols=usecols, chunksize=EDGE_CHUNK, low_memory=False):
        length = pd.to_numeric(chunk["edge_length_um"], errors="coerce").to_numpy(dtype=np.float64)
        length = np.nan_to_num(length, nan=0.0)
        total_length += float(length.sum())
        num_edges += int(len(chunk))
        if "is_stitch" in chunk.columns:
            stitch_mask = chunk["is_stitch"].fillna(False).astype(bool).to_numpy()
            num_stitch += int(stitch_mask.sum())
            stitch_length += float(length[stitch_mask].sum())
    return {
        "total_vessel_length_um": total_length,
        "num_edges": num_edges,
        "num_stitch_edges": num_stitch,
        "stitch_length_um": stitch_length,
    }


def stream_vertices(vertex_csv):
    degree_hist = {}
    num_vertices = 0
    num_endpoints = 0
    num_endpoints_non_boundary = 0
    num_branch_points = 0
    branch_degree_hist = {}
    usecols = ["degree", "in_core", "touches_core_boundary"]
    for chunk in pd.read_csv(vertex_csv, usecols=usecols, chunksize=VERTEX_CHUNK, low_memory=False):
        in_core = chunk["in_core"].fillna(False).astype(bool).to_numpy()
        if not in_core.any():
            continue
        chunk = chunk.loc[in_core]
        degree = pd.to_numeric(chunk["degree"], errors="coerce").to_numpy(dtype=np.float64)
        valid = ~np.isnan(degree)
        degree = degree[valid]
        num_vertices += int(len(degree))
        degree_int = degree.astype(np.int64)
        hist = np.bincount(degree_int, minlength=0)
        for k, count in enumerate(hist):
            if count:
                degree_hist[int(k)] = degree_hist.get(int(k), 0) + int(count)
        branch_mask = degree_int >= 3
        num_branch_points += int(branch_mask.sum())
        for k in np.unique(degree_int[branch_mask]):
            sub = branch_mask & (degree_int == k)
            branch_degree_hist[int(k)] = branch_degree_hist.get(int(k), 0) + int(sub.sum())
        endpoint_mask = degree_int == 1
        num_endpoints += int(endpoint_mask.sum())
        boundary = chunk["touches_core_boundary"].fillna(False).astype(bool).to_numpy()[valid]
        num_endpoints_non_boundary += int((endpoint_mask & ~boundary).sum())
    return {
        "num_vertices": num_vertices,
        "num_endpoints": num_endpoints,
        "num_endpoints_non_boundary": num_endpoints_non_boundary,
        "num_branch_points": num_branch_points,
        "degree_histogram": degree_hist,
        "branch_point_degree_histogram": branch_degree_hist,
    }


def bin_stats(values, bin_edges):
    edges = list(bin_edges) + [np.inf]
    counts = np.zeros(len(bin_edges), dtype=np.int64)
    sums = np.zeros(len(bin_edges), dtype=np.float64)
    for index, lower in enumerate(bin_edges):
        upper = edges[index + 1]
        mask = (values >= lower) & (values < upper)
        counts[index] = int(mask.sum())
        sums[index] = float(values[mask].sum())
    return counts, sums


def tortuosity_range(values, mask):
    sub = values[mask]
    if sub.size == 0:
        return np.nan, np.nan, np.nan
    return float(np.nanmin(sub)), float(np.nanmax(sub)), float(np.nanmean(sub))


def build_diameter_rows(branch_table):
    diameter = pd.to_numeric(branch_table["mean_radius_um"], errors="coerce").to_numpy(dtype=np.float64) * 2.0
    length = pd.to_numeric(branch_table["branch_length_um"], errors="coerce").to_numpy(dtype=np.float64)
    tortuosity = pd.to_numeric(branch_table["tortuosity"], errors="coerce").to_numpy(dtype=np.float64)
    valid_dia = np.isfinite(diameter) & (diameter > 0)
    length = np.where(np.isfinite(length), length, 0.0)
    rows = []
    for index, lower in enumerate(DEFAULT_DIAM_BIN_EDGES):
        upper = list(DEFAULT_DIAM_BIN_EDGES) + [np.inf]
        upper = upper[index + 1]
        mask = valid_dia & (diameter >= lower) & (diameter < upper)
        counts, _ = bin_stats(diameter[mask], (lower,))
        branch_count = int(counts[0])
        length_sum = float(length[mask].sum())
        tort_min, tort_max, tort_mean = tortuosity_range(tortuosity, mask)
        radii = diameter[mask] / 2.0
        segment_lengths = length[mask]
        volume = float(np.sum(np.pi * radii ** 2 * segment_lengths))
        surface = float(np.sum(2.0 * np.pi * radii * segment_lengths))
        rows.append(
            {
                "diameter_bin_um": fmt_interval(lower, upper),
                "lower_um": lower,
                "upper_um": upper if np.isfinite(upper) else np.nan,
                "branch_count": branch_count,
                "branch_percent": float(branch_count / max(int(mask.sum()), 1) * 100.0),
                "length_um": length_sum,
                "length_percent": float(length_sum / max(float(length[valid_dia].sum()), 1e-12) * 100.0),
                "tortuosity_min": tort_min,
                "tortuosity_max": tort_max,
                "tortuosity_mean": tort_mean,
                "vessel_volume_um3": volume,
                "surface_area_um2": surface,
            }
        )
    return pd.DataFrame(rows)


def build_length_rows(branch_table, branch_to_branch_only=True):
    length = pd.to_numeric(branch_table["branch_length_um"], errors="coerce").to_numpy(dtype=np.float64)
    if branch_to_branch_only and "is_branch_to_branch" in branch_table.columns:
        keep = branch_table["is_branch_to_branch"].fillna(False).astype(bool).to_numpy()
        length = np.where(keep, length, np.nan)
    total = float(np.nansum(length))
    rows = []
    for index, lower in enumerate(DEFAULT_LENGTH_BIN_EDGES):
        upper = list(DEFAULT_LENGTH_BIN_EDGES) + [np.inf]
        upper = upper[index + 1]
        mask = np.isfinite(length) & (length >= lower) & (length < upper)
        count = int(mask.sum())
        length_sum = float(np.nansum(length[mask]))
        rows.append(
            {
                "length_bin_um": fmt_interval(lower, upper),
                "lower_um": lower,
                "upper_um": upper if np.isfinite(upper) else np.nan,
                "segment_count": count,
                "segment_percent": float(count / max(int(np.isfinite(length).sum()), 1) * 100.0),
                "length_um": length_sum,
                "length_percent": float(length_sum / max(total, 1e-12) * 100.0),
            }
        )
    return pd.DataFrame(rows)


def build_tortuosity_diameter_rows(branch_table):
    diameter = pd.to_numeric(branch_table["mean_radius_um"], errors="coerce").to_numpy(dtype=np.float64) * 2.0
    length = pd.to_numeric(branch_table["branch_length_um"], errors="coerce").to_numpy(dtype=np.float64)
    tortuosity = pd.to_numeric(branch_table["tortuosity"], errors="coerce").to_numpy(dtype=np.float64)
    valid = np.isfinite(diameter) & (diameter > 0) & np.isfinite(tortuosity) & (tortuosity >= 1.0)
    length = np.where(np.isfinite(length), length, 0.0)
    total_length = float(length[valid].sum())
    rows = []
    for t_index, t_lower in enumerate(DEFAULT_TORT_BIN_EDGES):
        t_upper = list(DEFAULT_TORT_BIN_EDGES) + [np.inf]
        t_upper = t_upper[t_index + 1]
        t_mask = valid & (tortuosity >= t_lower) & (tortuosity < t_upper)
        for d_index, d_lower in enumerate(DEFAULT_DIAM_BIN_EDGES):
            d_upper = list(DEFAULT_DIAM_BIN_EDGES) + [np.inf]
            d_upper = d_upper[d_index + 1]
            d_mask = (diameter >= d_lower) & (diameter < d_upper)
            mask = t_mask & d_mask
            length_sum = float(length[mask].sum())
            rows.append(
                {
                    "tortuosity_bin": fmt_interval(t_lower, t_upper),
                    "diameter_bin_um": fmt_interval(d_lower, d_upper),
                    "length_um": length_sum,
                    "length_percent": float(length_sum / max(total_length, 1e-12) * 100.0),
                }
            )
    return pd.DataFrame(rows)


def save_figure(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_length_distribution(length_df, output_path):
    fig, ax = plt.subplots(figsize=(9, 5))
    labels = [row["length_bin_um"] for _, row in length_df.iterrows()]
    counts = length_df["segment_count"].to_numpy(dtype=np.float64)
    ax.bar(range(len(labels)), counts, color="#4C72B0")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_xlabel("Segment length (um)")
    ax.set_ylabel("Segment count")
    ax.set_title("Vessel segment length distribution (branch-point to branch-point)")
    ax.set_yscale("log")
    save_figure(fig, output_path)


def plot_diameter_distribution(diam_df, output_path):
    fig, ax = plt.subplots(figsize=(9, 5))
    labels = [row["diameter_bin_um"] for _, row in diam_df.iterrows()]
    x = np.arange(len(labels))
    length_pct = diam_df["length_percent"].to_numpy(dtype=np.float64)
    ax.bar(x, length_pct, color="#55A868")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_xlabel("Vessel diameter (um)")
    ax.set_ylabel("Length percent (%)")
    ax.set_title("Vessel length by diameter bin")
    save_figure(fig, output_path)


def plot_branch_degree_distribution(hist, output_path):
    fig, ax = plt.subplots(figsize=(9, 5))
    degrees = sorted(hist)
    counts = [hist[d] for d in degrees]
    ax.bar([str(d) for d in degrees], counts, color="#C44E52")
    ax.set_xlabel("Number of connected vessel segments at branch point")
    ax.set_ylabel("Branch point count")
    ax.set_title("Branch point degree distribution")
    ax.set_yscale("log")
    save_figure(fig, output_path)


def plot_tortuosity_diameter_heatmap(tort_diam_df, output_path):
    pivot = tort_diam_df.pivot_table(
        index="tortuosity_bin", columns="diameter_bin_um", values="length_percent", fill_value=0.0
    )
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(pivot.to_numpy(dtype=np.float64), aspect="auto", cmap="viridis")
    ax.set_xticks(range(pivot.shape[1]))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
    ax.set_yticks(range(pivot.shape[0]))
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel("Vessel diameter (um)")
    ax.set_ylabel("Tortuosity bin")
    ax.set_title("Vessel length percent by tortuosity and diameter")
    fig.colorbar(im, ax=ax, label="Length percent (%)")
    save_figure(fig, output_path)


def plot_tortuosity_histogram(tortuosity, output_path):
    clean = tortuosity[np.isfinite(tortuosity) & (tortuosity >= 1.0)]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(clean, bins=100, color="#8172B2")
    ax.set_xlabel("Tortuosity")
    ax.set_ylabel("Branch count")
    ax.set_title("Branch tortuosity distribution")
    ax.set_yscale("log")
    save_figure(fig, output_path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Compute comprehensive vessel network analysis report")
    parser.add_argument("--run_dir", required=True, help="Directory containing skeleton_*.csv and vessel_branch_metrics.csv")
    parser.add_argument("--sample_id", default="", help="Sample / mouse identifier")
    parser.add_argument("--output_dir", required=True, help="Directory for report outputs")
    parser.add_argument("--resolution_xyz", default="1.8,1.8,2.0", help="Voxel size in um as x,y,z")
    parser.add_argument("--region_volume_um3", default="", help="Optional whole-brain vessel volume in um3 from region scan")
    parser.add_argument("--hpf_volume_um3", default="", help="Optional HPF vessel volume in um3 from region scan")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    resolution = parse_resolution_xyz(args.resolution_xyz)
    voxel_volume = float(np.prod(resolution))

    edge_csv = run_dir / "skeleton_edges.csv"
    vertex_csv = run_dir / "skeleton_vertices.csv"
    branch_csv = run_dir / "vessel_branch_metrics.csv"

    summary = {}
    summary["sample_id"] = args.sample_id
    summary["resolution_xyz_um"] = list(resolution)

    print("Streaming edges ...")
    edge_stats = stream_edges(edge_csv)
    summary.update(edge_stats)

    print("Streaming vertices ...")
    vertex_stats = stream_vertices(vertex_csv)
    summary.update(
        {
            key: vertex_stats[key]
            for key in ("num_vertices", "num_endpoints", "num_endpoints_non_boundary", "num_branch_points")
        }
    )

    print("Loading branch table ...")
    branch_table = pd.read_csv(branch_csv, low_memory=False)
    length_um = pd.to_numeric(branch_table["branch_length_um"], errors="coerce").to_numpy(dtype=np.float64)
    radius_um = pd.to_numeric(branch_table["mean_radius_um"], errors="coerce").to_numpy(dtype=np.float64)
    tortuosity = pd.to_numeric(branch_table["tortuosity"], errors="coerce").to_numpy(dtype=np.float64)
    valid_radius = np.isfinite(radius_um) & (radius_um > 0)
    valid_length = np.where(np.isfinite(length_um), length_um, 0.0)

    summary["num_branches_total"] = int(len(branch_table))
    summary["num_branch_to_branch_segments"] = int(
        branch_table["is_branch_to_branch"].fillna(False).astype(bool).sum()
    )
    summary["num_terminal_branches"] = int(branch_table["is_terminal_branch"].fillna(False).astype(bool).sum())
    summary["num_vessel_loops"] = int(branch_table["is_loop"].fillna(False).astype(bool).sum())
    summary["branch_length_mean_um"] = float(np.nanmean(length_um)) if np.isfinite(length_um).any() else np.nan
    summary["branch_length_sd_um"] = float(np.nanstd(length_um)) if np.isfinite(length_um).any() else np.nan
    summary["branch_length_sum_um"] = float(np.nansum(length_um))
    summary["mean_diameter_um"] = float(np.nanmean(radius_um[valid_radius]) * 2.0) if valid_radius.any() else np.nan
    summary["vessel_volume_cylinder_um3"] = float(np.sum(np.pi * radius_um[valid_radius] ** 2 * valid_length[valid_radius]))
    summary["surface_area_total_um2"] = float(np.sum(2.0 * np.pi * radius_um[valid_radius] * valid_length[valid_radius]))
    summary["mean_radius_um"] = float(np.nanmean(radius_um)) if np.isfinite(radius_um).any() else np.nan
    summary["tortuosity_min"] = float(np.nanmin(tortuosity)) if np.isfinite(tortuosity).any() else np.nan
    summary["tortuosity_max"] = float(np.nanmax(tortuosity)) if np.isfinite(tortuosity).any() else np.nan
    summary["tortuosity_mean"] = float(np.nanmean(tortuosity)) if np.isfinite(tortuosity).any() else np.nan
    summary["tortuosity_median"] = float(np.nanmedian(tortuosity)) if np.isfinite(tortuosity).any() else np.nan

    mask_voxels = None
    run_summary_path = run_dir / "vessel_network_summary.json"
    if run_summary_path.exists():
        run_summary = json.loads(run_summary_path.read_text(encoding="utf-8"))
        mask_voxels = run_summary.get("mask_voxels")
        summary["mask_voxels"] = mask_voxels
        summary["vessel_volume_mask_um3"] = (
            float(mask_voxels * voxel_volume) if mask_voxels is not None else np.nan
        )
        summary["num_stitch_edges_run"] = run_summary.get("num_stitch_edges")
        summary["connected_components_run"] = run_summary.get("connected_components")
        summary["processed_chunks"] = run_summary.get("processed_chunks")

    if args.region_volume_um3:
        summary["vessel_volume_whole_brain_um3"] = float(args.region_volume_um3)
    if args.hpf_volume_um3:
        summary["vessel_volume_hpf_um3"] = float(args.hpf_volume_um3)

    length_df = build_length_rows(branch_table, branch_to_branch_only=True)
    diam_df = build_diameter_rows(branch_table)
    tort_diam_df = build_tortuosity_diameter_rows(branch_table)
    branch_degree_hist = vertex_stats["branch_point_degree_histogram"]

    print("Writing tables and figures ...")
    length_df.to_csv(output_dir / "length_distribution.csv", index=False)
    diam_df.to_csv(output_dir / "diameter_distribution.csv", index=False)
    tort_diam_df.to_csv(output_dir / "tortuosity_diameter_length_percent.csv", index=False)
    pd.DataFrame(
        [
            {"branch_point_degree": int(k), "count": int(v)}
            for k, v in sorted(branch_degree_hist.items())
        ]
    ).to_csv(output_dir / "branch_point_degree_distribution.csv", index=False)

    plot_length_distribution(length_df, figures_dir / "length_distribution.png")
    plot_diameter_distribution(diam_df, figures_dir / "diameter_distribution.png")
    plot_branch_degree_distribution(branch_degree_hist, figures_dir / "branch_point_degree_distribution.png")
    plot_tortuosity_diameter_heatmap(tort_diam_df, figures_dir / "tortuosity_diameter_length_percent.png")
    plot_tortuosity_histogram(tortuosity, figures_dir / "tortuosity_distribution.png")

    summary_path = output_dir / "vessel_analysis_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    xlsx_path = output_dir / "vessel_analysis_table.xlsx"
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        pd.DataFrame([summary]).to_excel(writer, sheet_name="mouse_summary", index=False)
        length_df.to_excel(writer, sheet_name="length_distribution", index=False)
        diam_df.to_excel(writer, sheet_name="diameter_distribution", index=False)
        tort_diam_df.to_excel(writer, sheet_name="tortuosity_x_diameter", index=False)
        pd.DataFrame(
            [
                {"branch_point_degree": int(k), "count": int(v)}
                for k, v in sorted(branch_degree_hist.items())
            ]
        ).to_excel(writer, sheet_name="branch_point_degree", index=False)

    print(f"Summary written to {summary_path}")
    print(f"Workbook written to {xlsx_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
