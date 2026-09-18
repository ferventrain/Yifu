"""Comprehensive vessel network analysis report from skeleton reconstruction outputs.

Computes length / diameter / tortuosity / branch-point / volume / surface / loop
metrics from kimimaro CSVs or EDT polyline ``skeleton_edges.csv`` and writes a
per-mouse statistics workbook plus distribution figures.
"""
from __future__ import annotations

import argparse
import json
import pickle
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
EDT_BRANCH_CHUNK = 500_000
SPILL_DIRNAME = "_chunk_spill"


def parse_resolution_xyz(value):
    return tuple(float(part) for part in str(value).split(","))


def fmt_interval(lower, upper):
    if np.isfinite(upper):
        return f"{lower:g}-<{upper:g}"
    return f">={lower:g}"


def bincount_with_nan(values):
    clean = values[~np.isnan(values)]
    return clean, np.bincount(clean.astype(np.int64), minlength=0)


def csv_columns(path):
    return list(pd.read_csv(path, nrows=0).columns)


def is_edt_polyline_edges(edge_csv):
    cols = set(csv_columns(edge_csv))
    return "length_um" in cols and "edge_length_um" not in cols


def jsonable(value):
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def empty_vertex_stats():
    return {
        "num_vertices": 0,
        "num_endpoints": 0,
        "num_endpoints_non_boundary": 0,
        "num_branch_points": 0,
        "degree_histogram": {},
        "branch_point_degree_histogram": {},
    }


def stream_edges(edge_csv):
    total_length = 0.0
    num_edges = 0
    num_stitch = 0
    stitch_length = 0.0
    cols = set(csv_columns(edge_csv))
    length_col = "length_um" if "length_um" in cols else "edge_length_um"
    usecols = [length_col]
    if "is_stitch" in cols:
        usecols.append("is_stitch")
    for chunk in pd.read_csv(edge_csv, usecols=usecols, chunksize=EDGE_CHUNK, low_memory=False):
        length = pd.to_numeric(chunk[length_col], errors="coerce").to_numpy(dtype=np.float64)
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
    columns = set(csv_columns(vertex_csv))
    if "degree" not in columns:
        return None
    degree_hist = {}
    num_vertices = 0
    num_endpoints = 0
    num_endpoints_non_boundary = 0
    num_branch_points = 0
    branch_degree_hist = {}
    usecols = ["degree"]
    if "in_core" in columns:
        usecols.append("in_core")
    has_boundary = "touches_core_boundary" in columns
    if has_boundary:
        usecols.append("touches_core_boundary")
    for chunk in pd.read_csv(vertex_csv, usecols=usecols, chunksize=VERTEX_CHUNK, low_memory=False):
        if "in_core" in chunk.columns:
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
        if has_boundary:
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


NODE_KEY_STRIDE = 1 << 40


def stream_vertex_stats_from_edges(edge_csv):
    """Vertex degree statistics derived from edge endpoint lists.

    Fallback for skeleton outputs whose vertices CSV has no kimimaro ``degree``
    column (e.g. vessel_express_reconstruction). Node identity is the
    (skeleton_id, node_id) pair packed into one int64 key.
    """
    cols = set(csv_columns(edge_csv))
    usecols = ["skeleton_id", "source_node", "target_node"]
    missing = sorted(set(usecols).difference(cols))
    if missing:
        raise ValueError(f"Edge CSV is missing required columns: {missing}")
    key_parts = []
    count_parts = []
    for chunk in pd.read_csv(edge_csv, usecols=usecols, chunksize=EDGE_CHUNK, low_memory=False):
        skeleton_id = pd.to_numeric(chunk["skeleton_id"], errors="coerce").fillna(0).to_numpy(dtype=np.int64)
        source = pd.to_numeric(chunk["source_node"], errors="coerce").fillna(-1).to_numpy(dtype=np.int64)
        target = pd.to_numeric(chunk["target_node"], errors="coerce").fillna(-1).to_numpy(dtype=np.int64)
        keys = np.concatenate([skeleton_id * NODE_KEY_STRIDE + source, skeleton_id * NODE_KEY_STRIDE + target])
        unique, counts = np.unique(keys, return_counts=True)
        key_parts.append(unique)
        count_parts.append(counts)
    if not key_parts:
        return {
            "num_vertices": 0,
            "num_endpoints": 0,
            "num_endpoints_non_boundary": None,
            "num_branch_points": 0,
            "degree_histogram": {},
            "branch_point_degree_histogram": {},
        }
    all_keys = np.concatenate(key_parts)
    all_counts = np.concatenate(count_parts)
    _, inverse = np.unique(all_keys, return_inverse=True)
    degrees = np.bincount(inverse, weights=all_counts.astype(np.float64)).astype(np.int64)
    hist = np.bincount(np.clip(degrees, 0, None), minlength=0)
    degree_hist = {int(k): int(c) for k, c in enumerate(hist) if c}
    endpoint_mask = degrees == 1
    branch_mask = degrees >= 3
    if branch_mask.any():
        values, counts = np.unique(degrees[branch_mask], return_counts=True)
        branch_degree_hist = {int(k): int(c) for k, c in zip(values, counts)}
    else:
        branch_degree_hist = {}
    return {
        "num_vertices": int(degrees.size),
        "num_endpoints": int(endpoint_mask.sum()),
        "num_endpoints_non_boundary": None,
        "num_branch_points": int(branch_mask.sum()),
        "degree_histogram": degree_hist,
        "branch_point_degree_histogram": branch_degree_hist,
    }


EUCLIDEAN_MIN_UM = 1e-6


def is_express_branch_metrics(branch_csv):
    """True for vessel_express vessel_branch_metrics.csv (per-branch polylines)."""
    cols = set(csv_columns(branch_csv))
    return "length_um" in cols and "mean_radius_um" in cols and "branch_length_um" not in cols


def load_express_branch_table(branch_csv):
    """Kimimaro-like branch table from a vessel_express vessel_branch_metrics.csv.

    Rows are branch-point-to-branch-point polylines with path length in
    ``length_um``; tortuosity is path length over endpoint euclidean distance.
    Loops (both ends at the same location) get degenerate tortuosity values, so
    they are flagged ``is_loop`` and excluded from tortuosity statistics.
    """
    table = pd.read_csv(branch_csv, low_memory=False)
    required = {"length_um", "mean_radius_um", "source_z_um", "source_y_um", "source_x_um", "target_z_um", "target_y_um", "target_x_um"}
    missing = sorted(required.difference(table.columns))
    if missing:
        raise ValueError(f"Branch CSV is missing required columns: {missing}")
    source = table[["source_z_um", "source_y_um", "source_x_um"]].to_numpy(dtype=np.float64)
    target = table[["target_z_um", "target_y_um", "target_x_um"]].to_numpy(dtype=np.float64)
    euclidean = np.linalg.norm(target - source, axis=1)
    length = pd.to_numeric(table["length_um"], errors="coerce").to_numpy(dtype=np.float64)
    tortuosity = np.divide(length, euclidean, out=np.full_like(length, np.nan), where=euclidean > EUCLIDEAN_MIN_UM)
    if {"start_degree", "end_degree"}.issubset(table.columns):
        start_degree = pd.to_numeric(table["start_degree"], errors="coerce").fillna(0).to_numpy(dtype=np.int64)
        end_degree = pd.to_numeric(table["end_degree"], errors="coerce").fillna(0).to_numpy(dtype=np.int64)
        is_branch_to_branch = (start_degree >= 3) & (end_degree >= 3)
        is_terminal_branch = (start_degree == 1) | (end_degree == 1)
    else:
        is_branch_to_branch = np.ones(len(table), dtype=bool)
        is_terminal_branch = np.zeros(len(table), dtype=bool)
    if {"start_node", "end_node"}.issubset(table.columns):
        start_node = pd.to_numeric(table["start_node"], errors="coerce").fillna(-1).to_numpy(dtype=np.int64)
        end_node = pd.to_numeric(table["end_node"], errors="coerce").fillna(-2).to_numpy(dtype=np.int64)
        is_loop = (start_node == end_node) & (start_node >= 0)
    else:
        is_loop = euclidean <= EUCLIDEAN_MIN_UM
    table = table.assign(
        branch_length_um=length,
        tortuosity=tortuosity,
        is_branch_to_branch=is_branch_to_branch,
        is_terminal_branch=is_terminal_branch,
        is_loop=is_loop,
    )
    return table


def stream_edt_branches(edge_csv):
    """Build a kimimaro-like branch table from EDT polyline skeleton_edges.csv."""
    usecols = [
        "source_z_um",
        "source_y_um",
        "source_x_um",
        "target_z_um",
        "target_y_um",
        "target_x_um",
        "length_um",
        "mean_radius_um",
        "is_stitch",
    ]
    length_parts = []
    radius_parts = []
    tort_parts = []
    loop_parts = []
    total_length = 0.0
    num_rows = 0
    num_stitch = 0
    stitch_length = 0.0
    n_chunks = 0
    for chunk in pd.read_csv(edge_csv, usecols=usecols, chunksize=EDT_BRANCH_CHUNK, low_memory=False):
        n_chunks += 1
        if n_chunks == 1 or n_chunks % 10 == 0:
            print(f"  EDT branch chunk {n_chunks}, rows so far {num_rows + len(chunk)}", flush=True)
        length = pd.to_numeric(chunk["length_um"], errors="coerce").to_numpy(dtype=np.float64)
        length = np.nan_to_num(length, nan=0.0)
        stitch = chunk["is_stitch"].fillna(False).astype(bool).to_numpy()
        total_length += float(length.sum())
        num_rows += int(len(chunk))
        num_stitch += int(stitch.sum())
        stitch_length += float(length[stitch].sum())
        keep = ~stitch
        if not keep.any():
            continue
        src = chunk.loc[keep, ["source_z_um", "source_y_um", "source_x_um"]].to_numpy(dtype=np.float64)
        dst = chunk.loc[keep, ["target_z_um", "target_y_um", "target_x_um"]].to_numpy(dtype=np.float64)
        path = length[keep]
        euclidean = np.linalg.norm(dst - src, axis=1)
        tortuosity = np.divide(path, euclidean, out=np.full_like(path, np.nan), where=euclidean > 1e-12)
        radius = pd.to_numeric(chunk.loc[keep, "mean_radius_um"], errors="coerce").to_numpy(dtype=np.float64)
        length_parts.append(path)
        radius_parts.append(radius)
        tort_parts.append(tortuosity)
        loop_parts.append(euclidean <= 1e-6)
    if length_parts:
        branch_length = np.concatenate(length_parts)
        mean_radius = np.concatenate(radius_parts)
        tortuosity = np.concatenate(tort_parts)
        is_loop = np.concatenate(loop_parts)
    else:
        branch_length = np.array([], dtype=np.float64)
        mean_radius = np.array([], dtype=np.float64)
        tortuosity = np.array([], dtype=np.float64)
        is_loop = np.array([], dtype=bool)
    branch_table = pd.DataFrame(
        {
            "branch_length_um": branch_length,
            "mean_radius_um": mean_radius,
            "tortuosity": tortuosity,
            "is_loop": is_loop,
            "is_branch_to_branch": np.ones(len(branch_length), dtype=bool),
            "is_terminal_branch": np.zeros(len(branch_length), dtype=bool),
        }
    )
    return {
        "edge_stats": {
            "total_vessel_length_um": total_length,
            "num_edges": num_rows,
            "num_stitch_edges": num_stitch,
            "stitch_length_um": stitch_length,
        },
        "branch_table": branch_table,
    }


def spill_degree_stats(spill_dir):
    """Degree histograms from EDT chunk spill pickles (core skeleton voxels)."""
    stats = empty_vertex_stats()
    spill_path = Path(spill_dir)
    if not spill_path.is_dir():
        return stats
    degree_hist = {}
    branch_degree_hist = {}
    num_vertices = 0
    num_endpoints = 0
    num_branch_points = 0
    n_spill = 0
    paths = sorted(spill_path.glob("*.pkl"))
    for path in paths:
        n_spill += 1
        if n_spill == 1 or n_spill % 1000 == 0 or n_spill == len(paths):
            print(f"  spill {n_spill}/{len(paths)}", flush=True)
        with path.open("rb") as handle:
            payload = pickle.load(handle)
        degrees = payload.get("degrees")
        if degrees is None or len(degrees) == 0:
            continue
        degree_int = np.asarray(degrees, dtype=np.int64)
        degree_int = degree_int[degree_int >= 0]
        if degree_int.size == 0:
            continue
        num_vertices += int(degree_int.size)
        hist = np.bincount(degree_int, minlength=0)
        for k, count in enumerate(hist):
            if count:
                degree_hist[int(k)] = degree_hist.get(int(k), 0) + int(count)
        branch_mask = degree_int >= 3
        num_branch_points += int(branch_mask.sum())
        if branch_mask.any():
            sub_hist = np.bincount(degree_int[branch_mask], minlength=0)
            for k, count in enumerate(sub_hist):
                if count:
                    branch_degree_hist[int(k)] = branch_degree_hist.get(int(k), 0) + int(count)
        num_endpoints += int((degree_int == 1).sum())
    stats.update(
        {
            "num_vertices": num_vertices,
            "num_endpoints": num_endpoints,
            "num_endpoints_non_boundary": num_endpoints,
            "num_branch_points": num_branch_points,
            "degree_histogram": degree_hist,
            "branch_point_degree_histogram": branch_degree_hist,
        }
    )
    return stats


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
    total_valid_count = int(valid_dia.sum())
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
                "branch_percent": float(branch_count / max(total_valid_count, 1) * 100.0),
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
    parser.add_argument("--run_dir", required=True, help="Directory containing skeleton_edges.csv (kimimaro or EDT)")
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
    edt_mode = is_edt_polyline_edges(edge_csv)

    summary = {}
    summary["sample_id"] = args.sample_id
    summary["resolution_xyz_um"] = list(resolution)
    express_mode = (not edt_mode) and is_express_branch_metrics(branch_csv)
    summary["skeleton_source"] = (
        "edt_polyline" if edt_mode else ("vessel_express" if express_mode else "kimimaro")
    )

    if edt_mode:
        print("EDT polyline edges detected; streaming branches ...")
        edt = stream_edt_branches(edge_csv)
        summary.update(edt["edge_stats"])
        branch_table = edt["branch_table"]
        print("Reading chunk spill degrees ...")
        vertex_stats = spill_degree_stats(run_dir / SPILL_DIRNAME)
    else:
        print("Streaming edges ...")
        summary.update(stream_edges(edge_csv))
        if express_mode:
            print("vessel_express branch metrics detected; building branch table ...")
            branch_table = load_express_branch_table(branch_csv)
        else:
            print("Loading branch table ...")
            branch_table = pd.read_csv(branch_csv, low_memory=False)
        print("Streaming vertices ...")
        vertex_stats = stream_vertices(vertex_csv)
        if vertex_stats is None:
            print("Vertices CSV has no degree column; deriving degree stats from edges ...")
            vertex_stats = stream_vertex_stats_from_edges(edge_csv)

    summary.update(
        {
            key: vertex_stats[key]
            for key in ("num_vertices", "num_endpoints", "num_endpoints_non_boundary", "num_branch_points")
        }
    )

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
    summary = jsonable(summary)
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
