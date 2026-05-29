#!/usr/bin/env python3
"""
Zarr-native block-graph brain region signal analysis.
"""

import argparse
import ast
import concurrent.futures
import json
import logging
import shutil
import time
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import zarr
from openpyxl.styles import Alignment, Font
from scipy import ndimage
from tqdm import tqdm

try:
    from pipeline_modules.utils.errors import ErrorCode, PipelineError
    from pipeline_modules.utils.run_manifest import write_run_manifest
except ImportError:
    PipelineError = None  # type: ignore[assignment,misc]
    ErrorCode = None  # type: ignore[assignment]
    write_run_manifest = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


CONNECTIVITY_3D = np.ones((3, 3, 3), dtype=np.uint8)
PAIR_DTYPE = np.dtype([("component", np.int64), ("region", np.int64)])
ROOT_REGION_DTYPE = np.dtype([("root", np.int64), ("region", np.int64)])
PAIR_HEMISPHERE_DTYPE = np.dtype(
    [("component", np.int64), ("region", np.int64), ("hemisphere", np.int8)]
)
ROOT_REGION_HEMISPHERE_DTYPE = np.dtype(
    [("root", np.int64), ("region", np.int64), ("hemisphere", np.int8)]
)
# Must match atlas_label_to_hemisphere.py: 0=background, 1=left, 2=right.
LEFT_HEMISPHERE_ID = np.int8(1)
RIGHT_HEMISPHERE_ID = np.int8(2)
HEMISPHERE_NAMES = {
    int(LEFT_HEMISPHERE_ID): "Left",
    int(RIGHT_HEMISPHERE_ID): "Right",
}
FORWARD_BLOCK_OFFSETS = [
    (dz, dy, dx)
    for dz, dy, dx in product((-1, 0, 1), repeat=3)
    if not (dz == 0 and dy == 0 and dx == 0)
    and (dz > 0 or (dz == 0 and dy > 0) or (dz == 0 and dy == 0 and dx > 0))
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Zarr-native block graph analysis for brain-region object statistics."
    )
    parser.add_argument("--mask_zarr", required=True, help="Mask Zarr path")
    parser.add_argument("--label_zarr", required=True, help="Registered label Zarr path")
    parser.add_argument("--signal_zarr", required=True, help="Signal Zarr path")
    parser.add_argument("--cfg", required=True, help="Path to region CSV file")
    parser.add_argument("--output", required=True, help="Output Excel file path")
    parser.add_argument("--dataset_name", default="0", help="Dataset name inside the Zarr group")
    parser.add_argument("--block_size", default="", help="Override block size as z,y,x")
    parser.add_argument("--foreground_label", type=int, default=1, help="Foreground label in equality mode")
    parser.add_argument(
        "--foreground_mode",
        choices=("equal", "nonzero"),
        default="equal",
        help="Mask interpretation",
    )
    parser.add_argument("--min_voxels", type=int, default=10, help="Minimum merged component size")
    parser.add_argument("--flush_every", type=int, default=25, help="Rewrite Excel after every N rows")
    parser.add_argument("--resolution_xyz", default="1,1,1", help="Voxel size in microns as x,y,z")
    parser.add_argument("--tmp_dir", default="", help="Temporary folder for block artifacts")
    parser.add_argument("--keep_tmp", action="store_true", help="Keep temporary block artifacts")
    parser.add_argument(
        "--hemisphere_zarr",
        default="",
        help="Optional hemisphere label Zarr. When provided, left/right stats are enabled.",
    )
    parser.add_argument(
        "--pass1_workers",
        type=int,
        default=1,
        help="Number of worker processes for Pass 1 block scanning",
    )
    parser.add_argument(
        "--json_logs",
        action="store_true",
        help="Emit NDJSON log records to stderr instead of plain text",
    )
    return parser.parse_args()


def parse_block_size(block_text):
    if not str(block_text).strip():
        return None
    parts = [part.strip() for part in str(block_text).split(",") if part.strip()]
    if len(parts) != 3:
        raise ValueError(f"block_size must have 3 comma-separated integers, got: {block_text}")
    block_size = tuple(int(part) for part in parts)
    if any(size <= 0 for size in block_size):
        raise ValueError(f"block_size values must be positive, got: {block_size}")
    return block_size


def parse_resolution_xyz(resolution_text):
    parts = [part.strip() for part in str(resolution_text).split(",") if part.strip()]
    if len(parts) != 3:
        raise ValueError(f"resolution_xyz must have 3 comma-separated values, got: {resolution_text}")
    resolution_xyz = tuple(float(part) for part in parts)
    if any(value <= 0 for value in resolution_xyz):
        raise ValueError(f"resolution_xyz values must be positive, got: {resolution_xyz}")
    return resolution_xyz


def open_zarr_dataset(path_like, dataset_name):
    path = Path(path_like)
    if not path.exists():
        raise FileNotFoundError(f"Zarr path not found: {path}")

    root = zarr.open(str(path), mode="r")
    if isinstance(root, zarr.Array):
        return root

    if dataset_name in root:
        dataset = root[dataset_name]
        if isinstance(dataset, zarr.Array):
            return dataset

    array_keys = list(root.array_keys())
    if len(array_keys) == 1:
        return root[array_keys[0]]

    raise ValueError(
        f"Could not resolve a Zarr array from {path}. "
        f"Available arrays: {array_keys}, requested dataset_name={dataset_name}"
    )


def validate_zarr_inputs(mask_zarr, label_zarr, signal_zarr):
    if mask_zarr.shape != label_zarr.shape or mask_zarr.shape != signal_zarr.shape:
        raise ValueError(
            f"Shape mismatch: mask={mask_zarr.shape}, label={label_zarr.shape}, signal={signal_zarr.shape}"
        )
    if len(mask_zarr.shape) != 3:
        raise ValueError(f"Expected 3D arrays, got shape={mask_zarr.shape}")
    logger.info("volume shape=%s", mask_zarr.shape)


def choose_block_shape(mask_zarr, label_zarr, signal_zarr, requested_block_shape):
    if requested_block_shape is not None:
        return requested_block_shape

    candidate_chunks = []
    for array in (mask_zarr, label_zarr, signal_zarr):
        chunks = array.chunks
        if chunks is None:
            continue
        candidate_chunks.append(tuple(int(value) for value in chunks[:3]))

    if candidate_chunks:
        block_shape = candidate_chunks[0]
        logger.info("Using Zarr chunk size as block size: %s", block_shape)
        return block_shape

    raise ValueError("Could not infer block size from Zarr chunks; please provide --block_size")


def iter_block_specs(volume_shape, block_shape):
    grid_shape = tuple((dim + block - 1) // block for dim, block in zip(volume_shape, block_shape))
    block_id = 0
    for gz in range(grid_shape[0]):
        z0 = gz * block_shape[0]
        z1 = min(z0 + block_shape[0], volume_shape[0])
        for gy in range(grid_shape[1]):
            y0 = gy * block_shape[1]
            y1 = min(y0 + block_shape[1], volume_shape[1])
            for gx in range(grid_shape[2]):
                x0 = gx * block_shape[2]
                x1 = min(x0 + block_shape[2], volume_shape[2])
                yield {
                    "block_id": block_id,
                    "grid_index": (gz, gy, gx),
                    "start": (z0, y0, x0),
                    "stop": (z1, y1, x1),
                }
                block_id += 1


def build_boundary_mask(block_shape):
    mask = np.zeros(block_shape, dtype=bool)
    mask[0, :, :] = True
    mask[-1, :, :] = True
    mask[:, 0, :] = True
    mask[:, -1, :] = True
    mask[:, :, 0] = True
    mask[:, :, -1] = True
    return mask


def parse_structure_id_path(path_text):
    path_values = ast.literal_eval(str(path_text))
    return [int(value) for value in path_values]


def parse_acronym_text(acronym_text):
    try:
        acronym_values = ast.literal_eval(str(acronym_text))
        if isinstance(acronym_values, list) and acronym_values:
            return str(acronym_values[-1])
    except Exception:
        pass
    return str(acronym_text)


def load_region_tree(cfg_path):
    region_df = pd.read_csv(cfg_path)
    region_df = region_df.reset_index(drop=True)

    nodes_by_id = {}
    for row_index, row in region_df.iterrows():
        structure_id = int(row["id"])
        structure_path = parse_structure_id_path(row["structure_id_path"])
        st_level = max(0, len(structure_path) - 1)
        parent_structure_id = structure_path[-2] if len(structure_path) >= 2 else None

        node = {
            "id": structure_id,
            "name": str(row["name"]) if pd.notna(row["name"]) else str(structure_id),
            "acronym": parse_acronym_text(row["acronym"]),
            "st_level": st_level,
            "parent_structure_id": parent_structure_id,
            "children": [],
        }
        nodes_by_id[structure_id] = node

    root_node = None
    for structure_id, node in nodes_by_id.items():
        parent_structure_id = node["parent_structure_id"]
        if parent_structure_id is None or parent_structure_id not in nodes_by_id:
            root_node = node
            continue
        nodes_by_id[parent_structure_id]["children"].append(node)

    if root_node is None and not region_df.empty:
        root_node = nodes_by_id[int(region_df.iloc[0]["id"])]

    if root_node is None:
        raise ValueError(f"Could not determine root node from CSV: {cfg_path}")

    return root_node


def get_region_label_id(node):
    value = node.get("id")
    if isinstance(value, (int, np.integer)) and value >= 0:
        return int(value)
    return None


def build_display_name(node):
    return node.get("name") or ""


def build_region_row(node, aggregated_stats):
    hemispheres = aggregated_stats.get("hemispheres")
    if hemispheres:
        row_stats = {
            "total_voxels": sum(int(stats["total_voxels"]) for stats in hemispheres.values()),
            "signal_voxels": sum(int(stats["signal_voxels"]) for stats in hemispheres.values()),
            "signal_count": sum(int(stats["signal_count"]) for stats in hemispheres.values()),
            "sum_intensity": sum(float(stats["sum_intensity"]) for stats in hemispheres.values()),
        }
    else:
        row_stats = aggregated_stats

    total_voxels = int(row_stats["total_voxels"])
    signal_voxels = int(row_stats["signal_voxels"])
    voxel_density = float(signal_voxels / total_voxels) if total_voxels > 0 else 0.0

    row = {
        "Name": build_display_name(node),
        "st_level": node.get("st_level"),
        "Total Voxels": total_voxels,
        "Signal Voxels": signal_voxels,
        "Voxel Density": voxel_density,
        "Signal Count": int(row_stats["signal_count"]),
        "Sum Intensity": float(row_stats["sum_intensity"]),
    }
    if hemispheres:
        for hemisphere_id, hemisphere_name in HEMISPHERE_NAMES.items():
            hemisphere_stats = hemispheres.get(
                hemisphere_id,
                {
                    "total_voxels": 0,
                    "signal_voxels": 0,
                    "signal_count": 0,
                    "sum_intensity": 0.0,
                },
            )
            hemisphere_total = int(hemisphere_stats["total_voxels"])
            hemisphere_signal = int(hemisphere_stats["signal_voxels"])
            row[f"{hemisphere_name} Total Voxels"] = hemisphere_total
            row[f"{hemisphere_name} Signal Voxels"] = hemisphere_signal
            row[f"{hemisphere_name} Voxel Density"] = (
                float(hemisphere_signal / hemisphere_total) if hemisphere_total > 0 else 0.0
            )
            row[f"{hemisphere_name} Signal Count"] = int(hemisphere_stats["signal_count"])
            row[f"{hemisphere_name} Sum Intensity"] = float(hemisphere_stats["sum_intensity"])
    return row


def empty_hemisphere_stats():
    return {
        int(LEFT_HEMISPHERE_ID): {
            "total_voxels": 0,
            "signal_voxels": 0,
            "signal_count": 0,
            "sum_intensity": 0.0,
        },
        int(RIGHT_HEMISPHERE_ID): {
            "total_voxels": 0,
            "signal_voxels": 0,
            "signal_count": 0,
            "sum_intensity": 0.0,
        },
    }


def flatten_region_rows(region_tree, direct_stats):
    rows = []
    hemisphere_enabled = "total_region_voxels_by_hemisphere" in direct_stats

    def visit(node):
        aggregated = {
            "total_voxels": 0,
            "signal_voxels": 0,
            "signal_count": 0,
            "sum_intensity": 0.0,
        }
        if hemisphere_enabled:
            aggregated["hemispheres"] = empty_hemisphere_stats()

        label_id = get_region_label_id(node)
        if label_id is not None and label_id > 0:
            aggregated["total_voxels"] += int(direct_stats["total_region_voxels"].get(label_id, 0))
            aggregated["signal_voxels"] += int(direct_stats["region_signal_voxels"].get(label_id, 0))
            aggregated["signal_count"] += int(direct_stats["region_signal_counts"].get(label_id, 0))
            aggregated["sum_intensity"] += float(direct_stats["region_sum_intensity"].get(label_id, 0.0))
            if hemisphere_enabled:
                for hemisphere_id in HEMISPHERE_NAMES:
                    hemisphere_key = (label_id, hemisphere_id)
                    aggregated["hemispheres"][hemisphere_id]["total_voxels"] += int(
                        direct_stats["total_region_voxels_by_hemisphere"].get(hemisphere_key, 0)
                    )
                    aggregated["hemispheres"][hemisphere_id]["signal_voxels"] += int(
                        direct_stats["region_signal_voxels_by_hemisphere"].get(hemisphere_key, 0)
                    )
                    aggregated["hemispheres"][hemisphere_id]["signal_count"] += int(
                        direct_stats["region_signal_counts_by_hemisphere"].get(hemisphere_key, 0)
                    )
                    aggregated["hemispheres"][hemisphere_id]["sum_intensity"] += float(
                        direct_stats["region_sum_intensity_by_hemisphere"].get(hemisphere_key, 0.0)
                    )

        for child in node.get("children", []):
            child_aggregated = visit(child)
            aggregated["total_voxels"] += child_aggregated["total_voxels"]
            aggregated["signal_voxels"] += child_aggregated["signal_voxels"]
            aggregated["signal_count"] += child_aggregated["signal_count"]
            aggregated["sum_intensity"] += child_aggregated["sum_intensity"]
            if hemisphere_enabled:
                for hemisphere_id in HEMISPHERE_NAMES:
                    aggregated["hemispheres"][hemisphere_id]["total_voxels"] += child_aggregated["hemispheres"][hemisphere_id]["total_voxels"]
                    aggregated["hemispheres"][hemisphere_id]["signal_voxels"] += child_aggregated["hemispheres"][hemisphere_id]["signal_voxels"]
                    aggregated["hemispheres"][hemisphere_id]["signal_count"] += child_aggregated["hemispheres"][hemisphere_id]["signal_count"]
                    aggregated["hemispheres"][hemisphere_id]["sum_intensity"] += child_aggregated["hemispheres"][hemisphere_id]["sum_intensity"]

        if label_id is not None and label_id > 0 and aggregated["total_voxels"] > 0:
            rows.append(build_region_row(node, aggregated))

        return aggregated

    visit(region_tree)
    return rows


def flush_rows_to_excel(rows, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataframe = pd.DataFrame(rows)

    if dataframe.empty:
        with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
            pd.DataFrame().to_excel(writer, index=False, sheet_name="Level_0")
        return

    dataframe = dataframe.sort_values(by=["st_level", "Name"], kind="stable").reset_index(drop=True)

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        unique_levels = [int(level) for level in sorted(dataframe["st_level"].dropna().unique().tolist())]

        for level in unique_levels:
            level_frame = dataframe[dataframe["st_level"] == level].copy().reset_index(drop=True)
            export_columns = [
                "Name",
                "Total Voxels",
                "Signal Voxels",
                "Voxel Density",
                "Signal Count",
                "Sum Intensity",
            ]
            if "Left Total Voxels" in level_frame.columns:
                export_columns += [
                    "Left Total Voxels",
                    "Left Signal Voxels",
                    "Left Voxel Density",
                    "Left Signal Count",
                    "Left Sum Intensity",
                    "Right Total Voxels",
                    "Right Signal Voxels",
                    "Right Voxel Density",
                    "Right Signal Count",
                    "Right Sum Intensity",
                ]
            export_frame = level_frame[export_columns]
            export_frame.to_excel(writer, index=False, sheet_name=f"Level_{level}")

            worksheet = writer.sheets[f"Level_{level}"]
            header_font = Font(name="Arial", size=11, bold=True)
            body_font = Font(name="Arial", size=11)
            for row in worksheet.iter_rows():
                for cell in row:
                    cell.alignment = Alignment(horizontal="left", vertical="center")
                    cell.font = header_font if cell.row == 1 else body_font

            for column in worksheet.columns:
                max_length = 0
                column_letter = column[0].column_letter
                for cell in column:
                    cell_value = "" if cell.value is None else str(cell.value)
                    if len(cell_value) > max_length:
                        max_length = len(cell_value)
                worksheet.column_dimensions[column_letter].width = min(max_length + 2, 80)


def build_binary_mask(mask_chunk, foreground_mode, foreground_label):
    if foreground_mode == "nonzero":
        return mask_chunk > 0
    return mask_chunk == foreground_label


def add_bincounts_to_dict(target, labels, weights=None):
    if labels.size == 0:
        return

    max_label = int(labels.max())
    if max_label <= 0:
        return

    counts = np.bincount(labels.astype(np.int64, copy=False), weights=weights, minlength=max_label + 1)
    nonzero_ids = np.nonzero(counts)[0]
    for region_id in nonzero_ids.tolist():
        if region_id <= 0:
            continue
        value = counts[region_id]
        if weights is None:
            target[int(region_id)] = target.get(int(region_id), 0) + int(value)
        else:
            target[int(region_id)] = target.get(int(region_id), 0.0) + float(value)


def aggregate_region_totals(total_region_voxels, total_region_voxels_by_hemisphere, label_chunk, hemisphere_chunk=None):
    positive_labels = label_chunk[label_chunk > 0]
    if positive_labels.size == 0:
        return

    add_bincounts_to_dict(total_region_voxels, positive_labels)
    if hemisphere_chunk is None or total_region_voxels_by_hemisphere is None:
        return

    valid_mask = hemisphere_chunk > 0
    if not np.any(valid_mask):
        return

    labels = label_chunk[valid_mask].astype(np.int64, copy=False)
    hemispheres = hemisphere_chunk[valid_mask].astype(np.int8, copy=False)
    for hemisphere_id in HEMISPHERE_NAMES:
        hemi_labels = labels[hemispheres == hemisphere_id]
        if hemi_labels.size == 0:
            continue
        before = {}
        add_bincounts_to_dict(before, hemi_labels)
        for region_id, count in before.items():
            pair_key = (int(region_id), int(hemisphere_id))
            total_region_voxels_by_hemisphere[pair_key] = total_region_voxels_by_hemisphere.get(pair_key, 0) + int(count)


def aggregate_signal_by_hemisphere(
    region_signal_voxels_by_hemisphere,
    region_sum_intensity_by_hemisphere,
    mask_chunk,
    label_chunk,
    signal_chunk,
    foreground_mode,
    foreground_label,
    hemisphere_chunk=None,
):
    binary_mask = build_binary_mask(mask_chunk, foreground_mode, foreground_label)
    if hemisphere_chunk is None:
        return

    valid_mask = binary_mask & (label_chunk > 0) & (hemisphere_chunk > 0)
    if not np.any(valid_mask):
        return

    labels = label_chunk[valid_mask].astype(np.int64, copy=False)
    intensities = signal_chunk[valid_mask].astype(np.float64, copy=False)
    hemispheres = hemisphere_chunk[valid_mask].astype(np.int8, copy=False)
    for hemisphere_id in HEMISPHERE_NAMES:
        hemi_mask = hemispheres == hemisphere_id
        if not np.any(hemi_mask):
            continue
        hemi_labels = labels[hemi_mask]
        hemi_intensities = intensities[hemi_mask]
        voxel_counts = {}
        intensity_sums = {}
        add_bincounts_to_dict(voxel_counts, hemi_labels)
        add_bincounts_to_dict(intensity_sums, hemi_labels, weights=hemi_intensities)
        for region_id, count in voxel_counts.items():
            pair_key = (int(region_id), int(hemisphere_id))
            region_signal_voxels_by_hemisphere[pair_key] = (
                region_signal_voxels_by_hemisphere.get(pair_key, 0) + int(count)
            )
        for region_id, intensity_sum in intensity_sums.items():
            pair_key = (int(region_id), int(hemisphere_id))
            region_sum_intensity_by_hemisphere[pair_key] = (
                region_sum_intensity_by_hemisphere.get(pair_key, 0.0) + float(intensity_sum)
            )


def compute_block_artifacts(
    mask_chunk,
    label_chunk,
    signal_chunk,
    start,
    foreground_mode,
    foreground_label,
    next_component_id,
    boundary_mask_cache,
    volume_shape,
    hemisphere_chunk=None,
):
    binary_mask = build_binary_mask(mask_chunk, foreground_mode, foreground_label)
    local_labels, num_local_components = ndimage.label(binary_mask, structure=CONNECTIVITY_3D)

    if num_local_components <= 0:
        empty_int64 = np.empty(0, dtype=np.int64)
        empty_float64 = np.empty(0, dtype=np.float64)
        return {
            "num_local_components": 0,
            "next_component_id": next_component_id,
            "component_ids": empty_int64,
            "component_sizes": empty_int64,
            "pair_component_ids": empty_int64,
            "pair_region_ids": empty_int64,
            "pair_voxel_counts": empty_int64,
            "pair_intensity_sums": empty_float64,
            "hemisphere_pair_component_ids": empty_int64,
            "hemisphere_pair_region_ids": empty_int64,
            "hemisphere_pair_hemisphere_ids": np.empty(0, dtype=np.int8),
            "hemisphere_pair_voxel_counts": empty_int64,
            "hemisphere_pair_intensity_sums": empty_float64,
            "boundary_z": empty_int64,
            "boundary_y": empty_int64,
            "boundary_x": empty_int64,
            "boundary_component_ids": empty_int64,
        }

    component_ids = np.arange(next_component_id, next_component_id + num_local_components, dtype=np.int64)
    next_component_id += num_local_components

    local_to_component = np.zeros(num_local_components + 1, dtype=np.int64)
    local_to_component[1:] = component_ids
    component_chunk = local_to_component[local_labels]
    component_sizes = np.bincount(local_labels.ravel())[1:].astype(np.int64, copy=False)

    valid_mask = (component_chunk > 0) & (label_chunk > 0)
    if np.any(valid_mask):
        valid_component_ids = component_chunk[valid_mask].astype(np.int64, copy=False)
        valid_region_ids = label_chunk[valid_mask].astype(np.int64, copy=False)
        valid_signal_values = signal_chunk[valid_mask].astype(np.float64, copy=False)

        pair_values = np.empty(valid_component_ids.size, dtype=PAIR_DTYPE)
        pair_values["component"] = valid_component_ids
        pair_values["region"] = valid_region_ids
        unique_pairs, inverse_index = np.unique(pair_values, return_inverse=True)

        pair_voxel_counts = np.zeros(unique_pairs.shape[0], dtype=np.int64)
        np.add.at(pair_voxel_counts, inverse_index, 1)

        pair_intensity_sums = np.zeros(unique_pairs.shape[0], dtype=np.float64)
        np.add.at(pair_intensity_sums, inverse_index, valid_signal_values)

        pair_component_ids = unique_pairs["component"].astype(np.int64, copy=False)
        pair_region_ids = unique_pairs["region"].astype(np.int64, copy=False)
    else:
        pair_component_ids = np.empty(0, dtype=np.int64)
        pair_region_ids = np.empty(0, dtype=np.int64)
        pair_voxel_counts = np.empty(0, dtype=np.int64)
        pair_intensity_sums = np.empty(0, dtype=np.float64)

    if hemisphere_chunk is not None:
        hemisphere_valid_mask = valid_mask & (hemisphere_chunk > 0)
    else:
        hemisphere_valid_mask = np.zeros(component_chunk.shape, dtype=bool)

    if np.any(hemisphere_valid_mask):
        valid_component_ids = component_chunk[hemisphere_valid_mask].astype(np.int64, copy=False)
        valid_region_ids = label_chunk[hemisphere_valid_mask].astype(np.int64, copy=False)
        valid_hemisphere_ids = hemisphere_chunk[hemisphere_valid_mask].astype(np.int8, copy=False)
        valid_signal_values = signal_chunk[hemisphere_valid_mask].astype(np.float64, copy=False)

        hemisphere_pair_values = np.empty(valid_component_ids.size, dtype=PAIR_HEMISPHERE_DTYPE)
        hemisphere_pair_values["component"] = valid_component_ids
        hemisphere_pair_values["region"] = valid_region_ids
        hemisphere_pair_values["hemisphere"] = valid_hemisphere_ids
        unique_hemisphere_pairs, hemisphere_inverse_index = np.unique(
            hemisphere_pair_values,
            return_inverse=True,
        )

        hemisphere_pair_voxel_counts = np.zeros(unique_hemisphere_pairs.shape[0], dtype=np.int64)
        np.add.at(hemisphere_pair_voxel_counts, hemisphere_inverse_index, 1)

        hemisphere_pair_intensity_sums = np.zeros(unique_hemisphere_pairs.shape[0], dtype=np.float64)
        np.add.at(hemisphere_pair_intensity_sums, hemisphere_inverse_index, valid_signal_values)

        hemisphere_pair_component_ids = unique_hemisphere_pairs["component"].astype(np.int64, copy=False)
        hemisphere_pair_region_ids = unique_hemisphere_pairs["region"].astype(np.int64, copy=False)
        hemisphere_pair_hemisphere_ids = unique_hemisphere_pairs["hemisphere"].astype(np.int8, copy=False)
    else:
        hemisphere_pair_component_ids = np.empty(0, dtype=np.int64)
        hemisphere_pair_region_ids = np.empty(0, dtype=np.int64)
        hemisphere_pair_hemisphere_ids = np.empty(0, dtype=np.int8)
        hemisphere_pair_voxel_counts = np.empty(0, dtype=np.int64)
        hemisphere_pair_intensity_sums = np.empty(0, dtype=np.float64)

    block_shape = tuple(int(size) for size in mask_chunk.shape)
    boundary_mask = boundary_mask_cache.setdefault(block_shape, build_boundary_mask(block_shape))
    boundary_foreground_mask = boundary_mask & (component_chunk > 0)

    if np.any(boundary_foreground_mask):
        boundary_z, boundary_y, boundary_x = np.nonzero(boundary_foreground_mask)
        boundary_z = boundary_z.astype(np.int64, copy=False) + int(start[0])
        boundary_y = boundary_y.astype(np.int64, copy=False) + int(start[1])
        boundary_x = boundary_x.astype(np.int64, copy=False) + int(start[2])
        boundary_component_ids = component_chunk[boundary_foreground_mask].astype(np.int64, copy=False)
    else:
        boundary_z = np.empty(0, dtype=np.int64)
        boundary_y = np.empty(0, dtype=np.int64)
        boundary_x = np.empty(0, dtype=np.int64)
        boundary_component_ids = np.empty(0, dtype=np.int64)

    return {
        "num_local_components": int(num_local_components),
        "next_component_id": int(next_component_id),
        "component_ids": component_ids,
        "component_sizes": component_sizes,
        "pair_component_ids": pair_component_ids,
        "pair_region_ids": pair_region_ids,
        "pair_voxel_counts": pair_voxel_counts,
        "pair_intensity_sums": pair_intensity_sums,
        "hemisphere_pair_component_ids": hemisphere_pair_component_ids,
        "hemisphere_pair_region_ids": hemisphere_pair_region_ids,
        "hemisphere_pair_hemisphere_ids": hemisphere_pair_hemisphere_ids,
        "hemisphere_pair_voxel_counts": hemisphere_pair_voxel_counts,
        "hemisphere_pair_intensity_sums": hemisphere_pair_intensity_sums,
        "boundary_z": boundary_z,
        "boundary_y": boundary_y,
        "boundary_x": boundary_x,
        "boundary_component_ids": boundary_component_ids,
    }


def save_block_artifact(artifact_path, artifact):
    np.savez(
        artifact_path,
        component_ids=artifact["component_ids"],
        component_sizes=artifact["component_sizes"],
        pair_component_ids=artifact["pair_component_ids"],
        pair_region_ids=artifact["pair_region_ids"],
        pair_voxel_counts=artifact["pair_voxel_counts"],
        pair_intensity_sums=artifact["pair_intensity_sums"],
        hemisphere_pair_component_ids=artifact["hemisphere_pair_component_ids"],
        hemisphere_pair_region_ids=artifact["hemisphere_pair_region_ids"],
        hemisphere_pair_hemisphere_ids=artifact["hemisphere_pair_hemisphere_ids"],
        hemisphere_pair_voxel_counts=artifact["hemisphere_pair_voxel_counts"],
        hemisphere_pair_intensity_sums=artifact["hemisphere_pair_intensity_sums"],
        boundary_z=artifact["boundary_z"],
        boundary_y=artifact["boundary_y"],
        boundary_x=artifact["boundary_x"],
        boundary_component_ids=artifact["boundary_component_ids"],
    )


def load_block_artifact_mutable(artifact_path):
    with np.load(artifact_path, allow_pickle=False) as data:
        return {key: np.array(data[key], copy=True) for key in data.files}


WORKER_MASK_ZARR = None
WORKER_LABEL_ZARR = None
WORKER_SIGNAL_ZARR = None
WORKER_DATASET_NAME = None
WORKER_HEMISPHERE_ZARR = None


def init_pass1_worker(mask_zarr_path, label_zarr_path, signal_zarr_path, hemisphere_zarr_path, dataset_name):
    global WORKER_MASK_ZARR, WORKER_LABEL_ZARR, WORKER_SIGNAL_ZARR, WORKER_HEMISPHERE_ZARR, WORKER_DATASET_NAME
    WORKER_DATASET_NAME = dataset_name
    WORKER_MASK_ZARR = open_zarr_dataset(mask_zarr_path, dataset_name)
    WORKER_LABEL_ZARR = open_zarr_dataset(label_zarr_path, dataset_name)
    WORKER_SIGNAL_ZARR = open_zarr_dataset(signal_zarr_path, dataset_name)
    WORKER_HEMISPHERE_ZARR = None
    if str(hemisphere_zarr_path).strip():
        WORKER_HEMISPHERE_ZARR = open_zarr_dataset(hemisphere_zarr_path, dataset_name)


def process_block_spec_worker(block_spec, foreground_mode, foreground_label):
    z0, y0, x0 = block_spec["start"]
    z1, y1, x1 = block_spec["stop"]

    mask_chunk = np.asarray(WORKER_MASK_ZARR[z0:z1, y0:y1, x0:x1])
    label_chunk = np.asarray(WORKER_LABEL_ZARR[z0:z1, y0:y1, x0:x1])
    signal_chunk = np.asarray(WORKER_SIGNAL_ZARR[z0:z1, y0:y1, x0:x1])
    hemisphere_chunk = None
    if WORKER_HEMISPHERE_ZARR is not None:
        hemisphere_chunk = np.asarray(WORKER_HEMISPHERE_ZARR[z0:z1, y0:y1, x0:x1])

    total_region_voxels = {}
    total_region_voxels_by_hemisphere = {}
    region_signal_voxels_by_hemisphere = {}
    region_sum_intensity_by_hemisphere = {}
    aggregate_region_totals(
        total_region_voxels,
        total_region_voxels_by_hemisphere,
        label_chunk,
        hemisphere_chunk,
    )
    aggregate_signal_by_hemisphere(
        region_signal_voxels_by_hemisphere,
        region_sum_intensity_by_hemisphere,
        mask_chunk,
        label_chunk,
        signal_chunk,
        foreground_mode,
        foreground_label,
        hemisphere_chunk,
    )

    artifact = compute_block_artifacts(
        mask_chunk=mask_chunk,
        label_chunk=label_chunk,
        signal_chunk=signal_chunk,
        start=block_spec["start"],
        foreground_mode=foreground_mode,
        foreground_label=foreground_label,
        next_component_id=1,
        boundary_mask_cache={},
        volume_shape=WORKER_MASK_ZARR.shape,
        hemisphere_chunk=hemisphere_chunk,
    )

    return {
        "block_spec": block_spec,
        "artifact": artifact,
        "total_region_voxels": total_region_voxels,
        "total_region_voxels_by_hemisphere": total_region_voxels_by_hemisphere,
        "region_signal_voxels_by_hemisphere": region_signal_voxels_by_hemisphere,
        "region_sum_intensity_by_hemisphere": region_sum_intensity_by_hemisphere,
    }


def merge_region_totals(target_totals, source_totals):
    for region_id, count in source_totals.items():
        target_totals[region_id] = target_totals.get(region_id, 0) + int(count)


def merge_pair_totals(target_totals, source_totals):
    for pair_key, count in source_totals.items():
        normalized_key = (int(pair_key[0]), int(pair_key[1]))
        target_totals[normalized_key] = target_totals.get(normalized_key, 0) + int(count)


def apply_component_offset(artifact, component_offset):
    if component_offset == 0:
        return artifact

    remapped = dict(artifact)
    for key in (
        "component_ids",
        "pair_component_ids",
        "hemisphere_pair_component_ids",
        "boundary_component_ids",
    ):
        values = remapped[key]
        if values.size > 0:
            remapped[key] = values + np.int64(component_offset)
    return remapped


def scan_blocks_and_write_artifacts(
    mask_zarr,
    label_zarr,
    signal_zarr,
    hemisphere_zarr,
    mask_zarr_path,
    label_zarr_path,
    signal_zarr_path,
    hemisphere_zarr_path,
    dataset_name,
    block_shape,
    foreground_mode,
    foreground_label,
    tmp_dir,
    pass1_workers,
):
    logger.info("Pass 1/3: scanning blocks and writing block artifacts...")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    blocks_dir = tmp_dir / "blocks"
    blocks_dir.mkdir(parents=True, exist_ok=True)

    total_region_voxels = {}
    total_region_voxels_by_hemisphere = {}
    region_signal_voxels_by_hemisphere = {}
    region_sum_intensity_by_hemisphere = {}
    manifest = []
    next_component_id = 1
    use_hemisphere_label = hemisphere_zarr is not None

    all_block_specs = list(iter_block_specs(mask_zarr.shape, block_shape))
    with tqdm(total=len(all_block_specs), desc="Block scan", unit="block") as progress_bar:
        if int(pass1_workers) <= 1:
            boundary_mask_cache = {}
            for block_spec in all_block_specs:
                z0, y0, x0 = block_spec["start"]
                z1, y1, x1 = block_spec["stop"]

                mask_chunk = np.asarray(mask_zarr[z0:z1, y0:y1, x0:x1])
                label_chunk = np.asarray(label_zarr[z0:z1, y0:y1, x0:x1])
                signal_chunk = np.asarray(signal_zarr[z0:z1, y0:y1, x0:x1])
                hemisphere_chunk = None
                if use_hemisphere_label:
                    hemisphere_chunk = np.asarray(hemisphere_zarr[z0:z1, y0:y1, x0:x1])

                aggregate_region_totals(
                    total_region_voxels,
                    total_region_voxels_by_hemisphere,
                    label_chunk,
                    hemisphere_chunk,
                )
                aggregate_signal_by_hemisphere(
                    region_signal_voxels_by_hemisphere,
                    region_sum_intensity_by_hemisphere,
                    mask_chunk,
                    label_chunk,
                    signal_chunk,
                    foreground_mode,
                    foreground_label,
                    hemisphere_chunk,
                )

                artifact = compute_block_artifacts(
                    mask_chunk=mask_chunk,
                    label_chunk=label_chunk,
                    signal_chunk=signal_chunk,
                    start=block_spec["start"],
                    foreground_mode=foreground_mode,
                    foreground_label=foreground_label,
                    next_component_id=next_component_id,
                    boundary_mask_cache=boundary_mask_cache,
                    volume_shape=mask_zarr.shape,
                    hemisphere_chunk=hemisphere_chunk,
                )
                next_component_id = artifact["next_component_id"]

                artifact_path = blocks_dir / (
                    f"block_z{block_spec['grid_index'][0]:04d}_"
                    f"y{block_spec['grid_index'][1]:04d}_"
                    f"x{block_spec['grid_index'][2]:04d}.npz"
                )
                save_block_artifact(artifact_path, artifact)

                manifest.append(
                    {
                        "block_id": int(block_spec["block_id"]),
                        "grid_index": list(block_spec["grid_index"]),
                        "start": list(block_spec["start"]),
                        "stop": list(block_spec["stop"]),
                        "artifact_path": str(artifact_path),
                        "component_count": int(artifact["num_local_components"]),
                        "boundary_count": int(artifact["boundary_component_ids"].size),
                    }
                )
                progress_bar.update(1)
                progress_bar.set_postfix(
                    components=artifact["num_local_components"],
                    boundary_voxels=artifact["boundary_component_ids"].size,
                    refresh=False,
                )
        else:
            max_in_flight = max(int(pass1_workers) * 2, 1)
            logger.info("Pass 1 parallel workers: %s, in-flight tasks: %s", pass1_workers, max_in_flight)
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=int(pass1_workers),
                initializer=init_pass1_worker,
                initargs=(
                    mask_zarr_path,
                    label_zarr_path,
                    signal_zarr_path,
                    hemisphere_zarr_path,
                    dataset_name,
                ),
            ) as executor:
                future_to_block = {}
                spec_iter = iter(all_block_specs)

                while True:
                    while len(future_to_block) < max_in_flight:
                        try:
                            block_spec = next(spec_iter)
                        except StopIteration:
                            break
                        future = executor.submit(
                            process_block_spec_worker,
                            block_spec,
                            foreground_mode,
                            foreground_label,
                        )
                        future_to_block[future] = block_spec

                    if not future_to_block:
                        break

                    done, _ = concurrent.futures.wait(
                        future_to_block,
                        return_when=concurrent.futures.FIRST_COMPLETED,
                    )
                    for future in done:
                        block_spec = future_to_block.pop(future)
                        result = future.result()
                        merge_region_totals(total_region_voxels, result["total_region_voxels"])
                        merge_pair_totals(
                            total_region_voxels_by_hemisphere,
                            result["total_region_voxels_by_hemisphere"],
                        )
                        merge_pair_totals(
                            region_signal_voxels_by_hemisphere,
                            result["region_signal_voxels_by_hemisphere"],
                        )
                        for pair_key, intensity_sum in result["region_sum_intensity_by_hemisphere"].items():
                            normalized_key = (int(pair_key[0]), int(pair_key[1]))
                            region_sum_intensity_by_hemisphere[normalized_key] = (
                                region_sum_intensity_by_hemisphere.get(normalized_key, 0.0) + float(intensity_sum)
                            )

                        local_artifact = result["artifact"]
                        component_offset = next_component_id - 1
                        artifact = apply_component_offset(local_artifact, component_offset)
                        next_component_id += int(local_artifact["num_local_components"])
                        artifact["next_component_id"] = int(next_component_id)

                        artifact_path = blocks_dir / (
                            f"block_z{block_spec['grid_index'][0]:04d}_"
                            f"y{block_spec['grid_index'][1]:04d}_"
                            f"x{block_spec['grid_index'][2]:04d}.npz"
                        )
                        save_block_artifact(artifact_path, artifact)

                        manifest.append(
                            {
                                "block_id": int(block_spec["block_id"]),
                                "grid_index": list(block_spec["grid_index"]),
                                "start": list(block_spec["start"]),
                                "stop": list(block_spec["stop"]),
                                "artifact_path": str(artifact_path),
                                "component_count": int(local_artifact["num_local_components"]),
                                "boundary_count": int(local_artifact["boundary_component_ids"].size),
                            }
                        )
                        progress_bar.update(1)
                        progress_bar.set_postfix(
                            components=local_artifact["num_local_components"],
                            boundary_voxels=local_artifact["boundary_component_ids"].size,
                            refresh=False,
                        )

    manifest.sort(key=lambda block: block["block_id"])

    manifest_payload = {
        "volume_shape": list(mask_zarr.shape),
        "block_shape": list(block_shape),
        "total_components": int(next_component_id - 1),
        "blocks": manifest,
        "total_region_voxels": {str(key): int(value) for key, value in total_region_voxels.items()},
        "total_region_voxels_by_hemisphere": {
            f"{int(key[0])}:{int(key[1])}": int(value)
            for key, value in total_region_voxels_by_hemisphere.items()
        },
        "region_signal_voxels_by_hemisphere": {
            f"{int(key[0])}:{int(key[1])}": int(value)
            for key, value in region_signal_voxels_by_hemisphere.items()
        },
        "region_sum_intensity_by_hemisphere": {
            f"{int(key[0])}:{int(key[1])}": float(value)
            for key, value in region_sum_intensity_by_hemisphere.items()
        },
    }

    with open(tmp_dir / "manifest.json", "w", encoding="utf-8") as handle:
        json.dump(manifest_payload, handle, indent=2)

    return manifest_payload


def load_block_arrays(artifact_path):
    with np.load(artifact_path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def find_root(parent, node_id):
    while parent[node_id] != node_id:
        parent[node_id] = parent[parent[node_id]]
        node_id = parent[node_id]
    return node_id


def union_roots(parent, rank, node_a, node_b):
    root_a = find_root(parent, int(node_a))
    root_b = find_root(parent, int(node_b))
    if root_a == root_b:
        return

    if rank[root_a] < rank[root_b]:
        root_a, root_b = root_b, root_a

    parent[root_b] = root_a
    if rank[root_a] == rank[root_b]:
        rank[root_a] += 1


def build_shift_values(block_delta):
    shift_options = []
    for axis_delta in block_delta:
        if axis_delta > 0:
            shift_options.append((-1,))
        elif axis_delta < 0:
            shift_options.append((1,))
        else:
            shift_options.append((-1, 0, 1))
    return list(product(*shift_options))


def compare_neighbor_boundaries(arrays_a, arrays_b, block_delta, volume_shape, parent, rank):
    if arrays_a["boundary_component_ids"].size == 0 or arrays_b["boundary_component_ids"].size == 0:
        return 0

    linear_a = np.ravel_multi_index(
        (arrays_a["boundary_z"], arrays_a["boundary_y"], arrays_a["boundary_x"]),
        dims=volume_shape,
    )
    ids_a = arrays_a["boundary_component_ids"]

    matches = 0
    for shift_z, shift_y, shift_x in build_shift_values(block_delta):
        shifted_z = arrays_b["boundary_z"] + shift_z
        shifted_y = arrays_b["boundary_y"] + shift_y
        shifted_x = arrays_b["boundary_x"] + shift_x

        valid_mask = (
            (shifted_z >= 0)
            & (shifted_z < volume_shape[0])
            & (shifted_y >= 0)
            & (shifted_y < volume_shape[1])
            & (shifted_x >= 0)
            & (shifted_x < volume_shape[2])
        )
        if not np.any(valid_mask):
            continue

        valid_indices = np.nonzero(valid_mask)[0]
        shifted_linear_b = np.ravel_multi_index(
            (shifted_z[valid_mask], shifted_y[valid_mask], shifted_x[valid_mask]),
            dims=volume_shape,
        )

        _, index_a, index_b = np.intersect1d(
            linear_a,
            shifted_linear_b,
            assume_unique=True,
            return_indices=True,
        )
        if index_a.size == 0:
            continue

        matched_indices_b = valid_indices[index_b]
        for component_a, component_b in zip(ids_a[index_a].tolist(), arrays_b["boundary_component_ids"][matched_indices_b].tolist()):
            union_roots(parent, rank, component_a, component_b)
            matches += 1

    return matches


def stitch_block_boundaries(manifest_payload):
    logger.info("Pass 2/3: stitching neighboring blocks through block boundaries...")
    volume_shape = tuple(int(value) for value in manifest_payload["volume_shape"])
    total_components = int(manifest_payload["total_components"])
    block_lookup = {tuple(block["grid_index"]): block for block in manifest_payload["blocks"]}

    parent = np.arange(total_components + 1, dtype=np.int64)
    rank = np.zeros(total_components + 1, dtype=np.uint8)
    compared_pairs = 0
    matched_voxels = 0

    with tqdm(total=len(manifest_payload["blocks"]), desc="Boundary stitch", unit="block") as progress_bar:
        for block in manifest_payload["blocks"]:
            grid_index = tuple(block["grid_index"])
            arrays_a = None

            for delta in FORWARD_BLOCK_OFFSETS:
                neighbor_index = (
                    grid_index[0] + delta[0],
                    grid_index[1] + delta[1],
                    grid_index[2] + delta[2],
                )
                neighbor_block = block_lookup.get(neighbor_index)
                if neighbor_block is None:
                    continue

                if arrays_a is None:
                    arrays_a = load_block_arrays(block["artifact_path"])
                arrays_b = load_block_arrays(neighbor_block["artifact_path"])
                matched_voxels += compare_neighbor_boundaries(
                    arrays_a=arrays_a,
                    arrays_b=arrays_b,
                    block_delta=delta,
                    volume_shape=volume_shape,
                    parent=parent,
                    rank=rank,
                )
                compared_pairs += 1

            progress_bar.update(1)
            progress_bar.set_postfix(
                compared_pairs=compared_pairs,
                matched_voxels=matched_voxels,
                refresh=False,
            )

    logger.info("Compressing union-find roots...")
    while True:
        compressed_parent = parent[parent]
        if np.array_equal(compressed_parent, parent):
            break
        parent[:] = compressed_parent

    return parent


def build_root_sizes(manifest_payload, parent):
    logger.info("Pass 3/3a: aggregating merged component sizes...")
    root_sizes = np.zeros(parent.shape[0], dtype=np.int64)

    with tqdm(total=len(manifest_payload["blocks"]), desc="Root sizes", unit="block") as progress_bar:
        for block in manifest_payload["blocks"]:
            arrays = load_block_arrays(block["artifact_path"])
            if arrays["component_ids"].size > 0:
                root_ids = parent[arrays["component_ids"]]
                np.add.at(root_sizes, root_ids, arrays["component_sizes"])
            progress_bar.update(1)

    return root_sizes


def choose_majority_hemisphere(left_voxels, right_voxels):
    if left_voxels <= 0 and right_voxels <= 0:
        raise ValueError("Cannot choose a majority hemisphere from zero voxels.")
    if left_voxels >= right_voxels:
        return int(LEFT_HEMISPHERE_ID)
    return int(RIGHT_HEMISPHERE_ID)


def choose_majority_region(region_voxels):
    if not region_voxels:
        raise ValueError("Cannot choose a majority region from zero voxels.")
    best_voxels = max(region_voxels.values())
    tied_regions = [int(region_id) for region_id, count in region_voxels.items() if count == best_voxels]
    return min(tied_regions)


def collapse_component_stats_by_majority(
    *,
    region_pair_voxels=None,
    region_pair_intensity=None,
    region_hemisphere_pair_voxels=None,
    region_hemisphere_pair_intensity=None,
    assign_hemisphere=False,
):
    """Assign each connected object to one brain region and optionally one hemisphere."""
    component_regions = {}
    component_hemispheres = {}
    component_intensity = {}

    if assign_hemisphere and region_hemisphere_pair_voxels:
        for (root_id, region_id, hemisphere_id), voxel_count in region_hemisphere_pair_voxels.items():
            root_key = int(root_id)
            region_map = component_regions.setdefault(root_key, {})
            region_map[int(region_id)] = region_map.get(int(region_id), 0) + int(voxel_count)
            hemi_map = component_hemispheres.setdefault(root_key, {})
            hemi_map[int(hemisphere_id)] = hemi_map.get(int(hemisphere_id), 0) + int(voxel_count)
        for (root_id, _region_id, _hemisphere_id), intensity_sum in region_hemisphere_pair_intensity.items():
            root_key = int(root_id)
            component_intensity[root_key] = component_intensity.get(root_key, 0.0) + float(intensity_sum)
    else:
        for (root_id, region_id), voxel_count in (region_pair_voxels or {}).items():
            root_key = int(root_id)
            region_map = component_regions.setdefault(root_key, {})
            region_map[int(region_id)] = region_map.get(int(region_id), 0) + int(voxel_count)
        for (root_id, _region_id), intensity_sum in (region_pair_intensity or {}).items():
            root_key = int(root_id)
            component_intensity[root_key] = component_intensity.get(root_key, 0.0) + float(intensity_sum)

    region_signal_voxels = {}
    region_signal_counts = {}
    region_sum_intensity = {}
    region_signal_voxels_by_hemisphere = {}
    region_signal_counts_by_hemisphere = {}
    region_sum_intensity_by_hemisphere = {}

    for root_id, region_map in component_regions.items():
        chosen_region = choose_majority_region(region_map)
        total_voxels = sum(int(count) for count in region_map.values())
        total_intensity = float(component_intensity.get(root_id, 0.0))

        region_signal_voxels[chosen_region] = region_signal_voxels.get(chosen_region, 0) + total_voxels
        region_signal_counts[chosen_region] = region_signal_counts.get(chosen_region, 0) + 1
        region_sum_intensity[chosen_region] = region_sum_intensity.get(chosen_region, 0.0) + total_intensity

        if assign_hemisphere:
            hemi_map = component_hemispheres.get(root_id, {})
            left_voxels = int(hemi_map.get(int(LEFT_HEMISPHERE_ID), 0))
            right_voxels = int(hemi_map.get(int(RIGHT_HEMISPHERE_ID), 0))
            chosen_hemisphere = choose_majority_hemisphere(left_voxels, right_voxels)
            pair_key = (chosen_region, chosen_hemisphere)
            region_signal_voxels_by_hemisphere[pair_key] = (
                region_signal_voxels_by_hemisphere.get(pair_key, 0) + total_voxels
            )
            region_signal_counts_by_hemisphere[pair_key] = (
                region_signal_counts_by_hemisphere.get(pair_key, 0) + 1
            )
            region_sum_intensity_by_hemisphere[pair_key] = (
                region_sum_intensity_by_hemisphere.get(pair_key, 0.0) + total_intensity
            )

    result = {
        "region_signal_voxels": region_signal_voxels,
        "region_signal_counts": region_signal_counts,
        "region_sum_intensity": region_sum_intensity,
    }
    if assign_hemisphere:
        result["region_signal_voxels_by_hemisphere"] = region_signal_voxels_by_hemisphere
        result["region_signal_counts_by_hemisphere"] = region_signal_counts_by_hemisphere
        result["region_sum_intensity_by_hemisphere"] = region_sum_intensity_by_hemisphere
    return result


def aggregate_final_region_stats(manifest_payload, parent, root_sizes, min_voxels):
    logger.info("Pass 3/3b: collapsing merged objects into per-region statistics...")
    kept_root_mask = root_sizes >= int(min_voxels)
    kept_root_mask[0] = False

    region_pair_voxels = {}
    region_pair_intensity = {}
    region_hemisphere_pair_voxels = {}
    region_hemisphere_pair_intensity = {}
    with tqdm(total=len(manifest_payload["blocks"]), desc="Region collapse", unit="block") as progress_bar:
        for block in manifest_payload["blocks"]:
            arrays = load_block_arrays(block["artifact_path"])
            if arrays["pair_component_ids"].size == 0:
                progress_bar.update(1)
                continue

            root_ids = parent[arrays["pair_component_ids"]]
            keep_mask = kept_root_mask[root_ids]
            if not np.any(keep_mask):
                progress_bar.update(1)
                continue

            kept_roots = root_ids[keep_mask].astype(np.int64, copy=False)
            kept_regions = arrays["pair_region_ids"][keep_mask].astype(np.int64, copy=False)
            kept_counts = arrays["pair_voxel_counts"][keep_mask].astype(np.int64, copy=False)
            kept_intensity = arrays["pair_intensity_sums"][keep_mask].astype(np.float64, copy=False)

            pair_values = np.empty(kept_roots.size, dtype=ROOT_REGION_DTYPE)
            pair_values["root"] = kept_roots
            pair_values["region"] = kept_regions
            unique_pairs, inverse_index = np.unique(pair_values, return_inverse=True)

            merged_counts = np.zeros(unique_pairs.shape[0], dtype=np.int64)
            np.add.at(merged_counts, inverse_index, kept_counts)

            merged_intensity = np.zeros(unique_pairs.shape[0], dtype=np.float64)
            np.add.at(merged_intensity, inverse_index, kept_intensity)

            for pair, voxel_count, intensity_sum in zip(unique_pairs.tolist(), merged_counts.tolist(), merged_intensity.tolist()):
                pair_key = (int(pair[0]), int(pair[1]))
                region_pair_voxels[pair_key] = region_pair_voxels.get(pair_key, 0) + int(voxel_count)
                region_pair_intensity[pair_key] = region_pair_intensity.get(pair_key, 0.0) + float(intensity_sum)

            if arrays.get("hemisphere_pair_component_ids", np.empty(0, dtype=np.int64)).size > 0:
                hemisphere_root_ids = parent[arrays["hemisphere_pair_component_ids"]]
                hemisphere_keep_mask = kept_root_mask[hemisphere_root_ids]
                if np.any(hemisphere_keep_mask):
                    kept_hemisphere_roots = hemisphere_root_ids[hemisphere_keep_mask].astype(np.int64, copy=False)
                    kept_hemisphere_regions = arrays["hemisphere_pair_region_ids"][hemisphere_keep_mask].astype(np.int64, copy=False)
                    kept_hemisphere_ids = arrays["hemisphere_pair_hemisphere_ids"][hemisphere_keep_mask].astype(np.int8, copy=False)
                    kept_hemisphere_counts = arrays["hemisphere_pair_voxel_counts"][hemisphere_keep_mask].astype(np.int64, copy=False)
                    kept_hemisphere_intensity = arrays["hemisphere_pair_intensity_sums"][hemisphere_keep_mask].astype(np.float64, copy=False)

                    hemisphere_pair_values = np.empty(
                        kept_hemisphere_roots.size,
                        dtype=ROOT_REGION_HEMISPHERE_DTYPE,
                    )
                    hemisphere_pair_values["root"] = kept_hemisphere_roots
                    hemisphere_pair_values["region"] = kept_hemisphere_regions
                    hemisphere_pair_values["hemisphere"] = kept_hemisphere_ids
                    unique_hemisphere_pairs, hemisphere_inverse_index = np.unique(
                        hemisphere_pair_values,
                        return_inverse=True,
                    )

                    merged_hemisphere_counts = np.zeros(unique_hemisphere_pairs.shape[0], dtype=np.int64)
                    np.add.at(merged_hemisphere_counts, hemisphere_inverse_index, kept_hemisphere_counts)

                    merged_hemisphere_intensity = np.zeros(unique_hemisphere_pairs.shape[0], dtype=np.float64)
                    np.add.at(
                        merged_hemisphere_intensity,
                        hemisphere_inverse_index,
                        kept_hemisphere_intensity,
                    )

                    for pair, voxel_count, intensity_sum in zip(
                        unique_hemisphere_pairs.tolist(),
                        merged_hemisphere_counts.tolist(),
                        merged_hemisphere_intensity.tolist(),
                    ):
                        pair_key = (int(pair[0]), int(pair[1]), int(pair[2]))
                        region_hemisphere_pair_voxels[pair_key] = (
                            region_hemisphere_pair_voxels.get(pair_key, 0) + int(voxel_count)
                        )
                        region_hemisphere_pair_intensity[pair_key] = (
                            region_hemisphere_pair_intensity.get(pair_key, 0.0) + float(intensity_sum)
                        )

            progress_bar.update(1)

    assign_hemisphere = bool(region_hemisphere_pair_voxels)
    if assign_hemisphere:
        collapsed_stats = collapse_component_stats_by_majority(
            region_hemisphere_pair_voxels=region_hemisphere_pair_voxels,
            region_hemisphere_pair_intensity=region_hemisphere_pair_intensity,
            assign_hemisphere=True,
        )
    else:
        collapsed_stats = collapse_component_stats_by_majority(
            region_pair_voxels=region_pair_voxels,
            region_pair_intensity=region_pair_intensity,
            assign_hemisphere=False,
        )

    region_signal_voxels = collapsed_stats["region_signal_voxels"]
    region_signal_counts = collapsed_stats["region_signal_counts"]
    region_sum_intensity = collapsed_stats["region_sum_intensity"]
    region_signal_voxels_by_hemisphere = collapsed_stats.get("region_signal_voxels_by_hemisphere", {})
    region_signal_counts_by_hemisphere = collapsed_stats.get("region_signal_counts_by_hemisphere", {})
    region_sum_intensity_by_hemisphere = collapsed_stats.get("region_sum_intensity_by_hemisphere", {})

    kept_components = int(np.count_nonzero((parent == np.arange(parent.shape[0], dtype=np.int64)) & kept_root_mask))
    kept_voxels = int(root_sizes[kept_root_mask].sum(dtype=np.int64))
    logger.info(
        "Kept %d connected components with >= %d voxels, covering %d voxels",
        kept_components, min_voxels, kept_voxels,
    )

    total_region_voxels = {
        int(key): int(value) for key, value in manifest_payload["total_region_voxels"].items()
    }
    if "total_region_voxels_by_hemisphere" in manifest_payload:
        total_region_voxels_by_hemisphere = {}
        for key, value in manifest_payload.get("total_region_voxels_by_hemisphere", {}).items():
            region_id, hemisphere_id = str(key).split(":", 1)
            total_region_voxels_by_hemisphere[(int(region_id), int(hemisphere_id))] = int(value)
    else:
        total_region_voxels_by_hemisphere = {}
        region_signal_voxels_by_hemisphere = {}
        region_signal_counts_by_hemisphere = {}
        region_sum_intensity_by_hemisphere = {}

    return {
        "total_region_voxels": total_region_voxels,
        "region_signal_voxels": region_signal_voxels,
        "region_signal_counts": region_signal_counts,
        "region_sum_intensity": region_sum_intensity,
        **({"total_region_voxels_by_hemisphere": total_region_voxels_by_hemisphere} if total_region_voxels_by_hemisphere else {}),
        **({"region_signal_voxels_by_hemisphere": region_signal_voxels_by_hemisphere} if region_signal_voxels_by_hemisphere else {}),
        **({"region_signal_counts_by_hemisphere": region_signal_counts_by_hemisphere} if region_signal_counts_by_hemisphere else {}),
        **({"region_sum_intensity_by_hemisphere": region_sum_intensity_by_hemisphere} if region_sum_intensity_by_hemisphere else {}),
    }


def export_region_excel(region_tree, direct_stats, output_path, flush_every):
    rows = []
    all_rows = flatten_region_rows(region_tree, direct_stats)
    for index, row in enumerate(all_rows, start=1):
        rows.append(row)
        if flush_every > 0 and index % flush_every == 0:
            logger.info("Flushing %d rows to %s", len(rows), output_path)
            flush_rows_to_excel(rows, output_path)

    logger.info("Final flush with %d rows to %s", len(rows), output_path)
    flush_rows_to_excel(rows, output_path)


def analyze_zarr_graph(
    mask_zarr_path,
    label_zarr_path,
    signal_zarr_path,
    cfg_path,
    output_path,
    dataset_name,
    block_size,
    foreground_mode,
    foreground_label,
    min_voxels,
    flush_every,
    resolution_xyz,
    tmp_dir,
    keep_tmp,
    pass1_workers,
    hemisphere_zarr_path="",
):
    _ = resolution_xyz
    mask_zarr = open_zarr_dataset(mask_zarr_path, dataset_name)
    label_zarr = open_zarr_dataset(label_zarr_path, dataset_name)
    signal_zarr = open_zarr_dataset(signal_zarr_path, dataset_name)
    hemisphere_zarr = None
    if str(hemisphere_zarr_path).strip():
        hemisphere_zarr = open_zarr_dataset(hemisphere_zarr_path, dataset_name)
        if hemisphere_zarr.shape != mask_zarr.shape:
            raise ValueError(
                f"Hemisphere Zarr shape mismatch: hemisphere={hemisphere_zarr.shape}, mask={mask_zarr.shape}"
            )
    validate_zarr_inputs(mask_zarr, label_zarr, signal_zarr)

    block_shape = choose_block_shape(mask_zarr, label_zarr, signal_zarr, block_size)
    region_tree = load_region_tree(cfg_path)
    if hemisphere_zarr is None:
        logger.info("Hemisphere analysis disabled; only total region statistics will be computed.")
    else:
        logger.info("Hemisphere analysis enabled via hemisphere label Zarr: %s", hemisphere_zarr_path)

    tmp_root = Path(tmp_dir) if str(tmp_dir).strip() else Path(f"{output_path}_zarr_graph_tmp")
    if tmp_root.exists():
        logger.info("Cleaning existing temporary directory: %s", tmp_root)
        shutil.rmtree(tmp_root)

    total_start_time = time.perf_counter()
    try:
        pass1_start_time = time.perf_counter()
        manifest_payload = scan_blocks_and_write_artifacts(
            mask_zarr=mask_zarr,
            label_zarr=label_zarr,
            signal_zarr=signal_zarr,
            hemisphere_zarr=hemisphere_zarr,
            mask_zarr_path=mask_zarr_path,
            label_zarr_path=label_zarr_path,
            signal_zarr_path=signal_zarr_path,
            hemisphere_zarr_path=hemisphere_zarr_path,
            dataset_name=dataset_name,
            block_shape=block_shape,
            foreground_mode=foreground_mode,
            foreground_label=foreground_label,
            tmp_dir=tmp_root,
            pass1_workers=pass1_workers,
        )
        logger.info("Timing | Pass 1 scan: %.2fs", time.perf_counter() - pass1_start_time)

        pass2_start_time = time.perf_counter()
        parent = stitch_block_boundaries(manifest_payload)
        logger.info("Timing | Pass 2 stitch: %.2fs", time.perf_counter() - pass2_start_time)

        pass3a_start_time = time.perf_counter()
        root_sizes = build_root_sizes(manifest_payload, parent)
        logger.info("Timing | Pass 3a root sizes: %.2fs", time.perf_counter() - pass3a_start_time)

        pass3b_start_time = time.perf_counter()
        direct_stats = aggregate_final_region_stats(
            manifest_payload=manifest_payload,
            parent=parent,
            root_sizes=root_sizes,
            min_voxels=min_voxels,
        )
        logger.info("Timing | Pass 3b region collapse: %.2fs", time.perf_counter() - pass3b_start_time)

        export_start_time = time.perf_counter()
        export_region_excel(region_tree, direct_stats, output_path, flush_every)
        logger.info("Timing | Excel export: %.2fs", time.perf_counter() - export_start_time)
    finally:
        logger.info("Timing | Total analysis: %.2fs", time.perf_counter() - total_start_time)
        if keep_tmp:
            logger.info("Temporary block artifacts kept at %s", tmp_root)
        elif tmp_root.exists():
            logger.info("Removing temporary block artifacts at %s", tmp_root)
            shutil.rmtree(tmp_root)


def main():
    import sys as _sys

    args = parse_args()

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
        _started_at = time.time()

        analyze_zarr_graph(
            mask_zarr_path=args.mask_zarr,
            label_zarr_path=args.label_zarr,
            signal_zarr_path=args.signal_zarr,
            cfg_path=args.cfg,
            output_path=args.output,
            dataset_name=args.dataset_name,
            block_size=parse_block_size(args.block_size),
            foreground_mode=args.foreground_mode,
            foreground_label=args.foreground_label,
            min_voxels=args.min_voxels,
            flush_every=args.flush_every,
            resolution_xyz=parse_resolution_xyz(args.resolution_xyz),
            tmp_dir=args.tmp_dir,
            keep_tmp=args.keep_tmp,
            pass1_workers=args.pass1_workers,
            hemisphere_zarr_path=args.hemisphere_zarr,
        )

        if write_run_manifest is not None:
            output_dir = Path(args.output).parent
            write_run_manifest(
                output_dir,
                module="registration.region_signal_analysis_zarr_graph",
                entrypoint="analyze_zarr_graph",
                inputs={
                    "mask_zarr_path": args.mask_zarr,
                    "label_zarr_path": args.label_zarr,
                    "signal_zarr_path": args.signal_zarr,
                    "cfg_path": args.cfg,
                    "output_path": args.output,
                    "dataset_name": args.dataset_name,
                    "foreground_mode": args.foreground_mode,
                    "foreground_label": args.foreground_label,
                    "min_voxels": args.min_voxels,
                    "resolution_xyz": args.resolution_xyz,
                    "hemisphere_zarr_path": args.hemisphere_zarr,
                },
                outputs=[Path(args.output)],
                started_at=_started_at,
            )

        logger.info("Analysis finished. Excel saved to %s", args.output)
    except Exception as exc:
        if PipelineError is not None and isinstance(exc, PipelineError):
            print(json.dumps({"error_code": exc.code.value, "message": str(exc.message)}), file=_sys.stderr)
            _sys.exit(exc.exit_code)
        logger.exception("Unhandled error: %s", exc)
        _sys.exit(1)


if __name__ == "__main__":
    main()
