#!/usr/bin/env python3
"""
Streamed brain-region signal analysis.

Design goals:
1. Traverse regions defined in add_id_ytw.json
2. Avoid loading the full brain into memory
3. Preserve 3D connected-component semantics for signal counting
4. Export one Excel sheet per st_level
"""

import argparse
import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from openpyxl.styles import Alignment, Font
from scipy import ndimage
from tqdm import tqdm


TIFF_PATTERNS = ("*.tif", "*.tiff", "*.TIF", "*.TIFF")
CONNECTIVITY_3D = np.ones((3, 3, 3), dtype=np.uint8)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze signal voxels, counts, and intensity for each brain region."
    )
    parser.add_argument("--mask_path", required=True, help="Mask TIFF folder or 3D TIFF file")
    parser.add_argument("--label_path", required=True, help="Label TIFF folder or 3D TIFF file")
    parser.add_argument("--signal_path", required=True, help="Raw signal TIFF folder or 3D TIFF file")
    parser.add_argument("--cfg", required=True, help="Path to region CSV file")
    parser.add_argument("--output", required=True, help="Output Excel file path")
    parser.add_argument(
        "--min_voxels",
        type=int,
        default=0,
        help="Remove connected components smaller than this voxel count; 0 disables filtering",
    )
    parser.add_argument(
        "--flush_every",
        type=int,
        default=25,
        help="Rewrite the Excel file after every N finished regions",
    )
    parser.add_argument(
        "--chunk_depth",
        type=int,
        default=16,
        help="Number of z-slices processed per 3D chunk",
    )
    parser.add_argument(
        "--foreground_label",
        type=int,
        default=1,
        help="Foreground label value for ilastik Simple Segmentation masks",
    )
    parser.add_argument(
        "--resolution_xyz",
        default="1,1,1",
        help="Voxel size in microns as x,y,z, for example 1.8,1.8,2.0",
    )
    return parser.parse_args()


def list_tiff_files(folder_path):
    files = []
    for pattern in TIFF_PATTERNS:
        files.extend(folder_path.glob(pattern))
    return sorted({file_path.resolve() for file_path in files})


def resolve_input(path_like, volume_name):
    path = Path(path_like)
    if not path.exists():
        raise FileNotFoundError(f"{volume_name} path not found: {path}")
    return path


def get_volume_shape(path_like, volume_name):
    path = resolve_input(path_like, volume_name)
    if path.is_dir():
        tiff_files = list_tiff_files(path)
        if not tiff_files:
            raise FileNotFoundError(f"No TIFF files found in {path}")
        first_slice = tifffile.imread(str(tiff_files[0]))
        if first_slice.ndim != 2:
            raise ValueError(f"{volume_name} folder should contain 2D slices, got {first_slice.shape}")
        return (len(tiff_files),) + first_slice.shape

    with tifffile.TiffFile(str(path)) as tif:
        series_shape = tif.series[0].shape

    if len(series_shape) == 2:
        return (1,) + tuple(series_shape)
    if len(series_shape) == 3:
        return tuple(series_shape)
    raise ValueError(f"{volume_name} must be 3D, got shape {series_shape}")


def iter_volume_slices(path_like, volume_name):
    path = resolve_input(path_like, volume_name)
    if path.is_dir():
        tiff_files = list_tiff_files(path)
        if not tiff_files:
            raise FileNotFoundError(f"No TIFF files found in {path}")
        for tiff_file in tiff_files:
            yield tifffile.imread(str(tiff_file))
        return

    with tifffile.TiffFile(str(path)) as tif:
        for page in tif.pages:
            slice_data = page.asarray()
            if slice_data.ndim == 2:
                yield slice_data
            else:
                raise ValueError(f"{volume_name} pages must be 2D, got {slice_data.shape}")


def iter_volume_chunks(mask_path, label_path, signal_path, chunk_depth):
    mask_iter = iter_volume_slices(mask_path, "mask volume")
    label_iter = iter_volume_slices(label_path, "label volume")
    signal_iter = iter_volume_slices(signal_path, "signal volume")

    while True:
        mask_slices = []
        label_slices = []
        signal_slices = []

        for _ in range(chunk_depth):
            try:
                mask_slice = next(mask_iter)
                label_slice = next(label_iter)
                signal_slice = next(signal_iter)
            except StopIteration:
                break

            if mask_slice.shape != label_slice.shape or mask_slice.shape != signal_slice.shape:
                raise ValueError(
                    f"Chunk slice shape mismatch: "
                    f"mask={mask_slice.shape}, label={label_slice.shape}, signal={signal_slice.shape}"
                )

            mask_slices.append(mask_slice)
            label_slices.append(label_slice)
            signal_slices.append(signal_slice)

        if not mask_slices:
            break

        yield (
            np.stack(mask_slices, axis=0),
            np.stack(label_slices, axis=0),
            np.stack(signal_slices, axis=0),
        )


def validate_shapes(mask_path, label_path, signal_path):
    mask_shape = get_volume_shape(mask_path, "mask volume")
    label_shape = get_volume_shape(label_path, "label volume")
    signal_shape = get_volume_shape(signal_path, "signal volume")

    if mask_shape != label_shape:
        raise ValueError(f"Mask and label shape mismatch: {mask_shape} vs {label_shape}")
    if mask_shape != signal_shape:
        raise ValueError(f"Mask and signal shape mismatch: {mask_shape} vs {signal_shape}")

    print(f"volume shape={mask_shape}")
    return mask_shape


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
    row_order_by_id = {}
    for row_index, row in region_df.iterrows():
        structure_id = int(row["id"])
        structure_path = parse_structure_id_path(row["structure_id_path"])
        st_level = max(0, len(structure_path) - 1)
        parent_structure_id = structure_path[-2] if len(structure_path) >= 2 else None

        node = {
            "id": structure_id,
            "id_ytw": structure_id,
            "atlas_id": structure_id,
            "name": str(row["name"]).split(",")[0] if pd.notna(row["name"]) else str(structure_id),
            "acronym": parse_acronym_text(row["acronym"]),
            "st_level": st_level,
            "parent_structure_id": parent_structure_id,
            "graph_order": row_index,
            "children": [],
        }
        nodes_by_id[structure_id] = node
        row_order_by_id[structure_id] = row_index

    root_node = None
    for structure_id, node in nodes_by_id.items():
        parent_structure_id = node["parent_structure_id"]
        if parent_structure_id is None or parent_structure_id not in nodes_by_id:
            root_node = node
            continue
        nodes_by_id[parent_structure_id]["children"].append(node)

    if root_node is None:
        raise ValueError(f"Could not determine root node from CSV: {cfg_path}")

    for node in nodes_by_id.values():
        node["children"].sort(key=lambda child: row_order_by_id.get(child["id"], 0))

    return root_node


def parse_resolution_xyz(resolution_text):
    parts = [part.strip() for part in str(resolution_text).split(",") if part.strip()]
    if len(parts) != 3:
        raise ValueError(f"resolution_xyz must have 3 comma-separated values, got: {resolution_text}")
    resolution_xyz = tuple(float(part) for part in parts)
    if any(value <= 0 for value in resolution_xyz):
        raise ValueError(f"resolution_xyz values must be positive, got: {resolution_xyz}")
    return resolution_xyz


def get_region_label_id(node):
    for key in ("id", "id_ytw", "atlas_id"):
        value = node.get(key)
        if isinstance(value, (int, np.integer)) and value >= 0:
            return int(value)
    return None


def make_component_record():
    return {
        "size": 0,
        "region_voxels": {},
        "region_intensity": {},
    }


def make_union_find():
    return {
        "parent": {},
        "size_rank": {},
        "data": {},
        "next_id": 1,
    }


def create_component(union_find):
    component_id = union_find["next_id"]
    union_find["next_id"] += 1
    union_find["parent"][component_id] = component_id
    union_find["size_rank"][component_id] = 1
    union_find["data"][component_id] = make_component_record()
    return component_id


def find_root(union_find, component_id):
    parent = union_find["parent"][component_id]
    if parent != component_id:
        union_find["parent"][component_id] = find_root(union_find, parent)
    return union_find["parent"][component_id]


def merge_region_dict(target_dict, source_dict):
    for key, value in source_dict.items():
        target_dict[key] = target_dict.get(key, 0) + value


def union_components(union_find, component_a, component_b):
    root_a = find_root(union_find, component_a)
    root_b = find_root(union_find, component_b)
    if root_a == root_b:
        return root_a

    rank_a = union_find["size_rank"][root_a]
    rank_b = union_find["size_rank"][root_b]
    if rank_a < rank_b:
        root_a, root_b = root_b, root_a

    union_find["parent"][root_b] = root_a
    if rank_a == rank_b:
        union_find["size_rank"][root_a] += 1

    data_a = union_find["data"][root_a]
    data_b = union_find["data"].pop(root_b)
    data_a["size"] += data_b["size"]
    merge_region_dict(data_a["region_voxels"], data_b["region_voxels"])
    merge_region_dict(data_a["region_intensity"], data_b["region_intensity"])
    union_find["size_rank"].pop(root_b, None)
    return root_a


def update_component_stats(component_record, region_values, signal_values):
    component_record["size"] += int(region_values.size)

    if region_values.size == 0:
        return

    valid_mask = region_values > 0
    valid_regions = region_values[valid_mask]
    valid_signal = signal_values[valid_mask]
    if valid_regions.size == 0:
        return

    unique_regions, inverse_index = np.unique(valid_regions, return_inverse=True)
    region_counts = np.bincount(inverse_index)
    region_sums = np.bincount(inverse_index, weights=valid_signal.astype(np.float64, copy=False))

    for idx, region_id in enumerate(unique_regions.tolist()):
        region_id = int(region_id)
        component_record["region_voxels"][region_id] = (
            component_record["region_voxels"].get(region_id, 0) + int(region_counts[idx])
        )
        component_record["region_intensity"][region_id] = (
            component_record["region_intensity"].get(region_id, 0.0) + float(region_sums[idx])
        )


def add_pair_stats(component_record, region_ids, counts, intensity_sums):
    for region_id, count, intensity_sum in zip(region_ids, counts, intensity_sums):
        region_id = int(region_id)
        component_record["region_voxels"][region_id] = (
            component_record["region_voxels"].get(region_id, 0) + int(count)
        )
        component_record["region_intensity"][region_id] = (
            component_record["region_intensity"].get(region_id, 0.0) + float(intensity_sum)
        )


def accumulate_chunk_stats(union_find, global_id, local_component_mask, label_chunk, signal_chunk):
    component_record = union_find["data"][global_id]
    region_values = label_chunk[local_component_mask]
    signal_values = signal_chunk[local_component_mask]
    update_component_stats(component_record, region_values, signal_values)


def build_boundary_pairs(previous_global_last_slice, current_first_slice_labels, union_find):
    if previous_global_last_slice is None:
        return {}

    boundary_pairs = {}
    current_nonzero = current_first_slice_labels > 0
    if not np.any(current_nonzero):
        return boundary_pairs

    for row_offset in (-1, 0, 1):
        for col_offset in (-1, 0, 1):
            shifted_previous = np.roll(previous_global_last_slice, shift=(row_offset, col_offset), axis=(0, 1))

            if row_offset > 0:
                shifted_previous[:row_offset, :] = 0
            elif row_offset < 0:
                shifted_previous[row_offset:, :] = 0

            if col_offset > 0:
                shifted_previous[:, :col_offset] = 0
            elif col_offset < 0:
                shifted_previous[:, col_offset:] = 0

            overlap_mask = (shifted_previous > 0) & current_nonzero
            if not np.any(overlap_mask):
                continue

            local_ids = current_first_slice_labels[overlap_mask].astype(np.int64, copy=False)
            prev_ids = shifted_previous[overlap_mask].astype(np.int64, copy=False)

            pair_keys = (local_ids << 32) | prev_ids
            unique_keys = np.unique(pair_keys)
            for key in unique_keys.tolist():
                local_id = int(key >> 32)
                prev_id = int(key & 0xFFFFFFFF)
                prev_root = find_root(union_find, prev_id)
                if local_id not in boundary_pairs:
                    boundary_pairs[local_id] = set()
                boundary_pairs[local_id].add(prev_root)

    return boundary_pairs


def build_connected_components(mask_path, label_path, signal_path, chunk_depth, total_slices, foreground_label):
    print("Streaming chunks and building 3D connected components...")
    union_find = make_union_find()
    total_region_voxels = {}
    previous_global_last_slice = None
    processed_slices = 0
    total_chunks = max(1, (total_slices + chunk_depth - 1) // chunk_depth)

    with tqdm(total=total_slices, desc="Processing slices", unit="slice") as progress_bar:
        for chunk_index, (mask_chunk, label_chunk, signal_chunk) in enumerate(
            iter_volume_chunks(mask_path, label_path, signal_path, chunk_depth),
            start=1,
        ):
            positive_labels = label_chunk[label_chunk > 0]
            if positive_labels.size > 0:
                unique_labels, counts = np.unique(positive_labels, return_counts=True)
                for region_id, count in zip(unique_labels.tolist(), counts.tolist()):
                    total_region_voxels[int(region_id)] = total_region_voxels.get(int(region_id), 0) + int(count)

            binary_mask = mask_chunk == foreground_label
            local_labels, num_local_components = ndimage.label(binary_mask, structure=CONNECTIVITY_3D)
            global_chunk = np.zeros_like(local_labels, dtype=np.int32)

            if num_local_components > 0:
                boundary_pairs = build_boundary_pairs(
                    previous_global_last_slice,
                    local_labels[0],
                    union_find,
                )

                local_to_global = np.zeros(num_local_components + 1, dtype=np.int32)
                for local_id in range(1, num_local_components + 1):
                    neighbor_roots = boundary_pairs.get(local_id, set())
                    if neighbor_roots:
                        sorted_roots = sorted(neighbor_roots)
                        global_id = sorted_roots[0]
                        for other_root in sorted_roots[1:]:
                            global_id = union_components(union_find, global_id, other_root)
                        global_id = find_root(union_find, global_id)
                    else:
                        global_id = create_component(union_find)
                    local_to_global[local_id] = global_id

                for local_id in range(1, num_local_components + 1):
                    global_id = find_root(union_find, int(local_to_global[local_id]))
                    local_to_global[local_id] = global_id
                    local_component_mask = local_labels == local_id
                    global_chunk[local_component_mask] = global_id
                    accumulate_chunk_stats(
                        union_find,
                        global_id,
                        local_component_mask,
                        label_chunk,
                        signal_chunk,
                    )

            previous_global_last_slice = global_chunk[-1] if global_chunk.shape[0] > 0 else previous_global_last_slice
            processed_slices += mask_chunk.shape[0]
            progress_bar.update(mask_chunk.shape[0])
            progress_bar.set_postfix(
                chunk=chunk_index,
                total_chunks=total_chunks,
                components=num_local_components,
                refresh=False,
            )

    return total_region_voxels, union_find


def finalize_region_statistics(total_region_voxels, union_find, min_voxels):
    print("Collapsing kept components into per-region statistics...")
    kept_components = 0
    kept_voxels = 0
    region_signal_voxels = {}
    region_signal_counts = {}
    region_sum_intensity = {}

    for root_id, component_record in union_find["data"].items():
        if find_root(union_find, root_id) != root_id:
            continue
        if component_record["size"] < min_voxels:
            continue

        kept_components += 1
        kept_voxels += component_record["size"]

        for region_id, voxel_count in component_record["region_voxels"].items():
            region_signal_voxels[region_id] = region_signal_voxels.get(region_id, 0) + int(voxel_count)
            region_signal_counts[region_id] = region_signal_counts.get(region_id, 0) + 1

        for region_id, intensity_sum in component_record["region_intensity"].items():
            region_sum_intensity[region_id] = region_sum_intensity.get(region_id, 0.0) + float(intensity_sum)

    print(
        f"Kept {kept_components} connected components with >= {min_voxels} voxels, "
        f"covering {kept_voxels} voxels"
    )

    return {
        "total_region_voxels": total_region_voxels,
        "region_signal_voxels": region_signal_voxels,
        "region_signal_counts": region_signal_counts,
        "region_sum_intensity": region_sum_intensity,
    }


def build_display_name(node, label_id):
    name = node.get("name") or ""
    acronym = node.get("acronym") or ""
    return f"{name},{acronym}"


def build_region_row(node, label_id, aggregated_stats, voxel_volume_um3):
    total_voxels = int(aggregated_stats["total_voxels"])
    signal_voxels = int(aggregated_stats["signal_voxels"])
    voxel_density = float(signal_voxels / total_voxels) if total_voxels > 0 else 0.0

    return {
        "Name": build_display_name(node, label_id),
        "st_level": node.get("st_level"),
        "graph_order": node.get("graph_order", 0),
        "Total Voxels": total_voxels,
        "Signal Voxels": signal_voxels,
        "Voxel Density": voxel_density,
        "Signal Count": int(aggregated_stats["signal_count"]),
        "Sum Intensity": float(aggregated_stats["sum_intensity"]),
    }


def flatten_region_rows(region_tree, direct_stats, voxel_volume_um3):
    rows = []

    def visit(node):
        aggregated = {
            "total_voxels": 0,
            "signal_voxels": 0,
            "signal_count": 0,
            "sum_intensity": 0.0,
        }

        label_id = get_region_label_id(node)
        if label_id is not None and label_id > 0:
            aggregated["total_voxels"] += int(direct_stats["total_region_voxels"].get(label_id, 0))
            aggregated["signal_voxels"] += int(direct_stats["region_signal_voxels"].get(label_id, 0))
            aggregated["signal_count"] += int(direct_stats["region_signal_counts"].get(label_id, 0))
            aggregated["sum_intensity"] += float(direct_stats["region_sum_intensity"].get(label_id, 0.0))

        for child in node.get("children", []):
            child_aggregated = visit(child)
            aggregated["total_voxels"] += child_aggregated["total_voxels"]
            aggregated["signal_voxels"] += child_aggregated["signal_voxels"]
            aggregated["signal_count"] += child_aggregated["signal_count"]
            aggregated["sum_intensity"] += child_aggregated["sum_intensity"]

        if label_id is not None and label_id > 0:
            rows.append(build_region_row(node, label_id, aggregated, voxel_volume_um3))

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

    dataframe = dataframe.sort_values(by=["st_level", "graph_order", "Name"], kind="stable").reset_index(drop=True)

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        unique_levels = [
            int(level)
            for level in sorted(dataframe["st_level"].dropna().unique().tolist())
        ]

        for level in unique_levels:
            level_frame = dataframe[dataframe["st_level"] == level].copy().reset_index(drop=True)
            export_frame = level_frame[
                [
                    "Name",
                    "Total Voxels",
                    "Signal Voxels",
                    "Voxel Density",
                    "Signal Count",
                    "Sum Intensity",
                ]
            ]
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


def analyze_regions(
    mask_path,
    label_path,
    signal_path,
    cfg_path,
    output_path,
    min_voxels,
    flush_every,
    chunk_depth,
    foreground_label,
    resolution_xyz,
):
    volume_shape = validate_shapes(mask_path, label_path, signal_path)
    region_tree = load_region_tree(cfg_path)
    voxel_volume_um3 = float(resolution_xyz[0] * resolution_xyz[1] * resolution_xyz[2])

    total_region_voxels, union_find = build_connected_components(
        mask_path,
        label_path,
        signal_path,
        chunk_depth,
        volume_shape[0],
        foreground_label,
    )
    direct_stats = finalize_region_statistics(total_region_voxels, union_find, min_voxels)

    rows = []
    all_rows = flatten_region_rows(region_tree, direct_stats, voxel_volume_um3)
    for index, row in enumerate(all_rows, start=1):
        rows.append(row)
        if flush_every > 0 and index % flush_every == 0:
            print(f"Flushing {len(rows)} rows to {output_path}")
            flush_rows_to_excel(rows, output_path)

    print(f"Final flush with {len(rows)} rows to {output_path}")
    flush_rows_to_excel(rows, output_path)


def main():
    args = parse_args()
    analyze_regions(
        mask_path=args.mask_path,
        label_path=args.label_path,
        signal_path=args.signal_path,
        cfg_path=args.cfg,
        output_path=args.output,
        min_voxels=args.min_voxels,
        flush_every=args.flush_every,
        chunk_depth=args.chunk_depth,
        foreground_label=args.foreground_label,
        resolution_xyz=parse_resolution_xyz(args.resolution_xyz),
    )
    print(f"Analysis finished. Excel saved to {args.output}")


if __name__ == "__main__":
    main()
