#!/usr/bin/env python3
"""
Inspect whether a target region and its descendants are present in a label volume.
Supports Zarr and NIfTI reference/sample labels.
"""

import argparse
import ast
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description="Check voxel coverage of a target region subtree inside a label volume."
    )
    parser.add_argument("--label_zarr", required=True, help="Registered label volume path (.zarr or .nii/.nii.gz)")
    parser.add_argument("--reference_label_zarr", help="Optional reference label volume path (.zarr or .nii/.nii.gz)")
    parser.add_argument("--cfg", required=True, help="Path to region CSV file")
    parser.add_argument("--dataset_name", default="0", help="Dataset name inside the Zarr group")
    parser.add_argument("--region_id", type=int, help="Target region id")
    parser.add_argument("--acronym", help="Target region acronym")
    parser.add_argument(
        "--block_size",
        default="",
        help="Optional block size override as z,y,x. Default uses the label Zarr chunk size.",
    )
    parser.add_argument(
        "--top_zero",
        type=int,
        default=20,
        help="How many zero-voxel descendants to show in the report",
    )
    return parser.parse_args()


def open_zarr_dataset(path_like, dataset_name):
    import zarr

    path = Path(path_like)
    if not path.exists():
        raise FileNotFoundError(f"Zarr path not found: {path}")

    root = zarr.open(str(path), mode="r")
    if isinstance(root, zarr.Array):
        return root

    if dataset_name in root and isinstance(root[dataset_name], zarr.Array):
        return root[dataset_name]

    array_keys = list(root.array_keys())
    if len(array_keys) == 1:
        return root[array_keys[0]]

    raise ValueError(
        f"Could not resolve a Zarr array from {path}. "
        f"Available arrays: {array_keys}, requested dataset_name={dataset_name}"
    )


class NiftiVolume:
    def __init__(self, array):
        self._array = np.asarray(array)
        self.shape = self._array.shape
        self.chunks = None

    def __getitem__(self, item):
        return self._array[item]


def open_nifti_dataset(path_like):
    path = Path(path_like)
    if not path.exists():
        raise FileNotFoundError(f"NIfTI path not found: {path}")

    try:
        import nibabel as nib
        img = nib.load(str(path))
        data = np.asanyarray(img.dataobj)
    except ModuleNotFoundError:
        try:
            import ants
            data = ants.image_read(str(path)).numpy()
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "Reading .nii/.nii.gz requires nibabel or ants to be installed in this Python environment."
            ) from exc

    if data.ndim != 3:
        raise ValueError(f"Expected 3D NIfTI volume, got shape={data.shape}")
    return NiftiVolume(data)


def open_label_volume(path_like, dataset_name):
    path = Path(path_like)
    suffixes = [suffix.lower() for suffix in path.suffixes]
    if path.suffix.lower() == ".zarr" or path.is_dir():
        return open_zarr_dataset(path, dataset_name)
    if suffixes[-2:] == [".nii", ".gz"] or path.suffix.lower() == ".nii":
        return open_nifti_dataset(path)
    raise ValueError(f"Unsupported label volume format: {path}")


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


def choose_block_shape(label_volume, requested_block_shape):
    if requested_block_shape is not None:
        return requested_block_shape
    chunks = getattr(label_volume, "chunks", None)
    if chunks is None:
        shape = tuple(int(value) for value in label_volume.shape)
        return shape
    return tuple(int(value) for value in chunks[:3])


def iter_block_slices(volume_shape, block_shape):
    for z0 in range(0, volume_shape[0], block_shape[0]):
        z1 = min(z0 + block_shape[0], volume_shape[0])
        for y0 in range(0, volume_shape[1], block_shape[1]):
            y1 = min(y0 + block_shape[1], volume_shape[1])
            for x0 in range(0, volume_shape[2], block_shape[2]):
                x1 = min(x0 + block_shape[2], volume_shape[2])
                yield (slice(z0, z1), slice(y0, y1), slice(x0, x1))


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
    acronym_to_ids = {}
    for _, row in region_df.iterrows():
        structure_id = int(row["id"])
        structure_path = parse_structure_id_path(row["structure_id_path"])
        parent_structure_id = structure_path[-2] if len(structure_path) >= 2 else None
        acronym = parse_acronym_text(row["acronym"])
        node = {
            "id": structure_id,
            "name": str(row["name"]) if pd.notna(row["name"]) else str(structure_id),
            "acronym": acronym,
            "parent_structure_id": parent_structure_id,
            "children": [],
        }
        nodes_by_id[structure_id] = node
        acronym_to_ids.setdefault(acronym.lower(), []).append(structure_id)

    for node in nodes_by_id.values():
        parent_structure_id = node["parent_structure_id"]
        if parent_structure_id in nodes_by_id:
            nodes_by_id[parent_structure_id]["children"].append(node)

    return nodes_by_id, acronym_to_ids


def resolve_target_node(nodes_by_id, acronym_to_ids, region_id=None, acronym=None):
    if region_id is not None:
        if region_id not in nodes_by_id:
            raise KeyError(f"Region id not found in CSV: {region_id}")
        return nodes_by_id[region_id]

    if acronym is None:
        raise ValueError("Please provide either --region_id or --acronym")

    matched_ids = acronym_to_ids.get(acronym.lower(), [])
    if not matched_ids:
        raise KeyError(f"Acronym not found in CSV: {acronym}")
    if len(matched_ids) > 1:
        raise ValueError(f"Acronym is ambiguous: {acronym} -> {matched_ids}. Please use --region_id.")
    return nodes_by_id[matched_ids[0]]


def collect_subtree_nodes(node):
    nodes = [node]
    for child in node["children"]:
        nodes.extend(collect_subtree_nodes(child))
    return nodes


def count_label_voxels(label_volume, block_shape):
    counts = {}
    all_blocks = list(iter_block_slices(label_volume.shape, block_shape))
    with tqdm(total=len(all_blocks), desc="Scanning label blocks", unit="block") as progress_bar:
        for block_slices in all_blocks:
            block = np.asarray(label_volume[block_slices])
            positive_labels = block[block > 0]
            if positive_labels.size > 0:
                unique_labels, unique_counts = np.unique(positive_labels, return_counts=True)
                for label_id, label_count in zip(unique_labels.tolist(), unique_counts.tolist()):
                    counts[int(label_id)] = counts.get(int(label_id), 0) + int(label_count)
            progress_bar.update(1)
    return counts


def summarize_subtree_counts(subtree_nodes, voxel_counts):
    direct_counts = {node["id"]: int(voxel_counts.get(node["id"], 0)) for node in subtree_nodes}
    subtree_total = int(sum(direct_counts.values()))
    nonzero_nodes = [node for node in subtree_nodes if direct_counts[node["id"]] > 0]
    zero_nodes = [node for node in subtree_nodes if direct_counts[node["id"]] == 0]
    return {
        "direct_counts": direct_counts,
        "subtree_total": subtree_total,
        "nonzero_nodes": nonzero_nodes,
        "zero_nodes": zero_nodes,
    }


def format_node_label(node):
    return f"{node['name']},{node['acronym']} (id={node['id']})"


def main():
    args = parse_args()
    label_volume = open_label_volume(args.label_zarr, args.dataset_name)
    if len(label_volume.shape) != 3:
        raise ValueError(f"Expected 3D label volume, got shape={label_volume.shape}")

    block_shape = choose_block_shape(label_volume, parse_block_size(args.block_size))
    print(f"Label shape: {label_volume.shape}")
    print(f"Block shape: {block_shape}")

    nodes_by_id, acronym_to_ids = load_region_tree(args.cfg)
    target_node = resolve_target_node(
        nodes_by_id,
        acronym_to_ids,
        region_id=args.region_id,
        acronym=args.acronym,
    )

    voxel_counts = count_label_voxels(label_volume, block_shape)
    subtree_nodes = collect_subtree_nodes(target_node)
    sample_summary = summarize_subtree_counts(subtree_nodes, voxel_counts)
    subtree_total_voxels = sample_summary["subtree_total"]
    direct_voxels = sample_summary["direct_counts"][target_node["id"]]
    nonzero_nodes = sample_summary["nonzero_nodes"]
    zero_nodes = sample_summary["zero_nodes"]

    print("\n=== Target Region Coverage ===")
    print(f"Target: {format_node_label(target_node)}")
    print(f"Direct voxels on target id: {direct_voxels}")
    print(f"Subtree voxel total: {subtree_total_voxels}")
    print(f"Subtree node count: {len(subtree_nodes)}")
    print(f"Subtree nodes with voxels > 0: {len(nonzero_nodes)}")
    print(f"Subtree nodes with voxels = 0: {len(zero_nodes)}")

    if direct_voxels == 0 and subtree_total_voxels == 0:
        print("Interpretation: this region subtree is absent from the current label Zarr.")
    elif direct_voxels == 0 and subtree_total_voxels > 0:
        print("Interpretation: the parent region id has no direct voxels, but some descendants are present.")
    else:
        print("Interpretation: this region id itself is present in the current label Zarr.")

    print("\n=== Present Descendants ===")
    for node in sorted(nonzero_nodes, key=lambda item: (-voxel_counts.get(item["id"], 0), item["name"]))[:20]:
        print(f"{voxel_counts.get(node['id'], 0):>12} voxels | {format_node_label(node)}")

    print("\n=== Zero-Voxel Descendants ===")
    for node in sorted(zero_nodes, key=lambda item: item["name"])[: max(0, args.top_zero)]:
        print(f"{0:>12} voxels | {format_node_label(node)}")

    missing_label_ids = sorted(
        node["id"]
        for node in subtree_nodes
        if node["id"] not in voxel_counts
    )
    if missing_label_ids:
        print("\n=== Missing IDs In Label Zarr ===")
        print(",".join(str(region_id) for region_id in missing_label_ids[:100]))

    if args.reference_label_zarr:
        print("\n=== Reference Comparison ===")
        reference_label_volume = open_label_volume(args.reference_label_zarr, args.dataset_name)
        if len(reference_label_volume.shape) != 3:
            raise ValueError(f"Expected 3D reference label volume, got shape={reference_label_volume.shape}")

        reference_block_shape = choose_block_shape(reference_label_volume, parse_block_size(args.block_size))
        print(f"Reference label shape: {reference_label_volume.shape}")
        print(f"Reference block shape: {reference_block_shape}")
        reference_voxel_counts = count_label_voxels(reference_label_volume, reference_block_shape)
        reference_summary = summarize_subtree_counts(subtree_nodes, reference_voxel_counts)

        print(f"Reference direct voxels on target id: {reference_summary['direct_counts'][target_node['id']]}")
        print(f"Reference subtree voxel total: {reference_summary['subtree_total']}")
        print(f"Reference subtree nodes with voxels > 0: {len(reference_summary['nonzero_nodes'])}")
        print(f"Reference subtree nodes with voxels = 0: {len(reference_summary['zero_nodes'])}")

        atlas_present_sample_missing = [
            node for node in subtree_nodes
            if reference_summary["direct_counts"][node["id"]] > 0 and sample_summary["direct_counts"][node["id"]] == 0
        ]
        sample_present_reference_missing = [
            node for node in subtree_nodes
            if reference_summary["direct_counts"][node["id"]] == 0 and sample_summary["direct_counts"][node["id"]] > 0
        ]

        print("\n=== Present In Reference But Missing In Sample ===")
        for node in sorted(
            atlas_present_sample_missing,
            key=lambda item: (-reference_summary["direct_counts"][item["id"]], item["name"])
        )[:50]:
            print(
                f"{reference_summary['direct_counts'][node['id']]:>12} ref voxels | "
                f"{0:>12} sample voxels | {format_node_label(node)}"
            )

        print("\n=== Present In Sample But Missing In Reference ===")
        for node in sorted(
            sample_present_reference_missing,
            key=lambda item: (-sample_summary["direct_counts"][item["id"]], item["name"])
        )[:50]:
            print(
                f"{0:>12} ref voxels | "
                f"{sample_summary['direct_counts'][node['id']]:>12} sample voxels | {format_node_label(node)}"
            )


if __name__ == "__main__":
    main()
