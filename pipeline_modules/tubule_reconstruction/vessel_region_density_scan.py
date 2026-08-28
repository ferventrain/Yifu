"""Per-atlas-region vessel volume density scan.

Counts vessel mask foreground voxels per atlas label (scanning only chunks that
contain foreground) and region total voxels from a downsampled atlas label Zarr,
then reports vessel volume density for the requested atlas regions.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import zarr

from pipeline_modules.tubule_reconstruction.region_vessel_analysis import (
    _collect_subtree_ids,
    load_region_tree_with_lookups,
    resolve_region_query,
)


def open_zarr_dataset(zarr_path, dataset_name="0"):
    group = zarr.open(str(zarr_path), mode="r")
    if dataset_name in group:
        return group[dataset_name]
    arrays = list(group.arrays())
    if len(arrays) == 1:
        return arrays[0][1]
    raise ValueError(f"Dataset {dataset_name!r} not found in {zarr_path}")


def load_foreground_chunks(cache_path):
    payload = json.loads(Path(cache_path).read_text(encoding="utf-8"))
    chunks = payload.get("foreground_chunks")
    if not chunks:
        return []
    return [tuple(int(part) for part in str(value).split(".")) for value in chunks]


def region_lookups(cfg_path):
    return load_region_tree_with_lookups(cfg_path)


def parse_resolution_xyz(value):
    return tuple(float(part) for part in str(value).split(","))


def main() -> int:
    parser = argparse.ArgumentParser(description="Scan vessel voxels per atlas region and report volume density")
    parser.add_argument("--mask_zarr", required=True)
    parser.add_argument("--annotation_zarr", required=True, help="Full-resolution registered atlas label Zarr")
    parser.add_argument("--annotation_dataset_name", default="0")
    parser.add_argument("--annotation_resolution_xyz", default="1.8,1.8,2.0")
    parser.add_argument("--foreground_chunks_json", required=True, help="foreground_chunks_min100.json from reconstruction")
    parser.add_argument("--cfg", required=True, help="Allen region CSV")
    parser.add_argument("--regions", default="CTX;CNU;CB;BS;TH;HY;PAL;OLF;HPF", help="Region queries to aggregate")
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    resolution = parse_resolution_xyz(args.annotation_resolution_xyz)
    voxel_volume = float(np.prod(resolution))

    mask = open_zarr_dataset(args.mask_zarr)
    label = open_zarr_dataset(args.annotation_zarr, dataset_name=args.annotation_dataset_name)
    total_chunks = np.ceil(np.asarray(label.shape) / np.asarray(label.chunks)).astype(int)

    nodes_by_id, acronym_to_ids, name_to_ids = region_lookups(args.cfg)
    region_queries = [query.strip() for query in args.regions.split(";") if query.strip()]
    resolved_regions = []
    for query in region_queries:
        try:
            node = resolve_region_query(query, nodes_by_id, acronym_to_ids, name_to_ids)
        except Exception as exc:
            print(f"Warning: region {query!r} not found ({exc})", file=sys.stderr)
            continue
        resolved_regions.append((query, node, _collect_subtree_ids(node)))

    all_subtree_ids = sorted({int(value) for _, _, ids in resolved_regions for value in ids})
    max_label = int(all_subtree_ids[-1]) if all_subtree_ids else 0

    vessel_voxels_by_label = np.zeros(max_label + 1, dtype=np.int64)
    region_voxels_by_label = np.zeros(max_label + 1, dtype=np.int64)

    chunk_indices = load_foreground_chunks(args.foreground_chunks_json)
    if not chunk_indices:
        grid = np.ceil(np.asarray(mask.shape) / np.asarray(mask.chunks)).astype(int)
        chunk_indices = [
            tuple(index)
            for index in np.ndindex(tuple(grid))
            if all(index[axis] * mask.chunks[axis] < mask.shape[axis] for axis in range(3))
        ]

    for chunk_index in chunk_indices:
        slices = tuple(
            slice(int(chunk_index[axis]) * int(mask.chunks[axis]), min(int((chunk_index[axis] + 1) * mask.chunks[axis]), int(mask.shape[axis])))
            for axis in range(3)
        )
        mask_block = np.asarray(mask[slices])
        if not np.any(mask_block):
            continue
        label_block = np.asarray(label[slices]).astype(np.int64)
        values = label_block[mask_block > 0]
        if values.size == 0:
            continue
        values = values[values <= max_label]
        if values.size == 0:
            continue
        counts = np.bincount(values)
        vessel_voxels_by_label[: len(counts)] += counts

    print("Label histogram from full-resolution atlas ...")
    for index in np.ndindex(tuple(total_chunks)):
        slices = tuple(
            slice(int(index[axis]) * int(label.chunks[axis]), min(int((index[axis] + 1) * label.chunks[axis]), int(label.shape[axis])))
            for axis in range(3)
        )
        block = np.asarray(label[slices]).astype(np.int64)
        counts = np.bincount(block.ravel())
        region_voxels_by_label[: len(counts)] += counts[: max_label + 1]

    nodes_by_id, acronym_to_ids, name_to_ids = region_lookups(args.cfg)
    rows = []
    for query in args.regions.split(";"):
        query = query.strip()
        if not query:
            continue
        try:
            node = resolve_region_query(query, nodes_by_id, acronym_to_ids, name_to_ids)
        except Exception as exc:
            print(f"Warning: region {query!r} not found ({exc})", file=sys.stderr)
            continue
        region_id = int(node["id"])
        ids = [int(value) for value in _collect_subtree_ids(node) if int(value) <= max_label]
        mask_voxels = int(vessel_voxels_by_label[ids].sum())
        region_voxels = int(region_voxels_by_label[ids].sum())
        vessel_volume = float(mask_voxels * voxel_volume)
        region_volume = float(region_voxels * voxel_volume)
        density_voxel = float(mask_voxels / region_voxels) if region_voxels else np.nan
        density_um3_per_um3 = density_voxel
        density_um3_per_mm3 = float(vessel_volume / (region_volume / 1e9)) if region_volume else np.nan
        rows.append(
            {
                "query": query,
                "region_id": region_id,
                "region_acronym": node.get("acronym", ""),
                "region_name": node.get("name", ""),
                "num_subtree_ids": len(ids),
                "vessel_mask_voxels": mask_voxels,
                "region_voxels": region_voxels,
                "vessel_volume_um3": vessel_volume,
                "region_volume_um3": region_volume,
                "density_vessel_fraction": density_voxel,
                "density_um3_per_mm3": density_um3_per_mm3,
            }
        )

    table = pd.DataFrame(rows)
    table.to_csv(output_dir / "region_vessel_volume_density.csv", index=False)
    payload = {"regions": rows}
    (output_dir / "region_vessel_volume_density.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Wrote {len(rows)} region densities to {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
