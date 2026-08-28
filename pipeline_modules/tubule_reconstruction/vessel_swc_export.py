"""Streaming SWC export from skeleton CSVs (whole brain and atlas-region filtered).

Imaris imports SWC files as filaments; each skeleton is written as a tree with
parent=-1 roots and globally unique node ids. Optional region filtering keeps
only vertices whose atlas label belongs to a requested region subtree.
"""
from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import zarr

VERTEX_CHUNK = 3_000_000
EDGE_CHUNK = 3_000_000
LABEL_CHUNK_CACHE_SIZE = 64


def parse_resolution_xyz(value):
    return tuple(float(part) for part in str(value).split(","))


def open_zarr_dataset(zarr_path, dataset_name="0"):
    group = zarr.open(str(zarr_path), mode="r")
    if dataset_name in group:
        return group[dataset_name]
    arrays = list(group.arrays())
    if len(arrays) == 1:
        return arrays[0][1]
    raise ValueError(f"Dataset {dataset_name!r} not found in {zarr_path}")


def grouped_reader(csv_path, usecols, chunksize):
    reader = pd.read_csv(csv_path, usecols=usecols, chunksize=chunksize, low_memory=False)
    pending_key = None
    pending_rows = []
    for chunk in reader:
        chunk = chunk.dropna(subset=["skeleton_id"])
        chunk["skeleton_id"] = chunk["skeleton_id"].astype(np.int64)
        for key, group in itertools.groupby(chunk.itertuples(index=False), key=lambda row: int(row.skeleton_id)):
            group_rows = list(group)
            if pending_key is None:
                pending_key = key
                pending_rows = group_rows
            elif key == pending_key:
                pending_rows.extend(group_rows)
            else:
                yield pending_key, pending_rows
                pending_key = key
                pending_rows = group_rows
    if pending_key is not None:
        yield pending_key, pending_rows


class RegionFilter:
    def __init__(self, annotation_zarr_path, subtree_ids, resolution_zyx):
        self.label = open_zarr_dataset(annotation_zarr_path)
        self.subtree = set(int(value) for value in subtree_ids)
        self.resolution_zyx = np.asarray(resolution_zyx, dtype=np.float64)
        self.chunk_cache = {}
        self.cache_order = []

    def _load_chunk(self, chunk_index):
        if chunk_index in self.chunk_cache:
            self.cache_order.remove(chunk_index)
            self.cache_order.append(chunk_index)
            return self.chunk_cache[chunk_index]
        slices = tuple(
            slice(int(chunk_index[axis]) * int(self.label.chunks[axis]), min(int((int(chunk_index[axis]) + 1) * self.label.chunks[axis]), int(self.label.shape[axis])))
            for axis in range(3)
        )
        block = np.asarray(self.label[slices]).astype(np.int64)
        self.chunk_cache[chunk_index] = block
        self.cache_order.append(chunk_index)
        while len(self.cache_order) > LABEL_CHUNK_CACHE_SIZE:
            oldest = self.cache_order.pop(0)
            self.chunk_cache.pop(oldest, None)
        return block

    def keep(self, z_um, y_um, x_um):
        voxel = np.floor(np.asarray([z_um, y_um, x_um], dtype=np.float64) / self.resolution_zyx).astype(np.int64)
        chunk_index = tuple(int(v // self.label.chunks[axis]) for axis, v in enumerate(voxel))
        block = self._load_chunk(chunk_index)
        local = (
            int(voxel[0]) - int(chunk_index[0]) * int(self.label.chunks[0]),
            int(voxel[1]) - int(chunk_index[1]) * int(self.label.chunks[1]),
            int(voxel[2]) - int(chunk_index[2]) * int(self.label.chunks[2]),
        )
        return int(block[local]) in self.subtree


def bfs_parents(node_ids, edges):
    node_ids = sorted(set(node_ids))
    adjacency = {nid: [] for nid in node_ids}
    for src, tgt in edges:
        if src in adjacency and tgt in adjacency:
            adjacency[src].append(tgt)
            adjacency[tgt].append(src)

    order = []
    parent_map = {}
    unvisited = set(node_ids)
    while unvisited:
        root = next(iter(unvisited))
        parent_map[root] = -1
        order.append(root)
        queue = [root]
        visited = {root}
        unvisited.discard(root)
        while queue:
            current = queue.pop(0)
            for neighbor in adjacency[current]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    unvisited.discard(neighbor)
                    parent_map[neighbor] = current
                    order.append(neighbor)
                    queue.append(neighbor)
    return parent_map, order


def main() -> int:
    parser = argparse.ArgumentParser(description="Export merged SWC files from skeleton CSVs")
    parser.add_argument("--run_dir", required=True, help="Directory containing skeleton_vertices.csv / skeleton_edges.csv")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--output_prefix", default="wholebrain")
    parser.add_argument("--resolution_xyz", default="1.8,1.8,2.0")
    parser.add_argument("--filter_subtree_ids", default="", help="Comma-separated atlas label ids to keep (region subtree)")
    parser.add_argument("--annotation_zarr", default="", help="Full-resolution atlas label Zarr (required with --filter_subtree_ids)")
    parser.add_argument("--annotation_dataset_name", default="0")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    resolution_xyz = parse_resolution_xyz(args.resolution_xyz)
    resolution_zyx = resolution_xyz[::-1]

    region_filter = None
    if args.filter_subtree_ids:
        subtree_ids = [int(part) for part in args.filter_subtree_ids.split(",") if part.strip()]
        region_filter = RegionFilter(args.annotation_zarr, subtree_ids, resolution_zyx)

    stitch_rows = {}
    stitch_csv = run_dir / "skeleton_stitch_edges.csv"
    if stitch_csv.exists():
        for row in pd.read_csv(stitch_csv, low_memory=False).itertuples(index=False):
            key = int(row.skeleton_id)
            stitch_rows.setdefault(key, []).append((int(row.source_node), int(row.target_node)))

    vertex_usecols = ["skeleton_id", "node_id", "z_um", "y_um", "x_um", "radius_um"]
    edge_usecols = ["skeleton_id", "source_node", "target_node"]

    vertex_stream = grouped_reader(run_dir / "skeleton_vertices.csv", vertex_usecols, VERTEX_CHUNK)
    edge_stream = grouped_reader(run_dir / "skeleton_edges.csv", edge_usecols, EDGE_CHUNK)

    whole_path = output_dir / f"{args.output_prefix}.swc"
    filtered_path = output_dir / f"{args.output_prefix}_filtered.swc" if region_filter else None

    next_global_id = 1
    whole_written = 0
    filtered_written = 0
    filtered_global_ids = set()

    header = f"# merged skeleton export, resolution_xyz={args.resolution_xyz}\n# id type x y z radius parent\n"
    edge_iter = iter(edge_stream)
    current_edges = None

    with open(whole_path, "w", encoding="utf-8") as whole_fh:
        whole_fh.write(header)
        filtered_fh = open(filtered_path, "w", encoding="utf-8") if filtered_path else None
        if filtered_fh:
            filtered_fh.write(header)

        for skeleton_id, vertex_rows in vertex_stream:
            while current_edges is None or current_edges[0] < skeleton_id:
                try:
                    current_edges = next(edge_iter)
                except StopIteration:
                    current_edges = None
                    break
            edges = []
            if current_edges is not None and current_edges[0] == skeleton_id:
                edges = [(int(row.source_node), int(row.target_node)) for row in current_edges[1]]
                current_edges = None
            edges.extend(stitch_rows.get(skeleton_id, []))

            vertex_map = {int(row.node_id): row for row in vertex_rows}
            parent_map, bfs_order = bfs_parents(list(vertex_map), edges)
            local_to_global = {node_id: next_global_id + index for index, node_id in enumerate(bfs_order)}

            for node_id in bfs_order:
                row = vertex_map[node_id]
                radius = float(row.radius_um) if not pd.isna(row.radius_um) else 1.0
                parent = parent_map.get(node_id, -1)
                parent_id = local_to_global.get(parent, -1) if parent >= 0 else -1
                line = (
                    f"{local_to_global[node_id]} 0 {float(row.x_um):.6f} {float(row.y_um):.6f} "
                    f"{float(row.z_um):.6f} {radius:.6f} {parent_id}\n"
                )
                whole_fh.write(line)
                whole_written += 1
                if filtered_fh:
                    keep = region_filter.keep(float(row.z_um), float(row.y_um), float(row.x_um))
                    if keep:
                        if parent_id >= 0 and parent_id not in filtered_global_ids:
                            line = (
                                f"{local_to_global[node_id]} 0 {float(row.x_um):.6f} {float(row.y_um):.6f} "
                                f"{float(row.z_um):.6f} {radius:.6f} -1\n"
                            )
                        filtered_fh.write(line)
                        filtered_global_ids.add(local_to_global[node_id])
                        filtered_written += 1

            next_global_id += len(bfs_order)

        if filtered_fh:
            filtered_fh.close()

    print(f"Whole-brain SWC: {whole_path} ({whole_written} nodes)")
    if filtered_path:
        print(f"Filtered SWC: {filtered_path} ({filtered_written} nodes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
