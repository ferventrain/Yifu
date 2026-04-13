import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import zarr
from scipy import ndimage


DEFAULT_TEASAR_PARAMS = {
    "scale": 1.5,
    "const": 300,
    "pdrf_scale": 100000,
    "pdrf_exponent": 4,
    "soma_acceptance_threshold": 3500,
    "soma_detection_threshold": 750,
    "soma_invalidation_scale": 1.0,
    "soma_invalidation_const": 300,
    "max_paths": None,
}


def open_zarr_dataset(path_like, dataset_name="0"):
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


def parse_resolution_xyz(resolution_text):
    if isinstance(resolution_text, (tuple, list)):
        if len(resolution_text) != 3:
            raise ValueError(f"resolution_xyz must have 3 values, got: {resolution_text}")
        return tuple(float(v) for v in resolution_text)

    parts = [part.strip() for part in str(resolution_text).split(",") if part.strip()]
    if len(parts) != 3:
        raise ValueError(f"resolution_xyz must have 3 comma-separated values, got: {resolution_text}")
    return tuple(float(part) for part in parts)


def parse_roi(roi_text):
    if not str(roi_text).strip():
        return None

    roi = json.loads(roi_text)
    if not isinstance(roi, dict):
        raise ValueError("ROI must be a JSON object like {'z':[0,100],'y':[0,200],'x':[0,200]}")

    slices = []
    for axis in ("z", "y", "x"):
        if axis not in roi:
            raise ValueError(f"ROI is missing axis: {axis}")
        bounds = roi[axis]
        if not isinstance(bounds, list) or len(bounds) != 2:
            raise ValueError(f"ROI axis '{axis}' must be [start, stop]")
        slices.append(slice(int(bounds[0]), int(bounds[1])))
    return tuple(slices)


def list_existing_chunk_indices(mask_zarr):
    """List chunk indices that physically exist in the Zarr store."""
    store = mask_zarr.store
    array_path = getattr(mask_zarr, "path", "")
    dim_sep = getattr(mask_zarr, "_dimension_separator", ".")
    ndim = len(mask_zarr.shape)

    prefix = f"{array_path}/" if array_path else ""
    existing = set()

    for raw_key in store.keys():
        key = str(raw_key)
        if prefix and not key.startswith(prefix):
            continue

        rel = key[len(prefix):] if prefix else key
        if rel in {".zarray", ".zattrs", ".zgroup", "zarr.json"}:
            continue
        if rel.startswith("."):
            continue

        parts = rel.split(dim_sep)
        if len(parts) != ndim:
            continue

        try:
            idx = tuple(int(p) for p in parts)
        except ValueError:
            continue

        existing.add(idx)

    return sorted(existing)


def chunk_index_to_slices(chunk_index, chunks, shape):
    slices = []
    for axis, chunk_idx in enumerate(chunk_index):
        start = int(chunk_idx) * int(chunks[axis])
        stop = min(start + int(chunks[axis]), int(shape[axis]))
        slices.append(slice(start, stop))
    return tuple(slices)


def _extract_binary_mask(mask_zarr, roi=None, foreground_label=1):
    if roi is None:
        mask = np.asarray(mask_zarr[:])
    else:
        mask = np.asarray(mask_zarr[roi])

    if mask.ndim != 3:
        raise ValueError(f"Expected a 3D mask volume, got shape={mask.shape}")

    return mask == foreground_label if foreground_label is not None else mask > 0


def _binary_to_component_labels(binary_mask):
    labeled, num_features = ndimage.label(binary_mask.astype(np.uint8))
    return labeled.astype(np.uint32, copy=False), int(num_features)


def _import_kimimaro():
    try:
        import kimimaro
    except ImportError as exc:
        raise ImportError(
            "kimimaro is required for tubule reconstruction. "
            "Please install it before running this module."
        ) from exc
    return kimimaro


def skeletonize_binary_mask(
    binary_mask,
    resolution_xyz=(1.0, 1.0, 1.0),
    dust_threshold=0,
    fix_borders=True,
    parallel=1,
    teasar_params=None,
):
    kimimaro = _import_kimimaro()

    labels, num_components = _binary_to_component_labels(binary_mask)
    if num_components == 0:
        return {}, {"num_components": 0}

    resolution_xyz = tuple(float(v) for v in resolution_xyz)
    anisotropy_xyz = resolution_xyz
    teasar_cfg = dict(DEFAULT_TEASAR_PARAMS)
    if teasar_params:
        teasar_cfg.update(teasar_params)

    skeletons = kimimaro.skeletonize(
        labels,
        teasar_params=teasar_cfg,
        dust_threshold=int(dust_threshold),
        anisotropy=anisotropy_xyz,
        fix_branching=True,
        fix_borders=bool(fix_borders),
        progress=False,
        parallel=int(parallel),
    )

    return skeletons, {"num_components": num_components}


def _get_skeleton_radii(skeleton):
    radii = getattr(skeleton, "radii", None)
    if radii is None:
        return None
    radii = np.asarray(radii)
    if radii.size == 0:
        return None
    return radii.astype(np.float64, copy=False)


def _build_adjacency(num_vertices, edges):
    adjacency = [[] for _ in range(num_vertices)]
    for edge_index, (u, v) in enumerate(edges):
        adjacency[int(u)].append((int(v), edge_index))
        adjacency[int(v)].append((int(u), edge_index))
    return adjacency


def _edge_length(vertices, u, v):
    return float(np.linalg.norm(vertices[int(v)] - vertices[int(u)]))


def _collect_path(adjacency, edges, start_node, next_node, visited_edges):
    path_nodes = [start_node, next_node]
    path_edges = []
    prev_node = start_node
    current_node = next_node

    edge_lookup = {}
    for edge_index, (u, v) in enumerate(edges):
        edge_lookup[(int(u), int(v))] = edge_index
        edge_lookup[(int(v), int(u))] = edge_index

    while True:
        edge_index = edge_lookup[(prev_node, current_node)]
        if edge_index in visited_edges:
            break

        visited_edges.add(edge_index)
        path_edges.append(edge_index)

        current_degree = len(adjacency[current_node])
        if current_degree != 2:
            break

        candidates = [neighbor for neighbor, _ in adjacency[current_node] if neighbor != prev_node]
        if not candidates:
            break

        next_candidate = candidates[0]
        if next_candidate == start_node:
            path_nodes.append(next_candidate)
            loop_edge = edge_lookup[(current_node, next_candidate)]
            if loop_edge not in visited_edges:
                visited_edges.add(loop_edge)
                path_edges.append(loop_edge)
            break

        path_nodes.append(next_candidate)
        prev_node, current_node = current_node, next_candidate

    return path_nodes, path_edges


def _extract_branches_from_skeleton(skeleton_id, skeleton):
    vertices = np.asarray(getattr(skeleton, "vertices", np.empty((0, 3))), dtype=np.float64)
    edges = np.asarray(getattr(skeleton, "edges", np.empty((0, 2))), dtype=np.int64)
    radii = _get_skeleton_radii(skeleton)

    if vertices.size == 0 or edges.size == 0:
        return []

    adjacency = _build_adjacency(len(vertices), edges)
    degrees = np.array([len(neighbors) for neighbors in adjacency], dtype=np.int64)
    branch_nodes = set(np.where(degrees != 2)[0].tolist())
    visited_edges = set()
    branches = []
    branch_id = 0

    for node in sorted(branch_nodes):
        for neighbor, edge_index in adjacency[node]:
            if edge_index in visited_edges:
                continue

            path_nodes, path_edges = _collect_path(adjacency, edges, node, neighbor, visited_edges)
            if not path_edges:
                continue

            branches.append(
                _branch_record_from_path(
                    skeleton_id=skeleton_id,
                    branch_id=branch_id,
                    path_nodes=path_nodes,
                    path_edges=path_edges,
                    vertices=vertices,
                    radii=radii,
                )
            )
            branch_id += 1

    for edge_index, (u, v) in enumerate(edges):
        if edge_index in visited_edges:
            continue

        path_nodes, path_edges = _collect_loop(adjacency, edges, int(u), int(v), visited_edges)
        if not path_edges:
            continue

        branches.append(
            _branch_record_from_path(
                skeleton_id=skeleton_id,
                branch_id=branch_id,
                path_nodes=path_nodes,
                path_edges=path_edges,
                vertices=vertices,
                radii=radii,
                is_loop=True,
            )
        )
        branch_id += 1

    return branches


def _collect_loop(adjacency, edges, start_node, next_node, visited_edges):
    edge_lookup = {}
    for edge_index, (u, v) in enumerate(edges):
        edge_lookup[(int(u), int(v))] = edge_index
        edge_lookup[(int(v), int(u))] = edge_index

    path_nodes = [start_node, next_node]
    path_edges = []
    prev_node = start_node
    current_node = next_node

    while True:
        edge_index = edge_lookup[(prev_node, current_node)]
        if edge_index in visited_edges:
            break

        visited_edges.add(edge_index)
        path_edges.append(edge_index)

        candidates = [neighbor for neighbor, _ in adjacency[current_node] if neighbor != prev_node]
        if not candidates:
            break

        next_candidate = candidates[0]
        if next_candidate == start_node:
            closing_edge = edge_lookup[(current_node, next_candidate)]
            if closing_edge not in visited_edges:
                visited_edges.add(closing_edge)
                path_edges.append(closing_edge)
            path_nodes.append(next_candidate)
            break

        path_nodes.append(next_candidate)
        prev_node, current_node = current_node, next_candidate

    return path_nodes, path_edges


def _branch_record_from_path(
    skeleton_id,
    branch_id,
    path_nodes,
    path_edges,
    vertices,
    radii,
    is_loop=False,
):
    points = vertices[np.asarray(path_nodes[:-1] if is_loop else path_nodes, dtype=np.int64)]
    if is_loop:
        points = np.vstack([points, vertices[path_nodes[-1]]])

    segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1) if len(points) > 1 else np.array([], dtype=np.float64)
    branch_length = float(segment_lengths.sum())
    euclidean_length = float(np.linalg.norm(points[-1] - points[0])) if len(points) > 1 else 0.0
    tortuosity = float(branch_length / euclidean_length) if euclidean_length > 0 else np.nan

    radius_values = None
    if radii is not None:
        radius_values = radii[np.asarray(path_nodes[:-1] if is_loop else path_nodes, dtype=np.int64)]

    return {
        "skeleton_id": int(skeleton_id),
        "branch_id": int(branch_id),
        "start_node": int(path_nodes[0]),
        "end_node": int(path_nodes[-1]),
        "num_points": int(len(points)),
        "num_edges": int(len(path_edges)),
        "is_loop": bool(is_loop),
        "branch_length_um": branch_length,
        "euclidean_length_um": euclidean_length,
        "tortuosity": tortuosity,
        "mean_radius_um": float(np.mean(radius_values)) if radius_values is not None else np.nan,
        "max_radius_um": float(np.max(radius_values)) if radius_values is not None else np.nan,
        "min_radius_um": float(np.min(radius_values)) if radius_values is not None else np.nan,
    }


def branch_table_from_skeletons(skeletons):
    rows = []
    for skeleton_id, skeleton in skeletons.items():
        rows.extend(_extract_branches_from_skeleton(skeleton_id, skeleton))

    if not rows:
        return pd.DataFrame(
            columns=[
                "skeleton_id",
                "branch_id",
                "start_node",
                "end_node",
                "num_points",
                "num_edges",
                "is_loop",
                "branch_length_um",
                "euclidean_length_um",
                "tortuosity",
                "mean_radius_um",
                "max_radius_um",
                "min_radius_um",
            ]
        )

    return pd.DataFrame(rows)


def _reindex_branch_table(branch_table, skeleton_offset=0, extra_columns=None):
    if branch_table.empty:
        table = branch_table.copy()
    else:
        table = branch_table.copy()
        table["skeleton_id"] = table["skeleton_id"].astype(np.int64) + int(skeleton_offset)

    if extra_columns:
        for key, value in extra_columns.items():
            table[key] = value

    return table


def summarize_vessel_network(binary_mask, skeletons, branch_table, resolution_xyz=(1.0, 1.0, 1.0)):
    voxel_volume = float(np.prod(tuple(float(v) for v in resolution_xyz)))
    mask_voxels = int(np.count_nonzero(binary_mask))

    total_vertices = 0
    total_edges = 0
    total_branch_points = 0
    total_end_points = 0
    mean_radii = []

    for skeleton in skeletons.values():
        vertices = np.asarray(getattr(skeleton, "vertices", np.empty((0, 3))))
        edges = np.asarray(getattr(skeleton, "edges", np.empty((0, 2))))
        total_vertices += int(len(vertices))
        total_edges += int(len(edges))

        adjacency = _build_adjacency(len(vertices), edges) if len(vertices) > 0 else []
        degrees = np.array([len(neighbors) for neighbors in adjacency], dtype=np.int64) if adjacency else np.empty(0, dtype=np.int64)
        total_branch_points += int(np.sum(degrees >= 3))
        total_end_points += int(np.sum(degrees == 1))

        radii = _get_skeleton_radii(skeleton)
        if radii is not None:
            mean_radii.append(float(np.mean(radii)))

    branch_lengths = branch_table["branch_length_um"].to_numpy(dtype=np.float64) if not branch_table.empty else np.empty(0, dtype=np.float64)
    tortuosities = branch_table["tortuosity"].to_numpy(dtype=np.float64) if not branch_table.empty else np.empty(0, dtype=np.float64)
    valid_tortuosities = tortuosities[np.isfinite(tortuosities)]

    summary = {
        "num_skeletons": int(len(skeletons)),
        "num_branches": int(len(branch_table)),
        "num_vertices": int(total_vertices),
        "num_edges": int(total_edges),
        "num_branch_points": int(total_branch_points),
        "num_end_points": int(total_end_points),
        "mask_voxels": int(mask_voxels),
        "mask_volume_um3": float(mask_voxels * voxel_volume),
        "total_branch_length_um": float(branch_lengths.sum()) if branch_lengths.size > 0 else 0.0,
        "mean_branch_length_um": float(branch_lengths.mean()) if branch_lengths.size > 0 else 0.0,
        "median_branch_length_um": float(np.median(branch_lengths)) if branch_lengths.size > 0 else 0.0,
        "max_branch_length_um": float(branch_lengths.max()) if branch_lengths.size > 0 else 0.0,
        "mean_tortuosity": float(valid_tortuosities.mean()) if valid_tortuosities.size > 0 else np.nan,
        "mean_skeleton_radius_um": float(np.mean(mean_radii)) if mean_radii else np.nan,
    }
    return summary


def summarize_branch_table(branch_table):
    if branch_table.empty:
        return {
            "num_skeletons": 0,
            "num_branches": 0,
            "num_vertices": 0,
            "num_edges": 0,
            "num_branch_points": 0,
            "num_end_points": 0,
            "mask_voxels": 0,
            "mask_volume_um3": 0.0,
            "total_branch_length_um": 0.0,
            "mean_branch_length_um": 0.0,
            "median_branch_length_um": 0.0,
            "max_branch_length_um": 0.0,
            "mean_tortuosity": np.nan,
            "mean_skeleton_radius_um": np.nan,
        }

    lengths = branch_table["branch_length_um"].to_numpy(dtype=np.float64)
    tortuosities = branch_table["tortuosity"].to_numpy(dtype=np.float64)
    valid_tortuosities = tortuosities[np.isfinite(tortuosities)]
    radius_values = branch_table["mean_radius_um"].to_numpy(dtype=np.float64)
    valid_radii = radius_values[np.isfinite(radius_values)]

    return {
        "num_skeletons": int(branch_table["skeleton_id"].nunique()),
        "num_branches": int(len(branch_table)),
        "num_vertices": None,
        "num_edges": None,
        "num_branch_points": None,
        "num_end_points": None,
        "mask_voxels": None,
        "mask_volume_um3": None,
        "total_branch_length_um": float(lengths.sum()),
        "mean_branch_length_um": float(lengths.mean()),
        "median_branch_length_um": float(np.median(lengths)),
        "max_branch_length_um": float(lengths.max()),
        "mean_tortuosity": float(valid_tortuosities.mean()) if valid_tortuosities.size > 0 else np.nan,
        "mean_skeleton_radius_um": float(valid_radii.mean()) if valid_radii.size > 0 else np.nan,
    }


def _write_summary_json(summary, output_path):
    sanitized = {}
    for key, value in summary.items():
        if isinstance(value, (np.floating, float)) and np.isnan(value):
            sanitized[key] = None
        elif isinstance(value, np.generic):
            sanitized[key] = value.item()
        else:
            sanitized[key] = value

    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(sanitized, handle, indent=2, ensure_ascii=False)


def analyze_binary_mask_zarr(
    mask_zarr_path,
    output_dir,
    dataset_name="0",
    resolution_xyz=(1.0, 1.0, 1.0),
    foreground_label=1,
    roi=None,
    dust_threshold=0,
    fix_borders=True,
    parallel=1,
    teasar_params=None,
):
    resolution_xyz = parse_resolution_xyz(resolution_xyz)
    mask_zarr = open_zarr_dataset(mask_zarr_path, dataset_name=dataset_name)
    binary_mask = _extract_binary_mask(mask_zarr, roi=roi, foreground_label=foreground_label)

    skeletons, meta = skeletonize_binary_mask(
        binary_mask=binary_mask,
        resolution_xyz=resolution_xyz,
        dust_threshold=dust_threshold,
        fix_borders=fix_borders,
        parallel=parallel,
        teasar_params=teasar_params,
    )

    branch_table = branch_table_from_skeletons(skeletons)
    summary = summarize_vessel_network(
        binary_mask=binary_mask,
        skeletons=skeletons,
        branch_table=branch_table,
        resolution_xyz=resolution_xyz,
    )
    summary["connected_components"] = meta["num_components"]

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    branch_csv_path = output_root / "vessel_branch_metrics.csv"
    summary_json_path = output_root / "vessel_network_summary.json"
    branch_table.to_csv(branch_csv_path, index=False)
    _write_summary_json(summary, summary_json_path)

    return {
        "summary": summary,
        "branch_table": branch_table,
        "branch_csv_path": branch_csv_path,
        "summary_json_path": summary_json_path,
    }


def analyze_binary_mask_zarr_test_mode(
    mask_zarr_path,
    output_dir,
    dataset_name="0",
    resolution_xyz=(1.0, 1.0, 1.0),
    foreground_label=1,
    dust_threshold=0,
    fix_borders=True,
    parallel=1,
    teasar_params=None,
):
    resolution_xyz = parse_resolution_xyz(resolution_xyz)
    mask_zarr = open_zarr_dataset(mask_zarr_path, dataset_name=dataset_name)
    existing_chunks = list_existing_chunk_indices(mask_zarr)

    if not existing_chunks:
        raise ValueError("No physical chunks found in the input mask Zarr store.")

    all_branch_tables = []
    chunk_rows = []
    skeleton_offset = 0
    total_mask_voxels = 0
    total_components = 0
    chunks = mask_zarr.chunks
    shape = mask_zarr.shape

    for chunk_index in existing_chunks:
        chunk_slices = chunk_index_to_slices(chunk_index, chunks, shape)
        binary_mask = _extract_binary_mask(mask_zarr, roi=chunk_slices, foreground_label=foreground_label)
        mask_voxels = int(np.count_nonzero(binary_mask))
        total_mask_voxels += mask_voxels

        if mask_voxels == 0:
            chunk_rows.append(
                {
                    "chunk_index": ".".join(str(v) for v in chunk_index),
                    "chunk_start_zyx": ",".join(str(s.start) for s in chunk_slices),
                    "chunk_stop_zyx": ",".join(str(s.stop) for s in chunk_slices),
                    "mask_voxels": 0,
                    "connected_components": 0,
                    "num_skeletons": 0,
                    "num_branches": 0,
                    "total_branch_length_um": 0.0,
                }
            )
            continue

        skeletons, meta = skeletonize_binary_mask(
            binary_mask=binary_mask,
            resolution_xyz=resolution_xyz,
            dust_threshold=dust_threshold,
            fix_borders=fix_borders,
            parallel=parallel,
            teasar_params=teasar_params,
        )
        total_components += int(meta["num_components"])

        branch_table = branch_table_from_skeletons(skeletons)
        branch_table = _reindex_branch_table(
            branch_table,
            skeleton_offset=skeleton_offset,
            extra_columns={
                "chunk_index": ".".join(str(v) for v in chunk_index),
                "chunk_start_zyx": ",".join(str(s.start) for s in chunk_slices),
                "chunk_stop_zyx": ",".join(str(s.stop) for s in chunk_slices),
            },
        )
        all_branch_tables.append(branch_table)

        chunk_summary = summarize_vessel_network(
            binary_mask=binary_mask,
            skeletons=skeletons,
            branch_table=branch_table,
            resolution_xyz=resolution_xyz,
        )
        chunk_summary["chunk_index"] = ".".join(str(v) for v in chunk_index)
        chunk_summary["chunk_start_zyx"] = ",".join(str(s.start) for s in chunk_slices)
        chunk_summary["chunk_stop_zyx"] = ",".join(str(s.stop) for s in chunk_slices)
        chunk_summary["connected_components"] = int(meta["num_components"])
        chunk_rows.append(chunk_summary)

        skeleton_offset += len(skeletons)

    combined_branch_table = (
        pd.concat(all_branch_tables, ignore_index=True)
        if all_branch_tables
        else branch_table_from_skeletons({})
    )
    summary = summarize_branch_table(combined_branch_table)
    summary["mode"] = "test_chunkwise"
    summary["processed_chunks"] = int(len(existing_chunks))
    summary["connected_components"] = int(total_components)
    summary["mask_voxels"] = int(total_mask_voxels)
    summary["mask_volume_um3"] = float(total_mask_voxels * np.prod(tuple(float(v) for v in resolution_xyz)))

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    branch_csv_path = output_root / "vessel_branch_metrics.csv"
    chunk_csv_path = output_root / "vessel_chunk_metrics.csv"
    summary_json_path = output_root / "vessel_network_summary.json"

    combined_branch_table.to_csv(branch_csv_path, index=False)
    pd.DataFrame(chunk_rows).to_csv(chunk_csv_path, index=False)
    _write_summary_json(summary, summary_json_path)

    return {
        "summary": summary,
        "branch_table": combined_branch_table,
        "branch_csv_path": branch_csv_path,
        "chunk_csv_path": chunk_csv_path,
        "summary_json_path": summary_json_path,
    }


def build_argparser():
    parser = argparse.ArgumentParser(description="Reconstruct vessel network metrics from a binary mask Zarr using kimimaro.")
    parser.add_argument("--mask_zarr", required=True, help="Path to input binary mask Zarr")
    parser.add_argument("--output_dir", required=True, help="Directory for CSV and JSON outputs")
    parser.add_argument("--dataset_name", default="0", help="Dataset name inside the Zarr group")
    parser.add_argument("--resolution_xyz", default="1,1,1", help="Voxel size in microns as x,y,z")
    parser.add_argument("--foreground_label", type=int, default=1, help="Foreground label value in the mask")
    parser.add_argument("--roi", default="", help="Optional ROI JSON string, e.g. {\"z\":[0,100],\"y\":[0,256],\"x\":[0,256]}")
    parser.add_argument("--dust_threshold", type=int, default=0, help="Minimum component size for kimimaro")
    parser.add_argument("--parallel", type=int, default=1, help="Kimimaro worker count")
    parser.add_argument("--no_fix_borders", action="store_true", help="Disable border fixing in kimimaro")
    parser.add_argument(
        "--test",
        action="store_true",
        help="Smoke-test mode: only process chunks that physically exist in the input mask store",
    )
    return parser


def main():
    parser = build_argparser()
    args = parser.parse_args()

    roi = parse_roi(args.roi) if args.roi else None
    if args.test:
        result = analyze_binary_mask_zarr_test_mode(
            mask_zarr_path=args.mask_zarr,
            output_dir=args.output_dir,
            dataset_name=args.dataset_name,
            resolution_xyz=args.resolution_xyz,
            foreground_label=args.foreground_label,
            dust_threshold=args.dust_threshold,
            fix_borders=not args.no_fix_borders,
            parallel=args.parallel,
        )
    else:
        result = analyze_binary_mask_zarr(
            mask_zarr_path=args.mask_zarr,
            output_dir=args.output_dir,
            dataset_name=args.dataset_name,
            resolution_xyz=args.resolution_xyz,
            foreground_label=args.foreground_label,
            roi=roi,
            dust_threshold=args.dust_threshold,
            fix_borders=not args.no_fix_borders,
            parallel=args.parallel,
        )

    print("Vessel reconstruction completed.")
    print(f"Summary JSON: {result['summary_json_path']}")
    print(f"Branch CSV: {result['branch_csv_path']}")
    if "chunk_csv_path" in result:
        print(f"Chunk CSV: {result['chunk_csv_path']}")


if __name__ == "__main__":
    main()
