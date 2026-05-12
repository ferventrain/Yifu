import argparse
import concurrent.futures
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import zarr
from scipy import ndimage

try:
    from pipeline_modules.utils.errors import ErrorCode, PipelineError
    from pipeline_modules.utils.run_manifest import write_run_manifest
except ImportError:  # running the file directly without project root on sys.path
    PipelineError = None  # type: ignore[assignment,misc]
    ErrorCode = None  # type: ignore[assignment]
    write_run_manifest = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


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


def resolution_xyz_to_zyx(resolution_xyz):
    resolution_xyz = parse_resolution_xyz(resolution_xyz)
    return (float(resolution_xyz[2]), float(resolution_xyz[1]), float(resolution_xyz[0]))


def parse_triplet_int(text):
    if isinstance(text, (tuple, list)):
        if len(text) != 3:
            raise ValueError(f"Expected 3 integers, got: {text}")
        return tuple(int(v) for v in text)

    parts = [part.strip() for part in str(text).split(",") if part.strip()]
    if len(parts) != 3:
        raise ValueError(f"Expected 3 comma-separated integers, got: {text}")
    return tuple(int(part) for part in parts)


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


def iter_all_chunk_indices(shape, chunks):
    grid_shape = tuple((int(dim) + int(chunk) - 1) // int(chunk) for dim, chunk in zip(shape, chunks))
    for z in range(grid_shape[0]):
        for y in range(grid_shape[1]):
            for x in range(grid_shape[2]):
                yield (z, y, x)


def expand_slices(core_slices, halo_zyx, shape):
    expanded = []
    for axis, (core_slice, halo, dim) in enumerate(zip(core_slices, halo_zyx, shape)):
        start = max(0, int(core_slice.start) - int(halo))
        stop = min(int(dim), int(core_slice.stop) + int(halo))
        expanded.append(slice(start, stop))
    return tuple(expanded)


def _extract_binary_mask(mask_zarr, roi=None, foreground_label=1):
    if roi is None:
        mask = np.asarray(mask_zarr[:])
    else:
        mask = np.asarray(mask_zarr[roi])

    if mask.ndim != 3:
        raise ValueError(f"Expected a 3D mask volume, got shape={mask.shape}")

    return mask == foreground_label if foreground_label is not None else mask > 0


def _make_chunk_metadata(chunk_index, chunk_slices, expanded_slices):
    return {
        "chunk_index": ".".join(str(v) for v in chunk_index),
        "chunk_start_zyx": ",".join(str(s.start) for s in chunk_slices),
        "chunk_stop_zyx": ",".join(str(s.stop) for s in chunk_slices),
        "expanded_start_zyx": ",".join(str(s.start) for s in expanded_slices),
        "expanded_stop_zyx": ",".join(str(s.stop) for s in expanded_slices),
    }


def _process_chunk_task(task):
    resolution_xyz = parse_resolution_xyz(task["resolution_xyz"])
    mask_zarr = open_zarr_dataset(task["mask_zarr_path"], dataset_name=task["dataset_name"])
    chunk_index = tuple(task["chunk_index"])
    chunks = tuple(task["chunks"])
    shape = tuple(task["shape"])
    halo_zyx = tuple(task["halo_zyx"])

    chunk_slices = chunk_index_to_slices(chunk_index, chunks, shape)
    expanded_slices = expand_slices(chunk_slices, halo_zyx, shape)
    metadata = _make_chunk_metadata(chunk_index, chunk_slices, expanded_slices)

    core_binary_mask = _extract_binary_mask(
        mask_zarr,
        roi=chunk_slices,
        foreground_label=task["foreground_label"],
    )
    core_mask_voxels = int(np.count_nonzero(core_binary_mask))

    if core_mask_voxels == 0:
        chunk_summary = {
            **metadata,
            "mask_voxels": 0,
            "expanded_mask_voxels": 0,
            "connected_components": 0,
            "num_skeletons": 0,
            "num_branches": 0,
            "total_branch_length_um": 0.0,
        }
        return {
            "chunk_index": chunk_index,
            "core_mask_voxels": 0,
            "connected_components": 0,
            "branch_table": branch_table_from_skeletons({}),
            "vertex_table": pd.DataFrame(),
            "edge_table": pd.DataFrame(),
            "chunk_summary": chunk_summary,
            "num_skeletons": 0,
        }

    binary_mask = _extract_binary_mask(
        mask_zarr,
        roi=expanded_slices,
        foreground_label=task["foreground_label"],
    )
    expanded_mask_voxels = int(np.count_nonzero(binary_mask))

    skeletons, meta = skeletonize_binary_mask(
        binary_mask=binary_mask,
        resolution_xyz=resolution_xyz,
        dust_threshold=task["dust_threshold"],
        fix_borders=task["fix_borders"],
        parallel=task["kimimaro_parallel"],
        teasar_params=task["teasar_params"],
    )

    branch_table = branch_table_from_skeletons(skeletons)
    branch_table = _reindex_branch_table(branch_table, skeleton_offset=0, extra_columns=metadata)

    if task["save_skeleton"]:
        vertex_table, edge_table = skeleton_tables_from_skeletons(
            skeletons,
            extra_columns=metadata,
            offset_zyx_um=np.asarray([s.start for s in expanded_slices], dtype=np.float64)
            * np.asarray(resolution_xyz_to_zyx(resolution_xyz), dtype=np.float64),
        )
        vertex_table, edge_table = _annotate_core_membership(
            vertex_table,
            edge_table,
            core_slices=chunk_slices,
            resolution_xyz=resolution_xyz,
        )
        vertex_table, edge_table = _filter_skeleton_tables_to_core(vertex_table, edge_table)
    else:
        vertex_table, edge_table = pd.DataFrame(), pd.DataFrame()

    chunk_summary = summarize_vessel_network(
        binary_mask=core_binary_mask,
        skeletons=skeletons,
        branch_table=branch_table,
        resolution_xyz=resolution_xyz,
    )
    chunk_summary.update(metadata)
    chunk_summary["mask_voxels"] = core_mask_voxels
    chunk_summary["expanded_mask_voxels"] = expanded_mask_voxels
    chunk_summary["connected_components"] = int(meta["num_components"])

    return {
        "chunk_index": chunk_index,
        "core_mask_voxels": core_mask_voxels,
        "connected_components": int(meta["num_components"]),
        "branch_table": branch_table,
        "vertex_table": vertex_table,
        "edge_table": edge_table,
        "chunk_summary": chunk_summary,
        "num_skeletons": len(skeletons),
    }


def _binary_to_component_labels(binary_mask):
    labeled, num_features = ndimage.label(binary_mask.astype(np.uint8))
    # Keep a signed integer dtype for kimimaro. Unsigned labels can trigger
    # dtype casting issues when kimimaro applies ROI offsets to skeleton vertices.
    if num_features <= np.iinfo(np.int32).max:
        return labeled.astype(np.int32, copy=False), int(num_features)
    return labeled.astype(np.int64, copy=False), int(num_features)


def _import_kimimaro():
    try:
        import kimimaro
    except ImportError as exc:
        raise ImportError(
            "kimimaro is required for tubule reconstruction. "
            "Please install it before running this module."
        ) from exc
    _patch_kimimaro_roi_cast_bug(kimimaro)
    return kimimaro


def _patch_kimimaro_roi_cast_bug(kimimaro):
    """Patch kimimaro ROI offset handling to avoid uint32/int64 casting errors.

    Some kimimaro builds return skeleton.vertices as uint32 inside
    intake.skeletonize_subset. When ROI offsets are applied with
    `skeleton.vertices += roi.minpt`, NumPy can raise a casting error.
    We patch the subset function to promote both operands to int64 first.
    """
    intake = kimimaro.intake
    if getattr(intake, "_yifu_roi_cast_patch", False):
        return

    def patched_skeletonize_subset(
        all_dbf,
        cc_labels,
        voxel_graph,
        remapping,
        teasar_params,
        anisotropy,
        all_slices,
        border_targets,
        extra_targets_before,
        extra_targets_after,
        progress,
        fix_borders,
        fix_branching,
        cc_segids,
    ):
        skeletons = intake.defaultdict(list)

        with intake.tqdm(cc_segids, disable=(not progress), desc="Skeletonizing Labels") as pbar:
            for segid in pbar:
                pbar.set_postfix(label=str(remapping[segid]))

                slices = all_slices[segid - 1]
                if slices is None:
                    continue

                roi = intake.Bbox.from_slices(slices)
                if roi.volume() <= 1:
                    continue

                labels = cc_labels[slices]
                labels = labels == segid
                dbf = (labels * all_dbf[slices]).astype(np.float32)
                cropped_voxel_graph = voxel_graph[slices] if voxel_graph is not None else None

                manual_targets_before = []
                manual_targets_after = []
                root = None

                def translate_to_roi(targets):
                    targets = np.array(targets, dtype=np.int64)
                    targets -= roi.minpt.astype(np.int64)
                    return targets.tolist()

                if len(border_targets[segid]) > 0:
                    manual_targets_before = translate_to_roi(border_targets[segid])
                    root = manual_targets_before.pop()

                if segid in extra_targets_before and len(extra_targets_before[segid]) > 0:
                    manual_targets_before.extend(translate_to_roi(extra_targets_before[segid]))

                if segid in extra_targets_after and len(extra_targets_after[segid]) > 0:
                    manual_targets_after.extend(translate_to_roi(extra_targets_after[segid]))

                skeleton = kimimaro.trace.trace(
                    labels,
                    dbf,
                    anisotropy=anisotropy,
                    fix_branching=fix_branching,
                    manual_targets_before=manual_targets_before,
                    manual_targets_after=manual_targets_after,
                    root=root,
                    voxel_graph=cropped_voxel_graph,
                    **teasar_params,
                )

                if skeleton.empty():
                    continue

                skeleton.vertices = skeleton.vertices.astype(np.int64, copy=False)
                skeleton.vertices += roi.minpt.astype(np.int64)

                orig_segid = remapping[segid]
                skeleton.id = orig_segid
                skeleton.vertices = skeleton.vertices.astype(np.float32, copy=False)
                skeleton.vertices *= anisotropy
                skeletons[orig_segid].append(skeleton)

        return intake.merge(skeletons)

    intake.skeletonize_subset = patched_skeletonize_subset
    intake._yifu_roi_cast_patch = True


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

    anisotropy_zyx = resolution_xyz_to_zyx(resolution_xyz)
    teasar_cfg = dict(DEFAULT_TEASAR_PARAMS)
    if teasar_params:
        teasar_cfg.update(teasar_params)

    skeletons = kimimaro.skeletonize(
        labels,
        teasar_params=teasar_cfg,
        dust_threshold=int(dust_threshold),
        anisotropy=anisotropy_zyx,
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


def _compute_branch_depths(branches, degrees, vertices):
    if not branches:
        return []

    branch_nodes = set()
    for branch in branches:
        branch_nodes.add(int(branch["start_node"]))
        branch_nodes.add(int(branch["end_node"]))

    node_graph = {node: set() for node in branch_nodes}
    for branch in branches:
        if branch.get("is_loop", False):
            continue
        start_node = int(branch["start_node"])
        end_node = int(branch["end_node"])
        node_graph.setdefault(start_node, set()).add(end_node)
        node_graph.setdefault(end_node, set()).add(start_node)

    endpoint_nodes = [node for node in branch_nodes if int(degrees[node]) == 1]
    if endpoint_nodes:
        root_node = min(endpoint_nodes, key=lambda node: float(vertices[node][0]))
    else:
        root_node = min(branch_nodes, key=lambda node: float(vertices[node][0]))

    node_depths = {int(root_node): 0}
    queue = [int(root_node)]
    queue_index = 0

    while queue_index < len(queue):
        current = queue[queue_index]
        queue_index += 1
        for neighbor in node_graph.get(current, ()):
            if neighbor in node_depths:
                continue
            node_depths[neighbor] = node_depths[current] + 1
            queue.append(neighbor)

    branch_depths = []
    for branch in branches:
        if branch.get("is_loop", False):
            branch_depths.append(np.nan)
            continue

        start_depth = node_depths.get(int(branch["start_node"]))
        end_depth = node_depths.get(int(branch["end_node"]))
        if start_depth is None or end_depth is None:
            branch_depths.append(np.nan)
        else:
            branch_depths.append(int(max(start_depth, end_depth)))

    return branch_depths


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

    branch_depths = _compute_branch_depths(branches, degrees, vertices)
    for branch, branch_depth in zip(branches, branch_depths):
        start_degree = int(degrees[int(branch["start_node"])])
        end_degree = int(degrees[int(branch["end_node"])])
        branch["start_degree"] = start_degree
        branch["end_degree"] = end_degree
        branch["branch_depth"] = branch_depth
        branch["is_terminal_branch"] = bool((start_degree == 1) or (end_degree == 1))
        branch["is_branch_to_branch"] = bool((start_degree >= 3) and (end_degree >= 3))
        branch["is_root_branch"] = bool(np.isfinite(branch_depth) and int(branch_depth) == 1)

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
                "start_degree",
                "end_degree",
                "branch_depth",
                "is_terminal_branch",
                "is_branch_to_branch",
                "is_root_branch",
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


def skeleton_tables_from_skeletons(skeletons, extra_columns=None, offset_zyx_um=None):
    vertex_rows = []
    edge_rows = []
    offset = np.asarray(offset_zyx_um if offset_zyx_um is not None else (0.0, 0.0, 0.0), dtype=np.float64)

    for skeleton_id, skeleton in skeletons.items():
        vertices = np.asarray(getattr(skeleton, "vertices", np.empty((0, 3))), dtype=np.float64)
        edges = np.asarray(getattr(skeleton, "edges", np.empty((0, 2))), dtype=np.int64)
        radii = _get_skeleton_radii(skeleton)

        for node_id, point in enumerate(vertices):
            global_point = point + offset
            row = {
                "skeleton_id": int(skeleton_id),
                "node_id": int(node_id),
                "z_um": float(global_point[0]),
                "y_um": float(global_point[1]),
                "x_um": float(global_point[2]),
                "radius_um": float(radii[node_id]) if radii is not None else np.nan,
            }
            if extra_columns:
                row.update(extra_columns)
            vertex_rows.append(row)

        for edge_id, (source_node, target_node) in enumerate(edges):
            source_point = vertices[int(source_node)] + offset
            target_point = vertices[int(target_node)] + offset
            row = {
                "skeleton_id": int(skeleton_id),
                "edge_id": int(edge_id),
                "source_node": int(source_node),
                "target_node": int(target_node),
                "source_z_um": float(source_point[0]),
                "source_y_um": float(source_point[1]),
                "source_x_um": float(source_point[2]),
                "target_z_um": float(target_point[0]),
                "target_y_um": float(target_point[1]),
                "target_x_um": float(target_point[2]),
                "edge_length_um": float(np.linalg.norm(target_point - source_point)),
            }
            if extra_columns:
                row.update(extra_columns)
            edge_rows.append(row)

    vertex_table = pd.DataFrame(vertex_rows)
    edge_table = pd.DataFrame(edge_rows)
    return vertex_table, edge_table


def _reindex_skeleton_tables(vertex_table, edge_table, skeleton_offset=0):
    if not vertex_table.empty:
        vertex_table = vertex_table.copy()
        vertex_table["skeleton_id"] = vertex_table["skeleton_id"].astype(np.int64) + int(skeleton_offset)
    if not edge_table.empty:
        edge_table = edge_table.copy()
        edge_table["skeleton_id"] = edge_table["skeleton_id"].astype(np.int64) + int(skeleton_offset)
    return vertex_table, edge_table


def _write_skeleton_tables(vertex_table, edge_table, output_root):
    vertex_csv_path = output_root / "skeleton_vertices.csv"
    edge_csv_path = output_root / "skeleton_edges.csv"
    vertex_table.to_csv(vertex_csv_path, index=False)
    edge_table.to_csv(edge_csv_path, index=False)
    return vertex_csv_path, edge_csv_path


def _write_single_swc(vertex_table, edge_table, swc_path):
    if vertex_table.empty:
        return

    vertex_table = vertex_table.copy().sort_values(["node_id"]).reset_index(drop=True)
    node_ids = set(vertex_table["node_id"].astype(int).tolist())

    # Build undirected adjacency from edge table
    adj = {nid: [] for nid in node_ids}
    if not edge_table.empty:
        for row in edge_table.itertuples(index=False):
            src, tgt = int(row.source_node), int(row.target_node)
            if src in node_ids and tgt in node_ids:
                adj[src].append(tgt)
                adj[tgt].append(src)

    # BFS from root to assign parent for every reachable node
    parent_map = {}
    visited = set()

    unvisited = set(node_ids)
    while unvisited:
        # Pick root: prefer node with degree 1 (endpoint), else smallest id
        root = None
        for nid in sorted(unvisited):
            if len(adj.get(nid, [])) == 1:
                root = nid
                break
        if root is None:
            root = min(unvisited)

        parent_map[root] = -1
        queue = [root]
        visited.add(root)
        unvisited.discard(root)

        while queue:
            current = queue.pop(0)
            for neighbor in adj[current]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    unvisited.discard(neighbor)
                    parent_map[neighbor] = current
                    queue.append(neighbor)

    lines = ["# id type x y z radius parent"]
    for row in vertex_table.itertuples(index=False):
        node_id = int(row.node_id)
        swc_id = node_id + 1
        parent_node = parent_map.get(node_id, -1)
        parent_id = parent_node + 1 if parent_node >= 0 else -1
        radius = float(row.radius_um) if not pd.isna(row.radius_um) else 1.0

        lines.append(
            f"{swc_id} 0 {float(row.x_um):.6f} {float(row.y_um):.6f} "
            f"{float(row.z_um):.6f} {radius:.6f} {parent_id}"
        )

    swc_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_swc_files(vertex_table, edge_table, output_root):
    swc_dir = output_root / "swc"
    swc_dir.mkdir(parents=True, exist_ok=True)

    if vertex_table.empty:
        return swc_dir, []

    edge_table = _normalize_edge_table_schema(edge_table)
    written_paths = []

    for skeleton_id, skeleton_vertices in vertex_table.groupby("skeleton_id"):
        if edge_table.empty:
            skeleton_edges = edge_table.iloc[0:0].copy()
        else:
            skeleton_edges = edge_table.loc[
                (edge_table["source_skeleton_id"].astype(int) == int(skeleton_id))
                | (edge_table["target_skeleton_id"].astype(int) == int(skeleton_id))
                | (edge_table["skeleton_id"].astype(int) == int(skeleton_id))
            ].copy()

        swc_path = swc_dir / f"skeleton_{int(skeleton_id):06d}.swc"
        _write_single_swc(skeleton_vertices, skeleton_edges, swc_path)
        written_paths.append(swc_path)

    return swc_dir, written_paths


def _annotate_core_membership(vertex_table, edge_table, core_slices, resolution_xyz):
    if vertex_table.empty and edge_table.empty:
        return vertex_table, edge_table

    resolution_zyx = np.asarray(resolution_xyz_to_zyx(resolution_xyz), dtype=np.float64)
    core_start_um = np.asarray([s.start for s in core_slices], dtype=np.float64) * resolution_zyx
    core_stop_um = np.asarray([s.stop for s in core_slices], dtype=np.float64) * resolution_zyx

    if not vertex_table.empty:
        vertex_table = vertex_table.copy()
        coords = vertex_table[["z_um", "y_um", "x_um"]].to_numpy(dtype=np.float64)
        in_core = np.all((coords >= core_start_um) & (coords < core_stop_um), axis=1)
        z_min = np.isclose(coords[:, 0], core_start_um[0], atol=resolution_zyx[0] * 0.5)
        z_max = np.isclose(coords[:, 0], core_stop_um[0] - resolution_zyx[0], atol=resolution_zyx[0] * 0.5)
        y_min = np.isclose(coords[:, 1], core_start_um[1], atol=resolution_zyx[1] * 0.5)
        y_max = np.isclose(coords[:, 1], core_stop_um[1] - resolution_zyx[1], atol=resolution_zyx[1] * 0.5)
        x_min = np.isclose(coords[:, 2], core_start_um[2], atol=resolution_zyx[2] * 0.5)
        x_max = np.isclose(coords[:, 2], core_stop_um[2] - resolution_zyx[2], atol=resolution_zyx[2] * 0.5)
        touches_boundary = z_min | z_max | y_min | y_max | x_min | x_max
        vertex_table["in_core"] = in_core
        vertex_table["touches_core_boundary"] = touches_boundary
        vertex_table["touch_z_min"] = z_min
        vertex_table["touch_z_max"] = z_max
        vertex_table["touch_y_min"] = y_min
        vertex_table["touch_y_max"] = y_max
        vertex_table["touch_x_min"] = x_min
        vertex_table["touch_x_max"] = x_max

    if not edge_table.empty:
        edge_table = edge_table.copy()
        source = edge_table[["source_z_um", "source_y_um", "source_x_um"]].to_numpy(dtype=np.float64)
        target = edge_table[["target_z_um", "target_y_um", "target_x_um"]].to_numpy(dtype=np.float64)
        midpoint = (source + target) / 2.0
        midpoint_in_core = np.all((midpoint >= core_start_um) & (midpoint < core_stop_um), axis=1)
        touches_boundary = np.any(
            np.isclose(source, core_start_um[None, :], atol=resolution_zyx[None, :] * 0.5)
            | np.isclose(source, core_stop_um[None, :] - resolution_zyx[None, :], atol=resolution_zyx[None, :] * 0.5)
            | np.isclose(target, core_start_um[None, :], atol=resolution_zyx[None, :] * 0.5)
            | np.isclose(target, core_stop_um[None, :] - resolution_zyx[None, :], atol=resolution_zyx[None, :] * 0.5),
            axis=1,
        )
        edge_table["midpoint_in_core"] = midpoint_in_core
        edge_table["touches_core_boundary"] = touches_boundary

    return vertex_table, edge_table


def _filter_skeleton_tables_to_core(vertex_table, edge_table):
    if edge_table.empty:
        return vertex_table.iloc[0:0].copy() if not vertex_table.empty else pd.DataFrame(), edge_table

    keep_edges = (
        edge_table["midpoint_in_core"].astype(bool)
        if "midpoint_in_core" in edge_table.columns
        else np.ones(len(edge_table), dtype=bool)
    )
    filtered_edges = edge_table.loc[keep_edges].copy()

    if vertex_table.empty or filtered_edges.empty:
        return pd.DataFrame(columns=vertex_table.columns if not vertex_table.empty else []), filtered_edges

    keep_pairs = set(zip(filtered_edges["skeleton_id"].tolist(), filtered_edges["source_node"].tolist()))
    keep_pairs.update(zip(filtered_edges["skeleton_id"].tolist(), filtered_edges["target_node"].tolist()))
    keep_vertex_mask = [
        (int(row.skeleton_id), int(row.node_id)) in keep_pairs
        for row in vertex_table.itertuples(index=False)
    ]
    filtered_vertices = vertex_table.loc[keep_vertex_mask].copy()
    return filtered_vertices, filtered_edges


def _parse_chunk_index_text(chunk_index_text):
    return tuple(int(part) for part in str(chunk_index_text).split("."))


def _compute_vertex_degrees_from_edges(vertex_table, edge_table):
    if vertex_table.empty:
        return vertex_table

    vertex_table = vertex_table.copy()
    degree_map = {}

    if not edge_table.empty:
        for row in edge_table.itertuples(index=False):
            degree_map[(int(row.skeleton_id), int(row.source_node))] = degree_map.get((int(row.skeleton_id), int(row.source_node)), 0) + 1
            degree_map[(int(row.skeleton_id), int(row.target_node))] = degree_map.get((int(row.skeleton_id), int(row.target_node)), 0) + 1

    vertex_table["degree"] = [
        degree_map.get((int(row.skeleton_id), int(row.node_id)), 0)
        for row in vertex_table.itertuples(index=False)
    ]
    return vertex_table


def _normalize_edge_table_schema(edge_table):
    edge_table = edge_table.copy()
    if "source_skeleton_id" not in edge_table.columns:
        edge_table["source_skeleton_id"] = edge_table["skeleton_id"]
    if "target_skeleton_id" not in edge_table.columns:
        edge_table["target_skeleton_id"] = edge_table["skeleton_id"]
    if "is_stitch" not in edge_table.columns:
        edge_table["is_stitch"] = False
    return edge_table


# ---------------------------------------------------------------------------
# Skeleton graph postprocessing: merge nearby branch points + prune short spurs
# ---------------------------------------------------------------------------


def _build_graph_from_tables(vertex_table, edge_table):
    """Build mutable adjacency structures from vertex/edge DataFrames.

    Returns
    -------
    coords : dict[(skel_id, node_id)] -> np.array([z, y, x])
    radii  : dict[(skel_id, node_id)] -> float
    adj    : dict[(skel_id, node_id)] -> set[(skel_id, node_id)]
    """
    coords = {}
    radii = {}
    adj = {}

    for row in vertex_table.itertuples(index=False):
        key = (int(row.skeleton_id), int(row.node_id))
        coords[key] = np.array([row.z_um, row.y_um, row.x_um], dtype=np.float64)
        radii[key] = float(row.radius_um) if hasattr(row, "radius_um") and not pd.isna(row.radius_um) else 0.0
        adj[key] = set()

    for row in edge_table.itertuples(index=False):
        sk = int(row.skeleton_id)
        src = (sk, int(row.source_node))
        tgt = (sk, int(row.target_node))
        if src not in adj:
            adj[src] = set()
        if tgt not in adj:
            adj[tgt] = set()
        adj[src].add(tgt)
        adj[tgt].add(src)

    return coords, radii, adj


def _merge_nearby_branch_points(coords, radii, adj, distance_um):
    """Merge branch points (degree >= 3) that are within distance_um of each other.

    Uses greedy clustering within each connected component of branch points.
    Returns the number of merges performed.
    """
    if distance_um <= 0:
        return 0

    bp_keys = [k for k, neighbors in adj.items() if len(neighbors) >= 3]
    if not bp_keys:
        return 0

    from scipy.spatial import cKDTree

    bp_coords_arr = np.array([coords[k] for k in bp_keys], dtype=np.float64)
    tree = cKDTree(bp_coords_arr)
    pairs = tree.query_pairs(r=distance_um)

    if not pairs:
        return 0

    # Union-find for clustering
    parent = list(range(len(bp_keys)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    # Only merge within same skeleton
    for i, j in pairs:
        if bp_keys[i][0] == bp_keys[j][0]:
            union(i, j)

    clusters = {}
    for i in range(len(bp_keys)):
        root = find(i)
        clusters.setdefault(root, []).append(i)

    num_merges = 0
    for members in clusters.values():
        if len(members) < 2:
            continue

        member_keys = [bp_keys[m] for m in members]
        degrees = [len(adj[k]) for k in member_keys]
        best_idx = int(np.argmax(degrees))
        representative = member_keys[best_idx]

        # Update representative coordinate to cluster centroid
        centroid = np.mean([coords[k] for k in member_keys], axis=0)
        coords[representative] = centroid

        for k in member_keys:
            if k == representative:
                continue

            # Rewire all neighbors of k to representative
            for neighbor in list(adj[k]):
                adj[neighbor].discard(k)
                if neighbor != representative:
                    adj[neighbor].add(representative)
                    adj[representative].add(neighbor)

            # Remove k from graph
            del adj[k]
            del coords[k]
            del radii[k]
            num_merges += 1

    # Remove self-loops
    for k in list(adj.keys()):
        adj[k].discard(k)

    return num_merges


def _prune_short_spurs(coords, adj, max_length_um):
    """Iteratively remove terminal branches shorter than max_length_um.

    A spur is a path from a degree-1 node to the nearest degree!=2 node
    whose total length <= max_length_um.

    Returns the number of spur branches pruned.
    """
    if max_length_um <= 0:
        return 0

    total_pruned = 0

    while True:
        pruned_this_pass = 0
        terminals = [k for k, neighbors in adj.items() if len(neighbors) == 1]

        for terminal in terminals:
            if terminal not in adj or len(adj[terminal]) != 1:
                continue

            # Trace path from terminal until we hit a non-degree-2 node
            path = [terminal]
            current = terminal
            prev = None
            length = 0.0

            while True:
                neighbors = [n for n in adj[current] if n != prev]
                if not neighbors:
                    break
                next_node = neighbors[0]
                length += float(np.linalg.norm(coords[next_node] - coords[current]))
                if length > max_length_um:
                    break
                path.append(next_node)
                if len(adj[next_node]) != 2:
                    # Reached a junction or another terminal
                    break
                prev = current
                current = next_node

            # Only prune if: path ends at a junction (degree >= 3) and length <= threshold
            if length > max_length_um:
                continue
            if len(path) < 2:
                continue
            junction = path[-1]
            if junction not in adj or len(adj[junction]) < 3:
                continue

            # Remove all nodes in path except the junction
            for node in path[:-1]:
                for neighbor in list(adj.get(node, set())):
                    adj[neighbor].discard(node)
                if node in adj:
                    del adj[node]
                if node in coords:
                    del coords[node]

            pruned_this_pass += 1

        total_pruned += pruned_this_pass
        if pruned_this_pass == 0:
            break

    return total_pruned


def _rebuild_tables_from_graph(coords, radii, adj, original_vertex_table):
    """Rebuild vertex_table and edge_table from the cleaned graph structures."""
    # Preserve extra columns from original vertex table
    extra_cols = [c for c in original_vertex_table.columns
                  if c not in ("skeleton_id", "node_id", "z_um", "y_um", "x_um", "radius_um", "degree")]

    # Build lookup for extra columns from original table
    orig_extra = {}
    if extra_cols:
        for row in original_vertex_table.itertuples(index=False):
            key = (int(row.skeleton_id), int(row.node_id))
            orig_extra[key] = {col: getattr(row, col) for col in extra_cols}

    vertex_rows = []
    for key in sorted(coords.keys()):
        sk_id, node_id = key
        c = coords[key]
        row = {
            "skeleton_id": sk_id,
            "node_id": node_id,
            "z_um": float(c[0]),
            "y_um": float(c[1]),
            "x_um": float(c[2]),
            "radius_um": radii.get(key, np.nan),
        }
        if key in orig_extra:
            row.update(orig_extra[key])
        vertex_rows.append(row)

    edge_rows = []
    seen_edges = set()
    edge_id_counter = 0
    for key in sorted(adj.keys()):
        sk_id, node_id = key
        for neighbor in sorted(adj[key]):
            edge_pair = (min(key, neighbor), max(key, neighbor))
            if edge_pair in seen_edges:
                continue
            seen_edges.add(edge_pair)

            n_sk, n_node = neighbor
            src_c = coords[key]
            tgt_c = coords[neighbor]
            edge_rows.append({
                "skeleton_id": sk_id,
                "edge_id": edge_id_counter,
                "source_node": node_id,
                "target_node": n_node,
                "source_z_um": float(src_c[0]),
                "source_y_um": float(src_c[1]),
                "source_x_um": float(src_c[2]),
                "target_z_um": float(tgt_c[0]),
                "target_y_um": float(tgt_c[1]),
                "target_x_um": float(tgt_c[2]),
                "edge_length_um": float(np.linalg.norm(tgt_c - src_c)),
            })
            edge_id_counter += 1

    new_vertex_table = pd.DataFrame(vertex_rows) if vertex_rows else pd.DataFrame(
        columns=list(original_vertex_table.columns)
    )
    new_edge_table = pd.DataFrame(edge_rows) if edge_rows else pd.DataFrame(
        columns=["skeleton_id", "edge_id", "source_node", "target_node",
                 "source_z_um", "source_y_um", "source_x_um",
                 "target_z_um", "target_y_um", "target_x_um", "edge_length_um"]
    )

    return new_vertex_table, new_edge_table


def _branch_table_from_tables(vertex_table, edge_table):
    """Recompute branch metrics from cleaned vertex/edge tables."""
    if vertex_table.empty or edge_table.empty:
        return branch_table_from_skeletons({})

    all_branches = []
    for skeleton_id, skel_verts in vertex_table.groupby("skeleton_id"):
        skel_edges = edge_table[edge_table["skeleton_id"] == skeleton_id]
        if skel_edges.empty:
            continue

        node_ids = skel_verts["node_id"].tolist()
        node_id_set = set(node_ids)
        node_coords = {}
        node_radii = {}
        for row in skel_verts.itertuples(index=False):
            nid = int(row.node_id)
            node_coords[nid] = np.array([row.z_um, row.y_um, row.x_um], dtype=np.float64)
            node_radii[nid] = float(row.radius_um) if not pd.isna(row.radius_um) else np.nan

        local_adj = {nid: [] for nid in node_id_set}
        edge_list = []
        for row in skel_edges.itertuples(index=False):
            src, tgt = int(row.source_node), int(row.target_node)
            if src in node_id_set and tgt in node_id_set:
                eidx = len(edge_list)
                edge_list.append((src, tgt))
                local_adj[src].append((tgt, eidx))
                local_adj[tgt].append((src, eidx))

        degrees = {nid: len(local_adj[nid]) for nid in node_id_set}
        branch_nodes = {nid for nid, d in degrees.items() if d != 2}
        visited_edges = set()
        branch_id = 0

        for node in sorted(branch_nodes):
            for neighbor, eidx in local_adj[node]:
                if eidx in visited_edges:
                    continue

                path_nodes = [node, neighbor]
                path_edges = [eidx]
                visited_edges.add(eidx)
                prev, current = node, neighbor

                while degrees.get(current, 0) == 2:
                    candidates = [(n, ei) for n, ei in local_adj[current] if n != prev and ei not in visited_edges]
                    if not candidates:
                        break
                    next_n, next_ei = candidates[0]
                    visited_edges.add(next_ei)
                    path_nodes.append(next_n)
                    path_edges.append(next_ei)
                    prev, current = current, next_n

                if len(path_nodes) < 2:
                    continue

                pts = np.array([node_coords[n] for n in path_nodes], dtype=np.float64)
                seg_lengths = np.linalg.norm(pts[1:] - pts[:-1], axis=1)
                branch_length = float(seg_lengths.sum())
                euclidean = float(np.linalg.norm(pts[-1] - pts[0]))
                tortuosity = branch_length / euclidean if euclidean > 1e-12 else np.nan

                path_radii = [node_radii[n] for n in path_nodes if not np.isnan(node_radii.get(n, np.nan))]

                start_node = path_nodes[0]
                end_node = path_nodes[-1]
                start_deg = degrees.get(start_node, 0)
                end_deg = degrees.get(end_node, 0)

                all_branches.append({
                    "skeleton_id": int(skeleton_id),
                    "branch_id": branch_id,
                    "start_node": start_node,
                    "end_node": end_node,
                    "num_points": len(path_nodes),
                    "num_edges": len(path_edges),
                    "is_loop": False,
                    "start_degree": start_deg,
                    "end_degree": end_deg,
                    "branch_depth": np.nan,
                    "is_terminal_branch": bool(start_deg == 1 or end_deg == 1),
                    "is_branch_to_branch": bool(start_deg >= 3 and end_deg >= 3),
                    "is_root_branch": False,
                    "branch_length_um": branch_length,
                    "euclidean_length_um": euclidean,
                    "tortuosity": tortuosity,
                    "mean_radius_um": float(np.mean(path_radii)) if path_radii else np.nan,
                    "max_radius_um": float(np.max(path_radii)) if path_radii else np.nan,
                    "min_radius_um": float(np.min(path_radii)) if path_radii else np.nan,
                })
                branch_id += 1

    if not all_branches:
        return branch_table_from_skeletons({})
    return pd.DataFrame(all_branches)


def postprocess_skeleton_tables(
    vertex_table,
    edge_table,
    merge_branch_points_distance_um=0.0,
    prune_spurs_max_length_um=0.0,
):
    """Apply graph-level cleanup to skeleton tables.

    Parameters
    ----------
    vertex_table, edge_table : pd.DataFrame
        As produced by skeleton_tables_from_skeletons or chunkwise assembly.
    merge_branch_points_distance_um : float
        Cluster and merge branch points within this distance. 0 = disabled.
    prune_spurs_max_length_um : float
        Remove terminal branches shorter than this. 0 = disabled.

    Returns
    -------
    cleaned_vertex_table, cleaned_edge_table, cleaned_branch_table, cleanup_stats
    """
    if merge_branch_points_distance_um <= 0 and prune_spurs_max_length_um <= 0:
        branch_table = _branch_table_from_tables(vertex_table, edge_table)
        return vertex_table, edge_table, branch_table, {}

    if vertex_table.empty or edge_table.empty:
        branch_table = branch_table_from_skeletons({})
        return vertex_table, edge_table, branch_table, {}

    n_verts_before = len(vertex_table)
    n_edges_before = len(edge_table)

    coords, radii, adj = _build_graph_from_tables(vertex_table, edge_table)

    num_merges = _merge_nearby_branch_points(coords, radii, adj, merge_branch_points_distance_um)
    num_spurs_pruned = _prune_short_spurs(coords, adj, prune_spurs_max_length_um)

    # Remove isolated nodes (degree 0)
    isolated = [k for k, neighbors in adj.items() if len(neighbors) == 0]
    for k in isolated:
        del adj[k]
        if k in coords:
            del coords[k]

    cleaned_vertex_table, cleaned_edge_table = _rebuild_tables_from_graph(coords, radii, adj, vertex_table)
    cleaned_branch_table = _branch_table_from_tables(cleaned_vertex_table, cleaned_edge_table)

    stats = {
        "postprocess_enabled": True,
        "merge_branch_points_distance_um": float(merge_branch_points_distance_um),
        "prune_spurs_max_length_um": float(prune_spurs_max_length_um),
        "num_branchpoint_merges": int(num_merges),
        "num_spur_branches_pruned": int(num_spurs_pruned),
        "num_vertices_removed_postprocess": int(n_verts_before - len(cleaned_vertex_table)),
        "num_edges_removed_postprocess": int(n_edges_before - len(cleaned_edge_table)),
    }

    logger.info(
        "Postprocess: merged %d branch points, pruned %d spurs, removed %d vertices / %d edges",
        num_merges, num_spurs_pruned,
        stats["num_vertices_removed_postprocess"],
        stats["num_edges_removed_postprocess"],
    )

    return cleaned_vertex_table, cleaned_edge_table, cleaned_branch_table, stats


def stitch_skeleton_edges_across_chunks(vertex_table, edge_table, max_distance_um):
    if vertex_table.empty or edge_table.empty:
        return vertex_table, _normalize_edge_table_schema(edge_table), pd.DataFrame()

    vertex_table = _compute_vertex_degrees_from_edges(vertex_table, edge_table)
    edge_table = _normalize_edge_table_schema(edge_table)

    endpoint_mask = (
        vertex_table["in_core"].astype(bool)
        & vertex_table["touches_core_boundary"].astype(bool)
        & (vertex_table["degree"].astype(int) == 1)
    )
    endpoints = vertex_table.loc[endpoint_mask].copy()
    if endpoints.empty:
        return vertex_table, edge_table, pd.DataFrame()

    endpoints["chunk_index_tuple"] = endpoints["chunk_index"].map(_parse_chunk_index_text)
    chunk_set = set(endpoints["chunk_index_tuple"].tolist())
    stitch_rows = []
    used_nodes = set()
    next_edge_id = int(edge_table["edge_id"].max()) + 1 if not edge_table.empty else 0

    face_pairs = [
        ("touch_z_max", "touch_z_min", (1, 0, 0)),
        ("touch_y_max", "touch_y_min", (0, 1, 0)),
        ("touch_x_max", "touch_x_min", (0, 0, 1)),
    ]

    for chunk_index in sorted(chunk_set):
        chunk_rows = endpoints.loc[endpoints["chunk_index_tuple"] == chunk_index]
        for face_a, face_b, delta in face_pairs:
            neighbor_index = tuple(chunk_index[i] + delta[i] for i in range(3))
            if neighbor_index not in chunk_set:
                continue

            a_rows = chunk_rows.loc[chunk_rows[face_a].astype(bool)]
            if a_rows.empty:
                continue

            b_rows = endpoints.loc[
                (endpoints["chunk_index_tuple"] == neighbor_index)
                & endpoints[face_b].astype(bool)
            ]
            if b_rows.empty:
                continue

            candidate_pairs = []
            for a in a_rows.itertuples(index=False):
                a_key = (int(a.skeleton_id), int(a.node_id))
                if a_key in used_nodes:
                    continue
                a_point = np.array([a.z_um, a.y_um, a.x_um], dtype=np.float64)

                for b in b_rows.itertuples(index=False):
                    b_key = (int(b.skeleton_id), int(b.node_id))
                    if b_key in used_nodes:
                        continue
                    b_point = np.array([b.z_um, b.y_um, b.x_um], dtype=np.float64)
                    distance_um = float(np.linalg.norm(a_point - b_point))
                    if distance_um <= float(max_distance_um):
                        candidate_pairs.append((distance_um, a, b))

            candidate_pairs.sort(key=lambda item: item[0])
            for distance_um, a, b in candidate_pairs:
                a_key = (int(a.skeleton_id), int(a.node_id))
                b_key = (int(b.skeleton_id), int(b.node_id))
                if a_key in used_nodes or b_key in used_nodes:
                    continue

                used_nodes.add(a_key)
                used_nodes.add(b_key)
                stitch_rows.append(
                    {
                        "skeleton_id": int(a.skeleton_id),
                        "edge_id": int(next_edge_id),
                        "source_node": int(a.node_id),
                        "target_node": int(b.node_id),
                        "source_z_um": float(a.z_um),
                        "source_y_um": float(a.y_um),
                        "source_x_um": float(a.x_um),
                        "target_z_um": float(b.z_um),
                        "target_y_um": float(b.y_um),
                        "target_x_um": float(b.x_um),
                        "edge_length_um": float(distance_um),
                        "chunk_index": str(a.chunk_index),
                        "chunk_start_zyx": str(a.chunk_start_zyx),
                        "chunk_stop_zyx": str(a.chunk_stop_zyx),
                        "expanded_start_zyx": str(a.expanded_start_zyx),
                        "expanded_stop_zyx": str(a.expanded_stop_zyx),
                        "source_skeleton_id": int(a.skeleton_id),
                        "target_skeleton_id": int(b.skeleton_id),
                        "is_stitch": True,
                    }
                )
                next_edge_id += 1

    stitch_edge_table = pd.DataFrame(stitch_rows)
    if stitch_edge_table.empty:
        return vertex_table, edge_table, stitch_edge_table

    combined_edge_table = pd.concat([edge_table, stitch_edge_table], ignore_index=True, sort=False)
    return vertex_table, combined_edge_table, stitch_edge_table


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
            "num_terminal_branches": 0,
            "max_branch_depth": None,
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
        "num_terminal_branches": int(branch_table["is_terminal_branch"].fillna(False).astype(bool).sum())
        if "is_terminal_branch" in branch_table.columns
        else None,
        "max_branch_depth": int(np.nanmax(branch_table["branch_depth"].to_numpy(dtype=np.float64)))
        if ("branch_depth" in branch_table.columns and np.any(np.isfinite(branch_table["branch_depth"].to_numpy(dtype=np.float64))))
        else None,
    }


def summarize_graph_from_skeleton_tables(vertex_table, edge_table):
    if vertex_table.empty:
        return {
            "num_vertices": 0,
            "num_edges": 0,
            "num_branch_points": 0,
            "num_end_points": 0,
            "num_end_points_non_boundary": 0,
        }

    vertex_table = _compute_vertex_degrees_from_edges(vertex_table, edge_table)
    degrees = vertex_table["degree"].astype(int)
    boundary_mask = (
        vertex_table["touches_core_boundary"].astype(bool)
        if "touches_core_boundary" in vertex_table.columns
        else np.zeros(len(vertex_table), dtype=bool)
    )

    return {
        "num_vertices": int(len(vertex_table)),
        "num_edges": int(len(edge_table)),
        "num_branch_points": int((degrees >= 3).sum()),
        "num_end_points": int((degrees == 1).sum()),
        "num_end_points_non_boundary": int(((degrees == 1) & (~boundary_mask)).sum()),
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
    save_skeleton=False,
    save_swc=False,
    merge_branch_points_distance_um=0.0,
    prune_spurs_max_length_um=0.0,
):
    _started_at = time.time()
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

    result = {
        "summary": summary,
        "branch_table": branch_table,
        "branch_csv_path": branch_csv_path,
        "summary_json_path": summary_json_path,
    }

    if save_skeleton:
        vertex_table, edge_table = skeleton_tables_from_skeletons(skeletons)

        postprocess_active = (merge_branch_points_distance_um > 0 or prune_spurs_max_length_um > 0)
        if postprocess_active:
            vertex_table, edge_table, branch_table, cleanup_stats = postprocess_skeleton_tables(
                vertex_table, edge_table,
                merge_branch_points_distance_um=merge_branch_points_distance_um,
                prune_spurs_max_length_um=prune_spurs_max_length_um,
            )
            summary.update(cleanup_stats)
            # Rewrite branch CSV with cleaned data
            branch_table.to_csv(branch_csv_path, index=False)
            result["branch_table"] = branch_table

        vertex_csv_path, edge_csv_path = _write_skeleton_tables(vertex_table, edge_table, output_root)
        result["vertex_table"] = vertex_table
        result["edge_table"] = edge_table
        result["vertex_csv_path"] = vertex_csv_path
        result["edge_csv_path"] = edge_csv_path
        if save_swc:
            swc_dir, swc_paths = write_swc_files(vertex_table, edge_table, output_root)
            result["swc_dir"] = swc_dir
            result["swc_paths"] = swc_paths

    if write_run_manifest is not None:
        _output_files = [v for k, v in result.items() if k.endswith("_path") or k == "swc_dir"]
        result["manifest_path"] = write_run_manifest(
            output_root,
            module="tubule_reconstruction.kimimaro_reconstruction",
            entrypoint="analyze_binary_mask_zarr",
            inputs={
                "mask_zarr_path": str(mask_zarr_path),
                "dataset_name": dataset_name,
                "resolution_xyz": resolution_xyz,
                "foreground_label": foreground_label,
                "dust_threshold": dust_threshold,
                "fix_borders": fix_borders,
                "save_skeleton": save_skeleton,
                "save_swc": save_swc,
            },
            outputs=_output_files,
            started_at=_started_at,
        )

    return result


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
    save_skeleton=False,
    save_swc=False,
):
    return analyze_binary_mask_zarr_chunkwise(
        mask_zarr_path=mask_zarr_path,
        output_dir=output_dir,
        dataset_name=dataset_name,
        resolution_xyz=resolution_xyz,
        foreground_label=foreground_label,
        dust_threshold=dust_threshold,
        fix_borders=fix_borders,
        parallel=parallel,
        teasar_params=teasar_params,
        save_skeleton=save_skeleton,
        save_swc=save_swc,
        process_existing_only=True,
        halo_zyx=(0, 0, 0),
        mode_name="test_chunkwise",
    )


def analyze_binary_mask_zarr_chunkwise(
    mask_zarr_path,
    output_dir,
    dataset_name="0",
    resolution_xyz=(1.0, 1.0, 1.0),
    foreground_label=1,
    dust_threshold=0,
    fix_borders=True,
    parallel=1,
    teasar_params=None,
    save_skeleton=False,
    save_swc=False,
    process_existing_only=False,
    halo_zyx=(0, 0, 0),
    mode_name="chunkwise",
    stitch=True,
    stitch_max_distance_um=5.0,
    chunk_workers=1,
    merge_branch_points_distance_um=0.0,
    prune_spurs_max_length_um=0.0,
):
    _started_at = time.time()
    resolution_xyz = parse_resolution_xyz(resolution_xyz)
    chunk_workers = max(1, int(chunk_workers))
    mask_zarr = open_zarr_dataset(mask_zarr_path, dataset_name=dataset_name)
    halo_zyx = parse_triplet_int(halo_zyx)
    existing_chunks = list_existing_chunk_indices(mask_zarr)
    if not existing_chunks and process_existing_only:
        raise ValueError("No physical chunks found in the input mask Zarr store.")

    chunks = mask_zarr.chunks
    shape = mask_zarr.shape
    chunk_indices = existing_chunks if process_existing_only else list(iter_all_chunk_indices(shape, chunks))
    kimimaro_parallel = int(parallel) if chunk_workers <= 1 else 1

    task_payloads = [
        {
            "mask_zarr_path": str(mask_zarr_path),
            "dataset_name": dataset_name,
            "resolution_xyz": resolution_xyz,
            "foreground_label": foreground_label,
            "dust_threshold": dust_threshold,
            "fix_borders": fix_borders,
            "kimimaro_parallel": kimimaro_parallel,
            "teasar_params": teasar_params,
            "save_skeleton": save_skeleton,
            "chunk_index": chunk_index,
            "chunks": chunks,
            "shape": shape,
            "halo_zyx": halo_zyx,
        }
        for chunk_index in chunk_indices
    ]

    if chunk_workers <= 1:
        chunk_results = [_process_chunk_task(task) for task in task_payloads]
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=chunk_workers) as executor:
            chunk_results = list(executor.map(_process_chunk_task, task_payloads))

    chunk_results.sort(key=lambda item: tuple(item["chunk_index"]))

    all_branch_tables = []
    all_vertex_tables = []
    all_edge_tables = []
    chunk_rows = []
    skeleton_offset = 0
    total_mask_voxels = 0
    total_components = 0

    for chunk_result in chunk_results:
        total_mask_voxels += int(chunk_result["core_mask_voxels"])
        total_components += int(chunk_result["connected_components"])

        branch_table = _reindex_branch_table(
            chunk_result["branch_table"],
            skeleton_offset=skeleton_offset,
        )
        all_branch_tables.append(branch_table)

        if save_skeleton:
            vertex_table, edge_table = _reindex_skeleton_tables(
                chunk_result["vertex_table"],
                chunk_result["edge_table"],
                skeleton_offset=skeleton_offset,
            )
            all_vertex_tables.append(vertex_table)
            all_edge_tables.append(edge_table)

        chunk_rows.append(chunk_result["chunk_summary"])
        skeleton_offset += int(chunk_result["num_skeletons"])

    combined_branch_table = (
        pd.concat(all_branch_tables, ignore_index=True)
        if all_branch_tables
        else branch_table_from_skeletons({})
    )
    combined_vertex_table = (
        pd.concat(all_vertex_tables, ignore_index=True)
        if all_vertex_tables
        else pd.DataFrame()
    )
    combined_edge_table = (
        pd.concat(all_edge_tables, ignore_index=True)
        if all_edge_tables
        else pd.DataFrame()
    )
    summary = summarize_branch_table(combined_branch_table)
    summary["mode"] = mode_name
    summary["processed_chunks"] = int(len(chunk_indices))
    summary["connected_components"] = int(total_components)
    summary["mask_voxels"] = int(total_mask_voxels)
    summary["mask_volume_um3"] = float(total_mask_voxels * np.prod(tuple(float(v) for v in resolution_xyz)))
    summary["halo_zyx"] = [int(v) for v in halo_zyx]
    summary["process_existing_only"] = bool(process_existing_only)
    summary["stitch_enabled"] = bool(stitch)
    summary["stitch_max_distance_um"] = float(stitch_max_distance_um)
    summary["chunk_workers"] = int(chunk_workers)
    summary["kimimaro_parallel_per_chunk"] = int(kimimaro_parallel)

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    branch_csv_path = output_root / "vessel_branch_metrics.csv"
    chunk_csv_path = output_root / "vessel_chunk_metrics.csv"
    summary_json_path = output_root / "vessel_network_summary.json"

    combined_branch_table.to_csv(branch_csv_path, index=False)
    pd.DataFrame(chunk_rows).to_csv(chunk_csv_path, index=False)

    result = {
        "summary": summary,
        "branch_table": combined_branch_table,
        "branch_csv_path": branch_csv_path,
        "chunk_csv_path": chunk_csv_path,
        "summary_json_path": summary_json_path,
    }

    if save_skeleton:
        stitch_edge_table = pd.DataFrame()
        if stitch:
            combined_vertex_table, combined_edge_table, stitch_edge_table = stitch_skeleton_edges_across_chunks(
                combined_vertex_table,
                combined_edge_table,
                max_distance_um=stitch_max_distance_um,
            )
        else:
            combined_edge_table = _normalize_edge_table_schema(combined_edge_table)

        vertex_csv_path, edge_csv_path = _write_skeleton_tables(combined_vertex_table, combined_edge_table, output_root)
        result["vertex_table"] = combined_vertex_table
        result["edge_table"] = combined_edge_table
        result["vertex_csv_path"] = vertex_csv_path
        result["edge_csv_path"] = edge_csv_path
        summary["num_stitch_edges"] = int(len(stitch_edge_table))
        if not stitch_edge_table.empty:
            stitch_csv_path = output_root / "skeleton_stitch_edges.csv"
            stitch_edge_table.to_csv(stitch_csv_path, index=False)
            result["stitch_edge_table"] = stitch_edge_table
            result["stitch_csv_path"] = stitch_csv_path
            logger.info("Skeleton stitch CSV: %s", stitch_csv_path)
        else:
            summary["num_stitch_edges"] = 0

        postprocess_active = (merge_branch_points_distance_um > 0 or prune_spurs_max_length_um > 0)
        if postprocess_active:
            combined_vertex_table, combined_edge_table, cleaned_branch_table, cleanup_stats = postprocess_skeleton_tables(
                combined_vertex_table, combined_edge_table,
                merge_branch_points_distance_um=merge_branch_points_distance_um,
                prune_spurs_max_length_um=prune_spurs_max_length_um,
            )
            summary.update(cleanup_stats)
            combined_branch_table = cleaned_branch_table
            combined_branch_table.to_csv(branch_csv_path, index=False)
            result["branch_table"] = combined_branch_table
            # Rewrite skeleton tables with cleaned data
            vertex_csv_path, edge_csv_path = _write_skeleton_tables(combined_vertex_table, combined_edge_table, output_root)
            result["vertex_table"] = combined_vertex_table
            result["edge_table"] = combined_edge_table
            result["vertex_csv_path"] = vertex_csv_path
            result["edge_csv_path"] = edge_csv_path

        if save_swc:
            swc_dir, swc_paths = write_swc_files(combined_vertex_table, combined_edge_table, output_root)
            result["swc_dir"] = swc_dir
            result["swc_paths"] = swc_paths

        graph_summary = summarize_graph_from_skeleton_tables(combined_vertex_table, combined_edge_table)
        summary.update(graph_summary)

    _write_summary_json(summary, summary_json_path)

    if write_run_manifest is not None:
        _output_files = [v for k, v in result.items() if k.endswith("_path") or k == "swc_dir"]
        result["manifest_path"] = write_run_manifest(
            output_root,
            module="tubule_reconstruction.kimimaro_reconstruction",
            entrypoint="analyze_binary_mask_zarr_chunkwise",
            inputs={
                "mask_zarr_path": str(mask_zarr_path),
                "dataset_name": dataset_name,
                "resolution_xyz": resolution_xyz,
                "foreground_label": foreground_label,
                "dust_threshold": dust_threshold,
                "fix_borders": fix_borders,
                "save_skeleton": save_skeleton,
                "save_swc": save_swc,
                "stitch": stitch,
                "stitch_max_distance_um": stitch_max_distance_um,
            },
            outputs=_output_files,
            started_at=_started_at,
        )

    return result


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
    parser.add_argument("--save_skeleton", action="store_true", help="Export skeleton vertices and edges as CSV")
    parser.add_argument("--save_swc", action="store_true", help="Export one SWC file per skeleton")
    parser.add_argument("--chunkwise", action="store_true", help="Process the mask chunk-by-chunk instead of loading the full volume")
    parser.add_argument("--chunk_workers", type=int, default=1, help="Number of worker processes for chunkwise processing")
    parser.add_argument("--existing_only", action="store_true", help="In chunkwise mode, only process chunks that physically exist in the store")
    parser.add_argument("--halo_zyx", default="0,0,0", help="Halo overlap in voxels for chunkwise processing as z,y,x")
    parser.add_argument("--no_stitch", action="store_true", help="Disable cross-chunk skeleton stitching in chunkwise mode")
    parser.add_argument("--stitch_max_distance_um", type=float, default=5.0, help="Maximum distance in microns for cross-chunk endpoint stitching")
    parser.add_argument("--merge_branch_points_distance_um", type=float, default=0.0, help="Merge branch points within this distance (um). 0=disabled")
    parser.add_argument("--prune_spurs_max_length_um", type=float, default=0.0, help="Prune terminal branches shorter than this (um). 0=disabled")
    parser.add_argument(
        "--test",
        action="store_true",
        help="Smoke-test mode: only process chunks that physically exist in the input mask store",
    )
    parser.add_argument(
        "--json_logs",
        action="store_true",
        help="Emit NDJSON log records to stderr instead of plain text",
    )
    return parser


def main():
    parser = build_argparser()
    args = parser.parse_args()

    if args.json_logs:
        import sys

        class _JsonFormatter(logging.Formatter):
            def format(self, record):
                import json as _json
                return _json.dumps({
                    "level": record.levelname,
                    "logger": record.name,
                    "message": record.getMessage(),
                })

        _handler = logging.StreamHandler(sys.stderr)
        _handler.setFormatter(_JsonFormatter())
        logging.root.addHandler(_handler)
        logging.root.setLevel(logging.INFO)
    else:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    import sys as _sys

    try:
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
                save_skeleton=args.save_skeleton,
                save_swc=args.save_swc,
            )
        elif args.chunkwise:
            result = analyze_binary_mask_zarr_chunkwise(
                mask_zarr_path=args.mask_zarr,
                output_dir=args.output_dir,
                dataset_name=args.dataset_name,
                resolution_xyz=args.resolution_xyz,
                foreground_label=args.foreground_label,
                dust_threshold=args.dust_threshold,
                fix_borders=not args.no_fix_borders,
                parallel=args.parallel,
                save_skeleton=args.save_skeleton,
                save_swc=args.save_swc,
                process_existing_only=args.existing_only,
                halo_zyx=args.halo_zyx,
                mode_name="chunkwise",
                stitch=not args.no_stitch,
                stitch_max_distance_um=args.stitch_max_distance_um,
                chunk_workers=args.chunk_workers,
                merge_branch_points_distance_um=args.merge_branch_points_distance_um,
                prune_spurs_max_length_um=args.prune_spurs_max_length_um,
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
                save_skeleton=args.save_skeleton,
                save_swc=args.save_swc,
                merge_branch_points_distance_um=args.merge_branch_points_distance_um,
                prune_spurs_max_length_um=args.prune_spurs_max_length_um,
            )

        logger.info("Vessel reconstruction completed.")
        logger.info("Summary JSON: %s", result["summary_json_path"])
        logger.info("Branch CSV: %s", result["branch_csv_path"])
        if "chunk_csv_path" in result:
            logger.info("Chunk CSV: %s", result["chunk_csv_path"])
        if "vertex_csv_path" in result:
            logger.info("Skeleton vertices CSV: %s", result["vertex_csv_path"])
        if "edge_csv_path" in result:
            logger.info("Skeleton edges CSV: %s", result["edge_csv_path"])
        if "stitch_csv_path" in result:
            logger.info("Skeleton stitch CSV: %s", result["stitch_csv_path"])
        if "swc_dir" in result:
            logger.info("SWC directory: %s", result["swc_dir"])
    except Exception as exc:
        if PipelineError is not None and isinstance(exc, PipelineError):
            import json as _json
            print(_json.dumps({"error_code": exc.code.value, "message": str(exc.message)}), file=_sys.stderr)
            _sys.exit(exc.exit_code)
        logger.exception("Unhandled error: %s", exc)
        _sys.exit(1)


if __name__ == "__main__":
    main()
