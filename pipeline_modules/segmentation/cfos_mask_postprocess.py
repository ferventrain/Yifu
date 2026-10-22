from __future__ import annotations

import argparse
from pathlib import Path
from typing import Literal

import cc3d
import numpy as np
from skimage.measure import block_reduce, marching_cubes
from tqdm import tqdm

try:
    from pipeline_modules.segmentation.zarr_utils import (
        create_output_zarr,
        export_zarr_to_tiff,
        open_zarr_dataset,
    )
    from pipeline_modules.tubule_reconstruction.region_vessel_analysis import (
        _collect_subtree_ids,
        load_region_tree_with_lookups,
        resolve_region_query,
    )
except ImportError:  # pragma: no cover
    from .zarr_utils import (
        create_output_zarr,
        export_zarr_to_tiff,
        open_zarr_dataset,
    )
    from pipeline_modules.tubule_reconstruction.region_vessel_analysis import (
        _collect_subtree_ids,
        load_region_tree_with_lookups,
        resolve_region_query,
    )

DEFAULT_REGION_CFG = (
    Path(__file__).resolve().parents[1] / "registration" / "Region_Csv_Rev2_updated.CSV"
)


def _extent_ratio(bbox: tuple[slice, slice, slice]) -> float:
    extents = [max(int(sl.stop) - int(sl.start), 1) for sl in bbox]
    return float(max(extents)) / float(min(extents))


def resolve_region_subtree_ids(region_query: str, *, cfg_path: Path) -> tuple[set[int], str]:
    nodes_by_id, acronym_to_ids, name_to_ids = load_region_tree_with_lookups(cfg_path)
    node = resolve_region_query(region_query, nodes_by_id, acronym_to_ids, name_to_ids)
    subtree_ids = set(_collect_subtree_ids(node))
    display_name = str(node.get("name") or node.get("acronym") or node["id"])
    return subtree_ids, display_name


def _build_region_slice(label_slice: np.ndarray, region_id_array: np.ndarray) -> np.ndarray:
    return np.isin(label_slice, region_id_array)


def downsample_mask_zarr(
    mask_in,
    *,
    factor: int,
    label_in=None,
    region_id_array: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None, list[int]]:
    depth, height, width = (int(mask_in.shape[0]), int(mask_in.shape[1]), int(mask_in.shape[2]))
    if (label_in is None) ^ (region_id_array is None):
        raise ValueError("label_in and region_id_array must be provided together")

    ds_shape = (-(-depth // factor), -(-height // factor), -(-width // factor))
    ds_mask = np.zeros(ds_shape, dtype=np.uint8)
    ds_region = None if region_id_array is None else np.zeros(ds_shape, dtype=np.uint8)
    active_z_indices: list[int] = []

    # Read z blocks aligned to the source chunk depth: factor-slice reads on
    # deep-chunked Zarrs decompress every touched chunk once per 4 slices.
    # Cap the block by bytes so uint32 labels never materialize huge slabs.
    chunk_z = max(1, int(mask_in.chunks[0]) if getattr(mask_in, "chunks", None) else factor)
    per_slice = int(height) * int(width) * int(np.dtype(mask_in.dtype).itemsize)
    slice_cap = max(factor, (4 * 1024**3) // max(per_slice, 1))
    slice_cap -= slice_cap % factor
    z_block = max(factor, min(chunk_z, slice_cap))
    for zb0 in tqdm(range(0, depth, z_block), desc="Downsample mask", unit="block"):
        zb1 = min(zb0 + z_block, depth)
        slab = (np.asarray(mask_in[zb0:zb1], dtype=np.uint8) > 0).astype(np.uint8)
        pad = [(-s) % factor for s in slab.shape]
        if any(pad):
            slab = np.pad(slab, [(0, p) for p in pad])
        ds_mask[zb0 // factor : zb1 // factor] = block_reduce(
            slab,
            block_size=(factor, factor, factor),
            func=np.max,
        )
        if ds_region is not None:
            label_slab = np.asarray(label_in[zb0:zb1])
            region_slab = _build_region_slice(label_slab, region_id_array).astype(np.uint8)
            if any(pad):
                region_slab = np.pad(region_slab, [(0, p) for p in pad])
            ds_region[zb0 // factor : zb1 // factor] = block_reduce(
                region_slab,
                block_size=(factor, factor, factor),
                func=np.max,
            )
            active = np.nonzero(ds_region[zb0 // factor : zb1 // factor].any(axis=(1, 2)))[0]
            for ds_z in active:
                z0 = (zb0 // factor + int(ds_z)) * factor
                active_z_indices.extend(range(z0, z0 + factor))

    return ds_mask, ds_region, active_z_indices


def select_keep_labels_3d(
    labels: np.ndarray,
    stats: dict,
    *,
    max_voxels: int,
    min_voxels: int,
    max_extent_ratio: float,
    downsample_factor: int,
    max_single_slice_voxels: int = 0,
    max_slice_counts_ds: np.ndarray | None = None,
    edge_hits: np.ndarray | None = None,
) -> tuple[set[int], int, int, int, int]:
    """Decide which 3D components survive the standard filters.

    Volumes/areas are given at full resolution and converted to the
    downsampled grid. A filter value <= 0 disables that filter:

    - ``min_voxels`` / ``max_voxels``: 3D object volume bounds
    - ``max_extent_ratio``: bounding-box max/min axis ratio
    - ``max_single_slice_voxels``: drop objects whose largest single-slice
      footprint (vessels, surface sheets) exceeds this many voxels
    - ``edge_hits``: per-label flags from the brain-boundary exclusion
      (edge shrink); any hit removes the object
    """
    keep_labels: set[int] = set()
    removed_volume = 0
    removed_extent = 0
    removed_single_slice = 0
    removed_edge = 0

    scale_volume = downsample_factor ** 3
    min_voxels_ds = max(int(min_voxels) // scale_volume, 1) if min_voxels > 0 else 0
    max_voxels_ds = max(int(max_voxels) // scale_volume, 1) if max_voxels > 0 else 0
    single_slice_threshold_ds = (
        float(max_single_slice_voxels) / float(downsample_factor ** 2) if max_single_slice_voxels > 0 else 0.0
    )

    voxel_counts = stats["voxel_counts"]
    bounding_boxes = stats["bounding_boxes"]

    for label_idx in range(1, len(voxel_counts)):
        count_ds = int(voxel_counts[label_idx])
        if min_voxels_ds and count_ds < min_voxels_ds:
            removed_volume += 1
            continue
        if max_voxels_ds and count_ds > max_voxels_ds:
            removed_volume += 1
            continue
        if (
            single_slice_threshold_ds > 0
            and max_slice_counts_ds is not None
            and int(max_slice_counts_ds[label_idx]) >= single_slice_threshold_ds
        ):
            removed_single_slice += 1
            continue
        if max_extent_ratio > 0:
            ratio = _extent_ratio(bounding_boxes[label_idx])
            if ratio > max_extent_ratio:
                removed_extent += 1
                continue
        if edge_hits is not None and int(edge_hits[label_idx]) > 0:
            removed_edge += 1
            continue

        keep_labels.add(label_idx)

    return keep_labels, removed_volume, removed_extent, removed_single_slice, removed_edge


def upsample_keep_slice(ds_keep: np.ndarray, factor: int, height: int, width: int) -> np.ndarray:
    upsampled = np.repeat(np.repeat(ds_keep, factor, axis=0), factor, axis=1)
    return upsampled[:height, :width]


def _stream_pool_label_foreground(label_arr, factor: int, ds_shape: tuple[int, int, int]) -> np.ndarray:
    """Max-pool (label > 0) onto the downsampled grid, streaming z blocks
    aligned to the source chunk depth (factor-slice reads on deep-chunked
    Zarrs re-decompress every chunk once per few slices). Blocks are
    byte-capped so uint32 labels never materialize huge slabs."""
    ds = np.zeros(ds_shape, dtype=np.uint8)
    depth = int(label_arr.shape[0])
    per_slice = int(label_arr.shape[1]) * int(label_arr.shape[2]) * int(np.dtype(label_arr.dtype).itemsize)
    chunk_z = max(1, int(label_arr.chunks[0]) if getattr(label_arr, "chunks", None) else factor)
    slice_cap = max(factor, (4 * 1024**3) // max(per_slice, 1))
    slice_cap -= slice_cap % factor
    z_block = max(factor, min(chunk_z, slice_cap))
    for zb0 in tqdm(range(0, depth, z_block), desc="Pool label foreground", unit="block"):
        zb1 = min(zb0 + z_block, depth)
        slab = (np.asarray(label_arr[zb0:zb1]) > 0).astype(np.uint8)
        pad = [(-s) % factor for s in slab.shape]
        if any(pad):
            slab = np.pad(slab, [(0, p) for p in pad])
        ds[zb0 // factor : zb1 // factor] = block_reduce(
            slab,
            block_size=(factor, factor, factor),
            func=np.max,
        )
    return ds


def _per_component_max_slice_counts(labels: np.ndarray, n_labels: int) -> np.ndarray:
    """Largest single downsampled-slice footprint per component label."""
    max_counts = np.zeros(n_labels, dtype=np.int64)
    for z in range(labels.shape[0]):
        counts = np.bincount(labels[z].astype(np.int64).ravel(), minlength=n_labels)
        np.maximum(max_counts, counts[:n_labels], out=max_counts)
    return max_counts


def _resolve_ds_brain_mask(
    label_zarr: Path | None,
    brain_mask_ds: Path | None,
    downsample_factor: int,
) -> tuple[np.ndarray, dict, Path]:
    """Load the factor-4 brain-mask cache, building and persisting it from the
    label zarr (one full label read) when the cache does not exist yet."""
    from pipeline_modules.registration.brain_mask_ds import (
        ATTRS_KEY,
        DS_FACTOR,
        brain_mask_ds_path,
        open_brain_mask_ds,
    )
    from pipeline_modules.utils.zarr_io import create_output_zarr

    cache_path = brain_mask_ds or (brain_mask_ds_path(label_zarr) if label_zarr is not None else None)
    if cache_path is None:
        raise ValueError("edge_shrink_factor requires a brain-mask cache or --label_zarr")
    if cache_path.exists():
        mask, meta = open_brain_mask_ds(cache_path)
        return mask, meta, cache_path

    if label_zarr is None or not Path(label_zarr).exists():
        raise FileNotFoundError(
            f"Brain-mask cache not found at {cache_path} and no label zarr to build it from. "
            "Rerun registration (it writes the cache as a side product) or pass --label_zarr."
        )
    print(f"Building factor-{DS_FACTOR} brain-mask cache from {label_zarr} (one-time full label read)...", flush=True)
    label_arr = open_zarr_dataset(label_zarr, dataset_name="0")
    ds_shape = tuple(int(-(-int(s) // DS_FACTOR)) for s in label_arr.shape)
    mask = _stream_pool_label_foreground(label_arr, DS_FACTOR, ds_shape)
    _, out_arr = create_output_zarr(cache_path, mask.shape, (min(8, mask.shape[0]), 1024, 1024), "uint8", data=mask)
    del out_arr
    root_attrs = {
        ATTRS_KEY: {
            "factor": DS_FACTOR,
            "target_shape": [int(s) for s in label_arr.shape],
            "ds_shape": [int(s) for s in mask.shape],
            "spacing_zyx_um": None,
            "source": "label_zarr_fallback",
        }
    }
    import zarr

    try:
        grp = zarr.open_group(str(cache_path), mode="r+")
    except TypeError:  # zarr 2.x has no zarr_format kwarg
        grp = zarr.open_group(str(cache_path), mode="r+")
    grp.attrs.update(root_attrs)
    return mask, dict(root_attrs[ATTRS_KEY]), cache_path


def _edge_shrink_hits(
    ds_brain: np.ndarray,
    spacing_zyx,
    centroids: np.ndarray,
    n_labels: int,
    shrink: float,
    cache_factor: int,
    downsample_factor: int,
    coarse_factor: int = 4,
) -> np.ndarray:
    """AIND-style brain-boundary exclusion (crop_cells).

    Builds a surface mesh from the ds brain mask (max-pooled onto a coarse
    grid, marching cubes), shrinks it about its center of mass by ``shrink``,
    and marks every component whose centroid falls outside the shrunken mesh.
    Points are evaluated in micrometers when the cache carries spacing,
    otherwise in ds-grid voxel units (the shrink factor is dimensionless).

    Returns a 0/1 hit array indexed like cc3d labels (0 = background).
    """
    import vedo

    spacing = tuple(float(v) for v in (spacing_zyx or (1.0, 1.0, 1.0)))

    pad = [(-s) % coarse_factor for s in ds_brain.shape]
    coarse = ds_brain if not any(pad) else np.pad(ds_brain, [(0, p) for p in pad])
    coarse = block_reduce(coarse, block_size=(coarse_factor,) * 3, func=np.max)
    # One empty border guarantees a closed marching-cubes surface.
    coarse = np.pad(coarse, 1)
    if not coarse.any():
        return np.zeros(n_labels, dtype=np.int64)
    verts, faces, _, _ = marching_cubes(
        coarse.astype(np.float32),
        level=0.5,
        spacing=tuple(s * coarse_factor for s in spacing),
    )

    mesh = vedo.Mesh([verts, faces])
    com = mesh.center_of_mass()
    verts_shrunk = com + float(shrink) * (verts - com)
    shrunk = vedo.Mesh([verts_shrunk, faces])

    # centroids are in the postprocess ds grid; convert to cache-grid
    # coordinates, then to mesh coordinates (padded by one coarse voxel).
    pts = centroids * (float(downsample_factor) / float(cache_factor)) * np.asarray(spacing)
    pts = pts + np.asarray([s * coarse_factor for s in spacing])
    inside_ids = shrunk.inside_points(pts=pts, return_ids=True)
    # inside_ids index the centroid array (component label = index + 1) and are
    # the components to KEEP; everything else falls outside the shrunken brain.
    hits = np.ones(n_labels, dtype=np.int64)
    hits[0] = 0
    hits[np.asarray(inside_ids, dtype=int) + 1] = 0
    return hits


def postprocess_cfos_mask_3d(
    *,
    signal_zarr: Path,
    mask_zarr: Path,
    output_mask_zarr: Path,
    masked_signal_zarr: Path | None,
    masked_tiff_dir: Path | None,
    max_voxels: int,
    min_voxels: int,
    max_extent_ratio: float,
    downsample_factor: int,
    edge_shrink_factor: float = 0.0,
    max_single_slice_voxels: int = 0,
    label_zarr: Path | None = None,
    brain_mask_ds: Path | None = None,
    region_query: str | None = None,
    region_cfg: Path | None = None,
    region_outside: Literal["keep", "zero"] = "keep",
    dataset_name: str = "0",
    export_prefix: str = "masked_",
) -> dict[str, str | int | float]:
    signal = open_zarr_dataset(signal_zarr, dataset_name=dataset_name)
    mask_in = open_zarr_dataset(mask_zarr, dataset_name=dataset_name)
    if signal.shape != mask_in.shape:
        raise ValueError(f"Shape mismatch: signal={signal.shape}, mask={mask_in.shape}")

    label_in = None
    region_id_array = None
    region_name = None
    region_ids: set[int] = set()
    if region_query:
        if label_zarr is None:
            raise ValueError("--region requires --label_zarr")
        label_in = open_zarr_dataset(label_zarr, dataset_name=dataset_name)
        if label_in.shape != mask_in.shape:
            raise ValueError(f"Shape mismatch: mask={mask_in.shape}, label={label_in.shape}")
        region_cfg = region_cfg or DEFAULT_REGION_CFG
        region_ids, region_name = resolve_region_subtree_ids(region_query, cfg_path=region_cfg)
        region_id_array = np.asarray(sorted(region_ids))

    ds_mask, ds_region, active_z_indices = downsample_mask_zarr(
        mask_in,
        factor=downsample_factor,
        label_in=label_in if region_id_array is not None else None,
        region_id_array=region_id_array,
    )
    if ds_region is not None:
        ds_mask = (ds_mask > 0) & (ds_region > 0)
        ds_mask = ds_mask.astype(np.uint8)
    labels = cc3d.connected_components(ds_mask, connectivity=26)
    stats = cc3d.statistics(labels)
    n_labels = int(len(stats["voxel_counts"]))
    print(f"3D connected components: {n_labels - 1} objects in filter scope", flush=True)

    edge_hits = None
    brain_mask_cache_used = ""
    if edge_shrink_factor > 0:
        ds_brain, cache_meta, cache_path = _resolve_ds_brain_mask(label_zarr, brain_mask_ds, downsample_factor)
        cache_factor = int(cache_meta.get("factor", 4))
        spacing = cache_meta.get("spacing_zyx_um") or None
        centroids = np.asarray(stats["centroids"])[1:]
        coarse_factor = 4 if min(ds_brain.shape) >= 32 else 1
        edge_hits = _edge_shrink_hits(
            ds_brain,
            spacing,
            centroids,
            n_labels,
            shrink=edge_shrink_factor,
            cache_factor=cache_factor,
            downsample_factor=downsample_factor,
            coarse_factor=coarse_factor,
        )
        brain_mask_cache_used = str(cache_path)
    max_slice_counts_ds = None
    if max_single_slice_voxels > 0:
        max_slice_counts_ds = _per_component_max_slice_counts(labels, n_labels)

    keep_labels, removed_volume, removed_extent, removed_single_slice, removed_edge = select_keep_labels_3d(
        labels,
        stats,
        max_voxels=max_voxels,
        min_voxels=min_voxels,
        max_extent_ratio=max_extent_ratio,
        downsample_factor=downsample_factor,
        max_single_slice_voxels=max_single_slice_voxels,
        max_slice_counts_ds=max_slice_counts_ds,
        edge_hits=edge_hits,
    )
    ds_keep = np.isin(labels, list(keep_labels)).astype(np.uint8)

    _, mask_out = create_output_zarr(
        output_mask_zarr,
        shape=mask_in.shape,
        chunks=mask_in.chunks,
        dtype="uint8",
        dataset_name=dataset_name,
    )
    masked_out = None
    if masked_signal_zarr is not None:
        _, masked_out = create_output_zarr(
            masked_signal_zarr,
            shape=signal.shape,
            chunks=signal.chunks,
            dtype=signal.dtype,
            dataset_name=dataset_name,
        )

    depth, height, width = (int(mask_in.shape[0]), int(mask_in.shape[1]), int(mask_in.shape[2]))
    active_z_set = set(active_z_indices) if active_z_indices else None

    def _keep_block(z0: int, z1: int, y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
        """ds_keep tile upsampled by factor on every axis, cropped to the tile.

        Voxel-wise identical to the per-slice upsample_keep_slice: the value at
        (z, y, x) is ds_keep[z // factor, y // factor, x // factor]."""
        f = int(downsample_factor)
        block = ds_keep[z0 // f : -(-z1 // f), y0 // f : -(-y1 // f), x0 // f : -(-x1 // f)]
        if f > 1:
            block = np.repeat(np.repeat(np.repeat(block, f, axis=0), f, axis=1), f, axis=2)
        return block[: z1 - z0, : y1 - y0, : x1 - x0]

    # Chunk-aligned tiles: reading/writing per z-slice on deep-chunked Zarr
    # re-decompresses every touched chunk once per slice (~65 s/slice on
    # 256^3 chunks); a whole-chunk tile is decompressed exactly once.
    chunk_zyx = tuple(max(1, int(c)) for c in mask_in.chunks)
    z_edges = list(range(0, depth, chunk_zyx[0])) + [depth]
    y_edges = list(range(0, height, chunk_zyx[1])) + [height]
    x_edges = list(range(0, width, chunk_zyx[2])) + [width]
    tiles = [
        (z0, z1, y0, y1, x0, x1)
        for z0, z1 in zip(z_edges, z_edges[1:])
        for y0, y1 in zip(y_edges, y_edges[1:])
        for x0, x1 in zip(x_edges, x_edges[1:])
    ]
    for z0, z1, y0, y1, x0, x1 in tqdm(tiles, desc="Apply 3D filter", unit="tile"):
        mask_block = np.asarray(mask_in[z0:z1, y0:y1, x0:x1], dtype=np.uint8) > 0
        if label_in is not None and region_id_array is not None:
            label_block = np.asarray(label_in[z0:z1, y0:y1, x0:x1])
            filtered = np.empty(mask_block.shape, dtype=np.uint8)
            for zi, z_idx in enumerate(range(z0, z1)):
                mask_slice = mask_block[zi]
                if active_z_set is not None and z_idx not in active_z_set:
                    filtered[zi] = mask_slice
                    continue
                keep_slice = _keep_block(z_idx, z_idx + 1, y0, y1, x0, x1)[0]
                region_slice = _build_region_slice(label_block[zi], region_id_array)
                filtered_inside = mask_slice & (keep_slice > 0)
                if region_outside == "keep":
                    filtered[zi] = np.where(region_slice, filtered_inside, mask_slice)
                else:
                    filtered[zi] = filtered_inside & region_slice
        else:
            keep_block = _keep_block(z0, z1, y0, y1, x0, x1)
            filtered = (mask_block & (keep_block > 0)).astype(np.uint8)
        mask_out[z0:z1, y0:y1, x0:x1] = filtered
        if masked_out is not None:
            signal_block = np.asarray(signal[z0:z1, y0:y1, x0:x1])
            masked_out[z0:z1, y0:y1, x0:x1] = np.where(filtered > 0, signal_block, 0)

    exports: dict[str, str] = {"filtered_mask_zarr": str(output_mask_zarr)}
    if masked_signal_zarr is not None:
        exports["masked_signal_zarr"] = str(masked_signal_zarr)
    if masked_tiff_dir is not None:
        if masked_signal_zarr is None:
            raise ValueError("--masked_tiff_dir requires --masked_signal_zarr")
        masked_tiff_dir.mkdir(parents=True, exist_ok=True)
        export_zarr_to_tiff(
            masked_signal_zarr,
            masked_tiff_dir,
            dataset_name=dataset_name,
            prefix=export_prefix,
        )
        exports["masked_tiff_dir"] = str(masked_tiff_dir)

    return {
        **exports,
        "downsample_factor": int(downsample_factor),
        "max_voxels_full": int(max_voxels),
        "max_voxels_downsampled": int(max(max_voxels // (downsample_factor**3), 1)),
        "max_extent_ratio": float(max_extent_ratio),
        "edge_shrink_factor": float(edge_shrink_factor),
        "brain_mask_ds": brain_mask_cache_used,
        "max_single_slice_voxels": int(max_single_slice_voxels),
        "labels_total_3d": int(n_labels - 1),
        "labels_kept_3d": int(len(keep_labels)),
        "labels_removed_volume_3d": int(removed_volume),
        "labels_removed_extent_3d": int(removed_extent),
        "labels_removed_single_slice_3d": int(removed_single_slice),
        "labels_removed_edge_3d": int(removed_edge),
        "region_query": region_query or "",
        "region_name": region_name or "",
        "region_ids_count": int(len(region_ids)),
        "region_outside": region_outside,
        "active_slices": int(len(active_z_indices) if active_z_indices else depth),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="3D cc3d mask filtering on a downsampled mask, then apply to signal.",
    )
    parser.add_argument("--signal_zarr", type=Path, required=True)
    parser.add_argument("--mask_zarr", type=Path, required=True)
    parser.add_argument("--output_mask_zarr", type=Path, required=True)
    parser.add_argument("--masked_signal_zarr", type=Path, default=None)
    parser.add_argument("--masked_tiff_dir", type=Path, default=None)
    parser.add_argument(
        "--max_voxels",
        type=int,
        default=1000,
        help="Maximum 3D object volume at full resolution.",
    )
    parser.add_argument(
        "--min_voxels",
        type=int,
        default=1,
        help="Minimum 3D object volume at full resolution.",
    )
    parser.add_argument(
        "--max_extent_ratio",
        type=float,
        default=3.0,
        help="Maximum allowed max-axis/min-axis ratio from 3D bounding box (0 disables).",
    )
    parser.add_argument(
        "--edge_shrink_factor",
        type=float,
        default=0.0,
        help="AIND-style brain-boundary exclusion: shrink the brain-surface mesh "
        "about its center of mass by this factor and drop components whose "
        "centroid falls outside it (e.g. 0.99; 0 disables). Uses the factor-4 "
        "brain-mask cache written by registration (or built once from --label_zarr).",
    )
    parser.add_argument(
        "--brain_mask_ds",
        type=Path,
        default=None,
        help="Factor-4 brain-mask cache zarr (default: <label_zarr>_ds_brain_mask.zarr).",
    )
    parser.add_argument(
        "--max_single_slice_voxels",
        type=int,
        default=0,
        help="Drop objects whose largest single-slice footprint exceeds this many "
        "full-resolution voxels (0 disables).",
    )
    parser.add_argument(
        "--downsample_factor",
        type=int,
        default=4,
        help="Isotropic downsample factor before 3D connected components.",
    )
    parser.add_argument(
        "--label_zarr",
        type=Path,
        default=None,
        help="Registered atlas label Zarr in sample space (required with --region; "
        "also used to build the brain-mask cache when it does not exist yet).",
    )
    parser.add_argument(
        "--region",
        default=None,
        help="Brain region acronym/name/id; filtering runs only inside this subtree (e.g. cc).",
    )
    parser.add_argument(
        "--region_cfg",
        type=Path,
        default=None,
        help=f"Allen region CSV for --region lookup (default: {DEFAULT_REGION_CFG}).",
    )
    parser.add_argument(
        "--region_outside",
        choices=("keep", "zero"),
        default="keep",
        help="Outside --region: keep original mask (keep) or set to zero (zero).",
    )
    parser.add_argument("--dataset_name", default="0")
    parser.add_argument("--export_prefix", default="masked_")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = postprocess_cfos_mask_3d(
        signal_zarr=args.signal_zarr,
        mask_zarr=args.mask_zarr,
        output_mask_zarr=args.output_mask_zarr,
        masked_signal_zarr=args.masked_signal_zarr,
        masked_tiff_dir=args.masked_tiff_dir,
        max_voxels=args.max_voxels,
        min_voxels=args.min_voxels,
        max_extent_ratio=args.max_extent_ratio,
        edge_shrink_factor=args.edge_shrink_factor,
        max_single_slice_voxels=args.max_single_slice_voxels,
        downsample_factor=args.downsample_factor,
        label_zarr=args.label_zarr,
        brain_mask_ds=args.brain_mask_ds,
        region_query=args.region,
        region_cfg=args.region_cfg,
        region_outside=args.region_outside,
        dataset_name=args.dataset_name,
        export_prefix=args.export_prefix,
    )
    for key, value in result.items():
        print(f"{key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
