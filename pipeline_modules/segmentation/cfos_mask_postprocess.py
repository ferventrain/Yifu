from __future__ import annotations

import argparse
from pathlib import Path
from typing import Literal

import cc3d
import numpy as np
from skimage.measure import block_reduce
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
    Path(__file__).resolve().parents[1] / "registration" / "Region_Csv_Rev1_updated.CSV"
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
    if depth % factor or height % factor or width % factor:
        raise ValueError(
            f"Mask shape {(depth, height, width)} must be divisible by downsample factor {factor}"
        )
    if (label_in is None) ^ (region_id_array is None):
        raise ValueError("label_in and region_id_array must be provided together")

    ds_shape = (depth // factor, height // factor, width // factor)
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
        ds_mask[zb0 // factor : zb1 // factor] = block_reduce(
            slab,
            block_size=(factor, factor, factor),
            func=np.max,
        )
        if ds_region is not None:
            label_slab = np.asarray(label_in[zb0:zb1])
            region_slab = _build_region_slice(label_slab, region_id_array).astype(np.uint8)
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
    shell_hits: np.ndarray | None = None,
) -> tuple[set[int], int, int, int, int]:
    """Decide which 3D components survive the standard filters.

    Volumes/areas are given at full resolution and converted to the
    downsampled grid. A filter value <= 0 disables that filter:

    - ``min_voxels`` / ``max_voxels``: 3D object volume bounds
    - ``max_extent_ratio``: bounding-box max/min axis ratio
    - ``max_single_slice_voxels``: drop objects whose largest single-slice
      footprint (vessels, surface sheets) exceeds this many voxels
    - ``shell_hits``: per-label voxel counts inside the excluded near-boundary
      shell; any hit removes the object
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
        if shell_hits is not None and int(shell_hits[label_idx]) > 0:
            removed_edge += 1
            continue

        keep_labels.add(label_idx)

    return keep_labels, removed_volume, removed_extent, removed_single_slice, removed_edge


def upsample_keep_slice(ds_keep: np.ndarray, factor: int, height: int, width: int) -> np.ndarray:
    upsampled = np.repeat(np.repeat(ds_keep, factor, axis=0), factor, axis=1)
    return upsampled[:height, :width]


def _downsample_foreground(label_arr, factor: int, ds_shape: tuple[int, int, int]) -> np.ndarray:
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
    for zb0 in range(0, depth, z_block):
        zb1 = min(zb0 + z_block, depth)
        slab = (np.asarray(label_arr[zb0:zb1]) > 0).astype(np.uint8)
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


def _boundary_shell_hits(
    label_in,
    *,
    factor: int,
    ds_shape: tuple[int, int, int],
    labels: np.ndarray,
    n_labels: int,
    exclude_edge_px: int,
) -> tuple[np.ndarray, int]:
    """Per-label voxel counts inside the near-brain-surface exclusion shell.

    The brain surface comes from the registered atlas label (label > 0); it is
    downsampled, eroded by ceil(exclude_edge_px / factor) voxels, and the
    shell between the surface and the eroded core defines the exclusion zone.
    """
    from scipy import ndimage as ndi

    ds_brain = _downsample_foreground(label_in, factor, ds_shape)
    iterations = max(1, -(-int(exclude_edge_px) // int(factor)))
    core = ndi.binary_erosion(
        ds_brain.astype(bool),
        structure=ndi.generate_binary_structure(3, 1),
        iterations=iterations,
    )
    shell = (ds_brain > 0) & ~core
    if not np.any(shell):
        return np.zeros(n_labels, dtype=np.int64), iterations
    shell_hits = np.bincount(labels[shell].astype(np.int64).ravel(), minlength=n_labels)
    return shell_hits[:n_labels], iterations


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
    exclude_edge_px: int = 0,
    max_single_slice_voxels: int = 0,
    label_zarr: Path | None = None,
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
    if region_query or exclude_edge_px > 0:
        if label_zarr is None:
            raise ValueError("--region and --exclude_edge_px require --label_zarr")
        label_in = open_zarr_dataset(label_zarr, dataset_name=dataset_name)
        if label_in.shape != mask_in.shape:
            raise ValueError(f"Shape mismatch: mask={mask_in.shape}, label={label_in.shape}")
        if region_query:
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

    shell_hits = None
    erosion_iterations = 0
    if exclude_edge_px > 0:
        shell_hits, erosion_iterations = _boundary_shell_hits(
            label_in,
            factor=downsample_factor,
            ds_shape=tuple(int(v) for v in ds_mask.shape),
            labels=labels,
            n_labels=n_labels,
            exclude_edge_px=exclude_edge_px,
        )
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
        shell_hits=shell_hits,
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
        "exclude_edge_px": int(exclude_edge_px),
        "edge_shell_erosion_iterations": int(erosion_iterations),
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
        "--exclude_edge_px",
        type=int,
        default=0,
        help="Drop objects with any voxel within this many pixels inside the brain "
        "surface (atlas label boundary); requires --label_zarr (0 disables).",
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
        help="Registered atlas label Zarr in sample space (required with --region "
        "or --exclude_edge_px).",
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
        exclude_edge_px=args.exclude_edge_px,
        max_single_slice_voxels=args.max_single_slice_voxels,
        downsample_factor=args.downsample_factor,
        label_zarr=args.label_zarr,
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
