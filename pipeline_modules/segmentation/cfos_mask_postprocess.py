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

    for ds_z in tqdm(range(ds_shape[0]), desc="Downsample mask", unit="slab"):
        z0 = ds_z * factor
        slab = np.asarray(mask_in[z0 : z0 + factor], dtype=np.uint8) > 0
        ds_mask[ds_z] = block_reduce(
            slab.astype(np.uint8),
            block_size=(factor, factor, factor),
            func=np.max,
        )[0]
        if ds_region is not None:
            label_slab = np.asarray(label_in[z0 : z0 + factor])
            region_slab = _build_region_slice(label_slab, region_id_array).astype(np.uint8)
            ds_region[ds_z] = block_reduce(
                region_slab,
                block_size=(factor, factor, factor),
                func=np.max,
            )[0]
            if ds_region[ds_z].any():
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
) -> tuple[set[int], int, int]:
    keep_labels: set[int] = set()
    removed_volume = 0
    removed_extent = 0

    scale_volume = downsample_factor ** 3
    max_voxels_ds = max(int(max_voxels // scale_volume), 1)
    min_voxels_ds = max(int(min_voxels // scale_volume), 1)

    voxel_counts = stats["voxel_counts"]
    bounding_boxes = stats["bounding_boxes"]

    for label_idx in range(1, len(voxel_counts)):
        count_ds = int(voxel_counts[label_idx])
        if count_ds < min_voxels_ds:
            removed_volume += 1
            continue
        if count_ds > max_voxels_ds:
            removed_volume += 1
            continue

        ratio = _extent_ratio(bounding_boxes[label_idx])
        if ratio > max_extent_ratio:
            removed_extent += 1
            continue

        keep_labels.add(label_idx)

    return keep_labels, removed_volume, removed_extent


def upsample_keep_slice(ds_keep: np.ndarray, factor: int, height: int, width: int) -> np.ndarray:
    upsampled = np.repeat(np.repeat(ds_keep, factor, axis=0), factor, axis=1)
    return upsampled[:height, :width]


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
        label_in=label_in,
        region_id_array=region_id_array,
    )
    if ds_region is not None:
        ds_mask = (ds_mask > 0) & (ds_region > 0)
        ds_mask = ds_mask.astype(np.uint8)
    labels = cc3d.connected_components(ds_mask, connectivity=26)
    stats = cc3d.statistics(labels)
    print(f"3D connected components: {len(stats['voxel_counts']) - 1} objects in filter scope", flush=True)

    keep_labels, removed_volume, removed_extent = select_keep_labels_3d(
        labels,
        stats,
        max_voxels=max_voxels,
        min_voxels=min_voxels,
        max_extent_ratio=max_extent_ratio,
        downsample_factor=downsample_factor,
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
    for z_idx in tqdm(range(depth), desc="Apply 3D filter", unit="slice"):
        mask_slice = np.asarray(mask_in[z_idx], dtype=np.uint8) > 0
        if label_in is not None and region_id_array is not None:
            if active_z_set is not None and z_idx not in active_z_set:
                filtered = mask_slice.astype(np.uint8)
            else:
                ds_z = z_idx // downsample_factor
                keep_slice = upsample_keep_slice(
                    ds_keep[ds_z],
                    downsample_factor,
                    height,
                    width,
                )
                region_slice = _build_region_slice(np.asarray(label_in[z_idx]), region_id_array)
                filtered_inside = mask_slice & (keep_slice > 0)
                if region_outside == "keep":
                    filtered = np.where(region_slice, filtered_inside, mask_slice).astype(np.uint8)
                else:
                    filtered = (filtered_inside & region_slice).astype(np.uint8)
        else:
            ds_z = z_idx // downsample_factor
            keep_slice = upsample_keep_slice(
                ds_keep[ds_z],
                downsample_factor,
                height,
                width,
            )
            filtered = (mask_slice & (keep_slice > 0)).astype(np.uint8)
        mask_out[z_idx] = filtered
        if masked_out is not None:
            signal_slice = np.asarray(signal[z_idx])
            masked_out[z_idx] = np.where(filtered > 0, signal_slice, 0)

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
        "labels_total_3d": int(len(stats["voxel_counts"]) - 1),
        "labels_kept_3d": int(len(keep_labels)),
        "labels_removed_volume_3d": int(removed_volume),
        "labels_removed_extent_3d": int(removed_extent),
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
        help="Maximum allowed max-axis/min-axis ratio from 3D bounding box.",
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
        help="Registered atlas label Zarr in sample space (required with --region).",
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
