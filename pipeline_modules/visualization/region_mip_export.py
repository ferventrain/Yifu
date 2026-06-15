"""Export maximum-intensity projections for a masked Allen brain region in sample space."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import tifffile

from pipeline_modules.segmentation.zarr_utils import open_zarr_dataset
from pipeline_modules.tubule_reconstruction.region_vessel_analysis import (
    _collect_subtree_ids,
    load_region_tree_with_lookups,
    resolve_region_query,
)
from pipeline_modules.utils.sample_layout import SampleLayout
from pipeline_modules.visualization.heatmap import resolve_sample_stack_defaults
from pipeline_modules.visualization.warp_mask_zarr_to_atlas_points import iter_block_slices

PLANES = ("horizontal", "coronal", "sagittal")
DEFAULT_REGION_CFG = (
    Path(__file__).resolve().parents[2] / "pipeline_modules" / "registration" / "Region_Csv_Rev1_updated.CSV"
)


def _sanitize_slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip())
    return slug.strip("_") or "region"


def _parse_block_shape(value: str) -> tuple[int, int, int]:
    parts = [int(part.strip()) for part in str(value).split(",") if part.strip()]
    if len(parts) != 3:
        raise ValueError(f"block_shape must be z,y,x with three integers, got: {value}")
    if any(part <= 0 for part in parts):
        raise ValueError(f"block_shape values must be positive, got: {parts}")
    return tuple(parts)  # type: ignore[return-value]


def resolve_region_subtree_ids(region_query: str, *, cfg_path: str | Path) -> tuple[set[int], str, str]:
    nodes_by_id, acronym_to_ids, name_to_ids = load_region_tree_with_lookups(cfg_path)
    node = resolve_region_query(region_query, nodes_by_id, acronym_to_ids, name_to_ids)
    subtree_ids = set(_collect_subtree_ids(node))
    slug = _sanitize_slug(node.get("acronym") or node.get("name") or str(node["id"]))
    display_name = str(node.get("name") or slug)
    return subtree_ids, slug, display_name


def _mip_output_shapes(volume_shape: tuple[int, int, int]) -> dict[str, tuple[int, int]]:
    z_size, y_size, x_size = volume_shape
    return {
        "horizontal": (y_size, x_size),
        "coronal": (z_size, x_size),
        "sagittal": (z_size, y_size),
    }


def _init_mip_buffers(volume_shape: tuple[int, int, int], *, dtype: np.dtype) -> dict[str, np.ndarray]:
    buffers: dict[str, np.ndarray] = {}
    for plane, out_shape in _mip_output_shapes(volume_shape).items():
        buffers[plane] = np.zeros(out_shape, dtype=dtype)
    return buffers


def _update_mip_buffers(
    buffers: dict[str, np.ndarray],
    masked_block: np.ndarray,
    block_slices: tuple[slice, slice, slice],
) -> None:
    z_slice, y_slice, x_slice = block_slices
    block_horizontal = np.max(masked_block, axis=0)
    block_coronal = np.max(masked_block, axis=1)
    block_sagittal = np.max(masked_block, axis=2)

    buffers["horizontal"][y_slice, x_slice] = np.maximum(
        buffers["horizontal"][y_slice, x_slice],
        block_horizontal,
    )
    buffers["coronal"][z_slice, x_slice] = np.maximum(
        buffers["coronal"][z_slice, x_slice],
        block_coronal,
    )
    buffers["sagittal"][z_slice, y_slice] = np.maximum(
        buffers["sagittal"][z_slice, y_slice],
        block_sagittal,
    )


def compute_region_mips(
    signal_volume: np.ndarray,
    label_volume: np.ndarray,
    region_ids: Iterable[int],
    *,
    mask_volume: np.ndarray | None = None,
    foreground_label: int = 1,
    block_shape: tuple[int, int, int] = (128, 256, 256),
) -> dict[str, np.ndarray]:
    if signal_volume.shape != label_volume.shape:
        raise ValueError(
            f"signal and label shapes must match, got {signal_volume.shape} vs {label_volume.shape}"
        )
    if mask_volume is not None and mask_volume.shape != signal_volume.shape:
        raise ValueError(
            f"mask shape must match signal shape, got {mask_volume.shape} vs {signal_volume.shape}"
        )

    region_id_array = np.asarray(list(region_ids), dtype=label_volume.dtype)
    accumulation_dtype = np.result_type(signal_volume.dtype, np.uint32)
    buffers = _init_mip_buffers(signal_volume.shape, dtype=accumulation_dtype)

    for block_slices in iter_block_slices(signal_volume.shape, block_shape):
        signal_block = np.asarray(signal_volume[block_slices])
        label_block = np.asarray(label_volume[block_slices])
        region_mask = np.isin(label_block, region_id_array)
        if mask_volume is not None:
            region_mask &= np.asarray(mask_volume[block_slices]) == foreground_label
        if not np.any(region_mask):
            continue
        masked_block = np.where(region_mask, signal_block, 0).astype(accumulation_dtype, copy=False)
        _update_mip_buffers(buffers, masked_block, block_slices)

    return buffers


def compute_region_mips_from_zarr(
    *,
    signal_zarr_path: str | Path,
    label_zarr_path: str | Path,
    region_ids: Iterable[int],
    mask_zarr_path: str | Path | None = None,
    dataset_name: str = "0",
    foreground_label: int = 1,
    block_shape: tuple[int, int, int] = (128, 256, 256),
) -> dict[str, np.ndarray]:
    signal_arr = open_zarr_dataset(signal_zarr_path, dataset_name=dataset_name)
    label_arr = open_zarr_dataset(label_zarr_path, dataset_name=dataset_name)
    if signal_arr.shape != label_arr.shape:
        raise ValueError(
            f"signal and label Zarr shapes must match, got {signal_arr.shape} vs {label_arr.shape}"
        )

    mask_arr = open_zarr_dataset(mask_zarr_path, dataset_name=dataset_name) if mask_zarr_path else None
    if mask_arr is not None and mask_arr.shape != signal_arr.shape:
        raise ValueError(
            f"mask Zarr shape must match signal shape, got {mask_arr.shape} vs {signal_arr.shape}"
        )

    region_id_array = np.asarray(list(region_ids), dtype=label_arr.dtype)
    accumulation_dtype = np.result_type(signal_arr.dtype, np.uint32)
    buffers = _init_mip_buffers(signal_arr.shape, dtype=accumulation_dtype)

    for block_slices in iter_block_slices(signal_arr.shape, block_shape):
        signal_block = np.asarray(signal_arr[block_slices])
        label_block = np.asarray(label_arr[block_slices])
        region_mask = np.isin(label_block, region_id_array)
        if mask_arr is not None:
            region_mask &= np.asarray(mask_arr[block_slices]) == foreground_label
        if not np.any(region_mask):
            continue
        masked_block = np.where(region_mask, signal_block, 0).astype(accumulation_dtype, copy=False)
        _update_mip_buffers(buffers, masked_block, block_slices)

    return buffers


def export_region_mip_tiffs(
    *,
    sample_dir: str | Path,
    region_query: str,
    output_dir: str | Path | None = None,
    cfg_path: str | Path | None = None,
    signal_zarr_path: str | Path | None = None,
    label_zarr_path: str | Path | None = None,
    mask_zarr_path: str | Path | None = None,
    config_path: str | Path | None = None,
    signal_ch: str | None = None,
    dataset_name: str = "0",
    foreground_label: int = 1,
    block_shape: tuple[int, int, int] = (128, 256, 256),
    output_dtype: str = "preserve",
) -> dict[str, object]:
    sample_dir = Path(sample_dir)
    defaults = resolve_sample_stack_defaults(sample_dir, config_path=config_path)
    resolved_signal_ch = signal_ch or str(defaults["signal_ch"])
    layout = SampleLayout(sample_dir=sample_dir, signal_ch=resolved_signal_ch, reg_ch=str(defaults["register_ch"]))

    signal_path = Path(signal_zarr_path or layout.signal_zarr)
    label_path = Path(label_zarr_path or layout.atlas_label_zarr)
    mask_path = Path(mask_zarr_path) if mask_zarr_path else None
    region_cfg = Path(cfg_path or DEFAULT_REGION_CFG)

    subtree_ids, region_slug, region_name = resolve_region_subtree_ids(region_query, cfg_path=region_cfg)
    mips = compute_region_mips_from_zarr(
        signal_zarr_path=signal_path,
        label_zarr_path=label_path,
        region_ids=subtree_ids,
        mask_zarr_path=mask_path,
        dataset_name=dataset_name,
        foreground_label=foreground_label,
        block_shape=block_shape,
    )

    out_dir = Path(output_dir or (sample_dir / "visualization" / "region_mip"))
    out_dir.mkdir(parents=True, exist_ok=True)

    signal_arr = open_zarr_dataset(signal_path, dataset_name=dataset_name)
    if output_dtype == "preserve":
        save_dtype = np.dtype(signal_arr.dtype)
    else:
        save_dtype = np.dtype(output_dtype)

    outputs: dict[str, str] = {}
    for plane in PLANES:
        mip = mips[plane]
        if save_dtype != mip.dtype:
            if np.issubdtype(save_dtype, np.integer):
                info = np.iinfo(save_dtype)
                mip = np.clip(np.rint(mip), info.min, info.max).astype(save_dtype)
            else:
                mip = mip.astype(save_dtype)
        filename = f"{sample_dir.name}_{region_slug}_{plane}.tiff"
        output_path = out_dir / filename
        tifffile.imwrite(str(output_path), mip, compression="lzw")
        outputs[plane] = str(output_path)

    summary = {
        "sample_dir": str(sample_dir),
        "region_query": region_query,
        "region_name": region_name,
        "region_slug": region_slug,
        "region_ids_count": len(subtree_ids),
        "signal_zarr": str(signal_path),
        "label_zarr": str(label_path),
        "mask_zarr": str(mask_path) if mask_path else None,
        "block_shape": list(block_shape),
        "outputs": outputs,
    }
    summary_path = out_dir / f"{sample_dir.name}_{region_slug}_mip_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    summary["summary_json"] = str(summary_path)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Mask one Allen brain region in sample space and export horizontal/coronal/sagittal "
            "maximum-intensity projection TIFFs."
        )
    )
    parser.add_argument("--sample-dir", required=True, help="Sample root directory")
    parser.add_argument(
        "--region",
        required=True,
        help="Brain region acronym, full name, or numeric Allen id (includes subtree by default)",
    )
    parser.add_argument(
        "--cfg",
        default=str(DEFAULT_REGION_CFG),
        help="Allen region CSV used to resolve --region",
    )
    parser.add_argument("--config", default="", help="Optional pipeline config.json for channel defaults")
    parser.add_argument("--signal-ch", default="", help="Signal channel label, e.g. ch1")
    parser.add_argument("--signal-zarr", default="", help="Override signal Zarr path")
    parser.add_argument("--label-zarr", default="", help="Override atlas label Zarr path")
    parser.add_argument("--mask-zarr", default="", help="Optional segmentation mask Zarr for foreground gating")
    parser.add_argument("--dataset-name", default="0", help="Zarr dataset name")
    parser.add_argument("--foreground-label", type=int, default=1, help="Foreground label when --mask-zarr is set")
    parser.add_argument(
        "--block-shape",
        default="128,256,256",
        help="Chunk shape for streaming MIP in z,y,x order",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="Output directory; default is sample_dir/visualization/region_mip",
    )
    parser.add_argument(
        "--output-dtype",
        default="preserve",
        help="TIFF dtype: preserve (default) or an explicit numpy dtype such as uint16",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    payload = export_region_mip_tiffs(
        sample_dir=args.sample_dir,
        region_query=args.region,
        output_dir=args.output_dir or None,
        cfg_path=args.cfg,
        signal_zarr_path=args.signal_zarr or None,
        label_zarr_path=args.label_zarr or None,
        mask_zarr_path=args.mask_zarr or None,
        config_path=args.config or None,
        signal_ch=args.signal_ch or None,
        dataset_name=args.dataset_name,
        foreground_label=int(args.foreground_label),
        block_shape=_parse_block_shape(args.block_shape),
        output_dtype=args.output_dtype,
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
