"""Export a sample-space brain region as a masked 3D TIFF stack."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import tifffile
from tqdm import tqdm

from pipeline_modules.preprocessing.tiff_to_zarr import (
    convert_sample_channels_to_zarr,
    parse_channel_list,
)
from pipeline_modules.segmentation.zarr_utils import open_zarr_dataset
from pipeline_modules.tubule_reconstruction.region_vessel_analysis import (
    _collect_subtree_ids,
    load_region_tree_with_lookups,
    resolve_region_query,
)
from pipeline_modules.utils.sample_layout import SampleLayout
from pipeline_modules.visualization.heatmap import resolve_sample_stack_defaults
from pipeline_modules.visualization.region_mip_export import DEFAULT_REGION_CFG

EXPORT_MODES = ("signal", "mask", "region")


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


def _cast_slice_for_save(array: np.ndarray, *, output_dtype: np.dtype) -> np.ndarray:
    if array.dtype == output_dtype:
        return array
    if np.issubdtype(output_dtype, np.integer):
        info = np.iinfo(output_dtype)
        return np.clip(np.rint(array), info.min, info.max).astype(output_dtype)
    return array.astype(output_dtype)


def _build_region_slice(
    label_slice: np.ndarray,
    region_id_array: np.ndarray,
) -> np.ndarray:
    return np.isin(label_slice, region_id_array)


def _masked_signal_slice(
    signal_slice: np.ndarray,
    label_slice: np.ndarray,
    region_id_array: np.ndarray,
    *,
    mask_slice: np.ndarray | None,
    foreground_label: int,
) -> np.ndarray:
    region_mask = _build_region_slice(label_slice, region_id_array)
    if mask_slice is not None:
        region_mask &= mask_slice == foreground_label
    return np.where(region_mask, signal_slice, 0)


def export_region_masked_volume_tiffs(
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
    export_mode: str = "signal",
    filename_prefix: str = "C1",
    z_pad: int = 6,
    output_dtype: str = "preserve",
    compression: str = "lzw",
) -> dict[str, object]:
    if export_mode not in EXPORT_MODES:
        raise ValueError(f"export_mode must be one of {EXPORT_MODES}, got: {export_mode!r}")

    sample_dir = Path(sample_dir)
    defaults = resolve_sample_stack_defaults(sample_dir, config_path=config_path)
    resolved_signal_ch = signal_ch or str(defaults["signal_ch"])
    layout = SampleLayout(sample_dir=sample_dir, signal_ch=resolved_signal_ch, reg_ch=str(defaults["register_ch"]))

    label_path = Path(label_zarr_path or layout.atlas_label_zarr)
    signal_path = Path(signal_zarr_path or layout.signal_zarr) if export_mode == "signal" else None
    mask_path = Path(mask_zarr_path) if mask_zarr_path else None
    if export_mode == "mask" and mask_path is None:
        mask_path = layout.mask_zarr
    region_cfg = Path(cfg_path or DEFAULT_REGION_CFG)

    label_arr = open_zarr_dataset(label_path, dataset_name=dataset_name)
    signal_arr = open_zarr_dataset(signal_path, dataset_name=dataset_name) if signal_path else None
    mask_arr = open_zarr_dataset(mask_path, dataset_name=dataset_name) if mask_path else None

    if export_mode == "signal":
        if signal_arr is None:
            raise ValueError("export_mode=signal requires a signal Zarr.")
        if signal_arr.shape != label_arr.shape:
            raise ValueError(
                f"signal and label Zarr shapes must match, got {signal_arr.shape} vs {label_arr.shape}"
            )
    if mask_arr is not None and mask_arr.shape != label_arr.shape:
        raise ValueError(
            f"mask Zarr shape must match label shape, got {mask_arr.shape} vs {label_arr.shape}"
        )

    subtree_ids, region_slug, region_name = resolve_region_subtree_ids(region_query, cfg_path=region_cfg)
    region_id_array = np.asarray(list(subtree_ids), dtype=label_arr.dtype)

    if output_dtype == "preserve":
        if export_mode == "signal" and signal_arr is not None:
            save_dtype = np.dtype(signal_arr.dtype)
        elif export_mode == "mask":
            save_dtype = np.dtype(mask_arr.dtype if mask_arr is not None else np.uint8)
        else:
            save_dtype = np.dtype(label_arr.dtype)
    else:
        save_dtype = np.dtype(output_dtype)

    out_dir = Path(
        output_dir
        or (sample_dir / "visualization" / "region_masked" / f"{resolved_signal_ch}_{region_slug}_{export_mode}")
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    depth = int(label_arr.shape[0])
    written_slices = 0
    nonzero_voxels = 0
    for z_index in tqdm(range(depth), desc=f"Export {region_slug} {export_mode}", unit="slice", leave=False, file=sys.stderr):
        label_slice = np.asarray(label_arr[z_index])
        region_mask = _build_region_slice(label_slice, region_id_array)

        if export_mode == "region":
            slice_out = region_mask.astype(save_dtype, copy=False)
        elif export_mode == "mask":
            if mask_arr is None:
                raise ValueError("export_mode=mask requires --mask-zarr or sample_dir/<signal_ch>_mask.zarr")
            mask_slice = np.asarray(mask_arr[z_index])
            slice_out = np.where(region_mask & (mask_slice == foreground_label), foreground_label, 0).astype(
                save_dtype,
                copy=False,
            )
        else:
            if signal_arr is None:
                raise ValueError("export_mode=signal requires a signal Zarr.")
            signal_slice = np.asarray(signal_arr[z_index])
            mask_slice = np.asarray(mask_arr[z_index]) if mask_arr is not None else None
            slice_out = _masked_signal_slice(
                signal_slice,
                label_slice,
                region_id_array,
                mask_slice=mask_slice,
                foreground_label=foreground_label,
            )

        slice_out = _cast_slice_for_save(slice_out, output_dtype=save_dtype)
        if np.any(slice_out):
            written_slices += 1
            nonzero_voxels += int(np.count_nonzero(slice_out))
        output_path = out_dir / f"{filename_prefix}{z_index:0{z_pad}d}.tiff"
        tifffile.imwrite(str(output_path), slice_out, compression=compression)

    summary = {
        "sample_dir": str(sample_dir),
        "region_query": region_query,
        "region_name": region_name,
        "region_slug": region_slug,
        "region_ids_count": len(subtree_ids),
        "export_mode": export_mode,
        "signal_ch": resolved_signal_ch,
        "signal_zarr": str(signal_path) if signal_path else None,
        "label_zarr": str(label_path),
        "mask_zarr": str(mask_path) if mask_path else None,
        "output_dir": str(out_dir),
        "filename_prefix": filename_prefix,
        "slice_count": depth,
        "nonempty_slices": written_slices,
        "nonzero_voxels": nonzero_voxels,
        "output_dtype": str(save_dtype),
    }
    summary_path = out_dir / f"{resolved_signal_ch}_{region_slug}_{export_mode}_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    summary["summary_json"] = str(summary_path)
    return summary


def export_region_masked_volume_tiffs_for_channels(
    *,
    sample_dir: str | Path,
    region_query: str,
    channels: str | list[str],
    cfg_path: str | Path | None = None,
    label_zarr_path: str | Path | None = None,
    config_path: str | Path | None = None,
    dataset_name: str = "0",
    foreground_label: int = 1,
    export_mode: str = "signal",
    filename_prefix: str = "C1",
    z_pad: int = 6,
    output_dtype: str = "preserve",
    compression: str = "lzw",
    convert_tiff_workers: int = 0,
    chunk_size: tuple[int, int, int] = (256, 256, 256),
    skip_existing_zarr: bool = True,
    output_root: str | Path | None = None,
) -> dict[str, object]:
    sample_dir = Path(sample_dir)
    channel_labels = parse_channel_list(channels)
    conversion_summary = convert_sample_channels_to_zarr(
        sample_dir,
        channel_labels,
        chunk_size=chunk_size,
        workers=convert_tiff_workers,
        dataset_name=dataset_name,
        skip_existing=skip_existing_zarr,
    )

    channel_results: dict[str, dict[str, object]] = {}
    skipped_channels: dict[str, str] = {}
    for channel in channel_labels:
        zarr_path = sample_dir / f"{channel}.zarr"
        if not zarr_path.exists():
            skipped_channels[channel] = "Missing Zarr after conversion attempt"
            continue
        output_dir = (
            Path(output_root) / f"{channel}_{_sanitize_slug(region_query)}_{export_mode}"
            if output_root
            else sample_dir / "visualization" / "region_masked" / f"{channel}_{_sanitize_slug(region_query)}_{export_mode}"
        )
        channel_results[channel] = export_region_masked_volume_tiffs(
            sample_dir=sample_dir,
            region_query=region_query,
            output_dir=output_dir,
            cfg_path=cfg_path,
            signal_zarr_path=zarr_path,
            label_zarr_path=label_zarr_path,
            mask_zarr_path=None,
            config_path=config_path,
            signal_ch=channel,
            dataset_name=dataset_name,
            foreground_label=foreground_label,
            export_mode=export_mode,
            filename_prefix=filename_prefix,
            z_pad=z_pad,
            output_dtype=output_dtype,
            compression=compression,
        )

    if not channel_results:
        raise ValueError(
            f"No channels exported. Requested={channel_labels}, conversion={conversion_summary}, skipped={skipped_channels}"
        )

    subtree_ids, region_slug, region_name = resolve_region_subtree_ids(
        region_query,
        cfg_path=Path(cfg_path or DEFAULT_REGION_CFG),
    )
    return {
        "sample_dir": str(sample_dir),
        "region_query": region_query,
        "region_name": region_name,
        "region_slug": region_slug,
        "export_mode": export_mode,
        "channels_requested": channel_labels,
        "conversion": conversion_summary,
        "skipped_channels": skipped_channels,
        "channels": channel_results,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Mask one Allen brain region in sample space using upsampled_atlas_label.zarr "
            "and export a full 3D TIFF stack."
        )
    )
    parser.add_argument("--sample-dir", required=True, help="Sample root directory")
    parser.add_argument(
        "--region",
        required=True,
        help="Brain region acronym, full name, or numeric Allen id (includes subtree by default)",
    )
    parser.add_argument(
        "--export-mode",
        choices=EXPORT_MODES,
        default="signal",
        help="signal=masked signal (default), mask=cFos/segmentation mask inside region, region=binary atlas mask",
    )
    parser.add_argument(
        "--cfg",
        default=str(DEFAULT_REGION_CFG),
        help="Allen region CSV used to resolve --region",
    )
    parser.add_argument("--config", default="", help="Optional pipeline config.json for channel defaults")
    parser.add_argument(
        "--channels",
        default="",
        help="Comma-separated channels to export, e.g. 0,1,2,3. Converts missing chX.zarr from chX/ TIFF folders first.",
    )
    parser.add_argument("--signal-ch", default="", help="Single signal channel label when --channels is omitted, e.g. ch3")
    parser.add_argument("--signal-zarr", default="", help="Override signal Zarr path")
    parser.add_argument("--label-zarr", default="", help="Override atlas label Zarr path")
    parser.add_argument("--mask-zarr", default="", help="Optional segmentation mask Zarr for export-mode signal/mask")
    parser.add_argument("--dataset-name", default="0", help="Zarr dataset name")
    parser.add_argument("--foreground-label", type=int, default=1, help="Foreground label when using mask Zarr")
    parser.add_argument(
        "--output-dir",
        default="",
        help="Output TIFF directory for single-channel mode; default is sample_dir/visualization/region_masked/{ch}_{region}_{mode}",
    )
    parser.add_argument(
        "--output-root",
        default="",
        help="Root directory for multi-channel exports; default is sample_dir/visualization/region_masked",
    )
    parser.add_argument(
        "--convert-tiff-workers",
        type=int,
        default=0,
        help="Worker threads when converting chX/ TIFF folders to chX.zarr. 0 uses cpu_count // 4.",
    )
    parser.add_argument(
        "--chunk-size",
        default="256,256,256",
        help="Zarr chunk size z,y,x used when converting missing TIFF folders",
    )
    parser.add_argument(
        "--force-reconvert-zarr",
        action="store_true",
        help="Rebuild chX.zarr even when it already exists (multi-channel mode)",
    )
    parser.add_argument("--filename-prefix", default="C1", help="Output slice prefix, e.g. C1 -> C1000001.tiff")
    parser.add_argument("--z-pad", type=int, default=6, help="Zero-padding width for slice index in filenames")
    parser.add_argument(
        "--output-dtype",
        default="preserve",
        help="TIFF dtype: preserve (default) or an explicit numpy dtype such as uint16",
    )
    parser.add_argument(
        "--compression",
        default="lzw",
        help="TIFF compression passed to tifffile.imwrite (use '' for none)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.channels:
        payload = export_region_masked_volume_tiffs_for_channels(
            sample_dir=args.sample_dir,
            region_query=args.region,
            channels=args.channels,
            cfg_path=args.cfg,
            label_zarr_path=args.label_zarr or None,
            config_path=args.config or None,
            dataset_name=args.dataset_name,
            foreground_label=int(args.foreground_label),
            export_mode=args.export_mode,
            filename_prefix=args.filename_prefix,
            z_pad=int(args.z_pad),
            output_dtype=args.output_dtype,
            compression=args.compression or None,
            convert_tiff_workers=int(args.convert_tiff_workers),
            chunk_size=_parse_block_shape(args.chunk_size),
            skip_existing_zarr=not bool(args.force_reconvert_zarr),
            output_root=args.output_root or None,
        )
    else:
        payload = export_region_masked_volume_tiffs(
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
            export_mode=args.export_mode,
            filename_prefix=args.filename_prefix,
            z_pad=int(args.z_pad),
            output_dtype=args.output_dtype,
            compression=args.compression or None,
        )
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
