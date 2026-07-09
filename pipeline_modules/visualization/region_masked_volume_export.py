"""Export a sample-space brain region as a masked 3D TIFF stack."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Literal

import numpy as np
import tifffile
from tqdm import tqdm

from pipeline_modules.preprocessing.tiff_to_zarr import parse_channel_list, resolve_tiff_workers
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
SOURCE_CHOICES = ("auto", "tiff", "zarr")


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


def _list_tiff_stack(path: Path) -> list[Path]:
    files = sorted(path.glob("*.tif*"))
    if not files:
        raise FileNotFoundError(f"No TIFF files found in {path}")
    return files


def _pair_tiff_stacks(
    signal_dir: Path,
    label_dir: Path,
    mask_dir: Path | None = None,
) -> tuple[list[Path], list[Path], list[Path] | None]:
    signal_files = _list_tiff_stack(signal_dir)
    label_files = _list_tiff_stack(label_dir)
    if len(signal_files) != len(label_files):
        raise ValueError(
            f"Signal and label TIFF stacks must have the same slice count, "
            f"got {len(signal_files)} vs {len(label_files)} "
            f"({signal_dir} vs {label_dir})"
        )
    mask_files = None
    if mask_dir is not None:
        mask_files = _list_tiff_stack(mask_dir)
        if len(mask_files) != len(signal_files):
            raise ValueError(
                f"Mask TIFF stack must match signal slice count, "
                f"got {len(mask_files)} vs {len(signal_files)} ({mask_dir})"
            )
    return signal_files, label_files, mask_files


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


def _resolve_output_dtype(
    *,
    export_mode: str,
    output_dtype: str,
    signal_sample: np.ndarray | None,
    mask_sample: np.ndarray | None,
    label_sample: np.ndarray,
) -> np.dtype:
    if output_dtype != "preserve":
        return np.dtype(output_dtype)
    if export_mode == "signal" and signal_sample is not None:
        return np.dtype(signal_sample.dtype)
    if export_mode == "mask":
        return np.dtype(mask_sample.dtype if mask_sample is not None else np.uint8)
    return np.dtype(label_sample.dtype)


def _render_masked_slice(
    *,
    export_mode: str,
    label_slice: np.ndarray,
    region_id_array: np.ndarray,
    save_dtype: np.dtype,
    signal_slice: np.ndarray | None = None,
    mask_slice: np.ndarray | None = None,
    foreground_label: int = 1,
) -> np.ndarray:
    region_mask = _build_region_slice(label_slice, region_id_array)
    if export_mode == "region":
        slice_out = region_mask.astype(save_dtype, copy=False)
    elif export_mode == "mask":
        if mask_slice is None:
            raise ValueError("export_mode=mask requires a segmentation mask stack.")
        slice_out = np.where(region_mask & (mask_slice == foreground_label), foreground_label, 0).astype(
            save_dtype,
            copy=False,
        )
    else:
        if signal_slice is None:
            raise ValueError("export_mode=signal requires a signal stack.")
        slice_out = _masked_signal_slice(
            signal_slice,
            label_slice,
            region_id_array,
            mask_slice=mask_slice,
            foreground_label=foreground_label,
        )
    return _cast_slice_for_save(slice_out, output_dtype=save_dtype)


def _resolve_source_mode(
    *,
    source: str,
    layout: SampleLayout,
    signal_tiff_dir: Path | None,
    label_tiff_dir: Path | None,
    signal_zarr_path: Path | None,
    label_zarr_path: Path | None,
) -> Literal["tiff", "zarr"]:
    mode = str(source).strip().lower()
    if mode not in SOURCE_CHOICES:
        raise ValueError(f"source must be one of {SOURCE_CHOICES}, got: {source!r}")

    signal_tiff = signal_tiff_dir or layout.signal_tiff_dir
    label_tiff = label_tiff_dir or layout.atlas_label_tiff_dir
    signal_zarr = signal_zarr_path or layout.signal_zarr
    label_zarr = label_zarr_path or layout.atlas_label_zarr

    tiff_ready = signal_tiff.is_dir() and label_tiff.is_dir()
    zarr_ready = label_zarr.exists() and (signal_zarr.exists() if signal_zarr else True)

    if mode == "tiff":
        if not tiff_ready:
            raise FileNotFoundError(
                f"TIFF export requested but stacks are missing: signal={signal_tiff}, label={label_tiff}"
            )
        return "tiff"
    if mode == "zarr":
        if not label_zarr.exists():
            raise FileNotFoundError(f"Zarr export requested but label Zarr is missing: {label_zarr}")
        return "zarr"
    if tiff_ready:
        return "tiff"
    if zarr_ready:
        return "zarr"
    raise FileNotFoundError(
        "Could not resolve export source. Expected TIFF stacks "
        f"({signal_tiff} + {label_tiff}) or Zarr stores ({signal_zarr} + {label_zarr})."
    )


def _default_output_dir(
    sample_dir: Path,
    *,
    signal_ch: str,
    region_slug: str,
    export_mode: str,
) -> Path:
    return sample_dir / "visualization" / "region_masked" / f"{signal_ch}_{region_slug}_{export_mode}"


def _process_tiff_slice_job(job: dict[str, object]) -> tuple[int, int, int]:
    label_slice = np.asarray(tifffile.imread(str(job["label_path"])))
    signal_slice = (
        np.asarray(tifffile.imread(str(job["signal_path"])))
        if job.get("signal_path") is not None
        else None
    )
    mask_slice = (
        np.asarray(tifffile.imread(str(job["mask_path"])))
        if job.get("mask_path") is not None
        else None
    )
    slice_out = _render_masked_slice(
        export_mode=str(job["export_mode"]),
        label_slice=label_slice,
        region_id_array=job["region_id_array"],  # type: ignore[arg-type]
        save_dtype=job["save_dtype"],  # type: ignore[arg-type]
        signal_slice=signal_slice,
        mask_slice=mask_slice,
        foreground_label=int(job["foreground_label"]),
    )
    tifffile.imwrite(str(job["output_path"]), slice_out, compression=job["compression"])
    return (
        int(job["z_index"]),
        int(np.any(slice_out)),
        int(np.count_nonzero(slice_out)),
    )


def export_region_masked_volume_tiffs_from_tiff(
    *,
    sample_dir: str | Path,
    region_query: str,
    output_dir: str | Path | None = None,
    cfg_path: str | Path | None = None,
    signal_tiff_dir: str | Path | None = None,
    label_tiff_dir: str | Path | None = None,
    mask_tiff_dir: str | Path | None = None,
    config_path: str | Path | None = None,
    signal_ch: str | None = None,
    foreground_label: int = 1,
    export_mode: str = "signal",
    filename_prefix: str = "C1",
    z_pad: int = 6,
    output_dtype: str = "preserve",
    compression: str = "lzw",
    workers: int = 0,
    mirror_input_filenames: bool = False,
) -> dict[str, object]:
    if export_mode not in EXPORT_MODES:
        raise ValueError(f"export_mode must be one of {EXPORT_MODES}, got: {export_mode!r}")

    sample_dir = Path(sample_dir)
    defaults = resolve_sample_stack_defaults(sample_dir, config_path=config_path)
    resolved_signal_ch = signal_ch or str(defaults["signal_ch"])
    layout = SampleLayout(sample_dir=sample_dir, signal_ch=resolved_signal_ch, reg_ch=str(defaults["register_ch"]))

    signal_dir = Path(signal_tiff_dir or layout.signal_tiff_dir)
    label_dir = Path(label_tiff_dir or layout.atlas_label_tiff_dir)
    mask_dir = Path(mask_tiff_dir) if mask_tiff_dir else None
    if export_mode == "mask" and mask_dir is None:
        mask_dir = layout.mask_tiff_dir
    if export_mode == "signal" and not signal_dir.is_dir():
        raise FileNotFoundError(f"Signal TIFF directory not found: {signal_dir}")
    if not label_dir.is_dir():
        raise FileNotFoundError(f"Atlas label TIFF directory not found: {label_dir}")

    signal_files: list[Path] | None
    if export_mode == "signal":
        signal_files, label_files, mask_files = _pair_tiff_stacks(signal_dir, label_dir, mask_dir)
    else:
        label_files = _list_tiff_stack(label_dir)
        signal_files = None
        mask_files = _list_tiff_stack(mask_dir) if mask_dir is not None else None
        if mask_files is not None and len(mask_files) != len(label_files):
            raise ValueError(
                f"Mask TIFF stack must match label slice count, got {len(mask_files)} vs {len(label_files)}"
            )

    region_cfg = Path(cfg_path or DEFAULT_REGION_CFG)
    subtree_ids, region_slug, region_name = resolve_region_subtree_ids(region_query, cfg_path=region_cfg)
    label_sample = np.asarray(tifffile.imread(str(label_files[0])))
    signal_sample = (
        np.asarray(tifffile.imread(str(signal_files[0])))
        if signal_files is not None
        else None
    )
    mask_sample = (
        np.asarray(tifffile.imread(str(mask_files[0])))
        if mask_files is not None
        else None
    )
    region_id_array = np.asarray(list(subtree_ids), dtype=label_sample.dtype)
    save_dtype = _resolve_output_dtype(
        export_mode=export_mode,
        output_dtype=output_dtype,
        signal_sample=signal_sample,
        mask_sample=mask_sample,
        label_sample=label_sample,
    )

    out_dir = Path(
        output_dir
        or _default_output_dir(sample_dir, signal_ch=resolved_signal_ch, region_slug=region_slug, export_mode=export_mode)
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    jobs: list[dict[str, object]] = []
    for z_index, label_path in enumerate(label_files):
        if mirror_input_filenames and signal_files is not None:
            output_name = signal_files[z_index].name
        else:
            output_name = f"{filename_prefix}{z_index:0{z_pad}d}.tiff"
        jobs.append(
            {
                "z_index": z_index,
                "label_path": label_path,
                "signal_path": signal_files[z_index] if signal_files is not None else None,
                "mask_path": mask_files[z_index] if mask_files is not None else None,
                "output_path": out_dir / output_name,
                "export_mode": export_mode,
                "region_id_array": region_id_array,
                "save_dtype": save_dtype,
                "foreground_label": foreground_label,
                "compression": compression,
            }
        )

    worker_count = resolve_tiff_workers(workers)
    written_slices = 0
    nonzero_voxels = 0
    if worker_count <= 1:
        iterator = jobs
        if sys.stderr.isatty():
            iterator = tqdm(jobs, desc=f"Export {region_slug} {export_mode}", unit="slice", leave=False, file=sys.stderr)
        for job in iterator:
            _, nonempty, nonzero = _process_tiff_slice_job(job)
            written_slices += nonempty
            nonzero_voxels += nonzero
    else:
        max_in_flight = max(1, min(worker_count + 1, len(jobs)))
        pending: dict[object, None] = {}
        job_iter = iter(jobs)
        with ThreadPoolExecutor(max_workers=worker_count) as pool:
            def submit_next() -> None:
                try:
                    pending[pool.submit(_process_tiff_slice_job, next(job_iter))] = None
                except StopIteration:
                    return

            for _ in range(max_in_flight):
                submit_next()
            progress = tqdm(total=len(jobs), desc=f"Export {region_slug} {export_mode}", unit="slice", leave=False, file=sys.stderr)
            try:
                while pending:
                    for future in as_completed(list(pending)):
                        pending.pop(future)
                        _, nonempty, nonzero = future.result()
                        written_slices += nonempty
                        nonzero_voxels += nonzero
                        progress.update(1)
                        submit_next()
                        break
            finally:
                progress.close()

    summary = {
        "sample_dir": str(sample_dir),
        "source": "tiff",
        "region_query": region_query,
        "region_name": region_name,
        "region_slug": region_slug,
        "region_ids_count": len(subtree_ids),
        "export_mode": export_mode,
        "signal_ch": resolved_signal_ch,
        "signal_tiff_dir": str(signal_dir) if export_mode == "signal" else None,
        "label_tiff_dir": str(label_dir),
        "mask_tiff_dir": str(mask_dir) if mask_dir else None,
        "output_dir": str(out_dir),
        "filename_prefix": filename_prefix,
        "mirror_input_filenames": mirror_input_filenames,
        "slice_count": len(label_files),
        "nonempty_slices": written_slices,
        "nonzero_voxels": nonzero_voxels,
        "output_dtype": str(save_dtype),
        "workers": worker_count,
    }
    summary_path = out_dir / f"{resolved_signal_ch}_{region_slug}_{export_mode}_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    summary["summary_json"] = str(summary_path)
    return summary


def export_region_masked_volume_tiffs_from_zarr(
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
    save_dtype = _resolve_output_dtype(
        export_mode=export_mode,
        output_dtype=output_dtype,
        signal_sample=np.asarray(label_arr[0]) if export_mode != "signal" else np.asarray(signal_arr[0]),
        mask_sample=np.asarray(mask_arr[0]) if mask_arr is not None else None,
        label_sample=np.asarray(label_arr[0]),
    )

    out_dir = Path(
        output_dir
        or _default_output_dir(sample_dir, signal_ch=resolved_signal_ch, region_slug=region_slug, export_mode=export_mode)
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    depth = int(label_arr.shape[0])
    written_slices = 0
    nonzero_voxels = 0
    for z_index in tqdm(range(depth), desc=f"Export {region_slug} {export_mode}", unit="slice", leave=False, file=sys.stderr):
        label_slice = np.asarray(label_arr[z_index])
        signal_slice = np.asarray(signal_arr[z_index]) if signal_arr is not None else None
        mask_slice = np.asarray(mask_arr[z_index]) if mask_arr is not None else None
        slice_out = _render_masked_slice(
            export_mode=export_mode,
            label_slice=label_slice,
            region_id_array=region_id_array,
            save_dtype=save_dtype,
            signal_slice=signal_slice,
            mask_slice=mask_slice,
            foreground_label=foreground_label,
        )
        if np.any(slice_out):
            written_slices += 1
            nonzero_voxels += int(np.count_nonzero(slice_out))
        output_path = out_dir / f"{filename_prefix}{z_index:0{z_pad}d}.tiff"
        tifffile.imwrite(str(output_path), slice_out, compression=compression)

    summary = {
        "sample_dir": str(sample_dir),
        "source": "zarr",
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


def export_region_masked_volume_tiffs(
    *,
    sample_dir: str | Path,
    region_query: str,
    source: str = "auto",
    output_dir: str | Path | None = None,
    cfg_path: str | Path | None = None,
    signal_tiff_dir: str | Path | None = None,
    label_tiff_dir: str | Path | None = None,
    mask_tiff_dir: str | Path | None = None,
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
    workers: int = 0,
    mirror_input_filenames: bool = False,
) -> dict[str, object]:
    sample_dir = Path(sample_dir)
    defaults = resolve_sample_stack_defaults(sample_dir, config_path=config_path)
    resolved_signal_ch = signal_ch or str(defaults["signal_ch"])
    layout = SampleLayout(sample_dir=sample_dir, signal_ch=resolved_signal_ch, reg_ch=str(defaults["register_ch"]))
    resolved_source = _resolve_source_mode(
        source=source,
        layout=layout,
        signal_tiff_dir=Path(signal_tiff_dir) if signal_tiff_dir else None,
        label_tiff_dir=Path(label_tiff_dir) if label_tiff_dir else None,
        signal_zarr_path=Path(signal_zarr_path) if signal_zarr_path else None,
        label_zarr_path=Path(label_zarr_path) if label_zarr_path else None,
    )
    if resolved_source == "tiff":
        return export_region_masked_volume_tiffs_from_tiff(
            sample_dir=sample_dir,
            region_query=region_query,
            output_dir=output_dir,
            cfg_path=cfg_path,
            signal_tiff_dir=signal_tiff_dir,
            label_tiff_dir=label_tiff_dir,
            mask_tiff_dir=mask_tiff_dir,
            config_path=config_path,
            signal_ch=resolved_signal_ch,
            foreground_label=foreground_label,
            export_mode=export_mode,
            filename_prefix=filename_prefix,
            z_pad=z_pad,
            output_dtype=output_dtype,
            compression=compression,
            workers=workers,
            mirror_input_filenames=mirror_input_filenames,
        )
    return export_region_masked_volume_tiffs_from_zarr(
        sample_dir=sample_dir,
        region_query=region_query,
        output_dir=output_dir,
        cfg_path=cfg_path,
        signal_zarr_path=signal_zarr_path,
        label_zarr_path=label_zarr_path,
        mask_zarr_path=mask_zarr_path,
        config_path=config_path,
        signal_ch=resolved_signal_ch,
        dataset_name=dataset_name,
        foreground_label=foreground_label,
        export_mode=export_mode,
        filename_prefix=filename_prefix,
        z_pad=z_pad,
        output_dtype=output_dtype,
        compression=compression,
    )


def export_region_masked_volume_tiffs_for_channels(
    *,
    sample_dir: str | Path,
    region_query: str,
    channels: str | list[str],
    cfg_path: str | Path | None = None,
    label_tiff_dir: str | Path | None = None,
    label_zarr_path: str | Path | None = None,
    config_path: str | Path | None = None,
    dataset_name: str = "0",
    foreground_label: int = 1,
    export_mode: str = "signal",
    filename_prefix: str = "C1",
    z_pad: int = 6,
    output_dtype: str = "preserve",
    compression: str = "lzw",
    workers: int = 0,
    source: str = "auto",
    mirror_input_filenames: bool = False,
    output_root: str | Path | None = None,
) -> dict[str, object]:
    sample_dir = Path(sample_dir)
    channel_labels = parse_channel_list(channels)
    channel_results: dict[str, dict[str, object]] = {}
    skipped_channels: dict[str, str] = {}

    for channel in channel_labels:
        try:
            output_dir = (
                Path(output_root) / f"{channel}_{_sanitize_slug(region_query)}_{export_mode}"
                if output_root
                else _default_output_dir(
                    sample_dir,
                    signal_ch=channel,
                    region_slug=_sanitize_slug(region_query),
                    export_mode=export_mode,
                )
            )
            channel_results[channel] = export_region_masked_volume_tiffs(
                sample_dir=sample_dir,
                region_query=region_query,
                source=source,
                output_dir=output_dir,
                cfg_path=cfg_path,
                label_tiff_dir=label_tiff_dir,
                label_zarr_path=label_zarr_path,
                config_path=config_path,
                signal_ch=channel,
                dataset_name=dataset_name,
                foreground_label=foreground_label,
                export_mode=export_mode,
                filename_prefix=filename_prefix,
                z_pad=z_pad,
                output_dtype=output_dtype,
                compression=compression,
                workers=workers,
                mirror_input_filenames=mirror_input_filenames,
            )
        except (FileNotFoundError, ValueError) as exc:
            skipped_channels[channel] = str(exc)

    if not channel_results:
        raise ValueError(
            f"No channels exported. Requested={channel_labels}, skipped={skipped_channels}"
        )

    _, region_slug, region_name = resolve_region_subtree_ids(
        region_query,
        cfg_path=Path(cfg_path or DEFAULT_REGION_CFG),
    )
    return {
        "sample_dir": str(sample_dir),
        "source": source,
        "region_query": region_query,
        "region_name": region_name,
        "region_slug": region_slug,
        "export_mode": export_mode,
        "channels_requested": channel_labels,
        "skipped_channels": skipped_channels,
        "channels": channel_results,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Mask one Allen brain region in sample space and export a full 3D masked TIFF stack. "
            "Defaults to TIFF stacks (chX/ + upsampled_atlas_label/) without Zarr conversion."
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
        help="signal=masked signal (default), mask=segmentation mask inside region, region=binary atlas mask",
    )
    parser.add_argument(
        "--source",
        choices=SOURCE_CHOICES,
        default="auto",
        help="auto prefers TIFF stacks; use zarr only when TIFF folders are unavailable",
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
        help="Comma-separated channels to export from TIFF stacks, e.g. '1,2,3'",
    )
    parser.add_argument("--signal-ch", default="", help="Single signal channel label when --channels is omitted, e.g. ch3")
    parser.add_argument("--signal-tiff-dir", default="", help="Override signal TIFF folder")
    parser.add_argument(
        "--label-tiff-dir",
        default="",
        help="Override atlas label TIFF folder; default is sample_dir/upsampled_atlas_label",
    )
    parser.add_argument("--mask-tiff-dir", default="", help="Optional segmentation mask TIFF folder")
    parser.add_argument("--signal-zarr", default="", help="Override signal Zarr path when --source zarr")
    parser.add_argument("--label-zarr", default="", help="Override atlas label Zarr path when --source zarr")
    parser.add_argument("--mask-zarr", default="", help="Optional segmentation mask Zarr when --source zarr")
    parser.add_argument("--dataset-name", default="0", help="Zarr dataset name when --source zarr")
    parser.add_argument("--foreground-label", type=int, default=1, help="Foreground label when using mask stack")
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
        "--workers",
        type=int,
        default=0,
        help="Parallel slice workers for TIFF mode. 0 uses cpu_count // 2.",
    )
    parser.add_argument(
        "--mirror-input-filenames",
        action="store_true",
        help="Write outputs using the same filenames as the input signal TIFF slices",
    )
    parser.add_argument("--filename-prefix", default="C1", help="Output slice prefix when not mirroring input names")
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
    common_kwargs = {
        "sample_dir": args.sample_dir,
        "region_query": args.region,
        "source": args.source,
        "cfg_path": args.cfg,
        "label_tiff_dir": args.label_tiff_dir or None,
        "label_zarr_path": args.label_zarr or None,
        "config_path": args.config or None,
        "dataset_name": args.dataset_name,
        "foreground_label": int(args.foreground_label),
        "export_mode": args.export_mode,
        "filename_prefix": args.filename_prefix,
        "z_pad": int(args.z_pad),
        "output_dtype": args.output_dtype,
        "compression": args.compression or None,
        "workers": int(args.workers),
        "mirror_input_filenames": bool(args.mirror_input_filenames),
    }
    if args.channels:
        payload = export_region_masked_volume_tiffs_for_channels(
            channels=args.channels,
            output_root=args.output_root or None,
            **common_kwargs,
        )
    else:
        payload = export_region_masked_volume_tiffs(
            output_dir=args.output_dir or None,
            signal_tiff_dir=args.signal_tiff_dir or None,
            mask_tiff_dir=args.mask_tiff_dir or None,
            signal_zarr_path=args.signal_zarr or None,
            mask_zarr_path=args.mask_zarr or None,
            signal_ch=args.signal_ch or None,
            **common_kwargs,
        )
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
