from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

try:
    from pipeline_modules.qc.grading import grade_qc_results, load_threshold_rules
    from pipeline_modules.qc.ims_io import (
        DEFAULT_IMS_CHANNEL,
        DEFAULT_IMS_HISTOGRAM_Z_CHUNKS,
        DEFAULT_IMS_HDF_CACHE_MB,
        DEFAULT_IMS_MAX_SLICES,
        DEFAULT_IMS_RESOLUTION_LEVEL,
        DEFAULT_IMS_TIMEPOINT,
        DEFAULT_IMS_Z_CHUNK,
        effective_z_chunk,
        group_z_indices_into_read_ranges,
        ims_source_meta,
        iter_ims_histogram_blocks,
        open_ims_dataset,
        select_histogram_z_chunk_ids,
        select_qc_z_indices,
        stack_ordered_slices,
    )
    from pipeline_modules.qc.progress import QcProgressTracker, count_zarr_blocks
    from pipeline_modules.qc.metrics import (
        DEFAULT_DARK_PIXEL_THRESHOLD,
        DEFAULT_LOW_CONTRAST_THRESHOLD,
        aggregate_metric_dicts,
        compute_exposure_dynamic_range_metrics,
        compute_slice_metrics,
        flatten_metric_groups,
    )
    from pipeline_modules.segmentation.zarr_utils import open_zarr_dataset
    from pipeline_modules.utils.errors import ErrorCode, PipelineError
    from pipeline_modules.utils.run_manifest import write_run_manifest
    from pipeline_modules.utils.tiff_range_to_niigz import list_tiff_files, load_tiff_as_zyx
except ImportError:  # pragma: no cover
    from .grading import grade_qc_results, load_threshold_rules
    from .ims_io import (
        DEFAULT_IMS_CHANNEL,
        DEFAULT_IMS_HISTOGRAM_Z_CHUNKS,
        DEFAULT_IMS_HDF_CACHE_MB,
        DEFAULT_IMS_MAX_SLICES,
        DEFAULT_IMS_RESOLUTION_LEVEL,
        DEFAULT_IMS_TIMEPOINT,
        DEFAULT_IMS_Z_CHUNK,
        effective_z_chunk,
        group_z_indices_into_read_ranges,
        ims_source_meta,
        iter_ims_histogram_blocks,
        open_ims_dataset,
        select_histogram_z_chunk_ids,
        select_qc_z_indices,
        stack_ordered_slices,
    )
    from .progress import QcProgressTracker, count_zarr_blocks
    from .metrics import (
        DEFAULT_DARK_PIXEL_THRESHOLD,
        DEFAULT_LOW_CONTRAST_THRESHOLD,
        aggregate_metric_dicts,
        compute_exposure_dynamic_range_metrics,
        compute_slice_metrics,
        flatten_metric_groups,
    )
    from ..segmentation.zarr_utils import open_zarr_dataset
    from ..utils.errors import ErrorCode, PipelineError
    from ..utils.run_manifest import write_run_manifest
    from ..utils.tiff_range_to_niigz import list_tiff_files, load_tiff_as_zyx


DEFAULT_TILE_SIZE = 256
DEFAULT_N_SLABS = 32
DEFAULT_MAX_SLICES = 16
DEFAULT_HISTOGRAM_BINS = 65536
DEFAULT_THRESHOLDS_PATH = Path(__file__).resolve().parent / "qc_thresholds.json"


@dataclass
class ImageQcConfig:
    tile_size: int = DEFAULT_TILE_SIZE
    n_slabs: int = DEFAULT_N_SLABS
    max_slices: int = DEFAULT_MAX_SLICES
    projection: str = "none"
    fft_enabled: bool = False
    histogram_bins: int = DEFAULT_HISTOGRAM_BINS
    saturation_margin: float = 0.001
    dark_pixel_threshold: float = DEFAULT_DARK_PIXEL_THRESHOLD
    low_contrast_threshold: float = DEFAULT_LOW_CONTRAST_THRESHOLD
    dataset_name: str = "0"
    thresholds_path: str | Path | None = str(DEFAULT_THRESHOLDS_PATH)
    grading_enabled: bool = True
    show_progress: bool = True
    ims_resolution_level: int = DEFAULT_IMS_RESOLUTION_LEVEL
    ims_channel: int = DEFAULT_IMS_CHANNEL
    ims_timepoint: int = DEFAULT_IMS_TIMEPOINT
    ims_histogram_z_chunks: int = DEFAULT_IMS_HISTOGRAM_Z_CHUNKS
    ims_z_chunk: int = 0
    ims_hdf_cache_mb: int = DEFAULT_IMS_HDF_CACHE_MB
    ims_z_chunk_align: bool = True
    nas_qc: bool = False


def _select_z_indices(shape_z: int, max_slices: int) -> list[int]:
    z_size = int(shape_z)
    if z_size <= 0:
        return []
    count = min(max(int(max_slices), 1), z_size)
    if count == 1:
        return [z_size // 2]
    return sorted({int(round(v)) for v in np.linspace(0, z_size - 1, count)})


def _projection_2d(volume_zyx: np.ndarray, projection: str) -> np.ndarray:
    if projection == "max":
        return np.max(volume_zyx, axis=0)
    if projection == "mean":
        return np.mean(volume_zyx, axis=0)
    raise PipelineError(
        ErrorCode.ARGUMENT_INVALID,
        "projection must be one of: none, max, mean",
        {"projection": projection},
    )


def _resolve_input_source(
    *,
    input_zarr: str | Path | None,
    input_tiff_dir: str | Path | None,
    input_tiff: str | Path | None,
    input_ims: str | Path | None,
) -> tuple[str, Path]:
    sources = [input_zarr, input_tiff_dir, input_tiff, input_ims]
    provided = [source for source in sources if source]
    if len(provided) != 1:
        raise PipelineError(
            ErrorCode.ARGUMENT_INVALID,
            "Provide exactly one of --input_zarr, --input_tiff_dir, --input_tiff, or --input_ims",
            {
                "input_zarr": input_zarr,
                "input_tiff_dir": input_tiff_dir,
                "input_tiff": input_tiff,
                "input_ims": input_ims,
            },
        )
    if input_zarr:
        path = Path(input_zarr)
        if not path.exists():
            raise PipelineError(ErrorCode.INPUT_NOT_FOUND, "Input Zarr not found", {"input_zarr": str(path)})
        return "zarr", path
    if input_tiff_dir:
        path = Path(input_tiff_dir)
        if not path.exists():
            raise PipelineError(
                ErrorCode.INPUT_NOT_FOUND,
                "Input TIFF directory not found",
                {"input_tiff_dir": str(path)},
            )
        return "tiff_dir", path
    if input_ims:
        path = Path(input_ims)
        if not path.exists():
            raise PipelineError(ErrorCode.INPUT_NOT_FOUND, "Input IMS not found", {"input_ims": str(path)})
        return "ims", path
    path = Path(input_tiff)
    if not path.exists():
        raise PipelineError(ErrorCode.INPUT_NOT_FOUND, "Input TIFF not found", {"input_tiff": str(path)})
    return "tiff", path


def _apply_nas_qc_defaults(config: ImageQcConfig) -> ImageQcConfig:
    if not config.nas_qc:
        return config
    config.max_slices = DEFAULT_IMS_MAX_SLICES
    config.projection = "none"
    config.ims_resolution_level = DEFAULT_IMS_RESOLUTION_LEVEL
    config.ims_histogram_z_chunks = DEFAULT_IMS_HISTOGRAM_Z_CHUNKS
    config.ims_z_chunk_align = True
    if config.ims_z_chunk <= 0:
        config.ims_z_chunk = DEFAULT_IMS_Z_CHUNK
    return config


def _histogram_edges(values: np.ndarray, bins: int) -> tuple[np.ndarray, int]:
    arr = np.asarray(values)
    if np.issubdtype(arr.dtype, np.integer):
        info = np.iinfo(arr.dtype)
        if bins >= (info.max - info.min + 1):
            edges = np.arange(info.min, info.max + 2, dtype=np.float64)
            return edges, len(edges) - 1
    vmin = float(np.min(values))
    vmax = float(np.max(values))
    if vmax <= vmin:
        vmax = vmin + 1.0
    return np.linspace(vmin, vmax, int(bins) + 1, dtype=np.float64), int(bins)


def _histogram_edges_from_dtype(dtype: np.dtype, bins: int) -> np.ndarray:
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        if bins >= (info.max - info.min + 1):
            return np.arange(info.min, info.max + 2, dtype=np.float64)
    return np.linspace(0.0, 1.0, int(bins) + 1, dtype=np.float64)


def _percentiles_from_histogram(counts: np.ndarray, edges: np.ndarray, percentiles: Iterable[int]) -> dict[str, float]:
    total = float(np.sum(counts))
    if total <= 0:
        return {f"p{p}": 0.0 for p in percentiles}
    cumulative = np.cumsum(counts)
    result: dict[str, float] = {}
    for p in percentiles:
        target = total * (float(p) / 100.0)
        idx = int(np.searchsorted(cumulative, target, side="left"))
        idx = min(max(idx, 0), len(edges) - 2)
        left = edges[idx]
        right = edges[idx + 1]
        prev = cumulative[idx - 1] if idx > 0 else 0.0
        bin_count = counts[idx]
        if bin_count <= 0 or right <= left:
            value = left
        else:
            fraction = (target - prev) / float(bin_count)
            value = left + fraction * (right - left)
        result[f"p{int(p)}"] = float(value)
    return result


def _saturation_threshold_from_edges(edges: np.ndarray, saturation_margin: float) -> float:
    dtype_min = float(edges[0])
    dtype_max = float(edges[-1])
    span = max(dtype_max - dtype_min, 1.0)
    return dtype_max - saturation_margin * span


def _accumulate_block_stats(
    blocks: Iterable[np.ndarray],
    *,
    edges: np.ndarray,
    dark_pixel_threshold: float,
    saturation_threshold: float,
) -> tuple[np.ndarray, int, int, int]:
    counts = np.zeros(len(edges) - 1, dtype=np.int64)
    dark_count = 0
    saturated_count = 0
    total_count = 0
    for block in blocks:
        arr = np.asarray(block)
        if arr.size == 0:
            continue
        flat = arr.ravel()
        counts += np.histogram(flat, bins=edges)[0]
        dark_count += int(np.sum(flat < dark_pixel_threshold))
        saturated_count += int(np.sum(flat >= saturation_threshold))
        total_count += int(flat.size)
    return counts, dark_count, saturated_count, total_count


def _global_exposure_from_accumulators(
    counts: np.ndarray,
    edges: np.ndarray,
    *,
    dark_count: int,
    saturated_count: int,
    total_count: int,
    dark_pixel_threshold: float,
) -> dict[str, float]:
    if total_count <= 0:
        return compute_exposure_dynamic_range_metrics(
            np.zeros(1, dtype=np.float64),
            dark_pixel_threshold=dark_pixel_threshold,
        )

    centers = (edges[:-1] + edges[1:]) / 2.0
    total = float(total_count)
    mean = float(np.sum(centers * counts) / max(float(np.sum(counts)), 1.0))
    percentiles = _percentiles_from_histogram(counts, edges, (1, 5, 10, 25, 50, 75, 90, 95, 99))
    p1 = percentiles["p1"]
    p99 = percentiles["p99"]
    robust_dynamic_range = max(p99 - p1, 0.0)
    dtype_min = float(edges[0])
    dtype_max = float(edges[-1])
    span = max(dtype_max - dtype_min, 1.0)

    return {
        "mean": mean,
        "median": percentiles["p50"],
        **{k: v for k, v in percentiles.items() if k in {"p1", "p5", "p25", "p50", "p75", "p95", "p99"}},
        "saturated_pixel_ratio": saturated_count / total,
        "dark_pixel_ratio": dark_count / total,
        "robust_dynamic_range": robust_dynamic_range,
        "dynamic_range_utilization": robust_dynamic_range / span,
        "dark_pixel_threshold": float(dark_pixel_threshold),
    }


def compute_global_exposure_metrics(
    volume_zyx: np.ndarray,
    *,
    histogram_bins: int,
    saturation_margin: float,
    dark_pixel_threshold: float,
) -> dict[str, float]:
    arr = np.asarray(volume_zyx)
    edges, _ = _histogram_edges(arr, histogram_bins)
    saturation_threshold = _saturation_threshold_from_edges(edges, saturation_margin)
    counts, dark_count, saturated_count, total_count = _accumulate_block_stats(
        [arr],
        edges=edges,
        dark_pixel_threshold=dark_pixel_threshold,
        saturation_threshold=saturation_threshold,
    )
    return _global_exposure_from_accumulators(
        counts,
        edges,
        dark_count=dark_count,
        saturated_count=saturated_count,
        total_count=total_count,
        dark_pixel_threshold=dark_pixel_threshold,
    )


def _iter_zarr_blocks(dataset: Any):
    shape = tuple(int(v) for v in dataset.shape)
    chunks = tuple(int(v) for v in getattr(dataset, "chunks", shape))
    for z0 in range(0, shape[0], chunks[0]):
        z1 = min(z0 + chunks[0], shape[0])
        for y0 in range(0, shape[1], chunks[1]):
            y1 = min(y0 + chunks[1], shape[1])
            for x0 in range(0, shape[2], chunks[2]):
                x1 = min(x0 + chunks[2], shape[2])
                yield np.asarray(dataset[z0:z1, y0:y1, x0:x1])


def _iter_tiff_dir_blocks(path: Path):
    for file_path in list_tiff_files(path):
        yield load_tiff_as_zyx(file_path)


def _compute_global_exposure_streaming(
    blocks: Iterable[np.ndarray],
    *,
    dtype: np.dtype,
    histogram_bins: int,
    saturation_margin: float,
    dark_pixel_threshold: float,
    progress: QcProgressTracker | None = None,
    progress_desc: str = "Histogram chunks",
    progress_total: int | None = None,
) -> dict[str, float]:
    block_iter = blocks
    if progress is not None:
        block_iter = progress.iter(
            blocks,
            desc=progress_desc,
            total=progress_total,
            unit="chunk",
        )
    edges = _histogram_edges_from_dtype(dtype, histogram_bins)
    saturation_threshold = _saturation_threshold_from_edges(edges, saturation_margin)
    counts, dark_count, saturated_count, total_count = _accumulate_block_stats(
        block_iter,
        edges=edges,
        dark_pixel_threshold=dark_pixel_threshold,
        saturation_threshold=saturation_threshold,
    )
    return _global_exposure_from_accumulators(
        counts,
        edges,
        dark_count=dark_count,
        saturated_count=saturated_count,
        total_count=total_count,
        dark_pixel_threshold=dark_pixel_threshold,
    )


def _global_percentiles_from_exposure(global_exposure: dict[str, float]) -> dict[str, float]:
    p1 = global_exposure.get("p1", 0.0)
    p99 = global_exposure.get("p99", 0.0)
    span = max(p99 - p1, 1e-6)
    return {
        "p10": float(global_exposure.get("p10", p1 + 0.09 * span)),
        "p25": float(global_exposure.get("p25", p1 + 0.24 * span)),
        "p50": float(global_exposure.get("median", p1 + 0.49 * span)),
        "p75": float(global_exposure.get("p75", p1 + 0.74 * span)),
        "p90": float(global_exposure.get("p90", p1 + 0.89 * span)),
    }


def _compute_slice_records(
    slices_zyx: np.ndarray,
    *,
    z_indices: list[int],
    global_exposure: dict[str, float],
    config: ImageQcConfig,
    progress: QcProgressTracker | None = None,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    global_percentiles = _global_percentiles_from_exposure(global_exposure)
    slice_records: list[dict[str, Any]] = []
    flat_metric_rows: list[dict[str, float]] = []

    iterator = enumerate(z_indices)
    if progress is not None:
        iterator = progress.iter(
            list(enumerate(z_indices)),
            desc="Slice metrics",
            total=len(z_indices),
            unit="slice",
        )

    for offset, z_index in iterator:
        slice_2d = slices_zyx[offset]
        groups = compute_slice_metrics(
            slice_2d,
            tile_size=config.tile_size,
            n_slabs=config.n_slabs,
            fft_enabled=config.fft_enabled,
            global_percentiles=global_percentiles,
            dark_pixel_threshold=config.dark_pixel_threshold,
            low_contrast_threshold=config.low_contrast_threshold,
        )
        flat = flatten_metric_groups(groups)
        flat_metric_rows.append(flat)
        slice_records.append(
            {
                "z_index": int(z_index),
                "metrics": groups,
                "metrics_flat": flat,
            }
        )

    return slice_records, aggregate_metric_dicts(flat_metric_rows) if flat_metric_rows else {}


def _finalize_qc_results(
    *,
    shape_zyx: tuple[int, int, int],
    dtype: str,
    z_indices: list[int],
    global_exposure: dict[str, float],
    slice_records: list[dict[str, Any]],
    slice_aggregate: dict[str, float],
    projection_metrics: dict[str, Any] | None,
    config: ImageQcConfig,
    progress: QcProgressTracker | None = None,
) -> dict[str, Any]:
    results = {
        "shape_zyx": list(shape_zyx),
        "dtype": dtype,
        "sampled_z_indices": z_indices,
        "global_exposure_dynamic_range": global_exposure,
        "slice_metrics": slice_records,
        "slice_aggregate": slice_aggregate,
        "projection_metrics": projection_metrics,
        "config": {
            "tile_size": config.tile_size,
            "n_slabs": config.n_slabs,
            "max_slices": config.max_slices,
            "projection": config.projection,
            "fft_enabled": config.fft_enabled,
            "histogram_bins": config.histogram_bins,
            "dark_pixel_threshold": config.dark_pixel_threshold,
            "low_contrast_threshold": config.low_contrast_threshold,
            "grading_enabled": config.grading_enabled,
            "thresholds_path": str(config.thresholds_path) if config.thresholds_path else None,
            "show_progress": config.show_progress,
            "nas_qc": config.nas_qc,
            "ims_resolution_level": config.ims_resolution_level,
            "ims_channel": config.ims_channel,
            "ims_timepoint": config.ims_timepoint,
            "ims_histogram_z_chunks": config.ims_histogram_z_chunks,
            "ims_z_chunk": config.ims_z_chunk,
            "ims_hdf_cache_mb": config.ims_hdf_cache_mb,
            "ims_z_chunk_align": config.ims_z_chunk_align,
        },
    }
    if config.grading_enabled:
        if progress is not None:
            with progress.step("grading", "Grading"):
                rules = load_threshold_rules(config.thresholds_path)
                results["grading"] = grade_qc_results(results, rules=rules)
        else:
            rules = load_threshold_rules(config.thresholds_path)
            results["grading"] = grade_qc_results(results, rules=rules)
    return results


def compute_image_qc(
    volume_zyx: np.ndarray,
    *,
    config: ImageQcConfig | None = None,
    progress: QcProgressTracker | None = None,
) -> dict[str, Any]:
    cfg = config or ImageQcConfig()
    tracker = progress if progress is not None else QcProgressTracker(enabled=cfg.show_progress)
    volume = np.asarray(volume_zyx)
    if volume.ndim == 2:
        volume = volume[np.newaxis, ...]
    if volume.ndim != 3:
        raise PipelineError(
            ErrorCode.INPUT_FORMAT_INVALID,
            "Expected a 2D or 3D image volume",
            {"shape": list(volume.shape)},
        )

    with tracker.step("global_histogram", "Global histogram"):
        global_exposure = compute_global_exposure_metrics(
            volume,
            histogram_bins=cfg.histogram_bins,
            saturation_margin=cfg.saturation_margin,
            dark_pixel_threshold=cfg.dark_pixel_threshold,
        )
    z_indices = _select_z_indices(volume.shape[0], cfg.max_slices)
    selected = volume[z_indices] if z_indices else volume[:0]
    with tracker.step("slice_metrics", "Slice metrics"):
        slice_records, slice_aggregate = _compute_slice_records(
            selected,
            z_indices=z_indices,
            global_exposure=global_exposure,
            config=cfg,
            progress=tracker,
        )

    projection_metrics: dict[str, Any] | None = None
    if cfg.projection in {"max", "mean"}:
        with tracker.step("projection", "Volume projection"):
            projection_2d = _projection_2d(volume, cfg.projection)
            projection_groups = compute_slice_metrics(
                projection_2d,
                tile_size=cfg.tile_size,
                n_slabs=cfg.n_slabs,
                fft_enabled=cfg.fft_enabled,
                global_percentiles=_global_percentiles_from_exposure(global_exposure),
                dark_pixel_threshold=cfg.dark_pixel_threshold,
                low_contrast_threshold=cfg.low_contrast_threshold,
            )
            projection_metrics = {
                "projection": cfg.projection,
                "metrics": projection_groups,
                "metrics_flat": flatten_metric_groups(projection_groups, prefix="projection."),
            }

    return _finalize_qc_results(
        shape_zyx=tuple(volume.shape),
        dtype=str(volume.dtype),
        z_indices=z_indices,
        global_exposure=global_exposure,
        slice_records=slice_records,
        slice_aggregate=slice_aggregate,
        projection_metrics=projection_metrics,
        config=cfg,
        progress=tracker,
    )


def _compute_image_qc_zarr(
    dataset: Any,
    *,
    config: ImageQcConfig,
    progress: QcProgressTracker | None = None,
) -> dict[str, Any]:
    tracker = progress if progress is not None else QcProgressTracker(enabled=config.show_progress)
    shape = tuple(int(v) for v in dataset.shape)
    chunks = tuple(int(v) for v in getattr(dataset, "chunks", shape))
    z_indices = _select_z_indices(shape[0], config.max_slices)

    with tracker.step("global_histogram", "Global histogram"):
        global_exposure = _compute_global_exposure_streaming(
            _iter_zarr_blocks(dataset),
            dtype=np.dtype(dataset.dtype),
            histogram_bins=config.histogram_bins,
            saturation_margin=config.saturation_margin,
            dark_pixel_threshold=config.dark_pixel_threshold,
            progress=tracker,
            progress_desc="Histogram chunks",
            progress_total=count_zarr_blocks(shape, chunks),
        )

    with tracker.step("load_sampled_slices", "Load sampled slices"):
        selected = (
            np.stack([np.asarray(dataset[z_index]) for z_index in z_indices], axis=0)
            if z_indices
            else np.empty((0, *shape[1:]))
        )

    with tracker.step("slice_metrics", "Slice metrics"):
        slice_records, slice_aggregate = _compute_slice_records(
            selected,
            z_indices=z_indices,
            global_exposure=global_exposure,
            config=config,
            progress=tracker,
        )

    projection_metrics: dict[str, Any] | None = None
    if config.projection in {"max", "mean"}:
        with tracker.step("projection", "Volume projection"):
            projection_2d: np.ndarray | None = None
            z_iter = tracker.iter(
                range(shape[0]),
                desc="Projection slices",
                total=shape[0],
                unit="slice",
            )
            for z_index in z_iter:
                sl = np.asarray(dataset[z_index], dtype=np.float64)
                if projection_2d is None:
                    projection_2d = sl.copy()
                elif config.projection == "max":
                    np.maximum(projection_2d, sl, out=projection_2d)
                else:
                    projection_2d += sl
            if projection_2d is not None and config.projection == "mean":
                projection_2d /= float(shape[0])
            if projection_2d is not None:
                projection_groups = compute_slice_metrics(
                    projection_2d,
                    tile_size=config.tile_size,
                    n_slabs=config.n_slabs,
                    fft_enabled=config.fft_enabled,
                    global_percentiles=_global_percentiles_from_exposure(global_exposure),
                    dark_pixel_threshold=config.dark_pixel_threshold,
                    low_contrast_threshold=config.low_contrast_threshold,
                )
                projection_metrics = {
                    "projection": config.projection,
                    "metrics": projection_groups,
                    "metrics_flat": flatten_metric_groups(projection_groups, prefix="projection."),
                }

    return _finalize_qc_results(
        shape_zyx=shape,
        dtype=str(dataset.dtype),
        z_indices=z_indices,
        global_exposure=global_exposure,
        slice_records=slice_records,
        slice_aggregate=slice_aggregate,
        projection_metrics=projection_metrics,
        config=config,
        progress=tracker,
    )


def _compute_image_qc_tiff_dir(
    path: Path,
    *,
    config: ImageQcConfig,
    progress: QcProgressTracker | None = None,
) -> dict[str, Any]:
    tracker = progress if progress is not None else QcProgressTracker(enabled=config.show_progress)
    files = list_tiff_files(path)
    if not files:
        raise PipelineError(
            ErrorCode.INPUT_FORMAT_INVALID,
            "No TIFF files found in input directory",
            {"input_tiff_dir": str(path)},
        )
    sample = load_tiff_as_zyx(files[0])
    dtype = sample.dtype
    shape = (len(files), sample.shape[0], sample.shape[1])
    z_indices = _select_z_indices(shape[0], config.max_slices)

    with tracker.step("global_histogram", "Global histogram"):
        global_exposure = _compute_global_exposure_streaming(
            _iter_tiff_dir_blocks(path),
            dtype=dtype,
            histogram_bins=config.histogram_bins,
            saturation_margin=config.saturation_margin,
            dark_pixel_threshold=config.dark_pixel_threshold,
            progress=tracker,
            progress_desc="Histogram slices",
            progress_total=len(files),
        )

    with tracker.step("load_sampled_slices", "Load sampled slices"):
        selected = (
            np.stack([load_tiff_as_zyx(files[z_index])[0] for z_index in z_indices], axis=0)
            if z_indices
            else np.empty((0, shape[1], shape[2]), dtype=dtype)
        )

    with tracker.step("slice_metrics", "Slice metrics"):
        slice_records, slice_aggregate = _compute_slice_records(
            selected,
            z_indices=z_indices,
            global_exposure=global_exposure,
            config=config,
            progress=tracker,
        )

    projection_metrics: dict[str, Any] | None = None
    if config.projection in {"max", "mean"}:
        with tracker.step("projection", "Volume projection"):
            projection_2d: np.ndarray | None = None
            for file_path in tracker.iter(files, desc="Projection slices", total=len(files), unit="slice"):
                sl = load_tiff_as_zyx(file_path)[0].astype(np.float64, copy=False)
                if projection_2d is None:
                    projection_2d = sl.copy()
                elif config.projection == "max":
                    np.maximum(projection_2d, sl, out=projection_2d)
                else:
                    projection_2d += sl
            if projection_2d is not None and config.projection == "mean":
                projection_2d /= float(shape[0])
            if projection_2d is not None:
                projection_groups = compute_slice_metrics(
                    projection_2d,
                    tile_size=config.tile_size,
                    n_slabs=config.n_slabs,
                    fft_enabled=config.fft_enabled,
                    global_percentiles=_global_percentiles_from_exposure(global_exposure),
                    dark_pixel_threshold=config.dark_pixel_threshold,
                    low_contrast_threshold=config.low_contrast_threshold,
                )
                projection_metrics = {
                    "projection": config.projection,
                    "metrics": projection_groups,
                    "metrics_flat": flatten_metric_groups(projection_groups, prefix="projection."),
                }

    return _finalize_qc_results(
        shape_zyx=shape,
        dtype=str(dtype),
        z_indices=z_indices,
        global_exposure=global_exposure,
        slice_records=slice_records,
        slice_aggregate=slice_aggregate,
        projection_metrics=projection_metrics,
        config=config,
        progress=tracker,
    )


def _compute_image_qc_ims(
    ims_path: Path,
    *,
    config: ImageQcConfig,
    progress: QcProgressTracker | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    tracker = progress if progress is not None else QcProgressTracker(enabled=config.show_progress)

    with open_ims_dataset(
        ims_path,
        resolution_level=config.ims_resolution_level,
        channel=config.ims_channel,
        timepoint=config.ims_timepoint,
        hdf_cache_mb=config.ims_hdf_cache_mb,
    ) as (dataset, info):
        z_chunk = effective_z_chunk(info, config.ims_z_chunk if config.ims_z_chunk > 0 else None)
        shape = info.shape_zyx
        z_indices = select_qc_z_indices(
            shape[0],
            config.max_slices,
            z_chunk,
            align_to_chunk=config.ims_z_chunk_align,
        )
        histogram_chunk_ids = select_histogram_z_chunk_ids(
            shape[0],
            z_chunk,
            config.ims_histogram_z_chunks,
        )

        with tracker.step("global_histogram", "Global histogram"):
            global_exposure = _compute_global_exposure_streaming(
                iter_ims_histogram_blocks(dataset, z_chunk=z_chunk, z_chunk_ids=histogram_chunk_ids),
                dtype=np.dtype(info.dtype),
                histogram_bins=config.histogram_bins,
                saturation_margin=config.saturation_margin,
                dark_pixel_threshold=config.dark_pixel_threshold,
                progress=tracker,
                progress_desc="IMS histogram z-chunks",
                progress_total=len(histogram_chunk_ids),
            )

        with tracker.step("load_sampled_slices", "Load sampled slices"):
            read_ranges = group_z_indices_into_read_ranges(z_indices, shape_z=shape[0], z_chunk=z_chunk)
            slice_map: dict[int, np.ndarray] = {}
            range_iter = tracker.iter(
                read_ranges,
                desc="IMS chunk reads",
                total=len(read_ranges),
                unit="chunk",
            )
            for z0, z1 in range_iter:
                block = np.asarray(dataset[z0:z1, :, :])
                for z_index in z_indices:
                    z_value = int(z_index)
                    if z0 <= z_value < z1:
                        slice_map[z_value] = block[z_value - z0]
            selected = stack_ordered_slices(slice_map, z_indices)

        with tracker.step("slice_metrics", "Slice metrics"):
            slice_records, slice_aggregate = _compute_slice_records(
                selected,
                z_indices=z_indices,
                global_exposure=global_exposure,
                config=config,
                progress=tracker,
            )

        source_meta = ims_source_meta(
            info,
            z_chunk=z_chunk,
            read_strategy="chunk_aligned_z_batch",
        )
        source_meta["histogram_z_chunk_ids"] = histogram_chunk_ids
        source_meta["histogram_z_chunks_requested"] = int(config.ims_histogram_z_chunks)
        source_meta["histogram_z_chunks_read"] = len(histogram_chunk_ids)
        source_meta["slice_read_ranges_zy"] = read_ranges
        source_meta["sampled_z_indices"] = z_indices

        results = _finalize_qc_results(
            shape_zyx=shape,
            dtype=info.dtype,
            z_indices=z_indices,
            global_exposure=global_exposure,
            slice_records=slice_records,
            slice_aggregate=slice_aggregate,
            projection_metrics=None,
            config=config,
            progress=tracker,
        )
        if config.projection != "none":
            results.setdefault("warnings", []).append(
                "IMS NAS QC ignores --projection; use TIFF/Zarr locally for projection metrics.",
            )
        return results, source_meta


def _write_slice_metrics_csv(path: Path, slice_records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["z_index"]
    if slice_records:
        fieldnames.extend(sorted(slice_records[0]["metrics_flat"].keys()))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in slice_records:
            row = {"z_index": record["z_index"]}
            row.update(record["metrics_flat"])
            writer.writerow(row)


def run_image_qc(
    *,
    input_zarr: str | Path | None = None,
    input_tiff_dir: str | Path | None = None,
    input_tiff: str | Path | None = None,
    input_ims: str | Path | None = None,
    output_json: str | Path,
    output_csv: str | Path | None = None,
    sample_id: str | None = None,
    config: ImageQcConfig | None = None,
) -> dict[str, Any]:
    started_at = time.time()
    source_kind, source_path = _resolve_input_source(
        input_zarr=input_zarr,
        input_tiff_dir=input_tiff_dir,
        input_tiff=input_tiff,
        input_ims=input_ims,
    )
    cfg = _apply_nas_qc_defaults(config or ImageQcConfig())
    if source_kind == "ims" and not cfg.nas_qc and cfg.max_slices == DEFAULT_MAX_SLICES:
        cfg.max_slices = DEFAULT_IMS_MAX_SLICES
    progress = QcProgressTracker(enabled=cfg.show_progress)

    with progress.step("open_source", "Open source"):
        if source_kind == "ims":
            results, source_meta = _compute_image_qc_ims(source_path, config=cfg, progress=progress)
        elif source_kind == "zarr":
            dataset = open_zarr_dataset(source_path, dataset_name=cfg.dataset_name)
            source_meta = {
                "source_kind": "zarr",
                "source_path": str(source_path),
                "shape_zyx": [int(v) for v in dataset.shape],
                "dtype": str(dataset.dtype),
                "chunks_zyx": [int(v) for v in getattr(dataset, "chunks", dataset.shape)],
                "streaming": True,
            }
            results = _compute_image_qc_zarr(dataset, config=cfg, progress=progress)
        elif source_kind == "tiff_dir":
            source_meta = {
                "source_kind": "tiff_dir",
                "source_path": str(source_path),
                "streaming": True,
            }
            results = _compute_image_qc_tiff_dir(source_path, config=cfg, progress=progress)
            source_meta["shape_zyx"] = results["shape_zyx"]
            source_meta["dtype"] = results["dtype"]
        else:
            volume = load_tiff_as_zyx(source_path)
            source_meta = {
                "source_kind": "tiff",
                "source_path": str(source_path),
                "shape_zyx": list(volume.shape),
                "dtype": str(volume.dtype),
                "streaming": False,
            }
            results = compute_image_qc(volume, config=cfg, progress=progress)

    results["sample_id"] = sample_id or source_path.stem
    results["source"] = source_meta

    with progress.step("write_outputs", "Write outputs"):
        output_path = Path(output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        results["runtime_seconds"] = round(time.time() - started_at, 3)
        results["timing_breakdown"] = progress.to_dict()
        output_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

        csv_path: Path | None = None
        if output_csv:
            csv_path = Path(output_csv)
            _write_slice_metrics_csv(csv_path, results["slice_metrics"])

        write_run_manifest(
            output_path.parent,
            module="qc",
            entrypoint="image_qc",
            inputs={
                "source_kind": source_kind,
                "source_path": str(source_path),
                "sample_id": results["sample_id"],
                "config": cfg.__dict__,
            },
            outputs=[output_path] + ([csv_path] if csv_path is not None else []),
            started_at=started_at,
        )

    progress.print_summary(total_seconds=results["runtime_seconds"])
    return results


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute no-reference image quality metrics for LSFM TIFF/Zarr volumes.",
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input_zarr", help="Input signal Zarr path")
    source.add_argument("--input_tiff_dir", help="Input TIFF slice directory")
    source.add_argument("--input_tiff", help="Single multi-page or 2D TIFF path")
    source.add_argument("--input_ims", help="Imaris IMS file on NAS/local path")
    parser.add_argument("--output_json", required=True, help="Output JSON metrics path")
    parser.add_argument("--output_csv", default=None, help="Optional per-slice CSV output path")
    parser.add_argument("--sample_id", default=None, help="Sample identifier for the report")
    parser.add_argument("--dataset_name", default="0", help="Zarr dataset name")
    parser.add_argument("--tile_size", type=int, default=DEFAULT_TILE_SIZE)
    parser.add_argument("--n_slabs", type=int, default=DEFAULT_N_SLABS)
    parser.add_argument("--max_slices", type=int, default=DEFAULT_MAX_SLICES)
    parser.add_argument(
        "--projection",
        choices=("none", "max", "mean"),
        default="none",
        help="Optional whole-volume projection metrics",
    )
    parser.add_argument("--fft_enabled", action="store_true", help="Enable FFT stripe periodicity metrics")
    parser.add_argument("--histogram_bins", type=int, default=DEFAULT_HISTOGRAM_BINS)
    parser.add_argument("--dark_pixel_threshold", type=float, default=DEFAULT_DARK_PIXEL_THRESHOLD)
    parser.add_argument("--low_contrast_threshold", type=float, default=DEFAULT_LOW_CONTRAST_THRESHOLD)
    parser.add_argument(
        "--thresholds",
        default=str(DEFAULT_THRESHOLDS_PATH),
        help="JSON file with pass/warn/fail thresholds",
    )
    parser.add_argument("--no_grading", action="store_true", help="Disable pass/warn/fail grading")
    parser.add_argument("--quiet", action="store_true", help="Disable progress bars and timing summary")
    parser.add_argument(
        "--nas_qc",
        action="store_true",
        help="NAS/IMS optimized defaults: ResolutionLevel 2, 8 aligned slices, 24 histogram z-chunks, no projection",
    )
    parser.add_argument("--ims_resolution_level", type=int, default=DEFAULT_IMS_RESOLUTION_LEVEL)
    parser.add_argument("--ims_channel", type=int, default=DEFAULT_IMS_CHANNEL)
    parser.add_argument("--ims_timepoint", type=int, default=DEFAULT_IMS_TIMEPOINT)
    parser.add_argument(
        "--ims_histogram_z_chunks",
        type=int,
        default=DEFAULT_IMS_HISTOGRAM_Z_CHUNKS,
        help="Number of Z chunks to read for global histogram (IMS/NAS mode)",
    )
    parser.add_argument(
        "--ims_z_chunk",
        type=int,
        default=0,
        help="IMS read alignment in Z; 0 means auto-detect from IMS chunks",
    )
    parser.add_argument("--ims_hdf_cache_mb", type=int, default=DEFAULT_IMS_HDF_CACHE_MB)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = ImageQcConfig(
        tile_size=args.tile_size,
        n_slabs=args.n_slabs,
        max_slices=args.max_slices,
        projection=args.projection,
        fft_enabled=bool(args.fft_enabled),
        histogram_bins=args.histogram_bins,
        dark_pixel_threshold=args.dark_pixel_threshold,
        low_contrast_threshold=args.low_contrast_threshold,
        dataset_name=args.dataset_name,
        thresholds_path=args.thresholds,
        grading_enabled=not bool(args.no_grading),
        show_progress=not bool(args.quiet),
        nas_qc=bool(args.nas_qc),
        ims_resolution_level=args.ims_resolution_level,
        ims_channel=args.ims_channel,
        ims_timepoint=args.ims_timepoint,
        ims_histogram_z_chunks=args.ims_histogram_z_chunks,
        ims_z_chunk=args.ims_z_chunk,
        ims_hdf_cache_mb=args.ims_hdf_cache_mb,
    )
    try:
        run_image_qc(
            input_zarr=args.input_zarr,
            input_tiff_dir=args.input_tiff_dir,
            input_tiff=args.input_tiff,
            input_ims=args.input_ims,
            output_json=args.output_json,
            output_csv=args.output_csv,
            sample_id=args.sample_id,
            config=config,
        )
    except PipelineError as exc:
        print(json.dumps(exc.to_dict(), ensure_ascii=False), file=sys.stderr)
        return exc.exit_code
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
