from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

try:
    from pipeline_modules.utils.errors import ErrorCode, PipelineError
    from pipeline_modules.segmentation.zarr_utils import open_zarr_dataset
except ImportError:  # pragma: no cover
    from ..utils.errors import ErrorCode, PipelineError
    from .zarr_utils import open_zarr_dataset


DEFAULT_WEIGHTS = {
    "uncertainty": 0.70,
    "small_components": 0.30,
}


def _iter_chunk_slices(shape: tuple[int, int, int], chunks: tuple[int, int, int]):
    for z in range(0, shape[0], chunks[0]):
        z_end = min(z + chunks[0], shape[0])
        for y in range(0, shape[1], chunks[1]):
            y_end = min(y + chunks[1], shape[1])
            for x in range(0, shape[2], chunks[2]):
                x_end = min(x + chunks[2], shape[2])
                yield (slice(z, z_end), slice(y, y_end), slice(x, x_end))



def _iter_blocks(shape: tuple[int, int, int], chunks: tuple[int, int, int]):
    block_number = 1
    z_index = 0
    for z in range(0, shape[0], chunks[0]):
        z_end = min(z + chunks[0], shape[0])
        y_index = 0
        for y in range(0, shape[1], chunks[1]):
            y_end = min(y + chunks[1], shape[1])
            x_index = 0
            for x in range(0, shape[2], chunks[2]):
                x_end = min(x + chunks[2], shape[2])
                slices = (slice(z, z_end), slice(y, y_end), slice(x, x_end))
                start = (z, y, x)
                stop = (z_end, y_end, x_end)
                yield {
                    "block_id": f"block_{block_number:06d}",
                    "chunk_index": f"{z_index}.{y_index}.{x_index}",
                    "block_start_zyx": _format_triplet(start),
                    "block_stop_zyx": _format_triplet(stop),
                    "block_shape_zyx": _format_triplet((z_end - z, y_end - y, x_end - x)),
                    "_slices": slices,
                    "_start": start,
                    "_stop": stop,
                }
                block_number += 1
                x_index += 1
            y_index += 1
        z_index += 1


def _parse_triplet(value: str) -> tuple[int, int, int] | None:
    if not str(value).strip():
        return None
    parts = [part.strip() for part in str(value).split(",") if part.strip()]
    if len(parts) != 3:
        raise PipelineError(ErrorCode.ARGUMENT_INVALID, "Expected three comma-separated integers", {"value": value})
    return (int(parts[0]), int(parts[1]), int(parts[2]))


def _coerce_triplet(value: Any, *, field_name: str) -> tuple[int, int, int]:
    if isinstance(value, (tuple, list)) and len(value) == 3:
        return (int(value[0]), int(value[1]), int(value[2]))
    parsed = _parse_triplet(str(value))
    if parsed is None:
        raise PipelineError(
            ErrorCode.ARGUMENT_INVALID,
            f"Expected three integers for {field_name}",
            {field_name: value},
        )
    return parsed


def _format_triplet(value: tuple[int, int, int]) -> str:
    return ",".join(str(int(v)) for v in value)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(parsed) or math.isinf(parsed):
        return default
    return parsed


def _component_stats(mask_bool, *, small_component_max_voxels: int) -> tuple[int, int]:
    if not mask_bool.any():
        return 0, 0
    try:
        import numpy as np
        from scipy import ndimage
    except ModuleNotFoundError:
        return 1, 0

    labeled, num_components = ndimage.label(mask_bool)
    if num_components == 0:
        return 0, 0
    sizes = np.bincount(labeled.ravel())[1:]
    small_components = int((sizes <= int(small_component_max_voxels)).sum())
    return int(num_components), small_components


def _resolve_optional_path(value: Any) -> Path | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return Path(text)


def _load_qc_arrays(
    *,
    mask_zarr: str | Path,
    image_zarr: str | Path | None = None,
    probability_zarr: str | Path | None = None,
    threshold_zarr: str | Path | None = None,
    dataset_name: str = "0",
):
    mask_path = Path(mask_zarr)
    if not mask_path.exists():
        raise PipelineError(ErrorCode.INPUT_NOT_FOUND, "Mask Zarr not found", {"mask_zarr": str(mask_path)})

    image_path = _resolve_optional_path(image_zarr)
    probability_path = _resolve_optional_path(probability_zarr)
    threshold_path = _resolve_optional_path(threshold_zarr)

    if image_path is not None and not image_path.exists():
        raise PipelineError(ErrorCode.INPUT_NOT_FOUND, "Image Zarr not found", {"image_zarr": str(image_path)})
    if probability_path is not None and not probability_path.exists():
        raise PipelineError(
            ErrorCode.INPUT_NOT_FOUND,
            "Probability Zarr not found",
            {"probability_zarr": str(probability_path)},
        )
    if threshold_path is not None and not threshold_path.exists():
        raise PipelineError(
            ErrorCode.INPUT_NOT_FOUND,
            "Threshold Zarr not found",
            {"threshold_zarr": str(threshold_path)},
        )

    mask_data = open_zarr_dataset(mask_path, dataset_name=dataset_name)
    image_data = open_zarr_dataset(image_path, dataset_name=dataset_name) if image_path is not None else None
    probability_data = (
        open_zarr_dataset(probability_path, dataset_name=dataset_name) if probability_path is not None else None
    )
    threshold_data = open_zarr_dataset(threshold_path, dataset_name=dataset_name) if threshold_path is not None else None

    shape = tuple(int(v) for v in mask_data.shape)
    if image_data is not None and tuple(int(v) for v in image_data.shape) != shape:
        raise PipelineError(
            ErrorCode.ARGUMENT_INVALID,
            "Image Zarr shape does not match mask Zarr",
            {"mask_shape": shape, "image_shape": tuple(image_data.shape)},
        )
    if probability_data is not None and tuple(int(v) for v in probability_data.shape) != shape:
        raise PipelineError(
            ErrorCode.ARGUMENT_INVALID,
            "Probability Zarr shape does not match mask Zarr",
            {"mask_shape": shape, "probability_shape": tuple(probability_data.shape)},
        )
    if threshold_data is not None and tuple(int(v) for v in threshold_data.shape) != shape:
        raise PipelineError(
            ErrorCode.ARGUMENT_INVALID,
            "Threshold Zarr shape does not match mask Zarr",
            {"mask_shape": shape, "threshold_shape": tuple(threshold_data.shape)},
        )

    return {
        "mask_path": mask_path,
        "image_path": image_path,
        "probability_path": probability_path,
        "threshold_path": threshold_path,
        "mask_data": mask_data,
        "image_data": image_data,
        "probability_data": probability_data,
        "threshold_data": threshold_data,
        "shape": shape,
    }


def compute_block_qc_metrics(
    *,
    sample_id: str,
    mask_zarr: str | Path,
    image_zarr: str | Path | None = None,
    probability_zarr: str | Path | None = None,
    threshold_zarr: str | Path | None = None,
    dataset_name: str = "0",
    chunk_size: tuple[int, int, int] | None = None,
    uncertainty_low: float = 0.4,
    uncertainty_high: float = 0.6,
    small_component_max_voxels: int = 32,
    skip_empty_blocks: bool = True,
    workers: int = 32,
) -> list[dict[str, Any]]:
    import numpy as np

    arrays = _load_qc_arrays(
        mask_zarr=mask_zarr,
        image_zarr=image_zarr,
        probability_zarr=probability_zarr,
        threshold_zarr=threshold_zarr,
        dataset_name=dataset_name,
    )
    mask_path = arrays["mask_path"]
    image_path = arrays["image_path"]
    probability_path = arrays["probability_path"]
    threshold_path = arrays["threshold_path"]
    image_data = arrays["image_data"]
    mask_data = arrays["mask_data"]
    probability_data = arrays["probability_data"]
    threshold_data = arrays["threshold_data"]
    shape = arrays["shape"]
    chunks = chunk_size or tuple(int(v) for v in getattr(mask_data, "chunks", shape))

    import math
    from concurrent.futures import ThreadPoolExecutor
    from tqdm import tqdm

    total_blocks = math.prod(math.ceil(s / c) for s, c in zip(shape, chunks))
    all_blocks = list(_iter_blocks(shape, chunks))

    def _scan_one_block(block, _skip_empty=skip_empty_blocks):
        slices = block["_slices"]
        mask_chunk = np.asarray(mask_data[slices]) > 0
        total_voxels = max(int(mask_chunk.size), 1)
        foreground_voxels = int(mask_chunk.sum())

        if _skip_empty:
            if image_data is None:
                print("ERROR: --no_skip_empty_blocks not set but image_zarr (ch1_preprocessed.zarr) is missing", file=sys.stderr)
                sys.exit(1)
            image_chunk = np.asarray(image_data[slices])
            if float(image_chunk.max()) <= 240.0:
                return None

        uncertain_voxels = 0
        if probability_data is not None:
            probability_chunk = np.asarray(probability_data[slices])
            uncertain = (probability_chunk >= float(uncertainty_low)) & (
                probability_chunk <= float(uncertainty_high)
            )
            uncertain_voxels = int(uncertain.sum())

        xor_voxels = 0
        union_voxels = 0
        if threshold_data is not None:
            threshold_chunk = np.asarray(threshold_data[slices]) > 0
            xor_voxels = int(np.logical_xor(mask_chunk, threshold_chunk).sum())
            union_voxels = int(np.logical_or(mask_chunk, threshold_chunk).sum())

        num_components, small_components = _component_stats(
            mask_chunk,
            small_component_max_voxels=int(small_component_max_voxels),
        )
        threshold_disagreement = (xor_voxels / union_voxels) if union_voxels else 0.0

        return {
            "sample_id": sample_id,
            "block_id": f"{sample_id}_{str(block['chunk_index']).replace('.', '-')}",
            "chunk_index": block["chunk_index"],
            "block_start_zyx": block["block_start_zyx"],
            "block_stop_zyx": block["block_stop_zyx"],
            "block_shape_zyx": block["block_shape_zyx"],
            "image_zarr": str(image_path) if image_path is not None else "",
            "mask_zarr": str(mask_path),
            "probability_zarr": str(probability_path) if probability_path is not None else "",
            "threshold_zarr": str(threshold_path) if threshold_path is not None else "",
            "total_voxels": total_voxels,
            "foreground_voxels": foreground_voxels,
            "fg_ratio": foreground_voxels / total_voxels,
            "uncertain_voxels": uncertain_voxels,
            "uncertain_ratio": uncertain_voxels / total_voxels,
            "num_components": num_components,
            "components_per_million_voxels": num_components / (total_voxels / 1_000_000.0),
            "small_components": small_components,
            "small_component_ratio": small_components / max(num_components, 1),
            "threshold_disagreement": threshold_disagreement,
            "preview_image_tiff": "",
            "preview_mask_tiff": "",
            "_shape": shape,
            "_slices": slices,
            "_xor_voxels": xor_voxels,
            "_union_voxels": union_voxels,
        }

    from concurrent.futures import as_completed

    metrics = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_scan_one_block, b): b for b in all_blocks}
        for future in tqdm(
            as_completed(futures),
            total=total_blocks,
            desc=f"Scanning blocks ({sample_id})",
            unit="block",
        ):
            result = future.result()
            if result is not None:
                metrics.append(result)

    return metrics


def compute_sample_qc_metrics(
    *,
    sample_id: str,
    mask_zarr: str | Path,
    image_zarr: str | Path | None = None,
    probability_zarr: str | Path | None = None,
    threshold_zarr: str | Path | None = None,
    dataset_name: str = "0",
    chunk_size: tuple[int, int, int] | None = None,
    uncertainty_low: float = 0.4,
    uncertainty_high: float = 0.6,
    small_component_max_voxels: int = 32,
) -> dict[str, Any]:
    block_metrics = compute_block_qc_metrics(
        sample_id=sample_id,
        mask_zarr=mask_zarr,
        image_zarr=image_zarr,
        probability_zarr=probability_zarr,
        threshold_zarr=threshold_zarr,
        dataset_name=dataset_name,
        chunk_size=chunk_size,
        uncertainty_low=uncertainty_low,
        uncertainty_high=uncertainty_high,
        small_component_max_voxels=small_component_max_voxels,
    )
    arrays = _load_qc_arrays(
        mask_zarr=mask_zarr,
        image_zarr=image_zarr,
        probability_zarr=probability_zarr,
        threshold_zarr=threshold_zarr,
        dataset_name=dataset_name,
    )
    shape = arrays["shape"]

    total_voxels = max(sum(int(record["total_voxels"]) for record in block_metrics), 1)
    foreground_voxels = sum(int(record["foreground_voxels"]) for record in block_metrics)
    uncertain_voxels = sum(int(record["uncertain_voxels"]) for record in block_metrics)
    num_components = sum(int(record["num_components"]) for record in block_metrics)
    small_components = sum(int(record["small_components"]) for record in block_metrics)
    xor_voxels = sum(int(record["_xor_voxels"]) for record in block_metrics)
    union_voxels = sum(int(record["_union_voxels"]) for record in block_metrics)
    threshold_disagreement = (xor_voxels / union_voxels) if union_voxels else 0.0

    return {
        "sample_id": sample_id,
        "image_zarr": str(arrays["image_path"]) if arrays["image_path"] is not None else "",
        "mask_zarr": str(arrays["mask_path"]),
        "probability_zarr": str(arrays["probability_path"]) if arrays["probability_path"] is not None else "",
        "threshold_zarr": str(arrays["threshold_path"]) if arrays["threshold_path"] is not None else "",
        "shape": "x".join(str(v) for v in shape),
        "total_voxels": total_voxels,
        "foreground_voxels": foreground_voxels,
        "fg_ratio": foreground_voxels / total_voxels,
        "uncertain_voxels": uncertain_voxels,
        "uncertain_ratio": uncertain_voxels / total_voxels,
        "num_components": num_components,
        "components_per_million_voxels": num_components / (total_voxels / 1_000_000.0),
        "small_components": small_components,
        "small_component_ratio": small_components / max(num_components, 1),
        "threshold_disagreement": threshold_disagreement,
    }


def _robust_outlier_scores(records: list[dict[str, Any]], key: str) -> list[float]:
    import numpy as np

    values = np.asarray([_safe_float(record.get(key)) for record in records], dtype=np.float64)
    if values.size == 0:
        return []
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    scale = max(1.4826 * mad, 1e-12)
    return [min(abs(float(value) - median) / (3.0 * scale), 1.0) for value in values]


def _relative_high_scores(records: list[dict[str, Any]], key: str) -> list[float]:
    values = [_safe_float(record.get(key)) for record in records]
    high = max(values) if values else 0.0
    if high <= 0:
        return [0.0 for _ in values]
    return [min(max(value / high, 0.0), 1.0) for value in values]


def score_qc_records(
    records: list[dict[str, Any]],
    *,
    weights: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    weights = weights or DEFAULT_WEIGHTS
    uncertainty_scores = _relative_high_scores(records, "uncertain_ratio")

    scored = []
    for index, record in enumerate(records):
        uncertainty_score = uncertainty_scores[index] if index < len(uncertainty_scores) else 0.0
        foreground_outlier_score = 0.0
        small_component_score = min(max(_safe_float(record.get("small_component_ratio")), 0.0), 1.0)
        threshold_score = 0.0
        review_score = (
            weights["uncertainty"] * uncertainty_score
            + weights["small_components"] * small_component_score
        )
        scored_record = dict(record)
        scored_record.update(
            {
                "uncertainty_score": uncertainty_score,
                "foreground_outlier_score": foreground_outlier_score,
                "small_component_score": small_component_score,
                "threshold_disagreement_score": threshold_score,
                "review_score": review_score,
            }
        )
        scored.append(scored_record)

    return sorted(scored, key=lambda row: _safe_float(row.get("review_score")), reverse=True)


def _read_records_csv(path: Path) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as fh:
        records = list(csv.DictReader(fh))
    base_dir = path.parent
    for record in records:
        for key in ("image_zarr", "mask_zarr", "probability_zarr", "threshold_zarr"):
            value = str(record.get(key, "")).strip()
            if value and not Path(value).is_absolute():
                record[key] = str(base_dir / value)
    return records


def _records_from_sample_dirs(
    sample_dirs: list[Path],
    *,
    signal_ch: str,
    mask_suffix: str,
    probability_suffix: str,
    threshold_suffix: str,
) -> list[dict[str, str]]:
    records = []
    for sample_dir in sample_dirs:
        records.append(
            {
                "sample_id": sample_dir.name,
                "image_zarr": str(sample_dir / f"ch{signal_ch}.zarr"),
                "mask_zarr": str(sample_dir / f"ch{signal_ch}{mask_suffix}"),
                "probability_zarr": str(sample_dir / f"ch{signal_ch}{probability_suffix}"),
                "threshold_zarr": str(sample_dir / f"ch{signal_ch}{threshold_suffix}") if threshold_suffix else "",
            }
        )
    return records


def _write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "rank",
        "sample_id",
        "block_id",
        "chunk_index",
        "block_start_zyx",
        "block_stop_zyx",
        "block_shape_zyx",
        "review_score",
        "uncertain_ratio",
        "fg_ratio",
        "num_components",
        "components_per_million_voxels",
        "small_component_ratio",
        "threshold_disagreement",
        "image_zarr",
        "mask_zarr",
        "probability_zarr",
        "threshold_zarr",
        "preview_image_tiff",
        "preview_mask_tiff",
        "total_voxels",
        "foreground_voxels",
        "uncertain_voxels",
        "small_components",
        "uncertainty_score",
        "foreground_outlier_score",
        "small_component_score",
        "threshold_disagreement_score",
    ]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for rank, record in enumerate(records, start=1):
            row = dict(record)
            row["rank"] = rank
            writer.writerow(row)


def _large_grid_key(
    record: dict[str, Any],
    *,
    large_grid_size: tuple[int, int, int] = (1024, 1024, 1024),
    chunk_size: tuple[int, int, int] = (256, 256, 256),
) -> str:
    parts = str(record.get("chunk_index") or "").split(".")
    if len(parts) >= 3:
        cz, cy, cx = int(parts[0]), int(parts[1]), int(parts[2])
        lgz, lgy, lgx = large_grid_size
        cz_size = chunk_size[0]
        cy_size = chunk_size[1]
        cx_size = chunk_size[2]
        grid_z = cz // (lgz // cz_size)
        grid_y = cy // (lgy // cy_size)
        grid_x = cx // (lgx // cx_size)
        return f"{grid_z}-{grid_y}-{grid_x}"
    return str(record.get("sample_id") or "")


def select_top_records(
    records: list[dict[str, Any]],
    *,
    top_n: int,
    max_per_large_grid: int,
    large_grid_size: tuple[int, int, int] = (1024, 1024, 1024),
    chunk_size: tuple[int, int, int] = (256, 256, 256),
) -> list[dict[str, Any]]:
    selected = []
    grid_counts: dict[str, int] = {}
    for record in records:
        if len(selected) >= int(top_n):
            break
        grid_key = _large_grid_key(record, large_grid_size=large_grid_size, chunk_size=chunk_size)
        if grid_counts.get(grid_key, 0) >= int(max_per_large_grid):
            continue
        selected.append(record)
        grid_counts[grid_key] = grid_counts.get(grid_key, 0) + 1
    return selected


def build_review_queue(
    records: list[dict[str, Any]],
    *,
    dataset_name: str = "0",
    chunk_size: tuple[int, int, int] | None = None,
    uncertainty_low: float = 0.4,
    uncertainty_high: float = 0.6,
    small_component_max_voxels: int = 32,
    skip_missing: bool = False,
    skip_empty_blocks: bool = True,
    workers: int = 32,
) -> list[dict[str, Any]]:
    metrics = []
    for record in records:
        mask_path = Path(str(record.get("mask_zarr", "")))
        image_path = _resolve_optional_path(record.get("image_zarr"))
        probability_path = _resolve_optional_path(record.get("probability_zarr"))
        threshold_path = _resolve_optional_path(record.get("threshold_zarr"))
        missing_image = image_path is not None and not image_path.exists()
        missing_probability = probability_path is not None and not probability_path.exists()
        missing_threshold = threshold_path is not None and not threshold_path.exists()
        if skip_missing and (not mask_path.exists() or missing_image or missing_probability or missing_threshold):
            continue
        metrics.extend(
            compute_block_qc_metrics(
                sample_id=str(record.get("sample_id") or mask_path.parent.name or mask_path.stem),
                image_zarr=image_path,
                mask_zarr=mask_path,
                probability_zarr=probability_path,
                threshold_zarr=threshold_path,
                dataset_name=dataset_name,
                chunk_size=chunk_size,
                uncertainty_low=uncertainty_low,
                uncertainty_high=uncertainty_high,
                small_component_max_voxels=small_component_max_voxels,
                skip_empty_blocks=skip_empty_blocks,
                workers=workers,
            )
        )
    return score_qc_records(metrics)


def export_block_previews(
    records: list[dict[str, Any]],
    preview_dir: str | Path,
    *,
    dataset_name: str = "0",
    skip_missing: bool = False,
) -> Path:
    import numpy as np
    import tifffile

    preview_path = Path(preview_dir)
    preview_path.mkdir(parents=True, exist_ok=True)
    image_cache: dict[str, Any] = {}
    mask_cache: dict[str, Any] = {}

    def _open_cached(cache: dict[str, Any], path: Path):
        key = str(path)
        if key not in cache:
            cache[key] = open_zarr_dataset(path, dataset_name=dataset_name)
        return cache[key]

    from concurrent.futures import ThreadPoolExecutor
    from tqdm import tqdm

    def _export_one(record):
        record["preview_image_tiff"] = ""
        record["preview_mask_tiff"] = ""

        image_path = _resolve_optional_path(record.get("image_zarr"))
        mask_path = _resolve_optional_path(record.get("mask_zarr"))
        if image_path is None or not image_path.exists():
            if skip_missing:
                return record
            raise PipelineError(
                ErrorCode.INPUT_NOT_FOUND,
                "Image Zarr not found for QC preview export",
                {"image_zarr": str(image_path) if image_path is not None else ""},
            )
        if mask_path is None or not mask_path.exists():
            if skip_missing:
                return record
            raise PipelineError(
                ErrorCode.INPUT_NOT_FOUND,
                "Mask Zarr not found for QC preview export",
                {"mask_zarr": str(mask_path) if mask_path is not None else ""},
            )

        image_data = _open_cached(image_cache, image_path)
        mask_data = _open_cached(mask_cache, mask_path)
        if tuple(int(v) for v in image_data.shape) != tuple(int(v) for v in mask_data.shape):
            raise PipelineError(
                ErrorCode.ARGUMENT_INVALID,
                "Image Zarr shape does not match mask Zarr for QC preview export",
                {"image_shape": tuple(image_data.shape), "mask_shape": tuple(mask_data.shape)},
            )

        slices = record.get("_slices")
        if slices is None:
            start = _coerce_triplet(record.get("block_start_zyx"), field_name="block_start_zyx")
            stop = _coerce_triplet(record.get("block_stop_zyx"), field_name="block_stop_zyx")
            slices = tuple(slice(start[idx], stop[idx]) for idx in range(3))

        image_chunk = np.asarray(image_data[slices])
        mask_chunk = np.asarray(mask_data[slices])
        sample_preview_dir = preview_path
        sample_preview_dir.mkdir(parents=True, exist_ok=True)
        block_id = str(record.get("block_id") or "block")
        image_preview_path = sample_preview_dir / 'image' / f"{block_id}_image.tiff"
        mask_preview_path = sample_preview_dir / 'mask' / f"{block_id}_mask.tiff"
        image_preview_path.parent.mkdir(parents=True, exist_ok=True)
        mask_preview_path.parent.mkdir(parents=True, exist_ok=True)
        tifffile.imwrite(str(image_preview_path), image_chunk)
        tifffile.imwrite(str(mask_preview_path), mask_chunk)
        record["preview_image_tiff"] = str(image_preview_path)
        record["preview_mask_tiff"] = str(mask_preview_path)
        return record

    from concurrent.futures import as_completed

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(_export_one, r): r for r in records}
        for future in tqdm(as_completed(futures), total=len(records), desc="Exporting previews", unit="block"):
            future.result()

    return preview_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rank worst cFos U-Net blocks for manual review")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--records_csv",
        help="CSV with sample_id, image_zarr, mask_zarr, probability_zarr, optional threshold_zarr",
    )
    source.add_argument("--sample_root", help="Folder containing per-sample subdirectories")
    source.add_argument("--sample_dirs", nargs="+", help="Explicit sample directories")
    parser.add_argument("--signal_ch", default="2", help="Signal channel id used for automatic sample-dir paths")
    parser.add_argument("--mask_suffix", default="_mask.zarr")
    parser.add_argument("--probability_suffix", default="_prob.zarr")
    parser.add_argument("--threshold_suffix", default="", help="Optional suffix such as _threshold_mask.zarr")
    parser.add_argument("--dataset_name", default="0")
    parser.add_argument("--chunk_size", default="", help="Optional z,y,x block size for QC scanning")
    parser.add_argument("--uncertainty_low", type=float, default=0.4)
    parser.add_argument("--uncertainty_high", type=float, default=0.6)
    parser.add_argument("--small_component_max_voxels", type=int, default=32)
    parser.add_argument("--top_n", type=int, default=30)
    parser.add_argument("--max_per_large_grid", type=int, default=10)
    parser.add_argument("--large_grid_size", default="1024,1024,1024", help="Large grid size z,y,x (default 1024,1024,1024)")
    parser.add_argument("--output_csv", default="review_queue.csv")
    parser.add_argument("--top_csv", default="", help="Optional separate CSV for the top-N rows")
    parser.add_argument("--preview_dir", default="", help="Optional directory for top-N block image and mask TIFF previews")
    parser.add_argument(
        "--skip_missing",
        action="store_true",
        help="Skip records with missing image, mask, probability, or threshold paths",
    )
    parser.add_argument(
        "--no_skip_empty_blocks",
        action="store_true",
        help="Include all-zero mask blocks in QC scanning (default: skip them)",
    )
    parser.add_argument("--workers", type=int, default=32, help="Thread pool workers for block scanning (default 32)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.records_csv:
            records = _read_records_csv(Path(args.records_csv))
        else:
            if args.sample_root:
                root = Path(args.sample_root)
                sample_dirs = sorted(path for path in root.iterdir() if path.is_dir())
            else:
                sample_dirs = [Path(path) for path in args.sample_dirs]
            records = _records_from_sample_dirs(
                sample_dirs,
                signal_ch=args.signal_ch,
                mask_suffix=args.mask_suffix,
                probability_suffix=args.probability_suffix,
                threshold_suffix=args.threshold_suffix,
            )

        ranked = build_review_queue(
            records,
            dataset_name=args.dataset_name,
            chunk_size=_parse_triplet(args.chunk_size),
            uncertainty_low=args.uncertainty_low,
            uncertainty_high=args.uncertainty_high,
            small_component_max_voxels=args.small_component_max_voxels,
            skip_missing=args.skip_missing,
            skip_empty_blocks=not args.no_skip_empty_blocks,
            workers=args.workers,
        )
        output_csv = Path(args.output_csv)
        parsed_chunk_size = _parse_triplet(args.chunk_size) or (256, 256, 256)
        parsed_large_grid_size = _parse_triplet(args.large_grid_size) or (1024, 1024, 1024)
        top_records = select_top_records(
            ranked,
            top_n=max(int(args.top_n), 0),
            max_per_large_grid=max(int(args.max_per_large_grid), 0),
            large_grid_size=parsed_large_grid_size,
            chunk_size=parsed_chunk_size,
        )
        top_csv = Path(args.top_csv) if args.top_csv else output_csv.with_name(f"top{args.top_n}_{output_csv.name}")
        preview_dir = Path(args.preview_dir) if args.preview_dir else top_csv.with_name(f"{top_csv.stem}_previews")
        export_block_previews(
            top_records,
            preview_dir,
            dataset_name=args.dataset_name,
            skip_missing=args.skip_missing,
        )
        _write_csv(output_csv, ranked)
        _write_csv(top_csv, top_records)
        print(
            json.dumps(
                {
                    "success": True,
                    "records": len(ranked),
                    "output_csv": str(output_csv),
                    "top_csv": str(top_csv),
                    "preview_dir": str(preview_dir),
                    "top_n": int(args.top_n),
                    "preview_records": len(top_records),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    except PipelineError as exc:
        print(json.dumps(exc.to_dict(), ensure_ascii=False), file=sys.stderr)
        return exc.exit_code
    except Exception as exc:  # pragma: no cover
        wrapped = PipelineError(ErrorCode.INTERNAL_ERROR, "Unhandled cfos_unet QC error", {"error": str(exc)})
        print(json.dumps(wrapped.to_dict(), ensure_ascii=False), file=sys.stderr)
        return wrapped.exit_code


if __name__ == "__main__":
    sys.exit(main())
