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
    "uncertainty": 0.45,
    "foreground_outlier": 0.25,
    "small_components": 0.20,
    "threshold_disagreement": 0.10,
}


def _iter_chunk_slices(shape: tuple[int, int, int], chunks: tuple[int, int, int]):
    for z in range(0, shape[0], chunks[0]):
        z_end = min(z + chunks[0], shape[0])
        for y in range(0, shape[1], chunks[1]):
            y_end = min(y + chunks[1], shape[1])
            for x in range(0, shape[2], chunks[2]):
                x_end = min(x + chunks[2], shape[2])
                yield (slice(z, z_end), slice(y, y_end), slice(x, x_end))


def _parse_triplet(value: str) -> tuple[int, int, int] | None:
    if not str(value).strip():
        return None
    parts = [part.strip() for part in str(value).split(",") if part.strip()]
    if len(parts) != 3:
        raise PipelineError(ErrorCode.ARGUMENT_INVALID, "Expected three comma-separated integers", {"value": value})
    return (int(parts[0]), int(parts[1]), int(parts[2]))


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
        # Fallback keeps QC usable without scipy, but only reports one coarse
        # foreground component per processed chunk.
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


def compute_sample_qc_metrics(
    *,
    sample_id: str,
    mask_zarr: str | Path,
    probability_zarr: str | Path | None = None,
    threshold_zarr: str | Path | None = None,
    dataset_name: str = "0",
    chunk_size: tuple[int, int, int] | None = None,
    uncertainty_low: float = 0.4,
    uncertainty_high: float = 0.6,
    small_component_max_voxels: int = 32,
) -> dict[str, Any]:
    import numpy as np

    mask_path = Path(mask_zarr)
    if not mask_path.exists():
        raise PipelineError(ErrorCode.INPUT_NOT_FOUND, "Mask Zarr not found", {"mask_zarr": str(mask_path)})

    prob_path = _resolve_optional_path(probability_zarr)
    threshold_path = _resolve_optional_path(threshold_zarr)
    mask_data = open_zarr_dataset(mask_path, dataset_name=dataset_name)
    if prob_path and not prob_path.exists():
        raise PipelineError(
            ErrorCode.INPUT_NOT_FOUND,
            "Probability Zarr not found",
            {"probability_zarr": str(prob_path)},
        )
    prob_data = open_zarr_dataset(prob_path, dataset_name=dataset_name) if prob_path and prob_path.exists() else None
    threshold_data = (
        open_zarr_dataset(threshold_path, dataset_name=dataset_name)
        if threshold_path and threshold_path.exists()
        else None
    )

    shape = tuple(int(v) for v in mask_data.shape)
    chunks = chunk_size or tuple(int(v) for v in getattr(mask_data, "chunks", shape))
    if prob_data is not None and tuple(int(v) for v in prob_data.shape) != shape:
        raise PipelineError(
            ErrorCode.ARGUMENT_INVALID,
            "Probability Zarr shape does not match mask Zarr",
            {"mask_shape": shape, "probability_shape": tuple(prob_data.shape)},
        )
    if threshold_data is not None and tuple(int(v) for v in threshold_data.shape) != shape:
        raise PipelineError(
            ErrorCode.ARGUMENT_INVALID,
            "Threshold Zarr shape does not match mask Zarr",
            {"mask_shape": shape, "threshold_shape": tuple(threshold_data.shape)},
        )

    total_voxels = 0
    foreground_voxels = 0
    uncertain_voxels = 0
    xor_voxels = 0
    union_voxels = 0
    num_components = 0
    small_components = 0

    for slices in _iter_chunk_slices(shape, chunks):
        mask_chunk = np.asarray(mask_data[slices]) > 0
        total_voxels += int(mask_chunk.size)
        foreground_voxels += int(mask_chunk.sum())

        if prob_data is not None:
            prob_chunk = np.asarray(prob_data[slices])
            uncertain = (prob_chunk >= float(uncertainty_low)) & (prob_chunk <= float(uncertainty_high))
            uncertain_voxels += int(uncertain.sum())

        if threshold_data is not None:
            threshold_chunk = np.asarray(threshold_data[slices]) > 0
            xor_voxels += int(np.logical_xor(mask_chunk, threshold_chunk).sum())
            union_voxels += int(np.logical_or(mask_chunk, threshold_chunk).sum())

        chunk_components, chunk_small_components = _component_stats(
            mask_chunk,
            small_component_max_voxels=int(small_component_max_voxels),
        )
        num_components += chunk_components
        small_components += chunk_small_components

    total_voxels = max(total_voxels, 1)
    threshold_disagreement = (xor_voxels / union_voxels) if union_voxels else 0.0
    return {
        "sample_id": sample_id,
        "mask_zarr": str(mask_path),
        "probability_zarr": str(prob_path) if prob_path else "",
        "threshold_zarr": str(threshold_path) if threshold_path else "",
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
    foreground_outlier_scores = _robust_outlier_scores(records, "fg_ratio")

    scored = []
    for index, record in enumerate(records):
        uncertainty_score = uncertainty_scores[index] if index < len(uncertainty_scores) else 0.0
        foreground_outlier_score = (
            foreground_outlier_scores[index] if index < len(foreground_outlier_scores) else 0.0
        )
        small_component_score = min(max(_safe_float(record.get("small_component_ratio")), 0.0), 1.0)
        threshold_score = min(max(_safe_float(record.get("threshold_disagreement")), 0.0), 1.0)
        review_score = (
            weights["uncertainty"] * uncertainty_score
            + weights["foreground_outlier"] * foreground_outlier_score
            + weights["small_components"] * small_component_score
            + weights["threshold_disagreement"] * threshold_score
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
        for key in ("mask_zarr", "probability_zarr", "threshold_zarr"):
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
        "review_score",
        "uncertain_ratio",
        "fg_ratio",
        "num_components",
        "components_per_million_voxels",
        "small_component_ratio",
        "threshold_disagreement",
        "mask_zarr",
        "probability_zarr",
        "threshold_zarr",
        "shape",
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


def build_review_queue(
    records: list[dict[str, Any]],
    *,
    dataset_name: str = "0",
    chunk_size: tuple[int, int, int] | None = None,
    uncertainty_low: float = 0.4,
    uncertainty_high: float = 0.6,
    small_component_max_voxels: int = 32,
    skip_missing: bool = False,
) -> list[dict[str, Any]]:
    metrics = []
    for record in records:
        mask_path = Path(str(record.get("mask_zarr", "")))
        probability_path = _resolve_optional_path(record.get("probability_zarr"))
        missing_probability = probability_path is not None and not probability_path.exists()
        if skip_missing and (not mask_path.exists() or missing_probability):
            continue
        metrics.append(
            compute_sample_qc_metrics(
                sample_id=str(record.get("sample_id") or mask_path.parent.name or mask_path.stem),
                mask_zarr=mask_path,
                probability_zarr=probability_path,
                threshold_zarr=_resolve_optional_path(record.get("threshold_zarr")),
                dataset_name=dataset_name,
                chunk_size=chunk_size,
                uncertainty_low=uncertainty_low,
                uncertainty_high=uncertainty_high,
                small_component_max_voxels=small_component_max_voxels,
            )
        )
    return score_qc_records(metrics)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rank cFos U-Net outputs for active-learning review")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--records_csv", help="CSV with sample_id, mask_zarr, probability_zarr, optional threshold_zarr")
    source.add_argument("--sample_root", help="Folder containing per-sample subdirectories")
    source.add_argument("--sample_dirs", nargs="+", help="Explicit sample directories")
    parser.add_argument("--signal_ch", default="2", help="Signal channel id used for automatic sample-dir paths")
    parser.add_argument("--mask_suffix", default="_mask.zarr")
    parser.add_argument("--probability_suffix", default="_prob.zarr")
    parser.add_argument("--threshold_suffix", default="", help="Optional suffix such as _threshold_mask.zarr")
    parser.add_argument("--dataset_name", default="0")
    parser.add_argument("--chunk_size", default="", help="Optional z,y,x chunks for QC scanning")
    parser.add_argument("--uncertainty_low", type=float, default=0.4)
    parser.add_argument("--uncertainty_high", type=float, default=0.6)
    parser.add_argument("--small_component_max_voxels", type=int, default=32)
    parser.add_argument("--top_n", type=int, default=10)
    parser.add_argument("--output_csv", default="review_queue.csv")
    parser.add_argument("--top_csv", default="", help="Optional separate CSV for the top-N rows")
    parser.add_argument("--skip_missing", action="store_true", help="Skip records with missing mask/probability paths")
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
        )
        output_csv = Path(args.output_csv)
        _write_csv(output_csv, ranked)
        top_records = ranked[: max(int(args.top_n), 0)]
        top_csv = Path(args.top_csv) if args.top_csv else output_csv.with_name(f"top{args.top_n}_{output_csv.name}")
        _write_csv(top_csv, top_records)
        print(
            json.dumps(
                {
                    "success": True,
                    "records": len(ranked),
                    "output_csv": str(output_csv),
                    "top_csv": str(top_csv),
                    "top_n": int(args.top_n),
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
