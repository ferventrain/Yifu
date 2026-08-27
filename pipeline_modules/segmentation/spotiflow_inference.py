from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

try:
    from pipeline_modules.segmentation.zarr_utils import open_zarr_dataset
    from pipeline_modules.utils.errors import ErrorCode, PipelineError
    from pipeline_modules.utils.run_manifest import write_run_manifest
except ImportError:  # pragma: no cover
    from .zarr_utils import open_zarr_dataset
    from ..utils.errors import ErrorCode, PipelineError
    from ..utils.run_manifest import write_run_manifest

logger = logging.getLogger(__name__)


DEFAULT_MODEL_DIR = Path(__file__).resolve().parents[2] / "model" / "spotiflow"
DEFAULT_CFG = Path(__file__).resolve().parents[1] / "registration" / "Region_Csv_Rev1_updated.CSV"
DEFAULT_CHECKPOINT_TILES = 32
DEFAULT_BATCH_SIZE = 1


def _configure_logging(json_logs: bool) -> None:
    if json_logs:
        class _JsonFormatter(logging.Formatter):
            def format(self, record: logging.LogRecord) -> str:
                return json.dumps(
                    {
                        "level": record.levelname,
                        "logger": record.name,
                        "message": record.getMessage(),
                    },
                    ensure_ascii=False,
                )

        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(_JsonFormatter())
        logging.root.handlers.clear()
        logging.root.addHandler(handler)
        logging.root.setLevel(logging.INFO)
    else:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


def _parse_triplet(value: str) -> tuple[int, int, int] | None:
    if not str(value).strip():
        return None
    parts = [part.strip() for part in str(value).split(",") if part.strip()]
    if len(parts) != 3:
        raise PipelineError(ErrorCode.ARGUMENT_INVALID, "Expected z,y,x triplet", {"value": value})
    return (int(parts[0]), int(parts[1]), int(parts[2]))


def _iter_tile_slices(shape: tuple[int, int, int], tile_size: tuple[int, int, int], overlap: int):
    step = tuple(max(1, int(size) - int(overlap) * 2) for size in tile_size)
    starts_by_axis = []
    for axis_length, tile, axis_step in zip(shape, tile_size, step):
        if axis_length <= tile:
            starts_by_axis.append([0])
            continue
        starts = list(range(0, axis_length - tile + 1, axis_step))
        if starts[-1] != axis_length - tile:
            starts.append(axis_length - tile)
        starts_by_axis.append(starts)

    for z0 in starts_by_axis[0]:
        z1 = min(z0 + tile_size[0], shape[0])
        for y0 in starts_by_axis[1]:
            y1 = min(y0 + tile_size[1], shape[1])
            for x0 in starts_by_axis[2]:
                x1 = min(x0 + tile_size[2], shape[2])
                yield (slice(z0, z1), slice(y0, y1), slice(x0, x1))


def _point_inside_keep_window(
    point: np.ndarray,
    slices: tuple[slice, slice, slice],
    shape: tuple[int, int, int],
    overlap: int,
) -> bool:
    if overlap <= 0:
        return True

    for axis, axis_slice in enumerate(slices):
        local = float(point[axis])
        start = int(axis_slice.start or 0)
        stop = int(axis_slice.stop or shape[axis])
        length = stop - start
        lower = 0 if start == 0 else overlap
        upper = length if stop >= shape[axis] else length - overlap
        if local < lower or local >= upper:
            return False
    return True


def _load_region_lookup(cfg_path: str | Path) -> dict[int, dict[str, str]]:
    import pandas as pd

    cfg_path = Path(cfg_path)
    if not cfg_path.exists():
        raise PipelineError(ErrorCode.INPUT_NOT_FOUND, "Region CSV not found", {"cfg": str(cfg_path)})

    region_df = pd.read_csv(cfg_path)
    required = {"id", "name", "acronym"}
    missing = required.difference(region_df.columns)
    if missing:
        raise PipelineError(
            ErrorCode.ARGUMENT_INVALID,
            "Region CSV missing required columns",
            {"missing": sorted(missing), "cfg": str(cfg_path)},
        )

    lookup: dict[int, dict[str, str]] = {}
    for _, row in region_df.iterrows():
        region_id = int(row["id"])
        lookup[region_id] = {
            "region_id": str(region_id),
            "region_name": str(row["name"]),
            "region_acronym": _parse_acronym_text(row["acronym"]),
        }
    return lookup


def _parse_acronym_text(acronym_text: object) -> str:
    import ast

    try:
        acronym_values = ast.literal_eval(str(acronym_text))
        if isinstance(acronym_values, list) and acronym_values:
            return str(acronym_values[-1]).strip()
    except Exception:
        pass
    return str(acronym_text).strip()


def _write_points_csv(points: list[dict[str, Any]], output_csv: str | Path) -> Path:
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "z",
        "y",
        "x",
        "region_id",
        "region_name",
        "region_acronym",
        "tile_z0",
        "tile_y0",
        "tile_x0",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(points)
    return output_path


def _write_region_counts_csv(region_rows: list[dict[str, Any]], output_csv: str | Path) -> Path:
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["region_id", "region_name", "region_acronym", "signal_count"]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(region_rows)
    return output_path


def _write_summary_json(summary: dict[str, Any], output_json: str | Path) -> Path:
    output_path = Path(output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return output_path


def _resolve_output_path(path_value: str | Path | None, fallback: str | Path) -> Path:
    if path_value is None or not str(path_value).strip():
        return Path(fallback)
    return Path(path_value)


def _checkpoint_dir_for(output_path: Path, run_identity: dict[str, Any]) -> Path:
    return output_path.parent / f"{output_path.stem}_spotiflow_resume"


def _legacy_checkpoint_dirs(output_path: Path) -> list[Path]:
    pattern = f"{output_path.stem}_spotiflow_resume_*"
    return sorted(output_path.parent.glob(pattern), key=lambda path: path.stat().st_mtime, reverse=True)


def _load_checkpoint_meta(meta_path: Path) -> dict[str, Any] | None:
    if not meta_path.exists():
        return None
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _checkpoint_meta_matches(existing: dict[str, Any] | None, current: dict[str, Any]) -> bool:
    if existing is None:
        return True
    return existing == current


def _resolve_checkpoint_dir(output_path: Path, run_identity: dict[str, Any]) -> Path:
    checkpoint_dir = _checkpoint_dir_for(output_path, run_identity)
    if checkpoint_dir.exists():
        return checkpoint_dir

    for legacy_dir in _legacy_checkpoint_dirs(output_path):
        existing = _load_checkpoint_meta(legacy_dir / "checkpoint_meta.json")
        if _checkpoint_meta_matches(existing, run_identity):
            return legacy_dir

    return checkpoint_dir


def _batch_csv_path(checkpoint_dir: Path, batch_index: int) -> Path:
    return checkpoint_dir / f"batch_{batch_index:06d}.csv"


def _batch_qc_csv_path(checkpoint_dir: Path, batch_index: int) -> Path:
    return checkpoint_dir / f"batch_{batch_index:06d}_tile_qc.csv"


def _tile_qc_fieldnames() -> list[str]:
    return [
        "tile_index",
        "batch_index",
        "z0",
        "z1",
        "y0",
        "y1",
        "x0",
        "x1",
        "skipped",
        "raw_max",
        "raw_mean",
        "point_count",
        "prob_thresh",
        "prob_min",
        "prob_mean",
        "prob_median",
        "near_threshold_count",
        "uncertainty_score",
        "preview_path",
    ]


def _write_tile_qc_batch_csv(rows: list[dict[str, Any]], output_csv: str | Path) -> Path:
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=_tile_qc_fieldnames())
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def _merge_tile_qc_csvs(batch_paths: list[Path], output_csv: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=_tile_qc_fieldnames())
        writer.writeheader()
        for batch_path in batch_paths:
            if not batch_path.exists():
                continue
            with batch_path.open("r", newline="", encoding="utf-8") as batch_handle:
                reader = csv.DictReader(batch_handle)
                for row in reader:
                    writer.writerow(row)
                    rows.append(row)
    return rows


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(parsed) or math.isinf(parsed):
        return default
    return parsed


def _coerce_optional_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (list, tuple, np.ndarray)):
        arr = np.asarray(value, dtype=np.float64).reshape(-1)
        if arr.size == 0:
            return None
        return _safe_float(arr[0], default=0.0)
    return _safe_float(value, default=0.0)


def _resolved_model_prob_thresh(model: Any, override: float | None) -> float | None:
    if override is not None:
        return float(override)
    for attr in ("_prob_thresh", "prob_thresh", "threshold"):
        resolved = _coerce_optional_float(getattr(model, attr, None))
        if resolved is not None:
            return resolved
    return None


def _extract_spot_probabilities(details: Any, point_count: int) -> np.ndarray:
    for attr in ("prob", "probs", "probability", "probabilities", "score", "scores"):
        values = getattr(details, attr, None)
        if values is None:
            continue
        arr = np.asarray(values, dtype=np.float64).reshape(-1)
        if arr.size == point_count:
            return arr
    return np.asarray([], dtype=np.float64)


def _tile_bounds(slices: tuple[slice, slice, slice], shape: tuple[int, int, int]) -> tuple[int, int, int, int, int, int]:
    z0 = int(slices[0].start or 0)
    z1 = int(slices[0].stop or shape[0])
    y0 = int(slices[1].start or 0)
    y1 = int(slices[1].stop or shape[1])
    x0 = int(slices[2].start or 0)
    x1 = int(slices[2].stop or shape[2])
    return z0, z1, y0, y1, x0, x1


def _tile_qc_row(
    *,
    tile_index: int,
    batch_index: int,
    slices: tuple[slice, slice, slice],
    shape: tuple[int, int, int],
    tile: np.ndarray,
    point_count: int,
    details: Any | None,
    prob_thresh: float | None,
    skipped: bool,
) -> dict[str, Any]:
    z0, z1, y0, y1, x0, x1 = _tile_bounds(slices, shape)
    probs = _extract_spot_probabilities(details, point_count) if details is not None else np.asarray([], dtype=np.float64)
    prob_min = float(np.min(probs)) if probs.size else ""
    prob_mean = float(np.mean(probs)) if probs.size else ""
    prob_median = float(np.median(probs)) if probs.size else ""
    near_threshold_count = 0
    uncertainty_score = 0.0
    if probs.size and prob_thresh is not None:
        margin = max(0.02, abs(float(prob_thresh)) * 0.15)
        distances = np.abs(probs - float(prob_thresh))
        near = distances <= margin
        near_threshold_count = int(near.sum())
        if near_threshold_count:
            uncertainty_score = float(np.sum(1.0 - distances[near] / margin))

    return {
        "tile_index": tile_index,
        "batch_index": batch_index,
        "z0": z0,
        "z1": z1,
        "y0": y0,
        "y1": y1,
        "x0": x0,
        "x1": x1,
        "skipped": int(skipped),
        "raw_max": float(tile.max(initial=0)),
        "raw_mean": float(tile.mean()) if tile.size else 0.0,
        "point_count": int(point_count),
        "prob_thresh": "" if prob_thresh is None else float(prob_thresh),
        "prob_min": prob_min,
        "prob_mean": prob_mean,
        "prob_median": prob_median,
        "near_threshold_count": near_threshold_count,
        "uncertainty_score": uncertainty_score,
        "preview_path": "",
    }


def _select_top_uncertain_tile_rows(rows: list[dict[str, Any]], top_n: int) -> list[dict[str, Any]]:
    candidates = [dict(row) for row in rows if int(_safe_float(row.get("skipped"), 0)) == 0]
    candidates.sort(
        key=lambda row: (
            _safe_float(row.get("uncertainty_score")),
            _safe_float(row.get("near_threshold_count")),
            _safe_float(row.get("point_count")),
            _safe_float(row.get("raw_max")),
        ),
        reverse=True,
    )
    return candidates[: max(0, int(top_n))]


def _write_top_tile_csv(rows: list[dict[str, Any]], output_csv: str | Path) -> Path:
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=_tile_qc_fieldnames())
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def _export_tile_qc_previews(
    rows: list[dict[str, Any]],
    *,
    data_in: Any,
    preview_dir: str | Path,
    mode: str = "mip",
) -> list[dict[str, Any]]:
    import tifffile

    preview_path = Path(preview_dir)
    preview_path.mkdir(parents=True, exist_ok=True)
    normalized_mode = str(mode).strip().lower()
    updated_rows: list[dict[str, Any]] = []
    for row in rows:
        z0, z1 = int(row["z0"]), int(row["z1"])
        y0, y1 = int(row["y0"]), int(row["y1"])
        x0, x1 = int(row["x0"]), int(row["x1"])
        tile = np.asarray(data_in[(slice(z0, z1), slice(y0, y1), slice(x0, x1))])
        block_id = f"tile_{int(row['tile_index']):06d}_z{z0}-{z1}_y{y0}-{y1}_x{x0}-{x1}"
        if normalized_mode == "volume":
            output_path = preview_path / f"{block_id}.tiff"
            tifffile.imwrite(str(output_path), tile)
        else:
            output_path = preview_path / f"{block_id}_mip_z.tiff"
            tifffile.imwrite(str(output_path), tile.max(axis=0))
        updated = dict(row)
        updated["preview_path"] = str(output_path)
        updated_rows.append(updated)
    return updated_rows


def _predict_tiles_batch(
    model: Any,
    tiles: list[np.ndarray],
    *,
    prob_thresh: float | None,
    min_distance: int,
    peak_mode: str,
    normalizer: Any,
    subpix_radius: int,
    device: str,
    use_tuned_tile_overlap: bool,
) -> list[tuple[np.ndarray, Any]]:
    from spotiflow.utils import center_crop, center_pad, flow_to_vector, normalize, prob_to_points, subpixel_offset

    actual_n_dims = 3
    corr_grid = np.asarray(model.config.grid, dtype=np.float64)
    div_by = tuple(
        model.config.downsample_factors[0][0] ** model.config.levels for _ in range(actual_n_dims)
    ) + (1,)
    if model.config.is_3d and any(int(g) > 1 for g in model.config.grid):
        div_by = tuple(int(g) * int(d) for g, d in zip((*model.config.grid, 1), div_by))

    normalizer_fn = normalizer
    if isinstance(normalizer, str) and normalizer == "auto":
        normalizer_fn = normalize

    prepared: list[np.ndarray] = []
    paddings: list[Any] = []
    for tile in tiles:
        x = tile.astype(np.float32)
        x = x[..., None]
        if callable(normalizer_fn):
            x = normalizer_fn(x)
        pad_shape = tuple(int(int(d) * np.ceil(s / int(d))) for s, d in zip(x.shape, div_by))
        x, padding = center_pad(x, pad_shape, mode="reflect")
        prepared.append(x)
        paddings.append(padding)

    img_t = torch.from_numpy(np.stack(prepared)).to(device)
    img_t = img_t.permute(0, 4, 1, 2, 3)

    model.eval()
    with torch.inference_mode():
        out = model(img_t)

    resolved_thresh: float
    if prob_thresh is None:
        _pt = getattr(model, "_prob_thresh", 0.5)
        if isinstance(_pt, (list, tuple, np.ndarray)):
            resolved_thresh = float(_pt[0])
        else:
            resolved_thresh = float(_pt)
    elif isinstance(prob_thresh, (list, tuple, np.ndarray)):
        resolved_thresh = float(prob_thresh[0])
    else:
        resolved_thresh = float(prob_thresh)

    # flow[0] has shape (4, D', H', W') regardless of batch size (no batch dim)
    flow_3d_t = out["flow"][0].permute(1, 2, 3, 0).detach().cpu().numpy() if subpix_radius >= 0 else None

    results: list[tuple[np.ndarray, Any]] = []
    for i, tile in enumerate(tiles):
        padding = paddings[i]
        orig_shape = np.asarray(tile.shape[:actual_n_dims], dtype=np.int64)

        if model.config.is_3d and int(model.config.out_channels) > 1:
            y = model._sigmoid(out["heatmaps"][0][i]).detach().cpu().numpy()
        else:
            y = model._sigmoid(out["heatmaps"][0][i].squeeze(0)).detach().cpu().numpy()

        out_shape = tuple(int(s) // int(g) for s, g in zip(orig_shape, corr_grid))
        y = center_crop(y, out_shape)

        pts = prob_to_points(
            y,
            prob_thresh=resolved_thresh,
            exclude_border=False,
            mode=peak_mode,
            min_distance=min_distance,
        )
        probs = y[tuple(pts.astype(int).T)].tolist() if pts.size else []

        subpix_out: Any = None
        if subpix_radius >= 0 and pts.size and flow_3d_t is not None:
            subpix_tile = flow_to_vector(flow_3d_t, sigma=model.config.sigma)
            subpix_tile = center_crop(subpix_tile, out_shape)
            offset = subpixel_offset(pts, subpix_tile, y, radius=subpix_radius)
            pts = pts + offset
            subpix_out = subpix_tile

        pad_correction = np.array([int(p[0]) for p in padding[:actual_n_dims]], dtype=np.float64)[None] / corr_grid
        pts = pts - pad_correction

        if model.config.is_3d and any(int(g) > 1 for g in model.config.grid):
            pts = pts * np.asarray(model.config.grid, dtype=np.float64)

        from types import SimpleNamespace

        details = SimpleNamespace(
            prob=np.asarray(probs, dtype=np.float64),
            heatmap=y,
            subpix=subpix_out,
            flow=None,
        )
        results.append((pts, details))

    return results


def _append_timing_row(timing_csv: Path, row: dict[str, Any]) -> None:
    timing_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "batch_index",
        "start_tile",
        "end_tile",
        "tiles",
        "skipped_tiles",
        "points",
        "read_seconds",
        "predict_seconds",
        "post_seconds",
        "write_seconds",
        "total_seconds",
    ]
    write_header = not timing_csv.exists()
    with timing_csv.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _write_points_batch_csv(points: list[dict[str, Any]], output_csv: str | Path) -> Path:
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "z",
        "y",
        "x",
        "region_id",
        "region_name",
        "region_acronym",
        "tile_z0",
        "tile_y0",
        "tile_x0",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(points)
    return output_path


def _merge_batch_csvs(batch_paths: list[Path], output_csv: Path) -> list[dict[str, Any]]:
    points: list[dict[str, Any]] = []
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "z",
        "y",
        "x",
        "region_id",
        "region_name",
        "region_acronym",
        "tile_z0",
        "tile_y0",
        "tile_x0",
    ]
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for batch_path in batch_paths:
            if not batch_path.exists():
                continue
            with batch_path.open("r", newline="", encoding="utf-8") as batch_handle:
                reader = csv.DictReader(batch_handle)
                for row in reader:
                    writer.writerow(row)
                    points.append(row)
    return points


def _count_points_by_region(
    points: list[dict[str, Any]],
    region_lookup: dict[int, dict[str, str]],
) -> list[dict[str, Any]]:
    counts: dict[int, int] = {}
    for point in points:
        region_id = int(point.get("region_id", 0))
        if region_id <= 0:
            continue
        counts[region_id] = counts.get(region_id, 0) + 1

    rows = []
    for region_id, count in sorted(counts.items()):
        region_info = region_lookup.get(
            region_id,
            {"region_id": str(region_id), "region_name": str(region_id), "region_acronym": ""},
        )
        rows.append(
            {
                "region_id": region_id,
                "region_name": region_info["region_name"],
                "region_acronym": region_info["region_acronym"],
                "signal_count": int(count),
            }
        )
    return rows


def run_spotiflow_inference(
    *,
    input_zarr: str | Path,
    output_csv: str | Path | None = None,
    model_dir: str | Path = DEFAULT_MODEL_DIR,
    label_zarr: str | Path | None = None,
    region_counts_csv: str | Path | None = None,
    summary_json: str | Path | None = None,
    cfg: str | Path = DEFAULT_CFG,
    dataset_name: str = "0",
    which: str = "best",
    prob_thresh: float | None = None,
    min_distance: int = 1,
    tile_size: tuple[int, int, int] | None = None,
    tile_overlap: int = 16,
    skip_below_threshold: float | None = 100.0,
    device: str = "auto",
    peak_mode: str = "fast",
    normalizer: str | None = "auto",
    subpix: bool | None = None,
    use_tuned_tile_overlap: bool = False,
    batch_size: int = DEFAULT_BATCH_SIZE,
    checkpoint_tiles: int = DEFAULT_CHECKPOINT_TILES,
    qc_top_n: int = 0,
    qc_tile_csv: str | Path | None = None,
    qc_top_csv: str | Path | None = None,
    qc_preview_dir: str | Path | None = None,
    qc_preview_mode: str = "mip",
) -> dict[str, Any]:
    started_at = time.time()
    input_path = Path(input_zarr)
    model_path = Path(model_dir)
    if not input_path.exists():
        raise PipelineError(ErrorCode.INPUT_NOT_FOUND, "Input Zarr not found", {"input_zarr": str(input_path)})
    if not model_path.exists():
        raise PipelineError(ErrorCode.INPUT_NOT_FOUND, "Spotiflow model folder not found", {"model_dir": str(model_path)})
    if not (model_path / "config.yaml").exists() or not (model_path / f"{which}.pt").exists():
        raise PipelineError(
            ErrorCode.INPUT_NOT_FOUND,
            "Spotiflow model folder must contain config.yaml and checkpoint",
            {"model_dir": str(model_path), "which": which},
        )

    output_path = _resolve_output_path(output_csv, input_path.with_name(f"{input_path.stem}_spotiflow_points.csv"))
    region_counts_path = _resolve_output_path(
        region_counts_csv,
        output_path.with_name(output_path.stem.replace("_points", "") + "_region_counts.csv"),
    )
    summary_path = _resolve_output_path(
        summary_json,
        output_path.with_name(output_path.stem.replace("_points", "") + "_summary.json"),
    )
    checkpoint_tiles = max(1, int(checkpoint_tiles))
    qc_top_n = max(0, int(qc_top_n or 0))

    data_in = open_zarr_dataset(input_path, dataset_name=dataset_name)
    shape = tuple(int(v) for v in data_in.shape)
    if len(shape) != 3:
        raise PipelineError(ErrorCode.ARGUMENT_INVALID, "Spotiflow inference expects a 3D Zarr array", {"shape": shape})

    label_data = None
    region_lookup: dict[int, dict[str, str]] = {}
    label_path = Path(label_zarr) if label_zarr else None
    if label_path:
        if not label_path.exists():
            raise PipelineError(ErrorCode.INPUT_NOT_FOUND, "Label Zarr not found", {"label_zarr": str(label_path)})
        label_data = open_zarr_dataset(label_path, dataset_name=dataset_name)
        if tuple(int(v) for v in label_data.shape) != shape:
            raise PipelineError(
                ErrorCode.ARGUMENT_INVALID,
                "Input Zarr and label Zarr shapes differ",
                {"input_shape": shape, "label_shape": tuple(int(v) for v in label_data.shape)},
            )
        region_lookup = _load_region_lookup(cfg)

    try:
        from spotiflow.model import Spotiflow
    except ModuleNotFoundError as exc:
        raise PipelineError(
            ErrorCode.DEPENDENCY_MISSING,
            "Spotiflow is not installed in the current Python environment",
            {"install_hint": "pip install spotiflow"},
        ) from exc

    logger.info("Loading Spotiflow model from %s", model_path)
    model = Spotiflow.from_folder(str(model_path), which=which, map_location=device, verbose=True)
    resolved_prob_thresh = _resolved_model_prob_thresh(model, prob_thresh)
    chunk_shape = tuple(int(v) for v in getattr(data_in, "chunks", shape))
    resolved_tile_size = tile_size or chunk_shape
    tile_slices = list(_iter_tile_slices(shape, resolved_tile_size, max(0, int(tile_overlap))))

    run_identity = {
        "input_zarr": str(input_path),
        "model_dir": str(model_path),
        "label_zarr": str(label_path) if label_path else "",
        "dataset_name": dataset_name,
        "which": which,
        "prob_thresh": prob_thresh,
        "min_distance": min_distance,
        "tile_size": list(resolved_tile_size),
        "tile_overlap": int(tile_overlap),
        "skip_below_threshold": skip_below_threshold,
        "device": device,
        "peak_mode": peak_mode,
        "normalizer": normalizer,
        "subpix": subpix,
        "use_tuned_tile_overlap": use_tuned_tile_overlap,
        "checkpoint_tiles": checkpoint_tiles,
        "shape": list(shape),
    }
    checkpoint_dir = _resolve_checkpoint_dir(output_path, run_identity)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    meta_path = checkpoint_dir / "checkpoint_meta.json"
    if not _checkpoint_meta_matches(_load_checkpoint_meta(meta_path), run_identity):
        raise PipelineError(
            ErrorCode.ARGUMENT_INVALID,
            "Existing Spotiflow checkpoint was created with different parameters",
            {
                "checkpoint_dir": str(checkpoint_dir),
                "hint": "Remove the checkpoint directory to start a fresh run, or restore the previous config to resume it.",
            },
        )
    meta_path.write_text(json.dumps(run_identity, indent=2, ensure_ascii=False), encoding="utf-8")
    timing_csv = checkpoint_dir / "batch_timing.csv"
    qc_tile_path = _resolve_output_path(
        qc_tile_csv,
        output_path.with_name(output_path.stem.replace("_points", "") + "_tile_qc.csv"),
    )
    qc_top_path = _resolve_output_path(
        qc_top_csv,
        output_path.with_name(output_path.stem.replace("_points", "") + f"_top{qc_top_n}_uncertain_tiles.csv"),
    )
    qc_preview_path = Path(qc_preview_dir) if qc_preview_dir else output_path.parent / (
        output_path.stem.replace("_points", "") + f"_top{qc_top_n}_uncertain_tiles"
    )

    batch_paths: list[Path] = []
    batch_qc_paths: list[Path] = []
    skipped_tiles = 0
    total_batches = (len(tile_slices) + checkpoint_tiles - 1) // checkpoint_tiles
    for batch_index in tqdm(range(total_batches), desc="Spotiflow inference", unit="batch"):
        batch_path = _batch_csv_path(checkpoint_dir, batch_index)
        batch_qc_path = _batch_qc_csv_path(checkpoint_dir, batch_index)
        batch_paths.append(batch_path)
        batch_qc_paths.append(batch_qc_path)
        if batch_path.exists() and (qc_top_n <= 0 or batch_qc_path.exists()):
            continue

        batch_points: list[dict[str, Any]] = []
        batch_qc_rows: list[dict[str, Any]] = []
        start_idx = batch_index * checkpoint_tiles
        end_idx = min(start_idx + checkpoint_tiles, len(tile_slices))
        batch_started = time.perf_counter()
        read_seconds = 0.0
        predict_seconds = 0.0
        post_seconds = 0.0
        batch_skipped_tiles = 0
        subpix_is_false = subpix is False
        subpix_is_true = subpix is True
        if subpix_is_false:
            subpix_radius = -1
        elif subpix_is_true:
            subpix_radius = 0
        elif subpix is None:
            subpix_radius = 0 if getattr(model.config, "compute_flow", True) else -1
        else:
            subpix_radius = 0

        batched_pending: list[tuple[np.ndarray, int, tuple[slice, slice, slice]]] = []

        def _flush_batched() -> None:
            nonlocal predict_seconds, post_seconds
            if not batched_pending:
                return
            tiles_batch = [p[0] for p in batched_pending]
            pred_started = time.perf_counter()
            batch_results = _predict_tiles_batch(
                model,
                tiles_batch,
                prob_thresh=prob_thresh,
                min_distance=int(min_distance),
                peak_mode=peak_mode,
                normalizer=normalizer,
                subpix_radius=subpix_radius,
                device=device,
                use_tuned_tile_overlap=use_tuned_tile_overlap,
            )
            predict_seconds += time.perf_counter() - pred_started

            post_started = time.perf_counter()
            for (pred_pts, det), (_tile, ti, sl) in zip(batch_results, batched_pending):
                _predicted_points = np.asarray(pred_pts)
                _point_count = 0
                if _predicted_points.size == 0:
                    if qc_top_n > 0:
                        batch_qc_rows.append(
                            _tile_qc_row(
                                tile_index=ti, batch_index=batch_index, slices=sl, shape=shape, tile=_tile,
                                point_count=0, details=det, prob_thresh=resolved_prob_thresh, skipped=False,
                            )
                        )
                    continue
                if _predicted_points.ndim == 1:
                    _predicted_points = _predicted_points.reshape(1, -1)
                _point_count = int(_predicted_points.shape[0])

                tile_origin = np.array([int(slc.start or 0) for slc in sl], dtype=np.float64)
                label_tile = np.asarray(label_data[sl]) if label_data is not None else None
                for local_point in _predicted_points[:, :3]:
                    if not _point_inside_keep_window(local_point, sl, shape, int(tile_overlap)):
                        continue
                    global_point = local_point.astype(np.float64, copy=False) + tile_origin
                    rounded = np.rint(global_point).astype(np.int64)
                    if np.any(rounded < 0) or np.any(rounded >= np.asarray(shape)):
                        continue
                    region_id = 0
                    region_name = ""
                    region_acronym = ""
                    if label_tile is not None:
                        local_rounded = np.rint(local_point).astype(np.int64)
                        if np.any(local_rounded < 0) or np.any(local_rounded >= np.asarray(label_tile.shape)):
                            continue
                        region_id = int(label_tile[tuple(local_rounded.tolist())])
                        region_info = region_lookup.get(region_id)
                        if region_info:
                            region_name = region_info["region_name"]
                            region_acronym = region_info["region_acronym"]
                    batch_points.append(
                        {
                            "z": float(global_point[0]), "y": float(global_point[1]), "x": float(global_point[2]),
                            "region_id": region_id, "region_name": region_name, "region_acronym": region_acronym,
                            "tile_z0": int(tile_origin[0]), "tile_y0": int(tile_origin[1]), "tile_x0": int(tile_origin[2]),
                        }
                    )
                if qc_top_n > 0:
                    batch_qc_rows.append(
                        _tile_qc_row(
                            tile_index=ti, batch_index=batch_index, slices=sl, shape=shape, tile=_tile,
                            point_count=_point_count, details=det, prob_thresh=resolved_prob_thresh, skipped=False,
                        )
                    )
            post_seconds += time.perf_counter() - post_started
            batched_pending.clear()

        for tile_index, slices in enumerate(tile_slices[start_idx:end_idx], start=start_idx):
            read_started = time.perf_counter()
            tile = np.asarray(data_in[slices])
            read_seconds += time.perf_counter() - read_started
            if skip_below_threshold is not None and float(tile.max(initial=0)) < float(skip_below_threshold):
                skipped_tiles += 1
                batch_skipped_tiles += 1
                if qc_top_n > 0:
                    batch_qc_rows.append(
                        _tile_qc_row(
                            tile_index=tile_index, batch_index=batch_index,
                            slices=slices, shape=shape, tile=tile,
                            point_count=0, details=None,
                            prob_thresh=resolved_prob_thresh, skipped=True,
                        )
                    )
                continue

            if batch_size > 1 and subpix_radius < 0:
                batched_pending.append((tile, tile_index, slices))
                if len(batched_pending) >= batch_size:
                    _flush_batched()
            else:
                predict_started = time.perf_counter()
                predicted_points, details = model.predict(
                    tile,
                    prob_thresh=prob_thresh,
                    min_distance=int(min_distance),
                    exclude_border=False,
                    peak_mode=peak_mode,
                    normalizer=normalizer,
                    subpix=subpix,
                    verbose=False,
                    device=device,
                    use_tuned_tile_overlap=use_tuned_tile_overlap,
                )
                predict_seconds += time.perf_counter() - predict_started

                post_started = time.perf_counter()
                predicted_points = np.asarray(predicted_points)
                point_count_before_keep_window = 0
                if predicted_points.size == 0:
                    if qc_top_n > 0:
                        batch_qc_rows.append(
                            _tile_qc_row(
                                tile_index=tile_index, batch_index=batch_index,
                                slices=slices, shape=shape, tile=tile,
                                point_count=0, details=details,
                                prob_thresh=resolved_prob_thresh, skipped=False,
                            )
                        )
                    post_seconds += time.perf_counter() - post_started
                    continue
                if predicted_points.ndim == 1:
                    predicted_points = predicted_points.reshape(1, -1)
                point_count_before_keep_window = int(predicted_points.shape[0])

                tile_origin = np.array([int(slc.start or 0) for slc in slices], dtype=np.float64)
                label_tile = np.asarray(label_data[slices]) if label_data is not None else None
                for local_point in predicted_points[:, :3]:
                    if not _point_inside_keep_window(local_point, slices, shape, int(tile_overlap)):
                        continue
                    global_point = local_point.astype(np.float64, copy=False) + tile_origin
                    rounded = np.rint(global_point).astype(np.int64)
                    if np.any(rounded < 0) or np.any(rounded >= np.asarray(shape)):
                        continue
                    region_id = 0
                    region_name = ""
                    region_acronym = ""
                    if label_tile is not None:
                        local_rounded = np.rint(local_point).astype(np.int64)
                        if np.any(local_rounded < 0) or np.any(local_rounded >= np.asarray(label_tile.shape)):
                            continue
                        region_id = int(label_tile[tuple(local_rounded.tolist())])
                        region_info = region_lookup.get(region_id)
                        if region_info:
                            region_name = region_info["region_name"]
                            region_acronym = region_info["region_acronym"]
                    batch_points.append(
                        {
                            "z": float(global_point[0]), "y": float(global_point[1]), "x": float(global_point[2]),
                            "region_id": region_id, "region_name": region_name, "region_acronym": region_acronym,
                            "tile_z0": int(tile_origin[0]), "tile_y0": int(tile_origin[1]), "tile_x0": int(tile_origin[2]),
                        }
                    )
                if qc_top_n > 0:
                    batch_qc_rows.append(
                        _tile_qc_row(
                            tile_index=tile_index, batch_index=batch_index,
                            slices=slices, shape=shape, tile=tile,
                            point_count=point_count_before_keep_window, details=details,
                            prob_thresh=resolved_prob_thresh, skipped=False,
                        )
                    )
                post_seconds += time.perf_counter() - post_started

        if batch_size > 1 and batched_pending:
            _flush_batched()

        write_started = time.perf_counter()
        _write_points_batch_csv(batch_points, batch_path)
        if qc_top_n > 0:
            _write_tile_qc_batch_csv(batch_qc_rows, batch_qc_path)
        write_seconds = time.perf_counter() - write_started
        _append_timing_row(
            timing_csv,
            {
                "batch_index": batch_index,
                "start_tile": start_idx,
                "end_tile": end_idx,
                "tiles": end_idx - start_idx,
                "skipped_tiles": batch_skipped_tiles,
                "points": len(batch_points),
                "read_seconds": f"{read_seconds:.3f}",
                "predict_seconds": f"{predict_seconds:.3f}",
                "post_seconds": f"{post_seconds:.3f}",
                "write_seconds": f"{write_seconds:.3f}",
                "total_seconds": f"{time.perf_counter() - batch_started:.3f}",
            },
        )

    points = _merge_batch_csvs(batch_paths, output_path)
    qc_rows: list[dict[str, Any]] = []
    top_qc_rows: list[dict[str, Any]] = []
    if qc_top_n > 0:
        qc_rows = _merge_tile_qc_csvs(batch_qc_paths, qc_tile_path)
        top_qc_rows = _select_top_uncertain_tile_rows(qc_rows, qc_top_n)
        if top_qc_rows:
            top_qc_rows = _export_tile_qc_previews(
                top_qc_rows,
                data_in=data_in,
                preview_dir=qc_preview_path,
                mode=qc_preview_mode,
            )
        qc_top_path = _write_top_tile_csv(top_qc_rows, qc_top_path)
    region_rows = _count_points_by_region(points, region_lookup) if label_data is not None else []
    if label_data is not None:
        region_counts_path = _write_region_counts_csv(region_rows, region_counts_path)

    summary = {
        "success": True,
        "input_zarr": str(input_path),
        "model_dir": str(model_path),
        "label_zarr": str(label_path) if label_path else "",
        "output_csv": str(output_path),
        "region_counts_csv": str(region_counts_path) if label_data is not None else "",
        "summary_json": str(summary_path),
        "dataset_name": dataset_name,
        "which": which,
        "prob_thresh": prob_thresh,
        "resolved_prob_thresh": resolved_prob_thresh,
        "min_distance": min_distance,
        "tile_size": list(resolved_tile_size),
        "tile_overlap": int(tile_overlap),
        "shape": list(shape),
        "total_signal_count": len(points),
        "region_count_rows": len(region_rows),
        "skipped_tiles": skipped_tiles,
        "processed_tiles": len(tile_slices) - skipped_tiles,
        "checkpoint_dir": str(checkpoint_dir),
        "checkpoint_tiles": checkpoint_tiles,
        "batch_size": batch_size,
        "batch_timing_csv": str(timing_csv),
        "qc_tile_csv": str(qc_tile_path) if qc_top_n > 0 else "",
        "qc_top_csv": str(qc_top_path) if qc_top_n > 0 else "",
        "qc_preview_dir": str(qc_preview_path) if qc_top_n > 0 else "",
        "qc_top_n": qc_top_n,
        "qc_preview_mode": qc_preview_mode if qc_top_n > 0 else "",
        "device": device,
    }
    summary_path = _write_summary_json(summary, summary_path)

    manifest_outputs = [output_path, summary_path]
    if label_data is not None:
        manifest_outputs.append(region_counts_path)

    manifest_path = write_run_manifest(
        output_path.parent,
        module="segmentation",
        entrypoint="run_spotiflow_inference",
        inputs={
            "input_zarr": str(input_path),
            "model_dir": str(model_path),
            "label_zarr": str(label_path) if label_path else "",
            "cfg": str(cfg),
            "dataset_name": dataset_name,
            "which": which,
            "prob_thresh": prob_thresh,
            "min_distance": min_distance,
            "tile_size": resolved_tile_size,
            "tile_overlap": tile_overlap,
            "skip_below_threshold": skip_below_threshold,
            "device": device,
            "checkpoint_tiles": checkpoint_tiles,
            "checkpoint_dir": str(checkpoint_dir),
            "qc_top_n": qc_top_n,
            "qc_preview_mode": qc_preview_mode,
        },
        outputs=manifest_outputs,
        started_at=started_at,
        extra=summary,
    )
    summary["manifest_path"] = str(manifest_path)
    _write_summary_json(summary, summary_path)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Spotiflow 3D spot detection on a whole-brain Zarr volume")
    parser.add_argument("--input_zarr", required=True, help="Input signal Zarr path")
    parser.add_argument("--output_csv", required=True, help="Output CSV path for detected points")
    parser.add_argument("--model_dir", default=str(DEFAULT_MODEL_DIR), help="Spotiflow model folder")
    parser.add_argument("--label_zarr", default="", help="Optional registered atlas label Zarr for region counts")
    parser.add_argument("--region_counts_csv", default="", help="Optional output CSV path for per-region counts")
    parser.add_argument("--summary_json", default="", help="Optional output JSON summary path")
    parser.add_argument("--cfg", default=str(DEFAULT_CFG), help="Region CSV path")
    parser.add_argument("--dataset_name", default="0", help="Dataset name inside Zarr groups")
    parser.add_argument("--which", choices=["best", "last"], default="best", help="Spotiflow checkpoint name")
    parser.add_argument("--prob_thresh", type=float, default=None, help="Override probability threshold")
    parser.add_argument("--min_distance", type=int, default=1, help="Minimum distance between detected spots")
    parser.add_argument("--tile_size", default="", help="Tile size as z,y,x. Defaults to input Zarr chunks")
    parser.add_argument("--tile_overlap", type=int, default=16, help="Overlap voxels per tile side for de-duplication")
    parser.add_argument("--skip_below_threshold", type=float, default=100.0, help="Skip tiles below raw max intensity threshold; use a negative value to disable")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--peak_mode", choices=["fast", "skimage"], default="fast")
    parser.add_argument("--normalizer", default="auto", help="Spotiflow normalizer; use none to disable")
    parser.add_argument("--subpix", choices=["auto", "true", "false"], default="auto")
    parser.add_argument("--use_tuned_tile_overlap", action="store_true")
    parser.add_argument("--checkpoint_tiles", type=int, default=DEFAULT_CHECKPOINT_TILES, help="Number of tiles per resumable checkpoint batch")
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE, help="Number of tiles processed in a single GPU forward pass; higher values improve GPU utilization")
    parser.add_argument("--qc_top_n", type=int, default=0, help="Write tile-level QC and export top-N uncertain tile previews")
    parser.add_argument("--qc_tile_csv", default="", help="Optional output CSV path for all tile-level Spotiflow QC rows")
    parser.add_argument("--qc_top_csv", default="", help="Optional output CSV path for top uncertain Spotiflow tile rows")
    parser.add_argument("--qc_preview_dir", default="", help="Optional output directory for top uncertain tile previews")
    parser.add_argument("--qc_preview_mode", choices=["mip", "volume"], default="mip", help="Save MIP preview stack or full tile volume")
    parser.add_argument("--json_logs", action="store_true")
    return parser.parse_args()


def _parse_subpix(value: str) -> bool | None:
    text = str(value).strip().lower()
    if text == "auto":
        return None
    return text == "true"


def main() -> int:
    args = parse_args()
    _configure_logging(args.json_logs)
    try:
        result = run_spotiflow_inference(
            input_zarr=args.input_zarr,
            output_csv=args.output_csv,
            model_dir=args.model_dir,
            label_zarr=args.label_zarr or None,
            region_counts_csv=args.region_counts_csv or None,
            summary_json=args.summary_json or None,
            cfg=args.cfg,
            dataset_name=args.dataset_name,
            which=args.which,
            prob_thresh=args.prob_thresh,
            min_distance=args.min_distance,
            tile_size=_parse_triplet(args.tile_size),
            tile_overlap=args.tile_overlap,
            skip_below_threshold=None if args.skip_below_threshold < 0 else args.skip_below_threshold,
            device=args.device,
            peak_mode=args.peak_mode,
            normalizer=None if str(args.normalizer).lower() == "none" else args.normalizer,
            subpix=_parse_subpix(args.subpix),
            use_tuned_tile_overlap=args.use_tuned_tile_overlap,
            checkpoint_tiles=args.checkpoint_tiles,
            batch_size=args.batch_size,
            qc_top_n=args.qc_top_n,
            qc_tile_csv=args.qc_tile_csv or None,
            qc_top_csv=args.qc_top_csv or None,
            qc_preview_dir=args.qc_preview_dir or None,
            qc_preview_mode=args.qc_preview_mode,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0
    except PipelineError as exc:
        print(json.dumps(exc.to_dict(), ensure_ascii=False), file=sys.stderr)
        return exc.exit_code
    except Exception as exc:  # pragma: no cover
        logger.exception("Unhandled Spotiflow inference error: %s", exc)
        wrapped = PipelineError(ErrorCode.INTERNAL_ERROR, "Unhandled Spotiflow inference error", {"error": str(exc)})
        print(json.dumps(wrapped.to_dict(), ensure_ascii=False), file=sys.stderr)
        return wrapped.exit_code


if __name__ == "__main__":
    sys.exit(main())
