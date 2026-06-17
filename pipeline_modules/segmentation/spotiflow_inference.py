from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
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
    skip_below_threshold: float | None = None,
    device: str = "auto",
    peak_mode: str = "fast",
    normalizer: str | None = "auto",
    subpix: bool | None = None,
    use_tuned_tile_overlap: bool = False,
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
    chunk_shape = tuple(int(v) for v in getattr(data_in, "chunks", shape))
    resolved_tile_size = tile_size or chunk_shape
    tile_slices = list(_iter_tile_slices(shape, resolved_tile_size, max(0, int(tile_overlap))))

    points: list[dict[str, Any]] = []
    skipped_tiles = 0
    for slices in tqdm(tile_slices, desc="Spotiflow inference", unit="tile"):
        tile = np.asarray(data_in[slices])
        if skip_below_threshold is not None and float(tile.max(initial=0)) < float(skip_below_threshold):
            skipped_tiles += 1
            continue

        predicted_points, _details = model.predict(
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
        predicted_points = np.asarray(predicted_points)
        if predicted_points.size == 0:
            continue
        if predicted_points.ndim == 1:
            predicted_points = predicted_points.reshape(1, -1)

        tile_origin = np.array([int(slc.start or 0) for slc in slices], dtype=np.float64)
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
            if label_data is not None:
                region_id = int(label_data[tuple(rounded.tolist())])
                region_info = region_lookup.get(region_id)
                if region_info:
                    region_name = region_info["region_name"]
                    region_acronym = region_info["region_acronym"]

            points.append(
                {
                    "z": float(global_point[0]),
                    "y": float(global_point[1]),
                    "x": float(global_point[2]),
                    "region_id": region_id,
                    "region_name": region_name,
                    "region_acronym": region_acronym,
                    "tile_z0": int(tile_origin[0]),
                    "tile_y0": int(tile_origin[1]),
                    "tile_x0": int(tile_origin[2]),
                }
            )

    output_path = _write_points_csv(points, output_path)
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
        "min_distance": min_distance,
        "tile_size": list(resolved_tile_size),
        "tile_overlap": int(tile_overlap),
        "shape": list(shape),
        "total_signal_count": len(points),
        "region_count_rows": len(region_rows),
        "skipped_tiles": skipped_tiles,
        "processed_tiles": len(tile_slices) - skipped_tiles,
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
    parser.add_argument("--skip_below_threshold", type=float, default=None, help="Skip tiles below raw max intensity threshold")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--peak_mode", choices=["fast", "skimage"], default="fast")
    parser.add_argument("--normalizer", default="auto", help="Spotiflow normalizer; use none to disable")
    parser.add_argument("--subpix", choices=["auto", "true", "false"], default="auto")
    parser.add_argument("--use_tuned_tile_overlap", action="store_true")
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
            skip_below_threshold=args.skip_below_threshold,
            device=args.device,
            peak_mode=args.peak_mode,
            normalizer=None if str(args.normalizer).lower() == "none" else args.normalizer,
            subpix=_parse_subpix(args.subpix),
            use_tuned_tile_overlap=args.use_tuned_tile_overlap,
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
