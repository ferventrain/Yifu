"""Convert a binary signal mask Zarr into a 25 um point-cloud CSV for brainrender."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    from pipeline_modules.segmentation.zarr_utils import open_zarr_dataset
    from pipeline_modules.utils.sample_layout import SampleLayout
except ImportError:  # pragma: no cover
    from ..segmentation.zarr_utils import open_zarr_dataset
    from ..utils.sample_layout import SampleLayout


logger = logging.getLogger(__name__)


@dataclass
class BinAccumulator:
    count: int = 0
    x_sum: float = 0.0
    y_sum: float = 0.0
    z_sum: float = 0.0


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


def parse_triplet(value: str | tuple[float, float, float], *, name: str) -> tuple[float, float, float]:
    if isinstance(value, tuple):
        parts = value
    else:
        raw_parts = [part.strip() for part in str(value).split(",") if part.strip()]
        if len(raw_parts) != 3:
            raise ValueError(f"{name} must have 3 comma-separated values in x,y,z order, got: {value}")
        parts = tuple(float(part) for part in raw_parts)

    if any(part <= 0 for part in parts):
        raise ValueError(f"{name} values must be positive, got: {parts}")
    return parts[0], parts[1], parts[2]


def parse_block_shape(value: str, fallback: tuple[int, int, int]) -> tuple[int, int, int]:
    if not str(value).strip():
        return tuple(int(v) for v in fallback)
    parts = [part.strip() for part in str(value).split(",") if part.strip()]
    if len(parts) != 3:
        raise ValueError(f"block_shape must have 3 comma-separated values in z,y,x order, got: {value}")
    block_shape = tuple(int(part) for part in parts)
    if any(size <= 0 for size in block_shape):
        raise ValueError(f"block_shape values must be positive, got: {block_shape}")
    return block_shape


def resolve_mask_zarr(
    *,
    sample_dir: str | Path | None,
    signal_ch: str,
    mask_zarr: str | Path | None,
) -> Path:
    if mask_zarr:
        path = Path(mask_zarr)
    else:
        if not sample_dir:
            raise ValueError("Either --sample_dir or --mask_zarr is required.")
        layout = SampleLayout(sample_dir=Path(sample_dir), signal_ch=signal_ch, require_exists=True)
        path = layout.mask_zarr

    if not path.exists():
        raise FileNotFoundError(f"Mask Zarr not found: {path}")
    return path


def iter_block_slices(shape: tuple[int, int, int], block_shape: tuple[int, int, int]):
    z_size, y_size, x_size = block_shape
    for z0 in range(0, shape[0], z_size):
        z1 = min(z0 + z_size, shape[0])
        for y0 in range(0, shape[1], y_size):
            y1 = min(y0 + y_size, shape[1])
            for x0 in range(0, shape[2], x_size):
                x1 = min(x0 + x_size, shape[2])
                yield (slice(z0, z1), slice(y0, y1), slice(x0, x1))


def foreground_mask(block: np.ndarray, *, mode: str, label: int) -> np.ndarray:
    if mode == "equal":
        return block == label
    if mode == "nonzero":
        return block != 0
    raise ValueError(f"Unsupported foreground mode: {mode}")


def _accumulate_unique_bins(
    accumulators: dict[int, BinAccumulator],
    *,
    flat_keys: np.ndarray,
    x_um: np.ndarray,
    y_um: np.ndarray,
    z_um: np.ndarray,
) -> None:
    unique_keys, inverse = np.unique(flat_keys, return_inverse=True)
    counts = np.bincount(inverse)
    x_sums = np.bincount(inverse, weights=x_um)
    y_sums = np.bincount(inverse, weights=y_um)
    z_sums = np.bincount(inverse, weights=z_um)

    for idx, raw_key in enumerate(unique_keys):
        key = int(raw_key)
        accumulator = accumulators.get(key)
        if accumulator is None:
            accumulator = BinAccumulator()
            accumulators[key] = accumulator
        accumulator.count += int(counts[idx])
        accumulator.x_sum += float(x_sums[idx])
        accumulator.y_sum += float(y_sums[idx])
        accumulator.z_sum += float(z_sums[idx])


def mask_zarr_to_points_table(
    mask_zarr_path: str | Path,
    *,
    resolution_xyz: tuple[float, float, float],
    target_resolution_xyz: tuple[float, float, float] = (25.0, 25.0, 25.0),
    dataset_name: str = "0",
    foreground_mode: str = "nonzero",
    foreground_label: int = 1,
    block_shape: tuple[int, int, int] | None = None,
    min_voxels_per_point: int = 1,
    max_points: int = 150_000,
    coordinate_space: str = "sample",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Aggregate foreground voxels into target-resolution physical centroids.

    Input Zarr arrays are interpreted as ``(z, y, x)``. ``resolution_xyz`` and
    ``target_resolution_xyz`` are both in microns and use ``(x, y, z)`` order.
    """

    started_at = time.time()
    arr = open_zarr_dataset(mask_zarr_path, dataset_name=dataset_name)
    shape = tuple(int(v) for v in arr.shape)
    if len(shape) != 3:
        raise ValueError(f"Expected a 3D mask Zarr, got shape: {shape}")

    chunks = getattr(arr, "chunks", None) or shape
    block_shape = block_shape or tuple(int(v) for v in chunks)
    block_shape = tuple(min(int(block_shape[idx]), shape[idx]) for idx in range(3))

    res_x, res_y, res_z = resolution_xyz
    target_x, target_y, target_z = target_resolution_xyz
    grid_shape_zyx = tuple(
        int(np.ceil((shape[idx] * (res_z, res_y, res_x)[idx]) / (target_z, target_y, target_x)[idx]))
        for idx in range(3)
    )
    grid_z, grid_y, grid_x = grid_shape_zyx
    key_stride_y = grid_x
    key_stride_z = grid_y * grid_x

    accumulators: dict[int, BinAccumulator] = {}
    foreground_voxels = 0
    blocks = list(iter_block_slices(shape, block_shape))

    for z_slice, y_slice, x_slice in tqdm(blocks, desc="Binning mask", unit="block", leave=False, file=sys.stderr):
        block = np.asarray(arr[z_slice, y_slice, x_slice])
        fg = foreground_mask(block, mode=foreground_mode, label=foreground_label)
        if not fg.any():
            continue

        local_z, local_y, local_x = np.nonzero(fg)
        foreground_voxels += int(local_z.size)

        z_indices = local_z + z_slice.start
        y_indices = local_y + y_slice.start
        x_indices = local_x + x_slice.start

        x_um = (x_indices.astype(np.float64) + 0.5) * res_x
        y_um = (y_indices.astype(np.float64) + 0.5) * res_y
        z_um = (z_indices.astype(np.float64) + 0.5) * res_z

        gx = np.floor(x_um / target_x).astype(np.int64)
        gy = np.floor(y_um / target_y).astype(np.int64)
        gz = np.floor(z_um / target_z).astype(np.int64)
        flat_keys = gz * key_stride_z + gy * key_stride_y + gx

        _accumulate_unique_bins(
            accumulators,
            flat_keys=flat_keys,
            x_um=x_um,
            y_um=y_um,
            z_um=z_um,
        )

    rows: list[dict[str, Any]] = []
    voxel_volume_um3 = float(res_x * res_y * res_z)
    for flat_key, accumulator in accumulators.items():
        if accumulator.count < min_voxels_per_point:
            continue
        gz = flat_key // key_stride_z
        rem = flat_key % key_stride_z
        gy = rem // key_stride_y
        gx = rem % key_stride_y
        rows.append(
            {
                "x": accumulator.x_sum / accumulator.count,
                "y": accumulator.y_sum / accumulator.count,
                "z": accumulator.z_sum / accumulator.count,
                "grid_x": int(gx),
                "grid_y": int(gy),
                "grid_z": int(gz),
                "voxel_count": int(accumulator.count),
                "signal_volume_um3": accumulator.count * voxel_volume_um3,
                "coordinate_space": coordinate_space,
            }
        )

    table = pd.DataFrame(rows)
    if not table.empty:
        table = table.sort_values("voxel_count", ascending=False, kind="mergesort")
        if max_points and max_points > 0 and len(table) > max_points:
            table = table.head(int(max_points)).copy()
        table = table.sort_values(["grid_z", "grid_y", "grid_x"], kind="mergesort").reset_index(drop=True)

    summary = {
        "success": True,
        "mask_zarr": str(mask_zarr_path),
        "dataset_name": dataset_name,
        "shape_zyx": list(shape),
        "chunks_zyx": list(chunks),
        "block_shape_zyx": list(block_shape),
        "resolution_xyz": list(resolution_xyz),
        "target_resolution_xyz": list(target_resolution_xyz),
        "foreground_mode": foreground_mode,
        "foreground_label": foreground_label,
        "foreground_voxels": int(foreground_voxels),
        "occupied_target_bins": int(len(accumulators)),
        "exported_points": int(len(table)),
        "min_voxels_per_point": int(min_voxels_per_point),
        "max_points": int(max_points),
        "coordinate_space": coordinate_space,
        "duration_seconds": time.time() - started_at,
    }
    return table, summary


def default_output_path(mask_zarr_path: Path) -> Path:
    sample_dir = mask_zarr_path.parent
    stem = mask_zarr_path.name.replace(".zarr", "")
    return sample_dir / f"{stem}_atlas_points.csv"


def write_outputs(table: pd.DataFrame, summary: dict[str, Any], output_csv: str | Path) -> dict[str, Path]:
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_path, index=False)

    summary_path = output_path.with_suffix(".json")
    summary = dict(summary)
    summary["output_csv"] = str(output_path)
    summary["summary_json"] = str(summary_path)
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    return {"csv": output_path, "summary": summary_path}


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Aggregate sample_dir/chX_mask.zarr into a target-resolution point CSV for brainrender."
    )
    parser.add_argument("--sample_dir", default="", help="Sample root directory containing chX_mask.zarr.")
    parser.add_argument("--signal_ch", default="ch0", help="Signal channel label, e.g. ch1.")
    parser.add_argument("--mask_zarr", default="", help="Explicit mask Zarr path. Overrides --sample_dir/--signal_ch.")
    parser.add_argument("--dataset_name", default="0", help="Dataset name inside the Zarr group.")
    parser.add_argument("--resolution_xyz", required=True, help="Input mask voxel size in microns as x,y,z.")
    parser.add_argument("--target_resolution_xyz", default="25,25,25", help="Output grid size in microns as x,y,z.")
    parser.add_argument("--foreground_mode", choices=("nonzero", "equal"), default="nonzero")
    parser.add_argument("--foreground_label", type=int, default=1)
    parser.add_argument("--block_shape", default="", help="Optional block shape in z,y,x order.")
    parser.add_argument("--min_voxels_per_point", type=int, default=1)
    parser.add_argument("--max_points", type=int, default=150_000, help="Keep densest N points; 0 disables the cap.")
    parser.add_argument(
        "--coordinate_space",
        choices=("sample", "atlas"),
        default="sample",
        help="Metadata tag for the output coordinates. Use atlas only when the mask is already in atlas space.",
    )
    parser.add_argument("--output", default="", help="Output CSV path. Defaults next to the mask Zarr.")
    return parser


def main() -> int:
    configure_logging()
    parser = build_argparser()
    args = parser.parse_args()

    resolution_xyz = parse_triplet(args.resolution_xyz, name="resolution_xyz")
    target_resolution_xyz = parse_triplet(args.target_resolution_xyz, name="target_resolution_xyz")
    mask_zarr_path = resolve_mask_zarr(
        sample_dir=args.sample_dir or None,
        signal_ch=args.signal_ch,
        mask_zarr=args.mask_zarr or None,
    )

    arr = open_zarr_dataset(mask_zarr_path, dataset_name=args.dataset_name)
    fallback_block_shape = tuple(int(v) for v in (getattr(arr, "chunks", None) or arr.shape))
    block_shape = parse_block_shape(args.block_shape, fallback_block_shape)

    table, summary = mask_zarr_to_points_table(
        mask_zarr_path,
        resolution_xyz=resolution_xyz,
        target_resolution_xyz=target_resolution_xyz,
        dataset_name=args.dataset_name,
        foreground_mode=args.foreground_mode,
        foreground_label=args.foreground_label,
        block_shape=block_shape,
        min_voxels_per_point=args.min_voxels_per_point,
        max_points=args.max_points,
        coordinate_space=args.coordinate_space,
    )

    output_csv = Path(args.output) if args.output else default_output_path(mask_zarr_path)
    outputs = write_outputs(table, summary, output_csv)
    logger.info("Saved %d points to %s", len(table), outputs["csv"])
    logger.info("Saved summary to %s", outputs["summary"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
