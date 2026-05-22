"""Warp a sample-space mask Zarr into atlas space and export brainrender points."""

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
    import ants
except ModuleNotFoundError as exc:  # pragma: no cover
    raise ModuleNotFoundError("ANTsPy is required. Run this in the registration/napari environment.") from exc

try:
    from brainglobe_atlasapi.bg_atlas import BrainGlobeAtlas
except ModuleNotFoundError:  # pragma: no cover
    BrainGlobeAtlas = None  # type: ignore[assignment]

try:
    from pipeline_modules.segmentation.zarr_utils import open_zarr_dataset
except ImportError:  # pragma: no cover
    from ..segmentation.zarr_utils import open_zarr_dataset


logger = logging.getLogger(__name__)


@dataclass
class BinAccumulator:
    count: int = 0


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


def default_output_path(mask_zarr_path: Path) -> Path:
    return mask_zarr_path.parent / "visualization" / "points.csv"


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
        path = Path(sample_dir) / f"{signal_ch}_mask.zarr"
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


def resolve_sample_reference_nii(
    *,
    sample_dir: str | Path | None,
    register_ch: str,
    sample_reference_nii: str | Path | None,
) -> Path:
    if sample_reference_nii:
        path = Path(sample_reference_nii)
    else:
        if not sample_dir:
            raise ValueError("Either --sample_dir or --sample_reference_nii is required.")
        path = Path(sample_dir) / f"{register_ch}_downsample" / "volume.nii.gz"
    if not path.exists():
        raise FileNotFoundError(f"Sample reference NIfTI not found: {path}")
    return path


def resolve_inverse_transforms(transforms_dir: str | Path, transforms: str) -> list[str]:
    if transforms.strip():
        paths = [Path(part.strip()) for part in transforms.split(",") if part.strip()]
    else:
        root = Path(transforms_dir)
        if not root.exists():
            raise FileNotFoundError(f"Transforms directory not found: {root}")
        paths = sorted(root.glob("inv_*"), key=lambda path: path.name)
    if not paths:
        raise FileNotFoundError("No inverse transforms found. Pass --transforms or check --transforms_dir.")
    missing = [path for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Transform file(s) not found: {missing}")
    return [str(path) for path in paths]


def accumulate_sample_grid(
    mask_zarr_path: str | Path,
    *,
    resolution_xyz: tuple[float, float, float],
    target_resolution_xyz: tuple[float, float, float],
    output_shape_zyx: tuple[int, int, int],
    dataset_name: str,
    foreground_mode: str,
    foreground_label: int,
    block_shape: tuple[int, int, int],
    min_voxels_per_point: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    arr = open_zarr_dataset(mask_zarr_path, dataset_name=dataset_name)
    shape = tuple(int(v) for v in arr.shape)
    if len(shape) != 3:
        raise ValueError(f"Expected a 3D mask Zarr, got shape: {shape}")

    res_x, res_y, res_z = resolution_xyz
    target_x, target_y, target_z = target_resolution_xyz
    grid_z, grid_y, grid_x = output_shape_zyx
    key_stride_y = grid_x
    key_stride_z = grid_y * grid_x

    accumulators: dict[int, BinAccumulator] = {}
    foreground_voxels = 0
    clipped_voxels = 0
    blocks = list(iter_block_slices(shape, block_shape))

    for z_slice, y_slice, x_slice in tqdm(blocks, desc="Binning sample mask", unit="block", leave=False, file=sys.stderr):
        block = np.asarray(arr[z_slice, y_slice, x_slice])
        fg = foreground_mask(block, mode=foreground_mode, label=foreground_label)
        if not fg.any():
            continue

        local_z, local_y, local_x = np.nonzero(fg)
        foreground_voxels += int(local_z.size)
        z_indices = local_z + z_slice.start
        y_indices = local_y + y_slice.start
        x_indices = local_x + x_slice.start

        gx = np.floor(((x_indices.astype(np.float64) + 0.5) * res_x) / target_x).astype(np.int64)
        gy = np.floor(((y_indices.astype(np.float64) + 0.5) * res_y) / target_y).astype(np.int64)
        gz = np.floor(((z_indices.astype(np.float64) + 0.5) * res_z) / target_z).astype(np.int64)
        in_bounds = (gx >= 0) & (gx < grid_x) & (gy >= 0) & (gy < grid_y) & (gz >= 0) & (gz < grid_z)
        clipped_voxels += int(np.count_nonzero(~in_bounds))
        if not np.any(in_bounds):
            continue

        flat_keys = gz[in_bounds] * key_stride_z + gy[in_bounds] * key_stride_y + gx[in_bounds]
        unique_keys, inverse = np.unique(flat_keys, return_inverse=True)
        counts = np.bincount(inverse)
        for idx, raw_key in enumerate(unique_keys):
            key = int(raw_key)
            accumulator = accumulators.get(key)
            if accumulator is None:
                accumulator = BinAccumulator()
                accumulators[key] = accumulator
            accumulator.count += int(counts[idx])

    volume = np.zeros(output_shape_zyx, dtype=np.uint8)
    occupied_bins = 0
    for flat_key, accumulator in accumulators.items():
        if accumulator.count < min_voxels_per_point:
            continue
        gz = flat_key // key_stride_z
        rem = flat_key % key_stride_z
        gy = rem // key_stride_y
        gx = rem % key_stride_y
        volume[int(gz), int(gy), int(gx)] = 1
        occupied_bins += 1

    summary = {
        "mask_zarr": str(mask_zarr_path),
        "shape_zyx": list(shape),
        "sample_grid_shape_zyx": list(output_shape_zyx),
        "resolution_xyz": list(resolution_xyz),
        "target_resolution_xyz": list(target_resolution_xyz),
        "foreground_mode": foreground_mode,
        "foreground_label": foreground_label,
        "foreground_voxels": foreground_voxels,
        "clipped_voxels": clipped_voxels,
        "occupied_sample_bins": occupied_bins,
        "min_voxels_per_point": min_voxels_per_point,
    }
    return volume, summary


def warp_sample_grid_to_atlas(
    sample_volume_zyx: np.ndarray,
    *,
    sample_reference_nii: str | Path,
    atlas_image: str | Path,
    transformlist: list[str],
) -> tuple[np.ndarray, tuple[float, float, float]]:
    sample_ref = ants.image_read(str(sample_reference_nii))
    atlas_ref = ants.image_read(str(atlas_image))
    identity = np.eye(3)
    sample_ref.set_direction(identity)
    atlas_ref.set_direction(identity)

    moving = ants.from_numpy(
        np.transpose(sample_volume_zyx.astype(np.float32), (2, 1, 0)),
        origin=sample_ref.origin,
        spacing=sample_ref.spacing,
        direction=identity,
    )
    warped = ants.apply_transforms(
        fixed=atlas_ref,
        moving=moving,
        transformlist=transformlist,
        interpolator="nearestNeighbor",
    )
    atlas_volume_zyx = np.transpose(warped.numpy(), (2, 1, 0)) > 0
    atlas_resolution_xyz = tuple(float(v) for v in atlas_ref.spacing)
    return atlas_volume_zyx.astype(np.uint8), atlas_resolution_xyz


def resolve_atlas_resolution_xyz(atlas_name: str, atlas_resolution_xyz: str) -> tuple[float, float, float]:
    if atlas_resolution_xyz.strip():
        return parse_triplet(atlas_resolution_xyz, name="atlas_resolution_xyz")
    if atlas_name:
        if BrainGlobeAtlas is None:
            raise ModuleNotFoundError(
                "brainglobe-atlasapi is required to resolve --atlas_name. "
                "Pass --atlas_resolution_xyz explicitly instead."
            )
        atlas = BrainGlobeAtlas(atlas_name, check_latest=False)
        resolution = tuple(float(value) for value in atlas.resolution)
        return resolution[0], resolution[1], resolution[2]
    return (25.0, 25.0, 25.0)


def atlas_volume_to_points(
    atlas_volume_zyx: np.ndarray,
    *,
    atlas_resolution_xyz: tuple[float, float, float],
    max_points: int,
) -> pd.DataFrame:
    z_idx, y_idx, x_idx = np.nonzero(atlas_volume_zyx)
    # ANTs/NIfTI data arrive here as array axes (DV, AP, ML) after converting
    # from ANTs' (ML, AP, DV) numpy order. brainrender/BrainGlobe point
    # coordinates are expected as (AP, DV, ML), exposed as CSV x,y,z below.
    ap_idx = y_idx
    dv_idx = z_idx
    ml_idx = x_idx
    table = pd.DataFrame(
        {
            "x": (ap_idx.astype(np.float64) + 0.5) * atlas_resolution_xyz[0],
            "y": (dv_idx.astype(np.float64) + 0.5) * atlas_resolution_xyz[1],
            "z": (ml_idx.astype(np.float64) + 0.5) * atlas_resolution_xyz[2],
            "grid_x": ap_idx.astype(np.int64),
            "grid_y": dv_idx.astype(np.int64),
            "grid_z": ml_idx.astype(np.int64),
            "voxel_count": np.ones_like(x_idx, dtype=np.int64),
            "signal_volume_um3": np.full_like(x_idx, np.prod(atlas_resolution_xyz), dtype=np.float64),
            "coordinate_space": "atlas",
        }
    )
    if max_points and max_points > 0 and len(table) > max_points:
        table = table.sample(n=int(max_points), random_state=0).sort_values(["grid_z", "grid_y", "grid_x"])
    return table.reset_index(drop=True)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Warp sample-space mask Zarr into atlas space and export points.")
    parser.add_argument("--sample_dir", default="", help="Sample root directory.")
    parser.add_argument("--signal_ch", default="ch0", help="Signal channel label, e.g. ch1.")
    parser.add_argument("--register_ch", default="ch0", help="Registration channel label, e.g. ch0.")
    parser.add_argument("--mask_zarr", default="", help="Explicit mask Zarr path. Overrides --sample_dir/--signal_ch.")
    parser.add_argument("--dataset_name", default="0")
    parser.add_argument("--sample_reference_nii", default="", help="Downsampled sample NIfTI used for registration.")
    parser.add_argument("--atlas_image", required=True, help="Atlas image NIfTI used in registration.")
    parser.add_argument("--atlas_name", default="allen_mouse_25um", help="BrainGlobe atlas name used to resolve output point scale.")
    parser.add_argument(
        "--atlas_resolution_xyz",
        default="",
        help="Output atlas voxel size in microns as x,y,z. Defaults to --atlas_name resolution.",
    )
    parser.add_argument("--transforms_dir", default="", help="Directory containing inv_* transforms.")
    parser.add_argument("--transforms", default="", help="Comma-separated inverse transform paths in ANTs order.")
    parser.add_argument("--resolution_xyz", required=True, help="Input mask voxel size in microns as x,y,z.")
    parser.add_argument("--target_resolution_xyz", default="25,25,25", help="Sample grid size in microns as x,y,z.")
    parser.add_argument("--foreground_mode", choices=("nonzero", "equal"), default="nonzero")
    parser.add_argument("--foreground_label", type=int, default=1)
    parser.add_argument("--block_shape", default="", help="Optional block shape in z,y,x order.")
    parser.add_argument("--min_voxels_per_point", type=int, default=1)
    parser.add_argument("--max_points", type=int, default=150_000, help="Randomly keep N atlas voxels; 0 disables cap.")
    parser.add_argument("--output", default="", help="Output atlas-space CSV path.")
    return parser


def main() -> int:
    configure_logging()
    started_at = time.time()
    args = build_argparser().parse_args()

    sample_dir = args.sample_dir or None
    mask_zarr_path = resolve_mask_zarr(sample_dir=sample_dir, signal_ch=args.signal_ch, mask_zarr=args.mask_zarr or None)
    sample_reference_nii = resolve_sample_reference_nii(
        sample_dir=sample_dir,
        register_ch=args.register_ch,
        sample_reference_nii=args.sample_reference_nii or None,
    )
    transforms_dir = args.transforms_dir or str(Path(sample_reference_nii).parents[1] / "transforms")
    transformlist = resolve_inverse_transforms(transforms_dir, args.transforms)

    sample_ref = ants.image_read(str(sample_reference_nii))
    sample_shape_zyx = tuple(int(v) for v in sample_ref.shape[::-1])
    resolution_xyz = parse_triplet(args.resolution_xyz, name="resolution_xyz")
    target_resolution_xyz = parse_triplet(args.target_resolution_xyz, name="target_resolution_xyz")

    arr = open_zarr_dataset(mask_zarr_path, dataset_name=args.dataset_name)
    fallback_block_shape = tuple(int(v) for v in (getattr(arr, "chunks", None) or arr.shape))
    block_shape = parse_block_shape(args.block_shape, fallback_block_shape)

    sample_volume, summary = accumulate_sample_grid(
        mask_zarr_path,
        resolution_xyz=resolution_xyz,
        target_resolution_xyz=target_resolution_xyz,
        output_shape_zyx=sample_shape_zyx,
        dataset_name=args.dataset_name,
        foreground_mode=args.foreground_mode,
        foreground_label=args.foreground_label,
        block_shape=block_shape,
        min_voxels_per_point=args.min_voxels_per_point,
    )
    atlas_volume, raw_atlas_spacing_xyz = warp_sample_grid_to_atlas(
        sample_volume,
        sample_reference_nii=sample_reference_nii,
        atlas_image=args.atlas_image,
        transformlist=transformlist,
    )
    atlas_resolution_xyz = resolve_atlas_resolution_xyz(args.atlas_name, args.atlas_resolution_xyz)
    table = atlas_volume_to_points(atlas_volume, atlas_resolution_xyz=atlas_resolution_xyz, max_points=args.max_points)

    summary.update(
        {
            "success": True,
            "sample_reference_nii": str(sample_reference_nii),
            "atlas_image": str(args.atlas_image),
            "atlas_name": args.atlas_name,
            "transformlist": transformlist,
            "atlas_shape_zyx": list(atlas_volume.shape),
            "raw_atlas_image_spacing_xyz": list(raw_atlas_spacing_xyz),
            "atlas_resolution_xyz": list(atlas_resolution_xyz),
            "exported_points": int(len(table)),
            "max_points": int(args.max_points),
            "coordinate_space": "atlas",
            "duration_seconds": time.time() - started_at,
        }
    )

    output_csv = Path(args.output) if args.output else default_output_path(Path(mask_zarr_path))
    outputs = write_outputs(table, summary, output_csv)
    logger.info("Saved %d atlas-space points to %s", len(table), outputs["csv"])
    logger.info("Saved summary to %s", outputs["summary"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
