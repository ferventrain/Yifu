"""Hysteresis thresholding for vessel segmentation on 3D Zarr volumes.

Uses dual thresholds: high threshold seeds confident vessel regions,
low threshold extends connectivity. Only low-threshold regions connected
to high-threshold seeds are retained. This bridges weak-signal gaps
without introducing isolated noise.

Optionally applies morphological gap bridging (closing) to fill small
discontinuities in the resulting binary mask.

CLI::

    micromamba run -n yifu python -m pipeline_modules.segmentation.hysteresis_threshold \
        --input_zarr ch2.zarr --output_zarr ch2_mask.zarr \
        --high 800 --low 300

    # With morphological gap bridging
    micromamba run -n yifu python -m pipeline_modules.segmentation.hysteresis_threshold \
        --input_zarr ch2.zarr --output_zarr ch2_mask.zarr \
        --high 800 --low 300 --close_radius 3
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm as _tqdm

logger = logging.getLogger(__name__)


def _get_array(z) -> Any:
    import zarr
    if isinstance(z, zarr.Array):
        return z
    if "0" in z:
        return z["0"]
    arrays = list(z.arrays())
    if len(arrays) == 1:
        return arrays[0][1]
    raise KeyError(f"Zarr group has {len(arrays)} arrays; cannot determine which to use")


def hysteresis_threshold_3d(
    volume: np.ndarray,
    high: float,
    low: float,
) -> np.ndarray:
    """Apply hysteresis thresholding to a 3D volume.

    Voxels above `high` are seeds. Voxels above `low` that are connected
    (26-connectivity) to any seed are kept. Everything else is background.
    """
    from skimage.filters import apply_hysteresis_threshold
    return apply_hysteresis_threshold(volume, low, high)


def morphological_bridge(
    mask: np.ndarray,
    close_radius: int = 3,
    dilate_radius: int = 0,
) -> np.ndarray:
    """Bridge small gaps in a binary mask using morphological closing.

    Parameters
    ----------
    mask : binary 3D array
    close_radius : radius of the ball structuring element for closing
    dilate_radius : if > 0, apply extra dilation before closing then
                    erode back (helps bridge slightly larger gaps)
    """
    from scipy import ndimage as ndi
    from skimage.morphology import ball

    if close_radius <= 0 and dilate_radius <= 0:
        return mask

    result = mask.astype(bool)

    if dilate_radius > 0:
        struct_d = ball(dilate_radius)
        result = ndi.binary_dilation(result, structure=struct_d)

    if close_radius > 0:
        struct_c = ball(close_radius)
        result = ndi.binary_closing(result, structure=struct_c)

    if dilate_radius > 0:
        struct_d = ball(dilate_radius)
        result = ndi.binary_erosion(result, structure=struct_d)

    return result.astype(np.uint8)


def segment_hysteresis_zarr(
    input_zarr: str | Path,
    output_zarr: str | Path,
    *,
    high: float,
    low: float,
    close_radius: int = 0,
    slab_depth: int = 64,
    overlap: int = 0,
) -> dict[str, Any]:
    """Apply hysteresis thresholding to a Zarr volume.

    For volumes that fit in memory, processes the full volume at once
    (required for correct global connectivity). For very large volumes,
    processes in overlapping slabs with cross-boundary merging.

    Parameters
    ----------
    input_zarr : path to input signal Zarr
    output_zarr : path for output binary mask Zarr
    high : high (seed) threshold
    low : low (extension) threshold
    close_radius : morphological closing radius (0 = skip)
    slab_depth : Z-depth per processing slab (only used if volume > 4GB)
    overlap : slab overlap for connectivity (auto-set if 0)
    """
    import zarr

    started_at = time.time()
    input_path = Path(input_zarr)
    output_path = Path(output_zarr)

    z_in = zarr.open(str(input_path), mode="r")
    arr_in = _get_array(z_in)
    shape = arr_in.shape
    dtype_in = arr_in.dtype

    if len(shape) != 3:
        raise ValueError(f"Expected 3D Zarr, got shape {shape}")

    nbytes = int(np.prod(shape)) * dtype_in.itemsize
    use_full_volume = nbytes < 4 * 1024**3  # < 4 GB → load all at once

    chunks = (min(64, shape[0]), min(256, shape[1]), min(256, shape[2]))
    store_out = zarr.DirectoryStore(str(output_path))
    root_out = zarr.group(store=store_out, overwrite=True)
    try:
        from numcodecs import Blosc
        compressor = Blosc(cname="zstd", clevel=5, shuffle=Blosc.SHUFFLE)
    except (ImportError, ModuleNotFoundError):
        compressor = None
    out_arr = root_out.create_dataset("0", shape=shape, chunks=chunks, dtype=np.uint8,
                                       compressor=compressor)
    root_out.attrs["multiscales"] = [{"version": "0.4", "datasets": [{"path": "0"}]}]

    if use_full_volume:
        logger.info("Loading full volume (%d MB) for global hysteresis...",
                    nbytes // (1024 * 1024))
        vol = arr_in[:]

        # Diagnostics: show what each threshold captures
        n_total = vol.size
        n_above_low = int((vol >= low).sum())
        n_above_high = int((vol >= high).sum())
        logger.info("Threshold diagnostics:")
        logger.info("  Volume range: [%d, %d], mean=%.1f", vol.min(), vol.max(), vol.mean())
        logger.info("  Above LOW  (>= %.0f): %d voxels (%.2f%%)",
                    low, n_above_low, 100.0 * n_above_low / n_total)
        logger.info("  Above HIGH (>= %.0f): %d voxels (%.2f%%)",
                    high, n_above_high, 100.0 * n_above_high / n_total)

        if n_above_low / n_total > 0.5:
            logger.warning("WARNING: >50%% of voxels pass the LOW threshold! "
                           "Result will be mostly foreground. Raise --low.")

        mask = hysteresis_threshold_3d(vol, high=high, low=low).astype(np.uint8)
        del vol

        if close_radius > 0:
            logger.info("Applying morphological closing (radius=%d)...", close_radius)
            mask = morphological_bridge(mask, close_radius=close_radius)

        out_arr[:] = mask
        n_fg = int((mask > 0).sum())
        logger.info("Foreground voxels: %d / %d (%.2f%%)",
                    n_fg, mask.size, 100.0 * n_fg / mask.size)
    else:
        if overlap == 0:
            overlap = max(close_radius * 2, 16)

        depth = shape[0]
        total_slabs = max(1, (depth + slab_depth - 1) // slab_depth)
        logger.info("Volume too large for single pass (%d MB). "
                    "Processing %d slabs with overlap=%d. "
                    "Note: connectivity across slab boundaries may be incomplete.",
                    nbytes // (1024 * 1024), total_slabs, overlap)

        for slab_idx in _tqdm(
            range(total_slabs),
            desc="Hysteresis threshold",
            unit="slab",
            leave=False,
            file=sys.stderr,
        ):
            write_start = slab_idx * slab_depth
            write_end = min(write_start + slab_depth, depth)
            read_start = max(0, write_start - overlap)
            read_end = min(depth, write_end + overlap)

            slab = arr_in[read_start:read_end]
            mask_slab = hysteresis_threshold_3d(slab, high=high, low=low).astype(np.uint8)

            if close_radius > 0:
                mask_slab = morphological_bridge(mask_slab, close_radius=close_radius)

            inner_start = write_start - read_start
            inner_end = inner_start + (write_end - write_start)
            out_arr[write_start:write_end] = mask_slab[inner_start:inner_end]

    duration = time.time() - started_at
    logger.info("Hysteresis threshold done: %.1f s", duration)

    return {
        "success": True,
        "input": str(input_path),
        "output": str(output_path),
        "high": high,
        "low": low,
        "close_radius": close_radius,
        "shape": list(shape),
        "full_volume_mode": use_full_volume,
        "duration_seconds": round(duration, 1),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Hysteresis thresholding for vessel segmentation",
    )
    parser.add_argument("--input_zarr", required=True)
    parser.add_argument("--output_zarr", required=True)
    parser.add_argument("--high", type=float, required=True,
                        help="High (seed) threshold")
    parser.add_argument("--low", type=float, required=True,
                        help="Low (extension) threshold")
    parser.add_argument("--close_radius", type=int, default=0,
                        help="Morphological closing ball radius (0=skip, 3=typical)")
    parser.add_argument("--slab_depth", type=int, default=64,
                        help="Z-slab depth for large volumes (default: 64)")
    parser.add_argument("--export_tiff", default=None,
                        help="Export mask as TIFF slices to this directory")
    return parser


def main() -> int:
    args = build_parser().parse_args()

    result = segment_hysteresis_zarr(
        args.input_zarr,
        args.output_zarr,
        high=args.high,
        low=args.low,
        close_radius=args.close_radius,
        slab_depth=args.slab_depth,
    )

    if args.export_tiff:
        import tifffile
        import zarr
        tiff_dir = Path(args.export_tiff)
        tiff_dir.mkdir(parents=True, exist_ok=True)
        z = zarr.open(str(args.output_zarr), mode="r")
        arr = _get_array(z)
        for idx in _tqdm(range(arr.shape[0]), desc="Export TIFF", file=sys.stderr):
            tifffile.imwrite(str(tiff_dir / f"slice_{idx:04d}.tif"), arr[idx])
        result["tiff_export"] = str(tiff_dir)

    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["success"] else 1


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    sys.exit(main())
