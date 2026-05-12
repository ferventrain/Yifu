"""3D tubular enhancement for neural fiber structures using Hessian-based filters.

Uses skimage Frangi / Meijering / Sato filters on 3D Zarr volumes to
enhance tubular structures (neural fibers) while suppressing sheet-like
membrane background and punctate noise.

Example
-------
CLI (standalone)::

    python pipeline_modules/preprocessing/tubular_enhancement.py \\
        --input_zarr ch1.zarr --output_zarr ch1_enhanced.zarr \\
        --method frangi --sigmas "1,2,4,8"

    # With TIFF export
    python pipeline_modules/preprocessing/tubular_enhancement.py \\
        --input_zarr ch1.zarr --output_zarr ch1_enhanced.zarr \\
        --method meijering --sigmas "2,4" --export_tiff ./enhanced_tiff

Python API::

    from pipeline_modules.preprocessing.tubular_enhancement import (
        enhance_tubular_zarr, export_to_tiff
    )
    result = enhance_tubular_zarr(
        "ch1.zarr", "ch1_enhanced.zarr",
        method="frangi", sigmas=[1, 2, 4, 8],
    )
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm as _tqdm

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_array(z) -> Any:
    """Resolve a Zarr group to the single array it wraps."""
    import zarr

    if isinstance(z, zarr.Array):
        return z
    if "0" in z:
        return z["0"]
    arrays = list(z.arrays())
    if len(arrays) == 1:
        return arrays[0][1]
    raise KeyError(
        f"Zarr group has {len(arrays)} arrays; cannot determine which to use"
    )


def _apply_3d_enhancement(
    volume: np.ndarray,
    method: str,
    sigmas: list[float],
    black_ridges: bool,
) -> np.ndarray:
    """Apply a 3D Hessian-based tubular enhancement filter.

    All scikit-image filters return float64 in [0, 1].
    """
    from skimage.filters import frangi, meijering, sato

    if method == "frangi":
        return frangi(
            volume, sigmas=sigmas, black_ridges=black_ridges,
            mode="constant", cval=0,
        )
    if method == "meijering":
        return meijering(
            volume, sigmas=sigmas, black_ridges=black_ridges,
            mode="constant", cval=0,
        )
    if method == "sato":
        return sato(
            volume, sigmas=sigmas, black_ridges=black_ridges,
            mode="constant", cval=0,
        )
    raise ValueError(f"Unknown tubular enhancement method: {method!r}")


def _scale_to_dtype(data: np.ndarray, target_dtype: np.dtype) -> np.ndarray:
    """Scale float [0, 1] filter output back to an integer dtype range."""
    if np.issubdtype(target_dtype, np.floating):
        return data.astype(target_dtype)
    info = np.iinfo(target_dtype)
    return np.clip(data * info.max, 0, info.max).astype(target_dtype)


def _estimate_throughput_mbs(
    shape: tuple[int, ...], dtype: np.dtype, elapsed: float,
) -> float:
    nbytes = float(np.prod(shape)) * dtype.itemsize
    return nbytes / (1024 * 1024) / max(elapsed, 1e-9)


def _normalize_3d_size(
    value: int | str | tuple[int, int, int],
    *,
    name: str,
) -> tuple[int, int, int]:
    """Normalize a scalar or comma-separated value into a 3D size tuple."""
    if isinstance(value, tuple):
        sizes = value
    elif isinstance(value, int):
        sizes = (value, value, value)
    else:
        parts = [part.strip() for part in str(value).split(",") if part.strip()]
        if len(parts) == 1:
            scalar = int(parts[0])
            sizes = (scalar, scalar, scalar)
        elif len(parts) == 3:
            sizes = tuple(int(part) for part in parts)
        else:
            raise ValueError(
                f"{name} must be an integer or three comma-separated integers"
            )

    if any(size <= 0 for size in sizes):
        raise ValueError(f"{name} values must all be positive")
    return sizes


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def enhance_tubular_zarr(
    input_zarr: str | Path,
    output_zarr: str | Path,
    *,
    method: str = "frangi",
    sigmas: list[float] | None = None,
    black_ridges: bool = False,
    slab_depth: int = 32,
    tile_size: int | tuple[int, int, int] = (256, 256, 256),
    overlap: int | None = None,
    output_dtype: str | None = None,
) -> dict[str, Any]:
    """Apply 3D tubular enhancement to a Zarr volume, writing an enhanced copy.

    Parameters
    ----------
    input_zarr:
        Path to the input Zarr store (3D, any integer / float dtype).
    output_zarr:
        Path for the output Zarr store (will be created / overwritten).
    method:
        ``"frangi"`` (default), ``"meijering"``, or ``"sato"``.
    sigmas:
        Scales (pixel radii) for the multiscale Hessian analysis.
        Default: ``[1.0, 2.0, 4.0, 8.0]``.
    black_ridges:
        If True, enhance dark tubular structures on a bright background.
        Default (False): enhance bright structures on dark background.
    slab_depth:
        Deprecated compatibility alias for Z-only slab processing. Retained
        for older callers; ignored when ``tile_size`` is provided.
    tile_size:
        3D processing tile size ``(z, y, x)``. Each tile is processed with
        halo overlap on all sides, and only its center region is written back.
    overlap:
        Halo overlap between adjacent tiles. Auto-derived if *None*.
    output_dtype:
        Dtype of the output Zarr.  Default: same as the input dtype.

    Returns
    -------
    dict
        Summary with keys ``success``, ``shape``, ``duration_seconds``, etc.
    """
    if sigmas is None:
        sigmas = [1.0, 2.0, 4.0, 8.0]

    import zarr

    started_at = time.time()
    input_path = Path(input_zarr)
    output_path = Path(output_zarr)

    z_in = zarr.open(str(input_path), mode="r")
    arr_in = _get_array(z_in)
    shape = arr_in.shape
    dtype_in = arr_in.dtype

    if len(shape) != 3:
        raise ValueError(f"Expected a 3D Zarr array, got shape {shape}")

    if overlap is None:
        overlap = int(max(sigmas) * 3) + 1
    if tile_size is None:
        tile_size = (slab_depth, shape[1], shape[2])
    tile_size = _normalize_3d_size(tile_size, name="tile_size")

    dtype_out = np.dtype(output_dtype) if output_dtype else dtype_in

    # Inherit chunking but cap each dimension to the processing tile size.
    in_chunks = tuple(min(chunk, tile) for chunk, tile in zip(arr_in.chunks, tile_size))

    zarr.open(
        str(output_path),
        mode="w",
        shape=shape,
        chunks=in_chunks,
        dtype=dtype_out,
    )
    z_out = zarr.open(str(output_path), mode="a")
    out_arr = _get_array(z_out)

    grid_shape = tuple(
        max(1, math.ceil(axis_size / axis_tile))
        for axis_size, axis_tile in zip(shape, tile_size)
    )
    total_tiles = int(np.prod(grid_shape))

    logger.info(
        "Tubular enhancement: method=%s, sigmas=%s, shape=%s, "
        "tile_size=%s, overlap=%d, tiles=%d, %s -> %s",
        method, sigmas, shape, tile_size, overlap, total_tiles,
        dtype_in, dtype_out,
    )

    for tile_idx in _tqdm(
        np.ndindex(grid_shape),
        desc="Tubular enhancement",
        unit="tile",
        leave=False,
        total=total_tiles,
        file=sys.stderr,
    ):
        write_starts = tuple(index * tile for index, tile in zip(tile_idx, tile_size))
        write_ends = tuple(
            min(start + tile, axis_size)
            for start, tile, axis_size in zip(write_starts, tile_size, shape)
        )
        read_starts = tuple(max(0, start - overlap) for start in write_starts)
        read_ends = tuple(
            min(axis_size, end + overlap)
            for end, axis_size in zip(write_ends, shape)
        )

        read_slices = tuple(slice(start, end) for start, end in zip(read_starts, read_ends))
        write_slices = tuple(slice(start, end) for start, end in zip(write_starts, write_ends))
        inner_slices = tuple(
            slice(write_start - read_start, write_end - read_start)
            for write_start, write_end, read_start in zip(write_starts, write_ends, read_starts)
        )

        tile = arr_in[read_slices]
        enhanced = _apply_3d_enhancement(tile, method, sigmas, black_ridges)
        out_arr[write_slices] = _scale_to_dtype(enhanced[inner_slices], dtype_out)

    duration = time.time() - started_at
    mbs = _estimate_throughput_mbs(shape, dtype_in, duration)
    logger.info("Tubular enhancement done: %.1f s  (%.1f MB/s)", duration, mbs)

    return {
        "success": True,
        "input": str(input_path),
        "output": str(output_path),
        "method": method,
        "sigmas": sigmas,
        "black_ridges": black_ridges,
        "shape": list(shape),
        "dtype_in": str(dtype_in),
        "dtype_out": str(dtype_out),
        "duration_seconds": duration,
    }


def export_to_tiff(
    zarr_path: str | Path,
    output_dir: str | Path,
    *,
    slice_axis: int = 0,
    dtype: str | None = None,
) -> Path:
    """Export a 3D Zarr volume to a directory of TIFF slices.

    Parameters
    ----------
    zarr_path:
        Path to the input Zarr store (3D).
    output_dir:
        Directory to write TIFF slices into (created if missing).
    slice_axis:
        Axis to slice along (0 = Z, default; 2 = last axis).
    dtype:
        Optional output dtype override (e.g. ``"uint16"``).

    Returns
    -------
    Path
        The output directory.
    """
    import zarr
    import tifffile

    zarr_path = Path(zarr_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    z = zarr.open(str(zarr_path), mode="r")
    arr = _get_array(z)
    shape = arr.shape

    axis = slice_axis
    n = shape[axis]
    for idx in range(n):
        sl = (idx, ...) if axis == 0 else (..., idx) if axis == 2 else ...
        slice_data = arr[sl]
        if dtype:
            slice_data = slice_data.astype(np.dtype(dtype))
        tiff_path = output_dir / f"slice_{idx:04d}.tif"
        tifffile.imwrite(str(tiff_path), slice_data)

    return output_dir


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="3D tubular enhancement for neural fiber structures",
    )
    parser.add_argument("--input_zarr", required=True)
    parser.add_argument("--output_zarr", required=True)
    parser.add_argument(
        "--method", default="frangi",
        choices=["frangi", "meijering", "sato"],
        help="Hessian-based enhancement method (default: frangi)",
    )
    parser.add_argument(
        "--sigmas", default="1,2,4,8",
        help="Comma-separated sigma values (default: 1,2,4,8)",
    )
    parser.add_argument(
        "--black_ridges", action="store_true",
        help="Enhance dark ridges on bright background (default: bright ridges)",
    )
    parser.add_argument(
        "--slab_depth", type=int, default=32,
        help="Deprecated compatibility alias for old Z-only processing",
    )
    parser.add_argument(
        "--tile_size", default="256,256,256",
        help="3D processing tile size as N or z,y,x (default: 256,256,256)",
    )
    parser.add_argument(
        "--output_dtype",
        help='Output dtype, e.g. "uint16" (default: same as input)',
    )
    parser.add_argument(
        "--export_tiff",
        help="Export result as TIFF slices to this directory",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    sigmas = [float(s.strip()) for s in args.sigmas.split(",") if s.strip()]

    result = enhance_tubular_zarr(
        args.input_zarr,
        args.output_zarr,
        method=args.method,
        sigmas=sigmas,
        black_ridges=args.black_ridges,
        slab_depth=args.slab_depth,
        tile_size=_normalize_3d_size(args.tile_size, name="tile_size"),
        output_dtype=args.output_dtype,
    )

    if args.export_tiff:
        export_to_tiff(args.output_zarr, args.export_tiff)
        result["tiff_export"] = str(args.export_tiff)

    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["success"] else 1


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    sys.exit(main())
