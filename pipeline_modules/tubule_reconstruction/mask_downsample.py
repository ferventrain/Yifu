"""Chunked binary-mask downsample with occupancy control.

Max-pool (OR-reduce) keeps 1-voxel capillaries but paints every coarse voxel that
a vessel merely grazes, so measured diameters inflate by about one coarse voxel.
Occupancy thresholding drops those low-fill border voxels while still keeping
axis-aligned filaments whose fill is about ``1 / factor**2``.
"""

from __future__ import annotations

import logging
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import zarr
from tqdm import tqdm

from pipeline_modules.tubule_reconstruction.kimimaro_reconstruction import (
    chunk_index_to_slices,
    iter_all_chunk_indices,
    open_zarr_dataset,
)

logger = logging.getLogger(__name__)

VALID_METHODS = ("max_pool", "occupancy", "majority")
BYTES_PER_GIB = 1024.0 ** 3
# Whole-volume Lee thinning typically keeps ~3 uint8 copies of the array.
LEE_PEAK_COPIES = 3


def default_occupancy_threshold(factor: int) -> float:
    """Fill fraction that still keeps a 1-voxel-wide axis-aligned filament."""
    factor = max(1, int(factor))
    return 1.0 / float(factor * factor)


def downsampled_shape(shape_zyx, factor: int) -> tuple[int, int, int]:
    factor = int(factor)
    if factor < 1:
        raise ValueError(f"downsample factor must be >= 1, got {factor}")
    return tuple(int(np.ceil(int(dim) / factor)) for dim in shape_zyx)


def estimate_downsampled_memory(shape_zyx, factor: int = 4, *, copies: int = LEE_PEAK_COPIES) -> dict:
    """RAM estimate for a uint8 mask after isotropic integer downsample."""
    native = tuple(int(v) for v in shape_zyx)
    factor = max(1, int(factor))
    ds_shape = downsampled_shape(native, factor)
    native_voxels = int(np.prod(np.asarray(native, dtype=np.int64)))
    ds_voxels = int(np.prod(np.asarray(ds_shape, dtype=np.int64)))
    return {
        "native_shape_zyx": list(native),
        "downsampled_shape_zyx": list(ds_shape),
        "downsample_factor": int(factor),
        "native_voxels": native_voxels,
        "downsampled_voxels": ds_voxels,
        "uint8_gib": ds_voxels / BYTES_PER_GIB,
        "lee_peak_gib_est": (ds_voxels * int(copies)) / BYTES_PER_GIB,
        "fits_128gib_uint8_array": ds_voxels < 128 * BYTES_PER_GIB,
    }


def block_reduce_binary(
    binary_mask: np.ndarray,
    factor: int,
    *,
    method: str = "occupancy",
    occupancy_threshold: float | None = None,
) -> np.ndarray:
    """Isotropic integer downsample of a 3D binary block.

    ``max_pool``
        Any foreground voxel in the ``factor^3`` window survives.
    ``occupancy``
        Survive if the fill fraction is ``>= occupancy_threshold``
        (default ``1 / factor**2``).
    ``majority``
        Survive if fill fraction is ``>= 0.5``. Drops capillaries thinner
        than about half a coarse voxel.
    """
    factor = int(factor)
    method = str(method)
    if method not in VALID_METHODS:
        raise ValueError(f"method must be one of {VALID_METHODS}, got {method!r}")
    binary = np.asarray(binary_mask, dtype=bool)
    if binary.ndim != 3:
        raise ValueError(f"Expected 3D mask, got shape={binary.shape}")
    if factor <= 1:
        return binary

    if occupancy_threshold is None:
        occupancy_threshold = default_occupancy_threshold(factor)

    shape = np.asarray(binary.shape, dtype=np.int64)
    pad = [(-int(dim) % factor) for dim in shape]
    if any(pad):
        binary = np.pad(
            binary,
            ((0, pad[0]), (0, pad[1]), (0, pad[2])),
            mode="constant",
            constant_values=False,
        )
    padded = binary.shape
    reshaped = binary.reshape(
        padded[0] // factor,
        factor,
        padded[1] // factor,
        factor,
        padded[2] // factor,
        factor,
    )
    if method == "max_pool":
        return reshaped.any(axis=(1, 3, 5))
    occupancy = reshaped.mean(axis=(1, 3, 5))
    if method == "majority":
        return occupancy >= 0.5
    return occupancy >= float(occupancy_threshold)


def downsample_binary_mask_zarr(
    mask_zarr_path,
    output_zarr_path,
    factor: int = 4,
    *,
    dataset_name: str = "0",
    foreground_label: int | None = 1,
    method: str = "occupancy",
    occupancy_threshold: float | None = None,
    chunks=None,
    workers: int = 1,
    progress_callback=None,
):
    """Write an isotropically downsampled binary mask Zarr.

    Returns ``(output_path, out_shape, out_chunks)``.
    """
    factor = int(factor)
    method = str(method)
    if factor < 1:
        raise ValueError(f"downsample factor must be >= 1, got {factor}")
    if method not in VALID_METHODS:
        raise ValueError(f"method must be one of {VALID_METHODS}, got {method!r}")
    if occupancy_threshold is None:
        occupancy_threshold = default_occupancy_threshold(factor)

    src = open_zarr_dataset(mask_zarr_path, dataset_name=dataset_name)
    if src.ndim != 3:
        raise ValueError(f"Expected 3D mask, got shape={src.shape}")

    out_shape = downsampled_shape(src.shape, factor)
    src_chunks = tuple(int(v) for v in (chunks or getattr(src, "chunks", None) or (64, 64, 64)))
    out_chunks = tuple(max(1, int(np.ceil(c / factor))) for c in src_chunks)
    out_chunks = tuple(min(out_chunks[i], out_shape[i]) for i in range(3))

    output_path = Path(output_zarr_path)
    if output_path.exists():
        shutil.rmtree(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    root = zarr.open_group(str(output_path), mode="w")
    out = root.create_dataset(
        "0",
        shape=out_shape,
        chunks=out_chunks,
        dtype=np.uint8,
        overwrite=True,
    )

    indices = list(iter_all_chunk_indices(out_shape, out_chunks))
    workers = max(1, int(workers))

    def _reduce_one(out_index):
        out_slices = chunk_index_to_slices(out_index, out_chunks, out_shape)
        src_slices = tuple(
            slice(int(s.start) * factor, min(int(s.stop) * factor, int(src.shape[axis])))
            for axis, s in enumerate(out_slices)
        )
        block = np.asarray(src[src_slices])
        if foreground_label is None:
            binary = block > 0
        else:
            binary = block == foreground_label
        if not np.any(binary):
            return out_index, None
        reduced = block_reduce_binary(
            binary,
            factor,
            method=method,
            occupancy_threshold=occupancy_threshold,
        )
        target_shape = tuple(s.stop - s.start for s in out_slices)
        reduced = reduced[: target_shape[0], : target_shape[1], : target_shape[2]]
        return out_index, reduced.astype(np.uint8)

    done = 0
    total = len(indices)
    if progress_callback is not None:
        progress_callback(done, total)

    def _commit(out_index, reduced):
        nonlocal done
        if reduced is not None:
            out_slices = chunk_index_to_slices(out_index, out_chunks, out_shape)
            out[out_slices] = reduced
        done += 1
        if progress_callback is not None:
            progress_callback(done, total)

    if workers == 1 or total <= 1:
        for out_index in tqdm(indices, desc="Downsample mask"):
            _, reduced = _reduce_one(out_index)
            _commit(out_index, reduced)
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_reduce_one, out_index) for out_index in indices]
            for future in tqdm(as_completed(futures), total=total, desc="Downsample mask"):
                out_index, reduced = future.result()
                _commit(out_index, reduced)

    root.attrs["downsample_factor"] = int(factor)
    root.attrs["source_mask"] = str(mask_zarr_path)
    root.attrs["method"] = str(method)
    root.attrs["occupancy_threshold"] = float(occupancy_threshold)
    logger.info(
        "Wrote downsampled mask %s shape=%s method=%s threshold=%.4f",
        output_path,
        out_shape,
        method,
        occupancy_threshold,
    )
    return output_path, out_shape, out_chunks
