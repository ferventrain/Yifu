"""Factor-4 downsampled brain-mask cache derived from the warped atlas label.

Registration writes this cache next to the upsampled atlas label zarr
(`<label>_ds_brain_mask.zarr`) so downstream consumers (cfos postprocess edge
exclusion) can get a whole-brain mask at ~29 um without re-streaming the
multi-hundred-GB full-resolution label.

The ds mask is built from the same geometry the upsampler uses: every target
slice is a nearest resize of a mapped source (registration-resolution) slice,
so per ds row the union of its <=4 mapped source slices, resized to the ds xy
grid, equals the max-pooled label mask to within one ds voxel.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

DS_FACTOR = 4
CACHE_SUFFIX = "_ds_brain_mask"
ATTRS_KEY = "brain_mask_ds"


def brain_mask_ds_path(label_zarr: Path | str) -> Path:
    label_zarr = Path(label_zarr)
    return label_zarr.parent / f"{label_zarr.stem}{CACHE_SUFFIX}.zarr"


def ds_grid_shape(target_shape, factor: int = DS_FACTOR) -> tuple[int, ...]:
    return tuple(int(-(-int(s) // factor)) for s in target_shape)


def write_brain_mask_ds_from_source(
    warped_mask_zyx: np.ndarray,
    z_indices: np.ndarray,
    target_shape,
    out_zarr: Path | str,
    spacing_zyx_um=None,
    factor: int = DS_FACTOR,
) -> Path:
    """Write the ds brain-mask cache from the registration-resolution label mask.

    ``warped_mask_zyx`` is the boolean (label > 0) volume at registration
    resolution; ``z_indices`` maps every target z slice to its source slice
    (same linspace formula the upsampler uses). ``spacing_zyx_um`` is the
    target-resolution voxel size (one value per axis); stored in the cache
    attrs so consumers can work in micrometers.
    """
    import cv2

    from pipeline_modules.utils.zarr_io import create_output_zarr

    out_zarr = Path(out_zarr)
    src = warped_mask_zyx
    tgt = tuple(int(v) for v in target_shape)
    ds_shape = ds_grid_shape(tgt, factor)
    chunk = (min(8, ds_shape[0]), min(1024, ds_shape[1]), min(1024, ds_shape[2]))
    root, arr = create_output_zarr(out_zarr, ds_shape, chunk, "uint8")

    ds_xy = (ds_shape[2], ds_shape[1])  # cv2 (x, y)
    for row, z0 in enumerate(range(0, tgt[0], factor)):
        mapped = np.unique(z_indices[z0 : z0 + factor])
        union = np.zeros(src.shape[1:], dtype=bool)
        for sz in mapped:
            union |= src[int(sz)]
        arr[row] = cv2.resize(union.astype(np.uint8), ds_xy, interpolation=cv2.INTER_NEAREST)

    root.attrs[ATTRS_KEY] = {
        "factor": factor,
        "target_shape": list(tgt),
        "ds_shape": list(ds_shape),
        "spacing_zyx_um": [float(v) for v in spacing_zyx_um] if spacing_zyx_um is not None else None,
        "source": "warped_atlas_label",
    }
    return out_zarr


def open_brain_mask_ds(path: Path | str) -> tuple[np.ndarray, dict]:
    """Load the ds brain mask (whole array) and its attrs."""
    import zarr

    path = Path(path)
    try:
        grp = zarr.open_group(str(path), mode="r", zarr_format=2)
    except TypeError:  # zarr 2.x has no zarr_format kwarg
        grp = zarr.open_group(str(path), mode="r")
    meta = dict(grp.attrs.get(ATTRS_KEY, {}) or {})
    mask = np.asarray(grp["0"])
    return mask, meta
