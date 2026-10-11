from __future__ import annotations

from pathlib import Path

import numpy as np

from pipeline_modules.registration.brain_mask_ds import (
    DS_FACTOR,
    brain_mask_ds_path,
    ds_grid_shape,
    open_brain_mask_ds,
    write_brain_mask_ds_from_source,
)


def test_brain_mask_ds_path_is_label_sibling():
    label = Path("S:/sample/upsampled_atlas_label.zarr")
    assert brain_mask_ds_path(label) == Path("S:/sample/upsampled_atlas_label_ds_brain_mask.zarr")


def test_ds_grid_shape_uses_ceiling():
    assert ds_grid_shape((4356, 10281, 7451), 4) == (1089, 2571, 1863)
    assert ds_grid_shape((16, 64, 64), 4) == (4, 16, 16)


def test_write_brain_mask_ds_from_source_geometry(tmp_path):
    # source (2, 8, 8): slice 0 has a left half, slice 1 is full
    src = np.zeros((2, 8, 8), dtype=bool)
    src[0][:, :4] = True
    src[1][:] = True
    # target (8, 32, 32): round(linspace(0, 1, 8)) maps rows 0-3 -> source 0, rows 4-7 -> source 1
    z_indices = np.round(np.linspace(0, 1, 8)).astype(int)

    out = tmp_path / "mask_ds.zarr"
    write_brain_mask_ds_from_source(
        src, z_indices, (8, 32, 32), out, spacing_zyx_um=(8.0, 7.2, 7.2)
    )

    mask, meta = open_brain_mask_ds(out)
    assert mask.shape == (2, 8, 8)
    # ds row 0 = union of the 4 mapped slices of source 0 = left half
    assert mask[0][:, :4].all()
    assert not mask[0][:, 4:].any()
    # ds row 1 = source 1 = full
    assert mask[1].all()
    assert meta["factor"] == DS_FACTOR
    assert meta["spacing_zyx_um"] == [8.0, 7.2, 7.2]
    assert meta["target_shape"] == [8, 32, 32]
