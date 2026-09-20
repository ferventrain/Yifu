from __future__ import annotations

from pathlib import Path

import numpy as np

from pipeline_modules.utils.zarr_io import (
    create_output_zarr,
    list_array_keys,
    list_existing_chunk_indices,
    normalize_zarr_dtype,
    open_group,
    open_zarr_array,
    write_array,
)


def test_create_array_accepts_tifffile_uint32_dtype(tmp_path: Path):
    import tifffile

    tiff_path = tmp_path / "label.tif"
    tifffile.imwrite(tiff_path, np.array([[70000]], dtype=np.uint32))
    sample_dtype = tifffile.imread(tiff_path).dtype
    assert normalize_zarr_dtype(sample_dtype) is np.uint32

    root, dataset = create_output_zarr(
        tmp_path / "labels.zarr",
        (2, 4, 4),
        (1, 4, 4),
        sample_dtype,
    )
    dataset[0] = 70000
    opened = open_zarr_array(tmp_path / "labels.zarr")
    assert opened.dtype == np.uint32
    assert int(opened[0, 0, 0]) == 70000


def test_write_array_keeps_zarr_v2_layout(tmp_path: Path):
    path = tmp_path / "volume.zarr"
    data = np.arange(24, dtype=np.uint16).reshape(2, 3, 4)
    write_array(path, data, chunks=(1, 3, 4))

    assert (path / ".zgroup").exists()
    assert (path / "0" / ".zarray").exists()
    assert not (path / "zarr.json").exists()

    arr = open_zarr_array(path)
    assert arr.shape == (2, 3, 4)
    assert int(arr[1, 0, 0]) == 12
    assert list_array_keys(open_group(path)) == ["0"]


def test_create_output_zarr_roundtrip(tmp_path: Path):
    path = tmp_path / "mask.zarr"
    root, dataset = create_output_zarr(path, (4, 4, 4), (2, 4, 4), np.uint8)
    dataset[0:2] = 1
    assert "0" in list_array_keys(root)
    opened = open_zarr_array(path)
    assert int(opened[0, 0, 0]) == 1
    assert int(opened[3, 0, 0]) == 0
    existing = list_existing_chunk_indices(opened)
    assert (0, 0, 0) in existing
