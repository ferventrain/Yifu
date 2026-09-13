from __future__ import annotations

from pathlib import Path

import numpy as np

from pipeline_modules.preprocessing.zarr_pyramid import add_resolution_pyramid, downsample_nearest2
from pipeline_modules.utils.zarr_io import write_array


def test_downsample_nearest2_keeps_labels():
    vol = np.zeros((4, 4, 4), dtype=np.uint16)
    vol[0:2, 0:2, 0:2] = 7
    down = downsample_nearest2(vol)
    assert down.shape == (2, 2, 2)
    assert int(down[0, 0, 0]) == 7


def test_add_resolution_pyramid_writes_ngff_levels(tmp_path: Path):
    path = tmp_path / "vol.zarr"
    data = np.arange(8 * 8 * 8, dtype=np.uint16).reshape(8, 8, 8)
    write_array(path, data, chunks=(4, 8, 8))
    result = add_resolution_pyramid(
        path,
        max_levels=3,
        min_size=2,
        z_block=4,
        write_manifest=False,
        method="nearest",
        scale_zyx=(2.0, 0.71, 0.71),
    )
    assert result["levels"] >= 2
    import zarr

    root = zarr.open_group(str(path), mode="r")
    assert "0" in root and "1" in root
    scales = root.attrs["multiscales"]
    assert scales[0]["axes"][0]["name"] == "z"
    assert len(scales[0]["datasets"]) == result["levels"]
    assert scales[0]["datasets"][1]["coordinateTransformations"][0]["scale"][0] == 4.0
