from __future__ import annotations

import pytest

from pipeline_modules.preprocessing.ims_to_zarr import (
    _coarse_reg_output_path,
    ims_level_stride_xyz,
    resolve_reg_channel,
)
from pipeline_modules.preprocessing.zarr_to_registration_nii import resolve_input_resolution_xyz
from pipeline_modules.utils.errors import PipelineError
from pipeline_modules.utils.zarr_io import create_output_zarr


def _write_fake_ims(path, shape_l0=(64, 64, 64), stride=4, level=2):
    """Minimal IMS-like HDF5: DataSet/ResolutionLevel N/TimePoint 0/Channel 0/Data."""
    import h5py

    shape_coarse = tuple(max(s // stride, 1) for s in shape_l0)
    with h5py.File(str(path), "w") as handle:
        for res_level, shape in ((0, shape_l0), (level, shape_coarse)):
            tp = handle.require_group(f"DataSet/ResolutionLevel {res_level}/TimePoint 0")
            tp.require_group("Channel 0").create_dataset("Data", shape=shape, dtype="uint16")


def test_ims_level_stride_from_shapes(tmp_path):
    ims = tmp_path / "sample.ims"
    _write_fake_ims(ims, shape_l0=(64, 64, 64), stride=4, level=2)
    assert ims_level_stride_xyz(ims, 2) == [4.0, 4.0, 4.0]
    assert ims_level_stride_xyz(ims, 0) == [1.0, 1.0, 1.0]


def test_ims_level_stride_non_cubic(tmp_path):
    ims = tmp_path / "sample.ims"
    _write_fake_ims(ims, shape_l0=(80, 64, 32), stride=8, level=2)
    assert ims_level_stride_xyz(ims, 2) == [8.0, 8.0, 8.0]


def test_resolve_reg_channel_modes(monkeypatch):
    assert resolve_reg_channel("none", reg_resolution_level=2) is None
    assert resolve_reg_channel("0", reg_resolution_level=2) == 0
    assert resolve_reg_channel("", reg_resolution_level=2, tty=False) is None
    monkeypatch.setattr("builtins.input", lambda prompt: "y")
    assert resolve_reg_channel("", reg_resolution_level=2, tty=True) == 0
    monkeypatch.setattr("builtins.input", lambda prompt: "")
    assert resolve_reg_channel("", reg_resolution_level=2, tty=True) is None
    with pytest.raises(PipelineError):
        resolve_reg_channel("abc", reg_resolution_level=2)


def test_coarse_reg_output_path_naming():
    assert _coarse_reg_output_path("s/ch1.zarr", [1], 0).name == "ch0_downsampled.zarr"
    multi = _coarse_reg_output_path("s/sample.zarr", [1, 2], 0)
    assert multi.name == "ch0_downsampled.zarr" and multi.parent.name == "sample"


def test_resolve_input_resolution_prefers_stride_attrs(tmp_path):
    zarr_path = tmp_path / "ch0_downsampled.zarr"
    _, arr = create_output_zarr(zarr_path, (8, 32, 24), (4, 8, 8), "uint16")
    arr.attrs["ims_resolution_level"] = 2
    arr.attrs["ims_stride_xyz"] = [8.0, 8.0, 4.0]

    import zarr

    group = zarr.open_group(str(zarr_path), mode="r+")
    group.attrs["ims_resolution_level"] = 2
    group.attrs["ims_stride_xyz"] = [8.0, 8.0, 4.0]

    res, source = resolve_input_resolution_xyz(zarr_path, (1.8, 1.8, 2.0))
    assert res == (14.4, 14.4, 8.0)
    assert "ims_stride_xyz" in source


def test_resolve_input_resolution_level_fallback(tmp_path):
    zarr_path = tmp_path / "ch0_downsampled.zarr"
    create_output_zarr(zarr_path, (8, 32, 24), (4, 8, 8), "uint16")

    import zarr

    group = zarr.open_group(str(zarr_path), mode="r+")
    group.attrs["ims_resolution_level"] = 2

    res, source = resolve_input_resolution_xyz(zarr_path, (1.8, 1.8, 2.0))
    assert res == (7.2, 7.2, 8.0)
    assert "2^level" in source


def test_resolve_input_resolution_native_without_attrs(tmp_path):
    zarr_path = tmp_path / "ch0.zarr"
    create_output_zarr(zarr_path, (8, 32, 24), (4, 8, 8), "uint16")
    res, source = resolve_input_resolution_xyz(zarr_path, (1.8, 1.8, 2.0))
    assert res == (1.8, 1.8, 2.0)
    assert "native" in source
