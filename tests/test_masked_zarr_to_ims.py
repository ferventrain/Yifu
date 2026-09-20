from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from pipeline_modules.utils.errors import PipelineError
from pipeline_modules.utils.masked_zarr_to_ims import (
    _hdf_attr_text,
    clamp_ims_chunks,
    downsample_block_majority,
    iter_z_ranges,
    write_masked_ims,
)
from pipeline_modules.utils.zarr_io import create_output_zarr
from pipeline_modules.qc.ims_io import open_ims_dataset


MAX_CHUNK_BYTES = 4 * 1024 * 1024


def _write_mask_zarr(path: Path, volume: np.ndarray, chunks: tuple[int, int, int] = (4, 8, 8)) -> Path:
    _, array = create_output_zarr(path, tuple(volume.shape), chunks, volume.dtype)
    array[:] = volume
    return path


def _write_synthetic_ims(
    path: Path,
    volumes: dict[int, np.ndarray],
    *,
    chunks: tuple[int, int, int] = (4, 8, 8),
    channel_names: dict[int, str] | None = None,
) -> None:
    with h5py.File(path, "w") as handle:
        tp = handle.create_group("DataSet").create_group("ResolutionLevel 0").create_group("TimePoint 0")
        for channel, volume in volumes.items():
            tp.create_group(f"Channel {channel}").create_dataset(
                "Data", data=volume, chunks=chunks, compression="gzip"
            )
        info = handle.create_group("DataSetInfo")
        info.create_group("Imaris").attrs["Version"] = "9.5.0"
        depth, height, width = volumes[next(iter(volumes))].shape
        image = info.create_group("Image")
        image.attrs["X"] = str(width)
        image.attrs["Y"] = str(height)
        image.attrs["Z"] = str(depth)
        for channel, volume in volumes.items():
            name = (channel_names or {}).get(channel, f"Channel {channel}")
            info.create_group(f"Channel {channel}").attrs["Name"] = name


def _read_ims_channel(path: Path, channel: int) -> np.ndarray:
    with open_ims_dataset(path, resolution_level=0, channel=channel) as (dataset, info):
        return np.asarray(dataset[:])


def test_clamp_ims_chunks_respects_byte_budget():
    assert clamp_ims_chunks((3808, 10368, 7552), 2, (64, 256, 256)) == (32, 256, 256)
    assert clamp_ims_chunks((3808, 10368, 7552), 1, (64, 256, 256)) == (64, 256, 256)
    chunks = clamp_ims_chunks((3808, 10368, 7552), 2, (256, 256, 256))
    assert chunks[0] * chunks[1] * chunks[2] * 2 <= MAX_CHUNK_BYTES
    assert clamp_ims_chunks((8, 16, 16), 2, (64, 256, 256)) == (8, 16, 16)


def test_iter_z_ranges_covers_depth():
    assert list(iter_z_ranges(10, 4)) == [(0, 4), (4, 8), (8, 10)]
    assert list(iter_z_ranges(0, 4)) == []
    assert list(iter_z_ranges(1, 32)) == [(0, 1)]


def test_downsample_block_majority_threshold():
    block = np.zeros((2, 2, 2), dtype=np.uint16)
    assert int(downsample_block_majority(block)[0, 0, 0]) == 0
    block[0, 0, 0] = 1
    assert int(downsample_block_majority(block)[0, 0, 0]) == 0  # mean 0.125 <= 0.5
    for offset in range(4):
        block[offset // 2, offset % 2, 0] = 1
    assert int(downsample_block_majority(block)[0, 0, 0]) == 0  # mean 0.5 is not > 0.5
    block[1, 1, 1] = 1
    assert int(downsample_block_majority(block)[0, 0, 0]) == 1  # 5 of 8 voxels set


def test_write_mask_ims_single_channel(tmp_path: Path):
    mask = np.zeros((8, 16, 16), dtype=np.uint16)
    mask[2:5, 3:9, 4:12] = 1
    mask_zarr = _write_mask_zarr(tmp_path / "mask.zarr", mask)
    output_ims = tmp_path / "masked.ims"

    result = write_masked_ims(mask_zarr, output_ims, chunk_size=(4, 8, 8), z_block=4)

    assert result["success"]
    assert result["pyramid_levels"] == 1
    assert [entry["name"] for entry in result["channels"]] == ["ch0"]
    assert np.array_equal(_read_ims_channel(output_ims, 0), mask)

    with h5py.File(output_ims, "r") as handle:
        data = handle["DataSet/ResolutionLevel 0/TimePoint 0/Channel 0/Data"]
        assert data.chunks is not None
        assert np.prod(data.chunks) * 2 <= MAX_CHUNK_BYTES
        assert data.compression == "gzip"
        info = handle["DataSetInfo"]
        assert _hdf_attr_text(info["Imaris"].attrs["Version"]) == "5.5.0"
        assert _hdf_attr_text(info["Image"].attrs["X"]) == "16"
        assert _hdf_attr_text(info["Image"].attrs["Y"]) == "16"
        assert _hdf_attr_text(info["Image"].attrs["Z"]) == "8"
        assert _hdf_attr_text(info["Image"].attrs["Noc"]) == "1"
        # Voxel extents must exist (Imaris computes voxel size from them).
        assert _hdf_attr_text(info["Image"].attrs["ExtMin0"]) == "0"
        assert _hdf_attr_text(info["Image"].attrs["ExtMax0"]) == "16"
        assert _hdf_attr_text(info["Image"].attrs["ExtMax2"]) == "8"
        assert _hdf_attr_text(info["Image"].attrs["Unit"]) == "um"
        assert _hdf_attr_text(info["Channel 0"].attrs["Name"]) == "ch0"
        # Real histogram datasets + display range from actual values.
        assert "Histogram" in handle["DataSet/ResolutionLevel 0/TimePoint 0/Channel 0"]
        assert "Histogram1024" in handle["DataSet/ResolutionLevel 0/TimePoint 0/Channel 0"]
        assert _hdf_attr_text(info["Channel 0"].attrs["ColorRange"]) == "0.000 1.000"
        assert "TimePoint1" in info["TimeInfo"].attrs
        assert _hdf_attr_text(info["ImarisDataSet"].attrs["Creator"]) == "Imaris x64"
        assert "Entries" in info["Log"].attrs
        # Root format attributes Imaris needs to identify the file.
        assert _hdf_attr_text(handle.attrs["ImarisVersion"]) == "5.5.0"
        assert _hdf_attr_text(handle.attrs["DataSetDirectoryName"]) == "DataSet"
        assert handle.attrs["NumberOfDataSets"].shape == (1,)
        # Attr char arrays must be sized exactly to the text: real Imaris files
        # carry NO trailing NUL, and "DataSet\x00" fails Imaris's lookups.
        assert len(handle.attrs["ImarisVersion"]) == len("5.5.0")
        assert len(handle.attrs["DataSetDirectoryName"]) == len("DataSet")
        assert len(info["Image"].attrs["X"]) == len("16")
        # Every attr Imaris parses must be an |S1 char array, not vlen str.
        assert handle.attrs["ImarisVersion"].dtype.kind == "S"
        assert info["Image"].attrs["X"].dtype.kind == "S"
        assert handle["Thumbnail/Data"].shape == (256, 1024)  # W x 4W RGBA
        assert "ResolutionLevel 1" not in handle["DataSet"]


def test_write_mask_ims_replaces_source_channel(tmp_path: Path):
    shape = (8, 16, 16)
    source_volumes = {
        0: (np.arange(int(np.prod(shape))).reshape(shape) % 1000).astype(np.uint16),
        1: (np.arange(int(np.prod(shape))).reshape(shape) % 997).astype(np.uint16),
    }
    mask = np.zeros(shape, dtype=np.uint16)
    mask[1:4, 2:6, 3:9] = 1

    source_ims = tmp_path / "source.ims"
    _write_synthetic_ims(source_ims, source_volumes, channel_names={0: "Green", 1: "Red"})
    mask_zarr = _write_mask_zarr(tmp_path / "mask.zarr", mask)
    output_ims = tmp_path / "masked.ims"

    result = write_masked_ims(
        mask_zarr,
        output_ims,
        source_ims=source_ims,
        mask_channel=1,
        chunk_size=(4, 8, 8),
        z_block=4,
    )

    assert [entry["index"] for entry in result["channels"]] == [0, 1]
    assert result["channels"][0] == {"index": 0, "name": "Green", "source": "ims", "dtype": "uint16"}
    assert result["channels"][1] == {"index": 1, "name": "ch1", "source": "mask", "dtype": "uint16"}
    assert np.array_equal(_read_ims_channel(output_ims, 0), source_volumes[0])
    assert np.array_equal(_read_ims_channel(output_ims, 1), mask)

    # The source file must be untouched.
    with h5py.File(source_ims, "r") as handle:
        for channel, volume in source_volumes.items():
            assert np.array_equal(
                np.asarray(handle[f"DataSet/ResolutionLevel 0/TimePoint 0/Channel {channel}/Data"][:]),
                volume,
            )
        # Voxel metadata carried over from the source Image group.
        assert _hdf_attr_text(handle["DataSetInfo/Image"].attrs["X"]) == "16"


def test_write_mask_ims_appends_mask_as_new_channel(tmp_path: Path):
    shape = (8, 16, 16)
    source_volume = (np.arange(int(np.prod(shape))).reshape(shape) % 1000).astype(np.uint16)
    mask = np.zeros(shape, dtype=np.uint16)
    mask[0:2, :, :] = 1

    source_ims = tmp_path / "source.ims"
    _write_synthetic_ims(source_ims, {0: source_volume})
    mask_zarr = _write_mask_zarr(tmp_path / "mask.zarr", mask)
    output_ims = tmp_path / "masked.ims"

    result = write_masked_ims(
        mask_zarr,
        output_ims,
        source_ims=source_ims,
        append_as_new_channel=True,
        chunk_size=(4, 8, 8),
        z_block=4,
    )

    assert [entry["index"] for entry in result["channels"]] == [0, 1]
    assert result["channels"][1]["name"] == "ch1"
    assert np.array_equal(_read_ims_channel(output_ims, 0), source_volume)
    assert np.array_equal(_read_ims_channel(output_ims, 1), mask)


def test_source_channel_name_from_byte_array(tmp_path: Path):
    """Real Imaris files store DataSetInfo attrs as |S1 byte arrays."""
    shape = (8, 16, 16)
    source_volume = (np.arange(int(np.prod(shape))).reshape(shape) % 1000).astype(np.uint16)
    mask = np.zeros(shape, dtype=np.uint16)
    mask[0:2, :, :] = 1

    source_ims = tmp_path / "source.ims"
    _write_synthetic_ims(source_ims, {0: source_volume})
    with h5py.File(source_ims, "a") as handle:
        handle["DataSetInfo/Channel 0"].attrs["Name"] = np.array(
            [bytes(char, "ascii") for char in "(name not specified)"], dtype="S1"
        )
    mask_zarr = _write_mask_zarr(tmp_path / "mask.zarr", mask)
    output_ims = tmp_path / "masked.ims"

    result = write_masked_ims(
        mask_zarr,
        output_ims,
        source_ims=source_ims,
        append_as_new_channel=True,
        chunk_size=(4, 8, 8),
        z_block=4,
    )

    assert result["channels"][0]["name"] == "(name not specified)"
    with h5py.File(output_ims, "r") as handle:
        # Creator app version is carried from the source; format version stays at the root.
        assert _hdf_attr_text(handle["DataSetInfo/Imaris"].attrs["Version"]) == "9.5.0"
        assert _hdf_attr_text(handle["DataSetInfo/Channel 0"].attrs["Name"]) == "(name not specified)"


def test_write_mask_ims_shape_mismatch_raises(tmp_path: Path):
    shape = (8, 16, 16)
    source_ims = tmp_path / "source.ims"
    _write_synthetic_ims(source_ims, {0: np.zeros(shape, dtype=np.uint16)})
    mask_zarr = _write_mask_zarr(tmp_path / "mask.zarr", np.zeros((4, 8, 8), dtype=np.uint16))

    with pytest.raises(PipelineError):
        write_masked_ims(mask_zarr, tmp_path / "masked.ims", source_ims=source_ims)


def test_write_mask_ims_rejects_mask_channel_missing_from_source(tmp_path: Path):
    shape = (8, 16, 16)
    source_ims = tmp_path / "source.ims"
    _write_synthetic_ims(source_ims, {0: np.zeros(shape, dtype=np.uint16)})
    mask_zarr = _write_mask_zarr(tmp_path / "mask.zarr", np.zeros(shape, dtype=np.uint16))

    with pytest.raises(PipelineError):
        write_masked_ims(mask_zarr, tmp_path / "masked.ims", source_ims=source_ims, mask_channel=3)


def test_write_mask_ims_output_guards(tmp_path: Path):
    shape = (8, 16, 16)
    source_ims = tmp_path / "source.ims"
    _write_synthetic_ims(source_ims, {0: np.zeros(shape, dtype=np.uint16)})
    mask_zarr = _write_mask_zarr(tmp_path / "mask.zarr", np.zeros(shape, dtype=np.uint16))
    output_ims = tmp_path / "masked.ims"

    # Never write into the source .ims.
    with pytest.raises(PipelineError):
        write_masked_ims(mask_zarr, source_ims, source_ims=source_ims)

    write_masked_ims(mask_zarr, output_ims, chunk_size=(4, 8, 8), z_block=4)
    with pytest.raises(PipelineError):
        write_masked_ims(mask_zarr, output_ims, chunk_size=(4, 8, 8), z_block=4)
    write_masked_ims(mask_zarr, output_ims, chunk_size=(4, 8, 8), z_block=4, overwrite=True)


def test_write_mask_ims_pyramid_max_levels(tmp_path: Path):
    mask = np.zeros((8, 16, 16), dtype=np.uint16)
    mask[0:2, 0:2, 0:2] = 1  # 8 of 8 -> kept
    mask[2:4, 0:2, 0:2] = 1
    mask[2, 0, 1] = 0  # 7 of 8 -> kept
    mask[4:6, 0:2, 0:2] = 1
    mask[4, 0, 0] = 0
    mask[4, 0, 1] = 0
    mask[4, 1, 0] = 0
    mask[5, 1, 1] = 0  # 4 of 8 -> MAX pooling still keeps the dot visible
    mask_zarr = _write_mask_zarr(tmp_path / "mask.zarr", mask)
    output_ims = tmp_path / "masked.ims"

    result = write_masked_ims(
        mask_zarr,
        output_ims,
        pyramid_levels=2,
        chunk_size=(4, 8, 8),
        z_block=4,
    )

    assert result["level_shapes"] == [[8, 16, 16], [4, 8, 8], [2, 4, 4]]

    with h5py.File(output_ims, "r") as handle:
        level1 = handle["DataSet/ResolutionLevel 1/TimePoint 0/Channel 0/Data"]
        assert level1.shape == (4, 8, 8)
        level1_data = np.asarray(level1[:])
        assert int(level1_data[0, 0, 0]) == 1
        assert int(level1_data[1, 0, 0]) == 1
        assert int(level1_data[2, 0, 0]) == 1  # max pooling keeps sparse dots
        level2 = handle["DataSet/ResolutionLevel 2/TimePoint 0/Channel 0/Data"]
        assert level2.shape == (2, 4, 4)
        level2_data = np.asarray(level2[:])
        assert int(level2_data[1, 1, 1]) == 1  # grown footprint at coarse level
        assert "ResolutionLevel 3" not in handle["DataSet"]
        level2_channel = handle["DataSet/ResolutionLevel 2/TimePoint 0/Channel 0"]
        assert _hdf_attr_text(level2_channel.attrs["ImageSizeZ"]) == "2"
