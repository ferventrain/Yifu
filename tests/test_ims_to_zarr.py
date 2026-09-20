from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from pipeline_modules.preprocessing.ims_to_zarr import (
    convert_ims_channel_to_zarr,
    convert_ims_to_zarr,
    list_ims_channel_indices,
    parse_channel_spec,
    resolve_output_paths,
)
from pipeline_modules.utils.errors import PipelineError
from pipeline_modules.utils.zarr_io import list_existing_chunk_indices, open_zarr_array


def _write_synthetic_ims(
    path: Path,
    *,
    shape: tuple[int, int, int] = (12, 24, 20),
    chunks: tuple[int, int, int] = (4, 8, 10),
    channels: tuple[int, ...] = (0,),
) -> dict[int, np.ndarray]:
    volumes: dict[int, np.ndarray] = {}
    with h5py.File(path, "w") as handle:
        tp = handle.create_group("DataSet").create_group("ResolutionLevel 0").create_group("TimePoint 0")
        for channel in channels:
            volume = (
                np.arange(int(np.prod(shape)), dtype=np.uint16).reshape(shape) + 100 * (channel + 1)
            ).astype(np.uint16)
            tp.create_group(f"Channel {channel}").create_dataset(
                "Data",
                data=volume,
                chunks=chunks,
                compression="gzip",
            )
            volumes[channel] = volume
    return volumes


def test_list_ims_channel_indices(tmp_path: Path):
    ims_path = tmp_path / "sample.ims"
    _write_synthetic_ims(ims_path, channels=(0, 2))
    assert list_ims_channel_indices(ims_path) == [0, 2]


def test_parse_channel_spec_all_and_list():
    assert parse_channel_spec("all", [0, 1, 3]) == [0, 1, 3]
    assert parse_channel_spec("3,0", [0, 1, 3]) == [0, 3]
    assert parse_channel_spec([1], [0, 1]) == [1]
    with pytest.raises(PipelineError):
        parse_channel_spec("9", [0, 1])


def test_resolve_output_paths_single_and_multi():
    assert resolve_output_paths(Path("out/ch0.zarr"), [0]) == [Path("out/ch0.zarr")]
    multi = resolve_output_paths(Path("out/sample.zarr"), [0, 1])
    assert multi == [Path("out/sample/ch0.zarr"), Path("out/sample/ch1.zarr")]


def test_convert_ims_channel_to_zarr_roundtrip(tmp_path: Path):
    ims_path = tmp_path / "sample.ims"
    volumes = _write_synthetic_ims(ims_path, shape=(12, 24, 20), chunks=(4, 8, 10))
    output_zarr = tmp_path / "sample.zarr"

    result = convert_ims_channel_to_zarr(
        ims_path,
        output_zarr,
        channel=0,
        chunk_size=(4, 8, 10),
        write_manifest=False,
    )

    assert result["success"]
    assert result["shape"] == [12, 24, 20]
    assert result["written_chunks"] == result["total_chunks"]
    assert result["skipped_chunks"] == 0
    array = open_zarr_array(output_zarr)
    assert array.shape == (12, 24, 20)
    assert array.dtype == np.uint16
    assert np.array_equal(np.asarray(array[:]), volumes[0])

    compressor_meta = json.loads((output_zarr / "0" / ".zarray").read_text())["compressor"]
    assert compressor_meta["id"] == "gzip"


def test_convert_ims_channel_to_zarr_resume_skips_written_chunks(tmp_path: Path):
    ims_path = tmp_path / "sample.ims"
    volumes = _write_synthetic_ims(ims_path, shape=(12, 24, 20), chunks=(4, 8, 10))
    output_zarr = tmp_path / "sample.zarr"

    result = convert_ims_channel_to_zarr(ims_path, output_zarr, channel=0, chunk_size=(4, 8, 10), write_manifest=False)
    assert result["success"]

    rerun = convert_ims_channel_to_zarr(ims_path, output_zarr, channel=0, chunk_size=(4, 8, 10), write_manifest=False)
    assert rerun["skipped_chunks"] == result["total_chunks"]
    assert rerun["written_chunks"] == 0
    assert np.array_equal(np.asarray(open_zarr_array(output_zarr)[:]), volumes[0])


def test_convert_ims_channel_to_zarr_resume_with_partial_store(tmp_path: Path):
    ims_path = tmp_path / "sample.ims"
    volumes = _write_synthetic_ims(ims_path, shape=(12, 24, 20), chunks=(4, 8, 10))
    output_zarr = tmp_path / "sample.zarr"

    # Pre-populate one chunk row with correct data, as an interrupted run would.
    from pipeline_modules.utils.zarr_io import create_output_zarr

    _, array = create_output_zarr(output_zarr, (12, 24, 20), (4, 8, 10), np.uint16)
    array[0:4] = volumes[0][0:4]
    pre_existing = len(list_existing_chunk_indices(array))
    assert pre_existing == 6  # (4/8) * (20/10) tiles in the first z-row

    result = convert_ims_channel_to_zarr(ims_path, output_zarr, channel=0, chunk_size=(4, 8, 10), write_manifest=False)
    assert result["skipped_chunks"] == pre_existing
    assert result["written_chunks"] == result["total_chunks"] - pre_existing
    assert np.array_equal(np.asarray(open_zarr_array(output_zarr)[:]), volumes[0])


def test_convert_ims_channel_to_zarr_rebuilds_stale_store(tmp_path: Path):
    ims_path = tmp_path / "sample.ims"
    volumes = _write_synthetic_ims(ims_path, shape=(12, 24, 20), chunks=(4, 8, 10))
    output_zarr = tmp_path / "sample.zarr"

    from pipeline_modules.utils.zarr_io import create_output_zarr

    create_output_zarr(output_zarr, (2, 2, 2), (2, 2, 2), np.uint8)

    result = convert_ims_channel_to_zarr(ims_path, output_zarr, channel=0, chunk_size=(4, 8, 10), write_manifest=False)
    assert result["success"]
    array = open_zarr_array(output_zarr)
    assert array.shape == (12, 24, 20)
    assert np.array_equal(np.asarray(array[:]), volumes[0])


def test_convert_ims_to_zarr_multi_channel_dir_mode(tmp_path: Path):
    ims_path = tmp_path / "sample.ims"
    volumes = _write_synthetic_ims(ims_path, shape=(8, 16, 16), chunks=(4, 8, 8), channels=(0, 1))
    output_root = tmp_path / "batch"

    summary = convert_ims_to_zarr(ims_path, tmp_path / "batch.zarr", "all", chunk_size=(4, 8, 8))

    assert summary["success"]
    assert summary["channels_requested"] == [0, 1]
    assert not summary["failed"]
    for channel in (0, 1):
        channel_zarr = output_root / f"ch{channel}.zarr"
        assert channel_zarr.exists()
        array = open_zarr_array(channel_zarr)
        assert np.array_equal(np.asarray(array[:]), volumes[channel])


def test_convert_ims_to_zarr_single_channel_writes_exact_output(tmp_path: Path):
    ims_path = tmp_path / "sample.ims"
    volumes = _write_synthetic_ims(ims_path, shape=(8, 16, 16), chunks=(4, 8, 8), channels=(0,))
    output_zarr = tmp_path / "out" / "direct.zarr"

    summary = convert_ims_to_zarr(ims_path, output_zarr, "0", chunk_size=(4, 8, 8))

    assert summary["success"]
    assert np.array_equal(np.asarray(open_zarr_array(output_zarr)[:]), volumes[0])


def test_convert_ims_channel_missing_input_raises(tmp_path: Path):
    with pytest.raises(PipelineError):
        convert_ims_channel_to_zarr(tmp_path / "missing.ims", tmp_path / "out.zarr", channel=0)
