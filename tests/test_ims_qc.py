from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pipeline_modules.qc.ims_io import (
    group_z_indices_into_read_ranges,
    open_ims_dataset,
    select_histogram_z_chunk_ids,
    select_qc_z_indices,
)
from pipeline_modules.qc.image_qc import ImageQcConfig, run_image_qc


def _write_synthetic_ims(
    path: Path,
    *,
    shape: tuple[int, int, int] = (128, 256, 256),
    chunks: tuple[int, int, int] = (32, 128, 128),
) -> None:
    import h5py

    volume = np.linspace(0, 65535, num=int(np.prod(shape)), dtype=np.uint16).reshape(shape)
    with h5py.File(path, "w") as handle:
        tp = handle.create_group("DataSet").create_group("ResolutionLevel 0").create_group("TimePoint 0")
        ch = tp.create_group("Channel 0")
        ch.create_dataset(
            "Data",
            data=volume,
            chunks=chunks,
            compression="gzip",
        )


def test_select_qc_z_indices_align_to_chunk_centers():
    indices = select_qc_z_indices(3808, 8, 32, align_to_chunk=True)
    assert len(indices) == 8
    assert indices == sorted(indices)
    for z_index in indices:
        assert z_index % 32 == 16 or z_index == 3807


def test_group_z_indices_into_read_ranges_merges_neighbors():
    z_indices = [16, 48, 4000]
    ranges = group_z_indices_into_read_ranges(z_indices, shape_z=3808, z_chunk=32)
    assert ranges == [(0, 64), (4000, 3808)]


def test_select_histogram_z_chunk_ids_limits_reads():
    chunk_ids = select_histogram_z_chunk_ids(3808, 32, 24)
    assert len(chunk_ids) == 24
    assert chunk_ids == sorted(chunk_ids)
    assert chunk_ids[0] == 0
    assert chunk_ids[-1] == 118


def test_run_image_qc_ims_nas_mode(tmp_path: Path):
    ims_path = tmp_path / "sample.ims"
    _write_synthetic_ims(ims_path, shape=(256, 256, 256))
    output_json = tmp_path / "qc.json"
    results = run_image_qc(
        input_ims=ims_path,
        output_json=output_json,
        config=ImageQcConfig(
            max_slices=8,
            ims_resolution_level=0,
            ims_histogram_z_chunks=8,
            projection="none",
            ims_channel=0,
            grading_enabled=True,
            show_progress=False,
        ),
    )
    assert output_json.exists()
    assert results["source"]["source_kind"] == "ims"
    assert results["source"]["read_strategy"] == "chunk_aligned_z_batch"
    assert results["source"]["histogram_z_chunks_read"] <= 24
    assert len(results["slice_metrics"]) == 8
    assert results["config"]["ims_histogram_z_chunks"] == 8
    assert "timing_breakdown" in results


def test_open_ims_dataset_reads_expected_shape(tmp_path: Path):
    ims_path = tmp_path / "sample.ims"
    _write_synthetic_ims(ims_path, shape=(64, 128, 128))
    with open_ims_dataset(ims_path, resolution_level=0, channel=0) as (dataset, info):
        assert dataset.shape == (64, 128, 128)
        assert info.chunks_zyx == (32, 128, 128)
