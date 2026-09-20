"""Tests for ``pipeline_modules.registration.spinal_segment_signal_stats``."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pipeline_modules.registration.spinal_segment_signal_stats import (
    load_segment_legend_csv,
    resolve_segment_label_zarr,
    run_spinal_segment_analysis,
    upsample_segments_to_signal_grid,
    write_region_csv,
)
from pipeline_modules.utils.zarr_io import create_output_zarr, open_zarr_array

SIGNAL_SHAPE = (4, 8, 8)


def _write_legend(path, pairs):
    pd.DataFrame(pairs, columns=["segment_id", "name"]).to_csv(path, index=False)
    return path


@pytest.fixture()
def sample_dir(tmp_path):
    path = tmp_path / "sample01"
    path.mkdir()
    return path


@pytest.fixture()
def legend_csv(sample_dir):
    return _write_legend(
        sample_dir / "segment_id_legend.csv",
        [(2, "C3"), (1, "C1"), (4, "T2"), (3, "T1")],
    )


def test_load_segment_legend_csv_sorts_anatomically(legend_csv):
    pairs = load_segment_legend_csv(legend_csv)
    assert [name for _, name in pairs] == ["C1", "C3", "T1", "T2"]
    assert [segment_id for segment_id, _ in pairs] == [1, 2, 3, 4]


def test_load_segment_legend_csv_accepts_plain_segment_column(tmp_path):
    csv_path = tmp_path / "segments_reference.csv"
    pd.DataFrame({"Segment": ["T2", "C1", "C2"]}).to_csv(csv_path, index=False)
    pairs = load_segment_legend_csv(csv_path)
    assert pairs == [(1, "C1"), (2, "C2"), (3, "T2")]


def test_write_region_csv_builds_two_level_hierarchy(tmp_path, legend_csv):
    pairs = load_segment_legend_csv(legend_csv)
    out = write_region_csv(pairs, tmp_path / "regions.csv")

    frame = pd.read_csv(out)
    root = frame[frame["id"] == 999999].iloc[0]
    assert root["name"] == "Spinal cord all segments"
    assert root["structure_id_path"] == "[999999]"

    child = frame[frame["id"] == 1].iloc[0]
    assert child["name"] == "C1"
    assert child["structure_id_path"] == "[999999, 1]"
    assert len(frame) == 1 + len(pairs)


def _create_segments_zarr(path, shape, *, attrs=None, values=None):
    root, arr = create_output_zarr(path, shape, shape, "uint16")
    if values is not None:
        arr[:] = values
    if attrs:
        root.attrs.update(attrs)
    return path


def test_resolve_prefers_native_shape_match(sample_dir, tmp_path):
    _create_segments_zarr(sample_dir / "spinalj_segments_native.zarr", SIGNAL_SHAPE)
    _create_segments_zarr(sample_dir / "spinalj_segments.zarr", (2, 4, 4))

    path, upsampled = resolve_segment_label_zarr(sample_dir, "spinalj_segments", SIGNAL_SHAPE)
    assert upsampled is False
    assert path.name == "spinalj_segments_native.zarr"


def test_resolve_falls_back_to_midres_for_upsampling(sample_dir):
    _create_segments_zarr(
        sample_dir / "spinalj_segments.zarr",
        (2, 4, 4),
        attrs={"native_shape_zyx": list(SIGNAL_SHAPE), "z_step": 2},
    )

    path, upsampled = resolve_segment_label_zarr(sample_dir, "spinalj_segments", SIGNAL_SHAPE)
    assert upsampled is True
    assert path.name == "spinalj_segments.zarr"


def test_resolve_raises_on_missing_and_mismatch(sample_dir):
    with pytest.raises(FileNotFoundError):
        resolve_segment_label_zarr(sample_dir, "spinalj_segments", SIGNAL_SHAPE)

    _create_segments_zarr(sample_dir / "spinalj_segments.zarr", (3, 5, 5))
    with pytest.raises(ValueError):
        resolve_segment_label_zarr(sample_dir, "spinalj_segments", SIGNAL_SHAPE)


def test_upsample_segments_to_signal_grid(sample_dir):
    mid = _create_segments_zarr(
        sample_dir / "spinalj_segments.zarr",
        (2, 4, 4),
        attrs={"native_shape_zyx": list(SIGNAL_SHAPE), "z_step": 2},
        values=np.stack([np.ones((4, 4), dtype=np.uint16), 2 * np.ones((4, 4), dtype=np.uint16)]),
    )
    signal = sample_dir / "ch0.zarr"
    create_output_zarr(signal, SIGNAL_SHAPE, SIGNAL_SHAPE, "uint16")

    out = upsample_segments_to_signal_grid(
        mid, signal, sample_dir / "spinalj_segments_upsampled.zarr", (1.0, 1.0, 2.0)
    )
    arr = open_zarr_array(out)
    assert arr.shape == SIGNAL_SHAPE
    assert np.all(np.asarray(arr[0:2]) == 1)
    assert np.all(np.asarray(arr[2:4]) == 2)


def _build_signal_and_mask(sample_dir):
    signal = sample_dir / "ch0.zarr"
    mask = sample_dir / "ch0_mask.zarr"
    _, signal_arr = create_output_zarr(signal, SIGNAL_SHAPE, SIGNAL_SHAPE, "uint16")
    _, mask_arr = create_output_zarr(mask, SIGNAL_SHAPE, SIGNAL_SHAPE, "uint8")

    mask_arr[1, 1:3, 1:3] = 1           # small: 4 voxels, dropped (< min 5)
    mask_arr[1, 4:7, 4:7] = 1           # medium: 9 voxels, kept -> segment C1 (z < 2)
    mask_arr[3, 1:5, 1:5] = 1           # large: 16 voxels, dropped (> max 12) -> segment C2
    signal_arr[1, 1:3, 1:3] = 100
    signal_arr[1, 4:7, 4:7] = 100
    signal_arr[3, 1:5, 1:5] = 100
    return signal, mask


def _create_segment_labels(sample_dir):
    labels = np.zeros(SIGNAL_SHAPE, dtype=np.uint16)
    labels[:2] = 1  # C1
    labels[2:] = 2  # C2
    return _create_segments_zarr(
        sample_dir / "spinalj_segments.zarr",
        SIGNAL_SHAPE,
        attrs={"label_kind": "vertebral_segments"},
        values=labels,
    )


def test_run_spinal_segment_analysis_end_to_end(sample_dir):
    legend_csv = _write_legend(sample_dir / "segment_id_legend.csv", [(1, "C1"), (2, "C2")])
    signal, mask = _build_signal_and_mask(sample_dir)
    _create_segment_labels(sample_dir)

    summary = run_spinal_segment_analysis(
        sample_dir=sample_dir,
        signal_ch="0",
        signal_zarr_path=signal,
        mask_zarr_path=mask,
        resolution_xyz=(1.0, 1.0, 2.0),
        min_voxels=5,
        max_voxels=12,
    )

    assert summary["segment_count"] == 2
    assert summary["legend_csv"] == str(legend_csv)

    sheets = pd.read_excel(summary["output_excel"], sheet_name=None)
    assert set(sheets) == {"Level_0", "Level_1"}

    c1 = sheets["Level_1"][sheets["Level_1"]["Name"] == "C1"].iloc[0]
    assert int(c1["Signal Count"]) == 1
    assert int(c1["Signal Voxels"]) == 9
    assert int(c1["Total Voxels"]) == 128
    assert float(c1["Signal Volume (um3)"]) == pytest.approx(18.0)

    c2 = sheets["Level_1"][sheets["Level_1"]["Name"] == "C2"].iloc[0]
    assert int(c2["Signal Count"]) == 0

    root = sheets["Level_0"].iloc[0]
    assert int(root["Signal Count"]) == 1

    flat = pd.read_csv(summary["output_csv"])
    assert {"Level", "Name", "Signal Count"} <= set(flat.columns)
    assert len(flat) == 3  # root + C1 + C2

    assert (sample_dir / "results" / "_run_manifest.json").exists()


def test_run_spinal_segment_analysis_upsamples_midres(sample_dir):
    _write_legend(sample_dir / "segment_id_legend.csv", [(1, "C1"), (2, "C2")])
    signal, mask = _build_signal_and_mask(sample_dir)

    mid_labels = np.zeros((2, 4, 4), dtype=np.uint16)
    mid_labels[0] = 1  # C1
    mid_labels[1] = 2  # C2
    _create_segments_zarr(
        sample_dir / "spinalj_segments.zarr",
        (2, 4, 4),
        attrs={"native_shape_zyx": list(SIGNAL_SHAPE), "z_step": 2, "label_kind": "vertebral_segments"},
        values=mid_labels,
    )

    summary = run_spinal_segment_analysis(
        sample_dir=sample_dir,
        signal_ch="0",
        signal_zarr_path=signal,
        mask_zarr_path=mask,
        resolution_xyz=(1.0, 1.0, 2.0),
        min_voxels=5,
        max_voxels=12,
    )

    upsampled = sample_dir / "spinalj_segments_upsampled.zarr"
    assert upsampled.exists()
    assert summary["segments_zarr"] == str(upsampled)

    sheets = pd.read_excel(summary["output_excel"], sheet_name=None)
    c1 = sheets["Level_1"][sheets["Level_1"]["Name"] == "C1"].iloc[0]
    assert int(c1["Signal Count"]) == 1
    assert int(c1["Total Voxels"]) == 128
