from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import tifffile

from pipeline_modules.visualization.atlas_slice import index_to_bregma_mm
from pipeline_modules.visualization.cfos_report_spatial import (
    build_axis_histogram,
    build_brain_outline_surface_payload,
    build_points_viewer_payload,
    build_region_slice_focus_payload,
    build_region_surface_payload,
    build_spatial_payload,
    load_points_frame,
    slice_linkage_for_bin,
)


def test_load_points_frame_from_grid_columns(tmp_path):
    csv_path = tmp_path / "points.csv"
    pd.DataFrame(
        {
            "grid_x": [10, 20],
            "grid_y": [30, 40],
            "grid_z": [50, 60],
        }
    ).to_csv(csv_path, index=False)

    frame = load_points_frame(csv_path, "points_csv")
    assert len(frame) == 2
    assert frame.iloc[0]["index_ap"] == pytest.approx(10)
    assert frame.iloc[0]["index_dv"] == pytest.approx(30)
    assert frame.iloc[0]["index_ml"] == pytest.approx(50)


def test_build_axis_histogram_from_points():
    frame = pd.DataFrame(
        {
            "index_ap": [10, 10, 11, 50],
            "index_dv": [5, 6, 7, 8],
            "index_ml": [100, 101, 102, 103],
        }
    )
    histogram = build_axis_histogram(
        axis="AP",
        points_frame=frame,
        atlas_shape=(456, 528, 320),
        bins=8,
    )
    assert histogram["axis"] == "AP"
    assert histogram["plane"] == "coronal"
    assert len(histogram["counts"]) == 8
    assert histogram["total"] == 4


def test_build_spatial_payload_from_volume(tmp_path):
    volume = np.zeros((8, 10, 6), dtype=np.uint8)
    volume[2:6, 3:8, 2:5] = 1
    volume_path = tmp_path / "heatmap_volume.tiff"
    tifffile.imwrite(str(volume_path), volume)

    payload = build_spatial_payload(
        {"atlas_volume_tiff": str(volume_path)},
        bins=4,
        max_points=1000,
    )
    assert payload["available"] is True
    assert payload["source_kind"] == "atlas_volume_tiff"
    assert payload["histogram_source_kind"] == "atlas_volume_tiff"
    assert payload["measure"] == "cfos_count"
    assert payload["axes"]["AP"]["source"] == "volume"
    assert len(payload["axes"]["DV"]["counts"]) == 4
    assert payload["axes"]["AP"]["bin_centers_bregma_mm"]


def test_build_points_viewer_payload(tmp_path):
    csv_path = tmp_path / "points.csv"
    pd.DataFrame(
        {
            "grid_x": [1, 2, 3],
            "grid_y": [4, 5, 6],
            "grid_z": [7, 8, 9],
        }
    ).to_csv(csv_path, index=False)

    payload = build_points_viewer_payload({"points_csv": str(csv_path)}, max_points=10)
    assert payload["available"] is True
    assert payload["display_count"] == 3
    assert payload["points"][0]["ap"] == pytest.approx(1)
    assert "region_id" in payload["points"][0]


def test_build_region_surface_payload(tmp_path):
    labels = np.zeros((12, 14, 10), dtype=np.uint16)
    labels[2:10, 3:11, 2:8] = 315
    atlas_path = tmp_path / "atlas_label.tiff"
    tifffile.imwrite(str(atlas_path), labels)

    payload = build_region_surface_payload(
        atlas_label=atlas_path,
        region_ids=frozenset({315}),
        stride=1,
    )
    assert payload["available"] is True
    assert payload["vertex_count"] > 0
    assert payload["face_count"] > 0
    assert payload["region_ids"] == [315]
    assert "dv" in payload["vertices"][0]
    assert len(payload["faces"][0]) == 3


def test_build_brain_outline_surface_payload(tmp_path):
    labels = np.zeros((12, 14, 10), dtype=np.uint16)
    labels[2:10, 3:11, 2:8] = 315
    atlas_path = tmp_path / "atlas_label.tiff"
    tifffile.imwrite(str(atlas_path), labels)

    payload = build_brain_outline_surface_payload(atlas_label=atlas_path, stride=1)
    assert payload["available"] is True
    assert payload["kind"] == "brain_outline"
    assert payload["vertex_count"] > 0
    assert payload["atlas_resolution_um_dv_ap_ml"] == [25.0, 25.0, 25.0]


def test_slice_linkage_for_bin():
    linkage = slice_linkage_for_bin("AP", 216.5)
    assert linkage["plane"] == "coronal"
    assert linkage["coordinate"] == pytest.approx(216.5)


def test_index_to_bregma_mm_ap():
    assert index_to_bregma_mm("AP", 216, bregma_index=(18, 216, 228), resolution_um=25.0) == pytest.approx(0.0)
    assert index_to_bregma_mm("AP", 196, bregma_index=(18, 216, 228), resolution_um=25.0) == pytest.approx(0.5)


def test_build_region_slice_focus_payload_float32_labels(tmp_path):
    labels = np.zeros((12, 14, 10), dtype=np.float32)
    labels[2:10, 4:10, 2:8] = 315.0
    atlas_path = tmp_path / "atlas_label_float.tiff"
    tifffile.imwrite(str(atlas_path), labels)

    payload = build_region_slice_focus_payload(
        frozenset({315}),
        atlas_label=atlas_path,
        bregma_index=(18, 216, 228),
        resolution_um=25.0,
    )
    assert payload["available"] is True
    assert payload["recommended_index_ap"] == pytest.approx(6, abs=1.0)


def test_build_region_slice_focus_payload(tmp_path):
    labels = np.zeros((12, 14, 10), dtype=np.uint16)
    labels[2:10, 4:10, 2:8] = 315
    atlas_path = tmp_path / "atlas_label.tiff"
    tifffile.imwrite(str(atlas_path), labels)

    payload = build_region_slice_focus_payload(
        frozenset({315}),
        atlas_label=atlas_path,
        bregma_index=(18, 216, 228),
        resolution_um=25.0,
    )
    assert payload["available"] is True
    assert payload["recommended_plane"] == "coronal"
    assert payload["recommended_index_ap"] == pytest.approx(6, abs=1.0)
    assert "bregma_mm_ap" in payload
