from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

from pipeline_modules.visualization.atlas_slice import (
    build_hemisphere_ab_ratio_lookups,
    count_region_connected_components,
    is_paired_region,
    load_region_pairing_reference,
    paint_hemisphere_ratio_slice,
    ratio_region_metric_values,
    save_region_pairing_reference,
)
from pipeline_modules.visualization.render_ab_hemisphere_ratio_heatmap import (
    render_painted_ratio_slice_png,
)


def _synthetic_pairing_volume() -> np.ndarray:
    """Region 1: two lateral blobs (paired). Region 2: one midline blob (unpaired)."""
    volume = np.zeros((6, 8, 10), dtype=np.uint16)
    # Left and right components for region 1
    volume[1:5, 2:6, 1:3] = 1
    volume[1:5, 2:6, 7:9] = 1
    # Midline single component for region 2
    volume[2:5, 3:6, 4:6] = 2
    return volume


def test_analyze_region_cc_geometry_flags_ap_split():
    from pipeline_modules.visualization.atlas_slice import analyze_region_cc_geometry

    volume = np.zeros((6, 12, 10), dtype=np.uint16)
    # True L/R pair for region 1
    volume[1:5, 4:8, 1:3] = 1
    volume[1:5, 4:8, 7:9] = 1
    # AP-split midline-ish region 2 (two blobs along AP, both near center ML)
    volume[2:5, 1:3, 4:6] = 2
    volume[2:5, 9:11, 4:6] = 2

    records = analyze_region_cc_geometry(volume, ml_mid_index=5, min_cc_voxels=5, resolution_um=25.0)
    by_id = {int(row["region_id"]): row for row in records}
    assert by_id[1]["geometry"] == "lr_pair"
    assert by_id[1]["suspect_force_unpaired"] is False
    assert by_id[2]["geometry"] == "ap_split"
    assert by_id[2]["suspect_force_unpaired"] is True


def test_count_region_connected_components_paired_vs_midline():
    volume = _synthetic_pairing_volume()
    counts = count_region_connected_components(volume, connectivity=26)
    assert counts[1] == 2
    assert counts[2] == 1
    assert is_paired_region(counts[1]) is True
    assert is_paired_region(counts[2]) is False


def test_ratio_and_hemisphere_ab_lookups_use_pairing_rules():
    a_left = {1: 10.0, 2: 4.0}
    a_right = {1: 20.0, 2: 6.0}
    b_left = {1: 5.0, 2: 1.0}
    b_right = {1: 5.0, 2: 1.0}
    n_cc = {1: 2, 2: 1}

    left, right = build_hemisphere_ab_ratio_lookups(
        a_left=a_left,
        a_right=a_right,
        b_left=b_left,
        b_right=b_right,
        n_cc_by_region_id=n_cc,
        pseudocount=1.0,
    )

    # Paired region 1: (10+1)/(5+1)=11/6, (20+1)/(5+1)=21/6
    assert left[1] == (10.0 + 1.0) / (5.0 + 1.0)
    assert right[1] == (20.0 + 1.0) / (5.0 + 1.0)
    # Unpaired region 2: whole (4+6+1)/(1+1+1)=11/3
    whole = (4.0 + 6.0 + 1.0) / (1.0 + 1.0 + 1.0)
    assert left[2] == whole
    assert right[2] == whole

    direct = ratio_region_metric_values({1: 3.0}, {1: 1.0}, pseudocount=1.0)
    assert direct[1] == (3.0 + 1.0) / (1.0 + 1.0)


def test_paint_paired_split_vs_unpaired_whole(tmp_path: Path):
    volume = _synthetic_pairing_volume()
    # Coronal-like 2D slice through AP index 4: rows=DV, cols=ML
    label_slice = volume[:, 4, :]
    ml_mid = label_slice.shape[1] // 2  # 5

    left_values = {1: 2.0, 2: 7.0}
    right_values = {1: 8.0, 2: 7.0}
    painted = paint_hemisphere_ratio_slice(
        label_slice,
        left_values,
        right_values,
        paired_by_region_id={1: True, 2: False},
        ml_mid_index=ml_mid,
    )

    left_region1 = painted[(label_slice == 1) & (np.arange(label_slice.shape[1])[None, :] < ml_mid)]
    right_region1 = painted[(label_slice == 1) & (np.arange(label_slice.shape[1])[None, :] >= ml_mid)]
    assert left_region1.size > 0 and right_region1.size > 0
    assert np.allclose(left_region1, 2.0)
    assert np.allclose(right_region1, 8.0)

    region2 = painted[label_slice == 2]
    assert region2.size > 0
    assert np.allclose(region2, 7.0)

    out = tmp_path / "ratio.png"
    render_painted_ratio_slice_png(
        painted,
        label_slice,
        out,
        vmin=0.5,
        vmax=2.0,
        vcenter=1.0,
        pixel_scale=2,
        supersample=2,
        contour_smooth=1.0,
        colorbar_label="A/B test",
    )
    assert out.exists() and out.stat().st_size > 0


def test_midplane_crossing_blob_not_geometrically_split():
    from pipeline_modules.visualization.atlas_slice import (
        find_midplane_bisected_paired_regions,
        paint_hemisphere_ratio_slice,
    )

    # One continuous bar across mid for a "paired" region.
    labels = np.zeros((8, 12), dtype=np.uint16)
    labels[2:6, 2:10] = 5
    painted = paint_hemisphere_ratio_slice(
        labels,
        {5: 2.0},
        {5: 8.0},
        {5: True},
        ml_mid_index=6,
        min_side_fraction=0.15,
    )
    vals = painted[labels == 5]
    assert vals.size > 0
    assert np.allclose(vals, 5.0)  # mean of 2 and 8; no hard L/R knife cut

    volume = np.zeros((6, 10, 12), dtype=np.uint16)
    volume[:, 3:7, 2:10] = 5
    flagged = find_midplane_bisected_paired_regions(
        volume,
        {5: True},
        ml_mid_index=6,
        min_region_slice_pixels=5,
        min_bisected_fraction=0.1,
        min_bisected_slices=1,
        min_side_fraction=0.15,
    )
    assert any(int(row["region_id"]) == 5 for row in flagged)


def test_pairing_reference_roundtrip(tmp_path: Path):
    volume = _synthetic_pairing_volume()
    n_cc = count_region_connected_components(volume)
    path = tmp_path / "region_pairing.json"
    save_region_pairing_reference(
        path,
        n_cc,
        atlas_label="synthetic.tiff",
        connectivity=26,
        name_by_region_id={1: "Paired Region", 2: "Midline Region"},
    )
    loaded_n_cc, loaded_paired = load_region_pairing_reference(path)
    assert loaded_n_cc[1] == 2
    assert loaded_n_cc[2] == 1
    assert loaded_paired[1] is True
    assert loaded_paired[2] is False

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["paired_rule"] == "n_cc >= 2"
    assert payload["regions"]["1"]["name"] == "Paired Region"


def test_build_region_pairing_cli(tmp_path: Path):
    from pipeline_modules.visualization.build_region_pairing_reference import main as build_main

    atlas_path = tmp_path / "atlas_label.tiff"
    tifffile.imwrite(str(atlas_path), _synthetic_pairing_volume())
    out = tmp_path / "pairing.json"
    rc = build_main(
        [
            "--atlas_label",
            str(atlas_path),
            "--output",
            str(out),
            "--progress_every",
            "0",
            "--cfg",
            str(tmp_path / "missing.csv"),
        ]
    )
    assert rc == 0
    n_cc, paired = load_region_pairing_reference(out)
    assert n_cc[1] == 2 and paired[1] is True
    assert n_cc[2] == 1 and paired[2] is False


def test_render_ab_cli_uses_pairing_reference(tmp_path: Path):
    from pipeline_modules.visualization.render_ab_hemisphere_ratio_heatmap import main as render_main

    atlas_path = tmp_path / "atlas_label.tiff"
    volume = _synthetic_pairing_volume()
    tifffile.imwrite(str(atlas_path), volume)

    cfg = tmp_path / "regions.csv"
    cfg.write_text(
        "id,name,acronym,graph_id,rgb_triplet,structure_id_path,structure_set_ids\n"
        "1,Region One,R1,1,\"[255,255,255]\",\"[1]\",\"[]\"\n"
        "2,Region Two,R2,1,\"[255,0,0]\",\"[1,2]\",\"[]\"\n",
        encoding="utf-8",
    )

    def write_excel(path: Path, left1: float, right1: float, left2: float, right2: float) -> None:
        with pd.ExcelWriter(path) as writer:
            pd.DataFrame(
                {
                    "Name": ["Region One", "Region Two"],
                    "Left Signal Count": [left1, left2],
                    "Right Signal Count": [right1, right2],
                }
            ).to_excel(writer, sheet_name="Level_1", index=False)

    excel_a = tmp_path / "a.xlsx"
    excel_b = tmp_path / "b.xlsx"
    write_excel(excel_a, 10, 20, 4, 6)
    write_excel(excel_b, 5, 5, 1, 1)

    pairing = tmp_path / "pairing.json"
    save_region_pairing_reference(
        pairing,
        count_region_connected_components(volume),
        atlas_label=atlas_path,
    )

    out_png = tmp_path / "ab_ratio.png"
    # AP index ~4 at bregma-mm with default resolution; use index via coordinate hack:
    # bregma_index AP=4, coordinate=0 -> index 4
    rc = render_main(
        [
            "--excel_a",
            str(excel_a),
            "--excel_b",
            str(excel_b),
            "--pairing_reference",
            str(pairing),
            "--atlas_label",
            str(atlas_path),
            "--cfg",
            str(cfg),
            "--metric",
            "signal_count",
            "--bregma_mm",
            "0",
            "--bregma_index",
            "2,4,5",
            "--output",
            str(out_png),
        ]
    )
    assert rc == 0
    assert out_png.exists() and out_png.stat().st_size > 0
    summary = json.loads(out_png.with_suffix(".json").read_text(encoding="utf-8"))
    assert summary["pairing_reference"] == str(pairing)
    assert summary["vcenter"] == 1.0
