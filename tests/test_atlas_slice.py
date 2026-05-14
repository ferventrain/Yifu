from __future__ import annotations

import json
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import tifffile
from PIL import Image

from pipeline_modules.visualization.atlas_slice import (
    AtlasSliceSpec,
    build_atlas_slice_heatmap,
    coordinate_to_index,
    extract_atlas_slice,
    render_atlas_slice,
    render_atlas_slice_heatmap,
)


def write_synthetic_label(path: Path) -> np.ndarray:
    volume = np.zeros((10, 12, 14), dtype=np.uint16)
    volume[2:8, 3:10, 4:11] = 1
    volume[2:5, 3:7, 4:8] = 2
    volume[5:8, 7:10, 8:11] = 3
    tifffile.imwrite(str(path), volume)
    return volume


def write_region_cfg(path: Path) -> None:
    path.write_text(
        "id,name,acronym,graph_id,rgb_triplet,structure_id_path,structure_set_ids\n"
        "1,Region One,R1,1,\"[255,255,255]\",\"[1]\",\"[]\"\n"
        "2,Region Two,R2,1,\"[255,0,0]\",\"[1,2]\",\"[]\"\n"
        "3,Region Three,R3,1,\"[0,0,0]\",\"[1,3]\",\"[]\"\n",
        encoding="utf-8",
    )


def write_metric_excel(path: Path) -> None:
    with pd.ExcelWriter(path) as writer:
        pd.DataFrame(
            {
                "Name": ["Region One", "Region Three"],
                "Voxel Density": [0.1, 0.9],
            }
        ).to_excel(writer, sheet_name="Level_1", index=False)


def test_coordinate_to_index_supports_bregma_ccf_and_index():
    shape = (10, 12, 14)
    bregma = (2, 6, 7)

    assert coordinate_to_index(
        AtlasSliceSpec("coronal", "bregma-mm", -0.05, bregma_index=bregma),
        shape,
    ) == 8
    assert coordinate_to_index(
        AtlasSliceSpec("sagittal", "bregma-mm", 0.05, bregma_index=bregma),
        shape,
    ) == 9
    assert coordinate_to_index(
        AtlasSliceSpec("horizontal", "bregma-mm", 0.05, bregma_index=bregma),
        shape,
    ) == 4
    assert coordinate_to_index(AtlasSliceSpec("coronal", "ccf-um", 75), shape) == 3
    assert coordinate_to_index(AtlasSliceSpec("sagittal", "index", 5), shape) == 5


def test_extract_atlas_slice_shapes_for_three_planes(tmp_path):
    label_path = tmp_path / "label.tiff"
    volume = write_synthetic_label(label_path)

    coronal = extract_atlas_slice(label_path, AtlasSliceSpec("coronal", "index", 5))
    sagittal = extract_atlas_slice(label_path, AtlasSliceSpec("sagittal", "index", 6))
    horizontal = extract_atlas_slice(label_path, AtlasSliceSpec("horizontal", "index", 4))

    assert coronal.image.shape == (volume.shape[0], volume.shape[2])
    assert sagittal.image.shape == (volume.shape[0], volume.shape[1])
    assert horizontal.image.shape == (volume.shape[1], volume.shape[2])
    assert coronal.x_axis == "ML"
    assert coronal.y_axis == "DV"
    assert sagittal.x_axis == "AP"
    assert horizontal.y_axis == "AP"


def test_coordinate_to_index_raises_clear_out_of_bounds_error():
    with pytest.raises(ValueError, match="plane=coronal.*index=46.*shape=\\(10, 12, 14\\)"):
        coordinate_to_index(
            AtlasSliceSpec("coronal", "bregma-mm", -1.0, bregma_index=(2, 6, 7)),
            (10, 12, 14),
        )


def test_render_atlas_slice_writes_white_background_black_boundaries(tmp_path):
    label_path = tmp_path / "label.tiff"
    write_synthetic_label(label_path)
    atlas_slice = extract_atlas_slice(label_path, AtlasSliceSpec("coronal", "index", 5))
    output_path = render_atlas_slice(atlas_slice, tmp_path / "slice.png", dpi=80)

    assert output_path.exists()
    assert output_path.stat().st_size > 0

    rendered = np.asarray(Image.open(output_path).convert("RGB"))
    assert np.any(np.any(rendered < 245, axis=-1))
    assert np.any(np.all(rendered > 245, axis=-1))


def test_cli_smoke_writes_output_and_json(tmp_path):
    label_path = tmp_path / "label.tiff"
    write_synthetic_label(label_path)
    output_path = tmp_path / "cli_slice.png"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pipeline_modules.visualization.atlas_slice",
            "--label",
            str(label_path),
            "--plane",
            "coronal",
            "--coord-system",
            "index",
            "--coord",
            "5",
            "--output",
            str(output_path),
            "--dpi",
            "80",
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=True,
    )

    payload = json.loads(result.stdout)
    assert output_path.exists()
    assert payload["index"] == 5
    assert payload["plane"] == "coronal"


def test_render_atlas_slice_svg_native(tmp_path):
    label_path = tmp_path / "label.tiff"
    write_synthetic_label(label_path)
    atlas_slice = extract_atlas_slice(label_path, AtlasSliceSpec("coronal", "index", 5))
    output_path = render_atlas_slice(atlas_slice, tmp_path / "slice.svg")

    assert output_path.exists()
    assert output_path.stat().st_size > 0

    tree = ET.parse(output_path)
    root = tree.getroot()
    ns = {"svg": "http://www.w3.org/2000/svg"}

    rect = root.find(".svg:rect", ns)
    assert rect is not None
    assert rect.get("fill") == "white"

    paths = root.findall(".//svg:path", ns)
    assert len(paths) > 0
    for path in paths:
        assert path.get("d", "").startswith("M ")
        assert path.get("fill", "none") == "none"
        assert path.get("stroke") == "black"


def test_render_atlas_slice_svg_hide_regions(tmp_path):
    label_path = tmp_path / "label.tiff"
    write_synthetic_label(label_path)
    atlas_slice = extract_atlas_slice(label_path, AtlasSliceSpec("coronal", "index", 5))
    output_path = render_atlas_slice(atlas_slice, tmp_path / "no_regions.svg", show_regions=False)

    ns = {"svg": "http://www.w3.org/2000/svg"}
    tree = ET.parse(output_path)
    paths = tree.getroot().findall(".//svg:path", ns)
    assert len(paths) > 0


def test_cli_smoke_svg_output(tmp_path):
    label_path = tmp_path / "label.tiff"
    write_synthetic_label(label_path)
    output_path = tmp_path / "cli_slice.svg"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pipeline_modules.visualization.atlas_slice",
            "--label",
            str(label_path),
            "--plane",
            "coronal",
            "--coord-system",
            "index",
            "--coord",
            "5",
            "--output",
            str(output_path),
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=True,
    )

    payload = json.loads(result.stdout)
    assert output_path.exists()
    assert payload["index"] == 5
    assert output_path.stat().st_size > 0


def test_render_atlas_slice_heatmap_svg(tmp_path):
    label_path = tmp_path / "label.tiff"
    cfg_path = tmp_path / "regions.csv"
    excel_path = tmp_path / "metrics.xlsx"
    output_path = tmp_path / "heatmap.svg"

    write_synthetic_label(label_path)
    write_region_cfg(cfg_path)
    write_metric_excel(excel_path)

    atlas_slice = extract_atlas_slice(label_path, AtlasSliceSpec("coronal", "index", 5))
    atlas_heatmap = build_atlas_slice_heatmap(
        atlas_slice,
        input_excel=excel_path,
        cfg_path=cfg_path,
        metric="Voxel Density",
        vmin=0.0,
        vmax=1.0,
    )
    render_atlas_slice_heatmap(atlas_heatmap, output_path)

    ns = {"svg": "http://www.w3.org/2000/svg"}
    tree = ET.parse(output_path)
    root = tree.getroot()
    paths = root.findall(".//svg:path", ns)
    rects = root.findall(".//svg:rect", ns)
    texts = [node.text for node in root.findall(".//svg:text", ns)]

    assert output_path.exists()
    assert atlas_heatmap.region_values[2] == pytest.approx(0.1)
    assert atlas_heatmap.region_values[3] == pytest.approx(0.9)
    assert any(path.get("fill") not in {None, "none"} for path in paths)
    assert any(rect.get("fill", "").startswith("url(#colorbar-gradient)") for rect in rects)
    assert "Voxel Density" in texts
