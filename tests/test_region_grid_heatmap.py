"""Hermetic tests for region_grid_heatmap: grouping, density math, packing, render."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from pipeline_modules.visualization.region_grid_heatmap import (
    collect_region_density,
    grey_group_of,
    pack_group_panels,
    render_region_grid,
    strip_common_prefix,
)
from pipeline_modules.visualization.top_level7_density_ratio import load_structure_tree

# minimal structure tree: grey > CTX > CTXpl > FRP/MOp; grey > IB > TH > VL;
# grey > IB > HY > LHA; root > fiber tracts > cc (must be excluded)
TREE_ROWS = [
    (0, "root,root", [0]),
    (1, "Basic cell groups and regions,grey", [0, 1]),
    (2, "Cerebral cortex,CTX", [0, 1, 2]),
    (3, "Neocortex,CTXpl", [0, 1, 2, 3]),
    (4, "Frontal pole, cerebral cortex,FRP", [0, 1, 2, 3, 4]),
    (5, "Primary motor area,MOp", [0, 1, 2, 3, 5]),
    (6, "Interbrain,IB", [0, 1, 6]),
    (7, "Thalamus,TH", [0, 1, 6, 7]),
    (8, "Ventrolateral thalamic nucleus,VL", [0, 1, 6, 7, 8]),
    (9, "Hypothalamus,HY", [0, 1, 6, 9]),
    (10, "Lateral hypothalamic area,LHA", [0, 1, 6, 9, 10]),
    (11, "fiber tracts,fiber tracts", [0, 11]),
    (12, "corpus callosum,cc", [0, 11, 12]),
]
REGION_NAMES = {
    4: "Frontal pole, cerebral cortex,FRP",
    5: "Primary motor area,MOp",
    8: "Ventrolateral thalamic nucleus,VL",
    10: "Lateral hypothalamic area,LHA",
    12: "corpus callosum,cc",
}


def write_tree_csv(path: Path) -> Path:
    frame = pd.DataFrame(
        [
            {"id": rid, "name": name, "structure_id_path": str(path_ids)}
            for rid, name, path_ids in TREE_ROWS
        ]
    )
    frame.to_csv(path, index=False)
    return path


def write_workbook(path: Path, frp=(10.0, 1000.0), mopa=(20.0, 1000.0), vl=(5.0, 500.0), lha=(1.0, 200.0)) -> Path:
    rows = []
    for name, (sv, tv) in zip(
        ["FRP", "MOp", "VL", "LHA", "cc"],
        [frp, mopa, vl, lha, (0.0, 50.0)],
    ):
        rows.append(
            {
                "Name": REGION_NAMES[{4: 4, 5: 5, 8: 8, 10: 10, 12: 12}[{"FRP": 4, "MOp": 5, "VL": 8, "LHA": 10, "cc": 12}[name]]],
                "Total Voxels": tv,
                "Signal Voxels": sv,
            }
        )
    pd.DataFrame(rows).to_excel(path, sheet_name="Level_5", index=False)
    return path


def test_grey_group_of_maps_major_divisions(tmp_path: Path) -> None:
    csv = write_tree_csv(tmp_path / "tree.csv")
    nodes, _root = load_structure_tree(csv)
    assert grey_group_of(4, nodes) == "Isocortex"
    assert grey_group_of(8, nodes) == "TH"
    assert grey_group_of(10, nodes) == "HY"
    assert grey_group_of(12, nodes) is None  # fiber tract excluded


def test_collect_region_density_values_and_groups(tmp_path: Path) -> None:
    csv = write_tree_csv(tmp_path / "tree.csv")
    wb_a = write_workbook(tmp_path / "a.xlsx", frp=(10.0, 1000.0), mopa=(20.0, 1000.0), lha=(0.0, 0.0))
    wb_b = write_workbook(tmp_path / "b.xlsx", frp=(30.0, 1000.0), mopa=(5.0, 1000.0), lha=(4.0, 800.0))
    rows, mat, group_sizes = collect_region_density([wb_a, wb_b], ["a", "b"], csv)

    names = [r[0] for r in rows]
    assert names == ["FRP", "MOp", "VL", "LHA"]  # cc excluded, tree order kept
    assert [r[2] for r in rows] == ["Isocortex", "Isocortex", "TH", "HY"]
    assert np.isclose(mat[0, 0], 1.0) and np.isclose(mat[0, 1], 3.0)  # FRP 10/1000, 30/1000 %
    # VL present in both, LHA only in b (NaN in a)
    assert not np.isnan(mat[2, 0]) and np.isnan(mat[3, 0]) and not np.isnan(mat[3, 1])
    # group layout rows include the 2 band rows: Isocortex 2+2, TH 1+2, HY 1+2
    assert {g: n for g, _s, _e, n in group_sizes} == {"Isocortex": 4, "TH": 3, "HY": 3}


def test_pack_group_panels_keeps_groups_whole_and_ordered() -> None:
    groups = [("Isocortex", 0, 10, 12), ("TH", 10, 15, 7), ("HY", 15, 20, 7)]
    panels = pack_group_panels(groups, max_panels=5)
    flat = [g for panel in panels for (g, _s, _e, _n) in panel]
    assert flat == ["Isocortex", "TH", "HY"]  # order preserved
    assert all(len({g for g, _s, _e, _n in panel}) == len(panel) for panel in panels)  # whole groups
    # capacity must fit the biggest group
    assert all(sum(n for _g, _s, _e, n in panel) <= 12 + 6 for panel in panels)


def test_render_region_grid_smoke(tmp_path: Path) -> None:
    csv = write_tree_csv(tmp_path / "tree.csv")
    wb_a = write_workbook(tmp_path / "a.xlsx")
    wb_b = write_workbook(tmp_path / "b.xlsx")
    rows, mat, group_sizes = collect_region_density([wb_a, wb_b], ["a", "b"], csv)
    panels = pack_group_panels(group_sizes, max_panels=5)
    png = tmp_path / "grid_heatmap.png"
    csv_out = tmp_path / "grid_density.csv"
    render_region_grid(rows, mat, ["a", "b"], panels, png, csv_out)
    assert png.exists() and png.stat().st_size > 10_000
    out = pd.read_csv(csv_out)
    assert list(out.columns) == ["acronym", "region", "group", "density_pct_a", "density_pct_b"]
    assert len(out) == 4


def test_strip_common_prefix() -> None:
    assert strip_common_prefix(["B_x_SY1", "B_x_SY2"]) == ["SY1", "SY2"]
    assert strip_common_prefix(["alpha"]) == ["alpha"]
    assert strip_common_prefix(["alpha", "beta"]) == ["alpha", "beta"]  # no common prefix
