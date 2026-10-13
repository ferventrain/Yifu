from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pipeline_modules.visualization.cfos_standard_report import (
    DEFAULT_PARCELLATION_CSV,
    _block_sum,
    build_comparison,
    coarse_table,
    load_density_fraction_grid,
    parcellation_table,
    rank_top_regions,
    system_of_regions,
)


def test_block_sum_pools_correct_axes():
    volume = np.arange(2 * 4 * 2 * 4 * 2 * 4, dtype=np.float32).reshape(2, 4, 2, 4, 2, 4).transpose(0, 1, 2, 3, 4, 5)
    grid = np.zeros((8, 8, 8), dtype=np.float32)
    for i in range(8):
        for j in range(8):
            for k in range(8):
                grid[i, j, k] = i * 100 + j * 10 + k
    summed = _block_sum(grid, 4)
    assert summed.shape == (2, 2, 2)
    # block (0,0,0) covers grid[0:4, 0:4, 0:4]
    assert summed[0, 0, 0] == pytest.approx(grid[0:4, 0:4, 0:4].sum())
    assert summed[1, 1, 1] == pytest.approx(grid[4:8, 4:8, 4:8].sum())
    assert summed[0, 1, 0] == pytest.approx(grid[0:4, 4:8, 0:4].sum())


def test_parcellation_table_remainder_math_count_and_volume():
    parcellation = [
        {"name": "Parent,P", "acronym": "P", "id": 1, "level": 3, "remainder_children": ["Child C,CC", "Child D,DD"]},
        {"name": "Child C,CC", "acronym": "CC", "id": 2, "level": 4, "remainder_children": None},
        {"name": "Child D,DD", "acronym": "DD", "id": 3, "level": 4, "remainder_children": None},
    ]
    metrics = {
        "Parent,P": {"count": 100.0, "volume_voxels": 1000.0},
        "Child C,CC": {"count": 30.0, "volume_voxels": 300.0},
        "Child D,DD": {"count": 50.0, "volume_voxels": 500.0},
    }
    table = parcellation_table(metrics, voxel_um3=8.0, parcellation=parcellation)
    by_name = table.set_index("region_name")
    parent = by_name.loc["Parent,P"]
    assert parent["count"] == 100.0 - 30.0 - 50.0
    assert parent["volume_mm3"] == pytest.approx((1000.0 - 300.0 - 500.0) * 8.0 / 1e9)
    assert parent["density_cells_per_mm3"] == pytest.approx(20.0 / (200.0 * 8.0 / 1e9))
    assert by_name.loc["Child C,CC"]["count"] == 30.0
    # exclusive region set: no duplicates, no parent-child double counting
    assert table["region_name"].is_unique


def test_build_comparison_ranking_and_flags():
    table_a = pd.DataFrame({
        "region_name": ["R1,R1", "R2,R2", "R3,R3"],
        "acronym": ["R1", "R2", "R3"],
        "region_id": [1, 2, 3],
        "count": [10.0, 200.0, 5.0],
        "volume_mm3": [1.0, 1.0, 1e-4],
        "density_cells_per_mm3": [10.0, 200.0, 50000.0],
        "is_remainder": [False, False, False],
    })
    table_b = table_a.copy()
    table_b["count"] = [110.0, 100.0, 5.0]
    table_b["density_cells_per_mm3"] = [110.0, 100.0, 50000.0]
    comparison = build_comparison(table_a, table_b, min_volume_mm3=0.005, low_count_warning=20.0)
    assert comparison["acronym"].tolist() == ["R1", "R2", "R3"]
    r1 = comparison.set_index("acronym").loc["R1"]
    assert r1["delta_density"] == pytest.approx(110.0 - 10.0)
    assert bool(r1["qc_usable"]) is True
    assert bool(comparison.set_index("acronym").loc["R3"]["qc_usable"]) is False
    assert bool(comparison.set_index("acronym").loc["R1"]["low_count"]) is True  # count_a=10 < 20
    top = rank_top_regions(comparison, top_n=2)
    assert top["acronym"].tolist() == ["R1", "R2"]  # |Δ|=100 both; tie -> name order, tiny R3 excluded
    assert top["delta_density"].iloc[0] == pytest.approx(100.0)


def test_density_fraction_grid_capped_and_binned(tmp_path):
    import tifffile

    counts = np.zeros((8, 8, 8), dtype=np.float32)
    counts[0:4, 0:4, 0:4] = 4000.0  # huge -> occupancy capped at 1.0
    counts[4:8, 4:8, 4:8] = 100.0
    tif = tmp_path / "count.tif"
    tifffile.imwrite(tif, counts)
    grid = load_density_fraction_grid(tif, voxel_um3=6.48, bin_um=100.0, sigma_um=0.0)
    assert grid.shape == (2, 2, 2)
    assert grid.max() <= 1.0 + 1e-6
    # saturated block is full occupancy, sparse block = 100*6.48/15625
    assert grid[0, 0, 0] == pytest.approx(1.0)
    assert grid[1, 1, 1] == pytest.approx(100.0 * 6.48 / 15625.0, rel=1e-3)


@pytest.mark.skipif(not DEFAULT_PARCELLATION_CSV.exists(), reason="production structure CSV not available")
def test_production_parcellation_system_mapping_and_tables():
    from pipeline_modules.visualization.top_level7_density_ratio import build_default_parcellation, load_all_level_counts

    parcellation = None  # built lazily inside parcellation_table
    system_of = system_of_regions()
    assert system_of.get("Primary visual area,VISp") == "Isocortex"
    assert system_of.get("Field CA1,CA1") == "Hippocampal formation"
    assert system_of.get("Medial geniculate complex,MG") == "Thalamus"

    metrics = {
        "Primary visual area,VISp": {"count": 100.0, "volume_voxels": 1e6},
        "Field CA1,CA1": {"count": 50.0, "volume_voxels": 5e5},
        "Hippocampal formation,HPF": {"count": 400.0, "volume_voxels": 4e6},
    }
    table = parcellation_table(metrics, voxel_um3=6.48, parcellation=parcellation)
    coarse = coarse_table(metrics, 6.48)
    hp = coarse.set_index("system_en").loc["Hippocampal formation"]
    assert hp["count"] == 400.0
    assert hp["density_cells_per_mm3"] == pytest.approx(400.0 / (4e6 * 6.48 / 1e9))
    _ = load_all_level_counts  # import sanity for the paired module
