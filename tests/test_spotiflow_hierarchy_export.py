from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from main import export_spotiflow_region_count_hierarchy, spotiflow_model_mtime


def test_spotiflow_model_mtime_uses_model_files_not_child_dirs(tmp_path: Path):
    model_dir = tmp_path / "spotiflow"
    model_dir.mkdir()
    child_dir = model_dir / "for_napari"
    child_dir.mkdir()
    for name in ("best.pt", "config.yaml", "thresholds.yaml"):
        path = model_dir / name
        path.write_text("model", encoding="utf-8")
        # Older than the child directory. Directory mtime should not force rerun.
        path.touch()

    old_time = 1_700_000_000
    child_time = 1_800_000_000
    for name in ("best.pt", "config.yaml", "thresholds.yaml"):
        path = model_dir / name
        os.utime(path, (old_time, old_time))
    os.utime(child_dir, (child_time, child_time))

    assert spotiflow_model_mtime({"model_dir": str(model_dir), "which": "best"}) == old_time


def test_export_spotiflow_region_count_hierarchy_rolls_up_counts(tmp_path: Path):
    cfg = tmp_path / "regions.csv"
    pd.DataFrame(
        [
            {
                "id": 997,
                "name": "root,root",
                "acronym": "['root']",
                "structure_id_path": "[997]",
            },
            {
                "id": 8,
                "name": "Basic cell groups and regions,grey",
                "acronym": "['grey']",
                "structure_id_path": "[997, 8]",
            },
            {
                "id": 10,
                "name": "Region A,RA",
                "acronym": "['RA']",
                "structure_id_path": "[997, 8, 10]",
            },
            {
                "id": 20,
                "name": "Region B,RB",
                "acronym": "['RB']",
                "structure_id_path": "[997, 8, 20]",
            },
            {
                "id": 30,
                "name": "Region A child,RAC",
                "acronym": "['RAC']",
                "structure_id_path": "[997, 8, 10, 30]",
            },
        ]
    ).to_csv(cfg, index=False)

    counts_csv = tmp_path / "ch3_spotiflow_region_counts.csv"
    pd.DataFrame(
        [
            {"region_id": 30, "region_name": "Region A child,RAC", "region_acronym": "RAC", "signal_count": 5},
            {"region_id": 20, "region_name": "Region B,RB", "region_acronym": "RB", "signal_count": 7},
            {"region_id": 9999, "region_name": "Missing", "region_acronym": "", "signal_count": 3},
        ]
    ).to_csv(counts_csv, index=False)

    output_excel = tmp_path / "sample_ch3_brain_distribution_stats.xlsx"
    summary = export_spotiflow_region_count_hierarchy(counts_csv, cfg, output_excel)

    assert summary["direct_region_count"] == 15
    assert summary["hierarchy_region_count"] == 12
    assert summary["unmapped_counts"] == {9999: 3}

    level_0 = pd.read_excel(output_excel, sheet_name="Level_0")
    level_1 = pd.read_excel(output_excel, sheet_name="Level_1")
    level_2 = pd.read_excel(output_excel, sheet_name="Level_2")
    level_3 = pd.read_excel(output_excel, sheet_name="Level_3")

    assert list(level_0.columns) == ["Name", "Signal Count"]
    assert level_0.loc[0, "Signal Count"] == 12
    assert level_1.loc[0, "Signal Count"] == 12
    assert dict(zip(level_2["Name"], level_2["Signal Count"])) == {
        "Region A,RA": 5,
        "Region B,RB": 7,
    }
    assert level_3.loc[0, "Signal Count"] == 5
