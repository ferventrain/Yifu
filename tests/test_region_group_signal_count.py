from __future__ import annotations

import json

import pandas as pd
import pytest

from pipeline_modules.visualization.region_group_signal_count import (
    build_region_group_signal_count_table,
    find_density_excel,
    load_region_groups,
)


def write_region_csv(tmp_path):
    rows = [
        {
            "id": 1,
            "name": "Frontal pole, cerebral cortex,FRP",
            "acronym": "['cerebral cortex', 'FRP']",
            "structure_id_path": "[997, 8, 1]",
        },
        {
            "id": 2,
            "name": "Anterior cingulate area,ACA",
            "acronym": "['ACA']",
            "structure_id_path": "[997, 8, 2]",
        },
        {
            "id": 3,
            "name": "Prelimbic area,PL",
            "acronym": "['PL']",
            "structure_id_path": "[997, 8, 3]",
        },
    ]
    csv_path = tmp_path / "regions.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    return csv_path


def write_density_excel(sample_dir):
    excel_path = sample_dir / f"{sample_dir.name}_result.xlsx"
    with pd.ExcelWriter(excel_path) as writer:
        pd.DataFrame(
            {
                "Name": [
                    "Frontal pole, cerebral cortex,FRP",
                    "Anterior cingulate area,ACA",
                ],
                "Signal Count": [10, 20],
            }
        ).to_excel(writer, sheet_name="Level_7", index=False)
        pd.DataFrame(
            {
                "Name": ["Prelimbic area,PL"],
                "Signal Count": [5],
            }
        ).to_excel(writer, sheet_name="Level_6", index=False)
        pd.DataFrame({"Name": ["ignored"], "Signal Count": [999]}).to_excel(
            writer,
            sheet_name="Notes",
            index=False,
        )
    return excel_path


def write_groups_json(tmp_path):
    groups_path = tmp_path / "groups.json"
    groups_path.write_text(
        json.dumps(
            {
                "PFC": {"acronyms": ["FRP", "ACA", "PL"], "color": "#fff"},
                "ACC": ["ACA"],
                "Missing": {"acronyms": ["NOPE"]},
            }
        ),
        encoding="utf-8",
    )
    return groups_path


def test_load_region_groups_accepts_list_and_object_specs(tmp_path):
    groups_path = write_groups_json(tmp_path)

    groups = load_region_groups(groups_path)

    assert groups["PFC"] == ["FRP", "ACA", "PL"]
    assert groups["ACC"] == ["ACA"]


def test_build_region_group_signal_count_table_sums_acronyms(tmp_path):
    sample_dir = tmp_path / "sample_a"
    sample_dir.mkdir()
    excel_path = write_density_excel(sample_dir)
    cfg_path = write_region_csv(tmp_path)
    groups_path = write_groups_json(tmp_path)

    table = build_region_group_signal_count_table(
        excel_path,
        groups_json=groups_path,
        cfg=cfg_path,
        warn_missing=False,
    )

    by_group = table.set_index("group")
    assert by_group.loc["PFC", "signal_count"] == pytest.approx(35)
    assert by_group.loc["ACC", "signal_count"] == pytest.approx(20)
    assert by_group.loc["Missing", "signal_count"] == pytest.approx(0)
    assert by_group.loc["Missing", "missing_acronyms"] == "NOPE"
    assert by_group.loc["PFC", "sample"] == "sample_a"


def test_find_density_excel_prefers_sample_named_result(tmp_path):
    sample_dir = tmp_path / "sample_b"
    sample_dir.mkdir()
    preferred = write_density_excel(sample_dir)
    other = sample_dir / "newer_density.xlsx"
    other.write_text("placeholder")

    assert find_density_excel(sample_dir) == preferred
