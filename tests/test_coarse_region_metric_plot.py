from __future__ import annotations

import pandas as pd
import pytest

from pipeline_modules.visualization.coarse_region_metric_plot import (
    DEFAULT_REGION_IDS,
    build_coarse_region_table,
    generate_coarse_region_outputs,
    load_level_sheets,
    load_region_names,
)


REGION_NAMES = {
    315: "Isocortex,Isocortex",
    1089: "Hippocampal formation,HPF",
    698: "Olfactory areas,OLF",
    703: "Cortical subplate,CTXsp",
    477: "Striatum,STR",
    803: "Pallidum,PAL",
    549: "Thalamus,TH",
    1097: "Hypothalamus,HY",
    313: "Midbrain,MB",
    771: "Pons,P",
    354: "Medulla,MY",
    512: "Cerebellum,CB",
    1009: "fiber tracts,fiber tracts",
    73: "ventricular systems,VS",
    1024: "grooves,grv",
    304325711: "retina,retina",
}


def write_region_csv(tmp_path):
    rows = [
        {
            "id": region_id,
            "name": name,
            "acronym": "[]",
            "graph_id": 1,
            "rgb_triplet": "[0, 0, 0]",
            "structure_id_path": f"[997, {region_id}]",
            "structure_set_ids": "[]",
        }
        for region_id, name in REGION_NAMES.items()
    ]
    csv_path = tmp_path / "regions.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    return csv_path


def write_region_excel(tmp_path):
    excel_path = tmp_path / "region_signal.xlsx"
    with pd.ExcelWriter(excel_path) as writer:
        pd.DataFrame(
            {
                "Name": [
                    "Isocortex,Isocortex",
                    "Hippocampal formation,HPF",
                    "fiber tracts,fiber tracts",
                ],
                "Voxel Density": [0.1, 0.2, 0.3],
                "Signal Voxels": [10, 20, 30],
            }
        ).to_excel(writer, sheet_name="Level_6", index=False)
        pd.DataFrame(
            {
                "Name": ["ventricular systems,VS", "grooves,grv"],
                "Voxel Density": [0.4, 99.0],
                "Signal Voxels": [40, 990],
            }
        ).to_excel(writer, sheet_name="Level_2", index=False)
        pd.DataFrame({"Name": ["ignored"], "Voxel Density": [1.0]}).to_excel(
            writer,
            sheet_name="Notes",
            index=False,
        )
    return excel_path


class TestLoadLevelSheets:
    def test_reads_all_level_sheets(self, tmp_path):
        excel_path = write_region_excel(tmp_path)

        table = load_level_sheets(excel_path)

        assert set(table["source_sheet"]) == {"Level_2", "Level_6"}
        assert "ignored" not in set(table["Name"])


class TestCoarseRegionTable:
    def test_builds_default_regions_and_excludes_grooves_retina(self, tmp_path):
        excel_path = write_region_excel(tmp_path)
        cfg_path = write_region_csv(tmp_path)
        level_table = load_level_sheets(excel_path)
        region_table = load_region_names(cfg_path)

        coarse = build_coarse_region_table(level_table, region_table, "Voxel Density", warn_missing=False)

        assert len(coarse) == 14
        assert coarse["region_id"].tolist() == DEFAULT_REGION_IDS
        assert "grooves" not in set(coarse["region_name"])
        assert "retina" not in set(coarse["region_name"])
        assert coarse.loc[coarse["region_id"] == 1009, "value"].iloc[0] == pytest.approx(0.3)
        assert coarse.loc[coarse["region_id"] == 73, "value"].iloc[0] == pytest.approx(0.4)
        assert coarse.loc[coarse["region_id"] == 698, "value"].iloc[0] == pytest.approx(0.0)

    def test_missing_metric_raises(self, tmp_path):
        excel_path = write_region_excel(tmp_path)
        cfg_path = write_region_csv(tmp_path)

        with pytest.raises(ValueError, match="Metric column not found"):
            build_coarse_region_table(
                load_level_sheets(excel_path),
                load_region_names(cfg_path),
                "Missing Metric",
                warn_missing=False,
            )


class TestGenerateCoarseRegionOutputs:
    def test_writes_tables_and_two_plots(self, tmp_path):
        excel_path = write_region_excel(tmp_path)
        cfg_path = write_region_csv(tmp_path)
        output_prefix = tmp_path / "result"

        outputs = generate_coarse_region_outputs(
            input_excel=excel_path,
            cfg=cfg_path,
            metric="Voxel Density",
            title="Regional Density",
            ylabel=None,
            output_prefix=output_prefix,
            dpi=80,
        )

        for key in ("csv_path", "xlsx_path", "atlas_plot_path", "sorted_plot_path"):
            path = outputs[key]
            assert path.exists()
            assert path.stat().st_size > 0

        output_table = pd.read_csv(outputs["csv_path"])
        assert len(output_table) == 14
        assert output_table.loc[0, "region_name"] == "Isocortex"

    def test_passes_custom_title_and_ylabel_to_plots(self, tmp_path, monkeypatch):
        excel_path = write_region_excel(tmp_path)
        cfg_path = write_region_csv(tmp_path)
        calls = []

        def fake_plot(coarse_table, *, output_path, title, ylabel, figsize, dpi):
            calls.append({"title": title, "ylabel": ylabel, "output_path": output_path})
            output_path.write_text("fake image")
            return output_path

        monkeypatch.setattr(
            "pipeline_modules.visualization.coarse_region_metric_plot.plot_coarse_region_bar",
            fake_plot,
        )

        generate_coarse_region_outputs(
            input_excel=excel_path,
            cfg=cfg_path,
            metric="Signal Voxels",
            title="Custom Title",
            ylabel="Custom Y",
            output_prefix=tmp_path / "custom",
        )

        assert [call["title"] for call in calls] == ["Custom Title", "Custom Title (sorted)"]
        assert [call["ylabel"] for call in calls] == ["Custom Y", "Custom Y"]
