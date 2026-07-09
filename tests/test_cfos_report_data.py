from __future__ import annotations

import json

import pandas as pd
import pytest

from pipeline_modules.visualization.cfos_report_data import (
    aggregate_metrics_to_coarse_regions,
    build_region_tree_nodes,
    build_report_bundle,
    choose_default_level,
    collect_subtree_region_ids,
    compute_leaf_region_stats,
    compute_overview,
    compute_system_metrics,
    export_region_metrics_csv,
    load_region_metadata_table,
    normalize_region_metrics,
    read_whole_brain_stats_from_excel,
    VOXEL_VOLUME_UM3,
)


def write_region_csv(tmp_path):
    rows = [
        {
            "id": 997,
            "name": "root",
            "acronym": "['root']",
            "structure_id_path": "[997]",
        },
        {
            "id": 315,
            "name": "Isocortex,ISO",
            "acronym": "['ISO']",
            "structure_id_path": "[997, 315]",
        },
        {
            "id": 512,
            "name": "Cerebellum,CB",
            "acronym": "['CB']",
            "structure_id_path": "[997, 512]",
        },
        {
            "id": 1,
            "name": "Frontal pole, cerebral cortex,FRP",
            "acronym": "['FRP']",
            "structure_id_path": "[997, 315, 1]",
        },
    ]
    csv_path = tmp_path / "regions.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    return csv_path


def write_density_excel(sample_dir):
    excel_path = sample_dir / "results" / f"{sample_dir.name}_ch1_brain_distribution_stats.xlsx"
    excel_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(excel_path) as writer:
        pd.DataFrame(
            {
                "Name": ["root,root"],
                "Total Voxels": [2000],
                "Signal Voxels": [300],
                "Voxel Density": [0.15],
                "Signal Count": [55],
                "Sum Intensity": [5500],
                "Left Signal Count": [30],
                "Right Signal Count": [25],
            }
        ).to_excel(writer, sheet_name="Level_0", index=False)
        pd.DataFrame(
            {
                "Name": [
                    "Isocortex,ISO",
                    "Cerebellum,CB",
                    "Frontal pole, cerebral cortex,FRP",
                ],
                "Total Voxels": [1000, 500, 100],
                "Signal Voxels": [100, 50, 20],
                "Voxel Density": [0.1, 0.1, 0.2],
                "Signal Count": [40, 10, 5],
                "Sum Intensity": [4000, 500, 250],
                "Left Signal Count": [20, 4, 3],
                "Right Signal Count": [20, 6, 2],
                "Left Voxel Density": [0.08, 0.08, 0.15],
                "Right Voxel Density": [0.12, 0.12, 0.25],
            }
        ).to_excel(writer, sheet_name="Level_2", index=False)
        pd.DataFrame(
            {
                "Name": ["Frontal pole, cerebral cortex,FRP"],
                "Total Voxels": [100],
                "Signal Voxels": [20],
                "Voxel Density": [0.2],
                "Signal Count": [5],
                "Sum Intensity": [250],
            }
        ).to_excel(writer, sheet_name="Level_6", index=False)
    return excel_path


def write_groups_json(tmp_path):
    groups_path = tmp_path / "groups.json"
    groups_path.write_text(
        json.dumps({"PFC": {"acronyms": ["FRP"], "color": "#fff"}}),
        encoding="utf-8",
    )
    return groups_path


def test_read_whole_brain_stats_from_excel_root_row(tmp_path):
    sample_dir = tmp_path / "mouse01"
    sample_dir.mkdir()
    excel = write_density_excel(sample_dir)
    stats = read_whole_brain_stats_from_excel(excel)
    assert stats["total_cfos_count"] == 55
    assert stats["signal_voxels"] == 300
    assert stats["total_region_volume_voxels"] == 2000
    assert stats["signal_volume_um3"] == pytest.approx(300 * VOXEL_VOLUME_UM3)
    assert stats["brain_volume_um3"] == pytest.approx(2000 * VOXEL_VOLUME_UM3)
    assert stats["whole_brain_count_laterality_index"] == pytest.approx((25 - 30) / 55)


def test_normalize_region_metrics_and_laterality(tmp_path):
    cfg = write_region_csv(tmp_path)
    sample_dir = tmp_path / "mouse01"
    sample_dir.mkdir()
    excel = write_density_excel(sample_dir)

    metrics = normalize_region_metrics(excel, sample_id="mouse01", cfg_path=cfg)
    level2 = [row for row in metrics if row["level"] == "Level_2"]
    frp = next(row for row in level2 if row["region_acronym"] == "FRP")

    assert frp["cfos_count"] == 5
    assert frp["mean_cfos_intensity"] == pytest.approx(12.5)
    assert frp["count_laterality_index"] == pytest.approx((2 - 3) / 5)
    assert frp["rank_by_count"] == 3


def test_build_report_bundle_contains_mvp_sections(tmp_path):
    cfg = write_region_csv(tmp_path)
    groups = write_groups_json(tmp_path)
    sample_dir = tmp_path / "mouse01"
    sample_dir.mkdir()
    write_density_excel(sample_dir)

    bundle = build_report_bundle(
        sample_dir,
        cfg_path=cfg,
        groups_json=groups,
        signal_ch="ch1",
        group_label="control",
    )

    assert bundle["sample"]["sample_id"] == "mouse01"
    assert bundle["sample"]["group"] == "control"
    assert bundle["sample"]["sample_dir"] == str(sample_dir.resolve())
    assert bundle["overview"]["total_cfos_count"] > 0
    assert bundle["summary"]["schema_version"] == "1"
    assert bundle["summary"]["headline_stats"]["total_cfos_count"] == 55
    assert bundle["summary"]["headline_stats"]["source"] == "excel_level_0_root"
    assert bundle["region_tree"]
    assert bundle["system_metrics"]
    assert any(item["id"] == "hotspot_clustering" for item in bundle["unavailable_modules"])
    assert bundle["parameters"]["default_level"] == "Level_6"


def test_export_region_metrics_csv(tmp_path):
    cfg = write_region_csv(tmp_path)
    sample_dir = tmp_path / "mouse01"
    sample_dir.mkdir()
    excel = write_density_excel(sample_dir)
    metrics = normalize_region_metrics(excel, sample_id="mouse01", cfg_path=cfg)
    level = choose_default_level(metrics)

    csv_text = export_region_metrics_csv(
        metrics,
        region_ids=[1],
        level=level,
        sample_id="mouse01",
        group="control",
        source_paths={"density_excel": str(excel)},
    )
    assert "sample_id" in csv_text
    assert "mouse01" in csv_text
    assert "FRP" in csv_text or "Frontal pole" in csv_text


def test_build_region_tree_nodes_uses_allen_roots(tmp_path):
    cfg = write_region_csv(tmp_path)
    region_table = load_region_metadata_table(cfg)
    tree = build_region_tree_nodes(region_table)
    assert len(tree) == 1
    assert tree[0]["region_id"] == 997
    child_ids = {child["region_id"] for child in tree[0]["children"]}
    assert child_ids == {315, 512}


def test_compute_leaf_region_stats_uses_finest_structural_leaves(tmp_path):
    cfg = write_region_csv(tmp_path)
    sample_dir = tmp_path / "mouse01"
    sample_dir.mkdir()
    excel = write_density_excel(sample_dir)
    metrics = normalize_region_metrics(excel, sample_id="mouse01", cfg_path=cfg)

    leaf_stats = compute_leaf_region_stats(metrics, cfg_path=cfg)
    assert leaf_stats["scope"] == "all_levels_finest_available"
    assert leaf_stats["total_region_count"] == 2
    assert leaf_stats["activated_region_count"] == 2


def test_aggregate_metrics_to_coarse_regions_uses_path_prefix(tmp_path):
    cfg = write_region_csv(tmp_path)
    lookup = load_region_metadata_table(cfg)
    metrics = [
        {
            "sample_id": "mouse01",
            "region_id": 1,
            "region_name": "Frontal pole",
            "region_acronym": "FRP",
            "structure_id_path": [997, 315, 1],
            "level": "Level_6",
            "cfos_count": 10.0,
            "signal_voxels": 20.0,
            "region_volume_voxels": 100.0,
            "sum_intensity": 100.0,
        },
        {
            "sample_id": "mouse01",
            "region_id": 512,
            "region_name": "Cerebellum",
            "region_acronym": "CB",
            "structure_id_path": [997, 512],
            "level": "Level_6",
            "cfos_count": 5.0,
            "signal_voxels": 8.0,
            "region_volume_voxels": 50.0,
            "sum_intensity": 40.0,
        },
    ]
    aggregated = aggregate_metrics_to_coarse_regions(metrics, lookup)
    by_id = {row["region_id"]: row for row in aggregated}
    assert by_id[315]["cfos_count"] == pytest.approx(10.0)
    assert by_id[512]["cfos_count"] == pytest.approx(5.0)


def test_compute_system_metrics_reads_coarse_rows_from_excel(tmp_path):
    cfg = write_region_csv(tmp_path)
    sample_dir = tmp_path / "mouse01"
    sample_dir.mkdir()
    excel_path = sample_dir / "results" / f"{sample_dir.name}_ch1_brain_distribution_stats.xlsx"
    excel_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(excel_path) as writer:
        pd.DataFrame(
            {
                "Name": ["root,root"],
                "Total Voxels": [2000],
                "Signal Voxels": [300],
                "Voxel Density": [0.15],
                "Signal Count": [55],
            }
        ).to_excel(writer, sheet_name="Level_0", index=False)
        pd.DataFrame(
            {
                "Name": ["Isocortex,ISO", "Cerebellum,CB"],
                "Total Voxels": [1000, 500],
                "Signal Voxels": [100, 50],
                "Voxel Density": [0.1, 0.1],
                "Signal Count": [40, 10],
            }
        ).to_excel(writer, sheet_name="Level_2", index=False)
        pd.DataFrame(
            {
                "Name": ["Frontal pole, cerebral cortex,FRP"],
                "Total Voxels": [100],
                "Signal Voxels": [20],
                "Voxel Density": [0.2],
                "Signal Count": [5],
            }
        ).to_excel(writer, sheet_name="Level_6", index=False)

    metrics = normalize_region_metrics(excel_path, sample_id="mouse01", cfg_path=cfg)
    systems = compute_system_metrics(excel_path, metrics, cfg_path=cfg)
    by_acronym = {row["system_acronym"]: row for row in systems}
    assert by_acronym["ISO"]["system_cfos_count"] == pytest.approx(40.0)
    assert by_acronym["CB"]["system_cfos_count"] == pytest.approx(10.0)


def test_compute_overview_and_system_metrics(tmp_path):
    cfg = write_region_csv(tmp_path)
    sample_dir = tmp_path / "mouse01"
    sample_dir.mkdir()
    excel = write_density_excel(sample_dir)
    metrics = normalize_region_metrics(excel, sample_id="mouse01", cfg_path=cfg)
    level = "Level_2"

    overview = compute_overview(metrics, level=level)
    assert overview["activated_region_count"] == 3
    assert overview["whole_brain_count_laterality_index"] is not None

    systems = compute_system_metrics(excel, metrics, level=level, cfg_path=cfg)
    assert any(item["system_name"] == "Isocortex" for item in systems)
    assert any(item["system_name"] == "Cerebellum" for item in systems)
    assert all(item["source"] == "coarse_allen_region" for item in systems)
    assert len(systems) >= 2


def test_collect_subtree_region_ids(tmp_path):
    cfg = write_region_csv(tmp_path)
    subtree = collect_subtree_region_ids(315, cfg)
    assert subtree == frozenset({315, 1})
    assert collect_subtree_region_ids(512, cfg) == frozenset({512})
