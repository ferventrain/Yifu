from __future__ import annotations

import json

import pandas as pd
import pytest

from pipeline_modules.visualization.cfos_report_data import aggregate_region_metrics_to_level, load_region_metadata_table
from pipeline_modules.visualization.cfos_report_group_stats import (
    build_group_analysis_payload,
    build_pairwise_manifest,
    build_pairwise_scatter_payload,
    compute_differential_regions,
    load_group_manifest,
    parse_group_manifest_json,
    select_top_differential_regions,
)


def _write_region_csv(tmp_path):
    rows = [
        {"id": 997, "name": "root", "acronym": "['root']", "structure_id_path": "[997]"},
        {"id": 315, "name": "Isocortex,ISO", "acronym": "['ISO']", "structure_id_path": "[997, 315]"},
        {"id": 512, "name": "Sub1,S1", "acronym": "['S1']", "structure_id_path": "[997, 315, 512]"},
        {"id": 520, "name": "Sub2,S2", "acronym": "['S2']", "structure_id_path": "[997, 315, 520]"},
        {"id": 600, "name": "Cerebellum,CB", "acronym": "['CB']", "structure_id_path": "[997, 600]"},
    ]
    csv_path = tmp_path / "regions.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    return csv_path


def _write_sample_excel(tmp_path, sample_name: str, region_rows: list[dict]) -> str:
    sample_dir = tmp_path / sample_name
    results = sample_dir / "results"
    results.mkdir(parents=True)
    excel_path = results / f"{sample_name}_ch1_brain_distribution_stats.xlsx"
    frame = pd.DataFrame(region_rows)
    with pd.ExcelWriter(excel_path) as writer:
        frame.to_excel(writer, sheet_name="Level_8", index=False)
    return str(sample_dir)


def test_parse_group_manifest_json():
    manifest = parse_group_manifest_json(
        json.dumps(
            [
                {"sample_dir": "S:/a", "group": "control"},
                {"sample_dir": "S:/b", "group": "treatment"},
            ]
        )
    )
    assert len(manifest) == 2
    assert manifest[0]["group"] == "control"


def test_compute_differential_regions():
    long_df = pd.DataFrame(
        [
            {"sample_id": "a1", "group": "control", "region_id": 315, "region_name": "ISO", "region_acronym": "ISO", "value": 10.0},
            {"sample_id": "a2", "group": "control", "region_id": 315, "region_name": "ISO", "region_acronym": "ISO", "value": 12.0},
            {"sample_id": "b1", "group": "treat", "region_id": 315, "region_name": "ISO", "region_acronym": "ISO", "value": 40.0},
            {"sample_id": "b2", "group": "treat", "region_id": 315, "region_name": "ISO", "region_acronym": "ISO", "value": 44.0},
            {"sample_id": "a1", "group": "control", "region_id": 600, "region_name": "CB", "region_acronym": "CB", "value": 5.0},
            {"sample_id": "b1", "group": "treat", "region_id": 600, "region_name": "CB", "region_acronym": "CB", "value": 6.0},
        ]
    )
    results = compute_differential_regions(long_df, group_a="control", group_b="treat")
    iso = next(row for row in results if row["region_id"] == 315)
    assert iso["mean_a"] == pytest.approx(11.0)
    assert iso["mean_b"] == pytest.approx(42.0)
    assert iso["log2_fold_change"] > 0
    assert iso["delta"] == pytest.approx(31.0)
    assert "p_value" not in iso
    assert "q_value" not in iso


def test_aggregate_region_metrics_to_level(tmp_path):
    cfg = _write_region_csv(tmp_path)
    metrics = [
        {
            "sample_id": "mouse_a",
            "region_id": 512,
            "region_name": "Sub1",
            "region_acronym": "S1",
            "structure_id_path": [997, 315, 512],
            "level": "Level_3",
            "cfos_count": 10.0,
            "signal_voxels": 10.0,
            "region_volume_voxels": 100.0,
            "sum_intensity": 20.0,
            "has_hemisphere": False,
        },
        {
            "sample_id": "mouse_a",
            "region_id": 520,
            "region_name": "Sub2",
            "region_acronym": "S2",
            "structure_id_path": [997, 315, 520],
            "level": "Level_3",
            "cfos_count": 20.0,
            "signal_voxels": 20.0,
            "region_volume_voxels": 100.0,
            "sum_intensity": 40.0,
            "has_hemisphere": False,
        },
        {
            "sample_id": "mouse_a",
            "region_id": 600,
            "region_name": "Cerebellum",
            "region_acronym": "CB",
            "structure_id_path": [997, 600],
            "level": "Level_1",
            "cfos_count": 5.0,
            "signal_voxels": 5.0,
            "region_volume_voxels": 50.0,
            "sum_intensity": 10.0,
            "has_hemisphere": False,
        },
    ]
    aggregated = aggregate_region_metrics_to_level(metrics, "Level_1", load_region_metadata_table(cfg))
    by_id = {row["region_id"]: row for row in aggregated}
    assert by_id[315]["cfos_count"] == pytest.approx(30.0)
    assert by_id[600]["cfos_count"] == pytest.approx(5.0)


def test_build_pairwise_manifest(tmp_path):
    sample_a = tmp_path / "mouse_a"
    sample_b = tmp_path / "mouse_b"
    sample_a.mkdir()
    sample_b.mkdir()
    manifest = build_pairwise_manifest(sample_a, sample_b, group_a="control", group_b="treat")
    assert len(manifest) == 2
    assert manifest[0]["group"] == "control"
    assert manifest[1]["sample_dir"] == str(sample_b)


def test_build_group_analysis_payload(tmp_path):
    cfg = _write_region_csv(tmp_path)
    iso_name = "Isocortex,ISO"
    cb_name = "Cerebellum,CB"
    control_dir = _write_sample_excel(
        tmp_path,
        "mouse_c1",
        [
            {"Name": iso_name, "Signal Count": 20, "Signal Voxels": 100, "Voxel Density": 0.1, "Total Voxels": 1000, "Sum Intensity": 200},
            {"Name": cb_name, "Signal Count": 5, "Signal Voxels": 50, "Voxel Density": 0.05, "Total Voxels": 1000, "Sum Intensity": 50},
        ],
    )
    treat_dir = _write_sample_excel(
        tmp_path,
        "mouse_t1",
        [
            {"Name": iso_name, "Signal Count": 80, "Signal Voxels": 400, "Voxel Density": 0.4, "Total Voxels": 1000, "Sum Intensity": 800},
            {"Name": cb_name, "Signal Count": 6, "Signal Voxels": 60, "Voxel Density": 0.06, "Total Voxels": 1000, "Sum Intensity": 60},
        ],
    )
    manifest_path = tmp_path / "manifest.csv"
    pd.DataFrame(
        [
            {"sample_dir": control_dir, "group": "control", "signal_ch": "ch1"},
            {"sample_dir": treat_dir, "group": "treatment", "signal_ch": "ch1"},
        ]
    ).to_csv(manifest_path, index=False)

    payload = build_group_analysis_payload(
        load_group_manifest(manifest_path),
        cfg_path=cfg,
        level="Level_1",
        metric="cfos_count",
        group_a="control",
        group_b="treatment",
        top_n=8,
    )
    assert payload["available"] is True
    assert payload["heatmap"]["matrix"]
    assert payload["top_differential_regions"]
    assert payload["pairwise_scatter"]["available"] is True
    assert "pearson_r" in payload["pairwise_scatter"]
    assert payload["sample_correlation"]["available"] is True
    assert "volcano" not in payload
    assert "pca" not in payload
    iso = next(row for row in payload["differential_regions"] if row["region_acronym"] == "ISO")
    assert iso["mean_b"] > iso["mean_a"]


def test_build_sample_correlation_payload():
    long_df = pd.DataFrame(
        [
            {"sample_id": "a1", "group": "control", "region_id": 315, "region_name": "ISO", "region_acronym": "ISO", "value": 10.0},
            {"sample_id": "a2", "group": "control", "region_id": 315, "region_name": "ISO", "region_acronym": "ISO", "value": 12.0},
            {"sample_id": "b1", "group": "treat", "region_id": 315, "region_name": "ISO", "region_acronym": "ISO", "value": 40.0},
            {"sample_id": "b1", "group": "treat", "region_id": 600, "region_name": "CB", "region_acronym": "CB", "value": 6.0},
            {"sample_id": "a1", "group": "control", "region_id": 600, "region_name": "CB", "region_acronym": "CB", "value": 5.0},
            {"sample_id": "a2", "group": "control", "region_id": 600, "region_name": "CB", "region_acronym": "CB", "value": 7.0},
        ]
    )
    from pipeline_modules.visualization.cfos_report_group_stats import build_sample_correlation_payload

    payload = build_sample_correlation_payload(long_df)
    assert payload["available"] is True
    assert len(payload["samples"]) == 3
    assert len(payload["matrix"]) == 3


def test_resolve_region_scope_ids(tmp_path):
    from pipeline_modules.visualization.cfos_report_group_stats import resolve_region_scope_ids

    cfg = _write_region_csv(tmp_path)
    long_df = pd.DataFrame(
        [
            {"sample_id": "a1", "group": "control", "region_id": 315, "region_name": "ISO", "region_acronym": "ISO", "value": 10.0},
            {"sample_id": "a1", "group": "control", "region_id": 512, "region_name": "S1", "region_acronym": "S1", "value": 4.0},
            {"sample_id": "a1", "group": "control", "region_id": 520, "region_name": "S2", "region_acronym": "S2", "value": 6.0},
        ]
    )
    scoped = resolve_region_scope_ids(long_df, focus_region_id=315, cfg_path=cfg)
    assert scoped == [315, 512, 520]


def test_pairwise_scatter_payload():
    long_df = pd.DataFrame(
        [
            {"sample_id": "a1", "group": "control", "region_id": 315, "region_name": "ISO", "region_acronym": "ISO", "value": 10.0},
            {"sample_id": "b1", "group": "treat", "region_id": 315, "region_name": "ISO", "region_acronym": "ISO", "value": 40.0},
            {"sample_id": "a1", "group": "control", "region_id": 600, "region_name": "CB", "region_acronym": "CB", "value": 5.0},
            {"sample_id": "b1", "group": "treat", "region_id": 600, "region_name": "CB", "region_acronym": "CB", "value": 6.0},
        ]
    )
    payload = build_pairwise_scatter_payload(long_df, group_a="control", group_b="treat")
    assert payload["available"] is True
    assert payload["mode"] == "pairwise"
    assert payload["pearson_r"] >= -1.0
    assert payload["n_regions"] == 2


def test_select_top_differential_regions():
    rows = [
        {"region_id": 1, "log2_fold_change": 0.2},
        {"region_id": 2, "log2_fold_change": -1.5},
        {"region_id": 3, "log2_fold_change": 0.9},
    ]
    top = select_top_differential_regions(rows, top_n=2)
    assert top[0]["region_id"] == 2
    assert len(top) == 2
