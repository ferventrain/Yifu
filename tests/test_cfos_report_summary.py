from __future__ import annotations

import pytest

from pipeline_modules.visualization.cfos_report_data import build_report_bundle
from pipeline_modules.visualization.cfos_report_summary import (
    SUMMARY_SCHEMA_VERSION,
    build_summary_payload,
    read_summary_json,
    summary_json_path,
    write_summary_json,
)
from tests.test_cfos_report_data import write_density_excel, write_groups_json, write_region_csv


def test_build_summary_payload_fields(tmp_path):
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
    summary = bundle["summary"]

    assert summary["schema_version"] == SUMMARY_SCHEMA_VERSION
    assert summary["sample"]["sample_id"] == "mouse01"
    assert summary["sample"]["sample_dir"] == str(sample_dir.resolve())
    assert summary["sample"]["signal_ch"] == "ch1"
    assert summary["sample"]["group"] == "control"
    assert summary["headline_stats"]["total_cfos_count"] > 0
    assert summary["headline_stats"].get("signal_volume_um3") is not None
    assert summary["headline_stats"].get("brain_volume_um3") is not None
    assert summary["headline_stats"]["whole_brain_voxel_density"] >= 0
    assert summary["top_regions_by_count"]
    assert isinstance(summary["systems"], list)
    level2 = build_summary_payload(bundle, sample_dir=sample_dir, level="Level_2")
    assert level2["systems"]
    assert level2["laterality"] is not None
    assert summary["data_availability"]["density_excel"] is True


def test_summary_json_roundtrip(tmp_path):
    cfg = write_region_csv(tmp_path)
    groups = write_groups_json(tmp_path)
    sample_dir = tmp_path / "mouse01"
    sample_dir.mkdir()
    write_density_excel(sample_dir)

    bundle = build_report_bundle(sample_dir, cfg_path=cfg, groups_json=groups, signal_ch="ch1")
    path = summary_json_path(sample_dir, "ch1")
    assert path.name == "mouse01_ch1_summary.json"
    assert path.is_file()

    loaded = read_summary_json(path)
    assert loaded["sample"]["sample_id"] == "mouse01"
    assert loaded["schema_version"] == SUMMARY_SCHEMA_VERSION


def test_build_summary_payload_level_override(tmp_path):
    cfg = write_region_csv(tmp_path)
    groups = write_groups_json(tmp_path)
    sample_dir = tmp_path / "mouse01"
    sample_dir.mkdir()
    write_density_excel(sample_dir)

    bundle = build_report_bundle(sample_dir, cfg_path=cfg, groups_json=groups, signal_ch="ch1")
    level2 = build_summary_payload(bundle, sample_dir=sample_dir, level="Level_2")
    assert level2["atlas"]["level"] == "Level_2"
    assert level2["headline_stats"]["activated_region_count"] == 2
    assert level2["headline_stats"]["total_region_count"] == 2


def test_write_summary_json_rejects_unknown_schema(tmp_path):
    path = tmp_path / "bad_summary.json"
    write_summary_json(path, {"schema_version": "999", "sample": {"sample_id": "x"}})
    with pytest.raises(ValueError, match="Unsupported summary schema"):
        read_summary_json(path)
