"""Partial run-configs must resolve through the layering chain in harness code."""

from __future__ import annotations

import json
from pathlib import Path

from pipeline_modules.harness.results import layout_from_config, load_config
from pipeline_modules.harness.verify import required_outputs


def _write(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_load_config_merges_partial_run_config(tmp_path: Path) -> None:
    base = _write(
        tmp_path / "config.json",
        {"project_name": "s", "input": {"channels": {"signal": "1", "registration": "0"}}},
    )
    partial = _write(tmp_path / "config_1.json", {"segmentation": {"cfos_unet": {"normalize_scope": "chunk"}}})
    merged = load_config(partial)
    assert merged["input"]["channels"]["signal"] == "1"  # inherited from sibling base
    assert merged["segmentation"]["cfos_unet"]["normalize_scope"] == "chunk"  # override kept
    assert load_config(base)["input"]["channels"]["signal"] == "1"  # base stays raw


def test_required_outputs_uses_merged_signal_channel(tmp_path: Path) -> None:
    sample = tmp_path / "sample01"
    sample.mkdir()
    _write(
        sample / "config.json",
        {"input": {"channels": {"signal": "1", "registration": "0"}}},
    )
    partial = _write(sample / "config_1.json", {"segmentation": {"cfos_unet": {"normalize_scope": "chunk"}}})
    cfg = load_config(partial)
    assert layout_from_config(sample, cfg).signal_ch == "ch1"
    outputs = required_outputs(sample, cfg)
    assert any(p.name == "sample01_ch1_brain_distribution_stats.xlsx" for p in outputs)
