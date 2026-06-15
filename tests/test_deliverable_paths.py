"""Tests for pipeline_modules.utils.deliverable_paths."""

from __future__ import annotations

from pathlib import Path

from pipeline_modules.utils.deliverable_paths import (
    brain_distribution_stats_xlsx,
    heatmap_2d_dir,
    heatmap_3d_png,
    heatmap_3d_volume_tiff,
    normalize_channel,
)


def test_normalize_channel_variants():
    assert normalize_channel("ch1") == "ch1"
    assert normalize_channel("ch2+ch3") == "ch2_ch3"
    assert normalize_channel("all") == "all"


def test_brain_distribution_stats_path(tmp_path: Path):
    sample = tmp_path / "sham"
    sample.mkdir()
    assert brain_distribution_stats_xlsx(sample, "ch1") == (
        sample / "results" / "sham_ch1_brain_distribution_stats.xlsx"
    )


def test_heatmap_paths(tmp_path: Path):
    sample = tmp_path / "nao_1"
    sample.mkdir()
    assert heatmap_2d_dir(sample, "ch1") == sample / "visualization" / "nao_1_ch1_heatmap_2d"
    assert heatmap_3d_png(sample, "ch1") == sample / "visualization" / "nao_1_ch1_heatmap_3d.png"
    assert heatmap_3d_volume_tiff(sample, "ch1") == (
        sample / "visualization" / "nao_1_ch1_heatmap_3d_volume.tiff"
    )
