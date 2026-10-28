"""Guard: the repo's default region CSV and atlas annotation stay on the v2 lineage.

Rev1 -> Rev2 was a full id-space migration (2026-10): Region_Csv_Rev2_updated.CSV
pairs with atlas_label_v2_uint16.tiff via atlas_v2_id_map.json. Nothing new may
default back to the Rev1 CSV or the old atlas_label.tiff.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import tifffile

REPO = Path(__file__).resolve().parents[1]


def test_default_parcellation_csv_points_at_rev2() -> None:
    from pipeline_modules.visualization.top_level7_density_ratio import DEFAULT_PARCELLATION_CSV

    assert "Rev2" in DEFAULT_PARCELLATION_CSV.name
    frame = pd.read_csv(DEFAULT_PARCELLATION_CSV)
    assert "official_id" in frame.columns  # Rev2-only column
    assert len(frame) == 1327


def test_resolve_atlas_label_path_prefers_v2(tmp_path, monkeypatch) -> None:
    from pipeline_modules.utils import data_paths

    fake_data = tmp_path / "Yifu_data" / "reference"
    fake_data.mkdir(parents=True)
    (fake_data / "atlas_label_v2_uint16.tiff").write_bytes(b"x")
    (fake_data / "atlas_label.tiff").write_bytes(b"old")
    monkeypatch.setenv("YIFU_DATA_DIR", str(tmp_path / "Yifu_data"))
    monkeypatch.delenv("YIFU_ANALYSIS_ROOT", raising=False)
    resolved = data_paths.resolve_atlas_label_path()
    assert resolved.name == "atlas_label_v2_uint16.tiff"


def test_no_rev1_defaults_in_tracked_sources() -> None:
    """Grep-guard: tracked python/config sources must not default to Rev1 artifacts."""
    import subprocess

    result = subprocess.run(
        ["git", "grep", "-n", "-E", r"Region_Csv_Rev1_updated|reference/atlas_label\.tiff"],
        cwd=REPO,
        capture_output=True,
        text=True,
    )

    def is_allowed(match_line: str) -> bool:
        # 守卫文件自身提到模式；napari_atlas_overlay 的 LEGACY_REGION_CSV_CANDIDATES
        # 与示例/文档里的 _frangi_trials2 引用是运行时兜底（老 id label 配本地
        # Rev1 副本），默认仍是 Rev2
        return (
            "test_atlas_v2_defaults" in match_line
            or "napari_atlas_overlay.py" in match_line
            or "_frangi_trials2" in match_line
        )

    offenders = [line for line in result.stdout.splitlines() if line.strip() and not is_allowed(line)]
    assert not offenders, f"Rev1/old-label defaults crept back into: {offenders}"


def test_v2_reference_and_idmap_exist() -> None:
    v2_tiff = Path(r"H:/Yifu_data/reference/atlas_label_v2_uint16.tiff")
    id_map = Path(r"H:/Yifu_data/reference/atlas_v2_id_map.json")
    if not v2_tiff.exists():
        pytest.skip("Yifu_data reference not mounted on this machine")
    assert tifffile.memmap(str(v2_tiff)).shape == (320, 528, 456)
    assert id_map.exists()
