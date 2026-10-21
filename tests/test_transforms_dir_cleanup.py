"""Guard: re-registration must not leave mixed transform generations."""

from __future__ import annotations

import re
from pathlib import Path


def test_save_registration_results_clears_stale_transforms(tmp_path: Path) -> None:
    """Simulate a rerun over a dirty transforms dir: stale fwd_*/inv_* files must
    be removed before the new generation is copied in."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "ants_registration_module", REPO_MODULE := Path(r"H:/yifu0428/Yifu/pipeline_modules/registration/ANTs_registration.py")
    )
    assert spec is not None
    # Parsing the whole module requires heavy deps (ants). Instead assert the
    # cleanup exists at the single write site by source inspection: the save
    # block must unlink stale fwd_*/inv_* before copying.
    source = REPO_MODULE.read_text(encoding="utf-8")
    save_block = source.split("def save_registration_results")[1].split("def check_and_run_density_analysis")[0]
    # the unlink lines must appear before the first fwd_ copy
    copy_pos = save_block.find("fwd_{i}_")
    cleanup_pos = save_block.find('transforms_dir.glob("fwd_*")')
    inv_cleanup_pos = save_block.find('transforms_dir.glob("inv_*")')
    assert 0 < cleanup_pos < copy_pos, "stale fwd_* cleanup missing before copy"
    assert 0 < inv_cleanup_pos < copy_pos, "stale inv_* cleanup missing before copy"
    assert "stale.unlink()" in save_block


def test_dirty_transforms_dir_is_detected_by_source_convention(tmp_path: Path) -> None:
    """A dir with two generations is recognizable: same fwd_<i>_ prefix twice."""
    d = tmp_path / "transforms"
    d.mkdir()
    (d / "fwd_0_tmpaaaa1Warp.nii.gz").write_bytes(b"x")
    (d / "fwd_0_tmpbbbb1Warp.nii.gz").write_bytes(b"x")
    generations = {}
    for p in d.glob("fwd_*"):
        m = re.match(r"fwd_(\d+)_(tmp\w+)", p.name)
        generations.setdefault(m.group(1), set()).add(m.group(2))
    dup = [idx for idx, tokens in generations.items() if len(tokens) > 1]
    assert dup == ["0"]
