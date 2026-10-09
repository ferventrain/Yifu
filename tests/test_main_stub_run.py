"""Stub-run wiring test: the full 6-step pipeline must traverse every step,
build every command, and touch every declared output — without running any
algorithm — on a bare sample directory.

This is the cheap regression net for path/key conventions between steps:
rename an output, drop a config key, or break a branch condition and this
test fails in seconds instead of three hours into a real run.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

EXPECTED_COMMANDS = [
    "1.1 Downsample registration channel",
    "2.1 ANTs registration (atlas -> image)",
    "2.3 Convert atlas label to hemisphere Zarr (single-plane fallback)",
    "3.3 Convert signal TIFF to Zarr",
    "4.1 Segmentation (cfos_unet)",
    "4.2 Standard mask postprocess (edge/single-slice filters)",
    "5.2 Region density analysis",
]


def _build_stub_config(tmp_path: Path) -> Path:
    template = json.loads(
        (REPO_ROOT / "config" / "config_cfos_template.json").read_text(encoding="utf-8-sig")
    )
    template["project_name"] = "stub-wiring-test"
    # Point reference/model paths at the temp tree so the test never needs
    # YIFU_DATA_DIR or the real atlas on disk.
    template["registration"]["atlas_path"] = str(tmp_path / "reference" / "atlas.tiff")
    template["registration"]["annotation_path"] = str(tmp_path / "reference" / "atlas_label.tiff")
    template["segmentation"]["cfos_unet"]["checkpoint_path"] = str(tmp_path / "models" / "best_model.pt")

    sample_dir = tmp_path / "SAMPLE_STUB"
    sample_dir.mkdir(parents=True)
    config_path = sample_dir / "config.json"
    config_path.write_text(json.dumps(template, indent=1), encoding="utf-8")
    return config_path


def test_main_stub_run_traverses_all_steps(tmp_path: Path, capsys=None):
    config_path = _build_stub_config(tmp_path)
    sample_dir = config_path.parent

    proc = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "main.py"),
            "--config",
            str(config_path),
            "--sample_dir",
            str(sample_dir),
            "--stub",
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert proc.returncode == 0, f"stub run failed:\n{proc.stdout[-3000:]}\n{proc.stderr[-3000:]}"
    stdout = proc.stdout

    # All six step banners were reached.
    for step in range(1, 7):
        assert f"Step {step}/6" in stdout, f"step {step} banner missing"

    # Every expected command was built and recorded.
    summary_lines = [line for line in stdout.splitlines() if line.startswith("STUB_SUMMARY ")]
    assert summary_lines, "STUB_SUMMARY line missing"
    summary = json.loads(summary_lines[-1][len("STUB_SUMMARY ") :])
    assert summary["stub"] is True
    for desc in EXPECTED_COMMANDS:
        assert desc in summary["commands"], f"command not built: {desc}"
    assert summary["command_count"] == len(summary["commands"])

    # Stub outputs were materialized so downstream exists() gates passed.
    for output in (
        sample_dir / "ch0_downsample" / "volume.nii.gz",
        sample_dir / "upsampled_atlas_label.zarr",
        sample_dir / "atlas_label_hemisphere.zarr",
        sample_dir / "ch1.zarr",
        sample_dir / "ch1_mask.zarr",
    ):
        assert output.exists(), f"stub output missing: {output}"

    # No real command may have executed: nothing beyond .stub placeholders
    # may live in the touched zarr directories.
    touched = list((sample_dir / "ch1.zarr").iterdir())
    assert [p.name for p in touched] == [".stub"]

    # No completion markers are written in stub mode.
    assert not list(sample_dir.glob("*.done.json"))


def test_skip_registration_bypasses_both_registration_steps(tmp_path: Path):
    """--skip_registration must skip BOTH the registration-channel downsample
    (step 1) and ANTs registration (step 2).

    Stats-only rerun folders keep only registration results (upsampled_atlas_label.zarr,
    transforms) — no ch0/ TIFFs, no ch0_downsample/volume.nii.gz — so step 1 has
    nothing to skip on and nothing to rebuild from. Regression: production runs
    failed twice at step 1 with INPUT_NOT_FOUND ch0 before this branch existed.
    """
    config_path = _build_stub_config(tmp_path)
    sample_dir = config_path.parent

    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    cfg["segmentation"]["method"] = "threshold"
    cfg["segmentation"]["threshold"] = {"value": "250", "sigma": 0, "min_object_size": 10}
    cfg["analysis"]["use_hemisphere_label"] = False
    cfg["registration"]["save_upsampled_label_hemisphere_zarr"] = False
    config_path.write_text(json.dumps(cfg, indent=1), encoding="utf-8")

    # The one registration product such folders keep.
    (sample_dir / "upsampled_atlas_label.zarr").mkdir()

    proc = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "main.py"),
            "--config",
            str(config_path),
            "--sample_dir",
            str(sample_dir),
            "--stub",
            "--skip_registration",
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert proc.returncode == 0, f"stub run failed:\n{proc.stdout[-3000:]}\n{proc.stderr[-3000:]}"
    stdout = proc.stdout
    assert "Registration channel downsample skipped (--skip_registration)." in stdout
    assert "Registration skipped (--skip_registration)." in stdout

    summary_lines = [line for line in stdout.splitlines() if line.startswith("STUB_SUMMARY ")]
    assert summary_lines, "STUB_SUMMARY line missing"
    commands = json.loads(summary_lines[-1][len("STUB_SUMMARY ") :])["commands"]
    assert not any(str(c).startswith("1.1 ") for c in commands), "step 1 downsample must not run"
    assert not any(str(c).startswith("2.1 ") for c in commands), "ANTs registration must not run"
    assert "5.2 Region density analysis" in commands, "analysis must still run off existing label zarr"
    assert (sample_dir / "ch1_mask.zarr").exists(), "segmentation must still run"
