"""Per-run partial configs: inheritance merge, config-level skip flags."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from pipeline_modules.utils.config_layering import deep_merge, resolve_partial_config

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_deep_merge_dicts_recurse_scalars_override():
    base = {"a": {"x": 1, "y": 2}, "b": [1, 2], "c": "old"}
    override = {"a": {"y": 9}, "b": [3], "d": "new"}
    merged = deep_merge(base, override)
    assert merged == {"a": {"x": 1, "y": 9}, "b": [3], "c": "old", "d": "new"}


def test_partial_config_inherits_sibling_then_template(tmp_path: Path):
    sample = tmp_path / "sample"
    sample.mkdir()
    (sample / "config.json").write_text(
        json.dumps({"input": {"channels": {"signal": "2"}}, "project_name": "base"}),
        encoding="utf-8",
    )
    (sample / "config_1.json").write_text(
        json.dumps({"project_name": "rerun-1", "registration": {"grad_step": 0.2}}),
        encoding="utf-8",
    )

    _path, cfg = resolve_partial_config(sample / "config_1.json")
    # override wins
    assert cfg["project_name"] == "rerun-1"
    assert cfg["registration"]["grad_step"] == 0.2
    # sibling config.json fills the input section
    assert cfg["input"]["channels"]["signal"] == "2"
    # template fills everything else
    assert cfg["registration"]["syn_iterations"] == [70, 50, 30, 10]
    assert cfg["segmentation"]["method"] == "cfos_unet"
    assert "zarr" in cfg["preprocessing"]

    # the sample's own config.json is never merged over anything
    _path, base_cfg = resolve_partial_config(sample / "config.json")
    assert base_cfg == {"input": {"channels": {"signal": "2"}}, "project_name": "base"}


def test_harness_validate_config_accepts_partial(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    root = tmp_path / "arivis-analysis"
    sample = root / "mouse01"
    sample.mkdir(parents=True)
    (sample / "config.json").write_text(
        json.dumps({"input": {"channels": {"signal": "1", "registration": "0"}}}),
        encoding="utf-8",
    )
    (sample / "config_1.json").write_text(json.dumps({"only_tubule": True}), encoding="utf-8")
    monkeypatch.setenv("YIFU_ANALYSIS_ROOT", str(root))
    monkeypatch.setenv("YIFU_ACTIVE_DIR", str(root / "_active"))

    from pipeline_modules.harness.queue import ActiveStore

    path, cfg = ActiveStore().validate_config(sample / "config_1.json")
    assert path.name == "config_1.json"
    assert cfg["input"]["channels"]["signal"] == "1"  # inherited
    assert cfg["only_tubule"] is True


def test_required_outputs_respects_config_level_flags(tmp_path: Path):
    from pipeline_modules.harness.verify import required_outputs

    sample = tmp_path / "mouse01"
    sample.mkdir()
    cfg = {"input": {"channels": {"signal": "1", "registration": "0"}}, "only_tubule": True}
    outputs = required_outputs(sample, cfg, [])
    assert len(outputs) == 1
    assert outputs[0].name.endswith(".csv")


def test_stub_run_with_partial_config_and_skip_registration(tmp_path: Path):
    """A one-file run config (skip + two overrides) drives the full pipeline
    wiring with everything else inherited from config.json -> template."""
    sample = tmp_path / "SAMPLE_PARTIAL"
    sample.mkdir(parents=True)
    (sample / "config.json").write_text(
        json.dumps({"input": {"channels": {"signal": "1", "registration": "0"}}}),
        encoding="utf-8",
    )
    checkpoint = tmp_path / "best_model.pt"
    (sample / "config_1.json").write_text(
        json.dumps(
            {
                "project_name": "stub-partial",
                "skip_registration": True,
                "segmentation": {
                    "cfos_unet": {"checkpoint_path": str(checkpoint)},
                    "postprocess": {"exclude_edge_px": 0, "edge_shrink_factor": 0},
                },
            }
        ),
        encoding="utf-8",
    )

    proc = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "main.py"),
            "--config",
            str(sample / "config_1.json"),
            "--sample_dir",
            str(sample),
            "--stub",
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-2000:]
    summary_line = [line for line in proc.stdout.splitlines() if line.startswith("STUB_SUMMARY ")][-1]
    summary = json.loads(summary_line[len("STUB_SUMMARY ") :])
    commands = summary["commands"]
    # registration family skipped via config flag (incl. the channel
    # downsample, which only feeds registration)
    assert not any("registration" in command for command in commands)
    assert commands == [
        "3.3 Convert signal TIFF to Zarr",
        "4.1 Segmentation (cfos_unet)",
        "4.2 Standard mask postprocess (edge/single-slice filters)",
    ]
