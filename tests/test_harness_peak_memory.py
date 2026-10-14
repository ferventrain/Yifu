"""Peak-memory recording: process-tree sampler, per-step progress fields,
timing history memory lane, and the run_command module-id parser."""

from __future__ import annotations

import json
from pathlib import Path

from pipeline_modules.harness.proc import tree_memory_mb
from pipeline_modules.harness.progress import ProgressWriter, touch_heartbeat
from pipeline_modules.harness.timing import (
    load_memory_history,
    load_timing_history,
    record_step_timings,
)


def test_tree_memory_mb_own_process():
    import os

    value = tree_memory_mb(os.getpid())
    assert value is not None and value > 10.0


def test_tree_memory_mb_invalid_pids():
    assert tree_memory_mb(None) is None
    assert tree_memory_mb(0) is None
    assert tree_memory_mb(-5) is None


def test_module_id_from_command():
    from main import module_id_from_command

    cmd = (
        '"C:\\python.exe" -m pipeline_modules.segmentation.cfos_mask_postprocess '
        "--signal_zarr s/ch1.zarr"
    )
    assert module_id_from_command(cmd) == "segmentation.cfos_mask_postprocess"
    assert module_id_from_command("python main.py --config x") is None
    # single-component package path (e.g. the harness itself) is not a step module
    assert module_id_from_command("python -m pipeline_modules.harness run") is None


def test_runcontext_records_step_peak_and_module(tmp_path: Path):
    progress = tmp_path / "progress.json"
    ctx = ProgressWriter(progress, step_total=2)
    ctx.start_run(config_path=str(tmp_path / "config.json"))
    ctx.begin_step(1, "4.2 Standard mask postprocess")
    ctx.record_step_peak(2048.4)
    ctx.record_step_peak(1024.0)  # lower sample must not shrink the max
    ctx.annotate_current_module("segmentation.cfos_mask_postprocess")
    ctx.annotate_current_module("other.module")  # first annotation wins
    ctx.begin_step(2, "5 Region stats")
    state = json.loads(progress.read_text(encoding="utf-8"))
    step42 = state["steps"][0]
    assert step42["peak_mem_mb"] == 2048.4
    assert step42["module"] == "segmentation.cfos_mask_postprocess"
    assert state["steps"][1]["peak_mem_mb"] is None


def test_touch_heartbeat_merges_peak(tmp_path: Path):
    progress = tmp_path / "progress.json"
    ctx = ProgressWriter(progress, step_total=1)
    ctx.start_run(config_path=str(tmp_path / "config.json"))
    touch_heartbeat(progress, peak_mem_mb=5000.0)
    touch_heartbeat(progress, peak_mem_mb=4000.0)  # must keep the max
    touch_heartbeat(progress)  # no peak -> unchanged
    state = json.loads(progress.read_text(encoding="utf-8"))
    assert state["peak_mem_mb"] == 5000.0
    assert state["heartbeat_at"]


def test_timing_history_records_memory(tmp_path: Path):
    active = tmp_path / "_active"
    active.mkdir()
    progress = {
        "steps": [
            {"name": "4.2 Standard mask postprocess", "seconds": 3600.0, "peak_mem_mb": 80000.0},
            {"name": "4.2 Standard mask postprocess", "seconds": 1800.0, "peak_mem_mb": 60000.0},
            {"name": "SKIP: nothing", "seconds": 5.0, "skipped": True, "peak_mem_mb": 999.0},
            {"name": "2.1 ANTs registration", "seconds": 7200.0},  # no memory recorded
        ]
    }
    record_step_timings(progress, active=active)
    durations = load_timing_history(active)
    assert durations["4.2 Standard mask postprocess"] == [3600.0, 1800.0]
    mems = load_memory_history(active)
    assert mems["4.2 Standard mask postprocess"] == [80000.0, 60000.0]
    assert "2.1 ANTs registration" not in mems
    assert "SKIP: nothing" not in mems
    # legacy durations-only file (no steps_mem) still loads
    legacy = active / "timing_history.json"
    legacy.write_text(json.dumps({"schema_version": "1", "steps": {"x": [10.0]}}), encoding="utf-8")
    assert load_timing_history(active)["x"] == [10.0]
    assert load_memory_history(active) == {}
