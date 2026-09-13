from __future__ import annotations

import json
from pathlib import Path

import pytest

from pipeline_modules.harness.paths import active_dir, analysis_root
from pipeline_modules.harness.progress import (
    ProgressWriter,
    estimate_eta,
    planned_step_names,
)
from pipeline_modules.harness.queue import ActiveStore
from pipeline_modules.harness.results import collect_existing_results, normalize_channel_label
from pipeline_modules.harness.timing import record_step_timings
from pipeline_modules.utils.errors import ErrorCode, PipelineError


MINIMAL_CONFIG = {
    "project_name": "Test",
    "input": {"channels": {"signal": "0", "registration": "1"}},
    "segmentation": {"method": "threshold"},
}


@pytest.fixture
def analysis_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "arivis-analysis"
    root.mkdir()
    active = root / "_active"
    monkeypatch.setenv("YIFU_ANALYSIS_ROOT", str(root))
    monkeypatch.setenv("YIFU_ACTIVE_DIR", str(active))
    return root


def _write_sample(root: Path, name: str = "mouse01") -> Path:
    sample = root / name
    sample.mkdir()
    (sample / "config.json").write_text(json.dumps(MINIMAL_CONFIG), encoding="utf-8")
    return sample


def test_env_paths_follow_overrides(analysis_home: Path):
    assert analysis_root() == analysis_home
    assert active_dir() == analysis_home / "_active"


def test_add_job_requires_sample_under_root(analysis_home: Path, tmp_path: Path):
    store = ActiveStore()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "config.json").write_text(json.dumps(MINIMAL_CONFIG), encoding="utf-8")
    with pytest.raises(PipelineError) as exc:
        store.add_job(outside)
    assert exc.value.code == ErrorCode.ARGUMENT_INVALID


def test_add_job_and_list_views(analysis_home: Path):
    sample = _write_sample(analysis_home)
    store = ActiveStore()
    record = store.add_job(sample)
    assert record["status"] == "queued"
    assert record["extra_args"] == []
    views = store.list_job_views()
    assert len(views) == 1
    assert views[0]["id"] == record["id"]
    assert views[0]["progress"]["status"] == "pending"
    assert views[0]["eta"]["collecting"] is True


def test_progress_writer_and_eta(analysis_home: Path):
    sample = _write_sample(analysis_home)
    store = ActiveStore()
    job = store.add_job(sample)
    progress_file = Path(store.job_view(job)["progress_path"])
    writer = ProgressWriter(progress_file, step_total=6)
    writer.start_run(sample_dir=str(sample), config_path=str(sample / "config.json"))
    writer.begin_step(1, "Registration channel downsample")
    writer.state["steps"][-1]["seconds"] = 12.0
    writer.state["steps"][-1]["ended_at"] = writer.state["steps"][-1]["started_at"]
    writer.begin_step(2, "Atlas registration and label outputs")
    payload = writer.state
    history = {"Registration channel downsample": [10.0, 20.0], "Atlas registration and label outputs": [100.0]}
    eta = estimate_eta(payload, history, planned_names=planned_step_names(MINIMAL_CONFIG))
    assert eta["source"] in {"history", "partial"}
    assert eta["current_step_remaining_s"] is not None


def test_record_step_timings_skips_tiny_and_skipped(analysis_home: Path):
    sample = _write_sample(analysis_home)
    store = ActiveStore()
    store.add_job(sample)
    record_step_timings(
        {
            "steps": [
                {"name": "Segmentation", "seconds": 0.2, "skipped": False},
                {"name": "Segmentation", "seconds": 40.0, "skipped": False},
                {"name": "Region density analysis", "seconds": 12.0, "skipped": True},
            ]
        },
        active=store.active,
    )
    from pipeline_modules.harness.timing import load_timing_history

    history = load_timing_history(store.active)
    assert history["Segmentation"] == [40.0]
    assert "Region density analysis" not in history


def test_open_path_in_file_manager_opens_parent_for_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from pipeline_modules.harness.paths import open_path_in_file_manager

    sample = tmp_path / "file.xlsx"
    sample.write_bytes(b"x")
    called = []

    def fake_popen(cmd, *args, **kwargs):
        called.append(cmd)
        return None

    monkeypatch.setattr("pipeline_modules.harness.paths.subprocess.Popen", fake_popen)
    monkeypatch.setattr("pipeline_modules.harness.paths.os.name", "nt")
    folder = open_path_in_file_manager(sample)
    assert folder == tmp_path.resolve()
    assert called
    assert called[0][0] == "explorer"
    assert called[0][1].startswith("/select,")


def test_collect_existing_results(tmp_path: Path):
    sample = tmp_path / "mouse"
    sample.mkdir()
    excel_dir = sample / "results"
    excel_dir.mkdir()
    excel = excel_dir / "mouse_ch0_brain_distribution_stats.xlsx"
    excel.write_bytes(b"xlsx")
    (sample / "ch0.zarr").mkdir()
    rows = collect_existing_results(sample, MINIMAL_CONFIG)
    names = {row["name"] for row in rows}
    assert "signal_zarr" in names
    assert "brain_distribution_stats_xlsx" in names
    assert normalize_channel_label("1") == "ch1"


def test_spotiflow_planned_steps():
    names = planned_step_names({"segmentation": {"method": "spotiflow"}})
    assert names[4] == "Spotiflow signal count summary"


def test_sample_nested_under_active_is_allowed(analysis_home: Path):
    sample = analysis_home / "_active" / "YF2025120501" / "dbm1"
    sample.mkdir(parents=True)
    (sample / "config.json").write_text(json.dumps(MINIMAL_CONFIG), encoding="utf-8")
    store = ActiveStore()
    record = store.add_job(sample)
    assert record["status"] == "queued"
    assert Path(record["sample_dir"]) == sample.resolve()


def test_jobs_bookkeeping_dir_is_rejected(analysis_home: Path):
    store = ActiveStore()
    store.ensure_layout()
    with pytest.raises(PipelineError) as exc:
        store.add_job(store.active / "jobs")
    assert exc.value.code == ErrorCode.ARGUMENT_INVALID


def test_attach_external_reads_status_and_live_pid(analysis_home: Path):
    import os

    sample = analysis_home / "_active" / "YF2025120501" / "dbm1"
    sample.mkdir(parents=True)
    (sample / "config.json").write_text(json.dumps(MINIMAL_CONFIG), encoding="utf-8")
    status = analysis_home / "_active" / "YF2025120501" / "_dbm1_pipeline_status.txt"
    status.write_text("RUNNING Step 2: ANTs registration atlas2image SyN\n2026-08-28T13:34:58\ncmd\n", encoding="utf-8")
    log = analysis_home / "_active" / "YF2025120501" / "_dbm1_pipeline.log"
    log.write_text("log line\n", encoding="utf-8")
    store = ActiveStore()
    record = store.attach_external(
        sample,
        title="dbm1 全脑+海马血管重建",
        pid=os.getpid(),
        log_path=log,
        status_path=status,
        config_path=sample / "config.json",
    )
    assert record["status"] == "running"
    assert record["step_index"] == 2
    views = store.list_job_views()
    assert views[0]["progress"]["step_name"].startswith("Step 2")
    assert views[0]["log_path"].endswith("_dbm1_pipeline.log")
