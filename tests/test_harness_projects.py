"""Tests for the project board: /api/projects grouping and notes POST."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pipeline_modules.harness.queue import ActiveStore


@pytest.fixture
def analysis_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "arivis-analysis"
    root.mkdir()
    active = root / "_active"
    monkeypatch.setenv("YIFU_ANALYSIS_ROOT", str(root))
    monkeypatch.setenv("YIFU_ACTIVE_DIR", str(active))
    return root


def _write_sample(root: Path, name: str, config_names=("config.json",), project_name=None) -> Path:
    sample = root / name
    sample.mkdir(parents=True)
    for config_name in config_names:
        config = {"input": {"channels": {"signal": "1", "registration": "0"}}}
        if project_name:
            config["project_name"] = project_name
        (sample / config_name).write_text(json.dumps(config), encoding="utf-8")
    return sample


def test_projects_groups_by_batch_and_sample(analysis_home: Path, monkeypatch: pytest.MonkeyPatch):
    pytest.importorskip("fastapi")
    batch = analysis_home / "YF20991001"
    ot = _write_sample(batch, "ot", config_names=("config.json", "config_1.json"), project_name="p_ot")
    sham = _write_sample(batch, "sham", project_name="p_sham")
    lone = _write_sample(analysis_home, "solo_sample")

    store = ActiveStore()
    store.add_job(ot)
    store.add_job(ot, ot / "config_1.json")
    store.add_job(sham)
    store.add_job(lone)

    import importlib

    monitor = importlib.import_module("apps.pipeline_harness.main")
    monkeypatch.setattr(monitor, "store", store)
    payload = monitor.api_projects()

    batches = {b["batch"]: b for b in payload["batches"]}
    assert set(batches) == {"YF20991001", "arivis-analysis"}
    batch_projects = {p["sample_name"]: p for p in batches["YF20991001"]["projects"]}
    assert set(batch_projects) == {"ot", "sham"}
    # two runs on ot grouped under one project, newest first
    assert len(batch_projects["ot"]["runs"]) == 2
    # config listing for one-click start: config.json first, then config_1
    assert batch_projects["ot"]["configs"] == ["config.json", "config_1.json"]
    # aggregate status over queued runs
    assert batch_projects["ot"]["status"] == "queued"


def test_projects_notes_post_and_roundtrip(analysis_home: Path, monkeypatch: pytest.MonkeyPatch):
    pytest.importorskip("fastapi")
    sample = _write_sample(analysis_home, "ot")
    store = ActiveStore()
    store.add_job(sample)

    import importlib

    monitor = importlib.import_module("apps.pipeline_harness.main")
    monkeypatch.setattr(monitor, "store", store)

    body = monitor.ProjectNotesBody(
        sample_dir=str(sample), requirements="OT 组 cfos 定量", retro="", append_note="首次提交"
    )
    result = monitor.api_project_notes(body)
    assert result["notes"]["requirements"] == "OT 组 cfos 定量"
    assert result["notes"]["log"][0]["text"] == "首次提交"

    # notes surface in the projects payload
    payload = monitor.api_projects()
    project = payload["batches"][0]["projects"][0]
    assert project["notes"]["requirements"] == "OT 组 cfos 定量"
    assert project["notes"]["log"][0]["text"] == "首次提交"

    # paths outside the analysis root are rejected
    with pytest.raises(Exception):
        monitor.api_project_notes(
            monitor.ProjectNotesBody(sample_dir=str(Path(__file__).absolute()), requirements="x")
        )


def test_projects_get_never_starts_worker(analysis_home: Path, monkeypatch: pytest.MonkeyPatch):
    pytest.importorskip("fastapi")
    sample = _write_sample(analysis_home, "ot")
    store = ActiveStore()
    store.add_job(sample)

    import importlib

    monitor = importlib.import_module("apps.pipeline_harness.main")
    calls = []
    monkeypatch.setattr(monitor, "store", store)
    monkeypatch.setattr(
        monitor, "ensure_worker_running", lambda *a, **k: calls.append(1) or {"started": True}
    )
    monitor.api_projects()
    assert calls == []


def test_projects_includes_manifest_summary(analysis_home: Path, monkeypatch: pytest.MonkeyPatch):
    pytest.importorskip("fastapi")
    sample = _write_sample(analysis_home, "ot")
    manifest = {
        "module": "registration.ANTs_registration",
        "entrypoint": "run_full_pipeline",
        "ended_at": "2026-10-09T12:00:00+00:00",
        "duration_seconds": 620.0,
        "code": {"git_commit": "abcdef1234567890", "git_dirty": False},
        "outputs": [{"path": "x.nii", "type": "file"}],
    }
    (sample / "qc" / "_run_manifest.json").parent.mkdir(exist_ok=True)
    (sample / "qc" / "_run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    store = ActiveStore()
    store.add_job(sample)
    import importlib

    monitor = importlib.import_module("apps.pipeline_harness.main")
    monkeypatch.setattr(monitor, "store", store)
    payload = monitor.api_projects()
    project = payload["batches"][0]["projects"][0]
    assert project["last_manifest"]["module"] == "registration.ANTs_registration"
    assert project["last_manifest"]["git_commit"].startswith("abcdef")
