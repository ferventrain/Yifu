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


def test_project_status_follows_newest_run_not_history(analysis_home: Path, monkeypatch: pytest.MonkeyPatch):
    """A sample whose latest rerun succeeded must show succeeded, not stay
    red forever because older runs failed; history stays countable."""
    pytest.importorskip("fastapi")
    sample = _write_sample(analysis_home, "ot")

    store = ActiveStore()
    old = store.add_job(sample)
    old["status"] = "failed"
    old["error"] = {"code": "INTERNAL_ERROR", "message": "older failure"}
    old["started_at"] = "2026-10-01T00:00:00+00:00"
    old["ended_at"] = "2026-10-01T00:00:00+00:00"
    store.save_job(old)
    new = store.add_job(sample, sample / "config.json")
    new["status"] = "succeeded"
    new["started_at"] = "2026-10-02T00:00:00+00:00"
    new["ended_at"] = "2026-10-02T00:00:00+00:00"
    store.save_job(new)

    import importlib

    monitor = importlib.import_module("apps.pipeline_harness.main")
    monkeypatch.setattr(monitor, "store", store)
    payload = monitor.api_projects()
    project = payload["batches"][0]["projects"][0]
    assert project["status"] == "succeeded"
    assert project["failed_count"] == 1
    assert len(project["error_history"]) == 1


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

    # notes surface in the projects payload — at the BATCH (project) level
    payload = monitor.api_projects()
    batch = payload["batches"][0]
    assert batch["notes"]["requirements"] == "OT 组 cfos 定量"
    assert batch["notes"]["log"][0]["text"] == "首次提交"
    assert "notes" not in batch["projects"][0] or batch["projects"][0].get("notes") is None

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


# ---------------------------------------------------------------- archive (归档)


def _make_archivable_sample(root: Path, name: str = "sample") -> Path:
    """A finished sample shaped like real production folders: v2 GROUP zarr
    stores (root .zgroup + 0/.zarray, the layout archive must detect), a
    completion marker, raw ims, transforms, and kept deliverables."""
    sample = root / name
    (sample / "ch1.zarr" / "0").mkdir(parents=True)
    (sample / "ch1.zarr" / ".zgroup").write_text("{}", encoding="utf-8")
    (sample / "ch1.zarr" / "0" / ".zarray").write_text("{}", encoding="utf-8")
    (sample / "atlas_label_hemisphere.zarr").mkdir()
    (sample / "atlas_label_hemisphere.zarr" / ".zgroup").write_text("{}", encoding="utf-8")
    (sample / "transforms").mkdir()
    (sample / "raw.ims").write_text("x", encoding="utf-8")
    (sample / "ch1.zarr.done.json").write_text("{}", encoding="utf-8")
    (sample / "config.json").write_text(
        json.dumps({"input": {"channels": {"signal": "1", "registration": "0"}}}), encoding="utf-8"
    )
    (sample / "results").mkdir()
    (sample / "results" / "stats.xlsx").write_text("x", encoding="utf-8")
    (sample / "qc").mkdir()
    (sample / "qc" / "check.png").write_text("x", encoding="utf-8")
    return sample


def test_archive_deletes_v2_group_zarr_stores(tmp_path: Path):
    from apps.pipeline_harness.projects import archive_sample

    sample = _make_archivable_sample(tmp_path)
    summary = archive_sample(sample, tmp_path / "_archived")

    # the production layout (group store) IS detected and deleted
    assert "ch1.zarr/" in summary["deleted"]
    assert "atlas_label_hemisphere.zarr/" in summary["deleted"]
    assert "transforms/" in summary["deleted"]
    assert "raw.ims" in summary["deleted"]
    assert "ch1.zarr.done.json" in summary["deleted"]  # .done.json suffix matches
    # deliverables kept and the folder moved
    dest = tmp_path / "_archived" / "sample"
    assert (dest / "results" / "stats.xlsx").exists()
    assert (dest / "qc" / "check.png").exists()
    assert (dest / "config.json").exists()
    assert not sample.exists()


def test_archive_dry_run_touches_nothing(tmp_path: Path):
    from apps.pipeline_harness.projects import archive_sample

    sample = _make_archivable_sample(tmp_path)
    summary = archive_sample(sample, tmp_path / "_archived", dry_run=True)

    assert summary["dry_run"] is True
    assert summary["deleted_count"] == len(summary["would_delete"]) == 5
    # nothing was deleted and the folder did not move
    assert (sample / "ch1.zarr" / ".zgroup").exists()
    assert (sample / "transforms").exists()
    assert not (tmp_path / "_archived").exists()


def test_archive_endpoint_refuses_unfinished_jobs(analysis_home: Path, monkeypatch: pytest.MonkeyPatch):
    pytest.importorskip("fastapi")
    sample = _make_archivable_sample(analysis_home)
    store = ActiveStore()
    job = store.add_job(sample)
    job["status"] = "running"
    store.save_job(job)

    import importlib

    monitor = importlib.import_module("apps.pipeline_harness.main")
    monkeypatch.setattr(monitor, "store", store)

    with pytest.raises(Exception):
        monitor.api_project_archive(monitor.ProjectArchiveBody(sample_dir=str(sample)))


def test_archive_endpoint_dry_run_and_full_roundtrip(analysis_home: Path, monkeypatch: pytest.MonkeyPatch):
    pytest.importorskip("fastapi")
    sample = _make_archivable_sample(analysis_home)
    store = ActiveStore()
    job = store.add_job(sample)
    job["status"] = "done"
    store.save_job(job)

    import importlib

    monitor = importlib.import_module("apps.pipeline_harness.main")
    monkeypatch.setattr(monitor, "store", store)

    preview = monitor.api_project_archive(
        monitor.ProjectArchiveBody(sample_dir=str(sample), dry_run=True)
    )
    assert preview["dry_run"] is True
    assert "ch1.zarr/" in preview["would_delete"]
    assert (sample / "ch1.zarr").exists()  # preview deleted nothing

    result = monitor.api_project_archive(monitor.ProjectArchiveBody(sample_dir=str(sample)))
    assert "ch1.zarr/" in result["deleted"]
    assert (analysis_home / "_archived" / "sample" / "config.json").exists()
    # job records for the archived sample are gone
    assert all(str(j.get("sample_dir")) != str(sample) for j in store.list_jobs())
