"""Board iteration tests: drag priority, anomaly sorting, estimates, stage times."""

from __future__ import annotations

import json
import shutil
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


def _write_sample(root: Path, name: str) -> Path:
    sample = root / name
    sample.mkdir(parents=True)
    (sample / "config.json").write_text(
        json.dumps({"input": {"channels": {"signal": "1", "registration": "0"}}}),
        encoding="utf-8",
    )
    return sample


def test_board_order_persists_and_sorts(analysis_home: Path):
    pytest.importorskip("fastapi")
    import importlib

    batch = analysis_home / "YF20991001"
    a = _write_sample(batch, "aaa")
    b = _write_sample(batch, "bbb")
    c = _write_sample(batch, "ccc")

    store = ActiveStore()
    for sample in (a, b, c):
        store.add_job(sample)

    monitor = importlib.import_module("apps.pipeline_harness.main")
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(monitor, "store", store)

    result = monitor.api_projects_order(monitor.BoardOrderBody(ordered=[str(b), str(a), str(c)]))
    assert result["order"]["sample_order"][0] == str(b)

    payload = monitor.api_projects()
    names = [p["sample_name"] for p in payload["batches"][0]["projects"]]
    assert names == ["bbb", "aaa", "ccc"], names

    # only [c, b] pinned -> aaa (unpinned) sorts after pinned ones
    monitor.api_projects_order(monitor.BoardOrderBody(ordered=[str(c), str(b)]))
    payload = monitor.api_projects()
    names = [p["sample_name"] for p in payload["batches"][0]["projects"]]
    assert names == ["ccc", "bbb", "aaa"], names

    with pytest.raises(Exception):
        monitor.api_projects_order(monitor.BoardOrderBody(ordered=[str(Path(__file__).absolute())]))
    monkeypatch.undo()


def test_anomalous_sample_sorts_last(analysis_home: Path):
    pytest.importorskip("fastapi")
    import importlib

    batch = analysis_home / "YF20991001"
    ok = _write_sample(batch, "healthy")
    broken = _write_sample(batch, "ghost")

    store = ActiveStore()
    store.add_job(ok)
    store.add_job(broken)
    # the anomaly arises when the dir disappears AFTER the run was recorded
    shutil.rmtree(broken)

    monitor = importlib.import_module("apps.pipeline_harness.main")
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(monitor, "store", store)
    payload = monitor.api_projects()
    projects = payload["batches"][0]["projects"]
    assert projects[-1]["sample_name"] == "ghost"
    assert projects[-1]["anomaly"] == "样本目录不存在"
    assert projects[0]["anomaly"] is None
    monkeypatch.undo()


def test_estimate_from_timing_history(analysis_home: Path):
    pytest.importorskip("fastapi")
    import importlib

    sample = _write_sample(analysis_home / "batchX", "ot")
    from pipeline_modules.harness.timing import record_step_timings

    record_step_timings(
        {
            "steps": [
                {"name": "Registration channel downsample", "seconds": 600.0},
                {"name": "Atlas registration and label outputs", "seconds": 1200.0},
                {"name": "Signal preprocessing and Zarr conversion", "seconds": 1800.0},
                {"name": "Segmentation", "seconds": 3600.0},
            ]
        },
        active=analysis_home / "_active",
    )

    store = ActiveStore()
    store.add_job(sample)
    monitor = importlib.import_module("apps.pipeline_harness.main")
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(monitor, "store", store)
    payload = monitor.api_projects()
    project = payload["batches"][0]["projects"][0]
    assert project["estimate"] is not None
    assert project["estimate"]["seconds"] == 7200.0
    assert "历史中位" in project["estimate"]["basis"]
    monkeypatch.undo()


def test_active_batches_flagged_first(analysis_home: Path):
    pytest.importorskip("fastapi")
    import importlib

    active_sample = _write_sample(analysis_home / "_active" / "YF20991001", "in_active")
    outside_sample = _write_sample(analysis_home / "elsewhere", "outside")

    store = ActiveStore()
    store.add_job(active_sample)
    store.add_job(outside_sample)

    monitor = importlib.import_module("apps.pipeline_harness.main")
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(monitor, "store", store)
    payload = monitor.api_projects()
    flags = [(b["batch"], bool(b.get("under_active"))) for b in payload["batches"]]
    assert ("YF20991001", True) in flags
    assert ("elsewhere", False) in flags
    # under_active batches come first
    assert flags == sorted(flags, key=lambda item: not item[1])
    monkeypatch.undo()


def test_stage_times_aggregates_manifests(analysis_home: Path):
    pytest.importorskip("fastapi")
    import apps.pipeline_harness.projects as projects_mod
    import importlib

    projects_mod._STAGE_CACHE["payload"] = None
    projects_mod._STAGE_CACHE["at"] = 0.0

    for name, duration in (("ot", 100.0), ("sham", 300.0)):
        sample = _write_sample(analysis_home, name)
        (sample / "_run_manifest.json").write_text(
            json.dumps({"module": "registration.ANTs_registration", "duration_seconds": duration}),
            encoding="utf-8",
        )
    sample = _write_sample(analysis_home, "tmcao")
    (sample / "_run_manifest.json").write_text(
        json.dumps({"module": "segmentation.cfos_unet_inference", "duration_seconds": 900.0}),
        encoding="utf-8",
    )

    monitor = importlib.import_module("apps.pipeline_harness.main")
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(monitor, "store", ActiveStore())
    payload = monitor.api_stage_times()
    by_module = {s["module"]: s for s in payload["stages"]}
    assert by_module["registration.ANTs_registration"]["count"] == 2
    assert by_module["registration.ANTs_registration"]["median_s"] == 200.0
    assert by_module["segmentation.cfos_unet_inference"]["median_s"] == 900.0
    monkeypatch.undo()
    projects_mod._STAGE_CACHE["payload"] = None
    projects_mod._STAGE_CACHE["at"] = 0.0
