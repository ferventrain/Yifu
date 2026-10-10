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


def test_board_priority_lives_on_batches_and_sorts(analysis_home: Path):
    pytest.importorskip("fastapi")
    import importlib

    batchA = analysis_home / "YF20991101"  # will get P1
    batchB = analysis_home / "YF20991202"  # will get P0
    batchC = analysis_home / "YF20991303"  # unprioritized
    for sample_dir in (batchA / "s1", batchB / "s1", batchC / "s1"):
        _write_sample(sample_dir.parent, sample_dir.name)

    store = ActiveStore()
    for batch in (batchA, batchB, batchC):
        store.add_job(batch / "s1")

    monitor = importlib.import_module("apps.pipeline_harness.main")
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(monitor, "store", store)

    # priority is set on the PROJECT (batch dir), not the sample
    monitor.api_projects_priority(monitor.ProjectPriorityBody(sample_dir=str(batchA), level=1))
    monitor.api_projects_priority(monitor.ProjectPriorityBody(sample_dir=str(batchB), level=0))

    payload = monitor.api_projects()
    batches = payload["batches"]
    # P0 batch first, then P1, then unprioritized
    assert [b["batch"] for b in batches] == ["YF20991202", "YF20991101", "YF20991303"]
    assert batches[0]["priority"] == 0
    assert batches[1]["priority"] == 1
    assert batches[2]["priority"] is None
    # samples inside a batch carry no priority of their own anymore
    assert all(p.get("priority") is None for b in batches for p in b["projects"])

    # clearing (level=-1) drops the batch back among the unprioritized
    monitor.api_projects_priority(monitor.ProjectPriorityBody(sample_dir=str(batchB), level=-1))
    payload = monitor.api_projects()
    assert [b["batch"] for b in payload["batches"]][0] == "YF20991101"

    with pytest.raises(Exception):
        monitor.api_projects_priority(
            monitor.ProjectPriorityBody(sample_dir=str(Path(__file__).absolute()), level=0)
        )
    monkeypatch.undo()


def test_resources_payload_reports_draw_and_capacity(analysis_home: Path):
    pytest.importorskip("fastapi")
    import importlib

    from datetime import datetime, timedelta, timezone

    sample = _write_sample(analysis_home / "batchX", "ot")
    (sample / "config.json").write_text(
        json.dumps({"input": {"channels": {"signal": "1", "registration": "0"}},
                    "segmentation": {"method": "cfos_unet"}}),
        encoding="utf-8",
    )
    store = ActiveStore()
    job = store.add_job(sample)
    job["status"] = "running"
    job["started_at"] = (datetime.now(timezone.utc) - timedelta(minutes=10)).isoformat()
    store.save_job(job)

    monitor = importlib.import_module("apps.pipeline_harness.main")
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(monitor, "store", store)
    payload = monitor.api_projects()
    resources = payload["resources"]
    assert resources["capacity"]["cpu"] >= 1
    assert resources["capacity"]["gpu"] >= 0
    # one running cfos-unet task draws cpu 8 / gpu 1
    assert resources["used"] == {"cpu": 8, "gpu": 1}
    assert resources["active_tasks"][0]["profile"] == "cfos-unet"
    labels = {cost["label"] for cost in resources["profile_costs"]}
    assert {"血管", "cFos UNet", "cFos 阈值"} <= labels
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
    assert "统一估时" in project["estimate"]["basis"]
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


def test_manifest_status_field_and_incomplete_exclusion(analysis_home: Path):
    pytest.importorskip("fastapi")
    import apps.pipeline_harness.projects as projects_mod
    import importlib
    from pipeline_modules.utils.run_manifest import (
        MANIFEST_STATUS_INCOMPLETE,
        build_run_manifest,
        write_run_manifest,
    )

    # default manifests are "completed"
    manifest = build_run_manifest(
        module="m", entrypoint="e", inputs={}, outputs=[], started_at=100.0, ended_at=160.0
    )
    assert manifest["status"] == "completed"
    assert manifest["duration_seconds"] == 60.0

    # an interrupted run writes an incomplete manifest
    sample = _write_sample(analysis_home, "ot")
    write_run_manifest(
        sample, module="main_pipeline", entrypoint="main", inputs={}, outputs=[],
        started_at=100.0, ended_at=900.0, status=MANIFEST_STATUS_INCOMPLETE,
    )
    payload = json.loads((sample / "_run_manifest.json").read_text(encoding="utf-8"))
    assert payload["status"] == "incomplete"

    # stage-times skips incomplete manifests even with a plausible duration
    other = _write_sample(analysis_home, "sham")
    (other / "_run_manifest.json").write_text(
        json.dumps({"module": "main_pipeline", "duration_seconds": 500.0, "status": "incomplete"}),
        encoding="utf-8",
    )
    good = _write_sample(analysis_home, "tmcao")
    (good / "_run_manifest.json").write_text(
        json.dumps({"module": "registration.ANTs_registration", "duration_seconds": 700.0}),
        encoding="utf-8",
    )

    projects_mod._STAGE_CACHE["payload"] = None
    projects_mod._STAGE_CACHE["at"] = 0.0
    monitor = importlib.import_module("apps.pipeline_harness.main")
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(monitor, "store", ActiveStore())
    result = monitor.api_stage_times()
    by_module = {s["module"]: s for s in result["stages"]}
    assert "main_pipeline" not in by_module  # incomplete excluded
    assert by_module["registration.ANTs_registration"]["median_s"] == 700.0
    assert result["incomplete_skipped"] == 2
    monkeypatch.undo()
    projects_mod._STAGE_CACHE["payload"] = None
    projects_mod._STAGE_CACHE["at"] = 0.0


def test_estimate_is_config_driven(analysis_home: Path):
    """Estimates come from config content x manifest stage medians, NOT from
    the sample's own run history: unet costs more than threshold, postprocess
    and skip_registration adjust the sum."""
    pytest.importorskip("fastapi")
    import apps.pipeline_harness.projects as projects_mod
    import importlib

    projects_mod._STAGE_CACHE["payload"] = None
    projects_mod._STAGE_CACHE["at"] = 0.0

    # seed per-stage medians via run manifests with the fine module names
    stage_seconds = {
        "preprocessing": 600.0,
        "pipeline_modules.preprocessing.zarr_to_registration_nii": 300.0,
        "registration.ANTs_registration": 3600.0,
        "segmentation.cfos_unet_inference": 7200.0,
        "segmentation.intensity_threshold": 600.0,
        "registration.region_signal_analysis_zarr_graph": 3000.0,
        "visualization.pipeline_qc": 60.0,
    }
    for name, seconds in stage_seconds.items():
        sample = analysis_home / f"stage_{name.split('.')[-1]}"
        sample.mkdir(parents=True, exist_ok=True)
        (sample / "_run_manifest.json").write_text(
            json.dumps({"module": name, "duration_seconds": seconds}), encoding="utf-8"
        )

    unet_cfg = {
        "input": {"channels": {"signal": "1", "registration": "0"}},
        "segmentation": {"method": "cfos_unet", "postprocess": {"enabled": True}},
    }
    thr_cfg = {
        "input": {"channels": {"signal": "1", "registration": "0"}},
        "segmentation": {"method": "threshold", "postprocess": {"enabled": False}},
    }
    skip_cfg = dict(unet_cfg)
    skip_cfg["skip_registration"] = True

    for name, cfg in (("unetA", unet_cfg), ("thrA", thr_cfg), ("skipA", skip_cfg)):
        sample = analysis_home / "bx" / name
        sample.mkdir(parents=True, exist_ok=True)
        (sample / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
        ActiveStore().add_job(sample)

    monitor = importlib.import_module("apps.pipeline_harness.main")
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(monitor, "store", ActiveStore())
    payload = monitor.api_projects()
    estimates = {}
    for batch in payload["batches"]:
        for project in batch["projects"]:
            estimates[project["sample_name"]] = project["estimate"]

    common = 600 + 300 + 3000 + 60
    assert estimates["unetA"]["seconds"] == common + 3600 + 7200
    assert estimates["thrA"]["seconds"] == common + 3600 + 600
    assert estimates["skipA"]["seconds"] == common + 7200
    assert "UNet" in estimates["unetA"]["basis"]
    assert "阈值" in estimates["thrA"]["basis"]
    assert "按 config 估算" in estimates["unetA"]["basis"]
    monkeypatch.undo()
    projects_mod._STAGE_CACHE["payload"] = None
    projects_mod._STAGE_CACHE["at"] = 0.0
