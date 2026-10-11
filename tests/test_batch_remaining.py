"""Batch remaining time = sum of UNFINISHED step history medians (not dynamic ETA)."""

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


def _write_sample(root: Path, name: str) -> Path:
    sample = root / name
    sample.mkdir(parents=True)
    (sample / "config.json").write_text(
        json.dumps({"input": {"channels": {"signal": "1", "registration": "0"}}}),
        encoding="utf-8",
    )
    return sample


def _seed_running(store: ActiveStore, sample: Path, step_index: int) -> None:
    """Add a job whose progress.json looks mid-pipeline at step_index."""
    from datetime import datetime, timezone

    job = store.add_job(sample)
    job_id = job["id"]
    job["status"] = "running"
    job["started_at"] = datetime.now(timezone.utc).isoformat()
    store.save_job(job)

    from pipeline_modules.harness.paths import progress_path

    progress = json.loads(progress_path(job_id, store.active).read_text(encoding="utf-8"))
    progress["status"] = "running"
    progress["step_index"] = step_index
    progress["step_total"] = 6
    progress["step_name"] = "Segmentation"
    progress["steps"] = [
        {"index": i + 1, "name": name, "ended_at": "x" if i + 1 < step_index else None}
        for i, name in enumerate(
            [
                "Registration channel downsample",
                "Atlas registration and label outputs",
                "Signal preprocessing and Zarr conversion",
                "Segmentation",
                "Region density analysis",
                "Vessel network reconstruction and region morphology",
            ]
        )
    ]
    progress_path(job_id, store.active).write_text(json.dumps(progress), encoding="utf-8")


def test_batch_remaining_uses_unfinished_step_medians(analysis_home: Path):
    pytest.importorskip("fastapi")
    import importlib

    from pipeline_modules.harness.timing import record_step_timings

    # historical step medians: steps 4-6 (Segmentation onward) sum to 1h
    record_step_timings(
        {
            "steps": [
                {"name": "Registration channel downsample", "seconds": 600.0},
                {"name": "Atlas registration and label outputs", "seconds": 1200.0},
                {"name": "Signal preprocessing and Zarr conversion", "seconds": 1800.0},
                {"name": "Segmentation", "seconds": 1200.0},
                {"name": "Region density analysis", "seconds": 1500.0},
                {"name": "Vessel network reconstruction and region morphology", "seconds": 900.0},
            ]
        },
        active=analysis_home / "_active",
    )

    sample = _write_sample(analysis_home / "batchX", "ot")
    store = ActiveStore()
    _seed_running(store, sample, step_index=4)  # currently ON step 4 (Segmentation)

    monitor = importlib.import_module("apps.pipeline_harness.main")
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(monitor, "store", store)
    payload = monitor.api_projects()
    batch = payload["batches"][0]
    # remaining = step4 + step5 + step6 = 1200 + 1500 + 900 = 3600s
    assert batch["remaining_s"] == pytest.approx(3600.0), batch["remaining_s"]
    monkeypatch.undo()
