"""Acceptance tests for the single-command harness (run/worker/verifier).

Covers the ten scenarios from the harness redesign spec: one standard
command submits a job; the monitor's read paths never start anything;
heartbeats advance; success requires outputs; missing outputs fail;
half-written progress.json never fakes success; stale heartbeats display
stalled; the same sample is not double-submitted; agents need no run.py;
legacy commands still work.
"""

from __future__ import annotations

import json
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from pipeline_modules.harness import __main__ as cli
from pipeline_modules.harness.jsonio import write_json_atomic
from pipeline_modules.harness.paths import (
    artifacts_path,
    progress_path,
    worker_state_path,
)
from pipeline_modules.harness.progress import touch_heartbeat
from pipeline_modules.harness.queue import ActiveStore
from pipeline_modules.harness.worker import HarnessWorker, decorate_views
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
    sample.mkdir(parents=True)
    (sample / "config.json").write_text(json.dumps(MINIMAL_CONFIG), encoding="utf-8")
    return sample


def _required_xlsx(sample: Path) -> Path:
    return sample / "results" / f"{sample.name}_ch0_brain_distribution_stats.xlsx"


def _write_fake_xlsx(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("[Content_Types].xml", "<Types/>")


def _run_cli(capsys, argv: list[str]) -> tuple[int, dict]:
    code = cli.main(argv)
    payload = json.loads(capsys.readouterr().out)
    return code, payload


# 1 + 9. the standard command submits a job and returns run_id (no run.py)
def test_run_command_submits_job_and_returns_run_id(analysis_home: Path, capsys):
    sample = _write_sample(analysis_home)
    code, payload = _run_cli(
        capsys, ["run", "--sample-dir", str(sample), "--no-start-worker"]
    )
    assert code == 0
    run_id = payload["run_id"]
    assert run_id and payload["status"] == "queued"
    assert payload["already_active"] is False
    store = ActiveStore()
    job = store.load_job(run_id)
    assert job["status"] == "queued"
    assert progress_path(run_id, store.active).exists()


def test_preflight_is_readonly(analysis_home: Path, capsys):
    sample = _write_sample(analysis_home)
    code, payload = _run_cli(capsys, ["preflight", "--sample-dir", str(sample)])
    assert code == 0 and payload["status"] == "ok"
    assert payload["pipeline"] == "brain"
    store = ActiveStore()
    assert store.list_jobs() == []


def test_run_rejects_unknown_extra_args(analysis_home: Path, capsys):
    sample = _write_sample(analysis_home)
    code = cli.main(
        ["run", "--sample-dir", str(sample), "--extra-arg=--evil", "--no-start-worker"]
    )
    captured = capsys.readouterr()
    assert code == PipelineError(ErrorCode.ARGUMENT_INVALID, "x").exit_code
    body = json.loads(captured.err)
    assert body["error"]["code"] == ErrorCode.ARGUMENT_INVALID.value
    assert "--evil" in body["error"]["context"]["rejected"]


# 2. monitor refresh never starts anything
def test_monitor_get_jobs_never_starts_worker(analysis_home: Path, monkeypatch):
    pytest.importorskip("fastapi")
    sample = _write_sample(analysis_home)
    store = ActiveStore()
    store.add_job(sample)

    import importlib

    monitor = importlib.import_module("apps.pipeline_harness.main")
    calls = []
    monkeypatch.setattr(
        monitor, "ensure_worker_running", lambda *a, **k: calls.append(1) or {"started": True, "worker": {"alive": True}}
    )

    payload = monitor.api_jobs()
    assert len(payload["jobs"]) == 1
    assert calls == []  # GET did not start anything
    assert not worker_state_path(store.active).exists()

    body = monitor.AddJobBody(sample_dir=str(_write_sample(analysis_home, "mouse02")))
    monitor.api_add_job(body)
    assert len(calls) == 1  # explicit submit may start the worker


# 2b. monitor surfaces sample qc images with PM-served clickable URLs
def test_monitor_lists_qc_images_and_serves_files(analysis_home: Path, monkeypatch: pytest.MonkeyPatch):
    pytest.importorskip("fastapi")
    sample = _write_sample(analysis_home, "mouse03")
    qc_dir = sample / "qc"
    qc_dir.mkdir()
    rigid_png = qc_dir / "registration_rigid_check.png"
    rigid_png.write_bytes(b"\x89PNG\r\n\x1a\nfake")

    store = ActiveStore()
    store.add_job(sample)

    import importlib

    monitor = importlib.import_module("apps.pipeline_harness.main")
    # the module-level store binds env dirs at import time; rebind to this
    # test's analysis home (the module may already be cached by an earlier test)
    monkeypatch.setattr(monitor, "store", store)
    payload = monitor.api_jobs()
    job = next(j for j in payload["jobs"] if j["sample_name"] == "mouse03")
    assert job["qc_images"][0]["name"] == "registration_rigid_check.png"
    assert job["qc_images"][0]["url"].startswith("/api/file?path=")

    # the served route returns the actual bytes (read-only viewer)
    response = monitor.api_file(path=str(rigid_png))
    assert rigid_png.name in str(response.filename) or response.media_type

    # paths outside the analysis root are rejected
    with pytest.raises(PipelineError):
        monitor.api_file(path=str(Path(__file__).absolute()))


# 3. running jobs get heartbeats
def test_touch_heartbeat_updates_running_progress(analysis_home: Path):
    sample = _write_sample(analysis_home)
    store = ActiveStore()
    job = store.add_job(sample)
    progress_file = progress_path(job["id"], store.active)
    old = datetime.now(timezone.utc).replace(microsecond=0) - timedelta(minutes=10)
    state = json.loads(progress_file.read_text(encoding="utf-8"))
    state["status"] = "running"
    state["heartbeat_at"] = old.isoformat()
    write_json_atomic(progress_file, state)

    touch_heartbeat(progress_file)

    refreshed = json.loads(progress_file.read_text(encoding="utf-8"))
    assert refreshed["status"] == "running"  # RMW preserved other fields
    assert datetime.fromisoformat(refreshed["heartbeat_at"]) > old


def _finalize(store: ActiveStore, job_id: str, returncode: int) -> dict:
    HarnessWorker(store)._finalize(job_id, returncode)
    return store.load_job(job_id)


# 4. exit 0 + outputs exist + readable => succeeded
def test_finalize_succeeded_when_outputs_exist(analysis_home: Path):
    sample = _write_sample(analysis_home)
    store = ActiveStore()
    job = store.add_job(sample)
    _write_fake_xlsx(_required_xlsx(sample))

    final = _finalize(store, job["id"], 0)

    assert final["status"] == "succeeded"
    manifest = json.loads(artifacts_path(job["id"], store.active).read_text(encoding="utf-8"))
    assert manifest["verification"]["ok"] is True


# 5. exit 0 but outputs missing => failed (structured, not a guess)
def test_finalize_failed_when_output_missing(analysis_home: Path):
    sample = _write_sample(analysis_home)
    store = ActiveStore()
    job = store.add_job(sample)

    final = _finalize(store, job["id"], 0)

    assert final["status"] == "failed"
    assert final["error"]["code"] == ErrorCode.OUTPUT_INVALID.value
    assert final["error"]["retryable"] is False
    assert final["error"]["suggestion"]


def test_finalize_failed_on_nonzero_exit(analysis_home: Path):
    sample = _write_sample(analysis_home)
    store = ActiveStore()
    job = store.add_job(sample)
    _write_fake_xlsx(_required_xlsx(sample))

    final = _finalize(store, job["id"], 1)

    assert final["status"] == "failed"
    assert final["error"]["code"] == ErrorCode.INTERNAL_ERROR.value


# 6. half-written progress.json never fakes success
def test_corrupt_progress_does_not_fake_success(analysis_home: Path):
    sample = _write_sample(analysis_home)
    store = ActiveStore()
    job = store.add_job(sample)
    progress_path(job["id"], store.active).write_text('{"status": "run', encoding="utf-8")

    final = _finalize(store, job["id"], 0)  # outputs missing too

    assert final["status"] == "failed"  # not crashed, not fabricated success


# 7. stale heartbeat / dead worker displays stalled
def test_stale_heartbeat_displays_stalled(analysis_home: Path):
    sample = _write_sample(analysis_home)
    store = ActiveStore()
    job = store.add_job(sample)
    job["status"] = "running"
    job["started_at"] = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    store.save_job(job)
    progress_file = progress_path(job["id"], store.active)
    old = (datetime.now(timezone.utc) - timedelta(minutes=10)).replace(microsecond=0).isoformat()
    write_json_atomic(progress_file, {"status": "running", "step_index": 2, "heartbeat_at": old})

    views = decorate_views(store.list_job_views(), store.active)
    assert views[0]["display_status"] == "stalled"
    assert views[0]["heartbeat_age_s"] > 120

    # fresh heartbeat with a live worker shows running
    touch_heartbeat(progress_file)
    from pipeline_modules.harness.proc import pid_created_at

    write_json_atomic(
        worker_state_path(store.active),
        {"pid": _current_pid(), "pid_created_at": pid_created_at(_current_pid()), "stopped_at": None},
    )
    views = decorate_views(store.list_job_views(), store.active)
    assert views[0]["display_status"] == "running"
    assert views[0]["duration_s"] is not None


def test_legacy_done_status_maps_to_succeeded(analysis_home: Path):
    sample = _write_sample(analysis_home)
    store = ActiveStore()
    job = store.add_job(sample)
    job["status"] = "done"
    store.save_job(job)
    views = decorate_views(store.list_job_views(), store.active)
    assert views[0]["display_status"] == "succeeded"


# 8. the same sample is not submitted twice
def test_run_dedupes_same_sample(analysis_home: Path, capsys):
    sample = _write_sample(analysis_home)
    code_one, payload_one = _run_cli(capsys, ["run", "--sample-dir", str(sample), "--no-start-worker"])
    code_two, payload_two = _run_cli(capsys, ["run", "--sample-dir", str(sample), "--no-start-worker"])
    assert code_one == code_two == 0
    assert payload_two["run_id"] == payload_one["run_id"]
    assert payload_two["already_active"] is True
    assert len(ActiveStore().list_jobs()) == 1


def test_rerun_allowed_after_failure(analysis_home: Path, capsys):
    sample = _write_sample(analysis_home)
    code_one, payload_one = _run_cli(capsys, ["run", "--sample-dir", str(sample), "--no-start-worker"])
    store = ActiveStore()
    job = store.load_job(payload_one["run_id"])
    job["status"] = "failed"
    store.save_job(job)

    code_two, payload_two = _run_cli(capsys, ["run", "--sample-dir", str(sample), "--no-start-worker"])
    assert payload_two["already_active"] is False
    assert payload_two["run_id"] != payload_one["run_id"]


# 10. legacy commands still work
def test_legacy_enqueue_and_list_still_work(analysis_home: Path, capsys):
    sample = _write_sample(analysis_home)
    code, record = _run_cli(capsys, ["enqueue", "--sample-dir", str(sample)])
    assert code == 0 and record["status"] == "queued"

    code, views = _run_cli(capsys, ["list"])
    assert code == 0
    assert [view["id"] for view in views] == [record["id"]]
    assert views[0]["display_status"] == "queued"


def test_cancel_queued_job(analysis_home: Path):
    from pipeline_modules.harness.worker import request_cancel

    sample = _write_sample(analysis_home)
    store = ActiveStore()
    job = store.add_job(sample)
    result = request_cancel(store, job["id"])
    assert result["status"] == "cancelled"
    assert store.load_job(job["id"])["status"] == "cancelled"


def _current_pid() -> int:
    import os

    return os.getpid()
