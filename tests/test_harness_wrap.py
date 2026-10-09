"""Tests for the wrap CLI, the minimal report() API, and the lenient reader."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from pipeline_modules.harness.jsonio import write_json_atomic
from pipeline_modules.harness.progress import (
    PM_PROGRESS_FILENAME,
    report,
    report_done,
    report_failed,
)
from pipeline_modules.harness.queue import ActiveStore, _read_module_progress


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
    (sample / "config.json").write_text(
        json.dumps({"input": {"channels": {"signal": "1", "registration": "0"}}}),
        encoding="utf-8",
    )
    return sample


# ------------------------------------------------------------------- report()


def test_report_writes_pm_progress_schema2(analysis_home: Path, monkeypatch: pytest.MonkeyPatch):
    sample = _write_sample(analysis_home)
    monkeypatch.delenv("YIFU_PROGRESS_FILE", raising=False)

    report(done=3, total=10, phase="apply mask", sample_dir=sample)
    payload = json.loads((sample / PM_PROGRESS_FILENAME).read_text(encoding="utf-8"))
    assert payload["done"] == 3
    assert payload["total"] == 10
    assert payload["phase"] == "apply mask"
    assert payload["status"] == "running"
    assert payload["schema_version"] == "2"

    report_done(sample_dir=sample)
    payload = json.loads((sample / PM_PROGRESS_FILENAME).read_text(encoding="utf-8"))
    assert payload["status"] == "done"
    assert payload["done"] == payload["total"] == 10

    report_failed("boom", sample_dir=sample)
    payload = json.loads((sample / PM_PROGRESS_FILENAME).read_text(encoding="utf-8"))
    assert payload["status"] == "failed"
    assert payload["error"] == "boom"


def test_report_prefers_env_progress_file(analysis_home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    sample = _write_sample(analysis_home)
    env_progress = tmp_path / "run_progress.json"
    env_progress.write_text(json.dumps({"status": "running", "unit_done": 0, "unit_total": 0}), encoding="utf-8")
    monkeypatch.setenv("YIFU_PROGRESS_FILE", str(env_progress))

    report(done=5, total=20, phase="phase-a")
    # throttle: report_child_units may skip non-forced updates within 2s, so
    # check the file only if it was updated; otherwise force via total==done
    report(done=20, total=20, phase="phase-a")
    payload = json.loads(env_progress.read_text(encoding="utf-8"))
    assert payload["unit_total"] == 20

    # pm_progress.json must NOT have been written when env is set
    assert not (sample / PM_PROGRESS_FILENAME).exists()


# ------------------------------------------------------- lenient module reader


def test_read_module_progress_accepts_all_schemas(tmp_path: Path):
    schema1 = tmp_path / "s1.json"
    write_json_atomic(schema1, {"unit_done": 4, "unit_total": 9, "phase": "skeleton", "phase_started_at": "2026-10-09T00:00:00+00:00"})
    got = _read_module_progress(str(schema1))
    assert got == {"unit_done": 4, "unit_total": 9, "phase": "skeleton", "phase_started_at": "2026-10-09T00:00:00+00:00"}

    schema2 = tmp_path / "s2.json"
    write_json_atomic(schema2, {"done": 7, "total": 12, "phase": "ims rewrite", "status": "running"})
    got = _read_module_progress(str(schema2))
    assert got["unit_done"] == 7 and got["unit_total"] == 12 and got["phase"] == "ims rewrite"

    percent = tmp_path / "s3.json"
    write_json_atomic(percent, {"percent": 42.0})
    got = _read_module_progress(str(percent))
    assert got["unit_total"] == 100 and got["unit_done"] == 42

    assert _read_module_progress(str(tmp_path / "missing.json")) is None
    garbage = tmp_path / "s4.json"
    write_json_atomic(garbage, {"something": "else"})
    assert _read_module_progress(str(garbage)) is None  # no progress signal at all


def test_external_view_falls_back_to_pm_progress(analysis_home: Path):
    sample = _write_sample(analysis_home)
    report(done=2, total=5, phase="manual chain", sample_dir=sample)

    store = ActiveStore()
    store.attach_external(
        sample,
        title="manual chain",
        pid=0,  # dead pid: status file absent -> failed; we only need the view
        log_path=None,
        status_path=None,
    )
    view = store.list_job_views()[0]
    # unit progress from pm_progress.json must surface in the external view
    assert view["progress"]["unit_done"] == 2
    assert view["progress"]["unit_total"] == 5
    assert view["progress"]["unit_phase"] == "manual chain"


def test_external_running_job_gets_heartbeat(analysis_home: Path, tmp_path: Path):
    sample = _write_sample(analysis_home)
    status = tmp_path / "status.txt"
    status.write_text(f"RUNNING Step 1/2: rewrite\n2026-10-09T10:00:00\n", encoding="utf-8")

    store = ActiveStore()
    store.attach_external(
        sample,
        title="chain",
        pid=0 + 1,  # placeholder; replaced below with the live test process pid
        status_path=status,
    )
    # re-attach with OUR pid so the liveness check passes
    job = store.list_jobs()[0]
    job["pid"] = 0
    store.attach_external(
        sample,
        title="chain",
        pid=0 or None,
        status_path=status,
    )
    # simpler: directly exercise sync with a live pid
    job = store.list_jobs()[0]
    import os

    job["pid"] = os.getpid()
    job["pid_created_at"] = None
    synced = store.sync_external_job(job)
    assert synced["status"] == "running"
    assert synced.get("heartbeat_at"), "live external job must carry a heartbeat"
    view = store.job_view(synced)
    assert view["progress"].get("heartbeat_at")


# ------------------------------------------------------------------ wrap CLI


def _wrap_command(script: str) -> list[str]:
    return [sys.executable, "-c", script]


def test_wrap_tracks_success_and_progress_lines(analysis_home: Path, capsys):
    from pipeline_modules.harness.__main__ import main as harness_main

    sample = _write_sample(analysis_home, "mouse02")
    script = (
        "print('hello')\n"
        "print('PROGRESS 3/10 apply mask')\n"
        "print('PROGRESS 10/10 apply mask')\n"
    )
    code = harness_main(
        [
            "wrap",
            "--sample-dir", str(sample),
            "--title", "unit wrap success",
            "--", *_wrap_command(script),
        ]
    )
    assert code == 0
    out = capsys.readouterr().out
    job_id = json.loads([line for line in out.splitlines() if line.startswith("{")][0])["wrap"]["job_id"]

    store = ActiveStore()
    job = store.load_job(job_id)
    assert job["status"] == "done"
    assert job["title"] == "unit wrap success"
    # progress file harvested from stdout lines
    progress = json.loads(Path(job["module_progress_path"]).read_text(encoding="utf-8"))
    assert progress["unit_done"] == 10 and progress["unit_total"] == 10
    # log captured the child output
    log_text = Path(job["log_path"]).read_text(encoding="utf-8")
    assert "hello" in log_text and "PROGRESS 10/10" in log_text


def test_wrap_maps_nonzero_exit_to_failed(analysis_home: Path, capsys):
    from pipeline_modules.harness.__main__ import main as harness_main

    sample = _write_sample(analysis_home, "mouse03")
    code = harness_main(
        [
            "wrap",
            "--sample-dir", str(sample),
            "--title", "unit wrap failure",
            "--", *_wrap_command("import sys; print('boom'); sys.exit(3)"),
        ]
    )
    assert code == 3
    out = capsys.readouterr().out
    job_id = json.loads([line for line in out.splitlines() if line.startswith("{")][0])["wrap"]["job_id"]
    store = ActiveStore()
    job = store.load_job(job_id)
    assert job["status"] == "failed"
    assert "exit code 3" in str(job["error"])


def test_wrap_requires_command(analysis_home: Path):
    from pipeline_modules.harness.__main__ import main as harness_main

    sample = _write_sample(analysis_home, "mouse04")
    code = harness_main(["wrap", "--sample-dir", str(sample), "--"])
    assert code == 2
