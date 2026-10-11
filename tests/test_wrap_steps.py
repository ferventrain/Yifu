"""Wrap --steps protocol test."""

from __future__ import annotations

import json
import sys
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


def _write_sample(root: Path, name: str = "mouse01") -> Path:
    sample = root / name
    sample.mkdir(parents=True)
    (sample / "config.json").write_text(
        json.dumps({"input": {"channels": {"signal": "1", "registration": "0"}}}),
        encoding="utf-8",
    )
    return sample


def test_wrap_steps_protocol(analysis_home: Path, capsys):
    """--steps declares the coarse checklist; STEP n lines advance it and the
    terminal state stamps step_states for the stepper UI."""
    from pipeline_modules.harness.__main__ import main as harness_main

    sample = _write_sample(analysis_home, "mouse05")
    script = (
        "print('STEP 1 download')\n"
        "print('PROGRESS 3/10 download')\n"
        "print('STEP 2 register')\n"
    )
    code = harness_main(
        [
            "wrap",
            "--sample-dir", str(sample),
            "--title", "unit steps",
            "--steps", "download,register,write-ims",
            "--", sys.executable, "-c", script,
        ]
    )
    assert code == 0
    out = capsys.readouterr().out
    job_id = json.loads([line for line in out.splitlines() if line.startswith("{")][0])["wrap"]["job_id"]

    store = ActiveStore()
    job = store.load_job(job_id)
    assert job["status"] == "done"
    progress = json.loads(Path(job["module_progress_path"]).read_text(encoding="utf-8"))
    assert progress["steps"] == ["download", "register", "write-ims"]
    assert progress["step_states"] == ["done", "done", "done"]

    view = store.job_view(job)
    assert view["progress"]["steps"] == ["download", "register", "write-ims"]
    assert view["progress"]["step_states"] == ["done", "done", "done"]


def test_wrap_steps_failure_marks_current_failed(analysis_home: Path, capsys):
    from pipeline_modules.harness.__main__ import main as harness_main

    sample = _write_sample(analysis_home, "mouse06")
    script = "import sys\nprint('STEP 2 register')\nsys.exit(3)"
    code = harness_main(
        [
            "wrap",
            "--sample-dir", str(sample),
            "--title", "unit steps fail",
            "--steps", "download,register",
            "--", sys.executable, "-c", script,
        ]
    )
    assert code == 3
    out = capsys.readouterr().out
    job_id = json.loads([line for line in out.splitlines() if line.startswith("{")][0])["wrap"]["job_id"]

    store = ActiveStore()
    job = store.load_job(job_id)
    assert job["status"] == "failed"
    progress = json.loads(Path(job["module_progress_path"]).read_text(encoding="utf-8"))
    assert progress["step_states"] == ["done", "failed"]
