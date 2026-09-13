"""Analysis-root and Active-queue locations for the pipeline harness."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

ANALYSIS_ROOT_ENV = "YIFU_ANALYSIS_ROOT"
ACTIVE_DIR_ENV = "YIFU_ACTIVE_DIR"
DEFAULT_ANALYSIS_ROOT = Path(r"H:\arivis-analysis")
ACTIVE_DIRNAME = "_active"


def analysis_root() -> Path:
    raw = os.environ.get(ANALYSIS_ROOT_ENV, "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    return DEFAULT_ANALYSIS_ROOT.resolve()


def active_dir(*, root: Path | None = None) -> Path:
    raw = os.environ.get(ACTIVE_DIR_ENV, "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    return (root or analysis_root()) / ACTIVE_DIRNAME


def jobs_dir(active: Path | None = None) -> Path:
    return (active or active_dir()) / "jobs"


def runs_dir(active: Path | None = None) -> Path:
    return (active or active_dir()) / "runs"


def queue_path(active: Path | None = None) -> Path:
    return (active or active_dir()) / "queue.json"


def timing_history_path(active: Path | None = None) -> Path:
    return (active or active_dir()) / "timing_history.json"


def job_file(job_id: str, active: Path | None = None) -> Path:
    return jobs_dir(active) / f"{job_id}.json"


def run_dir(job_id: str, active: Path | None = None) -> Path:
    return runs_dir(active) / job_id


def progress_path(job_id: str, active: Path | None = None) -> Path:
    return run_dir(job_id, active) / "progress.json"


def stdout_log_path(job_id: str, active: Path | None = None) -> Path:
    return run_dir(job_id, active) / "stdout.log"


def open_path_in_file_manager(path: Path) -> Path:
    """Open ``path`` in the OS file manager. Files open their parent folder."""
    target = Path(path).expanduser()
    if not target.exists():
        raise FileNotFoundError(str(target))
    folder = target if target.is_dir() else target.parent
    folder = folder.resolve()
    if os.name == "nt":
        # explorer.exe returns a non-zero code even when it succeeds.
        if target.is_file():
            subprocess.Popen(["explorer", f"/select,{os.path.normpath(str(target.resolve()))}"])
        else:
            subprocess.Popen(["explorer", os.path.normpath(str(folder))])
    else:
        subprocess.Popen(["xdg-open", str(folder)])
    return folder
