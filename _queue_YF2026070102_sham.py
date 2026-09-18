"""Wait for the YF2026070102 (SICM) cFos run, then launch the sham run.

Polls the SICM driver status file / job pid every 60 seconds. When the SICM
run finishes (ALL DONE / FAILED / CANCELLED, or the driver died unexpectedly),
launches the sham driver and waits for it. Exit code mirrors the sham driver.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

os.environ.setdefault("YIFU_ANALYSIS_ROOT", r"S:\Arivis_Analysis")
sys.path.insert(0, r"S:\Yifu")

from pipeline_modules.harness.proc import pid_is_alive

SICM_SAMPLE = Path(r"S:\Arivis_Analysis\YF2026070102")
SICM_STATUS = SICM_SAMPLE / "cfos_pipeline.status.txt"
SHAM_DRIVER = Path(r"S:\Yifu\_run_YF2026070102_sham_cfos.py")
SHAM_LOG = Path(r"S:\Arivis_Analysis\YF2026070102_sham\cfos_pipeline.log")
REPO = Path(r"S:\Yifu")
PYTHON = sys.executable
TERMINAL_PREFIXES = ("ALL DONE", "FAILED", "CANCELLED")


def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def sicm_first_line() -> str:
    if not SICM_STATUS.exists():
        return ""
    try:
        text = SICM_STATUS.read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return ""
    return text.splitlines()[0].strip() if text else ""


def sicm_driver_pid() -> int | None:
    jobs_dir = Path(os.environ["YIFU_ANALYSIS_ROOT"]) / "_active" / "jobs"
    if not jobs_dir.exists():
        return None
    candidates = []
    for path in jobs_dir.glob("*.json"):
        if "YF2026070102" not in path.stem or "sham" in path.stem:
            continue
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if str(record.get("kind")) == "external" and str(record.get("sample_dir", "")).lower() == str(SICM_SAMPLE).lower():
            candidates.append(record)
    if not candidates:
        return None
    latest = max(candidates, key=lambda record: str(record.get("created_at") or ""))
    pid = latest.get("pid")
    try:
        return int(pid) if pid else None
    except (TypeError, ValueError):
        return None


def sicm_finished() -> str | None:
    first = sicm_first_line()
    for prefix in TERMINAL_PREFIXES:
        if first.startswith(prefix):
            return f"status file says: {first}"
    pid = sicm_driver_pid()
    if pid is not None and not pid_is_alive(pid) and first.startswith("RUNNING"):
        return f"driver pid {pid} died while status says RUNNING"
    return None


def main() -> int:
    print(f"[{now()}] watching SICM run (status={SICM_STATUS})", flush=True)
    while True:
        reason = sicm_finished()
        if reason:
            break
        time.sleep(60)
    print(f"[{now()}] SICM finished: {reason}", flush=True)
    time.sleep(15)

    SHAM_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(SHAM_LOG, "a", encoding="utf-8") as log_file:
        proc = subprocess.Popen(
            [PYTHON, str(SHAM_DRIVER)],
            cwd=str(REPO),
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )
    print(f"[{now()}] launched sham driver pid={proc.pid}", flush=True)
    code = proc.wait()
    print(f"[{now()}] sham driver exited with code {code}", flush=True)
    return 0 if code == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
