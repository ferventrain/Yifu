"""Serial orchestrator for the six YF2025063002 vessel bilateral runs.

Waits while any harness job is running (main or external), then launches
_run_YF2025063002_vessel.py for MPTP_1, MPTP_2, SEBL_1, SEBL_2, PBS_1, PBS_2
one after another. Runs one sample at a time; failures do not block the
remaining samples. The final exit code is non-zero if any sample failed.

Detached usage (survives this console):
  start /b <python> _queue_YF2025063002_vessel.py
Log: S:\\Arivis_Analysis\\_active\\YF2025063002\\vessel_queue.log
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

REPO = Path(r"S:\Yifu")
BATCH_DIR = Path(r"S:\Arivis_Analysis\_active\YF2025063002")
QUEUE_LOG = BATCH_DIR / "vessel_queue.log"
DRIVER = REPO / "_run_YF2025063002_vessel.py"
SAMPLES = ["MPTP_1", "MPTP_2", "SEBL_1", "SEBL_2", "PBS_1", "PBS_2"]

os.environ["YIFU_ANALYSIS_ROOT"] = r"S:\Arivis_Analysis"
os.environ["PYTHONUNBUFFERED"] = "1"
sys.path.insert(0, str(REPO))

LOG_FILE = open(QUEUE_LOG, "a", encoding="utf-8")
sys.stdout = LOG_FILE  # detached-safe: no console needed
sys.stderr = LOG_FILE


def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(msg: str) -> None:
    text = f"[{now()}] [queue] {msg}"
    print(text, flush=True)
    LOG_FILE.flush()


def wait_until_idle() -> None:
    from pipeline_modules.harness.queue import ActiveStore

    store = ActiveStore()
    while True:
        running = [job for job in store.list_jobs() if str(job.get("status")) == "running"]
        if not running:
            return
        names = ", ".join(str(job.get("title") or job.get("id")) for job in running)
        log(f"harness busy (running: {names}); waiting 60 s")
        time.sleep(60)


def main() -> int:
    log(f"=== YF2025063002 vessel queue start, samples={SAMPLES} ===")
    results: dict[str, int] = {}
    for index, sample in enumerate(SAMPLES, start=1):
        wait_until_idle()
        log(f"({index}/{len(SAMPLES)}) launching {sample}")
        proc = subprocess.run(
            [sys.executable, str(DRIVER), sample],
            cwd=str(REPO),
        )
        results[sample] = proc.returncode
        log(f"({index}/{len(SAMPLES)}) {sample} driver exited with code {proc.returncode}")
        time.sleep(15)

    failed = [name for name, code in results.items() if code != 0]
    log(f"=== queue finished: {len(results) - len(failed)}/{len(results)} ok"
        + (f", failed: {', '.join(failed)}" if failed else "") + " ===")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
