"""Watchdog for the vessel bilateral queue: self-heal after ANY kill event.

Checks every run (scheduled every 15 min):
  1. Any expected sample whose status file does not say ALL DONE?
  2. Is an orchestrator (_queue_YF2025063002_vessel.py) alive?
If work remains and no orchestrator is alive, relaunch the queue scheduled
task. Whatever killed the previous chain (NAS drop, external kill, reboot
cleanup), the queue resumes from markers within one watchdog period.

Log: <YF2025063002 batch>/vessel_watchdog.log
"""
from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, r"S:\Yifu")

QUEUE_TASK = "YifuVesselQueueRerun"
BATCH_DIR = Path(r"S:\Arivis_Analysis\_active\YF2025063002")
LOG_PATH = BATCH_DIR / "vessel_watchdog.log"
SAMPLE_STATUS = {
    "YF2025063002_MPTP_1": BATCH_DIR / "YF2025063002_MPTP_1" / "vessel_pipeline.status.txt",
    "YF2025063002_MPTP_2": BATCH_DIR / "YF2025063002_MPTP_2" / "vessel_pipeline.status.txt",
    "YF2025063002_SEBL_1": BATCH_DIR / "YF2025063002_SEBL_1" / "vessel_pipeline.status.txt",
    "YF2025063002_SEBL_2": BATCH_DIR / "YF2025063002_SEBL_2" / "vessel_pipeline.status.txt",
    "YF2025063002_PBS_1": BATCH_DIR / "YF2025063002_PBS_1" / "vessel_pipeline.status.txt",
    "YF2025063002_PBS_2": BATCH_DIR / "YF2025063002_PBS_2" / "vessel_pipeline.status.txt",
    "YF2026051202_A125": Path(
        r"S:\Arivis_Analysis\_active\YF2026051202\YF2026051202_A125\vessel_pipeline.status.txt"
    ),
}


def log(message: str) -> None:
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(f"[{stamp}] {message}\n")


def pending_samples() -> list[str]:
    pending = []
    for name, status in SAMPLE_STATUS.items():
        try:
            first = status.read_text(encoding="utf-8", errors="replace").splitlines()[0].strip()
        except (OSError, IndexError):
            pending.append(name)
            continue
        if not first.startswith("ALL DONE"):
            pending.append(name)
    return pending


def orchestrator_alive() -> bool:
    try:
        import psutil
    except ImportError:
        return True  # cannot check; do not risk double-launch
    me = os.getpid()
    for proc in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            if proc.info["pid"] == me or "python" not in (proc.info["name"] or "").lower():
                continue
            cmd_list = proc.info["cmdline"] or []
            if len(cmd_list) >= 2 and cmd_list[1].endswith("_queue_YF2025063002_vessel.py"):
                return True
        except Exception:
            continue
    return False


def main() -> int:
    pending = pending_samples()
    if not pending:
        log("all samples ALL DONE; watchdog idle")
        return 0
    if orchestrator_alive():
        log(f"orchestrator alive; pending={pending}")
        return 0
    result = subprocess.run(
        ["schtasks", "/Run", "/TN", QUEUE_TASK],
        capture_output=True, text=True, encoding="gbk", errors="replace",
    )
    log(f"orchestrator DEAD with pending={pending}; schtasks /Run -> rc={result.returncode}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
