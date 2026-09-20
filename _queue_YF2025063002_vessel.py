"""Serial orchestrator for vessel bilateral-statistics runs.

Launches _run_YF2025063002_vessel.py for each sample one after another,
waiting while any PM job is running and while the source NAS is unreachable.
Sample specs are 'YF<id>_<SAMPLE>' (e.g. YF2025063002_MPTP_1, YF2026051202_A125)
or bare '<SAMPLE>' for the default dataset YF2025063002.

Usage:
  python _queue_YF2025063002_vessel.py [SAMPLE ...]
  (no args = the six YF2025063002 samples)

Queue log: <batch_dir>/vessel_queue.log of the first sample's dataset.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

REPO = Path(r"S:\Yifu")
ANALYSIS_ROOT = Path(r"S:\Arivis_Analysis")
DRIVER = REPO / "_run_YF2025063002_vessel.py"
DEFAULT_SAMPLES = ["MPTP_1", "MPTP_2", "SEBL_1", "SEBL_2", "PBS_1", "PBS_2"]

DATASETS = {
    "YF2025063002": {
        "batch_dir": Path(r"S:\Arivis_Analysis\_active\YF2025063002"),
        "nas_root": Path(r"//192.168.110.4/Yifu/YF2025063002"),
    },
    "YF2026051202": {
        "batch_dir": Path(r"S:\Arivis_Analysis\_active\YF2026051202"),
        "nas_root": Path(r"//192.168.110.17/MegaSpim/YF2026051202"),
    },
}
DEFAULT_DATASET = "YF2025063002"

os.environ["YIFU_ANALYSIS_ROOT"] = str(ANALYSIS_ROOT)
os.environ["PYTHONUNBUFFERED"] = "1"
sys.path.insert(0, str(REPO))


def dataset_of(sample: str) -> str:
    match = re.match(r"(YF\d+)_", sample)
    return match.group(1) if match else DEFAULT_DATASET


SAMPLES = [arg for arg in sys.argv[1:] if not arg.startswith("--")] or DEFAULT_SAMPLES
BATCH_DIR = DATASETS[dataset_of(SAMPLES[0])]["batch_dir"]
QUEUE_LOG = BATCH_DIR / "vessel_queue.log"

LOG_FILE = open(QUEUE_LOG, "a", encoding="utf-8")
sys.stdout = LOG_FILE  # detached-safe: no console needed
sys.stderr = LOG_FILE


def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(msg: str) -> None:
    print(f"[{now()}] [queue] {msg}", flush=True)
    LOG_FILE.flush()


def wait_until_idle() -> None:
    from pipeline_modules.harness.queue import ActiveStore

    store = ActiveStore()
    while True:
        running = [
            job for job in store.list_job_views()  # job_view syncs external status
            if str(job.get("status")) == "running"
        ]
        if not running:
            return
        names = ", ".join(str(job.get("title") or job.get("id")) for job in running)
        log(f"harness busy (running: {names}); waiting 60 s")
        time.sleep(60)


def wait_for_nas(nas_root: Path) -> None:
    """Block until the NAS share answers again (a drop killed a whole pass once)."""
    while True:
        try:
            if any(nas_root.glob("*_Destripe_DONE")):
                return
        except OSError:
            pass
        log(f"NAS unreachable ({nas_root}); waiting 60 s")
        time.sleep(60)


def sample_finished(sample: str) -> bool:
    """The status file, not the exit code, is the verdict on a run."""
    dir_name = sample if sample.startswith("YF") else f"{dataset_of(sample)}_{sample}"
    status = DATASETS[dataset_of(sample)]["batch_dir"] / dir_name / "vessel_pipeline.status.txt"
    try:
        first = status.read_text(encoding="utf-8", errors="replace").splitlines()[0].strip()
    except (OSError, IndexError):
        return False
    return first.startswith("ALL DONE")


def run_pass(samples: list[str]) -> dict[str, int]:
    results: dict[str, int] = {}
    for index, sample in enumerate(samples, start=1):
        wait_until_idle()
        wait_for_nas(DATASETS[dataset_of(sample)]["nas_root"])
        log(f"({index}/{len(samples)}) launching {sample}")
        proc = subprocess.run(
            [sys.executable, str(DRIVER), sample],
            cwd=str(REPO),
        )
        if sample_finished(sample):
            results[sample] = 0
            code_note = f"(exit {proc.returncode}, status file says ALL DONE -> ok)"
        else:
            results[sample] = proc.returncode or 1
            code_note = f"(exit {proc.returncode}, status file NOT done)"
        log(f"({index}/{len(samples)}) {sample} driver finished {code_note}")
        time.sleep(15)
    return results


def main() -> int:
    log(f"=== vessel queue start, samples={SAMPLES} ===")
    results: dict[str, int] = {}
    pending = list(SAMPLES)
    for round_number in range(1, 4):  # transient killer events: retry failed samples up to 3 passes
        pass_results = run_pass(pending)
        results.update(pass_results)
        pending = [name for name, code in pass_results.items() if code != 0]
        if not pending:
            break
        log(f"pass {round_number} finished with failures: {', '.join(pending)}; retrying in 5 min")
        time.sleep(300)

    failed = [name for name, code in results.items() if code != 0]
    log(f"=== queue finished: {len(SAMPLES) - len(failed)}/{len(SAMPLES)} ok"
        + (f", failed: {', '.join(failed)}" if failed else "") + " ===")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
