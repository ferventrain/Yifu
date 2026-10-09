"""``harness wrap``: run ANY command as a tracked job with progress.

The default way to give a program Pipeline Monitor progress — zero changes to
the program::

    python -m pipeline_modules.harness wrap --sample-dir "S:/path/sample" \
        --title "rim mask chain" -- python my_script.py --args

What wrap does:

* attaches an external job (log/progress/status files owned by wrap, so the
  protocol is always exactly right),
* tees child stdout into ``runs/<id>/stdout.log`` and converts explicit
  ``PROGRESS <done>/<total> [phase]`` lines (and ``progress.report()`` calls —
  wrap exports ``YIFU_PROGRESS_FILE`` to the child) into unit progress,
* keeps a heartbeat while the child lives (the monitor stamps it via the
  tracked pid),
* maps the exit code to ALL DONE / FAILED and honours the monitor's cancel
  flag (kills the whole child process tree).

A program that prints nothing still shows liveness, elapsed time and a log
tail in the monitor.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

from pipeline_modules.harness.jsonio import utc_now_iso, write_json_atomic
from pipeline_modules.harness.paths import cancel_flag_path
from pipeline_modules.harness.queue import ActiveStore

PROGRESS_LINE_RE = re.compile(r"^\s*PROGRESS\s+(\d+)\s*/\s*(\d+)(?:\s+(.+?))?\s*$", re.IGNORECASE)
CANCEL_POLL_S = 2.0


class _WrapFiles:
    """Scratch log/status/progress files owned by wrap (under <active>/logs)."""

    def __init__(self, active: Path) -> None:
        stamp = time.strftime("%Y%m%d-%H%M%S")
        self.dir = active / "logs" / f"wrap-{stamp}-{os.getpid()}"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.log = self.dir / "stdout.log"
        self.status = self.dir / "status.txt"
        self.progress = self.dir / "module_progress.json"

    def write_status(self, first_line: str) -> None:
        self.status.write_text(f"{first_line}\n{utc_now_iso()}\n", encoding="utf-8")

    def write_progress(self, done: int | None, total: int | None, phase: str = "") -> None:
        payload: dict[str, Any] = {"updated_at": utc_now_iso()}
        if done is not None:
            payload["unit_done"] = int(done)
        if total is not None:
            payload["unit_total"] = int(total)
        if phase:
            payload["phase"] = str(phase)
            payload.setdefault("phase_started_at", utc_now_iso())
        try:
            # merge: keep a previously stamped phase_started_at so the unit
            # ETA does not restart on every line
            from pipeline_modules.harness.jsonio import read_json

            existing = read_json(self.progress, default={}) or {}
            merged = {**existing, **payload}
            if "phase_started_at" in existing and existing.get("phase") == payload.get("phase"):
                merged["phase_started_at"] = existing["phase_started_at"]
            write_json_atomic(self.progress, merged)
        except OSError:
            pass


def _tail_stream(stream, log_handle, files: _WrapFiles) -> None:
    """Tee child output into the log file and harvest PROGRESS lines."""
    for raw in iter(stream.readline, ""):
        log_handle.write(raw)
        log_handle.flush()
        match = PROGRESS_LINE_RE.match(raw)
        if match:
            files.write_progress(int(match.group(1)), int(match.group(2)), (match.group(3) or "").strip())
    stream.close()


def run_wrap(args: argparse.Namespace) -> int:
    command = list(args.command or [])
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        print("wrap: no command given (put it after '--')", file=sys.stderr)
        return 2

    store = ActiveStore()
    store.ensure_layout()
    sample = store.validate_sample_dir(args.sample_dir)

    files = _WrapFiles(store.active)
    files.write_status(f"RUNNING {args.title or 'wrapped command'}")
    files.write_progress(0, 0, args.title or "starting")

    env = os.environ.copy()
    # Children that use progress.report()/report_child_units write unit
    # progress directly; plain stdout PROGRESS lines are harvested below.
    env["YIFU_PROGRESS_FILE"] = str(files.progress)

    popen_kwargs: dict[str, Any] = {
        "args": command,
        "cwd": str(sample),
        "stdout": subprocess.PIPE,
        "stderr": subprocess.STDOUT,
        "text": True,
        "encoding": "utf-8",
        "errors": "replace",
        "env": env,
        "stdin": subprocess.DEVNULL,
    }
    if sys.platform == "win32":
        popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP

    log_handle = open(files.log, "a", encoding="utf-8")
    try:
        proc = subprocess.Popen(**popen_kwargs)
    except OSError as exc:
        log_handle.close()
        files.write_status(f"FAILED could not start: {exc}")
        print(f"wrap: could not start command: {exc}", file=sys.stderr)
        return 2

    record = store.attach_external(
        sample,
        title=args.title or " ".join(command[:3]),
        pid=proc.pid,
        log_path=files.log,
        status_path=files.status,
        module_progress_path=files.progress,
        command=" ".join(command),
        step_total=int(args.step_total or 1),
    )
    job_id = str(record["id"])
    print(json.dumps({"wrap": {"job_id": job_id, "log": str(files.log)}}, ensure_ascii=False), flush=True)

    reader = threading.Thread(target=_tail_stream, args=(proc.stdout, log_handle, files), daemon=True)
    reader.start()

    def _kill_tree() -> None:
        if sys.platform == "win32" and proc.pid:
            subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], capture_output=True, check=False)
        else:
            proc.terminate()

    returncode: int | None = None
    cancelled = False
    try:
        while proc.poll() is None:
            if cancel_flag_path(job_id, store.active).exists():
                cancelled = True
                _kill_tree()
                break
            time.sleep(CANCEL_POLL_S)
        returncode = proc.wait()
    except KeyboardInterrupt:
        cancelled = True
        _kill_tree()
        returncode = proc.wait()
    finally:
        reader.join(timeout=5)
        log_handle.close()

    def _sync_terminal_record() -> None:
        """Push the terminal status file into the job record right away
        (otherwise it stays 'running' until the monitor's next sync)."""
        try:
            store.sync_external_job(store.load_job(job_id))
        except Exception:
            pass

    if cancelled:
        files.write_status("CANCELLED by user")
        cancel_flag_path(job_id, store.active).unlink(missing_ok=True)
        _sync_terminal_record()
        print(json.dumps({"wrap": {"job_id": job_id, "status": "cancelled"}}), flush=True)
        return 130
    if returncode == 0:
        files.write_status("ALL DONE")
        _sync_terminal_record()
        print(json.dumps({"wrap": {"job_id": job_id, "status": "done"}}), flush=True)
        return 0
    files.write_status(f"FAILED exit code {returncode}")
    _sync_terminal_record()
    print(json.dumps({"wrap": {"job_id": job_id, "status": "failed", "exit_code": int(returncode or 1)}}), flush=True)
    return int(returncode or 1)


def build_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--sample-dir", required=True, help="Sample directory the command belongs to")
    parser.add_argument("--title", default="", help="Job title shown in the monitor")
    parser.add_argument("--step-total", type=int, default=0, help="Optional step count for the progress bar denominator")
    parser.add_argument("command", nargs=argparse.REMAINDER, help="Command after '--'")
    return parser
