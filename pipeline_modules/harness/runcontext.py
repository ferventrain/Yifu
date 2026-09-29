"""RunContext: the thin facade pipeline modules adopt for progress reporting.

Existing modules already report through ``--progress_file`` /
``report_child_units``. RunContext is the SAME protocol expressed as an
object, so a module can migrate mechanically without touching algorithms:

    ctx = RunContext.from_env()          # no-op when no progress file is set
    ctx.start_step("segmentation", total=640)
    ctx.progress(done=182, total=640, message="processing blocks")
    ctx.heartbeat()
    ctx.log("...")
    ctx.register_output(path)
    ctx.check_cancelled()                # raises Cancelled
    ctx.finish_step()

When ``unit_total`` is unknown, pass nothing: the monitor shows
"progress unknown" instead of a fabricated percentage.
"""

from __future__ import annotations

import os
from pathlib import Path

from pipeline_modules.harness.jsonio import utc_now_iso
from pipeline_modules.harness.progress import ProgressWriter, touch_heartbeat

PROGRESS_FILE_ENV = "YIFU_PROGRESS_FILE"


class Cancelled(Exception):
    """Raised by :meth:`RunContext.check_cancelled` when the user cancelled."""


class RunContext:
    def __init__(self, progress_file: Path | None) -> None:
        self.progress_file = progress_file
        self._writer: ProgressWriter | None = (
            ProgressWriter(progress_file) if progress_file is not None else None
        )

    @classmethod
    def from_env(cls) -> "RunContext":
        raw = os.environ.get(PROGRESS_FILE_ENV, "").strip()
        return cls(Path(raw) if raw else None)

    @property
    def enabled(self) -> bool:
        return self._writer is not None

    # ------------------------------------------------------------------ API

    def start_step(self, step_id: str, *, total: int | None = None, title: str = "") -> None:
        if self._writer is None:
            return
        index = int(self._writer.state.get("step_index") or 0) + 1
        self._writer.begin_step(index, title or str(step_id))
        if total is not None:
            self._writer.set_units(0, int(total), phase=str(step_id))

    def progress(self, *, done: int | None, total: int | None, message: str = "") -> None:
        if self._writer is None:
            return
        if done is None or total is None:
            if message:
                self._writer.note(message)
            return
        self._writer.set_units(int(done), int(total))
        if message:
            self._writer.note(message)

    def heartbeat(self) -> None:
        if self.progress_file is not None:
            touch_heartbeat(self.progress_file)

    def log(self, message: str) -> None:
        if self._writer is not None:
            self._writer.note(str(message))

    def register_output(self, path: str | Path) -> None:
        """Record an output in progress.json; the artifact manifest is written
        by the worker's verifier after the run ends."""
        if self._writer is None:
            return
        outputs = self._writer.state.setdefault("outputs", [])
        resolved = str(Path(path).resolve())
        if resolved not in {row.get("path") for row in outputs}:
            outputs.append({"path": resolved, "registered_at": utc_now_iso()})
        self._writer._flush()

    def check_cancelled(self) -> None:
        if self.progress_file is None:
            return
        if (self.progress_file.parent / "cancel.flag").exists():
            raise Cancelled("cancel.flag present")

    def finish_step(self) -> None:
        if self._writer is not None:
            self._writer._close_current_step()
            self._writer._flush()
