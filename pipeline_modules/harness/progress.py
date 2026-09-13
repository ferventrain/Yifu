"""Structured progress.json written by main.py and read by the harness UI."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pipeline_modules.harness.jsonio import read_json, utc_now_iso, write_json_atomic

PROGRESS_SCHEMA_VERSION = "1"

CANONICAL_STEPS = [
    "Registration channel downsample",
    "Atlas registration and label outputs",
    "Signal preprocessing and Zarr conversion",
    "Segmentation",
    "Region density analysis",
    "Vessel network reconstruction and region morphology",
]

SPOTIFLOW_STEP_5 = "Spotiflow signal count summary"


def planned_step_names(config: dict[str, Any] | None = None) -> list[str]:
    names = list(CANONICAL_STEPS)
    method = ""
    if isinstance(config, dict):
        method = str((config.get("segmentation") or {}).get("method") or "").strip().lower()
    if method == "spotiflow":
        names[4] = SPOTIFLOW_STEP_5
    return names


def empty_progress(*, step_total: int = 6) -> dict[str, Any]:
    return {
        "schema_version": PROGRESS_SCHEMA_VERSION,
        "status": "pending",
        "step_index": 0,
        "step_total": int(step_total),
        "step_name": "",
        "step_started_at": None,
        "run_started_at": None,
        "run_ended_at": None,
        "message": "",
        "error": None,
        "skipped": False,
        "steps": [],
        "results": None,
        "sample_dir": None,
        "config_path": None,
    }


def read_progress(path: Path) -> dict[str, Any]:
    payload = read_json(path, default=None)
    if not isinstance(payload, dict):
        return empty_progress()
    return payload


class ProgressWriter:
    """Append-only-ish writer used by the main orchestrator."""

    def __init__(self, path: str | Path, *, step_total: int = 6) -> None:
        self.path = Path(path)
        self.state = empty_progress(step_total=step_total)

    def _flush(self) -> None:
        write_json_atomic(self.path, self.state)

    def start_run(
        self,
        *,
        sample_dir: str | None = None,
        config_path: str | None = None,
        step_total: int | None = None,
    ) -> None:
        if step_total is not None:
            self.state["step_total"] = int(step_total)
        self.state["status"] = "running"
        self.state["run_started_at"] = utc_now_iso()
        self.state["sample_dir"] = sample_dir
        self.state["config_path"] = config_path
        self._flush()

    def _close_current_step(self) -> None:
        steps = self.state["steps"]
        if not steps:
            return
        current = steps[-1]
        if current.get("ended_at"):
            return
        ended = utc_now_iso()
        current["ended_at"] = ended
        started = current.get("started_at")
        current["seconds"] = _elapsed_seconds(started, ended)

    def begin_step(self, step_index: int, title: str) -> None:
        self._close_current_step()
        now = utc_now_iso()
        self.state["status"] = "running"
        self.state["step_index"] = int(step_index)
        self.state["step_name"] = str(title)
        self.state["step_started_at"] = now
        self.state["skipped"] = False
        self.state["message"] = ""
        self.state["steps"].append(
            {
                "index": int(step_index),
                "name": str(title),
                "started_at": now,
                "ended_at": None,
                "seconds": None,
                "skipped": False,
            }
        )
        self._flush()

    def note(self, message: str) -> None:
        self.state["message"] = str(message)
        self._flush()

    def set_units(
        self,
        unit_done: int,
        unit_total: int,
        *,
        phase: str = "",
        phase_started_at: str | None = None,
    ) -> None:
        self.state["unit_done"] = int(unit_done)
        self.state["unit_total"] = int(unit_total)
        if phase:
            self.state["unit_phase"] = str(phase)
            self.state["message"] = f"{phase} {int(unit_done)}/{int(unit_total)}"
        if phase_started_at:
            self.state["phase_started_at"] = phase_started_at
        self._flush()

    def mark_current_skipped(self) -> None:
        self.state["skipped"] = True
        steps = self.state["steps"]
        if steps:
            steps[-1]["skipped"] = True
        self._flush()

    def finish(self, results: list[dict[str, Any]] | None = None) -> None:
        self._close_current_step()
        self.state["status"] = "done"
        self.state["run_ended_at"] = utc_now_iso()
        self.state["error"] = None
        self.state["results"] = results or []
        self.state["unit_done"] = int(self.state.get("unit_total") or 0) or self.state.get("unit_done")
        self._flush()

    def fail(self, error: str) -> None:
        self._close_current_step()
        self.state["status"] = "failed"
        self.state["run_ended_at"] = utc_now_iso()
        self.state["error"] = str(error)
        self._flush()


def _parse_iso(value: str | None):
    if not value:
        return None
    try:
        from datetime import datetime

        text = value.replace("Z", "+00:00")
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def _as_aware(dt):
    """Attach local timezone to naive datetimes so ETA math stays consistent."""
    from datetime import datetime

    if dt is None or dt.tzinfo is not None:
        return dt
    return dt.replace(tzinfo=datetime.now().astimezone().tzinfo)


def _elapsed_seconds(started_at: str | None, ended_at: str | None) -> float | None:
    start = _as_aware(_parse_iso(started_at))
    end = _as_aware(_parse_iso(ended_at))
    if start is None or end is None:
        return None
    return max((end - start).total_seconds(), 0.0)


def current_step_elapsed_s(progress: dict[str, Any], *, now_iso: str | None = None) -> float | None:
    if progress.get("status") != "running":
        return None
    started = progress.get("step_started_at")
    end = now_iso or utc_now_iso()
    return _elapsed_seconds(started, end)


def estimate_eta(
    progress: dict[str, Any],
    history: dict[str, list[float]],
    *,
    planned_names: list[str] | None = None,
    now_iso: str | None = None,
) -> dict[str, Any]:
    """Estimate remaining seconds from historical median step times.

    Missing history returns null remaining values and ``source=collecting``.
    """
    names = planned_names or planned_step_names()
    step_index = int(progress.get("step_index") or 0)
    current_name = str(progress.get("step_name") or "")
    elapsed = current_step_elapsed_s(progress, now_iso=now_iso) or 0.0

    current_median = _median(history.get(current_name) or []) if current_name else None
    current_remaining = None
    if current_median is not None and progress.get("status") == "running":
        current_remaining = max(current_median - elapsed, 0.0)

    future_remaining = 0.0
    future_known = True
    remaining_names: list[str] = []
    if current_name and current_name in names:
        remaining_names = names[names.index(current_name) + 1 :]
    elif step_index > 0:
        remaining_names = names[step_index:]
    else:
        remaining_names = names

    for name in remaining_names:
        median = _median(history.get(name) or [])
        if median is None:
            future_known = False
            break
        future_remaining += median

    total_remaining = None
    source = "collecting"
    if progress.get("status") in {"done", "failed", "cancelled"}:
        total_remaining = 0.0
        current_remaining = 0.0
        source = "history"
    elif current_remaining is not None and future_known:
        total_remaining = current_remaining + future_remaining
        source = "history"
    elif current_remaining is not None and not remaining_names:
        total_remaining = current_remaining
        source = "history"
    elif current_remaining is not None:
        source = "partial"

    return {
        "current_step_remaining_s": None if current_remaining is None else round(current_remaining, 1),
        "total_remaining_s": None if total_remaining is None else round(total_remaining, 1),
        "current_step_elapsed_s": None if progress.get("status") != "running" else round(elapsed, 1),
        "source": source,
        "collecting": source == "collecting",
    }


def estimate_unit_eta(
    *,
    unit_done: int,
    unit_total: int,
    phase_started_at: str | None,
    now_iso: str | None = None,
) -> dict[str, Any]:
    """ETA from completed/total units within the current phase."""
    done = max(0, int(unit_done or 0))
    total = max(0, int(unit_total or 0))
    now = now_iso or utc_now_iso()
    elapsed = _elapsed_seconds(phase_started_at, now)
    if total <= 0 or done <= 0 or elapsed is None or elapsed <= 0:
        return {
            "current_step_remaining_s": None,
            "total_remaining_s": None,
            "current_step_elapsed_s": None if elapsed is None else round(elapsed, 1),
            "source": "collecting",
            "collecting": True,
            "unit_done": done,
            "unit_total": total,
        }
    if done >= total:
        return {
            "current_step_remaining_s": 0.0,
            "total_remaining_s": 0.0,
            "current_step_elapsed_s": round(elapsed, 1),
            "source": "units",
            "collecting": False,
            "unit_done": done,
            "unit_total": total,
        }
    remaining = elapsed * (total - done) / done
    return {
        "current_step_remaining_s": round(remaining, 1),
        "total_remaining_s": round(remaining, 1),
        "current_step_elapsed_s": round(elapsed, 1),
        "source": "units",
        "collecting": False,
        "unit_done": done,
        "unit_total": total,
    }


def _median(values: list[float]) -> float | None:
    numbers = [float(v) for v in values if v is not None and float(v) > 0]
    if not numbers:
        return None
    numbers.sort()
    n = len(numbers)
    mid = n // 2
    if n % 2:
        return numbers[mid]
    return (numbers[mid - 1] + numbers[mid]) / 2.0
