"""Historical per-step timings used for harness ETA."""

from __future__ import annotations

from typing import Any

from pipeline_modules.harness.jsonio import read_json, write_json_atomic
from pipeline_modules.harness.paths import timing_history_path

HISTORY_SCHEMA_VERSION = "1"
MAX_SAMPLES_PER_STEP = 20


def load_timing_history(active=None) -> dict[str, list[float]]:
    payload = read_json(timing_history_path(active), default={})
    if not isinstance(payload, dict):
        return {}
    steps = payload.get("steps")
    if not isinstance(steps, dict):
        return {}
    history: dict[str, list[float]] = {}
    for name, values in steps.items():
        if not isinstance(values, list):
            continue
        cleaned: list[float] = []
        for value in values:
            try:
                number = float(value)
            except (TypeError, ValueError):
                continue
            if number > 0:
                cleaned.append(number)
        if cleaned:
            history[str(name)] = cleaned[-MAX_SAMPLES_PER_STEP:]
    return history


def load_memory_history(active=None) -> dict[str, list[float]]:
    """Per-step-name peak-memory samples (MB), recorded alongside durations."""
    payload = read_json(timing_history_path(active), default={})
    if not isinstance(payload, dict):
        return {}
    steps_mem = payload.get("steps_mem")
    if not isinstance(steps_mem, dict):
        return {}
    history: dict[str, list[float]] = {}
    for name, values in steps_mem.items():
        if not isinstance(values, list):
            continue
        cleaned = []
        for value in values:
            try:
                number = float(value)
            except (TypeError, ValueError):
                continue
            if number > 0:
                cleaned.append(number)
        if cleaned:
            history[str(name)] = cleaned[-MAX_SAMPLES_PER_STEP:]
    return history


def record_step_timings(progress: dict[str, Any], *, active=None) -> None:
    """Append completed, non-skipped step durations (and peak memory, when the
    worker recorded it) into timing_history.json."""
    steps = progress.get("steps") or []
    if not steps:
        return
    path = timing_history_path(active)
    payload = read_json(path, default={})
    if not isinstance(payload, dict):
        payload = {}
    history = load_timing_history(active)
    mem_history = load_memory_history(active)
    changed = False
    for step in steps:
        if not isinstance(step, dict):
            continue
        if step.get("skipped"):
            continue
        name = str(step.get("name") or "").strip()
        seconds = step.get("seconds")
        try:
            duration = float(seconds)
        except (TypeError, ValueError):
            duration = 0.0
        if not name:
            continue
        if duration > 1.0:
            history.setdefault(name, []).append(duration)
            history[name] = history[name][-MAX_SAMPLES_PER_STEP:]
            changed = True
        try:
            peak = float(step.get("peak_mem_mb"))
        except (TypeError, ValueError):
            peak = 0.0
        if peak > 0:
            mem_history.setdefault(name, []).append(peak)
            mem_history[name] = mem_history[name][-MAX_SAMPLES_PER_STEP:]
            changed = True
    if not changed:
        return
    payload["schema_version"] = HISTORY_SCHEMA_VERSION
    payload["steps"] = history
    payload["steps_mem"] = mem_history
    write_json_atomic(path, payload)
