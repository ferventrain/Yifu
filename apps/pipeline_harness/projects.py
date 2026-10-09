"""Project board aggregation for the Pipeline Monitor.

A project = one sample directory; a batch = the sample's parent directory
(experiment grouping). Everything here is READ-ONLY over existing job
records/views plus a few small files in the sample dir:

* ``<sample>/pm_project.json`` — human notes (requirements / retro / log),
  edited via POST /api/projects/notes (the only writer);
* ``config*.json`` in the sample dir — the run configs the card can start;
* the newest ``_run_manifest.json`` within depth 2 — provenance summary.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pipeline_modules.harness.jsonio import read_json, utc_now_iso, write_json_atomic

NOTES_FILENAME = "pm_project.json"
NOTES_SCHEMA_VERSION = "1"

_AGGREGATE_RANK = {
    "running": 0,
    "verifying": 0,
    "stalled": 1,
    "queued": 2,
    "failed": 3,
    "cancelled": 4,
    "succeeded": 5,
    "done": 5,
}

_MANIFEST_CANDIDATE_LIMIT = 60


def _norm_key(path_value: str) -> str:
    return str(path_value).replace("\\", "/").rstrip("/").lower()


def read_project_notes(sample_dir: Path) -> dict[str, Any] | None:
    payload = read_json(sample_dir / NOTES_FILENAME, default=None)
    return payload if isinstance(payload, dict) else None


def write_project_notes(
    sample_dir: Path,
    *,
    requirements: str | None = None,
    retro: str | None = None,
    append_note: str | None = None,
) -> dict[str, Any]:
    """Create/update ``<sample>/pm_project.json`` (the notes writer)."""
    path = sample_dir / NOTES_FILENAME
    state = read_json(path, default=None) or {}
    if not isinstance(state, dict):
        state = {}
    if requirements is not None:
        state["requirements"] = str(requirements)
    if retro is not None:
        state["retro"] = str(retro)
    if append_note:
        entry = {"at": utc_now_iso(), "text": str(append_note)}
        state.setdefault("log", []).append(entry)
    state["schema_version"] = NOTES_SCHEMA_VERSION
    state["updated_at"] = utc_now_iso()
    write_json_atomic(path, state)
    return state


def list_sample_configs(sample_dir: Path) -> list[str]:
    """config*.json names available for one-click start (config.json, config_1.json, ...)."""
    try:
        names = sorted(p.name for p in sample_dir.glob("config*.json") if p.is_file())
    except OSError:
        return []
    # config.json first, then config_1, config_2 ...
    def rank(name: str) -> tuple[int, str]:
        if name == "config.json":
            return (0, "")
        return (1, name)

    return sorted(names, key=rank)


def latest_run_manifest_summary(sample_dir: Path) -> dict[str, Any] | None:
    """Newest _run_manifest.json within depth 2, as a compact summary."""
    try:
        candidates = list(sample_dir.glob("_run_manifest.json"))
        candidates.extend(sample_dir.glob("*/_run_manifest.json"))
    except OSError:
        return None
    candidates = [p for p in candidates if p.is_file()][:_MANIFEST_CANDIDATE_LIMIT]
    if not candidates:
        return None
    newest = max(candidates, key=lambda p: p.stat().st_mtime)
    payload = read_json(newest, default=None)
    if not isinstance(payload, dict):
        return None
    outputs = payload.get("outputs") or []
    code = payload.get("code") or {}
    return {
        "path": str(newest),
        "module": payload.get("module"),
        "entrypoint": payload.get("entrypoint"),
        "ended_at": payload.get("ended_at"),
        "duration_seconds": payload.get("duration_seconds"),
        "git_commit": code.get("git_commit"),
        "git_dirty": code.get("git_dirty"),
        "outputs": [
            {"path": item.get("path"), "type": item.get("type")}
            for item in outputs
            if isinstance(item, dict)
        ][:12],
    }


def _error_summary(error: Any) -> str | None:
    if not error:
        return None
    if isinstance(error, dict):
        code = str(error.get("code") or "")
        message = str(error.get("message") or "")
        return f"[{code}] {message}" if code else message or None
    return str(error)[:300]


def _run_row(view: dict[str, Any]) -> dict[str, Any]:
    config_path = str(view.get("config_path") or "")
    code_version = view.get("code_version") or {}
    return {
        "id": view.get("id"),
        "title": view.get("title") or view.get("sample_name"),
        "kind": view.get("kind"),
        "status": view.get("status"),
        "display_status": view.get("display_status") or view.get("status"),
        "config_name": Path(config_path).name if config_path else "",
        "config_path": config_path or None,
        "created_at": view.get("created_at"),
        "started_at": view.get("started_at"),
        "ended_at": view.get("ended_at"),
        "duration_s": view.get("duration_s"),
        "error": _error_summary(view.get("error")),
        "git_commit": (code_version.get("git_commit") or "")[:12] or None,
        "qc_images": view.get("qc_images") or [],
        "results": view.get("results") or [],
        "progress": view.get("progress") or {},
        "eta": view.get("eta"),
        "log_path": view.get("log_path"),
        "worker_alive": view.get("worker_alive"),
        "heartbeat_age_s": view.get("heartbeat_age_s"),
        "current_step_duration_s": view.get("current_step_duration_s"),
        # full decorated view for the expanded rich card (local UI only)
        "full_view": view,
    }


def _project_sort_key(project: dict[str, Any]) -> tuple[int, str]:
    newest = project["runs"][0] if project["runs"] else {}
    stamp = newest.get("started_at") or newest.get("created_at") or ""
    return (0, str(stamp))


def group_into_batches(views: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Group decorated job views into batches -> projects (read-only).

    Run rows are slim; only the newest run's full decorated view is kept
    (``latest_full_view``) so the expanded rich card renders without a
    second request.
    """
    by_sample: dict[str, dict[str, Any]] = {}
    for view in views:
        sample_dir = str(view.get("sample_dir") or "").strip()
        if not sample_dir:
            continue
        key = _norm_key(sample_dir)
        project = by_sample.setdefault(
            key,
            {
                "sample_dir": sample_dir,
                "sample_name": view.get("sample_name") or Path(sample_dir).name,
                "runs": [],
            },
        )
        row = _run_row(view)
        row["full_view"] = view  # kept for this build; stripped below
        project["runs"].append(row)

    batches: dict[str, dict[str, Any]] = {}
    for project in by_sample.values():
        # newest run first (started_at falls back to created_at)
        project["runs"].sort(
            key=lambda run: str(run.get("started_at") or run.get("created_at") or ""),
            reverse=True,
        )
        # keep the newest run's full view for the expanded card; drop the rest
        latest_view = None
        for index, run in enumerate(project["runs"]):
            full = run.pop("full_view", None)
            if index == 0:
                latest_view = full
        project["latest_full_view"] = latest_view

        active = [r for r in project["runs"] if str(r["status"]) in {"queued", "running", "verifying"}]
        project["active_run_id"] = active[0]["id"] if active else None
        statuses = [str(r["display_status"] or r["status"]) for r in project["runs"]]
        project["status"] = (
            min(statuses, key=lambda s: _AGGREGATE_RANK.get(s, 9)) if statuses else "idle"
        )
        project["failed_count"] = sum(1 for s in statuses if s in {"failed", "stalled"})
        project["error_history"] = [
            {"run_id": r["id"], "error": r["error"], "ended_at": r["ended_at"]}
            for r in project["runs"]
            if r["error"]
        ][:20]

        sample_path = Path(project["sample_dir"])
        project["notes"] = read_project_notes(sample_path)
        project["configs"] = list_sample_configs(sample_path)
        project["last_manifest"] = latest_run_manifest_summary(sample_path)

        batch_dir = sample_path.parent
        batch_key = _norm_key(str(batch_dir))
        batch = batches.setdefault(
            batch_key,
            {
                "batch": batch_dir.name or str(batch_dir),
                "batch_dir": str(batch_dir),
                "projects": [],
            },
        )
        batch["projects"].append(project)

    batch_list = sorted(batches.values(), key=lambda b: b["batch"].lower())
    for batch in batch_list:
        batch["projects"].sort(key=_project_sort_key, reverse=True)
    return batch_list


def projects_payload(store: Any, views: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "batches": group_into_batches(views),
        "worker": None,  # filled by caller
        "analysis_root": str(store.root),
        "active_dir": str(store.active),
    }
