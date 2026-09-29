"""Pipeline harness: single-command run, serial worker, progress, monitor API.

Standard entry::

    python -m pipeline_modules.harness run --sample-dir ... --config ...

Core objects: RunSpec (what to run), HarnessWorker/Runner (runs main.py),
RunContext (module-facing progress facade), ProgressWriter (progress.json),
verify/artifact manifest, ActiveStore (file-backed queue) and the read-only
Pipeline Monitor app in ``apps.pipeline_harness``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pipeline_modules.harness.paths import (
    ACTIVE_DIR_ENV,
    ANALYSIS_ROOT_ENV,
    active_dir,
    analysis_root,
)
from pipeline_modules.harness.progress import ProgressWriter, estimate_eta, planned_step_names, read_progress
from pipeline_modules.harness.queue import ActiveStore
from pipeline_modules.harness.results import collect_existing_results, layout_from_config
from pipeline_modules.harness.runcontext import Cancelled, RunContext
from pipeline_modules.harness.runspec import RunSpec
from pipeline_modules.harness.worker import HarnessWorker, ensure_worker_running, worker_snapshot

__all__ = [
    "ACTIVE_DIR_ENV",
    "ANALYSIS_ROOT_ENV",
    "ActiveStore",
    "Cancelled",
    "HarnessWorker",
    "ProgressWriter",
    "RunContext",
    "RunSpec",
    "active_dir",
    "analysis_root",
    "collect_existing_results",
    "ensure_worker_running",
    "estimate_eta",
    "layout_from_config",
    "load_capability_manifest",
    "planned_step_names",
    "read_progress",
    "worker_snapshot",
]


def load_capability_manifest() -> dict[str, Any]:
    path = Path(__file__).with_name("capability_manifest.json")
    return json.loads(path.read_text(encoding="utf-8"))
