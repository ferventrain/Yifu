"""Pipeline harness: Active queue, progress files, and serial main.py runner."""

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
from pipeline_modules.harness.runner import QueueRunner

__all__ = [
    "ACTIVE_DIR_ENV",
    "ANALYSIS_ROOT_ENV",
    "ActiveStore",
    "ProgressWriter",
    "QueueRunner",
    "active_dir",
    "analysis_root",
    "collect_existing_results",
    "estimate_eta",
    "layout_from_config",
    "load_capability_manifest",
    "planned_step_names",
    "read_progress",
]


def load_capability_manifest() -> dict[str, Any]:
    path = Path(__file__).with_name("capability_manifest.json")
    return json.loads(path.read_text(encoding="utf-8"))
