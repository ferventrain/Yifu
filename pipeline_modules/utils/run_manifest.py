"""Per-run output manifests.

Every pipeline module should call :func:`write_run_manifest` at the end of a
successful run. The resulting ``_run_manifest.json`` sits next to the other
outputs and describes, in a uniform shape, what went in and what came out.

This lets agents (and humans) answer three questions without rereading code:

1. Did this directory come from a specific pipeline run?
2. What inputs and parameters produced it?
3. What output files are expected to exist, and at what sizes?
"""

from __future__ import annotations

import json
import os
import platform
import socket
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping


MANIFEST_FILENAME = "_run_manifest.json"
MANIFEST_SCHEMA_VERSION = "1"


def _sanitize_for_json(value: Any) -> Any:
    """Best-effort conversion of arbitrary values to JSON-serializable types."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): _sanitize_for_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_sanitize_for_json(v) for v in value]
    try:
        import numpy as np  # local import keeps this module light
    except ImportError:  # pragma: no cover
        np = None
    if np is not None:
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
    return repr(value)


def _describe_output(path: Path) -> dict[str, Any]:
    record: dict[str, Any] = {"path": str(path.resolve())}
    try:
        if path.is_dir():
            record["type"] = "directory"
            total_bytes = 0
            file_count = 0
            for entry in path.rglob("*"):
                if entry.is_file():
                    total_bytes += entry.stat().st_size
                    file_count += 1
            record["size_bytes"] = int(total_bytes)
            record["file_count"] = int(file_count)
        elif path.is_file():
            record["type"] = "file"
            record["size_bytes"] = int(path.stat().st_size)
        else:
            record["type"] = "missing"
    except OSError as exc:
        record["type"] = "error"
        record["error"] = repr(exc)
    return record


def build_run_manifest(
    *,
    module: str,
    entrypoint: str,
    inputs: Mapping[str, Any],
    outputs: Iterable[Path | str],
    started_at: float,
    ended_at: float | None = None,
    warnings: Iterable[str] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble a manifest dict without writing it to disk.

    Useful when the caller wants to embed the manifest in a larger structure
    or return it alongside the module's primary result.
    """
    ended_at = ended_at if ended_at is not None else time.time()
    outputs_records = [_describe_output(Path(p)) for p in outputs]
    warning_list = list(warnings) if warnings else []

    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "module": module,
        "entrypoint": entrypoint,
        "started_at": datetime.fromtimestamp(started_at, tz=timezone.utc).isoformat(),
        "ended_at": datetime.fromtimestamp(ended_at, tz=timezone.utc).isoformat(),
        "duration_seconds": float(ended_at - started_at),
        "host": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "python": sys.version.split()[0],
            "pid": os.getpid(),
        },
        "inputs": _sanitize_for_json(inputs),
        "outputs": outputs_records,
        "warnings": warning_list,
    }
    if extra:
        manifest["extra"] = _sanitize_for_json(extra)
    return manifest


def write_run_manifest(
    output_dir: Path | str,
    *,
    module: str,
    entrypoint: str,
    inputs: Mapping[str, Any],
    outputs: Iterable[Path | str],
    started_at: float,
    ended_at: float | None = None,
    warnings: Iterable[str] | None = None,
    extra: Mapping[str, Any] | None = None,
    filename: str = MANIFEST_FILENAME,
) -> Path:
    """Write a ``_run_manifest.json`` under ``output_dir`` and return its path.

    ``output_dir`` is created if missing. Existing manifest files are
    overwritten on purpose; callers that care about history should rename the
    old file before invoking this helper.
    """
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    manifest = build_run_manifest(
        module=module,
        entrypoint=entrypoint,
        inputs=inputs,
        outputs=outputs,
        started_at=started_at,
        ended_at=ended_at,
        warnings=warnings,
        extra=extra,
    )

    manifest_path = output_root / filename
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
    return manifest_path
