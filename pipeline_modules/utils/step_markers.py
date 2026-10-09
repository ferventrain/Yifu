"""Hash-based completion markers for pipeline steps.

``exists()``-based skipping cannot tell a finished output from an interrupted
one, and never reruns when only the config changed. Each heavy ``ensure_*``
step in ``main.py`` therefore leaves a ``<output>.done.json`` next to its
primary output recording:

* the step's parameter payload (hashed, plus the raw dict for humans),
* a cheap signature (size/mtime, or newest-mtime/file-count for directories)
  of every input,
* the code pin (git commit) at the time the step finished.

A step counts as complete only when every output still exists, the marker is
readable, and the payload and input signatures all match. Git commit changes
do NOT invalidate a marker (pulling new code should not rerun a 10-hour
segmentation); the commit is recorded for provenance only.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

from pipeline_modules.utils.run_manifest import collect_code_version

MARKER_SCHEMA_VERSION = "1"
#: Cap for directory signature walks so pathological stores cannot stall the
#: skip check. Production zarrs hold thousands of chunks, well under this.
MAX_DIR_ENTRIES = 200_000


def marker_path_for(output: Path | str) -> Path:
    return Path(str(output) + ".done.json")


def _canonical_hash(payload: Any) -> str:
    text = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _file_signature(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {"path": str(path), "kind": "file", "size": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns)}


def _dir_signature(path: Path) -> dict[str, Any]:
    newest_ns = int(path.stat().st_mtime_ns)
    count = 0
    for child in path.rglob("*"):
        count += 1
        if count > MAX_DIR_ENTRIES:
            break
        try:
            newest_ns = max(newest_ns, int(child.stat().st_mtime_ns))
        except OSError:
            continue
    return {"path": str(path), "kind": "dir", "file_count": count, "newest_mtime_ns": newest_ns}


def _input_signature(path: Path | str) -> dict[str, Any] | None:
    resolved = Path(path)
    if not resolved.exists():
        return None
    if resolved.is_file():
        return _file_signature(resolved)
    return _dir_signature(resolved)


def step_is_complete(
    outputs: Sequence[Path | str],
    inputs: Iterable[Path | str],
    payload: dict[str, Any],
) -> tuple[bool, str]:
    """Check outputs + marker + input/payload signatures. Never raises.

    Returns ``(complete, human_reason)``; the reason doubles as the rerun
    justification when incomplete.
    """
    output_paths = [Path(p) for p in outputs]
    for output in output_paths:
        if not output.exists():
            return False, f"output missing: {output}"
    marker = marker_path_for(output_paths[0])
    if not marker.exists():
        return False, "no completion marker yet"
    try:
        record = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False, "completion marker unreadable"
    if str(record.get("schema_version")) != MARKER_SCHEMA_VERSION:
        return False, "completion marker schema mismatch"
    if record.get("payload_hash") != _canonical_hash(payload):
        return False, "step parameters changed since the last run"
    recorded_outputs = [str(p) for p in record.get("outputs", [])]
    if recorded_outputs != [str(p) for p in output_paths]:
        return False, "output set changed"
    recorded_inputs = {str(item.get("path")): item for item in record.get("inputs", []) if isinstance(item, dict)}
    for raw in inputs:
        signature = _input_signature(raw)
        if signature is None:
            return False, f"input missing: {raw}"
        previous = recorded_inputs.get(str(Path(raw)))
        if previous is None:
            return False, f"input not tracked by marker: {raw}"
        comparable = {key: value for key, value in signature.items() if key != "path"}
        if any(previous.get(key) != value for key, value in comparable.items()):
            return False, f"input changed: {raw}"
    return True, "completion marker valid"


#: Reasons that mean "no usable marker state" rather than "state invalidated".
LEGACY_REASONS = (
    "no completion marker yet",
    "completion marker unreadable",
    "completion marker schema mismatch",
)


def step_or_legacy_complete(
    outputs: Sequence[Path | str],
    inputs: Iterable[Path | str],
    payload: dict[str, Any],
    outputs_present,
) -> tuple[bool, str]:
    """``step_is_complete`` plus a legacy lane for pre-marker outputs.

    Outputs produced before markers existed would otherwise be redone from
    scratch on their next pipeline run. When the only problem is a missing or
    unreadable marker and the caller-supplied ``outputs_present()`` says the
    outputs are all there (the old ``exists()`` contract), record a marker for
    them now and report complete.
    """
    complete, reason = step_is_complete(outputs, inputs, payload)
    if not complete and reason in LEGACY_REASONS and outputs_present():
        mark_step_complete(outputs, inputs, payload)
        return True, "legacy outputs present; completion marker recorded"
    return complete, reason


def mark_step_complete(
    outputs: Sequence[Path | str],
    inputs: Iterable[Path | str],
    payload: dict[str, Any],
) -> Path:
    """Write the completion marker for the step's primary output.

    Call only after the step's producing commands succeeded. Inputs that no
    longer exist are recorded with a null signature (the next check treats
    that as incomplete and reruns). The write is atomic so an interrupted
    write can never leave a half-readable marker.
    """
    output_paths = [Path(p) for p in outputs]
    marker = marker_path_for(output_paths[0])
    marker.parent.mkdir(parents=True, exist_ok=True)
    signatures = [_input_signature(p) for p in inputs]
    record = {
        "schema_version": MARKER_SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "outputs": [str(p) for p in output_paths],
        "inputs": [sig if sig is not None else {"path": str(Path(p)), "kind": "missing"} for sig, p in zip(signatures, inputs)],
        "payload": payload,
        "payload_hash": _canonical_hash(payload),
        "code_version": collect_code_version(),
    }
    tmp_path = marker.with_suffix(".tmp")
    tmp_path.write_text(json.dumps(record, indent=1, ensure_ascii=False), encoding="utf-8")
    tmp_path.replace(marker)
    return marker
