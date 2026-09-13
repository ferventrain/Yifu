"""Atomic JSON helpers used by the harness queue and progress files."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    last_error: OSError | None = None
    for _ in range(8):
        try:
            os.replace(tmp, path)
            return
        except (PermissionError, FileNotFoundError, OSError) as exc:
            last_error = exc
            import time

            time.sleep(0.05)
            if not tmp.exists():
                tmp.write_text(text, encoding="utf-8")
    if last_error:
        raise last_error
