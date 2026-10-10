"""Per-run partial configs: missing keys inherit from base configs.

The board workflow writes one config per run (config.json, config_1.json,
config_2.json ...). A run config only needs the fields it OVERRIDES —
``resolve_partial_config`` deep-merges it over, in order:

1. the sibling ``config.json`` in the same directory (the sample's base),
2. the repo's ``config/config_cfos_template.json``.

A full config behaves identically (every key it defines wins). The sample's
own ``config.json`` is never merged over anything — it IS the base.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_TEMPLATE = Path(__file__).resolve().parents[2] / "config" / "config_cfos_template.json"


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursive merge: dict values combine; anything present in ``override``
    (lists and scalars included) replaces the base value wholesale."""
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    return payload if isinstance(payload, dict) else {}


def resolve_partial_config(path: str | Path) -> tuple[Path, dict[str, Any]]:
    """Load a config, inheriting missing keys from base configs.

    Returns ``(path, merged_config)``; on any read/parse problem the raw
    payload (or {}) is returned so the caller's own validation reports the
    error against the original file.
    """
    path = Path(path)
    try:
        cfg = _read_json(path)
    except (OSError, json.JSONDecodeError):
        return path, {}
    if path.name == "config.json":
        return path, cfg
    for base_path in (path.parent / "config.json", _TEMPLATE):
        try:
            if base_path.is_file() and base_path.resolve() != path.resolve():
                cfg = deep_merge(_read_json(base_path), cfg)
        except (OSError, json.JSONDecodeError):
            continue
    return path, cfg
