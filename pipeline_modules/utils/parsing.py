"""Small value parsers shared across pipeline modules."""

from __future__ import annotations

import re


def parse_resolution_xyz(value) -> tuple[float, float, float]:
    """Parse ``x,y,z`` (or a 3-sequence); every value must be positive."""
    if isinstance(value, (tuple, list)):
        parts = list(value)
    else:
        parts = [part.strip() for part in str(value).split(",") if part.strip()]
    if len(parts) != 3:
        raise ValueError(f"resolution_xyz must have 3 values, got: {value!r}")
    values = tuple(float(part) for part in parts)
    if any(item <= 0 for item in values):
        raise ValueError(f"resolution_xyz values must be positive: {value!r}")
    return values


def parse_block_shape(value, *, allow_empty: bool = False) -> tuple[int, int, int] | None:
    """Parse ``z,y,x``; empty input returns None only when ``allow_empty``."""
    text = str(value).strip()
    if not text and allow_empty:
        return None
    parts = [part.strip() for part in text.split(",") if part.strip()]
    if len(parts) != 3:
        raise ValueError(f"block_shape must be z,y,x with three integers, got: {value!r}")
    values = tuple(int(part) for part in parts)
    if any(item <= 0 for item in values):
        raise ValueError(f"block_shape values must be positive, got: {values}")
    return values


def sanitize_slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip())
    return slug.strip("_") or "region"
