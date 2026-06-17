"""Resolve paths under the external Yifu data directory (YIFU_DATA_DIR)."""

from __future__ import annotations

import os
import re
from pathlib import Path

YIFU_DATA_DIR_ENV = "YIFU_DATA_DIR"
_ENV_PATTERN = re.compile(r"\$\{YIFU_DATA_DIR\}|%YIFU_DATA_DIR%", re.IGNORECASE)


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def get_yifu_data_dir(*, required: bool = True) -> Path | None:
    raw = os.environ.get(YIFU_DATA_DIR_ENV, "").strip()
    if not raw:
        if required:
            raise RuntimeError(
                f"Environment variable {YIFU_DATA_DIR_ENV} is not set. "
                "Set it to the folder that contains reference/ and models/ "
                "(for example S:/Yifu_data on Windows)."
            )
        return None
    return Path(raw).expanduser().resolve()


def expand_config_path(path_value: str | Path, *, project_root_override: Path | None = None) -> Path:
    """Expand ${YIFU_DATA_DIR} placeholders, then resolve relative to repo root."""
    text = str(path_value).strip()
    if not text:
        return Path(text)

    if _ENV_PATTERN.search(text):
        data_dir = get_yifu_data_dir(required=True)
        text = _ENV_PATTERN.sub(lambda _match: str(data_dir), text)

    path = Path(os.path.expandvars(text)).expanduser()
    if path.is_absolute():
        return path.resolve()

    root = project_root_override or project_root()
    return (root / path).resolve()


def reference_dir() -> Path:
    return get_yifu_data_dir() / "reference"


def cfos_checkpoint_path(filename: str = "best_model.pt") -> Path:
    return get_yifu_data_dir() / "models" / "cfos" / filename
