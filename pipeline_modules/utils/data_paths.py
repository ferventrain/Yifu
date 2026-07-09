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


def resolve_atlas_label_path() -> Path:
    """Find Allen atlas_label.tiff from config, YIFU_DATA_DIR, or repo data/."""
    candidates: list[Path] = []
    data_dir = get_yifu_data_dir(required=False)

    config_path = project_root() / "config" / "config.json"
    if config_path.exists():
        import json

        try:
            payload = json.loads(config_path.read_text(encoding="utf-8"))
            annotation = payload.get("registration", {}).get("annotation_path")
            if annotation:
                if _ENV_PATTERN.search(str(annotation)):
                    if data_dir is not None:
                        candidates.append(expand_config_path(annotation))
                else:
                    try:
                        candidates.append(expand_config_path(annotation))
                    except RuntimeError:
                        candidates.append(project_root() / str(annotation))
        except Exception:
            pass

    candidates.append(project_root() / "data" / "reference" / "atlas_label.tiff")
    if data_dir is not None:
        candidates.append(data_dir / "reference" / "atlas_label.tiff")

    seen: set[Path] = set()
    for path in candidates:
        resolved = path.expanduser().resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.exists():
            return resolved

    tried = "\n  ".join(str(path) for path in seen)
    raise FileNotFoundError(
        "Allen atlas label TIFF not found. Tried:\n  "
        f"{tried}\n"
        f"Set {YIFU_DATA_DIR_ENV} to your data root (with reference/atlas_label.tiff) "
        "or update config/registration annotation_path."
    )
