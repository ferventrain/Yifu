"""Collect existing SampleLayout artefacts after a pipeline run."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pipeline_modules.utils.sample_layout import SampleLayout

RESULT_KEYS = [
    "brain_distribution_stats_xlsx",
    "density_results_xlsx",
    "mask_zarr",
    "signal_zarr",
    "atlas_label_zarr",
    "atlas_label_tiff_dir",
    "reg_downsample_nii",
    "results_dir",
    "visualization_dir",
    "heatmap_2d_dir",
    "heatmap_3d_png",
    "tubule_reconstruction_dir",
    "tubule_summary_json",
    "tubule_region_summary_csv",
]


def normalize_channel_label(value: Any) -> str:
    text = str(value or "").strip().lower().replace(" ", "")
    if not text:
        return "ch0"
    if text.startswith("ch"):
        return text
    return f"ch{text}"


def load_config(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path)
    return json.loads(path.read_text(encoding="utf-8-sig"))


def layout_from_config(sample_dir: str | Path, config: dict[str, Any]) -> SampleLayout:
    channels = config.get("input", {}).get("channels", {})
    signal_ch = normalize_channel_label(channels.get("signal", "0"))
    reg_ch = normalize_channel_label(channels.get("registration", "1"))
    return SampleLayout(sample_dir=Path(sample_dir), signal_ch=signal_ch, reg_ch=reg_ch, require_exists=False)


def _describe_path(name: str, path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    kind = "directory" if path.is_dir() else "file"
    record: dict[str, Any] = {
        "name": name,
        "path": str(path),
        "kind": kind,
    }
    try:
        if path.is_file():
            record["size_bytes"] = int(path.stat().st_size)
        elif path.is_dir():
            record["size_bytes"] = None
    except OSError:
        record["size_bytes"] = None
    return record


def collect_existing_results(sample_dir: str | Path, config: dict[str, Any] | str | Path) -> list[dict[str, Any]]:
    if not isinstance(config, dict):
        config = load_config(config)
    layout = layout_from_config(sample_dir, config)
    mapping = layout.as_dict()
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for key in RESULT_KEYS:
        raw = mapping.get(key)
        if not raw or raw in seen:
            continue
        record = _describe_path(key, Path(raw))
        if record is None:
            continue
        seen.add(raw)
        rows.append(record)
    return rows
