"""Normalize cFos pipeline Excel outputs into a report-ready JSON model."""

from __future__ import annotations

import argparse
import ast
import io
import json
import sys
import zipfile
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import tifffile

from pipeline_modules.utils.data_paths import resolve_atlas_label_path
from pipeline_modules.utils.deliverable_paths import (
    heatmap_3d_volume_tiff,
    visualization_dir,
)
from pipeline_modules.visualization.atlas_slice import (
    DEFAULT_ATLAS_LABEL,
    DEFAULT_BREGMA_INDEX,
    AtlasSliceSpec,
    bregma_mm_for_plane_index,
    build_region_metric_lookup,
    compute_symmetric_metric_limits,
    extract_atlas_slice,
    fold_change_region_metric_values,
    index_to_bregma_mm,
    paint_hemisphere_split_slice,
    paint_lr_sample_split_slice,
    resolve_slice_region_values,
    subtract_region_metric_values,
)
from pipeline_modules.visualization.coarse_region_metric_plot import (
    DEFAULT_CFG,
    DEFAULT_REGION_IDS,
    build_coarse_region_table,
    load_level_sheets,
    load_region_names,
    parse_structure_id_path,
    split_name_and_acronym,
)
from pipeline_modules.visualization.cfos_report_spatial import get_atlas_shape, resolve_spatial_source
from pipeline_modules.visualization.heatmap import resolve_density_excel_path
from pipeline_modules.visualization.cfos_report_summary import (
    build_summary_payload,
    summary_json_path,
    write_summary_json,
)
from pipeline_modules.visualization.region_group_signal_count import (
    DEFAULT_GROUPS,
    parse_acronym_text,
)

DEFAULT_ATLAS_VERSION = "allen_mouse_25um"
DEFAULT_SIGNAL_SLICE_SIGMA = 10.0
DEFAULT_SIGNAL_SLICE_ALPHA = 2.0

_SIGNAL_VOLUME_CACHE: dict[str, np.ndarray] = {}

EXCEL_TO_FRONTEND = {
    "Signal Count": "cfos_count",
    "Signal Voxels": "signal_voxels",
    "Voxel Density": "voxel_density",
    "Total Voxels": "region_volume_voxels",
    "Sum Intensity": "sum_intensity",
    "Left Signal Count": "left_cfos_count",
    "Right Signal Count": "right_cfos_count",
    "Left Voxel Density": "left_voxel_density",
    "Right Voxel Density": "right_voxel_density",
    "Left Signal Voxels": "left_signal_voxels",
    "Right Signal Voxels": "right_signal_voxels",
    "Left Sum Intensity": "left_sum_intensity",
    "Right Sum Intensity": "right_sum_intensity",
    "Left Total Voxels": "left_region_volume_voxels",
    "Right Total Voxels": "right_region_volume_voxels",
}

FRONTEND_TO_EXCEL = {
    "cfos_count": "Signal Count",
    "signal_voxels": "Signal Voxels",
    "voxel_density": "Voxel Density",
    "mean_cfos_intensity": "Sum Intensity",
    "sum_intensity": "Sum Intensity",
    "left_cfos_count": "Left Signal Count",
    "right_cfos_count": "Right Signal Count",
    "left_voxel_density": "Left Voxel Density",
    "right_voxel_density": "Right Voxel Density",
    "laterality_index": "Signal Count",
    "count_laterality_index": "Signal Count",
    "density_laterality_index": "Voxel Density",
}

UNAVAILABLE_MODULES = [
    {
        "id": "hotspot_clustering",
        "title": "Hotspot / cluster analysis",
        "status": "unavailable",
        "reason": "Requires DBSCAN/HDBSCAN clustering over atlas-space points.",
    },
    {
        "id": "multi_marker",
        "title": "Multi-marker colocalization",
        "status": "unavailable",
        "reason": "Requires per-marker region metrics and colocalization tables.",
    },
    {
        "id": "group_statistics",
        "title": "Multi-sample group statistics",
        "status": "available",
        "reason": "Load a group manifest CSV/JSON with sample_dir and group columns to run differential analysis.",
    },
    {
        "id": "web_3d_viewer",
        "title": "Full brainrender mesh viewer",
        "status": "partial",
        "reason": "Browser point-cloud view is available when points CSV or atlas volume exists; offline brainrender mesh rendering remains separate.",
    },
    {
        "id": "chinese_aliases",
        "title": "Chinese region name search",
        "status": "unavailable",
        "reason": "Requires a Chinese alias table keyed by Allen region ID.",
    },
]


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator) / float(denominator)


def _laterality_index(left: float, right: float) -> float | None:
    total = float(left) + float(right)
    if total == 0:
        return None
    return (float(right) - float(left)) / total


def load_region_metadata_table(cfg_path: str | Path = DEFAULT_CFG) -> pd.DataFrame:
    cfg_path = Path(cfg_path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Region CSV not found: {cfg_path}")

    region_df = pd.read_csv(cfg_path)
    required = {"id", "name", "acronym", "structure_id_path"}
    missing = required.difference(region_df.columns)
    if missing:
        raise ValueError(f"Region CSV missing required column(s): {sorted(missing)}")

    rows = []
    for _, row in region_df.iterrows():
        display_name, name_acronym = split_name_and_acronym(row["name"])
        acronym = parse_acronym_text(row["acronym"]) or name_acronym
        path = parse_structure_id_path(row["structure_id_path"])
        rows.append(
            {
                "region_id": int(row["id"]),
                "region_name": display_name,
                "region_acronym": acronym,
                "excel_name": str(row["name"]),
                "structure_id_path": path,
                "parent_id": int(path[-2]) if len(path) >= 2 else None,
            }
        )
    return pd.DataFrame(rows)


def build_region_tree_nodes(region_table: pd.DataFrame) -> list[dict[str, Any]]:
    by_id = {int(row.region_id): row for _, row in region_table.iterrows()}
    children_by_parent: dict[int | None, list[int]] = {}
    for _, row in region_table.iterrows():
        parent_id = row.parent_id if pd.notna(row.parent_id) else None
        children_by_parent.setdefault(parent_id, []).append(int(row.region_id))

    for child_ids in children_by_parent.values():
        child_ids.sort(key=lambda region_id: str(by_id[region_id].region_name))

    def build_node(region_id: int) -> dict[str, Any]:
        row = by_id[region_id]
        child_ids = children_by_parent.get(region_id, [])
        return {
            "region_id": int(region_id),
            "region_name": str(row.region_name),
            "region_acronym": str(row.region_acronym),
            "parent_id": int(row.parent_id) if pd.notna(row.parent_id) else None,
            "structure_id_path": [int(value) for value in row.structure_id_path],
            "children": [build_node(child_id) for child_id in child_ids],
        }

    roots = [
        region_id
        for region_id in by_id
        if pd.isna(by_id[region_id].parent_id) or int(by_id[region_id].parent_id) not in by_id
    ]
    if not roots and 997 in by_id:
        roots = [997]
    if not roots:
        roots = children_by_parent.get(None, [])
    return [build_node(region_id) for region_id in roots]


def load_normalized_level_frames(input_excel: str | Path) -> dict[str, pd.DataFrame]:
    input_excel = Path(input_excel)
    sheets = pd.read_excel(input_excel, sheet_name=None)
    level_frames: dict[str, pd.DataFrame] = {}
    for sheet_name, frame in sheets.items():
        if not str(sheet_name).startswith("Level_"):
            continue
        if "Name" not in frame.columns:
            continue
        level_frames[str(sheet_name)] = frame.copy()
    if not level_frames:
        raise ValueError(f"No Level_* sheets found in Excel workbook: {input_excel}")
    return level_frames


def _normalize_region_row(
    excel_row: pd.Series,
    *,
    sample_id: str,
    level: str,
    region_lookup: pd.DataFrame,
) -> dict[str, Any] | None:
    excel_name = str(excel_row["Name"])
    matches = region_lookup[region_lookup["excel_name"] == excel_name]
    if matches.empty:
        return None
    region = matches.iloc[0]

    payload: dict[str, Any] = {
        "sample_id": sample_id,
        "region_id": int(region["region_id"]),
        "region_name": str(region["region_name"]),
        "region_acronym": str(region["region_acronym"]),
        "excel_name": excel_name,
        "structure_id_path": [int(value) for value in region["structure_id_path"]],
        "level": level,
    }

    signal_voxels = float(pd.to_numeric(excel_row.get("Signal Voxels", 0), errors="coerce") or 0)
    sum_intensity = float(pd.to_numeric(excel_row.get("Sum Intensity", 0), errors="coerce") or 0)
    payload["cfos_count"] = float(pd.to_numeric(excel_row.get("Signal Count", 0), errors="coerce") or 0)
    payload["signal_voxels"] = signal_voxels
    payload["voxel_density"] = float(pd.to_numeric(excel_row.get("Voxel Density", 0), errors="coerce") or 0)
    payload["region_volume_voxels"] = float(pd.to_numeric(excel_row.get("Total Voxels", 0), errors="coerce") or 0)
    payload["sum_intensity"] = sum_intensity
    payload["mean_cfos_intensity"] = _safe_ratio(sum_intensity, signal_voxels)

    if "Left Signal Count" in excel_row.index and "Right Signal Count" in excel_row.index:
        left_count = float(pd.to_numeric(excel_row.get("Left Signal Count", 0), errors="coerce") or 0)
        right_count = float(pd.to_numeric(excel_row.get("Right Signal Count", 0), errors="coerce") or 0)
        left_density = float(pd.to_numeric(excel_row.get("Left Voxel Density", 0), errors="coerce") or 0)
        right_density = float(pd.to_numeric(excel_row.get("Right Voxel Density", 0), errors="coerce") or 0)
        payload["left_cfos_count"] = left_count
        payload["right_cfos_count"] = right_count
        payload["left_voxel_density"] = left_density
        payload["right_voxel_density"] = right_density
        payload["count_laterality_index"] = _laterality_index(left_count, right_count)
        payload["density_laterality_index"] = _laterality_index(left_density, right_density)
        payload["laterality_index"] = payload["count_laterality_index"]
        payload["has_hemisphere"] = True
    else:
        payload["has_hemisphere"] = False

    return payload


def normalize_region_metrics(
    input_excel: str | Path,
    *,
    sample_id: str,
    cfg_path: str | Path = DEFAULT_CFG,
) -> list[dict[str, Any]]:
    region_lookup = load_region_metadata_table(cfg_path)
    level_frames = load_normalized_level_frames(input_excel)

    metrics: list[dict[str, Any]] = []
    for level, frame in sorted(level_frames.items(), key=lambda item: item[0]):
        for _, row in frame.iterrows():
            normalized = _normalize_region_row(row, sample_id=sample_id, level=level, region_lookup=region_lookup)
            if normalized is not None:
                metrics.append(normalized)

    for level in sorted({metric["level"] for metric in metrics}):
        level_rows = [metric for metric in metrics if metric["level"] == level]
        by_count = sorted(level_rows, key=lambda item: item["cfos_count"], reverse=True)
        by_density = sorted(level_rows, key=lambda item: item["voxel_density"], reverse=True)
        count_rank = {row["region_id"]: index + 1 for index, row in enumerate(by_count)}
        density_rank = {row["region_id"]: index + 1 for index, row in enumerate(by_density)}
        for metric in metrics:
            if metric["level"] != level:
                continue
            metric["rank_by_count"] = count_rank[metric["region_id"]]
            metric["rank_by_density"] = density_rank[metric["region_id"]]

    return metrics


def _metrics_at_level(metrics: list[dict[str, Any]], level: str) -> list[dict[str, Any]]:
    return [metric for metric in metrics if metric["level"] == level]


def parse_level_number(level: str) -> int:
    return int(str(level).split("_", 1)[1])


def region_st_level(structure_id_path: list[int]) -> int:
    return max(0, len(structure_id_path) - 1)


def ancestor_region_id_at_level(structure_id_path: list[int], target_level: int) -> int:
    path = [int(value) for value in structure_id_path]
    native_level = region_st_level(path)
    if native_level < target_level:
        return int(path[-1])
    if target_level < 0 or target_level >= len(path):
        return int(path[-1])
    return int(path[target_level])


def _metric_has_finer_descendant(
    region_id: int,
    metrics_by_id: dict[int, dict[str, Any]],
    paths_by_id: dict[int, list[int]],
) -> bool:
    path = paths_by_id.get(region_id)
    if not path:
        return False
    for other_id, other_path in paths_by_id.items():
        if other_id == region_id or other_id not in metrics_by_id:
            continue
        if len(other_path) > len(path) and other_path[: len(path)] == path:
            return True
    return False


def finest_coarse_region_id(
    structure_id_path: list[int],
    coarse_ids: set[int] | frozenset[int],
) -> int | None:
    """Return the deepest coarse Allen region present on a structure path."""
    path = [int(value) for value in structure_id_path]
    matching = [region_id for region_id in path if region_id in coarse_ids]
    return int(matching[-1]) if matching else None


def aggregate_metrics_to_coarse_regions(
    metrics: list[dict[str, Any]],
    region_lookup: pd.DataFrame,
    *,
    coarse_region_ids: list[int] | None = None,
) -> list[dict[str, Any]]:
    """Roll finest-available region metrics up to DEFAULT_REGION_IDS coarse systems."""
    coarse_ids = frozenset(coarse_region_ids or DEFAULT_REGION_IDS)
    paths_by_id = {
        int(row.region_id): [int(value) for value in row.structure_id_path]
        for _, row in region_lookup.iterrows()
    }
    meta_by_id = {int(row.region_id): row for _, row in region_lookup.iterrows()}
    metrics_by_id = {int(metric["region_id"]): metric for metric in metrics}

    source_rows: list[dict[str, Any]] = []
    for metric in metrics:
        region_id = int(metric["region_id"])
        if _metric_has_finer_descendant(region_id, metrics_by_id, paths_by_id):
            continue
        source_rows.append(metric)

    buckets: dict[int, dict[str, Any]] = {}
    for metric in source_rows:
        bucket_id = finest_coarse_region_id(metric["structure_id_path"], coarse_ids)
        if bucket_id is None:
            continue
        if bucket_id not in buckets:
            meta = meta_by_id.get(bucket_id)
            buckets[bucket_id] = {
                "sample_id": metric["sample_id"],
                "region_id": bucket_id,
                "region_name": str(meta.region_name) if meta is not None else str(metric["region_name"]),
                "region_acronym": str(meta.region_acronym) if meta is not None else str(metric["region_acronym"]),
                "structure_id_path": paths_by_id.get(bucket_id, metric["structure_id_path"]),
                "level": COARSE_SYSTEM_LEVEL,
                "cfos_count": 0.0,
                "signal_voxels": 0.0,
                "region_volume_voxels": 0.0,
                "sum_intensity": 0.0,
                "has_hemisphere": False,
                "left_cfos_count": 0.0,
                "right_cfos_count": 0.0,
            }
        bucket = buckets[bucket_id]
        bucket["cfos_count"] += float(metric["cfos_count"])
        bucket["signal_voxels"] += float(metric["signal_voxels"])
        bucket["region_volume_voxels"] += float(metric["region_volume_voxels"])
        bucket["sum_intensity"] += float(metric.get("sum_intensity", 0.0))
        if metric.get("has_hemisphere"):
            bucket["has_hemisphere"] = True
            bucket["left_cfos_count"] += float(metric.get("left_cfos_count", 0.0))
            bucket["right_cfos_count"] += float(metric.get("right_cfos_count", 0.0))

    for bucket in buckets.values():
        bucket["voxel_density"] = _safe_ratio(bucket["signal_voxels"], bucket["region_volume_voxels"])
        bucket["mean_cfos_intensity"] = _safe_ratio(bucket["sum_intensity"], bucket["signal_voxels"])
        if bucket.get("has_hemisphere"):
            bucket["count_laterality_index"] = _laterality_index(
                bucket["left_cfos_count"],
                bucket["right_cfos_count"],
            )
            bucket["laterality_index"] = bucket["count_laterality_index"]

    return sorted(buckets.values(), key=lambda item: int(item["region_id"]))


def aggregate_region_metrics_to_level(
    metrics: list[dict[str, Any]],
    target_level: str,
    region_lookup: pd.DataFrame,
) -> list[dict[str, Any]]:
    """Roll up region metrics to a target atlas level.

    Finer regions are summed into their level-X ancestor. Regions coarser than
    the target level are kept at their native (finest available) partition.
    """
    target = parse_level_number(target_level)
    paths_by_id = {
        int(row.region_id): [int(value) for value in row.structure_id_path]
        for _, row in region_lookup.iterrows()
    }
    meta_by_id = {int(row.region_id): row for _, row in region_lookup.iterrows()}
    metrics_by_id = {int(metric["region_id"]): metric for metric in metrics}

    source_rows: list[dict[str, Any]] = []
    for metric in metrics:
        region_id = int(metric["region_id"])
        if _metric_has_finer_descendant(region_id, metrics_by_id, paths_by_id):
            continue
        source_rows.append(metric)

    buckets: dict[int, dict[str, Any]] = {}
    for metric in source_rows:
        path = [int(value) for value in metric["structure_id_path"]]
        bucket_id = ancestor_region_id_at_level(path, target)
        if bucket_id not in buckets:
            meta = meta_by_id.get(bucket_id)
            buckets[bucket_id] = {
                "sample_id": metric["sample_id"],
                "region_id": bucket_id,
                "region_name": str(meta.region_name) if meta is not None else str(metric["region_name"]),
                "region_acronym": str(meta.region_acronym) if meta is not None else str(metric["region_acronym"]),
                "structure_id_path": paths_by_id.get(bucket_id, path),
                "level": target_level,
                "cfos_count": 0.0,
                "signal_voxels": 0.0,
                "region_volume_voxels": 0.0,
                "sum_intensity": 0.0,
                "has_hemisphere": False,
                "left_cfos_count": 0.0,
                "right_cfos_count": 0.0,
            }
        bucket = buckets[bucket_id]
        bucket["cfos_count"] += float(metric["cfos_count"])
        bucket["signal_voxels"] += float(metric["signal_voxels"])
        bucket["region_volume_voxels"] += float(metric["region_volume_voxels"])
        bucket["sum_intensity"] += float(metric.get("sum_intensity", 0.0))
        if metric.get("has_hemisphere"):
            bucket["has_hemisphere"] = True
            bucket["left_cfos_count"] += float(metric.get("left_cfos_count", 0.0))
            bucket["right_cfos_count"] += float(metric.get("right_cfos_count", 0.0))

    for bucket in buckets.values():
        bucket["voxel_density"] = _safe_ratio(bucket["signal_voxels"], bucket["region_volume_voxels"])
        bucket["mean_cfos_intensity"] = _safe_ratio(bucket["sum_intensity"], bucket["signal_voxels"])
        if bucket.get("has_hemisphere"):
            bucket["count_laterality_index"] = _laterality_index(
                bucket["left_cfos_count"],
                bucket["right_cfos_count"],
            )
            bucket["laterality_index"] = bucket["count_laterality_index"]

    return sorted(buckets.values(), key=lambda item: int(item["region_id"]))


def choose_default_level(metrics: list[dict[str, Any]]) -> str:
    levels = sorted({metric["level"] for metric in metrics}, key=lambda name: int(name.split("_", 1)[1]))
    if not levels:
        raise ValueError("No region metrics available.")
    return levels[-1]


def choose_coarse_system_level(metrics: list[dict[str, Any]], *, default_level: str | None = None) -> str:
    """Pick a level where Allen coarse parent regions are usually present."""
    available = {metric["level"] for metric in metrics}
    for candidate in ("Level_2", default_level or "", "Level_6"):
        if candidate and candidate in available:
            return candidate
    return choose_default_level(metrics)


VOXEL_VOLUME_UM3 = 6.48


def read_whole_brain_stats_from_excel(input_excel: str | Path) -> dict[str, Any]:
    """Read whole-brain totals from the Excel root row (Level_0), not level sums."""
    input_excel = Path(input_excel)
    frame = pd.read_excel(input_excel, sheet_name="Level_0")
    if frame.empty:
        raise ValueError(f"No Level_0 sheet rows in {input_excel}")

    root_rows = frame[frame["Name"].astype(str).str.contains("root", case=False, na=False)]
    row = root_rows.iloc[0] if not root_rows.empty else frame.iloc[0]

    signal_voxels = float(pd.to_numeric(row.get("Signal Voxels", 0), errors="coerce") or 0.0)
    total_voxels = float(pd.to_numeric(row.get("Total Voxels", 0), errors="coerce") or 0.0)
    total_cfos_count = float(pd.to_numeric(row.get("Signal Count", 0), errors="coerce") or 0.0)
    left_count = float(pd.to_numeric(row.get("Left Signal Count", 0), errors="coerce") or 0.0)
    right_count = float(pd.to_numeric(row.get("Right Signal Count", 0), errors="coerce") or 0.0)

    payload: dict[str, Any] = {
        "excel_name": str(row.get("Name") or ""),
        "total_cfos_count": total_cfos_count,
        "signal_voxels": signal_voxels,
        "total_region_volume_voxels": total_voxels,
        "signal_volume_um3": signal_voxels * VOXEL_VOLUME_UM3,
        "brain_volume_um3": total_voxels * VOXEL_VOLUME_UM3,
        "voxel_volume_um3": VOXEL_VOLUME_UM3,
    }
    if "Left Signal Count" in row.index and "Right Signal Count" in row.index:
        payload["left_total_cfos_count"] = left_count
        payload["right_total_cfos_count"] = right_count
        payload["whole_brain_count_laterality_index"] = _laterality_index(left_count, right_count)
    return payload


def resolve_display_region_id(
    region_id: int,
    *,
    level: str,
    cfg_path: str | Path = DEFAULT_CFG,
) -> int:
    """Map an atlas label id to the analysis level used in the region browser."""
    table = load_region_metadata_table(cfg_path)
    matches = table[table["region_id"] == int(region_id)]
    if matches.empty:
        return int(region_id)
    path = [int(value) for value in matches.iloc[0]["structure_id_path"]]
    return ancestor_region_id_at_level(path, parse_level_number(level))


def _has_finer_present_descendant(
    region_id: int,
    present_ids: frozenset[int],
    paths_by_id: dict[int, list[int]],
) -> bool:
    """True when a finer region from the Excel workbooks is nested under region_id."""
    path = paths_by_id.get(int(region_id))
    if not path:
        return False
    for other_id in present_ids:
        if other_id == int(region_id):
            continue
        other_path = paths_by_id.get(other_id)
        if other_path and len(other_path) > len(path) and other_path[: len(path)] == path:
            return True
    return False


def _pick_finest_metric_for_region(metrics: list[dict[str, Any]], region_id: int) -> dict[str, Any] | None:
    candidates = [metric for metric in metrics if int(metric["region_id"]) == int(region_id)]
    if not candidates:
        return None
    return max(candidates, key=lambda metric: parse_level_number(metric["level"]))


def compute_leaf_region_stats(
    metrics: list[dict[str, Any]],
    *,
    cfg_path: str | Path = DEFAULT_CFG,
    activation_threshold: float = 0.0,
) -> dict[str, Any]:
    """Count activated regions among finest-available nodes across all Level_* sheets.

    For each branch of the atlas tree, keep the deepest region that appears anywhere
    in Level_0..Level_9. Parent rows are excluded when a finer descendant row exists
    in the workbook (e.g. Level_2 ISO is dropped when FRP or finer regions exist).
    """
    region_lookup = load_region_metadata_table(cfg_path)
    paths_by_id = {
        int(row.region_id): [int(value) for value in row.structure_id_path]
        for _, row in region_lookup.iterrows()
    }

    present_ids = frozenset(
        int(metric["region_id"])
        for metric in metrics
        if int(metric["region_id"]) not in {0, 997}
        and "root" not in str(metric.get("excel_name") or "").lower()
    )

    leaf_ids = sorted(
        region_id
        for region_id in present_ids
        if not _has_finer_present_descendant(region_id, present_ids, paths_by_id)
    )
    leaf_metrics = [
        metric for metric in (_pick_finest_metric_for_region(metrics, region_id) for region_id in leaf_ids)
        if metric is not None
    ]
    activated = [metric for metric in leaf_metrics if metric["cfos_count"] > activation_threshold]
    return {
        "scope": "all_levels_finest_available",
        "activated_region_count": len(activated),
        "total_region_count": len(leaf_metrics),
        "activation_threshold": activation_threshold,
    }


def compute_overview(metrics: list[dict[str, Any]], *, level: str, activation_threshold: float = 0.0) -> dict[str, Any]:
    level_metrics = _metrics_at_level(metrics, level)
    activated = [metric for metric in level_metrics if metric["cfos_count"] > activation_threshold]

    overview: dict[str, Any] = {
        "level": level,
        "total_cfos_count": float(sum(metric["cfos_count"] for metric in level_metrics)),
        "total_signal_voxels": float(sum(metric["signal_voxels"] for metric in level_metrics)),
        "total_region_volume_voxels": float(sum(metric["region_volume_voxels"] for metric in level_metrics)),
        "activated_region_count": len(activated),
        "total_region_count": len(level_metrics),
        "activation_threshold": activation_threshold,
    }

    if any(metric.get("has_hemisphere") for metric in level_metrics):
        left_total = float(sum(metric.get("left_cfos_count", 0.0) for metric in level_metrics))
        right_total = float(sum(metric.get("right_cfos_count", 0.0) for metric in level_metrics))
        overview["left_total_cfos_count"] = left_total
        overview["right_total_cfos_count"] = right_total
        overview["whole_brain_count_laterality_index"] = _laterality_index(left_total, right_total)

    top_by_count = sorted(level_metrics, key=lambda item: item["cfos_count"], reverse=True)[:15]
    top_by_density = sorted(level_metrics, key=lambda item: item["voxel_density"], reverse=True)[:15]
    overview["top_by_cfos_count"] = top_by_count
    overview["top_by_voxel_density"] = top_by_density
    return overview


COARSE_SYSTEM_LEVEL = "Level_2"


def _load_coarse_region_metrics_from_excel(
    input_excel: str | Path,
    cfg_path: str | Path = DEFAULT_CFG,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read coarse Allen system metrics from Excel level sheets by region name."""
    cfg_path = Path(cfg_path)
    region_df = pd.read_csv(cfg_path)
    available_coarse_ids = [
        region_id
        for region_id in DEFAULT_REGION_IDS
        if region_id in set(region_df["id"].astype(int))
    ]
    if not available_coarse_ids:
        return pd.DataFrame(), pd.DataFrame()

    coarse_regions = load_region_names(cfg_path, available_coarse_ids)
    level_table = load_level_sheets(input_excel)
    table_kwargs = {"warn_missing": False}
    count_table = build_coarse_region_table(level_table, coarse_regions, "Signal Count", **table_kwargs)
    voxels_table = build_coarse_region_table(level_table, coarse_regions, "Signal Voxels", **table_kwargs)
    total_table = build_coarse_region_table(level_table, coarse_regions, "Total Voxels", **table_kwargs)
    density_table = build_coarse_region_table(level_table, coarse_regions, "Voxel Density", **table_kwargs)

    metrics_table = coarse_regions.copy()
    metrics_table["cfos_count"] = count_table["value"].to_numpy(dtype=float)
    metrics_table["signal_voxels"] = voxels_table["value"].to_numpy(dtype=float)
    metrics_table["region_volume_voxels"] = total_table["value"].to_numpy(dtype=float)
    metrics_table["voxel_density"] = density_table["value"].to_numpy(dtype=float)
    return metrics_table, coarse_regions


def compute_system_metrics(
    input_excel: str | Path,
    metrics: list[dict[str, Any]],
    *,
    level: str | None = None,
    cfg_path: str | Path = DEFAULT_CFG,
    activation_threshold: float = 0.0,
) -> list[dict[str, Any]]:
    del level, metrics  # coarse systems always come from Excel level-sheet lookup
    metrics_table, coarse_regions = _load_coarse_region_metrics_from_excel(input_excel, cfg_path)
    if metrics_table.empty or coarse_regions.empty:
        return []

    try:
        whole_brain = read_whole_brain_stats_from_excel(input_excel)
        whole_brain_cfos = float(whole_brain["total_cfos_count"])
        whole_brain_voxels = float(whole_brain["total_region_volume_voxels"])
    except Exception:
        whole_brain_cfos = float(metrics_table["cfos_count"].sum())
        whole_brain_voxels = float(metrics_table["region_volume_voxels"].sum())

    metrics_by_id = {int(row["region_id"]): row.to_dict() for _, row in metrics_table.iterrows()}
    systems: list[dict[str, Any]] = []

    for _, row in coarse_regions.iterrows():
        region_id = int(row["region_id"])
        region_metric = metrics_by_id.get(region_id)
        if region_metric is None:
            system_cfos = 0.0
            system_signal_voxels = 0.0
            system_total_voxels = 0.0
            system_voxel_density = 0.0
        else:
            system_cfos = float(region_metric["cfos_count"])
            system_signal_voxels = float(region_metric["signal_voxels"])
            system_total_voxels = float(region_metric["region_volume_voxels"])
            system_voxel_density = float(region_metric["voxel_density"])
        activation_load = _safe_ratio(system_cfos, whole_brain_cfos)
        enrichment_score = _safe_ratio(
            activation_load,
            _safe_ratio(system_total_voxels, whole_brain_voxels),
        )
        systems.append(
            {
                "system_name": str(row["region_name"]),
                "system_acronym": str(row["region_acronym"]),
                "region_id": region_id,
                "member_region_ids": [region_id],
                "system_cfos_count": system_cfos,
                "system_signal_voxels": system_signal_voxels,
                "system_total_voxels": system_total_voxels,
                "system_voxel_density": system_voxel_density,
                "activation_load": activation_load,
                "enrichment_score": enrichment_score,
                "activated_region_count": 1 if system_cfos > activation_threshold else 0,
                "top_region": {
                    "region_id": region_id,
                    "region_name": str(row["region_name"]),
                    "region_acronym": str(row["region_acronym"]),
                    "cfos_count": system_cfos,
                },
                "source": "coarse_allen_region",
            }
        )

    systems.sort(key=lambda item: item["activation_load"], reverse=True)
    return systems


def compute_exploratory_findings(metrics: list[dict[str, Any]], systems: list[dict[str, Any]], *, level: str) -> list[dict[str, Any]]:
    level_metrics = _metrics_at_level(metrics, level)
    findings: list[dict[str, Any]] = []

    counts = np.array([metric["cfos_count"] for metric in level_metrics], dtype=float)
    densities = np.array([metric["voxel_density"] for metric in level_metrics], dtype=float)
    if counts.size:
        count_cutoff = float(np.percentile(counts[counts > 0], 95)) if np.any(counts > 0) else 0.0
        density_cutoff = float(np.percentile(densities[densities > 0], 95)) if np.any(densities > 0) else 0.0
        for metric in level_metrics:
            if metric["cfos_count"] >= count_cutoff and metric["cfos_count"] > 0:
                findings.append(
                    {
                        "kind": "high_activation_count",
                        "region_id": metric["region_id"],
                        "region_name": metric["region_name"],
                        "region_acronym": metric["region_acronym"],
                        "metric": "cfos_count",
                        "value": metric["cfos_count"],
                        "message": f"High cFos count in {metric['region_acronym']}",
                    }
                )
            if metric["voxel_density"] >= density_cutoff and metric["voxel_density"] > 0:
                findings.append(
                    {
                        "kind": "high_activation_density",
                        "region_id": metric["region_id"],
                        "region_name": metric["region_name"],
                        "region_acronym": metric["region_acronym"],
                        "metric": "voxel_density",
                        "value": metric["voxel_density"],
                        "message": f"High voxel density in {metric['region_acronym']}",
                    }
                )

    for metric in level_metrics:
        li = metric.get("count_laterality_index")
        if li is not None and abs(li) >= 0.35:
            findings.append(
                {
                    "kind": "strong_laterality",
                    "region_id": metric["region_id"],
                    "region_name": metric["region_name"],
                    "region_acronym": metric["region_acronym"],
                    "metric": "count_laterality_index",
                    "value": li,
                    "message": f"Strong laterality in {metric['region_acronym']} (LI={li:.2f})",
                }
            )

    for system in systems:
        if system.get("source") != "coarse_allen_region":
            continue
        if system["enrichment_score"] >= 1.5 and system["system_cfos_count"] > 0:
            findings.append(
                {
                    "kind": "high_system_enrichment",
                    "region_id": system["region_id"],
                    "region_name": system["system_name"],
                    "region_acronym": system["system_acronym"],
                    "metric": "enrichment_score",
                    "value": system["enrichment_score"],
                    "message": f"High enrichment in {system['system_acronym']}",
                }
            )

    return findings


def discover_optional_assets(sample_dir: Path, signal_ch: str) -> dict[str, Any]:
    sample_dir = Path(sample_dir)
    viz_dir = visualization_dir(sample_dir)
    points_csv = viz_dir / "points.csv"
    spotiflow_candidates = sorted(sample_dir.glob("**/spotiflow*points*.csv"))
    spotiflow_points_csv = spotiflow_candidates[0] if spotiflow_candidates else None
    atlas_volume = heatmap_3d_volume_tiff(sample_dir, signal_ch)
    return {
        "atlas_volume_tiff": str(atlas_volume) if atlas_volume.exists() else None,
        "points_csv": str(points_csv) if points_csv.exists() else None,
        "spotiflow_points_csv": str(spotiflow_points_csv) if spotiflow_points_csv else None,
    }


def build_report_bundle(
    sample_dir: str | Path,
    *,
    input_excel: str | Path | None = None,
    cfg_path: str | Path = DEFAULT_CFG,
    groups_json: str | Path = DEFAULT_GROUPS,
    signal_ch: str = "ch1",
    group_label: str | None = None,
    activation_threshold: float = 0.0,
    atlas_version: str = DEFAULT_ATLAS_VERSION,
) -> dict[str, Any]:
    sample_dir = Path(sample_dir)
    sample_id = sample_dir.name
    density_excel = resolve_density_excel_path(sample_dir, input_excel, signal_ch=signal_ch)
    region_table = load_region_metadata_table(cfg_path)
    metrics = normalize_region_metrics(density_excel, sample_id=sample_id, cfg_path=cfg_path)
    default_level = choose_default_level(metrics)
    systems = compute_system_metrics(
        density_excel,
        metrics,
        level=default_level,
        cfg_path=cfg_path,
        activation_threshold=activation_threshold,
    )
    overview = compute_overview(metrics, level=default_level, activation_threshold=activation_threshold)
    findings = compute_exploratory_findings(metrics, systems, level=default_level)
    assets = discover_optional_assets(sample_dir, signal_ch)
    spatial_source_kind, spatial_source_path = resolve_spatial_source(assets)
    atlas_label_error: str | None = None
    try:
        atlas_label_path = resolve_atlas_label_path()
        atlas_shape = get_atlas_shape(atlas_label_path)
    except FileNotFoundError as exc:
        atlas_label_path = DEFAULT_ATLAS_LABEL
        atlas_label_error = str(exc)
        atlas_shape = get_atlas_shape(DEFAULT_ATLAS_LABEL)

    bundle = {
        "schema_version": "1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "sample": {
            "sample_id": sample_id,
            "sample_dir": str(sample_dir.resolve()),
            "group": group_label,
            "density_excel": str(density_excel),
            "atlas_version": atlas_version,
            **assets,
        },
        "parameters": {
            "cfg_path": str(cfg_path),
            "groups_json": str(groups_json),
            "signal_ch": signal_ch,
            "activation_threshold": activation_threshold,
            "default_level": default_level,
            "atlas_label_tiff": str(atlas_label_path),
            "atlas_label_available": atlas_label_error is None,
            "atlas_label_error": atlas_label_error,
            "bregma_index": list(DEFAULT_BREGMA_INDEX),
            "atlas_shape_dv_ap_ml": list(atlas_shape),
            "atlas_resolution_um_dv_ap_ml": [25.0, 25.0, 25.0],
            "slice_ranges": {
                "DV": [0, atlas_shape[0] - 1],
                "AP": [0, atlas_shape[1] - 1],
                "ML": [0, atlas_shape[2] - 1],
            },
        },
        "spatial": {
            "available": spatial_source_kind != "none",
            "source_kind": spatial_source_kind,
            "source_path": str(spatial_source_path) if spatial_source_path else None,
        },
        "levels": sorted({metric["level"] for metric in metrics}, key=lambda name: int(name.split("_", 1)[1])),
        "region_tree": build_region_tree_nodes(region_table),
        "region_metrics": metrics,
        "overview": overview,
        "system_metrics": systems,
        "findings": findings,
        "unavailable_modules": UNAVAILABLE_MODULES,
        "metric_labels": {
            "cfos_count": "cFos count (Signal Count)",
            "signal_voxels": "Signal voxels",
            "voxel_density": "Voxel density",
            "mean_cfos_intensity": "Mean cFos intensity",
            "laterality_index": "Count laterality index",
            "count_laterality_index": "Count laterality index",
            "density_laterality_index": "Density laterality index",
        },
    }
    summary = build_summary_payload(bundle, sample_dir=sample_dir, level=default_level)
    bundle["summary"] = summary
    try:
        write_summary_json(summary_json_path(sample_dir, signal_ch), summary)
    except OSError:
        pass
    return bundle


def export_region_metrics_csv(
    metrics: list[dict[str, Any]],
    *,
    region_ids: list[int] | None = None,
    level: str | None = None,
    sample_id: str,
    group: str | None = None,
    atlas_version: str = DEFAULT_ATLAS_VERSION,
    source_paths: dict[str, str] | None = None,
) -> str:
    rows = metrics
    if level:
        rows = [row for row in rows if row["level"] == level]
    if region_ids:
        wanted = set(region_ids)
        rows = [row for row in rows if int(row["region_id"]) in wanted]

    export_rows = []
    for row in rows:
        export_row = dict(row)
        export_row["sample_id"] = sample_id
        export_row["group"] = group or ""
        export_row["atlas_version"] = atlas_version
        export_row["analysis_timestamp"] = datetime.now(timezone.utc).isoformat()
        if source_paths:
            export_row["source_density_excel"] = source_paths.get("density_excel", "")
        export_rows.append(export_row)

    frame = pd.DataFrame(export_rows)
    return frame.to_csv(index=False)


def _build_laterality_lookup(metrics: list[dict[str, Any]], level: str) -> dict[int, float]:
    lookup: dict[int, float] = {}
    for metric in _metrics_at_level(metrics, level):
        li = metric.get("count_laterality_index")
        if li is not None:
            lookup[int(metric["region_id"])] = float(li)
    return lookup


def _build_mean_intensity_lookup(metrics: list[dict[str, Any]], level: str) -> dict[int, float]:
    return {
        int(metric["region_id"]): float(metric["mean_cfos_intensity"])
        for metric in _metrics_at_level(metrics, level)
    }


def collect_subtree_region_ids(region_id: int, cfg_path: str | Path = DEFAULT_CFG) -> frozenset[int]:
    table = load_region_metadata_table(cfg_path)
    target = int(region_id)
    subtree: set[int] = set()
    for _, row in table.iterrows():
        path = [int(value) for value in row["structure_id_path"]]
        if target in path:
            subtree.add(int(row["region_id"]))
    if not subtree:
        subtree.add(target)
    return frozenset(subtree)


def load_smoothed_signal_volume(
    atlas_volume_path: str | Path,
    atlas_label_path: str | Path,
    *,
    sigma: float = DEFAULT_SIGNAL_SLICE_SIGMA,
    alpha: float = DEFAULT_SIGNAL_SLICE_ALPHA,
) -> np.ndarray:
    from pipeline_modules.visualization.heatmap import build_local_signal_volume, read_tiff_stack

    atlas_volume_path = Path(atlas_volume_path)
    atlas_label_path = Path(atlas_label_path)
    if not atlas_volume_path.exists():
        raise FileNotFoundError(f"Atlas signal volume not found: {atlas_volume_path}")
    if not atlas_label_path.exists():
        raise FileNotFoundError(f"Atlas label not found: {atlas_label_path}")

    cache_key = (
        f"{atlas_volume_path.resolve()}|{atlas_volume_path.stat().st_mtime_ns}|"
        f"{atlas_label_path.resolve()}|{atlas_label_path.stat().st_mtime_ns}|"
        f"{sigma}|{alpha}"
    )
    cached = _SIGNAL_VOLUME_CACHE.get(cache_key)
    if cached is not None:
        return cached

    volume = np.asarray(read_tiff_stack(atlas_volume_path), dtype=np.float32)
    labels = tifffile.memmap(str(atlas_label_path))
    atlas_mask = np.asarray(labels) > 0
    local_signal = build_local_signal_volume(
        volume,
        sigma=float(sigma),
        alpha=float(alpha),
        atlas_mask=atlas_mask,
        normalize=False,
    )
    _SIGNAL_VOLUME_CACHE[cache_key] = local_signal
    return local_signal


def render_metric_slice_png_bytes(
    *,
    input_excel: str | Path,
    cfg_path: str | Path = DEFAULT_CFG,
    plane: str = "coronal",
    coordinate_system: str = "index",
    coordinate: float = DEFAULT_BREGMA_INDEX[1],
    metric: str = "cfos_count",
    level: str | None = None,
    sample_id: str | None = None,
    atlas_label: str | Path | None = None,
    atlas_volume_tiff: str | Path | None = None,
    color_mode: str = "region",
    compare_input_excel: str | Path | None = None,
    compare_sample_id: str | None = None,
    dpi: int = 90,
    focus_region_id: int | None = None,
    focus_only: bool = False,
    bregma_index: tuple[int, int, int] = DEFAULT_BREGMA_INDEX,
    resolution_um: float = 25.0,
    interactive: bool = False,
) -> bytes:
    png_bytes, _layout = render_metric_slice_png_with_layout(
        input_excel=input_excel,
        cfg_path=cfg_path,
        plane=plane,
        coordinate_system=coordinate_system,
        coordinate=coordinate,
        metric=metric,
        level=level,
        sample_id=sample_id,
        atlas_label=atlas_label,
        atlas_volume_tiff=atlas_volume_tiff,
        color_mode=color_mode,
        compare_input_excel=compare_input_excel,
        compare_sample_id=compare_sample_id,
        dpi=dpi,
        focus_region_id=focus_region_id,
        focus_only=focus_only,
        bregma_index=bregma_index,
        resolution_um=resolution_um,
        interactive=interactive,
    )
    return png_bytes


def render_metric_slice_png_with_layout(
    *,
    input_excel: str | Path,
    cfg_path: str | Path = DEFAULT_CFG,
    plane: str = "coronal",
    coordinate_system: str = "index",
    coordinate: float = DEFAULT_BREGMA_INDEX[1],
    metric: str = "cfos_count",
    level: str | None = None,
    sample_id: str | None = None,
    atlas_label: str | Path | None = None,
    atlas_volume_tiff: str | Path | None = None,
    color_mode: str = "region",
    compare_input_excel: str | Path | None = None,
    compare_sample_id: str | None = None,
    dpi: int = 90,
    focus_region_id: int | None = None,
    focus_only: bool = False,
    bregma_index: tuple[int, int, int] = DEFAULT_BREGMA_INDEX,
    resolution_um: float = 25.0,
    interactive: bool = False,
) -> tuple[bytes, dict[str, int]]:
    focus_key = int(focus_region_id) if focus_region_id else 0
    return _cached_render_metric_slice_png_with_layout(
        str(input_excel),
        str(cfg_path),
        plane,
        coordinate_system,
        float(coordinate),
        metric,
        str(level) if level else "",
        str(sample_id) if sample_id else "",
        str(atlas_label) if atlas_label else "",
        str(atlas_volume_tiff) if atlas_volume_tiff else "",
        str(color_mode),
        str(compare_input_excel) if compare_input_excel else "",
        str(compare_sample_id) if compare_sample_id else "",
        int(dpi),
        focus_key,
        bool(focus_only),
        tuple(int(value) for value in bregma_index),
        float(resolution_um),
        bool(interactive),
    )


@lru_cache(maxsize=256)
def _cached_render_metric_slice_png_with_layout(
    input_excel: str,
    cfg_path: str,
    plane: str,
    coordinate_system: str,
    coordinate: float,
    metric: str,
    level: str,
    sample_id: str,
    atlas_label: str,
    atlas_volume_tiff: str,
    color_mode: str,
    compare_input_excel: str,
    compare_sample_id: str,
    dpi: int,
    focus_region_id: int,
    focus_only: bool,
    bregma_index: tuple[int, int, int],
    resolution_um: float,
    interactive: bool,
) -> tuple[bytes, dict[str, int]]:
    resolved_level = level or None
    resolved_sample_id = sample_id or None
    resolved_atlas = atlas_label or None
    resolved_volume = atlas_volume_tiff or None
    focus_ids = (
        collect_subtree_region_ids(focus_region_id, cfg_path) if focus_region_id else None
    )
    return _render_metric_slice_png_bytes_impl(
        input_excel=Path(input_excel),
        cfg_path=Path(cfg_path),
        plane=plane,
        coordinate_system=coordinate_system,
        coordinate=coordinate,
        metric=metric,
        level=resolved_level,
        sample_id=resolved_sample_id,
        atlas_label=Path(resolved_atlas) if resolved_atlas else None,
        atlas_volume_tiff=Path(resolved_volume) if resolved_volume else None,
        color_mode=color_mode,
        compare_input_excel=Path(compare_input_excel) if compare_input_excel else None,
        compare_sample_id=compare_sample_id or None,
        dpi=dpi,
        focus_region_ids=focus_ids,
        focus_only=focus_only and focus_ids is not None,
        bregma_index=bregma_index,
        resolution_um=resolution_um,
        interactive=interactive,
    )


def _render_metric_slice_png_bytes_impl(
    *,
    input_excel: str | Path,
    cfg_path: str | Path = DEFAULT_CFG,
    plane: str = "coronal",
    coordinate_system: str = "index",
    coordinate: float = DEFAULT_BREGMA_INDEX[1],
    metric: str = "cfos_count",
    level: str | None = None,
    sample_id: str | None = None,
    atlas_label: str | Path | None = None,
    atlas_volume_tiff: str | Path | None = None,
    color_mode: str = "region",
    compare_input_excel: str | Path | None = None,
    compare_sample_id: str | None = None,
    dpi: int = 90,
    focus_region_ids: frozenset[int] | None = None,
    focus_only: bool = False,
    bregma_index: tuple[int, int, int] = DEFAULT_BREGMA_INDEX,
    resolution_um: float = 25.0,
    interactive: bool = False,
) -> tuple[bytes, dict[str, int]]:
    from PIL import Image

    from pipeline_modules.visualization.heatmap import (
        _render_local_slice_array,
        _render_region_metric_slice_array,
        _slice_signal_volume,
    )

    include_colorbar = True

    input_excel = Path(input_excel)
    label_path = Path(atlas_label) if atlas_label else resolve_atlas_label_path()
    atlas_slice = extract_atlas_slice(
        label_path,
        AtlasSliceSpec(
            plane=plane,  # type: ignore[arg-type]
            coordinate_system=coordinate_system,  # type: ignore[arg-type]
            coordinate=float(coordinate),
            bregma_index=bregma_index,
            atlas_resolution_um=resolution_um,
        ),
    )

    atlas_h, atlas_w = atlas_slice.image.shape
    slice_layout: dict[str, int] = {
        "image_width": atlas_w,
        "image_height": atlas_h,
        "slice_left": 0,
        "slice_top": 0,
        "slice_width": atlas_w,
        "slice_height": atlas_h,
        "atlas_width": atlas_w,
        "atlas_height": atlas_h,
    }

    normalized_color_mode = str(color_mode or "region").strip().lower()
    compare_modes = {"dual", "split_lr", "diff", "fold"}
    if normalized_color_mode in compare_modes and compare_input_excel is None:
        raise ValueError(f"color_mode={normalized_color_mode} requires compare_sample_dir.")

    if normalized_color_mode == "signal":
        if atlas_volume_tiff is None:
            raise ValueError(
                "Local signal density heatmap requires sample atlas volume TIFF "
                "(visualization/*_heatmap_3d_volume.tiff)."
            )
        local_signal = load_smoothed_signal_volume(atlas_volume_tiff, label_path)
        signal_slice = _slice_signal_volume(local_signal, atlas_slice)
        if signal_slice.shape != atlas_slice.image.shape:
            raise ValueError(
                f"Signal slice shape {signal_slice.shape} does not match atlas slice shape {atlas_slice.image.shape}"
            )
        brain_mask = atlas_slice.image > 0
        positive = signal_slice[brain_mask]
        positive = positive[positive > 0]
        vmax = float(np.nanpercentile(positive, 99.5)) if positive.size else 1.0
        if vmax <= 0:
            vmax = 1.0
        rendered, slice_layout = _render_local_slice_array(
            signal_slice,
            atlas_slice.image,
            cmap_name="white_blue_red",
            vmin=0.0,
            vmax=vmax,
            dpi=dpi,
            line_width=0.16,
            brain_outline_width=0.42,
            colorbar_label="Local signal density",
            show_region_contours=True,
            include_colorbar=include_colorbar,
        )
    elif normalized_color_mode == "hemisphere":
        if plane != "coronal":
            raise ValueError("Hemisphere split coloring is currently supported for coronal slices only.")
        left_metric, right_metric = (
            ("Left Voxel Density", "Right Voxel Density")
            if metric == "voxel_density"
            else ("Left Signal Count", "Right Signal Count")
        )
        lookup_left, path_by_region_id = build_region_metric_lookup(
            input_excel,
            cfg_path=cfg_path,
            metric=left_metric,
        )
        lookup_right, _ = build_region_metric_lookup(
            input_excel,
            cfg_path=cfg_path,
            metric=right_metric,
        )
        painted = paint_hemisphere_split_slice(
            atlas_slice.image,
            lookup_left,
            lookup_right,
            path_by_region_id,
            ml_mid_index=int(bregma_index[2]),
        )
        finite = [value for value in painted[np.isfinite(painted)] if np.isfinite(value)]
        vmax = max(finite) if finite else 1.0
        rendered, slice_layout = _render_local_slice_array(
            painted,
            atlas_slice.image,
            cmap_name="white_orange_red_black",
            vmin=0.0,
            vmax=vmax,
            dpi=dpi,
            line_width=0.16,
            brain_outline_width=0.42,
            colorbar_label=f"L:{left_metric} / R:{right_metric}",
            show_region_contours=True,
            include_colorbar=include_colorbar,
        )
    elif normalized_color_mode in compare_modes:
        excel_metric = FRONTEND_TO_EXCEL.get(metric, metric)
        lookup_a, path_by_region_id = build_region_metric_lookup(
            input_excel,
            cfg_path=cfg_path,
            metric=excel_metric,
        )
        lookup_b, _ = build_region_metric_lookup(
            Path(compare_input_excel),
            cfg_path=cfg_path,
            metric=excel_metric,
        )
        sample_a_label = str(sample_id or input_excel.parent.parent.name)
        sample_b_label = str(compare_sample_id or Path(compare_input_excel).parent.parent.name)

        if normalized_color_mode == "dual":
            region_values_a = resolve_slice_region_values(atlas_slice.image, lookup_a, path_by_region_id)
            region_values_b = resolve_slice_region_values(atlas_slice.image, lookup_b, path_by_region_id)
            finite = [
                value
                for mapping in (region_values_a, region_values_b)
                for value in mapping.values()
                if np.isfinite(value)
            ]
            vmax = max(finite) if finite else 1.0
            rendered_a, _layout_a = _render_region_metric_slice_array(
                atlas_slice.image,
                region_values_a,
                cmap_name="white_orange_red_black",
                vmin=0.0,
                vmax=vmax,
                dpi=dpi,
                line_width=0.16,
                brain_outline_width=0.42,
                colorbar_label=f"{sample_a_label} · {excel_metric}",
                focus_region_ids=focus_region_ids,
                focus_only=focus_only,
                include_colorbar=include_colorbar,
            )
            rendered_b, _layout_b = _render_region_metric_slice_array(
                atlas_slice.image,
                region_values_b,
                cmap_name="white_orange_red_black",
                vmin=0.0,
                vmax=vmax,
                dpi=dpi,
                line_width=0.16,
                brain_outline_width=0.42,
                colorbar_label=f"{sample_b_label} · {excel_metric}",
                focus_region_ids=focus_region_ids,
                focus_only=focus_only,
                include_colorbar=include_colorbar,
            )
            gap = np.zeros((rendered_a.shape[0], 10, 4), dtype=np.uint8)
            rendered = np.concatenate([rendered_a, gap, rendered_b], axis=1)
        elif normalized_color_mode == "split_lr":
            if plane != "coronal":
                raise ValueError("split_lr mode is currently supported for coronal slices only.")
            region_values_a = resolve_slice_region_values(atlas_slice.image, lookup_a, path_by_region_id)
            region_values_b = resolve_slice_region_values(atlas_slice.image, lookup_b, path_by_region_id)
            painted = paint_lr_sample_split_slice(
                atlas_slice.image,
                region_values_a,
                region_values_b,
                ml_mid_index=int(bregma_index[2]),
            )
            finite = [value for value in painted[np.isfinite(painted)] if np.isfinite(value)]
            vmax = max(finite) if finite else 1.0
            rendered, slice_layout = _render_local_slice_array(
                painted,
                atlas_slice.image,
                cmap_name="white_orange_red_black",
                vmin=0.0,
                vmax=vmax,
                dpi=dpi,
                line_width=0.16,
                brain_outline_width=0.42,
                colorbar_label=f"L:{sample_a_label} / R:{sample_b_label} · {excel_metric}",
                show_region_contours=True,
                include_colorbar=include_colorbar,
            )
        elif normalized_color_mode == "diff":
            diff_lookup = subtract_region_metric_values(lookup_a, lookup_b)
            region_values = resolve_slice_region_values(atlas_slice.image, diff_lookup, path_by_region_id)
            vmin, vmax = compute_symmetric_metric_limits(list(region_values.values()))
            rendered, slice_layout = _render_region_metric_slice_array(
                atlas_slice.image,
                region_values,
                cmap_name="signal_count_diff",
                vmin=vmin,
                vmax=vmax,
                dpi=dpi,
                line_width=0.16,
                brain_outline_width=0.42,
                colorbar_label=f"{sample_a_label} - {sample_b_label} · {excel_metric}",
                focus_region_ids=focus_region_ids,
                focus_only=focus_only,
                include_colorbar=include_colorbar,
            )
        else:  # fold
            fold_lookup = fold_change_region_metric_values(lookup_a, lookup_b)
            region_values = resolve_slice_region_values(atlas_slice.image, fold_lookup, path_by_region_id)
            finite = [abs(value) for value in region_values.values() if np.isfinite(value)]
            limit = max(finite) if finite else 1.0
            rendered, slice_layout = _render_region_metric_slice_array(
                atlas_slice.image,
                region_values,
                cmap_name="signal_count_diff",
                vmin=-limit,
                vmax=limit,
                dpi=dpi,
                line_width=0.16,
                brain_outline_width=0.42,
                colorbar_label=f"log2({sample_a_label}/{sample_b_label}) · {excel_metric}",
                focus_region_ids=focus_region_ids,
                focus_only=focus_only,
                include_colorbar=include_colorbar,
            )
    elif normalized_color_mode != "region":
        raise ValueError(
            "color_mode must be one of: region, signal, hemisphere, dual, split_lr, diff, fold"
        )

    else:
        path_by_region_id = {
            int(row["region_id"]): [int(value) for value in row["structure_id_path"]]
            for _, row in load_region_metadata_table(cfg_path).iterrows()
        }

        if metric in {"laterality_index", "count_laterality_index", "density_laterality_index", "mean_cfos_intensity"}:
            resolved_sample_id = sample_id or input_excel.parent.name
            metrics = normalize_region_metrics(input_excel, sample_id=resolved_sample_id, cfg_path=cfg_path)
            chosen_level = level or choose_default_level(metrics)
            if metric == "density_laterality_index":
                value_by_region_id = {
                    int(item["region_id"]): float(item.get("density_laterality_index") or 0.0)
                    for item in _metrics_at_level(metrics, chosen_level)
                }
                colorbar_label = "Density laterality index"
                cmap_name = "signal_count_diff"
                vmin = -max((abs(value) for value in value_by_region_id.values()), default=1.0)
                vmax = -vmin
            elif metric == "mean_cfos_intensity":
                value_by_region_id = _build_mean_intensity_lookup(metrics, chosen_level)
                colorbar_label = "Mean cFos intensity"
                cmap_name = "white_orange_red_black"
                finite = [value for value in value_by_region_id.values() if np.isfinite(value) and value > 0]
                vmin = 0.0
                vmax = max(finite) if finite else 1.0
            else:
                value_by_region_id = _build_laterality_lookup(metrics, chosen_level)
                colorbar_label = "Count laterality index"
                cmap_name = "signal_count_diff"
                finite = [abs(value) for value in value_by_region_id.values() if np.isfinite(value)]
                limit = max(finite) if finite else 1.0
                vmin, vmax = -limit, limit
            region_values = resolve_slice_region_values(atlas_slice.image, value_by_region_id, path_by_region_id)
            rendered, slice_layout = _render_region_metric_slice_array(
                atlas_slice.image,
                region_values,
                cmap_name=cmap_name,
                vmin=vmin,
                vmax=vmax,
                dpi=dpi,
                line_width=0.16,
                brain_outline_width=0.42,
                colorbar_label=colorbar_label,
                focus_region_ids=focus_region_ids,
                focus_only=focus_only,
                include_colorbar=include_colorbar,
            )
        else:
            excel_metric = FRONTEND_TO_EXCEL.get(metric, metric)
            value_by_region_id, path_by_region_id = build_region_metric_lookup(
                input_excel,
                cfg_path=cfg_path,
                metric=excel_metric,
            )
            region_values = resolve_slice_region_values(atlas_slice.image, value_by_region_id, path_by_region_id)
            finite = [value for value in region_values.values() if np.isfinite(value)]
            vmax = max(finite) if finite else 1.0
            colorbar_label = "Signal Density" if excel_metric == "Voxel Density" else excel_metric
            rendered, slice_layout = _render_region_metric_slice_array(
                atlas_slice.image,
                region_values,
                cmap_name="white_orange_red_black",
                vmin=0.0,
                vmax=vmax,
                dpi=dpi,
                line_width=0.16,
                brain_outline_width=0.42,
                colorbar_label=colorbar_label,
                focus_region_ids=focus_region_ids,
                focus_only=focus_only,
                include_colorbar=include_colorbar,
            )

    image = Image.fromarray(rendered)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue(), slice_layout


def export_slice_bookmarks_zip(
    *,
    sample_id: str,
    input_excel: str | Path,
    cfg_path: str | Path,
    atlas_label: str | Path,
    atlas_volume_tiff: str | Path | None,
    bookmarks: list[dict[str, Any]],
    color_modes: list[str],
    metric: str,
    level: str | None,
    bregma_index: tuple[int, int, int] = DEFAULT_BREGMA_INDEX,
    resolution_um: float = 25.0,
    dpi: int = 150,
    focus_region_id: int | None = None,
) -> bytes:
    if not bookmarks:
        raise ValueError("At least one slice bookmark is required.")
    normalized_modes = []
    for mode in color_modes:
        normalized = str(mode).strip().lower()
        if normalized not in {"region", "signal"}:
            raise ValueError("color_modes entries must be 'region' or 'signal'")
        if normalized not in normalized_modes:
            normalized_modes.append(normalized)
    if not normalized_modes:
        raise ValueError("At least one color mode is required.")

    manifest_rows: list[dict[str, Any]] = []
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        for bookmark in bookmarks:
            plane = str(bookmark.get("plane") or "coronal")
            coordinate = float(bookmark.get("coordinate"))
            label = str(bookmark.get("label") or f"slice_{int(round(coordinate))}").strip()
            safe_label = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in label) or "slice"
            bregma_mm = bookmark.get("bregma_mm")
            if bregma_mm is None:
                bregma_mm = bregma_mm_for_plane_index(
                    plane,  # type: ignore[arg-type]
                    coordinate,
                    bregma_index=bregma_index,
                    resolution_um=resolution_um,
                )

            bookmark_modes = bookmark.get("color_modes")
            if bookmark_modes:
                item_modes = [
                    mode
                    for mode in bookmark_modes
                    if str(mode).strip().lower() in normalized_modes
                ]
            else:
                item_modes = normalized_modes
            if not item_modes:
                continue

            for color_mode in item_modes:
                png_bytes = render_metric_slice_png_bytes(
                    input_excel=input_excel,
                    cfg_path=cfg_path,
                    plane=plane,
                    coordinate_system=str(bookmark.get("coordinate_system") or "index"),
                    coordinate=coordinate,
                    metric=metric,
                    level=level,
                    sample_id=sample_id,
                    atlas_label=atlas_label,
                    atlas_volume_tiff=atlas_volume_tiff,
                    color_mode=color_mode,
                    dpi=dpi,
                    focus_region_id=focus_region_id,
                    focus_only=False,
                    bregma_index=bregma_index,
                    resolution_um=resolution_um,
                )
                filename = (
                    f"{safe_label}_{plane}_idx{int(round(coordinate))}_"
                    f"bregma{bregma_mm:+.2f}mm_{color_mode}.png"
                )
                archive.writestr(filename, png_bytes)
                manifest_rows.append(
                    {
                        "label": label,
                        "plane": plane,
                        "coordinate_system": str(bookmark.get("coordinate_system") or "index"),
                        "coordinate": coordinate,
                        "bregma_mm": float(bregma_mm),
                        "color_mode": color_mode,
                        "metric": metric,
                        "level": level,
                        "filename": filename,
                    }
                )

        archive.writestr(
            "manifest.json",
            json.dumps(
                {
                    "sample_id": sample_id,
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                    "bookmarks": manifest_rows,
                },
                indent=2,
            ),
        )
    return buffer.getvalue()


def write_report_bundle(output_path: str | Path, bundle: dict[str, Any]) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(bundle, indent=2), encoding="utf-8")
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build normalized cFos visual report JSON from pipeline Excel output.")
    parser.add_argument("--sample_dir", required=True, help="Sample directory containing density Excel output")
    parser.add_argument("--input_excel", default=None, help="Optional explicit density Excel path")
    parser.add_argument("--signal_ch", default="ch1", help="Signal channel token for deliverable path resolution")
    parser.add_argument("--group", default=None, help="Optional sample group label")
    parser.add_argument("--cfg", default=str(DEFAULT_CFG), help="Allen region CSV path")
    parser.add_argument("--groups_json", default=str(DEFAULT_GROUPS), help="Region groups JSON path")
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSON path. Defaults to sample_dir/visualization/<sample>_cfos_report.json",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    sample_dir = Path(args.sample_dir)
    output = (
        Path(args.output)
        if args.output
        else sample_dir / "visualization" / f"{sample_dir.name}_cfos_report.json"
    )
    try:
        bundle = build_report_bundle(
            sample_dir,
            input_excel=args.input_excel,
            cfg_path=args.cfg,
            groups_json=args.groups_json,
            signal_ch=args.signal_ch,
            group_label=args.group,
        )
        write_report_bundle(output, bundle)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Saved report bundle to: {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
