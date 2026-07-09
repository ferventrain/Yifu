"""Summary dashboard payload for the cFos visual report."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pipeline_modules.utils.deliverable_paths import visualization_dir

VOXEL_VOLUME_UM3 = 6.48

SUMMARY_SCHEMA_VERSION = "1"
SUMMARY_TOP_REGIONS = 12
SUMMARY_TOP_FINDINGS = 8
SUMMARY_SYSTEM_SOURCES = frozenset({"coarse_allen_region"})


def summary_json_path(sample_dir: str | Path, signal_ch: str = "ch1") -> Path:
    sample_dir = Path(sample_dir)
    return visualization_dir(sample_dir) / f"{sample_dir.name}_{signal_ch}_summary.json"


def _compact_region_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "region_id": int(row["region_id"]),
        "region_name": str(row.get("region_name") or ""),
        "region_acronym": str(row.get("region_acronym") or ""),
        "cfos_count": float(row.get("cfos_count") or 0.0),
        "voxel_density": float(row.get("voxel_density") or 0.0),
        "signal_voxels": float(row.get("signal_voxels") or 0.0),
    }


def _compact_system_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "region_id": int(row["region_id"]) if row.get("region_id") is not None else None,
        "system_name": str(row.get("system_name") or ""),
        "system_acronym": str(row.get("system_acronym") or ""),
        "system_cfos_count": float(row.get("system_cfos_count") or 0.0),
        "activation_load": float(row.get("activation_load") or 0.0),
        "enrichment_score": float(row.get("enrichment_score") or 0.0),
        "system_voxel_density": float(row.get("system_voxel_density") or 0.0),
        "activated_region_count": int(row.get("activated_region_count") or 0),
    }


def _compact_finding(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "kind": str(row.get("kind") or ""),
        "region_id": int(row["region_id"]) if row.get("region_id") is not None else None,
        "region_name": str(row.get("region_name") or ""),
        "region_acronym": str(row.get("region_acronym") or ""),
        "metric": str(row.get("metric") or ""),
        "value": float(row["value"]) if row.get("value") is not None else None,
        "message": str(row.get("message") or ""),
    }


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator) / float(denominator)


def _resolve_summary_system_rows(
    bundle: dict[str, Any],
    metrics: list[dict[str, Any]],
    sample: dict[str, Any],
    params: dict[str, Any],
    *,
    activation_threshold: float,
) -> list[dict[str, Any]]:
    """Roll finest-level metrics up to Level_2 coarse Allen systems."""
    density_excel = sample.get("density_excel")
    if not density_excel or not metrics:
        return [
            _compact_system_row(row)
            for row in bundle.get("system_metrics") or []
            if row.get("source") in SUMMARY_SYSTEM_SOURCES
        ]

    from pipeline_modules.visualization.cfos_report_data import compute_system_metrics

    return [
        _compact_system_row(row)
        for row in compute_system_metrics(
            density_excel,
            metrics,
            level=params.get("default_level"),
            cfg_path=params.get("cfg_path"),
            activation_threshold=activation_threshold,
        )
        if row.get("source") in SUMMARY_SYSTEM_SOURCES
    ]


def build_summary_payload(
    bundle: dict[str, Any],
    *,
    sample_dir: str | Path | None = None,
    level: str | None = None,
) -> dict[str, Any]:
    """Build the Summary-tab payload from a full report bundle."""
    params = bundle.get("parameters") or {}
    sample = dict(bundle.get("sample") or {})
    overview = dict(bundle.get("overview") or {})
    metrics = list(bundle.get("region_metrics") or [])
    activation_threshold = float(
        overview.get("activation_threshold") or params.get("activation_threshold") or 0.0
    )
    resolved_level = level or overview.get("level") or params.get("default_level") or ""
    resolved_sample_dir = str(sample_dir or sample.get("sample_dir") or "")
    signal_ch = str(params.get("signal_ch") or "ch1")

    if resolved_level and resolved_level != overview.get("level") and metrics:
        from pipeline_modules.visualization.cfos_report_data import compute_overview

        overview = compute_overview(metrics, level=resolved_level, activation_threshold=activation_threshold)

    leaf_stats = None
    cfg_path = params.get("cfg_path")
    if metrics and cfg_path:
        from pipeline_modules.visualization.cfos_report_data import compute_leaf_region_stats

        leaf_stats = compute_leaf_region_stats(
            metrics,
            cfg_path=cfg_path,
            activation_threshold=activation_threshold,
        )

    systems = _resolve_summary_system_rows(
        bundle,
        metrics,
        sample,
        params,
        activation_threshold=activation_threshold,
    )

    whole_brain = None
    density_excel = sample.get("density_excel")
    if density_excel:
        try:
            from pipeline_modules.visualization.cfos_report_data import read_whole_brain_stats_from_excel

            whole_brain = read_whole_brain_stats_from_excel(density_excel)
        except Exception:
            whole_brain = None

    if whole_brain:
        headline_cfos = float(whole_brain["total_cfos_count"])
        headline_signal_voxels = float(whole_brain["signal_voxels"])
        headline_brain_voxels = float(whole_brain["total_region_volume_voxels"])
        signal_volume_um3 = float(whole_brain["signal_volume_um3"])
        brain_volume_um3 = float(whole_brain["brain_volume_um3"])
    else:
        headline_cfos = float(overview.get("total_cfos_count") or 0.0)
        headline_signal_voxels = float(overview.get("total_signal_voxels") or 0.0)
        headline_brain_voxels = float(overview.get("total_region_volume_voxels") or 0.0)
        signal_volume_um3 = headline_signal_voxels * VOXEL_VOLUME_UM3
        brain_volume_um3 = headline_brain_voxels * VOXEL_VOLUME_UM3

    laterality = None
    if whole_brain and whole_brain.get("whole_brain_count_laterality_index") is not None:
        laterality = {
            "left_total_cfos_count": float(whole_brain.get("left_total_cfos_count") or 0.0),
            "right_total_cfos_count": float(whole_brain.get("right_total_cfos_count") or 0.0),
            "whole_brain_count_laterality_index": whole_brain.get("whole_brain_count_laterality_index"),
        }
    elif overview.get("whole_brain_count_laterality_index") is not None:
        laterality = {
            "left_total_cfos_count": float(overview.get("left_total_cfos_count") or 0.0),
            "right_total_cfos_count": float(overview.get("right_total_cfos_count") or 0.0),
            "whole_brain_count_laterality_index": overview.get("whole_brain_count_laterality_index"),
        }

    spatial = bundle.get("spatial") or {}
    sample_assets = sample

    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "sample": {
            "sample_id": str(sample.get("sample_id") or Path(resolved_sample_dir).name),
            "sample_dir": resolved_sample_dir,
            "signal_ch": signal_ch,
            "group": sample.get("group"),
            "density_excel": str(sample.get("density_excel") or ""),
            "atlas_version": str(sample.get("atlas_version") or ""),
        },
        "atlas": {
            "level": resolved_level,
            "default_level": str(params.get("default_level") or resolved_level),
            "atlas_label_available": bool(params.get("atlas_label_available", True)),
            "atlas_label_error": params.get("atlas_label_error"),
            "atlas_shape_dv_ap_ml": list(params.get("atlas_shape_dv_ap_ml") or []),
        },
        "data_availability": {
            "density_excel": bool(sample.get("density_excel")),
            "atlas_label": bool(params.get("atlas_label_available", True)),
            "spatial_axes": bool(spatial.get("available")),
            "points_csv": bool(sample_assets.get("points_csv")),
            "spotiflow_points_csv": bool(sample_assets.get("spotiflow_points_csv")),
            "atlas_volume_tiff": bool(sample_assets.get("atlas_volume_tiff")),
        },
        "headline_stats": {
            "total_cfos_count": headline_cfos,
            "signal_voxels": headline_signal_voxels,
            "total_region_volume_voxels": headline_brain_voxels,
            "signal_volume_um3": signal_volume_um3,
            "brain_volume_um3": brain_volume_um3,
            "voxel_volume_um3": VOXEL_VOLUME_UM3,
            "whole_brain_voxel_density": _safe_ratio(headline_signal_voxels, headline_brain_voxels),
            "activated_region_count": int(
                (leaf_stats or overview).get("activated_region_count") or 0
            ),
            "total_region_count": int((leaf_stats or overview).get("total_region_count") or 0),
            "activation_threshold": float(
                (leaf_stats or overview).get("activation_threshold")
                or overview.get("activation_threshold")
                or 0.0
            ),
            "leaf_region_scope": (leaf_stats or {}).get("scope"),
            "systems_scope": "coarse_excel_lookup",
            "source": "excel_level_0_root" if whole_brain else "level_aggregate",
        },
        "laterality": laterality,
        "top_regions_by_count": [
            _compact_region_row(row) for row in (overview.get("top_by_cfos_count") or [])[:SUMMARY_TOP_REGIONS]
        ],
        "top_regions_by_density": [
            _compact_region_row(row) for row in (overview.get("top_by_voxel_density") or [])[:SUMMARY_TOP_REGIONS]
        ],
        "systems": systems,
        "findings": [_compact_finding(row) for row in (bundle.get("findings") or [])[:SUMMARY_TOP_FINDINGS]],
    }
    return summary


def write_summary_json(output_path: str | Path, summary: dict[str, Any]) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return output_path


def read_summary_json(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != SUMMARY_SCHEMA_VERSION:
        raise ValueError(f"Unsupported summary schema: {payload.get('schema_version')}")
    return payload


def read_summary_json_if_exists(path: str | Path) -> dict[str, Any] | None:
    path = Path(path)
    if not path.is_file():
        return None
    try:
        return read_summary_json(path)
    except (OSError, json.JSONDecodeError, ValueError):
        return None
