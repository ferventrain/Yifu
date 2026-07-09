"""Multi-sample group statistics for the cFos visual report (P3)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy import stats

from pipeline_modules.visualization.cfos_report_data import (
    DEFAULT_CFG,
    aggregate_region_metrics_to_level,
    collect_subtree_region_ids,
    load_region_metadata_table,
    normalize_region_metrics,
)
from pipeline_modules.visualization.heatmap import resolve_density_excel_path
from pipeline_modules.visualization.region_group_signal_count import DEFAULT_GROUPS, load_region_groups

MetricName = Literal[
    "cfos_count",
    "signal_voxels",
    "voxel_density",
    "mean_cfos_intensity",
    "sum_intensity",
]

METRIC_FIELDS = {
    "cfos_count": "cfos_count",
    "signal_voxels": "signal_voxels",
    "voxel_density": "voxel_density",
    "mean_cfos_intensity": "mean_cfos_intensity",
    "sum_intensity": "sum_intensity",
}


def _log2_fold_change(mean_a: float, mean_b: float, *, pseudocount: float = 1.0) -> float:
    return float(np.log2((mean_b + pseudocount) / (mean_a + pseudocount)))

def load_group_manifest(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Group manifest not found: {path}")

    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and "samples" in payload:
            entries = payload["samples"]
        elif isinstance(payload, list):
            entries = payload
        else:
            raise ValueError("JSON manifest must be a list or an object with a 'samples' array.")
    else:
        frame = pd.read_csv(path)
        required = {"sample_dir", "group"}
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"Manifest CSV missing required column(s): {sorted(missing)}")
        entries = frame.to_dict(orient="records")

    normalized: list[dict[str, Any]] = []
    for entry in entries:
        sample_dir = Path(str(entry["sample_dir"]))
        group = str(entry["group"]).strip()
        if not group:
            raise ValueError(f"Empty group label for sample_dir={sample_dir}")
        signal_ch = str(entry.get("signal_ch", "ch1")).strip() or "ch1"
        sample_id = str(entry.get("sample_id") or sample_dir.name)
        normalized.append(
            {
                "sample_id": sample_id,
                "sample_dir": str(sample_dir),
                "group": group,
                "signal_ch": signal_ch,
            }
        )
    if len(normalized) < 2:
        raise ValueError("Group manifest must contain at least two samples.")
    return normalized


def parse_group_manifest_json(text: str) -> list[dict[str, Any]]:
    payload = json.loads(text)
    if isinstance(payload, list):
        entries = payload
    elif isinstance(payload, dict) and "samples" in payload:
        entries = payload["samples"]
    else:
        raise ValueError("Manifest JSON must be a list or an object with a 'samples' array.")

    normalized: list[dict[str, Any]] = []
    for entry in entries:
        sample_dir = Path(str(entry["sample_dir"]))
        group = str(entry["group"]).strip()
        if not group:
            raise ValueError(f"Empty group label for sample_dir={sample_dir}")
        signal_ch = str(entry.get("signal_ch", "ch1")).strip() or "ch1"
        sample_id = str(entry.get("sample_id") or sample_dir.name)
        normalized.append(
            {
                "sample_id": sample_id,
                "sample_dir": str(sample_dir),
                "group": group,
                "signal_ch": signal_ch,
            }
        )
    if len(normalized) < 2:
        raise ValueError("Group manifest must contain at least two samples.")
    return normalized


def build_long_format_metrics(
    manifest: list[dict[str, Any]],
    *,
    cfg_path: str | Path = DEFAULT_CFG,
    level: str,
    metric: str = "cfos_count",
) -> pd.DataFrame:
    if metric not in METRIC_FIELDS:
        raise ValueError(f"Unsupported metric: {metric}")

    rows: list[dict[str, Any]] = []
    region_lookup = load_region_metadata_table(cfg_path)
    for entry in manifest:
        sample_dir = Path(entry["sample_dir"])
        signal_ch = entry.get("signal_ch", "ch1")
        density_excel = resolve_density_excel_path(sample_dir, None, signal_ch=signal_ch)
        metrics = normalize_region_metrics(
            density_excel,
            sample_id=entry["sample_id"],
            cfg_path=cfg_path,
        )
        aggregated = aggregate_region_metrics_to_level(metrics, level, region_lookup)
        for row in aggregated:
            rows.append(
                {
                    "sample_id": entry["sample_id"],
                    "group": entry["group"],
                    "region_id": int(row["region_id"]),
                    "region_name": row["region_name"],
                    "region_acronym": row["region_acronym"],
                    "value": float(row[metric]),
                }
            )
    if not rows:
        raise ValueError(f"No region metrics found for level={level}")
    return pd.DataFrame(rows)


def compute_differential_regions(
    long_df: pd.DataFrame,
    *,
    group_a: str,
    group_b: str,
    min_mean: float = 0.0,
    pseudocount: float = 1.0,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for region_id, region_frame in long_df.groupby("region_id"):
        values_a = region_frame.loc[region_frame["group"] == group_a, "value"].to_numpy(dtype=float)
        values_b = region_frame.loc[region_frame["group"] == group_b, "value"].to_numpy(dtype=float)
        if values_a.size == 0 or values_b.size == 0:
            continue
        mean_a = float(np.mean(values_a))
        mean_b = float(np.mean(values_b))
        if max(mean_a, mean_b) < min_mean:
            continue
        log2_fc = _log2_fold_change(mean_a, mean_b, pseudocount=pseudocount)
        meta = region_frame.iloc[0]
        rows.append(
            {
                "region_id": int(region_id),
                "region_name": str(meta["region_name"]),
                "region_acronym": str(meta["region_acronym"]),
                "group_a": group_a,
                "group_b": group_b,
                "mean_a": mean_a,
                "mean_b": mean_b,
                "delta": mean_b - mean_a,
                "fold_change": float((mean_b + pseudocount) / (mean_a + pseudocount)),
                "log2_fold_change": log2_fc,
                "n_a": int(values_a.size),
                "n_b": int(values_b.size),
            }
        )

    rows.sort(key=lambda item: abs(float(item["log2_fold_change"])), reverse=True)
    return rows


def compute_differential_systems(
    manifest: list[dict[str, Any]],
    *,
    cfg_path: str | Path = DEFAULT_CFG,
    groups_json: str | Path = DEFAULT_GROUPS,
    level: str,
    metric: str,
    group_a: str,
    group_b: str,
) -> list[dict[str, Any]]:
    groups = load_region_groups(groups_json)
    region_lookup = load_region_metadata_table(cfg_path)
    lookup_by_acronym = region_lookup.drop_duplicates("region_acronym", keep="first").set_index("region_acronym")

    system_rows: list[dict[str, Any]] = []
    for entry in manifest:
        sample_dir = Path(entry["sample_dir"])
        signal_ch = entry.get("signal_ch", "ch1")
        density_excel = resolve_density_excel_path(sample_dir, None, signal_ch=signal_ch)
        metrics = normalize_region_metrics(
            density_excel,
            sample_id=entry["sample_id"],
            cfg_path=cfg_path,
        )
        metrics_by_id = {
            int(item["region_id"]): item
            for item in metrics
            if item["level"] == level
        }

        for group_name, acronyms in groups.items():
            member_cfos = 0.0
            member_signal_voxels = 0.0
            member_total_voxels = 0.0
            matched = False
            for acronym in acronyms:
                if acronym not in lookup_by_acronym.index:
                    continue
                region_id = int(lookup_by_acronym.loc[acronym, "region_id"])
                region_metric = metrics_by_id.get(region_id)
                if region_metric is None:
                    continue
                matched = True
                member_cfos += float(region_metric["cfos_count"])
                member_signal_voxels += float(region_metric["signal_voxels"])
                member_total_voxels += float(region_metric["region_volume_voxels"])

            if not matched:
                continue

            if metric == "voxel_density":
                value = float(member_signal_voxels / member_total_voxels) if member_total_voxels else 0.0
            elif metric == "signal_voxels":
                value = member_signal_voxels
            else:
                value = member_cfos

            system_rows.append(
                {
                    "sample_id": entry["sample_id"],
                    "group": entry["group"],
                    "system_name": group_name,
                    "value": value,
                }
            )

    if not system_rows:
        return []

    frame = pd.DataFrame(system_rows)
    results: list[dict[str, Any]] = []
    for system_name, system_frame in frame.groupby("system_name"):
        values_a = system_frame.loc[system_frame["group"] == group_a, "value"].to_numpy(dtype=float)
        values_b = system_frame.loc[system_frame["group"] == group_b, "value"].to_numpy(dtype=float)
        if values_a.size == 0 or values_b.size == 0:
            continue
        mean_a = float(np.mean(values_a))
        mean_b = float(np.mean(values_b))
        results.append(
            {
                "system_name": str(system_name),
                "group_a": group_a,
                "group_b": group_b,
                "mean_a": mean_a,
                "mean_b": mean_b,
                "delta": mean_b - mean_a,
                "log2_fold_change": _log2_fold_change(mean_a, mean_b),
                "n_a": int(values_a.size),
                "n_b": int(values_b.size),
            }
        )

    results.sort(key=lambda item: abs(float(item["log2_fold_change"])), reverse=True)
    return results


HeatmapMode = Literal["differential", "absolute"]


def resolve_region_scope_ids(
    long_df: pd.DataFrame,
    *,
    focus_region_id: int | None,
    cfg_path: str | Path,
) -> list[int] | None:
    """Return ordered region ids for analysis scope, or None for all regions at level."""
    if focus_region_id is None:
        return None

    focus_region_id = int(focus_region_id)
    subtree = collect_subtree_region_ids(focus_region_id, cfg_path)
    present = {int(value) for value in long_df["region_id"].unique()}
    scoped = sorted(present.intersection(subtree))
    if not scoped:
        return [focus_region_id] if focus_region_id in present else []

    ordered: list[int] = []
    if focus_region_id in present:
        ordered.append(focus_region_id)
    meta = long_df.drop_duplicates("region_id").set_index("region_id")
    children = [region_id for region_id in scoped if region_id != focus_region_id]
    children.sort(
        key=lambda region_id: (
            str(meta.loc[region_id, "region_acronym"]) if region_id in meta.index else "",
            region_id,
        )
    )
    ordered.extend(children)
    return ordered


def filter_long_df_to_scope(
    long_df: pd.DataFrame,
    region_ids: list[int] | None,
) -> pd.DataFrame:
    if not region_ids:
        return long_df
    allowed = {int(value) for value in region_ids}
    return long_df[long_df["region_id"].isin(allowed)].copy()


def build_group_heatmap(
    long_df: pd.DataFrame,
    *,
    top_n: int = 36,
    differential: list[dict[str, Any]] | None = None,
    region_ids: list[int] | None = None,
    heatmap_mode: HeatmapMode = "differential",
) -> dict[str, Any]:
    if region_ids:
        top_regions = [int(value) for value in region_ids]
    elif heatmap_mode == "absolute":
        region_means = long_df.groupby("region_id")["value"].mean().sort_values(ascending=False)
        top_regions = [int(region_id) for region_id in region_means.head(top_n).index.tolist()]
    elif differential:
        top_regions = [row["region_id"] for row in differential[:top_n]]
    else:
        region_means = long_df.groupby("region_id")["value"].mean().sort_values(ascending=False)
        top_regions = [int(region_id) for region_id in region_means.head(top_n).index.tolist()]

    samples = sorted(long_df["sample_id"].unique().tolist())
    sample_groups = (
        long_df.drop_duplicates("sample_id")
        .set_index("sample_id")["group"]
        .to_dict()
    )
    matrix: list[list[float]] = []
    region_labels: list[str] = []
    for region_id in top_regions:
        region_frame = long_df[long_df["region_id"] == int(region_id)]
        if region_frame.empty:
            continue
        region_labels.append(str(region_frame.iloc[0]["region_acronym"]))
        matrix.append(
            [
                float(region_frame.loc[region_frame["sample_id"] == sample_id, "value"].sum())
                for sample_id in samples
            ]
        )
    flat = [value for row in matrix for value in row]
    value_min = float(min(flat)) if flat else 0.0
    value_max = float(max(flat)) if flat else 0.0
    return {
        "mode": heatmap_mode,
        "samples": samples,
        "sample_groups": sample_groups,
        "region_ids": [int(value) for value in top_regions[: len(region_labels)]],
        "region_labels": region_labels,
        "matrix": matrix,
        "value_min": value_min,
        "value_max": value_max,
    }


def select_top_differential_regions(
    differential: list[dict[str, Any]],
    *,
    top_n: int = 20,
) -> list[dict[str, Any]]:
    ranked = sorted(
        differential,
        key=lambda row: abs(float(row.get("log2_fold_change", 0.0))),
        reverse=True,
    )
    return ranked[: int(top_n)]


def build_group_comparison_scatter(
    long_df: pd.DataFrame,
    *,
    group_a: str,
    group_b: str,
    region_ids: list[int] | None = None,
) -> dict[str, Any]:
    frame = filter_long_df_to_scope(long_df, region_ids)
    samples_a = sorted(frame.loc[frame["group"] == group_a, "sample_id"].unique().tolist())
    samples_b = sorted(frame.loc[frame["group"] == group_b, "sample_id"].unique().tolist())
    if not samples_a or not samples_b:
        return {
            "available": False,
            "reason": f"Both groups must have at least one sample ({group_a}, {group_b}).",
            "points": [],
        }

    group_means = (
        frame.groupby(["region_id", "group"], as_index=False)["value"]
        .mean()
        .pivot(index="region_id", columns="group", values="value")
    )
    if group_a not in group_means.columns or group_b not in group_means.columns:
        return {
            "available": False,
            "reason": "Selected groups are missing from the region table.",
            "points": [],
        }

    meta = frame.drop_duplicates("region_id").set_index("region_id")
    points: list[dict[str, Any]] = []
    xs: list[float] = []
    ys: list[float] = []
    for region_id in sorted(int(value) for value in group_means.index.tolist()):
        value_a = float(group_means.loc[region_id, group_a])
        value_b = float(group_means.loc[region_id, group_b])
        if not (np.isfinite(value_a) and np.isfinite(value_b)):
            continue
        xs.append(value_a)
        ys.append(value_b)
        row = meta.loc[region_id] if region_id in meta.index else None
        points.append(
            {
                "region_id": region_id,
                "region_acronym": str(row["region_acronym"]) if row is not None else str(region_id),
                "region_name": str(row["region_name"]) if row is not None else str(region_id),
                "x": value_a,
                "y": value_b,
            }
        )

    if len(xs) < 2:
        return {
            "available": False,
            "reason": "Not enough overlapping regions for Pearson correlation.",
            "points": points,
        }

    pearson_r, pearson_p = stats.pearsonr(np.asarray(xs, dtype=float), np.asarray(ys, dtype=float))
    mode = "pairwise" if len(samples_a) == 1 and len(samples_b) == 1 else "group_mean"
    return {
        "available": True,
        "mode": mode,
        "sample_a": str(samples_a[0]) if len(samples_a) == 1 else None,
        "sample_b": str(samples_b[0]) if len(samples_b) == 1 else None,
        "samples_a": samples_a,
        "samples_b": samples_b,
        "group_a": group_a,
        "group_b": group_b,
        "points": points,
        "pearson_r": float(pearson_r),
        "pearson_p": float(pearson_p),
        "n_regions": len(points),
    }


def build_pairwise_scatter_payload(
    long_df: pd.DataFrame,
    *,
    group_a: str,
    group_b: str,
    region_ids: list[int] | None = None,
) -> dict[str, Any]:
    return build_group_comparison_scatter(
        long_df,
        group_a=group_a,
        group_b=group_b,
        region_ids=region_ids,
    )


def build_sample_correlation_payload(
    long_df: pd.DataFrame,
    *,
    region_ids: list[int] | None = None,
) -> dict[str, Any]:
    frame = filter_long_df_to_scope(long_df, region_ids)
    samples = sorted(frame["sample_id"].unique().tolist())
    if len(samples) < 2:
        return {
            "available": False,
            "reason": "Sample correlation requires at least two samples.",
            "samples": samples,
            "matrix": [],
        }

    pivot = frame.pivot_table(index="sample_id", columns="region_id", values="value", aggfunc="mean")
    pivot = pivot.dropna(axis=1, how="all")
    if pivot.shape[1] < 2:
        return {
            "available": False,
            "reason": "Not enough overlapping regions for sample correlation.",
            "samples": samples,
            "matrix": [],
        }

    corr = pivot.T.corr(method="pearson")
    sample_groups = (
        frame.drop_duplicates("sample_id")
        .set_index("sample_id")["group"]
        .to_dict()
    )
    ordered_samples = [sample_id for sample_id in samples if sample_id in corr.index]
    corr = corr.loc[ordered_samples, ordered_samples]
    return {
        "available": True,
        "samples": ordered_samples,
        "sample_groups": {sample_id: sample_groups.get(sample_id, "") for sample_id in ordered_samples},
        "matrix": corr.fillna(0.0).astype(float).values.tolist(),
        "n_regions": int(pivot.shape[1]),
    }


def build_pairwise_manifest(
    sample_a_dir: str | Path,
    sample_b_dir: str | Path,
    *,
    signal_ch_a: str = "ch1",
    signal_ch_b: str | None = None,
    group_a: str | None = None,
    group_b: str | None = None,
) -> list[dict[str, Any]]:
    sample_a_dir = Path(sample_a_dir)
    sample_b_dir = Path(sample_b_dir)
    if not sample_a_dir.is_dir():
        raise FileNotFoundError(f"Sample A directory not found: {sample_a_dir}")
    if not sample_b_dir.is_dir():
        raise FileNotFoundError(f"Sample B directory not found: {sample_b_dir}")
    if sample_a_dir.resolve() == sample_b_dir.resolve():
        raise ValueError("Sample A and Sample B must be different directories.")

    signal_ch_b = signal_ch_b or signal_ch_a
    label_a = group_a or sample_a_dir.name
    label_b = group_b or sample_b_dir.name
    return [
        {
            "sample_id": sample_a_dir.name,
            "sample_dir": str(sample_a_dir),
            "group": label_a,
            "signal_ch": signal_ch_a,
        },
        {
            "sample_id": sample_b_dir.name,
            "sample_dir": str(sample_b_dir),
            "group": label_b,
            "signal_ch": signal_ch_b,
        },
    ]


def build_group_analysis_payload(
    manifest: list[dict[str, Any]],
    *,
    cfg_path: str | Path = DEFAULT_CFG,
    groups_json: str | Path = DEFAULT_GROUPS,
    level: str,
    metric: str = "cfos_count",
    group_a: str | None = None,
    group_b: str | None = None,
    top_n: int = 36,
    focus_region_id: int | None = None,
    heatmap_mode: HeatmapMode = "differential",
) -> dict[str, Any]:
    long_df = build_long_format_metrics(
        manifest,
        cfg_path=cfg_path,
        level=level,
        metric=metric,
    )
    region_scope_ids = resolve_region_scope_ids(
        long_df,
        focus_region_id=focus_region_id,
        cfg_path=cfg_path,
    )
    scoped_df = filter_long_df_to_scope(long_df, region_scope_ids)

    groups = sorted(long_df["group"].unique().tolist())
    if len(groups) < 2:
        raise ValueError("At least two distinct groups are required for group analysis.")

    resolved_a = group_a or groups[0]
    resolved_b = group_b or groups[1]
    if resolved_a not in groups or resolved_b not in groups:
        raise ValueError(f"Groups must be chosen from {groups}")

    differential_regions = compute_differential_regions(
        scoped_df,
        group_a=resolved_a,
        group_b=resolved_b,
    )
    differential_systems = compute_differential_systems(
        manifest,
        cfg_path=cfg_path,
        groups_json=groups_json,
        level=level,
        metric=metric,
        group_a=resolved_a,
        group_b=resolved_b,
    )
    heatmap_region_ids = region_scope_ids
    if heatmap_mode == "differential" and region_scope_ids:
        ranked = [row["region_id"] for row in differential_regions]
        scoped_set = set(region_scope_ids)
        heatmap_region_ids = [region_id for region_id in ranked if region_id in scoped_set]
        if focus_region_id is not None and int(focus_region_id) in scoped_set:
            parent_id = int(focus_region_id)
            if parent_id not in heatmap_region_ids:
                heatmap_region_ids.insert(0, parent_id)
            else:
                heatmap_region_ids = [parent_id] + [
                    region_id for region_id in heatmap_region_ids if region_id != parent_id
                ]

    heatmap = build_group_heatmap(
        scoped_df,
        top_n=top_n,
        differential=differential_regions,
        region_ids=heatmap_region_ids,
        heatmap_mode=heatmap_mode,
    )
    top_differential_regions = select_top_differential_regions(differential_regions, top_n=min(top_n, 20))
    pairwise_scatter = build_group_comparison_scatter(
        scoped_df,
        group_a=resolved_a,
        group_b=resolved_b,
        region_ids=region_scope_ids,
    )
    sample_correlation = build_sample_correlation_payload(scoped_df, region_ids=region_scope_ids)

    focus_meta = None
    if focus_region_id is not None:
        lookup = load_region_metadata_table(cfg_path)
        matches = lookup[lookup["region_id"] == int(focus_region_id)]
        if not matches.empty:
            row = matches.iloc[0]
            focus_meta = {
                "region_id": int(focus_region_id),
                "region_name": str(row["region_name"]),
                "region_acronym": str(row["region_acronym"]),
            }

    return {
        "available": True,
        "level": level,
        "metric": metric,
        "heatmap_mode": heatmap_mode,
        "focus_region": focus_meta,
        "region_scope_ids": region_scope_ids or [],
        "groups": groups,
        "comparison": {"group_a": resolved_a, "group_b": resolved_b},
        "sample_count": len(manifest),
        "is_pairwise": len(manifest) == 2,
        "region_count": int(scoped_df["region_id"].nunique()),
        "differential_regions": differential_regions,
        "top_differential_regions": top_differential_regions,
        "differential_systems": differential_systems,
        "heatmap": heatmap,
        "pairwise_scatter": pairwise_scatter,
        "sample_correlation": sample_correlation,
        "manifest": manifest,
    }


def export_differential_regions_csv(differential: list[dict[str, Any]]) -> str:
    frame = pd.DataFrame(differential)
    return frame.to_csv(index=False)
