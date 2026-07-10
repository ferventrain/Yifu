"""Compute cFos / segmentation statistics restricted to one Allen brain region subtree."""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pipeline_modules.registration.region_signal_analysis_zarr_graph import (
    aggregate_final_region_stats,
    build_binary_mask,
    build_root_sizes,
    choose_block_shape,
    scan_blocks_and_write_artifacts,
    stitch_block_boundaries,
    validate_zarr_inputs,
)
from pipeline_modules.segmentation.zarr_utils import open_zarr_dataset as open_seg_zarr_dataset
from pipeline_modules.tubule_reconstruction.region_vessel_analysis import (
    _collect_subtree_ids,
    load_region_tree_with_lookups,
    resolve_region_query,
)
from pipeline_modules.utils.sample_layout import SampleLayout
from pipeline_modules.visualization.heatmap import resolve_sample_stack_defaults
from pipeline_modules.visualization.region_mip_export import DEFAULT_REGION_CFG
from pipeline_modules.visualization.warp_mask_zarr_to_atlas_points import iter_block_slices

logger = logging.getLogger(__name__)


def resolve_region_subtree_ids(region_query: str, *, cfg_path: str | Path) -> tuple[set[int], str, str, dict[int, dict[str, Any]]]:
    nodes_by_id, acronym_to_ids, name_to_ids = load_region_tree_with_lookups(cfg_path)
    node = resolve_region_query(region_query, nodes_by_id, acronym_to_ids, name_to_ids)
    subtree_ids = set(_collect_subtree_ids(node))
    slug = str(node.get("acronym") or node.get("name") or node["id"]).strip()
    display_name = str(node.get("name") or slug)
    nodes = {
        int(region_id): {
            "id": int(region_id),
            "name": str(nodes_by_id[region_id].get("name") or region_id),
            "acronym": str(nodes_by_id[region_id].get("acronym") or ""),
        }
        for region_id in subtree_ids
        if int(region_id) in nodes_by_id
    }
    return subtree_ids, slug, display_name, nodes


def _parse_resolution_xyz(value: str) -> tuple[float, float, float]:
    parts = [float(part.strip()) for part in str(value).split(",") if part.strip()]
    if len(parts) != 3:
        raise ValueError(f"resolution_xyz must be x,y,z, got: {value}")
    return parts[0], parts[1], parts[2]


def _parse_block_shape(value: str) -> tuple[int, int, int] | None:
    text = str(value).strip()
    if not text:
        return None
    parts = [int(part.strip()) for part in text.split(",") if part.strip()]
    if len(parts) != 3:
        raise ValueError(f"block_shape must be z,y,x, got: {value}")
    return parts[0], parts[1], parts[2]


def _gate_mask_to_scope(
    mask_chunk: np.ndarray,
    label_chunk: np.ndarray,
    scope_ids: np.ndarray,
    *,
    foreground_mode: str,
    foreground_label: int,
) -> np.ndarray:
    scoped = build_binary_mask(mask_chunk, foreground_mode, foreground_label) & np.isin(label_chunk, scope_ids)
    if foreground_mode == "nonzero":
        gated = mask_chunk.copy()
        gated[~scoped] = 0
        return gated
    gated = np.zeros_like(mask_chunk)
    gated[scoped] = foreground_label
    return gated


def _accumulate_scope_voxel_stats(
    *,
    mask_arr: Any,
    label_arr: Any,
    signal_arr: Any,
    scope_ids: set[int],
    block_shape: tuple[int, int, int],
    foreground_mode: str,
    foreground_label: int,
) -> dict[str, Any]:
    scope_array = np.asarray(sorted(scope_ids), dtype=np.int64)
    region_voxels: dict[int, int] = {}
    signal_voxels: dict[int, int] = {}
    sum_intensity: dict[int, float] = {}

    for block_slices in iter_block_slices(mask_arr.shape, block_shape):
        label_chunk = np.asarray(label_arr[block_slices])
        in_scope = np.isin(label_chunk, scope_array)
        if not np.any(in_scope):
            continue

        scoped_labels = label_chunk[in_scope].astype(np.int64, copy=False)
        region_counts = np.bincount(scoped_labels, minlength=int(scope_array.max()) + 1)
        for region_id in scope_array:
            count = int(region_counts[int(region_id)])
            if count:
                region_voxels[int(region_id)] = region_voxels.get(int(region_id), 0) + count

        mask_chunk = np.asarray(mask_arr[block_slices])
        signal_chunk = np.asarray(signal_arr[block_slices])
        gated_mask = _gate_mask_to_scope(
            mask_chunk,
            label_chunk,
            scope_array,
            foreground_mode=foreground_mode,
            foreground_label=foreground_label,
        )
        foreground = build_binary_mask(gated_mask, foreground_mode, foreground_label) & in_scope
        if not np.any(foreground):
            continue

        active_labels = label_chunk[foreground].astype(np.int64, copy=False)
        active_signal = signal_chunk[foreground].astype(np.float64, copy=False)
        signal_counts = np.bincount(active_labels, minlength=int(scope_array.max()) + 1)
        intensity_sums = np.bincount(active_labels, weights=active_signal, minlength=int(scope_array.max()) + 1)
        for region_id in scope_array:
            svox = int(signal_counts[int(region_id)])
            if svox:
                signal_voxels[int(region_id)] = signal_voxels.get(int(region_id), 0) + svox
                sum_intensity[int(region_id)] = sum_intensity.get(int(region_id), 0.0) + float(intensity_sums[int(region_id)])

    total_region_voxels = int(sum(region_voxels.values()))
    total_signal_voxels = int(sum(signal_voxels.values()))
    total_sum_intensity = float(sum(sum_intensity.values()))
    return {
        "total_region_voxels": total_region_voxels,
        "total_signal_voxels": total_signal_voxels,
        "total_sum_intensity": total_sum_intensity,
        "region_voxels": region_voxels,
        "signal_voxels": signal_voxels,
        "sum_intensity": sum_intensity,
    }


def _scan_blocks_region_scope(
    *,
    mask_zarr,
    label_zarr,
    signal_zarr,
    hemisphere_zarr,
    mask_zarr_path: str,
    label_zarr_path: str,
    signal_zarr_path: str,
    hemisphere_zarr_path: str,
    dataset_name: str,
    block_shape: tuple[int, int, int],
    foreground_mode: str,
    foreground_label: int,
    scope_ids: set[int],
    tmp_dir: Path,
    pass1_workers: int,
):
    scope_array = np.asarray(sorted(scope_ids), dtype=np.int64)

    class _ScopedMask:
        def __init__(self, source):
            self._source = source
            self.shape = source.shape
            self.dtype = source.dtype

        def __getitem__(self, key):
            mask_chunk = np.asarray(self._source[key])
            label_chunk = np.asarray(label_zarr[key])
            return _gate_mask_to_scope(
                mask_chunk,
                label_chunk,
                scope_array,
                foreground_mode=foreground_mode,
                foreground_label=foreground_label,
            )

    scoped_mask = _ScopedMask(mask_zarr)
    return scan_blocks_and_write_artifacts(
        mask_zarr=scoped_mask,
        label_zarr=label_zarr,
        signal_zarr=signal_zarr,
        hemisphere_zarr=hemisphere_zarr,
        mask_zarr_path=mask_zarr_path,
        label_zarr_path=label_zarr_path,
        signal_zarr_path=signal_zarr_path,
        hemisphere_zarr_path=hemisphere_zarr_path,
        dataset_name=dataset_name,
        block_shape=block_shape,
        foreground_mode=foreground_mode,
        foreground_label=foreground_label,
        tmp_dir=tmp_dir,
        pass1_workers=pass1_workers,
    )


def _build_subregion_frame(
    *,
    scope_nodes: dict[int, dict[str, Any]],
    voxel_stats: dict[str, Any],
    signal_stats: dict[str, Any] | None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for region_id in sorted(scope_nodes):
        node = scope_nodes[region_id]
        total_voxels = int(voxel_stats["region_voxels"].get(region_id, 0))
        sig_voxels = int(voxel_stats["signal_voxels"].get(region_id, 0))
        sum_int = float(voxel_stats["sum_intensity"].get(region_id, 0.0))
        row = {
            "Region ID": region_id,
            "Acronym": node.get("acronym") or "",
            "Name": node.get("name") or "",
            "Total Voxels": total_voxels,
            "Signal Voxels": sig_voxels,
            "Voxel Density": float(sig_voxels / total_voxels) if total_voxels else 0.0,
            "Sum Intensity": sum_int,
            "Mean Intensity": float(sum_int / sig_voxels) if sig_voxels else 0.0,
        }
        if signal_stats is not None:
            row["Signal Count"] = int(signal_stats["region_signal_counts"].get(region_id, 0))
        rows.append(row)
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame = frame.sort_values(["Signal Voxels", "Total Voxels"], ascending=False).reset_index(drop=True)
    return frame


def compute_region_scope_signal_stats(
    *,
    sample_dir: str | Path,
    region_query: str,
    signal_ch: str | None = None,
    mask_zarr_path: str | Path | None = None,
    label_zarr_path: str | Path | None = None,
    signal_zarr_path: str | Path | None = None,
    hemisphere_zarr_path: str | Path | None = None,
    cfg_path: str | Path | None = None,
    config_path: str | Path | None = None,
    dataset_name: str = "0",
    block_shape: tuple[int, int, int] | None = None,
    foreground_mode: str = "equal",
    foreground_label: int = 1,
    min_voxels: int = 10,
    pass1_workers: int = 4,
    resolution_xyz: tuple[float, float, float] = (1.8, 1.8, 2.0),
    include_signal_count: bool = True,
    output_dir: str | Path | None = None,
    keep_tmp: bool = False,
) -> dict[str, Any]:
    sample_dir = Path(sample_dir)
    defaults = resolve_sample_stack_defaults(sample_dir, config_path=config_path)
    resolved_signal_ch = signal_ch or str(defaults["signal_ch"])
    layout = SampleLayout(sample_dir=sample_dir, signal_ch=resolved_signal_ch, reg_ch=str(defaults["register_ch"]))

    mask_path = Path(mask_zarr_path or layout.mask_zarr)
    label_path = Path(label_zarr_path or layout.atlas_label_zarr)
    signal_path = Path(signal_zarr_path or layout.signal_zarr)
    hemisphere_path = Path(hemisphere_zarr_path) if hemisphere_zarr_path else layout.atlas_label_hemisphere_zarr
    region_cfg = Path(cfg_path or DEFAULT_REGION_CFG)

    mask_arr = open_seg_zarr_dataset(mask_path, dataset_name=dataset_name)
    label_arr = open_seg_zarr_dataset(label_path, dataset_name=dataset_name)
    signal_arr = open_seg_zarr_dataset(signal_path, dataset_name=dataset_name)
    validate_zarr_inputs(mask_arr, label_arr, signal_arr)

    hemisphere_arr = None
    if hemisphere_path.exists():
        hemisphere_arr = open_seg_zarr_dataset(hemisphere_path, dataset_name=dataset_name)
        if hemisphere_arr.shape != mask_arr.shape:
            raise ValueError(f"Hemisphere Zarr shape mismatch: {hemisphere_arr.shape} vs {mask_arr.shape}")

    scope_ids, region_slug, region_name, scope_nodes = resolve_region_subtree_ids(region_query, cfg_path=region_cfg)
    chosen_block = block_shape or choose_block_shape(mask_arr, label_arr, signal_arr, None)

    started = time.time()
    voxel_stats = _accumulate_scope_voxel_stats(
        mask_arr=mask_arr,
        label_arr=label_arr,
        signal_arr=signal_arr,
        scope_ids=scope_ids,
        block_shape=chosen_block,
        foreground_mode=foreground_mode,
        foreground_label=foreground_label,
    )

    signal_stats = None
    tmp_root: Path | None = None
    if include_signal_count:
        out_parent = Path(output_dir or (sample_dir / "visualization" / "region_stats"))
        out_parent.mkdir(parents=True, exist_ok=True)
        tmp_root = out_parent / f".tmp_{resolved_signal_ch}_{region_slug}_graph"
        if tmp_root.exists():
            shutil.rmtree(tmp_root)
        manifest_payload = _scan_blocks_region_scope(
            mask_zarr=mask_arr,
            label_zarr=label_arr,
            signal_zarr=signal_arr,
            hemisphere_zarr=hemisphere_arr,
            mask_zarr_path=str(mask_path),
            label_zarr_path=str(label_path),
            signal_zarr_path=str(signal_path),
            hemisphere_zarr_path=str(hemisphere_path) if hemisphere_arr is not None else "",
            dataset_name=dataset_name,
            block_shape=chosen_block,
            foreground_mode=foreground_mode,
            foreground_label=foreground_label,
            scope_ids=scope_ids,
            tmp_dir=tmp_root,
            pass1_workers=pass1_workers,
        )
        parent = stitch_block_boundaries(manifest_payload)
        root_sizes = build_root_sizes(manifest_payload, parent)
        signal_stats = aggregate_final_region_stats(
            manifest_payload=manifest_payload,
            parent=parent,
            root_sizes=root_sizes,
            min_voxels=min_voxels,
        )
        if not keep_tmp and tmp_root.exists():
            shutil.rmtree(tmp_root)

    total_voxels = int(voxel_stats["total_region_voxels"])
    total_signal_voxels = int(voxel_stats["total_signal_voxels"])
    total_sum_intensity = float(voxel_stats["total_sum_intensity"])
    voxel_volume_um3 = float(resolution_xyz[0] * resolution_xyz[1] * resolution_xyz[2])
    region_volume_mm3 = total_voxels * voxel_volume_um3 / 1_000_000_000.0

    signal_count = 0
    if signal_stats is not None:
        for region_id in scope_ids:
            signal_count += int(signal_stats["region_signal_counts"].get(int(region_id), 0))

    summary = {
        "sample_dir": str(sample_dir),
        "signal_ch": resolved_signal_ch,
        "region_query": region_query,
        "region_name": region_name,
        "region_slug": region_slug,
        "region_ids_count": len(scope_ids),
        "mask_zarr": str(mask_path),
        "label_zarr": str(label_path),
        "signal_zarr": str(signal_path),
        "total_voxels": total_voxels,
        "signal_voxels": total_signal_voxels,
        "voxel_density": float(total_signal_voxels / total_voxels) if total_voxels else 0.0,
        "signal_count": int(signal_count),
        "sum_intensity": total_sum_intensity,
        "mean_intensity": float(total_sum_intensity / total_signal_voxels) if total_signal_voxels else 0.0,
        "resolution_xyz_um": list(resolution_xyz),
        "region_volume_mm3": region_volume_mm3,
        "foreground_mode": foreground_mode,
        "foreground_label": foreground_label,
        "min_voxels": int(min_voxels),
        "block_shape": list(chosen_block),
        "duration_seconds": round(time.time() - started, 3),
    }

    subregions = _build_subregion_frame(
        scope_nodes=scope_nodes,
        voxel_stats=voxel_stats,
        signal_stats=signal_stats,
    )

    out_dir = Path(output_dir or (sample_dir / "visualization" / "region_stats"))
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{sample_dir.name}_{resolved_signal_ch}_{region_slug}_stats.json"
    xlsx_path = out_dir / f"{sample_dir.name}_{resolved_signal_ch}_{region_slug}_stats.xlsx"
    csv_path = out_dir / f"{sample_dir.name}_{resolved_signal_ch}_{region_slug}_subregions.csv"

    payload = {
        "summary": summary,
        "subregions": subregions.to_dict(orient="records"),
        "output_json": str(json_path),
        "output_xlsx": str(xlsx_path),
        "output_subregions_csv": str(csv_path),
    }
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)

    summary_frame = pd.DataFrame([summary])
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        summary_frame.to_excel(writer, index=False, sheet_name="Summary")
        subregions.to_excel(writer, index=False, sheet_name="Subregions")
    subregions.to_csv(csv_path, index=False)

    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute mask/signal statistics for one Allen brain region subtree (e.g. HIP only)."
    )
    parser.add_argument("--sample-dir", required=True)
    parser.add_argument("--region", required=True, help="Allen acronym/name/id, e.g. HIP")
    parser.add_argument("--signal-ch", default="", help="Signal channel, e.g. ch3")
    parser.add_argument("--mask-zarr", default="")
    parser.add_argument("--label-zarr", default="")
    parser.add_argument("--signal-zarr", default="")
    parser.add_argument("--hemisphere-zarr", default="")
    parser.add_argument("--cfg", default=str(DEFAULT_REGION_CFG))
    parser.add_argument("--config", default="", help="Optional pipeline config.json")
    parser.add_argument("--dataset-name", default="0")
    parser.add_argument("--block-shape", default="", help="Override block size z,y,x")
    parser.add_argument("--foreground-mode", choices=("equal", "nonzero"), default="equal")
    parser.add_argument("--foreground-label", type=int, default=1)
    parser.add_argument("--min-voxels", type=int, default=10, help="Minimum connected-component size")
    parser.add_argument("--pass1-workers", type=int, default=4)
    parser.add_argument("--resolution-xyz", default="1.8,1.8,2.0", help="Voxel size in microns as x,y,z")
    parser.add_argument("--skip-signal-count", action="store_true", help="Fast voxel-only stats")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--keep-tmp", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = build_parser()
    args = parser.parse_args(argv)
    payload = compute_region_scope_signal_stats(
        sample_dir=args.sample_dir,
        region_query=args.region,
        signal_ch=args.signal_ch or None,
        mask_zarr_path=args.mask_zarr or None,
        label_zarr_path=args.label_zarr or None,
        signal_zarr_path=args.signal_zarr or None,
        hemisphere_zarr_path=args.hemisphere_zarr or None,
        cfg_path=args.cfg,
        config_path=args.config or None,
        dataset_name=args.dataset_name,
        block_shape=_parse_block_shape(args.block_shape),
        foreground_mode=args.foreground_mode,
        foreground_label=int(args.foreground_label),
        min_voxels=int(args.min_voxels),
        pass1_workers=int(args.pass1_workers),
        resolution_xyz=_parse_resolution_xyz(args.resolution_xyz),
        include_signal_count=not bool(args.skip_signal_count),
        output_dir=args.output_dir or None,
        keep_tmp=bool(args.keep_tmp),
    )
    print(json.dumps(payload["summary"], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
