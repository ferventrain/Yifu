#!/usr/bin/env python3
"""Per-vertebra spinal cord signal statistics.

Consumes a vertebral-segment label volume (default: the SpinalJ wizard output
``spinalj_segments.zarr`` written into the sample directory), a threshold
segmentation mask, and the signal Zarr, then produces per-vertebra positive
signal count / volume / density tables through the block-graph region engine
(``region_signal_analysis_zarr_graph.analyze_zarr_graph``).

The mid-res (~10 um) wizard label Zarr is nearest-upsampled to the signal grid
automatically; a ``*_native.zarr`` label volume is used as-is when present.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline_modules.registration.region_signal_analysis_zarr_graph import (
    analyze_zarr_graph,
    parse_resolution_xyz,
)
from pipeline_modules.registration.spinalj_labels_to_native import (
    _segment_sort_key,
    upsample_midres_to_native_zarr,
)
from pipeline_modules.utils.deliverable_paths import (
    spinal_segment_region_csv,
    spinal_segment_stats_csv,
    spinal_segment_stats_xlsx,
)
from pipeline_modules.utils.run_manifest import write_run_manifest

try:
    from pipeline_modules.utils.zarr_io import open_group, open_zarr_array
except ImportError:  # pragma: no cover - direct-script fallback
    from ..utils.zarr_io import open_group, open_zarr_array

logger = logging.getLogger(__name__)

DEFAULT_SEGMENTS_ZARR = "spinalj_segments"
LEGEND_FILENAME = "segment_id_legend.csv"
# Sentinel id for the root row. Must be > 0 (the engine skips non-positive
# label ids) and must not collide with wizard segment ids, which are 1..N.
ROOT_REGION_ID = 999999
ROOT_REGION_NAME = "Spinal cord all segments"


def load_segment_legend_csv(legend_csv: Path) -> list[tuple[int, str]]:
    """Read a vertebral legend CSV into sorted ``(segment_id, name)`` pairs.

    Accepts either the wizard output format (``segment_id,name``) or a plain
    ``Segment`` column (ids are then assigned in C->T->L->S order).
    """
    legend_csv = Path(legend_csv)
    frame = pd.read_csv(legend_csv)
    columns = {str(col).strip().lower() for col in frame.columns}

    if "segment_id" in columns:
        id_col = next(col for col in frame.columns if str(col).strip().lower() == "segment_id")
        name_col = next(col for col in frame.columns if str(col).strip().lower() == "name")
        pairs = [(int(row[id_col]), str(row[name_col]).strip()) for _, row in frame.iterrows()]
    elif "segment" in columns:
        seg_col = next(col for col in frame.columns if str(col).strip().lower() == "segment")
        names = sorted({str(row[seg_col]).strip() for _, row in frame.iterrows()}, key=_segment_sort_key)
        pairs = [(index + 1, name) for index, name in enumerate(names)]
    else:
        raise ValueError(
            f"Legend CSV {legend_csv} must have a 'segment_id' + 'name' or a 'Segment' column; "
            f"found: {list(frame.columns)}"
        )

    if not pairs:
        raise ValueError(f"Legend CSV {legend_csv} contains no segments")
    pairs.sort(key=lambda pair: _segment_sort_key(pair[1]))
    return pairs


def write_region_csv(legend_pairs: list[tuple[int, str]], output_csv: Path) -> Path:
    """Convert ``(segment_id, name)`` pairs into the flat region CSV the engine loads."""
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "id": ROOT_REGION_ID,
            "name": ROOT_REGION_NAME,
            "acronym": ROOT_REGION_NAME,
            "structure_id_path": f"[{ROOT_REGION_ID}]",
            "graph_id": "",
            "rgb_triplet": "",
            "structure_set_ids": "",
        }
    ]
    for segment_id, name in legend_pairs:
        rows.append(
            {
                "id": segment_id,
                "name": name,
                "acronym": name,
                "structure_id_path": f"[{ROOT_REGION_ID}, {segment_id}]",
                "graph_id": "",
                "rgb_triplet": "",
                "structure_set_ids": "",
            }
        )
    pd.DataFrame(rows).to_csv(output_csv, index=False)
    logger.info("Wrote spinal segment region CSV: %s (%d segments)", output_csv, len(legend_pairs))
    return output_csv


def shape_as_tuple(shape) -> tuple:
    return tuple(int(v) for v in shape)


def read_zarr_attrs(zarr_path: Path) -> dict:
    group = open_group(zarr_path, mode="r")
    return dict(group.attrs)


def resolve_legend_csv(segments_zarr_path: Path, explicit_csv: str | Path | None, sample_dir: Path) -> Path:
    """Locate the vertebral legend: explicit path, zarr attrs, or wizard defaults."""
    if explicit_csv:
        candidate = Path(explicit_csv)
        if not candidate.is_absolute():
            candidate = sample_dir / candidate
        if not candidate.exists():
            raise FileNotFoundError(f"Segments legend CSV not found: {candidate}")
        return candidate

    attrs = read_zarr_attrs(segments_zarr_path)
    attrs_legend = str(attrs.get("legend_csv", "") or "").strip()
    if attrs_legend and Path(attrs_legend).exists():
        return Path(attrs_legend)

    candidates = [
        sample_dir / LEGEND_FILENAME,
        sample_dir / "_spinalj_wizard" / "labels_native" / LEGEND_FILENAME,
        segments_zarr_path.parent / LEGEND_FILENAME,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Vertebral segment legend not found. Pass --legend_csv explicitly (expected columns "
        "segment_id,name as written by spinalj_labels_to_native.py)."
    )


def resolve_segment_label_zarr(
    sample_dir: Path,
    segments_zarr: str | Path,
    signal_shape: tuple[int, int, int],
    dataset_name: str = "0",
) -> tuple[Path, bool]:
    """Return ``(label_zarr_path, upsampled)`` aligned to the signal grid.

    Candidates are tried in order: the configured path itself, its
    ``{stem}_native.zarr`` sibling, then its ``{stem}.zarr`` sibling. A volume
    whose shape already matches the signal grid wins immediately; a mid-res
    wizard volume whose ``native_shape_zyx`` attribute matches the signal grid
    is kept as an upsample fallback.
    """
    sample_dir = Path(sample_dir)
    configured = Path(str(segments_zarr).strip() or DEFAULT_SEGMENTS_ZARR)
    signal_shape = shape_as_tuple(signal_shape)
    if configured.is_absolute():
        candidates = [configured]
    else:
        stem = configured.stem
        candidates = [
            sample_dir / configured,
            sample_dir / f"{stem}_native.zarr",
            sample_dir / f"{stem}.zarr",
        ]

    upsample_fallback = None
    checked: list[str] = []
    for candidate in candidates:
        if not candidate.exists():
            continue
        checked.append(str(candidate))
        array = open_zarr_array(candidate, dataset_name=dataset_name)
        if shape_as_tuple(array.shape) == signal_shape:
            return candidate, False
        attrs = read_zarr_attrs(candidate)
        native_shape = shape_as_tuple(attrs.get("native_shape_zyx", ()) or ())
        if native_shape == signal_shape and upsample_fallback is None:
            upsample_fallback = candidate
    if upsample_fallback is not None:
        return upsample_fallback, True

    if checked:
        raise ValueError(
            "Segments label Zarr shape does not match the signal volume and cannot be upsampled. "
            f"Checked: {', '.join(checked)}; signal shape: {signal_shape}. Re-run the SpinalJ label "
            "back-projection (spinalj_labels_to_native.py) for this sample."
        )
    searched = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(
        f"Vertebral segment label Zarr not found (searched: {searched}). Run the SpinalJ wizard "
        "(spinalj_wizard_ui.py + spinalj_labels_to_native.py --write_zarr) first, or point "
        "spinal_cord.segments_zarr at an existing label volume."
    )


def upsample_segments_to_signal_grid(
    midres_zarr_path: Path,
    signal_zarr_path: Path,
    output_zarr_path: Path,
    resolution_xyz: tuple[float, float, float],
    dataset_name: str = "0",
) -> Path:
    """Nearest-upsample the mid-res wizard segment labels onto the signal grid."""
    mid_array = open_zarr_array(midres_zarr_path, dataset_name=dataset_name)
    signal_array = open_zarr_array(signal_zarr_path, dataset_name=dataset_name)
    attrs = read_zarr_attrs(midres_zarr_path)
    native_shape = shape_as_tuple(attrs.get("native_shape_zyx") or signal_array.shape)
    z_step = int(attrs.get("z_step") or 1)
    if native_shape != shape_as_tuple(signal_array.shape):
        raise ValueError(
            f"Cannot upsample segments {midres_zarr_path}: native_shape_zyx {native_shape} != "
            f"signal shape {shape_as_tuple(signal_array.shape)}"
        )
    mid = np.asarray(mid_array[:])
    spacing_zyx_um = (resolution_xyz[2], resolution_xyz[1], resolution_xyz[0])
    logger.info(
        "Upsampling mid-res segments %s %s -> %s (z_step=%d)",
        midres_zarr_path.name, mid.shape, native_shape, z_step,
    )
    upsample_midres_to_native_zarr(
        mid,
        Path(output_zarr_path),
        native_shape_zyx=native_shape,
        z_step=z_step,
        chunks=(1, 2048, 2048),
        spacing_zyx_um=spacing_zyx_um,
        extra_attrs={
            "label_kind": "vertebral_segments",
            "legend_csv": str(attrs.get("legend_csv", "") or ""),
            "upsampled_for_analysis": True,
        },
    )
    return Path(output_zarr_path)


def export_flat_csv(output_excel: Path, output_csv: Path) -> Path:
    """Flatten the per-level Excel sheets into one CSV with a Level column."""
    sheets = pd.read_excel(output_excel, sheet_name=None)
    frames = []
    for sheet_name, frame in sheets.items():
        if not isinstance(sheet_name, str) or not sheet_name.startswith("Level_"):
            continue
        level = sheet_name[len("Level_"):]
        frame = frame.copy()
        frame.insert(0, "Level", int(level) if str(level).isdigit() else level)
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    combined.to_csv(output_csv, index=False)
    return output_csv


def run_spinal_segment_analysis(
    *,
    sample_dir: str | Path,
    signal_ch: str | int,
    signal_zarr_path: str | Path,
    mask_zarr_path: str | Path,
    resolution_xyz: tuple[float, float, float],
    segments_zarr: str | Path = DEFAULT_SEGMENTS_ZARR,
    legend_csv: str | Path | None = None,
    min_voxels: int = 300,
    max_voxels: int = 5000,
    dataset_name: str = "0",
    pass1_workers: int = 1,
    output_excel: str | Path | None = None,
    output_csv: str | Path | None = None,
    region_csv: str | Path | None = None,
) -> dict:
    """Run the per-vertebra signal analysis and return a summary dict."""
    started_at = time.time()
    sample_dir = Path(sample_dir)
    channel = f"ch{signal_ch}" if not str(signal_ch).startswith("ch") else str(signal_ch)

    signal_array = open_zarr_array(signal_zarr_path, dataset_name=dataset_name)
    signal_shape = shape_as_tuple(signal_array.shape)

    segments_zarr_path, needs_upsample = resolve_segment_label_zarr(
        sample_dir, segments_zarr, signal_shape, dataset_name=dataset_name
    )
    if needs_upsample:
        upsampled_path = sample_dir / f"{segments_zarr_path.stem}_upsampled.zarr"
        segments_zarr_path = upsample_segments_to_signal_grid(
            segments_zarr_path,
            signal_zarr_path,
            upsampled_path,
            resolution_xyz,
            dataset_name=dataset_name,
        )

    legend_path = resolve_legend_csv(segments_zarr_path, legend_csv, sample_dir)
    legend_pairs = load_segment_legend_csv(legend_path)

    output_excel = Path(output_excel) if output_excel else spinal_segment_stats_xlsx(sample_dir, channel)
    output_csv = Path(output_csv) if output_csv else spinal_segment_stats_csv(sample_dir, channel)
    region_csv = Path(region_csv) if region_csv else spinal_segment_region_csv(sample_dir)
    output_excel.parent.mkdir(parents=True, exist_ok=True)

    write_region_csv(legend_pairs, region_csv)

    logger.info(
        "Analyzing per-vertebra signal stats: segments=%d min_voxels=%d max_voxels=%d resolution=%s",
        len(legend_pairs), int(min_voxels), int(max_voxels), resolution_xyz,
    )
    analyze_zarr_graph(
        mask_zarr_path=str(mask_zarr_path),
        label_zarr_path=str(segments_zarr_path),
        signal_zarr_path=str(signal_zarr_path),
        cfg_path=str(region_csv),
        output_path=str(output_excel),
        dataset_name=dataset_name,
        block_size=None,
        foreground_mode="equal",
        foreground_label=1,
        min_voxels=int(min_voxels),
        flush_every=0,
        resolution_xyz=resolution_xyz,
        tmp_dir="",
        keep_tmp=False,
        pass1_workers=int(pass1_workers),
        max_voxels=int(max_voxels),
        report_physical_volume=True,
    )
    export_flat_csv(output_excel, output_csv)

    summary = {
        "output_excel": str(output_excel),
        "output_csv": str(output_csv),
        "region_csv": str(region_csv),
        "segments_zarr": str(segments_zarr_path),
        "legend_csv": str(legend_path),
        "segment_count": len(legend_pairs),
        "min_voxels": int(min_voxels),
        "max_voxels": int(max_voxels),
        "resolution_xyz": list(resolution_xyz),
    }
    write_run_manifest(
        output_excel.parent,
        module="registration.spinal_segment_signal_stats",
        entrypoint="run_spinal_segment_analysis",
        inputs={
            "sample_dir": str(sample_dir),
            "signal_channel": channel,
            "signal_zarr_path": str(signal_zarr_path),
            "mask_zarr_path": str(mask_zarr_path),
            "segments_zarr_path": str(segments_zarr_path),
            "legend_csv": str(legend_path),
            "region_csv": str(region_csv),
            "min_voxels": int(min_voxels),
            "max_voxels": int(max_voxels),
            "resolution_xyz": list(resolution_xyz),
            "dataset_name": dataset_name,
        },
        outputs=[output_excel, output_csv, region_csv],
        started_at=started_at,
        extra=summary,
    )
    logger.info("Per-vertebra signal stats written: %s", output_excel)
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample_dir", required=True, help="Sample directory")
    parser.add_argument("--signal_ch", required=True, help="Signal channel number, e.g. 0")
    parser.add_argument("--signal_zarr", required=True, help="Signal Zarr path")
    parser.add_argument("--mask_zarr", required=True, help="Threshold segmentation mask Zarr path")
    parser.add_argument(
        "--segments_zarr",
        default=DEFAULT_SEGMENTS_ZARR,
        help="Vertebral segment label Zarr (name under sample_dir or absolute path)",
    )
    parser.add_argument("--legend_csv", default="", help="Segment legend CSV (default: zarr attrs / wizard output)")
    parser.add_argument("--min_voxels", type=int, default=300, help="Minimum object size in voxels")
    parser.add_argument("--max_voxels", type=int, default=5000, help="Maximum object size in voxels (0 disables)")
    parser.add_argument("--resolution_xyz", default="1,1,1", help="Voxel size in microns as x,y,z")
    parser.add_argument("--dataset_name", default="0")
    parser.add_argument("--pass1_workers", type=int, default=1)
    parser.add_argument("--output_excel", default="", help="Default: results/<sample>_ch<ch>_spinal_segment_stats.xlsx")
    parser.add_argument("--output_csv", default="", help="Default: same stem as the Excel with .csv")
    parser.add_argument("--region_csv", default="", help="Default: results/<sample>_spinal_segment_regions.csv")
    parser.add_argument("--json_logs", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.json_logs:
        import sys

        class _JsonFormatter(logging.Formatter):
            def format(self, record):
                return json.dumps(
                    {"level": record.levelname, "logger": record.name, "message": record.getMessage()}
                )

        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(_JsonFormatter())
        logging.root.addHandler(handler)
        logging.root.setLevel(logging.INFO)
    else:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    summary = run_spinal_segment_analysis(
        sample_dir=args.sample_dir,
        signal_ch=args.signal_ch,
        signal_zarr_path=args.signal_zarr,
        mask_zarr_path=args.mask_zarr,
        resolution_xyz=parse_resolution_xyz(args.resolution_xyz),
        segments_zarr=args.segments_zarr,
        legend_csv=args.legend_csv or None,
        min_voxels=args.min_voxels,
        max_voxels=args.max_voxels,
        dataset_name=args.dataset_name,
        pass1_workers=args.pass1_workers,
        output_excel=args.output_excel or None,
        output_csv=args.output_csv or None,
        region_csv=args.region_csv or None,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
