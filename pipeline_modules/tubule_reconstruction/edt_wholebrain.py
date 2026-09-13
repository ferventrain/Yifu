"""Whole-brain EDT+thinning reconstruction from a binary mask Zarr."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import logging
import os
import pickle
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from pipeline_modules.tubule_reconstruction.config import default_chunk_workers
from pipeline_modules.tubule_reconstruction.edt_reconstruction import (
    graph_to_polylines,
    process_chunk,
    stitch_face_endpoints,
)
from pipeline_modules.tubule_reconstruction.kimimaro_reconstruction import (
    list_existing_chunk_indices,
    open_zarr_dataset,
    parse_resolution_xyz,
    parse_triplet_int,
    resolution_xyz_to_zyx,
)

logger = logging.getLogger(__name__)
SPILL_DIRNAME = "_chunk_spill"
PROGRESS_NAME = "_progress.json"


def _now_iso() -> str:
    return datetime.now().astimezone().replace(microsecond=0).isoformat()


def _spill_key(chunk_index) -> str:
    z, y, x = (int(v) for v in chunk_index)
    return f"z{z:05d}_y{y:05d}_x{x:05d}"


def _write_progress(output_root: Path, payload: dict) -> None:
    path = output_root / PROGRESS_NAME
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _compact_result(result: dict) -> dict:
    slices = result["core_slices"]
    return {
        "chunk_index": tuple(int(v) for v in result["chunk_index"]),
        "core_start_zyx": [int(s.start) for s in slices],
        "core_stop_zyx": [int(s.stop) for s in slices],
        "core_shape": tuple(int(v) for v in result["core_shape"]),
        "mask_voxels": int(result["mask_voxels"]),
        "num_skeleton_voxels": int(result["num_skeleton_voxels"]),
        "num_edges": int(result["num_edges"]),
        "local_coords": result["local_coords"],
        "edges": result["edges"],
        "degrees": result["degrees"],
        "global_coords_um": result["global_coords_um"],
        "radii_um": result["radii_um"],
        "face_endpoints_um": result.get("face_endpoints_um") or {},
    }


def _edt_chunk_worker(task: dict) -> dict:
    spill_path = Path(task["spill_path"])
    if spill_path.is_file():
        with spill_path.open("rb") as handle:
            payload = pickle.load(handle)
        return {
            "chunk_index": tuple(payload["chunk_index"]),
            "mask_voxels": int(payload.get("mask_voxels") or 0),
            "num_skeleton_voxels": int(payload.get("num_skeleton_voxels") or 0),
            "num_edges": int(payload.get("num_edges") or 0),
            "resumed": True,
            "spill_path": str(spill_path),
        }

    mask_zarr = open_zarr_dataset(task["mask_zarr_path"], dataset_name=task["dataset_name"])
    result = process_chunk(
        mask_zarr,
        task["chunk_index"],
        resolution_xyz=task["resolution_xyz"],
        foreground_label=task["foreground_label"],
        dust_threshold=task["dust_threshold"],
        halo_zyx=tuple(task["halo_zyx"]),
        build_edge_table=False,
    )
    payload = _compact_result(result)
    spill_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = spill_path.with_suffix(".pkl.tmp")
    with tmp.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, spill_path)
    return {
        "chunk_index": payload["chunk_index"],
        "mask_voxels": payload["mask_voxels"],
        "num_skeleton_voxels": payload["num_skeleton_voxels"],
        "num_edges": payload["num_edges"],
        "resumed": False,
        "spill_path": str(spill_path),
    }


def _load_spill(path: Path) -> dict:
    with path.open("rb") as handle:
        return pickle.load(handle)


def _branch_rows_from_spill(payload: dict, resolution_xyz) -> list[dict]:
    coords = payload["local_coords"]
    edges = payload["edges"]
    degrees = payload["degrees"]
    if len(coords) == 0:
        return []
    scale_zyx = np.array(resolution_xyz_to_zyx(resolution_xyz), dtype=np.float64)
    origin = np.array(payload["core_start_zyx"], dtype=np.float64)
    radii = payload["radii_um"]
    chunk_text = ".".join(str(v) for v in payload["chunk_index"])
    index_of = {tuple(int(v) for v in row): i for i, row in enumerate(coords)}
    rows = []
    for branch_id, path in enumerate(graph_to_polylines(coords, edges, degrees)):
        if len(path) < 2:
            continue
        um = path.astype(np.float64) * scale_zyx + origin * scale_zyx
        seg = np.linalg.norm(np.diff(um, axis=0), axis=1)
        src, dst = um[0], um[-1]
        radius_vals = []
        for zyx in np.rint(path).astype(np.int32):
            i = index_of.get((int(zyx[0]), int(zyx[1]), int(zyx[2])))
            if i is not None and i < len(radii):
                radius_vals.append(float(radii[i]))
        rows.append(
            {
                "chunk_index": chunk_text,
                "branch_id": int(branch_id),
                "source_z_um": float(src[0]),
                "source_y_um": float(src[1]),
                "source_x_um": float(src[2]),
                "target_z_um": float(dst[0]),
                "target_y_um": float(dst[1]),
                "target_x_um": float(dst[2]),
                "length_um": float(seg.sum()) if len(seg) else 0.0,
                "num_points": int(len(path)),
                "mean_radius_um": float(np.mean(radius_vals)) if radius_vals else np.nan,
                "is_stitch": False,
            }
        )
    return rows


def run_wholebrain(
    mask_zarr_path,
    output_dir,
    *,
    dataset_name="0",
    resolution_xyz=(1.8, 1.8, 2.0),
    foreground_label=1,
    dust_threshold=100,
    halo_zyx=(2, 4, 4),
    stitch_max_distance_um=5.0,
    chunk_workers=None,
) -> dict:
    started = time.time()
    resolution_xyz = parse_resolution_xyz(resolution_xyz)
    halo_zyx = parse_triplet_int(halo_zyx)
    if chunk_workers is None:
        chunk_workers = default_chunk_workers()
    chunk_workers = max(1, min(8, int(chunk_workers)))
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    spill_dir = output_root / SPILL_DIRNAME
    spill_dir.mkdir(parents=True, exist_ok=True)

    mask_zarr = open_zarr_dataset(mask_zarr_path, dataset_name=dataset_name)
    chunk_indices = list_existing_chunk_indices(mask_zarr)
    if not chunk_indices:
        raise ValueError(f"No physical chunks in {mask_zarr_path}")
    logger.info("Whole-brain EDT: %d chunks, workers=%d, no downsample", len(chunk_indices), chunk_workers)

    tasks = []
    for chunk_index in chunk_indices:
        spill_path = spill_dir / f"{_spill_key(chunk_index)}.pkl"
        tasks.append(
            {
                "mask_zarr_path": str(mask_zarr_path),
                "dataset_name": dataset_name,
                "chunk_index": tuple(int(v) for v in chunk_index),
                "resolution_xyz": resolution_xyz,
                "foreground_label": foreground_label,
                "dust_threshold": dust_threshold,
                "halo_zyx": tuple(int(v) for v in halo_zyx),
                "spill_path": str(spill_path),
            }
        )

    phase_started = _now_iso()
    total = len(tasks)
    max_inflight = max(chunk_workers * 4, chunk_workers)

    def emit(done_count: int) -> None:
        _write_progress(
            output_root,
            {
                "phase": "Processing chunks",
                "unit_done": int(done_count),
                "unit_total": int(total),
                "phase_started_at": phase_started,
                "updated_at": _now_iso(),
                "chunk_workers": int(chunk_workers),
                "downsample_factor": 1,
            },
        )

    pending = [t for t in tasks if not Path(t["spill_path"]).is_file()]
    done = total - len(pending)
    logger.info("Resume: %d already spilled, %d remaining", done, len(pending))
    emit(done)

    if chunk_workers <= 1:
        for task in tqdm(pending, desc="EDT chunks"):
            _edt_chunk_worker(task)
            done += 1
            if done % 25 == 0 or done == total:
                emit(done)
    else:
        inflight: dict[concurrent.futures.Future, dict] = {}
        pending_tasks = list(pending)
        with concurrent.futures.ProcessPoolExecutor(max_workers=chunk_workers) as executor:
            with tqdm(total=len(pending), desc="EDT chunks") as bar:
                while pending_tasks or inflight:
                    while pending_tasks and len(inflight) < max_inflight:
                        task = pending_tasks.pop()
                        fut = executor.submit(_edt_chunk_worker, task)
                        inflight[fut] = task
                    finished, _ = concurrent.futures.wait(
                        inflight,
                        return_when=concurrent.futures.FIRST_COMPLETED,
                    )
                    for fut in finished:
                        inflight.pop(fut)
                        fut.result()
                        done += 1
                        bar.update(1)
                        if done % 25 == 0 or done == total:
                            emit(done)

    emit(total)
    logger.info("Chunk pass finished in %.1f min; stitching faces", (time.time() - started) / 60.0)

    chunk_rows = []
    endpoint_metas = []
    branch_csv = output_root / "skeleton_edges.csv"
    if branch_csv.exists():
        branch_csv.unlink()
    wrote_header = False
    total_skel = 0
    total_mask = 0
    for path in tqdm(sorted(spill_dir.glob("*.pkl")), desc="Writing edges / stitch inputs"):
        payload = _load_spill(path)
        chunk_rows.append(
            {
                "chunk_index": ".".join(str(v) for v in payload["chunk_index"]),
                "mask_voxels": int(payload.get("mask_voxels") or 0),
                "num_skeleton_voxels": int(payload.get("num_skeleton_voxels") or 0),
                "num_edges": int(payload.get("num_edges") or 0),
            }
        )
        total_skel += int(payload.get("num_skeleton_voxels") or 0)
        total_mask += int(payload.get("mask_voxels") or 0)
        endpoint_metas.append(
            {
                "chunk_index": tuple(payload["chunk_index"]),
                "face_endpoints_um": payload.get("face_endpoints_um") or {},
            }
        )
        rows = _branch_rows_from_spill(payload, resolution_xyz)
        if rows:
            frame = pd.DataFrame(rows)
            frame.to_csv(branch_csv, mode="a", header=not wrote_header, index=False)
            wrote_header = True

    stitch_table = stitch_face_endpoints(endpoint_metas, max_distance_um=stitch_max_distance_um)
    stitch_csv = output_root / "stitch_edges.csv"
    if not stitch_table.empty:
        stitch_table.to_csv(stitch_csv, index=False)
        extra = stitch_table.rename(columns={"chunk_a": "chunk_index"}).copy()
        extra["branch_id"] = -1
        extra["num_points"] = 2
        extra["mean_radius_um"] = np.nan
        keep = [
            "chunk_index",
            "branch_id",
            "source_z_um",
            "source_y_um",
            "source_x_um",
            "target_z_um",
            "target_y_um",
            "target_x_um",
            "length_um",
            "num_points",
            "mean_radius_um",
            "is_stitch",
        ]
        extra[keep].to_csv(branch_csv, mode="a", header=not wrote_header, index=False)

    pd.DataFrame(chunk_rows).to_csv(output_root / "vessel_chunk_metrics.csv", index=False)
    summary = {
        "mode": "edt_chunkwise",
        "downsample_factor": 1,
        "processed_chunks": int(len(chunk_indices)),
        "mask_voxels": int(total_mask),
        "num_skeleton_voxels": int(total_skel),
        "num_stitch_edges": int(len(stitch_table)),
        "chunk_workers": int(chunk_workers),
        "dust_threshold": int(dust_threshold),
        "halo_zyx": list(halo_zyx),
        "resolution_xyz_um": list(resolution_xyz),
        "stitch_max_distance_um": float(stitch_max_distance_um),
        "elapsed_min": round((time.time() - started) / 60.0, 2),
    }
    (output_root / "vessel_network_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    _write_progress(
        output_root,
        {
            "phase": "Done",
            "unit_done": int(total),
            "unit_total": int(total),
            "phase_started_at": phase_started,
            "updated_at": _now_iso(),
            "chunk_workers": int(chunk_workers),
            "downsample_factor": 1,
        },
    )
    logger.info("Whole-brain EDT done in %.1f min. Stitch edges=%d", summary["elapsed_min"], len(stitch_table))
    return summary


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Whole-brain EDT+thinning from a binary mask Zarr.")
    parser.add_argument("--mask_zarr", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--dataset_name", default="0")
    parser.add_argument("--resolution_xyz", default="1.8,1.8,2.0")
    parser.add_argument("--dust_threshold", type=int, default=100)
    parser.add_argument("--halo_zyx", default="2,4,4")
    parser.add_argument("--stitch_max_distance_um", type=float, default=5.0)
    parser.add_argument("--chunk_workers", type=int, default=default_chunk_workers())
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    run_wholebrain(
        args.mask_zarr,
        args.output_dir,
        dataset_name=args.dataset_name,
        resolution_xyz=args.resolution_xyz,
        dust_threshold=args.dust_threshold,
        halo_zyx=args.halo_zyx,
        stitch_max_distance_um=args.stitch_max_distance_um,
        chunk_workers=args.chunk_workers,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
