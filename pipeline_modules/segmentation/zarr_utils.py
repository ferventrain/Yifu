from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from pipeline_modules.utils.tiff_stack_io import (
    iter_batch_ranges,
    normalize_tiff_compression,
    resolve_slice_batch,
    resolve_stack_workers,
    run_bounded_batches,
    write_tiff_stack_batch,
)

_DATASET_CACHE: dict[str, Any] = {}


def _require_zarr_stack():
    try:
        import numpy as np
        import tifffile
        import zarr
        from numcodecs import Blosc
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ModuleNotFoundError(
            "numpy, tifffile, zarr, and numcodecs are required for segmentation Zarr I/O"
        ) from exc
    return np, tifffile, zarr, Blosc


def open_zarr_dataset(path_like, dataset_name: str = "0"):
    _, _, zarr, _ = _require_zarr_stack()
    path = Path(path_like)
    if not path.exists():
        raise FileNotFoundError(f"Zarr path not found: {path}")

    root = zarr.open(str(path), mode="r")
    if isinstance(root, zarr.Array):
        return root
    if dataset_name in root and isinstance(root[dataset_name], zarr.Array):
        return root[dataset_name]

    array_keys = list(root.array_keys())
    if len(array_keys) == 1:
        return root[array_keys[0]]

    raise ValueError(
        f"Could not resolve a Zarr array from {path}. "
        f"Available arrays: {array_keys}, requested dataset_name={dataset_name}"
    )


def create_output_zarr(output_zarr, shape, chunks, dtype, *, dataset_name: str = "0", compressor="default"):
    _, _, zarr, Blosc = _require_zarr_stack()
    output_path = Path(output_zarr)
    store_out = zarr.DirectoryStore(str(output_path))
    root_out = zarr.group(store=store_out, overwrite=True)
    if compressor == "default":
        compressor = Blosc(cname="zstd", clevel=5, shuffle=Blosc.SHUFFLE)
    data_out = root_out.create_dataset(dataset_name, shape=shape, chunks=chunks, dtype=dtype, compressor=compressor)
    root_out.attrs["multiscales"] = [{
        "version": "0.4",
        "datasets": [{"path": dataset_name}],
    }]
    return root_out, data_out


def list_existing_chunk_indices(data_in):
    store = data_in.store
    array_path = getattr(data_in, "path", "")
    dim_sep = getattr(data_in, "_dimension_separator", ".")
    ndim = len(data_in.shape)

    prefix = f"{array_path}/" if array_path else ""
    existing = set()
    for raw_key in store.keys():
        key = str(raw_key)
        if prefix and not key.startswith(prefix):
            continue
        rel = key[len(prefix):] if prefix else key
        if rel in {".zarray", ".zattrs", ".zgroup", "zarr.json"}:
            continue
        if rel.startswith("."):
            continue
        parts = rel.split(dim_sep)
        if len(parts) != ndim:
            continue
        try:
            idx = tuple(int(part) for part in parts)
        except ValueError:
            continue
        existing.add(idx)
    return sorted(existing)


def _get_cached_dataset(input_zarr: str, dataset_name: str):
    cache_key = f"{input_zarr}:{dataset_name}"
    if cache_key not in _DATASET_CACHE:
        _DATASET_CACHE[cache_key] = open_zarr_dataset(input_zarr, dataset_name=dataset_name)
    return _DATASET_CACHE[cache_key]


def _export_zarr_batch_job(job: dict[str, object]) -> int:
    data_in = _get_cached_dataset(str(job["input_zarr"]), str(job["dataset_name"]))
    start = int(job["start"])
    end = int(job["end"])
    batch = np.asarray(data_in[start:end])
    write_tiff_stack_batch(
        batch,
        job["output_paths"],  # type: ignore[arg-type]
        compression=job.get("compression"),  # type: ignore[arg-type]
    )
    return end - start


def export_zarr_to_tiff(
    input_zarr,
    output_dir,
    *,
    dataset_name: str = "0",
    prefix: str = "mask_",
    slice_names: list[str] | None = None,
    workers: int = 0,
    slice_batch: int | None = None,
    compression: str | None = None,
) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    data_in = open_zarr_dataset(input_zarr, dataset_name=dataset_name)
    depth = int(data_in.shape[0])
    if slice_names is not None and len(slice_names) != depth:
        raise ValueError(
            f"slice_names length ({len(slice_names)}) must match Zarr depth ({depth})"
        )
    read_batch = resolve_slice_batch(slice_batch, default=16)
    worker_count = resolve_stack_workers(workers)
    comp = normalize_tiff_compression(compression)

    jobs: list[dict[str, object]] = []
    for start, end in iter_batch_ranges(depth, read_batch):
        if slice_names is None:
            output_paths = [output_path / f"{prefix}{z_idx:04d}.tiff" for z_idx in range(start, end)]
        else:
            output_paths = [output_path / slice_names[z_idx] for z_idx in range(start, end)]
        jobs.append(
            {
                "input_zarr": str(Path(input_zarr)),
                "dataset_name": dataset_name,
                "start": start,
                "end": end,
                "output_paths": output_paths,
                "compression": comp,
            }
        )

    run_bounded_batches(
        jobs,
        _export_zarr_batch_job,
        worker_count=worker_count,
        progress_total=depth,
        desc="Export Zarr to TIFF",
        unit="slice",
    )
    return output_path
