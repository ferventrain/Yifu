from __future__ import annotations

from pathlib import Path


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


def export_zarr_to_tiff(
    input_zarr,
    output_dir,
    *,
    dataset_name: str = "0",
    prefix: str = "mask_",
):
    np, tifffile, _, _ = _require_zarr_stack()
    data_in = open_zarr_dataset(input_zarr, dataset_name=dataset_name)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    for z_idx in range(int(data_in.shape[0])):
        slice_2d = np.asarray(data_in[z_idx])
        tifffile.imwrite(str(output_path / f"{prefix}{z_idx:04d}.tiff"), slice_2d)
    return output_path
