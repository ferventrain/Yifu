"""Zarr v2/v3 compatible open/create helpers.

The Python library may be zarr 2 or zarr 3. Pipeline outputs keep the on-disk
Zarr v2 / OME-NGFF 0.4 layout so existing samples stay readable. zarr 3 can
open those stores, which is what ``napari-ome-zarr`` 0.10 needs.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import numpy as np


def _zarr():
    try:
        import zarr
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ModuleNotFoundError("zarr is required") from exc
    return zarr


def is_zarr_array(obj: Any) -> bool:
    zarr = _zarr()
    return isinstance(obj, zarr.Array)


def list_array_keys(group: Any) -> list[str]:
    if hasattr(group, "array_keys"):
        try:
            return [str(key) for key in group.array_keys()]
        except Exception:
            pass
    zarr = _zarr()
    keys: list[str] = []
    try:
        names = list(group.keys())
    except Exception:
        return keys
    for key in names:
        try:
            member = group[key]
        except Exception:
            continue
        if isinstance(member, zarr.Array):
            keys.append(str(key))
    return keys


def open_store(path: str | Path, mode: str = "r") -> Any:
    zarr = _zarr()
    return zarr.open(str(path), mode=mode)


def open_group(path: str | Path, mode: str = "r") -> Any:
    zarr = _zarr()
    return zarr.open_group(str(path), mode=mode)


def _group_member(group: Any, name: str) -> Any | None:
    try:
        return group[name]
    except (KeyError, FileNotFoundError, TypeError):
        return None


def open_zarr_array(path_like: str | Path, dataset_name: str = "0") -> Any:
    """Open a Zarr group or array and return the image array (default dataset ``0``)."""
    zarr = _zarr()
    path = Path(path_like)
    if not path.exists():
        raise FileNotFoundError(f"Zarr path not found: {path}")
    root = zarr.open(str(path), mode="r")
    if isinstance(root, zarr.Array):
        return root
    dataset = _group_member(root, dataset_name)
    if dataset is not None and isinstance(dataset, zarr.Array):
        return dataset
    keys = list_array_keys(root)
    if len(keys) == 1:
        return root[keys[0]]
    raise ValueError(
        f"Could not resolve a Zarr array from {path}. "
        f"Available arrays: {keys}, requested dataset_name={dataset_name}"
    )


def resolve_compressor(compressor: Any = "default") -> Any | None:
    if compressor is None or compressor == "none":
        return None
    if compressor == "default":
        from numcodecs import Blosc

        return Blosc(cname="zstd", clevel=5, shuffle=Blosc.SHUFFLE)
    if compressor == "fast":
        from numcodecs import Blosc

        return Blosc(cname="lz4", clevel=1, shuffle=Blosc.SHUFFLE)
    return compressor


def array_compressor(array: Any) -> Any | None:
    value = getattr(array, "compressor", None)
    if value is not None:
        return value
    compressors = getattr(array, "compressors", None)
    if not compressors:
        return None
    if isinstance(compressors, (list, tuple)):
        return compressors[0] if compressors else None
    return compressors


def open_output_group(path: str | Path, *, overwrite: bool = True) -> Any:
    """Create a Zarr group, writing Zarr v2 on-disk layout when using zarr>=3."""
    zarr = _zarr()
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if overwrite and output.exists():
        if output.is_dir():
            shutil.rmtree(output)
        else:
            output.unlink()
    mode = "w" if overwrite else "w-"
    try:
        return zarr.open_group(str(output), mode=mode, zarr_format=2)
    except TypeError:
        if hasattr(zarr, "DirectoryStore"):
            store = zarr.DirectoryStore(str(output))
            return zarr.group(store=store, overwrite=overwrite)
        return zarr.open_group(str(output), mode=mode)


def create_array(
    root: Any,
    dataset_name: str,
    *,
    shape: tuple[int, ...],
    chunks: tuple[int, ...],
    dtype: Any,
    compressor: Any = "default",
    data: np.ndarray | None = None,
) -> Any:
    if isinstance(compressor, str) or compressor is None:
        resolved = resolve_compressor("none" if compressor is None else compressor)
    else:
        resolved = compressor
    if hasattr(root, "create_array"):
        try:
            arr = root.create_array(
                dataset_name,
                shape=shape,
                chunks=chunks,
                dtype=dtype,
                compressors=resolved,
            )
        except TypeError:
            arr = root.create_array(
                dataset_name,
                shape=shape,
                chunks=chunks,
                dtype=dtype,
                compressor=resolved,
            )
        if data is not None:
            arr[:] = data
        return arr
    kwargs: dict[str, Any] = {
        "shape": shape,
        "chunks": chunks,
        "dtype": dtype,
        "compressor": resolved,
    }
    if data is not None:
        kwargs["data"] = data
    return root.create_dataset(dataset_name, **kwargs)


def ome_ngff_multiscales(
    datasets: list[str],
    *,
    ndim: int,
    base_scale: tuple[float, ...] | None = None,
    name: str = "volume",
) -> list[dict[str, Any]]:
    """OME-NGFF 0.4 multiscales with axes + scale so napari-ome-zarr can open the store.

    Missing axes makes napari-ome-zarr assume 5D (t,c,z,y,x) and crash with
    ``NoneType has no attribute affine_matrix``.
    """
    if ndim == 2:
        axes = [{"name": "y", "type": "space"}, {"name": "x", "type": "space"}]
    elif ndim == 3:
        axes = [
            {"name": "z", "type": "space"},
            {"name": "y", "type": "space"},
            {"name": "x", "type": "space"},
        ]
    elif ndim == 4:
        axes = [
            {"name": "c", "type": "channel"},
            {"name": "z", "type": "space"},
            {"name": "y", "type": "space"},
            {"name": "x", "type": "space"},
        ]
    else:
        axes = [
            {"name": "t", "type": "time"},
            {"name": "c", "type": "channel"},
            {"name": "z", "type": "space"},
            {"name": "y", "type": "space"},
            {"name": "x", "type": "space"},
        ][:ndim]
    scale = list(base_scale) if base_scale is not None else [1.0] * ndim
    if len(scale) != ndim:
        raise ValueError(f"base_scale length {len(scale)} does not match ndim={ndim}")
    out = []
    spatial = [i for i, axis in enumerate(axes) if axis.get("type") == "space"]
    for level, path in enumerate(datasets):
        factor = float(2**level)
        level_scale = [float(s) * factor if i in spatial else float(s) for i, s in enumerate(scale)]
        out.append(
            {
                "path": str(path),
                "coordinateTransformations": [{"type": "scale", "scale": level_scale}],
            }
        )
    return [
        {
            "version": "0.4",
            "name": name,
            "axes": axes,
            "datasets": out,
        }
    ]


def create_output_zarr(
    output_zarr: str | Path,
    shape: tuple[int, ...],
    chunks: tuple[int, ...],
    dtype: Any,
    *,
    dataset_name: str = "0",
    compressor: Any = "default",
    data: np.ndarray | None = None,
) -> tuple[Any, Any]:
    root = open_output_group(output_zarr, overwrite=True)
    dataset = create_array(
        root,
        dataset_name,
        shape=shape,
        chunks=chunks,
        dtype=dtype,
        compressor=compressor,
        data=data,
    )
    root.attrs["multiscales"] = ome_ngff_multiscales(
        [dataset_name],
        ndim=len(shape),
    )
    return root, dataset


def write_array(
    output_zarr: str | Path,
    data: np.ndarray,
    *,
    chunks: tuple[int, ...] | None = None,
    dataset_name: str = "0",
    compressor: Any = "default",
) -> Any:
    array = np.asarray(data)
    resolved_chunks = tuple(chunks or array.shape)
    _, dataset = create_output_zarr(
        output_zarr,
        tuple(array.shape),
        resolved_chunks,
        array.dtype,
        dataset_name=dataset_name,
        compressor=compressor,
        data=array,
    )
    return dataset


def list_existing_chunk_indices(array: Any) -> list[tuple[int, ...]]:
    """Return chunk indices that already exist on disk (v2 directory stores)."""
    ndim = len(array.shape)
    dim_sep = getattr(array, "_dimension_separator", None) or "."
    existing: set[tuple[int, ...]] = set()

    store = getattr(array, "store", None)
    array_path = str(getattr(array, "path", "") or "")
    prefix = f"{array_path}/" if array_path else ""
    keys: list[str] = []
    if store is not None and hasattr(store, "keys"):
        try:
            keys = [str(k) for k in store.keys()]
        except Exception:
            keys = []
    if not keys:
        fs_path = _array_filesystem_dir(array)
        if fs_path is not None:
            for entry in fs_path.rglob("*"):
                if entry.is_file() and not entry.name.startswith("."):
                    rel = str(entry.relative_to(fs_path)).replace("\\", "/")
                    keys.append(rel)

    for key in keys:
        rel = key[len(prefix) :] if prefix and key.startswith(prefix) else key
        if rel in {".zarray", ".zattrs", ".zgroup", "zarr.json"} or rel.startswith("."):
            continue
        parts = rel.split(dim_sep)
        if len(parts) != ndim:
            continue
        try:
            existing.add(tuple(int(part) for part in parts))
        except ValueError:
            continue
    return sorted(existing)


def _array_filesystem_dir(array: Any) -> Path | None:
    store = getattr(array, "store", None)
    path = getattr(array, "path", "") or ""
    for attr in ("path", "root"):
        value = getattr(store, attr, None)
        if isinstance(value, (str, Path)) and Path(value).exists():
            base = Path(value)
            return (base / path) if path else base
    return None
