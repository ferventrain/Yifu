from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np

try:
    from pipeline_modules.utils.errors import ErrorCode, PipelineError
except ImportError:  # pragma: no cover
    from ..utils.errors import ErrorCode, PipelineError


DEFAULT_IMS_RESOLUTION_LEVEL = 2
DEFAULT_IMS_CHANNEL = 0
DEFAULT_IMS_TIMEPOINT = 0
DEFAULT_IMS_MAX_SLICES = 8
DEFAULT_IMS_HISTOGRAM_Z_CHUNKS = 24
DEFAULT_IMS_Z_CHUNK = 32
DEFAULT_IMS_HDF_CACHE_MB = 1024


@dataclass(frozen=True)
class ImsDatasetInfo:
    path: str
    resolution_level: int
    channel: int
    timepoint: int
    shape_zyx: tuple[int, int, int]
    chunks_zyx: tuple[int, int, int]
    dtype: str
    compression: str | None


def _require_h5py():
    try:
        import h5py
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise PipelineError(
            ErrorCode.DEPENDENCY_MISSING,
            "h5py is required for IMS QC",
            {"dependency": "h5py"},
        ) from exc
    return h5py


def _channel_key(timepoint_group: Any, channel: int) -> str:
    for key in timepoint_group.keys():
        if not str(key).startswith("Channel"):
            continue
        parts = str(key).split(" ")
        if len(parts) > 1 and int(parts[1]) == int(channel):
            return str(key)
    raise PipelineError(
        ErrorCode.INPUT_FORMAT_INVALID,
        "IMS channel not found",
        {"channel": channel, "available": list(timepoint_group.keys())},
    )


def resolve_ims_dataset(
    file_handle: Any,
    *,
    resolution_level: int,
    channel: int,
    timepoint: int,
) -> tuple[Any, ImsDatasetInfo]:
    if "DataSet" not in file_handle:
        raise PipelineError(ErrorCode.INPUT_FORMAT_INVALID, "Invalid IMS file: missing DataSet group")

    resolution_key = f"ResolutionLevel {int(resolution_level)}"
    dataset_root = file_handle["DataSet"]
    if resolution_key not in dataset_root:
        available = [str(key) for key in dataset_root.keys() if str(key).startswith("ResolutionLevel")]
        raise PipelineError(
            ErrorCode.INPUT_FORMAT_INVALID,
            "IMS resolution level not found",
            {"resolution_level": resolution_level, "available": available},
        )

    timepoint_key = f"TimePoint {int(timepoint)}"
    resolution_group = dataset_root[resolution_key]
    if timepoint_key not in resolution_group:
        raise PipelineError(
            ErrorCode.INPUT_FORMAT_INVALID,
            "IMS timepoint not found",
            {"timepoint": timepoint, "available": list(resolution_group.keys())},
        )

    tp_group = resolution_group[timepoint_key]
    channel_key = _channel_key(tp_group, channel)
    if "Data" not in tp_group[channel_key]:
        raise PipelineError(
            ErrorCode.INPUT_FORMAT_INVALID,
            "IMS channel Data dataset not found",
            {"channel_key": channel_key},
        )

    dataset = tp_group[channel_key]["Data"]
    if len(dataset.shape) != 3:
        raise PipelineError(
            ErrorCode.INPUT_FORMAT_INVALID,
            "IMS Data must be 3D ZYX",
            {"shape": list(dataset.shape)},
        )

    chunks = tuple(int(v) for v in (dataset.chunks or (DEFAULT_IMS_Z_CHUNK, 128, 128)))
    info = ImsDatasetInfo(
        path=str(getattr(file_handle, "filename", "")),
        resolution_level=int(resolution_level),
        channel=int(channel),
        timepoint=int(timepoint),
        shape_zyx=tuple(int(v) for v in dataset.shape),
        chunks_zyx=chunks,
        dtype=str(dataset.dtype),
        compression=str(dataset.compression) if dataset.compression is not None else None,
    )
    return dataset, info


@contextmanager
def open_ims_dataset(
    path: str | Path,
    *,
    resolution_level: int = DEFAULT_IMS_RESOLUTION_LEVEL,
    channel: int = DEFAULT_IMS_CHANNEL,
    timepoint: int = DEFAULT_IMS_TIMEPOINT,
    hdf_cache_mb: int = DEFAULT_IMS_HDF_CACHE_MB,
):
    h5py = _require_h5py()
    ims_path = Path(path)
    if not ims_path.exists():
        raise PipelineError(ErrorCode.INPUT_NOT_FOUND, "IMS file not found", {"input_ims": str(ims_path)})

    cache_settings = {
        "rdcc_nbytes": max(int(hdf_cache_mb), 64) * 1024 * 1024,
        "rdcc_nslots": 52000,
    }
    with h5py.File(str(ims_path), "r", **cache_settings) as file_handle:
        dataset, info = resolve_ims_dataset(
            file_handle,
            resolution_level=resolution_level,
            channel=channel,
            timepoint=timepoint,
        )
        yield dataset, info


def effective_z_chunk(info: ImsDatasetInfo, requested_z_chunk: int | None = None) -> int:
    if requested_z_chunk is not None and int(requested_z_chunk) > 0:
        return int(requested_z_chunk)
    return max(int(info.chunks_zyx[0]), 1)


def count_z_chunks(shape_z: int, z_chunk: int) -> int:
    chunk = max(int(z_chunk), 1)
    return max((int(shape_z) + chunk - 1) // chunk, 1)


def select_histogram_z_chunk_ids(shape_z: int, z_chunk: int, max_z_chunks: int) -> list[int]:
    total_chunks = count_z_chunks(shape_z, z_chunk)
    count = min(max(int(max_z_chunks), 1), total_chunks)
    if count == 1:
        return [total_chunks // 2]
    return sorted({int(round(v)) for v in np.linspace(0, total_chunks - 1, count)})


def select_qc_z_indices(
    shape_z: int,
    max_slices: int,
    z_chunk: int,
    *,
    align_to_chunk: bool = True,
) -> list[int]:
    z_size = int(shape_z)
    chunk = max(int(z_chunk), 1)
    count = min(max(int(max_slices), 1), z_size)
    if count == 1:
        if align_to_chunk:
            chunk_id = (z_size // 2) // chunk
            return [min(chunk_id * chunk + chunk // 2, z_size - 1)]
        return [z_size // 2]

    total_chunks = count_z_chunks(z_size, chunk)
    if align_to_chunk:
        chunk_ids = np.linspace(0, total_chunks - 1, count)
        indices = {
            min(int(round(chunk_id)) * chunk + chunk // 2, z_size - 1)
            for chunk_id in chunk_ids
        }
        return sorted(indices)

    return sorted({int(round(v)) for v in np.linspace(0, z_size - 1, count)})


def _merge_z_ranges(ranges: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not ranges:
        return []
    ordered = sorted(ranges, key=lambda item: item[0])
    merged: list[tuple[int, int]] = [ordered[0]]
    for z0, z1 in ordered[1:]:
        prev_z0, prev_z1 = merged[-1]
        if z0 <= prev_z1:
            merged[-1] = (prev_z0, max(prev_z1, z1))
        else:
            merged.append((z0, z1))
    return merged


def group_z_indices_into_read_ranges(
    z_indices: list[int],
    *,
    shape_z: int,
    z_chunk: int,
) -> list[tuple[int, int]]:
    if not z_indices:
        return []
    ranges: list[tuple[int, int]] = []
    chunk = max(int(z_chunk), 1)
    for z_index in z_indices:
        z0 = (int(z_index) // chunk) * chunk
        z1 = min(z0 + chunk, int(shape_z))
        ranges.append((z0, z1))
    return _merge_z_ranges(ranges)


def iter_ims_histogram_blocks(
    dataset: Any,
    *,
    z_chunk: int,
    z_chunk_ids: list[int],
) -> Iterator[np.ndarray]:
    shape_z = int(dataset.shape[0])
    chunk = max(int(z_chunk), 1)
    for chunk_id in z_chunk_ids:
        z0 = int(chunk_id) * chunk
        if z0 >= shape_z:
            continue
        z1 = min(z0 + chunk, shape_z)
        yield np.asarray(dataset[z0:z1, :, :])


def read_slices_chunk_aligned(
    dataset: Any,
    z_indices: list[int],
    *,
    z_chunk: int,
) -> dict[int, np.ndarray]:
    if not z_indices:
        return {}
    shape_z = int(dataset.shape[0])
    slices: dict[int, np.ndarray] = {}
    for z0, z1 in group_z_indices_into_read_ranges(z_indices, shape_z=shape_z, z_chunk=z_chunk):
        block = np.asarray(dataset[z0:z1, :, :])
        for z_index in z_indices:
            if z0 <= int(z_index) < z1:
                slices[int(z_index)] = block[int(z_index) - z0]
    return slices


def stack_ordered_slices(slice_map: dict[int, np.ndarray], z_indices: list[int]) -> np.ndarray:
    if not z_indices:
        shape_yx = next(iter(slice_map.values())).shape if slice_map else (0, 0)
        return np.empty((0, *shape_yx))
    return np.stack([slice_map[int(z_index)] for z_index in z_indices], axis=0)


def ims_source_meta(info: ImsDatasetInfo, *, z_chunk: int, read_strategy: str) -> dict[str, Any]:
    return {
        "source_kind": "ims",
        "source_path": info.path,
        "shape_zyx": list(info.shape_zyx),
        "dtype": info.dtype,
        "chunks_zyx": list(info.chunks_zyx),
        "compression": info.compression,
        "resolution_level": info.resolution_level,
        "channel": info.channel,
        "timepoint": info.timepoint,
        "effective_z_chunk": int(z_chunk),
        "streaming": True,
        "read_strategy": read_strategy,
    }
