"""Write a masked Zarr volume back into a fresh Imaris ``.ims`` (HDF5 5.5 format).

Closes the cfos segmentation loop without a TIFF layer:
``ims -> zarr -> segmentation -> masked zarr -> ims``.

The output is always a NEW ``.ims`` file; neither the mask Zarr nor any source
``.ims`` is ever modified. ``Data`` datasets are chunked (<= 4 MB per chunk)
and gzip-compressed, which Imaris requires, and only ``ResolutionLevel 0`` is
written unless ``--pyramid 1|2`` adds 2x levels (mask channel: max pooling;
source channels: mean).

Two modes:

- single channel: the mask becomes the only channel of the new file.
- ``--from_source_ims``: other channels of the original IMS are streamed into
  the new file and the mask replaces channel ``--channel`` (or is appended via
  ``--append_as_new_channel``).
CLI::

    python -m pipeline_modules.utils.masked_zarr_to_ims \
        --input mask.zarr --output masked.ims

    python -m pipeline_modules.utils.masked_zarr_to_ims \
        --input mask.zarr --output masked.ims \
        --from_source_ims original.ims --channel 1 --pyramid 1
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

import numpy as np
from tqdm import tqdm

try:
    import hdf5plugin  # noqa: F401  # registers LZ4/ZSTD HDF5 filters when present
except ImportError:  # pragma: no cover - optional dependency
    hdf5plugin = None

try:
    from pipeline_modules.qc.ims_io import DEFAULT_IMS_HDF_CACHE_MB, resolve_ims_dataset
    from pipeline_modules.utils.errors import ErrorCode, PipelineError
    from pipeline_modules.utils.zarr_io import open_zarr_array
except ImportError:  # pragma: no cover
    from ..qc.ims_io import DEFAULT_IMS_HDF_CACHE_MB, resolve_ims_dataset
    from ..utils.errors import ErrorCode, PipelineError
    from ..utils.zarr_io import open_zarr_array

logger = logging.getLogger(__name__)

DEFAULT_MASK_CHUNK = (64, 256, 256)
MAX_HDF_CHUNK_BYTES = 4 * 1024 * 1024
DEFAULT_Z_BLOCK = 32
DEFAULT_IMS_FORMAT_VERSION = "5.5.0"
DEFAULT_RESOLUTION_LEVEL = 0
DEFAULT_TIMEPOINT = 0


def _require_h5py():
    try:
        import h5py
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise PipelineError(
            ErrorCode.DEPENDENCY_MISSING,
            "h5py is required for Zarr-to-IMS conversion",
            {"dependency": "h5py"},
        ) from exc
    return h5py


def _coerce_chunk_size(value: str | tuple[int, int, int]) -> tuple[int, int, int]:
    if isinstance(value, tuple):
        parts = [int(part) for part in value]
    else:
        parts = [int(part.strip()) for part in str(value).split(",") if part.strip()]
    if len(parts) != 3 or any(part <= 0 for part in parts):
        raise PipelineError(
            ErrorCode.ARGUMENT_INVALID,
            "chunk_size must be three positive integers",
            {"chunk_size": value},
        )
    return (parts[0], parts[1], parts[2])


def clamp_ims_chunks(
    shape_zyx: tuple[int, int, int],
    itemsize: int,
    requested: tuple[int, int, int] = DEFAULT_MASK_CHUNK,
    max_bytes: int = MAX_HDF_CHUNK_BYTES,
) -> tuple[int, int, int]:
    """Clamp chunks to the volume extent and shrink z/y/x (in that order) until
    each chunk stays within ``max_bytes``. Imaris rejects oversized chunks."""
    chunks = [max(1, min(int(requested[i]), int(shape_zyx[i]))) for i in range(3)]
    for i in range(3):
        while chunks[0] * chunks[1] * chunks[2] * int(itemsize) > int(max_bytes) and chunks[i] > 1:
            chunks[i] //= 2
    if chunks[0] * chunks[1] * chunks[2] * int(itemsize) > int(max_bytes):  # pragma: no cover - 1x1x1 always fits
        raise PipelineError(
            ErrorCode.ARGUMENT_INVALID,
            "Cannot satisfy the IMS chunk byte budget",
            {"shape_zyx": list(shape_zyx), "itemsize": int(itemsize), "max_bytes": int(max_bytes)},
        )
    return (chunks[0], chunks[1], chunks[2])


def iter_z_ranges(depth: int, z_block: int) -> Iterator[tuple[int, int]]:
    """Yield ``(z0, z1)`` streaming windows covering ``depth`` rows."""
    step = max(1, int(z_block))
    for z0 in range(0, int(depth), step):
        yield z0, min(z0 + step, int(depth))


def downsample_block_mean(block: np.ndarray) -> np.ndarray:
    """2x isotropic mean of a slab with even extent, dtype preserved."""
    z, y, x = block.shape
    z2, y2, x2 = z - z % 2, y - y % 2, x - x % 2
    work = block[:z2, :y2, :x2]
    if work.size == 0:
        return np.empty((0, 0, 0), dtype=block.dtype)
    stacked = work.reshape(z2 // 2, 2, y2 // 2, 2, x2 // 2, 2)
    if np.dtype(block.dtype).kind in "iub":
        sums = stacked.sum(axis=(1, 3, 5), dtype=np.int64)
        return ((sums + 4) // 8).astype(block.dtype, copy=False)
    return stacked.mean(axis=(1, 3, 5), dtype=np.float64).astype(block.dtype, copy=False)


def downsample_block_majority(block: np.ndarray) -> np.ndarray:
    """2x isotropic majority vote (mean > 0.5 -> 1) of a binary slab."""
    z, y, x = block.shape
    z2, y2, x2 = z - z % 2, y - y % 2, x - x % 2
    work = block[:z2, :y2, :x2]
    if work.size == 0:
        return np.empty((0, 0, 0), dtype=block.dtype)
    stacked = work.reshape(z2 // 2, 2, y2 // 2, 2, x2 // 2, 2)
    sums = stacked.sum(axis=(1, 3, 5), dtype=np.int64)
    return (sums > 4).astype(block.dtype, copy=False)


def downsample_block_max(block: np.ndarray) -> np.ndarray:
    """2x isotropic max pooling. Use for sparse signals (e.g. masked cfos):
    mean dilutes a few-voxel dot to black within 2-3 levels, max keeps its
    peak intensity visible at every pyramid level."""
    z, y, x = block.shape
    z2, y2, x2 = z - z % 2, y - y % 2, x - x % 2
    work = block[:z2, :y2, :x2]
    if work.size == 0:
        return np.empty((z2 // 2, y2 // 2, x2 // 2), dtype=block.dtype)
    return work.reshape(z2 // 2, 2, y2 // 2, 2, x2 // 2, 2).max(axis=(1, 3, 5))


@dataclass
class OutputChannel:
    index: int
    name: str
    source: str  # "mask" or "ims"
    dtype: np.dtype
    source_dataset: Any | None = None  # h5py dataset for source == "ims"


def _list_source_channels(handle: Any) -> list[int]:
    data_set = handle.get("DataSet")
    if data_set is None:
        raise PipelineError(
            ErrorCode.INPUT_FORMAT_INVALID,
            "Invalid IMS file: missing DataSet group",
            {"keys": [str(key) for key in handle.keys()]},
        )
    resolution_key = f"ResolutionLevel {DEFAULT_RESOLUTION_LEVEL}"
    if resolution_key not in data_set:
        available = [str(key) for key in data_set.keys() if str(key).startswith("ResolutionLevel")]
        raise PipelineError(
            ErrorCode.INPUT_FORMAT_INVALID,
            "IMS resolution level not found in source file",
            {"resolution_level": DEFAULT_RESOLUTION_LEVEL, "available": available},
        )
    resolution = data_set[resolution_key]
    if "TimePoint 0" not in resolution:
        raise PipelineError(
            ErrorCode.INPUT_FORMAT_INVALID,
            "IMS timepoint 0 not found in source file",
            {"available": list(resolution.keys())},
        )
    timepoint_group = resolution["TimePoint 0"]
    indices: list[int] = []
    for key in timepoint_group.keys():
        if not str(key).startswith("Channel"):
            continue
        parts = str(key).split(" ")
        if len(parts) > 1:
            try:
                indices.append(int(parts[1]))
            except ValueError:
                continue
    return sorted(indices)


def _ims_attr(text: Any) -> np.ndarray:
    """Encode text the way Imaris does: an |S1 char array, WITHOUT a trailing NUL.

    Real Imaris files store every DataSetInfo value as ``|S1`` arrays sized
    exactly to the text (e.g. 'DataSet' is 7 elements). h5py variable-length
    strings are not recognized, and appending a C-style NUL makes values like
    'DataSet\\x00' fail Imaris's byte-exact lookups ("no dataset").
    """
    raw = str(text).encode("utf-8")
    if not raw:
        return np.array([b""], dtype="S1")
    return np.frombuffer(raw, dtype=np.uint8).view("S1")


def _set_attr(group: Any, name: str, text: Any) -> None:
    group.attrs[name] = _ims_attr(text)


def _hdf_attr_text(value: Any) -> str:
    """Decode an HDF5 attr that Imaris may store as str, bytes, or a |S1 array."""
    if isinstance(value, np.ndarray):
        if value.dtype.kind == "S":
            return b"".join(value.tolist()).decode("utf-8", "ignore")
        value = value.item() if value.ndim == 0 else str(value)
    if isinstance(value, bytes):
        return value.decode("utf-8", "ignore")
    return str(value)


def _source_channel_name(handle: Any, channel: int) -> str:
    group = handle.get(f"DataSetInfo/Channel {channel}")
    if group is not None:
        name = group.attrs.get("Name")
        if name is not None:
            text = _hdf_attr_text(name)
            if text.strip():
                return text
    return f"Channel {channel}"


def build_output_channels(
    source_handle: Any | None,
    mask_shape: tuple[int, int, int],
    mask_dtype: np.dtype,
    *,
    mask_channel: int,
    append_as_new_channel: bool,
    resolution_level: int = DEFAULT_RESOLUTION_LEVEL,
    timepoint: int = DEFAULT_TIMEPOINT,
) -> list[OutputChannel]:
    """Plan the channels of the new file. Validates shapes against the mask."""
    if source_handle is None:
        return [OutputChannel(index=0, name="ch0", source="mask", dtype=mask_dtype)]

    channels: list[OutputChannel] = []
    for index in _list_source_channels(source_handle):
        dataset, _ = resolve_ims_dataset(
            source_handle,
            resolution_level=resolution_level,
            channel=index,
            timepoint=timepoint,
        )
        if tuple(int(v) for v in dataset.shape) != tuple(mask_shape):
            raise PipelineError(
                ErrorCode.INPUT_FORMAT_INVALID,
                "Mask Zarr shape does not match source IMS channel shape",
                {
                    "mask_shape": list(mask_shape),
                    "source_channel": index,
                    "source_shape": [int(v) for v in dataset.shape],
                },
            )
        channels.append(
            OutputChannel(
                index=index,
                name=_source_channel_name(source_handle, index),
                source="ims",
                dtype=np.dtype(dataset.dtype),
                source_dataset=dataset,
            )
        )

    if append_as_new_channel:
        mask_index = max((entry.index for entry in channels), default=-1) + 1
    else:
        mask_index = int(mask_channel)
        if not any(entry.index == mask_index for entry in channels):
            raise PipelineError(
                ErrorCode.ARGUMENT_INVALID,
                "mask_channel not present in source IMS; use --append_as_new_channel to add it",
                {"mask_channel": mask_index, "source_channels": [entry.index for entry in channels]},
            )

    planned: list[OutputChannel] = []
    for entry in channels:
        if entry.index == mask_index:
            planned.append(
                OutputChannel(index=entry.index, name=f"ch{entry.index}", source="mask", dtype=mask_dtype)
            )
        else:
            planned.append(entry)
    if not any(entry.source == "mask" for entry in planned):
        planned.append(
            OutputChannel(index=mask_index, name=f"ch{mask_index}", source="mask", dtype=mask_dtype)
        )
    return sorted(planned, key=lambda entry: entry.index)


def _init_ims_skeleton(
    handle: Any,
    *,
    channels: list[OutputChannel],
    shape_zyx: tuple[int, int, int],
    n_levels: int,
    source_handle: Any | None,
    format_version: str,
    gzip_level: int,
    chunk_size: tuple[int, int, int],
) -> dict[int, Any]:
    """Create the full Imaris 5.5 metadata layout and return {channel: level-0 Data}.

    Mirrors what real Imaris files contain: root format attributes, DataSetInfo
    (Imaris / ImarisDataSet / Log / Image / Channel N / TimeInfo), per-channel
    size and histogram-range attributes, all encoded as |S1 char arrays.
    """
    _write_root_attrs(handle, format_version=format_version)
    timepoint_group = handle.require_group("DataSet").require_group("ResolutionLevel 0").require_group("TimePoint 0")
    depth, height, width = (int(v) for v in shape_zyx)
    data_by_channel: dict[int, Any] = {}
    for entry in channels:
        chunks = clamp_ims_chunks(shape_zyx, np.dtype(entry.dtype).itemsize, chunk_size)
        channel_group = timepoint_group.require_group(f"Channel {entry.index}")
        data_by_channel[entry.index] = channel_group.create_dataset(
            "Data",
            shape=tuple(int(v) for v in shape_zyx),
            chunks=chunks,
            dtype=entry.dtype,
            compression="gzip",
            compression_opts=max(1, int(gzip_level)),
        )
        _set_attr(channel_group, "ImageSizeX", str(width))
        _set_attr(channel_group, "ImageSizeY", str(height))
        _set_attr(channel_group, "ImageSizeZ", str(depth))
        hist_max = 1 if entry.source == "mask" else int(np.iinfo(np.dtype(entry.dtype)).max)
        _set_attr(channel_group, "HistogramMin", "0.000")
        _set_attr(channel_group, "HistogramMax", f"{hist_max:.3f}")
        _set_attr(channel_group, "HistogramMin1024", "0.000")
        _set_attr(channel_group, "HistogramMax1024", f"{hist_max:.3f}")

    info = handle.require_group("DataSetInfo")

    imaris = info.require_group("Imaris")
    _set_attr(imaris, "Version", _source_imaris_version(source_handle) or format_version)
    _set_attr(imaris, "ImageId", "100001")
    _set_attr(imaris, "ThumbnailMode", "thumbnailMIP")
    _set_attr(imaris, "ThumbnailSize", "256")

    imaris_data_set = info.require_group("ImarisDataSet")
    _set_attr(imaris_data_set, "Creator", "Imaris x64")
    _set_attr(imaris_data_set, "NumberOfImages", "1")
    _set_attr(imaris_data_set, "Version", format_version)

    _set_attr(info.require_group("Log"), "Entries", "0")

    image = info.require_group("Image")
    if source_handle is not None:
        # Carry voxel size and any other Image metadata over from the source.
        source_image = source_handle.get("DataSetInfo/Image")
        if source_image is not None:
            for key, value in source_image.attrs.items():
                image.attrs[str(key)] = value
    _set_attr(image, "X", str(width))
    _set_attr(image, "Y", str(height))
    _set_attr(image, "Z", str(depth))
    _set_attr(image, "Noc", str(len(channels)))
    _set_attr(image, "Not", "1")
    if "ExtMin0" not in image.attrs:
        # Extents drive Imaris's voxel-size computation; without them the image
        # is invalid and Imaris shows "no dataset". Default to 1 um isotropic
        # when the source carries no extents.
        for axis, dim in enumerate((width, height, depth)):
            _set_attr(image, f"ExtMin{axis}", "0")
            _set_attr(image, f"ExtMax{axis}", str(dim))
    if "Unit" not in image.attrs:
        _set_attr(image, "Unit", "um")
    if "RecordingDate" not in image.attrs:
        _set_attr(image, "RecordingDate", datetime.now().strftime("%Y-%m-%d %H:%M:%S.000"))

    for entry in channels:
        channel_info = info.require_group(f"Channel {entry.index}")
        if entry.source == "ims" and source_handle is not None:
            source_channel_info = source_handle.get(f"DataSetInfo/Channel {entry.index}")
            if source_channel_info is not None:
                for key, value in source_channel_info.attrs.items():
                    channel_info.attrs[str(key)] = value
        if "Color" not in channel_info.attrs:
            _set_attr(channel_info, "Color", "0.000 1.000 0.000" if entry.source == "mask" else "1.000 0.000 0.000")
        for key, value in (
            ("ColorMode", "BaseColor"),
            ("ColorOpacity", "1.000"),
            ("GammaCorrection", "1.000"),
            ("Description", "(description not specified)"),
            ("ColorRange", "0.000 1.000" if entry.source == "mask" else "0.000 65535.000"),
        ):
            if key not in channel_info.attrs:
                _set_attr(channel_info, key, value)
        _set_attr(channel_info, "Name", entry.name)

    time_info = info.require_group("TimeInfo")
    _set_attr(time_info, "DatasetTimePoints", "1")
    _set_attr(time_info, "FileTimePoints", "1")
    copied_time = False
    if source_handle is not None:
        source_time_info = source_handle.get("DataSetInfo/TimeInfo")
        if source_time_info is not None:
            for key, value in source_time_info.attrs.items():
                time_info.attrs[str(key)] = value
                copied_time = True
    if not copied_time or "TimePoint1" not in time_info.attrs:
        # Imaris numbers TimePointN from 1, not 0.
        _set_attr(time_info, "TimePoint1", datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"))
    if "TimePoint0" in time_info.attrs:  # self-heal files from the old writer
        del time_info.attrs["TimePoint0"]
    return data_by_channel


def _write_root_attrs(handle: Any, *, format_version: str) -> None:
    """Root attributes every real Imaris file carries; without them Imaris
    cannot even identify the file ("no supported images found")."""
    handle.attrs["ImarisVersion"] = _ims_attr(format_version)
    handle.attrs["DataSetDirectoryName"] = _ims_attr("DataSet")
    handle.attrs["DataSetInfoDirectoryName"] = _ims_attr("DataSetInfo")
    handle.attrs["ImarisDataSet"] = _ims_attr("ImarisDataSet")
    handle.attrs["ThumbnailDirectoryName"] = _ims_attr("Thumbnail")
    handle.attrs["NumberOfDataSets"] = np.array([1], dtype=np.uint32)


def _source_imaris_version(source_handle: Any | None) -> str:
    """Creator application version of the source file (e.g. "10.1")."""
    if source_handle is None:
        return ""
    source_imaris = source_handle.get("DataSetInfo/Imaris")
    if source_imaris is None:
        return ""
    version = source_imaris.attrs.get("Version")
    return _hdf_attr_text(version).strip() if version is not None else ""


def _write_thumbnail(handle: Any, channels: list[OutputChannel], data_by_channel: dict[int, Any]) -> None:
    """Write /Thumbnail/Data as the W x 4W RGBA bitmap the format requires.

    Red plane from the first channel, green from the second (mask), sampled
    from the coarsest available level, contrast-stretched to uint8.
    """
    width = 256
    planes: list[np.ndarray] = []
    for entry in channels[:2]:
        data = data_by_channel[entry.index]
        depth = int(data.shape[0])
        z_step = max(1, depth // width)
        mip: np.ndarray | None = None
        for z0 in range(0, depth, z_step * 16):
            slab = np.asarray(data[z0 : min(z0 + z_step * 16, depth) : z_step])
            block_max = slab.max(axis=0)
            mip = block_max if mip is None else np.maximum(mip, block_max)
            del slab, block_max
        assert mip is not None
        rows = (np.linspace(0, mip.shape[0] - 1, width)).astype(np.int64)
        cols = (np.linspace(0, mip.shape[1] - 1, width)).astype(np.int64)
        small = mip[np.ix_(rows, cols)].astype(np.float32)
        lo, hi = np.percentile(small, 1.0), np.percentile(small, 99.5)
        if hi <= lo:
            hi = lo + 1.0
        planes.append(np.clip((small - lo) / (hi - lo) * 255.0, 0.0, 255.0).astype(np.uint8))
    rgba = np.zeros((width, width, 4), dtype=np.uint8)
    rgba[..., 0] = planes[0]
    if len(planes) > 1:
        rgba[..., 1] = planes[1]
    rgba[..., 3] = 255
    handle.require_group("Thumbnail").create_dataset("Data", data=rgba.reshape(width, width * 4))


def _copy_level_zero(
    channels: list[OutputChannel],
    data_by_channel: dict[int, Any],
    mask_array: Any,
    *,
    shape_zyx: tuple[int, int, int],
    chunk_size: tuple[int, int, int],
    z_block: int,
) -> dict[int, np.ndarray]:
    """Stream level-0 data; returns per-channel value histograms for the
    small integer dtypes Imaris displays via Histogram datasets."""
    depth, height, _ = (int(v) for v in shape_zyx)
    band_rows = max(1, min(int(chunk_size[1]), height))
    hist_bins = {
        entry.index: (65536 if np.dtype(entry.dtype).itemsize == 2 else 256)
        for entry in channels
        if np.dtype(entry.dtype).kind in "ui" and np.dtype(entry.dtype).itemsize <= 2
    }
    hist = {index: np.zeros(bins, dtype=np.int64) for index, bins in hist_bins.items()}
    progress = tqdm(total=depth, desc="Zarr -> IMS level 0", unit="slice", file=sys.stderr)
    try:
        for z0, z1 in iter_z_ranges(depth, z_block):
            for entry in channels:
                target = data_by_channel[entry.index]
                for y0 in range(0, height, band_rows):
                    y1 = min(y0 + band_rows, height)
                    if entry.source == "mask":
                        band = np.asarray(mask_array[z0:z1, y0:y1, :])
                    else:
                        band = np.asarray(entry.source_dataset[z0:z1, y0:y1, :])
                    target[z0:z1, y0:y1, :] = band
                    if entry.index in hist:
                        hist[entry.index] += np.bincount(
                            band.ravel(), minlength=hist_bins[entry.index]
                        )
            progress.update(z1 - z0)
    finally:
        progress.close()
    return hist


def _write_channel_hist(group: Any, counts: np.ndarray, vmax: int) -> None:
    for nbins, name in ((256, "Histogram"), (1024, "Histogram1024")):
        live = counts[: vmax + 1]
        idx = np.clip((np.arange(len(live), dtype=np.int64) * nbins) // (vmax + 1), 0, nbins - 1)
        binned = np.zeros(nbins, dtype=np.int64)
        np.add.at(binned, idx, live)
        if name in group:
            del group[name]
        group.create_dataset(name, data=binned.astype(np.uint64))
    _set_attr(group, "HistogramMin", "0.000")
    _set_attr(group, "HistogramMax", f"{vmax:.3f}")
    _set_attr(group, "HistogramMin1024", "0.000")
    _set_attr(group, "HistogramMax1024", f"{vmax:.3f}")


def _percentile_nonzero(counts: np.ndarray, q: float) -> float:
    nonzero = int(counts[1:].sum())
    if nonzero <= 0:
        return 1.0
    cum = np.cumsum(counts[1:])
    return float(max(np.searchsorted(cum, q * nonzero) + 1, 1))


def _downsample_channel_level(
    previous: Any,
    target: Any,
    *,
    method: str,
    chunk_size: tuple[int, int, int],
    z_block: int,
    label: str,
    grow: bool = False,
) -> tuple[int, int, int]:
    """Downsample one level; ``grow`` applies 3x3x3 maximum-filter growth so
    sparse dots keep a visible footprint at coarse levels."""
    from scipy.ndimage import maximum_filter

    previous_shape = tuple(int(v) for v in previous.shape)
    out_shape = tuple(v // 2 for v in previous_shape)
    depth, height, _ = out_shape
    band_rows = max(1, min(int(chunk_size[1]), height))
    down_fn = {"max": downsample_block_max, "mean": downsample_block_mean}[str(method)]
    progress = tqdm(total=depth, desc=f"IMS pyramid {label}", unit="slice", file=sys.stderr)
    try:
        for z0, z1 in iter_z_ranges(depth, z_block):
            hz0, hz1 = (max(0, z0 - 1), min(depth, z1 + 1)) if grow else (z0, z1)
            for y0 in range(0, height, band_rows):
                y1 = min(y0 + band_rows, height)
                hy0, hy1 = (max(0, y0 - 1), min(height, y1 + 1)) if grow else (y0, y1)
                slab = np.asarray(previous[2 * hz0 : 2 * hz1, 2 * hy0 : 2 * hy1, : 2 * out_shape[2]])
                down = down_fn(slab)
                if grow:
                    down = maximum_filter(down, size=3)
                target[z0:z1, y0:y1, :] = down[
                    (z0 - hz0) : (z0 - hz0) + (z1 - z0), (y0 - hy0) : (y0 - hy0) + (y1 - y0)
                ]
                del slab, down
            progress.update(z1 - z0)
    finally:
        progress.close()
    return out_shape


def _write_pyramid_levels(
    handle: Any,
    channels: list[OutputChannel],
    data_by_channel: dict[int, Any],
    *,
    shape_zyx: tuple[int, int, int],
    pyramid_levels: int,
    chunk_size: tuple[int, int, int],
    z_block: int,
    gzip_level: int,
) -> tuple[list[list[int]], dict[int, Any]]:
    level_shapes = [list(int(v) for v in shape_zyx)]
    current: dict[int, Any] = dict(data_by_channel)
    max_itemsize = max(np.dtype(entry.dtype).itemsize for entry in channels)
    level = 0
    while True:
        # Keep writing levels while explicitly requested, and beyond that while
        # the coarsest level still exceeds ~1MB: Imaris previews from the
        # coarsest level, so a huge one makes opening scan a full-size volume.
        requested_more = level + 1 <= int(pyramid_levels)
        oversize = int(np.prod(level_shapes[-1])) * max_itemsize > 1024 * 1024
        if level + 1 >= 8 or not (requested_more or oversize):
            break
        out_shape = tuple(v // 2 for v in level_shapes[-1])
        if min(out_shape) < 1:
            logger.warning("Stopping pyramid at level %d: remaining extent too small", level + 1)
            break
        level += 1
        level_group = handle["DataSet"].require_group(f"ResolutionLevel {level}").require_group("TimePoint 0")
        next_level: dict[int, Any] = {}
        for entry in channels:
            chunks = clamp_ims_chunks(out_shape, np.dtype(entry.dtype).itemsize, chunk_size)
            dataset = level_group.require_group(f"Channel {entry.index}").create_dataset(
                "Data",
                shape=out_shape,
                chunks=chunks,
                dtype=entry.dtype,
                compression="gzip",
                compression_opts=max(1, int(gzip_level)),
            )
            _downsample_channel_level(
                current[entry.index],
                dataset,
                # Sparse masked signals need max pooling to stay visible at
                # coarse levels; dense source channels read naturally with mean.
                method="max" if entry.source == "mask" else "mean",
                chunk_size=chunk_size,
                z_block=z_block,
                label=f"L{level} ch{entry.index}",
                grow=entry.source == "mask" and level >= 2,
            )
            _set_attr(
                level_group[f"Channel {entry.index}"],
                "ImageSizeX",
                str(out_shape[2]),
            )
            _set_attr(level_group[f"Channel {entry.index}"], "ImageSizeY", str(out_shape[1]))
            _set_attr(level_group[f"Channel {entry.index}"], "ImageSizeZ", str(out_shape[0]))
            hist_max = 1 if entry.source == "mask" else int(np.iinfo(np.dtype(entry.dtype)).max)
            _set_attr(level_group[f"Channel {entry.index}"], "HistogramMin", "0.000")
            _set_attr(level_group[f"Channel {entry.index}"], "HistogramMax", f"{hist_max:.3f}")
            next_level[entry.index] = dataset
        current = next_level
        level_shapes.append(list(out_shape))
    return level_shapes, current


def write_masked_ims(
    mask_zarr: str | Path,
    output_ims: str | Path,
    *,
    source_ims: str | Path | None = None,
    mask_channel: int = 0,
    append_as_new_channel: bool = False,
    pyramid_levels: int = 0,
    chunk_size: tuple[int, int, int] | str = DEFAULT_MASK_CHUNK,
    z_block: int = DEFAULT_Z_BLOCK,
    gzip_level: int = 4,
    hdf_cache_mb: int = DEFAULT_IMS_HDF_CACHE_MB,
    overwrite: bool = False,
    format_version: str = DEFAULT_IMS_FORMAT_VERSION,
) -> dict[str, Any]:
    """Write the masked Zarr into a brand-new Imaris ``.ims`` file."""
    h5py = _require_h5py()
    started_at = time.time()
    mask_path = Path(mask_zarr)
    output_path = Path(output_ims)
    source_path = Path(source_ims) if source_ims is not None else None
    chunks_requested = _coerce_chunk_size(chunk_size)
    pyramid_levels = int(pyramid_levels)

    if not mask_path.exists():
        raise PipelineError(ErrorCode.INPUT_NOT_FOUND, "Mask Zarr path not found", {"mask_zarr": str(mask_path)})
    if source_path is not None and not source_path.exists():
        raise PipelineError(ErrorCode.INPUT_NOT_FOUND, "Source IMS file not found", {"source_ims": str(source_path)})
    if source_path is not None and output_path.resolve() == source_path.resolve():
        raise PipelineError(
            ErrorCode.ARGUMENT_INVALID,
            "Refusing to overwrite the source IMS file; choose a different --output",
            {"output_ims": str(output_path), "source_ims": str(source_path)},
        )
    if output_path.exists() and not overwrite:
        raise PipelineError(
            ErrorCode.ARGUMENT_INVALID,
            "Output IMS file already exists; pass overwrite=True / --overwrite to replace it",
            {"output_ims": str(output_path)},
        )
    if pyramid_levels not in (0, 1, 2):
        raise PipelineError(ErrorCode.ARGUMENT_INVALID, "pyramid_levels must be 0, 1 or 2", {"pyramid_levels": pyramid_levels})

    mask_array = open_zarr_array(mask_path)
    if mask_array.ndim != 3:
        raise PipelineError(
            ErrorCode.INPUT_FORMAT_INVALID,
            "Mask Zarr array must be 3D ZYX",
            {"shape": list(int(v) for v in mask_array.shape)},
        )
    mask_shape = tuple(int(v) for v in mask_array.shape)
    mask_dtype = np.dtype(mask_array.dtype)

    kwargs = dict(
        rdcc_nbytes=max(int(hdf_cache_mb), 64) * 1024 * 1024,
        rdcc_nslots=52000,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(str(output_path), "w", **kwargs) as out_handle:
        source_handle = (
            h5py.File(str(source_path), "r", **kwargs) if source_path is not None else None
        )
        try:
            channels = build_output_channels(
                source_handle,
                mask_shape,
                mask_dtype,
                mask_channel=mask_channel,
                append_as_new_channel=append_as_new_channel,
            )
            data_by_channel = _init_ims_skeleton(
                out_handle,
                channels=channels,
                shape_zyx=mask_shape,
                n_levels=pyramid_levels + 1,
                source_handle=source_handle,
                format_version=format_version,
                gzip_level=gzip_level,
                chunk_size=chunks_requested,
            )
            hist = _copy_level_zero(
                channels,
                data_by_channel,
                mask_array,
                shape_zyx=mask_shape,
                chunk_size=chunks_requested,
                z_block=z_block,
            )
            for entry in channels:
                counts = hist.get(entry.index)
                if counts is None:
                    continue
                # Unwritten chunks read back as the 0 fill value.
                counts[0] += int(np.prod(mask_shape)) - int(counts.sum())
                nonzero = np.nonzero(counts)[0]
                vmax = int(nonzero[-1]) if len(nonzero) else 1
                _write_channel_hist(
                    out_handle["DataSet/ResolutionLevel 0/TimePoint 0"][f"Channel {entry.index}"],
                    counts,
                    vmax,
                )
                _set_attr(
                    out_handle[f"DataSetInfo/Channel {entry.index}"],
                    "ColorRange",
                    f"0.000 {_percentile_nonzero(counts, 0.99):.3f}",
                )
            level_shapes = [list(mask_shape)]
            coarsest: dict[int, Any] = dict(data_by_channel)
            if pyramid_levels > 0:
                level_shapes, coarsest = _write_pyramid_levels(
                    out_handle,
                    channels,
                    data_by_channel,
                    shape_zyx=mask_shape,
                    pyramid_levels=pyramid_levels,
                    chunk_size=chunks_requested,
                    z_block=z_block,
                    gzip_level=gzip_level,
                )
            _write_thumbnail(out_handle, channels, coarsest)
        finally:
            if source_handle is not None:
                source_handle.close()

    result = {
        "success": True,
        "mask_zarr": str(mask_path),
        "output_ims": str(output_path),
        "source_ims": str(source_path) if source_path is not None else None,
        "shape": list(mask_shape),
        "mask_dtype": str(mask_dtype),
        "channels": [
            {"index": entry.index, "name": entry.name, "source": entry.source, "dtype": str(entry.dtype)}
            for entry in channels
        ],
        "chunk_size": list(clamp_ims_chunks(mask_shape, mask_dtype.itemsize, chunks_requested)),
        "max_chunk_bytes": MAX_HDF_CHUNK_BYTES,
        "gzip_level": int(gzip_level),
        "pyramid_levels": len(level_shapes),
        "level_shapes": level_shapes,
    }
    result["duration_seconds"] = time.time() - started_at
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write a masked Zarr volume into a new Imaris .ims file")
    parser.add_argument("--input", "-i", required=True, help="Input masked Zarr (uint16 mask)")
    parser.add_argument("--output", "-o", required=True, help="Output .ims path (always a new file)")
    parser.add_argument("--from_source_ims", default="", help="Original .ims whose other channels are streamed into the output")
    parser.add_argument("--channel", type=int, default=0, help="Output channel index that receives the mask (with --from_source_ims)")
    parser.add_argument("--append_as_new_channel", action="store_true", help="Append the mask as a new channel instead of replacing --channel")
    parser.add_argument("--pyramid", type=int, default=0, choices=(0, 1, 2), help="Extra 2x majority-vote resolution levels to write (default: 0)")
    parser.add_argument(
        "--chunk_size",
        default=",".join(str(v) for v in DEFAULT_MASK_CHUNK),
        help="HDF5 chunk size z,y,x; auto-shrunk to the 4MB Imaris budget (default: 64,256,256)",
    )
    parser.add_argument("--z_block", type=int, default=DEFAULT_Z_BLOCK, help="Z slices streamed per pass (default: 32)")
    parser.add_argument("--gzip_level", type=int, default=4, help="HDF5 gzip compression level (default: 4)")
    parser.add_argument("--hdf_cache_mb", type=int, default=DEFAULT_IMS_HDF_CACHE_MB, help="HDF5 chunk cache size in MB")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing output .ims")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    try:
        result = write_masked_ims(
            args.input,
            args.output,
            source_ims=args.from_source_ims or None,
            mask_channel=int(args.channel),
            append_as_new_channel=bool(args.append_as_new_channel),
            pyramid_levels=int(args.pyramid),
            chunk_size=args.chunk_size,
            z_block=int(args.z_block),
            gzip_level=int(args.gzip_level),
            hdf_cache_mb=int(args.hdf_cache_mb),
            overwrite=bool(args.overwrite),
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0
    except PipelineError as exc:
        print(json.dumps(exc.to_dict(), ensure_ascii=False), file=sys.stderr)
        return exc.exit_code
    except Exception as exc:  # pragma: no cover - defensive CLI boundary
        logger.exception("Unhandled Zarr-to-IMS error: %s", exc)
        wrapped = PipelineError(ErrorCode.INTERNAL_ERROR, "Unhandled Zarr-to-IMS error", {"error": str(exc)})
        print(json.dumps(wrapped.to_dict(), ensure_ascii=False), file=sys.stderr)
        return wrapped.exit_code


if __name__ == "__main__":
    sys.exit(main())
