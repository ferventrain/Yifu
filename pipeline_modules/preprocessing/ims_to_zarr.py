"""Stream an Imaris IMS (HDF5 5.5) file into per-channel Zarr volumes.

Replaces the ``ims -> tiff`` leg of the cfos segmentation pipeline with a
direct ``ims -> zarr`` conversion (``ims -> zarr -> segmentation -> masked
zarr -> ims``). Data is read from the IMS ``Data`` dataset in z-slab batches
aligned to the Zarr z-chunk and never materialised in full.

Resume support: chunks already present on disk in the output Zarr store are
skipped, so an interrupted conversion can simply be re-run.

CLI::

    python -m pipeline_modules.preprocessing.ims_to_zarr \
        --input sample.ims --output sample.zarr --channels 0 \
        --chunk_size 32,256,256

Single ``--channels`` values write the Zarr store to ``--output`` directly;
multi-channel specs write ``<output>/ch{N}.zarr`` per channel (a trailing
``.zarr`` suffix on ``--output`` is dropped for the directory form).

Registration fast path: ``--reg_channel 0`` additionally streams channel 0
from a coarse IMS pyramid level (``--reg_resolution_level``, default 2) into
``ch0_downsampled.zarr`` with ``ims_resolution_level``/``ims_stride_xyz``
attrs, which ``main.py`` downsamples to the registration NIfTI directly.
Interactive runs without the flag are asked whether to extract it.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

try:
    import hdf5plugin  # noqa: F401  # registers LZ4/ZSTD HDF5 filters when present
except ImportError:  # pragma: no cover - optional dependency
    hdf5plugin = None

try:
    from pipeline_modules.preprocessing.tiff_to_zarr import _configure_logging
    from pipeline_modules.qc.ims_io import (
        DEFAULT_IMS_HDF_CACHE_MB,
        open_ims_dataset,
    )
    from pipeline_modules.utils.errors import ErrorCode, PipelineError
    from pipeline_modules.utils.run_manifest import write_run_manifest
    from pipeline_modules.utils.zarr_io import (
        _group_member,
        create_output_zarr,
        list_array_keys,
        list_existing_chunk_indices,
    )
except ImportError:  # pragma: no cover
    from .tiff_to_zarr import _configure_logging
    from ..qc.ims_io import (
        DEFAULT_IMS_HDF_CACHE_MB,
        open_ims_dataset,
    )
    from ..utils.errors import ErrorCode, PipelineError
    from ..utils.run_manifest import write_run_manifest
    from ..utils.zarr_io import (
        _group_member,
        create_output_zarr,
        list_array_keys,
        list_existing_chunk_indices,
    )

logger = logging.getLogger(__name__)

DEFAULT_RESOLUTION_LEVEL = 0
DEFAULT_TIMEPOINT = 0
DEFAULT_IMS_CHUNK_SIZE = (32, 256, 256)


def _require_h5py():
    try:
        import h5py
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise PipelineError(
            ErrorCode.DEPENDENCY_MISSING,
            "h5py is required for IMS-to-Zarr conversion",
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


def list_ims_channel_indices(ims_path: str | Path, *, resolution_level: int = DEFAULT_RESOLUTION_LEVEL) -> list[int]:
    """Return sorted channel indices under ``ResolutionLevel N / TimePoint 0``."""
    h5py = _require_h5py()
    path = Path(ims_path)
    if not path.exists():
        raise PipelineError(ErrorCode.INPUT_NOT_FOUND, "IMS file not found", {"input_ims": str(path)})
    with h5py.File(str(path), "r") as handle:
        try:
            timepoint_group = handle["DataSet"][f"ResolutionLevel {int(resolution_level)}"]["TimePoint 0"]
        except KeyError:
            raise PipelineError(
                ErrorCode.INPUT_FORMAT_INVALID,
                "Invalid IMS file: missing ResolutionLevel/TimePoint",
                {
                    "input_ims": str(path),
                    "resolution_level": int(resolution_level),
                    "data_set_keys": [str(key) for key in handle.get("DataSet", {}).keys()],
                },
            ) from None
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


def parse_channel_spec(spec: str | list[int], available: list[int]) -> list[int]:
    """Resolve a channel spec (``all``, ``0,1`` or list) against available channels."""
    if isinstance(spec, list):
        channels = sorted({int(value) for value in spec})
    else:
        text = str(spec).strip().lower()
        if text in ("", "all"):
            channels = sorted({int(value) for value in available})
        else:
            try:
                channels = sorted({int(part) for part in re.split(r"[,\s]+", text) if part.strip()})
            except ValueError as exc:
                raise PipelineError(
                    ErrorCode.ARGUMENT_INVALID,
                    "channels must be 'all' or comma-separated integers",
                    {"channels": spec},
                ) from exc
    if not channels:
        raise PipelineError(
            ErrorCode.ARGUMENT_INVALID,
            "No channels resolved for IMS-to-Zarr conversion",
            {"channels": spec, "available": available},
        )
    missing = [channel for channel in channels if channel not in set(available)]
    if missing:
        raise PipelineError(
            ErrorCode.INPUT_FORMAT_INVALID,
            "Requested channels missing from IMS file",
            {"missing": missing, "available": available},
        )
    return channels


def resolve_output_paths(output: str | Path, channels: list[int]) -> list[Path]:
    """One channel writes ``--output`` directly; several write ``<output>/ch{N}.zarr``."""
    out = Path(output)
    if len(channels) == 1:
        return [out]
    root = out.with_suffix("") if out.suffix == ".zarr" else out
    return [root / f"ch{channel}.zarr" for channel in channels]


def ims_level_stride_xyz(ims_path: str | Path, resolution_level: int) -> list[float]:
    """Per-axis stride of an IMS pyramid level relative to level 0.

    Reads dataset *shapes* only (no voxel data), so the exact stride is
    computed from the file itself instead of assuming a power-of-two rule.
    """
    h5py = _require_h5py()
    level = int(resolution_level)
    path = Path(ims_path)

    def _shape_for_level(handle, target_level: int) -> tuple[int, int, int]:
        try:
            timepoint_group = handle["DataSet"][f"ResolutionLevel {target_level}"]["TimePoint 0"]
        except KeyError:
            raise PipelineError(
                ErrorCode.INPUT_FORMAT_INVALID,
                "Invalid IMS file: missing ResolutionLevel/TimePoint",
                {"input_ims": str(path), "resolution_level": target_level},
            ) from None
        for key in timepoint_group.keys():
            if not str(key).startswith("Channel"):
                continue
            data = timepoint_group[key].get("Data")
            if data is None:
                continue
            return tuple(int(v) for v in data.shape)
        raise PipelineError(
            ErrorCode.INPUT_FORMAT_INVALID,
            "Invalid IMS file: no channel data at resolution level",
            {"input_ims": str(path), "resolution_level": target_level},
        )

    with h5py.File(str(path), "r") as handle:
        shape_coarse = _shape_for_level(handle, level)
        if level == 0:
            return [1.0, 1.0, 1.0]
        shape_full = _shape_for_level(handle, 0)
    return [shape_full[i] / max(shape_coarse[i], 1) for i in range(3)]


def _write_ims_coarse_attrs(
    output_zarr: str | Path,
    *,
    channel: int,
    resolution_level: int,
    stride_xyz: list[float],
) -> None:
    """Record level/stride metadata so downstream can recover the voxel size."""
    import zarr

    group = zarr.open_group(str(output_zarr), mode="r+")
    group.attrs["ims_channel"] = int(channel)
    group.attrs["ims_resolution_level"] = int(resolution_level)
    group.attrs["ims_stride_xyz"] = [float(v) for v in stride_xyz]


def _coarse_reg_output_path(output: str | Path, channels: list[int], reg_channel: int) -> Path:
    out = Path(output)
    if len(channels) == 1:
        return out.parent / f"ch{reg_channel}_downsampled.zarr"
    root = out.with_suffix("") if out.suffix == ".zarr" else out
    return root / f"ch{reg_channel}_downsampled.zarr"


def _open_existing_writable(output_path: Path) -> Any:
    """Open an existing Zarr group/array in read-write mode (resume path)."""
    import zarr

    root = zarr.open(str(output_path), mode="r+")
    if isinstance(root, zarr.Array):
        return root
    dataset = _group_member(root, "0")
    if dataset is not None and isinstance(dataset, zarr.Array):
        return dataset
    keys = list_array_keys(root)
    if len(keys) == 1:
        return root[keys[0]]
    raise ValueError(f"Could not resolve a writable Zarr array from {output_path}")


def _open_or_create_target(
    output_path: Path,
    *,
    shape: tuple[int, int, int],
    chunks: tuple[int, int, int],
    dtype: Any,
    compressor: Any,
) -> tuple[Any, set[tuple[int, int, int]]]:
    """Create the output array, or reuse an existing store with matching geometry.

    Returns the array plus the set of chunk indices already on disk (empty for
    a fresh store). Stale stores with mismatched shape/dtype/chunks are rebuilt.
    """
    if output_path.exists():
        try:
            existing = _open_existing_writable(output_path)
            same_geometry = (
                tuple(int(v) for v in existing.shape) == tuple(shape)
                and np.dtype(existing.dtype) == np.dtype(dtype)
                and tuple(int(v) for v in existing.chunks) == tuple(chunks)
            )
            if same_geometry:
                logger.info("Resuming %s (%d chunks already on disk)", output_path, len(list_existing_chunk_indices(existing)))
                return existing, set(list_existing_chunk_indices(existing))
        except PipelineError:
            raise
        except Exception:
            logger.info("Existing store at %s is unusable; rebuilding", output_path)
    _, array = create_output_zarr(output_path, shape, chunks, dtype, compressor=compressor)
    return array, set()


def convert_ims_channel_to_zarr(
    input_ims: str | Path,
    output_zarr: str | Path,
    *,
    channel: int = 0,
    resolution_level: int = DEFAULT_RESOLUTION_LEVEL,
    timepoint: int = DEFAULT_TIMEPOINT,
    chunk_size: tuple[int, int, int] | str = DEFAULT_IMS_CHUNK_SIZE,
    gzip_level: int = 1,
    hdf_cache_mb: int = DEFAULT_IMS_HDF_CACHE_MB,
    write_manifest: bool = True,
) -> dict[str, Any]:
    """Stream one IMS channel into a Zarr volume, skipping chunks already written."""
    from numcodecs import GZip

    started_at = time.time()
    input_path = Path(input_ims)
    output_path = Path(output_zarr)
    chunks_requested = _coerce_chunk_size(chunk_size)

    try:
        compressor = GZip(level=max(1, int(gzip_level)))
    except Exception as exc:  # pragma: no cover - numcodecs without GZip
        raise PipelineError(
            ErrorCode.DEPENDENCY_MISSING,
            "numcodecs with GZip support is required",
            {"dependency": "numcodecs", "error": str(exc)},
        ) from exc

    with open_ims_dataset(
        input_path,
        resolution_level=resolution_level,
        channel=channel,
        timepoint=timepoint,
        hdf_cache_mb=hdf_cache_mb,
    ) as (dataset, info):
        shape = tuple(int(v) for v in dataset.shape)
        dtype = np.dtype(dataset.dtype)
        chunks = tuple(min(int(c), int(s)) for c, s in zip(chunks_requested, shape))

        target, existing_chunks = _open_or_create_target(
            output_path,
            shape=shape,
            chunks=chunks,
            dtype=dtype,
            compressor=compressor,
        )

        grid = tuple(max(1, (shape[i] + chunks[i] - 1) // chunks[i]) for i in range(3))
        total_chunks = int(grid[0] * grid[1] * grid[2])
        skipped = 0
        written = 0
        progress = tqdm(total=grid[0], desc=f"IMS ch{channel} -> Zarr", unit="z-chunk-row", file=sys.stderr)
        try:
            for cz in range(grid[0]):
                z0 = cz * chunks[0]
                z1 = min(z0 + chunks[0], shape[0])
                row_tiles = [(cy, cx) for cy in range(grid[1]) for cx in range(grid[2])]
                missing = [tile for tile in row_tiles if (cz, tile[0], tile[1]) not in existing_chunks]
                if not missing:
                    skipped += len(row_tiles)
                    progress.update(1)
                    continue
                if len(missing) == len(row_tiles):
                    # Fast path: whole chunk row missing, stream full-width slab.
                    slab = np.asarray(dataset[z0:z1, :, :])
                    target[z0:z1, :, :] = slab
                    del slab
                    written += len(row_tiles)
                else:
                    # Banded path: read one y-band at a time so memory stays at
                    # (chunk_z x chunk_y x full X) instead of a full z-slab.
                    for cy in range(grid[1]):
                        tiles = [tile for tile in missing if tile[0] == cy]
                        if not tiles:
                            continue
                        y0 = cy * chunks[1]
                        y1 = min(y0 + chunks[1], shape[1])
                        band = np.asarray(dataset[z0:z1, y0:y1, :])
                        for _, cx in tiles:
                            x0 = cx * chunks[2]
                            x1 = min(x0 + chunks[2], shape[2])
                            target[z0:z1, y0:y1, x0:x1] = band[:, :, x0:x1]
                        del band
                        written += len(tiles)
                    skipped += len(row_tiles) - len(missing)
                progress.update(1)
        finally:
            progress.close()

    result = {
        "success": True,
        "input_ims": str(input_path),
        "output_zarr": str(output_path),
        "channel": int(channel),
        "resolution_level": int(resolution_level),
        "timepoint": int(timepoint),
        "shape": list(shape),
        "dtype": str(dtype),
        "chunk_size": list(chunks),
        "total_chunks": total_chunks,
        "written_chunks": written,
        "skipped_chunks": skipped,
        "compressor": f"gzip(level={gzip_level})",
        "source_compression": info.compression,
    }
    result["duration_seconds"] = time.time() - started_at
    if write_manifest:
        manifest_path = write_run_manifest(
            output_path,
            module="preprocessing",
            entrypoint="convert_ims_channel_to_zarr",
            inputs={
                "input_ims": str(input_path),
                "output_zarr": str(output_path),
                "channel": int(channel),
                "resolution_level": int(resolution_level),
                "chunk_size": list(chunks),
            },
            outputs=[output_path],
            started_at=started_at,
            extra=result,
        )
        result["manifest_path"] = str(manifest_path)
    return result


def convert_ims_to_zarr(
    input_ims: str | Path,
    output: str | Path,
    channels: str | list[int] = "0",
    *,
    chunk_size: tuple[int, int, int] | str = DEFAULT_IMS_CHUNK_SIZE,
    resolution_level: int = DEFAULT_RESOLUTION_LEVEL,
    timepoint: int = DEFAULT_TIMEPOINT,
    gzip_level: int = 1,
    hdf_cache_mb: int = DEFAULT_IMS_HDF_CACHE_MB,
    reg_channel: int | None = None,
    reg_resolution_level: int = 2,
) -> dict[str, Any]:
    """Convert selected IMS channels to Zarr. Fails only if no channel converts.

    When ``reg_channel`` is set, that channel is additionally streamed from a
    coarse IMS pyramid level (``reg_resolution_level``) into
    ``ch{N}_downsampled.zarr`` next to the main outputs, for fast registration
    downsampling. Its zarr attrs record ``ims_resolution_level`` and the exact
    per-axis ``ims_stride_xyz`` relative to level 0.
    """
    available = list_ims_channel_indices(input_ims, resolution_level=resolution_level)
    selected = parse_channel_spec(channels, available)
    output_paths = resolve_output_paths(output, selected)

    converted: dict[str, dict[str, Any]] = {}
    failed: dict[str, str] = {}
    for channel, output_path in zip(selected, output_paths):
        try:
            converted[str(channel)] = convert_ims_channel_to_zarr(
                input_ims,
                output_path,
                channel=channel,
                resolution_level=resolution_level,
                timepoint=timepoint,
                chunk_size=chunk_size,
                gzip_level=gzip_level,
                hdf_cache_mb=hdf_cache_mb,
            )
        except PipelineError as exc:
            failed[str(channel)] = str(exc.message)
        except Exception as exc:  # pragma: no cover - defensive batch boundary
            failed[str(channel)] = str(exc)

    if reg_channel is not None:
        try:
            reg_path = _coarse_reg_output_path(output, selected, int(reg_channel))
            converted[f"{int(reg_channel)}_downsampled"] = convert_ims_channel_to_zarr(
                input_ims,
                reg_path,
                channel=int(reg_channel),
                resolution_level=int(reg_resolution_level),
                timepoint=timepoint,
                chunk_size=chunk_size,
                gzip_level=gzip_level,
                hdf_cache_mb=hdf_cache_mb,
            )
            stride_xyz = ims_level_stride_xyz(input_ims, int(reg_resolution_level))
            _write_ims_coarse_attrs(
                reg_path,
                channel=int(reg_channel),
                resolution_level=int(reg_resolution_level),
                stride_xyz=stride_xyz,
            )
        except PipelineError as exc:
            failed[f"{int(reg_channel)}_downsampled"] = str(exc.message)
        except Exception as exc:  # pragma: no cover - defensive batch boundary
            failed[f"{int(reg_channel)}_downsampled"] = str(exc)

    if not converted:
        raise PipelineError(
            ErrorCode.INPUT_FORMAT_INVALID,
            "No IMS channels were converted to Zarr",
            {"input_ims": str(input_ims), "failed": failed},
        )
    return {
        "success": True,
        "input_ims": str(input_ims),
        "channels_requested": selected,
        "converted": converted,
        "failed": failed,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stream an Imaris IMS file into per-channel Zarr volumes")
    parser.add_argument("--input", "-i", required=True, help="Input .ims file")
    parser.add_argument("--output", "-o", required=True, help="Output .zarr path (single channel) or directory root (multi channel)")
    parser.add_argument("--channels", default="0", help="Channel spec: all, 0, or 0,1 (default: 0)")
    parser.add_argument(
        "--chunk_size",
        default=",".join(str(v) for v in DEFAULT_IMS_CHUNK_SIZE),
        help="Zarr chunk size z,y,x (default: 32,256,256). z sets the streaming read slab.",
    )
    parser.add_argument("--resolution_level", type=int, default=DEFAULT_RESOLUTION_LEVEL, help="IMS ResolutionLevel to read")
    parser.add_argument(
        "--reg_channel",
        default="",
        help="Also stream this channel from a coarse pyramid level for registration "
        "(e.g. 0 writes ch0_downsampled.zarr). Empty: ask interactively on a TTY; "
        "'none' disables.",
    )
    parser.add_argument(
        "--reg_resolution_level",
        type=int,
        default=2,
        help="IMS ResolutionLevel for --reg_channel (default: 2)",
    )
    parser.add_argument("--timepoint", type=int, default=DEFAULT_TIMEPOINT, help="IMS TimePoint to read")
    parser.add_argument("--gzip_level", type=int, default=1, help="Zarr gzip compression level (default: 1)")
    parser.add_argument("--hdf_cache_mb", type=int, default=DEFAULT_IMS_HDF_CACHE_MB, help="HDF5 chunk cache size in MB")
    parser.add_argument("--json_logs", action="store_true", help="Emit NDJSON log records to stderr")
    return parser.parse_args()


def resolve_reg_channel(spec: str, *, reg_resolution_level: int, tty: bool | None = None) -> int | None:
    """Resolve --reg_channel into a channel index (None = skip).

    Empty spec asks interactively (TTY only, default No); 'none' always skips.
    """
    text = str(spec).strip().lower()
    if text in ("none", "no", "false", "off"):
        return None
    if text:
        try:
            return int(text)
        except ValueError as exc:
            raise PipelineError(
                ErrorCode.ARGUMENT_INVALID,
                "--reg_channel must be an integer channel index or 'none'",
                {"reg_channel": spec},
            ) from exc
    is_tty = sys.stdin.isatty() if tty is None else tty
    if not is_tty:
        logger.info(
            "Non-interactive run: skipping coarse registration-channel extraction "
            "(pass --reg_channel 0 to extract ch0_downsampled.zarr)."
        )
        return None
    try:
        answer = input(
            f"是否同时提取配准通道 ch0 的降采样 IMS 层 (ResolutionLevel {reg_resolution_level}) "
            f"到 ch0_downsampled.zarr 用于快速配准? [y/N]: "
        )
    except (EOFError, OSError):
        return None
    if answer.strip().lower() not in ("y", "yes"):
        return None
    return 0


def main() -> int:
    args = parse_args()
    _configure_logging(args.json_logs)

    try:
        reg_channel = resolve_reg_channel(args.reg_channel, reg_resolution_level=int(args.reg_resolution_level))
        result = convert_ims_to_zarr(
            args.input,
            args.output,
            args.channels,
            chunk_size=args.chunk_size,
            resolution_level=int(args.resolution_level),
            timepoint=int(args.timepoint),
            gzip_level=int(args.gzip_level),
            hdf_cache_mb=int(args.hdf_cache_mb),
            reg_channel=reg_channel,
            reg_resolution_level=int(args.reg_resolution_level),
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0
    except PipelineError as exc:
        print(json.dumps(exc.to_dict(), ensure_ascii=False), file=sys.stderr)
        return exc.exit_code
    except Exception as exc:  # pragma: no cover - defensive CLI boundary
        logger.exception("Unhandled IMS-to-Zarr error: %s", exc)
        wrapped = PipelineError(ErrorCode.INTERNAL_ERROR, "Unhandled IMS-to-Zarr error", {"error": str(exc)})
        print(json.dumps(wrapped.to_dict(), ensure_ascii=False), file=sys.stderr)
        return wrapped.exit_code


if __name__ == "__main__":
    sys.exit(main())
