"""Pack an HE falsecolor RGB tiff stack into a brand-new Imaris ``.ims``.

Input: a directory of ``<prefix>_HE.tif(f)`` RGB uint8 slices as produced by
``pipeline_modules.HE.falsecolor_batch``. Output: a NEW 3-channel (R/G/B,
uint8) Imaris 5.5 file, streamed slice by slice — the full volume is never
held in memory.

``--from_source_ims`` (recommended when converting an export of an existing
ims) inherits voxel size / extents / recording date from that file's
``DataSetInfo/Image`` group, so the HE volume lands at the correct physical
scale in Imaris. Channel metadata is NOT copied from the source (the HE
channels are R/G/B base colors, not the original fluorophore channels).

Pyramid: plain MAX pooling at every level (keeps the stain vivid at coarse
zoom); levels are written automatically until the coarsest is ~1 MB so Imaris
previews fast (same rule as masked_zarr_to_ims).

CLI::

    python -m pipeline_modules.HE.stack_to_ims \
        --input_dir <dir with *_HE.tif> --output <out.ims> \
        [--from_source_ims src.ims] [--gzip 4] [--overwrite]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

try:
    import hdf5plugin  # noqa: F401  # registers LZ4/ZSTD filters when present
except ImportError:  # pragma: no cover - optional dependency
    hdf5plugin = None

from skimage.io import imread

from pipeline_modules.utils.errors import ErrorCode, PipelineError
from pipeline_modules.utils.masked_zarr_to_ims import (
    DEFAULT_IMS_FORMAT_VERSION,
    DEFAULT_Z_BLOCK,
    OutputChannel,
    _hdf_attr_text,
    _init_ims_skeleton,
    _percentile_nonzero,
    _require_h5py,
    _set_attr,
    _write_channel_hist,
    _write_pyramid_levels,
)

HE_SUFFIX_RE = re.compile(r"^(?P<prefix>.+)_HE\.tiff?$", re.IGNORECASE)
CHANNEL_NAMES = ("Red", "Green", "Blue")
CHANNEL_COLORS = ("1.000 0.000 0.000", "0.000 1.000 0.000", "0.000 0.000 1.000")
DEFAULT_CHUNK = (4, 1024, 1024)


def discover_he_slices(input_dir: Path) -> list[tuple[str, Path]]:
    """Return sorted (prefix, path) for <prefix>_HE.tif(f) files."""
    import os

    found: list[tuple[str, Path]] = []
    with os.scandir(input_dir) as entries:
        for entry in entries:
            match = HE_SUFFIX_RE.match(entry.name) if entry.is_file() else None
            if match is not None:
                found.append((match.group("prefix"), Path(entry.path)))
    found.sort(key=lambda item: item[0])
    if not found:
        raise PipelineError(
            ErrorCode.INPUT_NOT_FOUND,
            "No <prefix>_HE.tif slices found in input dir",
            {"input_dir": str(input_dir)},
        )
    return found


def _inherit_source_geometry(
    source_handle: Any, width: int, height: int, depth: int
) -> dict[str, str]:
    """ExtMin/ExtMax attrs scaled to the stack dims, from the source's voxel spacing.

    The export stack may be cropped relative to the source ims (dims differ);
    spacing survives that, absolute position does not — extents are anchored at
    the source's ExtMin and scaled per axis.
    """
    image = source_handle.get("DataSetInfo/Image")
    attrs: dict[str, str] = {}
    if image is None:
        return attrs
    dims = {"X": width, "Y": height, "Z": depth}
    for axis_idx, axis in enumerate(("X", "Y", "Z")):
        try:
            src_dim = int(float(_hdf_attr_text(image.attrs.get(axis, "")).strip() or "0"))
            ext_min = float(_hdf_attr_text(image.attrs.get(f"ExtMin{axis_idx}", "")).strip() or "0")
            ext_max = float(_hdf_attr_text(image.attrs.get(f"ExtMax{axis_idx}", "")).strip() or "0")
        except ValueError:
            continue
        if src_dim <= 0 or ext_max <= ext_min:
            continue
        spacing = (ext_max - ext_min) / src_dim
        if src_dim != dims[axis]:
            print(
                f"Source {axis}={src_dim} vs stack {dims[axis]}: "
                f"extents scaled at {spacing:.4f} um/px",
                flush=True,
            )
        attrs[f"ExtMin{axis_idx}"] = f"{ext_min:.9f}"
        attrs[f"ExtMax{axis_idx}"] = f"{ext_min + dims[axis] * spacing:.9f}"
    return attrs


def _write_rgb_thumbnail(handle: Any, coarsest: dict[int, Any], channels: list[OutputChannel]) -> None:
    """/Thumbnail/Data as W x 4W RGBA; R/G/B from the three HE channels."""
    width = 256
    planes: dict[int, np.ndarray] = {}
    for entry in channels[:3]:
        data = coarsest[entry.index]
        depth = int(data.shape[0])
        z_step = max(1, depth // width)
        mip: np.ndarray | None = None
        for z0 in range(0, depth, z_step * 16):
            slab = np.asarray(data[z0 : min(z0 + z_step * 16, depth) : z_step])
            block_max = slab.max(axis=0)
            mip = block_max if mip is None else np.maximum(mip, block_max)
            del slab, block_max
        rows = np.linspace(0, mip.shape[0] - 1, width).astype(np.int64)
        cols = np.linspace(0, mip.shape[1] - 1, width).astype(np.int64)
        small = mip[np.ix_(rows, cols)].astype(np.float32)
        lo, hi = np.percentile(small, 1.0), np.percentile(small, 99.5)
        if hi <= lo:
            hi = lo + 1.0
        planes[entry.index] = np.clip((small - lo) / (hi - lo) * 255.0, 0.0, 255.0).astype(np.uint8)

    rgba = np.zeros((width, width, 4), dtype=np.uint8)
    for i, entry in enumerate(channels[:3]):
        rgba[..., i] = planes[entry.index]
    rgba[..., 3] = 255
    handle.require_group("Thumbnail").create_dataset("Data", data=rgba.reshape(width, width * 4))


def write_stack_ims(
    he_tiff_dir: str | Path,
    output_ims: str | Path,
    *,
    source_ims: str | Path | None = None,
    pyramid_levels: int = 0,
    chunk_size: tuple[int, int, int] = DEFAULT_CHUNK,
    z_block: int = DEFAULT_Z_BLOCK,
    gzip_level: int = 4,
    hdf_cache_mb: int = 128,
    overwrite: bool = False,
    format_version: str = DEFAULT_IMS_FORMAT_VERSION,
) -> dict[str, Any]:
    """Stream the HE tiff stack into a new .ims; returns a summary dict."""
    h5py = _require_h5py()
    started_at = time.time()
    input_dir = Path(he_tiff_dir)
    output_path = Path(output_ims)
    source_path = Path(source_ims) if source_ims is not None else None

    if not input_dir.exists():
        raise PipelineError(ErrorCode.INPUT_NOT_FOUND, "HE tiff dir not found", {"input_dir": str(input_dir)})
    if source_path is not None and not source_path.exists():
        raise PipelineError(ErrorCode.INPUT_NOT_FOUND, "Source IMS file not found", {"source_ims": str(source_path)})
    if source_path is not None and output_path.resolve() == source_path.resolve():
        raise PipelineError(
            ErrorCode.ARGUMENT_INVALID,
            "Refusing to overwrite the source IMS file; choose a different --output",
            {"output_ims": str(output_path)},
        )
    if output_path.exists() and not overwrite:
        raise PipelineError(
            ErrorCode.ARGUMENT_INVALID,
            "Output IMS file already exists; pass overwrite=True / --overwrite to replace it",
            {"output_ims": str(output_path)},
        )

    slices = discover_he_slices(input_dir)
    first = imread(str(slices[0][1]))
    if first.ndim != 3 or first.shape[2] != 3:
        raise PipelineError(
            ErrorCode.INPUT_FORMAT_INVALID,
            "HE slices must be RGB (H, W, 3) uint8",
            {"file": str(slices[0][1]), "shape": list(first.shape)},
        )
    height, width = int(first.shape[0]), int(first.shape[1])
    depth = len(slices)
    shape_zyx = (depth, height, width)

    channels = [
        OutputChannel(index=i, name=CHANNEL_NAMES[i], source="mask", dtype=np.dtype(np.uint8))
        for i in range(3)
    ]

    kwargs = dict(rdcc_nbytes=max(int(hdf_cache_mb), 64) * 1024 * 1024, rdcc_nslots=52000)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(str(output_path), "w", **kwargs) as out_handle:
        source_handle = h5py.File(str(source_path), "r", **kwargs) if source_path is not None else None
        try:
            data_by_channel = _init_ims_skeleton(
                out_handle,
                channels=channels,
                shape_zyx=shape_zyx,
                n_levels=1,  # pyramid levels appended by _write_pyramid_levels
                source_handle=source_handle,
                format_version=format_version,
                gzip_level=gzip_level,
                chunk_size=chunk_size,
            )
            if source_handle is not None:
                # Skeleton copies the source Image attrs verbatim (source dims);
                # re-anchor extents to the stack dims at the source's spacing.
                image_info = out_handle["DataSetInfo/Image"]
                for key, value in _inherit_source_geometry(source_handle, width, height, depth).items():
                    _set_attr(image_info, key, value)
            # HE channels: R/G/B base colors, uint8 range (skeleton's "mask"
            # defaults assume a binary channel).
            for i, entry in enumerate(channels):
                group = out_handle["DataSet/ResolutionLevel 0/TimePoint 0"][f"Channel {entry.index}"]
                _set_attr(group, "HistogramMax", "255.000")
                _set_attr(group, "HistogramMax1024", "255.000")
                info_group = out_handle[f"DataSetInfo/Channel {entry.index}"]
                _set_attr(info_group, "Color", CHANNEL_COLORS[i])
                _set_attr(info_group, "ColorRange", "0.000 255.000")

            counts = {entry.index: np.zeros(256, dtype=np.int64) for entry in channels}
            print(f"Writing {depth} slices ({width}x{height}x3 uint8) -> {output_path}", flush=True)
            for z, (prefix, path) in enumerate(slices):
                rgb = imread(str(path))
                if rgb.shape[:2] != (height, width):
                    raise PipelineError(
                        ErrorCode.INPUT_FORMAT_INVALID,
                        "Slice geometry differs from the first slice",
                        {"file": str(path), "shape": list(rgb.shape), "expected": [height, width]},
                    )
                for entry in channels:
                    plane = rgb[..., entry.index]
                    counts[entry.index] += np.bincount(plane.ravel(), minlength=256)
                    data_by_channel[entry.index][z, :, :] = plane
                if (z + 1) % 100 == 0 or z + 1 == depth:
                    print(f"[{z + 1}/{depth}] {prefix}", flush=True)

            for entry in channels:
                group = out_handle["DataSet/ResolutionLevel 0/TimePoint 0"][f"Channel {entry.index}"]
                _write_channel_hist(group, counts[entry.index], 255)
                _set_attr(
                    out_handle[f"DataSetInfo/Channel {entry.index}"],
                    "ColorRange",
                    f"0.000 {_percentile_nonzero(counts[entry.index], 0.99):.3f}",
                )

            level_shapes, coarsest = _write_pyramid_levels(
                out_handle,
                channels,
                data_by_channel,
                shape_zyx=shape_zyx,
                pyramid_levels=pyramid_levels,
                chunk_size=chunk_size,
                z_block=z_block,
                gzip_level=gzip_level,
            )
            _write_rgb_thumbnail(out_handle, coarsest, channels)
        finally:
            if source_handle is not None:
                source_handle.close()

    return {
        "success": True,
        "input_dir": str(input_dir),
        "output_ims": str(output_path),
        "source_ims": str(source_path) if source_path is not None else None,
        "shape": [int(v) for v in shape_zyx],
        "channels": [entry.name for entry in channels],
        "pyramid_levels": len(level_shapes),
        "level_shapes": level_shapes,
        "gzip_level": int(gzip_level),
        "duration_seconds": time.time() - started_at,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Pack HE falsecolor RGB tiff stack into Imaris .ims.")
    parser.add_argument("--input_dir", type=Path, required=True, help="Dir with <prefix>_HE.tif slices")
    parser.add_argument("--output", type=Path, required=True, help="Output .ims path")
    parser.add_argument("--from_source_ims", type=Path, default=None, help="Inherit voxel size/metadata")
    parser.add_argument("--pyramid_levels", type=int, default=0, help="0 = auto until coarsest ~1MB")
    parser.add_argument("--gzip", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    result = write_stack_ims(
        args.input_dir,
        args.output,
        source_ims=args.from_source_ims,
        pyramid_levels=args.pyramid_levels,
        gzip_level=args.gzip,
        overwrite=args.overwrite,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
