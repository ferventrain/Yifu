from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from pipeline_modules.preprocessing.tiff_to_zarr import convert_tiff_to_zarr
    from pipeline_modules.utils.errors import ErrorCode, PipelineError
    from pipeline_modules.utils.run_manifest import write_run_manifest
except ImportError:  # pragma: no cover
    convert_tiff_to_zarr = None  # type: ignore[assignment]
    PipelineError = None  # type: ignore[assignment,misc]
    ErrorCode = None  # type: ignore[assignment]
    write_run_manifest = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

LEFT_ID = np.uint8(1)
RIGHT_ID = np.uint8(2)


def _configure_logging(json_logs: bool) -> None:
    if json_logs:
        class _JsonFormatter(logging.Formatter):
            def format(self, record: logging.LogRecord) -> str:
                return json.dumps(
                    {
                        "level": record.levelname,
                        "logger": record.name,
                        "message": record.getMessage(),
                    },
                    ensure_ascii=False,
                )

        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(_JsonFormatter())
        logging.root.handlers.clear()
        logging.root.addHandler(handler)
        logging.root.setLevel(logging.INFO)
    else:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


def _resolve_default_atlas_label() -> Path:
    import os

    explicit = os.environ.get("YIFU_ATLAS_LABEL")
    if explicit:
        return Path(explicit)
    data_dir = os.environ.get("YIFU_DATA_DIR")
    if data_dir:
        candidate = Path(data_dir) / "reference" / "atlas_label.tiff"
        if candidate.exists():
            return candidate
    return Path(r"H:\Yifu_data\reference\atlas_label.tiff")


def _standard_space_hemi(label_arr_xyz: np.ndarray) -> tuple[np.ndarray, float]:
    """Paint left/right on a standard-space annotation in ANTs order (X,Y,Z)
    around the annotation's x centroid. The reference annotation merges
    left/right structure pairs (every label straddles the midline), so the
    global x centroid IS the midline."""
    mask = label_arr_xyz > 0
    xs = np.nonzero(mask.any(axis=(1, 2)))[0]
    if xs.size == 0:
        raise ValueError("atlas label volume is empty")
    weights = mask.sum(axis=(1, 2)).astype(np.float64)
    mid = float((xs * weights[xs]).sum() / weights[xs].sum())
    hemi = np.zeros(label_arr_xyz.shape, dtype=np.uint8)
    left_plane = np.arange(label_arr_xyz.shape[0])[:, None, None] < mid
    hemi[mask & left_plane] = LEFT_ID
    hemi[mask & ~left_plane] = RIGHT_ID
    return hemi, mid


def _load_standard_label(
    atlas_label_tiff: str | Path, flip_axes: Sequence[int] = ()
) -> np.ndarray:
    """Load the atlas annotation in ANTs (X,Y,Z) order and apply the same
    axis flips registration used. Without this, an ML flip would make the
    hemisphere labels report the mirrored side as left."""
    from pipeline_modules.registration.label_codec import load_label_array_preserving_ids

    axes = tuple(int(axis) for axis in flip_axes)
    if any(axis < 0 or axis > 2 for axis in axes):
        raise ValueError(f"flip_axes must be axis indices 0..2 (ML, AP, DV), got {axes}")
    label_arr_xyz = load_label_array_preserving_ids(atlas_label_tiff)
    if axes:
        label_arr_xyz = np.flip(label_arr_xyz, axis=axes)
        logger.info("Applied atlas flip axes %s before painting hemisphere labels", axes)
    return label_arr_xyz


def _upsample_hemi_and_write(warped_zyx: np.ndarray, target_shape, output_zarr: Path,
                             chunk=(256, 256, 256), flip_axes: Sequence[int] = ()) -> np.ndarray:
    """Nearest-neighbour upsample to the sample grid (z: round-linspace index,
    xy: cv2 nearest) and write in z-batches so each zarr chunk compresses once.
    Returns the per-z left/right boundary curve."""
    import cv2

    from pipeline_modules.utils.zarr_io import create_output_zarr

    z_indices = np.round(np.linspace(0, warped_zyx.shape[0] - 1, target_shape[0])).astype(int)
    root, dataset = create_output_zarr(output_zarr, list(target_shape), chunk, np.dtype("uint8"))
    boundary = np.full(target_shape[0], np.nan)
    target_xy = (int(target_shape[2]), int(target_shape[1]))
    batch_z = int(chunk[0])
    for z0 in range(0, target_shape[0], batch_z):
        z1 = min(z0 + batch_z, target_shape[0])
        batch = np.empty((z1 - z0, target_shape[1], target_shape[2]), dtype=np.uint8)
        for i, sz in enumerate(z_indices[z0:z1]):
            plane = warped_zyx[sz]
            if plane.shape[1] != target_shape[1] or plane.shape[2] != target_shape[2]:
                plane = cv2.resize(plane, target_xy, interpolation=cv2.INTER_NEAREST)
            batch[i] = plane
            cols_l = np.nonzero((plane == int(LEFT_ID)).any(axis=0))[0]
            cols_r = np.nonzero((plane == int(RIGHT_ID)).any(axis=0))[0]
            if cols_l.size and cols_r.size:
                boundary[z0 + i] = (cols_l.max() + 1 + cols_r.min()) / 2.0
        dataset[z0:z1] = batch
    good = ~np.isnan(boundary)
    if good.any():
        idx = np.arange(boundary.shape[0])
        boundary = np.interp(idx, idx[good], boundary[good])
    root.attrs["midline_method"] = "warped standard-space midplane (nearest-neighbour)"
    root.attrs["split_x"] = int(np.ceil(np.nanmedian(boundary))) if good.any() else 0
    root.attrs["split_x_per_z"] = [round(float(v), 2) for v in boundary]
    root.attrs["source"] = "hemisphere_from_transform"
    root.attrs["atlas_flip_axes"] = [int(axis) for axis in flip_axes]
    return boundary


def convert_hemisphere_from_transform(
    sample_dir: str | Path,
    output_zarr: str | Path | None = None,
    atlas_label_tiff: str | Path | None = None,
    *,
    transformlist: list | None = None,
    fixed_image=None,
    atlas_image=None,
    target_shape=None,
    flip_axes: Sequence[int] = (),
) -> dict[str, Any]:
    """Preferred hemisphere method: paint the STANDARD-SPACE midline plane on
    the annotation and warp it with the exact transform that produced
    upsampled_atlas_label. The split surface is the registered midline itself
    (bends and yaws with the sample, poles included).

    Post-run CLI usage reads everything from sample_dir (transforms/ +
    ch0_downsample/volume.nii.gz). Inside the registration flow the in-memory
    objects can be passed instead: transformlist=reg_result['fwdtransforms'],
    fixed_image=register_image, atlas_image=atlas_image,
    target_shape=original_shape, flip_axes=axes_from_flip(effective_flip_atlas).
    """
    import ants

    started_at = time.time()
    sample_dir = Path(sample_dir)
    output_zarr = Path(output_zarr) if output_zarr else sample_dir / "atlas_label_hemisphere.zarr"
    atlas_label_tiff = Path(atlas_label_tiff) if atlas_label_tiff else _resolve_default_atlas_label()
    atlas_tiff = atlas_label_tiff.with_name("atlas.tiff")
    if not atlas_label_tiff.exists():
        raise FileNotFoundError(str(atlas_label_tiff))

    if transformlist is None:
        transforms_dir = sample_dir / "transforms"
        transformlist = [
            str(p)
            for p in sorted(
                (q for q in transforms_dir.iterdir() if q.name.startswith("fwd_")),
                key=lambda q: int(q.name.split("_")[1]),
            )
        ]
        if not transformlist:
            raise FileNotFoundError(f"no fwd_* transforms under {transforms_dir}")
    if fixed_image is None:
        from pipeline_modules.registration.ANTs_registration import _ants_image_read

        reference_nii = sample_dir / "ch0_downsample" / "volume.nii.gz"
        if not reference_nii.exists():
            raise FileNotFoundError(str(reference_nii))
        fixed_image = _ants_image_read(reference_nii)
        fixed_image.set_direction(np.eye(3))  # same direction forcing as registration
    if target_shape is None:
        label_meta = sample_dir / "upsampled_atlas_label.zarr" / "0" / ".zarray"
        if not label_meta.exists():
            raise FileNotFoundError(str(label_meta))
        target_shape = tuple(json.loads(label_meta.read_text(encoding="utf-8"))["shape"])

    label_arr_xyz = _load_standard_label(atlas_label_tiff, flip_axes)  # (X, Y, Z)
    hemi_xyz, mid = _standard_space_hemi(label_arr_xyz)
    if atlas_image is None:
        if not atlas_tiff.exists():
            raise FileNotFoundError(str(atlas_tiff))
        atlas_image = ants.image_read(str(atlas_tiff))
    hemi_img = ants.from_numpy(
        hemi_xyz,
        spacing=atlas_image.spacing,
        origin=atlas_image.origin,
        direction=atlas_image.direction,
    )
    logger.info("standard-space midline x=%.1f; warping with fwd transforms", mid)
    warped = ants.apply_transforms(
        fixed=fixed_image,
        moving=hemi_img,
        transformlist=[str(t) for t in transformlist],
        interpolator="nearestNeighbor",
    )
    warped_zyx = np.transpose(np.asarray(warped.numpy(), dtype=np.uint8), (2, 1, 0))  # -> z,y,x
    if not np.any(warped_zyx > 0):
        raise RuntimeError("warped hemisphere volume is empty - transform/grid mismatch")

    boundary = _upsample_hemi_and_write(
        warped_zyx, target_shape, output_zarr, flip_axes=flip_axes
    )
    logger.info(
        "hemisphere zarr written: %s (boundary median %d, curve %.0f..%.0f, %.0fs)",
        output_zarr,
        int(np.nanmedian(boundary)),
        np.nanmin(boundary),
        np.nanmax(boundary),
        time.time() - started_at,
    )
    return {
        "success": True,
        "output_zarr": str(output_zarr),
        "target_shape": list(target_shape),
        "standard_midline_x": round(mid, 2),
        "split_x": int(np.ceil(np.nanmedian(boundary))),
        "atlas_flip_axes": [int(axis) for axis in flip_axes],
        "elapsed_s": round(time.time() - started_at, 1),
    }


def _parse_flip_axes(value: str) -> tuple[int, ...]:
    if not value.strip():
        return ()
    try:
        axes = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("flip axes must be comma-separated integers") from exc
    if any(axis < 0 or axis > 2 for axis in axes):
        raise argparse.ArgumentTypeError("flip axes must be 0 (ML), 1 (AP) or 2 (DV)")
    return axes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert atlas label to hemisphere label Zarr (warped standard-space midline)"
    )
    parser.add_argument("--sample_dir", required=True, help="Sample dir with transforms/ + ch0_downsample/volume.nii.gz")
    parser.add_argument("--output", required=True, help="Output hemisphere .zarr path")
    parser.add_argument("--atlas_label", default="", help="Standard-space annotation tiff override")
    parser.add_argument(
        "--flip_axes",
        type=_parse_flip_axes,
        default=(),
        help="Atlas axis indices flipped before registration (0=ML, 1=AP, 2=DV)",
    )
    parser.add_argument("--json_logs", action="store_true", help="Emit NDJSON log records to stderr")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    _configure_logging(args.json_logs)
    try:
        result = convert_hemisphere_from_transform(
            args.sample_dir,
            args.output,
            args.atlas_label or None,
            flip_axes=args.flip_axes,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0
    except Exception as exc:  # pragma: no cover
        if PipelineError is not None and isinstance(exc, PipelineError):
            print(json.dumps(exc.to_dict(), ensure_ascii=False), file=sys.stderr)
            return exc.exit_code
        logger.exception("Unhandled hemisphere conversion error: %s", exc)
        if PipelineError is not None and ErrorCode is not None:
            wrapped = PipelineError(ErrorCode.INTERNAL_ERROR, "Unhandled hemisphere conversion error", {"error": str(exc)})
            print(json.dumps(wrapped.to_dict(), ensure_ascii=False), file=sys.stderr)
            return wrapped.exit_code
        print(
            json.dumps(
                {
                    "error": {
                        "code": "INTERNAL_ERROR",
                        "message": "Unhandled hemisphere conversion error",
                        "context": {"error": str(exc)},
                    }
                },
                ensure_ascii=False,
            ),
            file=sys.stderr,
        )
        return 5


if __name__ == "__main__":
    sys.exit(main())
