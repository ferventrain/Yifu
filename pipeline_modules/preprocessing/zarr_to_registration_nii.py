"""Convert a coarse registration-channel Zarr into the registration NIfTI.

Streams the input Zarr (typically an Imaris pyramid level written by
ims_to_zarr) slab by slab, rescales xy per slab and z once on the assembled
stack, and writes ``volume.nii.gz`` with the production registration NIfTI
convention (xyz transpose + UNIT affine — see the comment in
``convert_zarr_to_registration_nii``), so ANTs_registration can consume it in
place of the TIFF-folder downsample.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline_modules.utils.run_manifest import write_run_manifest
from pipeline_modules.utils.zarr_io import open_zarr_array, read_zarr_attrs

logger = logging.getLogger(__name__)


def parse_resolution_xyz(resolution_text: str) -> tuple[float, float, float]:
    parts = [part.strip() for part in str(resolution_text).split(",") if part.strip()]
    if len(parts) != 3:
        raise ValueError(f"Resolution must be x,y,z with 3 values: {resolution_text!r}")
    values = tuple(float(part) for part in parts)
    if any(value <= 0 for value in values):
        raise ValueError(f"Resolution values must be positive: {resolution_text!r}")
    return values


def resolve_input_resolution_xyz(
    input_zarr: str | Path,
    native_resolution_xyz: tuple[float, float, float],
) -> tuple[tuple[float, float, float], str]:
    """Resolution of a registration-input Zarr, coarse-level aware.

    ``ims_to_zarr --reg_channel`` records ``ims_stride_xyz`` (exact per-axis
    stride vs IMS level 0) and ``ims_resolution_level`` on the output zarr;
    prefer them, fall back to ``2**level``, else assume native resolution.
    Returns ``(resolution_xyz, source_note)``.
    """
    import zarr

    native = tuple(float(v) for v in native_resolution_xyz)
    try:
        attrs = read_zarr_attrs(input_zarr)
        stride = attrs.get("ims_stride_xyz")
        if stride and len(stride) == 3 and all(float(v) > 0 for v in stride):
            res = tuple(native[i] * float(stride[i]) for i in range(3))
            return res, f"ims_stride_xyz={tuple(round(float(v), 2) for v in stride)}"
        level = attrs.get("ims_resolution_level")
        if level is not None:
            factor = 2 ** int(level)
            return (
                tuple(v * factor for v in native),
                f"ims_resolution_level={int(level)} (2^level fallback)",
            )
    except Exception:  # noqa: BLE001 - metadata lookup is best-effort
        pass
    return native, "native (no ims attrs)"


def _zoom_to_shape(volume: np.ndarray, target_shape: tuple[int, int, int]) -> np.ndarray:
    from scipy import ndimage

    if tuple(volume.shape) == target_shape:
        return volume
    factors = [target / current for target, current in zip(target_shape, volume.shape)]
    return ndimage.zoom(volume, factors, order=1, mode="nearest", prefilter=True)


def _otsu_threshold(values: np.ndarray) -> float:
    """Otsu inter-class variance maximum over a 256-bin histogram.

    Same algorithm as skimage.filters.threshold_otsu, kept local so the
    masking does not depend on an optional dependency.
    """
    hist, edges = np.histogram(values, bins=256)
    centers = (edges[:-1] + edges[1:]) / 2
    weights = hist.astype(np.float64)
    total = weights.sum()
    if total == 0:
        return float(centers[0])
    best, threshold = -1.0, float(centers[0])
    cum_w = np.cumsum(weights)
    cum_m = np.cumsum(weights * centers)
    end_m = cum_m[-1]
    for i in range(1, len(centers)):
        w0, w1 = cum_w[i - 1], total - cum_w[i - 1]
        if w0 == 0 or w1 == 0:
            continue
        m0 = cum_m[i - 1] / w0
        m1 = (end_m - cum_m[i - 1]) / w1
        var = w0 * w1 * (m0 - m1) ** 2
        if var > best:
            best, threshold = var, float(centers[i])
    return threshold


def compute_halo_brain_mask(volume: np.ndarray) -> tuple[np.ndarray, dict]:
    """Auto brain mask that cuts the scattering halo around the tissue.

    Rationale: LSFM backgrounds are not zero (noise floor + a dim halo of
    scattered signal around the tissue). Mutual information treats that rim
    as "dark tissue", which lets the atlas inflate past the true boundary.
    The mask zeroes everything outside the brain so the registration metric
    sees a real background on both sides.

    Gaussian smooth (sigma 1.2) -> Otsu threshold on positive voxels, floored
    at 0.35 * p99 -> binary closing (2 iters) -> fill holes -> largest
    connected component -> erode 2 voxels (50 um at the 25 um grid) to remove
    the halo rim itself from the registration volume.
    """
    from scipy import ndimage

    smooth = ndimage.gaussian_filter(volume.astype(np.float32), sigma=1.2)
    positive = smooth[smooth > 0]
    if positive.size == 0:
        stats = {"applied": True, "threshold": 0.0, "otsu": 0.0, "p99": 0.0, "voxels": int(volume.size)}
        return np.ones(volume.shape, dtype=bool), stats
    p99 = float(np.percentile(positive, 99))
    otsu = _otsu_threshold(positive)
    threshold = max(otsu, 0.35 * p99)

    mask = smooth > threshold
    structure = ndimage.generate_binary_structure(3, 1)
    mask = ndimage.binary_closing(mask, structure=structure, iterations=2)
    mask = ndimage.binary_fill_holes(mask)
    labels, count = ndimage.label(mask)
    if count > 1:
        sizes = ndimage.sum(mask, labels, range(1, count + 1))
        mask = labels == (1 + int(np.argmax(sizes)))
    mask = ndimage.binary_erosion(mask, structure=structure, iterations=2)

    stats = {
        "applied": True,
        "threshold": round(float(threshold), 3),
        "otsu": round(float(otsu), 3),
        "p99": round(p99, 3),
        "voxels": int(mask.sum()),
    }
    logger.info(
        "Halo mask: otsu=%.1f p99=%.1f -> thr=%.1f, kept %d/%d voxels",
        otsu,
        p99,
        threshold,
        int(mask.sum()),
        mask.size,
    )
    return mask, stats


def convert_zarr_to_registration_nii(
    input_zarr: str | Path,
    output_nii: str | Path,
    *,
    input_resolution_xyz: tuple[float, float, float],
    target_resolution_xyz: tuple[float, float, float],
    dataset_name: str = "0",
    flip_y: bool = False,
    halo_mask: bool = True,
) -> dict:
    """Rescale a Zarr volume to the target isotropic spacing and write a NIfTI.

    ``flip_y`` mirrors along the sample's y axis: for samples whose xy plane
    is mounted upside-down relative to the atlas (ANTs with
    allow_reflection=false cannot fix a handedness flip).

    ``halo_mask`` (default on) zeroes everything outside an auto-detected
    brain mask before writing, so the scattering halo around the tissue never
    reaches the registration metric. The mask is also saved next to the NIfTI
    as ``brain_mask.nii.gz``."""
    import nibabel as nib
    from scipy import ndimage

    input_path = Path(input_zarr)
    output_path = Path(output_nii)
    if not input_path.exists():
        raise FileNotFoundError(f"Input Zarr not found: {input_path}")
    if output_path.exists():
        raise FileExistsError(f"Output NIfTI already exists: {output_path}")

    array = open_zarr_array(input_path, dataset_name=dataset_name)
    try:  # fail fast with a clear message instead of async task errors mid-run
        _ = np.asarray(array[tuple(0 for _ in array.shape)])
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"Opened {input_path} but cannot read its data ({type(exc).__name__}: {exc}). "
            "The store format is not readable by this zarr build; re-export it with "
            "ims_to_zarr (group store, dataset '0', blosc compressor)."
        ) from exc
    nz, ny, nx = (int(value) for value in array.shape)
    res_x, res_y, res_z = input_resolution_xyz
    tgt_x, tgt_y, tgt_z = target_resolution_xyz

    out_ny = max(int(round(ny * res_y / tgt_y)), 1)
    out_nx = max(int(round(nx * res_x / tgt_x)), 1)
    out_nz = max(int(round(nz * res_z / tgt_z)), 1)

    logger.info(
        "Input %s shape=%s spacing=(%.4f, %.4f, %.4f) um -> output shape=%s spacing=(%.3f, %.3f, %.3f) um",
        input_path,
        (nz, ny, nx),
        res_x,
        res_y,
        res_z,
        (out_nz, out_ny, out_nx),
        tgt_x,
        tgt_y,
        tgt_z,
    )

    slabs: list[np.ndarray] = []
    for z in range(nz):
        slab = np.asarray(array[z], dtype=np.float32)
        slabs.append(_zoom_to_shape(slab[np.newaxis, :, :], (1, out_ny, out_nx))[0])
        if (z + 1) % 128 == 0 or z + 1 == nz:
            logger.info("Rescaled xy %d/%d slabs", z + 1, nz)
    volume = np.stack(slabs, axis=0)
    del slabs

    volume = _zoom_to_shape(volume, (out_nz, out_ny, out_nx))
    volume_u16 = np.clip(np.rint(volume), 0, 65535).astype(np.uint16)
    del volume

    halo_stats: dict = {"applied": False}
    if halo_mask:
        mask, halo_stats = compute_halo_brain_mask(volume_u16)
        volume_u16 = np.where(mask, volume_u16, 0).astype(np.uint16)
        del mask

    if flip_y:
        # Mirror along the sample's y axis: for samples whose xy plane is
        # mounted upside-down relative to the atlas (ANTs with
        # allow_reflection=false cannot fix a handedness flip).
        volume_u16 = volume_u16[:, ::-1, :]
        logger.info("flip_y applied: volume mirrored along y")

    # NIfTI convention of the proven dbdb36 production registration: (z, y, x)
    # transposed to (x, y, z) with a UNIT diagonal affine. The reference atlas
    # TIFF carries no spacing metadata and the registration loads it at unit
    # spacing, so registration runs voxel-isometrically. Writing a true
    # physical affine (e.g. 25 um) makes the atlas' physical extent land
    # outside the sample volume and ANTs warps it into an empty volume. The
    # true voxel size is recorded in original_shape.json instead.
    volume_xyz = np.transpose(volume_u16, (2, 1, 0))
    affine = np.eye(4)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(volume_xyz, affine), str(output_path))

    if halo_mask:
        # Same geometry as volume.nii.gz (xyz transpose + unit affine) so the
        # mask can be overlaid on the registration volume directly.
        mask_nii_path = output_path.parent / "brain_mask.nii.gz"
        nib.save(nib.Nifti1Image(np.where(volume_xyz > 0, 1, 0).astype(np.uint8), affine), str(mask_nii_path))

    with (output_path.parent / "original_shape.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "original_shape": [nz, ny, nx],
                "spacing_xyz": [res_x, res_y, res_z],
                "halo_mask": halo_stats,
            },
            handle,
        )

    return {
        "input_zarr": str(input_path),
        "output_nii": str(output_nii),
        "input_shape_zyx": [nz, ny, nx],
        "output_shape_zyx": [out_nz, out_ny, out_nx],
        "input_resolution_xyz": [res_x, res_y, res_z],
        "target_resolution_xyz": [tgt_x, tgt_y, tgt_z],
        "halo_mask": halo_stats,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rescale a coarse registration-channel Zarr into the registration NIfTI"
    )
    parser.add_argument("--input_zarr", required=True, help="Input .zarr directory (dataset '0')")
    parser.add_argument("--output_nii", required=True, help="Output volume.nii.gz path")
    parser.add_argument("--input_resolution_xyz", required=True, help='Input voxel size um "x,y,z"')
    parser.add_argument("--target_resolution_xyz", default="25.0,25.0,25.0", help='Target voxel size um "x,y,z"')
    parser.add_argument("--dataset_name", default="0", help="Dataset name inside the Zarr group")
    parser.add_argument(
        "--flip_y",
        action="store_true",
        help="Mirror along the sample y axis (upside-down xy plane samples)",
    )
    parser.add_argument(
        "--halo_mask",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Auto brain mask before writing: zero out the scattering halo "
        "around the tissue so registration cannot inflate past the boundary "
        "(default on; --no-halo_mask to disable)",
    )
    parser.add_argument("--json_logs", action="store_true", help="Emit NDJSON log records to stderr")
    return parser


def main() -> int:
    args = parse_args().parse_args()
    if args.json_logs:
        logging.basicConfig(level=logging.INFO, format="%(message)s")
    else:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    started_at = time.time()
    result = convert_zarr_to_registration_nii(
        args.input_zarr,
        args.output_nii,
        input_resolution_xyz=parse_resolution_xyz(args.input_resolution_xyz),
        target_resolution_xyz=parse_resolution_xyz(args.target_resolution_xyz),
        dataset_name=args.dataset_name,
        flip_y=args.flip_y,
        halo_mask=args.halo_mask,
    )
    write_run_manifest(
        Path(args.output_nii).parent,
        module="pipeline_modules.preprocessing.zarr_to_registration_nii",
        entrypoint="convert_zarr_to_registration_nii",
        inputs={
            "input_zarr": args.input_zarr,
            "input_resolution_xyz": args.input_resolution_xyz,
            "target_resolution_xyz": args.target_resolution_xyz,
            "dataset_name": args.dataset_name,
            "flip_y": args.flip_y,
            "halo_mask": args.halo_mask,
        },
        outputs=[Path(args.output_nii)],
        started_at=started_at,
        extra=result,
    )
    logger.info("Wrote %s", args.output_nii)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
