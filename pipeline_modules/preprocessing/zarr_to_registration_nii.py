"""Convert a coarse registration-channel Zarr into the registration NIfTI.

Streams the input Zarr (typically an Imaris pyramid level written by
ims_to_zarr) slab by slab, rescales xy per slab and z once on the assembled
stack, and writes ``volume.nii.gz`` with the production registration NIfTI
convention (xyz transpose + unit affine — see the comment in
``convert_zarr_to_registration_nii``), so ANTs_registration can consume it in
place of the TIFF-folder downsample.

The DL brain mask (``pipeline_modules.segmentation.brain_mask_aind``) is the
only mask source when used: ``--brain_mask`` zeroes everything outside the
brain and is mirrored into ``brain_mask.nii.gz`` (N4 + orientation preflight
consumers). Omitting ``--brain_mask`` runs unmasked (no zeroing, no
brain_mask.nii.gz; ANTs falls back to unmasked behaviour). The old Otsu halo
mask was removed.
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

from pipeline_modules.utils.parsing import parse_resolution_xyz
from pipeline_modules.utils.run_manifest import write_run_manifest
from pipeline_modules.utils.zarr_io import open_zarr_array, read_zarr_attrs

logger = logging.getLogger(__name__)

NO_FLIP = (False, False, False)


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
    logger.warning(
        "Coarse zarr %s has no ims_resolution_level/ims_stride_xyz attrs; "
        "assuming NATIVE resolution %s um. If this zarr is an ims pyramid "
        "level, the output nii physical size will be wrong (atlas appears "
        "magnified). Make coarse reg inputs with --reg_channel or write the "
        "attrs yourself.",
        input_zarr,
        native,
    )
    return native, "native (no ims attrs)"


def _zoom_to_shape(volume: np.ndarray, target_shape: tuple[int, int, int]) -> np.ndarray:
    from scipy import ndimage

    if tuple(volume.shape) == target_shape:
        return volume
    factors = [target / current for target, current in zip(target_shape, volume.shape)]
    return ndimage.zoom(volume, factors, order=1, mode="nearest", prefilter=True)


def _zoom_mask_to_shape(mask: np.ndarray, target_shape: tuple[int, int, int]) -> np.ndarray:
    """Nearest-neighbour zoom for binary masks (order=0 keeps components sharp)."""
    from scipy import ndimage

    if tuple(mask.shape) == target_shape:
        return mask
    factors = [target / current for target, current in zip(target_shape, mask.shape)]
    zoomed = ndimage.zoom(mask.astype(np.uint8), factors, order=0, mode="nearest")
    return zoomed > 0


def load_external_brain_mask(mask_zarr: str | Path, target_shape: tuple[int, int, int], erode: int = 1):
    """Map an external (aind fusion3 DL) coarse-grid brain mask onto the
    registration grid and erode ``erode`` voxels off the boundary rim.

    The former keep-largest-3D-component cleanup was removed: it was a
    workaround for the 0.7-threshold era when the fused mask fragmented;
    at the 0.5 threshold the mask is solid and strays are already dropped
    per-slice by fuse_slab's relative-size filter.

    Returns (mask bool array, stats dict).
    """
    from scipy import ndimage

    mask_path = Path(mask_zarr)
    if not mask_path.exists():
        raise FileNotFoundError(f"Brain mask Zarr not found: {mask_path}")
    arr = open_zarr_array(mask_path, dataset_name="0")
    depth = arr.shape[0]
    slabs = []
    for z in range(depth):
        slab = (np.asarray(arr[z]) > 0).astype(np.uint8)[None]  # (1, ny, nx)
        slabs.append(_zoom_mask_to_shape(slab, (1, target_shape[1], target_shape[2]))[0].astype(np.uint8))
        if (z + 1) % 256 == 0 or z + 1 == depth:
            logger.info("Brain mask xy mapping %d/%d slabs", z + 1, depth)
    mask = _zoom_mask_to_shape(np.stack(slabs, axis=0), target_shape)
    del slabs
    if erode > 0:
        structure = ndimage.generate_binary_structure(3, 1)
        mask = ndimage.binary_erosion(mask, structure=structure, iterations=erode)

    stats = {
        "applied": True,
        "source": str(mask_path),
        "erode": int(erode),
        "voxels": int(mask.sum()),
    }
    logger.info(
        "DL brain mask: %s, kept %d/%d voxels after erode=%d",
        mask_path.name,
        int(mask.sum()),
        int(mask.size),
        erode,
    )
    return mask, stats


def read_flip_atlas(config_path: str | Path | None) -> tuple[bool, bool, bool]:
    """Read ``registration.flip_atlas`` (the ``[ML, AP, DV]`` booleans already
    used by ``pipeline_modules/registration/ANTs_registration.py``) from the
    sample config.

    No config / missing key / all-false -> ``(False, False, False)`` (no flip).
    A malformed entry (not 3 values) raises ``ValueError``.
    """
    if not config_path:
        return NO_FLIP
    path = Path(config_path)
    if not path.exists():
        logger.warning("Config not found (%s); no flip applied", path)
        return NO_FLIP
    with open(path, encoding="utf-8") as handle:
        config = json.load(handle)
    flip = (config.get("registration") or {}).get("flip_atlas")
    if flip is None:
        return NO_FLIP
    flip = [bool(value) for value in flip]
    if len(flip) != 3:
        raise ValueError(f"registration.flip_atlas must have 3 entries [ML, AP, DV], got {flip!r}")
    return (flip[0], flip[1], flip[2])


def flip_axes_from_flip_atlas(flip_atlas: tuple[bool, bool, bool]) -> tuple[int, ...]:
    """Map ``[ML, AP, DV]`` to axis indices of this tool's ``(z, y, x)`` array.

    Follows ``registration/orientation.py::axes_from_flip`` exactly as used by
    ``ANTs_registration._apply_atlas_flip`` (element i flips numpy axis i of
    the index-aligned registration volume): ML -> axis 0 (stack z),
    AP -> axis 1 (y, the historical ``--flip_y`` behaviour), DV -> axis 2 (x).
    """
    return tuple(i for i, flag in enumerate(flip_atlas) if flag)


def convert_zarr_to_registration_nii(
    input_zarr: str | Path,
    output_nii: str | Path,
    *,
    input_resolution_xyz: tuple[float, float, float],
    target_resolution_xyz: tuple[float, float, float],
    dataset_name: str = "0",
    brain_mask: str | Path | None = None,
    brain_mask_erode: int = 1,
    flip_atlas: tuple[bool, bool, bool] = NO_FLIP,
) -> dict:
    """Rescale a Zarr volume to the target isotropic spacing and write a NIfTI.

    ``brain_mask`` (optional) is the coarse-grid DL brain mask (e.g.
    ``ch0_brain_mask_dl.zarr`` from pipeline_modules.segmentation.brain_mask_aind):
    when given it is mapped onto the registration grid, eroded ``brain_mask_erode``
    voxels, used to zero everything
    outside the brain before writing, and mirrored into ``brain_mask.nii.gz``
    next to the NIfTI (N4 + orientation preflight consumers). When ``brain_mask``
    is ``None`` nothing is zeroed and no brain_mask.nii.gz is written (a stale
    one is removed) — unmasked registration, ANTs falls back to its unmasked
    behaviour. An existing ``volume.nii.gz`` (and its side outputs) is
    overwritten, never refused.

    ``flip_atlas`` mirrors the volume along the axes selected by the sample
    config's ``registration.flip_atlas`` ``[ML, AP, DV]`` booleans (see
    :func:`flip_axes_from_flip_atlas`); for samples mounted with a handedness
    flip ANTs cannot fix (``allow_reflection=false``). All-false = no flip.

    Existing side outputs (``brain_mask.nii.gz``, ``original_shape.json``) are
    overwritten consistently with ``volume.nii.gz``."""
    import nibabel as nib

    input_path = Path(input_zarr)
    output_path = Path(output_nii)
    if not input_path.exists():
        raise FileNotFoundError(f"Input Zarr not found: {input_path}")

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

    if brain_mask:
        mask, brain_mask_stats = load_external_brain_mask(
            brain_mask, (out_nz, out_ny, out_nx), erode=brain_mask_erode
        )
        volume_u16 = np.where(mask, volume_u16, 0).astype(np.uint16)
        del mask
    else:
        # Unmasked registration: raw volume straight into ANTs (which itself
        # falls back to its unmasked behaviour when brain_mask.nii.gz is absent).
        brain_mask_stats = {"applied": False, "source": None, "erode": 0, "voxels": int(np.count_nonzero(volume_u16))}

    axes = flip_axes_from_flip_atlas(tuple(bool(v) for v in flip_atlas))
    if axes:
        volume_u16 = np.flip(volume_u16, axis=axes)
        logger.info("flip_atlas %s applied: volume mirrored along axes %s (z,y,x)", list(flip_atlas), list(axes))

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

    # Same geometry as volume.nii.gz (xyz transpose + unit affine) so the
    # mask can be overlaid on the registration volume directly. In the
    # unmasked variant no brain_mask.nii.gz is written — and a stale one from
    # an earlier masked run is removed so ANTs cannot pick it up silently.
    mask_nii_path = output_path.parent / "brain_mask.nii.gz"
    if brain_mask:
        nib.save(nib.Nifti1Image(np.where(volume_xyz > 0, 1, 0).astype(np.uint8), affine), str(mask_nii_path))
    elif mask_nii_path.exists():
        mask_nii_path.unlink()

    with (output_path.parent / "original_shape.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "original_shape": [nz, ny, nx],
                "spacing_xyz": [res_x, res_y, res_z],
                "brain_mask": brain_mask_stats,
                "flip_atlas": [bool(v) for v in flip_atlas],
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
        "brain_mask": brain_mask_stats,
        "flip_atlas": [bool(v) for v in flip_atlas],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rescale a coarse registration-channel Zarr into the registration NIfTI"
    )
    parser.add_argument("--input_zarr", required=True, help="Input .zarr directory (dataset '0')")
    parser.add_argument("--output_nii", required=True, help="Output volume.nii.gz path (overwritten if present)")
    parser.add_argument("--input_resolution_xyz", required=True, help='Input voxel size um "x,y,z"')
    parser.add_argument("--target_resolution_xyz", default="25.0,25.0,25.0", help='Target voxel size um "x,y,z"')
    parser.add_argument("--dataset_name", default="0", help="Dataset name inside the Zarr group")
    parser.add_argument(
        "--brain_mask",
        default=None,
        help="Coarse-grid DL brain-mask Zarr (e.g. ch0_brain_mask_dl.zarr from "
        "pipeline_modules.segmentation.brain_mask_aind); zeroes the registration "
        "NIfTI outside the brain and becomes brain_mask.nii.gz. Omit for unmasked "
        "registration (no zeroing, no brain_mask.nii.gz; stale one removed)",
    )
    parser.add_argument(
        "--brain_mask_erode",
        type=int,
        default=1,
        help="Voxels eroded off the brain-mask boundary (default 1; registration.brain_mask_erode)",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Sample config.json; registration.flip_atlas [ML, AP, DV] decides the volume flip "
        "(missing config or all-false = no flip)",
    )
    parser.add_argument("--json_logs", action="store_true", help="Emit NDJSON log records to stderr")
    return parser


def main() -> int:
    args = parse_args().parse_args()
    if args.json_logs:
        logging.basicConfig(level=logging.INFO, format="%(message)s")
    else:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    flip_atlas = read_flip_atlas(args.config)
    started_at = time.time()
    result = convert_zarr_to_registration_nii(
        args.input_zarr,
        args.output_nii,
        input_resolution_xyz=parse_resolution_xyz(args.input_resolution_xyz),
        target_resolution_xyz=parse_resolution_xyz(args.target_resolution_xyz),
        dataset_name=args.dataset_name,
        brain_mask=args.brain_mask,
        brain_mask_erode=args.brain_mask_erode,
        flip_atlas=flip_atlas,
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
            "brain_mask": args.brain_mask,
            "brain_mask_erode": args.brain_mask_erode,
            "config": args.config,
        },
        outputs=[Path(args.output_nii)],
        started_at=started_at,
        extra=result,
    )
    logger.info("Wrote %s", args.output_nii)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
