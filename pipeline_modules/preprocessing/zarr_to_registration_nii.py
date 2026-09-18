"""Convert a coarse registration-channel Zarr into the registration NIfTI.

Streams the input Zarr (typically an Imaris pyramid level written by
ims_to_zarr) slab by slab, rescales xy per slab and z once on the assembled
stack, and writes ``volume.nii.gz`` with the same NIfTI convention as
preprocessing.downsample (xyz transpose + diagonal affine), so
ANTs_registration can consume it in place of the TIFF-folder downsample.
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
from pipeline_modules.utils.zarr_io import open_zarr_array

logger = logging.getLogger(__name__)


def parse_resolution_xyz(resolution_text: str) -> tuple[float, float, float]:
    parts = [part.strip() for part in str(resolution_text).split(",") if part.strip()]
    if len(parts) != 3:
        raise ValueError(f"Resolution must be x,y,z with 3 values: {resolution_text!r}")
    values = tuple(float(part) for part in parts)
    if any(value <= 0 for value in values):
        raise ValueError(f"Resolution values must be positive: {resolution_text!r}")
    return values


def _zoom_to_shape(volume: np.ndarray, target_shape: tuple[int, int, int]) -> np.ndarray:
    from scipy import ndimage

    if tuple(volume.shape) == target_shape:
        return volume
    factors = [target / current for target, current in zip(target_shape, volume.shape)]
    return ndimage.zoom(volume, factors, order=1, mode="nearest", prefilter=True)


def convert_zarr_to_registration_nii(
    input_zarr: str | Path,
    output_nii: str | Path,
    *,
    input_resolution_xyz: tuple[float, float, float],
    target_resolution_xyz: tuple[float, float, float],
    dataset_name: str = "0",
) -> dict:
    """Rescale a Zarr volume to the target isotropic spacing and write a NIfTI."""
    import nibabel as nib
    from scipy import ndimage

    input_path = Path(input_zarr)
    output_path = Path(output_nii)
    if not input_path.exists():
        raise FileNotFoundError(f"Input Zarr not found: {input_path}")
    if output_path.exists():
        raise FileExistsError(f"Output NIfTI already exists: {output_path}")

    array = open_zarr_array(input_path, dataset_name=dataset_name)
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

    # Same NIfTI convention as preprocessing.downsample: (z, y, x) -> (x, y, z)
    # with a diagonal spacing affine.
    volume_xyz = np.transpose(volume_u16, (2, 1, 0))
    affine = np.eye(4)
    affine[0, 0] = tgt_x
    affine[1, 1] = tgt_y
    affine[2, 2] = tgt_z
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(volume_xyz, affine), str(output_path))

    with (output_path.parent / "original_shape.json").open("w", encoding="utf-8") as handle:
        json.dump({"original_shape": [nz, ny, nx], "spacing_xyz": [res_x, res_y, res_z]}, handle)

    return {
        "input_zarr": str(input_path),
        "output_nii": str(output_path),
        "input_shape_zyx": [nz, ny, nx],
        "output_shape_zyx": [out_nz, out_ny, out_nx],
        "input_resolution_xyz": [res_x, res_y, res_z],
        "target_resolution_xyz": [tgt_x, tgt_y, tgt_z],
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
        },
        outputs=[Path(args.output_nii)],
        started_at=started_at,
        extra=result,
    )
    logger.info("Wrote %s", args.output_nii)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
