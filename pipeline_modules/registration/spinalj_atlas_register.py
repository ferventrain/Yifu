#!/usr/bin/env python3
"""Trial registration of LSFM spinal-cord volumes to the SpinalJ SC_P56 atlas.

Uses SpinalJ atlas files only (Template.tif + Annotation.tif). Does not use the
SpinalJ Fiji plugin. Designed for MegaSpim-style stacks where the long SC axis
is in-plane (Y) rather than the stack Z axis.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path

import numpy as np
import tifffile

logger = logging.getLogger(__name__)

# SpinalJ atlas voxel size (from SC_P56_Atlas_10x10x20_* folder name / paper).
ATLAS_SPACING_XYZ_UM = (10.0, 10.0, 20.0)


def _configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


def find_atlas_files(atlas_dir: Path) -> dict[str, Path]:
    atlas_dir = atlas_dir.resolve()
    template = atlas_dir / "Template.tif"
    annotation = atlas_dir / "Annotation.tif"
    if not template.exists():
        matches = list(atlas_dir.rglob("Template.tif"))
        if not matches:
            raise FileNotFoundError(f"Template.tif not found under {atlas_dir}")
        atlas_dir = matches[0].parent
        template = matches[0]
        annotation = atlas_dir / "Annotation.tif"
    if not annotation.exists():
        # Some releases may use plural filename.
        alt = atlas_dir / "Annotations.tif"
        if alt.exists():
            annotation = alt
        else:
            raise FileNotFoundError(f"Annotation.tif not found in {atlas_dir}")
    return {"atlas_dir": atlas_dir, "template": template, "annotation": annotation}


def load_tiff_stack_as_zyx(path: Path) -> np.ndarray:
    arr = tifffile.imread(str(path))
    if arr.ndim == 2:
        arr = arr[None, ...]
    elif arr.ndim == 3:
        # ImageJ/SpinalJ stacks are usually (Z, Y, X).
        pass
    elif arr.ndim == 4:
        # (Z, C, Y, X) or (C, Z, Y, X) — take first channel-like axis of size small.
        if arr.shape[1] <= 4 and arr.shape[0] > arr.shape[1]:
            arr = arr[:, 0]
        else:
            arr = arr[0]
    else:
        raise ValueError(f"Unexpected TIFF ndim={arr.ndim} shape={arr.shape} for {path}")
    return np.asarray(arr)


def convert_atlas_to_nifti(atlas_dir: Path, out_dir: Path) -> dict[str, Path]:
    import nibabel as nib

    files = find_atlas_files(atlas_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    template = load_tiff_stack_as_zyx(files["template"]).astype(np.float32)
    annotation = load_tiff_stack_as_zyx(files["annotation"])
    if annotation.dtype.kind == "f":
        annotation = np.rint(annotation).astype(np.int32)
    else:
        annotation = annotation.astype(np.int32)

    # ANTs/nibabel volumes are written as (X, Y, Z); our arrays are (Z, Y, X).
    sx, sy, sz = ATLAS_SPACING_XYZ_UM
    affine = np.diag([sx, sy, sz, 1.0])

    template_xyz = np.transpose(template, (2, 1, 0))
    annotation_xyz = np.transpose(annotation, (2, 1, 0))

    template_nii = out_dir / "Template.nii.gz"
    annotation_nii = out_dir / "Annotation.nii.gz"
    nib.save(nib.Nifti1Image(template_xyz, affine), str(template_nii))
    nib.save(nib.Nifti1Image(annotation_xyz, affine), str(annotation_nii))

    meta = {
        "atlas_dir": str(files["atlas_dir"]),
        "template_shape_zyx": list(template.shape),
        "annotation_shape_zyx": list(annotation.shape),
        "spacing_xyz_um": list(ATLAS_SPACING_XYZ_UM),
        "physical_size_xyz_mm": [
            template.shape[2] * sx / 1000.0,
            template.shape[1] * sy / 1000.0,
            template.shape[0] * sz / 1000.0,
        ],
    }
    (out_dir / "atlas_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    logger.info(
        "Atlas ZYX=%s spacing_xyz=%s physical_mm=%s",
        template.shape,
        ATLAS_SPACING_XYZ_UM,
        meta["physical_size_xyz_mm"],
    )
    return {"template": template_nii, "annotation": annotation_nii, "meta": out_dir / "atlas_meta.json"}


def list_channel_tiffs(tiff_dir: Path, channel_token: str = "_C0_") -> list[Path]:
    files = sorted(p for p in tiff_dir.glob("*.tif*") if channel_token in p.name)
    if not files:
        files = sorted(tiff_dir.glob("*.tif*"))
    if not files:
        raise FileNotFoundError(f"No TIFF slices in {tiff_dir}")
    return files


def _resize_yx_preserve_peaks(
    img: np.ndarray,
    *,
    out_w: int,
    out_h: int,
    peak_blend: float = 0.65,
) -> np.ndarray:
    """Downsample a 2D plane while keeping bright landmarks visible.

    INTER_AREA alone averages sparse bright spots into neighbors and makes
    landmark features hard to see. Blend area (anti-aliased mean) with a
    max-pool nearest resize so peaks survive without fully abandoning
    anti-aliasing for registration.
    """
    import cv2

    img_f = np.asarray(img, dtype=np.float32)
    mean_ds = cv2.resize(img_f, (out_w, out_h), interpolation=cv2.INTER_AREA)
    if peak_blend <= 0.0:
        return mean_ds

    # Integer-ish max pool when reducing a lot; then nearest to exact size.
    h0, w0 = img_f.shape
    fy = max(int(round(h0 / out_h)), 1)
    fx = max(int(round(w0 / out_w)), 1)
    if fy > 1 or fx > 1:
        # Trim to whole blocks so reshape-max is exact.
        h_use = (h0 // fy) * fy
        w_use = (w0 // fx) * fx
        block = img_f[:h_use, :w_use].reshape(h_use // fy, fy, w_use // fx, fx).max(axis=(1, 3))
        peak_ds = cv2.resize(block, (out_w, out_h), interpolation=cv2.INTER_NEAREST)
    else:
        peak_ds = cv2.resize(img_f, (out_w, out_h), interpolation=cv2.INTER_LINEAR)

    blend = float(np.clip(peak_blend, 0.0, 1.0))
    return (1.0 - blend) * mean_ds + blend * np.maximum(mean_ds, peak_ds)


def downsample_reorient_lsfm_stack(
    tiff_dir: Path,
    out_nii: Path,
    *,
    source_spacing_xyz_um: tuple[float, float, float],
    target_spacing_xyz_um: tuple[float, float, float],
    channel_token: str = "_C0_",
    # Sample array axes before reorient: (Z_stack, Y_long, X_lateral)
    # Default maps long in-plane axis -> atlas Z (rostrocaudal).
    permute_zyx_to_atlas_zyx: tuple[int, int, int] = (1, 0, 2),
    max_rc_mm: float | None = None,
    preserve_peaks: bool = True,
    peak_blend: float = 0.65,
) -> dict:
    """Stream-downsample LSFM TIFF stack and reorient into atlas-like ZYX.

    Default assumption for MegaSpim WT jisui:
      - each TIFF is (Y, X) = (long RC axis ~35 mm, lateral ~6.7 mm)
      - stack Z is the other transverse axis (~2.9 mm)
      - SpinalJ atlas is coronal: Z=RC (~31 mm), Y=~2.5 mm, X=~3.2 mm
    Default permute (1,0,2): atlas_z<-sample_y, atlas_y<-sample_z, atlas_x<-sample_x

    Intensity is kept as float32 (not 8-bit). With preserve_peaks=True (default),
    Z uses max-reduce over each slice window and XY blends area+peak so sparse
    bright landmarks stay visible after downsampling.
    """
    import nibabel as nib
    from scipy import ndimage

    files = list_channel_tiffs(tiff_dir, channel_token=channel_token)
    first = tifffile.imread(str(files[0]))
    if first.ndim != 2:
        raise ValueError(f"Expected 2D slices, got shape {first.shape}")

    n_z, y0, x0 = len(files), int(first.shape[0]), int(first.shape[1])
    src_x, src_y, src_z = source_spacing_xyz_um
    # Source spacing for array axes (Z, Y, X): stack-Z, row-Y, col-X
    src_spacing_zyx = np.array([src_z, src_y, src_x], dtype=np.float64)
    tgt_spacing_zyx_atlas = np.array(
        [target_spacing_xyz_um[2], target_spacing_xyz_um[1], target_spacing_xyz_um[0]],
        dtype=np.float64,
    )

    logger.info("Input slices=%d shape_YX=%sx%s spacing_xyz_um=%s", n_z, y0, x0, source_spacing_xyz_um)

    # Keep atlas-like target spacing after permute: build an intermediate volume in
    # sample ZYX at approximately isotropic target spacing, then permute.
    # Use the minimum target spacing for intermediate axes so later permute is accurate.
    # z_step = how many source slices per output Z plane so stack-Z spacing ≈ tgt_iso.
    tgt_iso = float(min(target_spacing_xyz_um))
    z_step = max(int(round(tgt_iso / src_spacing_zyx[0])), 1)
    y_out = max(int(round(y0 * src_spacing_zyx[1] / tgt_iso)), 1)
    x_out = max(int(round(x0 * src_spacing_zyx[2] / tgt_iso)), 1)
    z_starts = list(range(0, n_z, z_step))
    effective_z_spacing = float(src_spacing_zyx[0] * z_step)

    logger.info(
        "Intermediate iso≈%.2fum (stack-Z effective=%.2fum) out_ZYX~=%d x %d x %d "
        "(z_step=%d, preserve_peaks=%s peak_blend=%.2f)",
        tgt_iso,
        effective_z_spacing,
        len(z_starts),
        y_out,
        x_out,
        z_step,
        preserve_peaks,
        peak_blend if preserve_peaks else 0.0,
    )

    import cv2

    slices: list[np.ndarray] = []
    for i, z0 in enumerate(z_starts):
        z1 = min(z0 + z_step, n_z)
        if preserve_peaks and (z1 - z0) > 1:
            # Max across the Z window so bright landmarks in skipped slices survive.
            acc = None
            for zi in range(z0, z1):
                img = tifffile.imread(str(files[zi]))
                if img.shape != (y0, x0):
                    raise ValueError(f"Slice shape mismatch at {files[zi]}: {img.shape}")
                img_f = np.asarray(img, dtype=np.float32)
                acc = img_f if acc is None else np.maximum(acc, img_f)
            plane = acc
        else:
            plane = tifffile.imread(str(files[z0]))
            if plane.shape != (y0, x0):
                raise ValueError(f"Slice shape mismatch at {files[z0]}: {plane.shape}")

        if preserve_peaks:
            resized = _resize_yx_preserve_peaks(
                plane, out_w=x_out, out_h=y_out, peak_blend=peak_blend
            )
        else:
            resized = cv2.resize(
                np.asarray(plane, dtype=np.float32),
                (x_out, y_out),
                interpolation=cv2.INTER_AREA,
            )
        slices.append(np.asarray(resized, dtype=np.float32))
        if (i + 1) % 50 == 0 or i + 1 == len(z_starts):
            logger.info("  read/resize %d/%d", i + 1, len(z_starts))

    vol_zyx = np.stack(slices, axis=0)
    del slices

    # Optional crop along long axis (array axis 1 = Y) before reorient.
    if max_rc_mm is not None:
        max_y = int(round((max_rc_mm * 1000.0) / tgt_iso))
        if vol_zyx.shape[1] > max_y:
            start = (vol_zyx.shape[1] - max_y) // 2
            vol_zyx = vol_zyx[:, start : start + max_y, :]
            logger.info("Cropped RC axis to %d voxels (%.2f mm)", max_y, max_rc_mm)

    # Intermediate spacings in sample ZYX (stack-Z may be slightly off tgt_iso due to integer step).
    spacing_sample_zyx = np.array([effective_z_spacing, tgt_iso, tgt_iso], dtype=np.float64)

    # Reorient sample ZYX -> atlas-like ZYX via permute.
    # permute_zyx_to_atlas_zyx[i] = source axis for output axis i.
    vol_atlas_zyx = np.transpose(vol_zyx, permute_zyx_to_atlas_zyx)
    del vol_zyx
    spacing_after = spacing_sample_zyx[list(permute_zyx_to_atlas_zyx)]

    # Resample to exact atlas target spacing (Z,Y,X of atlas).
    # order=1 keeps edges sharper than cubic blur; peaks already boosted upstream.
    zoom = spacing_after / tgt_spacing_zyx_atlas
    if not np.allclose(zoom, 1.0, rtol=0.02, atol=0.02):
        logger.info("Zoom to atlas spacing with factors ZYX=%s (from spacing %s)", zoom, spacing_after)
        vol_atlas_zyx = ndimage.zoom(vol_atlas_zyx, zoom=zoom, order=1).astype(np.float32)

    out_nii.parent.mkdir(parents=True, exist_ok=True)
    sx, sy, sz = target_spacing_xyz_um
    affine = np.diag([sx, sy, sz, 1.0])
    vol_xyz = np.transpose(vol_atlas_zyx, (2, 1, 0))
    nib.save(nib.Nifti1Image(vol_xyz, affine), str(out_nii))

    meta = {
        "input_dir": str(tiff_dir),
        "n_input_slices": n_z,
        "input_yx": [y0, x0],
        "channel_token": channel_token,
        "source_spacing_xyz_um": list(source_spacing_xyz_um),
        "target_spacing_xyz_um": list(target_spacing_xyz_um),
        "permute_zyx_to_atlas_zyx": list(permute_zyx_to_atlas_zyx),
        "preserve_peaks": bool(preserve_peaks),
        "peak_blend": float(peak_blend) if preserve_peaks else 0.0,
        "z_reduce": "max" if preserve_peaks else "stride",
        "output_shape_zyx": list(vol_atlas_zyx.shape),
        "output_physical_mm_xyz": [
            vol_atlas_zyx.shape[2] * sx / 1000.0,
            vol_atlas_zyx.shape[1] * sy / 1000.0,
            vol_atlas_zyx.shape[0] * sz / 1000.0,
        ],
        "output_dtype": "float32",
        "output_nii": str(out_nii),
    }
    # volume.nii.gz -> volume_meta.json
    meta_path = out_nii.parent / (out_nii.name.replace(".nii.gz", "") + "_meta.json")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    logger.info(
        "Wrote %s shape_zyx=%s dtype=float32 physical_mm=%s",
        out_nii,
        vol_atlas_zyx.shape,
        meta["output_physical_mm_xyz"],
    )
    return meta


def make_tissue_mask_zyx(
    vol_zyx: np.ndarray,
    *,
    smooth_sigma: float = 1.5,
    thr_frac_of_p99: float = 0.05,
    close_radius: int = 3,
    open_radius: int = 1,
) -> np.ndarray:
    """Binary tissue mask from intensity (largest connected component)."""
    from scipy import ndimage

    vol = np.asarray(vol_zyx, dtype=np.float32)
    smooth = ndimage.gaussian_filter(vol, sigma=smooth_sigma)
    p99 = float(np.percentile(smooth, 99.0))
    thr = max(p99 * thr_frac_of_p99, 1e-6)
    mask = smooth > thr

    if close_radius > 0:
        mask = ndimage.binary_closing(mask, structure=np.ones((close_radius,) * 3))
    if open_radius > 0:
        mask = ndimage.binary_opening(mask, structure=np.ones((open_radius,) * 3))

    labeled, nlab = ndimage.label(mask)
    if nlab == 0:
        logger.warning("Tissue mask empty at thr=%.4g (p99=%.4g); using full FOV", thr, p99)
        return np.ones(vol.shape, dtype=bool)
    counts = np.bincount(labeled.ravel())
    counts[0] = 0
    keep = int(np.argmax(counts))
    mask = labeled == keep
    # Fill holes slice-wise along RC (Z) for a cleaner cord outline.
    for z in range(mask.shape[0]):
        mask[z] = ndimage.binary_fill_holes(mask[z])
    logger.info(
        "Tissue mask thr=%.4g (%.1f%% of p99) fg_frac=%.3f",
        thr,
        100.0 * thr_frac_of_p99,
        float(mask.mean()),
    )
    return mask


def mask_and_crop_volume_nii(
    volume_nii: Path,
    out_dir: Path,
    *,
    thr_frac_of_p99: float = 0.05,
    pad_vox: int = 8,
) -> dict[str, Path]:
    """Zero outside tissue, crop to mask bbox, write volume+mask NIfTIs."""
    import nibabel as nib

    out_dir.mkdir(parents=True, exist_ok=True)
    nii = nib.load(str(volume_nii))
    vol_xyz = np.asanyarray(nii.dataobj).astype(np.float32)
    affine = nii.affine.copy()
    spacing = tuple(float(abs(affine[i, i])) for i in range(3))

    vol_zyx = np.transpose(vol_xyz, (2, 1, 0))
    mask_zyx = make_tissue_mask_zyx(vol_zyx, thr_frac_of_p99=thr_frac_of_p99)
    vol_zyx = vol_zyx * mask_zyx.astype(np.float32)

    zs, ys, xs = np.where(mask_zyx)
    z0, z1 = int(zs.min()), int(zs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    z0 = max(0, z0 - pad_vox)
    y0 = max(0, y0 - pad_vox)
    x0 = max(0, x0 - pad_vox)
    z1 = min(mask_zyx.shape[0], z1 + pad_vox)
    y1 = min(mask_zyx.shape[1], y1 + pad_vox)
    x1 = min(mask_zyx.shape[2], x1 + pad_vox)

    vol_zyx = vol_zyx[z0:z1, y0:y1, x0:x1]
    mask_zyx = mask_zyx[z0:z1, y0:y1, x0:x1]

    # Keep origin shifted in affine (XYZ order).
    new_affine = affine.copy()
    new_affine[0, 3] = affine[0, 3] + x0 * spacing[0]
    new_affine[1, 3] = affine[1, 3] + y0 * spacing[1]
    new_affine[2, 3] = affine[2, 3] + z0 * spacing[2]

    vol_xyz = np.transpose(vol_zyx, (2, 1, 0))
    mask_xyz = np.transpose(mask_zyx.astype(np.uint8), (2, 1, 0))

    out_vol = out_dir / "volume_masked.nii.gz"
    out_mask = out_dir / "tissue_mask.nii.gz"
    nib.save(nib.Nifti1Image(vol_xyz, new_affine), str(out_vol))
    nib.save(nib.Nifti1Image(mask_xyz, new_affine), str(out_mask))

    # QC mid-slices
    qc_dir = out_dir / "mask_qc"
    qc_dir.mkdir(exist_ok=True)
    zmid = vol_zyx.shape[0] // 2
    ymid = vol_zyx.shape[1] // 2
    tifffile.imwrite(qc_dir / "masked_zmid.tif", _to_u16(vol_zyx[zmid]))
    tifffile.imwrite(qc_dir / "mask_zmid.tif", (mask_zyx[zmid].astype(np.uint8) * 255))
    tifffile.imwrite(qc_dir / "masked_ymid.tif", _to_u16(vol_zyx[:, ymid, :]))
    _write_qc_png_pair(qc_dir / "mask_overlay_zmid.png", vol_zyx[zmid], mask_zyx[zmid].astype(np.float32) * float(np.percentile(vol_zyx, 99) + 1))

    meta = {
        "source_volume": str(volume_nii),
        "crop_zyx": [[z0, z1], [y0, y1], [x0, x1]],
        "shape_zyx_before": list(np.transpose(np.asanyarray(nii.dataobj), (2, 1, 0)).shape),
        "shape_zyx_after": list(vol_zyx.shape),
        "physical_mm_xyz_after": [
            vol_zyx.shape[2] * spacing[0] / 1000.0,
            vol_zyx.shape[1] * spacing[1] / 1000.0,
            vol_zyx.shape[0] * spacing[2] / 1000.0,
        ],
        "fg_frac": float(mask_zyx.mean()),
        "thr_frac_of_p99": thr_frac_of_p99,
        "pad_vox": pad_vox,
    }
    (out_dir / "mask_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    logger.info(
        "Masked+cropped volume ZYX %s -> %s physical_mm=%s",
        meta["shape_zyx_before"],
        meta["shape_zyx_after"],
        meta["physical_mm_xyz_after"],
    )
    return {"volume": out_vol, "mask": out_mask, "meta": out_dir / "mask_meta.json"}


def load_landmark_pairs(landmarks_json: Path) -> list[dict]:
    """Return complete sample/atlas voxel pairs from landmark UI JSON."""
    payload = json.loads(Path(landmarks_json).read_text(encoding="utf-8"))
    pairs: list[dict] = []
    for item in payload.get("landmarks", []):
        sample = item.get("sample") or {}
        atlas = item.get("atlas") or {}
        if not all(k in sample and k in atlas for k in ("x", "y", "z")):
            continue
        pairs.append(
            {
                "id": str(item.get("id", "")),
                "label": str(item.get("label", item.get("id", ""))),
                "sample": {k: float(sample[k]) for k in ("x", "y", "z")},
                "atlas": {k: float(atlas[k]) for k in ("x", "y", "z")},
            }
        )
    if len(pairs) < 2:
        raise ValueError(f"Need >=2 complete landmark pairs in {landmarks_json}, got {len(pairs)}")
    return pairs


def _scale_landmark_z(
    z: float,
    *,
    src_n: int | None,
    dst_n: int | None,
) -> float:
    if not src_n or not dst_n or src_n == dst_n or src_n <= 1 or dst_n <= 1:
        return float(z)
    return float(z) * float(dst_n - 1) / float(src_n - 1)


def _zyx_vox_to_phys(img, x: float, y: float, z: float) -> np.ndarray:
    """Convert landmark UI ZYX voxels to ANTs physical XYZ (spacing units)."""
    index = np.array([float(x), float(y), float(z)], dtype=np.float64)
    spacing = np.asarray(img.spacing, dtype=np.float64)
    origin = np.asarray(img.origin, dtype=np.float64)
    direction = np.asarray(img.direction, dtype=np.float64).reshape(3, 3)
    return origin + direction @ (index * spacing)


def write_landmark_initial_transform(
    *,
    landmarks_json: Path,
    fixed_img,
    moving_img,
    out_path: Path,
    transform_type: str = "similarity",
) -> dict:
    """Fit moving(atlas)→fixed(sample) linear transform from landmark pairs."""
    import ants

    payload = json.loads(Path(landmarks_json).read_text(encoding="utf-8"))
    pairs = load_landmark_pairs(landmarks_json)
    src_z = None
    shape = payload.get("sample_shape_zyx") or []
    if len(shape) >= 1:
        src_z = int(shape[0])
    dst_z = int(fixed_img.shape[2])
    src_atlas_z = None
    atlas_shape = payload.get("atlas_shape_zyx") or []
    if len(atlas_shape) >= 1:
        src_atlas_z = int(atlas_shape[0])
    dst_atlas_z = int(moving_img.shape[2])

    moving_pts = []
    fixed_pts = []
    used = []
    for pair in pairs:
        sx, sy, sz = pair["sample"]["x"], pair["sample"]["y"], pair["sample"]["z"]
        ax, ay, az = pair["atlas"]["x"], pair["atlas"]["y"], pair["atlas"]["z"]
        sz = _scale_landmark_z(sz, src_n=src_z, dst_n=dst_z)
        az = _scale_landmark_z(az, src_n=src_atlas_z, dst_n=dst_atlas_z)
        fixed_pts.append(_zyx_vox_to_phys(fixed_img, sx, sy, sz))
        moving_pts.append(_zyx_vox_to_phys(moving_img, ax, ay, az))
        used.append(
            {
                "id": pair["id"],
                "label": pair["label"],
                "sample_xyz_vox": [sx, sy, sz],
                "atlas_xyz_vox": [ax, ay, az],
                "sample_phys": [float(v) for v in fixed_pts[-1]],
                "atlas_phys": [float(v) for v in moving_pts[-1]],
            }
        )

    moving_pts = np.vstack(moving_pts)
    fixed_pts = np.vstack(fixed_pts)
    tx_type = str(transform_type).strip().lower()
    if tx_type not in ("rigid", "similarity", "affine"):
        raise ValueError("landmark transform_type must be rigid, similarity, or affine")
    logger.info(
        "Fitting landmark %s from %d pairs (sample Z %s→%s)",
        tx_type,
        len(used),
        src_z,
        dst_z,
    )
    tx = ants.fit_transform_to_paired_points(
        moving_pts,
        fixed_pts,
        transform_type=tx_type,
    )
    if isinstance(tx, (list, tuple)):
        tx = tx[0]
    # fit_transform_to_paired_points returns fixed→moving; ANTs -r wants moving→fixed.
    tx = ants.invert_ants_transform(tx)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ants.write_transform(tx, str(out_path))

    mapped = np.array(
        [ants.apply_ants_transform_to_point(tx, list(p)) for p in moving_pts],
        dtype=np.float64,
    )
    err = mapped - fixed_pts
    rmse = float(np.sqrt(np.mean(np.sum(err * err, axis=1))))
    rmse_z = float(np.sqrt(np.mean((err[:, 2]) ** 2)))
    meta = {
        "landmarks_json": str(landmarks_json),
        "transform_type": tx_type,
        "n_pairs": len(used),
        "pairs": used,
        "rmse_mm": rmse / 1000.0 if rmse > 50 else rmse,
        "rmse_phys": rmse,
        "rmse_z_phys": rmse_z,
        "transform": str(out_path),
        "sample_z_scaled": src_z not in (None, dst_z),
    }
    logger.info(
        "Landmark init RMSE=%.3f (Z=%.3f) → %s",
        rmse,
        rmse_z,
        out_path,
    )
    return meta


def _flip_ants_volume_z(img):
    import ants

    data = np.ascontiguousarray(img.numpy()[:, :, ::-1])
    return ants.from_numpy(
        data.astype(np.float32, copy=False),
        spacing=img.spacing,
        origin=img.origin,
        direction=np.asarray(img.direction, dtype=np.float64).copy(),
    )


def _landmarks_need_moving_z_flip(landmarks_json: Path) -> bool:
    payload = json.loads(Path(landmarks_json).read_text(encoding="utf-8"))
    sz, az = [], []
    for item in payload.get("landmarks", []):
        sample = item.get("sample") or {}
        atlas = item.get("atlas") or {}
        if "z" in sample and "z" in atlas:
            sz.append(float(sample["z"]))
            az.append(float(atlas["z"]))
    if len(sz) < 2:
        return False
    a = np.asarray(az, dtype=np.float64)
    s = np.asarray(sz, dtype=np.float64)
    A = np.column_stack([a, np.ones_like(a)])
    scale, _shift = np.linalg.lstsq(A, s, rcond=None)[0]
    return bool(scale < 0)


def _rewrite_landmarks_after_moving_z_flip(
    landmarks_json: Path,
    *,
    moving_z_n: int,
    out_path: Path,
) -> Path:
    payload = json.loads(Path(landmarks_json).read_text(encoding="utf-8"))
    n = int(moving_z_n)
    for item in payload.get("landmarks", []):
        atlas = item.get("atlas") or {}
        if "z" in atlas:
            atlas["z"] = float((n - 1) - float(atlas["z"]))
            item["atlas"] = atlas
    payload["moving_z_flipped_to_match_landmarks"] = True
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path


def run_ants_registration(
    fixed_nii: Path,
    moving_template_nii: Path,
    moving_annotation_nii: Path,
    out_dir: Path,
    *,
    transform: str = "SyNRA",
    fixed_mask_nii: Path | None = None,
    allow_reflection: bool = False,
    direction_y: float = 1.0,
    landmarks_json: Path | None = None,
    landmark_transform_type: str = "similarity",
) -> dict:
    import ants
    import nibabel as nib

    out_dir.mkdir(parents=True, exist_ok=True)
    # Sample (fixed) direction: diag(1, direction_y, 1). Default +1 = no Y flip vs atlas.
    # Pass direction_y=-1 only if sample appears mirrored along Y relative to atlas.
    fixed_direction = np.diag([1.0, float(direction_y), 1.0]).astype(np.float64)
    moving_direction = np.eye(3, dtype=np.float64)

    def load_nii(
        path: Path,
        *,
        as_float: bool = True,
        direction: np.ndarray,
    ) -> "ants.ANTsImage":
        nii = nib.load(str(path))
        data = np.asanyarray(nii.dataobj)
        if as_float and not np.issubdtype(data.dtype, np.floating):
            data = data.astype(np.float32)
        elif not as_float:
            data = (data > 0).astype(np.float32)
        spacing = tuple(float(abs(nii.affine[i, i])) for i in range(3))
        data_f = data.astype(np.float32, copy=False)
        # Keep physical FOV on the +Y side when direction_yy is negative.
        origin = [0.0, 0.0, 0.0]
        if float(direction[1, 1]) < 0:
            origin[1] = float((data_f.shape[1] - 1) * spacing[1])
        return ants.from_numpy(
            data_f,
            spacing=spacing,
            origin=tuple(origin),
            direction=np.asarray(direction, dtype=np.float64).copy(),
        )

    fixed = load_nii(fixed_nii, direction=fixed_direction)
    moving = load_nii(moving_template_nii, direction=moving_direction)
    label = load_nii(moving_annotation_nii, direction=moving_direction)
    flipped_moving_z = False
    lm_path = Path(landmarks_json) if landmarks_json is not None else None
    if lm_path is not None:
        if not lm_path.exists():
            raise FileNotFoundError(f"landmarks_json not found: {lm_path}")
        if _landmarks_need_moving_z_flip(lm_path):
            logger.info(
                "Landmark Z mapping is reversed (sample high-Z ↔ atlas low-Z); "
                "flipping moving atlas along Z before registration"
            )
            moving = _flip_ants_volume_z(moving)
            label = _flip_ants_volume_z(label)
            lm_path = _rewrite_landmarks_after_moving_z_flip(
                lm_path,
                moving_z_n=int(moving.shape[2]),
                out_path=out_dir / "landmarks_after_moving_z_flip.json",
            )
            flipped_moving_z = True

    fixed_mask = (
        load_nii(fixed_mask_nii, as_float=False, direction=fixed_direction)
        if fixed_mask_nii is not None
        else None
    )
    # Atlas foreground mask helps ignore empty template margins.
    moving_mask = ants.threshold_image(moving, 1e-6, 1e12, 1, 0)

    logger.info("Fixed shape/spacing=%s/%s direction=\n%s", fixed.shape, fixed.spacing, fixed.direction)
    logger.info("Moving shape/spacing=%s/%s direction=\n%s", moving.shape, moving.spacing, moving.direction)
    if fixed_mask is not None:
        logger.info("Using fixed tissue mask fg_frac=%.3f", float(np.mean(fixed_mask.numpy() > 0)))

    landmark_meta = None
    initial_transform = None
    if lm_path is not None:
        landmark_meta = write_landmark_initial_transform(
            landmarks_json=lm_path,
            fixed_img=fixed,
            moving_img=moving,
            out_path=out_dir / "landmark_init.mat",
            transform_type=landmark_transform_type,
        )
        landmark_meta["moving_z_flipped"] = bool(flipped_moving_z)
        initial_transform = landmark_meta["transform"]

    fixed_matched = ants.histogram_match_image(fixed, moving)
    # SyNRA = Rigid + Affine + SyN: rigid stage can recover ~180° in-plane rotations.
    logger.info(
        "Running ANTs registration type=%s (atlas -> image, metric=mattes MI, "
        "aff_do_reflection=%s, direction_y=%s, landmark_init=%s)",
        transform,
        allow_reflection,
        direction_y,
        initial_transform,
    )
    reg_kwargs = dict(
        fixed=fixed_matched,
        moving=moving,
        type_of_transform=transform,
        aff_do_reflection=bool(allow_reflection),
        moving_mask=moving_mask,
    )
    if initial_transform is not None:
        reg_kwargs["initial_transform"] = initial_transform
    if fixed_mask is not None:
        reg_kwargs["mask"] = fixed_mask
    reg = ants.registration(**reg_kwargs)

    warped_template = reg["warpedmovout"]
    warped_label = ants.apply_transforms(
        fixed=fixed_matched,
        moving=label,
        transformlist=reg["fwdtransforms"],
        interpolator="nearestNeighbor",
    )

    warped_template_path = out_dir / "warped_template.nii.gz"
    warped_label_path = out_dir / "warped_annotation.nii.gz"
    ants.image_write(warped_template, str(warped_template_path))
    ants.image_write(warped_label, str(warped_label_path))

    # Save a couple of QC mid-slices as TIFF for quick visual check.
    fixed_zyx = np.transpose(fixed_matched.numpy(), (2, 1, 0))
    warped_zyx = np.transpose(warped_template.numpy(), (2, 1, 0))
    zmid = fixed_zyx.shape[0] // 2
    ymid = fixed_zyx.shape[1] // 2
    qc_dir = out_dir / "qc_slices"
    qc_dir.mkdir(exist_ok=True)
    tifffile.imwrite(qc_dir / "fixed_zmid.tif", _to_u16(fixed_zyx[zmid]))
    tifffile.imwrite(qc_dir / "warped_template_zmid.tif", _to_u16(warped_zyx[zmid]))
    tifffile.imwrite(qc_dir / "fixed_ymid.tif", _to_u16(fixed_zyx[:, ymid, :]))
    tifffile.imwrite(qc_dir / "warped_template_ymid.tif", _to_u16(warped_zyx[:, ymid, :]))
    _write_qc_png_pair(qc_dir / "overlay_zmid.png", fixed_zyx[zmid], warped_zyx[zmid])
    _write_qc_png_pair(qc_dir / "overlay_ymid.png", fixed_zyx[:, ymid, :], warped_zyx[:, ymid, :])

    # Persist transform files if present.
    transforms_dir = out_dir / "transforms"
    transforms_dir.mkdir(exist_ok=True)
    saved_transforms = []
    for i, tpath in enumerate(reg.get("fwdtransforms", [])):
        src = Path(tpath)
        if src.exists():
            dst = transforms_dir / f"fwd_{i}_{src.name}"
            shutil.copy2(src, dst)
            saved_transforms.append(str(dst))

    summary = {
        "transform": transform,
        "allow_reflection": bool(allow_reflection),
        "direction_y": float(direction_y),
        "fixed_direction": fixed_direction.tolist(),
        "moving_direction": moving_direction.tolist(),
        "fixed": str(fixed_nii),
        "moving_template": str(moving_template_nii),
        "warped_template": str(warped_template_path),
        "warped_annotation": str(warped_label_path),
        "fwdtransforms": saved_transforms,
        "fixed_shape": list(fixed.shape),
        "moving_shape": list(moving.shape),
        "landmarks": landmark_meta,
    }
    (out_dir / "registration_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("Registration finished. QC slices in %s", qc_dir)
    return summary


def _to_u16(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    lo, hi = np.percentile(arr, (1, 99.5))
    if hi <= lo:
        hi = lo + 1.0
    scaled = np.clip((arr - lo) / (hi - lo), 0, 1)
    return (scaled * 65535.0).astype(np.uint16)


def _to_u8(arr: np.ndarray) -> np.ndarray:
    return (_to_u16(arr).astype(np.float32) / 65535.0 * 255.0).astype(np.uint8)


def _write_qc_png_pair(path: Path, fixed: np.ndarray, warped: np.ndarray) -> None:
    """Write an RGB overlay PNG: fixed=magenta, warped=green."""
    import cv2

    f = _to_u8(fixed)
    w = _to_u8(warped)
    if f.shape != w.shape:
        out_h = max(f.shape[0], w.shape[0])
        out_w = max(f.shape[1], w.shape[1])
        f2 = np.zeros((out_h, out_w), dtype=np.uint8)
        w2 = np.zeros((out_h, out_w), dtype=np.uint8)
        f2[: f.shape[0], : f.shape[1]] = f
        w2[: w.shape[0], : w.shape[1]] = w
        f, w = f2, w2
    # BGR for OpenCV: B=fixed, G=warped, R=fixed → magenta/green overlay.
    bgr = np.stack([f, w, f], axis=-1)
    cv2.imwrite(str(path), bgr)


def parse_float_triplet(text: str) -> tuple[float, float, float]:
    parts = [p.strip() for p in text.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("expected three comma-separated numbers")
    return float(parts[0]), float(parts[1]), float(parts[2])


def parse_int_triplet(text: str) -> tuple[int, int, int]:
    parts = [int(p.strip()) for p in text.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("expected three comma-separated ints")
    return parts[0], parts[1], parts[2]


def main() -> None:
    _configure_logging()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atlas_dir", required=True, help="SpinalJ SC_P56 atlas folder containing Template.tif")
    parser.add_argument("--sample_tiff_dir", required=True, help="Folder of TIFF slices (e.g. ch0_masked)")
    parser.add_argument("--out_dir", required=True, help="Output directory for this trial")
    parser.add_argument("--channel_token", default="_C0_", help="Substring used to pick registration channel TIFFs")
    parser.add_argument(
        "--source_spacing_xyz_um",
        type=parse_float_triplet,
        default=(0.71, 0.71, 2.0),
        help="Sample spacing X,Y,Z in microns",
    )
    parser.add_argument(
        "--target_spacing_xyz_um",
        type=parse_float_triplet,
        default=(10.0, 10.0, 20.0),
        help="Registration working spacing X,Y,Z in microns (default matches SpinalJ atlas native spacing)",
    )
    parser.add_argument(
        "--permute_zyx_to_atlas_zyx",
        type=parse_int_triplet,
        default=(1, 0, 2),
        help="Permute sample Z,Y,X axes into atlas-like Z,Y,X (default matches WT jisui physical axes)",
    )
    parser.add_argument(
        "--transform",
        default="SyNRA",
        help="ANTs type_of_transform: SyNRA (Rigid+Affine+SyN, default), SyN, Affine, Rigid, ...",
    )
    parser.add_argument(
        "--allow_reflection",
        action="store_true",
        help="Allow affine reflection (det=-1 flips). Off by default; use if LR appears mirrored.",
    )
    parser.add_argument(
        "--direction_y",
        type=float,
        default=1.0,
        help="ANTs direction matrix YY when loading fixed NIfTI (default +1 = no Y flip; use -1 to flip Y)",
    )
    parser.add_argument("--skip_downsample", action="store_true")
    parser.add_argument("--skip_register", action="store_true")
    parser.add_argument(
        "--no_preserve_peaks",
        action="store_true",
        help="Disable peak-preserving downsample (old INTER_AREA + Z stride only)",
    )
    parser.add_argument(
        "--peak_blend",
        type=float,
        default=0.65,
        help="XY blend of max-pool peaks vs area mean (0=area only, 1=strong peaks; default 0.65)",
    )
    parser.add_argument(
        "--apply_mask",
        action="store_true",
        help="Build tissue mask, zero outside, crop to bbox, and pass mask to ANTs",
    )
    parser.add_argument(
        "--mask_thr_frac",
        type=float,
        default=0.05,
        help="Mask threshold as fraction of intensity p99 (default 0.05)",
    )
    parser.add_argument("--mask_pad_vox", type=int, default=8, help="Padding voxels around mask bbox")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    atlas_nifti_dir = out_dir / "atlas_nifti"
    # Always write atlas at native 10x10x20, then ANTs will resample via spacing metadata.
    atlas_paths = convert_atlas_to_nifti(Path(args.atlas_dir), atlas_nifti_dir)

    sample_nii = out_dir / "sample_for_reg" / "volume.nii.gz"
    if args.skip_downsample and sample_nii.exists():
        logger.info("Reusing existing sample volume %s", sample_nii)
    else:
        downsample_reorient_lsfm_stack(
            Path(args.sample_tiff_dir),
            sample_nii,
            source_spacing_xyz_um=args.source_spacing_xyz_um,
            target_spacing_xyz_um=args.target_spacing_xyz_um,
            channel_token=args.channel_token,
            permute_zyx_to_atlas_zyx=args.permute_zyx_to_atlas_zyx,
            preserve_peaks=not bool(args.no_preserve_peaks),
            peak_blend=float(args.peak_blend),
        )

    fixed_mask_nii = None
    fixed_for_reg = sample_nii
    if args.apply_mask:
        mask_paths = mask_and_crop_volume_nii(
            sample_nii,
            out_dir / "sample_for_reg",
            thr_frac_of_p99=args.mask_thr_frac,
            pad_vox=args.mask_pad_vox,
        )
        fixed_for_reg = mask_paths["volume"]
        fixed_mask_nii = mask_paths["mask"]

    if not args.skip_register:
        # Resample atlas nifti to the working spacing used for the sample, so shapes are comparable.
        import ants
        import nibabel as nib

        work_dir = out_dir / "reg_work"
        work_dir.mkdir(exist_ok=True)
        sx, sy, sz = args.target_spacing_xyz_um

        def resample_to_spacing(src: Path, dst: Path, is_label: bool) -> Path:
            nii = nib.load(str(src))
            spacing = tuple(float(abs(nii.affine[i, i])) for i in range(3))
            data = np.asanyarray(nii.dataobj).astype(np.float32)
            img = ants.from_numpy(data, spacing=spacing)
            img.set_direction(np.eye(3))
            ref = ants.from_numpy(
                np.zeros(
                    (
                        max(1, int(round(img.shape[0] * spacing[0] / sx))),
                        max(1, int(round(img.shape[1] * spacing[1] / sy))),
                        max(1, int(round(img.shape[2] * spacing[2] / sz))),
                    ),
                    dtype=np.float32,
                ),
                spacing=(sx, sy, sz),
            )
            ref.set_direction(np.eye(3))
            interp = "nearestNeighbor" if is_label else "linear"
            out = ants.resample_image_to_target(img, ref, interp_type=interp)
            ants.image_write(out, str(dst))
            return dst

        template_work = resample_to_spacing(
            atlas_paths["template"], work_dir / "Template_work.nii.gz", is_label=False
        )
        annotation_work = resample_to_spacing(
            atlas_paths["annotation"], work_dir / "Annotation_work.nii.gz", is_label=True
        )

        run_ants_registration(
            fixed_for_reg,
            template_work,
            annotation_work,
            out_dir / "ants_out",
            transform=args.transform,
            fixed_mask_nii=fixed_mask_nii,
            allow_reflection=bool(args.allow_reflection),
            direction_y=float(args.direction_y),
        )


if __name__ == "__main__":
    main()
