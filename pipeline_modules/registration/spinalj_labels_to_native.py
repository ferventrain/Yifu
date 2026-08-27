#!/usr/bin/env python3
"""Warp SpinalJ vertebral segment labels back toward the original sample.

Pipeline (wizard registration outputs):
  1. Build a segment ID volume from Segments_Reference.csv (atlas Z → C1/T5/…)
  2. Apply saved ANTs fwd transforms → straightened+flipped space
  3. Undo flip_z/x/y
  4. Unstraighten along the same centerline frames → sample_for_reg space
  5. Optional inverse permute → LSFM stack-axis orientation at reg resolution

Does NOT upsample to native MegaSpim TIFF resolution (that would be huge);
overlays sit on the downsampled volume derived from the original stack.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def _configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


def _segment_sort_key(name: str) -> tuple:
    name = str(name).strip()
    prefix = "".join(c for c in name if c.isalpha()) or "Z"
    digits = "".join(c for c in name if c.isdigit()) or "0"
    order = {"C": 0, "T": 1, "L": 2, "S": 3, "Co": 4}.get(prefix, 9)
    return (order, int(digits), name)


def load_segment_legend(segments_csv: Path) -> tuple[list[str], dict[str, int], np.ndarray]:
    """Return (names_in_order, name→id, per-atlas-Z id array)."""
    names_z: list[str] = []
    with segments_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "Segment" not in reader.fieldnames:
            raise RuntimeError(f"Expected Segment column in {segments_csv}")
        for row in reader:
            names_z.append(str(row["Segment"]).strip())
    if not names_z:
        raise RuntimeError(f"No segments in {segments_csv}")
    unique = sorted(set(names_z), key=_segment_sort_key)
    name_to_id = {n: i + 1 for i, n in enumerate(unique)}
    z_ids = np.asarray([name_to_id[n] for n in names_z], dtype=np.uint16)
    return unique, name_to_id, z_ids


def build_segment_volume_xyz(
    template_work_nii: Path,
    z_ids: np.ndarray,
    *,
    thr: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """XYZ uint16 segment IDs, filled where template > thr. Affine from template."""
    import nibabel as nib

    nii = nib.load(str(template_work_nii))
    tmpl = np.asanyarray(nii.dataobj).astype(np.float32)
    if tmpl.ndim != 3:
        raise ValueError(f"Expected 3D template, got {tmpl.shape}")
    nx, ny, nz = tmpl.shape
    if len(z_ids) != nz:
        # Resample 1D segment axis to template Z count (nearest).
        src = np.linspace(0, len(z_ids) - 1, num=len(z_ids))
        dst = np.linspace(0, len(z_ids) - 1, num=nz)
        idx = np.clip(np.rint(np.interp(dst, src, src)).astype(int), 0, len(z_ids) - 1)
        z_use = z_ids[idx]
        logger.warning(
            "Segments_Reference length %d != template Z=%d; nearest-resampled Z axis",
            len(z_ids),
            nz,
        )
    else:
        z_use = z_ids
    seg = np.zeros(tmpl.shape, dtype=np.uint16)
    mask = tmpl > float(thr)
    for z in range(nz):
        if not np.any(mask[:, :, z]):
            continue
        seg[:, :, z][mask[:, :, z]] = z_use[z]
    return seg, nii.affine.copy()


def apply_ants_to_fixed(
    *,
    fixed_nii: Path,
    moving_xyz: np.ndarray,
    moving_affine: np.ndarray,
    fwdtransforms: list[str],
    fixed_direction: np.ndarray,
    moving_direction: np.ndarray,
) -> np.ndarray:
    """Return warped label as ZYX uint16 in fixed space."""
    import ants
    import nibabel as nib

    def load_fixed(path: Path) -> "ants.ANTsImage":
        nii = nib.load(str(path))
        data = np.asanyarray(nii.dataobj).astype(np.float32)
        spacing = tuple(float(abs(nii.affine[i, i])) for i in range(3))
        origin = [0.0, 0.0, 0.0]
        if float(fixed_direction[1, 1]) < 0:
            origin[1] = float((data.shape[1] - 1) * spacing[1])
        return ants.from_numpy(
            data,
            spacing=spacing,
            origin=tuple(origin),
            direction=np.asarray(fixed_direction, dtype=np.float64).copy(),
        )

    spacing = tuple(float(abs(moving_affine[i, i])) for i in range(3))
    moving = ants.from_numpy(
        moving_xyz.astype(np.float32),
        spacing=spacing,
        origin=(0.0, 0.0, 0.0),
        direction=np.asarray(moving_direction, dtype=np.float64).copy(),
    )
    fixed = load_fixed(fixed_nii)
    warped = ants.apply_transforms(
        fixed=fixed,
        moving=moving,
        transformlist=list(fwdtransforms),
        interpolator="nearestNeighbor",
    )
    # ANTs numpy is XYZ → ZYX
    return np.rint(np.transpose(warped.numpy(), (2, 1, 0))).astype(np.uint16)


def undo_flips_zyx(vol: np.ndarray, flip_meta: dict) -> np.ndarray:
    out = vol
    if flip_meta.get("flip_z"):
        out = out[::-1]
    if flip_meta.get("flip_y"):
        out = out[:, ::-1, :]
    if flip_meta.get("flip_x"):
        out = out[:, :, ::-1]
    return np.ascontiguousarray(out)


def unstraighten_labels_zyx(
    label_straight_zyx: np.ndarray,
    centerline_zyx: np.ndarray,
    target_shape_zyx: tuple[int, int, int],
    *,
    spacing_zyx_um: tuple[float, float, float],
    out_radius_yx_vox: int,
    step_um: float,
    smooth_centerline_um: float,
    tangent_smooth_window: int,
    frame_smooth_window: int,
) -> np.ndarray:
    """Scatter straightened labels back into curved sample_for_reg ZYX."""
    from pipeline_modules.registration.spinalj_straighten_register import (
        _build_smoothed_frames,
        _resample_centerline_uniform,
        _smooth_centerline_spline,
    )

    cl_in = np.asarray(centerline_zyx, dtype=np.float64)
    if smooth_centerline_um and smooth_centerline_um > 0:
        cl_in = _smooth_centerline_spline(cl_in, spacing_zyx_um, smooth_um=float(smooth_centerline_um))
    cl = _resample_centerline_uniform(cl_in, spacing_zyx_um, float(step_um))
    n_planes = len(cl)
    if label_straight_zyx.shape[0] != n_planes:
        raise RuntimeError(
            f"Straight label Z={label_straight_zyx.shape[0]} != rebuilt planes {n_planes}. "
            "Check centerline / straighten_meta match."
        )
    r = int(out_radius_yx_vox)
    sy, sx = spacing_zyx_um[1], spacing_zyx_um[2]
    pitch = 0.5 * (sy + sx)
    sp = np.asarray(spacing_zyx_um, dtype=np.float64)
    pts_um = cl * sp
    _tang, normals, binormals = _build_smoothed_frames(
        pts_um,
        tangent_smooth_window=int(tangent_smooth_window),
        frame_smooth_window=int(frame_smooth_window),
    )

    out = np.zeros(target_shape_zyx, dtype=np.uint16)
    zz, yy, xx = target_shape_zyx
    for i in range(n_planes):
        plane = label_straight_zyx[i]
        ys, xs = np.nonzero(plane)
        if ys.size == 0:
            continue
        vals = plane[ys, xs].astype(np.uint16, copy=False)
        # plane coords: iy in [0,2r), mapped to off = (iy-r)*pitch
        off_y_um = (ys.astype(np.float64) - r) * pitch
        off_x_um = (xs.astype(np.float64) - r) * pitch
        n_hat = normals[i]
        b_hat = binormals[i]
        c_um = pts_um[i]
        oz = (c_um[0] + n_hat[0] * off_y_um + b_hat[0] * off_x_um) / sp[0]
        oy = (c_um[1] + n_hat[1] * off_y_um + b_hat[1] * off_x_um) / sp[1]
        ox = (c_um[2] + n_hat[2] * off_y_um + b_hat[2] * off_x_um) / sp[2]
        zi = np.rint(oz).astype(np.int32)
        yi = np.rint(oy).astype(np.int32)
        xi = np.rint(ox).astype(np.int32)
        m = (zi >= 0) & (zi < zz) & (yi >= 0) & (yi < yy) & (xi >= 0) & (xi < xx)
        if not np.any(m):
            continue
        out[zi[m], yi[m], xi[m]] = vals[m]
        if (i + 1) % 200 == 0 or i + 1 == n_planes:
            logger.info("  unstraighten plane %d/%d (labeled px=%d)", i + 1, n_planes, int(ys.size))
    return out


def inverse_permute_to_lsfm(
    vol_atlas_zyx: np.ndarray,
    permute_zyx_to_atlas_zyx: tuple[int, int, int],
) -> np.ndarray:
    """Undo sample→atlas permute used in downsample_reorient_lsfm_stack."""
    p = tuple(int(x) for x in permute_zyx_to_atlas_zyx)
    inv = [0, 0, 0]
    for out_ax, src_ax in enumerate(p):
        inv[src_ax] = out_ax
    return np.transpose(vol_atlas_zyx, tuple(inv))


def _to_u8(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    pos = arr[arr > 0]
    if pos.size == 0:
        return np.zeros(arr.shape, dtype=np.uint8)
    lo, hi = np.percentile(pos, (1, 99.5))
    if hi <= lo:
        hi = lo + 1.0
    return (np.clip((arr - lo) / (hi - lo), 0, 1) * 255.0).astype(np.uint8)


def write_qc_overlays(
    qc_dir: Path,
    sample_zyx: np.ndarray,
    seg_zyx: np.ndarray,
    id_to_name: dict[int, str],
) -> None:
    qc_dir.mkdir(parents=True, exist_ok=True)
    # Distinct colors for up to ~40 segments.
    rng = np.random.default_rng(0)
    max_id = int(seg_zyx.max()) if seg_zyx.size else 0
    lut = np.zeros((max_id + 1, 3), dtype=np.uint8)
    if max_id > 0:
        lut[1:] = rng.integers(40, 255, size=(max_id, 3), dtype=np.uint8)

    def colorize(lab2d: np.ndarray, gray2d: np.ndarray) -> np.ndarray:
        g = _to_u8(gray2d)
        rgb = np.stack([g, g, g], axis=-1)
        m = lab2d > 0
        if np.any(m):
            cols = lut[lab2d[m]]
            rgb[m] = (0.35 * rgb[m].astype(np.float32) + 0.65 * cols.astype(np.float32)).astype(np.uint8)
        return rgb

    zmid = sample_zyx.shape[0] // 2
    ymid = sample_zyx.shape[1] // 2
    cv2.imwrite(str(qc_dir / "overlay_zmid_bgr.png"), colorize(seg_zyx[zmid], sample_zyx[zmid])[:, :, ::-1])
    cv2.imwrite(
        str(qc_dir / "overlay_ymid_bgr.png"),
        colorize(seg_zyx[:, ymid, :], sample_zyx[:, ymid, :])[:, :, ::-1],
    )
    # Sagittal MIP: intensity max along X; label = first nonzero along X (fast proxy).
    mip = sample_zyx.max(axis=2)
    ribbon = np.max(seg_zyx, axis=2).astype(np.uint16)
    step_z = max(1, mip.shape[0] // 800)
    step_y = max(1, mip.shape[1] // 400)
    cv2.imwrite(
        str(qc_dir / "overlay_sagittal_mip_bgr.png"),
        colorize(ribbon[::step_z, ::step_y], mip[::step_z, ::step_y])[:, :, ::-1],
    )
    # Legend image
    names = [id_to_name[i] for i in range(1, max_id + 1) if i in id_to_name]
    legend = np.zeros((max(1, 24 * len(names)), 220, 3), dtype=np.uint8)
    for i, name in enumerate(names):
        sid = next(k for k, v in id_to_name.items() if v == name)
        y0 = i * 24
        legend[y0 : y0 + 20, 8:28] = lut[sid]
        cv2.putText(
            legend,
            f"{sid:02d} {name}",
            (36, y0 + 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )
    cv2.imwrite(str(qc_dir / "segment_legend_bgr.png"), legend)


def save_nii_zyx(path: Path, vol_zyx: np.ndarray, spacing_zyx_um: tuple[float, float, float]) -> None:
    import nibabel as nib

    # Store as XYZ with affine diag(sx,sy,sz) matching existing spinalj convention.
    sx, sy, sz = float(spacing_zyx_um[2]), float(spacing_zyx_um[1]), float(spacing_zyx_um[0])
    vol_xyz = np.transpose(vol_zyx, (2, 1, 0))
    affine = np.diag([sx, sy, sz, 1.0])
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(vol_xyz.astype(np.uint16), affine), str(path))


def load_nii_as_zyx_u16(path: Path) -> np.ndarray:
    import nibabel as nib

    xyz = np.asanyarray(nib.load(str(path)).dataobj)
    return np.rint(np.transpose(xyz, (2, 1, 0))).astype(np.uint16)


def midres_shape_from_volume_meta(vmeta: dict) -> tuple[tuple[int, int, int], dict]:
    """LSFM stack-axis shape at ~target iso µm (before atlas spacing zoom)."""
    n_z = int(vmeta["n_input_slices"])
    y0, x0 = [int(v) for v in vmeta["input_yx"]]
    src_x, src_y, src_z = [float(v) for v in vmeta["source_spacing_xyz_um"]]
    tgt = [float(v) for v in vmeta["target_spacing_xyz_um"]]
    tgt_iso = float(min(tgt))
    z_step = max(int(round(tgt_iso / src_z)), 1)
    y_out = max(int(round(y0 * src_y / tgt_iso)), 1)
    x_out = max(int(round(x0 * src_x / tgt_iso)), 1)
    z_out = len(range(0, n_z, z_step))
    info = {
        "z_step": z_step,
        "tgt_iso_um": tgt_iso,
        "source_spacing_xyz_um": [src_x, src_y, src_z],
        "native_shape_zyx": [n_z, y0, x0],
        "midres_shape_zyx": [z_out, y_out, x_out],
        "spacing_zyx_um": [src_z * z_step, tgt_iso, tgt_iso],
    }
    return (z_out, y_out, x_out), info


def sample_for_reg_to_midres_lsfm(
    labels_atlas_zyx: np.ndarray,
    vmeta: dict,
) -> tuple[np.ndarray, dict]:
    """Undo atlas spacing zoom + permute → LSFM mid-res labels (nearest)."""
    from scipy import ndimage

    mid_shape, info = midres_shape_from_volume_meta(vmeta)
    perm = tuple(int(x) for x in vmeta.get("permute_zyx_to_atlas_zyx", [1, 0, 2]))
    # After permute, atlas axes were ~midres remapped then zoomed to sample_for_reg.
    # Inverse zoom on atlas ZYX to the pre-zoom atlas shape, then inverse permute.
    atlas_pre = [0, 0, 0]
    for out_ax, src_ax in enumerate(perm):
        atlas_pre[out_ax] = mid_shape[src_ax]
    atlas_pre_shape = tuple(int(v) for v in atlas_pre)
    zoom = tuple(t / s if s > 0 else 1.0 for t, s in zip(atlas_pre_shape, labels_atlas_zyx.shape))
    logger.info(
        "Upsample sample_for_reg %s → pre-zoom atlas %s (zoom=%s)",
        labels_atlas_zyx.shape,
        atlas_pre_shape,
        tuple(round(z, 4) for z in zoom),
    )
    if not np.allclose(zoom, 1.0, rtol=0.01, atol=0.01):
        up_atlas = ndimage.zoom(labels_atlas_zyx, zoom=zoom, order=0, mode="nearest").astype(np.uint16)
        # Fix off-by-one from rounding.
        if up_atlas.shape != atlas_pre_shape:
            fixed = np.zeros(atlas_pre_shape, dtype=np.uint16)
            slices = tuple(slice(0, min(a, b)) for a, b in zip(up_atlas.shape, atlas_pre_shape))
            fixed[slices] = up_atlas[slices]
            up_atlas = fixed
    else:
        up_atlas = labels_atlas_zyx.astype(np.uint16, copy=False)
    mid = inverse_permute_to_lsfm(up_atlas, perm).astype(np.uint16, copy=False)
    if mid.shape != mid_shape:
        fixed = np.zeros(mid_shape, dtype=np.uint16)
        slices = tuple(slice(0, min(a, b)) for a, b in zip(mid.shape, mid_shape))
        fixed[slices] = mid[slices]
        mid = fixed
    return mid, info


def write_label_zarr(
    path: Path,
    vol_zyx: np.ndarray,
    *,
    chunks: tuple[int, int, int] = (64, 256, 256),
    spacing_zyx_um: tuple[float, float, float] | None = None,
    extra_attrs: dict | None = None,
) -> Path:
    from pipeline_modules.segmentation.zarr_utils import create_output_zarr

    path = Path(path)
    if path.exists():
        import shutil

        shutil.rmtree(path)
    root, arr = create_output_zarr(path, vol_zyx.shape, chunks, np.uint16, dataset_name="0")
    # Write in Z slabs to bound memory.
    cz = int(chunks[0])
    for z0 in range(0, vol_zyx.shape[0], cz):
        z1 = min(z0 + cz, vol_zyx.shape[0])
        arr[z0:z1] = vol_zyx[z0:z1]
        if (z0 // cz) % 10 == 0:
            logger.info("  write zarr %s Z %d:%d / %d", path.name, z0, z1, vol_zyx.shape[0])
    attrs = {
        "dtype": "uint16",
        "shape_zyx": list(vol_zyx.shape),
        "spacing_zyx_um": list(spacing_zyx_um) if spacing_zyx_um else None,
    }
    if extra_attrs:
        attrs.update(extra_attrs)
    root.attrs.update({k: v for k, v in attrs.items() if v is not None})
    arr.attrs.update({k: v for k, v in attrs.items() if v is not None})
    return path


def upsample_midres_to_native_zarr(
    mid_zyx: np.ndarray,
    out_zarr: Path,
    *,
    native_shape_zyx: tuple[int, int, int],
    z_step: int,
    chunks: tuple[int, int, int] = (1, 2048, 2048),
    spacing_zyx_um: tuple[float, float, float] | None = None,
    extra_attrs: dict | None = None,
) -> Path:
    """Nearest upsample mid-res LSFM labels to native TIFF stack shape (chunked, sparse-friendly)."""
    from scipy import ndimage

    from pipeline_modules.segmentation.zarr_utils import create_output_zarr

    out_zarr = Path(out_zarr)
    if out_zarr.exists():
        import shutil

        shutil.rmtree(out_zarr)
    n_z, n_y, n_x = native_shape_zyx
    root, arr = create_output_zarr(out_zarr, (n_z, n_y, n_x), chunks, np.uint16, dataset_name="0")
    # Try to skip empty chunk materialization when supported.
    try:
        arr._write_empty_chunks = False  # noqa: SLF001 — zarr v2 best-effort
    except Exception:
        pass

    zy, zx = mid_zyx.shape[1], mid_zyx.shape[2]
    labeled_z = np.where(mid_zyx.max(axis=(1, 2)) > 0)[0]
    logger.info(
        "Native upsample → %s; mid labeled Z planes=%d/%d",
        native_shape_zyx,
        len(labeled_z),
        mid_zyx.shape[0],
    )
    for i, z_lr in enumerate(labeled_z):
        sl = mid_zyx[int(z_lr)]
        up = ndimage.zoom(sl, zoom=(n_y / max(zy, 1), n_x / max(zx, 1)), order=0, mode="nearest")
        if up.shape != (n_y, n_x):
            fixed = np.zeros((n_y, n_x), dtype=np.uint16)
            fixed[: min(n_y, up.shape[0]), : min(n_x, up.shape[1])] = up[
                : min(n_y, up.shape[0]), : min(n_x, up.shape[1])
            ]
            up = fixed
        z0 = int(z_lr) * int(z_step)
        z1 = min(z0 + int(z_step), n_z)
        for z in range(z0, z1):
            # Write in Y strips to limit peak RAM on wide slices.
            strip = int(chunks[1])
            for y0 in range(0, n_y, strip):
                y1 = min(y0 + strip, n_y)
                block = up[y0:y1]
                if np.any(block):
                    arr[z, y0:y1, :] = block
        if (i + 1) % 20 == 0 or i + 1 == len(labeled_z):
            logger.info("  native plane %d/%d (mid Z=%d → native Z[%d:%d))", i + 1, len(labeled_z), z_lr, z0, z1)

    attrs = {
        "dtype": "uint16",
        "shape_zyx": [n_z, n_y, n_x],
        "spacing_zyx_um": list(spacing_zyx_um) if spacing_zyx_um else None,
        "upsampled_from_midres": True,
        "z_step": int(z_step),
    }
    if extra_attrs:
        attrs.update(extra_attrs)
    root.attrs.update({k: v for k, v in attrs.items() if v is not None})
    arr.attrs.update({k: v for k, v in attrs.items() if v is not None})
    return out_zarr


def unstraighten_label_volume(
    label_flipped_zyx: np.ndarray,
    *,
    flip_meta: dict,
    centerline: np.ndarray,
    target_shape_zyx: tuple[int, int, int],
    spacing_zyx_um: tuple[float, float, float],
    straighten_meta: dict,
) -> np.ndarray:
    straight = undo_flips_zyx(label_flipped_zyx, flip_meta)
    return unstraighten_labels_zyx(
        straight,
        centerline,
        target_shape_zyx,
        spacing_zyx_um=spacing_zyx_um,
        out_radius_yx_vox=int(straighten_meta["out_radius_yx_vox"]),
        step_um=float(straighten_meta["step_um"]),
        smooth_centerline_um=float(straighten_meta.get("smooth_centerline_um", 40.0)),
        tangent_smooth_window=int(straighten_meta.get("tangent_smooth_window", 31)),
        frame_smooth_window=int(straighten_meta.get("frame_smooth_window", 31)),
    )


def run_wizard_backproject(
    *,
    wizard_dir: Path,
    sample_nii: Path,
    atlas_dir: Path,
    out_dir: Path,
    volume_meta_json: Path | None = None,
    sample_dir: Path | None = None,
    write_zarr: bool = False,
    zarr_mode: str = "segments",
    full_native: bool = False,
    zarr_chunks: tuple[int, int, int] = (64, 256, 256),
) -> dict:
    import nibabel as nib

    from pipeline_modules.registration.spinalj_straighten_register import load_centerline_csv

    wizard_dir = Path(wizard_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    reg_summary = json.loads((wizard_dir / "ants_out" / "registration_summary.json").read_text(encoding="utf-8"))
    flip_meta = json.loads((wizard_dir / "straightened" / "flip_meta.json").read_text(encoding="utf-8"))
    straighten_meta = json.loads((wizard_dir / "straightened" / "straighten_meta.json").read_text(encoding="utf-8"))
    centerline = load_centerline_csv(wizard_dir / "centerline_zyx.csv")

    segments_csv = Path(atlas_dir) / "Segments_Reference.csv"
    unique_names, _name_to_id, z_ids = load_segment_legend(segments_csv)
    id_to_name = {i + 1: n for i, n in enumerate(unique_names)}
    legend_path = out_dir / "segment_id_legend.csv"
    with legend_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["segment_id", "name"])
        for i, n in enumerate(unique_names):
            w.writerow([i + 1, n])

    template_work = wizard_dir / "reg_work" / "Template_work.nii.gz"
    logger.info("Building segment volume from %s ...", segments_csv)
    seg_xyz, seg_aff = build_segment_volume_xyz(template_work, z_ids)
    seg_moving_path = out_dir / "segments_atlas_work.nii.gz"
    nib.save(nib.Nifti1Image(seg_xyz, seg_aff), str(seg_moving_path))

    fixed_nii = Path(reg_summary["fixed"])
    fwd = list(reg_summary["fwdtransforms"])
    for t in fwd:
        if not Path(t).exists():
            raise FileNotFoundError(f"Missing transform: {t}")
    fixed_direction = np.asarray(reg_summary["fixed_direction"], dtype=np.float64)
    moving_direction = np.asarray(reg_summary["moving_direction"], dtype=np.float64)

    logger.info("Applying ANTs transforms → straightened/flipped space ...")
    warped_flipped_zyx = apply_ants_to_fixed(
        fixed_nii=fixed_nii,
        moving_xyz=seg_xyz,
        moving_affine=seg_aff,
        fwdtransforms=fwd,
        fixed_direction=fixed_direction,
        moving_direction=moving_direction,
    )
    save_nii_zyx(
        out_dir / "segments_in_straight_flipped.nii.gz",
        warped_flipped_zyx,
        tuple(straighten_meta["spacing_zyx_um_out"]),
    )

    logger.info("Undo flips %s ...", flip_meta)
    warped_straight_zyx = undo_flips_zyx(warped_flipped_zyx, flip_meta)
    save_nii_zyx(
        out_dir / "segments_in_straight.nii.gz",
        warped_straight_zyx,
        tuple(straighten_meta["spacing_zyx_um_out"]),
    )

    sample = nib.load(str(sample_nii))
    sample_xyz = np.asanyarray(sample.dataobj)
    sample_zyx = np.transpose(sample_xyz, (2, 1, 0))
    sp = tuple(float(abs(sample.affine[i, i])) for i in range(3))
    # NIfTI affine XYZ → spacing ZYX
    spacing_zyx_um = (sp[2], sp[1], sp[0])

    logger.info(
        "Unstraighten segments → sample_for_reg ZYX=%s spacing_zyx_um=%s ...",
        sample_zyx.shape,
        spacing_zyx_um,
    )
    seg_sample = unstraighten_label_volume(
        warped_flipped_zyx,
        flip_meta=flip_meta,
        centerline=centerline,
        target_shape_zyx=tuple(sample_zyx.shape),
        spacing_zyx_um=spacing_zyx_um,
        straighten_meta=straighten_meta,
    )
    sample_seg_path = out_dir / "segments_in_sample_for_reg.nii.gz"
    save_nii_zyx(sample_seg_path, seg_sample, spacing_zyx_um)

    ann_sample = None
    sample_ann_path = None
    if zarr_mode in ("annotation", "both"):
        warped_ann = wizard_dir / "ants_out" / "warped_annotation.nii.gz"
        if not warped_ann.exists():
            raise FileNotFoundError(f"Missing {warped_ann}")
        logger.info("Unstraighten annotation from %s ...", warped_ann)
        ann_flipped = load_nii_as_zyx_u16(warped_ann)
        ann_sample = unstraighten_label_volume(
            ann_flipped,
            flip_meta=flip_meta,
            centerline=centerline,
            target_shape_zyx=tuple(sample_zyx.shape),
            spacing_zyx_um=spacing_zyx_um,
            straighten_meta=straighten_meta,
        )
        sample_ann_path = out_dir / "annotation_in_sample_for_reg.nii.gz"
        save_nii_zyx(sample_ann_path, ann_sample, spacing_zyx_um)

    vmeta = None
    if volume_meta_json and Path(volume_meta_json).exists():
        vmeta = json.loads(Path(volume_meta_json).read_text(encoding="utf-8"))

    lsfm_path = None
    if vmeta is not None:
        # Keep a quick NIfTI at inverse-permute only (reg-res), plus mid-res for zarr.
        perm = tuple(vmeta.get("permute_zyx_to_atlas_zyx", [1, 0, 2]))
        seg_lsfm = inverse_permute_to_lsfm(seg_sample, perm)
        iso = float(min(vmeta.get("target_spacing_xyz_um", [10.0, 10.0, 20.0])))
        lsfm_path = out_dir / "segments_in_lsfm_stack_axes.nii.gz"
        save_nii_zyx(lsfm_path, seg_lsfm, (iso, iso, iso))
        logger.info("Wrote LSFM-axis labels %s shape=%s", lsfm_path, seg_lsfm.shape)

    write_qc_overlays(out_dir / "qc", sample_zyx.astype(np.float32), seg_sample, id_to_name)

    zarr_paths: dict[str, str] = {}
    if write_zarr:
        if sample_dir is None:
            sample_dir = wizard_dir.parent
        sample_dir = Path(sample_dir)
        sample_dir.mkdir(parents=True, exist_ok=True)
        if vmeta is None:
            raise RuntimeError("--write_zarr requires volume_meta.json to recover LSFM geometry")

        def _emit(name: str, labels_atlas: np.ndarray, extra: dict) -> Path:
            mid, info = sample_for_reg_to_midres_lsfm(labels_atlas, vmeta)
            mid_path = sample_dir / f"{name}.zarr"
            logger.info("Writing mid-res zarr %s shape=%s ...", mid_path, mid.shape)
            write_label_zarr(
                mid_path,
                mid,
                chunks=zarr_chunks,
                spacing_zyx_um=tuple(info["spacing_zyx_um"]),
                extra_attrs={
                    **extra,
                    **{k: info[k] for k in ("z_step", "tgt_iso_um", "native_shape_zyx", "midres_shape_zyx")},
                    "volume_meta": str(volume_meta_json),
                    "axes": "ZYX_LSFM_stack",
                },
            )
            zarr_paths[name] = str(mid_path)
            if full_native:
                nat_path = sample_dir / f"{name}_native.zarr"
                src = [float(v) for v in vmeta["source_spacing_xyz_um"]]
                logger.info("Writing native-resolution zarr %s (sparse chunked) ...", nat_path)
                upsample_midres_to_native_zarr(
                    mid,
                    nat_path,
                    native_shape_zyx=tuple(info["native_shape_zyx"]),
                    z_step=int(info["z_step"]),
                    chunks=(1, min(2048, zarr_chunks[1] * 4), min(2048, zarr_chunks[2] * 4)),
                    spacing_zyx_um=(src[2], src[1], src[0]),
                    extra_attrs={**extra, "axes": "ZYX_native_TIFF_stack"},
                )
                zarr_paths[f"{name}_native"] = str(nat_path)
            return mid_path

        if zarr_mode in ("segments", "both"):
            _emit(
                "spinalj_segments",
                seg_sample,
                {"label_kind": "vertebral_segments", "legend_csv": str(legend_path)},
            )
        if zarr_mode in ("annotation", "both"):
            if ann_sample is None:
                raise RuntimeError("annotation labels missing")
            _emit(
                "spinalj_annotation",
                ann_sample,
                {"label_kind": "spinalj_annotation_regions"},
            )

    fg = int(np.count_nonzero(seg_sample))
    present = sorted({int(v) for v in np.unique(seg_sample) if int(v) > 0})
    summary = {
        "sample_nii": str(sample_nii),
        "wizard_dir": str(wizard_dir),
        "segments_in_sample_for_reg": str(sample_seg_path),
        "annotation_in_sample_for_reg": str(sample_ann_path) if sample_ann_path else None,
        "segments_in_lsfm_stack_axes": str(lsfm_path) if lsfm_path else None,
        "legend": str(legend_path),
        "n_labeled_voxels": fg,
        "segment_ids_present": present,
        "segment_names_present": [id_to_name[i] for i in present if i in id_to_name],
        "zarr": zarr_paths,
        "note": (
            "Labels back-projected with centerline frames. Mid-res zarr matches original "
            "TIFF stack axes at ~target iso µm. Optional *_native.zarr matches full TIFF shape "
            "(very large; sparse write)."
        ),
    }
    (out_dir / "backproject_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("Done. Labeled voxels=%d segments=%s zarr=%s", fg, summary["segment_names_present"], zarr_paths)
    return summary


def main() -> None:
    _configure_logging()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--wizard_dir", required=True, help=".../SOD1/_spinalj_wizard")
    p.add_argument(
        "--sample_nii",
        default="",
        help="sample_for_reg/volume.nii.gz (default: sibling _spinalj_reg/...)",
    )
    p.add_argument(
        "--atlas_dir",
        default=r"S:\Yifu_data\reference\SC_P56_Atlas_10x10x20_v5_2020",
        help="SpinalJ atlas folder containing Segments_Reference.csv",
    )
    p.add_argument("--out_dir", default="", help="Default: <wizard_dir>/labels_native")
    p.add_argument(
        "--volume_meta_json",
        default="",
        help="volume_meta.json from downsample (for LSFM-axis reorient)",
    )
    p.add_argument(
        "--sample_dir",
        default="",
        help="Original sample folder to write zarr into (default: parent of wizard_dir)",
    )
    p.add_argument(
        "--write_zarr",
        action="store_true",
        help="Write upsampled label zarr under --sample_dir",
    )
    p.add_argument(
        "--zarr_mode",
        choices=("segments", "annotation", "both"),
        default="both",
        help="Which labels to write as zarr (default both)",
    )
    p.add_argument(
        "--full_native",
        action="store_true",
        help="Also write native TIFF-shape zarr (huge; chunked). Default is mid-res ~10µm only.",
    )
    p.add_argument("--zarr_chunks", default="64,256,256", help="Zarr chunks Z,Y,X")
    p.add_argument(
        "--reuse_sample_labels",
        action="store_true",
        help="Reuse existing segments/annotation NIfTI in out_dir (skip ANTs/unstraighten)",
    )
    args = p.parse_args()

    wizard_dir = Path(args.wizard_dir)
    sample_nii = Path(args.sample_nii) if args.sample_nii else (
        wizard_dir.parent / "_spinalj_reg" / "sample_for_reg" / "volume.nii.gz"
    )
    out_dir = Path(args.out_dir) if args.out_dir else (wizard_dir / "labels_native")
    volume_meta = Path(args.volume_meta_json) if args.volume_meta_json else (
        sample_nii.parent / "volume_meta.json"
    )
    chunks = tuple(int(x) for x in args.zarr_chunks.split(","))
    if len(chunks) != 3:
        raise SystemExit("--zarr_chunks must be Z,Y,X")

    if args.reuse_sample_labels:
        _run_zarr_only_from_existing(
            out_dir=out_dir,
            volume_meta_json=volume_meta,
            sample_dir=Path(args.sample_dir) if args.sample_dir else wizard_dir.parent,
            zarr_mode=args.zarr_mode,
            full_native=bool(args.full_native),
            zarr_chunks=chunks,
        )
        return

    run_wizard_backproject(
        wizard_dir=wizard_dir,
        sample_nii=sample_nii,
        atlas_dir=Path(args.atlas_dir),
        out_dir=out_dir,
        volume_meta_json=volume_meta if volume_meta.exists() else None,
        sample_dir=Path(args.sample_dir) if args.sample_dir else wizard_dir.parent,
        write_zarr=bool(args.write_zarr),
        zarr_mode=args.zarr_mode,
        full_native=bool(args.full_native),
        zarr_chunks=chunks,
    )


def _run_zarr_only_from_existing(
    *,
    out_dir: Path,
    volume_meta_json: Path,
    sample_dir: Path,
    zarr_mode: str,
    full_native: bool,
    zarr_chunks: tuple[int, int, int],
) -> None:
    """Fast path: upsample already-backprojected NIfTIs to sample-dir zarr."""
    if not volume_meta_json.exists():
        raise SystemExit(f"Missing volume meta: {volume_meta_json}")
    vmeta = json.loads(volume_meta_json.read_text(encoding="utf-8"))
    sample_dir = Path(sample_dir)
    sample_dir.mkdir(parents=True, exist_ok=True)
    legend_path = out_dir / "segment_id_legend.csv"
    zarr_paths: dict[str, str] = {}

    def _one(nifti_name: str, zarr_name: str, extra: dict) -> None:
        path = out_dir / nifti_name
        if not path.exists():
            raise SystemExit(f"Missing {path}; run without --reuse_sample_labels first")
        labels = load_nii_as_zyx_u16(path)
        mid, info = sample_for_reg_to_midres_lsfm(labels, vmeta)
        mid_path = sample_dir / f"{zarr_name}.zarr"
        logger.info("Writing %s shape=%s", mid_path, mid.shape)
        write_label_zarr(
            mid_path,
            mid,
            chunks=zarr_chunks,
            spacing_zyx_um=tuple(info["spacing_zyx_um"]),
            extra_attrs={
                **extra,
                **{k: info[k] for k in ("z_step", "tgt_iso_um", "native_shape_zyx", "midres_shape_zyx")},
                "volume_meta": str(volume_meta_json),
                "axes": "ZYX_LSFM_stack",
            },
        )
        zarr_paths[zarr_name] = str(mid_path)
        if full_native:
            nat_path = sample_dir / f"{zarr_name}_native.zarr"
            src = [float(v) for v in vmeta["source_spacing_xyz_um"]]
            upsample_midres_to_native_zarr(
                mid,
                nat_path,
                native_shape_zyx=tuple(info["native_shape_zyx"]),
                z_step=int(info["z_step"]),
                chunks=(1, 2048, 2048),
                spacing_zyx_um=(src[2], src[1], src[0]),
                extra_attrs={**extra, "axes": "ZYX_native_TIFF_stack"},
            )
            zarr_paths[f"{zarr_name}_native"] = str(nat_path)

    if zarr_mode in ("segments", "both"):
        _one(
            "segments_in_sample_for_reg.nii.gz",
            "spinalj_segments",
            {"label_kind": "vertebral_segments", "legend_csv": str(legend_path)},
        )
    if zarr_mode in ("annotation", "both"):
        _one(
            "annotation_in_sample_for_reg.nii.gz",
            "spinalj_annotation",
            {"label_kind": "spinalj_annotation_regions"},
        )
    summary_path = sample_dir / "spinalj_labels_zarr_summary.json"
    summary_path.write_text(json.dumps({"zarr": zarr_paths}, indent=2), encoding="utf-8")
    logger.info("Zarr-only done: %s", zarr_paths)


if __name__ == "__main__":
    main()
