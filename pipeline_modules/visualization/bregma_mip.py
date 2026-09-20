"""Render coronal slab-MIP screenshots from a brain Zarr at bregma AP coordinates.

The tool intentionally avoids re-running registration. Two anchor modes:

* **manual (default line)**: state where bregma sits on the sample AP axis
  (``--bregma-index`` or ``--bregma-offset-mm`` + ``--anterior``) and correct
  the cutting angle with ``--tilt-deg``. Works on any Zarr, needs only voxel
  size, and renders a sheared slab maximum-intensity projection.
* **transforms (best-effort enhancement)**: if the sample was already
  registered by the pipeline, the stored ANTs ``transforms/`` files are tried
  as a bregma anchor, cross-validated against ``upsampled_atlas_label.zarr``
  on an interior atlas landmark. Fresh registration is never run (it takes
  ~45 min per sample); when the check cannot pass -- no matching label zarr,
  or the physical-space bookkeeping does not verify -- the tool demands a
  manual anchor instead of guessing.

Sample Zarr convention (same as the rest of the repo): arrays are ``(z, y, x)``
with z = DV (the TIFF stack axis), y = AP, x = ML. A coronal MIP therefore has
rows = DV and cols = ML, matching atlas coronal slices in ``atlas_slice.py``.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from matplotlib import pyplot as plt  # noqa: E402

from pipeline_modules.segmentation.zarr_utils import open_zarr_dataset  # noqa: E402

logger = logging.getLogger(__name__)

DEFAULT_VOXEL_XYZ_UM = (1.8, 1.8, 2.0)
# Same approximate Allen CCF 25 um bregma voxel as atlas_slice.py.
DEFAULT_BREGMA_DV_AP_ML = (18, 216, 228)


class AnchorError(RuntimeError):
    """Raised when no usable bregma anchor could be resolved."""


@dataclass
class ApAnchor:
    """Resolved position of bregma on the sample AP (zarr axis 1) grid.

    ``index`` is always expressed on the level-0 (full resolution) grid;
    ``index_for`` converts to the grid actually being rendered.
    """

    index: float  # continuous AP index of bregma at level 0
    anterior: str  # "low" | "high": which end of axis 1 is anterior
    source: str  # "bregma-index" | "bregma-offset" | "transforms"
    bregma_dv: float | None = None  # optional bregma DV row (level 0)
    bregma_ml: float | None = None  # optional bregma ML col (level 0)
    label_check: str = "skipped"

    def index_for(self, bregma_mm: float, ap_vox_um: float, *, level: int = 0) -> float:
        """AP index (on the rendered level grid) of a mm coordinate; anterior positive."""
        scale = 2**level
        sign = 1.0 if self.anterior == "high" else -1.0
        return self.index / scale + sign * bregma_mm * 1000.0 / (ap_vox_um * scale)


# ---------------------------------------------------------------------------
# Voxel-size resolution
# ---------------------------------------------------------------------------


def _multiscale_scale_zyx(group) -> tuple[float, float, float] | None:
    """Best-effort (z, y, x) micron scale from NGFF multiscales attrs."""
    try:
        multi = group.attrs.get("multiscales")
        if not multi:
            return None
        entry = multi[0]
        scale = entry.get("base_scale_zyx") or entry.get("scale")
        if scale:
            values = [float(v) for v in scale]
            if len(values) == 3:
                return tuple(values)  # type: ignore[return-value]
        for ds in entry.get("datasets") or []:
            ds_scale = ds.get("scale") or ds.get("coordinateTransformations")
            if isinstance(ds_scale, list):  # coordinateTransformations form
                for tf in ds_scale:
                    if isinstance(tf, dict) and tf.get("type") == "scale":
                        ds_scale = tf.get("scale")
                        break
            if ds_scale and len(ds_scale) == 3:
                return tuple(float(v) for v in ds_scale)  # type: ignore[return-value]
    except Exception:
        return None
    return None


def resolve_voxel_xyz_um(
    zarr_path: str | Path,
    *,
    explicit: str = "",
    config_path: str | Path | None = None,
    sample_dir: str | Path | None = None,
) -> tuple[tuple[float, float, float], str]:
    """Resolve native (x=ML, y=AP, z=DV) voxel size in microns.

    Priority: explicit flag > NGFF multiscales attrs > config.json
    ``input.resolution_xyz`` > repo default (with a warning).
    """
    if explicit.strip():
        parts = [float(p) for p in explicit.split(",")]
        if len(parts) != 3 or any(p <= 0 for p in parts):
            raise ValueError(f"--voxel-xyz-um must be three positive numbers, got: {explicit}")
        return (parts[0], parts[1], parts[2]), "flag"

    import zarr

    try:
        group = zarr.open_group(str(zarr_path), mode="r")
        scale_zyx = _multiscale_scale_zyx(group)
    except Exception:
        scale_zyx = None
    if scale_zyx:
        z_um, y_um, x_um = scale_zyx
        return (x_um, y_um, z_um), "zarr-multiscales"

    if config_path is None and sample_dir is not None:
        candidate = Path(sample_dir).parent / "config.json"
        if candidate.exists():
            config_path = candidate
    if config_path is not None and Path(config_path).exists():
        with open(config_path, encoding="utf-8-sig") as handle:
            cfg = json.load(handle)
        values = cfg.get("input", {}).get("resolution_xyz")
        if values and len(values) == 3:
            return (float(values[0]), float(values[1]), float(values[2])), f"config:{config_path}"

    logger.warning(
        "Voxel size not found in zarr attrs or config; using repo default %s um (x,y,z). "
        "Pass --voxel-xyz-um to override.",
        DEFAULT_VOXEL_XYZ_UM,
    )
    return DEFAULT_VOXEL_XYZ_UM, "default"


# ---------------------------------------------------------------------------
# Anchors
# ---------------------------------------------------------------------------


def resolve_manual_anchor(
    *,
    n_ap_level: int,
    n_ap_level0: int,
    level: int,
    ap_vox_um: float,
    bregma_index: float | None,
    bregma_offset_mm: float | None,
    anterior: str,
) -> ApAnchor:
    if anterior not in ("low", "high"):
        raise ValueError("--anterior must be 'low' or 'high'")
    scale = 2**level
    if bregma_index is not None:
        idx_level = float(bregma_index)
        if not 0 <= idx_level < n_ap_level:
            raise AnchorError(
                f"--bregma-index {bregma_index} outside AP axis 0..{n_ap_level - 1} at level {level}"
            )
        return ApAnchor(index=idx_level * scale, anterior=anterior, source="bregma-index")
    if bregma_offset_mm is not None:
        offset_vox = abs(float(bregma_offset_mm)) * 1000.0 / ap_vox_um
        edge = 0.0 if anterior == "low" else float(n_ap_level0) - 1.0
        index0 = edge + offset_vox if anterior == "low" else edge - offset_vox
        if not 0 <= index0 < n_ap_level0:
            raise AnchorError(
                f"bregma offset {bregma_offset_mm} mm lands at AP index {index0:.1f}, "
                f"outside AP axis 0..{n_ap_level0 - 1}"
            )
        return ApAnchor(index=index0, anterior=anterior, source="bregma-offset")
    raise AnchorError(
        "No bregma anchor: pass --bregma-index, or --bregma-offset-mm (with --anterior), "
        "or point --transforms-dir at stored pipeline transforms."
    )


def map_bregma_through_transforms(
    *,
    transforms_dir: str | Path,
    atlas_image_path: str | Path,
    voxel_xyz_um: tuple[float, float, float],
    native_shape_zyx: tuple[int, int, int],
    bregma_dv_ap_ml: tuple[int, int, int] = DEFAULT_BREGMA_DV_AP_ML,
    label_zarr_path: str | Path | None = None,
    atlas_label_path: str | Path | None = None,
) -> ApAnchor:
    """Map the atlas bregma voxel into native sample space with stored fwd transforms.

    The pipeline registers with fixed=downsampled sample, moving=atlas
    (``atlas2image``), so the ``fwd_*`` files map atlas physical points to
    sample physical points. The downsample NIfTI physical space is microns with
    origin 0, hence native index = physical microns / native voxel microns.
    """
    import ants
    import pandas as pd

    transforms_dir = Path(transforms_dir)
    fwd = sorted(transforms_dir.glob("fwd_*"), key=lambda p: p.name)
    inv = sorted(transforms_dir.glob("inv_*"), key=lambda p: p.name)
    if not fwd:
        raise AnchorError(f"No fwd_* transforms in {transforms_dir}")

    label_arr = None
    if label_zarr_path and Path(label_zarr_path).exists():
        candidate = open_zarr_dataset(label_zarr_path)
        if tuple(int(v) for v in candidate.shape) == tuple(native_shape_zyx):
            label_arr = candidate
    if label_arr is None:
        raise AnchorError(
            "Transforms anchor requires a shape-matching upsampled_atlas_label.zarr "
            "for validation; pass a manual anchor (--bregma-index / --bregma-offset-mm)."
        )

    # Bregma sits at the cortical surface where the atlas label is empty, so
    # validate on a deep interior landmark instead.
    landmark = _atlas_interior_landmark(atlas_label_path) if atlas_label_path else None
    if landmark is None:
        raise AnchorError(
            "Could not pick an interior atlas landmark for validation; pass a manual anchor."
        )
    (l_dv, l_ap, l_ml), expected = landmark

    atlas = ants.image_read(str(atlas_image_path))
    atlas.set_direction(np.eye(3))
    vox_ml, vox_ap, vox_dv = voxel_xyz_um

    def atlas_point(dv_ap_ml_point: tuple[float, float, float]) -> pd.DataFrame:
        p_dv, p_ap, p_ml = dv_ap_ml_point
        # ANTs numpy order for both spaces is (x=ML, y=AP, z=DV).
        numpy_idx = np.array([p_ml, p_ap, p_dv], dtype=np.float64)
        phys = (
            np.asarray(atlas.origin, dtype=np.float64)
            + np.asarray(atlas.spacing, dtype=np.float64) * numpy_idx
        )
        return pd.DataFrame({"x": [phys[0]], "y": [phys[1]], "z": [phys[2]]})

    def map_um(dv_ap_ml_point, transformlist) -> tuple[float, float, float]:
        out = ants.apply_transforms_to_points(
            3, atlas_point(dv_ap_ml_point), [str(p) for p in transformlist]
        )
        # ANTs physical components are (x=ML, y=AP, z=DV) microns.
        return float(out["z"][0]), float(out["y"][0]), float(out["x"][0])  # -> (dv, ap, ml) um

    def landmark_agrees(transformlist) -> bool:
        m_dv_um, m_ap_um, m_ml_um = map_um((l_dv, l_ap, l_ml), transformlist)
        z = int(round(m_dv_um / vox_dv))
        y = int(round(m_ap_um / vox_ap))
        x = int(round(m_ml_um / vox_ml))
        n_dv, n_ap, n_ml = native_shape_zyx
        if not (0 <= z < n_dv and 0 <= y < n_ap and 0 <= x < n_ml):
            return False
        # Tolerance of roughly one 25 um downsample voxel in each axis.
        r, c, k = (max(1, int(round(25.0 / v))) for v in (vox_dv, vox_ap, vox_ml))
        box = np.asarray(
            label_arr[max(0, z - r) : z + r + 1, max(0, y - c) : y + c + 1, max(0, x - k) : x + k + 1]
        )
        return bool(np.any(box == expected))

    # apply_transforms_to_points semantics differ across antspyx versions for
    # deformation fields, so adopt the list convention that actually maps the
    # interior atlas landmark onto its own warped label.
    candidates: list[tuple[str, list[Path]]] = [("fwd", fwd)]
    if inv:
        candidates.append(("inv", inv))
    candidates.extend([("fwd-reversed", fwd[::-1]), ("inv-reversed", inv[::-1])])
    chosen = next(((name, tl) for name, tl in candidates if tl and landmark_agrees(tl)), None)
    if chosen is None:
        raise AnchorError(
            "Stored transforms failed the interior-landmark label check under every "
            "transform-list convention; pass --bregma-index or --bregma-offset-mm instead."
        )
    convention, transformlist = chosen

    dv_i, ap_i, ml_i = (float(v) for v in bregma_dv_ap_ml)
    b_dv_um, b_ap_um, b_ml_um = map_um((dv_i, ap_i, ml_i), transformlist)
    anchor = ApAnchor(
        index=b_ap_um / vox_ap,
        anterior="low",
        source="transforms",
        bregma_dv=b_dv_um / vox_dv,
        bregma_ml=b_ml_um / vox_ml,
    )
    n_dv, n_ap, n_ml = native_shape_zyx
    if not (
        0 <= anchor.index < n_ap and 0 <= anchor.bregma_dv < n_dv and 0 <= anchor.bregma_ml < n_ml
    ):
        raise AnchorError(
            "Transform-mapped bregma point lands outside the zarr grid "
            f"(dv {anchor.bregma_dv:.0f}, ap {anchor.index:.0f}, ml {anchor.bregma_ml:.0f} "
            f"vs shape {native_shape_zyx}); transforms may not match this zarr."
        )

    # Resolve AP orientation: map a point 1 mm anterior of bregma (atlas AP:
    # anterior = lower index; atlas voxels are 25 um) and see which way the
    # sample AP index moves.
    _pd, p_ap_um, _pm = map_um((dv_i, ap_i - 1000.0 / 25.0, ml_i), transformlist)
    anchor.anterior = "high" if p_ap_um / vox_ap > anchor.index else "low"
    anchor.label_check = f"ok via interior landmark (region {expected}, convention {convention})"
    return anchor


def _atlas_interior_landmark(atlas_label_path: str | Path) -> tuple[tuple[int, int, int], int] | None:
    """A labeled atlas voxel near the volume middle, for transform cross-checks."""
    try:
        import tifffile

        with tifffile.TiffFile(str(atlas_label_path)) as tif:
            n_pages = len(tif.pages)
            page_shape = tuple(int(v) for v in tif.pages[0].shape)
        if n_pages < 8 or len(page_shape) != 2:
            return None
        cy, cx = page_shape[0] / 2.0, page_shape[1] / 2.0
        for dv in range(int(n_pages * 0.35), int(n_pages * 0.65), max(1, n_pages // 24)):
            page = np.asarray(tifffile.imread(str(atlas_label_path), key=dv))
            ys, xs = np.nonzero(page)
            if ys.size == 0:
                continue
            best = int(np.argmin((ys - cy) ** 2 + (xs - cx) ** 2))
            return (int(dv), int(ys[best]), int(xs[best])), int(page[ys[best], xs[best]])
    except Exception as exc:  # noqa: BLE001 - diagnostic only
        logger.warning("Interior landmark lookup failed: %s", exc)
    return None


# ---------------------------------------------------------------------------
# Slab MIP
# ---------------------------------------------------------------------------


def coronal_row_ap_centers(
    *,
    n_dv: int,
    base_ap_index: float,
    anterior: str,
    tilt_deg: float,
    dv_vox_um: float,
    ap_vox_um: float,
    tilt_ref_row: float | None = None,
) -> np.ndarray:
    """Per-DV-row AP slab center; positive tilt samples more anterior tissue ventrally."""
    rows = np.arange(n_dv, dtype=np.float64)
    if tilt_deg == 0.0:
        return np.full(n_dv, float(base_ap_index))
    if tilt_ref_row is None:
        tilt_ref_row = (n_dv - 1) / 2.0
    dv_delta_mm = (rows - float(tilt_ref_row)) * dv_vox_um / 1000.0
    ap_delta_mm = math.tan(math.radians(float(tilt_deg))) * dv_delta_mm
    sign = 1.0 if anterior == "high" else -1.0
    return base_ap_index + sign * ap_delta_mm * 1000.0 / ap_vox_um


def compute_coronal_slab_mip(
    arr,
    *,
    row_ap_centers: np.ndarray,
    half_window_vox: float,
    dv_block: int = 128,
) -> np.ndarray:
    """Slab MIP with rows = DV, cols = ML; each row maxes its own (tilted) AP window."""
    n_dv, n_ap, n_ml = arr.shape
    if half_window_vox <= 0:
        centers = np.rint(row_ap_centers).astype(np.int64)
        lo = np.clip(centers, 0, n_ap - 1)
        hi = lo + 1
    else:
        lo = np.clip(np.floor(row_ap_centers - half_window_vox).astype(np.int64), 0, n_ap - 1)
        hi = np.clip(np.ceil(row_ap_centers + half_window_vox).astype(np.int64) + 1, 1, n_ap)  # exclusive
    out_dtype = np.result_type(arr.dtype, np.uint16)
    out = np.zeros((n_dv, n_ml), dtype=out_dtype)
    for r0 in range(0, n_dv, int(dv_block)):
        r1 = min(r0 + int(dv_block), n_dv)
        c_lo = int(lo[r0:r1].min())
        c_hi = int(hi[r0:r1].max())
        block = np.asarray(arr[r0:r1, c_lo:c_hi, :])
        for i, r in enumerate(range(r0, r1)):
            out[r] = block[i, lo[r] - c_lo : hi[r] - c_lo].max(axis=0)
    return out


def _max_pool(image: np.ndarray, factor: int) -> np.ndarray:
    if factor <= 1:
        return image
    rows, cols = image.shape
    rows, cols = rows - rows % factor, cols - cols % factor
    trimmed = image[:rows, :cols]
    return trimmed.reshape(rows // factor, factor, cols // factor, factor).max(axis=(1, 3))


def label_boundary_mask(label_slice: np.ndarray) -> np.ndarray:
    """White-pixel boundary mask between differing atlas regions on a coronal slice."""
    labels = np.asarray(label_slice)
    mask = np.zeros(labels.shape, dtype=bool)
    diff_rows = labels[:-1, :] != labels[1:, :]
    mask[:-1, :] |= diff_rows
    mask[1:, :] |= diff_rows
    diff_cols = labels[:, :-1] != labels[:, 1:]
    mask[:, :-1] |= diff_cols
    mask[:, 1:] |= diff_cols
    return mask


def resolve_display_vmax(mip: np.ndarray, *, percentile: float, explicit: float | None) -> float:
    if explicit is not None:
        return float(explicit)
    values = mip[mip > 0]
    if values.size == 0:
        return 1.0
    vmax = float(np.percentile(values, percentile))
    return vmax if vmax > 0 else 1.0


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


@dataclass
class RenderMeta:
    bregma_mm: float
    thickness_mm: float
    tilt_deg: float
    anchor: ApAnchor
    voxel_xyz_um: tuple[float, float, float]
    voxel_source: str
    ap_lo: int
    ap_hi: int
    vmax: float
    zarr_path: Path
    level: int
    atlas_panel: str = ""


def render_bregma_png(
    mip: np.ndarray,
    *,
    output_path: Path,
    meta: RenderMeta,
    flip_dv: bool = False,
    flip_ml: bool = False,
    boundary_mask: np.ndarray | None = None,
    atlas_slice_image: np.ndarray | None = None,
    pool_factor: int = 1,
    dpi: int = 200,
) -> Path:
    if pool_factor > 1:
        mip = _max_pool(mip, pool_factor)
        if boundary_mask is not None:
            boundary_mask = _max_pool(boundary_mask.astype(np.uint8), pool_factor).astype(bool)
    image = np.flipud(mip) if flip_dv else mip
    image = np.fliplr(image) if flip_ml else image
    boundary = None
    if boundary_mask is not None:
        boundary = np.flipud(boundary_mask) if flip_dv else boundary_mask
        boundary = np.fliplr(boundary) if flip_ml else boundary

    scale = 2**meta.level
    extent_ml_mm = image.shape[1] * meta.voxel_xyz_um[0] * scale / 1000.0
    extent_dv_mm = image.shape[0] * meta.voxel_xyz_um[2] * scale / 1000.0

    n_panels = 2 if atlas_slice_image is not None else 1
    aspect = extent_ml_mm / max(extent_dv_mm, 1e-6)
    panel_h = 8.0
    fig_w = panel_h * aspect * n_panels + max(panel_h * aspect * 0.35, 2.5)
    fig, axes = plt.subplots(1, n_panels, figsize=(fig_w, panel_h), dpi=dpi)
    axes = np.atleast_1d(axes)
    for ax in axes:
        ax.set_facecolor("black")

    ax = axes[0]
    ax.imshow(
        image, cmap="gray", vmin=0.0, vmax=meta.vmax,
        extent=(0, extent_ml_mm, extent_dv_mm, 0), interpolation="nearest",
    )
    if boundary is not None and boundary.any():
        overlay = np.zeros((*boundary.shape, 4), dtype=np.float32)
        overlay[boundary] = (1.0, 1.0, 1.0, 0.85)
        ax.imshow(overlay, extent=(0, extent_ml_mm, extent_dv_mm, 0), interpolation="nearest")
    if meta.anchor.bregma_dv is not None and meta.anchor.bregma_ml is not None:
        dv_mm = meta.anchor.bregma_dv * meta.voxel_xyz_um[2] / 1000.0
        ml_mm = meta.anchor.bregma_ml * meta.voxel_xyz_um[0] / 1000.0
        if flip_dv:
            dv_mm = extent_dv_mm - dv_mm
        if flip_ml:
            ml_mm = extent_ml_mm - ml_mm
        ax.plot([ml_mm - 0.2, ml_mm + 0.2], [dv_mm, dv_mm], color="cyan", linewidth=1.2)
        ax.plot([ml_mm, ml_mm], [dv_mm - 0.2, dv_mm + 0.2], color="cyan", linewidth=1.2)
    _draw_scale_bar(ax, extent_ml_mm, extent_dv_mm)

    ax.set_title(
        f"Bregma AP {meta.bregma_mm:+.2f} mm  |  slab {meta.thickness_mm:.2f} mm",
        color="white", fontsize=11,
    )
    ax.text(
        0.5, -0.04,
        (
            f"anchor: {meta.anchor.source} (AP idx {meta.anchor.index / scale:.0f}, "
            f"anterior={meta.anchor.anterior})  tilt {meta.tilt_deg:+.1f} deg  "
            f"window AP [{meta.ap_lo}, {meta.ap_hi}]\n"
            f"voxel {meta.voxel_xyz_um} um ({meta.voxel_source})  level {meta.level}  "
            f"vmax={meta.vmax:.0f}  {meta.zarr_path.name}"
        ),
        transform=ax.transAxes, ha="center", va="top", color="0.75", fontsize=7,
    )

    if atlas_slice_image is not None:
        from pipeline_modules.visualization.atlas_slice import _add_lines, _label_contour_lines

        atlas_ax = axes[1]
        atlas_ax.imshow(
            np.zeros(atlas_slice_image.shape, dtype=np.uint8), cmap="gray", vmin=0, vmax=1,
            interpolation="nearest",
        )
        lines = _label_contour_lines(atlas_slice_image, smoothing=1.2)
        _add_lines(atlas_ax, lines, linewidth=0.25)
        atlas_ax.set_xlim(-0.5, atlas_slice_image.shape[1] - 0.5)
        atlas_ax.set_ylim(atlas_slice_image.shape[0] - 0.5, -0.5)
        atlas_ax.set_title(f"Allen atlas {meta.atlas_panel}", color="white", fontsize=10)

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color("0.4")
    fig.patch.set_facecolor("black")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, facecolor="black", bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    return output_path


def _draw_scale_bar(ax, extent_ml_mm: float, extent_dv_mm: float) -> None:
    bar_mm = 1.0 if extent_ml_mm >= 2.0 else 0.5
    x1 = extent_ml_mm * 0.96
    x0 = x1 - bar_mm
    y = extent_dv_mm * 0.95
    ax.plot([x0, x1], [y, y], color="white", linewidth=2.5)
    ax.text((x0 + x1) / 2.0, y - extent_dv_mm * 0.02, f"{bar_mm:g} mm", color="white",
            ha="center", va="bottom", fontsize=8)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def render_bregma_mips(
    *,
    zarr_path: str | Path,
    bregma_mm: list[float],
    thickness_mm: float,
    bregma_index: float | None = None,
    bregma_offset_mm: float | None = None,
    anterior: str = "low",
    tilt_deg: float = 0.0,
    tilt_ref_row: float | None = None,
    voxel_xyz_um: str = "",
    config_path: str | Path | None = None,
    level: int = 0,
    dataset: str = "0",
    transforms_dir: str | Path | None = None,
    atlas_image_path: str | Path | None = None,
    atlas_label_path: str | Path | None = None,
    label_zarr_path: str | Path | None = None,
    use_transforms: bool = True,
    overlay_labels: bool = True,
    atlas_panel: bool = False,
    percentile: float = 99.5,
    vmax: float | None = None,
    flip_dv: bool = False,
    flip_ml: bool = False,
    output_dir: str | Path | None = None,
    max_pixels: int = 3000,
    dv_block: int = 128,
) -> dict[str, object]:
    zarr_path = Path(zarr_path)
    sample_dir = zarr_path.parent
    voxel, voxel_source = resolve_voxel_xyz_um(
        zarr_path, explicit=voxel_xyz_um, config_path=config_path, sample_dir=sample_dir
    )
    vox_ml, vox_ap, vox_dv = voxel

    base_arr = open_zarr_dataset(zarr_path, dataset_name=dataset)
    if base_arr.ndim != 3:
        raise ValueError(f"Expected a 3D (z, y, x) zarr, got shape {base_arr.shape}")
    native_shape = tuple(int(v) for v in base_arr.shape)
    if level:
        try:
            arr = open_zarr_dataset(zarr_path, dataset_name=str(level))
        except Exception as exc:
            raise ValueError(f"Pyramid level {level} not found in {zarr_path}: {exc}") from exc
        scale = 2**level
    else:
        arr = base_arr
        scale = 1
    n_dv, n_ap, n_ml = (int(v) for v in arr.shape)

    transforms_dir = Path(transforms_dir) if transforms_dir else sample_dir / "transforms"
    label_zarr = Path(label_zarr_path) if label_zarr_path else sample_dir / "upsampled_atlas_label.zarr"
    if label_zarr_path is None and not label_zarr.exists():
        label_zarr = None

    if atlas_image_path is None or atlas_label_path is None:
        try:
            from pipeline_modules.utils.data_paths import reference_dir

            ref_dir = reference_dir()
            atlas_image_path = atlas_image_path or ref_dir / "atlas.tiff"
            atlas_label_path = atlas_label_path or ref_dir / "atlas_label.tiff"
        except Exception as exc:  # noqa: BLE001 - optional conveniences
            logger.warning("Reference dir unavailable: %s", exc)

    anchor: ApAnchor | None = None
    if bregma_index is None and bregma_offset_mm is None and use_transforms and transforms_dir.exists():
        try:
            anchor = map_bregma_through_transforms(
                transforms_dir=transforms_dir,
                atlas_image_path=atlas_image_path,
                voxel_xyz_um=voxel,
                native_shape_zyx=native_shape,
                label_zarr_path=label_zarr,
                atlas_label_path=atlas_label_path,
            )
            logger.info(
                "Transform-anchored bregma at AP index %.1f (anterior=%s, label check: %s)",
                anchor.index, anchor.anterior, anchor.label_check,
            )
        except Exception as exc:  # noqa: BLE001 - anchor is an enhancement, fall back
            logger.warning("Transform anchor unavailable (%s); a manual anchor is required.", exc)
            anchor = None
    if anchor is None:
        anchor = resolve_manual_anchor(
            n_ap_level=n_ap,
            n_ap_level0=native_shape[1],
            level=level,
            ap_vox_um=vox_ap,
            bregma_index=bregma_index,
            bregma_offset_mm=bregma_offset_mm,
            anterior=anterior,
        )

    half_window_vox = (thickness_mm / 2.0) * 1000.0 / (vox_ap * scale)
    out_dir = Path(output_dir) if output_dir else sample_dir / "visualization" / "bregma_mip"

    label_arr = None
    if overlay_labels and label_zarr is not None and label_zarr.exists() and level == 0:
        candidate = open_zarr_dataset(label_zarr, dataset_name=dataset)
        if tuple(int(v) for v in candidate.shape) == (n_dv, n_ap, n_ml):
            label_arr = candidate
        else:
            logger.warning(
                "Label zarr shape %s differs from signal %s; skipping overlay.",
                candidate.shape, (n_dv, n_ap, n_ml),
            )

    outputs: list[dict[str, object]] = []
    for bregma in bregma_mm:
        base_index = anchor.index_for(bregma, vox_ap, level=level)
        centers = coronal_row_ap_centers(
            n_dv=n_dv,
            base_ap_index=base_index,
            anterior=anchor.anterior,
            tilt_deg=tilt_deg,
            dv_vox_um=vox_dv * scale,
            ap_vox_um=vox_ap * scale,
            tilt_ref_row=tilt_ref_row,
        )
        if centers.max() < 0 or centers.min() > n_ap - 1:
            raise AnchorError(
                f"Bregma {bregma:+.2f} mm maps to AP index {base_index:.1f}, outside 0..{n_ap - 1}"
            )
        if centers.min() < 0 or centers.max() > n_ap - 1:
            logger.warning("Tilted window extends past the AP axis for bregma %+.2f; clipping.", bregma)
        mip = compute_coronal_slab_mip(
            arr, row_ap_centers=centers, half_window_vox=half_window_vox, dv_block=dv_block
        )

        lo = int(np.clip(np.floor(centers.min() - half_window_vox), 0, n_ap - 1))
        hi = int(np.clip(np.ceil(centers.max() + half_window_vox), 0, n_ap - 1))
        meta = RenderMeta(
            bregma_mm=bregma,
            thickness_mm=thickness_mm,
            tilt_deg=tilt_deg,
            anchor=anchor,
            voxel_xyz_um=voxel,
            voxel_source=voxel_source,
            ap_lo=lo,
            ap_hi=hi,
            vmax=resolve_display_vmax(mip, percentile=percentile, explicit=vmax),
            zarr_path=zarr_path,
            level=level,
        )

        boundary = None
        if label_arr is not None:
            center_row = int(np.clip(round((n_dv - 1) / 2.0), 0, n_dv - 1))
            center_ap = int(np.clip(round(centers[center_row]), 0, n_ap - 1))
            boundary = label_boundary_mask(np.asarray(label_arr[:, center_ap, :]))

        atlas_image = None
        if atlas_panel:
            atlas_image = _atlas_coronal_image(bregma, atlas_label_path)
            if atlas_image is not None:
                meta.atlas_panel = f"coronal AP {bregma:+.2f} mm"

        pool_factor = 1
        if max_pixels > 0 and mip.shape[0] * mip.shape[1] > max_pixels * max_pixels:
            pool_factor = int(math.ceil(math.sqrt(mip.shape[0] * mip.shape[1]) / max_pixels))
        out_path = out_dir / (
            f"{sample_dir.name}_{zarr_path.stem}_bregma{bregma:+.2f}mm_thick{thickness_mm:.2f}mm.png"
        )
        render_bregma_png(
            mip,
            output_path=out_path,
            meta=meta,
            flip_dv=flip_dv,
            flip_ml=flip_ml,
            boundary_mask=boundary,
            atlas_slice_image=atlas_image,
            pool_factor=pool_factor,
        )
        outputs.append(
            {
                "bregma_mm": bregma,
                "ap_index_center": round(base_index, 1),
                "ap_window": [lo, hi],
                "output": str(out_path),
                "boundary_overlay": boundary is not None,
                "atlas_panel": atlas_image is not None,
                "pool_factor": pool_factor,
            }
        )
        logger.info("Rendered %s", out_path)

    return {
        "zarr": str(zarr_path),
        "level": level,
        "dataset": dataset,
        "voxel_xyz_um": list(voxel),
        "voxel_source": voxel_source,
        "shape_zyx": [n_dv, n_ap, n_ml],
        "anchor_source": anchor.source,
        "anchor_index_level0": round(anchor.index, 1),
        "anterior": anchor.anterior,
        "label_check": anchor.label_check,
        "thickness_mm": thickness_mm,
        "tilt_deg": tilt_deg,
        "outputs": outputs,
    }


def _atlas_coronal_image(bregma_mm: float, atlas_label_path: str | Path | None) -> np.ndarray | None:
    if not atlas_label_path or not Path(atlas_label_path).exists():
        logger.warning("Atlas label not found (%s); skipping atlas panel.", atlas_label_path)
        return None
    try:
        from pipeline_modules.visualization.atlas_slice import AtlasSliceSpec, extract_atlas_slice

        spec = AtlasSliceSpec(plane="coronal", coordinate_system="bregma-mm", coordinate=bregma_mm)
        return extract_atlas_slice(atlas_label_path, spec).image
    except Exception as exc:  # noqa: BLE001 - panel is optional decoration
        logger.warning("Atlas panel failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_bregma_list(text: str) -> list[float]:
    values = [float(p) for p in str(text).split(",") if p.strip()]
    if not values:
        raise argparse.ArgumentTypeError("--bregma-mm needs at least one number")
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Render coronal slab-MIP PNG screenshots of a brain Zarr at bregma AP "
            "coordinates. Sample zarr axes are (z=DV, y=AP, x=ML)."
        )
    )
    parser.add_argument("--zarr", required=True, help="Signal zarr path, e.g. sample/ch1.zarr")
    parser.add_argument(
        "--bregma-mm", required=True, type=parse_bregma_list,
        help="AP mm relative to bregma (anterior positive); comma list allowed",
    )
    parser.add_argument("--thickness-mm", required=True, type=float,
                        help="Slab thickness along AP that the MIP covers")
    parser.add_argument("--bregma-index", type=float, default=None,
                        help="AP index of bregma on the rendered grid (manual anchor)")
    parser.add_argument("--bregma-offset-mm", type=float, default=None,
                        help="Distance from the volume's anterior edge to bregma (manual anchor)")
    parser.add_argument("--anterior", choices=("low", "high"), default="low",
                        help="Which end of AP axis 1 is anterior (default: low)")
    parser.add_argument(
        "--tilt-deg", type=float, default=0.0,
        help="Cutting-angle correction: positive samples more anterior tissue at higher "
             "DV rows; flip sign if boundaries tilt the wrong way",
    )
    parser.add_argument("--tilt-ref-row", type=float, default=None,
                        help="DV row the tilt pivots on (default: volume DV midpoint)")
    parser.add_argument(
        "--voxel-xyz-um", default="",
        help="Native voxel size x,y,z in microns; else multiscales attrs, config, or "
             "repo default 1.8,1.8,2.0",
    )
    parser.add_argument("--config", default=None, help="Pipeline config.json for voxel size")
    parser.add_argument(
        "--level", type=int, default=0,
        help="Zarr pyramid level (0=full res); voxel and indices scale by 2^level",
    )
    parser.add_argument("--dataset", default="0", help="Zarr dataset name of the base level")
    parser.add_argument(
        "--transforms-dir", default=None,
        help="transforms/ dir with fwd_* files; default: <zarr parent>/transforms",
    )
    parser.add_argument("--no-transforms", action="store_true",
                        help="Never use stored transforms; require a manual anchor")
    parser.add_argument("--no-labels", action="store_true",
                        help="Skip the warped atlas-region boundary overlay")
    parser.add_argument("--atlas-panel", action="store_true",
                        help="Add a side-by-side Allen atlas coronal slice at the same bregma")
    parser.add_argument("--atlas-image", default=None, help="Override atlas.tiff path")
    parser.add_argument("--atlas-label", default=None, help="Override atlas_label.tiff path")
    parser.add_argument("--label-zarr", default=None,
                        help="Override upsampled_atlas_label.zarr path")
    parser.add_argument("--percentile", type=float, default=99.5)
    parser.add_argument("--vmax", type=float, default=None)
    parser.add_argument("--flip-dv", action="store_true", help="Flip image vertically for display")
    parser.add_argument("--flip-ml", action="store_true",
                        help="Flip image horizontally for display")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--max-pixels", type=int, default=3000,
                        help="Cap long image side via max-pooling (0 disables; default 3000)")
    parser.add_argument("--dv-block", type=int, default=128)
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = build_parser().parse_args(argv)
    try:
        payload = render_bregma_mips(
            zarr_path=args.zarr,
            bregma_mm=args.bregma_mm,
            thickness_mm=args.thickness_mm,
            bregma_index=args.bregma_index,
            bregma_offset_mm=args.bregma_offset_mm,
            anterior=args.anterior,
            tilt_deg=args.tilt_deg,
            tilt_ref_row=args.tilt_ref_row,
            voxel_xyz_um=args.voxel_xyz_um,
            config_path=args.config,
            level=args.level,
            dataset=args.dataset,
            transforms_dir=args.transforms_dir,
            atlas_image_path=args.atlas_image,
            atlas_label_path=args.atlas_label,
            label_zarr_path=args.label_zarr,
            use_transforms=not args.no_transforms,
            overlay_labels=not args.no_labels,
            atlas_panel=args.atlas_panel,
            percentile=args.percentile,
            vmax=args.vmax,
            flip_dv=args.flip_dv,
            flip_ml=args.flip_ml,
            output_dir=args.output_dir,
            max_pixels=args.max_pixels,
            dv_block=args.dv_block,
        )
    except (AnchorError, ValueError, FileNotFoundError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
