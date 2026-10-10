"""Render coronal slab-MIP screenshots from a brain Zarr at bregma AP coordinates.

The tool intentionally avoids re-running registration. Two spaces:

* **atlas (default, ``--space atlas``)**: the slab is cut in standard atlas
  space (AP window around the bregma coordinate), every atlas-grid point of
  the slab is mapped back to native image space with the stored pipeline
  ``transforms/`` (fwd, atlas->downsample grid, then the same index rescale
  the pipeline used to write ``upsampled_atlas_label.zarr``), the full-res
  signal Zarr is sampled at those points, and the MIP is taken along the
  atlas AP axis -- an orthogonal camera view down the atlas AP axis. The
  transform convention is cross-validated against ``upsampled_atlas_label.zarr``
  on an interior atlas landmark before any sampling. Several same-sample
  signal zarrs render as an additive false-color composite (``--cmap`` per
  channel); atlas region boundaries draw as thin smooth dashed vector lines
  (``--boundary-color/-linewidth/-style``).
* **native (``--space native``)**: slab taken directly on the sample AP axis
  with a manual bregma anchor (``--bregma-index`` or ``--bregma-offset-mm`` +
  ``--anterior``) and optional ``--tilt-deg`` angle correction; transforms
  only serve as an anchor. Works on any Zarr without registration outputs.

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
from matplotlib.collections import LineCollection  # noqa: E402

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


def _skeleton_paths(skel: np.ndarray) -> list[np.ndarray]:
    """Split a 1-px boolean skeleton into (N, 2) row/col polylines.

    Junction pixels (3+ neighbors) are removed first so every remaining
    connected component is a simple path or loop; each is walked end to end.
    """
    from scipy import ndimage as ndi

    s = np.asarray(skel, dtype=bool)
    if not s.any():
        return []
    kernel = np.ones((3, 3), dtype=np.uint8)
    kernel[1, 1] = 0
    neighbor_count = ndi.convolve(s.astype(np.uint8), kernel, mode="constant")
    junction = s & (neighbor_count >= 3)
    comp_labels, n_comp = ndi.label(s & ~junction, structure=np.ones((3, 3), dtype=bool))

    def neighbors8(pt: tuple[int, int]) -> list[tuple[int, int]]:
        y, x = pt
        out = []
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                q = (y + dy, x + dx)
                if 0 <= q[0] < s.shape[0] and 0 <= q[1] < s.shape[1] and s[q]:
                    out.append(q)
        return out

    paths: list[np.ndarray] = []
    for ci in range(1, n_comp + 1):
        pts = set(map(tuple, np.argwhere(comp_labels == ci)))
        if not pts:
            continue
        ends = [p for p in pts if sum(1 for q in neighbors8(p) if q in pts) == 1]
        cur = ends[0] if ends else next(iter(pts))
        path = [cur]
        pts.discard(cur)
        while pts:
            nxt = [q for q in neighbors8(cur) if q in pts]
            if not nxt:
                break
            cur = nxt[0]
            pts.discard(cur)
            path.append(cur)
        if len(path) >= 2:
            paths.append(np.asarray(path, dtype=np.float64))
    return paths


def label_boundary_lines_mm(
    label_slice: np.ndarray,
    *,
    row_vox_um: float,
    col_vox_um: float,
    upsample: int = 5,
    smoothing: float = 1.2,
) -> list[np.ndarray]:
    """Single smooth centerline per region boundary of a coronal label slice, in mm.

    Rows are DV and cols are ML on the slice; the returned (N, 2) arrays hold
    (x=ML mm, y=DV mm) with the origin at the slice corner, ready to draw on an
    imshow with extent (0, W, H, 0).

    Per-region contour extraction draws every shared border twice (once per
    adjoining region). Instead the boundary band -- voxels where the label
    changes between neighbors -- is upsampled with linear interpolation,
    gaussian-blurred (rounding the 25 um label staircase, the heatmap slice
    renderer recipe) and skeletonized, so each border yields exactly one
    centerline. ``smoothing`` adds a gaussian along each line in label voxels
    (heatmap default 1.2; 0 disables).
    """
    from scipy import ndimage as ndi
    from skimage.morphology import skeletonize

    from pipeline_modules.visualization.atlas_slice import _smooth_contour

    labels = np.asarray(label_slice)
    band = label_boundary_mask(labels).astype(np.float32)
    if not band.any():
        return []
    fine = ndi.zoom(band, float(upsample), order=1)
    fine = ndi.gaussian_filter(fine, sigma=max(1.0, upsample / 4.0))
    skel = skeletonize(fine >= 0.5)
    sigma_fine = float(smoothing) * float(upsample)
    min_pts = max(8, 2 * upsample)
    lines = []
    for path in _skeleton_paths(skel):
        if len(path) < min_pts:
            continue
        if sigma_fine > 0:
            path = _smooth_contour(path, sigma_fine)
        lines.append(np.column_stack([path[:, 1], path[:, 0]]))  # (row, col) -> (x, y)
    x_mm = col_vox_um / (1000.0 * upsample)
    y_mm = row_vox_um / (1000.0 * upsample)
    max_x = (labels.shape[1] - 1) * col_vox_um / 1000.0
    max_y = (labels.shape[0] - 1) * row_vox_um / 1000.0
    scaled = [np.asarray(ln, dtype=np.float64) * (x_mm, y_mm) for ln in lines]
    # Smoothing can push edge-running lines a fraction of a fine voxel outside
    # the slice; clamp so every line stays drawable in-extent.
    return [np.clip(ln, (0.0, 0.0), (max_x, max_y)) for ln in scaled]


MULTI_CHANNEL_PALETTE = ("green", "magenta", "cyan", "yellow", "red", "blue", "orange")


def resolve_channel_cmaps(n_channels: int, cmaps: list[str] | str | None) -> list[str]:
    """Per-channel colormap list; defaults to gray, or a palette for composites."""
    if cmaps is None:
        return ["gray"] if n_channels == 1 else [
            MULTI_CHANNEL_PALETTE[i % len(MULTI_CHANNEL_PALETTE)] for i in range(n_channels)
        ]
    if isinstance(cmaps, str):
        cmaps = [p.strip() for p in cmaps.split(",") if p.strip()]
    if len(cmaps) == 1 and n_channels > 1:
        cmaps = cmaps * n_channels
    if len(cmaps) != n_channels:
        raise ValueError(
            f"{n_channels} channel(s) need {n_channels} colormaps, got: {cmaps}"
        )
    return list(cmaps)


def _channel_colormap(name: str):
    """Matplotlib colormap, or a black->color ramp when ``name`` is a plain color.

    Lets ``--cmap green`` / ``magenta`` work as false-color channels while real
    colormaps (gray, viridis, ...) pass through unchanged.
    """
    try:
        return plt.get_cmap(name)
    except ValueError:
        from matplotlib import colors as mcolors
        from matplotlib.colors import LinearSegmentedColormap

        if not mcolors.is_color_like(name):
            raise
        rgb = np.asarray(mcolors.to_rgb(name), dtype=np.float64)
        peak = float(rgb.max())
        if peak > 0:
            rgb = rgb / peak  # CSS names like 'green' are half-brightness; saturate
        return LinearSegmentedColormap.from_list(name, [(0.0, 0.0, 0.0), tuple(rgb)])


def _compose_rgb(mips: list[np.ndarray], cmaps: list[str], vmaxs: list[float]) -> np.ndarray:
    """Additive false-color composite of per-channel MIPs, clipped to [0, 1]."""
    rgb = None
    for mip, cmap_name, vmax in zip(mips, cmaps, vmaxs):
        cmap = _channel_colormap(cmap_name)
        normed = np.clip(mip.astype(np.float32) / max(float(vmax), 1e-6), 0.0, 1.0)
        layer = np.asarray(cmap(normed), dtype=np.float32)[..., :3]
        rgb = layer if rgb is None else rgb + layer
    return np.clip(rgb, 0.0, 1.0)


def _draw_boundary_lines(
    ax,
    lines: list[np.ndarray] | None,
    *,
    extent_ml_mm: float,
    extent_dv_mm: float,
    color: str,
    linewidth: float,
    style: str,
    flip_dv: bool = False,
    flip_ml: bool = False,
) -> None:
    if not lines or linewidth <= 0:
        return
    segments = []
    for line in lines:
        x = extent_ml_mm - line[:, 0] if flip_ml else line[:, 0]
        y = extent_dv_mm - line[:, 1] if flip_dv else line[:, 1]
        segments.append(np.column_stack([x, y]))
    collection = LineCollection(
        segments,
        colors=color,
        linewidths=linewidth,
        linestyles=(0, (3.0, 2.4)) if style == "dashed" else "solid",
        capstyle="round",
        joinstyle="round",
        antialiaseds=True,
    )
    ax.add_collection(collection)


def resolve_display_vmax(mip: np.ndarray, *, percentile: float, explicit: float | None) -> float:
    if explicit is not None:
        return float(explicit)
    values = mip[mip > 0]
    if values.size == 0:
        return 1.0
    vmax = float(np.percentile(values, percentile))
    return vmax if vmax > 0 else 1.0


# ---------------------------------------------------------------------------
# Atlas-space slab MIP (slab cut in atlas space, sampled back in native space)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AtlasSlabGeometry:
    """Fine sampling grid of an atlas-space coronal slab, in atlas voxel units."""

    dv: np.ndarray  # (n_rows,) atlas DV positions, 0..n_dv-1
    ml: np.ndarray  # (n_cols,) atlas ML positions
    ap: np.ndarray  # (n_planes,) atlas AP positions spanning the slab thickness
    coarse_ap: np.ndarray  # integer atlas AP planes mapped through the transforms
    ap_center: float


def atlas_slab_geometry(
    *,
    bregma_mm: float,
    thickness_mm: float,
    pitch_um: float,
    atlas_shape_dv_ap_ml: tuple[int, int, int],
    bregma_dv_ap_ml: tuple[int, int, int] = DEFAULT_BREGMA_DV_AP_ML,
    atlas_res_um: float = 25.0,
) -> AtlasSlabGeometry:
    n_dv, n_ap, n_ml = atlas_shape_dv_ap_ml
    if thickness_mm <= 0:
        raise ValueError(f"--thickness-mm must be positive, got {thickness_mm}")
    if pitch_um <= 0:
        raise ValueError(f"pitch must be positive, got {pitch_um}")
    # Anterior is positive bregma mm and anterior atlas AP indices are lower.
    ap_center = float(bregma_dv_ap_ml[1]) - float(bregma_mm) * 1000.0 / atlas_res_um
    half_vox = thickness_mm * 1000.0 / 2.0 / atlas_res_um
    ap_lo, ap_hi = ap_center - half_vox, ap_center + half_vox
    if ap_hi < 0 or ap_lo > n_ap - 1:
        raise AnchorError(
            f"Bregma {bregma_mm:+.2f} mm maps to atlas AP index {ap_center:.1f}, "
            f"outside 0..{n_ap - 1}"
        )
    if ap_lo < 0 or ap_hi > n_ap - 1:
        logger.warning("Slab extends past the atlas AP axis for bregma %+.2f; clipping.", bregma_mm)
    n_rows = int(math.floor((n_dv - 1) * atlas_res_um / pitch_um)) + 1
    n_cols = int(math.floor((n_ml - 1) * atlas_res_um / pitch_um)) + 1
    thickness_um = thickness_mm * 1000.0
    n_planes = max(2, int(math.ceil(thickness_um / pitch_um)) + 1)
    dv = np.arange(n_rows, dtype=np.float64) * pitch_um / atlas_res_um
    ml = np.arange(n_cols, dtype=np.float64) * pitch_um / atlas_res_um
    ap = np.linspace(ap_lo, ap_hi, n_planes)
    coarse_ap = np.unique(np.clip(np.rint(ap), 0, n_ap - 1).astype(np.int64))
    return AtlasSlabGeometry(dv=dv, ml=ml, ap=ap, coarse_ap=coarse_ap, ap_center=ap_center)


def interp_coarse_to_fine(
    coarse: np.ndarray,
    *,
    coarse_ap: np.ndarray,
    geometry: AtlasSlabGeometry,
) -> np.ndarray:
    """Upsample a per-plane quantity from the coarse atlas grid to the fine slab grid.

    ``coarse`` has shape ``(n_dv, len(coarse_ap), n_ml)`` — axis order (DV, AP,
    ML), matching ``meshgrid(dv, coarse_ap, ml, indexing="ij")``; the result has
    shape ``(n_rows, n_planes, n_cols)``. Linear interpolation is exact here
    because the SyN deformation is smooth far below the 25 um coarse spacing.
    """
    from scipy import ndimage as ndi

    ap_coord = np.interp(geometry.ap, coarse_ap, np.arange(len(coarse_ap), dtype=np.float64))
    dv_g, ap_g, ml_g = np.meshgrid(geometry.dv, ap_coord, geometry.ml, indexing="ij")
    coords = [np.asarray(c, dtype=np.float32) for c in (dv_g, ap_g, ml_g)]
    return ndi.map_coordinates(np.asarray(coarse, dtype=np.float32), coords, order=1, mode="nearest")


def sample_mip_from_native_coords(
    arrays,
    *,
    z: np.ndarray,
    y: np.ndarray,
    x: np.ndarray,
    tile_z: int = 32,
    tile_x: int = 256,
) -> tuple[list[np.ndarray], dict[str, float]]:
    """Max-project native-space samples of a slab into atlas-aligned MIPs.

    ``z/y/x`` are continuous native indices with shape ``(n_rows, n_planes,
    n_cols)``; ``arrays`` is one signal zarr per channel (same shape), each
    yielding its own MIP. Points are grouped into tiles aligned to the zarr
    chunk grid so each tile's bounding box is read once and shared across
    channels; every MIP maxes over the plane (slab-thickness) axis.
    """
    from scipy import ndimage as ndi

    if not arrays:
        raise ValueError("sample_mip_from_native_coords needs at least one channel array")
    shape0 = tuple(int(v) for v in arrays[0].shape)
    for a in arrays[1:]:
        if tuple(int(v) for v in a.shape) != shape0:
            raise ValueError("All channel zarrs must share the same (z, y, x) shape")
    n_z, n_y, n_x = shape0
    n_rows, n_planes, n_cols = z.shape
    z_f = z.ravel()
    y_f = y.ravel()
    x_f = x.ravel()
    valid = (z_f >= 0) & (z_f <= n_z - 1) & (y_f >= 0) & (y_f <= n_y - 1) & (x_f >= 0) & (x_f <= n_x - 1)
    keep = np.nonzero(valid)[0]
    z_k, y_k, x_k = z_f[keep], y_f[keep], x_f[keep]
    row_stride = n_planes * n_cols
    rows_k = keep // row_stride
    rem = keep % row_stride
    planes_k = rem // n_cols
    cols_k = rem % n_cols

    zr = np.rint(z_k).astype(np.int64)
    xr = np.rint(x_k).astype(np.int64)
    n_tx = n_x // tile_x + 2
    keys = (zr // tile_z) * n_tx + (xr // tile_x)
    order = np.argsort(keys, kind="stable")
    z_s, y_s, x_s = z_k[order], y_k[order], x_k[order]
    rows_s, planes_s, cols_s = rows_k[order], planes_k[order], cols_k[order]
    uniq, starts = np.unique(keys[order], return_index=True)
    bounds = list(starts) + [len(keep)]

    outs = [
        np.zeros((n_rows, n_cols), dtype=np.result_type(a.dtype, np.uint16)) for a in arrays
    ]
    for key, i0, i1 in zip(uniq, bounds[:-1], bounds[1:]):
        tz, tx = int(key) // n_tx, int(key) % n_tx
        zg, yg, xg = z_s[i0:i1], y_s[i0:i1], x_s[i0:i1]
        zs = max(0, tz * tile_z - 2)
        ze = min(n_z, (tz + 1) * tile_z + 3)
        xs = max(0, tx * tile_x - 2)
        xe = min(n_x, (tx + 1) * tile_x + 3)
        ys = max(0, int(math.floor(yg.min())) - 2)
        ye = min(n_y, int(math.ceil(yg.max())) + 3)
        regions = [np.asarray(a[zs:ze, ys:ye, xs:xe]) for a in arrays]
        for plane in np.unique(planes_s[i0:i1]):
            m = planes_s[i0:i1] == plane
            rz, ry, rx = zg[m] - zs, yg[m] - ys, xg[m] - xs
            r, c = rows_s[i0:i1][m], cols_s[i0:i1][m]
            for ci, region in enumerate(regions):
                vals = ndi.map_coordinates(
                    region, [rz, ry, rx], order=1, mode="constant", cval=0, prefilter=False
                )
                outs[ci][r, c] = np.maximum(outs[ci][r, c], vals)

    stats = {
        "valid_fraction": float(keep.size) / float(z_f.size),
        "tiles": int(uniq.size),
    }
    return outs, stats


@dataclass
class AtlasNativeMapper:
    """Maps atlas voxel coordinates (DV, AP, ML) to native zarr indices.

    The pipeline registers with fixed = downsampled sample and moving = atlas
    in an index-unit physical space (both NIfTIs carry spacing/origin as
    written on disk). fwd transforms therefore map atlas voxel coordinates to
    downsample-grid indices; the rescale to native indices mirrors exactly how
    ``upsampled_atlas_label.zarr`` was produced (linspace endpoints on the DV
    axis, (i + 0.5) * T/S - 0.5 on AP/ML).
    """

    transformlist: list[Path]
    convention: str
    atlas_origin_xyz: tuple[float, float, float]  # ANTs components (x=ML, y=AP, z=DV)
    atlas_spacing_xyz: tuple[float, float, float]
    fixed_origin_xyz: tuple[float, float, float]
    fixed_spacing_xyz: tuple[float, float, float]
    fixed_shape_xyz: tuple[int, int, int]  # (ML, AP, DV) downsample grid
    native_shape_zyx: tuple[int, int, int]
    label_check: str = "skipped"
    chunk_points: int = 250_000

    def _native_scales(self) -> tuple[float, float, float]:
        """Per-axis factors (DV, AP, ML) mapping downsample indices to native."""
        t_z, t_y, t_x = self.native_shape_zyx
        s_x, s_y, s_z = self.fixed_shape_xyz
        f_z = (t_z - 1.0) / (s_z - 1.0) if s_z > 1 else float(t_z)
        return f_z, t_y / s_y, t_x / s_x

    def map(self, dv: np.ndarray, ap: np.ndarray, ml: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Map atlas coordinates (any shape, broadcast together) to native (z, y, x)."""
        import ants
        import pandas as pd

        dv = np.broadcast_to(np.asarray(dv, dtype=np.float64), np.broadcast_shapes(dv.shape, ap.shape, ml.shape))
        ap = np.broadcast_to(np.asarray(ap, dtype=np.float64), dv.shape)
        ml = np.broadcast_to(np.asarray(ml, dtype=np.float64), dv.shape)
        flat_dv, flat_ap, flat_ml = dv.ravel(), ap.ravel(), ml.ravel()
        out = np.empty((3, flat_dv.size), dtype=np.float64)
        ao_x, ao_y, ao_z = self.atlas_origin_xyz
        as_x, as_y, as_z = self.atlas_spacing_xyz
        fo_x, fo_y, fo_z = self.fixed_origin_xyz
        fs_x, fs_y, fs_z = self.fixed_spacing_xyz
        f_z, f_y, f_x = self._native_scales()
        for i0 in range(0, flat_dv.size, self.chunk_points):
            i1 = min(i0 + self.chunk_points, flat_dv.size)
            points = pd.DataFrame(
                {
                    "x": ao_x + as_x * flat_ml[i0:i1],
                    "y": ao_y + as_y * flat_ap[i0:i1],
                    "z": ao_z + as_z * flat_dv[i0:i1],
                }
            )
            res = ants.apply_transforms_to_points(
                3, points, [str(p) for p in self.transformlist]
            )
            down_ml = (res["x"].to_numpy(dtype=np.float64) - fo_x) / fs_x
            down_ap = (res["y"].to_numpy(dtype=np.float64) - fo_y) / fs_y
            down_dv = (res["z"].to_numpy(dtype=np.float64) - fo_z) / fs_z
            out[0, i0:i1] = down_dv * f_z
            out[1, i0:i1] = (down_ap + 0.5) * f_y - 0.5
            out[2, i0:i1] = (down_ml + 0.5) * f_x - 0.5
        return out[0].reshape(dv.shape), out[1].reshape(dv.shape), out[2].reshape(dv.shape)


def _read_fixed_grid_nii(path: Path) -> tuple[tuple[float, float, float], tuple[float, float, float], tuple[int, int, int]]:
    import ants

    img = ants.image_read(str(path))
    img.set_direction(np.eye(3))
    return tuple(float(v) for v in img.origin), tuple(float(v) for v in img.spacing), tuple(int(v) for v in img.shape)


def resolve_atlas_native_mapper(
    *,
    sample_dir: str | Path,
    transforms_dir: str | Path,
    atlas_image_path: str | Path,
    atlas_label_path: str | Path,
    label_zarr_path: str | Path,
    native_shape_zyx: tuple[int, int, int],
    fixed_nii: str | Path | None = None,
    chunk_points: int = 250_000,
) -> AtlasNativeMapper:
    """Build an atlas->native point mapper validated on an interior landmark."""
    import ants

    transforms_dir = Path(transforms_dir)
    fwd = sorted(transforms_dir.glob("fwd_*"), key=lambda p: p.name)
    inv = sorted(transforms_dir.glob("inv_*"), key=lambda p: p.name)
    if not fwd:
        raise AnchorError(f"No fwd_* transforms in {transforms_dir}")

    label_arr = None
    if label_zarr_path and Path(label_zarr_path).exists():
        candidate = open_zarr_dataset(Path(label_zarr_path))
        if tuple(int(v) for v in candidate.shape) == tuple(native_shape_zyx):
            label_arr = candidate
    if label_arr is None:
        raise AnchorError(
            "Atlas-space rendering requires a shape-matching upsampled_atlas_label.zarr "
            "for validation; re-run registration or use --space native."
        )

    landmark = _atlas_interior_landmark(atlas_label_path)
    if landmark is None:
        raise AnchorError("Could not pick an interior atlas landmark for validation; use --space native.")
    (l_dv, l_ap, l_ml), expected = landmark

    if fixed_nii is not None:
        fixed_path = Path(fixed_nii)
    else:
        sample_dir = Path(sample_dir)
        candidates = sorted(sample_dir.glob("*_downsample/volume.nii.gz"))
        if not candidates:
            candidates = sorted(transforms_dir.glob("fwd_*Warp.nii.gz"))
        if not candidates:
            raise AnchorError(
                "No fixed-grid NIfTI found (expected <sample>/*_downsample/volume.nii.gz); "
                "pass --fixed-nii or use --space native."
            )
        fixed_path = candidates[0]
    fixed_origin, fixed_spacing, fixed_shape = _read_fixed_grid_nii(fixed_path)

    atlas = ants.image_read(str(atlas_image_path))
    atlas.set_direction(np.eye(3))
    atlas_origin = tuple(float(v) for v in atlas.origin)
    atlas_spacing = tuple(float(v) for v in atlas.spacing)

    def make(transformlist: list[Path]) -> AtlasNativeMapper:
        return AtlasNativeMapper(
            transformlist=transformlist,
            convention="",
            atlas_origin_xyz=atlas_origin,
            atlas_spacing_xyz=atlas_spacing,
            fixed_origin_xyz=fixed_origin,
            fixed_spacing_xyz=fixed_spacing,
            fixed_shape_xyz=fixed_shape,
            native_shape_zyx=native_shape_zyx,
            chunk_points=chunk_points,
        )

    def landmark_agrees(mapper: AtlasNativeMapper) -> bool:
        m_z, m_y, m_x = mapper.map(np.array([l_dv]), np.array([l_ap]), np.array([l_ml]))
        z, y, x = int(round(float(m_z[0]))), int(round(float(m_y[0]))), int(round(float(m_x[0])))
        n_dv, n_ap, n_ml = native_shape_zyx
        if not (0 <= z < n_dv and 0 <= y < n_ap and 0 <= x < n_ml):
            return False
        # Tolerance of roughly one 25 um downsample voxel per axis.
        f_z, f_y, f_x = mapper._native_scales()
        r, c, k = (max(1, int(round(f))) for f in (f_z, f_y, f_x))
        box = np.asarray(
            label_arr[max(0, z - r) : z + r + 1, max(0, y - c) : y + c + 1, max(0, x - k) : x + k + 1]
        )
        return bool(np.any(box == expected))

    candidates: list[tuple[str, list[Path]]] = [("fwd", fwd)]
    if inv:
        candidates.append(("inv", inv))
    candidates.extend([("fwd-reversed", fwd[::-1]), ("inv-reversed", inv[::-1])])
    chosen = next(((name, tl) for name, tl in candidates if tl and landmark_agrees(make(tl))), None)
    if chosen is None:
        raise AnchorError(
            "Stored transforms failed the interior-landmark label check under every "
            "transform-list convention; use --space native with a manual anchor."
        )
    convention, transformlist = chosen
    mapper = make(transformlist)
    mapper.convention = convention
    mapper.label_check = f"ok via interior landmark (region {expected}, convention {convention}, fixed grid {fixed_path.name})"
    return mapper


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
    cmap: str = "gray",
    boundary_lines: list[np.ndarray] | None = None,
    boundary_color: str = "white",
    boundary_linewidth: float = 0.35,
    boundary_style: str = "dashed",
    atlas_slice_image: np.ndarray | None = None,
    pool_factor: int = 1,
    dpi: int = 200,
) -> Path:
    if pool_factor > 1:
        mip = _max_pool(mip, pool_factor)
    image = np.flipud(mip) if flip_dv else mip
    image = np.fliplr(image) if flip_ml else image

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
        image, cmap=cmap, vmin=0.0, vmax=meta.vmax,
        extent=(0, extent_ml_mm, extent_dv_mm, 0), interpolation="nearest",
    )
    _draw_boundary_lines(
        ax,
        boundary_lines,
        extent_ml_mm=extent_ml_mm,
        extent_dv_mm=extent_dv_mm,
        color=boundary_color,
        linewidth=boundary_linewidth,
        style=boundary_style,
        flip_dv=flip_dv,
        flip_ml=flip_ml,
    )
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


def render_atlas_slab_png(
    mips: list[np.ndarray],
    *,
    output_path: Path,
    bregma_mm: float,
    thickness_mm: float,
    pitch_um: float,
    atlas_shape_dv_ap_ml: tuple[int, int, int],
    atlas_res_um: float,
    vmaxs: list[float],
    cmaps: list[str],
    zarr_paths: list[Path],
    mapper_info: str,
    boundary_lines: list[np.ndarray] | None = None,
    boundary_color: str = "white",
    boundary_linewidth: float = 0.35,
    boundary_style: str = "dashed",
    flip_dv: bool = False,
    flip_ml: bool = False,
    pool_factor: int = 1,
    dpi: int = 200,
) -> Path:
    if pool_factor > 1:
        mips = [_max_pool(m, pool_factor) for m in mips]
    image = _compose_rgb(mips, cmaps, vmaxs)
    image = np.flipud(image) if flip_dv else image
    image = np.fliplr(image) if flip_ml else image

    n_dv, n_ap, n_ml = atlas_shape_dv_ap_ml
    extent_ml_mm = n_ml * atlas_res_um / 1000.0
    extent_dv_mm = n_dv * atlas_res_um / 1000.0
    aspect = extent_ml_mm / max(extent_dv_mm, 1e-6)
    panel_h = 8.0
    fig_w = panel_h * aspect + max(panel_h * aspect * 0.35, 2.5)
    fig, ax = plt.subplots(1, 1, figsize=(fig_w, panel_h), dpi=dpi)
    ax.set_facecolor("black")

    ax.imshow(
        image, extent=(0, extent_ml_mm, extent_dv_mm, 0), interpolation="nearest",
    )
    _draw_boundary_lines(
        ax,
        boundary_lines,
        extent_ml_mm=extent_ml_mm,
        extent_dv_mm=extent_dv_mm,
        color=boundary_color,
        linewidth=boundary_linewidth,
        style=boundary_style,
        flip_dv=flip_dv,
        flip_ml=flip_ml,
    )
    _draw_scale_bar(ax, extent_ml_mm, extent_dv_mm)

    channel_summary = " + ".join(
        f"{p.stem}[{cmap}] vmax {vmax:.0f}"
        for p, cmap, vmax in zip(zarr_paths, cmaps, vmaxs)
    )
    ax.set_title(
        f"Atlas-space slab MIP  |  bregma AP {bregma_mm:+.2f} mm  |  slab {thickness_mm:.2f} mm",
        color="white", fontsize=11,
    )
    ax.text(
        0.5, -0.04,
        (
            f"orthogonal projection along atlas AP  |  pitch {pitch_um:g} um  "
            f"atlas {n_dv}x{n_ap}x{n_ml} @ {atlas_res_um:g} um\n"
            f"{mapper_info}\n{channel_summary}"
        ),
        transform=ax.transAxes, ha="center", va="top", color="0.75", fontsize=7,
    )
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("0.4")
    fig.patch.set_facecolor("black")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, facecolor="black", bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    return output_path


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def render_bregma_mips(
    *,
    zarr_path: str | Path,
    bregma_mm: list[float],
    thickness_mm: float,
    space: str = "native",
    pitch_um: float = 5.0,
    cmaps: list[str] | str | None = None,
    boundary_color: str = "white",
    boundary_linewidth: float = 0.35,
    boundary_style: str = "dashed",
    boundary_smooth: float = 1.2,
    fixed_nii: str | Path | None = None,
    chunk_points: int = 250_000,
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
    if space == "atlas":
        return render_atlas_space_mips(
            zarr_path=zarr_path,
            bregma_mm=bregma_mm,
            thickness_mm=thickness_mm,
            pitch_um=pitch_um,
            cmaps=cmaps,
            transforms_dir=transforms_dir,
            atlas_image_path=atlas_image_path,
            atlas_label_path=atlas_label_path,
            label_zarr_path=label_zarr_path,
            fixed_nii=fixed_nii,
            chunk_points=chunk_points,
            overlay_labels=overlay_labels,
            boundary_color=boundary_color,
            boundary_linewidth=boundary_linewidth,
            boundary_style=boundary_style,
            boundary_smooth=boundary_smooth,
            percentile=percentile,
            vmax=vmax,
            flip_dv=flip_dv,
            flip_ml=flip_ml,
            output_dir=output_dir,
            max_pixels=max_pixels,
        )
    if space != "native":
        raise ValueError(f"space must be 'atlas' or 'native', got {space}")
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
            boundary = label_boundary_lines_mm(
                np.asarray(label_arr[:, center_ap, :]),
                row_vox_um=vox_dv,
                col_vox_um=vox_ml,
                smoothing=boundary_smooth,
            )

        atlas_image = None
        if atlas_panel:
            atlas_image = _atlas_coronal_image(bregma, atlas_label_path)
            if atlas_image is not None:
                meta.atlas_panel = f"coronal AP {bregma:+.2f} mm"

        pool_factor = 1
        if max_pixels > 0 and mip.shape[0] * mip.shape[1] > max_pixels * max_pixels:
            pool_factor = int(math.ceil(math.sqrt(mip.shape[0] * mip.shape[1]) / max_pixels))
        out_path = out_dir / (
            f"{sample_dir.name}_{zarr_path.stem}_bregma{bregma:+.2f}mm_thick{thickness_mm:.2f}mm_{boundary_style}.png"
        )
        render_bregma_png(
            mip,
            output_path=out_path,
            meta=meta,
            flip_dv=flip_dv,
            flip_ml=flip_ml,
            cmap=resolve_channel_cmaps(1, cmaps)[0] if cmaps else "gray",
            boundary_lines=boundary,
            boundary_color=boundary_color,
            boundary_linewidth=boundary_linewidth,
            boundary_style=boundary_style,
            atlas_slice_image=atlas_image,
            pool_factor=pool_factor,
        )
        outputs.append(
            {
                "bregma_mm": bregma,
                "ap_index_center": round(base_index, 1),
                "ap_window": [lo, hi],
                "output": str(out_path),
                "boundary_overlay": bool(boundary),
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


def _atlas_label_shape_dv_ap_ml(atlas_label_path: str | Path) -> tuple[int, int, int]:
    import tifffile

    with tifffile.TiffFile(str(atlas_label_path)) as tif:
        n_pages = len(tif.pages)
        page_shape = tuple(int(v) for v in tif.pages[0].shape)
    if len(page_shape) != 2:
        raise ValueError(f"Atlas label pages must be 2D, got {page_shape}")
    n_ap, n_ml = page_shape
    return n_pages, n_ap, n_ml


def render_atlas_space_mips(
    *,
    zarr_path: str | Path | list[str | Path],
    bregma_mm: list[float],
    thickness_mm: float,
    pitch_um: float = 5.0,
    cmaps: list[str] | str | None = None,
    transforms_dir: str | Path | None = None,
    atlas_image_path: str | Path | None = None,
    atlas_label_path: str | Path | None = None,
    label_zarr_path: str | Path | None = None,
    fixed_nii: str | Path | None = None,
    chunk_points: int = 250_000,
    overlay_labels: bool = True,
    boundary_color: str = "white",
    boundary_linewidth: float = 0.35,
    boundary_style: str = "dashed",
    boundary_smooth: float = 1.2,
    percentile: float = 99.5,
    vmax: float | None = None,
    flip_dv: bool = False,
    flip_ml: bool = False,
    output_dir: str | Path | None = None,
    max_pixels: int = 3000,
    tile_z: int = 32,
    tile_x: int = 256,
) -> dict[str, object]:
    """Slab cut in atlas space, sampled back into native space, orthogonal AP MIP.

    ``zarr_path`` accepts one signal zarr or several (comma-separated string or
    list) for a false-color multi-channel composite; all channels must be zarrs
    of the same sample (same registration outputs and shape), each drawn with
    its own colormap from ``cmaps``.
    """
    if isinstance(zarr_path, str):
        zarr_paths = [Path(p.strip()) for p in zarr_path.split(",") if p.strip()]
    elif isinstance(zarr_path, Path):
        zarr_paths = [zarr_path]
    else:
        zarr_paths = [Path(p) for p in zarr_path]
    if not zarr_paths:
        raise ValueError("At least one signal zarr is required")
    sample_dir = zarr_paths[0].parent
    if any(p.parent != sample_dir for p in zarr_paths):
        raise ValueError(
            "Multi-channel rendering needs zarrs of the same sample directory "
            f"(got parents: {sorted({str(p.parent) for p in zarr_paths})})"
        )
    cmaps = resolve_channel_cmaps(len(zarr_paths), cmaps)

    arrays = [open_zarr_dataset(p) for p in zarr_paths]
    for p, arr in zip(zarr_paths, arrays):
        if arr.ndim != 3:
            raise ValueError(f"Expected a 3D (z, y, x) zarr, got shape {arr.shape} for {p}")
    native_shape = tuple(int(v) for v in arrays[0].shape)

    transforms_dir = Path(transforms_dir) if transforms_dir else sample_dir / "transforms"
    label_zarr = Path(label_zarr_path) if label_zarr_path else sample_dir / "upsampled_atlas_label.zarr"

    if atlas_image_path is None or atlas_label_path is None:
        from pipeline_modules.utils.data_paths import reference_dir

        ref_dir = reference_dir()
        atlas_image_path = atlas_image_path or ref_dir / "atlas.tiff"
        atlas_label_path = atlas_label_path or ref_dir / "atlas_label.tiff"
    atlas_image_path, atlas_label_path = Path(atlas_image_path), Path(atlas_label_path)

    atlas_shape = _atlas_label_shape_dv_ap_ml(atlas_label_path)
    atlas_res_um = 25.0
    mapper = resolve_atlas_native_mapper(
        sample_dir=sample_dir,
        transforms_dir=transforms_dir,
        atlas_image_path=atlas_image_path,
        atlas_label_path=atlas_label_path,
        label_zarr_path=label_zarr,
        native_shape_zyx=native_shape,
        fixed_nii=fixed_nii,
        chunk_points=chunk_points,
    )
    logger.info(
        "Atlas->native mapper ready (%s; bregma at atlas AP index %.1f)",
        mapper.label_check,
        DEFAULT_BREGMA_DV_AP_ML[1],
    )

    out_dir = Path(output_dir) if output_dir else sample_dir / "visualization" / "bregma_mip"
    atlas_label_volume = None
    if overlay_labels:
        import tifffile

        atlas_label_volume = tifffile.imread(str(atlas_label_path))
    boundary_upsample = int(min(8, max(2, round(atlas_res_um / pitch_um))))

    outputs: list[dict[str, object]] = []
    for bregma in bregma_mm:
        geometry = atlas_slab_geometry(
            bregma_mm=bregma,
            thickness_mm=thickness_mm,
            pitch_um=pitch_um,
            atlas_shape_dv_ap_ml=atlas_shape,
        )
        logger.info(
            "Mapping %d coarse atlas planes (AP %s) through transforms...",
            geometry.coarse_ap.size, list(geometry.coarse_ap),
        )
        coarse_ap_idx = geometry.coarse_ap.astype(np.float64)
        dv_grid = np.arange(atlas_shape[0], dtype=np.float64)
        ml_grid = np.arange(atlas_shape[2], dtype=np.float64)
        c_dv, c_ap, c_ml = np.meshgrid(dv_grid, coarse_ap_idx, ml_grid, indexing="ij")
        c_z, c_y, c_x = mapper.map(c_dv, c_ap, c_ml)
        fine_z = interp_coarse_to_fine(c_z, coarse_ap=coarse_ap_idx, geometry=geometry)
        fine_y = interp_coarse_to_fine(c_y, coarse_ap=coarse_ap_idx, geometry=geometry)
        fine_x = interp_coarse_to_fine(c_x, coarse_ap=coarse_ap_idx, geometry=geometry)
        del c_z, c_y, c_x
        mips, sample_stats = sample_mip_from_native_coords(
            arrays, z=fine_z, y=fine_y, x=fine_x, tile_z=tile_z, tile_x=tile_x
        )
        del fine_z, fine_y, fine_x

        channel_infos = []
        vmaxs = []
        for mip, cmap_name, path in zip(mips, cmaps, zarr_paths):
            nonzero_fraction = float(np.count_nonzero(mip)) / float(mip.size)
            eff_vmax = resolve_display_vmax(mip, percentile=percentile, explicit=vmax)
            vmaxs.append(eff_vmax)
            channel_infos.append(
                {
                    "zarr": path.name,
                    "cmap": cmap_name,
                    "vmax": round(eff_vmax, 1),
                    "nonzero_fraction": round(nonzero_fraction, 4),
                }
            )

        boundary = None
        if atlas_label_volume is not None:
            center_ap = int(np.clip(round(geometry.ap_center), 0, atlas_shape[1] - 1))
            label_slice = np.asarray(atlas_label_volume[:, center_ap, :])
            boundary = label_boundary_lines_mm(
                label_slice,
                row_vox_um=atlas_res_um,
                col_vox_um=atlas_res_um,
                upsample=boundary_upsample,
                smoothing=boundary_smooth,
            )

        pool_factor = 1
        if max_pixels > 0 and mips[0].shape[0] * mips[0].shape[1] > max_pixels * max_pixels:
            pool_factor = int(math.ceil(math.sqrt(mips[0].shape[0] * mips[0].shape[1]) / max_pixels))
        stem = "+".join(p.stem for p in zarr_paths)
        out_path = out_dir / (
            f"{sample_dir.name}_{stem}_atlas_bregma{bregma:+.2f}mm_"
            f"thick{thickness_mm:.2f}mm_pitch{pitch_um:g}um_{boundary_style}.png"
        )
        render_atlas_slab_png(
            mips,
            output_path=out_path,
            bregma_mm=bregma,
            thickness_mm=thickness_mm,
            pitch_um=pitch_um,
            atlas_shape_dv_ap_ml=atlas_shape,
            atlas_res_um=atlas_res_um,
            vmaxs=vmaxs,
            cmaps=cmaps,
            zarr_paths=zarr_paths,
            mapper_info=mapper.label_check,
            boundary_lines=boundary,
            boundary_color=boundary_color,
            boundary_linewidth=boundary_linewidth,
            boundary_style=boundary_style,
            flip_dv=flip_dv,
            flip_ml=flip_ml,
            pool_factor=pool_factor,
        )
        outputs.append(
            {
                "bregma_mm": bregma,
                "atlas_ap_center_index": round(geometry.ap_center, 1),
                "atlas_ap_window": [round(float(geometry.ap.min()), 1), round(float(geometry.ap.max()), 1)],
                "planes": int(geometry.ap.size),
                "grid_rows_cols": [int(mips[0].shape[0]), int(mips[0].shape[1])],
                "valid_sample_fraction": round(sample_stats["valid_fraction"], 4),
                "tiles": int(sample_stats["tiles"]),
                "channels": channel_infos,
                "pool_factor": pool_factor,
                "boundary_overlay": bool(boundary),
                "output": str(out_path),
            }
        )
        logger.info("Rendered %s", out_path)

    return {
        "zarr": [str(p) for p in zarr_paths],
        "space": "atlas",
        "shape_zyx": list(native_shape),
        "atlas_shape_dv_ap_ml": list(atlas_shape),
        "atlas_res_um": atlas_res_um,
        "pitch_um": pitch_um,
        "thickness_mm": thickness_mm,
        "cmaps": cmaps,
        "transform_convention": mapper.convention,
        "label_check": mapper.label_check,
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
            "coordinates. Sample zarr axes are (z=DV, y=AP, x=ML). Default --space atlas "
            "cuts the slab in standard atlas space, maps it back to the native image "
            "through the stored registration transforms, and renders an orthogonal "
            "camera MIP along the atlas AP axis."
        )
    )
    parser.add_argument(
        "--zarr", required=True,
        help="Signal zarr path, e.g. sample/ch1.zarr; comma-separate several channels "
             "of the same sample for a false-color composite (e.g. ch1.zarr,ch2.zarr)",
    )
    parser.add_argument(
        "--cmap", default="",
        help="Colormap per channel, comma-separated (e.g. green,magenta); default gray "
             "for one channel or a color palette when several zarrs are given",
    )
    parser.add_argument(
        "--boundary-color", default="white",
        help="Color of the atlas region boundary overlay (default white)",
    )
    parser.add_argument(
        "--boundary-linewidth", type=float, default=0.35,
        help="Boundary line width in points (default 0.35)",
    )
    parser.add_argument(
        "--boundary-style", choices=("dashed", "solid"), default="dashed",
        help="Boundary line style (default dashed)",
    )
    parser.add_argument(
        "--boundary-smooth", type=float, default=1.2,
        help="Boundary smoothing gaussian in atlas label voxels, heatmap-style "
             "(default 1.2; 0 disables)",
    )
    parser.add_argument(
        "--bregma-mm", required=True, type=parse_bregma_list,
        help="AP mm relative to bregma (anterior positive); comma list allowed",
    )
    parser.add_argument("--thickness-mm", required=True, type=float,
                        help="Slab thickness along AP that the MIP covers")
    parser.add_argument(
        "--space", choices=("atlas", "native"), default="atlas",
        help="atlas: slab cut in atlas space, warped back to native space, orthogonal "
             "AP MIP (requires pipeline transforms + upsampled_atlas_label.zarr). "
             "native: slab on the sample AP axis with a manual anchor (default: atlas)",
    )
    parser.add_argument(
        "--pitch-um", type=float, default=5.0,
        help="Atlas-space sampling pitch in microns for --space atlas (default 5)",
    )
    parser.add_argument(
        "--fixed-nii", default=None,
        help="Downsample NIfTI defining the registration fixed grid; default: "
             "<sample>/*_downsample/volume.nii.gz",
    )
    parser.add_argument(
        "--chunk-points", type=int, default=250000,
        help="Points per ANTs point-mapping call for --space atlas",
    )
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
            space=args.space,
            pitch_um=args.pitch_um,
            cmaps=args.cmap or None,
            boundary_color=args.boundary_color,
            boundary_linewidth=args.boundary_linewidth,
            boundary_style=args.boundary_style,
            boundary_smooth=args.boundary_smooth,
            fixed_nii=args.fixed_nii,
            chunk_points=args.chunk_points,
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
