#!/usr/bin/env python3
"""Napari 3-step SpinalJ registration wizard.

Step 1 — Straighten: auto centerline, or edit sparse anchors (interp fills
         neighbors), MIP pick, then straighten (no untwist by default).
Step 2 — Untwist: confirm whether untwist is needed; skip by default.
Step 3 — Register: Flip X/Y/Z with atlas side-by-side preview, then SyNRA
         or landmark UI.

Example:
  python -m pipeline_modules.registration.spinalj_wizard_ui --volume_nii ".../volume.nii.gz" --out_dir ".../_spinalj_wizard" --atlas_dir "S:/Yifu_data/reference/SC_P56_Atlas_10x10x20_v5_2020"
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import subprocess
import sys
import threading
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_ATLAS = r"S:\Yifu_data\reference\SC_P56_Atlas_10x10x20_v5_2020"


def _configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


def load_volume_zyx(path: Path) -> tuple[np.ndarray, np.ndarray, tuple[float, float, float]]:
    import nibabel as nib

    nii = nib.load(str(path))
    xyz = np.asanyarray(nii.dataobj).astype(np.float32)
    aff = np.asarray(nii.affine)
    spacing_xyz = tuple(float(abs(aff[i, i])) for i in range(3))
    zyx = np.transpose(xyz, (2, 1, 0))
    spacing_zyx = (spacing_xyz[2], spacing_xyz[1], spacing_xyz[0])
    return zyx, aff, spacing_zyx


def save_volume_zyx(path: Path, vol_zyx: np.ndarray, spacing_zyx_um: tuple[float, float, float]) -> None:
    import nibabel as nib

    sz, sy, sx = spacing_zyx_um
    affine = np.diag([sx, sy, sz, 1.0])
    xyz = np.transpose(np.asarray(vol_zyx, dtype=np.float32), (2, 1, 0))
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(xyz, affine), str(path))


def load_centerline_csv(path: Path) -> np.ndarray:
    """Load Nx3 (z,y,x) voxels from wizard/preview CSV."""
    from pipeline_modules.registration.spinalj_straighten_register import (
        load_centerline_csv as _load,
    )

    return _load(Path(path))


def write_centerline_csv(path: Path, pts_zyx: np.ndarray, spacing_zyx_um: tuple[float, float, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["z_vox", "y_vox", "x_vox", "z_mm", "y_mm", "x_mm", "unreliable", "extrapolated"])
        for z, y, x in np.asarray(pts_zyx, dtype=np.float64):
            w.writerow(
                [
                    f"{z:.3f}",
                    f"{y:.3f}",
                    f"{x:.3f}",
                    f"{z * spacing_zyx_um[0] / 1000.0:.5f}",
                    f"{y * spacing_zyx_um[1] / 1000.0:.5f}",
                    f"{x * spacing_zyx_um[2] / 1000.0:.5f}",
                    0,
                    0,
                ]
            )


def auto_centerline_zyx(vol_zyx: np.ndarray) -> np.ndarray:
    from pipeline_modules.registration.spinalj_mask_centerline_preview import (
        extract_centerline_zyx,
        make_tissue_mask_zyx,
        reject_centerline_jumps,
        smooth_centerline,
    )

    mask, _ = make_tissue_mask_zyx(vol_zyx, thr_frac_of_p99=0.10, filter_2d_cc=True)
    pts = extract_centerline_zyx(mask, vol_zyx, mode="intensity")
    pts = reject_centerline_jumps(pts, max_jump_vox=12.0)
    pts = smooth_centerline(pts, window=31)
    return pts


def merge_mip_points(
    sag_zy: np.ndarray,
    cor_zx: np.ndarray,
    *,
    shape_zyx: tuple[int, int, int],
) -> np.ndarray:
    """Merge sagittal (z,y) and coronal (z,x) clicks into (z,y,x) by nearest Z."""
    if len(sag_zy) == 0 and len(cor_zx) == 0:
        return np.zeros((0, 3), dtype=np.float64)
    nz, ny, nx = shape_zyx
    zs = sorted(set(int(round(z)) for z, _y in sag_zy) | set(int(round(z)) for z, _x in cor_zx))
    zs = [z for z in zs if 0 <= z < nz]
    sag_map = {int(round(z)): float(y) for z, y in sag_zy}
    cor_map = {int(round(z)): float(x) for z, x in cor_zx}
    out = []
    for z in zs:
        y = sag_map.get(z)
        x = cor_map.get(z)
        if y is None and sag_map:
            keys = np.array(sorted(sag_map))
            y = float(np.interp(z, keys, [sag_map[int(k)] for k in keys]))
        if x is None and cor_map:
            keys = np.array(sorted(cor_map))
            x = float(np.interp(z, keys, [cor_map[int(k)] for k in keys]))
        if y is None:
            y = (ny - 1) / 2.0
        if x is None:
            x = (nx - 1) / 2.0
        out.append((float(z), float(np.clip(y, 0, ny - 1)), float(np.clip(x, 0, nx - 1))))
    pts = np.asarray(out, dtype=np.float64)
    if len(pts) >= 2:
        pts = pts[np.argsort(pts[:, 0])]
    return pts


def densify_from_anchors(
    anchors_zyx: np.ndarray,
    *,
    z_grid: np.ndarray | None = None,
    step_z: float = 1.0,
    shape_zyx: tuple[int, int, int] | None = None,
) -> np.ndarray:
    """Build a dense centerline by interpolating Y/X along Z from sparse anchors.

    Edit 1–2 (or more) anchors; neighbors between them transition smoothly.
    """
    anc = np.asarray(anchors_zyx, dtype=np.float64).reshape(-1, 3)
    if len(anc) == 0:
        return np.zeros((0, 3), dtype=np.float64)
    anc = anc[np.argsort(anc[:, 0])]
    # Collapse duplicate Z (keep last).
    z_u, y_u, x_u = [], [], []
    for z, y, x in anc:
        zi = float(z)
        if z_u and abs(zi - z_u[-1]) < 1e-6:
            y_u[-1], x_u[-1] = float(y), float(x)
        else:
            z_u.append(zi)
            y_u.append(float(y))
            x_u.append(float(x))
    z_a = np.asarray(z_u, dtype=np.float64)
    y_a = np.asarray(y_u, dtype=np.float64)
    x_a = np.asarray(x_u, dtype=np.float64)

    if z_grid is None:
        z0, z1 = float(z_a.min()), float(z_a.max())
        if shape_zyx is not None:
            z0 = max(0.0, z0)
            z1 = min(float(shape_zyx[0] - 1), z1)
        z_grid = np.arange(z0, z1 + 0.5 * step_z, step_z, dtype=np.float64)
    else:
        z_grid = np.asarray(z_grid, dtype=np.float64)

    if len(z_a) == 1:
        y = np.full_like(z_grid, y_a[0])
        x = np.full_like(z_grid, x_a[0])
    else:
        y = np.interp(z_grid, z_a, y_a)
        x = np.interp(z_grid, z_a, x_a)

    if shape_zyx is not None:
        ny, nx = shape_zyx[1], shape_zyx[2]
        y = np.clip(y, 0, ny - 1)
        x = np.clip(x, 0, nx - 1)
    return np.column_stack([z_grid, y, x])


def subsample_anchors(pts: np.ndarray, every: int = 40, *, ends: bool = True) -> np.ndarray:
    """Pick sparse anchors from a dense centerline (for editing)."""
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 3)
    if len(pts) == 0:
        return pts
    every = max(1, int(every))
    idx = list(range(0, len(pts), every))
    if ends and (len(pts) - 1) not in idx:
        idx.append(len(pts) - 1)
    return pts[np.asarray(idx, dtype=int)]


def apply_flips_zyx(vol: np.ndarray, *, flip_x: bool, flip_y: bool, flip_z: bool) -> np.ndarray:
    out = np.asarray(vol)
    if flip_z:
        out = out[::-1, :, :]
    if flip_y:
        out = out[:, ::-1, :]
    if flip_x:
        out = out[:, :, ::-1]
    return np.ascontiguousarray(out)


def _to_u8_preview(arr: np.ndarray) -> np.ndarray:
    a = np.asarray(arr, dtype=np.float32)
    m = a > 0
    if not np.any(m):
        return np.zeros(a.shape, dtype=np.uint8)
    lo, hi = np.percentile(a[m], (1, 99.5))
    if hi <= lo:
        hi = lo + 1.0
    out = np.zeros_like(a, dtype=np.float32)
    out[m] = np.clip((a[m] - lo) / (hi - lo), 0, 1) * 255.0
    return out.astype(np.uint8)


def _contrast_limits_from_volume(vol: np.ndarray) -> tuple[float, float]:
    """Percentile contrast so sparse bright landmarks are visible in napari."""
    a = np.asarray(vol, dtype=np.float32)
    # Subsample for speed on large volumes.
    step = max(1, int(np.prod(a.shape) // 2_000_000))
    sample = a.ravel()[::step]
    pos = sample[sample > 0]
    if pos.size < 32:
        return float(np.min(a)), float(max(np.max(a), 1.0))
    lo, hi = np.percentile(pos, (1.0, 99.8))
    if hi <= lo:
        hi = lo + 1.0
    return float(lo), float(hi)


def load_atlas_template_zyx(atlas_dir: Path, cache_dir: Path) -> tuple[np.ndarray, tuple[float, float, float]]:
    import nibabel as nib

    from pipeline_modules.registration.spinalj_atlas_register import convert_atlas_to_nifti

    paths = convert_atlas_to_nifti(Path(atlas_dir), Path(cache_dir))
    nii = nib.load(str(paths["template"]))
    xyz = np.asanyarray(nii.dataobj).astype(np.float32)
    spacing_xyz = tuple(float(abs(nii.affine[i, i])) for i in range(3))
    zyx = np.transpose(xyz, (2, 1, 0))
    return zyx, (spacing_xyz[2], spacing_xyz[1], spacing_xyz[0])


class SpinalJWizard:
    def __init__(
        self,
        *,
        volume_nii: Path,
        out_dir: Path,
        atlas_dir: Path,
        transform: str = "SyNRA",
        centerline_csv: Path | None = None,
    ) -> None:
        import napari
        from qtpy.QtCore import Qt
        from qtpy.QtGui import QImage, QPixmap
        from qtpy.QtWidgets import (
            QCheckBox,
            QComboBox,
            QGroupBox,
            QHBoxLayout,
            QLabel,
            QListWidget,
            QPushButton,
            QTextEdit,
            QVBoxLayout,
            QWidget,
        )

        self.volume_nii = Path(volume_nii)
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.atlas_dir = Path(atlas_dir)
        self.transform = transform

        self.vol_zyx, self.affine, self.spacing_zyx = load_volume_zyx(self.volume_nii)
        self.centerline = np.zeros((0, 3), dtype=np.float64)
        self.anchors = np.zeros((0, 3), dtype=np.float64)
        self.straight_nii: Path | None = None
        self.straight_zyx: np.ndarray | None = None
        self.straight_spacing: tuple[float, float, float] | None = None
        self.fixed_nii: Path | None = None
        self.need_untwist = False
        self.allow_reflection = False
        self.flip_x = False
        self.flip_y = False
        self.flip_z = False
        self.atlas_zyx: np.ndarray | None = None
        self.step = 1
        self._reg_thread: threading.Thread | None = None
        self._anchor_live = True

        self.viewer = napari.Viewer(title="SpinalJ Wizard (3 steps)")
        clim = _contrast_limits_from_volume(self.vol_zyx)
        self.viewer.add_image(
            self.vol_zyx,
            name="volume",
            colormap="gray",
            blending="additive",
            rendering="mip",
            contrast_limits=clim,
        )
        self.cl_layer = self.viewer.add_points(
            np.zeros((0, 3)),
            name="centerline",
            size=4,
            face_color="cyan",
            border_color="white",
            ndim=3,
            opacity=0.9,
            out_of_slice_display=True,
        )
        self.anchor_layer = self.viewer.add_points(
            np.zeros((0, 3)),
            name="anchors",
            size=10,
            face_color="magenta",
            border_color="yellow",
            ndim=3,
            out_of_slice_display=True,
        )
        # Live crosshair on the current Z slice (always visible while scrolling).
        self.cross_layer = self.viewer.add_shapes(
            None,
            name="cl_crosshair",
            shape_type="line",
            edge_color="cyan",
            edge_width=2,
            opacity=0.95,
            ndim=3,
        )
        self.cl_path_layer = self.viewer.add_shapes(
            None,
            name="cl_path",
            shape_type="path",
            edge_color="cyan",
            edge_width=1.5,
            opacity=0.7,
            ndim=3,
        )

        # ----- dock -----
        dock = QWidget()
        layout = QVBoxLayout(dock)
        self.step_label = QLabel("Step 1/3 — Straighten (centerline)")
        self.step_label.setStyleSheet("font-weight:600; font-size:14px;")
        layout.addWidget(self.step_label)

        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumHeight(100)
        layout.addWidget(self.log)

        # Step 1
        g1 = QGroupBox("Step1 centerline")
        g1l = QVBoxLayout(g1)
        self.btn_auto_cl = QPushButton("1a. Auto centerline")
        self.btn_mip = QPushButton("1b. Manual MIP pick (sparse anchors)")
        self.btn_add_anchor = QPushButton("Add/update anchor at current Z  [A]")
        self.btn_interp = QPushButton("Interp anchors → full centerline")
        self.chk_live_interp = QCheckBox("Live interp when anchors change")
        self.chk_live_interp.setChecked(True)
        self.btn_straighten = QPushButton("1d. Straighten (no untwist)")
        for b in (
            self.btn_auto_cl,
            self.btn_mip,
            self.btn_add_anchor,
            self.btn_interp,
            self.chk_live_interp,
        ):
            g1l.addWidget(b)

        # Fast navigation + slice preview for anchors
        nav_row = QHBoxLayout()
        self.btn_prev_anchor = QPushButton("◀ Prev anchor  [")
        self.btn_next_anchor = QPushButton("Next anchor ▶  ]")
        self.btn_jump_worst = QPushButton("Jump worst bend")
        self.btn_jump_gap = QPushButton("Jump biggest gap")
        nav_row.addWidget(self.btn_prev_anchor)
        nav_row.addWidget(self.btn_next_anchor)
        g1l.addLayout(nav_row)
        nav_row2 = QHBoxLayout()
        nav_row2.addWidget(self.btn_jump_worst)
        nav_row2.addWidget(self.btn_jump_gap)
        g1l.addLayout(nav_row2)

        self.lbl_slice_preview = QLabel("slice preview")
        self.lbl_slice_preview.setMinimumSize(180, 180)
        self.lbl_slice_preview.setAlignment(Qt.AlignCenter)
        self.lbl_slice_preview.setStyleSheet("background:#111; color:#aaa; border:1px solid #444;")
        g1l.addWidget(self.lbl_slice_preview)
        self.lbl_anchor_info = QLabel("anchors: 0")
        self.lbl_anchor_info.setWordWrap(True)
        g1l.addWidget(self.lbl_anchor_info)

        self.list_anchors = QListWidget()
        self.list_anchors.setMaximumHeight(110)
        self.list_anchors.setToolTip("Click an anchor to jump viewer to that Z and show preview")
        g1l.addWidget(self.list_anchors)

        g1l.addWidget(self.btn_straighten)
        layout.addWidget(g1)

        self._anchor_nav_i = 0
        self.chk_auto_jump = QCheckBox("After add/edit: jump to next suggested Z")
        self.chk_auto_jump.setChecked(True)
        g1l.insertWidget(g1l.indexOf(self.btn_straighten), self.chk_auto_jump)

        # Step 2
        g2 = QGroupBox("Step2 untwist")
        g2l = QVBoxLayout(g2)
        self.chk_untwist = QCheckBox("需要解扭（默认不勾选 = 不解扭）")
        self.chk_untwist.setChecked(False)
        self.btn_open_untwist = QPushButton("打开手动解扭界面")
        self.btn_open_untwist.setEnabled(False)
        self.cmb_untwist = QComboBox()
        self.cmb_untwist.addItems(["manual", "atlas"])
        g2l.addWidget(self.chk_untwist)
        g2l.addWidget(self.cmb_untwist)
        g2l.addWidget(self.btn_open_untwist)
        layout.addWidget(g2)

        # Step 3 flip + atlas compare
        g3 = QGroupBox("Step3 flip + atlas compare + register")
        g3l = QVBoxLayout(g3)
        flip_row = QHBoxLayout()
        self.chk_flip_x = QCheckBox("Flip X")
        self.chk_flip_y = QCheckBox("Flip Y")
        self.chk_flip_z = QCheckBox("Flip Z")
        for c in (self.chk_flip_x, self.chk_flip_y, self.chk_flip_z):
            flip_row.addWidget(c)
        g3l.addLayout(flip_row)
        self.chk_reflect = QCheckBox("allow_reflection (ANTs)")
        g3l.addWidget(self.chk_reflect)

        cmp_row = QHBoxLayout()
        self.lbl_sample = QLabel("sample")
        self.lbl_atlas = QLabel("atlas")
        for lbl in (self.lbl_sample, self.lbl_atlas):
            lbl.setMinimumSize(160, 160)
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setStyleSheet("background:#111; color:#aaa; border:1px solid #444;")
            cmp_row.addWidget(lbl)
        g3l.addLayout(cmp_row)
        self.lbl_flip_hint = QLabel("对照 atlas：解剖左右/背腹是否一致；不一致再勾 Flip")
        self.lbl_flip_hint.setWordWrap(True)
        g3l.addWidget(self.lbl_flip_hint)

        self.btn_load_compare = QPushButton("刷新 sample↔atlas 对照")
        self.btn_landmarks = QPushButton("手动点选特征截面 (landmark UI)")
        self.btn_register = QPushButton("自动配准 SyNRA")
        self.btn_preview_qc = QPushButton("预览配准 QC")
        for b in (self.btn_load_compare, self.btn_landmarks, self.btn_register, self.btn_preview_qc):
            g3l.addWidget(b)
        layout.addWidget(g3)

        nav = QHBoxLayout()
        self.btn_prev = QPushButton("← Prev")
        self.btn_next = QPushButton("Next →")
        nav.addWidget(self.btn_prev)
        nav.addWidget(self.btn_next)
        layout.addLayout(nav)
        layout.addStretch(1)

        self.viewer.window.add_dock_widget(dock, name="SpinalJ steps", area="right")

        self.btn_auto_cl.clicked.connect(self.on_auto_centerline)
        self.btn_mip.clicked.connect(self.on_mip_pick)
        self.btn_add_anchor.clicked.connect(self.on_add_anchor_at_z)
        self.btn_interp.clicked.connect(self.on_interp_anchors)
        self.chk_live_interp.stateChanged.connect(self._on_live_toggle)
        self.btn_prev_anchor.clicked.connect(lambda: self.jump_anchor(-1))
        self.btn_next_anchor.clicked.connect(lambda: self.jump_anchor(+1))
        self.btn_jump_worst.clicked.connect(self.jump_worst_bend)
        self.btn_jump_gap.clicked.connect(self.jump_biggest_gap)
        self.list_anchors.currentRowChanged.connect(self._on_anchor_list_row)
        self.btn_straighten.clicked.connect(self.on_straighten)
        self.chk_untwist.stateChanged.connect(self.on_untwist_toggle)
        self.btn_open_untwist.clicked.connect(self.on_open_untwist)
        self.chk_flip_x.stateChanged.connect(self.on_flip_changed)
        self.chk_flip_y.stateChanged.connect(self.on_flip_changed)
        self.chk_flip_z.stateChanged.connect(self.on_flip_changed)
        self.chk_reflect.stateChanged.connect(self.on_flip_changed)
        self.btn_load_compare.clicked.connect(self.refresh_flip_compare)
        self.btn_landmarks.clicked.connect(self.on_landmarks)
        self.btn_register.clicked.connect(self.on_register)
        self.btn_preview_qc.clicked.connect(self.on_preview_qc)
        self.btn_prev.clicked.connect(lambda: self.set_step(self.step - 1))
        self.btn_next.clicked.connect(self.go_next)

        self.anchor_layer.events.data.connect(self._on_anchors_data)
        self.viewer.dims.events.current_step.connect(self._on_dims_step)
        self.viewer.bind_key("[", lambda _v: self.jump_anchor(-1))
        self.viewer.bind_key("]", lambda _v: self.jump_anchor(+1))
        self.viewer.bind_key("a", lambda _v: self.on_add_anchor_at_z())
        self.viewer.bind_key("n", lambda _v: self.jump_anchor(+1))
        self.viewer.bind_key("p", lambda _v: self.jump_anchor(-1))
        self.viewer.bind_key("w", lambda _v: self.jump_worst_bend())
        self.viewer.bind_key("g", lambda _v: self.jump_biggest_gap())

        self.set_step(1)
        self._log(
            f"Loaded {self.volume_nii.name} ZYX={tuple(self.vol_zyx.shape)} "
            f"spacing_um={self.spacing_zyx}"
        )
        if centerline_csv is not None:
            self.load_centerline_from_csv(Path(centerline_csv))
        self._log(
            "Anchors: drag magenta points. Keys: [ ] / n p = prev/next anchor, "
            "A=add at Z, W=worst bend, G=biggest gap."
        )

    def load_centerline_from_csv(self, path: Path) -> None:
        """Load a finished centerline CSV into cyan + magenta layers (skip 1a)."""
        path = Path(path)
        if not path.exists():
            self._log(f"Centerline CSV not found: {path}")
            return
        try:
            pts = load_centerline_csv(path)
            self._set_centerline_layer(pts)
            anc = subsample_anchors(pts, every=max(20, len(pts) // 25))
            self._set_anchors_layer(anc)
            # Keep a copy under out_dir so Straighten always finds it.
            dst = self.out_dir / "centerline_zyx.csv"
            write_centerline_csv(dst, pts, self.spacing_zyx)
            self.refresh_anchor_list()
            self.refresh_slice_preview()
            if len(anc):
                self._anchor_nav_i = 0
                self.jump_anchor(0)
            self._log(
                f"Loaded centerline CSV: {path.name} → {len(pts)} pts, {len(anc)} anchors. "
                "You can edit anchors then Straighten (1d), or Straighten directly."
            )
        except Exception as e:
            self._log(f"Load centerline FAILED: {e}")

    def _log(self, msg: str) -> None:
        logger.info("%s", msg)
        self.log.append(msg)

    def _on_live_toggle(self, *_a) -> None:
        self._anchor_live = bool(self.chk_live_interp.isChecked())

    def set_step(self, step: int) -> None:
        self.step = int(np.clip(step, 1, 3))
        titles = {
            1: "Step 1/3 — Straighten (anchors + interp)",
            2: "Step 2/3 — Confirm untwist (default: skip)",
            3: "Step 3/3 — Flip X/Y/Z vs atlas + register",
        }
        self.step_label.setText(titles[self.step])
        s1, s2, s3 = self.step == 1, self.step == 2, self.step == 3
        for w in (
            self.btn_auto_cl,
            self.btn_mip,
            self.btn_add_anchor,
            self.btn_interp,
            self.chk_live_interp,
            self.btn_prev_anchor,
            self.btn_next_anchor,
            self.btn_jump_worst,
            self.btn_jump_gap,
            self.list_anchors,
            self.chk_auto_jump,
            self.btn_straighten,
        ):
            w.setEnabled(s1)
        self.chk_untwist.setEnabled(s2)
        self.cmb_untwist.setEnabled(s2)
        self.btn_open_untwist.setEnabled(s2 and self.chk_untwist.isChecked())
        for w in (
            self.chk_flip_x,
            self.chk_flip_y,
            self.chk_flip_z,
            self.chk_reflect,
            self.btn_load_compare,
            self.btn_landmarks,
            self.btn_register,
            self.btn_preview_qc,
        ):
            w.setEnabled(s3)
        self.btn_prev.setEnabled(self.step > 1)
        self.btn_next.setEnabled(self.step < 3)
        if s3:
            self.refresh_flip_compare()

    def _sorted_anchors(self) -> np.ndarray:
        anc = np.asarray(self.anchor_layer.data, dtype=np.float64).reshape(-1, 3)
        if len(anc) == 0:
            return anc
        return anc[np.argsort(anc[:, 0])]

    def _centerline_yx_at_z(self, z: float) -> tuple[float, float] | None:
        """YX of dense centerline (else nearby anchor) at Z."""
        z = float(z)
        if len(self.centerline):
            i = int(np.argmin(np.abs(self.centerline[:, 0] - z)))
            return float(self.centerline[i, 1]), float(self.centerline[i, 2])
        anc = self._sorted_anchors()
        if len(anc):
            i = int(np.argmin(np.abs(anc[:, 0] - z)))
            return float(anc[i, 1]), float(anc[i, 2])
        return None

    def refresh_cl_overlay(self) -> None:
        """Update on-viewer crosshair + path so centerline is visible while scrolling."""
        try:
            z = float(self.viewer.dims.current_step[0])
            yx = self._centerline_yx_at_z(z)
            if yx is None:
                self.cross_layer.data = []
            else:
                y, x = yx
                half = 28.0
                self.cross_layer.data = [
                    np.array([[z, y, x - half], [z, y, x + half]], dtype=np.float64),
                    np.array([[z, y - half, x], [z, y + half, x]], dtype=np.float64),
                ]
                self.cross_layer.shape_type = ["line", "line"]
            if len(self.centerline) >= 2:
                step = max(1, len(self.centerline) // 200)
                path = self.centerline[::step]
                if len(path) < 2 or not np.allclose(path[-1], self.centerline[-1]):
                    path = np.vstack([path, self.centerline[-1]])
                self.cl_path_layer.data = [path.astype(np.float64)]
                self.cl_path_layer.shape_type = ["path"]
            else:
                self.cl_path_layer.data = []
        except Exception as e:
            logger.debug("refresh_cl_overlay: %s", e)

    def _goto_zyx(self, z: float, y: float | None = None, x: float | None = None) -> None:
        """Jump dims to Z (and optionally center camera on y,x)."""
        z = float(np.clip(round(z), 0, self.vol_zyx.shape[0] - 1))
        step = list(self.viewer.dims.current_step)
        step[0] = int(z)
        self.viewer.dims.current_step = tuple(step)
        if y is not None and x is not None:
            try:
                # napari camera center is in data coordinates (z, y, x) for 3D.
                self.viewer.camera.center = (z, float(y), float(x))
            except Exception:
                pass
        self.refresh_cl_overlay()
        self.refresh_slice_preview()

    def refresh_slice_preview(self) -> None:
        """Show current transverse slice with crosshair at centerline/anchor."""
        from qtpy.QtGui import QImage, QPixmap

        z = int(round(float(self.viewer.dims.current_step[0])))
        z = int(np.clip(z, 0, self.vol_zyx.shape[0] - 1))
        sl = _to_u8_preview(self.vol_zyx[z])
        # Prefer anchor at this Z, else dense centerline.
        y = x = None
        anc = self._sorted_anchors()
        if len(anc):
            i = int(np.argmin(np.abs(anc[:, 0] - z)))
            if abs(anc[i, 0] - z) <= 1.5:
                y, x = float(anc[i, 1]), float(anc[i, 2])
        if y is None:
            yx = self._centerline_yx_at_z(z)
            if yx is not None:
                y, x = yx
        if y is None:
            y = (sl.shape[0] - 1) / 2.0
            x = (sl.shape[1] - 1) / 2.0
        rgb = np.stack([sl, sl, sl], axis=-1)
        yy, xx = int(round(y)), int(round(x))
        # cyan crosshair
        h, w = sl.shape
        t = 2
        rgb[yy, max(0, xx - 12) : min(w, xx + 13)] = (0, 255, 255)
        rgb[max(0, yy - 12) : min(h, yy + 13), xx] = (0, 255, 255)
        rgb[
            max(0, yy - t) : min(h, yy + t + 1),
            max(0, xx - 12) : min(w, xx + 13),
        ] = (0, 255, 255)
        rgb[
            max(0, yy - 12) : min(h, yy + 13),
            max(0, xx - t) : min(w, xx + t + 1),
        ] = (0, 255, 255)
        # downscale for label
        try:
            import cv2

            scale = min(200 / max(h, 1), 200 / max(w, 1), 1.0)
            nh, nw = max(1, int(h * scale)), max(1, int(w * scale))
            show = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA)
        except Exception:
            show = rgb[:: max(1, h // 180), :: max(1, w // 180)]
        show = np.ascontiguousarray(show)
        qimg = QImage(
            show.data,
            int(show.shape[1]),
            int(show.shape[0]),
            int(show.strides[0]),
            QImage.Format_RGB888,
        )
        self.lbl_slice_preview.setPixmap(QPixmap.fromImage(qimg.copy()))
        self.lbl_slice_preview.setToolTip(f"Z={z} crosshair at y={y:.1f}, x={x:.1f}")

    def refresh_anchor_list(self) -> None:
        anc = self._sorted_anchors()
        self.list_anchors.blockSignals(True)
        self.list_anchors.clear()
        for i, (z, y, x) in enumerate(anc):
            self.list_anchors.addItem(f"{i:02d}  Z={z:.0f}  Y={y:.1f}  X={x:.1f}")
        self.list_anchors.blockSignals(False)
        self.lbl_anchor_info.setText(
            f"anchors: {len(anc)}   (click list / [ ] to jump; preview updates)"
        )
        if len(anc):
            self._anchor_nav_i = int(np.clip(self._anchor_nav_i, 0, len(anc) - 1))

    def _on_dims_step(self, _event=None) -> None:
        if self.step == 1:
            self.refresh_cl_overlay()
            self.refresh_slice_preview()

    def _on_anchor_list_row(self, row: int) -> None:
        anc = self._sorted_anchors()
        if row < 0 or row >= len(anc):
            return
        self._anchor_nav_i = row
        z, y, x = anc[row]
        self._goto_zyx(z, y, x)
        self._log(f"Jump list → Z={z:.0f}")

    def jump_anchor(self, delta: int) -> None:
        anc = self._sorted_anchors()
        if len(anc) == 0:
            self._log("No anchors yet.")
            return
        # If currently near an anchor, step from there; else from closest.
        z_cur = float(self.viewer.dims.current_step[0])
        i_near = int(np.argmin(np.abs(anc[:, 0] - z_cur)))
        if abs(anc[i_near, 0] - z_cur) <= 1.5:
            self._anchor_nav_i = i_near
        self._anchor_nav_i = int((self._anchor_nav_i + delta) % len(anc))
        z, y, x = anc[self._anchor_nav_i]
        self.list_anchors.blockSignals(True)
        self.list_anchors.setCurrentRow(self._anchor_nav_i)
        self.list_anchors.blockSignals(False)
        self._goto_zyx(z, y, x)
        self._log(f"Anchor {self._anchor_nav_i + 1}/{len(anc)} Z={z:.0f}")

    def _suggest_zs(self) -> list[tuple[float, str]]:
        """Suggested Z to edit: large bends and large gaps between anchors."""
        sug: list[tuple[float, str]] = []
        anc = self._sorted_anchors()
        cl = self.centerline
        if len(cl) >= 5:
            # Lateral jump along dense line (physical-ish using equal voxel weight).
            d = np.linalg.norm(np.diff(cl[:, 1:3], axis=0), axis=1)
            # local max every ~window
            for _ in range(min(8, max(1, len(d) // 50))):
                i = int(np.argmax(d))
                if d[i] < 0.5:
                    break
                sug.append((float(cl[i, 0]), f"bend Δyx={d[i]:.1f}"))
                lo, hi = max(0, i - 30), min(len(d), i + 31)
                d[lo:hi] = 0
        if len(anc) >= 2:
            gaps = np.diff(anc[:, 0])
            for _ in range(min(5, len(gaps))):
                i = int(np.argmax(gaps))
                if gaps[i] < 40:
                    break
                z_mid = 0.5 * (anc[i, 0] + anc[i + 1, 0])
                sug.append((float(z_mid), f"gap Δz={gaps[i]:.0f}"))
                gaps[i] = 0
        # unique by Z bucket
        out = []
        seen = set()
        for z, reason in sorted(sug, key=lambda t: t[0]):
            key = int(round(z / 10.0))
            if key in seen:
                continue
            seen.add(key)
            out.append((z, reason))
        return out

    def jump_worst_bend(self) -> None:
        if len(self.centerline) < 5:
            self._log("Need dense centerline first (auto or interp).")
            return
        cl = self.centerline
        d = np.linalg.norm(np.diff(cl[:, 1:3], axis=0), axis=1)
        i = int(np.argmax(d))
        z, y, x = cl[i]
        self._goto_zyx(z, y, x)
        self._log(f"Worst bend @ Z={z:.0f} Δyx={d[i]:.2f} — fix with A or drag nearby anchor")
        self.refresh_slice_preview()

    def jump_biggest_gap(self) -> None:
        anc = self._sorted_anchors()
        if len(anc) < 2:
            self._log("Need ≥2 anchors.")
            return
        gaps = np.diff(anc[:, 0])
        i = int(np.argmax(gaps))
        z_mid = 0.5 * (anc[i, 0] + anc[i + 1, 0])
        # yx from dense centerline if available
        if len(self.centerline):
            j = int(np.argmin(np.abs(self.centerline[:, 0] - z_mid)))
            y, x = self.centerline[j, 1], self.centerline[j, 2]
        else:
            y = 0.5 * (anc[i, 1] + anc[i + 1, 1])
            x = 0.5 * (anc[i, 2] + anc[i + 1, 2])
        self._goto_zyx(z_mid, y, x)
        self._log(f"Biggest gap mid Z={z_mid:.0f} (Δz={gaps[i]:.0f}) — press A to drop anchor")
        self.refresh_slice_preview()

    def _set_centerline_layer(self, pts: np.ndarray) -> None:
        self.centerline = np.asarray(pts, dtype=np.float64).reshape(-1, 3)
        self.cl_layer.data = self.centerline.copy()
        self.refresh_cl_overlay()
        self.refresh_slice_preview()

    def _set_anchors_layer(self, pts: np.ndarray, *, trigger_interp: bool = False) -> None:
        self.anchors = np.asarray(pts, dtype=np.float64).reshape(-1, 3)
        prev = self._anchor_live
        self._anchor_live = False
        self.anchor_layer.data = self.anchors.copy()
        self._anchor_live = prev
        self.refresh_anchor_list()
        if trigger_interp:
            self.on_interp_anchors(silent=True)
        self.refresh_slice_preview()

    def _on_anchors_data(self, _event=None) -> None:
        if not self._anchor_live or self.step != 1:
            return
        self.anchors = np.asarray(self.anchor_layer.data, dtype=np.float64).reshape(-1, 3)
        self.refresh_anchor_list()
        if self.chk_live_interp.isChecked() and len(self.anchors) >= 2:
            self.on_interp_anchors(silent=True)
        self.refresh_slice_preview()

    def on_auto_centerline(self) -> None:
        self._log("Running auto centerline...")
        try:
            pts = auto_centerline_zyx(self.vol_zyx)
            self._set_centerline_layer(pts)
            anc = subsample_anchors(pts, every=max(20, len(pts) // 25))
            self._set_anchors_layer(anc)
            csv_path = self.out_dir / "centerline_zyx.csv"
            write_centerline_csv(csv_path, pts, self.spacing_zyx)
            self._log(f"Auto centerline: {len(pts)} pts, {len(anc)} anchors → {csv_path}")
            self._log("Use Next anchor / Jump worst bend to review quickly.")
            if len(anc):
                self._anchor_nav_i = 0
                self.jump_anchor(0)
        except Exception as e:
            self._log(f"Auto centerline FAILED: {e}")

    def on_add_anchor_at_z(self) -> None:
        """Insert/update an anchor at the viewer's current Z from dense centerline."""
        z = int(round(float(self.viewer.dims.current_step[0])))
        if len(self.centerline) == 0:
            ny, nx = self.vol_zyx.shape[1], self.vol_zyx.shape[2]
            pt = np.array([[z, (ny - 1) / 2.0, (nx - 1) / 2.0]], dtype=np.float64)
        else:
            i = int(np.argmin(np.abs(self.centerline[:, 0] - z)))
            pt = self.centerline[i : i + 1].copy()
            pt[0, 0] = float(z)
        anc = np.asarray(self.anchor_layer.data, dtype=np.float64).reshape(-1, 3)
        if len(anc):
            keep = np.abs(anc[:, 0] - z) > 0.6
            anc = anc[keep]
            anc = np.vstack([anc, pt])
        else:
            anc = pt
        anc = anc[np.argsort(anc[:, 0])]
        self._set_anchors_layer(anc, trigger_interp=True)
        self._log(f"Anchor at Z={z}: {pt[0].round(1).tolist()}")
        # Jump to next place that still needs attention.
        if self.chk_auto_jump.isChecked():
            sug = self._suggest_zs()
            # pick first suggestion far from current z
            nxt = None
            for zs, reason in sug:
                if abs(zs - z) > 15:
                    nxt = (zs, reason)
                    break
            if nxt is None:
                # fall back: next anchor after current
                if len(anc) >= 2:
                    after = anc[anc[:, 0] > z + 0.5]
                    if len(after):
                        self._goto_zyx(after[0, 0], after[0, 1], after[0, 2])
                        self._log(f"Auto-jump next anchor Z={after[0, 0]:.0f}")
                        return
                self._goto_zyx(z, pt[0, 1], pt[0, 2])
            else:
                zs, reason = nxt
                if len(self.centerline):
                    j = int(np.argmin(np.abs(self.centerline[:, 0] - zs)))
                    self._goto_zyx(zs, self.centerline[j, 1], self.centerline[j, 2])
                else:
                    self._goto_zyx(zs)
                self._log(f"Auto-jump suggested Z={zs:.0f} ({reason}) — adjust then press A")
        else:
            self._goto_zyx(z, pt[0, 1], pt[0, 2])

    def on_interp_anchors(self, silent: bool = False) -> None:
        anc = np.asarray(self.anchor_layer.data, dtype=np.float64).reshape(-1, 3)
        if len(anc) < 2:
            if not silent:
                self._log("Need ≥2 anchors to interpolate.")
            return
        # Keep original Z span if we already have a dense line; else anchors span.
        if len(self.centerline) >= 2:
            z_grid = self.centerline[:, 0]
        else:
            z_grid = None
        dense = densify_from_anchors(
            anc,
            z_grid=z_grid,
            step_z=1.0,
            shape_zyx=self.vol_zyx.shape,
        )
        self.anchors = anc[np.argsort(anc[:, 0])]
        self._set_centerline_layer(dense)
        csv_path = self.out_dir / "centerline_zyx.csv"
        write_centerline_csv(csv_path, dense, self.spacing_zyx)
        if not silent:
            self._log(f"Interp: {len(anc)} anchors → {len(dense)} centerline points")

    def on_mip_pick(self) -> None:
        """MIP window: sparse clicks become anchors, then dense interp."""
        import napari
        from qtpy.QtWidgets import QMessageBox, QPushButton

        sag = self.vol_zyx.max(axis=2)
        cor = self.vol_zyx.max(axis=1)
        gap = 20
        canvas = np.zeros((sag.shape[0], sag.shape[1] + gap + cor.shape[1]), dtype=np.float32)
        canvas[:, : sag.shape[1]] = sag
        canvas[:, sag.shape[1] + gap :] = cor

        v = napari.Viewer(title="MIP anchors: LEFT=sagittal  RIGHT=coronal — few clicks, then Accept")
        v.add_image(canvas, name="mips", colormap="gray")
        seed = []
        src = self.anchors if len(self.anchors) else subsample_anchors(self.centerline, every=50)
        for z, y, x in src:
            seed.append([z, y])
            seed.append([z, sag.shape[1] + gap + x])
        pl = v.add_points(
            np.asarray(seed, dtype=np.float64).reshape(-1, 2) if seed else np.zeros((0, 2)),
            name="mip_anchors",
            size=8,
            face_color="magenta",
            ndim=2,
        )
        v.dims.ndisplay = 2

        def _accept():
            data = np.asarray(pl.data, dtype=np.float64).reshape(-1, 2)
            sag_zy, cor_zx = [], []
            split = sag.shape[1] + gap / 2.0
            for r, c in data:
                if c < split:
                    sag_zy.append((r, c))
                else:
                    cor_zx.append((r, c - (sag.shape[1] + gap)))
            anc = merge_mip_points(
                np.asarray(sag_zy, dtype=np.float64).reshape(-1, 2),
                np.asarray(cor_zx, dtype=np.float64).reshape(-1, 2),
                shape_zyx=self.vol_zyx.shape,
            )
            if len(anc) < 2:
                QMessageBox.warning(None, "MIP pick", "Need ≥2 Z anchors (ideally on both sides).")
                return
            self._set_anchors_layer(anc)
            # Dense over full occupied Z of volume mask-ish: use anchor span
            dense = densify_from_anchors(anc, step_z=1.0, shape_zyx=self.vol_zyx.shape)
            self._set_centerline_layer(dense)
            write_centerline_csv(self.out_dir / "centerline_zyx.csv", dense, self.spacing_zyx)
            self._log(f"MIP anchors {len(anc)} → dense {len(dense)}")
            v.close()

        btn = QPushButton("Accept → interp dense centerline")
        btn.clicked.connect(_accept)
        v.window.add_dock_widget(btn, area="right")
        self._log("MIP: place a few anchors on left/right, Accept interpolates all slices.")

    def on_straighten(self) -> None:
        # Ensure latest interp
        if len(self.anchor_layer.data) >= 2:
            self.on_interp_anchors(silent=True)
        if len(self.centerline) < 3:
            self._log("No centerline — auto / MIP / anchors+interp first.")
            return
        csv_path = self.out_dir / "centerline_zyx.csv"
        write_centerline_csv(csv_path, self.centerline, self.spacing_zyx)
        straight_dir = self.out_dir / "straightened"
        straight_dir.mkdir(exist_ok=True)
        self.straight_nii = straight_dir / "volume_straight.nii.gz"
        # Same CLI path as before (not in napari GUI thread): keeps UI responsive.
        cmd = [
            sys.executable,
            "-m",
            "pipeline_modules.registration.spinalj_straighten_register",
            "--volume_nii",
            str(self.volume_nii),
            "--centerline_csv",
            str(csv_path),
            "--atlas_dir",
            str(self.atlas_dir),
            "--out_dir",
            str(self.out_dir),
            "--skip_register",
            "--no_untwist",
            "--out_radius_yx_vox",
            "160",
            "--straighten_interp_order",
            "1",
            # Manual / anchor centerlines must NOT be spline-smoothed again:
            # smooth_um>0 cuts corners on bends → residual S-curve in sagittal QC.
            "--straighten_smooth_um",
            "0",
            "--straighten_tangent_window",
            "11",
            "--straighten_frame_window",
            "11",
        ]
        log_path = straight_dir / "straighten_cli.log"
        self._log(
            "Straighten started in background (CLI, order=1, centerline_smooth=0 so your anchors stay). "
            f"Log: {log_path}"
        )
        self.btn_straighten.setEnabled(False)

        def _job():
            try:
                with log_path.open("w", encoding="utf-8") as lf:
                    proc = subprocess.run(
                        cmd,
                        cwd=str(Path(__file__).resolve().parents[2]),
                        stdout=lf,
                        stderr=subprocess.STDOUT,
                        check=False,
                    )
                if proc.returncode != 0:
                    self._log(f"Straighten FAILED (exit {proc.returncode}). See {log_path}")
                    return
                if not self.straight_nii.exists():
                    self._log(f"Straighten finished but missing {self.straight_nii}")
                    return
                vol, _aff, sp = load_volume_zyx(self.straight_nii)
                self.straight_zyx = vol
                self.straight_spacing = sp
                self.fixed_nii = self.straight_nii
                meta_path = straight_dir / "straighten_meta.json"
                if meta_path.exists():
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                    self._log(
                        f"Straightened OK shape={meta.get('out_shape_zyx')} "
                        f"smooth_um={meta.get('smooth_centerline_um')} → {self.straight_nii}"
                    )
                else:
                    self._log(f"Straightened OK → {self.straight_nii}")
                qc_sag = straight_dir / "qc" / "straight_sagittal_mip.png"
                if qc_sag.exists():
                    self._log(f"QC updated: {qc_sag} mtime={qc_sag.stat().st_mtime}")
                try:
                    from qtpy.QtCore import QTimer

                    def _add_layer():
                        if "straightened" in self.viewer.layers:
                            self.viewer.layers["straightened"].data = vol
                            self.viewer.layers["straightened"].contrast_limits = _contrast_limits_from_volume(vol)
                        else:
                            self.viewer.add_image(
                                vol,
                                name="straightened",
                                colormap="magenta",
                                blending="additive",
                                contrast_limits=_contrast_limits_from_volume(vol),
                            )
                        # Force-refresh QC preview in napari (avoids stale OS image-viewer cache).
                        qc_path = straight_dir / "qc" / "straight_sagittal_mip.png"
                        if qc_path.exists():
                            import cv2

                            qc_img = cv2.imread(str(qc_path), cv2.IMREAD_GRAYSCALE)
                            if qc_img is not None:
                                name = "qc_straight_sagittal"
                                if name in self.viewer.layers:
                                    self.viewer.layers[name].data = qc_img
                                else:
                                    self.viewer.add_image(qc_img, name=name, colormap="gray")
                                self._log(f"Loaded QC layer '{name}' from disk")
                        self.btn_straighten.setEnabled(True)
                        self.set_step(2)

                    QTimer.singleShot(0, _add_layer)
                except Exception:
                    self.btn_straighten.setEnabled(True)
            except Exception as e:
                self._log(f"Straighten FAILED: {e}")
                self.btn_straighten.setEnabled(True)

        threading.Thread(target=_job, daemon=True).start()

    def on_untwist_toggle(self, _state: int = 0) -> None:
        self.need_untwist = bool(self.chk_untwist.isChecked())
        self.btn_open_untwist.setEnabled(self.step == 2 and self.need_untwist)
        self._log("Untwist: " + ("requested" if self.need_untwist else "SKIP"))

    def on_open_untwist(self) -> None:
        if not self.need_untwist:
            self._log("Untwist not confirmed — skipping.")
            return
        if self.straight_nii is None or not self.straight_nii.exists():
            self._log("Straighten first.")
            return
        mode = self.cmb_untwist.currentText()
        untwist_out = self.out_dir / "manual_untwist"
        untwist_out.mkdir(exist_ok=True)
        if mode == "atlas":
            cmd = [
                sys.executable,
                "-m",
                "pipeline_modules.registration.spinalj_straighten_register",
                "--volume_nii",
                str(self.volume_nii),
                "--centerline_csv",
                str(self.out_dir / "centerline_zyx.csv"),
                "--atlas_dir",
                str(self.atlas_dir),
                "--out_dir",
                str(self.out_dir / "atlas_untwist"),
                "--untwist",
                "--untwist_method",
                "atlas",
                "--skip_register",
            ]
            subprocess.Popen(cmd, cwd=str(Path(__file__).resolve().parents[2]))
            self.fixed_nii = self.out_dir / "atlas_untwist" / "straightened" / "volume_straight.nii.gz"
            self._log("Atlas untwist started in background.")
        else:
            cmd = [
                sys.executable,
                "-m",
                "pipeline_modules.registration.spinalj_manual_untwist_ui",
                "--volume_nii",
                str(self.straight_nii),
                "--out_dir",
                str(untwist_out),
                "--atlas_dir",
                str(self.atlas_dir),
                "--transform",
                self.transform,
            ]
            subprocess.Popen(cmd, cwd=str(Path(__file__).resolve().parents[2]))
            self.fixed_nii = untwist_out / "volume_straight_manual_untwist_xyalign.nii.gz"
            self._log("Manual untwist UI launched.")

    def _ensure_atlas(self) -> bool:
        if self.atlas_zyx is not None:
            return True
        try:
            self._log("Loading atlas Template for flip compare...")
            self.atlas_zyx, _sp = load_atlas_template_zyx(self.atlas_dir, self.out_dir / "atlas_nifti")
            self._log(f"Atlas ZYX={self.atlas_zyx.shape}")
            return True
        except Exception as e:
            self._log(f"Atlas load FAILED: {e}")
            return False

    def _sample_for_compare(self) -> np.ndarray | None:
        # Prefer in-memory straighten; else load fixed.
        if self.straight_zyx is not None:
            return self.straight_zyx
        path = self._resolve_fixed()
        if path is None:
            return None
        vol, _aff, sp = load_volume_zyx(path)
        self.straight_zyx = vol
        self.straight_spacing = sp
        return vol

    def on_flip_changed(self, *_args) -> None:
        self.flip_x = bool(self.chk_flip_x.isChecked())
        self.flip_y = bool(self.chk_flip_y.isChecked())
        self.flip_z = bool(self.chk_flip_z.isChecked())
        self.allow_reflection = bool(self.chk_reflect.isChecked())
        self._log(
            f"Flips: X={self.flip_x} Y={self.flip_y} Z={self.flip_z} "
            f"allow_reflection={self.allow_reflection}"
        )
        self.refresh_flip_compare()

    def refresh_flip_compare(self) -> None:
        from qtpy.QtGui import QImage, QPixmap
        from qtpy.QtCore import Qt

        if not self._ensure_atlas():
            return
        sample = self._sample_for_compare()
        if sample is None or self.atlas_zyx is None:
            self._log("No sample volume for compare.")
            return
        sample_f = apply_flips_zyx(
            sample, flip_x=self.flip_x, flip_y=self.flip_y, flip_z=self.flip_z
        )
        zs = sample_f[sample_f.shape[0] // 2]
        za = self.atlas_zyx[self.atlas_zyx.shape[0] // 2]
        # Show in napari too
        for name, arr, cmap in (
            ("sample_flip_preview", sample_f, "magenta"),
            ("atlas_ref", self.atlas_zyx, "green"),
        ):
            if name in self.viewer.layers:
                self.viewer.layers[name].data = arr
            else:
                self.viewer.add_image(arr, name=name, colormap=cmap, blending="additive", opacity=0.7)

        def _set_label(lbl, img2d: np.ndarray, title: str) -> None:
            u8 = _to_u8_preview(img2d)
            # Resize-ish for label: keep aspect, max 200
            h, w = u8.shape
            scale = min(200 / max(h, 1), 200 / max(w, 1), 1.0)
            nh, nw = max(1, int(h * scale)), max(1, int(w * scale))
            try:
                import cv2

                u8s = cv2.resize(u8, (nw, nh), interpolation=cv2.INTER_AREA)
            except Exception:
                u8s = u8[:: max(1, h // nh), :: max(1, w // nw)]
            qimg = QImage(u8s.data, u8s.shape[1], u8s.shape[0], u8s.strides[0], QImage.Format_Grayscale8)
            pix = QPixmap.fromImage(qimg.copy())
            lbl.setPixmap(pix)
            lbl.setToolTip(title)

        _set_label(self.lbl_sample, zs, f"sample midZ (flips XYZ={self.flip_x},{self.flip_y},{self.flip_z})")
        _set_label(self.lbl_atlas, za, "atlas Template midZ")

    def _write_flipped_fixed(self) -> Path | None:
        sample = self._sample_for_compare()
        if sample is None:
            return None
        sp = self.straight_spacing or (20.0, 10.0, 10.0)
        if not (self.flip_x or self.flip_y or self.flip_z):
            return self._resolve_fixed()
        flipped = apply_flips_zyx(
            sample, flip_x=self.flip_x, flip_y=self.flip_y, flip_z=self.flip_z
        )
        out = self.out_dir / "straightened" / "volume_straight_flipped.nii.gz"
        save_volume_zyx(out, flipped, sp)
        meta = {"flip_x": self.flip_x, "flip_y": self.flip_y, "flip_z": self.flip_z}
        (self.out_dir / "straightened" / "flip_meta.json").write_text(
            json.dumps(meta, indent=2), encoding="utf-8"
        )
        self.fixed_nii = out
        self._log(f"Wrote flipped fixed → {out}")
        return out

    def _resolve_fixed(self) -> Path | None:
        candidates = []
        if self.fixed_nii is not None:
            candidates.append(self.fixed_nii)
        candidates.extend(
            [
                self.out_dir / "straightened" / "volume_straight_flipped.nii.gz",
                self.out_dir / "manual_untwist" / "volume_straight_manual_untwist_xyalign.nii.gz",
                self.out_dir / "manual_untwist" / "volume_straight_manual_untwist.nii.gz",
                self.out_dir / "atlas_untwist" / "straightened" / "volume_straight.nii.gz",
                self.out_dir / "straightened" / "volume_straight.nii.gz",
            ]
        )
        for c in candidates:
            if c is not None and Path(c).exists():
                return Path(c)
        return None

    def on_landmarks(self) -> None:
        fixed = self._write_flipped_fixed() or self._resolve_fixed()
        if fixed is None:
            self._log("No straightened volume yet.")
            return
        lm_out = self.out_dir / "landmarks"
        cmd = [
            sys.executable,
            "-m",
            "pipeline_modules.registration.spinalj_landmark_ui",
            "--volume_nii",
            str(fixed),
            "--atlas_dir",
            str(self.atlas_dir),
            "--out_dir",
            str(lm_out),
        ]
        subprocess.Popen(cmd, cwd=str(Path(__file__).resolve().parents[2]))
        self._log(f"Landmark UI launched → {lm_out}")

    def on_register(self) -> None:
        fixed = self._write_flipped_fixed() or self._resolve_fixed()
        if fixed is None:
            self._log("No fixed volume for registration.")
            return
        if self._reg_thread and self._reg_thread.is_alive():
            self._log("Registration already running.")
            return
        self.allow_reflection = bool(self.chk_reflect.isChecked())
        # Physical flips already applied to volume; identity direction (no ANTs Y flip).
        direction_y = 1.0
        ants_out = self.out_dir / "ants_out"
        self._log(
            f"Registering fixed={fixed.name} flips=XYZ({self.flip_x},{self.flip_y},{self.flip_z}) "
            f"reflect={self.allow_reflection} ..."
        )

        def _job():
            try:
                from pipeline_modules.registration.spinalj_manual_untwist_ui import (
                    run_formal_register_job,
                )

                summary = run_formal_register_job(
                    fixed_nii=fixed,
                    atlas_dir=self.atlas_dir,
                    out_dir=self.out_dir,
                    transform=self.transform,
                    allow_reflection=self.allow_reflection,
                    direction_y=direction_y,
                    ants_out_name="ants_out",
                )
                summary["flip_x"] = self.flip_x
                summary["flip_y"] = self.flip_y
                summary["flip_z"] = self.flip_z
                (self.out_dir / "wizard_register_summary.json").write_text(
                    json.dumps(summary, indent=2), encoding="utf-8"
                )
                self._log(f"Registration DONE → {ants_out}")
            except Exception as e:
                self._log(f"Registration FAILED: {e}")

        self._reg_thread = threading.Thread(target=_job, daemon=True)
        self._reg_thread.start()

    def on_preview_qc(self) -> None:
        qc = self.out_dir / "ants_out" / "qc_slices"
        if not qc.exists():
            self._log("No QC yet — run register first.")
            return
        try:
            from skimage.io import imread

            overlay = qc / "overlay_zmid.png"
            if overlay.exists():
                img = imread(str(overlay))
                name = "qc_overlay_zmid"
                if name in self.viewer.layers:
                    self.viewer.layers.remove(name)
                self.viewer.add_image(img, name=name, rgb=True)
                self._log(f"Loaded {overlay}")
            else:
                self._log(f"Missing {overlay}")
        except Exception as e:
            self._log(f"Preview failed: {e}")

    def go_next(self) -> None:
        if self.step == 1:
            if self.straight_nii is None or not Path(self.straight_nii).exists():
                self._log("Step 1 incomplete — straighten first.")
                return
            self.set_step(2)
            return
        if self.step == 2:
            self.need_untwist = bool(self.chk_untwist.isChecked())
            if not self.need_untwist:
                self.fixed_nii = self.straight_nii
                self._log("Step 2: untwist skipped.")
            else:
                resolved = self._resolve_fixed()
                if resolved is None:
                    self._log("Untwist confirmed but output missing — open untwist UI first.")
                    return
                self.fixed_nii = resolved
            self.set_step(3)
            return

    def show(self) -> None:
        import napari

        napari.run()


def main() -> None:
    _configure_logging()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--volume_nii", required=True, help="Downsampled sample volume.nii.gz")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--atlas_dir", default=DEFAULT_ATLAS)
    p.add_argument("--transform", default="SyNRA")
    p.add_argument(
        "--centerline_csv",
        default="",
        help="Precomputed centerline_zyx.csv — skip auto/MIP; load cyan+magenta on start",
    )
    args = p.parse_args()

    wiz = SpinalJWizard(
        volume_nii=Path(args.volume_nii),
        out_dir=Path(args.out_dir),
        atlas_dir=Path(args.atlas_dir),
        transform=args.transform,
        centerline_csv=Path(args.centerline_csv) if args.centerline_csv else None,
    )
    wiz.show()


if __name__ == "__main__":
    main()
