#!/usr/bin/env python3
"""Landmark picking UI for SpinalJ atlas ↔ straightened cord registration.

Left/right transverse slices for placing named anchors; bottom row shows
low-resolution sagittal MIPs of sample and atlas with landmark Z markers.

Default landmarks are only rostral start (开头) and caudal end (结尾).
Users can add more landmarks with custom names.

Example:
  python -m pipeline_modules.registration.spinalj_landmark_ui --volume_nii ".../volume_straight_manual_untwist_xyalign.nii.gz" --atlas_dir "S:/Yifu_data/reference/SC_P56_Atlas_10x10x20_v5_2020" --out_dir ".../landmarks"
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# Display name, id, matplotlib color — only start / end by default.
DEFAULT_PRESETS: list[tuple[str, str, str]] = [
    ("开头 (rostral start)", "rostral_start", "#e41a1c"),
    ("结尾 (caudal end)", "caudal_end", "#984ea3"),
]
LOCKED_IDS = frozenset({"rostral_start", "caudal_end"})
CUSTOM_COLORS = (
    "#377eb8",
    "#4daf4a",
    "#ff7f00",
    "#a65628",
    "#f781bf",
    "#66c2a5",
    "#fc8d62",
    "#8da0cb",
)


def _configure_cjk_font() -> None:
    """Prefer a CJK-capable font so 开头/结尾 render in the UI."""
    import matplotlib

    for name in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Arial Unicode MS"):
        try:
            from matplotlib import font_manager

            path = font_manager.findfont(name, fallback_to_default=False)
            if path and "DejaVu" not in path:
                matplotlib.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
                matplotlib.rcParams["axes.unicode_minus"] = False
                return
        except Exception:
            continue


def _slugify(name: str) -> str:
    s = re.sub(r"[^\w]+", "_", name.strip(), flags=re.UNICODE)
    s = s.strip("_").lower()
    return s or "landmark"


def _auto_contrast_u8(arr: np.ndarray, *, p_lo: float = 1.0, p_hi: float = 99.0) -> np.ndarray:
    a = np.asarray(arr, dtype=np.float32)
    m = a > 0
    if not np.any(m):
        return np.zeros(a.shape, dtype=np.uint8)
    lo, hi = np.percentile(a[m], (p_lo, p_hi))
    if hi <= lo:
        hi = lo + 1.0
    out = np.zeros_like(a, dtype=np.float32)
    out[m] = np.clip((a[m] - lo) / (hi - lo), 0, 1) * 255.0
    return out.astype(np.uint8)


def _slice_centroid_yx(img: np.ndarray) -> tuple[float, float] | None:
    img = np.asarray(img, dtype=np.float32)
    pos = img[img > 0]
    if pos.size < 32:
        return None
    thr = float(np.percentile(pos, 99) * 0.05)
    w = np.where(img > thr, img, 0.0)
    wsum = float(w.sum())
    if wsum <= 0:
        return None
    yy, xx = np.indices(img.shape, dtype=np.float64)
    return float((w * yy).sum() / wsum), float((w * xx).sum() / wsum)


def lowres_sagittal_mip(
    vol_zyx: np.ndarray,
    *,
    downsample: int = 4,
) -> tuple[np.ndarray, float, float]:
    """Sagittal MIP (max over X): image is (Y, Z), low-res.

    Returns (mip_u8_yz, z_scale, y_scale) mapping full-res index → mip index.
    """
    ds = max(1, int(downsample))
    vol = np.asarray(vol_zyx)
    sub = vol[::ds, ::ds, ::ds]
    mip_zy = sub.max(axis=2)  # (Z', Y')
    mip_yz = np.transpose(mip_zy, (1, 0))  # (Y', Z') — Z horizontal
    return _auto_contrast_u8(mip_yz), 1.0 / ds, 1.0 / ds


def estimate_z_affine(
    sample_z: list[float],
    atlas_z: list[float],
) -> dict | None:
    """Fit sample_z ≈ scale * atlas_z + shift (atlas → sample Z)."""
    if len(sample_z) < 2:
        return None
    a = np.asarray(atlas_z, dtype=np.float64)
    s = np.asarray(sample_z, dtype=np.float64)
    A = np.column_stack([a, np.ones_like(a)])
    scale, shift = np.linalg.lstsq(A, s, rcond=None)[0]
    pred = scale * a + shift
    resid = float(np.sqrt(np.mean((pred - s) ** 2)))
    return {
        "atlas_to_sample_z_scale": float(scale),
        "atlas_to_sample_z_shift": float(shift),
        "rmse_z_vox": resid,
        "n_points": int(len(sample_z)),
    }


def load_volume_zyx(path: Path) -> tuple[np.ndarray, np.ndarray]:
    import nibabel as nib

    nii = nib.load(str(path))
    xyz = np.asanyarray(nii.dataobj).astype(np.float32)
    return np.transpose(xyz, (2, 1, 0)), np.asarray(nii.affine)


def load_atlas_template_zyx(atlas_dir: Path, *, cache_dir: Path | None = None) -> np.ndarray:
    """Load SpinalJ Template as ZYX float32 (native 10×10×20)."""
    import nibabel as nib

    from pipeline_modules.registration.spinalj_atlas_register import convert_atlas_to_nifti

    cache = cache_dir or (Path(atlas_dir) / "_nifti_cache")
    paths = convert_atlas_to_nifti(Path(atlas_dir), cache)
    xyz = np.asanyarray(nib.load(str(paths["template"])).dataobj).astype(np.float32)
    return np.transpose(xyz, (2, 1, 0))


class LandmarkUI:
    def __init__(
        self,
        sample_zyx: np.ndarray,
        atlas_zyx: np.ndarray,
        *,
        out_dir: Path,
        sample_path: Path,
        atlas_dir: Path,
        presets: list[tuple[str, str, str]] | None = None,
        mip_downsample: int = 4,
        session_path: Path | None = None,
    ) -> None:
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Button, RadioButtons, Slider

        _configure_cjk_font()

        self.sample_zyx = sample_zyx
        self.atlas_zyx = atlas_zyx
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.sample_path = Path(sample_path)
        self.atlas_dir = Path(atlas_dir)
        # Ordered list: (label, id, color)
        self.landmarks: list[tuple[str, str, str]] = list(presets or DEFAULT_PRESETS)
        self.session_path = session_path or (self.out_dir / "landmarks.json")
        self.mip_downsample = int(mip_downsample)
        self._custom_color_i = 0

        self.points: dict[str, dict] = {
            pid: {"sample": None, "atlas": None, "label": lab, "color": col}
            for lab, pid, col in self.landmarks
        }
        self.active_id = self.landmarks[0][1]
        self.z_sample = int(self.sample_zyx.shape[0] // 2)
        self.z_atlas = int(self.atlas_zyx.shape[0] // 2)

        logger.info("Building low-res sagittal MIPs (ds=%d)...", self.mip_downsample)
        self.mip_s, self.zs_s, self.ys_s = lowres_sagittal_mip(
            self.sample_zyx, downsample=self.mip_downsample
        )
        self.mip_a, self.zs_a, self.ys_a = lowres_sagittal_mip(
            self.atlas_zyx, downsample=self.mip_downsample
        )

        if self.session_path.exists():
            self._load_session()

        self.fig = plt.figure(figsize=(14, 9))
        gs = self.fig.add_gridspec(
            3,
            2,
            height_ratios=[1.15, 0.85, 0.08],
            hspace=0.28,
            wspace=0.18,
            left=0.06,
            right=0.78,
            top=0.92,
            bottom=0.14,
        )
        self.ax_s = self.fig.add_subplot(gs[0, 0])
        self.ax_a = self.fig.add_subplot(gs[0, 1])
        self.ax_mip_s = self.fig.add_subplot(gs[1, 0])
        self.ax_mip_a = self.fig.add_subplot(gs[1, 1])

        self.ax_s.set_title("Sample transverse (click to place)")
        self.ax_a.set_title("Atlas transverse (click to place)")
        self.ax_mip_s.set_title("Sample sagittal MIP (side) — lines = landmarks")
        self.ax_mip_a.set_title("Atlas sagittal MIP (side) — lines = landmarks")
        for ax in (self.ax_s, self.ax_a):
            ax.axis("off")

        self.im_s = self.ax_s.imshow(np.zeros((10, 10), dtype=np.uint8), cmap="gray", vmin=0, vmax=255)
        self.im_a = self.ax_a.imshow(np.zeros((10, 10), dtype=np.uint8), cmap="gray", vmin=0, vmax=255)
        self.im_mip_s = self.ax_mip_s.imshow(
            self.mip_s, cmap="gray", vmin=0, vmax=255, aspect="auto", origin="upper"
        )
        self.im_mip_a = self.ax_mip_a.imshow(
            self.mip_a, cmap="gray", vmin=0, vmax=255, aspect="auto", origin="upper"
        )
        self.ax_mip_s.set_xlabel("Z (along cord)")
        self.ax_mip_a.set_xlabel("Z (along cord)")
        self.ax_mip_s.set_ylabel("Y")
        self.ax_mip_a.set_ylabel("Y")

        (self.pt_s,) = self.ax_s.plot([], [], "o", color="cyan", markersize=10, markeredgecolor="white")
        (self.pt_a,) = self.ax_a.plot([], [], "o", color="lime", markersize=10, markeredgecolor="white")
        self.cur_s = self.ax_mip_s.axvline(0, color="yellow", lw=1.0, alpha=0.8)
        self.cur_a = self.ax_mip_a.axvline(0, color="yellow", lw=1.0, alpha=0.8)
        self.mip_lines_s: list = []
        self.mip_lines_a: list = []

        self.rax = self.fig.add_axes([0.80, 0.42, 0.18, 0.46])
        self.radio: RadioButtons | None = None
        self._rebuild_radio(active_id=self.active_id)

        ax_zs = self.fig.add_axes([0.10, 0.07, 0.28, 0.03])
        ax_za = self.fig.add_axes([0.48, 0.07, 0.28, 0.03])
        self.slider_s = Slider(
            ax_zs, "Sample Z", 0, self.sample_zyx.shape[0] - 1, valinit=self.z_sample, valstep=1
        )
        self.slider_a = Slider(
            ax_za, "Atlas Z", 0, self.atlas_zyx.shape[0] - 1, valinit=self.z_atlas, valstep=1
        )
        self.slider_s.on_changed(self._on_z_sample)
        self.slider_a.on_changed(self._on_z_atlas)

        self.btn_add = Button(self.fig.add_axes([0.80, 0.34, 0.16, 0.045]), "Add landmark")
        self.btn_del = Button(self.fig.add_axes([0.80, 0.28, 0.16, 0.045]), "Delete custom")
        self.btn_save = Button(self.fig.add_axes([0.80, 0.22, 0.16, 0.045]), "Save")
        self.btn_clear = Button(self.fig.add_axes([0.80, 0.16, 0.16, 0.045]), "Clear active")
        self.btn_place_z = Button(self.fig.add_axes([0.80, 0.10, 0.16, 0.045]), "Place at Z")
        self.btn_add.on_clicked(lambda _e: self._add_landmark_dialog())
        self.btn_del.on_clicked(lambda _e: self._delete_active_custom())
        self.btn_save.on_clicked(lambda _e: self._save_session())
        self.btn_clear.on_clicked(lambda _e: self._clear_active())
        self.btn_place_z.on_clicked(lambda _e: self._place_at_current_z())

        self.status = self.fig.text(0.02, 0.005, "", fontsize=9)
        self.fig.canvas.mpl_connect("button_press_event", self._on_click)
        self.fig.canvas.mpl_connect("scroll_event", self._on_scroll)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        self.fig.suptitle(
            "Landmarks: 开头/结尾 fixed; Add landmark for custom names.  "
            "Select → scrub Z → click slice (or Place at Z).  MIP click jumps Z.  s=save",
            fontsize=10,
        )
        self._refresh()

    def _labels(self) -> list[str]:
        return [lab for lab, _pid, _col in self.landmarks]

    def _ids(self) -> list[str]:
        return [pid for _lab, pid, _col in self.landmarks]

    def _rebuild_radio(self, *, active_id: str | None = None) -> None:
        from matplotlib.widgets import RadioButtons

        self.rax.cla()
        labels = self._labels()
        active = 0
        if active_id is not None and active_id in self._ids():
            active = self._ids().index(active_id)
        self.radio = RadioButtons(self.rax, labels, active=active)
        self.radio.on_clicked(self._on_preset)
        self.active_id = self._ids()[active]
        self.rax.set_title("Landmarks", fontsize=9)

    def _next_color(self) -> str:
        col = CUSTOM_COLORS[self._custom_color_i % len(CUSTOM_COLORS)]
        self._custom_color_i += 1
        return col

    def _unique_id(self, base: str) -> str:
        pid = base
        n = 2
        existing = set(self._ids())
        while pid in existing:
            pid = f"{base}_{n}"
            n += 1
        return pid

    def _add_landmark(self, label: str, *, pid: str | None = None, color: str | None = None) -> str:
        label = str(label).strip()
        if not label:
            raise ValueError("empty landmark name")
        new_id = self._unique_id(pid or _slugify(label))
        col = color or self._next_color()
        self.landmarks.append((label, new_id, col))
        self.points[new_id] = {"sample": None, "atlas": None, "label": label, "color": col}
        return new_id

    def _add_landmark_dialog(self) -> None:
        import tkinter as tk
        from tkinter import simpledialog

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        name = simpledialog.askstring(
            "Add landmark",
            "Custom landmark name:",
            parent=root,
        )
        root.destroy()
        if not name or not str(name).strip():
            return
        new_id = self._add_landmark(str(name).strip())
        self._rebuild_radio(active_id=new_id)
        self._refresh()
        logger.info("Added landmark %s (%s)", name, new_id)

    def _delete_active_custom(self) -> None:
        if self.active_id in LOCKED_IDS:
            self.status.set_text("Cannot delete 开头/结尾 — only custom landmarks")
            self.fig.canvas.draw_idle()
            return
        if len(self.landmarks) <= 2:
            return
        self.landmarks = [t for t in self.landmarks if t[1] != self.active_id]
        self.points.pop(self.active_id, None)
        self._rebuild_radio(active_id=self.landmarks[0][1])
        self._refresh()

    def _on_preset(self, label: str) -> None:
        for lab, pid, _col in self.landmarks:
            if lab == label:
                self.active_id = pid
                break
        rec = self.points[self.active_id]
        if rec["sample"] is not None:
            self.slider_s.set_val(int(rec["sample"]["z"]))
        if rec["atlas"] is not None:
            self.slider_a.set_val(int(rec["atlas"]["z"]))
        self._refresh()

    def _on_z_sample(self, val: float) -> None:
        self.z_sample = int(val)
        self._refresh(update_sliders=False)

    def _on_z_atlas(self, val: float) -> None:
        self.z_atlas = int(val)
        self._refresh(update_sliders=False)

    def _on_scroll(self, event) -> None:
        if event.inaxes == self.ax_s or event.inaxes == self.ax_mip_s:
            d = -1 if event.button == "up" else 1
            self.slider_s.set_val(int(np.clip(self.z_sample + d * 5, 0, self.sample_zyx.shape[0] - 1)))
        elif event.inaxes == self.ax_a or event.inaxes == self.ax_mip_a:
            d = -1 if event.button == "up" else 1
            self.slider_a.set_val(int(np.clip(self.z_atlas + d * 5, 0, self.atlas_zyx.shape[0] - 1)))

    def _on_key(self, event) -> None:
        if event.key == "n":
            ids = self._ids()
            i = ids.index(self.active_id)
            nxt = ids[(i + 1) % len(ids)]
            self._rebuild_radio(active_id=nxt)
            self._on_preset(next(lab for lab, pid, _c in self.landmarks if pid == nxt))
        elif event.key == "a":
            self._add_landmark_dialog()
        elif event.key == "]":
            self.slider_s.set_val(min(self.z_sample + 1, self.sample_zyx.shape[0] - 1))
        elif event.key == "[":
            self.slider_s.set_val(max(self.z_sample - 1, 0))
        elif event.key == "s":
            self._save_session()

    def _on_click(self, event) -> None:
        if event.inaxes is None or event.button != 1:
            return
        if event.xdata is None or event.ydata is None:
            return

        if event.inaxes == self.ax_mip_s:
            z = int(np.clip(round(event.xdata / self.zs_s), 0, self.sample_zyx.shape[0] - 1))
            self.slider_s.set_val(z)
            return
        if event.inaxes == self.ax_mip_a:
            z = int(np.clip(round(event.xdata / self.zs_a), 0, self.atlas_zyx.shape[0] - 1))
            self.slider_a.set_val(z)
            return

        if event.inaxes == self.ax_s:
            self.points[self.active_id]["sample"] = {
                "z": int(self.z_sample),
                "y": float(event.ydata),
                "x": float(event.xdata),
            }
            self._refresh()
            return
        if event.inaxes == self.ax_a:
            self.points[self.active_id]["atlas"] = {
                "z": int(self.z_atlas),
                "y": float(event.ydata),
                "x": float(event.xdata),
            }
            self._refresh()
            return

    def _place_at_current_z(self) -> None:
        """Place both sides at current slider Z using slice centroids for XY."""
        cs = _slice_centroid_yx(self.sample_zyx[self.z_sample])
        ca = _slice_centroid_yx(self.atlas_zyx[self.z_atlas])
        if cs is None:
            cs = (
                (self.sample_zyx.shape[1] - 1) / 2.0,
                (self.sample_zyx.shape[2] - 1) / 2.0,
            )
        if ca is None:
            ca = (
                (self.atlas_zyx.shape[1] - 1) / 2.0,
                (self.atlas_zyx.shape[2] - 1) / 2.0,
            )
        self.points[self.active_id]["sample"] = {
            "z": int(self.z_sample),
            "y": float(cs[0]),
            "x": float(cs[1]),
        }
        self.points[self.active_id]["atlas"] = {
            "z": int(self.z_atlas),
            "y": float(ca[0]),
            "x": float(ca[1]),
        }
        self._refresh()

    def _clear_active(self) -> None:
        self.points[self.active_id]["sample"] = None
        self.points[self.active_id]["atlas"] = None
        self._refresh()

    def _paired_zs(self) -> tuple[list[float], list[float]]:
        sz, az = [], []
        for _lab, pid, _col in self.landmarks:
            rec = self.points[pid]
            if rec["sample"] is not None and rec["atlas"] is not None:
                sz.append(float(rec["sample"]["z"]))
                az.append(float(rec["atlas"]["z"]))
        return sz, az

    def _short_tag(self, label: str, pid: str) -> str:
        if pid == "rostral_start":
            return "start"
        if pid == "caudal_end":
            return "end"
        return (label or pid)[:6]

    def _draw_mip_landmarks(self) -> None:
        for ln in self.mip_lines_s:
            ln.remove()
        for ln in self.mip_lines_a:
            ln.remove()
        self.mip_lines_s.clear()
        self.mip_lines_a.clear()
        for lab, pid, col in self.landmarks:
            rec = self.points[pid]
            tag = self._short_tag(lab, pid)
            if rec["sample"] is not None:
                xz = float(rec["sample"]["z"]) * self.zs_s
                self.mip_lines_s.append(self.ax_mip_s.axvline(xz, color=col, lw=2.0, alpha=0.95))
                self.mip_lines_s.append(
                    self.ax_mip_s.text(
                        xz, 3, tag, color=col, fontsize=8, rotation=90, va="bottom", ha="right"
                    )
                )
            if rec["atlas"] is not None:
                xz = float(rec["atlas"]["z"]) * self.zs_a
                self.mip_lines_a.append(self.ax_mip_a.axvline(xz, color=col, lw=2.0, alpha=0.95))
                self.mip_lines_a.append(
                    self.ax_mip_a.text(
                        xz, 3, tag, color=col, fontsize=8, rotation=90, va="bottom", ha="right"
                    )
                )

    def _refresh(self, update_sliders: bool = True) -> None:
        if update_sliders:
            if abs(self.slider_s.val - self.z_sample) > 0.5:
                self.slider_s.set_val(self.z_sample)
                return
            if abs(self.slider_a.val - self.z_atlas) > 0.5:
                self.slider_a.set_val(self.z_atlas)
                return

        u8s = _auto_contrast_u8(self.sample_zyx[self.z_sample])
        u8a = _auto_contrast_u8(self.atlas_zyx[self.z_atlas])
        self.im_s.set_data(u8s)
        self.im_s.set_extent((-0.5, u8s.shape[1] - 0.5, u8s.shape[0] - 0.5, -0.5))
        self.im_a.set_data(u8a)
        self.im_a.set_extent((-0.5, u8a.shape[1] - 0.5, u8a.shape[0] - 0.5, -0.5))

        rec = self.points[self.active_id]
        if rec["sample"] is not None and int(rec["sample"]["z"]) == self.z_sample:
            self.pt_s.set_data([rec["sample"]["x"]], [rec["sample"]["y"]])
        else:
            self.pt_s.set_data([], [])
        if rec["atlas"] is not None and int(rec["atlas"]["z"]) == self.z_atlas:
            self.pt_a.set_data([rec["atlas"]["x"]], [rec["atlas"]["y"]])
        else:
            self.pt_a.set_data([], [])

        self.cur_s.set_xdata([self.z_sample * self.zs_s, self.z_sample * self.zs_s])
        self.cur_a.set_xdata([self.z_atlas * self.zs_a, self.z_atlas * self.zs_a])
        self._draw_mip_landmarks()

        sz, az = self._paired_zs()
        fit = estimate_z_affine(sz, az)
        done = []
        for lab, pid, _c in self.landmarks:
            r = self.points[pid]
            mark = ("S" if r["sample"] else "-") + ("A" if r["atlas"] else "-")
            done.append(f"{lab}:{mark}")
        fit_txt = ""
        if fit is not None:
            fit_txt = (
                f" | Z fit atlas→sample: scale={fit['atlas_to_sample_z_scale']:.3f} "
                f"shift={fit['atlas_to_sample_z_shift']:.1f} rmse={fit['rmse_z_vox']:.1f}vox"
            )
        active_lab = self.points[self.active_id]["label"]
        self.status.set_text(
            f"active={active_lab}  sampleZ={self.z_sample} atlasZ={self.z_atlas}  "
            f"[{'; '.join(done)}]{fit_txt}"
        )
        self.ax_s.set_title(f"Sample Z={self.z_sample}  active={active_lab}")
        self.ax_a.set_title(f"Atlas Z={self.z_atlas}  active={active_lab}")
        self.fig.canvas.draw_idle()

    def _load_session(self) -> None:
        data = json.loads(self.session_path.read_text(encoding="utf-8"))
        # Rebuild landmark list from file (keep 开头/结尾, drop old cervical/lumbar presets
        # unless they were custom-saved under other ids).
        loaded: list[tuple[str, str, str]] = []
        points: dict[str, dict] = {}
        for item in data.get("landmarks", []):
            pid = str(item.get("id", "")).strip()
            lab = str(item.get("label", pid)).strip()
            col = str(item.get("color", self._next_color()))
            if not pid:
                continue
            # Skip legacy enlargement presets if present in old sessions.
            if pid in ("cervical_enlargement", "lumbar_enlargement"):
                continue
            loaded.append((lab, pid, col))
            points[pid] = {
                "sample": item.get("sample"),
                "atlas": item.get("atlas"),
                "label": lab,
                "color": col,
            }
        # Ensure locked start/end always exist.
        have = {pid for _l, pid, _c in loaded}
        for lab, pid, col in DEFAULT_PRESETS:
            if pid not in have:
                loaded.insert(0 if pid == "rostral_start" else len(loaded), (lab, pid, col))
                points[pid] = {"sample": None, "atlas": None, "label": lab, "color": col}
        # Stable order: start, customs..., end
        start = [t for t in loaded if t[1] == "rostral_start"]
        end = [t for t in loaded if t[1] == "caudal_end"]
        mid = [t for t in loaded if t[1] not in LOCKED_IDS]
        self.landmarks = start + mid + end
        self.points = {pid: points[pid] for _l, pid, _c in self.landmarks}
        self.active_id = self.landmarks[0][1]
        logger.info("Loaded landmarks from %s (%d points)", self.session_path, len(self.landmarks))

    def _save_session(self) -> None:
        landmarks = []
        for lab, pid, col in self.landmarks:
            landmarks.append(
                {
                    "id": pid,
                    "label": lab,
                    "color": col,
                    "sample": self.points[pid]["sample"],
                    "atlas": self.points[pid]["atlas"],
                }
            )
        sz, az = self._paired_zs()
        fit = estimate_z_affine(sz, az)
        payload = {
            "sample_nii": str(self.sample_path),
            "atlas_dir": str(self.atlas_dir),
            "sample_shape_zyx": list(self.sample_zyx.shape),
            "atlas_shape_zyx": list(self.atlas_zyx.shape),
            "landmarks": landmarks,
            "z_affine_atlas_to_sample": fit,
        }
        self.session_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        qc = self.out_dir / "qc"
        qc.mkdir(exist_ok=True)
        self.fig.savefig(qc / "landmarks_panel.png", dpi=120)
        self.status.set_text(f"Saved {self.session_path.name}" + (f" | {fit}" if fit else ""))
        self.fig.canvas.draw_idle()
        logger.info("Wrote %s", self.session_path)

    def show(self) -> None:
        import matplotlib.pyplot as plt

        plt.show()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--volume_nii", required=True, help="Straightened sample NIfTI (XYZ)")
    p.add_argument(
        "--atlas_dir",
        default=r"S:\Yifu_data\reference\SC_P56_Atlas_10x10x20_v5_2020",
        help="SpinalJ atlas folder",
    )
    p.add_argument("--out_dir", required=True)
    p.add_argument("--mip_downsample", type=int, default=4, help="Sagittal MIP downsample (default 4)")
    p.add_argument("--atlas_cache_dir", default="", help="Optional NIfTI cache for atlas")
    args = p.parse_args()

    import matplotlib

    try:
        matplotlib.use("TkAgg")
    except Exception:
        pass
    _configure_cjk_font()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sample_zyx, _aff = load_volume_zyx(Path(args.volume_nii))
    cache = Path(args.atlas_cache_dir) if args.atlas_cache_dir else (out_dir / "atlas_nifti")
    sibling = Path(args.volume_nii).resolve().parent / "atlas_nifti"
    if (sibling / "Template.nii.gz").exists() and not args.atlas_cache_dir:
        cache = sibling
    atlas_zyx = load_atlas_template_zyx(Path(args.atlas_dir), cache_dir=cache)
    logger.info("Sample ZYX=%s  Atlas ZYX=%s", sample_zyx.shape, atlas_zyx.shape)

    ui = LandmarkUI(
        sample_zyx,
        atlas_zyx,
        out_dir=out_dir,
        sample_path=Path(args.volume_nii),
        atlas_dir=Path(args.atlas_dir),
        mip_downsample=args.mip_downsample,
    )
    ui.show()


if __name__ == "__main__":
    main()
