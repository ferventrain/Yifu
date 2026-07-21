"""Shared high-quality atlas-slice heatmap rendering.

Fill and region contours use the same smoothed vector paths (anti-aliased,
supersampled), so color stays inside outlines. Brain outer outline is off by
default (often discontinuous on Allen labels).
"""

from __future__ import annotations

import io
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors as mcolors
from matplotlib.collections import LineCollection
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Polygon
from PIL import Image
from scipy import ndimage as ndi

from pipeline_modules.visualization.atlas_slice import (
    SIGNAL_COUNT_DIFF_CMAP,
    _colormap_by_name,
    _mask_contour_lines,
)

DEFAULT_PIXEL_SCALE = 4
DEFAULT_SUPERSAMPLE = 2
DEFAULT_CONTOUR_SMOOTH = 1.2
DEFAULT_REGION_LINE_WIDTH = 0.7
DEFAULT_BRAIN_OUTLINE_WIDTH = 0.0  # off: outer outline is often broken


def _resolve_cmap(cmap_name: str):
    if cmap_name == "signal_count_diff":
        return SIGNAL_COUNT_DIFF_CMAP.copy()
    return _colormap_by_name(cmap_name).copy()


def _resolve_norm(vmin: float, vmax: float, vcenter: float | None):
    lower = float(vmin)
    upper = float(vmax)
    if vcenter is not None and lower < float(vcenter) < upper:
        return TwoSlopeNorm(vmin=lower, vcenter=float(vcenter), vmax=upper)
    return mcolors.Normalize(vmin=lower, vmax=upper, clip=True)


def render_labeled_value_slice_rgb(
    painted: np.ndarray,
    label_slice: np.ndarray,
    *,
    vmin: float,
    vmax: float,
    vcenter: float | None = None,
    cmap_name: str = "white_orange_red_black",
    pixel_scale: int = DEFAULT_PIXEL_SCALE,
    supersample: int = DEFAULT_SUPERSAMPLE,
    contour_smooth: float = DEFAULT_CONTOUR_SMOOTH,
    line_width: float = DEFAULT_REGION_LINE_WIDTH,
    brain_outline_width: float = DEFAULT_BRAIN_OUTLINE_WIDTH,
    show_region_contours: bool = True,
    background: str = "white",
    contour_color: str | None = None,
) -> np.ndarray:
    """Return RGB uint8 image of the slice panel (no colorbar)."""
    labels = np.asarray(label_slice)
    values = np.asarray(painted, dtype=np.float32)
    if labels.shape != values.shape:
        raise ValueError(f"painted shape {values.shape} != label shape {labels.shape}")

    scale = max(int(pixel_scale), 1)
    ss = max(int(supersample), 1)
    height, width = labels.shape

    if background == "white":
        bg_rgb = (1.0, 1.0, 1.0)
        default_contour = "#334155"
    else:
        bg_rgb = (0.0, 0.0, 0.0)
        default_contour = "white"
    stroke_color = contour_color or default_contour

    cmap = _resolve_cmap(cmap_name)
    norm = _resolve_norm(vmin, vmax, vcenter)
    smooth = float(contour_smooth)

    region_paths: list[tuple[tuple[float, float, float], list[np.ndarray]]] = []
    for region_id in np.unique(labels):
        rid = int(region_id)
        if rid == 0:
            continue
        mask = labels == rid
        region_vals = values[mask]
        finite = region_vals[np.isfinite(region_vals)]
        if finite.size == 0:
            continue
        for color_val in np.unique(np.round(finite.astype(np.float64), decimals=6)):
            sub = mask & np.isfinite(values) & (np.round(values.astype(np.float64), decimals=6) == float(color_val))
            if not np.any(sub):
                continue
            rgba = cmap(norm(float(color_val)))
            face = (float(rgba[0]), float(rgba[1]), float(rgba[2]))
            polys = _mask_contour_lines(sub, smoothing=smooth, min_points=6)
            if polys:
                region_paths.append((face, polys))

    brain_paths = _mask_contour_lines(labels > 0, smoothing=max(smooth, 0.8), min_points=8)

    render_scale = scale * ss
    dpi = 100.0
    fig_w = (width * render_scale) / dpi
    fig_h = (height * render_scale) / dpi
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
    try:
        fig.patch.set_facecolor(bg_rgb)
        ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
        ax.set_facecolor(bg_rgb)
        ax.set_xlim(-0.5, width - 0.5)
        ax.set_ylim(height - 0.5, -0.5)
        ax.set_aspect("equal")
        ax.set_axis_off()

        for face, polys in region_paths:
            for poly in polys:
                if poly is None or len(poly) < 3:
                    continue
                ax.add_patch(
                    Polygon(
                        poly,
                        closed=True,
                        facecolor=face,
                        edgecolor="none",
                        linewidth=0.0,
                        antialiased=True,
                        zorder=1,
                    )
                )

        if show_region_contours and line_width > 0:
            stroke_lines: list[np.ndarray] = []
            for _face, polys in region_paths:
                stroke_lines.extend(polys)
            if stroke_lines:
                ax.add_collection(
                    LineCollection(
                        stroke_lines,
                        colors=stroke_color,
                        linewidths=float(line_width) * (render_scale / 4.0),
                        antialiased=True,
                        zorder=2,
                    )
                )

        if brain_outline_width > 0 and brain_paths:
            ax.add_collection(
                LineCollection(
                    brain_paths,
                    colors=stroke_color,
                    linewidths=float(brain_outline_width) * (render_scale / 4.0),
                    antialiased=True,
                    zorder=3,
                )
            )

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi, facecolor=bg_rgb, pad_inches=0)
        buf.seek(0)
        image = Image.open(buf).convert("RGB")
    finally:
        plt.close(fig)

    target_size = (width * scale, height * scale)
    if image.size != target_size:
        image = image.resize(target_size, resample=Image.Resampling.LANCZOS)

    brain = np.repeat(np.repeat((labels > 0).astype(np.uint8), scale, axis=0), scale, axis=1).astype(bool)
    keep = ndi.binary_dilation(brain, iterations=1)
    rgb = np.asarray(image, dtype=np.uint8).copy()
    if rgb.shape[0] != keep.shape[0] or rgb.shape[1] != keep.shape[1]:
        image = image.resize((keep.shape[1], keep.shape[0]), resample=Image.Resampling.LANCZOS)
        rgb = np.asarray(image, dtype=np.uint8).copy()
    bg_u8 = np.clip(np.round(np.array(bg_rgb) * 255.0), 0, 255).astype(np.uint8)
    rgb[~keep] = bg_u8
    return rgb


def render_continuous_signal_slice_rgb(
    signal_slice: np.ndarray,
    label_slice: np.ndarray,
    *,
    vmin: float,
    vmax: float,
    cmap_name: str = "white_blue_red",
    pixel_scale: int = DEFAULT_PIXEL_SCALE,
    supersample: int = DEFAULT_SUPERSAMPLE,
    contour_smooth: float = DEFAULT_CONTOUR_SMOOTH,
    line_width: float = DEFAULT_REGION_LINE_WIDTH,
    brain_outline_width: float = DEFAULT_BRAIN_OUTLINE_WIDTH,
    show_region_contours: bool = True,
    background: str = "white",
    contour_color: str | None = None,
) -> np.ndarray:
    """High-res continuous signal slice: supersampled nearest + LANCZOS, vector contours."""
    from pipeline_modules.visualization.atlas_slice import _label_contour_lines

    labels = np.asarray(label_slice)
    signal = np.asarray(signal_slice, dtype=np.float32)
    if labels.shape != signal.shape:
        raise ValueError(f"signal shape {signal.shape} != label shape {labels.shape}")

    scale = max(int(pixel_scale), 1)
    ss = max(int(supersample), 1)
    height, width = labels.shape
    if background == "white":
        bg_rgb = (1.0, 1.0, 1.0)
        default_contour = "#334155"
    else:
        bg_rgb = (0.0, 0.0, 0.0)
        default_contour = "white"
    stroke_color = contour_color or default_contour

    cmap = _resolve_cmap(cmap_name)
    norm = mcolors.Normalize(vmin=float(vmin), vmax=float(vmax), clip=True)
    rgba = cmap(norm(signal))
    rgb = rgba[..., :3].astype(np.float32)
    outside = labels <= 0
    rgb[outside] = np.array(bg_rgb, dtype=np.float32)
    rgb_u8 = np.clip(np.round(rgb * 255.0), 0, 255).astype(np.uint8)

    hi = scale * ss
    rgb_hi = np.repeat(np.repeat(rgb_u8, hi, axis=0), hi, axis=1)
    image = Image.fromarray(rgb_hi, mode="RGB").resize(
        (width * scale, height * scale),
        resample=Image.Resampling.LANCZOS,
    )
    rgb_out = np.asarray(image, dtype=np.uint8).copy()

    if (show_region_contours and line_width > 0) or brain_outline_width > 0:
        fig_w = (width * scale) / 100.0
        fig_h = (height * scale) / 100.0
        fig = plt.figure(figsize=(fig_w, fig_h), dpi=100)
        try:
            ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
            ax.imshow(rgb_out, extent=(-0.5, width - 0.5, height - 0.5, -0.5), origin="upper")
            ax.set_xlim(-0.5, width - 0.5)
            ax.set_ylim(height - 0.5, -0.5)
            ax.set_axis_off()
            if show_region_contours and line_width > 0:
                lines = _label_contour_lines(labels, smoothing=float(contour_smooth))
                if lines:
                    ax.add_collection(
                        LineCollection(
                            lines,
                            colors=stroke_color,
                            linewidths=float(line_width) * (scale / 4.0),
                            antialiased=True,
                        )
                    )
            if brain_outline_width > 0:
                brain_lines = _mask_contour_lines(labels > 0, smoothing=max(float(contour_smooth), 0.8))
                if brain_lines:
                    ax.add_collection(
                        LineCollection(
                            brain_lines,
                            colors=stroke_color,
                            linewidths=float(brain_outline_width) * (scale / 4.0),
                            antialiased=True,
                        )
                    )
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=100, pad_inches=0)
            buf.seek(0)
            rgb_out = np.asarray(Image.open(buf).convert("RGB"), dtype=np.uint8)
        finally:
            plt.close(fig)

    brain = np.repeat(np.repeat((labels > 0), scale, axis=0), scale, axis=1)
    keep = ndi.binary_dilation(brain, iterations=1)
    if rgb_out.shape[0] != keep.shape[0] or rgb_out.shape[1] != keep.shape[1]:
        rgb_out = np.asarray(
            Image.fromarray(rgb_out).resize((keep.shape[1], keep.shape[0]), resample=Image.Resampling.LANCZOS),
            dtype=np.uint8,
        )
    bg_u8 = np.clip(np.round(np.array(bg_rgb) * 255.0), 0, 255).astype(np.uint8)
    rgb_out = rgb_out.copy()
    rgb_out[~keep] = bg_u8
    return rgb_out


def attach_colorbar_to_slice_rgb(
    slice_rgb: np.ndarray,
    *,
    vmin: float,
    vmax: float,
    vcenter: float | None = None,
    cmap_name: str,
    colorbar_label: str,
    background: str = "white",
    colorbar_width_px: int = 72,
) -> tuple[np.ndarray, dict[str, int]]:
    """Paste a colorbar beside a slice RGB and return (RGBA canvas, layout)."""
    if background == "white":
        theme = {
            "background": "white",
            "cbar_tick": "#334155",
            "cbar_edge": "#64748b",
            "cbar_label": "#1e293b",
        }
        bg_rgb = (1.0, 1.0, 1.0)
    else:
        theme = {
            "background": "black",
            "cbar_tick": "white",
            "cbar_edge": "white",
            "cbar_label": "white",
        }
        bg_rgb = (0.0, 0.0, 0.0)

    cmap = _resolve_cmap(cmap_name)
    norm = _resolve_norm(vmin, vmax, vcenter)
    lower = float(vmin)
    upper = float(vmax)
    center = float(vcenter) if vcenter is not None else None

    height, width = slice_rgb.shape[:2]
    cbar_h = max(int(round(height * 0.72)), 80)
    cbar_w = max(int(colorbar_width_px), 40)
    fig = plt.figure(figsize=(cbar_w / 100.0, cbar_h / 100.0), dpi=100)
    try:
        fig.patch.set_facecolor(bg_rgb)
        cax = fig.add_axes([0.35, 0.05, 0.25, 0.90])
        gradient = np.linspace(upper, lower, 256).reshape(256, 1)
        cax.imshow(gradient, aspect="auto", cmap=cmap, norm=norm)
        cax.set_xticks([])
        if center is not None and lower < center < upper:
            tick_vals = [lower, center, upper]
        else:
            tick_vals = [lower, upper]
        tick_pos = [
            (upper - value) / (upper - lower) * 255.0 if upper != lower else 0.0
            for value in tick_vals
        ]
        cax.set_yticks(tick_pos)
        cax.set_yticklabels([f"{value:g}" for value in tick_vals], color=theme["cbar_tick"], fontsize=8)
        cax.yaxis.set_label_position("right")
        cax.yaxis.tick_right()
        cax.set_ylabel(colorbar_label, color=theme["cbar_label"], fontsize=8)
        for spine in cax.spines.values():
            spine.set_color(theme["cbar_edge"])
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100, facecolor=bg_rgb)
        buf.seek(0)
        cbar_img = Image.open(buf).convert("RGB")
    finally:
        plt.close(fig)

    pad = 18
    out_w = width + pad + cbar_img.width + 8
    out_h = max(height, cbar_img.height + 20)
    bg_u8 = tuple(int(round(c * 255)) for c in bg_rgb)
    canvas = Image.new("RGB", (out_w, out_h), bg_u8)
    slice_top = (out_h - height) // 2
    canvas.paste(Image.fromarray(slice_rgb, mode="RGB"), (0, slice_top))
    canvas.paste(cbar_img, (width + pad, (out_h - cbar_img.height) // 2))
    rgba = np.asarray(canvas.convert("RGBA"), dtype=np.uint8)
    layout = {
        "image_width": int(out_w),
        "image_height": int(out_h),
        "slice_left": 0,
        "slice_top": int(slice_top),
        "slice_width": int(width),
        "slice_height": int(height),
        "atlas_width": int(width),  # caller should overwrite with native atlas size
        "atlas_height": int(height),
    }
    return rgba, layout


def save_slice_heatmap_png(
    painted_or_signal: np.ndarray,
    label_slice: np.ndarray,
    output_path: str | Path,
    *,
    mode: str = "labeled",
    vmin: float,
    vmax: float,
    vcenter: float | None = None,
    cmap_name: str = "white_orange_red_black",
    pixel_scale: int = DEFAULT_PIXEL_SCALE,
    supersample: int = DEFAULT_SUPERSAMPLE,
    contour_smooth: float = DEFAULT_CONTOUR_SMOOTH,
    line_width: float = DEFAULT_REGION_LINE_WIDTH,
    brain_outline_width: float = DEFAULT_BRAIN_OUTLINE_WIDTH,
    show_region_contours: bool = True,
    colorbar_label: str = "Value",
    background: str = "white",
    include_colorbar: bool = True,
) -> Path:
    """Render and save a heatmap PNG using the shared high-quality path."""
    if mode == "continuous":
        slice_rgb = render_continuous_signal_slice_rgb(
            painted_or_signal,
            label_slice,
            vmin=vmin,
            vmax=vmax,
            cmap_name=cmap_name,
            pixel_scale=pixel_scale,
            supersample=supersample,
            contour_smooth=contour_smooth,
            line_width=line_width,
            brain_outline_width=brain_outline_width,
            show_region_contours=show_region_contours,
            background=background,
        )
    else:
        slice_rgb = render_labeled_value_slice_rgb(
            painted_or_signal,
            label_slice,
            vmin=vmin,
            vmax=vmax,
            vcenter=vcenter,
            cmap_name=cmap_name,
            pixel_scale=pixel_scale,
            supersample=supersample,
            contour_smooth=contour_smooth,
            line_width=line_width,
            brain_outline_width=brain_outline_width,
            show_region_contours=show_region_contours,
            background=background,
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if include_colorbar:
        rgba, _layout = attach_colorbar_to_slice_rgb(
            slice_rgb,
            vmin=vmin,
            vmax=vmax,
            vcenter=vcenter,
            cmap_name=cmap_name,
            colorbar_label=colorbar_label,
            background=background,
        )
        Image.fromarray(rgba).save(output_path)
    else:
        Image.fromarray(slice_rgb).save(output_path)
    return output_path
