from __future__ import annotations

import argparse
import base64
import io
import json
import os
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy import ndimage
import tifffile
import torch
import torch.nn as nn

try:
    import ants
except ImportError:  # pragma: no cover - only needed for registration mode
    ants = None

from pipeline_modules.visualization.atlas_slice import (
    AXIS_NAMES,
    DEFAULT_ATLAS_LABEL,
    PLANE_TO_FIXED_AXIS,
    AtlasSlice,
    AtlasSliceSpec,
    _colormap_by_name,
    _contours_to_svg_d,
    _format_svg_color,
    _label_contour_lines,
    _mask_contour_lines,
    extract_atlas_slice,
)

SVG_NS = "http://www.w3.org/2000/svg"
ET.register_namespace("", SVG_NS)


def create_gaussian_kernel_3d(kernel_size=11, sigma=1.5):
    ax = np.arange(-kernel_size // 2 + 1.0, kernel_size // 2 + 1.0)
    xx, yy, zz = np.meshgrid(ax, ax, ax)
    kernel = np.exp(-(xx**2 + yy**2 + zz**2) / (2 * sigma**2))
    return kernel


def read_tiff_stack(path):
    path = Path(path)
    if path.is_dir():
        files = sorted(list(path.glob("*.tif*")))
        if not files:
            raise FileNotFoundError(f"No TIFF files found in directory: {path}")
        images = [np.array(Image.open(f)) for f in files]
        return np.array(images)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    img = Image.open(path)
    images = []
    for i in range(img.n_frames):
        img.seek(i)
        images.append(np.array(img))
    return np.array(images)


def downsample_mask(mask_array, config_path):
    if not os.path.exists(config_path):
        print(f"Warning: Config not found at {config_path}. Skipping downsampling.")
        return mask_array

    with open(config_path, "r") as f:
        cfg = json.load(f)

    input_res = None
    target_res = None
    if "input" in cfg and "resolution_xyz" in cfg["input"]:
        input_res = cfg["input"]["resolution_xyz"]
    elif "source_resolution" in cfg:
        input_res = cfg["source_resolution"]

    if "preprocessing" in cfg and "downsample" in cfg["preprocessing"] and "target_resolution_xyz" in cfg["preprocessing"]["downsample"]:
        target_res = cfg["preprocessing"]["downsample"]["target_resolution_xyz"]
    elif "target_resolution" in cfg:
        target_res = cfg["target_resolution"]

    if input_res is None or target_res is None:
        print("Warning: Could not find resolution settings in config. Skipping downsampling.")
        return mask_array

    factors = [s / t for s, t in zip(input_res, target_res)]
    factors_zyx = factors[::-1]
    print(f"Downsampling mask with factors (z,y,x): {factors_zyx}")
    print(f"Original shape: {mask_array.shape}")
    downsampled = ndimage.zoom(mask_array, factors_zyx, order=0)
    downsampled = (downsampled > 0).astype(np.uint8) * 255
    print(f"Downsampled shape: {downsampled.shape}")
    return downsampled


def apply_registration(mask_array, reference_path, transforms):
    if ants is None:
        raise ImportError("ANTsPy is required for --transforms registration but is not installed in this environment.")

    print("\nApplying registration transforms...")
    print(f"Reference: {reference_path}")
    print(f"Transforms: {transforms}")

    if not os.path.exists(reference_path):
        raise FileNotFoundError(f"Reference image not found: {reference_path}")
    fixed = ants.image_read(reference_path)
    mask_ants_data = np.transpose(mask_array, (2, 1, 0)).astype("float32")
    moving = ants.from_numpy(mask_ants_data, origin=[0, 0, 0], spacing=[1, 1, 1], direction=np.eye(3))
    warped = ants.apply_transforms(
        fixed=fixed,
        moving=moving,
        transformlist=transforms,
        interpolator="nearestNeighbor",
    )
    return np.transpose(warped.numpy(), (2, 1, 0))


def build_local_signal_volume(signal_volume, sigma=1.5, alpha=1.0, atlas_mask=None, kernel_size=11, normalize=False):
    volume = np.asarray(signal_volume, dtype=np.float32).copy()
    if atlas_mask is not None:
        mask = np.asarray(atlas_mask) > 0
        if mask.shape != volume.shape:
            raise ValueError(f"Signal volume shape {volume.shape} does not match atlas mask shape {mask.shape}")
        volume[~mask] = 0

    radiation_matrix = create_gaussian_kernel_3d(kernel_size, sigma)
    radiation_matrix = radiation_matrix[np.newaxis, np.newaxis, ...]
    conv = nn.Conv3d(1, 1, kernel_size, 1, padding=kernel_size // 2, bias=False)
    conv.weight = nn.Parameter(torch.Tensor(radiation_matrix), requires_grad=False)

    if torch.cuda.is_available():
        conv = conv.cuda()
        input_tensor = torch.Tensor(volume[np.newaxis, np.newaxis, ...]).cuda()
    else:
        input_tensor = torch.Tensor(volume[np.newaxis, np.newaxis, ...])

    with torch.no_grad():
        img_out = conv(input_tensor)
    local_signal = img_out.cpu().numpy().squeeze().astype(np.float32) * float(alpha)
    if atlas_mask is not None:
        local_signal[~mask] = 0
    if normalize:
        max_value = float(local_signal.max())
        if max_value > 0:
            local_signal /= max_value
    return local_signal


def _legacy_rgb_heat_volume(local_signal, edge, atlas_mask):
    heatimg = np.zeros(local_signal.shape, dtype=np.float32)
    heatimg = np.array([heatimg, heatimg, heatimg]).transpose((1, 2, 3, 0))
    edge_rgb = np.array([edge, edge, edge]).transpose((1, 2, 3, 0))

    heatimg[..., 0][local_signal > 255] = local_signal[local_signal > 255] - 255
    heatimg[..., 2][local_signal > 255] = 255
    heatimg[..., 2][local_signal > 255 * 2] -= local_signal[local_signal > 255 * 2] - 255 * 2
    heatimg[..., 2][local_signal <= 255] = local_signal[local_signal <= 255]
    heatimg[edge_rgb != 0] = edge_rgb[edge_rgb != 0]
    heatimg[atlas_mask == 0] = 0
    return np.clip(heatimg, 0, 255).astype(np.uint8)


def heatmap(save_img_path, edge_path, atlas_mask_path, save_path, alpha, sigma=1.5, resolution_cfg=None, transforms=None, reference=None):
    print(f"Loading input mask: {save_img_path}")
    img = read_tiff_stack(save_img_path)

    if resolution_cfg:
        img = downsample_mask(img, resolution_cfg)
    if transforms and reference:
        img = apply_registration(img, reference, transforms)

    print(f"Loading edge reference: {edge_path}")
    edge = read_tiff_stack(edge_path)
    print(f"Loading atlas mask: {atlas_mask_path}")
    atlas_mask = read_tiff_stack(atlas_mask_path)

    if img.shape != atlas_mask.shape:
        print(f"Warning: Shape mismatch. Input: {img.shape}, Atlas: {atlas_mask.shape}")

    print(f"Processing volume shape: {img.shape}")
    local_signal = build_local_signal_volume(img, sigma=sigma, alpha=alpha, atlas_mask=atlas_mask)
    heatimg = _legacy_rgb_heat_volume(local_signal, edge, atlas_mask)

    print(f"Saving heatmap to: {save_path}")
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(save_path, heatimg, compression="lzw")
    print("Done!")


def _slice_signal_volume(local_signal_volume: np.ndarray, atlas_slice: AtlasSlice) -> np.ndarray:
    if local_signal_volume.ndim != 3:
        raise ValueError(f"local_signal_volume must be 3D, got shape: {local_signal_volume.shape}")
    if atlas_slice.plane == "horizontal":
        return np.asarray(local_signal_volume[atlas_slice.index, :, :], dtype=np.float32)
    if atlas_slice.plane == "coronal":
        return np.asarray(local_signal_volume[:, atlas_slice.index, :], dtype=np.float32)
    return np.asarray(local_signal_volume[:, :, atlas_slice.index], dtype=np.float32)


def _render_local_slice_array(
    signal_slice: np.ndarray,
    label_slice: np.ndarray,
    *,
    cmap_name: str,
    vmin: float,
    vmax: float,
    dpi: int,
    line_width: float,
    brain_outline_width: float,
    colorbar_label: str,
) -> np.ndarray:
    height, width = label_slice.shape
    aspect = width / max(height, 1)
    long_side = 6.2
    figsize = (long_side, max(long_side / aspect, 1.0)) if aspect >= 1 else (max(long_side * aspect, 1.0), long_side)
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    cmap = _colormap_by_name(cmap_name).copy()
    cmap.set_bad((1, 1, 1, 0))
    masked_signal = np.ma.masked_where((label_slice <= 0) | (signal_slice <= vmin), signal_slice)
    image = ax.imshow(masked_signal, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="bilinear")

    region_lines = _label_contour_lines(label_slice, smoothing=1.4)
    if region_lines and line_width > 0:
        from matplotlib.collections import LineCollection

        ax.add_collection(
            LineCollection(
                region_lines,
                colors="black",
                linewidths=line_width,
                alpha=0.55,
                antialiaseds=True,
                capstyle="round",
                joinstyle="round",
            )
        )

    brain_lines = _mask_contour_lines(label_slice > 0, smoothing=1.8)
    if brain_lines and brain_outline_width > 0:
        from matplotlib.collections import LineCollection

        ax.add_collection(
            LineCollection(
                brain_lines,
                colors="black",
                linewidths=brain_outline_width,
                alpha=0.95,
                antialiaseds=True,
                capstyle="round",
                joinstyle="round",
            )
        )

    ax.set_axis_off()
    ax.set_xlim(-0.5, width - 0.5)
    ax.set_ylim(height - 0.5, -0.5)
    cbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.025, shrink=0.55)
    cbar.ax.tick_params(labelsize=7, length=2, width=0.6)
    cbar.outline.set_linewidth(0.6)
    cbar.set_label(colorbar_label, fontsize=8, labelpad=6)

    fig.tight_layout(pad=0.08)
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=dpi, facecolor="white", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    buffer.seek(0)
    return np.asarray(Image.open(buffer).convert("RGBA"))


def render_local_signal_atlas_slice(
    local_signal_volume: np.ndarray,
    label_path: str | Path,
    spec: AtlasSliceSpec,
    output_path: str | Path,
    *,
    cmap_name: str = "coolwarm",
    vmin: float = 0.0,
    vmax: float | None = None,
    dpi: int = 300,
    line_width: float = 0.16,
    brain_outline_width: float = 0.42,
    colorbar_label: str = "Signal Intensity",
) -> Path:
    atlas_slice = extract_atlas_slice(label_path, spec)
    signal_slice = _slice_signal_volume(local_signal_volume, atlas_slice)
    if signal_slice.shape != atlas_slice.image.shape:
        raise ValueError(f"Signal slice shape {signal_slice.shape} does not match atlas slice shape {atlas_slice.image.shape}")

    upper = float(vmax) if vmax is not None else float(np.nanpercentile(signal_slice[atlas_slice.image > 0], 99.5))
    if upper <= vmin:
        upper = float(vmin) + 1e-6

    rendered = _render_local_slice_array(
        signal_slice,
        atlas_slice.image,
        cmap_name=cmap_name,
        vmin=float(vmin),
        vmax=upper,
        dpi=int(dpi),
        line_width=float(line_width),
        brain_outline_width=float(brain_outline_width),
        colorbar_label=colorbar_label,
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() == ".svg":
        return _write_raster_svg_with_vector_outlines(
            rendered,
            atlas_slice,
            output_path,
            line_width=line_width,
            brain_outline_width=brain_outline_width,
        )

    Image.fromarray(rendered).save(output_path)
    return output_path


def _write_raster_svg_with_vector_outlines(
    rendered_rgba: np.ndarray,
    atlas_slice: AtlasSlice,
    output_path: Path,
    *,
    line_width: float,
    brain_outline_width: float,
) -> Path:
    img = Image.fromarray(rendered_rgba)
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    height, width = rendered_rgba.shape[:2]

    root = ET.Element(f"{{{SVG_NS}}}svg", {"width": str(width), "height": str(height), "viewBox": f"0 0 {width} {height}"})
    ET.SubElement(root, f"{{{SVG_NS}}}rect", {"x": "0", "y": "0", "width": str(width), "height": str(height), "fill": "white"})
    ET.SubElement(
        root,
        f"{{{SVG_NS}}}image",
        {
            "x": "0",
            "y": "0",
            "width": str(width),
            "height": str(height),
            "href": f"data:image/png;base64,{encoded}",
        },
    )

    ET.ElementTree(root).write(output_path, encoding="utf-8", xml_declaration=True)
    return output_path


def _parse_triplet(value: str) -> tuple[float, float, float]:
    parts = [part.strip() for part in str(value).split(",") if part.strip()]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("value must contain three comma-separated numbers")
    return tuple(float(part) for part in parts)  # type: ignore[return-value]


def _add_gaussian_soma(
    volume: np.ndarray,
    center: tuple[float, float, float],
    *,
    radius: tuple[float, float, float],
    amplitude: float,
) -> None:
    shape = volume.shape
    cd, ca, cm = center
    rd, ra, rm = radius
    d0, d1 = max(int(cd - rd * 3), 0), min(int(cd + rd * 3) + 1, shape[0])
    a0, a1 = max(int(ca - ra * 3), 0), min(int(ca + ra * 3) + 1, shape[1])
    m0, m1 = max(int(cm - rm * 3), 0), min(int(cm + rm * 3) + 1, shape[2])
    d, a, m = np.ogrid[d0:d1, a0:a1, m0:m1]
    soma = amplitude * np.exp(-0.5 * (((d - cd) / rd) ** 2 + ((a - ca) / ra) ** 2 + ((m - cm) / rm) ** 2))
    volume[d0:d1, a0:a1, m0:m1] += soma.astype(np.float32)


def _bezier_points(
    start: np.ndarray,
    control: np.ndarray,
    end: np.ndarray,
    count: int,
) -> np.ndarray:
    t = np.linspace(0.0, 1.0, count, dtype=np.float32)[:, None]
    return (1 - t) ** 2 * start + 2 * (1 - t) * t * control + t**2 * end


def _add_fiber_trace(
    volume: np.ndarray,
    points: np.ndarray,
    *,
    amplitude: float,
    radius: tuple[float, float, float] = (0.8, 1.6, 1.6),
    step: float = 0.7,
) -> None:
    shape = np.array(volume.shape)
    rd, ra, rm = radius
    if len(points) < 2:
        return

    dense_points = [points[0].astype(np.float32)]
    for start, end in zip(points[:-1], points[1:]):
        segment = end - start
        distance = float(np.linalg.norm(segment))
        count = max(int(np.ceil(distance / max(step, 1e-3))), 1)
        for t in np.linspace(0.0, 1.0, count + 1, dtype=np.float32)[1:]:
            dense_points.append((start + segment * t).astype(np.float32))

    for idx, point in enumerate(dense_points):
        taper = float(np.interp(idx, [0, max(len(dense_points) - 1, 1)], [amplitude, amplitude * 0.42]))
        cd, ca, cm = point
        d0, d1 = max(int(cd - rd * 2.5), 0), min(int(cd + rd * 2.5) + 1, shape[0])
        a0, a1 = max(int(ca - ra * 2.5), 0), min(int(ca + ra * 2.5) + 1, shape[1])
        m0, m1 = max(int(cm - rm * 2.5), 0), min(int(cm + rm * 2.5) + 1, shape[2])
        d, a, m = np.ogrid[d0:d1, a0:a1, m0:m1]
        brush = taper * np.exp(-0.5 * (((d - cd) / rd) ** 2 + ((a - ca) / ra) ** 2 + ((m - cm) / rm) ** 2))
        volume[d0:d1, a0:a1, m0:m1] += brush.astype(np.float32)


def build_synthetic_prv_signal(
    atlas_labels: np.ndarray,
    *,
    center: tuple[float, float, float] | None = None,
    spread: tuple[float, float, float] = (15.0, 70.0, 55.0),
    neuron_count: int = 100,
) -> np.ndarray:
    shape = atlas_labels.shape
    if center is None:
        center = (shape[0] * 0.46, shape[1] * 0.50, shape[2] * 0.55)

    rng = np.random.default_rng(7)
    signal = np.zeros(shape, dtype=np.float32)
    fiber_seed = np.zeros(shape, dtype=np.float32)
    center_arr = np.asarray(center, dtype=np.float32)
    dv_spread, ap_spread, ml_spread = spread

    cluster_count = max(6, min(14, max(int(neuron_count) // 12, 1)))
    cluster_centers: list[np.ndarray] = []
    for idx in range(cluster_count):
        if idx == 0:
            cluster_centers.append(center_arr.copy())
            continue
        angle = rng.uniform(0.0, np.pi * 2.0)
        radial = np.sqrt(rng.uniform(0.08, 1.0))
        candidate = center_arr + np.array(
            [
                rng.normal(0, dv_spread * 0.35),
                np.cos(angle) * ap_spread * radial * 1.1 + rng.normal(0, ap_spread * 0.18),
                np.sin(angle) * ml_spread * radial * 1.15 + rng.normal(0, ml_spread * 0.18),
            ],
            dtype=np.float32,
        )
        candidate = np.clip(candidate, [0, 0, 0], np.array(shape, dtype=np.float32) - 1)
        if atlas_labels[tuple(np.round(candidate).astype(int))] > 0:
            cluster_centers.append(candidate.astype(np.float32))

    soma_centers: list[np.ndarray] = []
    attempts = 0
    target_count = max(int(neuron_count), 1)
    while len(soma_centers) < target_count and attempts < target_count * 60:
        attempts += 1
        cluster = cluster_centers[attempts % len(cluster_centers)]
        candidate = cluster + np.array(
            [
                rng.normal(0, max(dv_spread * 0.22, 2.0)),
                rng.normal(0, max(ap_spread * 0.16, 5.0)),
                rng.normal(0, max(ml_spread * 0.16, 5.0)),
            ],
            dtype=np.float32,
        )
        candidate = np.clip(candidate, [0, 0, 0], np.array(shape, dtype=np.float32) - 1)
        idx = tuple(np.round(candidate).astype(int))
        if atlas_labels[idx] <= 0:
            continue
        if any(np.linalg.norm(candidate - existing) < 3.2 for existing in soma_centers):
            continue
        soma_centers.append(candidate.astype(np.float32))

    if not soma_centers:
        raise ValueError("Could not place any synthetic neurons inside atlas mask")

    for soma in soma_centers:
        distance = float(np.linalg.norm((soma - center_arr) / np.maximum(np.asarray(spread, dtype=np.float32), 1.0)))
        amp = float(np.clip(1.06 - 0.18 * distance + rng.normal(0, 0.045), 0.34, 1.0))
        radius = (
            float(rng.uniform(1.1, 1.8)),
            float(rng.uniform(2.1, 3.4)),
            float(rng.uniform(2.1, 3.6)),
        )
        _add_gaussian_soma(signal, tuple(soma), radius=radius, amplitude=amp)

        nearest_cluster = min(cluster_centers, key=lambda cluster: float(np.linalg.norm(cluster - soma)))
        start_anchor = nearest_cluster + np.array(
            [
                rng.normal(0, dv_spread * 0.08),
                rng.normal(0, ap_spread * 0.08),
                rng.normal(0, ml_spread * 0.08),
            ],
            dtype=np.float32,
        )
        start_anchor = np.clip(start_anchor, [0, 0, 0], np.array(shape, dtype=np.float32) - 1)
        midpoint = (start_anchor + soma) / 2
        control = midpoint + np.array(
            [
                rng.normal(0, dv_spread * 0.18),
                rng.normal(0, ap_spread * 0.24),
                rng.normal(0, ml_spread * 0.24),
            ],
            dtype=np.float32,
        )
        trunk_points = _bezier_points(start_anchor, control, soma, 180)
        trunk_amp = float(rng.uniform(0.22, 0.42))
        _add_fiber_trace(
            fiber_seed,
            trunk_points,
            amplitude=trunk_amp,
            radius=(0.42, 0.86, 0.86),
            step=0.38,
        )

        branch_total = int(rng.integers(3, 7))
        for _ in range(branch_total):
            branch_scale = float(rng.uniform(0.16, 0.9))
            branch_start = trunk_points[int(len(trunk_points) * branch_scale)]
            branch_direction = soma - nearest_cluster
            lateral = np.array(
                [
                    rng.normal(0, dv_spread * 0.18),
                    rng.normal(0, ap_spread * 0.34),
                    rng.normal(0, ml_spread * 0.34),
                ],
                dtype=np.float32,
            )
            branch_end = branch_start + branch_direction * rng.uniform(0.06, 0.18) + lateral
            branch_end = np.clip(branch_end, [0, 0, 0], np.array(shape, dtype=np.float32) - 1)
            if atlas_labels[tuple(np.round(branch_end).astype(int))] <= 0:
                continue
            branch_control = (branch_start + branch_end) / 2 + np.array(
                [
                    rng.normal(0, dv_spread * 0.1),
                    rng.normal(0, ap_spread * 0.16),
                    rng.normal(0, ml_spread * 0.16),
                ],
                dtype=np.float32,
            )
            branch_points = _bezier_points(branch_start, branch_control, branch_end, 110)
            branch_amp = float(rng.uniform(0.08, 0.16))
            _add_fiber_trace(
                fiber_seed,
                branch_points,
                amplitude=branch_amp,
                radius=(0.34, 0.68, 0.68),
                step=0.34,
            )

            twig_total = int(rng.integers(1, 4))
            for _ in range(twig_total):
                twig_scale = float(rng.uniform(0.22, 0.88))
                twig_start = branch_points[int(len(branch_points) * twig_scale)]
                twig_end = twig_start + np.array(
                    [
                        rng.normal(0, dv_spread * 0.08),
                        rng.normal(0, ap_spread * 0.18),
                        rng.normal(0, ml_spread * 0.18),
                    ],
                    dtype=np.float32,
                )
                twig_end = np.clip(twig_end, [0, 0, 0], np.array(shape, dtype=np.float32) - 1)
                if atlas_labels[tuple(np.round(twig_end).astype(int))] <= 0:
                    continue
                twig_control = (twig_start + twig_end) / 2 + np.array(
                    [
                        rng.normal(0, dv_spread * 0.05),
                        rng.normal(0, ap_spread * 0.08),
                        rng.normal(0, ml_spread * 0.08),
                    ],
                    dtype=np.float32,
                )
                twig_points = _bezier_points(twig_start, twig_control, twig_end, 72)
                _add_fiber_trace(
                    fiber_seed,
                    twig_points,
                    amplitude=float(rng.uniform(0.04, 0.09)),
                    radius=(0.28, 0.52, 0.52),
                    step=0.3,
                )

    fiber_signal = ndimage.gaussian_filter(fiber_seed, sigma=(0.42, 0.72, 0.72), mode="constant")
    fiber_core = ndimage.gaussian_filter(fiber_seed, sigma=(0.12, 0.18, 0.18), mode="constant")
    signal += fiber_signal * 1.02 + fiber_core * 0.28
    signal += ndimage.gaussian_filter(signal, sigma=(0.24, 0.34, 0.34), mode="constant") * 0.04
    signal[atlas_labels <= 0] = 0
    signal -= float(signal.min())
    max_value = float(signal.max())
    if max_value > 0:
        signal /= max_value
    return signal.astype(np.float32)


def generate_prv_sample(
    *,
    label_path: str | Path = DEFAULT_ATLAS_LABEL,
    output_dir: str | Path = "S:/可视化素材/heatmap",
    sample_count: int = 10,
    cmap_name: str = "white_blue_red",
    dpi: int = 300,
    center: tuple[float, float, float] | None = None,
    spread: tuple[float, float, float] = (15.0, 70.0, 55.0),
    neuron_count: int = 100,
) -> list[Path]:
    label_path = Path(label_path)
    labels = np.asarray(tifffile.memmap(str(label_path)))
    signal = build_synthetic_prv_signal(labels, center=center, spread=spread, neuron_count=neuron_count)
    brain_dv = np.flatnonzero(np.any(labels > 0, axis=(1, 2)))
    if len(brain_dv) == 0:
        raise ValueError("Atlas label contains no nonzero brain voxels")

    start = int(np.quantile(brain_dv, 0.22))
    stop = int(np.quantile(brain_dv, 0.78))
    indices = np.linspace(start, stop, int(sample_count)).round().astype(int)
    peak_index = int(np.argmax(signal.max(axis=(1, 2))))
    if len(indices) > 0:
        indices[len(indices) // 2] = peak_index
        indices = np.clip(np.sort(indices), int(brain_dv.min()), int(brain_dv.max()))

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for order, index in enumerate(indices):
        spec = AtlasSliceSpec("horizontal", "index", float(index))
        for suffix in (".png", ".svg"):
            output_path = output_dir / f"prv_horizontal_{order:02d}{suffix}"
            outputs.append(
                render_local_signal_atlas_slice(
                    signal,
                    label_path,
                    spec,
                    output_path,
                    cmap_name=cmap_name,
                    vmin=0.0,
                    vmax=1.0,
                    dpi=dpi,
                    line_width=0.12,
                    brain_outline_width=0.36,
                    colorbar_label="Signal Intensity",
                )
            )
    return outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate 3D or atlas-slice signal heatmaps")
    parser.add_argument("--mode", choices=["stack", "atlas-slice", "prv-sample"], default="stack")
    parser.add_argument("--input", help="Path to input mask/density image (TIFF stack or folder)")
    parser.add_argument("--output", help="Path to save output heatmap")

    default_atlas_dir = Path(__file__).parent.parent / "Allen_brainatlas"
    parser.add_argument("--edge", default=str(default_atlas_dir / "edge.tiff"), help="Path to edge reference image")
    parser.add_argument("--atlas_mask", default=str(default_atlas_dir / "atlas_mask.tiff"), help="Path to atlas mask image")
    parser.add_argument("--alpha", type=float, default=2.0, help="Intensity scaling factor")
    parser.add_argument("--sigma", type=float, default=2.0, help="Gaussian smoothing sigma")
    parser.add_argument("--config", help="Path to config.json for downsampling")
    parser.add_argument("--transforms", nargs="+", help="List of inverse transform files (Image -> Atlas)")
    parser.add_argument("--reference", help="Path to reference atlas image (for registration)")

    parser.add_argument("--label", default=str(DEFAULT_ATLAS_LABEL), help="3D atlas label TIFF path")
    parser.add_argument("--plane", default="horizontal", choices=["coronal", "sagittal", "horizontal"], help="Atlas slice plane")
    parser.add_argument("--coord-system", default="index", choices=["bregma-mm", "ccf-um", "index"], help="Coordinate system")
    parser.add_argument("--coord", type=float, help="Coordinate value for atlas-slice mode")
    parser.add_argument("--atlas-resolution-um", type=float, default=25.0)
    parser.add_argument("--bregma-index", default="18,216,228")
    parser.add_argument("--vmin", type=float, default=0.0)
    parser.add_argument("--vmax", type=float, default=None)
    parser.add_argument("--cmap", default="white_blue_red")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--line-width", type=float, default=0.16)
    parser.add_argument("--brain-outline-width", type=float, default=0.42)
    parser.add_argument("--colorbar-label", default="Signal Intensity")

    parser.add_argument("--sample-output-dir", default="S:/可视化素材/heatmap")
    parser.add_argument("--sample-count", type=int, default=10)
    parser.add_argument("--sample-center", type=_parse_triplet, default=None, help="Synthetic PRV center as dv,ap,ml index")
    parser.add_argument("--sample-spread", type=_parse_triplet, default=(15.0, 70.0, 55.0), help="Synthetic PRV spread as dv,ap,ml voxels")
    parser.add_argument("--sample-neuron-count", type=int, default=100, help="Approximate number of synthetic neuron somas")
    return parser


def parse_bregma_index(value: str) -> tuple[int, int, int]:
    parts = [part.strip() for part in str(value).split(",") if part.strip()]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("bregma index must be dv,ap,ml")
    return tuple(int(part) for part in parts)  # type: ignore[return-value]


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if args.mode == "stack":
            if not args.input or not args.output:
                parser.error("--input and --output are required for --mode stack")
            heatmap(args.input, args.edge, args.atlas_mask, args.output, args.alpha, args.sigma, args.config, args.transforms, args.reference)
            payload = {"mode": args.mode, "output": str(args.output)}
        elif args.mode == "atlas-slice":
            if not args.input or not args.output or args.coord is None:
                parser.error("--input, --output, and --coord are required for --mode atlas-slice")
            signal = read_tiff_stack(args.input).astype(np.float32)
            labels = np.asarray(tifffile.memmap(str(args.label)))
            local_signal = build_local_signal_volume(signal, sigma=args.sigma, alpha=args.alpha, atlas_mask=labels, normalize=True)
            spec = AtlasSliceSpec(args.plane, args.coord_system, args.coord, args.atlas_resolution_um, parse_bregma_index(args.bregma_index))
            output_path = render_local_signal_atlas_slice(
                local_signal,
                args.label,
                spec,
                args.output,
                cmap_name=args.cmap,
                vmin=args.vmin,
                vmax=args.vmax,
                dpi=args.dpi,
                line_width=args.line_width,
                brain_outline_width=args.brain_outline_width,
                colorbar_label=args.colorbar_label,
            )
            payload = {"mode": args.mode, "output": str(output_path), "plane": args.plane, "coordinate": args.coord}
        else:
            outputs = generate_prv_sample(
                label_path=args.label,
                output_dir=args.sample_output_dir,
                sample_count=args.sample_count,
                cmap_name=args.cmap,
                dpi=args.dpi,
                center=args.sample_center,
                spread=args.sample_spread,
                neuron_count=args.sample_neuron_count,
            )
            payload = {"mode": args.mode, "output_dir": str(args.sample_output_dir), "outputs": [str(path) for path in outputs]}
    except Exception as exc:
        print(f"Error: {exc}")
        return 1

    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
