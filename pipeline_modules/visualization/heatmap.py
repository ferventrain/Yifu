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


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_reference_dir() -> Path:
    return project_root() / "data" / "reference"


def load_json(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_sample_config(sample_dir: str | Path) -> Path:
    sample_dir = Path(sample_dir)
    candidates = [
        sample_dir.parent / "config.json",
        sample_dir.parent / "config" / "config.json",
        project_root() / "config" / "config.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not find config.json near sample_dir: {sample_dir}")


def default_sample_stack_output(sample_dir: str | Path) -> Path:
    sample_dir = Path(sample_dir)
    return sample_dir / "visualization" / f"{sample_dir.name}_heatmap3d_stack.tiff"


def default_sample_stack_volume(sample_dir: str | Path) -> Path:
    sample_dir = Path(sample_dir)
    return sample_dir / "visualization" / f"{sample_dir.name}_heatmap3d_volume.tiff"


def default_sample_points_csv(sample_dir: str | Path) -> Path:
    return Path(sample_dir) / "visualization" / "points.csv"


def resolve_sample_stack_defaults(
    sample_dir: str | Path,
    *,
    config_path: str | Path | None = None,
) -> dict[str, object]:
    sample_dir = Path(sample_dir)
    cfg_path = Path(config_path) if config_path else resolve_sample_config(sample_dir)
    cfg = load_json(cfg_path)
    input_cfg = cfg.get("input", {})
    preprocessing_cfg = cfg.get("preprocessing", {})

    channels = input_cfg.get("channels", {})
    signal_ch = f"ch{channels.get('signal', '1')}"
    register_ch = f"ch{channels.get('registration', '0')}"
    resolution_xyz = tuple(float(value) for value in input_cfg.get("resolution_xyz", (1.8, 1.8, 2.0)))
    target_resolution_xyz = tuple(
        float(value) for value in preprocessing_cfg.get("downsample", {}).get("target_resolution_xyz", (25.0, 25.0, 25.0))
    )
    return {
        "config_path": cfg_path,
        "signal_ch": signal_ch,
        "register_ch": register_ch,
        "resolution_xyz": resolution_xyz,
        "target_resolution_xyz": target_resolution_xyz,
        "mask_zarr": sample_dir / f"{signal_ch}_mask.zarr",
        "sample_reference_nii": sample_dir / f"{register_ch}_downsample" / "volume.nii.gz",
        "transforms_dir": sample_dir / "transforms",
        "atlas_image": default_reference_dir() / "atlas_label.tiff",
        "edge": default_reference_dir() / "atlas_edge.tiff",
        "atlas_mask": default_reference_dir() / "atlas_label.tiff",
        "output": default_sample_stack_output(sample_dir),
        "output_volume": default_sample_stack_volume(sample_dir),
        "points_csv": default_sample_points_csv(sample_dir),
    }


def create_gaussian_kernel_3d(kernel_size=7, sigma=1.0):
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
    return np.asarray(tifffile.imread(str(path)))


def read_volume(path, *, dataset_name: str = "0"):
    path = Path(path)
    if path.suffix.lower() == ".zarr":
        _ = dataset_name
        raise ValueError(
            "Direct Zarr loading was removed to avoid materializing full-resolution volumes. "
            "Use --mode sample-stack for sample-space mask Zarr inputs, or pass an atlas-space TIFF volume."
        )
    return read_tiff_stack(path)


def build_local_signal_volume(signal_volume, sigma=1.0, alpha=1.0, atlas_mask=None, kernel_size=7, normalize=False):
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


_HEATMAP_CMAP_STOPS = [
    (0.000, np.array([0x00, 0x00, 0x00])),
    (0.125, np.array([0x1e, 0x09, 0x4f])),
    (0.250, np.array([0x3f, 0x07, 0x61])),
    (0.375, np.array([0x71, 0x17, 0x6e])),
    (0.500, np.array([0xbd, 0x33, 0x4e])),
    (0.625, np.array([0xe0, 0x4f, 0x31])),
    (0.750, np.array([0xf9, 0x8b, 0x0e])),
    (0.875, np.array([0xeb, 0xf3, 0x77])),
    (1.000, np.array([0xff, 0xff, 0xff])),
]


def _build_heatmap_lut(num_entries: int = 256) -> np.ndarray:
    positions = np.array([pos for pos, _ in _HEATMAP_CMAP_STOPS])
    colors = np.array([color for _, color in _HEATMAP_CMAP_STOPS], dtype=np.float32)
    lut = np.zeros((num_entries, 3), dtype=np.float32)
    x = np.linspace(0.0, 1.0, num_entries)
    for channel in range(3):
        lut[:, channel] = np.interp(x, positions, colors[:, channel])
    return np.round(lut).astype(np.uint8)


def _legacy_rgb_heat_volume(local_signal, edge, atlas_mask):
    signal = np.asarray(local_signal, dtype=np.float32)
    max_val = signal.max()
    if max_val > 0:
        normalized = np.clip(signal / max_val, 0.0, 1.0)
    else:
        normalized = signal

    lut = _build_heatmap_lut()
    idx = (normalized * (len(lut) - 1)).astype(np.int32)
    heatimg = lut[idx]

    edge_values = np.asarray(edge)
    edge_mask = edge_values != 0
    if np.any(edge_mask):
        edge_uint8 = np.clip(edge_values, 0, 255).astype(np.uint8)
        heatimg[edge_mask, 0] = edge_uint8[edge_mask]
        heatimg[edge_mask, 1] = edge_uint8[edge_mask]
        heatimg[edge_mask, 2] = edge_uint8[edge_mask]

    heatimg[np.asarray(atlas_mask) == 0] = 0
    return heatimg


def heatmap(
    save_img_path,
    edge_path,
    atlas_mask_path,
    save_path,
    alpha,
    sigma=1.0,
    resolution_cfg=None,
    transforms=None,
    reference=None,
    dataset_name="0",
    atlas_dataset_name="0",
    edge_dataset_name="0",
    normalize=False,
    vmax=None,
    scale_max=510.0,
):
    print(f"Loading input mask: {save_img_path}")
    img = read_volume(save_img_path, dataset_name=dataset_name)

    if resolution_cfg or transforms or reference:
        raise ValueError(
            "Legacy in-memory downsample/registration was removed. "
            "Use --mode sample-stack to warp sample-space mask Zarr block-wise before rendering."
        )

    return render_heatmap_stack(
        img,
        edge_path=edge_path,
        atlas_mask_path=atlas_mask_path,
        save_path=save_path,
        alpha=alpha,
        sigma=sigma,
        atlas_dataset_name=atlas_dataset_name,
        edge_dataset_name=edge_dataset_name,
        normalize=normalize,
        vmax=vmax,
        scale_max=scale_max,
    )


def render_heatmap_stack(
    img,
    *,
    edge_path,
    atlas_mask_path,
    save_path,
    alpha,
    sigma=1.0,
    atlas_dataset_name="0",
    edge_dataset_name="0",
    normalize=False,
    vmax=None,
    scale_max=510.0,
):
    img = np.asarray(img)

    print(f"Loading edge reference: {edge_path}")
    edge = read_volume(edge_path, dataset_name=edge_dataset_name)
    print(f"Loading atlas mask: {atlas_mask_path}")
    atlas_mask = read_volume(atlas_mask_path, dataset_name=atlas_dataset_name)

    if img.shape != atlas_mask.shape:
        raise ValueError(
            f"Input volume shape {img.shape} does not match atlas mask shape {atlas_mask.shape}. "
            "Use an atlas-space mask/density volume, or warp/downsample the sample-space mask first."
        )
    if edge.shape != atlas_mask.shape:
        raise ValueError(f"Edge volume shape {edge.shape} does not match atlas mask shape {atlas_mask.shape}")

    print(f"Processing volume shape: {img.shape}")
    local_signal = build_local_signal_volume(img, sigma=sigma, alpha=alpha, atlas_mask=atlas_mask)
    max_signal = float(local_signal.max())
    print(f"Local signal range before display scaling: min={float(local_signal.min()):.6g}, max={max_signal:.6g}")
    if normalize:
        display_vmax = float(vmax) if vmax is not None else max_signal
        if display_vmax > 0:
            local_signal = np.clip(local_signal / display_vmax, 0, 1) * float(scale_max)
            print(f"Normalized local signal for display: vmax={display_vmax:.6g}, scale_max={float(scale_max):.6g}")
        else:
            print("Warning: local signal max is 0; skipping normalization.")
    heatimg = _legacy_rgb_heat_volume(local_signal, edge, atlas_mask)

    print(f"Saving heatmap to: {save_path}")
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(save_path, heatimg, compression="lzw")
    print("Done!")


def generate_sample_stack_heatmap(
    *,
    sample_dir: str | Path | None,
    signal_ch: str,
    register_ch: str,
    mask_zarr: str | Path | None,
    dataset_name: str,
    sample_reference_nii: str | Path | None,
    atlas_image: str | Path,
    transforms_dir: str | Path | None,
    transforms: str,
    resolution_xyz: str,
    target_resolution_xyz: str,
    foreground_mode: str,
    foreground_label: int,
    block_shape: str,
    min_voxels_per_point: int,
    volume_mode: str,
    output_volume: str | Path | None,
    edge_path: str | Path,
    atlas_mask_path: str | Path,
    output: str | Path,
    alpha: float,
    sigma: float,
    atlas_dataset_name: str = "0",
    edge_dataset_name: str = "0",
    normalize: bool = False,
    vmax: float | None = None,
    scale_max: float = 510.0,
) -> dict[str, object]:
    if ants is None:
        raise ImportError("ANTsPy is required for --mode sample-stack.")

    from pipeline_modules.visualization.warp_mask_zarr_to_atlas_points import (
        accumulate_sample_grid,
        atlas_volume_to_points,
        parse_block_shape,
        parse_triplet,
        resolve_inverse_transforms,
        resolve_mask_zarr,
        resolve_sample_reference_nii,
        resolve_atlas_resolution_xyz,
        write_outputs,
        warp_sample_grid_to_atlas,
        write_volume_output,
    )
    from pipeline_modules.segmentation.zarr_utils import open_zarr_dataset

    print("Stage 1/4: resolve sample inputs")
    sample_dir_value = sample_dir if sample_dir else None
    mask_zarr_path = resolve_mask_zarr(
        sample_dir=sample_dir_value,
        signal_ch=signal_ch,
        mask_zarr=mask_zarr,
    )
    sample_reference_path = resolve_sample_reference_nii(
        sample_dir=sample_dir_value,
        register_ch=register_ch,
        sample_reference_nii=sample_reference_nii,
    )
    transforms_root = transforms_dir or str(Path(sample_reference_path).parents[1] / "transforms")
    transformlist = resolve_inverse_transforms(transforms_root, transforms)
    if not transformlist:
        raise ValueError(f"No inverse transforms found under: {transforms_root}")

    print(f"Using sample mask: {mask_zarr_path}")
    print(f"Using sample reference: {sample_reference_path}")
    print(f"Using transforms: {transformlist}")

    cached_volume_path = Path(output_volume) if output_volume else default_sample_stack_volume(sample_dir)
    points_csv_path = default_sample_points_csv(sample_dir)
    summary: dict[str, object] = {
        "success": True,
        "mode": "sample-stack",
        "sample_reference_nii": str(sample_reference_path),
        "atlas_image": str(atlas_image),
        "transformlist": transformlist,
        "cached_volume_path": str(cached_volume_path),
    }

    if cached_volume_path.exists():
        print(f"Using cached atlas-space volume: {cached_volume_path}")
        atlas_volume = read_tiff_stack(cached_volume_path)
        raw_atlas_spacing_xyz = tuple()
        summary["cache_hit"] = True
    else:
        sample_ref = ants.image_read(str(sample_reference_path))
        sample_shape_zyx = tuple(int(value) for value in sample_ref.shape[::-1])
        resolution = parse_triplet(resolution_xyz, name="resolution_xyz")
        target_resolution = parse_triplet(target_resolution_xyz, name="target_resolution_xyz")

        arr = open_zarr_dataset(mask_zarr_path, dataset_name=dataset_name)
        fallback_block_shape = tuple(int(value) for value in (getattr(arr, "chunks", None) or arr.shape))
        resolved_block_shape = parse_block_shape(block_shape, fallback_block_shape)

        print("Stage 2/4: binning sample mask")
        sample_volume, summary = accumulate_sample_grid(
            mask_zarr_path,
            resolution_xyz=resolution,
            target_resolution_xyz=target_resolution,
            output_shape_zyx=sample_shape_zyx,
            dataset_name=dataset_name,
            foreground_mode=foreground_mode,
            foreground_label=foreground_label,
            block_shape=resolved_block_shape,
            min_voxels_per_point=min_voxels_per_point,
            volume_mode=volume_mode,
        )
        print("Stage 3/4: warping binned volume into atlas space")
        atlas_volume, raw_atlas_spacing_xyz = warp_sample_grid_to_atlas(
            sample_volume,
            sample_reference_nii=sample_reference_path,
            atlas_image=atlas_image,
            transformlist=transformlist,
            interpolator="linear" if volume_mode == "count" else "nearestNeighbor",
            binarize=volume_mode == "binary",
        )

        print(f"Writing cached atlas-space volume to: {cached_volume_path}")
        cached_volume_path.parent.mkdir(parents=True, exist_ok=True)
        write_volume_output(atlas_volume, cached_volume_path)
        summary["cache_hit"] = False
        summary["output_volume"] = str(cached_volume_path)

    if points_csv_path.exists() and summary.get("cache_hit"):
        print(f"Using cached atlas-space points CSV: {points_csv_path}")
    else:
        print(f"Writing atlas-space points CSV to: {points_csv_path}")
        atlas_resolution_xyz = resolve_atlas_resolution_xyz("", "25,25,25")
        table = atlas_volume_to_points(atlas_volume, atlas_resolution_xyz=atlas_resolution_xyz, max_points=150_000)
        point_outputs = write_outputs(table, summary, points_csv_path)
        summary["points_csv"] = str(point_outputs["csv"])
        summary["exported_points"] = int(len(table))

    print("Stage 4/4: rendering heatmap")
    render_heatmap_stack(
        atlas_volume,
        edge_path=edge_path,
        atlas_mask_path=atlas_mask_path,
        save_path=output,
        alpha=alpha,
        sigma=sigma,
        atlas_dataset_name=atlas_dataset_name,
        edge_dataset_name=edge_dataset_name,
        normalize=normalize,
        vmax=vmax,
        scale_max=scale_max,
    )

    summary.update(
        {
            "atlas_shape_zyx": list(atlas_volume.shape),
            "raw_atlas_image_spacing_xyz": list(raw_atlas_spacing_xyz) if raw_atlas_spacing_xyz else [],
            "heatmap_output": str(output),
        }
    )
    summary_path = Path(output).with_suffix(".json")
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    print(f"Saved summary to: {summary_path}")
    return {"output": str(output), "summary": str(summary_path), "atlas_shape_zyx": list(atlas_volume.shape)}


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
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")

    cmap = _colormap_by_name(cmap_name).copy()
    cmap.set_bad((0, 0, 0, 0))
    masked_signal = np.ma.masked_where((label_slice <= 0) | (signal_slice <= vmin), signal_slice)
    image = ax.imshow(masked_signal, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")

    region_lines = _label_contour_lines(label_slice, smoothing=1.4)
    if region_lines and line_width > 0:
        from matplotlib.collections import LineCollection

        ax.add_collection(
            LineCollection(
                region_lines,
                colors="white",
                linewidths=line_width,
                alpha=0.95,
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
                colors="white",
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
    cbar.ax.tick_params(labelsize=7, length=2, width=0.6, colors="white")
    cbar.outline.set_linewidth(0.6)
    cbar.outline.set_edgecolor("white")
    cbar.set_label(colorbar_label, fontsize=8, labelpad=6, color="white")

    fig.tight_layout(pad=0.08)
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=dpi, facecolor="black", bbox_inches="tight", pad_inches=0.02)
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
    line_width: float = 0.3,
    brain_outline_width: float = 0.3,
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
    ET.SubElement(root, f"{{{SVG_NS}}}rect", {"x": "0", "y": "0", "width": str(width), "height": str(height), "fill": "black"})
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
                    line_width=0.3,
                    brain_outline_width=0.3,
                    colorbar_label="Signal Intensity",
                )
            )
    return outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate 3D or atlas-slice signal heatmaps")
    parser.add_argument("--mode", choices=["stack", "sample-stack", "atlas-slice", "prv-sample"], default="stack")
    parser.add_argument("--input", help="Path to input mask/density image (TIFF stack or folder)")
    parser.add_argument("--output", help="Path to save output heatmap")

    parser.add_argument("--config", default="", help="Path to project config.json used to resolve sample defaults")
    parser.add_argument("--edge", default="", help="Path to atlas-space edge reference image")
    parser.add_argument("--atlas_mask", default="", help="Path to atlas-space brain mask/label image")
    parser.add_argument("--alpha", type=float, default=2.0, help="Intensity scaling factor")
    parser.add_argument("--sigma", type=float, default=2.0, help="Gaussian smoothing sigma")
    parser.add_argument("--transforms", default="", help="Comma-separated inverse transform paths in ANTs order for --mode sample-stack")
    parser.add_argument("--reference", help=argparse.SUPPRESS)
    parser.add_argument("--dataset-name", default="0", help="Dataset name when --input is a Zarr group")
    parser.add_argument("--atlas-dataset-name", default="0", help="Dataset name when --atlas_mask is a Zarr group")
    parser.add_argument("--edge-dataset-name", default="0", help="Dataset name when --edge is a Zarr group")
    parser.add_argument("--normalize", action="store_true", help="Normalize stack local signal before RGB display scaling")
    parser.add_argument("--display-vmax", type=float, default=None, help="Value mapped to --scale-max when --normalize is used")
    parser.add_argument("--scale-max", type=float, default=510.0, help="Display scale maximum used by --normalize")

    parser.add_argument("--sample-dir", default="", help="Sample directory for --mode sample-stack")
    parser.add_argument("--signal-ch", default="ch1", help="Signal channel label for --mode sample-stack")
    parser.add_argument("--register-ch", default="ch0", help="Registration channel label for --mode sample-stack")
    parser.add_argument("--sample-reference-nii", default="", help="Downsampled sample NIfTI used for registration")
    parser.add_argument("--atlas-image", default="", help="Atlas fixed image; defaults to data/reference/atlas_label.tiff")
    parser.add_argument("--transforms-dir", default="", help="Directory containing inverse transforms")
    parser.add_argument("--resolution-xyz", default="", help="Input mask voxel size in microns as x,y,z")
    parser.add_argument("--target-resolution-xyz", default="", help="Sample grid voxel size in microns as x,y,z")
    parser.add_argument("--foreground-mode", choices=("nonzero", "equal"), default="equal")
    parser.add_argument("--foreground-label", type=int, default=1)
    parser.add_argument("--block-shape", default="", help="Optional block shape in z,y,x order")
    parser.add_argument("--min-voxels-per-point", type=int, default=1)
    parser.add_argument("--volume-mode", choices=("binary", "count"), default="binary")
    parser.add_argument("--output-volume", default="", help="Optional atlas-space volume TIFF written before heatmap rendering")

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
            heatmap(
                args.input,
                args.edge,
                args.atlas_mask,
                args.output,
                args.alpha,
                args.sigma,
                args.config,
                args.transforms,
                args.reference,
                args.dataset_name,
                args.atlas_dataset_name,
                args.edge_dataset_name,
                args.normalize,
                args.display_vmax,
                args.scale_max,
            )
            payload = {"mode": args.mode, "output": str(args.output)}
        elif args.mode == "sample-stack":
            if not args.sample_dir:
                parser.error("--sample-dir is required for --mode sample-stack")
            defaults = resolve_sample_stack_defaults(args.sample_dir, config_path=args.config or None)
            payload = generate_sample_stack_heatmap(
                sample_dir=args.sample_dir,
                signal_ch=defaults["signal_ch"],
                register_ch=defaults["register_ch"],
                mask_zarr=args.input or defaults["mask_zarr"],
                dataset_name=args.dataset_name,
                sample_reference_nii=args.sample_reference_nii or defaults["sample_reference_nii"],
                atlas_image=args.atlas_image or defaults["atlas_image"],
                transforms_dir=args.transforms_dir or defaults["transforms_dir"],
                transforms=args.transforms,
                resolution_xyz=args.resolution_xyz or defaults["resolution_xyz"],
                target_resolution_xyz=args.target_resolution_xyz or defaults["target_resolution_xyz"],
                foreground_mode=args.foreground_mode,
                foreground_label=args.foreground_label,
                block_shape=args.block_shape,
                min_voxels_per_point=args.min_voxels_per_point,
                volume_mode=args.volume_mode,
                output_volume=args.output_volume or None,
                edge_path=args.edge or defaults["edge"],
                atlas_mask_path=args.atlas_mask or defaults["atlas_mask"],
                output=args.output or defaults["output"],
                alpha=args.alpha,
                sigma=args.sigma,
                atlas_dataset_name=args.atlas_dataset_name,
                edge_dataset_name=args.edge_dataset_name,
                normalize=args.normalize,
                vmax=args.display_vmax,
                scale_max=args.scale_max,
            )
        elif args.mode == "atlas-slice":
            if not args.input or not args.output or args.coord is None:
                parser.error("--input, --output, and --coord are required for --mode atlas-slice")
            signal = read_volume(args.input, dataset_name=args.dataset_name).astype(np.float32)
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
