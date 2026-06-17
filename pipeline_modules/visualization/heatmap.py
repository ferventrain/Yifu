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
    build_region_metric_lookup,
    build_region_name_lookup,
    collect_regions_missing_metric_data,
    compute_symmetric_metric_limits,
    extract_atlas_slice,
    resolve_slice_region_values,
    subtract_region_metric_values,
)

from pipeline_modules.utils.deliverable_paths import (
    brain_distribution_stats_xlsx,
    heatmap_2d_dir,
    heatmap_3d_colorbar_png,
    heatmap_3d_png,
    heatmap_3d_stack_tiff,
    heatmap_3d_summary_json,
    heatmap_3d_volume_tiff,
    legacy_brain_distribution_candidates,
    legacy_heatmap_3d_volume_candidates,
)

SVG_NS = "http://www.w3.org/2000/svg"
ET.register_namespace("", SVG_NS)

BATCH_SLICE_DEFAULT_PERCENTILE = 99.5
SUBTRACT_DIFF_DEFAULT_PERCENTILE = 95.0


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_reference_dir() -> Path:
    from pipeline_modules.utils.data_paths import reference_dir

    return reference_dir()


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


def default_sample_stack_output(sample_dir: str | Path, signal_ch: str = "ch1") -> Path:
    return heatmap_3d_stack_tiff(sample_dir, signal_ch)


def default_sample_stack_volume(sample_dir: str | Path, signal_ch: str = "ch1") -> Path:
    return heatmap_3d_volume_tiff(sample_dir, signal_ch)


def default_sample_stack_colorbar(sample_dir: str | Path, signal_ch: str = "ch1") -> Path:
    return heatmap_3d_colorbar_png(sample_dir, signal_ch)


def default_sample_heatmap_3d_png(sample_dir: str | Path, signal_ch: str = "ch1") -> Path:
    return heatmap_3d_png(sample_dir, signal_ch)


def atlas_bin_volume_mm3(target_resolution_xyz: tuple[float, float, float]) -> float:
    res_x, res_y, res_z = target_resolution_xyz
    return float(res_x / 1000.0) * float(res_y / 1000.0) * float(res_z / 1000.0)


def counts_to_density_volume(
    atlas_volume: np.ndarray,
    *,
    target_resolution_xyz: tuple[float, float, float],
    volume_mode: str,
) -> np.ndarray:
    volume = np.asarray(atlas_volume, dtype=np.float32)
    if volume_mode != "count":
        raise ValueError(
            "Signal density in count/volume requires --volume-mode count. "
            f"Got volume_mode={volume_mode!r}."
        )
    bin_volume_mm3 = atlas_bin_volume_mm3(target_resolution_xyz)
    if bin_volume_mm3 <= 0:
        raise ValueError(f"Invalid atlas bin volume: {bin_volume_mm3} mm³")
    return volume / bin_volume_mm3


def mean_voxels_per_cell(
    resolution_xyz_um: tuple[float, float, float],
    mean_cell_volume_um3: float,
) -> float:
    res_x, res_y, res_z = resolution_xyz_um
    native_voxel_um3 = float(res_x) * float(res_y) * float(res_z)
    if native_voxel_um3 <= 0:
        raise ValueError(f"Invalid native voxel volume from resolution_xyz_um={resolution_xyz_um}")
    if mean_cell_volume_um3 <= 0:
        raise ValueError(f"mean_cell_volume_um3 must be > 0, got {mean_cell_volume_um3}")
    return float(mean_cell_volume_um3) / native_voxel_um3


def voxel_density_to_cell_density(
    voxel_density: np.ndarray,
    *,
    resolution_xyz_um: tuple[float, float, float],
    mean_cell_volume_um3: float,
) -> np.ndarray:
    """Convert foreground voxel density (voxels/mm³) to cell density (cells/mm³)."""
    voxels_per_cell = mean_voxels_per_cell(resolution_xyz_um, mean_cell_volume_um3)
    return np.asarray(voxel_density, dtype=np.float32) / voxels_per_cell


def linspace_bregma_coords(start_mm: float, end_mm: float, count: int) -> list[float]:
    if count < 1:
        raise ValueError("slice count must be >= 1")
    if count == 1:
        return [float(start_mm)]
    values = np.linspace(float(start_mm), float(end_mm), int(count), dtype=np.float64)
    return [float(round(value, 4)) for value in values]


def sample_has_batch_inputs(sample_dir: str | Path, signal_ch: str = "ch1") -> bool:
    sample_dir = Path(sample_dir)
    if not sample_dir.is_dir():
        return False
    if any(sample_dir.glob("*_mask.zarr")):
        return True
    for candidate in legacy_heatmap_3d_volume_candidates(sample_dir, signal_ch):
        if candidate.exists():
            return True
    return default_sample_stack_volume(sample_dir, signal_ch).exists()


def default_sample_density_excel(sample_dir: str | Path, signal_ch: str = "ch1") -> Path:
    return brain_distribution_stats_xlsx(sample_dir, signal_ch)


def default_region_cfg_path() -> Path:
    return project_root() / "pipeline_modules" / "registration" / "Region_Csv_Rev1_updated.CSV"


def resolve_density_excel_path(
    sample_dir: str | Path,
    input_excel: str | Path | None = None,
    signal_ch: str = "ch1",
) -> Path:
    if input_excel:
        path = Path(input_excel)
        if not path.exists():
            raise FileNotFoundError(f"Density Excel not found: {path}")
        return path
    for candidate in legacy_brain_distribution_candidates(sample_dir, signal_ch):
        if candidate.exists():
            return candidate
    default_path = default_sample_density_excel(sample_dir, signal_ch)
    raise FileNotFoundError(
        f"No density Excel found for sample_dir={sample_dir}. "
        f"Expected {default_path} or a legacy *density*.xlsx workbook"
    )


def sample_has_density_excel(sample_dir: str | Path, signal_ch: str = "ch1") -> bool:
    sample_dir = Path(sample_dir)
    if default_sample_density_excel(sample_dir, signal_ch).exists():
        return True
    return any(path.exists() for path in legacy_brain_distribution_candidates(sample_dir, signal_ch)[1:])


def discover_sample_dirs(samples_root: str | Path, *, require_volume: bool = True) -> list[Path]:
    samples_root = Path(samples_root)
    if not samples_root.is_dir():
        raise NotADirectoryError(f"samples_root is not a directory: {samples_root}")

    def _matches(child: Path) -> bool:
        if sample_has_batch_inputs(child):
            return True
        if not require_volume and sample_has_density_excel(child):
            return True
        return False

    discovered = [child for child in sorted(samples_root.iterdir()) if child.is_dir() and _matches(child)]
    if not discovered:
        if require_volume:
            raise FileNotFoundError(
                "No sample directories with *_mask.zarr or visualization/*_heatmap_3d_volume.tiff "
                f"found under: {samples_root}"
            )
        raise FileNotFoundError(
            "No sample directories with density Excel workbooks found under: "
            f"{samples_root}"
        )
    return discovered


def _paint_region_values_on_slice(label_slice: np.ndarray, region_values: dict[int, float]) -> np.ndarray:
    painted = np.full(label_slice.shape, np.nan, dtype=np.float32)
    labels = np.asarray(label_slice)
    for region_id, value in region_values.items():
        mask = labels == int(region_id)
        if np.any(mask):
            painted[mask] = float(value)
    inside_brain = labels > 0
    painted[inside_brain & np.isnan(painted)] = 0.0
    return painted


def prepare_smoothed_voxel_density_volume(
    atlas_count_volume: np.ndarray,
    atlas_mask: np.ndarray,
    *,
    target_resolution_xyz: tuple[float, float, float],
    volume_mode: str,
    sigma: float,
    alpha: float,
) -> np.ndarray:
    density_input = counts_to_density_volume(
        np.asarray(atlas_count_volume, dtype=np.float32),
        target_resolution_xyz=target_resolution_xyz,
        volume_mode=volume_mode,
    )
    local_signal = build_local_signal_volume(
        density_input,
        sigma=float(sigma),
        alpha=float(alpha),
        atlas_mask=atlas_mask,
        normalize=False,
    )
    return local_signal / max(float(alpha), 1e-12)


def compute_shared_density_vmax(
    density_volumes: list[np.ndarray],
    atlas_mask: np.ndarray,
    *,
    density_vmin: float = 0.0,
    percentile: float = 99.5,
    explicit_vmax: float | None = None,
) -> float:
    if explicit_vmax is not None:
        return float(explicit_vmax)
    brain = np.asarray(atlas_mask) > 0
    max_value = float(density_vmin)
    for volume in density_volumes:
        values = np.asarray(volume, dtype=np.float32)[brain]
        positive = values[values > float(density_vmin)]
        if positive.size:
            max_value = max(max_value, float(np.percentile(positive, float(percentile))))
    if max_value <= float(density_vmin):
        max_value = float(density_vmin) + 1e-6
    return max_value


def default_cell_density_slice_dir(sample_dir: str | Path) -> Path:
    return Path(sample_dir) / "visualization" / "cell_density_slices"


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
        "output": default_sample_stack_output(sample_dir, signal_ch),
        "output_volume": default_sample_stack_volume(sample_dir, signal_ch),
        "output_png": default_sample_heatmap_3d_png(sample_dir, signal_ch),
        "points_csv": default_sample_points_csv(sample_dir),
    }


def resolve_slice_output_dir(
    sample_dir: str | Path,
    *,
    config_path: str | Path | None = None,
    output_subdir: str = "",
    signal_ch: str | None = None,
) -> Path:
    if output_subdir:
        return Path(sample_dir) / "visualization" / output_subdir
    if signal_ch is None:
        defaults = resolve_sample_stack_defaults(sample_dir, config_path=config_path)
        signal_ch = str(defaults["signal_ch"])
    return heatmap_2d_dir(sample_dir, signal_ch)


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


def _matplotlib_cmap_from_heatmap_lut():
    from matplotlib.colors import ListedColormap

    lut = _build_heatmap_lut() / 255.0
    return ListedColormap(lut)


def _resolve_density_range(
    density_signal: np.ndarray,
    atlas_mask: np.ndarray,
    *,
    vmin: float | None = None,
    vmax: float | None = None,
) -> tuple[float, float]:
    brain = np.asarray(atlas_mask) > 0
    values = np.asarray(density_signal, dtype=np.float32)[brain]
    positive = values[values > 0]
    resolved_min = 0.0 if vmin is None else float(vmin)
    if vmax is not None:
        resolved_max = float(vmax)
    elif positive.size:
        resolved_max = float(np.percentile(positive, 99.5))
    else:
        resolved_max = float(values.max()) if values.size else 1.0
    if resolved_max <= resolved_min:
        resolved_max = resolved_min + 1e-6
    return resolved_min, resolved_max


def save_density_colorbar(
    output_path: str | Path,
    *,
    vmin: float,
    vmax: float,
    unit_label: str = "count/mm³",
    dpi: int = 150,
) -> Path:
    from matplotlib.colors import Normalize

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(1.35, 4.8), dpi=dpi)
    fig.patch.set_facecolor("white")
    cmap = _matplotlib_cmap_from_heatmap_lut()
    norm = Normalize(vmin=vmin, vmax=vmax)
    colorbar = fig.colorbar(
        matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap),
        cax=ax,
        orientation="vertical",
    )
    colorbar.set_label(unit_label, fontsize=11, labelpad=10)
    colorbar.ax.tick_params(labelsize=9, length=4, width=0.8)
    colorbar.set_ticks([vmin, vmax])
    colorbar.set_ticklabels([f"{vmin:.4g}", f"{vmax:.4g}"])
    fig.tight_layout(pad=0.2)
    fig.savefig(output_path, dpi=dpi, facecolor="white", bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    return output_path


def _load_cached_volume_mode(cached_volume_path: Path) -> str | None:
    meta_path = cached_volume_path.with_suffix(".json")
    if not meta_path.exists():
        return None
    try:
        return str(load_json(meta_path).get("volume_mode", "")).strip() or None
    except (OSError, json.JSONDecodeError):
        return None


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
    target_resolution_xyz=None,
    volume_mode="binary",
    save_colorbar=True,
    colorbar_output=None,
    colorbar_unit="count/mm³",
    density_vmin=None,
    density_vmax=None,
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
        target_resolution_xyz=target_resolution_xyz,
        volume_mode=volume_mode,
        save_colorbar=save_colorbar,
        colorbar_output=colorbar_output,
        colorbar_unit=colorbar_unit,
        density_vmin=density_vmin,
        density_vmax=density_vmax,
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
    target_resolution_xyz: tuple[float, float, float] | None = None,
    volume_mode: str = "binary",
    save_colorbar: bool = True,
    colorbar_output: str | Path | None = None,
    colorbar_unit: str = "count/mm³",
    density_vmin: float | None = None,
    density_vmax: float | None = None,
    preview_png_path: str | Path | None = None,
) -> dict[str, float | str | None]:
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
    if target_resolution_xyz is None:
        density_input = np.asarray(img, dtype=np.float32)
    else:
        density_input = counts_to_density_volume(
            img,
            target_resolution_xyz=target_resolution_xyz,
            volume_mode=volume_mode,
        )
        bin_volume_mm3 = atlas_bin_volume_mm3(target_resolution_xyz)
        print(
            f"Converted atlas counts to signal density using bin volume "
            f"{bin_volume_mm3:.6g} mm³ ({colorbar_unit})"
        )

    local_signal = build_local_signal_volume(density_input, sigma=sigma, alpha=alpha, atlas_mask=atlas_mask)
    density_signal = local_signal / max(float(alpha), 1e-12)
    resolved_density_min, resolved_density_max = _resolve_density_range(
        density_signal,
        atlas_mask,
        vmin=density_vmin,
        vmax=density_vmax if density_vmax is not None else vmax,
    )
    print(
        "Smoothed signal density range inside atlas mask: "
        f"min={resolved_density_min:.6g}, max={resolved_density_max:.6g} {colorbar_unit}"
    )

    display_signal = local_signal
    max_signal = float(local_signal.max())
    print(f"Local signal range before display scaling: min={float(local_signal.min()):.6g}, max={max_signal:.6g}")
    if normalize:
        display_vmax = float(vmax) if vmax is not None else max_signal
        if display_vmax > 0:
            display_signal = np.clip(local_signal / display_vmax, 0, 1) * float(scale_max)
            print(f"Normalized local signal for display: vmax={display_vmax:.6g}, scale_max={float(scale_max):.6g}")
        else:
            print("Warning: local signal max is 0; skipping normalization.")
    heatimg = _legacy_rgb_heat_volume(display_signal, edge, atlas_mask)

    print(f"Saving heatmap stack to: {save_path}")
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(save_path, heatimg, compression="lzw")

    preview_saved = None
    if preview_png_path:
        preview_saved = Path(preview_png_path)
        preview_saved.parent.mkdir(parents=True, exist_ok=True)
        preview_image = np.max(np.asarray(heatimg), axis=0)
        if preview_image.dtype != np.uint8:
            preview_image = np.clip(preview_image, 0, 255).astype(np.uint8)
        Image.fromarray(preview_image).save(preview_saved)
        print(f"Saved 3D heatmap preview PNG to: {preview_saved}")

    colorbar_path = None
    if save_colorbar:
        colorbar_path = Path(colorbar_output) if colorbar_output else save_path.with_name(f"{save_path.stem}_colorbar.png")
        save_density_colorbar(
            colorbar_path,
            vmin=resolved_density_min,
            vmax=resolved_density_max,
            unit_label=colorbar_unit,
        )
        print(f"Saved density colorbar to: {colorbar_path}")

    print("Done!")
    return {
        "density_min": resolved_density_min,
        "density_max": resolved_density_max,
        "density_unit": colorbar_unit,
        "colorbar_output": str(colorbar_path) if colorbar_path else None,
        "preview_png_output": str(preview_saved) if preview_saved else None,
    }


def ensure_atlas_count_volume(
    *,
    sample_dir: str | Path,
    signal_ch: str,
    register_ch: str,
    mask_zarr: str | Path | None,
    dataset_name: str,
    sample_reference_nii: str | Path | None,
    atlas_image: str | Path,
    transforms_dir: str | Path | None,
    transforms: str,
    resolution_xyz: str | tuple[float, float, float],
    target_resolution_xyz: str | tuple[float, float, float],
    foreground_mode: str,
    foreground_label: int,
    block_shape: str,
    min_voxels_per_point: int,
    volume_mode: str,
    output_volume: str | Path | None = None,
) -> tuple[np.ndarray, Path, dict[str, object]]:
    if ants is None:
        raise ImportError("ANTsPy is required to build atlas-space volumes.")

    from pipeline_modules.visualization.warp_mask_zarr_to_atlas_points import (
        accumulate_sample_grid,
        parse_block_shape,
        parse_triplet,
        resolve_inverse_transforms,
        resolve_mask_zarr,
        resolve_sample_reference_nii,
        warp_sample_grid_to_atlas,
        write_volume_output,
    )
    from pipeline_modules.segmentation.zarr_utils import open_zarr_dataset

    sample_dir = Path(sample_dir)
    sample_dir_value = sample_dir if sample_dir else None
    cached_volume_path = Path(output_volume) if output_volume else default_sample_stack_volume(sample_dir, signal_ch)
    if isinstance(target_resolution_xyz, tuple):
        target_resolution = tuple(float(value) for value in target_resolution_xyz)
    else:
        target_resolution = parse_triplet(target_resolution_xyz, name="target_resolution_xyz")
    if isinstance(resolution_xyz, tuple):
        resolution = tuple(float(value) for value in resolution_xyz)
    else:
        resolution = parse_triplet(resolution_xyz, name="resolution_xyz")

    summary: dict[str, object] = {
        "cached_volume_path": str(cached_volume_path),
        "volume_mode": volume_mode,
        "target_resolution_xyz": list(target_resolution),
        "resolution_xyz": list(resolution),
        "atlas_image": str(atlas_image),
    }

    cached_mode = _load_cached_volume_mode(cached_volume_path)
    use_cached_volume = cached_volume_path.exists() and (cached_mode is None or cached_mode == volume_mode)
    if cached_volume_path.exists() and cached_mode is not None and cached_mode != volume_mode:
        print(
            f"Cached atlas volume uses volume_mode={cached_mode!r}; "
            f"recomputing with requested volume_mode={volume_mode!r}."
        )

    if use_cached_volume:
        print(f"Using cached atlas-space volume: {cached_volume_path}")
        atlas_volume = read_tiff_stack(cached_volume_path)
        summary["cache_hit"] = True
        summary["output_volume"] = str(cached_volume_path)
        return atlas_volume, cached_volume_path, summary

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

    summary.update(
        {
            "sample_reference_nii": str(sample_reference_path),
            "transformlist": transformlist,
        }
    )

    sample_ref = ants.image_read(str(sample_reference_path))
    sample_shape_zyx = tuple(int(value) for value in sample_ref.shape[::-1])
    arr = open_zarr_dataset(mask_zarr_path, dataset_name=dataset_name)
    fallback_block_shape = tuple(int(value) for value in (getattr(arr, "chunks", None) or arr.shape))
    resolved_block_shape = parse_block_shape(block_shape, fallback_block_shape)

    print("Binning sample mask into atlas registration grid")
    sample_volume, bin_summary = accumulate_sample_grid(
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
    summary.update(bin_summary)
    print("Warping binned volume into atlas space")
    atlas_volume, raw_atlas_spacing_xyz = warp_sample_grid_to_atlas(
        sample_volume,
        sample_reference_nii=sample_reference_path,
        atlas_image=atlas_image,
        transformlist=transformlist,
        interpolator="linear" if volume_mode == "count" else "nearestNeighbor",
        binarize=volume_mode == "binary",
    )
    summary["raw_atlas_image_spacing_xyz"] = list(raw_atlas_spacing_xyz)

    print(f"Writing cached atlas-space volume to: {cached_volume_path}")
    cached_volume_path.parent.mkdir(parents=True, exist_ok=True)
    write_volume_output(atlas_volume, cached_volume_path)
    volume_meta_path = cached_volume_path.with_suffix(".json")
    with volume_meta_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "volume_mode": volume_mode,
                "target_resolution_xyz": list(target_resolution),
                "output_volume": str(cached_volume_path),
            },
            handle,
            indent=2,
            ensure_ascii=False,
        )
    summary["cache_hit"] = False
    summary["output_volume"] = str(cached_volume_path)
    summary["volume_meta"] = str(volume_meta_path)
    return atlas_volume, cached_volume_path, summary


def _report_missing_region_metric_data(
    *,
    sample_label: str,
    missing_region_ids: set[int],
    region_name_by_id: dict[int, str],
    region_metric: str,
) -> None:
    if not missing_region_ids:
        return
    formatted = [
        f"{region_id} ({region_name_by_id.get(region_id, 'unknown')})"
        for region_id in sorted(missing_region_ids)
    ]
    print(
        f"Sample '{sample_label}' has no {region_metric} data for "
        f"{len(missing_region_ids)} brain region(s) on rendered slices; using 0: "
        + ", ".join(formatted)
    )


def _accumulate_missing_regions_for_slices(
    *,
    label_path: Path,
    bregma_coords: list[float],
    plane: str,
    coord_system: str,
    atlas_resolution_um: float,
    bregma_index: tuple[int, int, int],
    value_by_region_id: dict[int, float],
    path_by_region_id: dict[int, list[int]],
) -> set[int]:
    missing: set[int] = set()
    for ap_mm in bregma_coords:
        spec = AtlasSliceSpec(
            plane=plane,
            coordinate_system=coord_system,
            coordinate=ap_mm,
            atlas_resolution_um=atlas_resolution_um,
            bregma_index=bregma_index,
        )
        atlas_slice = extract_atlas_slice(label_path, spec)
        missing.update(
            collect_regions_missing_metric_data(
                atlas_slice.image,
                value_by_region_id,
                path_by_region_id,
            )
        )
    return missing


def _resolve_subtract_diff_percentile(cli_value: float) -> float:
    """Signal-count diff maps default to P95 symmetric limits."""
    if cli_value == BATCH_SLICE_DEFAULT_PERCENTILE:
        return SUBTRACT_DIFF_DEFAULT_PERCENTILE
    return float(cli_value)


def _resolve_shared_metric_limits(
    metric_values: list[float],
    *,
    density_vmin: float | None,
    density_vmax: float | None,
    density_percentile: float,
    symmetric: bool = False,
) -> tuple[float, float]:
    if symmetric:
        explicit = density_vmax if density_vmax is not None else None
        return compute_symmetric_metric_limits(
            metric_values,
            percentile=density_percentile,
            explicit_vmax=explicit,
        )
    vmin = float(density_vmin or 0.0)
    if density_vmax is not None:
        vmax = float(density_vmax)
    else:
        finite = [float(value) for value in metric_values if np.isfinite(value) and value > vmin]
        vmax = float(np.percentile(finite, float(density_percentile))) if finite else vmin + 1e-6
    if vmax <= vmin:
        vmax = vmin + 1e-6
    return vmin, vmax


def generate_batch_region_metric_slices(
    *,
    samples_root: str | Path,
    region_metric: str = "Signal Count",
    region_cfg_path: str | Path | None = None,
    label_path: str | Path | None = None,
    bregma_start: float = 1.1,
    bregma_end: float = -5.2,
    slice_count: int = 12,
    plane: str = "coronal",
    coord_system: str = "bregma-mm",
    atlas_resolution_um: float = 25.0,
    bregma_index: tuple[int, int, int] = (18, 216, 228),
    density_vmin: float | None = None,
    density_vmax: float | None = None,
    density_percentile: float = 99.5,
    cmap_name: str = "white_orange_red_black",
    dpi: int = 300,
    line_width: float = 0.16,
    brain_outline_width: float = 0.42,
    show_region_contours: bool = True,
    colorbar_label: str | None = None,
    output_subdir: str = "",
) -> dict[str, object]:
    sample_dirs = discover_sample_dirs(samples_root, require_volume=False)
    label_path = Path(label_path or DEFAULT_ATLAS_LABEL)
    region_cfg_path = Path(region_cfg_path or default_region_cfg_path())
    bregma_coords = linspace_bregma_coords(bregma_start, bregma_end, slice_count)
    region_name_by_id = build_region_name_lookup(region_cfg_path)

    lookups: dict[str, tuple[dict[int, float], dict[int, list[int]]]] = {}
    all_values: list[float] = []
    for sample_dir in sample_dirs:
        excel_path = resolve_density_excel_path(sample_dir)
        value_by_region_id, path_by_region_id = build_region_metric_lookup(
            excel_path,
            cfg_path=region_cfg_path,
            metric=region_metric,
        )
        lookups[str(sample_dir)] = (value_by_region_id, path_by_region_id)
        all_values.extend(float(value) for value in value_by_region_id.values())
        missing_regions = _accumulate_missing_regions_for_slices(
            label_path=label_path,
            bregma_coords=bregma_coords,
            plane=plane,
            coord_system=coord_system,
            atlas_resolution_um=atlas_resolution_um,
            bregma_index=bregma_index,
            value_by_region_id=value_by_region_id,
            path_by_region_id=path_by_region_id,
        )
        _report_missing_region_metric_data(
            sample_label=sample_dir.name,
            missing_region_ids=missing_regions,
            region_name_by_id=region_name_by_id,
            region_metric=region_metric,
        )

    shared_vmin, shared_vmax = _resolve_shared_metric_limits(
        all_values,
        density_vmin=density_vmin,
        density_vmax=density_vmax,
        density_percentile=density_percentile,
        symmetric=False,
    )
    resolved_colorbar_label = colorbar_label or region_metric
    print(
        f"Shared region metric color scale ({region_metric}): "
        f"min={shared_vmin:g}, max={shared_vmax:g} (samples={len(sample_dirs)})"
    )

    outputs_by_sample: dict[str, list[str]] = {}
    for sample_dir in sample_dirs:
        value_by_region_id, path_by_region_id = lookups[str(sample_dir)]
        defaults = resolve_sample_stack_defaults(sample_dir)
        out_dir = resolve_slice_output_dir(
            sample_dir,
            output_subdir=output_subdir,
            signal_ch=str(defaults["signal_ch"]),
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        sample_outputs: list[str] = []
        for ap_mm in bregma_coords:
            spec = AtlasSliceSpec(
                plane=plane,
                coordinate_system=coord_system,
                coordinate=ap_mm,
                atlas_resolution_um=atlas_resolution_um,
                bregma_index=bregma_index,
            )
            atlas_slice = extract_atlas_slice(label_path, spec)
            slice_region_values = resolve_slice_region_values(
                atlas_slice.image,
                value_by_region_id,
                path_by_region_id,
            )
            output_path = out_dir / f"bregma_{ap_mm}mm.png"
            render_region_metric_atlas_slice(
                label_path,
                spec,
                slice_region_values,
                output_path,
                cmap_name=cmap_name,
                vmin=float(shared_vmin),
                vmax=float(shared_vmax),
                dpi=int(dpi),
                line_width=float(line_width),
                brain_outline_width=float(brain_outline_width),
                show_region_contours=show_region_contours,
                colorbar_label=resolved_colorbar_label,
            )
            sample_outputs.append(str(output_path))
        outputs_by_sample[str(sample_dir)] = sample_outputs
        print(f"Wrote {len(sample_outputs)} region metric slices to: {out_dir}")

    summary_path = Path(samples_root) / "visualization" / "batch_region_metric_slices.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "mode": "batch-cell-density-slices",
        "slice_color_mode": "region",
        "samples_root": str(samples_root),
        "sample_dirs": [str(path) for path in sample_dirs],
        "region_metric": region_metric,
        "region_cfg_path": str(region_cfg_path),
        "bregma_coords_mm": bregma_coords,
        "shared_metric_vmin": float(shared_vmin),
        "shared_metric_vmax": float(shared_vmax),
        "cmap_name": cmap_name,
        "colorbar_label": resolved_colorbar_label,
        "show_region_contours": bool(show_region_contours),
        "outputs_by_sample": outputs_by_sample,
    }
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    payload["summary_json"] = str(summary_path)
    return payload


def generate_batch_signal_count_diff_slices(
    *,
    sample_dir: str | Path,
    subtract_sample_dir: str | Path,
    region_metric: str = "Signal Count",
    region_cfg_path: str | Path | None = None,
    input_excel: str | Path | None = None,
    subtract_input_excel: str | Path | None = None,
    label_path: str | Path | None = None,
    bregma_start: float = 1.1,
    bregma_end: float = -5.2,
    slice_count: int = 12,
    plane: str = "coronal",
    coord_system: str = "bregma-mm",
    atlas_resolution_um: float = 25.0,
    bregma_index: tuple[int, int, int] = (18, 216, 228),
    density_vmin: float | None = None,
    density_vmax: float | None = None,
    density_percentile: float = SUBTRACT_DIFF_DEFAULT_PERCENTILE,
    cmap_name: str = "signal_count_diff",
    dpi: int = 300,
    line_width: float = 0.16,
    brain_outline_width: float = 0.42,
    show_region_contours: bool = True,
    colorbar_label: str | None = None,
    output_subdir: str = "",
) -> dict[str, object]:
    sample_dir = Path(sample_dir)
    subtract_sample_dir = Path(subtract_sample_dir)
    label_path = Path(label_path or DEFAULT_ATLAS_LABEL)
    region_cfg_path = Path(region_cfg_path or default_region_cfg_path())
    bregma_coords = linspace_bregma_coords(bregma_start, bregma_end, slice_count)

    excel_a = resolve_density_excel_path(sample_dir, input_excel)
    excel_b = resolve_density_excel_path(subtract_sample_dir, subtract_input_excel)
    lookup_a, path_by_region_id = build_region_metric_lookup(excel_a, cfg_path=region_cfg_path, metric=region_metric)
    lookup_b, path_by_region_id_b = build_region_metric_lookup(excel_b, cfg_path=region_cfg_path, metric=region_metric)
    diff_lookup = subtract_region_metric_values(lookup_a, lookup_b)
    region_name_by_id = build_region_name_lookup(region_cfg_path)
    missing_a = _accumulate_missing_regions_for_slices(
        label_path=label_path,
        bregma_coords=bregma_coords,
        plane=plane,
        coord_system=coord_system,
        atlas_resolution_um=atlas_resolution_um,
        bregma_index=bregma_index,
        value_by_region_id=lookup_a,
        path_by_region_id=path_by_region_id,
    )
    missing_b = _accumulate_missing_regions_for_slices(
        label_path=label_path,
        bregma_coords=bregma_coords,
        plane=plane,
        coord_system=coord_system,
        atlas_resolution_um=atlas_resolution_um,
        bregma_index=bregma_index,
        value_by_region_id=lookup_b,
        path_by_region_id=path_by_region_id_b,
    )
    _report_missing_region_metric_data(
        sample_label=sample_dir.name,
        missing_region_ids=missing_a,
        region_name_by_id=region_name_by_id,
        region_metric=region_metric,
    )
    _report_missing_region_metric_data(
        sample_label=subtract_sample_dir.name,
        missing_region_ids=missing_b,
        region_name_by_id=region_name_by_id,
        region_metric=region_metric,
    )

    shared_vmin, shared_vmax = _resolve_shared_metric_limits(
        list(diff_lookup.values()),
        density_vmin=density_vmin,
        density_vmax=density_vmax,
        density_percentile=density_percentile,
        symmetric=True,
    )
    resolved_colorbar_label = colorbar_label or f"{region_metric} diff ({sample_dir.name} - {subtract_sample_dir.name})"
    print(
        f"Shared signal-count diff color scale: min={shared_vmin:g}, max={shared_vmax:g} "
        f"({sample_dir.name} minus {subtract_sample_dir.name})"
    )

    out_dir = resolve_slice_output_dir(
        sample_dir,
        output_subdir=output_subdir,
        signal_ch=str(resolve_sample_stack_defaults(sample_dir)["signal_ch"]),
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[str] = []
    for ap_mm in bregma_coords:
        spec = AtlasSliceSpec(
            plane=plane,
            coordinate_system=coord_system,
            coordinate=ap_mm,
            atlas_resolution_um=atlas_resolution_um,
            bregma_index=bregma_index,
        )
        atlas_slice = extract_atlas_slice(label_path, spec)
        slice_region_values = resolve_slice_region_values(
            atlas_slice.image,
            diff_lookup,
            path_by_region_id,
        )
        output_path = out_dir / f"bregma_{ap_mm}mm_{sample_dir.name}_minus_{subtract_sample_dir.name}.png"
        render_region_metric_atlas_slice(
            label_path,
            spec,
            slice_region_values,
            output_path,
            cmap_name=cmap_name,
            vmin=float(shared_vmin),
            vmax=float(shared_vmax),
            dpi=int(dpi),
            line_width=float(line_width),
            brain_outline_width=float(brain_outline_width),
            show_region_contours=show_region_contours,
            colorbar_label=resolved_colorbar_label,
        )
        outputs.append(str(output_path))
    print(f"Wrote {len(outputs)} signal-count diff slices to: {out_dir}")

    summary_path = out_dir / f"{sample_dir.name}_minus_{subtract_sample_dir.name}_summary.json"
    payload = {
        "mode": "batch-cell-density-slices",
        "slice_color_mode": "region",
        "subtract_mode": True,
        "sample_dir": str(sample_dir),
        "subtract_sample_dir": str(subtract_sample_dir),
        "input_excel": str(excel_a),
        "subtract_input_excel": str(excel_b),
        "region_metric": region_metric,
        "region_cfg_path": str(region_cfg_path),
        "bregma_coords_mm": bregma_coords,
        "shared_metric_vmin": float(shared_vmin),
        "shared_metric_vmax": float(shared_vmax),
        "cmap_name": cmap_name,
        "colorbar_label": resolved_colorbar_label,
        "show_region_contours": bool(show_region_contours),
        "outputs": outputs,
    }
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    payload["summary_json"] = str(summary_path)
    return payload


def generate_batch_cell_density_slices(
    *,
    samples_root: str | Path,
    mean_cell_volume_um3: float,
    config_path: str | Path | None = None,
    label_path: str | Path | None = None,
    atlas_image: str | Path | None = None,
    bregma_start: float = 1.1,
    bregma_end: float = -5.2,
    slice_count: int = 12,
    plane: str = "coronal",
    coord_system: str = "bregma-mm",
    atlas_resolution_um: float = 25.0,
    bregma_index: tuple[int, int, int] = (18, 216, 228),
    sigma: float = 10.0,
    alpha: float = 2.0,
    volume_mode: str = "count",
    density_vmin: float = 0.0,
    density_vmax: float | None = None,
    density_percentile: float = 99.5,
    cmap_name: str = "white_blue_red",
    dpi: int = 300,
    line_width: float = 0.16,
    brain_outline_width: float = 0.42,
    show_region_contours: bool = True,
    colorbar_label: str = "cell density (cells/mm³)",
    output_subdir: str = "",
    foreground_mode: str = "equal",
    foreground_label: int = 1,
    block_shape: str = "",
    min_voxels_per_point: int = 1,
    dataset_name: str = "0",
) -> dict[str, object]:
    sample_dirs = discover_sample_dirs(samples_root)
    label_path = Path(label_path or DEFAULT_ATLAS_LABEL)
    labels = np.asarray(tifffile.imread(str(label_path)))
    atlas_image_path = Path(atlas_image or default_reference_dir() / "atlas.nii.gz")
    bregma_coords = linspace_bregma_coords(bregma_start, bregma_end, slice_count)

    cell_density_volumes: dict[str, np.ndarray] = {}
    sample_summaries: dict[str, object] = {}
    for sample_dir in sample_dirs:
        defaults = resolve_sample_stack_defaults(sample_dir, config_path=config_path)
        resolution_xyz = tuple(float(value) for value in defaults["resolution_xyz"])  # type: ignore[arg-type]
        target_resolution_xyz = tuple(float(value) for value in defaults["target_resolution_xyz"])  # type: ignore[arg-type]
        atlas_volume, cached_volume_path, volume_summary = ensure_atlas_count_volume(
            sample_dir=sample_dir,
            signal_ch=str(defaults["signal_ch"]),
            register_ch=str(defaults["register_ch"]),
            mask_zarr=defaults["mask_zarr"],
            dataset_name=dataset_name,
            sample_reference_nii=defaults["sample_reference_nii"],
            atlas_image=atlas_image_path,
            transforms_dir=defaults["transforms_dir"],
            transforms="",
            resolution_xyz=resolution_xyz,
            target_resolution_xyz=target_resolution_xyz,
            foreground_mode=foreground_mode,
            foreground_label=foreground_label,
            block_shape=block_shape,
            min_voxels_per_point=min_voxels_per_point,
            volume_mode=volume_mode,
        )
        voxel_density = prepare_smoothed_voxel_density_volume(
            atlas_volume,
            labels,
            target_resolution_xyz=target_resolution_xyz,
            volume_mode=volume_mode,
            sigma=sigma,
            alpha=alpha,
        )
        cell_density = voxel_density_to_cell_density(
            voxel_density,
            resolution_xyz_um=resolution_xyz,
            mean_cell_volume_um3=mean_cell_volume_um3,
        )
        cell_density_volumes[str(sample_dir)] = cell_density
        sample_summaries[str(sample_dir)] = {
            "cached_volume": str(cached_volume_path),
            "resolution_xyz": list(resolution_xyz),
            "mean_cell_volume_um3": float(mean_cell_volume_um3),
            "voxels_per_cell": mean_voxels_per_cell(resolution_xyz, mean_cell_volume_um3),
            **volume_summary,
        }

    shared_vmax = compute_shared_density_vmax(
        list(cell_density_volumes.values()),
        labels,
        density_vmin=density_vmin,
        percentile=density_percentile,
        explicit_vmax=density_vmax,
    )
    print(
        "Shared cell density color scale: "
        f"min={density_vmin:g}, max={shared_vmax:g} cells/mm³ "
        f"(percentile={density_percentile}, samples={len(sample_dirs)})"
    )

    outputs_by_sample: dict[str, list[str]] = {}
    for sample_dir in sample_dirs:
        cell_density = cell_density_volumes[str(sample_dir)]
        out_dir = resolve_slice_output_dir(
            sample_dir,
            config_path=config_path,
            output_subdir=output_subdir,
            signal_ch=str(defaults["signal_ch"]),
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        sample_outputs: list[str] = []
        for ap_mm in bregma_coords:
            output_path = out_dir / f"bregma_{ap_mm}mm.png"
            spec = AtlasSliceSpec(
                plane=plane,
                coordinate_system=coord_system,
                coordinate=ap_mm,
                atlas_resolution_um=atlas_resolution_um,
                bregma_index=bregma_index,
            )
            render_local_signal_atlas_slice(
                cell_density,
                label_path,
                spec,
                output_path,
                cmap_name=cmap_name,
                vmin=float(density_vmin),
                vmax=float(shared_vmax),
                dpi=int(dpi),
                line_width=float(line_width),
                brain_outline_width=float(brain_outline_width),
                show_region_contours=show_region_contours,
                colorbar_label=colorbar_label,
            )
            sample_outputs.append(str(output_path))
        outputs_by_sample[str(sample_dir)] = sample_outputs
        print(f"Wrote {len(sample_outputs)} cell density slices to: {out_dir}")

    summary_path = Path(samples_root) / "visualization" / "batch_cell_density_slices.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "mode": "batch-cell-density-slices",
        "samples_root": str(samples_root),
        "sample_dirs": [str(path) for path in sample_dirs],
        "mean_cell_volume_um3": float(mean_cell_volume_um3),
        "bregma_coords_mm": bregma_coords,
        "shared_density_vmin": float(density_vmin),
        "shared_density_vmax": float(shared_vmax),
        "density_percentile": float(density_percentile),
        "show_region_contours": bool(show_region_contours),
        "colorbar_label": colorbar_label,
        "outputs_by_sample": outputs_by_sample,
        "sample_summaries": sample_summaries,
    }
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    payload["summary_json"] = str(summary_path)
    return payload


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
    save_colorbar: bool = True,
    colorbar_output: str | Path | None = None,
    colorbar_unit: str = "count/mm³",
    density_vmin: float | None = None,
    density_vmax: float | None = None,
) -> dict[str, object]:
    if ants is None:
        raise ImportError("ANTsPy is required for --mode sample-stack.")

    from pipeline_modules.visualization.warp_mask_zarr_to_atlas_points import (
        atlas_volume_to_points,
        parse_triplet,
        resolve_atlas_resolution_xyz,
        write_outputs,
    )

    print("Stage 1/4: resolve sample inputs and atlas count volume")
    if isinstance(target_resolution_xyz, tuple):
        target_resolution = tuple(float(value) for value in target_resolution_xyz)
    else:
        target_resolution = parse_triplet(target_resolution_xyz, name="target_resolution_xyz")
    atlas_volume, cached_volume_path, volume_summary = ensure_atlas_count_volume(
        sample_dir=sample_dir,
        signal_ch=signal_ch,
        register_ch=register_ch,
        mask_zarr=mask_zarr,
        dataset_name=dataset_name,
        sample_reference_nii=sample_reference_nii,
        atlas_image=atlas_image,
        transforms_dir=transforms_dir,
        transforms=transforms,
        resolution_xyz=resolution_xyz,
        target_resolution_xyz=target_resolution,
        foreground_mode=foreground_mode,
        foreground_label=foreground_label,
        block_shape=block_shape,
        min_voxels_per_point=min_voxels_per_point,
        volume_mode=volume_mode,
        output_volume=output_volume,
    )
    points_csv_path = default_sample_points_csv(sample_dir)
    summary: dict[str, object] = {
        "success": True,
        "mode": "sample-stack",
        "cached_volume_path": str(cached_volume_path),
        "volume_mode": volume_mode,
        "target_resolution_xyz": list(target_resolution),
        "density_unit": colorbar_unit,
    }
    summary.update(volume_summary)
    raw_atlas_spacing_xyz = tuple(volume_summary.get("raw_atlas_image_spacing_xyz") or ())

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
    resolved_colorbar_output = (
        Path(colorbar_output)
        if colorbar_output
        else default_sample_stack_colorbar(sample_dir, signal_ch)
    )
    preview_png_path = default_sample_heatmap_3d_png(sample_dir, signal_ch)
    render_stats = render_heatmap_stack(
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
        target_resolution_xyz=target_resolution,
        volume_mode=volume_mode,
        save_colorbar=save_colorbar,
        colorbar_output=resolved_colorbar_output,
        colorbar_unit=colorbar_unit,
        density_vmin=density_vmin,
        density_vmax=density_vmax,
        preview_png_path=preview_png_path,
    )

    summary.update(
        {
            "atlas_shape_zyx": list(atlas_volume.shape),
            "raw_atlas_image_spacing_xyz": list(raw_atlas_spacing_xyz) if raw_atlas_spacing_xyz else [],
            "heatmap_stack_output": str(output),
            "heatmap_output": str(render_stats.get("preview_png_output") or preview_png_path),
            "density_min": render_stats["density_min"],
            "density_max": render_stats["density_max"],
            "colorbar_output": render_stats["colorbar_output"],
        }
    )
    summary_path = heatmap_3d_summary_json(sample_dir, signal_ch)
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
    show_region_contours: bool = True,
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

    region_lines = _label_contour_lines(label_slice, smoothing=1.4) if show_region_contours else []
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


def _render_region_metric_slice_array(
    label_slice: np.ndarray,
    region_values: dict[int, float],
    *,
    cmap_name: str,
    vmin: float,
    vmax: float,
    dpi: int,
    line_width: float,
    brain_outline_width: float,
    colorbar_label: str,
    show_region_contours: bool = True,
) -> np.ndarray:
    painted = _paint_region_values_on_slice(label_slice, region_values)
    height, width = label_slice.shape
    aspect = width / max(height, 1)
    long_side = 6.2
    figsize = (long_side, max(long_side / aspect, 1.0)) if aspect >= 1 else (max(long_side * aspect, 1.0), long_side)
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")

    cmap = _colormap_by_name(cmap_name).copy()
    cmap.set_bad((0, 0, 0, 0))
    masked = np.ma.masked_where(np.asarray(label_slice) <= 0, painted)
    image = ax.imshow(masked, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")

    region_lines = _label_contour_lines(label_slice, smoothing=1.4) if show_region_contours else []
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


def render_region_metric_atlas_slice(
    label_path: str | Path,
    spec: AtlasSliceSpec,
    region_values: dict[int, float],
    output_path: str | Path,
    *,
    cmap_name: str = "white_orange_red_black",
    vmin: float = 0.0,
    vmax: float | None = None,
    dpi: int = 300,
    line_width: float = 0.16,
    brain_outline_width: float = 0.42,
    show_region_contours: bool = True,
    colorbar_label: str = "Signal Count",
) -> Path:
    atlas_slice = extract_atlas_slice(label_path, spec)
    finite_values = [float(value) for value in region_values.values() if np.isfinite(value)]
    upper = float(vmax) if vmax is not None else (max(finite_values) if finite_values else float(vmin) + 1e-6)
    lower = float(vmin)
    if upper <= lower:
        upper = lower + 1e-6

    rendered = _render_region_metric_slice_array(
        atlas_slice.image,
        region_values,
        cmap_name=cmap_name,
        vmin=lower,
        vmax=upper,
        dpi=int(dpi),
        line_width=float(line_width),
        brain_outline_width=float(brain_outline_width),
        colorbar_label=colorbar_label,
        show_region_contours=show_region_contours,
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rendered).save(output_path)
    return output_path


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
    show_region_contours: bool = True,
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
        show_region_contours=show_region_contours,
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
    parser.add_argument(
        "--mode",
        choices=["stack", "sample-stack", "atlas-slice", "batch-cell-density-slices", "prv-sample"],
        default="stack",
    )
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
    parser.add_argument("--save-colorbar", dest="save_colorbar", action="store_true", default=True, help="Save a density colorbar PNG alongside stack heatmaps")
    parser.add_argument("--no-colorbar", dest="save_colorbar", action="store_false", help="Do not write a density colorbar PNG")
    parser.add_argument("--colorbar-output", default="", help="Optional colorbar PNG path")
    parser.add_argument("--colorbar-unit", default="count/mm³", help="Unit label shown on the density colorbar")
    parser.add_argument("--density-vmin", type=float, default=None, help="Minimum signal density shown on the colorbar")
    parser.add_argument("--density-vmax", type=float, default=None, help="Maximum signal density shown on the colorbar")

    parser.add_argument("--sample-dir", default="", help="Sample directory for --mode sample-stack")
    parser.add_argument(
        "--samples-root",
        default="",
        help="Parent directory containing multiple sample subdirectories for --mode batch-cell-density-slices",
    )
    parser.add_argument(
        "--mean-cell-volume-um3",
        type=float,
        default=None,
        help="Mean cell volume in µm³; converts local voxel density to cell density (cells/mm³)",
    )
    parser.add_argument("--bregma-start", type=float, default=1.1, help="Start AP coordinate in mm for batch slice mode")
    parser.add_argument("--bregma-end", type=float, default=-5.2, help="End AP coordinate in mm for batch slice mode")
    parser.add_argument("--slice-count", type=int, default=12, help="Number of evenly spaced bregma slices in batch mode")
    parser.add_argument(
        "--density-percentile",
        type=float,
        default=BATCH_SLICE_DEFAULT_PERCENTILE,
        help="Percentile for shared color scale when --density-vmax is omitted; subtract diff mode defaults to 95 unless this flag is set",
    )
    parser.add_argument(
        "--output-subdir",
        default="",
        help="Subdirectory under visualization/ for batch slice PNG outputs; defaults depend on slice-color-mode",
    )
    parser.add_argument(
        "--slice-color-mode",
        choices=["signal", "region"],
        default="signal",
        help="signal=smoothed atlas-space density slices; region=fill Allen brain areas from density Excel metrics",
    )
    parser.add_argument(
        "--subtract-sample-dir",
        default="",
        help="Subtract this sample's region metric from --sample-dir and render one diff heatmap series",
    )
    parser.add_argument(
        "--region-metric",
        default="Signal Count",
        help="Metric column from density Excel Level_* sheets used by --slice-color-mode region",
    )
    parser.add_argument(
        "--region-cfg",
        default="",
        help="Allen region CSV for mapping Excel rows to atlas region ids",
    )
    parser.add_argument(
        "--input-excel",
        default="",
        help="Optional density Excel override for --sample-dir in region or diff modes",
    )
    parser.add_argument(
        "--subtract-input-excel",
        default="",
        help="Optional density Excel override for --subtract-sample-dir",
    )
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
    parser.add_argument(
        "--hide-region-contours",
        action="store_true",
        help="Omit Allen internal region outlines on 2D atlas-slice heatmaps (outer brain outline is kept)",
    )
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
            target_resolution = None
            if args.target_resolution_xyz:
                target_resolution = _parse_triplet(args.target_resolution_xyz)
            render_stats = heatmap(
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
                target_resolution,
                args.volume_mode,
                args.save_colorbar,
                args.colorbar_output or None,
                args.colorbar_unit,
                args.density_vmin,
                args.density_vmax,
            )
            payload = {"mode": args.mode, "output": str(args.output), **render_stats}
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
                save_colorbar=args.save_colorbar,
                colorbar_output=args.colorbar_output or None,
                colorbar_unit=args.colorbar_unit,
                density_vmin=args.density_vmin,
                density_vmax=args.density_vmax,
            )
        elif args.mode == "atlas-slice":
            if not args.input or not args.output or args.coord is None:
                parser.error("--input, --output, and --coord are required for --mode atlas-slice")
            signal = read_volume(args.input, dataset_name=args.dataset_name).astype(np.float32)
            labels = np.asarray(tifffile.memmap(str(args.label)))
            target_resolution = _parse_triplet(args.target_resolution_xyz) if args.target_resolution_xyz else None
            density_input = np.asarray(signal, dtype=np.float32)
            if target_resolution is not None:
                density_input = counts_to_density_volume(
                    density_input,
                    target_resolution_xyz=target_resolution,
                    volume_mode=args.volume_mode,
                )
                local_signal = build_local_signal_volume(
                    density_input,
                    sigma=args.sigma,
                    alpha=args.alpha,
                    atlas_mask=labels,
                    normalize=False,
                )
                display_signal = local_signal / max(float(args.alpha), 1e-12)
                resolved_vmin, resolved_vmax = _resolve_density_range(
                    display_signal,
                    labels,
                    vmin=args.density_vmin if args.density_vmin is not None else args.vmin,
                    vmax=args.density_vmax if args.density_vmax is not None else args.vmax,
                )
                colorbar_label = args.colorbar_label
                if colorbar_label == "Signal Intensity":
                    colorbar_label = f"signal density ({args.colorbar_unit})"
                if args.mean_cell_volume_um3 is not None:
                    if not args.resolution_xyz:
                        parser.error("--resolution-xyz is required when --mean-cell-volume-um3 is set")
                    resolution = _parse_triplet(args.resolution_xyz)
                    display_signal = voxel_density_to_cell_density(
                        display_signal,
                        resolution_xyz_um=resolution,
                        mean_cell_volume_um3=float(args.mean_cell_volume_um3),
                    )
                    if args.colorbar_label == "Signal Intensity":
                        colorbar_label = "cell density (cells/mm³)"
                    resolved_vmin, resolved_vmax = _resolve_density_range(
                        display_signal,
                        labels,
                        vmin=args.density_vmin if args.density_vmin is not None else args.vmin,
                        vmax=args.density_vmax if args.density_vmax is not None else args.vmax,
                    )
            else:
                display_signal = build_local_signal_volume(
                    signal,
                    sigma=args.sigma,
                    alpha=args.alpha,
                    atlas_mask=labels,
                    normalize=True,
                )
                resolved_vmin = float(args.vmin)
                resolved_vmax = float(args.vmax) if args.vmax is not None else None
                colorbar_label = args.colorbar_label
            spec = AtlasSliceSpec(args.plane, args.coord_system, args.coord, args.atlas_resolution_um, parse_bregma_index(args.bregma_index))
            output_path = render_local_signal_atlas_slice(
                display_signal,
                args.label,
                spec,
                args.output,
                cmap_name=args.cmap,
                vmin=resolved_vmin,
                vmax=resolved_vmax,
                dpi=args.dpi,
                line_width=args.line_width,
                brain_outline_width=args.brain_outline_width,
                show_region_contours=not args.hide_region_contours,
                colorbar_label=colorbar_label,
            )
            payload = {
                "mode": args.mode,
                "output": str(output_path),
                "plane": args.plane,
                "coordinate": args.coord,
                "density_vmin": resolved_vmin,
                "density_vmax": resolved_vmax,
                "colorbar_label": colorbar_label,
            }
        elif args.mode == "batch-cell-density-slices":
            batch_plane = args.plane
            batch_coord_system = args.coord_system
            if batch_plane == "horizontal" and batch_coord_system == "index":
                batch_plane = "coronal"
                batch_coord_system = "bregma-mm"
            batch_kwargs = dict(
                label_path=args.label,
                bregma_start=float(args.bregma_start),
                bregma_end=float(args.bregma_end),
                slice_count=int(args.slice_count),
                plane=batch_plane,
                coord_system=batch_coord_system,
                atlas_resolution_um=float(args.atlas_resolution_um),
                bregma_index=parse_bregma_index(args.bregma_index),
                density_vmin=args.density_vmin,
                density_vmax=args.density_vmax,
                density_percentile=float(args.density_percentile),
                dpi=int(args.dpi),
                line_width=float(args.line_width),
                brain_outline_width=float(args.brain_outline_width),
                show_region_contours=not args.hide_region_contours,
                region_metric=args.region_metric,
                region_cfg_path=args.region_cfg or None,
                output_subdir=args.output_subdir or "",
            )
            if args.subtract_sample_dir:
                if not args.sample_dir:
                    parser.error("--sample-dir is required when --subtract-sample-dir is set")
                diff_cmap = args.cmap if args.cmap != "white_blue_red" else "signal_count_diff"
                diff_colorbar = (
                    args.colorbar_label
                    if args.colorbar_label != "Signal Intensity"
                    else None
                )
                subtract_kwargs = {
                    key: value
                    for key, value in batch_kwargs.items()
                    if key != "density_percentile" and value is not None
                }
                payload = generate_batch_signal_count_diff_slices(
                    sample_dir=args.sample_dir,
                    subtract_sample_dir=args.subtract_sample_dir,
                    input_excel=args.input_excel or None,
                    subtract_input_excel=args.subtract_input_excel or None,
                    cmap_name=diff_cmap,
                    colorbar_label=diff_colorbar,
                    output_subdir=args.output_subdir or "",
                    density_percentile=_resolve_subtract_diff_percentile(float(args.density_percentile)),
                    **subtract_kwargs,
                )
            elif args.slice_color_mode == "region":
                if not args.samples_root:
                    parser.error("--samples-root is required for --slice-color-mode region")
                region_colorbar = (
                    args.colorbar_label
                    if args.colorbar_label != "Signal Intensity"
                    else None
                )
                payload = generate_batch_region_metric_slices(
                    samples_root=args.samples_root,
                    cmap_name=args.cmap if args.cmap != "white_blue_red" else "white_orange_red_black",
                    colorbar_label=region_colorbar,
                    output_subdir=args.output_subdir or "",
                    **{key: value for key, value in batch_kwargs.items() if value is not None},
                )
            else:
                if not args.samples_root:
                    parser.error("--samples-root is required for --slice-color-mode signal")
                if args.mean_cell_volume_um3 is None or args.mean_cell_volume_um3 <= 0:
                    parser.error("--mean-cell-volume-um3 must be > 0 for --slice-color-mode signal")
                payload = generate_batch_cell_density_slices(
                    samples_root=args.samples_root,
                    mean_cell_volume_um3=float(args.mean_cell_volume_um3),
                    config_path=args.config or None,
                    atlas_image=args.atlas_image or None,
                    sigma=float(args.sigma),
                    alpha=float(args.alpha),
                    volume_mode=args.volume_mode,
                    cmap_name=args.cmap,
                    colorbar_label=(
                        args.colorbar_label
                        if args.colorbar_label != "Signal Intensity"
                        else "cell density (cells/mm³)"
                    ),
                    output_subdir=args.output_subdir or "",
                    foreground_mode=args.foreground_mode,
                    foreground_label=int(args.foreground_label),
                    block_shape=args.block_shape,
                    min_voxels_per_point=int(args.min_voxels_per_point),
                    dataset_name=args.dataset_name,
                    **{key: value for key, value in batch_kwargs.items() if key not in {"region_metric", "region_cfg_path"} and value is not None},
                )
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
