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


def build_local_signal_volume(signal_volume, sigma=1.5, alpha=1.0, atlas_mask=None, kernel_size=11):
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
    cmap_name: str = "white_orange_red_black",
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


def build_synthetic_prv_signal(
    atlas_labels: np.ndarray,
    *,
    center: tuple[float, float, float] | None = None,
    spread: tuple[float, float, float] = (15.0, 70.0, 55.0),
) -> np.ndarray:
    shape = atlas_labels.shape
    if center is None:
        center = (shape[0] * 0.46, shape[1] * 0.48, shape[2] * 0.56)
    d, a, m = np.ogrid[: shape[0], : shape[1], : shape[2]]
    signal = np.zeros(shape, dtype=np.float32)

    components = [
        (center, spread, 1.0),
        ((center[0] - 9, center[1] - 42, center[2] + 36), (spread[0] * 0.95, spread[1] * 0.75, spread[2] * 0.62), 0.58),
        ((center[0] + 8, center[1] + 64, center[2] - 42), (spread[0] * 0.9, spread[1] * 0.85, spread[2] * 0.7), 0.42),
        ((center[0] - 18, center[1] + 12, center[2] - 76), (spread[0] * 0.65, spread[1] * 0.55, spread[2] * 0.45), 0.28),
    ]
    for component_center, component_spread, weight in components:
        cd, ca, cm = component_center
        sd, sa, sm = component_spread
        signal += weight * np.exp(-0.5 * (((d - cd) / sd) ** 2 + ((a - ca) / sa) ** 2 + ((m - cm) / sm) ** 2)).astype(np.float32)

    texture = 1.0 + 0.08 * np.sin(a / 16.0) + 0.06 * np.cos(m / 19.0)
    signal *= texture.astype(np.float32)
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
    cmap_name: str = "white_orange_red_black",
    dpi: int = 300,
    center: tuple[float, float, float] | None = None,
    spread: tuple[float, float, float] = (15.0, 70.0, 55.0),
) -> list[Path]:
    label_path = Path(label_path)
    labels = np.asarray(tifffile.memmap(str(label_path)))
    signal = build_synthetic_prv_signal(labels, center=center, spread=spread)
    brain_dv = np.flatnonzero(np.any(labels > 0, axis=(1, 2)))
    if len(brain_dv) == 0:
        raise ValueError("Atlas label contains no nonzero brain voxels")

    start = int(np.quantile(brain_dv, 0.22))
    stop = int(np.quantile(brain_dv, 0.78))
    indices = np.linspace(start, stop, int(sample_count)).round().astype(int)

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
    parser.add_argument("--cmap", default="white_orange_red_black")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--line-width", type=float, default=0.16)
    parser.add_argument("--brain-outline-width", type=float, default=0.42)
    parser.add_argument("--colorbar-label", default="Signal Intensity")

    parser.add_argument("--sample-output-dir", default="S:/可视化素材/heatmap")
    parser.add_argument("--sample-count", type=int, default=10)
    parser.add_argument("--sample-center", type=_parse_triplet, default=None, help="Synthetic PRV center as dv,ap,ml index")
    parser.add_argument("--sample-spread", type=_parse_triplet, default=(15.0, 70.0, 55.0), help="Synthetic PRV spread as dv,ap,ml voxels")
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
            local_signal = build_local_signal_volume(signal, sigma=args.sigma, alpha=args.alpha, atlas_mask=labels)
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
            )
            payload = {"mode": args.mode, "output_dir": str(args.sample_output_dir), "outputs": [str(path) for path in outputs]}
    except Exception as exc:
        print(f"Error: {exc}")
        return 1

    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
