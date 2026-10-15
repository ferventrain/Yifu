"""Interactive 3D heatmap of an atlas-space c-Fos count volume (MIP volume rendering).

Takes the warp output ``<sample>_ch1_count_volume.tif`` (axes SI, AP, LR at
25 um), masks it to the Allen annotation, smooths with a fixed physical
sigma, and renders a see-through heatmap: default mode is maximum intensity
projection with the magma colormap, so the colour along each view ray is the
strongest signal through the brain. Interactive window by default;
``--output`` writes an offscreen screenshot instead.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import tifffile

ATLAS_VOXEL_UM = 25.0
MODE_CHOICES = ("mip", "composite")
DEFAULT_CMAP = "magma"
DEFAULT_PERCENTILE = 99.5
DEFAULT_POWER = 1.2
DEFAULT_SIGMA_UM = 150.0
DEFAULT_ALPHA_STOPS = ((0.0, 0.0), (0.3, 0.02), (0.6, 0.15), (1.0, 0.9))


def prepare_heatmap_grid(
    count_volume_path: str | Path,
    *,
    annotation: np.ndarray | None,
    sigma_um: float = DEFAULT_SIGMA_UM,
) -> np.ndarray:
    """Count tif -> masked smoothed grid with axes (AP, SI, LR).

    The warp writes counts as (SI, AP, LR); transposing to (AP, SI, LR)
    matches the BrainGlobe root-mesh space. Values outside the annotation
    (interpolation leakage / far-field smear) stay exactly zero.
    """

    counts = tifffile.imread(str(count_volume_path)).astype(np.float32)
    grid = np.transpose(counts, (1, 0, 2)).copy()
    if annotation is not None:
        annotation = np.asarray(annotation)
        if annotation.shape != grid.shape:
            raise ValueError(f"annotation shape {annotation.shape} does not match grid {grid.shape} (need AP, SI, LR)")
        grid[~annotation] = 0.0
    from scipy.ndimage import gaussian_filter

    smoothed = gaussian_filter(grid, sigma_um / ATLAS_VOXEL_UM)
    if annotation is not None:
        smoothed[~annotation] = 0.0
    return smoothed


def load_annotation_ap_si_lr() -> np.ndarray:
    from brainglobe_atlasapi.bg_atlas import BrainGlobeAtlas

    return np.asarray(BrainGlobeAtlas("allen_mouse_25um", check_latest=False).annotation) > 0


def _cmap_anchor_list(cmap_name: str, stops: int = 9) -> list[tuple[float, tuple[float, float, float]]]:
    import matplotlib

    cmap = matplotlib.colormaps[cmap_name]
    return [(round(float(x), 3), tuple(float(c) for c in cmap(x)[:3])) for x in np.linspace(0.0, 1.0, stops)]


def _root_brain_mesh(atlas_name: str):
    import vedo
    from brainglobe_atlasapi.bg_atlas import BrainGlobeAtlas

    atlas = BrainGlobeAtlas(atlas_name, check_latest=False)
    meshio_mesh = atlas.structures["root"]["mesh"]
    verts = np.asarray(meshio_mesh.points, dtype=np.float32)
    faces = np.vstack([c.data for c in meshio_mesh.cells if c.type == "triangle"]).astype(np.int64)
    mesh = vedo.Mesh([verts, faces])
    mesh.color("lightgrey")
    mesh.alpha(0.05)
    return mesh


def render_heatmap(
    grid: np.ndarray,
    *,
    cmap: str = DEFAULT_CMAP,
    percentile: float = DEFAULT_PERCENTILE,
    power: float = DEFAULT_POWER,
    mode: str = "mip",
    sigma_alpha_stops: tuple[tuple[float, float], ...] = DEFAULT_ALPHA_STOPS,
    output: str | Path | None = None,
    window_size: tuple[int, int] = (1500, 1000),
    atlas_name: str = "allen_mouse_25um",
    title: str = "c-Fos 3D heatmap - L-drag rotate / wheel zoom / q close",
) -> Path | None:
    """Render the prepared (AP, SI, LR) grid; interactive window unless output is given."""

    if mode not in MODE_CHOICES:
        raise ValueError(f"mode must be one of {MODE_CHOICES}, got {mode}")
    import vedo

    positive = grid[grid > 0]
    if positive.size == 0:
        raise ValueError("Heatmap grid is empty; nothing to render.")
    vmax = float(np.percentile(positive, percentile))
    normalized = np.clip(grid / vmax, 0.0, 1.0) ** power

    volume = vedo.Volume(np.ascontiguousarray(normalized, dtype=np.float32), spacing=(ATLAS_VOXEL_UM,) * 3)
    volume.color(_cmap_anchor_list(cmap))
    volume.alpha([tuple(stop) for stop in sigma_alpha_stops])
    volume.mode(1 if mode == "mip" else 0)

    brain_mesh = _root_brain_mesh(atlas_name)
    offscreen = output is not None
    plt = vedo.Plotter(offscreen=offscreen, size=window_size, bg="white")
    plt.show(volume, brain_mesh, title, interactive=not offscreen, axes=0)
    if offscreen:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        plt.screenshot(str(output))
        return output
    return None


def resolve_count_volume(sample_dir: str | Path) -> Path:
    sample_dir = Path(sample_dir)
    candidates = sorted(sample_dir.glob("visualization/*count_volume.tif"))
    if not candidates:
        raise FileNotFoundError(f"No visualization/*count_volume.tif under {sample_dir}; run warp_mask_zarr_to_atlas_points --volume_mode count first.")
    return candidates[0]


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Interactive 3D heatmap (MIP volume rendering) of an atlas-space c-Fos count volume."
    )
    parser.add_argument("--count_volume", default="", help="Atlas-space count volume TIFF (SI, AP, LR). Alternative to --sample_dir.")
    parser.add_argument("--sample_dir", default="", help="Sample root; uses visualization/*count_volume.tif.")
    parser.add_argument("--cmap", default=DEFAULT_CMAP, help=f"Matplotlib colormap. Default: {DEFAULT_CMAP}.")
    parser.add_argument("--percentile", type=float, default=DEFAULT_PERCENTILE,
                        help="Normalization percentile of nonzero voxels (higher = smaller peak region). Default: 99.5.")
    parser.add_argument("--power", type=float, default=DEFAULT_POWER,
                        help="Gamma on normalized values (higher = dimmer background). Default: 1.2.")
    parser.add_argument("--mode", choices=MODE_CHOICES, default="mip",
                        help="mip: see-through maximum intensity projection; composite: surface-shaded. Default: mip.")
    parser.add_argument("--sigma_um", type=float, default=DEFAULT_SIGMA_UM, help="Smoothing sigma in microns. Default: 150.")
    parser.add_argument("--output", default="", help="Offscreen screenshot path; omit for an interactive window.")
    parser.add_argument("--window", default="1500,1000", help="Window size as width,height. Default: 1500,1000.")
    return parser


def main() -> int:
    args = build_argparser().parse_args()
    try:
        if args.count_volume:
            count_volume = Path(args.count_volume)
        elif args.sample_dir:
            count_volume = resolve_count_volume(args.sample_dir)
        else:
            raise ValueError("Pass either --count_volume or --sample_dir.")
        parts = [part.strip() for part in str(args.window).split(",") if part.strip()]
        if len(parts) != 2:
            raise ValueError(f"--window must be width,height, got: {args.window}")
        window_size = (int(parts[0]), int(parts[1]))
        grid = prepare_heatmap_grid(count_volume, annotation=load_annotation_ap_si_lr(), sigma_um=args.sigma_um)
        result = render_heatmap(
            grid,
            cmap=args.cmap,
            percentile=args.percentile,
            power=args.power,
            mode=args.mode,
            output=(Path(args.output) if args.output else None),
            window_size=window_size,
        )
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    if result is not None:
        print(f"Saved heatmap screenshot to: {result}")
    else:
        print("Closed interactive heatmap window.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
