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
REGION_CSV = Path(__file__).resolve().parents[1] / "registration" / "Region_Csv_Rev1_updated.CSV"


def region_mask_from_acronyms(acronyms: list[str], annotation: np.ndarray, csv_path: str | Path = REGION_CSV) -> np.ndarray:
    """Boolean mask of the atlas structures for the given acronyms (+ descendants).

    Ids come from the repo region CSV (Allen CCF id space, matching the
    BrainGlobe annotation used for masking).
    """

    import ast

    import pandas as pd

    frame = pd.read_csv(csv_path)
    wanted = {a.strip().lower() for a in acronyms}
    roots: set[int] = set()
    rows = []
    for _, row in frame.iterrows():
        base, acro = (str(row["name"]).rsplit(",", 1) + [""])[:2] if "," in str(row["name"]) else (str(row["name"]), "")
        path = ast.literal_eval(row["structure_id_path"]) if isinstance(row["structure_id_path"], str) else list(row["structure_id_path"])
        rows.append((int(row["id"]), acro.strip(), [int(v) for v in path]))
        if acro.strip().lower() in wanted:
            roots.add(int(row["id"]))
    if not roots:
        raise ValueError(f"No region acronyms matched in {csv_path}: {sorted(acronyms)}")
    keep_ids = set()
    for rid, _acro, path in rows:
        if any(r in path for r in roots):
            keep_ids.add(rid)
    id_arr = np.fromiter(keep_ids, dtype=np.int64, count=len(keep_ids))
    ann_flat = annotation.reshape(-1)
    mask_flat = np.isin(ann_flat, id_arr)
    return mask_flat.reshape(annotation.shape)


def camera_position(shape_zyx: tuple[int, int, int], azimuth_deg: float, elevation_deg: float, radius_factor: float = 1.7, center=None):
    """Spherical camera around a focal center in the (AP, SI, LR) frame.

    az=0/el=0 looks from anterior (+AP); az=90 from the left (+LR); el=90
    looks straight down from the top. ``center`` defaults to the volume
    center (pass a region centroid to focus a sub-volume). Returns
    (pos, focal, viewup).
    """

    spacing = ATLAS_VOXEL_UM
    if center is None:
        center = (
            (shape_zyx[0] - 1) / 2.0 * spacing,
            (shape_zyx[1] - 1) / 2.0 * spacing,
            (shape_zyx[2] - 1) / 2.0 * spacing,
        )
    else:
        center = tuple(float(c) for c in center)
    radius = radius_factor * float(np.max(shape_zyx)) * spacing
    az, el = np.radians(azimuth_deg), np.radians(elevation_deg)
    # spherical direction in (AP, SI, LR): el in the AP-SI plane, LR via az
    dir_ap = np.cos(el) * np.cos(az)
    dir_si = np.sin(el)
    dir_lr = np.cos(el) * np.sin(az)
    pos = (center[0] + radius * dir_ap, center[1] + radius * dir_si, center[2] + radius * dir_lr)
    viewup = (0.0, 1.0, 0.0) if abs(np.sin(el)) < 0.99 else (0.0, 0.0, 1.0)
    return pos, center, viewup


def mask_surface_mesh(mask: np.ndarray, color: str = "crimson", alpha: float = 0.9):
    """Marching-cubes surface of a boolean (AP, SI, LR) mask, in um coordinates."""

    import vedo
    from skimage import measure

    pad = np.pad(np.asarray(mask, dtype=np.float32), 1)
    verts, faces, _normals, _values = measure.marching_cubes(pad, level=0.5, spacing=(ATLAS_VOXEL_UM,) * 3)
    verts -= ATLAS_VOXEL_UM  # undo the padding offset
    mesh = vedo.Mesh([verts, faces.astype(np.int64)])
    mesh.color(color)
    mesh.alpha(alpha)
    centroid = verts.mean(axis=0)
    return mesh, centroid


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
        inside = annotation != 0
        if inside.shape != grid.shape:
            raise ValueError(f"annotation shape {annotation.shape} does not match grid {grid.shape} (need AP, SI, LR)")
        grid[~inside] = 0.0
    from scipy.ndimage import gaussian_filter

    smoothed = gaussian_filter(grid, sigma_um / ATLAS_VOXEL_UM)
    if annotation is not None:
        smoothed[~inside] = 0.0
    return smoothed


def load_annotation_ap_si_lr() -> np.ndarray:
    """Raw BrainGlobe annotation (Allen structure ids, AP, SI, LR)."""

    from brainglobe_atlasapi.bg_atlas import BrainGlobeAtlas

    return np.asarray(BrainGlobeAtlas("allen_mouse_25um", check_latest=False).annotation)


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
    camera: tuple | None = None,
    brain_alpha: float = 0.05,
    extra_actors: tuple = (),
    include_volume: bool = True,
    show_brain: bool = True,
    title: str = "c-Fos 3D heatmap - L-drag rotate / wheel zoom / q close",
) -> Path | None:
    """Render the prepared (AP, SI, LR) grid; interactive window unless output is given.

    ``camera`` = (pos, focal, viewup) as produced by :func:`camera_position`
    pins the offscreen view (ignored in the interactive window).
    ``include_volume=False`` renders only the meshes (location views).
    """

    if mode not in MODE_CHOICES:
        raise ValueError(f"mode must be one of {MODE_CHOICES}, got: {mode}")
    import vedo

    actors = []
    if include_volume:
        positive = grid[grid > 0]
        if positive.size == 0:
            raise ValueError("Heatmap grid is empty; nothing to render.")
        vmax = float(np.percentile(positive, percentile))
        normalized = np.clip(grid / vmax, 0.0, 1.0) ** power

        volume = vedo.Volume(np.ascontiguousarray(normalized, dtype=np.float32), spacing=(ATLAS_VOXEL_UM,) * 3)
        volume.color(_cmap_anchor_list(cmap))
        volume.alpha([tuple(stop) for stop in sigma_alpha_stops])
        volume.mode(1 if mode == "mip" else 0)
        actors.append(volume)
    actors.extend(extra_actors)

    if show_brain:
        brain_mesh = _root_brain_mesh(atlas_name)
        brain_mesh.alpha(brain_alpha)
        actors.append(brain_mesh)
    offscreen = output is not None
    plt = vedo.Plotter(offscreen=offscreen, size=window_size, bg="white")
    if camera is not None:
        pos, focal, viewup = camera
        cam_kw = dict(camera=dict(pos=pos, focalPoint=focal, viewup=viewup))
    else:
        cam_kw = {}
    plt.show(*actors, title, interactive=not offscreen, axes=0, **cam_kw)
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
    parser.add_argument(
        "--camera", default="",
        help="Offscreen view as azimuth_deg,elevation_deg[,radius_factor]. az=0 anterior, 90 left, el=90 top-down. "
        "E.g. '90,0' lateral view. Ignored in the interactive window.",
    )
    parser.add_argument(
        "--regions", default="",
        help="Comma-separated region acronyms (repo CSV, e.g. 'VTA,LHA'); restricts the heatmap to those "
        "structures incl. descendants. Pair with a higher --brain_alpha to show their location.",
    )
    parser.add_argument("--brain_alpha", type=float, default=-1.0,
                        help="Root brain mesh alpha (0-1). Default: 0.05, or 0.15 when --regions is given.")
    parser.add_argument(
        "--region_view", choices=("heat", "location", "zoom"), default="heat",
        help="With --regions: heat (default) = region heatmap inside the translucent brain; "
        "location = brain outline + solid colored region surface (where is it); "
        "zoom = translucent region shell + its internal heatmap, camera pulled close.",
    )
    parser.add_argument("--region_color", default="#D55E00",
                        help="Region surface color for location/zoom views (name or #hex). Default vermillion #D55E00.")
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
        annotation = load_annotation_ap_si_lr()
        grid = prepare_heatmap_grid(count_volume, annotation=annotation, sigma_um=args.sigma_um)
        region_centroid = None
        extra_actors: tuple = ()
        include_volume = True
        if args.regions:
            acronyms = [a.strip() for a in args.regions.split(",") if a.strip()]
            region_mask = region_mask_from_acronyms(acronyms, annotation)
            grid = grid * region_mask
            if args.region_view in ("location", "zoom"):
                region_mesh, region_centroid = mask_surface_mesh(region_mask)
                region_mesh.color(args.region_color)
                if args.region_view == "location":
                    region_mesh.alpha(1.0)  # opaque: always visible through the shell
                    include_volume = False
                else:  # zoom: translucent region shell around its own heatmap, no brain shell
                    region_mesh.alpha(0.18)  # translucent vermillion outer shell
                extra_actors = (region_mesh,)
        camera = None
        if args.camera:
            cam_parts = [p.strip() for p in args.camera.split(",") if p.strip()]
            if len(cam_parts) not in (2, 3):
                raise ValueError(f"--camera must be az_deg,el_deg[,radius_factor], got: {args.camera}")
            radius_factor = float(cam_parts[2]) if len(cam_parts) == 3 else (0.55 if args.region_view == "zoom" else 1.7)
            camera = camera_position(
                grid.shape,
                float(cam_parts[0]),
                float(cam_parts[1]),
                radius_factor,
                # only the zoom view orbits the region; location/heat frame the whole brain
                center=region_centroid if args.region_view == "zoom" else None,
            )
        brain_alpha = args.brain_alpha if args.brain_alpha >= 0 else (
            0.06 if args.region_view == "location" else (0.15 if args.regions else 0.05)
        )
        result = render_heatmap(
            grid,
            cmap=args.cmap,
            percentile=args.percentile,
            power=args.power,
            mode=args.mode,
            output=(Path(args.output) if args.output else None),
            window_size=window_size,
            camera=camera,
            brain_alpha=brain_alpha,
            extra_actors=extra_actors,
            include_volume=include_volume,
            show_brain=args.region_view != "zoom",
            title="c-Fos 3D heatmap" + (f" - {args.regions}" if args.regions else ""),
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
