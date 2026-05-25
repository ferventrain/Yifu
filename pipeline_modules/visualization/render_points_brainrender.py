"""Render atlas-space punctate signals inside a transparent standard brain with brainrender."""

from __future__ import annotations

import argparse
import configparser
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


def _prepare_brainrender_runtime() -> None:
    """Redirect brainrender cache/log files to a private temp directory.

    This avoids startup failures on Windows when another process is holding
    ``~/.brainglobe/brainrender/log.log`` open, while still reusing the user's
    existing BrainGlobe atlas cache.
    """

    original_home = Path(os.environ.get("USERPROFILE") or str(Path.home()))
    original_brainglobe_dir = original_home / ".brainglobe"
    runtime_home = Path(tempfile.gettempdir()) / "brainrender_runtime_yifu"
    runtime_home.mkdir(parents=True, exist_ok=True)

    config_dir = runtime_home / ".config" / "brainglobe"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / "bg_config.conf"

    conf = configparser.ConfigParser()
    conf["default_dirs"] = {
        "brainglobe_dir": str(original_brainglobe_dir),
        "interm_download_dir": str(original_brainglobe_dir),
    }
    with config_path.open("w", encoding="utf-8") as handle:
        conf.write(handle)

    os.environ["BRAINGLOBE_CONFIG_DIR"] = str(config_dir)
    os.environ.setdefault("HOME", str(runtime_home))
    os.environ.setdefault("USERPROFILE", str(runtime_home))


_prepare_brainrender_runtime()

from brainrender import Scene, settings  # noqa: E402
from brainrender.actors import Points  # noqa: E402
from brainglobe_atlasapi.bg_atlas import BrainGlobeAtlas  # noqa: E402
from brainglobe_atlasapi.list_atlases import get_downloaded_atlases  # noqa: E402
from pipeline_modules.visualization.coarse_region_metric_plot import DEFAULT_REGION_IDS  # noqa: E402


DEFAULT_OUTPUT = "brainrender_points.png"
DEFAULT_COLUMNS = ("x", "y", "z")
ALIAS_GROUPS: dict[str, tuple[str, ...]] = {
    "x": ("x", "ap", "anterior_posterior", "anteriorposterior"),
    "y": ("y", "dv", "dorsal_ventral", "dorsoventral"),
    "z": ("z", "ml", "lr", "mediolateral", "left_right", "left-right"),
}
COARSE_REGION_COLORS = [
    "#0072B2",
    "#D55E00",
    "#009E73",
    "#CC79A7",
    "#E69F00",
    "#56B4E9",
    "#7F3C8D",
    "#11A579",
    "#3969AC",
    "#F2B701",
    "#E73F74",
    "#80BA5A",
    "#E68310",
    "#008695",
]


def parse_columns(value: str) -> tuple[str, str, str]:
    parts = tuple(part.strip() for part in str(value).split(",") if part.strip())
    if len(parts) != 3:
        raise ValueError(f"--columns must contain exactly 3 column names, got: {value}")
    return parts[0], parts[1], parts[2]


def parse_triplet(value: str) -> tuple[float, float, float]:
    parts = tuple(part.strip() for part in str(value).split(",") if part.strip())
    if len(parts) != 3:
        raise ValueError(f"Expected three comma-separated values, got: {value}")
    parsed = tuple(float(part) for part in parts)
    if any(part <= 0 for part in parsed):
        raise ValueError(f"Values must be positive, got: {parsed}")
    return parsed[0], parsed[1], parsed[2]


def normalize_name(name: str) -> str:
    return name.strip().lower().replace(" ", "_").replace("-", "_")


def infer_columns(table: pd.DataFrame) -> tuple[str, str, str]:
    normalized_to_original = {normalize_name(col): col for col in table.columns}
    resolved: list[str] = []
    for axis in ("x", "y", "z"):
        candidates = ALIAS_GROUPS[axis]
        match = next((normalized_to_original[key] for key in candidates if key in normalized_to_original), None)
        if match is None:
            raise ValueError(
                "Could not infer coordinate columns from CSV. "
                f"Available columns: {list(table.columns)}. "
                "Please pass --columns x_col,y_col,z_col explicitly."
            )
        resolved.append(match)
    return resolved[0], resolved[1], resolved[2]


def load_points(csv_path: str | Path, columns: tuple[str, str, str] | None) -> np.ndarray:
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Points CSV not found: {csv_path}")

    table = pd.read_csv(csv_path)
    if table.empty:
        raise ValueError(f"Points CSV is empty: {csv_path}")

    x_col, y_col, z_col = columns if columns is not None else infer_columns(table)
    missing = [col for col in (x_col, y_col, z_col) if col not in table.columns]
    if missing:
        raise KeyError(f"Missing coordinate column(s) in {csv_path}: {missing}")

    coords = table.loc[:, [x_col, y_col, z_col]].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
    valid = np.all(np.isfinite(coords), axis=1)
    coords = coords[valid]
    if coords.size == 0:
        raise ValueError(f"No valid numeric point coordinates found in {csv_path}")
    return coords


def configure_brainrender(*, background: str, root_alpha: float, show_axes: bool, offscreen: bool) -> None:
    settings.BACKGROUND_COLOR = background
    settings.ROOT_ALPHA = float(root_alpha)
    settings.ROOT_COLOR = [0.82, 0.82, 0.82]
    settings.SHADER_STYLE = "plastic"
    settings.SHOW_AXES = bool(show_axes)
    settings.OFFSCREEN = bool(offscreen)
    settings.WHOLE_SCREEN = False
    settings.INTERACTIVE = not offscreen
    settings.DEFAULT_CAMERA = "three_quarters"


def ensure_atlas_available(atlas_name: str) -> None:
    downloaded = set(get_downloaded_atlases())
    if atlas_name not in downloaded:
        raise FileNotFoundError(
            f"BrainGlobe atlas '{atlas_name}' is not downloaded on this machine. "
            f"Currently downloaded atlases: {sorted(downloaded) if downloaded else 'none'}. "
            "Please download the atlas once in the napari environment, then rerun this script."
        )


def atlas_resolution_xyz(atlas_name: str) -> tuple[float, float, float]:
    atlas = BrainGlobeAtlas(atlas_name, check_latest=False)
    resolution = tuple(float(value) for value in atlas.resolution)
    if len(resolution) != 3:
        raise ValueError(f"Unexpected atlas resolution for {atlas_name}: {resolution}")
    return resolution[0], resolution[1], resolution[2]


def convert_coordinate_units(
    points: np.ndarray,
    *,
    coordinate_units: str,
    atlas_name: str,
    atlas_resolution: tuple[float, float, float] | None,
) -> tuple[np.ndarray, str]:
    if coordinate_units not in {"auto", "um", "voxel"}:
        raise ValueError(f"Unsupported coordinate units: {coordinate_units}")

    resolution = np.asarray(atlas_resolution or atlas_resolution_xyz(atlas_name), dtype=np.float64)
    if coordinate_units == "auto":
        # Brainrender uses atlas-space physical coordinates. A max coordinate
        # below ~1000 is almost certainly voxel indices for a 25 um atlas.
        coordinate_units = "voxel" if float(np.nanmax(points)) < 1000.0 and float(np.max(resolution)) > 1.0 else "um"

    if coordinate_units == "voxel":
        points = points * resolution[None, :]
    return points, coordinate_units


def filter_points_to_brain(
    points: np.ndarray,
    *,
    atlas_name: str,
    atlas_resolution: tuple[float, float, float] | None,
) -> tuple[np.ndarray, int, int]:
    atlas = BrainGlobeAtlas(atlas_name, check_latest=False)
    annotation = atlas.annotation
    resolution = np.asarray(atlas_resolution or atlas.resolution, dtype=np.float64)

    indices = np.floor(points / resolution[None, :]).astype(np.int64)
    in_bounds = (
        (indices[:, 0] >= 0)
        & (indices[:, 0] < annotation.shape[0])
        & (indices[:, 1] >= 0)
        & (indices[:, 1] < annotation.shape[1])
        & (indices[:, 2] >= 0)
        & (indices[:, 2] < annotation.shape[2])
    )
    inside = np.zeros(len(points), dtype=bool)
    inside[in_bounds] = annotation[indices[in_bounds, 0], indices[in_bounds, 1], indices[in_bounds, 2]] > 0
    return points[inside], int(np.count_nonzero(inside)), int(len(points))


def brain_mask_fraction(
    points: np.ndarray,
    *,
    atlas: BrainGlobeAtlas,
    atlas_resolution: tuple[float, float, float] | None,
) -> float:
    annotation = atlas.annotation
    resolution = np.asarray(atlas_resolution or atlas.resolution, dtype=np.float64)
    indices = np.floor(points / resolution[None, :]).astype(np.int64)
    in_bounds = (
        (indices[:, 0] >= 0)
        & (indices[:, 0] < annotation.shape[0])
        & (indices[:, 1] >= 0)
        & (indices[:, 1] < annotation.shape[1])
        & (indices[:, 2] >= 0)
        & (indices[:, 2] < annotation.shape[2])
    )
    if not np.any(in_bounds):
        return 0.0
    inside = np.zeros(len(points), dtype=bool)
    inside[in_bounds] = annotation[indices[in_bounds, 0], indices[in_bounds, 1], indices[in_bounds, 2]] > 0
    return float(np.count_nonzero(inside)) / float(len(points))


def auto_reorder_axes_to_atlas(
    points: np.ndarray,
    *,
    atlas_name: str,
    atlas_resolution: tuple[float, float, float] | None,
) -> tuple[np.ndarray, str]:
    atlas = BrainGlobeAtlas(atlas_name, check_latest=False)
    orders = {
        "x,y,z": (0, 1, 2),
        "x,z,y": (0, 2, 1),
        "y,x,z": (1, 0, 2),
        "y,z,x": (1, 2, 0),
        "z,x,y": (2, 0, 1),
        "z,y,x": (2, 1, 0),
    }
    best_order = "x,y,z"
    best_fraction = -1.0
    for name, indices in orders.items():
        fraction = brain_mask_fraction(points[:, indices], atlas=atlas, atlas_resolution=atlas_resolution)
        if fraction > best_fraction:
            best_order = name
            best_fraction = fraction
    if best_order != "x,y,z":
        print(f"Auto axis order: using {best_order} ({best_fraction:.1%} points inside atlas mask).")
    return points[:, orders[best_order]], best_order


def resolve_region_id(atlas: BrainGlobeAtlas, region: str, region_id: int | None) -> int | None:
    if region_id is not None:
        if region_id not in atlas.structures:
            raise ValueError(f"Region id {region_id} is not present in atlas {atlas.atlas_name}.")
        return int(region_id)
    if not region.strip():
        return None

    query = region.strip()
    lookup = atlas.lookup_df
    normalized = lookup.assign(
        acronym_norm=lookup["acronym"].astype(str).map(normalize_name),
        name_norm=lookup["name"].astype(str).map(normalize_name),
    )
    query_norm = normalize_name(query)
    matches = normalized[(normalized["acronym_norm"] == query_norm) | (normalized["name_norm"] == query_norm)]
    if matches.empty:
        matches = normalized[
            normalized["acronym_norm"].str.contains(query_norm, regex=False)
            | normalized["name_norm"].str.contains(query_norm, regex=False)
        ]
    if matches.empty:
        raise ValueError(f"Could not find atlas region matching: {region}")
    if len(matches) > 1:
        preview = ", ".join(f"{row.acronym}({int(row.id)})" for row in matches.head(10).itertuples())
        raise ValueError(f"Region query matched multiple regions. Use --region_id. Matches: {preview}")
    return int(matches.iloc[0]["id"])


def descendant_region_ids(atlas: BrainGlobeAtlas, region_id: int, include_descendants: bool) -> set[int]:
    if not include_descendants:
        return {int(region_id)}
    tree = atlas.structures.tree
    subtree = tree.subtree(region_id)
    return {int(node.identifier) for node in subtree.all_nodes()}


def region_acronym(atlas: BrainGlobeAtlas, region_id: int) -> str:
    structure = atlas.structures[int(region_id)]
    return str(structure["acronym"])


def filter_points_to_region(
    points: np.ndarray,
    *,
    atlas_name: str,
    atlas_resolution: tuple[float, float, float] | None,
    region: str,
    region_id: int | None,
    include_descendants: bool,
) -> tuple[np.ndarray, int, int, int, str]:
    atlas = BrainGlobeAtlas(atlas_name, check_latest=False)
    resolved_id = resolve_region_id(atlas, region, region_id)
    if resolved_id is None:
        return points, len(points), len(points), 0, ""

    ids = descendant_region_ids(atlas, resolved_id, include_descendants)
    annotation = atlas.annotation
    resolution = np.asarray(atlas_resolution or atlas.resolution, dtype=np.float64)
    indices = np.floor(points / resolution[None, :]).astype(np.int64)
    in_bounds = (
        (indices[:, 0] >= 0)
        & (indices[:, 0] < annotation.shape[0])
        & (indices[:, 1] >= 0)
        & (indices[:, 1] < annotation.shape[1])
        & (indices[:, 2] >= 0)
        & (indices[:, 2] < annotation.shape[2])
    )
    inside = np.zeros(len(points), dtype=bool)
    labels = annotation[indices[in_bounds, 0], indices[in_bounds, 1], indices[in_bounds, 2]]
    inside[in_bounds] = np.isin(labels, list(ids))
    return points[inside], int(np.count_nonzero(inside)), int(len(points)), resolved_id, region_acronym(atlas, resolved_id)


def point_annotation_labels(
    points: np.ndarray,
    *,
    atlas: BrainGlobeAtlas,
    atlas_resolution: tuple[float, float, float] | None,
) -> np.ndarray:
    annotation = atlas.annotation
    resolution = np.asarray(atlas_resolution or atlas.resolution, dtype=np.float64)
    indices = np.floor(points / resolution[None, :]).astype(np.int64)
    in_bounds = (
        (indices[:, 0] >= 0)
        & (indices[:, 0] < annotation.shape[0])
        & (indices[:, 1] >= 0)
        & (indices[:, 1] < annotation.shape[1])
        & (indices[:, 2] >= 0)
        & (indices[:, 2] < annotation.shape[2])
    )
    labels = np.zeros(len(points), dtype=np.int64)
    labels[in_bounds] = annotation[indices[in_bounds, 0], indices[in_bounds, 1], indices[in_bounds, 2]].astype(np.int64)
    return labels


def coarse_region_for_label(atlas: BrainGlobeAtlas, label: int, coarse_ids: set[int]) -> int | None:
    if label <= 0 or int(label) not in atlas.structures:
        return None
    path = atlas.structures[int(label)].get("structure_id_path", [])
    matches = [int(region_id) for region_id in path if int(region_id) in coarse_ids]
    return matches[-1] if matches else None


def coarse_region_colors_for_points(
    points: np.ndarray,
    *,
    atlas_name: str,
    atlas_resolution: tuple[float, float, float] | None,
    drop_unassigned: bool,
) -> tuple[np.ndarray, list[str], dict[int, int]]:
    atlas = BrainGlobeAtlas(atlas_name, check_latest=False)
    labels = point_annotation_labels(points, atlas=atlas, atlas_resolution=atlas_resolution)
    coarse_ids = {int(region_id) for region_id in DEFAULT_REGION_IDS}
    color_by_region = {
        int(region_id): COARSE_REGION_COLORS[index % len(COARSE_REGION_COLORS)]
        for index, region_id in enumerate(DEFAULT_REGION_IDS)
    }

    coarse_for_label: dict[int, int | None] = {}
    colors: list[str] = []
    assigned = np.zeros(len(points), dtype=bool)
    counts: dict[int, int] = {int(region_id): 0 for region_id in DEFAULT_REGION_IDS}

    for index, label in enumerate(labels):
        label_id = int(label)
        if label_id not in coarse_for_label:
            coarse_for_label[label_id] = coarse_region_for_label(atlas, label_id, coarse_ids)
        coarse_id = coarse_for_label[label_id]
        if coarse_id is None:
            colors.append("#9a9a9a")
            continue
        colors.append(color_by_region[coarse_id])
        assigned[index] = True
        counts[coarse_id] += 1

    if drop_unassigned:
        points = points[assigned]
        colors = [color for color, keep in zip(colors, assigned) if keep]
    counts = {region_id: count for region_id, count in counts.items() if count > 0}
    return points, colors, counts


def reorder_axes(points: np.ndarray, axis_order: str) -> np.ndarray:
    order_text = axis_order.strip().lower()
    if not order_text:
        return points
    aliases = {"ap": "x", "dv": "y", "ml": "z"}
    parts = [aliases.get(part.strip(), part.strip()) for part in order_text.split(",") if part.strip()]
    if len(parts) != 3 or set(parts) != {"x", "y", "z"}:
        raise ValueError(f"--axis_order must be a permutation of x,y,z or ap,dv,ml, got: {axis_order}")
    indices = [{"x": 0, "y": 1, "z": 2}[part] for part in parts]
    return points[:, indices]


def summarize_points(points: np.ndarray) -> str:
    mins = np.min(points, axis=0)
    maxs = np.max(points, axis=0)
    return (
        f"x={mins[0]:.1f}..{maxs[0]:.1f}, "
        f"y={mins[1]:.1f}..{maxs[1]:.1f}, "
        f"z={mins[2]:.1f}..{maxs[2]:.1f}, n={len(points)}"
    )


def render_points_scene(
    points: np.ndarray,
    *,
    atlas_name: str,
    point_radius: float,
    point_color: str | list[str],
    point_alpha: float,
    root_alpha: float,
    title: str | None,
    background: str,
    show_axes: bool,
    root: bool,
    whole_brain_silhouette: bool,
    region_mesh: str,
    region_mesh_id: int | None,
    region_alpha: float,
    region_color: str,
    region_silhouette: bool,
    hemisphere: str,
) -> Scene:
    scene = Scene(
        root=root,
        atlas_name=atlas_name,
        check_latest=False,
        inset=False,
        title=title,
    )
    if whole_brain_silhouette and not root:
        print("--whole_brain_silhouette keeps the standard transparent whole-brain mesh; ignoring --no_root.")

    if region_mesh or region_mesh_id is not None:
        atlas = BrainGlobeAtlas(atlas_name, check_latest=False)
        resolved_id = resolve_region_id(atlas, region_mesh, region_mesh_id)
        if resolved_id is None:
            raise ValueError("--region_mesh or --region_mesh_id is required to render a region mesh.")
        scene.add_brain_region(
            region_acronym(atlas, resolved_id),
            alpha=region_alpha,
            color=region_color or None,
            silhouette=region_silhouette,
            hemisphere=hemisphere,
            force=True,
        )

    scene.add(
        Points(
            points,
            name="signal_points",
            colors=point_color,
            alpha=point_alpha,
            radius=point_radius,
            res=10,
        )
    )
    return scene


def resolve_output_path(csv_path: Path, output: str | None) -> Path:
    if output:
        return Path(output)
    return csv_path.with_name(DEFAULT_OUTPUT)


def resolve_render_output_path(args: argparse.Namespace, points_csv: Path) -> Path | None:
    if not args.output:
        return None
    output = Path(args.output)
    if args.sample_dir and not output.is_absolute() and output.parent == Path("."):
        return Path(args.sample_dir) / "visualization" / output
    return resolve_output_path(points_csv, args.output)


def default_points_csv_for_sample(sample_dir: str | Path, signal_ch: str) -> Path:
    _ = signal_ch
    return Path(sample_dir) / "visualization" / "points.csv"


def default_heatmap_volume_for_sample(sample_dir: str | Path) -> Path:
    sample_dir = Path(sample_dir)
    return sample_dir / "visualization" / f"{sample_dir.name}_heatmap3d_volume.tiff"


def default_reference_atlas_image() -> Path:
    return Path(__file__).resolve().parents[2] / "data" / "reference" / "atlas_label.tiff"


def resolve_points_csv(args: argparse.Namespace) -> Path:
    if args.points_csv:
        return Path(args.points_csv)
    if not args.sample_dir:
        raise ValueError("Pass either --points_csv or --sample_dir.")

    sample_dir = Path(args.sample_dir)
    points_csv = default_points_csv_for_sample(sample_dir, args.signal_ch)
    if points_csv.exists() and not args.force_warp:
        print(f"Using existing atlas-space points CSV: {points_csv}")
        return points_csv

    cached_volume = default_heatmap_volume_for_sample(sample_dir)
    if cached_volume.exists() and not args.force_warp:
        print(f"Generating atlas-space points CSV from cached volume: {cached_volume}")
        from pipeline_modules.visualization.warp_mask_zarr_to_atlas_points import (
            atlas_volume_to_points,
            write_outputs,
        )
        import tifffile

        atlas_volume = tifffile.imread(str(cached_volume))
        atlas_resolution = parse_triplet(args.atlas_resolution_xyz or "25,25,25")
        table = atlas_volume_to_points(atlas_volume, atlas_resolution_xyz=atlas_resolution, max_points=150_000)
        write_outputs(
            table,
            {
                "source_volume": str(cached_volume),
                "atlas_resolution_xyz": list(atlas_resolution),
                "exported_points": int(len(table)),
                "coordinate_space": "atlas",
            },
            points_csv,
        )
        return points_csv

    mask_zarr = sample_dir / f"{args.signal_ch}_mask.zarr"
    sample_reference = sample_dir / f"{args.register_ch}_downsample" / "volume.nii.gz"
    output_volume = default_heatmap_volume_for_sample(sample_dir)
    if args.warp_with_micromamba:
        command = [
            "micromamba",
            "run",
            "-n",
            args.warp_env,
            "python",
        ]
    else:
        command = [sys.executable]
    command.extend([
        "-m",
        "pipeline_modules.visualization.warp_mask_zarr_to_atlas_points",
        "--sample_dir",
        str(sample_dir),
        "--signal_ch",
        args.signal_ch,
        "--register_ch",
        args.register_ch,
        "--mask_zarr",
        str(mask_zarr),
        "--sample_reference_nii",
        str(sample_reference),
        "--atlas_image",
        args.atlas_image,
        "--atlas_name",
        args.atlas_name,
        "--atlas_resolution_xyz",
        args.atlas_resolution_xyz or "25,25,25",
        "--resolution_xyz",
        args.resolution_xyz,
        "--target_resolution_xyz",
        args.target_resolution_xyz,
        "--foreground_mode",
        args.foreground_mode,
        "--foreground_label",
        str(args.foreground_label),
        "--output",
        str(points_csv),
        "--output_volume",
        str(output_volume),
    ])
    print(f"Generating atlas-space points CSV: {points_csv}")
    try:
        subprocess.run(command, cwd=Path(__file__).resolve().parents[2], check=True)
    except FileNotFoundError as exc:
        raise RuntimeError(
            "Could not start the point-generation command. "
            "Pass --points_csv if you already generated points, or use --warp_with_micromamba only when micromamba is on PATH."
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"Failed to generate atlas-space points CSV with exit code {exc.returncode}.") from exc

    if not points_csv.exists():
        raise FileNotFoundError(f"Warp command finished but points CSV was not created: {points_csv}")
    return points_csv


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render atlas-space punctate signals as colored points inside a transparent standard brain."
    )
    parser.add_argument("--points_csv", default="", help="CSV containing atlas-space point coordinates.")
    parser.add_argument("--sample_dir", default="", help="Sample root directory. If passed, points are generated automatically.")
    parser.add_argument("--signal_ch", default="ch1", help="Signal channel label used with --sample_dir.")
    parser.add_argument("--register_ch", default="ch0", help="Registration channel label used with --sample_dir.")
    parser.add_argument("--force_warp", action="store_true", help="Regenerate atlas-space points even if the CSV already exists.")
    parser.add_argument("--warp_env", default="yifu", help=argparse.SUPPRESS)
    parser.add_argument("--warp_with_micromamba", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--atlas_image", default=str(default_reference_atlas_image()), help=argparse.SUPPRESS)
    parser.add_argument("--resolution_xyz", default="1.8,1.8,2.0", help=argparse.SUPPRESS)
    parser.add_argument("--target_resolution_xyz", default="25,25,25", help=argparse.SUPPRESS)
    parser.add_argument("--foreground_mode", choices=("nonzero", "equal"), default="equal", help=argparse.SUPPRESS)
    parser.add_argument("--foreground_label", type=int, default=1, help=argparse.SUPPRESS)
    parser.add_argument(
        "--columns",
        default="",
        help="Coordinate columns as x_col,y_col,z_col. If omitted, tries x/y/z or ap/dv/ml style names.",
    )
    parser.add_argument("--atlas_name", default="allen_mouse_25um", help="BrainGlobe atlas name.")
    parser.add_argument("--point_color", default="#ff4d6d", help="Point color.")
    parser.add_argument("--point_alpha", type=float, default=0.95, help="Point opacity from 0 to 1.")
    parser.add_argument("--point_radius", type=float, default=40.0, help="Point sphere radius in atlas units (um).")
    parser.add_argument(
        "--color_by_coarse_region",
        action="store_true",
        help="Color points by the coarse brain-region groups used by coarse_region_metric_plot.py.",
    )
    parser.add_argument(
        "--drop_unassigned_coarse_points",
        action="store_true",
        help="When using --color_by_coarse_region, drop points outside the coarse groups.",
    )
    parser.add_argument("--filter_to_brain", action="store_true", help="Drop points outside the atlas annotation mask.")
    parser.add_argument("--only_region", default="", help="Render only points inside this atlas region acronym/name.")
    parser.add_argument("--show_region", default="", help="Render this atlas region mesh by acronym/name.")
    parser.add_argument("--region_outline", action="store_true", help="Draw an outline around --show_region.")
    parser.add_argument("--hide_whole_brain", action="store_true", help="Hide the transparent whole-brain mesh.")
    parser.add_argument("--region_alpha", type=float, default=0.18, help="Rendered region opacity.")
    parser.add_argument("--region_color", default="#4cc9f0", help="Rendered region color.")
    parser.add_argument("--hemisphere", choices=("both", "left", "right"), default="both", help="Region mesh hemisphere.")
    parser.add_argument("--region", default="", help=argparse.SUPPRESS)
    parser.add_argument("--region_id", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument(
        "--no_region_descendants",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--no_root", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--whole_brain_silhouette",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--region_mesh", default="", help=argparse.SUPPRESS)
    parser.add_argument("--region_mesh_id", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--region_silhouette", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--coordinate_units",
        choices=("auto", "um", "voxel"),
        default="auto",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--axis_order",
        default="",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--atlas_resolution_xyz",
        default="",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--root_alpha", type=float, default=0.12, help="Brain transparency from 0 to 1.")
    parser.add_argument("--background", default="white", help="Background color.")
    parser.add_argument("--camera", default="three_quarters", help="brainrender camera preset name.")
    parser.add_argument("--title", default="", help="Optional scene title.")
    parser.add_argument("--output", default="", help="PNG output path. If omitted, opens an interactive viewer.")
    parser.add_argument("--screenshot_scale", type=int, default=2, help="Screenshot scale factor when --output is used.")
    parser.add_argument("--show_axes", action="store_true", help="Show atlas coordinate axes.")
    return parser


def main() -> int:
    parser = build_argparser()
    args = parser.parse_args()

    columns = parse_columns(args.columns) if args.columns else None
    points_csv = resolve_points_csv(args)
    points = load_points(points_csv, columns)
    ensure_atlas_available(args.atlas_name)
    atlas_resolution = parse_triplet(args.atlas_resolution_xyz) if args.atlas_resolution_xyz else None
    requested_region = args.only_region or args.region
    requested_region_mesh = args.show_region or args.region_mesh
    hide_whole_brain = args.hide_whole_brain or args.no_root
    region_silhouette = args.region_outline or args.region_silhouette

    if args.axis_order:
        points = reorder_axes(points, args.axis_order)
    points, resolved_units = convert_coordinate_units(
        points,
        coordinate_units=args.coordinate_units,
        atlas_name=args.atlas_name,
        atlas_resolution=atlas_resolution,
    )
    if not args.axis_order:
        points, _ = auto_reorder_axes_to_atlas(
            points,
            atlas_name=args.atlas_name,
            atlas_resolution=atlas_resolution,
        )
    if args.filter_to_brain:
        points, kept_points, total_points = filter_points_to_brain(
            points,
            atlas_name=args.atlas_name,
            atlas_resolution=atlas_resolution,
        )
        if len(points) == 0:
            raise ValueError(f"All {total_points} points were outside the atlas annotation mask.")
        print(f"Filtered atlas mask: kept {kept_points}/{total_points} points inside brain.")
    if requested_region or args.region_id is not None:
        points, kept_points, total_points, resolved_region_id, resolved_region = filter_points_to_region(
            points,
            atlas_name=args.atlas_name,
            atlas_resolution=atlas_resolution,
            region=requested_region,
            region_id=args.region_id,
            include_descendants=not args.no_region_descendants,
        )
        if len(points) == 0:
            raise ValueError(f"All {total_points} points were outside region {resolved_region} ({resolved_region_id}).")
        print(f"Filtered region {resolved_region} ({resolved_region_id}): kept {kept_points}/{total_points} points.")
    point_color: str | list[str] = args.point_color
    if args.color_by_coarse_region:
        if args.point_alpha == parser.get_default("point_alpha"):
            args.point_alpha = 0.9
        points, point_color, coarse_counts = coarse_region_colors_for_points(
            points,
            atlas_name=args.atlas_name,
            atlas_resolution=atlas_resolution,
            drop_unassigned=args.drop_unassigned_coarse_points,
        )
        if len(points) == 0:
            raise ValueError("No points remained after coarse region coloring/filtering.")
        atlas_for_names = BrainGlobeAtlas(args.atlas_name, check_latest=False)
        summary = ", ".join(
            f"{region_acronym(atlas_for_names, region_id)}:{count}"
            for region_id, count in sorted(coarse_counts.items(), key=lambda item: DEFAULT_REGION_IDS.index(item[0]))
        )
        print(f"Colored by coarse region groups: {summary}")
    print(f"Loaded {resolved_units} coordinates. Render point bounds: {summarize_points(points)}")

    output_path = resolve_render_output_path(args, points_csv)
    if output_path is not None:
        configure_brainrender(
            background=args.background,
            root_alpha=args.root_alpha,
            show_axes=args.show_axes,
            offscreen=True,
        )
    else:
        configure_brainrender(
            background=args.background,
            root_alpha=args.root_alpha,
            show_axes=args.show_axes,
            offscreen=False,
        )

    scene = render_points_scene(
        points,
        atlas_name=args.atlas_name,
        point_radius=args.point_radius,
        point_color=point_color,
        point_alpha=args.point_alpha,
        root_alpha=args.root_alpha,
        title=args.title or None,
        background=args.background,
        show_axes=args.show_axes,
        root=not hide_whole_brain,
        whole_brain_silhouette=False,
        region_mesh=requested_region_mesh,
        region_mesh_id=args.region_mesh_id,
        region_alpha=args.region_alpha,
        region_color=args.region_color,
        region_silhouette=region_silhouette,
        hemisphere=args.hemisphere,
    )

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        scene.screenshots_folder = output_path.parent
        saved_path = scene.screenshot(
            name=output_path.name,
            scale=args.screenshot_scale,
            camera=args.camera,
        )
        scene.close()
        print(f"Saved brainrender point visualization to: {saved_path}")
    else:
        scene.render(interactive=True, camera=args.camera)
        print("Closed brainrender point viewer.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
