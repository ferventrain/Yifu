"""Render atlas-space punctate signals inside a transparent standard brain with brainrender."""

from __future__ import annotations

import argparse
import configparser
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
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
from brainrender.camera import check_camera_param  # noqa: E402
from brainglobe_atlasapi.bg_atlas import BrainGlobeAtlas  # noqa: E402
from brainglobe_atlasapi.list_atlases import get_downloaded_atlases  # noqa: E402
from pipeline_modules.visualization.coarse_region_metric_plot import DEFAULT_REGION_IDS  # noqa: E402


DEFAULT_OUTPUT = "brainrender_points.png"
DEFAULT_REGION_GROUPS = Path(__file__).resolve().parents[2] / "config" / "region_groups.json"
DEFAULT_CAMERA_VIEW_ARG = "__default__"
DEFAULT_CAMERA_VIEW_FILENAME = "{sample_name}_brainrender_view.json"
DEFAULT_COLUMNS = ("x", "y", "z")
ALIAS_GROUPS: dict[str, tuple[str, ...]] = {
    "x": ("x", "ap", "anterior_posterior", "anteriorposterior"),
    "y": ("y", "dv", "dorsal_ventral", "dorsoventral"),
    "z": ("z", "ml", "lr", "mediolateral", "left_right", "left-right"),
}
@dataclass(frozen=True)
class RegionGroup:
    name: str
    acronyms: tuple[str, ...]
    color: str | None = None
    description: str = ""


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


def normalize_background(background: str) -> str:
    """Map CLI background names to brainrender BACKGROUND_COLOR values."""
    value = str(background or "white").strip().lower()
    aliases = {
        "white": "white",
        "w": "white",
        "black": "black",
        "k": "black",
        "dark": "black",
    }
    if value in aliases:
        return aliases[value]
    return str(background).strip()


def parse_color_rgb(value: str) -> list[float] | str:
    """Accept a named color, #hex, or comma-separated 0-1 / 0-255 RGB."""
    text = str(value or "").strip()
    if not text:
        raise ValueError("Empty color value.")
    if "," in text:
        parts = [float(part.strip()) for part in text.split(",")]
        if len(parts) != 3:
            raise ValueError(f"Expected 3 RGB components, got {len(parts)}: {value!r}")
        if max(parts) > 1.0:
            parts = [channel / 255.0 for channel in parts]
        return [max(0.0, min(1.0, channel)) for channel in parts]
    return text


def default_root_color(background: str) -> list[float]:
    """Brighter outline on dark scenes; mid-gray on white so the silhouette stays readable."""
    bg = normalize_background(background)
    if bg == "black":
        return [0.92, 0.94, 0.98]
    return [0.72, 0.72, 0.74]


def configure_brainrender(
    *,
    background: str,
    root_alpha: float,
    show_axes: bool,
    offscreen: bool,
    shader_style: str = "glossy",
    root_color: str | list[float] | None = None,
) -> None:
    bg = normalize_background(background)
    settings.BACKGROUND_COLOR = bg
    settings.ROOT_ALPHA = float(root_alpha)
    if root_color is None or root_color == "":
        settings.ROOT_COLOR = default_root_color(bg)
    elif isinstance(root_color, (list, tuple)):
        settings.ROOT_COLOR = [float(channel) for channel in root_color]
    else:
        settings.ROOT_COLOR = parse_color_rgb(str(root_color))
    settings.SHADER_STYLE = shader_style
    settings.SHOW_AXES = bool(show_axes)
    settings.OFFSCREEN = bool(offscreen)
    settings.WHOLE_SCREEN = False
    settings.INTERACTIVE = not offscreen
    settings.DEFAULT_CAMERA = "three_quarters"


def install_lighting_controls(
    scene: Scene,
    *,
    light_intensity: float = 1.0,
    ambient: float | None = None,
) -> None:
    """Scale renderer lights and optionally override material ambient after style is applied."""
    intensity = float(light_intensity)
    if intensity <= 0:
        raise ValueError(f"--light_intensity must be > 0, got {light_intensity}")
    if ambient is not None and not (0.0 <= float(ambient) <= 1.0):
        raise ValueError(f"--ambient must be in [0, 1], got {ambient}")

    if abs(intensity - 1.0) < 1e-9 and ambient is None:
        return

    original_apply_style = scene._apply_style
    base_intensities: dict[int, float] = {}

    def _apply_style_with_lighting() -> None:
        original_apply_style()
        if ambient is not None:
            ambient_value = float(ambient)
            for actor in scene.clean_actors:
                try:
                    actor.mesh.properties.SetAmbient(ambient_value)
                except Exception:
                    pass
                try:
                    actor._mesh.properties.SetAmbient(ambient_value)
                except Exception:
                    pass
        if abs(intensity - 1.0) < 1e-9:
            return
        if scene.plotter is None:
            scene._get_plotter()
        renderers = list(getattr(scene.plotter, "renderers", None) or [])
        if not renderers and getattr(scene.plotter, "renderer", None) is not None:
            renderers = [scene.plotter.renderer]
        for ren in renderers:
            lights = ren.GetLights()
            lights.InitTraversal()
            light = lights.GetNextItem()
            while light is not None:
                key = id(light)
                if key not in base_intensities:
                    base_intensities[key] = float(light.GetIntensity())
                light.SetIntensity(base_intensities[key] * intensity)
                light = lights.GetNextItem()

    scene._apply_style = _apply_style_with_lighting


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
    acronyms = lookup["acronym"].astype(str)
    names = lookup["name"].astype(str)

    # Prefer exact acronym (case-sensitive) so CM != cm in Allen atlas.
    exact_acronym = lookup[acronyms == query]
    if len(exact_acronym) == 1:
        return int(exact_acronym.iloc[0]["id"])

    exact_name = lookup[names == query]
    if len(exact_name) == 1:
        return int(exact_name.iloc[0]["id"])

    normalized = lookup.assign(
        acronym_norm=acronyms.map(normalize_name),
        name_norm=names.map(normalize_name),
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
        # If case-insensitive search is ambiguous, fall back to exact acronym among matches.
        exact_among = matches[matches["acronym"].astype(str) == query]
        if len(exact_among) == 1:
            return int(exact_among.iloc[0]["id"])
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


def parse_group_names(value: str) -> list[str] | None:
    parts = [part.strip() for part in str(value).split(",") if part.strip()]
    return parts or None


def parse_group_colors(value: str, count: int) -> list[str]:
    parts = [part.strip() for part in str(value).split(",") if part.strip()]
    if not parts:
        return [COARSE_REGION_COLORS[index % len(COARSE_REGION_COLORS)] for index in range(count)]
    if len(parts) != count:
        raise ValueError(f"Expected {count} group colors, got {len(parts)}.")
    return parts


def load_region_groups(path: str | Path) -> dict[str, RegionGroup]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Region groups JSON not found: {path}")

    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"Region groups JSON must be an object, got: {type(payload).__name__}")

    groups: dict[str, RegionGroup] = {}
    for index, (name, spec) in enumerate(payload.items()):
        if isinstance(spec, list):
            acronyms = tuple(str(item).strip() for item in spec if str(item).strip())
            color = COARSE_REGION_COLORS[index % len(COARSE_REGION_COLORS)]
            description = ""
        elif isinstance(spec, dict):
            raw_acronyms = spec.get("acronyms", [])
            if not isinstance(raw_acronyms, list):
                raise ValueError(f"Group {name!r} acronyms must be a list.")
            acronyms = tuple(str(item).strip() for item in raw_acronyms if str(item).strip())
            color = str(spec.get("color") or COARSE_REGION_COLORS[index % len(COARSE_REGION_COLORS)])
            description = str(spec.get("description") or "")
        else:
            raise ValueError(f"Group {name!r} must be a list of acronyms or an object with acronyms.")
        if not acronyms:
            raise ValueError(f"Group {name!r} has no atlas acronyms.")
        groups[str(name)] = RegionGroup(name=str(name), acronyms=acronyms, color=color, description=description)
    return groups


def select_region_groups(
    groups: dict[str, RegionGroup],
    group_names: list[str] | None,
    group_colors: list[str] | None = None,
) -> list[RegionGroup]:
    if group_names is None:
        selected = list(groups.values())
    else:
        missing = [name for name in group_names if name not in groups]
        if missing:
            raise KeyError(f"Unknown region group(s): {missing}. Available: {sorted(groups)}")
        selected = [groups[name] for name in group_names]
    if group_colors is None:
        return selected
    return [
        RegionGroup(name=group.name, acronyms=group.acronyms, color=color, description=group.description)
        for group, color in zip(selected, group_colors)
    ]


def group_region_ids(
    atlas: BrainGlobeAtlas,
    groups: list[RegionGroup],
    *,
    include_descendants: bool,
) -> dict[str, set[int]]:
    ids_by_group: dict[str, set[int]] = {}
    for group in groups:
        ids: set[int] = set()
        for acronym in group.acronyms:
            region_id = resolve_region_id(atlas, acronym, None)
            if region_id is None:
                continue
            ids.update(descendant_region_ids(atlas, region_id, include_descendants))
        ids_by_group[group.name] = ids
    return ids_by_group


def filter_points_to_groups(
    points: np.ndarray,
    *,
    atlas_name: str,
    atlas_resolution: tuple[float, float, float] | None,
    groups: list[RegionGroup],
    include_descendants: bool,
) -> tuple[np.ndarray, dict[str, int], int]:
    atlas = BrainGlobeAtlas(atlas_name, check_latest=False)
    ids_by_group = group_region_ids(atlas, groups, include_descendants=include_descendants)
    union_ids = set().union(*ids_by_group.values()) if ids_by_group else set()

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
    inside[in_bounds] = np.isin(labels, list(union_ids))
    filtered = points[inside]

    counts: dict[str, int] = {}
    if len(filtered) > 0:
        filtered_indices = np.floor(filtered / resolution[None, :]).astype(np.int64)
        filtered_labels = annotation[
            filtered_indices[:, 0], filtered_indices[:, 1], filtered_indices[:, 2]
        ]
        for group in groups:
            counts[group.name] = int(np.count_nonzero(np.isin(filtered_labels, list(ids_by_group[group.name]))))
    else:
        counts = {group.name: 0 for group in groups}
    return filtered, counts, int(len(points))


def filter_points_to_single_group(
    points: np.ndarray,
    *,
    atlas_name: str,
    atlas_resolution: tuple[float, float, float] | None,
    group: RegionGroup,
    include_descendants: bool,
) -> tuple[np.ndarray, int, int]:
    filtered, counts, total = filter_points_to_groups(
        points,
        atlas_name=atlas_name,
        atlas_resolution=atlas_resolution,
        groups=[group],
        include_descendants=include_descendants,
    )
    kept = counts.get(group.name, 0)
    return filtered, kept, total


def color_points_by_groups(
    points: np.ndarray,
    *,
    atlas_name: str,
    atlas_resolution: tuple[float, float, float] | None,
    groups: list[RegionGroup],
    include_descendants: bool,
    drop_unassigned: bool,
) -> tuple[np.ndarray, list[str], dict[str, int]]:
    atlas = BrainGlobeAtlas(atlas_name, check_latest=False)
    labels = point_annotation_labels(points, atlas=atlas, atlas_resolution=atlas_resolution)
    ids_by_group = group_region_ids(atlas, groups, include_descendants=include_descendants)
    color_by_group = {group.name: group.color or "#9a9a9a" for group in groups}

    colors: list[str] = []
    assigned = np.zeros(len(points), dtype=bool)
    counts: dict[str, int] = {group.name: 0 for group in groups}

    for index, label in enumerate(labels):
        label_id = int(label)
        matched_group: str | None = None
        if label_id > 0:
            for group in groups:
                if label_id in ids_by_group[group.name]:
                    matched_group = group.name
                    break
        if matched_group is None:
            colors.append("#9a9a9a")
            continue
        colors.append(str(color_by_group[matched_group]))
        assigned[index] = True
        counts[matched_group] += 1

    if drop_unassigned:
        points = points[assigned]
        colors = [color for color, keep in zip(colors, assigned) if keep]
        counts = {name: count for name, count in counts.items() if count > 0}
    else:
        counts = {name: count for name, count in counts.items() if count > 0}
    return points, colors, counts


def _clean_camera_value(val):
    if isinstance(val, tuple):
        return tuple(round(float(v), 4) if isinstance(v, float) else int(round(v)) for v in val)
    if isinstance(val, float):
        return round(float(val), 4)
    return val


def extract_camera_params(scene: Scene) -> dict:
    """Capture camera params including ViewAngle (mouse-wheel zoom).

    brainrender's get_camera_params omits ViewAngle, so interactive zoom is lost
    when replaying a saved camera JSON into offscreen screenshots.
    """
    if not scene.is_rendered:
        scene.render(interactive=False)
    cam = scene.plotter.camera
    params = {
        "pos": _clean_camera_value(cam.GetPosition()),
        "focal_point": _clean_camera_value(cam.GetFocalPoint()),
        "viewup": _clean_camera_value(cam.GetViewUp()),
        "distance": _clean_camera_value(cam.GetDistance()),
        "clipping_range": _clean_camera_value(cam.GetClippingRange()),
        "view_angle": _clean_camera_value(cam.GetViewAngle()),
    }
    window_size = getattr(scene.plotter, "window_size", None) or getattr(scene.plotter, "size", None)
    if window_size is not None:
        try:
            params["window_size"] = [int(window_size[0]), int(window_size[1])]
        except Exception:
            pass
    return params


def apply_camera_params(scene: Scene, camera: str | dict) -> dict:
    """Apply camera dict/preset, including optional view_angle from our JSON format."""
    from brainrender.camera import set_camera_params

    if isinstance(camera, str):
        params = check_camera_param(camera)
    else:
        params = dict(camera)
        check_camera_param(params)

    if scene.plotter is None:
        scene._get_plotter()
    set_camera_params(scene.plotter.camera, params)
    view_angle = params.get("view_angle")
    if view_angle is not None:
        scene.plotter.camera.SetViewAngle(float(view_angle))
    return params


def load_camera_view(path: str | Path) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Camera view JSON not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Camera view JSON must be an object, got: {type(payload).__name__}")
    # Keep extended keys (view_angle, window_size) while validating required fields.
    validated = check_camera_param(dict(payload))
    for key, value in payload.items():
        validated.setdefault(key, value)
    return validated


def save_camera_view(path: str | Path, params: dict, *, name: str | None = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(params)
    if name:
        payload["name"] = name
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def resolve_camera(camera_preset: str, camera_view: str, *, sample_dir: Path | None) -> str | dict:
    camera_path = resolve_camera_view_path(camera_view, sample_dir)
    if camera_path is not None:
        return load_camera_view(camera_path)
    return camera_preset


def infer_sample_dir(*, sample_dir: str, points_csv: Path) -> Path | None:
    if sample_dir.strip():
        return Path(sample_dir).resolve()
    csv_path = points_csv.resolve()
    if csv_path.parent.name.lower() == "visualization":
        return csv_path.parent.parent
    return None


def default_camera_view_path(sample_dir: Path) -> Path:
    sample_dir = sample_dir.resolve()
    return sample_dir / "visualization" / DEFAULT_CAMERA_VIEW_FILENAME.format(sample_name=sample_dir.name)


def resolve_camera_view_path(value: str, sample_dir: Path | None) -> Path | None:
    if not value:
        return None
    if value == DEFAULT_CAMERA_VIEW_ARG:
        if sample_dir is None:
            raise ValueError(
                "Could not infer sample_dir for the default camera view path. "
                "Pass --sample_dir, or use --points_csv located under sample_dir/visualization/."
            )
        return default_camera_view_path(sample_dir)
    return Path(value)


def install_camera_export_hook(scene: Scene, output_path: str | Path) -> None:
    """Register a vedo KeyPress callback to save the current camera on V."""
    output_path = Path(output_path)
    original_render = scene.render

    def render(*args, **kwargs):
        if scene.plotter is None:
            scene._get_plotter()
        plotter = scene.plotter

        def on_key_press(evt) -> None:
            if str(getattr(evt, "keypress", "")).lower() != "v":
                return
            params = extract_camera_params(scene)
            saved = save_camera_view(output_path, params, name=output_path.stem)
            print(f"Saved camera view to: {saved}")
            if "view_angle" in params:
                print(f"  view_angle={params['view_angle']} (mouse-wheel zoom)")
            print("Reuse with: --camera_view", saved)

        original_show = plotter.show

        def show_with_hook(*show_args, **show_kwargs):
            original_init = plotter.initialize_interactor

            def init_with_hook(*init_args, **init_kwargs):
                original_init(*init_args, **init_kwargs)
                plotter.add_callback("KeyPress", on_key_press)

            plotter.initialize_interactor = init_with_hook
            try:
                return original_show(*show_args, **show_kwargs)
            finally:
                plotter.initialize_interactor = original_init
                plotter.show = original_show

        plotter.show = show_with_hook
        return original_render(*args, **kwargs)

    scene.render = render


def _actor_list(actors) -> list:
    if actors is None:
        return []
    return actors if isinstance(actors, list) else [actors]


def _mesh_x_bounds(actors: list) -> tuple[float, float]:
    bounds: list[np.ndarray] = []
    for actor in actors:
        mesh = getattr(actor, "_mesh", None) or getattr(actor, "mesh", None)
        if mesh is None:
            continue
        bounds.append(np.asarray(mesh.bounds(), dtype=np.float64).reshape((3, 2))[0])
    if not bounds:
        raise ValueError("Could not determine region mesh bounds for coronal slab clipping.")
    stacked = np.vstack(bounds)
    return float(np.min(stacked[:, 0])), float(np.max(stacked[:, 1]))


def _cut_actor_between_x(actor, x_start: float, x_stop: float) -> None:
    mesh = actor._mesh
    cut = getattr(mesh, "cut_with_plane", None) or getattr(mesh, "cutWithPlane", None)
    if cut is None:
        raise AttributeError("Actor mesh does not support cut_with_plane/cutWithPlane.")
    cut(origin=(x_start, 0, 0), normal=(1, 0, 0))
    cut(origin=(x_stop, 0, 0), normal=(-1, 0, 0))
    actor.cap()


def clip_root_to_region_coronal_slab(scene: Scene, region_actors) -> tuple[float, float]:
    slab_start, slab_stop = _mesh_x_bounds(_actor_list(region_actors))
    if slab_start >= slab_stop:
        raise ValueError(
            f"Invalid coronal slab bounds from region mesh: start={slab_start:.1f}, stop={slab_stop:.1f}."
        )
    _cut_actor_between_x(scene.root, slab_start, slab_stop)
    return slab_start, slab_stop


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
    region_groups: list[RegionGroup] | None = None,
    region_coronal_slab: bool = False,
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

    if region_groups:
        atlas = BrainGlobeAtlas(atlas_name, check_latest=False)
        for group in region_groups:
            mesh_color = group.color or region_color or None
            for acronym in group.acronyms:
                resolved_id = resolve_region_id(atlas, acronym, None)
                if resolved_id is None:
                    continue
                actors = scene.add_brain_region(
                    region_acronym(atlas, resolved_id),
                    alpha=region_alpha,
                    color=mesh_color,
                    silhouette=region_silhouette,
                    hemisphere=hemisphere,
                    force=True,
                )
                if region_coronal_slab:
                    clip_root_to_region_coronal_slab(scene, actors)
    elif region_mesh or region_mesh_id is not None:
        atlas = BrainGlobeAtlas(atlas_name, check_latest=False)
        resolved_id = resolve_region_id(atlas, region_mesh, region_mesh_id)
        if resolved_id is None:
            raise ValueError("--region_mesh or --region_mesh_id is required to render a region mesh.")
        actors = scene.add_brain_region(
            region_acronym(atlas, resolved_id),
            alpha=region_alpha,
            color=region_color or None,
            silhouette=region_silhouette,
            hemisphere=hemisphere,
            force=True,
        )
        if region_coronal_slab:
            slab_start, slab_stop = clip_root_to_region_coronal_slab(scene, actors)
            print(
                f"Coronal slab clipped whole-brain outline from {region_acronym(atlas, resolved_id)} bounds: "
                f"x={slab_start:.1f}..{slab_stop:.1f} um"
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
    if args.screenshot_per_group:
        return None
    if not args.output:
        return None
    output = Path(args.output)
    if args.sample_dir and not output.is_absolute() and output.parent == Path("."):
        return Path(args.sample_dir) / "visualization" / output
    return resolve_output_path(points_csv, args.output)


def resolve_output_dir(args: argparse.Namespace, points_csv: Path, sample_dir: Path | None) -> Path:
    if args.output_dir:
        output_dir = Path(args.output_dir)
    elif args.output:
        output_dir = Path(args.output).parent
    elif sample_dir is not None:
        output_dir = sample_dir / "visualization" / "brainrender"
    else:
        output_dir = points_csv.parent / "brainrender"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def default_points_csv_for_sample(sample_dir: str | Path, signal_ch: str) -> Path:
    _ = signal_ch
    return Path(sample_dir) / "visualization" / "points.csv"


def default_heatmap_volume_for_sample(sample_dir: str | Path, signal_ch: str = "ch1") -> Path:
    from pipeline_modules.utils.deliverable_paths import heatmap_3d_volume_tiff

    return heatmap_3d_volume_tiff(sample_dir, signal_ch)


def default_reference_atlas_image() -> Path:
    return Path(__file__).resolve().parents[2] / "data" / "reference" / "atlas_label.tiff"


def default_mask_atlas_volume_for_sample(sample_dir: str | Path, signal_ch: str) -> Path:
    return Path(sample_dir) / "visualization" / f"{signal_ch}_mask_atlas_volume.tiff"


def default_mask_atlas_volume_meta_for_sample(sample_dir: str | Path, signal_ch: str) -> Path:
    return Path(sample_dir) / "visualization" / f"{signal_ch}_mask_atlas_volume.json"


def generate_points_csv_from_atlas_volume(
    *,
    atlas_volume_path: Path,
    atlas_resolution_xyz: tuple[float, float, float],
    points_csv: Path,
    source_label: str,
) -> Path:
    from pipeline_modules.visualization.warp_mask_zarr_to_atlas_points import (
        atlas_volume_to_points,
        write_outputs,
    )
    import tifffile

    atlas_volume = tifffile.imread(str(atlas_volume_path))
    table = atlas_volume_to_points(
        atlas_volume,
        atlas_resolution_xyz=atlas_resolution_xyz,
        max_points=150_000,
    )
    write_outputs(
        table,
        {
            "source_volume": str(atlas_volume_path),
            "atlas_resolution_xyz": list(atlas_resolution_xyz),
            "exported_points": int(len(table)),
            "coordinate_space": "atlas",
        },
        points_csv,
    )
    print(f"Generated atlas-space points CSV from {source_label}: {points_csv} ({len(table)} points)")
    return points_csv


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
    mask_atlas_volume = default_mask_atlas_volume_for_sample(sample_dir, args.signal_ch)
    mask_atlas_meta = default_mask_atlas_volume_meta_for_sample(sample_dir, args.signal_ch)
    if mask_atlas_volume.exists() and mask_atlas_meta.exists() and not args.force_warp:
        meta = json.loads(mask_atlas_meta.read_text(encoding="utf-8"))
        atlas_resolution = tuple(float(value) for value in meta.get("atlas_resolution_xyz", (25.0, 25.0, 25.0)))
        if len(atlas_resolution) != 3:
            raise ValueError(f"Invalid atlas_resolution_xyz in {mask_atlas_meta}: {atlas_resolution}")
        return generate_points_csv_from_atlas_volume(
            atlas_volume_path=mask_atlas_volume,
            atlas_resolution_xyz=atlas_resolution,
            points_csv=points_csv,
            source_label=str(mask_atlas_volume),
        )

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
        "--bin_workers",
        str(getattr(args, "bin_workers", 0) or 0),
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
    parser.add_argument("--points_csv", default="", help="Atlas-space point CSV. With --sample_dir, defaults to sample_dir/visualization/points.csv.")
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
    parser.add_argument(
        "--region_coronal_slab",
        action="store_true",
        help="Clip the whole-brain outline to the selected region's x/AP coronal extent.",
    )
    parser.add_argument("--hide_whole_brain", action="store_true", help="Hide the transparent whole-brain mesh.")
    parser.add_argument("--region_alpha", type=float, default=0.35, help="Rendered region opacity.")
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
    parser.add_argument("--root_alpha", type=float, default=0.25, help="Brain transparency from 0 to 1.")
    parser.add_argument(
        "--root_color",
        default="",
        help=(
            "Whole-brain outline color: name, #hex, or R,G,B (0-1 or 0-255). "
            "Default is brighter near-white on black backgrounds."
        ),
    )
    parser.add_argument(
        "--light_intensity",
        type=float,
        default=1.0,
        help="Scale scene light brightness (1.0=default, try 1.5–2.5 to brighten).",
    )
    parser.add_argument(
        "--ambient",
        type=float,
        default=None,
        help="Override material ambient fill in [0, 1] (higher = flatter, brighter fill light).",
    )
    parser.add_argument(
        "--shader_style",
        choices=("glossy", "shiny", "plastic", "metallic", "cartoon"),
        default="glossy",
        help=(
            "Mesh lighting style. glossy≈glass highlight (default); shiny=bright specular; "
            "plastic=matte; metallic=metal; cartoon=flat + black silhouette edges."
        ),
    )
    parser.add_argument(
        "--background",
        choices=("white", "black"),
        default="white",
        help="Scene background: white (default) or black.",
    )
    parser.add_argument("--camera", default="three_quarters", help="brainrender camera preset name.")
    parser.add_argument(
        "--camera_view",
        nargs="?",
        const=DEFAULT_CAMERA_VIEW_ARG,
        default="",
        metavar="PATH",
        help="Load camera JSON. Omit PATH to use sample_dir/visualization/{sample}_brainrender_view.json.",
    )
    parser.add_argument(
        "--export_camera_view",
        nargs="?",
        const=DEFAULT_CAMERA_VIEW_ARG,
        default="",
        metavar="PATH",
        help="Interactive export. Omit PATH to save to sample_dir/visualization/{sample}_brainrender_view.json.",
    )
    parser.add_argument(
        "--region_groups",
        default="",
        help=f"JSON file defining named atlas region groups (default: {DEFAULT_REGION_GROUPS}).",
    )
    parser.add_argument(
        "--group_names",
        default="",
        help="Comma-separated subset of region group names to render (default: all groups in JSON).",
    )
    parser.add_argument(
        "--group_colors",
        default="",
        help="Comma-separated colors overriding group colors for the selected groups.",
    )
    parser.add_argument(
        "--filter_points_by_group",
        action="store_true",
        help="Keep only points inside the selected region groups.",
    )
    parser.add_argument(
        "--color_points_by_group",
        action="store_true",
        help="Color points by the selected region groups.",
    )
    parser.add_argument(
        "--drop_unassigned_group_points",
        action="store_true",
        help="When using --color_points_by_group, drop points outside the selected groups.",
    )
    parser.add_argument(
        "--screenshot_per_group",
        action="store_true",
        help="Save one PNG per selected region group using the same camera view.",
    )
    parser.add_argument(
        "--output_dir",
        default="",
        help="Output directory for --screenshot_per_group (default: sample visualization/brainrender).",
    )
    parser.add_argument("--title", default="", help="Optional scene title.")
    parser.add_argument("--output", default="", help="PNG output path. If omitted, opens an interactive viewer.")
    parser.add_argument("--screenshot_scale", type=int, default=2, help="Screenshot scale factor when --output is used.")
    parser.add_argument("--show_axes", action="store_true", help="Show atlas coordinate axes.")
    return parser


def save_scene_screenshot(
    scene: Scene,
    output_path: Path,
    *,
    camera: str | dict,
    screenshot_scale: int,
) -> str:
    """Render offscreen and save PNG, preserving extended camera fields like view_angle."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scene.screenshots_folder = output_path.parent

    # Force a controlled render path so we can re-apply ViewAngle after brainrender's
    # set_camera_params (which only knows pos/viewup/clipping/focal/distance).
    if not scene.is_rendered:
        scene.render(interactive=False, camera=camera, zoom=1)
    if isinstance(camera, dict):
        apply_camera_params(scene, camera)

    print(f"\nSaving new screenshot at {output_path.name}\n")
    savepath = str(output_path)
    scene.plotter.screenshot(filename=savepath, scale=screenshot_scale)
    return savepath


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
    if args.region_coronal_slab and not requested_region_mesh and args.region_mesh_id is None:
        raise ValueError("--region_coronal_slab requires --show_region or --region_mesh_id.")
    if args.region_coronal_slab and hide_whole_brain:
        raise ValueError("--region_coronal_slab clips the whole-brain outline, so do not combine it with --hide_whole_brain.")

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
        if args.filter_points_by_group or args.color_points_by_group or args.screenshot_per_group:
            raise ValueError("Use either --only_region or region-group options, not both.")
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

    use_region_groups = bool(
        args.region_groups
        or args.group_names
        or args.filter_points_by_group
        or args.color_points_by_group
        or args.screenshot_per_group
    )
    selected_groups: list[RegionGroup] = []
    if use_region_groups:
        groups_path = Path(args.region_groups) if args.region_groups else DEFAULT_REGION_GROUPS
        all_groups = load_region_groups(groups_path)
        group_names = parse_group_names(args.group_names)
        group_colors = parse_group_colors(args.group_colors, len(group_names or all_groups)) if args.group_colors else None
        selected_groups = select_region_groups(all_groups, group_names, group_colors)
        print(
            "Region groups: "
            + ", ".join(f"{group.name}({','.join(group.acronyms)})" for group in selected_groups)
        )
        if args.filter_points_by_group or args.screenshot_per_group:
            points, group_counts, total_points = filter_points_to_groups(
                points,
                atlas_name=args.atlas_name,
                atlas_resolution=atlas_resolution,
                groups=selected_groups,
                include_descendants=not args.no_region_descendants,
            )
            if len(points) == 0:
                raise ValueError(f"All {total_points} points were outside the selected region groups.")
            summary = ", ".join(f"{name}:{count}" for name, count in group_counts.items())
            print(f"Filtered region groups: kept {len(points)}/{total_points} points. Per group: {summary}")

    point_color: str | list[str] = args.point_color
    if args.color_by_coarse_region and args.color_points_by_group:
        raise ValueError("Use either --color_by_coarse_region or --color_points_by_group, not both.")
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
    elif args.color_points_by_group:
        if not selected_groups:
            raise ValueError("--color_points_by_group requires --region_groups or --group_names.")
        if args.point_alpha == parser.get_default("point_alpha"):
            args.point_alpha = 0.9
        points, point_color, group_counts = color_points_by_groups(
            points,
            atlas_name=args.atlas_name,
            atlas_resolution=atlas_resolution,
            groups=selected_groups,
            include_descendants=not args.no_region_descendants,
            drop_unassigned=args.drop_unassigned_group_points,
        )
        if len(points) == 0:
            raise ValueError("No points remained after region-group coloring/filtering.")
        summary = ", ".join(f"{name}:{count}" for name, count in group_counts.items())
        print(f"Colored by region groups: {summary}")
    print(f"Loaded {resolved_units} coordinates. Render point bounds: {summarize_points(points)}")

    sample_dir = infer_sample_dir(sample_dir=args.sample_dir, points_csv=points_csv)
    camera = resolve_camera(args.camera, args.camera_view, sample_dir=sample_dir)
    export_camera_path = resolve_camera_view_path(args.export_camera_view, sample_dir)
    if export_camera_path is not None:
        print(f"Camera view path: {export_camera_path}")
    output_path = resolve_render_output_path(args, points_csv)
    interactive_mode = output_path is None and not args.screenshot_per_group
    if export_camera_path is not None:
        interactive_mode = True
    if args.screenshot_per_group and export_camera_path is not None:
        raise ValueError("Use either --export_camera_view or --screenshot_per_group, not both.")

    configure_brainrender(
        background=args.background,
        root_alpha=args.root_alpha,
        show_axes=args.show_axes,
        offscreen=not interactive_mode,
        shader_style=args.shader_style,
        root_color=args.root_color or None,
    )

    region_groups_for_mesh = selected_groups if selected_groups else None
    if region_groups_for_mesh and requested_region_mesh:
        print("Using --region_groups meshes; ignoring --show_region.")
        requested_region_mesh = ""

    if args.screenshot_per_group:
        if not selected_groups:
            raise ValueError("--screenshot_per_group requires --region_groups or --group_names.")
        output_dir = resolve_output_dir(args, points_csv, sample_dir)
        base_points = points
        for group in selected_groups:
            group_points, kept, total = filter_points_to_single_group(
                base_points,
                atlas_name=args.atlas_name,
                atlas_resolution=atlas_resolution,
                group=group,
                include_descendants=not args.no_region_descendants,
            )
            if len(group_points) == 0:
                print(f"Skipping {group.name}: no points inside group ({total} input points).")
                continue
            scene = render_points_scene(
                group_points,
                atlas_name=args.atlas_name,
                point_radius=args.point_radius,
                point_color=group.color or args.point_color,
                point_alpha=args.point_alpha,
                root_alpha=args.root_alpha,
                title=args.title or group.name,
                background=args.background,
                show_axes=args.show_axes,
                root=not hide_whole_brain,
                whole_brain_silhouette=False,
                region_mesh="",
                region_mesh_id=None,
                region_alpha=args.region_alpha,
                region_color=args.region_color,
                region_silhouette=region_silhouette,
                hemisphere=args.hemisphere,
                region_groups=[group],
                region_coronal_slab=args.region_coronal_slab,
            )
            install_lighting_controls(
                scene,
                light_intensity=args.light_intensity,
                ambient=args.ambient,
            )
            output_file = output_dir / f"{group.name}_brainrender.png"
            saved_path = save_scene_screenshot(
                scene,
                output_file,
                camera=camera,
                screenshot_scale=args.screenshot_scale,
            )
            scene.close()
            print(f"Saved {group.name}: {kept}/{total} points -> {saved_path}")
        return 0

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
        region_groups=region_groups_for_mesh,
        region_coronal_slab=args.region_coronal_slab,
    )
    install_lighting_controls(
        scene,
        light_intensity=args.light_intensity,
        ambient=args.ambient,
    )

    if export_camera_path is not None:
        install_camera_export_hook(scene, export_camera_path)
        print(
            "Camera export mode: adjust the view, press V to save "
            f"{export_camera_path}, Shift+C to print camera dict, Q/Esc to close."
        )
        scene.render(interactive=True, camera=camera)
        print("Closed brainrender point viewer.")
        return 0

    if output_path is not None:
        saved_path = save_scene_screenshot(
            scene,
            output_path,
            camera=camera,
            screenshot_scale=args.screenshot_scale,
        )
        scene.close()
        print(f"Saved brainrender point visualization to: {saved_path}")
    else:
        scene.render(interactive=True, camera=camera)
        print("Closed brainrender point viewer.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
