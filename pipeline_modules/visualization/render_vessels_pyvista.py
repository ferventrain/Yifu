"""Render SWC vessel skeletons as smooth 3D topology curves with PyVista."""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np


PAPER_BACKGROUND = "#fbfaf5"
PAPER_TEXT = "#252525"
PAPER_FRAME = "#7b7f84"
PAPER_BRANCH_POINT = "#8f2432"
PAPER_DIAMETER_CMAP = [
    "#12355b",
    "#1f6f8b",
    "#2fa58d",
    "#8ecf6b",
    "#f2c14e",
]
DARK_BACKGROUND = "#0b1626"
DARK_TEXT = "#edf4fb"
DARK_FRAME = "#9fb0c4"
DARK_BRANCH_POINT = "#ff8a8a"
BRIGHT_DIAMETER_CMAP = [
    "#4cc9f0",
    "#72efdd",
    "#c7f9cc",
    "#fff3b0",
    "#f4a261",
    "#e76f51",
]


@dataclass(frozen=True)
class SwcNode:
    node_id: int
    x: float
    y: float
    z: float
    radius: float
    parent_id: int

    @property
    def point(self) -> np.ndarray:
        return np.asarray([self.x, self.y, self.z], dtype=np.float64)


@dataclass(frozen=True)
class Branch:
    points: np.ndarray
    radii: np.ndarray
    source_id: int
    target_id: int
    swc_path: Path

    @property
    def mean_diameter_um(self) -> float:
        valid = self.radii[np.isfinite(self.radii) & (self.radii > 0)]
        if valid.size == 0:
            return 0.0
        return float(np.mean(valid) * 2.0)


def parse_window_size(value: str) -> tuple[int, int]:
    parts = [part.strip() for part in str(value).split(",") if part.strip()]
    if len(parts) != 2:
        raise ValueError(f"window_size must be width,height, got: {value}")
    width, height = int(parts[0]), int(parts[1])
    if width <= 0 or height <= 0:
        raise ValueError(f"window_size values must be positive, got: {value}")
    return width, height


def load_swc(path: str | Path) -> dict[int, SwcNode]:
    path = Path(path)
    nodes: dict[int, SwcNode] = {}

    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split()
            if len(parts) < 7:
                raise ValueError(f"Malformed SWC line {line_no} in {path}: {raw_line.rstrip()}")

            node_id = int(float(parts[0]))
            nodes[node_id] = SwcNode(
                node_id=node_id,
                x=float(parts[2]),
                y=float(parts[3]),
                z=float(parts[4]),
                radius=float(parts[5]),
                parent_id=int(float(parts[6])),
            )

    if not nodes:
        raise ValueError(f"No SWC nodes found in {path}")
    return nodes


def build_children(nodes: dict[int, SwcNode]) -> dict[int, list[int]]:
    children: dict[int, list[int]] = defaultdict(list)
    for node in nodes.values():
        if node.parent_id in nodes:
            children[node.parent_id].append(node.node_id)
    return children


def node_degree(node_id: int, nodes: dict[int, SwcNode], children: dict[int, list[int]]) -> int:
    parent_count = 1 if nodes[node_id].parent_id in nodes else 0
    return parent_count + len(children.get(node_id, []))


def extract_branches(nodes: dict[int, SwcNode], swc_path: Path) -> tuple[list[Branch], list[np.ndarray]]:
    """Collapse degree-2 chains into topology branches."""
    children = build_children(nodes)
    endpoint_ids = {
        node_id
        for node_id in nodes
        if node_degree(node_id, nodes, children) != 2
    }
    branch_point_coords = [
        nodes[node_id].point
        for node_id in nodes
        if len(children.get(node_id, [])) > 1
    ]

    branches: list[Branch] = []
    visited_edges: set[tuple[int, int]] = set()

    for start_id in sorted(endpoint_ids):
        neighbor_ids = []
        parent_id = nodes[start_id].parent_id
        if parent_id in nodes:
            neighbor_ids.append(parent_id)
        neighbor_ids.extend(children.get(start_id, []))

        for next_id in sorted(neighbor_ids):
            edge_key = tuple(sorted((start_id, next_id)))
            if edge_key in visited_edges:
                continue

            path_ids = [start_id, next_id]
            visited_edges.add(edge_key)
            previous_id, current_id = start_id, next_id

            while current_id not in endpoint_ids:
                neighbors = []
                parent_id = nodes[current_id].parent_id
                if parent_id in nodes:
                    neighbors.append(parent_id)
                neighbors.extend(children.get(current_id, []))

                forward = [node_id for node_id in neighbors if node_id != previous_id]
                if not forward:
                    break

                following_id = forward[0]
                visited_edges.add(tuple(sorted((current_id, following_id))))
                path_ids.append(following_id)
                previous_id, current_id = current_id, following_id

            points = np.stack([nodes[node_id].point for node_id in path_ids], axis=0)
            radii = np.asarray([nodes[node_id].radius for node_id in path_ids], dtype=np.float64)
            branches.append(
                Branch(
                    points=points,
                    radii=radii,
                    source_id=path_ids[0],
                    target_id=path_ids[-1],
                    swc_path=swc_path,
                )
            )

    return branches, branch_point_coords


def filter_min_spacing(points: np.ndarray, min_spacing: float) -> np.ndarray:
    if min_spacing <= 0 or len(points) <= 2:
        return points

    kept = [points[0]]
    for point in points[1:-1]:
        if np.linalg.norm(point - kept[-1]) >= min_spacing:
            kept.append(point)
    kept.append(points[-1])
    return np.asarray(kept, dtype=np.float64)


def smooth_points(points: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(points) <= 2:
        return points

    if window % 2 == 0:
        window += 1

    radius = window // 2
    smoothed = points.copy()
    for idx in range(1, len(points) - 1):
        start = max(0, idx - radius)
        stop = min(len(points), idx + radius + 1)
        smoothed[idx] = points[start:stop].mean(axis=0)
    smoothed[0] = points[0]
    smoothed[-1] = points[-1]
    return smoothed


def prepare_branch_points(
    points: np.ndarray,
    *,
    min_spacing: float,
    smooth_window: int,
) -> np.ndarray:
    prepared = filter_min_spacing(points, min_spacing=min_spacing)
    return smooth_points(prepared, window=smooth_window)


def discover_swc_files(swc_dir: str | Path, max_files: int = 0) -> list[Path]:
    swc_path = Path(swc_dir)
    if not swc_path.exists():
        raise FileNotFoundError(f"SWC directory not found: {swc_path}")
    if not swc_path.is_dir():
        raise NotADirectoryError(f"SWC input must be a directory: {swc_path}")

    files = sorted(swc_path.glob("*.swc"))
    if max_files and max_files > 0:
        files = files[: int(max_files)]
    if not files:
        raise FileNotFoundError(f"No .swc files found in {swc_path}")
    return files


def load_branches(swc_dir: str | Path, max_files: int = 0) -> tuple[list[Branch], np.ndarray]:
    all_branches: list[Branch] = []
    branch_points: list[np.ndarray] = []

    for swc_file in discover_swc_files(swc_dir, max_files=max_files):
        nodes = load_swc(swc_file)
        branches, points = extract_branches(nodes, swc_file)
        all_branches.extend(branches)
        branch_points.extend(points)

    if not all_branches:
        raise ValueError(f"No renderable branches found in {swc_dir}")

    branch_point_array = (
        np.stack(branch_points, axis=0)
        if branch_points
        else np.empty((0, 3), dtype=np.float64)
    )
    return all_branches, branch_point_array


def color_for_branch(index: int, total: int, *, color_mode: str, branch_color: str, colormap: str):
    if color_mode == "solid":
        return branch_color

    import matplotlib.pyplot as plt

    cmap = plt.get_cmap(colormap)
    value = index / max(1, total - 1)
    rgba = cmap(value)
    return tuple(float(v) for v in rgba[:3])


def diameter_clim(
    branches: list[Branch],
    clip_percentiles: tuple[float, float] = (2.0, 98.0),
) -> tuple[float, float]:
    diameters = np.asarray([branch.mean_diameter_um for branch in branches], dtype=np.float64)
    valid = diameters[np.isfinite(diameters) & (diameters > 0)]
    if valid.size == 0:
        return 0.0, 1.0

    lo_pct, hi_pct = clip_percentiles
    if 0 <= lo_pct < hi_pct <= 100:
        lo = float(np.percentile(valid, lo_pct))
        hi = float(np.percentile(valid, hi_pct))
    else:
        lo = float(np.min(valid))
        hi = float(np.max(valid))
    if hi <= lo:
        pad = max(0.5, abs(lo) * 0.1)
        return lo - pad, hi + pad
    return lo, hi


def parse_clip_percentiles(value: str | tuple[float, float]) -> tuple[float, float]:
    if isinstance(value, tuple):
        lo, hi = value
    else:
        parts = [part.strip() for part in str(value).split(",") if part.strip()]
        if len(parts) != 2:
            raise ValueError(f"diameter_clip_percentiles must be low,high, got: {value}")
        lo, hi = float(parts[0]), float(parts[1])
    if not (0 <= lo < hi <= 100):
        raise ValueError(f"diameter_clip_percentiles must satisfy 0 <= low < high <= 100, got: {value}")
    return float(lo), float(hi)


def apply_camera_preset(plotter, preset: str) -> None:
    preset = str(preset).lower()
    if preset == "iso":
        plotter.camera_position = "iso"
    elif preset == "xy":
        plotter.view_xy()
    elif preset == "xz":
        plotter.view_xz()
    elif preset == "yz":
        plotter.view_yz()
    else:
        raise ValueError(f"Unknown camera preset: {preset}")
    plotter.reset_camera()


def all_branch_bounds(branches: list[Branch], padding_fraction: float = 0.035) -> tuple[float, float, float, float, float, float]:
    points = np.concatenate([branch.points for branch in branches], axis=0)
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    spans = np.maximum(maxs - mins, 1.0)
    pad = spans * float(padding_fraction)
    mins = mins - pad
    maxs = maxs + pad
    return (
        float(mins[0]),
        float(maxs[0]),
        float(mins[1]),
        float(maxs[1]),
        float(mins[2]),
        float(maxs[2]),
    )


def resolve_background_color(background: str) -> str:
    if background == "dark":
        return DARK_BACKGROUND
    if background == "paper":
        return PAPER_BACKGROUND
    if background == "white":
        return "#ffffff"
    if background == "black":
        return "#000000"
    return background


def color_to_rgb01(color: str) -> tuple[float, float, float]:
    color = resolve_background_color(color)
    if not color.startswith("#") or len(color) != 7:
        raise ValueError(f"Expected a #RRGGBB color, got: {color}")
    return (
        int(color[1:3], 16) / 255.0,
        int(color[3:5], 16) / 255.0,
        int(color[5:7], 16) / 255.0,
    )


def resolve_colormap(colormap: str):
    from matplotlib.colors import LinearSegmentedColormap

    if colormap == "paper":
        return LinearSegmentedColormap.from_list(
            "yifu_paper_diameter",
            PAPER_DIAMETER_CMAP,
            N=256,
        )
    if colormap == "bright":
        return LinearSegmentedColormap.from_list(
            "yifu_bright_diameter",
            BRIGHT_DIAMETER_CMAP,
            N=256,
        )
    return colormap


def text_color_for_background(background: str) -> str:
    return DARK_TEXT if background in {"black", "dark"} else PAPER_TEXT


def frame_color_for_background(background: str) -> str:
    return DARK_FRAME if background in {"black", "dark"} else PAPER_FRAME


def branch_point_color_for_background(background: str) -> str:
    return DARK_BRANCH_POINT if background in {"black", "dark"} else PAPER_BRANCH_POINT


def render_vessels(
    *,
    swc_dir: str | Path,
    max_files: int = 0,
    spline_factor: int = 12,
    smooth_window: int = 5,
    min_spacing: float = 1.0,
    line_width: float = 5.0,
    branch_point_size: float = 2.0,
    color_mode: str = "diameter",
    branch_color: str = "#0072b2",
    colormap: str = "bright",
    background: str = "dark",
    diameter_clip_percentiles: tuple[float, float] = (2.0, 98.0),
    output: str | Path | None = None,
    dpi: int = 300,
    window_size: tuple[int, int] = (1920, 1080),
    show_frame: bool = True,
    frame_line_width: float = 5,
    camera: str = "iso",
    parallel_projection: bool = False,
    save_on_close: str | Path | None = None,
) -> dict[str, object]:
    try:
        import pyvista as pv
    except ImportError as exc:
        raise ImportError("pyvista is required for rendering. Install pyvista in the active environment.") from exc

    background_color = resolve_background_color(background)
    background_rgb = color_to_rgb01(background)
    cmap = resolve_colormap(colormap)

    pv.global_theme.background = background_color
    pv.global_theme.window_size = window_size

    branches, branch_points = load_branches(swc_dir, max_files=max_files)

    if output:
        pv.OFF_SCREEN = True

    plotter = pv.Plotter(off_screen=bool(output), window_size=window_size)
    plotter.set_background(background_color, top=background_color)
    for renderer in plotter.renderers:
        renderer.SetBackground(*background_rgb)
    try:
        plotter.enable_anti_aliasing("ssaa")
    except Exception:
        pass
    frame_color = frame_color_for_background(background)

    clim = diameter_clim(branches, clip_percentiles=diameter_clip_percentiles)
    scalar_bar_args = {
        "title": "Diameter (um)",
        "position_x": 0.80,
        "position_y": 0.055,
        "width": 0.14,
        "height": 0.022,
        "n_labels": 4,
        "color": text_color_for_background(background),
        "title_font_size": 10,
        "label_font_size": 8,
        "fmt": "%.2g",
    }
    scalar_bar_added = False
    rendered_branches = 0
    for index, branch in enumerate(branches):
        points = prepare_branch_points(
            branch.points,
            min_spacing=min_spacing,
            smooth_window=smooth_window,
        )
        if len(points) < 2:
            continue

        if len(points) >= 3 and spline_factor > 1:
            n_points = max(len(points) * int(spline_factor), len(points))
            mesh = pv.Spline(points, n_points)
        else:
            mesh = pv.lines_from_points(points)

        if color_mode == "diameter":
            mesh.point_data["diameter_um"] = np.full(
                mesh.n_points,
                np.clip(branch.mean_diameter_um, clim[0], clim[1]),
                dtype=np.float64,
            )
            plotter.add_mesh(
                mesh,
                scalars="diameter_um",
                cmap=cmap,
                clim=clim,
                line_width=line_width,
                render_lines_as_tubes=True,
                show_scalar_bar=not scalar_bar_added,
                scalar_bar_args=scalar_bar_args,
                ambient=0.12,
                diffuse=0.78,
                specular=0.32,
                specular_power=32,
            )
            scalar_bar_added = True
        else:
            plotter.add_mesh(
                mesh,
                color=color_for_branch(
                    index,
                    len(branches),
                    color_mode=color_mode,
                    branch_color=branch_color,
                    colormap=colormap if colormap != "paper" else "viridis",
                ),
                line_width=line_width,
                render_lines_as_tubes=True,
                ambient=0.12,
                diffuse=0.78,
                specular=0.32,
                specular_power=32,
            )
        rendered_branches += 1

    if len(branch_points) > 0 and branch_point_size > 0:
        cloud = pv.PolyData(branch_points)
        cloud["branch_point_size"] = np.full(len(branch_points), float(branch_point_size), dtype=np.float64)
        sphere = pv.Sphere(radius=float(branch_point_size), theta_resolution=16, phi_resolution=16)
        glyphs = cloud.glyph(geom=sphere, scale="branch_point_size", orient=False, factor=1.0)
        plotter.add_mesh(
            glyphs,
            color=branch_point_color_for_background(background),
            smooth_shading=True,
            ambient=0.18,
            diffuse=0.68,
            specular=0.38,
            specular_power=36,
        )

    if show_frame:
        plotter.add_mesh(
            pv.Box(bounds=all_branch_bounds(branches)),
            style="wireframe",
            color=frame_color,
            line_width=frame_line_width,
        )
    apply_camera_preset(plotter, camera)
    if parallel_projection:
        plotter.enable_parallel_projection()

    output_path = None
    if output:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        scale = max(1, int(round(float(dpi) / 100.0)))
        plotter.screenshot(
            str(output_path),
            window_size=window_size,
            scale=scale,
            transparent_background=False,
        )
        plotter.close()
    elif save_on_close:
        output_path = Path(save_on_close)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        scale = max(1, int(round(float(dpi) / 100.0)))

        def save_current_view():
            plotter.screenshot(
                str(output_path),
                window_size=window_size,
                scale=scale,
                transparent_background=False,
            )
            print(f"Saved current view to: {output_path}")

        plotter.add_text(
            "Press S to save current view",
            position="upper_left",
            font_size=10,
            color=text_color_for_background(background),
        )
        plotter.add_key_event("s", save_current_view)
        plotter.show()
    else:
        plotter.show()

    return {
        "swc_dir": str(swc_dir),
        "num_branches": int(len(branches)),
        "rendered_branches": int(rendered_branches),
        "num_branch_points": int(len(branch_points)),
        "diameter_clim_um": [float(clim[0]), float(clim[1])],
        "output": str(output_path) if output_path else "",
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render SWC vessel skeletons as smooth 3D branch curves with PyVista."
    )
    parser.add_argument("--swc_dir", required=True, help="Directory containing .swc files")
    parser.add_argument("--max_files", type=int, default=0, help="Limit number of SWC files loaded (0 = all)")
    parser.add_argument("--spline_factor", type=int, default=12, help="Interpolation density multiplier")
    parser.add_argument("--smooth_window", type=int, default=5, help="Moving-average smoothing window")
    parser.add_argument("--min_spacing", type=float, default=1.0, help="Minimum point spacing in microns")
    parser.add_argument("--line_width", type=float, default=5.0, help="Rendered curve line width")
    parser.add_argument("--branch_point_size", type=float, default=2.0, help="Red branch-point sphere radius in microns")
    parser.add_argument("--color_mode", choices=("diameter", "solid", "branch"), default="diameter")
    parser.add_argument("--branch_color", default="#0072b2", help="Color used when color_mode=solid")
    parser.add_argument("--colormap", default="bright", help="Matplotlib colormap used for branch coloring")
    parser.add_argument("--background", choices=("dark", "paper", "black", "white"), default="dark", help="Scene background")
    parser.add_argument(
        "--diameter_clip_percentiles",
        default="2,98",
        help="Percentile range used for diameter color clipping as low,high. Use 0,100 to disable clipping.",
    )
    parser.add_argument("--output", default=None, help="Save screenshot to this path instead of opening a viewer")
    parser.add_argument("--dpi", type=int, default=300, help="Screenshot resolution scale")
    parser.add_argument("--window_size", default="1920,1080", help="Window size as width,height")
    parser.add_argument("--no_frame", action="store_true", help="Disable the scene bounding frame")
    parser.add_argument("--frame_line_width", type=float, default=2.4, help="Scene frame line width")
    parser.add_argument("--camera", choices=("iso", "xy", "xz", "yz"), default="iso", help="Camera preset for non-interactive rendering")
    parser.add_argument("--parallel_projection", action="store_true", help="Use orthographic instead of perspective projection")
    parser.add_argument(
        "--save_on_close",
        default=None,
        help="Interactive mode only: press S in the viewer to save the current view to this path",
    )
    parser.add_argument(
        "--interactive_output",
        default=None,
        help="Interactive mode only: press S in the viewer to save the current view to this path",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        result = render_vessels(
            swc_dir=args.swc_dir,
            max_files=args.max_files,
            spline_factor=args.spline_factor,
            smooth_window=args.smooth_window,
            min_spacing=args.min_spacing,
            line_width=args.line_width,
            branch_point_size=args.branch_point_size,
            color_mode=args.color_mode,
            branch_color=args.branch_color,
            colormap=args.colormap,
            background=args.background,
            diameter_clip_percentiles=parse_clip_percentiles(args.diameter_clip_percentiles),
            output=args.output,
            dpi=args.dpi,
            window_size=parse_window_size(args.window_size),
            show_frame=not args.no_frame,
            frame_line_width=args.frame_line_width,
            camera=args.camera,
            parallel_projection=args.parallel_projection,
            save_on_close=args.interactive_output or args.save_on_close,
        )
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if result["output"]:
        print(f"Saved vessel render to: {result['output']}")
    else:
        print("Closed PyVista vessel renderer.")
    print(
        "Rendered "
        f"{result['rendered_branches']} branches with {result['num_branch_points']} branch points."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
