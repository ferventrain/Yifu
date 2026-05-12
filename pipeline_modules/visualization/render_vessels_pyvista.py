"""Render SWC vessel topology as smooth branch curves with branch-point markers.

This renderer ignores varying vessel radius and instead focuses on topology:
- each branch segment between branch points is rendered as one smooth spline
- branch points are shown as small red spheres

This is a cleaner visualization when trace continuity matters more than tube
thickness realism.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class SwcNode:
    node_id: int
    x: float
    y: float
    z: float
    radius: float
    parent_id: int


@dataclass
class Branch:
    points: np.ndarray
    branch_id: int
    mean_radius: float


@dataclass
class TreeData:
    branches: list[Branch]
    branch_points: np.ndarray


def parse_swc(filepath: Path) -> list[SwcNode]:
    nodes: list[SwcNode] = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 7:
                continue
            nodes.append(
                SwcNode(
                    node_id=int(parts[0]),
                    x=float(parts[2]),
                    y=float(parts[3]),
                    z=float(parts[4]),
                    radius=float(parts[5]),
                    parent_id=int(parts[6]),
                )
            )
    return nodes


def swc_to_tree_data(nodes: list[SwcNode], branch_id_start: int = 0) -> TreeData:
    if not nodes:
        return TreeData(branches=[], branch_points=np.empty((0, 3), dtype=np.float64))

    id_to_index = {node.node_id: i for i, node in enumerate(nodes)}
    children: dict[int, list[int]] = {i: [] for i in range(len(nodes))}
    roots: list[int] = []

    for i, node in enumerate(nodes):
        if node.parent_id == -1 or node.parent_id not in id_to_index:
            roots.append(i)
        else:
            children[id_to_index[node.parent_id]].append(i)

    coords = np.array([[node.x, node.y, node.z] for node in nodes], dtype=np.float64)
    radii = np.array([node.radius for node in nodes], dtype=np.float64)

    branch_points_idx = sorted(
        {
            i
            for i, kids in children.items()
            if len(kids) > 1
        }
    )
    branch_points = coords[branch_points_idx] if branch_points_idx else np.empty((0, 3), dtype=np.float64)

    branches: list[Branch] = []
    next_branch_id = branch_id_start

    def trace_path(start_index: int, prepend_index: int | None = None) -> None:
        nonlocal next_branch_id
        path = []
        if prepend_index is not None:
            path.append(prepend_index)
        path.append(start_index)

        current = start_index
        while len(children[current]) == 1:
            current = children[current][0]
            path.append(current)

        if len(path) >= 2:
            branches.append(Branch(points=coords[path], branch_id=next_branch_id, mean_radius=float(np.mean(radii[path]))))
            next_branch_id += 1

        for child in children[current]:
            trace_path(child, prepend_index=current)

    for root in roots:
        if not children[root]:
            continue
        if len(children[root]) == 1:
            trace_path(root)
        else:
            for child in children[root]:
                trace_path(child, prepend_index=root)

    return TreeData(branches=branches, branch_points=branch_points)


def load_tree_data(swc_dir: Path, max_files: int = 0) -> TreeData:
    swc_files = sorted(swc_dir.glob("*.swc"))
    if max_files > 0:
        swc_files = swc_files[:max_files]

    all_branches: list[Branch] = []
    all_branch_points: list[np.ndarray] = []
    branch_id_start = 0

    for swc_path in swc_files:
        tree = swc_to_tree_data(parse_swc(swc_path), branch_id_start=branch_id_start)
        all_branches.extend(tree.branches)
        if len(tree.branch_points) > 0:
            all_branch_points.append(tree.branch_points)
        if tree.branches:
            branch_id_start = max(branch.branch_id for branch in tree.branches) + 1

    branch_points = (
        np.vstack(all_branch_points)
        if all_branch_points
        else np.empty((0, 3), dtype=np.float64)
    )
    return TreeData(branches=all_branches, branch_points=branch_points)


def resample_polyline(points: np.ndarray, samples_per_edge: int) -> np.ndarray:
    if len(points) < 2:
        return points

    seg_lengths = np.linalg.norm(points[1:] - points[:-1], axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(seg_lengths)])
    total_length = cumulative[-1]
    if total_length <= 0:
        return points

    sample_count = max((len(points) - 1) * samples_per_edge + 1, len(points))
    s_values = np.linspace(0.0, total_length, sample_count)
    sampled = np.empty((sample_count, 3), dtype=np.float64)

    seg_idx = 0
    for i, s in enumerate(s_values):
        while seg_idx < len(seg_lengths) - 1 and s > cumulative[seg_idx + 1]:
            seg_idx += 1
        seg_len = max(seg_lengths[seg_idx], 1e-12)
        t = (s - cumulative[seg_idx]) / seg_len
        sampled[i] = points[seg_idx] * (1.0 - t) + points[seg_idx + 1] * t

    return sampled


def _smooth_branch_points(points: np.ndarray, min_spacing: float = 0.0, smooth_window: int = 5) -> np.ndarray:
    """Downsample dense points and apply moving-average smoothing to reduce jitter."""
    if len(points) < 3:
        return points

    if min_spacing > 0:
        kept = [0]
        for i in range(1, len(points)):
            if np.linalg.norm(points[i] - points[kept[-1]]) >= min_spacing:
                kept.append(i)
        if kept[-1] != len(points) - 1:
            kept.append(len(points) - 1)
        points = points[kept]

    if len(points) < 3:
        return points

    if smooth_window > 1 and len(points) > smooth_window:
        smoothed = points.copy()
        half = smooth_window // 2
        for i in range(half, len(points) - half):
            smoothed[i] = points[i - half:i + half + 1].mean(axis=0)
        points = smoothed

    return points


def _get_branch_cmap(cmap_name: str = "neon"):
    from matplotlib import cm
    from matplotlib.colors import LinearSegmentedColormap

    if cmap_name in {"bio", "bio_blue_purple"}:
        return LinearSegmentedColormap.from_list(
            "bio_blue_purple",
            ["#12345f", "#1f5aa6", "#4a6cf0", "#8b5cf6", "#d1a6ff"],
        )
    if cmap_name in {"neon", "neon_fusion"}:
        return LinearSegmentedColormap.from_list(
            "neon_fusion",
            ["#06111f", "#0066ff", "#00d5ff", "#7b5cff", "#ff4fd8", "#ffd166"],
        )
    return cm.get_cmap(cmap_name)


def _branch_colormap(value: float, cmap_name: str = "bio", color_gamma: float = 0.65) -> tuple[float, float, float]:
    value = float(np.clip(value, 0.0, 1.0))
    value = float(np.clip(value ** max(color_gamma, 1e-3), 0.0, 1.0))
    cmap = _get_branch_cmap(cmap_name)
    r, g, b, _ = cmap(value)
    return float(r), float(g), float(b)


def build_curve_mesh(
    branches: list[Branch],
    spline_factor: int,
    smooth_window: int = 5,
    min_spacing: float = 1.0,
    clip_percentile: float = 2.0,
    color_gamma: float = 0.65,
):
    import pyvista as pv

    multiblock = pv.MultiBlock()
    radii = np.array([branch.mean_radius for branch in branches], dtype=np.float64) if branches else np.empty(0)
    if radii.size > 0 and np.isfinite(radii).any():
        lo = float(np.nanpercentile(radii, clip_percentile))
        hi = float(np.nanpercentile(radii, 100.0 - clip_percentile))
        if not np.isfinite(lo):
            lo = float(np.nanmin(radii))
        if not np.isfinite(hi):
            hi = float(np.nanmax(radii))
    else:
        lo = 0.0
        hi = 1.0
    denom = max(hi - lo, 1e-12)

    for branch in branches:
        pts = _smooth_branch_points(branch.points, min_spacing=min_spacing, smooth_window=smooth_window)
        if len(pts) < 2:
            continue
        n_spline = max(len(pts) * spline_factor, 20)
        try:
            spline = pv.Spline(pts, n_spline)
        except Exception:
            continue
        spline["branch_id"] = np.full(spline.n_points, branch.branch_id, dtype=np.int32)
        spline["branch_radius"] = np.full(spline.n_points, branch.mean_radius, dtype=np.float64)
        radius_norm = (branch.mean_radius - lo) / denom
        spline["branch_radius_norm"] = np.full(spline.n_points, np.clip(radius_norm, 0.0, 1.0), dtype=np.float64)
        multiblock.append(spline)

    return multiblock


def build_branch_point_mesh(branch_points: np.ndarray, point_size: float):
    import pyvista as pv

    if len(branch_points) == 0:
        return pv.PolyData()

    spheres = pv.MultiBlock()
    for center in branch_points:
        spheres.append(pv.Sphere(radius=point_size, center=center, theta_resolution=18, phi_resolution=18))
    return spheres.combine()


def render_scene(
    curve_blocks,
    branch_point_mesh,
    *,
    line_width: float,
    branch_color: str,
    color_mode: str,
    cmap_name: str,
    color_gamma: float,
    background: str,
    output: str | None,
    dpi: int,
    window_size: tuple[int, int],
):
    import pyvista as pv

    plotter = pv.Plotter(window_size=window_size, off_screen=bool(output))
    try:
        plotter.set_background("#08111f", top="#172338")
    except TypeError:
        plotter.set_background("#08111f")
    if background == "white":
        plotter.set_background("#f5f7fb")

    scalar_bar_added = False
    cmap = _get_branch_cmap(cmap_name)
    scalar_bar_args = {
        "title": "Thickness",
        "vertical": False,
        "position_x": 0.75,
        "position_y": 0.05,
        "width": 0.18,
        "height": 0.08,
        "title_font_size": 20,
        "label_font_size": 20,
        "n_labels": 4,
        "fmt": "%.2f",
    }

    for i in range(curve_blocks.n_blocks):
        curve = curve_blocks[i]
        if curve is None or curve.n_points == 0:
            continue

        if color_mode == "branch":
            value = float(curve["branch_radius_norm"][0]) if "branch_radius_norm" in curve.array_names else 0.0
            color = _branch_colormap(value, cmap_name=cmap_name, color_gamma=color_gamma)
            plotter.add_mesh(
                curve,
                scalars="branch_radius_norm",
                cmap=cmap,
                clim=(0.0, 1.0),
                show_scalar_bar=not scalar_bar_added,
                scalar_bar_args=scalar_bar_args if not scalar_bar_added else None,
                line_width=line_width,
                render_lines_as_tubes=True,
                smooth_shading=True,
                opacity=0.98,
                pbr=True,
                metallic=0.06,
                roughness=0.28,
                specular=0.5,
                specular_power=42,
                diffuse=0.9,
                ambient=0.24,
            )
            scalar_bar_added = True
        else:
            color = branch_color
            plotter.add_mesh(
                curve,
                color=color,
                line_width=line_width,
                render_lines_as_tubes=True,
                smooth_shading=True,
                opacity=0.98,
                pbr=True,
                metallic=0.06,
                roughness=0.28,
                specular=0.5,
                specular_power=42,
                diffuse=0.9,
                ambient=0.24,
            )

    if branch_point_mesh.n_points > 0:
        plotter.add_mesh(
            branch_point_mesh,
            color="#d24d57",
            smooth_shading=True,
            opacity=0.92,
            specular=0.45,
            specular_power=24,
            ambient=0.22,
        )

    try:
        plotter.enable_anti_aliasing("ssaa")
    except Exception:
        pass
    try:
        plotter.enable_eye_dome_lighting()
    except Exception:
        pass

    plotter.add_light(pv.Light(position=(1, 1, 1), intensity=1.1, color=(1.0, 0.98, 0.96)))
    plotter.add_light(pv.Light(position=(-1, -1, 0.5), intensity=0.55, color=(0.72, 0.82, 1.0)))
    plotter.add_light(pv.Light(position=(0.0, 0.0, 1.5), intensity=0.28, color=(1.0, 1.0, 1.0)))
    plotter.camera.zoom(1.2)

    if output:
        plotter.screenshot(str(Path(output)), scale=max(1, dpi // 72))
        print(f"Saved to {output}")
        plotter.close()
    else:
        plotter.show()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render SWC branches as smooth curves with red branch points")
    parser.add_argument("--swc_dir", required=True)
    parser.add_argument("--max_files", type=int, default=0)
    parser.add_argument("--spline_factor", type=int, default=12, help="Smoothing density along each branch")
    parser.add_argument("--smooth_window", type=int, default=5, help="Moving-average window size for jitter removal")
    parser.add_argument("--min_spacing", type=float, default=1.0, help="Minimum spacing between control points (um)")
    parser.add_argument("--line_width", type=float, default=14.0, help="Displayed curve width")
    parser.add_argument("--branch_point_size", type=float, default=3, help="Radius of red branch-point markers")
    parser.add_argument("--color_mode", choices=["solid", "branch"], default="branch", help="Use one solid color or a different color per branch")
    parser.add_argument("--branch_color", default="deepskyblue")
    parser.add_argument("--colormap", default="neon", help="Matplotlib colormap for thickness coloring (neon = vivid cyan→violet→pink)")
    parser.add_argument("--clip_percentile", type=float, default=2.0, help="Clip branch thickness using low/high percentiles")
    parser.add_argument("--color_gamma", type=float, default=0.35, help="Boost color contrast; lower = stronger contrast")
    parser.add_argument("--background", choices=["black", "white"], default="black")
    parser.add_argument("--output", default=None)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--window_size", default="1920,1080")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    swc_dir = Path(args.swc_dir)
    if not swc_dir.is_dir():
        print(f"Error: {swc_dir} is not a directory", file=sys.stderr)
        return 1

    tree = load_tree_data(swc_dir, max_files=args.max_files)
    print(f"Loaded {len(tree.branches)} branches and {len(tree.branch_points)} branch points")

    curve_blocks = build_curve_mesh(
        tree.branches,
        spline_factor=args.spline_factor,
        smooth_window=args.smooth_window,
        min_spacing=args.min_spacing,
        clip_percentile=args.clip_percentile,
        color_gamma=args.color_gamma,
    )
    branch_point_mesh = build_branch_point_mesh(tree.branch_points, point_size=args.branch_point_size)

    window_size = tuple(int(x) for x in args.window_size.split(","))
    render_scene(
        curve_blocks,
        branch_point_mesh,
        line_width=args.line_width,
        branch_color=args.branch_color,
        color_mode=args.color_mode,
        cmap_name=args.colormap,
        color_gamma=args.color_gamma,
        background=args.background,
        output=args.output,
        dpi=args.dpi,
        window_size=window_size,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
