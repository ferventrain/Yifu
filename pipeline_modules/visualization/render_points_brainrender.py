"""Render atlas-space punctate signals inside a transparent standard brain with brainrender."""

from __future__ import annotations

import argparse
import configparser
import os
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
from brainglobe_atlasapi.list_atlases import get_downloaded_atlases  # noqa: E402


DEFAULT_OUTPUT = "brainrender_points.png"
DEFAULT_COLUMNS = ("x", "y", "z")
ALIAS_GROUPS: dict[str, tuple[str, ...]] = {
    "x": ("x", "ap", "anterior_posterior", "anteriorposterior"),
    "y": ("y", "dv", "dorsal_ventral", "dorsoventral"),
    "z": ("z", "ml", "lr", "mediolateral", "left_right", "left-right"),
}


def parse_columns(value: str) -> tuple[str, str, str]:
    parts = tuple(part.strip() for part in str(value).split(",") if part.strip())
    if len(parts) != 3:
        raise ValueError(f"--columns must contain exactly 3 column names, got: {value}")
    return parts[0], parts[1], parts[2]


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


def render_points_scene(
    points: np.ndarray,
    *,
    atlas_name: str,
    point_radius: float,
    point_color: str,
    point_alpha: float,
    root_alpha: float,
    title: str | None,
    background: str,
    show_axes: bool,
) -> Scene:
    scene = Scene(
        root=True,
        atlas_name=atlas_name,
        check_latest=False,
        inset=False,
        title=title,
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


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render atlas-space punctate signals as colored points inside a transparent standard brain."
    )
    parser.add_argument("--points_csv", required=True, help="CSV containing atlas-space point coordinates.")
    parser.add_argument(
        "--columns",
        default="",
        help="Coordinate columns as x_col,y_col,z_col. If omitted, tries x/y/z or ap/dv/ml style names.",
    )
    parser.add_argument("--atlas_name", default="allen_mouse_25um", help="BrainGlobe atlas name.")
    parser.add_argument("--point_color", default="#ff4d6d", help="Point color.")
    parser.add_argument("--point_alpha", type=float, default=0.95, help="Point opacity from 0 to 1.")
    parser.add_argument("--point_radius", type=float, default=40.0, help="Point sphere radius in atlas units (um).")
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
    points_csv = Path(args.points_csv)
    points = load_points(points_csv, columns)
    ensure_atlas_available(args.atlas_name)

    output_path = resolve_output_path(points_csv, args.output or None) if args.output else None
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
        point_color=args.point_color,
        point_alpha=args.point_alpha,
        root_alpha=args.root_alpha,
        title=args.title or None,
        background=args.background,
        show_axes=args.show_axes,
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
