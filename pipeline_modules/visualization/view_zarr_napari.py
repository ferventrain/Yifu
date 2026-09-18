"""Open a Zarr volume in napari.

Pyramid stores (datasets ``0``, ``1``, ``2``, ...) are opened with
``napari-ome-zarr``. Single-array stores, including 4D QC stacks, are loaded
directly so incomplete NGFF metadata cannot crash the viewer.

CLI::

    python -m pipeline_modules.visualization.view_zarr_napari S:/path/volume.zarr
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _looks_like_ome_pyramid(path: Path) -> bool:
    try:
        import zarr
    except ModuleNotFoundError:
        return False
    try:
        root = zarr.open_group(str(path), mode="r")
    except Exception:
        return False
    attrs = dict(root.attrs)
    if "ome" in attrs and isinstance(attrs["ome"], dict):
        attrs = attrs["ome"]
    scales = attrs.get("multiscales") or []
    if not scales:
        return False
    datasets = scales[0].get("datasets") or []
    if len(datasets) < 2:
        return False
    first = datasets[0]
    has_transform = bool(first.get("coordinateTransformations")) or bool(scales[0].get("axes"))
    return has_transform


def _open_in_viewer(viewer, path: Path, plugin: str | None) -> None:
    if plugin:
        viewer.open(str(path), plugin=plugin)
        return
    if _looks_like_ome_pyramid(path):
        try:
            viewer.open(str(path), plugin="napari-ome-zarr")
            return
        except Exception as exc:
            print(f"napari-ome-zarr failed ({exc}); opening array 0 directly", file=sys.stderr)

    try:
        from pipeline_modules.utils.zarr_io import open_zarr_array
    except ImportError:  # pragma: no cover
        from ..utils.zarr_io import open_zarr_array

    arr = open_zarr_array(path)
    attrs = dict(getattr(arr, "attrs", {}) or {})
    names = attrs.get("channel_names")
    downsample = float(attrs.get("downsample") or 1.0)
    scale = (downsample, downsample, downsample) if downsample > 1 else None
    kwargs: dict = {"name": Path(path).stem}
    if scale is not None:
        kwargs["scale"] = scale
    if arr.ndim == 4:
        kwargs["channel_axis"] = int(attrs.get("channel_axis") or 0)
        if isinstance(names, (list, tuple)) and names:
            kwargs["name"] = list(names)
        viewer.add_image(arr, **kwargs)
        return
    viewer.add_image(arr, **kwargs)


def main() -> int:
    parser = argparse.ArgumentParser(description="Open a Zarr volume in napari")
    parser.add_argument("zarr", help="Path to a .zarr store")
    parser.add_argument(
        "--plugin",
        default=None,
        help="Force a napari reader plugin. Default: ome-zarr for pyramids, direct array otherwise.",
    )
    args = parser.parse_args()
    path = Path(args.zarr)

    try:
        import napari
    except ModuleNotFoundError:
        print("napari is not installed in this environment", file=sys.stderr)
        return 1

    viewer = napari.Viewer()
    _open_in_viewer(viewer, path, args.plugin)
    napari.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
