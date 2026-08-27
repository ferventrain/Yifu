from __future__ import annotations

import argparse
import json
import pathlib
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_CROP_SIZE = (256, 256, 256)
PREVIEW_LAYER_NAME = "crop_preview"
PREVIEW_EDGE_COLOR_VALID = "cyan"
PREVIEW_EDGE_COLOR_INVALID = "orange"
PREVIEW_EDGE_WIDTH = 2


def intended_crop_bounds(
    point_zyx: tuple[float, float, float] | list[float] | np.ndarray,
    crop_size: tuple[int, int, int] = DEFAULT_CROP_SIZE,
) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    """Return [start, stop) crop bounds without validating against image shape."""
    if len(point_zyx) < 3:
        raise ValueError("point_zyx must contain z,y,x coordinates")
    start = tuple(int(round(float(value))) for value in point_zyx[:3])
    crop = tuple(int(value) for value in crop_size)
    if any(value <= 0 for value in crop):
        raise ValueError("crop_size values must be positive")
    stop = tuple(start[axis] + crop[axis] for axis in range(3))
    return start, stop


def crop_bounds_in_shape(
    start: tuple[int, int, int],
    stop: tuple[int, int, int],
    shape: tuple[int, int, int],
) -> bool:
    if len(shape) != 3:
        raise ValueError("Only 3D images are supported")
    if any(value < 0 for value in start):
        return False
    return all(stop[axis] <= int(shape[axis]) for axis in range(3))


def crop_bounds_from_point(
    point_zyx: tuple[float, float, float] | list[float] | np.ndarray,
    shape: tuple[int, int, int],
    crop_size: tuple[int, int, int] = DEFAULT_CROP_SIZE,
) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    if len(shape) != 3:
        raise ValueError("Only 3D images are supported")

    start, stop = intended_crop_bounds(point_zyx, crop_size)
    if any(value < 0 for value in start):
        raise ValueError(f"Crop start must be non-negative, got {start}")
    if not crop_bounds_in_shape(start, stop, shape):
        raise ValueError(
            f"Crop {start}->{stop} exceeds image shape {tuple(int(v) for v in shape)}. "
            "Move the point farther from the lower/right/back edge or reduce crop size."
        )
    return start, stop


def crop_box_face_rects(
    start: tuple[int, int, int] | list[int],
    stop: tuple[int, int, int] | list[int],
) -> list[np.ndarray]:
    """Build 6 rectangular faces for a [start, stop) crop box in z,y,x order."""
    z0, y0, x0 = (float(value) for value in start)
    z1, y1, x1 = (float(value) for value in stop)
    faces = (
        ((z0, y0, x0), (z0, y0, x1), (z0, y1, x1), (z0, y1, x0)),
        ((z1, y0, x0), (z1, y0, x1), (z1, y1, x1), (z1, y1, x0)),
        ((z0, y0, x0), (z0, y0, x1), (z1, y0, x1), (z1, y0, x0)),
        ((z0, y1, x0), (z0, y1, x1), (z1, y1, x1), (z1, y1, x0)),
        ((z0, y0, x0), (z0, y1, x0), (z1, y1, x0), (z1, y0, x0)),
        ((z0, y0, x1), (z0, y1, x1), (z1, y1, x1), (z1, y0, x1)),
    )
    return [np.asarray(face, dtype=np.float64) for face in faces]


def preview_shapes_from_points(
    points_zyx: np.ndarray | list[list[float]],
    crop_size: tuple[int, int, int],
    shape: tuple[int, int, int] | None = None,
) -> tuple[list[np.ndarray], list[str]]:
    """Return rectangle faces and edge colors for each crop preview box."""
    points = np.asarray(points_zyx, dtype=np.float64)
    if points.size == 0:
        return [], []
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError("points_zyx must be an (N, >=3) array of z,y,x coordinates")

    shapes: list[np.ndarray] = []
    edge_colors: list[str] = []
    for point in points:
        start, stop = intended_crop_bounds(point[:3], crop_size)
        valid = True if shape is None else crop_bounds_in_shape(start, stop, shape)
        color = PREVIEW_EDGE_COLOR_VALID if valid else PREVIEW_EDGE_COLOR_INVALID
        for face in crop_box_face_rects(start, stop):
            shapes.append(face)
            edge_colors.append(color)
    return shapes, edge_colors


def crop_array_from_point(
    image_data: Any,
    point_zyx: tuple[float, float, float] | list[float] | np.ndarray,
    crop_size: tuple[int, int, int] = DEFAULT_CROP_SIZE,
) -> tuple[np.ndarray, dict[str, Any]]:
    shape = tuple(int(value) for value in image_data.shape[:3])
    start, stop = crop_bounds_from_point(point_zyx, shape, crop_size)
    slices = tuple(slice(start[axis], stop[axis]) for axis in range(3))
    crop = np.asarray(image_data[slices])
    metadata = {
        "start_zyx": list(start),
        "stop_zyx": list(stop),
        "crop_size_zyx": [int(value) for value in crop_size],
        "source_shape_zyx": list(shape),
    }
    return crop, metadata


def _selected_point(points_layer: Any) -> np.ndarray:
    selected = sorted(getattr(points_layer, "selected_data", set()))
    data = np.asarray(points_layer.data)
    if data.size == 0:
        raise ValueError("Points layer has no points. Add a point first.")
    if selected:
        return np.asarray(data[selected[-1]], dtype=np.float64)
    if len(data) == 1:
        return np.asarray(data[0], dtype=np.float64)
    raise ValueError("Select exactly one point in the Points layer, or leave a single point in the layer.")


def _all_points(points_layer: Any) -> np.ndarray:
    data = np.asarray(points_layer.data, dtype=np.float64)
    if data.size == 0:
        raise ValueError("Points layer has no points. Add at least one point first.")
    if data.ndim != 2 or data.shape[1] < 3:
        raise ValueError("Points layer data must contain z,y,x coordinates.")
    return data


def _save_crop(crop: np.ndarray, metadata: dict[str, Any], output_dir: Path, prefix: str, fmt: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    start = metadata["start_zyx"]
    stem = f"{prefix}_z{start[0]}_y{start[1]}_x{start[2]}"
    fmt = fmt.lower()
    if fmt in {"tif", "tiff"}:
        import tifffile

        output_path = output_dir / f"{stem}.tiff"
        tifffile.imwrite(str(output_path), crop)
    elif fmt == "npy":
        output_path = output_dir / f"{stem}.npy"
        np.save(output_path, crop)
    else:
        raise ValueError(f"Unsupported output format: {fmt}")

    metadata_path = output_dir / f"{stem}.json"
    metadata_with_output = dict(metadata)
    metadata_with_output["output_path"] = str(output_path)
    metadata_path.write_text(json.dumps(metadata_with_output, indent=2), encoding="utf-8")
    return output_path


def _sanitize_filename_stem(value: str) -> str:
    cleaned = "".join(char if char.isalnum() or char in {"-", "_", "."} else "_" for char in str(value).strip())
    cleaned = cleaned.strip("._")
    return cleaned or "image"


def _layer_source_path(layer: Any) -> Path | None:
    source = getattr(layer, "source", None)
    path_value = getattr(source, "path", None)
    if not path_value:
        return None
    if isinstance(path_value, (list, tuple)):
        if not path_value:
            return None
        path_value = path_value[0]
    try:
        return Path(path_value)
    except TypeError:
        return None


def _zarr_container_path(source_path: Path) -> Path | None:
    for candidate in (source_path, *source_path.parents):
        if candidate.suffix.lower() == ".zarr":
            return candidate
    return None


def image_layer_stem(layer: Any) -> str:
    source_path = _layer_source_path(layer)
    if source_path is not None:
        zarr_path = _zarr_container_path(source_path)
        if zarr_path is not None and zarr_path.parent != zarr_path:
            return _sanitize_filename_stem(zarr_path.parent.name)
        return _sanitize_filename_stem(source_path.stem)
    return _sanitize_filename_stem(getattr(layer, "name", "image"))


def default_crop_output_dir(layer: Any | None) -> Path:
    if layer is not None:
        source_path = _layer_source_path(layer)
        if source_path is not None:
            zarr_path = _zarr_container_path(source_path)
            if zarr_path is not None:
                return zarr_path.parent
            if source_path.parent != source_path:
                return source_path.parent
    return Path.cwd()


def _points_layer_choice(viewer: Any) -> str:
    if _layer_by_name(viewer, "crop_anchors_zyx") is not None:
        return "crop_anchors_zyx"
    for layer in viewer.layers:
        if layer.__class__.__name__ == "Points":
            return layer.name
    return ""


def _layer_by_name(viewer: Any, name: str) -> Any | None:
    for layer in viewer.layers:
        if layer.name == name:
            return layer
    return None


def ensure_crop_points_layer(viewer: Any) -> Any:
    existing = _layer_by_name(viewer, "crop_anchors_zyx")
    if existing is not None:
        return existing
    return viewer.add_points(
        np.empty((0, 3), dtype=np.float32),
        name="crop_anchors_zyx",
        ndim=3,
        size=12,
        face_color="magenta",
        border_color="white",
    )


def ensure_crop_preview_layer(viewer: Any) -> Any:
    existing = _layer_by_name(viewer, PREVIEW_LAYER_NAME)
    if existing is not None:
        return existing
    layer = viewer.add_shapes(
        [],
        name=PREVIEW_LAYER_NAME,
        ndim=3,
        shape_type="rectangle",
        edge_color=PREVIEW_EDGE_COLOR_VALID,
        face_color=[0, 0, 0, 0],
        edge_width=PREVIEW_EDGE_WIDTH,
        opacity=0.9,
    )
    # Preview is driven by points + crop size; keep it non-interactive.
    if hasattr(layer, "editable"):
        layer.editable = False
    return layer


def _set_preview_shapes(preview_layer: Any, shapes: list[np.ndarray], edge_colors: list[str]) -> None:
    if not shapes:
        preview_layer.data = []
        return
    preview_layer.data = shapes
    preview_layer.shape_type = ["rectangle"] * len(shapes)
    preview_layer.edge_color = edge_colors
    preview_layer.face_color = [[0, 0, 0, 0]] * len(shapes)
    preview_layer.edge_width = PREVIEW_EDGE_WIDTH


def _build_cropper(viewer: Any) -> Any:
    try:
        from magicgui import magicgui
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("magicgui is required to use the napari point cropper widget.") from exc

    points_layer = ensure_crop_points_layer(viewer)
    ensure_crop_preview_layer(viewer)
    preview_connections: dict[str, Any] = {"points": None}

    def _image_layer_names(widget=None) -> list[str]:
        return [layer.name for layer in viewer.layers if layer.__class__.__name__ == "Image"]

    def _points_layer_names(widget=None) -> list[str]:
        return [layer.name for layer in viewer.layers if layer.__class__.__name__ == "Points"]

    def _current_image_layer() -> Any | None:
        active = getattr(viewer.layers.selection, "active", None)
        if active is not None and active.__class__.__name__ == "Image":
            return active
        for layer in viewer.layers:
            if layer.__class__.__name__ == "Image":
                return layer
        return None

    default_image = _current_image_layer()
    image_layer_names = _image_layer_names()
    if default_image is not None and default_image.name in image_layer_names:
        default_image_layer = default_image.name
    elif image_layer_names:
        default_image_layer = image_layer_names[0]
    else:
        default_image_layer = None

    points_layer_names = _points_layer_names()
    default_points_layer = _points_layer_choice(viewer) or points_layer.name
    if default_points_layer not in points_layer_names:
        default_points_layer = points_layer_names[0] if points_layer_names else None

    @magicgui(
        call_button="Save all crops",
        image_layer={"choices": _image_layer_names, "nullable": True},
        points_layer_name={"choices": _points_layer_names, "nullable": True},
        output_format={"choices": ["tiff", "npy"]},
    )
    def cropper(
        image_layer: str | None = default_image_layer,
        points_layer_name: str | None = default_points_layer,
        save_dir: pathlib.Path = default_crop_output_dir(default_image),
        size_z: int = DEFAULT_CROP_SIZE[0],
        size_y: int = DEFAULT_CROP_SIZE[1],
        size_x: int = DEFAULT_CROP_SIZE[2],
        output_format: str = "tiff",
    ) -> None:
        if not image_layer:
            raise ValueError("Choose an image layer to crop from.")
        image = _layer_by_name(viewer, image_layer)
        if image is None:
            raise ValueError(f"Image layer not found: {image_layer}")
        pts = _layer_by_name(viewer, points_layer_name) if points_layer_name else points_layer
        if pts is None:
            raise ValueError(f"Points layer not found: {points_layer_name}")
        prefix = image_layer_stem(image)
        points = _all_points(pts)
        saved_paths = []
        for point_index, point in enumerate(points):
            crop, metadata = crop_array_from_point(image.data, point[:3], (size_z, size_y, size_x))
            metadata.update(
                {
                    "anchor_point_zyx": [float(v) for v in point[:3]],
                    "point_index": int(point_index),
                    "point_count": int(len(points)),
                    "image_layer": image.name,
                    "points_layer": pts.name,
                    "source_path": str(_layer_source_path(image) or ""),
                }
            )
            output_path = _save_crop(crop, metadata, Path(save_dir), prefix, output_format)
            saved_paths.append(output_path)
            print(f"Saved crop: {output_path}")
        print(f"Saved {len(saved_paths)} crop(s).")

    def _active_points_layer() -> Any | None:
        name = cropper.points_layer_name.value
        if name:
            layer = _layer_by_name(viewer, name)
            if layer is not None:
                return layer
        return points_layer

    def _active_image_shape() -> tuple[int, int, int] | None:
        name = cropper.image_layer.value
        image = _layer_by_name(viewer, name) if name else _current_image_layer()
        if image is None:
            return None
        data = getattr(image, "data", None)
        if data is None or len(getattr(data, "shape", ())) < 3:
            return None
        return tuple(int(value) for value in data.shape[:3])

    def _update_preview(*_args: Any, **_kwargs: Any) -> None:
        pts = _active_points_layer()
        preview = _layer_by_name(viewer, PREVIEW_LAYER_NAME) or ensure_crop_preview_layer(viewer)
        if pts is None:
            _set_preview_shapes(preview, [], [])
            return
        data = np.asarray(pts.data, dtype=np.float64)
        if data.size == 0:
            _set_preview_shapes(preview, [], [])
            return
        crop_size = (
            int(cropper.size_z.value),
            int(cropper.size_y.value),
            int(cropper.size_x.value),
        )
        if any(value <= 0 for value in crop_size):
            _set_preview_shapes(preview, [], [])
            return
        shapes, edge_colors = preview_shapes_from_points(data, crop_size, _active_image_shape())
        _set_preview_shapes(preview, shapes, edge_colors)

    def _disconnect_points_preview() -> None:
        connected = preview_connections.get("points")
        if connected is None:
            return
        layer, callback = connected
        try:
            layer.events.data.disconnect(callback)
        except (TypeError, RuntimeError, ValueError):
            pass
        preview_connections["points"] = None

    def _connect_points_preview(layer: Any | None) -> None:
        _disconnect_points_preview()
        if layer is None:
            return

        def _on_points_changed(*_args: Any, **_kwargs: Any) -> None:
            _update_preview()

        layer.events.data.connect(_on_points_changed)
        preview_connections["points"] = (layer, _on_points_changed)

    def _on_points_layer_changed(event: Any = None) -> None:
        _connect_points_preview(_active_points_layer())
        _update_preview()

    cropper.size_z.changed.connect(_update_preview)
    cropper.size_y.changed.connect(_update_preview)
    cropper.size_x.changed.connect(_update_preview)
    cropper.image_layer.changed.connect(_update_preview)
    cropper.points_layer_name.changed.connect(_on_points_layer_changed)
    viewer.layers.events.inserted.connect(_update_preview)
    viewer.layers.events.removed.connect(_update_preview)

    _connect_points_preview(_active_points_layer())
    _update_preview()
    # Keep the interactive points layer selected after creating the preview overlay.
    if points_layer in viewer.layers:
        viewer.layers.selection.active = points_layer
    return cropper


def make_point_cropper_widget() -> Any:
    import napari
    viewer = napari.current_viewer()
    return _build_cropper(viewer)


def launch_point_cropper(
    image_path: str | Path | None = None,
    *,
    dataset_name: str = "0",
    output_dir: str | Path | None = None,
    crop_size: tuple[int, int, int] = DEFAULT_CROP_SIZE,
) -> None:
    try:
        import napari
        from magicgui import magicgui
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("napari and magicgui are required. Run this in the napari environment.") from exc

    viewer = napari.Viewer(ndisplay=3)
    if image_path:
        viewer.open(str(image_path))
    cropper = _build_cropper(viewer)
    if output_dir:
        cropper.save_dir.value = Path(output_dir)
    cropper.size_z.value = int(crop_size[0])
    cropper.size_y.value = int(crop_size[1])
    cropper.size_x.value = int(crop_size[2])
    viewer.window.add_dock_widget(cropper, area="right", name="Point cropper")
    napari.run()


def _parse_triplet(value: str) -> tuple[int, int, int]:
    parts = [part.strip() for part in str(value).split(",") if part.strip()]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("Expected z,y,x, for example 256,256,256")
    return (int(parts[0]), int(parts[1]), int(parts[2]))


def main() -> int:
    parser = argparse.ArgumentParser(description="Launch napari with a point-anchored 3D crop exporter.")
    parser.add_argument("--image", default="", help="Optional image/Zarr/TIFF path to open at startup")
    parser.add_argument("--dataset_name", default="0", help="Reserved for future Zarr dataset selection")
    parser.add_argument("--output_dir", default="", help="Default crop output directory")
    parser.add_argument("--crop_size", type=_parse_triplet, default=DEFAULT_CROP_SIZE, help="Crop size as z,y,x")
    args = parser.parse_args()
    launch_point_cropper(
        args.image or None,
        dataset_name=args.dataset_name,
        output_dir=args.output_dir or None,
        crop_size=args.crop_size,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
