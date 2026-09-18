"""Open a downsampled volume in micro-SAM Annotator 3D and save a tissue mask.

Draw a box on one slice, segment that slice, then propagate through Z with
Segment All Slices (Shift+S). This is meant for tissue envelopes that automatic
hysteresis cannot close across a notch or gap.

CLI::

    python -m pipeline_modules.visualization.annotate_tissue_sam_napari --input ch0_surface_homogenized_qc.zarr --output ch0_tissue_mask.zarr
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# Windows: torchvision can load JPEG/PNG DLLs that break Pillow unless Image is imported first.
import PIL.Image  # noqa: F401


USAGE = """
micro-SAM Annotator 3D
----------------------
1. Wait for embeddings (first launch also downloads vit_b_lm weights).
2. Select the 'prompts' layer. Draw a rectangle that covers the tissue on a mid slice.
3. Click Segment Slice. Check the 'current_object' overlay.
4. In Segment ND, set projection to 'box', then Segment All Slices (Shift+S).
5. If a notch is still missing, go to that slice, draw another box, Segment Slice, then Segment All Slices again.
6. Click Commit, or interpolate with napari-label-interpolator, then pick the layer in Save tissue mask.
7. Click **Save tissue mask** (choose any Labels layer, e.g. `*- interpolated`).

Then run homogenize with --tissue_mask pointing at the saved zarr, using the same --downsample as this QC volume.
""".strip()


def _default_output(input_path: Path) -> Path:
    stem = input_path.stem.removesuffix("_qc")
    return input_path.with_name(f"{stem}_tissue_mask.zarr")


def _to_uint8(volume: np.ndarray) -> np.ndarray:
    work = np.asarray(volume, dtype=np.float32)
    finite = np.isfinite(work)
    if not finite.any():
        return np.zeros(work.shape, dtype=np.uint8)
    lo, hi = np.percentile(work[finite], (1.0, 99.8))
    if not np.isfinite(hi) or hi <= lo:
        lo = float(work[finite].min())
        hi = float(work[finite].max())
    if hi <= lo:
        return np.zeros(work.shape, dtype=np.uint8)
    scaled = np.clip((work - lo) / (hi - lo), 0.0, 1.0)
    return (scaled * 255.0).astype(np.uint8)


def load_volume_for_annotation(path: Path, channel: str | None = None) -> tuple[np.ndarray, dict]:
    try:
        from pipeline_modules.utils.zarr_io import open_zarr_array
    except ImportError:  # pragma: no cover
        from ..utils.zarr_io import open_zarr_array

    arr = open_zarr_array(path)
    attrs = dict(getattr(arr, "attrs", {}) or {})
    names = list(attrs.get("channel_names") or [])
    data = np.asarray(arr)
    if data.ndim == 4:
        axis = int(attrs.get("channel_axis") or 0)
        if axis != 0:
            data = np.moveaxis(data, axis, 0)
        if channel and channel in names:
            index = names.index(channel)
        elif "downsampled_input" in names:
            index = names.index("downsampled_input")
        else:
            index = 0
        data = data[index]
        attrs["selected_channel"] = names[index] if index < len(names) else str(index)
    elif data.ndim != 3:
        raise ValueError(f"Expected a 3D or 4D QC volume, got shape {data.shape} from {path}")
    attrs.setdefault("downsample", 1)
    return data, attrs


def labels_to_tissue(labels: np.ndarray) -> np.ndarray:
    return np.asarray(labels) > 0


def save_tissue_mask_zarr(
    path: Path,
    mask: np.ndarray,
    *,
    downsample: int = 1,
    source: str = "",
) -> Path:
    try:
        from pipeline_modules.utils.zarr_io import (
            create_array,
            ome_ngff_multiscales,
            open_output_group,
        )
    except ImportError:  # pragma: no cover
        from ..utils.zarr_io import create_array, ome_ngff_multiscales, open_output_group

    binary = labels_to_tissue(mask).astype(np.uint8)
    if not binary.any():
        raise ValueError("Selected labels layer is empty.")
    path = Path(path)
    if path.suffix.lower() != ".zarr":
        path = path.with_suffix(".zarr")
    root = open_output_group(path, overwrite=True)
    ds = float(max(int(downsample), 1))
    root.attrs["multiscales"] = ome_ngff_multiscales(
        ["0"],
        ndim=3,
        base_scale=(ds, ds, ds),
        name="tissue_mask",
    )
    chunks = (
        min(64, int(binary.shape[0])),
        min(128, int(binary.shape[1])),
        min(128, int(binary.shape[2])),
    )
    arr = create_array(
        root,
        "0",
        shape=binary.shape,
        chunks=chunks,
        dtype=np.uint8,
        compressor="default",
        data=binary,
    )
    arr.attrs["downsample"] = int(ds)
    arr.attrs["source"] = str(source)
    arr.attrs["tissue_fraction"] = float(binary.mean())
    print(f"Saved tissue mask {binary.shape} fraction={binary.mean():.4f} -> {path}")
    return path


def _labels_layer_names(viewer) -> list[str]:
    from napari.layers import Labels

    return [str(layer.name) for layer in viewer.layers if isinstance(layer, Labels)]


def _default_labels_layer(viewer) -> str:
    names = _labels_layer_names(viewer)
    if not names:
        return ""
    for preferred in (
        "committed_objects - interpolated",
        "current_object - interpolated",
        "committed_objects",
        "current_object",
    ):
        if preferred in names:
            return preferred
    return names[-1]


def _labels_from_layer(viewer, layer_name: str) -> np.ndarray:
    if layer_name not in viewer.layers:
        raise ValueError(f"Layer not found: {layer_name}")
    from napari.layers import Labels

    layer = viewer.layers[layer_name]
    if not isinstance(layer, Labels):
        raise ValueError(f"Layer {layer_name!r} is not a Labels layer")
    data = np.asarray(layer.data)
    if data.ndim != 3:
        raise ValueError(f"Layer {layer_name!r} must be 3D, got shape {data.shape}")
    return data


def launch_annotator(
    image: np.ndarray,
    *,
    output: Path,
    embedding_path: Path | None,
    model_type: str,
    downsample: int,
    source: str,
) -> int:
    try:
        from micro_sam.sam_annotator import annotator_3d
    except ModuleNotFoundError:
        print("micro-sam is not installed. Run: python -m pip install micro-sam", file=sys.stderr)
        return 1

    from magicgui import magicgui
    import napari

    print(USAGE)
    print(f"Volume shape={image.shape} dtype={image.dtype} model={model_type}")
    print("Computing embeddings (first run downloads the SAM checkpoint)...")

    viewer = annotator_3d(
        image,
        embedding_path=str(embedding_path) if embedding_path is not None else None,
        model_type=model_type,
        return_viewer=True,
        prefer_decoder=False,
    )

    @magicgui(
        call_button="Save tissue mask",
        layer_name={"label": "Labels layer", "choices": []},
    )
    def save_mask(layer_name: str = "") -> None:
        name = str(layer_name).strip()
        if not name:
            napari.utils.notifications.show_error("Choose a Labels layer to save.")
            return
        try:
            labels = _labels_from_layer(viewer, name)
            save_tissue_mask_zarr(output, labels, downsample=downsample, source=source)
        except Exception as exc:
            napari.utils.notifications.show_error(str(exc))
            return
        napari.utils.notifications.show_info(f"Saved {output} from layer {name!r}")

    def _refresh_save_layer_choices(_event=None) -> None:
        choices = _labels_layer_names(viewer)
        save_mask.layer_name.choices = choices
        current = str(save_mask.layer_name.value or "")
        if current in choices:
            return
        default = _default_labels_layer(viewer)
        if default:
            save_mask.layer_name.value = default

    viewer.layers.events.inserted.connect(_refresh_save_layer_choices)
    viewer.layers.events.removed.connect(_refresh_save_layer_choices)
    _refresh_save_layer_choices()

    viewer.window.add_dock_widget(save_mask, name="Save tissue mask", area="right")
    napari.run()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Annotate a tissue envelope with micro-SAM 3D box prompts and save a mask Zarr",
    )
    parser.add_argument(
        "--input",
        required=True,
        help="QC Zarr from surface homogenize (channel downsampled_input) or a 3D Zarr volume",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output mask Zarr (default: <input without _qc>_tissue_mask.zarr)",
    )
    parser.add_argument(
        "--channel",
        default=None,
        help="QC channel name to annotate (default: downsampled_input)",
    )
    parser.add_argument(
        "--model_type",
        default="vit_b_lm",
        help="micro-SAM model; vit_b_lm is the light-microscopy default",
    )
    parser.add_argument(
        "--embedding_path",
        default=None,
        help="Directory to cache SAM embeddings (default: next to --output)",
    )
    parser.add_argument(
        "--no_stretch",
        action="store_true",
        help="Do not percentile-stretch to uint8 before SAM",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Input not found: {input_path}", file=sys.stderr)
        return 1
    output = Path(args.output) if args.output else _default_output(input_path)
    embedding_path = Path(args.embedding_path) if args.embedding_path else output.with_name(f"{output.stem}_embeddings")

    volume, attrs = load_volume_for_annotation(input_path, channel=args.channel)
    downsample = int(attrs.get("downsample") or 1)
    image = np.asarray(volume) if args.no_stretch else _to_uint8(volume)
    return launch_annotator(
        image,
        output=output,
        embedding_path=embedding_path,
        model_type=str(args.model_type),
        downsample=downsample,
        source=str(input_path),
    )


if __name__ == "__main__":
    sys.exit(main())
