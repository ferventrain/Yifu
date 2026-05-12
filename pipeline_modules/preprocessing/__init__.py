"""Preprocessing module public API."""

from .config import (
    ChannelSubtractionCfg,
    ClaheCfg,
    DownsampleCfg,
    EdgeSignalRemovalCfg,
    MedianFilterCfg,
    PreprocessingCfg,
    RollingBallCfg,
    ScatteringRemovalCfg,
    TophatCfg,
    TubularEnhancementCfg,
    ZarrCfg,
    export_json_schema,
    layout_for_sample,
    load_capability_manifest,
)
try:
    from .preprocessor import (
        Preprocessor,
        apply_processing_steps,
        normalize_channel_list,
        run_preprocessing,
    )
except ModuleNotFoundError as exc:  # pragma: no cover - optional runtime dependency missing
    Preprocessor = None  # type: ignore[assignment]
    _preprocessor_import_error = str(exc)

    def apply_processing_steps(*args, **kwargs):  # type: ignore[no-redef]
        raise ModuleNotFoundError(f"preprocessing support is unavailable: {_preprocessor_import_error}")

    def normalize_channel_list(*args, **kwargs):  # type: ignore[no-redef]
        raise ModuleNotFoundError(f"preprocessing support is unavailable: {_preprocessor_import_error}")

    def run_preprocessing(*args, **kwargs):  # type: ignore[no-redef]
        raise ModuleNotFoundError(f"preprocessing support is unavailable: {_preprocessor_import_error}")

try:
    from .downsample import ImageDownsampler, downsample_folder
except ModuleNotFoundError as exc:  # pragma: no cover - optional runtime dependency missing
    ImageDownsampler = None  # type: ignore[assignment]
    _downsample_import_error = str(exc)

    def downsample_folder(*args, **kwargs):  # type: ignore[no-redef]
        raise ModuleNotFoundError(f"downsample support is unavailable: {_downsample_import_error}")

try:
    from .tubular_enhancement import enhance_tubular_zarr, export_to_tiff
except ModuleNotFoundError as exc:  # pragma: no cover - optional runtime dependency missing
    _tubular_import_error = str(exc)

    def enhance_tubular_zarr(*args, **kwargs):  # type: ignore[no-redef]
        raise ModuleNotFoundError(f"tubular enhancement support is unavailable: {_tubular_import_error}")

    def export_to_tiff(*args, **kwargs):  # type: ignore[no-redef]
        raise ModuleNotFoundError(f"tubular enhancement support is unavailable: {_tubular_import_error}")

try:
    from .tiff_to_zarr import convert_tiff_to_zarr
except ModuleNotFoundError as exc:  # pragma: no cover - optional runtime dependency missing
    _tiff_to_zarr_import_error = str(exc)

    def convert_tiff_to_zarr(*args, **kwargs):  # type: ignore[no-redef]
        raise ModuleNotFoundError(f"TIFF-to-Zarr support is unavailable: {_tiff_to_zarr_import_error}")

try:
    from .edge_signal_removal import remove_edge_signal
except ModuleNotFoundError as exc:  # pragma: no cover - optional runtime dependency missing
    _edge_import_error = str(exc)

    def remove_edge_signal(*args, **kwargs):  # type: ignore[no-redef]
        raise ModuleNotFoundError(f"edge signal removal support is unavailable: {_edge_import_error}")

__all__ = [
    "ChannelSubtractionCfg",
    "ClaheCfg",
    "DownsampleCfg",
    "EdgeSignalRemovalCfg",
    "ImageDownsampler",
    "MedianFilterCfg",
    "PreprocessingCfg",
    "Preprocessor",
    "RollingBallCfg",
    "ScatteringRemovalCfg",
    "TophatCfg",
    "TubularEnhancementCfg",
    "ZarrCfg",
    "apply_processing_steps",
    "convert_tiff_to_zarr",
    "downsample_folder",
    "enhance_tubular_zarr",
    "export_to_tiff",
    "export_json_schema",
    "layout_for_sample",
    "load_capability_manifest",
    "normalize_channel_list",
    "remove_edge_signal",
    "run_preprocessing",
]
