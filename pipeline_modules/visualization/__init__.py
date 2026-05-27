"""Visualization helpers for atlas, vessel, and region outputs."""

from __future__ import annotations

__all__ = [
    "AtlasSlice",
    "AtlasSliceSpec",
    "coordinate_to_index",
    "download_all_svgs",
    "extract_atlas_slice",
    "render_bregma_ap_svg",
    "render_atlas_slice",
]


def __getattr__(name: str):
    if name in {
        "AtlasSlice",
        "AtlasSliceSpec",
        "coordinate_to_index",
        "extract_atlas_slice",
        "render_atlas_slice",
    }:
        from . import atlas_slice

        return getattr(atlas_slice, name)
    if name in {"download_all_svgs", "render_bregma_ap_svg"}:
        from . import allen_svg_slice

        return getattr(allen_svg_slice, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
