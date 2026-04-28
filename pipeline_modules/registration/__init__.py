"""Registration & brain-region analysis module.

Public API — import these from ``pipeline_modules.registration``:

- :class:`~BidirectionalRegistration` — ANTs-based atlas ↔ sample registration
- :func:`~analyze_zarr_graph` — Zarr-native block-graph region signal analysis
- :func:`~check_region_coverage` — voxel-coverage check for a region subtree
- :func:`~merge_atlas_regions` — remap fine atlas labels into coarser target regions
"""

from .config import (
    AnalysisCfg,
    RegistrationCfg,
    export_json_schema,
    layout_for_sample,
    load_capability_manifest,
)

__all__ = [
    "AnalysisCfg",
    "RegistrationCfg",
    "export_json_schema",
    "layout_for_sample",
    "load_capability_manifest",
]
