from .config import (
    RegionVesselAnalysisCfg,
    TeasarParams,
    TubuleReconstructionCfg,
    export_json_schema,
    layout_for_sample,
    load_capability_manifest,
    output_dir_for_sample,
)
import importlib

def __getattr__(name):
    lazy = {
        "analyze_binary_mask_zarr": "kimimaro_reconstruction",
        "branch_table_from_skeletons": "kimimaro_reconstruction",
        "open_zarr_dataset": "kimimaro_reconstruction",
        "skeletonize_binary_mask": "kimimaro_reconstruction",
        "summarize_vessel_network": "kimimaro_reconstruction",
        "analyze_regions_from_skeleton": "region_vessel_analysis",
        "load_region_tree_with_lookups": "region_vessel_analysis",
        "resolve_region_query": "region_vessel_analysis",
    }
    module = lazy.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(importlib.import_module(f".{module}", __package__), name)

__all__ = [
    "RegionVesselAnalysisCfg",
    "TeasarParams",
    "TubuleReconstructionCfg",
    "analyze_binary_mask_zarr",
    "analyze_regions_from_skeleton",
    "branch_table_from_skeletons",
    "export_json_schema",
    "layout_for_sample",
    "load_capability_manifest",
    "load_region_tree_with_lookups",
    "open_zarr_dataset",
    "output_dir_for_sample",
    "resolve_region_query",
    "skeletonize_binary_mask",
    "summarize_vessel_network",
]
