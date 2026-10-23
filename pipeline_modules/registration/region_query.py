"""Resolve a region query against the Allen region CSV and collect its subtree."""

from __future__ import annotations

from pathlib import Path

from pipeline_modules.tubule_reconstruction.region_vessel_analysis import (
    _collect_subtree_ids,
    load_region_tree_with_lookups,
    resolve_region_query,
)


def resolve_region_subtree(region_query: str, cfg_path: str | Path):
    """Return ``(subtree_ids, node, nodes_by_id)`` for a query.

    Queries may be an acronym, full name, or integer id; the subtree includes
    every descendant of the matched node.
    """
    nodes_by_id, acronym_to_ids, name_to_ids = load_region_tree_with_lookups(cfg_path)
    node = resolve_region_query(region_query, nodes_by_id, acronym_to_ids, name_to_ids)
    return set(_collect_subtree_ids(node)), node, nodes_by_id
