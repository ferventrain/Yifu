"""Tests for atlas-guided chunk selection before vessel reconstruction."""

from __future__ import annotations

import pytest

from pipeline_modules.tubule_reconstruction.kimimaro_reconstruction import select_region_chunk_indices


def test_selects_only_intersecting_mask_chunks(tiny_annotation_zarr, tiny_mask_zarr, tiny_region_csv):
    indices, metadata = select_region_chunk_indices(
        tiny_annotation_zarr,
        tiny_region_csv,
        "RA",
        mask_shape=(8, 8, 8),
        mask_chunks=(4, 4, 4),
    )

    assert indices == [(0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1)]
    assert metadata["region_ids"] == [10]
    assert metadata["matching_chunks"] == 4
    assert metadata["selected_chunks"] == 4


def test_chunk_margin_adds_adjacent_chunks(tiny_annotation_zarr, tiny_mask_zarr, tiny_region_csv):
    indices, metadata = select_region_chunk_indices(
        tiny_annotation_zarr,
        tiny_region_csv,
        "RA",
        mask_shape=(8, 8, 8),
        mask_chunks=(4, 4, 4),
        chunk_margin=1,
    )

    assert indices == list(__import__("itertools").product(range(2), range(2), range(2)))
    assert metadata["selected_chunks"] == 8


def test_region_filter_rejects_shape_mismatch(tiny_annotation_zarr, tiny_region_csv):
    with pytest.raises(ValueError, match="shape must match"):
        select_region_chunk_indices(
            tiny_annotation_zarr,
            tiny_region_csv,
            "RA",
            mask_shape=(4, 4, 4),
            mask_chunks=(4, 4, 4),
        )
