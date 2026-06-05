"""Shared pytest fixtures for the Yifu pipeline test suite."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import zarr

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture()
def tmp_sample_dir(tmp_path):
    """Return a temporary directory that mimics a sample root."""
    return tmp_path / "sample01"


@pytest.fixture()
def tiny_annotation_zarr(tmp_path):
    """A small 3-D annotation label Zarr (shape 8x8x8, 25 um/vox).

    Labels: voxels in the lower half (z < 4) are 10, upper half are 20.
    """
    store = zarr.open(str(tmp_path / "annotation.zarr"), mode="w")
    arr = store.zeros("0", shape=(8, 8, 8), chunks=(8, 8, 8), dtype="int32")
    arr[:4, :, :] = 10
    arr[4:, :, :] = 20
    return tmp_path / "annotation.zarr"


@pytest.fixture()
def tiny_mask_zarr(tmp_path):
    """A small binary vessel mask Zarr matching tiny_annotation_zarr."""
    store = zarr.open(str(tmp_path / "mask.zarr"), mode="w")
    arr = store.zeros("0", shape=(8, 8, 8), chunks=(4, 4, 4), dtype="uint8")
    arr[1:3, 1:3, 1:3] = 1
    arr[5:7, 1:3, 1:3] = 1
    return tmp_path / "mask.zarr"


@pytest.fixture()
def tiny_skeleton_csvs(tmp_path):
    """Skeleton vertex + edge CSVs covering two regions (labels 10 and 20)."""
    n_vertices = 20
    rng = np.random.default_rng(0)

    vertex_data = {
        "skeleton_id": [1] * n_vertices,
        "node_id": list(range(n_vertices)),
        "z_um": rng.uniform(0, 200, n_vertices).tolist(),
        "y_um": rng.uniform(0, 200, n_vertices).tolist(),
        "x_um": rng.uniform(0, 200, n_vertices).tolist(),
        "radius_um": rng.uniform(1, 5, n_vertices).tolist(),
    }
    vertex_df = pd.DataFrame(vertex_data)
    vertex_csv = tmp_path / "skeleton_vertices.csv"
    vertex_df.to_csv(vertex_csv, index=False)

    n_edges = 10
    edge_data = {
        "skeleton_id":   [1] * n_edges,
        "source_node":   list(range(n_edges)),
        "target_node":   list(range(1, n_edges + 1)),
        "edge_length_um": rng.uniform(2, 20, n_edges).tolist(),
        "source_z_um":   rng.uniform(0, 200, n_edges).tolist(),
        "source_y_um":   rng.uniform(0, 200, n_edges).tolist(),
        "source_x_um":   rng.uniform(0, 200, n_edges).tolist(),
        "target_z_um":   rng.uniform(0, 200, n_edges).tolist(),
        "target_y_um":   rng.uniform(0, 200, n_edges).tolist(),
        "target_x_um":   rng.uniform(0, 200, n_edges).tolist(),
    }
    edge_df = pd.DataFrame(edge_data)
    edge_csv = tmp_path / "skeleton_edges.csv"
    edge_df.to_csv(edge_csv, index=False)

    return vertex_csv, edge_csv


@pytest.fixture()
def tiny_region_csv(tmp_path):
    """A minimal Allen-style region CSV with two leaf nodes."""
    rows = [
        {"id": 1,  "name": "root",      "acronym": "root", "structure_id_path": "/1/"},
        {"id": 10, "name": "RegionA",   "acronym": "RA",   "structure_id_path": "/1/10/"},
        {"id": 20, "name": "RegionB",   "acronym": "RB",   "structure_id_path": "/1/20/"},
    ]
    csv_path = tmp_path / "regions.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    return csv_path
