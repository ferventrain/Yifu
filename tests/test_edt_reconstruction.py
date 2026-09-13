from __future__ import annotations

import numpy as np

from pipeline_modules.tubule_reconstruction.edt_reconstruction import (
    skeleton_to_graph,
    skeletonize_binary_edt,
    stitch_chunk_graphs,
)


def test_straight_tube_skeleton_is_thin_and_centered():
    mask = np.zeros((21, 21, 41), dtype=bool)
    mask[8:13, 8:13, 2:39] = True
    skeleton, dbf = skeletonize_binary_edt(mask, resolution_xyz=(1.0, 1.0, 1.0), dust_threshold=0)
    coords = np.argwhere(skeleton)
    assert len(coords) > 10
    assert np.abs(coords[:, 0] - 10).mean() < 1.5
    assert np.abs(coords[:, 1] - 10).mean() < 1.5
    assert float(dbf[skeleton].mean()) >= 1.0


def test_graph_degrees_on_line():
    skeleton = np.zeros((5, 5, 8), dtype=bool)
    skeleton[2, 2, 1:7] = True
    coords, edges, degrees = skeleton_to_graph(skeleton)
    assert len(coords) == 6
    assert len(edges) == 5
    assert int((degrees == 1).sum()) == 2
    assert int((degrees == 2).sum()) == 4


def test_stitch_joins_adjacent_chunk_endpoints():
    def fake_chunk(chunk_index, local_xyz_um, face_axis_coord):
        coords = np.array(local_xyz_um, dtype=np.float64)
        local = np.array([[0, 0, face_axis_coord]], dtype=np.int32)
        return {
            "chunk_index": chunk_index,
            "local_coords": local,
            "degrees": np.array([1], dtype=np.int32),
            "core_shape": (4, 4, 4),
            "global_coords_um": coords,
        }

    left = fake_chunk((0, 0, 0), [[0.0, 0.0, 3.0]], 3)
    right = fake_chunk((0, 0, 1), [[0.0, 0.0, 4.2]], 0)
    stitched = stitch_chunk_graphs([left, right], max_distance_um=5.0)
    assert len(stitched) == 1
    assert bool(stitched.iloc[0]["is_stitch"])
    assert stitched.iloc[0]["chunk_a"] == "0.0.0"
    assert stitched.iloc[0]["chunk_b"] == "0.0.1"
