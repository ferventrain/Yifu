from __future__ import annotations

from pathlib import Path

import pandas as pd

from pipeline_modules.tubule_reconstruction.kimimaro_reconstruction import (
    _concat_tables_batched,
    _load_spilled_chunk,
    _spill_chunk_result,
)


def test_spill_chunk_result_roundtrip(tmp_path: Path):
    chunk_result = {
        "chunk_index": (1, 2, 3),
        "core_mask_voxels": 10,
        "connected_components": 2,
        "num_skeletons": 1,
        "chunk_summary": {"chunk_z": 1},
        "branch_table": pd.DataFrame({"skeleton_id": [0], "length_um": [1.5]}),
        "vertex_table": pd.DataFrame({"skeleton_id": [0], "x_um": [1.0]}),
        "edge_table": pd.DataFrame({"skeleton_id": [0], "src": [0], "dst": [1]}),
    }
    meta = _spill_chunk_result(tmp_path / "_chunk_spill", chunk_result)
    assert meta["spill_path"]
    assert Path(meta["spill_path"]).is_file()
    assert "branch_table" not in meta
    loaded = _load_spilled_chunk(meta)
    assert loaded["chunk_index"] == (1, 2, 3)
    assert list(loaded["branch_table"]["length_um"]) == [1.5]


def test_concat_tables_batched_matches_pandas():
    frames = [pd.DataFrame({"v": [i, i + 1]}) for i in range(0, 20, 2)]
    expected = pd.concat(frames, ignore_index=True)
    got = _concat_tables_batched(frames, batch_size=3)
    pd.testing.assert_frame_equal(got, expected)
