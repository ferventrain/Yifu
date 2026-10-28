import json

import numpy as np

from pipeline_modules.segmentation.neurofly_brain_seg import (
    append_sidecar,
    axis_ratios,
    coarse_atlas_mask,
    label_window_for_chunk,
    load_sidecar,
    pick_label_level,
    seg_from_jsonable,
    seg_to_jsonable,
)


def test_axis_ratios_from_neurofly_info():
    # spacing 是 µm/体素：L0 体素比 = spacing[level]/spacing[0]
    info = [
        {"spacing": [1.8, 1.8, 2.0]},
        {"spacing": [3.6, 3.6, 4.0]},
        {"spacing": [57.809, 57.650, 64.058]},
    ]
    assert axis_ratios(info, 0) == (1.0, 1.0, 1.0)
    assert axis_ratios(info, 1) == (2.0, 2.0, 2.0)
    r5 = axis_ratios(info, 2)
    assert abs(r5[0] - 57.809 / 1.8) < 1e-9  # ≈32.1
    assert abs(r5[2] - 64.058 / 2.0) < 1e-9


def test_label_window_for_chunk_maps_and_clips():
    shape = (4475, 32752, 18317)  # zyx
    ratios = (8.0, 8.0, 8.0)
    # chunk 起点 (100,100,100) 尺寸 300 → L0 窗口 800 宽
    win = label_window_for_chunk([100, 100, 100, 300, 300, 300], ratios, shape)
    assert win == ((800, 3200), (800, 3200), (800, 3200))
    # 越界夹边
    win2 = label_window_for_chunk([4000, 40000, 22000, 300, 300, 300], ratios, shape)
    assert win2[0][1] <= 4475 and win2[1][1] <= 32752 and win2[2][1] <= 18317
    assert all(lo <= hi for lo, hi in win2)



def test_coarse_atlas_mask_exact_any_pool():
    # label zyx：z 方向只有后半有标注；chunk z 轴 4 cells over 20 voxels
    # → 每 cell 足迹 5 voxels，cell2=[10,15) cell3=[15,20) 有标注
    label = np.zeros((20, 4, 4), dtype=np.uint16)
    label[10:] = 7
    gate = coarse_atlas_mask(label, (4, 4, 4))  # chunk xyz
    assert gate.shape == (4, 4, 4)
    assert gate.dtype == bool
    assert not gate[..., 0].any() and not gate[..., 1].any()
    assert gate[..., 2].all() and gate[..., 3].all()
    assert gate.mean() == 0.5
    # 单层细线（1 voxel 厚）不被 any-pool 漏掉
    thin = np.zeros((8, 8, 8), dtype=np.uint16)
    thin[4, 4, 4] = 3
    g2 = coarse_atlas_mask(thin, (4, 4, 4))
    assert g2.sum() == 1 and g2[2, 2, 2]


def test_pick_label_level_coarsest_not_coarser_than_image():
    assert pick_label_level([0, 1, 2, 3], (8.0, 8.0, 8.0)) == 3
    assert pick_label_level([0, 1, 2, 3, 4, 5, 6], (32.1, 32.0, 4.0)) == 5
    # 图像层比 label 最细层还细 → 退最细
    assert pick_label_level([2, 3, 4], (1.0, 1.0, 1.0)) == 2


def test_sidecar_roundtrip(tmp_path):
    path = tmp_path / "s.chunks.jsonl"
    segs = [
        {"sid": 0, "points": np.array([[1, 2, 3], [4, 5, 6]]),
         "sampled_points": np.array([[1, 2, 3]])},
        {"sid": 1, "points": np.zeros((0, 3), dtype=np.int64),
         "sampled_points": np.zeros((0, 3), dtype=np.int64)},
    ]
    append_sidecar(path, "0,0,0,64,64,64", segs)
    append_sidecar(path, "64,0,0,64,64,64", [])
    loaded = load_sidecar(path)
    assert set(loaded) == {"0,0,0,64,64,64", "64,0,0,64,64,64"}
    assert loaded["0,0,0,64,64,64"][0]["points"] == segs[0]["points"].tolist()
    assert loaded["64,0,0,64,64,64"] == []
    # jsonable 幂等
    assert seg_to_jsonable(segs[0])["points"] == [[1, 2, 3], [4, 5, 6]]
    back = seg_from_jsonable(seg_to_jsonable(segs[0]))
    assert back["points"] == segs[0]["points"].tolist()


def test_load_sidecar_skips_corrupt_lines(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text('{"key": "a", "segs": []}\nnot json\n{"wrong": 1}\n', encoding="utf-8")
    assert load_sidecar(path) == {"a": []}
