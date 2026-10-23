import json

import numpy as np

from pipeline_modules.visualization.napari_atlas_overlay import (
    RegionRow,
    choose_label_level,
    ensure_label_pyramid,
    filter_regions,
    label_levels,
    level_scale_zyx,
    region_centroid,
)


def _make_label_store(root, shape=(37, 45, 53)):
    import zarr

    root.mkdir(parents=True, exist_ok=True)
    group = zarr.open_group(str(root), mode="w")
    arr = group.create_array("0", shape=shape, chunks=(8, 8, 8), dtype="uint16", compressors=None)
    data = np.zeros(shape, dtype="uint16")
    data[10:20, 10:20, 10:20] = 5
    data[30:36, 30:44, 40:52] = 7
    arr[:] = data
    (root / ".zattrs").write_text(
        json.dumps({"multiscales": [{"version": "0.4", "datasets": [
            {"path": "0", "coordinateTransformations": [{"type": "scale", "scale": [2.0, 1.0, 1.0]}]}
        ]}]}),
        encoding="utf-8",
    )
    return data


def test_ensure_label_pyramid_builds_nearest_levels(tmp_path):
    root = tmp_path / "lab.zarr"
    data = _make_label_store(root)

    levels = ensure_label_pyramid(root, log=lambda *_: None)

    assert levels[0] == 0
    assert levels == sorted(levels)
    import zarr

    group = zarr.open_group(str(root), mode="r")
    for k in levels[1:]:
        src = np.asarray(group[str(k - 1)])
        dst = np.asarray(group[str(k)])
        assert dst.shape == tuple((s + 1) // 2 for s in src.shape)
        # 最近邻抽稀：粗层每个值来自细层的偶数位
        idx = tuple(slice(0, s, 2) for s in src.shape)
        np.testing.assert_array_equal(dst, src[idx])
    # id 5 在各层都存活（块够大）
    for k in levels:
        assert 5 in np.unique(np.asarray(group[str(k)]))

    attrs = json.loads((root / ".zattrs").read_text(encoding="utf-8"))
    paths = [d["path"] for d in attrs["multiscales"][0]["datasets"]]
    assert paths == [str(k) for k in levels]
    scale0 = attrs["multiscales"][0]["datasets"][0]["coordinateTransformations"][0]["scale"]
    scale2 = attrs["multiscales"][0]["datasets"][2]["coordinateTransformations"][0]["scale"]
    assert scale0 == [2.0, 1.0, 1.0]
    assert scale2 == [8.0, 4.0, 4.0]


def test_ensure_label_pyramid_idempotent(tmp_path):
    root = tmp_path / "lab.zarr"
    _make_label_store(root)
    first = ensure_label_pyramid(root, log=lambda *_: None)
    second = ensure_label_pyramid(root, log=lambda *_: None)
    assert first == second


def test_choose_label_level_matches_image_voxel():
    base = (2.0, 0.7, 0.7)
    # 图像体素足够粗时 label 取最细层（不过度欠采样即可）
    assert choose_label_level([0, 1, 2, 3], base, (5.6, 5.6, 5.6)) == 0
    assert choose_label_level([0, 1, 2, 3], base, (1.2, 1.2, 1.2)) == 0
    assert choose_label_level([0, 1, 2, 3], base, (0.6, 0.6, 0.6)) == 0
    # 只给粗层列表时取最粗可用层（都超 2× 时兜底）
    assert choose_label_level([3, 4], base, (0.5, 0.5, 0.5)) == 4


def test_level_scale_multiplies_by_power_of_two():
    assert level_scale_zyx((2.0, 1.0, 0.5), 3) == (16.0, 8.0, 4.0)


def test_filter_regions_ranking():
    rows = [
        RegionRow(1, "SSp-m6b", "Primary somatosensory area, mouth, layer 6b"),
        RegionRow(2, "SSp", "Primary somatosensory areas"),
        RegionRow(3, "VISpm2/3", "posteromedial visual area, layer 2/3"),
        RegionRow(7, "root", "root,root"),
    ]
    hits = filter_regions(rows, "ssp")
    assert [r.region_id for r in hits[:2]] == [2, 1]  # 前缀优先于包含
    assert filter_regions(rows, "visual")[0].region_id == 3
    assert filter_regions(rows, "7") == [rows[3]]  # id 命中
    assert filter_regions(rows, "") == []


def test_region_centroid(tmp_path):
    data = np.zeros((10, 10, 10), dtype="uint16")
    data[2:6, 4:8, 4:8] = 9
    c = region_centroid(data, 9)
    assert c == (3.5, 5.5, 5.5)
    assert region_centroid(data, 123) is None


def test_label_levels_skips_non_numeric(tmp_path):
    root = tmp_path / "lab.zarr"
    import zarr

    group = zarr.open_group(str(root), mode="w")
    group.create_array("0", shape=(4, 4, 4), dtype="uint8", compressors=None)
    group.create_array("1", shape=(2, 2, 2), dtype="uint8", compressors=None)
    group.create_array("extra", shape=(2, 2, 2), dtype="uint8", compressors=None)
    assert label_levels(root) == [0, 1]
