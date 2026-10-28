import json

import numpy as np

from pipeline_modules.visualization.napari_atlas_overlay import (
    RegionRow,
    _decode_ims_attr,
    choose_label_level,
    ensure_label_pyramid,
    filter_regions,
    label_levels,
    level_scale_zyx,
    match_label_scale,
    region_centroid,
)


def _make_label_store(root, shape=(37, 45, 53)):
    import zarr

    root.mkdir(parents=True, exist_ok=True)
    group = zarr.open_group(str(root), mode="w", zarr_format=2)
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
        idx = tuple(slice(0, s, 2) for s in src.shape)
        np.testing.assert_array_equal(dst, src[idx])
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


def test_choose_label_level_by_grid_extent():
    shapes = {0: (4475, 32752, 18317), 1: (2238, 16376, 9159), 2: (1119, 8188, 4580), 3: (560, 4094, 2290)}
    # 图像是 ims L3（≈全分辨率 1/8）：L0/L1 太细但满足 N_lab>=N_img/2 → 取最细 L0？
# 约束是内存：label 体素不超过图像 2 倍细。L0 相对图像体素 8 倍细，违反 → 应取接近的层
    assert choose_label_level([0, 1, 2, 3], shapes, (560, 4096, 2304)) == 2  # L2 每轴恰 <= 2×图像
    assert choose_label_level([0, 1, 2, 3], shapes, (545, 1286, 932)) == 3  # 更粗时退到 L3
    assert choose_label_level([0, 1, 2, 3], shapes, (4475, 32752, 18317)) == 0
    # 图像比 label 粗层还粗 → 兜底最粗
    assert choose_label_level([2, 3], shapes, (100, 100, 100)) == 3


def test_choose_label_level_requires_all_axes():
    shapes = {0: (100, 100, 100), 1: (50, 50, 50), 2: (25, 25, 25)}
    # z 轴不满足 2× 约束（60 > 50/2? 50>=30 满足）——全部满足则取最细
    assert choose_label_level([0, 1, 2], shapes, (60, 60, 60)) == 0
    # 图像 z 很粗时 L0 z=100 >= 30 仍满足 → 仍 L0
    assert choose_label_level([0, 1, 2], shapes, (30, 200, 200)) == 1


def test_match_label_scale_extent_alignment():
    # 同源网格：全分辨率 (100, 200, 200)，图像为 1/4 层（可能带 padding 26 vs 25）
    img_scale = (1.0, 1.0, 1.0)
    img_shape = (26, 50, 50)
    lab_shape = (100, 200, 200)
    scale = match_label_scale(img_scale, img_shape, lab_shape)
    # 世界范围一致：N_lab * s_lab == N_img * s_img
    for nl, sl, ni, si in zip(lab_shape, scale, img_shape, img_scale):
        assert abs(nl * sl - ni * si) < 1e-9


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
    assert [r.region_id for r in hits[:2]] == [2, 1]  # 精确== > 前缀
    assert filter_regions(rows, "visual")[0].region_id == 3
    assert filter_regions(rows, "7") == [rows[3]]
    assert filter_regions(rows, "") == []


def test_region_centroid():
    data = np.zeros((10, 10, 10), dtype="uint16")
    data[2:6, 4:8, 4:8] = 9
    assert region_centroid(data, 9) == (3.5, 5.5, 5.5)
    assert region_centroid(data, 123) is None


def test_label_levels_skips_non_numeric(tmp_path):
    root = tmp_path / "lab.zarr"
    import zarr

    group = zarr.open_group(str(root), mode="w", zarr_format=2)
    group.create_array("0", shape=(4, 4, 4), dtype="uint8", compressors=None)
    group.create_array("1", shape=(2, 2, 2), dtype="uint8", compressors=None)
    group.create_array("extra", shape=(2, 2, 2), dtype="uint8", compressors=None)
    assert label_levels(root) == [0, 1]


def test_decode_ims_attr_variants():
    # MegaSpim 把标量存成逐字符 |S1 数组
    char_array = np.frombuffer(b"13.0888888888", dtype="S1")
    assert _decode_ims_attr(char_array) == 13.0888888888
    assert _decode_ims_attr(np.float64(2.5)) == 2.5
    assert _decode_ims_attr(np.array([2.5], dtype="f4")) == 2.5
    assert _decode_ims_attr(None) is None
    assert _decode_ims_attr(np.frombuffer(b"abc", dtype="S1")) is None


def test_scan_level_centroids_one_pass():
    from pipeline_modules.visualization.napari_atlas_overlay import scan_level_centroids

    data = np.zeros((6, 8, 10), dtype="uint16")
    data[1:3, 2:6, 4:8] = 5
    data[4:6, 0:2, 0:2] = 9
    centroids, counts = scan_level_centroids(data)
    assert counts == {5: 32, 9: 8}
    assert centroids[5] == (1.5, 3.5, 5.5)
    assert centroids[9] == (4.5, 0.5, 0.5)


def test_resolve_display_set_self_children_or_none():
    from pipeline_modules.visualization.napari_atlas_overlay import (
        RegionRow,
        build_children_map,
        resolve_display_set,
    )

    rows = [
        RegionRow(1, "A", "Area A", (1,)),
        RegionRow(10, "A1", "Area A one", (1, 10)),
        RegionRow(11, "A2", "Area A two", (1, 11)),
        RegionRow(12, "A1a", "Area A one alpha", (1, 10, 12)),
        RegionRow(2, "B", "Area B", (2,)),
    ]
    children = build_children_map(rows)
    assert resolve_display_set(2, {2, 10}, children) == [2]
    assert resolve_display_set(1, {10, 12}, children) == [10, 12]
    assert resolve_display_set(1, set(), children) == []
    assert resolve_display_set(99, {10}, children) == []


def test_fix_layer_scales_sets_um_spacing():
    from pipeline_modules.visualization.napari_atlas_overlay import _fix_layer_scales

    class FakeLayer:
        scale = (1.0, 1.0, 1.0)

    class FakePano:
        scale = (1.0, 1.0, 1.0)
        visible = True

    class FakePanoPoints:
        scale = (1.0, 1.0, 1.0)

    class FakeImage:
        resolution_levels = ["L0", "L7"]

        def __init__(self, info):
            self.info = info
            self.rois = [[0, 0, 0, 18317, 32752, 4475], [0, 0, 0, 285, 512, 35]]

    class FakeLayer2:
        def __init__(self):
            self.scale = (1.0, 1.0, 1.0)

    class FakePanel:
        pass

    info = [
        {"spacing": [0.71, 0.71, 2.0], "image_size": [18317, 32752, 4475]},
        {"spacing": [64.27, 64.0, 128.0], "image_size": [285, 512, 35]},
    ]
    panel = FakePanel()
    panel.image = FakeImage(info)
    panel.image_layer = FakeLayer()
    panel.panorama_image = FakePano()
    panel.panorama_points = FakePanoPoints()
    panel.resolution_dropdown = None
    panel.level = None
    panel.point_layer = FakeLayer2()
    _fix_layer_scales(panel)
    assert np.allclose(panel.image_layer.scale, (0.71, 0.71, 2.0))
    # 配套层（点层）必须与 image 层同步 scale，db 体素坐标才能对齐
    assert np.allclose(panel.point_layer.scale, (0.71, 0.71, 2.0))
    # panorama 用第一个全轴 <1000 的层（此处 idx 1）
    assert np.allclose(panel.panorama_image.scale, (64.27, 64.0, 128.0))
    # panorama 点层坐标是 hr 层绝对体素 → scale 用 hr spacing（0.71/2.0 口径）
    assert np.allclose(panel.panorama_points.scale, (0.71, 0.71, 2.0))
    _fix_layer_scales(panel)
    assert np.allclose(panel.image_layer.scale, (0.71, 0.71, 2.0))
