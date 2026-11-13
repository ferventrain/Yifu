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
    load_db_nodes_and_segments,
    match_label_scale,
    nodes_in_region_mask,
    parse_db_node_coords,
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
        translate = (0.0, 0.0, 0.0)

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
            self.translate = (0.0, 0.0, 0.0)

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


def test_fix_layer_scales_labeling_block_translate():
    # labeling 模式：neurofly 把块原点的层级体素数塞进 image.translate（refresh:
    # c_coord - h_size）；napari world = data*scale + translate（translate 原样
    # 世界单位，offscreen 实证），必须乘 spacing 才是块的真实 µm 位置。db 点层
    # 坐标是绝对体素，translate 归 0。旧公式 it*(1-scale) 会把点钉在错位的块上
    #（q 进 labeling 后 atlas label 相对块漂移的根因）。
    from pipeline_modules.visualization.napari_atlas_overlay import _fix_layer_scales

    class FakeImage:
        resolution_levels = ["L0", "L7"]
        info = [{"spacing": [1.42, 1.42, 4.0]}, {"spacing": [64.27, 64.0, 128.0]}]
        rois = [[0, 0, 0, 18317, 32752, 4475]]

    class FakeLayer:
        def __init__(self, translate):
            self.scale = (1.0, 1.0, 1.0)
            self.translate = translate
            self.data = np.zeros((64, 64, 64))

    class FakePanel:
        pass

    panel = FakePanel()
    panel.image = FakeImage()
    panel.image_layer = FakeLayer((5826.0, 10171.0, 1012.0))
    # 旧补偿公式的残留值，修复后应归 0
    panel.point_layer = FakeLayer((5826.0 * 0.42, 10171.0 * 0.42, 1012.0 * 3.0))
    panel.resolution_dropdown = None
    panel.level = None

    class FakeCamera:
        def __init__(self):
            self.center = (0.0, 0.0, 0.0)

    class FakeViewer:
        def __init__(self):
            self.camera = FakeCamera()

    panel.viewer = FakeViewer()
    _fix_layer_scales(panel)
    assert np.allclose(panel.image_layer.scale, (1.42, 1.42, 4.0))
    assert np.allclose(panel.image_layer.translate, (5826 * 1.42, 10171 * 1.42, 1012 * 4.0))
    # 体素原值必须留档：put_point 把 image.translate 当体素偏移读
    assert np.allclose(panel.image_layer._atlas_voxel_translate, (5826.0, 10171.0, 1012.0))
    assert np.allclose(panel.point_layer.scale, (1.42, 1.42, 4.0))
    assert np.allclose(panel.point_layer.translate, (0.0, 0.0, 0.0))
    # 块被搬动时相机同步到块世界中心 = translate + shape*scale/2（64³ 块）
    assert np.allclose(
        panel.viewer.camera.center,
        (5826 * 1.42 + 32 * 1.42, 10171 * 1.42 + 32 * 1.42, 1012 * 4.0 + 32 * 4.0),
    )
    # 第二次运行：块没动（translate 已就位）→ 不拽相机，保住用户导航
    panel.viewer.camera.center = (1.0, 1.0, 1.0)
    _fix_layer_scales(panel)
    assert np.allclose(panel.viewer.camera.center, (1.0, 1.0, 1.0))


def test_parse_db_node_coords_variants():
    # 三种在库写法都收：原生 list / sqlite3.Binary 读回的 bytes / shard 老数据的
    # np 数组 str()（"[np.int64(2282), ...]" 与无逗号 "[ 47 190 69]" 两型）。
    # 非坐标行（恰好不是 3 个数）丢弃。
    rows = [
        ("[10, 20, 30]",),
        (b"[11, 21, 31]",),
        ("[np.int64(12), np.int64(22), np.int64(32)]",),
        ("[ 13 23 33 ]",),
        ("garbage",),
        ("[1, 2]",),
    ]
    arr = parse_db_node_coords(rows)
    assert arr.shape == (4, 3)
    assert arr[0].tolist() == [10.0, 20.0, 30.0]
    assert arr[1].tolist() == [11.0, 21.0, 31.0]
    assert arr[2].tolist() == [12.0, 22.0, 32.0]
    assert arr[3].tolist() == [13.0, 23.0, 33.0]


def test_nodes_in_region_axis_order_xyz():
    # db 坐标是层级体素 XYZ（neurofly from_roi 口径，RRR 实锤 2026-11-04）；
    # 扫描层数组是 zarr 原生 zyx。id 沿 z 分带（z<2 → 5，z≥2 → 9）。
    # n1/n4 的 x 远大于 z：按旧 zyx 解释会把 x 当 z 读进 id9 带（负样本判据）。
    scan = np.zeros((4, 6, 5), dtype=np.uint32)
    scan[:2] = 5
    scan[2:] = 9
    nodes = np.array(
        [
            [1.0, 1.0, 1.0],  # -> (z0,y0,x0) id5
            [4.0, 2.0, 1.0],  # -> (z0,y1,x2) id5；zyx 误读 z=2 -> id9
            [2.0, 5.0, 3.0],  # -> (z1,y2,x1) id5
            [1.0, 1.0, 6.0],  # -> (z3,y0,x0) id9
            [6.0, 1.0, 1.0],  # -> (z0,y0,x3) id5；zyx 误读 z=3 -> id9
        ]
    )
    mask = nodes_in_region_mask(nodes, scan, [5], node_level=1, scan_level=2)
    assert mask.tolist() == [True, True, True, False, True]
    # 换 id9 集合反向断言
    mask9 = nodes_in_region_mask(nodes, scan, [9], node_level=1, scan_level=2)
    assert mask9.tolist() == [False, False, False, True, False]


def test_load_db_nodes_and_segments(tmp_path):
    # nodes+edges → nid 对齐坐标 + 并查集分段；三种 coord 写法都收；
    # 坏行节点跳过（nid 不入列）；edges 引用缺失 nid 的边忽略。
    import sqlite3

    from pipeline_modules.visualization.napari_atlas_overlay import load_db_nodes_and_segments

    db = tmp_path / "t.db"
    conn = sqlite3.connect(str(db))
    conn.execute(
        "CREATE TABLE nodes (nid INTEGER PRIMARY KEY, coord TEXT, creator TEXT, "
        "status INTEGER, type INTEGER, date TIMESTAMP DEFAULT CURRENT_TIMESTAMP, checked INTEGER)"
    )
    conn.execute(
        "CREATE TABLE edges (src INTEGER, des INTEGER, "
        "date TIMESTAMP DEFAULT CURRENT_TIMESTAMP, creator TEXT, PRIMARY KEY (src,des))"
    )
    rows = [
        (1, "[10, 20, 30]"),
        (2, "[11, 21, 31]"),
        (3, "[np.int64(12), np.int64(22), np.int64(32)]"),
        (7, "[ 70 70 70 ]"),
        (8, b"[80, 81, 82]"),
        (9, "garbage"),
    ]
    for nid, coord in rows:
        conn.execute("INSERT INTO nodes (nid, coord) VALUES (?, ?)", (nid, coord))
    for src, des in [(1, 2), (2, 3), (7, 8), (999, 1)]:
        conn.execute("INSERT INTO edges (src, des) VALUES (?, ?)", (src, des))
    conn.commit()
    conn.close()

    nids, coords, segs = load_db_nodes_and_segments(str(db))
    assert nids.tolist() == [1, 2, 3, 7, 8]
    assert coords.shape == (5, 3)
    assert coords[0].tolist() == [10.0, 20.0, 30.0]
    assert coords[4].tolist() == [80.0, 81.0, 82.0]
    # 段 {1,2,3} 连通一段、{7,8} 一段（代表 nid 取决于并查集路径，用同段性断言）
    assert segs[0] == segs[1] == segs[2]
    assert segs[3] == segs[4]
    assert segs[0] != segs[3]


def test_wrap_neurofly_put_point_restores_voxel_translate():
    # µm 世界下 neurofly put_point 把 image_layer.translate 当体素偏移读；
    # 包装版调用前临时换回 _atlas_voxel_translate、调用后恢复 µm。
    import types

    from pipeline_modules.visualization.napari_atlas_overlay import wrap_neurofly_put_point

    class FakeLayer:
        def __init__(self):
            self.mouse_drag_callbacks = []
            self.translate = (100.0, 200.0, 400.0)  # 已修正的 µm 口径
            self._atlas_voxel_translate = (50.0, 100.0, 100.0)  # 体素口径
            self.seen_translate = []

    class FakeBound:
        """模拟登记在 mouse_drag_callbacks 里的 bound method。"""

        def __init__(self):
            self.calls = 0

        @property
        def __func__(self):
            return types.SimpleNamespace(__qualname__="Annotator.put_point")

        def __call__(self, lay, event):
            self.calls += 1
            lay.seen_translate.append(tuple(lay.translate))

    class FakePanel:
        pass

    layer = FakeLayer()
    bound = FakeBound()
    layer.mouse_drag_callbacks.append(bound)
    panel = FakePanel()
    panel.image_layer = layer
    assert wrap_neurofly_put_point(panel) is True
    # 列表里已是包装版 → 幂等
    assert wrap_neurofly_put_point(panel) is False
    wrapper = layer.mouse_drag_callbacks[0]
    assert getattr(wrapper, "_atlas_um_guard", False) is True
    wrapper(layer, None)
    # 原始 put_point 看到的是体素口径；调用完恢复 µm
    assert layer.seen_translate == [(50.0, 100.0, 100.0)]
    assert layer.translate == (100.0, 200.0, 400.0)
    assert bound.calls == 1


def test_ray_bbox_intersections():
    # napari 原生 get_ray_intersections 在点击位置投影落盒内时返回 None
    # （find_front_back_face 要求前后两面，3D 里相机焦点在块内时点击全落盒内
    # → put_point TypeError NoneType，2026-11-05 offscreen 复现：块中心合法点击
    # 也崩）。slab 兜底版对盒内/盒外/斜射/脱靶都要成立。
    from pipeline_modules.visualization.napari_atlas_overlay import _ray_bbox_intersections

    bbox = np.array([[0.0, 64.0], [0.0, 64.0], [0.0, 64.0]])
    # 盒内点沿 +z：near 钳到点击点本身，far=出射面
    near, far = _ray_bbox_intersections((32.0, 32.0, 32.0), (0.0, 0.0, 1.0), bbox)
    assert near is not None and np.allclose(near, (32.0, 32.0, 32.0))
    assert np.allclose(far, (32.0, 32.0, 64.0))
    # 盒外点沿 +z 朝块：near=入射面
    near2, far2 = _ray_bbox_intersections((32.0, 32.0, -10.0), (0.0, 0.0, 1.0), bbox)
    assert np.allclose(near2, (32.0, 32.0, 0.0)) and np.allclose(far2, (32.0, 32.0, 64.0))
    # 斜射线
    near3, far3 = _ray_bbox_intersections((-8.0, -8.0, -8.0), (1.0, 1.0, 1.0), bbox)
    assert near3 is not None and np.allclose(near3, (0.0, 0.0, 0.0))
    assert np.allclose(far3, (64.0, 64.0, 64.0))
    # 脱靶：平行线不穿盒
    miss, _ = _ray_bbox_intersections((500.0, 32.0, 32.0), (0.0, 0.0, 1.0), bbox)
    assert miss is None
