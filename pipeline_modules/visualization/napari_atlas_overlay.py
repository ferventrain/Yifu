"""napari 图像 + 图谱 label 叠加浏览器（跟随图像层级 / 脑区搜索 / 点击跳转）。

目的：在 neurofly（Segs Annotator）等任意 napari 工作流里叠加 atlas label。
napari Labels 层直接吃单尺度全分辨率 label（RRR 为 4475×32752×18317）会在
渲染时整体物化导致卡退，本工具给 label zarr 就地补最近邻抽稀金字塔，并
**跟随**当前图像层：用户在 neurofly 面板切换 resolution level 时，label 自动
换到匹配层级（按网格 extent 对齐，不依赖图像层 scale 的单位约定）。

用法（yifu env，与 neurofly 同环境；sample-dir 存在 label 时自动加载）::

    python -m pipeline_modules.visualization.napari_atlas_overlay \
        --sample-dir S:/Arivis_Analysis/_active/YF2026071002/ot

    # 或显式指定
    python -m pipeline_modules.visualization.napari_atlas_overlay \
        --label-zarr <...>/upsampled_atlas_label.zarr \
        --region-csv S:/Yifu/pipeline_modules/registration/Region_Csv_Rev2_updated.CSV

功能：
1. 自动/手动加载 label zarr，就地建最近邻金字塔（一次性成本）；
2. 跟随任意图像层（插入 / 换 data / 换 scale 均触发），label 层级与图像体素
   匹配（label 体素不超过图像 2×，取最细满足层），scale 按网格 extent 对齐；
3. 搜索脑区（缩写精确>前缀>包含 / 名称 / id），回车或点击跳质心，保持 zoom；
4. 状态栏显示鼠标下脑区名（id 配 region CSV 解析）。
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REGION_CSV = REPO_ROOT / "pipeline_modules" / "registration" / "Region_Csv_Rev2_updated.CSV"
LABEL_LAYER_NAME = "atlas_label"
PYRAMID_MAX_LEVEL = 6
PYRAMID_MIN_AXIS = 16  # stop decimating once an axis falls below this


# --------------------------------------------------------------------------
# label zarr 金字塔
# --------------------------------------------------------------------------

def _open_group(label_zarr: str | Path):
    import zarr

    return zarr.open_group(str(label_zarr), mode="a")


def label_levels(label_zarr: str | Path) -> list[int]:
    """已存在的金字塔层级号（数据集名 "0","1",...）。"""
    group = _open_group(label_zarr)
    levels = []
    for key in group.keys():
        if re.fullmatch(r"\d+", str(key)):
            levels.append(int(key))
    return sorted(levels)


def ensure_label_pyramid(
    label_zarr: str | Path,
    *,
    max_level: int = PYRAMID_MAX_LEVEL,
    min_axis: int = PYRAMID_MIN_AXIS,
    log=print,
) -> list[int]:
    """就地给 label zarr 补最近邻抽稀金字塔层级（label 是分类 id，只能抽稀不能平均）。

    已存在的层级跳过，断点续跑安全。写完把根 .zattrs 的 multiscales.datasets
    扩成全部层级（napari-ome-zarr 拖拽打开时也按多层读）。
    """
    group = _open_group(label_zarr)
    existing = label_levels(label_zarr)

    def _finer(level: int):
        return max(e for e in existing if e < level)

    level = len(existing)
    while level <= max_level:
        src_level = _finer(level)
        src = group[str(src_level)]
        shape = src.shape
        if min(shape) < min_axis:
            break
        new_shape = tuple(max(1, (s + 1) // 2) for s in shape)
        if new_shape == shape:
            break
        try:
            dst = group.create_array(
                str(level), shape=new_shape, chunks=_level_chunks(new_shape),
                dtype=src.dtype, compressors=None,
            )
        except TypeError:  # zarr 2 API
            dst = group.create_dataset(
                str(level), shape=new_shape, chunks=_level_chunks(new_shape),
                dtype=src.dtype, compressor=None,
            )
        # 非整数倍时最近邻取上取整采样，保证覆盖到最后一行/列
        idx = tuple(slice((i + 1) // 2 - 1 if i + 1 > 1 else 0, s, 2) for i, s in enumerate(shape))
        dst[:] = src[idx]
        existing.append(level)
        log(f"label pyramid level {level}: {shape} -> {new_shape}")
        level += 1

    _refresh_multiscales(label_zarr, existing)
    return existing


def _level_chunks(shape) -> tuple[int, ...]:
    # 粗层很小，chunk 取整层或 256³ 封顶
    return tuple(min(int(s), 256) for s in shape)


def _refresh_multiscales(label_zarr: str | Path, levels: list[int]) -> None:
    path = Path(label_zarr)
    attrs_path = path / ".zattrs"
    attrs = json.loads(attrs_path.read_text(encoding="utf-8")) if attrs_path.exists() else {}
    ms = attrs.get("multiscales") or [{"version": "0.4", "name": "volume"}]
    base_scale = [1.0, 1.0, 1.0]
    for tf in ms[0].get("datasets", [{}])[0].get("coordinateTransformations", []):
        if tf.get("type") == "scale":
            base_scale = [float(v) for v in tf["scale"]]
    ms[0]["datasets"] = [
        {
            "path": str(k),
            "coordinateTransformations": [{"type": "scale", "scale": [b * 2.0**k for b in base_scale]}],
        }
        for k in levels
    ]
    attrs["multiscales"] = ms
    attrs_path.write_text(json.dumps(attrs, indent=1), encoding="utf-8")


def level_scale_zyx(base_scale_zyx, level: int) -> tuple[float, float, float]:
    f = 2.0**level
    return tuple(b * f for b in base_scale_zyx)


def match_label_scale(image_scale, image_shape, label_shape):
    """网格 extent 对齐：label 与图像同源全分辨率网格时，两层世界范围相等。

    N_lab * s_lab = N_img * s_img  =>  s_lab = s_img * N_img / N_lab
    与图像层 scale 用什么单位无关（µm 或 1,1,1 都对），且自动吃掉金字塔层
    的取整 padding。
    """
    return tuple(
        float(si) * ni / nl for si, ni, nl in zip(image_scale, image_shape, label_shape)
    )


def choose_label_level(available: list[int], label_shapes: dict, image_shape) -> int:
    """选体素不比图像细超过 2× 的最细层级（N_lab <= 2·N_img，控制内存）。

    label_shapes: {level: shape_zyx}；image_shape 为参考图像层形状。
    图像比全部 label 层都粗时取最粗层（体素最少）。
    """
    for k in available:  # 升序，第一个满足的即最细且不超内存
        if all(nl <= 2.0 * ni for nl, ni in zip(label_shapes[k], image_shape)):
            return k
    return available[-1]


# --------------------------------------------------------------------------
# ims 元数据（体素尺寸读取，MegaSpim ExtMin/Max 是逐字符 |S1 数组）
# --------------------------------------------------------------------------

def _decode_ims_attr(value) -> float | None:
    arr = np.asarray(value)
    if arr.dtype.kind == "f" and arr.size == 1:
        return float(arr.item())
    if arr.dtype.kind in ("S", "U") or arr.dtype == np.dtype("S1"):
        text = arr.tobytes().decode(errors="ignore") if arr.dtype.kind == "S" else str(arr)
        try:
            return float(text)
        except ValueError:
            return None
    try:
        return float(arr.item())
    except Exception:
        return None


def ims_voxel_um_zyx(ims_path: str | Path, level: int) -> tuple[float, float, float] | None:
    """从 /DataSetInfo/Image 的 ExtMin/ExtMax（µm, x/y/z）+ 层级形状推该层体素 (z,y,x) µm。"""
    import h5py

    with h5py.File(str(ims_path), "r") as handle:
        info = handle.get("DataSetInfo", {}).get("Image")
        if info is None:
            return None
        attrs = dict(info.attrs)
        ext = []
        for ax in range(3):
            lo = _decode_ims_attr(attrs.get(f"ExtMin{ax}"))
            hi = _decode_ims_attr(attrs.get(f"ExtMax{ax}"))
            if lo is None or hi is None:
                return None
            ext.append(hi - lo)
    lv = next(item for item in ims_levels(ims_path) if item["level"] == level)
    shape = lv["shape_zyx"]  # (z, y, x)
    voxel_xyz = [ext[i] / max(shape[2 - i], 1) for i in range(3)]
    return (voxel_xyz[2], voxel_xyz[1], voxel_xyz[0])  # -> zyx


def ims_levels(ims_path: str | Path) -> list[dict]:
    """[(level, shape_zyx, stride_xyz)]，level 0 最细。"""
    import h5py

    out = []
    with h5py.File(str(ims_path), "r") as handle:
        ds = handle["DataSet"]
        level_keys = sorted(
            (int(k.split()[-1]) for k in ds.keys() if k.startswith("ResolutionLevel")),
        )
        shapes = {}
        for lv in level_keys:
            data = ds[f"ResolutionLevel {lv}"]["TimePoint 0"]["Channel 0"]["Data"]
            shapes[lv] = tuple(int(v) for v in data.shape)
        full = shapes[level_keys[0]]
        for lv in level_keys:
            stride = [full[i] / max(shapes[lv][i], 1) for i in range(3)]
            out.append({"level": lv, "shape_zyx": shapes[lv], "stride_xyz": stride})
    return out


# --------------------------------------------------------------------------
# region CSV / 搜索
# --------------------------------------------------------------------------

@dataclass
class RegionRow:
    region_id: int
    acronym: str
    name: str

    @property
    def text(self) -> str:
        return f"{self.acronym} | {self.name} | {self.region_id}"


def load_region_rows(csv_path: str | Path) -> list[RegionRow]:
    import csv

    rows = []
    with open(csv_path, encoding="utf-8-sig", newline="") as handle:
        for raw in csv.DictReader(handle):
            try:
                rid = int(raw["id"])
            except (TypeError, ValueError):
                continue
            acronym = str(raw.get("acronym") or "").strip()
            name = str(raw.get("name") or "").strip()
            rows.append(RegionRow(rid, acronym, name))
    return rows


def filter_regions(rows: list[RegionRow], query: str, limit: int = 60) -> list[RegionRow]:
    """精确 == 缩写 > 缩写前缀 > 缩写包含 > 名称/id 包含，大小写不敏感。"""
    q = query.strip().lower()
    if not q:
        return []
    exact, prefix, contains, name_hit = [], [], [], []
    for row in rows:
        ac = row.acronym.lower()
        if ac == q:
            exact.append(row)
        elif ac.startswith(q):
            prefix.append(row)
        elif q in ac:
            contains.append(row)
        elif q in row.name.lower() or q in str(row.region_id):
            name_hit.append(row)
    out = exact + prefix + contains + name_hit
    return out[:limit]


def region_centroid(level_array, region_id: int) -> tuple[float, float, float] | None:
    """某层 label 数组中该脑区的质心（体素坐标 zyx）。数组须可一次读入（粗层）。"""
    coords = np.argwhere(np.asarray(level_array) == region_id)
    if coords.size == 0:
        return None
    return tuple(float(v) for v in coords.mean(axis=0))


# --------------------------------------------------------------------------
# Qt widget
# --------------------------------------------------------------------------

def make_atlas_overlay_widget(
    viewer,
    *,
    label_zarr: str | Path | None = None,
    region_csv: str | Path = DEFAULT_REGION_CSV,
    auto_load: bool = True,
):
    """制造图谱叠加 dock。label 跟随 viewer 里任意 3D 图像层（neurofly 等）。"""
    from qtpy import QtCore, QtWidgets

    state = {
        "label_zarr": Path(label_zarr) if label_zarr else None,
        "levels": [],
        "shapes": {},      # level -> shape_zyx
        "scales": {},      # level -> 当前世界 scale（随参考图像层更新）
        "centroids": {},   # level -> {id: (z,y,x)}
        "rows": [],
        "id_to_row": {},
        "connected": set(),  # 已挂 data/scale 事件的层 id
    }

    box = QtWidgets.QWidget()
    layout = QtWidgets.QVBoxLayout(box)

    # --- 图谱控制 ---
    row1 = QtWidgets.QHBoxLayout()
    btn_label = QtWidgets.QPushButton("叠加图谱…")
    btn_reload = QtWidgets.QPushButton("重新匹配层级")
    check_follow = QtWidgets.QCheckBox("跟随图像层级")
    check_follow.setChecked(True)
    row1.addWidget(btn_label)
    row1.addWidget(btn_reload)
    row1.addWidget(check_follow)
    layout.addLayout(row1)

    # --- 搜索 ---
    edit_search = QtWidgets.QLineEdit()
    edit_search.setPlaceholderText("搜索脑区：缩写 / 名称 / id")
    layout.addWidget(edit_search)
    list_results = QtWidgets.QListWidget()
    list_results.setMaximumHeight(260)
    layout.addWidget(list_results)
    status = QtWidgets.QLabel("尚未加载图谱")
    status.setWordWrap(True)
    layout.addWidget(status)

    debounce = QtCore.QTimer(singleShot=True, interval=250)

    # ---------- 参考图像层 / 跟随 ----------

    def _is_label_layer(layer) -> bool:
        return layer.name == LABEL_LAYER_NAME

    def _reference_layer():
        """只认 3D Image 层（neurofly 的 Points 等层 data.shape=(N,3)，不能当图像）。"""
        try:
            from napari.layers import Image
        except ImportError:
            return None
        for layer in reversed(viewer.layers):
            if _is_label_layer(layer) or not isinstance(layer, Image):
                continue
            data = getattr(layer, "data", None)
            shape = getattr(data, "shape", None)
            scale = getattr(layer, "scale", None)
            if shape is not None and len(shape) == 3 and scale is not None and len(scale) == 3 \
                    and all(np.isfinite(s) and s > 0 for s in scale):
                return layer
        return None

    def _watch_layer(layer):
        if id(layer) in state["connected"] or _is_label_layer(layer):
            return
        state["connected"].add(id(layer))
        if hasattr(layer, "events"):
            layer.events.data.connect(lambda _e=None: debounce.start())
            layer.events.scale.connect(lambda _e=None: debounce.start())

    def _rematch():
        if not (check_follow.isChecked() and state["levels"]):
            return
        try:
            ref = _reference_layer()
            if ref is None:
                return
            wanted = choose_label_level(state["levels"], state["shapes"], ref.data.shape)
            layer = None
            for lyr in viewer.layers:
                if lyr.name == LABEL_LAYER_NAME:
                    layer = lyr
                    break
            if layer is None or layer.data.shape != state["shapes"][wanted]:
                _attach_level(wanted, ref)
            else:
                _rescale_layer(layer, wanted, ref)
        except Exception as exc:  # noqa: BLE001 - 跟随逻辑绝不崩 GUI
            status.setText(f"跟随图像失败: {exc}")

    debounce.timeout.connect(_rematch)

    def _on_layers_inserted(event):
        _watch_layer(event.value)
        debounce.start()

    viewer.layers.events.inserted.connect(_on_layers_inserted)

    def _on_layers_removed(event=None):
        debounce.start()

    viewer.layers.events.removed.connect(_on_layers_removed)
    for layer in viewer.layers:
        _watch_layer(layer)

    # ---------- 图谱加载 ----------

    def _rescale_layer(layer, level: int, ref):
        scale = match_label_scale(
            tuple(float(v) for v in ref.scale), tuple(ref.data.shape), state["shapes"][level],
        )
        state["scales"][level] = scale
        layer.scale = scale
        layer.refresh()

    def _attach_level(level: int, ref=None):
        import zarr

        scale = None
        if ref is not None:
            scale = match_label_scale(
                tuple(float(v) for v in ref.scale), tuple(ref.data.shape), state["shapes"][level],
            )
            if not all(np.isfinite(s) and s > 0 for s in scale):
                scale = None  # 图像层 scale 异常时退回占位，不喂非法值给 napari
        if scale is None:
            scale = tuple(2.0**level for _ in range(3))  # 无图像层时的占位相对 scale
        state["scales"][level] = scale
        group = zarr.open_group(store=str(state["label_zarr"]), mode="r")
        arr = group[str(level)]
        existing = None
        for lyr in viewer.layers:
            if lyr.name == LABEL_LAYER_NAME:
                existing = lyr
                break
        if existing is not None:
            existing.data = arr
            existing.scale = scale
            existing.refresh()
        else:
            viewer.add_labels(
                arr, name=LABEL_LAYER_NAME, scale=scale, opacity=0.35, blending="translucent",
            )
        status.setText(
            f"图谱 L{level}（{state['shapes'][level][0]}×{state['shapes'][level][1]}×{state['shapes'][level][2]}），"
            f"参考图像 {ref.name if ref else '无'}，scale {tuple(round(v, 4) for v in scale)}"
        )

    def _load_label(path: str | Path | None = None):
        if path is None:
            path, _ = QtWidgets.QFileDialog.getOpenFileName(
                box, "选择 label zarr", "", "Label Zarr (*upsampled_atlas_label.zarr *.zarr)",
            )
            if not path:
                return
        state["label_zarr"] = Path(path)
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            levels = ensure_label_pyramid(path)
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
        state["levels"] = levels
        import zarr

        group = zarr.open_group(store=str(path), mode="r")
        state["shapes"] = {k: tuple(int(v) for v in group[str(k)].shape) for k in levels}
        state["rows"] = load_region_rows(region_csv)
        state["id_to_row"] = {r.region_id: r for r in state["rows"]}
        coarse = np.asarray(group[str(levels[-1])])
        centroids = {}
        for rid in np.unique(coarse):
            if rid == 0:
                continue
            c = region_centroid(coarse, int(rid))
            if c is not None:
                centroids[int(rid)] = c
        state["centroids"] = {levels[-1]: centroids}
        state["scales"] = {}
        ref = _reference_layer()
        if ref is not None:
            _attach_level(choose_label_level(levels, state["shapes"], ref.data.shape), ref)
        else:
            status.setText(
                f"图谱就绪（{len(levels)} 层金字塔，{len(state['rows'])} 区域）。"
                "在 neurofly / 任意方式打开 3D 图像后自动叠加。"
            )

    btn_label.clicked.connect(lambda: _load_label())

    def _manual_rematch():
        ref = _reference_layer()
        if ref is None or not state["levels"]:
            status.setText("没有可参考的 3D 图像层（先在 neurofly / 其他方式加载图像）")
            return
        _attach_level(choose_label_level(state["levels"], state["shapes"], ref.data.shape), ref)

    btn_reload.clicked.connect(_manual_rematch)

    # ---------- 搜索 / 跳转 ----------

    def _refresh_results():
        list_results.clear()
        for row in filter_regions(state["rows"], edit_search.text()):
            list_results.addItem(row.text)

    edit_search.textChanged.connect(_refresh_results)

    def _world_centroid(row: RegionRow):
        """质心的世界坐标：优先用当前展示层的 scale；层缺失则按相对比例换算。"""
        import zarr

        for level in reversed(state["levels"]):
            cache = state["centroids"].setdefault(level, {})
            centroid = cache.get(row.region_id)
            if centroid is None:
                try:
                    arr = np.asarray(zarr.open(store=str(state["label_zarr"] / str(level)), mode="r"))
                except Exception:
                    continue
                centroid = region_centroid(arr, row.region_id)
                if centroid is None:
                    continue
                cache[row.region_id] = centroid
            scale = state["scales"].get(level)
            if scale is None:
                # 没有图像层参考：按层间 2 倍比例归一到 L0 相对坐标
                scale = tuple(1.0 / 2.0**level for _ in range(3))
            return tuple(c * s for c, s in zip(centroid, scale))
        return None

    def _jump(row: RegionRow):
        world = _world_centroid(row)
        if world is None:
            status.setText(f"未找到 {row.text} 的体素（该脑区在此 label 中无覆盖）")
            return
        if viewer.dims.ndisplay == 2:
            viewer.dims.point = world
        else:
            viewer.camera.center = world
        status.setText(f"跳转 → {row.text}")

    def _on_item(text_item):
        row = state["id_to_row"].get(int(text_item.text().rsplit("|", 1)[-1].strip()))
        if row:
            _jump(row)

    list_results.itemClicked.connect(_on_item)

    def _on_return():
        items = list_results.selectedItems() or ([list_results.item(0)] if list_results.count() else [])
        if items:
            _on_item(items[0])

    edit_search.returnPressed.connect(_on_return)

    # ---------- 鼠标位置读脑区 ----------

    def _on_cursor(event=None):
        layer = None
        for lyr in viewer.layers:
            if lyr.name == LABEL_LAYER_NAME:
                layer = lyr
                break
        if layer is None:
            return
        try:
            value = layer.get_value(viewer.cursor.position)
        except Exception:
            return
        if value is None:
            return
        rid = int(value)
        row = state["id_to_row"].get(rid)
        status.setText(f"当前脑区: {row.text if row else rid}")

    viewer.cursor.events.position.connect(_on_cursor)

    if auto_load and state["label_zarr"] is not None and Path(state["label_zarr"]).exists():
        QtCore.QTimer.singleShot(0, lambda: _load_label(state["label_zarr"]))

    return box


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def _find_label_zarr(sample_dir: Path | None) -> Path | None:
    if sample_dir is None:
        return None
    for name in ("upsampled_atlas_label.zarr", "upsampled_label_ngff.zarr"):
        candidate = sample_dir / name
        if candidate.exists():
            return candidate
    return None


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sample-dir", help="样本目录：自动加载 upsampled_atlas_label.zarr")
    parser.add_argument("--label-zarr", help="atlas label zarr 路径（默认 <sample-dir>/upsampled_atlas_label.zarr）")
    parser.add_argument("--region-csv", default=str(DEFAULT_REGION_CSV))
    args = parser.parse_args(argv)

    sample_dir = Path(args.sample_dir) if args.sample_dir else None
    label_zarr = Path(args.label_zarr) if args.label_zarr else _find_label_zarr(sample_dir)

    import napari

    viewer = napari.current_viewer() or napari.Viewer()
    widget = make_atlas_overlay_widget(
        viewer, label_zarr=label_zarr, region_csv=args.region_csv, auto_load=True,
    )
    viewer.window.add_dock_widget(widget, area="left", name="图谱叠加")
    napari.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
