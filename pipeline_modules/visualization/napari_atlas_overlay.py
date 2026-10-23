"""napari 图像 + 图谱 label 叠加浏览器（脑区搜索 / 点击跳转）。

目的：解决"neurofly/napari 里 imaris 图像 + 全分辨率 atlas label 无法共存"的问题。
napari Labels 层直接吃单尺度全分辨率 label（RRR 为 4475×32752×18317）会在渲染时
整体物化导致卡退，本工具给 label zarr 就地补最近邻抽稀金字塔，按当前图像层的
体素尺寸选择匹配层级加载，内存占用始终与图像同级。

用法（yifu env，neurofly 也在该环境，可在同一窗口内叠加使用）::

    python -m pipeline_modules.visualization.napari_atlas_overlay \
        --sample-dir S:/Arivis_Analysis/_active/YF202607100101_ZC_nao_RRR

    # 或显式指定
    python -m pipeline_modules.visualization.napari_atlas_overlay \
        --label-zarr <...>/upsampled_atlas_label.zarr \
        --label-scale-um 2.0,0.71,0.71 \
        --region-csv S:/Yifu/pipeline_modules/registration/Region_Csv_Rev2_updated.CSV

功能：
1. 「打开图像」：.ims（选金字塔层级，懒加载）或 .zarr（数组 0）；
2. 「叠加图谱」：给 label zarr 就地建最近邻金字塔（一次成本），按图像层级
   自动选 label 层级，scale 用 µm 世界坐标对齐；切换层级即换加载层；
3. 侧边栏搜索脑区（缩写/全名/id 子串），回车或点击跳转到该脑区质心，
   保持当前 zoom（2D 改 dims.point，3D 改 camera.center）；
4. 状态栏显示鼠标下脑区名（id 沿用 label 的 id 体系，配 region CSV 解析）。
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


def choose_label_level(available: list[int], base_scale_zyx, target_scale_zyx) -> int:
    """选体素尺寸不超过图像 2× 的最细层级（label 不过度欠采样）；都超则取最粗。"""
    target = min(target_scale_zyx) if target_scale_zyx else 1.0
    for k in available:  # 升序，第一个满足的即最细
        voxel = min(level_scale_zyx(base_scale_zyx, k))
        if voxel <= 2.0 * target:
            return k
    return available[-1]


# --------------------------------------------------------------------------
# ims 元数据
# --------------------------------------------------------------------------

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
            lo = attrs.get(f"ExtMin{ax}")
            hi = attrs.get(f"Max{ax}") or attrs.get(f"ExtMax{ax}")
            if lo is None or hi is None:
                return None
            ext.append(float(np.asarray(hi).item()) - float(np.asarray(lo).item()))
    lv = next(item for item in ims_levels(ims_path) if item["level"] == level)
    shape = lv["shape_zyx"]  # (z, y, x)
    voxel_xyz = [ext[i] / max(shape[2 - i], 1) for i in range(3)]
    return (voxel_xyz[2], voxel_xyz[1], voxel_xyz[0])  # -> zyx


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
    label_scale_zyx=None,
):
    from qtpy import QtCore, QtWidgets

    state = {
        "label_zarr": Path(label_zarr) if label_zarr else None,
        "base_scale": tuple(label_scale_zyx) if label_scale_zyx else None,
        "levels": [],
        "centroids": {},   # level -> {id: (z,y,x)}
        "rows": [],
        "id_to_row": {},
    }

    box = QtWidgets.QWidget()
    box.setWindowTitle("图谱叠加")  # dock 标题兜底
    layout = QtWidgets.QVBoxLayout(box)

    # --- 图像 ---
    btn_image = QtWidgets.QPushButton("打开图像（ims / zarr）…")
    layout.addWidget(btn_image)

    # --- 图谱 ---
    row1 = QtWidgets.QHBoxLayout()
    btn_label = QtWidgets.QPushButton("叠加图谱…")
    combo_level = QtWidgets.QComboBox()
    check_follow = QtWidgets.QCheckBox("跟随图像层级")
    check_follow.setChecked(True)
    row1.addWidget(btn_label, 1)
    row1.addWidget(combo_level, 1)
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

    # ---------- 图像打开 ----------

    def _open_image():
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            box, "选择图像", str(state["label_zarr"].parent if state["label_zarr"] else ""),
            "Images (*.ims *.zarr);;All (*.*)",
        )
        if not path:
            return
        p = Path(path)
        if p.suffix == ".ims":
            levels = ims_levels(p)
            labels = [
                f"L{item['level']}  {item['shape_zyx'][0]}×{item['shape_zyx'][1]}×{item['shape_zyx'][2]}"
                for item in levels
            ]
            sel, ok = QtWidgets.QInputDialog.getItem(
                box, "选择金字塔层级", "ims resolution level（0 最细，一般选 2-4）:", labels,
                min(3, len(labels) - 1), False,
            )
            if not ok:
                return
            item = levels[labels.index(sel)]
            import h5py

            handle = h5py.File(str(p), "r")
            data = handle["DataSet"][f"ResolutionLevel {item['level']}"]["TimePoint 0"]["Channel 0"]["Data"]
            voxel = ims_voxel_um_zyx(p, item["level"]) or (2.0, 1.8, 1.8)
            viewer.add_image(
                data, name=f"ims_L{item['level']}", scale=voxel,
                blending="additive", multiscale=False,
            )
            box._ims_handle = handle  # 防止 h5 句柄被 GC
            status.setText(f"图像 L{item['level']}，体素(zyx) {voxel} µm")
        else:
            import zarr

            arr = zarr.open(store=str(p / "0"), mode="r")
            voxel = state.get("base_scale") or (1.0, 1.0, 1.0)
            viewer.add_image(arr, name=p.name, scale=voxel, blending="additive")
            status.setText(f"图像 {p.name}")

    btn_image.clicked.connect(_open_image)

    # ---------- 图谱加载 ----------

    def _ensure_base_scale():
        if state["base_scale"] is None:
            text, ok = QtWidgets.QInputDialog.getText(
                box, "label 体素尺寸",
                "未从样本 config 读到 label 体素尺寸，输入 z,y,x µm（逗号分隔）:",
                text="2.0,1.8,1.8",
            )
            if ok and text.strip():
                state["base_scale"] = tuple(float(v) for v in text.split(","))
            else:
                state["base_scale"] = (1.0, 1.0, 1.0)

    def _attach_level(level: int):
        for layer in list(viewer.layers):
            if layer.name == LABEL_LAYER_NAME:
                viewer.layers.remove(layer)
                break
        import zarr

        group = zarr.open_group(str(state["label_zarr"]), mode="r")
        arr = group[str(level)]
        scale = level_scale_zyx(state["base_scale"], level)
        viewer.add_labels(
            arr, name=LABEL_LAYER_NAME, scale=scale, opacity=0.35, blending="translucent",
        )
        combo_level.blockSignals(True)
        combo_level.setCurrentText(f"L{level}")
        combo_level.blockSignals(False)
        status.setText(
            f"图谱 L{level}，体素(zyx) {tuple(round(v, 3) for v in scale)} µm，"
            f"共 {len(state['rows'])} 区域"
        )

    def _pick_level_for_current_image() -> int:
        target = None
        for layer in viewer.layers:
            if layer.name == LABEL_LAYER_NAME:
                continue
            if getattr(layer, "scale", None) and len(layer.scale) == 3:
                target = tuple(float(v) for v in layer.scale)
                break
        if target is None:
            return state["levels"][-1]
        return choose_label_level(state["levels"], state["base_scale"], target)

    def _load_label():
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            box, "选择 label zarr", "", "Label Zarr (*upsampled_atlas_label.zarr *.zarr)",
        )
        if not path:
            return
        state["label_zarr"] = Path(path)
        _ensure_base_scale()
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            levels = ensure_label_pyramid(path)
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
        state["levels"] = levels
        combo_level.blockSignals(True)
        combo_level.clear()
        combo_level.addItems([f"L{k}" for k in levels])
        combo_level.blockSignals(False)
        state["rows"] = load_region_rows(region_csv)
        state["id_to_row"] = {r.region_id: r for r in state["rows"]}
        # 粗层 id 集合 + 质心缓存（一次读入）
        import zarr

        coarse = zarr.open(store=str(state["label_zarr"] / str(levels[-1])), mode="r")
        coarse_np = np.asarray(coarse)
        centroids = {}
        for rid in np.unique(coarse_np):
            if rid == 0:
                continue
            c = region_centroid(coarse_np, int(rid))
            if c is not None:
                centroids[int(rid)] = c
        state["centroids"] = {levels[-1]: centroids}
        _attach_level(_pick_level_for_current_image())

    btn_label.clicked.connect(_load_label)

    def _on_level_changed(text: str):
        if text and state["levels"]:
            _attach_level(int(text[1:]))

    combo_level.currentTextChanged.connect(_on_level_changed)

    def _on_layers_inserted(event):
        # neurofly/其他方式新加了图像层 → 跟随换层级
        if not (check_follow.isChecked() and state["levels"]):
            return
        layer = event.value
        if layer.name == LABEL_LAYER_NAME:
            return
        target = tuple(float(v) for v in layer.scale) if len(getattr(layer, "scale", ())) == 3 else None
        if target is None:
            return
        wanted = choose_label_level(state["levels"], state["base_scale"], target)
        current = combo_level.currentText()
        if current and wanted != int(current[1:]):
            _attach_level(wanted)

    viewer.layers.events.inserted.connect(_on_layers_inserted)

    # ---------- 搜索 / 跳转 ----------

    def _refresh_results():
        list_results.clear()
        for row in filter_regions(state["rows"], edit_search.text()):
            list_results.addItem(row.text)

    edit_search.textChanged.connect(_refresh_results)

    def _jump(row: RegionRow):
        # 从粗层往上找该 id 的质心；粗层没有（小区被抽稀掉）则试更细层
        import zarr

        for level in reversed(state["levels"]):
            cache = state["centroids"].setdefault(level, {})
            if row.region_id in cache:
                centroid = cache[row.region_id]
            else:
                try:
                    arr = np.asarray(zarr.open(store=str(state["label_zarr"] / str(level)), mode="r"))
                except Exception:
                    continue
                centroid = region_centroid(arr, row.region_id)
                if centroid is None:
                    continue
                cache[row.region_id] = centroid
            world = tuple(c * s for c, s in zip(centroid, level_scale_zyx(state["base_scale"], level)))
            if viewer.dims.ndisplay == 2:
                viewer.dims.point = world
            else:
                viewer.camera.center = world
            status.setText(f"跳转 → {row.text}")
            return
        status.setText(f"未找到 {row.text} 的体素（该脑区在此 label 中无覆盖）")

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
        layer = viewer.layers.get(LABEL_LAYER_NAME) if hasattr(viewer.layers, "get") else None
        if layer is None:
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

    return box


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def _default_label_scale(sample_dir: Path | None):
    if sample_dir is None:
        return None
    cfg = sample_dir / "config.json"
    if not cfg.exists():
        return None
    try:
        data = json.loads(cfg.read_text(encoding="utf-8"))
        xyz = data["input"]["resolution_xyz"]
        return (float(xyz[2]), float(xyz[1]), float(xyz[0]))  # -> zyx
    except Exception:
        return None


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sample-dir", help="样本目录：预填 label zarr / 从 config.json 读体素尺寸")
    parser.add_argument("--label-zarr", help="atlas label zarr 路径（默认 <sample-dir>/upsampled_atlas_label.zarr）")
    parser.add_argument("--label-scale-um", help="label 体素 z,y,x µm（默认取样本 config 的 input.resolution_xyz）")
    parser.add_argument("--region-csv", default=str(DEFAULT_REGION_CSV))
    parser.add_argument("--image", help="启动即打开的图像（ims/zarr，可选）")
    args = parser.parse_args(argv)

    sample_dir = Path(args.sample_dir) if args.sample_dir else None
    label_zarr = Path(args.label_zarr) if args.label_zarr else (
        sample_dir / "upsampled_atlas_label.zarr" if sample_dir else None
    )
    if label_zarr is not None and not label_zarr.exists():
        label_zarr = sample_dir / "upsampled_label_ngff.zarr" if sample_dir else None
    scale = None
    if args.label_scale_um:
        scale = tuple(float(v) for v in args.label_scale_um.split(","))
    else:
        scale = _default_label_scale(sample_dir)

    import napari

    viewer = napari.CurrentViewer() or napari.Viewer()
    widget = make_atlas_overlay_widget(
        viewer, label_zarr=label_zarr, region_csv=args.region_csv, label_scale_zyx=scale,
    )
    viewer.window.add_dock_widget(widget, area="left", name="图谱叠加")
    if args.image:
        viewer.open(args.image)
    napari.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
