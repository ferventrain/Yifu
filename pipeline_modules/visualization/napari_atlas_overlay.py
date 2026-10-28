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
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REGION_CSV = REPO_ROOT / "pipeline_modules" / "registration" / "Region_Csv_Rev2_updated.CSV"
# Rev2 切换之前产的 label 用官方大 id，配 Rev1 CSV（仓库已删，用本地副本兜底）
LEGACY_REGION_CSV_CANDIDATES = (
    Path("S:/_frangi_trials2/Region_Csv_Rev1_updated.CSV"),
    REPO_ROOT.parent / "_frangi_trials2" / "Region_Csv_Rev1_updated.CSV",
)
LABEL_LAYER_NAME = "atlas_label"
PYRAMID_MAX_LEVEL = 6
PYRAMID_MIN_AXIS = 16  # stop decimating once an axis falls below this
# 质心扫描层的内存预算（字节）：预算内取最细层，一次向量化扫描出全部存活 id
CENTROID_SCAN_BUDGET = 1_000_000_000


# --------------------------------------------------------------------------
# label zarr 金字塔
# --------------------------------------------------------------------------

def _open_group(label_zarr: str | Path):
    import zarr

    # zarr 3.1.6 在既有 v2 store 上 mode='a' 不带 zarr_format 时，新建数组的
    # 写入会静默蒸发（.zarray 正常、0 个 chunk 落盘）。按 store 实际格式打开：
    # 有 .zgroup 是 v2（显式钉住），否则不传参（v3 store / zarr2 API）
    path = Path(label_zarr)
    force_v2 = (not path.exists()) or (path / ".zgroup").exists()
    try:
        if force_v2:
            return zarr.open_group(str(path), mode="a", zarr_format=2)
        return zarr.open_group(str(path), mode="a")
    except TypeError:
        return zarr.open_group(str(path), mode="a")


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

    写入不走 zarr3 的 setitem——它的 v2 异步写路径在本机大 store 上不可靠
    （偶发静默丢弃、asyncio 线程死转）；改为手工按 v2 规范落 chunk 文件
    （numcodecs zstd 压缩、全零 chunk 跳过），读取仍走 zarr 稳定的读路径。
    """
    existing = label_levels(label_zarr)

    def _finer(level: int):
        return max(e for e in existing if e < level)

    # 层名 = 2^k 抽稀倍数的 k，由源层推导（不能用 len(existing)：历史目录有
    # 空洞时 len 会整体错位一档，名字与倍率语义脱钩）
    level = (max(existing) + 1) if existing else 1
    while level <= max_level:
        if min(_open_read(label_zarr, str(_finer(level))).shape) < min_axis:
            break
        src_level = _finer(level)
        src_arr = _open_read(label_zarr, str(src_level))
        shape = src_arr.shape
        new_shape = tuple(max(1, (s + 1) // 2) for s in shape)
        if new_shape == shape:
            break
        chunks = _level_chunks(new_shape)
        _write_pyramid_level(
            label_zarr, level, new_shape, chunks, src_arr.dtype, src_arr, log
        )
        existing.append(level)
        log(f"label pyramid level {level}: {shape} -> {new_shape}")
        level += 1

    _refresh_multiscales(label_zarr, existing)
    return existing


def _open_read(label_zarr: str | Path, dataset: str):
    import zarr

    _tune_zarr_singlethread()
    return zarr.open(store=str(Path(label_zarr) / dataset), mode="r")


def _write_pyramid_level(store, level, shape, chunks, dtype, src_arr, log) -> None:
    """手工写一个 v2 层：.zarray + 逐 chunk 文件（zstd+shuffle，全零跳过）。"""
    import shutil
    import time

    from numcodecs import Blosc

    root = Path(store) / str(level)
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    codec = Blosc(cname="zstd", clevel=5, shuffle=Blosc.SHUFFLE)
    zarray = {
        "shape": [int(v) for v in shape],
        "chunks": [int(c) for c in chunks],
        "dtype": np.dtype(dtype).str,
        "compressor": {"id": "blosc", "cname": "zstd", "clevel": 5, "shuffle": 1, "blocksize": 0},
        "fill_value": 0,
        "order": "C",
        "filters": None,
        "dimension_separator": ".",
        "zarr_format": 2,
    }
    (root / ".zarray").write_text(json.dumps(zarray), encoding="utf-8")

    nz, ny, nx = (int(v) for v in shape)
    cz, cy, cx = (int(v) for v in chunks)
    written_chunks = 0
    t0 = time.time()
    for zo in range(nz):
        plane = np.ascontiguousarray(np.asarray(src_arr[2 * zo])[::2, ::2])
        for yo in range(0, ny, cy):
            for xo in range(0, nx, cx):
                sub = plane[yo:yo + cy, xo:xo + cx]
                if not sub.any():
                    continue  # fill_value=0 的 chunk 不落盘（v2 语义）
                if sub.shape != (cy, cx):
                    # 边缘 chunk 补零到整块：zarr3 的 v2 读取器不支持 v2 规范的
                    # 截断边缘块（reshape 整 chunk 尺寸失败），补零最兼容
                    padded = np.zeros((cy, cx), dtype=sub.dtype)
                    padded[: sub.shape[0], : sub.shape[1]] = sub
                    sub = padded
                if not sub.flags["C_CONTIGUOUS"]:
                    sub = np.ascontiguousarray(sub)
                (root / f"{zo}.{yo // cy}.{xo // cx}").write_bytes(codec.encode(sub))
                written_chunks += 1
        if (zo + 1) % 256 == 0:
            log(f"  L{level} plane {zo + 1}/{nz}, chunks {written_chunks}, {time.time() - t0:.0f}s")
    # 自验：中平面经 zarr 读回应与直接抽稀一致（防再次整层白跑）
    mid_direct = np.ascontiguousarray(np.asarray(src_arr[2 * (nz // 2)])[::2, ::2])
    mid_back = np.asarray(_open_read(store, str(level))[nz // 2])
    if not np.array_equal(mid_back, mid_direct):
        raise RuntimeError(f"label pyramid L{level} mid-plane readback mismatch")
    log(f"  L{level} done: {written_chunks} nonzero chunks in {time.time() - t0:.0f}s")


def _level_chunks(shape) -> tuple[int, ...]:
    # z 向整平面（逐平面流式写），xy 2048 封顶：小层自动退化为整层单 chunk
    nz = 1
    return (nz, min(int(shape[1]), 2048), min(int(shape[2]), 2048))


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
    path: tuple[int, ...] = ()

    @property
    def text(self) -> str:
        return f"{self.acronym} | {self.name} | {self.region_id}"


def load_region_rows(csv_path: str | Path) -> list[RegionRow]:
    import ast
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
            try:
                path = tuple(int(v) for v in ast.literal_eval(str(raw.get("structure_id_path") or "")))
            except Exception:
                path = ()
            rows.append(RegionRow(rid, acronym, name, path))
    return rows
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


def scan_level_centroids(arr_zyx) -> tuple[dict[int, tuple[float, float, float]], dict[int, int]]:
    """单次遍历一层 label，返回 {id: zyx 质心} 和 {id: 体素数}。

    逐 z 平面累加（平面内 unique + bincount），不物化三维坐标网格，内存只需
    两个平面级临时数组；被最近邻抽稀抹掉的小 id 自然不在结果里。
    """
    import numpy as _np

    arr = arr_zyx
    nz = arr.shape[0]
    ny, nx = int(arr.shape[1]), int(arr.shape[2])
    ygrid, xgrid = _np.meshgrid(
        _np.arange(ny, dtype=_np.float64), _np.arange(nx, dtype=_np.float64), indexing="ij"
    )
    sums: dict[int, list[float]] = {}  # id -> [count, sz, sy, sx]
    for z in range(nz):
        plane = _np.asarray(arr[z])
        if plane.size == 0 or not plane.any():
            continue
        ids, inverse = _np.unique(plane, return_inverse=True)
        counts = _np.bincount(inverse.ravel(), minlength=len(ids))
        sel = ids != 0
        if not sel.any():
            continue
        ids_nz = ids[sel]
        inv_flat = inverse.ravel()
        yw = _np.bincount(inv_flat, weights=ygrid.ravel(), minlength=len(ids))[sel]
        xw = _np.bincount(inv_flat, weights=xgrid.ravel(), minlength=len(ids))[sel]
        zw = counts[sel] * float(z)
        for i, rid in enumerate(ids_nz.tolist()):
            acc = sums.get(rid)
            if acc is None:
                sums[rid] = [float(counts[sel][i]), zw[i], yw[i], xw[i]]
            else:
                acc[0] += float(counts[sel][i])
                acc[1] += zw[i]
                acc[2] += yw[i]
                acc[3] += xw[i]
    centroids = {rid: (acc[1] / acc[0], acc[2] / acc[0], acc[3] / acc[0]) for rid, acc in sums.items()}
    counts_out = {rid: int(acc[0]) for rid, acc in sums.items()}
    return centroids, counts_out


def build_children_map(rows: list[RegionRow]) -> dict[int, list[int]]:
    """{直接父 id: [子 id 列表]}，path 为空（无谱系信息）的行不参与。"""
    children: dict[int, list[int]] = {}
    for row in rows:
        if len(row.path) >= 2:
            children.setdefault(int(row.path[-2]), []).append(row.region_id)
    return children


def collect_descendants(root_id: int, children_map: dict[int, list[int]]) -> set[int]:
    """root 的全部后代 id（不含 root 本身）。"""
    seen: set[int] = set()
    stack = list(children_map.get(root_id, ()))
    while stack:
        rid = stack.pop()
        if rid in seen:
            continue
        seen.add(rid)
        stack.extend(children_map.get(rid, ()))
    return seen


def resolve_display_set(
    region_id: int,
    present: set[int],
    children_map: dict[int, list[int]],
) -> list[int]:
    """点选脑区时实际要显示/跳转的 id 集合：本体有体素用本体；本体被抽稀
    抹掉则用存活子区的并集；子区也全灭则返回空（该行应从搜索列表剔除）。"""
    if region_id in present:
        return [region_id]
    alive = sorted(collect_descendants(region_id, children_map) & present)
    return alive


# --------------------------------------------------------------------------
# Qt widget
# --------------------------------------------------------------------------

_ZARR_THREADED_TUNED = False


def _tune_zarr_singlethread() -> None:
    """zarr3 的 async 线程池 × blosc 内部线程在 Qt 事件循环里偶发 MemoryError
    （非确定性，实测默认并发在 QTimer 回调中会随机挂）。label 读取量小，
    单线程化无感，换取稳定。"""
    global _ZARR_THREADED_TUNED
    if _ZARR_THREADED_TUNED:
        return
    try:
        import numcodecs.blosc as blosc_mod

        blosc_mod.set_nthreads(1)
    except Exception:
        pass
    try:
        import zarr

        zarr.config.set({"async.concurrency": 1})
    except Exception:
        pass
    _ZARR_THREADED_TUNED = True


def _guard_combo(combo, values, state):
    """monkeypatch 一个 magicgui ComboBox：reset 后若 choices 为空则回填。"""
    if combo is None or getattr(combo, "_atlas_guard", False) or not values:
        return
    original = combo.reset_choices

    def guarded_reset(*args, **kwargs):
        original(*args, **kwargs)
        try:
            if len(list(combo.choices)) == 0:
                combo.choices = values
                pos = state.get("sv_last_idx", 0)
                combo.value = values[pos if pos < len(values) else 0]
        except Exception:
            pass

    combo.reset_choices = guarded_reset
    combo._atlas_guard = True
    """monkeypatch 一个 magicgui ComboBox：reset 后若 choices 为空则回填。"""
    if combo is None or getattr(combo, "_atlas_guard", False) or not values:
        return
    original = combo.reset_choices

    def guarded_reset(*args, **kwargs):
        original(*args, **kwargs)
        try:
            if len(list(combo.choices)) == 0:
                combo.choices = values
                pos = state.get("sv_last_idx", 0)
                combo.value = values[pos if pos < len(values) else 0]
        except Exception:
            pass

    combo.reset_choices = guarded_reset
    combo._atlas_guard = True


def make_atlas_overlay_widget(
    viewer,
    *,
    label_zarr: str | Path | None = None,
    region_csv: str | Path | None = None,
    auto_load: bool = True,
):
    """制造图谱叠加 dock。label 跟随 viewer 里任意 3D 图像层（neurofly 等）。

    region_csv=None 时按 label 的 id 体系自动选 Rev2（小 id）或 Rev1 本地副本。
    """
    from qtpy import QtCore, QtWidgets

    state = {
        "label_zarr": Path(label_zarr) if label_zarr else None,
        "levels": [],
        "shapes": {},      # level -> shape_zyx
        "scales": {},      # level -> 当前世界 scale（随参考图像层更新）
        "centroids": {},   # {id: (z,y,x)}（质心扫描层一次算齐）
        "counts": {},      # {id: 体素数}
        "present": set(),
        "centroid_level": None,
        "display_map": {},  # {region_id: 实际显示的 id 集合}
        "usable_rows": [],
        "children": {},    # {parent_id: [child ids]}
        "rows": [],
        "id_to_row": {},
        "translate": (0.0, 0.0, 0.0),
        "sv_last_idx": 0,
        "combo_snap": {},   # id(panel) -> {id(combo): (combo, choices, value)}
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

    # QTimer 挂在 box 下：面板销毁时定时器随之失效，不再有僵尸回调
    debounce = QtCore.QTimer(box)
    debounce.setSingleShot(True)
    debounce.setInterval(250)
    cursor_debounce = QtCore.QTimer(box)
    cursor_debounce.setSingleShot(True)
    cursor_debounce.setInterval(150)
    # 周期巡检 neurofly 下拉框（napari 任意图层事件都可能清空它们）
    sweep = QtCore.QTimer(box)
    sweep.setInterval(1000)

    def _alive() -> bool:
        """面板关闭（Qt 对象销毁）后所有回调一律静默退出。"""
        try:
            _ = check_follow.isChecked()
            return True
        except RuntimeError:
            return False

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

    # SimpleViewer 强制 3D 显示，Labels 层 3D 渲染要吃整个体数据；
    # 超过这个体素数的 label 层级一律不挂（ot L5=10M ✓，L4=653M ✗）
    LABEL_VOXEL_BUDGET = 16_000_000

    def _wanted_level(ref) -> int:
        """label 层级 = 体素预算内的最细层，且尽量不超过 neurofly 层级。

        图像层比所有预算内 label 层都细（如 ims 加载后默认 L0）时，取预算内
        最细层——比图像粗只是 label 轮廓略糊，绝不能反向去物化 L0（1.3TB）。
        """
        if not state["levels"]:
            return 0
        img_level, _spacing, _spacing0 = _neurofly_context(ref)

        def voxels(k):
            s = state["shapes"][k]
            return s[0] * s[1] * s[2]

        within = [k for k in state["levels"] if voxels(k) <= LABEL_VOXEL_BUDGET]
        if not within:
            return state["levels"][-1]
        if img_level is not None:
            capped = [k for k in within if k <= img_level]
            return capped[-1] if capped else within[0]
        return within[0]

    def _rematch():
        if not _alive() or not (check_follow.isChecked() and state["levels"]):
            return
        try:
            ref = _reference_layer()
            if ref is None:
                return
            wanted = _wanted_level(ref)
            layer = None
            for lyr in viewer.layers:
                if lyr.name == LABEL_LAYER_NAME:
                    layer = lyr
                    break
            if layer is None or tuple(reversed(layer.data.shape)) != state["shapes"][wanted]:
                _attach_level(wanted, ref)
            else:
                _rescale_layer(layer, wanted, ref)
        except Exception as exc:  # noqa: BLE001 - 跟随逻辑绝不崩 GUI
            status.setText(f"跟随图像失败: {exc}")

    debounce.timeout.connect(_rematch)

    def _on_layers_inserted(event):
        if not _alive():
            return
        _watch_layer(event.value)
        _snapshot_neurofly_combos()
        debounce.start()

    layers_inserted_conn = viewer.layers.events.inserted.connect(_on_layers_inserted)

    def _on_layers_removed(event=None):
        if not _alive():
            return
        debounce.start()

    layers_removed_conn = viewer.layers.events.removed.connect(_on_layers_removed)
    for layer in viewer.layers:
        _watch_layer(layer)

    # ---------- 图谱加载 ----------

    def _rescale_layer(layer, level: int, ref):
        _, scale, translate = _label_transform(level, ref)
        layer.scale = scale
        layer.translate = translate
        layer.refresh()
        _restore_neurofly_combos()

    def _iter_neurofly_panels():
        """遍历 dock 里的 neurofly 面板（SimpleViewer / SegsAnnotator 等）。

        识别条件：同时有 image_layer 与 image 属性。候选顺序：dock_widgets 的
        value（可能是内层控件本身）→ .native → .widget()（Qt dock 包装时内层
        控件在 widget() 里）。
        """
        try:
            window = viewer.window
            docks = list(window.dock_widgets.values()) if hasattr(window, "dock_widgets") \
                else list(window._dock_widgets.values())
        except Exception:
            return
        for dock in docks:
            candidates = [dock, getattr(dock, "native", None)]
            if hasattr(dock, "widget"):
                try:
                    candidates.append(dock.widget())
                except Exception:
                    pass
            for candidate in candidates:
                if (
                    candidate is not None
                    and getattr(candidate, "image_layer", None) is not None
                    and getattr(candidate, "image", None) is not None
                ):
                    yield candidate
                    break

    def _panel_level_combo(panel):
        """面板的 resolution level 下拉框（SimpleViewer 叫 resolution_dropdown，
        SegsAnnotator 叫 level）。"""
        return getattr(panel, "resolution_dropdown", None) or getattr(panel, "level", None)

    def _panel_combos(panel):
        """容器里全部 ComboBox（level/channel/node_type/...），泛化守护用。"""
        try:
            return [w for w in panel if type(w).__name__ == "ComboBox"]
        except Exception:
            return []

    def _neurofly_widgets(ref):
        """产出 (panel, level_idx, spacing_xyz, spacing0)。

        level_idx 以面板自己的 level 下拉框为准（neurofly 的 ROI 数据零填充，
        图像层形状不可靠）；spacing 来自 panel.image.info（µm/体素，各层各轴
        非严格 2 倍），用于 label 的精确对齐。
        """
        for panel in _iter_neurofly_panels():
            image = getattr(panel, "image", None)
            image_layer = getattr(panel, "image_layer", None)
            if image is None:
                continue
            if image_layer is not None and ref is not None and image_layer is not ref:
                continue
            dropdown = _panel_level_combo(panel)
            try:
                levels = list(image.resolution_levels)
            except Exception:
                continue
            if not levels:
                continue
            idx = None
            if dropdown is not None:
                value = dropdown.value
                idx = levels.index(value) if value in levels else None
            spacing = None
            spacing0 = None
            info = getattr(image, "info", None)
            if isinstance(info, list) and idx is not None and idx < len(info):
                spacing = info[idx].get("spacing")
                if info:
                    spacing0 = info[0].get("spacing")
            yield panel, idx, spacing, spacing0

    def _neurofly_context(ref):
        for _panel, idx, spacing, spacing0 in _neurofly_widgets(ref):
            if idx is not None:
                state["sv_last_idx"] = idx
                return idx, spacing, spacing0
        return None, None, None

    # napari 图层事件会清空 dock 里所有 magicgui ComboBox 的 choices（neurofly
    # 的 choices 源是一次性的，清空即死：value=None、refresh 全崩，SegsAnnotator
    # 还会去读未加载的 db）。对策：对每个 neurofly 面板快照全部 ComboBox 的
    # choices+value，发现被清空就用 Qt blockSignals 静默回填（不触发任何 handler）。
    def _snapshot_neurofly_combos():
        for panel in _iter_neurofly_panels():
            snap = state["combo_snap"].setdefault(id(panel), {})
            for combo in _panel_combos(panel):
                try:
                    choices = list(combo.choices)
                except Exception:
                    continue
                if choices:
                    snap[id(combo)] = (combo, choices, combo.value)

    def _restore_neurofly_combos():
        for panel in _iter_neurofly_panels():
            snap = state["combo_snap"].get(id(panel), {})
            if snap:
                for _cid, (combo, choices, value) in list(snap.items()):
                    try:
                        if len(list(combo.choices)) != 0:
                            continue
                        # psygnal 信号级屏蔽：Qt blockSignals 挡不住 magicgui 内部直发；
                        # 不触发面板 handler（Annotator 未选 db 时 on_changing_level 会炸）
                        with combo.changed.blocked():
                            combo.choices = list(choices)
                            if value is not None and value in choices:
                                combo.value = value
                    except Exception:
                        continue
            _restore_key_combos_from_image(panel)

    def _restore_key_combos_from_image(panel):
        """无快照兜底：level / channel 两个关键框直接从 panel.image 回填。"""
        image = getattr(panel, "image", None)
        if image is None:
            return
        try:
            levels = list(image.resolution_levels)
            channels = list(image.channels)
        except Exception:
            return
        level_combo = _panel_level_combo(panel)
        channel_combo = getattr(panel, "channel_dropdown", None) or getattr(panel, "channel", None)
        if level_combo is not None and levels and len(list(level_combo.choices)) == 0:
            try:
                with level_combo.changed.blocked():
                    level_combo.choices = levels
                    pos = state.get("sv_last_idx", 0)
                    level_combo.value = levels[pos if pos < len(levels) else 0]
            except Exception:
                pass
        if channel_combo is not None and channels and len(list(channel_combo.choices)) == 0:
            try:
                with channel_combo.changed.blocked():
                    channel_combo.choices = channels
                    channel_combo.value = channels[0]
            except Exception:
                pass

    def _label_transform(level: int, ref):
        """label 层的 (scale, translate, data)。

        全系统 µm 世界：label 层 k 的体素物理尺寸恒为 2^k × spacing(L0)，与
        图像当前层级无关（neurofly 各层 scale 也已被补丁设为 µm spacing）。
        数据用 dask 懒转置成 XYZ，同步物化后交给 napari。
        """
        import dask.array as da

        _tune_zarr_singlethread()
        arr = da.from_zarr(str(state["label_zarr"] / str(level))).transpose(2, 1, 0)
        # napari 3D 模式会整体物化 Labels 层；dask 多线程并发解压 chunk 缓冲
        # 会瞬间申请大量内存。预算内的层本来就 ≤16M 体素，同步物化成 numpy
        # 一次到位；邻居进程（渲染类）瞬时冲击 commit 时重试一次。
        import time as _time

        for attempt in range(2):
            try:
                arr = arr.compute(scheduler="synchronous")
                break
            except MemoryError:
                if attempt:
                    raise
                _time.sleep(1.5)
        translate = (0.0, 0.0, 0.0)
        if ref is not None:
            img_level, spacing, spacing0 = _neurofly_context(ref)
            if (
                spacing0 and len(spacing0) == 3 and all(s and s > 0 for s in spacing0)
            ):
                # 全系统 µm 世界：label 层 k 的体素物理尺寸 = 2^k × spacing(L0)，
                # 恒定，与图像当前层级无关（图像层 scale 也是 µm spacing）
                scale = tuple(2.0**level * float(s0) for s0 in spacing0)  # xyz
            else:
                scale = tuple(2.0**level for _ in range(3))
        else:
            scale = tuple(2.0**level for _ in range(3))
        state["scales"][level] = scale
        state["translate"] = translate
        return arr, scale, translate

    def _attach_level(level: int, ref=None):
        arr, scale, translate = _label_transform(level, ref)
        existing = None
        for lyr in viewer.layers:
            if lyr.name == LABEL_LAYER_NAME:
                existing = lyr
                break
        if existing is not None:
            existing.data = arr
            existing.scale = scale
            existing.translate = translate
            existing.refresh()
        else:
            viewer.add_labels(
                arr, name=LABEL_LAYER_NAME, scale=scale, translate=translate,
                opacity=0.35, blending="translucent",
            )
        _restore_neurofly_combos()  # napari 图层事件会清空 neurofly 下拉框
        shape_zyx = state["shapes"][level]
        status.setText(
            f"图谱 L{level}（{shape_zyx[0]}×{shape_zyx[1]}×{shape_zyx[2]} zyx，"
            f"已转 XYZ 对齐 {ref.name if ref else '占位'}，scale "
            f"{tuple(round(v, 4) for v in scale)}，translate {tuple(round(v, 1) for v in translate)}）"
        )

    def _load_label(path: str | Path | None = None):
        if path is None:
            # zarr 是目录，必须用选目录对话框（getOpenFileName 选不中）
            picked = QtWidgets.QFileDialog.getExistingDirectory(
                box, "选择 atlas label zarr 目录（如 upsampled_atlas_label.zarr）",
                "",
            )
            if not picked:
                return
            path = picked
        path = Path(path)
        if not (path / ".zgroup").exists() and not (path / ".zattrs").exists():
            # 误选了样本根目录：自动找里面的 label zarr
            for name in ("upsampled_atlas_label.zarr", "upsampled_label_ngff.zarr"):
                candidate = path / name
                if (candidate / ".zgroup").exists() or (candidate / ".zattrs").exists():
                    path = candidate
                    break
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

        # ---- 质心扫描层：内存预算内最细的一层，一次向量化扫描出全部存活 id ----
        _tune_zarr_singlethread()
        scan_level = levels[-1]
        for k in levels:
            shape = state["shapes"][k]
            itemsize = np.dtype(group[str(k)].dtype).itemsize
            if int(np.prod(shape)) * itemsize <= CENTROID_SCAN_BUDGET:
                scan_level = k
                break
        import dask.array as da

        scan_arr = np.asarray(
            da.from_zarr(str(path / str(scan_level))).compute(scheduler="synchronous")
        )
        centroids, counts = scan_level_centroids(scan_arr)
        del scan_arr
        present = set(centroids)
        state["centroid_level"] = scan_level
        state["centroids"] = centroids
        state["counts"] = counts
        state["present"] = present

        # ---- CSV 按 label 的 id 体系自动选择（Rev2 小 id vs Rev1 官方大 id）----
        csv_path = region_csv
        if csv_path is None:
            max_id = max(present) if present else 0
            if max_id <= 2000:
                csv_path = DEFAULT_REGION_CSV
            else:
                csv_path = next(
                    (p for p in LEGACY_REGION_CSV_CANDIDATES if Path(p).exists()),
                    DEFAULT_REGION_CSV,
                )
        rows = load_region_rows(csv_path)
        state["rows"] = rows
        state["id_to_row"] = {r.region_id: r for r in rows}
        children_map = build_children_map(rows)
        state["children"] = children_map

        # ---- 可用性解析：本体无体素 → 子区并集；子区也全灭 → 从搜索列表剔除 ----
        display_map: dict[int, list[int]] = {}
        usable_rows = []
        for row in rows:
            display = resolve_display_set(row.region_id, present, children_map)
            if display:
                display_map[row.region_id] = display
                usable_rows.append(row)
        state["display_map"] = display_map
        state["usable_rows"] = usable_rows
        state["scales"] = {}
        ref = _reference_layer()
        if ref is not None:
            _attach_level(_wanted_level(ref), ref)
        else:
            status.setText(
                f"图谱就绪（{len(levels)} 层金字塔，扫描层 L{scan_level}："
                f"{len(present)} 个 id 存活，{len(usable_rows)}/{len(rows)} 区可搜索，"
                f"CSV: {Path(csv_path).name}）。在 neurofly / 任意方式打开 3D 图像后自动叠加。"
            )

    btn_label.clicked.connect(lambda: _load_label())

    def _manual_rematch():
        ref = _reference_layer()
        if ref is None or not state["levels"]:
            status.setText("没有可参考的 3D 图像层（先在 neurofly / 其他方式加载图像）")
            return
        _attach_level(_wanted_level(ref), ref)

    btn_reload.clicked.connect(_manual_rematch)

    # ---------- 搜索 / 跳转 ----------

    def _refresh_results():
        list_results.clear()
        for row in filter_regions(state["usable_rows"], edit_search.text()):
            list_results.addItem(row.text)

    edit_search.textChanged.connect(_refresh_results)

    def _centroid_world(display: list[int]):
        """显示 id 集合的加权质心 → 世界坐标（XYZ）。全部数据加载时预扫好了，
        这里只是字典查询，O(1)。"""
        counts = state["counts"]
        centroids = state["centroids"]
        total = sum(counts.get(i, 0) for i in display)
        if total <= 0:
            return None
        cz = sum(centroids[i][0] * counts[i] for i in display if i in counts) / total
        cy = sum(centroids[i][1] * counts[i] for i in display if i in counts) / total
        cx = sum(centroids[i][2] * counts[i] for i in display if i in counts) / total
        level = state["centroid_level"]
        ref = _reference_layer()
        scale = None
        if ref is not None:
            _img_level, spacing, spacing0 = _neurofly_context(ref)
            if (
                spacing and len(spacing) == 3 and all(s and s > 0 for s in spacing)
                and spacing0 and len(spacing0) == 3 and all(s and s > 0 for s in spacing0)
            ):
                scale = [2.0**level * s0 / s for s0, s in zip(spacing0, spacing)]  # xyz
        if scale is None:
            scale = [2.0**level for _ in range(3)]
        translate = state.get("translate", (0.0, 0.0, 0.0))
        world_xyz = (
            translate[0] + cx * scale[0],
            translate[1] + cy * scale[1],
            translate[2] + cz * scale[2],
        )
        return world_xyz

    def _set_isolation(display: list[int]):
        """只显示给定 id 集合的 label（其余存活 id 全透明），便于在脑区内选 goal。"""
        layer = None
        for lyr in viewer.layers:
            if lyr.name == LABEL_LAYER_NAME:
                layer = lyr
                break
        if layer is None:
            return
        present = state["present"]
        wanted = set(display) & present
        if not wanted:
            return
        try:
            from napari.utils.colormaps import DirectLabelColormap

            if "orig_colormap" not in state:
                state["orig_colormap"] = layer.colormap
            transparent = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
            highlight = np.array([1.0, 0.35, 0.25, 1.0], dtype=np.float32)
            color_dict = {int(rid): transparent for rid in present}
            color_dict[0] = transparent
            color_dict[None] = transparent  # 字典外的未知 id 一律透明
            for rid in wanted:
                color_dict[int(rid)] = highlight
            layer.colormap = DirectLabelColormap(color_dict=color_dict)
            state["isolated"] = sorted(wanted)
        except Exception as exc:  # noqa: BLE001
            state.pop("isolated", None)
            state["isolate_error"] = str(exc)

    def _clear_isolation():
        layer = None
        for lyr in viewer.layers:
            if lyr.name == LABEL_LAYER_NAME:
                layer = lyr
                break
        if layer is None:
            return
        try:
            if "orig_colormap" in state:
                layer.colormap = state.pop("orig_colormap")
        except Exception as exc:  # noqa: BLE001
            status.setText(f"恢复全部显示失败: {exc}")

    btn_show_all = QtWidgets.QPushButton("显示全部脑区")
    layout.insertWidget(layout.indexOf(list_results) + 1, btn_show_all)
    btn_show_all.clicked.connect(lambda: _clear_isolation() if _alive() else None)

    def _jump(row: RegionRow):
        display = state["display_map"].get(row.region_id) or []
        world = _centroid_world(display) if display else None
        if world is None:
            status.setText(f"{row.text} 在此 label 中无覆盖（本体与子区均无体素）")
            return
        if viewer.dims.ndisplay == 2:
            viewer.dims.point = world
        else:
            viewer.camera.center = world
        _set_isolation(display)
        extra = "" if len(display) == 1 else f"（本体无体素，显示 {len(display)} 个子区）"
        if state.get("isolated"):
            status.setText(f"跳转 → {row.text}，仅显示该脑区{extra}；点「显示全部脑区」恢复")
        else:
            status.setText(
                f"跳转 → {row.text}（隔离显示失败: {state.get('isolate_error', 'unknown')}）"
            )

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
        if not _alive():
            return
        cursor_debounce.start()

    def _read_cursor_region():
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

    cursor_debounce.timeout.connect(_read_cursor_region)
    cursor_conn = viewer.cursor.events.position.connect(_on_cursor)

    # 先给已存在的 neurofly 面板拍快照再挂 label：清空 choices 的正是我们的
    # 第一次图层插入，晚于此的快照只能拍到空列表
    _snapshot_neurofly_combos()
    if auto_load and state["label_zarr"] is not None and Path(state["label_zarr"]).exists():
        QtCore.QTimer.singleShot(0, lambda: _load_label(state["label_zarr"]) if _alive() else None)

    if auto_load and state["label_zarr"] is not None and Path(state["label_zarr"]).exists():
        QtCore.QTimer.singleShot(0, lambda: _load_label(state["label_zarr"]) if _alive() else None)

    def _sweep():
        if not _alive():
            return
        _snapshot_neurofly_combos()
        _restore_neurofly_combos()
        for widget, idx, _spacing, _spacing0 in _neurofly_widgets(None):
            try:
                rewire_neurofly_double_click(widget)
                # 完整纠正：image/点/边/goal 层 µm scale + panorama 粗层 +
                # panorama 点层（hr 口径）——与类补丁的 _fix_layer_scales 同一份逻辑
                _fix_layer_scales(widget)
            except Exception:
                continue

    sweep.timeout.connect(_sweep)
    sweep.start()

    def _cleanup():
        """面板销毁时断开 viewer 级连接，杜绝僵尸回调引用已删 Qt 对象。"""
        for conn in (layers_inserted_conn, layers_removed_conn, cursor_conn):
            try:
                conn.disconnect()
            except Exception:
                pass
        debounce.stop()
        cursor_debounce.stop()
        sweep.stop()

    box.destroyed.connect(lambda _o=None: _cleanup())

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


_SCALE_PATCHED = set()


def patch_neurofly_display_scale() -> bool:
    """类级补丁：包住 Annotator.refresh / SimpleViewer.refresh，刷新后把
    image 层与 panorama 层的 scale 设为该层真实 spacing（µm/体素）。

    neurofly 的层 scale 假设金字塔各向同性（源码 TODO 承认），z 向 2µm 的
    MegaSpim 数据会压扁。类级补丁不依赖面板实例发现（Plugins 菜单加载的
    实例被 Qt 包装层藏住，实例级遍历找不到）。"""
    try:
        from neurofly import segs_annotator, simple_viewer

        targets = []
        if hasattr(segs_annotator.Annotator, "refresh") \
                and "Annotator.refresh" not in _SCALE_PATCHED:
            targets.append((segs_annotator.Annotator, "refresh", "Annotator.refresh"))
        if hasattr(segs_annotator.Annotator, "refresh_panorama") \
                and "Annotator.refresh_panorama" not in _SCALE_PATCHED:
            # panorama 重建会重设 panorama 层 scale（各向同性假设）→ 每次重建后纠正
            targets.append((segs_annotator.Annotator, "refresh_panorama", "Annotator.refresh_panorama"))
        if hasattr(simple_viewer.SimpleViewer, "refresh") \
                and "SimpleViewer.refresh" not in _SCALE_PATCHED:
            targets.append((simple_viewer.SimpleViewer, "refresh", "SimpleViewer.refresh"))

        for cls, name, key in targets:
            original = getattr(cls, name)

            def patched(self, *args, _orig=original, **kwargs):
                result = _orig(self, *args, **kwargs)
                try:
                    _fix_layer_scales(self)
                except Exception:
                    pass
                return result

            patched._atlas_scale_guard = True  # type: ignore[attr-defined]
            setattr(cls, name, patched)
            _SCALE_PATCHED.add(key)
        return bool(targets)
    except Exception:
        return False


def _fix_layer_scales(panel) -> None:
    """把面板的 image 层与 panorama 层 scale 设为真实 spacing（µm/体素）。"""
    image = getattr(panel, "image", None)
    info = getattr(image, "info", None)
    if not isinstance(info, list) or not info:
        return
    image_layer = getattr(panel, "image_layer", None)
    if image_layer is None:
        return
    idx = 0
    dropdown = getattr(panel, "resolution_dropdown", None) or getattr(panel, "level", None)
    try:
        levels = list(image.resolution_levels)
        value = dropdown.value if dropdown is not None else None
        if value in levels:
            idx = levels.index(value)
    except Exception:
        pass
    if idx >= len(info):
        return
    spacing = info[idx].get("spacing")
    if spacing and len(spacing) == 3 and all(s and s > 0 for s in spacing):
        new_scale = tuple(float(v) for v in spacing)
        if not np.allclose(np.asarray(image_layer.scale, dtype=float),
                           np.asarray(new_scale, dtype=float), rtol=0.01):
            image_layer.scale = new_scale
        # db/标注坐标是层级体素索引，neurofly 靠全层 scale=1 对齐；image 层改成
        # µm 后，当前 level 的配套层（点/边/goal）必须同步同一 scale
        for attr in ("point_layer", "edge_layer", "ex_edge_layer", "goal_layer"):
            layer = getattr(panel, attr, None)
            if layer is not None and not np.allclose(
                    np.asarray(layer.scale, dtype=float), np.asarray(new_scale, dtype=float), rtol=0.01):
                layer.scale = new_scale

    # panorama 图像层是粗层数据 → scale 用 pano 层 spacing；而 panorama 的
    # db 点层坐标是 hr 层绝对体素（与 point_layer 同口径）→ scale 必须用
    # spacing_hr（origin 全 0 时 world = 坐标×spacing，两层在同一 µm 世界）
    pano = getattr(panel, "panorama_image", None)
    if pano is not None:
        try:
            pano_level = next(
                i for i, roi in enumerate(image.rois)
                if (np.array(roi[3:]) < np.array([1000, 1000, 1000])).all()
            )
            pano_spacing = info[pano_level].get("spacing")
            if pano_spacing and len(pano_spacing) == 3 and all(s and s > 0 for s in pano_spacing) \
                    and not np.allclose(np.asarray(pano.scale, dtype=float),
                                        np.asarray(pano_spacing, dtype=float), rtol=0.01):
                pano.scale = tuple(float(v) for v in pano_spacing)
            # panorama 的 db 点层坐标是 hr 层绝对体素（与 point_layer 同口径）
            # → scale 用 hr spacing（µm），与 pano 图像同处一个 µm 世界
            for attr in ("panorama_points", "panorama_view"):
                layer = getattr(panel, attr, None)
                if layer is not None and spacing and not np.allclose(
                        np.asarray(layer.scale, dtype=float),
                        np.asarray(spacing, dtype=float), rtol=0.01):
                    layer.scale = tuple(float(v) for v in spacing)
        except (StopIteration, Exception):
            pass


def patch_neurofly_double_click() -> bool:
    """兜住 neurofly SimpleViewer.on_double_click 的 None 崩溃（ray 未命中时
    near/far 为 None，原实现直接相减抛 TypeError）。返回是否打了补丁。

    只换类属性还不够：SimpleViewer 构造时就把原始 bound method 塞进了
    image 层的 mouse_double_click_callbacks 列表，后开的插件必须在列表里
    原位替换那份引用——由 rewire_neurofly_double_click(widget) 完成，widget
    的 1s 巡检会自动对每个发现的 neurofly 面板做一次。
    """
    try:
        from neurofly import simple_viewer

        if getattr(simple_viewer.SimpleViewer.on_double_click, "_atlas_guard", False):
            return True
        original = simple_viewer.SimpleViewer.on_double_click

        def guarded(self, layer, event):
            near, far = layer.get_ray_intersections(
                event.position, event.view_direction, event.dims_displayed
            )
            if near is None or far is None:
                return  # ray 未命中（点击处无图像体素），neurofly 原实现会崩
            return original(self, layer, event)

        guarded._atlas_guard = True  # type: ignore[attr-defined]
        simple_viewer.SimpleViewer.on_double_click = guarded
        return True
    except Exception:
        return False


def rewire_neurofly_double_click(sv_widget) -> bool:
    """把 sv 面板 image 层回调列表里已登记的原始 on_double_click 原位换成
    带 None 守卫的版本（列表持有的是构造时的 bound method，事后改类属性
    影响不到它）。"""
    layer = getattr(sv_widget, "image_layer", None)
    callbacks = getattr(layer, "mouse_double_click_callbacks", None)
    if not callbacks:
        return False
    rewired = False
    for index, cb in enumerate(list(callbacks)):
        func = getattr(cb, "__func__", None)
        qual = getattr(func, "__qualname__", "") if func is not None else ""
        if not qual.endswith("SimpleViewer.on_double_click") or getattr(func, "_atlas_guard", False):
            continue

        def guarded(layer_, event, _orig=cb):
            near, far = layer_.get_ray_intersections(
                event.position, event.view_direction, event.dims_displayed
            )
            if near is None or far is None:
                return
            return _orig(layer_, event)

        guarded._atlas_guard = True  # type: ignore[attr-defined]
        callbacks[index] = guarded
        rewired = True
    return rewired


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sample-dir", help="样本目录：自动加载 upsampled_atlas_label.zarr")
    parser.add_argument("--label-zarr", help="atlas label zarr 路径（默认 <sample-dir>/upsampled_atlas_label.zarr）")
    parser.add_argument("--region-csv", default=None, help="区域 CSV；缺省按 label id 体系自动选 Rev2/Rev1")
    parser.add_argument("--image", help="ims/zarr 图像：创建 neurofly 面板并自动加载（配合 --panel）")
    parser.add_argument("--db", help="分割 db：随 --image 一起填入面板")
    parser.add_argument("--panel", choices=["annotator", "simple", "none"], default="none",
                        help="随启动创建的 neurofly 面板（默认 none）")
    parser.add_argument("--level", default=None, help="面板的 resolution level（如 '2' 或 'ResolutionLevel 2'）")
    args = parser.parse_args(argv)

    sample_dir = Path(args.sample_dir) if args.sample_dir else None
    label_zarr = Path(args.label_zarr) if args.label_zarr else _find_label_zarr(sample_dir)

    import napari

    patch_neurofly_display_scale()
    patched = patch_neurofly_double_click()
    viewer = napari.current_viewer() or napari.Viewer()

    def _safe_assign(panel, attr, value):
        """面板半初始化状态下的赋值可能触发 neurofly handler 报错（EmitLoopError），
        GUI 里是日志噪音；自动化里吞掉，最后手动补一次刷新。"""
        try:
            setattr(panel, attr, value)
        except Exception as exc:  # noqa: BLE001
            print(f"[atlas-overlay] assign {attr} raised {type(exc).__name__} (logged)", flush=True)

    if args.panel != "none":
        if args.panel == "annotator":
            from neurofly.segs_annotator import Annotator

            panel = Annotator(viewer)
            viewer.window.add_dock_widget(panel, area="left", name="Segs Annotator")
            if args.db:
                _safe_assign(panel, "db_path", args.db)
        else:
            from neurofly.simple_viewer import SimpleViewer

            panel = SimpleViewer(viewer)
            viewer.window.add_dock_widget(panel, area="left", name="Simple Viewer")
        if args.image:
            _safe_assign(panel, "image_path", args.image)
        pump_ms = 8000 if args.image else 1000
        try:
            from qtpy.QtWidgets import QApplication

            end = time.time() + pump_ms / 1000
            while time.time() < end:
                QApplication.processEvents()
        except Exception:
            pass
        if args.image and args.level is not None:
            combo = getattr(panel, "resolution_dropdown", None) or getattr(panel, "level", None)
            if combo is not None:
                try:
                    choices = list(combo.choices)
                except Exception:
                    choices = []
                value = args.level
                if str(args.level).isdigit() and choices:
                    match = [c for c in choices if c.endswith(f" {args.level}")]
                    value = match[0] if match else str(args.level)
                _safe_assign(combo, "value", value)
        # 图像+db 都就位后补一次 db 刷新（初始化顺序噪音的自愈）
        if args.db and hasattr(panel, "on_reading_db"):
            try:
                panel.on_reading_db()
            except Exception as exc:  # noqa: BLE001
                print(f"[atlas-overlay] db refresh raised {type(exc).__name__} (logged)", flush=True)
        rewire_neurofly_double_click(panel)

    widget = make_atlas_overlay_widget(
        viewer, label_zarr=label_zarr, region_csv=args.region_csv, auto_load=True,
    )
    # 右侧 dock：neurofly 的面板在左侧，避免相互挤掉
    viewer.window.add_dock_widget(widget, area="right", name="图谱叠加")
    if patched:
        print("neurofly 双击兼容补丁已应用")
    napari.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
