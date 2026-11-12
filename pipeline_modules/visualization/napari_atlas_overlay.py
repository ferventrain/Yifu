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
REGION_NODES_LAYER = "region_nodes"
# 上次使用的路径持久化（记住 image/db，免得每次手填）
PATHS_JSON = Path(__file__).resolve().parents[2] / "_tmp" / "atlas_overlay_paths.json"
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


def _parse_coord_value(coord):
    """单个 coord 值 → [x, y, z]（float list）或 None。

    兼容三种写法：python list（原生 segs2db）、sqlite3.Binary 读回的 bytes、
    np 数组 str()（"[np.int64(2282), ...]" / "[ 47 190 69]"，shard 老数据
    literal_eval 解析不了——恰好 3 个数才按坐标收）。"""
    import ast
    import re

    values = None
    try:
        if isinstance(coord, bytes):
            coord = coord.decode("utf-8")
        parsed = ast.literal_eval(coord)
        if isinstance(parsed, (list, tuple)) and len(parsed) == 3:
            values = parsed
    except Exception:
        values = None
    if values is None:
        # (?<![\w.]) 挡掉 "np.int64(12)" 里 int64 的 64 被当坐标数字
        nums = re.findall(r"(?<![\w.])-?\d+(?:\.\d+)?(?:e[+-]?\d+)?", coord)
        if len(nums) == 3:
            values = nums
    if values is None:
        return None
    try:
        return [float(v) for v in values]
    except (TypeError, ValueError):
        return None


def parse_db_node_coords(rows) -> np.ndarray:
    """neurofly nodes 表 (coord,) 行 → (N,3) float，坐标口径 = 层级体素 **XYZ**。

    实锤（2026-11-04，RRR neurofly_segments.db 81.6 万节点 vs label L1 形状
    (2238,16376,9159)）：neurofly from_roi 把数据转置成 XYZ，db 按 x,y,z 存——
    按 zyx 解释第三列当 z 会越界。"""
    coords = []
    for (coord,) in rows:
        v = _parse_coord_value(coord)
        if v is not None:
            coords.append(v)
    return np.asarray(coords, dtype=np.float64) if coords else np.zeros((0, 3))


def load_db_nodes_and_segments(db_path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """neurofly db → (nids (N,), coords (N,3) 层级体素 XYZ, seg (N,) 段代表 nid)。

    seg 按 edges 表并查集分组：连通的节点 = 同一神经元段。"从脑区开始追一群
    神经元"的遍历单位是段（=神经元），节点只是段上的采样点。坏行节点（坐标
    解析失败）跳过；edges 引用缺失 nid 的边忽略。"""
    import sqlite3

    conn = sqlite3.connect(str(db_path))
    try:
        rows = conn.execute("SELECT nid, coord FROM nodes").fetchall()
        edges = conn.execute("SELECT src, des FROM edges").fetchall()
    finally:
        conn.close()
    nids: list[int] = []
    coords: list[list[float]] = []
    for nid, coord in rows:
        v = _parse_coord_value(coord)
        if v is not None:
            nids.append(int(nid))
            coords.append(v)
    parent = {n: n for n in nids}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for src, des in edges:
        try:
            a, b = find(int(src)), find(int(des))
        except KeyError:
            continue
        if a != b:
            parent[a] = b
    arr = np.asarray(coords, dtype=np.float64) if coords else np.zeros((0, 3))
    id_arr = np.asarray(nids, dtype=np.int64) if nids else np.zeros(0, dtype=np.int64)
    seg_arr = np.asarray([find(n) for n in nids], dtype=np.int64) if nids else np.zeros(0, dtype=np.int64)
    return id_arr, arr, seg_arr


def nodes_in_region_mask(
    nodes_xyz: np.ndarray,
    scan_arr_zyx,
    display_ids,
    node_level: int,
    scan_level: int,
) -> np.ndarray:
    """db 节点（node_level 体素 XYZ）落在给定 id 集合内的布尔 mask。

    金字塔各轴 ×2/层（RRR 实测 L1→L5 三轴比值差 <0.1%），单标量因子换算；
    查找在 zarr 原生 zyx 数组上做。db 本身不记录层级（推理产物=L1；手工库
    是 Annotator 面板当时层级）——层级用错症状是整体差 2 的倍数。"""
    if len(nodes_xyz) == 0:
        return np.zeros(0, dtype=bool)
    pts = nodes_xyz * (2.0 ** (node_level - scan_level))
    zi = np.clip(pts[:, 2].astype(np.int64), 0, scan_arr_zyx.shape[0] - 1)
    yi = np.clip(pts[:, 1].astype(np.int64), 0, scan_arr_zyx.shape[1] - 1)
    xi = np.clip(pts[:, 0].astype(np.int64), 0, scan_arr_zyx.shape[2] - 1)
    values = scan_arr_zyx[zi, yi, xi]
    return np.isin(values, np.asarray(display_ids, dtype=values.dtype))


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
    db_path: str | Path | None = None,
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
        "nodes_l1": None,   # (N,3) db 节点坐标（L1 体素 XYZ，neurofly from_roi 口径）
        "node_ids": None,   # (N,) nid，与 nodes_l1 同序
        "node_seg": None,   # (N,) 段代表 nid（edges 并查集；段=神经元）
        "db_path": None,
        "label_level": None,  # label 层当前挂载的金字塔层级（节点显示换算用）
        "scan_arr": None,   # 质心扫描层数组缓存（节点成员判定）
        "region_iter": None,  # 当前脑区的段遍历状态（「下一神经元」）
        "region_nodes_idx": None,  # 环层第 i 点 → nodes_l1 行号（点环选择用）
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

    # --- 分割 db 节点高亮 ---
    row_db = QtWidgets.QHBoxLayout()
    edit_db = QtWidgets.QLineEdit()
    edit_db.setPlaceholderText("db 路径（如 neurofly_segments.db，可留空自动找）")
    _db_prefill = db_path or _load_paths_json().get("db")
    if _db_prefill:
        edit_db.setText(str(_db_prefill))
    btn_nodes = QtWidgets.QPushButton("载入节点")
    row_db.addWidget(edit_db, 1)
    row_db.addWidget(btn_nodes)
    layout.addLayout(row_db)

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
        state["label_level"] = level
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
            prev_active = viewer.layers.selection.active
            viewer.add_labels(
                arr, name=LABEL_LAYER_NAME, scale=scale, translate=translate,
                opacity=0.35, blending="translucent",
            )
            # napari 自动把新层设为 active，会抢走 neurofly point_layer 的选中
            # （按 d 变成 viewer 级 refresh → 中心点乱切）；恢复原选中
            if prev_active is not None and prev_active.name != LABEL_LAYER_NAME:
                viewer.layers.selection.active = prev_active
            elif prev_active is None:
                viewer.layers.selection.active = viewer.layers["image"] if "image" in viewer.layers else None
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
        state["scan_arr"] = scan_arr
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
        _refresh_results()  # 加载完成即刷新列表——加载期间打过字的搜索框别留空
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

    btn_nodes.clicked.connect(lambda: _load_nodes_db(edit_db.text().strip()) if _alive() else None)
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
        _set_panorama_dim(False)
        state["region_iter"] = None
        state["region_nodes_idx"] = None
        for lyr in viewer.layers:
            if lyr.name == REGION_NODES_LAYER:
                lyr.data = np.zeros((0, 3))
                lyr.refresh()
                break
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

    def _resolve_db(path_text: str) -> Path | None:
        p = Path(path_text) if path_text else None
        if p is None or not p.exists():
            return None
        if p.is_file() and p.suffix == ".db":
            return p
        for name in ("neurofly_segments.db", "RRR_manual.db"):
            candidate = p / name
            if candidate.exists():
                return candidate
        hits = sorted(p.glob("*.db"))
        return hits[0] if hits else None

    def _load_nodes_db(path_text: str):
        resolved = _resolve_db(path_text)
        if resolved is None:
            status.setText("未找到 .db 文件：填样本目录或直接指向 .db")
            return
        n = _load_nodes(str(resolved))
        state["db_path"] = resolved
        _save_paths_json({"db": str(resolved)})
        n_seg = int(np.unique(state["node_seg"]).size) if n else 0
        status.setText(
            f"已载入 {n} 节点 / {n_seg} 神经元段（{resolved.name}）；"
            "点选脑区后用「下一神经元」逐段追"
        )

    def _load_nodes(db_path: str):
        """读 db 全部节点（L1 体素 XYZ）+ edges 并查集分段，缓存进 state。"""
        nids, coords, segs = load_db_nodes_and_segments(db_path)
        state["node_ids"] = nids
        state["nodes_l1"] = coords
        state["node_seg"] = segs
        return len(coords)

    def _update_region_nodes_layer(display: list[int]):
        """脑区内节点 → 可点击高亮点层。

        坐标直接用 **µm 世界**（= L1 体素 × 2 × spacing0，与显示层级无关），
        层 scale=(1,1,1)——napari 3D 里 marker 实际直径 = size × scale[-1]
        （vispy points.py #5582），跟 label 层走 scale[-1]=128µm 会把 size=3
        放大成 ~0.4mm 巨环；归一后 size 即纯世界 µm。size=6µm ≈ neurofly
        soma 标记（8）和普通节点（1-2）之间。点击环 = 选中该节点进 labeling。"""
        scan_arr = state.get("scan_arr")
        nodes_l1 = state.get("nodes_l1")
        label_layer = None
        for lyr in viewer.layers:
            if lyr.name == LABEL_LAYER_NAME:
                label_layer = lyr
                break
        if scan_arr is None or nodes_l1 is None or len(nodes_l1) == 0 or label_layer is None:
            inside = np.zeros((0, 3))
            idxs = np.zeros(0, dtype=np.int64)
        else:
            mask = nodes_in_region_mask(nodes_l1, scan_arr, display, 1, state["centroid_level"])
            idxs = np.nonzero(mask)[0]
            disp_level = state.get("label_level")
            if disp_level is None:
                disp_level = state["centroid_level"]
            # world µm = 显示层体素 × 该层 µm spacing = L1 体素 × 2 × spacing0
            um_per_disp_voxel = np.asarray(label_layer.scale, dtype=float)
            inside = nodes_l1[idxs] * (um_per_disp_voxel * (2.0 ** (1 - disp_level)))
        state["region_nodes_idx"] = idxs
        pts = None
        for lyr in viewer.layers:
            if lyr.name == REGION_NODES_LAYER:
                pts = lyr
                break
        if pts is None:
            prev_active = viewer.layers.selection.active
            pts = viewer.add_points(
                np.zeros((0, 3)), name=REGION_NODES_LAYER,
                face_color="#00ff88", size=6, symbol="ring", opacity=0.95,
                border_width=0, border_color="#004400",
            )
            pts.mouse_drag_callbacks.append(_on_ring_click)
            # napari 自动激活新层会抢走 neurofly point_layer 选中（d 失灵的
            # 老路数），同 label 层一样恢复
            if prev_active is not None and prev_active.name != REGION_NODES_LAYER:
                viewer.layers.selection.active = prev_active
        pts.data = inside
        pts.scale = (1.0, 1.0, 1.0)
        pts.translate = (0.0, 0.0, 0.0)
        pts.size = 6
        pts.refresh()
        return len(inside)

    def _trace_node(nid: int) -> str:
        """程序化"选点+按 q"：设 selected_node → 进 labeling（全景时）或重新
        切块居中（labeling 时，keep_image=False）。

        可行性已实证（2026-11-04 offscreen）：selected_node.value 赋值走 GUI
        同链路；switch_mode/refresh 后 active 层自动回到 neurofly point_layer，
        追踪快捷键不受影响。返回错误信息（成功返回 ""）。"""
        panel = next(_iter_neurofly_panels(), None)
        if panel is None or getattr(panel, "G", None) is None:
            return "先在 neurofly Annotator 载入同一个 db 再跳"
        try:
            panel.selected_node.value = str(int(nid))
            if getattr(panel.mode_switch, "mode", "panorama") == "panorama":
                panel.switch_mode(panel.viewer)
            else:
                # keep_image=False 必须显式给：默认 True 只刷点/边标记，
                # 不重新切块不移相机（node 跳了 image 不跟的根因）
                panel.refresh(panel.viewer, keep_image=False)
        except Exception as exc:  # noqa: BLE001
            return f"跳转失败: {type(exc).__name__}"
        return ""

    def _smart_d(viewer_):
        """viewer 级 'd' 重绑为 drive：neurofly 原绑定 refresh(keep_image=True)
        只刷任务标记不移动视图；active 层是 point_layer 时 napari 优先走它自己的
        predict_displacement，到不了这里——能走到说明 active 是别的层（image/
        atlas/rings），此时 drive 一次（重新切块居中）。"""
        panel = next(_iter_neurofly_panels(), None)
        if panel is None or getattr(panel, "G", None) is None:
            return
        try:
            if int(panel.selected_node.value) not in panel.G.nodes:
                return
            panel.refresh(panel.viewer, keep_image=False)
        except Exception:
            pass

    def _on_ring_click(layer, event):
        """点环 = 选中该节点（同 neurofly panorama 点选链路，候选只有区内节点）。"""
        if not _alive() or getattr(event, "button", 0) != 1:
            return
        idxs = state.get("region_nodes_idx")
        ids = state.get("node_ids")
        if idxs is None or ids is None or len(idxs) == 0:
            return
        try:
            idx = layer.click_get_value(
                event.position,
                view_direction=event.view_direction,
                dims_displayed=event.dims_displayed,
                world=True,
            )
        except Exception:
            idx = None
        if idx is None or idx < 0 or idx >= len(idxs):
            return
        nid = int(ids[int(idxs[int(idx)])])
        err = _trace_node(nid)
        status.setText(f"已定位 nid={nid}" + (f"（{err}）" if err else ""))

    def _begin_region_iter(display: list[int]):
        """当前脑区的段遍历状态：段间按最小 nid 升序（确定性遍历）。"""
        nodes_l1 = state.get("nodes_l1")
        ids = state.get("node_ids")
        seg = state.get("node_seg")
        if nodes_l1 is None or ids is None or seg is None or len(nodes_l1) == 0:
            state["region_iter"] = None
            return
        mask = nodes_in_region_mask(nodes_l1, state["scan_arr"], display, 1, state["centroid_level"])
        idxs = np.nonzero(mask)[0]
        if len(idxs) == 0:
            state["region_iter"] = None
            return
        seg_nodes: dict[int, list[int]] = {}
        for i in idxs:
            seg_nodes.setdefault(int(seg[i]), []).append(int(ids[i]))
        seg_ids = sorted(seg_nodes, key=lambda s: min(seg_nodes[s]))
        state["region_iter"] = {
            "seg_ids": seg_ids,
            "cursor": 0,
            "seg_nodes": {s: sorted(v) for s, v in seg_nodes.items()},
        }

    def _next_neuron():
        """「下一神经元」：取下一个段的区内起始节点，程序化选点+进 labeling。"""
        it = state.get("region_iter")
        panel = next(_iter_neurofly_panels(), None)
        if panel is None or getattr(panel, "G", None) is None:
            status.setText("先在 neurofly Annotator 载入同一个 db，再逐段追")
            return
        if not it or not it.get("seg_ids"):
            status.setText("先在列表里点选一个脑区，再逐段追")
            return
        seg_ids = it["seg_ids"]
        k = it["cursor"] % len(seg_ids)
        seg = seg_ids[k]
        nids = it["seg_nodes"][seg]
        it["cursor"] = k + 1
        nid = nids[0]
        err = _trace_node(nid)
        if err:
            status.setText(f"神经元段 {k + 1}/{len(seg_ids)}：{err}")
        else:
            status.setText(
                f"神经元段 {k + 1}/{len(seg_ids)}：区内 {len(nids)} 节点，起始 nid={nid}；"
                "再点「下一神经元」跳下一段"
            )

    def _set_panorama_dim(dim: bool):
        """选区时压暗 panorama 点云（826k 点太杂乱），「显示全部脑区」恢复。"""
        panel = next(_iter_neurofly_panels(), None)
        pp = getattr(panel, "panorama_points", None) if panel is not None else None
        if pp is None:
            return
        try:
            if dim:
                if "pano_opacity" not in state:
                    state["pano_opacity"] = float(getattr(pp, "opacity", 1.0))
                pp.opacity = 0.15
            elif "pano_opacity" in state:
                pp.opacity = state.pop("pano_opacity")
        except Exception:
            pass

    btn_next_seg = QtWidgets.QPushButton("下一神经元")
    btn_show_all = QtWidgets.QPushButton("显示全部脑区")
    row_iter = QtWidgets.QHBoxLayout()
    row_iter.addWidget(btn_next_seg)
    row_iter.addWidget(btn_show_all)
    layout.insertLayout(layout.indexOf(list_results) + 1, row_iter)
    btn_next_seg.clicked.connect(lambda: _next_neuron() if _alive() else None)
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
        n_in = _update_region_nodes_layer(display)
        _begin_region_iter(display)
        _set_panorama_dim(True)
        extra = "" if len(display) == 1 else f"（本体无体素，显示 {len(display)} 个子区）"
        if n_in:
            n_seg = state.get("region_iter") or {}
            n_txt = f"，高亮 {n_in} 节点 / {len(n_seg.get('seg_ids', []))} 段"
            status_txt = f"跳转 → {row.text}{n_txt}；「下一神经元」逐段追，「显示全部脑区」恢复"
        else:
            status_txt = f"跳转 → {row.text}{extra}"
        if not state.get("isolated") and state.get("isolate_error"):
            status_txt += f"（隔离失败: {state['isolate_error']}）"
        status.setText(status_txt)

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

    def _sweep():
        if not _alive():
            return
        _snapshot_neurofly_combos()
        _restore_neurofly_combos()
        for widget, idx, _spacing, _spacing0 in _neurofly_widgets(None):
            try:
                rewire_neurofly_double_click(widget)
                wrap_neurofly_put_point(widget)
                # 完整纠正：image/点/边/goal 层 µm scale + panorama 粗层 +
                # panorama 点层（hr 口径）——与类补丁的 _fix_layer_scales 同一份逻辑
                _fix_layer_scales(widget)
                if not state.get("d_bound"):
                    # 面板发现后重绑 viewer 'd' 为 drive（面板构造时会绑回
                    # refresh(keep_image=True)，须在其后覆盖）
                    viewer.bind_key("d", _smart_d, overwrite=True)
                    state["d_bound"] = True
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

    if db_path:
        # 单开一拍事件循环再解析（82 万节点 ~10s），别把窗口首帧卡在创建里
        QtCore.QTimer.singleShot(0, lambda: _load_nodes_db(str(db_path)) if _alive() else None)

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


# id(image_layer) -> 上次写入的 µm translate（幂等判据，见 _fix_layer_scales）
_LAST_IMG_TRANSLATE: dict[int, tuple] = {}


def _fix_layer_scales(panel) -> None:
    """把面板的 image 层与 panorama 层 scale 设为真实 spacing（µm/体素）。

    面板带 `_image_origin`（原生 µm 的 neurofly fork 标记，refresh 自己维护
    世界坐标）时直接返回——补偿是给体素口径的官方 0.1.4 用的，双重生效会把
    µm translate 再乘一遍 scale。"""
    if getattr(panel, "_image_origin", None) is not None:
        return
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
        # napari world = data*scale + translate，translate 原样当世界坐标用
        #（offscreen 实证 2026-11-04）。neurofly 塞进 image.translate 的是块
        # 原点的层级体素数（refresh: c_coord - h_size），乘 spacing 才是块的真实
        # µm 位置——不乘则整块在世界里错位 it*(scale-1)：labeling 模式 it≠0 时
        # atlas label/panorama 相对块漂移（q 进 labeling 首次暴露），panorama
        # 模式 it=0 掩盖。db 点/边/goal 坐标是绝对体素，translate 0 + µm scale。
        it = tuple(float(v) for v in image_layer.translate)
        # 区分 neurofly 刚写的原始体素 translate 和我们已修正的 µm translate：
        # 补丁每次 refresh 都跑，直接 it*scale 会二次相乘（非幂等）。记录上次
        # 写入值；与当前一致 = 已是 µm 口径，跳过。
        last = _LAST_IMG_TRANSLATE.get(id(image_layer))
        if last is not None and np.allclose(np.asarray(it, dtype=float), np.asarray(last, dtype=float), rtol=1e-6):
            img_translate = it
            recenter = False
            # 已是 µm 口径：体素原值 = µm / scale（给 put_point 等体素口径消费者）
            voxel_translate = tuple(t / s for t, s in zip(it, new_scale))
        else:
            img_translate = tuple(t * s for t, s in zip(it, new_scale))
            recenter = True  # 块被搬动：相机也得跟着换世界
            voxel_translate = it
            _LAST_IMG_TRANSLATE[id(image_layer)] = img_translate
        # neurofly put_point / SimpleViewer 双击把 image_layer.translate 当体素
        # 偏移读（segs_annotator.py:1214 / simple_viewer.py:267）——µm 世界下由
        # wrap_neurofly_put_point / 双击守卫临时换回体素口径，原始值存这里
        try:
            image_layer._atlas_voxel_translate = tuple(float(v) for v in voxel_translate)
        except Exception:
            pass
        if not np.allclose(
                np.asarray(image_layer.translate, dtype=float), np.asarray(img_translate, dtype=float), rtol=1e-6):
            image_layer.translate = img_translate
        for attr in ("point_layer", "edge_layer", "ex_edge_layer", "goal_layer"):
            layer = getattr(panel, attr, None)
            if layer is None:
                continue
            if not np.allclose(
                    np.asarray(layer.scale, dtype=float), np.asarray(new_scale, dtype=float), rtol=0.01):
                layer.scale = new_scale
            if not np.allclose(
                    np.asarray(layer.translate, dtype=float), np.zeros(3), rtol=1e-6):
                layer.translate = (0.0, 0.0, 0.0)
        if recenter:
            # neurofly 设 camera.center = c_coord（体素数当世界坐标）；块搬进真实
            # µm 世界后相机必须同步，否则 q 进 labeling 视野里没有块。新目标 =
            # 块世界中心 = translate + shape*scale/2（= c_coord×spacing）。只在块
            # 真被搬动时拽相机，不干扰用户手动导航。
            viewer = getattr(panel, "viewer", None)
            if viewer is not None:
                try:
                    shape = np.asarray(getattr(image_layer.data, "shape"), dtype=float)
                    center = np.asarray(img_translate, dtype=float) + shape * np.asarray(new_scale) / 2.0
                    viewer.camera.center = tuple(float(v) for v in center)
                except Exception:
                    pass

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
    影响不到它）。原生 µm fork 的 on_double_click 自带守卫，直接跳过。"""
    if getattr(sv_widget, "_image_origin", None) is not None:
        return False  # 原生 µm fork：on_double_click 自带 None 守卫
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
            # SimpleViewer.on_double_click 同样把 translate 当体素偏移读
            #（simple_viewer.py:267），µm 世界下临时换回体素口径再调原逻辑
            raw = getattr(layer_, "_atlas_voxel_translate", None)
            prev = None
            if raw is not None:
                prev = tuple(float(v) for v in layer_.translate)
                layer_.translate = tuple(float(v) for v in raw)
            try:
                return _orig(layer_, event)
            finally:
                if prev is not None:
                    layer_.translate = prev

        guarded._atlas_guard = True  # type: ignore[attr-defined]
        callbacks[index] = guarded
        rewired = True
    return rewired


def _ray_bbox_intersections(pos_d, dir_d, bbox):
    """点击射线与数据包围盒求交（slab 法）。

    napari `Layer.get_ray_intersections` 在点击位置投影**落在盒内**时只找得到
    背面（find_front_back_face 要求前后两面）→ 返回 (None, None)，neurofly 的
    put_point/双击直接崩。这里对盒内起点同样成立：near 钳到点击点本身。
    bbox 为 (3,2) 各轴 [min, max]（上限按 napari 口径已 +1）。"""
    pos_d = np.asarray(pos_d, dtype=float)
    dir_d = np.asarray(dir_d, dtype=float)
    t0, t1 = -np.inf, np.inf
    for ax in range(3):
        lo, hi = float(bbox[ax][0]), float(bbox[ax][1])
        d, p = dir_d[ax], pos_d[ax]
        if abs(d) < 1e-12:
            if p < lo or p > hi:
                return None, None
            continue
        ta, tb = (lo - p) / d, (hi - p) / d
        if ta > tb:
            ta, tb = tb, ta
        t0, t1 = max(t0, ta), min(t1, tb)
        if t0 > t1:
            return None, None
    t0 = max(t0, 0.0)
    if not np.isfinite(t1) or t1 <= 0:
        return None, None
    near = pos_d + t0 * dir_d
    far = pos_d + t1 * dir_d
    return near, far


def wrap_neurofly_put_point(panel) -> bool:
    """µm 世界下 neurofly put_point（image 层右键=手动加点）的口径包装。

    原生 µm 的 neurofly fork（面板带 `_image_origin`，put_point 自己用体素
    原点、ray 自守卫）不需要本包装——sweep 会在发现标记时直接跳过。

    对体素口径的官方 0.1.4（无标记），包装做三件事：①调用前在 **µm 世界**把
    点击射线换到数据坐标并与块 bbox 求交（slab 法）——必须先算，因为第②步会
    把 translate 换回体素口径、破坏 world↔data 映射，put_point 内部的原生 ray
    求交在坏映射下全判到盒外（2026-11-05 offscreen 复现）；②临时把 translate
    换回体素口径（_atlas_voxel_translate 由 _fix_layer_scales 维护，put_point
    把 translate 当体素偏移加回坐标）并把算好的 near/far 顶替进
    get_ray_intersections——napari 原生实现还要求点击位置投影落在盒**外**（
    find_front_back_face 找前后两面），3D 里相机焦点在块内时点击全落盒内 →
    原生返回 None → put_point 崩；③射线真没打到块时提示后不调用 put_point。"""
    layer = getattr(panel, "image_layer", None)
    if layer is None or not getattr(layer, "mouse_drag_callbacks", None):
        return False
    if getattr(panel, "_image_origin", None) is not None:
        return False  # 原生 µm fork：put_point 自洽，不包装
    callbacks = layer.mouse_drag_callbacks
    for index, cb in enumerate(list(callbacks)):
        if getattr(cb, "_atlas_um_guard", False):
            return False  # 已包装
        func = getattr(cb, "__func__", None)
        qual = getattr(func, "__qualname__", "") if func is not None else ""
        if not qual.endswith("Annotator.put_point"):
            continue
        original = cb

        def wrapped(lay, event, _orig=original):
            if getattr(panel, "_image_origin", None) is not None:
                return _orig(lay, event)  # 原生 µm fork：put_point 自洽，直通
            if event is not None and (
                getattr(event, "button", 0) != 2 or panel.proofreading_switch.value
            ):
                return _orig(lay, event)
            raw = getattr(lay, "_atlas_voxel_translate", None)
            if raw is None:
                return _orig(lay, event)
            # ① µm 映射还完好时把点击射线换到数据坐标并求交（无 event 跳过）
            near = far = None
            if event is not None:
                try:
                    dims = list(event.dims_displayed)
                    pos_d = np.asarray(
                        lay._world_to_displayed_data(np.asarray(event.position, dtype=float), dims),
                        dtype=float,
                    )
                    dir_d = np.asarray(
                        lay._world_to_displayed_data_ray(np.asarray(event.view_direction, dtype=float), dims),
                        dtype=float,
                    )
                    bbox = np.asarray(lay._extent_data)[:, dims].T
                    bbox[:, 1] += 1  # 与 napari get_ray_intersections 同口径
                    near, far = _ray_bbox_intersections(pos_d, dir_d, bbox)
                except Exception:
                    near, far = None, None
                if near is None or far is None:
                    try:
                        from napari.utils.notifications import show_info

                        show_info("右键加点：射线没有打到图像块上")
                    except Exception:
                        pass
                    return None
            # ② 体素口径临时换回；有 event 时 near/far 顶替（put_point 内部重算
            #    会拿到同样值），无 event 的异构调用只做口径交换
            prev = tuple(float(v) for v in lay.translate)

            def fixed_ray(position, view_direction, dims_displayed, world=True):
                return near, far

            lay.get_ray_intersections = fixed_ray
            lay.translate = tuple(float(v) for v in raw)
            try:
                return _orig(lay, event)
            finally:
                vars(lay).pop("get_ray_intersections", None)
                lay.translate = prev

        wrapped._atlas_um_guard = True  # type: ignore[attr-defined]
        callbacks[index] = wrapped
        return True
    return False


def _load_paths_json() -> dict:
    """路径记忆（上次成功载入的 db 等）；纯便利缓存，坏/缺静默。"""
    try:
        return json.loads(PATHS_JSON.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_paths_json(update: dict) -> None:
    try:
        data = _load_paths_json()
        data.update(update)
        PATHS_JSON.parent.mkdir(parents=True, exist_ok=True)
        PATHS_JSON.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    except Exception:
        pass  # 记不住路径不影响功能


def _print_console_snippet(image: str | None, db: str | None) -> None:
    """打印 napari console 里开 neurofly 面板的粘贴片段。

    CLI 不再代建面板（启动期半初始化状态下赋值路径不可靠且 dock 落左侧）；
    面板由用户 console 里这段创建。顺序先 db 后图像（refresh_panorama 建 G
    时两路径都读，db 先就位则图像加载后自动补全景）。"""
    db = db or _load_paths_json().get("db") or "<db 路径>"
    image = image or "<ims 路径>"
    print(
        "---- napari console 里粘贴这段（图像加载要等几秒，NAS 更久）----\n"
        "import napari\n"
        "from neurofly.segs_annotator import Annotator\n"
        "viewer = napari.current_viewer()\n"
        "panel = Annotator(viewer)\n"
        'viewer.window.add_dock_widget(panel, area="right", name="Segs Annotator")\n'
        f'panel.db_path.value = r"{db}"\n'
        f'panel.image_path.value = r"{image}"\n'
        "# 图像出来后切面板层级；推理 db 是 L1 体素坐标，面板层级须同级：\n"
        'panel.level.value = next(c for c in panel.level.choices if c.endswith(" 1"))\n'
        "------------------------------------------------------------------",
        flush=True,
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sample-dir", help="样本目录：自动加载 upsampled_atlas_label.zarr")
    parser.add_argument("--label-zarr", help="atlas label zarr 路径（默认 <sample-dir>/upsampled_atlas_label.zarr）")
    parser.add_argument("--region-csv", default=None, help="区域 CSV；缺省按 label id 体系自动选 Rev2/Rev1")
    parser.add_argument("--db", help="分割 db：overlay 直接载入节点，并打进 console 片段")
    parser.add_argument("--image", help="ims 路径：打进 console 片段（neurofly 面板由 console 片段创建）")
    args = parser.parse_args(argv)

    sample_dir = Path(args.sample_dir) if args.sample_dir else None
    label_zarr = Path(args.label_zarr) if args.label_zarr else _find_label_zarr(sample_dir)

    import napari

    patch_neurofly_display_scale()
    patched = patch_neurofly_double_click()
    viewer = napari.current_viewer() or napari.Viewer()

    widget = make_atlas_overlay_widget(
        viewer, label_zarr=label_zarr, region_csv=args.region_csv, auto_load=True,
        db_path=args.db,
    )
    viewer.window.add_dock_widget(widget, area="right", name="图谱叠加")
    if patched:
        print("neurofly 双击兼容补丁已应用")
    _print_console_snippet(image=args.image, db=args.db)
    napari.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
