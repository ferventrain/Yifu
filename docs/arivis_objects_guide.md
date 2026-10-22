# Arivis .objects 文件读写指南

`*.objects` 是 Arivis 的对象标注数据库（SQLite 格式），与同名 `.ims` / `.sis` 图像文件
配套使用。本文记录其内部结构、如何用 Python 读写，以及注意事项。

## 文件位置与命名

与 ims 文件同目录，命名通常为 `<图像名>.objects` 或 `<容器名>.objects`。
例如 `YF2026073001_AYD_nao.ims` 旁边的 `C1 (2).objects`。
打开 ims/sis 时 Arivis 会自动加载目录下所有 `.objects` 文件里的对象。

### 图像文件家族（同一基本名）

| 文件 | 内容 |
|---|---|
| `.sis` | 像素数据（私有分块格式；文本头 `# Single Image Stack` + `Header = N` 字节 XML 头） |
| `.metadata` | SQLite：tile 索引表、auditTrail（含图像尺寸/层数，见下文） |
| `.objects` | 对象标注（本文主题） |
| `.cache` / `.3dcache` / `.3dcache-lock` | 渲染缓存与锁；残留锁文件不影响换库 |

## 内部结构（SQLite）

用任意 SQLite 工具（或 Python `sqlite3`）打开，核心表：

| 表 | 作用 |
|---|---|
| `Objects` | 对象列表：Id、Name、包围盒（Bounds_X/Y/Z 1..2）、IsDeleted |
| `Geometries` | 逐 z 平面的几何数据：Object_Id、Bounds、Data（行程编码 BLOB） |
| `Objects2Tags` | 对象 ↔ 标签关联（Tag_Id → Tags 表） |
| `Tags` | 标签定义：如 手动 / 梗死 / C1 / C3 / 心脏 |
| `Features` / `FeatureValues` | 对象的测量特征（体积、强度等） |
| `Parent2Children` | 对象层级 |

### Geometries.Data 的行程编码格式

每个 z 平面对应一条记录，Data 是 `<i4` 小端整数数组：

```
[版本=1, X1, X2, Y1, Y2,               ← 5 个整数的平面头部（含边界框）
 k₀, x_start, x_end, x_start, x_end, ...,  ← 第 1 行 y=Y1：k 个区间
 k₁, x_start, x_end, ...,                  ← 第 2 行 y=Y1+1
 ...]
```

- 行数 = Y2 − Y1 + 1；区间为闭区间（含端点）
- 每个 k 后跟恰好 k 对整数
- 用 `np.frombuffer(blob, '<i4')` 解析

### 读取示例

```python
import json
import sqlite3

import numpy as np

DB = r"S:/path/to/C1.objects"
TAG_MAP = {"心脏": 16, "梗死": 2, "C1": 14, "C3": 10}  # 按 Tags 表实际值调整


def decode_plane(blob, x1, x2, y1, y2):
    """解析一个 z 平面 → bool mask (y2-y1+1, x2-x1+1)。"""
    i4 = np.frombuffer(blob, "<i4")
    assert i4[0] == 1
    body = i4[5:]
    h, w = y2 - y1 + 1, x2 - x1 + 1
    m = np.zeros((h, w), np.uint8)
    pos = 0
    for yy in range(h):
        k = int(body[pos]); pos += 1
        for _ in range(k):
            xa, xb = int(body[pos]), int(body[pos + 1]); pos += 2
            m[yy, xa - x1:xb - x1 + 1] = 1
    return m


def extract_tag(tag_id, db_path=DB):
    """提取一个标签下所有对象的全部平面 → {z: mask}。"""
    con = sqlite3.connect(db_path)
    out = {}
    for z, x1, x2, y1, y2, blob in con.execute(
        "SELECT g.Bounds_Z1, g.Bounds_X1, g.Bounds_X2, g.Bounds_Y1, g.Bounds_Y2, g.Data "
        "FROM Geometries g JOIN Objects2Tags t ON g.Object_Id = t.Object_Id "
        "WHERE t.Tag_Id = ? ORDER BY g.Bounds_Z1", (tag_id,)):
        out[int(z)] = decode_plane(blob, int(x1), int(x2), int(y1), int(y2))
    return out
```

完整可运行版本见 `_frangi_trials2/extract_arivis_masks.py`
（标签 → 逐 z 平面 mask → zarr，含断点续传）。

### 写入示例

按相同格式生成行程编码 BLOB 后，以新 Object_Id 写入 Geometries，
并在 Objects2Tags 里关联标签即可被 Arivis 识别。

注意：直接写入 .objects 前请确认 Arivis 当前**没有打开该工程**——不是看 tasklist 里有没有
arivis 进程，而是确认它打开的文档不是这个文件（最可靠：用下文脚本 API 查
`get_document().get_name()`）。实测 arivis 开着但打开的是别的文档时，本工程的
.objects 可以安全整体替换；残留的 `.3dcache-lock` 不碍事。

## 官方脚本 API（arivis Pro 自带 Python + gRPC）——读像素的正道

不要逆向 `.sis`，直接用 arivis Pro（4.4，装在 `S:/arivis Pro`）的脚本 API（2026-10 实测）。

1. 先启动并打开文档：`"S:/arivis Pro/arivisPro.exe" "<sis 路径>"`。
   1.7TB 的 sis 约 30 秒后才能接受脚本连接，脚本里要对连接做重试。
   注意 `arivisPro.exe --help` 没有命令行帮助——直接拉起 GUI，别在 shell 里探测它。
2. 用自带解释器跑脚本：`"S:/arivis Pro/Python3/python.exe"`，
   先 `sys.path.insert(0, r"S:/arivis Pro/Python3/Lib/site-packages")` 再 `import arivis`。
   连接经 Win32 消息自动发现，无需配置环境变量；arivis 没在跑时只报一句
   "Cannot connect to arivis Pro"，没有细节。
3. 自带环境：Python 3.8 + numpy 1.19.5 + pywin32，**没有 tifffile / scipy**；
   大数据用 npy 落盘中转，再用别的 env 转格式。

```python
import sys, time
sys.path.insert(0, r"S:/arivis Pro/Python3/Lib/site-packages")
import arivis

for _ in range(30):
    try:
        v = arivis.App.get_viewer_list()[0]; break
    except Exception:
        time.sleep(5)
doc = v.get_document()                # 当前打开的文档
iset = doc.get_default_imageset()
arr = iset.read_imagedata(((x, y, z), (sx, sy, sz)), channel, timepoint)
# 3D 读返回 (z, y, x)；64 层 × 2000×2300 的 slab 无压力，实测 ~100–135 MB/s
```

已踩过的 API 坑：

- `App.get_document(path)` 只能取**已打开**的文档，不会开新文件（文档没开时返回
  None，下一步就 AttributeError）。开新文档用 `viewer.open_document(path)`
  （会切换该 viewer 的当前文档，不开新窗口）。
- `get_bounding_rectangle()` 已废弃，且返回的 Bounds2D 没有 x1/y1 属性（直接
  AttributeError）；用 `get_bounding_box()`（Bounds4D，各轴闭区间）。
- `read_imagedata` 的 region 元组形式：start 必须 3D `(x, y, z)`，size 可 2D 或 3D
  （2D 视为 sz=1）；buffer=None 时直接返回 numpy 数组。

## .sis 私有格式逆向（负结果，勿再试）

- `.metadata` 的 `versioned_tile_descriptors` 表给出每个 chunk 的 `position / size`
  （chunk 固定 612,360 B，compression=0 未压缩），但 chunk 在文件里**乱序散布**；
  `.sis` 头部 XML 里的 `<Tiles width="630" height="648" tileNumX="12" tileNumY="16"/>`
  是逻辑缓存分块，**不是**物理存储布局。
- 2026-10 用「掩膜内外对比度 + 行相关周期 + 接缝梯度」穷举了全部重组假设
  （整幅光栅 / tile 网格 / 列条带 / 转置 / 蛇形 / 翻转），与 API 真值比对全部失败
  （同一 patch 解码均值 332 vs 真值 41.5）。结论：读像素走脚本 API。
- 不开 arivis 也能拿图像尺寸：`.metadata` 的 `auditTrail` 表 JSON 里有
  `"created image set '0': 16 bit, 3 channel(s), 1 frame(s), 3680 plane(s), 7552x10368 pixels"`。
- 层数陷阱：头报 plane 数（3680/4128）多于 tile 表里实际有数据的 plane 数
  （3655/4114），尾部若干层无 tile；引用 z 上限时以 tile 表为准。

## 裁剪导出配方（big bbox 外扩 → 独立图像集）

1. 从 .objects 读 big 的 Bounds（闭区间），三轴各外扩 20%（按图像尺寸夹边）。
2. 脚本 API `read_imagedata` 按 slab（64 层/次）读出 → npy → tifffile 转每通道 BigTIFF。
3. 复制 .objects 并整体平移坐标：Objects 的 6 个 Bounds 字段 + 每条 Geometry 的
   Data blob 头部 4 个 u32（X1,X2,Y1,Y2）与每对区间的 2 个 u32，全部减去
   (ox, oy, oz)；`Bounds_Z` 同步平移。
4. arivis 里把 3 个 BigTIFF **作为一个文档的 3 个通道**一起导入，文档名与 sidecar
   基本名严格一致（`<名>_crop.tif` ↔ `<名>_crop.objects`）；删掉 arivis 自生成的
   .objects，换成平移版再打开。

坑：外圈（outside）ROI 因逐层跟踪漂移可能越出 crop 边界（实测最多 181px），
统计这几个 ROI 时要么剔除要么只算落在体积内的部分。

## 注意事项

- 坐标为原生体素索引（非微米），分辨率看对应 ims 的元数据
  （本项目 = 1.8 × 1.8 × 2.0 µm，x, y, z 顺序）
- Bounds_X/Y/Z 是**闭区间**（含端点），即 X2 − X1 + 1 个体素
- 标签 Tag_Id 因文件而异，务必先读 `Tags` 表确认
- 孤儿行清理：某些文件里 `Objects.Name` 无标签的行是绘制草稿，可按需过滤
- zarr 3.2.1 的 v2 codec 不支持边缘裁剪块（读角区会报 reshape 错），
  读本项目 zarr 时请用 `_frangi_trials2/rawzarr.py`（已处理满块/裁剪块兼容）
- **软删语义**：arivis 里的一切删除都是 `IsDeleted=1`，不会物理删行；
  统计"现存对象"必须过滤 `IsDeleted=0`，否则数量和 GUI 对不上
- **GUI 改库**：用户在 arivis 里的一切操作（copy / split / Object Math / 手动挪动）
  都直接改写 .objects。任何重生成、批量替换前必须重新 inspect 当前 live 库；
  按名字前缀批量替换（如 `Auto %`）会误伤用户手工副本（"Auto inside #001 - copy"
  也命中前缀），应按显式 Id 列表替换，或替换后把用户对象恢复（IsDeleted 置 0）
- 换库流程建议：备份 live → 写副本 → 副本校验 → 整体替换；旧对象软删保留可回滚
