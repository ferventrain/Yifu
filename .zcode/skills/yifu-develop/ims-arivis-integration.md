# Imaris / arivis 对接指南（草稿，持续由用户维护）

面向需要读写 `.ims` / `.sis` / `.objects` 或与 Imaris、arivis Pro 共存的开发任务。
这里是格式与对接的已知事实，来源：逆向 + 实战验证（2026-09 ~ 2026-10）。

---

## 1. Imaris `.ims` 格式（HDF5）

**层级**（h5py 只读路径，read-only 零依赖 Imaris）：

```
/DataSet/ResolutionLevel {0..N}/TimePoint 0/Channel {0..M}/Data   # 体数据
/DataSetInfo/Image         # ExtMin0..2 / ExtMax0..2（µm, 轴序 X,Y,Z）
/DataSetInfo/Channel {i}   # 通道元数据
```

- **轴序**：Data 的 HDF5 shape 即 (z, y, x)。经 `ims_to_zarr` 落盘的 zarr 保持同序。
- **金字塔**：ResolutionLevel 0 最细；每层 shape 是**独立 padding** 的（凑 tile 倍数，
  L0 真实 7451×10281×4356 → h5 7552×10368×4384，L7 只有 64×128×64）。
  **不能用 h5 shape 推层级/spacing**，要用各层 `ImageSizeX/Y/Z` 属性（真实尺寸）。
- **ExtMin/Max 是逐字符 `|S1` 数组**（"13.0888888888" 存成 13 个字节元素），
  读取要字节拼接再 float（`napari_atlas_overlay._decode_ims_attr` 有现成实现）。
- **spacing 口径**：neurofly `image.info[level]['spacing']` = µm/体素（该层），
  各层各轴**不严格 2 倍**，跨层换算用 `spacing(L0)/spacing(当前层)` 比值。
- L2 常有零填充超出 extent/4 的区域（MegaSpim/SmartSpim 导出习惯）。

## 2. arivis `.sis`（场景文件）与 `.objects`（对象数据库）

- `.sis`：文本头 `#Single Image Stack\nHeader = <字节数>` + XML；`<Scope>` = imageset，
  每 plane 带 `<Rect x y w h>` = crop 在原图的偏移；**显示名看 visibleName**，
  内部 name 保留删除重建的旧编号（解析必须用 visibleName）。
- `.objects`：SQLite（Objects / Geometries / Tags 表）。几何格式：每 z 平面一条
  Geometries.Data：header `[1,X1,X2,Y1,Y2]` + 每 y 行 `[区间数k,(x1,x2)×k]`（x 闭区间）。
  完整解析与统计实战见 repo `docs/arivis_objects_guide.md`。
- arivis 自动化正道：**arivis Pro 自带 Python3 + gRPC（read_imagedata）**，
  `.sis` 逆向仅作只读解析，别再投入。

## 3. 对接场景与已知坑

| 场景 | 要点 |
|---|---|
| python 读 ims | h5py 直读（本仓库惯例）；注意懒加载，别 np.asarray 全量 |
| napari 显示 ims | neurofly 的 SimpleViewer/Segs Annotator 就是现成 ims reader |
| Fiji 开 NAS ims | 边车 BasePath 吞 UNC 斜杠（`file:/192...`→`S:\192...`）；改 `file:////host/` 或映射盘符；bdv.img.imaris 经 jhdf5 独立读 ims |
| Imaris 互相锁 | arivis/Imaris 锁定 .sis/.ims 时脚本要用后备路径 |
| 坐标互通 | arivis 对象坐标=原图体素坐标，按 Rect 包围盒归属 crop |

## 4. 待补充（用户维护区）

- Imaris 端操作惯例（人工环节：谁、在哪个软件、做什么检查）
- ims **写**回（Imaris 可打开）的清单——参见 TIFF→ims 转换器经验（root attrs、
  no-NUL `|S1`、ExtMin/Max、深金字塔、RGBA 缩略图），如需展开另起小节
