# Brainrender 点信号可视化

在 Allen 标准脑（`allen_mouse_25um`）中，把 atlas 空间的 punctate 荧光信号显示为透明脑内的彩色点，并支持：

- 自定义脑区分组 mesh + 组内点着色
- 交互式保存固定视角
- 按脑区组批量出图（同一视角）

脚本入口：`pipeline_modules/visualization/render_points_brainrender.py`

---

## 环境

| 用途 | 环境 |
|------|------|
| brainrender 渲染 | `napari` |
| 点 CSV 自动生成（warp / 体积转点） | `yifu` |

BrainGlobe atlas 需已下载（`napari` 环境里检查）：

```powershell
micromamba run -n napari python -c "from brainglobe_atlasapi.list_atlases import get_downloaded_atlases; print(get_downloaded_atlases())"
```

应包含 `allen_mouse_25um`。

在仓库根目录运行 brainrender 时，Windows 需设置 `PYTHONPATH`：

```powershell
cd S:\Yifu; $env:PYTHONPATH='S:\Yifu'
```

下文命令均假设已在 `S:\Yifu` 且已设置 `PYTHONPATH`（如未写则请自行加上）。

---

## 推荐用法：只传 `--sample_dir`

每个样本的所有 brainrender 产物默认落在该样本的 `visualization/` 下：

| 文件 | 默认路径 |
|------|----------|
| 点坐标 CSV | `{sample_dir}/visualization/points.csv` |
| 点坐标元数据 | `{sample_dir}/visualization/points.json` |
| 固定视角 JSON | `{sample_dir}/visualization/{sample_name}_brainrender_view.json` |
| 按组截图 PNG | `{sample_dir}/visualization/brainrender/{GROUP}_brainrender.png` |

示例样本：

```text
S:\Arivis_Analysis\YF2026030302_wangzhinao\nao_1
```

对应文件：

```text
...\nao_1\visualization\points.csv
...\nao_1\visualization\nao_1_brainrender_view.json
...\nao_1\visualization\brainrender\PFC_brainrender.png
```

换脑子时 **只改 `--sample_dir`** 即可。

---

## 样本前置条件

brainrender 需要 **atlas 空间** 的点坐标。pipeline 跑完后，样本目录通常已有：

```text
nao_1/
  ch0/                          # 配准通道 TIFF
  ch0_downsample/volume.nii.gz  # 配准用 NIfTI
  ch1/                          # 信号通道 TIFF
  ch1.zarr                      # 信号 Zarr
  ch1_mask.zarr                 # 分割 mask Zarr
  upsampled_atlas_label.zarr      # atlas label（样本空间）
  visualization/
    ch1_mask_atlas_volume.tiff  # mask 已 warp 到 atlas（常用输入）
    ch1_mask_atlas_volume.json
    points.csv                  # brainrender 输入（可自动生成）
```

### 自动生成 `points.csv`

传 `--sample_dir` 且 `visualization/points.csv` 不存在时，按顺序尝试：

1. **`{signal_ch}_mask_atlas_volume.tiff`** + 同名 `.json`（默认 `signal_ch=ch1`）
2. **`{sample_name}_heatmap3d_volume.tiff`**
3. 调用 `warp_mask_zarr_to_atlas_points` 从 mask zarr 完整 warp

已有 `points.csv` 时直接复用；强制重算加 `--force_warp`。

手动指定 CSV（覆盖默认路径）：

```powershell
micromamba run -n napari python pipeline_modules/visualization/render_points_brainrender.py --sample_dir "S:\path\to\nao_1" --points_csv "S:\path\to\custom_points.csv" ...
```

---

## 脑区分组配置

默认分组：`config/region_groups.json`

| 组名 | Allen 结构 | 说明 |
|------|-----------|------|
| `PFC` | FRP, ACA, PL, ILA, ORB, DP | 前额叶（含 ACA，与 ACC 组故意重叠） |
| `dlPFC` | PL, ACAd, ORBl | 背外侧 PFC 同源近似 |
| `vmPFC` | ILA, ACAv, ORBm, FRP | 腹内侧 PFC 同源近似 |
| `HIP` | HIP | 海马（含 CA/DG 等子区） |
| `AMY` | BLA, BMA, LA, PA, CEA, MEA, AAA, IA, COA, PAA | 杏仁核（含 COA/PAA） |
| `ACC` | ACA | 前扣回（与 PFC 重叠 intentional） |
| `NAc` | ACB | 伏隔核 |
| `LHb` | LH | 外侧缰核 |

JSON 支持两种格式：

**完整格式（推荐）：**

```json
{
  "PFC": {
    "acronyms": ["FRP", "ACA", "PL", "ILA", "ORB", "DP"],
    "color": "#4cc9f0",
    "description": "..."
  }
}
```

**简写格式：**

```json
{"PFC": ["FRP", "ACA", "PL", "ILA", "ORB", "DP"]}
```

自定义分组文件：

```powershell
--region_groups "S:\Yifu\config\my_groups.json"
```

---

## 完整工作流（推荐顺序）

以 `nao_1` 为例。

### Step 1 — 交互预览 + 按组着色

```powershell
cd S:\Yifu; $env:PYTHONPATH='S:\Yifu'; micromamba run -n napari python pipeline_modules/visualization/render_points_brainrender.py --sample_dir "S:\Arivis_Analysis\YF2026030302_wangzhinao\nao_1" --group_names "PFC,HIP,AMY,ACC,NAc,LHb" --color_points_by_group --filter_points_by_group
```

- 显示透明全脑 + 所选组的 mesh + 组内点（按组颜色）
- 不加 `--color_points_by_group` 时所有点为默认红色（`#ff4d6d`）

**PowerShell 注意：** `--group_names` 必须加引号，否则逗号会拆成多个参数。

### Step 2 — 保存固定视角（不是截图）

```powershell
cd S:\Yifu; $env:PYTHONPATH='S:\Yifu'; micromamba run -n napari python pipeline_modules/visualization/render_points_brainrender.py --sample_dir "S:\Arivis_Analysis\YF2026030302_wangzhinao\nao_1" --group_names "PFC" --color_points_by_group --export_camera_view
```

交互窗口快捷键：

| 按键 | 作用 |
|------|------|
| **V** | 保存视角到 `{sample}/visualization/{sample}_brainrender_view.json` |
| **Shift+C** | 在终端打印相机参数（备用） |
| **S** | 当前窗口截图（brainrender 默认行为） |
| **Q / Esc** | 关闭窗口 |

按 **V** 后终端应出现 `Saved camera view to: ...`。此步骤 **不** 生成按组 PNG。

也可显式指定路径：

```powershell
--export_camera_view "S:\path\to\custom_view.json"
```

### Step 3 — 同视角按组批量截图

```powershell
cd S:\Yifu; $env:PYTHONPATH='S:\Yifu'; micromamba run -n napari python pipeline_modules/visualization/render_points_brainrender.py --sample_dir "S:\Arivis_Analysis\YF2026030302_wangzhinao\nao_1" --group_names "PFC,HIP,AMY,ACC,NAc,LHb" --screenshot_per_group --camera_view
```

- `--camera_view` **不写路径** → 自动读取 Step 2 保存的 JSON
- 输出：`visualization/brainrender/PFC_brainrender.png` 等
- 某组无点则跳过
- 第一组 mesh 加载可能较慢（1–3 分钟正常）

尚未保存视角时，可先用预设：

```powershell
cd S:\Yifu; $env:PYTHONPATH='S:\Yifu'; micromamba run -n napari python pipeline_modules/visualization/render_points_brainrender.py --sample_dir "S:\Arivis_Analysis\YF2026030302_wangzhinao\nao_1" --group_names "PFC,HIP,AMY" --screenshot_per_group --camera three_quarters
```

---

## 其他常用模式

### 单张 PNG（非按组）

```powershell
cd S:\Yifu; $env:PYTHONPATH='S:\Yifu'; micromamba run -n napari python pipeline_modules/visualization/render_points_brainrender.py --sample_dir "S:\Arivis_Analysis\YF2026030302_wangzhinao\nao_1" --group_names "PFC,HIP" --color_points_by_group --output brainrender_preview.png --camera_view
```

`--output` 相对路径会写到 `{sample_dir}/visualization/`。

### 只显示单个 Allen 区域 mesh

```powershell
cd S:\Yifu; $env:PYTHONPATH='S:\Yifu'; micromamba run -n napari python pipeline_modules/visualization/render_points_brainrender.py --sample_dir "S:\Arivis_Analysis\YF2026030302_wangzhinao\nao_1" --show_region "HIP" --only_region "HIP"
```

### 按 coarse 全脑分区着色（wb 粗分区，非 region_groups）

```powershell
cd S:\Yifu; $env:PYTHONPATH='S:\Yifu'; micromamba run -n napari python pipeline_modules/visualization/render_points_brainrender.py --sample_dir "S:\Arivis_Analysis\YF2026030302_wangzhinao\nao_1" --color_by_coarse_region
```

与 `--color_points_by_group` 互斥。

### 从外部 CSV 渲染（不用 sample_dir）

```powershell
cd S:\Yifu; $env:PYTHONPATH='S:\Yifu'; micromamba run -n napari python pipeline_modules/visualization/render_points_brainrender.py --points_csv "S:\path\to\atlas_points.csv" --columns "x,y,z" --point_radius 40 --root_alpha 0.12
```

CSV 列名支持 `x,y,z` 或 `ap,dv,ml`，也可用 `--columns` 指定。

---

## 参数一览

### 样本与输入

| 参数 | 默认 | 说明 |
|------|------|------|
| `--sample_dir` | — | 样本根目录；启用所有默认路径 |
| `--points_csv` | `{sample_dir}/visualization/points.csv` | 点坐标 CSV |
| `--signal_ch` | `ch1` | 信号通道名（影响自动生成路径） |
| `--register_ch` | `ch0` | 配准通道名（warp 时用） |
| `--force_warp` | off | 强制重新生成 `points.csv` |
| `--columns` | 自动推断 | 坐标列，如 `x,y,z` |

### 点样式

| 参数 | 默认 | 说明 |
|------|------|------|
| `--point_color` | `#ff4d6d` | 点颜色（未按组着色时） |
| `--point_alpha` | `0.95` | 点透明度 |
| `--point_radius` | `40` | 球半径（µm，atlas 单位） |
| `--filter_to_brain` | off | 去掉 atlas mask 外的点 |

### 脑区分组

| 参数 | 默认 | 说明 |
|------|------|------|
| `--region_groups` | `config/region_groups.json` | 分组 JSON |
| `--group_names` | 全部组 | 逗号分隔，如 `"PFC,HIP,AMY"` |
| `--group_colors` | JSON 内颜色 | 逗号分隔覆盖色 |
| `--filter_points_by_group` | off | 只保留所选组内的点 |
| `--color_points_by_group` | off | 按组着色 |
| `--drop_unassigned_group_points` | off | 着色时丢弃组外点 |
| `--screenshot_per_group` | off | 每组输出一张 PNG |

### 区域 mesh

| 参数 | 默认 | 说明 |
|------|------|------|
| `--show_region` | — | 单个区域 mesh（acronym） |
| `--only_region` | — | 只显示该区域内的点 |
| `--region_outline` | off | 区域轮廓线 |
| `--hide_whole_brain` | off | 隐藏透明全脑 |
| `--region_alpha` | `0.18` | 区域 mesh 透明度 |
| `--region_color` | `#4cc9f0` | 单区域 mesh 颜色 |
| `--hemisphere` | `both` | `both` / `left` / `right` |

### 相机与输出

| 参数 | 默认 | 说明 |
|------|------|------|
| `--camera` | `three_quarters` | brainrender 预设视角 |
| `--camera_view` | — | 加载视角 JSON；不写路径则用样本默认可视化路径 |
| `--export_camera_view` | — | 交互保存视角；不写路径则存到样本默认可视化路径 |
| `--output` | — | 单张 PNG 路径（有则离屏渲染，无则交互窗口） |
| `--output_dir` | `{sample}/visualization/brainrender` | 按组截图目录 |
| `--screenshot_scale` | `2` | 截图分辨率倍数 |
| `--root_alpha` | `0.12` | 全脑透明度 |
| `--background` | `white` | 背景色 |
| `--title` | — | 窗口标题 |
| `--show_axes` | off | 显示坐标轴 |
| `--atlas_name` | `allen_mouse_25um` | BrainGlobe atlas 名 |

---

## 视角 JSON 格式

`nao_1_brainrender_view.json` 示例：

```json
{
  "pos": [-29176, 70303, 15030],
  "focal_point": [7830, 4296, -5694],
  "viewup": [-1, 0, 0],
  "distance": 78459,
  "clipping_range": [61887, 99152],
  "name": "nao_1_brainrender_view"
}
```

必填字段：`pos`, `viewup`, `clipping_range`（由 **V** 或 Shift+C 导出）。

---

## 常见问题

**全是红点**  
未加 `--color_points_by_group`。加上即可按 `region_groups.json` 里的颜色显示。

**按 V 没反应 / Step 3 报 Camera view JSON not found**  
Step 2 未成功保存视角。重新跑 `--export_camera_view`，确认终端出现 `Saved camera view to:`，或检查 `visualization/{sample}_brainrender_view.json` 是否存在。

**`unrecognized arguments: HIP AMY`**  
PowerShell 中 `--group_names` 未加引号。应写 `--group_names "PFC,HIP,AMY"`。

**`ModuleNotFoundError: pipeline_modules`**  
未设置 `PYTHONPATH=S:\Yifu` 或不在仓库根目录运行。

**`ModuleNotFoundError: brainrender`**  
用了 `yifu` 环境。brainrender 必须在 `napari` 环境运行。

**Atlas not downloaded**  
在 `napari` 环境下载一次 Allen atlas，或从已有机器复制 `~/.brainglobe` 缓存。

**ACC 与 PFC 点数重叠**  
设计如此：两组都含 `ACA`，同一点可同时计入两组统计；按组截图时各组独立过滤。

**dlPFC / vmPFC 是人类术语**  
在小鼠 Allen CCF 中是功能同源近似，见 `config/region_groups.json` 注释，非严格 one-to-one。

---

## 与 pipeline 的关系

- 本脚本 **不在** `main.py` 默认流程中；属于 `pipeline_modules/visualization/` 独立工具。
- 依赖上游产物：registration（transforms）、segmentation（mask zarr）、以及 atlas warp 后的体积或 `points.csv`。
- 坐标必须在 **atlas 空间**（µm）；脚本内部会做单位/轴序自动推断，一般无需手动翻转。

---

## 文件索引

| 路径 | 说明 |
|------|------|
| `pipeline_modules/visualization/render_points_brainrender.py` | 主脚本 |
| `config/region_groups.json` | 默认脑区分组 |
| `pipeline_modules/visualization/warp_mask_zarr_to_atlas_points.py` | mask → atlas 点/体积 |
| `pfc_groups.json` | 旧版 PFC 简写分组（仍兼容） |
