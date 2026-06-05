# 血管网络重建模块

本目录用于从已经完成分割的血管 `binary mask zarr` 中重建血管骨架，并输出分支级和全局级血管网络参数。

当前实现文件：

- `kimimaro_reconstruction.py`：从 binary mask 到 skeleton + 全局血管网络参数
- `region_vessel_analysis.py`：在已有 skeleton CSV 的基础上，按脑区（全名 / 缩写 / id）汇总血管参数

当前核心流程：

1. 读取二值血管 mask Zarr
2. 提取前景并做连通域标记
3. 使用 `kimimaro` 做 skeletonize
4. 从 skeleton 中提取分支
5. 输出分支表和整体汇总指标

## 配置示例

建议在 `config.json` 中加入如下配置：

```json
{
  "tubule_reconstruction": {
    "enabled": false,
    "mask_dataset_name": "0",
    "foreground_label": 1,
    "dust_threshold": 100,
    "parallel": 4,
    "fix_borders": true,
    "output_dirname": "tubule_reconstruction",
    "teasar_params": {
      "scale": 1.5,
      "const": 300,
      "pdrf_scale": 100000,
      "pdrf_exponent": 4,
      "soma_acceptance_threshold": 3500,
      "soma_detection_threshold": 750,
      "soma_invalidation_scale": 1.0,
      "soma_invalidation_const": 300,
      "max_paths": null
    }
  }
}
```

## 顶层参数说明

### `enabled`

- 类型：`bool`
- 作用：是否启用血管网络重建模块
- 建议：当前主流程还未自动接入，先作为配置占位保留

### `mask_dataset_name`

- 类型：`str`
- 作用：指定 Zarr group 中真正存放 mask 数据的数组名
- 常见值：`"0"`
- 说明：如果你的 Zarr 是 OME-Zarr 风格，通常数据在 `0` 这个 dataset 下

### `foreground_label`

- 类型：`int`
- 作用：指定 mask 中哪个像素值代表血管前景
- 常见值：`1`
- 说明：如果你的 mask 是 0/1 二值图，用 `1`；如果是 0/255，需要改成 `255`

### `dust_threshold`

- 类型：`int`
- 作用：过滤太小的连通域，避免噪声碎片进入骨架重建
- 建议：
  - 噪声较多时适当增大
  - 想保留细小末梢结构时适当减小
- 经验起点：`50` 到 `500`

### `parallel`

- 类型：`int`
- 作用：`kimimaro` 使用的并行 worker 数
- 建议：
  - 小数据可设 `1`
  - 大数据可按 CPU 核心数逐步增大
  - Windows 上建议保守一点，先从 `2` 或 `4` 开始

### `fix_borders`

- 类型：`bool`
- 作用：是否启用边界修正
- 含义：当结构靠近体数据边界时，帮助减少边界处不稳定骨架
- 建议：大多数情况下保持 `true`

### `output_dirname`

- 类型：`str`
- 作用：指定输出目录名称
- 当前用途：后续接入主流程时，可将结果统一输出到样本目录下的该子文件夹

## `teasar_params` 参数说明

`kimimaro` 底层使用 TEASAR 类方法提取 skeleton。以下参数会显著影响骨架的形态、分支数量和追踪稳定性。

### `scale`

- 类型：`float`
- 作用：控制骨架路径代价中的尺度项
- 影响：
  - 更大：更偏向平滑、主干优先
  - 更小：更容易保留细碎分支
- 建议：先用默认值 `1.5`

### `const`

- 类型：`int`
- 作用：TEASAR 路径代价中的常数项
- 影响：
  - 更大：骨架更保守
  - 更小：更容易延伸到细小结构
- 建议：先用默认值 `300`

### `pdrf_scale`

- 类型：`int`
- 作用：控制路径距离惩罚函数的强度
- 影响：会影响主干路径和分支路径的选择
- 建议：通常先不改，保留默认值 `100000`

### `pdrf_exponent`

- 类型：`int`
- 作用：控制路径距离惩罚函数的指数
- 影响：数值越大，惩罚变化越陡
- 建议：先保持 `4`

### `soma_acceptance_threshold`

- 类型：`int`
- 作用：原本更偏神经元大体结构识别的阈值，用于接受较大的团块区域
- 对血管任务的意义：
  - 如果存在粗大血管团块、膨大结构，可能会产生影响
  - 普通血管网络场景一般不需要频繁调整
- 建议：先保留默认值

### `soma_detection_threshold`

- 类型：`int`
- 作用：控制大块结构检测阈值
- 对血管任务的意义：更多是兼容 `kimimaro` 原始设计，血管数据中通常不作为首要调参项
- 建议：先保留默认值

### `soma_invalidation_scale`

- 类型：`float`
- 作用：控制大块结构附近路径失效范围的尺度
- 对血管任务的意义：通常不是血管任务中的优先调参项
- 建议：先保留 `1.0`

### `soma_invalidation_const`

- 类型：`int`
- 作用：控制大块结构附近路径失效范围的常数项
- 对血管任务的意义：通常不是首要调参项
- 建议：先保留默认值 `300`

### `max_paths`

- 类型：`int | null`
- 作用：限制每个连通域可生成的最大路径数
- 含义：
  - `null` 表示不限制
  - 设置为较小值可以降低复杂样本的运行时间
- 建议：
  - 小样本或调试阶段可以设置一个较小值
  - 正式分析建议先保持 `null`

## 哪些参数最值得先调

如果你后面要真正调参，优先级建议是：

1. `foreground_label`
2. `dust_threshold`
3. `scale`
4. `const`
5. `parallel`

也就是说，大多数血管数据第一轮不需要把所有参数都调一遍，先把“前景值、噪声过滤、骨架保守程度”调顺就够了。

## 当前输出文件

运行后会输出：

- `vessel_branch_metrics.csv`
  每一条血管分支的长度、半径、曲折度、branch depth、terminal branch 标记等指标
- `vessel_network_summary.json`
  整个样本的血管网络汇总指标，包括 branch point count、branch point 间 path length 统计、mask vessel volume、mean tortuosity 和 mean branch depth
- `skeleton_vertices.csv`
  骨架节点坐标表，包含每个节点的 `z_um / y_um / x_um / radius_um`
- `skeleton_edges.csv`
  骨架连边表，包含每条边连接的两个节点及对应空间坐标
- `swc/`
  如果加上 `--save_swc`，会导出每条 skeleton 对应的 `.swc` 文件，可用于 `kimimaro view`

如果需要输出 skeleton，可在命令行中加入：

```bash
--save_skeleton
```

如果还需要输出 SWC，可加入：

```bash
--save_swc
```

如果需要在 napari 中查看重建出来的 edge，可使用：

```bash
python pipeline_modules/tubule_reconstruction/view_skeleton_napari.py \
  --skeleton_edges_csv "path/to/skeleton_edges.csv" \
  --image_zarr "path/to/image.zarr" \
  --mask_zarr "path/to/mask.zarr" \
  --resolution_xyz "1.8,1.8,2.0"
```

如果是 `--test` 模式输出，脚本会优先按 `skeleton_edges.csv` 里的 `chunk_index` 和 `chunk_start_zyx` 自动加载对应 chunk。
当前脚本会使用 napari 的 `Vectors` 层来显示 skeleton edge，比把每条边当成单独 shape 更轻量。

如果你安装了 `kimimaro[view]`，可以直接查看某一条骨架：

```bash
kimimaro view path/to/output/swc/skeleton_000000.swc
```

## 当前版本说明

当前版本是第一版核心模块，特点如下：

- 已支持 `binary mask zarr -> skeleton -> branch metrics`
- 已支持按 `resolution_xyz` 输出带物理单位的长度参数
- 已支持 `--test` 模式，只处理物理存在的 chunk 并按 chunk 独立做重建
- 已支持 `--chunkwise` 模式，按 chunk 流式读取数据，避免一次性把整块体数据读入内存
- 已支持 `--halo_zyx`，在 chunkwise 模式下读取邻域 halo 来减轻边界伪影
- 已支持 chunkwise 下的跨 chunk skeleton stitching
- 还没有接入 `main.py`
- 还没有实现更高级的全局 graph merging 与拓扑清理

如果后续要处理超大体积全量数据，建议下一步补：

1. block-wise skeletonization
2. block 间 skeleton stitching
3. 与样本级 `config.json` 和 `main.py` 自动集成

## `--test` 模式说明

`--test` 模式适合 smoke test 和局部验证，行为是：

- 只扫描并处理输入 Zarr 中真实存在的 chunk
- 每个 chunk 独立做连通域、骨架和分支统计
- 输出总表和 `vessel_chunk_metrics.csv`
- 如果加上 `--save_skeleton`，还会额外输出包含 chunk 信息的 skeleton 顶点和边表

需要注意：

- `--test` 不会做跨 chunk 血管连接
- 因此它的结果适合验证“模块能不能跑通”，不适合作为最终全局血管网络结果

## `--chunkwise` 模式说明

`--chunkwise` 用于大体积 mask 的正式分块处理，行为是：

- 按 chunk 逐块读取，而不是整块载入内存
- 可通过 `--halo_zyx` 读取邻域上下文，例如 `--halo_zyx "8,32,32"`
- 如果加上 `--save_skeleton`，会优先导出位于 core chunk 内的 skeleton edge，减少邻域 halo 引入的重复显示
- 默认会尝试对相邻 chunk 边界上的 endpoint 做最近邻 stitching
- 可通过 `--stitch_max_distance_um` 控制跨 chunk 连边的最大距离
- 如果不希望 stitching，可加入 `--no_stitch`

需要注意：

- 当前版本已经做了“halo 读邻域 + core 区域导出 + 相邻 chunk endpoint stitching”
- 但还没有做更高级的全局 graph merging、环路修正和拓扑清理

## 关于 terminal / 神经末梢数量

如果你后面想把同样的方法迁移到神经末梢分析，`terminal count` 可以作为一个很有用的第一近似，但要注意：

- `num_end_points` 更接近“图上的末端节点数量”
- `num_end_points_non_boundary` 会进一步排除贴在 chunk 边界上的末端，更适合避免边界截断带来的假阳性
- `num_terminal_branches` 更接近“末端分支段数量”

如果你的网络近似树结构，并且已经做过噪声清理和短 spur 修剪，那么“末端节点数”通常可以近似看作“神经末梢数”。  
但如果存在：

- 环路
- 边界截断
- segmentation 噪声
- 很短的假分支

那就不能直接把 raw terminal count 当作最终神经末梢数量，最好至少再做一轮：

1. 短 spur 过滤
2. 边界 terminal 排除
3. 必要时按 terminal branch 长度再筛选

## 按脑区输出血管参数（`region_vessel_analysis.py`）

在完成一次带 `--save_skeleton` 的重建后，可以用 `region_vessel_analysis` 直接基于已有的 `skeleton_vertices.csv` 和 `skeleton_edges.csv`，按脑区输出血管参数，**无需重新跑骨架化**。

### 工作原理

1. 从 Allen region CSV 加载脑区树，建立 `id / 缩写 / 全名` 三套查找表
2. 将用户提供的脑区查询解析到节点，并收集该节点的**全部子树 id**
3. 读取 registered 的 annotation label Zarr，把每个 skeleton 顶点（按 `z_um/y_um/x_um`）和每条边的中点映射到对应体素，读取 annotation label
4. 落入子树 id 集合的顶点 / 边归入该脑区，按脑区汇总参数

顶点 / 边使用各自独立的 label 判定；边是否属于某脑区看其中点所在体素的 label（与 chunkwise core 判定保持一致）。

### 输入要求

- `skeleton_vertices.csv`：由 `--save_skeleton` 产出，必须包含 `skeleton_id / node_id / z_um / y_um / x_um`，可选 `radius_um`
- `skeleton_edges.csv`：必须包含 `skeleton_id / source_node / target_node / edge_length_um` 和源点、终点的 `*_z_um / *_y_um / *_x_um`
- `annotation_zarr`：已 registered 的 Allen label 体积，与 skeleton 的 um 坐标**共用原点**
- `region CSV`：如 `pipeline_modules/registration/Region_Csv_Rev1_updated.CSV`

**关键假设**：skeleton CSV 的 um 坐标和 annotation Zarr 在同一物理坐标系，点通过 `floor(point_um / annotation_resolution_zyx)` 映射到体素。如果两者坐标原点不同，需要先做对齐。

### 输出文件

写入 `--output_dir` 下：

- `region_vessel_summary.csv`
- `region_vessel_summary.json`

每行（每个脑区查询）包含的字段：

| 字段 | 含义 |
| --- | --- |
| `query` | 用户输入的原始查询字符串 |
| `region_id / region_acronym / region_name` | 解析到的脑区 |
| `num_subtree_ids` | 子树节点数（包括自身） |
| `num_branch_points` | 脑区内度 >= 3 的 branch point 数 |
| `branch_point_path_length_sum_um` | 两个 branch point 之间 path 长度之和 |
| `branch_point_path_length_mean_um` | 两个 branch point 之间 path 长度均值 |
| `branch_point_path_length_sd_um` | 两个 branch point 之间 path 长度 SD |
| `mask_voxels` | 该脑区内 vessel mask 前景体素数 |
| `vessel_volume_um3` | 直接用 mask voxel count * voxel volume 得到的血管体积 |
| `mean_tortuosity` | 该脑区内 branch path 的 tortuosity 均值 |
| `mean_branch_depth` | 该脑区内 branch depth 均值 |

### 命令行用法

```powershell
python -m pipeline_modules.tubule_reconstruction.region_vessel_analysis `
  --vertex_csv out/skeleton_vertices.csv `
  --edge_csv out/skeleton_edges.csv `
  --branch_csv out/vessel_branch_metrics.csv `
  --mask_zarr sample/ch1_mask.zarr `
  --annotation_zarr registered/annotation.zarr `
  --annotation_dataset_name 0 `
  --annotation_resolution_xyz 25,25,25 `
  --cfg pipeline_modules/registration/Region_Csv_Rev1_updated.CSV `
  --regions "CTX;HPF;Thalamus;315" `
  --output_dir out/region_vessels
```

参数说明：

- `--vertex_csv / --edge_csv`：前置重建步骤的 skeleton 输出（必须用 `--save_skeleton` 跑过）
- `--branch_csv`：前置重建步骤的 `vessel_branch_metrics.csv`；不提供时会尝试从 skeleton 表重建 branch path 统计
- `--mask_zarr / --mask_dataset_name / --foreground_label`：用于直接按 mask 体素统计血管 volume
- `--annotation_zarr / --annotation_dataset_name`：registered annotation label 的 Zarr 路径与内部 dataset 名
- `--annotation_resolution_xyz`：annotation 体素物理尺寸，单位 μm，顺序 `x,y,z`。25 μm Allen CCF 写 `25,25,25`
- `--cfg`：Allen region CSV
- `--regions`：脑区查询列表，支持 **acronym / 全名 / 整数 id** 三种形式任意混用；分隔符支持逗号、分号、换行；大小写不敏感；自动包含该脑区子树
- `--output_dir`：输出目录

### Python API 用法

```python
from pipeline_modules.tubule_reconstruction import analyze_regions_from_skeleton

result = analyze_regions_from_skeleton(
    vertex_csv_path="out/skeleton_vertices.csv",
    edge_csv_path="out/skeleton_edges.csv",
    branch_csv_path="out/vessel_branch_metrics.csv",
    mask_zarr_path="sample/ch1_mask.zarr",
    annotation_zarr_path="registered/annotation.zarr",
    region_cfg_csv="pipeline_modules/registration/Region_Csv_Rev1_updated.CSV",
    regions=["CTX", "Hippocampal formation", 315],
    output_dir="out/region_vessels",
    annotation_resolution_xyz="25,25,25",
)
summary_df = result["summary_table"]
```

### 查询匹配顺序与歧义处理

- 如果输入是纯整数，按 `region_id` 查
- 否则先查 acronym，再查 full name；均不区分大小写
- 如果一个字符串查询在同一层级上对应多个 id（歧义），会抛 `ValueError` 并建议改用 id
- 找不到时抛 `KeyError`

### 与 `check_region_coverage_zarr.py` 的区别

`registration/check_region_coverage_zarr.py` 做的是“这个脑区在 label 体里有多少体素”的体素计数；
`region_vessel_analysis.py` 做的是“这个脑区内血管骨架的长度 / 半径 / 分支拓扑”，侧重血管参数而非体素覆盖率。

---

## Agent-Native 集成接口

本模块已按"Agent-Native"原则改造，可被自动化 agent 直接发现和调用。

### 结构化配置（Pydantic）

```python
from pipeline_modules.tubule_reconstruction import TubuleReconstructionCfg, RegionVesselAnalysisCfg

# 从 dict（如解析自 config.json）构建并验证
cfg = TubuleReconstructionCfg(**config_json["tubule_reconstruction"])

# 导出 JSON Schema（供 agent / IDE 做参数校验）
from pipeline_modules.tubule_reconstruction import export_json_schema
schema = export_json_schema()
```

### SampleLayout — 统一路径管理

```python
from pipeline_modules.tubule_reconstruction import layout_for_sample

layout = layout_for_sample("/data/mouse01", signal_ch="ch0")
# 所有路径通过属性访问，无需拼接字符串：
layout.mask_zarr                   # /data/mouse01/ch0_mask.zarr
layout.tubule_reconstruction_dir   # /data/mouse01/tubule_reconstruction
layout.tubule_vertex_csv           # /data/mouse01/tubule_reconstruction/skeleton_vertices.csv
layout.atlas_label_zarr            # /data/mouse01/upsampled_atlas_label.zarr
```

### Capability Manifest — 机器可读能力说明

```python
from pipeline_modules.tubule_reconstruction import load_capability_manifest

manifest = load_capability_manifest()
# manifest["entrypoints"] 列出所有入口函数、输入参数、输出文件、前置条件
```

也可直接读取 JSON 文件：`pipeline_modules/tubule_reconstruction/capability_manifest.json`

全项目模块索引见：`capabilities.json`（项目根目录）。

### 结构化错误与运行记录

- 每次成功调用 `analyze_binary_mask_zarr[_chunkwise]` 或 `analyze_regions_from_skeleton` 后，`output_dir` 下自动写入 `_run_manifest.json`，记录输入参数、输出文件列表、耗时。
- CLI 加 `--json_logs` 可将日志以 NDJSON 格式输出到 stderr，便于 agent 解析。
- 发生结构化错误时，CLI 在 stderr 输出 `{"error_code": "...", "message": "..."}` 并以对应退出码退出（2 = 配置错误，3 = 输入缺失，1 = 运行时错误）。

### 测试

```powershell
# 需要激活 conda 环境后执行
pytest tests/test_utils.py tests/test_tubule_config.py tests/test_region_vessel_analysis.py -v
```

测试覆盖：`SampleLayout`、`PipelineError`/`ErrorCode`、`write_run_manifest`、Pydantic 配置模型、`analyze_regions_from_skeleton` 端到端 smoke test（全部使用合成数据，不依赖 GPU / kimimaro）。

