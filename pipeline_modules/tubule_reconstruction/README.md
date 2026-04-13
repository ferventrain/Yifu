# 血管网络重建模块

本目录用于从已经完成分割的血管 `binary mask zarr` 中重建血管骨架，并输出分支级和全局级血管网络参数。

当前实现文件：

- `kimimaro_reconstruction.py`

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
  每一条血管分支的长度、半径、曲折度等指标
- `vessel_network_summary.json`
  整个样本的血管网络汇总指标

## 当前版本说明

当前版本是第一版核心模块，特点如下：

- 已支持 `binary mask zarr -> skeleton -> branch metrics`
- 已支持按 `resolution_xyz` 输出带物理单位的长度参数
- 已支持 `--test` 模式，只处理物理存在的 chunk 并按 chunk 独立做重建
- 还没有接入 `main.py`
- 还没有实现真正的 block-wise skeleton stitching

如果后续要处理超大体积全量数据，建议下一步补：

1. block-wise skeletonization
2. block 间 skeleton stitching
3. 与样本级 `config.json` 和 `main.py` 自动集成

## `--test` 模式说明

`--test` 模式适合 smoke test 和局部验证，行为是：

- 只扫描并处理输入 Zarr 中真实存在的 chunk
- 每个 chunk 独立做连通域、骨架和分支统计
- 输出总表和 `vessel_chunk_metrics.csv`

需要注意：

- `--test` 不会做跨 chunk 血管连接
- 因此它的结果适合验证“模块能不能跑通”，不适合作为最终全局血管网络结果
