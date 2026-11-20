# LSFM 主流程使用说明

本仓库当前推荐通过 `main.py` 统一执行主流程。它会按顺序串起以下步骤：

1. 信号通道预处理
2. TIFF 转 Zarr
3. 配准通道下采样
4. 分割
5. ANTs 配准
6. 脑区统计分析

如果你的目标是“把一个样本从原始 TIFF 跑到最终统计结果”，优先看这份文档即可。

## 0. 项目核心设计宗旨（Agent-Native）

本仓库在重构方向上按下列五条原则执行。新写的代码、以及后续逐模块重构时，都应该遵守这些约定，以便人和 agent 都能稳定、可发现地调用。

### 0.1 每个模块都要暴露 Python API，argparse 只是薄壳

- 每个 `pipeline_modules/<module>/` 在 `__init__.py` 中显式 `__all__` 导出纯函数入口
- 纯函数**接收 Python 对象、返回 Python 对象**（通常是 `dict` / `dataclass`）
- 纯函数内部**不允许** `sys.exit` / `print` / 从 `argv` 读参数
- CLI 入口 (`if __name__ == "__main__"`) 退化为薄壳：解析参数 → 调用纯函数 → 把结果以 JSON 打到 stdout
- 这样 agent 可以 `from pipeline_modules.xxx import yyy` 直接调用并拿到返回值

### 0.2 结构化输入输出 + 结构化错误

- 每个模块运行完写一份 `<output_dir>/_run_manifest.json`：输入参数、输出文件列表（绝对路径 + 字节数）、耗时、模块版本、警告
- 统一的错误模型 `PipelineError(code, message, context)`，`code` 是枚举（如 `INPUT_NOT_FOUND / CONFIG_INVALID / CUDA_OOM`）
- `print` 改用 `logging`；CLI 失败时把结构化错误以 JSON 打到 stderr（便于 agent 解析）
- 失败路径**不**用 `sys.exit(1)` 吞掉上下文，应抛结构化异常并在 CLI 层映射成非零退出码 + JSON 错误体

### 0.3 配置用 Pydantic / dataclass，并导出 JSON Schema

- 每个模块的配置段由一个配置模型定义（各 `pipeline_modules/<module>/config.py`），统一契约见对应的 `capability_manifest.json`
- 顶层 `input / preprocessing / registration / segmentation / analysis / tubule_reconstruction` 由 `main.py` 组合
- 配置分层：补跑 config 缺省键自动继承 `config_N.json ← config.json ← config/config_cfos_template.json`
- `main.py` 启动时校验一次坏配置，而不是到中途某步才崩

### 0.4 机器可读的 capability manifest

- 根目录 `capabilities.json` 登记模块；每个模块目录下的 `capability_manifest.json` 是**唯一契约**：
  - `id`、自然语言描述
  - Python 入口路径、CLI 入口
  - 输入 / 输出、前置条件（依赖哪些上游产物）
- Agent 通过 manifest 做**能力发现**和**依赖推断**，不需要阅读源码就知道"当前能跑什么、结果会写到哪"
- 这也是把仓库包成 DSH 插件 / MCP server 时的数据源

### 0.5 样本目录布局由代码（而非注释）表达

- 在 `pipeline_modules/utils/sample_layout.py` 用 `SampleLayout` 类集中表达 `sample_dir/` 下所有约定路径（`chX.zarr`、`chX_mask.zarr`、`upsampled_atlas_label.zarr`、`tubule_reconstruction/` 等）
- 所有模块、`main.py`、agent 一律通过 `SampleLayout` 访问这些路径，不再在各处拼字符串
- `SampleLayout.as_dict()` 可列出全部约定产物路径，配合 `require_exists=False` 用于判断"哪些步骤已做 / 下一步能做什么"
- 路径命名保持与本文档第 7 节一致，避免破坏现有产物

### 改造优先级

新模块**从第一天起就按这五条写**；老模块按"杠杆最大"的顺序逐个重构：

1. `tubule_reconstruction`（已部分符合 0.1，作为改造模板）
2. `registration`
3. `segmentation`
4. `preprocessing`
5. 最后改 `main.py`，切换到 `SampleLayout` + Pydantic 配置 + 结构化错误

## 1. 环境准备

建议先创建 Conda 环境：

```bash
conda env create -f environment.yml
conda activate yifu
```

如果已经有环境，也至少要确保这些关键依赖可用：

- `antspyx`
- `cellpose`
- `torch`
- `tifffile`
- `zarr`
- `opencv-python-headless`
- `nibabel`

## 2. 样本目录结构

`main.py` 要求传入 `--sample_dir`，目录下至少要有通道文件夹，例如：

```text
sample_dir/
├── ch0/
├── ch1/
└── ...
```

约定如下：

- `signal` 通道：用于分割和最终统计
- `registration` 通道：用于和 atlas 做配准

如果配置里写的是：

```json
"channels": {
  "signal": "1",
  "registration": "0"
}
```

那么主流程会默认使用：

- `sample_dir/ch1/` 作为信号通道
- `sample_dir/ch0/` 作为配准通道

## 3. 配置文件

先复制模板：

```bash
copy config\config_template.json config.json
```

然后至少检查下面这些字段。它们是 `main.py` 当前真正依赖的核心配置。

### 最小可用示例

```json
{
  "input": {
    "resolution_xyz": [1.8, 1.8, 2.0],
    "channels": {
      "signal": "1",
      "registration": "0"
    }
  },
  "preprocessing": {
    "tophat": {
      "apply": false,
      "kernel_size": 21
    },
    "rolling_ball": {
      "apply": false,
      "radius": 50
    },
    "median_filter": {
      "apply": false,
      "kernel_size": 3
    },
    "scattering_removal": {
      "apply": false,
      "sigma": 50.0,
      "weight": 1.0
    },
    "clahe": {
      "apply": false,
      "clip_limit": 2.0,
      "tile_grid_size": 8
    },
    "downsample": {
      "target_resolution_xyz": [25.0, 25.0, 25.0]
    },
    "zarr": {
      "chunk_size": [128, 256, 256]
    }
  },
  "registration": {
    "atlas_path": "S:/Yifu/Allen_brainatlas/atlas.tiff",
    "annotation_path": "S:/Yifu/Allen_brainatlas/atlas_label.tiff"
  },
  "segmentation": {
    "method": "cellpose",
    "cellpose": {
      "model": "cyto3",
      "diameter": 30.0,
      "workers": 4
    },
    "threshold": {
      "value": 1000,
      "sigma": 0
    }
  },
  "analysis": {
    "density_config": "pipeline_modules/registration/Region_Csv_Rev2_updated.CSV"
  }
}
```

### 各字段作用

- `input.resolution_xyz`
  原始数据体素分辨率，格式为 `[x, y, z]`，单位通常是微米。
- `input.channels.signal`
  信号通道编号，不带 `ch` 前缀。
- `input.channels.registration`
  配准通道编号，不带 `ch` 前缀。
- `preprocessing.*.apply`
  控制是否对信号通道 TIFF 做增强。
- `preprocessing.downsample.target_resolution_xyz`
  配准通道下采样后的目标分辨率，`main.py` 会自动计算下采样比例。
- `preprocessing.zarr.chunk_size`
  TIFF 转 Zarr 时使用的 chunk。
- `registration.atlas_path`
  atlas 图像路径。
- `registration.annotation_path`
  atlas 标签路径。
- `segmentation.method`
  目前支持 `cellpose` 或 `threshold`。
- `segmentation.cellpose.model`
  Cellpose 模型名。
- `segmentation.cellpose.diameter`
  Cellpose 直径参数。
- `segmentation.cellpose.workers`
  Cellpose 并行 worker 数。
- `segmentation.threshold.value`
  阈值分割阈值。
- `segmentation.threshold.sigma`
  阈值分割前的平滑参数。
- `analysis.density_config`
  脑区层级 CSV 配置。

## 4. 运行主流程

**唯一推荐入口（一条命令，跑完可以关终端）**：

```bash
python -m pipeline_modules.harness run \
  --sample-dir "S:/Arivis_Analysis/<sample>" \
  --config "S:/Arivis_Analysis/<sample>/config.json"
```

它会校验输入、建任务、启动唯一 worker 并立即返回 `run_id`；同一样本重复提交会去重返回原
`run_id`（并确保 worker 在线）。之后在 DeepSeek Harness 里用 "Yifu 队列" 面板或
`yifu_jobs` / `yifu_job` 工具看当前步骤、运行时间、心跳和结果；旧监视页
（http://127.0.0.1:8766）迁移期保留，只读。

任务结束时只有"进程正常结束 + 结果文件存在且可读 + artifact manifest 写入"才会标记
`succeeded`，失败会给出结构化错误（哪一步 / 什么类别 / 能否重试 / 建议动作）。
提交前可先 `python -m pipeline_modules.harness preflight --sample-dir ...` 做一次只读校验；
取消用 `python -m pipeline_modules.harness cancel <run_id>`；查看任务
`python -m pipeline_modules.harness list`。

生产/全脑运行不要绕过队列裸跑：同一时间只跑一个 `main.py` 任务，Monitor 显示有任务运行中时
不要另启重型流程。

`main.py` 表达不了的定制程序用 wrap 挂到同一个队列（零改动）：

```bash
python -m pipeline_modules.harness wrap --sample-dir "S:/Arivis_Analysis/<sample>" \
  --title "<标题>" -- python my_script.py --args
```

旧的两行 status 文件 + `attach` 驱动契约属于兼容入口（wrap 内部仍在用），不要再用它新接流程，
也不要手工编辑 queue 文件。

直接启动（不经队列，前台阻塞，仅供调试单个步骤）：

```bash
python main.py --config config.json --sample_dir "S:\path\to\sample_dir"
```

这是原来的启动方式，仍然可用，但它不会被 PM 记录，也不会校验输出。`--sample_dir` 当前
基本是必填项。

### 常见变体

只检查 harness/模块能力清单能否加载（不跑流程）：

```bash
python main.py --test
```

## 5. 命令行参数

- `--config`
  配置文件路径，默认是样本目录下的 `config.json`。
- `--sample_dir`
  样本根目录，必须包含 `ch0`、`ch1` 这类通道文件夹。
- `--test`
  自检模式：加载模块能力清单后退出，不执行完整流程。

## 6. 主流程实际做了什么

`main.py` 当前的真实执行逻辑如下：

### 6.1 预处理

- 只对 `signal` 通道做增强
- 如果没有启用任何预处理步骤，则直接使用原始 TIFF
- 预处理输出目录为 `chX_preprocessed/`

当前 `Preprocessor` 真正支持的增强步骤是：

- `tophat`
- `rolling_ball`
- `median_filter`
- `scattering_removal`
- `clahe`
- `channel_subtraction`（通道相减，抑制自发荧光串扰）

### 6.2 信号通道转 Zarr

主流程会把信号通道转换成：

```text
sample_dir/chX.zarr
```

生产/全脑运行推荐 Imaris-direct：`python -m pipeline_modules.preprocessing.ims_to_zarr`
直接从 `.ims` 流式写 Zarr，绕过 TIFF 中间产物（见 `preprocessing/capability_manifest.json`）。
如果该 Zarr 已存在，会自动跳过这一阶段。

### 6.3 配准通道下采样

主流程会对 `registration` 通道做下采样，并生成：

```text
sample_dir/chY_downsample/
└── volume.nii.gz
```

这里的下采样比例由：

- `input.resolution_xyz`
- `preprocessing.downsample.target_resolution_xyz`

自动计算得到。

### 6.4 分割

`segmentation.method` 决定分割方式：`cellpose`（分布式 3D）、`threshold`、
`cfos_unet`（本地 checkpoint 3D U-Net）或 `spotiflow`（斑点检测）。分割结果写：

```text
sample_dir/chX_mask.zarr
```

之后主流程还会把它导出成 TIFF：

```text
sample_dir/chX_mask/
```

如果你已经提前准备好了以下任意结果，主流程会自动复用：

- 已有 `chX_mask/` 且目录非空：直接复用，不再重新分割
- 已有 `chX_mask.zarr` 但还没有 `chX_mask/`：不重新分割，但会补导出 TIFF

### 6.5 配准

主流程调用 `pipeline_modules/registration/ANTs_registration.py`，把 atlas 标签映射回样本空间，主要输出：

- `upsampled_atlas_label/`
- `transforms/`

如果 `upsampled_atlas_label/` 已经存在且非空，会自动跳过配准。

### 6.6 脑区统计分析

分析阶段会确保以下 Zarr 都存在：

- `chX.zarr`
- `chX_mask.zarr`
- `upsampled_atlas_label.zarr`

最终输出（`SampleLayout.brain_distribution_stats_xlsx`）：

```text
sample_dir/results/<sample>_ch<X>_brain_distribution_stats.xlsx
```

## 7. 运行后常见输出

假设：

- `signal = 1`
- `registration = 0`

那么一次完整运行后，常见结果包括：

- `ch1_preprocessed/`：预处理后的信号通道 TIFF
- `ch1.zarr/`：信号通道 Zarr
- `ch1_mask.zarr/`：分割结果 Zarr
- `ch1_mask/`：分割结果 TIFF
- `ch0_downsample/`：配准通道下采样结果
- `upsampled_atlas_label/`：映射回样本空间的 atlas 标签 TIFF
- `upsampled_atlas_label.zarr/`：atlas 标签 Zarr
- `results/<sample>_ch1_brain_distribution_stats.xlsx`：最终统计结果
- `transforms/`：ANTs 变换文件

## 8. 常见问题

### 8.1 `--sample_dir` 不传可以吗

完整主流程下不可以，当前必须显式传入 `--sample_dir`。只有 `--test` 模式不需要。

### 8.2 `registration.mode` 会生效吗

当前 `main.py` 内部固定使用 `atlas2image`。也就是说，即使配置文件里写了别的值，主流程仍按 `atlas2image` 跑。

### 8.3 `config_template.json` 里有些字段为什么没效果

模板比主流程更“宽”。按当前代码，下面这些字段不是 `main.py` 主流程的核心输入，或者不会被完整使用：

- `registration.mode`（固定 `atlas2image`）
- `segmentation.cellpose.use_gpu`
- `segmentation.cellpose.block_size`
- `preprocessing.homomorphic_filter`

`tubule_reconstruction`（第 6 步）与 `segmentation.threshold.min_object_size` 已经接入主流程，
配置后会生效。具体字段以各模块 `capability_manifest.json` 为准。

### 8.4 预处理会不会作用到配准通道

不会。增强预处理只作用于信号通道；配准通道用原始数据（生产推荐直接从 `.ims` 下采样成
`volume.nii.gz`，见 `pipeline_modules/preprocessing/capability_manifest.json`）。

## 9. 相关模块

如果你需要看某个阶段的独立说明，可以继续参考：

- `pipeline_modules/preprocessing/`
- `pipeline_modules/registration/`
- `pipeline_modules/segmentation/`

完整契约以每个模块的 `capability_manifest.json` 为准。日常跑样本用标准入口：

```bash
python -m pipeline_modules.harness run --sample-dir "<样本目录>" --config "<样本目录>/config.json"
```
