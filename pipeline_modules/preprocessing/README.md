# LSFM preprocessing 模块

本文档按仓库根目录 [README.md](../../README.md) 中的 Agent-Native 约束重写，分成两层内容：

1. 当前代码已经具备的 preprocessing 能力
2. preprocessing 模块为了满足 Agent-Native 目标，后续必须遵守的接口和文档约束

## 1. 当前状态

`pipeline_modules/preprocessing/` 已完成一轮 Agent-Native 改造，但还没有完全达到根 README 定义的最终目标。

当前已经存在、且可以真实使用的能力：

- `__init__.py`
  已提供 preprocessing 包级导出面，agent 和其他模块可以从 `pipeline_modules.preprocessing` 统一发现入口。
- `config.py`
  已提供 dataclass 风格的配置模型、schema 导出、capability manifest 加载和 `layout_for_sample(...)`。
- `preprocessor.py`
  配置驱动的 TIFF 批处理预处理，支持可选通道减法、若干增强步骤、结构化 CLI 返回和 per-output manifest。
- `channel_subtraction.py`
  独立通道背景减法工具。
- `downsample.py`
  3D TIFF 序列下采样，并导出 TIFF 栈、`volume.nii.gz` 和 `_run_manifest.json`。
- `tiff_to_zarr.py`
  TIFF 序列转 Zarr，并写 `_run_manifest.json`。
- `upsample_mask.py`
  低分辨率 mask 上采样到全分辨率 Zarr。
- 单图像处理脚本
  `tophat_background.py`、`rolling_ball_background.py`、`scattering_removal.py`、`median_filter.py`、`masked_clahe.py`。

当前还没有完成的 Agent-Native 能力：

- 不是所有脚本都已经改成“Python API 为主，CLI 为薄壳”；单图像脚本仍主要是脚本式接口。
- `channel_subtraction.py` 本身还没有单独接入统一的结构化 CLI 错误/manifest。
- 还没有并入仓库级的统一 `PipelineConfig` / `config_schema.py`。
- `main.py` 还没有完全切换到 preprocessing 的新导出 API、`SampleLayout` 和结构化错误路径。
- 部分能力仍依赖运行环境里安装 `opencv-python`、`nibabel`、`zarr` 等可选依赖。

## 2. Agent-Native 约束

根据根目录 [README.md](../../README.md)，preprocessing 文档和后续代码改造必须遵守以下原则。

### 2.1 Python API 是主入口，CLI 只是薄壳

preprocessing 模块的目标形态应当是：

- 在 `pipeline_modules/preprocessing/__init__.py` 中显式导出 `__all__`
- 主要能力以可导入的 Python API 形式提供，而不是只能通过 `python xxx.py` 调用
- API 接收 Python 对象，返回 Python 对象
- CLI 只负责：
  1. 解析参数
  2. 调用 Python API
  3. 将结果以 JSON 打到 stdout

当前状态：

- 已有 `pipeline_modules.preprocessing` 包级导出
- `preprocessor.py`、`downsample.py`、`tiff_to_zarr.py` 已同时暴露 Python API 和结构化 CLI
- 但单图像脚本和 `channel_subtraction.py` 还没有统一到这一层

### 2.2 结构化输出、日志和错误

preprocessing 后续应提供：

- 统一错误模型，例如 `PipelineError(code, message, context)`
- 统一 `logging` 输出，避免模块内部直接 `print`
- 可选 `--json_logs`，输出 NDJSON 事件流
- 每次运行写出 `<output_dir>/_run_manifest.json`

`_run_manifest.json` 至少应包含：

- 输入参数
- 输出文件列表
- 每个输出的绝对路径和字节数
- 耗时
- 模块版本
- warnings

当前状态：

- `preprocessor.py`、`downsample.py`、`tiff_to_zarr.py` 已接入结构化错误边界和 `_run_manifest.json`
- 但 preprocessing 目录下并不是所有脚本都已完成同样改造

### 2.3 配置必须可验证

根 README 要求配置最终由 Pydantic 或 dataclass 模型统一表达，并能导出 JSON Schema。对 preprocessing 来说，这意味着：

- 文档中的配置字段必须和顶层 `PipelineConfig.preprocessing` 对齐
- 文档不应再把“脚本私有参数”描述成长期稳定接口，除非它们已经进入统一配置模型
- `main.py` 启动时应先校验配置，再进入 preprocessing

当前状态：

- preprocessing 仍直接读取 JSON 字典
- 文档只能把当前字段视为“现状接口”，不能视为最终稳定 schema

### 2.4 capability manifest

preprocessing 最终应在仓库根目录的 capability manifest 中声明：

- `id`
- 自然语言描述
- Python 入口
- CLI 入口
- 输入 schema
- 输出 schema
- 前置依赖和产物依赖

这样 agent 不用读源码也能知道：

- preprocessing 能做什么
- 需要什么输入
- 会产出到哪里

当前状态：

- `pipeline_modules/preprocessing/capability_manifest.json` 已建立
- 根目录 [capabilities.json](../../capabilities.json) 也已把 preprocessing 标记为 `agent_native: true`

### 2.5 路径必须收敛到 `SampleLayout`

根 README 明确要求样本目录布局由代码统一表达。preprocessing 相关路径后续应通过 `SampleLayout` 访问，而不是散落在脚本里拼接字符串。

对 preprocessing 来说，至少包括：

- 原始通道目录，如 `ch0/`、`ch1/`
- 预处理输出目录，如 `chX_preprocessed/`
- Zarr 输出，如 `chX.zarr`
- 下采样输出，如 `chY_downsample/`

当前状态：

- 独立入口已开始使用 `layout_for_sample(...)`
- 但还没有做到 preprocessing 目录下所有路径都完全经由 `SampleLayout` 访问

## 3. 当前主流程里 preprocessing 的真实接入方式

这部分描述“现在代码实际怎么跑”，供使用者和 agent 对齐预期。

### 3.1 `main.py` 里真正用到的 preprocessing 能力

主流程 [main.py](../../main.py) 当前直接使用了三类 preprocessing 能力：

- 从 `pipeline_modules.preprocessing.preprocessor` import `Preprocessor`
- 调用 `pipeline_modules/preprocessing/downsample.py`
- 调用 `pipeline_modules/preprocessing/tiff_to_zarr.py`

其中，`main.py` 对信号通道的处理逻辑是：

- 如果 `Preprocessor(preprocessing_cfg).steps` 非空，则对 `signal` 通道做增强
- 输出目录为 `sample_dir/chX_preprocessed/`
- 然后把该目录转成 `sample_dir/chX.zarr`

需要特别注意：

- `main.py` 当前不会在主流程里调用 `channel_subtraction`
- `main.py` 当前只对 `signal` 通道做增强预处理
- `registration` 通道不会走这些增强步骤，而是直接做下采样

### 3.2 预处理步骤的执行顺序

`Preprocessor` 当前通过遍历 `config["preprocessing"]` 的键顺序构建步骤，并跳过以下非增强字段：

- `downsample`
- `zarr`
- `channel_subtraction`

因此，增强步骤的顺序不是硬编码常量，而是由 JSON 配置中的字段顺序决定。

当前支持的增强步骤只有：

- `tophat`
- `rolling_ball`
- `median_filter`
- `scattering_removal`
- `clahe`

## 4. 当前可用入口

### 4.1 推荐给现有代码使用者的入口

如果你的目标是运行仓库当前主流程，优先使用：

```bash
python main.py --config config.json --sample_dir /path/to/sample
```

这会触发和 preprocessing 直接相关的几步：

1. 对信号通道做可选增强，输出 `chX_preprocessed/`
2. 把信号通道 TIFF 栈转成 `chX.zarr`
3. 对配准通道做下采样，输出 `chY_downsample/`

### 4.2 preprocessing 独立入口

如果只想单独运行预处理，可使用：

```bash
python -m pipeline_modules.preprocessing.preprocessor --config config.json --sample_dir /path/to/sample
```

可选参数：

- `--channel 1`
  仅处理 `ch1`，覆盖 `preprocessing.channels`
- `--workers 16`
  并行进程数，默认 `CPU // 2`，Windows 上限 61
- `--no-resume`
  关闭断点续跑
- `--output_dir /path/to/output`
  自定义输出根目录

这个独立入口和 `main.py` 的差别很重要：

- `preprocessor.py` 可以按 `preprocessing.channels` 处理多个通道
- `preprocessor.py` 可以启用 `channel_subtraction`
- `main.py` 当前只会处理主配置里的 `signal` 通道，且不会走 `channel_subtraction`

## 5. 当前配置字段

下面这些字段是 preprocessing 目录下现有代码真实会读取的字段，但它们还不应被视为“最终稳定 schema”。

```json
{
  "preprocessing": {
    "channels": ["1", "2"],
    "channel_subtraction": {
      "apply": true,
      "background_channel": "ch0",
      "weight": 1.0,
      "adaptive": true,
      "save_plots": false,
      "sample_ratio": 0.005,
      "min_samples": 10,
      "max_samples": 50,
      "compression": "lzw"
    },
    "tophat": { "apply": true, "kernel_size": 21 },
    "rolling_ball": { "apply": false, "radius": 50 },
    "median_filter": { "apply": true, "kernel_size": 3 },
    "scattering_removal": { "apply": false, "sigma": 50.0, "weight": 1.0 },
    "clahe": { "apply": true, "clip_limit": 2.0, "tile_grid_size": 8 },
    "downsample": {
      "target_resolution_xyz": [25.0, 25.0, 25.0]
    },
    "zarr": {
      "chunk_size": [128, 256, 256]
    }
  }
}
```

字段说明：

- `preprocessing.channels`
  独立运行 `preprocessor.py` 时要处理的目标通道列表；可写 `"1"`、`1`、`"ch1"` 等形式，代码会归一化成 `ch1`
- `preprocessing.channel_subtraction.*`
  独立预处理入口会读取；`main.py` 当前不会使用
- `preprocessing.tophat` 到 `preprocessing.clahe`
  增强步骤定义；只有 `apply=true` 的项会进入处理链
- `preprocessing.downsample.target_resolution_xyz`
  注册通道下采样目标分辨率，供 [main.py](../../main.py) 和 [downsample.py](./downsample.py) 使用
- `preprocessing.zarr.chunk_size`
  TIFF 转 Zarr 时的 chunk 大小

## 6. 当前输入输出约定

### 6.1 输入目录

```text
sample_dir/
├── ch0/
├── ch1/
├── ch2/
└── ...
```

### 6.2 `preprocessor.py` 输出

- 无通道减法时：
  从 `sample_dir/chX/` 读取，输出到 `sample_dir/chX_preprocessed/`
- 有通道减法时：
  读取背景通道和目标通道，直接输出到 `sample_dir/chX_preprocessed/`

### 6.3 `downsample.py` 输出

默认输出目录为：

- 强度图：`<input_folder>_downsample/`
- mask：`<input_folder>_downsample_mask/`

输出内容包括：

- 下采样 TIFF 栈：`ds_0000.tiff` ...
- `volume.nii.gz`
- `original_shape.json`

### 6.4 `tiff_to_zarr.py` 输出

- 输出为一个目录型 Zarr store，例如 `sample_dir/ch1.zarr/`
- 数组路径为 `0`
- 会写基础 `multiscales` 元数据

## 7. 独立脚本用法

### 7.1 `channel_subtraction.py`

```bash
python pipeline_modules/preprocessing/channel_subtraction.py /path/to/sample --bg-channel ch0 --adaptive
python pipeline_modules/preprocessing/channel_subtraction.py /path/to/sample --weight 0.9 --workers 16
```

常用参数：

- `--bg-channel`
- `--adaptive`
- `--weight`
- `--sample-ratio`
- `--min-samples`
- `--max-samples`
- `--save-plots`
- `--compression`

### 7.2 `downsample.py`

```bash
python pipeline_modules/preprocessing/downsample.py --input_folder ./ch0 --resolution_config config.json
python pipeline_modules/preprocessing/downsample.py --input_folder ./ch0 --factor "0.5,0.5,0.5"
python pipeline_modules/preprocessing/downsample.py --input_folder ./mask --factor "0.5,0.5,0.5" --is_mask
```

实现备注：

- 手工 `--factor` 使用 `(z,y,x)` 顺序
- 配置文件中的 `resolution_xyz` / `target_resolution_xyz` 使用 `(x,y,z)` 顺序
- 当前实现对强度图和 mask 都使用最近邻插值

### 7.3 `tiff_to_zarr.py`

```bash
python pipeline_modules/preprocessing/tiff_to_zarr.py --input ./ch1_preprocessed --output ./ch1.zarr --chunk_size "128,256,256"
```

默认压缩器为 `Blosc(zstd, clevel=5)`。

### 7.4 `upsample_mask.py`

```bash
python pipeline_modules/preprocessing/upsample_mask.py \
  --input_mask_zarr ./ch1_mask_ds.zarr \
  --output_mask_zarr ./ch1_mask_fullres.zarr \
  --full_res_zarr ./ch1.zarr \
  --chunk_size "128,256,256"
```

### 7.5 单张图像处理脚本

```bash
python pipeline_modules/preprocessing/tophat_background.py input.tiff
python pipeline_modules/preprocessing/rolling_ball_background.py input.tiff --radius 50
python pipeline_modules/preprocessing/scattering_removal.py input.tiff --sigma 50 --weight 1.0
python pipeline_modules/preprocessing/median_filter.py input.tiff --kernel_size 3
python pipeline_modules/preprocessing/masked_clahe.py input.tiff --clip-limit 2.0 --grid-size 16
python pipeline_modules/preprocessing/masked_clahe.py input.tiff --masked
```

## 8. 对 agent 和后续重构的约束

在 preprocessing 完成全部 Agent-Native 改造之前，agent 使用本目录时应遵循下面这些现实约束：

- 把当前接口视为“实现现状”，不要视为最终稳定 API
- 如果只想复用增强逻辑，当前最稳定的可导入对象是 `pipeline_modules.preprocessing.preprocessor.Preprocessor`
- 如果需要 machine-readable 结果，当前代码还不会自动产出 manifest，需要调用方自行补充
- 如果需要可靠错误分类，当前代码还没有统一 `PipelineError`
- 如果需要判断样本目录状态，当前还不能依赖 `SampleLayout.status()`，需要调用方自己检查文件系统

## 9. 依赖

- `numpy`
- `opencv-python`
- `tifffile`
- `scipy`
- `tqdm`
- `scikit-learn`
- `matplotlib`
- `nibabel`
- `zarr`
- `numcodecs`

其中：

- `scikit-learn` 主要用于通道减法中的 RANSAC
- `matplotlib` 主要用于 `channel_subtraction` 的 `--save-plots`

## 10. 注意事项

- 通道减法的切片匹配依赖文件名中的 `_(C\d+)_Z(\d+)` 模式
- `preprocessor.py` 在独立运行时，如果既没传 `--channel`，又没在 `preprocessing.channels` 里配置目标通道，会直接报错退出
- 当前增强步骤顺序取决于 JSON 中字段顺序；如果顺序敏感，请显式调整配置文件中的键顺序
- `main.py` 当前不会自动复用 `channel_subtraction`
- 当前文档已经按 Agent-Native 目标重写术语，但不代表 preprocessing 代码本身已经完成这些改造
