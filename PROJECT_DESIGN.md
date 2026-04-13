# LSFM 数据处理流程 - 项目设计文档

更新时间：2026-04-09

## 1. 项目目标

本项目面向 TB 级光片荧光显微镜（LSFM）数据处理，目标是建立一套可配置、可扩展、适合大体积数据的全流程分析管线。当前仓库已经具备从原始 TIFF 数据到脑区统计结果的主流程能力，并保留了若干可独立调用的扩展脚本。

当前主流程由 `main.py` 统一调度，核心步骤为：

1. 配置读取
2. 预处理增强
3. TIFF 转 Zarr
4. 配准通道下采样
5. 分割
6. ANTs 配准
7. 脑区统计分析

## 2. 当前设计原则

### 2.1 配置驱动

主流程使用 JSON 配置文件驱动，默认读取根目录下的 `config.json`，也可通过命令行参数指定其他配置文件。当前代码中实际使用的是 JSON，不是 YAML。

### 2.2 模块化拆分

项目将能力拆分为以下目录：

- `pipeline_modules/preprocessing`：预处理、格式转换、下采样
- `pipeline_modules/segmentation`：细胞/信号分割与可视化
- `pipeline_modules/registration`：配准、脑区映射、区域统计
- `pipeline_modules/visualization`：热图生成
- `pipeline_modules/utils`：辅助工具
- `data/reference`：参考图谱数据

### 2.3 面向大体积数据

当前实现采用 TIFF、Zarr、NIfTI 混合存储策略：

- 原始数据通常以 TIFF 序列输入
- 分割与统计分析阶段优先使用 Zarr
- 配准阶段使用下采样后的 NIfTI 体数据

这意味着“统一使用 Zarr”是设计方向，但当前落地版本仍是多格式协同。

### 2.4 可跳步与复用中间结果

`main.py` 支持通过命令行参数跳过预处理、配准、分割、分析步骤；同时也会检测已有的 `mask`、`zarr`、配准输出，尽量避免重复计算。

## 3. 当前仓库结构与职责

### 3.1 根目录

- `main.py`：主流程入口
- `config.json`：默认配置
- `config_template.json`：配置模板
- `environment.yml`：Conda 环境定义
- `Dockerfile`：Docker 构建文件
- `hardware_requirements.md`：资源需求说明

### 3.2 参考数据

当前图谱数据位于 `data/reference`：

- `atlas.nii.gz`
- `atlas_label.nii.gz`
- `atlas.tiff`
- `atlas_label.tiff`

## 4. 输入与输出定义

### 4.1 输入

主流程默认要求样本目录至少包含以下通道文件夹：

```text
sample_dir/
├── ch0/
├── ch1/
└── ...
```

其中：

- `signal` 通道：用于分割和分析
- `registration` 通道：用于与参考图谱配准

### 4.2 当前实际使用的关键配置项

当前主流程直接依赖以下字段：

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
    "downsample": {
      "target_resolution_xyz": [25.0, 25.0, 25.0]
    },
    "zarr": {
      "chunk_size": [128, 256, 256]
    }
  },
  "registration": {
    "atlas_path": "path/to/atlas.nii.gz",
    "annotation_path": "path/to/atlas_label.nii.gz"
  },
  "segmentation": {
    "method": "cellpose or threshold"
  },
  "analysis": {
    "density_config": "pipeline_modules/registration/Region_Csv_Rev1_updated.CSV"
  }
}
```

### 4.3 典型输出

以 `signal=1`、`registration=0` 为例，主流程会生成或复用以下结果：

- `ch1_preprocessed/`：预处理增强后的信号通道 TIFF
- `ch1.zarr/`：信号通道 Zarr
- `ch1_mask.zarr/`：分割结果 Zarr
- `ch1_mask/`：导出的 mask TIFF 序列
- `ch0_downsample/`：配准通道下采样结果和 `volume.nii.gz`
- `upsampled_atlas_label/`：上采样回原始尺寸后的图谱标签 TIFF
- `upsampled_atlas_label.zarr/`：图谱标签的 Zarr 版本
- `density_results_ch1.xlsx`：脑区统计结果
- `transforms/` 或其他配准输出目录：ANTs 变换结果（当启用保存时）

## 5. 核心模块设计

### 5.1 总控模块

入口文件：`main.py`

当前职责：

- 读取配置文件和命令行参数
- 根据 `signal`、`registration` 通道自动拼接输入输出路径
- 调用预处理、分割、配准、分析脚本
- 支持 `--skip_preprocessing`、`--skip_registration`、`--skip_segmentation`、`--skip_analysis`
- 支持 `--test` 测试模式

当前主流程固定使用 `atlas2image` 作为默认配准方向，用于后续脑区统计。

### 5.2 预处理模块

目录：`pipeline_modules/preprocessing`

#### 已实现能力

- `preprocessor.py`
  支持按配置组合多步图像增强，当前代码中已实现：
  - Top-hat 背景校正
  - Rolling-ball 背景校正
  - 散射去除
  - 中值滤波
  - CLAHE 增强
- `tiff_to_zarr.py`
  将 TIFF 序列转换为 OME-Zarr 风格数据
- `downsample.py`
  按配置或手动因子下采样 TIFF 栈，并输出 `volume.nii.gz`
- `upsample_mask.py`
  用于将低分辨率 mask 上采样回全分辨率
- `channel_subtraction.py`
  支持背景通道减法和自适应权重估计

#### 当前接入状态

- `main.py` 已接入：增强预处理、TIFF 转 Zarr、配准通道下采样
- 已实现但未在 `main.py` 主流程中自动接入：通道减法、按多通道批处理的预处理入口

### 5.3 分割模块

目录：`pipeline_modules/segmentation`

#### 已实现能力

- `cellpose_distributed.py`
  基于 `cellpose.contrib.distributed_segmentation` 的分布式 Cellpose 分割
- `intensity_threshold_segmentor.py`
  基于阈值与连通域分析的 Zarr 分割
- `visualize_spots.py`
  根据坐标结果生成可视化 mask
- `test_single_image.py`
  单张图像的 Cellpose 快速验证
- `test_cellpose_time.py`
  Cellpose 性能测试脚本

#### 当前接入状态

`main.py` 当前支持两种分割方法：

- `cellpose`
- `threshold`

分割结果统一输出为 `mask.zarr`，必要时再导出为 TIFF 序列供后续步骤复用。

#### 当前边界

旧设计文档中提到的 Spotiflow、StarDist、`base_segmentor.py` 等内容，目前仓库中没有形成统一可切换的主流程实现，不应视为当前已落地能力。

### 5.4 配准模块

目录：`pipeline_modules/registration`

#### 已实现能力

- `ANTs_registration.py`
  基于 ANTs 的双向配准脚本，支持：
  - `atlas2image`
  - `image2atlas`
  - `Rigid`
  - `Affine`
  - `SyN`
  - `SyNRA`
- 支持保存配准后的图像、标签以及变换文件
- 支持将 atlas label 上采样回原始样本分辨率并导出 TIFF

#### 当前接入状态

- `main.py` 已接入 `atlas2image` 路径
- `image2atlas` 已实现，但当前主要作为独立脚本模式使用

### 5.5 分析模块

目录同样位于 `pipeline_modules/registration`

#### 已实现能力

- `region_signal_analysis_zarr_graph.py`
  基于 Zarr 的 block-graph 脑区统计分析
- `region_signal_analysis.py`
  较早版本的 TIFF/体数据统计脚本
- `check_region_coverage_zarr.py`
  脑区覆盖检查工具

#### 当前输出指标

当前 Zarr 统计脚本会生成按脑区层级组织的 Excel，核心字段包括：

- `Total Voxels`
- `Signal Voxels`
- `Voxel Density`
- `Signal Count`
- `Signal Sum`
- `Signal Mean`
- `Signal Density`

### 5.6 可视化模块

目录：`pipeline_modules/visualization`

#### 已实现能力

- `heatmap.py`
  可对 mask 做下采样、应用配准变换，并在 atlas 空间生成 3D 热图

#### 当前接入状态

热图生成功能已实现，但当前未接入 `main.py` 自动流程，属于独立调用工具。

### 5.7 工具模块

目录：`pipeline_modules/utils`

当前已实现的辅助工具包括：

- `channel_organizer.py`：通道整理
- `convert_niigz.py`：格式转换
- `count_mask_pixel.py`：mask 像素统计
- `ims_to_tiff.py`：IMS 转 TIFF
- `volume_calculator.py`：体积计算
- `zero_roi_pixels.py`：ROI 清零

## 6. 当前主流程

根据 `main.py` 的实际逻辑，当前主流程顺序如下：

1. 读取配置文件并解析通道映射
2. 对 `signal` 通道执行可配置增强预处理
3. 将预处理后的 `signal` 通道 TIFF 转换为 Zarr
4. 对 `registration` 通道执行下采样，并生成 `volume.nii.gz`
5. 对 `signal` 通道执行分割，输出 `mask.zarr`
6. 将 `mask.zarr` 导出为 TIFF 序列
7. 执行 `atlas2image` 配准，将 atlas label 映射到样本空间
8. 将上采样后的图谱标签再次转换为 Zarr
9. 基于 `mask.zarr`、`label.zarr`、`signal.zarr` 生成脑区统计 Excel

## 7. 部署与运行环境

### 7.1 环境定义

当前仓库已提供：

- `environment.yml`
- `Dockerfile`

主要依赖包括：

- Python 3.8
- numpy / scipy / pandas / openpyxl
- tifffile / zarr / dask / dask-image
- antspyx
- cellpose
- torch / torchvision
- opencv-python-headless
- nibabel

### 7.2 硬件建议

根据 `hardware_requirements.md` 当前记录：

- CPU：32 vCPU 以上
- 内存：128 GB 以上
- GPU：推荐 8 张 NVIDIA A100
- 单卡显存：建议 40 GB 以上

## 8. 当前版本边界说明

为了让设计文档与仓库现状一致，需要明确以下边界：

1. 当前主流程已经打通，但不是所有已实现脚本都已接入 `main.py`
2. 当前正式使用的配置格式是 JSON
3. 当前主流程中默认执行的是 `atlas2image` 配准路径
4. 通道减法、热图生成、`image2atlas` 配准等能力已经实现，但更适合作为独立工具使用
5. 旧文档中提到但仓库中未落地的模块，不再作为当前版本能力描述

## 9. 后续文档维护原则

后续如果继续推进项目，建议统一按以下原则维护文档：

- 设计文档只写“仓库当前真实存在且可运行的能力”
- 规划中的功能单独放到进度文档或路线图中
- 每新增一个模块，明确它是“已实现未接入主流程”还是“已接入主流程”
- 当配置项扩展时，同步更新 `config_template.json` 和文档示例

