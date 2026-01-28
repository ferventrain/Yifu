# LSFM 数据处理流程 - 项目设计文档

## 1. 项目概述

本项目旨在构建一个高吞吐、模块化、松耦合的自动化处理流程，专门用于处理 TB 级光片荧光显微镜 (LSFM) 数据（单通道可达 500GB+）。

**核心设计：**
*   **IO与算法解耦**：所有核心算法（分割、配准、分析）均不直接读取原始 TIFF 文件，而是统一基于 **OME-Zarr** 格式进行操作。
*   **算法/模型可插拔**：分割模型（Cellpose, Spotiflow 等）和配准算法（ANTs, Elastix 等）作为独立模块，可根据 Config 配置灵活替换，无需修改主流程代码。
*   **配置驱动**：所有处理参数由统一的配置文件控制，实现流程的可复现性。

## 2. 输入与输出定义

### 2.1 输入 (Input)
1.  **原始图像数据**:
    *   **格式**: TIFF 图像栈 (TIFF Stack) 或图像序列。
    *   **规模**: 典型单通道数据量约 500GB。
    *   **结构**: 多通道数据，通常按文件夹组织。
2.  **配置文件 (Config)**:
    *   包含实验元数据与处理参数的 JSON/YAML 文件。
    *   **必需字段**:
        *   `resolution`: 图像物理分辨率 (x, y, z 微米/像素)。
        *   `channels`: 通道数量及每个通道的类型（如：细胞核通道、血管通道）。
        *   `model`: 指定使用的分割模型 (e.g., `cyto3`, `nuclei`, `spotiflow`)。
        *   `sample_id`: 样本唯一编号。
        *   `group_name`: 课题组名称。

### 2.2 输出 (Output)
1.  **定量统计报表**:
    *   **格式**: Excel (.xlsx) 或 CSV。
    *   **内容**: 全脑各脑区（基于 Allen Brain Atlas）的细胞计数、密度、体积统计。
2.  **可视化图表**:
    *   **格式**: TIFF (3D Volume), PNG (Projections)。
    *   **内容**:
        *   全脑细胞密度热图 (3D Heatmap)。
        *   分割结果覆盖图 (Segmentation Mask Overlay)。
        *   配准质量评估图。

## 3. 核心模块设计

### 3.1 预处理模块 (`/preprocessing`)
*   **输入**: 原始 TIFF 图像栈 (高分辨率)。
*   **功能**:
    *   将 TIFF 转换为 Zarr 格式 (分块存储，支持高效并行读取)。
    *   生成降采样金字塔层级，用于可视化和配准。
*   **关键脚本**:
    *   `tiff_to_zarr.py`: 将 TIFF 文件夹转换为 `.zarr`。
    *   `downsample.py`: 生成低分辨率体积数据 (例如用于配准)。

### 3.2 分割模块 (`/segmentation`)
*   **输入**: Zarr 数组 (或 TIFF 栈)。
*   **功能**:
    *   在 3D 空间中检测并分割细胞/细胞核。
    *   **可插拔架构**: 支持通过 Config 切换不同的分割核心 (如 Cellpose, Spotiflow, StarDist)。
    *   支持分布式处理 (基于 Dask)，可处理超大体积数据。
    *   检测结果的可视化 (Mask 或 标记点)。
*   **关键脚本**:
    *   `cellpose_distributed.py`: 基于 Dask + Cellpose 的分布式分割程序。
    *   `test_single_image.py`: 单张2D图像测试工具。
    *   `visualize_spots.py`: Spotiflow 结果可视化工具，根据坐标 CSV 生成 Mask 图像。
    *   `base_segmentor.py`: 分割器基类定义。

### 3.3 配准模块 (`/registration`)
*   **输入**: 降采样后的全脑数据 + 参考图谱 (Allen Brain Atlas)。
*   **功能**:
    *   **图谱 -> 图像配准**: 将标准图谱标签变形映射到样本脑空间。
    *   **图像 -> 图谱配准**: 将样本脑变形映射到标准空间。
    *   使用 ANTs (Advanced Normalization Tools) 进行刚体、仿射和 SyN 非线性形变。
*   **关键脚本**:
    *   `ANTs_registration.py`: 主配准流程脚本。
    *   `analyze_density.py`: 脑区密度定量分析工具。

### 3.4 工具模块 (`/utils`)
*   **关键脚本**:
    *   `channel_organizer.py`: 通道组织工具，用于将多通道图像转换为单通道图像。
    *   `convert_niigz.py`: 格式转换工具。
    *   `count_mask_pixel.py` / `volume_calculator.py`: 像素统计与体积计算。
    *   `zero_roi_pixels.py`: ROI 区域处理。

### 3.5 可视化模块 (`/visualization`)
*   **关键脚本**:
    *   `heatmap.py`: 生成 3D 热图，可视化配准后的细胞密度或信号强度。

### 3.6 其他模块
*   `Allen_brainatlas/`: 存放标准图谱文件 (atlas, labels, mask)。
*   `Arivis_registration/`: Arivis 软件相关的配准脚本与区域定义。
*   `SOTA_test/`: 存放其他 SOTA 模型 (如 Spotiflow) 的测试结果与对比。

## 4. 部署架构

### 4.1 Docker 环境
*   **基础镜像**: `mambaorg/micromamba`
*   **依赖库**:
    *   **核心**: Python 3.8+, Numpy, Scipy, Pandas
    *   **图像处理**: Tifffile, Zarr, Dask, Dask-Image
    *   **AI/GPU**: PyTorch, Cellpose (支持 CUDA)
    *   **配准**: ANTsPy
*   **容器特性**:
    *   支持 NVIDIA GPU 直通。
    *   支持数据卷挂载。

## 5. 工作流示例
1.  **格式转换**: `Raw TIFF` -> `Input.zarr` (利用 Dask 转换 TB 级数据)
2.  **分割**: `Input.zarr` -> `Mask.zarr` (基于 Zarr 分块并行，算法可插拔)
3.  **降采样**: `Input.zarr` -> `Downsampled.nii.gz`
4.  **配准**: `Downsampled.nii.gz` + `Allen Atlas` -> `Warped_Labels.nii.gz`
5.  **上采样**: `Warped_Labels.nii.gz` -> `Full_Res_Labels.zarr`
6.  **定量分析**: 叠加 `Mask.zarr` 与 `Full_Res_Labels.zarr` -> 生成统计报表与可视化热图

