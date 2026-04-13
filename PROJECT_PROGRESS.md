# LSFM 项目进度

更新时间：2026-04-09

## 1. 当前进度总览

本项目主流程已具备从原始 TIFF 数据到脑区统计结果（Excel）的可运行链路，核心模块已打通。当前阶段重点是完善扩展能力、统一主流程接入和补充测试/文档。

## 2. 已完成模块

### 2.1 主流程编排

- [x] `main.py` 统一调度预处理、分割、配准、分析
- [x] 支持跳步参数：`--skip_preprocessing` / `--skip_segmentation` / `--skip_registration` / `--skip_analysis`
- [x] 支持已有中间结果自动复用（避免重复计算）

### 2.2 配置体系

- [x] `config.json` 可直接驱动全流程
- [x] `config_template.json` 提供配置模板
- [x] 主流程关键配置项已稳定（input / preprocessing / segmentation / registration / analysis）

### 2.3 预处理模块（`pipeline_modules/preprocessing`）

- [x] 可配置增强预处理（Top-hat / Rolling-ball / Scattering removal / Median filter / CLAHE）
- [x] TIFF -> Zarr 转换（`tiff_to_zarr.py`）
- [x] 配准通道下采样并导出 `volume.nii.gz`（`downsample.py`）
- [x] Mask 上采样工具（`upsample_mask.py`）
- [x] 通道减法工具（`channel_subtraction.py`）

### 2.4 分割模块（`pipeline_modules/segmentation`）

- [x] 阈值分割（`intensity_threshold_segmentor.py`）
- [x] 分割结果导出 TIFF（主流程内调用）
- [x] 单图测试与性能测试脚本

### 2.5 配准模块（`pipeline_modules/registration`）

- [x] ANTs 双向配准能力（`atlas2image` / `image2atlas`）
- [x] 支持保存变换和配准结果
- [x] 支持 atlas label 上采样并导出回样本空间 TIFF

### 2.6 脑区分析模块（`pipeline_modules/registration`）

- [x] Zarr-native block graph 脑区统计（`region_signal_analysis_zarr_graph.py`）
- [x] 脑区统计结果导出为 Excel（含层级脑区统计）
- [x] 覆盖检查工具（`check_region_coverage_zarr.py`）

### 2.7 可视化与工具模块

- [x] 3D 热图生成工具（`pipeline_modules/visualization/heatmap.py`）
- [x] 通道整理、ROI 清零、体积统计、格式转换等工具（`pipeline_modules/utils`）

### 2.8 工程化基础

- [x] `environment.yml` 环境定义
- [x] `Dockerfile` 容器化基础
- [x] 硬件需求文档（`hardware_requirements.md`）

## 3. 待完成模块（留空待补充）

> 下面各项预留为你后续填写，可按“目标 / 当前状态 / 计划完成时间 / 负责人”补充。

### 3.1 主流程能力扩展

- [ ] 模块名称：
  目标：
  当前状态：
  计划完成时间：
  负责人：

### 3.2 预处理能力扩展

- [ ] 模块名称：非特异性信号去除
  目标：去除脑样本中不属于特异性信号的高亮信号，用深度学习方法
  当前状态：未开始
  计划完成时间：6月之前
  负责人：余子涵，肖萌

### 3.3 分割能力扩展

- [ ] 模块名称：深度学习分割
  目标：用深度学习模型全自动分割图像(仅cfos信号)
  当前状态：模型训练中
  计划完成时间：5月之前
  负责人：余子涵，王钦

### 3.4 分析能力扩展

- [ ] 模块名称：血管分析
  目标：能够分析大型图像的血管，重建血管网络，输出血管参数
  当前状态：确定方法：tubemap/kimimaro
  计划完成时间：6月之前
  负责人：余子涵

### 3.5 分析与可视化扩展

- [ ] 模块名称：
  目标：
  当前状态：
  计划完成时间：
  负责人：

### 3.6 测试与质量保障

- [ ] 模块名称：
  目标：
  当前状态：
  计划完成时间：
  负责人：

### 3.7 文档与交付规范

- [ ] 模块名称：
  目标：
  当前状态：
  计划完成时间：
  负责人：

