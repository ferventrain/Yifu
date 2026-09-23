# cFos 标准流程

本文档描述 cFos 样本的标准处理流程。以下四条是标准约定，改动需经确认：

1. **分割块为 256³**
2. **先全脑 normalize，再逐块分割**
3. **统计前过滤：距脑表面 40 px 以内的 object 不算**
4. **单层 ≥200 voxel 的 object 滤掉**

## 流程总览

```
IMS (.ims)
  ├─ ims_to_zarr --channels {signal}            → ch{X}.zarr            (信号通道, 原生分辨率)
  └─ ims_to_zarr --reg_channel 0                → ch0_downsampled.zarr  (配准通道, IMS 粗层 L2)
main.py --config config_cfos_*.json --sample_dir <dir>
  ├─ 1  配准下采样: ch0_downsampled.zarr → ch0_downsample/volume.nii.gz   (秒级)
  ├─ 2  ANTs 配准 (atlas2image) → upsampled_atlas_label.zarr + transforms/
  ├─ 3  信号预处理 / Zarr
  ├─ 4  cFos U-Net 分割 + 标准 mask 过滤
  └─ 5  脑区信号统计 → density_results_ch{X}.xlsx
```

## 标准点详解

### 1. 分割块 256³

- `preprocessing.zarr.chunk_size = [256, 256, 256]`：zarr 物理分块。
- `segmentation.cfos_unet.patch_size = [256, 256, 256]`：网络滑窗 patch。
- 推理按 zarr 物理 chunk 逐块进行（`cfos_unet_inference.py`），chunk 与 patch 对齐。

### 2. 先全脑 normalize，再逐块分割

- **方法**（`cfos_unet_model.normalize_volume`）：百分位裁剪拉伸——以 [P1, P99.5]
  为上下界，先 clip 再线性拉伸到 [0, 1]（`normalize_percentiles: [1.0, 99.5]`）。
- **范围**（`normalize_scope: "global"`，默认）：分割前先对整个脑做一遍流式
  扫描（uint16 用精确直方图，逐块累加 65536-bin 计数），求出**全脑统一**的
  P1/P99.5 值；然后每个 256³ 块用同一组界值做裁剪拉伸。
  这保证各块的亮度尺度一致（逐块独立 normalize 会在块边界造成概率不一致）。
- 旧行为可用 `"normalize_scope": "chunk"` 恢复。
- 空块跳过逻辑（`skip_below_threshold`）不影响全局界值计算。

### 3+4. 标准 mask 过滤（统计前，`cfos_mask_postprocess`）

分割输出先经过标准过滤才成为统计用的正典 mask（`ch{X}_mask.zarr`）：

- **脑缘排除**（`exclude_edge_px: 40`）：以配准到样本空间的 atlas label
  （`upsampled_atlas_label.zarr`，label > 0）的脑表面为基准，向内 40 px 的
  壳层内出现过的 object **整个删除**。实现在 4× 降采样网格上：腐蚀
  k = ceil(40/4) = 10 体素，object 与壳层有任何重叠即删除。
- **单层大面积排除**（`max_single_slice_voxels: 200`）：object 在任一单层
  （z 切片）上的足迹 ≥ 200 个原生 voxel（血管断面、表面大片假阳性）→ 删除。
  4× 网格上阈值为 200/4² = 12.5 ds-voxel。
- 其余过滤器默认关闭（`max_voxels/min_voxels/max_extent_ratio = 0`）。

配置块（`segmentation.postprocess`）：

```json
"postprocess": {
  "enabled": true,
  "exclude_edge_px": 40,
  "max_single_slice_voxels": 200,
  "max_voxels": 0,
  "min_voxels": 0,
  "max_extent_ratio": 0.0,
  "downsample_factor": 4
}
```

产物链：`ch{X}_mask_raw.zarr`（U-Net 原始输出）→ `ch{X}_mask.zarr`（标准过滤后，
统计与导出的唯一输入）。下游统计（`region_signal_analysis_zarr_graph`）无需感知。

## 运行

```bash
# 1) IMS 直读 Zarr（交互式会询问是否提取 ch0 粗层；无人值守用 --reg_channel 0）
python -m pipeline_modules.preprocessing.ims_to_zarr --input <sample>.ims --output <sample_dir>/ch1.zarr --channels 1 --reg_channel 0

# 2) 主流程
python main.py --config config/config_cfos_YF2026060302_M2.json --sample_dir <sample_dir>
```

## 验收要点

- 全局 normalize：日志输出 `Global normalize bounds: low=… high=… (exact histogram over full volume)`；
  结果 JSON 含 `normalize_scope: "global"` 与 `normalize_bounds`。
- 标准过滤：postprocess 输出 `labels_removed_edge_3d`、`labels_removed_single_slice_3d`
  计数；`ch{X}_mask.zarr` 为过滤后结果。
