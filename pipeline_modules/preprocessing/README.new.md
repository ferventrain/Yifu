# LSFM 预处理模块（preprocessing）

本目录提供两类能力：

1. 配置驱动的批处理预处理（推荐，`preprocessor.py`）
2. 独立脚本工具（单步处理、格式转换、上下采样）

## 目录说明

- `preprocessor.py`：按 `config.json` 批量执行预处理（可选通道减法）
- `channel_subtraction.py`：独立的通道背景减法工具
- `tophat_background.py`：Top-hat 背景校正（单张）
- `rolling_ball_background.py`：Rolling-ball 背景校正（单张）
- `scattering_removal.py`：散射/光晕去除（单张）
- `median_filter.py`：中值滤波（单张）
- `masked_clahe.py`：CLAHE 对比度增强（单张）
- `downsample.py`：3D TIFF 序列下采样并导出 NIfTI
- `tiff_to_zarr.py`：TIFF 序列转 Zarr
- `upsample_mask.py`：低分辨率 mask 上采样到全分辨率

## 推荐主入口：preprocessor.py

### 命令

```bash
python -m pipeline_modules.preprocessing.preprocessor --config config.json --sample_dir /path/to/sample
```

可选参数：

- `--channel 1`：仅处理 `ch1`（覆盖配置）
- `--workers 16`：并行进程数（默认 `CPU//2`，Windows 上限 61）
- `--no-resume`：关闭断点续跑
- `--output_dir /path/to/output`：自定义输出根目录

### 输入目录约定

```text
sample_dir/
├── ch0/
├── ch1/
├── ch2/
└── ...
```

### config 中 preprocessing 的关键字段

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
    "clahe": { "apply": true, "clip_limit": 2.0, "tile_grid_size": 8 }
  }
}
```

说明：

- 启用 `channel_subtraction.apply=true` 时，流程为“通道减法 + 增强步骤”，输出到 `chX_preprocessed/`
- 未启用通道减法时，仅执行增强步骤
- 实际支持的增强步骤为：`tophat`、`rolling_ball`、`median_filter`、`scattering_removal`、`clahe`

## 独立脚本用法

### 1) channel_subtraction.py

```bash
python pipeline_modules/preprocessing/channel_subtraction.py /path/to/sample --bg-channel ch0 --adaptive
python pipeline_modules/preprocessing/channel_subtraction.py /path/to/sample --weight 0.9 --workers 16
```

主要参数：`--bg-channel`、`--adaptive`、`--weight`、`--sample-ratio`、`--min-samples`、`--max-samples`、`--save-plots`、`--compression`

### 2) downsample.py

```bash
python pipeline_modules/preprocessing/downsample.py --input_folder ./ch0 --resolution_config config.json
python pipeline_modules/preprocessing/downsample.py --input_folder ./ch0 --factor "0.5,0.5,0.5"
python pipeline_modules/preprocessing/downsample.py --input_folder ./mask --factor "0.5,0.5,0.5" --is_mask
```

输出：

- 下采样后的 TIFF 序列（`ds_0000.tiff`...）
- `volume.nii.gz`
- `original_shape.json`

### 3) tiff_to_zarr.py

```bash
python pipeline_modules/preprocessing/tiff_to_zarr.py --input ./ch1_preprocessed --output ./ch1.zarr --chunk_size "128,256,256"
```

默认压缩器为 `Blosc(zstd, clevel=5)`，写入 OME-Zarr 基础 `multiscales` 元数据。

### 4) upsample_mask.py

```bash
python pipeline_modules/preprocessing/upsample_mask.py \
  --input_mask_zarr ./ch1_mask_ds.zarr \
  --output_mask_zarr ./ch1_mask_fullres.zarr \
  --full_res_zarr ./ch1.zarr \
  --chunk_size "128,256,256"
```

### 5) 单张图像处理脚本

```bash
python pipeline_modules/preprocessing/tophat_background.py input.tiff
python pipeline_modules/preprocessing/rolling_ball_background.py input.tiff --radius 50
python pipeline_modules/preprocessing/scattering_removal.py input.tiff --sigma 50 --weight 1.0
python pipeline_modules/preprocessing/median_filter.py input.tiff --kernel_size 3
python pipeline_modules/preprocessing/masked_clahe.py input.tiff --clip-limit 2.0 --grid-size 16
python pipeline_modules/preprocessing/masked_clahe.py input.tiff --masked
```

## 依赖

- `numpy`
- `opencv-python`
- `tifffile`
- `scipy`
- `tqdm`
- `scikit-learn`（通道减法中 RANSAC）
- `matplotlib`（`--save-plots` 时）
- `nibabel`
- `zarr`
- `numcodecs`

## 注意事项

- 文件名解析依赖模式 `*_C1_Z0001.tif`（通道减法按 `Z` 匹配切片）
- 下采样统一使用最近邻插值（强度图和 mask 都是）
- Windows 多进程默认上限 61，已在代码中做限制
- `config_template.json` 里的字段可能滞后，建议以本 README 和源码行为为准
