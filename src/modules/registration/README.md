# 配准与分析流程 (Registration & Analysis Pipeline)

本文件夹包含了用于 LSFM（光片荧光显微镜）图像与 Allen Brain Atlas（Allen 脑图谱）之间双向配准以及脑区信号密度分析的工具。

## 1. 配准 (`ANTs_registration.py`)

该脚本负责处理样本图像与 Allen Brain Atlas 之间的配准。它支持两种模式：`atlas2image`（用于密度分析）和 `image2atlas`（用于热图生成）。

### 关键参数说明：
- `--sample_dir`: 样本的根目录路径。
- `--target_channel`: 您想要分析或配准的目标通道索引
- `--register_channel`: 用于配准对齐的参考通道索引
- `--atlas_image`: Allen Brain Atlas 模板图像的路径。
- `--atlas_label`: Allen Brain Atlas 标签图像的路径。
- `--mode`: 配准模式，可选 `atlas2image` 或 `image2atlas`。

### 使用场景：

#### A. 标准密度分析 (Atlas -> Image)
当您希望将图谱脑区映射到您的样本上，以计算每个脑区的信号密度时，请使用此模式。

**前提条件：**
- 存在 `chX_downsample` 文件夹（用于配准的自发光通道）。
- 存在 `chY_downsample_mask` 文件夹（用于分析的信号通道 Mask）。

**命令示例：**
```bash
python ANTs_registration.py \
  --sample_dir "S:\Path\To\Sample" \
  --target_channel "3" \
  --register_channel "5" \
  --atlas_image "S:\Path\To\Atlas\template.nii.gz" \
  --atlas_label "S:\Path\To\Atlas\annotation.nii.gz" \
  --mode "atlas2image" \
  --save_registered_image
```

**输出结果：**
1. 将 Atlas（Moving）配准到 Sample（Fixed）。
2. 将 Atlas Label 变换（Warp）到 Sample 空间。
3. 自动检测并读取 `ch3_downsample_mask`。
4. 计算每个脑区的信号密度。
5. 生成结果文件 `density_analysis_ch3.xlsx`。

#### B. 热图生成 (Image -> Atlas)
当您希望将样本的信号（Mask）映射到标准 Atlas 空间，以生成热图（例如用于多样本平均）时，请使用此模式。

**前提条件：**
- 存在 `chX_downsample` 文件夹（用于配准的自发光通道）。
- 存在 `chY_downsample_mask` 文件夹（需要被变换的信号通道 Mask）。

**命令示例：**
```bash
python ANTs_registration.py \
  --sample_dir "S:\Path\To\Sample" \
  --target_channel "3" \
  --register_channel "5" \
  --atlas_image "S:\Path\To\Atlas\template.nii.gz" \
  --atlas_label "S:\Path\To\Atlas\annotation.nii.gz" \
  --mode "image2atlas" \
  --save_registered_image
```

**输出结果：**
1. 将 Sample（Moving）配准到 Atlas（Fixed）。
2. 将 `ch3_downsample_mask` 变换（Warp）到 Atlas 空间。
3. 将变换后的 Mask 保存到 `ch3_warped_mask` 文件夹（TIFF 栈）。
4. 您随后可以使用 `utils/heatmap.py` 利用这些变换后的 Mask 生成热图。

## 2. 密度分析 (`analyze_density.py`)

通常情况下，该脚本会在 `ANTs_registration.py` 的 `atlas2image` 模式下自动被调用，但也可以单独运行。

**命令示例：**
```bash
python analyze_density.py \
  --mask_folder "S:\Sample\ch3_downsample_mask" \
  --label_folder "S:\Sample\ch3_atlas_label_downsampled" \
  --cfg "add_id_ytw.json" \
  --output "density_results.xlsx"
```

## 3. 分割 (`segmentation/`)

使用 `cellpose_segmentation.py` 生成分析所需的 Mask。

**命令示例：**
```bash
python segmentation/cellpose_segmentation.py \
  --sample_dir "S:\Path\To\Sample" \
  --channel "3" \
  --patch_size "128,256,256" \
  --batch_size 4
```
