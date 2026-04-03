# 图像处理预处理模块

本目录包含一系列用于光片荧光显微镜（LSFM）图像处理的预处理模块。这些模块按功能分类，可以单独使用也可以组合使用。

## 模块概览

| 模块                                                | 功能          | 适用场景                   |
| ------------------------------------------------- | ----------- | ---------------------- |
| [channel\_subtraction.py](#channel_subtractionpy) | 通道背景减法      | 去除参考通道（如自发荧光通道）的背景信号   |
| [downsample.py](#downsamplepy)                    | 3D图像降采样     | 减小数据体积，加速后续处理          |
| [masked\_clahe.py](#masked_clahepy)               | CLAHE对比度增强  | 增强图像对比度，提升细节可见性        |
| [scattering\_removal.py](#scattering_removalpy)   | 散射/光晕去除     | 去除光片散射造成的光晕背景          |
| [tophat\_background.py](#tophat_backgroundpy)     | Top-Hat背景校正 | 去除不均匀背景噪声              |
| [tiff\_to\_zarr.py](#tiff_to_zarrpy)              | TIFF转Zarr   | 将TIFF切片序列转换为OME-Zarr格式 |

***

## channel\_subtraction.py

**功能：** 从其他通道图像中并行减去背景通道（如自发荧光通道）图像，去除自身荧光背景偏移。支持固定权重和自适应权重估计两种模式。

**算法原理：**
背景通道（通常是无标记的通道C0）记录了系统的自身荧光和背景噪声。通过线性模型 `Cx = a × Cbg + b` 拟合背景像素，然后计算 `结果 = Cx - (weight × a) × Cbg`，最后裁剪负值到0。

- 使用Otsu自动识别背景像素
- 使用RANSAC鲁棒线性回归估计系数
- 支持全局自适应估计一个权重应用到所有切片

**适用场景：**

- 实验采集时包含一个自发荧光通道用于记录系统自身荧光背景
- 需要去除不同通道中共同的背景偏移信号
- 多通道3D图像栈的批量处理
- 背景强度随成像深度变化，需要自动估计最佳权重

**使用方法：**

```bash
# 基本用法（固定权重1.0）
python channel_subtraction.py /path/to/root

# 开启自适应全局权重估计（推荐）
python channel_subtraction.py /path/to/root --adaptive

# 自适应并保存拟合图表
python channel_subtraction.py /path/to/root --adaptive --save-plots

# 指定自定义背景通道名称
python channel_subtraction.py /path/to/root --bg-channel autofluorescence --adaptive

# 指定权重基数
python channel_subtraction.py /path/to/root --weight 0.9 --adaptive

# 指定并行工作进程数
python channel_subtraction.py /path/to/root --adaptive --workers 16

# 不使用压缩
python channel_subtraction.py /path/to/root --compression none
```

**目录结构要求：**

```
root/
├── ch0/                 # 参考通道文件夹（必须包含C0图像）
│   ├── *_C1_Z0001.tif
│   └── ...
├── ch1/                 # 其他通道文件夹
│   ├── *_C1_Z0001.tif
│   └── ...
└── ch2/
    └── ...
```

输出会创建 `ch1_subtracted/`, `ch2_subtracted/` 等文件夹保存结果。

**参数说明：**

- `path`: 包含通道文件夹的根目录路径（必需）
- `--bg-channel`: 背景通道文件夹名称（默认：`ch0`）
- `--workers`: 并行进程数（默认：CPU核心数/2，Windows限制最大61）
- `--compression`: TIFF压缩算法，可选 `lzw`, `none`, `zlib`（默认：lzw）
- `--weight`: 基础权重，最终权重 = `weight × 估计a`（默认：1.0）
- `--adaptive`: 开启全局自适应权重估计（默认：关闭，使用固定权重）
- `--save-plots`: 保存全局拟合图表（仅在 `--adaptive` 时有效）
- `--sample-ratio`: 用于估计的图像占总切片数的比例（默认：0.005）
- `--min-samples`: 最少采样图像数（默认：10）
- `--max-samples`: 最多采样图像数（默认：50）

**何时使用** **`--adaptive`：**

- ✅ 第一次处理该数据集，不知道最佳权重 - 一定要用
- ✅ 背景强度随Z深度变化明显 - 一定要用
- ✅ 想要自动估计最优权重 - 使用
- ❌ 已经知道最优权重，只想快速重跑 - 不需要

**文件名解析：**
从文件名如 `YF2025102901_..._C1_Z0051.tif` 中提取通道信息和Z索引，按Z索引匹配C0和对应通道。

***

## downsample.py

**功能：** 对3D TIFF图像栈或掩码进行降采样，可基于目标分辨率自动计算降采样因子。

**适用场景：**

- 原始数据分辨率过高，需要降采样用于预览或快速分割
- 多尺度分析需要不同分辨率的数据
- 减少内存占用，加速后续处理步骤

**使用方法：**

```bash
# 使用配置文件（推荐）
python downsample.py --input_folder ./raw --resolution_config config.json

# 手动指定降采样因子
python downsample.py --input_folder ./raw --factor "2,2,2"

# 统一降采样所有维度2倍
python downsample.py --input_folder ./raw --factor 2

# 处理掩码（自动使用最近邻插值）
python downsample.py --input_folder ./mask --resolution_config config.json --is_mask

# 指定输出文件夹
python downsample.py --input_folder ./raw --resolution_config config.json --output_folder ./output
```

**配置文件格式示例：**

```json
{
  "input": {
    "resolution_xyz": [0.325, 0.325, 2.0]
  },
  "preprocessing": {
    "downsample": {
      "target_resolution_xyz": [1.0, 1.0, 4.0]
    }
  }
}
```

**参数说明：**

- `--input_folder`: 输入TIFF文件夹路径（必需）
- `--resolution_config`: JSON配置文件路径（与 `--factor` 二选一）
- `--factor`: 手动指定降采样因子 `"z,y,x"` 或单个数字（如 `"2"` 表示所有维度2倍）
- `--output_folder`: 输出文件夹路径（可选，默认自动生成）
- `--is_mask`: 标记输入为掩码，使用最近邻插值（保持标签完整性）
- `--chunk_size`: 每次处理Z切片数量，控制内存使用（默认：100）

**输出：**

- 降采样后的TIFF切片序列
- `volume.nii.gz` - NIfTI格式的完整体积，方便在Neuroglancer等工具中查看
- `original_shape.json` - 原始尺寸信息

**注意事项：**

- 插值始终使用最近邻（order=0），无论是图像还是掩码，这有利于保持原始强度分布
- 分块处理Z轴可能在块边界引入轻微伪影，对于大降采样因子可忽略

***

## masked\_clahe.py

**功能：** 使用CLAHE（限制对比度自适应直方图均衡）进行对比度增强。

**适用场景：**

- 背景去除后（如Tophat后）需要增强前景对比度
- 图像动态范围低，细节不清晰
- 只增强前景区域，避免增强背景噪声

**使用方法：**

```bash
# 全局增强整张图像
python masked_clahe.py input.tiff

# 只增强前景区域（使用Otsu自动找mask）
python masked_clahe.py input.tiff --masked

# 调整参数
python masked_clahe.py input.tiff --masked --clip-limit 3.0 --grid-size 8
```

**参数说明：**

- `image_path`: 输入图像路径（必需）
- `--masked`: 是否只增强前景，使用Otsu自动分割前景（默认：False，增强整张图）
- `--clip-limit`: CLAHE剪切限制，值越大对比度越强（默认：2.0）
- `--grid-size`: CLAHE网格大小，越小局部适应性越强（默认：16）

**输出：**

- `*_clahe.tiff` - 全局增强结果
- `*_masked_clahe.tiff` - 掩码增强结果（使用 `--masked` 时）
- `*_otsu_mask.tiff` - Otsu分割得到的前景掩码（使用 `--masked` 时）

**何时使用** **`--masked`：**

- ✅ 图像大部分区域是黑色背景，只有小部分是前景 - 使用 `--masked`
- ✅ 不想让背景也被增强 - 使用 `--masked`
- ❌ 已经做过去背景，整个图像都是感兴趣区域 - 不需要

***

## scattering\_removal.py

**功能：** 基于高斯背景估计去除光片散射造成的光晕背景。

**适用场景：**

- Top-Hat变换后仍残留散射光晕
- 厚样品中光片散射造成不均匀背景
- 需要进一步压低背景，提升前景对比度

**使用方法：**

```bash
# 默认参数
python scattering_removal.py input_tophat.tiff

# 调整sigma和weight
python scattering_removal.py input_tophat.tiff --sigma 100 --weight 0.8
```

**算法思路：**

1. 用大sigma高斯模糊估计图像的背景（包含散射光晕）
2. `结果 = 原图 - weight × 背景`
3. 裁剪负值到0，保持数据类型不变

**参数说明：**

- `image_path`: 输入图像路径（必需，通常是Tophat处理后的图像）
- `--sigma`: 高斯sigma，越大背景越平滑（默认：50.0）。建议设置为感兴趣结构直径的2-3倍。
- `--weight`: 背景减法权重（默认：1.0）。如果背景去除过度可以降低到0.8-0.9。

**参数调优建议：**

- 如果光晕还是很明显 → 增大sigma或增大weight
- 如果前景被过度去除变暗 → 减小weight

**输出：** `*_scatter_removed.tiff`

***

## tophat\_background.py

**功能：** 使用形态学Top-Hat变换去除不均匀背景噪声。

**适用场景：**

- 光照不均匀造成的缓慢变化背景
- 大尺度背景偏移，需要保留小尺度特征
- 作为其他增强步骤（CLAHE、散射去除）的前置处理

**使用方法：**

```bash
# 基本用法（kernel_size=21）
python tophat_background.py input.tiff

# 在代码中调用修改kernel_size
from tophat_background import tophat_background_correction
tophat_background_correction("input.tiff", kernel_size=31)
```

**参数说明：**

- `image_path`: 输入图像路径（必需）
- `kernel_size`: 结构元素大小（代码中可配置，命令行默认21）

**参数选择：**

- kernel\_size应大于你想要保留的最大目标尺寸
- 目标越大，kernel\_size应该越大
- 常用值：15, 21, 31

**输出：** `*_tophat.tiff`

**推荐工作流：**

```
原始图像 → Top-Hat背景校正 → 散射去除 → CLAHE增强
```

***

## tiff\_to\_zarr.py

**功能：** 将TIFF切片文件夹转换为OME-Zarr格式，方便大数据集处理和可视化。

**适用场景：**

- 大型3D数据集需要高效存储和随机访问
- 准备数据以供Neuroglancer、napari等工具可视化
- 后续处理需要分块访问数据

**使用方法：**

```bash
# 默认块大小 (128, 256, 256)
python tiff_to_zarr.py --input ./tiff_folder --output ./volume.zarr

# 自定义块大小
python tiff_to_zarr.py --input ./tiff_folder --output ./volume.zarr --chunk_size "64,512,512"
```

**参数说明：**

- `--input`: 输入TIFF文件夹路径（必需）
- `--output`: 输出 `.zarr` 路径（必需）
- `--chunk_size`: Zarr分块大小 `z,y,x`（默认：`128,256,256`）

**输出：**

- 符合OME-Zarr v0.4格式的Zarr存储
- 使用Blosc/zstd压缩

**特点：**

- 分块处理，内存占用可控
- 支持非常大的数据集
- 输出可直接在napari中打开

***

## 推荐处理流程

根据不同的应用场景，推荐以下预处理流程：

### 1. 多通道背景去除 + 增强

```
原始采集数据
  └─→ channel_subtraction (减去C0背景)
       └─→ tophat_background (去除不均匀背景)
            └─→ scattering_removal (去除散射光晕)
                 └─→ masked_clahe (对比度增强)
                      └─→ downsample (可选，降采样用于分割/预览)
                           └─→ tiff_to_zarr (可选，转Zarr可视化)
```

### 2. 单通道数据仅背景校正

```
原始图像
  └─→ tophat_background
       └─→ scattering_removal
            └─→ clahe
```

### 3. 仅格式转换

```
TIFF切片序列
  └─→ tiff_to_zarr → Zarr for visualization
```

***

## 依赖

需要以下Python包：

```
numpy
opencv-python
tifffile
tqdm
scipy
scikit-learn  # 用于RANSAC鲁棒回归
matplotlib    # 用于绘图（--save-plots时需要）
nibabel
zarr
numcodecs
```

***

## 注意事项

1. **文件名格式**: `channel_subtraction.py` 期望文件名包含 `_Cx_Zyyyy` 格式的信息
2. **内存管理**: `downsample.py` 使用分块处理大体积数据，可通过 `chunk_size` 调整内存使用
3. **插值方法**: `downsample.py` 始终使用最近邻插值，这对于分割掩码是正确的选择，对于强度图像也能保持原始分布
4. **参数调优**: 散射去除的 `sigma` 参数建议根据目标结构大小调整，一般为目标直径的2-3倍

