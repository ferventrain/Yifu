import SimpleITK as sitk
import numpy as np
from PIL import Image
from pathlib import Path
import sys

def convert_tiff_to_nifti_with_spacing(
    input_path: Path, 
    output_path: Path, 
    spacing,
    origin,
    direction
):
    """
    读取TIFF图像(单个3D文件或2D切片文件夹),将其转换为NIfTI (.nii.gz)格式,
    并设置指定的物理空间信息 (Spacing, Origin, Direction)。

    Args:
        input_path (Path): 输入文件/文件夹的路径。
        output_path (Path): 输出 .nii.gz 文件的路径。
        spacing (tuple): (X, Y, Z) 轴的间距, 单位通常是毫米(mm)。
        origin (tuple): 图像原点在物理空间中的坐标。
        direction (tuple): 图像方向余弦矩阵 (一个9元素的元组)。
    """
    print(f"--- 开始处理: {input_path} ---")

    if not input_path.exists():
        print(f"错误: 输入路径不存在: {input_path}")
        return

    # 1. 读取TIFF文件并堆叠为NumPy数组
    image_slices = []
    try:
        if input_path.is_file():
            # --- 处理单个3D TIFF文件 ---
            print(f"检测到单个文件, 将其作为3D TIFF处理...")
            img = Image.open(input_path)
            for i in range(img.n_frames):
                img.seek(i)
                # 将PIL图像转换为NumPy数组
                slice_np = np.array(img)
                image_slices.append(slice_np)
            print(f"成功读取 {len(image_slices)} 个切片。")

        elif input_path.is_dir():
            # --- 处理2D切片文件夹 ---
            print(f"检测到文件夹, 将读取所有 .tif/.tiff 文件...")
            # 查找并排序所有TIFF文件, 确保Z轴顺序正确
            files = sorted(list(input_path.glob('*.tif')) + list(input_path.glob('*.tiff')))
            if not files:
                print(f"错误: 在文件夹 {input_path} 中未找到任何 .tif 或 .tiff 文件。")
                return
            
            print(f"找到 {len(files)} 个文件, 将按字母顺序读取...")
            for f in files:
                img = Image.open(f)
                slice_np = np.array(img)
                image_slices.append(slice_np)

        else:
            print(f"错误: 输入路径既不是文件也不是文件夹: {input_path}")
            return

        # 将切片列表堆叠成一个3D NumPy数组
        # 堆叠后数组的维度顺序是 (Z, Y, X)
        numpy_array = np.stack(image_slices, axis=0)
        print(f"成功创建NumPy数组, 形状为 (Z, Y, X): {numpy_array.shape}")

    except Exception as e:
        print(f"读取图像时发生错误: {e}")
        return

    # 2. 将NumPy数组转换为SimpleITK图像
    # SimpleITK默认将 (Z, Y, X) 的NumPy数组正确地解释为3D图像
    sitk_image = sitk.GetImageFromArray(numpy_array)

    # 3. 设置物理空间元数据 (Metadata)
    # 这是最关键的一步！
    # SimpleITK的SetSpacing方法需要一个 (X, Y, Z) 顺序的元组
    sitk_image.SetSpacing(spacing)
    sitk_image.SetOrigin(origin)
    sitk_image.SetDirection(direction)

    print(f"已为图像设置以下元数据:")
    print(f"  - Spacing (X, Y, Z): {sitk_image.GetSpacing()}")
    print(f"  - Origin (X, Y, Z):  {sitk_image.GetOrigin()}")
    print(f"  - Size (X, Y, Z):    {sitk_image.GetSize()}")

    # 4. 保存为NIfTI格式 (.nii.gz)
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sitk.WriteImage(sitk_image, str(output_path))
        print(f"成功将图像保存到: {output_path}")
    except Exception as e:
        print(f"保存NIfTI文件时发生错误: {e}")
    
    print("--- 处理完成 ---")


#    - 如果是单个3D TIFF文件, 写完整的文件路径, 例如: Path("data/atlas/allen_atlas.tif")
#    - 如果是包含多个2D切片的文件夹, 写文件夹的路径, 例如: Path("data/atlas_slices/")
INPUT_PATH = Path(r"S:\Yifu\Allen_brainatlas\atlas_mask.tiff") 

# 2. 设置输出路径
OUTPUT_PATH = Path(r"S:\Yifu\Allen_brainatlas\atlas_mask.nii.gz")

# 3. 设置你的Atlas的物理间距 (Spacing)
ATLAS_SPACING = (0.025, 0.025, 0.025) 

# --- 主程序入口 ---
if __name__ == "__main__":
    # 检查路径是否为示例路径
    convert_tiff_to_nifti_with_spacing(
        input_path=INPUT_PATH,
        output_path=OUTPUT_PATH,
        spacing=ATLAS_SPACING,
        origin=(0.0, 0.0, 0.0),
        direction=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    )

