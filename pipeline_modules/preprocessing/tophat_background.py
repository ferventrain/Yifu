import cv2
import numpy as np
import sys
import os
import tifffile


def tophat_background_correction(image_path: str, kernel_size: int = 15) -> None:
    """
    使用Top-Hat变换去除背景不均匀噪音
    
    Parameters:
        image_path: 输入图像路径
        kernel_size: 结构元素大小，控制背景估计的尺度
    """
    if image_path.lower().endswith(('.tif', '.tiff')):
        img = tifffile.imread(image_path)
    else:
        img = cv2.imread(image_path)
    
    if img is None:
        raise FileNotFoundError(f"无法读取图像: {image_path}")
    
    dtype = img.dtype
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    
    if len(img.shape) == 3:
        if img.shape[2] == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img[:, :, 0]
        tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
        result = tophat
    else:
        gray = img
        tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
        result = tophat
    
    result = result.astype(dtype)
    
    dir_path = os.path.dirname(image_path)
    filename = os.path.basename(image_path)
    name, ext = os.path.splitext(filename)
    output_path = os.path.join(dir_path, f"{name}_tophat{ext}")
    
    if output_path.lower().endswith(('.tif', '.tiff')):
        tifffile.imwrite(output_path, result, compression=None)
    else:
        cv2.imwrite(output_path, result)
    
    print(f"Top-Hat背景校正结果已保存至: {output_path} (dtype={dtype})")


def main():
    if len(sys.argv) != 2:
        print("用法: python tophat_background.py <图像路径>")
        print("示例: python tophat_background.py input.jpg")
        print("可选: 修改kernel_size参数调整背景估计尺度")
        sys.exit(1)
    
    image_path = sys.argv[1]
    tophat_background_correction(image_path, kernel_size=21)


if __name__ == "__main__":
    main()
