import cv2
import numpy as np
import sys
import os
import tifffile
import argparse


def remove_scattering(image_path: str, sigma: float = 50.0, weight: float = 1.0, accelerate: bool = True) -> None:
    """
    Remove scattering/halo background after Tophat using Gaussian background estimation.
    
    思路：Tophat去掉大背景后，仍有散射光晕。用高斯模糊估计局部背景，然后减法去掉光晕。
    
    Parameters:
        image_path: 输入图像路径
        sigma: 高斯sigma，越大背景越平滑。一般是你感兴趣结构直径的2-3倍。
        weight: 背景减法权重，result = img - weight * background
        accelerate: 如果True，用降采样加速（大sigma时加速明显，精度损失可忽略）
    """
    if image_path.lower().endswith(('.tif', '.tiff')):
        img = tifffile.imread(image_path)
    else:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    
    dtype = img.dtype
    if len(img.shape) == 3:
        if img.shape[2] == 1:
            img = img[:, :, 0]
    
    img_float = img.astype(np.float64)
    height, width = img_float.shape
    
    # Accelerate by downsampling for large sigma: Gaussian blur on smaller image is much faster
    if accelerate and sigma > 20 and min(height, width) > 500:
        # Calculate downsample ratio based on sigma
        downsample_ratio = max(2, int(sigma / 20))
        new_h = max(1, height // downsample_ratio)
        new_w = max(1, width // downsample_ratio)
        
        # Downsample
        img_down = cv2.resize(img_float, (new_w, new_h))
        sigma_down = sigma / downsample_ratio
        
        # Gaussian blur on small image
        background_down = cv2.GaussianBlur(img_down, ksize=(0, 0), sigmaX=sigma_down, sigmaY=sigma_down)
        
        # Upsample back to original size
        background = cv2.resize(background_down, (width, height), interpolation=cv2.INTER_CUBIC)
    else:
        # Direct blur on original size
        background = cv2.GaussianBlur(img_float, ksize=(0, 0), sigmaX=sigma, sigmaY=sigma)
    
    result = np.clip(img_float - weight * background, 0, None)
    
    if np.issubdtype(dtype, np.integer):
        max_val = np.iinfo(dtype).max
        result = np.clip(result, 0, max_val).astype(dtype)
    else:
        result = result.astype(dtype)
    
    dir_path = os.path.dirname(image_path) if os.path.dirname(image_path) else '.'
    filename = os.path.basename(image_path)
    name, ext = os.path.splitext(filename)
    output_path = os.path.join(dir_path, f"{name}_scatter_removed{ext}")
    
    if output_path.lower().endswith(('.tif', '.tiff')):
        tifffile.imwrite(output_path, result, compression=None)
    else:
        if result.dtype != np.uint8:
            result = (result / result.max() * 255).astype(np.uint8)
        cv2.imwrite(output_path, result)
    
    if accelerate and sigma > 20:
        print(f"Scattering removal result saved to: {output_path}")
        print(f"  Parameters: sigma={sigma}, weight={weight}, accelerated=Yes")
    else:
        print(f"Scattering removal result saved to: {output_path}")
        print(f"  Parameters: sigma={sigma}, weight={weight}, accelerated=No")


def main():
    parser = argparse.ArgumentParser(description="Remove scattering/halo background after Tophat.")
    parser.add_argument("image_path", type=str, help="Input image path (usually after Tophat)")
    parser.add_argument("--sigma", type=float, default=50.0, help="Gaussian sigma for background estimation (default: 50)")
    parser.add_argument("--weight", type=float, default=1.0, help="Subtraction weight: result = img - weight * background (default: 1.0)")
    parser.add_argument("--no-accelerate", action="store_true", help="Disable downsampling acceleration (slower but more accurate)")
    
    args = parser.parse_args()
    accelerate = not args.no_accelerate
    remove_scattering(args.image_path, sigma=args.sigma, weight=args.weight, accelerate=accelerate)


if __name__ == "__main__":
    main()
