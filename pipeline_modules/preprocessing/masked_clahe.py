import cv2
import numpy as np
import sys
import os
import tifffile
import argparse


def clahe_enhance(image_path: str, clip_limit: float = 2.0, grid_size: tuple = (8, 8), use_mask: bool = False) -> None:
    """
    CLAHE对比度增强。默认直接全局增强，适合已经做过去背景的图像（比如Tophat后）。
    如果use_mask=True，先用Otsu找前景mask，只增强前景。
    
    Parameters:
        image_path: 输入图像路径
        clip_limit: CLAHE的剪切限制
        grid_size: CLAHE网格块大小
        use_mask: 是否只增强前景（Otsu mask）
    """
    if image_path.lower().endswith(('.tif', '.tiff')):
        img = tifffile.imread(image_path)
    else:
        img = cv2.imread(image_path)
    
    if img is None:
        raise FileNotFoundError(f"无法读取图像: {image_path}")
    
    dtype = img.dtype
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    
    if len(img.shape) == 3 and img.shape[2] == 3:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l_channel = lab[:, :, 0]
        
        if use_mask:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            if gray.max() > 0:
                if gray.max() > 255:
                    gray_norm = (gray / gray.max() * 255).astype(np.uint8)
                else:
                    gray_norm = gray.astype(np.uint8)
                _, mask = cv2.threshold(gray_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            else:
                mask = np.zeros_like(gray, dtype=np.uint8)
            
            full_eq = clahe.apply(l_channel)
            l_channel[mask > 0] = full_eq[mask > 0]
        else:
            l_channel = clahe.apply(l_channel)
        
        lab[:, :, 0] = l_channel
        result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        mask_output = None
    else:
        if len(img.shape) == 3:
            gray = img[:, :, 0]
        else:
            gray = img
        
        if use_mask:
            if gray.max() > 0:
                if gray.max() > 255:
                    gray_norm = (gray / gray.max() * 255).astype(np.uint8)
                else:
                    gray_norm = gray.astype(np.uint8)
                _, mask = cv2.threshold(gray_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            else:
                mask = np.zeros_like(gray, dtype=np.uint8)
            
            full_eq = clahe.apply(gray.astype(np.uint16))
            result = gray.copy()
            result[mask > 0] = full_eq[mask > 0]
            mask_output = mask
        else:
            result = clahe.apply(gray.astype(np.uint16))
            mask_output = None
    
    result = result.astype(dtype)
    
    dir_path = os.path.dirname(image_path) if os.path.dirname(image_path) else '.'
    filename = os.path.basename(image_path)
    name, ext = os.path.splitext(filename)
    
    if use_mask:
        output_path = os.path.join(dir_path, f"{name}_masked_clahe{ext}")
        if mask_output is not None:
            mask_output_path = os.path.join(dir_path, f"{name}_otsu_mask{ext}")
    else:
        output_path = os.path.join(dir_path, f"{name}_clahe{ext}")
    
    if output_path.lower().endswith(('.tif', '.tiff')):
        tifffile.imwrite(output_path, result, compression=None)
        if use_mask and mask_output is not None:
            tifffile.imwrite(mask_output_path, mask_output, compression=None)
    else:
        cv2.imwrite(output_path, result)
        if use_mask and mask_output is not None:
            cv2.imwrite(mask_output_path, mask_output)
    
    if use_mask:
        print(f"Otsu mask已保存至: {mask_output_path}")
        print(f"masked CLAHE结果已保存至: {output_path} (dtype={dtype})")
    else:
        print(f"CLAHE结果已保存至: {output_path} (dtype={dtype})")


def main():
    parser = argparse.ArgumentParser(description="CLAHE contrast enhancement. Works great after Tophat background removal.")
    parser.add_argument("image_path", type=str, help="Input image path")
    parser.add_argument("--masked", action="store_true", help="Use Otsu mask to only enhance foreground (default: False, enhance whole image)")
    parser.add_argument("--clip-limit", type=float, default=2.0, help="CLAHE clip limit (default: 2.0)")
    parser.add_argument("--grid-size", type=int, default=16, help="CLAHE grid size (default: 8)")
    
    args = parser.parse_args()
    
    grid_size = (args.grid_size, args.grid_size)
    clahe_enhance(args.image_path, clip_limit=args.clip_limit, grid_size=grid_size, use_mask=args.masked)


if __name__ == "__main__":
    main()
