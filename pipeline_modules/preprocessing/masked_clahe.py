import cv2
import numpy as np
import sys
import os
import tifffile
import argparse


def clahe_array(
    image: np.ndarray,
    *,
    clip_limit: float = 2.0,
    grid_size: int | tuple[int, int] = 16,
    use_mask: bool = False,
) -> np.ndarray:
    """
    Apply CLAHE to a 2D grayscale (or single-channel) array and return the result.

    Preserves input dtype. For uint16 inputs OpenCV CLAHE is applied on uint16.
    """
    if image.ndim == 3 and image.shape[-1] in (1, 3, 4):
        if image.shape[-1] == 1:
            gray = image[..., 0]
        else:
            raise ValueError("clahe_array expects 2D grayscale; got multi-channel image")
    elif image.ndim == 2:
        gray = image
    else:
        raise ValueError(f"clahe_array expects 2D image, got shape={image.shape}")

    if isinstance(grid_size, int):
        tile = (grid_size, grid_size)
    else:
        tile = (int(grid_size[0]), int(grid_size[1]))

    original_dtype = gray.dtype
    clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=tile)

    if np.issubdtype(original_dtype, np.integer) and np.iinfo(original_dtype).max > 255:
        work = gray.astype(np.uint16, copy=False)
    else:
        # Map to uint8 for OpenCV when low bit-depth / float
        if np.issubdtype(original_dtype, np.floating):
            finite = gray[np.isfinite(gray)]
            vmax = float(finite.max()) if finite.size else 1.0
            vmax = vmax if vmax > 0 else 1.0
            work = np.clip(gray / vmax * 255.0, 0, 255).astype(np.uint8)
        else:
            work = gray.astype(np.uint8, copy=False)

    if use_mask:
        if work.dtype == np.uint16:
            norm = (work / max(int(work.max()), 1) * 255).astype(np.uint8)
        else:
            norm = work
        _, mask = cv2.threshold(norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        enhanced = clahe.apply(work)
        result = work.copy()
        result[mask > 0] = enhanced[mask > 0]
    else:
        result = clahe.apply(work)

    if original_dtype == result.dtype:
        return result
    if np.issubdtype(original_dtype, np.floating):
        return result.astype(original_dtype) / 255.0
    max_in = np.iinfo(original_dtype).max
    max_work = np.iinfo(result.dtype).max
    scaled = result.astype(np.float64) / max_work * max_in
    return np.clip(scaled, 0, max_in).astype(original_dtype)


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
    if len(img.shape) == 3 and img.shape[2] == 3:
        # Keep BGR path behavior for color images via file CLI.
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
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
            mask_output = mask
        else:
            l_channel = clahe.apply(l_channel)
            mask_output = None
        
        lab[:, :, 0] = l_channel
        result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    else:
        if len(img.shape) == 3:
            gray = img[:, :, 0]
        else:
            gray = img
        result = clahe_array(
            gray,
            clip_limit=clip_limit,
            grid_size=grid_size,
            use_mask=use_mask,
        )
        mask_output = None
        if use_mask:
            work = gray
            if work.max() > 255:
                gray_norm = (work / work.max() * 255).astype(np.uint8)
            else:
                gray_norm = work.astype(np.uint8)
            _, mask_output = cv2.threshold(gray_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
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
