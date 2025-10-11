import os
import json
from pathlib import Path
from typing import Tuple, Optional, Union
import numpy as np
from tqdm import tqdm
import tifffile
import SimpleITK as sitk
from scipy import ndimage


class ImageDownsampler:
    """图像下采样处理类"""
    
    INTERPOLATION_METHODS = {
        'nearest': sitk.sitkNearestNeighbor,
        'linear': sitk.sitkLinear,
        'bspline': sitk.sitkBSpline,
        'gaussian': sitk.sitkGaussian,
        'lanczos': sitk.sitkLanczosWindowedSinc
    }
    
    def __init__(self, resolution_config_path: str):
        """
        Args:
            resolution_config_path: 包含分辨率信息的JSON文件路径
        """
        with open(resolution_config_path, 'r') as f:
            self.config = json.load(f)
        self.source_resolution = self.config['source_resolution']  # [x, y, z] in μm
        self.target_resolution = self.config['target_resolution']  # Allen atlas resolution
        self.downsample_factors = self._calculate_downsample_factors()
    
    def _calculate_downsample_factors(self) -> Tuple[float, float, float]:
        """计算下采样因子"""
        factors = [
            source / target
            for source, target in zip(self.source_resolution, self.target_resolution)
        ]
        factors = factors[::-1]
        return tuple([round(factor, 3) for factor in factors])
    
    def downsample_tiff_stack(self, 
                            input_folder: str, 
                            method: str = 'linear',
                            chunk_size: int = 100,
                            downsample_mask: bool = True) -> None: 
        """
        对TIFF栈进行下采样
        
        Args:
            input_folder: 输入TIFF文件夹路径
            method: 插值方法
            chunk_size: 每个块的切片数量
            downsample_mask: 下采样mask
        """
        input_path = Path(input_folder)
        
        # 读取所有TIFF文件
        tiff_files = sorted(input_path.glob('*.tif*'))
        if not tiff_files:
            raise ValueError(f"No TIFF files found in {input_folder}")
        
        print(f"Found {len(tiff_files)} TIFF files")
        print(f"Downsample factors: {self.downsample_factors}")
        
        # 处理mask下采样
        mask_output_path = None
        mask_input_path = input_path.parent / f"{input_path.stem}_mask"
        if mask_input_path.exists():
            mask_output_path = mask_input_path.parent / f"{mask_input_path.stem}_downsample_mask"
            mask_output_path.mkdir(parents=True, exist_ok=True)
        if downsample_mask and os.listdir(mask_output_path) != os.listdir(mask_input_path):
            # 检查是单个ome.tiff还是多个tiff
            ome_tiff = list(mask_input_path.glob('*.ome.tiff'))
            if ome_tiff:
                # 单个ome.tiff文件
                print(f"Processing mask: {ome_tiff[0]}")
                mask_volume = tifffile.imread(ome_tiff[0])
                if mask_volume.ndim == 2:
                    mask_volume = mask_volume[np.newaxis, ...]
                
                # 下采样mask - 使用nearest避免插值产生中间值
                downsampled_mask = self._downsample_volume(mask_volume, 'nearest')
                
                # 保存mask
                for i in tqdm(range(downsampled_mask.shape[0]), desc="Saving mask slices"):
                    tifffile.imwrite(
                        mask_output_path / f"mask_{i:04d}.tiff",
                        downsampled_mask[i].astype(mask_volume.dtype)
                    )
                
                # 保存为NIfTI
                mask_nifti = sitk.GetImageFromArray(downsampled_mask)
                mask_nifti.SetSpacing(self.target_resolution[::-1])
                sitk.WriteImage(mask_nifti, str(mask_output_path / "mask.nii.gz"), useCompression=True)
                print(f"Mask saved to {mask_output_path}")
                
            else:
                # 多个tiff文件
                mask_files = sorted(mask_input_path.glob('*.tif*'))
                if mask_files:
                    print(f"Processing {len(mask_files)} mask files")
                    mask_volume = np.stack([tifffile.imread(f) for f in mask_files], axis=0)
                    
                    # 下采样mask
                    downsampled_mask = self._downsample_volume(mask_volume, 'nearest')
                    
                    # 保存mask
                    for i in range(downsampled_mask.shape[0]):
                        tifffile.imwrite(
                            mask_output_path / f"mask_{i:04d}.tiff",
                            downsampled_mask[i].astype(mask_volume.dtype)
                        )
                    
                    # 保存为NIfTI
                    mask_nifti = sitk.GetImageFromArray(downsampled_mask)
                    mask_nifti.SetSpacing(self.target_resolution[::-1])
                    sitk.WriteImage(mask_nifti, str(mask_output_path / "mask.nii.gz"), useCompression=True)
                    print(f"Mask saved to {mask_output_path}")
            del mask_volume, downsampled_mask
            # ==========================================
            
            # 获取图像维度
            first_img = tifffile.imread(tiff_files[0])
            height, width = first_img.shape
            depth = len(tiff_files)
            dtype = first_img.dtype
            
            # 计算输出维度
            out_depth = int(depth * self.downsample_factors[0])
            out_height = int(height * self.downsample_factors[1])
            out_width = int(width * self.downsample_factors[2])
            
            print(f"Input shape: ({depth}, {height}, {width})")
            print(f"Output shape: ({out_depth}, {out_height}, {out_width})")

            # 保存原始形状信息
            original_shape = (depth, height, width)
            with open(input_path.parent / 'original_shape.json', 'w') as f:
                json.dump({'original_shape': original_shape}, f)
            
            # 创建输出列表来收集所有下采样的切片
            all_downsampled_slices = []
            
            # 分块处理
            num_chunks = (depth + chunk_size - 1) // chunk_size
            
            for chunk_idx in tqdm(range(num_chunks), desc="Processing chunks"):
                start_idx = chunk_idx * chunk_size
                end_idx = min(start_idx + chunk_size, depth)
                
                # 读取当前块
                chunk_data = []
                for i in range(start_idx, end_idx):
                    img = tifffile.imread(tiff_files[i])
                    chunk_data.append(img)
                chunk_volume = np.stack(chunk_data, axis=0)
                
                # 下采样当前块
                downsampled_chunk = self._downsample_volume(chunk_volume, method)
                
                # 收集下采样后的切片
                for slice_idx in range(downsampled_chunk.shape[0]):
                    all_downsampled_slices.append(downsampled_chunk[slice_idx])

            # 保存所有下采样的切片
            print("Saving downsampled stack...")
            output_path = input_path.parent / f"{input_path.stem}_downsample"
            output_path.mkdir(parents=True, exist_ok=True)
            for i, slice_img in enumerate(tqdm(all_downsampled_slices, desc="Saving TIFFs")):
                output_file = output_path / f"downsampled_{i:04d}.tiff"
                tifffile.imwrite(output_file, slice_img.astype(dtype))
            
            # 保存为NIfTI
            nifti_path = output_path / "volume.nii.gz"
            self._save_as_nifti(all_downsampled_slices, nifti_path)
            print(f"Saved as NIfTI: {nifti_path}")



    def _downsample_volume(self, volume: np.ndarray, method: str) -> np.ndarray:
        """执行体积下采样"""
        new_shape = tuple(
            int(dim * factor) 
            for dim, factor in zip(volume.shape, self.downsample_factors)
        )
        
        # 主要使用scipy.ndimage.zoom，它速度快且内存效率高
        interpolation_order = {
            'nearest': 0,    # 最近邻
            'linear': 1,     # 线性（默认）
            'quadratic': 2,  # 二次
            'cubic': 3,      # 三次
        }
        
        if method in interpolation_order:
            # 直接使用scipy zoom
            return ndimage.zoom(
                volume, 
                self.downsample_factors, 
                order=interpolation_order[method],
                mode='constant',  # 边界处理
                prefilter=True    # 使用预滤波提高质量
            )
        else:
            # 默认线性插值
            return ndimage.zoom(volume, self.downsample_factors, order=1)


    def _save_as_nifti(self, slices_list: list, output_path: Path) -> None:
        """分块保存为NIfTI格式"""
        # 将列表转换为数组
        volume = np.stack(slices_list, axis=0)
        
        # 保存为NIfTI
        sitk_image = sitk.GetImageFromArray(volume)
        sitk_image.SetSpacing(self.target_resolution[::-1])
        
        # 使用压缩保存，路径不能包含中文
        sitk.WriteImage(sitk_image, str(output_path), useCompression=True)
        
        # 释放内存
        del volume


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Downsample LSFM images")
    parser.add_argument('--input_folder', required=True, help='Input folder containing TIFF files, should not contain Chinese characters')
    parser.add_argument('--resolution_config', required=True, help='JSON file with resolution info')
    parser.add_argument('--method', default='linear', 
                       choices=['nearest', 'linear', 'quadratic', 'cubic'],
                       help='Interpolation method')
    parser.add_argument('--chunk_size', type=int, default=100,
                       help='Number of slices per chunk (adjust based on available memory)')
    parser.add_argument('--downsample_mask', action='store_true', help='Downsample mask files')
    
    args = parser.parse_args()
    
    
    # 执行下采样
    downsampler = ImageDownsampler(args.resolution_config)
    downsampler.downsample_tiff_stack(
        args.input_folder, 
        args.method,
        args.chunk_size,
        args.downsample_mask
    )


if __name__ == "__main__":
    main()
