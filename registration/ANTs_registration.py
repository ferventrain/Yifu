import os
from pathlib import Path
from typing import Optional, Tuple, Union, Dict
import numpy as np
import ants
import SimpleITK as sitk
import tifffile
from scipy import ndimage
from tqdm import tqdm


class BidirectionalRegistration:
    """双向配准类：支持atlas到image和image到atlas的配准"""
    
    UPSAMPLING_METHODS = {
        'nearest': 0,
        'linear': 1,
        'cubic': 3,
        'quintic': 5
    }
    
    def __init__(self, 
                 atlas_image_path: str, 
                 atlas_label_path: str,
                 target_image_path: str,
                 original_shape: Tuple[int, int, int]):
        """
        Args:
            atlas_image_path: Allen脑图谱原始图像路径
            atlas_label_path: Allen脑图谱标签图像路径
            target_image_path: 目标图像路径（downsample后的图像）
            original_shape: 目标图像原始形状（用于上采样）
        """
        self.atlas_image = ants.image_read(atlas_image_path)
        self.atlas_label = ants.image_read(atlas_label_path)
        self.target_image = self.prepare_target_image(target_image_path)
        self.target_path = Path(target_image_path)
        self.original_shape = original_shape
        self.sample_dir = self.target_path.parent  # 包含Ch0，mask，配准结果等文件夹的文件夹
    
    def prepare_target_image(self, target_path: Union[str, Path]) -> ants.ANTsImage:
        """准备目标图像"""
        target_path = Path(target_path)
        
        if target_path.is_dir():
            # 如果是文件夹，转换TIFF栈为NIfTI
            print("Converting TIFF stack to NIfTI...")
            volume = self._load_tiff_stack(target_path)
            temp_nifti = target_path / "temp_volume.nii.gz"
            sitk.WriteImage(sitk.GetImageFromArray(volume), str(temp_nifti))
            target_image = ants.image_read(str(temp_nifti))
        else:
            # 直接读取NIfTI文件
            target_image = ants.image_read(str(target_path))
        
        return target_image
    
    def _load_tiff_stack(self, folder_path: Path) -> np.ndarray:
        """加载TIFF栈"""
        tiff_files = sorted(folder_path.glob('*.tif*'))
        stack = []
        for tiff_file in tiff_files:
            img = tifffile.imread(tiff_file)
            stack.append(img)
        return np.stack(stack, axis=0)
    
    def register(self,
                mode: str = 'atlas2image',
                registration_type: str = 'SyN',
                **kwargs) -> Dict:
        """
        执行配准
        
        Args:
            mode: 'atlas2image' 或 'image2atlas'
            registration_type: 配准类型
            
        Returns:
            包含配准结果的字典
        """
        if mode == 'atlas2image':
            return self._register_atlas_to_image(registration_type, **kwargs)
        elif mode == 'image2atlas':
            return self._register_image_to_atlas(registration_type, **kwargs)
        else:
            raise ValueError(f"Invalid mode: {mode}. Choose 'atlas2image' or 'image2atlas'")
    
    def _register_atlas_to_image(self, 
                                registration_type: str,
                                **kwargs) -> Dict:
        """将Atlas配准到图像空间"""
        print(f"Performing {registration_type} registration: Atlas → Image...")
        
        # 执行配准
        registration = ants.registration(
            fixed=self.target_image,
            moving=self.atlas_image,
            type_of_transform=registration_type,
            **kwargs
        )
        
        # 应用变换到脑图谱标签
        warped_label = ants.apply_transforms(
            fixed=self.target_image,
            moving=self.atlas_label,
            transformlist=registration['fwdtransforms'],
            interpolator='nearestNeighbor'
        )
        
        return {
            'warped_image': registration['warpedmovout'],
            'warped_label': warped_label,
            'transforms': registration,
            'mode': 'atlas2image'
        }
    
    def _register_image_to_atlas(self,
                                registration_type: str,
                                **kwargs) -> Dict:
        """将图像配准到Atlas空间"""
        print(f"Performing {registration_type} registration: Image → Atlas...")
        
        # 执行配准
        registration = ants.registration(
            fixed=self.atlas_image,
            moving=self.target_image,
            type_of_transform=registration_type,
            **kwargs
        )
        
        # 如果有下采样后目标图像的标签，也进行变换，用于后续制作热图
        target_label = None
        target_label_path = self.sample_dir / (f"{self.target_path.stem}_mask_downsampled.nii.gz")
        if target_label_path.exists():
            target_label = ants.image_read(str(target_label_path))
            target_label = ants.apply_transforms(
                fixed=self.atlas_image,
                moving=target_label,
                transformlist=registration['fwdtransforms'],
                interpolator='nearestNeighbor'
            )

        return {
            'warped_image': registration['warpedmovout'],
            'warped_label': target_label,  # 这里是图像的标签变换到atlas空间，用于制作热图
            'atlas_label': self.atlas_label,  # 原始atlas标签
            'transforms': registration,
            'mode': 'image2atlas'
        }
    
    def upsample_label_chunked(self,
                               label_image: ants.ANTsImage,
                               output_dir: str,
                               method: str = 'nearest',
                               chunk_size: int = 50) -> None:
        """
        分块上采样标签图像，直接保存每个块避免内存问题
        
        Args:
            label_image: ANTs标签图像
            original_shape: 原始图像形状 [z, y, x]
            output_dir: 输出目录
            method: 上采样方法
            chunk_size: 每个块的切片数量
        """
        output_path = Path(output_dir)
        tiff_output = output_path / f"{self.target_path.stem}_upsampled_label"
        tiff_output.mkdir(parents=True, exist_ok=True)
        
        # 获取标签数组
        label_array = label_image.numpy()
        current_shape = label_array.shape
        
        print(f"Upsampling from {current_shape} to {self.original_shape}...")
        
        # 计算缩放因子
        zoom_factors = [
            target / current 
            for target, current in zip(self.original_shape, current_shape)
        ]
        
        # 确定插值阶数
        order = self.UPSAMPLING_METHODS.get(method, 0)
        
        # 计算每个块在原始空间中的切片数
        slices_per_chunk_original = int(chunk_size * zoom_factors[0])
        
        # 跟踪输出切片索引
        output_slice_idx = 0
        
        # 分块处理
        num_chunks = (current_shape[0] + chunk_size - 1) // chunk_size
        
        for chunk_idx in tqdm(range(num_chunks), desc="Upsampling and saving chunks"):
            start_idx = chunk_idx * chunk_size
            end_idx = min(start_idx + chunk_size, current_shape[0])
            
            # 获取当前块
            chunk = label_array[start_idx:end_idx]
            
            # 上采样当前块
            upsampled_chunk = ndimage.zoom(chunk, zoom_factors, order=order)
            
            # 确保标签值保持整数
            if method == 'nearest':
                upsampled_chunk = np.round(upsampled_chunk).astype(np.uint16)
            else:
                upsampled_chunk = upsampled_chunk.astype(np.uint16)
            
            # 直接保存当前块的每个切片
            for slice_idx in range(upsampled_chunk.shape[0]):
                tifffile.imwrite(
                    str(tiff_output / f"label_{output_slice_idx:06d}.tiff"),
                    upsampled_chunk[slice_idx],
                    compression='lzw'  # 使用压缩
                )
                output_slice_idx += 1
            
            # 立即释放内存
            del chunk
            del upsampled_chunk
        
        print(f"Saved {output_slice_idx} upsampled slices to {tiff_output}")
    
    def save_registration_results(self,
                                 results: Dict,
                                 output_dir: str,
                                 save_transforms: bool = False) -> None:
        """保存配准结果"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        mode = results['mode']
        
        # 保存配准后的标签为TIFF栈
        # atlas2image mode下，label是atlas空间的标签，需要上采样到原始空间
        # image2atlas mode下，label是原始空间的标签，直接保存即可，后续用于制作热图
        if results['warped_label'] is not None:
            label_array = results['warped_label'].numpy()
            if mode == 'atlas2image':
                # 上采样atlas标签到原始空间
                label_dir = self.sample_dir / "atlas_upsampled"
                self.upsample_label_chunked(
                    label_image=results['warped_label'],
                    output_dir=str(label_dir)
                )
            else:
                # 保存下采样+变换后atlas空间的mask
                mask_dir = output_path / f"{self.target_path.stem}_warped_mask"
                mask_dir.mkdir(exist_ok=True)
                print(f"Saving warped mask to {mask_dir}...")
                for i in range(label_array.shape[0]):
                    tifffile.imwrite(
                        str(mask_dir / f"mask_{i:06d}.tiff"),
                        label_array[i].astype(np.uint16),
                        compression='lzw'
                    )
        
        # 保存变换参数（如果需要）
        if save_transforms and 'transforms' in results:
            transforms_dir = output_path / "transforms"
            transforms_dir.mkdir(exist_ok=True)
            # 复制变换文件
            for i, transform in enumerate(results['transforms']['fwdtransforms']):
                if os.path.exists(transform):
                    import shutil
                    shutil.copy(transform, transforms_dir / f"fwd_{i}_{os.path.basename(transform)}")
    
    def run_full_pipeline(self,
                         mode: str = 'atlas2image',
                         registration_type: str = 'SyN',
                         upsample_method: str = 'nearest',
                         chunk_size: int = 50,
                         save_transforms: bool = False) -> None:
        """
        运行完整的配准流程
        
        Args:
            mode: 'atlas2image' 或 'image2atlas'
            registration_type: 配准类型
            upsample_method: 上采样方法
            chunk_size: 分块大小
            save_transforms: 是否保存变换文件
        """
        # 执行配准
        results = self.register(mode, registration_type)
        output_dir = self.sample_dir
        # 保存配准结果
        self.save_registration_results(results, output_dir, save_transforms)
        # 清理临时文件
        self._cleanup_temp_files(results['transforms'])
    
    def _cleanup_temp_files(self, transforms: dict) -> None:
        """清理临时文件"""
        # 清理变换文件
        for transform in transforms.get('fwdtransforms', []):
            if os.path.exists(transform):
                os.remove(transform)
        for transform in transforms.get('invtransforms', []):
            if os.path.exists(transform):
                os.remove(transform)
        
        # 清理临时NIfTI文件
        if self.target_path.is_dir():
            temp_nifti = self.target_path / "temp_volume.nii.gz"
            if temp_nifti.exists():
                temp_nifti.unlink()


def main():
    import argparse
    import json
    
    parser = argparse.ArgumentParser(description="Bidirectional registration between Allen atlas and LSFM images")
    parser.add_argument('--target', required=True, 
                       help='Target image path (folder with TIFFs or .nii.gz file)')
    parser.add_argument('--atlas_image', required=True,
                       help='Allen atlas image path')
    parser.add_argument('--atlas_label', required=True,
                       help='Allen atlas label path')
    parser.add_argument('--mode', default='atlas2image',
                       choices=['atlas2image', 'image2atlas'],
                       help='Registration direction')
    parser.add_argument('--registration_type', default='SyN',
                       choices=['Rigid', 'Affine', 'SyN', 'SyNRA'],
                       help='Registration type')
    parser.add_argument('--upsample_method', default='nearest',
                       choices=['nearest', 'linear', 'cubic', 'quintic'],
                       help='Upsampling interpolation method')
    parser.add_argument('--chunk_size', type=int, default=50,
                       help='Number of slices per chunk for upsampling')
    parser.add_argument('--save_transforms', action='store_true',
                       help='Save transformation files')
    
    args = parser.parse_args()
    
    # 读取原始形状信息（如果提供）
    original_shape = None
    origin_shape_path = Path(args.target).parent / 'original_shape.json'
    if origin_shape_path.exists():
        with open(origin_shape_path, 'r') as f:
            config = json.load(f)
            original_shape = tuple(config['original_shape'])  # [z, y, x]
    else:
        print(f'Warning: {origin_shape_path} not found. Skipping original shape upsampling.')
    
    # 创建配准器并运行
    registrator = BidirectionalRegistration(
        args.atlas_image, 
        args.atlas_label,
        args.target,
        original_shape
    )
    
    registrator.run_full_pipeline(
        args.mode,
        original_shape,
        args.registration_type,
        args.upsample_method,
        args.chunk_size,
        args.save_transforms
    )


if __name__ == "__main__":
    main()
