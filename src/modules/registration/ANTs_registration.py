import os
import json
import shutil
from pathlib import Path
from typing import Optional, Tuple, Union, Dict
import numpy as np
import ants
import tifffile
from scipy import ndimage
from tqdm import tqdm
from analyze_density import BrainDensityAnalyzer


class BidirectionalRegistration:
    """双向配准类：支持atlas到image和image到atlas的配准"""
    
    UPSAMPLING_METHODS = {
        'nearest': 0,
        'linear': 1,
        'cubic': 3,
        'quintic': 5
    }
    
    def __init__(self, 
                 sample_dir: str,
                 target_channel: str,
                 atlas_image_path: str, 
                 atlas_label_path: str,
                 register_channel: str,
                 original_shape: Tuple[int, int, int],
                 density_cfg_path: Optional[str] = None):
        
        self.sample_dir = Path(sample_dir)
        self.target_channel = target_channel
        self.register_channel = register_channel
        self.original_shape = original_shape
        
        # Load Atlas
        self.atlas_image = ants.image_read(atlas_image_path)
        self.atlas_label = ants.image_read(atlas_label_path)
        
        # Load Config
        current_dir = Path(__file__).parent
        resolution_config_path = current_dir / 'resolution.json'
        with open(resolution_config_path, 'r') as f:
            self.config = json.load(f)
        self.source_resolution = self.config['source_resolution']
        self.atlas_resolution = self.config['target_resolution']
        
        # Load Target Image (used for image2atlas warping)
        self.target_image = self._find_and_load_target_image()
        if self.target_image is None:
            print("No target image or mask found!")
            return

        # Load Register Image (downsampled sample)
        reg_img_path = self.sample_dir / f"ch{self.register_channel}_downsample/volume.nii.gz"
        self.register_image = ants.image_read(str(reg_img_path))
        
        # Density Config
        self.density_cfg_path = density_cfg_path or os.path.join(os.path.dirname(os.path.abspath(__file__)), 'add_id_ytw.json')

    def _find_and_load_target_image(self) -> Optional[ants.ANTsImage]:
        """查找并加载目标图像"""
        # 优先级：downsample folder -> downsample_mask folder
        possible_dirs = [
            self.sample_dir / f"ch{self.target_channel}_downsample",
            self.sample_dir / f"ch{self.target_channel}_downsample_mask"
        ]
        
        for p in possible_dirs:
            if p.exists():
                return self.prepare_target_image(p)
        return None

    def prepare_target_image(self, target_path: Path) -> ants.ANTsImage:
        """准备目标图像：优先读取 NIfTI，否则从 TIFF 转换"""
        nifti_files = ["volume.nii.gz", "mask.nii.gz"]
        
        if target_path.is_dir():
            # Check for existing NIfTI
            for f in nifti_files:
                if (target_path / f).exists():
                    return ants.image_read(str(target_path / f))
            
            # Convert TIFF stack
            print(f"Converting TIFF stack in {target_path} to NIfTI...")
            volume = self._load_tiff_stack(target_path)
            temp_nifti = target_path / "volume.nii.gz"
            
            # Convert to ANTsImage
            # volume is (z, y, x), ANTs expects (x, y, z) for from_numpy
            volume_ants = np.transpose(volume, (2, 1, 0))
            
            ants_vol = ants.from_numpy(
                volume_ants,
                origin=[0, 0, 0],
                spacing=self.atlas_resolution,
                direction=np.eye(3)
            )
            
            ants.image_write(ants_vol, str(temp_nifti))
            
            return ants_vol
        else:
            return ants.image_read(str(target_path))
    
    def _load_tiff_stack(self, folder_path: Path) -> np.ndarray:
        """加载TIFF栈"""
        tiff_files = sorted(folder_path.glob('*.tif*'))
        stack = [tifffile.imread(f) for f in tiff_files]
        return np.stack(stack, axis=0)

    def _save_volume_as_tiff(self, data: Union[ants.ANTsImage, np.ndarray], output_dir: Path, prefix: str = "image"):
        """通用保存 TIFF 栈方法"""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if isinstance(data, ants.ANTsImage):
            arr = data.numpy()
            # ANTs numpy is (x, y, z), transpose to (y, x, z) for TIFF
            arr = np.transpose(arr, (1, 0, 2))
        else:
            arr = data
            
        # Ensure 3D (y, x, z)
        if arr.ndim != 3:
            raise ValueError(f"Expected 3D array, got shape {arr.shape}")
            
        print(f"Saving {prefix} TIFFs to {output_dir} (shape: {arr.shape})...")
        for i in range(arr.shape[2]):
            tifffile.imwrite(
                str(output_dir / f"{prefix}_{i:04d}.tiff"),
                arr[:, :, i].astype(np.uint16),
                compression='lzw'
            )

    def _perform_registration(self, fixed: ants.ANTsImage, moving: ants.ANTsImage, 
                            reg_type: str, **kwargs) -> Dict:
        """通用配准核心逻辑"""
        print(f"Performing {reg_type} registration...")
        print("--- METADATA VERIFICATION ---")
        print(f"Fixed Image  | Spacing: {fixed.spacing}, Origin: {fixed.origin}")
        print(f"Moving Image | Spacing: {moving.spacing}, Origin: {moving.origin}")
        
        # 始终将 Sample (register_image) 的直方图匹配到 Atlas
        # 注意：需要判断哪个是 sample。根据初始化逻辑，self.register_image 是 sample。
        # 如果 fixed 是 register_image，则它匹配 atlas (moving)。
        # 如果 moving 是 register_image，则它匹配 atlas (fixed)。
        
        # 在原代码中，无论方向如何，都执行了:
        # self.register_image = ants.histogram_match_image(self.register_image, self.atlas_image)
        # 这一步应该在调用此函数前或者在此函数内完成，但需要引用 self.register_image
        
        # 为了保持原逻辑，我们在外部做 histogram match。
        
        return ants.registration(
            fixed=fixed,
            moving=moving,
            type_of_transform=reg_type,
            grad_step=0.1,
            aff_random_sampling_rate=0.5,
            **kwargs
        )

    def register(self, mode: str = 'atlas2image', registration_type: str = 'SyN', **kwargs) -> Dict:
        """执行配准"""
        if mode not in ['atlas2image', 'image2atlas']:
            raise ValueError(f"Invalid mode: {mode}")

        # Histogram matching (Sample matches Atlas)
        print("Performing histogram matching (Sample -> Atlas)...")
        self.register_image = ants.histogram_match_image(self.register_image, self.atlas_image)

        if mode == 'atlas2image':
            print("Mode: Atlas -> Image")
            reg_result = self._perform_registration(
                fixed=self.register_image, 
                moving=self.atlas_image, 
                reg_type=registration_type, 
                **kwargs
            )
            
            # Apply transform to Atlas Label
            warped_label = ants.apply_transforms(
                fixed=self.register_image,
                moving=self.atlas_label,
                transformlist=reg_result['fwdtransforms'],
                interpolator='nearestNeighbor'
            )
            
            return {
                'warped_image': reg_result['warpedmovout'],
                'warped_label': warped_label,
                'transforms': reg_result,
                'mode': mode
            }
            
        else: # image2atlas
            print("Mode: Image -> Atlas")
            reg_result = self._perform_registration(
                fixed=self.atlas_image, 
                moving=self.register_image, 
                reg_type=registration_type, 
                **kwargs
            )
            
            # Apply transform to Target Image (Sample)
            warped_target = ants.apply_transforms(
                fixed=self.atlas_image,
                moving=self.target_image,
                transformlist=reg_result['fwdtransforms'],
                interpolator='nearestNeighbor'
            )
            
            return {
                'warped_image': reg_result['warpedmovout'],
                'warped_label': warped_target, # Here label is the warped target image/mask
                'atlas_label': self.atlas_label,
                'transforms': reg_result,
                'mode': mode
            }

    def upsample_label_chunked(self, label_image: ants.ANTsImage, output_dir: str, 
                             method: str = 'nearest', chunk_size: int = 50) -> None:
        """分块上采样标签图像"""
        output_path = Path(output_dir)
        tiff_output = output_path / f"{self.target_channel}_upsampled_label"
        tiff_output.mkdir(parents=True, exist_ok=True)
        
        label_array = label_image.numpy()
        current_shape = label_array.shape
        print(f"Upsampling from {current_shape} to {self.original_shape}...")
        
        zoom_factors = [t / c for t, c in zip(self.original_shape, current_shape)]
        order = self.UPSAMPLING_METHODS.get(method, 0)
        
        output_slice_idx = 0
        num_chunks = (current_shape[0] + chunk_size - 1) // chunk_size
        
        for chunk_idx in tqdm(range(num_chunks), desc="Upsampling chunks"):
            start_idx = chunk_idx * chunk_size
            end_idx = min(start_idx + chunk_size, current_shape[0])
            
            chunk = label_array[start_idx:end_idx]
            upsampled_chunk = ndimage.zoom(chunk, zoom_factors, order=order)
            
            if method == 'nearest':
                upsampled_chunk = np.round(upsampled_chunk).astype(np.uint16)
            else:
                upsampled_chunk = upsampled_chunk.astype(np.uint16)
            
            # Save slices
            for i in range(upsampled_chunk.shape[0]):
                tifffile.imwrite(
                    str(tiff_output / f"label_{output_slice_idx:06d}.tiff"),
                    upsampled_chunk[i],
                    compression='lzw'
                )
                output_slice_idx += 1
            
            del chunk, upsampled_chunk
            
        print(f"Saved {output_slice_idx} slices to {tiff_output}")

    def save_registration_results(self, results: Dict, save_transforms: bool = False, 
                                save_registered_image: bool = False) -> None:
        """保存配准结果"""
        mode = results['mode']
        
        # 1. Save Warped Label / Mask
        if results.get('warped_label') is not None:
            if mode == 'atlas2image':
                # Upsample atlas label to original space
                label_dir = self.sample_dir / f"{self.target_channel}_atlas_label_upsampled"
                self.upsample_label_chunked(results['warped_label'], str(label_dir))
            else:
                # Save warped mask (already in atlas space)
                mask_dir = self.sample_dir / f"ch{self.target_channel}_warped_mask"
                # For image2atlas, warped_label is actually the warped sample mask
                # We need to be careful about dimensions. 
                # warped_label is an ANTsImage. _save_volume_as_tiff handles it.
                self._save_volume_as_tiff(results['warped_label'], mask_dir, prefix="mask")

        # 2. Save Warped Image
        if save_registered_image:
            image_dir = self.sample_dir / f"ch{self.register_channel}_warped_image"
            self._save_volume_as_tiff(results['warped_image'], image_dir, prefix="image")
            
            # Also save NIfTI
            nii_path = image_dir / f"{self.register_channel}_warped_image.nii.gz"
            warped_img = results['warped_image']
            ants.image_write(warped_img, str(nii_path)) # Use ants.image_write directly

        # 3. Save Transforms
        if save_transforms and 'transforms' in results:
            transforms_dir = self.sample_dir / "transforms"
            transforms_dir.mkdir(exist_ok=True)
            for i, transform in enumerate(results['transforms']['fwdtransforms']):
                if os.path.exists(transform):
                    shutil.copy(transform, transforms_dir / f"fwd_{i}_{os.path.basename(transform)}")

    def check_and_run_density_analysis(self, results: Dict):
        """检查并运行密度分析"""
        mask_folder = self.sample_dir / f"ch{self.target_channel}_downsample_mask"
        
        if mask_folder.exists() and results.get('warped_label') is not None:
            print(f"\nFound mask folder: {mask_folder}. Starting density analysis...")
            
            # Save downsampled atlas label (registered)
            downsampled_label_dir = self.sample_dir / f"ch{self.target_channel}_atlas_label_downsampled"
            self._save_volume_as_tiff(results['warped_label'], downsampled_label_dir, prefix="label")
            
            try:
                analyzer = BrainDensityAnalyzer(self.density_cfg_path)
                analysis_results = analyzer.analyze(str(mask_folder), str(downsampled_label_dir))
                
                output_excel = self.sample_dir / f"density_analysis_ch{self.target_channel}.xlsx"
                analyzer.write_to_excel(analysis_results, str(output_excel))
                print(f"Density analysis completed. Saved to {output_excel}")
            except Exception as e:
                print(f"Error during density analysis: {e}")
                import traceback
                traceback.print_exc()
        elif not mask_folder.exists():
            print(f"Density analysis skipped. Mask folder {mask_folder} not found.")

    def run_full_pipeline(self, mode: str = 'atlas2image', registration_type: str = 'SyN',
                         save_registered_image: bool = True, save_transforms: bool = False) -> None:
        """运行完整流程"""
        results = self.register(mode, registration_type)
        self.save_registration_results(results, save_transforms, save_registered_image)
        
        if mode == 'atlas2image':
            self.check_and_run_density_analysis(results)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Bidirectional registration between Allen atlas and LSFM images")
    parser.add_argument('--target_channel', required=True, help='Target channel')
    parser.add_argument('--sample_dir', required=True, help='Sample root directory')
    parser.add_argument('--atlas_image', required=True, help='Allen atlas image path')
    parser.add_argument('--atlas_label', required=True, help='Allen atlas label path')
    parser.add_argument('--register_channel', required=True, help='Registration channel')
    parser.add_argument('--save_registered_image', action='store_true', help='Save registered image')
    parser.add_argument('--mode', default='atlas2image', choices=['atlas2image', 'image2atlas'], help='Direction')
    parser.add_argument('--registration_type', default='SyN', choices=['Rigid', 'Affine', 'SyN', 'SyNRA'])
    parser.add_argument('--upsample_method', default='nearest', choices=['nearest', 'linear', 'cubic', 'quintic'])
    parser.add_argument('--chunk_size', type=int, default=50, help='Chunk size for upsampling')
    parser.add_argument('--save_transforms', action='store_true', help='Save transforms')
    parser.add_argument('--density_cfg', help='Density analysis config path')
    
    args = parser.parse_args()
    
    # Load original shape if exists
    original_shape = None
    origin_shape_path = Path(args.sample_dir) / 'original_shape.json'
    if origin_shape_path.exists():
        with open(origin_shape_path, 'r') as f:
            original_shape = tuple(json.load(f)['original_shape'])
    else:
        print(f'Warning: {origin_shape_path} not found. Skipping upsampling.')
    
    registrator = BidirectionalRegistration(
        args.sample_dir, args.target_channel, args.atlas_image, args.atlas_label,
        args.register_channel, original_shape, args.density_cfg
    )
    
    registrator.run_full_pipeline(
        args.mode, args.registration_type, args.save_registered_image, args.save_transforms
    )


if __name__ == "__main__":
    main()
