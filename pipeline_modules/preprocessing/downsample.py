import os
import json
import argparse
from pathlib import Path
from typing import Tuple, List, Optional
import numpy as np
from tqdm import tqdm
import tifffile
import nibabel as nib
from scipy import ndimage


class ImageDownsampler:
    """Class for downsampling 3D TIFF stacks."""
    
    def __init__(self, resolution_config_path: Optional[str] = None, manual_factors: Optional[Tuple[float, float, float]] = None):
        """
        Args:
            resolution_config_path: Path to JSON config (config.json) containing resolution info
            manual_factors: Tuple of (z, y, x) downsample factors. If provided, overrides config.
        """
        self.manual_factors = manual_factors
        self.target_resolution = (1.0, 1.0, 1.0) # Default placeholder
        
        if manual_factors:
            self.downsample_factors = manual_factors
            print(f"Using manual downsample factors (z, y, x): {self.downsample_factors}")
        else:
            if not resolution_config_path:
                raise ValueError("Must provide either resolution_config_path or manual_factors")
                
            self.config_path = Path(resolution_config_path)
            if not self.config_path.exists():
                raise FileNotFoundError(f"Config file not found: {resolution_config_path}")
                
            with open(self.config_path, 'r') as f:
                self.config = json.load(f)
                
            # Parse resolution from main config structure (config.json) or legacy resolution.json
            if 'input' in self.config and 'resolution_xyz' in self.config['input']:
                self.source_resolution = self.config['input']['resolution_xyz']
            elif 'source_resolution' in self.config:
                self.source_resolution = self.config['source_resolution']
            else:
                 raise ValueError("Config must contain 'source_resolution' or 'input.resolution_xyz'")

            if 'preprocessing' in self.config and 'downsample' in self.config['preprocessing'] and 'target_resolution_xyz' in self.config['preprocessing']['downsample']:
                self.target_resolution = self.config['preprocessing']['downsample']['target_resolution_xyz']
            elif 'target_resolution' in self.config:
                self.target_resolution = self.config['target_resolution']
            else:
                raise ValueError("Config must contain 'target_resolution' or 'preprocessing.downsample.target_resolution_xyz'")
                
            self.downsample_factors = self._calculate_downsample_factors()
            
            print(f"Loaded config: {self.config}")
            print(f"Calculated downsample factors (z, y, x): {self.downsample_factors}")
    
    def _calculate_downsample_factors(self) -> Tuple[float, float, float]:
        """Calculate downsample factors for (z, y, x) axes."""
        # Config is typically [x, y, z], we need [z, y, x] for numpy
        factors = [
            source / target
            for source, target in zip(self.source_resolution, self.target_resolution)
        ]
        # Reverse to get (z, y, x)
        return tuple([round(f, 3) for f in factors[::-1]])
    
    def downsample_folder(self, 
                         input_folder: str, 
                         output_folder: Optional[str] = None,
                         is_mask: bool = False,
                         chunk_size: int = 100) -> None: 
        """
        Downsample a folder of TIFF files.
        
        Args:
            input_folder: Path to folder containing TIFFs
            output_folder: Path to save outputs. If None, defaults to input_folder + "_downsample"
            is_mask: If True, uses nearest neighbor interpolation
            chunk_size: Number of slices to process at once
        """
        input_path = Path(input_folder)
        if not input_path.exists():
            raise FileNotFoundError(f"Input folder not found: {input_folder}")

        # Determine output path
        if output_folder is None:
            suffix = "_downsample_mask" if is_mask else "_downsample"
            # If input is already xxx_mask, avoid xxx_mask_downsample_mask, just xxx_downsample_mask
            if is_mask and input_path.stem.endswith("_mask"):
                 output_path = input_path.parent / f"{input_path.stem.replace('_mask', '')}{suffix}"
            else:
                output_path = input_path.parent / f"{input_path.stem}{suffix}"
        else:
            output_path = Path(output_folder)
            
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Find files
        tiff_files = sorted(input_path.glob('*.tif*'))
        if not tiff_files:
            # Check if it's a single OME-TIFF
            ome_tiffs = list(input_path.glob('*.ome.tiff'))
            if ome_tiffs:
                print(f"Detected OME-TIFF: {ome_tiffs[0]}")
                self._process_single_file(ome_tiffs[0], output_path, is_mask)
                return
            else:
                raise ValueError(f"No TIFF files found in {input_folder}")
        
        print(f"Found {len(tiff_files)} TIFF files in {input_folder}")
        print(f"Processing as {'MASK' if is_mask else 'INTENSITY IMAGE'}")
        
        # Read metadata from first file
        first_img = tifffile.imread(tiff_files[0])
        original_shape = (len(tiff_files),) + first_img.shape
        dtype = first_img.dtype
        
        print(f"Original shape: {original_shape}")
        
        # Save original shape info
        with open(output_path / 'original_shape.json', 'w') as f:
            json.dump({'original_shape': original_shape}, f)
            
        # Choose interpolation method
        # order=0 is nearest neighbor (for masks), order=1 is linear (for images)
        interp_order = 0 if is_mask else 1
        
        # Process in chunks
        all_downsampled_slices = []
        
        num_chunks = (len(tiff_files) + chunk_size - 1) // chunk_size
        
        for chunk_idx in tqdm(range(num_chunks), desc="Downsampling chunks"):
            start_idx = chunk_idx * chunk_size
            end_idx = min(start_idx + chunk_size, len(tiff_files))
            
            # Load chunk
            chunk_files = tiff_files[start_idx:end_idx]
            chunk_data = [tifffile.imread(f) for f in chunk_files]
            chunk_volume = np.stack(chunk_data, axis=0)
            
            # Downsample chunk
            # Note: We use ndimage.zoom. 
            # Ideally for Z-axis, we should treat the whole stack, but for memory reasons we chunk.
            # This might introduce slight artifacts at chunk boundaries if Z factor != 1.
            # But for large downsampling (e.g. 10x), it's usually acceptable.
            
            downsampled_chunk = ndimage.zoom(
                chunk_volume, 
                self.downsample_factors, 
                order=interp_order,
                mode='nearest',
                prefilter=not is_mask # Disable prefilter for masks to avoid creating new values
            )
            
            # Collect results
            # We assume the result is small enough to fit in memory
            for i in range(downsampled_chunk.shape[0]):
                all_downsampled_slices.append(downsampled_chunk[i])
                
        # Stack full result
        full_downsampled_volume = np.stack(all_downsampled_slices, axis=0)
        print(f"Downsampled shape: {full_downsampled_volume.shape}")
        
        # Save TIFFs
        print("Saving TIFF stack...")
        for i in tqdm(range(full_downsampled_volume.shape[0]), desc="Writing TIFFs"):
            out_file = output_path / f"ds_{i:04d}.tiff"
            tifffile.imwrite(out_file, full_downsampled_volume[i].astype(dtype))
            
        # Save NIfTI
        print("Saving NIfTI volume...")
        nifti_path = output_path / "volume.nii.gz"
        self._save_as_nifti(full_downsampled_volume, nifti_path)
        print(f"Done! Results saved to {output_path}")

    def _process_single_file(self, file_path: Path, output_path: Path, is_mask: bool):
        """Handle single OME-TIFF or 3D TIFF file."""
        print(f"Reading {file_path}...")
        volume = tifffile.imread(file_path)
        
        if volume.ndim == 2:
            volume = volume[np.newaxis, ...]
            
        print(f"Original shape: {volume.shape}")
        
        interp_order = 0 if is_mask else 1
        
        downsampled_volume = ndimage.zoom(
            volume,
            self.downsample_factors,
            order=interp_order,
            mode='nearest',
            prefilter=not is_mask
        )
        
        print(f"Downsampled shape: {downsampled_volume.shape}")
        
        # Save TIFFs
        for i in tqdm(range(downsampled_volume.shape[0]), desc="Writing TIFFs"):
            out_file = output_path / f"ds_{i:04d}.tiff"
            tifffile.imwrite(out_file, downsampled_volume[i].astype(volume.dtype))
            
        # Save NIfTI
        nifti_path = output_path / "volume.nii.gz"
        self._save_as_nifti(downsampled_volume, nifti_path)

    def _save_as_nifti(self, volume: np.ndarray, output_path: Path) -> None:
        """Save numpy array as NIfTI with correct spacing."""
        # Nibabel expects (x, y, z) for NIfTI usually, but numpy is (z, y, x).
        # We need to transpose to (x, y, z) for nibabel to match standard orientation
        # Or we set the affine correctly. 
        # Standard: array index (k, j, i) -> (z, y, x). 
        # NIfTI affine usually maps (i, j, k) -> (x, y, z).
        # So we transpose volume from (z, y, x) to (x, y, z)
        
        volume_xyz = np.transpose(volume, (2, 1, 0))
        
        # Create affine matrix
        # Scaling factors on diagonal
        # spacing is (x, y, z)
        spacing = self.target_resolution
        affine = np.eye(4)
        affine[0, 0] = spacing[0]
        affine[1, 1] = spacing[1]
        affine[2, 2] = spacing[2]
        
        # Create image
        nifti_img = nib.Nifti1Image(volume_xyz, affine)
        
        # Save
        nib.save(nifti_img, output_path)


def main():
    parser = argparse.ArgumentParser(description="Downsample LSFM images or masks")
    parser.add_argument('--input_folder', required=True, help='Input folder containing TIFF files')
    parser.add_argument('--resolution_config', help='JSON file with resolution info')
    parser.add_argument('--factor', type=str, help='Manual downsample factors "z,y,x" (e.g. "0.5,0.5,0.5")')
    parser.add_argument('--output_folder', help='Optional output folder path')
    parser.add_argument('--is_mask', action='store_true', help='Set this flag if input is a mask (uses nearest neighbor)')
    parser.add_argument('--chunk_size', type=int, default=100, help='Z-slices per chunk for processing')
    
    args = parser.parse_args()
    
    try:
        manual_factors = None
        if args.factor:
            try:
                # Parse "0.5,0.5,0.5" into (0.5, 0.5, 0.5)
                parts = args.factor.split(',')
                if len(parts) == 1:
                     val = float(parts[0])
                     manual_factors = (val, val, val)
                elif len(parts) == 3:
                     manual_factors = tuple(float(x) for x in parts)
                else:
                    raise ValueError("Factor must be single number or 3 comma-separated numbers")
            except ValueError as e:
                print(f"Error parsing factor: {e}")
                exit(1)

        downsampler = ImageDownsampler(args.resolution_config, manual_factors)
        downsampler.downsample_folder(
            args.input_folder, 
            output_folder=args.output_folder,
            is_mask=args.is_mask,
            chunk_size=args.chunk_size
        )
    except Exception as e:
        print(f"Error: {e}")
        exit(1)

if __name__ == "__main__":
    main()
