import argparse
import zarr
import dask.array as da
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

from typing import Dict, Union, Tuple
import numpy as np
from skimage.filters import threshold_otsu
from scipy import ndimage
from src.core.base_segmentor import BaseSegmentor

class IntensityThresholdSegmentor(BaseSegmentor):
    """
    A simple intensity-based threshold segmentor.
    Useful for testing or segmenting very high-contrast structures (e.g., vessels).
    """
    
    def __init__(self, config: Dict):
        super().__init__(config)
        self.threshold = config.get('threshold', 'otsu')  # 'otsu' or float value
        self.min_size = config.get('min_size', 10)       # Remove small objects
        self.sigma = config.get('sigma', 1.0)            # Gaussian blur sigma
        
    def load_model(self):
        # No model to load
        print(f"IntensityThresholdSegmentor initialized. Threshold: {self.threshold}, Sigma: {self.sigma}")
        return True

    def predict_batch(self, batch_data: np.ndarray) -> np.ndarray:
        """
        Segment a batch of images using thresholding.
        batch_data: [Batch, Z, Y, X] or [Z, Y, X]
        """
        # Ensure numpy
        if hasattr(batch_data, 'compute'):
            batch_data = batch_data.compute()
            
        masks = []
        
        # Iterate over batch dimension (or Z if 3D)
        # Handle cases where batch_data might be 3D [Z, Y, X] directly
        if batch_data.ndim == 3:
            loop_range = batch_data.shape[0]
            indexer = lambda x: batch_data[x]
        elif batch_data.ndim == 4: # [B, Z, Y, X]
             # This depends on how map_blocks passes data. Usually it preserves dimensionality.
             # If map_blocks passes a chunk, it's (Z, Y, X) usually.
             loop_range = batch_data.shape[0]
             indexer = lambda x: batch_data[x]
        else:
            # Fallback
            loop_range = batch_data.shape[0]
            indexer = lambda x: batch_data[x]

        for i in range(loop_range):
            img = indexer(i)
            
            # 1. Preprocessing (Smoothing)
            if self.sigma > 0:
                img = ndimage.gaussian_filter(img, sigma=self.sigma)
                
            # 2. Thresholding
            if self.threshold == 'otsu':
                try:
                    thresh_val = threshold_otsu(img)
                except ValueError: # Empty image
                    thresh_val = 0
            else:
                thresh_val = float(self.threshold)
                
            binary_mask = img > thresh_val
            
            # 3. Post-processing (Labeling + Size filtering)
            labeled, num_features = ndimage.label(binary_mask)
            
            if self.min_size > 0:
                sizes = ndimage.sum(binary_mask, labeled, range(num_features + 1))
                mask_size = sizes < self.min_size
                remove_pixel = mask_size[labeled]
                labeled[remove_pixel] = 0
                
            masks.append(labeled.astype(np.uint16))
            
        return np.stack(masks)

    def predict_volume(self, volume_data: object) -> object:
        """
        Run on Dask array volume using map_blocks.
        """
        if not isinstance(volume_data, da.Array):
            raise ValueError("Input volume must be a Dask array")
            
        result = volume_data.map_blocks(
            self.predict_batch,
            dtype=np.uint16,
            chunks=volume_data.chunks
        )
        
        return result

def main():
    parser = argparse.ArgumentParser(description="Run Intensity Threshold Segmentation")
    parser.add_argument('--input_zarr', required=True, help='Path to input .zarr directory')
    parser.add_argument('--output_zarr', required=True, help='Path for output .zarr directory')
    
    # Config parameters (Can be passed individually or we could pass a json)
    # But since main.py constructs the command, arguments are fine.
    # The user wanted params in config file, but here we are CLI.
    # We will accept arguments that main.py parses from config.json and passes here.
    parser.add_argument('--threshold', default='otsu', help='Threshold value or "otsu"')
    parser.add_argument('--sigma', type=float, default=1.0, help='Gaussian smoothing sigma')
    parser.add_argument('--min_size', type=int, default=10, help='Minimum object size')
    
    args = parser.parse_args()
    
    # 1. Load Data
    print(f"Opening Input Zarr: {args.input_zarr}")
    input_zarr = zarr.open(args.input_zarr, mode='r')
    
    if isinstance(input_zarr, zarr.Group):
        if '0' in input_zarr:
            input_zarr = input_zarr['0']
            
    dask_arr = da.from_zarr(input_zarr)
    print(f"Input Shape: {dask_arr.shape}")
    
    # 2. Configure Segmentor
    config = {
        'threshold': float(args.threshold) if args.threshold != 'otsu' else 'otsu',
        'sigma': args.sigma,
        'min_size': args.min_size
    }
    
    segmentor = IntensityThresholdSegmentor(config)
    segmentor.load_model()
    
    # 3. Run Inference
    print("Running threshold segmentation (lazy)...")
    result_dask = segmentor.predict_volume(dask_arr)
    
    # 4. Save
    print(f"Saving to {args.output_zarr}...")
    da.to_zarr(result_dask, args.output_zarr, overwrite=True)
    
    print("Segmentation complete.")

if __name__ == "__main__":
    main()
