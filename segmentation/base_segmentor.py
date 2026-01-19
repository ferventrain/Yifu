import os
import time
from pathlib import Path
from typing import Tuple, List, Union, Optional
from abc import ABC, abstractmethod
import numpy as np
import tifffile
from tqdm import tqdm
import torch
import SimpleITK as sitk
import psutil
from cellpose.contrib.distributed_segmentation import distributed_eval

def compute_steps_for_sliding_window(patch_size: Tuple[int, ...],
                                     image_size: Tuple[int, ...],
                                     step_size: float) -> List[List[int]]:
    """
    Borrowed from Inference3D.py
    Compute the steps for the sliding window approach.
    """
    assert [i >= j for i, j in zip(image_size, patch_size)], "image size must be as large or larger than patch_size"
    assert 0 < step_size <= 1, 'step_size must be larger than 0 and smaller or equal to 1'

    # Our step width is patch_size*step_size at most, but can be narrower.
    target_step_sizes_in_voxels = [i * step_size for i in patch_size]

    num_steps = [int(np.ceil((i - k) / j)) + 1 for i, j, k in zip(image_size, target_step_sizes_in_voxels, patch_size)]

    steps = []
    for dim in range(len(patch_size)):
        # the highest step value for this dimension is
        max_step_value = image_size[dim] - patch_size[dim]
        if num_steps[dim] > 1:
            actual_step_size = max_step_value / (num_steps[dim] - 1)
        else:
            actual_step_size = 99999999999  # does not matter because there is only one step at 0

        steps_here = [int(np.round(actual_step_size * i)) for i in range(num_steps[dim])]
        steps.append(steps_here)

    return steps

import tempfile
import shutil

class LazyTiffStackLoader:
    """
    Lazy loader for a folder of TIFF files with Rolling Cache.
    Reads chunks of Z-slices into memory to reduce IO overhead.
    """
    def __init__(self, path: Path, cache_size: int = 64):
        self.path = Path(path)
        self.files = sorted(self.path.glob('*.tif*'))
        if not self.files:
            raise ValueError(f"No TIFF files found in {path}")
        
        # Read first file metadata
        with tifffile.TiffFile(self.files[0]) as tif:
            self.dtype = tif.pages[0].dtype
            self.shape = (len(self.files),) + tif.pages[0].shape
        
        self.ndim = 3
        
        # Cache settings
        self.cache_size = cache_size
        self.cache_start = -1
        self.cache_end = -1
        self.cache_data = None
        
        print(f"Initialized Lazy Loader for {len(self.files)} files. Shape: {self.shape}, Cache Size: {self.cache_size} slices")

    def _ensure_cache(self, z_start, z_stop):
        """
        Ensure the requested Z range is in cache.
        Optimized for sequential access (sliding window).
        """
        # Check if the requested range is fully within the current cache
        if (self.cache_data is not None and 
            z_start >= self.cache_start and 
            z_stop <= self.cache_end):
            return

        # If not in cache, load a new chunk
        # We try to load a chunk starting from z_start with length cache_size
        # But we must clip to file limits
        new_start = z_start
        new_end = min(self.shape[0], new_start + self.cache_size)
        
        # If the requested range is larger than cache_size, we just load what is needed
        # (Though in sliding window, patch_size_z should be < cache_size ideally)
        if z_stop > new_end:
            new_end = z_stop
            
        # print(f"DEBUG: Cache Miss for {z_start}-{z_stop}. Loading {new_start}-{new_end}...")
        
        stack = []
        for i in range(new_start, new_end):
            stack.append(tifffile.imread(self.files[i]))
            
        self.cache_data = np.array(stack)
        self.cache_start = new_start
        self.cache_end = new_end

    def __getitem__(self, key):
        # Handle simple integer indexing (z-slice)
        if isinstance(key, int):
            self._ensure_cache(key, key + 1)
            return self.cache_data[key - self.cache_start]
            
        # Handle slicing tuple (z, y, x)
        if not isinstance(key, tuple):
             # Fallback
             return tifffile.imread(self.files[key])
        
        # Extract slices
        z_slice = key[0]
        y_slice = key[1] if len(key) > 1 else slice(None)
        x_slice = key[2] if len(key) > 2 else slice(None)
        
        # Normalize Z slice
        z_start, z_stop, z_step = z_slice.indices(self.shape[0])
        
        # Ensure data is in memory
        self._ensure_cache(z_start, z_stop)
        
        # Calculate relative indices within the cache
        rel_start = z_start - self.cache_start
        rel_stop = z_stop - self.cache_start
        
        # Extract from cache
        # Note: self.cache_data is (D, H, W), so we slice on D first
        img_crop = self.cache_data[rel_start:rel_stop:z_step, y_slice, x_slice]
        
        return img_crop

class BaseSegmentor(ABC):
    """
    Abstract base class for 3D segmentation using sliding window inference.
    """
    def __init__(self):
        pass

    def load_image(self, path: Union[str, Path]) -> Union[np.ndarray, LazyTiffStackLoader]:
        """Load image from folder of TIFFs or single file"""
        path = Path(path)
        if path.is_dir():
            print(f"Initializing Lazy TIFF loader from {path}...")
            # Return lazy loader instead of full numpy array
            return LazyTiffStackLoader(path)
        elif path.suffix in ['.nii', '.nii.gz']:
            print(f"Loading NIfTI from {path}...")
            img = sitk.ReadImage(str(path))
            return sitk.GetArrayFromImage(img)
        else:
            print(f"Loading image from {path}...")
            return tifffile.imread(str(path))

    @abstractmethod
    def inference_batch(self, patches: List[np.ndarray], batch_size: int) -> List[np.ndarray]:
        """
        Perform inference on a batch of 3D patches.
        
        Args:
            patches: List of 3D numpy arrays (Z, Y, X)
            
        Returns:
            List of binary masks: List of 3D numpy arrays (Z, Y, X) of type uint8 (0 or 1)
        """
        pass

    def _estimate_batch_size(self, patch_size: Tuple[int, int, int], gpu_fraction: float = 0.8, multiplier: float = 150.0) -> int:
        """
        Estimate batch size based on GPU memory.
        """
        if not torch.cuda.is_available():
            print("CUDA not available, using default batch size 1")
            return 1
            
        try:
            # Get memory info (free, total) in bytes
            free_mem, total_mem = torch.cuda.mem_get_info()
            
            # Calculate target memory usage
            current_used = total_mem - free_mem
            target_limit = total_mem * gpu_fraction
            available_mem = target_limit - current_used
            
            if available_mem <= 0:
                print(f"Warning: Current memory usage ({current_used/1024**3:.2f}GB) already exceeds target ({target_limit/1024**3:.2f}GB). Using batch size 1.")
                return 1
            
            # Calculate patch memory
            # Assuming float32 input (4 bytes)
            pixels = np.prod(patch_size)
            patch_bytes = pixels * 4
            
            # Estimated memory per patch during inference
            est_patch_mem = patch_bytes * multiplier
            
            batch_size = int(available_mem / est_patch_mem)
            
            # Clamp batch size
            batch_size = max(1, batch_size)
            
            print(f"Auto-calculated batch size: {batch_size}")
            print(f"  Available Mem for Batch: {available_mem/1024**3:.2f} GB")
            print(f"  Est. Mem per Patch: {est_patch_mem/1024**3:.2f} GB (Multiplier: {multiplier})")
            
            return batch_size
            
        except Exception as e:
            print(f"Error estimating batch size: {e}. Using default 1.")
            return 1

    def predict_sliding_window(self, image: Union[np.ndarray, LazyTiffStackLoader], 
                             patch_size: Tuple[int, int, int], 
                             overlap: float = 0.5, 
                             batch_size: Optional[int] = None,
                             gpu_memory_fraction: float = 0.8,
                             mem_multiplier: float = 150.0) -> np.ndarray:
        """
        Run prediction using sliding window with batch processing.
        Uses disk-based storage (memmap) for the result to avoid RAM OOM.
        """
        print(f"Starting sliding window prediction. Image shape: {image.shape}, Patch size: {patch_size}")
        
        # Calculate steps
        steps = compute_steps_for_sliding_window(patch_size, image.shape, 1 - overlap)
        
        # Determine Batch Size
        if batch_size is None:
            batch_size = self._estimate_batch_size(patch_size, gpu_memory_fraction, mem_multiplier)
        
        # Generate all patch coordinates
        tiles = []
        for z in steps[0]:
            for y in steps[1]:
                for x in steps[2]:
                    tiles.append((z, y, x))
        
        total_tiles = len(tiles)
        print(f"Total tiles to process: {total_tiles}. Batch size: {batch_size}")
        
        # Initialize result array using Memory Mapping (Disk-based)
        # Create a temporary file for the memmap
        temp_dir = Path("temp_processing")
        temp_dir.mkdir(exist_ok=True)
        temp_file = temp_dir / f"temp_mask_{int(time.time())}.dat"
        
        print(f"Creating temporary memory-mapped file at {temp_file}...")
        result_mask = np.memmap(temp_file, dtype=np.uint8, mode='w+', shape=image.shape)
        
        # Process in batches
        pbar = tqdm(total=total_tiles, desc="Processing tiles")

        for i in range(0, total_tiles, batch_size):
            batch_tiles = tiles[i : i + batch_size]
            batch_patches = []
            batch_coords = []
            
            # Extract patches
            # If image is LazyTiffStackLoader, this reads from disk on demand
            for (z, y, x) in batch_tiles:
                lb_z, ub_z = z, z + patch_size[0]
                lb_y, ub_y = y, y + patch_size[1]
                lb_x, ub_x = x, x + patch_size[2]
                
                # Slicing the image (LazyLoader handles disk IO here)
                patch = image[lb_z:ub_z, lb_y:ub_y, lb_x:ub_x]
                batch_patches.append(patch)
                batch_coords.append((lb_z, ub_z, lb_y, ub_y, lb_x, ub_x))
        
            try:
                # Run Inference on Batch
                batch_masks = self.inference_batch(batch_patches, batch_size)
                
                # Merge results into Memmap
                # The OS handles the paging of the memmap file
                for mask, (lb_z, ub_z, lb_y, ub_y, lb_x, ub_x) in zip(batch_masks, batch_coords):
                    if mask.shape != (ub_z-lb_z, ub_y-lb_y, ub_x-lb_x):
                         print(f"Warning: Output shape {mask.shape} mismatch. Expected {(ub_z-lb_z, ub_y-lb_y, ub_x-lb_x)}")
                         continue
                         
                    # Read current region from disk (memmap)
                    current_region = result_mask[lb_z:ub_z, lb_y:ub_y, lb_x:ub_x]
                    # Compute max and write back to disk
                    result_mask[lb_z:ub_z, lb_y:ub_y, lb_x:ub_x] = np.maximum(current_region, mask)
                    
                # Optional: Flush changes to disk periodically (OS usually handles this)
                if i % (batch_size * 5) == 0:
                    result_mask.flush()
                    
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print("\nOOM Error detected! Try reducing batch size or GPU fraction.")
                    torch.cuda.empty_cache()
                    raise e
                else:
                    print(f"\nError processing batch starting at index {i}: {e}")
                    import traceback
                    traceback.print_exc()
            except Exception as e:
                print(f"\nError processing batch starting at index {i}: {e}")
                import traceback
                traceback.print_exc()
            
            pbar.update(len(batch_tiles))
        
        pbar.close()
        result_mask.flush()
        return result_mask

    def save_results(self, mask: np.ndarray, output_dir: Path):
        """Save result as TIFF stack and clean up memmap if applicable"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Saving results to {output_dir}...")
        
        # Save as TIFF stack (Z, Y, X) -> save slice by slice
        for i in tqdm(range(mask.shape[0]), desc="Saving TIFFs"):
            tifffile.imwrite(
                str(output_dir / f"mask_{i:04d}.tiff"),
                mask[i], # Reads from memmap
                compression='lzw'
            )
            
        # Cleanup memmap if it's a memmap
        if isinstance(mask, np.memmap):
            print("Cleaning up temporary memmap file...")
            try:
                filename = mask.filename
                # Delete the memmap object to close the file handle
                del mask
                # Remove the file
                if os.path.exists(filename):
                    os.remove(filename)
                    # Try to remove directory if empty
                    try:
                        os.rmdir(os.path.dirname(filename))
                    except:
                        pass
            except Exception as e:
                print(f"Warning: Could not clean up temp file: {e}")
