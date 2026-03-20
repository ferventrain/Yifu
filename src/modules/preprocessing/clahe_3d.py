import cv2
import numpy as np
import os
import argparse
import tifffile
from pathlib import Path
from tqdm import tqdm

class CLAHE3D:
    """
    A class to handle CLAHE for 2D images and 3D volumes.
    Optimized for memory efficiency and supports 16-bit images.
    """
    def __init__(self, clip_limit=2.0, tile_grid_size=(64, 64)):
        """
        Initialize CLAHE settings.
        
        Parameters:
        - clip_limit: Threshold for contrast limiting. 
                     For 16-bit, values around 1.0-5.0 are common.
        - tile_grid_size: Number of tiles in (rows, cols) for each slice.
        """
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
        self.clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)

    def process_2d(self, img):
        """Apply CLAHE to a single 2D image."""
        if img is None:
            return None
        return self.clahe.apply(img)

    def process_3d_slice_by_slice(self, volume):
        """
        Apply CLAHE to a 3D volume slice-by-slice (Z-direction).
        This is memory efficient for large volumes.
        
        Parameters:
        - volume: 3D numpy array (Z, Y, X)
        """
        z_depth = volume.shape[0]
        result = np.zeros_like(volume)
        
        for z in range(z_depth):
            result[z] = self.clahe.apply(volume[z])
            if (z + 1) % 10 == 0 or z == z_depth - 1:
                print(f"Processed slice {z+1}/{z_depth}", end='\r')
        print("\n3D slice-by-slice CLAHE completed.")
        return result

    def process_folder(self, input_dir, output_dir, pattern="*.tif"):
        """
        Process a folder of images one by one. 
        Useful for ultra-large volumes stored as image sequences.
        """
        import glob
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        files = sorted(glob.glob(os.path.join(input_dir, pattern)))
        print(f"Found {len(files)} files in {input_dir}")
        
        for i, f in enumerate(files):
            filename = os.path.basename(f)
            img = cv2.imread(f, cv2.IMREAD_UNCHANGED)
            if img is None:
                print(f"Skip {filename}: could not read.")
                continue
                
            res = self.process_2d(img)
            cv2.imwrite(os.path.join(output_dir, filename), res)
            
            if (i + 1) % 10 == 0 or i == len(files) - 1:
                print(f"Processed {i+1}/{len(files)} files", end='\r')
        print("\nFolder processing completed.")

    def process_file(self, input_path, output_path):
        """
        Load a file (2D or 3D TIF), apply CLAHE, and save.
        """
        print(f"Loading {input_path}...")
        
        # Check file extension
        ext = os.path.splitext(input_path)[1].lower()
        
        # Load image/volume
        # For large 3D volumes, tifffile is recommended. 
        # But we use OpenCV here as it's in the requirements.
        img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
        
        if img is None:
            # Try imreadmulti for multi-page TIF (3D volumes)
            ret, vol_list = cv2.imreadmulti(input_path, [], cv2.IMREAD_UNCHANGED)
            if not ret or not vol_list:
                raise FileNotFoundError(f"Could not read {input_path}. Ensure it's a valid image or TIF stack.")
            img = np.array(vol_list)
            print(f"Loaded 3D volume with shape {img.shape}")
        else:
            print(f"Loaded 2D image with shape {img.shape}")

        # Process based on dimension
        if len(img.shape) == 2:
            result = self.process_2d(img)
        elif len(img.shape) == 3:
            # Check if it's a color image (Y, X, C) or a grayscale volume (Z, Y, X)
            if img.shape[2] in [3, 4]:
                print("Processing color image in LAB space...")
                lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                l_clahe = self.process_2d(l)
                lab = cv2.merge((l_clahe, a, b))
                result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            else:
                result = self.process_3d_slice_by_slice(img)
        else:
            raise ValueError(f"Unsupported image shape: {img.shape}")

        print(f"Saving result to {output_path}...")
        
        # Save based on dimension
        if len(result.shape) == 3 and result.shape[0] > 1 and not (result.shape[2] in [3, 4]):
            # Multi-page TIF saving
            cv2.imwritemulti(output_path, list(result))
        else:
            cv2.imwrite(output_path, result)
        
        print("Done.")
        return result

def main():
    parser = argparse.ArgumentParser(description="Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to images.")
    parser.add_argument("--input", required=True, help="Input directory or file.")
    parser.add_argument("--output", required=True, help="Output directory for processed images.")
    parser.add_argument("--clip_limit", type=float, default=2.0, help="Contrast limit. Default: 2.0")
    parser.add_argument("--tile_grid_size", type=int, default=8, help="Tile grid size (e.g. 8 for 8x8). Default: 8")
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    processor = CLAHE3D(clip_limit=args.clip_limit, tile_grid_size=(args.tile_grid_size, args.tile_grid_size))
    
    if input_path.is_file():
        files = [input_path]
    else:
        files = sorted(list(input_path.glob("*.tif*")))
        
    print(f"Applying CLAHE to {len(files)} files...")
    for f in tqdm(files):
        img = tifffile.imread(str(f))
        processed = processor.process_2d(img)
        tifffile.imwrite(str(output_dir / f.name), processed, compression='lzw')

if __name__ == "__main__":
    main()
