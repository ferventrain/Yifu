import argparse
import pandas as pd
import numpy as np
import tifffile
from skimage.draw import disk
from pathlib import Path

def generate_spot_image(csv_path, tiff_path, output_path, radius=3):
    """
    Generate an image with circles at coordinates specified in CSV.
    The output image has the same (Y, X) dimensions as the reference TIFF.
    """
    # 1. Load CSV
    print(f"Loading CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Check columns
    if 'y' not in df.columns or 'x' not in df.columns:
        raise ValueError("CSV must contain 'y' and 'x' columns")
    
    # 2. Load Reference TIFF to get shape
    print(f"Loading Reference TIFF: {tiff_path}")
    ref_img = tifffile.imread(tiff_path)
    
    # Handle dimensions
    # If it's 3D (Z, Y, X), we assume we want a 2D projection or the CSV is for a specific slice.
    # But usually spotiflow results on 2D images are 2D coordinates.
    # We will use the last two dimensions for Y, X.
    if ref_img.ndim >= 2:
        h, w = ref_img.shape[-2], ref_img.shape[-1]
    else:
        raise ValueError(f"Reference image must be at least 2D. Got shape {ref_img.shape}")
        
    print(f"Reference Image Shape: {ref_img.shape} -> Using Output Shape: ({h}, {w})")
    
    # 3. Create empty image
    # Using uint8 (0-255)
    output_img = np.zeros((h, w), dtype=np.uint8)
    
    # 4. Draw circles
    print(f"Drawing {len(df)} spots with radius {radius}...")
    
    # Filter points that are out of bounds (just in case)
    valid_points = 0
    
    for idx, row in df.iterrows():
        y, x = row['y'], row['x']
        
        # Draw disk
        # disk center is (row, col) -> (y, x)
        rr, cc = disk((y, x), radius, shape=(h, w))
        
        output_img[rr, cc] = 255
        valid_points += 1
            
    # 5. Save output
    print(f"Saving output to: {output_path}")
    # Ensure output directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    tifffile.imwrite(output_path, output_img, compression='lzw')
    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate spot visualization from CSV coordinates")
    parser.add_argument('--csv', required=True, help='Path to spotiflow CSV results')
    parser.add_argument('--tiff', required=True, help='Path to reference TIFF image (to get dimensions)')
    parser.add_argument('--output', required=True, help='Path to output TIFF image')
    parser.add_argument('--radius', type=int, default=5, help='Radius of the circles (default: 5)')
    
    args = parser.parse_args()
    
    generate_spot_image(args.csv, args.tiff, args.output, args.radius)
