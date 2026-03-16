import os
import re
import argparse
import numpy as np
import tifffile
from pathlib import Path
from tqdm import tqdm

def parse_filename(filename):
    """
    Extract channel and Z-index from filename like YF2025102901_..._C1_Z0051.
    Returns (channel_str, z_index_str)
    """
    # Pattern to match Cx and Z####
    match = re.search(r'_(C\d+)_Z(\d+)', filename)
    if match:
        return match.group(1), match.group(2)
    return None, None

def process_channel_subtraction(root_path):
    root = Path(root_path)
    if not root.is_dir():
        print(f"Error: {root_path} is not a valid directory.")
        return

    # Find the ch0 folder
    # Looking for a folder that exactly matches "ch0" or starts with "ch0"
    ch0_folders = [f for f in root.iterdir() if f.is_dir() and (f.name == "ch0" or f.name.startswith("ch0_"))]
    if not ch0_folders:
        # Try any folder with "ch0" in it if the above fails
        ch0_folders = list(root.glob("*ch0*"))
    
    if not ch0_folders:
        print(f"Error: Could not find ch0 folder in {root_path}.")
        return
    
    ch0_dir = ch0_folders[0]
    print(f"Using {ch0_dir.name} as reference (C0).")

    # Map Z-index to file path in ch0
    ch0_files = {}
    for f in ch0_dir.glob("*.tif*"):
        _, z_idx = parse_filename(f.name)
        if z_idx:
            ch0_files[z_idx] = f
    
    if not ch0_files:
        print(f"Error: No valid TIFF files found in {ch0_dir}.")
        return
    
    print(f"Found {len(ch0_files)} reference files in {ch0_dir.name}.")

    # Traverse other folders (excluding ch0 and output folders)
    for folder in root.iterdir():
        if not folder.is_dir() or folder == ch0_dir or folder.name.endswith("_subtracted"):
            continue
        
        # Determine the target channel folder name
        # If folder name is "ch1", output is "ch1_subtracted"
        target_subtracted_dir = root / f"{folder.name}_subtracted"
        target_subtracted_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Processing folder: {folder.name} -> {target_subtracted_dir.name}")
        
        # Find files in current channel folder
        cx_files = list(folder.glob("*.tif*"))
        if not cx_files:
            print(f"No TIFF files found in {folder.name}, skipping.")
            continue
            
        for cx_file in tqdm(cx_files, desc=f"Subtracting {folder.name}"):
            _, z_idx = parse_filename(cx_file.name)
            
            if z_idx in ch0_files:
                c0_file = ch0_files[z_idx]
                
                try:
                    # Load images
                    img_cx = tifffile.imread(str(cx_file))
                    img_c0 = tifffile.imread(str(c0_file))
                    
                    # Ensure same shape
                    if img_cx.shape != img_c0.shape:
                        print(f"Warning: Shape mismatch for Z{z_idx}: {img_cx.shape} vs {img_c0.shape}. Skipping.")
                        continue
                    
                    # Perform subtraction: Cx - C0
                    # Use int32 to avoid overflow/underflow, then clip to 0 and original dtype max
                    # Most TIFFs are uint16 (0-65535) or uint8 (0-255)
                    dtype = img_cx.dtype
                    max_val = np.iinfo(dtype).max
                    
                    # Subtract and clip
                    subtracted = np.clip(img_cx.astype(np.int32) - img_c0.astype(np.int32), 0, max_val).astype(dtype)
                    
                    # Save result (filename remains the same)
                    output_path = target_subtracted_dir / cx_file.name
                    tifffile.imwrite(str(output_path), subtracted, compression='lzw')
                    
                except Exception as e:
                    print(f"Error processing {cx_file.name}: {e}")
            else:
                # No matching Z index in ch0
                # Optionally copy the file or skip
                # The user says "对每一个Cx-C0对应的图像", so we skip if C0 is missing.
                pass

    print("All processing complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Subtract ch0 image from other channel images based on Z-index.")
    parser.add_argument("path", type=str, help="Root path containing channel folders (ch0, ch1, etc.)")
    
    args = parser.parse_args()
    process_channel_subtraction(args.path)
