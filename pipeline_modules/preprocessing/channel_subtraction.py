import os
import re
import argparse
import numpy as np
import tifffile
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

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

def subtract_worker(cx_file, c0_file, output_path, compression='lzw'):
    """
    Worker function to perform subtraction for a single image pair.
    """
    try:
        # Load images
        img_cx = tifffile.imread(str(cx_file))
        img_c0 = tifffile.imread(str(c0_file))
        
        # Ensure same shape
        if img_cx.shape != img_c0.shape:
            return f"Error: Shape mismatch for {cx_file.name}"
        
        # Perform subtraction: Cx - C0
        dtype = img_cx.dtype
        max_val = np.iinfo(dtype).max
        
        # Subtract and clip using vectorized numpy operations
        subtracted = np.clip(img_cx.astype(np.int32) - img_c0.astype(np.int32), 0, max_val).astype(dtype)
        
        # Save result
        tifffile.imwrite(str(output_path), subtracted, compression=compression)
        return "success"
    except Exception as e:
        return f"Error processing {cx_file.name}: {str(e)}"

def process_channel_subtraction(root_path, max_workers=None, compression='lzw'):
    root = Path(root_path)
    if not root.is_dir():
        print(f"Error: {root_path} is not a valid directory.")
        return

    # Find the ch0 folder
    ch0_folders = [f for f in root.iterdir() if f.is_dir() and (f.name == "ch0" or f.name.startswith("ch0_"))]
    if not ch0_folders:
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

    if max_workers is None:
        max_workers = max(1, multiprocessing.cpu_count() // 2)
    
    # Windows has a limit of 61 for ProcessPoolExecutor's max_workers
    if os.name == 'nt' and max_workers > 61:
        print(f"Note: Capping max_workers to 61 due to Windows limitations (original: {max_workers})")
        max_workers = 61
    
    print(f"Using {max_workers} workers for parallel processing.")

    # Traverse other folders (excluding ch0 and output folders)
    for folder in root.iterdir():
        if not folder.is_dir() or folder == ch0_dir or folder.name.endswith("_subtracted"):
            continue
        
        target_subtracted_dir = root / f"{folder.name}_subtracted"
        target_subtracted_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Processing folder: {folder.name} -> {target_subtracted_dir.name}")
        
        # Find files in current channel folder
        cx_files = list(folder.glob("*.tif*"))
        if not cx_files:
            print(f"No TIFF files found in {folder.name}, skipping.")
            continue
            
        # Prepare tasks
        tasks = []
        for cx_file in cx_files:
            _, z_idx = parse_filename(cx_file.name)
            if z_idx in ch0_files:
                output_path = target_subtracted_dir / cx_file.name
                if not output_path.exists(): # Basic resume support
                    tasks.append((cx_file, ch0_files[z_idx], output_path, compression))
        
        # Execute tasks in parallel
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit tasks
            future_to_file = {executor.submit(subtract_worker, *task): task[0].name for task in tasks}
            
            # Use tqdm to monitor progress
            for future in tqdm(as_completed(future_to_file), total=len(tasks), desc=f"Subtracting {folder.name}"):
                result = future.result()
                if result != "success":
                    print(f"\n{result}")

    print("All processing complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parallel subtract ch0 image from other channel images.")
    parser.add_argument("path", type=str, help="Root path containing channel folders")
    parser.add_argument("--workers", type=int, default=None, help="Number of parallel processes (default: CPU_COUNT - 1)")
    parser.add_argument("--compression", type=str, default="lzw", choices=["lzw", "none", "zlib"], help="TIFF compression (default: lzw)")
    
    args = parser.parse_args()
    
    # Map 'none' to None for tifffile
    comp = None if args.compression == 'none' else args.compression
    
    process_channel_subtraction(args.path, max_workers=args.workers, compression=comp)
