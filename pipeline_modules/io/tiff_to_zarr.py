import argparse
import os
import glob
from pathlib import Path
import numpy as np
import tifffile
import zarr
from tqdm import tqdm
import numcodecs

def convert_tiff_to_zarr(input_dir, output_path, chunk_size=(128, 256, 256), num_workers=4):
    """
    Convert a folder of TIFF files to a Zarr array with 3D chunking.
    Reads in batches to avoid RAM overflow.
    """
    input_path = Path(input_dir)
    tiff_files = sorted(input_path.glob('*.tif*'))
    
    if not tiff_files:
        raise ValueError(f"No TIFF files found in {input_dir}")
        
    print(f"Found {len(tiff_files)} TIFF files.")
    
    # Read first file to get dimensions and dtype
    sample = tifffile.imread(tiff_files[0])
    dtype = sample.dtype
    shape = (len(tiff_files),) + sample.shape
    
    print(f"Data Shape: {shape}")
    print(f"Data Type: {dtype}")
    print(f"Target Chunk Size: {chunk_size}")
    
    # Initialize Zarr array on disk
    # Using Blosc compressor for speed and compression
    compressor = numcodecs.Blosc(cname='zstd', clevel=3, shuffle=numcodecs.Blosc.SHUFFLE)
    
    store = zarr.DirectoryStore(output_path)
    z = zarr.open(store, mode='w', shape=shape, chunks=chunk_size, dtype=dtype, compressor=compressor)
    
    print(f"Created Zarr store at {output_path}")
    
    # Process in batches equal to the Z-chunk size
    # This aligns writes with Zarr chunks for maximum performance
    z_chunk = chunk_size[0]
    
    for i in tqdm(range(0, shape[0], z_chunk), desc="Converting to Zarr"):
        # Determine batch range
        start_idx = i
        end_idx = min(i + z_chunk, shape[0])
        
        # Read batch of TIFFs into memory
        batch_files = tiff_files[start_idx:end_idx]
        batch_data = []
        
        for f in batch_files:
            batch_data.append(tifffile.imread(f))
            
        batch_stack = np.stack(batch_data, axis=0)
        
        # Write to Zarr
        z[start_idx:end_idx, :, :] = batch_stack
        
    print("Conversion complete.")
    print(f"Zarr info:\n{z.info}")

def main():
    parser = argparse.ArgumentParser(description="Convert TIFF stack to Chunked Zarr")
    parser.add_argument('--input', required=True, help='Input folder containing TIFF files')
    parser.add_argument('--output', required=True, help='Output path for .zarr directory')
    parser.add_argument('--chunk_size', type=str, default="128,256,256", help='Chunk size z,y,x (default: 128,256,256)')
    
    args = parser.parse_args()
    
    try:
        cz, cy, cx = map(int, args.chunk_size.split(','))
        chunk_size = (cz, cy, cx)
    except:
        print("Invalid chunk size format. Using default.")
        chunk_size = (128, 256, 256)
        
    convert_tiff_to_zarr(args.input, args.output, chunk_size)

if __name__ == "__main__":
    main()
