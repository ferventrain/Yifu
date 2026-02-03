import argparse
import os
import glob
from pathlib import Path
import zarr
import tifffile
import numpy as np
from numcodecs import Blosc
from tqdm import tqdm
import dask.array as da
import dask

def convert_tiff_to_zarr(input_dir, output_zarr, chunk_size=(128, 256, 256), compressor='default'):
    """
    Convert a folder of TIFF files to OME-Zarr.
    Handles large datasets by reading and writing lazily/chunked.
    """
    input_path = Path(input_dir)
    output_path = Path(output_zarr)
    
    print(f"Searching for TIFFs in {input_path}...")
    tiff_files = sorted(list(input_path.glob('*.tif*')))
    
    if not tiff_files:
        raise ValueError(f"No TIFF files found in {input_dir}")
        
    print(f"Found {len(tiff_files)} slices.")
    
    # Read first slice to get shape/dtype
    sample = tifffile.imread(tiff_files[0])
    dtype = sample.dtype
    shape = (len(tiff_files),) + sample.shape
    
    print(f"Volume Shape: {shape}")
    print(f"Data Type: {dtype}")
    
    # Create Zarr array
    store = zarr.DirectoryStore(str(output_path))
    root = zarr.group(store=store, overwrite=True)
    
    if compressor == 'default':
        compressor = Blosc(cname='zstd', clevel=5, shuffle=Blosc.SHUFFLE)
        
    # Initialize array
    z = root.create_dataset('0', shape=shape, chunks=chunk_size, dtype=dtype, compressor=compressor)
    
    # Write attributes (OME-Zarr style basic)
    root.attrs['multiscales'] = [{
        'version': '0.4',
        'datasets': [{'path': '0'}]
    }]
    
    # Processing in chunks (Z-axis)
    # We load N slices that match the Z-chunk size to minimize I/O fragmentation
    z_chunk = chunk_size[0]
    
    for i in tqdm(range(0, shape[0], z_chunk), desc="Writing to Zarr"):
        end_i = min(i + z_chunk, shape[0])
        
        # Load batch of TIFFs
        # Using dask.delayed or simple loop? Simple loop is safer for memory control here.
        stack = []
        for f in tiff_files[i:end_i]:
            stack.append(tifffile.imread(f))
            
        chunk_data = np.stack(stack)
        
        # Write to Zarr
        z[i:end_i] = chunk_data
        
    print(f"Conversion complete: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert TIFF folder to Zarr")
    parser.add_argument('--input', required=True, help='Input TIFF folder')
    parser.add_argument('--output', required=True, help='Output .zarr path')
    parser.add_argument('--chunk_size', default="128,256,256", help='Chunk size z,y,x')
    
    args = parser.parse_args()
    
    try:
        cz, cy, cx = map(int, args.chunk_size.split(','))
        chunk_size = (cz, cy, cx)
    except:
        chunk_size = (128, 256, 256)
        
    convert_tiff_to_zarr(args.input, args.output, chunk_size)
