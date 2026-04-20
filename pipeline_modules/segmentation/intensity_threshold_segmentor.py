import argparse
import zarr
import sys
from pathlib import Path
from tqdm import tqdm
import numpy as np
from skimage.filters import threshold_otsu
from scipy import ndimage
from numcodecs import Blosc

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

def segment_chunk(img, threshold, sigma, min_size, output_mode='label'):
    """
    Segment a single 3D chunk (or 2D slice)
    """
    # 1. Preprocessing (Smoothing)
    if sigma > 0:
        img = ndimage.gaussian_filter(img, sigma=sigma)
        
    # 2. Thresholding
    if threshold == 'otsu':
        try:
            thresh_val = threshold_otsu(img)
        except ValueError: # Empty image
            thresh_val = 0
    else:
        thresh_val = float(threshold)
        
    binary_mask = img > thresh_val
    
    # 3. Post-processing (Labeling + Size filtering)
    # Note: ndimage.label on a chunk might create edge artifacts when stitching back
    # But for simple thresholding segmentation it is often acceptable or handled later.
    # Ideally we should process with overlap, but for now let's keep it simple block-wise.
    
    # If we want instance segmentation (different IDs), labeling per chunk is problematic
    # because IDs will restart. 
    # If we just want binary mask (0/1), then it's fine.
    # The original code returned labeled matrix.
    # Let's return binary mask (0/255) or labeled if needed.
    # Given the output is a mask for density analysis, binary is usually what's needed first.
    # But density analysis counts "objects". 
    # If we label here chunk-wise, we can't count global objects easily.
    # However, 'intensity_threshold_segmentor' usually implies a semantic mask or local objects.
    
    labeled, num_features = ndimage.label(binary_mask)

    if min_size > 0:
        sizes = ndimage.sum(binary_mask, labeled, range(num_features + 1))
        mask_size = sizes < min_size
        remove_pixel = mask_size[labeled]
        labeled[remove_pixel] = 0

    if output_mode == 'binary':
        return (labeled > 0).astype(np.uint8)

    return labeled.astype(np.uint16)


def list_existing_chunk_indices(data_in):
    """List chunk indices that physically exist in the input store."""
    store = data_in.store
    array_path = getattr(data_in, "path", "")
    dim_sep = getattr(data_in, "_dimension_separator", ".")
    ndim = len(data_in.shape)

    prefix = f"{array_path}/" if array_path else ""
    existing = set()

    for raw_key in store.keys():
        key = str(raw_key)
        if prefix and not key.startswith(prefix):
            continue

        rel = key[len(prefix):] if prefix else key

        # Skip metadata and non-chunk files
        if rel in {".zarray", ".zattrs", ".zgroup", "zarr.json"}:
            continue
        if rel.startswith("."):
            continue

        parts = rel.split(dim_sep)
        if len(parts) != ndim:
            continue

        try:
            idx = tuple(int(p) for p in parts)
        except ValueError:
            continue

        existing.add(idx)

    return sorted(existing)

def main():
    parser = argparse.ArgumentParser(description="Run Intensity Threshold Segmentation (No Dask)")
    parser.add_argument('--input_zarr', required=True, help='Path to input .zarr directory')
    parser.add_argument('--output_zarr', required=True, help='Path for output .zarr directory')
    
    parser.add_argument('--threshold', default='otsu', help='Threshold value or "otsu"')
    parser.add_argument('--sigma', type=float, default=1.0, help='Gaussian smoothing sigma')
    parser.add_argument('--min_size', type=int, default=10, help='Minimum object size')
    parser.add_argument(
        '--output_mode',
        choices=['label', 'binary'],
        default='label',
        help='Output connected-component labels or a pure binary mask'
    )
    parser.add_argument(
        '--test',
        action='store_true',
        help='Smoke-test mode: only process chunks that physically exist in the input store'
    )
    
    args = parser.parse_args()
    
    # 1. Open Input Zarr
    print(f"Opening Input Zarr: {args.input_zarr}")
    # Use zarr.open which handles store and mode correctly
    root_in = zarr.open(args.input_zarr, mode='r')
    
    # Handle OME-Zarr structure (often in '0' group)
    if isinstance(root_in, zarr.Group) and '0' in root_in:
        data_in = root_in['0']
    else:
        data_in = root_in
        
    shape = data_in.shape
    chunks = data_in.chunks
    print(f"Input Shape: {shape}")
    print(f"Chunks: {chunks}")
    
    # 2. Prepare Output Zarr
    store_out = zarr.DirectoryStore(args.output_zarr)
    root_out = zarr.group(store=store_out, overwrite=True)
    
    compressor = Blosc(cname='zstd', clevel=5, shuffle=Blosc.SHUFFLE)
    output_dtype = np.uint8 if args.output_mode == 'binary' else np.uint16
    data_out = root_out.create_dataset('0', shape=shape, chunks=chunks, dtype=output_dtype, compressor=compressor)
    
    # Write metadata
    root_out.attrs['multiscales'] = [{
        'version': '0.4',
        'datasets': [{'path': '0'}]
    }]
    
    print(
        f"Running segmentation (Threshold={args.threshold}, Sigma={args.sigma}, "
        f"OutputMode={args.output_mode})..."
    )

    if args.test:
        existing_indices = list_existing_chunk_indices(data_in)
        if not existing_indices:
            print("No physical chunks found in input store. Nothing to process in --test mode.")
            return

        print(f"Test mode enabled: found {len(existing_indices)} physical chunks to process.")

        for idx in tqdm(existing_indices, desc="Segmenting existing chunks"):
            slices = []
            for axis, chunk_idx in enumerate(idx):
                start = chunk_idx * chunks[axis]
                stop = min(start + chunks[axis], shape[axis])
                slices.append(slice(start, stop))
            slices = tuple(slices)

            vol_chunk = np.asarray(data_in[slices])
            seg_chunk = segment_chunk(
                vol_chunk,
                args.threshold,
                args.sigma,
                args.min_size,
                output_mode=args.output_mode,
            )
            data_out[slices] = seg_chunk
    else:
        # 3. Process chunk-by-chunk along Z
        z_chunk_size = chunks[0]
        for z in tqdm(range(0, shape[0], z_chunk_size), desc="Segmenting Z-slices"):
            z_end = min(z + z_chunk_size, shape[0])

            # Note: this reads the full YX plane for each Z chunk.
            vol_chunk = data_in[z:z_end, :, :]
            seg_chunk = segment_chunk(
                vol_chunk,
                args.threshold,
                args.sigma,
                args.min_size,
                output_mode=args.output_mode,
            )
            data_out[z:z_end, :, :] = seg_chunk
        
    print(f"Segmentation complete. Saved to {args.output_zarr}")

if __name__ == "__main__":
    main()
