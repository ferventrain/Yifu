import argparse
import numpy as np
import zarr
from scipy import ndimage
from tqdm import tqdm


def upsample_mask(input_mask_zarr, output_mask_zarr, full_res_zarr, chunk_size=(128, 256, 256)):
    """Upsample a downsampled mask back to full resolution using nearest neighbor interpolation.
    
    Args:
        input_mask_zarr: Path to downsampled mask zarr
        output_mask_zarr: Path to output full resolution mask zarr
        full_res_zarr: Path to full resolution signal zarr (to get shape)
        chunk_size: Output chunk size
    """
    print(f"Loading downsampled mask: {input_mask_zarr}")
    ds_mask_z = zarr.open(input_mask_zarr, mode='r')
    if isinstance(ds_mask_z, zarr.Group) and '0' in ds_mask_z:
        ds_mask = ds_mask_z['0']
    else:
        ds_mask = ds_mask_z
    
    print(f"Loading full resolution zarr: {full_res_zarr}")
    full_z = zarr.open(full_res_zarr, mode='r')
    if isinstance(full_z, zarr.Group) and '0' in full_z:
        full_data = full_z['0']
    else:
        full_data = full_z
    
    full_shape = full_data.shape
    ds_shape = ds_mask.shape
    
    print(f"Downsampled shape: {ds_shape}")
    print(f"Full resolution shape: {full_shape}")
    
    zoom_factors = [full / ds for full, ds in zip(full_shape, ds_shape)]
    print(f"Zoom factors: {zoom_factors}")
    
    print("Creating output zarr...")
    store = zarr.DirectoryStore(str(output_mask_zarr))
    root = zarr.group(store=store, overwrite=True)
    
    out = root.create_dataset(
        '0',
        shape=full_shape,
        chunks=chunk_size,
        dtype=ds_mask.dtype,
        compressor=ds_mask.compressor
    )
    
    print("Upsampling by chunk...")
    chunk_size_z = chunk_size[0]
    n_chunks = (full_shape[0] + chunk_size_z - 1) // chunk_size_z
    
    for i in tqdm(range(n_chunks), desc="Upsampling chunks"):
        start_z = i * chunk_size_z
        end_z = min((i + 1) * chunk_size_z, full_shape[0])
        
        ds_start_z = int(np.floor(start_z / zoom_factors[0]))
        ds_end_z = int(np.ceil(end_z / zoom_factors[0]))
        
        ds_chunk = ds_mask[ds_start_z:ds_end_z, :, :]
        
        zoom_z = (end_z - start_z) / ds_chunk.shape[0]
        zoom_xy = zoom_factors[1:]
        zoom = (zoom_z, zoom_xy[0], zoom_xy[1])
        
        upsampled = ndimage.zoom(ds_chunk, zoom, order=0, mode='nearest')
        
        out[start_z:end_z, :, :] = upsampled
    
    print(f"Upsampling complete! Output saved to: {output_mask_zarr}")
    print(f"Final shape: {out.shape}")


def main():
    parser = argparse.ArgumentParser(description="Upsample downsampled mask back to full resolution")
    parser.add_argument('--input_mask_zarr', required=True, help='Path to input downsampled mask zarr')
    parser.add_argument('--output_mask_zarr', required=True, help='Path to output full resolution mask zarr')
    parser.add_argument('--full_res_zarr', required=True, help='Path to full resolution signal zarr (provides output shape)')
    parser.add_argument('--chunk_size', default="128,256,256", help='Output chunk size z,y,x')
    
    args = parser.parse_args()
    
    try:
        cz, cy, cx = map(int, args.chunk_size.split(','))
        chunk_size = (cz, cy, cx)
    except:
        chunk_size = (128, 256, 256)
    
    upsample_mask(
        args.input_mask_zarr,
        args.output_mask_zarr,
        args.full_res_zarr,
        chunk_size=chunk_size
    )


if __name__ == "__main__":
    main()
