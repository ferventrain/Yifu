import argparse
import os
import zarr
import numpy as np
import tifffile
from tqdm import tqdm
import torch
import torch.nn.functional as F

def richardson_lucy_gpu(image, psf, num_iter=10):
    """
    Richardson-Lucy deconvolution using PyTorch for GPU acceleration.
    
    Args:
        image (numpy.ndarray): Input 3D image chunk (Z, Y, X).
        psf (numpy.ndarray): Point Spread Function (3D kernel).
        num_iter (int): Number of iterations.
        
    Returns:
        numpy.ndarray: Deconvolved image (on CPU).
    """
    # Check for GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == 'cpu':
        print("Warning: CUDA not available. Running on CPU (slow).")

    # Convert to Tensor and move to device
    # Image shape: (Z, Y, X) -> (1, 1, Z, Y, X) for 3D convolution
    img_tensor = torch.from_numpy(image.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
    psf_tensor = torch.from_numpy(psf.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
    
    # Normalize PSF
    psf_tensor = psf_tensor / psf_tensor.sum()
    
    # Flip PSF for convolution (correlation vs convolution)
    # Standard RL uses convolution with flipped PSF for the backward step?
    # In PyTorch, conv3d is correlation. True convolution requires flipping kernel.
    # But usually for symmetric PSF it doesn't matter. Let's flip to be safe.
    psf_flipped = torch.flip(psf_tensor, [2, 3, 4])
    
    # Initial estimate (usually the input image)
    estimate = img_tensor.clone()
    
    # RL Iterations
    # O_k+1 = O_k * ( (I / (O_k * PSF)) * PSF_flipped )
    # * denotes convolution
    
    eps = 1e-6
    
    for i in range(num_iter):
        # 1. Forward projection: O_k * PSF
        # Padding is crucial. We use 'same' padding logic manually or via padding arg.
        # PyTorch conv3d doesn't support 'same' natively for asymmetric kernels easily, 
        # but for odd kernels we can calculate padding.
        pad_z = psf.shape[0] // 2
        pad_y = psf.shape[1] // 2
        pad_x = psf.shape[2] // 2
        
        # We need to pad input to handle boundaries
        # mode='replicate' is often better than zero padding for deconv to avoid edge artifacts
        
        # Forward convolution
        blurred_estimate = F.conv3d(
            F.pad(estimate, (pad_x, pad_x, pad_y, pad_y, pad_z, pad_z), mode='replicate'),
            psf_tensor
        )
        
        # 2. Relative blur: I / (O_k * PSF)
        # Avoid division by zero
        relative_blur = img_tensor / (blurred_estimate + eps)
        
        # 3. Backward projection: (I / ...) * PSF_flipped
        # Correlation with PSF is equivalent to Convolution with PSF_flipped
        error_back_projected = F.conv3d(
            F.pad(relative_blur, (pad_x, pad_x, pad_y, pad_y, pad_z, pad_z), mode='replicate'),
            psf_flipped
        )
        
        # 4. Update: O_k+1 = O_k * ...
        estimate = estimate * error_back_projected
        
    return estimate.squeeze().cpu().numpy()

def generate_gaussian_psf(shape=(31, 31, 31), sigma=(2.0, 2.0, 2.0)):
    """Generate a 3D Gaussian PSF."""
    z, y, x = np.meshgrid(
        np.arange(-shape[0]//2 + 1, shape[0]//2 + 1),
        np.arange(-shape[1]//2 + 1, shape[1]//2 + 1),
        np.arange(-shape[2]//2 + 1, shape[2]//2 + 1),
        indexing='ij'
    )
    psf = np.exp(-(z**2/(2*sigma[0]**2) + y**2/(2*sigma[1]**2) + x**2/(2*sigma[2]**2)))
    return psf / psf.sum()

def process_zarr_deconv(input_zarr_path, output_folder, num_iter=10, psf_sigma=(2.0, 2.0, 2.0), test_mode=False):
    """
    Process Zarr chunks with RL deconvolution and save as TIFF slices dynamically.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    print(f"Opening Zarr: {input_zarr_path}")
    
    try:
        # Open Zarr
        z_root = zarr.open(input_zarr_path, mode='r')
        
        # Handle OME-Zarr (usually data is in '0')
        if isinstance(z_root, zarr.Group) and '0' in z_root:
            dset = z_root['0']
        else:
            dset = z_root
            
        print(f"Data shape: {dset.shape}")
        print(f"Chunks: {dset.chunks}")
        
        # Generate PSF
        print("Generating PSF...")
        # PSF size should be odd
        psf = generate_gaussian_psf(shape=(15, 31, 31), sigma=psf_sigma)
        
        # Iterate over chunks (Z-axis)
        # We process slab by slab to save memory but allow GPU batching
        z_chunk_size = dset.chunks[0]
        total_z = dset.shape[0]
        
        if test_mode:
            # Calculate the start of the chunk in the middle of the volume
            mid_z = total_z // 2
            # Align to chunk boundary
            z_starts = [(mid_z // z_chunk_size) * z_chunk_size]
            print(f"🧪 TEST MODE: Processing only one slab around Z={mid_z} (Chunk start: {z_starts[0]})")
        else:
            z_starts = range(0, total_z, z_chunk_size)

        print(f"Starting Deconvolution (Iterations: {num_iter})...")
        
        for z_start in tqdm(z_starts, desc="Processing Chunks"):
            z_end = min(z_start + z_chunk_size, total_z)
            
            # Read chunk
            # Note: For deconvolution, we strictly need overlap (padding) from neighbors
            # to avoid block boundary artifacts.
            # Here we implement a simple version with minimal overlap read.
            overlap = psf.shape[0] // 2
            
            read_start = max(0, z_start - overlap)
            read_end = min(total_z, z_end + overlap)
            
            chunk_data = dset[read_start:read_end, :, :]
            
            # Run Deconvolution on GPU
            deconv_chunk = richardson_lucy_gpu(chunk_data, psf, num_iter=num_iter)
            
            # Crop back to valid range (remove overlap)
            valid_start = z_start - read_start
            valid_end = valid_start + (z_end - z_start)
            
            valid_data = deconv_chunk[valid_start:valid_end]
            
            # Save valid slices immediately
            for i in range(valid_data.shape[0]):
                z_global = z_start + i
                prefix = "test_" if test_mode else "deconv_"
                output_filename = f"{prefix}Z{z_global:04d}.tif"
                output_path = os.path.join(output_folder, output_filename)
                
                # Convert back to uint16 (assuming input was uint16)
                # RL output is float, we need to clip and cast
                slice_data = np.clip(valid_data[i], 0, 65535).astype(np.uint16)
                
                tifffile.imwrite(output_path, slice_data, compression='lzw')
                
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GPU Richardson-Lucy Deconvolution for Zarr")
    parser.add_argument('--input', required=True, help='Path to input .zarr file/folder')
    parser.add_argument('--output', required=True, help='Output directory for TIFFs')
    parser.add_argument('--iter', type=int, default=10, help='Number of iterations')
    parser.add_argument('--sigma', type=float, nargs=3, default=[1.5, 1.5, 1.5], help='PSF Sigma (z y x)')
    parser.add_argument('--test', action='store_true', help='Run test mode: process only one slab in the middle')
    
    args = parser.parse_args()
    
    process_zarr_deconv(args.input, args.output, args.iter, tuple(args.sigma), args.test)
