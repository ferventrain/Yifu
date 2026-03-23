import cv2
import numpy as np
import argparse
import tifffile
from scipy.ndimage import gaussian_filter as scipy_gaussian_filter
import dask.array as da
try:
    from dask_image.ndfilters import gaussian_filter as dask_gaussian_filter
except ImportError:
    dask_gaussian_filter = None
from pathlib import Path
from tqdm import tqdm

def estimate_d0_adaptive(img, energy_threshold=0.98):
    """
    Automatically estimate d0 by analyzing the energy distribution of the 
    downsampled image's spectrum.
    
    Parameters:
    - img: Input image.
    - energy_threshold: Cumulative energy threshold (0.95-0.99). 
                        Higher threshold leads to a larger d0.
    """
    # 1. Extreme downsampling for speed (256x256 is enough for frequency analysis)
    small_img = cv2.resize(np.float64(img), (256, 256))
    
    # 2. Compute spectrum in log domain
    f = np.fft.fft2(np.log1p(small_img))
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = np.abs(fshift)**2 # Power spectrum
    
    # 3. Create distance matrix from center
    rows, cols = magnitude_spectrum.shape
    crow, ccol = rows // 2 , cols // 2
    y, x = np.ogrid[-crow:rows-crow, -ccol:cols-ccol]
    dist = np.sqrt(x*x + y*y)
    
    # 4. Sort energy by distance
    dist_flat = dist.ravel()
    mag_flat = magnitude_spectrum.ravel()
    idx = np.argsort(dist_flat)
    
    dist_sorted = dist_flat[idx]
    mag_sorted = mag_flat[idx]
    
    # 5. Find radius where cumulative energy reaches threshold
    cum_energy = np.cumsum(mag_sorted)
    total_energy = cum_energy[-1]
    
    cutoff_idx = np.searchsorted(cum_energy, total_energy * energy_threshold)
    d0_small = dist_sorted[cutoff_idx]
    
    # 6. Map back to original image scale
    scale = img.shape[0] / 256.0   
    d0_estimated = d0_small * scale
    
    # 7. Sanity Check / Clamping
    # d0 shouldn't be too small (doing nothing) or too large (destroying signal)
    # Typical range: 0.5% to 20% of image dimension
    min_dim = min(img.shape)
    lower_bound = max(30.0, min_dim * 0.005) # At least 30 pixels, or 0.5%
    upper_bound = min_dim * 0.2           # Max 20% of image size
    
    d0_clamped = np.clip(d0_estimated, lower_bound, upper_bound)
    return d0_clamped

def homomorphic_filter(img, d0=None, rl=0.5, rh=2.0, c=1.0, return_uint8=False, verbose=False):
    """
    Apply Homomorphic Filtering to an image.
    
    Parameters:
    - img: Input grayscale image (can be uint8, uint16, or float).
    - d0: Cutoff frequency. If None, it will be estimated automatically.
    - rl: Low frequency gain (gamma_L), usually < 1.
    - rh: High frequency gain (gamma_H), usually > 1.
    - c: Sharpness constant.
    - return_uint8: If True, output is clipped to 0-255 uint8.
    - verbose: If True, print estimated d0 (default: False).
    
    Returns:
    - filtered_img: The enhanced image.
    """
    # Auto-estimate d0 if not provided
    if d0 is None:
        d0 = estimate_d0_adaptive(img)
        if verbose:
            print(f"d0: {d0}")
    
    # Get original range if it's uint
    orig_dtype = img.dtype
    
    # 1. Convert to float and add 1 to avoid log(0)
    img_float = np.float64(img) + 1.0
    
    # 2. Log transform
    img_log = np.log(img_float)
    
    # 3. FFT
    rows, cols = img.shape
    img_fft = np.fft.fft2(img_log)
    img_fft_shift = np.fft.fftshift(img_fft)
    
    # 4. Create High-Pass Filter (Butterworth-like or Gaussian-based)
    M, N = rows, cols
    u = np.arange(M)
    v = np.arange(N)
    u, v = np.meshgrid(u, v, indexing='ij')
    
    # Center coordinates
    u0, v0 = M // 2, N // 2
    
    # Distance from center
    Duv = np.sqrt((u - u0)**2 + (v - v0)**2)
    
    # Homomorphic filter function: H(u,v) = (rh - rl) * (1 - exp(-c * (D^2 / D0^2))) + rl
    Huv = (rh - rl) * (1 - np.exp(-c * (Duv**2 / (d0**2)))) + rl
    
    # 5. Apply filter in frequency domain
    filtered_fft_shift = img_fft_shift * Huv
    
    # 6. Inverse FFT
    filtered_fft = np.fft.ifftshift(filtered_fft_shift)
    img_filtered_log = np.fft.ifft2(filtered_fft)
    img_filtered_log = np.real(img_filtered_log)
    
    # 7. Exponential transform (Inverse log)
    img_filtered = np.exp(img_filtered_log) - 1.0
    
    # 8. Handle output
    if return_uint8:
        img_filtered = np.clip(img_filtered, 0, 255)
        return np.uint8(img_filtered)
    
    # If not uint8, we might want to scale it back to the original range or keep as float
    # For dark signals, returning float is best as it preserves all precision.
    return img_filtered

def homomorphic_filter_3d_adaptive(volume, sigma_xy=None, voxel_ratio=1.111, rl=0.5, rh=1.5, sample_rate=0.005):
    """
    3D Adaptive Homomorphic Filtering using Spatial Gaussian Approximation.
    Supports Dask arrays for memory-efficient out-of-core processing.
    
    Parameters:
    - volume: 3D array (numpy, zarr, or dask array)
    - sigma_xy: Smoothness scale for XY plane.
    - voxel_ratio: Anisotropy correction (PixelSize_Z / PixelSize_XY).
    - rl: Low frequency gain (illumination)
    - rh: High frequency gain (reflectance/signal)
    - sample_rate: Fraction of slices to use for parameter estimation (0.0-1.0).
    """
    # 1. Robust automatic estimation of sigma_xy
    if sigma_xy is None:
        print("Estimating sigma_xy from volume statistics...")
        z_depth = volume.shape[0]
        # Sample slices evenly across the volume
        num_samples = max(1, int(z_depth * sample_rate))
        indices = np.linspace(0, z_depth - 1, num_samples, dtype=int)
        
        d0_list = []
        for idx in indices:
            # Load single slice into memory for estimation
            try:
                slice_data = np.array(volume[idx])
                d0 = estimate_d0_adaptive(slice_data)
                d0_list.append(d0)
            except Exception as e:
                pass
        
        if not d0_list:
            d0_avg = 30.0
        else:
            d0_avg = np.median(d0_list)
            
        print(f"Estimated average d0: {d0_avg:.2f} from {len(d0_list)} slices.")
        sigma_xy = volume.shape[2] / (2 * np.pi * d0_avg)
        print(f"Calculated robust sigma_xy: {sigma_xy:.2f}")

    # 2. Anisotropy correction
    sigma_z = sigma_xy / voxel_ratio
    sigmas = (sigma_z, sigma_xy, sigma_xy)
    print(f"Applying 3D Gaussian with sigmas (Z, Y, X): {sigmas}")

    # 3. Handle as Dask array for memory efficiency
    if not isinstance(volume, da.Array):
        # Convert to dask array if not already one
        # Use chunks that align with Zarr or at least reasonable slice chunks
        if hasattr(volume, 'chunks'):
            chunks = volume.chunks
        else:
            chunks = (1, volume.shape[1], volume.shape[2])
        vol_da = da.from_array(volume, chunks=chunks)
    else:
        vol_da = volume

    # 4. Log transform lazily
    vol_log = da.log1p(vol_da.astype(np.float32))
    
    # 5. 3D Gaussian filtering in log domain
    # This computes the low-frequency illumination component
    if dask_gaussian_filter is not None:
        low_freq = dask_gaussian_filter(vol_log, sigma=sigmas)
    else:
        # Fallback if dask-image is missing (less memory efficient)
        print("Warning: dask-image not found. Falling back to scipy.ndimage (may use lots of RAM).")
        low_freq = vol_log.map_overlap(scipy_gaussian_filter, 
                                       depth=tuple(int(3*s + 1) for s in sigmas),
                                       sigma=sigmas)
    
    # 6. Homomorphic enhancement lazily
    res_log = rh * (vol_log - low_freq) + rl * low_freq
    
    # 7. Exponential restoration lazily
    result = da.expm1(res_log)
    
    return result

def main():
    parser = argparse.ArgumentParser(description="Apply Homomorphic Filter to a single image or a folder of TIFFs.")
    parser.add_argument("--input", required=True, help="Input TIFF file or directory.")
    parser.add_argument("--output", required=True, help="Output directory for results.")
    parser.add_argument("--rl", type=float, default=0.5, help="Low frequency gain (gamma_L). Default: 0.5")
    parser.add_argument("--rh", type=float, default=2.0, help="High frequency gain (gamma_H). Default: 2.0")
    parser.add_argument("--c", type=float, default=1.0, help="Sharpness constant. Default: 1.0")
    parser.add_argument("--d0", type=float, default=None, help="Cutoff frequency. If not set, estimated adaptively.")
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if input_path.is_file():
        files = [input_path]
    else:
        files = sorted(list(input_path.glob("*.tif*")))
        
    print(f"Processing {len(files)} files...")
    
    for f in tqdm(files):
        img = tifffile.imread(str(f))
        
        # Apply 2D homomorphic filter (standard)
        filtered = homomorphic_filter(img, d0=args.d0, rl=args.rl, rh=args.rh, c=args.c)
        
        # Clip to original range to avoid overflow
        dtype = img.dtype
        if np.issubdtype(dtype, np.integer):
            max_val = np.iinfo(dtype).max
            filtered = np.clip(filtered, 0, max_val).astype(dtype)
        
        tifffile.imwrite(str(output_dir / f.name), filtered, compression='lzw')

if __name__ == "__main__":
    main()
