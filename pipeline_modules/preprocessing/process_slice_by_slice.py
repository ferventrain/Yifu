import argparse
import numpy as np
import tifffile
import cv2
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

from homomorphic_filter import homomorphic_filter


class CLAHE2D:
    """CLAHE for 2D images supporting 8-bit and 16-bit.
    Automatically uses OpenCV CUDA GPU acceleration if available.
    """
    def __init__(self, clip_limit=2.0, tile_grid_size=(32, 32), use_gpu=True):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
        self.use_gpu = use_gpu
        self.gpu_available = False
        
        if self.use_gpu and cv2.cuda.getCudaEnabledDeviceCount() > 0:
            # GPU CLAHE is available
            self.gpu_available = True
            self.clahe = cv2.cuda.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
        else:
            # Fallback to CPU
            self.clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
    
    def process(self, img):
        if self.gpu_available:
            # GPU processing
            if img.dtype == np.uint16:
                gpu_mat = cv2.cuda_GpuMat(img)
                result_gpu = self.clahe.apply(gpu_mat)
                return result_gpu.download()
            elif img.dtype == np.uint8:
                gpu_mat = cv2.cuda_GpuMat(img)
                result_gpu = self.clahe.apply(gpu_mat)
                return result_gpu.download()
            else:
                if img.dtype in [np.float32, np.float64]:
                    if img.max() <= 1.0:
                        img_scaled = (img * 65535).astype(np.uint16)
                    else:
                        img_scaled = img.astype(np.uint16)
                    gpu_mat = cv2.cuda_GpuMat(img_scaled)
                    result_gpu = self.clahe.apply(gpu_mat)
                    result_scaled = result_gpu.download()
                    return result_scaled.astype(img.dtype)
                else:
                    raise ValueError(f"Unsupported dtype: {img.dtype}")
        else:
            # CPU processing
            if img.dtype == np.uint16:
                return self.clahe.apply(img)
            elif img.dtype == np.uint8:
                return self.clahe.apply(img)
            else:
                if img.dtype in [np.float32, np.float64]:
                    if img.max() <= 1.0:
                        img_scaled = (img * 65535).astype(np.uint16)
                    else:
                        img_scaled = img.astype(np.uint16)
                    result_scaled = self.clahe.apply(img_scaled)
                    return result_scaled.astype(img.dtype)
                else:
                    raise ValueError(f"Unsupported dtype: {img.dtype}")


def process_single_file(args):
    """Process a single TIFF slice: 2D Homomorphic -> 2D CLAHE."""
    input_path, output_path, d0, rl, rh, c, clip_limit, tile_grid_size, use_gpu = args
    
    img = tifffile.imread(str(input_path))
    
    filtered = homomorphic_filter(img, d0=d0, rl=rl, rh=rh, c=c, verbose=False)
    
    dtype = img.dtype
    if np.issubdtype(dtype, np.integer):
        max_val = np.iinfo(dtype).max
        filtered = np.clip(filtered, 0, max_val).astype(dtype)
    
    clahe = CLAHE2D(clip_limit=clip_limit, tile_grid_size=(tile_grid_size, tile_grid_size), use_gpu=use_gpu)
    final = clahe.process(filtered)
    
    tifffile.imwrite(str(output_path), final, compression='lzw')
    
    return input_path.name


def process_folder(input_dir, output_dir, d0=None, rl=0.5, rh=2.0, c=1.0, 
                   clip_limit=2.0, tile_grid_size=32, n_workers=None, compression='lzw', use_gpu=False):
    """Process all TIFF slices in a folder: 2D Homomorphic + 2D CLAHE slice-by-slice."""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    tiff_files = sorted(list(input_dir.glob("*.tif*")))
    if not tiff_files:
        print(f"No TIFF files found in {input_dir}")
        return
    
    print(f"Found {len(tiff_files)} TIFF files to process...")
    print(f"Parameters: d0={d0}, rl={rl}, rh={rh}, c={c}, clip_limit={clip_limit}, tile_grid_size={tile_grid_size}")
    print(f"GPU acceleration: {'enabled' if use_gpu else 'disabled'}")
    
    if n_workers is None:
        max_allowed = min(multiprocessing.cpu_count(), 61)
        n_workers = max(1, max_allowed - 2)
    print(f"Using {n_workers} parallel workers...")
    
    if d0 is None:
        print("Estimating d0 automatically from first slice...")
        first_img = tifffile.imread(str(tiff_files[0]))
        from homomorphic_filter import estimate_d0_adaptive
        d0 = estimate_d0_adaptive(first_img)
        print(f"Estimated d0: {d0}")
    
    tasks = []
    for f in tiff_files:
        out_path = output_dir / f.name
        tasks.append((f, out_path, d0, rl, rh, c, clip_limit, tile_grid_size, use_gpu))
    
    completed = 0
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = [executor.submit(process_single_file, task) for task in tasks]
        
        with tqdm(total=len(tasks), desc="Processing slices") as pbar:
            for future in as_completed(futures):
                fname = future.result()
                completed += 1
                pbar.update(1)
    
    print(f"\nCompleted {completed}/{len(tiff_files)} slices.")
    print(f"Results saved to {output_dir}")


def test_single_image(input_path, output_path, d0=None, rl=0.3, rh=2.0, c=2.0, 
                      clip_limit=2.0, tile_grid_size=16, use_gpu=False):
    """Test processing on a single image, saves both homo and homo+clahe results."""
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    img = tifffile.imread(str(input_path))
    
    print(f"Input image shape: {img.shape}, dtype: {img.dtype}")
    
    if d0 is None:
        print("Estimating d0 automatically...")
    
    filtered = homomorphic_filter(img, d0=d0, rl=rl, rh=rh, c=c, verbose=False)
    
    dtype = img.dtype
    if np.issubdtype(dtype, np.integer):
        max_val = np.iinfo(dtype).max
        filtered_clip = np.clip(filtered, 0, max_val).astype(dtype)
    else:
        filtered_clip = filtered
    
    homo_out = output_path.parent / f"{output_path.stem}_homo.tif"
    tifffile.imwrite(homo_out, filtered_clip, compression='lzw')
    print(f"Homomorphic result saved to {homo_out}")
    
    clahe = CLAHE2D(clip_limit=clip_limit, tile_grid_size=(tile_grid_size, tile_grid_size), use_gpu=use_gpu)
    if clahe.gpu_available:
        print("CLAHE: using GPU acceleration")
    else:
        print("CLAHE: using CPU" + (" (GPU requested but not available)" if use_gpu else ""))
    
    final = clahe.process(filtered_clip)
    
    clahe_out = output_path
    if not str(output_path).endswith('.tif') and not str(output_path).endswith('.tiff'):
        clahe_out = output_path.with_suffix('.tif')
    tifffile.imwrite(clahe_out, final, compression='lzw')
    print(f"Homomorphic + CLAHE result saved to {clahe_out}")
    
    return filtered_clip, final


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Slice-by-slice processing: 2D Homomorphic Filter -> 2D CLAHE.")
    parser.add_argument("--input", required=True, help="Input directory of TIFF slices or single TIFF file.")
    parser.add_argument("--output", required=True, help="Output directory or output file (if input is single file).")
    
    parser.add_argument("--d0", type=float, default=None, help="Cutoff frequency for homomorphic filter. Auto-estimated if not set.")
    parser.add_argument("--rl", type=float, default=0.5, help="Low frequency gain. Default: 0.5")
    parser.add_argument("--rh", type=float, default=2.0, help="High frequency gain. Default: 2.0")
    parser.add_argument("--c", type=float, default=2.0, help="Sharpness constant. Default: 2.0")
    
    parser.add_argument("--clip_limit", type=float, default=2.0, help="CLAHE clip limit. Default: 2.0")
    parser.add_argument("--tile_grid_size", type=int, default=16, help="CLAHE tile grid size. Default: 16")
    
    parser.add_argument("--n_workers", type=int, default=None, help="Number of parallel workers. Default: auto")
    parser.add_argument("--compression", default='lzw', help="Output compression. Default: lzw")
    parser.add_argument("--use_gpu", action='store_true', help="Use OpenCV CUDA GPU acceleration for CLAHE if available. Default: False")
    parser.add_argument("--test_single", action='store_true', help="Test mode: process single image, save intermediate homo result.")
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    if args.test_single or input_path.is_file():
        test_single_image(
            input_path, 
            output_path,
            d0=args.d0,
            rl=args.rl,
            rh=args.rh,
            c=args.c,
            clip_limit=args.clip_limit,
            tile_grid_size=args.tile_grid_size,
            use_gpu=args.use_gpu
        )
    else:
        process_folder(
            input_path,
            output_path,
            d0=args.d0,
            rl=args.rl,
            rh=args.rh,
            c=args.c,
            clip_limit=args.clip_limit,
            tile_grid_size=args.tile_grid_size,
            n_workers=args.n_workers,
            compression=args.compression,
            use_gpu=args.use_gpu
        )
