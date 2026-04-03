import os
import sys
import argparse
import json
import re
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import numpy as np
import tifffile

import cv2

from .tophat_background import tophat_background_correction
from .scattering_removal import remove_scattering
from .masked_clahe import clahe_enhance


def process_single_image(input_path, output_path, steps):
    """Process a single image through a series of preprocessing steps.
    
    Args:
        input_path: Path to input TIFF file
        output_path: Path to save processed TIFF
        steps: List of (func, kwargs) tuples to apply in order
    """
    try:
        img = tifffile.imread(str(input_path))
        
        if img.ndim == 3 and img.shape[2] == 1:
            img = img[:, :, 0]
        
        current_img = img
        dtype = img.dtype
        
        for func, kwargs in steps:
            if func == 'tophat':
                kernel_size = kwargs.get('kernel_size', 21)
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
                if len(current_img.shape) == 3:
                    gray = current_img[:, :, 0]
                else:
                    gray = current_img
                tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
                current_img = tophat.astype(dtype)
            
            elif func == 'scattering_removal':
                sigma = kwargs.get('sigma', 50.0)
                weight = kwargs.get('weight', 1.0)
                img_float = current_img.astype(np.float64)
                background = cv2.GaussianBlur(img_float, ksize=(0, 0), sigmaX=sigma, sigmaY=sigma)
                result = np.clip(img_float - weight * background, 0, None)
                if np.issubdtype(dtype, np.integer):
                    max_val = np.iinfo(dtype).max
                    result = np.clip(result, 0, max_val).astype(dtype)
                else:
                    result = result.astype(dtype)
                current_img = result
            
            elif func == 'clahe':
                clip_limit = kwargs.get('clip_limit', 2.0)
                tile_grid_size = kwargs.get('tile_grid_size', 8)
                clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid_size, tile_grid_size))
                
                if len(current_img.shape) == 3:
                    gray = current_img[:, :, 0]
                else:
                    gray = current_img
                
                result = clahe.apply(gray.astype(np.uint16))
                current_img = result.astype(dtype)
            
            elif func == 'homomorphic_filter':
                from .homomorphic_filter import homomorphic_filter
                rl = kwargs.get('rl', 0.5)
                rh = kwargs.get('rh', 2.0)
                c = kwargs.get('c', 1.0)
                d0 = kwargs.get('d0', None)
                result = homomorphic_filter(current_img, rl=rl, rh=rh, c=c, d0=d0)
                current_img = result.astype(dtype)
        
        tifffile.imwrite(str(output_path), current_img, compression=None)
        return True
    
    except Exception as e:
        print(f"Error processing {input_path}: {str(e)}")
        return False


class Preprocessor:
    """Configurable preprocessor that applies a sequence of enhancement steps.
    
    Supports parallel processing of multiple TIFF files with configurable steps.
    """
    
    def __init__(self, preprocessing_config):
        """Initialize preprocessor from config.
        
        Args:
            preprocessing_config: 'preprocessing' section from config.json
        """
        self.config = preprocessing_config
        self.steps = self._build_steps()
    
    def _build_steps(self):
        """Build the list of processing steps from config.
        
        Steps are executed in the order they appear in config.
        Only steps with "apply": true are included.
        """
        steps = []
        
        for step_name, step_config in self.config.items():
            if step_name in ['downsample', 'zarr']:
                continue
            
            if not isinstance(step_config, dict):
                continue
            
            apply = step_config.get('apply', False)
            if not apply:
                continue
            
            steps.append((step_name, step_config))
        
        return steps
    
    def process_folder(self, input_folder, output_folder, max_workers=None, resume=True):
        """Process all TIFF files in a folder.
        
        Args:
            input_folder: Path to input folder containing TIFF files
            output_folder: Path to output folder for processed TIFF files
            max_workers: Number of parallel workers (None = CPU count // 2)
            resume: If True, skip already processed files
        """
        input_path = Path(input_folder)
        output_path = Path(output_folder)
        output_path.mkdir(parents=True, exist_ok=True)
        
        tiff_files = sorted(list(input_path.glob('*.tif*')))
        if not tiff_files:
            print(f"No TIFF files found in {input_folder}")
            return False
        
        print(f"Found {len(tiff_files)} TIFF files to process")
        print(f"Processing steps (in order): {[name for name, _ in self.steps]}")
        
        if len(self.steps) == 0:
            print("No preprocessing steps enabled in config, skipping.")
            return True
        
        tasks = []
        for tiff_file in tiff_files:
            output_file = output_path / tiff_file.name
            if resume and output_file.exists():
                continue
            tasks.append((tiff_file, output_file, self.steps))
        
        if len(tasks) == 0:
            print(f"All {len(tiff_files)} files already processed, done.")
            return True
        
        print(f"Processing {len(tasks)} remaining files (resume={resume})")
        
        if max_workers is None:
            max_workers = max(1, os.cpu_count() // 2)
        
        if os.name == 'nt' and max_workers > 61:
            print(f"Note: Capping max_workers to 61 due to Windows limitations")
            max_workers = 61
        
        print(f"Using {max_workers} parallel workers")
        
        completed = 0
        failed = 0
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(process_single_image, *task): task[0].name 
                for task in tasks
            }
            
            for future in tqdm(as_completed(futures), total=len(futures), desc="Preprocessing"):
                result = future.result()
                if result:
                    completed += 1
                else:
                    failed += 1
        
        print(f"\nPreprocessing complete:")
        print(f"  Completed: {completed}")
        print(f"  Failed: {failed}")
        print(f"  Output: {output_folder}")
        
        return failed == 0


def parse_filename(filename):
    """Extract channel and Z-index from filename like YF2025102901_..._C1_Z0051."""
    match = re.search(r'_(C\d+)_Z(\d+)', filename)
    if match:
        return match.group(1), match.group(2)
    return None, None


def channel_subtraction_worker(cx_file, c0_file, output_path, weight, compression=None):
    """Worker to perform channel subtraction for a single image pair."""
    try:
        img_cx = tifffile.imread(str(cx_file))
        img_c0 = tifffile.imread(str(c0_file))
        
        if img_cx.shape != img_c0.shape:
            return f"Error: Shape mismatch for {cx_file.name}"
        
        dtype = img_cx.dtype
        max_val = np.iinfo(dtype).max
        
        subtracted = np.clip(img_cx.astype(np.int32) - weight * img_c0.astype(np.int32), 0, max_val).astype(dtype)
        tifffile.imwrite(str(output_path), subtracted, compression=compression)
        return "success"
    except Exception as e:
        return f"Error processing {cx_file.name}: {str(e)}"


def run_channel_subtraction(root_path, background_channel='ch0', weight=1.0, adaptive=False, 
                           save_plots=False, sample_ratio=0.005, min_samples=10, max_samples=50, 
                           max_workers=None, compression=None):
    """Run channel subtraction on all channels except background channel.
    
    Args:
        root_path: Root directory containing channel folders (ch0, ch1, ch2, ...)
        background_channel: Name of background channel folder (default: 'ch0')
        weight: Base weight for subtraction: result = Cx - (weight * a) * Cbg
        adaptive: Whether to estimate global adaptive weight
        save_plots: Save global fit plot (only with adaptive)
        sample_ratio: Ratio of images to sample for global estimation
        min_samples: Minimum number of images to sample
        max_samples: Maximum number of images to sample
        max_workers: Number of parallel workers
        compression: TIFF compression
    """
    from .channel_subtraction import estimate_global_weight
    
    root = Path(root_path)
    if not root.is_dir():
        print(f"Error: {root_path} is not a valid directory.")
        return False
    
    pattern = f"*{background_channel}*"
    bg_folders = [f for f in root.iterdir() if f.is_dir() and 
                 (f.name == background_channel or f.name.startswith(f"{background_channel}_"))]
    if not bg_folders:
        bg_folders = list(root.glob(pattern))
    
    if not bg_folders:
        print(f"Error: Could not find folder matching '{background_channel}' in {root_path}.")
        return False
    
    bg_dir = bg_folders[0]
    print(f"Using {bg_dir.name} as background channel.")
    
    bg_files = {}
    for f in bg_dir.glob("*.tif*"):
        _, z_idx = parse_filename(f.name)
        if z_idx:
            bg_files[z_idx] = f
    
    if not bg_files:
        print(f"Error: No valid TIFF files found in {bg_dir.name}.")
        return False
    
    print(f"Found {len(bg_files)} reference files in {bg_dir.name}.")
    
    if max_workers is None:
        max_workers = max(1, os.cpu_count() // 2)
    
    if os.name == 'nt' and max_workers > 61:
        print(f"Note: Capping max_workers to 61 due to Windows limitations")
        max_workers = 61
    
    print(f"Using {max_workers} workers for parallel processing.")
    
    for folder in root.iterdir():
        if not folder.is_dir() or folder == bg_dir or folder.name.endswith("_subtracted"):
            continue
        
        target_subtracted_dir = root / f"{folder.name}_subtracted"
        target_subtracted_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\nProcessing folder: {folder.name} -> {target_subtracted_dir.name}")
        
        cx_files = list(folder.glob("*.tif*"))
        if not cx_files:
            print(f"No TIFF files found in {folder.name}, skipping.")
            continue
        
        matched_pairs = []
        for cx_file in cx_files:
            _, z_idx = parse_filename(cx_file.name)
            if z_idx in bg_files:
                matched_pairs.append((cx_file, bg_files[z_idx], target_subtracted_dir / cx_file.name, weight, compression))
        
        if not matched_pairs:
            print(f"No matched files found in {folder.name}, skipping.")
            continue
        
        print(f"  Found {len(matched_pairs)} matched files")
        
        if adaptive:
            print(f"  Global adaptive weight estimation: ON")
            n_total = len(matched_pairs)
            n_sample = max(min_samples, min(max_samples, int(n_total * sample_ratio)))
            print(f"  Sampling {n_sample} images out of {n_total}")
            
            import random
            idx_sample = random.sample(range(n_total), n_sample)
            sample_pairs = [matched_pairs[i] for i in idx_sample]
            sample_cx = [p[0] for p in sample_pairs]
            sample_c0 = [p[1] for p in sample_pairs]
            
            if save_plots:
                plot_path = target_subtracted_dir / f"{folder.name}_global_fit.png"
                global_a = estimate_global_weight(sample_cx, sample_c0, plot_path=plot_path)
                print(f"  Global fit plot saved to: {plot_path}")
            else:
                global_a = estimate_global_weight(sample_cx, sample_c0)
            
            final_weight = weight * global_a
            print(f"  Estimated a = {global_a:.4f}, final effective weight = {final_weight:.4f}")
            
            for i in range(len(matched_pairs)):
                matched_pairs[i] = (
                    matched_pairs[i][0],
                    matched_pairs[i][1],
                    matched_pairs[i][2],
                    final_weight,
                    matched_pairs[i][4]
                )
        else:
            print(f"  Fixed weight: {weight}")
            final_weight = weight
        
        tasks = []
        for cx_file, c0_file, output_path, w, comp in matched_pairs:
            if not output_path.exists():
                tasks.append((cx_file, c0_file, output_path, w, comp))
        
        print(f"  Processing {len(tasks)} remaining files (resume enabled)")
        
        completed = 0
        failed = 0
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(channel_subtraction_worker, *task): task[0].name
                for task in tasks
            }
            
            for future in tqdm(as_completed(futures), total=len(futures), desc=f"  Subtracting"):
                result = future.result()
                if result == "success":
                    completed += 1
                else:
                    print(f"\n  {result}")
                    failed += 1
        
        print(f"  Done: {completed} completed, {failed} failed")
    
    print(f"\nChannel subtraction complete for all channels!")
    return True


def main():
    """Standalone main program for preprocessing.
    
    Usage:
        python -m pipeline_modules.preprocessing.preprocessor --config config.json --sample_dir /path/to/sample
    """
    parser = argparse.ArgumentParser(
        description="Standalone configurable preprocessor with channel subtraction and enhancement"
    )
    parser.add_argument('--config', required=True, help='Path to config.json')
    parser.add_argument('--sample_dir', required=True, help='Sample root directory containing channel folders')
    parser.add_argument('--channel', help='Only process this specific channel (e.g., "1" for ch1, overrides config)')
    parser.add_argument('--workers', type=int, default=None, help='Number of parallel workers (default: CPU//2)')
    parser.add_argument('--no-resume', action='store_true', help='Disable resume (reprocess all files)')
    parser.add_argument('--output_dir', help='Custom output directory (default: chX_preprocessed in sample_dir)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.config):
        print(f"Error: Config file not found: {args.config}")
        sys.exit(1)
    
    with open(args.config, 'r') as f:
        full_config = json.load(f)
    
    sample_dir = Path(args.sample_dir)
    if not sample_dir.exists():
        print(f"Error: Sample directory not found: {args.sample_dir}")
        sys.exit(1)
    
    print(f"Config: {args.config}")
    print(f"Sample dir: {sample_dir}")
    
    cfg = full_config
    preprocessing_cfg = cfg['preprocessing']
    channels_cfg = cfg['input']['channels']
    
    signal_ch = args.channel if args.channel else channels_cfg['signal']
    print(f"Processing channel: {signal_ch}")
    
    resume = not args.no_resume
    print(f"Resume enabled: {resume}")
    
    if 'channel_subtraction' in preprocessing_cfg:
        cs_cfg = preprocessing_cfg['channel_subtraction']
        if cs_cfg.get('apply', False):
            print("\n" + "="*50)
            print("Step: Channel Subtraction")
            print("="*50)
            
            background_channel = cs_cfg.get('background_channel', 'ch0')
            weight = cs_cfg.get('weight', 1.0)
            adaptive = cs_cfg.get('adaptive', False)
            save_plots = cs_cfg.get('save_plots', False)
            sample_ratio = cs_cfg.get('sample_ratio', 0.005)
            min_samples = cs_cfg.get('min_samples', 10)
            max_samples = cs_cfg.get('max_samples', 50)
            compression = cs_cfg.get('compression', 'lzw')
            comp = None if compression == 'none' else compression
            
            success = run_channel_subtraction(
                root_path=sample_dir,
                background_channel=background_channel,
                weight=weight,
                adaptive=adaptive,
                save_plots=save_plots,
                sample_ratio=sample_ratio,
                min_samples=min_samples,
                max_samples=max_samples,
                max_workers=args.workers,
                compression=comp
            )
            
            if not success:
                print("Channel subtraction failed, exiting.")
                sys.exit(1)
    
    print("\n" + "="*50)
    print("Step: Image Enhancement (Preprocessor)")
    print("="*50)
    
    input_folder = sample_dir / f"ch{signal_ch}"
    if 'channel_subtraction' in preprocessing_cfg and preprocessing_cfg['channel_subtraction'].get('apply', False):
        input_folder = sample_dir / f"ch{signal_ch}_subtracted"
        if not input_folder.exists():
            print(f"Error: Subtracted folder not found: {input_folder}")
            sys.exit(1)
    
    if args.output_dir:
        output_folder = Path(args.output_dir)
    else:
        output_folder = sample_dir / f"ch{signal_ch}_preprocessed"
    
    preprocessor = Preprocessor(preprocessing_cfg)
    
    if len(preprocessor.steps) == 0:
        print("No enhancement steps enabled, done.")
        sys.exit(0)
    
    success = preprocessor.process_folder(
        input_folder=input_folder,
        output_folder=output_folder,
        max_workers=args.workers,
        resume=resume
    )
    
    if success:
        print("\n" + "="*50)
        print("PREPROCESSING COMPLETE")
        print(f"Output: {output_folder}")
        print("="*50)
        sys.exit(0)
    else:
        print("\nPreprocessing failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
