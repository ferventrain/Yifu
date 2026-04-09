import os
import re
import argparse
import numpy as np
import tifffile
import cv2
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
from sklearn.linear_model import RANSACRegressor
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DEFAULT_RANDOM_SEED = 42

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

def _normalize_to_uint8(image):
    """Normalize an array to uint8 for thresholding, preserving relative contrast."""
    image = np.asarray(image)
    if image.size == 0:
        return np.zeros(0, dtype=np.uint8)

    image = image.astype(np.float32, copy=False)
    image_min = float(image.min())
    image_max = float(image.max())
    if image_max <= image_min:
        return np.zeros(image.shape, dtype=np.uint8)

    normalized = (image - image_min) / (image_max - image_min)
    return np.clip(normalized * 255.0, 0, 255).astype(np.uint8)


def _estimate_tissue_non_signal_mask(img_cx_sample, img_c0_sample):
    """
    Build a fitting mask that keeps tissue pixels while excluding likely signal pixels.

    The mask logic is:
    1. Detect tissue mainly from C0, which better reflects autofluorescent tissue content.
    2. Remove very bright Cx pixels that likely correspond to real signal.
    3. Keep only the middle-low intensity portion of tissue pixels for stable fitting.
    """
    if img_cx_sample.size == 0 or img_c0_sample.size == 0:
        return np.zeros_like(img_cx_sample, dtype=bool)

    c0_uint8 = _normalize_to_uint8(img_c0_sample)
    if c0_uint8.max() > 0:
        c0_2d = c0_uint8.reshape(-1, 1)
        _, tissue_mask = cv2.threshold(c0_2d, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        tissue_mask = tissue_mask.ravel() > 0
    else:
        tissue_mask = np.zeros_like(img_c0_sample, dtype=bool)

    if np.sum(tissue_mask) < 100:
        c0_thresh = np.percentile(img_c0_sample, 60)
        tissue_mask = img_c0_sample > c0_thresh

    if np.sum(tissue_mask) < 100:
        return np.zeros_like(img_cx_sample, dtype=bool)

    cx_tissue = img_cx_sample[tissue_mask]
    c0_tissue = img_c0_sample[tissue_mask]

    cx_signal_thresh = np.percentile(cx_tissue, 85)
    c0_low = np.percentile(c0_tissue, 5)
    c0_high = np.percentile(c0_tissue, 95)

    fit_mask = (
        tissue_mask
        & (img_cx_sample <= cx_signal_thresh)
        & (img_c0_sample >= c0_low)
        & (img_c0_sample <= c0_high)
    )

    if np.sum(fit_mask) < 100:
        fit_mask = tissue_mask & (img_cx_sample <= np.percentile(cx_tissue, 90))

    return fit_mask


def collect_background_pixels(cx_file, c0_file, sample_size=10000, random_seed=DEFAULT_RANDOM_SEED):
    """
    Collect background pixels from a single image for global estimation.
    Returns (cx_bg, c0_bg) arrays of background pixel intensities.
    """
    img_cx = tifffile.imread(str(cx_file))
    img_c0 = tifffile.imread(str(c0_file))
    
    if img_cx.ndim != 2 or img_cx.shape != img_c0.shape:
        return None, None
    
    img_cx_flat = img_cx.flatten()
    img_c0_flat = img_c0.flatten()
    
    if len(img_cx_flat) > sample_size:
        rng = np.random.default_rng(random_seed)
        idx = rng.choice(len(img_cx_flat), sample_size, replace=False)
        img_cx_sample = img_cx_flat[idx]
        img_c0_sample = img_c0_flat[idx]
    else:
        img_cx_sample = img_cx_flat
        img_c0_sample = img_c0_flat

    bg_mask = _estimate_tissue_non_signal_mask(img_cx_sample, img_c0_sample)

    if np.sum(bg_mask) < 50:
        return None, None
    
    cx_bg = img_cx_sample[bg_mask]
    c0_bg = img_c0_sample[bg_mask]
    
    return cx_bg, c0_bg


def estimate_global_weight(cx_files, c0_files, plot_path=None, random_seed=DEFAULT_RANDOM_SEED):
    """
    Estimate global weight from collected background pixels from multiple images.
    If plot_path is provided, saves a summary scatter plot.
    """
    all_cx_bg = []
    all_c0_bg = []
    
    for i, (cx_file, c0_file) in enumerate(zip(cx_files, c0_files)):
        cx_bg, c0_bg = collect_background_pixels(
            cx_file,
            c0_file,
            random_seed=random_seed + i,
        )
        if cx_bg is not None and len(cx_bg) > 0:
            all_cx_bg.extend(cx_bg)
            all_c0_bg.extend(c0_bg)
    
    if len(all_cx_bg) < 500:
        print(f"  Warning: too few background pixels collected, using a=1.0")
        return 1.0
    
    all_cx_bg = np.array(all_cx_bg)
    all_c0_bg = np.array(all_c0_bg)
    
    A = np.vstack([all_c0_bg, np.ones(len(all_c0_bg))]).T
    result = np.linalg.lstsq(A, all_cx_bg, rcond=None)
    coeff = result[0]
    a, b = coeff[0], coeff[1]
    
    residuals = np.abs(all_cx_bg - (a * all_c0_bg + b))
    mad = np.median(np.abs(residuals - np.median(residuals)))
    threshold = 5 * mad if mad > 0 else 100
    
    if len(all_cx_bg) > 5000:
        ransac = RANSACRegressor(random_state=42, max_trials=100, residual_threshold=threshold)
        ransac.fit(all_c0_bg.reshape(-1, 1), all_cx_bg)
        a = ransac.estimator_.coef_[0]
        b = ransac.estimator_.intercept_
        inlier_mask = ransac.inlier_mask_
    else:
        inlier_mask = np.ones_like(all_cx_bg, dtype=bool)
    
    a = max(0.2, min(a, 1.6))
    
    if plot_path is not None:
        plt.figure(figsize=(10, 7))
        
        sample_size = min(5000, len(all_cx_bg))
        rng = np.random.default_rng(random_seed)
        idx_sample = rng.choice(len(all_cx_bg), sample_size, replace=False)
        cx_sample = all_cx_bg[idx_sample]
        c0_sample = all_c0_bg[idx_sample]
        mask_sample = inlier_mask[idx_sample]
        
        if np.sum(~mask_sample) > 0:
            plt.scatter(c0_sample[~mask_sample], cx_sample[~mask_sample], 
                       c='orange', alpha=0.5, s=10, label='Outliers')
        plt.scatter(c0_sample[mask_sample], cx_sample[mask_sample], 
                   c='blue', alpha=0.5, s=10, label='Background inliers')
        
        x_min, x_max = 0, all_c0_bg.max()
        x_fit = np.linspace(x_min, x_max, 100)
        y_fit = a * x_fit + b
        plt.plot(x_fit, y_fit, 'k--', linewidth=2, label=f'Global fit: Cx = {a:.3f} * C0 + {b:.1f}')
        
        plt.xlabel('C0 (autofluorescence) intensity')
        plt.ylabel('Cx (signal channel) intensity')
        plt.title(f'Global weight estimation - {len(cx_files)} images, {np.sum(inlier_mask)} tissue non-signal pixels used')
        plt.legend(loc='upper left')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150)
        plt.close()

    print(
        f"  Weight fit diagnostics: slices={len(cx_files)}, "
        f"pixels={len(all_cx_bg)}, inliers={int(np.sum(inlier_mask))}, "
        f"a={a:.4f}, b={b:.2f}"
    )
    
    return a


def estimate_background_weight(img_cx, img_c0, sample_size=10000, plot_path=None, random_seed=DEFAULT_RANDOM_SEED):
    """
    Estimate weight a by linear regression on background pixels only: Cx = a * C0 + b
    Uses Otsu thresholding to identify background (non-signal) pixels.
    Uses RANSAC for robust estimation against outliers.
    If plot_path is provided, saves an intensity scatter plot.
    """
    if img_cx.ndim != 2:
        if plot_path is not None:
            plt.figure(figsize=(8, 6))
            plt.text(0.5, 0.5, f"Not 2D image\nfallback to weight=1.0", 
                    ha='center', va='center', fontsize=12, transform=plt.gca().transAxes)
            plt.xlabel('C0 intensity')
            plt.ylabel('Cx intensity')
            plt.title(f'Not 2D image, a=1.00')
            plt.tight_layout()
            plt.savefig(plot_path, dpi=150)
            plt.close()
        return 1.0
    
    img_cx_flat = img_cx.flatten()
    img_c0_flat = img_c0.flatten()
    
    if len(img_cx_flat) > sample_size:
        rng = np.random.default_rng(random_seed)
        idx = rng.choice(len(img_cx_flat), sample_size, replace=False)
        img_cx_sample = img_cx_flat[idx]
        img_c0_sample = img_c0_flat[idx]
    else:
        img_cx_sample = img_cx_flat
        img_c0_sample = img_c0_flat

    bg_mask = _estimate_tissue_non_signal_mask(img_cx_sample, img_c0_sample)
    fg_mask = ~bg_mask
    
    if np.sum(bg_mask) < 100:
        if plot_path is not None:
            plt.figure(figsize=(8, 6))
            plt.text(0.5, 0.5, f"Too few background pixels\nfallback to weight=1.0", 
                    ha='center', va='center', fontsize=12, transform=plt.gca().transAxes)
            plt.xlabel('C0 intensity')
            plt.ylabel('Cx intensity')
            plt.title(f'Too few tissue non-signal pixels, a=1.00')
            plt.tight_layout()
            plt.savefig(plot_path, dpi=150)
            plt.close()
        return 1.0
    
    cx_bg = img_cx_sample[bg_mask].reshape(-1, 1)
    c0_bg = img_c0_sample[bg_mask]
    
    A = np.vstack([c0_bg, np.ones(len(c0_bg))]).T
    result = np.linalg.lstsq(A, cx_bg.ravel(), rcond=None)
    coeff = result[0]
    a, b = coeff[0], coeff[1]
    
    if len(cx_bg) > 1000:
        residuals = np.abs(cx_bg.ravel() - (a * c0_bg + b))
        mad = np.median(np.abs(residuals - np.median(residuals)))
        threshold = 5 * mad if mad > 0 else 100
        
        ransac = RANSACRegressor(random_state=42, max_trials=100, residual_threshold=threshold)
        ransac.fit(c0_bg.reshape(-1, 1), cx_bg.ravel())
        a = ransac.estimator_.coef_[0]
        b = ransac.estimator_.intercept_
        inlier_mask = ransac.inlier_mask_
    else:
        inlier_mask = np.ones_like(cx_bg, dtype=bool).ravel()
    
    a = max(0.2, min(a, 1.6))
    
    if plot_path is not None:
        plt.figure(figsize=(10, 7))
        
        if np.sum(fg_mask) > 10:
            plt.scatter(img_c0_sample[fg_mask], img_cx_sample[fg_mask], 
                       c='red', alpha=0.5, s=10, label='Signal (foreground)')
        
        if len(cx_bg) > 1000:
            plt.scatter(c0_bg[~inlier_mask], cx_bg[~inlier_mask], 
                       c='orange', alpha=0.6, s=15, label='Fit outliers')
            plt.scatter(c0_bg[inlier_mask], cx_bg[inlier_mask], 
                       c='blue', alpha=0.6, s=15, label='Tissue non-signal inliers')
        else:
            plt.scatter(c0_bg, cx_bg, c='blue', alpha=0.6, s=15, label='Tissue non-signal (used)')
        
        x_min, x_max = 0, max(img_c0_sample.max(), c0_bg.max())
        x_fit = np.linspace(x_min, x_max, 100)
        y_fit = a * x_fit + b
        plt.plot(x_fit, y_fit, 'k--', linewidth=2, label=f'Fit: Cx = {a:.2f} * C0 + {b:.1f}')
        
        plt.xlabel('C0 (autofluorescence) intensity')
        plt.ylabel('Cx (signal channel) intensity')
        plt.title(f'Tissue non-signal fit - estimated weight a = {a:.3f}')
        plt.legend(loc='upper left')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150)
        plt.close()
    
    return a


def subtract_worker(cx_file, c0_file, output_path, compression='lzw', global_weight=1.0):
    """
    Worker function to perform subtraction for a single image pair.
    Uses pre-estimated global weight.
    """
    try:
        # Load images
        img_cx = tifffile.imread(str(cx_file))
        img_c0 = tifffile.imread(str(c0_file))
        
        # Ensure same shape
        if img_cx.shape != img_c0.shape:
            return f"Error: Shape mismatch for {cx_file.name}"
        
        # Perform subtraction: Cx - global_weight * C0
        dtype = img_cx.dtype
        max_val = np.iinfo(dtype).max
        
        # Subtract and clip using vectorized numpy operations
        subtracted = np.clip(img_cx.astype(np.int32) - global_weight * img_c0.astype(np.int32), 0, max_val).astype(dtype)
        
        # Save result
        tifffile.imwrite(str(output_path), subtracted, compression=compression)
        return "success"
    except Exception as e:
        return f"Error processing {cx_file.name}: {str(e)}"

def process_channel_subtraction(root_path, max_workers=None, compression='lzw', weight=1.0, adaptive=False, save_plots=False, 
                               sample_ratio=0.005, min_samples=10, max_samples=50, background_channel='ch0'):
    root = Path(root_path)
    if not root.is_dir():
        print(f"Error: {root_path} is not a valid directory.")
        return

    # Find the background channel folder
    pattern = f"*{background_channel}*"
    bg_folders = [f for f in root.iterdir() if f.is_dir() and (f.name == background_channel or f.name.startswith(f"{background_channel}_"))]
    if not bg_folders:
        bg_folders = list(root.glob(pattern))
    
    if not bg_folders:
        print(f"Error: Could not find folder matching '{background_channel}' in {root_path}.")
        return
    
    bg_dir = bg_folders[0]
    print(f"Using {bg_dir.name} as background channel (autofluorescence reference).")

    # Map Z-index to file path in background channel
    bg_files = {}
    for f in bg_dir.glob("*.tif*"):
        _, z_idx = parse_filename(f.name)
        if z_idx:
            bg_files[z_idx] = f
    
    if not bg_files:
        print(f"Error: No valid TIFF files found in {bg_dir.name}.")
        return
    
    print(f"Found {len(bg_files)} reference files in {bg_dir.name}.")



    if adaptive:
        print(f"Global adaptive weight estimation: ON (sample_ratio={sample_ratio}, min={min_samples}, max={max_samples})")
        if save_plots:
            print("Saving global fit plot: ON")
    else:
        print(f"Fixed weight: {weight}")

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
        
        # Match files with ch0
        matched_pairs = []
        for cx_file in cx_files:
            _, z_idx = parse_filename(cx_file.name)
            if z_idx in bg_files:
                matched_pairs.append((cx_file, bg_files[z_idx]))
        
        if not matched_pairs:
            print(f"No matched files found in {folder.name}, skipping.")
            continue
        
        print(f"  Found {len(matched_pairs)} matched files")
        
        # Global adaptive estimation: sample some images to estimate global weight
        if adaptive:
            n_total = len(matched_pairs)
            n_sample = max(min_samples, min(max_samples, int(n_total * sample_ratio)))
            print(f"  Sampling {n_sample} images out of {n_total} for global weight estimation")
            
            # Random sample
            idx_sample = np.random.choice(n_total, n_sample, replace=False)
            sample_pairs = [matched_pairs[i] for i in idx_sample]
            sample_cx = [p[0] for p in sample_pairs]
            sample_c0 = [p[1] for p in sample_pairs]
            
            # Estimate global weight
            if save_plots:
                plot_path = target_subtracted_dir / f"{folder.name}_global_fit.png"
                global_a = estimate_global_weight(sample_cx, sample_c0, plot_path=plot_path)
                print(f"  Global fit plot saved to: {plot_path}")
            else:
                global_a = estimate_global_weight(sample_cx, sample_c0)
            
            final_weight = weight * global_a
            print(f"  Estimated a = {global_a:.4f}, final effective weight = {final_weight:.4f}")
        else:
            final_weight = weight
        
        # Prepare tasks for parallel processing
        tasks = []
        for cx_file, c0_file in matched_pairs:
            output_path = target_subtracted_dir / cx_file.name
            if not output_path.exists(): # Basic resume support
                tasks.append((cx_file, c0_file, output_path, compression, final_weight))
        
        # Execute tasks in parallel
        print(f"  Processing {len(tasks)} remaining files (resume enabled)")
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_file = {executor.submit(subtract_worker, *task): task[0].name for task in tasks}
            
            for future in tqdm(as_completed(future_to_file), total=len(tasks), desc=f"  Subtracting"):
                result = future.result()
                if result != "success":
                    print(f"\n  {result}")

    print("All processing complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parallel subtract background channel from other channel images for autofluorescence removal.")
    parser.add_argument("path", type=str, help="Root path containing channel folders")
    parser.add_argument("--bg-channel", type=str, default="ch0", help="Background channel folder name (default: ch0)")
    parser.add_argument("--workers", type=int, default=None, help="Number of parallel processes (default: CPU_COUNT // 2)")
    parser.add_argument("--compression", type=str, default="lzw", choices=["lzw", "none", "zlib"], help="TIFF compression (default: lzw)")
    parser.add_argument("--weight", type=float, default=1.0, help="Base weight for subtraction: result = Cx - (weight * a) * Cbg (default: 1.0)")
    parser.add_argument("--adaptive", action="store_true", help="Global adaptive weight estimation (sample background from multiple slices)")
    parser.add_argument("--save-plots", action="store_true", help="Save global fit plot (only with --adaptive)")
    parser.add_argument("--sample-ratio", type=float, default=0.005, help="Ratio of images to sample for global estimation (default: 0.005)")
    parser.add_argument("--min-samples", type=int, default=10, help="Minimum number of images to sample (default: 10)")
    parser.add_argument("--max-samples", type=int, default=50, help="Maximum number of images to sample (default: 50)")
    
    args = parser.parse_args()
    
    # Map 'none' to None for tifffile
    comp = None if args.compression == 'none' else args.compression
    
    process_channel_subtraction(args.path, max_workers=args.workers, compression=comp, 
                               weight=args.weight, adaptive=args.adaptive, save_plots=args.save_plots,
                               sample_ratio=args.sample_ratio, min_samples=args.min_samples, max_samples=args.max_samples,
                               background_channel=args.bg_channel)
