import argparse
import time
import os
from pathlib import Path
import numpy as np
import tifffile
import matplotlib.pyplot as plt
from cellpose import models, io

def run_single_image_segmentation(input_path, output_dir, diameter=30.0, pretrained_model='cyto3', use_gpu=True):
    """
    Run Cellpose segmentation on a single TIFF image.
    """
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading image: {input_path}")
    img = tifffile.imread(str(input_path))
    print(f"Image shape: {img.shape}, dtype: {img.dtype}")
    
    # Initialize Cellpose model
    print(f"Initializing Cellpose model: {pretrained_model} (GPU={use_gpu})")
    try:
        model = models.CellposeModel(gpu=use_gpu, pretrained_model=pretrained_model)
    except Exception as e:
        print(f"Warning: Failed to load on GPU ({e}). Falling back to CPU.")
        model = models.CellposeModel(gpu=False, pretrained_model=pretrained_model)

    # Run inference
    print(f"Running inference (diameter={diameter})...")
    start_time = time.time()
    
    # eval() returns masks, flows, styles
    # channels=[0,0] means grayscale image (or infer from shape)
    masks, flows, styles = model.eval(
        img, 
        diameter=diameter,
        do_3D=(img.ndim==3), # Auto-detect 3D
        flow_threshold=0.8,
        cellprob_threshold=0.1,
    )
    
    end_time = time.time()
    print(f"Inference done in {end_time - start_time:.2f} seconds")
    print(f"Number of detected instances: {masks.max()}")

    # 1. Save Raw Mask (Integer labels, preserving instance IDs)
    # This file contains the exact instance IDs (1, 2, 3...)
    # Best for downstream analysis.
    raw_output_path = output_dir / f"{input_path.stem}_mask.tiff"
    print(f"Saving raw integer mask to: {raw_output_path}")
    tifffile.imwrite(str(raw_output_path), masks.astype(np.uint16), compression='lzw')

    # 2. Save Colorized Mask (Visualization)
    # This file is for human viewing, showing different instances in different colors.
    vis_output_path = output_dir / f"{input_path.stem}_mask_vis.png"
    print(f"Saving visualization to: {vis_output_path}")
    
    # Create a random color map for instances
    # We use a simple strategy: map each ID to a random color
    if masks.max() > 0:
        # Generate random colors
        np.random.seed(42)
        colors = np.random.randint(0, 255, size=(masks.max() + 1, 3), dtype=np.uint8)
        colors[0] = [0, 0, 0] # Background is black
        
        # Create RGB image
        if img.ndim == 2:
            # For 2D, simple color mapping
            h, w = masks.shape
            rgb_mask = np.zeros((h, w, 3), dtype=np.uint8)
            
            # Use advanced indexing for speed
            rgb_mask = colors[masks]
            
            # Save using matplotlib to ensure it works
            plt.imsave(str(vis_output_path), rgb_mask)
            
        elif img.ndim == 3:
            # For 3D, save the middle slice visualization
            mid_slice = masks.shape[0] // 2
            h, w = masks.shape[1], masks.shape[2]
            rgb_mask = colors[masks[mid_slice]]
            
            vis_output_path_3d = output_dir / f"{input_path.stem}_mask_vis_midslice.png"
            plt.imsave(str(vis_output_path_3d), rgb_mask)
            print(f"Saved middle slice visualization to {vis_output_path_3d}")

    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Cellpose on a single TIFF image")
    parser.add_argument('--input', required=True, help='Path to input TIFF image')
    parser.add_argument('--output', default='output_test', help='Output directory')
    parser.add_argument('--diameter', type=float, default=30.0, help='Cell diameter')
    parser.add_argument('--model', default='cpsam', help='Model type (cyto, cyto3, nuclei)')
    parser.add_argument('--no_gpu', action='store_true', help='Disable GPU')
    
    args = parser.parse_args()
    
    run_single_image_segmentation(
        args.input, 
        args.output, 
        diameter=args.diameter, 
        pretrained_model = args.model,
        use_gpu=not args.no_gpu
    )
