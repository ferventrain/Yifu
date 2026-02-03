import argparse
from pathlib import Path
from typing import List
import numpy as np
from base_segmentor import BaseSegmentor

class CellposeSegmentor(BaseSegmentor):
    def __init__(self, pretrained_model='cyto3', gpu=True, diameter=30):
        super().__init__()
        self.gpu = gpu
        self.diameter = diameter
        self.pretrained_model = pretrained_model
        
        # Delayed import to avoid dependency if not using Cellpose
        from cellpose import models
        
        print(f"Loading Cellpose model: {self.pretrained_model} (GPU={gpu})...")
        try:
            self.model = models.CellposeModel(gpu=gpu, pretrained_model=self.pretrained_model)
        except Exception as e:
            print(f"Error loading model on GPU: {e}. Switching to CPU.")
            self.model = models.CellposeModel(gpu=False, pretrained_model=self.pretrained_model)

    def inference_batch(self, patches: List[np.ndarray], batch_size: int) -> List[np.ndarray]:
        """
        Cellpose specific batch inference
        """
        # Cellpose eval supports list of images for 3D
        masks, flows, styles = self.model.eval(
            patches,
            batch_size=batch_size, # Increase internal batch size to maximize GPU usage (default is 8)
            diameter=self.diameter,
            do_3D=True,
            progress=False,
            # stitch_threshold=0.5,
            anisotropy=1.111,  # Normally z step=2um, xy step=1.8um, todo: calc this from metadata
            z_axis=0,  # Explicitly specify z_axis=0 for 3D images
        )
        
        # Check output type (if single patch, it returns array, if list, it returns list)
        if not isinstance(masks, list):
            masks = [masks]
            
        # Binarize masks
        binary_masks = [(m > 0).astype(np.uint8) for m in masks]
        return binary_masks

    def run_test(self, test_size=(64, 64, 64)):
        """
        Run a quick test segmentation on a dummy volume to verify model loading and GPU usage.
        """
        print(f"\n--- Running Cellpose Test (Size: {test_size}) ---")
        
        # Create random noise image (simulate cell-like blobs?)
        # Just random noise is enough to test if model runs
        test_img = np.random.randint(0, 255, test_size, dtype=np.uint8)
        
        print(f"Test Image Shape: {test_img.shape}")
        
        try:
            # Run inference
            print("Running inference...")
            masks, flows, styles = self.model.eval(
                test_img,
                diameter=self.diameter,
                do_3D=True,
                z_axis=0,
                progress=True
            )
            
            print(f"Inference successful. Mask shape: {masks.shape}, Max label: {masks.max()}")
            print("--- Test Complete ---\n")
            return True
        except Exception as e:
            print(f"Test failed: {e}")
            return False

def main():
    parser = argparse.ArgumentParser(description="3D Segmentation with Sliding Window")
    parser.add_argument('--test', action='store_true', help='Run a quick test on dummy data and exit')
    parser.add_argument('--sample_dir', help='Root directory of the sample')
    parser.add_argument('--channel', help='Channel name (e.g., "0" for ch0)')
    parser.add_argument('--patch_size', type=str, default="128,256,256", help='Patch size in Z,Y,X (comma separated)')
    parser.add_argument('--overlap', type=float, default=0.2, help='Overlap fraction (0-1)')
    
    # Batch processing args
    parser.add_argument('--batch_size', type=int, default=None, help='Batch size (default: auto-calculate)')
    parser.add_argument('--gpu_fraction', type=float, default=0.8, help='Target GPU memory usage fraction (default: 0.8)')
    
    # Cellpose specific args
    parser.add_argument('--model_provider', type=str, default='cellpose', choices=['cellpose'], help='Model provider')
    parser.add_argument('--diameter', type=float, default=30.0, help='Cell diameter (Cellpose)')
    parser.add_argument('--pretrained_model', type=str, default='cyto3', help='Pretrained model name (Cellpose)')  
    
    # Allow partial args if testing
    args, unknown = parser.parse_known_args()
    
    # Initialize Segmentor (needed for test too)
    # We need to handle defaults if args are missing during test
    if args.model_provider == 'cellpose':
        segmentor = CellposeSegmentor(
            pretrained_model=args.pretrained_model,
            diameter=args.diameter
        )
    
    if args.test:
        segmentor.run_test()
        return

    if not args.sample_dir or not args.channel:
        parser.error("--sample_dir and --channel are required unless --test is set")
        
    sample_dir = Path(args.sample_dir)
    channel = args.channel
    
    # Construct input path logic
    # Priority 1: ch{channel}_downsample folder (Existing logic)
    # Priority 2: ch{channel} folder (TIFF stack)
    # Priority 3: ch{channel}_downsample/volume.nii.gz (Existing logic)
    
    input_path = sample_dir / f"ch{channel}_downsample"
    if not input_path.exists():
        # Try raw chX folder
        input_path_raw = sample_dir / f"ch{channel}"
        if input_path_raw.exists() and input_path_raw.is_dir():
            print(f"Found raw TIFF folder: {input_path_raw}")
            input_path = input_path_raw
        else:
            # Try NIfTI
            input_path_nii = sample_dir / f"ch{channel}_downsample/volume.nii.gz"
            if input_path_nii.exists():
                 input_path = input_path_nii
            else:
                 raise FileNotFoundError(f"Could not find input data in {sample_dir} for channel {channel}. \nChecked: \n1. {input_path}\n2. {input_path_raw}\n3. {input_path_nii}")
    
    print(f"Processing input: {input_path}")
    
    # Output directory
    output_dir = sample_dir / f"ch{channel}_mask"
    
    # Parse patch size
    try:
        pz, py, px = map(int, args.patch_size.split(','))
        patch_size = (pz, py, px)
    except:
        print("Invalid patch size format. Using default 128,256,256")
        patch_size = (128, 256, 256)
        
    # Select Segmentor
    if args.model_provider == 'cellpose':
        segmentor = CellposeSegmentor(
            pretrained_model=args.pretrained_model,
            diameter=args.diameter
        )
    else:
        raise ValueError(f"Unknown model provider: {args.model_provider}")
    
    # Load
    image = segmentor.load_image(input_path)
    
    # Check if image is smaller than patch size, adjust if needed
    patch_size = tuple(min(i, p) for i, p in zip(image.shape, patch_size))
    
    # Run
    mask = segmentor.predict_sliding_window(
        image, 
        patch_size=patch_size, 
        overlap=args.overlap,
        batch_size=args.batch_size,
        gpu_memory_fraction=args.gpu_fraction
    )
    
    # Save
    segmentor.save_results(mask, output_dir)
    print("Segmentation complete!")

if __name__ == "__main__":
    main()
