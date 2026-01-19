import sys
import os
import numpy as np
import time

def print_section(title):
    print(f"\n{'='*20} {title} {'='*20}")

def test_imports():
    print_section("Testing Imports")
    modules = [
        'numpy', 'scipy', 'tifffile', 'dask', 'distributed',
        'ants', 'SimpleITK', 'cellpose', 'torch'
    ]
    
    for mod in modules:
        try:
            __import__(mod)
            print(f"[OK] {mod}")
        except ImportError as e:
            print(f"[FAIL] {mod}: {e}")
            sys.exit(1)

def test_torch_gpu():
    print_section("Testing PyTorch & GPU")
    import torch
    print(f"PyTorch Version: {torch.__version__}")
    if torch.cuda.is_available():
        print(f"[OK] CUDA Available: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
    else:
        print("[WARNING] CUDA NOT Available. Running on CPU.")

def test_ants_registration():
    print_section("Testing ANTs Registration (Simulation)")
    import ants
    
    # Create dummy 3D images
    shape = (64, 64, 64)
    print(f"Creating dummy images {shape}...")
    
    # Fixed image: a sphere
    fixed_np = np.zeros(shape, dtype='float32')
    c = 32
    r = 15
    z, y, x = np.ogrid[:64, :64, :64]
    mask = (x-c)**2 + (y-c)**2 + (z-c)**2 <= r**2
    fixed_np[mask] = 1.0
    fixed_img = ants.from_numpy(fixed_np)
    
    # Moving image: shifted sphere
    moving_np = np.zeros(shape, dtype='float32')
    c_move = 36 # Shifted by 4 pixels
    mask_move = (x-c_move)**2 + (y-c_move)**2 + (z-c)**2 <= r**2
    moving_np[mask_move] = 1.0
    moving_img = ants.from_numpy(moving_np)
    
    print("Running SyN registration (fast mode)...")
    try:
        mytx = ants.registration(fixed=fixed_img, moving=moving_img, type_of_transform='SyN')
        print(f"[OK] Registration output keys: {mytx.keys()}")
    except Exception as e:
        print(f"[FAIL] ANTs Registration failed: {e}")
        # Don't exit, maybe just ANTs issue

def test_cellpose_segmentation():
    print_section("Testing Cellpose Segmentation (Simulation)")
    from cellpose import models
    import torch
    
    use_gpu = torch.cuda.is_available()
    print(f"Initializing Cellpose model (GPU={use_gpu})...")
    
    try:
        # Using 'cyto' model which is standard. 
        # Note: This attempts to download the model if not present.
        # In a closed container without net, this might fail if model not cached.
        # We wrap in try-except.
        model = models.CellposeModel(gpu=use_gpu)
        
        # Create dummy image: 2D plane with some circles
        img = np.zeros((256, 256), dtype=np.uint8)
        # Draw some fake cells
        for r, c in [(50,50), (100,100), (150,150)]:
            y, x = np.ogrid[:256, :256]
            mask = (x-c)**2 + (y-r)**2 <= 20**2
            img[mask] = 255
            
        print("Running inference...")
        masks, flows, styles = model.eval(img, diameter=40, channels=[0,0])
        
        n_cells = masks.max()
        print(f"[OK] Segmentation complete. Found {n_cells} cells.")
        
    except Exception as e:
        print(f"[WARNING] Cellpose test failed (possibly model download issue): {e}")

if __name__ == "__main__":
    print(f"Starting Environment Test on {sys.platform}")
    test_imports()
    test_torch_gpu()
    test_ants_registration()
    test_cellpose_segmentation()
    print_section("Test Complete")
