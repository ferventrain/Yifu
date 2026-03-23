import argparse
import os
import shutil
from pathlib import Path
import numpy as np
import zarr
import dask.array as da
import tifffile
from tqdm import tqdm
import warnings

# Silence specific warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="google.auth")
warnings.filterwarnings("ignore", category=FutureWarning, module="google.oauth2")
warnings.filterwarnings("ignore", category=FutureWarning, module="google.api_core")

from cellpose.contrib.distributed_segmentation import distributed_eval, myLocalCluster

def run_distributed_segmentation(input_zarr_path, output_zarr_path, 
                               pretrained_model='cyto3', diameter=30, 
                               block_size=(128, 256, 256),
                               n_workers=4, gpu=True, memory_limit='128GB'):
    
    print(f"Opening Input Zarr: {input_zarr_path}")
    input_zarr = zarr.open(input_zarr_path, mode='r')
    
    # Handle OME-Zarr (Group with '0' dataset)
    if isinstance(input_zarr, zarr.Group):
        if '0' in input_zarr:
            input_zarr = input_zarr['0']
    
    print(f"Input Shape: {input_zarr.shape}")
    
    # Setup Dask Cluster
    # Note: ncpus is physical cores per worker. 
    # If you have 1 GPU and want 4 workers, they will share the GPU (Cellpose handles this via torch)
    # But usually for GPU bound tasks, n_workers=1 is best if you only have 1 GPU.
    # If you have 4 GPUs, n_workers=4 is great.
    
    print(f"Setting up Dask LocalCluster with {n_workers} workers...")
    
    # Determine ncpus per worker roughly
    import multiprocessing
    total_cores = multiprocessing.cpu_count()
    ncpus_per_worker = max(1, total_cores // n_workers)
    
    # Configure Cluster
    cluster_kwargs = {
        'n_workers': n_workers,
        'ncpus': ncpus_per_worker,
        'threads_per_worker': 1,
        'memory_limit': memory_limit, # Adjust based on your RAM    
    }
    
    # Cellpose Arguments
    eval_kwargs = {
        'diameter': diameter,
        'do_3D': False, # Changed to False for speed. Uses stitching instead of 3D inference.
        'z_axis': 0,
        'stitch_threshold': 0.5,
        'batch_size': 32,
    }
    
    model_kwargs = {
        'gpu': gpu,
        'pretrained_model': pretrained_model,
    }

    # Run Distributed Eval
    print("Starting distributed evaluation...")
    with myLocalCluster(**cluster_kwargs) as cluster:
        # Create output directory if needed
        output_path = Path(output_zarr_path)
        if output_path.exists():
            print(f"Warning: Output path {output_path} exists. It might be overwritten.")
            
        result_zarr, boxes = distributed_eval(
            input_zarr,
            blocksize=block_size,
            write_path=str(output_zarr_path),
            model_kwargs=model_kwargs,
            eval_kwargs=eval_kwargs,
            cluster=cluster
        )
        
    print(f"Segmentation complete. Results saved to {output_zarr_path}")
    return result_zarr

def export_zarr_to_tiff(zarr_path, output_folder):
    """Export Zarr array to TIFF stack"""
    print(f"Exporting {zarr_path} to TIFF stack in {output_folder}...")
    z = zarr.open(zarr_path, mode='r')
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    
    # Iterate and save
    for i in tqdm(range(z.shape[0]), desc="Exporting TIFFs"):
        img = z[i]
        tifffile.imwrite(output_folder / f"mask_{i:04d}.tiff", img, compression='lzw')

def main():
    parser = argparse.ArgumentParser(description="Run Distributed Cellpose Segmentation")
    parser.add_argument('--input_zarr', required=True, help='Path to input .zarr directory')
    parser.add_argument('--output_zarr', required=True, help='Path for output .zarr directory')
    parser.add_argument('--output_tiff', help='Optional path to export results as TIFF stack')
    parser.add_argument('--pretrained_model', default='cyto3', help='Cellpose pretrained model')
    parser.add_argument('--diameter', type=float, default=30.0, help='Cell diameter')
    parser.add_argument('--block_size', type=str, default="128,256,256", help='Block size z,y,x')
    parser.add_argument('--workers', type=int, default=4, help='Number of dask workers')
    parser.add_argument('--memory_limit', default='auto', help='Memory limit per worker (e.g. "16GB", "auto")')
    parser.add_argument('--no_gpu', action='store_true', help='Disable GPU usage')
    
    args = parser.parse_args()
    
    try:
        bz, by, bx = map(int, args.block_size.split(','))
        block_size = (bz, by, bx)
    except:
        block_size = (128, 256, 256)

    # Run segmentation
    run_distributed_segmentation(
        args.input_zarr,
        args.output_zarr,
        pretrained_model=args.pretrained_model,
        diameter=int(args.diameter), # Cast to int to avoid float slice indices error in distributed_segmentation
        block_size=block_size,
        n_workers=args.workers,
        gpu=not args.no_gpu,
        memory_limit=args.memory_limit
    )
    
    # Export if requested
    if args.output_tiff:
        export_zarr_to_tiff(args.output_zarr, args.output_tiff)

if __name__ == "__main__":
    main()
