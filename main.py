import argparse
import os
import sys
import json
import time
from pathlib import Path
import subprocess
import shutil

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

def run_command(cmd, desc):
    """Run a shell command and print output"""
    print(f"\n{'='*20} {desc} {'='*20}")
    print(f"Command: {cmd}")
    
    try:
        # Use shell=True for complex commands or Windows compatibility if needed
        # In python, list of args is safer than shell=True string
        # subprocess.check_call(cmd, shell=True) 
        
        # Using os.system for simplicity in this script context, or subprocess
        ret = os.system(cmd)
        if ret != 0:
            raise RuntimeError(f"Command failed with return code {ret}")
            
    except Exception as e:
        print(f"Error executing step: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="LSFM Full Pipeline: Registration -> Segmentation -> Analysis")
    
    # Input Data
    parser.add_argument('--sample_dir', required=True, help='Root directory of the sample (e.g., S:/Data/Sample01)')
    parser.add_argument('--target_channel', required=True, help='Target channel for segmentation (e.g., "0" or "1")')
    parser.add_argument('--register_channel', required=True, help='Channel used for registration (usually autofluorescence, e.g., "1")')
    
    # Configs
    parser.add_argument('--config', default='config.json', help='Main pipeline config')
    parser.add_argument('--density_cfg', default='src/modules/registration/add_id_ytw.json', help='Density analysis hierarchy config')
    
    # Flags to skip steps
    parser.add_argument('--skip_preprocessing', action='store_true', help='Skip Zarr conversion/downsampling')
    parser.add_argument('--skip_registration', action='store_true', help='Skip ANTs registration')
    parser.add_argument('--skip_segmentation', action='store_true', help='Skip Cellpose segmentation')
    parser.add_argument('--skip_analysis', action='store_true', help='Skip density analysis')
    
    # Parameters overrides
    parser.add_argument('--workers', type=int, default=4, help='Dask workers for segmentation')
    parser.add_argument('--test', action='store_true', help='Run in test mode (quick checks only)')
    
    args = parser.parse_args()
    
    # --- TEST MODE ---
    if args.test:
        print("Running Pipeline in TEST Mode...")
        # Test 1: Cellpose Model Load & Inference
        cmd = "python src/modules/segmentation/cellpose_dis.py --test"
        run_command(cmd, "Test 1: Cellpose Model Check")
        
        print("\nTest passed! Basic environment and models are working.")
        return
    # -----------------
    
    sample_dir = Path(args.sample_dir)
    target_ch = args.target_channel
    reg_ch = args.register_channel
    
    # Paths setup
    raw_tiff_dir = sample_dir / f"ch{target_ch}"
    zarr_path = sample_dir / f"ch{target_ch}.zarr"
    mask_zarr_path = sample_dir / f"ch{target_ch}_mask.zarr"
    mask_tiff_dir = sample_dir / f"ch{target_ch}_mask" # Exported TIFF mask
    downsample_mask_dir = sample_dir / f"ch{target_ch}_downsample_mask"
    
    # 1. Preprocessing (TIFF -> Zarr & Downsampling)
    if not args.skip_preprocessing:
        # 1.1 TIFF to Zarr
        if not zarr_path.exists():
            cmd = f"python src/modules/preprocessing/tiff_to_zarr.py --input \"{raw_tiff_dir}\" --output \"{zarr_path}\""
            run_command(cmd, "Step 1.1: Convert Raw TIFF to Zarr")
        else:
            print(f"Zarr file exists, skipping conversion: {zarr_path}")

        # 1.2 Downsample Registration Channel (for ANTs)
        reg_downsample_dir = sample_dir / f"ch{reg_ch}_downsample"
        
        if not (reg_downsample_dir / "volume.nii.gz").exists():
            # ... (downsample logic) ...
            # Assuming we have a config for resolution, or use manual factor
            # Here we hardcode a manual factor or need a resolution.json
            # Let's assume resolution.json exists in sample_dir or we use manual
            res_cfg = sample_dir / "resolution.json"
            if res_cfg.exists():
                cmd = f"python src/modules/preprocessing/downsample.py --input_folder \"{sample_dir}/ch{reg_ch}\" --resolution_config \"{res_cfg}\""
            else:
                # Fallback to manual downsample if no config (Adjust as needed!)
                print("Warning: resolution.json not found, using default downsampling (0.72, 0.72, 0.8)")
                cmd = f"python src/modules/preprocessing/downsample.py --input_folder \"{sample_dir}/ch{reg_ch}\" --factor \"0.72,0.72,0.8\""
            
            run_command(cmd, "Step 1.2: Downsample Registration Channel")
        
        # 1.3 Downsample Target Channel (Optional)
        # We don't strictly need it for density analysis if we map Atlas -> Image.
        # But if we map Image -> Atlas, we need it.
        # Let's skip for now unless needed.

    # 2. Registration (ANTs)
    if not args.skip_registration:
        # We need atlas paths. Assuming they are in a standard location or passed via config.
        # Hardcoding standard paths for now based on project structure
        atlas_dir = Path("s:/Yifu/Allen_brainatlas") # Adjust this absolute path!
        atlas_img = atlas_dir / "atlas.tiff" # or average_template_25.tif
        atlas_lbl = atlas_dir / "atlas_label.tiff" # or annotation_25.tif
        
        # Bidirectional registration (we usually need Atlas -> Image for density in native space)
        # But analyze_density.py usually expects: Mask (Native) and Label (Native/Warped)
        # So we run 'atlas2image' mode.
        
        cmd = f"python src/modules/registration/ANTs_registration.py \
            --sample_dir \"{sample_dir}\" \
            --target_channel {target_ch} \
            --register_channel {reg_ch} \
            --atlas_image \"{atlas_img}\" \
            --atlas_label \"{atlas_lbl}\" \
            --mode atlas2image \
            --save_registered_image"
            
        run_command(cmd, "Step 2: ANTs Registration (Atlas -> Image)")

    # 3. Segmentation (Cellpose Distributed)
    if not args.skip_segmentation:
        if not mask_zarr_path.exists():
            cmd = f"python src/modules/segmentation/cellpose_distributed.py \
                --input_zarr \"{zarr_path}\" \
                --output_zarr \"{mask_zarr_path}\" \
                --workers {args.workers} \
                --pretrained_model cpsam \
                --diameter 10" \
                # Add --output_tiff if you want immediate TIFF export, but we can do it separate
            
            run_command(cmd, "Step 3.1: Distributed Segmentation")
        
        # Export Mask Zarr to TIFF (Required for analyze_density.py currently as it reads TIFF folders)
        if not mask_tiff_dir.exists():
            print(f"Exporting Mask Zarr to TIFF folder: {mask_tiff_dir}")
            # Use inline python to call export function from cellpose_distributed
            cmd = f"python -c \"from src.modules.segmentation.cellpose_distributed import export_zarr_to_tiff; export_zarr_to_tiff(r'{mask_zarr_path}', r'{mask_tiff_dir}')\""
            run_command(cmd, "Step 3.2: Export Mask Zarr to TIFF")

    # 4. Density Analysis
    if not args.skip_analysis:
        # We need:
        # 1. Mask Folder (Native resolution, TIFF) -> mask_tiff_dir
        # 2. Label Folder (Native resolution, Warped from Atlas) -> Created by ANTs_registration in step 2
        
        # Updated path: chX_upsampled_label
        warped_label_dir = sample_dir / f"ch{target_ch}_upsampled_atlas_label"
        
        # Check if we need to downsample the mask?
        # analyze_density.py typically runs on full res mask + full res label.
        # But if we downsample the mask, we need to downsample the label too.
        # Or upsample the label to full res?
        # Running on 500GB full res mask + full res label is very slow and memory intensive.
        # Usually we downsample the MASK to match the Registration resolution, 
        # OR we upsample the Label to full resolution (which ANTs script does).
        
        # If we use the upsampled label (Step 2 output), we use full res mask.
        # Ensure full res mask exists (TIFF).
        if not mask_tiff_dir.exists():
             print(f"Exporting Mask Zarr to TIFF folder: {mask_tiff_dir}")
             # Use inline python to call export function from cellpose_distributed
             cmd = f"python -c \"from src.modules.segmentation.cellpose_distributed import export_zarr_to_tiff; export_zarr_to_tiff(r'{mask_zarr_path}', r'{mask_tiff_dir}')\""
             run_command(cmd, "Step 3.2: Export Mask Zarr to TIFF")
        
        if not warped_label_dir.exists():
            print(f"Error: Warped label folder not found at {warped_label_dir}. Registration failed?")
            sys.exit(1)
            
        output_excel = sample_dir / f"density_results_ch{target_ch}.xlsx"
        
        cmd = f"python src/modules/registration/analyze_density.py \
            --mask_folder \"{mask_tiff_dir}\" \
            --label_folder \"{warped_label_dir}\" \
            --cfg \"{args.density_cfg}\" \
            --output \"{output_excel}\""
            
        run_command(cmd, "Step 4: Density Analysis")

    print("\n" + "="*50)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("="*50)

if __name__ == "__main__":
    main()
