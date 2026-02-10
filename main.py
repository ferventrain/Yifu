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

def load_config(config_path):
    """Load JSON config"""
    with open(config_path, 'r') as f:
        return json.load(f)

def main():
    parser = argparse.ArgumentParser(description="LSFM Full Pipeline: Segmentation -> Registration -> Analysis")
    
    # Core Args
    parser.add_argument('--config', default='config.json', help='Path to config.json')
    parser.add_argument('--sample_dir', help='Root directory of the sample (Overrides config)')
    
    # Override Flags
    parser.add_argument('--skip_preprocessing', action='store_true', help='Skip Zarr conversion/downsampling')
    parser.add_argument('--skip_registration', action='store_true', help='Skip ANTs registration')
    parser.add_argument('--skip_segmentation', action='store_true', help='Skip Cellpose segmentation')
    parser.add_argument('--skip_analysis', action='store_true', help='Skip density analysis')
    parser.add_argument('--test', action='store_true', help='Run in test mode (quick checks only)')
    
    args = parser.parse_args()
    
    # 1. Load Config
    if not os.path.exists(args.config):
        print(f"Error: Config file not found: {args.config}")
        print("Please run with --config path/to/config.json or generate one from template.")
        sys.exit(1)
        
    print(f"Loading config: {args.config}")
    cfg = load_config(args.config)
    
    # 2. Resolve Parameters (CLI overrides Config)
    # Sample Dir
    if args.sample_dir:
        sample_dir = Path(args.sample_dir)
    else:
        # If not in CLI, check if it's in config (optional, usually sample_dir is dynamic)
        # But let's assume sample_dir is always CLI for flexibility
        print("Error: --sample_dir is required (unless hardcoded in script, but CLI is preferred)")
        sys.exit(1)
        
    # Channels
    signal_ch = cfg['input']['channels']['signal']
    reg_ch = cfg['input']['channels']['registration']
    
    # Preprocessing
    ds_cfg = cfg['preprocessing']['downsample']
    zarr_cfg = cfg['preprocessing']['zarr']
    
    # Registration
    reg_cfg = cfg['registration']
    mode = 'atlas2image' # Always default to atlas2image for main pipeline
    
    # Segmentation
    seg_cfg = cfg['segmentation']
    seg_method = seg_cfg['method']
    
    # Analysis
    analysis_cfg = cfg['analysis']
    density_cfg_path = analysis_cfg.get('density_config', 'src/modules/registration/add_id_ytw.json')

    # --- TEST MODE ---
    if args.test:
        print("Running Pipeline in TEST Mode...")
        cmd = "python src/modules/segmentation/cellpose_dis.py --test"
        run_command(cmd, "Test 1: Cellpose Model Check")
        return
    # -----------------
    
    # Paths setup
    raw_tiff_dir = sample_dir / f"ch{signal_ch}"
    zarr_path = sample_dir / f"ch{signal_ch}.zarr"
    mask_zarr_path = sample_dir / f"ch{signal_ch}_mask.zarr"
    mask_tiff_dir = sample_dir / f"ch{signal_ch}_mask" 
    
    # 1. Preprocessing (TIFF -> Zarr & Downsampling)
    if not args.skip_preprocessing:
        # 1.1 TIFF to Zarr
        if not zarr_path.exists():
            chunk_str = ",".join(map(str, zarr_cfg['chunk_size']))
            cmd = f"python src/modules/preprocessing/tiff_to_zarr.py \
                --input \"{raw_tiff_dir}\" \
                --output \"{zarr_path}\" \
                --chunk_size \"{chunk_str}\""
            run_command(cmd, "Step 1.1: Convert Raw TIFF to Zarr")
        else:
            print(f"Zarr file exists, skipping conversion: {zarr_path}")

        # 1.2 Downsample Registration Channel (for ANTs)
        reg_downsample_dir = sample_dir / f"ch{reg_ch}_downsample"
        
        if not (reg_downsample_dir / "volume.nii.gz").exists():
            # Calculate downsample factors from config
            try:
                input_res = cfg['input']['resolution_xyz'] # [x, y, z]
                target_res = ds_cfg['target_resolution_xyz'] # [x, y, z]
                
                # Calculate factors: source / target
                # Note: config is in microns usually, ratio is unit-independent
                factors = [s / t for s, t in zip(input_res, target_res)]
                
                # Convert to z, y, x for downsample.py (which uses ndimage.zoom on ZYX stack)
                factors_zyx = factors[::-1]
                factor_str = ",".join([f"{f:.4f}" for f in factors_zyx])
                
                print(f"Calculated downsample factors (z,y,x): {factor_str} from config")
                
                cmd = f"python src/modules/preprocessing/downsample.py --input_folder \"{sample_dir}/ch{reg_ch}\" --factor \"{factor_str}\""
            except Exception as e:
                print(f"Error calculating downsample factors from config: {e}")
                print("Config content:", cfg)
                sys.exit(1)
            
            run_command(cmd, "Step 1.2: Downsample Registration Channel")

    # 2. Segmentation
    if not args.skip_segmentation:
        if not mask_zarr_path.exists():
            
            if seg_method == 'cellpose':
                cp_cfg = seg_cfg['cellpose']
                workers = cp_cfg['workers']
                model = cp_cfg['model']
                diameter = cp_cfg['diameter']
                
                cmd = f"python src/modules/segmentation/cellpose_distributed.py \
                    --input_zarr \"{zarr_path}\" \
                    --output_zarr \"{mask_zarr_path}\" \
                    --workers {workers} \
                    --pretrained_model {model} \
                    --diameter {diameter}"
                    
            elif seg_method == 'threshold':
                th_cfg = seg_cfg['threshold']
                thresh_val = th_cfg['value']
                sigma = th_cfg['sigma']
                
                cmd = f"python src/modules/segmentation/intensity_threshold_segmentor.py \
                    --input_zarr \"{zarr_path}\" \
                    --output_zarr \"{mask_zarr_path}\" \
                    --threshold {thresh_val} \
                    --sigma {sigma}"
            else:
                print(f"Unknown segmentation method: {seg_method}")
                sys.exit(1)
            
            run_command(cmd, f"Step 2.1: Segmentation ({seg_method})")
        
        # Export Mask Zarr to TIFF
        if not mask_tiff_dir.exists():
            print(f"Exporting Mask Zarr to TIFF folder: {mask_tiff_dir}")
            cmd = f"python -c \"from src.modules.segmentation.cellpose_distributed import export_zarr_to_tiff; export_zarr_to_tiff(r'{mask_zarr_path}', r'{mask_tiff_dir}')\""
            run_command(cmd, "Step 2.2: Export Mask Zarr to TIFF")

    # 3. Registration (ANTs)
    if not args.skip_registration:
        atlas_img = reg_cfg['atlas_path']
        atlas_lbl = reg_cfg['annotation_path']
        # mode = reg_cfg['mode'] # Already loaded above
        
        cmd = f"python src/modules/registration/ANTs_registration.py \
            --sample_dir \"{sample_dir}\" \
            --signal_channel {signal_ch} \
            --register_channel {reg_ch} \
            --atlas_image \"{atlas_img}\" \
            --atlas_label \"{atlas_lbl}\" \
            --mode {mode} \
            --save_registered_image \
            --save_transforms \
            --config \"{args.config}\""
            
        run_command(cmd, "Step 3: ANTs Registration (Atlas -> Image)")

    # 4. Density Analysis
    if not args.skip_analysis:
        
        # Determine analysis mode and paths
        # Always atlas2image mode for main pipeline
        warped_label_dir = sample_dir / f"ch{signal_ch}_upsampled_label"
        label_folder_arg = warped_label_dir
        
        if not warped_label_dir.exists():
            print(f"Error: Warped label folder not found at {warped_label_dir}. Registration failed?")
            sys.exit(1)

        if not mask_tiff_dir.exists():
             print(f"Exporting Mask Zarr to TIFF folder: {mask_tiff_dir}")
             cmd = f"python -c \"from src.modules.segmentation.cellpose_distributed import export_zarr_to_tiff; export_zarr_to_tiff(r'{mask_zarr_path}', r'{mask_tiff_dir}')\""
             run_command(cmd, "Step 2.2: Export Mask Zarr to TIFF")
        
        output_excel = sample_dir / f"density_results_ch{signal_ch}.xlsx"
        
        cmd = f"python src/modules/registration/analyze_density.py \
            --mask_folder \"{mask_tiff_dir}\" \
            --label_folder \"{label_folder_arg}\" \
            --cfg \"{density_cfg_path}\" \
            --output \"{output_excel}\""
            
        run_command(cmd, "Step 4: Density Analysis")

    print("\n" + "="*50)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("="*50)

if __name__ == "__main__":
    main()
