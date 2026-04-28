import argparse
import json
import subprocess
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

MAIN_PIPELINE_REGISTRATION_MODE = "atlas2image"


def run_command(cmd, desc):
    """Run a shell command and print output."""
    print(f"\n{'=' * 20} {desc} {'=' * 20}")
    print(f"Command: {cmd}")

    try:
        subprocess.run(cmd, shell=True, check=True)
    except subprocess.CalledProcessError as exc:
        print(f"Error executing step: {exc}")
        sys.exit(1)


def load_config(config_path):
    """Load JSON config."""
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def directory_has_files(path):
    return path.exists() and any(path.iterdir())


def format_csv(values):
    return ",".join(map(str, values))


def run_tiff_to_zarr(input_path, output_path, chunk_size, desc):
    chunk_str = format_csv(chunk_size)
    cmd = (
        f'python pipeline_modules/preprocessing/tiff_to_zarr.py '
        f'--input "{input_path}" '
        f'--output "{output_path}" '
        f'--chunk_size "{chunk_str}"'
    )
    run_command(cmd, desc)


def resolve_density_cfg_path(analysis_cfg):
    density_cfg_raw = analysis_cfg.get(
        "density_config",
        "pipeline_modules/registration/Region_Csv_Rev1_updated.CSV",
    )
    density_cfg_path = Path(density_cfg_raw)
    if not density_cfg_path.is_absolute():
        density_cfg_path = project_root / density_cfg_path

    if density_cfg_path.exists():
        return density_cfg_path

    fallback_density_cfg = (
        project_root / "pipeline_modules" / "registration" / "Region_Csv_Rev1_updated.CSV"
    )
    if fallback_density_cfg.exists():
        return fallback_density_cfg

    print(f"Error: Density config not found: {density_cfg_path}")
    sys.exit(1)


def calculate_downsample_factor_str(input_res, target_res):
    try:
        factors = [source / target for source, target in zip(input_res, target_res)]
    except Exception as exc:
        raise ValueError(f"Failed to calculate downsample factors: {exc}") from exc

    factors_zyx = factors[::-1]
    return ",".join(f"{factor:.4f}" for factor in factors_zyx)


def ensure_signal_zarr(sample_dir, signal_ch, preprocessing_cfg, zarr_cfg):
    from pipeline_modules.preprocessing.preprocessor import Preprocessor

    raw_tiff_dir = sample_dir / f"ch{signal_ch}"
    zarr_path = sample_dir / f"ch{signal_ch}.zarr"
    current_signal_tiff_dir = raw_tiff_dir

    preprocessor = Preprocessor(preprocessing_cfg)
    if preprocessor.steps:
        enhanced_dir = sample_dir / f"ch{signal_ch}_preprocessed"
        success = preprocessor.process_folder(
            input_folder=current_signal_tiff_dir,
            output_folder=enhanced_dir,
            max_workers=None,
            resume=True,
        )
        if not success:
            print("Preprocessing failed, exiting.")
            sys.exit(1)
        current_signal_tiff_dir = enhanced_dir
    else:
        print("No preprocessing enhancement steps enabled, using raw TIFF directly.")

    if zarr_path.exists():
        print(f"Zarr file exists, skipping conversion: {zarr_path}")
    else:
        run_tiff_to_zarr(
            current_signal_tiff_dir,
            zarr_path,
            zarr_cfg["chunk_size"],
            "Step 1.1: Convert Signal TIFF to Zarr",
        )

    return zarr_path


def ensure_registration_downsample(sample_dir, reg_ch, input_res, target_res):
    reg_downsample_dir = sample_dir / f"ch{reg_ch}_downsample"
    reg_nifti_path = reg_downsample_dir / "volume.nii.gz"

    if reg_nifti_path.exists():
        print(f"Registration downsample exists, skipping: {reg_nifti_path}")
        return

    try:
        factor_str = calculate_downsample_factor_str(input_res, target_res)
        print(f"Calculated downsample factors (z,y,x): {factor_str} from config")
    except ValueError as exc:
        print(f"Error calculating downsample factors from config: {exc}")
        sys.exit(1)

    cmd = (
        f'python pipeline_modules/preprocessing/downsample.py '
        f'--input_folder "{sample_dir / f"ch{reg_ch}"}" '
        f'--factor "{factor_str}"'
    )
    run_command(cmd, "Step 1.2: Downsample Registration Channel")


def build_segmentation_command(seg_cfg, zarr_path, mask_zarr_path):
    seg_method = seg_cfg["method"]

    if seg_method == "cellpose":
        cellpose_script = project_root / "pipeline_modules" / "segmentation" / "cellpose_distributed.py"
        if not cellpose_script.exists():
            print(f"Error: cellpose segmentation script not found: {cellpose_script}")
            sys.exit(1)
        cp_cfg = seg_cfg["cellpose"]
        return (
            f'python pipeline_modules/segmentation/cellpose_distributed.py '
            f'--input_zarr "{zarr_path}" '
            f'--output_zarr "{mask_zarr_path}" '
            f'--workers {cp_cfg["workers"]} '
            f'--pretrained_model {cp_cfg["model"]} '
            f'--diameter {cp_cfg["diameter"]}'
        )

    if seg_method == "threshold":
        th_cfg = seg_cfg["threshold"]
        return (
            f'python pipeline_modules/segmentation/intensity_threshold_segmentor.py '
            f'--input_zarr "{zarr_path}" '
            f'--output_zarr "{mask_zarr_path}" '
            f'--threshold {th_cfg["value"]} '
            f'--sigma {th_cfg["sigma"]}'
        )

    if seg_method == "cfos_unet":
        model_cfg = seg_cfg["cfos_unet"]
        cmd = (
            f'python pipeline_modules/segmentation/cfos_unet_inference.py '
            f'--input_zarr "{zarr_path}" '
            f'--output_zarr "{mask_zarr_path}" '
            f'--checkpoint_path "{model_cfg["checkpoint_path"]}" '
            f'--dataset_name "{model_cfg.get("dataset_name", "0")}" '
            f'--overlap {model_cfg.get("overlap", 0.25)} '
            f'--batch_size {model_cfg.get("batch_size", 1)} '
            f'--device "{model_cfg.get("device", "auto")}" '
            f'--foreground_class {model_cfg.get("foreground_class", 1)} '
            f'--probability_threshold {model_cfg.get("probability_threshold", 0.5)} '
            f'--output_mode "{model_cfg.get("output_mode", "binary")}" '
            f'--output_dtype "{model_cfg.get("output_dtype", "uint8")}"'
        )
        if model_cfg.get("patch_size"):
            cmd += f' --patch_size "{format_csv(model_cfg["patch_size"])}"'
        if model_cfg.get("chunk_size"):
            cmd += f' --chunk_size "{format_csv(model_cfg["chunk_size"])}"'
        if model_cfg.get("process_existing_only", False):
            cmd += ' --process_existing_only'
        return cmd

    print(f"Unknown segmentation method: {seg_method}")
    sys.exit(1)


def ensure_segmentation_outputs(sample_dir, signal_ch, zarr_path, seg_cfg):
    mask_zarr_path = sample_dir / f"ch{signal_ch}_mask.zarr"
    mask_tiff_dir = sample_dir / f"ch{signal_ch}_mask"

    if directory_has_files(mask_tiff_dir):
        print(f"Found existing mask folder at {mask_tiff_dir}, skipping segmentation.")
        return mask_zarr_path, mask_tiff_dir

    if not mask_zarr_path.exists():
        seg_cmd = build_segmentation_command(seg_cfg, zarr_path, mask_zarr_path)
        run_command(seg_cmd, f'Step 2.1: Segmentation ({seg_cfg["method"]})')
    else:
        print(f"Found existing mask Zarr at {mask_zarr_path}, skipping segmentation.")

    if not directory_has_files(mask_tiff_dir):
        export_cmd = (
            f'python -c "from pipeline_modules.segmentation '
            f'import export_zarr_to_tiff; '
            f"export_zarr_to_tiff(r'{mask_zarr_path}', r'{mask_tiff_dir}')\""
        )
        run_command(export_cmd, "Step 2.2: Export Mask Zarr to TIFF")

    return mask_zarr_path, mask_tiff_dir


def ensure_registration_outputs(sample_dir, signal_ch, reg_ch, reg_cfg, config_path):
    warped_label_dir = sample_dir / "upsampled_atlas_label"

    if directory_has_files(warped_label_dir):
        print(f"Registration output exists at {warped_label_dir}. Skipping Step 3.")
        return warped_label_dir

    cmd = (
        f'python pipeline_modules/registration/ANTs_registration.py '
        f'--sample_dir "{sample_dir}" '
        f'--signal_channel {signal_ch} '
        f'--register_channel {reg_ch} '
        f'--atlas_image "{reg_cfg["atlas_path"]}" '
        f'--atlas_label "{reg_cfg["annotation_path"]}" '
        f'--mode {MAIN_PIPELINE_REGISTRATION_MODE} '
        f'--save_registered_image '
        f'--save_transforms '
        f'--config "{config_path}"'
    )
    run_command(cmd, "Step 3: ANTs Registration (Atlas -> Image)")
    return warped_label_dir


def ensure_mask_zarr(mask_tiff_dir, mask_zarr_path, zarr_cfg):
    if mask_zarr_path.exists():
        return

    if not directory_has_files(mask_tiff_dir):
        print(f"Error: Mask Zarr not found at {mask_zarr_path}, and mask TIFF folder is unavailable.")
        sys.exit(1)

    run_tiff_to_zarr(mask_tiff_dir, mask_zarr_path, zarr_cfg["chunk_size"], "Step 4.0: Convert Mask TIFF to Zarr")


def ensure_label_zarr(warped_label_dir, warped_label_zarr_path, zarr_cfg):
    if warped_label_zarr_path.exists():
        return

    run_tiff_to_zarr(
        warped_label_dir,
        warped_label_zarr_path,
        zarr_cfg["chunk_size"],
        "Step 4.1: Convert Registered Label TIFF to Zarr",
    )


def run_density_analysis(
    sample_dir,
    signal_ch,
    zarr_path,
    mask_zarr_path,
    zarr_cfg,
    density_cfg_path,
    resolution_xyz,
):
    warped_label_dir = sample_dir / "upsampled_atlas_label"
    warped_label_zarr_path = sample_dir / "upsampled_atlas_label.zarr"

    if not directory_has_files(warped_label_dir):
        print(f"Error: Warped label folder not found at {warped_label_dir}. Registration failed?")
        sys.exit(1)

    mask_tiff_dir = sample_dir / f"ch{signal_ch}_mask"
    ensure_mask_zarr(mask_tiff_dir, mask_zarr_path, zarr_cfg)
    ensure_label_zarr(warped_label_dir, warped_label_zarr_path, zarr_cfg)

    output_excel = sample_dir / f"density_results_ch{signal_ch}.xlsx"
    resolution_xyz_str = format_csv(resolution_xyz)

    cmd = (
        f'python pipeline_modules/registration/region_signal_analysis_zarr_graph.py '
        f'--mask_zarr "{mask_zarr_path}" '
        f'--label_zarr "{warped_label_zarr_path}" '
        f'--signal_zarr "{zarr_path}" '
        f'--cfg "{density_cfg_path}" '
        f'--output "{output_excel}" '
        f'--min_voxels 10 '
        f'--foreground_mode equal '
        f'--foreground_label 1 '
        f'--resolution_xyz "{resolution_xyz_str}" '
        f'--pass1_workers 4'
    )
    run_command(cmd, "Step 4: Density Analysis")


def main():
    parser = argparse.ArgumentParser(
        description="LSFM main pipeline: preprocessing -> segmentation -> registration -> analysis"
    )
    parser.add_argument("--config", default="config.json", help="Path to config.json")
    parser.add_argument("--sample_dir", help="Root directory of the sample")
    parser.add_argument("--test", action="store_true", help="Run in test mode (quick checks only)")
    args = parser.parse_args()

    if args.test:
        print("Running Pipeline in TEST Mode...")
        run_command(
            'python -c "from pipeline_modules.segmentation import load_capability_manifest; '
            'import json; print(json.dumps(load_capability_manifest(), ensure_ascii=False)[:200])"',
            "Test 1: Segmentation Module Import Check",
        )
        return

    if not args.sample_dir:
        print("Error: --sample_dir is required.")
        sys.exit(1)

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Config file not found: {args.config}")
        print("Please run with --config path/to/config.json or generate one from template.")
        sys.exit(1)

    print(f"Loading config: {config_path}")
    cfg = load_config(config_path)
    sample_dir = Path(args.sample_dir)

    signal_ch = cfg["input"]["channels"]["signal"]
    reg_ch = cfg["input"]["channels"]["registration"]
    preprocessing_cfg = cfg["preprocessing"]
    zarr_cfg = preprocessing_cfg["zarr"]
    seg_cfg = cfg["segmentation"]
    reg_cfg = cfg["registration"]
    density_cfg_path = resolve_density_cfg_path(cfg["analysis"])

    zarr_path = ensure_signal_zarr(sample_dir, signal_ch, preprocessing_cfg, zarr_cfg)
    ensure_registration_downsample(
        sample_dir,
        reg_ch,
        cfg["input"]["resolution_xyz"],
        preprocessing_cfg["downsample"]["target_resolution_xyz"],
    )
    mask_zarr_path, _ = ensure_segmentation_outputs(sample_dir, signal_ch, zarr_path, seg_cfg)
    ensure_registration_outputs(sample_dir, signal_ch, reg_ch, reg_cfg, config_path)
    run_density_analysis(
        sample_dir,
        signal_ch,
        zarr_path,
        mask_zarr_path,
        zarr_cfg,
        density_cfg_path,
        cfg["input"]["resolution_xyz"],
    )

    print("\n" + "=" * 50)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 50)


if __name__ == "__main__":
    main()
