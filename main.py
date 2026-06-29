import argparse
import json
import subprocess
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))
PYTHON_EXE = sys.executable

MAIN_PIPELINE_REGISTRATION_MODE = "atlas2image"
PIPELINE_STEP_COUNT = 5


def print_pipeline_banner(sample_dir, config_path):
    print("\n" + "=" * 60)
    print("LSFM Pipeline")
    print(f"  Sample : {sample_dir}")
    print(f"  Config : {config_path}")
    print("=" * 60)


def print_step(step_num, title):
    print(f"\n{'─' * 60}")
    print(f"Step {step_num}/{PIPELINE_STEP_COUNT}: {title}")
    print("─" * 60)


def print_skip(reason):
    print(f"  SKIP: {reason}")


def print_note(message):
    print(f"  NOTE: {message}")


def run_command(cmd, desc, *, show_command=True):
    """Run a shell command and print output."""
    print(f"\n  >> {desc}")
    if show_command:
        print(f"     Command: {cmd}")

    try:
        subprocess.run(cmd, shell=True, check=True)
    except subprocess.CalledProcessError as exc:
        print(f"  ERROR: command failed ({exc})")
        sys.exit(1)


def load_config(config_path):
    """Load JSON config."""
    with open(config_path, "r", encoding="utf-8-sig") as f:
        return json.load(f)


def resolve_project_path(path_value):
    """Resolve config paths: ${YIFU_DATA_DIR}/..., absolute, or repo-relative."""
    from pipeline_modules.utils.data_paths import expand_config_path

    return expand_config_path(path_value, project_root_override=project_root)


def directory_has_files(path):
    return path.exists() and any(path.iterdir())


def format_csv(values):
    return ",".join(map(str, values))


def remove_path(path):
    path = Path(path)
    if not path.exists():
        return
    if path.is_dir():
        for child in path.iterdir():
            remove_path(child)
        path.rmdir()
    else:
        path.unlink()


def newest_mtime(path):
    path = Path(path)
    if not path.exists():
        return None
    if path.is_file():
        return path.stat().st_mtime
    newest = path.stat().st_mtime
    for child in path.rglob("*"):
        try:
            newest = max(newest, child.stat().st_mtime)
        except OSError:
            continue
    return newest


def cfos_unet_checkpoint_path(model_cfg):
    checkpoint_value = str(model_cfg.get("checkpoint_path", "")).strip()
    if not checkpoint_value:
        return None
    return resolve_project_path(checkpoint_value)


def cfos_unet_outputs_are_stale(model_cfg, output_paths):
    if not model_cfg.get("rerun_if_model_updated", False):
        return False
    checkpoint_path = cfos_unet_checkpoint_path(model_cfg)
    if checkpoint_path is None:
        print("Warning: rerun_if_model_updated is enabled but checkpoint_path is empty.")
        return False
    if not checkpoint_path.exists():
        print(f"Warning: checkpoint not found for freshness check: {checkpoint_path}")
        return False
    checkpoint_mtime = checkpoint_path.stat().st_mtime
    existing_output_mtimes = [mtime for mtime in (newest_mtime(path) for path in output_paths if path) if mtime is not None]
    if not existing_output_mtimes:
        return True
    return checkpoint_mtime > min(existing_output_mtimes)


def spotiflow_model_mtime(model_cfg):
    model_value = str(model_cfg.get("model_dir", "model/spotiflow")).strip()
    if not model_value:
        return None
    model_path = Path(model_value)
    if not model_path.is_absolute():
        model_path = project_root / model_path
    return newest_mtime(model_path)


def spotiflow_outputs_are_stale(model_cfg, output_paths):
    if not model_cfg.get("rerun_if_model_updated", True):
        return False
    model_mtime = spotiflow_model_mtime(model_cfg)
    if model_mtime is None:
        print("Warning: Spotiflow model folder not found for freshness check.")
        return False
    existing_output_mtimes = [mtime for mtime in (newest_mtime(path) for path in output_paths if path) if mtime is not None]
    if not existing_output_mtimes:
        return True
    return model_mtime > min(existing_output_mtimes)


def run_tiff_to_zarr(input_path, output_path, chunk_size, desc):
    chunk_str = format_csv(chunk_size)
    cmd = (
        f'"{PYTHON_EXE}" -m pipeline_modules.preprocessing.tiff_to_zarr '
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


def ensure_signal_tiff_dir(sample_dir, signal_ch, preprocessing_cfg, *, output_dir=None):
    from pipeline_modules.preprocessing.preprocessor import Preprocessor

    raw_tiff_dir = sample_dir / f"ch{signal_ch}"
    current_signal_tiff_dir = raw_tiff_dir

    preprocessor = Preprocessor(preprocessing_cfg)
    if preprocessor.steps:
        enhanced_dir = Path(output_dir) if output_dir is not None else sample_dir / f"ch{signal_ch}_preprocessed"
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
        print_note("No TIFF enhancement steps enabled; using raw signal channel.")

    return current_signal_tiff_dir


def ensure_signal_zarr(sample_dir, signal_ch, signal_tiff_dir, zarr_cfg):
    zarr_path = sample_dir / f"ch{signal_ch}.zarr"

    if zarr_path.exists():
        print_skip(f"Signal Zarr already exists: {zarr_path}")
    else:
        run_tiff_to_zarr(
            signal_tiff_dir,
            zarr_path,
            zarr_cfg["chunk_size"],
            "3.3 Convert signal TIFF to Zarr",
        )

    return zarr_path


def ensure_registration_downsample(sample_dir, reg_ch, input_res, target_res):
    reg_downsample_dir = sample_dir / f"ch{reg_ch}_downsample"
    reg_nifti_path = reg_downsample_dir / "volume.nii.gz"

    if reg_nifti_path.exists():
        print_skip(f"Registration downsample already exists: {reg_nifti_path}")
        return

    try:
        factor_str = calculate_downsample_factor_str(input_res, target_res)
        print_note(f"Downsample factors (z,y,x): {factor_str}")
    except ValueError as exc:
        print(f"Error calculating downsample factors from config: {exc}")
        sys.exit(1)

    cmd = (
        f'"{PYTHON_EXE}" -m pipeline_modules.preprocessing.downsample '
        f'--input_folder "{sample_dir / f"ch{reg_ch}"}" '
        f'--factor "{factor_str}"'
    )
    run_command(cmd, "1.1 Downsample registration channel")


def build_segmentation_command(seg_cfg, zarr_path, mask_zarr_path, probability_zarr_path=None):
    seg_method = seg_cfg["method"]

    if seg_method == "cellpose":
        cellpose_script = project_root / "pipeline_modules" / "segmentation" / "cellpose_distributed.py"
        if not cellpose_script.exists():
            print(f"Error: cellpose segmentation script not found: {cellpose_script}")
            sys.exit(1)
        cp_cfg = seg_cfg["cellpose"]
        return (
            f'"{PYTHON_EXE}" -m pipeline_modules.segmentation.cellpose_distributed '
            f'--input_zarr "{zarr_path}" '
            f'--output_zarr "{mask_zarr_path}" '
            f'--workers {cp_cfg["workers"]} '
            f'--pretrained_model {cp_cfg["model"]} '
            f'--diameter {cp_cfg["diameter"]}'
        )

    if seg_method == "threshold":
        th_cfg = seg_cfg["threshold"]
        return (
            f'"{PYTHON_EXE}" -m pipeline_modules.segmentation.intensity_threshold_segmentor '
            f'--input_zarr "{zarr_path}" '
            f'--output_zarr "{mask_zarr_path}" '
            f'--threshold {th_cfg["value"]} '
            f'--sigma {th_cfg["sigma"]}'
        )

    if seg_method == "cfos_unet":
        model_cfg = seg_cfg["cfos_unet"]
        cmd = (
            f'"{PYTHON_EXE}" -m pipeline_modules.segmentation.cfos_unet_inference '
            f'--input_zarr "{zarr_path}" '
            f'--output_zarr "{mask_zarr_path}" '
            f'--checkpoint_path "{cfos_unet_checkpoint_path(model_cfg)}" '
            f'--dataset_name "{model_cfg.get("dataset_name", "0")}" '
            f'--overlap {model_cfg.get("overlap", 0.25)} '
            f'--batch_size {model_cfg.get("batch_size", 4)} '
            f'--device "{model_cfg.get("device", "auto")}" '
            f'--foreground_class {model_cfg.get("foreground_class", 1)} '
            f'--probability_threshold {model_cfg.get("probability_threshold", 0.5)} '
            f'--output_mode "{model_cfg.get("output_mode", "binary")}" '
            f'--output_dtype "{model_cfg.get("output_dtype", "uint8")}" '
            f'--probability_dtype "{model_cfg.get("probability_dtype", "float16")}"'
        )
        if probability_zarr_path:
            cmd += f' --probability_zarr "{probability_zarr_path}"'
        if model_cfg.get("patch_size"):
            cmd += f' --patch_size "{format_csv(model_cfg["patch_size"])}"'
        if model_cfg.get("chunk_size"):
            cmd += f' --chunk_size "{format_csv(model_cfg["chunk_size"])}"'
        if model_cfg.get("normalize_percentiles"):
            cmd += f' --normalize_percentiles "{format_csv(model_cfg["normalize_percentiles"])}"'
        if model_cfg.get("skip_below_threshold") is not None:
            cmd += f' --skip_below_threshold {model_cfg.get("skip_below_threshold")}'
        if model_cfg.get("process_existing_only", False):
            cmd += ' --process_existing_only'
        return cmd

    print(f"Unknown segmentation method: {seg_method}")
    sys.exit(1)


def resolve_spotiflow_outputs(sample_dir, signal_ch, model_cfg):
    output_csv = model_cfg.get("output_csv") or f"ch{signal_ch}_spotiflow_points.csv"
    region_counts_csv = model_cfg.get("region_counts_csv") or f"ch{signal_ch}_spotiflow_region_counts.csv"
    summary_json = model_cfg.get("summary_json") or f"ch{signal_ch}_spotiflow_summary.json"

    def _resolve(path_value):
        path = Path(path_value)
        if not path.is_absolute():
            path = sample_dir / path
        return path

    return _resolve(output_csv), _resolve(region_counts_csv), _resolve(summary_json)


def build_spotiflow_command(
    model_cfg,
    zarr_path,
    output_csv,
    region_counts_csv,
    summary_json,
    label_zarr_path=None,
    density_cfg_path=None,
):
    model_dir = Path(model_cfg.get("model_dir", "model/spotiflow"))
    if not model_dir.is_absolute():
        model_dir = project_root / model_dir

    cmd = (
        f'"{PYTHON_EXE}" -m pipeline_modules.segmentation.spotiflow_inference '
        f'--input_zarr "{zarr_path}" '
        f'--output_csv "{output_csv}" '
        f'--model_dir "{model_dir}" '
        f'--region_counts_csv "{region_counts_csv}" '
        f'--summary_json "{summary_json}" '
        f'--dataset_name "{model_cfg.get("dataset_name", "0")}" '
        f'--which "{model_cfg.get("which", "best")}" '
        f'--min_distance {model_cfg.get("min_distance", 1)} '
        f'--tile_overlap {model_cfg.get("tile_overlap", 16)} '
        f'--device "{model_cfg.get("device", "auto")}" '
        f'--peak_mode "{model_cfg.get("peak_mode", "fast")}"'
    )
    if model_cfg.get("prob_thresh") is not None:
        cmd += f' --prob_thresh {model_cfg.get("prob_thresh")}'
    if model_cfg.get("tile_size"):
        cmd += f' --tile_size "{format_csv(model_cfg["tile_size"])}"'
    if model_cfg.get("skip_below_threshold") is not None:
        cmd += f' --skip_below_threshold {model_cfg.get("skip_below_threshold")}'
    if model_cfg.get("normalizer") is None:
        cmd += ' --normalizer none'
    else:
        cmd += f' --normalizer "{model_cfg.get("normalizer", "auto")}"'
    if model_cfg.get("subpix") is True:
        cmd += ' --subpix true'
    elif model_cfg.get("subpix") is False:
        cmd += ' --subpix false'
    if model_cfg.get("use_tuned_tile_overlap", False):
        cmd += ' --use_tuned_tile_overlap'
    if model_cfg.get("checkpoint_tiles") is not None:
        cmd += f' --checkpoint_tiles {model_cfg.get("checkpoint_tiles")}'
    if model_cfg.get("qc_top_n"):
        cmd += f' --qc_top_n {model_cfg.get("qc_top_n")}'
        if model_cfg.get("qc_tile_csv"):
            qc_tile_csv = Path(model_cfg.get("qc_tile_csv"))
            if not qc_tile_csv.is_absolute():
                qc_tile_csv = zarr_path.parent / qc_tile_csv
            cmd += f' --qc_tile_csv "{qc_tile_csv}"'
        if model_cfg.get("qc_top_csv"):
            qc_top_csv = Path(model_cfg.get("qc_top_csv"))
            if not qc_top_csv.is_absolute():
                qc_top_csv = zarr_path.parent / qc_top_csv
            cmd += f' --qc_top_csv "{qc_top_csv}"'
        if model_cfg.get("qc_preview_dir"):
            qc_preview_dir = Path(model_cfg.get("qc_preview_dir"))
            if not qc_preview_dir.is_absolute():
                qc_preview_dir = zarr_path.parent / qc_preview_dir
            cmd += f' --qc_preview_dir "{qc_preview_dir}"'
        cmd += f' --qc_preview_mode "{model_cfg.get("qc_preview_mode", "mip")}"'
    if label_zarr_path:
        cmd += f' --label_zarr "{label_zarr_path}"'
    configured_label_zarr = model_cfg.get("label_zarr")
    if configured_label_zarr and not label_zarr_path:
        label_path = Path(configured_label_zarr)
        if not label_path.is_absolute():
            label_path = zarr_path.parent / label_path
        cmd += f' --label_zarr "{label_path}"'
    if density_cfg_path:
        cmd += f' --cfg "{density_cfg_path}"'
    elif model_cfg.get("cfg"):
        cfg_path = Path(model_cfg.get("cfg"))
        if not cfg_path.is_absolute():
            cfg_path = project_root / cfg_path
        cmd += f' --cfg "{cfg_path}"'
    return cmd


def ensure_spotiflow_outputs(
    sample_dir,
    signal_ch,
    zarr_path,
    model_cfg,
    label_zarr_path=None,
    density_cfg_path=None,
):
    output_csv, region_counts_csv, summary_json = resolve_spotiflow_outputs(sample_dir, signal_ch, model_cfg)
    outputs = [output_csv, summary_json]
    if label_zarr_path or model_cfg.get("label_zarr"):
        outputs.append(region_counts_csv)

    force_rerun = spotiflow_outputs_are_stale(model_cfg, outputs)
    outputs_ready = all(Path(path).exists() for path in outputs)
    if force_rerun:
        print_note("Spotiflow model is newer than existing outputs; rerunning detection.")

    if outputs_ready and not force_rerun:
        print_skip(f"Spotiflow outputs already exist: {output_csv}")
        return output_csv, region_counts_csv, summary_json

    cmd = build_spotiflow_command(
        model_cfg,
        zarr_path,
        output_csv,
        region_counts_csv,
        summary_json,
        label_zarr_path=label_zarr_path,
        density_cfg_path=density_cfg_path,
    )
    run_command(cmd, "4.1 Spotiflow whole-brain signal detection")
    return output_csv, region_counts_csv, summary_json


def ensure_segmentation_outputs(sample_dir, signal_ch, zarr_path, seg_cfg):
    mask_zarr_path = sample_dir / f"ch{signal_ch}_mask.zarr"
    mask_tiff_dir = sample_dir / f"ch{signal_ch}_mask"
    export_mask_tiff = bool(seg_cfg.get("export_mask_tiff", False))
    probability_zarr_path = None
    force_rerun = False
    if seg_cfg["method"] == "cfos_unet":
        model_cfg = seg_cfg["cfos_unet"]
        configured_probability_zarr = model_cfg.get("probability_zarr", "")
        if configured_probability_zarr:
            probability_zarr_path = Path(configured_probability_zarr)
            if not probability_zarr_path.is_absolute():
                probability_zarr_path = sample_dir / probability_zarr_path
        elif model_cfg.get("save_probability", False):
            probability_zarr_path = sample_dir / f"ch{signal_ch}_prob.zarr"
        freshness_outputs = [mask_zarr_path, probability_zarr_path]
        if export_mask_tiff:
            freshness_outputs.append(mask_tiff_dir)
        force_rerun = cfos_unet_outputs_are_stale(model_cfg, freshness_outputs)

    probability_ready = probability_zarr_path is None or probability_zarr_path.exists()
    if force_rerun:
        print_note("cFos U-Net checkpoint is newer than existing outputs; rerunning segmentation.")

    if not force_rerun and mask_zarr_path.exists() and probability_ready:
        print_skip(f"Mask Zarr already exists: {mask_zarr_path}")
        if not export_mask_tiff:
            return mask_zarr_path, mask_tiff_dir

    if export_mask_tiff and not force_rerun and directory_has_files(mask_tiff_dir) and probability_ready:
        print_skip(f"Mask TIFF folder already exists: {mask_tiff_dir}")
        return mask_zarr_path, mask_tiff_dir

    segmentation_ran = False
    if force_rerun or not mask_zarr_path.exists() or not probability_ready:
        seg_cmd = build_segmentation_command(seg_cfg, zarr_path, mask_zarr_path, probability_zarr_path)
        run_command(seg_cmd, f'4.1 Segmentation ({seg_cfg["method"]})')
        segmentation_ran = True

    if force_rerun and segmentation_ran and directory_has_files(mask_tiff_dir):
        remove_path(mask_tiff_dir)

    if export_mask_tiff and not directory_has_files(mask_tiff_dir):
        export_cmd = (
            f'"{PYTHON_EXE}" -c "from pipeline_modules.segmentation '
            f'import export_zarr_to_tiff; '
            f"export_zarr_to_tiff(r'{mask_zarr_path}', r'{mask_tiff_dir}')\""
        )
        run_command(export_cmd, "4.2 Export mask Zarr to TIFF")
    elif not export_mask_tiff:
        print_skip("Mask TIFF export disabled (segmentation.export_mask_tiff=false).")

    return mask_zarr_path, mask_tiff_dir


def ensure_registration_outputs(sample_dir, signal_ch, reg_ch, reg_cfg, zarr_cfg, config_path):
    """Run ANTs registration and ensure requested label outputs exist.

    Returns (warped_label_tiff_dir, warped_label_zarr_path).
    """
    warped_label_dir = sample_dir / "upsampled_atlas_label"
    warped_label_zarr_path = sample_dir / "upsampled_atlas_label.zarr"
    save_label_tiff = bool(reg_cfg.get("save_upsampled_label", True))
    save_label_zarr = bool(reg_cfg.get("save_upsampled_label_zarr", True))

    if not save_label_tiff and not save_label_zarr:
        print_skip("Atlas label outputs disabled (save_upsampled_label and save_upsampled_label_zarr are both false).")
        return None, None

    requested_outputs_exist = (
        (not save_label_tiff or directory_has_files(warped_label_dir))
        and (not save_label_zarr or warped_label_zarr_path.exists())
    )
    if requested_outputs_exist:
        print_skip("Registration label outputs already exist.")
    else:
        atlas_path = resolve_project_path(reg_cfg["atlas_path"])
        annotation_path = resolve_project_path(reg_cfg["annotation_path"])
        transform_type = reg_cfg.get("transform_type", "SyN")
        cmd = (
            f'"{PYTHON_EXE}" -m pipeline_modules.registration.ANTs_registration '
            f'--sample_dir "{sample_dir}" '
            f'--signal_channel {signal_ch} '
            f'--register_channel {reg_ch} '
            f'--atlas_image "{atlas_path}" '
            f'--atlas_label "{annotation_path}" '
            f'--mode {MAIN_PIPELINE_REGISTRATION_MODE} '
            f'--registration_type {transform_type} '
            f'--save_registered_image '
            f'--save_transforms '
            f'--config "{config_path}"'
        )
        run_command(cmd, "2.1 ANTs registration (atlas → image)")

    # Ensure label Zarr for downstream modules when requested. Older registration
    # runs may have produced only the TIFF stack.
    if save_label_zarr and not warped_label_zarr_path.exists():
        if not directory_has_files(warped_label_dir):
            print(f"Error: Label Zarr requested, but TIFF stack is unavailable at {warped_label_dir}.")
            sys.exit(1)
        run_tiff_to_zarr(
            warped_label_dir,
            warped_label_zarr_path,
            zarr_cfg["chunk_size"],
            "2.2 Convert atlas label TIFF to Zarr",
        )
    elif save_label_zarr:
        print_skip(f"Atlas label Zarr already exists: {warped_label_zarr_path}")

    save_hemisphere_zarr = bool(reg_cfg.get("save_upsampled_label_hemisphere_zarr", False))
    hemisphere_zarr_path = sample_dir / "atlas_label_hemisphere.zarr"
    if save_hemisphere_zarr and not hemisphere_zarr_path.exists():
        hemisphere_input = warped_label_zarr_path if warped_label_zarr_path.exists() else warped_label_dir
        if not Path(hemisphere_input).exists():
            print(f"Error: Hemisphere Zarr requested, but atlas label input is unavailable at {hemisphere_input}.")
            sys.exit(1)
        chunk_str = format_csv(zarr_cfg["chunk_size"])
        cmd = (
            f'"{PYTHON_EXE}" -m pipeline_modules.registration.atlas_label_to_hemisphere '
            f'--input "{hemisphere_input}" '
            f'--output "{hemisphere_zarr_path}" '
            f'--chunk_size "{chunk_str}" '
            f'--dataset_name "0"'
        )
        run_command(cmd, "2.3 Convert atlas label to hemisphere Zarr")
    elif save_hemisphere_zarr:
        print_skip(f"Hemisphere Zarr already exists: {hemisphere_zarr_path}")

    if not save_label_tiff and directory_has_files(warped_label_dir):
        remove_path(warped_label_dir)

    return (
        warped_label_dir if save_label_tiff else None,
        warped_label_zarr_path if save_label_zarr else None,
    )


def ensure_mask_zarr(mask_tiff_dir, mask_zarr_path, zarr_cfg):
    if mask_zarr_path.exists():
        return

    if not directory_has_files(mask_tiff_dir):
        print(f"Error: Mask Zarr not found at {mask_zarr_path}, and mask TIFF folder is unavailable.")
        sys.exit(1)

    run_tiff_to_zarr(mask_tiff_dir, mask_zarr_path, zarr_cfg["chunk_size"], "5.1 Convert mask TIFF to Zarr")


def ensure_hemisphere_label_zarr(warped_label_zarr_path, hemisphere_zarr_path, zarr_cfg):
    if hemisphere_zarr_path.exists():
        print(f"Hemisphere label Zarr exists, skipping conversion: {hemisphere_zarr_path}")
        return

    if not warped_label_zarr_path or not warped_label_zarr_path.exists():
        print(f"Error: Cannot create hemisphere Zarr because label Zarr is unavailable: {warped_label_zarr_path}")
        sys.exit(1)

    cmd = (
        f'"{PYTHON_EXE}" -m pipeline_modules.registration.atlas_label_to_hemisphere '
        f'--input "{warped_label_zarr_path}" '
        f'--output "{hemisphere_zarr_path}" '
        f'--chunk_size "{format_csv(zarr_cfg["chunk_size"])}"'
    )
    run_command(cmd, "Step 2.3: Convert Atlas Label Zarr to Hemisphere Zarr")


def run_density_analysis(
    sample_dir,
    signal_ch,
    zarr_path,
    mask_zarr_path,
    warped_label_zarr_path,
    hemisphere_zarr_path,
    zarr_cfg,
    density_cfg_path,
    resolution_xyz,
):
    if not warped_label_zarr_path.exists():
        print(f"Error: Label Zarr not found at {warped_label_zarr_path}.")
        sys.exit(1)
    if hemisphere_zarr_path and not hemisphere_zarr_path.exists():
        print(f"Error: Hemisphere Zarr not found at {hemisphere_zarr_path}.")
        sys.exit(1)

    mask_tiff_dir = sample_dir / f"ch{signal_ch}_mask"
    ensure_mask_zarr(mask_tiff_dir, mask_zarr_path, zarr_cfg)

    output_excel = sample_dir / "results" / f"{sample_dir.name}_ch{signal_ch}_brain_distribution_stats.xlsx"
    output_excel.parent.mkdir(parents=True, exist_ok=True)
    resolution_xyz_str = format_csv(resolution_xyz)

    cmd = (
        f'"{PYTHON_EXE}" -m pipeline_modules.registration.region_signal_analysis_zarr_graph '
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
    if hemisphere_zarr_path:
        cmd += f' --hemisphere_zarr "{hemisphere_zarr_path}"'
    run_command(cmd, "5.2 Region density analysis")


def main():
    parser = argparse.ArgumentParser(
        description="LSFM main pipeline: preprocessing -> registration -> edge removal -> segmentation -> analysis"
    )
    parser.add_argument("--config", default="config.json", help="Path to config.json")
    parser.add_argument("--sample_dir", help="Root directory of the sample")
    parser.add_argument("--test", action="store_true", help="Run in test mode (quick checks only)")
    parser.add_argument("--skip_registration", action="store_true", help="Skip ANTs registration")
    args = parser.parse_args()

    if args.test:
        print("Running Pipeline in TEST Mode...")
        run_command(
            f'"{PYTHON_EXE}" -c "from pipeline_modules.segmentation import load_capability_manifest; '
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

    cfg = load_config(config_path)
    sample_dir = Path(args.sample_dir)
    print_pipeline_banner(sample_dir, config_path)

    signal_ch = cfg["input"]["channels"]["signal"]
    reg_ch = cfg["input"]["channels"]["registration"]
    preprocessing_cfg = cfg["preprocessing"]
    zarr_cfg = preprocessing_cfg["zarr"]
    seg_cfg = cfg["segmentation"]
    reg_cfg = cfg["registration"]
    analysis_cfg = cfg["analysis"]
    density_cfg_path = resolve_density_cfg_path(analysis_cfg)
    use_hemisphere_label = bool(analysis_cfg.get("use_hemisphere_label", False))

    print_step(1, "Registration channel downsample")
    ensure_registration_downsample(
        sample_dir,
        reg_ch,
        cfg["input"]["resolution_xyz"],
        preprocessing_cfg["downsample"]["target_resolution_xyz"],
    )

    print_step(2, "Atlas registration and label outputs")
    warped_label_dir = warped_label_zarr = None
    hemisphere_label_zarr = None
    if use_hemisphere_label and not bool(reg_cfg.get("save_upsampled_label_hemisphere_zarr", False)):
        print(
            "Error: analysis.use_hemisphere_label=true requires "
            "registration.save_upsampled_label_hemisphere_zarr=true."
        )
        sys.exit(1)
    if not args.skip_registration:
        warped_label_dir, warped_label_zarr = ensure_registration_outputs(
            sample_dir, signal_ch, reg_ch, reg_cfg, zarr_cfg, config_path,
        )
        if use_hemisphere_label:
            hemisphere_label_zarr = sample_dir / "atlas_label_hemisphere.zarr"
    else:
        print_skip("Registration skipped (--skip_registration).")
        zarr_path_candidate = sample_dir / "upsampled_atlas_label.zarr"
        dir_path_candidate = sample_dir / "upsampled_atlas_label"
        if zarr_path_candidate.exists():
            warped_label_zarr = zarr_path_candidate
        hemisphere_candidate = sample_dir / "atlas_label_hemisphere.zarr"
        if use_hemisphere_label and hemisphere_candidate.exists():
            hemisphere_label_zarr = hemisphere_candidate
        if directory_has_files(dir_path_candidate):
            warped_label_dir = dir_path_candidate

    if use_hemisphere_label:
        hemisphere_label_zarr = sample_dir / "atlas_label_hemisphere.zarr"
        if warped_label_zarr:
            ensure_hemisphere_label_zarr(warped_label_zarr, hemisphere_label_zarr, zarr_cfg)

    print_step(3, "Signal preprocessing and Zarr conversion")
    esr_cfg = preprocessing_cfg.get("edge_signal_removal", {})
    edge_removal_enabled = esr_cfg.get("apply", False)
    if edge_removal_enabled and warped_label_dir:
        raw_tiff_dir = sample_dir / f"ch{signal_ch}"
        contour_tiff_dir = sample_dir / "ch0"
        prior_dir = sample_dir / "ch0_warped_image"
        clean_tiff_dir = sample_dir / f"ch{signal_ch}_preprocessed"
        clean_zarr_path = sample_dir / f"ch{signal_ch}.zarr"
        cmd = (
            f'"{PYTHON_EXE}" -m pipeline_modules.preprocessing.edge_signal_removal '
            f'--input_dir "{raw_tiff_dir}" '
            f'--contour_dir "{contour_tiff_dir}" '
            f'--output_dir "{clean_tiff_dir}" '
            f'--inward_px {esr_cfg.get("inward_px", 50)} '
            f'--outward_px {esr_cfg.get("outward_px", 0)} '
            f'--prior_dir "{prior_dir}" '
            f'--prior_curve_scale {esr_cfg.get("prior_curve_scale", 1.15)} '
            f'--prior_curve_smooth_sigma {esr_cfg.get("prior_curve_smooth_sigma", 2.0)} '
            f'--brightness_pct {esr_cfg.get("brightness_pct", 90.0)} '
            f'--contour_min_object_area_px {esr_cfg.get("contour_min_object_area_px", 50)} '
            f'--contour_morph_radius_px {esr_cfg.get("contour_morph_radius_px", 2)} '
            f'--contour_dilation_px {esr_cfg.get("contour_dilation_px", 2)} '
            f'--min_area_px {esr_cfg.get("min_area_px", 50)} '
            f'--suppression_weight {esr_cfg.get("suppression_weight", 1.0)} '
            f'--max_workers {esr_cfg.get("max_workers", 8)} '
            "--no_resume"
        )
        run_command(cmd, "3.2 Edge signal removal")
        signal_tiff_dir = clean_tiff_dir
        if clean_zarr_path.exists():
            remove_path(clean_zarr_path)
        zarr_path = ensure_signal_zarr(sample_dir, signal_ch, signal_tiff_dir, zarr_cfg)
    else:
        if edge_removal_enabled:
            print_note("Edge signal removal enabled, but atlas label TIFF is unavailable; using standard preprocessing.")
        print_note("3.1 TIFF preprocessing/enhancement")
        signal_tiff_dir = ensure_signal_tiff_dir(sample_dir, signal_ch, preprocessing_cfg)
        zarr_path = ensure_signal_zarr(sample_dir, signal_ch, signal_tiff_dir, zarr_cfg)

    print_step(4, "Segmentation")
    if seg_cfg["method"] == "spotiflow":
        spotiflow_label_zarr = warped_label_zarr
        if not spotiflow_label_zarr and seg_cfg.get("spotiflow", {}).get("label_zarr"):
            spotiflow_label_zarr = Path(seg_cfg["spotiflow"]["label_zarr"])
            if not spotiflow_label_zarr.is_absolute():
                spotiflow_label_zarr = sample_dir / spotiflow_label_zarr

        points_csv, region_counts_csv, summary_json = ensure_spotiflow_outputs(
            sample_dir,
            signal_ch,
            zarr_path,
            seg_cfg["spotiflow"],
            label_zarr_path=spotiflow_label_zarr if spotiflow_label_zarr and spotiflow_label_zarr.exists() else None,
            density_cfg_path=density_cfg_path,
        )
        print_step(5, "Spotiflow signal count summary")
        print_note(f"Whole-brain points CSV: {points_csv}")
        print_note(f"Summary JSON: {summary_json}")
        if spotiflow_label_zarr and spotiflow_label_zarr.exists():
            print_note(f"Per-region signal counts CSV: {region_counts_csv}")
        else:
            print_skip("Per-region counts skipped (atlas label Zarr unavailable).")
    else:
        mask_zarr_path, _ = ensure_segmentation_outputs(sample_dir, signal_ch, zarr_path, seg_cfg)

        if warped_label_zarr:
            print_step(5, "Region density analysis")
            run_density_analysis(
                sample_dir,
                signal_ch,
                zarr_path,
                mask_zarr_path,
                warped_label_zarr,
                hemisphere_label_zarr,
                zarr_cfg,
                density_cfg_path,
                cfg["input"]["resolution_xyz"],
            )
        else:
            print_step(5, "Region density analysis")
            print_skip("Density analysis skipped (atlas label Zarr unavailable).")

    print("\n" + "=" * 60)
    print("Pipeline completed successfully.")
    print("=" * 60)


if __name__ == "__main__":
    main()
