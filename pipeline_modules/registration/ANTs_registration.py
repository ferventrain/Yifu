import os
import json
import logging
import shutil
import time
from pathlib import Path
from typing import Optional, Tuple, Union, Dict

import numpy as np
import ants
import tifffile

try:
    from pipeline_modules.registration.label_codec import (
        build_label_id_codec,
        decode_label_codes,
        ensure_label_storage_dtype,
        load_label_array_preserving_ids,
    )
    from pipeline_modules.preprocessing.tiff_to_zarr import convert_tiff_to_zarr
    from pipeline_modules.utils.tiff_stack_io import (
        iter_batch_ranges,
        normalize_tiff_compression,
        resolve_slice_batch,
        resolve_stack_workers,
        run_bounded_batches,
    )
    from pipeline_modules.utils.errors import ErrorCode, PipelineError
    from pipeline_modules.utils.run_manifest import write_run_manifest
except ImportError:
    from .label_codec import (  # type: ignore[no-redef]
        build_label_id_codec,
        decode_label_codes,
        ensure_label_storage_dtype,
        load_label_array_preserving_ids,
    )

    convert_tiff_to_zarr = None  # type: ignore[assignment]
    PipelineError = None  # type: ignore[assignment,misc]
    ErrorCode = None  # type: ignore[assignment]
    write_run_manifest = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


def _nifti_spacing_from_affine(affine: np.ndarray):
    return tuple(float(abs(affine[i, i])) for i in range(3))


def _load_nifti_as_ants(path):
    """Load NIfTI via nibabel and convert to ANTs image.

    ANTs/ITK image_read can crash on Windows when the path contains non-ASCII
    characters (e.g. Chinese folder names on NAS paths).
    """
    try:
        import nibabel as nib
    except ModuleNotFoundError as exc:
        raise RuntimeError("nibabel is required to load NIfTI registration volumes") from exc

    nii = nib.load(str(path))
    data = np.asanyarray(nii.dataobj)
    if not np.issubdtype(data.dtype, np.floating):
        data = data.astype(np.float32)
    return ants.from_numpy(data, spacing=_nifti_spacing_from_affine(nii.affine))


def _ants_image_read(path):
    path_str = str(path)
    if path_str.lower().endswith((".nii", ".nii.gz")):
        return _load_nifti_as_ants(path)
    return ants.image_read(path_str)


def _write_tiff_volume_batch(job: dict[str, object]) -> int:
    arr = job["arr"]
    start = int(job["start"])
    end = int(job["end"])
    output_dtype = job["dtype"]
    compression = normalize_tiff_compression(job.get("compression"))  # type: ignore[arg-type]
    for offset, z_index in enumerate(range(start, end)):
        tifffile.imwrite(
            str(job["paths"][offset]),
            arr[z_index, :, :].astype(output_dtype, copy=False),
            compression=compression,
        )
    return end - start


def _upsample_label_batch(job: dict[str, object]) -> int:
    import cv2

    source_volume = job["source_volume"]
    z_indices = job["z_indices"]
    target_start = int(job["target_start"])
    target_xy = job["target_xy"]
    output_dtype = job["dtype"]
    output_dir = Path(job["output_dir"])
    compression = normalize_tiff_compression(job.get("compression"))  # type: ignore[arg-type]
    written = 0
    for local_offset, source_z in enumerate(z_indices):
        target_i = target_start + local_offset
        resized_slice = cv2.resize(
            source_volume[int(source_z)],
            target_xy,
            interpolation=cv2.INTER_NEAREST,
        )
        tifffile.imwrite(
            str(output_dir / f"label_{target_i:06d}.tiff"),
            np.rint(resized_slice).astype(output_dtype, copy=False),
            compression=compression,
        )
        written += 1
    return written


class BidirectionalRegistration:
    """双向配准类：支持atlas到image和image到atlas的配准"""
    
    UPSAMPLING_METHODS = {
        'nearest': 0,
        'linear': 1,
        'cubic': 3,
        'quintic': 5
    }
    
    def __init__(self, 
                 sample_dir: str,
                 signal_channel: str,
                 atlas_image_path: str, 
                 atlas_label_path: str,
                 register_channel: str,
                 original_shape: Optional[Tuple[int, int, int]] = None,
                 density_cfg_path: Optional[str] = None,
                 config_path: Optional[str] = None):
        
        self.sample_dir = Path(sample_dir)
        self.signal_channel = signal_channel
        self.register_channel = register_channel
        
        # Try to infer original shape if not provided
        if original_shape is None:
            self.original_shape = self._infer_original_shape()
        else:
            self.original_shape = original_shape
            
        if self.original_shape is None:
            logger.warning("Could not determine original shape. Upsampling will be skipped/fail.")
        else:
            logger.info("Original shape determined: %s", self.original_shape)

        # Load Atlas
        self.atlas_image = ants.image_read(atlas_image_path)
        self.atlas_label_id_lut: np.ndarray | None = None
        self.atlas_label = self._load_encoded_atlas_label(atlas_label_path)
        
        # Force direction matrix to identity to avoid flipping/reflection
        # [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        identity_direction = np.eye(3)
        self.atlas_image.set_direction(identity_direction)
        self.atlas_label.set_direction(identity_direction)
        logger.info("Forced atlas direction matrix to identity (no flipping)")
        
        # Load Config
        if config_path and os.path.exists(config_path):
             with open(config_path, 'r', encoding='utf-8-sig') as f:
                full_config = json.load(f)
                # Parse resolution from main config structure
                # Input resolution (Source)
                self.source_resolution = full_config['input']['resolution_xyz']
                # Target resolution (Downsample target)
                self.atlas_resolution = full_config['preprocessing']['downsample']['target_resolution_xyz']
        else:
            # Fallback to old behavior if config_path not provided (though we should enforce it)
            logger.warning("Config path not provided or not found. Falling back to resolution.json (Deprecated).")
            current_dir = Path(__file__).parent
            resolution_config_path = current_dir / 'resolution.json'
            if resolution_config_path.exists():
                with open(resolution_config_path, 'r', encoding='utf-8') as f:
                    self.config = json.load(f)
                self.source_resolution = self.config['source_resolution']
                self.atlas_resolution = self.config['target_resolution']
            else:
                # Absolute fallback default
                logger.error("No resolution config found. Using defaults.")
                self.source_resolution = [1.8, 1.8, 2.0]
                self.atlas_resolution = [25.0, 25.0, 25.0]

        logger.info("Source Resolution: %s", self.source_resolution)
        logger.info("Target Resolution: %s", self.atlas_resolution)
        registration_config = getattr(self, "config", {})
        if config_path and os.path.exists(config_path):
            registration_config = full_config.get("registration", {})
        self.registration_config = registration_config

        # Flip atlas if configured
        flip_atlas = registration_config.get("flip_atlas", [False, False, False])
        if any(flip_atlas):
            flip_axes = tuple(i for i, flip in enumerate(flip_atlas) if flip)
            logger.info("Flipping atlas along axis indices %s (x=%s, y=%s, z=%s)",
                        flip_axes, flip_atlas[0], flip_atlas[1], flip_atlas[2])
            self.atlas_image = ants.from_numpy(
                np.flip(self.atlas_image.numpy(), axis=flip_axes).copy(),
                spacing=self.atlas_image.spacing,
                origin=self.atlas_image.origin,
                direction=self.atlas_image.direction,
            )
            self.atlas_label = ants.from_numpy(
                np.flip(self.atlas_label.numpy(), axis=flip_axes).copy(),
                spacing=self.atlas_label.spacing,
                origin=self.atlas_label.origin,
                direction=self.atlas_label.direction,
            )

        self.save_upsampled_label = bool(registration_config.get("save_upsampled_label", True))
        self.save_upsampled_label_zarr = bool(registration_config.get("save_upsampled_label_zarr", True))
        self.zarr_chunk_size = tuple(
            full_config.get("preprocessing", {})
            .get("zarr", {})
            .get("chunk_size", (128, 256, 256))
        ) if config_path and os.path.exists(config_path) else (128, 256, 256)
        
        # Load Target Image (used for image2atlas warping)
        # Simplified: We don't load signal channel image by default to reduce dependencies
        self.target_image = None
        
        # Load Register Image (downsampled sample)
        reg_img_path = self.sample_dir / f"ch{self.register_channel}_downsample/volume.nii.gz"
        if not reg_img_path.exists():
            raise FileNotFoundError(f"Registration image not found at {reg_img_path}. Please run downsampling first.")
        
        self.register_image = _ants_image_read(reg_img_path)
        
        # Force direction matrix to identity to avoid flipping/reflection
        identity_direction = np.eye(3)
        self.register_image.set_direction(identity_direction)
        logger.info("Forced register image direction matrix to identity (no flipping)")
        
        # Density Config
        self.density_cfg_path = density_cfg_path or os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Region_Csv_Rev1_updated.CSV')

    def _load_encoded_atlas_label(self, atlas_label_path: str) -> ants.ANTsImage:
        label_array = load_label_array_preserving_ids(atlas_label_path)
        encoded, label_id_lut = build_label_id_codec(label_array)
        self.atlas_label_id_lut = label_id_lut
        logger.info(
            "Encoded atlas label ids for ANTs warp: %d original ids -> codes 0..%d",
            len(label_id_lut),
            len(label_id_lut) - 1,
        )
        return ants.from_numpy(
            encoded,
            spacing=self.atlas_image.spacing,
            origin=self.atlas_image.origin,
            direction=self.atlas_image.direction,
        )

    def _infer_original_shape(self) -> Optional[Tuple[int, int, int]]:
        """Try to infer original shape from raw TIFF files"""
        logger.info("Attempting to infer original shape from raw data...")
        
        # Try target channel folder first, then register channel folder
        possible_folders = [
            self.sample_dir / f"ch{self.signal_channel}",
            self.sample_dir / f"ch{self.register_channel}"
        ]
        
        for folder in possible_folders:
            if folder.exists() and folder.is_dir():
                tiff_files = sorted(list(folder.glob("*.tif*")))
                if tiff_files:
                    try:
                        first_img = tifffile.imread(str(tiff_files[0]))
                        # Shape is (Z, Y, X)
                        shape = (len(tiff_files),) + first_img.shape
                        logger.info("Found raw data in %s. Inferred shape: %s", folder, shape)
                        return shape
                    except Exception as e:
                        logger.error("Error reading %s: %s", folder, e)
        
        return None
    
    def _load_tiff_stack(self, folder_path: Path) -> np.ndarray:
        """加载TIFF栈"""
        tiff_files = sorted(folder_path.glob('*.tif*'))
        stack = [tifffile.imread(f) for f in tiff_files]
        return np.stack(stack, axis=0)

    def _save_volume_as_tiff(self, data: Union[ants.ANTsImage, np.ndarray], output_dir: Path, prefix: str = "image"):
        """通用保存 TIFF 栈方法"""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if isinstance(data, ants.ANTsImage):
            arr = data.numpy()
            # ANTs: (X, Y, Z) -> TIFF: (Z, Y, X)
            arr = np.transpose(arr, (2, 1, 0))
        else:
            arr = data
            
        # Ensure 3D (Z, Y, X) - each slice along Z is (Y, X)
        if arr.ndim != 3:
            raise ValueError(f"Expected 3D array, got shape {arr.shape}")

        output_dtype = ensure_label_storage_dtype(arr) if prefix in {"label", "mask"} else np.uint16
        logger.info("Saving dtype for %s: %s", prefix, output_dtype)
            
        logger.info("Saving %s TIFFs to %s (shape: %s)...", prefix, output_dir, arr.shape)
        read_batch = resolve_slice_batch(None, default=16)
        worker_count = resolve_stack_workers(0)
        jobs: list[dict[str, object]] = []
        for start, end in iter_batch_ranges(int(arr.shape[0]), read_batch):
            paths = [output_dir / f"{prefix}_{z_index:04d}.tiff" for z_index in range(start, end)]
            jobs.append(
                {
                    "arr": arr,
                    "start": start,
                    "end": end,
                    "paths": paths,
                    "dtype": output_dtype,
                    "compression": "lzw",
                }
            )
        run_bounded_batches(
            jobs,
            _write_tiff_volume_batch,
            worker_count=worker_count,
            progress_total=int(arr.shape[0]),
            desc=f"Save {prefix} TIFFs",
            unit="slice",
        )

    def _perform_registration(self, fixed: ants.ANTsImage, moving: ants.ANTsImage, 
                            reg_type: str, **kwargs) -> Dict:
        """通用配准核心逻辑"""
        logger.info("Performing %s registration...", reg_type)
        logger.info("--- METADATA VERIFICATION ---")
        logger.info("Fixed Image  | Shape: %s, Spacing: %s, Origin: %s, Direction: %s", fixed.shape, fixed.spacing, fixed.origin, fixed.direction)
        logger.info("Moving Image | Shape: %s, Spacing: %s, Origin: %s, Direction: %s", moving.shape, moving.spacing, moving.origin, moving.direction)
        logger.info("Constraining: reflection disabled (orientation is manually aligned)")
        
        return ants.registration(
            fixed=fixed,
            moving=moving,
            type_of_transform=reg_type,
            grad_step=0.1,
            aff_random_sampling_rate=0.5,
            aff_do_reflection=False,
            **kwargs
        )

    def register(self, mode: str = 'atlas2image', registration_type: str = 'SyN', **kwargs) -> Dict:
        """执行配准"""
        if mode not in ['atlas2image', 'image2atlas']:
            raise ValueError(f"Invalid mode: {mode}")

        # Histogram matching (Sample matches Atlas)
        logger.info("Performing histogram matching (Sample -> Atlas)...")
        self.register_image = ants.histogram_match_image(self.register_image, self.atlas_image)

        if mode == 'atlas2image':
            logger.info("Mode: Atlas -> Image")
            reg_result = self._perform_registration(
                fixed=self.register_image, 
                moving=self.atlas_image, 
                reg_type=registration_type, 
                **kwargs
            )
            
            # Apply transform to Atlas Label
            warped_label = ants.apply_transforms(
                fixed=self.register_image,
                moving=self.atlas_label,
                transformlist=reg_result['fwdtransforms'],
                interpolator='nearestNeighbor'
            )
            
            return {
                'warped_image': reg_result['warpedmovout'],
                'warped_label': warped_label,
                'transforms': reg_result,
                'mode': mode
            }
            
        else: # image2atlas
            logger.info("Mode: Image -> Atlas")
            reg_result = self._perform_registration(
                fixed=self.atlas_image, 
                moving=self.register_image, 
                reg_type=registration_type, 
                **kwargs
            )
            
            # Simplified: Target image warping logic moved to analysis/heatmap program
            logger.info("Note: Target image warping logic has been moved to the analysis program.")
            
            return {
                'warped_image': reg_result['warpedmovout'],
                'warped_label': None,
                'atlas_label': self.atlas_label,
                'transforms': reg_result,
                'mode': mode
            }

    def upsample_label_chunked(self, label_image: ants.ANTsImage, output_dir: str, 
                             method: str = 'nearest', chunk_size: int = 50,
                             label_id_lut: np.ndarray | None = None,
                             *,
                             max_workers: int = 0,
                             slice_batch: int | None = None) -> None:
        """分块上采样标签图像 - 强制对齐到原始尺寸"""
        _ = method, chunk_size
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        arr = label_image.numpy()
        # ANTs numpy is (X, Y, Z); pipeline volumes are saved as (Z, Y, X).
        source_volume = decode_label_codes(np.transpose(arr, (2, 1, 0)), label_id_lut)
        
        source_shape = source_volume.shape
        target_shape = self.original_shape # (Z, Y, X)
        output_dtype = ensure_label_storage_dtype(source_volume)
        
        logger.info("Upsampling from %s to %s...", source_shape, target_shape)
        logger.debug("ANTs raw shape = %s, after transpose = %s", arr.shape, source_volume.shape)
        logger.info("Preserving label dtype as %s", output_dtype)
        
        # Z-axis mapping: For each target slice i, which source slice does it correspond to?
        z_indices = np.round(np.linspace(0, source_shape[0] - 1, target_shape[0])).astype(int)
        target_xy = (target_shape[2], target_shape[1])  # (X, Y) for cv2.resize

        read_batch = resolve_slice_batch(slice_batch, default=16)
        worker_count = resolve_stack_workers(max_workers)
        logger.info(
            "Upsampling with %d worker(s), %d slice(s) per batch",
            worker_count,
            read_batch,
        )

        jobs: list[dict[str, object]] = []
        for start, end in iter_batch_ranges(len(z_indices), read_batch):
            jobs.append(
                {
                    "source_volume": source_volume,
                    "z_indices": z_indices[start:end],
                    "target_start": start,
                    "target_xy": target_xy,
                    "dtype": output_dtype,
                    "output_dir": output_path,
                    "compression": "lzw",
                }
            )

        run_bounded_batches(
            jobs,
            _upsample_label_batch,
            worker_count=worker_count,
            progress_total=len(z_indices),
            desc="Upsampling slices",
            unit="slice",
        )
            
        logger.info("Saved %d slices to %s", len(z_indices), output_path)

    def save_registration_results(self, results: Dict, save_transforms: bool = False, 
                                save_registered_image: bool = False) -> None:
        """保存配准结果"""
        mode = results['mode']
        
        # 1. Save Warped Label / Mask
        if results.get('warped_label') is not None:
            if mode == 'atlas2image':
                # Upsample atlas label to original space
                # Save once per sample so all signal channels can reuse the same atlas label
                label_dir = self.sample_dir / "upsampled_atlas_label"
                label_zarr = self.sample_dir / "upsampled_atlas_label.zarr"
                need_tiff = self.save_upsampled_label or self.save_upsampled_label_zarr
                if not need_tiff:
                    logger.info(
                        "Skipping upsampled atlas label output "
                        "(registration.save_upsampled_label=false, save_upsampled_label_zarr=false)."
                    )
                else:
                    if label_dir.exists() and any(label_dir.iterdir()):
                        logger.info("Upsampled atlas label already exists at %s. Skipping upsampling.", label_dir)
                    else:
                        self.upsample_label_chunked(
                            results['warped_label'],
                            str(label_dir),
                            label_id_lut=self.atlas_label_id_lut,
                        )

                    if self.save_upsampled_label_zarr:
                        if label_zarr.exists():
                            logger.info("Upsampled atlas label Zarr already exists at %s. Skipping conversion.", label_zarr)
                        elif convert_tiff_to_zarr is None:
                            raise RuntimeError("TIFF-to-Zarr conversion support is unavailable.")
                        else:
                            convert_tiff_to_zarr(
                                label_dir,
                                label_zarr,
                                chunk_size=self.zarr_chunk_size,
                                dataset_name="0",
                            )

                    if self.registration_config.get("save_upsampled_label_hemisphere_zarr", False):
                        hemisphere_zarr = self.sample_dir / "atlas_label_hemisphere.zarr"
                        if hemisphere_zarr.exists():
                            logger.info("Hemisphere atlas label Zarr already exists at %s. Skipping conversion.", hemisphere_zarr)
                        else:
                            from pipeline_modules.registration.atlas_label_to_hemisphere import convert_atlas_label_to_hemisphere

                            hemisphere_input = label_zarr if label_zarr.exists() else label_dir
                            convert_atlas_label_to_hemisphere(
                                hemisphere_input,
                                hemisphere_zarr,
                                chunk_size=self.zarr_chunk_size,
                                dataset_name="0",
                            )

                    if not self.save_upsampled_label and label_dir.exists():
                        shutil.rmtree(label_dir)
                        logger.info("Removed intermediate upsampled atlas label TIFF stack: %s", label_dir)
            else:
                # Save warped mask (already in atlas space)
                mask_dir = self.sample_dir / f"ch{self.signal_channel}_warped_mask"
                # For image2atlas, warped_label is actually the warped sample mask
                # We need to be careful about dimensions. 
                # warped_label is an ANTsImage. _save_volume_as_tiff handles it.
                self._save_volume_as_tiff(results['warped_label'], mask_dir, prefix="mask")

        # 2. Save Warped Image
        if save_registered_image:
            image_dir = self.sample_dir / f"ch{self.register_channel}_warped_image"
            self._save_volume_as_tiff(results['warped_image'], image_dir, prefix="image")
            
            # Also save NIfTI
            nii_path = image_dir / f"{self.register_channel}_warped_image.nii.gz"
            warped_img = results['warped_image']
            ants.image_write(warped_img, str(nii_path)) # Use ants.image_write directly

        # 3. Save Transforms
        if save_transforms and 'transforms' in results:
            transforms_dir = self.sample_dir / "transforms"
            transforms_dir.mkdir(exist_ok=True)
            
            # Save Forward Transforms
            if 'fwdtransforms' in results['transforms']:
                for i, transform in enumerate(results['transforms']['fwdtransforms']):
                    if os.path.exists(transform):
                        shutil.copy(transform, transforms_dir / f"fwd_{i}_{os.path.basename(transform)}")
            
            # Save Inverse Transforms (Crucial for Heatmap/Image2Atlas later)
            if 'invtransforms' in results['transforms']:
                for i, transform in enumerate(results['transforms']['invtransforms']):
                    if os.path.exists(transform):
                        shutil.copy(transform, transforms_dir / f"inv_{i}_{os.path.basename(transform)}")

    def check_and_run_density_analysis(self, results: Dict):
        """检查并运行密度分析"""
        mask_folder = self.sample_dir / f"ch{self.signal_channel}_downsample_mask"
        
        if mask_folder.exists() and results.get('warped_label') is not None:
            logger.info("Found mask folder: %s. Starting density analysis...", mask_folder)
            
            # Save downsampled atlas label (registered)
            downsampled_label_dir = self.sample_dir / f"ch{self.signal_channel}_atlas_label_downsampled"
            self._save_volume_as_tiff(results['warped_label'], downsampled_label_dir, prefix="label")
            
            try:
                analyzer = BrainDensityAnalyzer(self.density_cfg_path)
                analysis_results = analyzer.analyze(str(mask_folder), str(downsampled_label_dir))
                
                output_excel = self.sample_dir / f"density_analysis_ch{self.signal_channel}.xlsx"
                analyzer.write_to_excel(analysis_results, str(output_excel))
                logger.info("Density analysis completed. Saved to %s", output_excel)
            except Exception as e:
                logger.error("Error during density analysis: %s", e)
                import traceback
                traceback.print_exc()
        elif not mask_folder.exists():
            logger.info("Density analysis skipped. Mask folder %s not found.", mask_folder)

    def run_full_pipeline(self, mode: str = 'atlas2image', registration_type: str = 'SyN',
                         save_registered_image: bool = True, save_transforms: bool = False) -> None:
        """运行完整流程"""
        results = self.register(mode, registration_type)
        self.save_registration_results(results, save_transforms, save_registered_image)
        
        # Density analysis is now handled by main.py
        # if mode == 'atlas2image':
        #     self.check_and_run_density_analysis(results)


def main():
    import argparse
    import sys as _sys
    
    parser = argparse.ArgumentParser(description="Bidirectional registration between Allen atlas and LSFM images")
    parser.add_argument('--signal_channel', required=True, help='Signal channel (e.g. 0)')
    parser.add_argument('--sample_dir', required=True, help='Sample root directory')
    parser.add_argument('--atlas_image', required=True, help='Allen atlas image path')
    parser.add_argument('--atlas_label', required=True, help='Allen atlas label path')
    parser.add_argument('--register_channel', required=True, help='Registration channel')
    parser.add_argument('--save_registered_image', action='store_true', help='Save registered image')
    parser.add_argument('--mode', default='atlas2image', choices=['atlas2image', 'image2atlas'], help='Direction')
    parser.add_argument('--registration_type', default='SyN', choices=['Rigid', 'Affine', 'SyN', 'SyNRA', 'ElasticSyN'])
    parser.add_argument('--upsample_method', default='nearest', choices=['nearest', 'linear', 'cubic', 'quintic'])
    parser.add_argument('--chunk_size', type=int, default=50, help='Chunk size for upsampling')
    parser.add_argument('--save_transforms', action='store_true', help='Save transforms')
    parser.add_argument('--density_cfg', help='Density analysis config path')
    parser.add_argument('--config', help='Path to main config.json')
    parser.add_argument(
        '--json_logs',
        action='store_true',
        help='Emit NDJSON log records to stderr instead of plain text',
    )
    
    args = parser.parse_args()

    if args.json_logs:
        class _JsonFormatter(logging.Formatter):
            def format(self, record):
                return json.dumps({
                    "level": record.levelname,
                    "logger": record.name,
                    "message": record.getMessage(),
                })
        _handler = logging.StreamHandler(_sys.stderr)
        _handler.setFormatter(_JsonFormatter())
        logging.root.addHandler(_handler)
        logging.root.setLevel(logging.INFO)
    else:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    try:
        _started_at = time.time()

        # Load original shape if exists
        original_shape = None
        origin_shape_path = Path(args.sample_dir) / 'original_shape.json'
        if origin_shape_path.exists():
            with open(origin_shape_path, 'r', encoding='utf-8') as f:
                original_shape = tuple(json.load(f)['original_shape'])
            logger.info('Loaded original shape from %s: %s', origin_shape_path, original_shape)
        else:
            logger.info('Info: %s not found. Will infer original shape from raw data.', origin_shape_path)
        
        registrator = BidirectionalRegistration(
            args.sample_dir, args.signal_channel, args.atlas_image, args.atlas_label,
            args.register_channel, original_shape, args.density_cfg, args.config
        )
        
        registrator.run_full_pipeline(
            args.mode, args.registration_type, args.save_registered_image, args.save_transforms
        )

        # Write run manifest
        if write_run_manifest is not None:
            sample_dir = Path(args.sample_dir)
            _output_files = []
            label_dir = sample_dir / "upsampled_atlas_label"
            if label_dir.exists():
                _output_files.append(label_dir)
            label_zarr = sample_dir / "upsampled_atlas_label.zarr"
            if label_zarr.exists():
                _output_files.append(label_zarr)
            write_run_manifest(
                sample_dir,
                module="registration.ANTs_registration",
                entrypoint="run_full_pipeline",
                inputs={
                    "sample_dir": args.sample_dir,
                    "signal_channel": args.signal_channel,
                    "register_channel": args.register_channel,
                    "mode": args.mode,
                    "registration_type": args.registration_type,
                    "config": args.config,
                },
                outputs=_output_files,
                started_at=_started_at,
            )

        logger.info("Registration pipeline completed.")
    except Exception as exc:
        if PipelineError is not None and isinstance(exc, PipelineError):
            print(json.dumps({"error_code": exc.code.value, "message": str(exc.message)}), file=_sys.stderr)
            _sys.exit(exc.exit_code)
        logger.exception("Unhandled error: %s", exc)
        _sys.exit(1)


if __name__ == "__main__":
    main()
