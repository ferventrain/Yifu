# Transfer Guide

## Files to replace in target repository
1. src/modules/registration/analyze_density.py
2. main.py
3. environment.yml (optional, for env setup)

## Required atlas files
Ensure Allen_brainatlas contains real binary files (not Git LFS pointer text):
- atlas.nii.gz or atlas.tiff
- atlas_label.nii.gz or atlas_label.tiff

## Environment setup
conda create -n yifu python=3.10 -y
conda activate yifu
conda env update -n yifu -f environment.yml

If network is unstable, install in batches:
- conda install -n yifu -c conda-forge simpleitk distributed dask-image dask-jobqueue bokeh pyqt pyqtgraph -y
- conda run -n yifu python -m pip install antspyx cellpose torch torchvision opencv-python-headless fastremap roifile fill-voids natsort tifffile tqdm nibabel

## Run full pipeline
# Optional OpenMP workaround on Windows if Step 2.2 errors:
# set KMP_DUPLICATE_LIB_OK=TRUE
python main.py --config config.json --sample_dir "<SAMPLE_ROOT>"

## Output columns (per Level sheet)
- Brain regions
- Acronym
- Count                 # td-tomato voxel count in region
- Total voxels
- Density               # Count / Total voxels
- tdTomato Count        # 3D connected-component object count (majority-region assignment)
- Brightness            # sum intensity over td-tomato voxels in region
- Mean brightness       # Brightness / Count
