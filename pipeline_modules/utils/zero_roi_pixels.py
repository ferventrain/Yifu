"""
Program to zero out pixel values in a specified ROI (defined by x, y boundaries) 
for specific 2D TIFF image slices, one at a time to prevent memory issues.

This script processes individual TIFF files in a folder, zeros out ROI values based on 
x, y boundaries in specified slices, and supports filename parsing for channel and slice info.
File format: YF2025102901_nanjingyikedaxue_nao_C0_Z0051 where C# is channel and Z### is slice.
"""

import numpy as np
import argparse
import os
from pathlib import Path
import re
from typing import Dict, List, Tuple, Optional


try:
    import tifffile
except ImportError:
    print("tifffile module not found. Please install with: pip install tifffile")
    exit(1)


def parse_filename(filename: str) -> Tuple[Optional[int], Optional[int]]:
    """
    Parse filename to extract channel and slice indices.
    
    Args:
        filename: Input filename in format YF2025102901_nanjingyikedaxue_nao_C0_Z0051.tif
        
    Returns:
        Tuple of (channel_index, slice_index) or (None, None) if parsing fails
    """
    # Extract the stem without extension
    stem = Path(filename).stem
    
    # Search for patterns C<digit> and Z<digits>
    channel_match = re.search(r'C(\d+)', stem)
    slice_match = re.search(r'Z(\d+)', stem)
    
    channel_idx = int(channel_match.group(1)) if channel_match else None
    slice_idx = int(slice_match.group(1)) if slice_match else None
    
    return channel_idx, slice_idx


def process_single_tif_file(
    filepath: str,
    x_min: int,
    x_max: int,
    y_min: int,
    y_max: int,
    output_dir: str
) -> bool:
    """
    Process a single TIFF file by zeroing out the specified ROI.
    
    Args:
        filepath: Path to the TIFF file
        x_min, x_max: X boundary indices
        y_min, y_max: Y boundary indices
        output_dir: Directory to save the processed file
        
    Returns:
        True if processing was successful
    """
    try:
        # Read the image
        image = tifffile.imread(str(filepath))
        
        # Validate x, y bounds against image dimensions
        height, width = image.shape[:2]  # Handle both grayscale and multichannel images
        
        x_min = max(0, min(x_min, width - 1))
        x_max = max(0, min(x_max, width - 1))
        y_min = max(0, min(y_min, height - 1))
        y_max = max(0, min(y_max, height - 1))
        
        if x_min > x_max or y_min > y_max:
            print(f"Warning: Invalid ROI bounds for {filepath}, skipping...")
            return False
        
        # Create a copy of the image to modify
        modified_image = image.copy()
        
        # Handle multi-channel images (last dimension is usually channels)
        if len(modified_image.shape) == 3:
            # For multichannel images: modify ROI across all channels
            modified_image[y_min:y_max+1, x_min:x_max+1, :] = 0
        else:
            # For grayscale images
            modified_image[y_min:y_max+1, x_min:x_max+1] = 0
        
        # Save the modified image to output directory
        output_path = Path(output_dir) / Path(filepath).name
        tifffile.imwrite(str(output_path), modified_image)
        
        print(f"Processed: {filepath} -> {output_path}")
        return True
        
    except Exception as e:
        print(f"Error processing {filepath}: {str(e)}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Zero out pixel values in a specified ROI for 2D TIFF images, one at a time."
    )
    
    parser.add_argument(
        "--input-folder",
        type=str,
        required=True,
        help="Input folder containing TIFF files"
    )
    
    parser.add_argument(
        "--output-folder",
        type=str,
        required=True,
        help="Output folder to save processed TIFF files"
    )
    
    parser.add_argument(
        "--x-min", 
        type=int, 
        required=True, 
        help="Minimum X boundary index"
    )
    
    parser.add_argument(
        "--x-max", 
        type=int, 
        required=True, 
        help="Maximum X boundary index"
    )
    
    parser.add_argument(
        "--y-min", 
        type=int, 
        required=True, 
        help="Minimum Y boundary index"
    )
    
    parser.add_argument(
        "--y-max", 
        type=int, 
        required=True, 
        help="Maximum Y boundary index"
    )
    
    parser.add_argument(
        "--z-min",
        type=int,
        default=None,
        help="Minimum Z slice index to process (inclusive)"
    )
    
    parser.add_argument(
        "--z-max",
        type=int,
        default=None,
        help="Maximum Z slice index to process (inclusive)"
    )
    
    parser.add_argument(
        "--target-channel",
        type=int,
        default=None,
        help="Only process TIFF files with this channel index (use -1 for all channels)"
    )
    
    parser.add_argument(
        "--target-slice",
        type=int,
        default=None,
        help="Only process TIFF files with this slice index (use -1 for all slices)"
    )
    
    args = parser.parse_args()
    
    # Validate input folder
    input_dir = Path(args.input_folder)
    if not input_dir.is_dir():
        print(f"Error: Input folder does not exist: {args.input_folder}")
        return
    
    # Validate output folder
    output_dir = Path(args.output_folder)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Processing TIFF files from: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"ROI boundaries - X: [{args.x_min}, {args.x_max}], Y: [{args.y_min}, {args.y_max}]")
    
    if args.target_channel is not None:
        print(f"Target channel: {args.target_channel}")
    if args.target_slice is not None:
        print(f"Target slice: {args.target_slice}")
    if args.z_min is not None and args.z_max is not None:
        print(f"Target slice range: [{args.z_min}, {args.z_max}]")
    
    # Find all TIFF files in the input directory
    tif_files = list(input_dir.glob("*.tif")) + list(input_dir.glob("*.tiff"))
    
    if not tif_files:
        print(f"No TIFF files found in {input_dir}")
        return
    
    print(f"Found {len(tif_files)} TIFF files")
    
    # Process each file individually
    processed_count = 0
    skipped_count = 0
    
    for tif_file in tif_files:
        # Parse filename to extract channel and slice indices
        channel_idx, slice_idx = parse_filename(tif_file.name)
        
        # Apply filters if specified
        # Check if channel matches (if specified)
        if args.target_channel is not None and channel_idx is not None and channel_idx != args.target_channel:
            skipped_count += 1
            continue  # Skip this file if it doesn't match the target channel
        
        # Check if slice matches (if specified with target-slice)
        if args.target_slice is not None and slice_idx is not None and slice_idx != args.target_slice:
            skipped_count += 1
            continue  # Skip this file if it doesn't match the target slice
        
        # Check if slice is in the Z range (if z-min and/or z-max specified)
        if (args.z_min is not None and slice_idx is not None and slice_idx < args.z_min) or \
           (args.z_max is not None and slice_idx is not None and slice_idx > args.z_max):
            skipped_count += 1
            continue  # Skip this file if it's outside the Z range
        
        # Process the TIFF file
        success = process_single_tif_file(
            str(tif_file),
            args.x_min, args.x_max,
            args.y_min, args.y_max,
            str(output_dir)
        )
        
        if success:
            processed_count += 1
        else:
            skipped_count += 1
    
    print(f"Processing completed!")
    print(f"Processed: {processed_count} files")
    print(f"Skipped: {skipped_count} files")


if __name__ == "__main__":
    main()