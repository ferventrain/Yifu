#!/usr/bin/env python3
"""
Volume Calculator Tool

A program to calculate sample volume from TIFF image stacks using global thresholding.
Each pixel has dimensions of 1.8*1.8*2 cubic micrometers.

Usage:
    python volume_calculator.py --input_folder path/to/tiff_folder --threshold 100 --output result.txt
"""

import os
import sys
import argparse
import SimpleITK as sitk
import numpy as np
from pathlib import Path
from tqdm import tqdm


class VolumeCalculator:
    """Volume calculation tool using global thresholding"""
    
    def __init__(self, pixel_size_x=1.8, pixel_size_y=1.8, pixel_size_z=2.0):
        """
        Initialize calculator with pixel size
        
        Args:
            pixel_size_x: Size of pixel in x direction (micrometers)
            pixel_size_y: Size of pixel in y direction (micrometers) 
            pixel_size_z: Size of pixel in z direction (micrometers)
        """
        self.pixel_size_x = pixel_size_x
        self.pixel_size_y = pixel_size_y
        self.pixel_size_z = pixel_size_z
        self.pixel_volume = pixel_size_x * pixel_size_y * pixel_size_z  # in cubic micrometers
    
    def load_tiff_stack(self, folder_path, file_pattern="*.tif*"):
        """
        Load all TIFF files from a folder and stack them into a 3D array
        
        Args:
            folder_path: Path to folder containing TIFF files
            file_pattern: Pattern to match TIFF files
            
        Returns:
            3D numpy array (z, y, x)
        """
        folder = Path(folder_path)
        if not folder.exists():
            raise FileNotFoundError(f"Folder not found: {folder_path}")
        
        # Get all TIFF files and sort them
        tiff_files = sorted(folder.glob(file_pattern))
        
        if not tiff_files:
            raise ValueError(f"No TIFF files found in {folder_path}")
        
        print(f"Found {len(tiff_files)} TIFF files in {folder_path}")
        
        # Load all images
        images = []
        for tiff_file in tqdm(tiff_files, desc="Loading TIFF files"):
            img = sitk.ReadImage(str(tiff_file))
            arr = sitk.GetArrayFromImage(img)
            images.append(arr)
        
        # Stack into 3D array
        stack = np.array(images)
        
        # Handle different dimensions
        if stack.ndim == 3:
            return stack
        elif stack.ndim == 4:
            # If images are already 3D, reshape
            return stack.squeeze()
        else:
            raise ValueError(f"Unexpected array dimensions: {stack.shape}")
    
    def apply_threshold(self, img_array, threshold_value):
        """
        Apply global threshold to image array
        
        Args:
            img_array: 3D image array
            threshold_value: Threshold value for segmentation
            
        Returns:
            Binary mask array where True values are above threshold
        """
        print(f"Applying threshold of {threshold_value}...")
        binary_mask = img_array > threshold_value
        # Use int64 to handle large counts and avoid overflow
        pixel_count = np.count_nonzero(binary_mask)
        print(f"Thresholding completed. {pixel_count} pixels above threshold")
        return binary_mask
    
    def calculate_volume(self, binary_mask):
        """
        Calculate volume in cubic micrometers from binary mask
        
        Args:
            binary_mask: 3D binary array where True values represent sample
            
        Returns:
            Volume in cubic micrometers
        """
        # Use int64 to handle large counts and avoid overflow
        voxel_count = np.count_nonzero(binary_mask)
        volume = voxel_count * self.pixel_volume
        return voxel_count, volume


def main():
    """Main entry point for command line usage"""
    parser = argparse.ArgumentParser(
        description='Volume Calculator Tool using Global Thresholding',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    %(prog)s --input_folder images/ --threshold 100 --output volume_result.txt
    %(prog)s -i sample_images/ -t 150 -o results/volume.txt
        """
    )
    
    parser.add_argument('-i', '--input_folder', 
                        required=True,
                        help='Path to folder containing TIFF files')
    
    parser.add_argument('-t', '--threshold',
                        type=float,
                        required=True,
                        help='Threshold value for segmentation')
    
    parser.add_argument('-o', '--output',
                        default='volume_result.txt',
                        help='Output text file path (default: volume_result.txt)')
    
    parser.add_argument('--pixel_size_x',
                        type=float,
                        default=1.8,
                        help='Pixel size in x direction (micrometers, default: 1.8)')
    
    parser.add_argument('--pixel_size_y',
                        type=float,
                        default=1.8,
                        help='Pixel size in y direction (micrometers, default: 1.8)')
    
    parser.add_argument('--pixel_size_z',
                        type=float,
                        default=2.0,
                        help='Pixel size in z direction (micrometers, default: 2.0)')
    
    parser.add_argument('-v', '--verbose',
                        action='store_true',
                        help='Enable verbose output')
    
    args = parser.parse_args()
    
    try:
        # Initialize calculator
        calculator = VolumeCalculator(
            pixel_size_x=args.pixel_size_x,
            pixel_size_y=args.pixel_size_y,
            pixel_size_z=args.pixel_size_z
        )
        
        # Load TIFF stack
        print("Loading TIFF stack...")
        image_stack = calculator.load_tiff_stack(args.input_folder)
        print(f"Loaded image stack with shape: {image_stack.shape}")
        
        # Apply threshold
        binary_mask = calculator.apply_threshold(image_stack, args.threshold)
        
        # Calculate volume
        voxel_count, volume = calculator.calculate_volume(binary_mask)
        
        # Format results
        results = f"""Volume Calculation Results
======================
Input folder: {args.input_folder}
Threshold value: {args.threshold}
Pixel dimensions: {args.pixel_size_x} x {args.pixel_size_y} x {args.pixel_size_z} um
Stack dimensions: {image_stack.shape[2]} x {image_stack.shape[1]} x {image_stack.shape[0]} pixels

Results:
- Number of voxels above threshold: {voxel_count:,}
- Total volume: {volume:,.2f} cubic micrometers
- Total volume: {volume/1000000000:.2f} cubic millimeters
"""
        
        # Save results
        output_dir = os.path.dirname(args.output)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        with open(args.output, 'a', encoding='utf-8') as f:
            f.write(results)
        
        print(results)
        print(f"✅ Results saved to: {args.output}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()