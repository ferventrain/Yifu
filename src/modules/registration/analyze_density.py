#!/usr/bin/env python3
"""
Brain Region Density Analysis Tool

Usage:
    python analyze_density.py --mask_folder path/to/mask_folder --label_folder path/to/label_folder --cfg path/to/add_id_ytw.json --output path/to/output.xlsx
"""

import os
import sys
import argparse
import pandas as pd
import collections
import numpy as np
import json
import copy
from pathlib import Path
from tqdm import tqdm
import tifffile

from openpyxl.cell import MergedCell
from openpyxl.styles import Alignment, Border, Side, Font


class BrainDensityAnalyzer:
    """Brain region density analysis tool"""
    
    def __init__(self, cfg_path):
        """
        Initialize analyzer with configuration file
        
        Args:
            cfg_path: Path to add_id_ytw.json configuration file
        """
        self.cfg_path = cfg_path
        self.cfg = self._load_config()
        
    def _load_config(self):
        """Load configuration from JSON file"""
        if not os.path.exists(self.cfg_path):
            raise FileNotFoundError(f"Configuration file not found: {self.cfg_path}")
        
        with open(self.cfg_path, 'r') as f:
            return json.load(f)
    
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
            arr = tifffile.imread(str(tiff_file))
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
    
    def validate_shapes(self, mask_array, label_array):
        """
        Validate that mask and label arrays have the same shape
        
        Args:
            mask_array: 3D mask array
            label_array: 3D label array
            
        Returns:
            True if shapes match
        """
        if mask_array.shape != label_array.shape:
            raise ValueError(f"Shape mismatch! Mask: {mask_array.shape}, Label: {label_array.shape}")
        
        print(f"✓ Shape validation passed: {mask_array.shape}")
        return True
    
    def calculate_distribution(self, img):
        """
        Calculate pixel value distribution excluding background (0)
        
        Args:
            img: Input image array
            
        Returns:
            Counter object with pixel value counts
        """
        res = collections.Counter(img.flatten())
        if 0 in res:
            del res[0]
        return res
    
    def update_total_voxels_recur(self, cfg, distribution):
        """Recursively update total voxels in configuration tree"""
        if 'total_voxels' not in cfg:
            cfg['total_voxels'] = 0

        if len(cfg['children']):
            for child in cfg['children']:
                cfg['total_voxels'] += self.update_total_voxels_recur(child, distribution)
        
        if cfg['id_ytw'] in distribution:
            cfg['total_voxels'] += distribution[cfg['id_ytw']]

        return cfg['total_voxels']
    
    def update_seg_voxels_recur(self, cfg, distribution):
        """Recursively update segmentation voxels in configuration tree"""
        if 'seg_voxels' not in cfg:
            cfg['seg_voxels'] = 0

        if len(cfg['children']):
            for child in cfg['children']:
                cfg['seg_voxels'] += self.update_seg_voxels_recur(child, distribution)

        if cfg['id_ytw'] in distribution:
            cfg['seg_voxels'] += distribution[cfg['id_ytw']]

        return cfg['seg_voxels']
    
    def analyse_statistics(self, cfg, res=None):
        """Analyze statistics for all brain regions"""
        if res is None:
            keys = ['Brain regions', 'Acronym', 'Count', 'Total voxels', 'Density']
            deep_copied_dict = {key: [] for key in keys}
            res = {key: copy.deepcopy(deep_copied_dict) for key in range(12)}

        level = cfg['st_level']
        res[level]['Brain regions'].append(cfg['name'])
        res[level]['Acronym'].append(cfg['acronym'])
        res[level]['Count'].append(cfg['seg_voxels'])
        res[level]['Total voxels'].append(cfg['total_voxels'])
        
        if cfg['total_voxels'] > 0:
            res[level]['Density'].append(cfg['seg_voxels'] / cfg['total_voxels'])
        else:
            res[level]['Density'].append(0)

        if len(cfg['children']):
            for child in cfg['children']:
                self.analyse_statistics(child, res)
        
        return res
    
    def analyze(self, mask_folder, label_folder):
        """
        Main analysis function
        
        Args:
            mask_folder: Path to mask folder
            label_folder: Path to label folder
            
        Returns:
            Dictionary with analysis results
        """
        print("\n" + "="*50)
        print("Starting Brain Density Analysis")
        print("="*50)
        
        # **Load mask and label stacks**
        print("\n📁 Loading mask files...")
        mask_array = self.load_tiff_stack(mask_folder)
        
        print("\n📁 Loading label files...")
        label_array = self.load_tiff_stack(label_folder)
        
        # **Validate shapes**
        print("\n🔍 Validating array shapes...")
        self.validate_shapes(mask_array, label_array)
        
        # **Calculate label distribution：每个不同脑区的体积**
        print("\n📊 Calculating label distribution...")
        label_distribution = self.calculate_distribution(label_array)
        print(f"Found {len(label_distribution)} unique brain regions")
        
        # **Update total voxels in config**
        cfg_copy = copy.deepcopy(self.cfg)
        self.update_total_voxels_recur(cfg_copy, label_distribution)
        
        # **Process mask and calculate segmentation distribution**
        print("\n🧮 Processing mask and calculating densities...")
        
        # Binarize mask (any non-zero value becomes 1)
        mask_binary = (mask_array != 0).astype(np.uint8)
        
        # Apply mask to labels
        masked_labels = label_array * mask_binary
        
        # Calculate segmentation distribution
        seg_distribution = self.calculate_distribution(masked_labels)
        
        # **Update segmentation voxels in config**
        self.update_seg_voxels_recur(cfg_copy, seg_distribution)
        
        # **Analyze statistics**
        print("\n📈 Analyzing statistics...")
        results = self.analyse_statistics(cfg_copy)
        
        print("✅ Analysis completed successfully!")
        
        return results
    
    def col_num_to_letter(self, col_num):
        """Convert column number to Excel letter"""
        letter = ""
        while col_num > 0:
            col_num -= 1
            letter = chr(col_num % 26 + ord('A')) + letter
            col_num //= 26
        return letter
    
    def write_to_excel(self, results, output_path):
        """
        Write results to Excel file with separate sheets for each level
        
        Args:
            results: Analysis results dictionary
            output_path: Output Excel file path
        """
        print(f"\n💾 Saving results to: {output_path}")
        
        # Create parent directory if it doesn't exist
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Write to Excel with separate sheets for each level
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # Process each level that has data
            for level_index, data in results.items():
                if not data['Brain regions']:  # Skip empty levels
                    continue
                
                # Create DataFrame for this level
                df = pd.DataFrame(data)
                df.index = range(1, len(df) + 1)
                df.insert(0, 'Index', df.index)  # Add index column
                
                # Create sheet name for this level
                sheet_name = f'Level_{level_index}'
                
                # Write DataFrame to its own sheet
                df.to_excel(writer, index=False, sheet_name=sheet_name)
                
                # Get the worksheet for this level
                worksheet = writer.sheets[sheet_name]
                
                # Format the worksheet
                # Set font and alignment
                content_font = Font(name='Arial', size=11)
                for row in worksheet.iter_rows():
                    for cell in row:
                        cell.font = content_font
                        cell.alignment = Alignment(horizontal='left', wrap_text=False)

                # Set column widths
                for column in worksheet.columns:
                    max_length = 0
                    column_letter = column[0].column_letter
                    for cell in column:
                        try:
                            if len(str(cell.value)) > max_length:
                                max_length = len(str(cell.value))
                        except:
                            pass
                    adjusted_width = min(max_length + 2, 50)  # Limit width to 50
                    worksheet.column_dimensions[column_letter].width = adjusted_width
        
        print(f"✅ Results saved successfully with separate sheets for each level!")


def main():
    """Main entry point for command line usage"""
    parser = argparse.ArgumentParser(
        description='Brain Region Density Analysis Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    %(prog)s --mask_folder ch0_mask/ --label_folder ch0_atlas_label/ --cfg add_id_ytw.json --output density_analysis.xlsx
    %(prog)s -m ch0_downsampled_mask/ -l ch0_atlas_label_upsampled/ -c add_id_ytw.json -o output/density.xlsx
        """
    )
    
    parser.add_argument('-m', '--mask_folder', 
                        required=True,
                        help='Path to mask folder containing TIFF files')
    
    parser.add_argument('-l', '--label_folder',
                        required=True,
                        help='Path to label folder containing Allen brain atlas labels')
    
    parser.add_argument('-c', '--cfg',
                        default='add_id_ytw.json',
                        help='Path to configuration file (default: add_id_ytw.json)')
    
    parser.add_argument('-o', '--output',
                        default='density_analysis.xlsx',
                        help='Output Excel file path (default: density_analysis.xlsx)')
    
    parser.add_argument('-v', '--verbose',
                        action='store_true',
                        help='Enable verbose output')
    
    args = parser.parse_args()
    
    try:
        # Initialize analyzer
        analyzer = BrainDensityAnalyzer(args.cfg)
        
        # Run analysis
        results = analyzer.analyze(args.mask_folder, args.label_folder)
        
        # Save results
        analyzer.write_to_excel(results, args.output)
        
        print(f"\n🎉 Analysis complete! Results saved to: {args.output}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
