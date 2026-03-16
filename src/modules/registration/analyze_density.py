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
    
    def process_streaming(self, mask_folder, label_folder):
        """
        Stream process mask and label folders slice by slice to save memory.
        Calculates distributions without loading full volume.
        
        Args:
            mask_folder: Path to mask folder
            label_folder: Path to label folder
            
        Returns:
            Tuple(label_distribution, seg_distribution)
        """
        mask_path = Path(mask_folder)
        label_path = Path(label_folder)
        
        # Get file lists
        mask_files = sorted(list(mask_path.glob("*.tif*")))
        label_files = sorted(list(label_path.glob("*.tif*")))
        
        n_mask = len(mask_files)
        n_label = len(label_files)
        
        if n_mask != n_label:
            print(f"⚠️ Warning: File count mismatch! Mask: {n_mask}, Label: {n_label}")
            # Determine the common range
            n_common = min(n_mask, n_label)
            print(f"⚠️ Truncating/Limiting analysis to the first {n_common} slices.")
            
            # Slice the lists to match the minimum length
            mask_files = mask_files[:n_common]
            label_files = label_files[:n_common]
            
        print(f"Streaming processing {len(mask_files)} slices...")
        
        label_dist_total = collections.Counter()
        seg_dist_total = collections.Counter()
        
        for mf, lf in tqdm(zip(mask_files, label_files), total=len(mask_files), desc="Processing Slices"):
            # Load single slice
            mask_slice = tifffile.imread(str(mf))
            label_slice = tifffile.imread(str(lf))
            
            # Validate shape (slice level)
            if mask_slice.shape != label_slice.shape:
                raise ValueError(f"Slice shape mismatch: {mf.name} {mask_slice.shape} vs {lf.name} {label_slice.shape}")
                
            # 1. Update Label Distribution (Total Voxels per Region)
            # Filter out background (0) if needed, usually label 0 is background
            lbl_counts = collections.Counter(label_slice.flatten())
            if 0 in lbl_counts:
                del lbl_counts[0]
            label_dist_total.update(lbl_counts)
            
            # 2. Update Segmentation Distribution (Signal Voxels per Region)
            # Mask binary: >0 is signal
            mask_binary = mask_slice > 0
            
            # Get labels where mask is True
            # This extracts label values only at signal positions
            signal_labels = label_slice[mask_binary]
            
            seg_counts = collections.Counter(signal_labels)
            if 0 in seg_counts:
                del seg_counts[0]
            seg_dist_total.update(seg_counts)
            
        return label_dist_total, seg_dist_total

    def analyze(self, mask_folder, label_folder):
        """
        Main analysis function
        """
        print("\n" + "="*50)
        print("Starting Brain Density Analysis (Streaming Mode)")
        print("="*50)
        
        # **Stream Process**
        print("\n🔄 Processing files stream...")
        label_distribution, seg_distribution = self.process_streaming(mask_folder, label_folder)
        
        print(f"\n📊 Found {len(label_distribution)} unique brain regions in atlas")
        print(f"🧮 Found signal in {len(seg_distribution)} regions")
        
        # **Update total voxels in config**
        cfg_copy = copy.deepcopy(self.cfg)
        self.update_total_voxels_recur(cfg_copy, label_distribution)
        
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
