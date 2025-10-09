import os
import json
import copy
import collections
from pathlib import Path
from typing import Dict, Optional, Tuple, List
import numpy as np
import pandas as pd
import tifffile
from tqdm import tqdm
from openpyxl.cell import MergedCell
from openpyxl.styles import Alignment, Border, Side, Font


class BrainRegionAnalyzer:
    """脑区分析类：计算密度和强度统计"""
    
    def __init__(self, sample_dir: str, cfg_path: str = "add_id_ytw.json"):
        """
        Args:
            sample_dir: 样本根目录路径 (SAMPLE_ID/)
            cfg_path: 配置文件路径 (add_id_ytw.json)
        """
        self.sample_dir = Path(sample_dir)
        self.cfg_path = cfg_path
        
        # 验证目录结构
        self._validate_directory_structure()
        
        # 加载配置
        with open(cfg_path, 'r') as f:
            self.cfg = json.load(f)
        
        # 获取原始形状
        with open(self.sample_dir / "original_shape.json", 'r') as f:
            shape_data = json.load(f)
            self.original_shape = tuple(shape_data['original_shape'])  # [z, y, x]
    
    def _validate_directory_structure(self):
        """验证必要的目录结构"""
        required_dirs = ['Ch0', 'Ch0_mask', 'Ch0_downsampled_reg']
        for dir_name in required_dirs:
            dir_path = self.sample_dir / dir_name
            if not dir_path.exists():
                raise FileNotFoundError(f"Required directory not found: {dir_path}")
        
        if not (self.sample_dir / "original_shape.json").exists():
            raise FileNotFoundError("original_shape.json not found")
    
    def calculate_distribution_chunked(self, 
                                      image_dir: Path,
                                      mask_dir: Path,
                                      atlas_dir: Path,
                                      chunk_size: int = 50) -> Tuple[Dict, Dict]:
        """
        分块计算分布，返回密度分布和强度分布
        
        Args:
            image_dir: 原始图像目录 (Ch0/)
            mask_dir: 分割掩码目录 (Ch0_mask/)
            atlas_dir: 配准后的atlas标签目录 (Ch0_downsampled_reg/)
            chunk_size: 每次处理的切片数量
            
        Returns:
            (density_distribution, intensity_distribution)
        """
        # 获取文件列表
        image_files = sorted(image_dir.glob('*.tif*'))
        mask_files = sorted(mask_dir.glob('*.tif*'))
        atlas_files = sorted(atlas_dir.glob('*.tif*'))
        
        # 验证文件数量一致
        if not (len(image_files) == len(mask_files) == len(atlas_files)):
            raise ValueError(f"File count mismatch: images={len(image_files)}, "
                           f"masks={len(mask_files)}, atlas={len(atlas_files)}")
        
        # 初始化计数器
        density_distribution = collections.Counter()
        intensity_distribution = collections.Counter()
        
        # 分块处理
        num_slices = len(image_files)
        num_chunks = (num_slices + chunk_size - 1) // chunk_size
        
        print(f"Processing {num_slices} slices in {num_chunks} chunks...")
        
        for chunk_idx in tqdm(range(num_chunks), desc="Processing chunks"):
            start_idx = chunk_idx * chunk_size
            end_idx = min(start_idx + chunk_size, num_slices)
            
            # 读取当前块
            image_chunk = []
            mask_chunk = []
            atlas_chunk = []
            
            for i in range(start_idx, end_idx):
                image_chunk.append(tifffile.imread(image_files[i]))
                mask_chunk.append(tifffile.imread(mask_files[i]))
                atlas_chunk.append(tifffile.imread(atlas_files[i]))
            
            # 转换为numpy数组
            image_chunk = np.stack(image_chunk, axis=0)
            mask_chunk = np.stack(mask_chunk, axis=0)
            atlas_chunk = np.stack(atlas_chunk, axis=0)
            
            # 二值化mask
            mask_binary = (mask_chunk > 0).astype(np.uint8)
            
            # 计算密度：每个脑区内mask=1的体素数量
            masked_atlas = atlas_chunk * mask_binary
            
            # 更新密度分布
            unique_labels, counts = np.unique(masked_atlas[masked_atlas > 0], 
                                             return_counts=True)
            for label, count in zip(unique_labels, counts):
                density_distribution[int(label)] += int(count)
            
            # 计算强度：每个脑区内mask=1位置的强度总和
            for label in np.unique(atlas_chunk[atlas_chunk > 0]):
                label_mask = (atlas_chunk == label) & (mask_binary > 0)
                intensity_sum = np.sum(image_chunk[label_mask])
                intensity_distribution[int(label)] += float(intensity_sum)
            
            # 释放内存
            del image_chunk, mask_chunk, atlas_chunk, mask_binary, masked_atlas
        
        return density_distribution, intensity_distribution
    
    def calculate_atlas_distribution_chunked(self, atlas_dir: Path, chunk_size: int = 50) -> Dict:
        """
        分块计算atlas的总体素分布
        
        Args:
            atlas_dir: 配准后的atlas标签目录
            chunk_size: 每次处理的切片数量
            
        Returns:
            脑区体素分布字典
        """
        atlas_files = sorted(atlas_dir.glob('*.tif*'))
        atlas_distribution = collections.Counter()
        
        num_slices = len(atlas_files)
        num_chunks = (num_slices + chunk_size - 1) // chunk_size
        
        print(f"Calculating atlas distribution for {num_slices} slices...")
        
        for chunk_idx in tqdm(range(num_chunks), desc="Processing atlas chunks"):
            start_idx = chunk_idx * chunk_size
            end_idx = min(start_idx + chunk_size, num_slices)
            
            # 读取当前块
            atlas_chunk = []
            for i in range(start_idx, end_idx):
                atlas_chunk.append(tifffile.imread(atlas_files[i]))
            
            atlas_chunk = np.stack(atlas_chunk, axis=0)
            
            # 统计体素
            unique_labels, counts = np.unique(atlas_chunk[atlas_chunk > 0], 
                                             return_counts=True)
            for label, count in zip(unique_labels, counts):
                atlas_distribution[int(label)] += int(count)
            
            del atlas_chunk
        
        return atlas_distribution
    
    def update_cfg_with_distributions(self, 
                                     atlas_distribution: Dict,
                                     density_distribution: Dict,
                                     intensity_distribution: Dict):
        """更新配置树的统计信息"""
        # 深拷贝配置，避免修改原始配置
        cfg_density = copy.deepcopy(self.cfg)
        cfg_intensity = copy.deepcopy(self.cfg)
        
        # 更新总体素数
        self._update_total_voxels_recur(cfg_density, atlas_distribution)
        self._update_total_voxels_recur(cfg_intensity, atlas_distribution)
        
        # 更新分割体素数（密度）
        self._update_seg_voxels_recur(cfg_density, density_distribution)
        
        # 更新强度总和
        self._update_intensity_sum_recur(cfg_intensity, intensity_distribution)
        
        return cfg_density, cfg_intensity
    
    def _update_total_voxels_recur(self, cfg, distribution):
        """递归更新总体素数"""
        if 'total_voxels' not in cfg:
            cfg['total_voxels'] = 0
        
        if 'id_ytw' not in cfg:
            cfg['id_ytw'] = cfg.get('id', 0)
        
        if len(cfg.get('children', [])):
            for child in cfg['children']:
                cfg['total_voxels'] += self._update_total_voxels_recur(child, distribution)
        
        if cfg['id_ytw'] in distribution:
            cfg['total_voxels'] += distribution[cfg['id_ytw']]
        
        return cfg['total_voxels']
    
    def _update_seg_voxels_recur(self, cfg, distribution):
        """递归更新分割体素数"""
        if 'seg_voxels' not in cfg:
            cfg['seg_voxels'] = 0
        
        if 'id_ytw' not in cfg:
            cfg['id_ytw'] = cfg.get('id', 0)
        
        if len(cfg.get('children', [])):
            for child in cfg['children']:
                cfg['seg_voxels'] += self._update_seg_voxels_recur(child, distribution)
        
        if cfg['id_ytw'] in distribution:
            cfg['seg_voxels'] += distribution[cfg['id_ytw']]
        
        return cfg['seg_voxels']
    
    def _update_intensity_sum_recur(self, cfg, distribution):
        """递归更新强度总和"""
        if 'intensity_sum' not in cfg:
            cfg['intensity_sum'] = 0
        
        if 'id_ytw' not in cfg:
            cfg['id_ytw'] = cfg.get('id', 0)
        
        if len(cfg.get('children', [])):
            for child in cfg['children']:
                cfg['intensity_sum'] += self._update_intensity_sum_recur(child, distribution)
        
        if cfg['id_ytw'] in distribution:
            cfg['intensity_sum'] += distribution[cfg['id_ytw']]
        
        return cfg['intensity_sum']
    
    def analyse_statistics(self, cfg, stat_type='density'):
        """分析统计数据"""
        res = {}
        
        if stat_type == 'density':
            keys = ['Brain regions', 'Acronym', 'Number of seg voxels', 
                   'Number of brain region voxels', 'Density']
        else:  # intensity
            keys = ['Brain regions', 'Acronym', 'Intensity sum', 
                   'Number of brain region voxels', 'Mean intensity']
        
        for level in range(12):
            res[level] = {key: [] for key in keys}
        
        self._analyse_statistics_recur(cfg, res, stat_type)
        return res
    
    def _analyse_statistics_recur(self, cfg, res, stat_type='density'):
        """递归分析统计数据"""
        level = cfg.get('st_level', 0)
        
        res[level]['Brain regions'].append(cfg.get('name', 'Unknown'))
        res[level]['Acronym'].append(cfg.get('acronym', 'N/A'))
        
        if stat_type == 'density':
            res[level]['Number of seg voxels'].append(cfg.get('seg_voxels', 0))
            res[level]['Number of brain region voxels'].append(cfg.get('total_voxels', 0))
            if cfg.get('total_voxels', 0) > 0:
                res[level]['Density'].append(cfg['seg_voxels'] / cfg['total_voxels'])
            else:
                res[level]['Density'].append(0)
        else:  # intensity
            res[level]['Intensity sum'].append(cfg.get('intensity_sum', 0))
            res[level]['Number of brain region voxels'].append(cfg.get('total_voxels', 0))
            if cfg.get('total_voxels', 0) > 0:
                res[level]['Mean intensity'].append(cfg['intensity_sum'] / cfg['total_voxels'])
            else:
                res[level]['Mean intensity'].append(0)
        
        for child in cfg.get('children', []):
            self._analyse_statistics_recur(child, res, stat_type)
    
    def write_to_excel(self, data, file_path):
        """写入Excel文件"""
        res = None
        for index, level_data in data.items():
            df = pd.DataFrame(level_data)
            df.index = range(1, len(df) + 1)
            
            if res is not None:
                df.insert(0, '', df.index)
                res = pd.concat([res, df], axis=1)
            else:
                df.insert(0, '', df.index)
                res = df
        
        with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
            res.to_excel(writer, index=False, startrow=1, startcol=0, sheet_name='Sheet1')
            worksheet = writer.sheets['Sheet1']
            
            # 格式化Excel
            self._format_excel(worksheet)
    
    def _format_excel(self, worksheet):
        """格式化Excel工作表"""
        # 去除标题行的边框
        for cell in worksheet[2]:
            cell.border = Border(left=Side(border_style=None),
                                right=Side(border_style=None),
                                top=Side(border_style=None),
                                bottom=Side(border_style=None))
        
        content_font = Font(name='Arial', size=11)
        for row in worksheet.iter_rows():
            for cell in row:
                cell.font = content_font
                cell.alignment = Alignment(horizontal='left')
        
        # 合并同一个level的单元格
        for i in range(12):
            a = 6 * i + 2
            b = a + 4
            a = self._col_num_to_letter(a)
            b = self._col_num_to_letter(b)
            alignment = Alignment(horizontal='center', vertical='center')
            worksheet.merge_cells(f'{a}1:{b}1')
            worksheet[f'{a}1'] = f'Level-{i}'
            worksheet[f'{a}1'].alignment = alignment
        
        # 设置列宽
        for col_idx, col in enumerate(worksheet.columns, start=1):
            if col_idx % 6 == 1:
                continue
            for cell in col:
                if isinstance(cell, MergedCell):
                    continue
                column_letter = cell.column_letter
                worksheet.column_dimensions[column_letter].width = 17
                break
    
    @staticmethod
    def _col_num_to_letter(col_num):
        """列号转字母"""
        letter = ""
        while col_num > 0:
            col_num -= 1
            letter = chr(col_num % 26 + ord('A')) + letter
            col_num //= 26
        return letter
    
    def run_analysis(self, output_dir: str, chunk_size: int = 50):
        """
        运行完整分析流程
        
        Args:
            output_dir: 输出目录
            chunk_size: 分块大小
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 定义路径
        image_dir = self.sample_dir / "Ch0"
        mask_dir = self.sample_dir / "Ch0_mask"
        atlas_dir = self.sample_dir / "Ch0_downsampled_reg" / "upsampled_label_stack"
        
        # 如果upsampled_label_stack不存在，尝试使用warped_atlas2image_label
        if not atlas_dir.exists():
            atlas_dir = self.sample_dir / "Ch0_downsampled_reg" / "warped_atlas2image_label"
            if not atlas_dir.exists():
                raise FileNotFoundError(f"Atlas label directory not found: {atlas_dir}")
        
        print(f"Image directory: {image_dir}")
        print(f"Mask directory: {mask_dir}")
        print(f"Atlas directory: {atlas_dir}")
        
        # 计算atlas总体素分布
        print("\n=== Calculating atlas distribution ===")
        atlas_distribution = self.calculate_atlas_distribution_chunked(atlas_dir, chunk_size)
        
        # 计算密度和强度分布
        print("\n=== Calculating density and intensity distributions ===")
        density_distribution, intensity_distribution = self.calculate_distribution_chunked(
            image_dir, mask_dir, atlas_dir, chunk_size
        )
        
        # 更新配置
        print("\n=== Updating configurations ===")
        cfg_density, cfg_intensity = self.update_cfg_with_distributions(
            atlas_distribution, density_distribution, intensity_distribution
        )
        
        # 分析统计
        print("\n=== Analyzing statistics ===")
        density_stats = self.analyse_statistics(cfg_density, 'density')
        intensity_stats = self.analyse_statistics(cfg_intensity, 'intensity')
        
        # 保存结果
        print("\n=== Saving results ===")
        density_file = output_path / f"{self.sample_dir.name}_density_analysis.xlsx"
        intensity_file = output_path / f"{self.sample_dir.name}_intensity_analysis.xlsx"
        
        self.write_to_excel(density_stats, density_file)
        self.write_to_excel(intensity_stats, intensity_file)
        
        print(f"Density analysis saved to: {density_file}")
        print(f"Intensity analysis saved to: {intensity_file}")
        
        # 保存汇总统计
        summary = {
            'sample_id': str(self.sample_dir.name),
            'total_brain_voxels': sum(atlas_distribution.values()),
            'total_seg_voxels': sum(density_distribution.values()),
            'total_intensity_sum': sum(intensity_distribution.values()),
            'overall_density': sum(density_distribution.values()) / sum(atlas_distribution.values()) if sum(atlas_distribution.values()) > 0 else 0,
            'num_brain_regions': len(atlas_distribution),
            'num_active_regions': len(density_distribution)
        }
        
        summary_file = output_path / f"{self.sample_dir.name}_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"Summary saved to: {summary_file}")
        
        return density_stats, intensity_stats


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Brain region density and intensity analysis")
    parser.add_argument('--sample_dir', required=True,
                       help='Sample directory path (SAMPLE_ID/)')
    parser.add_argument('--cfg_path', default='add_id_ytw.json',
                       help='Configuration file path')
    parser.add_argument('--output_dir', required=True,
                       help='Output directory for results')
    parser.add_argument('--chunk_size', type=int, default=50,
                       help='Number of slices to process at once')
    
    args = parser.parse_args()
    
    # 创建分析器并运行
    analyzer = BrainRegionAnalyzer(args.sample_dir, args.cfg_path)
    analyzer.run_analysis(args.output_dir, args.chunk_size)


if __name__ == "__main__":
    main()
