#!/usr/bin/env python3
# file_organizer.py

import os
import sys
import re
import shutil
from pathlib import Path

def organize_files(directory='.'):
    """
    将匹配 *_Ch\d* 模式的文件整理到对应文件夹
    """
    # 转换为Path对象
    target_dir = Path(directory).resolve()
    
    if not target_dir.exists():
        print(f"错误: 路径不存在 - {target_dir}")
        return False
    
    if not target_dir.is_dir():
        print(f"错误: 不是文件夹 - {target_dir}")
        return False
    
    print(f"正在处理: {target_dir}")
    
    # 匹配 *_Ch\d* 或 *_C\d_* 模式的文件
    # 模式1: 旧模式 *_Ch\d*
    pattern_old = re.compile(r'.*_Ch\d+.*')
    # 模式2: 新模式 *_C\d_* (例如 ..._C0_...)
    pattern_new = re.compile(r'.*_C(\d+)_.*\.tif.*', re.IGNORECASE)
    
    moved_count = 0
    
    # 获取所有匹配的文件
    files_to_process = [f for f in target_dir.iterdir() if f.is_file()]
    
    if not files_to_process:
        print("文件夹为空")
        return True

    for file_path in files_to_process:
        try:
            folder_name = None
            
            # 尝试匹配新模式 (C0, C1...)
            # 例如: YF2025090501_1_lanzhoudaxue_nao_3_C0_Z0006.tif
            match_new = pattern_new.match(file_path.name)
            if match_new:
                # 提取 C 后面的数字，如 "0"
                channel_num = match_new.group(1)
                # 文件夹名为 ch0, ch1... (chx格式)
                folder_name = f"ch{int(channel_num)}"
                
            # 尝试匹配旧模式 (Ch0, Ch1...)
            elif pattern_old.match(file_path.name):
                # 按 '_' 或 '.' 分割文件名
                parts = re.split(r'[_\.]', file_path.name)
                if len(parts) >= 2:
                    # 获取倒数第2个部分作为文件夹名 (例如 Ch0)
                    raw_folder_name = parts[-2]
                    # 尝试标准化为 chx 格式
                    if raw_folder_name.lower().startswith('ch'):
                        try:
                            num = int(raw_folder_name[2:])
                            folder_name = f"ch{num}"
                        except ValueError:
                            folder_name = raw_folder_name
                    else:
                        folder_name = raw_folder_name

            if folder_name:
                # 创建目标文件夹
                target_folder = target_dir / folder_name
                target_folder.mkdir(exist_ok=True)
                
                # 移动文件
                dest_path = target_folder / file_path.name
                shutil.move(str(file_path), str(dest_path))
                
                print(f"  已移动: {file_path.name} -> {folder_name}/")
                moved_count += 1
                
        except Exception as e:
            print(f"  错误: 处理 {file_path.name} 时出错 - {e}")
    
    print(f"\nFinished! (移动了 {moved_count} 个文件)")
    return True

def main():
    """主函数，处理命令行参数"""
    # 获取目标路径
    if len(sys.argv) > 1:
        # 从命令行参数获取路径
        target_path = sys.argv[1]
    else:
        # 交互式输入
        target_path = input("请输入要整理的文件夹路径 (直接回车使用当前目录): ").strip()
        if not target_path:
            target_path = '.'
    
    # 执行整理
    success = organize_files(target_path)
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
