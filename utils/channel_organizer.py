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
    
    # 匹配 *_Ch\d* 模式的文件
    pattern = re.compile(r'.*_ch\d+.*')
    moved_count = 0
    
    # 获取所有匹配的文件
    matching_files = [f for f in target_dir.iterdir() 
                     if f.is_file() and pattern.match(f.name)]
    
    if not matching_files:
        print("没有找到匹配 *_ch* 的文件")
        return True
    
    for file_path in matching_files:
        try:
            # 按 '_' 或 '.' 分割文件名
            parts = re.split(r'[_\.]', file_path.name)
            
            if len(parts) >= 2:
                # 获取倒数第2个部分作为文件夹名
                folder_name = parts[-2]
                
                # 创建目标文件夹 (对应 md $d -ea 0)
                target_folder = target_dir / folder_name
                target_folder.mkdir(exist_ok=True)
                
                # 移动文件 (对应 mv $_ $d)
                dest_path = target_folder / file_path.name
                shutil.move(str(file_path), str(dest_path))
                
                print(f"  已移动: {file_path.name} -> {folder_name}/")
                moved_count += 1
            else:
                print(f"  跳过: {file_path.name} (无法解析文件夹名)")
                
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
