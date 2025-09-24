import subprocess
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

def process_single_image(tif_file, ilastik_path, project_path, output_dir):
    """处理单个图像"""
    output_path = os.path.join(output_dir, f"{tif_file.stem}_result.tiff")
    
    cmd = [
        ilastik_path,
        "--headless",
        f"--project={project_path}",
        "--output_format=tiff",
        f"--output_filename_format={output_path}",
        str(tif_file)
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        return tif_file.name, result.returncode == 0, result.stderr
    except Exception as e:
        return tif_file.name, False, str(e)

def process_images_parallel(max_workers=4):
    """并行处理图像"""
    ilastik_path = r"C:\Program Files\ilastik-1.4.1.post1\ilastik.exe"
    project_path = r"H:\arivis-analysis\huazichun\MyProject.ilp"
    input_dir = r"H:\arivis-analysis\huazichun\PBS_1\Ch0"
    output_dir = r"H:\arivis-analysis\huazichun\PBS_1\Ch0_Results"
    
    os.makedirs(output_dir, exist_ok=True)
    tif_files = list(Path(input_dir).glob("*.tif"))
    
    print(f"Starting parallel processing of {len(tif_files)} files with {max_workers} workers")
    
    start_time = time.time()
    success_count = 0
    failure_count = 0
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_file = {
            executor.submit(process_single_image, file, ilastik_path, project_path, output_dir): file 
            for file in tif_files
        }
        
        # 处理完成的任务
        for future in as_completed(future_to_file):
            file_name, success, error = future.result()
            if success:
                success_count += 1
                print(f"✓ Success: {file_name} ({success_count}/{len(tif_files)})")
            else:
                failure_count += 1
                print(f"✗ Failed: {file_name} - {error}")
    
    end_time = time.time()
    print(f"\nProcessing completed in {end_time - start_time:.2f} seconds")
    print(f"Success: {success_count}, Failed: {failure_count}")

if __name__ == "__main__":
    # 根据您的CPU核心数调整max_workers
    process_images_parallel(max_workers=48)