import argparse
import h5py
import os
import re
import tifffile
import time
import sys
import numpy as np
from concurrent.futures import ThreadPoolExecutor

# Configuration for performance
# Each slice is approx 150MB (10368*7552*2 bytes)
# High performance settings for powerful PC
# BATCH_SIZE = 128 means reading ~20GB at a time.
# 如果你的内存大于64GB，可以尝试这个值
BATCH_SIZE = 32
# 增加线程数以榨干CPU
MAX_WORKERS = 32


def get_ims_files(path):
    """Find IMS files in the given path (file or directory). Supports UNC/NAS paths."""
    # Normalize path to handle slashes correctly for the OS
    path = os.path.normpath(path)
    
    if os.path.isfile(path):
        if path.lower().endswith('.ims'):
            return [path]
        else:
            return []
    elif os.path.isdir(path):
        try:
            files = [os.path.join(path, f) for f in os.listdir(path) if f.lower().endswith('.ims')]
            return sorted(files)
        except Exception as e:
            print(f"Error accessing directory {path}: {e}")
            return []
    return []


def save_slice(args):
    """Worker function to save a single slice."""
    output_path, slice_data = args
    
    # 检查文件是否已存在（断点续传）
    if os.path.exists(output_path):
        return "exists"
    
    # 极速跳过空白：使用 max() 或 sum() 可能比 np.any() 更慢，但这里我们还是用 np.any()
    # 另一种更快的检查是看前几个像素
    # if slice_data[0,0] == 0 and not np.any(slice_data):
    #     return "skipped"
        
    try:
        # 禁用压缩以提高写入速度
        tifffile.imwrite(output_path, slice_data, compression=None, metadata={'axes': 'YX'})
        return "saved"
    except Exception as e:
        print(f"Error saving {output_path}: {e}")
        return "error"


def process_ims(ims_path, output_root, target_channel_idx):
    """Convert a specific channel from IMS to TIFF with batch processing."""
    print(f"Opening {ims_path}...")
    try:
        # Increase HDF5 chunk cache size for better read performance
        # 1GB cache for massive throughput
        cache_settings = {"rdcc_nbytes": 1024 * 1024 * 1024, "rdcc_nslots": 52000}
        with h5py.File(ims_path, "r", **cache_settings) as f:
            # Check structure
            if "DataSet" not in f or "ResolutionLevel 0" not in f["DataSet"]:
                 print(f"Error: Invalid IMS structure in {ims_path}")
                 return False

            res0 = f["DataSet"]["ResolutionLevel 0"]
            if "TimePoint 0" not in res0:
                print("Error: TimePoint 0 not found.")
                return False
            
            tp0 = res0["TimePoint 0"]
            
            # Find the channel key
            target_channel_key = None
            for key in tp0.keys():
                if key.startswith("Channel"):
                    try:
                        parts = key.split(" ")
                        if len(parts) > 1 and int(parts[1]) == target_channel_idx:
                            target_channel_key = key
                            break
                    except ValueError:
                        continue
            
            if not target_channel_key:
                print(f"Channel {target_channel_idx} not found in {ims_path}")
                return False

            # Create output subdirectory
            ch_dir = os.path.join(output_root, f"ch{target_channel_idx}")
            if not os.path.exists(ch_dir):
                os.makedirs(ch_dir)
                print(f"Created directory: {ch_dir}")

            # Get Data
            if "Data" not in tp0[target_channel_key]:
                print(f"Error: Data not found in {target_channel_key}")
                return False

            dataset = tp0[target_channel_key]["Data"]
            data_shape = dataset.shape
            
            if len(data_shape) == 3:
                z_dim, y_dim, x_dim = data_shape
            else:
                print(f"Unexpected data shape: {data_shape}")
                return False

            print(f"Processing Channel {target_channel_idx} in {os.path.basename(ims_path)}")
            print(f"Dimensions: Z={z_dim}, Y={y_dim}, X={x_dim}")
            print(f"Batch processing enabled (Batch Size: {BATCH_SIZE})")

            basename = os.path.splitext(os.path.basename(ims_path))[0]
            
            start_time = time.time()
            processed_count = 0
            skipped_count = 0
            exists_count = 0
            
            for z_start in range(0, z_dim, BATCH_SIZE):
                z_end = min(z_start + BATCH_SIZE, z_dim)
                current_batch_size = z_end - z_start
                
                # Check if all files in this batch already exist to skip reading
                # This is a huge optimization for resuming
                all_exist = True
                for i in range(current_batch_size):
                    z = z_start + i
                    filename = f"{basename}_C{target_channel_idx}_Z{z:04d}.tif"
                    output_path = os.path.join(ch_dir, filename)
                    if not os.path.exists(output_path):
                        all_exist = False
                        break
                
                if all_exist:
                    processed_count += current_batch_size
                    exists_count += current_batch_size
                    # print(f"Skipping existing batch {z_start}-{z_end}", end='\r')
                    continue

                # Read batch (Only if needed)
                chunk_data = dataset[z_start:z_end, :, :]
                
                # Prepare write tasks
                tasks = []
                # Pre-allocate output paths to avoid string concatenation in loop
                for i in range(current_batch_size):
                    z = z_start + i
                    # Use memoryview to avoid copy if possible
                    slice_data = chunk_data[i]
                    filename = f"{basename}_C{target_channel_idx}_Z{z:04d}.tif"
                    output_path = os.path.join(ch_dir, filename)
                    tasks.append((output_path, slice_data))
                
                # Parallel write
                # Use map to keep order but execution is async
                with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                    results = list(executor.map(save_slice, tasks))
                    
                skipped_count += results.count("skipped")
                exists_count += results.count("exists")
                
                processed_count += current_batch_size
                
                # Progress
                elapsed = time.time() - start_time
                if elapsed > 0:
                    rate = (processed_count - exists_count) / elapsed # Actual processing rate
                    if rate <= 0: rate = 0.1 # avoid div by zero
                    remaining = (z_dim - processed_count) / rate
                    print(f"Processed {processed_count}/{z_dim} ({(processed_count/z_dim*100):.1f}%) [Skipped: {skipped_count}, Exists: {exists_count}] - ETA: {remaining/60:.1f} min", end='\r')

            print(f"\nFinished {os.path.basename(ims_path)} Channel {target_channel_idx} in {(time.time() - start_time)/60:.1f} min")
            print(f"Summary: {exists_count} existed, {skipped_count} empty skipped.")
            return True

    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        raise
    except Exception as e:
        print(f"Error processing {ims_path}: {e}")
        import traceback
        traceback.print_exc()
        return False


def list_channel_indices(ims_path):
    """Return sorted channel indices from ResolutionLevel 0 / TimePoint 0."""
    with h5py.File(ims_path, "r") as f:
        if "DataSet" not in f or "ResolutionLevel 0" not in f["DataSet"] or "TimePoint 0" not in f["DataSet"]["ResolutionLevel 0"]:
            return []
        tp0 = f["DataSet"]["ResolutionLevel 0"]["TimePoint 0"]
        channels = [k for k in tp0.keys() if str(k).startswith("Channel")]
        indices = []
        for key in channels:
            parts = str(key).split(" ")
            if len(parts) > 1:
                indices.append(int(parts[1]))
        return sorted(indices)


def parse_channel_spec(spec, available):
    spec = (spec or "").strip().lower()
    if spec in ("", "all"):
        if not available:
            raise ValueError("No channels found in IMS file")
        return available
    parts = re.split(r"[,\s]+", spec)
    indices = [int(p) for p in parts if p.strip()]
    if not indices:
        raise ValueError("No channel indices parsed from: %s" % spec)
    return indices


def convert_ims_file(ims_path, output_dir, channel_indices):
    """Convert one IMS file, one channel after another. Returns False on failure."""
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n===== {os.path.basename(ims_path)} =====")
    print(f"IMS: {ims_path}")
    print(f"OUT: {output_dir}")
    print(f"Channels (sequential): {channel_indices}")
    for channel_idx in channel_indices:
        ok = process_ims(ims_path, output_dir, channel_idx)
        if not ok:
            print(f"FAILED: {os.path.basename(ims_path)} channel {channel_idx}")
            return False
    return True


def run_cli(args):
    ims_files = get_ims_files(os.path.normpath(args.input))
    if not ims_files:
        print("No .ims files found.")
        return 1
    output_dir = os.path.normpath(args.output)
    os.makedirs(output_dir, exist_ok=True)
    available = list_channel_indices(ims_files[0])
    print(f"Found {len(ims_files)} IMS file(s). Available channels: {available}")
    try:
        target_channels = parse_channel_spec(args.channels, available)
    except ValueError as exc:
        print(f"Invalid --channels: {exc}")
        return 1
    print(f"Starting sequential conversion for channels: {target_channels}")
    for ims_file in ims_files:
        if not convert_ims_file(ims_file, output_dir, target_channels):
            return 1
    print("\nAll done!")
    return 0


def run_interactive():
    print("--- IMS to TIFF Converter (Interactive & Fast) ---")

    # 1. Input IMS address
    while True:
        print("\n提示：支持输入本地路径 (如 D:\\Data) 或 NAS 路径 (如 \\\\192.168.110.31\\Yifu\\...)")
        ims_input = input("请输入ims存放地址 (Enter IMS file path or directory): ").strip()
        if ims_input:
            # Strip quotes if user copied as path with quotes
            if (ims_input.startswith('"') and ims_input.endswith('"')) or (ims_input.startswith("'") and ims_input.endswith("'")):
                ims_input = ims_input[1:-1]
            # Normalize path for Windows/NAS
            ims_input = os.path.normpath(ims_input)
            break

    ims_files = get_ims_files(ims_input)

    if not ims_files:
        print("未找到.ims文件 (No .ims files found).")
        input("按回车键退出 (Press Enter to exit)...")
        return 1

    print(f"找到 {len(ims_files)} 个IMS文件 (Found {len(ims_files)} IMS files).")

    # 2. Input Output address
    while True:
        output_dir = input("请输入tif想要存放的目标地址 (Enter LOCAL output directory): ").strip()
        if output_dir:
            if (output_dir.startswith('"') and output_dir.endswith('"')) or (output_dir.startswith("'") and output_dir.endswith("'")):
                output_dir = output_dir[1:-1]
            output_dir = os.path.normpath(output_dir)
            break

    if not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
            print(f"创建输出目录 (Created output directory): {output_dir}")
        except Exception as e:
            print(f"创建目录失败 (Error creating output directory): {e}")
            return 1

    # 3. Input Channels
    available_channel_indices = []
    try:
        available_channel_indices = list_channel_indices(ims_files[0])
        print(f"第一个文件中可用通道 (Available channels in first file): {available_channel_indices}")
    except Exception:
        pass

    target_channel_indices = []
    while True:
        channel_input = input("请输入想要输出的channel序号 (可输入 'all' 或多个序号如 '0,1,2'): ").strip().lower()
        if not channel_input:
            continue

        try:
            target_channel_indices = parse_channel_spec(channel_input, available_channel_indices)
            break
        except ValueError:
            print("无效的输入。请输入数字序号，如 '0' 或 '0,1,2'，或 'all'。")

    # Process one IMS file at a time, then one channel at a time
    print(f"\n开始转换通道 (Starting conversion for channels: {target_channel_indices})...")
    for ims_file in ims_files:
        if not convert_ims_file(ims_file, output_dir, target_channel_indices):
            return 1

    print("\n所有转换完成! (All done!)")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Stream Imaris IMS from NAS/local path to TIFF slices. Does not copy the IMS file."
    )
    parser.add_argument("--input", "-i", help="IMS file or directory containing IMS files")
    parser.add_argument("--output", "-o", help="Local output directory (writes ch0/, ch1/, ...)")
    parser.add_argument("--channels", default="all", help="Channel spec: all or 0,1")
    args = parser.parse_args(argv)

    try:
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)
    except Exception:
        pass

    if args.input or args.output:
        if not args.input or not args.output:
            parser.error("--input and --output are both required in CLI mode")
        return run_cli(args)
    return run_interactive()


if __name__ == "__main__":
    sys.exit(main() or 0)
