import torch
import torch.nn as nn
import numpy as np
import os
import argparse
from PIL import Image
import tifffile
from tqdm import tqdm
from pathlib import Path

def create_gaussian_kernel_3d(kernel_size=11, sigma=1.5):
    """创建一个3D高斯卷积核"""
    ax = np.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1.)
    xx, yy, zz = np.meshgrid(ax, ax, ax)
    kernel = np.exp(-(xx**2 + yy**2 + zz**2) / (2 * sigma**2))
    return kernel

def read_tiff_stack(path):
    path = Path(path)
    if path.is_dir():
        # Read directory of TIFFs
        files = sorted(list(path.glob('*.tif*')))
        if not files:
            raise FileNotFoundError(f"No TIFF files found in directory: {path}")
        images = [np.array(Image.open(f)) for f in files]
        return np.array(images)
    else:
        # Read single multi-page TIFF
        if not path.exists():
             raise FileNotFoundError(f"File not found: {path}")
        img = Image.open(path)
        images = []
        for i in range(img.n_frames):
            img.seek(i)
            slice = np.array(img)
            images.append(slice)
        return np.array(images)

def heatmap(save_img_path, edge_path, atlas_mask_path, save_path, alpha, sigma=1.5):
    '''
    :param sigma: 高斯核的标准差，控制平滑程度
    '''
    print(f"Loading input mask: {save_img_path}")
    img = read_tiff_stack(save_img_path)
    
    print(f"Loading edge reference: {edge_path}")
    edge = read_tiff_stack(edge_path)
    
    print(f"Loading atlas mask: {atlas_mask_path}")
    atlas_mask = read_tiff_stack(atlas_mask_path)
    
    # Ensure dimensions match
    if img.shape != atlas_mask.shape:
        print(f"Warning: Shape mismatch. Resizing input {img.shape} to match atlas {atlas_mask.shape} is not implemented.")
        # In a real pipeline, we might need resizing here if input isn't registered perfectly
    
    img[atlas_mask == 0] = 0
    print(f"Processing volume shape: {img.shape}")

    heatimg = np.zeros(img.shape)
    heatimg = np.array([heatimg, heatimg, heatimg]).transpose((1, 2, 3, 0))
    edge = np.array([edge, edge, edge]).transpose((1, 2, 3, 0))

    kernel_size = 11  # 卷积核大小
    radiation_matrix = create_gaussian_kernel_3d(kernel_size, sigma)

    radiation_matrix = radiation_matrix[np.newaxis, np.newaxis, ...]

    print("Applying 3D convolution (heatmap generation)...")
    conv = nn.Conv3d(1, 1, kernel_size, 1, padding=kernel_size//2, bias=False)
    conv.weight = nn.Parameter(torch.Tensor(radiation_matrix), requires_grad=False)
    
    # Use GPU if available for convolution
    if torch.cuda.is_available():
        conv = conv.cuda()
        input_tensor = torch.Tensor(img.astype(np.float32)[np.newaxis, np.newaxis, ...]).cuda()
    else:
        input_tensor = torch.Tensor(img.astype(np.float32)[np.newaxis, np.newaxis, ...])
        
    img_out = conv(input_tensor)

    if torch.cuda.is_available():
        img_out_heat = img_out.cpu().detach().numpy().squeeze()
    else:
        img_out_heat = img_out.detach().numpy().squeeze()
    
    # ==================== 核心修改：移除归一化 ====================
    # # 以下归一化步骤被移除，以保留绝对尺度用于样本间比较
    # if img_out_heat.max() > 0:
    #     img_out_heat = (img_out_heat / img_out_heat.max()) * 255
    # =============================================================
    
    img_out_heat *= alpha

    # 颜色映射逻辑保持不变，基于固定的阈值(255, 510, ...)
    # (0,0,255) --> (255,0,0)
    heatimg[..., 0][img_out_heat > 255] = img_out_heat[img_out_heat > 255] - 255
    heatimg[..., 2][img_out_heat > 255] = 255
    heatimg[..., 2][img_out_heat > 255 * 2] -= img_out_heat[img_out_heat > 255 * 2] - 255 * 2
    heatimg[..., 2][img_out_heat <= 255] = img_out_heat[img_out_heat <= 255]

    heatimg[edge != 0] = edge[edge != 0]
    heatimg[atlas_mask == 0] = 0

    heatimg[heatimg < 0] = 0
    heatimg[heatimg > 255] = 255

    print(f"Saving heatmap to: {save_path}")
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(save_path, np.uint8(heatimg), compression='lzw')
    print("Done!")


def main():
    parser = argparse.ArgumentParser(description="Generate 3D Heatmap from Cell Density/Mask")
    
    # Required arguments
    parser.add_argument('--input', required=True, help='Path to input mask/density image (TIFF stack or folder)')
    parser.add_argument('--output', required=True, help='Path to save output heatmap TIFF')
    
    # Optional arguments with defaults pointing to project structure
    # Assuming script is run from project root, default to s:\Yifu\Allen_brainatlas relative paths
    default_atlas_dir = Path(__file__).parent.parent / "Allen_brainatlas"
    
    parser.add_argument('--edge', default=str(default_atlas_dir / "edge.tiff"), 
                        help='Path to edge reference image')
    parser.add_argument('--atlas_mask', default=str(default_atlas_dir / "atlas_mask.tiff"), 
                        help='Path to atlas mask image')
    
    parser.add_argument('--alpha', type=float, default=2.0, help='Intensity scaling factor')
    parser.add_argument('--sigma', type=float, default=2.0, help='Gaussian smoothing sigma')
    
    args = parser.parse_args()
    
    heatmap(args.input, args.edge, args.atlas_mask, args.output, args.alpha, args.sigma)

if __name__ == "__main__":
    main()
