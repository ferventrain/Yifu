import torch
import torch.nn as nn
import numpy as np
import os
from PIL import Image
import tifffile
from tqdm import tqdm

def read_tiff_stack(path):
    if os.path.isdir(path):
        images = [np.array(Image.open(os.path.join(path, p))) for p in sorted(os.listdir(path))]
        return np.array(images)
    else:
        img = Image.open(path)
        images = []
        for i in range(img.n_frames):
            img.seek(i)
            slice = np.array(img)
            images.append(slice)
        return np.array(images)

def heatmap(save_img_path, edge_path, save_path, alpha):
    '''
    :param img_path: Segmentation image (axon or soma)
    :param edge_path: Edge of atlas, used as background lines
    :param save_path: Path to save heatmap
    :param alpha: Alpha of the heatmap
    :return: None
    '''
    img = read_tiff_stack(save_img_path)
    edge = read_tiff_stack(edge_path)
    # 新建热力图
    heatimg = np.zeros(img.shape)
    heatimg = np.array([heatimg, heatimg, heatimg]).transpose((1, 2, 3, 0))
    edge = np.array([edge, edge, edge]).transpose((1, 2, 3, 0))

    radiation_matrix = np.zeros((11, 11, 11))
    radiation_matrix[1:10, 1:10, 1:10] = 1
    radiation_matrix[4:7, 4:7, 4:7] = 2
    radiation_matrix[5, 5, 5] = 3

    radiation_matrix = radiation_matrix[np.newaxis, np.newaxis, ...]

    conv = nn.Conv3d(1, 1, 11, 1, padding=5, bias=False)
    conv.weight = nn.Parameter(torch.Tensor(radiation_matrix), requires_grad=False)
    img_out = conv(torch.Tensor(img.astype(np.float32)[np.newaxis, np.newaxis, ...]))

    img_out_heat = img_out.detach().numpy().squeeze()

    img_out_heat *= alpha


    # (0,0,255) --> (255,0,0)
    heatimg[..., 0][img_out_heat > 255] = img_out_heat[img_out_heat > 255] - 255
    heatimg[..., 2][img_out_heat > 255] = 255
    # 超过越多,蓝色越小,图片越红
    heatimg[..., 2][img_out_heat > 255 * 2] -= img_out_heat[img_out_heat > 255 * 2] - 255 * 2
    # 像素值越小，红色通道为0，蓝色通道值越小
    heatimg[..., 2][img_out_heat <= 255] = img_out_heat[img_out_heat <= 255]

    # 如果是为冠状面叠加脑区轮廓
    heatimg = np.array(heatimg).transpose(1, 0, 2, 3)
    # 叠加脑区轮廓
    heatimg[edge != 0] = edge[edge != 0]

    # 处理溢出值
    heatimg[heatimg < 0] = 0
    heatimg[heatimg > 255] = 255

    tifffile.imwrite(save_path, np.uint8(heatimg))


# executing
# ------------------------------------one image------------------------------------
img_path = r"C:\Users\Peiqi\Desktop\visual\processing\reg\new_version\P0_P28_final\collected_symmetry\heatmap\avg\horizon\image\P7_soma_avg_image.tiff"
edge_path = r"C:\Users\Peiqi\Desktop\visual\brainatlas_v4\P7\P7_edge.tiff"
save_path = r"C:\Users\Peiqi\Desktop\visual\processing\reg\new_version\P0_P28_final\collected_symmetry\heatmap\avg\horizon\image\P7_soma_avg_image_heatmap.tiff"

alpha = 1
if 'axon' in img_path:
    alpha = 0.02
elif 'soma' in img_path:
    alpha = 0.05

heatmap(img_path, edge_path, save_path, alpha)