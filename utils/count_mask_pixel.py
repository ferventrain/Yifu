import cv2
import numpy as np

def count_mask_pixel(mask):
    """
    统计mask中像素的数量
    :param mask: 输入的mask，shape为(H, W)或(N, H, W)
    :return: mask中像素的数量
    """
    if mask.ndim == 2:
        return np.sum(mask)
    elif mask.ndim == 3:
        return np.sum(mask, axis=(1, 2))
    else:
        raise ValueError("mask的维度必须为2或3")