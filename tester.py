import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

def test_pipeline(image):
    # denoise 
    denoised = image

    # threshold

def main():
    image_path = 'H:\arivis-analysis\huazichun\PBS_1\Ch0/632640_453670_038280_Ch0.tiff'
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if image is None:
        print(f"Failed to load image {image_path}")
        return
    print(f"Image shape: {image.shape}")
    image_processed = test_pipeline(image)
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap='gray')
    plt.title('Original Image')
    plt.subplot(1, 2, 2)
    plt.imshow(image_processed, cmap='gray')
    plt.title('Processed Image')
    plt.axis('off')
    plt.show()

if __name__ == '__main__':
    main()