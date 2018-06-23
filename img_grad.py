import numpy as np
from scipy import ndimage

def compute_image_grad(image):
    sx = ndimage.sobel(image, axis=0, mode='constant')
    sy = ndimage.sobel(image, axis=1, mode='constant')
    return np.sum(np.hypot(sx, sy))