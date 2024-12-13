import numpy as np
from scipy.signal import convolve2d


def box_filter(image, kernel_size=3):
    """
    Box filtresi uygular (ortalama filtresi).
    """
    kernel = np.ones((kernel_size, kernel_size)) / (kernel_size * kernel_size)
    filtered_image = convolve2d(image, kernel, mode="same", boundary="fill", fillvalue=0)
    return filtered_image


def median_filter(image, kernel_size=3):
    """
    Median filtresi uygular.
    """
    padded_image = np.pad(image, pad_width=kernel_size // 2, mode='constant', constant_values=0)
    filtered_image = np.zeros_like(image)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            local_window = padded_image[i:i + kernel_size, j:j + kernel_size]
            filtered_image[i, j] = np.median(local_window)

    return filtered_image


def gaussian_kernel(size, sigma=1):
    """
    Gaussian kernel oluşturur.
    """
    ax = np.linspace(-(size // 2), size // 2, size)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
    return kernel / np.sum(kernel)


def gaussian_filter(image, kernel_size=3, sigma=1):
    """
    Gaussian filtresi uygular.
    """
    kernel = gaussian_kernel(kernel_size, sigma)
    filtered_image = convolve2d(image, kernel, mode="same", boundary="fill", fillvalue=0)
    return filtered_image