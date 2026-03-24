"""
Convolution filter demonstrations.
Covers: Course Topic #20.
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt


def demo_convolution_filters(image, save_path=None):
    """Apply and visualize classical convolution filters on a food image."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gaussian = cv2.GaussianBlur(gray, (5, 5), 0)
    gabor_kernel = cv2.getGaborKernel((21, 21), 5.0, np.pi / 4, 10.0, 0.5, 0)
    gabor = cv2.filter2D(gray, cv2.CV_8UC3, gabor_kernel)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    images = [gray, sobel_x, sobel_y, gaussian, gabor, laplacian]
    titles = ["Original", "Sobel-X", "Sobel-Y", "Gaussian", "Gabor", "Laplacian"]

    for ax, img, title in zip(axes.flatten(), images, titles):
        ax.imshow(img, cmap="gray")
        ax.set_title(title, fontsize=14)
        ax.axis("off")

    plt.suptitle("Convolution Filter Operations on Food Image", fontsize=16)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
