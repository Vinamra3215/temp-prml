"""
LBP (Local Binary Pattern) feature extractor.
"""
import cv2
import numpy as np
from skimage.feature import local_binary_pattern
from src.features.base import FeatureExtractor


class LBPExtractor(FeatureExtractor):
    """Extract Local Binary Pattern histogram features."""

    def __init__(self, n_points=24, radius=3, method="uniform"):
        self.n_points = n_points
        self.radius = radius
        self.method = method

    def extract(self, image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        lbp = local_binary_pattern(gray, self.n_points, self.radius, method=self.method)
        n_bins = self.n_points + 2 if self.method == "uniform" else 256
        hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
        return hist.astype(np.float32)
