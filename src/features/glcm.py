"""
GLCM (Gray-Level Co-occurrence Matrix) texture feature extractor.
Covers: Course Topic #3.
"""
import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops
from src.features.base import FeatureExtractor


class GLCMExtractor(FeatureExtractor):
    """Extract GLCM-based texture features."""

    def __init__(self, distances=None, angles=None, properties=None):
        self.distances = distances or [1, 3]
        self.angles = angles or [0, np.pi/4, np.pi/2, 3*np.pi/4]
        self.properties = properties or [
            "contrast", "dissimilarity", "homogeneity", "energy", "correlation"
        ]

    def extract(self, image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        gray = (gray // 4).astype(np.uint8)
        glcm = graycomatrix(
            gray, distances=self.distances, angles=self.angles,
            levels=64, symmetric=True, normed=True,
        )
        features = []
        for prop in self.properties:
            features.append(graycoprops(glcm, prop).flatten())
        return np.concatenate(features).astype(np.float32)
