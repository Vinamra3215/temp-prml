"""
HOG (Histogram of Oriented Gradients) feature extractor.
Covers: Course Topics #2, #3.
"""
import cv2
import numpy as np
from skimage.feature import hog as skimage_hog
from src.features.base import FeatureExtractor


class HOGExtractor(FeatureExtractor):
    """Extract HOG features for texture/shape representation."""

    def __init__(self, orientations=9, pixels_per_cell=(8,8), cells_per_block=(2,2)):
        self.orientations = orientations
        self.pixels_per_cell = tuple(pixels_per_cell)
        self.cells_per_block = tuple(cells_per_block)

    def extract(self, image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        features = skimage_hog(
            gray, orientations=self.orientations,
            pixels_per_cell=self.pixels_per_cell,
            cells_per_block=self.cells_per_block,
            visualize=False, feature_vector=True,
        )
        return features.astype(np.float32)

    def extract_with_visualization(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        features, hog_image = skimage_hog(
            gray, orientations=self.orientations,
            pixels_per_cell=self.pixels_per_cell,
            cells_per_block=self.cells_per_block,
            visualize=True, feature_vector=True,
        )
        return features.astype(np.float32), hog_image
