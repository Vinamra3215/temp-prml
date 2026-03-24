"""
Color histogram feature extractor (RGB + HSV).
Covers: Course Topic #2 - Feature computation & classification using histogram.
"""
import cv2
import numpy as np
from src.features.base import FeatureExtractor


class ColorHistogramExtractor(FeatureExtractor):
    """Extract color histograms from RGB and HSV color spaces."""

    def __init__(self, bins: int = 32, color_spaces: list = None):
        self.bins = bins
        self.color_spaces = color_spaces or ["rgb", "hsv"]

    def extract(self, image: np.ndarray) -> np.ndarray:
        histograms = []

        if "rgb" in self.color_spaces:
            for channel in range(3):
                hist = cv2.calcHist([image], [channel], None, [self.bins], [0, 256])
                histograms.append(hist.flatten())

        if "hsv" in self.color_spaces:
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            hist_h = cv2.calcHist([hsv], [0], None, [self.bins], [0, 180])
            hist_s = cv2.calcHist([hsv], [1], None, [self.bins], [0, 256])
            hist_v = cv2.calcHist([hsv], [2], None, [self.bins], [0, 256])
            histograms.extend([hist_h.flatten(), hist_s.flatten(), hist_v.flatten()])

        feature = np.concatenate(histograms)
        norm = feature.sum()
        if norm > 0:
            feature = feature / norm
        return feature.astype(np.float32)
