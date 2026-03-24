"""
Feature fusion - concatenation of multiple feature spaces.
Covers: Course Topic #3 - Multi-dimensional features.
"""
import numpy as np
from src.features.base import FeatureExtractor
from src.features.histogram import ColorHistogramExtractor
from src.features.hog import HOGExtractor
from src.features.lbp import LBPExtractor
from src.features.glcm import GLCMExtractor


class FusedFeatureExtractor(FeatureExtractor):
    """Concatenate features from multiple extractors."""

    def __init__(self, extractors=None):
        if extractors is None:
            self.extractors = [
                ColorHistogramExtractor(bins=32),
                HOGExtractor(),
                LBPExtractor(),
                GLCMExtractor(),
            ]
        else:
            self.extractors = extractors

    def extract(self, image: np.ndarray) -> np.ndarray:
        features = [ext.extract(image) for ext in self.extractors]
        return np.concatenate(features).astype(np.float32)
