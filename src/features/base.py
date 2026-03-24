"""
Abstract base class for all feature extractors.
"""
from abc import ABC, abstractmethod

import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

from src.data.preprocess import load_image


class FeatureExtractor(ABC):
    """Base class for feature extraction."""

    @abstractmethod
    def extract(self, image: np.ndarray) -> np.ndarray:
        """
        Extract feature vector from a single image.

        Args:
            image: HWC uint8 RGB image array

        Returns:
            1D feature vector
        """
        ...

    def extract_from_path(self, path: str, size: int = 224) -> np.ndarray:
        """Extract features from an image file path."""
        image = load_image(path, size=size)
        return self.extract(image)

    def extract_dataset(
        self,
        image_paths: list,
        labels: np.ndarray,
        size: int = 224,
        n_jobs: int = -1,
    ) -> tuple:
        """
        Parallel extraction over a full dataset.

        Args:
            image_paths: List of image file paths
            labels: Corresponding labels
            size: Image resize dimension
            n_jobs: Number of parallel workers (-1 = all CPUs)

        Returns:
            (X, y) tuple where X is (n_samples, n_features)
        """
        if n_jobs == 1:
            features = []
            valid_labels = []
            for i, path in enumerate(tqdm(image_paths, desc=self.__class__.__name__)):
                try:
                    feat = self.extract_from_path(path, size)
                    features.append(feat)
                    valid_labels.append(labels[i])
                except Exception:
                    continue
        else:
            results = Parallel(n_jobs=n_jobs, prefer="threads")(
                delayed(self._safe_extract)(path, size)
                for path in tqdm(image_paths, desc=self.__class__.__name__)
            )
            features = []
            valid_labels = []
            for i, feat in enumerate(results):
                if feat is not None:
                    features.append(feat)
                    valid_labels.append(labels[i])

        X = np.array(features, dtype=np.float32)
        y = np.array(valid_labels)
        return X, y

    def _safe_extract(self, path: str, size: int) -> np.ndarray:
        """Extract features, returning None on failure."""
        try:
            return self.extract_from_path(path, size)
        except Exception:
            return None
