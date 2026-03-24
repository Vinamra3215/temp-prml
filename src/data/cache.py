"""
Feature matrix caching via HDF5 for fast experiment iteration.
"""
import os

import h5py
import numpy as np


def get_cache_path(cache_dir: str, feature_type: str, split: str) -> str:
    """Get the cache file path for a given feature type and split."""
    return os.path.join(cache_dir, feature_type, f"{split}.h5")


def cache_exists(cache_dir: str, feature_type: str, split: str) -> bool:
    """Check if a cached feature matrix exists."""
    return os.path.exists(get_cache_path(cache_dir, feature_type, split))


def save_features(
    X: np.ndarray,
    y: np.ndarray,
    cache_dir: str,
    feature_type: str,
    split: str,
    class_names: list = None,
):
    """
    Save feature matrix and labels to HDF5.

    Args:
        X: Feature matrix (n_samples, n_features)
        y: Labels (n_samples,)
        cache_dir: Root cache directory
        feature_type: Feature extractor name
        split: 'train', 'val', or 'test'
        class_names: List of class name strings
    """
    path = get_cache_path(cache_dir, feature_type, split)
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with h5py.File(path, "w") as f:
        f.create_dataset("X", data=X, compression="gzip", compression_opts=4)
        f.create_dataset("y", data=y)
        if class_names is not None:
            f.attrs["classes"] = class_names
        f.attrs["n_samples"] = X.shape[0]
        f.attrs["n_features"] = X.shape[1]

    print(f"Cached {split} features ({feature_type}): {X.shape} -> {path}")


def load_features(
    cache_dir: str, feature_type: str, split: str
) -> tuple:
    """
    Load cached feature matrix and labels from HDF5.

    Returns:
        (X, y) tuple of numpy arrays
    """
    path = get_cache_path(cache_dir, feature_type, split)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No cached features at {path}. Run extract_features.py first."
        )

    with h5py.File(path, "r") as f:
        X = np.array(f["X"])
        y = np.array(f["y"])

    print(f"Loaded {split} features ({feature_type}): {X.shape}")
    return X, y
