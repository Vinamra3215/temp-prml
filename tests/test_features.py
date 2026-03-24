"""Unit tests for feature extractors."""
import numpy as np
import pytest

# Create a dummy test image
def make_test_image(size=64):
    return np.random.randint(0, 255, (size, size, 3), dtype=np.uint8)


def test_color_histogram():
    from src.features.histogram import ColorHistogramExtractor
    ext = ColorHistogramExtractor(bins=16)
    feat = ext.extract(make_test_image())
    assert feat.ndim == 1
    assert len(feat) > 0
    assert np.isclose(feat.sum(), 1.0, atol=0.01)


def test_hog():
    from src.features.hog import HOGExtractor
    ext = HOGExtractor()
    feat = ext.extract(make_test_image())
    assert feat.ndim == 1
    assert len(feat) > 0


def test_lbp():
    from src.features.lbp import LBPExtractor
    ext = LBPExtractor()
    feat = ext.extract(make_test_image())
    assert feat.ndim == 1


def test_glcm():
    from src.features.glcm import GLCMExtractor
    ext = GLCMExtractor()
    feat = ext.extract(make_test_image())
    assert feat.ndim == 1


def test_fusion():
    from src.features.fusion import FusedFeatureExtractor
    ext = FusedFeatureExtractor()
    feat = ext.extract(make_test_image())
    assert feat.ndim == 1
    assert len(feat) > 100  # Should be concat of all features
