"""Unit tests for model pipeline."""
import numpy as np
from src.models.registry import build_pipeline


def test_build_pipeline_knn():
    pipe = build_pipeline("knn", {"n_neighbors": 3})
    X = np.random.randn(100, 50)
    y = np.random.randint(0, 5, 100)
    pipe.fit(X, y)
    preds = pipe.predict(X[:10])
    assert len(preds) == 10


def test_build_pipeline_with_pca():
    pipe = build_pipeline("logistic", {"max_iter": 500}, reducer_name="pca", reducer_params={"n_components": 10})
    X = np.random.randn(100, 50)
    y = np.random.randint(0, 5, 100)
    pipe.fit(X, y)
    preds = pipe.predict(X[:10])
    assert len(preds) == 10
