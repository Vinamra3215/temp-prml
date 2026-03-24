"""Unit tests for evaluation metrics."""
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from src.evaluation.metrics import evaluate


def test_evaluate():
    X = np.random.randn(100, 20)
    y = np.random.randint(0, 3, 100)
    pipe = Pipeline([("scaler", StandardScaler()), ("clf", KNeighborsClassifier(n_neighbors=3))])
    pipe.fit(X[:80], y[:80])
    metrics, y_pred, _ = evaluate(pipe, X[80:], y[80:])
    assert "accuracy" in metrics
    assert "f1_macro" in metrics
    assert 0 <= metrics["accuracy"] <= 1
