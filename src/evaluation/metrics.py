"""
Evaluation metrics for classification.
"""
import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report, top_k_accuracy_score,
)


def evaluate(pipeline, X_test, y_test):
    """Compute all evaluation metrics."""
    y_pred = pipeline.predict(X_test)
    y_prob = None
    if hasattr(pipeline, "predict_proba"):
        try:
            y_prob = pipeline.predict_proba(X_test)
        except Exception:
            pass

    results = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_macro": f1_score(y_test, y_pred, average="macro"),
        "f1_weighted": f1_score(y_test, y_pred, average="weighted"),
        "precision_macro": precision_score(y_test, y_pred, average="macro"),
        "recall_macro": recall_score(y_test, y_pred, average="macro"),
    }

    if y_prob is not None:
        try:
            results["top5_accuracy"] = top_k_accuracy_score(y_test, y_prob, k=5)
        except Exception:
            results["top5_accuracy"] = None

    return results, y_pred, y_prob


def get_confusion_matrix(y_true, y_pred):
    """Compute confusion matrix."""
    return confusion_matrix(y_true, y_pred)


def get_classification_report(y_true, y_pred, class_names=None):
    """Get detailed classification report."""
    return classification_report(y_true, y_pred, target_names=class_names)
