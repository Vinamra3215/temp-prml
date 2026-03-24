"""
Core experiment runner - builds pipeline, runs CV, logs results.
"""
import os
import time
import numpy as np
from src.data.cache import load_features
from src.models.registry import build_pipeline
from src.evaluation.metrics import evaluate
from src.evaluation.cross_val import stratified_cv, print_cv_results
from src.evaluation.comparison import append_result


def run_experiment(
    feature_type, model_name, model_params=None,
    reducer_name="none", reducer_params=None,
    cache_dir="data/cache", results_dir="results/metrics",
    cv_folds=5, run_id=None,
):
    """Run a single experiment: load features -> build pipeline -> CV -> evaluate."""
    print(f"\n{'='*60}")
    print(f"Experiment: {model_name} | Features: {feature_type} | Reducer: {reducer_name}")
    print(f"{'='*60}")

    X_train, y_train = load_features(cache_dir, feature_type, "train")
    X_test, y_test = load_features(cache_dir, feature_type, "test")

    pipeline = build_pipeline(model_name, model_params, reducer_name, reducer_params)
    start_time = time.time()

    cv_results = stratified_cv(pipeline, X_train, y_train, k=cv_folds)
    print_cv_results(cv_results, model_name)

    pipeline.fit(X_train, y_train)
    test_metrics, y_pred, y_prob = evaluate(pipeline, X_test, y_test)
    elapsed = time.time() - start_time

    print(f"\nTest Results:")
    for k, v in test_metrics.items():
        if v is not None:
            print(f"  {k}: {v:.4f}")
    print(f"  Time: {elapsed:.1f}s")

    run_data = {
        "run_id": run_id or f"{model_name}_{feature_type}_{reducer_name}",
        "feature": feature_type,
        "reducer": reducer_name,
        "model": model_name,
        "cv_accuracy": cv_results["test_accuracy"].mean(),
        "cv_f1": cv_results["test_f1_macro"].mean(),
        "test_accuracy": test_metrics["accuracy"],
        "test_f1": test_metrics["f1_macro"],
        "time_seconds": elapsed,
    }
    append_result(results_dir, run_data)

    return pipeline, test_metrics, y_pred
