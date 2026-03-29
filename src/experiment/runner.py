"""
Core experiment runner — GridSearchCV for hyperparameter tuning, then eval.

Pipeline:
  1. Load cached features
  2. Build pipeline (Scaler → [PCA/LDA] → Classifier)
  3. GridSearchCV to find best hyperparameters via cross-validation
  4. Refit on full training set with best params
  5. Evaluate on test set
  6. Log to CSV + W&B
"""
import os
import time
import json
import numpy as np
from sklearn.model_selection import GridSearchCV

from src.data.cache import load_features
from src.models.registry import build_pipeline, get_param_grid
from src.evaluation.metrics import evaluate
from src.evaluation.cross_val import stratified_cv, print_cv_results
from src.evaluation.comparison import append_result
from src.utils.logging import (
    init_wandb, log_wandb, finish_wandb, log_experiment_csv, Timer
)


def run_experiment(
    feature_type, model_name, model_params=None,
    reducer_name="none", reducer_params=None,
    cache_dir="data/cache", results_dir="results/metrics",
    cv_folds=5, run_id=None, use_wandb=False,
    output_csv="master.csv", use_grid_search=True,
):
    """
    Run a single experiment with GridSearchCV hyperparameter tuning.

    Args:
        feature_type: Feature type key (histogram, hog, etc.)
        model_name: Model registry key (knn, logistic, kde, etc.)
        model_params: Dict of model hyperparameters (ignored if use_grid_search=True)
        reducer_name: "none", "pca", or "lda"
        reducer_params: Dict of reducer hyperparameters
        cache_dir: Path to cached features
        results_dir: Path to save results
        cv_folds: Number of CV folds
        run_id: Unique identifier for this run
        use_wandb: Whether to log to W&B
        output_csv: Name of output CSV file
        use_grid_search: If True, use GridSearchCV to find best params

    Returns:
        (pipeline, test_metrics, y_pred, best_params)
    """
    model_params = model_params or {}
    reducer_params = reducer_params or {}
    rid = run_id or f"{model_name}_{feature_type}_{reducer_name}"

    print(f"\n{'=' * 60}")
    print(f"Experiment: {model_name} | Features: {feature_type} | Reducer: {reducer_name}")
    print(f"GridSearch: {'ON' if use_grid_search else 'OFF'}")
    print(f"{'=' * 60}")

    # ── W&B init ─────────────────────────────────────────────────
    if use_wandb:
        config = {
            "model": model_name,
            "features": feature_type,
            "reducer": reducer_name,
            "cv_folds": cv_folds,
            "grid_search": use_grid_search,
        }
        tags = [model_name, feature_type, reducer_name]
        init_wandb(config=config, run_name=rid, tags=tags)

    # ── Load data ────────────────────────────────────────────────
    X_train, y_train = load_features(cache_dir, feature_type, "train")
    X_test, y_test = load_features(cache_dir, feature_type, "test")

    best_params = {}

    with Timer("Total experiment") as timer:
        if use_grid_search:
            # ── GridSearchCV ─────────────────────────────────────
            param_grid = get_param_grid(model_name)
            if param_grid:
                pipeline = build_pipeline(model_name, {}, reducer_name, reducer_params)
                print(f"  GridSearchCV: {len(param_grid)} param groups, {cv_folds}-fold CV")

                grid_search = GridSearchCV(
                    pipeline,
                    param_grid,
                    cv=cv_folds,
                    scoring="f1_macro",
                    n_jobs=-1,
                    verbose=0,
                    refit=True,
                )
                grid_search.fit(X_train, y_train)

                pipeline = grid_search.best_estimator_
                best_params = grid_search.best_params_
                cv_f1 = grid_search.best_score_

                print(f"  Best params: {best_params}")
                print(f"  Best CV F1:  {cv_f1:.4f}")

                # Get CV accuracy too (re-run with best estimator)
                from sklearn.model_selection import cross_val_score
                cv_acc_scores = cross_val_score(
                    pipeline, X_train, y_train,
                    cv=cv_folds, scoring="accuracy", n_jobs=-1
                )
                cv_acc = cv_acc_scores.mean()
            else:
                # No param grid for this model (e.g. naive_bayes with trivial grid)
                pipeline = build_pipeline(model_name, model_params, reducer_name, reducer_params)
                cv_results = stratified_cv(pipeline, X_train, y_train, k=cv_folds)
                print_cv_results(cv_results, model_name)
                cv_acc = cv_results["test_accuracy"].mean()
                cv_f1 = cv_results["test_f1_macro"].mean()
                pipeline.fit(X_train, y_train)
        else:
            # ── Fixed params (no grid search) ────────────────────
            pipeline = build_pipeline(model_name, model_params, reducer_name, reducer_params)
            cv_results = stratified_cv(pipeline, X_train, y_train, k=cv_folds)
            print_cv_results(cv_results, model_name)
            cv_acc = cv_results["test_accuracy"].mean()
            cv_f1 = cv_results["test_f1_macro"].mean()
            pipeline.fit(X_train, y_train)

        # ── Test evaluation ──────────────────────────────────────
        test_metrics, y_pred, y_prob = evaluate(pipeline, X_test, y_test)

    elapsed = timer.elapsed

    # ── Print results ────────────────────────────────────────────
    print(f"\nTest Results:")
    for k, v in test_metrics.items():
        if v is not None:
            print(f"  {k}: {v:.4f}")
    print(f"  Time: {elapsed:.1f}s")

    # ── Log to W&B ───────────────────────────────────────────────
    if use_wandb:
        log_wandb({
            "cv_accuracy": cv_acc,
            "cv_f1": cv_f1,
            "test_accuracy": test_metrics["accuracy"],
            "test_f1_macro": test_metrics["f1_macro"],
            "runtime_seconds": elapsed,
            **{f"best_{k}": v for k, v in best_params.items()},
        })
        finish_wandb()

    # ── Log to CSV ───────────────────────────────────────────────
    run_data = {
        "run_id": rid,
        "feature": feature_type,
        "reducer": reducer_name,
        "model": model_name,
        "cv_accuracy": round(cv_acc, 4),
        "cv_f1": round(cv_f1, 4),
        "test_accuracy": round(test_metrics["accuracy"], 4),
        "test_f1": round(test_metrics["f1_macro"], 4),
        "time_seconds": round(elapsed, 2),
    }
    append_result(results_dir, run_data, csv_name=output_csv)

    # ── Detailed experiment log ──────────────────────────────────
    log_experiment_csv(results_dir, {
        "run_id": rid,
        "model": model_name,
        "feature": feature_type,
        "reducer": reducer_name,
        "params": best_params if best_params else model_params,
        "cv_accuracy": round(cv_acc, 4),
        "cv_f1": round(cv_f1, 4),
        "test_accuracy": round(test_metrics["accuracy"], 4),
        "test_f1": round(test_metrics["f1_macro"], 4),
        "test_precision": round(test_metrics.get("precision_macro", 0), 4),
        "test_recall": round(test_metrics.get("recall_macro", 0), 4),
        "runtime": round(elapsed, 2),
    })

    return pipeline, test_metrics, y_pred, best_params
