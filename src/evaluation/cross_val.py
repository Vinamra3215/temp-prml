"""
Stratified K-Fold cross-validation wrapper.
Covers: Course Topic #7.
"""
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_validate, learning_curve
import matplotlib.pyplot as plt


def stratified_cv(pipeline, X, y, k=5, scoring=None):
    """
    Stratified K-Fold CV. The entire Pipeline (scaler + reducer + clf)
    is fit independently on each fold. No data leakage.
    """
    scoring = scoring or ["accuracy", "f1_macro"]
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    results = cross_validate(
        pipeline, X, y, cv=skf,
        scoring=scoring,
        return_train_score=True,
        n_jobs=-1,
    )
    return results


def print_cv_results(results, model_name="Model"):
    """Print cross-validation results summary."""
    print(f"\n{'='*50}")
    print(f"{model_name}")
    print(f"{'='*50}")
    for key in results:
        if key.startswith("test_") or key.startswith("train_"):
            vals = results[key]
            print(f"  {key}: {vals.mean():.4f} +/- {vals.std():.4f}")


def plot_learning_curve(pipeline, X, y, title="Learning Curve", save_path=None):
    """Plot learning curve to analyze overfitting."""
    train_sizes, train_scores, val_scores = learning_curve(
        pipeline, X, y,
        train_sizes=np.linspace(0.1, 1.0, 10),
        cv=5, scoring="accuracy", n_jobs=-1,
    )
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_scores.mean(axis=1), "o-", label="Train")
    plt.plot(train_sizes, val_scores.mean(axis=1), "o-", label="Validation")
    plt.fill_between(train_sizes,
                     train_scores.mean(axis=1) - train_scores.std(axis=1),
                     train_scores.mean(axis=1) + train_scores.std(axis=1), alpha=0.1)
    plt.fill_between(train_sizes,
                     val_scores.mean(axis=1) - val_scores.std(axis=1),
                     val_scores.mean(axis=1) + val_scores.std(axis=1), alpha=0.1)
    plt.xlabel("Training Samples")
    plt.ylabel("Accuracy")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
