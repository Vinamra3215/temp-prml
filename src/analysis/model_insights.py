"""
Model interpretation and analysis utilities.
Provides tools for understanding model behavior, errors, and feature importance.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


def analyze_confusion_matrix(y_true, y_pred, class_names):
    """
    Analyze confusion matrix to find most confused class pairs.

    Returns:
        dict with analysis results:
          - confused_pairs: list of (true_class, pred_class, count, rate)
          - per_class_accuracy: dict of class -> accuracy
          - hardest_classes: list of classes with lowest accuracy
          - easiest_classes: list of classes with highest accuracy
    """
    cm = confusion_matrix(y_true, y_pred)
    n_classes = len(class_names)

    # Per-class accuracy
    per_class_acc = {}
    for i in range(n_classes):
        total = cm[i].sum()
        correct = cm[i, i]
        per_class_acc[class_names[i]] = round(correct / total, 4) if total > 0 else 0

    # Most confused pairs (off-diagonal)
    confused_pairs = []
    for i in range(n_classes):
        for j in range(n_classes):
            if i != j and cm[i, j] > 0:
                rate = cm[i, j] / cm[i].sum()
                confused_pairs.append({
                    "true_class": class_names[i],
                    "predicted_as": class_names[j],
                    "count": int(cm[i, j]),
                    "error_rate": round(rate, 4),
                })

    confused_pairs.sort(key=lambda x: x["count"], reverse=True)

    # Sort classes by accuracy
    sorted_classes = sorted(per_class_acc.items(), key=lambda x: x[1])
    hardest = sorted_classes[:5]
    easiest = sorted_classes[-5:]

    return {
        "confused_pairs": confused_pairs[:10],  # Top 10 most confused
        "per_class_accuracy": per_class_acc,
        "hardest_classes": hardest,
        "easiest_classes": easiest,
    }


def generate_auto_summary(results_df, feature_type="cnn"):
    """
    Generate automatic text summary of experiment results.

    Args:
        results_df: DataFrame with columns [model, feature, test_accuracy, test_f1, time_seconds]
        feature_type: Primary feature type to focus on

    Returns:
        str: Multi-line summary text
    """
    if results_df.empty:
        return "No results available."

    df = results_df.dropna(subset=["test_accuracy"])
    if df.empty:
        return "No valid results."

    # Overall best
    best_idx = df["test_accuracy"].idxmax()
    best = df.loc[best_idx]

    # Best per feature
    cnn_df = df[df["feature"] == "cnn"]
    hc_df = df[df["feature"] != "cnn"]

    # Fastest
    fastest_idx = df["time_seconds"].idxmin()
    fastest = df.loc[fastest_idx]

    lines = [
        "=" * 60,
        "  EXPERIMENT SUMMARY (Auto-generated)",
        "=" * 60,
        "",
        f"Total experiments: {len(df)}",
        f"Models tested: {df['model'].nunique()}",
        f"Features tested: {df['feature'].nunique()}",
        "",
        "--- BEST RESULTS ---",
        f"Best overall:  {best['model']} on {best['feature']} "
        f"→ {best['test_accuracy']:.2%} accuracy, {best['test_f1']:.2%} F1",
        "",
    ]

    if not cnn_df.empty:
        best_cnn = cnn_df.loc[cnn_df["test_accuracy"].idxmax()]
        lines.append(
            f"Best CNN-based: {best_cnn['model']} → {best_cnn['test_accuracy']:.2%}"
        )

    if not hc_df.empty:
        best_hc = hc_df.loc[hc_df["test_accuracy"].idxmax()]
        lines.append(
            f"Best handcrafted: {best_hc['model']} on {best_hc['feature']} "
            f"→ {best_hc['test_accuracy']:.2%}"
        )

    if not cnn_df.empty and not hc_df.empty:
        gap = cnn_df["test_accuracy"].max() - hc_df["test_accuracy"].max()
        lines.append(f"CNN advantage: +{gap:.2%} over best handcrafted")

    lines.extend([
        "",
        "--- EFFICIENCY ---",
        f"Fastest model: {fastest['model']} on {fastest['feature']} "
        f"({fastest['time_seconds']:.1f}s)",
        "",
        "--- KEY INSIGHT ---",
    ])

    if not cnn_df.empty and cnn_df["test_accuracy"].max() > 0.7:
        lines.append(
            "CNN features capture high-level semantic representations that "
            "significantly outperform handcrafted features (HOG, LBP, etc.), "
            "demonstrating the power of transfer learning for food classification."
        )
    else:
        lines.append(
            "Results suggest feature quality is the primary bottleneck. "
            "Consider deeper CNN backbones or data augmentation."
        )

    lines.append("=" * 60)
    return "\n".join(lines)


def print_model_ranking(results_df, top_n=5):
    """Print a clean ranking table of top models."""
    df = results_df.dropna(subset=["test_accuracy"])
    df = df.sort_values("test_accuracy", ascending=False).head(top_n)

    print(f"\n{'Rank':<6}{'Model':<22}{'Features':<12}{'Accuracy':<12}{'F1':<12}")
    print("-" * 64)
    for rank, (_, row) in enumerate(df.iterrows(), 1):
        print(f"{rank:<6}{row['model']:<22}{row['feature']:<12}"
              f"{row['test_accuracy']:.2%}{'':>4}{row['test_f1']:.2%}")
