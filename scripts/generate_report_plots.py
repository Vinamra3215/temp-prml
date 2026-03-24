"""
Generate all result artifacts: comparison tables, plots, confusion matrices, metrics summaries.
Saves everything to results/plots/ and results/metrics/.
"""
import sys
sys.path.insert(0, ".")
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from src.utils.seed import seed_everything
from src.data.cache import load_features
from src.data.dataset import Food101Dataset
from src.models.registry import build_pipeline
from src.evaluation.metrics import evaluate, get_classification_report

seed_everything(42)
sns.set_theme(style="whitegrid", font_scale=1.1)

PLOTS_DIR = "results/plots"
METRICS_DIR = "results/metrics"
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)

# Load master results
df = pd.read_csv(os.path.join(METRICS_DIR, "master.csv"))
df = df.sort_values("test_accuracy", ascending=False).reset_index(drop=True)

# Load dataset for class names
dataset = Food101Dataset(root="data/", n_classes=20, seed=42)
class_names = dataset.class_names

print(f"Loaded {len(df)} experiment results, {len(class_names)} classes")

# =================================================================
# 1. MODEL COMPARISON BAR CHART
# =================================================================
print("\n[1/8] Model comparison bar chart...")
fig, ax = plt.subplots(figsize=(14, 8))
colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(df)))
labels = df["model"] + " / " + df["feature"]
bars = ax.barh(range(len(df)), df["test_accuracy"], color=colors)
ax.set_yticks(range(len(df)))
ax.set_yticklabels(labels)
ax.set_xlabel("Test Accuracy", fontsize=13)
ax.set_title("Model Comparison — Test Accuracy (20 Food Classes)", fontsize=15, fontweight="bold")
ax.invert_yaxis()
for i, (bar, acc) in enumerate(zip(bars, df["test_accuracy"])):
    ax.text(acc + 0.005, bar.get_y() + bar.get_height()/2,
            f"{acc:.1%}", va="center", fontsize=10, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "model_comparison.png"), dpi=150, bbox_inches="tight")
plt.close()

# =================================================================
# 2. CNN vs HANDCRAFTED FEATURES COMPARISON
# =================================================================
print("[2/8] Feature comparison chart...")
feat_df = df.groupby("feature")["test_accuracy"].max().sort_values(ascending=True)
fig, ax = plt.subplots(figsize=(10, 6))
colors_feat = ["#e74c3c" if f != "cnn" else "#2ecc71" for f in feat_df.index]
feat_df.plot(kind="barh", ax=ax, color=colors_feat)
ax.set_xlabel("Best Test Accuracy", fontsize=12)
ax.set_title("Feature Type Comparison (Best Model per Feature)", fontsize=14, fontweight="bold")
for i, (val, name) in enumerate(zip(feat_df.values, feat_df.index)):
    ax.text(val + 0.005, i, f"{val:.1%}", va="center", fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "feature_comparison.png"), dpi=150, bbox_inches="tight")
plt.close()

# =================================================================
# 3. ACCURACY vs F1 SCATTER
# =================================================================
print("[3/8] Accuracy vs F1 scatter...")
fig, ax = plt.subplots(figsize=(10, 8))
scatter = ax.scatter(df["test_accuracy"], df["test_f1"],
                     s=100, c=range(len(df)), cmap="viridis", edgecolors="black", zorder=5)
for _, row in df.iterrows():
    ax.annotate(f"{row['model']}/{row['feature'][:4]}",
                (row["test_accuracy"], row["test_f1"]),
                fontsize=7, ha="left", va="bottom")
ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="y=x")
ax.set_xlabel("Test Accuracy", fontsize=12)
ax.set_ylabel("Test F1 (Macro)", fontsize=12)
ax.set_title("Accuracy vs F1 Score", fontsize=14, fontweight="bold")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "accuracy_vs_f1.png"), dpi=150, bbox_inches="tight")
plt.close()

# =================================================================
# 4. TRAINING TIME COMPARISON
# =================================================================
print("[4/8] Training time comparison...")
fig, ax = plt.subplots(figsize=(12, 7))
df_time = df.sort_values("time_seconds")
ax.barh(df_time["model"] + " / " + df_time["feature"],
        df_time["time_seconds"], color=plt.cm.plasma(np.linspace(0.2, 0.9, len(df_time))))
ax.set_xlabel("Time (seconds)", fontsize=12)
ax.set_title("Experiment Runtime", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "runtime_comparison.png"), dpi=150, bbox_inches="tight")
plt.close()

# =================================================================
# 5. CONFUSION MATRICES FOR TOP 3 MODELS
# =================================================================
print("[5/8] Confusion matrices for top models...")
top_models = [
    ("logistic", "cnn", {"max_iter": 2000}, "none", None),
    ("svm_rbf", "cnn", {"C": 1.0, "gamma": "scale", "probability": True}, "none", None),
    ("random_forest", "cnn", {"n_estimators": 200, "max_depth": 30, "n_jobs": -1}, "none", None),
]

X_train, y_train = load_features("data/cache", "cnn", "train")
X_test, y_test = load_features("data/cache", "cnn", "test")

for model_name, feat, params, red, red_params in top_models:
    pipe = build_pipeline(model_name, params, red, red_params)
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=class_names, yticklabels=class_names)
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title(f"Confusion Matrix — {model_name} (CNN features)", fontsize=14, fontweight="bold")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"confusion_{model_name}.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # Save classification report
    report = get_classification_report(y_test, y_pred, class_names=class_names)
    with open(os.path.join(METRICS_DIR, f"classification_report_{model_name}.txt"), "w") as f:
        f.write(f"Classification Report — {model_name} (CNN features)\n")
        f.write("=" * 60 + "\n")
        f.write(report)
    print(f"  Saved confusion matrix & report for {model_name}")

# =================================================================
# 6. PCA EXPLAINED VARIANCE
# =================================================================
print("[6/8] PCA explained variance plot...")
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)
pca_full = PCA(n_components=min(200, X_scaled.shape[1]))
pca_full.fit(X_scaled)
cumvar = np.cumsum(pca_full.explained_variance_ratio_)

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(range(1, len(cumvar)+1), cumvar, "b-o", markersize=3)
ax.axhline(y=0.95, color="r", linestyle="--", label="95% variance")
ax.axhline(y=0.99, color="g", linestyle="--", label="99% variance")
n_95 = np.argmax(cumvar >= 0.95) + 1
n_99 = np.argmax(cumvar >= 0.99) + 1
ax.axvline(x=n_95, color="r", linestyle=":", alpha=0.5)
ax.axvline(x=n_99, color="g", linestyle=":", alpha=0.5)
ax.set_xlabel("Number of Components", fontsize=12)
ax.set_ylabel("Cumulative Explained Variance", fontsize=12)
ax.set_title(f"PCA on CNN Features — 95% at {n_95} dims, 99% at {n_99} dims", fontsize=14, fontweight="bold")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "pca_explained_variance.png"), dpi=150, bbox_inches="tight")
plt.close()

# =================================================================
# 7. CV ACCURACY vs TEST ACCURACY (Overfitting Analysis)
# =================================================================
print("[7/8] Overfitting analysis plot...")
fig, ax = plt.subplots(figsize=(10, 8))
ax.scatter(df["cv_accuracy"], df["test_accuracy"], s=120,
           c=range(len(df)), cmap="viridis", edgecolors="black", zorder=5)
for _, row in df.iterrows():
    ax.annotate(f"{row['model'][:6]}", (row["cv_accuracy"], row["test_accuracy"]),
                fontsize=8, ha="left", va="bottom")
ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Perfect generalization")
ax.set_xlabel("CV Accuracy (5-Fold)", fontsize=12)
ax.set_ylabel("Test Accuracy", fontsize=12)
ax.set_title("CV vs Test Accuracy — Overfitting Check", fontsize=14, fontweight="bold")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "cv_vs_test_accuracy.png"), dpi=150, bbox_inches="tight")
plt.close()

# =================================================================
# 8. SAVE FORMATTED COMPARISON TABLE
# =================================================================
print("[8/8] Saving comparison tables...")
df_display = df[["model", "feature", "reducer", "cv_accuracy", "cv_f1",
                  "test_accuracy", "test_f1", "time_seconds"]].copy()
df_display.columns = ["Model", "Features", "Reducer", "CV Acc", "CV F1",
                       "Test Acc", "Test F1", "Time (s)"]
for col in ["CV Acc", "CV F1", "Test Acc", "Test F1"]:
    df_display[col] = df_display[col].apply(lambda x: f"{x:.4f}")
df_display["Time (s)"] = df_display["Time (s)"].apply(lambda x: f"{x:.1f}")

# Save as CSV
df_display.to_csv(os.path.join(METRICS_DIR, "comparison_table.csv"), index=False)

# Save as markdown table
md_lines = ["# Model Comparison Results\n"]
md_lines.append(f"**Dataset:** Food-101 (20 classes, {len(X_train)} train, {len(X_test)} test)\n")
md_lines.append(f"**Cross-validation:** Stratified 5-Fold\n\n")
md_lines.append(df_display.to_markdown(index=False))
md_lines.append("\n\n## Key Findings\n")
md_lines.append(f"- **Best model:** {df.iloc[0]['model']} on {df.iloc[0]['feature']} features — {df.iloc[0]['test_accuracy']:.2%} accuracy\n")
md_lines.append(f"- **CNN features dominate:** Best CNN-based ({df[df['feature']=='cnn']['test_accuracy'].max():.2%}) vs best handcrafted ({df[df['feature']!='cnn']['test_accuracy'].max():.2%})\n")
md_lines.append(f"- **Fastest model:** {df.loc[df['time_seconds'].idxmin(), 'model']} ({df['time_seconds'].min():.1f}s)\n")

with open(os.path.join(METRICS_DIR, "comparison_results.md"), "w") as f:
    f.write("\n".join(md_lines))

print(f"\n{'='*60}")
print("ALL RESULTS GENERATED!")
print(f"{'='*60}")
print(f"Plots saved to: {PLOTS_DIR}/")
print(f"  - model_comparison.png")
print(f"  - feature_comparison.png")
print(f"  - accuracy_vs_f1.png")
print(f"  - runtime_comparison.png")
print(f"  - confusion_logistic.png")
print(f"  - confusion_svm_rbf.png")
print(f"  - confusion_random_forest.png")
print(f"  - pca_explained_variance.png")
print(f"  - cv_vs_test_accuracy.png")
print(f"\nMetrics saved to: {METRICS_DIR}/")
print(f"  - master.csv")
print(f"  - comparison_table.csv")
print(f"  - comparison_results.md")
print(f"  - classification_report_logistic.txt")
print(f"  - classification_report_svm_rbf.txt")
print(f"  - classification_report_random_forest.txt")
