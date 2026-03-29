"""
Generate curated set of 8 presentation-ready plots for PRML evaluation.

No CNN/ResNet — uses only handcrafted features.

Reads from:
  results/metrics/master_no_pca.csv   (Table 1 — raw dimensions)
  results/metrics/master_with_pca.csv (Table 2 — PCA-200)

Generates exactly 8 plots in results/plots/:
  1. model_comparison.png        — All models ranked by test accuracy
  2. feature_comparison.png      — Feature ablation comparison
  3. pca_explained_variance.png  — Variance retained vs PCA components
  4. accuracy_vs_f1.png          — Accuracy-F1 relationship
  5. cv_vs_test_accuracy.png     — Generalization gap analysis
  6. confusion_best_model.png    — Confusion matrix for best model
  7. runtime_comparison.png      — Computational cost comparison
  8. cnn_loss_curve.png          — MLP training/validation loss dynamics

Also saves comparison tables as CSV and markdown.
"""
import sys
sys.path.insert(0, ".")
import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from src.utils.seed import seed_everything
from src.data.cache import load_features
from src.data.dataset import Food101Dataset
from src.models.registry import build_pipeline
from src.evaluation.metrics import evaluate, get_classification_report

seed_everything(42)

# ── Style ────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
    "axes.labelsize": 12,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
})
sns.set_theme(style="whitegrid", font_scale=1.05)

PLOTS_DIR = "results/plots"
METRICS_DIR = "results/metrics"
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)

# ── Load results ─────────────────────────────────────────────────────
no_pca_path = os.path.join(METRICS_DIR, "master_no_pca.csv")
with_pca_path = os.path.join(METRICS_DIR, "master_with_pca.csv")
legacy_path = os.path.join(METRICS_DIR, "master.csv")

df_no_pca, df_with_pca = None, None

if os.path.exists(no_pca_path):
    df_no_pca = pd.read_csv(no_pca_path).dropna(subset=["test_accuracy"])
    df_no_pca = df_no_pca.sort_values("test_accuracy", ascending=False).reset_index(drop=True)
    print(f"Loaded Table 1 (No PCA): {len(df_no_pca)} experiments")

if os.path.exists(with_pca_path):
    df_with_pca = pd.read_csv(with_pca_path).dropna(subset=["test_accuracy"])
    df_with_pca = df_with_pca.sort_values("test_accuracy", ascending=False).reset_index(drop=True)
    print(f"Loaded Table 2 (PCA):    {len(df_with_pca)} experiments")

if df_no_pca is not None:
    df = df_no_pca
elif df_with_pca is not None:
    df = df_with_pca
elif os.path.exists(legacy_path):
    df = pd.read_csv(legacy_path).dropna(subset=["test_accuracy"])
    df = df.sort_values("test_accuracy", ascending=False).reset_index(drop=True)
    print(f"Loaded legacy master.csv: {len(df)} experiments")
else:
    print("ERROR: No results CSV found. Run experiments first.")
    sys.exit(1)

dataset = Food101Dataset(root="data/", n_classes=20, seed=42)
class_names = dataset.class_names
print(f"Classes: {len(class_names)}\n")

# Find best model for confusion matrix
best_row = df.iloc[0]
BEST_MODEL = best_row["model"]
BEST_FEATURE = best_row["feature"]
print(f"Best model: {BEST_MODEL} on {BEST_FEATURE} ({best_row['test_accuracy']:.2%})\n")


# ── Helper ───────────────────────────────────────────────────────────
def save_comparison_table(df_in, label, csv_name, md_name):
    df_out = df_in[["model", "feature", "cv_accuracy", "cv_f1",
                     "test_accuracy", "test_f1", "time_seconds"]].copy()
    df_out.columns = ["Model", "Features", "CV Acc", "CV F1",
                       "Test Acc", "Test F1", "Time (s)"]
    for col in ["CV Acc", "CV F1", "Test Acc", "Test F1"]:
        df_out[col] = df_out[col].apply(lambda x: f"{x:.4f}")
    df_out["Time (s)"] = df_out["Time (s)"].apply(lambda x: f"{x:.1f}")
    df_out.to_csv(os.path.join(METRICS_DIR, csv_name), index=False)
    md = [f"# {label}\n", f"**Dataset:** Food-101 (20 classes)",
          "**Cross-validation:** Stratified 5-Fold\n\n",
          df_out.to_markdown(index=False),
          f"\n\n**Best:** {df_in.iloc[0]['model']} on {df_in.iloc[0]['feature']} "
          f"— {df_in.iloc[0]['test_accuracy']:.2%} accuracy\n"]
    with open(os.path.join(METRICS_DIR, md_name), "w") as f:
        f.write("\n".join(md))
    print(f"  Saved {csv_name} + {md_name}")


# ═════════════════════════════════════════════════════════════════════
# PLOT 1/8 — MODEL COMPARISON
# ═════════════════════════════════════════════════════════════════════
print("[1/8] model_comparison.png")
fig, ax = plt.subplots(figsize=(13, max(7, len(df) * 0.45)))
cmap = plt.cm.viridis(np.linspace(0.15, 0.85, len(df)))
labels = df["model"].str.replace("_", " ").str.title() + "  (" + df["feature"] + ")"
bars = ax.barh(range(len(df)), df["test_accuracy"], color=cmap, edgecolor="white", linewidth=0.5)
ax.set_yticks(range(len(df)))
ax.set_yticklabels(labels, fontsize=9)
ax.set_xlabel("Test Accuracy")
ax.set_title("Model Comparison — Test Accuracy (20 Food Classes)")
ax.invert_yaxis()
ax.set_xlim(0, max(df["test_accuracy"]) * 1.15)
ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
for bar, acc in zip(bars, df["test_accuracy"]):
    ax.text(acc + 0.005, bar.get_y() + bar.get_height() / 2,
            f"{acc:.1%}", va="center", fontsize=8, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "model_comparison.png"))
plt.close()

# ═════════════════════════════════════════════════════════════════════
# PLOT 2/8 — FEATURE COMPARISON (Ablation)
# ═════════════════════════════════════════════════════════════════════
print("[2/8] feature_comparison.png")
feat_df = df.groupby("feature")["test_accuracy"].max().sort_values(ascending=True)
fig, ax = plt.subplots(figsize=(10, max(4, len(feat_df) * 0.8)))
feat_colors = plt.cm.Set2(np.linspace(0, 1, len(feat_df)))
bars = ax.barh(feat_df.index.str.upper(), feat_df.values, color=feat_colors,
               edgecolor="white", linewidth=0.5, height=0.6)
ax.set_xlabel("Best Test Accuracy")
ax.set_title("Feature Ablation — Best Accuracy per Feature Type")
ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
for bar, val in zip(bars, feat_df.values):
    ax.text(val + 0.005, bar.get_y() + bar.get_height() / 2,
            f"{val:.1%}", va="center", fontweight="bold", fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "feature_comparison.png"))
plt.close()

# ═════════════════════════════════════════════════════════════════════
# PLOT 3/8 — PCA EXPLAINED VARIANCE
# ═════════════════════════════════════════════════════════════════════
print("[3/8] pca_explained_variance.png")
# Use fused features (highest-dim handcrafted) for PCA analysis
X_train, y_train = load_features("data/cache", "fused", "train")
X_test, y_test = load_features("data/cache", "fused", "test")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)
n_comp = min(300, X_scaled.shape[1])
pca_full = PCA(n_components=n_comp)
pca_full.fit(X_scaled)
cumvar = np.cumsum(pca_full.explained_variance_ratio_)

n_95 = int(np.argmax(cumvar >= 0.95) + 1)
n_99 = int(np.argmax(cumvar >= 0.99) + 1)

fig, ax = plt.subplots(figsize=(10, 5.5))
ax.plot(range(1, len(cumvar) + 1), cumvar, color="#3498db", linewidth=2)
ax.fill_between(range(1, len(cumvar) + 1), cumvar, alpha=0.1, color="#3498db")
ax.axhline(y=0.95, color="#e74c3c", linestyle="--", linewidth=1, label=f"95% variance (n={n_95})")
ax.axhline(y=0.99, color="#2ecc71", linestyle="--", linewidth=1, label=f"99% variance (n={n_99})")
ax.axvline(x=200, color="#9b59b6", linestyle=":", alpha=0.6, label="PCA-200 (our choice)")
ax.axvline(x=n_95, color="#e74c3c", linestyle=":", alpha=0.3)
ax.axvline(x=n_99, color="#2ecc71", linestyle=":", alpha=0.3)
ax.set_xlabel("Number of Principal Components")
ax.set_ylabel("Cumulative Explained Variance Ratio")
ax.set_title(f"PCA on Fused Features — 95% at {n_95}, 99% at {n_99} components")
ax.legend(fontsize=10)
ax.grid(True, alpha=0.25)
ax.set_ylim(0, 1.02)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "pca_explained_variance.png"))
plt.close()

# ═════════════════════════════════════════════════════════════════════
# PLOT 4/8 — ACCURACY vs F1
# ═════════════════════════════════════════════════════════════════════
print("[4/8] accuracy_vs_f1.png")
fig, ax = plt.subplots(figsize=(9, 7))
sc = ax.scatter(df["test_accuracy"], df["test_f1"], s=110,
                c=df["test_accuracy"], cmap="viridis", edgecolors="black",
                linewidths=0.8, zorder=5)
for _, row in df.iterrows():
    label = row["model"].replace("_", " ").title()
    ax.annotate(label, (row["test_accuracy"] + 0.003, row["test_f1"] - 0.003),
                fontsize=7.5, alpha=0.85)
lims = [min(df["test_accuracy"].min(), df["test_f1"].min()) - 0.05, 1.0]
ax.plot(lims, lims, "k--", alpha=0.25, label="y = x (perfect correlation)")
ax.set_xlabel("Test Accuracy")
ax.set_ylabel("Test F1 Score (Macro)")
ax.set_title("Accuracy vs F1 — Metric Agreement Analysis")
ax.legend(fontsize=10)
ax.grid(True, alpha=0.25)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "accuracy_vs_f1.png"))
plt.close()

# ═════════════════════════════════════════════════════════════════════
# PLOT 5/8 — CV vs TEST ACCURACY (Generalization)
# ═════════════════════════════════════════════════════════════════════
print("[5/8] cv_vs_test_accuracy.png")
fig, ax = plt.subplots(figsize=(9, 7))
sc = ax.scatter(df["cv_accuracy"], df["test_accuracy"], s=120,
                c=df["test_accuracy"], cmap="viridis", edgecolors="black",
                linewidths=0.8, zorder=5)
for _, row in df.iterrows():
    label = row["model"].replace("_", " ").split()[0].title()
    ax.annotate(label, (row["cv_accuracy"] + 0.003, row["test_accuracy"] - 0.003),
                fontsize=8, alpha=0.85)
lims2 = [min(df["cv_accuracy"].min(), df["test_accuracy"].min()) - 0.05, 1.0]
ax.plot(lims2, lims2, "k--", alpha=0.25, label="Perfect generalization (CV = Test)")
ax.set_xlabel("Cross-Validation Accuracy (5-Fold)")
ax.set_ylabel("Test Set Accuracy")
ax.set_title("Generalization Analysis — CV vs Test Accuracy")
ax.legend(fontsize=10, loc="upper left")
ax.grid(True, alpha=0.25)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "cv_vs_test_accuracy.png"))
plt.close()

# ═════════════════════════════════════════════════════════════════════
# PLOT 6/8 — CONFUSION MATRIX (Best Model)
# ═════════════════════════════════════════════════════════════════════
print(f"[6/8] confusion_best_model.png ({BEST_MODEL} on {BEST_FEATURE})")
X_tr_cm, y_tr_cm = load_features("data/cache", BEST_FEATURE, "train")
X_te_cm, y_te_cm = load_features("data/cache", BEST_FEATURE, "test")

# Build and train best model pipeline (with GridSearch best params if available)
pipe = build_pipeline(BEST_MODEL, {})
pipe.fit(X_tr_cm, y_tr_cm)
y_pred = pipe.predict(X_te_cm)

cm = confusion_matrix(y_te_cm, y_pred)
fig, ax = plt.subplots(figsize=(13, 11))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
            xticklabels=class_names, yticklabels=class_names,
            linewidths=0.3, linecolor="white")
ax.set_xlabel("Predicted Label", fontsize=12)
ax.set_ylabel("True Label", fontsize=12)
ax.set_title(f"Confusion Matrix — {BEST_MODEL.replace('_',' ').title()} "
             f"({BEST_FEATURE} features)", fontsize=14, fontweight="bold")
plt.xticks(rotation=45, ha="right", fontsize=9)
plt.yticks(rotation=0, fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "confusion_best_model.png"))
plt.close()

report = get_classification_report(y_te_cm, y_pred, class_names=class_names)
with open(os.path.join(METRICS_DIR, f"classification_report_{BEST_MODEL}.txt"), "w") as f:
    f.write(f"Classification Report — {BEST_MODEL} ({BEST_FEATURE} features)\n")
    f.write("=" * 60 + "\n")
    f.write(report)

# ═════════════════════════════════════════════════════════════════════
# PLOT 7/8 — RUNTIME COMPARISON
# ═════════════════════════════════════════════════════════════════════
print("[7/8] runtime_comparison.png")
fig, ax = plt.subplots(figsize=(12, max(6, len(df) * 0.4)))
df_time = df.sort_values("time_seconds")
labels_t = df_time["model"].str.replace("_", " ").str.title() + "  (" + df_time["feature"] + ")"
cmap_t = plt.cm.plasma(np.linspace(0.15, 0.85, len(df_time)))
bars_t = ax.barh(labels_t, df_time["time_seconds"], color=cmap_t, edgecolor="white", linewidth=0.5)
ax.set_xlabel("Time (seconds)")
ax.set_title("Computational Cost — Training + Evaluation Runtime")
for bar, t in zip(bars_t, df_time["time_seconds"]):
    ax.text(t + max(df_time["time_seconds"]) * 0.01, bar.get_y() + bar.get_height() / 2,
            f"{t:.0f}s", va="center", fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "runtime_comparison.png"))
plt.close()

# ═════════════════════════════════════════════════════════════════════
# PLOT 8/8 — MLP LOSS CURVE (from saved history or trained fresh)
# ═════════════════════════════════════════════════════════════════════
print("[8/8] cnn_loss_curve.png")

history_path = os.path.join(METRICS_DIR, "cnn_history.json")

if os.path.exists(history_path):
    with open(history_path) as f:
        history = json.load(f)
    train_losses = history["train_loss"]
    val_losses = history["val_loss"]
    print("  Loaded loss history from cnn_history.json")
else:
    print("  cnn_history.json not found — training MLP for loss curve...")
    from sklearn.metrics import log_loss as _log_loss
    from sklearn.neural_network import MLPClassifier

    pca_r = PCA(n_components=200, random_state=42)
    sc2 = StandardScaler()
    X_tr_pca = pca_r.fit_transform(sc2.fit_transform(X_train))
    X_te_pca = pca_r.transform(sc2.transform(X_test))

    mlp = MLPClassifier(
        hidden_layer_sizes=(512, 256), activation="relu", solver="adam",
        learning_rate_init=0.001, batch_size=256,
        max_iter=1, warm_start=True, random_state=42,
    )
    n_ep = 50
    train_losses, val_losses = [], []
    classes = np.unique(y_train)
    for epoch in range(n_ep):
        mlp.max_iter = epoch + 1
        mlp.fit(X_tr_pca, y_train)
        train_losses.append(float(mlp.loss_))
        val_proba = mlp.predict_proba(X_te_pca)
        val_losses.append(float(_log_loss(y_test, val_proba, labels=classes)))
    with open(history_path, "w") as f:
        json.dump({"train_loss": train_losses, "val_loss": val_losses}, f, indent=2)

n_epochs = len(train_losses)
fig, ax = plt.subplots(figsize=(10, 6))
epochs = range(1, n_epochs + 1)
ax.plot(epochs, train_losses, "b-", linewidth=2, label="Training Loss", marker="o", markersize=3)
ax.plot(epochs, val_losses, "r-", linewidth=2, label="Validation Loss", marker="s", markersize=3)
best_epoch = int(np.argmin(val_losses)) + 1
best_val = min(val_losses)
ax.axvline(x=best_epoch, color="green", linestyle=":", alpha=0.6,
           label=f"Best validation (epoch {best_epoch})")
ax.scatter([best_epoch], [best_val], color="green", s=80, zorder=5, edgecolors="black")
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss (Cross-Entropy)")
ax.set_title("Neural Network Training Dynamics — Training vs Validation Loss")
ax.legend(fontsize=10)
ax.grid(True, alpha=0.25)
ax.set_xlim(1, n_epochs)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "cnn_loss_curve.png"))
plt.close()

# ═════════════════════════════════════════════════════════════════════
# SAVE COMPARISON TABLES
# ═════════════════════════════════════════════════════════════════════
print("\nSaving comparison tables...")

if df_no_pca is not None:
    save_comparison_table(df_no_pca,
                          "Table 1: Model Comparison — No PCA (Raw Dimensions)",
                          "comparison_table_no_pca.csv", "comparison_no_pca.md")

if df_with_pca is not None:
    save_comparison_table(df_with_pca,
                          "Table 2: Model Comparison — With PCA (200 Components)",
                          "comparison_table_with_pca.csv", "comparison_with_pca.md")

# ═════════════════════════════════════════════════════════════════════
# CLEANUP
# ═════════════════════════════════════════════════════════════════════
print("\nCleaning up old plots...")
unwanted = [
    "confusion_logistic.png", "confusion_random_forest.png", "confusion_svm_rbf.png",
    "dendrogram_cnn.png", "elbow_cnn.png", "bic_aic_cnn.png",
]
for fname in unwanted:
    path = os.path.join(PLOTS_DIR, fname)
    if os.path.exists(path):
        os.remove(path)
        print(f"  Removed {fname}")

for fname in ["classification_report_logistic.txt", "classification_report_random_forest.txt",
              "classification_report_svm_rbf.txt"]:
    path = os.path.join(METRICS_DIR, fname)
    if os.path.exists(path):
        os.remove(path)
        print(f"  Removed {fname}")

# ═════════════════════════════════════════════════════════════════════
print(f"\n{'=' * 60}")
print("DONE — 8 plots generated")
print(f"{'=' * 60}")
final_plots = sorted(os.listdir(PLOTS_DIR))
for p in final_plots:
    size_kb = os.path.getsize(os.path.join(PLOTS_DIR, p)) // 1024
    print(f"  {p}  ({size_kb} KB)")
print(f"\nTotal: {len(final_plots)} plots in {PLOTS_DIR}/")
