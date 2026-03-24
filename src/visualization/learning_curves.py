"""Learning curves and accuracy-vs-k plots."""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier


def plot_knn_k_vs_accuracy(X_train, y_train, X_test, y_test,
                            k_range=range(1, 31), save_path=None):
    """Plot kNN accuracy vs k value."""
    accuracies_uniform = []
    accuracies_weighted = []
    for k in k_range:
        knn_u = KNeighborsClassifier(n_neighbors=k, weights="uniform")
        knn_u.fit(X_train, y_train)
        accuracies_uniform.append(knn_u.score(X_test, y_test))

        knn_w = KNeighborsClassifier(n_neighbors=k, weights="distance")
        knn_w.fit(X_train, y_train)
        accuracies_weighted.append(knn_w.score(X_test, y_test))

    plt.figure(figsize=(10, 6))
    plt.plot(list(k_range), accuracies_uniform, "b-o", markersize=4, label="Uniform")
    plt.plot(list(k_range), accuracies_weighted, "r-s", markersize=4, label="Distance-weighted")
    plt.xlabel("k")
    plt.ylabel("Accuracy")
    plt.title("kNN: k vs Accuracy")
    plt.legend()
    plt.grid(True, alpha=0.3)
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_training_history(history, title="Training History", save_path=None):
    """Plot MLP/CNN training loss and accuracy curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(history["train_loss"], label="Train")
    if "val_loss" in history:
        ax1.plot(history["val_loss"], label="Val")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss"); ax1.set_title("Loss"); ax1.legend(); ax1.grid(True, alpha=0.3)
    if "val_acc" in history:
        ax2.plot(history["val_acc"], label="Val Accuracy")
        ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy"); ax2.set_title("Accuracy"); ax2.legend(); ax2.grid(True, alpha=0.3)
    plt.suptitle(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
