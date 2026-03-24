"""
KMeans clustering with evaluation.
Covers: Course Topic #23.
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score


def run_kmeans(X, n_clusters=20, init="k-means++", n_init=10, random_state=42):
    """Run KMeans clustering and return model + labels."""
    kmeans = KMeans(n_clusters=n_clusters, init=init, n_init=n_init, random_state=random_state)
    labels = kmeans.fit_predict(X)
    return kmeans, labels


def run_minibatch_kmeans(X, n_clusters=20, batch_size=1000, random_state=42):
    """Run Mini-Batch KMeans (faster for large datasets)."""
    mb = MiniBatchKMeans(n_clusters=n_clusters, batch_size=batch_size, random_state=random_state)
    labels = mb.fit_predict(X)
    return mb, labels


def evaluate_clustering(X, pred_labels, true_labels=None):
    """Evaluate clustering quality."""
    results = {"silhouette": silhouette_score(X, pred_labels, sample_size=min(5000, len(X)))}
    if true_labels is not None:
        results["ari"] = adjusted_rand_score(true_labels, pred_labels)
        results["nmi"] = normalized_mutual_info_score(true_labels, pred_labels)
    return results


def elbow_plot(X, k_range=range(5, 50, 5), save_path=None):
    """Plot elbow method for optimal k."""
    inertias = []
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=5, random_state=42)
        km.fit(X)
        inertias.append(km.inertia_)
    plt.figure(figsize=(10, 5))
    plt.plot(list(k_range), inertias, "bx-")
    plt.xlabel("k")
    plt.ylabel("Inertia")
    plt.title("Elbow Method for Optimal k")
    plt.grid(True, alpha=0.3)
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
