"""
Agglomerative (hierarchical) clustering.
Covers: Course Topic #24.
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage


def run_agglomerative(X, n_clusters=20, linkage_type="ward"):
    """Run agglomerative clustering."""
    agg = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage_type)
    labels = agg.fit_predict(X)
    return agg, labels


def plot_dendrogram(X, method="ward", max_samples=500, truncate_p=30, save_path=None):
    """Plot hierarchical clustering dendrogram."""
    if len(X) > max_samples:
        indices = np.random.RandomState(42).choice(len(X), max_samples, replace=False)
        X_sub = X[indices]
    else:
        X_sub = X

    Z = linkage(X_sub, method=method)
    plt.figure(figsize=(15, 7))
    dendrogram(Z, truncate_mode="lastp", p=truncate_p)
    plt.title("Hierarchical Clustering Dendrogram")
    plt.xlabel("Cluster")
    plt.ylabel("Distance")
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
