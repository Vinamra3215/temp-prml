"""
PCA dimensionality reduction wrapper.
Covers: Course Topics #12, #13.
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


class PCAReducer:
    """Wrapped sklearn PCA with variance analysis plots."""

    def __init__(self, n_components=100, whiten=True):
        self.pca = PCA(n_components=n_components, whiten=whiten)
        self.is_fitted = False

    def fit(self, X):
        self.pca.fit(X)
        self.is_fitted = True
        return self

    def transform(self, X):
        return self.pca.transform(X)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def plot_explained_variance(self, save_path=None):
        """Plot cumulative explained variance (scree plot)."""
        if not self.is_fitted:
            raise RuntimeError("PCA not fitted yet.")
        cumvar = np.cumsum(self.pca.explained_variance_ratio_)
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, len(cumvar) + 1), cumvar, "b-o", markersize=3)
        plt.xlabel("Number of Components")
        plt.ylabel("Cumulative Explained Variance")
        plt.title("PCA Explained Variance")
        plt.grid(True, alpha=0.3)
        plt.axhline(y=0.95, color="r", linestyle="--", label="95% threshold")
        plt.legend()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

    def plot_2d(self, X, y, class_names=None, save_path=None):
        """2D PCA projection scatter plot."""
        pca_2d = PCA(n_components=2)
        X_2d = pca_2d.fit_transform(X)
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y, cmap="tab20", alpha=0.5, s=5)
        plt.colorbar(scatter)
        plt.title("PCA 2D Projection")
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
