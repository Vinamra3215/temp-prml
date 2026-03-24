"""
LDA dimensionality reduction wrapper.
Covers: Course Topic #13.
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


class LDAReducer:
    """Wrapped sklearn LDA with visualization."""

    def __init__(self, n_components=19):
        self.lda = LinearDiscriminantAnalysis(n_components=n_components)
        self.is_fitted = False

    def fit(self, X, y):
        self.lda.fit(X, y)
        self.is_fitted = True
        return self

    def transform(self, X):
        return self.lda.transform(X)

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)

    def plot_2d(self, X, y, class_names=None, save_path=None):
        """2D LDA projection scatter plot."""
        lda_2d = LinearDiscriminantAnalysis(n_components=2)
        X_2d = lda_2d.fit_transform(X, y)
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y, cmap="tab20", alpha=0.5, s=5)
        plt.colorbar(scatter)
        plt.title("LDA 2D Projection")
        plt.xlabel("LD1")
        plt.ylabel("LD2")
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
