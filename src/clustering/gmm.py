"""
Gaussian Mixture Model clustering (EM algorithm).
Covers: Course Topic #23 - KMeans (EM) and variants.
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture


def run_gmm(X, n_components=20, covariance_type="diag", random_state=42):
    """Run GMM clustering."""
    gmm = GaussianMixture(
        n_components=n_components, covariance_type=covariance_type,
        random_state=random_state, max_iter=200,
    )
    labels = gmm.fit_predict(X)
    return gmm, labels


def bic_aic_sweep(X, k_range=range(5, 50, 5), covariance_type="diag", save_path=None):
    """Sweep over k values and plot BIC/AIC for optimal selection."""
    bics, aics = [], []
    for k in k_range:
        gmm = GaussianMixture(n_components=k, covariance_type=covariance_type, random_state=42)
        gmm.fit(X)
        bics.append(gmm.bic(X))
        aics.append(gmm.aic(X))

    plt.figure(figsize=(10, 5))
    plt.plot(list(k_range), bics, "b-o", label="BIC")
    plt.plot(list(k_range), aics, "r-s", label="AIC")
    plt.xlabel("Number of Components")
    plt.ylabel("Score")
    plt.title("GMM Model Selection: BIC vs AIC")
    plt.legend()
    plt.grid(True, alpha=0.3)
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    return bics, aics
