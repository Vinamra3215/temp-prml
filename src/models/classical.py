"""
Classical ML model utilities.
Covers: Course Topics #2, #4-#10, #17-#19, #22, #25-#26.

Models:  KNN, Logistic, Naive Bayes, Decision Tree, Gradient Boosting,
         Perceptron, MLP, KDE Classifier (Parzen Window)
Removed: SVM RBF, Random Forest (per project scope)
"""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neighbors import KNeighborsClassifier, KernelDensity
from sklearn.linear_model import LogisticRegression, Perceptron, SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier


# ── KDE / Parzen Window Classifier ──────────────────────────────────

class KDEClassifier(BaseEstimator, ClassifierMixin):
    """
    Kernel Density Estimation (Parzen Window) Classifier.

    Implements a generative Bayesian classifier using KDE:
      1. Fit one KDE per class on training data
      2. At prediction time, compute log-likelihood under each class KDE
      3. Apply Bayes rule: P(class|x) ∝ P(x|class) * P(class)

    This directly implements Parzen Window estimation (Course Topic #2).

    Parameters:
        bandwidth: KDE bandwidth (smoothing parameter)
        kernel: Kernel function ("gaussian", "tophat", "epanechnikov", etc.)
    """

    def __init__(self, bandwidth=1.0, kernel="gaussian"):
        self.bandwidth = bandwidth
        self.kernel = kernel

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.models_ = {}
        self.priors_ = {}
        n_total = len(y)

        for cls in self.classes_:
            X_cls = X[y == cls]
            self.priors_[cls] = len(X_cls) / n_total
            kde = KernelDensity(bandwidth=self.bandwidth, kernel=self.kernel)
            kde.fit(X_cls)
            self.models_[cls] = kde

        return self

    def predict(self, X):
        log_probs = self._compute_log_probs(X)
        return self.classes_[np.argmax(log_probs, axis=1)]

    def predict_proba(self, X):
        log_probs = self._compute_log_probs(X)
        # Convert log-probs to probabilities via softmax
        log_probs -= log_probs.max(axis=1, keepdims=True)
        probs = np.exp(log_probs)
        probs /= probs.sum(axis=1, keepdims=True)
        return probs

    def _compute_log_probs(self, X):
        log_probs = np.zeros((len(X), len(self.classes_)))
        for i, cls in enumerate(self.classes_):
            log_likelihood = self.models_[cls].score_samples(X)
            log_prior = np.log(self.priors_[cls])
            log_probs[:, i] = log_likelihood + log_prior
        return log_probs


def get_all_classical_models():
    """Return dict of all classical models with default params (for quick comparison)."""
    return {
        "Naive Bayes": GaussianNB(),
        "kNN (k=5, euclidean)": KNeighborsClassifier(n_neighbors=5, metric="euclidean"),
        "kNN (k=5, manhattan)": KNeighborsClassifier(n_neighbors=5, metric="manhattan"),
        "Weighted kNN (k=7)": KNeighborsClassifier(n_neighbors=7, weights="distance"),
        "SGD Classifier": SGDClassifier(loss="hinge", max_iter=1000, random_state=42),
        "Logistic Regression": LogisticRegression(
            multi_class="multinomial", solver="lbfgs", max_iter=2000, random_state=42
        ),
        "Perceptron": Perceptron(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=20, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42
        ),
        "KDE (Gaussian)": KDEClassifier(bandwidth=1.0, kernel="gaussian"),
        "KDE (Tophat / Parzen)": KDEClassifier(bandwidth=1.0, kernel="tophat"),
    }
