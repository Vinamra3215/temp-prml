"""
Model registry with unified Pipeline interface + GridSearchCV parameter grids.

All hyperparameters are selected via GridSearchCV (no hardcoding).
Covers: Course Topics #4-#10, #17-#19, #22, #25-#26.
"""
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, Perceptron, SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from src.models.classical import KDEClassifier


# ── Model Registry ───────────────────────────────────────────────────

MODEL_REGISTRY = {
    "knn":               KNeighborsClassifier,
    "logistic":          LogisticRegression,
    "naive_bayes":       GaussianNB,
    "decision_tree":     DecisionTreeClassifier,
    "mlp_sklearn":       MLPClassifier,
    "perceptron":        Perceptron,
    "sgd":               SGDClassifier,
    "gradient_boosting": GradientBoostingClassifier,
    "kde":               KDEClassifier,
}

REDUCER_REGISTRY = {
    "pca": PCA,
    "lda": LinearDiscriminantAnalysis,
    "none": None,
}


# ── GridSearchCV Parameter Grids ─────────────────────────────────────
# These are the search spaces for hyperparameter tuning.
# GridSearchCV evaluates ALL combinations via cross-validation.

PARAM_GRIDS = {
    "knn": {
        "clf__n_neighbors": [1, 3, 5, 7, 9, 11, 15, 21],
        "clf__weights": ["uniform", "distance"],
        "clf__metric": ["euclidean", "manhattan"],
    },
    "logistic": {
        "clf__C": [0.01, 0.1, 1.0, 10.0],
        "clf__solver": ["lbfgs"],
        "clf__max_iter": [2000],
        "clf__multi_class": ["multinomial"],
    },
    "naive_bayes": {
        "clf__var_smoothing": [1e-9, 1e-8, 1e-7, 1e-6],
    },
    "decision_tree": {
        "clf__max_depth": [5, 10, 20, 30, None],
        "clf__min_samples_split": [2, 5, 10],
        "clf__min_samples_leaf": [1, 2, 5],
    },
    "gradient_boosting": {
        "clf__n_estimators": [50, 100, 200],
        "clf__max_depth": [3, 5, 7],
        "clf__learning_rate": [0.01, 0.1, 0.2],
    },
    "mlp_sklearn": {
        "clf__hidden_layer_sizes": [(128,), (256, 128), (512, 256, 128)],
        "clf__activation": ["relu", "tanh"],
        "clf__learning_rate_init": [0.001, 0.01],
    },
    "perceptron": {
        "clf__penalty": [None, "l2", "l1"],
        "clf__alpha": [0.0001, 0.001, 0.01],
        "clf__max_iter": [500, 1000],
    },
    "sgd": {
        "clf__loss": ["hinge", "modified_huber"],
        "clf__alpha": [0.0001, 0.001, 0.01],
        "clf__max_iter": [1000],
    },
    "kde": {
        "clf__bandwidth": [0.1, 0.5, 1.0, 2.0, 5.0],
        "clf__kernel": ["gaussian", "tophat", "epanechnikov"],
    },
}


def build_pipeline(model_name, model_params=None, reducer_name="none", reducer_params=None):
    """
    Build: StandardScaler -> [PCA/LDA/None] -> Classifier

    Args:
        model_name: Key from MODEL_REGISTRY
        model_params: Dict of model hyperparameters (overrides defaults)
        reducer_name: "none", "pca", or "lda"
        reducer_params: Dict of reducer params (e.g. {"n_components": 200})

    Returns:
        sklearn Pipeline (safe for cross-validation — no data leakage)
    """
    model_params = model_params or {}
    reducer_params = reducer_params or {}

    steps = [("scaler", StandardScaler())]

    if reducer_name != "none" and reducer_name in REDUCER_REGISTRY:
        reducer_cls = REDUCER_REGISTRY[reducer_name]
        if reducer_cls is not None:
            steps.append(("reducer", reducer_cls(**reducer_params)))

    model_cls = MODEL_REGISTRY[model_name]
    model = model_cls(**model_params)
    steps.append(("clf", model))

    return Pipeline(steps)


def get_param_grid(model_name):
    """Get GridSearchCV parameter grid for a model."""
    return PARAM_GRIDS.get(model_name, {})
