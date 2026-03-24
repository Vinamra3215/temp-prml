"""
Model registry with unified Pipeline interface.
Covers: Course Topics #4-#10, #14-#16, #25-#27.
"""
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, Perceptron, SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


MODEL_REGISTRY = {
    "knn": KNeighborsClassifier,
    "logistic": LogisticRegression,
    "naive_bayes": GaussianNB,
    "svm_linear": lambda **kw: SVC(kernel="linear", **kw),
    "svm_rbf": lambda **kw: SVC(kernel="rbf", **kw),
    "decision_tree": DecisionTreeClassifier,
    "random_forest": RandomForestClassifier,
    "mlp_sklearn": MLPClassifier,
    "perceptron": Perceptron,
    "sgd": SGDClassifier,
    "gradient_boosting": GradientBoostingClassifier,
}

REDUCER_REGISTRY = {
    "pca": PCA,
    "lda": LinearDiscriminantAnalysis,
    "none": None,
}


def build_pipeline(model_name, model_params=None, reducer_name="none", reducer_params=None):
    """
    Build: StandardScaler -> [PCA/LDA/None] -> Classifier
    All fit() calls are safe for cross-validation -- no data leakage.
    """
    model_params = model_params or {}
    reducer_params = reducer_params or {}

    steps = [("scaler", StandardScaler())]

    if reducer_name != "none" and reducer_name in REDUCER_REGISTRY:
        reducer_cls = REDUCER_REGISTRY[reducer_name]
        if reducer_cls is not None:
            steps.append(("reducer", reducer_cls(**reducer_params)))

    model_cls = MODEL_REGISTRY[model_name]
    if callable(model_cls) and not isinstance(model_cls, type):
        model = model_cls(**model_params)
    else:
        model = model_cls(**model_params)
    steps.append(("clf", model))

    return Pipeline(steps)
