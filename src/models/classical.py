"""
Classical ML model utilities.
Covers: Course Topics #4-#10, #14-#16, #25-#27.
"""
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, Perceptron, SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier


def get_all_classical_models():
    """Return dict of all classical models with default params."""
    return {
        "Naive Bayes": GaussianNB(),
        "kNN (k=5, euclidean)": KNeighborsClassifier(n_neighbors=5, metric="euclidean"),
        "kNN (k=5, manhattan)": KNeighborsClassifier(n_neighbors=5, metric="manhattan"),
        "kNN (k=5, cosine)": KNeighborsClassifier(n_neighbors=5, metric="cosine"),
        "Weighted kNN (k=7)": KNeighborsClassifier(n_neighbors=7, weights="distance"),
        "SGD Classifier": SGDClassifier(loss="hinge", max_iter=1000, random_state=42),
        "Logistic Regression": LogisticRegression(
            multi_class="multinomial", solver="lbfgs", max_iter=2000, C=1.0, random_state=42
        ),
        "Logistic (L1, C=0.01)": LogisticRegression(
            penalty="l1", C=0.01, solver="saga", max_iter=2000, random_state=42
        ),
        "Logistic (L2, C=0.01)": LogisticRegression(
            penalty="l2", C=0.01, solver="lbfgs", max_iter=2000, random_state=42
        ),
        "Perceptron": Perceptron(max_iter=1000, random_state=42),
        "Linear SVM": SVC(kernel="linear", C=1.0, probability=True, random_state=42),
        "RBF SVM": SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, random_state=42),
        "Poly SVM": SVC(kernel="poly", C=1.0, gamma="scale", probability=True, random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=20, min_samples_split=5, random_state=42),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=30, n_jobs=-1, random_state=42
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42
        ),
    }
