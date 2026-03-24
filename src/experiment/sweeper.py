"""
Optuna hyperparameter sweep wrapper.
"""
import optuna
from sklearn.model_selection import cross_val_score
from src.models.registry import build_pipeline
from src.data.cache import load_features


def create_objective(feature_type, model_name, cache_dir="data/cache", cv_folds=5):
    """Create Optuna objective function."""
    X_train, y_train = load_features(cache_dir, feature_type, "train")

    def objective(trial):
        if model_name == "knn":
            params = {
                "n_neighbors": trial.suggest_int("n_neighbors", 1, 21),
                "weights": trial.suggest_categorical("weights", ["uniform", "distance"]),
                "metric": trial.suggest_categorical("metric", ["euclidean", "manhattan", "cosine"]),
            }
        elif model_name == "svm_rbf":
            params = {
                "C": trial.suggest_float("C", 0.01, 100, log=True),
                "gamma": trial.suggest_categorical("gamma", ["scale", "auto"]),
                "probability": True,
            }
        elif model_name == "mlp_sklearn":
            n_layers = trial.suggest_int("n_layers", 1, 3)
            units = trial.suggest_categorical("units", [64, 128, 256, 512])
            params = {
                "hidden_layer_sizes": tuple([units] * n_layers),
                "max_iter": 500,
                "early_stopping": True,
            }
        else:
            params = {}

        pipeline = build_pipeline(model_name, params)
        scores = cross_val_score(pipeline, X_train, y_train, cv=cv_folds, scoring="f1_macro")
        return scores.mean()

    return objective, X_train, y_train


def run_sweep(feature_type, model_name, n_trials=30, cache_dir="data/cache"):
    """Run Optuna hyperparameter sweep."""
    objective, _, _ = create_objective(feature_type, model_name, cache_dir)
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f"\nBest trial: {study.best_trial.value:.4f}")
    print(f"Best params: {study.best_trial.params}")
    return study
