"""
Optuna hyperparameter sweep with W&B tracking.
Uses Bayesian optimization (TPE sampler) for efficient search.
"""
import json
import os
import optuna
from sklearn.model_selection import cross_val_score
from src.models.registry import build_pipeline
from src.data.cache import load_features
from src.utils.logging import init_wandb, log_wandb, finish_wandb


# ── Search Spaces ────────────────────────────────────────────────────

def _knn_params(trial):
    return {
        "n_neighbors": trial.suggest_int("n_neighbors", 1, 25, step=2),
        "weights": trial.suggest_categorical("weights", ["uniform", "distance"]),
        "metric": trial.suggest_categorical("metric", ["euclidean", "manhattan", "cosine"]),
    }


def _kde_params(trial):
    return {
        "bandwidth": trial.suggest_float("bandwidth", 0.05, 10.0, log=True),
        "kernel": trial.suggest_categorical("kernel", ["gaussian", "tophat", "epanechnikov"]),
    }


def _logistic_params(trial):
    return {
        "C": trial.suggest_float("C", 1e-3, 1e2, log=True),
        "multi_class": "multinomial",
        "solver": "lbfgs",
        "max_iter": 2000,
        "random_state": 42,
    }


def _decision_tree_params(trial):
    return {
        "max_depth": trial.suggest_int("max_depth", 3, 50),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        "random_state": 42,
    }




def _gradient_boosting_params(trial):
    return {
        "n_estimators": trial.suggest_int("n_estimators", 30, 200, step=10),
        "max_depth": trial.suggest_int("max_depth", 2, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "random_state": 42,
    }


def _mlp_params(trial):
    n_layers = trial.suggest_int("n_layers", 1, 3)
    units = trial.suggest_categorical("units", [64, 128, 256, 512])
    return {
        "hidden_layer_sizes": tuple([units] * n_layers),
        "activation": trial.suggest_categorical("activation", ["relu", "tanh"]),
        "solver": "adam",
        "max_iter": 500,
        "early_stopping": True,
        "random_state": 42,
    }


SEARCH_SPACES = {
    "knn": _knn_params,
    "logistic": _logistic_params,
    "decision_tree": _decision_tree_params,
    "gradient_boosting": _gradient_boosting_params,
    "mlp_sklearn": _mlp_params,
    "kde": _kde_params,
}


# ── Objective ────────────────────────────────────────────────────────

def create_objective(feature_type, model_name, cache_dir="data/cache",
                     cv_folds=5, reducer_name="none", reducer_params=None,
                     use_wandb=False):
    """Create Optuna objective function with optional W&B logging per trial."""
    X_train, y_train = load_features(cache_dir, feature_type, "train")

    param_fn = SEARCH_SPACES.get(model_name)
    if param_fn is None:
        raise ValueError(f"No search space defined for '{model_name}'. "
                         f"Available: {list(SEARCH_SPACES.keys())}")

    def objective(trial):
        params = param_fn(trial)
        pipeline = build_pipeline(model_name, params, reducer_name, reducer_params)
        scores = cross_val_score(pipeline, X_train, y_train,
                                 cv=cv_folds, scoring="f1_macro")
        mean_f1 = scores.mean()

        if use_wandb:
            log_wandb({
                "trial_number": trial.number,
                "trial_f1": mean_f1,
                "trial_f1_std": scores.std(),
                **{f"param_{k}": v for k, v in params.items()
                   if not isinstance(v, (tuple, list))},
            })

        return mean_f1

    return objective, X_train, y_train


# ── Sweep Runner ─────────────────────────────────────────────────────

def run_sweep(feature_type, model_name, n_trials=30, cache_dir="data/cache",
              cv_folds=5, reducer_name="none", reducer_params=None,
              use_wandb=False, results_dir="results/metrics"):
    """
    Run Optuna hyperparameter sweep.

    Args:
        feature_type: Feature type key
        model_name: Model name from registry
        n_trials: Number of Optuna trials
        cache_dir: Feature cache directory
        cv_folds: CV folds
        reducer_name: "none" or "pca"
        reducer_params: PCA params if applicable
        use_wandb: Enable W&B tracking
        results_dir: Where to save sweep results

    Returns:
        optuna.Study object
    """
    print(f"\n{'=' * 60}")
    print(f"Optuna Sweep: {model_name} | {feature_type} | {n_trials} trials")
    print(f"{'=' * 60}")

    # W&B for sweep
    if use_wandb:
        init_wandb(
            config={"model": model_name, "features": feature_type,
                    "n_trials": n_trials, "reducer": reducer_name},
            run_name=f"sweep_{model_name}_{feature_type}",
            tags=["sweep", model_name, feature_type],
        )

    objective, _, _ = create_objective(
        feature_type, model_name, cache_dir, cv_folds,
        reducer_name, reducer_params, use_wandb,
    )

    # Bayesian optimization (TPE sampler)
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        study_name=f"{model_name}_{feature_type}",
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    # Print results
    print(f"\nBest trial #{study.best_trial.number}:")
    print(f"  F1 (macro): {study.best_trial.value:.4f}")
    print(f"  Best params: {study.best_trial.params}")

    # Log best to W&B
    if use_wandb:
        log_wandb({
            "best_f1": study.best_trial.value,
            "best_trial": study.best_trial.number,
            **{f"best_{k}": v for k, v in study.best_trial.params.items()},
        })
        finish_wandb()

    # Save sweep results locally
    os.makedirs(results_dir, exist_ok=True)
    sweep_file = os.path.join(results_dir, f"sweep_{model_name}_{feature_type}.json")
    sweep_data = {
        "model": model_name,
        "feature": feature_type,
        "reducer": reducer_name,
        "n_trials": n_trials,
        "best_f1": round(study.best_trial.value, 4),
        "best_params": study.best_trial.params,
        "all_trials": [
            {"number": t.number, "value": round(t.value, 4), "params": t.params}
            for t in study.trials
        ],
    }
    with open(sweep_file, "w") as f:
        json.dump(sweep_data, f, indent=2, default=str)
    print(f"  Saved to {sweep_file}")

    return study
