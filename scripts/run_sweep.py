"""
Run Optuna hyperparameter sweep with W&B tracking.

Uses Bayesian optimization (TPE sampler) to efficiently search
hyperparameter space instead of exhaustive grid search.

Usage:
    python scripts/run_sweep.py --model svm_rbf --features cnn --n-trials 30
    python scripts/run_sweep.py --model knn --features cnn --wandb
    python scripts/run_sweep.py --model logistic --features cnn --reducer pca --n-trials 50
"""
import sys
sys.path.insert(0, ".")
import argparse
from src.utils.seed import seed_everything
from src.experiment.sweeper import run_sweep


def main():
    parser = argparse.ArgumentParser(description="Run Optuna hyperparameter sweep")
    parser.add_argument("--model", default="svm_rbf")
    parser.add_argument("--features", default="cnn")
    parser.add_argument("--n-trials", type=int, default=30)
    parser.add_argument("--cache-dir", default="data/cache")
    parser.add_argument("--results-dir", default="results/metrics")
    parser.add_argument("--reducer", default="none", help="none or pca")
    parser.add_argument("--pca-components", type=int, default=100)
    parser.add_argument("--wandb", action="store_true", help="Enable W&B tracking")
    args = parser.parse_args()

    seed_everything(42)

    reducer_params = None
    if args.reducer == "pca":
        reducer_params = {"n_components": args.pca_components}

    study = run_sweep(
        feature_type=args.features,
        model_name=args.model,
        n_trials=args.n_trials,
        cache_dir=args.cache_dir,
        cv_folds=5,
        reducer_name=args.reducer,
        reducer_params=reducer_params,
        use_wandb=args.wandb,
        results_dir=args.results_dir,
    )

    print(f"\n{'=' * 60}")
    print(f"  SWEEP COMPLETE: {args.model} on {args.features}")
    print(f"{'=' * 60}")
    print(f"  Best F1 (macro): {study.best_value:.4f}")
    print(f"  Best params:")
    for k, v in study.best_params.items():
        print(f"    {k}: {v}")
    print(f"\n  Use these params in run_experiment.py for final evaluation.")


if __name__ == "__main__":
    main()
