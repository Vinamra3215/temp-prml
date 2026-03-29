"""
Run a single classifier experiment with optional W&B tracking.

Usage:
    python scripts/run_experiment.py --model svm_rbf --features cnn
    python scripts/run_experiment.py --model logistic --features cnn --reducer pca --wandb
"""
import sys
sys.path.insert(0, ".")
import argparse
from src.utils.seed import seed_everything
from src.experiment.runner import run_experiment


def main():
    parser = argparse.ArgumentParser(description="Run classification experiment")
    parser.add_argument("--model", default="svm_rbf", help="Model name from registry")
    parser.add_argument("--features", default="cnn", help="Feature type")
    parser.add_argument("--reducer", default="none", help="Reducer: none, pca, lda")
    parser.add_argument("--pca-components", type=int, default=100, help="PCA components if reducer=pca")
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--cache-dir", default="data/cache")
    parser.add_argument("--results-dir", default="results/metrics")
    parser.add_argument("--wandb", action="store_true", help="Enable W&B tracking")
    args = parser.parse_args()

    seed_everything(42)

    reducer_params = None
    if args.reducer == "pca":
        reducer_params = {"n_components": args.pca_components}

    run_experiment(
        feature_type=args.features,
        model_name=args.model,
        reducer_name=args.reducer,
        reducer_params=reducer_params,
        cache_dir=args.cache_dir,
        results_dir=args.results_dir,
        cv_folds=args.cv_folds,
        use_wandb=args.wandb,
    )


if __name__ == "__main__":
    main()
