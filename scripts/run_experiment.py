"""
Main experiment entrypoint. Run classifier experiments.
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
    parser.add_argument("--reducer", default="none", help="Dimensionality reducer")
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--cache-dir", default="data/cache")
    parser.add_argument("--results-dir", default="results/metrics")
    args = parser.parse_args()

    seed_everything(42)
    run_experiment(
        feature_type=args.features,
        model_name=args.model,
        reducer_name=args.reducer,
        cache_dir=args.cache_dir,
        results_dir=args.results_dir,
        cv_folds=args.cv_folds,
    )


if __name__ == "__main__":
    main()
