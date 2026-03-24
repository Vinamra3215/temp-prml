"""
Launch Optuna hyperparameter sweep.
"""
import sys
sys.path.insert(0, ".")
import argparse
from src.utils.seed import seed_everything
from src.experiment.sweeper import run_sweep


def main():
    parser = argparse.ArgumentParser(description="Run hyperparameter sweep")
    parser.add_argument("--model", default="svm_rbf")
    parser.add_argument("--features", default="cnn")
    parser.add_argument("--n-trials", type=int, default=30)
    parser.add_argument("--cache-dir", default="data/cache")
    args = parser.parse_args()

    seed_everything(42)
    study = run_sweep(args.features, args.model, args.n_trials, args.cache_dir)
    print(f"\nBest value: {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")


if __name__ == "__main__":
    main()
