"""
Run clustering experiments.
"""
import sys
sys.path.insert(0, ".")
import argparse
from src.utils.seed import seed_everything
from src.experiment.clustering_runner import run_clustering_experiments


def main():
    parser = argparse.ArgumentParser(description="Run clustering experiments")
    parser.add_argument("--features", default="cnn")
    parser.add_argument("--cache-dir", default="data/cache")
    parser.add_argument("--plots-dir", default="results/plots")
    args = parser.parse_args()

    seed_everything(42)
    results = run_clustering_experiments(args.features, args.cache_dir, args.plots_dir)

    print("\nClustering Results Summary:")
    for algo, metrics in results.items():
        print(f"  {algo}: {metrics}")


if __name__ == "__main__":
    main()
