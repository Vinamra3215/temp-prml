"""
Precompute and cache all feature matrices. Run this once before experiments.
"""
import sys
sys.path.insert(0, ".")

from src.utils.seed import seed_everything
from src.data.dataset import Food101Dataset
from src.data.cache import save_features, cache_exists
from src.features.histogram import ColorHistogramExtractor
from src.features.hog import HOGExtractor
from src.features.lbp import LBPExtractor
from src.features.glcm import GLCMExtractor
from src.features.fusion import FusedFeatureExtractor


def main():
    seed_everything(42)
    cache_dir = "data/cache"
    dataset = Food101Dataset(root="data/", n_classes=20, seed=42)
    (train_paths, train_labels), (val_paths, val_labels), (test_paths, test_labels) = dataset.get_splits()

    extractors = {
        "histogram": ColorHistogramExtractor(bins=32),
        "hog": HOGExtractor(),
        "lbp": LBPExtractor(),
        "glcm": GLCMExtractor(),
        "fused": FusedFeatureExtractor(),
    }

    # Add CNN (works on CPU too, just slower)
    try:
        from src.features.cnn_embeddings import CNNEmbeddingExtractor
        extractors["cnn"] = CNNEmbeddingExtractor(backbone="resnet50", device="cpu")
        print("CNN extractor loaded (CPU mode)")
    except (ImportError, OSError) as e:
        print(f"Skipping CNN extraction: {e}")

    splits = {
        "train": (train_paths, train_labels),
        "val": (val_paths, val_labels),
        "test": (test_paths, test_labels),
    }

    for feat_name, extractor in extractors.items():
        for split_name, (paths, labels) in splits.items():
            if cache_exists(cache_dir, feat_name, split_name):
                print(f"[SKIP] {feat_name}/{split_name} already cached.")
                continue

            print(f"\n[EXTRACT] {feat_name} / {split_name} ({len(paths)} images)")
            X, y = extractor.extract_dataset(paths, labels, n_jobs=1)
            save_features(X, y, cache_dir, feat_name, split_name, dataset.class_names)

    print("\nAll feature extraction complete!")


if __name__ == "__main__":
    main()
