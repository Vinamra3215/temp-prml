"""
Food-101 dataset handler using Kaggle folder structure.
Loads from: data/images/<class_name>/<image_id>.jpg
Splits from: data/meta/meta/train.json, test.json
"""
import os
import json
import random
from pathlib import Path

import numpy as np


class Food101Dataset:
    """
    Handles loading, stratified splitting, and class subsetting
    from the Kaggle Food-101 folder layout.
    """

    def __init__(
        self,
        root: str = "data/",
        n_classes: int = 20,
        val_split: float = 0.15,
        seed: int = 42,
    ):
        self.root = root
        self.n_classes = n_classes
        self.val_split = val_split
        self.seed = seed

        self.images_dir = os.path.join(root, "images")
        self.meta_dir = os.path.join(root, "meta", "meta")

        self.class_names, self.class_to_idx = self._load_classes()
        self.train_paths, self.train_labels = self._load_split("train")
        self.test_paths, self.test_labels = self._load_split("test")

    def _load_classes(self):
        """Load and subset class names."""
        classes_file = os.path.join(self.meta_dir, "classes.txt")
        with open(classes_file, "r") as f:
            all_classes = sorted([line.strip() for line in f if line.strip()])

        # Deterministically select n_classes
        random.seed(self.seed)
        selected = sorted(random.sample(all_classes, min(self.n_classes, len(all_classes))))
        class_to_idx = {c: i for i, c in enumerate(selected)}
        return selected, class_to_idx

    def _load_split(self, split: str):
        """Load image paths and labels for a split (train or test)."""
        json_path = os.path.join(self.meta_dir, f"{split}.json")
        with open(json_path, "r") as f:
            data = json.load(f)

        paths = []
        labels = []
        for class_name, image_ids in data.items():
            if class_name not in self.class_to_idx:
                continue
            label = self.class_to_idx[class_name]
            for img_id in image_ids:
                img_path = os.path.join(self.images_dir, f"{img_id}.jpg")
                if os.path.exists(img_path):
                    paths.append(img_path)
                    labels.append(label)

        return paths, np.array(labels)

    def get_splits(self):
        """
        Returns stratified (train, val, test) splits.
        Train/test come from meta JSON; val is carved from train.
        Returns: (train_paths, train_labels), (val_paths, val_labels), (test_paths, test_labels)
        """
        from sklearn.model_selection import train_test_split

        paths = np.array(self.train_paths)
        labels = self.train_labels

        X_train, X_val, y_train, y_val = train_test_split(
            paths, labels,
            test_size=self.val_split,
            random_state=self.seed,
            stratify=labels,
        )

        X_test = np.array(self.test_paths)
        y_test = self.test_labels

        print(f"Dataset loaded: {len(self.class_names)} classes")
        print(f"  Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

        return (X_train.tolist(), y_train), (X_val.tolist(), y_val), (X_test.tolist(), y_test)

    def __len__(self):
        return len(self.train_paths) + len(self.test_paths)

    def __repr__(self):
        return (
            f"Food101Dataset(n_classes={self.n_classes}, "
            f"train={len(self.train_paths)}, test={len(self.test_paths)}, "
            f"classes={self.class_names[:5]}...)"
        )
