"""File I/O utilities."""
import os
import json
import numpy as np
import pandas as pd


def save_results_csv(results, filepath):
    """Save results dict to CSV."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df = pd.DataFrame([results])
    if os.path.exists(filepath):
        existing = pd.read_csv(filepath)
        df = pd.concat([existing, df], ignore_index=True)
    df.to_csv(filepath, index=False)


def save_json(data, filepath):
    """Save dict as JSON."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2, default=str)


def ensure_dir(path):
    """Create directory if it does not exist."""
    os.makedirs(path, exist_ok=True)
