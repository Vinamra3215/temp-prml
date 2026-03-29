"""
Logging, experiment tracking, and W&B integration.
Provides unified interface for local CSV logging + optional W&B cloud tracking.
"""
import logging
import os
import json
import time
from datetime import datetime


def get_logger(name, level=logging.INFO):
    """Get a configured logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("[%(asctime)s] %(name)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


# ── W&B Integration ──────────────────────────────────────────────────

_wandb_enabled = False


def init_wandb(project="prml-food-classification", config=None, run_name=None, tags=None):
    """
    Initialize W&B run. Gracefully skips if wandb not installed.

    Args:
        project: W&B project name
        config: Dict of hyperparameters to log
        run_name: Human-readable run name (e.g. "svm_rbf_cnn_none")
        tags: List of tags (e.g. ["cnn", "svm", "no-pca"])

    Returns:
        True if W&B initialized, False otherwise
    """
    global _wandb_enabled
    try:
        import wandb
        wandb.init(
            project=project,
            config=config or {},
            name=run_name,
            tags=tags or [],
            reinit=True,
        )
        _wandb_enabled = True
        return True
    except ImportError:
        print("W&B not installed. Tracking locally only. Install: pip install wandb")
        _wandb_enabled = False
        return False
    except Exception as e:
        print(f"W&B init failed: {e}. Tracking locally only.")
        _wandb_enabled = False
        return False


def log_wandb(metrics_dict):
    """Log metrics to W&B if enabled."""
    global _wandb_enabled
    if not _wandb_enabled:
        return
    try:
        import wandb
        wandb.log(metrics_dict)
    except Exception:
        pass


def finish_wandb():
    """Finish current W&B run."""
    global _wandb_enabled
    if not _wandb_enabled:
        return
    try:
        import wandb
        wandb.finish()
    except Exception:
        pass


def is_wandb_enabled():
    """Check if W&B is currently active."""
    return _wandb_enabled


# ── Local Experiment Log ─────────────────────────────────────────────

def log_experiment_csv(results_dir, run_data):
    """
    Append experiment to local experiment_log.csv with full tracking info.
    This is the detailed log (more fields than master.csv).
    """
    import pandas as pd

    log_path = os.path.join(results_dir, "experiment_log.csv")
    os.makedirs(results_dir, exist_ok=True)

    # Add metadata
    run_data["timestamp"] = datetime.now().isoformat()
    run_data["params_json"] = json.dumps(run_data.get("params", {}))

    df_new = pd.DataFrame([run_data])
    if os.path.exists(log_path):
        df_existing = pd.read_csv(log_path)
        df = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df = df_new
    df.to_csv(log_path, index=False)


# ── Timing Wrapper ───────────────────────────────────────────────────

class Timer:
    """Context manager for timing code blocks."""

    def __init__(self, label=""):
        self.label = label
        self.elapsed = 0

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.elapsed = time.time() - self.start
        if self.label:
            print(f"  [{self.label}] {self.elapsed:.1f}s")
