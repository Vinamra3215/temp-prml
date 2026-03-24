"""Logging and wandb initialization."""
import logging
import os


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


def init_wandb(project="food-prml", config=None, run_name=None):
    """Initialize wandb run."""
    try:
        import wandb
        wandb.init(project=project, config=config, name=run_name)
        return True
    except ImportError:
        print("wandb not installed. Skipping experiment tracking.")
        return False
