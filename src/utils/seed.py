"""Global seed setter for reproducibility."""
import random
import numpy as np


def seed_everything(seed=42):
    """Set seed for all random number generators."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except (ImportError, OSError):
        pass  # torch not available or DLL issue — skip