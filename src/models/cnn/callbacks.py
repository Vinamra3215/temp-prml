"""
CNN training callbacks.
"""
import pytorch_lightning as pl


class EarlyStoppingCallback(pl.Callback):
    """Simple early stopping based on validation loss."""

    def __init__(self, patience=7, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter = 0

    def on_validation_end(self, trainer, pl_module):
        val_loss = trainer.callback_metrics.get("val/loss", float("inf"))
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                trainer.should_stop = True
