"""
CNN fine-tuning with PyTorch Lightning.
"""
import sys
sys.path.insert(0, ".")
import argparse
import pytorch_lightning as pl
from src.utils.seed import seed_everything
from src.models.cnn.model import FoodCNNLightning
from src.models.cnn.datamodule import Food101DataModule


def main():
    parser = argparse.ArgumentParser(description="Fine-tune CNN on Food-101")
    parser.add_argument("--backbone", default="resnet50")
    parser.add_argument("--n-classes", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--data-root", default="data/")
    args = parser.parse_args()

    seed_everything(42)

    model = FoodCNNLightning(
        backbone=args.backbone, num_classes=args.n_classes,
        lr=args.lr, freeze_backbone=True,
    )
    dm = Food101DataModule(
        root=args.data_root, n_classes=args.n_classes,
        batch_size=args.batch_size,
    )

    trainer = pl.Trainer(
        max_epochs=args.epochs, accelerator="auto",
        callbacks=[pl.callbacks.EarlyStopping(monitor="val/loss", patience=5)],
        log_every_n_steps=10,
    )
    trainer.fit(model, dm)

    print("\nCNN training complete!")
    print(f"Best val/loss: {trainer.callback_metrics.get('val/loss', 'N/A')}")


if __name__ == "__main__":
    main()
