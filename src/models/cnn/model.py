"""
CNN model with Transfer Learning (Lightning wrapper).
Covers: Course Topics #20, #21.
"""
import torch
import torch.nn as nn
import pytorch_lightning as pl
import timm


class FoodCNN(nn.Module):
    """Custom CNN for food classification."""

    def __init__(self, num_classes=20):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2), nn.Dropout2d(0.25),

            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2), nn.Dropout2d(0.25),

            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2, 2), nn.Dropout2d(0.25),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(128, 512), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


class FoodCNNLightning(pl.LightningModule):
    """Lightning wrapper for CNN training with transfer learning."""

    def __init__(self, backbone="resnet50", num_classes=20, lr=0.001,
                 weight_decay=1e-4, freeze_backbone=True):
        super().__init__()
        self.save_hyperparameters()
        self.model = timm.create_model(backbone, pretrained=True, num_classes=num_classes)

        if freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False
            # Unfreeze final classifier
            classifier = self.model.get_classifier()
            for param in classifier.parameters():
                param.requires_grad = True

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("train/loss", loss, prog_bar=True)
        self.log("train/acc", acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("val/loss", loss, prog_bar=True)
        self.log("val/acc", acc, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=self.hparams.lr, weight_decay=self.hparams.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "monitor": "val/loss"}}
