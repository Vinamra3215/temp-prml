"""
Lightning DataModule for Food-101.
"""
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class Food101DataModule(pl.LightningDataModule):
    """Lightning DataModule for Food-101 dataset."""

    def __init__(self, root="data/", n_classes=20, batch_size=32, image_size=224, num_workers=4):
        super().__init__()
        self.root = root
        self.n_classes = n_classes
        self.batch_size = batch_size
        self.image_size = image_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        train_transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(0.2, 0.2, 0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        test_transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        self.train_ds = datasets.Food101(self.root, split="train", transform=train_transform, download=True)
        self.val_ds = datasets.Food101(self.root, split="test", transform=test_transform, download=True)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
