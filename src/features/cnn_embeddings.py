"""
CNN embedding extractor using pretrained backbones (timm).
Covers: Course Topic #21 — CNN.
"""
import numpy as np
import torch
from PIL import Image
from src.features.base import FeatureExtractor


class CNNEmbeddingExtractor(FeatureExtractor):
    """Extract deep feature embeddings from pretrained CNN backbones."""

    def __init__(self, backbone: str = "resnet50", device: str = "cuda"):
        import timm
        from torchvision import transforms

        self.device = device if torch.cuda.is_available() else "cpu"
        self.model = timm.create_model(backbone, pretrained=True, num_classes=0)
        self.model = self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    def extract(self, image: np.ndarray) -> np.ndarray:
        pil_image = Image.fromarray(image)
        tensor = self.transform(pil_image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            embedding = self.model(tensor).squeeze().cpu().numpy()

        return embedding.astype(np.float32)

    def extract_dataset(self, image_paths, labels, size=224, n_jobs=1, batch_size=32):
        """Override for GPU-batched extraction."""
        from torch.utils.data import DataLoader, Dataset

        transform = self.transform

        class PathDataset(Dataset):
            def __init__(self, paths, lbls, tfm):
                self.paths = paths
                self.labels = lbls
                self.tfm = tfm

            def __len__(self):
                return len(self.paths)

            def __getitem__(self, idx):
                img = Image.open(self.paths[idx]).convert("RGB")
                return self.tfm(img), self.labels[idx]

        dataset = PathDataset(image_paths, labels, transform)
        loader = DataLoader(
            dataset, batch_size=batch_size, num_workers=0, pin_memory=False
        )

        all_features = []
        all_labels = []
        self.model.eval()

        for batch_imgs, batch_labels in loader:
            batch_imgs = batch_imgs.to(self.device)
            with torch.no_grad():
                feats = self.model(batch_imgs).cpu().numpy()
            all_features.append(feats)
            all_labels.append(batch_labels.numpy())

        X = np.vstack(all_features).astype(np.float32)
        y = np.concatenate(all_labels)
        return X, y
