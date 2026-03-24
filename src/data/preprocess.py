"""
Image preprocessing and augmentation utilities.
"""
import cv2
import numpy as np
from torchvision import transforms


def get_train_transform(image_size: int = 224):
    """Training transform with data augmentation."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def get_test_transform(image_size: int = 224):
    """Test/validation transform without augmentation."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def load_image(path: str, size: int = 224) -> np.ndarray:
    """Load and resize image as RGB numpy array (HWC, uint8)."""
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Cannot load image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (size, size))
    return img


def load_images_batch(paths: list, size: int = 224) -> np.ndarray:
    """Load multiple images as a batch (N, H, W, C)."""
    images = []
    for p in paths:
        try:
            images.append(load_image(p, size))
        except FileNotFoundError:
            continue
    return np.array(images)
