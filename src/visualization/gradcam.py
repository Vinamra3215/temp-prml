"""Grad-CAM visualization for CNN interpretability."""
import numpy as np
import matplotlib.pyplot as plt
import torch


def visualize_gradcam(model, image_tensor, target_class, layer_name=None, save_path=None):
    """Generate and visualize Grad-CAM heatmap."""
    try:
        from pytorch_grad_cam import GradCAM
        from pytorch_grad_cam.utils.image import show_cam_on_image

        target_layers = [list(model.modules())[-3]]
        cam = GradCAM(model=model, target_layers=target_layers)
        targets = None
        grayscale_cam = cam(input_tensor=image_tensor.unsqueeze(0), targets=targets)
        grayscale_cam = grayscale_cam[0, :]

        img_np = image_tensor.permute(1, 2, 0).numpy()
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
        visualization = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)

        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(img_np)
        plt.title("Original")
        plt.axis("off")
        plt.subplot(1, 2, 2)
        plt.imshow(visualization)
        plt.title(f"Grad-CAM (class {target_class})")
        plt.axis("off")
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    except ImportError:
        print("Install pytorch-grad-cam: pip install grad-cam")
