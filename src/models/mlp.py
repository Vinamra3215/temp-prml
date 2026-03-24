"""
MLP classifier implementations (sklearn + PyTorch).
Covers: Course Topics #17, #18, #19.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.neural_network import MLPClassifier


def get_sklearn_mlp(hidden_layers=(512, 256, 128), max_iter=500):
    """Get sklearn MLPClassifier with specified architecture."""
    return MLPClassifier(
        hidden_layer_sizes=hidden_layers,
        activation="relu",
        solver="adam",
        max_iter=max_iter,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=42,
    )


class FoodMLP(nn.Module):
    """
    PyTorch MLP for food classification.
    Demonstrates backpropagation and gradient descent.
    """

    def __init__(self, input_dim, num_classes, hidden_dims=(512, 256, 128), dropout=0.3):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


def train_pytorch_mlp(model, X_train, y_train, X_val, y_val, epochs=50, lr=0.001):
    """Train PyTorch MLP with backpropagation."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    X_train_t = torch.FloatTensor(X_train).to(device)
    y_train_t = torch.LongTensor(y_train).to(device)
    X_val_t = torch.FloatTensor(X_val).to(device)
    y_val_t = torch.LongTensor(y_val).to(device)

    history = {"train_loss": [], "val_loss": [], "val_acc": []}

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_t)
        loss = criterion(outputs, y_train_t)
        loss.backward()       # Backpropagation
        optimizer.step()       # Gradient descent update

        model.eval()
        with torch.no_grad():
            val_out = model(X_val_t)
            val_loss = criterion(val_out, y_val_t)
            val_pred = val_out.argmax(dim=1)
            val_acc = (val_pred == y_val_t).float().mean().item()

        history["train_loss"].append(loss.item())
        history["val_loss"].append(val_loss.item())
        history["val_acc"].append(val_acc)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} - Loss: {loss:.4f} - Val Acc: {val_acc:.4f}")

    return model, history
