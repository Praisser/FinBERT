"""Training with early stopping, and inference."""

import copy

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .config import BATCH_SIZE, EPOCHS, LEARNING_RATE, PATIENCE
from .dataset import SequenceData


def train(model: nn.Module, data: SequenceData) -> nn.Module:
    """Fit on the training split; keep the weights with the lowest validation loss."""
    loader = DataLoader(
        TensorDataset(torch.from_numpy(data.X_train), torch.from_numpy(data.y_train)),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    X_val, y_val = torch.from_numpy(data.X_val), torch.from_numpy(data.y_val)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.MSELoss()

    best_loss, best_state, stale = float("inf"), None, 0
    for _ in range(EPOCHS):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            loss_fn(model(xb).squeeze(-1), yb).backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(X_val).squeeze(-1), y_val).item()
        if val_loss < best_loss:
            best_loss, best_state, stale = val_loss, copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
            if stale >= PATIENCE:
                break

    model.load_state_dict(best_state)
    return model.eval()


@torch.no_grad()
def predict(model: nn.Module, X: np.ndarray) -> np.ndarray:
    return model.eval()(torch.from_numpy(X)).squeeze(-1).numpy()
