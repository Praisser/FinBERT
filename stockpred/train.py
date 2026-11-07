"""Training with early stopping, hyperparameter tuning, and inference."""

import copy
import itertools
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .config import BATCH_SIZE, EPOCHS, LEARNING_RATE, PATIENCE, SEED, WINDOW
from .dataset import SequenceData


def train(model: nn.Module, data: SequenceData) -> tuple[nn.Module, float]:
    """Fit on the training split; keep the weights with the lowest validation loss.

    Returns the model and that validation loss.
    """
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
    return model.eval(), best_loss


@dataclass
class Tuned:
    model: nn.Module
    params: dict
    trials: list[dict]  # every setting tried: params, val_loss, selected


def tune(model_cls: type[nn.Module], data: SequenceData, space: dict[str, list]) -> Tuned:
    """Grid search: train one model per combination in `space` and keep the one with
    the lowest validation loss. The test split plays no part in the choice."""
    trials, best = [], None
    for values in itertools.product(*space.values()):
        params = dict(zip(space, values))
        torch.manual_seed(SEED)
        model, val_loss = train(model_cls(data.X_train.shape[2], WINDOW, **params), data)
        trials.append({"params": params, "val_loss": val_loss})
        if best is None or val_loss < best[1]:
            best = (model, val_loss, params)

    model, _, params = best
    for trial in trials:
        trial["selected"] = trial["params"] == params
    return Tuned(model, params, trials)


@torch.no_grad()
def predict(model: nn.Module, X: np.ndarray) -> np.ndarray:
    return model.eval()(torch.from_numpy(X)).squeeze(-1).numpy()
