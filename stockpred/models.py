"""The two forecasting models compared in this project."""

import torch
from torch import nn


class LSTMRegressor(nn.Module):
    def __init__(self, n_features: int, window: int, hidden: int = 64, layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, num_layers=layers, dropout=dropout, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.head(out[:, -1])


class TransformerRegressor(nn.Module):
    def __init__(self, n_features: int, window: int, d_model: int = 64, heads: int = 4, layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.embed = nn.Linear(n_features, d_model)
        self.position = nn.Parameter(torch.randn(1, window, d_model) * 0.02)  # learned positional encoding
        layer = nn.TransformerEncoderLayer(d_model, heads, dim_feedforward=2 * d_model, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(self.embed(x) + self.position)
        return self.head(h[:, -1])


MODELS = {"LSTM": LSTMRegressor, "Transformer": TransformerRegressor}
