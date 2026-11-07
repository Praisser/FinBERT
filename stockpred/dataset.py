"""Turn the daily dataset into scaled sliding windows with a chronological split."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .config import TARGET, TEST_FRACTION, VAL_FRACTION, WINDOW


@dataclass
class SequenceData:
    """Model-ready arrays. X has shape (samples, WINDOW, features); y is the scaled target."""

    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    X_latest: np.ndarray    # the most recent window, used for the next-day forecast
    test_rows: np.ndarray   # dataset row of the last day in each test window
    y_mean: float
    y_std: float

    def to_return(self, y_scaled: np.ndarray) -> np.ndarray:
        """Undo target scaling: model output -> log return."""
        return y_scaled * self.y_std + self.y_mean


def prepare(data: pd.DataFrame, features: list[str]) -> SequenceData:
    X = data[features].to_numpy(np.float64)
    y = data[TARGET].to_numpy(np.float64)

    window_ends = np.arange(WINDOW - 1, len(data))
    labelled = window_ends[~np.isnan(y[window_ends])]
    n_test = int(len(labelled) * TEST_FRACTION)
    n_val = int(len(labelled) * VAL_FRACTION)
    train, val, test = np.split(labelled, [len(labelled) - n_val - n_test, len(labelled) - n_test])

    # Fit scalers on training rows only, so nothing leaks from the future.
    scaler = StandardScaler().fit(X[: train[-1] + 1])
    y_mean, y_std = y[train].mean(), y[train].std()
    y_scaled = ((y - y_mean) / y_std).astype(np.float32)

    windows = np.lib.stride_tricks.sliding_window_view(scaler.transform(X).astype(np.float32), WINDOW, axis=0)
    windows = windows.transpose(0, 2, 1)  # (n, features, window) -> (n, window, features)

    def take(ends: np.ndarray) -> np.ndarray:
        return windows[ends - WINDOW + 1]

    return SequenceData(
        X_train=take(train), y_train=y_scaled[train],
        X_val=take(val), y_val=y_scaled[val],
        X_test=take(test), y_test=y_scaled[test],
        X_latest=take(window_ends[-1:]),
        test_rows=test,
        y_mean=float(y_mean), y_std=float(y_std),
    )
