"""SHAP explanations for the sequence models."""

import numpy as np
import shap
import torch
from torch import nn

from .config import SEED, SHAP_BACKGROUND


def feature_contributions(model: nn.Module, background: np.ndarray, samples: np.ndarray) -> np.ndarray:
    """SHAP value of each feature for each sample, shape (samples, features).

    Each input is a window of days, so SHAP gives one value per (day, feature);
    these are summed over the window to get a single value per feature.
    """
    rng = np.random.default_rng(SEED)
    baseline = background[rng.choice(len(background), min(SHAP_BACKGROUND, len(background)), replace=False)]

    explainer = shap.GradientExplainer(model.eval(), torch.from_numpy(baseline))
    values = np.asarray(explainer.shap_values(torch.from_numpy(samples), rseed=SEED))
    return values.reshape(samples.shape).sum(axis=1)
