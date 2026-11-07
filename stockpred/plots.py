"""Static Matplotlib charts for reports and the notebook (the dashboard uses interactive Plotly)."""

import matplotlib.dates as mdates
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .config import FEATURE_LABELS

# Same colorblind-safe colors as the dashboard
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
POSITIVE, NEGATIVE = "#2a78d6", "#e34948"
NAMES = {"actual": "Actual close", "LSTM": "LSTM prediction", "Transformer": "Transformer prediction"}


def _date_axis(ax: Axes) -> None:
    locator = mdates.AutoDateLocator()
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))


def prediction_figure(predictions: pd.DataFrame) -> Figure:
    """Actual vs predicted close over the test period."""
    fig = Figure(figsize=(10, 4.5), layout="constrained")
    ax = fig.subplots()
    for column, color in zip(predictions, SERIES):
        ax.plot(predictions.index, predictions[column], label=NAMES.get(column, column), color=color, linewidth=1.5)
    ax.set(title="Actual vs predicted close (test period)", ylabel="Close")
    ax.legend()
    ax.grid(alpha=0.3)
    _date_axis(ax)
    return fig


def sentiment_figure(data: pd.DataFrame) -> Figure:
    """Closing price above daily FinBERT sentiment."""
    fig = Figure(figsize=(10, 6), layout="constrained")
    price_ax, sent_ax = fig.subplots(2, 1, sharex=True, height_ratios=[3, 2])
    price_ax.plot(data.index, data["Close"], color=SERIES[0])
    price_ax.set(title="Closing price and daily news sentiment", ylabel="Close")
    sent_ax.bar(data.index, data["sentiment"], width=1.0, color=[POSITIVE if v >= 0 else NEGATIVE for v in data["sentiment"]])
    sent_ax.set(ylabel="Sentiment (positive minus negative)", ylim=(-1, 1))
    for ax in (price_ax, sent_ax):
        ax.grid(alpha=0.3)
    _date_axis(sent_ax)
    return fig


def comparison_figure(comparison: pd.DataFrame) -> Figure:
    """MAE, RMSE and R² for each model, with and without sentiment features."""
    table = comparison.pivot(index="model", columns="features", values=["MAE", "RMSE", "R2"])
    fig = Figure(figsize=(12, 4), layout="constrained")
    axes = fig.subplots(1, 3)
    for ax, metric in zip(axes, ["MAE", "RMSE", "R2"]):
        values = table[metric]
        width = 0.8 / len(values.columns)
        for i, variant in enumerate(values.columns):
            bars = ax.bar([x + i * width for x in range(len(values))], values[variant], width, label=variant, color=SERIES[i])
            ax.bar_label(bars, fmt="%.3f", fontsize=8)
        ax.set_xticks([x + width * (len(values.columns) - 1) / 2 for x in range(len(values))], values.index)
        ax.set_title("R²" if metric == "R2" else metric)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center", ncols=2)
    fig.suptitle("Model comparison on the test period")
    return fig


def shap_figure(shap_values: pd.DataFrame, model_name: str) -> Figure:
    """Mean absolute SHAP value of each feature over the test period."""
    importance = shap_values.drop(index="forecast", errors="ignore").abs().mean().rename(index=FEATURE_LABELS).sort_values()
    fig = Figure(figsize=(7, 0.4 * len(importance) + 1.2), layout="constrained")
    ax = fig.subplots()
    ax.barh(importance.index, importance.values, color=SERIES[0])
    ax.set(title=f"{model_name}: feature importance (SHAP)", xlabel="Mean absolute SHAP value (percentage points of return)")
    ax.grid(axis="x", alpha=0.3)
    return fig
