"""Plotly charts in two colors.

Ink: drawn in the placeholder color INK, which style.css repaints in the page's text color, so
charts follow light and dark mode the moment the theme changes.
Gray: one mid gray that reads on both backgrounds, for secondary series. Series that share a
color are told apart by line style or marker shape.
"""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from stockpred.config import FEATURE_LABELS

INK = "#010101"  # placeholder, repainted by style.css
GRAY = "#8c8c8c"
GRID = "rgba(140, 140, 140, 0.18)"
CONFIG = {"displayModeBar": False}  # hover tooltips stay; double-click resets zoom
FORECAST_STYLE = {"LSTM": ("solid", "circle"), "Transformer": ("dot", "diamond")}


def show(fig: go.Figure) -> None:
    st.plotly_chart(fig, config=CONFIG)


def _style(fig: go.Figure, height: int, **layout) -> go.Figure:
    fig.update_layout(
        height=height,
        font=dict(color=GRAY),
        margin=dict(t=28, b=0, l=0, r=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, title_text="", traceorder="normal"),
        hoverlabel=dict(namelength=-1),
    )
    fig.update_layout(**layout)
    fig.update_xaxes(zeroline=False, ticks="", gridcolor=GRID)
    fig.update_yaxes(zeroline=False, ticks="", gridcolor=GRID)
    return fig


def price_outlook(data: pd.DataFrame, forecast: dict[str, float], days: int = 90) -> go.Figure:
    """Recent closes and both models' forecasts for the next session."""
    recent = data["Close"].tail(days)
    last_day, last_close = recent.index[-1], recent.iloc[-1]
    next_day = last_day + pd.offsets.BDay(1)
    fig = go.Figure()
    fig.add_scatter(x=recent.index, y=recent, name="Close", mode="lines", line=dict(color=INK, width=1.8), hovertemplate="%{y:,.2f}")
    for name, price in forecast.items():
        dash, symbol = FORECAST_STYLE[name]
        fig.add_scatter(
            x=[last_day, next_day], y=[last_close, price], name=f"{name} forecast", mode="lines+markers",
            line=dict(color=GRAY, width=1.5, dash=dash), marker=dict(size=[0, 9], symbol=symbol, color=GRAY),
            hovertemplate=f"{name} forecast %{{y:,.2f}}<extra></extra>",
        )
    fig.update_xaxes(showgrid=False, rangebreaks=[dict(bounds=["sat", "mon"])])
    return _style(fig, 300, hovermode="x unified")


def prediction(predictions: pd.DataFrame) -> go.Figure:
    """Actual close against both models' predictions over the test period."""
    fig = go.Figure()
    fig.add_scatter(x=predictions.index, y=predictions["actual"], name="Actual close", mode="lines", line=dict(color=INK, width=2), hovertemplate="%{y:,.2f}")
    for name, (dash, _) in FORECAST_STYLE.items():
        fig.add_scatter(
            x=predictions.index, y=predictions[name], name=f"{name} prediction", mode="lines",
            line=dict(color=GRAY, width=1.5, dash=dash), hovertemplate="%{y:,.2f}",
        )
    return _style(fig, 440, hovermode="x unified")


def intraday(bars: pd.DataFrame, previous_close: float | None) -> go.Figure:
    """One-minute prices for the latest session, with the previous close as a reference line."""
    fig = go.Figure(
        go.Scatter(x=bars.index, y=bars["Close"], mode="lines", line=dict(color=INK, width=1.6), hovertemplate="%{x|%H:%M}  %{y:,.2f}<extra></extra>")
    )
    if previous_close is not None:
        fig.add_hline(
            y=previous_close, line_width=1, line_color=GRAY,
            annotation_text=f"Previous close {previous_close:,.2f}", annotation_position="top left",
            annotation_font=dict(color=GRAY, size=11),
        )
    return _style(fig, 320, showlegend=False, margin=dict(t=10, b=0, l=0, r=0))


def price_and_sentiment(data: pd.DataFrame) -> go.Figure:
    """Closing price above daily news sentiment and its 20-day average."""
    sentiment = data["sentiment"]
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.6, 0.4], vertical_spacing=0.08)
    fig.add_scatter(x=data.index, y=data["Close"], name="Close price", mode="lines", line=dict(color=INK, width=1.6), hovertemplate="%{y:,.2f}", row=1, col=1)
    fig.add_bar(
        x=data.index, y=sentiment, name="Daily sentiment", marker_color=GRAY, opacity=0.7, marker_line_width=0,
        hovertemplate="%{y:+.2f}", row=2, col=1,
    )
    fig.add_scatter(
        x=data.index, y=sentiment.rolling(20).mean(), name="20-day average", mode="lines", line=dict(color=INK, width=1.4),
        hovertemplate="%{y:+.2f}", row=2, col=1,
    )
    fig.update_yaxes(title_text="Close price", row=1, col=1)
    fig.update_yaxes(title_text="Sentiment", range=[-1, 1], row=2, col=1)
    return _style(fig, 540, hovermode="x unified", bargap=0)


def error_bars(comparison: pd.DataFrame, metric: str) -> go.Figure:
    """One error metric per model, with sentiment in ink and without it in gray."""
    fig = go.Figure()
    for features, group in comparison.groupby("features", sort=False):
        fig.add_bar(
            y=group["model"], x=group[metric], name=str(features).capitalize(), orientation="h",
            marker_color=INK if features == "with sentiment" else GRAY,
            text=group[metric], texttemplate="%{x:.2f}", textposition="outside", cliponaxis=False,
            hovertemplate=f"%{{y}}, {features}: %{{x:.3f}}<extra></extra>",
        )
    fig.update_xaxes(range=[0, comparison[metric].max() * 1.2], title_text=f"{metric}, lower is better")
    fig.update_yaxes(autorange="reversed")
    return _style(fig, 240, barmode="group", bargap=0.35, bargroupgap=0.1)


def importance(shap_values: pd.DataFrame) -> go.Figure:
    """Mean absolute SHAP value of each feature over the test period."""
    values = shap_values.drop(index="forecast").abs().mean().rename(index=FEATURE_LABELS).sort_values()
    fig = go.Figure(go.Bar(x=values, y=values.index, orientation="h", marker_color=INK, hovertemplate="%{y}: %{x:.3f}<extra></extra>"))
    fig.update_xaxes(title_text="Mean absolute SHAP value (percentage points)")
    return _style(fig, 80 + 30 * len(values), showlegend=False, bargap=0.4, margin=dict(t=8, b=0, l=0, r=0))


def contributions(values: pd.Series) -> go.Figure:
    """Signed SHAP values for one prediction: ink pushed it up, gray pulled it down."""
    values = values.rename(index=FEATURE_LABELS).sort_values()
    fig = go.Figure()
    for part, color in [(values[values >= 0], INK), (values[values < 0], GRAY)]:
        fig.add_bar(x=part, y=part.index, orientation="h", marker_color=color, hovertemplate="%{y}: %{x:+.3f}<extra></extra>")
    fig.update_xaxes(title_text="Effect on predicted return (percentage points)", zeroline=True, zerolinewidth=1, zerolinecolor=GRAY)
    fig.update_yaxes(categoryorder="array", categoryarray=list(values.index))
    return _style(fig, 80 + 30 * len(values), showlegend=False, bargap=0.4, barmode="overlay", margin=dict(t=8, b=0, l=0, r=0))
