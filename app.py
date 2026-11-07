"""Dashboard: run the pipeline for a stock and explore its predictions, news sentiment,
model comparison and SHAP explanations.

    streamlit run app.py
"""

import json
import logging

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from stockpred.config import ARTIFACTS_DIR, HISTORY_YEARS
from stockpred.models import MODELS
from stockpred.pipeline import run

UP, DOWN = "#2e7d32", "#c62828"

st.set_page_config(page_title="Stock Sentiment Predictor", layout="wide")


class StatusHandler(logging.Handler):
    """Streams pipeline log messages into a Streamlit status box."""

    def __init__(self, status):
        super().__init__()
        self.status = status

    def emit(self, record):
        self.status.write(record.getMessage())


def run_pipeline(ticker: str, years: int) -> bool:
    logger = logging.getLogger("stockpred")
    logger.setLevel(logging.INFO)
    with st.status(f"Analysing {ticker}…", expanded=True) as status:
        handler = StatusHandler(status)
        logger.addHandler(handler)
        try:
            run(ticker, years=years)
            status.update(label=f"{ticker} analysed", state="complete", expanded=False)
            return True
        except Exception as exc:
            status.update(label=f"Analysis failed: {exc}", state="error")
            return False
        finally:
            logger.removeHandler(handler)


@st.cache_data
def load_results(ticker: str, version: float) -> dict:
    """Read a ticker's pipeline output. `version` (file mtime) refreshes the cache after a re-run."""
    folder = ARTIFACTS_DIR / ticker
    return {
        "summary": json.loads((folder / "summary.json").read_text()),
        "predictions": pd.read_csv(folder / "predictions.csv", index_col="date", parse_dates=True),
        "dataset": pd.read_csv(folder / "dataset.csv", index_col="date", parse_dates=True),
        "news": pd.read_csv(folder / "news.csv", parse_dates=["published"]),
        "shap": {name: pd.read_csv(folder / f"shap_{name}.csv", index_col="date") for name in MODELS},
    }


def signed_bar(values: pd.Series, x_title: str) -> go.Figure:
    values = values.sort_values()
    fig = go.Figure(go.Bar(x=values, y=values.index, orientation="h", marker_color=[UP if v >= 0 else DOWN for v in values]))
    fig.update_layout(xaxis_title=x_title, height=60 + 32 * len(values), margin=dict(t=10, b=10))
    return fig


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Analyse a stock")
    with st.form("analyse"):
        new_ticker = st.text_input("Ticker (Yahoo Finance symbol)", placeholder="AAPL, MSFT, RELIANCE.NS")
        years = st.slider("Years of history", 2, 10, HISTORY_YEARS)
        submitted = st.form_submit_button("Run analysis", type="primary")
    st.caption("Collects data, scores news with FinBERT, trains both models and computes SHAP. Takes a few minutes.")

if submitted and new_ticker.strip():
    ticker = new_ticker.strip().upper()
    if run_pipeline(ticker, years):
        st.session_state["ticker"] = ticker

analysed = sorted(p.name for p in ARTIFACTS_DIR.glob("*") if (p / "summary.json").exists())
if not analysed:
    st.title("Stock Sentiment Predictor")
    st.info("Enter a ticker in the sidebar and click **Run analysis** to get started.")
    st.stop()

ticker = st.sidebar.selectbox("Analysed stocks", analysed, key="ticker")
results = load_results(ticker, (ARTIFACTS_DIR / ticker / "summary.json").stat().st_mtime)
summary = results["summary"]

# ---------------------------------------------------------------- header
st.title(f"{ticker} price prediction")
st.caption(f"Market data up to {summary['as_of']} · news search: “{summary['news_query']}”")

last_close = summary["last_close"]
cols = st.columns(1 + len(summary["forecast"]))
cols[0].metric(f"Close on {summary['as_of']}", f"{last_close:,.2f}")
for col, (name, price) in zip(cols[1:], summary["forecast"].items()):
    col.metric(f"{name} · next-day forecast", f"{price:,.2f}", f"{price / last_close - 1:+.2%}")

tab_pred, tab_news, tab_compare, tab_shap = st.tabs(["Prediction", "News sentiment", "Model comparison", "Explanation (SHAP)"])

# ---------------------------------------------------------------- prediction
with tab_pred:
    st.subheader("Actual vs predicted close (test period)")
    fig = px.line(results["predictions"], labels={"value": "Close", "date": "", "variable": ""})
    fig.update_layout(legend=dict(orientation="h", y=1.08))
    st.plotly_chart(fig)
    st.caption("Each point is predicted using only data available at the previous day's close.")

# ---------------------------------------------------------------- news sentiment
with tab_news:
    news, dataset = results["news"], results["dataset"]
    counts = news["label"].value_counts()
    cols = st.columns(4)
    cols[0].metric("Headlines", f"{len(news):,}")
    for col, label in zip(cols[1:], ["positive", "negative", "neutral"]):
        count = counts.get(label, 0)
        col.metric(f"{label.title()} ({count / max(len(news), 1):.0%})", f"{count:,}")

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.6, 0.4], vertical_spacing=0.05)
    fig.add_trace(go.Scatter(x=dataset.index, y=dataset["Close"], name="Close", line_color="#1f77b4"), row=1, col=1)
    fig.add_trace(
        go.Bar(
            x=dataset.index, y=dataset["sentiment"], name="Daily sentiment",
            marker_color=[UP if v >= 0 else DOWN for v in dataset["sentiment"]],
        ),
        row=2, col=1,
    )
    fig.update_yaxes(title_text="Close", row=1, col=1)
    fig.update_yaxes(title_text="Sentiment", range=[-1, 1], row=2, col=1)
    fig.update_layout(height=520, showlegend=False, margin=dict(t=10))
    st.plotly_chart(fig)
    st.caption("Daily sentiment = average FinBERT (positive − negative) probability across that day's headlines.")

    st.subheader("Latest headlines")
    latest = news.sort_values("published", ascending=False).head(200)
    st.dataframe(
        latest[["published", "headline", "label", "positive", "negative", "neutral"]],
        hide_index=True,
        column_config={
            "published": st.column_config.DatetimeColumn("Published", format="YYYY-MM-DD HH:mm"),
            "headline": "Headline",
            "label": "Sentiment",
            **{c: st.column_config.ProgressColumn(c.title(), min_value=0, max_value=1, format="%.2f") for c in ["positive", "negative", "neutral"]},
        },
    )

# ---------------------------------------------------------------- model comparison
with tab_compare:
    comparison = pd.DataFrame(summary["comparison"])
    best = comparison.loc[comparison["RMSE"].idxmin()]
    st.subheader("LSTM vs Transformer, with and without news sentiment")
    st.write(f"Lowest error on the test period: **{best['model']} ({best['features']})**, RMSE {best['RMSE']:.2f}.")
    st.dataframe(
        comparison,
        hide_index=True,
        column_config={
            "model": "Model",
            "features": "Features",
            "MAE": st.column_config.NumberColumn(format="%.3f"),
            "RMSE": st.column_config.NumberColumn(format="%.3f"),
            "R2": st.column_config.NumberColumn("R²", format="%.4f"),
        },
    )
    fig = px.bar(comparison, x="model", y="RMSE", color="features", barmode="group", labels={"model": "", "features": ""})
    st.plotly_chart(fig)
    st.caption("MAE and RMSE are in price units (lower is better). R² closer to 1 is better.")

# ---------------------------------------------------------------- SHAP
with tab_shap:
    model_name = st.radio("Model", list(MODELS), horizontal=True)
    shap_values = results["shap"][model_name]
    unit = "impact on predicted next-day return (percentage points)"

    left, right = st.columns(2)
    with left:
        st.subheader("Overall feature importance")
        importance = shap_values.drop(index="forecast").abs().mean().sort_values()
        fig = px.bar(x=importance.values, y=importance.index, orientation="h", labels={"x": "mean |SHAP|", "y": ""})
        fig.update_layout(height=60 + 32 * len(importance), margin=dict(t=10, b=10))
        st.plotly_chart(fig)
        st.caption("Average size of each feature's effect across the test period.")
    with right:
        st.subheader("Why this prediction?")
        dates = shap_values.index[::-1].tolist()
        day = st.selectbox("Prediction", dates, format_func=lambda d: "Next-day forecast" if d == "forecast" else d)
        st.plotly_chart(signed_bar(shap_values.loc[day], unit))
        st.caption("Green features pushed the prediction up, red pushed it down.")
