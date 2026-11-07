"""Dashboard: run the pipeline for a stock and present its predictions, live market data,
news sentiment, model comparison and SHAP explanations.

    streamlit run app.py
"""

import json
import logging
from html import escape

import pandas as pd
import streamlit as st

from dashboard import about, charts, ui
from dashboard.ui import day, pct, short_day
from stockpred import db, ingest
from stockpred.config import ARTIFACTS_DIR, HISTORY_YEARS, LIVE_REFRESH_SECONDS
from stockpred.models import MODELS
from stockpred.pipeline import run

st.set_page_config(page_title="Stock Price Prediction", page_icon=":material/show_chart:", layout="wide")
ui.inject_css()

PARAM_NAMES = {"hidden": "hidden size", "d_model": "model size", "layers": "layers"}


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
    with st.status(f"Analysing {ticker}. This takes a few minutes.", expanded=True) as status:
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
        "shap": {name: pd.read_csv(folder / f"shap_{name}.csv", index_col="date") for name in MODELS},
    }


def format_params(params: dict) -> str:
    return ", ".join(f"{PARAM_NAMES.get(key, key)} {value}" for key, value in params.items())


def day_change(closes: pd.Series) -> float:
    return closes.iloc[-1] / closes.iloc[-2] - 1


def headlines_table(news: pd.DataFrame) -> None:
    news = news.sort_values("published", ascending=False).assign(label=news["label"].str.capitalize())
    st.dataframe(
        news[["published", "headline", "label", "positive", "negative", "neutral"]],
        hide_index=True,
        height=420,
        column_config={
            "published": st.column_config.DatetimeColumn("Published (UTC)", format="D MMM YYYY, HH:mm"),
            "headline": st.column_config.TextColumn("Headline", width="large"),
            "label": "Sentiment",
            **{
                c: st.column_config.ProgressColumn(c.capitalize(), min_value=0, max_value=1, format="%.2f", width="small")
                for c in ["positive", "negative", "neutral"]
            },
        },
    )


@st.fragment(run_every=LIVE_REFRESH_SECONDS)
def live_market(ticker: str, query: str) -> None:
    """Re-runs on its own every LIVE_REFRESH_SECONDS while the page is open."""
    try:
        bars, news = ingest.refresh_live(ticker, query)
    except Exception as exc:
        st.warning(f"Live data is unavailable right now: {exc}")
        return

    sessions = bars.index.normalize()
    today = bars[sessions == sessions[-1]]
    earlier = bars[sessions < sessions[-1]]
    price, last_trade = today["Close"].iloc[-1], today.index[-1]
    previous_close = earlier["Close"].iloc[-1] if len(earlier) else None
    market_open = pd.Timestamp.now(tz=last_trade.tz) - last_trade < pd.Timedelta(minutes=15)
    sentiment = (news["positive"] - news["negative"]).mean() if len(news) else None

    st.html(
        f'<div class="fx-status"><b>Market {"open" if market_open else "closed"}</b>'
        f"<span>Last trade {short_day(last_trade)}, {last_trade:%H:%M %Z}</span>"
        f"<span>Updates every {LIVE_REFRESH_SECONDS} seconds</span></div>"
    )
    st.html(
        ui.figures([
            ("Last price", f"{price:,.2f}", "" if previous_close is None else f"{pct(price / previous_close - 1)} vs previous close"),
            ("Day range", f"{today['Low'].min():,.2f}<small>to {today['High'].max():,.2f}</small>", "this session"),
            ("Headlines", f"{len(news)}", "last 24 hours"),
            ("Sentiment", "None" if sentiment is None else f"{sentiment:+.2f}", "average, last 24 hours"),
        ])
    )
    with ui.panel("intraday"):
        ui.panel_header(f"Intraday price, {day(last_trade)}", "One-minute prices. The gray line marks the previous close.")
        charts.show(charts.intraday(today, previous_close))
    with ui.panel("live_news"):
        ui.panel_header("Latest headlines", "Scored with FinBERT and saved, so the next model run includes them.")
        st.html(ui.headline_list(news) if len(news) else '<div class="fx-panel-note">No headlines in the last 24 hours.</div>')


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.html(
        '<div class="fx-brand"><div class="fx-label">B.Tech major project</div>'
        '<div class="fx-brand-name">Stock Price Prediction</div>'
        '<div class="fx-panel-note">News sentiment with LSTM and Transformer models</div></div>'
    )
    picker = st.container()
    ui.side_label("Analyse a new stock")
    with st.form("analyse", border=False):
        new_ticker = st.text_input("Ticker", placeholder="AAPL, MSFT, RELIANCE.NS", help="A Yahoo Finance symbol.")
        years = st.slider("Years of history", 2, 10, HISTORY_YEARS)
        submitted = st.form_submit_button("Run analysis", type="primary", width="stretch")
    st.html(f'<div class="fx-side-foot">{about.DEPARTMENT}<br>{about.COLLEGE}</div>')

if submitted and new_ticker.strip():
    ticker = new_ticker.strip().upper()
    if run_pipeline(ticker, years):
        st.session_state["ticker"] = ticker

analysed = sorted(p.name for p in ARTIFACTS_DIR.glob("*") if (p / "summary.json").exists())
if not analysed:
    ui.section("Analyse your first stock", "Enter a Yahoo Finance ticker in the sidebar and click <b>Run analysis</b>. The first run takes about two minutes.")
    about.render()
    st.stop()

every = {t: load_results(t, (ARTIFACTS_DIR / t / "summary.json").stat().st_mtime) for t in analysed}
with picker:
    ui.side_label("Stocks")
    ticker = st.radio(
        "Stocks", analysed, key="ticker", label_visibility="collapsed",
        captions=[f"{every[t]['summary']['last_close']:,.2f}   {pct(day_change(every[t]['dataset']['Close']))}" for t in analysed],
    )

results = every[ticker]
summary, dataset = results["summary"], results["dataset"]
comparison = pd.DataFrame(summary["comparison"]).sort_values("RMSE", ignore_index=True)
best = comparison.iloc[0]
news = db.load_news(ticker, summary["news_query"], since=pd.Timestamp(summary["start"]))
last_close, forecast = summary["last_close"], summary["forecast"]

# ---------------------------------------------------------------- header
left, right = st.columns([1, 1.6], gap="large", vertical_alignment="bottom")
with left:
    st.html(
        f'<div class="fx-symbol">{escape(ticker)}</div>'
        f'<div class="fx-price-line"><span class="fx-price">{last_close:,.2f}</span>'
        f'<span class="fx-change">{pct(day_change(dataset["Close"]))}</span></div>'
        f'<div class="fx-meta">Close on {day(summary["as_of"])}<br>'
        f'{len(dataset):,} trading days of prices and news for "{escape(summary["news_query"])}"</div>'
    )
with right:
    charts.show(charts.price_outlook(dataset, forecast))

st.html(
    ui.figures([
        *[(f"{name} forecast", f"{price:,.2f}", f"{pct(price / last_close - 1)} next session") for name, price in forecast.items()],
        ("Most accurate model", best["model"], f"RMSE {best['RMSE']:.2f}, {best['features']}"),
        ("Headlines analysed", f"{len(news):,}", f"on {int(dataset['sentiment'].ne(0).sum()):,} trading days"),
    ])
)
st.html('<div style="height:20px"></div>')

tab_pred, tab_live, tab_news, tab_compare, tab_shap, tab_about = st.tabs(
    ["Prediction", "Live market", "News sentiment", "Model comparison", "Explainability", "About"]
)

# ---------------------------------------------------------------- prediction
with tab_pred:
    predictions = results["predictions"]
    start, end = day(predictions.index[0]), day(predictions.index[-1])
    ui.section(
        "Actual and predicted close",
        f"Every trading day from {start} to {end}, predicted using only data available at the previous day's close. "
        "None of these days were used for training.",
    )
    main = comparison[comparison["features"] == "with sentiment"].set_index("model")
    st.html(
        ui.figures(
            [("Test period", f"{len(predictions)}<small>days</small>", f"{start} to {end}")]
            + [
                (name, f"{main.loc[name, 'RMSE']:.2f}<small>RMSE</small>", f"MAE {main.loc[name, 'MAE']:.2f}, R² {main.loc[name, 'R2']:.3f}")
                for name in MODELS
            ]
        )
    )
    st.html('<div style="height:16px"></div>')
    with ui.panel("prediction"):
        ui.panel_header("Closing price over the test period", "The bold line is the actual close. Gray lines are the predictions: solid for LSTM, dotted for Transformer.")
        charts.show(charts.prediction(predictions))

# ---------------------------------------------------------------- live market
with tab_live:
    ui.section("Live market", "Prices and headlines update automatically while this page is open.")
    live_market(ticker, summary["news_query"])

# ---------------------------------------------------------------- news sentiment
with tab_news:
    counts = news["label"].value_counts()
    scores = news["positive"] - news["negative"]
    ui.section("News sentiment", f"FinBERT scored {len(news):,} headlines about {escape(ticker)} since {day(summary['start'])}.")
    st.html(
        ui.figures(
            [(label.capitalize(), f"{counts.get(label, 0) / max(len(news), 1):.0%}", f"{counts.get(label, 0):,} headlines") for label in ["positive", "neutral", "negative"]]
            + [("Average score", f"{scores.mean():+.2f}", "positive minus negative")]
        )
    )
    st.html('<div style="height:16px"></div>')
    with ui.panel("sentiment"):
        ui.panel_header(
            "Closing price and daily news sentiment",
            "Daily sentiment is the average of FinBERT's positive minus negative probability over that day's headlines. "
            "Days without news count as 0.",
        )
        charts.show(charts.price_and_sentiment(dataset))
    with ui.panel("headlines"):
        ui.panel_header("All headlines", "Sort by any column, or search with the icon in the table's corner.")
        headlines_table(news)

# ---------------------------------------------------------------- model comparison
with tab_compare:
    rmse = comparison.pivot(index="model", columns="features", values="RMSE")
    effects = [
        f"{'lowered' if row['with sentiment'] < row['without sentiment'] else 'raised'} the {model} error by "
        f"{abs(row['with sentiment'] - row['without sentiment']):.2f}"
        for model, row in rmse.iterrows()
    ]
    ui.section(
        "Model comparison",
        f"The <b>{best['model']} {best['features']}</b> has the lowest test error, with an RMSE of {best['RMSE']:.2f}. "
        f"Adding news sentiment {' and '.join(effects)}.",
    )
    with ui.panel("ranking"):
        ui.panel_header("Test results", "MAE and RMSE are in price units, lower is better. R² closer to 1 is better.")
        st.html(
            ui.table(
                [("Rank", ""), ("Model", ""), ("Features", ""), ("Tuned settings", "muted"), ("MAE", "num"), ("RMSE", "num"), ("R²", "num")],
                [
                    (
                        [str(rank), row.model, row.features, format_params(row.params), f"{row.MAE:.3f}", f"{row.RMSE:.3f}", f"{row.R2:.4f}"],
                        "best" if rank == 1 else "",
                    )
                    for rank, row in enumerate(comparison.itertuples(), 1)
                ],
            )
        )
    left, right = st.columns(2)
    for col, metric, title in [(left, "MAE", "Mean absolute error"), (right, "RMSE", "Root mean squared error")]:
        with col, ui.panel(f"error_{metric}"):
            ui.panel_header(title, "Bold bars use news sentiment, gray bars do not.")
            charts.show(charts.error_bars(comparison, metric))

    tuning = pd.DataFrame(summary["tuning"]).sort_values(["model", "features", "val_loss"], ignore_index=True)
    with ui.panel("tuning"):
        ui.panel_header(
            "Hyperparameter tuning",
            f"{len(tuning)} settings were trained. For each model and feature set, the one with the lowest validation "
            "loss was kept. The test period was not used for this choice.",
        )
        with st.expander(f"All {len(tuning)} tuning runs"):
            st.html(
                ui.table(
                    [("Model", ""), ("Features", ""), ("Settings", "muted"), ("Validation loss", "num"), ("", "num")],
                    [
                        ([row.model, row.features, format_params(row.params), f"{row.val_loss:.4f}", "Selected" if row.selected else ""], "best" if row.selected else "")
                        for row in tuning.itertuples()
                    ],
                )
            )

# ---------------------------------------------------------------- explainability
with tab_shap:
    ui.section(
        "Explainability",
        "SHAP splits each prediction into the contribution of every input feature, in percentage points of "
        "predicted next-day return, summed over the 30-day input window.",
    )
    model_name = st.segmented_control("Model", list(MODELS), default=list(MODELS)[0], required=True, label_visibility="collapsed")
    shap_values = results["shap"][model_name]
    left, right = st.columns(2)
    with left, ui.panel("importance"):
        ui.panel_header("Overall feature importance", "Average size of each feature's effect across the test period.")
        charts.show(charts.importance(shap_values))
    with right, ui.panel("contributions"):
        ui.panel_header("Why this prediction?", "Bold bars pushed the predicted return up, gray bars pulled it down.")
        chosen = st.selectbox(
            "Prediction", shap_values.index[::-1].tolist(), label_visibility="collapsed",
            format_func=lambda d: "Next-session forecast" if d == "forecast" else day(d),
        )
        charts.show(charts.contributions(shap_values.loc[chosen]))

# ---------------------------------------------------------------- about
with tab_about:
    about.render()
