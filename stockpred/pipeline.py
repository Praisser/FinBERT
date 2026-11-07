"""End-to-end pipeline: collect -> features -> tune & train -> evaluate -> explain -> save.

Each stage is a function, so the notebook can run them one at a time.
Results for each ticker are written to artifacts/<TICKER>/:
    dataset.csv       merged market + indicator + sentiment dataset
    predictions.csv   actual vs predicted close over the test period
    shap_<model>.csv  per-feature SHAP values (test period + next-day forecast)
    summary.json      model comparison, tuning results and next-day forecast
    figures/*.png     report charts
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from . import collect, db, explain, features, ingest, plots
from .config import ARTIFACTS_DIR, HISTORY_YEARS, MARKET_FEATURES, SEARCH_SPACE, SENTIMENT_FEATURES
from .dataset import prepare
from .evaluate import regression_metrics
from .models import MODELS
from .train import predict, tune

log = logging.getLogger(__name__)

# The main models use all features; the second set measures how much sentiment helps.
MAIN = "with sentiment"
FEATURE_SETS = {
    MAIN: MARKET_FEATURES + SENTIMENT_FEATURES,
    "without sentiment": MARKET_FEATURES,
}


@dataclass
class Results:
    predictions: pd.DataFrame = field(default_factory=pd.DataFrame)  # actual vs predicted close, test period
    comparison: list[dict] = field(default_factory=list)             # metrics per model and feature set
    tuning: list[dict] = field(default_factory=list)                 # every hyperparameter setting tried
    forecast: dict[str, float] = field(default_factory=dict)         # next-day close per model
    shap: dict[str, pd.DataFrame] = field(default_factory=dict)      # per-feature SHAP values per model


def collect_data(ticker: str, years: int = HISTORY_YEARS, news_query: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """Update the database for this stock, then read back prices, scored headlines and the news query used."""
    ticker = ticker.upper()
    end = date.today()
    start = end - timedelta(days=round(365.25 * years))
    query = news_query or collect.default_news_query(ticker)
    ingest.update(ticker, query, start, end)
    return db.load_prices(ticker, start), db.load_news(ticker, query, start), query


def build_dataset(prices: pd.DataFrame, news: pd.DataFrame) -> pd.DataFrame:
    if news.empty:
        log.warning("No headlines found; sentiment will be zero")
    return features.build_dataset(prices, features.daily_sentiment(news, prices.index))


def model_results(data: pd.DataFrame) -> Results:
    """Tune, train, evaluate and explain both models, with and without sentiment."""
    close = data["Close"].to_numpy()
    results = Results()

    for variant, cols in FEATURE_SETS.items():
        seq = prepare(data, cols)
        dates = data.index[seq.test_rows + 1]
        actual = close[seq.test_rows + 1]
        if variant == MAIN:
            results.predictions = pd.DataFrame({"actual": actual}, index=pd.Index(dates, name="date"))

        for name, model_cls in MODELS.items():
            log.info("Tuning %s (%s)", name, variant)
            tuned = tune(model_cls, seq, SEARCH_SPACE[name])
            results.tuning += [{"model": name, "features": variant, **trial} for trial in tuned.trials]

            # Models predict tomorrow's log return; convert it to a price.
            predicted = close[seq.test_rows] * np.exp(seq.to_return(predict(tuned.model, seq.X_test)))
            metrics = regression_metrics(actual, predicted)
            results.comparison.append({"model": name, "features": variant, "params": tuned.params, **metrics})

            if variant == MAIN:
                results.predictions[name] = predicted
                next_return = seq.to_return(predict(tuned.model, seq.X_latest))[0]
                results.forecast[name] = float(close[-1] * np.exp(next_return))

                log.info("Explaining %s predictions with SHAP", name)
                samples = np.concatenate([seq.X_test, seq.X_latest])
                shap_pct = explain.feature_contributions(tuned.model, seq.X_train, samples) * seq.y_std * 100
                index = pd.Index([*dates.strftime("%Y-%m-%d"), "forecast"], name="date")
                results.shap[name] = pd.DataFrame(shap_pct, index=index, columns=cols)

    return results


def save_results(ticker: str, news_query: str, data: pd.DataFrame, results: Results) -> Path:
    out = ARTIFACTS_DIR / ticker.upper()
    (out / "figures").mkdir(parents=True, exist_ok=True)

    data.to_csv(out / "dataset.csv")
    results.predictions.to_csv(out / "predictions.csv")
    for name, values in results.shap.items():
        values.to_csv(out / f"shap_{name}.csv")

    summary = {
        "ticker": ticker.upper(),
        "news_query": news_query,
        "start": data.index[0].strftime("%Y-%m-%d"),
        "as_of": data.index[-1].strftime("%Y-%m-%d"),
        "last_close": float(data["Close"].iloc[-1]),
        "forecast": results.forecast,
        "comparison": results.comparison,
        "tuning": results.tuning,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))

    figures = {
        "prediction": plots.prediction_figure(results.predictions),
        "sentiment": plots.sentiment_figure(data),
        "comparison": plots.comparison_figure(pd.DataFrame(results.comparison)),
        **{f"shap_{name}": plots.shap_figure(values, name) for name, values in results.shap.items()},
    }
    for name, fig in figures.items():
        fig.savefig(out / "figures" / f"{name}.png", dpi=150)
    return out


def run(ticker: str, years: int = HISTORY_YEARS, news_query: str | None = None) -> Path:
    prices, news, query = collect_data(ticker, years, news_query)
    log.info("Building features from %d trading days and %d headlines", len(prices), len(news))
    data = build_dataset(prices, news)
    results = model_results(data)
    out = save_results(ticker, query, data, results)
    log.info("Done. Results saved to %s", out)
    return out
