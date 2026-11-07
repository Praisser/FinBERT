"""End-to-end pipeline: collect -> sentiment -> features -> train -> evaluate -> explain.

Results for each ticker are written to artifacts/<TICKER>/:
    news.csv          scored headlines
    dataset.csv       merged market + indicator + sentiment dataset
    predictions.csv   actual vs predicted close over the test period
    shap_<model>.csv  per-feature SHAP values (test period + next-day forecast)
    summary.json      model comparison and next-day forecast
"""

import json
import logging
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from . import collect, explain, features, sentiment
from .config import ARTIFACTS_DIR, HISTORY_YEARS, MARKET_FEATURES, SEED, SENTIMENT_FEATURES, WINDOW
from .dataset import prepare
from .evaluate import regression_metrics
from .models import MODELS
from .train import predict, train

log = logging.getLogger(__name__)

# The main models use all features; the second set measures how much sentiment helps.
FEATURE_SETS = {
    "with sentiment": MARKET_FEATURES + SENTIMENT_FEATURES,
    "without sentiment": MARKET_FEATURES,
}


def run(ticker: str, years: int = HISTORY_YEARS, news_query: str | None = None) -> Path:
    ticker = ticker.upper()
    end = date.today()
    start = end - timedelta(days=round(365.25 * years))
    out = ARTIFACTS_DIR / ticker
    out.mkdir(parents=True, exist_ok=True)

    log.info("Downloading %s prices from %s to %s", ticker, start, end)
    prices = collect.fetch_prices(ticker, start, end)

    query = news_query or collect.default_news_query(ticker)
    log.info("Collecting news headlines for %r", query)
    news = collect.fetch_news(query, start, end)
    if news.empty:
        log.warning("No headlines found; sentiment features will be zero")

    log.info("Scoring %d headlines with FinBERT", len(news))
    news = news.join(sentiment.score_headlines(news["headline"].tolist()))
    news.to_csv(out / "news.csv", index=False)

    log.info("Building features")
    data = features.build_dataset(prices, features.daily_sentiment(news, prices.index))
    data.to_csv(out / "dataset.csv")

    close = data["Close"].to_numpy()
    comparison, forecast = [], {}
    predictions = pd.DataFrame()

    for variant, cols in FEATURE_SETS.items():
        seq = prepare(data, cols)
        dates = data.index[seq.test_rows + 1]
        actual = close[seq.test_rows + 1]
        is_main = variant == "with sentiment"
        if is_main:
            predictions = pd.DataFrame({"actual": actual}, index=pd.Index(dates, name="date"))

        for name, model_cls in MODELS.items():
            log.info("Training %s (%s)", name, variant)
            torch.manual_seed(SEED)
            model = train(model_cls(len(cols), WINDOW), seq)
            # Model predicts tomorrow's log return; convert it to a price.
            predicted = close[seq.test_rows] * np.exp(seq.to_return(predict(model, seq.X_test)))
            comparison.append({"model": name, "features": variant, **regression_metrics(actual, predicted)})

            if is_main:
                predictions[name] = predicted
                forecast[name] = float(close[-1] * np.exp(seq.to_return(predict(model, seq.X_latest))[0]))

                log.info("Explaining %s predictions with SHAP", name)
                samples = np.concatenate([seq.X_test, seq.X_latest])
                shap_pct = explain.feature_contributions(model, seq.X_train, samples) * seq.y_std * 100
                index = pd.Index([*dates.strftime("%Y-%m-%d"), "forecast"], name="date")
                pd.DataFrame(shap_pct, index=index, columns=cols).to_csv(out / f"shap_{name}.csv")

    predictions.to_csv(out / "predictions.csv")
    summary = {
        "ticker": ticker,
        "news_query": query,
        "as_of": data.index[-1].strftime("%Y-%m-%d"),
        "last_close": float(close[-1]),
        "forecast": forecast,
        "comparison": comparison,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    log.info("Done. Results saved to %s", out)
    return out
