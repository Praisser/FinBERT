"""Keeps the database current: fetch what is missing, score new headlines with FinBERT, store."""

import logging
from datetime import date, timedelta

import pandas as pd

from . import collect, db, sentiment
from .config import NEWS_WINDOW_DAYS

log = logging.getLogger(__name__)


def update(ticker: str, query: str, start: date, end: date) -> None:
    """Bring one stock's prices and headlines in the database up to date."""
    log.info("Downloading %s prices from %s to %s", ticker, start, end)
    db.save_prices(ticker, collect.fetch_prices(ticker, start, end))
    update_news(ticker, query, start, end)


def update_news(ticker: str, query: str, start: date, end: date) -> None:
    """Fetch only the periods not collected before. The most recent stored week is
    fetched again because Google keeps adding headlines for a few days."""
    covered = db.news_coverage(ticker, query)
    if covered is None or start > covered[1] or end < covered[0]:
        missing = [(start, end)]
        new_coverage = (start, end)
    else:
        missing = [(start, covered[0])] if start < covered[0] else []
        missing.append((max(covered[1] - timedelta(days=NEWS_WINDOW_DAYS), start), end))
        new_coverage = (min(start, covered[0]), max(end, covered[1]))

    for period_start, period_end in missing:
        log.info("Collecting %r headlines from %s to %s", query, period_start, period_end)
        store_headlines(ticker, query, collect.fetch_news(query, period_start, period_end))
    db.set_news_coverage(ticker, query, *new_coverage)


def store_headlines(ticker: str, query: str, news: pd.DataFrame) -> int:
    """Score headlines not yet in the database and save them. Returns how many were new."""
    new = news[~news["headline"].isin(db.stored_headlines(ticker, query))].reset_index(drop=True)
    if new.empty:
        return 0
    log.info("Scoring %d new headlines with FinBERT", len(new))
    db.save_news(ticker, query, new.join(sentiment.score_headlines(new["headline"].tolist())))
    return len(new)


def refresh_live(ticker: str, query: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Latest 1-minute prices and the last 24 hours of headlines.

    New headlines are scored and stored, so the next model run includes them.
    """
    bars = collect.fetch_intraday(ticker)
    store_headlines(ticker, query, collect.fetch_latest_news(query))
    news = db.load_news(ticker, query, since=pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=1))
    return bars, news
