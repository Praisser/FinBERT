"""SQLite storage for collected prices and FinBERT-scored headlines."""

import sqlite3
from contextlib import closing
from datetime import date, datetime

import pandas as pd

from .config import DB_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS prices (
    ticker TEXT NOT NULL,
    date   TEXT NOT NULL,               -- YYYY-MM-DD
    open REAL, high REAL, low REAL, close REAL, volume REAL,
    PRIMARY KEY (ticker, date)
);
CREATE TABLE IF NOT EXISTS news (
    ticker    TEXT NOT NULL,
    query     TEXT NOT NULL,            -- news search the headline came from
    published TEXT NOT NULL,            -- UTC, YYYY-MM-DD HH:MM:SS
    headline  TEXT NOT NULL,
    positive REAL, negative REAL, neutral REAL, label TEXT,
    PRIMARY KEY (ticker, query, headline)
);
CREATE TABLE IF NOT EXISTS news_coverage (  -- date range already collected for each search
    ticker     TEXT NOT NULL,
    query      TEXT NOT NULL,
    start_date TEXT NOT NULL,
    end_date   TEXT NOT NULL,
    PRIMARY KEY (ticker, query)
);
"""

PRICE_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]
NEWS_COLUMNS = ["published", "headline", "positive", "negative", "neutral", "label"]


def connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.executescript(SCHEMA)
    return conn


def _utc_text(moment: date | datetime | pd.Timestamp) -> str:
    ts = pd.Timestamp(moment)
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC")
    return ts.strftime("%Y-%m-%d %H:%M:%S")


def save_prices(ticker: str, prices: pd.DataFrame) -> None:
    """Replace a stock's price history. Adjusted prices change after dividends and splits,
    so the whole history is rewritten rather than appended to."""
    rows = [
        (ticker, day.strftime("%Y-%m-%d"), *values)
        for day, values in zip(prices.index, prices[PRICE_COLUMNS].astype(float).itertuples(index=False))
    ]
    with closing(connect()) as conn, conn:
        conn.execute("DELETE FROM prices WHERE ticker = ?", (ticker,))
        conn.executemany("INSERT INTO prices VALUES (?, ?, ?, ?, ?, ?, ?)", rows)


def load_prices(ticker: str, since: date) -> pd.DataFrame:
    sql = """SELECT date, open AS Open, high AS High, low AS Low, close AS Close, volume AS Volume
             FROM prices WHERE ticker = ? AND date >= ? ORDER BY date"""
    with closing(connect()) as conn:
        return pd.read_sql(sql, conn, params=(ticker, since.isoformat()), parse_dates=["date"], index_col="date")


def save_news(ticker: str, query: str, news: pd.DataFrame) -> None:
    """Add scored headlines; ones already stored are skipped."""
    news = news.assign(published=news["published"].map(_utc_text))
    rows = [(ticker, query, *values) for values in news[NEWS_COLUMNS].itertuples(index=False)]
    with closing(connect()) as conn, conn:
        conn.executemany("INSERT OR IGNORE INTO news VALUES (?, ?, ?, ?, ?, ?, ?, ?)", rows)


def load_news(ticker: str, query: str, since: date | datetime | pd.Timestamp) -> pd.DataFrame:
    sql = f"""SELECT {', '.join(NEWS_COLUMNS)} FROM news
              WHERE ticker = ? AND query = ? AND published >= ? ORDER BY published"""
    with closing(connect()) as conn:
        news = pd.read_sql(sql, conn, params=(ticker, query, _utc_text(since)))
    news["published"] = pd.to_datetime(news["published"], utc=True)
    return news


def stored_headlines(ticker: str, query: str) -> set[str]:
    with closing(connect()) as conn:
        rows = conn.execute("SELECT headline FROM news WHERE ticker = ? AND query = ?", (ticker, query))
        return {headline for (headline,) in rows}


def news_coverage(ticker: str, query: str) -> tuple[date, date] | None:
    with closing(connect()) as conn:
        row = conn.execute(
            "SELECT start_date, end_date FROM news_coverage WHERE ticker = ? AND query = ?", (ticker, query)
        ).fetchone()
    return (date.fromisoformat(row[0]), date.fromisoformat(row[1])) if row else None


def set_news_coverage(ticker: str, query: str, start: date, end: date) -> None:
    with closing(connect()) as conn, conn:
        conn.execute(
            "INSERT OR REPLACE INTO news_coverage VALUES (?, ?, ?, ?)",
            (ticker, query, start.isoformat(), end.isoformat()),
        )
