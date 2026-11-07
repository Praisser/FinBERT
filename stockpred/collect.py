"""Data sources: prices from Yahoo Finance and news headlines from Google News."""

import logging
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta

import pandas as pd
import requests
import yfinance as yf

from .config import NEWS_WINDOW_DAYS

log = logging.getLogger(__name__)

GOOGLE_NEWS_RSS = "https://news.google.com/rss/search"


def fetch_prices(ticker: str, start: date, end: date) -> pd.DataFrame:
    """Daily OHLCV (split/dividend adjusted), indexed by trading date."""
    prices = yf.Ticker(ticker).history(start=start, end=end + timedelta(days=1), auto_adjust=True)
    if prices.empty:
        raise ValueError(f"No price data found for ticker {ticker!r}")
    prices.index = prices.index.tz_localize(None).normalize()
    prices.index.name = "date"
    return prices[["Open", "High", "Low", "Close", "Volume"]].dropna()


def fetch_intraday(ticker: str) -> pd.DataFrame:
    """1-minute OHLCV bars for the last few sessions, in the exchange's time zone."""
    bars = yf.Ticker(ticker).history(period="5d", interval="1m")
    if bars.empty:
        raise ValueError(f"No intraday data found for ticker {ticker!r}")
    return bars[["Open", "High", "Low", "Close", "Volume"]]


def default_news_query(ticker: str) -> str:
    """'RELIANCE.NS' -> 'RELIANCE stock'."""
    return f"{ticker.split('.')[0].lstrip('^')} stock"


def fetch_news(query: str, start: date, end: date) -> pd.DataFrame:
    """Headlines matching `query` published between start and end.

    Returns columns: published (UTC timestamp), headline.
    """
    step = timedelta(days=NEWS_WINDOW_DAYS)
    searches = []
    day = start
    while day < end:
        searches.append(f"{query} after:{day} before:{min(day + step, end)}")
        day += step

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(_search, searches))

    failed = sum(r is None for r in results)
    if failed:
        log.warning("%d of %d news requests failed; continuing with the rest", failed, len(searches))
    return _to_frame([row for result in results if result for row in result])


def fetch_latest_news(query: str) -> pd.DataFrame:
    """Headlines from the last 24 hours."""
    rows = _search(f"{query} when:1d")
    if rows is None:
        raise ConnectionError("Could not reach Google News")
    return _to_frame(rows)


def _search(q: str, retries: int = 3) -> list[tuple[str, str]] | None:
    params = {"q": q, "hl": "en-US", "gl": "US", "ceid": "US:en"}
    for attempt in range(retries):
        try:
            resp = requests.get(GOOGLE_NEWS_RSS, params=params, headers={"User-Agent": "Mozilla/5.0"}, timeout=20)
            resp.raise_for_status()
            return [_parse_item(item) for item in ET.fromstring(resp.content).iter("item")]
        except (requests.RequestException, ET.ParseError):
            time.sleep(2**attempt)
    return None


def _parse_item(item: ET.Element) -> tuple[str, str]:
    title = item.findtext("title", "")
    source = item.findtext("source", "")
    # Google appends the publisher: "Apple hits record high - Reuters"
    if source and title.endswith(f" - {source}"):
        title = title[: -len(source) - 3]
    return item.findtext("pubDate", ""), title.strip()


def _to_frame(rows: list[tuple[str, str]]) -> pd.DataFrame:
    news = pd.DataFrame(rows, columns=["published", "headline"])
    news["published"] = pd.to_datetime(news["published"], utc=True, format="mixed", errors="coerce")
    news = news[news["published"].notna() & (news["headline"] != "")]
    return news.drop_duplicates("headline").sort_values("published").reset_index(drop=True)
