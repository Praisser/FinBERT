"""Feature engineering: technical indicators, daily news sentiment and the merged dataset."""

import numpy as np
import pandas as pd

from .config import MARKET_FEATURES, SENTIMENT_FEATURES, TARGET

MARKET_CLOSE_HOUR = 16  # US/Eastern


def technical_indicators(prices: pd.DataFrame) -> pd.DataFrame:
    close, volume = prices["Close"], prices["Volume"]
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26

    return pd.DataFrame(
        {
            "return": np.log(close).diff(),
            "intraday": np.log(close / prices["Open"]),
            "hl_range": (prices["High"] - prices["Low"]) / close,
            "volume_ratio": volume / volume.rolling(20).mean(),
            "rsi": rsi(close, 14),
            "macd": macd / close,
            "macd_signal": macd.ewm(span=9, adjust=False).mean() / close,
            "close_vs_sma": close / close.rolling(20).mean() - 1,
            "close_vs_ema": close / close.ewm(span=20, adjust=False).mean() - 1,
        },
        index=prices.index,
    )


def rsi(close: pd.Series, period: int) -> pd.Series:
    """Relative Strength Index with Wilder's smoothing."""
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    return 100 - 100 / (1 + gain / loss)


def daily_sentiment(news: pd.DataFrame, trading_days: pd.DatetimeIndex) -> pd.Series:
    """Average headline sentiment (positive - negative) per trading day; 0 on days without news.

    Each headline is assigned to the first trading session that closes after it
    was published, so a prediction made at a day's close never sees later news.
    """
    published = news["published"].dt.tz_convert("America/New_York")
    after_close = (published.dt.hour >= MARKET_CLOSE_HOUR).astype(int)
    day = published.dt.tz_localize(None).dt.normalize() + pd.to_timedelta(after_close, unit="D")

    pos = trading_days.searchsorted(day)
    in_range = pos < len(trading_days)
    scored = pd.DataFrame(
        {
            "date": trading_days[pos[in_range]],
            "score": (news["positive"] - news["negative"])[in_range].to_numpy(),
        }
    )
    daily = scored.groupby("date")["score"].mean().reindex(trading_days, fill_value=0.0)
    return daily.astype(float).rename("sentiment")


def build_dataset(prices: pd.DataFrame, sentiment: pd.Series) -> pd.DataFrame:
    """One row per trading day: OHLCV, indicators, sentiment and the next-day target.

    The last row has no target; it is the input for the next-day forecast.
    """
    data = prices.join(technical_indicators(prices)).join(sentiment)
    features = MARKET_FEATURES + SENTIMENT_FEATURES
    data[features] = data[features].replace([np.inf, -np.inf], np.nan).ffill()
    data[TARGET] = data["return"].shift(-1)
    return data.dropna(subset=features)  # drops the indicator warm-up rows
