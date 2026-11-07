"""Project-wide settings: paths, feature lists and training hyperparameters."""

from pathlib import Path

ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts"

# Data collection
HISTORY_YEARS = 5
NEWS_WINDOW_DAYS = 7  # Google News is queried one week at a time to reach older headlines

# Sentiment
FINBERT_MODEL = "ProsusAI/finbert"

# Features. Price-based features are relative (returns, ratios) so they stay on
# the same scale as the stock trends up or down over the years.
MARKET_FEATURES = [
    "return",         # log return Close/prev Close
    "intraday",       # log return Close/Open
    "hl_range",       # (High - Low) / Close
    "volume_ratio",   # Volume / 20-day average volume
    "rsi",            # 14-day RSI
    "macd",           # MACD line / Close
    "macd_signal",    # MACD signal line / Close
    "close_vs_sma",   # Close / 20-day SMA - 1
    "close_vs_ema",   # Close / 20-day EMA - 1
]
# Headline counts are deliberately not a feature: Google News keeps far more
# headlines for recent weeks than for past years, so counts would mostly reflect
# the age of the data rather than the market.
SENTIMENT_FEATURES = [
    "sentiment",      # mean FinBERT (positive - negative) of the day's headlines, 0 if none
]
TARGET = "target"     # next day's log return

# Training
WINDOW = 30           # trading days of history fed to the models
VAL_FRACTION = 0.15
TEST_FRACTION = 0.15
BATCH_SIZE = 32
EPOCHS = 100
PATIENCE = 10         # early-stopping patience (epochs)
LEARNING_RATE = 1e-3
SEED = 42

# Explainability
SHAP_BACKGROUND = 100  # training windows used as the SHAP baseline
