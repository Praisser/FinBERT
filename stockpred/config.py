"""Project-wide settings: paths, feature lists and training hyperparameters."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = ROOT / "artifacts"      # model results per ticker
DB_PATH = ROOT / "data" / "market.db"   # collected prices and scored headlines

# Data collection
HISTORY_YEARS = 5
NEWS_WINDOW_DAYS = 7  # Google News is queried one week at a time to reach older headlines
LIVE_REFRESH_SECONDS = 60

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

# Feature names as shown in charts
FEATURE_LABELS = {
    "return": "Daily return",
    "intraday": "Intraday return",
    "hl_range": "High-low range",
    "volume_ratio": "Volume vs 20-day average",
    "rsi": "RSI (14 days)",
    "macd": "MACD",
    "macd_signal": "MACD signal line",
    "close_vs_sma": "Close vs SMA (20 days)",
    "close_vs_ema": "Close vs EMA (20 days)",
    "sentiment": "News sentiment",
}

# Training
WINDOW = 30           # trading days of history fed to the models
VAL_FRACTION = 0.15
TEST_FRACTION = 0.15
BATCH_SIZE = 32
EPOCHS = 100
PATIENCE = 10         # early-stopping patience (epochs)
LEARNING_RATE = 1e-3
SEED = 42

# Tuning: every combination is trained and the one with the lowest validation
# loss is kept. The test period is never used for this choice.
SEARCH_SPACE = {
    "LSTM": {"hidden": [32, 64], "layers": [1, 2]},
    "Transformer": {"d_model": [32, 64], "layers": [1, 2]},
}

# Explainability
SHAP_BACKGROUND = 100  # training windows used as the SHAP baseline
