# Financial News Sentiment and Stock Price Prediction

Stock price prediction with NLP and deep learning: LSTM vs Transformer, explained with SHAP.

The system combines a stock's daily market data and technical indicators with FinBERT sentiment from
financial news headlines. It trains LSTM and Transformer models to predict the next day's closing
price, compares them with MAE, RMSE and R², and uses SHAP to explain which features drove each
prediction. A Streamlit dashboard shows everything.

Project report and review slides are in [`docs/`](docs/).

## Setup

Requires Python 3.10 or newer (tested with 3.12).

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

The FinBERT model (~440 MB) downloads automatically the first time it runs.

## Usage

**Dashboard.** Enter a ticker, run the analysis and explore the results:

```bash
streamlit run app.py
```

**Command line.** Runs the same pipeline and writes results to `artifacts/<TICKER>/`:

```bash
python -m stockpred AAPL
python -m stockpred RELIANCE.NS --years 3 --query "Reliance Industries"
```

A full run takes a few minutes: most of the time goes to collecting headlines, FinBERT scoring and
training four models.

## How it works

| Stage | Module | What happens |
|---|---|---|
| 1. Data collection | `collect.py` | Daily OHLCV from Yahoo Finance (`yfinance`); headlines from Google News, fetched one week at a time |
| 2. Sentiment analysis | `sentiment.py` | FinBERT (`ProsusAI/finbert`) gives each headline positive / negative / neutral probabilities |
| 3. Feature engineering | `features.py` | RSI, MACD, SMA, EMA, volume ratio, returns; headlines are averaged into a daily sentiment score |
| 4. Data integration | `features.py` | Market features and daily sentiment merged into one dataset, aligned by trading day |
| 5. Pre-processing | `dataset.py` | 30-day sliding windows, chronological 70/15/15 train/val/test split, scaling fitted on training data only |
| 6. Model training | `models.py`, `train.py` | LSTM and Transformer encoder (PyTorch), Adam + early stopping |
| 7. Evaluation | `evaluate.py` | MAE, RMSE, R² on the test period, for both models **with and without** sentiment features |
| 8. Explainability | `explain.py` | SHAP (GradientExplainer): contribution of every feature to every prediction |
| Dashboard | `app.py` | Prediction chart, next-day forecast, news sentiment, model comparison, SHAP explanations |

`pipeline.py` runs the stages in order. All settings (window length, split sizes, hyperparameters,
feature lists) are in `config.py`.

### Design notes

- **No look-ahead.** A headline published after the 4 pm (US/Eastern) close, or on a non-trading day,
  counts toward the next trading session. Scalers are fitted on training data only. The split is chronological.
- **Return-based target.** Models predict the next day's log return, which is converted back to a
  price (`close_today × e^return`). Metrics are reported on that price. Price features are expressed as
  returns and ratios (for example close vs SMA). This keeps the inputs on a stable scale even when a
  stock trades far above its historical range.
- **SHAP units.** SHAP values are reported in percentage points of predicted next-day return. A
  feature's value is summed over the 30-day input window.

## Output files

`artifacts/<TICKER>/`:

| File | Contents |
|---|---|
| `news.csv` | Every headline with its FinBERT probabilities and label |
| `dataset.csv` | The integrated dataset: OHLCV, indicators, daily sentiment, target |
| `predictions.csv` | Actual vs LSTM vs Transformer close over the test period |
| `shap_LSTM.csv`, `shap_Transformer.csv` | Per-feature SHAP values for each test prediction and the forecast |
| `summary.json` | Model comparison metrics and next-day forecast |

## Limitations

- Google News returns only about 1–3 headlines per trading day for older periods, and many more for
  the last few weeks. Days without news get a neutral sentiment of 0. Headline counts are not used as
  a feature, because they would reflect how old the data is rather than market activity.
- Results are for research and education. They are not investment advice.
