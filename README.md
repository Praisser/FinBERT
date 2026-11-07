# Financial News Sentiment and Stock Price Prediction

Stock price prediction with NLP and deep learning: LSTM vs Transformer, explained with SHAP.

The system combines a stock's daily market data and technical indicators with FinBERT sentiment from
financial news headlines. It tunes and trains LSTM and Transformer models to predict the next day's
closing price, compares them with MAE, RMSE and R², and uses SHAP to explain which features drove each
prediction. A Streamlit dashboard shows everything, including live prices and headlines.

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

Tabs: Live market (refreshes every minute), Prediction, News sentiment, Model comparison (with
tuning results), Explanation (SHAP).

**Notebook.** [`stock_prediction.ipynb`](stock_prediction.ipynb) runs the method one stage at a time
with Matplotlib charts. Open it with `jupyter notebook`, or in
[Google Colab](https://colab.research.google.com/github/Praisser/FinBERT/blob/main/stock_prediction.ipynb),
where its first cell installs everything.

**Command line.** Runs the full pipeline and writes results to `artifacts/<TICKER>/`:

```bash
python -m stockpred AAPL
python -m stockpred RELIANCE.NS --years 3 --query "Reliance Industries"
```

The first run for a stock takes about 2 minutes, mostly collecting five years of headlines. Later runs
only fetch news published since the last run, so they take under a minute.

## How it works

| Stage | Module | What happens |
|---|---|---|
| 1. Data collection | `collect.py`, `ingest.py`, `db.py` | Daily OHLCV from Yahoo Finance; headlines from Google News, one week at a time. Both are stored in a SQLite database (`data/market.db`), and only missing periods are fetched |
| 2. Sentiment analysis | `sentiment.py` | FinBERT (`ProsusAI/finbert`) gives each new headline positive / negative / neutral probabilities, stored with the headline |
| 3. Feature engineering | `features.py` | RSI, MACD, SMA, EMA, volume ratio, returns; headlines averaged into a daily sentiment score |
| 4. Data integration | `features.py` | Market features and daily sentiment merged into one dataset, aligned by trading day |
| 5. Pre-processing | `dataset.py` | 30-day sliding windows, chronological 70/15/15 train/val/test split, scaling fitted on training data only |
| 6. Training and tuning | `models.py`, `train.py` | LSTM and Transformer encoder (PyTorch). A grid search over model size and depth keeps the setting with the lowest validation loss |
| 7. Evaluation | `evaluate.py` | MAE, RMSE, R² on the test period, for both models **with and without** sentiment |
| 8. Explainability | `explain.py` | SHAP (GradientExplainer): contribution of every feature to every prediction |
| Dashboard | `app.py` | Live market, prediction chart and next-day forecast, news sentiment, model comparison, SHAP |
| Report figures | `plots.py` | Matplotlib charts saved as PNG files with every run and shown in the notebook |

`pipeline.py` runs the stages in order. All settings (window length, split sizes, tuning grid,
feature lists, live refresh interval) are in `config.py`.

### Real-time data

The dashboard's **Live market** tab refreshes every 60 seconds. Each refresh downloads 1-minute prices
and the last 24 hours of headlines. New headlines are scored with FinBERT and saved to the database, so
the next model run includes them.

### Design notes

- **No look-ahead.** A headline published after the 4 pm (US/Eastern) close, or on a non-trading day,
  counts toward the next trading session. Scalers are fitted on training data only. The split is
  chronological. Tuning uses the validation split, never the test split.
- **Return-based target.** Models predict the next day's log return, which is converted back to a
  price (`close_today × e^return`). Metrics are reported on that price. Price features are expressed as
  returns and ratios (for example close vs SMA). This keeps the inputs on a stable scale even when a
  stock trades far above its historical range.
- **SHAP units.** SHAP values are reported in percentage points of predicted next-day return. A
  feature's value is summed over the 30-day input window.

## Output files

`data/market.db` holds the collected data. It has three tables:
- `prices`: daily OHLCV per ticker
- `news`: headlines with their FinBERT scores
- `news_coverage`: date ranges already collected

`artifacts/<TICKER>/`:

| File | Contents |
|---|---|
| `dataset.csv` | The integrated dataset: OHLCV, indicators, daily sentiment, target |
| `predictions.csv` | Actual vs LSTM vs Transformer close over the test period |
| `shap_LSTM.csv`, `shap_Transformer.csv` | Per-feature SHAP values for each test prediction and the forecast |
| `summary.json` | Model comparison, every tuning trial, and the next-day forecast |
| `figures/*.png` | Prediction, sentiment, model comparison and SHAP charts for reports |

## Limitations

- Google News returns only about 1–3 headlines per trading day for older periods, and many more for
  the last few weeks. Days without news get a neutral sentiment of 0. Headline counts are not used as
  a feature, because they would reflect how old the data is rather than market activity.
- The news-to-session timing uses the US market close (4 pm Eastern). For stocks on other exchanges
  this alignment is approximate.
- Results are for research and education. They are not investment advice.
