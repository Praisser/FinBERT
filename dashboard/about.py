"""The About page: project aim, methodology, technology and team (from the project report)."""

from math import prod

import streamlit as st

from stockpred.config import MARKET_FEATURES, SEARCH_SPACE, SENTIMENT_FEATURES, WINDOW

from . import ui

TITLE = (
    "Financial News Sentiment and Stock Price Prediction using NLP and Deep Learning: "
    "LSTM, Transformer and Explainable AI"
)

AIM = (
    "Stock prices respond both to their own history and to the news. This project predicts a stock's "
    "next-day closing price by combining its price history and technical indicators with the sentiment "
    "of financial news headlines, measured with FinBERT. It compares an LSTM with a Transformer model, "
    "measures how much news sentiment improves their accuracy, and uses SHAP to explain each prediction."
)

STAGES = [
    ("Data collection", "Daily prices from Yahoo Finance and news headlines from Google News, stored in a SQLite database."),
    ("Sentiment analysis", "FinBERT classifies every headline as positive, negative or neutral."),
    ("Feature engineering", "Technical indicators: returns, RSI, MACD, moving averages and volume."),
    ("Data integration", "Headlines are matched to the trading session they could affect and averaged into a daily score."),
    ("Pre-processing", "30-day sliding windows, a time-ordered 70/15/15 split, and scaling fitted on training data only."),
    ("Training and tuning", "LSTM and Transformer networks, each tuned with a grid search on validation loss."),
    ("Evaluation", "MAE, RMSE and R² on unseen test data, with and without news sentiment."),
    ("Explainability", "SHAP measures how much each feature moved every prediction."),
]

STACK = [
    ("Market data", "Yahoo Finance (yfinance), SQLite"),
    ("News and NLP", "Google News RSS, FinBERT (ProsusAI/finbert), Hugging Face Transformers"),
    ("Deep learning", "PyTorch: LSTM and Transformer encoder, scikit-learn"),
    ("Explainability", "SHAP (GradientExplainer)"),
    ("Interface", "Streamlit, Plotly, Matplotlib, Jupyter Notebook"),
]

STUDENTS = [
    ("B. Anirudh Reddy", "23B81A6669"),
    ("Y. Eeshanya", "23B81A6679"),
    ("Sk. Mehawish Fathima", "23B81A6680"),
]
GUIDE = ("Mrs. Farhana Bano", "Sr. Assistant Professor")
DEPARTMENT = "Department of CSE (Artificial Intelligence and Machine Learning)"
COLLEGE = "CVR College of Engineering, Hyderabad"

TRIALS = 2 * sum(prod(len(values) for values in grid.values()) for grid in SEARCH_SPACE.values())  # x2 feature sets


def render() -> None:
    st.html(f'<div class="fx-label">B.Tech major project</div><div class="fx-about-title">{TITLE}</div><div class="fx-lead">{AIM}</div>')
    st.html(
        '<div style="height:28px"></div>'
        + ui.figures([
            ("Input features", str(len(MARKET_FEATURES + SENTIMENT_FEATURES)), "per trading day"),
            ("Input window", f"{WINDOW}<small>days</small>", "of history per prediction"),
            ("Models tuned", str(TRIALS), "per stock"),
            ("Pipeline", f"{len(STAGES)}<small>stages</small>", "run in order"),
        ])
    )

    ui.section("Methodology", "From raw data to explained predictions.")
    steps = "".join(
        f'<div class="fx-step"><div class="fx-step-n">{number:02d}</div>'
        f'<div class="fx-step-title">{name}</div><div class="fx-step-text">{text}</div></div>'
        for number, (name, text) in enumerate(STAGES, 1)
    )
    st.html(f'<div class="fx-steps">{steps}</div>')

    ui.section("Technology")
    st.html('<div class="fx-defs">' + "".join(f"<div>{area}</div><div>{tools}</div>" for area, tools in STACK) + "</div>")

    ui.section("Team", f"{DEPARTMENT}, {COLLEGE}")
    people = [("Student", f"{name}, {roll}") for name, roll in STUDENTS] + [("Project guide", f"{GUIDE[0]}, {GUIDE[1]}")]
    st.html('<div class="fx-defs">' + "".join(f"<div>{role}</div><div>{who}</div>" for role, who in people) + "</div>")
