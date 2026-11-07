"""Headline sentiment with FinBERT (positive / negative / neutral)."""

from functools import cache

import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .config import FINBERT_MODEL

LABELS = ["positive", "negative", "neutral"]


@cache
def _load_finbert():
    tokenizer = AutoTokenizer.from_pretrained(FINBERT_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(FINBERT_MODEL).eval()
    return tokenizer, model


@torch.no_grad()
def score_headlines(headlines: list[str], batch_size: int = 64) -> pd.DataFrame:
    """Class probabilities and the predicted label for each headline."""
    if not headlines:
        return pd.DataFrame(columns=[*LABELS, "label"])

    tokenizer, model = _load_finbert()
    probs = []
    for i in range(0, len(headlines), batch_size):
        batch = tokenizer(headlines[i : i + batch_size], padding=True, truncation=True, max_length=64, return_tensors="pt")
        probs.append(torch.softmax(model(**batch).logits, dim=-1))

    columns = [model.config.id2label[i].lower() for i in range(model.config.num_labels)]
    scores = pd.DataFrame(torch.cat(probs).numpy(), columns=columns)[LABELS]
    scores["label"] = scores.idxmax(axis=1)
    return scores
