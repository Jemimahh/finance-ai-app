"""Rule-based sentiment analysis (VADER) tuned for financial text.

VADER's default lexicon is built for social media, so we add a small set of
finance-specific terms (e.g. "downgrade", "beats", "guidance") before scoring.
"""
import csv
from pathlib import Path

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from .preprocess import split_sentences

FINANCE_LEXICON = {
    # positive
    "beat": 2.0, "beats": 2.0, "surge": 2.3, "surged": 2.3, "soar": 2.5, "soars": 2.5,
    "soared": 2.5, "jump": 1.8, "jumped": 1.8, "rally": 2.0, "rallies": 2.0, "upgrade": 2.0,
    "upgraded": 2.0, "outperform": 2.0, "record": 1.5, "rebound": 1.6, "expands": 1.2,
    "raised": 1.0, "buyback": 1.5, "approval": 1.5, "profit": 1.0, "growth": 1.2,
    "robust": 1.8, "solid": 1.2, "bullish": 2.0, "de-risk": 1.5, "lifted": 1.5, "expanded": 1.5,
    "doubled": 1.8, "highest": 1.5, "jumps": 1.8, "beating": 1.8, "higher": 0.6,
    # negative
    "miss": -2.0, "misses": -2.0, "missed": -2.0, "plunge": -2.8, "plunged": -2.8,
    "tumble": -2.5, "tumbles": -2.5, "slide": -1.8, "slides": -1.8, "slump": -2.3,
    "downgrade": -2.2, "downgraded": -2.2, "underperform": -2.0, "default": -2.5,
    "defaults": -2.5, "bankruptcy": -3.2, "probe": -1.8, "investigation": -1.8,
    "fraud": -3.0, "recall": -1.8, "layoffs": -2.0, "cut": -1.3, "cuts": -1.3,
    "warn": -1.6, "warned": -1.6, "warning": -1.8, "losses": -2.0, "loss": -1.8,
    "weak": -1.7, "weaker": -1.7, "decline": -1.5, "declines": -1.5, "declining": -1.5,
    "fell": -1.8, "falls": -1.8, "drop": -1.6, "dropped": -1.6, "bearish": -2.0,
    "nonperforming": -2.2, "delinquencies": -2.0, "volatility": -0.8, "slowing": -1.2,
    "pressure": -1.0, "shortage": -1.8, "downturn": -2.0, "restrictions": -0.8, "outflows": -1.5, "negative": -1.5, "tight": -0.5,
}

_analyzer = SentimentIntensityAnalyzer()
_analyzer.lexicon.update(FINANCE_LEXICON)

POS_THRESHOLD = 0.15
NEG_THRESHOLD = -0.15


def label_from_score(compound: float) -> str:
    if compound >= POS_THRESHOLD:
        return "positive"
    if compound <= NEG_THRESHOLD:
        return "negative"
    return "neutral"


def analyze_sentiment(text: str) -> dict:
    """Return document-level scores plus per-sentence scores."""
    scores = _analyzer.polarity_scores(text or "")
    sentences = []
    for s in split_sentences(text):
        c = _analyzer.polarity_scores(s)["compound"]
        sentences.append({"sentence": s, "compound": round(c, 3), "label": label_from_score(c)})

    # Documents are long; averaging sentence scores is more stable than VADER's
    # whole-text compound, which saturates toward +/-1.
    if sentences:
        mean = sum(x["compound"] for x in sentences) / len(sentences)
    else:
        mean = scores["compound"]
    return {
        "label": label_from_score(mean),
        "compound": round(mean, 3),
        "positive": round(scores["pos"], 3),
        "neutral": round(scores["neu"], 3),
        "negative": round(scores["neg"], 3),
        "sentences": sentences,
    }


def evaluate(csv_path: str | Path) -> dict:
    """Accuracy and per-class F1 on a labeled sentiment CSV (columns: label,text)."""
    from sklearn.metrics import accuracy_score, classification_report

    y_true, y_pred = [], []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            y_true.append(row["label"])
            y_pred.append(label_from_score(_analyzer.polarity_scores(row["text"])["compound"]))
    return {
        "n": len(y_true),
        "accuracy": round(accuracy_score(y_true, y_pred), 3),
        "report": classification_report(y_true, y_pred, zero_division=0),
    }
