"""Finance news analyzer: Flask backend.

Pipeline per request: clean/tokenize -> sentiment (VADER) -> sector (Naive Bayes)
-> LLM summary / insights / Q&A (Grok API, with an offline fallback).
"""
import json
import os
from pathlib import Path

from flask import Flask, jsonify, render_template, request

from nlp.classifier import SectorClassifier
from nlp.llm import llm_available, run_llm, MODEL
from nlp.preprocess import clean_text, preprocess
from nlp.sentiment import analyze_sentiment, evaluate as eval_sentiment

BASE = Path(__file__).parent
app = Flask(__name__)

ARTICLES = json.loads((BASE / "data" / "articles.json").read_text(encoding="utf-8"))
classifier = SectorClassifier().train()
SENTIMENT_METRICS = eval_sentiment(BASE / "data" / "labeled_sentiment.csv")

MAX_INPUT_CHARS = 20000


def _get_text():
    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()
    if not text:
        return None, None, (jsonify(error="Please provide some article text."), 400)
    return text[:MAX_INPUT_CHARS], data, None


def run_nlp(text: str) -> dict:
    sent = analyze_sentiment(text)
    sector = classifier.predict(text)
    tokens = preprocess(text)
    return {
        "sentiment_label": sent["label"],
        "sentiment_score": sent["compound"],
        "sentiment_detail": {k: sent[k] for k in ("positive", "neutral", "negative")},
        "sentences": sent["sentences"],
        "category": sector["category"],
        "category_confidence": sector["confidence"],
        "category_top": sector["top"],
        "stats": {
            "characters": len(text),
            "tokens_after_cleaning": len(tokens),
            "cleaned_preview": " ".join(tokens[:25]),
            "cleaned_text_sample": clean_text(text)[:160],
        },
    }


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/articles")
def articles():
    return jsonify(ARTICLES)


@app.get("/api/status")
def status():
    return jsonify(
        llm_mode=f"llm:{MODEL}" if llm_available() else "offline",
        sector_metrics={k: v for k, v in classifier.metrics.items() if k != "report"},
        sentiment_metrics={"n": SENTIMENT_METRICS["n"], "accuracy": SENTIMENT_METRICS["accuracy"]},
    )


@app.post("/api/analyze")
def analyze():
    text, _, err = _get_text()
    if err:
        return err
    return jsonify(run_nlp(text))


@app.post("/api/llm")
def llm():
    text, data, err = _get_text()
    if err:
        return err
    task = data.get("task")
    if task not in ("summarize", "insights", "ask"):
        return jsonify(error="task must be 'summarize', 'insights' or 'ask'."), 400
    question = (data.get("question") or "").strip()
    if task == "ask" and not question:
        return jsonify(error="Please type a question."), 400
    nlp = run_nlp(text)
    result = run_llm(task, text, question, nlp)
    return jsonify(result)


if __name__ == "__main__":
    print(f"LLM mode: {'xAI Grok API (' + MODEL + ')' if llm_available() else 'offline fallback (set GROK_API_KEY for the LLM)'}")
    app.run(debug=False, port=int(os.environ.get("PORT", 5000)))
