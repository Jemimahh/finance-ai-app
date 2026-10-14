"""Basic tests. Run from the project root:  python -m pytest -q   (or python tests/test_app.py)"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ["GROK_API_KEY"] = ""  # empty (not removed) so a real key in .env is never loaded; tests use offline mode

from app import app  # noqa: E402
from nlp.preprocess import clean_text, preprocess  # noqa: E402
from nlp.sentiment import analyze_sentiment  # noqa: E402

client = app.test_client()
POS = "Shares surged after the company beat estimates and raised full-year guidance on record sales."
NEG = "Shares plunged after the bank missed estimates, set aside large loan loss provisions and cut its dividend."


def test_clean_text_strips_html_and_urls():
    out = clean_text("<p>Stocks <b>JUMPED</b> 5%! See https://example.com</p>")
    assert "<" not in out and "http" not in out and out.startswith("stocks jumped 5%")


def test_preprocess_keeps_figures_and_drops_stopwords():
    toks = preprocess("The company said revenue rose 12% in the quarter")
    assert "12%" in toks and "the" not in toks


def test_sentiment_direction():
    assert analyze_sentiment(POS)["label"] == "positive"
    assert analyze_sentiment(NEG)["label"] == "negative"


def test_analyze_endpoint():
    r = client.post("/api/analyze", json={"text": NEG})
    j = r.get_json()
    assert r.status_code == 200
    assert j["sentiment_label"] == "negative"
    assert j["category"] == "Banking"


def test_analyze_rejects_empty():
    assert client.post("/api/analyze", json={"text": "  "}).status_code == 400


def test_llm_summarize_and_ask_offline():
    r = client.post("/api/llm", json={"task": "summarize", "text": POS + " Analysts were upbeat."})
    assert r.status_code == 200 and r.get_json()["mode"] == "offline" and r.get_json()["answer"]
    r = client.post("/api/llm", json={"task": "ask", "text": NEG, "question": "What happened to the dividend?"})
    assert "dividend" in r.get_json()["answer"].lower()


def test_llm_validation():
    assert client.post("/api/llm", json={"task": "ask", "text": POS}).status_code == 400
    assert client.post("/api/llm", json={"task": "nope", "text": POS}).status_code == 400


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
