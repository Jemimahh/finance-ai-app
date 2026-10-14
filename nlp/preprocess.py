"""Text cleaning, tokenization and lemmatization for finance text.

Uses NLTK when its data is available and falls back to a light built-in
tokenizer / stop-word list so the app still runs offline.
"""
import html
import re

_STOPWORDS_FALLBACK = {
    "a", "an", "the", "and", "or", "but", "if", "of", "to", "in", "on", "at", "by",
    "for", "with", "as", "is", "are", "was", "were", "be", "been", "it", "its",
    "this", "that", "these", "those", "from", "has", "have", "had", "will", "would",
    "than", "then", "so", "such", "into", "over", "after", "about", "said", "says",
}

try:
    # Uses NLTK data if it is already installed; never downloads at import time.
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer

    STOPWORDS = set(stopwords.words("english"))
    _lemmatizer = WordNetLemmatizer()
    _lemmatizer.lemmatize("tests")  # confirm wordnet is present
    _USE_NLTK = True
except Exception:  # NLTK or its data not installed -> built-in fallback
    STOPWORDS = _STOPWORDS_FALLBACK
    _lemmatizer = None
    _USE_NLTK = False

# Keep negations: they matter for sentiment.
STOPWORDS = STOPWORDS - {"no", "not", "nor", "against", "down", "up", "off", "over", "below", "above"}

_TAG_RE = re.compile(r"<[^>]+>")
_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_SPECIAL_RE = re.compile(r"[^a-z0-9%$.\s-]")
_SPACE_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Strip HTML, URLs and odd symbols; lowercase; normalize whitespace."""
    text = html.unescape(text or "")
    text = _TAG_RE.sub(" ", text)
    text = _URL_RE.sub(" ", text)
    text = text.lower()
    text = _SPECIAL_RE.sub(" ", text)
    return _SPACE_RE.sub(" ", text).strip()


def tokenize(text: str) -> list[str]:
    """Split cleaned text into word tokens (keeps $ and % so figures stay intact)."""
    tokens = re.findall(r"[a-z]+|\$?\d[\d,.]*%?", text)
    return [t for t in tokens if t]


def lemmatize(tokens: list[str]) -> list[str]:
    if not _USE_NLTK or _lemmatizer is None:
        return tokens
    return [_lemmatizer.lemmatize(t) for t in tokens]


def preprocess(text: str, remove_stopwords: bool = True) -> list[str]:
    """Full pipeline: clean -> tokenize -> (stop words) -> lemmatize."""
    tokens = tokenize(clean_text(text))
    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS]
    return lemmatize(tokens)


def preprocess_to_string(text: str) -> str:
    """Convenience for scikit-learn vectorizers."""
    return " ".join(preprocess(text))


def split_sentences(text: str) -> list[str]:
    """Simple sentence splitter that avoids breaking on decimals like 4.1%."""
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z\"])", (text or "").strip())
    return [p.strip() for p in parts if p.strip()]
