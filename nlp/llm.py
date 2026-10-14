"""LLM integration: summaries, insights and Q&A over a finance article.

If GROK_API_KEY is set, requests go to the xAI Grok API (OpenAI-compatible). Otherwise the app
falls back to a clearly labeled offline mode (extractive summary + keyword
retrieval) so the prototype still demonstrates the full workflow without a key.
"""
import os
import re
from pathlib import Path

ENV_PATH = Path(__file__).resolve().parent.parent / ".env"


def _load_env_file(path: Path = ENV_PATH) -> None:
    """Load KEY=VALUE lines from .env. Uses python-dotenv if installed, else a tiny built-in parser."""
    try:
        from dotenv import load_dotenv

        load_dotenv(path)
        return
    except ImportError:
        pass
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip().removeprefix("export ").strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        os.environ.setdefault(key, value)  # real environment variables win


_load_env_file()

from .preprocess import preprocess, split_sentences

MODEL = os.environ.get("FINANCE_LLM_MODEL", "grok-4.7")
XAI_BASE_URL = "https://api.groq.com/openai/v1"
MAX_ARTICLE_CHARS = 12000  # keep prompts within a safe length

SYSTEM_PROMPT = (
    "You are a financial news analyst assistant. Answer using ONLY the article "
    "text provided. If the article does not contain the answer, say so plainly. "
    "Be concise and specific (cite figures from the article when relevant). "
    "You are not a licensed financial advisor: do not give personalized "
    "investment advice, and end insight lists with a one-line reminder that "
    "this is informational, not financial advice."
)

TASK_PROMPTS = {
    "summarize": "Summarize the article in 3 sentences or fewer for a busy analyst.",
    "insights": (
        "List 4-5 bullet-point key takeaways, including the main risks and the main "
        "positives, and note what an analyst might watch next. Keep each bullet to one line."
    ),
}


def _api_key() -> str:
    """The xAI key, read from the GROK_API_KEY environment variable (or .env file)."""
    return os.environ.get("GROK_API_KEY", "").strip()


def llm_available() -> bool:
    return bool(_api_key())


def _truncate(text: str) -> str:
    return text if len(text) <= MAX_ARTICLE_CHARS else text[:MAX_ARTICLE_CHARS] + " [truncated]"


def _context_block(article: str, nlp: dict | None) -> str:
    ctx = ""
    if nlp:
        ctx = (
            f"Pre-computed NLP signals (rule-based, may be imperfect): "
            f"sentiment={nlp.get('sentiment_label')} (score {nlp.get('sentiment_score')}), "
            f"sector={nlp.get('category')}.\n\n"
        )
    return f"{ctx}<article>\n{_truncate(article)}\n</article>"


def _call_grok(user_content: str, max_tokens: int = 500) -> str:
    """Call xAI's Responses API through the OpenAI SDK."""
    from openai import OpenAI

    client = OpenAI(api_key=_api_key(), base_url=XAI_BASE_URL, timeout=60)
    resp = client.responses.create(
        model=MODEL,
        instructions=SYSTEM_PROMPT,
        input=user_content,
        max_output_tokens=max_tokens,
    )
    return (resp.output_text or "").strip()


# ---------- offline fallback ----------

def _rank_sentences(sentences: list[str], query_tokens: set[str] | None = None) -> list[tuple[float, int, str]]:
    """Score sentences by term overlap (query) or content-word density plus figures."""
    scored = []
    for i, s in enumerate(sentences):
        toks = set(preprocess(s))
        if query_tokens:
            score = len(toks & query_tokens) / (len(query_tokens) ** 0.5 or 1)
        else:
            score = len(toks) / 20 + (0.5 if re.search(r"\d", s) else 0) + (0.3 if i == 0 else 0)
        scored.append((score, i, s))
    return sorted(scored, key=lambda x: x[0], reverse=True)


def _offline(task: str, article: str, question: str | None = None) -> str:
    sentences = split_sentences(article)
    if not sentences:
        return "No text to analyze."
    if task == "ask":
        q = set(preprocess(question or ""))
        top = [x for x in _rank_sentences(sentences, q)[:2] if x[0] > 0]
        if not top:
            return "I couldn't find anything in the article that addresses that question."
        return " ".join(s for _, _, s in sorted(top, key=lambda x: x[1]))
    n = 3 if task == "summarize" else 5
    top = sorted(_rank_sentences(sentences)[:n], key=lambda x: x[1])
    if task == "summarize":
        return " ".join(s for _, _, s in top)
    bullets = "\n".join(f"- {s}" for _, _, s in top)
    return bullets + "\n\nInformational only, not financial advice."


def run_llm(task: str, article: str, question: str | None = None, nlp: dict | None = None) -> dict:
    """task in {'summarize', 'insights', 'ask'}. Returns {'answer', 'mode'}."""
    if task == "ask":
        prompt = f"{_context_block(article, nlp)}\n\nQuestion: {question}"
    else:
        prompt = f"{_context_block(article, nlp)}\n\n{TASK_PROMPTS[task]}"

    if llm_available():
        try:
            return {"answer": _call_grok(prompt), "mode": f"llm:{MODEL}"}
        except Exception as e:  # network/key/model issues shouldn't crash the demo
            fallback = _offline(task, article, question)
            return {"answer": fallback, "mode": "offline-fallback", "warning": f"LLM call failed: {type(e).__name__}"}
    return {"answer": _offline(task, article, question), "mode": "offline"}
