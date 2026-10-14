"""Check that your .env / API key works:  python check_llm.py"""
import os
import sys

from nlp import llm  # loads .env on import


def main() -> int:
    if not llm.llm_available():
        print("GROK_API_KEY is not set.")
        print("  1. Copy .env.example to .env   2. Paste your key after GROK_API_KEY=   3. Save and re-run.")
        print("The app still works without it, in offline mode.")
        return 1
    print(f"Key found. Calling Groq with model '{llm.MODEL}' ...")
    try:
        reply = llm._call_grok("Reply with the single word: OK", max_tokens=20)
    except Exception as e:
        print(f"FAILED: {type(e).__name__}: {e}")
        print("Common causes: wrong or expired key, or a model name your account can't use (change FINANCE_LLM_MODEL).")
        return 2
    print(f"Success. Model replied: {reply!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
