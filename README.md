# Finance News Analyzer (AI in Finance prototype)

A small proof-of-concept web app. A user picks or pastes a finance news article and gets:

1. **Sentiment** (positive / neutral / negative) from a rule-based model (VADER plus a finance lexicon), with per-sentence scores.
2. **Sector classification** (Banking, Tech Stocks, Energy, Healthcare / Biotech, Retail / Consumer, Macro / Rates, Crypto, Industrials) from a TF-IDF + Naive Bayes model trained with scikit-learn.
3. **LLM output**: a short summary, bullet-point insights, and answers to free-form questions about the article.

The NLP signals are computed first and passed to the LLM as context, so the three layers work as one pipeline.

## Project layout

```
finance-ai-app/
├── app.py                    Flask backend (REST API + page)
├── check_llm.py              checks that your .env / API key works
├── .env.example              template for your keys (copy to .env)
├── .gitignore                keeps .env out of git
├── nlp/
│   ├── preprocess.py         cleaning, tokenizing, stop words, lemmatizing, sentence splitting
│   ├── sentiment.py          VADER + finance lexicon, evaluation helper
│   ├── classifier.py         sector classifier (TF-IDF + Complement Naive Bayes, 5-fold CV)
│   └── llm.py                xAI Grok API calls, with an offline fallback
├── templates/index.html      single-page UI (HTML/CSS/JS)
├── data/
│   ├── articles.json         8 sample articles with date, source, topic metadata
│   ├── labeled_sectors.csv   128 labeled headlines for the sector classifier
│   └── labeled_sentiment.csv 36 labeled headlines for sentiment evaluation
├── tests/test_app.py
└── requirements.txt
```

## Setup walkthrough

You need Python 3.10 or newer. Run everything from the project folder.

**1. Create a virtual environment and install dependencies**

```bash
python -m venv .venv
source .venv/Scripts/activate    
pip install -r requirements.txt
```

**2. Get an GROQ API key** (skip this for offline mode)

Sign in at https://console.groq.com, create an API key.

**3. Create your `.env` file**

```bash
cp .env.example .env               # Windows: copy .env.example .env
```

Open `.env` in any text editor and paste your key after the equals sign, with no quotes and no spaces:

```
GROK_API_KEY=GROQ-your-key-here
FINANCE_LLM_MODEL="openai/gpt-oss-120b"
PORT=5000
```

| Variable | Required | What it does |
|---|---|---|
| `GROK_API_KEY` | For the real LLM | Your xAI key. Empty or missing means offline mode. |
| `FINANCE_LLM_MODEL` | No | Groq model name. Defaults to `openai/gpt-oss-120b`. |
| `PORT` | No | Port for the web app. Defaults to `5000`. |

The `.env` file is listed in `.gitignore`, so it won't be committed by accident. Don't paste your key into code, screenshots, or messages. If a key leaks, delete it in the xAI console and create a new one.

**4. Check that the key works**

```bash
python check_llm.py
```

You should see `Success. Model replied: 'OK'`. If it fails, the message names the likely cause: wrong key, no credits, or a model name your account can't use.

**5. Start the app**

```bash
python app.py
```

The terminal prints which mode it is in (`xAI Grok API (grok-4.7)` or `offline fallback`). Open http://localhost:5000. The footer of the page and the small tag under each LLM answer show the mode too.

**6. Use it**

Pick a sample article (or paste your own, or upload a `.txt` file), click **Analyze**, then **Summarize**, **Key insights**, or type a question and click **Ask**.

### Modes at a glance

| What you see | Meaning |
|---|---|
| `llm:grok-4.7` | The answer came from Grok. |
| `offline` | No key is set. Summaries and insights pick the most relevant sentences; Q&A returns the best-matching sentences. |
| `offline-fallback` + a warning | A key is set but the call failed (bad key, no credits, network). The app kept working and shows why. |

### Troubleshooting

- **Still says offline after adding the key:** make sure the file is named exactly `.env` (not `.env.txt`), sits next to `app.py`, and has no spaces around `=`. Restart the app after editing it.
- **`ModuleNotFoundError`:** the virtual environment isn't active, or `pip install -r requirements.txt` didn't finish.
- **Port already in use:** change `PORT` in `.env`.
- **Authentication or 401/403 errors:** the key is wrong or revoked. Create a new one.
- **Model not found:** set `FINANCE_LLM_MODEL` to a model listed at https://docs.x.ai/docs/models.
- **Corporate or school network blocks the call:** the app falls back to offline mode and still runs.

**Optional NLTK data.** If you have NLTK's `stopwords` and `wordnet` data installed (`python -m nltk.downloader stopwords wordnet`), the preprocessor uses them. If not, it uses a built-in stop-word list and skips lemmatization. The app never downloads anything on its own.

## How to test

```bash
python -m pytest -q
```

The tests always run in offline mode, even if your `.env` contains a real key, so they never spend credits.

Manual check in the browser:
1. Choose "Regional lender Harbor Trust tumbles as loan losses mount" and click **Analyze**. Expect negative sentiment and the Banking sector.
2. Click **Summarize** and **Key insights**.
3. Ask: "What happened to the dividend?" The answer should mention the one-third cut.
4. Paste your own article, or upload a `.txt` file, and repeat.

API endpoints: `GET /api/articles`, `GET /api/status`, `POST /api/analyze` (`{"text": ...}`), `POST /api/llm` (`{"task": "summarize" | "insights" | "ask", "text": ..., "question": ...}`).

## Pipeline details

- **Preprocessing:** strips HTML and URLs, lowercases, removes special characters (keeping `$` and `%` so figures survive), tokenizes, removes stop words (keeping negations and direction words like "down"), lemmatizes when NLTK data is present.
- **Sentiment:** VADER with about 80 added finance terms ("beats", "downgrade", "nonperforming", "probe", and so on). The document score is the average of sentence scores, because whole-text VADER scores saturate on long articles. Thresholds are ±0.15.
- **Sector classifier:** TF-IDF features into Complement Naive Bayes, trained on `labeled_sectors.csv`. Metrics come from 5-fold cross-validation, then the model is refit on all data.
- **LLM:** the system prompt tells the model to answer only from the article, say so when the article lacks the answer, and avoid personalized investment advice. Articles are truncated at 12,000 characters to stay within prompt limits.

## Measured results

| Component | Data | Result |
|---|---|---|
| Sentiment (VADER + finance lexicon) | 36 labeled headlines | 83% accuracy (negative F1 0.96, positive 0.83, neutral 0.70) |
| Sector classifier | 128 labeled headlines, 5-fold CV | 85% accuracy, macro F1 0.85 |

These datasets are small and were written for this assignment, so treat the numbers as a sanity check, not a benchmark. The lexicon was tuned while looking at the sample articles, which also makes the demo articles a friendlier test than real-world text. A real evaluation would use a public set such as the Financial PhraseBank.

## Business use case

**Scenario:** An analyst or news-aggregator team receives hundreds of articles a day and needs to triage them quickly.

**Value:** The app tags each article with sentiment and sector so feeds can be filtered and sorted, then produces a short summary and key points so an analyst can decide in seconds whether an article deserves a full read. The Q&A box lets them ask targeted questions ("What risks does the article mention?") without reading end to end. Possible extensions: portfolio-level sentiment dashboards, alerts when sentiment on a holding turns sharply negative, and risk monitoring across a watchlist.

**Limitations:**
- This is not financial advice and does not replace professional judgment.
- Rule-based sentiment misses context. For example, "rate cut" is good news for some sectors and bad for others, and articles mixing good and bad news often score near neutral (the Skyline Air profit warning scores about 0).
- The sector classifier only knows 8 categories and is trained on short headlines, so long multi-topic articles can be mislabeled.
- LLMs can misread or invent details, and news itself carries bias. Always check figures against the source.
- The sample articles are fictional and for demonstration only.
