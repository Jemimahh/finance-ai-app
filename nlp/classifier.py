"""Sector / topic classifier: TF-IDF + Complement Naive Bayes (scikit-learn)."""
import csv
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline

from .preprocess import preprocess_to_string

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "labeled_sectors.csv"


def _load(path: Path):
    texts, labels = [], []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            texts.append(row["text"])
            labels.append(row["sector"])
    return texts, labels


def _build() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(preprocessor=preprocess_to_string, sublinear_tf=True)),
        ("nb", ComplementNB(alpha=0.3)),
    ])


class SectorClassifier:
    def __init__(self, data_path: Path = DATA_PATH):
        self.data_path = Path(data_path)
        self.metrics: dict = {}
        self.model: Pipeline | None = None

    def train(self):
        texts, labels = _load(self.data_path)

        # With a small dataset, 5-fold cross-validation gives a steadier estimate
        # than a single train/test split. Then refit on all data for serving.
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        pred = cross_val_predict(_build(), texts, labels, cv=cv)
        self.metrics = {
            "n_samples": len(texts),
            "cv_folds": 5,
            "accuracy": round(accuracy_score(labels, pred), 3),
            "macro_f1": round(f1_score(labels, pred, average="macro", zero_division=0), 3),
            "report": classification_report(labels, pred, zero_division=0),
        }
        self.model = _build().fit(texts, labels)
        return self

    def predict(self, text: str, top_k: int = 3) -> dict:
        if self.model is None:
            self.train()
        probs = self.model.predict_proba([text])[0]
        classes = self.model.classes_
        ranked = sorted(zip(classes, probs), key=lambda x: x[1], reverse=True)[:top_k]
        return {
            "category": ranked[0][0],
            "confidence": round(float(ranked[0][1]), 3),
            "top": [{"category": c, "probability": round(float(p), 3)} for c, p in ranked],
        }
