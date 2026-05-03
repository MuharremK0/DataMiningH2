import json
import joblib
import re
from html import unescape
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

BASE_DIR = Path(__file__).resolve().parent
input_path = BASE_DIR / "step5_enriched_dataset.csv"
pipeline_path = BASE_DIR / "journal_recommender_pipeline.pkl"
meta_path = BASE_DIR / "journal_recommender_meta.json"


def clean_text(text):
    if pd.isna(text) or text is None:
        return ""
    text = unescape(str(text)) # Metindeki HTML kaçış karakterlerini (örneğin &amp; -> &) düzeltir.
    text = re.sub(r"<.*?>", " ", text) # Metin içindeki HTML etiketlerini (örneğin <p> veya <div>) temizler.
    text = text.lower()
    text = re.sub(r"[\r\n\t]+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text) # Sadece küçük harfleri ve sayıları bırakır; tüm noktalama işaretlerini (. , ! ?) siler.
    text = re.sub(r"\s+", " ", text).strip() # Kelimeler arasındaki gereksiz boşlukları temizler
    return text


def build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """title / abstract / keywords / subjects kanalları — ColumnTransformer ile uyumlu."""
    out = pd.DataFrame()
    out["title_channel"] = df["title_clean"].fillna("").map(clean_text)
    out["abstract_channel"] = df["abstract_clean"].fillna("").map(clean_text)
    kw = (
        df["keywords_clean"].fillna("").map(clean_text)
        + " "
        + df["keyword_plus_clean"].fillna("").map(clean_text)
    ).str.strip()
    out["keywords_channel"] = kw
    out["subjects_channel"] = df["subjects_clean"].fillna("").map(clean_text)
    return out


# veri yükle
df = pd.read_csv(input_path)

needed = [
    "JournalName",
    "title_clean",
    "abstract_clean",
    "keywords_clean",
    "keyword_plus_clean",
    "subjects_clean",
]
df = df[needed].dropna(subset=["JournalName", "abstract_clean"]).copy()

# az örnekli journal'ları filtrele
min_samples_per_journal = 5
journal_counts = df["JournalName"].value_counts()
valid_journals = journal_counts[journal_counts >= min_samples_per_journal].index
df = df[df["JournalName"].isin(valid_journals)].copy()
df = df.reset_index(drop=True)

X_df = build_feature_frame(df)
y_series = df["JournalName"]

# çok kısa abstract (gürültü) — step2 ile uyumlu düşünce
min_words = 20
word_counts = X_df["abstract_channel"].str.split().str.len()
mask = word_counts >= min_words
X_df = X_df.loc[mask].reset_index(drop=True)
y = y_series.loc[mask].reset_index(drop=True).to_numpy(dtype=str)

print("--- TRAINING DATA INFO ---", flush=True)
print("Shape:", X_df.shape, flush=True)
print("Unique journals:", len(np.unique(y)), flush=True)

column_transformer = ColumnTransformer(
    transformers=[
        (
            "title",
            TfidfVectorizer(
                stop_words="english",
                max_features=6000,
                ngram_range=(1, 2),
                min_df=2,
                sublinear_tf=True,
            ),
            "title_channel",
        ),
        (
            "abstract",
            TfidfVectorizer(
                stop_words="english",
                max_features=12000,
                ngram_range=(1, 2),
                min_df=2,
                sublinear_tf=True,
            ),
            "abstract_channel",
        ),
        (
            "keywords",
            TfidfVectorizer(
                stop_words="english",
                max_features=6000,
                ngram_range=(1, 2),
                min_df=2,
                sublinear_tf=True,
            ),
            "keywords_channel",
        ),
        (
            "subjects",
            TfidfVectorizer(
                stop_words="english",
                max_features=3000,
                ngram_range=(1, 1),
                min_df=2,
                sublinear_tf=True,
            ),
            "subjects_channel",
        ),
    ],
    remainder="drop",
    sparse_threshold=0.3,
)

# Büyük çok sınıflı seyrek veride LR (lbfgs) çok yavaş olabiliyor; log-loss SGD benzer kaliteyi daha hızlı verir.
pipeline = Pipeline(
    steps=[
        ("features", column_transformer),
        (
            "clf",
            SGDClassifier(
                loss="log_loss",
                penalty="l2",
                alpha=1e-4,
                max_iter=2000,
                tol=1e-3,
                random_state=42,
                n_jobs=-1,
                early_stopping=True,
                validation_fraction=0.05,
                n_iter_no_change=5,
            ),
        ),
    ]
)


def top1_top5_accuracy(model, X, y_labels):
    """Çok sınıflı dergi tahmini: Top-1 (accuracy) ve Top-5 isabet oranı."""
    y_pred = model.predict(X)
    top1 = accuracy_score(y_labels, y_pred)
    proba = model.predict_proba(X)
    classes = model.classes_
    top5_hits = 0
    n = len(y_labels)
    for i in range(n):
        idx = np.argsort(proba[i])[::-1][:5]
        top5_labels = set(classes[idx])
        if y_labels[i] in top5_labels:
            top5_hits += 1
    top5 = top5_hits / n if n else 0.0
    return top1, top5


X_train, X_test, y_train, y_test = train_test_split(
    X_df, y, test_size=0.2, random_state=42, stratify=y
)

pipeline.fit(X_train, y_train)

train_top1, train_top5 = top1_top5_accuracy(pipeline, X_train, y_train)
test_top1, test_top5 = top1_top5_accuracy(pipeline, X_test, y_test)

print("\n--- METRICS (stratified 80/20, same pipeline) ---", flush=True)
print(
    f"{'':12} {'Top-1':>10} {'Top-5':>10}   (örnek sayısı)",
    flush=True,
)
print(
    f"{'Train':12} {train_top1:10.4f} {train_top5:10.4f}   n={len(y_train)}",
    flush=True,
)
print(
    f"{'Test':12} {test_top1:10.4f} {test_top5:10.4f}   n={len(y_test)}",
    flush=True,
)
print(
    "Not: Train skorları aynı veri üzerinde ölçülür (beklenen olarak testten yüksek olabilir).\n",
    flush=True,
)

meta = {
    "n_articles": int(X_df.shape[0]),
    "n_journals": int(len(np.unique(y))),
    "min_words_abstract": min_words,
    "min_samples_per_journal": min_samples_per_journal,
    "holdout_test_size": 0.2,
    "holdout_random_state": 42,
    "train_n_samples": int(len(y_train)),
    "train_top1_accuracy": float(train_top1),
    "train_top5_accuracy": float(train_top5),
    "test_n_samples": int(len(y_test)),
    "holdout_top1_accuracy": float(test_top1),
    "holdout_top5_accuracy": float(test_top5),
}
meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
print("Meta saved:", meta_path, flush=True)

pipeline.fit(X_df, y)
print("Final pipeline retrained on full data.", flush=True)

joblib.dump(pipeline, pipeline_path)
print("Pipeline saved:", pipeline_path, flush=True)


def recommend_journals(
    abstract: str,
    title: str = "",
    keywords: str = "",
    subjects: str = "",
    top_k: int = 5,
):
    row = pd.DataFrame(
        [
            {
                "title_channel": clean_text(title),
                "abstract_channel": clean_text(abstract),
                "keywords_channel": clean_text(keywords),
                "subjects_channel": clean_text(subjects),
            }
        ]
    )
    probs = pipeline.predict_proba(row)[0]
    classes = pipeline.classes_
    top_idx = np.argsort(probs)[::-1][:top_k]
    return [(classes[i], probs[i]) for i in top_idx]


sample_abstract = """
This paper proposes a machine learning framework for discovering patterns in large-scale structured and unstructured data.
The study focuses on knowledge discovery, classification, feature extraction, and predictive modeling, with experiments on real-world datasets.
"""

print("\n--- SAMPLE (abstract only) ---")
for rank, (journal, score) in enumerate(
    recommend_journals(sample_abstract, top_k=5), start=1
):
    print(f"{rank}. {journal} -> {score:.4f}")
