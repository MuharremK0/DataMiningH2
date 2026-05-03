import re
from html import unescape

import numpy as np
import pandas as pd


def clean_text(text):
    if text is None:
        return ""
    text = unescape(str(text))
    text = re.sub(r"<.*?>", " ", text)
    text = text.lower()
    text = re.sub(r"[\r\n\t]+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def recommend_journals(
    pipeline,
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
    return [
        {"journal": classes[i], "score": float(probs[i])} for i in top_idx
    ]
