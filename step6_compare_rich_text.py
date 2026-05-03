import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict

input_path = r"D:\FourthYear_2\3-Introduction_to_Data_Mining\final_project\step5_enriched_dataset.csv"

df = pd.read_csv(input_path)

# gerekli kolonlar
df = df[[
    "AcademicRecordID",
    "JournalName",
    "text_title_abstract",
    "text_rich"
]].dropna().copy()

# az örnekli journal'ları filtrele
min_samples_per_journal = 5
journal_counts = df["JournalName"].value_counts()
valid_journals = journal_counts[journal_counts >= min_samples_per_journal].index
df = df[df["JournalName"].isin(valid_journals)].copy()

print("--- FILTERED DATA INFO ---")
print("Shape:", df.shape)
print("Unique journals:", df["JournalName"].nunique())

# aynı split her deney için ortak olsun
train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df["JournalName"]
)

print("\n--- SPLIT INFO ---")
print("Train shape:", train_df.shape)
print("Test shape :", test_df.shape)

def evaluate_representation(train_texts, test_texts, train_labels, test_labels, exp_name,
                            max_features=20000, top_k_docs=10, top_k_journals=5):

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=max_features,
        ngram_range=(1, 2),
        min_df=2
    )

    X_train = vectorizer.fit_transform(train_texts)
    X_test = vectorizer.transform(test_texts)

    similarity_matrix = cosine_similarity(X_test, X_train)

    def recommend_from_row(sim_row):
        top_doc_indices = np.argsort(sim_row)[::-1][:top_k_docs]
        journal_scores = defaultdict(float)

        for idx in top_doc_indices:
            journal = train_labels.iloc[idx]
            score = sim_row[idx]
            journal_scores[journal] += score

        ranked_journals = sorted(journal_scores.items(), key=lambda x: x[1], reverse=True)
        return ranked_journals[:top_k_journals]

    top1_correct = 0
    top5_correct = 0

    for i in range(len(test_labels)):
        sim_row = similarity_matrix[i]
        recommended = recommend_from_row(sim_row)
        recommended_journals = [journal for journal, score in recommended]
        true_journal = test_labels.iloc[i]

        if recommended_journals and recommended_journals[0] == true_journal:
            top1_correct += 1

        if true_journal in recommended_journals:
            top5_correct += 1

    top1_acc = top1_correct / len(test_labels)
    top5_acc = top5_correct / len(test_labels)

    return {
        "experiment": exp_name,
        "top1_accuracy": top1_acc,
        "top5_accuracy": top5_acc,
        "train_shape": X_train.shape,
        "test_shape": X_test.shape
    }

results = []

# 1) title + abstract
results.append(
    evaluate_representation(
        train_texts=train_df["text_title_abstract"],
        test_texts=test_df["text_title_abstract"],
        train_labels=train_df["JournalName"],
        test_labels=test_df["JournalName"],
        exp_name="title_plus_abstract"
    )
)

# 2) rich text
results.append(
    evaluate_representation(
        train_texts=train_df["text_rich"],
        test_texts=test_df["text_rich"],
        train_labels=train_df["JournalName"],
        test_labels=test_df["JournalName"],
        exp_name="rich_text"
    )
)

results_df = pd.DataFrame(results)

print("\n--- COMPARISON RESULTS ---")
print(results_df.to_string(index=False))