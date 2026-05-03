import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict

input_path = r"D:\FourthYear_2\3-Introduction_to_Data_Mining\final_project\step2_preprocessed_dataset.csv"

df = pd.read_csv(input_path)
df = df[["AcademicRecordID", "JournalName", "text_for_model"]].dropna().copy()

print("--- ORIGINAL DATA INFO ---")
print("Shape:", df.shape)
print("Unique journals:", df["JournalName"].nunique())

min_samples_per_journal = 5
journal_counts = df["JournalName"].value_counts()
valid_journals = journal_counts[journal_counts >= min_samples_per_journal].index
df = df[df["JournalName"].isin(valid_journals)].copy()

print("\n--- FILTERED DATA INFO ---")
print("Min samples per journal:", min_samples_per_journal)
print("Shape:", df.shape)
print("Unique journals:", df["JournalName"].nunique())

train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df["JournalName"]
)

print("\n--- SPLIT INFO ---")
print("Train shape:", train_df.shape)
print("Test shape:", test_df.shape)

vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=20000,
    ngram_range=(1, 2),
    min_df=2
)

X_train = vectorizer.fit_transform(train_df["text_for_model"])
X_test = vectorizer.transform(test_df["text_for_model"])

print("\n--- TF-IDF SHAPES ---")
print("X_train:", X_train.shape)
print("X_test :", X_test.shape)

# Tüm similarity değerlerini tek seferde hesapla
similarity_matrix = cosine_similarity(X_test, X_train)

def recommend_from_similarity_row(sim_row, top_k_docs=10, top_k_journals=5):
    top_doc_indices = np.argsort(sim_row)[::-1][:top_k_docs]
    journal_scores = defaultdict(float)

    for idx in top_doc_indices:
        journal = train_df.iloc[idx]["JournalName"]
        score = sim_row[idx]
        journal_scores[journal] += score

    ranked_journals = sorted(journal_scores.items(), key=lambda x: x[1], reverse=True)
    return ranked_journals[:top_k_journals]

print("\n--- SAMPLE RECOMMENDATIONS ---")
for i in range(3):
    sim_row = similarity_matrix[i]
    true_journal = test_df.iloc[i]["JournalName"]
    recommended = recommend_from_similarity_row(sim_row)

    print(f"\nTest item {i+1}")
    print("True journal:", true_journal)
    print("Recommended journals:")
    for rank, (journal, score) in enumerate(recommended, start=1):
        print(f"{rank}. {journal} -> {score:.4f}")

top1_correct = 0
top5_correct = 0

for i in range(len(test_df)):
    sim_row = similarity_matrix[i]
    recommended = recommend_from_similarity_row(sim_row)
    recommended_journals = [journal for journal, score in recommended]
    true_journal = test_df.iloc[i]["JournalName"]

    if recommended_journals and recommended_journals[0] == true_journal:
        top1_correct += 1

    if true_journal in recommended_journals:
        top5_correct += 1

top1_acc = top1_correct / len(test_df)
top5_acc = top5_correct / len(test_df)

print("\n--- EVALUATION ---")
print(f"Top-1 Accuracy: {top1_acc:.4f}")
print(f"Top-5 Accuracy: {top5_acc:.4f}")