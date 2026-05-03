import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

input_path = r"D:\FourthYear_2\3-Introduction_to_Data_Mining\final_project\step5_enriched_dataset.csv"

df = pd.read_csv(input_path)

# gerekli kolonlar
df = df[["AcademicRecordID", "JournalName", "text_rich"]].dropna().copy()

print("--- ORIGINAL DATA INFO ---")
print("Shape:", df.shape)
print("Unique journals:", df["JournalName"].nunique())

# az örnekli journal'ları filtrele
min_samples_per_journal = 5
journal_counts = df["JournalName"].value_counts()
valid_journals = journal_counts[journal_counts >= min_samples_per_journal].index
df = df[df["JournalName"].isin(valid_journals)].copy()

print("\n--- FILTERED DATA INFO ---")
print("Min samples per journal:", min_samples_per_journal)
print("Shape:", df.shape)
print("Unique journals:", df["JournalName"].nunique())

# split
train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df["JournalName"]
)

print("\n--- SPLIT INFO ---")
print("Train shape:", train_df.shape)
print("Test shape :", test_df.shape)

# TF-IDF
vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=20000,
    ngram_range=(1, 2),
    min_df=2
)

X_train = vectorizer.fit_transform(train_df["text_rich"])
X_test = vectorizer.transform(test_df["text_rich"])

y_train = train_df["JournalName"]
y_test = test_df["JournalName"]

print("\n--- TF-IDF SHAPES ---")
print("X_train:", X_train.shape)
print("X_test :", X_test.shape)

# model
# model
model = LogisticRegression(
    max_iter=1000,
    solver="lbfgs"
)

model.fit(X_train, y_train)

print("\nModel training completed.")

# class probabilities
probs = model.predict_proba(X_test)
classes = model.classes_

# Top-1 / Top-5 evaluation
top1_correct = 0
top5_correct = 0

print("\n--- SAMPLE PREDICTIONS ---")
for i in range(3):
    prob_row = probs[i]
    top5_idx = np.argsort(prob_row)[::-1][:5]
    top5_journals = classes[top5_idx]
    top5_scores = prob_row[top5_idx]

    print(f"\nTest item {i+1}")
    print("True journal:", y_test.iloc[i])
    print("Predicted journals:")
    for rank, (journal, score) in enumerate(zip(top5_journals, top5_scores), start=1):
        print(f"{rank}. {journal} -> {score:.4f}")

for i in range(len(y_test)):
    prob_row = probs[i]
    top5_idx = np.argsort(prob_row)[::-1][:5]
    top5_journals = classes[top5_idx]
    true_journal = y_test.iloc[i]

    if top5_journals[0] == true_journal:
        top1_correct += 1

    if true_journal in top5_journals:
        top5_correct += 1

top1_acc = top1_correct / len(y_test)
top5_acc = top5_correct / len(y_test)

print("\n--- EVALUATION ---")
print(f"Top-1 Accuracy: {top1_acc:.4f}")
print(f"Top-5 Accuracy: {top5_acc:.4f}")