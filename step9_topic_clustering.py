import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

input_path = r"D:\FourthYear_2\3-Introduction_to_Data_Mining\final_project\step5_enriched_dataset.csv"
output_path = r"D:\FourthYear_2\3-Introduction_to_Data_Mining\final_project\step9_clustered_dataset.csv"

df = pd.read_csv(input_path)

# gerekli kolonlar
df = df[["AcademicRecordID", "JournalName", "text_rich"]].dropna().copy()

# az örnekli journal filtresi burada zorunlu değil ama istersen kullanabilirsin
min_samples_per_journal = 5
journal_counts = df["JournalName"].value_counts()
valid_journals = journal_counts[journal_counts >= min_samples_per_journal].index
df = df[df["JournalName"].isin(valid_journals)].copy()

print("--- DATA INFO ---")
print("Shape:", df.shape)
print("Unique journals:", df["JournalName"].nunique())

# TF-IDF
vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=5000,
    ngram_range=(1, 2),
    min_df=3
)

X = vectorizer.fit_transform(df["text_rich"])

print("\nTF-IDF shape:", X.shape)

# KMeans
n_clusters = 10
kmeans = KMeans(
    n_clusters=n_clusters,
    random_state=42,
    n_init=10
)

df["cluster"] = kmeans.fit_predict(X)

print("\n--- CLUSTER COUNTS ---")
print(df["cluster"].value_counts().sort_index())

# top terimler
feature_names = vectorizer.get_feature_names_out()
order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]

print("\n--- TOP TERMS PER CLUSTER ---")
for i in range(n_clusters):
    top_terms = [feature_names[ind] for ind in order_centroids[i, :12]]
    print(f"\nCluster {i}:")
    print(", ".join(top_terms))

# her cluster için örnek journal dağılımı
print("\n--- TOP JOURNALS PER CLUSTER ---")
for i in range(n_clusters):
    print(f"\nCluster {i}:")
    print(df[df["cluster"] == i]["JournalName"].value_counts().head(10))

df.to_csv(output_path, index=False, encoding="utf-8")
print("\nSaved to:", output_path)