import pandas as pd
import re
from html import unescape

input_path = r"D:\FourthYear_2\3-Introduction_to_Data_Mining\final_project\step1_basic_joined_dataset.csv"
output_path = r"D:\FourthYear_2\3-Introduction_to_Data_Mining\final_project\step2_preprocessed_dataset.csv"

df = pd.read_csv(input_path)

print("--- INPUT INFO ---")
print("Shape:", df.shape)
print(df.head())

def clean_text(text):
    if pd.isna(text):
        return ""

    # html entity çöz
    text = unescape(str(text))

    # html tag kaldır
    text = re.sub(r"<.*?>", " ", text)

    # lowercase
    text = text.lower()

    # satır sonu / tab temizliği
    text = re.sub(r"[\r\n\t]+", " ", text)

    # harf, sayı ve boşluk dışındakileri temizle
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    # fazla boşlukları temizle
    text = re.sub(r"\s+", " ", text).strip()

    return text

# temiz kolonlar
df["abstract_clean"] = df["AbstractText"].apply(clean_text)
df["title_clean"] = df["Title"].apply(clean_text)

# model için birleşik text
df["text_for_model"] = (df["title_clean"] + " " + df["abstract_clean"]).str.strip()

# metin uzunluğu
df["abstract_word_count"] = df["abstract_clean"].apply(lambda x: len(x.split()))
df["text_word_count"] = df["text_for_model"].apply(lambda x: len(x.split()))

print("\n--- AFTER CLEANING ---")
print(df[["Title", "AbstractText", "abstract_clean"]].head(3))

print("\n--- WORD COUNT STATS ---")
print(df["abstract_word_count"].describe())

# çok kısa metinleri kontrol et
short_count = (df["abstract_word_count"] < 20).sum()
print(f"\n20 kelimeden kısa abstract sayısı: {short_count}")

# istersen çok kısa abstractları filtrele
df_clean = df[df["abstract_word_count"] >= 20].copy()

print("\n--- FILTERED DATASET INFO ---")
print("Shape:", df_clean.shape)
print(df_clean[["JournalName", "abstract_word_count", "text_word_count"]].head())

df_clean.to_csv(output_path, index=False, encoding="utf-8")

print("\nKaydedildi:", output_path)