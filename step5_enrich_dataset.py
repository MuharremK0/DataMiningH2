import sqlite3
import pandas as pd
import re
from html import unescape

db_path = r"D:\FourthYear_2\3-Introduction_to_Data_Mining\final_project\!sources\CompSciencePub.sqlite"
output_path = r"D:\FourthYear_2\3-Introduction_to_Data_Mining\final_project\step5_enriched_dataset.csv"

conn = sqlite3.connect(db_path)

query = """
SELECT
    ar.AcademicRecordID,
    ar.Title,
    ara.AbstractText,
    ar.PublicationId,
    p.Name AS JournalName,
    ar.PubYear,

    kw.keywords_text,
    kp.keyword_plus_text,
    subj.subjects_text

FROM AcademicRecord ar

LEFT JOIN AcademicRecordAbstract ara
    ON ar.AcademicRecordID = ara.AcademicRecordId

LEFT JOIN Publication p
    ON ar.PublicationId = p.PublicationID

LEFT JOIN (
    SELECT
        ark.AcademicRecordId,
        GROUP_CONCAT(ak.Name, ' ') AS keywords_text
    FROM AcademicRecordKeyword ark
    LEFT JOIN AcademicKeyword ak
        ON ark.AcademicKeywordId = ak.AcademicKeywordID
    GROUP BY ark.AcademicRecordId
) kw
    ON ar.AcademicRecordID = kw.AcademicRecordId

LEFT JOIN (
    SELECT
        arkp.AcademicRecordId,
        GROUP_CONCAT(akp.Name, ' ') AS keyword_plus_text
    FROM AcademicRecordKeywordPlus arkp
    LEFT JOIN AcademicKeywordPlus akp
        ON arkp.AcademicKeywordPlusId = akp.AcademicKeywordPlusID
    GROUP BY arkp.AcademicRecordId
) kp
    ON ar.AcademicRecordID = kp.AcademicRecordId

LEFT JOIN (
    SELECT
        ars.AcademicRecordId,
        GROUP_CONCAT(asu.NameEn, ' ') AS subjects_text
    FROM AcademicRecordSubject ars
    LEFT JOIN AcademicSubject asu
        ON ars.AcademicSubjectId = asu.AcademicSubjectID
    GROUP BY ars.AcademicRecordId
) subj
    ON ar.AcademicRecordID = subj.AcademicRecordId
"""

df = pd.read_sql_query(query, conn)
conn.close()

print("--- RAW ENRICHED DATASET ---")
print("Shape:", df.shape)
print(df.head())
print(df.isnull().sum())

# temel filtre
df = df.dropna(subset=["AbstractText", "JournalName"]).copy()
df = df.drop_duplicates(subset=["AcademicRecordID"]).copy()

def clean_text(text):
    if pd.isna(text):
        return ""
    text = unescape(str(text))
    text = re.sub(r"<.*?>", " ", text)
    text = text.lower()
    text = re.sub(r"[\r\n\t]+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# temiz kolonlar
df["title_clean"] = df["Title"].apply(clean_text)
df["abstract_clean"] = df["AbstractText"].apply(clean_text)
df["keywords_clean"] = df["keywords_text"].apply(clean_text)
df["keyword_plus_clean"] = df["keyword_plus_text"].apply(clean_text)
df["subjects_clean"] = df["subjects_text"].apply(clean_text)

# iki farklı temsil
df["text_title_abstract"] = (
    df["title_clean"] + " " + df["abstract_clean"]
).str.strip()

df["text_rich"] = (
    df["title_clean"] + " " +
    df["abstract_clean"] + " " +
    df["keywords_clean"] + " " +
    df["keyword_plus_clean"] + " " +
    df["subjects_clean"]
).str.strip()

df["rich_word_count"] = df["text_rich"].apply(lambda x: len(x.split()))

print("\n--- CLEAN ENRICHED DATASET ---")
print("Shape:", df.shape)
print(df[[
    "JournalName",
    "keywords_clean",
    "keyword_plus_clean",
    "subjects_clean",
    "text_rich"
]].head(3))

df.to_csv(output_path, index=False, encoding="utf-8")
print("\nSaved to:", output_path)