import sqlite3
import pandas as pd
import os

db_path = r"D:\FourthYear_2\3-Introduction_to_Data_Mining\final_project\!sources\CompSciencePub.sqlite"

print("Dosya var mı?:", os.path.exists(db_path))
print("Dosya boyutu:", os.path.getsize(db_path) if os.path.exists(db_path) else "YOK")

conn = sqlite3.connect(db_path)

# tablo kontrolü
tables_query = """
SELECT name
FROM sqlite_master
WHERE type='table'
ORDER BY name;
"""
tables_df = pd.read_sql_query(tables_query, conn)
print(tables_df)

# örnek veri kontrolü
print("\n--- AcademicRecord sample ---")
print(pd.read_sql_query("SELECT * FROM AcademicRecord LIMIT 3;", conn))

print("\n--- AcademicRecordAbstract sample ---")
print(pd.read_sql_query("SELECT * FROM AcademicRecordAbstract LIMIT 3;", conn))

print("\n--- Publication sample ---")
print(pd.read_sql_query("SELECT * FROM Publication LIMIT 3;", conn))

# ana joined dataset
query = """
SELECT
    ar.AcademicRecordID,
    ar.Title,
    ara.AbstractText,
    ar.PublicationId,
    p.Name AS JournalName,
    ar.PubYear
FROM AcademicRecord ar
LEFT JOIN AcademicRecordAbstract ara
    ON ar.AcademicRecordID = ara.AcademicRecordId
LEFT JOIN Publication p
    ON ar.PublicationId = p.PublicationID
"""

df = pd.read_sql_query(query, conn)

print("\n--- RAW DATASET INFO ---")
print("Shape:", df.shape)
print(df.head())
print(df.isnull().sum())

# temel temizlik
df = df.dropna(subset=["AbstractText", "JournalName"]).copy()
df = df.drop_duplicates(subset=["AcademicRecordID"]).copy()

print("\n--- CLEANED DATASET INFO ---")
print("Shape:", df.shape)
print(df.head())
print(df.isnull().sum())

# csv kaydet
output_path = r"D:\FourthYear_2\3-Introduction_to_Data_Mining\final_project\step1_basic_joined_dataset.csv"
df.to_csv(output_path, index=False, encoding="utf-8")

print("\nCSV kaydedildi:", output_path)

conn.close()