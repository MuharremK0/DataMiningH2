import sqlite3
import pandas as pd
import os

db_path = r"D:\FourthYear_2\3-Introduction_to_Data_Mining\final_project\!sources\CompSciencePub.sqlite"

print("Dosya var mı?:", os.path.exists(db_path))
print("Dosya boyutu:", os.path.getsize(db_path) if os.path.exists(db_path) else "YOK")

conn = sqlite3.connect(db_path)

tables_query = """
SELECT name
FROM sqlite_master
WHERE type='table'
ORDER BY name;
"""

tables_df = pd.read_sql_query(tables_query, conn)
print(tables_df)