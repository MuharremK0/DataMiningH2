# Veri Madenciliği Final Projesi — Teknik Rehber ve Dokümantasyon

Bu belge, projedeki **adım dosyalarının (Step 1–9)** işlevlerini özetler; **ara çıktılar**, **model çıktıları** ve **Streamlit uygulaması** hakkında teknik notlar içerir. 

---

## Veri kaynağı ve ek materyal

| Konum | Açıklama |
|--------|-----------|
| `!sources/CompSciencePub.sqlite` | Akademik kayıtlar, özetler, dergiler, anahtar kelime ve konu alanları için SQLite veritabanı. Birleştirme ve zenginleştirme adımları bu kaynağa dayanır. |
| `!ekler/yapilanlar.txt` | Proje sürecinin kısa özeti (Türkçe); rapor veya sunuma ek referans olarak kullanılabilir. |

---

## Step 1 — Veriyi keşfetme ve birleştirme

### `step1.py`

- Veritabanı dosyasının varlığı ve boyutunu doğrular.
- `sqlite_master` üzerinden **tablo listesini** üretir.
- Ana çalışma veri setini oluşturmaz; keşif ve doğrulama amaçlı yardımcı script.

### `step1.1.py`

- `AcademicRecord`, `AcademicRecordAbstract`, `Publication` tablolarını **JOIN** eder.
- Çıktı: makale kimliği, başlık, özet, dergi adı (`JournalName`), yıl vb.
- Özet veya dergi adı eksik kayıtlar ile yinelenen `AcademicRecordID` satırları temizlenir.
- **Çıktı dosyası:** `step1_basic_joined_dataset.csv`

### `step1.ipynb`

- Jupyter ortamında tablolar ve veri üzerinde **etkileşimli keşif** için kullanılabilir (içerik analiz planına göre güncellenebilir).

---

## Step 2 — Metin ön işleme

### `step2.py`

- Girdi: `step1_basic_joined_dataset.csv`
- Başlık ve özet için **HTML/varlık temizliği**, küçük harfe çevirme, özel karakter ve boşluk normalizasyonu.
- `text_for_model`: başlık ile özetin birleşimi.
- İsteğe bağlı: **20 kelimeden kısa** özetleri eleme.
- **Çıktı dosyası:** `step2_preprocessed_dataset.csv`

---

## Step 3 — İlk baseline: benzer makale → dergi önerisi

### `step3_baseline.py`

- Girdi: `step2_preprocessed_dataset.csv` (`text_for_model`).
- Dergi başına minimum örnek eşiği (`min_samples_per_journal = 5`) ile seyrek sınıflar elenir.
- Eğitim / test ayrımı (`JournalName` üzerinden stratify).
- **TF-IDF** vektörleri ile test makalesine **kosinüs benzerliği** üzerinden en yakın eğitim örnekleri; bunların dergilerinden **Top-5 dergi** önerisi.
- Sınıflandırıcı değil; **içerik tabanlı benzerlik (retrieval)** yaklaşımıdır.

---

## Step 4 — Metin temsillerini karşılaştırma

### `step4_compare_texts.py`

- Girdi: `step2_preprocessed_dataset.csv`
- Aynı bölünme ve filtre ile **farklı metin alanları** karşılaştırılır (ör. yalnızca özet ile başlık+özet `text_for_model`).
- Amaç: dergi eşlemesi için hangi temsilin daha uygun olduğunu **sayısal** olarak karşılaştırmak.

---

## Step 5 — Zenginleştirilmiş veri seti (ER diyagramına uyumlu join)

### `step5_enrich_dataset.py`

- SQLite üzerinden `AcademicRecord` merkezli sorgu: özet, yayın (dergi), **keyword**, **keyword plus**, **subject** (junction tablolar ve `GROUP_CONCAT`).
- Metin alanları aynı kurallarla temizlenir.
- `text_title_abstract`: başlık + özet.
- **`text_rich`**: başlık + özet + keyword + keyword plus + subject birleşimi.
- **Çıktı dosyası:** `step5_enriched_dataset.csv`

---

## Step 6 — Zengin metin ile başlık+özet karşılaştırması

### `step6_compare_rich_text.py`

- Girdi: `step5_enriched_dataset.csv`
- `text_title_abstract` ile **`text_rich`** arasında Step 3–4 ile uyumlu **retrieval / benzerlik** karşılaştırması.
- Amaç: ER’deki ek alanların (keyword, subject) **öneri kalitesine** etkisini göstermek.

---

## Step 7 — Lojistik regresyon baseline (tek metin sütunu)

### `step7_logreg_baseline.py`

- Girdi: `step5_enriched_dataset.csv` — `text_rich`, `JournalName`.
- TF-IDF + **Logistic Regression** ile **çok sınıflı dergi sınıflandırması**.
- Step 8 öncesi **sınıflandırıcı referans çizgisi**; ayrı bir “final pipeline” dosyası üretmeyebilir (proje sürümüne bağlı).

---

## Step 8 — Final dergi öneri modeli (çok kanallı pipeline)

### `step8_final_recommender.py`

- Girdi: `step5_enriched_dataset.csv`
- **Dört metin kanalı:** başlık, özet, keyword/keyword+, subject — her biri için ayrı **TF-IDF** (`ColumnTransformer`), ardından **`Pipeline`** içinde **SGDClassifier(loss="log_loss")** (çok sınıflı, olasılık çıktısı; büyük ölçekte LR’ye göre genelde daha hızlı yakınsama).
- Dergi başına minimum örnek, özet uzunluğu vb. filtreler tutarlılık için uygulanır.
- Holdout metrikleri konsola yazdırılır; tüm veriyle yeniden eğitim sonrası:
- **Çıktı dosyası:** `journal_recommender_pipeline.pkl` (vektörleştirme + sınıflandırıcı)

> **Not:** `journal_recommender_model.pkl` ve `journal_recommender_vectorizer.pkl` dosyaları önceki tek-sütun `text_rich` düzenine aitti. Güncel arayüz **`journal_recommender_pipeline.pkl`** kullanır.

---

## Step 9 — Konu kümeleme (topic / cluster)

### `step9_topic_clustering.py`

- Girdi: `step5_enriched_dataset.csv` (`text_rich`, kimlik ve dergi bilgisi).
- TF-IDF → **KMeans** (`n_clusters = 10`).
- Konsola küme başına örnek terimler ve dergi dağılımları yazdırılır.
- **Çıktı dosyası:** `step9_clustered_dataset.csv` (makale başına `cluster` etiketi)

---

## Arayüz (demo)

### `app.py`

- **Streamlit** uygulaması.
- **Sekme 1 — Journal Recommender:** `journal_recommender_pipeline.pkl` yüklenir; kullanıcı özet girer; isteğe bağlı başlık, keyword ve subject alanları genişletilebilir panelden girilebilir (eğitimdeki kanallarla uyumlu).
- **Sekme 2 — Topic Clusters:** `step9_clustered_dataset.csv` üzerinden küme seçimi, örnek dergi ve makale listeleri.

**Çalıştırma:**

```bash
py -m streamlit run app.py
```

Öneri sekmesinin çalışması için `step8_final_recommender.py` ile üretilmiş güncel `journal_recommender_pipeline.pkl` dosyasının mevcut olması gerekir.

---

## Ara çıktı ve model dosyaları (özet)

| Dosya | Kaynak adım | Kısa açıklama |
|--------|-------------|----------------|
| `step1_basic_joined_dataset.csv` | Step 1.1 | Temel join + temizlik |
| `step2_preprocessed_dataset.csv` | Step 2 | Temiz metin + uzunluk filtreli set |
| `step5_enriched_dataset.csv` | Step 5 | Keyword/subject zenginleştirilmiş set |
| `step9_clustered_dataset.csv` | Step 9 | KMeans küme etiketli set |
| `journal_recommender_pipeline.pkl` | Step 8 | Canlı öneri için kayıtlı pipeline |

---

## Önerilen çalıştırma sırası (sıfırdan üretim)

Veritabanı yolu doğruysa tipik sıra:

1. `step1.1.py` → `step1_basic_joined_dataset.csv`
2. `step2.py` → `step2_preprocessed_dataset.csv`
3. İsteğe bağlı: `step3`, `step4`, `step6`, `step7` (karşılaştırma ve raporlama)
4. `step5_enrich_dataset.py` → `step5_enriched_dataset.csv`
5. `step8_final_recommender.py` → `journal_recommender_pipeline.pkl`
6. `step9_topic_clustering.py` → `step9_clustered_dataset.csv`
7. `py -m streamlit run app.py` veya `streamlit run app.py`