import streamlit as st

from dashboard.data import (
    load_clustered_data,
    load_recommender_meta,
    load_recommender_pipeline,
)
from dashboard.journal_recommender_tab import render_journal_recommender_tab
from dashboard.topic_clusters_tab import render_topic_clusters_tab

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Computer Science Journal Finder",
    page_icon="📚",
    layout="wide",
)

# -----------------------------
# LOAD
# -----------------------------
recommender_pipeline = load_recommender_pipeline()
recommender_meta = load_recommender_meta()
cluster_df = load_clustered_data()

# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.title("Project Info")
st.sidebar.markdown(
    "**Model:** çok kanallı TF-IDF + lineer SGD (log-loss, çok sınıf)"
)
st.sidebar.markdown(
    "**Kanallar:** başlık · özet · keyword/keyword+ · subject"
)

n_journals_model = len(recommender_pipeline.named_steps["clf"].classes_)
st.sidebar.markdown(f"**Dergi sayısı (sınıf):** {n_journals_model}")

if recommender_meta:
    n_art = recommender_meta["n_articles"]
    mw = recommender_meta["min_words_abstract"]
    t1 = recommender_meta["holdout_top1_accuracy"]
    t5 = recommender_meta["holdout_top5_accuracy"]
    st.sidebar.markdown(
        f"**Makale (pipeline eğitimi):** {n_art:,} (abstract ≥{mw} kelime, "
        f"dergi başına ≥{recommender_meta['min_samples_per_journal']} örnek)"
    )
    st.sidebar.markdown(
        f"**Holdout (80/20):** Top-1 = {t1:.4f} · Top-5 = {t5:.4f} "
        "(`journal_recommender_meta.json`, son `step8` çalıştırması)"
    )
    if recommender_meta.get("n_journals") != n_journals_model:
        st.sidebar.caption(
            "Uyarı: meta dosyasındaki dergi sayısı ile model sınıfları eşleşmiyor; "
            "`step8` sonrası meta güncellenmemiş olabilir."
        )
else:
    st.sidebar.markdown(
        "**Makale / holdout:** `journal_recommender_meta.json` yok — sayılar için "
        "`step8_final_recommender.py` çalıştırın (sidebar’daki dergi sayısı yüklü modelden)."
    )

st.sidebar.markdown("---")
st.sidebar.markdown("**Project Features**")
st.sidebar.markdown("- Journal Recommendation")
st.sidebar.markdown("- Topic Clustering")
st.sidebar.markdown("- Top-5 Journal Prediction")

# -----------------------------
# MAIN TITLE
# -----------------------------
st.title("📚 Computer Science Journal Finder")
st.write(
    "Özet (abstract) girerek en uygun 5 dergiyi alın. İsterseniz genişletilmiş alanlara "
    "başlık, anahtar kelime ve konu da ekleyerek eğitimle daha uyumlu tahmin alabilirsiniz."
)

# -----------------------------
# TABS
# -----------------------------
tab1, tab2 = st.tabs(["Journal Recommender", "Topic Clusters"])

with tab1:
    render_journal_recommender_tab(recommender_pipeline)

with tab2:
    render_topic_clusters_tab(cluster_df)
