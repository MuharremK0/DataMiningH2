import streamlit as st

from dashboard.recommender_logic import recommend_journals


def render_journal_recommender_tab(recommender_pipeline):
    st.subheader("Journal Recommendation")

    user_input = st.text_area(
        "Makale özeti (abstract)",
        height=250,
        placeholder="Abstract metnini buraya yapıştırın...",
    )

    with st.expander("İsteğe bağlı: başlık, anahtar kelimeler, konu (model eğitimiyle hizalı tahmin)"):
        opt_title = st.text_input("Başlık", placeholder="Boş bırakılabilir")
        opt_keywords = st.text_area(
            "Anahtar kelimeler (virgül veya boşlukla)",
            height=80,
            placeholder="Örn: deep learning, graph neural networks",
        )
        opt_subjects = st.text_area(
            "Konu / subject (İngilizce terimler)",
            height=80,
            placeholder="Örn: Artificial Intelligence, Data Mining",
        )

    recommend_clicked = st.button("Recommend Journals")

    if recommend_clicked:
        if not user_input.strip():
            st.warning("Lütfen bir abstract girin.")
        else:
            recommendations = recommend_journals(
                recommender_pipeline,
                abstract=user_input,
                title=opt_title,
                keywords=opt_keywords,
                subjects=opt_subjects,
                top_k=5,
            )

            st.success("En uygun 5 dergi listelendi.")

            for i, rec in enumerate(recommendations, start=1):
                st.markdown(
                    f"""
                    **{i}. {rec['journal']}**  
                    Score: `{rec['score']:.4f}`
                    """
                )
                st.progress(min(rec["score"] * 10, 1.0))
