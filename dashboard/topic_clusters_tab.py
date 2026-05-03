import streamlit as st

from dashboard.clusters_meta import get_cluster_keywords, infer_cluster_name


def render_topic_clusters_tab(cluster_df):
    st.subheader("Topic Clusters")

    cluster_ids = sorted(cluster_df["cluster"].unique().tolist())
    selected_cluster = st.selectbox(
        "Select a cluster",
        cluster_ids,
        format_func=lambda x: f"{x} - {infer_cluster_name(x)}",
    )

    cluster_subset = cluster_df[cluster_df["cluster"] == selected_cluster].copy()

    st.markdown(f"**Cluster Name:** {infer_cluster_name(selected_cluster)}")

    keywords = get_cluster_keywords(selected_cluster)
    st.markdown("**Top Terms:** " + ", ".join(keywords))

    st.markdown(f"**Number of Articles in Cluster:** {len(cluster_subset)}")

    st.markdown("### Top Journals in This Cluster")
    top_journals = cluster_subset["JournalName"].value_counts().head(10).reset_index()
    top_journals.columns = ["JournalName", "Count"]
    st.dataframe(top_journals, width="stretch")

    st.markdown("### Sample Articles from This Cluster")
    sample_rows = cluster_subset[["AcademicRecordID", "JournalName"]].head(10)
    st.dataframe(sample_rows, width="stretch")
