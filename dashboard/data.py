import json
import joblib
import pandas as pd
import streamlit as st

from dashboard.config import CLUSTERED_DATA_PATH, PIPELINE_PATH, RECOMMENDER_META_PATH


@st.cache_resource
def load_recommender_pipeline():
    return joblib.load(PIPELINE_PATH)


@st.cache_data
def load_recommender_meta():
    if not RECOMMENDER_META_PATH.is_file():
        return None
    return json.loads(RECOMMENDER_META_PATH.read_text(encoding="utf-8"))


@st.cache_data
def load_clustered_data():
    return pd.read_csv(CLUSTERED_DATA_PATH)
