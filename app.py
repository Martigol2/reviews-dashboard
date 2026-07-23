import streamlit as st
import json
import pickle
import joblib
from pathlib import Path

import pandas as pd

st.set_page_config(
    page_title="Amazon Reviews Dashboard",
    page_icon="⭐",
    layout="wide",
)

# ==========================================================
# Paths
# ==========================================================

DATA_PATH = Path("data")

PACK_PATH = DATA_PATH / "evidence_pack.json"
ARTICLES_PATH = DATA_PATH / "articles.json"
REVIEWS_PATH = DATA_PATH / "reviews_slim.parquet"
MODEL_PATH = DATA_PATH / "svm_pipeline.pkl"



# ==========================================================
# Cached Loaders
# ==========================================================

@st.cache_data
def load_pack():
    """Load evidence pack."""

    with open(PACK_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_articles():
    """Load category summary articles."""

    if not ARTICLES_PATH.exists():
        return {}

    with open(ARTICLES_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_reviews():
    """Load review dataset."""

    return pd.read_parquet(REVIEWS_PATH)


@st.cache_resource
def load_model():
    """Load sentiment classification model."""

    return joblib.load(MODEL_PATH)


# ==========================================================
# Tabs
# ==========================================================

def render_summaries(pack, articles):

    clusters = pack["clusters"]

    category_names = [c["cluster_name"] for c in clusters]

    selected_category = st.selectbox(
        "Select Product Category",
        category_names
    )

    cluster = next(
        c for c in clusters
        if c["cluster_name"] == selected_category
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Reviews", cluster["n_reviews"])
    col2.metric("Average Rating", f'{cluster["mean_rating"]:.2f}')
    col3.metric("Positive %", f'{cluster["pct_positive"]:.1f}%')
    col4.metric("Negative %", f'{cluster["pct_negative"]:.1f}%')

    st.divider()

    st.subheader("Category Summary")

    st.markdown(articles[selected_category])


def render_sentiment(model):
    st.info("🚧 Sentiment Explorer - Coming soon")


def render_personas(pack, reviews):
    st.info("🚧 Customer Personas - Coming soon")


# ==========================================================
# Main
# ==========================================================

def main():

    st.title("Amazon Reviews Dashboard")

    pack = load_pack()
    articles = load_articles()
    reviews = load_reviews()
    model = load_model()

    tab1, tab2, tab3 = st.tabs(
        [
            "📊 Category Summaries",
            "😊 Sentiment Explorer",
            "👥 Customer Personas",
        ]
    )

    with tab1:
        render_summaries(pack, articles)

    with tab2:
        render_sentiment(model)

    with tab3:
        render_personas(pack, reviews)


if __name__ == "__main__":
    main()