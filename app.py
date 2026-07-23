import streamlit as st
import json
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
PERSONAS = {
    "Book lover":         {"reading", "library", "light", "sun", "glare"},
    "Parent / kids":      {"kids", "son", "daughter", "grandson", "year old", "games"},
    "Movie watcher":      {"movies", "streaming", "netflix", "prime", "video", "watch"},
    "Music / smart home": {"music", "alexa", "sound", "speaker", "fun"},
    "Gift buyer":         {"gift", "christmas", "present"},
    "Budget buyer":       {"price", "value", "cheap", "affordable", "deal"},
}
def best_match(pack, keywords):
    """Return (product, cluster_name, matched_words) or (None, None, set())."""
    best = None
    for cluster in pack["clusters"]:
        for prod in cluster["products"]:
            overlap = set(prod.get("loved_themes", [])) & keywords
            if overlap:
                score = (len(overlap), prod["mean_rating"])
                if best is None or score > best[0]:
                    best = (score, prod, cluster["cluster_name"], overlap)
    if best is None:
        return None, None, set()
    _, prod, cname, overlap = best
    return prod, cname, overlap

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

# ==========================================================
# Helper Functions
# ==========================================================

def explain_prediction(model, review, prediction):
    """Return the most influential words for a prediction."""

    vectorizer = model.named_steps["tfidf"]
    classifier = model.named_steps["clf"]

    review_vector = vectorizer.transform([review])

    feature_names = vectorizer.get_feature_names_out()

    # Find the row of the predicted class
    class_index = list(classifier.classes_).index(prediction)

    coefficients = classifier.coef_[class_index]

    present_indices = review_vector.nonzero()[1]

    contributions = []

    for i in present_indices:

        tfidf = review_vector[0, i]
        coef = coefficients[i]

        score = tfidf * coef

        contributions.append(
            (
                feature_names[i],
                score
            )
        )

    contributions.sort(
        key=lambda x: abs(x[1]),
        reverse=True
    )

    return contributions[:10]

def render_sentiment(model):

    st.subheader("Sentiment Explorer")

    review = st.text_area(
        "Paste a customer review",
        height=180
    )

    if st.button("Analyze Review"):

        if review.strip() == "":
            st.warning("Please enter a review.")
            return

        prediction = model.predict([review])[0]

        important_words = explain_prediction(model, review, prediction)

        # Show prediction ONCE
        if prediction == "Positive":
            st.success("🟢 Predicted Sentiment: Positive")

        elif prediction == "Neutral":
            st.warning("🟡 Predicted Sentiment: Neutral")

        else:
            st.error("🔴 Predicted Sentiment: Negative")

        st.subheader("Top Words Driving the Prediction")

        for word, score in important_words:
            st.write(
                f"• **{word}**  \n"
                f"<span style='color:gray;'>Importance: {abs(score):.2f}</span>",
                unsafe_allow_html=True,
            )



def render_personas(pack, reviews):
    st.subheader("Find a product by who it's for")
    st.caption("Matched from themes mined across 34,624 real reviews — nothing here is AI-generated.")

    choice = st.radio("Who is this for?", list(PERSONAS), horizontal=True)
    prod, cname, matched = best_match(pack, PERSONAS[choice])

    if prod is None:
        st.warning("Nothing in the catalogue matches that persona.")
        return

    # headline
    col1, col2 = st.columns([3, 1])
    col1.markdown(f"### {prod['name']}")
    col1.caption(f"{cname} · {prod['n_reviews']:,} reviews")
    col2.metric("Rating", f"{prod['mean_rating']}/5")

    # why it matched
    st.markdown("**Matched your persona on:** " + ", ".join(f"`{w}`" for w in sorted(matched)))

    # a real quote
    quotes = prod.get("quotes_positive", [])
    if quotes:
        q = quotes[0]
        st.markdown(f"> {q['text']}")
        st.caption(f"— genuine {int(q['rating'])}★ review")


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