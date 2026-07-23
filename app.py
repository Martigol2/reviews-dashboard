import streamlit as st

st.set_page_config(
    page_title="Amazon Reviews Dashboard",
    page_icon="⭐",
    layout="wide",
)

# ==========================================================
# Cached Loaders
# ==========================================================

@st.cache_data
def load_pack():
    return None


@st.cache_data
def load_articles():
    return None


@st.cache_data
def load_reviews():
    return None


@st.cache_resource
def load_model():
    return None


# ==========================================================
# Tabs
# ==========================================================

def render_summaries(pack, articles):
    st.info("🚧 Category Summaries - Coming soon")


def render_sentiment(model):
    st.info("🚧 Sentiment Explorer - Coming soon")


def render_personas(pack, reviews):
    st.info("🚧 Customer Personas - Coming soon")


# ==========================================================
# Main
# ==========================================================

def main():

    st.set_page_config(
        page_title="Amazon Reviews Dashboard",
        page_icon="⭐",
        layout="wide",
    )

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