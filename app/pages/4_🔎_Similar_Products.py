import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from common import load_sample, setup

setup("Similar Products", "🔎")
st.title("🔎 Similar Products")
st.caption("Find competing listings using text similarity (TF-IDF + cosine) on product titles.")

df = load_sample().drop_duplicates("title").reset_index(drop=True)


@st.cache_resource(show_spinner=False)
def build_index(titles: tuple):
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, stop_words="english", sublinear_tf=True)
    return vec, vec.fit_transform(titles)


vec, mat = build_index(tuple(df["title"]))
q = st.text_input("Describe a product", "wireless noise cancelling bluetooth headphones")
k = st.slider("Results", 3, 20, 8)
if q.strip():
    sims = cosine_similarity(vec.transform([q]), mat).ravel()
    idx = sims.argsort()[::-1][:k]
    out = df.iloc[idx][["title", "brand", "category", "price", "rating", "reviews", "units_sold_month"]].copy()
    out.insert(0, "match", (sims[idx] * 100).round(0).astype(int).astype(str) + "%")
    st.dataframe(out, width="stretch", hide_index=True)
