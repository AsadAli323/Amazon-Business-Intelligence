import streamlit as st

from common import load_metrics, load_sample, setup

setup("Home")
df = load_sample()
m = load_metrics()

st.markdown(
    """<div class="hero"><h1>🛒 ShelfSense</h1>
    <p>Demand &amp; price intelligence for online electronics listings.
    Explore the market, forecast how well a product will sell, and sanity-check its price.</p></div>""",
    unsafe_allow_html=True,
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Products in demo sample", f"{len(df):,}")
c2.metric("Categories", df["category"].nunique())
c3.metric("Median price", f"${df['price'].median():,.0f}")
c4.metric("Demand model macro-F1", f"{m['demand']['results'][m['demand']['best_model']]['macro_f1']:.2f}")

st.subheader("What can I do here?")
a, b, c = st.columns(3)
cards = [
    (a, "📊 Market Explorer", "Filter by category, brand and price, then see how rating, reviews, "
                              "discounts and sponsorship relate to sales volume."),
    (b, "🔮 Demand Predictor", "Describe a listing and get a Low / Medium / High monthly demand "
                               "forecast with class probabilities."),
    (c, "💲 Price Estimator", "Type a product title and get a fair-price estimate from the model."),
    (a, "🔎 Similar Products", "Find the closest competing listings to any product title."),
    (b, "📁 Batch Scoring", "Upload your own CSV, score every row, and download the results."),
    (c, "📈 Model Report", "See how the models were evaluated, against which baselines."),
]
for col, t, d in cards:
    col.markdown(f'<div class="card"><h4>{t}</h4><p>{d}</p></div>', unsafe_allow_html=True)
    col.write("")

st.info("Use the sidebar to switch pages. Predictions are statistical estimates learned from "
        "listing snapshots collected in August 2025, not guarantees.")
