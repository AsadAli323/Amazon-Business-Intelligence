import numpy as np
import pandas as pd
import streamlit as st

from common import CATEGORIES, load_models, load_sample, setup
from shelfsense.cleaning import assign_category
from shelfsense.train import add_features

setup("Price Estimator", "💲")
st.title("💲 Price Estimator")
st.caption("Estimate a fair market price from the product title and basic listing facts.")

_, price_model = load_models()
df = load_sample()

title = st.text_input("Product title", "Apple MacBook Air 13-inch Laptop M3 chip 16GB RAM 512GB SSD")
auto = assign_category(title)
c1, c2, c3 = st.columns(3)
category = c1.selectbox("Category", CATEGORIES, index=CATEGORIES.index(auto),
                        help="Auto-detected from the title; change it if it looks wrong.")
rating = c2.slider("Rating", 1.0, 5.0, 4.6, 0.1)
reviews = c3.number_input("Number of reviews", 0, 1_000_000, 500)
renewed = st.checkbox("Renewed / refurbished", value="renewed" in title.lower())

row = pd.DataFrame([{"title": title, "rating": rating, "reviews": reviews, "category": category,
                     "is_renewed": int(renewed), "title_words": len(title.split()),
                     "title_len": len(title), "is_sponsored": 0}])
est = float(np.expm1(price_model.predict(add_features(row))[0]))
st.markdown(f'<div class="result">Estimated price: <b>${est:,.2f}</b></div>', unsafe_allow_html=True)

peers = df[df["category"] == category]["price"].dropna()
if len(peers):
    st.write(f"Typical **{category}** listings in the sample: median **${peers.median():,.0f}**, "
             f"middle 50% between **${peers.quantile(.25):,.0f}** and **${peers.quantile(.75):,.0f}**.")
    pct = (peers < est).mean()
    st.progress(float(pct), text=f"Your estimate is higher than {pct:.0%} of listings in this category")
st.caption("Treat this as a rough benchmark: typical error on unseen products is tens of dollars, "
           "and larger for expensive items.")
