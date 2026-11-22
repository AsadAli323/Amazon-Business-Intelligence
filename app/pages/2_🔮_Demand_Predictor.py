import pandas as pd
import plotly.express as px
import streamlit as st

from common import CATEGORIES, load_models, setup
from shelfsense.train import add_features

setup("Demand Predictor", "🔮")
st.title("🔮 Demand Predictor")
st.caption("Forecast a listing's monthly sales tier: Low (≤100), Medium (200–999) or High (≥1,000 units).")

demand_model, _ = load_models()

with st.form("f"):
    title = st.text_input("Product title", "Anker 737 Power Bank 24000mAh 140W USB-C Portable Charger")
    c1, c2, c3 = st.columns(3)
    price = c1.number_input("Price (USD)", 1.0, 20000.0, 89.99)
    rating = c2.slider("Rating", 1.0, 5.0, 4.5, 0.1)
    reviews = c3.number_input("Number of reviews", 0, 1_000_000, 1200)
    c4, c5, c6 = st.columns(3)
    category = c4.selectbox("Category", CATEGORIES, index=CATEGORIES.index("Chargers, Cables & Power"))
    discount = c5.slider("Discount (%)", 0, 80, 10)
    words = c6.number_input("Title word count", 1, 60, len(title.split()))
    o1, o2, o3, o4 = st.columns(4)
    sponsored = o1.checkbox("Sponsored ad")
    best = o2.checkbox("Best Seller badge")
    coupon = o3.checkbox("Has coupon")
    renewed = o4.checkbox("Renewed / refurbished")
    go = st.form_submit_button("Predict demand", type="primary")

if go:
    row = pd.DataFrame([{
        "title": title, "price": price, "rating": rating, "reviews": reviews, "discount_pct": discount,
        "is_sponsored": int(sponsored), "is_best_seller": int(best), "is_amazons_choice": 0,
        "is_limited_deal": 0, "has_coupon": int(coupon), "has_sustainability_badge": 0,
        "is_renewed": int(renewed), "title_words": len(title.split()), "category": category,
    }])
    X = add_features(row)
    probs = demand_model.predict_proba(X)[0]
    classes = list(demand_model.classes_)
    top = classes[probs.argmax()]
    st.markdown(f'<div class="result">Predicted demand tier: <b>{top}</b> '
                f'({probs.max():.0%} confidence)</div>', unsafe_allow_html=True)
    p = pd.DataFrame({"Tier": classes, "Probability": probs})
    order = ["Low", "Medium", "High"]
    st.plotly_chart(px.bar(p, x="Tier", y="Probability", category_orders={"Tier": order},
                           range_y=[0, 1], text=p["Probability"].map("{:.0%}".format))
                    .update_layout(height=320), width="stretch")
    st.caption("Review count is the strongest signal: heavily reviewed products are usually "
               "popular because they have already sold a lot.")
