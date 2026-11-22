import numpy as np
import pandas as pd
import streamlit as st

from common import load_models, load_sample, setup, to_tidy
from shelfsense.train import add_features

setup("Batch Scoring", "📁")
st.title("📁 Batch Scoring")
st.caption("Upload a CSV of listings. Each row gets a predicted demand tier and a model price estimate.")

demand_model, price_model = load_models()
st.markdown("**Required columns:** `title`, `price`, `rating`, `reviews`. Optional: `category`, "
            "`discount_pct`, `is_sponsored`, `is_best_seller`, `has_coupon`, `is_renewed`. "
            "The original raw scraped export format is also accepted.")

sample = load_sample().head(200)[["title", "price", "rating", "reviews"]]
st.download_button("Download a template CSV", sample.to_csv(index=False), "template.csv")

up = st.file_uploader("Upload CSV", type="csv")
if up:
    df = to_tidy(pd.read_csv(up))
    missing = [c for c in ["title", "price", "rating", "reviews"] if c not in df.columns]
    if missing:
        st.error(f"Missing required column(s): {', '.join(missing)}")
        st.stop()
    from shelfsense.cleaning import assign_category
    for col in ["discount_pct", "is_sponsored", "is_best_seller", "is_amazons_choice", "is_limited_deal",
                "has_coupon", "has_sustainability_badge", "is_renewed"]:
        if col not in df:
            df[col] = 0
    if "category" not in df:
        df["category"] = df["title"].map(assign_category)
    df["title"] = df["title"].astype(str)
    df["title_words"] = df["title"].str.split().str.len()
    df["title_len"] = df["title"].str.len()
    df["price"] = pd.to_numeric(df["price"], errors="coerce").fillna(df["price"].median() if df["price"].notna().any() else 50)
    X = add_features(df)
    out = df.copy()
    out["predicted_demand_tier"] = demand_model.predict(X)
    out["model_price_estimate"] = np.expm1(price_model.predict(X)).round(2)
    out["price_vs_estimate_pct"] = ((out["price"] / out["model_price_estimate"] - 1) * 100).round(1)
    st.success(f"Scored {len(out):,} rows.")
    st.dataframe(out[["title", "category", "price", "model_price_estimate", "price_vs_estimate_pct",
                      "predicted_demand_tier"]], width="stretch", hide_index=True)
    st.bar_chart(out["predicted_demand_tier"].value_counts())
    st.download_button("Download scored CSV", out.to_csv(index=False), "scored_products.csv", type="primary")
