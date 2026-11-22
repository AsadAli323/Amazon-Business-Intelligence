import plotly.express as px
import streamlit as st

from common import load_sample, setup

setup("Market Explorer", "📊")
st.title("📊 Market Explorer")
df = load_sample()

with st.sidebar:
    st.header("Filters")
    cats = st.multiselect("Category", sorted(df["category"].unique()))
    top_brands = df["brand"].value_counts().head(40).index.tolist()
    brands = st.multiselect("Brand", top_brands)
    lo, hi = float(df["price"].min()), float(df["price"].quantile(0.99))
    price = st.slider("Price range (USD)", lo, hi, (lo, hi))
    only_sold = st.checkbox("Only products with sales data", value=False)

f = df[df["price"].between(*price) | df["price"].isna()]
if cats:
    f = f[f["category"].isin(cats)]
if brands:
    f = f[f["brand"].isin(brands)]
if only_sold:
    f = f[f["units_sold_month"].notna()]

k1, k2, k3, k4 = st.columns(4)
k1.metric("Products", f"{len(f):,}")
k2.metric("Avg rating", f"{f['rating'].mean():.2f}")
k3.metric("Median price", f"${f['price'].median():,.0f}")
k4.metric("Sponsored share", f"{f['is_sponsored'].mean():.0%}")

t1, t2, t3 = st.tabs(["Overview", "Sales drivers", "Data"])
with t1:
    l, r = st.columns(2)
    cc = f["category"].value_counts().reset_index()
    l.plotly_chart(px.bar(cc, x="count", y="category", orientation="h", title="Products per category")
                   .update_layout(yaxis={"categoryorder": "total ascending"}), width="stretch")
    r.plotly_chart(px.box(f.dropna(subset=["price"]), x="category", y="price", log_y=True,
                          title="Price by category (log scale)")
                   .update_layout(xaxis_tickangle=-40, showlegend=False), width="stretch")
with t2:
    s = f.dropna(subset=["units_sold_month", "reviews"])
    l, r = st.columns(2)
    l.plotly_chart(px.scatter(s, x="reviews", y="units_sold_month", color="category", log_x=True,
                              log_y=True, hover_name="title", opacity=.6,
                              title="Reviews vs monthly units sold"), width="stretch")
    g = s.groupby("is_sponsored")["units_sold_month"].median().rename({0: "Organic", 1: "Sponsored"})
    r.plotly_chart(px.bar(g.reset_index(), x="is_sponsored", y="units_sold_month",
                          title="Median monthly units: organic vs sponsored",
                          labels={"is_sponsored": "", "units_sold_month": "Median units / month"}),
                   width="stretch")
    s2 = s.assign(rating_band=s["rating"].round(1))
    st.plotly_chart(px.box(s2, x="rating_band", y="units_sold_month", log_y=True,
                           title="Units sold by rating"), width="stretch")
with t3:
    cols = ["title", "brand", "category", "price", "rating", "reviews", "units_sold_month",
            "discount_pct", "is_sponsored"]
    st.dataframe(f[cols], width="stretch", hide_index=True)
    st.download_button("Download filtered data (CSV)", f[cols].to_csv(index=False), "filtered.csv")
