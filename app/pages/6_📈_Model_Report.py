import pandas as pd
import streamlit as st

from common import load_metrics, setup

setup("Model Report", "📈")
st.title("📈 Model Report")
m = load_metrics()

st.subheader("Demand tier classifier")
d = m["demand"]
st.write(f"Three classes (Low / Medium / High) on **{d['rows']:,}** products, 80/20 stratified split. "
         f"Best model: **{d['best_model']}**. A majority-class guess scores "
         f"**{d['majority_baseline_accuracy']:.1%}** accuracy.")
st.dataframe(pd.DataFrame(d["results"]).T.rename_axis("Model"), width="stretch")

st.subheader("Price estimator")
p = m["price"]
st.write(f"Trained on **{p['rows']:,}** products with a log-price target. Best model: **{p['best_model']}**. "
         f"Predicting the median for everyone gives an MAE of **${p['median_baseline_mae_usd']}**.")
st.dataframe(pd.DataFrame(p["results"]).T.rename_axis("Model"), width="stretch")

st.subheader("Limitations")
st.markdown("""
- Sales volume is reported in rounded buckets (50+, 100+, 1K+ …), so it is an approximate target.
- Data is a single snapshot from late August 2025, so seasonality is not captured.
- Categories are assigned by keyword rules on the title and can be wrong for ambiguous products.
- Review count partly reflects past sales, so it is a strong but somewhat circular signal.
""")
