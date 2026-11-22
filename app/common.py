"""Shared helpers for the ShelfSense Streamlit app."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from shelfsense.cleaning import CATEGORY_RULES, DEFAULT_CATEGORY, clean  # noqa: E402
from shelfsense.train import add_features  # noqa: E402

CATEGORIES = [c for c, _ in CATEGORY_RULES] + [DEFAULT_CATEGORY]
ACCENT = "#FF9900"

CSS = """
<style>
.block-container {padding-top: 2rem; max-width: 1200px;}
.hero {background: linear-gradient(135deg,#0f1c2e 0%,#1d3557 100%); color:#fff;
       padding: 2rem 2.2rem; border-radius: 16px; margin-bottom: 1.2rem;}
.hero h1 {color:#fff; margin:0 0 .3rem 0; font-size: 2.2rem;}
.hero p {color:#cfd8e3; margin:0; font-size:1.05rem;}
.card {border:1px solid rgba(128,128,128,.25); border-radius:14px; padding:1.1rem 1.2rem;
       height:100%;}
.card h4 {margin:0 0 .3rem 0;}
.result {border-left:6px solid #FF9900; background:rgba(255,153,0,.08);
         padding:1rem 1.2rem; border-radius:10px; font-size:1.1rem;}
</style>
"""


def setup(title: str, icon: str = "🛒"):
    st.set_page_config(page_title=f"{title} · ShelfSense", page_icon=icon, layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def load_sample() -> pd.DataFrame:
    df = pd.read_csv(ROOT / "data" / "sample" / "products_sample.csv", parse_dates=["collected_at"])
    return df


@st.cache_resource(show_spinner=False)
def load_models():
    return (joblib.load(ROOT / "models" / "demand_model.joblib"),
            joblib.load(ROOT / "models" / "price_model.joblib"))


@st.cache_data(show_spinner=False)
def load_metrics() -> dict:
    return json.loads((ROOT / "reports" / "metrics.json").read_text())


def to_tidy(df: pd.DataFrame) -> pd.DataFrame:
    """Accept either the raw scraped export or an already-tidy table."""
    if "Title" in df.columns and "Current/discounted_price" in df.columns:
        df = clean(df)
    return df
