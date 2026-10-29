"""Amalgam Finale Streamlit entry point."""

from __future__ import annotations

from pathlib import Path
import sys

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

st.set_page_config(
    page_title="Amalgam Finale",
    page_icon="A",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .app-header {padding: 0.75rem 0; border-bottom: 1px solid #2a2d3a;}
    .metric-grid {display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 0.75rem;}
    .metric-card {background: #171923; border: 1px solid #2a2d3a; border-radius: 8px; padding: 1rem;}
    .metric-card h4 {margin: 0; color: #9aa0a6; font-size: 0.85rem; font-weight: 500;}
    .metric-card p {margin: 0.35rem 0 0; font-size: 1.35rem; font-weight: 600;}
    .status {border-radius: 6px; padding: 0.35rem 0.6rem; font-size: 0.85rem; display: inline-block;}
    .status.ok {background: #153a33; color: #7ee3d5;}
    .status.missing {background: #3a2418; color: #f0a882;}
    .stDataFrame {border: 1px solid #2a2d3a;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="app-header"><h1>Amalgam Finale</h1><p>Invariant polymer property prediction and molecular intelligence platform.</p></div>', unsafe_allow_html=True)

st.markdown(
    "Use the sidebar to navigate. The dashboard loads the configured training and testing CSVs without using PI1M.csv."
)

import pandas as pd
from src.data_loader import load_csv, summarize_dataset, REQUIRED_TRAIN_COLUMNS, REQUIRED_TEST_COLUMNS
from src.config import TRAIN_DEFAULT_PATH, TEST_DEFAULT_PATH

col1, col2 = st.columns(2)
train_result = None
test_result = None
try:
    train_result = load_csv(TRAIN_DEFAULT_PATH)
except Exception as exc:
    st.sidebar.error(f"Training CSV unavailable: {exc}")
try:
    test_result = load_csv(TEST_DEFAULT_PATH)
except Exception as exc:
    st.sidebar.error(f"Testing CSV unavailable: {exc}")

if train_result is not None:
    st.sidebar.success(f"Train loaded: {len(train_result.dataframe):,} rows")
if test_result is not None:
    st.sidebar.success(f"Test loaded: {len(test_result.dataframe):,} rows")

st.info("Navigate to pages through the sidebar. Model artifacts are optional; missing artifacts are reported honestly.")