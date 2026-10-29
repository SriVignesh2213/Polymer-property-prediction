"""Dataset explorer."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import plotly.express as px
import streamlit as st

from src.config import TARGET_LABELS, TRAIN_DEFAULT_PATH, TEST_DEFAULT_PATH
from src.data_loader import load_csv, summarize_dataset

st.title("Dataset Explorer")
st.caption("Explore training and testing polymer data without modifying it.")

dataset_name = st.selectbox("Dataset", ["Training", "Testing"])
uploaded = st.file_uploader("Upload CSV override", type="csv")
path = TRAIN_DEFAULT_PATH if dataset_name == "Training" else TEST_DEFAULT_PATH
result = None
try:
    result = load_csv(path, uploaded)
except Exception as exc:
    st.error(str(exc))
    st.stop()

df = result.dataframe
st.success(f"Loaded {len(df):,} rows from {result.source}")

summary = summarize_dataset(df)
col1, col2, col3, col4 = st.columns(4)
col1.metric("Rows", f"{summary['rows']:,}")
col2.metric("Columns", summary["columns"])
col3.metric("Missing cells", f"{summary['missing_cells']:,}")
col4.metric("Unique SMILES", f"{summary['unique_smiles']:,}")

filtered = df.copy()
if "target_type" in df.columns:
    selected_types = st.multiselect("Filter target_type", sorted(df.target_type.dropna().unique().tolist()), default=sorted(df.target_type.dropna().unique().tolist()))
    if selected_types:
        filtered = filtered[filtered.target_type.isin(selected_types)]

page_size = st.slider("Rows per page", 10, 100, 25)
page = st.number_input("Page", min_value=1, value=1, step=1)
start = (page - 1) * page_size
st.dataframe(filtered.iloc[start:start + page_size], use_container_width=True)

st.subheader("Missing values")
st.dataframe(df.isna().sum().reset_index().rename(columns={"index": "column", 0: "missing"}), use_container_width=True)

st.subheader("Target distribution")
if "target_type" in df.columns and "target" in df.columns:
    stats = df.groupby("target_type")["target"].agg(["count", "mean", "std", "min", "max"]).reset_index()
    st.dataframe(stats, use_container_width=True)
    fig = px.histogram(df, x="target", color="target_type", barmode="overlay", template="plotly_dark", title="Target distributions")
    st.plotly_chart(fig, use_container_width=True)
    fig2 = px.box(df, x="target_type", y="target", template="plotly_dark", title="Box plots")
    st.plotly_chart(fig2, use_container_width=True)
else:
    counts = df["target_type"].value_counts().reset_index()
    counts.columns = ["target_type", "count"]
    st.dataframe(counts, use_container_width=True)

download = filtered.to_csv(index=False).encode("utf-8")
st.download_button("Download filtered dataset", download, file_name="filtered_dataset.csv")