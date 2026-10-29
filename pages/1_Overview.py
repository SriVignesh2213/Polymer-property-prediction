"""Overview dashboard."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import plotly.express as px
import streamlit as st

from src.config import TARGET_LABELS, TRAIN_DEFAULT_PATH, TEST_DEFAULT_PATH
from src.data_loader import load_csv, summarize_dataset, TARGET_TYPES
from src.model_pipeline import load_artifact

st.title("Overview")
st.caption("Executive summary of datasets, targets, and model status.")

train_res = test_res = None
try:
    train_res = load_csv(TRAIN_DEFAULT_PATH)
except Exception as exc:
    st.warning(str(exc))
try:
    test_res = load_csv(TEST_DEFAULT_PATH)
except Exception as exc:
    st.warning(str(exc))

artifact = load_artifact()

if train_res is not None and test_res is not None:
    train_df = train_res.dataframe
    test_df = test_res.dataframe
    tr_summary = summarize_dataset(train_df)
    te_summary = summarize_dataset(test_df)
    st.markdown('<div class="metric-grid">', unsafe_allow_html=True)
    metrics = {
        "Training rows": len(train_df),
        "Testing rows": len(test_df),
        "Target properties": len(TARGET_TYPES),
        "Unique train SMILES": train_df["smiles"].nunique(),
    }
    cols = st.columns(len(metrics))
    for col, (label, value) in zip(cols, metrics.items()):
        col.metric(label, f"{value:,}")
    st.markdown("</div>", unsafe_allow_html=True)

    counts = []
    for target in TARGET_TYPES:
        counts.append({"property": target, "train": len(train_df[train_df.target_type == target]), "test": len(test_df[test_df.target_type == target])})
    counts_df = pd.DataFrame(counts)
    fig = px.bar(counts_df, x="property", y=["train", "test"], barmode="group", template="plotly_dark", title="Train/test counts by property")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Target distribution")
    dist = train_df.groupby("target_type")["target"].agg(["count", "mean", "std", "min", "max"]).reset_index()
    st.dataframe(dist, use_container_width=True)

    st.subheader("Artifact status")
    if artifact is None:
        st.markdown('<span class="status missing">Model artifacts not available. Training script has not been run.</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status ok">Model artifacts loaded.</span>', unsafe_allow_html=True)
        st.write("Cross-validation scores")
        scores = artifact.get("cv_scores", {})
        score_df = pd.DataFrame({"target": list(scores.keys()), "cv_r2": list(scores.values())})
        if not score_df.empty:
            st.dataframe(score_df, use_container_width=True)
            st.metric("Official mean R2", f"{artifact.get('official_mean', float('nan')):.4f}")
else:
    st.info("Configure or upload datasets in the sidebar workflow.")