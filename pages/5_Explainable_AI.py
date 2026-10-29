"""Explainability page."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from src.config import TARGET_LABELS, TRAIN_DEFAULT_PATH
from src.data_loader import load_csv
from src.feature_engineering import build_feature_matrix, feature_names
from src.model_pipeline import load_artifact
from src.explainability import shap_importance_for_target

st.title("Explainable AI")
st.caption("SHAP-based feature importance for tree models.")

artifact = load_artifact()
if artifact is None:
    st.info("Model artifacts unavailable.")
    st.stop()

target = st.selectbox("Property", ["tg", "egc"])
train_res = load_csv(TRAIN_DEFAULT_PATH)
train_df = train_res.dataframe
target_df = train_df[train_df.target_type == target].head(200)
if target_df.empty:
    st.warning("No training rows available for this target.")
    st.stop()

X, _, _ = build_feature_matrix(target_df.smiles.tolist())
importance = shap_importance_for_target(artifact, target, X[: min(100, len(X))])
if importance is None:
    st.warning("Feature importance is unavailable.")
else:
    names = feature_names()
    importance_frame = pd.DataFrame({"feature": names, "importance": importance}).sort_values("importance", ascending=False).head(20)
    fig = px.bar(importance_frame.sort_values("importance"), x="importance", y="feature", orientation="h", template="plotly_dark", title=f"Top SHAP features: {target}")
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(importance_frame, use_container_width=True)
    st.caption("Features are generated from RDKit descriptors, motifs, fingerprints, and physical ratios. Unnamed features are reported by internal feature index.")